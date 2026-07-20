use crate::config::ClientConfig;
use crate::domain::{
    AppliedCommandReceipt, ControllerSession, DisarmReceipt, MonotonicInstant,
    PendingPhysicalCommand, ReceiptTiming,
};
use crate::transport::{MonotonicClock, V2CommandTransport};
use robot_protocol::v2::{
    AcquireControl, AcquireResult, AcquireResultCode, ControlEpoch, ControllerBootId,
    ControllerCapabilities, ControllerFaults, ControllerUid, DeadlineRelation, ForceStopReason,
    HostCommand, HostCommandResult, HostCommandResultCode, HostStop, HostStopResult, Message,
    MessageKind, OutputState, RequestId, StatusCode, StatusQuery, StatusReport, StopResultCode,
    TargetBootId, TimerPwm, V2CommandLeaseMs, V2CommandSequence,
};
use std::fmt;
use std::time::Duration;

struct ClientCore<Transport, Clock> {
    transport: Transport,
    clock: Clock,
    config: ClientConfig,
    next_request_id: Option<RequestId>,
    next_sequence: Option<V2CommandSequence>,
    last_observed_time: MonotonicInstant,
}

impl<Transport, Clock> ClientCore<Transport, Clock>
where
    Transport: V2CommandTransport,
    Clock: MonotonicClock,
{
    fn observe_now(&mut self) -> Result<MonotonicInstant, FailureCause<Transport::Error>> {
        let observed = self.clock.now();
        if observed < self.last_observed_time {
            return Err(FailureCause::ClockRegressed {
                previous: self.last_observed_time,
                observed,
            });
        }
        self.last_observed_time = observed;
        Ok(observed)
    }

    fn allocate_request_id(&mut self) -> Result<RequestId, FailureCause<Transport::Error>> {
        let request_id = self
            .next_request_id
            .take()
            .ok_or(FailureCause::RequestIdExhausted)?;
        self.next_request_id = request_id.get().checked_add(1).map(RequestId::new);
        Ok(request_id)
    }

    fn reset_command_sequence(&mut self) {
        self.next_sequence = Some(V2CommandSequence::FIRST);
    }

    fn allocate_sequence(&mut self) -> Result<V2CommandSequence, FailureCause<Transport::Error>> {
        let sequence = self
            .next_sequence
            .take()
            .ok_or(FailureCause::CommandSequenceExhausted)?;
        self.next_sequence = sequence.checked_successor();
        Ok(sequence)
    }

    fn exchange_with_timeout(
        &mut self,
        request: Message,
        timeout: Duration,
    ) -> Result<(Message, MonotonicInstant, MonotonicInstant), FailureCause<Transport::Error>> {
        let started_at = self.observe_now()?;
        let deadline_exclusive = started_at
            .checked_add(timeout)
            .ok_or(FailureCause::MonotonicArithmeticOverflow)?;
        self.exchange_until(request, deadline_exclusive)
    }

    fn exchange_until(
        &mut self,
        request: Message,
        deadline_exclusive: MonotonicInstant,
    ) -> Result<(Message, MonotonicInstant, MonotonicInstant), FailureCause<Transport::Error>> {
        let sent_at = self.observe_now()?;
        if sent_at >= deadline_exclusive {
            return Err(FailureCause::DeadlineExpiredBeforeSend {
                now: sent_at,
                deadline_exclusive,
            });
        }
        let remaining = deadline_exclusive
            .checked_duration_since(sent_at)
            .ok_or(FailureCause::MonotonicArithmeticOverflow)?;
        let response = self
            .transport
            .exchange_once(request, remaining)
            .map_err(FailureCause::Transport)?;
        let acknowledged_at = self.observe_now()?;
        if acknowledged_at >= deadline_exclusive {
            return Err(FailureCause::ResponseAtOrAfterDeadline {
                acknowledged_at,
                deadline_exclusive,
            });
        }
        Ok((response, sent_at, acknowledged_at))
    }

    fn query_stopped_controller(
        &mut self,
    ) -> Result<ControllerBootId, FailureCause<Transport::Error>> {
        let request_id = self.allocate_request_id()?;
        let query = StatusQuery {
            expected_controller_uid: self.config.controller_uid(),
            request_id,
        };
        let (message, _, _) = self.exchange_with_timeout(
            Message::StatusQuery(query),
            self.config.status_timeout().as_duration(),
        )?;
        let Message::StatusReport(report) = message else {
            return Err(FailureCause::Evidence(EvidenceError::UnexpectedMessage {
                expected: MessageKind::StatusReport,
                actual: message.kind(),
            }));
        };
        verify_status_report(&self.config, request_id, report).map_err(FailureCause::Evidence)
    }

    fn acquire_control_epoch(
        &mut self,
        boot_id: ControllerBootId,
    ) -> Result<ControllerSession, FailureCause<Transport::Error>> {
        let request_id = self.allocate_request_id()?;
        let request = AcquireControl {
            expected_controller_uid: self.config.controller_uid(),
            expected_boot_id: boot_id,
            request_id,
            expected_firmware_abi: self.config.expected_firmware_abi(),
            expected_firmware_build_id: self.config.expected_firmware_build_id(),
            expected_actuator_config_fingerprint: self
                .config
                .expected_actuator_config_fingerprint(),
        };
        let (message, _, _) = self.exchange_with_timeout(
            Message::AcquireControl(request),
            self.config.acquire_timeout().as_duration(),
        )?;
        let Message::AcquireResult(result) = message else {
            return Err(FailureCause::Evidence(EvidenceError::UnexpectedMessage {
                expected: MessageKind::AcquireResult,
                actual: message.kind(),
            }));
        };
        let epoch = verify_acquire_result(&self.config, boot_id, request_id, result)
            .map_err(FailureCause::Evidence)?;
        Ok(ControllerSession::from_verified_acquisition(
            self.config.controller_uid(),
            boot_id,
            epoch,
        ))
    }

    fn command_once(
        &mut self,
        session: ControllerSession,
        requested_timer_pwm: TimerPwm,
        lease: V2CommandLeaseMs,
        acknowledgement_deadline_exclusive: Option<MonotonicInstant>,
    ) -> Result<AppliedCommandReceipt, FailureCause<Transport::Error>> {
        let now = self.observe_now()?;
        let timeout_deadline = now
            .checked_add(self.config.applied_ack_timeout().as_duration())
            .ok_or(FailureCause::MonotonicArithmeticOverflow)?;
        let effective_deadline = acknowledgement_deadline_exclusive
            .map_or(timeout_deadline, |valid_through| {
                timeout_deadline.min(valid_through)
            });
        if now >= effective_deadline {
            return Err(FailureCause::DeadlineExpiredBeforeSend {
                now,
                deadline_exclusive: effective_deadline,
            });
        }
        // Allocate before the transport call. Even a local codec error burns
        // the sequence, so an uncertain nonzero is never sent again.
        let sequence = self.allocate_sequence()?;
        let request = HostCommand {
            controller_uid: session.controller_uid(),
            boot_id: session.boot_id(),
            control_epoch: session.control_epoch(),
            sequence,
            lease,
            requested_timer_pwm,
        };
        let (message, sent_at, acknowledged_at) =
            self.exchange_until(Message::HostCommand(request), effective_deadline)?;
        let Message::HostCommandResult(result) = message else {
            return Err(FailureCause::Evidence(EvidenceError::UnexpectedMessage {
                expected: MessageKind::HostCommandResult,
                actual: message.kind(),
            }));
        };
        verify_command_result(session, sequence, requested_timer_pwm, lease, result)
            .map_err(FailureCause::Evidence)?;

        // `remaining_lease` is measured when the server emits the result. The
        // server emission occurred no earlier than host send, so subtracting
        // the complete host-observed round trip is conservative: the usable
        // horizon is host send plus the reported remaining lifetime.
        let reported_remaining = Duration::from_millis(u64::from(result.remaining_lease.get()));
        let reported_horizon = sent_at
            .checked_add(reported_remaining)
            .ok_or(FailureCause::MonotonicArithmeticOverflow)?;
        // The caller's acknowledgement deadline governs admission of this
        // decision only. A successful exact ACK independently proves the
        // conservative controller-reported lease horizon; carrying the old
        // decision deadline into that receipt would falsely expire the next
        // pre-solve applied-evidence gate.
        let known_active_through_exclusive = reported_horizon;
        if acknowledged_at >= known_active_through_exclusive {
            return Err(FailureCause::LeaseNotKnownActiveAtAcknowledgement {
                acknowledged_at,
                known_active_through_exclusive,
            });
        }

        Ok(AppliedCommandReceipt::new(
            session,
            lease,
            result,
            ReceiptTiming {
                sent_at,
                acknowledged_at,
                known_active_through_exclusive,
            },
        ))
    }

    fn stop_once(
        &mut self,
        reason: ForceStopReason,
        timeout: Duration,
    ) -> Result<DisarmReceipt, FailureCause<Transport::Error>> {
        let request_id = self.allocate_request_id()?;
        let request = HostStop {
            controller_uid: self.config.controller_uid(),
            target_boot_id: TargetBootId::Any,
            request_id,
            reason,
        };
        let sent_at = match self.observe_now() {
            Ok(sent_at) => sent_at,
            Err(cause) => {
                // The OS transport timeout remains a bounded clock independent
                // of this client clock. Send once to reduce physical risk, but
                // never manufacture a receipt from untrustworthy timing.
                let _ = self
                    .transport
                    .exchange_once(Message::HostStop(request), timeout);
                return Err(cause);
            }
        };
        let deadline_exclusive = match sent_at.checked_add(timeout) {
            Some(deadline) => deadline,
            None => {
                let _ = self
                    .transport
                    .exchange_once(Message::HostStop(request), timeout);
                return Err(FailureCause::MonotonicArithmeticOverflow);
            }
        };
        let message = self
            .transport
            .exchange_once(Message::HostStop(request), timeout)
            .map_err(FailureCause::Transport)?;
        let acknowledged_at = self.observe_now()?;
        if acknowledged_at >= deadline_exclusive {
            return Err(FailureCause::ResponseAtOrAfterDeadline {
                acknowledged_at,
                deadline_exclusive,
            });
        }
        let Message::HostStopResult(result) = message else {
            return Err(FailureCause::Evidence(EvidenceError::UnexpectedMessage {
                expected: MessageKind::HostStopResult,
                actual: message.kind(),
            }));
        };
        verify_stop_result(
            self.config.controller_uid(),
            request_id,
            result,
            acknowledged_at,
        )
        .map_err(FailureCause::Evidence)
    }

    fn recover_stop(
        &mut self,
        reason: ForceStopReason,
        maximum_attempts: u8,
    ) -> StopRecoveryReport<Transport::Error> {
        let timeout = self.config.stop_recovery().attempt_timeout().as_duration();
        let mut last_failure = None;
        let mut attempts_started = 0;
        for attempt in 1..=maximum_attempts {
            attempts_started = attempt;
            match self.stop_once(reason, timeout) {
                Ok(receipt) => {
                    return StopRecoveryReport {
                        attempts_started,
                        confirmed_stop: Some(receipt),
                        last_failure,
                    };
                }
                Err(cause) => {
                    last_failure = Some(RecoveryAttemptFailure { attempt, cause });
                }
            }
        }
        StopRecoveryReport {
            attempts_started,
            confirmed_stop: None,
            last_failure,
        }
    }
}

pub struct DisarmedCommandClient<Transport, Clock> {
    core: ClientCore<Transport, Clock>,
}

impl<Transport, Clock> DisarmedCommandClient<Transport, Clock>
where
    Transport: V2CommandTransport,
    Clock: MonotonicClock,
{
    pub fn new(transport: Transport, clock: Clock, config: ClientConfig) -> Self {
        let last_observed_time = clock.now();
        Self {
            core: ClientCore {
                transport,
                clock,
                config,
                next_request_id: Some(RequestId::new(0)),
                next_sequence: Some(V2CommandSequence::FIRST),
                last_observed_time,
            },
        }
    }

    pub fn acquire_zero(
        self,
    ) -> Result<
        (ArmedCommandClient<Transport, Clock>, AppliedCommandReceipt),
        AcquireFailure<Transport, Clock>,
    > {
        let mut core = self.core;
        let boot_id = match core.query_stopped_controller() {
            Ok(boot_id) => boot_id,
            Err(cause) => return Err(AcquireFailure::latch(core, cause)),
        };
        let session = match core.acquire_control_epoch(boot_id) {
            Ok(session) => session,
            Err(cause) => return Err(AcquireFailure::latch(core, cause)),
        };
        core.reset_command_sequence();
        let receipt = match core.command_once(
            session,
            TimerPwm::ZERO,
            core.config.zero_acquisition_lease(),
            None,
        ) {
            Ok(receipt) => receipt,
            Err(cause) => return Err(AcquireFailure::latch(core, cause)),
        };
        if receipt.sequence() != V2CommandSequence::FIRST || !receipt.is_confirmed_zero() {
            return Err(AcquireFailure::latch(
                core,
                FailureCause::Evidence(EvidenceError::InitialAcquisitionWasNotSequenceZero),
            ));
        }
        let confirmed = ConfirmedAppliedState::from_receipt(&receipt);
        Ok((
            ArmedCommandClient {
                core: Some(core),
                session,
                confirmed,
            },
            receipt,
        ))
    }

    pub fn config(&self) -> &ClientConfig {
        &self.core.config
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ConfirmedAppliedState {
    timer_pwm: TimerPwm,
    known_active_through_exclusive: MonotonicInstant,
}

impl ConfirmedAppliedState {
    fn from_receipt(receipt: &AppliedCommandReceipt) -> Self {
        Self {
            timer_pwm: receipt.applied_timer_pwm(),
            known_active_through_exclusive: receipt.known_active_through_exclusive(),
        }
    }
}

pub struct ArmedCommandClient<Transport, Clock>
where
    Transport: V2CommandTransport,
    Clock: MonotonicClock,
{
    core: Option<ClientCore<Transport, Clock>>,
    session: ControllerSession,
    confirmed: ConfirmedAppliedState,
}

impl<Transport, Clock> ArmedCommandClient<Transport, Clock>
where
    Transport: V2CommandTransport,
    Clock: MonotonicClock,
{
    /// Consumes the armed state and returns it only when the previous applied
    /// receipt still proves an active controller lease. Call this immediately
    /// before starting work that may produce the next motion decision; a stale
    /// receipt latches the session and performs bounded stop recovery without
    /// allowing another command to be constructed.
    pub fn require_current_applied_evidence(
        mut self,
    ) -> Result<Self, ApplyFailure<Transport, Clock>> {
        let mut core = self.core.take().expect("armed client core is present");
        let now = match core.observe_now() {
            Ok(now) => now,
            Err(cause) => return Err(ApplyFailure::latch(core, cause)),
        };
        if now >= self.confirmed.known_active_through_exclusive {
            return Err(ApplyFailure::latch(
                core,
                FailureCause::PreviousAppliedEvidenceExpired {
                    now,
                    known_active_through_exclusive: self.confirmed.known_active_through_exclusive,
                },
            ));
        }
        self.core = Some(core);
        Ok(self)
    }

    pub fn apply(
        mut self,
        pending: PendingPhysicalCommand,
    ) -> Result<(Self, AppliedCommandReceipt), ApplyFailure<Transport, Clock>> {
        let mut core = self.core.take().expect("armed client core is present");
        let now = match core.observe_now() {
            Ok(now) => now,
            Err(cause) => return Err(ApplyFailure::latch(core, cause)),
        };
        if now >= self.confirmed.known_active_through_exclusive {
            return Err(ApplyFailure::latch(
                core,
                FailureCause::PreviousAppliedEvidenceExpired {
                    now,
                    known_active_through_exclusive: self.confirmed.known_active_through_exclusive,
                },
            ));
        }
        let receipt = match core.command_once(
            self.session,
            pending.requested_timer_pwm(),
            pending.lease(),
            Some(pending.acknowledgement_deadline_exclusive()),
        ) {
            Ok(receipt) => receipt,
            Err(cause) => return Err(ApplyFailure::latch(core, cause)),
        };
        self.confirmed = ConfirmedAppliedState::from_receipt(&receipt);
        self.core = Some(core);
        Ok((self, receipt))
    }

    pub fn disarm(
        self,
    ) -> Result<
        (DisarmedCommandClient<Transport, Clock>, DisarmReceipt),
        DisarmFailure<Transport, Clock>,
    > {
        self.disarm_with_reason(ForceStopReason::Operator)
    }

    /// End this control session with an exact, caller-selected protocol reason.
    /// This is used when a local safety boundary rejects a decision before it
    /// can become a command. Success still requires a matching controller stop
    /// receipt; a transport write alone is never reported as disarmed.
    pub fn disarm_with_reason(
        mut self,
        reason: ForceStopReason,
    ) -> Result<
        (DisarmedCommandClient<Transport, Clock>, DisarmReceipt),
        DisarmFailure<Transport, Clock>,
    > {
        let mut core = self.core.take().expect("armed client core is present");
        let timeout = core.config.stop_recovery().attempt_timeout().as_duration();
        let receipt = match core.stop_once(reason, timeout) {
            Ok(receipt) => receipt,
            Err(cause) => {
                return Err(DisarmFailure::latch_after_first_attempt(
                    core, cause, reason,
                ));
            }
        };
        core.reset_command_sequence();
        Ok((DisarmedCommandClient { core }, receipt))
    }

    pub const fn confirmed_applied_timer_pwm(&self) -> TimerPwm {
        self.confirmed.timer_pwm
    }

    pub const fn confirmed_active_through_exclusive(&self) -> MonotonicInstant {
        self.confirmed.known_active_through_exclusive
    }

    pub const fn controller_session(&self) -> ControllerSession {
        self.session
    }
}

impl<Transport, Clock> Drop for ArmedCommandClient<Transport, Clock>
where
    Transport: V2CommandTransport,
    Clock: MonotonicClock,
{
    fn drop(&mut self) {
        // Bounded risk reduction only: Drop cannot return a receipt, persist
        // evidence, or truthfully claim that the controller stopped.
        if let Some(core) = self.core.as_mut() {
            let attempts = core.config.stop_recovery().attempts().get();
            let _ = core.recover_stop(ForceStopReason::Operator, attempts);
        }
    }
}

pub struct LatchedCommandClient<Transport, Clock>
where
    Transport: V2CommandTransport,
{
    _core: ClientCore<Transport, Clock>,
    recovery: StopRecoveryReport<Transport::Error>,
}

impl<Transport, Clock> LatchedCommandClient<Transport, Clock>
where
    Transport: V2CommandTransport,
{
    pub const fn stop_knowledge(&self) -> LatchedStopKnowledge {
        if self.recovery.confirmed_stop.is_some() {
            LatchedStopKnowledge::ConfirmedStop
        } else {
            LatchedStopKnowledge::Unconfirmed
        }
    }

    pub const fn recovery(&self) -> &StopRecoveryReport<Transport::Error> {
        &self.recovery
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LatchedStopKnowledge {
    ConfirmedStop,
    Unconfirmed,
}

pub struct StopRecoveryReport<TransportError> {
    attempts_started: u8,
    confirmed_stop: Option<DisarmReceipt>,
    last_failure: Option<RecoveryAttemptFailure<TransportError>>,
}

impl<TransportError> StopRecoveryReport<TransportError> {
    pub const fn attempts_started(&self) -> u8 {
        self.attempts_started
    }

    pub const fn confirmed_stop_receipt(&self) -> Option<&DisarmReceipt> {
        self.confirmed_stop.as_ref()
    }

    pub const fn last_failure(&self) -> Option<&RecoveryAttemptFailure<TransportError>> {
        self.last_failure.as_ref()
    }
}

pub struct RecoveryAttemptFailure<TransportError> {
    attempt: u8,
    cause: FailureCause<TransportError>,
}

impl<TransportError> RecoveryAttemptFailure<TransportError> {
    pub const fn attempt(&self) -> u8 {
        self.attempt
    }

    pub const fn cause(&self) -> &FailureCause<TransportError> {
        &self.cause
    }
}

#[derive(Debug)]
pub enum FailureCause<TransportError> {
    Transport(TransportError),
    Evidence(EvidenceError),
    ClockRegressed {
        previous: MonotonicInstant,
        observed: MonotonicInstant,
    },
    MonotonicArithmeticOverflow,
    RequestIdExhausted,
    CommandSequenceExhausted,
    DeadlineExpiredBeforeSend {
        now: MonotonicInstant,
        deadline_exclusive: MonotonicInstant,
    },
    ResponseAtOrAfterDeadline {
        acknowledged_at: MonotonicInstant,
        deadline_exclusive: MonotonicInstant,
    },
    LeaseNotKnownActiveAtAcknowledgement {
        acknowledged_at: MonotonicInstant,
        known_active_through_exclusive: MonotonicInstant,
    },
    PreviousAppliedEvidenceExpired {
        now: MonotonicInstant,
        known_active_through_exclusive: MonotonicInstant,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvidenceError {
    UnexpectedMessage {
        expected: MessageKind,
        actual: MessageKind,
    },
    ControllerUidMismatch {
        expected: ControllerUid,
        actual: ControllerUid,
    },
    RequestIdMismatch {
        expected: RequestId,
        actual: RequestId,
    },
    ControllerBootIdMismatch {
        expected: ControllerBootId,
        actual: ControllerBootId,
    },
    ControlEpochMismatch {
        expected: ControlEpoch,
        actual: ControlEpoch,
    },
    SequenceMismatch {
        expected: V2CommandSequence,
        actual: V2CommandSequence,
    },
    StatusNotReadyStopped(StatusCode),
    StatusHasControlEpoch(ControlEpoch),
    StatusHasNoExactBootId,
    RequiredCapabilitiesMissing(ControllerCapabilities),
    ControllerFaultsPresent(ControllerFaults),
    OutputNotSafe(OutputState),
    StatusTimerPwmNonzero(TimerPwm),
    StatusRemainingLeaseNonzero(u16),
    AcquireNotGranted(AcquireResultCode),
    GrantedAcquireMissingEpoch,
    FirmwareAbiMismatch {
        expected: u16,
        actual: u16,
    },
    FirmwareBuildIdMismatch {
        expected: u32,
        actual: u32,
    },
    ActuatorConfigFingerprintMismatch,
    InitialAcquisitionWasNotSequenceZero,
    CommandResultNotFreshlyApplied(HostCommandResultCode),
    RequestedTimerPwmMismatch {
        expected: TimerPwm,
        actual: TimerPwm,
    },
    ControllerTimerPwmMismatch {
        expected: TimerPwm,
        actual: TimerPwm,
    },
    OutputStateMismatch {
        expected: OutputState,
        actual: OutputState,
    },
    RemainingLeaseZero,
    RemainingLeaseAboveRequested {
        remaining_ms: u16,
        requested_ms: u16,
    },
    ControllerDeadlineNotFuture,
    ControllerDeadlineAboveRequested {
        deadline_delta_ms: u32,
        requested_ms: u16,
    },
    RemainingLeaseAboveControllerDeadline {
        remaining_ms: u16,
        deadline_delta_ms: u32,
    },
    StopNotConfirmed(StopResultCode),
    StopResultHasNoExactBootId,
}

fn verify_common_identity(
    expected_uid: ControllerUid,
    actual_uid: ControllerUid,
) -> Result<(), EvidenceError> {
    if actual_uid != expected_uid {
        return Err(EvidenceError::ControllerUidMismatch {
            expected: expected_uid,
            actual: actual_uid,
        });
    }
    Ok(())
}

fn verify_capabilities_and_faults(
    capabilities: ControllerCapabilities,
    faults: ControllerFaults,
) -> Result<(), EvidenceError> {
    if !capabilities.supports_required_safety() {
        return Err(EvidenceError::RequiredCapabilitiesMissing(capabilities));
    }
    if !faults.is_clear() {
        return Err(EvidenceError::ControllerFaultsPresent(faults));
    }
    Ok(())
}

fn verify_status_report(
    config: &ClientConfig,
    expected_request_id: RequestId,
    report: StatusReport,
) -> Result<ControllerBootId, EvidenceError> {
    verify_common_identity(config.controller_uid(), report.controller_uid)?;
    if report.request_id != expected_request_id {
        return Err(EvidenceError::RequestIdMismatch {
            expected: expected_request_id,
            actual: report.request_id,
        });
    }
    if report.status != StatusCode::ReadyStopped {
        return Err(EvidenceError::StatusNotReadyStopped(report.status));
    }
    if let Some(epoch) = report.control_epoch {
        return Err(EvidenceError::StatusHasControlEpoch(epoch));
    }
    let TargetBootId::Exact(boot_id) = report.observed_boot_id else {
        return Err(EvidenceError::StatusHasNoExactBootId);
    };
    verify_capabilities_and_faults(report.capabilities, report.faults)?;
    if !report.output_state.is_safe() {
        return Err(EvidenceError::OutputNotSafe(report.output_state));
    }
    if !report.controller_timer_pwm.is_zero() {
        return Err(EvidenceError::StatusTimerPwmNonzero(
            report.controller_timer_pwm,
        ));
    }
    if report.remaining_lease.get() != 0 {
        return Err(EvidenceError::StatusRemainingLeaseNonzero(
            report.remaining_lease.get(),
        ));
    }
    Ok(boot_id)
}

fn verify_acquire_result(
    config: &ClientConfig,
    expected_boot_id: ControllerBootId,
    expected_request_id: RequestId,
    result: AcquireResult,
) -> Result<ControlEpoch, EvidenceError> {
    verify_common_identity(config.controller_uid(), result.controller_uid)?;
    if result.boot_id != expected_boot_id {
        return Err(EvidenceError::ControllerBootIdMismatch {
            expected: expected_boot_id,
            actual: result.boot_id,
        });
    }
    if result.request_id != expected_request_id {
        return Err(EvidenceError::RequestIdMismatch {
            expected: expected_request_id,
            actual: result.request_id,
        });
    }
    if result.result != AcquireResultCode::Granted {
        return Err(EvidenceError::AcquireNotGranted(result.result));
    }
    let epoch = result
        .control_epoch
        .ok_or(EvidenceError::GrantedAcquireMissingEpoch)?;
    verify_capabilities_and_faults(result.capabilities, result.faults)?;
    if result.observed_firmware_abi != config.expected_firmware_abi() {
        return Err(EvidenceError::FirmwareAbiMismatch {
            expected: config.expected_firmware_abi(),
            actual: result.observed_firmware_abi,
        });
    }
    if result.observed_firmware_build_id != config.expected_firmware_build_id() {
        return Err(EvidenceError::FirmwareBuildIdMismatch {
            expected: config.expected_firmware_build_id(),
            actual: result.observed_firmware_build_id,
        });
    }
    if result.observed_actuator_config_fingerprint != config.expected_actuator_config_fingerprint()
    {
        return Err(EvidenceError::ActuatorConfigFingerprintMismatch);
    }
    Ok(epoch)
}

fn verify_command_result(
    session: ControllerSession,
    expected_sequence: V2CommandSequence,
    expected_pwm: TimerPwm,
    expected_lease: V2CommandLeaseMs,
    result: HostCommandResult,
) -> Result<(), EvidenceError> {
    verify_common_identity(session.controller_uid(), result.controller_uid)?;
    if result.boot_id != session.boot_id() {
        return Err(EvidenceError::ControllerBootIdMismatch {
            expected: session.boot_id(),
            actual: result.boot_id,
        });
    }
    if result.control_epoch != session.control_epoch() {
        return Err(EvidenceError::ControlEpochMismatch {
            expected: session.control_epoch(),
            actual: result.control_epoch,
        });
    }
    if result.sequence != expected_sequence {
        return Err(EvidenceError::SequenceMismatch {
            expected: expected_sequence,
            actual: result.sequence,
        });
    }
    let accepted_result = if expected_pwm.is_zero() {
        matches!(
            result.result,
            HostCommandResultCode::AppliedNew | HostCommandResultCode::Stopped
        )
    } else {
        result.result == HostCommandResultCode::AppliedNew
    };
    if !accepted_result {
        return Err(EvidenceError::CommandResultNotFreshlyApplied(result.result));
    }
    if result.requested_timer_pwm != expected_pwm {
        return Err(EvidenceError::RequestedTimerPwmMismatch {
            expected: expected_pwm,
            actual: result.requested_timer_pwm,
        });
    }
    if result.controller_timer_pwm != expected_pwm {
        return Err(EvidenceError::ControllerTimerPwmMismatch {
            expected: expected_pwm,
            actual: result.controller_timer_pwm,
        });
    }
    if expected_pwm.is_zero() && !result.output_state.is_safe() {
        return Err(EvidenceError::OutputNotSafe(result.output_state));
    }
    if !expected_pwm.is_zero() && result.output_state != OutputState::NonzeroPwm {
        return Err(EvidenceError::OutputStateMismatch {
            expected: OutputState::NonzeroPwm,
            actual: result.output_state,
        });
    }
    if !result.faults.is_clear() {
        return Err(EvidenceError::ControllerFaultsPresent(result.faults));
    }
    let remaining_ms = result.remaining_lease.get();
    if remaining_ms == 0 {
        return Err(EvidenceError::RemainingLeaseZero);
    }
    if remaining_ms > expected_lease.get() {
        return Err(EvidenceError::RemainingLeaseAboveRequested {
            remaining_ms,
            requested_ms: expected_lease.get(),
        });
    }
    let deadline_delta_ms = match result
        .controller_expires_at
        .relation_to(result.controller_applied_at)
    {
        DeadlineRelation::Future { remaining_ms } => remaining_ms,
        DeadlineRelation::Expired | DeadlineRelation::AmbiguousHalfRange => {
            return Err(EvidenceError::ControllerDeadlineNotFuture);
        }
    };
    if deadline_delta_ms > u32::from(expected_lease.get()) {
        return Err(EvidenceError::ControllerDeadlineAboveRequested {
            deadline_delta_ms,
            requested_ms: expected_lease.get(),
        });
    }
    if u32::from(remaining_ms) > deadline_delta_ms {
        return Err(EvidenceError::RemainingLeaseAboveControllerDeadline {
            remaining_ms,
            deadline_delta_ms,
        });
    }
    Ok(())
}

fn verify_stop_result(
    expected_uid: ControllerUid,
    expected_request_id: RequestId,
    result: HostStopResult,
    acknowledged_at: MonotonicInstant,
) -> Result<DisarmReceipt, EvidenceError> {
    verify_common_identity(expected_uid, result.controller_uid)?;
    if result.request_id != expected_request_id {
        return Err(EvidenceError::RequestIdMismatch {
            expected: expected_request_id,
            actual: result.request_id,
        });
    }
    if result.result != StopResultCode::ControllerConfirmed {
        return Err(EvidenceError::StopNotConfirmed(result.result));
    }
    if !result.output_state.is_safe() {
        return Err(EvidenceError::OutputNotSafe(result.output_state));
    }
    let TargetBootId::Exact(observed_boot_id) = result.observed_boot_id else {
        return Err(EvidenceError::StopResultHasNoExactBootId);
    };
    Ok(DisarmReceipt::new(
        result.controller_uid,
        observed_boot_id,
        result.request_id,
        result.output_state,
        result.faults,
        acknowledged_at,
    ))
}

impl fmt::Display for EvidenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for EvidenceError {}

impl<TransportError: fmt::Display> fmt::Display for FailureCause<TransportError> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Transport(source) => write!(formatter, "command transport uncertainty: {source}"),
            Self::Evidence(source) => write!(formatter, "command evidence mismatch: {source}"),
            Self::ClockRegressed { previous, observed } => write!(
                formatter,
                "monotonic clock regressed from {} ns to {} ns",
                previous.nanos_since_clock_start(),
                observed.nanos_since_clock_start()
            ),
            Self::MonotonicArithmeticOverflow => {
                formatter.write_str("monotonic deadline arithmetic overflowed")
            }
            Self::RequestIdExhausted => formatter.write_str("V2 request ID space is exhausted"),
            Self::CommandSequenceExhausted => {
                formatter.write_str("V2 command sequence space is exhausted")
            }
            Self::DeadlineExpiredBeforeSend {
                now,
                deadline_exclusive,
            } => write!(
                formatter,
                "command deadline {} ns was not later than send time {} ns",
                deadline_exclusive.nanos_since_clock_start(),
                now.nanos_since_clock_start()
            ),
            Self::ResponseAtOrAfterDeadline {
                acknowledged_at,
                deadline_exclusive,
            } => write!(
                formatter,
                "response at {} ns did not precede exclusive deadline {} ns",
                acknowledged_at.nanos_since_clock_start(),
                deadline_exclusive.nanos_since_clock_start()
            ),
            Self::LeaseNotKnownActiveAtAcknowledgement {
                acknowledged_at,
                known_active_through_exclusive,
            } => write!(
                formatter,
                "applied acknowledgement at {} ns did not precede conservative lease horizon {} ns",
                acknowledged_at.nanos_since_clock_start(),
                known_active_through_exclusive.nanos_since_clock_start()
            ),
            Self::PreviousAppliedEvidenceExpired {
                now,
                known_active_through_exclusive,
            } => write!(
                formatter,
                "previous applied evidence expired at {} ns before next command time {} ns",
                known_active_through_exclusive.nanos_since_clock_start(),
                now.nanos_since_clock_start()
            ),
        }
    }
}

impl<TransportError> FailureCause<TransportError> {
    fn stop_reason(&self) -> ForceStopReason {
        match self {
            Self::Evidence(error) => error.stop_reason(),
            Self::CommandSequenceExhausted => ForceStopReason::SequenceConflict,
            Self::DeadlineExpiredBeforeSend { .. }
            | Self::ResponseAtOrAfterDeadline { .. }
            | Self::LeaseNotKnownActiveAtAcknowledgement { .. }
            | Self::PreviousAppliedEvidenceExpired { .. } => ForceStopReason::LeaseExpired,
            Self::Transport(_)
            | Self::ClockRegressed { .. }
            | Self::MonotonicArithmeticOverflow
            | Self::RequestIdExhausted => ForceStopReason::TransportFault,
        }
    }
}

impl EvidenceError {
    const fn stop_reason(self) -> ForceStopReason {
        match self {
            Self::ControllerBootIdMismatch { .. }
            | Self::ControlEpochMismatch { .. }
            | Self::StatusHasControlEpoch(_)
            | Self::StatusHasNoExactBootId => ForceStopReason::SessionReset,
            Self::SequenceMismatch { .. } | Self::InitialAcquisitionWasNotSequenceZero => {
                ForceStopReason::SequenceConflict
            }
            Self::ControllerFaultsPresent(_) => ForceStopReason::ControllerFault,
            _ => ForceStopReason::TransportFault,
        }
    }
}

impl<TransportError> std::error::Error for FailureCause<TransportError>
where
    TransportError: std::error::Error + 'static,
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Transport(source) => Some(source),
            Self::Evidence(source) => Some(source),
            _ => None,
        }
    }
}

pub struct AcquireFailure<Transport, Clock>
where
    Transport: V2CommandTransport,
{
    cause: FailureCause<Transport::Error>,
    client: Box<LatchedCommandClient<Transport, Clock>>,
}

impl<Transport, Clock> AcquireFailure<Transport, Clock>
where
    Transport: V2CommandTransport,
    Clock: MonotonicClock,
{
    fn latch(
        mut core: ClientCore<Transport, Clock>,
        cause: FailureCause<Transport::Error>,
    ) -> Self {
        let attempts = core.config.stop_recovery().attempts().get();
        let recovery = core.recover_stop(cause.stop_reason(), attempts);
        Self {
            cause,
            client: Box::new(LatchedCommandClient {
                _core: core,
                recovery,
            }),
        }
    }

    pub const fn cause(&self) -> &FailureCause<Transport::Error> {
        &self.cause
    }

    pub const fn stop_knowledge(&self) -> LatchedStopKnowledge {
        self.client.stop_knowledge()
    }

    pub const fn recovery(&self) -> &StopRecoveryReport<Transport::Error> {
        self.client.recovery()
    }

    pub fn into_latched(self) -> LatchedCommandClient<Transport, Clock> {
        *self.client
    }
}

pub struct ApplyFailure<Transport, Clock>
where
    Transport: V2CommandTransport,
{
    cause: FailureCause<Transport::Error>,
    client: Box<LatchedCommandClient<Transport, Clock>>,
}

impl<Transport, Clock> ApplyFailure<Transport, Clock>
where
    Transport: V2CommandTransport,
    Clock: MonotonicClock,
{
    fn latch(
        mut core: ClientCore<Transport, Clock>,
        cause: FailureCause<Transport::Error>,
    ) -> Self {
        let attempts = core.config.stop_recovery().attempts().get();
        let recovery = core.recover_stop(cause.stop_reason(), attempts);
        Self {
            cause,
            client: Box::new(LatchedCommandClient {
                _core: core,
                recovery,
            }),
        }
    }

    pub const fn cause(&self) -> &FailureCause<Transport::Error> {
        &self.cause
    }

    pub const fn stop_knowledge(&self) -> LatchedStopKnowledge {
        self.client.stop_knowledge()
    }

    pub const fn recovery(&self) -> &StopRecoveryReport<Transport::Error> {
        self.client.recovery()
    }

    pub fn into_latched(self) -> LatchedCommandClient<Transport, Clock> {
        *self.client
    }
}

pub struct DisarmFailure<Transport, Clock>
where
    Transport: V2CommandTransport,
{
    cause: FailureCause<Transport::Error>,
    client: Box<LatchedCommandClient<Transport, Clock>>,
}

impl<Transport, Clock> DisarmFailure<Transport, Clock>
where
    Transport: V2CommandTransport,
    Clock: MonotonicClock,
{
    fn latch_after_first_attempt(
        mut core: ClientCore<Transport, Clock>,
        cause: FailureCause<Transport::Error>,
        reason: ForceStopReason,
    ) -> Self {
        let remaining_attempts = core
            .config
            .stop_recovery()
            .attempts()
            .get()
            .saturating_sub(1);
        let recovery = core.recover_stop(reason, remaining_attempts);
        Self {
            cause,
            client: Box::new(LatchedCommandClient {
                _core: core,
                recovery,
            }),
        }
    }

    pub const fn cause(&self) -> &FailureCause<Transport::Error> {
        &self.cause
    }

    pub const fn stop_knowledge(&self) -> LatchedStopKnowledge {
        self.client.stop_knowledge()
    }

    pub const fn recovery(&self) -> &StopRecoveryReport<Transport::Error> {
        self.client.recovery()
    }

    pub fn into_latched(self) -> LatchedCommandClient<Transport, Clock> {
        *self.client
    }
}
