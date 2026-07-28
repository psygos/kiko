//! Sole qualification-only owner of the candidate controller.
//!
//! This runtime accepts only raw, operator-authorized timer-duty events from
//! the qualification console. It has no autonomous, MPC, velocity, mapping,
//! or production-control input.

use std::fmt;
use std::time::Instant;

use robot_command_client::{AppliedCommandReceipt, DisarmReceipt, LatchedStopKnowledge};
use robot_protocol::v2::{
    ControlEpoch, ControllerBootId, ControllerUid, DomainError, TimerPwm, V2CommandSequence,
};

use super::actuation::{LiveActuationError, LocalRejectionStop};
use super::{
    AdmittedCandidatePwmTarget, CandidateActuationSessionError,
    CandidateActuationSessionStartError, CandidateCadenceOverflowStop, CandidatePwmAdmissionError,
    CandidatePwmRequest, CandidateRuntimeServiceIntervalError, ConsoleAppliedReceipt,
    ConsoleReceiptProjectionError, NavigationClockEpoch, NavigationIngressBoundaryError,
    NavigationIngressEvent, NavigationIngressRecord, OperatorClaimedWheelsOffAttestation,
    QualificationAppliedStepIngress, QualificationObservedAppliedZero, QualificationTimerPwmPair,
    StoppedWheelsOffCandidateController, WheelsOffCandidateActuationSession,
    WheelsOffCandidateLimits, WheelsOffCandidateRuntimeServiceInterval,
    WheelsOffQualificationAppliedStep, WheelsOffQualificationAppliedStepRecordError,
    WheelsOffQualificationCompletionError, WheelsOffQualificationConsoleHandle,
    WheelsOffQualificationDisconnectError, WheelsOffQualificationEventId,
    WheelsOffQualificationFrontendState, WheelsOffQualificationIngressEvent,
    WheelsOffQualificationIngressReceiver, WheelsOffQualificationMotionAuthorityEnableError,
    WheelsOffQualificationReceiveError, WheelsOffQualificationSnapshot,
    WheelsOffQualificationTerminalCompletion,
};
use crate::HostMonotonicTimestamp;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WheelsOffQualificationControllerState {
    StoppedWithExactReceipt,
    Active,
    StopConfirmedWithoutRetainedReceipt,
    StopUncertain,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WheelsOffQualificationRuntimeState {
    Running,
    Faulted,
    Shutdown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WheelsOffQualificationMotionAuthorityState {
    PendingAttestation,
    Enabled,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WheelsOffQualificationFailClosedStop {
    ReusedExactReceipt,
    NewExactReceipt,
    ControllerConfirmedWithoutRetainedReceipt,
    Uncertain,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WheelsOffQualificationRuntimeTick {
    Idle,
    CandidateTargetRetained {
        event_id: WheelsOffQualificationEventId,
    },
    CandidateStepApplied {
        pending: PendingWheelsOffQualificationAppliedStep,
    },
    TerminalStopCompleted {
        event_id: WheelsOffQualificationEventId,
        reused_stopped_receipt: bool,
    },
}

/// Exact evidence returned only after the command client has verified one
/// controller application. It cannot be published in the qualifier snapshot
/// until a matching navigation-journal record is supplied.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PendingWheelsOffQualificationAppliedStep {
    event_id: WheelsOffQualificationEventId,
    requested_target: QualificationTimerPwmPair,
    requested_target_timer_pwm: TimerPwm,
    controller_uid: ControllerUid,
    controller_boot_id: ControllerBootId,
    control_epoch: ControlEpoch,
    controller_sequence: V2CommandSequence,
    controller_requested_step: TimerPwm,
    actual_applied: TimerPwm,
    target_reached: bool,
    receipt: ConsoleAppliedReceipt,
}

impl PendingWheelsOffQualificationAppliedStep {
    pub const fn event_id(self) -> WheelsOffQualificationEventId {
        self.event_id
    }

    pub const fn requested_target(self) -> QualificationTimerPwmPair {
        self.requested_target
    }

    pub const fn controller_sequence(self) -> V2CommandSequence {
        self.controller_sequence
    }

    pub const fn actual_applied(self) -> TimerPwm {
        self.actual_applied
    }

    pub const fn target_reached(self) -> bool {
        self.target_reached
    }

    pub fn journal_event(
        self,
        clock_epoch: NavigationClockEpoch,
        observed_at: HostMonotonicTimestamp,
    ) -> Result<QualificationAppliedStepIngress, NavigationIngressBoundaryError> {
        QualificationAppliedStepIngress::parse(
            clock_epoch,
            observed_at,
            self.event_id.as_nonzero(),
            self.controller_uid,
            self.controller_boot_id,
            self.control_epoch,
            self.controller_sequence,
            self.requested_target_timer_pwm,
            self.controller_requested_step,
            self.actual_applied,
            self.target_reached,
        )
    }

    pub fn bind_journal_record(
        self,
        journal_event: QualificationAppliedStepIngress,
        record: NavigationIngressRecord,
    ) -> Result<WheelsOffQualificationAppliedStep, WheelsOffQualificationAppliedStepJournalError>
    {
        if record.event() != NavigationIngressEvent::QualificationAppliedStep(journal_event)
            || journal_event.qualification_event_id() != self.event_id.as_nonzero()
            || journal_event.controller_sequence() != self.controller_sequence
            || journal_event.actual_applied() != self.actual_applied
        {
            return Err(WheelsOffQualificationAppliedStepJournalError::RecordMismatch);
        }
        Ok(WheelsOffQualificationAppliedStep::from_journaled_parts(
            self.event_id,
            record.sequence().as_u64(),
            self.requested_target,
            self.target_reached,
            self.receipt,
        ))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WheelsOffQualificationAppliedStepJournalError {
    RecordMismatch,
}

impl fmt::Display for WheelsOffQualificationAppliedStepJournalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("qualification applied-step journal record does not match its receipt")
    }
}

impl std::error::Error for WheelsOffQualificationAppliedStepJournalError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WheelsOffQualificationRuntimeShutdown {
    pub terminal_completion: WheelsOffQualificationTerminalCompletion,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WheelsOffQualificationRuntimeStartError {
    InitialAppliedCommandWasNotExactSafeZero,
    InitialDisarmWasNotExactSafeStop,
    InitialReceiptControllerMismatch,
    InitialReceiptBootMismatch,
    ConsoleFrontendNotConnected,
    ConsoleFrontendDisconnected,
    ConsoleMotionAuthorityAlreadyEnabled,
    ConsoleMaximumPwmMismatch {
        console: u8,
        controller: u8,
    },
    ConsoleTestPwmMismatch {
        console: u8,
        controller: u8,
    },
    ConsoleDeadmanMismatch,
    RuntimeServiceMaximumMismatch {
        admitted: std::time::Duration,
        controller: std::time::Duration,
    },
    RuntimeServiceInterval(CandidateRuntimeServiceIntervalError),
}

impl fmt::Display for WheelsOffQualificationRuntimeStartError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "wheels-off qualification runtime start rejected: {self:?}"
        )
    }
}

impl std::error::Error for WheelsOffQualificationRuntimeStartError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::RuntimeServiceInterval(source) => Some(source),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WheelsOffQualificationMotionAuthorityEnableFailure {
    RuntimeNotRunning(WheelsOffQualificationRuntimeState),
    AlreadyEnabled,
    ControllerNotExactlyStopped(WheelsOffQualificationControllerState),
    Attestation(CandidatePwmAdmissionError),
    Console(WheelsOffQualificationMotionAuthorityEnableError),
}

impl fmt::Display for WheelsOffQualificationMotionAuthorityEnableFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "wheels-off qualification motion authority enablement rejected: {self:?}"
        )
    }
}

impl std::error::Error for WheelsOffQualificationMotionAuthorityEnableFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Attestation(source) => Some(source),
            Self::Console(source) => Some(source),
            Self::RuntimeNotRunning(_)
            | Self::AlreadyEnabled
            | Self::ControllerNotExactlyStopped(_) => None,
        }
    }
}

pub enum WheelsOffQualificationRuntimeFailure {
    RuntimeNotRunning(WheelsOffQualificationRuntimeState),
    MotionAuthorityPending,
    ReceiverDisconnected,
    CandidateRequest(DomainError),
    CandidateAdmission(CandidatePwmAdmissionError),
    Reacquire(Box<LiveActuationError>),
    ReacquireCadenceDeadlineOverflow,
    Apply(Box<LiveActuationError>),
    ReceiptProjection(ConsoleReceiptProjectionError),
    AppliedStepRecord(WheelsOffQualificationAppliedStepRecordError),
    ApplyCadenceDeadlineOverflow,
    RefreshDeadlineOverflow,
    TerminalCompletion(WheelsOffQualificationCompletionError),
    ShutdownDisconnect(WheelsOffQualificationDisconnectError),
    ExactStopReceiptUnavailable,
}

impl fmt::Debug for WheelsOffQualificationRuntimeFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, formatter)
    }
}

impl fmt::Display for WheelsOffQualificationRuntimeFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RuntimeNotRunning(state) => {
                write!(formatter, "runtime is not running ({state:?})")
            }
            Self::MotionAuthorityPending => {
                formatter.write_str("nonzero motion authority is pending attended attestation")
            }
            Self::ReceiverDisconnected => formatter.write_str("console receiver disconnected"),
            Self::CandidateRequest(source) => write!(formatter, "invalid candidate PWM: {source}"),
            Self::CandidateAdmission(source) => write!(formatter, "{source}"),
            Self::Reacquire(source) => write!(formatter, "zero reacquisition failed: {source}"),
            Self::ReacquireCadenceDeadlineOverflow => {
                formatter.write_str("zero reacquisition cadence deadline overflowed")
            }
            Self::Apply(source) => write!(formatter, "candidate application failed: {source}"),
            Self::ReceiptProjection(source) => {
                write!(formatter, "candidate receipt projection failed: {source}")
            }
            Self::AppliedStepRecord(source) => source.fmt(formatter),
            Self::ApplyCadenceDeadlineOverflow => {
                formatter.write_str("candidate refresh cadence deadline overflowed")
            }
            Self::RefreshDeadlineOverflow => {
                formatter.write_str("qualification runtime refresh deadline overflowed")
            }
            Self::TerminalCompletion(source) => write!(formatter, "{source}"),
            Self::ShutdownDisconnect(source) => write!(formatter, "{source}"),
            Self::ExactStopReceiptUnavailable => {
                formatter.write_str("an exact safe stop receipt is unavailable")
            }
        }
    }
}

impl std::error::Error for WheelsOffQualificationRuntimeFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::CandidateRequest(source) => Some(source),
            Self::CandidateAdmission(source) => Some(source),
            Self::Reacquire(source) | Self::Apply(source) => Some(source),
            Self::ReceiptProjection(source) => Some(source),
            Self::AppliedStepRecord(source) => Some(source),
            Self::TerminalCompletion(source) => Some(source),
            Self::ShutdownDisconnect(source) => Some(source),
            _ => None,
        }
    }
}

pub struct WheelsOffQualificationRuntimeError {
    failure: WheelsOffQualificationRuntimeFailure,
    fail_closed_stop: WheelsOffQualificationFailClosedStop,
    stop_error: Option<Box<LiveActuationError>>,
}

impl WheelsOffQualificationRuntimeError {
    pub const fn failure(&self) -> &WheelsOffQualificationRuntimeFailure {
        &self.failure
    }

    pub const fn fail_closed_stop(&self) -> WheelsOffQualificationFailClosedStop {
        self.fail_closed_stop
    }

    pub fn stop_error(&self) -> Option<&LiveActuationError> {
        self.stop_error.as_deref()
    }
}

impl fmt::Debug for WheelsOffQualificationRuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, formatter)
    }
}

impl fmt::Display for WheelsOffQualificationRuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{}; fail-closed stop: {:?}",
            self.failure, self.fail_closed_stop
        )?;
        if let Some(stop_error) = &self.stop_error {
            write!(formatter, " ({stop_error})")?;
        }
        Ok(())
    }
}

impl std::error::Error for WheelsOffQualificationRuntimeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.failure)
    }
}

enum CandidateControllerOwner {
    Stopped(Box<StoppedWheelsOffCandidateController>),
    Active(Box<WheelsOffCandidateActuationSession>),
    ConfirmedWithoutRetainedReceipt,
    Uncertain,
    Transitioning,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WheelsOffQualificationMotionAuthority {
    PendingAttestation,
    Enabled(OperatorClaimedWheelsOffAttestation),
}

impl WheelsOffQualificationMotionAuthority {
    const fn state(self) -> WheelsOffQualificationMotionAuthorityState {
        match self {
            Self::PendingAttestation => {
                WheelsOffQualificationMotionAuthorityState::PendingAttestation
            }
            Self::Enabled(_) => WheelsOffQualificationMotionAuthorityState::Enabled,
        }
    }

    const fn attestation(self) -> Option<OperatorClaimedWheelsOffAttestation> {
        match self {
            Self::PendingAttestation => None,
            Self::Enabled(attestation) => Some(attestation),
        }
    }
}

#[derive(Clone, Copy)]
struct RetainedTarget {
    event_id: WheelsOffQualificationEventId,
    requested_pwm: QualificationTimerPwmPair,
    target: AdmittedCandidatePwmTarget,
}

#[must_use = "qualification controller ownership requires explicit shutdown evidence"]
pub struct WheelsOffQualificationRuntime {
    controller: CandidateControllerOwner,
    stopped_applied: Option<AppliedCommandReceipt>,
    last_stop: Option<DisarmReceipt>,
    limits: WheelsOffCandidateLimits,
    motion_authority: WheelsOffQualificationMotionAuthority,
    console: WheelsOffQualificationConsoleHandle,
    receiver: WheelsOffQualificationIngressReceiver,
    target: Option<RetainedTarget>,
    next_refresh_not_before: Option<Instant>,
    state: WheelsOffQualificationRuntimeState,
}

impl WheelsOffQualificationRuntime {
    #[allow(clippy::too_many_arguments)]
    pub fn try_new_pending(
        stopped: StoppedWheelsOffCandidateController,
        initial_applied: AppliedCommandReceipt,
        initial_stop: DisarmReceipt,
        limits: WheelsOffCandidateLimits,
        admitted_runtime_service_interval: WheelsOffCandidateRuntimeServiceInterval,
        actual_runtime_service_interval: std::time::Duration,
        console: WheelsOffQualificationConsoleHandle,
        receiver: WheelsOffQualificationIngressReceiver,
    ) -> Result<Self, WheelsOffQualificationRuntimeStartError> {
        if !exact_safe_applied_zero(&initial_applied) {
            return Err(
                WheelsOffQualificationRuntimeStartError::InitialAppliedCommandWasNotExactSafeZero,
            );
        }
        if !exact_safe_disarm(&initial_stop) {
            return Err(WheelsOffQualificationRuntimeStartError::InitialDisarmWasNotExactSafeStop);
        }
        let session = initial_applied.controller_session();
        if session.controller_uid() != initial_stop.controller_uid() {
            return Err(WheelsOffQualificationRuntimeStartError::InitialReceiptControllerMismatch);
        }
        if session.boot_id() != initial_stop.observed_boot_id() {
            return Err(WheelsOffQualificationRuntimeStartError::InitialReceiptBootMismatch);
        }
        let console_snapshot = console.snapshot();
        admit_console_before_runtime(&console_snapshot)?;
        let profile = console_snapshot.control_profile;
        if profile.maximum_abs_timer_pwm_percent() != limits.effective_max_abs_pwm_percent() {
            return Err(
                WheelsOffQualificationRuntimeStartError::ConsoleMaximumPwmMismatch {
                    console: profile.maximum_abs_timer_pwm_percent(),
                    controller: limits.effective_max_abs_pwm_percent(),
                },
            );
        }
        if profile.manual_test_magnitude_timer_pwm_percent()
            != limits.manual_test_magnitude_timer_pwm_percent()
        {
            return Err(
                WheelsOffQualificationRuntimeStartError::ConsoleTestPwmMismatch {
                    console: profile.manual_test_magnitude_timer_pwm_percent(),
                    controller: limits.manual_test_magnitude_timer_pwm_percent(),
                },
            );
        }
        if std::time::Duration::from_millis(profile.manual_deadman_ms()) != limits.manual_deadman()
        {
            return Err(WheelsOffQualificationRuntimeStartError::ConsoleDeadmanMismatch);
        }
        if admitted_runtime_service_interval.maximum() != limits.maximum_runtime_service_interval()
        {
            return Err(
                WheelsOffQualificationRuntimeStartError::RuntimeServiceMaximumMismatch {
                    admitted: admitted_runtime_service_interval.maximum(),
                    controller: limits.maximum_runtime_service_interval(),
                },
            );
        }
        admitted_runtime_service_interval
            .require_exact_runtime_interval(actual_runtime_service_interval)
            .map_err(WheelsOffQualificationRuntimeStartError::RuntimeServiceInterval)?;
        Ok(Self {
            controller: CandidateControllerOwner::Stopped(Box::new(stopped)),
            stopped_applied: Some(initial_applied),
            last_stop: Some(initial_stop),
            limits,
            motion_authority: WheelsOffQualificationMotionAuthority::PendingAttestation,
            console,
            receiver,
            target: None,
            next_refresh_not_before: None,
            state: WheelsOffQualificationRuntimeState::Running,
        })
    }

    pub const fn limits(&self) -> WheelsOffCandidateLimits {
        self.limits
    }

    pub const fn state(&self) -> WheelsOffQualificationRuntimeState {
        self.state
    }

    pub const fn motion_authority_state(&self) -> WheelsOffQualificationMotionAuthorityState {
        self.motion_authority.state()
    }

    pub fn enable_motion_authority(
        &mut self,
        attestation: OperatorClaimedWheelsOffAttestation,
    ) -> Result<(), WheelsOffQualificationMotionAuthorityEnableFailure> {
        if self.state != WheelsOffQualificationRuntimeState::Running {
            return Err(
                WheelsOffQualificationMotionAuthorityEnableFailure::RuntimeNotRunning(self.state),
            );
        }
        if self.motion_authority != WheelsOffQualificationMotionAuthority::PendingAttestation {
            return Err(WheelsOffQualificationMotionAuthorityEnableFailure::AlreadyEnabled);
        }
        let CandidateControllerOwner::Stopped(stopped) = &self.controller else {
            return Err(
                WheelsOffQualificationMotionAuthorityEnableFailure::ControllerNotExactlyStopped(
                    self.controller_state(),
                ),
            );
        };
        stopped
            .require_fresh_attestation(&attestation, Instant::now())
            .map_err(WheelsOffQualificationMotionAuthorityEnableFailure::Attestation)?;
        // The runtime owner is single-threaded, while HTTP submission can
        // concurrently lock the console. Install the runtime-side authority
        // first while the console is still closed, then publish the console
        // generation. A concurrent request therefore sees either the old
        // closed generation or a console generation whose sole consumer
        // already carries this attestation. No candidate event is consumed
        // until a later owner tick. Roll back if the console refuses the
        // transition so the two authorities cannot remain split.
        self.motion_authority = WheelsOffQualificationMotionAuthority::Enabled(attestation);
        if let Err(source) = self.console.enable_motion_authority() {
            self.motion_authority = WheelsOffQualificationMotionAuthority::PendingAttestation;
            return Err(WheelsOffQualificationMotionAuthorityEnableFailure::Console(
                source,
            ));
        }
        Ok(())
    }

    pub fn controller_state(&self) -> WheelsOffQualificationControllerState {
        match self.controller {
            CandidateControllerOwner::Stopped(_) => {
                WheelsOffQualificationControllerState::StoppedWithExactReceipt
            }
            CandidateControllerOwner::Active(_) => WheelsOffQualificationControllerState::Active,
            CandidateControllerOwner::ConfirmedWithoutRetainedReceipt => {
                WheelsOffQualificationControllerState::StopConfirmedWithoutRetainedReceipt
            }
            CandidateControllerOwner::Uncertain | CandidateControllerOwner::Transitioning => {
                WheelsOffQualificationControllerState::StopUncertain
            }
        }
    }

    pub fn last_applied(&self) -> Option<&AppliedCommandReceipt> {
        match &self.controller {
            CandidateControllerOwner::Stopped(_) => self.stopped_applied.as_ref(),
            CandidateControllerOwner::Active(session) => Some(session.last_applied()),
            _ => None,
        }
    }

    pub const fn last_stop(&self) -> Option<&DisarmReceipt> {
        self.last_stop.as_ref()
    }

    pub fn console_snapshot(&self) -> WheelsOffQualificationSnapshot {
        self.console.snapshot()
    }

    /// Queue the console's existing process-lifetime fail-closed terminal
    /// transition. The next runtime tick drains this stop before any retained
    /// candidate command.
    pub fn signal_internal_fail_closed(&self, observed_at: Option<HostMonotonicTimestamp>) {
        self.console.signal_internal_fail_closed(observed_at);
    }

    pub fn tick(
        &mut self,
    ) -> Result<WheelsOffQualificationRuntimeTick, WheelsOffQualificationRuntimeError> {
        if self.state != WheelsOffQualificationRuntimeState::Running {
            return Err(self.error_without_new_stop(
                WheelsOffQualificationRuntimeFailure::RuntimeNotRunning(self.state),
            ));
        }
        let now = Instant::now();

        match self.receiver.try_recv() {
            Ok(WheelsOffQualificationIngressEvent::TerminalStop(event)) => {
                self.handle_terminal(event.event_id())
            }
            Ok(WheelsOffQualificationIngressEvent::CandidatePwm(event)) => {
                let pair = event.requested_pwm();
                let requested = CandidatePwmRequest::try_new(
                    pair.left_timer_pwm_percent.get(),
                    pair.right_timer_pwm_percent.get(),
                )
                .map_err(|source| {
                    self.fail_closed(WheelsOffQualificationRuntimeFailure::CandidateRequest(
                        source,
                    ))
                })?;
                self.handle_candidate(event.event_id(), pair, requested, now)
            }
            Err(WheelsOffQualificationReceiveError::Empty) => self.refresh_retained_target(now),
            Err(WheelsOffQualificationReceiveError::RuntimeReceiverDisconnected) => {
                Err(self.fail_closed(WheelsOffQualificationRuntimeFailure::ReceiverDisconnected))
            }
        }
    }

    pub fn shutdown(
        &mut self,
    ) -> Result<WheelsOffQualificationRuntimeShutdown, WheelsOffQualificationRuntimeError> {
        if self.state == WheelsOffQualificationRuntimeState::Shutdown {
            return Err(self.error_without_new_stop(
                WheelsOffQualificationRuntimeFailure::RuntimeNotRunning(self.state),
            ));
        }
        self.target = None;
        self.next_refresh_not_before = None;
        let (stop, stop_error) = self.attempt_stop();
        if !matches!(
            stop,
            WheelsOffQualificationFailClosedStop::ReusedExactReceipt
                | WheelsOffQualificationFailClosedStop::NewExactReceipt
        ) {
            self.state = WheelsOffQualificationRuntimeState::Faulted;
            return Err(WheelsOffQualificationRuntimeError {
                failure: WheelsOffQualificationRuntimeFailure::ExactStopReceiptUnavailable,
                fail_closed_stop: stop,
                stop_error,
            });
        }
        let observed = self.observed_exact_stop().ok_or_else(|| {
            self.state = WheelsOffQualificationRuntimeState::Faulted;
            WheelsOffQualificationRuntimeError {
                failure: WheelsOffQualificationRuntimeFailure::ExactStopReceiptUnavailable,
                fail_closed_stop: stop,
                stop_error: None,
            }
        })?;
        let terminal_completion = self
            .receiver
            .disconnect_after_confirmed_zero(observed)
            .map_err(|source| {
                self.state = WheelsOffQualificationRuntimeState::Faulted;
                WheelsOffQualificationRuntimeError {
                    failure: WheelsOffQualificationRuntimeFailure::ShutdownDisconnect(source),
                    fail_closed_stop: stop,
                    stop_error: None,
                }
            })?;
        self.state = WheelsOffQualificationRuntimeState::Shutdown;
        Ok(WheelsOffQualificationRuntimeShutdown {
            terminal_completion,
        })
    }

    pub fn record_journaled_applied_step(
        &mut self,
        step: WheelsOffQualificationAppliedStep,
    ) -> Result<(), WheelsOffQualificationRuntimeError> {
        if self.state != WheelsOffQualificationRuntimeState::Running {
            return Err(self.error_without_new_stop(
                WheelsOffQualificationRuntimeFailure::RuntimeNotRunning(self.state),
            ));
        }
        self.console.record_applied_step(step).map_err(|source| {
            self.fail_closed(WheelsOffQualificationRuntimeFailure::AppliedStepRecord(
                source,
            ))
        })
    }

    fn handle_terminal(
        &mut self,
        event_id: WheelsOffQualificationEventId,
    ) -> Result<WheelsOffQualificationRuntimeTick, WheelsOffQualificationRuntimeError> {
        self.target = None;
        self.next_refresh_not_before = None;
        let was_stopped = matches!(self.controller, CandidateControllerOwner::Stopped(_));
        let (stop, stop_error) = self.attempt_stop();
        if !matches!(
            stop,
            WheelsOffQualificationFailClosedStop::ReusedExactReceipt
                | WheelsOffQualificationFailClosedStop::NewExactReceipt
        ) {
            self.console.signal_internal_fail_closed(None);
            self.state = WheelsOffQualificationRuntimeState::Faulted;
            return Err(WheelsOffQualificationRuntimeError {
                failure: WheelsOffQualificationRuntimeFailure::ExactStopReceiptUnavailable,
                fail_closed_stop: stop,
                stop_error,
            });
        }
        let observed = self.observed_exact_stop().ok_or_else(|| {
            self.console.signal_internal_fail_closed(None);
            self.state = WheelsOffQualificationRuntimeState::Faulted;
            WheelsOffQualificationRuntimeError {
                failure: WheelsOffQualificationRuntimeFailure::ExactStopReceiptUnavailable,
                fail_closed_stop: stop,
                stop_error: None,
            }
        })?;
        self.receiver
            .complete_terminal_stop(event_id, observed)
            .map_err(|source| {
                self.fail_closed(WheelsOffQualificationRuntimeFailure::TerminalCompletion(
                    source,
                ))
            })?;
        Ok(WheelsOffQualificationRuntimeTick::TerminalStopCompleted {
            event_id,
            reused_stopped_receipt: was_stopped,
        })
    }

    fn handle_candidate(
        &mut self,
        event_id: WheelsOffQualificationEventId,
        requested_pwm: QualificationTimerPwmPair,
        requested: CandidatePwmRequest,
        now: Instant,
    ) -> Result<WheelsOffQualificationRuntimeTick, WheelsOffQualificationRuntimeError> {
        let Some(attestation) = self.motion_authority.attestation() else {
            return Err(
                self.fail_closed(WheelsOffQualificationRuntimeFailure::MotionAuthorityPending)
            );
        };
        if let CandidateControllerOwner::Stopped(stopped) = &self.controller
            && let Err(source) = stopped.require_fresh_attestation(&attestation, now)
        {
            return Err(self.fail_closed(
                WheelsOffQualificationRuntimeFailure::CandidateAdmission(source),
            ));
        }
        if matches!(self.controller, CandidateControllerOwner::Stopped(_)) {
            let previous = std::mem::replace(
                &mut self.controller,
                CandidateControllerOwner::Transitioning,
            );
            let CandidateControllerOwner::Stopped(stopped) = previous else {
                unreachable!("stopped match and linear take agree");
            };
            match (*stopped).reacquire_zero() {
                Ok(session) => {
                    self.controller = CandidateControllerOwner::Active(Box::new(session));
                    self.next_refresh_not_before = now.checked_add(self.limits.command_interval());
                    if self.next_refresh_not_before.is_none() {
                        return Err(self.fail_closed(
                            WheelsOffQualificationRuntimeFailure::RefreshDeadlineOverflow,
                        ));
                    }
                }
                Err(CandidateActuationSessionStartError::Actuation(source)) => {
                    let stop = stop_knowledge_from_live_error(&source);
                    self.controller = owner_after_unretained_stop(stop);
                    self.state = WheelsOffQualificationRuntimeState::Faulted;
                    self.console.signal_internal_fail_closed(None);
                    return Err(WheelsOffQualificationRuntimeError {
                        failure: WheelsOffQualificationRuntimeFailure::Reacquire(Box::new(source)),
                        fail_closed_stop: stop,
                        stop_error: None,
                    });
                }
                Err(CandidateActuationSessionStartError::CadenceDeadlineOverflow { stop }) => {
                    let (stop, stop_error) = self.install_cadence_stop(stop);
                    self.state = WheelsOffQualificationRuntimeState::Faulted;
                    self.console.signal_internal_fail_closed(None);
                    return Err(WheelsOffQualificationRuntimeError {
                        failure:
                            WheelsOffQualificationRuntimeFailure::ReacquireCadenceDeadlineOverflow,
                        fail_closed_stop: stop,
                        stop_error,
                    });
                }
            }
        }
        let admitted = match &self.controller {
            CandidateControllerOwner::Active(session) => session
                .admit_target(requested, Some(&attestation))
                .map_err(|source| {
                    self.fail_closed(WheelsOffQualificationRuntimeFailure::CandidateAdmission(
                        source,
                    ))
                })?,
            _ => {
                return Err(self.fail_closed(
                    WheelsOffQualificationRuntimeFailure::ExactStopReceiptUnavailable,
                ));
            }
        };
        self.target = Some(RetainedTarget {
            event_id,
            requested_pwm,
            target: admitted,
        });
        match self.refresh_retained_target(now)? {
            WheelsOffQualificationRuntimeTick::Idle => {
                Ok(WheelsOffQualificationRuntimeTick::CandidateTargetRetained { event_id })
            }
            outcome => Ok(outcome),
        }
    }

    fn refresh_retained_target(
        &mut self,
        now: Instant,
    ) -> Result<WheelsOffQualificationRuntimeTick, WheelsOffQualificationRuntimeError> {
        let Some(retained) = self.target else {
            return Ok(WheelsOffQualificationRuntimeTick::Idle);
        };
        if let Err(source) = retained.target.require_fresh(now) {
            return Err(self.fail_closed(
                WheelsOffQualificationRuntimeFailure::CandidateAdmission(source),
            ));
        }
        if !refresh_due(now, self.next_refresh_not_before) {
            return Ok(WheelsOffQualificationRuntimeTick::Idle);
        }
        let result = match &mut self.controller {
            CandidateControllerOwner::Active(session) => session.apply_next_step(retained.target),
            _ => {
                return Err(self.fail_closed(
                    WheelsOffQualificationRuntimeFailure::ExactStopReceiptUnavailable,
                ));
            }
        };
        match result {
            Ok(receipt) => {
                let controller_session = receipt.controller_session();
                let verified_result = receipt.verified_host_result();
                let controller_sequence = receipt.sequence();
                let actual_applied = receipt.applied_timer_pwm();
                let projected_receipt = match ConsoleAppliedReceipt::from_verified(receipt) {
                    Ok(receipt) => receipt,
                    Err(source) => {
                        return Err(self.fail_closed(
                            WheelsOffQualificationRuntimeFailure::ReceiptProjection(source),
                        ));
                    }
                };
                self.next_refresh_not_before =
                    Instant::now().checked_add(self.limits.command_interval());
                if self.next_refresh_not_before.is_none() {
                    return Err(self.fail_closed(
                        WheelsOffQualificationRuntimeFailure::RefreshDeadlineOverflow,
                    ));
                }
                Ok(WheelsOffQualificationRuntimeTick::CandidateStepApplied {
                    pending: PendingWheelsOffQualificationAppliedStep {
                        event_id: retained.event_id,
                        requested_target: retained.requested_pwm,
                        requested_target_timer_pwm: retained.target.timer_pwm(),
                        controller_uid: controller_session.controller_uid(),
                        controller_boot_id: controller_session.boot_id(),
                        control_epoch: controller_session.control_epoch(),
                        controller_sequence,
                        controller_requested_step: verified_result.requested_timer_pwm,
                        actual_applied,
                        target_reached: actual_applied == retained.target.timer_pwm(),
                        receipt: projected_receipt,
                    },
                })
            }
            Err(CandidateActuationSessionError::CommandCadenceNotElapsed { remaining, .. }) => {
                self.next_refresh_not_before = Instant::now().checked_add(remaining);
                if self.next_refresh_not_before.is_none() {
                    return Err(self.fail_closed(
                        WheelsOffQualificationRuntimeFailure::RefreshDeadlineOverflow,
                    ));
                }
                Ok(WheelsOffQualificationRuntimeTick::Idle)
            }
            Err(CandidateActuationSessionError::Admission(source)) => Err(self.fail_closed(
                WheelsOffQualificationRuntimeFailure::CandidateAdmission(source),
            )),
            Err(CandidateActuationSessionError::Actuation(source)) => Err(self.fail_closed(
                WheelsOffQualificationRuntimeFailure::Apply(Box::new(source)),
            )),
            Err(CandidateActuationSessionError::CadenceDeadlineOverflow { stop }) => {
                let (stop, stop_error) = self.install_cadence_stop(stop);
                self.state = WheelsOffQualificationRuntimeState::Faulted;
                self.console.signal_internal_fail_closed(None);
                Err(WheelsOffQualificationRuntimeError {
                    failure: WheelsOffQualificationRuntimeFailure::ApplyCadenceDeadlineOverflow,
                    fail_closed_stop: stop,
                    stop_error,
                })
            }
        }
    }

    fn fail_closed(
        &mut self,
        failure: WheelsOffQualificationRuntimeFailure,
    ) -> WheelsOffQualificationRuntimeError {
        let primary_stop = failure_live_stop_knowledge(&failure);
        self.console.signal_internal_fail_closed(None);
        self.target = None;
        self.next_refresh_not_before = None;
        let (mut fail_closed_stop, stop_error) = self.attempt_stop();
        if fail_closed_stop == WheelsOffQualificationFailClosedStop::Uncertain
            && primary_stop
                == Some(
                    WheelsOffQualificationFailClosedStop::ControllerConfirmedWithoutRetainedReceipt,
                )
        {
            fail_closed_stop =
                WheelsOffQualificationFailClosedStop::ControllerConfirmedWithoutRetainedReceipt;
            self.controller = CandidateControllerOwner::ConfirmedWithoutRetainedReceipt;
        }
        self.state = WheelsOffQualificationRuntimeState::Faulted;
        WheelsOffQualificationRuntimeError {
            failure,
            fail_closed_stop,
            stop_error,
        }
    }

    fn error_without_new_stop(
        &self,
        failure: WheelsOffQualificationRuntimeFailure,
    ) -> WheelsOffQualificationRuntimeError {
        WheelsOffQualificationRuntimeError {
            failure,
            fail_closed_stop: match self.controller {
                CandidateControllerOwner::Stopped(_) => {
                    WheelsOffQualificationFailClosedStop::ReusedExactReceipt
                }
                CandidateControllerOwner::ConfirmedWithoutRetainedReceipt => {
                    WheelsOffQualificationFailClosedStop::ControllerConfirmedWithoutRetainedReceipt
                }
                CandidateControllerOwner::Active(_)
                | CandidateControllerOwner::Uncertain
                | CandidateControllerOwner::Transitioning => {
                    WheelsOffQualificationFailClosedStop::Uncertain
                }
            },
            stop_error: None,
        }
    }

    fn attempt_stop(
        &mut self,
    ) -> (
        WheelsOffQualificationFailClosedStop,
        Option<Box<LiveActuationError>>,
    ) {
        let owner = std::mem::replace(
            &mut self.controller,
            CandidateControllerOwner::Transitioning,
        );
        match owner {
            CandidateControllerOwner::Stopped(stopped) => {
                if self.last_stop.as_ref().is_some_and(exact_safe_disarm) {
                    self.controller = CandidateControllerOwner::Stopped(stopped);
                    (
                        WheelsOffQualificationFailClosedStop::ReusedExactReceipt,
                        None,
                    )
                } else {
                    self.controller = CandidateControllerOwner::Uncertain;
                    (WheelsOffQualificationFailClosedStop::Uncertain, None)
                }
            }
            CandidateControllerOwner::Active(session) => {
                match (*session).stop_now_with_last_applied() {
                    Ok((stopped, applied, receipt)) if exact_safe_disarm(&receipt) => {
                        self.controller = CandidateControllerOwner::Stopped(Box::new(stopped));
                        self.stopped_applied = Some(applied);
                        self.last_stop = Some(receipt);
                        (WheelsOffQualificationFailClosedStop::NewExactReceipt, None)
                    }
                    Ok((_stopped, _applied, _receipt)) => {
                        self.controller = CandidateControllerOwner::Uncertain;
                        (WheelsOffQualificationFailClosedStop::Uncertain, None)
                    }
                    Err(source) => {
                        let stop = stop_knowledge_from_live_error(&source);
                        self.controller = owner_after_unretained_stop(stop);
                        (stop, Some(Box::new(source)))
                    }
                }
            }
            CandidateControllerOwner::ConfirmedWithoutRetainedReceipt => {
                self.controller = CandidateControllerOwner::ConfirmedWithoutRetainedReceipt;
                (
                    WheelsOffQualificationFailClosedStop::ControllerConfirmedWithoutRetainedReceipt,
                    None,
                )
            }
            CandidateControllerOwner::Uncertain | CandidateControllerOwner::Transitioning => {
                self.controller = CandidateControllerOwner::Uncertain;
                (WheelsOffQualificationFailClosedStop::Uncertain, None)
            }
        }
    }

    fn install_cadence_stop(
        &mut self,
        stop: CandidateCadenceOverflowStop,
    ) -> (
        WheelsOffQualificationFailClosedStop,
        Option<Box<LiveActuationError>>,
    ) {
        match stop {
            CandidateCadenceOverflowStop::Confirmed {
                controller,
                receipt,
            } if exact_safe_disarm(&receipt) => {
                self.controller = CandidateControllerOwner::Stopped(controller);
                self.stopped_applied = None;
                self.last_stop = Some(receipt);
                (WheelsOffQualificationFailClosedStop::NewExactReceipt, None)
            }
            CandidateCadenceOverflowStop::Confirmed { .. } => {
                self.controller = CandidateControllerOwner::Uncertain;
                (WheelsOffQualificationFailClosedStop::Uncertain, None)
            }
            CandidateCadenceOverflowStop::Uncertain(source) => {
                let source = *source;
                let stop = stop_knowledge_from_live_error(&source);
                self.controller = owner_after_unretained_stop(stop);
                (stop, Some(Box::new(source)))
            }
        }
    }

    fn observed_exact_stop(&self) -> Option<QualificationObservedAppliedZero> {
        let receipt = self.last_stop.as_ref()?;
        if !matches!(self.controller, CandidateControllerOwner::Stopped(_))
            || !exact_safe_disarm(receipt)
        {
            return None;
        }
        QualificationObservedAppliedZero::parse(u64::from(receipt.request_id().get()), 0, 0).ok()
    }
}

impl Drop for WheelsOffQualificationRuntime {
    fn drop(&mut self) {
        if self.state == WheelsOffQualificationRuntimeState::Shutdown {
            return;
        }
        self.console.signal_internal_fail_closed(None);
        self.target = None;
        let (stop, _) = self.attempt_stop();
        if matches!(
            stop,
            WheelsOffQualificationFailClosedStop::ReusedExactReceipt
                | WheelsOffQualificationFailClosedStop::NewExactReceipt
        ) && let Some(observed) = self.observed_exact_stop()
        {
            let _ = self.receiver.disconnect_after_confirmed_zero(observed);
        }
    }
}

fn exact_safe_applied_zero(receipt: &AppliedCommandReceipt) -> bool {
    receipt.verified_host_result().requested_timer_pwm.is_zero() && receipt.is_confirmed_zero()
}

fn admit_console_before_runtime(
    snapshot: &WheelsOffQualificationSnapshot,
) -> Result<(), WheelsOffQualificationRuntimeStartError> {
    match snapshot.frontend_state {
        WheelsOffQualificationFrontendState::AwaitingConnection => {
            return Err(WheelsOffQualificationRuntimeStartError::ConsoleFrontendNotConnected);
        }
        WheelsOffQualificationFrontendState::Connected => {}
        WheelsOffQualificationFrontendState::Disconnected => {
            return Err(WheelsOffQualificationRuntimeStartError::ConsoleFrontendDisconnected);
        }
    }
    if snapshot.motion_authority_enabled {
        return Err(WheelsOffQualificationRuntimeStartError::ConsoleMotionAuthorityAlreadyEnabled);
    }
    Ok(())
}

fn exact_safe_disarm(receipt: &DisarmReceipt) -> bool {
    receipt.output_state().is_safe() && receipt.controller_faults().is_clear()
}

fn owner_after_unretained_stop(
    stop: WheelsOffQualificationFailClosedStop,
) -> CandidateControllerOwner {
    match stop {
        WheelsOffQualificationFailClosedStop::ControllerConfirmedWithoutRetainedReceipt => {
            CandidateControllerOwner::ConfirmedWithoutRetainedReceipt
        }
        _ => CandidateControllerOwner::Uncertain,
    }
}

fn stop_knowledge_from_live_error(
    error: &LiveActuationError,
) -> WheelsOffQualificationFailClosedStop {
    let confirmed = match error {
        LiveActuationError::Acquire(failure) => {
            failure.stop_knowledge() == LatchedStopKnowledge::ConfirmedStop
        }
        LiveActuationError::Preflight(failure) | LiveActuationError::Apply(failure) => {
            failure.stop_knowledge() == LatchedStopKnowledge::ConfirmedStop
        }
        LiveActuationError::DecisionRejected { stop, .. } => match stop {
            LocalRejectionStop::Confirmed(_) => true,
            LocalRejectionStop::DisarmFailed(failure) => {
                failure.stop_knowledge() == LatchedStopKnowledge::ConfirmedStop
            }
            LocalRejectionStop::SessionAlreadyConsumed => false,
        },
        LiveActuationError::Disarm(failure) => {
            failure.stop_knowledge() == LatchedStopKnowledge::ConfirmedStop
        }
        LiveActuationError::TransportBuild(_) | LiveActuationError::SessionConsumed => false,
    };
    if confirmed {
        WheelsOffQualificationFailClosedStop::ControllerConfirmedWithoutRetainedReceipt
    } else {
        WheelsOffQualificationFailClosedStop::Uncertain
    }
}

fn failure_live_stop_knowledge(
    failure: &WheelsOffQualificationRuntimeFailure,
) -> Option<WheelsOffQualificationFailClosedStop> {
    match failure {
        WheelsOffQualificationRuntimeFailure::Reacquire(source)
        | WheelsOffQualificationRuntimeFailure::Apply(source) => {
            Some(stop_knowledge_from_live_error(source))
        }
        _ => None,
    }
}

fn refresh_due(now: Instant, next_refresh_not_before: Option<Instant>) -> bool {
    next_refresh_not_before.is_none_or(|deadline| now >= deadline)
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn cadence_scheduler_is_idle_before_deadline() {
        let now = Instant::now();
        let deadline = now + Duration::from_millis(20);
        assert!(!refresh_due(now, Some(deadline)));
        assert!(refresh_due(deadline, Some(deadline)));
        assert!(refresh_due(now, None));
    }

    #[test]
    fn pending_motion_authority_carries_no_attestation() {
        let pending = WheelsOffQualificationMotionAuthority::PendingAttestation;
        assert_eq!(
            pending.state(),
            WheelsOffQualificationMotionAuthorityState::PendingAttestation
        );
        assert!(pending.attestation().is_none());

        let attestation = OperatorClaimedWheelsOffAttestation::try_new(
            true,
            true,
            true,
            true,
            true,
            Instant::now(),
        )
        .expect("explicit test attestation");
        let enabled = WheelsOffQualificationMotionAuthority::Enabled(attestation);
        assert_eq!(
            enabled.state(),
            WheelsOffQualificationMotionAuthorityState::Enabled
        );
        assert_eq!(enabled.attestation(), Some(attestation));
    }

    #[test]
    fn controller_stop_knowledge_never_promotes_uncertain_evidence() {
        assert_eq!(
            owner_after_unretained_stop(WheelsOffQualificationFailClosedStop::Uncertain)
                .state_for_test(),
            WheelsOffQualificationControllerState::StopUncertain
        );
    }

    #[test]
    fn runtime_start_admission_rejects_frontend_loss_before_motion_enablement() {
        let profile = super::super::WheelsOffQualificationControlProfile::parse(30, 10, 250)
            .expect("test profile");
        let (console, _receiver) = super::super::wheels_off_qualification_console(profile);
        console
            .report_frontend_connection_lost(HostMonotonicTimestamp::from_nanos(1))
            .expect("no authority means no stop event is needed");

        assert_eq!(
            admit_console_before_runtime(&console.snapshot()),
            Err(WheelsOffQualificationRuntimeStartError::ConsoleFrontendDisconnected)
        );
    }

    #[test]
    fn runtime_start_admission_rejects_console_without_frontend_readiness() {
        let profile = super::super::WheelsOffQualificationControlProfile::parse(30, 10, 250)
            .expect("test profile");
        let (console, _receiver) = super::super::wheels_off_qualification_console(profile);

        assert_eq!(
            admit_console_before_runtime(&console.snapshot()),
            Err(WheelsOffQualificationRuntimeStartError::ConsoleFrontendNotConnected)
        );
    }

    impl CandidateControllerOwner {
        fn state_for_test(&self) -> WheelsOffQualificationControllerState {
            match self {
                Self::Stopped(_) => WheelsOffQualificationControllerState::StoppedWithExactReceipt,
                Self::Active(_) => WheelsOffQualificationControllerState::Active,
                Self::ConfirmedWithoutRetainedReceipt => {
                    WheelsOffQualificationControllerState::StopConfirmedWithoutRetainedReceipt
                }
                Self::Uncertain | Self::Transitioning => {
                    WheelsOffQualificationControllerState::StopUncertain
                }
            }
        }
    }
}
