//! Linear adapter between the authenticated operator console and the sole
//! production motion owner.
//!
//! The console parses weak HTTP input once. This adapter accepts only the
//! resulting domain commands and submits them through the one non-clone typed
//! ingress created with the control socket. Process-local typed keys, claim
//! rendezvous, response channels, and console completion tokens never escape
//! this module.

use std::collections::HashMap;
use std::fmt;

use crossbeam_channel::TryRecvError;
use robot_command_client::AppliedCommandReceipt;

use super::control_socket::{
    AgentControlTypedIngress, AgentControlTypedRequestKey, AgentControlTypedSubmission,
    AgentControlTypedSubmissionPollError, AgentControlTypedSubmitError,
};
use super::live_motion_owner::{
    LiveLifecycleZeroReason, LiveMotionActuationPort, LiveMotionCompletedSafetyAction,
    LiveMotionOwner, LiveMotionOwnerError, LiveMotionOwnerOutcome, LiveMotionOwnerTerminalReport,
    LiveMotionTerminalShutdownEvidence, LivePhysicalStateEvent, LiveSoftwareEmergencyStopApplied,
};
use super::mpc::HostMonotonicClock;
use super::{
    AgentAutonomousMode, AgentControlCommandKindV1, AgentControlCompletionV1,
    AgentControlResponseKindV1, AgentControlResponseV1, ConsoleDownstreamRequestId,
    ConsoleResponseBindError, ConsoleResponseCompletionError, ConsoleResponseRejectionCode,
    ConsoleStopCause, ConsoleVerifiedCompletionError, NavigationIngressSink,
    OperatorConsoleAuthorityGeneration, OperatorConsoleCommand, OperatorConsoleCompletion,
    OperatorConsoleHandle, OperatorConsoleIngressItem, OperatorConsoleIngressReceiver,
    OperatorConsoleResponseToken,
};

/// Result of admitting at most one console item before one owner period.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperatorConsoleIngressDisposition {
    Idle,
    Submitted {
        downstream_request_id: ConsoleDownstreamRequestId,
        command: AgentControlCommandKindV1,
    },
    RejectedBackpressure {
        downstream_request_id: ConsoleDownstreamRequestId,
        command: AgentControlCommandKindV1,
    },
    /// The supervisor emergency-stop latch and exact fresh zero have already
    /// been applied. The caller must borrow the owner's retained physical
    /// event for diagnostics, then call
    /// [`OperatorConsoleRuntimeAdapter::complete_software_emergency_stop`].
    SoftwareEmergencyStopApplied,
}

/// Observable adapter transition after one owner result has been reconciled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperatorConsoleProcessDisposition {
    UnrelatedRuntimeOutcome,
    ResponseCompleted {
        downstream_request_id: ConsoleDownstreamRequestId,
    },
    ResponseRejected {
        downstream_request_id: ConsoleDownstreamRequestId,
    },
    AuthorityActivated {
        downstream_request_id: ConsoleDownstreamRequestId,
    },
    AuthorityCompleted {
        downstream_request_id: ConsoleDownstreamRequestId,
    },
    AuthorityCancelled {
        downstream_request_id: ConsoleDownstreamRequestId,
    },
    SaveMapPersistenceRequired {
        downstream_request_id: ConsoleDownstreamRequestId,
    },
}

/// A precise adapter invariant or channel failure.
#[derive(Debug)]
pub enum OperatorConsoleRuntimeAdapterError {
    EmergencyStopCompletionPending,
    NoEmergencyStopCompletionPending,
    SaveMapCompletionPending,
    NoSaveMapCompletionPending,
    CorrelationExhausted,
    RuntimeIngressDisconnected,
    ResponseTokenAlreadyBound,
    CorrelationCollision,
    MissingPendingCorrelation,
    UnexpectedTypedCorrelation,
    ClaimNotObserved,
    CancellationAfterClaim,
    ClaimChannelDisconnected,
    ResponseNotObserved,
    ResponseChannelDisconnected,
    RequestIdentityMismatch,
    ResponseIdentityMismatch,
    ResponseCommandMismatch,
    ResponseOutcomeMismatch,
    OwnerOutcomeMismatch,
    PhysicalEvidenceRequired,
    UnexpectedPhysicalEvidence,
    PhysicalEvidenceCorrelationMismatch,
    PhysicalEvidenceKindMismatch,
    PhysicalEvidenceReasonMismatch,
    UnsafeLifecycleZero,
    VerifiedCompletion(ConsoleVerifiedCompletionError),
    Completion(ConsoleResponseCompletionError),
    AuthorityConflict,
    AuthorityMissing,
    AuthorityModeMismatch,
    AuthoritySessionMissing,
    InternalEmergencyStopNotLatched,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperatorConsoleShutdownAuthorityState {
    None,
    CancelledAfterConfirmedControllerStop {
        downstream_request_id: ConsoleDownstreamRequestId,
    },
    RejectedBecauseControllerStopUncertain {
        downstream_request_id: ConsoleDownstreamRequestId,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OperatorConsoleRuntimeShutdownOutcome {
    pub controller_stop_confirmed: bool,
    pub lifecycle_cleanup_failed: bool,
    pub rejected_requests: usize,
    pub emergency_completion_finished: bool,
    pub authority: OperatorConsoleShutdownAuthorityState,
}

impl fmt::Display for OperatorConsoleRuntimeAdapterError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "operator-console runtime adapter failed: {self:?}"
        )
    }
}

impl std::error::Error for OperatorConsoleRuntimeAdapterError {}

/// A direct emergency stop can fail in either the adapter or the sole live
/// owner. The latter retains the exact typed owner error rather than flattening
/// it into an ambiguous string or boolean.
#[derive(Debug)]
pub enum OperatorConsoleRuntimeIngressError<E, J> {
    Adapter(OperatorConsoleRuntimeAdapterError),
    EmergencyStop(LiveMotionOwnerError<E, J>),
}

impl<E: fmt::Debug, J: fmt::Debug> fmt::Display for OperatorConsoleRuntimeIngressError<E, J> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Adapter(source) => source.fmt(formatter),
            Self::EmergencyStop(source) => source.fmt(formatter),
        }
    }
}

impl<E, J> std::error::Error for OperatorConsoleRuntimeIngressError<E, J>
where
    E: std::error::Error + 'static,
    J: std::error::Error + 'static,
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Adapter(source) => Some(source),
            Self::EmergencyStop(source) => Some(source),
        }
    }
}

struct PendingConsoleSubmission {
    downstream_request_id: ConsoleDownstreamRequestId,
    command: OperatorConsoleCommand,
    command_kind: AgentControlCommandKindV1,
    submission: AgentControlTypedSubmission,
    response: OperatorConsoleResponseToken,
}

struct PendingEmergencyStop {
    key: AgentControlTypedRequestKey,
    downstream_request_id: Option<ConsoleDownstreamRequestId>,
    response: Option<OperatorConsoleResponseToken>,
    evidence: LiveSoftwareEmergencyStopApplied,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperatorConsoleRetainedAuthorityKind {
    Manual,
    Autonomous(AgentAutonomousMode),
}

struct RetainedAuthority {
    kind: OperatorConsoleRetainedAuthorityKind,
    guard: OperatorConsoleAuthorityGeneration,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OperatorConsoleRetainedAuthority {
    kind: OperatorConsoleRetainedAuthorityKind,
    downstream_request_id: ConsoleDownstreamRequestId,
    source: super::ConsoleSourceKind,
}

impl OperatorConsoleRetainedAuthority {
    pub const fn kind(self) -> OperatorConsoleRetainedAuthorityKind {
        self.kind
    }

    pub const fn downstream_request_id(self) -> ConsoleDownstreamRequestId {
        self.downstream_request_id
    }

    pub const fn source(self) -> super::ConsoleSourceKind {
        self.source
    }
}

/// Sole production adapter for console arbitration and exact response
/// completion.
///
/// This value is intentionally not `Clone`. It owns the only typed console
/// ingress, receiver, pending-correlation table, and active console authority
/// guard. Production drains at most one console item before each owner period.
pub struct OperatorConsoleRuntimeAdapter {
    handle: OperatorConsoleHandle,
    receiver: OperatorConsoleIngressReceiver,
    ingress: AgentControlTypedIngress,
    pending: HashMap<AgentControlTypedRequestKey, PendingConsoleSubmission>,
    awaiting_save_map: Option<PendingConsoleSubmission>,
    pending_emergency_stop: Option<PendingEmergencyStop>,
    authority: Option<RetainedAuthority>,
}

impl OperatorConsoleRuntimeAdapter {
    pub fn new(
        handle: OperatorConsoleHandle,
        receiver: OperatorConsoleIngressReceiver,
        ingress: AgentControlTypedIngress,
    ) -> Self {
        Self {
            handle,
            receiver,
            ingress,
            pending: HashMap::new(),
            awaiting_save_map: None,
            pending_emergency_stop: None,
            authority: None,
        }
    }

    pub const fn handle(&self) -> &OperatorConsoleHandle {
        &self.handle
    }

    pub fn pending_submission_count(&self) -> usize {
        self.pending.len() + usize::from(self.awaiting_save_map.is_some())
    }

    pub const fn has_active_console_authority(&self) -> bool {
        self.authority.is_some()
    }

    /// Exact console authority guard currently retained by this adapter.
    pub fn retained_authority(
        &self,
    ) -> Result<Option<OperatorConsoleRetainedAuthority>, OperatorConsoleRuntimeAdapterError> {
        let Some(authority) = self.authority.as_ref() else {
            return Ok(None);
        };
        let source = match self.handle.source_kind(authority.guard.session_id()) {
            Ok(source) => source,
            Err(_) => {
                return self.fail(OperatorConsoleRuntimeAdapterError::AuthoritySessionMissing);
            }
        };
        Ok(Some(OperatorConsoleRetainedAuthority {
            kind: authority.kind,
            downstream_request_id: authority.guard.generation(),
            source,
        }))
    }

    /// Resolve an owner's process-local typed correlation while the matching
    /// linear submission is still pending. Periodic and raw-socket events have
    /// no console request identity and return `None`.
    pub fn correlated_downstream_request_id<D>(
        &self,
        event: &LivePhysicalStateEvent<AppliedCommandReceipt, D>,
    ) -> Result<Option<ConsoleDownstreamRequestId>, OperatorConsoleRuntimeAdapterError> {
        let Some(key) = event.typed_request_key() else {
            return Ok(None);
        };
        if let Some(pending) = self.pending.get(&key) {
            return Ok(Some(pending.downstream_request_id));
        }
        if let Some(pending) = self.pending_emergency_stop.as_ref()
            && pending.key == key
        {
            return Ok(pending.downstream_request_id);
        }
        self.fail(OperatorConsoleRuntimeAdapterError::MissingPendingCorrelation)
    }

    /// Drain and admit no more than one console item before the next owner
    /// process/tick operation.
    pub fn drain_one_before_owner<J, P, C>(
        &mut self,
        owner: &mut LiveMotionOwner<J, P, C>,
    ) -> Result<
        OperatorConsoleIngressDisposition,
        OperatorConsoleRuntimeIngressError<P::Error, J::Error>,
    >
    where
        J: NavigationIngressSink,
        C: HostMonotonicClock,
        P: LiveMotionActuationPort<J, C, Receipt = AppliedCommandReceipt>,
    {
        if self.pending_emergency_stop.is_some() {
            return self
                .fail_ingress(OperatorConsoleRuntimeAdapterError::EmergencyStopCompletionPending);
        }
        if self.awaiting_save_map.is_some() && !self.handle.software_safety_stop_latched() {
            return self.fail_ingress(OperatorConsoleRuntimeAdapterError::SaveMapCompletionPending);
        }
        let item = match self.receiver.try_next() {
            Ok(item) => item,
            Err(TryRecvError::Empty) => return Ok(OperatorConsoleIngressDisposition::Idle),
            Err(TryRecvError::Disconnected) => {
                return self
                    .fail_ingress(OperatorConsoleRuntimeAdapterError::RuntimeIngressDisconnected);
            }
        };
        match item {
            OperatorConsoleIngressItem::Dispatch(dispatch) => {
                let downstream_request_id = dispatch.downstream_request_id();
                let command = dispatch.command();
                if let Some(reason) = priority_cancellation_reason(command) {
                    self.cancel_older_unclaimed(downstream_request_id, reason)
                        .map_err(OperatorConsoleRuntimeIngressError::Adapter)?;
                }
                let (request, _source, received_at, response) = dispatch.into_parts();
                let command_kind = request.command().kind();
                let submission = match self.ingress.try_submit(request, received_at) {
                    Ok(submission) => submission,
                    Err(AgentControlTypedSubmitError::QueueFull { submission }) => {
                        response.reject(ConsoleResponseRejectionCode::RuntimeRejected);
                        drop(submission);
                        if command_is_safety_reducing(command) {
                            self.handle.signal_internal_fail_closed();
                        }
                        return Ok(OperatorConsoleIngressDisposition::RejectedBackpressure {
                            downstream_request_id,
                            command: command_kind,
                        });
                    }
                    Err(AgentControlTypedSubmitError::RuntimeDisconnected { submission }) => {
                        response.reject(ConsoleResponseRejectionCode::InternalFault);
                        drop(submission);
                        return self.fail_ingress(
                            OperatorConsoleRuntimeAdapterError::RuntimeIngressDisconnected,
                        );
                    }
                    Err(AgentControlTypedSubmitError::CorrelationExhausted) => {
                        response.reject(ConsoleResponseRejectionCode::InternalFault);
                        return self.fail_ingress(
                            OperatorConsoleRuntimeAdapterError::CorrelationExhausted,
                        );
                    }
                };
                let key = submission.typed_request_key();
                let response = match response.bind_typed_request_key(key) {
                    Ok(response) => response,
                    Err(ConsoleResponseBindError::AlreadyBound) => {
                        drop(submission);
                        return self.fail_ingress(
                            OperatorConsoleRuntimeAdapterError::ResponseTokenAlreadyBound,
                        );
                    }
                };
                let pending = PendingConsoleSubmission {
                    downstream_request_id,
                    command,
                    command_kind,
                    submission,
                    response,
                };
                if self.pending.contains_key(&key) {
                    return self.fail_pending_ingress(
                        pending,
                        OperatorConsoleRuntimeAdapterError::CorrelationCollision,
                    );
                }
                self.pending.insert(key, pending);
                Ok(OperatorConsoleIngressDisposition::Submitted {
                    downstream_request_id,
                    command: command_kind,
                })
            }
            OperatorConsoleIngressItem::SoftwareSafetyStop(stop) => {
                self.apply_direct_software_emergency_stop(owner, stop)
            }
        }
    }

    /// Reconcile one `process_one` result. `physical_event` is consumed here
    /// only after the caller has borrowed it for Rerun/diagnostics.
    pub fn complete_processed_owner_outcome<J, P, C>(
        &mut self,
        owner: &mut LiveMotionOwner<J, P, C>,
        outcome: &LiveMotionOwnerOutcome,
        physical_event: Option<LivePhysicalStateEvent<AppliedCommandReceipt, P::Diagnostic>>,
    ) -> Result<OperatorConsoleProcessDisposition, OperatorConsoleRuntimeAdapterError>
    where
        J: NavigationIngressSink,
        C: HostMonotonicClock,
        P: LiveMotionActuationPort<J, C, Receipt = AppliedCommandReceipt>,
    {
        self.ensure_no_deferred_completion()?;
        match owner.take_last_processed_typed_request_key() {
            Some(key) => self.complete_correlated(key, outcome, physical_event),
            None => self.complete_uncorrelated_process(outcome, physical_event),
        }
    }

    /// Finish a `SaveMap` token only after the outer persistence owner has
    /// sent its truthful accepted/rejected response through the claimed
    /// request returned by `LiveMotionOwnerOutcome::SaveMapRequested`.
    pub fn complete_save_map_response(
        &mut self,
    ) -> Result<OperatorConsoleProcessDisposition, OperatorConsoleRuntimeAdapterError> {
        if self.pending_emergency_stop.is_some() {
            return self.fail(OperatorConsoleRuntimeAdapterError::EmergencyStopCompletionPending);
        }
        let Some(pending) = self.awaiting_save_map.take() else {
            return self.fail(OperatorConsoleRuntimeAdapterError::NoSaveMapCompletionPending);
        };
        let response = match pending.submission.try_take_response() {
            Ok(response) => response,
            Err(AgentControlTypedSubmissionPollError::Pending) => {
                return self.fail_pending(
                    pending,
                    OperatorConsoleRuntimeAdapterError::ResponseNotObserved,
                );
            }
            Err(AgentControlTypedSubmissionPollError::ResponseDisconnected) => {
                return self.fail_pending(
                    pending,
                    OperatorConsoleRuntimeAdapterError::ResponseChannelDisconnected,
                );
            }
            Err(AgentControlTypedSubmissionPollError::ClaimDisconnected) => {
                return self.fail_pending(
                    pending,
                    OperatorConsoleRuntimeAdapterError::ClaimChannelDisconnected,
                );
            }
        };
        if let Err(source) = self.validate_response_identity(&pending, response) {
            return self.fail_pending(pending, source);
        }
        match response.response() {
            AgentControlResponseKindV1::Accepted {
                command: AgentControlCommandKindV1::SaveMap,
                completion: AgentControlCompletionV1::Completed,
            } => {
                let id = pending.downstream_request_id;
                self.finish_nonphysical_completion(pending)?;
                Ok(OperatorConsoleProcessDisposition::ResponseCompleted {
                    downstream_request_id: id,
                })
            }
            AgentControlResponseKindV1::Rejected { .. } => {
                let id = pending.downstream_request_id;
                pending
                    .response
                    .reject(ConsoleResponseRejectionCode::RuntimeRejected);
                Ok(OperatorConsoleProcessDisposition::ResponseRejected {
                    downstream_request_id: id,
                })
            }
            AgentControlResponseKindV1::Accepted { .. }
            | AgentControlResponseKindV1::Status { .. } => self.fail_pending(
                pending,
                OperatorConsoleRuntimeAdapterError::ResponseOutcomeMismatch,
            ),
        }
    }

    /// Consume the direct emergency-stop proof after the caller has emitted
    /// the owner's retained lifecycle-zero event.
    pub fn complete_software_emergency_stop(
        &mut self,
    ) -> Result<(), OperatorConsoleRuntimeAdapterError> {
        let Some(pending) = self.pending_emergency_stop.take() else {
            return self.fail(OperatorConsoleRuntimeAdapterError::NoEmergencyStopCompletionPending);
        };
        let completion = match pending.response {
            Some(response) => response
                .completed_with_verified_emergency_stop(pending.evidence)
                .map_err(OperatorConsoleRuntimeAdapterError::VerifiedCompletion),
            None => self
                .handle
                .complete_internal_fail_closed_with_verified_emergency_stop(
                    pending.key,
                    pending.evidence,
                )
                .map_err(OperatorConsoleRuntimeAdapterError::VerifiedCompletion)
                .and_then(|completed| {
                    completed
                        .then_some(())
                        .ok_or(OperatorConsoleRuntimeAdapterError::InternalEmergencyStopNotLatched)
                }),
        };
        match completion {
            Ok(()) => Ok(()),
            Err(source) => self.fail(source),
        }
    }

    /// Reconcile a later `tick_motion` result. Such a tick has no console
    /// request key; it can only maintain or terminate the one retained
    /// authority generation.
    pub fn complete_periodic_owner_outcome<D>(
        &mut self,
        outcome: &LiveMotionOwnerOutcome,
        physical_event: Option<LivePhysicalStateEvent<AppliedCommandReceipt, D>>,
    ) -> Result<OperatorConsoleProcessDisposition, OperatorConsoleRuntimeAdapterError> {
        self.ensure_no_deferred_completion()?;
        self.validate_uncorrelated_event(&physical_event)?;
        match outcome {
            LiveMotionOwnerOutcome::Idle => {
                if physical_event.is_some() {
                    return self
                        .fail(OperatorConsoleRuntimeAdapterError::UnexpectedPhysicalEvidence);
                }
                Ok(OperatorConsoleProcessDisposition::UnrelatedRuntimeOutcome)
            }
            LiveMotionOwnerOutcome::PeriodicManualApplied => {
                self.require_authority_kind(OperatorConsoleRetainedAuthorityKind::Manual)?;
                self.require_coordinator_event(physical_event)?;
                Ok(OperatorConsoleProcessDisposition::UnrelatedRuntimeOutcome)
            }
            LiveMotionOwnerOutcome::PeriodicManualStopped => {
                self.require_authority_kind(OperatorConsoleRetainedAuthorityKind::Manual)?;
                self.require_safe_termination_event(physical_event)?;
                self.cancel_authority()
            }
            LiveMotionOwnerOutcome::PeriodicAutonomousApplied { mode } => {
                self.require_authority_kind(OperatorConsoleRetainedAuthorityKind::Autonomous(
                    *mode,
                ))?;
                self.require_coordinator_event(physical_event)?;
                Ok(OperatorConsoleProcessDisposition::UnrelatedRuntimeOutcome)
            }
            LiveMotionOwnerOutcome::PeriodicAutonomousStopped { mode } => {
                self.require_authority_kind(OperatorConsoleRetainedAuthorityKind::Autonomous(
                    *mode,
                ))?;
                self.require_safe_termination_event(physical_event)?;
                self.cancel_authority()
            }
            LiveMotionOwnerOutcome::AutonomousCompleted { mode } => {
                self.require_authority_kind(OperatorConsoleRetainedAuthorityKind::Autonomous(
                    *mode,
                ))?;
                self.require_safe_termination_event(physical_event)?;
                self.complete_authority()
            }
            LiveMotionOwnerOutcome::ClientUnavailableBeforeClaim
            | LiveMotionOwnerOutcome::StatusReplied(_)
            | LiveMotionOwnerOutcome::Rejected { .. }
            | LiveMotionOwnerOutcome::Completed(_)
            | LiveMotionOwnerOutcome::SaveMapRequested { .. }
            | LiveMotionOwnerOutcome::ShutdownRequested
            | LiveMotionOwnerOutcome::AutonomousAccepted { .. } => {
                self.fail(OperatorConsoleRuntimeAdapterError::OwnerOutcomeMismatch)
            }
        }
    }

    /// Explicitly terminalize the exact console request whose owner operation
    /// returned an error before a normal outcome could be reconciled.
    pub fn fail_processed_owner_operation<J, P, C>(
        &mut self,
        owner: &mut LiveMotionOwner<J, P, C>,
    ) -> Result<(), OperatorConsoleRuntimeAdapterError>
    where
        J: NavigationIngressSink,
        C: HostMonotonicClock,
        P: LiveMotionActuationPort<J, C, Receipt = AppliedCommandReceipt>,
    {
        let Some(key) = owner.take_last_processed_typed_request_key() else {
            self.handle.signal_internal_fail_closed();
            return Ok(());
        };
        let Some(pending) = self.pending.remove(&key) else {
            return self.fail(OperatorConsoleRuntimeAdapterError::MissingPendingCorrelation);
        };
        pending
            .response
            .reject(ConsoleResponseRejectionCode::InternalFault);
        drop(pending.submission);
        self.handle.signal_internal_fail_closed();
        Ok(())
    }

    /// Consume the adapter after the HTTP server has stopped accepting work
    /// and after the sole motion owner has produced its terminal report.
    ///
    /// Only confirmed controller-stop knowledge allows an active authority
    /// generation to transition to `Cancelled`. This includes recovery which
    /// proved stop after a failed disarm operation; the operation error remains
    /// in the terminal report. An uncertain terminal stop rejects the
    /// generation through its linear guard and latches the console's internal
    /// fault instead of displaying a false safe cancellation.
    pub fn shutdown<J, StopReceipt, StopError, D>(
        self,
        terminal: &LiveMotionOwnerTerminalReport<
            J,
            StopReceipt,
            StopError,
            AppliedCommandReceipt,
            D,
        >,
    ) -> OperatorConsoleRuntimeShutdownOutcome
    where
        J: NavigationIngressSink,
    {
        self.shutdown_with_terminal_evidence(terminal.shutdown_evidence())
    }

    /// Consume the adapter from the copyable non-physical facts retained
    /// before a terminal report releases its coordinator for journal
    /// finalization. This is used by a terminal map checkpoint, whose client
    /// response must remain pending until the exact dataset manifest has been
    /// published.
    pub fn shutdown_with_terminal_evidence(
        mut self,
        terminal: LiveMotionTerminalShutdownEvidence,
    ) -> OperatorConsoleRuntimeShutdownOutcome {
        let controller_stop_confirmed = terminal.controller_stop_confirmed();
        let lifecycle_cleanup_failed = terminal.lifecycle_cleanup_failed();
        let rejection = if controller_stop_confirmed {
            ConsoleResponseRejectionCode::CancelledByPriorityStop
        } else {
            ConsoleResponseRejectionCode::InternalFault
        };
        let mut rejected_requests = 0_usize;

        for (_, pending) in self.pending.drain() {
            pending.response.reject(rejection);
            drop(pending.submission);
            rejected_requests = rejected_requests.saturating_add(1);
        }
        if let Some(pending) = self.awaiting_save_map.take() {
            pending.response.reject(rejection);
            drop(pending.submission);
            rejected_requests = rejected_requests.saturating_add(1);
        }
        loop {
            match self.receiver.try_next() {
                Ok(OperatorConsoleIngressItem::Dispatch(dispatch)) => {
                    let (_request, _source, _received_at, response) = dispatch.into_parts();
                    response.reject(rejection);
                    rejected_requests = rejected_requests.saturating_add(1);
                }
                Ok(OperatorConsoleIngressItem::SoftwareSafetyStop(stop)) => {
                    let (_id, _source, _received_at, _latch, response) =
                        stop.into_emergency_parts();
                    if let Some(response) = response {
                        response.reject(rejection);
                        rejected_requests = rejected_requests.saturating_add(1);
                    }
                }
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
            }
        }

        let emergency_completion_finished =
            if let Some(pending) = self.pending_emergency_stop.take() {
                let completed = match pending.response {
                    Some(response) => response
                        .completed_with_verified_emergency_stop(pending.evidence)
                        .is_ok(),
                    None => self
                        .handle
                        .complete_internal_fail_closed_with_verified_emergency_stop(
                            pending.key,
                            pending.evidence,
                        )
                        .is_ok_and(|completed| completed),
                };
                if !completed {
                    self.handle.signal_internal_fail_closed();
                }
                completed
            } else {
                false
            };

        let authority = match (controller_stop_confirmed, self.authority.take()) {
            (_, None) => OperatorConsoleShutdownAuthorityState::None,
            (true, Some(authority)) => {
                let downstream_request_id = authority.guard.generation();
                authority.guard.cancelled();
                OperatorConsoleShutdownAuthorityState::CancelledAfterConfirmedControllerStop {
                    downstream_request_id,
                }
            }
            (false, Some(authority)) => {
                let downstream_request_id = authority.guard.generation();
                self.handle.signal_internal_fail_closed();
                // Deliberately do not call `cancelled`: dropping the live guard
                // records an internal-fault rejection, not a false safe stop.
                drop(authority);
                OperatorConsoleShutdownAuthorityState::RejectedBecauseControllerStopUncertain {
                    downstream_request_id,
                }
            }
        };
        if !controller_stop_confirmed || lifecycle_cleanup_failed {
            self.handle.signal_internal_fail_closed();
        }

        OperatorConsoleRuntimeShutdownOutcome {
            controller_stop_confirmed,
            lifecycle_cleanup_failed,
            rejected_requests,
            emergency_completion_finished,
            authority,
        }
    }

    fn apply_direct_software_emergency_stop<J, P, C>(
        &mut self,
        owner: &mut LiveMotionOwner<J, P, C>,
        stop: super::OperatorConsoleSoftwareSafetyStop,
    ) -> Result<
        OperatorConsoleIngressDisposition,
        OperatorConsoleRuntimeIngressError<P::Error, J::Error>,
    >
    where
        J: NavigationIngressSink,
        C: HostMonotonicClock,
        P: LiveMotionActuationPort<J, C, Receipt = AppliedCommandReceipt>,
    {
        let (downstream_request_id, _source, _received_at, _required_latch, response) =
            stop.into_emergency_parts();
        let key = match self.ingress.reserve_direct_safety_key() {
            Ok(key) => key,
            Err(AgentControlTypedSubmitError::CorrelationExhausted) => {
                if let Some(response) = response {
                    response.reject(ConsoleResponseRejectionCode::InternalFault);
                }
                return self.fail_ingress(OperatorConsoleRuntimeAdapterError::CorrelationExhausted);
            }
            Err(AgentControlTypedSubmitError::QueueFull { submission }) => {
                drop(submission);
                unreachable!("direct key reservation cannot observe queue state")
            }
            Err(AgentControlTypedSubmitError::RuntimeDisconnected { submission }) => {
                drop(submission);
                unreachable!("direct key reservation cannot observe queue state")
            }
        };
        let response = match response {
            Some(response) => match response.bind_typed_request_key(key) {
                Ok(response) => Some(response),
                Err(ConsoleResponseBindError::AlreadyBound) => {
                    return self.fail_ingress(
                        OperatorConsoleRuntimeAdapterError::ResponseTokenAlreadyBound,
                    );
                }
            },
            None => None,
        };

        self.cancel_all_pending(ConsoleResponseRejectionCode::CancelledBySoftwareSafetyStop);
        let authority = self.authority.take();

        let evidence = match owner.apply_software_emergency_stop(key) {
            Ok(evidence) => {
                if let Some(authority) = authority {
                    authority.guard.cancelled();
                }
                evidence
            }
            Err(source) => {
                if let Some(response) = response {
                    response.reject(ConsoleResponseRejectionCode::InternalFault);
                }
                // Do not record a safe cancellation without a confirmed zero.
                // Dropping the live guard truthfully rejects the generation
                // and latches the console's internal-fault stop.
                drop(authority);
                self.handle.signal_internal_fail_closed();
                return Err(OperatorConsoleRuntimeIngressError::EmergencyStop(source));
            }
        };
        self.pending_emergency_stop = Some(PendingEmergencyStop {
            key,
            downstream_request_id,
            response,
            evidence,
        });
        Ok(OperatorConsoleIngressDisposition::SoftwareEmergencyStopApplied)
    }

    fn complete_correlated<D>(
        &mut self,
        key: AgentControlTypedRequestKey,
        outcome: &LiveMotionOwnerOutcome,
        physical_event: Option<LivePhysicalStateEvent<AppliedCommandReceipt, D>>,
    ) -> Result<OperatorConsoleProcessDisposition, OperatorConsoleRuntimeAdapterError> {
        let Some(pending) = self.pending.remove(&key) else {
            return self.fail(OperatorConsoleRuntimeAdapterError::MissingPendingCorrelation);
        };
        if pending.submission.typed_request_key() != key {
            return self.fail_pending(
                pending,
                OperatorConsoleRuntimeAdapterError::UnexpectedTypedCorrelation,
            );
        }
        match pending.submission.try_take_claim() {
            Ok(()) => {}
            Err(AgentControlTypedSubmissionPollError::Pending) => {
                return self.fail_pending(
                    pending,
                    OperatorConsoleRuntimeAdapterError::ClaimNotObserved,
                );
            }
            Err(AgentControlTypedSubmissionPollError::ClaimDisconnected) => {
                return self.fail_pending(
                    pending,
                    OperatorConsoleRuntimeAdapterError::ClaimChannelDisconnected,
                );
            }
            Err(AgentControlTypedSubmissionPollError::ResponseDisconnected) => {
                return self.fail_pending(
                    pending,
                    OperatorConsoleRuntimeAdapterError::ResponseChannelDisconnected,
                );
            }
        }
        if matches!(outcome, LiveMotionOwnerOutcome::SaveMapRequested { .. }) {
            if pending.command_kind != AgentControlCommandKindV1::SaveMap {
                return self.fail_pending(
                    pending,
                    OperatorConsoleRuntimeAdapterError::OwnerOutcomeMismatch,
                );
            }
            if physical_event.is_some() {
                return self.fail_pending(
                    pending,
                    OperatorConsoleRuntimeAdapterError::UnexpectedPhysicalEvidence,
                );
            }
            let id = pending.downstream_request_id;
            self.awaiting_save_map = Some(pending);
            return Ok(
                OperatorConsoleProcessDisposition::SaveMapPersistenceRequired {
                    downstream_request_id: id,
                },
            );
        }
        let response = match pending.submission.try_take_response() {
            Ok(response) => response,
            Err(AgentControlTypedSubmissionPollError::Pending) => {
                return self.fail_pending(
                    pending,
                    OperatorConsoleRuntimeAdapterError::ResponseNotObserved,
                );
            }
            Err(AgentControlTypedSubmissionPollError::ResponseDisconnected) => {
                return self.fail_pending(
                    pending,
                    OperatorConsoleRuntimeAdapterError::ResponseChannelDisconnected,
                );
            }
            Err(AgentControlTypedSubmissionPollError::ClaimDisconnected) => {
                return self.fail_pending(
                    pending,
                    OperatorConsoleRuntimeAdapterError::ClaimChannelDisconnected,
                );
            }
        };
        if let Err(source) = self.validate_response_identity(&pending, response) {
            return self.fail_pending(pending, source);
        }
        if let Err(source) = self.validate_owner_response(&pending, outcome, response) {
            return self.fail_pending(pending, source);
        }

        if matches!(
            (outcome, response.response(), &physical_event),
            (
                LiveMotionOwnerOutcome::AutonomousCompleted {
                    mode: AgentAutonomousMode::Explore,
                },
                AgentControlResponseKindV1::Accepted {
                    command: AgentControlCommandKindV1::FrontierExplore,
                    completion: AgentControlCompletionV1::Completed,
                },
                None,
            )
        ) {
            if self.authority.is_some() {
                return self.fail_pending(
                    pending,
                    OperatorConsoleRuntimeAdapterError::AuthorityConflict,
                );
            }
            let id = pending.downstream_request_id;
            if let Err(source) = pending
                .response
                .completed_exploration_without_authority(key)
            {
                self.handle.signal_internal_fail_closed();
                return Err(OperatorConsoleRuntimeAdapterError::VerifiedCompletion(
                    source,
                ));
            }
            return Ok(OperatorConsoleProcessDisposition::ResponseCompleted {
                downstream_request_id: id,
            });
        }

        if matches!(
            response.response(),
            AgentControlResponseKindV1::Rejected { .. }
        ) {
            self.validate_event_key_for_rejection(key, pending.command, &physical_event)?;
            let terminates_manual_authority = matches!(
                (pending.command, &physical_event),
                (
                    OperatorConsoleCommand::ManualVelocity { .. },
                    Some(LivePhysicalStateEvent::LifecycleZero(applied))
                ) if applied.reason() == LiveLifecycleZeroReason::ManualRelease
            );
            let id = pending.downstream_request_id;
            pending
                .response
                .reject(ConsoleResponseRejectionCode::RuntimeRejected);
            if terminates_manual_authority {
                self.require_authority_kind(OperatorConsoleRetainedAuthorityKind::Manual)?;
                return self.cancel_authority();
            }
            return Ok(OperatorConsoleProcessDisposition::ResponseRejected {
                downstream_request_id: id,
            });
        }

        let id = pending.downstream_request_id;
        let command = pending.command;
        let completion = match physical_event {
            Some(LivePhysicalStateEvent::CoordinatorTick(evidence)) => pending
                .response
                .completed_with_verified_motion(evidence)
                .map(|(completion, _diagnostic)| completion)
                .map_err(OperatorConsoleRuntimeAdapterError::VerifiedCompletion),
            Some(LivePhysicalStateEvent::LifecycleZero(evidence)) => pending
                .response
                .completed_with_verified_lifecycle_zero(evidence)
                .map_err(OperatorConsoleRuntimeAdapterError::VerifiedCompletion),
            Some(LivePhysicalStateEvent::ActuationFault { .. }) => {
                Err(OperatorConsoleRuntimeAdapterError::PhysicalEvidenceKindMismatch)
            }
            None => Err(OperatorConsoleRuntimeAdapterError::PhysicalEvidenceRequired),
        };
        let completion = match completion {
            Ok(completion) => completion,
            Err(source) => {
                self.handle.signal_internal_fail_closed();
                return Err(source);
            }
        };
        match completion {
            OperatorConsoleCompletion::Completed => {
                if command_is_safety_reducing(command) {
                    let authority = self.cancel_authority_if_present();
                    if authority.is_some() {
                        return Ok(OperatorConsoleProcessDisposition::AuthorityCancelled {
                            downstream_request_id: id,
                        });
                    }
                }
                Ok(OperatorConsoleProcessDisposition::ResponseCompleted {
                    downstream_request_id: id,
                })
            }
            OperatorConsoleCompletion::Authority(guard) => {
                let Some(kind) = authority_kind_for_command(command) else {
                    guard.cancelled();
                    self.handle.signal_internal_fail_closed();
                    return Err(OperatorConsoleRuntimeAdapterError::OwnerOutcomeMismatch);
                };
                if let Some(existing) = self.authority.take() {
                    existing.guard.cancelled();
                    guard.cancelled();
                    return self.fail(OperatorConsoleRuntimeAdapterError::AuthorityConflict);
                }
                self.authority = Some(RetainedAuthority { kind, guard });
                Ok(OperatorConsoleProcessDisposition::AuthorityActivated {
                    downstream_request_id: id,
                })
            }
        }
    }

    fn complete_uncorrelated_process<D>(
        &mut self,
        outcome: &LiveMotionOwnerOutcome,
        physical_event: Option<LivePhysicalStateEvent<AppliedCommandReceipt, D>>,
    ) -> Result<OperatorConsoleProcessDisposition, OperatorConsoleRuntimeAdapterError> {
        self.validate_uncorrelated_event(&physical_event)?;
        match outcome {
            LiveMotionOwnerOutcome::Idle
            | LiveMotionOwnerOutcome::ClientUnavailableBeforeClaim
            | LiveMotionOwnerOutcome::StatusReplied(_)
            | LiveMotionOwnerOutcome::Rejected { .. } => {
                if physical_event.is_some() {
                    return self
                        .fail(OperatorConsoleRuntimeAdapterError::UnexpectedPhysicalEvidence);
                }
                Ok(OperatorConsoleProcessDisposition::UnrelatedRuntimeOutcome)
            }
            LiveMotionOwnerOutcome::Completed(action) if socket_safety_action(*action) => {
                self.require_socket_safety_event(*action, physical_event)?;
                self.receiver.discard_queued_for_external_priority_stop(
                    ConsoleResponseRejectionCode::CancelledByPriorityStop,
                );
                self.cancel_all_unclaimed_authority_requests(
                    ConsoleResponseRejectionCode::CancelledByPriorityStop,
                )?;
                self.cancel_authority()
            }
            LiveMotionOwnerOutcome::ShutdownRequested => {
                self.require_socket_shutdown_event(physical_event)?;
                self.cancel_authority()
            }
            LiveMotionOwnerOutcome::Completed(_)
            | LiveMotionOwnerOutcome::SaveMapRequested { .. }
            | LiveMotionOwnerOutcome::AutonomousAccepted { .. }
            | LiveMotionOwnerOutcome::PeriodicManualApplied
            | LiveMotionOwnerOutcome::PeriodicManualStopped
            | LiveMotionOwnerOutcome::PeriodicAutonomousApplied { .. }
            | LiveMotionOwnerOutcome::PeriodicAutonomousStopped { .. }
            | LiveMotionOwnerOutcome::AutonomousCompleted { .. } => {
                self.fail(OperatorConsoleRuntimeAdapterError::OwnerOutcomeMismatch)
            }
        }
    }

    fn validate_response_identity(
        &self,
        pending: &PendingConsoleSubmission,
        response: AgentControlResponseV1,
    ) -> Result<(), OperatorConsoleRuntimeAdapterError> {
        if pending.submission.request_id().get() != pending.downstream_request_id.get() {
            return Err(OperatorConsoleRuntimeAdapterError::RequestIdentityMismatch);
        }
        if response.request_id() != Some(pending.submission.request_id()) {
            return Err(OperatorConsoleRuntimeAdapterError::ResponseIdentityMismatch);
        }
        match response.response() {
            AgentControlResponseKindV1::Accepted { command, .. }
                if command != pending.command_kind =>
            {
                Err(OperatorConsoleRuntimeAdapterError::ResponseCommandMismatch)
            }
            AgentControlResponseKindV1::Status { .. } => {
                Err(OperatorConsoleRuntimeAdapterError::ResponseOutcomeMismatch)
            }
            AgentControlResponseKindV1::Accepted { .. }
            | AgentControlResponseKindV1::Rejected { .. } => Ok(()),
        }
    }

    fn validate_owner_response(
        &self,
        pending: &PendingConsoleSubmission,
        outcome: &LiveMotionOwnerOutcome,
        response: AgentControlResponseV1,
    ) -> Result<(), OperatorConsoleRuntimeAdapterError> {
        match (outcome, response.response()) {
            (
                LiveMotionOwnerOutcome::Rejected {
                    code: expected_code,
                    retryable: expected_retryable,
                },
                AgentControlResponseKindV1::Rejected { code, retryable },
            ) if code == *expected_code && retryable == *expected_retryable => Ok(()),
            (
                LiveMotionOwnerOutcome::Completed(action),
                AgentControlResponseKindV1::Accepted {
                    completion: AgentControlCompletionV1::Completed,
                    ..
                },
            ) if action_matches_command(*action, pending.command) => Ok(()),
            (
                LiveMotionOwnerOutcome::AutonomousAccepted { mode },
                AgentControlResponseKindV1::Accepted {
                    completion: AgentControlCompletionV1::AcceptedForProcessing,
                    ..
                },
            ) if autonomous_mode_matches_command(*mode, pending.command) => Ok(()),
            (
                LiveMotionOwnerOutcome::AutonomousCompleted { mode },
                AgentControlResponseKindV1::Accepted {
                    completion: AgentControlCompletionV1::Completed,
                    ..
                },
            ) if autonomous_mode_matches_command(*mode, pending.command) => Ok(()),
            (LiveMotionOwnerOutcome::ShutdownRequested, _)
            | (LiveMotionOwnerOutcome::Idle, _)
            | (LiveMotionOwnerOutcome::ClientUnavailableBeforeClaim, _)
            | (LiveMotionOwnerOutcome::StatusReplied(_), _)
            | (LiveMotionOwnerOutcome::SaveMapRequested { .. }, _)
            | (LiveMotionOwnerOutcome::PeriodicManualApplied, _)
            | (LiveMotionOwnerOutcome::PeriodicManualStopped, _)
            | (LiveMotionOwnerOutcome::PeriodicAutonomousApplied { .. }, _)
            | (LiveMotionOwnerOutcome::PeriodicAutonomousStopped { .. }, _) => {
                Err(OperatorConsoleRuntimeAdapterError::OwnerOutcomeMismatch)
            }
            _ => Err(OperatorConsoleRuntimeAdapterError::ResponseOutcomeMismatch),
        }
    }

    fn validate_event_key_for_rejection<D>(
        &self,
        expected: AgentControlTypedRequestKey,
        command: OperatorConsoleCommand,
        event: &Option<LivePhysicalStateEvent<AppliedCommandReceipt, D>>,
    ) -> Result<(), OperatorConsoleRuntimeAdapterError> {
        match event {
            Some(LivePhysicalStateEvent::CoordinatorTick(applied)) => {
                if applied.typed_request_key() != Some(expected) {
                    return self.fail(
                        OperatorConsoleRuntimeAdapterError::PhysicalEvidenceCorrelationMismatch,
                    );
                }
                if !applied.stopped() || !exact_safe_zero(applied.receipt()) {
                    return self.fail(OperatorConsoleRuntimeAdapterError::UnsafeLifecycleZero);
                }
                Ok(())
            }
            Some(LivePhysicalStateEvent::LifecycleZero(applied)) => {
                if applied.typed_request_key() != Some(expected) {
                    return self.fail(
                        OperatorConsoleRuntimeAdapterError::PhysicalEvidenceCorrelationMismatch,
                    );
                }
                let expected_reason = match command {
                    OperatorConsoleCommand::ManualVelocity { .. } => {
                        LiveLifecycleZeroReason::ManualRelease
                    }
                    OperatorConsoleCommand::AutonomousFrontierExplore
                    | OperatorConsoleCommand::AutonomousPointGoal(_) => {
                        LiveLifecycleZeroReason::AutonomousRelease
                    }
                    OperatorConsoleCommand::Arm
                    | OperatorConsoleCommand::Disarm
                    | OperatorConsoleCommand::BeginManual
                    | OperatorConsoleCommand::AutonomousMapOnly
                    | OperatorConsoleCommand::Stop { .. }
                    | OperatorConsoleCommand::SaveMap => {
                        return self.fail(
                            OperatorConsoleRuntimeAdapterError::PhysicalEvidenceReasonMismatch,
                        );
                    }
                };
                if applied.reason() != expected_reason {
                    return self
                        .fail(OperatorConsoleRuntimeAdapterError::PhysicalEvidenceReasonMismatch);
                }
                if !exact_safe_zero(applied.receipt()) {
                    return self.fail(OperatorConsoleRuntimeAdapterError::UnsafeLifecycleZero);
                }
                Ok(())
            }
            None => Ok(()),
            Some(LivePhysicalStateEvent::ActuationFault { .. }) => {
                self.fail(OperatorConsoleRuntimeAdapterError::PhysicalEvidenceKindMismatch)
            }
        }
    }

    fn validate_uncorrelated_event<D>(
        &self,
        event: &Option<LivePhysicalStateEvent<AppliedCommandReceipt, D>>,
    ) -> Result<(), OperatorConsoleRuntimeAdapterError> {
        let correlated = match event {
            Some(LivePhysicalStateEvent::CoordinatorTick(applied)) => {
                applied.typed_request_key().is_some()
            }
            Some(LivePhysicalStateEvent::LifecycleZero(applied)) => {
                applied.typed_request_key().is_some()
            }
            Some(LivePhysicalStateEvent::ActuationFault { .. }) | None => false,
        };
        if correlated {
            self.fail(OperatorConsoleRuntimeAdapterError::UnexpectedTypedCorrelation)
        } else {
            Ok(())
        }
    }

    fn require_coordinator_event<D>(
        &self,
        event: Option<LivePhysicalStateEvent<AppliedCommandReceipt, D>>,
    ) -> Result<(), OperatorConsoleRuntimeAdapterError> {
        match event {
            Some(LivePhysicalStateEvent::CoordinatorTick(_)) => Ok(()),
            Some(LivePhysicalStateEvent::LifecycleZero(_))
            | Some(LivePhysicalStateEvent::ActuationFault { .. }) => {
                self.fail(OperatorConsoleRuntimeAdapterError::PhysicalEvidenceKindMismatch)
            }
            None => self.fail(OperatorConsoleRuntimeAdapterError::PhysicalEvidenceRequired),
        }
    }

    fn require_safe_termination_event<D>(
        &self,
        event: Option<LivePhysicalStateEvent<AppliedCommandReceipt, D>>,
    ) -> Result<(), OperatorConsoleRuntimeAdapterError> {
        match event {
            Some(LivePhysicalStateEvent::CoordinatorTick(applied)) => {
                if applied.stopped() && exact_safe_zero(applied.receipt()) {
                    Ok(())
                } else {
                    self.fail(OperatorConsoleRuntimeAdapterError::UnsafeLifecycleZero)
                }
            }
            Some(LivePhysicalStateEvent::LifecycleZero(applied)) => {
                if matches!(
                    applied.reason(),
                    LiveLifecycleZeroReason::ManualRelease
                        | LiveLifecycleZeroReason::AutonomousRelease
                ) && exact_safe_zero(applied.receipt())
                {
                    Ok(())
                } else {
                    self.fail(OperatorConsoleRuntimeAdapterError::PhysicalEvidenceReasonMismatch)
                }
            }
            Some(LivePhysicalStateEvent::ActuationFault { .. }) => {
                self.fail(OperatorConsoleRuntimeAdapterError::PhysicalEvidenceKindMismatch)
            }
            None => self.fail(OperatorConsoleRuntimeAdapterError::PhysicalEvidenceRequired),
        }
    }

    fn require_socket_safety_event<D>(
        &self,
        action: LiveMotionCompletedSafetyAction,
        event: Option<LivePhysicalStateEvent<AppliedCommandReceipt, D>>,
    ) -> Result<(), OperatorConsoleRuntimeAdapterError> {
        let Some(LivePhysicalStateEvent::LifecycleZero(applied)) = event else {
            return self.fail(OperatorConsoleRuntimeAdapterError::PhysicalEvidenceKindMismatch);
        };
        let expected = match action {
            LiveMotionCompletedSafetyAction::MappingOnlyStopped => {
                LiveLifecycleZeroReason::MappingOnlyRequest
            }
            LiveMotionCompletedSafetyAction::GlobalStopped => {
                LiveLifecycleZeroReason::GlobalStopRequest
            }
            LiveMotionCompletedSafetyAction::ManualStopped => {
                LiveLifecycleZeroReason::ManualRelease
            }
            LiveMotionCompletedSafetyAction::Disarmed => LiveLifecycleZeroReason::DisarmRequest,
            _ => {
                return self.fail(OperatorConsoleRuntimeAdapterError::OwnerOutcomeMismatch);
            }
        };
        if applied.reason() != expected {
            return self.fail(OperatorConsoleRuntimeAdapterError::PhysicalEvidenceReasonMismatch);
        }
        if !exact_safe_zero(applied.receipt()) {
            return self.fail(OperatorConsoleRuntimeAdapterError::UnsafeLifecycleZero);
        }
        Ok(())
    }

    fn require_socket_shutdown_event<D>(
        &self,
        event: Option<LivePhysicalStateEvent<AppliedCommandReceipt, D>>,
    ) -> Result<(), OperatorConsoleRuntimeAdapterError> {
        let Some(LivePhysicalStateEvent::LifecycleZero(applied)) = event else {
            return self.fail(OperatorConsoleRuntimeAdapterError::PhysicalEvidenceKindMismatch);
        };
        if applied.reason() != LiveLifecycleZeroReason::ShutdownRequest {
            return self.fail(OperatorConsoleRuntimeAdapterError::PhysicalEvidenceReasonMismatch);
        }
        if !exact_safe_zero(applied.receipt()) {
            return self.fail(OperatorConsoleRuntimeAdapterError::UnsafeLifecycleZero);
        }
        Ok(())
    }

    fn cancel_older_unclaimed(
        &mut self,
        barrier: ConsoleDownstreamRequestId,
        reason: ConsoleResponseRejectionCode,
    ) -> Result<(), OperatorConsoleRuntimeAdapterError> {
        let keys = self
            .pending
            .iter()
            .filter_map(|(key, pending)| {
                (pending.downstream_request_id < barrier
                    && command_can_acquire_or_renew_authority(pending.command))
                .then_some(*key)
            })
            .collect::<Vec<_>>();
        for key in keys {
            let Some(pending) = self.pending.remove(&key) else {
                return self.fail(OperatorConsoleRuntimeAdapterError::MissingPendingCorrelation);
            };
            match pending.submission.try_take_claim() {
                Err(AgentControlTypedSubmissionPollError::Pending) => {
                    pending.response.reject(reason);
                    drop(pending.submission);
                }
                Ok(()) => {
                    return self.fail_pending(
                        pending,
                        OperatorConsoleRuntimeAdapterError::CancellationAfterClaim,
                    );
                }
                Err(AgentControlTypedSubmissionPollError::ClaimDisconnected) => {
                    return self.fail_pending(
                        pending,
                        OperatorConsoleRuntimeAdapterError::ClaimChannelDisconnected,
                    );
                }
                Err(AgentControlTypedSubmissionPollError::ResponseDisconnected) => {
                    return self.fail_pending(
                        pending,
                        OperatorConsoleRuntimeAdapterError::ResponseChannelDisconnected,
                    );
                }
            }
        }
        Ok(())
    }

    /// A raw-socket safety command can overtake typed console requests because
    /// the runtime has a dedicated priority lane. Once its exact zero is
    /// observed, no previously queued authority acquisition may survive and
    /// execute after that safety barrier.
    fn cancel_all_unclaimed_authority_requests(
        &mut self,
        reason: ConsoleResponseRejectionCode,
    ) -> Result<(), OperatorConsoleRuntimeAdapterError> {
        let keys = self
            .pending
            .iter()
            .filter_map(|(key, pending)| {
                command_can_acquire_or_renew_authority(pending.command).then_some(*key)
            })
            .collect::<Vec<_>>();
        for key in keys {
            let Some(pending) = self.pending.remove(&key) else {
                return self.fail(OperatorConsoleRuntimeAdapterError::MissingPendingCorrelation);
            };
            match pending.submission.try_take_claim() {
                Err(AgentControlTypedSubmissionPollError::Pending) => {
                    pending.response.reject(reason);
                    drop(pending.submission);
                }
                Ok(()) => {
                    return self.fail_pending(
                        pending,
                        OperatorConsoleRuntimeAdapterError::CancellationAfterClaim,
                    );
                }
                Err(AgentControlTypedSubmissionPollError::ClaimDisconnected) => {
                    return self.fail_pending(
                        pending,
                        OperatorConsoleRuntimeAdapterError::ClaimChannelDisconnected,
                    );
                }
                Err(AgentControlTypedSubmissionPollError::ResponseDisconnected) => {
                    return self.fail_pending(
                        pending,
                        OperatorConsoleRuntimeAdapterError::ResponseChannelDisconnected,
                    );
                }
            }
        }
        Ok(())
    }

    fn cancel_all_pending(&mut self, reason: ConsoleResponseRejectionCode) {
        for (_, pending) in self.pending.drain() {
            pending.response.reject(reason);
            drop(pending.submission);
        }
        if let Some(pending) = self.awaiting_save_map.take() {
            pending.response.reject(reason);
            drop(pending.submission);
        }
    }

    fn finish_nonphysical_completion(
        &mut self,
        pending: PendingConsoleSubmission,
    ) -> Result<(), OperatorConsoleRuntimeAdapterError> {
        match pending.response.completed() {
            Ok(OperatorConsoleCompletion::Completed) => Ok(()),
            Ok(OperatorConsoleCompletion::Authority(guard)) => {
                guard.cancelled();
                self.fail(OperatorConsoleRuntimeAdapterError::AuthorityConflict)
            }
            Err(source) => self.fail(OperatorConsoleRuntimeAdapterError::Completion(source)),
        }
    }

    fn require_authority_kind(
        &self,
        expected: OperatorConsoleRetainedAuthorityKind,
    ) -> Result<(), OperatorConsoleRuntimeAdapterError> {
        let Some(authority) = self.authority.as_ref() else {
            return self.fail(OperatorConsoleRuntimeAdapterError::AuthorityMissing);
        };
        if authority.kind != expected {
            return self.fail(OperatorConsoleRuntimeAdapterError::AuthorityModeMismatch);
        }
        Ok(())
    }

    fn cancel_authority(
        &mut self,
    ) -> Result<OperatorConsoleProcessDisposition, OperatorConsoleRuntimeAdapterError> {
        let Some(authority) = self.authority.take() else {
            return Ok(OperatorConsoleProcessDisposition::UnrelatedRuntimeOutcome);
        };
        let id = authority.guard.generation();
        authority.guard.cancelled();
        Ok(OperatorConsoleProcessDisposition::AuthorityCancelled {
            downstream_request_id: id,
        })
    }

    fn cancel_authority_if_present(&mut self) -> Option<ConsoleDownstreamRequestId> {
        self.authority.take().map(|authority| {
            let id = authority.guard.generation();
            authority.guard.cancelled();
            id
        })
    }

    fn complete_authority(
        &mut self,
    ) -> Result<OperatorConsoleProcessDisposition, OperatorConsoleRuntimeAdapterError> {
        let Some(authority) = self.authority.take() else {
            return self.fail(OperatorConsoleRuntimeAdapterError::AuthorityMissing);
        };
        let id = authority.guard.generation();
        authority.guard.completed();
        Ok(OperatorConsoleProcessDisposition::AuthorityCompleted {
            downstream_request_id: id,
        })
    }

    fn ensure_no_deferred_completion(&self) -> Result<(), OperatorConsoleRuntimeAdapterError> {
        if self.pending_emergency_stop.is_some() {
            return self.fail(OperatorConsoleRuntimeAdapterError::EmergencyStopCompletionPending);
        }
        if self.awaiting_save_map.is_some() {
            return self.fail(OperatorConsoleRuntimeAdapterError::SaveMapCompletionPending);
        }
        Ok(())
    }

    fn fail_pending<T>(
        &self,
        pending: PendingConsoleSubmission,
        source: OperatorConsoleRuntimeAdapterError,
    ) -> Result<T, OperatorConsoleRuntimeAdapterError> {
        pending
            .response
            .reject(ConsoleResponseRejectionCode::InternalFault);
        drop(pending.submission);
        self.fail(source)
    }

    fn fail<T>(
        &self,
        source: OperatorConsoleRuntimeAdapterError,
    ) -> Result<T, OperatorConsoleRuntimeAdapterError> {
        self.handle.signal_internal_fail_closed();
        Err(source)
    }

    fn fail_ingress<T, E, J>(
        &self,
        source: OperatorConsoleRuntimeAdapterError,
    ) -> Result<T, OperatorConsoleRuntimeIngressError<E, J>> {
        self.handle.signal_internal_fail_closed();
        Err(OperatorConsoleRuntimeIngressError::Adapter(source))
    }

    fn fail_pending_ingress<T, E, J>(
        &self,
        pending: PendingConsoleSubmission,
        source: OperatorConsoleRuntimeAdapterError,
    ) -> Result<T, OperatorConsoleRuntimeIngressError<E, J>> {
        pending
            .response
            .reject(ConsoleResponseRejectionCode::InternalFault);
        drop(pending.submission);
        self.fail_ingress(source)
    }
}

fn priority_cancellation_reason(
    command: OperatorConsoleCommand,
) -> Option<ConsoleResponseRejectionCode> {
    match command {
        OperatorConsoleCommand::Stop {
            cause: ConsoleStopCause::ManualDeadman,
        } => Some(ConsoleResponseRejectionCode::CancelledByManualDeadman),
        OperatorConsoleCommand::Stop { .. }
        | OperatorConsoleCommand::Disarm
        | OperatorConsoleCommand::AutonomousMapOnly => {
            Some(ConsoleResponseRejectionCode::CancelledByPriorityStop)
        }
        OperatorConsoleCommand::Arm
        | OperatorConsoleCommand::BeginManual
        | OperatorConsoleCommand::ManualVelocity { .. }
        | OperatorConsoleCommand::AutonomousFrontierExplore
        | OperatorConsoleCommand::AutonomousPointGoal(_)
        | OperatorConsoleCommand::SaveMap => None,
    }
}

fn command_can_acquire_or_renew_authority(command: OperatorConsoleCommand) -> bool {
    matches!(
        command,
        OperatorConsoleCommand::Arm
            | OperatorConsoleCommand::BeginManual
            | OperatorConsoleCommand::ManualVelocity { .. }
            | OperatorConsoleCommand::AutonomousFrontierExplore
            | OperatorConsoleCommand::AutonomousPointGoal(_)
    )
}

fn command_is_safety_reducing(command: OperatorConsoleCommand) -> bool {
    matches!(
        command,
        OperatorConsoleCommand::Disarm
            | OperatorConsoleCommand::AutonomousMapOnly
            | OperatorConsoleCommand::Stop { .. }
    )
}

fn authority_kind_for_command(
    command: OperatorConsoleCommand,
) -> Option<OperatorConsoleRetainedAuthorityKind> {
    match command {
        OperatorConsoleCommand::BeginManual => Some(OperatorConsoleRetainedAuthorityKind::Manual),
        OperatorConsoleCommand::AutonomousFrontierExplore => Some(
            OperatorConsoleRetainedAuthorityKind::Autonomous(AgentAutonomousMode::Explore),
        ),
        OperatorConsoleCommand::AutonomousPointGoal(_) => Some(
            OperatorConsoleRetainedAuthorityKind::Autonomous(AgentAutonomousMode::PointGoal),
        ),
        OperatorConsoleCommand::Arm
        | OperatorConsoleCommand::Disarm
        | OperatorConsoleCommand::ManualVelocity { .. }
        | OperatorConsoleCommand::AutonomousMapOnly
        | OperatorConsoleCommand::Stop { .. }
        | OperatorConsoleCommand::SaveMap => None,
    }
}

fn autonomous_mode_matches_command(
    mode: AgentAutonomousMode,
    command: OperatorConsoleCommand,
) -> bool {
    matches!(
        (mode, command),
        (
            AgentAutonomousMode::Explore,
            OperatorConsoleCommand::AutonomousFrontierExplore
        ) | (
            AgentAutonomousMode::PointGoal,
            OperatorConsoleCommand::AutonomousPointGoal(_)
        )
    )
}

fn action_matches_command(
    action: LiveMotionCompletedSafetyAction,
    command: OperatorConsoleCommand,
) -> bool {
    matches!(
        (action, command),
        (
            LiveMotionCompletedSafetyAction::Armed,
            OperatorConsoleCommand::Arm
        ) | (
            LiveMotionCompletedSafetyAction::Disarmed,
            OperatorConsoleCommand::Disarm
        ) | (
            LiveMotionCompletedSafetyAction::ManualStarted { .. },
            OperatorConsoleCommand::BeginManual
        ) | (
            LiveMotionCompletedSafetyAction::ManualCommandApplied,
            OperatorConsoleCommand::ManualVelocity { .. }
        ) | (
            LiveMotionCompletedSafetyAction::MappingOnlyStopped,
            OperatorConsoleCommand::AutonomousMapOnly
        ) | (
            LiveMotionCompletedSafetyAction::GlobalStopped,
            OperatorConsoleCommand::Stop { .. }
        )
    )
}

fn socket_safety_action(action: LiveMotionCompletedSafetyAction) -> bool {
    matches!(
        action,
        LiveMotionCompletedSafetyAction::MappingOnlyStopped
            | LiveMotionCompletedSafetyAction::GlobalStopped
            | LiveMotionCompletedSafetyAction::ManualStopped
            | LiveMotionCompletedSafetyAction::Disarmed
    )
}

fn exact_safe_zero(receipt: &AppliedCommandReceipt) -> bool {
    receipt.verified_host_result().requested_timer_pwm.is_zero() && receipt.is_confirmed_zero()
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::num::NonZeroU32;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::time::{Duration, Instant};

    use kiko_supervisor_core::{
        AuthorityDuration, ReadinessBinding, ReadinessEpoch, Sha256Digest, SupervisorConfig,
    };
    use robot_command_client::fake::{FakeClock, FakeStep, FakeTransport};
    use robot_command_client::{
        ClientConfig, ClientConfigInput, DisarmedCommandClient, MonotonicClock,
        PendingPhysicalCommand,
    };
    use robot_protocol::ControllerUptimeMsWrapping;
    use robot_protocol::v2::{
        AcquireResult, AcquireResultCode, ActuatorConfigFingerprint, ControlEpoch,
        ControllerBootId, ControllerCapabilities, ControllerDeadlineMsWrapping, ControllerFaults,
        ControllerUid, HostCommandResult, HostCommandResultCode, HostStopResult, Message,
        MessageKind, OutputState, RemainingLeaseMs, RequestId, StatusCode, StatusReport,
        StopResultCode, TargetBootId, TimerPwm, V2CommandLeaseMs, V2CommandSequence,
    };
    use serde_json::json;

    use super::super::live_motion_owner::LiveMotionTerminalStop;

    use super::super::control_socket::{
        agent_control_test_runtime_with_typed_ingress, enqueue_agent_control_test_request,
        enqueue_agent_control_test_request_through_runtime_lanes,
    };
    use super::super::mpc::HostMonotonicClockReadError;
    use super::*;
    use crate::dense::occupancy::{
        DepthCameraModel, DepthToTrackingCamera, OccupancyCell, OccupancyGridGeometry,
        OccupancyGridSnapshot,
    };
    use crate::map::SlamMap;
    use crate::navigation::{
        AGENT_CONTROL_SCHEMA_V1, AgentAuthoritySupervisor, AgentControlDispatcher,
        AgentControlMonotonicOrigin, AgentControlRequestParser, AgentControlRuntimeQueueCapacity,
        AgentControllerStopKnowledge, AgentLiveActuationFaultKind, AgentManualControlCore,
        AgentManualRuntimePolicy, AgentMapStateV1, AgentRuntimeStateV1, ConsoleIdempotencyKey,
        ConsoleRuntimeResponseState, ConsoleSessionCapability, ConsoleSnapshotRevision,
        ConsoleSourceKind, ConsoleSourceSequence, LiveMotionActuationFaultEvidence,
        MANUAL_DRIVE_CONFIG_V1, ManualDriveConfigV1, ManualDriveConfigV1Dto, NanoLiveModePolicy,
        NavigationClockEpoch, NavigationIngressCapacity, NavigationIngressLog,
        NavigationRecordingId, OperatorConsoleIntent, OperatorConsoleLimits,
        OperatorConsoleSnapshot, PathReferenceBuilderV1, PendingVisualAttemptIngress,
        PlanarOdometry, ShadowNavigationConfigV2, ShadowNavigationCoordinator,
        ShadowSafetySupervisor, VisualAdmission, VisualAdmissionOutcome, VisualAttemptOutcome,
        operator_console,
    };
    use crate::{
        DeviceSessionId, Frame, FrameDimensions, FrameId, HostMonotonicTimestamp, MapLocalization,
        PairingWindowNs, PinholeIntrinsics, Pose, SensorId, StereoObservation, StereoPair,
        Timestamp, VisualFrameStamp, WorldToCamera,
    };

    const BASE_NS: u64 = 1_000_000_000;
    const OWNER_NS: u64 = BASE_NS + 1_000_000;
    const UID_BYTES: [u8; 12] = [0x11; 12];
    const FINGERPRINT_BYTES: [u8; 16] = [0x22; 16];
    const RESPONSE_DELAY: Duration = Duration::from_millis(1);

    fn at(nanos: u64) -> HostMonotonicTimestamp {
        HostMonotonicTimestamp::from_nanos(nanos)
    }

    fn duration(nanos: u64) -> AuthorityDuration {
        AuthorityDuration::try_from_nanos(nanos).expect("nonzero authority duration")
    }

    fn uid() -> ControllerUid {
        ControllerUid::try_new(UID_BYTES).expect("fixture controller UID")
    }

    fn boot() -> ControllerBootId {
        ControllerBootId::try_new(17).expect("fixture boot ID")
    }

    fn epoch() -> ControlEpoch {
        ControlEpoch::try_new(23).expect("fixture control epoch")
    }

    fn capabilities() -> ControllerCapabilities {
        ControllerCapabilities::try_from_bits(ControllerCapabilities::REQUIRED_BITS)
            .expect("production capability bits")
    }

    fn lease() -> V2CommandLeaseMs {
        V2CommandLeaseMs::try_new(100).expect("fixture command lease")
    }

    fn client_config() -> ClientConfig {
        ClientConfig::parse(ClientConfigInput {
            command_endpoint: "127.0.0.1:8080",
            controller_uid_hex: "111111111111111111111111",
            expected_firmware_abi: "7",
            expected_firmware_build_id: "9",
            expected_actuator_config_fingerprint_hex: "22222222222222222222222222222222",
            status_timeout_ns: "50000000",
            acquire_timeout_ns: "50000000",
            applied_ack_timeout_ns: "50000000",
            stop_attempt_timeout_ns: "50000000",
            max_stop_recovery_attempts: "3",
            zero_acquisition_lease_ms: "100",
        })
        .expect("fixture client config")
    }

    fn status_report() -> Message {
        Message::StatusReport(StatusReport {
            controller_uid: uid(),
            observed_boot_id: TargetBootId::Exact(boot()),
            request_id: RequestId::new(0),
            status: StatusCode::ReadyStopped,
            control_epoch: None,
            controller_uptime: ControllerUptimeMsWrapping::new(1_000),
            capabilities: capabilities(),
            output_state: OutputState::Disabled,
            controller_timer_pwm: TimerPwm::ZERO,
            remaining_lease: RemainingLeaseMs::ZERO,
            faults: ControllerFaults::NONE,
        })
    }

    fn acquire_result() -> Message {
        Message::AcquireResult(AcquireResult {
            controller_uid: uid(),
            boot_id: boot(),
            request_id: RequestId::new(1),
            control_epoch: Some(epoch()),
            result: AcquireResultCode::Granted,
            capabilities: capabilities(),
            faults: ControllerFaults::NONE,
            observed_firmware_abi: 7,
            observed_firmware_build_id: 9,
            observed_actuator_config_fingerprint: ActuatorConfigFingerprint::try_new(
                FINGERPRINT_BYTES,
            )
            .expect("fixture fingerprint"),
        })
    }

    fn command_result(sequence: u32) -> Message {
        Message::HostCommandResult(HostCommandResult {
            controller_uid: uid(),
            boot_id: boot(),
            control_epoch: epoch(),
            sequence: V2CommandSequence::new(sequence),
            result: HostCommandResultCode::AppliedNew,
            requested_timer_pwm: TimerPwm::ZERO,
            controller_timer_pwm: TimerPwm::ZERO,
            output_state: OutputState::ZeroPwm,
            controller_applied_at: ControllerUptimeMsWrapping::new(2_000 + sequence),
            controller_expires_at: ControllerDeadlineMsWrapping::new(2_100 + sequence),
            remaining_lease: RemainingLeaseMs::try_new(90).expect("remaining fixture lease"),
            faults: ControllerFaults::NONE,
        })
    }

    fn stop_result() -> Message {
        Message::HostStopResult(HostStopResult {
            controller_uid: uid(),
            observed_boot_id: TargetBootId::Exact(boot()),
            request_id: RequestId::new(2),
            result: StopResultCode::ControllerConfirmed,
            output_state: OutputState::Disabled,
            controller_uptime: ControllerUptimeMsWrapping::new(3_000),
            faults: ControllerFaults::NONE,
        })
    }

    fn applied_zero_receipts(count: usize) -> VecDeque<AppliedCommandReceipt> {
        assert!(count > 0);
        let clock = FakeClock::default();
        let mut steps = vec![
            FakeStep::respond(MessageKind::StatusQuery, RESPONSE_DELAY, status_report()),
            FakeStep::respond(
                MessageKind::AcquireControl,
                RESPONSE_DELAY,
                acquire_result(),
            ),
            FakeStep::respond(MessageKind::HostCommand, RESPONSE_DELAY, command_result(0)),
        ];
        for sequence in 1..count {
            steps.push(FakeStep::respond(
                MessageKind::HostCommand,
                RESPONSE_DELAY,
                command_result(u32::try_from(sequence).expect("bounded fixture sequence")),
            ));
        }
        steps.push(FakeStep::respond(
            MessageKind::HostStop,
            RESPONSE_DELAY,
            stop_result(),
        ));
        let (transport, _probe) = FakeTransport::scripted(clock.clone(), steps);
        let client = DisarmedCommandClient::new(transport, clock.clone(), client_config());
        let (mut armed, first) = client
            .acquire_zero()
            .ok()
            .expect("fixture zero acquisition");
        let mut receipts = VecDeque::from([first]);
        for _ in 1..count {
            let deadline = clock
                .now()
                .checked_add(Duration::from_millis(50))
                .expect("fixture acknowledgement deadline");
            let pending = PendingPhysicalCommand::new(TimerPwm::ZERO, lease(), deadline);
            let (next, receipt) = armed.apply(pending).ok().expect("fixture zero refresh");
            armed = next;
            receipts.push_back(receipt);
        }
        drop(armed);
        receipts
    }

    fn supervisor_host_zero(sequence: u32) -> HostCommandResult {
        HostCommandResult {
            controller_uid: uid(),
            boot_id: boot(),
            control_epoch: epoch(),
            sequence: V2CommandSequence::new(sequence),
            result: HostCommandResultCode::AppliedNew,
            requested_timer_pwm: TimerPwm::ZERO,
            controller_timer_pwm: TimerPwm::ZERO,
            output_state: OutputState::ZeroPwm,
            controller_applied_at: ControllerUptimeMsWrapping::new(sequence),
            controller_expires_at: ControllerDeadlineMsWrapping::new(sequence + 100),
            remaining_lease: RemainingLeaseMs::try_new(90).expect("supervisor fixture lease"),
            faults: ControllerFaults::NONE,
        }
    }

    fn disarmed_supervisor() -> AgentAuthoritySupervisor {
        let config =
            SupervisorConfig::new(duration(10_000), duration(100)).expect("supervisor config");
        let mut authority =
            AgentAuthoritySupervisor::new(config, NavigationClockEpoch::new(at(BASE_NS)));
        authority
            .begin_inventory(at(BASE_NS + 1))
            .expect("inventory transition");
        authority
            .admit_readiness(
                ReadinessBinding::new(
                    ReadinessEpoch::try_new(1).expect("readiness epoch"),
                    uid(),
                    boot(),
                    epoch(),
                    Sha256Digest::try_new([2; 32]).expect("hardware digest"),
                    Sha256Digest::try_new([3; 32]).expect("calibration digest"),
                ),
                at(BASE_NS + 2),
            )
            .expect("readiness admission");
        authority
    }

    fn ready_supervisor() -> AgentAuthoritySupervisor {
        let mut authority = disarmed_supervisor();
        authority.arm(at(BASE_NS + 3)).expect("begin arm");
        authority
            .admit_applied_zero(supervisor_host_zero(1), at(BASE_NS + 4), at(BASE_NS + 4))
            .expect("arm zero");
        authority
    }

    fn ready_supervisor_with_recent_zero() -> AgentAuthoritySupervisor {
        let mut authority = disarmed_supervisor();
        authority
            .arm(at(OWNER_NS - 2))
            .expect("begin recently armed fixture");
        authority
            .admit_applied_zero(supervisor_host_zero(1), at(OWNER_NS - 1), at(OWNER_NS - 1))
            .expect("recent arm zero");
        authority
    }

    fn manual_policy() -> AgentManualRuntimePolicy {
        AgentManualRuntimePolicy::for_test(
            duration(1_000),
            ManualDriveConfigV1::parse(ManualDriveConfigV1Dto {
                schema_version: MANUAL_DRIVE_CONFIG_V1,
                maximum_abs_forward_velocity_mps: 0.5,
                maximum_abs_yaw_rate_rad_s: 1.0,
                maximum_command_age_ns: 50,
                deadman_timeout_ns: 50,
            })
            .expect("manual policy"),
        )
    }

    #[derive(Clone)]
    struct TestHostClock {
        next: Arc<AtomicU64>,
    }

    impl TestHostClock {
        fn new(nanos: u64) -> Self {
            Self {
                next: Arc::new(AtomicU64::new(nanos)),
            }
        }

        fn peek(&self) -> HostMonotonicTimestamp {
            at(self.next.load(Ordering::SeqCst))
        }

        fn advance(&self, nanos: u64) {
            self.next.fetch_add(nanos, Ordering::SeqCst);
        }
    }

    impl HostMonotonicClock for TestHostClock {
        fn try_now(&mut self) -> Result<HostMonotonicTimestamp, HostMonotonicClockReadError> {
            Ok(at(self.next.fetch_add(1, Ordering::SeqCst)))
        }
    }

    #[derive(Debug)]
    struct TestPortError {
        stop_confirmed: bool,
    }

    impl TestPortError {
        const UNCERTAIN: Self = Self {
            stop_confirmed: false,
        };
        const CONFIRMED_STOP: Self = Self {
            stop_confirmed: true,
        };
    }

    impl fmt::Display for TestPortError {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("fixture receipt queue exhausted")
        }
    }

    impl std::error::Error for TestPortError {}

    struct TestReceiptPort {
        receipts: VecDeque<AppliedCommandReceipt>,
        fail_disarm: bool,
        disarm_failure_stop_confirmed: bool,
        fail_next_zero: Arc<AtomicBool>,
        force_tick_stop: Arc<AtomicBool>,
    }

    impl<J> super::super::live_motion_owner::LiveMotionTerminalActuationPort<J, TestHostClock>
        for TestReceiptPort
    where
        J: NavigationIngressSink,
    {
        type StopEvidence = ();

        fn disarm(&mut self) -> Result<Self::StopEvidence, Self::Error> {
            if self.fail_disarm {
                Err(if self.disarm_failure_stop_confirmed {
                    TestPortError::CONFIRMED_STOP
                } else {
                    TestPortError::UNCERTAIN
                })
            } else {
                Ok(())
            }
        }
    }

    impl<J> LiveMotionActuationPort<J, TestHostClock> for TestReceiptPort
    where
        J: NavigationIngressSink,
    {
        type Receipt = AppliedCommandReceipt;
        type Diagnostic = ();
        type Error = TestPortError;

        fn apply_fresh_zero(&mut self) -> Result<Self::Receipt, Self::Error> {
            if self.fail_next_zero.swap(false, Ordering::SeqCst) {
                return Err(TestPortError::UNCERTAIN);
            }
            self.receipts.pop_front().ok_or(TestPortError::UNCERTAIN)
        }

        fn tick_manual(
            &mut self,
            _coordinator: &mut ShadowNavigationCoordinator<J>,
            tick: HostMonotonicTimestamp,
            command: super::super::ManualDriveOutput<kiko_supervisor_core::AuthorityLeaseId>,
            _clock: &mut TestHostClock,
        ) -> super::super::live_motion_owner::LiveMotionPortTickResult<
            Self::Receipt,
            Self::Diagnostic,
            Self::Error,
        > {
            let requested_stop = command.target().is_stop();
            let stopped = requested_stop || self.force_tick_stop.swap(false, Ordering::SeqCst);
            let receipt = self.receipts.pop_front().ok_or(
                super::super::live_motion_owner::LiveMotionPortTickError::Actuation(
                    TestPortError::UNCERTAIN,
                ),
            )?;
            Ok(
                super::super::live_motion_owner::LiveMotionApplied::for_test(
                    tick,
                    receipt,
                    (),
                    stopped,
                    !requested_stop && stopped,
                    false,
                ),
            )
        }

        fn tick_point_goal(
            &mut self,
            _coordinator: &mut ShadowNavigationCoordinator<J>,
            _tick: HostMonotonicTimestamp,
            _clock: &mut TestHostClock,
        ) -> super::super::live_motion_owner::LiveMotionPortTickResult<
            Self::Receipt,
            Self::Diagnostic,
            Self::Error,
        > {
            panic!("point-goal ticking is outside this adapter fixture")
        }

        fn tick_frontier_yaw(
            &mut self,
            _coordinator: &mut ShadowNavigationCoordinator<J>,
            _tick: HostMonotonicTimestamp,
            _command: super::super::FrontierYawScanCommandV1,
            _clock: &mut TestHostClock,
        ) -> super::super::live_motion_owner::LiveMotionPortTickResult<
            Self::Receipt,
            Self::Diagnostic,
            Self::Error,
        > {
            panic!("frontier ticking is outside this adapter fixture")
        }

        fn classify_error(source: &Self::Error) -> LiveMotionActuationFaultEvidence {
            LiveMotionActuationFaultEvidence::new(
                AgentLiveActuationFaultKind::TransportUnavailable,
                if source.stop_confirmed {
                    AgentControllerStopKnowledge::Confirmed
                } else {
                    AgentControllerStopKnowledge::Uncertain
                },
            )
        }
    }

    type Journal = NavigationIngressLog;
    type TestOwner = LiveMotionOwner<Journal, TestReceiptPort, TestHostClock>;

    fn coordinator() -> ShadowNavigationCoordinator<Journal> {
        let dimensions = FrameDimensions::try_new(640, 400).expect("depth dimensions");
        let camera = DepthCameraModel::new(
            PinholeIntrinsics::try_new(400.0, 400.0, 320.0, 200.0).expect("depth intrinsics"),
            dimensions,
            DepthToTrackingCamera::new(Pose::identity()),
        );
        let parsed = ShadowNavigationConfigV2::parse_json(
            include_bytes!("../../../../configs/navigation-shadow-v2.example.json"),
            camera,
        )
        .expect("navigation policy");
        let parts = parsed.into_runtime_parts();
        let mpc_config = parts.mpc_solver.config();
        let journal = NavigationIngressLog::new(
            NavigationRecordingId::try_new([0x5a; 16]).expect("recording ID"),
            NavigationIngressCapacity::try_new(256).expect("journal capacity"),
        );
        let odometry = PlanarOdometry::new(parts.odometry);
        let local_costmap = super::super::LocalCostmap::try_new(
            parts.local_costmap,
            DeviceSessionId::try_new(7).expect("device session"),
        )
        .expect("local costmap");
        let reference_builder = PathReferenceBuilderV1::new(parts.path_reference);
        let safety = ShadowSafetySupervisor::try_new(parts.mpc_solver, parts.shadow_command)
            .expect("safety supervisor");
        ShadowNavigationCoordinator::new_without_goal(
            NavigationClockEpoch::new(at(BASE_NS)),
            journal,
            odometry,
            local_costmap,
            parts.global_planner,
            reference_builder,
            mpc_config,
            parts.solver_budget,
            safety,
        )
    }

    struct Fixture {
        raw_sender: super::super::AgentControlRuntimeSender,
        owner: TestOwner,
        adapter: OperatorConsoleRuntimeAdapter,
        handle: OperatorConsoleHandle,
        session: super::super::ConsoleSessionId,
        capability: ConsoleSessionCapability,
        fail_next_zero: Arc<AtomicBool>,
        force_tick_stop: Arc<AtomicBool>,
        clock: TestHostClock,
    }

    fn fixture(authority: AgentAuthoritySupervisor, runtime_capacity: usize) -> Fixture {
        fixture_with_terminal_stop(authority, runtime_capacity, false)
    }

    fn fixture_with_terminal_stop(
        authority: AgentAuthoritySupervisor,
        runtime_capacity: usize,
        fail_disarm: bool,
    ) -> Fixture {
        fixture_with_terminal_stop_evidence(authority, runtime_capacity, fail_disarm, false)
    }

    fn fixture_with_terminal_stop_evidence(
        authority: AgentAuthoritySupervisor,
        runtime_capacity: usize,
        fail_disarm: bool,
        disarm_failure_stop_confirmed: bool,
    ) -> Fixture {
        let capacity = AgentControlRuntimeQueueCapacity::try_new(runtime_capacity)
            .expect("runtime queue capacity");
        let (raw_sender, receiver, typed_ingress) =
            agent_control_test_runtime_with_typed_ingress(capacity);
        let dispatcher = AgentControlDispatcher::new_with_unified_console_authority(
            receiver,
            AgentControlMonotonicOrigin::new(Instant::now(), at(OWNER_NS)),
            AgentManualControlCore::new(authority, Some(manual_policy())),
        );
        let clock = TestHostClock::new(OWNER_NS);
        let mut receipts = applied_zero_receipts(10);
        // The supervisor readiness fixture already observed sequence 1. The
        // production client would be the same retained session; discard the
        // acquisition/sequence-1 construction receipts so every owner event is
        // strictly newer.
        receipts.pop_front().expect("acquisition receipt");
        receipts.pop_front().expect("sequence-one receipt");
        let fail_next_zero = Arc::new(AtomicBool::new(false));
        let force_tick_stop = Arc::new(AtomicBool::new(false));
        let owner = LiveMotionOwner::new(
            dispatcher,
            coordinator(),
            TestReceiptPort {
                receipts,
                fail_disarm,
                disarm_failure_stop_confirmed,
                fail_next_zero: Arc::clone(&fail_next_zero),
                force_tick_stop: Arc::clone(&force_tick_stop),
            },
            clock.clone(),
            NanoLiveModePolicy::autonomous_for_test(
                duration(1_000),
                Duration::from_secs(60),
                0.1,
                NonZeroU32::new(2).expect("frontier goal budget"),
            ),
        );
        let (handle, console_receiver) = operator_console(
            OperatorConsoleLimits::default(),
            OperatorConsoleSnapshot::unknown(ConsoleSnapshotRevision::parse(1).unwrap()),
        );
        let capability = ConsoleSessionCapability::from_bytes([0x5a; 32]);
        let session = handle
            .open_session(ConsoleSourceKind::Operator, capability, at(OWNER_NS))
            .expect("console session");
        let adapter =
            OperatorConsoleRuntimeAdapter::new(handle.clone(), console_receiver, typed_ingress);
        Fixture {
            raw_sender,
            owner,
            adapter,
            handle,
            session,
            capability,
            fail_next_zero,
            force_tick_stop,
            clock,
        }
    }

    fn anchor_and_admit_fully_free_map(fixture: &mut Fixture) {
        let geometry = OccupancyGridGeometry::try_new(0.25, [-2.0, -2.0], 20, 16, 320)
            .expect("bounded global map");
        let map = SlamMap::new().snapshot();
        let timestamp = Timestamp::from_nanos(100);
        let left = Frame::new(
            SensorId::StereoLeft,
            FrameId::new(1),
            timestamp,
            1,
            1,
            vec![0],
        )
        .expect("left anchor frame");
        let right = Frame::new(
            SensorId::StereoRight,
            FrameId::new(2),
            timestamp,
            1,
            1,
            vec![0],
        )
        .expect("right anchor frame");
        let pair = StereoPair::try_new(
            left,
            right,
            PairingWindowNs::try_from_u64(1).expect("nonzero pairing window"),
        )
        .expect("exact-time stereo pair");
        let observed_at = fixture.clock.peek();
        let observation = StereoObservation::parse(
            DeviceSessionId::try_new(7).expect("device session"),
            observed_at,
            pair,
        )
        .expect("typed stereo observation");
        let ingress = PendingVisualAttemptIngress::from_observation(
            NavigationClockEpoch::new(at(BASE_NS)),
            &observation,
        )
        .expect("observation after navigation epoch")
        .complete(VisualAttemptOutcome::LocalizationOnly);
        let localization = MapLocalization::new(
            VisualFrameStamp::new(FrameId::new(1), timestamp),
            map,
            WorldToCamera::identity(),
        );
        let admission = VisualAdmission::localization_only(ingress, localization)
            .expect("matching localization identity");
        assert!(matches!(
            fixture
                .owner
                .accept_visual(admission, observed_at)
                .expect("anchor admission"),
            VisualAdmissionOutcome::Reanchored(_)
        ));

        fixture.clock.advance(1);
        let cells = vec![OccupancyCell::Free; geometry.cell_count()];
        let snapshot =
            OccupancyGridSnapshot::from_test_cells(geometry, &cells, map.instance_id(), 1);
        fixture
            .owner
            .accept_global_map(fixture.clock.peek(), timestamp, &snapshot)
            .expect("exact retained map admission");
        fixture.clock.advance(1);
    }

    fn activate_manual(fixture: &mut Fixture) -> ConsoleDownstreamRequestId {
        let begin_id = submit(fixture, 1, OperatorConsoleIntent::BeginManual);
        fixture
            .adapter
            .drain_one_before_owner(&mut fixture.owner)
            .expect("queue begin-manual");
        let outcome = fixture
            .owner
            .process_one(AgentMapStateV1::UNAVAILABLE)
            .expect("admit manual authority");
        let event = fixture.owner.take_last_physical_state();
        assert!(matches!(
            fixture
                .adapter
                .complete_processed_owner_outcome(&mut fixture.owner, &outcome, event,),
            Ok(OperatorConsoleProcessDisposition::AuthorityActivated { .. })
        ));
        assert!(fixture.adapter.has_active_console_authority());
        begin_id
    }

    fn submit(
        fixture: &Fixture,
        sequence: u64,
        intent: OperatorConsoleIntent,
    ) -> ConsoleDownstreamRequestId {
        fixture
            .handle
            .submit(
                fixture.session,
                fixture.capability,
                ConsoleSourceSequence::parse(sequence).unwrap(),
                ConsoleIdempotencyKey::parse(sequence).unwrap(),
                intent,
                at(OWNER_NS + sequence),
            )
            .expect("console submission")
            .downstream_request_id()
    }

    fn raw_status_request() -> super::super::AgentControlRequestV1 {
        AgentControlRequestParser::new()
            .parse_next(
                &serde_json::to_vec(&json!({
                    "schema_version": AGENT_CONTROL_SCHEMA_V1,
                    "request_id": 900,
                    "command": {"kind": "query_status"}
                }))
                .expect("raw request JSON"),
            )
            .expect("raw request")
    }

    fn raw_stop_request() -> super::super::AgentControlRequestV1 {
        AgentControlRequestParser::new()
            .parse_next(
                &serde_json::to_vec(&json!({
                    "schema_version": AGENT_CONTROL_SCHEMA_V1,
                    "request_id": 901,
                    "command": {"kind": "stop"}
                }))
                .expect("raw request JSON"),
            )
            .expect("raw request")
    }

    fn raw_manual_stop_request(sequence: u64) -> super::super::AgentControlRequestV1 {
        AgentControlRequestParser::new()
            .parse_next(
                &serde_json::to_vec(&json!({
                    "schema_version": AGENT_CONTROL_SCHEMA_V1,
                    "request_id": 902,
                    "command": {"kind": "manual_stop", "sequence": sequence}
                }))
                .expect("raw request JSON"),
            )
            .expect("raw manual-stop request")
    }

    #[test]
    fn typed_begin_manual_forces_request_correlated_zero_after_recent_supervisor_zero() {
        let mut fixture = fixture(ready_supervisor_with_recent_zero(), 2);
        let begin_id = submit(&fixture, 1, OperatorConsoleIntent::BeginManual);
        fixture
            .adapter
            .drain_one_before_owner(&mut fixture.owner)
            .expect("queue begin-manual");

        let outcome = fixture
            .owner
            .process_one(AgentMapStateV1::UNAVAILABLE)
            .expect("admit manual authority");
        assert!(matches!(
            outcome,
            LiveMotionOwnerOutcome::Completed(
                LiveMotionCompletedSafetyAction::ManualStarted { .. }
            )
        ));
        let event = fixture
            .owner
            .take_last_physical_state()
            .expect("console-correlated admission zero");
        assert!(matches!(
            &event,
            LivePhysicalStateEvent::LifecycleZero(applied)
                if applied.reason() == LiveLifecycleZeroReason::ManualAdmission
                    && exact_safe_zero(applied.receipt())
        ));
        assert_eq!(
            fixture
                .adapter
                .correlated_downstream_request_id(&event)
                .expect("exact pending correlation"),
            Some(begin_id)
        );
        assert!(matches!(
            fixture.adapter.complete_processed_owner_outcome(
                &mut fixture.owner,
                &outcome,
                Some(event),
            ),
            Ok(OperatorConsoleProcessDisposition::AuthorityActivated {
                downstream_request_id,
            }) if downstream_request_id == begin_id
        ));
    }

    #[test]
    fn correlated_safety_stopped_manual_velocity_rejects_command_and_cancels_authority() {
        let mut fixture = fixture(ready_supervisor(), 2);
        let begin_id = activate_manual(&mut fixture);
        let command_id = submit(
            &fixture,
            2,
            OperatorConsoleIntent::ManualVelocity(
                super::super::FiniteManualVelocityV1::parse(0.1, 0.0)
                    .expect("finite manual target"),
            ),
        );
        fixture
            .adapter
            .drain_one_before_owner(&mut fixture.owner)
            .expect("queue manual velocity");
        fixture.force_tick_stop.store(true, Ordering::SeqCst);

        let outcome = fixture
            .owner
            .process_one(AgentMapStateV1::UNAVAILABLE)
            .expect("safety-stop manual command");
        assert!(matches!(outcome, LiveMotionOwnerOutcome::Rejected { .. }));
        let event = fixture
            .owner
            .take_last_physical_state()
            .expect("manual release zero");
        assert!(matches!(
            &event,
            LivePhysicalStateEvent::LifecycleZero(applied)
                if applied.reason() == LiveLifecycleZeroReason::ManualRelease
                    && exact_safe_zero(applied.receipt())
        ));
        assert!(matches!(
            fixture.adapter.complete_processed_owner_outcome(
                &mut fixture.owner,
                &outcome,
                Some(event),
            ),
            Ok(OperatorConsoleProcessDisposition::AuthorityCancelled {
                downstream_request_id,
            }) if downstream_request_id == begin_id
        ));
        assert!(!fixture.adapter.has_active_console_authority());
        assert_eq!(
            fixture
                .handle
                .response_record(command_id)
                .expect("manual command response")
                .state,
            ConsoleRuntimeResponseState::Rejected
        );
        assert_eq!(
            fixture
                .handle
                .response_record(begin_id)
                .expect("manual authority response")
                .state,
            ConsoleRuntimeResponseState::Cancelled
        );
    }

    #[test]
    fn raw_manual_stop_cancels_retained_console_manual_authority_after_exact_zero() {
        let mut fixture = fixture(ready_supervisor(), 2);
        let begin_id = activate_manual(&mut fixture);
        let raw_peer = enqueue_agent_control_test_request_through_runtime_lanes(
            &fixture.raw_sender,
            raw_manual_stop_request(1),
            fixture.clock.peek(),
        );

        let outcome = fixture
            .owner
            .process_one(AgentMapStateV1::UNAVAILABLE)
            .expect("raw manual stop");
        assert!(
            matches!(
                outcome,
                LiveMotionOwnerOutcome::Completed(LiveMotionCompletedSafetyAction::ManualStopped)
            ),
            "unexpected raw manual-stop outcome: {outcome:?}"
        );
        let event = fixture
            .owner
            .take_last_physical_state()
            .expect("raw manual release zero");
        assert!(matches!(
            &event,
            LivePhysicalStateEvent::LifecycleZero(applied)
                if applied.reason() == LiveLifecycleZeroReason::ManualRelease
                    && exact_safe_zero(applied.receipt())
        ));
        assert!(matches!(
            fixture.adapter.complete_processed_owner_outcome(
                &mut fixture.owner,
                &outcome,
                Some(event),
            ),
            Ok(OperatorConsoleProcessDisposition::AuthorityCancelled {
                downstream_request_id,
            }) if downstream_request_id == begin_id
        ));
        assert!(raw_peer.join().expect("raw peer").is_some());
        assert!(!fixture.adapter.has_active_console_authority());
    }

    #[test]
    fn raw_priority_stop_cancels_queued_console_acquisition_before_it_can_rearm() {
        let mut fixture = fixture(ready_supervisor(), 2);
        let begin_id = submit(&fixture, 1, OperatorConsoleIntent::BeginManual);
        fixture
            .adapter
            .drain_one_before_owner(&mut fixture.owner)
            .expect("queue console begin");
        let raw_peer = enqueue_agent_control_test_request_through_runtime_lanes(
            &fixture.raw_sender,
            raw_stop_request(),
            at(OWNER_NS + 20),
        );

        let stop_outcome = fixture
            .owner
            .process_one(AgentMapStateV1::UNAVAILABLE)
            .expect("priority raw stop");
        assert!(matches!(
            stop_outcome,
            LiveMotionOwnerOutcome::Completed(LiveMotionCompletedSafetyAction::GlobalStopped)
        ));
        let stop_event = fixture
            .owner
            .take_last_physical_state()
            .expect("raw global-stop zero");
        fixture
            .adapter
            .complete_processed_owner_outcome(&mut fixture.owner, &stop_outcome, Some(stop_event))
            .expect("cancel overtaken console acquisition");
        assert!(raw_peer.join().expect("raw peer").is_some());
        assert_eq!(
            fixture
                .handle
                .response_record(begin_id)
                .expect("cancelled console begin")
                .rejection_code,
            Some(ConsoleResponseRejectionCode::CancelledByPriorityStop)
        );

        let stale_begin = fixture
            .owner
            .process_one(AgentMapStateV1::UNAVAILABLE)
            .expect("dropped console begin reaches claim barrier");
        assert!(matches!(
            stale_begin,
            LiveMotionOwnerOutcome::ClientUnavailableBeforeClaim
        ));
        fixture
            .adapter
            .complete_processed_owner_outcome(&mut fixture.owner, &stale_begin, None)
            .expect("stale acquisition cannot execute");
        assert!(
            fixture
                .owner
                .dispatcher()
                .manual()
                .authority()
                .active_lease()
                .is_none()
        );
    }

    #[test]
    fn raw_priority_stop_also_purges_console_acquisition_cached_in_receiver() {
        let mut fixture = fixture(ready_supervisor(), 3);
        let arm_id = submit(&fixture, 1, OperatorConsoleIntent::Arm);
        let begin_id = submit(&fixture, 2, OperatorConsoleIntent::BeginManual);
        assert!(matches!(
            fixture
                .adapter
                .drain_one_before_owner(&mut fixture.owner)
                .expect("minimum console ID is submitted"),
            OperatorConsoleIngressDisposition::Submitted {
                downstream_request_id,
                command: AgentControlCommandKindV1::Arm,
            } if downstream_request_id == arm_id
        ));
        let raw_peer = enqueue_agent_control_test_request_through_runtime_lanes(
            &fixture.raw_sender,
            raw_stop_request(),
            fixture.clock.peek(),
        );

        let stop_outcome = fixture
            .owner
            .process_one(AgentMapStateV1::UNAVAILABLE)
            .expect("priority raw stop");
        let stop_event = fixture
            .owner
            .take_last_physical_state()
            .expect("raw global-stop zero");
        fixture
            .adapter
            .complete_processed_owner_outcome(&mut fixture.owner, &stop_outcome, Some(stop_event))
            .expect("purge submitted and receiver-cached acquisitions");
        assert!(raw_peer.join().expect("raw peer").is_some());

        for request_id in [arm_id, begin_id] {
            assert_eq!(
                fixture
                    .handle
                    .response_record(request_id)
                    .expect("priority-cancelled console response")
                    .rejection_code,
                Some(ConsoleResponseRejectionCode::CancelledByPriorityStop)
            );
        }
        assert!(fixture.handle.requested_owner().is_none());
        assert_eq!(fixture.adapter.pending_submission_count(), 0);
        assert_eq!(
            fixture
                .adapter
                .drain_one_before_owner(&mut fixture.owner)
                .expect("receiver cache was purged"),
            OperatorConsoleIngressDisposition::Idle
        );

        let stale_arm = fixture
            .owner
            .process_one(AgentMapStateV1::UNAVAILABLE)
            .expect("dropped arm reaches claim barrier");
        assert!(matches!(
            stale_arm,
            LiveMotionOwnerOutcome::ClientUnavailableBeforeClaim
        ));
        fixture
            .adapter
            .complete_processed_owner_outcome(&mut fixture.owner, &stale_arm, None)
            .expect("stale arm is unrelated");
        assert!(
            fixture
                .owner
                .dispatcher()
                .manual()
                .authority()
                .active_lease()
                .is_none()
        );
    }

    #[test]
    fn priority_map_only_cancels_queued_arm_without_latching_or_stale_rearm() {
        let mut fixture = fixture(disarmed_supervisor(), 4);
        let raw_peer = enqueue_agent_control_test_request(
            &fixture.raw_sender,
            raw_status_request(),
            at(OWNER_NS),
        );
        let arm_id = submit(&fixture, 1, OperatorConsoleIntent::Arm);
        assert!(matches!(
            fixture
                .adapter
                .drain_one_before_owner(&mut fixture.owner)
                .expect("queue arm"),
            OperatorConsoleIngressDisposition::Submitted {
                command: AgentControlCommandKindV1::Arm,
                ..
            }
        ));

        let map_only_id = submit(&fixture, 2, OperatorConsoleIntent::AutonomousMapOnly);
        assert!(matches!(
            fixture
                .adapter
                .drain_one_before_owner(&mut fixture.owner)
                .expect("queue map-only barrier"),
            OperatorConsoleIngressDisposition::Submitted {
                command: AgentControlCommandKindV1::MapOnly,
                ..
            }
        ));
        let cancelled = fixture
            .handle
            .response_record(arm_id)
            .expect("cancelled arm response");
        assert_eq!(cancelled.state, ConsoleRuntimeResponseState::Rejected);
        assert_eq!(
            cancelled.rejection_code,
            Some(ConsoleResponseRejectionCode::CancelledByPriorityStop)
        );
        assert!(
            !fixture.handle.software_safety_stop_latched(),
            "intentional priority cancellation is not an internal fault"
        );

        let map_outcome = fixture
            .owner
            .process_one(AgentMapStateV1::UNAVAILABLE)
            .expect("map-only safety barrier");
        assert!(matches!(
            map_outcome,
            LiveMotionOwnerOutcome::Completed(LiveMotionCompletedSafetyAction::MappingOnlyStopped)
        ));
        let map_event = fixture.owner.take_last_physical_state();
        fixture
            .adapter
            .complete_processed_owner_outcome(&mut fixture.owner, &map_outcome, map_event)
            .expect("complete exact map-only zero");
        assert_eq!(
            fixture
                .handle
                .response_record(map_only_id)
                .expect("map-only response")
                .state,
            ConsoleRuntimeResponseState::Completed
        );

        let raw_outcome = fixture
            .owner
            .process_one(AgentMapStateV1::UNAVAILABLE)
            .expect("raw status backlog");
        fixture
            .adapter
            .complete_processed_owner_outcome(&mut fixture.owner, &raw_outcome, None)
            .expect("reconcile raw status");
        assert!(raw_peer.join().expect("raw peer").is_some());

        let stale_arm = fixture
            .owner
            .process_one(AgentMapStateV1::UNAVAILABLE)
            .expect("cancelled arm reaches claim barrier");
        assert!(matches!(
            stale_arm,
            LiveMotionOwnerOutcome::ClientUnavailableBeforeClaim
        ));
        fixture
            .adapter
            .complete_processed_owner_outcome(&mut fixture.owner, &stale_arm, None)
            .expect("cancelled arm is unrelated after failed claim");
        assert_eq!(
            fixture
                .owner
                .dispatcher()
                .control_status(AgentMapStateV1::UNAVAILABLE)
                .runtime(),
            AgentRuntimeStateV1::Disarmed,
            "the cancelled Arm never re-arms after MapOnly"
        );
    }

    #[test]
    fn dispatcher_rejection_retains_exact_typed_correlation_and_terminalizes_token() {
        let mut fixture = fixture(disarmed_supervisor(), 2);
        let begin_id = submit(&fixture, 1, OperatorConsoleIntent::BeginManual);
        assert!(matches!(
            fixture
                .adapter
                .drain_one_before_owner(&mut fixture.owner)
                .expect("queue begin-manual"),
            OperatorConsoleIngressDisposition::Submitted {
                command: AgentControlCommandKindV1::BeginManual,
                ..
            }
        ));

        let outcome = fixture
            .owner
            .process_one(AgentMapStateV1::UNAVAILABLE)
            .expect("truthful dispatcher rejection");
        assert!(matches!(outcome, LiveMotionOwnerOutcome::Rejected { .. }));
        let event = fixture.owner.take_last_physical_state();
        assert!(matches!(
            fixture.adapter.complete_processed_owner_outcome(
                &mut fixture.owner,
                &outcome,
                event,
            ),
            Ok(OperatorConsoleProcessDisposition::ResponseRejected {
                downstream_request_id,
            }) if downstream_request_id == begin_id
        ));
        assert_eq!(fixture.adapter.pending_submission_count(), 0);
        assert_eq!(
            fixture
                .handle
                .response_record(begin_id)
                .expect("terminal begin-manual response")
                .state,
            ConsoleRuntimeResponseState::Rejected
        );
        assert!(
            !fixture.handle.software_safety_stop_latched(),
            "a truthful ordinary rejection is not an internal fault"
        );
    }

    #[test]
    fn exploration_without_reachable_frontier_completes_without_fake_authority_or_receipt() {
        let mut fixture = fixture(ready_supervisor(), 2);
        anchor_and_admit_fully_free_map(&mut fixture);
        let explore_id = submit(
            &fixture,
            1,
            OperatorConsoleIntent::AutonomousFrontierExplore,
        );
        fixture
            .adapter
            .drain_one_before_owner(&mut fixture.owner)
            .expect("queue frontier exploration");
        let outcome = fixture
            .owner
            .process_one(AgentMapStateV1::UNAVAILABLE)
            .expect("complete empty frontier search");
        assert!(matches!(
            outcome,
            LiveMotionOwnerOutcome::AutonomousCompleted {
                mode: AgentAutonomousMode::Explore,
            }
        ));
        assert!(
            fixture.owner.take_last_physical_state().is_none(),
            "no authority was acquired and no command was physically applied"
        );
        assert!(matches!(
            fixture.adapter.complete_processed_owner_outcome(
                &mut fixture.owner,
                &outcome,
                None,
            ),
            Ok(OperatorConsoleProcessDisposition::ResponseCompleted {
                downstream_request_id,
            }) if downstream_request_id == explore_id
        ));
        assert!(!fixture.adapter.has_active_console_authority());
        let response = fixture
            .handle
            .response_record(explore_id)
            .expect("completed exploration response");
        assert_eq!(response.state, ConsoleRuntimeResponseState::Completed);
        assert!(
            !response.applied,
            "a no-frontier result must not invent a physical receipt"
        );
        assert!(
            fixture
                .owner
                .dispatcher()
                .manual()
                .authority()
                .active_lease()
                .is_none()
        );
        assert!(
            fixture
                .handle
                .submit(
                    fixture.session,
                    fixture.capability,
                    ConsoleSourceSequence::parse(2).unwrap(),
                    ConsoleIdempotencyKey::parse(2).unwrap(),
                    OperatorConsoleIntent::BeginManual,
                    at(OWNER_NS + 2),
                )
                .is_ok(),
            "the exact autonomous owner generation was released"
        );
    }

    #[test]
    fn safety_reduction_backpressure_escalates_to_direct_emergency_stop() {
        let mut fixture = fixture(ready_supervisor(), 1);
        let priority_blocker = fixture
            .adapter
            .ingress
            .try_submit(raw_stop_request(), at(OWNER_NS))
            .expect("fill priority runtime lane");
        let map_only_id = submit(&fixture, 1, OperatorConsoleIntent::AutonomousMapOnly);
        assert!(matches!(
            fixture
                .adapter
                .drain_one_before_owner(&mut fixture.owner)
                .expect("reject saturated safety-reducing command"),
            OperatorConsoleIngressDisposition::RejectedBackpressure {
                downstream_request_id,
                command: AgentControlCommandKindV1::MapOnly,
            } if downstream_request_id == map_only_id
        ));
        assert!(fixture.handle.software_safety_stop_latched());
        assert_eq!(
            fixture
                .handle
                .response_record(map_only_id)
                .expect("map-only backpressure response")
                .rejection_code,
            Some(ConsoleResponseRejectionCode::RuntimeRejected)
        );
        drop(priority_blocker);

        assert!(matches!(
            fixture
                .adapter
                .drain_one_before_owner(&mut fixture.owner)
                .expect("execute fail-closed stop outside saturated lane"),
            OperatorConsoleIngressDisposition::SoftwareEmergencyStopApplied
        ));
        let event = fixture
            .owner
            .take_last_physical_state()
            .expect("direct emergency zero");
        assert!(matches!(
            &event,
            LivePhysicalStateEvent::LifecycleZero(applied)
                if applied.reason() == LiveLifecycleZeroReason::SoftwareEmergencyStop
                    && exact_safe_zero(applied.receipt())
        ));
        fixture
            .adapter
            .complete_software_emergency_stop()
            .expect("complete exact fail-closed stop");

        let stale_stop = fixture
            .owner
            .process_one(AgentMapStateV1::UNAVAILABLE)
            .expect("disconnected blocker reaches claim barrier");
        assert!(matches!(
            stale_stop,
            LiveMotionOwnerOutcome::ClientUnavailableBeforeClaim
        ));
        fixture
            .adapter
            .complete_processed_owner_outcome(&mut fixture.owner, &stale_stop, None)
            .expect("stale stop cannot execute after direct emergency");
    }

    #[test]
    fn direct_emergency_stop_preempts_deferred_map_persistence() {
        let mut fixture = fixture(ready_supervisor(), 2);
        let save_id = submit(&fixture, 1, OperatorConsoleIntent::SaveMap);
        fixture
            .adapter
            .drain_one_before_owner(&mut fixture.owner)
            .expect("queue save-map");
        let save_outcome = fixture
            .owner
            .process_one(AgentMapStateV1::UNAVAILABLE)
            .expect("claim outer save-map action");
        assert!(matches!(
            fixture.adapter.complete_processed_owner_outcome(
                &mut fixture.owner,
                &save_outcome,
                None,
            ),
            Ok(
                OperatorConsoleProcessDisposition::SaveMapPersistenceRequired {
                    downstream_request_id,
                }
            ) if downstream_request_id == save_id
        ));

        let emergency_id = submit(&fixture, 2, OperatorConsoleIntent::SoftwareSafetyStop);
        assert!(matches!(
            fixture
                .adapter
                .drain_one_before_owner(&mut fixture.owner)
                .expect("emergency bypasses deferred persistence"),
            OperatorConsoleIngressDisposition::SoftwareEmergencyStopApplied
        ));
        assert_eq!(
            fixture
                .handle
                .response_record(save_id)
                .expect("cancelled save-map response")
                .rejection_code,
            Some(ConsoleResponseRejectionCode::CancelledBySoftwareSafetyStop)
        );
        let event = fixture
            .owner
            .take_last_physical_state()
            .expect("direct emergency zero");
        assert!(matches!(
            &event,
            LivePhysicalStateEvent::LifecycleZero(applied)
                if applied.reason() == LiveLifecycleZeroReason::SoftwareEmergencyStop
                    && exact_safe_zero(applied.receipt())
        ));
        fixture
            .adapter
            .complete_software_emergency_stop()
            .expect("complete emergency while save-map responder is abandoned");
        assert_eq!(
            fixture
                .handle
                .response_record(emergency_id)
                .expect("emergency response")
                .state,
            ConsoleRuntimeResponseState::Completed
        );
        drop(save_outcome);
    }

    #[test]
    fn direct_software_stop_bypasses_full_runtime_queue_and_completes_exactly() {
        let mut fixture = fixture(ready_supervisor(), 1);
        let arm_id = submit(&fixture, 1, OperatorConsoleIntent::Arm);
        fixture
            .adapter
            .drain_one_before_owner(&mut fixture.owner)
            .expect("fill typed runtime queue");
        assert_eq!(fixture.adapter.pending_submission_count(), 1);

        let emergency_id = submit(&fixture, 2, OperatorConsoleIntent::SoftwareSafetyStop);
        assert!(matches!(
            fixture
                .adapter
                .drain_one_before_owner(&mut fixture.owner)
                .expect("direct emergency stop"),
            OperatorConsoleIngressDisposition::SoftwareEmergencyStopApplied
        ));
        assert_eq!(fixture.adapter.pending_submission_count(), 0);
        let event = fixture
            .owner
            .take_last_physical_state()
            .expect("emergency lifecycle zero");
        assert!(matches!(
            &event,
            LivePhysicalStateEvent::LifecycleZero(applied)
                if applied.reason() == LiveLifecycleZeroReason::SoftwareEmergencyStop
                    && exact_safe_zero(applied.receipt())
        ));
        fixture
            .adapter
            .complete_software_emergency_stop()
            .expect("exact emergency completion");

        assert_eq!(
            fixture
                .handle
                .response_record(arm_id)
                .expect("cancelled queued arm")
                .rejection_code,
            Some(ConsoleResponseRejectionCode::CancelledBySoftwareSafetyStop)
        );
        let emergency = fixture
            .handle
            .response_record(emergency_id)
            .expect("emergency response");
        assert_eq!(emergency.state, ConsoleRuntimeResponseState::Completed);
        assert!(emergency.applied);
        assert_eq!(
            fixture
                .owner
                .dispatcher()
                .control_status(AgentMapStateV1::UNAVAILABLE)
                .runtime(),
            AgentRuntimeStateV1::Faulted
        );
    }

    #[test]
    fn failed_emergency_zero_rejects_authority_instead_of_claiming_safe_cancellation() {
        let mut fixture = fixture(ready_supervisor(), 2);
        let begin_id = activate_manual(&mut fixture);
        fixture.fail_next_zero.store(true, Ordering::SeqCst);
        let emergency_id = submit(&fixture, 2, OperatorConsoleIntent::SoftwareSafetyStop);

        assert!(matches!(
            fixture.adapter.drain_one_before_owner(&mut fixture.owner),
            Err(OperatorConsoleRuntimeIngressError::EmergencyStop(_))
        ));
        assert!(!fixture.adapter.has_active_console_authority());
        let authority = fixture
            .handle
            .response_record(begin_id)
            .expect("manual authority terminal state");
        assert_eq!(authority.state, ConsoleRuntimeResponseState::Rejected);
        assert_eq!(
            authority.rejection_code,
            Some(ConsoleResponseRejectionCode::InternalFault)
        );
        let emergency = fixture
            .handle
            .response_record(emergency_id)
            .expect("failed emergency response");
        assert_eq!(emergency.state, ConsoleRuntimeResponseState::Rejected);
        assert_eq!(
            emergency.rejection_code,
            Some(ConsoleResponseRejectionCode::InternalFault)
        );
        assert!(fixture.handle.software_safety_stop_latched());
        assert!(matches!(
            fixture.owner.take_last_physical_state(),
            Some(LivePhysicalStateEvent::ActuationFault { .. })
        ));
    }

    #[test]
    fn manual_release_truthfully_uses_conservative_global_stop_contract() {
        let mut fixture = fixture(ready_supervisor(), 2);
        let begin_id = activate_manual(&mut fixture);
        let release_id = submit(&fixture, 2, OperatorConsoleIntent::ReleaseManual);
        fixture
            .adapter
            .drain_one_before_owner(&mut fixture.owner)
            .expect("queue conservative manual release");
        let outcome = fixture
            .owner
            .process_one(AgentMapStateV1::UNAVAILABLE)
            .expect("apply global stop for manual release");
        assert!(matches!(
            outcome,
            LiveMotionOwnerOutcome::Completed(LiveMotionCompletedSafetyAction::GlobalStopped)
        ));
        let event = fixture
            .owner
            .take_last_physical_state()
            .expect("global stop zero");
        assert!(matches!(
            &event,
            LivePhysicalStateEvent::LifecycleZero(applied)
                if applied.reason() == LiveLifecycleZeroReason::GlobalStopRequest
                    && exact_safe_zero(applied.receipt())
        ));
        assert!(matches!(
            fixture.adapter.complete_processed_owner_outcome(
                &mut fixture.owner,
                &outcome,
                Some(event),
            ),
            Ok(OperatorConsoleProcessDisposition::AuthorityCancelled {
                downstream_request_id,
            }) if downstream_request_id == release_id
        ));
        assert_eq!(
            fixture
                .handle
                .response_record(begin_id)
                .expect("manual authority response")
                .state,
            ConsoleRuntimeResponseState::Cancelled
        );
        assert_eq!(
            fixture
                .handle
                .response_record(release_id)
                .expect("release response")
                .state,
            ConsoleRuntimeResponseState::Completed
        );
        assert!(!fixture.handle.software_safety_stop_latched());
    }

    #[test]
    fn browser_deadman_truthfully_uses_conservative_global_stop_contract() {
        let mut fixture = fixture(ready_supervisor(), 2);
        let begin_id = activate_manual(&mut fixture);
        assert!(
            fixture
                .handle
                .tick_deadman(at(OWNER_NS + 1_000_000_000))
                .expect("advance browser deadman")
        );
        let deadman_id = match fixture
            .adapter
            .drain_one_before_owner(&mut fixture.owner)
            .expect("queue deadman global stop")
        {
            OperatorConsoleIngressDisposition::Submitted {
                downstream_request_id,
                command: AgentControlCommandKindV1::Stop,
            } => downstream_request_id,
            other => panic!("unexpected deadman ingress disposition: {other:?}"),
        };
        let outcome = fixture
            .owner
            .process_one(AgentMapStateV1::UNAVAILABLE)
            .expect("apply global stop for browser deadman");
        assert!(matches!(
            outcome,
            LiveMotionOwnerOutcome::Completed(LiveMotionCompletedSafetyAction::GlobalStopped)
        ));
        let event = fixture
            .owner
            .take_last_physical_state()
            .expect("deadman global stop zero");
        assert!(matches!(
            &event,
            LivePhysicalStateEvent::LifecycleZero(applied)
                if applied.reason() == LiveLifecycleZeroReason::GlobalStopRequest
                    && exact_safe_zero(applied.receipt())
        ));
        assert!(matches!(
            fixture.adapter.complete_processed_owner_outcome(
                &mut fixture.owner,
                &outcome,
                Some(event),
            ),
            Ok(OperatorConsoleProcessDisposition::AuthorityCancelled {
                downstream_request_id,
            }) if downstream_request_id == deadman_id
        ));
        assert_eq!(
            fixture
                .handle
                .response_record(begin_id)
                .expect("manual authority response")
                .state,
            ConsoleRuntimeResponseState::Cancelled
        );
        assert_eq!(
            fixture
                .handle
                .response_record(deadman_id)
                .expect("deadman response")
                .state,
            ConsoleRuntimeResponseState::Completed
        );
        assert!(!fixture.handle.software_safety_stop_latched());
    }

    #[test]
    fn mismatched_physical_correlation_fails_closed() {
        let mut fixture = fixture(disarmed_supervisor(), 2);
        let arm_id = submit(&fixture, 1, OperatorConsoleIntent::Arm);
        fixture
            .adapter
            .drain_one_before_owner(&mut fixture.owner)
            .expect("queue arm");
        let outcome = fixture
            .owner
            .process_one(AgentMapStateV1::UNAVAILABLE)
            .expect("arm owner transition");
        let mut event = fixture
            .owner
            .take_last_physical_state()
            .expect("arm zero evidence");
        event.replace_typed_request_key_for_test(AgentControlTypedRequestKey::for_test(999));

        assert!(matches!(
            fixture.adapter.complete_processed_owner_outcome(
                &mut fixture.owner,
                &outcome,
                Some(event),
            ),
            Err(OperatorConsoleRuntimeAdapterError::VerifiedCompletion(
                ConsoleVerifiedCompletionError::TypedRequestKeyMismatch
            ))
        ));
        assert!(fixture.handle.software_safety_stop_latched());
        assert_eq!(
            fixture
                .handle
                .response_record(arm_id)
                .expect("failed arm response")
                .state,
            ConsoleRuntimeResponseState::Rejected
        );
    }

    #[test]
    fn shutdown_cancels_authority_only_after_confirmed_controller_stop() {
        let mut fixture = fixture(ready_supervisor(), 2);
        let begin_id = activate_manual(&mut fixture);
        let terminal = fixture.owner.shutdown();
        let shutdown = fixture.adapter.shutdown(&terminal);

        assert!(shutdown.controller_stop_confirmed);
        assert!(!shutdown.lifecycle_cleanup_failed);
        assert!(matches!(
            shutdown.authority,
            OperatorConsoleShutdownAuthorityState::CancelledAfterConfirmedControllerStop {
                downstream_request_id
            } if downstream_request_id == begin_id
        ));
        assert_eq!(
            fixture
                .handle
                .response_record(begin_id)
                .expect("manual authority response")
                .state,
            ConsoleRuntimeResponseState::Cancelled
        );
        assert!(!fixture.handle.software_safety_stop_latched());
    }

    #[test]
    fn shutdown_cancels_authority_after_failed_disarm_with_confirmed_recovery_stop() {
        let mut fixture = fixture_with_terminal_stop_evidence(ready_supervisor(), 2, true, true);
        let begin_id = activate_manual(&mut fixture);
        let terminal = fixture.owner.shutdown();
        assert!(matches!(
            terminal.controller_stop(),
            LiveMotionTerminalStop::DisarmFailedStopConfirmed(_)
        ));

        let shutdown = fixture.adapter.shutdown(&terminal);

        assert!(shutdown.controller_stop_confirmed);
        assert!(!shutdown.lifecycle_cleanup_failed);
        assert!(matches!(
            shutdown.authority,
            OperatorConsoleShutdownAuthorityState::CancelledAfterConfirmedControllerStop {
                downstream_request_id
            } if downstream_request_id == begin_id
        ));
        assert_eq!(
            fixture
                .handle
                .response_record(begin_id)
                .expect("manual authority response")
                .state,
            ConsoleRuntimeResponseState::Cancelled
        );
        // Operation failure remains in the terminal report even though its
        // recovery evidence proved the physical stop.
        assert!(terminal.controller_stop().is_confirmed());
        assert!(!fixture.handle.software_safety_stop_latched());
    }

    #[test]
    fn shutdown_with_uncertain_controller_stop_rejects_authority_and_latches_fault() {
        let mut fixture = fixture_with_terminal_stop(ready_supervisor(), 2, true);
        let begin_id = activate_manual(&mut fixture);
        let terminal = fixture.owner.shutdown();
        let shutdown = fixture.adapter.shutdown(&terminal);

        assert!(!shutdown.controller_stop_confirmed);
        assert!(matches!(
            shutdown.authority,
            OperatorConsoleShutdownAuthorityState::RejectedBecauseControllerStopUncertain {
                downstream_request_id
            } if downstream_request_id == begin_id
        ));
        let record = fixture
            .handle
            .response_record(begin_id)
            .expect("manual authority response");
        assert_eq!(record.state, ConsoleRuntimeResponseState::Rejected);
        assert_eq!(
            record.rejection_code,
            Some(ConsoleResponseRejectionCode::InternalFault)
        );
        assert!(fixture.handle.software_safety_stop_latched());
    }
}
