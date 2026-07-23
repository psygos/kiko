//! Single-threaded live owner for the bounded wheels-off motion slice.
//!
//! This type is the only seam which simultaneously owns control dispatch,
//! supervisor-backed manual state, coordinator reference ownership, the
//! receipt-gated actuator, and the host clock used for lifecycle and control
//! ticks. It never reports completion before the exact controller receipt
//! required by that operation has been obtained.

use core::fmt;
use std::num::NonZeroU64;

use kiko_supervisor_core::{AuthorityLeaseId, AuthorityMode, FaultKind, SupervisorAction};
use robot_command_client::{AppliedCommandReceipt, DisarmReceipt};
use robot_protocol::v2::HostCommandResult;

use super::mpc::{HostMonotonicClock, HostMonotonicClockReadError};
use super::{
    AgentControlClaimedRequest, AgentControlCommandV1, AgentControlDispatchResponseError,
    AgentControlDispatcher, AgentControlDispatcherError, AgentControlRejectionCodeV1,
    AgentControlStatusV1, AgentControllerStopKnowledge, AgentDispatchOutcome,
    AgentLiveActuationDisposition, AgentLiveActuationFaultKind, AgentManualControlError,
    AgentManualGlobalStopRequirement, AgentMapStateV1, BeginManualTransition,
    CoordinatorAdmissionError, CoordinatorMotionModeError, CoordinatorMotionModeV1,
    CoordinatorTickError, DepthAdmissionOutcome, GlobalMapAdmissionOutcome, ImuAdmissionOutcome,
    LiveMpcControlDriver, LiveMpcControlError, ManualDriveAcceptedIntent,
    ManualDriveAcceptedTargetKindError, ManualDriveOutput, ManualMpcCommandError,
    NavigationIngressSink, ShadowNavigationCoordinator, VisualAdmission, VisualAdmissionOutcome,
    classify_live_actuation_error,
};
use crate::dense::occupancy::OccupancyGridSnapshot;
use crate::{DepthObservation, HostMonotonicTimestamp, ImuReport, Timestamp};

/// Receipt evidence required by lifecycle gates.
///
/// Implementations must return the exact result already checked by their
/// command client. A weak wire DTO must not implement this trait directly.
pub trait LiveMotionAppliedReceipt: receipt_sealed::Sealed {
    fn verified_host_result(&self) -> HostCommandResult;
}

mod receipt_sealed {
    pub trait Sealed {}

    impl Sealed for robot_command_client::AppliedCommandReceipt {}
}

impl LiveMotionAppliedReceipt for AppliedCommandReceipt {
    fn verified_host_result(&self) -> HostCommandResult {
        AppliedCommandReceipt::verified_host_result(self)
    }
}

/// Exact result of one coordinator/safety/application cycle.
pub struct LiveMotionApplied<R> {
    receipt: R,
    stopped: bool,
}

impl<R> LiveMotionApplied<R> {
    pub const fn receipt(&self) -> &R {
        &self.receipt
    }

    pub const fn stopped(&self) -> bool {
        self.stopped
    }

    pub fn into_receipt(self) -> R {
        self.receipt
    }
}

/// Tick failures preserve whether the failure happened before physical
/// application, in a typed manual conversion, or in the actuator itself.
#[derive(Debug)]
pub enum LiveMotionPortTickError<E> {
    Actuation(E),
    Coordinator(CoordinatorTickError),
    ManualCommand(ManualMpcCommandError),
    ManualStop(ManualDriveAcceptedTargetKindError),
}

impl<E: fmt::Display> fmt::Display for LiveMotionPortTickError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Actuation(source) => write!(formatter, "physical application failed: {source}"),
            Self::Coordinator(source) => source.fmt(formatter),
            Self::ManualCommand(source) => source.fmt(formatter),
            Self::ManualStop(source) => source.fmt(formatter),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for LiveMotionPortTickError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Actuation(source) => Some(source),
            Self::Coordinator(source) => Some(source),
            Self::ManualCommand(source) => Some(source),
            Self::ManualStop(source) => Some(source),
        }
    }
}

/// Exact fault classification retained from a failed physical operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LiveMotionActuationFaultEvidence {
    kind: AgentLiveActuationFaultKind,
    controller_stop: AgentControllerStopKnowledge,
}

impl LiveMotionActuationFaultEvidence {
    pub const fn new(
        kind: AgentLiveActuationFaultKind,
        controller_stop: AgentControllerStopKnowledge,
    ) -> Self {
        Self {
            kind,
            controller_stop,
        }
    }

    pub const fn kind(self) -> AgentLiveActuationFaultKind {
        self.kind
    }

    pub const fn supervisor_fault(self) -> FaultKind {
        self.kind.supervisor_fault()
    }

    pub const fn controller_stop(self) -> AgentControllerStopKnowledge {
        self.controller_stop
    }
}

/// Receipt-gated actuator used by the owner.
///
/// The fake implementation used by tests has the same shape as the physical
/// driver: a successful tick always includes a receipt, while physical errors
/// carry a stable fault and stop-knowledge classification.
pub trait LiveMotionActuationPort<J, C>
where
    J: NavigationIngressSink,
    C: HostMonotonicClock,
{
    type Receipt: LiveMotionAppliedReceipt;
    type Error: std::error::Error + 'static;

    fn apply_fresh_zero(&mut self) -> Result<Self::Receipt, Self::Error>;

    fn tick_manual(
        &mut self,
        coordinator: &mut ShadowNavigationCoordinator<J>,
        tick: HostMonotonicTimestamp,
        command: ManualDriveOutput<AuthorityLeaseId>,
        clock: &mut C,
    ) -> Result<LiveMotionApplied<Self::Receipt>, LiveMotionPortTickError<Self::Error>>;

    fn classify_error(source: &Self::Error) -> LiveMotionActuationFaultEvidence;
}

/// Terminal controller operation required before a live owner may be
/// dismantled.
///
/// A successful result is the controller client's exact disarm receipt. An
/// error is retained as uncertain stop evidence; callers must never translate
/// it into a successful shutdown merely because an earlier zero was applied.
pub trait LiveMotionTerminalActuationPort<J, C>: LiveMotionActuationPort<J, C>
where
    J: NavigationIngressSink,
    C: HostMonotonicClock,
{
    type StopEvidence;

    fn disarm(&mut self) -> Result<Self::StopEvidence, Self::Error>;
}

impl<J, C> LiveMotionActuationPort<J, C> for LiveMpcControlDriver
where
    J: NavigationIngressSink,
    C: HostMonotonicClock,
{
    type Receipt = AppliedCommandReceipt;
    type Error = super::actuation::LiveActuationError;

    fn apply_fresh_zero(&mut self) -> Result<Self::Receipt, Self::Error> {
        LiveMpcControlDriver::apply_fresh_zero(self)
    }

    fn tick_manual(
        &mut self,
        coordinator: &mut ShadowNavigationCoordinator<J>,
        tick: HostMonotonicTimestamp,
        command: ManualDriveOutput<AuthorityLeaseId>,
        clock: &mut C,
    ) -> Result<LiveMotionApplied<Self::Receipt>, LiveMotionPortTickError<Self::Error>> {
        let applied = LiveMpcControlDriver::tick_manual(self, coordinator, tick, command, clock)
            .map_err(|source| match source {
                LiveMpcControlError::Preflight(source) | LiveMpcControlError::Apply(source) => {
                    LiveMotionPortTickError::Actuation(source)
                }
                LiveMpcControlError::Coordinator(source) => {
                    LiveMotionPortTickError::Coordinator(source)
                }
                LiveMpcControlError::ManualCommand(source) => {
                    LiveMotionPortTickError::ManualCommand(source)
                }
                LiveMpcControlError::ManualStop(source) => {
                    LiveMotionPortTickError::ManualStop(source)
                }
            })?;
        let (outcome, receipt) = applied.into_parts();
        Ok(LiveMotionApplied {
            stopped: outcome.decision().record().pwm().is_stop(),
            receipt,
        })
    }

    fn classify_error(source: &Self::Error) -> LiveMotionActuationFaultEvidence {
        let AgentLiveActuationDisposition::LatchFault(fault) =
            classify_live_actuation_error(source);
        LiveMotionActuationFaultEvidence::new(fault.kind(), fault.controller_stop())
    }
}

impl<J, C> LiveMotionTerminalActuationPort<J, C> for LiveMpcControlDriver
where
    J: NavigationIngressSink,
    C: HostMonotonicClock,
{
    type StopEvidence = DisarmReceipt;

    fn disarm(&mut self) -> Result<Self::StopEvidence, Self::Error> {
        LiveMpcControlDriver::disarm(self)
    }
}

/// Result of latching the supervisor fault corresponding to a physical error.
#[derive(Debug)]
pub enum LiveMotionFaultLatch {
    Latched {
        fault: FaultKind,
        obligation: SupervisorAction,
    },
    Failed(AgentManualControlError),
}

/// A fully performed safety action retained when only client response delivery
/// failed. The state mutation and controller receipt are not rolled back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LiveMotionCompletedSafetyAction {
    Armed,
    ManualStarted { lease_id: AuthorityLeaseId },
    ManualCommandApplied,
    ManualStopped,
    MappingOnlyStopped,
    GlobalStopped,
    Disarmed,
    ShutdownStopped,
}

#[derive(Debug)]
pub enum LiveMotionOperationError<E> {
    Clock(HostMonotonicClockReadError),
    Manual(AgentManualControlError),
    CoordinatorMode(CoordinatorMotionModeError),
    Coordinator(CoordinatorTickError),
    ManualCommandInvariant(ManualMpcCommandError),
    ManualStopInvariant(ManualDriveAcceptedTargetKindError),
    ActuationFault {
        source: E,
        evidence: LiveMotionActuationFaultEvidence,
        latch: LiveMotionFaultLatch,
    },
    AppliedZeroClock {
        source: HostMonotonicClockReadError,
        result: HostCommandResult,
    },
    PrimaryAndCleanup {
        primary: Box<Self>,
        cleanup: Box<Self>,
    },
    CoordinatorDirectWithoutAuthority {
        mode: CoordinatorMotionModeV1,
    },
    ActiveAuthorityOutsideManualLifecycle {
        mode: AuthorityMode,
        latch: LiveMotionFaultLatch,
    },
}

impl<E: fmt::Debug> fmt::Display for LiveMotionOperationError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "live motion operation failed: {self:?}")
    }
}

impl<E: std::error::Error + 'static> std::error::Error for LiveMotionOperationError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Clock(source) => Some(source),
            Self::Manual(source) => Some(source),
            Self::CoordinatorMode(source) => Some(source),
            Self::Coordinator(source) => Some(source),
            Self::ManualCommandInvariant(source) => Some(source),
            Self::ManualStopInvariant(source) => Some(source),
            Self::ActuationFault { source, .. } => Some(source),
            Self::AppliedZeroClock { source, .. } => Some(source),
            Self::PrimaryAndCleanup { primary, .. } => Some(primary),
            Self::CoordinatorDirectWithoutAuthority { .. }
            | Self::ActiveAuthorityOutsideManualLifecycle { .. } => None,
        }
    }
}

impl<E> From<HostMonotonicClockReadError> for LiveMotionOperationError<E> {
    fn from(source: HostMonotonicClockReadError) -> Self {
        Self::Clock(source)
    }
}

#[derive(Debug)]
pub enum LiveMotionOwnerError<E> {
    Dispatch(Box<AgentControlDispatcherError>),
    Operation(LiveMotionOperationError<E>),
    OperationAndResponse {
        operation: LiveMotionOperationError<E>,
        response: AgentControlDispatchResponseError,
    },
    ResponseAfterSafety {
        safety: LiveMotionCompletedSafetyAction,
        response: AgentControlDispatchResponseError,
    },
}

impl<E: fmt::Debug> fmt::Display for LiveMotionOwnerError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "live motion owner failed: {self:?}")
    }
}

impl<E: std::error::Error + 'static> std::error::Error for LiveMotionOwnerError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Dispatch(source) => Some(source.as_ref()),
            Self::Operation(source) => Some(source),
            Self::OperationAndResponse { operation, .. } => Some(operation),
            Self::ResponseAfterSafety { response, .. } => Some(response),
        }
    }
}

#[derive(Debug)]
pub enum LiveMotionOwnerOutcome {
    Idle,
    ClientUnavailableBeforeClaim,
    StatusReplied(AgentControlStatusV1),
    Rejected {
        code: AgentControlRejectionCodeV1,
        retryable: bool,
    },
    Completed(LiveMotionCompletedSafetyAction),
    /// Persistence remains outside this bounded owner. The outer runtime must
    /// persist its preconfigured destination and consume this token with the
    /// truthful final response.
    SaveMapRequested {
        claimed: AgentControlClaimedRequest,
    },
    ShutdownRequested,
    PeriodicManualApplied,
    PeriodicManualStopped,
}

/// Exact result of the final controller-disarm attempt.
#[derive(Debug)]
pub enum LiveMotionTerminalStop<Receipt, Error> {
    Confirmed(Receipt),
    Uncertain(Error),
}

impl<Receipt, Error> LiveMotionTerminalStop<Receipt, Error> {
    pub const fn is_confirmed(&self) -> bool {
        matches!(self, Self::Confirmed(_))
    }
}

/// Consuming shutdown result for the sole live owner.
///
/// The coordinator is returned together with, never separately from, both
/// lifecycle-cleanup and controller-disarm evidence. This lets the outer
/// runtime finalize its journal without erasing an uncertain physical stop.
#[must_use = "terminal motion-owner evidence must be inspected before publishing the journal"]
pub struct LiveMotionOwnerTerminalReport<J, Receipt, Error>
where
    J: NavigationIngressSink,
{
    coordinator: ShadowNavigationCoordinator<J>,
    lifecycle_cleanup: Option<LiveMotionOperationError<Error>>,
    controller_stop: LiveMotionTerminalStop<Receipt, Error>,
}

impl<J, Receipt, Error> LiveMotionOwnerTerminalReport<J, Receipt, Error>
where
    J: NavigationIngressSink,
{
    pub const fn coordinator(&self) -> &ShadowNavigationCoordinator<J> {
        &self.coordinator
    }

    pub const fn lifecycle_cleanup(&self) -> Option<&LiveMotionOperationError<Error>> {
        self.lifecycle_cleanup.as_ref()
    }

    pub const fn controller_stop(&self) -> &LiveMotionTerminalStop<Receipt, Error> {
        &self.controller_stop
    }

    pub fn into_parts(
        self,
    ) -> (
        ShadowNavigationCoordinator<J>,
        Option<LiveMotionOperationError<Error>>,
        LiveMotionTerminalStop<Receipt, Error>,
    ) {
        (
            self.coordinator,
            self.lifecycle_cleanup,
            self.controller_stop,
        )
    }
}

/// Sole live owner for the currently implemented motion slice.
pub struct LiveMotionOwner<J, P, C>
where
    J: NavigationIngressSink,
    C: HostMonotonicClock,
    P: LiveMotionActuationPort<J, C>,
{
    dispatcher: AgentControlDispatcher,
    coordinator: ShadowNavigationCoordinator<J>,
    actuation: P,
    clock: C,
    last_actuation_fault: Option<LiveMotionActuationFaultEvidence>,
}

impl<J, P, C> LiveMotionOwner<J, P, C>
where
    J: NavigationIngressSink,
    C: HostMonotonicClock,
    P: LiveMotionActuationPort<J, C>,
{
    pub const fn new(
        dispatcher: AgentControlDispatcher,
        coordinator: ShadowNavigationCoordinator<J>,
        actuation: P,
        clock: C,
    ) -> Self {
        Self {
            dispatcher,
            coordinator,
            actuation,
            clock,
            last_actuation_fault: None,
        }
    }

    pub const fn coordinator(&self) -> &ShadowNavigationCoordinator<J> {
        &self.coordinator
    }

    pub const fn dispatcher(&self) -> &AgentControlDispatcher {
        &self.dispatcher
    }

    pub const fn actuation(&self) -> &P {
        &self.actuation
    }

    pub const fn last_actuation_fault(&self) -> Option<LiveMotionActuationFaultEvidence> {
        self.last_actuation_fault
    }

    /// Admit one already-typed visual observation without exposing a mutable
    /// coordinator or a second control-tick path.
    pub fn accept_visual(
        &mut self,
        admission: VisualAdmission,
        now: HostMonotonicTimestamp,
    ) -> Result<VisualAdmissionOutcome, CoordinatorAdmissionError<J::Error>> {
        self.coordinator.accept_visual(admission, now)
    }

    /// Admit one already-typed depth observation without exposing a mutable
    /// coordinator or a second control-tick path.
    pub fn accept_depth(
        &mut self,
        observation: DepthObservation,
        now: HostMonotonicTimestamp,
    ) -> Result<DepthAdmissionOutcome, CoordinatorAdmissionError<J::Error>> {
        self.coordinator.accept_depth(observation, now)
    }

    /// Admit one already-typed IMU report without exposing a mutable
    /// coordinator or a second control-tick path.
    pub fn accept_imu(
        &mut self,
        report: ImuReport,
        now: HostMonotonicTimestamp,
    ) -> Result<ImuAdmissionOutcome, CoordinatorAdmissionError<J::Error>> {
        self.coordinator.accept_imu(report, now)
    }

    /// Admit one immutable global-map snapshot without exposing a mutable
    /// coordinator or a second control-tick path.
    pub fn accept_global_map(
        &mut self,
        host_arrival: HostMonotonicTimestamp,
        source_capture_timestamp: Timestamp,
        snapshot: &OccupancyGridSnapshot,
    ) -> Result<GlobalMapAdmissionOutcome, CoordinatorAdmissionError<J::Error>> {
        self.coordinator
            .accept_global_map(host_arrival, source_capture_timestamp, snapshot)
    }

    /// Claim and finish at most one command.
    pub fn process_one(
        &mut self,
        map: AgentMapStateV1,
    ) -> Result<LiveMotionOwnerOutcome, LiveMotionOwnerError<P::Error>>
    where
        P::Error: fmt::Debug,
    {
        let observed_at = self.now().map_err(|source| {
            LiveMotionOwnerError::Operation(LiveMotionOperationError::Clock(source))
        })?;
        let outcome = self
            .dispatcher
            .try_dispatch_one_at(map, observed_at)
            .map_err(|source| LiveMotionOwnerError::Dispatch(Box::new(source)))?;
        match outcome {
            AgentDispatchOutcome::Idle => Ok(LiveMotionOwnerOutcome::Idle),
            AgentDispatchOutcome::ClientUnavailableBeforeClaim => {
                Ok(LiveMotionOwnerOutcome::ClientUnavailableBeforeClaim)
            }
            AgentDispatchOutcome::RepliedStatus { status } => {
                Ok(LiveMotionOwnerOutcome::StatusReplied(status))
            }
            AgentDispatchOutcome::RejectedManual { code, retryable } => {
                Ok(LiveMotionOwnerOutcome::Rejected { code, retryable })
            }
            AgentDispatchOutcome::ArmRequested { claimed } => self.arm(claimed),
            AgentDispatchOutcome::DisarmRequested { claimed, manual } => {
                self.disarm(claimed, manual)
            }
            AgentDispatchOutcome::BeginManual {
                claimed,
                transition,
            } => self.begin_manual(claimed, transition),
            AgentDispatchOutcome::ManualCommand { claimed, output } => {
                self.manual_command(claimed, output)
            }
            AgentDispatchOutcome::GlobalStopRequested { claimed, manual } => self.global_stop(
                claimed,
                manual,
                LiveMotionCompletedSafetyAction::GlobalStopped,
            ),
            AgentDispatchOutcome::Shutdown { claimed } => {
                let stopped =
                    self.stop_all_motion(self.dispatcher.manual().global_stop_requirement());
                match stopped {
                    Ok(()) => self.respond_completed(
                        claimed,
                        LiveMotionCompletedSafetyAction::ShutdownStopped,
                        LiveMotionOwnerOutcome::ShutdownRequested,
                    ),
                    Err(source) => self.reject_operation(
                        claimed,
                        source,
                        AgentControlRejectionCodeV1::InternalFault,
                        false,
                    ),
                }
            }
            AgentDispatchOutcome::OtherMode { claimed, command } => match command {
                AgentControlCommandV1::MapOnly => self.global_stop(
                    claimed,
                    self.dispatcher.manual().global_stop_requirement(),
                    LiveMotionCompletedSafetyAction::MappingOnlyStopped,
                ),
                AgentControlCommandV1::SaveMap => {
                    Ok(LiveMotionOwnerOutcome::SaveMapRequested { claimed })
                }
                AgentControlCommandV1::FrontierExplore
                | AgentControlCommandV1::SelectMapPoint(_) => self.reject_without_operation(
                    claimed,
                    AgentControlRejectionCodeV1::NotReady,
                    false,
                ),
                AgentControlCommandV1::QueryStatus
                | AgentControlCommandV1::Arm
                | AgentControlCommandV1::Disarm
                | AgentControlCommandV1::Stop
                | AgentControlCommandV1::BeginManual
                | AgentControlCommandV1::ManualVelocity(_)
                | AgentControlCommandV1::ManualStop(_)
                | AgentControlCommandV1::Shutdown => {
                    unreachable!("dispatcher classifies every implemented command")
                }
            },
            AgentDispatchOutcome::ManualFault { claimed, source } => {
                let operation =
                    self.stop_after_unadmitted_fault(LiveMotionOperationError::Manual(source));
                self.reject_operation(
                    claimed,
                    operation,
                    AgentControlRejectionCodeV1::InternalFault,
                    false,
                )
            }
        }
    }

    /// Advance the active manual lease and deadman even when no socket command
    /// is available.
    pub fn tick_manual(&mut self) -> Result<LiveMotionOwnerOutcome, LiveMotionOwnerError<P::Error>>
    where
        P::Error: fmt::Debug,
    {
        let Some(active_lease) = self.dispatcher.manual().active_lease() else {
            return Ok(LiveMotionOwnerOutcome::Idle);
        };
        let tick = self.now().map_err(|source| {
            LiveMotionOwnerError::Operation(LiveMotionOperationError::Clock(source))
        })?;
        let command = match self.dispatcher.manual_mut().tick(tick) {
            Ok(tick) => tick.output(),
            Err(source) => {
                self.leave_manual_mode_if_bound(active_lease.id());
                self.coordinator.clear_goal();
                let primary =
                    self.stop_after_unadmitted_fault(LiveMotionOperationError::Manual(source));
                return Err(LiveMotionOwnerError::Operation(primary));
            }
        };
        let applied =
            match self
                .actuation
                .tick_manual(&mut self.coordinator, tick, command, &mut self.clock)
            {
                Ok(applied) => applied,
                Err(source) => {
                    let primary = self.port_tick_error(source, tick);
                    let lease_id = manual_output_lease(command);
                    let primary = self.cleanup_after_tick_failure(primary, lease_id);
                    return Err(LiveMotionOwnerError::Operation(primary));
                }
            };
        if applied.stopped() || command.target().is_stop() {
            self.finish_manual_stop(manual_output_lease(command))
                .map_err(LiveMotionOwnerError::Operation)?;
            Ok(LiveMotionOwnerOutcome::PeriodicManualStopped)
        } else {
            Ok(LiveMotionOwnerOutcome::PeriodicManualApplied)
        }
    }

    /// Consume the sole owner through an ordered lifecycle stop, supervisor
    /// disarm, and controller-session disarm.
    ///
    /// The controller disarm is attempted even if lifecycle cleanup fails.
    /// Both failures are retained independently because a coordinator error
    /// does not erase physical stop uncertainty, and a confirmed controller
    /// stop does not erase an internal lifecycle fault.
    pub fn shutdown(mut self) -> LiveMotionOwnerTerminalReport<J, P::StopEvidence, P::Error>
    where
        P: LiveMotionTerminalActuationPort<J, C>,
    {
        let lifecycle_cleanup = self.shutdown_lifecycle().err();
        let controller_stop = match self.actuation.disarm() {
            Ok(receipt) => LiveMotionTerminalStop::Confirmed(receipt),
            Err(source) => LiveMotionTerminalStop::Uncertain(source),
        };
        LiveMotionOwnerTerminalReport {
            coordinator: self.coordinator,
            lifecycle_cleanup,
            controller_stop,
        }
    }

    fn shutdown_lifecycle(&mut self) -> Result<(), LiveMotionOperationError<P::Error>> {
        self.stop_all_motion(self.dispatcher.manual().global_stop_requirement())?;
        let requested_at = self.now()?;
        let action = self
            .dispatcher
            .begin_disarm(requested_at)
            .map_err(LiveMotionOperationError::Manual)?;
        if matches!(action, SupervisorAction::Disarmed) {
            return Ok(());
        }
        let (result, observed_at) = self.fresh_zero(requested_at)?;
        self.dispatcher
            .complete_disarm_with_applied_zero(result, observed_at, observed_at)
            .map_err(LiveMotionOperationError::Manual)?;
        Ok(())
    }

    fn arm(
        &mut self,
        claimed: AgentControlClaimedRequest,
    ) -> Result<LiveMotionOwnerOutcome, LiveMotionOwnerError<P::Error>> {
        let operation = (|| {
            let now = self.now()?;
            self.dispatcher
                .begin_arm(now)
                .map_err(LiveMotionOperationError::Manual)?;
            let (result, observed_at) = self.fresh_zero(now)?;
            self.dispatcher
                .complete_arm_with_applied_zero(result, observed_at, observed_at)
                .map_err(LiveMotionOperationError::Manual)?;
            Ok(())
        })();
        match operation {
            Ok(()) => self.respond_completed(
                claimed,
                LiveMotionCompletedSafetyAction::Armed,
                LiveMotionOwnerOutcome::Completed(LiveMotionCompletedSafetyAction::Armed),
            ),
            Err(source) => self.reject_operation(
                claimed,
                source,
                AgentControlRejectionCodeV1::InternalFault,
                false,
            ),
        }
    }

    fn begin_manual(
        &mut self,
        claimed: AgentControlClaimedRequest,
        transition: BeginManualTransition,
    ) -> Result<LiveMotionOwnerOutcome, LiveMotionOwnerError<P::Error>> {
        let operation = (|| {
            let lease = match transition {
                BeginManualTransition::Granted { lease } => lease,
                BeginManualTransition::FreshAppliedZeroRequired => {
                    let requested_at = self.now()?;
                    let (result, observed_at) = self.fresh_zero(requested_at)?;
                    self.dispatcher
                        .manual_mut()
                        .complete_begin_with_applied_zero(result, observed_at, observed_at)
                        .map_err(LiveMotionOperationError::Manual)?
                }
            };
            let lease_id = lease.id();
            self.coordinator
                .enter_manual_mode(nonzero_lease(lease_id))
                .map_err(LiveMotionOperationError::CoordinatorMode)?;
            Ok(lease_id)
        })();
        match operation {
            Ok(lease_id) => {
                let safety = LiveMotionCompletedSafetyAction::ManualStarted { lease_id };
                self.respond_completed(claimed, safety, LiveMotionOwnerOutcome::Completed(safety))
            }
            Err(source) => {
                let source = self.cleanup_after_begin_failure(source);
                self.reject_operation(
                    claimed,
                    source,
                    AgentControlRejectionCodeV1::ModeConflict,
                    false,
                )
            }
        }
    }

    fn manual_command(
        &mut self,
        claimed: AgentControlClaimedRequest,
        output: ManualDriveOutput<AuthorityLeaseId>,
    ) -> Result<LiveMotionOwnerOutcome, LiveMotionOwnerError<P::Error>> {
        let tick = match self.now() {
            Ok(tick) => tick,
            Err(source) => {
                return self.reject_operation(
                    claimed,
                    LiveMotionOperationError::Clock(source),
                    AgentControlRejectionCodeV1::InternalFault,
                    false,
                );
            }
        };
        let explicit_stop = matches!(
            output,
            ManualDriveOutput::Accepted(accepted)
                if accepted.intent() == ManualDriveAcceptedIntent::ExplicitStop
        );
        let applied =
            match self
                .actuation
                .tick_manual(&mut self.coordinator, tick, output, &mut self.clock)
            {
                Ok(applied) => applied,
                Err(source) => {
                    let operation = self.port_tick_error(source, tick);
                    let operation =
                        self.cleanup_after_tick_failure(operation, manual_output_lease(output));
                    return self.reject_operation(
                        claimed,
                        operation,
                        AgentControlRejectionCodeV1::SafetyStopped,
                        false,
                    );
                }
            };
        if applied.stopped() || output.target().is_stop() {
            let operation = self.finish_manual_stop(manual_output_lease(output));
            if let Err(source) = operation {
                return self.reject_operation(
                    claimed,
                    source,
                    AgentControlRejectionCodeV1::InternalFault,
                    false,
                );
            }
            let safety = LiveMotionCompletedSafetyAction::ManualStopped;
            if explicit_stop {
                self.respond_completed(claimed, safety, LiveMotionOwnerOutcome::Completed(safety))
            } else {
                self.reject_after_safety(
                    claimed,
                    safety,
                    AgentControlRejectionCodeV1::SafetyStopped,
                    true,
                )
            }
        } else {
            let safety = LiveMotionCompletedSafetyAction::ManualCommandApplied;
            self.respond_completed(claimed, safety, LiveMotionOwnerOutcome::Completed(safety))
        }
    }

    fn global_stop(
        &mut self,
        claimed: AgentControlClaimedRequest,
        manual: AgentManualGlobalStopRequirement,
        safety: LiveMotionCompletedSafetyAction,
    ) -> Result<LiveMotionOwnerOutcome, LiveMotionOwnerError<P::Error>> {
        match self.stop_all_motion(manual) {
            Ok(()) => {
                self.respond_completed(claimed, safety, LiveMotionOwnerOutcome::Completed(safety))
            }
            Err(source) => self.reject_operation(
                claimed,
                source,
                AgentControlRejectionCodeV1::InternalFault,
                false,
            ),
        }
    }

    fn disarm(
        &mut self,
        claimed: AgentControlClaimedRequest,
        manual: AgentManualGlobalStopRequirement,
    ) -> Result<LiveMotionOwnerOutcome, LiveMotionOwnerError<P::Error>> {
        let operation = (|| {
            self.stop_all_motion(manual)?;
            let requested_at = self.now()?;
            let action = self
                .dispatcher
                .begin_disarm(requested_at)
                .map_err(LiveMotionOperationError::Manual)?;
            if matches!(action, SupervisorAction::Disarmed) {
                return Ok(());
            }
            let (result, observed_at) = self.fresh_zero(requested_at)?;
            self.dispatcher
                .complete_disarm_with_applied_zero(result, observed_at, observed_at)
                .map_err(LiveMotionOperationError::Manual)?;
            Ok(())
        })();
        match operation {
            Ok(()) => self.respond_completed(
                claimed,
                LiveMotionCompletedSafetyAction::Disarmed,
                LiveMotionOwnerOutcome::Completed(LiveMotionCompletedSafetyAction::Disarmed),
            ),
            Err(source) => self.reject_operation(
                claimed,
                source,
                AgentControlRejectionCodeV1::InternalFault,
                false,
            ),
        }
    }

    fn stop_all_motion(
        &mut self,
        manual: AgentManualGlobalStopRequirement,
    ) -> Result<(), LiveMotionOperationError<P::Error>> {
        match manual {
            AgentManualGlobalStopRequirement::NoManualTransition => {
                if let Some(active) = self.dispatcher.manual().authority().active_lease() {
                    let fault_at = self.now()?;
                    let fault = FaultKind::InternalInvariant;
                    let latch = match self.dispatcher.manual_mut().latch_fault(fault, fault_at) {
                        Ok(obligation) => LiveMotionFaultLatch::Latched { fault, obligation },
                        Err(source) => LiveMotionFaultLatch::Failed(source),
                    };
                    if let CoordinatorMotionModeV1::Manual { authority_lease_id }
                    | CoordinatorMotionModeV1::FrontierInPlaceYaw { authority_lease_id } =
                        self.coordinator.motion_mode()
                    {
                        let _ = self.coordinator.leave_direct_mode(authority_lease_id);
                    }
                    self.coordinator.clear_goal();
                    let primary = LiveMotionOperationError::ActiveAuthorityOutsideManualLifecycle {
                        mode: active.mode(),
                        latch,
                    };
                    return match self.fresh_zero(fault_at) {
                        Ok(_) => Err(primary),
                        Err(cleanup) => Err(LiveMotionOperationError::PrimaryAndCleanup {
                            primary: Box::new(primary),
                            cleanup: Box::new(cleanup),
                        }),
                    };
                }
                let mode = self.coordinator.motion_mode();
                if let CoordinatorMotionModeV1::Manual { authority_lease_id }
                | CoordinatorMotionModeV1::FrontierInPlaceYaw { authority_lease_id } = mode
                {
                    self.coordinator
                        .leave_direct_mode(authority_lease_id)
                        .map_err(LiveMotionOperationError::CoordinatorMode)?;
                    self.coordinator.clear_goal();
                    let requested_at = self.now()?;
                    self.fresh_zero(requested_at)?;
                    return Err(
                        LiveMotionOperationError::CoordinatorDirectWithoutAuthority { mode },
                    );
                }
                self.coordinator.clear_goal();
                let requested_at = self.now()?;
                self.fresh_zero(requested_at)?;
                Ok(())
            }
            AgentManualGlobalStopRequirement::PendingBeginMustBeCancelled => {
                let requested_at = self.now()?;
                self.dispatcher
                    .manual_mut()
                    .cancel_pending_begin(requested_at)
                    .map_err(LiveMotionOperationError::Manual)?;
                self.coordinator.clear_goal();
                let (result, observed_at) = self.fresh_zero(requested_at)?;
                self.dispatcher
                    .manual_mut()
                    .complete_cancelled_begin_with_applied_zero(result, observed_at, observed_at)
                    .map_err(LiveMotionOperationError::Manual)
            }
            AgentManualGlobalStopRequirement::FreshAppliedCancelledBeginZeroRequired => {
                let requested_at = self.now()?;
                self.coordinator.clear_goal();
                let (result, observed_at) = self.fresh_zero(requested_at)?;
                self.dispatcher
                    .manual_mut()
                    .complete_cancelled_begin_with_applied_zero(result, observed_at, observed_at)
                    .map_err(LiveMotionOperationError::Manual)
            }
            AgentManualGlobalStopRequirement::ReleaseActive { lease_id } => {
                let requested_at = self.now()?;
                let actual = self
                    .dispatcher
                    .manual_mut()
                    .begin_release(requested_at)
                    .map_err(LiveMotionOperationError::Manual)?;
                if actual != lease_id {
                    return Err(LiveMotionOperationError::Manual(
                        AgentManualControlError::ModeConflict,
                    ));
                }
                self.leave_manual_mode_for_release(lease_id)?;
                self.coordinator.clear_goal();
                let (result, observed_at) = self.fresh_zero(requested_at)?;
                self.dispatcher
                    .manual_mut()
                    .complete_release_with_applied_zero(result, observed_at, observed_at)
                    .map(|_| ())
                    .map_err(LiveMotionOperationError::Manual)
            }
            AgentManualGlobalStopRequirement::FreshAppliedReleaseZeroRequired { lease_id } => {
                self.leave_manual_mode_for_release(lease_id)?;
                self.coordinator.clear_goal();
                let requested_at = self.now()?;
                let (result, observed_at) = self.fresh_zero(requested_at)?;
                self.dispatcher
                    .manual_mut()
                    .complete_release_with_applied_zero(result, observed_at, observed_at)
                    .map(|_| ())
                    .map_err(LiveMotionOperationError::Manual)
            }
        }
    }

    fn finish_manual_stop(
        &mut self,
        lease_id: AuthorityLeaseId,
    ) -> Result<(), LiveMotionOperationError<P::Error>> {
        let requirement = self.dispatcher.manual().global_stop_requirement();
        match requirement {
            AgentManualGlobalStopRequirement::ReleaseActive { lease_id: actual }
                if actual == lease_id => {}
            AgentManualGlobalStopRequirement::FreshAppliedReleaseZeroRequired {
                lease_id: actual,
            } if actual == lease_id => {}
            _ => {
                return Err(LiveMotionOperationError::Manual(
                    AgentManualControlError::ModeConflict,
                ));
            }
        }
        self.stop_all_motion(requirement)
    }

    fn cleanup_after_begin_failure(
        &mut self,
        primary: LiveMotionOperationError<P::Error>,
    ) -> LiveMotionOperationError<P::Error> {
        let requirement = self.dispatcher.manual().global_stop_requirement();
        if matches!(
            requirement,
            AgentManualGlobalStopRequirement::NoManualTransition
        ) {
            return primary;
        }
        match self.stop_all_motion(requirement) {
            Ok(()) => primary,
            Err(cleanup) => LiveMotionOperationError::PrimaryAndCleanup {
                primary: Box::new(primary),
                cleanup: Box::new(cleanup),
            },
        }
    }

    fn cleanup_after_tick_failure(
        &mut self,
        primary: LiveMotionOperationError<P::Error>,
        lease_id: AuthorityLeaseId,
    ) -> LiveMotionOperationError<P::Error> {
        if matches!(&primary, LiveMotionOperationError::ActuationFault { .. }) {
            self.leave_manual_mode_if_bound(lease_id);
            self.coordinator.clear_goal();
            return primary;
        }
        let requirement = self.dispatcher.manual().global_stop_requirement();
        match self.stop_all_motion(requirement) {
            Ok(()) => primary,
            Err(cleanup) => LiveMotionOperationError::PrimaryAndCleanup {
                primary: Box::new(primary),
                cleanup: Box::new(cleanup),
            },
        }
    }

    fn stop_after_unadmitted_fault(
        &mut self,
        primary: LiveMotionOperationError<P::Error>,
    ) -> LiveMotionOperationError<P::Error> {
        if let CoordinatorMotionModeV1::Manual { authority_lease_id }
        | CoordinatorMotionModeV1::FrontierInPlaceYaw { authority_lease_id } =
            self.coordinator.motion_mode()
        {
            let _ = self.coordinator.leave_direct_mode(authority_lease_id);
        }
        self.coordinator.clear_goal();
        let requested_at = match self.now() {
            Ok(now) => now,
            Err(source) => {
                return LiveMotionOperationError::PrimaryAndCleanup {
                    primary: Box::new(primary),
                    cleanup: Box::new(LiveMotionOperationError::Clock(source)),
                };
            }
        };
        match self.fresh_zero(requested_at) {
            Ok(_) => primary,
            Err(cleanup) => LiveMotionOperationError::PrimaryAndCleanup {
                primary: Box::new(primary),
                cleanup: Box::new(cleanup),
            },
        }
    }

    fn fresh_zero(
        &mut self,
        requested_at: HostMonotonicTimestamp,
    ) -> Result<(HostCommandResult, HostMonotonicTimestamp), LiveMotionOperationError<P::Error>>
    {
        let receipt = match self.actuation.apply_fresh_zero() {
            Ok(receipt) => receipt,
            Err(source) => return Err(self.latch_actuation_fault(source, requested_at)),
        };
        let result = receipt.verified_host_result();
        let observed_at = self
            .now()
            .map_err(|source| LiveMotionOperationError::AppliedZeroClock { source, result })?;
        Ok((result, observed_at))
    }

    fn port_tick_error(
        &mut self,
        source: LiveMotionPortTickError<P::Error>,
        fault_at: HostMonotonicTimestamp,
    ) -> LiveMotionOperationError<P::Error> {
        match source {
            LiveMotionPortTickError::Actuation(source) => {
                self.latch_actuation_fault(source, fault_at)
            }
            LiveMotionPortTickError::Coordinator(source) => {
                LiveMotionOperationError::Coordinator(source)
            }
            LiveMotionPortTickError::ManualCommand(source) => {
                LiveMotionOperationError::ManualCommandInvariant(source)
            }
            LiveMotionPortTickError::ManualStop(source) => {
                LiveMotionOperationError::ManualStopInvariant(source)
            }
        }
    }

    fn latch_actuation_fault(
        &mut self,
        source: P::Error,
        fault_at: HostMonotonicTimestamp,
    ) -> LiveMotionOperationError<P::Error> {
        let evidence = P::classify_error(&source);
        self.last_actuation_fault = Some(evidence);
        let fault = evidence.supervisor_fault();
        let latch = match self.dispatcher.manual_mut().latch_fault(fault, fault_at) {
            Ok(obligation) => LiveMotionFaultLatch::Latched { fault, obligation },
            Err(source) => LiveMotionFaultLatch::Failed(source),
        };
        LiveMotionOperationError::ActuationFault {
            source,
            evidence,
            latch,
        }
    }

    fn leave_manual_mode(
        &mut self,
        lease_id: AuthorityLeaseId,
    ) -> Result<(), LiveMotionOperationError<P::Error>> {
        self.coordinator
            .leave_direct_mode(nonzero_lease(lease_id))
            .map_err(LiveMotionOperationError::CoordinatorMode)
    }

    fn leave_manual_mode_for_release(
        &mut self,
        lease_id: AuthorityLeaseId,
    ) -> Result<(), LiveMotionOperationError<P::Error>> {
        match self.coordinator.motion_mode() {
            CoordinatorMotionModeV1::Manual { authority_lease_id }
                if authority_lease_id == nonzero_lease(lease_id) =>
            {
                self.leave_manual_mode(lease_id)
            }
            CoordinatorMotionModeV1::MappingOnly | CoordinatorMotionModeV1::PointGoal => {
                self.coordinator.clear_goal();
                Ok(())
            }
            actual => Err(LiveMotionOperationError::CoordinatorMode(
                CoordinatorMotionModeError::NotDirectControl { actual },
            )),
        }
    }

    fn leave_manual_mode_if_bound(&mut self, lease_id: AuthorityLeaseId) {
        if self.coordinator.motion_mode()
            == (CoordinatorMotionModeV1::Manual {
                authority_lease_id: nonzero_lease(lease_id),
            })
        {
            let _ = self.coordinator.leave_direct_mode(nonzero_lease(lease_id));
        }
    }

    fn now(&mut self) -> Result<HostMonotonicTimestamp, HostMonotonicClockReadError> {
        self.clock.try_now()
    }

    fn respond_completed(
        &self,
        claimed: AgentControlClaimedRequest,
        safety: LiveMotionCompletedSafetyAction,
        outcome: LiveMotionOwnerOutcome,
    ) -> Result<LiveMotionOwnerOutcome, LiveMotionOwnerError<P::Error>> {
        claimed
            .respond_completed()
            .map(|()| outcome)
            .map_err(|response| LiveMotionOwnerError::ResponseAfterSafety { safety, response })
    }

    fn reject_after_safety(
        &self,
        claimed: AgentControlClaimedRequest,
        safety: LiveMotionCompletedSafetyAction,
        code: AgentControlRejectionCodeV1,
        retryable: bool,
    ) -> Result<LiveMotionOwnerOutcome, LiveMotionOwnerError<P::Error>> {
        claimed
            .reject(code, retryable)
            .map(|()| LiveMotionOwnerOutcome::Rejected { code, retryable })
            .map_err(|response| LiveMotionOwnerError::ResponseAfterSafety { safety, response })
    }

    fn reject_without_operation(
        &self,
        claimed: AgentControlClaimedRequest,
        code: AgentControlRejectionCodeV1,
        retryable: bool,
    ) -> Result<LiveMotionOwnerOutcome, LiveMotionOwnerError<P::Error>> {
        claimed
            .reject(code, retryable)
            .map(|()| LiveMotionOwnerOutcome::Rejected { code, retryable })
            .map_err(|response| {
                LiveMotionOwnerError::Dispatch(Box::new(AgentControlDispatcherError::Response(
                    response,
                )))
            })
    }

    fn reject_operation(
        &self,
        claimed: AgentControlClaimedRequest,
        operation: LiveMotionOperationError<P::Error>,
        code: AgentControlRejectionCodeV1,
        retryable: bool,
    ) -> Result<LiveMotionOwnerOutcome, LiveMotionOwnerError<P::Error>> {
        match claimed.reject(code, retryable) {
            Ok(()) => Err(LiveMotionOwnerError::Operation(operation)),
            Err(response) => Err(LiveMotionOwnerError::OperationAndResponse {
                operation,
                response,
            }),
        }
    }
}

fn nonzero_lease(lease_id: AuthorityLeaseId) -> NonZeroU64 {
    NonZeroU64::new(lease_id.get()).expect("supervisor authority lease IDs are nonzero")
}

fn manual_output_lease(output: ManualDriveOutput<AuthorityLeaseId>) -> AuthorityLeaseId {
    match output {
        ManualDriveOutput::Accepted(accepted) => accepted.authority_lease_id(),
        ManualDriveOutput::Stopped(stopped) => stopped.bound_authority_lease_id(),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Instant;

    use kiko_supervisor_core::{
        AuthorityDuration, ReadinessBinding, ReadinessEpoch, Sha256Digest, SupervisorConfig,
    };
    use robot_protocol::ControllerUptimeMsWrapping;
    use robot_protocol::v2::{
        ControlEpoch, ControllerBootId, ControllerDeadlineMsWrapping, ControllerFaults,
        ControllerUid, HostCommandResultCode, OutputState, RemainingLeaseMs, TimerPwm,
        V2CommandSequence,
    };
    use serde_json::{Value, json};

    use super::super::control_socket::{
        enqueue_agent_control_test_request,
        enqueue_agent_control_test_request_with_expired_response,
    };
    use super::super::mpc::HostMonotonicClockReadError;
    use super::*;
    use crate::dense::occupancy::{DepthCameraModel, DepthToTrackingCamera};
    use crate::navigation::{
        AGENT_CONTROL_SCHEMA_V1, AgentAuthoritySupervisor, AgentControlCommandKindV1,
        AgentControlCompletionV1, AgentControlMonotonicOrigin, AgentControlRequestParser,
        AgentControlResponseKindV1, AgentControlRuntimeQueueCapacity, AgentControlRuntimeSender,
        AgentManualControlCore, AgentManualRuntimePolicy, MANUAL_DRIVE_CONFIG_V1,
        ManualDriveConfigV1, ManualDriveConfigV1Dto, NavigationClockEpoch,
        NavigationIngressCapacity, NavigationIngressLog, NavigationRecordingId,
        PathReferenceBuilderV1, PlanarOdometry, ShadowNavigationConfigV1, ShadowSafetySupervisor,
        agent_control_runtime_queue,
    };
    use crate::{DeviceSessionId, FrameDimensions, PinholeIntrinsics, Pose};

    const BASE_NS: u64 = 1_000_000_000;
    const OWNER_START_NS: u64 = BASE_NS + 1_000_000;

    fn at(nanos: u64) -> HostMonotonicTimestamp {
        HostMonotonicTimestamp::from_nanos(nanos)
    }

    fn duration(nanos: u64) -> AuthorityDuration {
        AuthorityDuration::try_from_nanos(nanos).expect("nonzero duration")
    }

    fn uid() -> ControllerUid {
        ControllerUid::try_new([1; 12]).expect("controller UID")
    }

    fn boot() -> ControllerBootId {
        ControllerBootId::try_new(7).expect("boot ID")
    }

    fn host_result(sequence: u32, stopped: bool) -> HostCommandResult {
        let pwm = if stopped {
            TimerPwm::ZERO
        } else {
            TimerPwm::try_new(10, 10).expect("bounded motion fixture")
        };
        HostCommandResult {
            controller_uid: uid(),
            boot_id: boot(),
            control_epoch: ControlEpoch::try_new(9).expect("control epoch"),
            sequence: V2CommandSequence::new(sequence),
            result: HostCommandResultCode::AppliedNew,
            requested_timer_pwm: pwm,
            controller_timer_pwm: pwm,
            output_state: if stopped {
                OutputState::ZeroPwm
            } else {
                OutputState::NonzeroPwm
            },
            controller_applied_at: ControllerUptimeMsWrapping::new(sequence),
            controller_expires_at: ControllerDeadlineMsWrapping::new(sequence.wrapping_add(10)),
            remaining_lease: if stopped {
                RemainingLeaseMs::ZERO
            } else {
                RemainingLeaseMs::try_new(10).expect("bounded remaining lease")
            },
            faults: ControllerFaults::NONE,
        }
    }

    fn disarmed() -> AgentAuthoritySupervisor {
        let config =
            SupervisorConfig::new(duration(10_000), duration(100)).expect("supervisor policy");
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
                    ControlEpoch::try_new(9).expect("control epoch"),
                    Sha256Digest::try_new([2; 32]).expect("hardware digest"),
                    Sha256Digest::try_new([3; 32]).expect("calibration digest"),
                ),
                at(BASE_NS + 2),
            )
            .expect("readiness");
        authority
    }

    fn ready() -> AgentAuthoritySupervisor {
        let mut authority = disarmed();
        authority.arm(at(BASE_NS + 3)).expect("arm transition");
        authority
            .admit_applied_zero(host_result(1, true), at(BASE_NS + 4), at(BASE_NS + 4))
            .expect("initial zero");
        authority
    }

    fn active_point_goal() -> AgentAuthoritySupervisor {
        let mut authority = ready();
        assert!(matches!(
            authority
                .request_mode(AuthorityMode::PointGoal, duration(1_000), at(BASE_NS + 5))
                .expect("point-goal authority fixture"),
            SupervisorAction::AuthorityGranted { .. }
        ));
        authority
    }

    fn policy() -> AgentManualRuntimePolicy {
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
    struct ScriptedClock {
        next_ns: Arc<AtomicU64>,
    }

    impl ScriptedClock {
        fn new(nanos: u64) -> Self {
            Self {
                next_ns: Arc::new(AtomicU64::new(nanos)),
            }
        }

        fn peek(&self) -> HostMonotonicTimestamp {
            at(self.next_ns.load(Ordering::SeqCst))
        }

        fn advance(&self, nanos: u64) {
            self.next_ns.fetch_add(nanos, Ordering::SeqCst);
        }
    }

    impl HostMonotonicClock for ScriptedClock {
        fn try_now(&mut self) -> Result<HostMonotonicTimestamp, HostMonotonicClockReadError> {
            Ok(at(self.next_ns.fetch_add(1, Ordering::SeqCst)))
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum FakeError {
        Injected,
    }

    impl fmt::Display for FakeError {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("injected fake actuation failure")
        }
    }

    impl std::error::Error for FakeError {}

    struct FakeReceipt(HostCommandResult);

    impl receipt_sealed::Sealed for FakeReceipt {}

    impl LiveMotionAppliedReceipt for FakeReceipt {
        fn verified_host_result(&self) -> HostCommandResult {
            self.0
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum FakeEvent {
        FreshZero,
        Disarm,
        ManualTick {
            requested_stop: bool,
            applied_stop: bool,
        },
    }

    #[derive(Clone)]
    struct FakePortState {
        events: Arc<Mutex<Vec<FakeEvent>>>,
        fail_next_zero: Arc<AtomicBool>,
        fail_next_tick: Arc<AtomicBool>,
        fail_disarm: Arc<AtomicBool>,
        force_tick_stop: Arc<AtomicBool>,
    }

    impl FakePortState {
        fn new() -> Self {
            Self {
                events: Arc::new(Mutex::new(Vec::new())),
                fail_next_zero: Arc::new(AtomicBool::new(false)),
                fail_next_tick: Arc::new(AtomicBool::new(false)),
                fail_disarm: Arc::new(AtomicBool::new(false)),
                force_tick_stop: Arc::new(AtomicBool::new(false)),
            }
        }

        fn events(&self) -> Vec<FakeEvent> {
            self.events.lock().expect("event lock").clone()
        }
    }

    struct FakePort {
        state: FakePortState,
        next_sequence: u32,
    }

    impl FakePort {
        fn new(state: FakePortState) -> Self {
            Self {
                state,
                next_sequence: 10,
            }
        }

        fn receipt(&mut self, stopped: bool) -> FakeReceipt {
            let sequence = self.next_sequence;
            self.next_sequence = self
                .next_sequence
                .checked_add(1)
                .expect("fixture sequence space");
            FakeReceipt(host_result(sequence, stopped))
        }
    }

    impl<J> LiveMotionActuationPort<J, ScriptedClock> for FakePort
    where
        J: NavigationIngressSink,
    {
        type Receipt = FakeReceipt;
        type Error = FakeError;

        fn apply_fresh_zero(&mut self) -> Result<Self::Receipt, Self::Error> {
            self.state
                .events
                .lock()
                .expect("event lock")
                .push(FakeEvent::FreshZero);
            if self.state.fail_next_zero.swap(false, Ordering::SeqCst) {
                Err(FakeError::Injected)
            } else {
                Ok(self.receipt(true))
            }
        }

        fn tick_manual(
            &mut self,
            _coordinator: &mut ShadowNavigationCoordinator<J>,
            _tick: HostMonotonicTimestamp,
            command: ManualDriveOutput<AuthorityLeaseId>,
            _clock: &mut ScriptedClock,
        ) -> Result<LiveMotionApplied<Self::Receipt>, LiveMotionPortTickError<Self::Error>>
        {
            if self.state.fail_next_tick.swap(false, Ordering::SeqCst) {
                return Err(LiveMotionPortTickError::Actuation(FakeError::Injected));
            }
            let requested_stop = command.target().is_stop();
            let applied_stop = requested_stop || self.state.force_tick_stop.load(Ordering::SeqCst);
            self.state
                .events
                .lock()
                .expect("event lock")
                .push(FakeEvent::ManualTick {
                    requested_stop,
                    applied_stop,
                });
            Ok(LiveMotionApplied {
                receipt: self.receipt(applied_stop),
                stopped: applied_stop,
            })
        }

        fn classify_error(_source: &Self::Error) -> LiveMotionActuationFaultEvidence {
            LiveMotionActuationFaultEvidence::new(
                AgentLiveActuationFaultKind::TransportUnavailable,
                AgentControllerStopKnowledge::Uncertain,
            )
        }
    }

    impl<J> LiveMotionTerminalActuationPort<J, ScriptedClock> for FakePort
    where
        J: NavigationIngressSink,
    {
        type StopEvidence = u32;

        fn disarm(&mut self) -> Result<Self::StopEvidence, Self::Error> {
            self.state
                .events
                .lock()
                .expect("event lock")
                .push(FakeEvent::Disarm);
            if self.state.fail_disarm.load(Ordering::SeqCst) {
                Err(FakeError::Injected)
            } else {
                Ok(self.next_sequence)
            }
        }
    }

    type Journal = NavigationIngressLog;
    type Owner = LiveMotionOwner<Journal, FakePort, ScriptedClock>;

    fn coordinator() -> ShadowNavigationCoordinator<Journal> {
        let dimensions = FrameDimensions::try_new(640, 400).expect("depth dimensions");
        let camera = DepthCameraModel::new(
            PinholeIntrinsics::try_new(400.0, 400.0, 320.0, 200.0).expect("depth intrinsics"),
            dimensions,
            DepthToTrackingCamera::new(Pose::identity()),
        );
        let parsed = ShadowNavigationConfigV1::parse_json(
            include_bytes!("../../../../configs/navigation-shadow-v1.example.json"),
            camera,
        )
        .expect("example navigation policy");
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

    fn fixture(
        authority: AgentAuthoritySupervisor,
    ) -> (
        AgentControlRuntimeSender,
        Owner,
        ScriptedClock,
        FakePortState,
    ) {
        let (sender, receiver) = agent_control_runtime_queue(
            AgentControlRuntimeQueueCapacity::try_new(8).expect("queue capacity"),
        );
        let control_clock = AgentControlMonotonicOrigin::new(Instant::now(), at(OWNER_START_NS));
        let dispatcher = AgentControlDispatcher::new(
            receiver,
            control_clock,
            AgentManualControlCore::new(authority, Some(policy())),
        );
        let clock = ScriptedClock::new(OWNER_START_NS);
        let state = FakePortState::new();
        let owner = LiveMotionOwner::new(
            dispatcher,
            coordinator(),
            FakePort::new(state.clone()),
            clock.clone(),
        );
        (sender, owner, clock, state)
    }

    fn request(
        parser: &mut AgentControlRequestParser,
        request_id: u64,
        command: Value,
    ) -> super::super::AgentControlRequestV1 {
        parser
            .parse_next(
                &serde_json::to_vec(&json!({
                    "schema_version": AGENT_CONTROL_SCHEMA_V1,
                    "request_id": request_id,
                    "command": command,
                }))
                .expect("request JSON"),
            )
            .expect("parsed request")
    }

    fn enqueue(
        sender: &AgentControlRuntimeSender,
        parser: &mut AgentControlRequestParser,
        request_id: u64,
        command: Value,
        received_at: HostMonotonicTimestamp,
    ) -> std::thread::JoinHandle<Option<super::super::AgentControlResponseV1>> {
        enqueue_agent_control_test_request(
            sender,
            request(parser, request_id, command),
            received_at,
        )
    }

    fn assert_completed(
        peer: std::thread::JoinHandle<Option<super::super::AgentControlResponseV1>>,
        command: AgentControlCommandKindV1,
    ) {
        assert!(matches!(
            peer.join().expect("response peer"),
            Some(response)
                if matches!(
                    response.response(),
                    AgentControlResponseKindV1::Accepted {
                        command: actual,
                        completion: AgentControlCompletionV1::Completed,
                    } if actual == command
                )
        ));
    }

    #[test]
    fn arm_completes_only_after_a_new_zero_receipt() {
        let (sender, mut owner, clock, state) = fixture(disarmed());
        let mut parser = AgentControlRequestParser::new();
        let peer = enqueue(
            &sender,
            &mut parser,
            1,
            json!({"kind": "arm"}),
            clock.peek(),
        );

        assert!(matches!(
            owner.process_one(AgentMapStateV1::UNAVAILABLE),
            Ok(LiveMotionOwnerOutcome::Completed(
                LiveMotionCompletedSafetyAction::Armed
            ))
        ));
        assert_eq!(state.events(), [FakeEvent::FreshZero]);
        assert_completed(peer, AgentControlCommandKindV1::Arm);
        assert!(matches!(
            owner
                .dispatcher()
                .control_status(AgentMapStateV1::UNAVAILABLE)
                .runtime(),
            super::super::AgentRuntimeStateV1::ReadyStopped
        ));
    }

    #[test]
    fn begin_velocity_and_explicit_stop_are_receipt_gated_and_release_manual() {
        let (sender, mut owner, clock, state) = fixture(ready());
        let mut parser = AgentControlRequestParser::new();

        let begin = enqueue(
            &sender,
            &mut parser,
            1,
            json!({"kind": "begin_manual"}),
            clock.peek(),
        );
        assert!(matches!(
            owner.process_one(AgentMapStateV1::UNAVAILABLE),
            Ok(LiveMotionOwnerOutcome::Completed(
                LiveMotionCompletedSafetyAction::ManualStarted { .. }
            ))
        ));
        assert_completed(begin, AgentControlCommandKindV1::BeginManual);

        let velocity = enqueue(
            &sender,
            &mut parser,
            2,
            json!({
                "kind": "manual_velocity",
                "sequence": 1,
                "forward_velocity_mps": 0.2,
                "yaw_rate_rad_s": 0.1
            }),
            clock.peek(),
        );
        assert!(matches!(
            owner.process_one(AgentMapStateV1::UNAVAILABLE),
            Ok(LiveMotionOwnerOutcome::Completed(
                LiveMotionCompletedSafetyAction::ManualCommandApplied
            ))
        ));
        assert_completed(velocity, AgentControlCommandKindV1::ManualVelocity);

        let stop = enqueue(
            &sender,
            &mut parser,
            3,
            json!({"kind": "manual_stop", "sequence": 2}),
            clock.peek(),
        );
        assert!(matches!(
            owner.process_one(AgentMapStateV1::UNAVAILABLE),
            Ok(LiveMotionOwnerOutcome::Completed(
                LiveMotionCompletedSafetyAction::ManualStopped
            ))
        ));
        assert_completed(stop, AgentControlCommandKindV1::ManualStop);
        assert_eq!(
            state.events(),
            [
                FakeEvent::FreshZero,
                FakeEvent::ManualTick {
                    requested_stop: false,
                    applied_stop: false,
                },
                FakeEvent::ManualTick {
                    requested_stop: true,
                    applied_stop: true,
                },
                FakeEvent::FreshZero,
            ]
        );
        assert_eq!(
            owner.coordinator().motion_mode(),
            CoordinatorMotionModeV1::MappingOnly
        );
        assert!(owner.dispatcher().manual().active_lease().is_none());
    }

    #[test]
    fn global_stop_cancels_a_pending_begin_before_admitting_its_zero() {
        let (sender, receiver) = agent_control_runtime_queue(
            AgentControlRuntimeQueueCapacity::try_new(4).expect("queue capacity"),
        );
        let mut manual = AgentManualControlCore::new(ready(), Some(policy()));
        assert_eq!(
            manual
                .begin_manual(at(OWNER_START_NS))
                .expect("pending begin"),
            BeginManualTransition::FreshAppliedZeroRequired
        );
        let clock = ScriptedClock::new(OWNER_START_NS + 1);
        let state = FakePortState::new();
        let mut owner = LiveMotionOwner::new(
            AgentControlDispatcher::new(
                receiver,
                AgentControlMonotonicOrigin::new(Instant::now(), at(OWNER_START_NS)),
                manual,
            ),
            coordinator(),
            FakePort::new(state.clone()),
            clock.clone(),
        );
        let mut parser = AgentControlRequestParser::new();
        let peer = enqueue(
            &sender,
            &mut parser,
            1,
            json!({"kind": "stop"}),
            clock.peek(),
        );

        assert!(matches!(
            owner.process_one(AgentMapStateV1::UNAVAILABLE),
            Ok(LiveMotionOwnerOutcome::Completed(
                LiveMotionCompletedSafetyAction::GlobalStopped
            ))
        ));
        assert_completed(peer, AgentControlCommandKindV1::Stop);
        assert_eq!(state.events(), [FakeEvent::FreshZero]);
        assert!(owner.dispatcher().manual().active_lease().is_none());
    }

    #[test]
    fn periodic_deadman_stop_is_applied_then_released_with_a_fresh_zero() {
        let (sender, mut owner, clock, state) = fixture(ready());
        let mut parser = AgentControlRequestParser::new();
        let begin = enqueue(
            &sender,
            &mut parser,
            1,
            json!({"kind": "begin_manual"}),
            clock.peek(),
        );
        owner
            .process_one(AgentMapStateV1::UNAVAILABLE)
            .expect("begin manual");
        assert_completed(begin, AgentControlCommandKindV1::BeginManual);
        let velocity = enqueue(
            &sender,
            &mut parser,
            2,
            json!({
                "kind": "manual_velocity",
                "sequence": 1,
                "forward_velocity_mps": 0.2,
                "yaw_rate_rad_s": 0.0
            }),
            clock.peek(),
        );
        owner
            .process_one(AgentMapStateV1::UNAVAILABLE)
            .expect("velocity");
        assert_completed(velocity, AgentControlCommandKindV1::ManualVelocity);

        clock.advance(60);
        assert!(matches!(
            owner.tick_manual(),
            Ok(LiveMotionOwnerOutcome::PeriodicManualStopped)
        ));
        assert!(state.events().ends_with(&[
            FakeEvent::ManualTick {
                requested_stop: true,
                applied_stop: true,
            },
            FakeEvent::FreshZero,
        ]));
        assert_eq!(
            owner.coordinator().motion_mode(),
            CoordinatorMotionModeV1::MappingOnly
        );
    }

    #[test]
    fn coordinator_safety_stop_releases_authority_before_rejecting_velocity() {
        let (sender, mut owner, clock, state) = fixture(ready());
        let mut parser = AgentControlRequestParser::new();
        let begin = enqueue(
            &sender,
            &mut parser,
            1,
            json!({"kind": "begin_manual"}),
            clock.peek(),
        );
        owner
            .process_one(AgentMapStateV1::UNAVAILABLE)
            .expect("begin manual");
        assert_completed(begin, AgentControlCommandKindV1::BeginManual);

        state.force_tick_stop.store(true, Ordering::SeqCst);
        let velocity = enqueue(
            &sender,
            &mut parser,
            2,
            json!({
                "kind": "manual_velocity",
                "sequence": 1,
                "forward_velocity_mps": 0.2,
                "yaw_rate_rad_s": 0.0
            }),
            clock.peek(),
        );
        assert!(matches!(
            owner.process_one(AgentMapStateV1::UNAVAILABLE),
            Ok(LiveMotionOwnerOutcome::Rejected {
                code: AgentControlRejectionCodeV1::SafetyStopped,
                retryable: true,
            })
        ));
        assert!(matches!(
            velocity.join().expect("velocity peer"),
            Some(response)
                if matches!(
                    response.response(),
                    AgentControlResponseKindV1::Rejected {
                        code: AgentControlRejectionCodeV1::SafetyStopped,
                        retryable: true,
                    }
                )
        ));
        assert!(state.events().ends_with(&[
            FakeEvent::ManualTick {
                requested_stop: false,
                applied_stop: true,
            },
            FakeEvent::FreshZero,
        ]));
        assert!(owner.dispatcher().manual().active_lease().is_none());
        assert_eq!(
            owner.coordinator().motion_mode(),
            CoordinatorMotionModeV1::MappingOnly
        );
    }

    #[test]
    fn physical_tick_failure_latches_exact_fault_and_retains_stop_uncertainty() {
        let (sender, mut owner, clock, state) = fixture(ready());
        let mut parser = AgentControlRequestParser::new();
        let begin = enqueue(
            &sender,
            &mut parser,
            1,
            json!({"kind": "begin_manual"}),
            clock.peek(),
        );
        owner
            .process_one(AgentMapStateV1::UNAVAILABLE)
            .expect("begin manual");
        assert_completed(begin, AgentControlCommandKindV1::BeginManual);

        state.fail_next_tick.store(true, Ordering::SeqCst);
        let velocity = enqueue(
            &sender,
            &mut parser,
            2,
            json!({
                "kind": "manual_velocity",
                "sequence": 1,
                "forward_velocity_mps": 0.2,
                "yaw_rate_rad_s": 0.0
            }),
            clock.peek(),
        );
        assert!(matches!(
            owner.process_one(AgentMapStateV1::UNAVAILABLE),
            Err(LiveMotionOwnerError::Operation(
                LiveMotionOperationError::ActuationFault {
                    evidence,
                    latch: LiveMotionFaultLatch::Latched {
                        fault: FaultKind::HardwareReadinessLost,
                        ..
                    },
                    ..
                }
            )) if evidence.controller_stop() == AgentControllerStopKnowledge::Uncertain
        ));
        assert!(matches!(
            velocity.join().expect("velocity peer"),
            Some(response)
                if matches!(
                    response.response(),
                    AgentControlResponseKindV1::Rejected {
                        code: AgentControlRejectionCodeV1::SafetyStopped,
                        retryable: false,
                    }
                )
        ));
        assert!(matches!(
            owner
                .dispatcher()
                .control_status(AgentMapStateV1::UNAVAILABLE)
                .runtime(),
            super::super::AgentRuntimeStateV1::Faulted
        ));
        assert_eq!(
            owner.coordinator().motion_mode(),
            CoordinatorMotionModeV1::MappingOnly
        );
    }

    #[test]
    fn disarm_and_shutdown_finish_their_stop_receipts_before_responding() {
        let (sender, mut owner, clock, state) = fixture(ready());
        let mut parser = AgentControlRequestParser::new();
        let disarm = enqueue(
            &sender,
            &mut parser,
            1,
            json!({"kind": "disarm"}),
            clock.peek(),
        );
        assert!(matches!(
            owner.process_one(AgentMapStateV1::UNAVAILABLE),
            Ok(LiveMotionOwnerOutcome::Completed(
                LiveMotionCompletedSafetyAction::Disarmed
            ))
        ));
        assert_completed(disarm, AgentControlCommandKindV1::Disarm);
        assert_eq!(state.events(), [FakeEvent::FreshZero, FakeEvent::FreshZero]);

        let (sender, mut owner, clock, state) = fixture(ready());
        let mut parser = AgentControlRequestParser::new();
        let shutdown = enqueue(
            &sender,
            &mut parser,
            1,
            json!({"kind": "shutdown"}),
            clock.peek(),
        );
        assert!(matches!(
            owner.process_one(AgentMapStateV1::UNAVAILABLE),
            Ok(LiveMotionOwnerOutcome::ShutdownRequested)
        ));
        assert_completed(shutdown, AgentControlCommandKindV1::Shutdown);
        assert_eq!(state.events(), [FakeEvent::FreshZero]);
    }

    #[test]
    fn response_drop_keeps_safety_effect_and_dual_failure_keeps_both_causes() {
        let (sender, mut owner, clock, state) = fixture(disarmed());
        let mut parser = AgentControlRequestParser::new();
        let peer = enqueue_agent_control_test_request_with_expired_response(
            &sender,
            request(&mut parser, 1, json!({"kind": "arm"})),
            clock.peek(),
        );
        assert!(matches!(
            owner.process_one(AgentMapStateV1::UNAVAILABLE),
            Err(LiveMotionOwnerError::ResponseAfterSafety {
                safety: LiveMotionCompletedSafetyAction::Armed,
                response: AgentControlDispatchResponseError::ClientUnavailable,
            })
        ));
        peer.join().expect("expired peer");
        assert_eq!(state.events(), [FakeEvent::FreshZero]);
        assert!(matches!(
            owner
                .dispatcher()
                .control_status(AgentMapStateV1::UNAVAILABLE)
                .runtime(),
            super::super::AgentRuntimeStateV1::ReadyStopped
        ));

        let (sender, mut owner, clock, state) = fixture(disarmed());
        state.fail_next_zero.store(true, Ordering::SeqCst);
        let mut parser = AgentControlRequestParser::new();
        let peer = enqueue_agent_control_test_request_with_expired_response(
            &sender,
            request(&mut parser, 1, json!({"kind": "arm"})),
            clock.peek(),
        );
        assert!(matches!(
            owner.process_one(AgentMapStateV1::UNAVAILABLE),
            Err(LiveMotionOwnerError::OperationAndResponse {
                operation: LiveMotionOperationError::ActuationFault {
                    evidence,
                    latch: LiveMotionFaultLatch::Latched { .. },
                    ..
                },
                response: AgentControlDispatchResponseError::ClientUnavailable,
            }) if evidence.kind() == AgentLiveActuationFaultKind::TransportUnavailable
        ));
        peer.join().expect("expired peer");
        assert_eq!(
            owner.last_actuation_fault().map(|fault| fault.kind()),
            Some(AgentLiveActuationFaultKind::TransportUnavailable)
        );
    }

    #[test]
    fn unimplemented_motion_modes_are_rejected_and_save_map_is_an_outer_action() {
        let (sender, mut owner, clock, _state) = fixture(ready());
        let mut parser = AgentControlRequestParser::new();
        let frontier = enqueue(
            &sender,
            &mut parser,
            1,
            json!({"kind": "frontier_explore"}),
            clock.peek(),
        );
        assert!(matches!(
            owner.process_one(AgentMapStateV1::UNAVAILABLE),
            Ok(LiveMotionOwnerOutcome::Rejected {
                code: AgentControlRejectionCodeV1::NotReady,
                retryable: false,
            })
        ));
        assert!(matches!(
            frontier.join().expect("frontier peer"),
            Some(response)
                if matches!(
                    response.response(),
                    AgentControlResponseKindV1::Rejected {
                        code: AgentControlRejectionCodeV1::NotReady,
                        retryable: false,
                    }
                )
        ));

        let save = enqueue(
            &sender,
            &mut parser,
            2,
            json!({"kind": "save_map"}),
            clock.peek(),
        );
        let LiveMotionOwnerOutcome::SaveMapRequested { claimed } = owner
            .process_one(AgentMapStateV1::UNAVAILABLE)
            .expect("outer persistence action")
        else {
            panic!("save-map outer action")
        };
        claimed
            .reject(AgentControlRejectionCodeV1::PersistenceFailed, true)
            .expect("outer persistence response");
        assert!(save.join().expect("save peer").is_some());
    }

    #[test]
    fn unsupported_preexisting_authority_is_stopped_faulted_and_never_completed() {
        let (sender, mut owner, clock, state) = fixture(active_point_goal());
        let mut parser = AgentControlRequestParser::new();
        let stop = enqueue(
            &sender,
            &mut parser,
            1,
            json!({"kind": "stop"}),
            clock.peek(),
        );
        assert!(matches!(
            owner.process_one(AgentMapStateV1::UNAVAILABLE),
            Err(LiveMotionOwnerError::Operation(
                LiveMotionOperationError::ActiveAuthorityOutsideManualLifecycle {
                    mode: AuthorityMode::PointGoal,
                    latch: LiveMotionFaultLatch::Latched {
                        fault: FaultKind::InternalInvariant,
                        ..
                    },
                }
            ))
        ));
        assert!(matches!(
            stop.join().expect("stop peer"),
            Some(response)
                if matches!(
                    response.response(),
                    AgentControlResponseKindV1::Rejected {
                        code: AgentControlRejectionCodeV1::InternalFault,
                        retryable: false,
                    }
                )
        ));
        assert_eq!(state.events(), [FakeEvent::FreshZero]);
        assert!(matches!(
            owner
                .dispatcher()
                .control_status(AgentMapStateV1::UNAVAILABLE)
                .runtime(),
            super::super::AgentRuntimeStateV1::Faulted
        ));
    }

    #[test]
    fn terminal_shutdown_orders_lifecycle_zeros_before_confirmed_controller_disarm() {
        let (_sender, owner, _clock, state) = fixture(ready());

        let report = owner.shutdown();

        assert!(report.lifecycle_cleanup().is_none());
        assert!(report.controller_stop().is_confirmed());
        assert_eq!(
            state.events(),
            [
                FakeEvent::FreshZero,
                FakeEvent::FreshZero,
                FakeEvent::Disarm
            ]
        );
        let (coordinator, lifecycle, stop) = report.into_parts();
        assert!(lifecycle.is_none());
        assert!(matches!(stop, LiveMotionTerminalStop::Confirmed(_)));
        assert_eq!(
            coordinator.motion_mode(),
            CoordinatorMotionModeV1::MappingOnly
        );
    }

    #[test]
    fn terminal_shutdown_retains_lifecycle_failure_and_still_attempts_disarm() {
        let (_sender, owner, _clock, state) = fixture(ready());
        state.fail_next_zero.store(true, Ordering::SeqCst);

        let report = owner.shutdown();

        assert!(matches!(
            report.lifecycle_cleanup(),
            Some(LiveMotionOperationError::ActuationFault {
                evidence,
                latch: LiveMotionFaultLatch::Latched { .. },
                ..
            }) if evidence.controller_stop() == AgentControllerStopKnowledge::Uncertain
        ));
        assert!(report.controller_stop().is_confirmed());
        assert_eq!(state.events(), [FakeEvent::FreshZero, FakeEvent::Disarm]);
    }

    #[test]
    fn terminal_shutdown_never_masks_an_uncertain_controller_disarm() {
        let (_sender, owner, _clock, state) = fixture(disarmed());
        state.fail_disarm.store(true, Ordering::SeqCst);

        let report = owner.shutdown();

        assert!(report.lifecycle_cleanup().is_none());
        assert!(matches!(
            report.controller_stop(),
            LiveMotionTerminalStop::Uncertain(FakeError::Injected)
        ));
        assert_eq!(state.events(), [FakeEvent::FreshZero, FakeEvent::Disarm]);
    }
}
