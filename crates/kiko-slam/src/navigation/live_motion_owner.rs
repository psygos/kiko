//! Single-threaded live owner for the bounded wheels-off motion slice.
//!
//! This type is the only seam which simultaneously owns control dispatch,
//! supervisor-backed manual state, coordinator reference ownership, the
//! receipt-gated actuator, and the host clock used for lifecycle and control
//! ticks. It never reports completion before the exact controller receipt
//! required by that operation has been obtained.

use core::fmt;
use std::num::NonZeroU64;

use kiko_supervisor_core::{
    AuthorityDuration, AuthorityLeaseId, AuthorityMode, FaultKind, SupervisorAction,
};
use robot_command_client::{AppliedCommandReceipt, DisarmReceipt};
use robot_protocol::v2::HostCommandResult;

use super::agent_manual::{
    AgentAutonomousAuthority, AgentAutonomousControlError, AgentAutonomousMode,
    AgentAutonomousRequest, AgentAutonomousTick, PendingAgentAutonomousGrant,
    PendingAgentAutonomousStop,
};
#[cfg(feature = "operator-console")]
use super::control_socket::AgentControlTypedRequestKey;
use super::mpc::{HostMonotonicClock, HostMonotonicClockReadError};
use super::{
    AgentControlClaimedRequest, AgentControlCommandV1, AgentControlDispatchResponseError,
    AgentControlDispatcher, AgentControlDispatcherError, AgentControlRejectionCodeV1,
    AgentControlStatusV1, AgentControllerStopKnowledge, AgentDispatchOutcome,
    AgentLiveActuationDisposition, AgentLiveActuationFaultKind, AgentManualControlError,
    AgentManualGlobalStopRequirement, AgentMapStateV1, BeginManualTransition,
    CoordinatorAdmissionError, CoordinatorMotionModeError, CoordinatorMotionModeV1,
    CoordinatorTickBlocker, CoordinatorTickError, DepthAdmissionOutcome, FrontierBuildError,
    FrontierInPlaceScan, FrontierSearchError, FrontierSearchOutcome, FrontierUnknownDirection,
    FrontierYawReferenceBuildError, FrontierYawScanCommandError, FrontierYawScanCommandV1,
    GlobalMapAdmissionOutcome, ImuAdmissionOutcome, LiveMpcControlDriver, LiveMpcControlError,
    ManualDriveAcceptedIntent, ManualDriveAcceptedTargetKindError, ManualDriveOutput,
    ManualMpcCommandError, MapPointGoalSelection, NanoBoundaryFrontierExplorer,
    NanoFrontierExploreConfig, NanoFrontierExplorePolicy, NanoLiveModePolicy, NanoMotionModePolicy,
    NavigationIngressSink, PlanStartBuildError, ShadowNavigationCoordinator, VisualAdmission,
    VisualAdmissionOutcome, classify_live_actuation_error,
};
use crate::dense::occupancy::{OccupancyError, OccupancyGridSnapshot};
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
pub struct LiveMotionApplied<R, D> {
    tick: HostMonotonicTimestamp,
    receipt: R,
    diagnostic: D,
    stopped: bool,
    blocked: bool,
    frontier_yaw_target_reached: bool,
    #[cfg(feature = "operator-console")]
    typed_request_key: Option<AgentControlTypedRequestKey>,
}

impl<R, D> LiveMotionApplied<R, D> {
    #[cfg(all(test, feature = "operator-console"))]
    pub(crate) fn for_test(
        tick: HostMonotonicTimestamp,
        receipt: R,
        diagnostic: D,
        stopped: bool,
        blocked: bool,
        frontier_yaw_target_reached: bool,
    ) -> Self {
        Self {
            tick,
            receipt,
            diagnostic,
            stopped,
            blocked,
            frontier_yaw_target_reached,
            #[cfg(feature = "operator-console")]
            typed_request_key: None,
        }
    }

    pub const fn tick(&self) -> HostMonotonicTimestamp {
        self.tick
    }

    pub const fn receipt(&self) -> &R {
        &self.receipt
    }

    /// Exact authoritative coordinator outcome produced by this physical tick.
    ///
    /// The live owner retains and transfers this value only after the matching
    /// controller receipt exists, so diagnostics never need to recompute MPC.
    pub const fn diagnostic(&self) -> &D {
        &self.diagnostic
    }

    pub const fn stopped(&self) -> bool {
        self.stopped
    }

    pub const fn blocked(&self) -> bool {
        self.blocked
    }

    pub const fn frontier_yaw_target_reached(&self) -> bool {
        self.frontier_yaw_target_reached
    }

    #[cfg(feature = "operator-console")]
    pub(crate) const fn typed_request_key(&self) -> Option<AgentControlTypedRequestKey> {
        self.typed_request_key
    }

    pub fn into_parts(self) -> (R, D) {
        (self.receipt, self.diagnostic)
    }

    #[cfg(feature = "operator-console")]
    pub(crate) fn into_correlated_parts(self) -> (Option<AgentControlTypedRequestKey>, R, D) {
        (self.typed_request_key, self.receipt, self.diagnostic)
    }

    #[cfg(feature = "operator-console")]
    fn bind_typed_request_key(&mut self, key: Option<AgentControlTypedRequestKey>) {
        self.typed_request_key = key;
    }
}

/// Why the sole owner issued a fresh, receipt-gated zero outside a normal
/// coordinator tick.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LiveLifecycleZeroReason {
    ArmAdmission,
    ManualAdmission,
    AutonomousAdmission,
    ManualRelease,
    AutonomousRelease,
    FrontierPhaseTransition,
    GlobalStopRequest,
    MappingOnlyRequest,
    DisarmRequest,
    ShutdownRequest,
    TerminalShutdown,
    SoftwareEmergencyStop,
    FaultContainment,
    CoordinatorInvariantRecovery,
}

/// Exact receipt and monotonic request boundary for a lifecycle zero.
pub struct LiveLifecycleZeroApplied<R> {
    requested_at: HostMonotonicTimestamp,
    receipt: R,
    reason: LiveLifecycleZeroReason,
    #[cfg(feature = "operator-console")]
    typed_request_key: Option<AgentControlTypedRequestKey>,
}

impl<R> LiveLifecycleZeroApplied<R> {
    pub const fn requested_at(&self) -> HostMonotonicTimestamp {
        self.requested_at
    }

    pub const fn receipt(&self) -> &R {
        &self.receipt
    }

    pub const fn reason(&self) -> LiveLifecycleZeroReason {
        self.reason
    }

    #[cfg(feature = "operator-console")]
    pub(crate) const fn typed_request_key(&self) -> Option<AgentControlTypedRequestKey> {
        self.typed_request_key
    }

    #[cfg(feature = "operator-console")]
    pub(crate) fn into_correlated_parts(
        self,
    ) -> (
        Option<AgentControlTypedRequestKey>,
        HostMonotonicTimestamp,
        R,
        LiveLifecycleZeroReason,
    ) {
        (
            self.typed_request_key,
            self.requested_at,
            self.receipt,
            self.reason,
        )
    }
}

/// Latest physical-state evidence produced by the sole owner.
///
/// Coordinator ticks carry the exact MPC/safety diagnostic and matching
/// receipt. Lifecycle zeros deliberately carry no fabricated coordinator
/// outcome. Actuation faults carry only the stop knowledge actually proved by
/// the failed driver operation.
pub enum LivePhysicalStateEvent<R, D> {
    CoordinatorTick(LiveMotionApplied<R, D>),
    LifecycleZero(LiveLifecycleZeroApplied<R>),
    ActuationFault {
        observed_at: HostMonotonicTimestamp,
        evidence: LiveMotionActuationFaultEvidence,
    },
}

impl<R, D> LivePhysicalStateEvent<R, D> {
    /// Best available host-monotonic boundary for this exact event.
    pub const fn observed_at(&self) -> HostMonotonicTimestamp {
        match self {
            Self::CoordinatorTick(applied) => applied.tick(),
            Self::LifecycleZero(applied) => applied.requested_at(),
            Self::ActuationFault { observed_at, .. } => *observed_at,
        }
    }

    #[cfg(feature = "operator-console")]
    pub(crate) const fn typed_request_key(&self) -> Option<AgentControlTypedRequestKey> {
        match self {
            Self::CoordinatorTick(applied) => applied.typed_request_key(),
            Self::LifecycleZero(applied) => applied.typed_request_key(),
            Self::ActuationFault { .. } => None,
        }
    }

    #[cfg(all(test, feature = "operator-console"))]
    pub(crate) fn replace_typed_request_key_for_test(&mut self, key: AgentControlTypedRequestKey) {
        match self {
            Self::CoordinatorTick(applied) => applied.typed_request_key = Some(key),
            Self::LifecycleZero(applied) => applied.typed_request_key = Some(key),
            Self::ActuationFault { .. } => {
                panic!("actuation-fault evidence has no applied-command correlation key")
            }
        }
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
pub type LiveMotionPortTickResult<R, D, E> =
    Result<LiveMotionApplied<R, D>, LiveMotionPortTickError<E>>;

pub trait LiveMotionActuationPort<J, C>
where
    J: NavigationIngressSink,
    C: HostMonotonicClock,
{
    type Receipt: LiveMotionAppliedReceipt;
    type Diagnostic;
    type Error: std::error::Error + 'static;

    fn apply_fresh_zero(&mut self) -> Result<Self::Receipt, Self::Error>;

    fn tick_manual(
        &mut self,
        coordinator: &mut ShadowNavigationCoordinator<J>,
        tick: HostMonotonicTimestamp,
        command: ManualDriveOutput<AuthorityLeaseId>,
        clock: &mut C,
    ) -> LiveMotionPortTickResult<Self::Receipt, Self::Diagnostic, Self::Error>;

    fn tick_point_goal(
        &mut self,
        coordinator: &mut ShadowNavigationCoordinator<J>,
        tick: HostMonotonicTimestamp,
        clock: &mut C,
    ) -> LiveMotionPortTickResult<Self::Receipt, Self::Diagnostic, Self::Error>;

    fn tick_frontier_yaw(
        &mut self,
        coordinator: &mut ShadowNavigationCoordinator<J>,
        tick: HostMonotonicTimestamp,
        command: FrontierYawScanCommandV1,
        clock: &mut C,
    ) -> LiveMotionPortTickResult<Self::Receipt, Self::Diagnostic, Self::Error>;

    fn classify_error(source: &Self::Error) -> LiveMotionActuationFaultEvidence;
}

/// Terminal controller operation required before a live owner may be
/// dismantled.
///
/// A successful result is the controller client's exact disarm receipt. A
/// failed disarm can nevertheless carry independent recovery evidence proving
/// the controller stopped; callers must retain both facts without fabricating
/// a successful disarm.
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
    type Diagnostic = super::CoordinatorTickOutcome<J::Error>;
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
    ) -> Result<
        LiveMotionApplied<Self::Receipt, Self::Diagnostic>,
        LiveMotionPortTickError<Self::Error>,
    > {
        let applied = LiveMpcControlDriver::tick_manual(self, coordinator, tick, command, clock)
            .map_err(map_live_mpc_port_error)?;
        let (outcome, receipt) = applied.into_parts();
        Ok(LiveMotionApplied {
            tick,
            stopped: outcome.decision().record().pwm().is_stop(),
            blocked: outcome.blocker().is_some(),
            frontier_yaw_target_reached: false,
            #[cfg(feature = "operator-console")]
            typed_request_key: None,
            receipt,
            diagnostic: outcome,
        })
    }

    fn tick_point_goal(
        &mut self,
        coordinator: &mut ShadowNavigationCoordinator<J>,
        tick: HostMonotonicTimestamp,
        clock: &mut C,
    ) -> Result<
        LiveMotionApplied<Self::Receipt, Self::Diagnostic>,
        LiveMotionPortTickError<Self::Error>,
    > {
        let applied = LiveMpcControlDriver::tick_point_goal(self, coordinator, tick, clock)
            .map_err(map_live_mpc_port_error)?;
        let (outcome, receipt) = applied.into_parts();
        Ok(LiveMotionApplied {
            tick,
            stopped: outcome.decision().record().pwm().is_stop(),
            blocked: outcome.blocker().is_some(),
            frontier_yaw_target_reached: false,
            #[cfg(feature = "operator-console")]
            typed_request_key: None,
            receipt,
            diagnostic: outcome,
        })
    }

    fn tick_frontier_yaw(
        &mut self,
        coordinator: &mut ShadowNavigationCoordinator<J>,
        tick: HostMonotonicTimestamp,
        command: FrontierYawScanCommandV1,
        clock: &mut C,
    ) -> Result<
        LiveMotionApplied<Self::Receipt, Self::Diagnostic>,
        LiveMotionPortTickError<Self::Error>,
    > {
        let applied =
            LiveMpcControlDriver::tick_frontier_yaw(self, coordinator, tick, command, clock)
                .map_err(map_live_mpc_port_error)?;
        let (outcome, receipt) = applied.into_parts();
        let frontier_yaw_target_reached = matches!(
            outcome.blocker(),
            Some(CoordinatorTickBlocker::FrontierYawReference(
                FrontierYawReferenceBuildError::TargetAlreadyReached { .. }
            ))
        );
        Ok(LiveMotionApplied {
            tick,
            stopped: outcome.decision().record().pwm().is_stop(),
            blocked: outcome.blocker().is_some(),
            frontier_yaw_target_reached,
            #[cfg(feature = "operator-console")]
            typed_request_key: None,
            receipt,
            diagnostic: outcome,
        })
    }

    fn classify_error(source: &Self::Error) -> LiveMotionActuationFaultEvidence {
        let AgentLiveActuationDisposition::LatchFault(fault) =
            classify_live_actuation_error(source);
        LiveMotionActuationFaultEvidence::new(fault.kind(), fault.controller_stop())
    }
}

fn map_live_mpc_port_error(
    source: LiveMpcControlError,
) -> LiveMotionPortTickError<super::actuation::LiveActuationError> {
    match source {
        LiveMpcControlError::Preflight(source) | LiveMpcControlError::Apply(source) => {
            LiveMotionPortTickError::Actuation(source)
        }
        LiveMpcControlError::Coordinator(source) => LiveMotionPortTickError::Coordinator(source),
        LiveMpcControlError::ManualCommand(source) => {
            LiveMotionPortTickError::ManualCommand(source)
        }
        LiveMpcControlError::ManualStop(source) => LiveMotionPortTickError::ManualStop(source),
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
    PointGoalStarted,
    PointGoalCompleted,
    FrontierExploreStarted,
    FrontierExploreCompleted,
    AutonomousStopped,
    MappingOnlyStopped,
    GlobalStopped,
    Disarmed,
    ShutdownStopped,
}

#[derive(Debug)]
pub enum LiveMotionOperationError<E, J> {
    Clock(HostMonotonicClockReadError),
    Manual(AgentManualControlError),
    Autonomous(AgentAutonomousControlError),
    CoordinatorMode(CoordinatorMotionModeError),
    CoordinatorAdmission(Box<CoordinatorAdmissionError<J>>),
    Coordinator(CoordinatorTickError),
    MotionStartReadiness(CoordinatorTickBlocker),
    FrontierBuild(FrontierBuildError),
    FrontierSearch(FrontierSearchError),
    FrontierStart(PlanStartBuildError),
    FrontierYawCommand(FrontierYawScanCommandError),
    AutonomousDeadlineOverflow {
        started_at: HostMonotonicTimestamp,
        maximum_runtime_ns: u64,
    },
    RetainedMapUnavailable,
    AutonomousGoalUnavailable,
    FrontierScanSequenceExhausted,
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

impl<E: fmt::Debug, J: fmt::Debug> fmt::Display for LiveMotionOperationError<E, J> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "live motion operation failed: {self:?}")
    }
}

impl<E, J> std::error::Error for LiveMotionOperationError<E, J>
where
    E: std::error::Error + 'static,
    J: std::error::Error + 'static,
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Clock(source) => Some(source),
            Self::Manual(source) => Some(source),
            Self::Autonomous(source) => Some(source),
            Self::CoordinatorMode(source) => Some(source),
            Self::CoordinatorAdmission(source) => Some(source.as_ref()),
            Self::Coordinator(source) => Some(source),
            Self::FrontierBuild(source) => Some(source),
            Self::FrontierSearch(source) => Some(source),
            Self::FrontierStart(source) => Some(source),
            Self::FrontierYawCommand(source) => Some(source),
            Self::ManualCommandInvariant(source) => Some(source),
            Self::ManualStopInvariant(source) => Some(source),
            Self::ActuationFault { source, .. } => Some(source),
            Self::AppliedZeroClock { source, .. } => Some(source),
            Self::PrimaryAndCleanup { primary, .. } => Some(primary),
            Self::CoordinatorDirectWithoutAuthority { .. }
            | Self::ActiveAuthorityOutsideManualLifecycle { .. }
            | Self::MotionStartReadiness(_)
            | Self::AutonomousDeadlineOverflow { .. }
            | Self::RetainedMapUnavailable
            | Self::AutonomousGoalUnavailable
            | Self::FrontierScanSequenceExhausted => None,
        }
    }
}

fn primary_with_cleanup<E, J>(
    primary: LiveMotionOperationError<E, J>,
    cleanup: LiveMotionOperationError<E, J>,
) -> LiveMotionOperationError<E, J> {
    LiveMotionOperationError::PrimaryAndCleanup {
        primary: Box::new(primary),
        cleanup: Box::new(cleanup),
    }
}

impl<E, J> From<HostMonotonicClockReadError> for LiveMotionOperationError<E, J> {
    fn from(source: HostMonotonicClockReadError) -> Self {
        Self::Clock(source)
    }
}

#[derive(Debug)]
pub enum LiveMotionOwnerError<E, J> {
    Dispatch(Box<AgentControlDispatcherError>),
    Operation(LiveMotionOperationError<E, J>),
    OperationAndResponse {
        operation: LiveMotionOperationError<E, J>,
        response: AgentControlDispatchResponseError,
    },
    ResponseAfterSafety {
        safety: LiveMotionCompletedSafetyAction,
        response: AgentControlDispatchResponseError,
    },
}

impl<E: fmt::Debug, J: fmt::Debug> fmt::Display for LiveMotionOwnerError<E, J> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "live motion owner failed: {self:?}")
    }
}

impl<E, J> std::error::Error for LiveMotionOwnerError<E, J>
where
    E: std::error::Error + 'static,
    J: std::error::Error + 'static,
{
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
pub enum LiveMotionMapAdmissionError<E> {
    SnapshotRetention(OccupancyError),
    Coordinator(CoordinatorAdmissionError<E>),
}

impl<E: fmt::Debug> fmt::Display for LiveMotionMapAdmissionError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "live motion map admission failed: {self:?}")
    }
}

impl<E: std::error::Error + 'static> std::error::Error for LiveMotionMapAdmissionError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::SnapshotRetention(source) => Some(source),
            Self::Coordinator(source) => Some(source),
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
    AutonomousAccepted {
        mode: AgentAutonomousMode,
    },
    PeriodicAutonomousApplied {
        mode: AgentAutonomousMode,
    },
    PeriodicAutonomousStopped {
        mode: AgentAutonomousMode,
    },
    AutonomousCompleted {
        mode: AgentAutonomousMode,
    },
}

/// Exact non-recoverable software emergency-stop transition performed by the
/// sole live owner.
///
/// This proves that the supervisor latched [`FaultKind::EmergencyStop`] and
/// that the matching fresh-zero command was accepted by the controller. It is
/// not evidence for an independent physical emergency-stop circuit.
#[cfg(feature = "operator-console")]
#[derive(Debug)]
pub struct LiveSoftwareEmergencyStopApplied {
    typed_request_key: AgentControlTypedRequestKey,
    fault: FaultKind,
    result: HostCommandResult,
    observed_at: HostMonotonicTimestamp,
}

#[cfg(feature = "operator-console")]
impl LiveSoftwareEmergencyStopApplied {
    pub(crate) const fn typed_request_key(&self) -> AgentControlTypedRequestKey {
        self.typed_request_key
    }

    pub const fn fault(&self) -> FaultKind {
        self.fault
    }

    pub const fn result(&self) -> HostCommandResult {
        self.result
    }

    pub const fn observed_at(&self) -> HostMonotonicTimestamp {
        self.observed_at
    }
}

impl LiveMotionOwnerOutcome {
    /// A successful manual-begin transition has applied a fresh zero and
    /// established the lease, but deliberately has no velocity target yet.
    ///
    /// The outer host loop gives the already-queued next request one control
    /// period to provide that first target. Deferral applies only to the begin
    /// period: on the next period a missing or stale target follows the normal
    /// deadman path and releases manual authority.
    pub const fn defers_periodic_motion_tick(&self) -> bool {
        matches!(
            self,
            Self::Completed(LiveMotionCompletedSafetyAction::ManualStarted { .. })
        )
    }
}

/// Exact result of the final controller-disarm operation.
///
/// Command-operation success and controller-stop knowledge are independent:
/// recovery after a failed first exchange can prove stop without producing a
/// successful disarm receipt.
#[derive(Debug)]
pub enum LiveMotionTerminalStop<Receipt, Error> {
    Confirmed(Receipt),
    DisarmFailedStopConfirmed(Error),
    Uncertain(Error),
}

impl<Receipt, Error> LiveMotionTerminalStop<Receipt, Error> {
    pub const fn is_confirmed(&self) -> bool {
        matches!(
            self,
            Self::Confirmed(_) | Self::DisarmFailedStopConfirmed(_)
        )
    }
}

/// Copyable terminal facts needed by non-physical adapters after the linear
/// terminal report has returned its coordinator for durable journal
/// finalization.
///
/// This carries no receipt and cannot be used to command or reconstruct the
/// physical owner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LiveMotionTerminalShutdownEvidence {
    controller_stop_confirmed: bool,
    lifecycle_cleanup_failed: bool,
}

impl LiveMotionTerminalShutdownEvidence {
    pub const fn controller_stop_confirmed(self) -> bool {
        self.controller_stop_confirmed
    }

    pub const fn lifecycle_cleanup_failed(self) -> bool {
        self.lifecycle_cleanup_failed
    }
}

/// Consuming shutdown result for the sole live owner.
///
/// The coordinator is returned together with, never separately from, both
/// lifecycle-cleanup and controller-disarm evidence. This lets the outer
/// runtime finalize its journal without erasing an uncertain physical stop.
#[must_use = "terminal motion-owner evidence must be inspected before publishing the journal"]
pub struct LiveMotionOwnerTerminalReport<J, StopReceipt, Error, AppliedReceipt, Diagnostic>
where
    J: NavigationIngressSink,
{
    coordinator: ShadowNavigationCoordinator<J>,
    lifecycle_cleanup: Option<LiveMotionOperationError<Error, J::Error>>,
    controller_stop: LiveMotionTerminalStop<StopReceipt, Error>,
    last_physical_state: Option<LivePhysicalStateEvent<AppliedReceipt, Diagnostic>>,
}

pub type LiveMotionOwnerTerminalParts<J, StopReceipt, Error, AppliedReceipt, Diagnostic> = (
    ShadowNavigationCoordinator<J>,
    Option<LiveMotionOperationError<Error, <J as NavigationIngressSink>::Error>>,
    LiveMotionTerminalStop<StopReceipt, Error>,
    Option<LivePhysicalStateEvent<AppliedReceipt, Diagnostic>>,
);

impl<J, StopReceipt, Error, AppliedReceipt, Diagnostic>
    LiveMotionOwnerTerminalReport<J, StopReceipt, Error, AppliedReceipt, Diagnostic>
where
    J: NavigationIngressSink,
{
    pub const fn coordinator(&self) -> &ShadowNavigationCoordinator<J> {
        &self.coordinator
    }

    pub const fn lifecycle_cleanup(&self) -> Option<&LiveMotionOperationError<Error, J::Error>> {
        self.lifecycle_cleanup.as_ref()
    }

    pub const fn controller_stop(&self) -> &LiveMotionTerminalStop<StopReceipt, Error> {
        &self.controller_stop
    }

    pub const fn last_physical_state(
        &self,
    ) -> Option<&LivePhysicalStateEvent<AppliedReceipt, Diagnostic>> {
        self.last_physical_state.as_ref()
    }

    pub const fn shutdown_evidence(&self) -> LiveMotionTerminalShutdownEvidence {
        LiveMotionTerminalShutdownEvidence {
            controller_stop_confirmed: self.controller_stop.is_confirmed(),
            lifecycle_cleanup_failed: self.lifecycle_cleanup.is_some(),
        }
    }

    pub fn into_parts(
        self,
    ) -> LiveMotionOwnerTerminalParts<J, StopReceipt, Error, AppliedReceipt, Diagnostic> {
        (
            self.coordinator,
            self.lifecycle_cleanup,
            self.controller_stop,
            self.last_physical_state,
        )
    }
}

struct RetainedLiveMotionMap {
    snapshot: OccupancyGridSnapshot,
}

enum LiveAutonomousLifecycle {
    Inactive,
    PendingGrant(PendingAgentAutonomousGrant),
    Active(LiveAutonomousSession),
    PendingStop(PendingAgentAutonomousStop),
}

struct LiveAutonomousSession {
    authority: AgentAutonomousAuthority,
    execution: LiveAutonomousExecution,
}

enum LiveAutonomousExecution {
    PointGoal {
        authority_lease: AuthorityDuration,
        deadline_exclusive: HostMonotonicTimestamp,
        arrival_tolerance_m: f64,
    },
    Frontier(Box<LiveFrontierExecution>),
}

struct LiveFrontierExecution {
    config: NanoFrontierExploreConfig,
    deadline_exclusive: HostMonotonicTimestamp,
    goals_started: u32,
    next_scan_sequence: u64,
    attempted_scan_directions: u8,
    phase: LiveFrontierPhase,
}

enum LiveFrontierPhase {
    PointGoal,
    InPlaceYaw { command: FrontierYawScanCommandV1 },
}

#[derive(Clone, Copy)]
enum FrontierAdvanceReason {
    PointGoalReached,
    MapUpdated,
    YawTargetReached,
}

/// Exact supervisor-backed motion authority retained by the sole owner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LiveMotionAuthorityState {
    Manual {
        lease_id: AuthorityLeaseId,
    },
    Autonomous {
        lease_id: AuthorityLeaseId,
        mode: AgentAutonomousMode,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LiveMotionAuthorityStateError {
    ManualAndAutonomousActive,
}

impl fmt::Display for LiveMotionAuthorityStateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("manual and autonomous motion authority are active simultaneously")
    }
}

impl std::error::Error for LiveMotionAuthorityStateError {}

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
    policy: NanoLiveModePolicy,
    retained_map: Option<RetainedLiveMotionMap>,
    autonomous: LiveAutonomousLifecycle,
    last_physical_state: Option<LivePhysicalStateEvent<P::Receipt, P::Diagnostic>>,
    last_actuation_fault: Option<LiveMotionActuationFaultEvidence>,
    #[cfg(feature = "operator-console")]
    processing_typed_request_key: Option<AgentControlTypedRequestKey>,
    #[cfg(feature = "operator-console")]
    last_processed_typed_request_key: Option<AgentControlTypedRequestKey>,
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
        policy: NanoLiveModePolicy,
    ) -> Self {
        Self {
            dispatcher,
            coordinator,
            actuation,
            clock,
            policy,
            retained_map: None,
            autonomous: LiveAutonomousLifecycle::Inactive,
            last_physical_state: None,
            last_actuation_fault: None,
            #[cfg(feature = "operator-console")]
            processing_typed_request_key: None,
            #[cfg(feature = "operator-console")]
            last_processed_typed_request_key: None,
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

    /// Observe only an actually retained supervisor authority token.
    ///
    /// Pending requests and coordinator modes are not authority evidence.
    pub fn active_motion_authority(
        &self,
    ) -> Result<Option<LiveMotionAuthorityState>, LiveMotionAuthorityStateError> {
        let manual = self.dispatcher.manual().active_lease();
        let autonomous = match &self.autonomous {
            LiveAutonomousLifecycle::Active(active) => Some(&active.authority),
            LiveAutonomousLifecycle::Inactive
            | LiveAutonomousLifecycle::PendingGrant(_)
            | LiveAutonomousLifecycle::PendingStop(_) => None,
        };
        match (manual, autonomous) {
            (None, None) => Ok(None),
            (Some(lease), None) => Ok(Some(LiveMotionAuthorityState::Manual {
                lease_id: lease.id(),
            })),
            (None, Some(authority)) => Ok(Some(LiveMotionAuthorityState::Autonomous {
                lease_id: authority.lease().id(),
                mode: authority.mode(),
            })),
            (Some(_), Some(_)) => Err(LiveMotionAuthorityStateError::ManualAndAutonomousActive),
        }
    }

    /// Transfer the most recent authoritative physical-state evidence.
    ///
    /// Production diagnostics call this after `process_one`/`tick_motion`.
    /// Taking the evidence prevents an old nonzero tick, lifecycle zero, or
    /// fault classification from being emitted again on a later idle period.
    pub fn take_last_physical_state(
        &mut self,
    ) -> Option<LivePhysicalStateEvent<P::Receipt, P::Diagnostic>> {
        self.last_physical_state.take()
    }

    /// Transfer the process-local typed identity claimed by the most recent
    /// `process_one` call. Socket traffic and idle polls return `None`.
    ///
    /// This is response correlation only. It is not authority or physical
    /// application evidence; exact completion still requires the matching
    /// linear physical-state event where the command contract demands one.
    #[cfg(feature = "operator-console")]
    pub(crate) fn take_last_processed_typed_request_key(
        &mut self,
    ) -> Option<AgentControlTypedRequestKey> {
        self.last_processed_typed_request_key.take()
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
    ) -> Result<GlobalMapAdmissionOutcome, LiveMotionMapAdmissionError<J::Error>> {
        let retained = snapshot
            .try_duplicate()
            .map_err(LiveMotionMapAdmissionError::SnapshotRetention)?;
        let outcome = self
            .coordinator
            .accept_global_map(host_arrival, source_capture_timestamp, snapshot)
            .map_err(LiveMotionMapAdmissionError::Coordinator)?;
        self.retained_map = Some(RetainedLiveMotionMap { snapshot: retained });
        Ok(outcome)
    }

    /// Claim and finish at most one command.
    pub fn process_one(
        &mut self,
        map: AgentMapStateV1,
    ) -> Result<LiveMotionOwnerOutcome, LiveMotionOwnerError<P::Error, J::Error>>
    where
        P::Error: fmt::Debug,
    {
        self.process_one_with_start_gate(map, false)
    }

    /// Claim and finish at most one production command while requiring the
    /// coordinator's exact at-now sensor readiness before any new manual or
    /// autonomous authority can be granted.
    pub fn process_one_with_motion_start_readiness(
        &mut self,
        map: AgentMapStateV1,
    ) -> Result<LiveMotionOwnerOutcome, LiveMotionOwnerError<P::Error, J::Error>>
    where
        P::Error: fmt::Debug,
    {
        self.process_one_with_start_gate(map, true)
    }

    fn process_one_with_start_gate(
        &mut self,
        map: AgentMapStateV1,
        require_motion_start_readiness: bool,
    ) -> Result<LiveMotionOwnerOutcome, LiveMotionOwnerError<P::Error, J::Error>>
    where
        P::Error: fmt::Debug,
    {
        #[cfg(feature = "operator-console")]
        {
            self.last_processed_typed_request_key = None;
        }
        let observed_at = self.now().map_err(|source| {
            LiveMotionOwnerError::Operation(LiveMotionOperationError::Clock(source))
        })?;
        let motion_start_readiness = require_motion_start_readiness
            .then(|| self.coordinator.motion_start_readiness_at(observed_at));
        let outcome = self.dispatcher.try_dispatch_one_at(map, observed_at);
        #[cfg(feature = "operator-console")]
        {
            self.processing_typed_request_key =
                self.dispatcher.take_last_claimed_typed_request_key();
            self.last_processed_typed_request_key = self.processing_typed_request_key;
        }
        let outcome = match outcome {
            Ok(outcome) => outcome,
            Err(source) => {
                #[cfg(feature = "operator-console")]
                {
                    self.processing_typed_request_key = None;
                }
                return Err(LiveMotionOwnerError::Dispatch(Box::new(source)));
            }
        };
        let result = match outcome {
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
            } => match motion_start_readiness {
                Some(Err(source)) => self.reject_unready_manual_start(claimed, source),
                Some(Ok(())) | None => self.begin_manual(claimed, transition),
            },
            AgentDispatchOutcome::ManualCommand { claimed, output } => {
                self.manual_command(claimed, output)
            }
            AgentDispatchOutcome::GlobalStopRequested { claimed, manual } => self.global_stop(
                claimed,
                manual,
                LiveMotionCompletedSafetyAction::GlobalStopped,
            ),
            AgentDispatchOutcome::Shutdown { claimed } => {
                let stopped = self.stop_all_motion(
                    self.dispatcher.manual().global_stop_requirement(),
                    LiveLifecycleZeroReason::ShutdownRequest,
                );
                match stopped {
                    Ok(()) => self.respond_shutdown_completed(claimed),
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
                AgentControlCommandV1::SaveMap => self.request_save_map(claimed),
                AgentControlCommandV1::FrontierExplore => match motion_start_readiness {
                    Some(Err(_)) => self.reject_without_operation(
                        claimed,
                        AgentControlRejectionCodeV1::NotReady,
                        true,
                    ),
                    Some(Ok(())) | None => self.begin_frontier(claimed),
                },
                AgentControlCommandV1::SelectMapPoint(selection) => match motion_start_readiness {
                    Some(Err(_)) => self.reject_without_operation(
                        claimed,
                        AgentControlRejectionCodeV1::NotReady,
                        true,
                    ),
                    Some(Ok(())) | None => self.begin_point_goal(claimed, selection),
                },
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
        };
        #[cfg(feature = "operator-console")]
        {
            self.processing_typed_request_key = None;
        }
        result
    }

    /// Irreversibly latch the process supervisor and apply a fresh physical
    /// zero through this owner's existing controller session.
    ///
    /// Queued or future socket/console commands cannot reacquire authority
    /// from the resulting fault state. Recovery must cross the existing
    /// clear-to-inventory and complete re-admission path; this method provides
    /// no in-place reset.
    #[cfg(feature = "operator-console")]
    pub(crate) fn apply_software_emergency_stop(
        &mut self,
        typed_request_key: AgentControlTypedRequestKey,
    ) -> Result<LiveSoftwareEmergencyStopApplied, LiveMotionOwnerError<P::Error, J::Error>> {
        let fault_at = self.now().map_err(|source| {
            LiveMotionOwnerError::Operation(LiveMotionOperationError::Clock(source))
        })?;

        // Destroy every cached autonomous/direct-control continuation before
        // latching. A failed mode cleanup is reported after the mandatory zero
        // attempt; it cannot preserve authority because the supervisor fault
        // transition below clears the active lease.
        let direct_mode_cleanup = match self.coordinator.motion_mode() {
            CoordinatorMotionModeV1::Manual { authority_lease_id }
            | CoordinatorMotionModeV1::FrontierInPlaceYaw { authority_lease_id } => self
                .coordinator
                .leave_direct_mode(authority_lease_id)
                .map_err(LiveMotionOperationError::CoordinatorMode)
                .err(),
            CoordinatorMotionModeV1::MappingOnly | CoordinatorMotionModeV1::PointGoal => None,
        };
        self.coordinator.clear_goal();
        self.autonomous = LiveAutonomousLifecycle::Inactive;

        let latch_error = match self
            .dispatcher
            .manual_mut()
            .latch_fault(FaultKind::EmergencyStop, fault_at)
        {
            Ok(SupervisorAction::FaultStopRequired {
                fault: FaultKind::EmergencyStop,
            }) => None,
            Ok(action) => Some(LiveMotionOperationError::Manual(
                AgentManualControlError::UnexpectedSupervisorAction {
                    operation: "latch software emergency stop",
                    action,
                },
            )),
            Err(source) => Some(LiveMotionOperationError::Manual(source)),
        };

        let prior_key = self.processing_typed_request_key.replace(typed_request_key);
        let zero = self.fresh_zero(fault_at, LiveLifecycleZeroReason::SoftwareEmergencyStop);
        self.processing_typed_request_key = prior_key;
        let primary = match (direct_mode_cleanup, latch_error) {
            (Some(primary), Some(cleanup)) => Some(LiveMotionOperationError::PrimaryAndCleanup {
                primary: Box::new(primary),
                cleanup: Box::new(cleanup),
            }),
            (Some(primary), None) | (None, Some(primary)) => Some(primary),
            (None, None) => None,
        };
        match (primary, zero) {
            (None, Ok((result, observed_at))) => Ok(LiveSoftwareEmergencyStopApplied {
                typed_request_key,
                fault: FaultKind::EmergencyStop,
                result,
                observed_at,
            }),
            (Some(primary), Ok(_)) => Err(LiveMotionOwnerError::Operation(primary)),
            (None, Err(cleanup)) => Err(LiveMotionOwnerError::Operation(cleanup)),
            (Some(primary), Err(cleanup)) => Err(LiveMotionOwnerError::Operation(
                LiveMotionOperationError::PrimaryAndCleanup {
                    primary: Box::new(primary),
                    cleanup: Box::new(cleanup),
                },
            )),
        }
    }

    /// Advance whichever motion mode owns the sole supervisor lease.
    ///
    /// Production must call this on every control period. Autonomous and
    /// manual modes are mutually exclusive; pending autonomous zero
    /// continuations are completed before any other command can run.
    pub fn tick_motion(
        &mut self,
    ) -> Result<LiveMotionOwnerOutcome, LiveMotionOwnerError<P::Error, J::Error>>
    where
        P::Error: fmt::Debug,
    {
        if matches!(self.autonomous, LiveAutonomousLifecycle::Inactive) {
            return self.tick_manual();
        }
        let requested_at = self.now().map_err(|source| {
            LiveMotionOwnerError::Operation(LiveMotionOperationError::Clock(source))
        })?;
        let autonomous = std::mem::replace(&mut self.autonomous, LiveAutonomousLifecycle::Inactive);
        match autonomous {
            LiveAutonomousLifecycle::Inactive => unreachable!("autonomous state checked above"),
            LiveAutonomousLifecycle::PendingGrant(pending) => {
                let mode = pending.mode();
                self.cancel_and_stop_pending_grant(
                    pending,
                    requested_at,
                    LiveLifecycleZeroReason::AutonomousRelease,
                )
                .map_err(LiveMotionOwnerError::Operation)?;
                Ok(LiveMotionOwnerOutcome::PeriodicAutonomousStopped { mode })
            }
            LiveAutonomousLifecycle::PendingStop(pending) => {
                let mode = pending.mode();
                self.complete_pending_autonomous_stop(
                    pending,
                    requested_at,
                    LiveLifecycleZeroReason::AutonomousRelease,
                )
                .map_err(LiveMotionOwnerError::Operation)?;
                Ok(LiveMotionOwnerOutcome::PeriodicAutonomousStopped { mode })
            }
            LiveAutonomousLifecycle::Active(session) => {
                self.tick_active_autonomous(session, requested_at)
            }
        }
    }

    /// Advance the active manual lease and deadman even when no socket command
    /// is available.
    pub fn tick_manual(
        &mut self,
    ) -> Result<LiveMotionOwnerOutcome, LiveMotionOwnerError<P::Error, J::Error>>
    where
        P::Error: fmt::Debug,
    {
        if self.dispatcher.manual().active_lease().is_none() {
            return Ok(LiveMotionOwnerOutcome::Idle);
        }
        let tick = self.now().map_err(|source| {
            LiveMotionOwnerError::Operation(LiveMotionOperationError::Clock(source))
        })?;
        let command = match self.dispatcher.manual_mut().tick(tick) {
            Ok(tick) => tick.output(),
            Err(source) => {
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
        #[cfg(feature = "operator-console")]
        let mut applied = applied;
        #[cfg(feature = "operator-console")]
        applied.bind_typed_request_key(self.processing_typed_request_key);
        let stopped = applied.stopped();
        self.last_physical_state = Some(LivePhysicalStateEvent::CoordinatorTick(applied));
        if stopped || command.target().is_stop() {
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
    pub fn shutdown(
        mut self,
    ) -> LiveMotionOwnerTerminalReport<J, P::StopEvidence, P::Error, P::Receipt, P::Diagnostic>
    where
        P: LiveMotionTerminalActuationPort<J, C>,
    {
        let lifecycle_cleanup = self.shutdown_lifecycle().err();
        let controller_stop = match self.actuation.disarm() {
            Ok(receipt) => LiveMotionTerminalStop::Confirmed(receipt),
            Err(source) => {
                if P::classify_error(&source).controller_stop()
                    == AgentControllerStopKnowledge::Confirmed
                {
                    LiveMotionTerminalStop::DisarmFailedStopConfirmed(source)
                } else {
                    LiveMotionTerminalStop::Uncertain(source)
                }
            }
        };
        LiveMotionOwnerTerminalReport {
            coordinator: self.coordinator,
            lifecycle_cleanup,
            controller_stop,
            last_physical_state: self.last_physical_state,
        }
    }

    fn shutdown_lifecycle(&mut self) -> Result<(), LiveMotionOperationError<P::Error, J::Error>> {
        self.stop_all_motion(
            self.dispatcher.manual().global_stop_requirement(),
            LiveLifecycleZeroReason::TerminalShutdown,
        )?;
        let requested_at = self.now()?;
        let action = self
            .dispatcher
            .begin_disarm(requested_at)
            .map_err(LiveMotionOperationError::Manual)?;
        if matches!(action, SupervisorAction::Disarmed) {
            return Ok(());
        }
        let (result, observed_at) =
            self.fresh_zero(requested_at, LiveLifecycleZeroReason::TerminalShutdown)?;
        self.dispatcher
            .complete_disarm_with_applied_zero(result, observed_at, observed_at)
            .map_err(LiveMotionOperationError::Manual)?;
        Ok(())
    }

    fn begin_point_goal(
        &mut self,
        claimed: AgentControlClaimedRequest,
        selection: MapPointGoalSelection,
    ) -> Result<LiveMotionOwnerOutcome, LiveMotionOwnerError<P::Error, J::Error>>
    where
        P::Error: fmt::Debug,
    {
        let NanoMotionModePolicy::ControlApi {
            authority_lease,
            maximum_runtime,
            arrival_tolerance_m,
        } = self.policy.point_goal()
        else {
            return self.reject_without_operation(
                claimed,
                AgentControlRejectionCodeV1::AuthorityDenied,
                false,
            );
        };
        if !matches!(self.autonomous, LiveAutonomousLifecycle::Inactive) {
            return self.reject_without_operation(
                claimed,
                AgentControlRejectionCodeV1::ModeConflict,
                false,
            );
        }
        if self.dispatcher.manual().active_lease().is_some() {
            return self.reject_without_operation(
                claimed,
                AgentControlRejectionCodeV1::ModeConflict,
                false,
            );
        }
        let requested_at = match self.now() {
            Ok(now) => now,
            Err(source) => {
                return self.reject_operation(
                    claimed,
                    LiveMotionOperationError::Clock(source),
                    AgentControlRejectionCodeV1::InternalFault,
                    false,
                );
            }
        };
        let deadline_exclusive = match autonomous_deadline(requested_at, maximum_runtime) {
            Ok(deadline) => deadline,
            Err(source) => {
                return self.reject_operation(
                    claimed,
                    source,
                    AgentControlRejectionCodeV1::InternalFault,
                    false,
                );
            }
        };
        let Some(retained) = self.retained_map.as_ref() else {
            return self.reject_without_operation(
                claimed,
                AgentControlRejectionCodeV1::MapUnavailable,
                true,
            );
        };
        let prepared = match self
            .coordinator
            .prepare_map_point_goal(selection, &retained.snapshot)
        {
            Ok(prepared) => prepared,
            Err(source) => return self.reject_coordinator_admission(claimed, source),
        };
        let authority = {
            let Self {
                dispatcher,
                actuation,
                clock,
                autonomous,
                last_actuation_fault,
                last_physical_state,
                #[cfg(feature = "operator-console")]
                processing_typed_request_key,
                ..
            } = self;
            match Self::acquire_autonomous_parts(
                dispatcher,
                actuation,
                clock,
                autonomous,
                last_actuation_fault,
                last_physical_state,
                #[cfg(feature = "operator-console")]
                *processing_typed_request_key,
                AgentAutonomousMode::PointGoal,
                authority_lease,
                requested_at,
            ) {
                Ok(authority) => authority,
                Err(source) => {
                    return self.reject_operation(
                        claimed,
                        source,
                        AgentControlRejectionCodeV1::NotReady,
                        true,
                    );
                }
            }
        };
        if let Err(source) = self
            .coordinator
            .commit_prepared_map_point_goal(requested_at, prepared)
        {
            self.autonomous = LiveAutonomousLifecycle::Active(LiveAutonomousSession {
                authority,
                execution: LiveAutonomousExecution::PointGoal {
                    authority_lease,
                    deadline_exclusive,
                    arrival_tolerance_m,
                },
            });
            let primary = LiveMotionOperationError::CoordinatorAdmission(Box::new(source));
            let operation = match self.stop_autonomous(LiveLifecycleZeroReason::FaultContainment) {
                Ok(()) => primary,
                Err(cleanup) => LiveMotionOperationError::PrimaryAndCleanup {
                    primary: Box::new(primary),
                    cleanup: Box::new(cleanup),
                },
            };
            return self.reject_operation(
                claimed,
                operation,
                AgentControlRejectionCodeV1::InternalFault,
                false,
            );
        }
        self.autonomous = LiveAutonomousLifecycle::Active(LiveAutonomousSession {
            authority,
            execution: LiveAutonomousExecution::PointGoal {
                authority_lease,
                deadline_exclusive,
                arrival_tolerance_m,
            },
        });
        self.finish_autonomous_start(
            claimed,
            AgentAutonomousMode::PointGoal,
            LiveMotionCompletedSafetyAction::PointGoalStarted,
            LiveMotionCompletedSafetyAction::PointGoalCompleted,
        )
    }

    fn begin_frontier(
        &mut self,
        claimed: AgentControlClaimedRequest,
    ) -> Result<LiveMotionOwnerOutcome, LiveMotionOwnerError<P::Error, J::Error>>
    where
        P::Error: fmt::Debug,
    {
        let NanoFrontierExplorePolicy::ControlApi(config) = self.policy.frontier_explore() else {
            return self.reject_without_operation(
                claimed,
                AgentControlRejectionCodeV1::AuthorityDenied,
                false,
            );
        };
        if !matches!(self.autonomous, LiveAutonomousLifecycle::Inactive)
            || self.dispatcher.manual().active_lease().is_some()
        {
            return self.reject_without_operation(
                claimed,
                AgentControlRejectionCodeV1::ModeConflict,
                false,
            );
        }
        let requested_at = match self.now() {
            Ok(now) => now,
            Err(source) => {
                return self.reject_operation(
                    claimed,
                    LiveMotionOperationError::Clock(source),
                    AgentControlRejectionCodeV1::InternalFault,
                    false,
                );
            }
        };
        let deadline_exclusive = match autonomous_deadline(requested_at, config.maximum_runtime()) {
            Ok(deadline) => deadline,
            Err(source) => {
                return self.reject_operation(
                    claimed,
                    source,
                    AgentControlRejectionCodeV1::InternalFault,
                    false,
                );
            }
        };
        let Some(retained) = self.retained_map.as_ref() else {
            return self.reject_without_operation(
                claimed,
                AgentControlRejectionCodeV1::MapUnavailable,
                true,
            );
        };
        let start = match self.coordinator.plan_start_for_snapshot(&retained.snapshot) {
            Ok(start) => start,
            Err(source) => {
                return self.reject_operation(
                    claimed,
                    LiveMotionOperationError::FrontierStart(source),
                    AgentControlRejectionCodeV1::NotReady,
                    true,
                );
            }
        };
        let selection = match NanoBoundaryFrontierExplorer::try_new(
            &retained.snapshot,
            config.explorer(),
            config.boundary_m(),
        ) {
            Ok(mut explorer) => match explorer.select(start) {
                Ok(selection) => selection,
                Err(source) => {
                    return self.reject_operation(
                        claimed,
                        LiveMotionOperationError::FrontierSearch(source),
                        AgentControlRejectionCodeV1::NotReady,
                        true,
                    );
                }
            },
            Err(source) => {
                return self.reject_operation(
                    claimed,
                    LiveMotionOperationError::FrontierBuild(source),
                    AgentControlRejectionCodeV1::NotReady,
                    true,
                );
            }
        };
        if matches!(selection, FrontierSearchOutcome::NoReachableFrontier { .. }) {
            let safety = LiveMotionCompletedSafetyAction::FrontierExploreCompleted;
            return self.respond_completed(
                claimed,
                safety,
                LiveMotionOwnerOutcome::AutonomousCompleted {
                    mode: AgentAutonomousMode::Explore,
                },
            );
        }

        let prepared = match selection {
            FrontierSearchOutcome::Selected(frontier) => {
                match self
                    .coordinator
                    .prepare_frontier_goal(frontier, &retained.snapshot)
                {
                    Ok(prepared) => Some(prepared),
                    Err(source) => return self.reject_coordinator_admission(claimed, source),
                }
            }
            FrontierSearchOutcome::InPlaceScanRequired(_) => None,
            FrontierSearchOutcome::NoReachableFrontier { .. } => unreachable!(),
        };
        let authority = {
            let Self {
                dispatcher,
                actuation,
                clock,
                autonomous,
                last_actuation_fault,
                last_physical_state,
                #[cfg(feature = "operator-console")]
                processing_typed_request_key,
                ..
            } = self;
            match Self::acquire_autonomous_parts(
                dispatcher,
                actuation,
                clock,
                autonomous,
                last_actuation_fault,
                last_physical_state,
                #[cfg(feature = "operator-console")]
                *processing_typed_request_key,
                AgentAutonomousMode::Explore,
                config.authority_lease(),
                requested_at,
            ) {
                Ok(authority) => authority,
                Err(source) => {
                    return self.reject_operation(
                        claimed,
                        source,
                        AgentControlRejectionCodeV1::NotReady,
                        true,
                    );
                }
            }
        };

        let (phase, goals_started, next_scan_sequence, attempted_scan_directions) =
            match (selection, prepared) {
                (FrontierSearchOutcome::Selected(_), Some(prepared)) => {
                    if let Err(source) = self
                        .coordinator
                        .commit_prepared_map_point_goal(requested_at, prepared)
                    {
                        let operation = self.cleanup_failed_autonomous_start(
                            authority,
                            LiveAutonomousExecution::Frontier(Box::new(LiveFrontierExecution {
                                config,
                                deadline_exclusive,
                                goals_started: 0,
                                next_scan_sequence: 1,
                                attempted_scan_directions: 0,
                                phase: LiveFrontierPhase::PointGoal,
                            })),
                            LiveMotionOperationError::CoordinatorAdmission(Box::new(source)),
                        );
                        return self.reject_operation(
                            claimed,
                            operation,
                            AgentControlRejectionCodeV1::InternalFault,
                            false,
                        );
                    }
                    (LiveFrontierPhase::PointGoal, 1, 1, 0)
                }
                (FrontierSearchOutcome::InPlaceScanRequired(scan), None) => {
                    let target_direction = scan
                        .unknown_directions()
                        .iter()
                        .next()
                        .expect("frontier scan evidence is nonempty");
                    let command = match FrontierYawScanCommandV1::try_new(
                        authority.lease().id(),
                        1,
                        scan,
                        target_direction,
                        config.yaw_turn_direction(),
                        requested_at,
                        HostMonotonicTimestamp::from_nanos(
                            authority.lease().expires_at_exclusive().as_nanos(),
                        ),
                        config.yaw_scan_budget(),
                    ) {
                        Ok(command) => command,
                        Err(source) => {
                            let operation = self.cleanup_failed_autonomous_start(
                                authority,
                                LiveAutonomousExecution::Frontier(Box::new(
                                    LiveFrontierExecution {
                                        config,
                                        deadline_exclusive,
                                        goals_started: 0,
                                        next_scan_sequence: 1,
                                        attempted_scan_directions: 0,
                                        phase: LiveFrontierPhase::PointGoal,
                                    },
                                )),
                                LiveMotionOperationError::FrontierYawCommand(source),
                            );
                            return self.reject_operation(
                                claimed,
                                operation,
                                AgentControlRejectionCodeV1::NotReady,
                                true,
                            );
                        }
                    };
                    if let Err(source) = self
                        .coordinator
                        .enter_frontier_yaw_mode(nonzero_lease(authority.lease().id()))
                    {
                        let operation = self.cleanup_failed_autonomous_start(
                            authority,
                            LiveAutonomousExecution::Frontier(Box::new(LiveFrontierExecution {
                                config,
                                deadline_exclusive,
                                goals_started: 0,
                                next_scan_sequence: 2,
                                attempted_scan_directions: frontier_direction_bit(target_direction),
                                phase: LiveFrontierPhase::InPlaceYaw { command },
                            })),
                            LiveMotionOperationError::CoordinatorMode(source),
                        );
                        return self.reject_operation(
                            claimed,
                            operation,
                            AgentControlRejectionCodeV1::ModeConflict,
                            false,
                        );
                    }
                    (
                        LiveFrontierPhase::InPlaceYaw { command },
                        0,
                        2,
                        frontier_direction_bit(target_direction),
                    )
                }
                _ => unreachable!("frontier preparation matches its search result"),
            };
        self.autonomous = LiveAutonomousLifecycle::Active(LiveAutonomousSession {
            authority,
            execution: LiveAutonomousExecution::Frontier(Box::new(LiveFrontierExecution {
                config,
                deadline_exclusive,
                goals_started,
                next_scan_sequence,
                attempted_scan_directions,
                phase,
            })),
        });
        self.finish_autonomous_start(
            claimed,
            AgentAutonomousMode::Explore,
            LiveMotionCompletedSafetyAction::FrontierExploreStarted,
            LiveMotionCompletedSafetyAction::FrontierExploreCompleted,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn acquire_autonomous_parts(
        dispatcher: &mut AgentControlDispatcher,
        actuation: &mut P,
        clock: &mut C,
        autonomous: &mut LiveAutonomousLifecycle,
        last_actuation_fault: &mut Option<LiveMotionActuationFaultEvidence>,
        last_physical_state: &mut Option<LivePhysicalStateEvent<P::Receipt, P::Diagnostic>>,
        #[cfg(feature = "operator-console")] typed_request_key: Option<AgentControlTypedRequestKey>,
        mode: AgentAutonomousMode,
        duration: AuthorityDuration,
        requested_at: HostMonotonicTimestamp,
    ) -> Result<AgentAutonomousAuthority, LiveMotionOperationError<P::Error, J::Error>> {
        match dispatcher
            .request_autonomous(mode, duration, requested_at)
            .map_err(LiveMotionOperationError::Autonomous)?
        {
            AgentAutonomousRequest::Granted(authority) => Ok(authority),
            AgentAutonomousRequest::FreshAppliedZeroRequired(pending) => {
                *autonomous = LiveAutonomousLifecycle::PendingGrant(pending);
                let receipt = match actuation.apply_fresh_zero() {
                    Ok(receipt) => receipt,
                    Err(source) => {
                        let evidence = P::classify_error(&source);
                        *last_actuation_fault = Some(evidence);
                        *last_physical_state = Some(LivePhysicalStateEvent::ActuationFault {
                            observed_at: requested_at,
                            evidence,
                        });
                        let fault = evidence.supervisor_fault();
                        let latch = match dispatcher.manual_mut().latch_fault(fault, requested_at) {
                            Ok(obligation) => LiveMotionFaultLatch::Latched { fault, obligation },
                            Err(source) => LiveMotionFaultLatch::Failed(source),
                        };
                        *autonomous = LiveAutonomousLifecycle::Inactive;
                        return Err(LiveMotionOperationError::ActuationFault {
                            source,
                            evidence,
                            latch,
                        });
                    }
                };
                let result = receipt.verified_host_result();
                *last_physical_state = Some(LivePhysicalStateEvent::LifecycleZero(
                    LiveLifecycleZeroApplied {
                        requested_at,
                        receipt,
                        reason: LiveLifecycleZeroReason::AutonomousAdmission,
                        #[cfg(feature = "operator-console")]
                        typed_request_key,
                    },
                ));
                let observed_at = match clock.try_now() {
                    Ok(observed_at) => observed_at,
                    Err(source) => {
                        return Err(LiveMotionOperationError::AppliedZeroClock { source, result });
                    }
                };
                let pending = match autonomous {
                    LiveAutonomousLifecycle::PendingGrant(pending) => pending,
                    _ => unreachable!("grant continuation is retained until exact completion"),
                };
                match dispatcher.complete_autonomous_grant_with_applied_zero(
                    pending,
                    result,
                    observed_at,
                    observed_at,
                ) {
                    Ok(authority) => {
                        *autonomous = LiveAutonomousLifecycle::Inactive;
                        Ok(authority)
                    }
                    Err(source) => {
                        if matches!(
                            &source,
                            AgentAutonomousControlError::PendingAuthorityExpired
                                | AgentAutonomousControlError::UnexpectedSupervisorAction { .. }
                        ) {
                            *autonomous = LiveAutonomousLifecycle::Inactive;
                        }
                        Err(LiveMotionOperationError::Autonomous(source))
                    }
                }
            }
        }
    }

    fn finish_autonomous_start(
        &mut self,
        claimed: AgentControlClaimedRequest,
        mode: AgentAutonomousMode,
        started: LiveMotionCompletedSafetyAction,
        completed: LiveMotionCompletedSafetyAction,
    ) -> Result<LiveMotionOwnerOutcome, LiveMotionOwnerError<P::Error, J::Error>>
    where
        P::Error: fmt::Debug,
    {
        match self.tick_motion() {
            Ok(LiveMotionOwnerOutcome::PeriodicAutonomousApplied { .. }) => self.respond_accepted(
                claimed,
                started,
                LiveMotionOwnerOutcome::AutonomousAccepted { mode },
            ),
            Ok(LiveMotionOwnerOutcome::AutonomousCompleted { .. }) => self.respond_completed(
                claimed,
                completed,
                LiveMotionOwnerOutcome::AutonomousCompleted { mode },
            ),
            Ok(LiveMotionOwnerOutcome::PeriodicAutonomousStopped { .. }) => self
                .reject_after_safety(
                    claimed,
                    LiveMotionCompletedSafetyAction::AutonomousStopped,
                    AgentControlRejectionCodeV1::NotReady,
                    true,
                ),
            Ok(_) => {
                let operation = self.stop_after_unadmitted_fault(
                    LiveMotionOperationError::CoordinatorDirectWithoutAuthority {
                        mode: self.coordinator.motion_mode(),
                    },
                );
                self.reject_operation(
                    claimed,
                    operation,
                    AgentControlRejectionCodeV1::InternalFault,
                    false,
                )
            }
            Err(LiveMotionOwnerError::Operation(operation)) => self.reject_operation(
                claimed,
                operation,
                AgentControlRejectionCodeV1::SafetyStopped,
                true,
            ),
            Err(source) => Err(source),
        }
    }

    fn tick_active_autonomous(
        &mut self,
        mut session: LiveAutonomousSession,
        tick: HostMonotonicTimestamp,
    ) -> Result<LiveMotionOwnerOutcome, LiveMotionOwnerError<P::Error, J::Error>>
    where
        P::Error: fmt::Debug,
    {
        let mode = session.authority.mode();
        let authority_tick = match self.dispatcher.tick_autonomous(&session.authority, tick) {
            Ok(authority_tick) => authority_tick,
            Err(source) => {
                let primary = LiveMotionOperationError::Autonomous(source);
                let operation = self.cleanup_active_autonomous_failure(session, primary);
                return Err(LiveMotionOwnerError::Operation(operation));
            }
        };
        match authority_tick {
            AgentAutonomousTick::Active => {}
            AgentAutonomousTick::FreshAppliedStopRequired(pending) => {
                self.finish_autonomous_stop_transition(
                    &session,
                    pending,
                    tick,
                    LiveLifecycleZeroReason::AutonomousRelease,
                )
                .map_err(LiveMotionOwnerError::Operation)?;
                return Ok(LiveMotionOwnerOutcome::PeriodicAutonomousStopped { mode });
            }
        }

        let deadline_exclusive = match &session.execution {
            LiveAutonomousExecution::PointGoal {
                deadline_exclusive, ..
            } => *deadline_exclusive,
            LiveAutonomousExecution::Frontier(frontier) => frontier.deadline_exclusive,
        };
        if tick >= deadline_exclusive {
            self.release_autonomous_session(
                session,
                tick,
                LiveLifecycleZeroReason::AutonomousRelease,
            )
            .map_err(LiveMotionOwnerError::Operation)?;
            return Ok(LiveMotionOwnerOutcome::AutonomousCompleted { mode });
        }

        let authority_lease = match &session.execution {
            LiveAutonomousExecution::PointGoal {
                authority_lease, ..
            } => *authority_lease,
            LiveAutonomousExecution::Frontier(frontier) => frontier.config.authority_lease(),
        };
        if let Err(source) =
            self.dispatcher
                .renew_autonomous(&mut session.authority, authority_lease, tick)
        {
            if let AgentAutonomousControlError::LeaseExpired { pending } = source {
                self.finish_autonomous_stop_transition(
                    &session,
                    pending,
                    tick,
                    LiveLifecycleZeroReason::AutonomousRelease,
                )
                .map_err(LiveMotionOwnerError::Operation)?;
                return Ok(LiveMotionOwnerOutcome::PeriodicAutonomousStopped { mode });
            }
            let operation = self.cleanup_active_autonomous_failure(
                session,
                LiveMotionOperationError::Autonomous(source),
            );
            return Err(LiveMotionOwnerError::Operation(operation));
        }
        if let LiveAutonomousExecution::Frontier(frontier) = &mut session.execution
            && let LiveFrontierPhase::InPlaceYaw { command } = &mut frontier.phase
        {
            let rebound = FrontierYawScanCommandV1::try_new(
                session.authority.lease().id(),
                command.scan_sequence(),
                command.scan(),
                command.target_direction(),
                command.turn_direction(),
                command.started_at(),
                HostMonotonicTimestamp::from_nanos(
                    session.authority.lease().expires_at_exclusive().as_nanos(),
                ),
                command.budget(),
            );
            match rebound {
                Ok(rebound) => *command = rebound,
                Err(source) => {
                    let operation = self.cleanup_active_autonomous_failure(
                        session,
                        LiveMotionOperationError::FrontierYawCommand(source),
                    );
                    return Err(LiveMotionOwnerError::Operation(operation));
                }
            }
        }

        match &session.execution {
            LiveAutonomousExecution::PointGoal {
                arrival_tolerance_m,
                ..
            } => match self.current_goal_reached(*arrival_tolerance_m) {
                Ok(true) => {
                    self.release_autonomous_session(
                        session,
                        tick,
                        LiveLifecycleZeroReason::AutonomousRelease,
                    )
                    .map_err(LiveMotionOwnerError::Operation)?;
                    return Ok(LiveMotionOwnerOutcome::AutonomousCompleted { mode });
                }
                Ok(false) => {}
                Err(primary) => {
                    let operation = self.cleanup_active_autonomous_failure(session, primary);
                    return Err(LiveMotionOwnerError::Operation(operation));
                }
            },
            LiveAutonomousExecution::Frontier(frontier) => {
                let advance_reason = match &frontier.phase {
                    LiveFrontierPhase::PointGoal => {
                        match self.current_goal_reached(frontier.config.arrival_tolerance_m()) {
                            Ok(true) => Some(FrontierAdvanceReason::PointGoalReached),
                            Ok(false) => None,
                            Err(primary) => {
                                let operation =
                                    self.cleanup_active_autonomous_failure(session, primary);
                                return Err(LiveMotionOwnerError::Operation(operation));
                            }
                        }
                    }
                    LiveFrontierPhase::InPlaceYaw { command } => self
                        .retained_map
                        .as_ref()
                        .is_some_and(|retained| {
                            retained.snapshot.map_instance_id()
                                != Some(command.scan().map_instance_id())
                                || retained.snapshot.revision() != command.scan().map_revision()
                        })
                        .then_some(FrontierAdvanceReason::MapUpdated),
                };
                if let Some(reason) = advance_reason {
                    match self.advance_frontier(&mut session, tick, reason) {
                        Ok(true) => {
                            self.release_autonomous_session(
                                session,
                                tick,
                                LiveLifecycleZeroReason::AutonomousRelease,
                            )
                            .map_err(LiveMotionOwnerError::Operation)?;
                            return Ok(LiveMotionOwnerOutcome::AutonomousCompleted { mode });
                        }
                        Ok(false) => {}
                        Err(primary) => {
                            let operation =
                                self.cleanup_active_autonomous_failure(session, primary);
                            return Err(LiveMotionOwnerError::Operation(operation));
                        }
                    }
                }
            }
        }

        let applied = match &session.execution {
            LiveAutonomousExecution::PointGoal { .. } => {
                self.actuation
                    .tick_point_goal(&mut self.coordinator, tick, &mut self.clock)
            }
            LiveAutonomousExecution::Frontier(frontier) => match &frontier.phase {
                LiveFrontierPhase::PointGoal => {
                    self.actuation
                        .tick_point_goal(&mut self.coordinator, tick, &mut self.clock)
                }
                LiveFrontierPhase::InPlaceYaw { command } => self.actuation.tick_frontier_yaw(
                    &mut self.coordinator,
                    tick,
                    *command,
                    &mut self.clock,
                ),
            },
        };
        let applied = match applied {
            Ok(applied) => applied,
            Err(source) => {
                let primary = self.port_tick_error(source, tick);
                let operation = self.cleanup_active_autonomous_failure(session, primary);
                return Err(LiveMotionOwnerError::Operation(operation));
            }
        };
        #[cfg(feature = "operator-console")]
        let mut applied = applied;
        #[cfg(feature = "operator-console")]
        applied.bind_typed_request_key(self.processing_typed_request_key);
        let frontier_yaw_target_reached = applied.frontier_yaw_target_reached();
        let blocked = applied.blocked();
        self.last_physical_state = Some(LivePhysicalStateEvent::CoordinatorTick(applied));
        if frontier_yaw_target_reached {
            match self.advance_frontier(&mut session, tick, FrontierAdvanceReason::YawTargetReached)
            {
                Ok(true) => {
                    self.release_autonomous_session(
                        session,
                        tick,
                        LiveLifecycleZeroReason::AutonomousRelease,
                    )
                    .map_err(LiveMotionOwnerError::Operation)?;
                    return Ok(LiveMotionOwnerOutcome::AutonomousCompleted { mode });
                }
                Ok(false) => {
                    self.autonomous = LiveAutonomousLifecycle::Active(session);
                    return Ok(LiveMotionOwnerOutcome::PeriodicAutonomousApplied { mode });
                }
                Err(primary) => {
                    let operation = self.cleanup_active_autonomous_failure(session, primary);
                    return Err(LiveMotionOwnerError::Operation(operation));
                }
            }
        }
        if blocked {
            self.release_autonomous_session(
                session,
                tick,
                LiveLifecycleZeroReason::AutonomousRelease,
            )
            .map_err(LiveMotionOwnerError::Operation)?;
            Ok(LiveMotionOwnerOutcome::PeriodicAutonomousStopped { mode })
        } else {
            self.autonomous = LiveAutonomousLifecycle::Active(session);
            Ok(LiveMotionOwnerOutcome::PeriodicAutonomousApplied { mode })
        }
    }

    fn current_goal_reached(
        &self,
        arrival_tolerance_m: f64,
    ) -> Result<bool, LiveMotionOperationError<P::Error, J::Error>> {
        let retained = self
            .retained_map
            .as_ref()
            .ok_or(LiveMotionOperationError::RetainedMapUnavailable)?;
        let goal = self
            .coordinator
            .current_goal()
            .ok_or(LiveMotionOperationError::AutonomousGoalUnavailable)?;
        let start = self
            .coordinator
            .plan_start_for_snapshot(&retained.snapshot)
            .map_err(LiveMotionOperationError::FrontierStart)?;
        let delta_x_m = start.point().x_m() - goal.point().x_m();
        let delta_y_m = start.point().y_m() - goal.point().y_m();
        Ok(delta_x_m.hypot(delta_y_m) <= arrival_tolerance_m)
    }

    fn advance_frontier(
        &mut self,
        session: &mut LiveAutonomousSession,
        now: HostMonotonicTimestamp,
        reason: FrontierAdvanceReason,
    ) -> Result<bool, LiveMotionOperationError<P::Error, J::Error>> {
        let LiveAutonomousExecution::Frontier(frontier) = &mut session.execution else {
            return Err(LiveMotionOperationError::AutonomousGoalUnavailable);
        };
        let LiveFrontierExecution {
            config,
            goals_started,
            next_scan_sequence,
            attempted_scan_directions,
            phase,
            ..
        } = frontier.as_mut();
        let previous_yaw = match phase {
            LiveFrontierPhase::InPlaceYaw { command } => Some(*command),
            LiveFrontierPhase::PointGoal => None,
        };
        if matches!(reason, FrontierAdvanceReason::PointGoalReached) {
            *attempted_scan_directions = 0;
        }
        let left_direct_mode = previous_yaw.is_some();
        if left_direct_mode {
            self.fresh_zero(now, LiveLifecycleZeroReason::FrontierPhaseTransition)?;
            self.coordinator
                .leave_direct_mode(nonzero_lease(session.authority.lease().id()))
                .map_err(LiveMotionOperationError::CoordinatorMode)?;
        } else {
            self.coordinator.clear_goal();
        }
        let retained = self
            .retained_map
            .as_ref()
            .ok_or(LiveMotionOperationError::RetainedMapUnavailable)?;
        let start = self
            .coordinator
            .plan_start_for_snapshot(&retained.snapshot)
            .map_err(LiveMotionOperationError::FrontierStart)?;
        let selection = {
            let mut explorer = NanoBoundaryFrontierExplorer::try_new(
                &retained.snapshot,
                config.explorer(),
                config.boundary_m(),
            )
            .map_err(LiveMotionOperationError::FrontierBuild)?;
            explorer
                .select(start)
                .map_err(LiveMotionOperationError::FrontierSearch)?
        };
        match selection {
            FrontierSearchOutcome::NoReachableFrontier { .. } => Ok(true),
            FrontierSearchOutcome::Selected(frontier) => {
                if *goals_started == config.maximum_frontier_goals().get() {
                    return Ok(true);
                }
                self.coordinator
                    .select_frontier_goal(now, frontier, &retained.snapshot)
                    .map_err(|source| {
                        LiveMotionOperationError::CoordinatorAdmission(Box::new(source))
                    })?;
                *goals_started += 1;
                *attempted_scan_directions = 0;
                *phase = LiveFrontierPhase::PointGoal;
                Ok(false)
            }
            FrontierSearchOutcome::InPlaceScanRequired(scan) => {
                let preserved_target = previous_yaw
                    .filter(|_| matches!(reason, FrontierAdvanceReason::MapUpdated))
                    .map(FrontierYawScanCommandV1::target_direction)
                    .filter(|target| scan.unknown_directions().contains(*target));
                let (target_direction, started_at, budget, newly_attempted) =
                    if let Some(target_direction) = preserved_target {
                        let previous =
                            previous_yaw.expect("preserved target requires a prior yaw command");
                        (
                            target_direction,
                            previous.started_at(),
                            previous.budget(),
                            false,
                        )
                    } else {
                        let Some(target_direction) =
                            next_unattempted_scan_direction(scan, *attempted_scan_directions)
                        else {
                            return Ok(true);
                        };
                        (target_direction, now, config.yaw_scan_budget(), true)
                    };
                let sequence = *next_scan_sequence;
                *next_scan_sequence = next_scan_sequence
                    .checked_add(1)
                    .ok_or(LiveMotionOperationError::FrontierScanSequenceExhausted)?;
                let command = FrontierYawScanCommandV1::try_new(
                    session.authority.lease().id(),
                    sequence,
                    scan,
                    target_direction,
                    config.yaw_turn_direction(),
                    started_at,
                    HostMonotonicTimestamp::from_nanos(
                        session.authority.lease().expires_at_exclusive().as_nanos(),
                    ),
                    budget,
                )
                .map_err(LiveMotionOperationError::FrontierYawCommand)?;
                if newly_attempted {
                    *attempted_scan_directions |= frontier_direction_bit(target_direction);
                }
                if !left_direct_mode {
                    self.fresh_zero(now, LiveLifecycleZeroReason::FrontierPhaseTransition)?;
                }
                self.coordinator
                    .enter_frontier_yaw_mode(nonzero_lease(session.authority.lease().id()))
                    .map_err(LiveMotionOperationError::CoordinatorMode)?;
                *phase = LiveFrontierPhase::InPlaceYaw { command };
                Ok(false)
            }
        }
    }

    fn cleanup_failed_autonomous_start(
        &mut self,
        authority: AgentAutonomousAuthority,
        execution: LiveAutonomousExecution,
        primary: LiveMotionOperationError<P::Error, J::Error>,
    ) -> LiveMotionOperationError<P::Error, J::Error> {
        let session = LiveAutonomousSession {
            authority,
            execution,
        };
        self.cleanup_active_autonomous_failure(session, primary)
    }

    fn cleanup_active_autonomous_failure(
        &mut self,
        session: LiveAutonomousSession,
        primary: LiveMotionOperationError<P::Error, J::Error>,
    ) -> LiveMotionOperationError<P::Error, J::Error> {
        if matches!(&primary, LiveMotionOperationError::ActuationFault { .. }) {
            let coordinator_cleanup = self.clear_autonomous_coordinator_mode(&session);
            self.autonomous = LiveAutonomousLifecycle::Inactive;
            return match coordinator_cleanup {
                Ok(()) => primary,
                Err(cleanup) => LiveMotionOperationError::PrimaryAndCleanup {
                    primary: Box::new(primary),
                    cleanup: Box::new(cleanup),
                },
            };
        }
        self.autonomous = LiveAutonomousLifecycle::Active(session);
        match self.stop_autonomous(LiveLifecycleZeroReason::FaultContainment) {
            Ok(()) => primary,
            Err(cleanup) => LiveMotionOperationError::PrimaryAndCleanup {
                primary: Box::new(primary),
                cleanup: Box::new(cleanup),
            },
        }
    }

    fn release_autonomous_session(
        &mut self,
        session: LiveAutonomousSession,
        requested_at: HostMonotonicTimestamp,
        zero_reason: LiveLifecycleZeroReason,
    ) -> Result<(), LiveMotionOperationError<P::Error, J::Error>> {
        let pending = match self
            .dispatcher
            .begin_autonomous_release(&session.authority, requested_at)
        {
            Ok(pending) => pending,
            Err(source) => {
                if matches!(
                    &source,
                    AgentAutonomousControlError::UnexpectedSupervisorAction { .. }
                ) {
                    self.autonomous = LiveAutonomousLifecycle::Inactive;
                } else {
                    self.autonomous = LiveAutonomousLifecycle::Active(session);
                }
                return Err(LiveMotionOperationError::Autonomous(source));
            }
        };
        self.finish_autonomous_stop_transition(&session, pending, requested_at, zero_reason)
    }

    fn finish_autonomous_stop_transition(
        &mut self,
        session: &LiveAutonomousSession,
        pending: PendingAgentAutonomousStop,
        requested_at: HostMonotonicTimestamp,
        zero_reason: LiveLifecycleZeroReason,
    ) -> Result<(), LiveMotionOperationError<P::Error, J::Error>> {
        self.autonomous = LiveAutonomousLifecycle::PendingStop(pending);
        let coordinator_cleanup = self.clear_autonomous_coordinator_mode(session);
        let pending =
            match std::mem::replace(&mut self.autonomous, LiveAutonomousLifecycle::Inactive) {
                LiveAutonomousLifecycle::PendingStop(pending) => pending,
                _ => unreachable!("release retains its exact stop continuation"),
            };
        let stop = self.complete_pending_autonomous_stop(pending, requested_at, zero_reason);
        match (coordinator_cleanup, stop) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(primary), Ok(())) => Err(primary),
            (Ok(()), Err(cleanup)) => Err(cleanup),
            (Err(primary), Err(cleanup)) => Err(LiveMotionOperationError::PrimaryAndCleanup {
                primary: Box::new(primary),
                cleanup: Box::new(cleanup),
            }),
        }
    }

    fn clear_autonomous_coordinator_mode(
        &mut self,
        session: &LiveAutonomousSession,
    ) -> Result<(), LiveMotionOperationError<P::Error, J::Error>> {
        match self.coordinator.motion_mode() {
            CoordinatorMotionModeV1::FrontierInPlaceYaw { authority_lease_id } => {
                let expected = nonzero_lease(session.authority.lease().id());
                if authority_lease_id != expected {
                    return Err(LiveMotionOperationError::CoordinatorMode(
                        CoordinatorMotionModeError::AuthorityLeaseMismatch {
                            bound: authority_lease_id,
                            supplied: expected,
                        },
                    ));
                }
                self.coordinator
                    .leave_direct_mode(expected)
                    .map_err(LiveMotionOperationError::CoordinatorMode)?;
                self.coordinator.clear_goal();
                Ok(())
            }
            CoordinatorMotionModeV1::MappingOnly | CoordinatorMotionModeV1::PointGoal => {
                self.coordinator.clear_goal();
                Ok(())
            }
            actual @ CoordinatorMotionModeV1::Manual { .. } => {
                Err(LiveMotionOperationError::CoordinatorMode(
                    CoordinatorMotionModeError::NotDirectControl { actual },
                ))
            }
        }
    }

    fn cancel_and_stop_pending_grant(
        &mut self,
        pending: PendingAgentAutonomousGrant,
        requested_at: HostMonotonicTimestamp,
        zero_reason: LiveLifecycleZeroReason,
    ) -> Result<(), LiveMotionOperationError<P::Error, J::Error>> {
        self.autonomous = LiveAutonomousLifecycle::PendingGrant(pending);
        let pending_stop = {
            let pending = match &self.autonomous {
                LiveAutonomousLifecycle::PendingGrant(pending) => pending,
                _ => unreachable!("pending grant retained for cancellation"),
            };
            match self
                .dispatcher
                .cancel_pending_autonomous_grant(pending, requested_at)
            {
                Ok(pending_stop) => pending_stop,
                Err(source) => {
                    if matches!(
                        &source,
                        AgentAutonomousControlError::UnexpectedSupervisorAction { .. }
                    ) {
                        self.autonomous = LiveAutonomousLifecycle::Inactive;
                    }
                    return Err(LiveMotionOperationError::Autonomous(source));
                }
            }
        };
        self.autonomous = LiveAutonomousLifecycle::PendingStop(pending_stop);
        let pending_stop =
            match std::mem::replace(&mut self.autonomous, LiveAutonomousLifecycle::Inactive) {
                LiveAutonomousLifecycle::PendingStop(pending) => pending,
                _ => unreachable!("cancelled grant retains its stop continuation"),
            };
        self.complete_pending_autonomous_stop(pending_stop, requested_at, zero_reason)
    }

    fn complete_pending_autonomous_stop(
        &mut self,
        pending: PendingAgentAutonomousStop,
        requested_at: HostMonotonicTimestamp,
        zero_reason: LiveLifecycleZeroReason,
    ) -> Result<(), LiveMotionOperationError<P::Error, J::Error>> {
        self.autonomous = LiveAutonomousLifecycle::PendingStop(pending);
        let (result, observed_at) = match self.fresh_zero(requested_at, zero_reason) {
            Ok(evidence) => evidence,
            Err(source) => {
                if matches!(&source, LiveMotionOperationError::ActuationFault { .. }) {
                    self.autonomous = LiveAutonomousLifecycle::Inactive;
                }
                return Err(source);
            }
        };
        let pending = match &self.autonomous {
            LiveAutonomousLifecycle::PendingStop(pending) => pending,
            _ => unreachable!("stop continuation retained until exact completion"),
        };
        match self.dispatcher.complete_autonomous_stop_with_applied_zero(
            pending,
            result,
            observed_at,
            observed_at,
        ) {
            Ok(()) => {
                self.autonomous = LiveAutonomousLifecycle::Inactive;
                Ok(())
            }
            Err(source) => {
                if matches!(
                    &source,
                    AgentAutonomousControlError::UnexpectedSupervisorAction { .. }
                ) {
                    self.autonomous = LiveAutonomousLifecycle::Inactive;
                }
                Err(LiveMotionOperationError::Autonomous(source))
            }
        }
    }

    fn stop_autonomous(
        &mut self,
        zero_reason: LiveLifecycleZeroReason,
    ) -> Result<(), LiveMotionOperationError<P::Error, J::Error>> {
        if matches!(self.autonomous, LiveAutonomousLifecycle::Inactive) {
            return Ok(());
        }
        let requested_at = self.now()?;
        let lifecycle = std::mem::replace(&mut self.autonomous, LiveAutonomousLifecycle::Inactive);
        match lifecycle {
            LiveAutonomousLifecycle::Inactive => unreachable!("autonomous state checked above"),
            LiveAutonomousLifecycle::PendingGrant(pending) => {
                self.cancel_and_stop_pending_grant(pending, requested_at, zero_reason)
            }
            LiveAutonomousLifecycle::PendingStop(pending) => {
                self.complete_pending_autonomous_stop(pending, requested_at, zero_reason)
            }
            LiveAutonomousLifecycle::Active(session) => {
                self.release_autonomous_session(session, requested_at, zero_reason)
            }
        }
    }

    fn reject_coordinator_admission(
        &self,
        claimed: AgentControlClaimedRequest,
        source: CoordinatorAdmissionError<J::Error>,
    ) -> Result<LiveMotionOwnerOutcome, LiveMotionOwnerError<P::Error, J::Error>> {
        let expected = match &source {
            CoordinatorAdmissionError::NoCurrentMapForGoal => {
                Some((AgentControlRejectionCodeV1::MapUnavailable, true))
            }
            CoordinatorAdmissionError::GoalMapEpochMismatch { .. }
            | CoordinatorAdmissionError::GoalDisplayedRevisionMismatch { .. }
            | CoordinatorAdmissionError::GoalSnapshotMapMismatch { .. }
            | CoordinatorAdmissionError::GoalSnapshotRevisionMismatch { .. }
            | CoordinatorAdmissionError::FrontierGoalMapMismatch { .. }
            | CoordinatorAdmissionError::FrontierGoalRevisionMismatch { .. }
            | CoordinatorAdmissionError::PointGoalPreparationStale { .. } => {
                Some((AgentControlRejectionCodeV1::StaleMapSelection, true))
            }
            CoordinatorAdmissionError::PointGoalModeConflict { .. } => {
                Some((AgentControlRejectionCodeV1::ModeConflict, false))
            }
            CoordinatorAdmissionError::Plan(_)
            | CoordinatorAdmissionError::FrontierGoalMissingTraversalBoundary => {
                Some((AgentControlRejectionCodeV1::NotReady, true))
            }
            CoordinatorAdmissionError::Latched(_)
            | CoordinatorAdmissionError::Boundary(_)
            | CoordinatorAdmissionError::Journal(_)
            | CoordinatorAdmissionError::ReplayClock(_)
            | CoordinatorAdmissionError::SegmentIdExhausted
            | CoordinatorAdmissionError::MapRevisionNotIncreasing { .. } => None,
        };
        if let Some((code, retryable)) = expected {
            return self.reject_without_operation(claimed, code, retryable);
        }
        self.reject_operation(
            claimed,
            LiveMotionOperationError::CoordinatorAdmission(Box::new(source)),
            AgentControlRejectionCodeV1::InternalFault,
            false,
        )
    }

    fn arm(
        &mut self,
        claimed: AgentControlClaimedRequest,
    ) -> Result<LiveMotionOwnerOutcome, LiveMotionOwnerError<P::Error, J::Error>> {
        let operation = (|| {
            let now = self.now()?;
            self.dispatcher
                .begin_arm(now)
                .map_err(LiveMotionOperationError::Manual)?;
            let (result, observed_at) =
                self.fresh_zero(now, LiveLifecycleZeroReason::ArmAdmission)?;
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
    ) -> Result<LiveMotionOwnerOutcome, LiveMotionOwnerError<P::Error, J::Error>> {
        let operation = (|| {
            let lease = match transition {
                BeginManualTransition::Granted { lease } => {
                    // A console authority token is completed only from evidence
                    // correlated to that exact typed request. The supervisor may
                    // grant from a still-fresh earlier zero, which is sufficient
                    // for its internal admission but cannot prove this console
                    // request crossed the physical boundary. Apply one fresh,
                    // request-bound zero before exposing the authority.
                    #[cfg(feature = "operator-console")]
                    if self.processing_typed_request_key.is_some() {
                        let requested_at = self.now()?;
                        self.fresh_zero(requested_at, LiveLifecycleZeroReason::ManualAdmission)?;
                    }
                    lease
                }
                BeginManualTransition::FreshAppliedZeroRequired => {
                    let requested_at = self.now()?;
                    let (result, observed_at) =
                        self.fresh_zero(requested_at, LiveLifecycleZeroReason::ManualAdmission)?;
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

    fn request_save_map(
        &self,
        claimed: AgentControlClaimedRequest,
    ) -> Result<LiveMotionOwnerOutcome, LiveMotionOwnerError<P::Error, J::Error>> {
        let manual_inactive = matches!(
            self.dispatcher.manual().global_stop_requirement(),
            AgentManualGlobalStopRequirement::NoManualTransition
        );
        let autonomous_inactive = matches!(&self.autonomous, LiveAutonomousLifecycle::Inactive);
        let coordinator_mapping_only =
            self.coordinator.motion_mode() == CoordinatorMotionModeV1::MappingOnly;
        if manual_inactive && autonomous_inactive && coordinator_mapping_only {
            Ok(LiveMotionOwnerOutcome::SaveMapRequested { claimed })
        } else {
            // Durable encoding, quota scans, rename, and fsync run
            // synchronously. They are admitted only while no motion lifecycle
            // depends on this thread for MPC/deadman refresh.
            self.reject_without_operation(claimed, AgentControlRejectionCodeV1::ModeConflict, true)
        }
    }

    fn manual_command(
        &mut self,
        claimed: AgentControlClaimedRequest,
        output: ManualDriveOutput<AuthorityLeaseId>,
    ) -> Result<LiveMotionOwnerOutcome, LiveMotionOwnerError<P::Error, J::Error>> {
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
        #[cfg(feature = "operator-console")]
        let mut applied = applied;
        #[cfg(feature = "operator-console")]
        applied.bind_typed_request_key(self.processing_typed_request_key);
        let stopped = applied.stopped();
        self.last_physical_state = Some(LivePhysicalStateEvent::CoordinatorTick(applied));
        if stopped || output.target().is_stop() {
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
    ) -> Result<LiveMotionOwnerOutcome, LiveMotionOwnerError<P::Error, J::Error>> {
        let zero_reason = match safety {
            LiveMotionCompletedSafetyAction::MappingOnlyStopped => {
                LiveLifecycleZeroReason::MappingOnlyRequest
            }
            LiveMotionCompletedSafetyAction::GlobalStopped => {
                LiveLifecycleZeroReason::GlobalStopRequest
            }
            _ => LiveLifecycleZeroReason::FaultContainment,
        };
        match self.stop_all_motion(manual, zero_reason) {
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
    ) -> Result<LiveMotionOwnerOutcome, LiveMotionOwnerError<P::Error, J::Error>> {
        let operation = (|| {
            self.stop_all_motion(manual, LiveLifecycleZeroReason::DisarmRequest)?;
            let requested_at = self.now()?;
            let action = self
                .dispatcher
                .begin_disarm(requested_at)
                .map_err(LiveMotionOperationError::Manual)?;
            if matches!(action, SupervisorAction::Disarmed) {
                return Ok(());
            }
            let (result, observed_at) =
                self.fresh_zero(requested_at, LiveLifecycleZeroReason::DisarmRequest)?;
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
        zero_reason: LiveLifecycleZeroReason,
    ) -> Result<(), LiveMotionOperationError<P::Error, J::Error>> {
        let had_autonomous = !matches!(self.autonomous, LiveAutonomousLifecycle::Inactive);
        self.stop_autonomous(zero_reason)?;
        if had_autonomous && matches!(manual, AgentManualGlobalStopRequirement::NoManualTransition)
        {
            self.coordinator.clear_goal();
            return Ok(());
        }
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
                        && let Err(source) = self.coordinator.leave_direct_mode(authority_lease_id)
                    {
                        let cleanup = LiveMotionOperationError::CoordinatorMode(source);
                        let primary =
                            LiveMotionOperationError::ActiveAuthorityOutsideManualLifecycle {
                                mode: active.mode(),
                                latch,
                            };
                        self.coordinator.clear_goal();
                        let primary = primary_with_cleanup(primary, cleanup);
                        return match self.fresh_zero(
                            fault_at,
                            LiveLifecycleZeroReason::CoordinatorInvariantRecovery,
                        ) {
                            Ok(_) => Err(primary),
                            Err(cleanup) => Err(primary_with_cleanup(primary, cleanup)),
                        };
                    }
                    self.coordinator.clear_goal();
                    let primary = LiveMotionOperationError::ActiveAuthorityOutsideManualLifecycle {
                        mode: active.mode(),
                        latch,
                    };
                    return match self.fresh_zero(
                        fault_at,
                        LiveLifecycleZeroReason::CoordinatorInvariantRecovery,
                    ) {
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
                    self.fresh_zero(
                        requested_at,
                        LiveLifecycleZeroReason::CoordinatorInvariantRecovery,
                    )?;
                    return Err(
                        LiveMotionOperationError::CoordinatorDirectWithoutAuthority { mode },
                    );
                }
                self.coordinator.clear_goal();
                let requested_at = self.now()?;
                self.fresh_zero(requested_at, zero_reason)?;
                Ok(())
            }
            AgentManualGlobalStopRequirement::PendingBeginMustBeCancelled => {
                let requested_at = self.now()?;
                self.dispatcher
                    .manual_mut()
                    .cancel_pending_begin(requested_at)
                    .map_err(LiveMotionOperationError::Manual)?;
                self.coordinator.clear_goal();
                let (result, observed_at) = self.fresh_zero(requested_at, zero_reason)?;
                self.dispatcher
                    .manual_mut()
                    .complete_cancelled_begin_with_applied_zero(result, observed_at, observed_at)
                    .map_err(LiveMotionOperationError::Manual)
            }
            AgentManualGlobalStopRequirement::FreshAppliedCancelledBeginZeroRequired => {
                let requested_at = self.now()?;
                self.coordinator.clear_goal();
                let (result, observed_at) = self.fresh_zero(requested_at, zero_reason)?;
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
                let (result, observed_at) = self.fresh_zero(requested_at, zero_reason)?;
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
                let (result, observed_at) = self.fresh_zero(requested_at, zero_reason)?;
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
    ) -> Result<(), LiveMotionOperationError<P::Error, J::Error>> {
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
        self.stop_all_motion(requirement, LiveLifecycleZeroReason::ManualRelease)
    }

    fn cleanup_after_begin_failure(
        &mut self,
        primary: LiveMotionOperationError<P::Error, J::Error>,
    ) -> LiveMotionOperationError<P::Error, J::Error> {
        let requirement = self.dispatcher.manual().global_stop_requirement();
        if matches!(
            requirement,
            AgentManualGlobalStopRequirement::NoManualTransition
        ) {
            return primary;
        }
        match self.stop_all_motion(requirement, LiveLifecycleZeroReason::FaultContainment) {
            Ok(()) => primary,
            Err(cleanup) => LiveMotionOperationError::PrimaryAndCleanup {
                primary: Box::new(primary),
                cleanup: Box::new(cleanup),
            },
        }
    }

    fn reject_unready_manual_start(
        &mut self,
        claimed: AgentControlClaimedRequest,
        readiness: CoordinatorTickBlocker,
    ) -> Result<LiveMotionOwnerOutcome, LiveMotionOwnerError<P::Error, J::Error>> {
        let requirement = self.dispatcher.manual().global_stop_requirement();
        if matches!(
            requirement,
            AgentManualGlobalStopRequirement::NoManualTransition
        ) {
            return self.reject_without_operation(
                claimed,
                AgentControlRejectionCodeV1::NotReady,
                true,
            );
        }
        match self.stop_all_motion(requirement, LiveLifecycleZeroReason::FaultContainment) {
            Ok(()) => {
                self.reject_without_operation(claimed, AgentControlRejectionCodeV1::NotReady, true)
            }
            Err(cleanup) => self.reject_operation(
                claimed,
                LiveMotionOperationError::PrimaryAndCleanup {
                    primary: Box::new(LiveMotionOperationError::MotionStartReadiness(readiness)),
                    cleanup: Box::new(cleanup),
                },
                AgentControlRejectionCodeV1::InternalFault,
                false,
            ),
        }
    }

    fn cleanup_after_tick_failure(
        &mut self,
        primary: LiveMotionOperationError<P::Error, J::Error>,
        lease_id: AuthorityLeaseId,
    ) -> LiveMotionOperationError<P::Error, J::Error> {
        if matches!(&primary, LiveMotionOperationError::ActuationFault { .. }) {
            let cleanup = self.leave_manual_mode_if_bound(lease_id).err();
            self.coordinator.clear_goal();
            return match cleanup {
                Some(cleanup) => primary_with_cleanup(primary, cleanup),
                None => primary,
            };
        }
        let requirement = self.dispatcher.manual().global_stop_requirement();
        match self.stop_all_motion(requirement, LiveLifecycleZeroReason::FaultContainment) {
            Ok(()) => primary,
            Err(cleanup) => LiveMotionOperationError::PrimaryAndCleanup {
                primary: Box::new(primary),
                cleanup: Box::new(cleanup),
            },
        }
    }

    fn stop_after_unadmitted_fault(
        &mut self,
        primary: LiveMotionOperationError<P::Error, J::Error>,
    ) -> LiveMotionOperationError<P::Error, J::Error> {
        let mut primary = primary;
        if let CoordinatorMotionModeV1::Manual { authority_lease_id }
        | CoordinatorMotionModeV1::FrontierInPlaceYaw { authority_lease_id } =
            self.coordinator.motion_mode()
            && let Err(source) = self.coordinator.leave_direct_mode(authority_lease_id)
        {
            primary =
                primary_with_cleanup(primary, LiveMotionOperationError::CoordinatorMode(source));
        }
        self.coordinator.clear_goal();
        let requested_at = match self.now() {
            Ok(now) => now,
            Err(source) => {
                return primary_with_cleanup(primary, LiveMotionOperationError::Clock(source));
            }
        };
        match self.fresh_zero(requested_at, LiveLifecycleZeroReason::FaultContainment) {
            Ok(_) => primary,
            Err(cleanup) => primary_with_cleanup(primary, cleanup),
        }
    }

    fn fresh_zero(
        &mut self,
        requested_at: HostMonotonicTimestamp,
        reason: LiveLifecycleZeroReason,
    ) -> Result<
        (HostCommandResult, HostMonotonicTimestamp),
        LiveMotionOperationError<P::Error, J::Error>,
    > {
        let receipt = match self.actuation.apply_fresh_zero() {
            Ok(receipt) => receipt,
            Err(source) => return Err(self.latch_actuation_fault(source, requested_at)),
        };
        let result = receipt.verified_host_result();
        self.last_physical_state = Some(LivePhysicalStateEvent::LifecycleZero(
            LiveLifecycleZeroApplied {
                requested_at,
                receipt,
                reason,
                #[cfg(feature = "operator-console")]
                typed_request_key: self.processing_typed_request_key,
            },
        ));
        let observed_at = self
            .now()
            .map_err(|source| LiveMotionOperationError::AppliedZeroClock { source, result })?;
        Ok((result, observed_at))
    }

    fn port_tick_error(
        &mut self,
        source: LiveMotionPortTickError<P::Error>,
        fault_at: HostMonotonicTimestamp,
    ) -> LiveMotionOperationError<P::Error, J::Error> {
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
    ) -> LiveMotionOperationError<P::Error, J::Error> {
        let evidence = P::classify_error(&source);
        self.last_actuation_fault = Some(evidence);
        self.last_physical_state = Some(LivePhysicalStateEvent::ActuationFault {
            observed_at: fault_at,
            evidence,
        });
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
    ) -> Result<(), LiveMotionOperationError<P::Error, J::Error>> {
        self.coordinator
            .leave_direct_mode(nonzero_lease(lease_id))
            .map_err(LiveMotionOperationError::CoordinatorMode)
    }

    fn leave_manual_mode_for_release(
        &mut self,
        lease_id: AuthorityLeaseId,
    ) -> Result<(), LiveMotionOperationError<P::Error, J::Error>> {
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

    fn leave_manual_mode_if_bound(
        &mut self,
        lease_id: AuthorityLeaseId,
    ) -> Result<(), LiveMotionOperationError<P::Error, J::Error>> {
        if self.coordinator.motion_mode()
            == (CoordinatorMotionModeV1::Manual {
                authority_lease_id: nonzero_lease(lease_id),
            })
        {
            self.coordinator
                .leave_direct_mode(nonzero_lease(lease_id))
                .map_err(LiveMotionOperationError::CoordinatorMode)?;
        }
        Ok(())
    }

    fn now(&mut self) -> Result<HostMonotonicTimestamp, HostMonotonicClockReadError> {
        self.clock.try_now()
    }

    fn respond_completed(
        &self,
        claimed: AgentControlClaimedRequest,
        safety: LiveMotionCompletedSafetyAction,
        outcome: LiveMotionOwnerOutcome,
    ) -> Result<LiveMotionOwnerOutcome, LiveMotionOwnerError<P::Error, J::Error>> {
        claimed
            .respond_completed()
            .map(|()| outcome)
            .map_err(|response| LiveMotionOwnerError::ResponseAfterSafety { safety, response })
    }

    fn respond_shutdown_completed(
        &self,
        claimed: AgentControlClaimedRequest,
    ) -> Result<LiveMotionOwnerOutcome, LiveMotionOwnerError<P::Error, J::Error>> {
        claimed
            .respond_completed_after_wire_delivery()
            .map(|()| LiveMotionOwnerOutcome::ShutdownRequested)
            .map_err(|response| LiveMotionOwnerError::ResponseAfterSafety {
                safety: LiveMotionCompletedSafetyAction::ShutdownStopped,
                response,
            })
    }

    fn respond_accepted(
        &self,
        claimed: AgentControlClaimedRequest,
        safety: LiveMotionCompletedSafetyAction,
        outcome: LiveMotionOwnerOutcome,
    ) -> Result<LiveMotionOwnerOutcome, LiveMotionOwnerError<P::Error, J::Error>> {
        claimed
            .respond_accepted_for_processing()
            .map(|_| outcome)
            .map_err(|response| LiveMotionOwnerError::ResponseAfterSafety { safety, response })
    }

    fn reject_after_safety(
        &self,
        claimed: AgentControlClaimedRequest,
        safety: LiveMotionCompletedSafetyAction,
        code: AgentControlRejectionCodeV1,
        retryable: bool,
    ) -> Result<LiveMotionOwnerOutcome, LiveMotionOwnerError<P::Error, J::Error>> {
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
    ) -> Result<LiveMotionOwnerOutcome, LiveMotionOwnerError<P::Error, J::Error>> {
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
        operation: LiveMotionOperationError<P::Error, J::Error>,
        code: AgentControlRejectionCodeV1,
        retryable: bool,
    ) -> Result<LiveMotionOwnerOutcome, LiveMotionOwnerError<P::Error, J::Error>> {
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

const fn frontier_direction_bit(direction: FrontierUnknownDirection) -> u8 {
    match direction {
        FrontierUnknownDirection::NegativeMapY => 1 << 0,
        FrontierUnknownDirection::NegativeMapX => 1 << 1,
        FrontierUnknownDirection::PositiveMapX => 1 << 2,
        FrontierUnknownDirection::PositiveMapY => 1 << 3,
    }
}

fn next_unattempted_scan_direction(
    scan: FrontierInPlaceScan,
    attempted: u8,
) -> Option<FrontierUnknownDirection> {
    scan.unknown_directions()
        .iter()
        .find(|direction| attempted & frontier_direction_bit(*direction) == 0)
}

fn autonomous_deadline<E, J>(
    started_at: HostMonotonicTimestamp,
    maximum_runtime: std::time::Duration,
) -> Result<HostMonotonicTimestamp, LiveMotionOperationError<E, J>> {
    let maximum_runtime_ns = u64::try_from(maximum_runtime.as_nanos()).map_err(|_| {
        LiveMotionOperationError::AutonomousDeadlineOverflow {
            started_at,
            maximum_runtime_ns: u64::MAX,
        }
    })?;
    let deadline_ns = started_at
        .as_nanos()
        .checked_add(maximum_runtime_ns)
        .ok_or(LiveMotionOperationError::AutonomousDeadlineOverflow {
            started_at,
            maximum_runtime_ns,
        })?;
    Ok(HostMonotonicTimestamp::from_nanos(deadline_ns))
}

fn manual_output_lease(output: ManualDriveOutput<AuthorityLeaseId>) -> AuthorityLeaseId {
    match output {
        ManualDriveOutput::Accepted(accepted) => accepted.authority_lease_id(),
        ManualDriveOutput::Stopped(stopped) => stopped.bound_authority_lease_id(),
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

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
        enqueue_agent_control_test_request_with_failed_wire_delivery,
    };
    use super::super::mpc::HostMonotonicClockReadError;
    use super::*;
    use crate::dense::occupancy::{
        DepthCameraModel, DepthToTrackingCamera, OccupancyCell, OccupancyGridGeometry,
    };
    use crate::map::SlamMap;
    #[cfg(feature = "operator-console")]
    use crate::navigation::AgentRuntimeStateV1;
    use crate::navigation::{
        AGENT_CONTROL_SCHEMA_V1, AgentAuthoritySupervisor, AgentControlCommandKindV1,
        AgentControlCompletionV1, AgentControlMonotonicOrigin, AgentControlRequestParser,
        AgentControlResponseKindV1, AgentControlRuntimeQueueCapacity, AgentControlRuntimeSender,
        AgentManualControlCore, AgentManualRuntimePolicy, MANUAL_DRIVE_CONFIG_V1,
        ManualDriveConfigV1, ManualDriveConfigV1Dto, NavigationClockEpoch,
        NavigationIngressCapacity, NavigationIngressLog, NavigationRecordingId,
        PathReferenceBuilderV1, PendingVisualAttemptIngress, PlanarOdometry,
        ShadowNavigationConfigV1, ShadowSafetySupervisor, VisualAttemptOutcome,
        agent_control_runtime_queue,
    };
    use crate::{
        DeviceSessionId, Frame, FrameDimensions, FrameId, MapLocalization, PairingWindowNs,
        PinholeIntrinsics, Pose, SensorId, StereoObservation, StereoPair, VisualFrameStamp,
        WorldToCamera,
    };

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

    fn live_policy() -> NanoLiveModePolicy {
        live_policy_with_point_goal_runtime(Duration::from_secs(60))
    }

    fn live_policy_with_point_goal_runtime(maximum_runtime: Duration) -> NanoLiveModePolicy {
        NanoLiveModePolicy::autonomous_for_test(
            duration(1_000),
            maximum_runtime,
            0.1,
            NonZeroU32::new(2).expect("nonzero frontier goal budget"),
        )
    }

    #[derive(Clone)]
    struct ScriptedClock {
        next_ns: Arc<AtomicU64>,
        fail_next: Arc<AtomicBool>,
    }

    impl ScriptedClock {
        fn new(nanos: u64) -> Self {
            Self {
                next_ns: Arc::new(AtomicU64::new(nanos)),
                fail_next: Arc::new(AtomicBool::new(false)),
            }
        }

        fn peek(&self) -> HostMonotonicTimestamp {
            at(self.next_ns.load(Ordering::SeqCst))
        }

        fn advance(&self, nanos: u64) {
            self.next_ns.fetch_add(nanos, Ordering::SeqCst);
        }

        fn set(&self, nanos: u64) {
            self.next_ns.store(nanos, Ordering::SeqCst);
        }

        fn fail_next(&self) {
            self.fail_next.store(true, Ordering::SeqCst);
        }
    }

    impl HostMonotonicClock for ScriptedClock {
        fn try_now(&mut self) -> Result<HostMonotonicTimestamp, HostMonotonicClockReadError> {
            if self.fail_next.swap(false, Ordering::SeqCst) {
                return Err(HostMonotonicClockReadError::ElapsedNanosecondsOutOfRange {
                    elapsed_nanoseconds: u128::from(u64::MAX) + 1,
                });
            }
            Ok(at(self.next_ns.fetch_add(1, Ordering::SeqCst)))
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum FakeError {
        Injected,
        InjectedWithConfirmedStop,
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
        PointGoalTick {
            blocked: bool,
        },
        FrontierYawTick {
            target_direction: FrontierUnknownDirection,
            map_revision: u64,
            started_at: HostMonotonicTimestamp,
            valid_through_exclusive: HostMonotonicTimestamp,
            blocked: bool,
            target_reached: bool,
        },
    }

    #[derive(Clone)]
    struct FakePortState {
        events: Arc<Mutex<Vec<FakeEvent>>>,
        fail_next_zero: Arc<AtomicBool>,
        fail_next_tick: Arc<AtomicBool>,
        fail_disarm: Arc<AtomicBool>,
        fail_disarm_with_confirmed_stop: Arc<AtomicBool>,
        force_tick_stop: Arc<AtomicBool>,
        force_yaw_target_reached: Arc<AtomicBool>,
    }

    impl FakePortState {
        fn new() -> Self {
            Self {
                events: Arc::new(Mutex::new(Vec::new())),
                fail_next_zero: Arc::new(AtomicBool::new(false)),
                fail_next_tick: Arc::new(AtomicBool::new(false)),
                fail_disarm: Arc::new(AtomicBool::new(false)),
                fail_disarm_with_confirmed_stop: Arc::new(AtomicBool::new(false)),
                force_tick_stop: Arc::new(AtomicBool::new(false)),
                force_yaw_target_reached: Arc::new(AtomicBool::new(false)),
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
        type Diagnostic = ();
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
            tick: HostMonotonicTimestamp,
            command: ManualDriveOutput<AuthorityLeaseId>,
            _clock: &mut ScriptedClock,
        ) -> Result<
            LiveMotionApplied<Self::Receipt, Self::Diagnostic>,
            LiveMotionPortTickError<Self::Error>,
        > {
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
                tick,
                receipt: self.receipt(applied_stop),
                diagnostic: (),
                stopped: applied_stop,
                blocked: !requested_stop && applied_stop,
                frontier_yaw_target_reached: false,
                #[cfg(feature = "operator-console")]
                typed_request_key: None,
            })
        }

        fn tick_point_goal(
            &mut self,
            _coordinator: &mut ShadowNavigationCoordinator<J>,
            tick: HostMonotonicTimestamp,
            _clock: &mut ScriptedClock,
        ) -> Result<
            LiveMotionApplied<Self::Receipt, Self::Diagnostic>,
            LiveMotionPortTickError<Self::Error>,
        > {
            if self.state.fail_next_tick.swap(false, Ordering::SeqCst) {
                return Err(LiveMotionPortTickError::Actuation(FakeError::Injected));
            }
            let blocked = self.state.force_tick_stop.load(Ordering::SeqCst);
            self.state
                .events
                .lock()
                .expect("event lock")
                .push(FakeEvent::PointGoalTick { blocked });
            Ok(LiveMotionApplied {
                tick,
                receipt: self.receipt(blocked),
                diagnostic: (),
                stopped: blocked,
                blocked,
                frontier_yaw_target_reached: false,
                #[cfg(feature = "operator-console")]
                typed_request_key: None,
            })
        }

        fn tick_frontier_yaw(
            &mut self,
            _coordinator: &mut ShadowNavigationCoordinator<J>,
            tick: HostMonotonicTimestamp,
            command: FrontierYawScanCommandV1,
            _clock: &mut ScriptedClock,
        ) -> Result<
            LiveMotionApplied<Self::Receipt, Self::Diagnostic>,
            LiveMotionPortTickError<Self::Error>,
        > {
            if self.state.fail_next_tick.swap(false, Ordering::SeqCst) {
                return Err(LiveMotionPortTickError::Actuation(FakeError::Injected));
            }
            let target_reached = self
                .state
                .force_yaw_target_reached
                .swap(false, Ordering::SeqCst);
            let blocked = target_reached || self.state.force_tick_stop.load(Ordering::SeqCst);
            self.state
                .events
                .lock()
                .expect("event lock")
                .push(FakeEvent::FrontierYawTick {
                    target_direction: command.target_direction(),
                    map_revision: command.scan().map_revision(),
                    started_at: command.started_at(),
                    valid_through_exclusive: command.valid_through_exclusive(),
                    blocked,
                    target_reached,
                });
            Ok(LiveMotionApplied {
                tick,
                receipt: self.receipt(blocked),
                diagnostic: (),
                stopped: blocked,
                blocked,
                frontier_yaw_target_reached: target_reached,
                #[cfg(feature = "operator-console")]
                typed_request_key: None,
            })
        }

        fn classify_error(source: &Self::Error) -> LiveMotionActuationFaultEvidence {
            LiveMotionActuationFaultEvidence::new(
                AgentLiveActuationFaultKind::TransportUnavailable,
                match source {
                    FakeError::Injected => AgentControllerStopKnowledge::Uncertain,
                    FakeError::InjectedWithConfirmedStop => AgentControllerStopKnowledge::Confirmed,
                },
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
            if self
                .state
                .fail_disarm_with_confirmed_stop
                .load(Ordering::SeqCst)
            {
                Err(FakeError::InjectedWithConfirmedStop)
            } else if self.state.fail_disarm.load(Ordering::SeqCst) {
                Err(FakeError::Injected)
            } else {
                Ok(self.next_sequence)
            }
        }
    }

    type Journal = NavigationIngressLog;
    type Owner = LiveMotionOwner<Journal, FakePort, ScriptedClock>;

    fn assert_take_lifecycle_zero(owner: &mut Owner, expected: LiveLifecycleZeroReason) {
        let event = owner
            .take_last_physical_state()
            .expect("latest physical-state event");
        let LivePhysicalStateEvent::LifecycleZero(applied) = event else {
            panic!("expected a lifecycle zero event");
        };
        assert_eq!(applied.reason(), expected);
        assert_eq!(
            applied.receipt().verified_host_result().output_state,
            OutputState::ZeroPwm
        );
        assert!(
            owner.take_last_physical_state().is_none(),
            "physical-state evidence transfers exactly once"
        );
    }

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

    fn anchor_and_admit_map(
        owner: &mut Owner,
        clock: &ScriptedClock,
        cells: Vec<OccupancyCell>,
        geometry: OccupancyGridGeometry,
    ) -> OccupancyGridSnapshot {
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
        let observed_at = clock.peek();
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
            owner
                .accept_visual(admission, observed_at)
                .expect("anchor admission"),
            VisualAdmissionOutcome::Reanchored(_)
        ));

        clock.advance(1);
        let snapshot =
            OccupancyGridSnapshot::from_test_cells(geometry, &cells, map.instance_id(), 1);
        owner
            .accept_global_map(clock.peek(), timestamp, &snapshot)
            .expect("exact retained map admission");
        clock.advance(1);
        snapshot
    }

    fn free_map_geometry() -> OccupancyGridGeometry {
        OccupancyGridGeometry::try_new(0.25, [-2.0, -2.0], 20, 16, 320).expect("bounded global map")
    }

    fn fixture(
        authority: AgentAuthoritySupervisor,
    ) -> (
        AgentControlRuntimeSender,
        Owner,
        ScriptedClock,
        FakePortState,
    ) {
        fixture_with_live_policy(authority, live_policy())
    }

    fn fixture_with_live_policy(
        authority: AgentAuthoritySupervisor,
        live_policy: NanoLiveModePolicy,
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
            live_policy,
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

    fn assert_accepted_for_processing(
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
                        completion: AgentControlCompletionV1::AcceptedForProcessing,
                    } if actual == command
            )
        ));
    }

    #[cfg(feature = "operator-console")]
    #[test]
    fn software_emergency_stop_latches_supervisor_and_retains_exact_zero() {
        let (_sender, mut owner, _clock, state) = fixture(ready());

        let applied = owner
            .apply_software_emergency_stop(AgentControlTypedRequestKey::for_test(1))
            .expect("emergency stop applies");

        assert_eq!(applied.fault(), FaultKind::EmergencyStop);
        assert!(applied.result().requested_timer_pwm.is_zero());
        assert!(applied.result().controller_timer_pwm.is_zero());
        assert!(applied.result().output_state.is_safe());
        assert_eq!(
            owner
                .dispatcher()
                .control_status(AgentMapStateV1::UNAVAILABLE)
                .runtime(),
            AgentRuntimeStateV1::Faulted
        );
        assert_eq!(state.events(), [FakeEvent::FreshZero]);
        assert_take_lifecycle_zero(&mut owner, LiveLifecycleZeroReason::SoftwareEmergencyStop);
    }

    #[cfg(feature = "operator-console")]
    #[test]
    fn failed_emergency_zero_never_reports_emergency_completion() {
        let (_sender, mut owner, _clock, state) = fixture(ready());
        state.fail_next_zero.store(true, Ordering::SeqCst);

        assert!(matches!(
            owner.apply_software_emergency_stop(AgentControlTypedRequestKey::for_test(1)),
            Err(LiveMotionOwnerError::Operation(
                LiveMotionOperationError::ActuationFault { .. }
            ))
        ));
        assert_eq!(
            owner
                .dispatcher()
                .control_status(AgentMapStateV1::UNAVAILABLE)
                .runtime(),
            AgentRuntimeStateV1::Faulted
        );
        assert_eq!(state.events(), [FakeEvent::FreshZero]);
        assert!(matches!(
            owner.take_last_physical_state(),
            Some(LivePhysicalStateEvent::ActuationFault { .. })
        ));
    }

    fn begin_reachable_point_goal(
        sender: &AgentControlRuntimeSender,
        owner: &mut Owner,
        clock: &ScriptedClock,
        parser: &mut AgentControlRequestParser,
        request_id: u64,
    ) {
        let geometry = free_map_geometry();
        let snapshot = anchor_and_admit_map(
            owner,
            clock,
            vec![OccupancyCell::Free; geometry.cell_count()],
            geometry,
        );
        let binding = owner
            .coordinator()
            .current_map_binding()
            .expect("retained map binding");
        let point = enqueue(
            sender,
            parser,
            request_id,
            json!({
                "kind": "select_map_point",
                "map_epoch_id": binding.map_epoch_id().as_u64(),
                "displayed_revision": snapshot.revision(),
                "x_m": 1.0,
                "y_m": 0.0
            }),
            clock.peek(),
        );
        assert!(matches!(
            owner.process_one(AgentMapStateV1::UNAVAILABLE),
            Ok(LiveMotionOwnerOutcome::AutonomousAccepted {
                mode: AgentAutonomousMode::PointGoal,
            })
        ));
        assert_accepted_for_processing(point, AgentControlCommandKindV1::SelectMapPoint);
    }

    fn active_point_goal_deadline(owner: &Owner) -> HostMonotonicTimestamp {
        match &owner.autonomous {
            LiveAutonomousLifecycle::Active(LiveAutonomousSession {
                execution:
                    LiveAutonomousExecution::PointGoal {
                        deadline_exclusive, ..
                    },
                ..
            }) => *deadline_exclusive,
            _ => panic!("active point-goal execution"),
        }
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
    fn production_motion_start_gate_rejects_unready_manual_without_retaining_authority() {
        let (sender, mut owner, clock, state) = fixture(ready());
        let mut parser = AgentControlRequestParser::new();
        let begin = enqueue(
            &sender,
            &mut parser,
            1,
            json!({"kind": "begin_manual"}),
            clock.peek(),
        );

        let outcome = owner.process_one_with_motion_start_readiness(AgentMapStateV1::UNAVAILABLE);
        assert!(
            matches!(
                outcome,
                Ok(LiveMotionOwnerOutcome::Rejected {
                    code: AgentControlRejectionCodeV1::NotReady,
                    retryable: true,
                })
            ),
            "{outcome:?}"
        );
        assert!(matches!(
            begin.join().expect("begin response"),
            Some(response)
                if matches!(
                    response.response(),
                    AgentControlResponseKindV1::Rejected {
                        code: AgentControlRejectionCodeV1::NotReady,
                        retryable: true,
                    }
                )
        ));
        assert!(owner.dispatcher().manual().active_lease().is_none());
        assert!(matches!(
            owner.coordinator().motion_mode(),
            CoordinatorMotionModeV1::MappingOnly
        ));
        assert_eq!(state.events(), [FakeEvent::FreshZero]);
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
        let begin_outcome = owner
            .process_one(AgentMapStateV1::UNAVAILABLE)
            .expect("manual begin");
        assert!(matches!(
            &begin_outcome,
            LiveMotionOwnerOutcome::Completed(
                LiveMotionCompletedSafetyAction::ManualStarted { .. }
            )
        ));
        assert!(
            begin_outcome.defers_periodic_motion_tick(),
            "the begin period must not immediately deadman its new lease"
        );
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
        let velocity_outcome = owner
            .process_one(AgentMapStateV1::UNAVAILABLE)
            .expect("manual velocity");
        assert!(matches!(
            &velocity_outcome,
            LiveMotionOwnerOutcome::Completed(
                LiveMotionCompletedSafetyAction::ManualCommandApplied
            )
        ));
        assert!(!velocity_outcome.defers_periodic_motion_tick());
        assert_completed(velocity, AgentControlCommandKindV1::ManualVelocity);
        let applied_velocity = match owner
            .take_last_physical_state()
            .expect("manual command transfers its exact applied tick")
        {
            LivePhysicalStateEvent::CoordinatorTick(applied) => applied,
            _ => panic!("manual velocity must retain its coordinator tick"),
        };
        assert!(applied_velocity.tick().as_nanos() < clock.peek().as_nanos());
        assert!(!applied_velocity.stopped());
        assert_eq!(
            applied_velocity
                .receipt()
                .verified_host_result()
                .output_state,
            OutputState::NonzeroPwm
        );
        assert!(
            owner.take_last_physical_state().is_none(),
            "applied evidence is transferred at most once"
        );

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
        let applied_stop = match owner
            .take_last_physical_state()
            .expect("manual stop transfers its final lifecycle zero")
        {
            LivePhysicalStateEvent::LifecycleZero(applied) => applied,
            _ => panic!("manual stop must end at its fresh lifecycle zero"),
        };
        assert!(applied_stop.requested_at().as_nanos() < clock.peek().as_nanos());
        assert_eq!(
            applied_stop.reason(),
            LiveLifecycleZeroReason::ManualRelease
        );
        assert_eq!(
            applied_stop.receipt().verified_host_result().output_state,
            OutputState::ZeroPwm
        );
        assert!(owner.take_last_physical_state().is_none());
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
            live_policy(),
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
        assert_take_lifecycle_zero(&mut owner, LiveLifecycleZeroReason::GlobalStopRequest);
        assert!(owner.dispatcher().manual().active_lease().is_none());
    }

    #[test]
    fn mapping_only_command_retains_its_exact_one_shot_zero_event() {
        let (sender, mut owner, clock, state) = fixture(ready());
        let mut parser = AgentControlRequestParser::new();
        let peer = enqueue(
            &sender,
            &mut parser,
            1,
            json!({"kind": "map_only"}),
            clock.peek(),
        );

        assert!(matches!(
            owner.process_one(AgentMapStateV1::UNAVAILABLE),
            Ok(LiveMotionOwnerOutcome::Completed(
                LiveMotionCompletedSafetyAction::MappingOnlyStopped
            ))
        ));
        assert_completed(peer, AgentControlCommandKindV1::MapOnly);
        assert_eq!(state.events(), [FakeEvent::FreshZero]);
        assert_take_lifecycle_zero(&mut owner, LiveLifecycleZeroReason::MappingOnlyRequest);
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
        assert_take_lifecycle_zero(&mut owner, LiveLifecycleZeroReason::ManualRelease);
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
        assert_take_lifecycle_zero(&mut owner, LiveLifecycleZeroReason::ManualRelease);
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
        assert!(matches!(
            owner.take_last_physical_state(),
            Some(LivePhysicalStateEvent::ActuationFault {
                evidence,
                ..
            }) if evidence.controller_stop() == AgentControllerStopKnowledge::Uncertain
        ));
        assert!(owner.take_last_physical_state().is_none());
    }

    #[test]
    fn physical_tick_and_direct_mode_cleanup_failures_are_both_reported() {
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

        owner.coordinator.exhaust_motion_mode_generation_for_test();
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
                LiveMotionOperationError::PrimaryAndCleanup {
                    primary,
                    cleanup,
                }
            )) if matches!(
                primary.as_ref(),
                LiveMotionOperationError::ActuationFault {
                    evidence,
                    latch: LiveMotionFaultLatch::Latched {
                        fault: FaultKind::HardwareReadinessLost,
                        ..
                    },
                    ..
                } if evidence.controller_stop() == AgentControllerStopKnowledge::Uncertain
            ) && matches!(
                cleanup.as_ref(),
                LiveMotionOperationError::CoordinatorMode(
                    CoordinatorMotionModeError::GenerationExhausted
                )
            )
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
            owner.coordinator().motion_mode(),
            CoordinatorMotionModeV1::Manual { .. }
        ));
        assert!(matches!(
            owner.take_last_physical_state(),
            Some(LivePhysicalStateEvent::ActuationFault {
                evidence,
                ..
            }) if evidence.controller_stop() == AgentControllerStopKnowledge::Uncertain
        ));
        assert!(owner.take_last_physical_state().is_none());
        assert_eq!(
            state.events(),
            [FakeEvent::FreshZero],
            "the injected physical tick fails before emitting an applied receipt"
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
        assert_take_lifecycle_zero(&mut owner, LiveLifecycleZeroReason::DisarmRequest);

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
        assert_take_lifecycle_zero(&mut owner, LiveLifecycleZeroReason::ShutdownRequest);
    }

    #[test]
    fn shutdown_wire_uncertainty_preserves_the_completed_physical_stop() {
        let (sender, mut owner, clock, state) = fixture(ready());
        let mut parser = AgentControlRequestParser::new();
        let peer = enqueue_agent_control_test_request_with_failed_wire_delivery(
            &sender,
            request(&mut parser, 1, json!({"kind": "shutdown"})),
            clock.peek(),
        );

        assert!(matches!(
            owner.process_one(AgentMapStateV1::UNAVAILABLE),
            Err(LiveMotionOwnerError::ResponseAfterSafety {
                safety: LiveMotionCompletedSafetyAction::ShutdownStopped,
                response: AgentControlDispatchResponseError::WireDeliveryUncertain,
            })
        ));
        assert_eq!(
            state.events(),
            [FakeEvent::FreshZero],
            "the exact zero is applied before response delivery is attempted"
        );
        assert_take_lifecycle_zero(&mut owner, LiveLifecycleZeroReason::ShutdownRequest);
        assert!(matches!(
            peer.join().expect("wire-failure peer"),
            Some(response)
                if matches!(
                    response.response(),
                    AgentControlResponseKindV1::Accepted {
                        command: AgentControlCommandKindV1::Shutdown,
                        completion: AgentControlCompletionV1::Completed,
                    }
                )
        ));
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
    fn save_map_is_an_outer_action_only_while_every_motion_lifecycle_is_inactive() {
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
                code: AgentControlRejectionCodeV1::MapUnavailable,
                retryable: true,
            })
        ));
        assert!(matches!(
            frontier.join().expect("frontier peer"),
            Some(response)
                if matches!(
                    response.response(),
                    AgentControlResponseKindV1::Rejected {
                        code: AgentControlRejectionCodeV1::MapUnavailable,
                        retryable: true,
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

        let begin = enqueue(
            &sender,
            &mut parser,
            3,
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

        let active_save = enqueue(
            &sender,
            &mut parser,
            4,
            json!({"kind": "save_map"}),
            clock.peek(),
        );
        assert!(matches!(
            owner.process_one(AgentMapStateV1::UNAVAILABLE),
            Ok(LiveMotionOwnerOutcome::Rejected {
                code: AgentControlRejectionCodeV1::ModeConflict,
                retryable: true,
            })
        ));
        assert!(matches!(
            active_save.join().expect("active save peer"),
            Some(response)
                if matches!(
                    response.response(),
                    AgentControlResponseKindV1::Rejected {
                        code: AgentControlRejectionCodeV1::ModeConflict,
                        retryable: true,
                    }
                )
        ));
        assert!(owner.dispatcher().manual().active_lease().is_some());
    }

    #[test]
    fn point_goal_is_accepted_only_after_its_first_receipt_and_global_stop_releases_it() {
        let (sender, mut owner, clock, state) = fixture(ready());
        let geometry = free_map_geometry();
        let snapshot = anchor_and_admit_map(
            &mut owner,
            &clock,
            vec![OccupancyCell::Free; geometry.cell_count()],
            geometry,
        );
        let binding = owner
            .coordinator()
            .current_map_binding()
            .expect("retained map binding");
        let mut parser = AgentControlRequestParser::new();
        let point = enqueue(
            &sender,
            &mut parser,
            1,
            json!({
                "kind": "select_map_point",
                "map_epoch_id": binding.map_epoch_id().as_u64(),
                "displayed_revision": snapshot.revision(),
                "x_m": 1.0,
                "y_m": 0.0
            }),
            clock.peek(),
        );

        assert!(matches!(
            owner.process_one(AgentMapStateV1::UNAVAILABLE),
            Ok(LiveMotionOwnerOutcome::AutonomousAccepted {
                mode: AgentAutonomousMode::PointGoal,
            })
        ));
        assert_accepted_for_processing(point, AgentControlCommandKindV1::SelectMapPoint);
        assert_eq!(
            state.events(),
            [
                FakeEvent::FreshZero,
                FakeEvent::PointGoalTick { blocked: false },
            ]
        );
        assert!(matches!(
            owner.tick_motion(),
            Ok(LiveMotionOwnerOutcome::PeriodicAutonomousApplied {
                mode: AgentAutonomousMode::PointGoal,
            })
        ));

        let stop = enqueue(
            &sender,
            &mut parser,
            2,
            json!({"kind": "stop"}),
            clock.peek(),
        );
        assert!(matches!(
            owner.process_one(AgentMapStateV1::UNAVAILABLE),
            Ok(LiveMotionOwnerOutcome::Completed(
                LiveMotionCompletedSafetyAction::GlobalStopped
            ))
        ));
        assert_completed(stop, AgentControlCommandKindV1::Stop);
        assert!(state.events().ends_with(&[
            FakeEvent::PointGoalTick { blocked: false },
            FakeEvent::FreshZero,
        ]));
        assert_take_lifecycle_zero(&mut owner, LiveLifecycleZeroReason::GlobalStopRequest);
        assert!(
            owner
                .dispatcher()
                .manual()
                .authority()
                .active_lease()
                .is_none()
        );
        assert_eq!(
            owner.coordinator().motion_mode(),
            CoordinatorMotionModeV1::MappingOnly
        );
    }

    #[test]
    fn point_goal_deadline_is_exclusive_and_stops_at_or_after_the_boundary() {
        let point_goal_runtime = Duration::from_nanos(100);
        let (sender, mut owner, clock, state) = fixture_with_live_policy(
            ready(),
            live_policy_with_point_goal_runtime(point_goal_runtime),
        );
        let mut parser = AgentControlRequestParser::new();
        begin_reachable_point_goal(&sender, &mut owner, &clock, &mut parser, 1);
        let deadline_exclusive = active_point_goal_deadline(&owner);

        clock.set(
            deadline_exclusive
                .as_nanos()
                .checked_sub(1)
                .expect("nonzero deadline"),
        );
        assert!(matches!(
            owner.tick_motion(),
            Ok(LiveMotionOwnerOutcome::PeriodicAutonomousApplied {
                mode: AgentAutonomousMode::PointGoal,
            })
        ));
        let active_status = owner
            .dispatcher()
            .control_status(AgentMapStateV1::UNAVAILABLE);
        assert_eq!(
            active_status.runtime(),
            super::super::AgentRuntimeStateV1::Active {
                mode: super::super::AgentOperatingModeV1::PointGoal,
            }
        );
        assert_eq!(
            active_status.base_command(),
            super::super::AgentBaseCommandStateV1::Unknown
        );

        clock.set(deadline_exclusive.as_nanos());
        assert!(matches!(
            owner.tick_motion(),
            Ok(LiveMotionOwnerOutcome::AutonomousCompleted {
                mode: AgentAutonomousMode::PointGoal,
            })
        ));
        assert!(state.events().ends_with(&[
            FakeEvent::PointGoalTick { blocked: false },
            FakeEvent::FreshZero,
        ]));
        assert!(
            owner
                .dispatcher()
                .manual()
                .authority()
                .active_lease()
                .is_none()
        );
        assert_eq!(
            owner.coordinator().motion_mode(),
            CoordinatorMotionModeV1::MappingOnly
        );
        assert!(owner.coordinator().current_goal().is_none());

        let status_request = enqueue(
            &sender,
            &mut parser,
            2,
            json!({"kind": "query_status"}),
            clock.peek(),
        );
        let status = match owner
            .process_one(AgentMapStateV1::UNAVAILABLE)
            .expect("post-deadline status response")
        {
            LiveMotionOwnerOutcome::StatusReplied(status) => status,
            outcome => panic!("unexpected status outcome: {outcome:?}"),
        };
        assert_eq!(
            status.runtime(),
            super::super::AgentRuntimeStateV1::ReadyStopped
        );
        assert_eq!(
            status.base_command(),
            super::super::AgentBaseCommandStateV1::ConfirmedStopped
        );
        assert!(matches!(
            status_request.join().expect("status response peer"),
            Some(response)
                if matches!(
                    response.response(),
                    AgentControlResponseKindV1::Status {
                        status: response_status,
                    } if response_status == status
                )
        ));

        let events_after_completion = state.events();
        clock.set(
            deadline_exclusive
                .as_nanos()
                .checked_add(1)
                .expect("test deadline headroom"),
        );
        assert!(matches!(
            owner.tick_motion(),
            Ok(LiveMotionOwnerOutcome::Idle)
        ));
        assert_eq!(state.events(), events_after_completion);

        let (sender, mut owner, clock, state) = fixture_with_live_policy(
            ready(),
            live_policy_with_point_goal_runtime(point_goal_runtime),
        );
        let mut parser = AgentControlRequestParser::new();
        begin_reachable_point_goal(&sender, &mut owner, &clock, &mut parser, 1);
        let deadline_exclusive = active_point_goal_deadline(&owner);
        clock.set(
            deadline_exclusive
                .as_nanos()
                .checked_add(1)
                .expect("test deadline headroom"),
        );
        assert!(matches!(
            owner.tick_motion(),
            Ok(LiveMotionOwnerOutcome::AutonomousCompleted {
                mode: AgentAutonomousMode::PointGoal,
            })
        ));
        assert!(state.events().ends_with(&[
            FakeEvent::PointGoalTick { blocked: false },
            FakeEvent::FreshZero,
        ]));
    }

    #[test]
    fn point_goal_deadline_overflow_is_typed_and_never_acquires_authority() {
        let started_at = at(u64::MAX - 5);
        assert!(matches!(
            autonomous_deadline::<FakeError, FakeError>(started_at, Duration::from_nanos(10)),
            Err(LiveMotionOperationError::AutonomousDeadlineOverflow {
                started_at: actual,
                maximum_runtime_ns: 10,
            }) if actual == started_at
        ));

        let (sender, mut owner, clock, state) = fixture_with_live_policy(
            ready(),
            live_policy_with_point_goal_runtime(Duration::from_nanos(10)),
        );
        clock.set(u64::MAX - 5);
        let mut parser = AgentControlRequestParser::new();
        let point = enqueue(
            &sender,
            &mut parser,
            1,
            json!({
                "kind": "select_map_point",
                "map_epoch_id": 1,
                "displayed_revision": 1,
                "x_m": 1.0,
                "y_m": 0.0
            }),
            clock.peek(),
        );
        assert!(matches!(
            owner.process_one(AgentMapStateV1::UNAVAILABLE),
            Err(LiveMotionOwnerError::Operation(
                LiveMotionOperationError::AutonomousDeadlineOverflow {
                    started_at,
                    maximum_runtime_ns: 10,
                }
            )) if started_at.as_nanos() == u64::MAX - 4
        ));
        assert!(matches!(
            point.join().expect("overflow response peer"),
            Some(response)
                if matches!(
                    response.response(),
                    AgentControlResponseKindV1::Rejected {
                        code: AgentControlRejectionCodeV1::InternalFault,
                        retryable: false,
                    }
                )
        ));
        assert!(state.events().is_empty());
        assert!(
            owner
                .dispatcher()
                .manual()
                .authority()
                .active_lease()
                .is_none()
        );
        let status = owner
            .dispatcher()
            .control_status(AgentMapStateV1::UNAVAILABLE);
        assert_eq!(
            status.runtime(),
            super::super::AgentRuntimeStateV1::ReadyStopped
        );
        assert_eq!(
            status.base_command(),
            super::super::AgentBaseCommandStateV1::ConfirmedStopped
        );
    }

    #[test]
    fn clock_failures_preserve_the_active_autonomous_token_for_retry_and_stop() {
        let (sender, mut owner, clock, state) = fixture(ready());
        let geometry = free_map_geometry();
        let snapshot = anchor_and_admit_map(
            &mut owner,
            &clock,
            vec![OccupancyCell::Free; geometry.cell_count()],
            geometry,
        );
        let binding = owner
            .coordinator()
            .current_map_binding()
            .expect("retained map binding");
        let mut parser = AgentControlRequestParser::new();
        let point = enqueue(
            &sender,
            &mut parser,
            1,
            json!({
                "kind": "select_map_point",
                "map_epoch_id": binding.map_epoch_id().as_u64(),
                "displayed_revision": snapshot.revision(),
                "x_m": 1.0,
                "y_m": 0.0
            }),
            clock.peek(),
        );
        assert!(matches!(
            owner.process_one(AgentMapStateV1::UNAVAILABLE),
            Ok(LiveMotionOwnerOutcome::AutonomousAccepted {
                mode: AgentAutonomousMode::PointGoal,
            })
        ));
        assert_accepted_for_processing(point, AgentControlCommandKindV1::SelectMapPoint);

        let active_lease = owner
            .dispatcher()
            .manual()
            .authority()
            .active_lease()
            .expect("active autonomous lease");
        clock.fail_next();
        assert!(matches!(
            owner.tick_motion(),
            Err(LiveMotionOwnerError::Operation(
                LiveMotionOperationError::Clock(_)
            ))
        ));
        assert_eq!(
            owner.dispatcher().manual().authority().active_lease(),
            Some(active_lease)
        );
        assert!(matches!(
            owner.tick_motion(),
            Ok(LiveMotionOwnerOutcome::PeriodicAutonomousApplied {
                mode: AgentAutonomousMode::PointGoal,
            })
        ));

        clock.fail_next();
        assert!(matches!(
            owner.stop_autonomous(LiveLifecycleZeroReason::AutonomousRelease),
            Err(LiveMotionOperationError::Clock(_))
        ));
        assert_eq!(
            owner
                .dispatcher()
                .manual()
                .authority()
                .active_lease()
                .expect("active token remains installed")
                .id(),
            active_lease.id()
        );
        owner
            .stop_autonomous(LiveLifecycleZeroReason::AutonomousRelease)
            .expect("retryable ordered stop");
        assert!(
            owner
                .dispatcher()
                .manual()
                .authority()
                .active_lease()
                .is_none()
        );
        assert!(state.events().ends_with(&[FakeEvent::FreshZero]));
    }

    #[test]
    fn frontier_selection_uses_the_same_point_goal_receipt_path_and_is_bounded_by_stop() {
        let (sender, mut owner, clock, state) = fixture(ready());
        let geometry = free_map_geometry();
        let mut cells = vec![OccupancyCell::Free; geometry.cell_count()];
        cells[8 * 20 + 12] = OccupancyCell::Unknown;
        anchor_and_admit_map(&mut owner, &clock, cells, geometry);
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
            Ok(LiveMotionOwnerOutcome::AutonomousAccepted {
                mode: AgentAutonomousMode::Explore,
            })
        ));
        assert_accepted_for_processing(frontier, AgentControlCommandKindV1::FrontierExplore);
        assert_eq!(
            state.events(),
            [
                FakeEvent::FreshZero,
                FakeEvent::PointGoalTick { blocked: false },
            ]
        );

        let stop = enqueue(
            &sender,
            &mut parser,
            2,
            json!({"kind": "stop"}),
            clock.peek(),
        );
        assert!(matches!(
            owner.process_one(AgentMapStateV1::UNAVAILABLE),
            Ok(LiveMotionOwnerOutcome::Completed(
                LiveMotionCompletedSafetyAction::GlobalStopped
            ))
        ));
        assert_completed(stop, AgentControlCommandKindV1::Stop);
        assert!(state.events().ends_with(&[FakeEvent::FreshZero]));
        assert!(
            owner
                .dispatcher()
                .manual()
                .authority()
                .active_lease()
                .is_none()
        );
    }

    #[test]
    fn in_place_frontier_scan_attempts_each_map_direction_once_then_completes() {
        let (sender, mut owner, clock, state) = fixture(ready());
        let geometry = free_map_geometry();
        let mut cells = vec![OccupancyCell::Unknown; geometry.cell_count()];
        cells[8 * 20 + 8] = OccupancyCell::Free;
        anchor_and_admit_map(&mut owner, &clock, cells, geometry);
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
            Ok(LiveMotionOwnerOutcome::AutonomousAccepted {
                mode: AgentAutonomousMode::Explore,
            })
        ));
        assert_accepted_for_processing(frontier, AgentControlCommandKindV1::FrontierExplore);

        for remaining_after_tick in (0..4).rev() {
            state.force_yaw_target_reached.store(true, Ordering::SeqCst);
            let outcome = owner.tick_motion().expect("bounded yaw scan tick");
            if remaining_after_tick == 0 {
                assert!(matches!(
                    outcome,
                    LiveMotionOwnerOutcome::AutonomousCompleted {
                        mode: AgentAutonomousMode::Explore,
                    }
                ));
            } else {
                assert!(matches!(
                    outcome,
                    LiveMotionOwnerOutcome::PeriodicAutonomousApplied {
                        mode: AgentAutonomousMode::Explore,
                    }
                ));
            }
        }

        let directions: Vec<_> = state
            .events()
            .into_iter()
            .filter_map(|event| match event {
                FakeEvent::FrontierYawTick {
                    target_direction, ..
                } => Some(target_direction),
                _ => None,
            })
            .collect();
        assert_eq!(
            directions,
            [
                FrontierUnknownDirection::NegativeMapY,
                FrontierUnknownDirection::NegativeMapY,
                FrontierUnknownDirection::NegativeMapX,
                FrontierUnknownDirection::PositiveMapX,
                FrontierUnknownDirection::PositiveMapY,
            ]
        );
        assert!(
            owner
                .dispatcher()
                .manual()
                .authority()
                .active_lease()
                .is_none()
        );
        assert_eq!(
            owner.coordinator().motion_mode(),
            CoordinatorMotionModeV1::MappingOnly
        );
    }

    #[test]
    fn map_revision_rebinds_the_same_yaw_episode_without_resetting_its_start() {
        let (sender, mut owner, clock, state) = fixture(ready());
        let geometry = free_map_geometry();
        let mut cells = vec![OccupancyCell::Unknown; geometry.cell_count()];
        cells[8 * 20 + 8] = OccupancyCell::Free;
        let first = anchor_and_admit_map(&mut owner, &clock, cells.clone(), geometry);
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
            Ok(LiveMotionOwnerOutcome::AutonomousAccepted {
                mode: AgentAutonomousMode::Explore,
            })
        ));
        assert_accepted_for_processing(frontier, AgentControlCommandKindV1::FrontierExplore);
        let first_command = state
            .events()
            .into_iter()
            .find_map(|event| match event {
                FakeEvent::FrontierYawTick {
                    target_direction,
                    map_revision,
                    started_at,
                    valid_through_exclusive,
                    ..
                } => Some((
                    target_direction,
                    map_revision,
                    started_at,
                    valid_through_exclusive,
                )),
                _ => None,
            })
            .expect("first yaw command");

        let second = OccupancyGridSnapshot::from_test_cells(
            geometry,
            &cells,
            first.map_instance_id().expect("map-bound snapshot"),
            2,
        );
        owner
            .accept_global_map(clock.peek(), Timestamp::from_nanos(101), &second)
            .expect("newer exact map");
        clock.advance(1);
        assert!(matches!(
            owner.tick_motion(),
            Ok(LiveMotionOwnerOutcome::PeriodicAutonomousApplied {
                mode: AgentAutonomousMode::Explore,
            })
        ));
        let commands: Vec<_> = state
            .events()
            .into_iter()
            .filter_map(|event| match event {
                FakeEvent::FrontierYawTick {
                    target_direction,
                    map_revision,
                    started_at,
                    valid_through_exclusive,
                    ..
                } => Some((
                    target_direction,
                    map_revision,
                    started_at,
                    valid_through_exclusive,
                )),
                _ => None,
            })
            .collect();
        assert_eq!(commands.len(), 2);
        assert_eq!(commands[0], first_command);
        assert_eq!(commands[1].0, first_command.0);
        assert_eq!(commands[1].1, 2);
        assert_eq!(commands[1].2, first_command.2);
        assert!(commands[1].3 > first_command.3);

        let stop = enqueue(
            &sender,
            &mut parser,
            2,
            json!({"kind": "stop"}),
            clock.peek(),
        );
        assert!(matches!(
            owner.process_one(AgentMapStateV1::UNAVAILABLE),
            Ok(LiveMotionOwnerOutcome::Completed(
                LiveMotionCompletedSafetyAction::GlobalStopped
            ))
        ));
        assert_completed(stop, AgentControlCommandKindV1::Stop);
    }

    #[test]
    fn blocked_first_autonomous_tick_rejects_after_release_and_fresh_zero() {
        let (sender, mut owner, clock, state) = fixture(ready());
        let geometry = free_map_geometry();
        let snapshot = anchor_and_admit_map(
            &mut owner,
            &clock,
            vec![OccupancyCell::Free; geometry.cell_count()],
            geometry,
        );
        let binding = owner
            .coordinator()
            .current_map_binding()
            .expect("retained map binding");
        state.force_tick_stop.store(true, Ordering::SeqCst);
        let mut parser = AgentControlRequestParser::new();
        let point = enqueue(
            &sender,
            &mut parser,
            1,
            json!({
                "kind": "select_map_point",
                "map_epoch_id": binding.map_epoch_id().as_u64(),
                "displayed_revision": snapshot.revision(),
                "x_m": 1.0,
                "y_m": 0.0
            }),
            clock.peek(),
        );

        assert!(matches!(
            owner.process_one(AgentMapStateV1::UNAVAILABLE),
            Ok(LiveMotionOwnerOutcome::Rejected {
                code: AgentControlRejectionCodeV1::NotReady,
                retryable: true,
            })
        ));
        assert!(matches!(
            point.join().expect("point response"),
            Some(response)
                if matches!(
                    response.response(),
                    AgentControlResponseKindV1::Rejected {
                        code: AgentControlRejectionCodeV1::NotReady,
                        retryable: true,
                    }
                )
        ));
        assert_eq!(
            state.events(),
            [
                FakeEvent::FreshZero,
                FakeEvent::PointGoalTick { blocked: true },
                FakeEvent::FreshZero,
            ]
        );
        assert!(
            owner
                .dispatcher()
                .manual()
                .authority()
                .active_lease()
                .is_none()
        );
        assert_eq!(
            owner.coordinator().motion_mode(),
            CoordinatorMotionModeV1::MappingOnly
        );
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
        let (coordinator, lifecycle, stop, last_physical_state) = report.into_parts();
        assert!(lifecycle.is_none());
        assert!(matches!(stop, LiveMotionTerminalStop::Confirmed(_)));
        assert!(matches!(
            last_physical_state,
            Some(LivePhysicalStateEvent::LifecycleZero(applied))
                if applied.reason() == LiveLifecycleZeroReason::TerminalShutdown
        ));
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

    #[test]
    fn terminal_shutdown_preserves_failed_disarm_and_independent_confirmed_stop() {
        let (_sender, owner, _clock, state) = fixture(disarmed());
        state
            .fail_disarm_with_confirmed_stop
            .store(true, Ordering::SeqCst);

        let report = owner.shutdown();

        assert!(report.lifecycle_cleanup().is_none());
        assert!(report.controller_stop().is_confirmed());
        assert!(matches!(
            report.controller_stop(),
            LiveMotionTerminalStop::DisarmFailedStopConfirmed(FakeError::InjectedWithConfirmedStop)
        ));
        assert_eq!(state.events(), [FakeEvent::FreshZero, FakeEvent::Disarm]);
    }
}
