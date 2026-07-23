//! Supervisor-owned manual authority and command ingress for the live agent.
//!
//! This module deliberately owns no socket, coordinator, or actuator handle.
//! It is the typed state seam used by the live owner which owns those objects:
//! socket requests enter here only after claim, accepted targets leave only for
//! `ShadowNavigationCoordinator::tick_manual`, and every zero obligation must
//! be completed with exact applied evidence before another mode can begin.

use std::fmt;

use kiko_supervisor_core::{
    AuthorityDuration, AuthorityLease, AuthorityLeaseId, AuthorityMode, FaultKind, StopReason,
    SupervisorAction, SupervisorState, SupervisorStateKind,
};
use robot_command_client::{EvidenceError, FailureCause, LatchedStopKnowledge};
use robot_protocol::v2::HostCommandResult;

use super::actuation::{LiveActuationError, LocalRejectionStop, PhysicalDecisionError};
use super::mpc::BoundedId;
use super::{
    AgentAuthorityError, AgentAuthoritySupervisor, AgentControlStatusV1, AgentManualStopV1,
    AgentManualVelocityV1, AgentMapStateV1, ManualDriveConfigV1, ManualDriveCore,
    ManualDriveOutput, ManualDriveStopCause, PlantBoundNanoManualControlApiConfig,
};
use crate::HostMonotonicTimestamp;

/// Plant-bound policy stripped to the values used on every hot-path command.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AgentManualRuntimePolicy {
    authority_lease: AuthorityDuration,
    drive: ManualDriveConfigV1,
    plant_model_id: BoundedId,
    plant_model_version: std::num::NonZeroU32,
}

impl From<PlantBoundNanoManualControlApiConfig> for AgentManualRuntimePolicy {
    fn from(value: PlantBoundNanoManualControlApiConfig) -> Self {
        Self {
            authority_lease: value.authority_lease(),
            drive: value.drive(),
            plant_model_id: value.plant_model_id(),
            plant_model_version: value.plant_model_version(),
        }
    }
}

impl AgentManualRuntimePolicy {
    pub const fn authority_lease(self) -> AuthorityDuration {
        self.authority_lease
    }

    pub const fn drive(self) -> ManualDriveConfigV1 {
        self.drive
    }

    pub const fn plant_model_id(self) -> BoundedId {
        self.plant_model_id
    }

    pub const fn plant_model_version(self) -> std::num::NonZeroU32 {
        self.plant_model_version
    }

    #[cfg(test)]
    pub(crate) fn for_test(authority_lease: AuthorityDuration, drive: ManualDriveConfigV1) -> Self {
        Self {
            authority_lease,
            drive,
            plant_model_id: BoundedId::parse("model_id", "manual-test-plant".to_owned())
                .expect("test plant identity"),
            plant_model_version: std::num::NonZeroU32::new(1).expect("nonzero test version"),
        }
    }
}

struct ActiveManualSession {
    lease: AuthorityLease,
    core: ManualDriveCore<AuthorityLeaseId>,
}

enum ManualLifecycle {
    Inactive,
    AwaitingBeginZero,
    AwaitingCancelledBeginZero,
    Active(ActiveManualSession),
    AwaitingReleaseZero { lease_id: AuthorityLeaseId },
}

/// Autonomous modes which may share the sole process-wide authority owner.
///
/// Manual and commissioning authority are deliberately unrepresentable here:
/// manual has its own command/deadman lifecycle below, while commissioning is
/// outside the live navigation runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AgentAutonomousMode {
    PointGoal,
    Explore,
}

impl AgentAutonomousMode {
    const fn authority_mode(self) -> AuthorityMode {
        match self {
            Self::PointGoal => AuthorityMode::PointGoal,
            Self::Explore => AuthorityMode::Explore,
        }
    }
}

/// Active autonomous lease token retained by the sole live motion owner.
///
/// This token is intentionally not `Clone` or `Copy`. Renewal and release
/// consume the token so two independent continuations cannot claim the same
/// supervisor lease.
#[derive(Debug)]
pub struct AgentAutonomousAuthority {
    mode: AgentAutonomousMode,
    lease: AuthorityLease,
}

impl AgentAutonomousAuthority {
    pub const fn mode(&self) -> AgentAutonomousMode {
        self.mode
    }

    pub const fn lease(&self) -> AuthorityLease {
        self.lease
    }
}

/// Exact supervisor continuation waiting for a newly applied grant zero.
#[derive(Debug)]
pub struct PendingAgentAutonomousGrant {
    mode: AgentAutonomousMode,
    reason: StopReason,
}

impl PendingAgentAutonomousGrant {
    pub const fn mode(&self) -> AgentAutonomousMode {
        self.mode
    }

    pub const fn reason(&self) -> StopReason {
        self.reason
    }
}

/// Exact supervisor continuation waiting for a newly applied stop zero.
#[derive(Debug)]
pub struct PendingAgentAutonomousStop {
    mode: AgentAutonomousMode,
    lease_id: Option<AuthorityLeaseId>,
    reason: StopReason,
}

impl PendingAgentAutonomousStop {
    pub const fn mode(&self) -> AgentAutonomousMode {
        self.mode
    }

    pub const fn lease_id(&self) -> Option<AuthorityLeaseId> {
        self.lease_id
    }

    pub const fn reason(&self) -> StopReason {
        self.reason
    }
}

#[derive(Debug)]
pub(crate) enum AgentAutonomousRequest {
    Granted(AgentAutonomousAuthority),
    FreshAppliedZeroRequired(PendingAgentAutonomousGrant),
}

#[derive(Debug)]
pub(crate) enum AgentAutonomousTick {
    Active,
    FreshAppliedStopRequired(PendingAgentAutonomousStop),
}

/// Sole manual state owner around the process-wide authority supervisor.
pub struct AgentManualControlCore {
    authority: AgentAuthoritySupervisor,
    policy: Option<AgentManualRuntimePolicy>,
    lifecycle: ManualLifecycle,
}

impl AgentManualControlCore {
    pub fn new(
        authority: AgentAuthoritySupervisor,
        policy: Option<AgentManualRuntimePolicy>,
    ) -> Self {
        Self {
            authority,
            policy,
            lifecycle: ManualLifecycle::Inactive,
        }
    }

    pub const fn authority(&self) -> &AgentAuthoritySupervisor {
        &self.authority
    }

    /// Request a fresh autonomous lease from the stopped lifecycle state.
    ///
    /// Same-mode renewal is deliberately a separate operation requiring the
    /// exact retained active token. Cross-mode transitions must first release
    /// the old token and satisfy its stop zero, so this entry point cannot
    /// silently hand over authority without the owner accounting for cleanup.
    pub(crate) fn request_autonomous(
        &mut self,
        mode: AgentAutonomousMode,
        duration: AuthorityDuration,
        now: HostMonotonicTimestamp,
    ) -> Result<AgentAutonomousRequest, AgentAutonomousControlError> {
        self.require_manual_inactive_for_autonomy()?;
        if !matches!(self.authority.state(), SupervisorState::ReadyStopped { .. }) {
            return Err(AgentAutonomousControlError::AuthorityNotReady {
                actual: self.authority.state().kind(),
                obligation: self.authority.pending_obligation(),
            });
        }
        let action = self
            .authority
            .request_mode(mode.authority_mode(), duration, now)
            .map_err(AgentAutonomousControlError::Authority)?;
        match action {
            SupervisorAction::AuthorityGranted { lease }
                if lease.mode() == mode.authority_mode() =>
            {
                Ok(AgentAutonomousRequest::Granted(AgentAutonomousAuthority {
                    mode,
                    lease,
                }))
            }
            SupervisorAction::BaseZeroRequired {
                reason: StopReason::ZeroEvidenceStale,
            } => Ok(AgentAutonomousRequest::FreshAppliedZeroRequired(
                PendingAgentAutonomousGrant {
                    mode,
                    reason: StopReason::ZeroEvidenceStale,
                },
            )),
            action => Err(self.autonomous_invariant("request autonomous authority", action, now)),
        }
    }

    /// Complete the exact pending autonomous grant with newly applied zero
    /// evidence. A rejected receipt leaves the borrowed token available to the
    /// owner for cancellation or retry.
    pub(crate) fn complete_autonomous_grant_with_applied_zero(
        &mut self,
        pending: &PendingAgentAutonomousGrant,
        result: HostCommandResult,
        observed_at: HostMonotonicTimestamp,
        now: HostMonotonicTimestamp,
    ) -> Result<AgentAutonomousAuthority, AgentAutonomousControlError> {
        self.require_manual_inactive_for_autonomy()?;
        self.require_pending_reason("complete autonomous grant zero", pending.reason)?;
        let action = self
            .authority
            .admit_applied_zero(result, observed_at, now)
            .map_err(AgentAutonomousControlError::Authority)?;
        match action {
            SupervisorAction::AuthorityGranted { lease }
                if lease.mode() == pending.mode.authority_mode() =>
            {
                Ok(AgentAutonomousAuthority {
                    mode: pending.mode,
                    lease,
                })
            }
            SupervisorAction::PendingAuthorityExpired => {
                Err(AgentAutonomousControlError::PendingAuthorityExpired)
            }
            action => Err(self.autonomous_invariant("complete autonomous grant zero", action, now)),
        }
    }

    /// Renew only the exact active autonomous lease retained by the owner.
    ///
    /// The token is updated only after a same-ID, same-mode renewal. Any
    /// failure leaves the prior token intact as diagnostic/cleanup evidence.
    pub(crate) fn renew_autonomous(
        &mut self,
        active: &mut AgentAutonomousAuthority,
        duration: AuthorityDuration,
        now: HostMonotonicTimestamp,
    ) -> Result<(), AgentAutonomousControlError> {
        self.require_manual_inactive_for_autonomy()?;
        self.require_exact_autonomous(active)?;
        let action = self
            .authority
            .request_mode(active.mode.authority_mode(), duration, now)
            .map_err(AgentAutonomousControlError::Authority)?;
        match action {
            SupervisorAction::AuthorityRenewed { lease }
                if lease.id() == active.lease.id()
                    && lease.mode() == active.mode.authority_mode() =>
            {
                active.lease = lease;
                Ok(())
            }
            SupervisorAction::BaseZeroRequired {
                reason: StopReason::AuthorityLeaseExpired,
            } => Err(AgentAutonomousControlError::LeaseExpired {
                pending: PendingAgentAutonomousStop {
                    mode: active.mode,
                    lease_id: Some(active.lease.id()),
                    reason: StopReason::AuthorityLeaseExpired,
                },
            }),
            action => Err(self.autonomous_invariant("renew autonomous authority", action, now)),
        }
    }

    /// Advance the exact autonomous lease expiry without applying motion.
    pub(crate) fn tick_autonomous(
        &mut self,
        active: &AgentAutonomousAuthority,
        now: HostMonotonicTimestamp,
    ) -> Result<AgentAutonomousTick, AgentAutonomousControlError> {
        self.require_manual_inactive_for_autonomy()?;
        self.require_exact_autonomous(active)?;
        let action = self
            .authority
            .tick(now)
            .map_err(AgentAutonomousControlError::Authority)?;
        match action {
            SupervisorAction::None => {
                self.require_exact_autonomous(active)?;
                Ok(AgentAutonomousTick::Active)
            }
            SupervisorAction::BaseZeroRequired {
                reason: StopReason::AuthorityLeaseExpired,
            } => Ok(AgentAutonomousTick::FreshAppliedStopRequired(
                PendingAgentAutonomousStop {
                    mode: active.mode,
                    lease_id: Some(active.lease.id()),
                    reason: StopReason::AuthorityLeaseExpired,
                },
            )),
            action => Err(self.autonomous_invariant("tick autonomous authority", action, now)),
        }
    }

    /// Release the exact active autonomous lease. The returned token is only
    /// an obligation; a newly applied zero must still complete it.
    pub(crate) fn begin_autonomous_release(
        &mut self,
        active: &AgentAutonomousAuthority,
        now: HostMonotonicTimestamp,
    ) -> Result<PendingAgentAutonomousStop, AgentAutonomousControlError> {
        self.require_manual_inactive_for_autonomy()?;
        self.require_exact_autonomous(active)?;
        let action = self
            .authority
            .release_authority(active.lease.id(), now)
            .map_err(AgentAutonomousControlError::Authority)?;
        match action {
            SupervisorAction::BaseZeroRequired {
                reason: StopReason::AuthorityReleased,
            } => Ok(PendingAgentAutonomousStop {
                mode: active.mode,
                lease_id: Some(active.lease.id()),
                reason: StopReason::AuthorityReleased,
            }),
            action => Err(self.autonomous_invariant("release autonomous authority", action, now)),
        }
    }

    /// Cancel an autonomous grant which is still waiting behind its grant
    /// zero. The cancellation moves the zero barrier and returns a distinct
    /// stop continuation.
    pub(crate) fn cancel_pending_autonomous_grant(
        &mut self,
        pending: &PendingAgentAutonomousGrant,
        now: HostMonotonicTimestamp,
    ) -> Result<PendingAgentAutonomousStop, AgentAutonomousControlError> {
        self.require_manual_inactive_for_autonomy()?;
        self.require_pending_reason("cancel pending autonomous grant", pending.reason)?;
        let action = self
            .authority
            .cancel_pending_authority(now)
            .map_err(AgentAutonomousControlError::Authority)?;
        match action {
            SupervisorAction::BaseZeroRequired {
                reason: StopReason::PendingAuthorityCancelled,
            } => Ok(PendingAgentAutonomousStop {
                mode: pending.mode,
                lease_id: None,
                reason: StopReason::PendingAuthorityCancelled,
            }),
            action => {
                Err(self.autonomous_invariant("cancel pending autonomous grant", action, now))
            }
        }
    }

    /// Complete a release, expiry, or cancelled-grant stop continuation with
    /// exact new zero evidence. Rejected evidence leaves the token available.
    pub(crate) fn complete_autonomous_stop_with_applied_zero(
        &mut self,
        pending: &PendingAgentAutonomousStop,
        result: HostCommandResult,
        observed_at: HostMonotonicTimestamp,
        now: HostMonotonicTimestamp,
    ) -> Result<(), AgentAutonomousControlError> {
        self.require_manual_inactive_for_autonomy()?;
        self.require_pending_reason("complete autonomous stop zero", pending.reason)?;
        let action = self
            .authority
            .admit_applied_zero(result, observed_at, now)
            .map_err(AgentAutonomousControlError::Authority)?;
        match action {
            SupervisorAction::ReadyStopped => Ok(()),
            action => Err(self.autonomous_invariant("complete autonomous stop zero", action, now)),
        }
    }

    /// Conservatively project the process-wide lifecycle and this owner's
    /// retained manual state into the public status schema.
    pub fn control_status(&self, map: AgentMapStateV1) -> AgentControlStatusV1 {
        self.authority.control_status(map)
    }

    /// Begin explicit lifecycle arming without exposing mutable supervisor
    /// access. Success is only the mandatory fresh-zero obligation.
    pub fn begin_arm(
        &mut self,
        now: HostMonotonicTimestamp,
    ) -> Result<SupervisorAction, AgentManualControlError> {
        if !matches!(self.lifecycle, ManualLifecycle::Inactive) {
            return Err(AgentManualControlError::ModeConflict);
        }
        let action = self
            .authority
            .arm(now)
            .map_err(AgentManualControlError::Authority)?;
        match action {
            SupervisorAction::BaseZeroRequired {
                reason: StopReason::Arming,
            } => Ok(action),
            action => Err(AgentManualControlError::UnexpectedSupervisorAction {
                operation: "begin explicit arm",
                action,
            }),
        }
    }

    /// Complete explicit arming only with a newly applied exact zero receipt.
    pub fn complete_arm_with_applied_zero(
        &mut self,
        result: HostCommandResult,
        observed_at: HostMonotonicTimestamp,
        now: HostMonotonicTimestamp,
    ) -> Result<SupervisorAction, AgentManualControlError> {
        self.require_lifecycle_zero("complete explicit arm", StopReason::Arming)?;
        let action = match self.authority.admit_applied_zero(result, observed_at, now) {
            Ok(action) => action,
            Err(source) => return Err(self.pending_authority_error(source)),
        };
        match action {
            SupervisorAction::ReadyStopped => Ok(action),
            action => Err(AgentManualControlError::UnexpectedSupervisorAction {
                operation: "complete explicit arm",
                action,
            }),
        }
    }

    /// Begin explicit lifecycle disarming without exposing mutable supervisor
    /// access. An active manual core is destroyed before the supervisor enters
    /// its fresh-zero gate, preventing a cached command from being resumed.
    ///
    /// A manual transition already waiting for a different zero must be
    /// completed or faulted first; replacing that continuation would make its
    /// exact stop obligation ambiguous.
    pub fn begin_disarm(
        &mut self,
        now: HostMonotonicTimestamp,
    ) -> Result<SupervisorAction, AgentManualControlError> {
        if matches!(
            self.lifecycle,
            ManualLifecycle::AwaitingBeginZero
                | ManualLifecycle::AwaitingCancelledBeginZero
                | ManualLifecycle::AwaitingReleaseZero { .. }
        ) {
            return Err(AgentManualControlError::ModeConflict);
        }
        self.lifecycle = ManualLifecycle::Inactive;
        let action = self
            .authority
            .disarm(now)
            .map_err(AgentManualControlError::Authority)?;
        match action {
            SupervisorAction::Disarmed
            | SupervisorAction::BaseZeroRequired {
                reason: StopReason::ExplicitDisarm,
            } => Ok(action),
            action => Err(AgentManualControlError::UnexpectedSupervisorAction {
                operation: "begin explicit disarm",
                action,
            }),
        }
    }

    /// Complete explicit disarming only with a newly applied exact zero
    /// receipt produced after the disarm request.
    pub fn complete_disarm_with_applied_zero(
        &mut self,
        result: HostCommandResult,
        observed_at: HostMonotonicTimestamp,
        now: HostMonotonicTimestamp,
    ) -> Result<SupervisorAction, AgentManualControlError> {
        self.require_lifecycle_zero("complete explicit disarm", StopReason::ExplicitDisarm)?;
        let action = match self.authority.admit_applied_zero(result, observed_at, now) {
            Ok(action) => action,
            Err(source) => return Err(self.pending_authority_error(source)),
        };
        match action {
            SupervisorAction::Disarmed => Ok(action),
            action => Err(AgentManualControlError::UnexpectedSupervisorAction {
                operation: "complete explicit disarm",
                action,
            }),
        }
    }

    pub fn active_lease(&self) -> Option<AuthorityLease> {
        match (&self.lifecycle, self.authority.active_lease()) {
            (ManualLifecycle::Active(active), Some(supervisor_lease))
                if active.lease == supervisor_lease =>
            {
                Some(active.lease)
            }
            _ => None,
        }
    }

    /// Describe the manual portion of a process-wide stop without claiming
    /// that a zero has been sent or applied.
    ///
    /// The live owner must combine this with every other active mode, then
    /// complete the exact release/zero or fault obligation before responding
    /// `completed` to the control client. In particular, a begin already
    /// waiting for zero cannot be silently treated as inactive: admitting that
    /// zero would activate the pending lease.
    pub fn global_stop_requirement(&self) -> AgentManualGlobalStopRequirement {
        match &self.lifecycle {
            ManualLifecycle::Inactive => AgentManualGlobalStopRequirement::NoManualTransition,
            ManualLifecycle::AwaitingBeginZero => {
                AgentManualGlobalStopRequirement::PendingBeginMustBeCancelled
            }
            ManualLifecycle::AwaitingCancelledBeginZero => {
                AgentManualGlobalStopRequirement::FreshAppliedCancelledBeginZeroRequired
            }
            ManualLifecycle::Active(active) => AgentManualGlobalStopRequirement::ReleaseActive {
                lease_id: active.lease.id(),
            },
            ManualLifecycle::AwaitingReleaseZero { lease_id } => {
                AgentManualGlobalStopRequirement::FreshAppliedReleaseZeroRequired {
                    lease_id: *lease_id,
                }
            }
        }
    }

    /// Explicitly request manual authority. No velocity API calls this method.
    pub fn begin_manual(
        &mut self,
        now: HostMonotonicTimestamp,
    ) -> Result<BeginManualTransition, AgentManualControlError> {
        let policy = self.policy.ok_or(AgentManualControlError::ManualDisabled)?;
        if !matches!(self.lifecycle, ManualLifecycle::Inactive) {
            return Err(AgentManualControlError::ModeConflict);
        }
        if !matches!(self.authority.state(), SupervisorState::ReadyStopped { .. }) {
            return Err(AgentManualControlError::AuthorityNotReady {
                actual: self.authority.state().kind(),
            });
        }
        match self
            .authority
            .request_mode(AuthorityMode::Manual, policy.authority_lease, now)
            .map_err(AgentManualControlError::Authority)?
        {
            SupervisorAction::AuthorityGranted { lease } => {
                self.activate(lease, policy, now)?;
                Ok(BeginManualTransition::Granted { lease })
            }
            SupervisorAction::BaseZeroRequired { .. } => {
                self.lifecycle = ManualLifecycle::AwaitingBeginZero;
                Ok(BeginManualTransition::FreshAppliedZeroRequired)
            }
            action => Err(AgentManualControlError::UnexpectedSupervisorAction {
                operation: "begin manual",
                action,
            }),
        }
    }

    /// Complete a pending begin only with exact newly applied zero evidence.
    pub fn complete_begin_with_applied_zero(
        &mut self,
        result: HostCommandResult,
        observed_at: HostMonotonicTimestamp,
        now: HostMonotonicTimestamp,
    ) -> Result<AuthorityLease, AgentManualControlError> {
        if !matches!(self.lifecycle, ManualLifecycle::AwaitingBeginZero) {
            return Err(AgentManualControlError::NoPendingBegin);
        }
        let policy = self.policy.ok_or(AgentManualControlError::ManualDisabled)?;
        let action = match self.authority.admit_applied_zero(result, observed_at, now) {
            Ok(action) => action,
            Err(source) => return Err(self.pending_authority_error(source)),
        };
        match action {
            SupervisorAction::AuthorityGranted { lease } => {
                self.activate(lease, policy, now)?;
                Ok(lease)
            }
            SupervisorAction::PendingAuthorityExpired => {
                self.lifecycle = ManualLifecycle::Inactive;
                Err(AgentManualControlError::PendingAuthorityExpired)
            }
            action => Err(AgentManualControlError::UnexpectedSupervisorAction {
                operation: "complete manual begin zero",
                action,
            }),
        }
    }

    /// Cancel a manual lease which is still waiting for its begin-zero gate.
    ///
    /// The supervisor moves the zero barrier to `now` and replaces activation
    /// with a return to `ReadyStopped`. A newly applied zero is still required;
    /// cancellation itself never claims that hardware stopped.
    pub fn cancel_pending_begin(
        &mut self,
        now: HostMonotonicTimestamp,
    ) -> Result<SupervisorAction, AgentManualControlError> {
        if !matches!(self.lifecycle, ManualLifecycle::AwaitingBeginZero) {
            return Err(AgentManualControlError::NoPendingBegin);
        }
        let action = match self.authority.cancel_pending_authority(now) {
            Ok(action) => action,
            Err(source) => return Err(self.pending_authority_error(source)),
        };
        match action {
            SupervisorAction::BaseZeroRequired {
                reason: kiko_supervisor_core::StopReason::PendingAuthorityCancelled,
            } => {
                self.lifecycle = ManualLifecycle::AwaitingCancelledBeginZero;
                Ok(action)
            }
            action => Err(AgentManualControlError::UnexpectedSupervisorAction {
                operation: "cancel pending manual begin",
                action,
            }),
        }
    }

    /// Complete a cancelled begin with zero evidence produced strictly after
    /// cancellation. Success can only return the supervisor to `ReadyStopped`.
    pub fn complete_cancelled_begin_with_applied_zero(
        &mut self,
        result: HostCommandResult,
        observed_at: HostMonotonicTimestamp,
        now: HostMonotonicTimestamp,
    ) -> Result<(), AgentManualControlError> {
        if !matches!(self.lifecycle, ManualLifecycle::AwaitingCancelledBeginZero) {
            return Err(AgentManualControlError::NoPendingCancelledBegin);
        }
        let action = match self.authority.admit_applied_zero(result, observed_at, now) {
            Ok(action) => action,
            Err(source) => return Err(self.pending_authority_error(source)),
        };
        match action {
            SupervisorAction::ReadyStopped => {
                self.lifecycle = ManualLifecycle::Inactive;
                Ok(())
            }
            action => Err(AgentManualControlError::UnexpectedSupervisorAction {
                operation: "complete cancelled manual begin zero",
                action,
            }),
        }
    }

    pub fn ingest_velocity(
        &mut self,
        command: AgentManualVelocityV1,
        received_at: HostMonotonicTimestamp,
        observed_at: HostMonotonicTimestamp,
    ) -> Result<ManualDriveOutput<AuthorityLeaseId>, AgentManualControlError> {
        let lease_id = self.require_active_lease_id()?;
        self.ingest(
            command.bind_to_manual_lease(lease_id),
            received_at,
            observed_at,
        )
    }

    pub fn ingest_stop(
        &mut self,
        command: AgentManualStopV1,
        received_at: HostMonotonicTimestamp,
        observed_at: HostMonotonicTimestamp,
    ) -> Result<ManualDriveOutput<AuthorityLeaseId>, AgentManualControlError> {
        let lease_id = self.require_active_lease_id()?;
        self.ingest(
            command.bind_to_manual_lease(lease_id),
            received_at,
            observed_at,
        )
    }

    /// Advance lease expiry and deadman from the same host clock.
    pub fn tick(
        &mut self,
        now: HostMonotonicTimestamp,
    ) -> Result<ManualControlTick, AgentManualControlError> {
        let lease_id = self.require_active_lease_id()?;
        let mut supervisor_action = match self.authority.tick(now) {
            Ok(action) => action,
            Err(source) => return Err(self.fail_active_authority(source, lease_id)),
        };
        let snapshot = self.authority.manual_authority_snapshot();
        let ManualLifecycle::Active(active) = &mut self.lifecycle else {
            return Err(AgentManualControlError::ManualNotActive);
        };
        let output = active.core.tick(now, snapshot);
        if matches!(supervisor_action, SupervisorAction::BaseZeroRequired { .. }) {
            self.lifecycle = ManualLifecycle::AwaitingReleaseZero { lease_id };
        } else if matches!(output, ManualDriveOutput::Stopped(_)) {
            supervisor_action = match self.authority.release_authority(lease_id, now) {
                Ok(action @ SupervisorAction::BaseZeroRequired { .. }) => action,
                Ok(action) => {
                    self.lifecycle = ManualLifecycle::Inactive;
                    return Err(AgentManualControlError::UnexpectedSupervisorAction {
                        operation: "release manual after fail-closed tick",
                        action,
                    });
                }
                Err(source) => return Err(self.fail_active_authority(source, lease_id)),
            };
            self.lifecycle = ManualLifecycle::AwaitingReleaseZero { lease_id };
        }
        Ok(ManualControlTick {
            output,
            supervisor_action,
        })
    }

    /// Release the exact active lease and destroy its command/deadman state.
    pub fn begin_release(
        &mut self,
        now: HostMonotonicTimestamp,
    ) -> Result<AuthorityLeaseId, AgentManualControlError> {
        let lifecycle = std::mem::replace(&mut self.lifecycle, ManualLifecycle::Inactive);
        let ManualLifecycle::Active(active) = lifecycle else {
            self.lifecycle = lifecycle;
            return Err(AgentManualControlError::ManualNotActive);
        };
        let lease_id = active.lease.id();
        match self
            .authority
            .release_authority(lease_id, now)
            .map_err(AgentManualControlError::Authority)?
        {
            SupervisorAction::BaseZeroRequired { .. } => {
                self.lifecycle = ManualLifecycle::AwaitingReleaseZero { lease_id };
                Ok(lease_id)
            }
            action => Err(AgentManualControlError::UnexpectedSupervisorAction {
                operation: "release manual authority",
                action,
            }),
        }
    }

    /// A stopped coordinator decision is terminal for this manual session.
    /// Destroying the core prevents cached-target auto-resume after sensor
    /// recovery; the owner must apply and admit the resulting release zero.
    pub fn begin_safety_stop_release(
        &mut self,
        now: HostMonotonicTimestamp,
    ) -> Result<AuthorityLeaseId, AgentManualControlError> {
        self.begin_release(now)
    }

    /// Drop every manual command immediately and latch a typed supervisor
    /// fault. This is the controller reset/identity-change path after a
    /// preflight or apply failure makes further receipt use unsafe.
    pub fn latch_fault(
        &mut self,
        fault: FaultKind,
        now: HostMonotonicTimestamp,
    ) -> Result<SupervisorAction, AgentManualControlError> {
        self.lifecycle = ManualLifecycle::Inactive;
        let action = self
            .authority
            .latch_fault(fault, now)
            .map_err(AgentManualControlError::Authority)?;
        match action {
            SupervisorAction::FaultStopRequired { fault: actual } if actual == fault => Ok(action),
            action => Err(AgentManualControlError::UnexpectedSupervisorAction {
                operation: "latch manual/controller fault",
                action,
            }),
        }
    }

    /// Clear a handled fault only as far as inventory. No authority lease or
    /// command state can be restored in place.
    pub fn clear_fault_for_inventory(
        &mut self,
        now: HostMonotonicTimestamp,
    ) -> Result<SupervisorAction, AgentManualControlError> {
        self.lifecycle = ManualLifecycle::Inactive;
        self.authority
            .clear_fault_for_inventory(now)
            .map_err(AgentManualControlError::Authority)
    }

    /// Satisfy the release gate with the single zero receipt applied after the
    /// release request.
    pub fn complete_release_with_applied_zero(
        &mut self,
        result: HostCommandResult,
        observed_at: HostMonotonicTimestamp,
        now: HostMonotonicTimestamp,
    ) -> Result<AuthorityLeaseId, AgentManualControlError> {
        let ManualLifecycle::AwaitingReleaseZero { lease_id } = &self.lifecycle else {
            return Err(AgentManualControlError::NoPendingRelease);
        };
        let lease_id = *lease_id;
        let action = match self.authority.admit_applied_zero(result, observed_at, now) {
            Ok(action) => action,
            Err(source) => return Err(self.pending_authority_error(source)),
        };
        match action {
            SupervisorAction::ReadyStopped => {
                self.lifecycle = ManualLifecycle::Inactive;
                Ok(lease_id)
            }
            action => Err(AgentManualControlError::UnexpectedSupervisorAction {
                operation: "complete manual release zero",
                action,
            }),
        }
    }

    fn require_manual_inactive_for_autonomy(&self) -> Result<(), AgentAutonomousControlError> {
        if matches!(self.lifecycle, ManualLifecycle::Inactive) {
            Ok(())
        } else {
            Err(AgentAutonomousControlError::ManualLifecycleConflict)
        }
    }

    fn require_exact_autonomous(
        &self,
        expected: &AgentAutonomousAuthority,
    ) -> Result<(), AgentAutonomousControlError> {
        if expected.lease.mode() != expected.mode.authority_mode() {
            return Err(AgentAutonomousControlError::TokenModeMismatch {
                token_mode: expected.mode,
                lease_mode: expected.lease.mode(),
                obligation: self.authority.pending_obligation(),
            });
        }
        let actual = self.authority.active_lease();
        if actual == Some(expected.lease) {
            Ok(())
        } else {
            Err(AgentAutonomousControlError::ActiveLeaseMismatch {
                expected: expected.lease,
                actual: Box::new(actual),
                obligation: self.authority.pending_obligation(),
            })
        }
    }

    fn require_pending_reason(
        &self,
        operation: &'static str,
        expected: StopReason,
    ) -> Result<(), AgentAutonomousControlError> {
        let actual = self.authority.pending_obligation();
        if actual == (SupervisorAction::BaseZeroRequired { reason: expected }) {
            Ok(())
        } else {
            Err(AgentAutonomousControlError::PendingZeroObligationMismatch {
                operation,
                expected,
                actual,
            })
        }
    }

    fn autonomous_invariant(
        &mut self,
        operation: &'static str,
        action: SupervisorAction,
        now: HostMonotonicTimestamp,
    ) -> AgentAutonomousControlError {
        self.lifecycle = ManualLifecycle::Inactive;
        let latch = self
            .authority
            .latch_fault(FaultKind::InternalInvariant, now);
        AgentAutonomousControlError::UnexpectedSupervisorAction {
            operation,
            action,
            latch: Box::new(latch),
        }
    }

    fn pending_authority_error(&mut self, source: AgentAuthorityError) -> AgentManualControlError {
        if !matches!(self.authority.state(), SupervisorState::AwaitingZero { .. }) {
            self.lifecycle = ManualLifecycle::Inactive;
        }
        AgentManualControlError::Authority(source)
    }

    fn require_lifecycle_zero(
        &self,
        operation: &'static str,
        expected: StopReason,
    ) -> Result<(), AgentManualControlError> {
        if !matches!(self.lifecycle, ManualLifecycle::Inactive) {
            return Err(AgentManualControlError::ModeConflict);
        }
        let actual = self.authority.pending_obligation();
        if actual != (SupervisorAction::BaseZeroRequired { reason: expected }) {
            return Err(AgentManualControlError::LifecycleZeroObligationMismatch {
                operation,
                expected,
                actual,
            });
        }
        Ok(())
    }

    fn ingest(
        &mut self,
        command: super::ManualDriveParsedCommand<AuthorityLeaseId>,
        received_at: HostMonotonicTimestamp,
        observed_at: HostMonotonicTimestamp,
    ) -> Result<ManualDriveOutput<AuthorityLeaseId>, AgentManualControlError> {
        let policy = self.policy.ok_or(AgentManualControlError::ManualDisabled)?;
        let snapshot = self.authority.manual_authority_snapshot();
        let ManualLifecycle::Active(active) = &mut self.lifecycle else {
            return Err(AgentManualControlError::ManualNotActive);
        };
        let lease_id = active.lease.id();
        let output = active
            .core
            .ingest_parsed(command, received_at, observed_at, snapshot);
        if let ManualDriveOutput::Stopped(stopped) = output
            && matches!(
                stopped.cause(),
                ManualDriveStopCause::ClockRegression { .. }
                    | ManualDriveStopCause::ClockFaultLatched
            )
        {
            let cause = stopped.cause();
            self.lifecycle = ManualLifecycle::Inactive;
            let action = self
                .authority
                .latch_fault(FaultKind::ClockRegression, observed_at)
                .map_err(AgentManualControlError::Authority)?;
            return match action {
                SupervisorAction::FaultStopRequired {
                    fault: FaultKind::ClockRegression,
                } => Err(AgentManualControlError::ManualClockFault {
                    cause,
                    obligation: action,
                }),
                action => Err(AgentManualControlError::UnexpectedSupervisorAction {
                    operation: "latch manual ingress clock fault",
                    action,
                }),
            };
        }
        let ManualDriveOutput::Accepted(_) = output else {
            return Ok(output);
        };

        let action = match self.authority.request_mode(
            AuthorityMode::Manual,
            policy.authority_lease,
            observed_at,
        ) {
            Ok(action) => action,
            Err(source) => {
                return Err(self.fail_active_renewal(source, lease_id, observed_at));
            }
        };
        let SupervisorAction::AuthorityRenewed { lease } = action else {
            self.reconcile_active_obligation(action, lease_id);
            return Err(AgentManualControlError::UnexpectedSupervisorAction {
                operation: "renew admitted manual command",
                action,
            });
        };
        if lease.id() != lease_id {
            self.lifecycle = ManualLifecycle::Inactive;
            let obligation = self
                .authority
                .latch_fault(FaultKind::InternalInvariant, observed_at)
                .unwrap_or_else(|source| source.obligation());
            return Err(AgentManualControlError::RenewedLeaseIdentityChanged {
                previous: lease_id,
                renewed: lease.id(),
                obligation,
            });
        }
        let ManualLifecycle::Active(active) = &mut self.lifecycle else {
            return Err(AgentManualControlError::ManualNotActive);
        };
        active.lease = lease;
        Ok(active
            .core
            .admit_authority_renewal(observed_at, self.authority.manual_authority_snapshot()))
    }

    fn require_active_lease_id(&self) -> Result<AuthorityLeaseId, AgentManualControlError> {
        self.active_lease()
            .map(AuthorityLease::id)
            .ok_or(AgentManualControlError::ManualNotActive)
    }

    fn fail_active_authority(
        &mut self,
        source: AgentAuthorityError,
        lease_id: AuthorityLeaseId,
    ) -> AgentManualControlError {
        self.reconcile_active_obligation(source.obligation(), lease_id);
        AgentManualControlError::Authority(source)
    }

    fn fail_active_renewal(
        &mut self,
        source: AgentAuthorityError,
        lease_id: AuthorityLeaseId,
        now: HostMonotonicTimestamp,
    ) -> AgentManualControlError {
        self.reconcile_active_obligation(source.obligation(), lease_id);
        if !matches!(self.lifecycle, ManualLifecycle::Active(_)) {
            return AgentManualControlError::Authority(source);
        }

        // `ingest_parsed` has already accepted the command into the manual
        // core. If renewal then fails without moving the supervisor to an
        // explicit stop/fault state, retaining that core would make the cached
        // target tick-resumable. Destroy it and convert the still-active
        // supervisor state into a process-wide fault-stop obligation.
        self.lifecycle = ManualLifecycle::Inactive;
        match self
            .authority
            .latch_fault(FaultKind::InternalInvariant, now)
        {
            Ok(obligation) => {
                AgentManualControlError::AuthorityRenewalFaultLatched { source, obligation }
            }
            Err(escalation) => AgentManualControlError::AuthorityRenewalFaultEscalationFailed {
                source,
                escalation: Box::new(escalation),
            },
        }
    }

    fn reconcile_active_obligation(
        &mut self,
        obligation: SupervisorAction,
        lease_id: AuthorityLeaseId,
    ) {
        match obligation {
            SupervisorAction::BaseZeroRequired { .. } => {
                self.lifecycle = ManualLifecycle::AwaitingReleaseZero { lease_id };
            }
            SupervisorAction::FaultStopRequired { .. }
            | SupervisorAction::InventoryRequired
            | SupervisorAction::Disarmed
            | SupervisorAction::ReadyStopped
            | SupervisorAction::PendingAuthorityExpired => {
                self.lifecycle = ManualLifecycle::Inactive;
            }
            SupervisorAction::None
            | SupervisorAction::AuthorityGranted { .. }
            | SupervisorAction::AuthorityRenewed { .. } => {
                let still_exact = self.authority.active_lease().is_some_and(|lease| {
                    lease.id() == lease_id && lease.mode() == AuthorityMode::Manual
                });
                if !still_exact {
                    self.lifecycle = ManualLifecycle::Inactive;
                }
            }
        }
    }

    fn activate(
        &mut self,
        lease: AuthorityLease,
        policy: AgentManualRuntimePolicy,
        activated_at: HostMonotonicTimestamp,
    ) -> Result<(), AgentManualControlError> {
        if lease.mode() != AuthorityMode::Manual {
            return Err(AgentManualControlError::GrantedWrongMode {
                actual: lease.mode(),
            });
        }
        self.lifecycle = ManualLifecycle::Active(ActiveManualSession {
            lease,
            core: ManualDriveCore::new(policy.drive, lease.id(), activated_at),
        });
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BeginManualTransition {
    FreshAppliedZeroRequired,
    Granted { lease: AuthorityLease },
}

/// Manual contribution to one process-wide global-stop transaction.
///
/// This is an obligation snapshot, not physical stop evidence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AgentManualGlobalStopRequirement {
    NoManualTransition,
    PendingBeginMustBeCancelled,
    FreshAppliedCancelledBeginZeroRequired,
    ReleaseActive { lease_id: AuthorityLeaseId },
    FreshAppliedReleaseZeroRequired { lease_id: AuthorityLeaseId },
}

/// Stable typed reason retained from a consumed physical-session failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AgentLiveActuationFaultKind {
    TransportUnavailable,
    ControllerIdentityChanged,
    ControllerEvidenceRejected,
    ControllerSessionEnded,
    ReceiptDeadlineLost,
    HostClockRegression,
    HostInvariant,
    SafetyStopUncertain,
    SessionConsumed,
}

impl AgentLiveActuationFaultKind {
    /// Map exact transport evidence to the supervisor's intentionally smaller
    /// fault vocabulary without relabelling receipt timeouts as identity
    /// changes or host invariants.
    pub const fn supervisor_fault(self) -> FaultKind {
        match self {
            Self::ControllerIdentityChanged => FaultKind::ControllerIdentityChanged,
            Self::HostClockRegression => FaultKind::ClockRegression,
            Self::HostInvariant | Self::SessionConsumed => FaultKind::InternalInvariant,
            Self::TransportUnavailable
            | Self::ControllerEvidenceRejected
            | Self::ControllerSessionEnded
            | Self::ReceiptDeadlineLost
            | Self::SafetyStopUncertain => FaultKind::HardwareReadinessLost,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AgentControllerStopKnowledge {
    Confirmed,
    Uncertain,
}

impl From<LatchedStopKnowledge> for AgentControllerStopKnowledge {
    fn from(value: LatchedStopKnowledge) -> Self {
        match value {
            LatchedStopKnowledge::ConfirmedStop => Self::Confirmed,
            LatchedStopKnowledge::Unconfirmed => Self::Uncertain,
        }
    }
}

/// Fault plus the independent fact of whether recovery proved controller stop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AgentLiveActuationFault {
    kind: AgentLiveActuationFaultKind,
    controller_stop: AgentControllerStopKnowledge,
}

impl AgentLiveActuationFault {
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

/// Result of classifying a consumed physical-session failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AgentLiveActuationDisposition {
    /// Reinventory/fault handling is required before any new authority.
    LatchFault(AgentLiveActuationFault),
}

/// Classify the concrete physical bridge failure once at its ownership seam.
///
/// Receipt deadline failures remain distinguishable from transport loss and
/// controller identity changes. A confirmed stop is reported separately from
/// the fault kind; it never makes a failed command application successful.
pub fn classify_live_actuation_error(source: &LiveActuationError) -> AgentLiveActuationDisposition {
    match source {
        LiveActuationError::TransportBuild(_) => fault_disposition(
            AgentLiveActuationFaultKind::TransportUnavailable,
            AgentControllerStopKnowledge::Uncertain,
        ),
        LiveActuationError::Acquire(failure) => {
            classify_client_failure(failure.cause(), failure.stop_knowledge().into())
        }
        LiveActuationError::Preflight(failure) | LiveActuationError::Apply(failure) => {
            classify_client_failure(failure.cause(), failure.stop_knowledge().into())
        }
        LiveActuationError::Disarm(failure) => {
            classify_client_failure(failure.cause(), failure.stop_knowledge().into())
        }
        LiveActuationError::DecisionRejected { source, stop } => {
            let controller_stop = match stop {
                LocalRejectionStop::Confirmed(_) => AgentControllerStopKnowledge::Confirmed,
                LocalRejectionStop::DisarmFailed(failure) => failure.stop_knowledge().into(),
                LocalRejectionStop::SessionAlreadyConsumed => {
                    AgentControllerStopKnowledge::Uncertain
                }
            };
            classify_local_decision_rejection(source, controller_stop)
        }
        LiveActuationError::SessionConsumed => fault_disposition(
            AgentLiveActuationFaultKind::SessionConsumed,
            AgentControllerStopKnowledge::Uncertain,
        ),
    }
}

fn classify_local_decision_rejection(
    source: &PhysicalDecisionError,
    controller_stop: AgentControllerStopKnowledge,
) -> AgentLiveActuationDisposition {
    let kind = match (source, controller_stop) {
        (
            PhysicalDecisionError::CollisionValidityExpired { .. },
            AgentControllerStopKnowledge::Confirmed,
        ) => {
            // `reject_local_decision` obtained that stop by consuming and
            // disarming the physical session. The old readiness/control epoch
            // can no longer apply the fresh zero required for another lease.
            AgentLiveActuationFaultKind::ControllerSessionEnded
        }
        (PhysicalDecisionError::CollisionValidityExpired { .. }, _) => {
            AgentLiveActuationFaultKind::SafetyStopUncertain
        }
        _ => AgentLiveActuationFaultKind::HostInvariant,
    };
    fault_disposition(kind, controller_stop)
}

fn classify_client_failure<TransportError>(
    source: &FailureCause<TransportError>,
    controller_stop: AgentControllerStopKnowledge,
) -> AgentLiveActuationDisposition {
    let kind = match source {
        FailureCause::Transport(_) => AgentLiveActuationFaultKind::TransportUnavailable,
        FailureCause::Evidence(source) if evidence_is_identity_change(source) => {
            AgentLiveActuationFaultKind::ControllerIdentityChanged
        }
        FailureCause::Evidence(_) => AgentLiveActuationFaultKind::ControllerEvidenceRejected,
        FailureCause::ClockRegressed { .. } => AgentLiveActuationFaultKind::HostClockRegression,
        FailureCause::DeadlineExpiredBeforeSend { .. }
        | FailureCause::ResponseAtOrAfterDeadline { .. }
        | FailureCause::LeaseNotKnownActiveAtAcknowledgement { .. }
        | FailureCause::PreviousAppliedEvidenceExpired { .. } => {
            AgentLiveActuationFaultKind::ReceiptDeadlineLost
        }
        FailureCause::MonotonicArithmeticOverflow
        | FailureCause::RequestIdExhausted
        | FailureCause::CommandSequenceExhausted => AgentLiveActuationFaultKind::HostInvariant,
    };
    fault_disposition(kind, controller_stop)
}

fn evidence_is_identity_change(source: &EvidenceError) -> bool {
    matches!(
        source,
        EvidenceError::ControllerUidMismatch { .. }
            | EvidenceError::ControllerBootIdMismatch { .. }
            | EvidenceError::ControlEpochMismatch { .. }
    )
}

const fn fault_disposition(
    kind: AgentLiveActuationFaultKind,
    controller_stop: AgentControllerStopKnowledge,
) -> AgentLiveActuationDisposition {
    AgentLiveActuationDisposition::LatchFault(AgentLiveActuationFault {
        kind,
        controller_stop,
    })
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ManualControlTick {
    output: ManualDriveOutput<AuthorityLeaseId>,
    supervisor_action: SupervisorAction,
}

impl ManualControlTick {
    pub const fn output(self) -> ManualDriveOutput<AuthorityLeaseId> {
        self.output
    }

    pub const fn supervisor_action(self) -> SupervisorAction {
        self.supervisor_action
    }
}

#[derive(Debug)]
pub enum AgentAutonomousControlError {
    ManualLifecycleConflict,
    AuthorityNotReady {
        actual: SupervisorStateKind,
        obligation: SupervisorAction,
    },
    PendingAuthorityExpired,
    TokenModeMismatch {
        token_mode: AgentAutonomousMode,
        lease_mode: AuthorityMode,
        obligation: SupervisorAction,
    },
    ActiveLeaseMismatch {
        expected: AuthorityLease,
        actual: Box<Option<AuthorityLease>>,
        obligation: SupervisorAction,
    },
    PendingZeroObligationMismatch {
        operation: &'static str,
        expected: StopReason,
        actual: SupervisorAction,
    },
    LeaseExpired {
        pending: PendingAgentAutonomousStop,
    },
    UnexpectedSupervisorAction {
        operation: &'static str,
        action: SupervisorAction,
        latch: Box<Result<SupervisorAction, AgentAuthorityError>>,
    },
    Authority(AgentAuthorityError),
}

impl fmt::Display for AgentAutonomousControlError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "autonomous authority transition failed: {self:?}"
        )
    }
}

impl std::error::Error for AgentAutonomousControlError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Authority(source) => Some(source),
            Self::UnexpectedSupervisorAction { latch, .. } => match latch.as_ref() {
                Ok(_) => None,
                Err(source) => Some(source),
            },
            Self::ManualLifecycleConflict
            | Self::AuthorityNotReady { .. }
            | Self::PendingAuthorityExpired
            | Self::TokenModeMismatch { .. }
            | Self::ActiveLeaseMismatch { .. }
            | Self::PendingZeroObligationMismatch { .. }
            | Self::LeaseExpired { .. } => None,
        }
    }
}

#[derive(Debug)]
pub enum AgentManualControlError {
    ManualDisabled,
    ManualNotActive,
    ModeConflict,
    NoPendingBegin,
    NoPendingCancelledBegin,
    NoPendingRelease,
    LifecycleZeroObligationMismatch {
        operation: &'static str,
        expected: StopReason,
        actual: SupervisorAction,
    },
    PendingAuthorityExpired,
    ManualClockFault {
        cause: ManualDriveStopCause<AuthorityLeaseId>,
        obligation: SupervisorAction,
    },
    AuthorityNotReady {
        actual: SupervisorStateKind,
    },
    GrantedWrongMode {
        actual: AuthorityMode,
    },
    RenewedLeaseIdentityChanged {
        previous: AuthorityLeaseId,
        renewed: AuthorityLeaseId,
        obligation: SupervisorAction,
    },
    AuthorityRenewalFaultLatched {
        source: AgentAuthorityError,
        obligation: SupervisorAction,
    },
    AuthorityRenewalFaultEscalationFailed {
        source: AgentAuthorityError,
        escalation: Box<AgentAuthorityError>,
    },
    UnexpectedSupervisorAction {
        operation: &'static str,
        action: SupervisorAction,
    },
    Authority(AgentAuthorityError),
}

impl fmt::Display for AgentManualControlError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "manual control transition failed: {self:?}")
    }
}

impl std::error::Error for AgentManualControlError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Authority(source)
            | Self::AuthorityRenewalFaultLatched { source, .. }
            | Self::AuthorityRenewalFaultEscalationFailed { source, .. } => Some(source),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use kiko_supervisor_core::{
        ReadinessBinding, ReadinessEpoch, Sha256Digest, StopReason, SupervisorConfig,
        SupervisorError,
    };
    use robot_command_client::MonotonicInstant;
    use robot_protocol::ControllerUptimeMsWrapping;
    use robot_protocol::v2::{
        ControlEpoch, ControllerBootId, ControllerDeadlineMsWrapping, ControllerFaults,
        ControllerUid, HostCommandResultCode, OutputState, RemainingLeaseMs, TimerPwm,
        V2CommandSequence,
    };
    use serde_json::json;

    use super::*;
    use crate::navigation::{
        AGENT_CONTROL_SCHEMA_V1, AgentControlCommandV1, AgentControlRequestParser,
        MANUAL_DRIVE_CONFIG_V1, ManualDriveConfigV1Dto, ManualDriveStopCause, NavigationClockEpoch,
    };

    fn at(nanos: u64) -> HostMonotonicTimestamp {
        HostMonotonicTimestamp::from_nanos(nanos)
    }

    fn duration(nanos: u64) -> AuthorityDuration {
        AuthorityDuration::try_from_nanos(nanos).unwrap()
    }

    fn uid() -> ControllerUid {
        ControllerUid::try_new([1; 12]).unwrap()
    }

    fn boot() -> ControllerBootId {
        ControllerBootId::try_new(7).unwrap()
    }

    fn host_zero(sequence: u32) -> HostCommandResult {
        HostCommandResult {
            controller_uid: uid(),
            boot_id: boot(),
            control_epoch: ControlEpoch::try_new(9).unwrap(),
            sequence: V2CommandSequence::new(sequence),
            result: HostCommandResultCode::AppliedNew,
            requested_timer_pwm: TimerPwm::ZERO,
            controller_timer_pwm: TimerPwm::ZERO,
            output_state: OutputState::ZeroPwm,
            controller_applied_at: ControllerUptimeMsWrapping::new(10),
            controller_expires_at: ControllerDeadlineMsWrapping::new(20),
            remaining_lease: RemainingLeaseMs::ZERO,
            faults: ControllerFaults::NONE,
        }
    }

    fn disarmed() -> AgentAuthoritySupervisor {
        let config = SupervisorConfig::new(duration(1_000), duration(100)).unwrap();
        let mut authority = AgentAuthoritySupervisor::new(
            config,
            NavigationClockEpoch::new(HostMonotonicTimestamp::from_nanos(100)),
        );
        assert_eq!(
            authority.begin_inventory(at(101)).unwrap(),
            SupervisorAction::InventoryRequired
        );
        let readiness = ReadinessBinding::new(
            ReadinessEpoch::try_new(1).unwrap(),
            uid(),
            boot(),
            ControlEpoch::try_new(9).unwrap(),
            Sha256Digest::try_new([2; 32]).unwrap(),
            Sha256Digest::try_new([3; 32]).unwrap(),
        );
        assert_eq!(
            authority.admit_readiness(readiness, at(102)).unwrap(),
            SupervisorAction::Disarmed
        );
        authority
    }

    fn ready() -> AgentAuthoritySupervisor {
        let mut authority = disarmed();
        assert_eq!(
            authority.arm(at(103)).unwrap(),
            SupervisorAction::BaseZeroRequired {
                reason: StopReason::Arming
            }
        );
        assert_eq!(
            authority
                .admit_applied_zero(host_zero(0), at(104), at(104))
                .unwrap(),
            SupervisorAction::ReadyStopped
        );
        authority
    }

    #[test]
    fn autonomous_grant_renew_tick_and_release_require_exact_retained_tokens_and_zeros() {
        let mut control = AgentManualControlCore::new(ready(), Some(policy()));
        let pending = match control
            .request_autonomous(AgentAutonomousMode::PointGoal, duration(50), at(205))
            .expect("stale stopped evidence creates an exact grant continuation")
        {
            AgentAutonomousRequest::FreshAppliedZeroRequired(pending) => pending,
            AgentAutonomousRequest::Granted(_) => panic!("stale zero must not grant authority"),
        };
        assert_eq!(pending.mode(), AgentAutonomousMode::PointGoal);
        assert_eq!(pending.reason(), StopReason::ZeroEvidenceStale);

        let mut authority = control
            .complete_autonomous_grant_with_applied_zero(&pending, host_zero(1), at(206), at(206))
            .expect("fresh exact grant zero");
        let lease_id = authority.lease().id();
        assert_eq!(authority.mode(), AgentAutonomousMode::PointGoal);
        assert_eq!(control.authority().active_lease(), Some(authority.lease()));
        assert!(matches!(
            control.begin_manual(at(207)),
            Err(AgentManualControlError::AuthorityNotReady { .. })
        ));

        control
            .renew_autonomous(&mut authority, duration(50), at(207))
            .expect("same-token autonomous renewal");
        assert_eq!(authority.lease().id(), lease_id);
        assert!(matches!(
            control
                .tick_autonomous(&authority, at(208))
                .expect("active autonomous tick"),
            AgentAutonomousTick::Active
        ));

        let stop = control
            .begin_autonomous_release(&authority, at(209))
            .expect("exact autonomous release");
        assert_eq!(stop.mode(), AgentAutonomousMode::PointGoal);
        assert_eq!(stop.lease_id(), Some(lease_id));
        assert_eq!(stop.reason(), StopReason::AuthorityReleased);
        assert!(matches!(
            control.complete_autonomous_stop_with_applied_zero(
                &stop,
                host_zero(1),
                at(210),
                at(210),
            ),
            Err(AgentAutonomousControlError::Authority(_))
        ));
        control
            .complete_autonomous_stop_with_applied_zero(&stop, host_zero(2), at(211), at(211))
            .expect("new release zero");
        assert!(matches!(
            control.authority().state(),
            SupervisorState::ReadyStopped { .. }
        ));
    }

    #[test]
    fn pending_autonomous_grants_can_be_cancelled_or_expire_without_activating_motion() {
        let mut control = AgentManualControlCore::new(ready(), Some(policy()));
        let pending = match control
            .request_autonomous(AgentAutonomousMode::Explore, duration(50), at(205))
            .expect("stale stopped evidence")
        {
            AgentAutonomousRequest::FreshAppliedZeroRequired(pending) => pending,
            AgentAutonomousRequest::Granted(_) => panic!("stale zero must not grant"),
        };
        let stop = control
            .cancel_pending_autonomous_grant(&pending, at(206))
            .expect("cancel pending grant");
        assert_eq!(stop.mode(), AgentAutonomousMode::Explore);
        assert_eq!(stop.lease_id(), None);
        assert_eq!(stop.reason(), StopReason::PendingAuthorityCancelled);
        control
            .complete_autonomous_stop_with_applied_zero(&stop, host_zero(1), at(207), at(207))
            .expect("cancel stop zero");
        assert!(control.authority().active_lease().is_none());

        let pending = match control
            .request_autonomous(AgentAutonomousMode::PointGoal, duration(1), at(308))
            .expect("stale stopped evidence")
        {
            AgentAutonomousRequest::FreshAppliedZeroRequired(pending) => pending,
            AgentAutonomousRequest::Granted(_) => panic!("stale zero must not grant"),
        };
        assert!(matches!(
            control.complete_autonomous_grant_with_applied_zero(
                &pending,
                host_zero(2),
                at(310),
                at(310),
            ),
            Err(AgentAutonomousControlError::PendingAuthorityExpired)
        ));
        assert!(control.authority().active_lease().is_none());
        assert!(matches!(
            control.authority().state(),
            SupervisorState::ReadyStopped { .. }
        ));
    }

    #[test]
    fn explicit_arm_and_disarm_each_require_their_own_fresh_exact_zero() {
        let mut control = AgentManualControlCore::new(disarmed(), Some(policy()));

        assert_eq!(
            control.begin_arm(at(103)).expect("begin arm"),
            SupervisorAction::BaseZeroRequired {
                reason: StopReason::Arming,
            }
        );
        assert!(matches!(
            control.authority().state(),
            SupervisorState::AwaitingZero {
                reason: StopReason::Arming,
                ..
            }
        ));
        assert_eq!(
            control
                .complete_arm_with_applied_zero(host_zero(0), at(104), at(104))
                .expect("complete arm"),
            SupervisorAction::ReadyStopped
        );
        assert!(matches!(
            control.authority().state(),
            SupervisorState::ReadyStopped { .. }
        ));

        assert_eq!(
            control.begin_disarm(at(105)).expect("begin disarm"),
            SupervisorAction::BaseZeroRequired {
                reason: StopReason::ExplicitDisarm,
            }
        );
        assert!(matches!(
            control.complete_arm_with_applied_zero(host_zero(1), at(106), at(106)),
            Err(AgentManualControlError::LifecycleZeroObligationMismatch {
                operation: "complete explicit arm",
                expected: StopReason::Arming,
                actual: SupervisorAction::BaseZeroRequired {
                    reason: StopReason::ExplicitDisarm,
                },
            })
        ));
        assert_eq!(
            control
                .complete_disarm_with_applied_zero(host_zero(1), at(106), at(106))
                .expect("complete disarm"),
            SupervisorAction::Disarmed
        );
        assert!(matches!(
            control.authority().state(),
            SupervisorState::Disarmed { .. }
        ));
    }

    fn ready_near_timestamp_limit() -> AgentAuthoritySupervisor {
        const ORIGIN_NS: u64 = u64::MAX - 100;
        let config = SupervisorConfig::new(duration(100), duration(100)).unwrap();
        let mut authority =
            AgentAuthoritySupervisor::new(config, NavigationClockEpoch::new(at(ORIGIN_NS)));
        authority.begin_inventory(at(ORIGIN_NS + 1)).unwrap();
        let readiness = ReadinessBinding::new(
            ReadinessEpoch::try_new(1).unwrap(),
            uid(),
            boot(),
            ControlEpoch::try_new(9).unwrap(),
            Sha256Digest::try_new([2; 32]).unwrap(),
            Sha256Digest::try_new([3; 32]).unwrap(),
        );
        authority
            .admit_readiness(readiness, at(ORIGIN_NS + 2))
            .unwrap();
        authority.arm(at(ORIGIN_NS + 3)).unwrap();
        authority
            .admit_applied_zero(host_zero(0), at(ORIGIN_NS + 4), at(ORIGIN_NS + 4))
            .unwrap();
        authority
    }

    fn policy() -> AgentManualRuntimePolicy {
        AgentManualRuntimePolicy::for_test(
            duration(50),
            ManualDriveConfigV1::parse(ManualDriveConfigV1Dto {
                schema_version: MANUAL_DRIVE_CONFIG_V1,
                maximum_abs_forward_velocity_mps: 0.5,
                maximum_abs_yaw_rate_rad_s: 1.0,
                maximum_command_age_ns: 10,
                deadman_timeout_ns: 20,
            })
            .unwrap(),
        )
    }

    fn velocity(request_id: u64, sequence: u64) -> AgentManualVelocityV1 {
        let bytes = serde_json::to_vec(&json!({
            "schema_version": AGENT_CONTROL_SCHEMA_V1,
            "request_id": request_id,
            "command": {
                "kind": "manual_velocity",
                "sequence": sequence,
                "forward_velocity_mps": 0.2,
                "yaw_rate_rad_s": 0.1
            }
        }))
        .unwrap();
        let request = AgentControlRequestParser::new().parse_next(&bytes).unwrap();
        let AgentControlCommandV1::ManualVelocity(command) = request.command() else {
            panic!("velocity command")
        };
        command
    }

    fn stop(request_id: u64, sequence: u64) -> AgentManualStopV1 {
        let bytes = serde_json::to_vec(&json!({
            "schema_version": AGENT_CONTROL_SCHEMA_V1,
            "request_id": request_id,
            "command": {
                "kind": "manual_stop",
                "sequence": sequence
            }
        }))
        .unwrap();
        let request = AgentControlRequestParser::new().parse_next(&bytes).unwrap();
        let AgentControlCommandV1::ManualStop(command) = request.command() else {
            panic!("manual stop command")
        };
        command
    }

    #[test]
    fn begin_is_explicit_and_admitted_traffic_renews_same_lease_only_to_deadman() {
        let mut control = AgentManualControlCore::new(ready(), Some(policy()));
        let BeginManualTransition::Granted { lease } = control.begin_manual(at(105)).unwrap()
        else {
            panic!("fresh ready zero grants")
        };
        assert_eq!(lease.issued_at().as_nanos(), 105);
        assert_eq!(
            control.global_stop_requirement(),
            AgentManualGlobalStopRequirement::ReleaseActive {
                lease_id: lease.id(),
            }
        );
        assert!(matches!(
            control.begin_manual(at(106)),
            Err(AgentManualControlError::ModeConflict)
        ));

        let output = control
            .ingest_velocity(velocity(1, 1), at(106), at(106))
            .unwrap();
        let ManualDriveOutput::Accepted(target) = output else {
            panic!("fresh bounded velocity")
        };
        assert_eq!(target.authority_lease_id(), lease.id());
        assert_eq!(target.valid_through_exclusive(), at(126));
        let renewed = control.active_lease().unwrap();
        assert_eq!(renewed.id(), lease.id());
        assert_eq!(renewed.expires_at_exclusive().as_nanos(), 156);

        let tick = control.tick(at(126)).unwrap();
        assert_eq!(
            tick.supervisor_action(),
            SupervisorAction::BaseZeroRequired {
                reason: StopReason::AuthorityReleased,
            }
        );
        assert!(matches!(
            tick.output(),
            ManualDriveOutput::Stopped(stopped)
                if matches!(stopped.cause(), ManualDriveStopCause::DeadmanExpired { .. })
        ));
        assert_eq!(control.active_lease(), None);
        assert!(matches!(
            control.ingest_velocity(velocity(2, 2), at(127), at(127)),
            Err(AgentManualControlError::ManualNotActive)
        ));
        assert_eq!(
            control
                .complete_release_with_applied_zero(host_zero(1), at(128), at(128))
                .unwrap(),
            lease.id()
        );
    }

    #[test]
    fn queued_pre_session_command_stops_without_renewing_or_consuming_sequence() {
        let mut control = AgentManualControlCore::new(ready(), Some(policy()));
        let BeginManualTransition::Granted { lease } = control.begin_manual(at(105)).unwrap()
        else {
            panic!("manual grant")
        };
        let output = control
            .ingest_velocity(velocity(1, u64::MAX), at(104), at(105))
            .unwrap();
        assert!(matches!(
            output,
            ManualDriveOutput::Stopped(stopped)
                if matches!(
                    stopped.cause(),
                    ManualDriveStopCause::ReceiptBeforeManualSession { .. }
                )
        ));
        assert_eq!(
            control
                .active_lease()
                .unwrap()
                .expires_at_exclusive()
                .as_nanos(),
            lease.expires_at_exclusive().as_nanos()
        );
    }

    #[test]
    fn renewal_failure_destroys_cached_motion_and_latches_a_stop_obligation() {
        const BEGIN_NS: u64 = u64::MAX - 95;
        const COMMAND_NS: u64 = u64::MAX - 49;
        let mut control = AgentManualControlCore::new(ready_near_timestamp_limit(), Some(policy()));
        assert!(matches!(
            control.begin_manual(at(BEGIN_NS)),
            Ok(BeginManualTransition::Granted { .. })
        ));

        assert!(matches!(
            control.ingest_velocity(velocity(1, 1), at(COMMAND_NS), at(COMMAND_NS)),
            Err(AgentManualControlError::AuthorityRenewalFaultLatched {
                source: AgentAuthorityError::Supervisor {
                    source: SupervisorError::AuthorityDeadlineOverflow {
                        issued_at_ns: COMMAND_NS,
                        duration_ns: 50,
                    },
                    obligation: SupervisorAction::None,
                },
                obligation: SupervisorAction::FaultStopRequired {
                    fault: FaultKind::InternalInvariant,
                },
            })
        ));
        assert_eq!(control.active_lease(), None);
        assert_eq!(
            control.global_stop_requirement(),
            AgentManualGlobalStopRequirement::NoManualTransition
        );
        assert!(matches!(
            control.authority().state(),
            SupervisorState::FaultLatched {
                fault: FaultKind::InternalInvariant,
            }
        ));
        assert!(matches!(
            control.tick(at(COMMAND_NS)),
            Err(AgentManualControlError::ManualNotActive)
        ));
    }

    #[test]
    fn ordered_manual_stop_is_zero_but_retains_the_explicit_manual_session() {
        let mut control = AgentManualControlCore::new(ready(), Some(policy()));
        let BeginManualTransition::Granted { lease } = control.begin_manual(at(105)).unwrap()
        else {
            panic!("manual grant")
        };
        assert!(matches!(
            control
                .ingest_velocity(velocity(1, 1), at(106), at(106))
                .unwrap(),
            ManualDriveOutput::Accepted(_)
        ));
        let ManualDriveOutput::Accepted(accepted_stop) =
            control.ingest_stop(stop(2, 2), at(107), at(107)).unwrap()
        else {
            panic!("ordered manual stop")
        };
        let explicit = accepted_stop
            .into_explicit_stop()
            .expect("stop variant remains typed");
        assert_eq!(explicit.authority_lease_id(), lease.id());
        assert_eq!(explicit.sequence().get(), 2);
        assert_eq!(
            control.active_lease().map(|active| active.id()),
            Some(lease.id())
        );

        assert!(matches!(
            control
                .ingest_velocity(velocity(3, 3), at(108), at(108))
                .unwrap(),
            ManualDriveOutput::Accepted(target) if !target.target().is_stop()
        ));
    }

    #[test]
    fn stale_ready_zero_requires_new_evidence_and_release_requires_another_new_zero() {
        let mut control = AgentManualControlCore::new(ready(), Some(policy()));
        assert_eq!(
            control.begin_manual(at(204)).unwrap(),
            BeginManualTransition::FreshAppliedZeroRequired
        );
        assert_eq!(
            control.global_stop_requirement(),
            AgentManualGlobalStopRequirement::PendingBeginMustBeCancelled
        );
        let lease = control
            .complete_begin_with_applied_zero(host_zero(1), at(205), at(205))
            .unwrap();
        assert_eq!(control.active_lease(), Some(lease));
        assert!(matches!(
            control
                .ingest_velocity(velocity(1, 1), at(204), at(205))
                .unwrap(),
            ManualDriveOutput::Stopped(stopped)
                if matches!(
                    stopped.cause(),
                    ManualDriveStopCause::ReceiptBeforeManualSession {
                        session_started_at,
                        received_at,
                    } if session_started_at == at(205) && received_at == at(204)
                )
        ));

        assert_eq!(control.begin_release(at(206)).unwrap(), lease.id());
        assert_eq!(
            control.global_stop_requirement(),
            AgentManualGlobalStopRequirement::FreshAppliedReleaseZeroRequired {
                lease_id: lease.id(),
            }
        );
        assert!(matches!(
            control.complete_release_with_applied_zero(host_zero(1), at(207), at(207)),
            Err(AgentManualControlError::Authority(_))
        ));
        assert_eq!(
            control
                .complete_release_with_applied_zero(host_zero(2), at(208), at(208))
                .unwrap(),
            lease.id()
        );
        assert!(matches!(
            control.authority().state(),
            SupervisorState::ReadyStopped { .. }
        ));
    }

    #[test]
    fn global_stop_can_cancel_a_pending_begin_without_ever_granting_authority() {
        let mut control = AgentManualControlCore::new(ready(), Some(policy()));
        assert_eq!(
            control.begin_manual(at(204)).unwrap(),
            BeginManualTransition::FreshAppliedZeroRequired
        );
        assert_eq!(
            control.cancel_pending_begin(at(205)).unwrap(),
            SupervisorAction::BaseZeroRequired {
                reason: StopReason::PendingAuthorityCancelled,
            }
        );
        assert_eq!(
            control.global_stop_requirement(),
            AgentManualGlobalStopRequirement::FreshAppliedCancelledBeginZeroRequired
        );
        assert!(matches!(
            control.complete_cancelled_begin_with_applied_zero(host_zero(1), at(205), at(205)),
            Err(AgentManualControlError::Authority(
                AgentAuthorityError::Supervisor {
                    source:
                        kiko_supervisor_core::SupervisorError::ZeroEvidencePredatesStopRequest {
                            observed_at_ns: 205,
                            required_after_ns: 205,
                        },
                    obligation: SupervisorAction::BaseZeroRequired {
                        reason: StopReason::PendingAuthorityCancelled,
                    },
                }
            ))
        ));
        control
            .complete_cancelled_begin_with_applied_zero(host_zero(2), at(206), at(206))
            .unwrap();
        assert_eq!(control.active_lease(), None);
        assert_eq!(
            control.global_stop_requirement(),
            AgentManualGlobalStopRequirement::NoManualTransition
        );
        assert!(matches!(
            control.authority().state(),
            SupervisorState::ReadyStopped { .. }
        ));
    }

    #[test]
    fn clock_faults_during_pending_zero_completion_destroy_stale_lifecycle_claims() {
        let mut begin = AgentManualControlCore::new(ready(), Some(policy()));
        assert_eq!(
            begin.begin_manual(at(204)).unwrap(),
            BeginManualTransition::FreshAppliedZeroRequired
        );
        assert!(matches!(
            begin.complete_begin_with_applied_zero(host_zero(1), at(203), at(203)),
            Err(AgentManualControlError::Authority(
                AgentAuthorityError::ClockRegression { .. }
            ))
        ));
        assert_eq!(
            begin.global_stop_requirement(),
            AgentManualGlobalStopRequirement::NoManualTransition
        );
        assert!(matches!(
            begin.authority().state(),
            SupervisorState::FaultLatched {
                fault: FaultKind::ClockRegression
            }
        ));

        let mut release = AgentManualControlCore::new(ready(), Some(policy()));
        release.begin_manual(at(105)).unwrap();
        release.begin_release(at(106)).unwrap();
        assert!(matches!(
            release.complete_release_with_applied_zero(host_zero(1), at(105), at(105)),
            Err(AgentManualControlError::Authority(
                AgentAuthorityError::ClockRegression { .. }
            ))
        ));
        assert_eq!(
            release.global_stop_requirement(),
            AgentManualGlobalStopRequirement::NoManualTransition
        );
        assert!(matches!(
            release.authority().state(),
            SupervisorState::FaultLatched {
                fault: FaultKind::ClockRegression
            }
        ));
    }

    #[test]
    fn lease_expiry_drops_command_state_and_zero_completion_returns_ready() {
        let mut control = AgentManualControlCore::new(ready(), Some(policy()));
        let BeginManualTransition::Granted { lease } = control.begin_manual(at(105)).unwrap()
        else {
            panic!("manual grant")
        };
        assert_eq!(lease.expires_at_exclusive().as_nanos(), 155);
        let expired = control.tick(at(155)).unwrap();
        assert!(matches!(
            expired.supervisor_action(),
            SupervisorAction::BaseZeroRequired {
                reason: StopReason::AuthorityLeaseExpired
            }
        ));
        assert!(matches!(
            expired.output(),
            ManualDriveOutput::Stopped(stopped)
                if matches!(stopped.cause(), ManualDriveStopCause::AuthorityNotActiveManual)
        ));
        assert_eq!(control.active_lease(), None);
        assert_eq!(
            control
                .complete_release_with_applied_zero(host_zero(1), at(156), at(156))
                .unwrap(),
            lease.id()
        );
        assert!(matches!(
            control.authority().state(),
            SupervisorState::ReadyStopped { .. }
        ));
        assert!(matches!(
            control.begin_manual(at(157)),
            Ok(BeginManualTransition::Granted { .. })
        ));
    }

    #[test]
    fn pending_begin_timeout_never_activates_from_a_late_zero_receipt() {
        let mut control = AgentManualControlCore::new(ready(), Some(policy()));
        assert_eq!(
            control.begin_manual(at(204)).unwrap(),
            BeginManualTransition::FreshAppliedZeroRequired
        );
        assert!(matches!(
            control.complete_begin_with_applied_zero(host_zero(1), at(254), at(254)),
            Err(AgentManualControlError::PendingAuthorityExpired)
        ));
        assert_eq!(control.active_lease(), None);
        assert!(matches!(
            control.authority().state(),
            SupervisorState::ReadyStopped { .. }
        ));
    }

    #[test]
    fn disabled_policy_cannot_acquire_manual_authority() {
        let mut control = AgentManualControlCore::new(ready(), None);
        assert!(matches!(
            control.begin_manual(at(105)),
            Err(AgentManualControlError::ManualDisabled)
        ));
        assert_eq!(control.active_lease(), None);
    }

    #[test]
    fn controller_identity_fault_destroys_motion_and_requires_reinventory() {
        let mut control = AgentManualControlCore::new(ready(), Some(policy()));
        control.begin_manual(at(105)).unwrap();
        assert!(matches!(
            control
                .ingest_velocity(velocity(1, 1), at(106), at(106))
                .unwrap(),
            ManualDriveOutput::Accepted(_)
        ));
        assert_eq!(
            control
                .latch_fault(FaultKind::ControllerIdentityChanged, at(107))
                .unwrap(),
            SupervisorAction::FaultStopRequired {
                fault: FaultKind::ControllerIdentityChanged
            }
        );
        assert_eq!(control.active_lease(), None);
        assert!(matches!(
            control.ingest_velocity(velocity(2, 2), at(108), at(108)),
            Err(AgentManualControlError::ManualNotActive)
        ));
        assert!(matches!(
            control.authority().state(),
            SupervisorState::FaultLatched {
                fault: FaultKind::ControllerIdentityChanged
            }
        ));
        assert!(matches!(
            control.begin_manual(at(109)),
            Err(AgentManualControlError::AuthorityNotReady {
                actual: SupervisorStateKind::FaultLatched
            })
        ));
        assert_eq!(
            control.clear_fault_for_inventory(at(110)).unwrap(),
            SupervisorAction::InventoryRequired
        );
        assert!(matches!(
            control.authority().state(),
            SupervisorState::Inventory
        ));
    }

    #[test]
    fn hardware_readiness_fault_destroys_motion_and_requires_reinventory() {
        let mut control = AgentManualControlCore::new(ready(), Some(policy()));
        control.begin_manual(at(105)).unwrap();
        control
            .ingest_velocity(velocity(1, 1), at(106), at(106))
            .unwrap();
        assert_eq!(
            control
                .latch_fault(FaultKind::HardwareReadinessLost, at(107))
                .unwrap(),
            SupervisorAction::FaultStopRequired {
                fault: FaultKind::HardwareReadinessLost
            }
        );
        assert_eq!(control.active_lease(), None);
        assert!(matches!(
            control.authority().state(),
            SupervisorState::FaultLatched {
                fault: FaultKind::HardwareReadinessLost
            }
        ));
        assert!(matches!(
            control.ingest_velocity(velocity(2, 2), at(108), at(108)),
            Err(AgentManualControlError::ManualNotActive)
        ));
    }

    #[test]
    fn authority_clock_fault_destroys_the_manual_core_and_cannot_report_stale_active() {
        let mut control = AgentManualControlCore::new(ready(), Some(policy()));
        control.begin_manual(at(105)).unwrap();
        control
            .ingest_velocity(velocity(1, 1), at(106), at(106))
            .unwrap();

        assert!(matches!(
            control.tick(at(105)),
            Err(AgentManualControlError::Authority(
                AgentAuthorityError::ClockRegression {
                    obligation: SupervisorAction::FaultStopRequired {
                        fault: FaultKind::ClockRegression
                    },
                    ..
                }
            ))
        ));
        assert_eq!(control.active_lease(), None);
        assert!(matches!(
            control.authority().state(),
            SupervisorState::FaultLatched {
                fault: FaultKind::ClockRegression
            }
        ));
        assert!(matches!(
            control.ingest_velocity(velocity(2, 2), at(107), at(107)),
            Err(AgentManualControlError::ManualNotActive)
        ));
    }

    #[test]
    fn manual_ingress_clock_divergence_latches_fault_and_destroys_cached_target() {
        let mut control = AgentManualControlCore::new(ready(), Some(policy()));
        control.begin_manual(at(105)).unwrap();
        assert!(matches!(
            control
                .ingest_velocity(velocity(1, 1), at(106), at(106))
                .unwrap(),
            ManualDriveOutput::Accepted(_)
        ));

        // A duplicate is stopped by the manual stream and advances only its
        // observation clock; rejected traffic cannot renew authority.
        assert!(matches!(
            control
                .ingest_velocity(velocity(2, 1), at(107), at(107))
                .unwrap(),
            ManualDriveOutput::Stopped(stopped)
                if matches!(stopped.cause(), ManualDriveStopCause::DuplicateSequence { .. })
        ));
        assert!(matches!(
            control.ingest_velocity(velocity(3, 2), at(106), at(106)),
            Err(AgentManualControlError::ManualClockFault {
                cause: ManualDriveStopCause::ClockRegression { .. },
                obligation: SupervisorAction::FaultStopRequired {
                    fault: FaultKind::ClockRegression,
                },
            })
        ));
        assert_eq!(control.active_lease(), None);
        assert!(matches!(
            control.authority().state(),
            SupervisorState::FaultLatched {
                fault: FaultKind::ClockRegression
            }
        ));
        assert!(matches!(
            control.ingest_velocity(velocity(4, 3), at(108), at(108)),
            Err(AgentManualControlError::ManualNotActive)
        ));
    }

    #[test]
    fn coordinator_safety_stop_terminates_session_instead_of_auto_resuming_cache() {
        let mut control = AgentManualControlCore::new(ready(), Some(policy()));
        let BeginManualTransition::Granted { lease } = control.begin_manual(at(105)).unwrap()
        else {
            panic!("manual grant")
        };
        assert!(matches!(
            control
                .ingest_velocity(velocity(1, 7), at(106), at(106))
                .unwrap(),
            ManualDriveOutput::Accepted(_)
        ));
        assert_eq!(
            control.begin_safety_stop_release(at(107)).unwrap(),
            lease.id()
        );
        assert_eq!(control.active_lease(), None);
        assert!(matches!(
            control.ingest_velocity(velocity(2, 7), at(108), at(108)),
            Err(AgentManualControlError::ManualNotActive)
        ));
        control
            .complete_release_with_applied_zero(host_zero(1), at(109), at(109))
            .unwrap();
        assert!(matches!(
            control.begin_manual(at(110)),
            Ok(BeginManualTransition::Granted { .. })
        ));
    }

    fn classified_fault(
        source: &FailureCause<Infallible>,
        controller_stop: AgentControllerStopKnowledge,
    ) -> AgentLiveActuationFault {
        let AgentLiveActuationDisposition::LatchFault(fault) =
            classify_client_failure(source, controller_stop);
        fault
    }

    #[test]
    fn physical_receipt_deadlines_remain_typed_and_require_reinventory() {
        let deadline = MonotonicInstant::from_nanos_since_clock_start(200);
        let acknowledged = MonotonicInstant::from_nanos_since_clock_start(200);
        for source in [
            FailureCause::<Infallible>::DeadlineExpiredBeforeSend {
                now: deadline,
                deadline_exclusive: deadline,
            },
            FailureCause::ResponseAtOrAfterDeadline {
                acknowledged_at: acknowledged,
                deadline_exclusive: deadline,
            },
            FailureCause::LeaseNotKnownActiveAtAcknowledgement {
                acknowledged_at: acknowledged,
                known_active_through_exclusive: deadline,
            },
            FailureCause::PreviousAppliedEvidenceExpired {
                now: deadline,
                known_active_through_exclusive: deadline,
            },
        ] {
            let fault = classified_fault(&source, AgentControllerStopKnowledge::Confirmed);
            assert_eq!(
                fault.kind(),
                AgentLiveActuationFaultKind::ReceiptDeadlineLost
            );
            assert_eq!(fault.supervisor_fault(), FaultKind::HardwareReadinessLost);
            assert_eq!(
                fault.controller_stop(),
                AgentControllerStopKnowledge::Confirmed,
                "confirmed recovery stop does not turn the failed receipt into success"
            );
        }
    }

    #[test]
    fn physical_identity_clock_and_host_invariants_are_not_flattened() {
        let identity =
            FailureCause::<Infallible>::Evidence(EvidenceError::ControllerBootIdMismatch {
                expected: boot(),
                actual: ControllerBootId::try_new(8).unwrap(),
            });
        let identity = classified_fault(&identity, AgentControllerStopKnowledge::Uncertain);
        assert_eq!(
            identity.kind(),
            AgentLiveActuationFaultKind::ControllerIdentityChanged
        );
        assert_eq!(
            identity.supervisor_fault(),
            FaultKind::ControllerIdentityChanged
        );

        let clock = FailureCause::<Infallible>::ClockRegressed {
            previous: MonotonicInstant::from_nanos_since_clock_start(11),
            observed: MonotonicInstant::from_nanos_since_clock_start(10),
        };
        let clock = classified_fault(&clock, AgentControllerStopKnowledge::Confirmed);
        assert_eq!(
            clock.kind(),
            AgentLiveActuationFaultKind::HostClockRegression
        );
        assert_eq!(clock.supervisor_fault(), FaultKind::ClockRegression);

        let exhausted = FailureCause::<Infallible>::CommandSequenceExhausted;
        let exhausted = classified_fault(&exhausted, AgentControllerStopKnowledge::Confirmed);
        assert_eq!(exhausted.kind(), AgentLiveActuationFaultKind::HostInvariant);
        assert_eq!(exhausted.supervisor_fault(), FaultKind::InternalInvariant);
    }

    #[test]
    fn confirmed_local_expiry_stop_still_requires_a_new_controller_session() {
        let source = PhysicalDecisionError::CollisionValidityExpired {
            now: at(20),
            collision_valid_through: at(20),
        };
        let AgentLiveActuationDisposition::LatchFault(confirmed) =
            classify_local_decision_rejection(&source, AgentControllerStopKnowledge::Confirmed);
        assert_eq!(
            confirmed.kind(),
            AgentLiveActuationFaultKind::ControllerSessionEnded
        );
        assert_eq!(
            confirmed.supervisor_fault(),
            FaultKind::HardwareReadinessLost
        );
        assert_eq!(
            confirmed.controller_stop(),
            AgentControllerStopKnowledge::Confirmed
        );

        let AgentLiveActuationDisposition::LatchFault(uncertain) =
            classify_local_decision_rejection(&source, AgentControllerStopKnowledge::Uncertain);
        assert_eq!(
            uncertain.kind(),
            AgentLiveActuationFaultKind::SafetyStopUncertain
        );
        assert_eq!(
            uncertain.supervisor_fault(),
            FaultKind::HardwareReadinessLost
        );
    }

    #[test]
    fn missing_or_busy_identity_evidence_is_not_mislabeled_as_identity_change() {
        for evidence in [
            EvidenceError::StatusHasControlEpoch(ControlEpoch::try_new(9).unwrap()),
            EvidenceError::StatusHasNoExactBootId,
            EvidenceError::GrantedAcquireMissingEpoch,
            EvidenceError::StopResultHasNoExactBootId,
        ] {
            let source = FailureCause::<Infallible>::Evidence(evidence);
            let fault = classified_fault(&source, AgentControllerStopKnowledge::Uncertain);
            assert_eq!(
                fault.kind(),
                AgentLiveActuationFaultKind::ControllerEvidenceRejected
            );
            assert_eq!(fault.supervisor_fault(), FaultKind::HardwareReadinessLost);
        }
    }
}
