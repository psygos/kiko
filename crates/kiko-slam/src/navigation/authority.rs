//! Sole, transport-free adapter between the agent runtime and motion authority.
//!
//! The adapter intentionally exposes no actuator transport. Every returned
//! [`SupervisorAction`] is an obligation for the runtime to perform; it is
//! never evidence that inventory, a base-zero command, a hardware stop, or an
//! authority change actually happened. In particular,
//! [`SupervisorAction::BaseZeroRequired`] remains pending until
//! [`Self::admit_applied_zero`] accepts a new exact host command result.
//!
//! All timestamps accepted here are from the one host-process monotonic epoch
//! supplied at construction. The adapter preserves those absolute nanosecond
//! values when crossing into `kiko-supervisor-core`, so the exclusive authority
//! deadline copied into [`ManualAuthoritySnapshot`] is directly comparable to
//! the timestamps used by [`super::ManualDriveCore`].

use core::fmt;

use kiko_supervisor_core::{
    AuthorityDuration, AuthorityLease, AuthorityLeaseId, AuthorityMode, ConfirmedBaseZero,
    FaultKind, MonotonicInstant, ReadinessBinding, ReadinessEpoch, RobotSupervisor,
    SupervisorAction, SupervisorConfig, SupervisorError, SupervisorState, SupervisorStateKind,
    ZeroEvidenceError,
};
use robot_protocol::v2::{HostCommandResult, V2CommandSequence};

use super::{
    AgentBaseCommandStateV1, AgentControlStatusV1, AgentMapStateV1, AgentOperatingModeV1,
    AgentRuntimeStateV1, ManualAuthoritySnapshot, NavigationClockEpoch,
};
use crate::HostMonotonicTimestamp;

/// Fail-closed owner of lifecycle, exclusive motion mode, and lease identity.
///
/// Lease IDs are monotonically allocated for the lifetime of this value and
/// are never reset by reinventory. Exhaustion latches an internal fault before
/// it is reported. The adapter is deliberately not `Clone`: one process must
/// have exactly one mutable owner.
pub struct AgentAuthoritySupervisor {
    supervisor: RobotSupervisor,
    clock_epoch: NavigationClockEpoch,
    last_observed_at: HostMonotonicTimestamp,
    last_readiness_epoch: Option<ReadinessEpoch>,
    next_lease_id: Option<u64>,
    last_zero_receipt: Option<ZeroReceiptIdentity>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ZeroReceiptIdentity {
    controller_uid: robot_protocol::v2::ControllerUid,
    controller_boot_id: robot_protocol::v2::ControllerBootId,
    control_epoch: robot_protocol::v2::ControlEpoch,
    sequence: V2CommandSequence,
}

impl ZeroReceiptIdentity {
    const fn from_zero(zero: ConfirmedBaseZero) -> Self {
        Self {
            controller_uid: zero.controller_uid(),
            controller_boot_id: zero.controller_boot_id(),
            control_epoch: zero.control_epoch(),
            sequence: zero.sequence(),
        }
    }

    fn same_controller_epoch(self, other: Self) -> bool {
        self.controller_uid == other.controller_uid
            && self.controller_boot_id == other.controller_boot_id
            && self.control_epoch == other.control_epoch
    }
}

impl AgentAuthoritySupervisor {
    /// Create a booting supervisor bound to one live host-clock epoch.
    pub fn new(config: SupervisorConfig, clock_epoch: NavigationClockEpoch) -> Self {
        Self {
            supervisor: RobotSupervisor::new(config),
            clock_epoch,
            last_observed_at: clock_epoch.origin(),
            last_readiness_epoch: None,
            next_lease_id: Some(1),
            last_zero_receipt: None,
        }
    }

    /// Return the exact underlying lifecycle state without implying I/O.
    pub const fn state(&self) -> SupervisorState {
        self.supervisor.state()
    }

    /// Return the host monotonic epoch to which every adapter timestamp belongs.
    pub const fn clock_epoch(&self) -> NavigationClockEpoch {
        self.clock_epoch
    }

    /// Return the active lease, if and only if the supervisor is active.
    pub const fn active_lease(&self) -> Option<AuthorityLease> {
        match self.supervisor.state() {
            SupervisorState::Active { lease, .. } => Some(lease),
            _ => None,
        }
    }

    /// Project the exact active manual lease into `ManualDriveCore`'s input.
    ///
    /// The exclusive deadline is copied, not recomputed. An expired lease may
    /// remain visible until [`Self::tick`] runs, but `ManualDriveCore` compares
    /// the same deadline against its observation time and therefore stops.
    pub fn manual_authority_snapshot(&self) -> ManualAuthoritySnapshot<AuthorityLeaseId> {
        match self.supervisor.state() {
            SupervisorState::Active { lease, .. } if lease.mode() == AuthorityMode::Manual => {
                ManualAuthoritySnapshot::active_manual(
                    lease.id(),
                    HostMonotonicTimestamp::from_nanos(lease.expires_at_exclusive().as_nanos()),
                )
            }
            _ => ManualAuthoritySnapshot::NotActiveManual,
        }
    }

    /// Return the currently outstanding fail-closed obligation.
    ///
    /// `None` means only that the supervisor has no repeated lifecycle action
    /// to request. It does not prove that hardware is healthy or stationary.
    pub const fn pending_obligation(&self) -> SupervisorAction {
        match self.supervisor.state() {
            SupervisorState::Inventory => SupervisorAction::InventoryRequired,
            SupervisorState::AwaitingZero { reason, .. } => {
                SupervisorAction::BaseZeroRequired { reason }
            }
            SupervisorState::FaultLatched { fault } => {
                SupervisorAction::FaultStopRequired { fault }
            }
            _ => SupervisorAction::None,
        }
    }

    /// Enter the inventory gate. The returned action requests inventory; it
    /// does not claim inventory has run.
    pub fn begin_inventory(
        &mut self,
        now: HostMonotonicTimestamp,
    ) -> Result<SupervisorAction, AgentAuthorityError> {
        let now = self.observe_time(now)?;
        let result = self.supervisor.begin_inventory(now);
        self.finish(result)
    }

    /// Admit an externally established readiness binding.
    ///
    /// Readiness epochs must strictly increase for this process lifetime,
    /// including after a fault and reinventory. The inventory owner remains
    /// responsible for proving the binding's device and artifact contents.
    pub fn admit_readiness(
        &mut self,
        readiness: ReadinessBinding,
        now: HostMonotonicTimestamp,
    ) -> Result<SupervisorAction, AgentAuthorityError> {
        let now = self.observe_time(now)?;
        if let Some(previous) = self.last_readiness_epoch
            && readiness.epoch().get() <= previous.get()
        {
            return Err(AgentAuthorityError::ReadinessEpochNotIncreasing {
                previous,
                actual: readiness.epoch(),
                obligation: self.pending_obligation(),
            });
        }
        let result = self.supervisor.admit_readiness(readiness, now);
        let action = self.finish(result)?;
        self.last_readiness_epoch = Some(readiness.epoch());
        Ok(action)
    }

    /// Arm only as far as the mandatory post-arm zero gate.
    pub fn arm(
        &mut self,
        now: HostMonotonicTimestamp,
    ) -> Result<SupervisorAction, AgentAuthorityError> {
        let now = self.observe_time(now)?;
        let result = self.supervisor.arm(now);
        self.finish(result)
    }

    /// Parse and admit one exact base-zero receipt from the host command path.
    ///
    /// This is the adapter's only zero-admission API. The weak result is parsed
    /// exactly once through
    /// [`ConfirmedBaseZero::try_from_host_command_result`]. A result must prove
    /// a requested zero was newly applied (or explicitly stopped), be newer
    /// than the stop request, match readiness, and carry a strictly increasing
    /// controller command sequence within the same controller/control epoch.
    pub fn admit_applied_zero(
        &mut self,
        result: HostCommandResult,
        observed_at: HostMonotonicTimestamp,
        now: HostMonotonicTimestamp,
    ) -> Result<SupervisorAction, AgentAuthorityError> {
        let now = self.observe_time(now)?;
        let zero = ConfirmedBaseZero::try_from_host_command_result(
            result,
            to_supervisor_time(observed_at),
        )
        .map_err(|source| AgentAuthorityError::ZeroEvidence {
            source,
            obligation: self.pending_obligation(),
        })?;
        let identity = ZeroReceiptIdentity::from_zero(zero);
        if let Some(previous) = self.last_zero_receipt
            && identity.same_controller_epoch(previous)
            && identity.sequence.get() <= previous.sequence.get()
        {
            return Err(AgentAuthorityError::ZeroReceiptNotNew {
                previous_sequence: previous.sequence,
                actual_sequence: identity.sequence,
                obligation: self.pending_obligation(),
            });
        }

        let admitted = self.supervisor.admit_confirmed_zero(zero, now);
        let action = self.finish(admitted)?;
        self.last_zero_receipt = Some(identity);
        Ok(action)
    }

    /// Acquire, renew, or hand over the sole motion authority.
    ///
    /// A same-mode request renews the exact active lease ID. A different mode
    /// allocates a new ID and enters the mandatory handover-zero gate. From
    /// `ReadyStopped`, a new ID is allocated and normal freshness rules decide
    /// whether it can be granted immediately. No two modes can be active.
    pub fn request_mode(
        &mut self,
        mode: AuthorityMode,
        duration: AuthorityDuration,
        now: HostMonotonicTimestamp,
    ) -> Result<SupervisorAction, AgentAuthorityError> {
        let now = self.observe_time(now)?;
        let result = match self.supervisor.state() {
            SupervisorState::ReadyStopped { .. } => {
                let id = self.allocate_lease_id(now)?;
                self.supervisor.request_authority(id, mode, duration, now)
            }
            SupervisorState::Active { lease, .. } if lease.mode() == mode => {
                self.supervisor.renew_authority(lease.id(), duration, now)
            }
            SupervisorState::Active { .. } => {
                let id = self.allocate_lease_id(now)?;
                self.supervisor.handover(id, mode, duration, now)
            }
            state => {
                return Err(AgentAuthorityError::AuthorityUnavailable {
                    state: state.kind(),
                    obligation: self.pending_obligation(),
                });
            }
        };
        self.finish(result)
    }

    /// Release the exact active lease and require a newly applied zero.
    pub fn release_authority(
        &mut self,
        id: AuthorityLeaseId,
        now: HostMonotonicTimestamp,
    ) -> Result<SupervisorAction, AgentAuthorityError> {
        let now = self.observe_time(now)?;
        let result = self.supervisor.release_authority(id, now);
        self.finish(result)
    }

    /// Disarm through the core's explicit post-stop zero gate.
    pub fn disarm(
        &mut self,
        now: HostMonotonicTimestamp,
    ) -> Result<SupervisorAction, AgentAuthorityError> {
        let now = self.observe_time(now)?;
        let result = self.supervisor.disarm(now);
        self.finish(result)
    }

    /// Advance expiry handling. An expired active lease always becomes a
    /// base-zero obligation; ticking alone is not evidence that it stopped.
    pub fn tick(
        &mut self,
        now: HostMonotonicTimestamp,
    ) -> Result<SupervisorAction, AgentAuthorityError> {
        let now = self.observe_time(now)?;
        let result = self.supervisor.tick(now);
        self.finish(result)
    }

    /// Latch a typed fault and return the corresponding stop obligation.
    pub fn latch_fault(
        &mut self,
        fault: FaultKind,
        now: HostMonotonicTimestamp,
    ) -> Result<SupervisorAction, AgentAuthorityError> {
        let now = self.observe_time(now)?;
        let result = self.supervisor.latch_fault(fault, now);
        self.finish(result)
    }

    /// Clear a fault only as far as the inventory gate. Prior readiness and
    /// authority cannot be restored in place.
    pub fn clear_fault_for_inventory(
        &mut self,
        now: HostMonotonicTimestamp,
    ) -> Result<SupervisorAction, AgentAuthorityError> {
        let now = self.observe_time(now)?;
        let result = self.supervisor.clear_fault_for_inventory(now);
        self.finish(result)
    }

    /// Conservatively project supervisor state into the fixed control status.
    ///
    /// Only `ReadyStopped` has retained exact zero evidence, so it is the only
    /// state reported as [`AgentBaseCommandStateV1::ConfirmedStopped`]. Active
    /// authority does not imply that a hardware command was sent, and awaiting
    /// zero does not imply that the stop obligation was executed; both report
    /// `Unknown`. Lifecycle and operating-mode variants are mapped exactly.
    pub fn control_status(&self, map: AgentMapStateV1) -> AgentControlStatusV1 {
        let (runtime, base_command) = match self.supervisor.state() {
            SupervisorState::Booting => (
                AgentRuntimeStateV1::Booting,
                AgentBaseCommandStateV1::Unknown,
            ),
            SupervisorState::Inventory => (
                AgentRuntimeStateV1::Inventory,
                AgentBaseCommandStateV1::Unknown,
            ),
            SupervisorState::Disarmed { .. } => (
                AgentRuntimeStateV1::Disarmed,
                AgentBaseCommandStateV1::Unknown,
            ),
            SupervisorState::AwaitingZero { .. } => (
                AgentRuntimeStateV1::AwaitingZero,
                AgentBaseCommandStateV1::Unknown,
            ),
            SupervisorState::ReadyStopped { .. } => (
                AgentRuntimeStateV1::ReadyStopped,
                AgentBaseCommandStateV1::ConfirmedStopped,
            ),
            SupervisorState::Active { lease, .. } => {
                let mode = match lease.mode() {
                    AuthorityMode::Manual => AgentOperatingModeV1::Manual,
                    AuthorityMode::PointGoal => AgentOperatingModeV1::PointGoal,
                    AuthorityMode::Explore => AgentOperatingModeV1::FrontierExplore,
                    AuthorityMode::Commissioning => AgentOperatingModeV1::Commissioning,
                };
                (
                    AgentRuntimeStateV1::Active { mode },
                    AgentBaseCommandStateV1::Unknown,
                )
            }
            SupervisorState::FaultLatched { .. } => (
                AgentRuntimeStateV1::Faulted,
                AgentBaseCommandStateV1::Unknown,
            ),
        };
        AgentControlStatusV1::new(runtime, base_command, map)
    }

    fn observe_time(
        &mut self,
        now: HostMonotonicTimestamp,
    ) -> Result<MonotonicInstant, AgentAuthorityError> {
        let origin_ns = self.clock_epoch.origin().as_nanos();
        let actual_ns = now.as_nanos();
        let previous_ns = self.last_observed_at.as_nanos();
        if actual_ns < origin_ns {
            let obligation = self.latch_clock_fault_at_last_time()?;
            return Err(AgentAuthorityError::TimeBeforeClockEpoch {
                origin_ns,
                actual_ns,
                obligation,
            });
        }
        if actual_ns < previous_ns {
            let obligation = self.latch_clock_fault_at_last_time()?;
            return Err(AgentAuthorityError::ClockRegression {
                previous_ns,
                actual_ns,
                obligation,
            });
        }
        self.last_observed_at = now;
        Ok(to_supervisor_time(now))
    }

    fn latch_clock_fault_at_last_time(&mut self) -> Result<SupervisorAction, AgentAuthorityError> {
        let result = self.supervisor.latch_fault(
            FaultKind::ClockRegression,
            to_supervisor_time(self.last_observed_at),
        );
        self.finish(result)
    }

    fn allocate_lease_id(
        &mut self,
        now: MonotonicInstant,
    ) -> Result<AuthorityLeaseId, AgentAuthorityError> {
        let Some(raw) = self.next_lease_id else {
            let result = self
                .supervisor
                .latch_fault(FaultKind::InternalInvariant, now);
            let obligation = self.finish(result)?;
            return Err(AgentAuthorityError::LeaseIdExhausted { obligation });
        };
        let id =
            AuthorityLeaseId::try_new(raw).map_err(|source| AgentAuthorityError::Supervisor {
                source,
                obligation: self.pending_obligation(),
            })?;
        self.next_lease_id = raw.checked_add(1);
        Ok(id)
    }

    fn finish(
        &self,
        result: Result<SupervisorAction, SupervisorError>,
    ) -> Result<SupervisorAction, AgentAuthorityError> {
        result.map_err(|source| AgentAuthorityError::Supervisor {
            source,
            obligation: self.pending_obligation(),
        })
    }

    #[cfg(test)]
    fn set_next_lease_id_for_test(&mut self, next: Option<u64>) {
        self.next_lease_id = next;
    }
}

fn to_supervisor_time(timestamp: HostMonotonicTimestamp) -> MonotonicInstant {
    MonotonicInstant::from_nanos_since_process_start(timestamp.as_nanos())
}

/// Exact adapter failure plus any action the runtime still owes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AgentAuthorityError {
    /// The underlying lifecycle rejected a transition.
    Supervisor {
        source: SupervisorError,
        obligation: SupervisorAction,
    },
    /// The exact host result did not prove a newly applied safe zero.
    ZeroEvidence {
        source: ZeroEvidenceError,
        obligation: SupervisorAction,
    },
    /// A replayed or regressed controller sequence cannot satisfy a new gate.
    ZeroReceiptNotNew {
        previous_sequence: V2CommandSequence,
        actual_sequence: V2CommandSequence,
        obligation: SupervisorAction,
    },
    /// A readiness binding cannot reuse or regress its process-lifetime epoch.
    ReadinessEpochNotIncreasing {
        previous: ReadinessEpoch,
        actual: ReadinessEpoch,
        obligation: SupervisorAction,
    },
    /// A timestamp did not belong to the configured live clock epoch.
    TimeBeforeClockEpoch {
        origin_ns: u64,
        actual_ns: u64,
        obligation: SupervisorAction,
    },
    /// A timestamp regressed within the configured live clock epoch.
    ClockRegression {
        previous_ns: u64,
        actual_ns: u64,
        obligation: SupervisorAction,
    },
    /// Authority was requested outside `ReadyStopped` or `Active`.
    AuthorityUnavailable {
        state: SupervisorStateKind,
        obligation: SupervisorAction,
    },
    /// Every nonzero process-lifetime lease ID has been consumed.
    LeaseIdExhausted { obligation: SupervisorAction },
}

impl AgentAuthorityError {
    /// Return an action the runtime still owes after this failure.
    ///
    /// The action remains an obligation, not evidence that it ran.
    pub const fn obligation(&self) -> SupervisorAction {
        match self {
            Self::Supervisor { obligation, .. }
            | Self::ZeroEvidence { obligation, .. }
            | Self::ZeroReceiptNotNew { obligation, .. }
            | Self::ReadinessEpochNotIncreasing { obligation, .. }
            | Self::TimeBeforeClockEpoch { obligation, .. }
            | Self::ClockRegression { obligation, .. }
            | Self::AuthorityUnavailable { obligation, .. }
            | Self::LeaseIdExhausted { obligation } => *obligation,
        }
    }
}

impl fmt::Display for AgentAuthorityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "agent authority transition failed: {self:?}")
    }
}

impl std::error::Error for AgentAuthorityError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Supervisor { source, .. } => Some(source),
            Self::ZeroEvidence { source, .. } => Some(source),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use kiko_supervisor_core::{Sha256Digest, StopReason, SupervisorConfig};
    use robot_protocol::ControllerUptimeMsWrapping;
    use robot_protocol::v2::{
        ControlEpoch, ControllerBootId, ControllerDeadlineMsWrapping, ControllerFaults,
        ControllerUid, HostCommandResultCode, OutputState, RemainingLeaseMs, TimerPwm,
    };

    use super::*;

    const ORIGIN_NS: u64 = 100;

    fn at(nanos: u64) -> HostMonotonicTimestamp {
        HostMonotonicTimestamp::from_nanos(nanos)
    }

    fn duration(nanos: u64) -> AuthorityDuration {
        AuthorityDuration::try_from_nanos(nanos).expect("nonzero fixture duration")
    }

    fn config() -> SupervisorConfig {
        SupervisorConfig::new(duration(1_000), duration(100)).expect("valid fixture limits")
    }

    fn uid() -> ControllerUid {
        ControllerUid::try_new([1; 12]).expect("nonzero controller UID")
    }

    fn boot() -> ControllerBootId {
        ControllerBootId::try_new(7).expect("nonzero controller boot ID")
    }

    fn readiness(epoch: u64, control_epoch: u32) -> ReadinessBinding {
        ReadinessBinding::new(
            ReadinessEpoch::try_new(epoch).expect("nonzero readiness epoch"),
            uid(),
            boot(),
            ControlEpoch::try_new(control_epoch).expect("nonzero control epoch"),
            Sha256Digest::try_new([2; 32]).expect("nonzero hardware hash"),
            Sha256Digest::try_new([3; 32]).expect("nonzero calibration hash"),
        )
    }

    fn host_zero(sequence: u32, control_epoch: u32) -> HostCommandResult {
        HostCommandResult {
            controller_uid: uid(),
            boot_id: boot(),
            control_epoch: ControlEpoch::try_new(control_epoch).expect("nonzero control epoch"),
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

    fn booting() -> AgentAuthoritySupervisor {
        AgentAuthoritySupervisor::new(config(), NavigationClockEpoch::new(at(ORIGIN_NS)))
    }

    fn ready() -> AgentAuthoritySupervisor {
        let mut adapter = booting();
        assert_eq!(
            adapter.begin_inventory(at(101)),
            Ok(SupervisorAction::InventoryRequired)
        );
        assert_eq!(
            adapter.admit_readiness(readiness(1, 9), at(102)),
            Ok(SupervisorAction::Disarmed)
        );
        assert_eq!(
            adapter.arm(at(103)),
            Ok(SupervisorAction::BaseZeroRequired {
                reason: StopReason::Arming
            })
        );
        assert_eq!(
            adapter.admit_applied_zero(host_zero(0, 9), at(104), at(104)),
            Ok(SupervisorAction::ReadyStopped)
        );
        adapter
    }

    #[test]
    fn boot_inventory_readiness_arm_and_fresh_zero_are_strict_gates() {
        let mut adapter = booting();
        let denied = adapter
            .request_mode(AuthorityMode::Manual, duration(50), at(101))
            .expect_err("booting cannot grant authority");
        assert!(matches!(
            denied,
            AgentAuthorityError::AuthorityUnavailable {
                state: SupervisorStateKind::Booting,
                ..
            }
        ));
        adapter.begin_inventory(at(102)).expect("begin inventory");
        adapter
            .admit_readiness(readiness(1, 9), at(103))
            .expect("admit readiness");
        adapter.arm(at(104)).expect("arm to zero gate");

        let stale = adapter
            .admit_applied_zero(host_zero(0, 9), at(104), at(105))
            .expect_err("receipt must follow stop request");
        assert!(matches!(
            stale,
            AgentAuthorityError::Supervisor {
                source: SupervisorError::ZeroEvidencePredatesStopRequest { .. },
                obligation: SupervisorAction::BaseZeroRequired {
                    reason: StopReason::Arming
                }
            }
        ));
        assert_eq!(
            adapter.admit_applied_zero(host_zero(0, 9), at(106), at(106)),
            Ok(SupervisorAction::ReadyStopped)
        );
    }

    #[test]
    fn only_exact_new_host_zero_results_cross_the_boundary() {
        let mut adapter = booting();
        adapter.begin_inventory(at(101)).expect("inventory");
        adapter
            .admit_readiness(readiness(1, 9), at(102))
            .expect("readiness");
        let disarmed = adapter.control_status(AgentMapStateV1::UNAVAILABLE);
        assert_eq!(disarmed.runtime(), AgentRuntimeStateV1::Disarmed);
        assert_eq!(disarmed.base_command(), AgentBaseCommandStateV1::Unknown);
        adapter.arm(at(103)).expect("arm");

        let mut cached = host_zero(0, 9);
        cached.result = HostCommandResultCode::DuplicateCached;
        let cached = adapter
            .admit_applied_zero(cached, at(104), at(104))
            .expect_err("cached result is not fresh application evidence");
        assert!(matches!(
            cached,
            AgentAuthorityError::ZeroEvidence {
                source: ZeroEvidenceError::HostResultDoesNotProveFreshApplication { .. },
                ..
            }
        ));

        let mut nonzero = host_zero(0, 9);
        nonzero.requested_timer_pwm = TimerPwm::try_new(1, 0).expect("valid nonzero PWM");
        let nonzero = adapter
            .admit_applied_zero(nonzero, at(105), at(105))
            .expect_err("nonzero request cannot prove deliberate zero");
        assert!(matches!(
            nonzero,
            AgentAuthorityError::ZeroEvidence {
                source: ZeroEvidenceError::RequestedNonzeroPwm { .. },
                ..
            }
        ));
        adapter
            .admit_applied_zero(host_zero(0, 9), at(106), at(106))
            .expect("first exact zero");
        let lease = match adapter
            .request_mode(AuthorityMode::Manual, duration(50), at(107))
            .expect("grant manual")
        {
            SupervisorAction::AuthorityGranted { lease } => lease,
            action => panic!("unexpected action: {action:?}"),
        };
        adapter
            .release_authority(lease.id(), at(108))
            .expect("release enters new zero gate");

        let replay = adapter
            .admit_applied_zero(host_zero(0, 9), at(109), at(109))
            .expect_err("same applied result cannot be replayed as a new zero");
        assert!(matches!(
            replay,
            AgentAuthorityError::ZeroReceiptNotNew {
                previous_sequence,
                actual_sequence,
                obligation: SupervisorAction::BaseZeroRequired {
                    reason: StopReason::AuthorityReleased
                }
            } if previous_sequence.get() == 0 && actual_sequence.get() == 0
        ));
        adapter
            .admit_applied_zero(host_zero(1, 9), at(110), at(110))
            .expect("strictly newer zero result");
    }

    #[test]
    fn same_mode_renews_and_cross_mode_handover_requires_new_zero() {
        let mut adapter = ready();
        let first = match adapter
            .request_mode(AuthorityMode::Manual, duration(50), at(105))
            .expect("grant manual")
        {
            SupervisorAction::AuthorityGranted { lease } => lease,
            action => panic!("unexpected action: {action:?}"),
        };
        let renewed = match adapter
            .request_mode(AuthorityMode::Manual, duration(60), at(106))
            .expect("renew same mode")
        {
            SupervisorAction::AuthorityRenewed { lease } => lease,
            action => panic!("unexpected action: {action:?}"),
        };
        assert_eq!(renewed.id(), first.id());
        assert_eq!(renewed.expires_at_exclusive().as_nanos(), 166);
        assert!(matches!(
            adapter.manual_authority_snapshot(),
            ManualAuthoritySnapshot::ActiveManual {
                lease_id,
                expires_at_exclusive
            } if lease_id == first.id() && expires_at_exclusive == at(166)
        ));

        assert_eq!(
            adapter.request_mode(AuthorityMode::Explore, duration(70), at(107)),
            Ok(SupervisorAction::BaseZeroRequired {
                reason: StopReason::AuthorityHandover
            })
        );
        assert_eq!(
            adapter.manual_authority_snapshot(),
            ManualAuthoritySnapshot::NotActiveManual
        );
        let granted = match adapter
            .admit_applied_zero(host_zero(1, 9), at(108), at(108))
            .expect("new applied zero completes handover")
        {
            SupervisorAction::AuthorityGranted { lease } => lease,
            action => panic!("unexpected action: {action:?}"),
        };
        assert_ne!(granted.id(), first.id());
        assert_eq!(granted.mode(), AuthorityMode::Explore);
    }

    #[test]
    fn release_and_expiry_each_reenter_a_new_zero_gate() {
        let mut adapter = ready();
        let manual = match adapter
            .request_mode(AuthorityMode::Manual, duration(10), at(105))
            .expect("grant manual")
        {
            SupervisorAction::AuthorityGranted { lease } => lease,
            action => panic!("unexpected action: {action:?}"),
        };
        assert_eq!(
            adapter.release_authority(manual.id(), at(106)),
            Ok(SupervisorAction::BaseZeroRequired {
                reason: StopReason::AuthorityReleased
            })
        );
        assert!(matches!(
            adapter.request_mode(AuthorityMode::PointGoal, duration(10), at(107)),
            Err(AgentAuthorityError::AuthorityUnavailable {
                state: SupervisorStateKind::AwaitingZero,
                ..
            })
        ));
        adapter
            .admit_applied_zero(host_zero(1, 9), at(108), at(108))
            .expect("release zero");
        adapter
            .request_mode(AuthorityMode::PointGoal, duration(10), at(109))
            .expect("point authority");
        assert_eq!(adapter.tick(at(118)), Ok(SupervisorAction::None));
        assert_eq!(
            adapter.tick(at(119)),
            Ok(SupervisorAction::BaseZeroRequired {
                reason: StopReason::AuthorityLeaseExpired
            })
        );
        assert!(matches!(
            adapter.state(),
            SupervisorState::AwaitingZero {
                reason: StopReason::AuthorityLeaseExpired,
                ..
            }
        ));
        adapter
            .admit_applied_zero(host_zero(2, 9), at(120), at(120))
            .expect("expiry zero");
    }

    #[test]
    fn all_motion_modes_remain_mutually_exclusive_across_many_transitions() {
        let modes = [
            AuthorityMode::Commissioning,
            AuthorityMode::Manual,
            AuthorityMode::PointGoal,
            AuthorityMode::Explore,
        ];
        let mut adapter = ready();
        let mut now = 105_u64;
        let mut zero_sequence = 1_u32;
        let mut previous_id = None;

        for index in 0..128_usize {
            let mode = modes[index % modes.len()];
            let action = adapter
                .request_mode(mode, duration(500), at(now))
                .expect("deterministic mode request");
            now += 1;
            let lease = match action {
                SupervisorAction::AuthorityGranted { lease }
                | SupervisorAction::AuthorityRenewed { lease } => lease,
                SupervisorAction::BaseZeroRequired {
                    reason: StopReason::AuthorityHandover,
                } => {
                    let action = adapter
                        .admit_applied_zero(host_zero(zero_sequence, 9), at(now), at(now))
                        .expect("fresh handover zero");
                    zero_sequence += 1;
                    now += 1;
                    match action {
                        SupervisorAction::AuthorityGranted { lease } => lease,
                        action => panic!("unexpected post-zero action: {action:?}"),
                    }
                }
                action => panic!("unexpected authority action: {action:?}"),
            };
            assert_eq!(lease.mode(), mode);
            if let Some(previous) = previous_id
                && modes[(index - 1) % modes.len()] != mode
            {
                assert_ne!(lease.id(), previous);
            }
            previous_id = Some(lease.id());
            assert!(matches!(
                adapter.state(),
                SupervisorState::Active { lease: active, .. }
                    if active.id() == lease.id() && active.mode() == mode
            ));
        }
    }

    #[test]
    fn lease_id_exhaustion_latches_a_stop_required_fault() {
        let mut adapter = ready();
        adapter.set_next_lease_id_for_test(Some(u64::MAX));
        let lease = match adapter
            .request_mode(AuthorityMode::Manual, duration(50), at(105))
            .expect("last nonzero lease ID remains valid")
        {
            SupervisorAction::AuthorityGranted { lease } => lease,
            action => panic!("unexpected action: {action:?}"),
        };
        assert_eq!(lease.id().get(), u64::MAX);
        adapter
            .release_authority(lease.id(), at(106))
            .expect("release last ID");
        adapter
            .admit_applied_zero(host_zero(1, 9), at(107), at(107))
            .expect("post-release zero");

        let exhausted = adapter
            .request_mode(AuthorityMode::Explore, duration(50), at(108))
            .expect_err("no lease ID may wrap or be reused");
        assert_eq!(
            exhausted,
            AgentAuthorityError::LeaseIdExhausted {
                obligation: SupervisorAction::FaultStopRequired {
                    fault: FaultKind::InternalInvariant
                }
            }
        );
        assert_eq!(
            adapter.state(),
            SupervisorState::FaultLatched {
                fault: FaultKind::InternalInvariant
            }
        );
    }

    #[test]
    fn clock_regression_latches_fault_and_removes_manual_authority() {
        let mut adapter = ready();
        adapter
            .request_mode(AuthorityMode::Manual, duration(50), at(105))
            .expect("manual authority");
        let error = adapter.tick(at(104)).expect_err("clock must not regress");
        assert_eq!(
            error,
            AgentAuthorityError::ClockRegression {
                previous_ns: 105,
                actual_ns: 104,
                obligation: SupervisorAction::FaultStopRequired {
                    fault: FaultKind::ClockRegression
                }
            }
        );
        assert_eq!(
            adapter.manual_authority_snapshot(),
            ManualAuthoritySnapshot::NotActiveManual
        );
    }

    #[test]
    fn readiness_epoch_must_increase_after_reinventory() {
        let mut adapter = booting();
        adapter.begin_inventory(at(101)).expect("inventory");
        adapter
            .admit_readiness(readiness(7, 9), at(102))
            .expect("first readiness");
        adapter
            .latch_fault(FaultKind::HardwareReadinessLost, at(103))
            .expect("latch readiness fault");
        adapter
            .clear_fault_for_inventory(at(104))
            .expect("return to inventory");
        let replay = adapter
            .admit_readiness(readiness(7, 10), at(105))
            .expect_err("readiness epoch cannot be reused");
        assert!(matches!(
            replay,
            AgentAuthorityError::ReadinessEpochNotIncreasing {
                previous,
                actual,
                obligation: SupervisorAction::InventoryRequired,
            } if previous.get() == 7 && actual.get() == 7
        ));
        adapter
            .admit_readiness(readiness(8, 10), at(106))
            .expect("new readiness epoch");
    }

    #[test]
    fn control_status_preserves_lifecycle_and_never_invents_base_evidence() {
        let mut adapter = booting();
        let boot = adapter.control_status(AgentMapStateV1::UNAVAILABLE);
        assert_eq!(boot.runtime(), AgentRuntimeStateV1::Booting);
        assert_eq!(boot.base_command(), AgentBaseCommandStateV1::Unknown);

        adapter.begin_inventory(at(101)).expect("inventory");
        adapter
            .admit_readiness(readiness(1, 9), at(102))
            .expect("readiness");
        adapter.arm(at(103)).expect("arm");
        let awaiting = adapter.control_status(AgentMapStateV1::UNAVAILABLE);
        assert_eq!(awaiting.runtime(), AgentRuntimeStateV1::AwaitingZero);
        assert_eq!(awaiting.base_command(), AgentBaseCommandStateV1::Unknown);

        adapter
            .admit_applied_zero(host_zero(0, 9), at(104), at(104))
            .expect("ready zero");
        let ready = adapter.control_status(AgentMapStateV1::UNAVAILABLE);
        assert_eq!(
            ready.base_command(),
            AgentBaseCommandStateV1::ConfirmedStopped
        );
        adapter
            .request_mode(AuthorityMode::Commissioning, duration(50), at(105))
            .expect("commissioning authority");
        let commissioning = adapter.control_status(AgentMapStateV1::UNAVAILABLE);
        assert_eq!(
            commissioning.runtime(),
            AgentRuntimeStateV1::Active {
                mode: AgentOperatingModeV1::Commissioning
            }
        );
        assert_eq!(
            commissioning.base_command(),
            AgentBaseCommandStateV1::Unknown
        );
    }
}
