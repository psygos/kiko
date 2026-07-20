use core::fmt;
use core::num::NonZeroU64;

use crate::{
    AuthorityDuration, ConfirmedBaseZero, MonotonicInstant, ReadinessBinding, ReadinessEpoch,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AuthorityLeaseId(NonZeroU64);

impl AuthorityLeaseId {
    pub fn try_new(value: u64) -> Result<Self, SupervisorError> {
        NonZeroU64::new(value)
            .map(Self)
            .ok_or(SupervisorError::ZeroAuthorityLeaseId)
    }

    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AuthorityMode {
    Commissioning,
    Manual,
    PointGoal,
    Explore,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AuthorityLease {
    id: AuthorityLeaseId,
    mode: AuthorityMode,
    readiness_epoch: ReadinessEpoch,
    issued_at: MonotonicInstant,
    expires_at_exclusive: MonotonicInstant,
}

impl AuthorityLease {
    pub const fn id(self) -> AuthorityLeaseId {
        self.id
    }

    pub const fn mode(self) -> AuthorityMode {
        self.mode
    }

    pub const fn readiness_epoch(self) -> ReadinessEpoch {
        self.readiness_epoch
    }

    pub const fn issued_at(self) -> MonotonicInstant {
        self.issued_at
    }

    pub const fn expires_at_exclusive(self) -> MonotonicInstant {
        self.expires_at_exclusive
    }

    pub const fn is_alive_at(self, now: MonotonicInstant) -> bool {
        now.as_nanos() < self.expires_at_exclusive.as_nanos()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    Arming,
    AuthorityHandover,
    AuthorityReleased,
    AuthorityLeaseExpired,
    ExplicitDisarm,
    ZeroEvidenceStale,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaultKind {
    EmergencyStop,
    HardwareReadinessLost,
    ControllerIdentityChanged,
    ClockRegression,
    SafetyJournalFailed,
    InternalInvariant,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AfterZero {
    Disarmed,
    Ready,
    Activate(AuthorityLease),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SupervisorState {
    Booting,
    Inventory,
    Disarmed {
        readiness: ReadinessBinding,
    },
    AwaitingZero {
        readiness: ReadinessBinding,
        reason: StopReason,
        zero_required_after: MonotonicInstant,
    },
    ReadyStopped {
        readiness: ReadinessBinding,
        zero: ConfirmedBaseZero,
    },
    Active {
        readiness: ReadinessBinding,
        lease: AuthorityLease,
    },
    FaultLatched {
        fault: FaultKind,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SupervisorStateKind {
    Booting,
    Inventory,
    Disarmed,
    AwaitingZero,
    ReadyStopped,
    Active,
    FaultLatched,
}

impl SupervisorState {
    pub const fn kind(self) -> SupervisorStateKind {
        match self {
            Self::Booting => SupervisorStateKind::Booting,
            Self::Inventory => SupervisorStateKind::Inventory,
            Self::Disarmed { .. } => SupervisorStateKind::Disarmed,
            Self::AwaitingZero { .. } => SupervisorStateKind::AwaitingZero,
            Self::ReadyStopped { .. } => SupervisorStateKind::ReadyStopped,
            Self::Active { .. } => SupervisorStateKind::Active,
            Self::FaultLatched { .. } => SupervisorStateKind::FaultLatched,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SupervisorAction {
    None,
    InventoryRequired,
    BaseZeroRequired { reason: StopReason },
    ReadyStopped,
    AuthorityGranted { lease: AuthorityLease },
    AuthorityRenewed { lease: AuthorityLease },
    PendingAuthorityExpired,
    Disarmed,
    FaultStopRequired { fault: FaultKind },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SupervisorConfig {
    maximum_authority_lease: AuthorityDuration,
    maximum_zero_age: AuthorityDuration,
}

impl SupervisorConfig {
    pub const fn new(
        maximum_authority_lease: AuthorityDuration,
        maximum_zero_age: AuthorityDuration,
    ) -> Result<Self, SupervisorConfigError> {
        if maximum_zero_age.as_nanos() > maximum_authority_lease.as_nanos() {
            return Err(SupervisorConfigError::ZeroAgeExceedsAuthorityLease {
                maximum_zero_age_ns: maximum_zero_age.as_nanos(),
                maximum_authority_lease_ns: maximum_authority_lease.as_nanos(),
            });
        }
        Ok(Self {
            maximum_authority_lease,
            maximum_zero_age,
        })
    }

    pub const fn maximum_authority_lease(self) -> AuthorityDuration {
        self.maximum_authority_lease
    }

    pub const fn maximum_zero_age(self) -> AuthorityDuration {
        self.maximum_zero_age
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SupervisorConfigError {
    ZeroAgeExceedsAuthorityLease {
        maximum_zero_age_ns: u64,
        maximum_authority_lease_ns: u64,
    },
}

impl fmt::Display for SupervisorConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid supervisor configuration: {self:?}")
    }
}

impl core::error::Error for SupervisorConfigError {}

pub struct RobotSupervisor {
    config: SupervisorConfig,
    state: SupervisorState,
    after_zero: Option<AfterZero>,
    last_observed_time: MonotonicInstant,
}

impl RobotSupervisor {
    pub const fn new(config: SupervisorConfig) -> Self {
        Self {
            config,
            state: SupervisorState::Booting,
            after_zero: None,
            last_observed_time: MonotonicInstant::ZERO,
        }
    }

    pub const fn state(&self) -> SupervisorState {
        self.state
    }

    pub fn begin_inventory(
        &mut self,
        now: MonotonicInstant,
    ) -> Result<SupervisorAction, SupervisorError> {
        self.require_time(now)?;
        if !matches!(self.state, SupervisorState::Booting) {
            return Err(self.invalid_transition("begin inventory"));
        }
        self.commit_time(now);
        self.state = SupervisorState::Inventory;
        Ok(SupervisorAction::InventoryRequired)
    }

    pub fn admit_readiness(
        &mut self,
        readiness: ReadinessBinding,
        now: MonotonicInstant,
    ) -> Result<SupervisorAction, SupervisorError> {
        self.require_time(now)?;
        if !matches!(self.state, SupervisorState::Inventory) {
            return Err(self.invalid_transition("admit readiness"));
        }
        self.commit_time(now);
        self.state = SupervisorState::Disarmed { readiness };
        Ok(SupervisorAction::Disarmed)
    }

    pub fn arm(&mut self, now: MonotonicInstant) -> Result<SupervisorAction, SupervisorError> {
        self.require_time(now)?;
        let SupervisorState::Disarmed { readiness } = self.state else {
            return Err(self.invalid_transition("arm"));
        };
        self.commit_time(now);
        Ok(self.await_zero(readiness, StopReason::Arming, AfterZero::Ready, now))
    }

    pub fn request_authority(
        &mut self,
        id: AuthorityLeaseId,
        mode: AuthorityMode,
        duration: AuthorityDuration,
        now: MonotonicInstant,
    ) -> Result<SupervisorAction, SupervisorError> {
        self.require_time(now)?;
        let SupervisorState::ReadyStopped { readiness, zero } = self.state else {
            return Err(self.invalid_transition("request authority"));
        };
        let lease = self.build_lease(id, mode, readiness.epoch(), duration, now)?;
        let zero_is_fresh = self.zero_is_fresh_and_bound(zero, readiness, now)?;
        self.commit_time(now);
        if !zero_is_fresh {
            return Ok(self.await_zero(
                readiness,
                StopReason::ZeroEvidenceStale,
                AfterZero::Activate(lease),
                now,
            ));
        }
        self.state = SupervisorState::Active { readiness, lease };
        Ok(SupervisorAction::AuthorityGranted { lease })
    }

    pub fn handover(
        &mut self,
        id: AuthorityLeaseId,
        mode: AuthorityMode,
        duration: AuthorityDuration,
        now: MonotonicInstant,
    ) -> Result<SupervisorAction, SupervisorError> {
        self.require_time(now)?;
        let SupervisorState::Active { readiness, .. } = self.state else {
            return Err(self.invalid_transition("handover authority"));
        };
        let lease = self.build_lease(id, mode, readiness.epoch(), duration, now)?;
        self.commit_time(now);
        Ok(self.await_zero(
            readiness,
            StopReason::AuthorityHandover,
            AfterZero::Activate(lease),
            now,
        ))
    }

    pub fn renew_authority(
        &mut self,
        id: AuthorityLeaseId,
        duration: AuthorityDuration,
        now: MonotonicInstant,
    ) -> Result<SupervisorAction, SupervisorError> {
        self.require_time(now)?;
        let SupervisorState::Active { readiness, lease } = self.state else {
            return Err(self.invalid_transition("renew authority"));
        };
        if lease.id != id {
            return Err(SupervisorError::AuthorityLeaseMismatch {
                expected: lease.id,
                actual: id,
            });
        }
        if !lease.is_alive_at(now) {
            self.commit_time(now);
            return Ok(self.await_zero(
                readiness,
                StopReason::AuthorityLeaseExpired,
                AfterZero::Ready,
                now,
            ));
        }
        let renewed = self.build_lease(id, lease.mode, readiness.epoch(), duration, now)?;
        self.commit_time(now);
        self.state = SupervisorState::Active {
            readiness,
            lease: renewed,
        };
        Ok(SupervisorAction::AuthorityRenewed { lease: renewed })
    }

    pub fn release_authority(
        &mut self,
        id: AuthorityLeaseId,
        now: MonotonicInstant,
    ) -> Result<SupervisorAction, SupervisorError> {
        self.require_time(now)?;
        let SupervisorState::Active { readiness, lease } = self.state else {
            return Err(self.invalid_transition("release authority"));
        };
        if lease.id != id {
            return Err(SupervisorError::AuthorityLeaseMismatch {
                expected: lease.id,
                actual: id,
            });
        }
        self.commit_time(now);
        Ok(self.await_zero(
            readiness,
            StopReason::AuthorityReleased,
            AfterZero::Ready,
            now,
        ))
    }

    pub fn disarm(&mut self, now: MonotonicInstant) -> Result<SupervisorAction, SupervisorError> {
        self.require_time(now)?;
        match self.state {
            SupervisorState::Disarmed { .. } => {
                self.commit_time(now);
                Ok(SupervisorAction::Disarmed)
            }
            SupervisorState::ReadyStopped { readiness, .. }
            | SupervisorState::Active { readiness, .. } => {
                self.commit_time(now);
                Ok(self.await_zero(
                    readiness,
                    StopReason::ExplicitDisarm,
                    AfterZero::Disarmed,
                    now,
                ))
            }
            _ => Err(self.invalid_transition("disarm")),
        }
    }

    pub fn admit_confirmed_zero(
        &mut self,
        zero: ConfirmedBaseZero,
        now: MonotonicInstant,
    ) -> Result<SupervisorAction, SupervisorError> {
        self.require_time(now)?;
        let SupervisorState::AwaitingZero {
            readiness,
            zero_required_after,
            ..
        } = self.state
        else {
            return Err(self.invalid_transition("admit confirmed zero"));
        };
        if zero.observed_at() <= zero_required_after {
            return Err(SupervisorError::ZeroEvidencePredatesStopRequest {
                observed_at_ns: zero.observed_at().as_nanos(),
                required_after_ns: zero_required_after.as_nanos(),
            });
        }
        if !self.zero_is_fresh_and_bound(zero, readiness, now)? {
            return Err(SupervisorError::ZeroEvidenceStale);
        }
        let after_zero = self
            .after_zero
            .ok_or(SupervisorError::MissingZeroContinuation)?;
        self.commit_time(now);
        self.after_zero = None;
        match after_zero {
            AfterZero::Disarmed => {
                self.state = SupervisorState::Disarmed { readiness };
                Ok(SupervisorAction::Disarmed)
            }
            AfterZero::Ready => {
                self.state = SupervisorState::ReadyStopped { readiness, zero };
                Ok(SupervisorAction::ReadyStopped)
            }
            AfterZero::Activate(lease) if lease.is_alive_at(now) => {
                self.state = SupervisorState::Active { readiness, lease };
                Ok(SupervisorAction::AuthorityGranted { lease })
            }
            AfterZero::Activate(_) => {
                self.state = SupervisorState::ReadyStopped { readiness, zero };
                Ok(SupervisorAction::PendingAuthorityExpired)
            }
        }
    }

    pub fn tick(&mut self, now: MonotonicInstant) -> Result<SupervisorAction, SupervisorError> {
        self.require_time(now)?;
        let action = match self.state {
            SupervisorState::Active { readiness, lease } if !lease.is_alive_at(now) => self
                .await_zero(
                    readiness,
                    StopReason::AuthorityLeaseExpired,
                    AfterZero::Ready,
                    now,
                ),
            _ => SupervisorAction::None,
        };
        self.commit_time(now);
        Ok(action)
    }

    pub fn latch_fault(
        &mut self,
        fault: FaultKind,
        now: MonotonicInstant,
    ) -> Result<SupervisorAction, SupervisorError> {
        self.require_time(now)?;
        self.commit_time(now);
        self.after_zero = None;
        self.state = SupervisorState::FaultLatched { fault };
        Ok(SupervisorAction::FaultStopRequired { fault })
    }

    pub fn clear_fault_for_inventory(
        &mut self,
        now: MonotonicInstant,
    ) -> Result<SupervisorAction, SupervisorError> {
        self.require_time(now)?;
        if !matches!(self.state, SupervisorState::FaultLatched { .. }) {
            return Err(self.invalid_transition("clear fault"));
        }
        self.commit_time(now);
        self.after_zero = None;
        self.state = SupervisorState::Inventory;
        Ok(SupervisorAction::InventoryRequired)
    }

    fn build_lease(
        &self,
        id: AuthorityLeaseId,
        mode: AuthorityMode,
        readiness_epoch: ReadinessEpoch,
        duration: AuthorityDuration,
        now: MonotonicInstant,
    ) -> Result<AuthorityLease, SupervisorError> {
        if duration.as_nanos() > self.config.maximum_authority_lease.as_nanos() {
            return Err(SupervisorError::AuthorityLeaseTooLong {
                actual_ns: duration.as_nanos(),
                maximum_ns: self.config.maximum_authority_lease.as_nanos(),
            });
        }
        let expires_at_exclusive =
            now.checked_add(duration)
                .ok_or(SupervisorError::AuthorityDeadlineOverflow {
                    issued_at_ns: now.as_nanos(),
                    duration_ns: duration.as_nanos(),
                })?;
        Ok(AuthorityLease {
            id,
            mode,
            readiness_epoch,
            issued_at: now,
            expires_at_exclusive,
        })
    }

    fn zero_is_fresh_and_bound(
        &self,
        zero: ConfirmedBaseZero,
        readiness: ReadinessBinding,
        now: MonotonicInstant,
    ) -> Result<bool, SupervisorError> {
        if zero.controller_uid() != readiness.controller_uid()
            || zero.controller_boot_id() != readiness.controller_boot_id()
        {
            return Err(SupervisorError::ZeroControllerIdentityMismatch);
        }
        if zero.control_epoch() != readiness.control_epoch() {
            return Err(SupervisorError::ZeroControlEpochMismatch {
                expected: readiness.control_epoch(),
                actual: zero.control_epoch(),
            });
        }
        let Some(age_ns) = now.checked_elapsed_since(zero.observed_at()) else {
            return Err(SupervisorError::ZeroObservedInFuture {
                observed_at_ns: zero.observed_at().as_nanos(),
                now_ns: now.as_nanos(),
            });
        };
        Ok(age_ns < self.config.maximum_zero_age.as_nanos())
    }

    fn await_zero(
        &mut self,
        readiness: ReadinessBinding,
        reason: StopReason,
        after_zero: AfterZero,
        zero_required_after: MonotonicInstant,
    ) -> SupervisorAction {
        self.after_zero = Some(after_zero);
        self.state = SupervisorState::AwaitingZero {
            readiness,
            reason,
            zero_required_after,
        };
        SupervisorAction::BaseZeroRequired { reason }
    }

    fn require_time(&mut self, now: MonotonicInstant) -> Result<(), SupervisorError> {
        if now < self.last_observed_time {
            let previous_ns = self.last_observed_time.as_nanos();
            let actual_ns = now.as_nanos();
            self.after_zero = None;
            self.state = SupervisorState::FaultLatched {
                fault: FaultKind::ClockRegression,
            };
            return Err(SupervisorError::ClockRegression {
                previous_ns,
                actual_ns,
            });
        }
        Ok(())
    }

    fn commit_time(&mut self, now: MonotonicInstant) {
        self.last_observed_time = now;
    }

    fn invalid_transition(&self, operation: &'static str) -> SupervisorError {
        SupervisorError::InvalidTransition {
            operation,
            state: self.state.kind(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SupervisorError {
    ZeroAuthorityLeaseId,
    InvalidTransition {
        operation: &'static str,
        state: SupervisorStateKind,
    },
    ClockRegression {
        previous_ns: u64,
        actual_ns: u64,
    },
    AuthorityLeaseTooLong {
        actual_ns: u64,
        maximum_ns: u64,
    },
    AuthorityDeadlineOverflow {
        issued_at_ns: u64,
        duration_ns: u64,
    },
    AuthorityLeaseMismatch {
        expected: AuthorityLeaseId,
        actual: AuthorityLeaseId,
    },
    ZeroControllerIdentityMismatch,
    ZeroControlEpochMismatch {
        expected: robot_protocol::v2::ControlEpoch,
        actual: robot_protocol::v2::ControlEpoch,
    },
    ZeroEvidencePredatesStopRequest {
        observed_at_ns: u64,
        required_after_ns: u64,
    },
    ZeroObservedInFuture {
        observed_at_ns: u64,
        now_ns: u64,
    },
    ZeroEvidenceStale,
    MissingZeroContinuation,
}

impl fmt::Display for SupervisorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "robot supervisor transition failed: {self:?}")
    }
}

impl core::error::Error for SupervisorError {}

#[cfg(test)]
mod tests {
    extern crate std;

    use robot_protocol::ControllerUptimeMsWrapping;
    use robot_protocol::v2::{
        AppliedResult, AppliedResultCode, ControlEpoch, ControllerBootId,
        ControllerDeadlineMsWrapping, ControllerFaults, ControllerUid, HostCommandResult,
        HostCommandResultCode, OutputState, RemainingLeaseMs, TimerPwm, V2CommandSequence,
    };

    use super::*;
    use crate::{Sha256Digest, ZeroEvidenceError};

    fn at(nanos: u64) -> MonotonicInstant {
        MonotonicInstant::from_nanos_since_process_start(nanos)
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

    fn readiness(epoch: u64) -> ReadinessBinding {
        ReadinessBinding::new(
            ReadinessEpoch::try_new(epoch).unwrap(),
            uid(),
            boot(),
            ControlEpoch::try_new(9).unwrap(),
            Sha256Digest::try_new([2; 32]).unwrap(),
            Sha256Digest::try_new([3; 32]).unwrap(),
        )
    }

    fn zero(observed_at: u64) -> ConfirmedBaseZero {
        zero_in_epoch(observed_at, 9)
    }

    fn zero_in_epoch(observed_at: u64, control_epoch: u32) -> ConfirmedBaseZero {
        ConfirmedBaseZero::try_from_applied_result(
            AppliedResult {
                controller_uid: uid(),
                boot_id: boot(),
                control_epoch: ControlEpoch::try_new(control_epoch).unwrap(),
                sequence: V2CommandSequence::FIRST,
                result: AppliedResultCode::AppliedNew,
                timer_pwm: TimerPwm::ZERO,
                output_state: OutputState::ZeroPwm,
                applied_at: ControllerUptimeMsWrapping::new(10),
                expires_at: ControllerDeadlineMsWrapping::new(20),
                faults: ControllerFaults::NONE,
            },
            at(observed_at),
        )
        .unwrap()
    }

    fn supervisor() -> RobotSupervisor {
        RobotSupervisor::new(SupervisorConfig::new(duration(100), duration(20)).unwrap())
    }

    fn ready() -> RobotSupervisor {
        let mut supervisor = supervisor();
        supervisor.begin_inventory(at(1)).unwrap();
        supervisor.admit_readiness(readiness(1), at(2)).unwrap();
        supervisor.arm(at(3)).unwrap();
        supervisor.admit_confirmed_zero(zero(4), at(4)).unwrap();
        supervisor
    }

    #[test]
    fn boot_inventory_arm_and_zero_are_distinct_gates() {
        let mut supervisor = supervisor();
        assert_eq!(supervisor.state(), SupervisorState::Booting);
        assert_eq!(
            supervisor.begin_inventory(at(1)),
            Ok(SupervisorAction::InventoryRequired)
        );
        assert_eq!(
            supervisor.admit_readiness(readiness(1), at(2)),
            Ok(SupervisorAction::Disarmed)
        );
        assert!(
            supervisor
                .request_authority(
                    AuthorityLeaseId::try_new(1).unwrap(),
                    AuthorityMode::Manual,
                    duration(10),
                    at(3)
                )
                .is_err()
        );
        assert_eq!(
            supervisor.arm(at(3)),
            Ok(SupervisorAction::BaseZeroRequired {
                reason: StopReason::Arming
            })
        );
        assert_eq!(
            supervisor.admit_confirmed_zero(zero(4), at(4)),
            Ok(SupervisorAction::ReadyStopped)
        );
    }

    #[test]
    fn authority_is_exclusive_and_handover_requires_new_zero() {
        let mut supervisor = ready();
        let manual = AuthorityLeaseId::try_new(1).unwrap();
        let explore = AuthorityLeaseId::try_new(2).unwrap();
        assert!(matches!(
            supervisor.request_authority(manual, AuthorityMode::Manual, duration(50), at(5)),
            Ok(SupervisorAction::AuthorityGranted { lease }) if lease.mode() == AuthorityMode::Manual
        ));
        assert!(
            supervisor
                .request_authority(explore, AuthorityMode::Explore, duration(50), at(6))
                .is_err()
        );
        assert_eq!(
            supervisor.handover(explore, AuthorityMode::Explore, duration(50), at(6)),
            Ok(SupervisorAction::BaseZeroRequired {
                reason: StopReason::AuthorityHandover
            })
        );
        assert!(matches!(
            supervisor.admit_confirmed_zero(zero(7), at(7)),
            Ok(SupervisorAction::AuthorityGranted { lease }) if lease.id() == explore
        ));
    }

    #[test]
    fn lease_deadline_is_exclusive_and_requires_confirmed_stop() {
        let mut supervisor = ready();
        let lease = AuthorityLeaseId::try_new(1).unwrap();
        supervisor
            .request_authority(lease, AuthorityMode::PointGoal, duration(10), at(5))
            .unwrap();
        assert_eq!(supervisor.tick(at(14)), Ok(SupervisorAction::None));
        assert_eq!(
            supervisor.tick(at(15)),
            Ok(SupervisorAction::BaseZeroRequired {
                reason: StopReason::AuthorityLeaseExpired
            })
        );
        assert!(matches!(
            supervisor.state(),
            SupervisorState::AwaitingZero { .. }
        ));
    }

    #[test]
    fn authority_expiring_while_zero_is_pending_stays_stopped() {
        let mut supervisor = ready();
        let lease = AuthorityLeaseId::try_new(1).unwrap();
        assert_eq!(
            supervisor.request_authority(lease, AuthorityMode::Explore, duration(10), at(25)),
            Ok(SupervisorAction::BaseZeroRequired {
                reason: StopReason::ZeroEvidenceStale
            })
        );
        assert_eq!(
            supervisor.admit_confirmed_zero(zero(35), at(35)),
            Ok(SupervisorAction::PendingAuthorityExpired)
        );
        assert!(matches!(
            supervisor.state(),
            SupervisorState::ReadyStopped { .. }
        ));
    }

    #[test]
    fn stale_future_and_wrong_controller_zero_never_grant_authority() {
        let mut stale = supervisor();
        stale.begin_inventory(at(1)).unwrap();
        stale.admit_readiness(readiness(1), at(2)).unwrap();
        stale.arm(at(3)).unwrap();
        assert_eq!(
            stale.admit_confirmed_zero(zero(4), at(24)),
            Err(SupervisorError::ZeroEvidenceStale)
        );
        assert!(matches!(
            stale.state(),
            SupervisorState::AwaitingZero { .. }
        ));

        let mut future = supervisor();
        future.begin_inventory(at(1)).unwrap();
        future.admit_readiness(readiness(1), at(2)).unwrap();
        future.arm(at(3)).unwrap();
        assert!(matches!(
            future.admit_confirmed_zero(zero(5), at(4)),
            Err(SupervisorError::ZeroObservedInFuture { .. })
        ));

        let wrong = AppliedResult {
            controller_uid: ControllerUid::try_new([9; 12]).unwrap(),
            boot_id: boot(),
            control_epoch: ControlEpoch::try_new(9).unwrap(),
            sequence: V2CommandSequence::FIRST,
            result: AppliedResultCode::Stopped,
            timer_pwm: TimerPwm::ZERO,
            output_state: OutputState::Disabled,
            applied_at: ControllerUptimeMsWrapping::new(1),
            expires_at: ControllerDeadlineMsWrapping::new(1),
            faults: ControllerFaults::NONE,
        };
        let wrong = ConfirmedBaseZero::try_from_applied_result(wrong, at(4)).unwrap();
        assert_eq!(
            future.admit_confirmed_zero(wrong, at(5)),
            Err(SupervisorError::ZeroControllerIdentityMismatch)
        );
    }

    #[test]
    fn zero_must_follow_the_stop_request_and_match_the_inventory_control_epoch() {
        let mut supervisor = supervisor();
        supervisor.begin_inventory(at(1)).unwrap();
        supervisor.admit_readiness(readiness(1), at(2)).unwrap();
        supervisor.arm(at(3)).unwrap();

        assert_eq!(
            supervisor.admit_confirmed_zero(zero(3), at(4)),
            Err(SupervisorError::ZeroEvidencePredatesStopRequest {
                observed_at_ns: 3,
                required_after_ns: 3,
            })
        );
        assert_eq!(
            supervisor.admit_confirmed_zero(zero_in_epoch(4, 10), at(4)),
            Err(SupervisorError::ZeroControlEpochMismatch {
                expected: ControlEpoch::try_new(9).unwrap(),
                actual: ControlEpoch::try_new(10).unwrap(),
            })
        );
        assert!(matches!(
            supervisor.state(),
            SupervisorState::AwaitingZero { .. }
        ));
    }

    #[test]
    fn clock_regression_and_fault_clear_cannot_resume_motion() {
        let mut supervisor = ready();
        let id = AuthorityLeaseId::try_new(1).unwrap();
        supervisor
            .request_authority(id, AuthorityMode::Explore, duration(50), at(5))
            .unwrap();
        assert!(matches!(
            supervisor.tick(at(4)),
            Err(SupervisorError::ClockRegression { .. })
        ));
        assert_eq!(
            supervisor.state(),
            SupervisorState::FaultLatched {
                fault: FaultKind::ClockRegression
            }
        );
        assert_eq!(
            supervisor.clear_fault_for_inventory(at(6)),
            Ok(SupervisorAction::InventoryRequired)
        );
        assert_eq!(supervisor.state(), SupervisorState::Inventory);
    }

    #[test]
    fn malformed_applied_results_cannot_become_zero_evidence() {
        let base = AppliedResult {
            controller_uid: uid(),
            boot_id: boot(),
            control_epoch: ControlEpoch::try_new(9).unwrap(),
            sequence: V2CommandSequence::FIRST,
            result: AppliedResultCode::AppliedNew,
            timer_pwm: TimerPwm::try_new(1, 0).unwrap(),
            output_state: OutputState::NonzeroPwm,
            applied_at: ControllerUptimeMsWrapping::new(1),
            expires_at: ControllerDeadlineMsWrapping::new(2),
            faults: ControllerFaults::NONE,
        };
        assert!(matches!(
            ConfirmedBaseZero::try_from_applied_result(base, at(1)),
            Err(ZeroEvidenceError::NonzeroPwm { .. })
        ));

        let mut rejected = base;
        rejected.timer_pwm = TimerPwm::ZERO;
        rejected.output_state = OutputState::ZeroPwm;
        rejected.result = AppliedResultCode::RejectedSequence;
        assert!(matches!(
            ConfirmedBaseZero::try_from_applied_result(rejected, at(1)),
            Err(ZeroEvidenceError::ResultDoesNotProveApplication { .. })
        ));

        let mut cached = rejected;
        cached.result = AppliedResultCode::DuplicateCached;
        assert!(matches!(
            ConfirmedBaseZero::try_from_applied_result(cached, at(1)),
            Err(ZeroEvidenceError::ResultDoesNotProveApplication {
                result: AppliedResultCode::DuplicateCached
            })
        ));

        let host = HostCommandResult {
            controller_uid: uid(),
            boot_id: boot(),
            control_epoch: ControlEpoch::try_new(9).unwrap(),
            sequence: V2CommandSequence::FIRST,
            result: HostCommandResultCode::AppliedNew,
            requested_timer_pwm: TimerPwm::ZERO,
            controller_timer_pwm: TimerPwm::ZERO,
            output_state: OutputState::ZeroPwm,
            controller_applied_at: ControllerUptimeMsWrapping::new(1),
            controller_expires_at: ControllerDeadlineMsWrapping::new(2),
            remaining_lease: RemainingLeaseMs::ZERO,
            faults: ControllerFaults::NONE,
        };
        assert!(ConfirmedBaseZero::try_from_host_command_result(host, at(1)).is_ok());

        let mut cached_host = host;
        cached_host.result = HostCommandResultCode::DuplicateCached;
        assert!(matches!(
            ConfirmedBaseZero::try_from_host_command_result(cached_host, at(1)),
            Err(ZeroEvidenceError::HostResultDoesNotProveFreshApplication {
                result: HostCommandResultCode::DuplicateCached
            })
        ));

        let mut nonzero_request = host;
        nonzero_request.requested_timer_pwm = TimerPwm::try_new(1, -1).unwrap();
        assert!(matches!(
            ConfirmedBaseZero::try_from_host_command_result(nonzero_request, at(1)),
            Err(ZeroEvidenceError::RequestedNonzeroPwm { left: 1, right: -1 })
        ));
    }
}
