//! Shared exclusion between the sole physical base owner and head-gaze writes.
//!
//! The interlock starts only from parsed, freshly applied base-zero evidence.
//! It does not estimate motion from IMU data and it does not infer "stopped"
//! from elapsed time. Every later base transport transaction must enter the
//! state machine before sending and commit its exact verified outcome after
//! acknowledgement. Dropping an unfinished transaction faults the interlock.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

use kiko_supervisor_core::ConfirmedBaseZero;

const BASE_ZERO: u8 = 0;
const BASE_TRANSACTION: u8 = 1;
const BASE_MAY_MOVE: u8 = 2;
const HEAD_GAZE_TRANSACTION: u8 = 3;
const FAULTED: u8 = 4;

#[derive(Debug)]
struct SharedBaseMotionState {
    phase: AtomicU8,
}

impl SharedBaseMotionState {
    const fn new() -> Self {
        Self {
            phase: AtomicU8::new(BASE_ZERO),
        }
    }

    fn phase(&self) -> HeadGazeBaseMotionPhase {
        HeadGazeBaseMotionPhase::from_raw(self.phase.load(Ordering::SeqCst))
    }

    fn fault(&self) {
        self.phase.store(FAULTED, Ordering::SeqCst);
    }
}

/// Observable state of the head/base exclusion boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HeadGazeBaseMotionPhase {
    ConfirmedZero,
    BaseTransaction,
    BaseMayMove,
    HeadGazeTransaction,
    Faulted,
}

impl HeadGazeBaseMotionPhase {
    fn from_raw(raw: u8) -> Self {
        match raw {
            BASE_ZERO => Self::ConfirmedZero,
            BASE_TRANSACTION => Self::BaseTransaction,
            BASE_MAY_MOVE => Self::BaseMayMove,
            HEAD_GAZE_TRANSACTION => Self::HeadGazeTransaction,
            FAULTED => Self::Faulted,
            _ => Self::Faulted,
        }
    }
}

/// Non-cloneable endpoint retained by the sole physical base-command owner.
#[derive(Debug)]
pub struct HeadGazeBaseMotionInterlock {
    shared: Arc<SharedBaseMotionState>,
    controller_zero_origin: ConfirmedBaseZero,
}

/// Cloneable request endpoint for the physical head-gaze owner.
///
/// Cloning this endpoint creates no authority. Exactly one live lease can be
/// minted, and only while the base owner retains confirmed-zero state.
#[derive(Clone, Debug)]
pub struct HeadGazeBaseZeroExclusiveLeaseIssuer {
    shared: Arc<SharedBaseMotionState>,
}

impl HeadGazeBaseMotionInterlock {
    /// Create the interlock from freshly parsed base-zero evidence.
    ///
    /// `ConfirmedBaseZero` has no unchecked constructor: it is admitted from
    /// an exact applied controller result by `kiko-supervisor-core`.
    pub fn from_confirmed_zero(
        controller_zero_origin: ConfirmedBaseZero,
    ) -> (Self, HeadGazeBaseZeroExclusiveLeaseIssuer) {
        let shared = Arc::new(SharedBaseMotionState::new());
        (
            Self {
                shared: Arc::clone(&shared),
                controller_zero_origin,
            },
            HeadGazeBaseZeroExclusiveLeaseIssuer { shared },
        )
    }

    pub fn phase(&self) -> HeadGazeBaseMotionPhase {
        self.shared.phase()
    }

    pub const fn controller_zero_origin(&self) -> ConfirmedBaseZero {
        self.controller_zero_origin
    }

    /// Reserve the interlock before starting any base transport transaction.
    pub fn begin_base_transaction(
        &mut self,
    ) -> Result<HeadGazeBaseCommandTransaction, HeadGazeBaseInterlockError> {
        let mut observed = self.shared.phase.load(Ordering::SeqCst);
        loop {
            match observed {
                BASE_ZERO | BASE_MAY_MOVE => {
                    match self.shared.phase.compare_exchange(
                        observed,
                        BASE_TRANSACTION,
                        Ordering::SeqCst,
                        Ordering::SeqCst,
                    ) {
                        Ok(_) => {
                            return Ok(HeadGazeBaseCommandTransaction {
                                shared: Arc::clone(&self.shared),
                                controller_zero_origin: self.controller_zero_origin,
                                completed: false,
                            });
                        }
                        Err(actual) => observed = actual,
                    }
                }
                BASE_TRANSACTION => {
                    return Err(HeadGazeBaseInterlockError::BaseTransactionAlreadyActive);
                }
                HEAD_GAZE_TRANSACTION => {
                    return Err(HeadGazeBaseInterlockError::HeadGazeLeaseActive);
                }
                FAULTED => return Err(HeadGazeBaseInterlockError::Faulted),
                _ => {
                    self.shared.fault();
                    return Err(HeadGazeBaseInterlockError::InvalidInternalState(observed));
                }
            }
        }
    }

    /// Permanently fail closed after a base-owner invariant or transport
    /// failure. A new controller acquisition and fresh zero are required.
    pub fn fault(&mut self) {
        self.shared.fault();
    }
}

impl Drop for HeadGazeBaseMotionInterlock {
    fn drop(&mut self) {
        // A surviving cloneable issuer is never authority after the sole base
        // owner disappears. The shared state must become absorbing before the
        // non-cloneable endpoint releases its final ownership evidence.
        self.shared.fault();
    }
}

impl HeadGazeBaseZeroExclusiveLeaseIssuer {
    pub fn phase(&self) -> HeadGazeBaseMotionPhase {
        self.shared.phase()
    }

    /// Mint one exclusive lease only from confirmed-zero state.
    pub fn try_acquire(
        &self,
    ) -> Result<HeadGazeBaseZeroExclusiveLease, HeadGazeBaseInterlockError> {
        match self.shared.phase.compare_exchange(
            BASE_ZERO,
            HEAD_GAZE_TRANSACTION,
            Ordering::SeqCst,
            Ordering::SeqCst,
        ) {
            Ok(_) => Ok(HeadGazeBaseZeroExclusiveLease {
                shared: Arc::clone(&self.shared),
            }),
            Err(BASE_TRANSACTION) => Err(HeadGazeBaseInterlockError::BaseTransactionAlreadyActive),
            Err(BASE_MAY_MOVE) => Err(HeadGazeBaseInterlockError::BaseMayMove),
            Err(HEAD_GAZE_TRANSACTION) => Err(HeadGazeBaseInterlockError::HeadGazeLeaseActive),
            Err(FAULTED) => Err(HeadGazeBaseInterlockError::Faulted),
            Err(raw) => {
                self.shared.fault();
                Err(HeadGazeBaseInterlockError::InvalidInternalState(raw))
            }
        }
    }
}

/// Exclusive capability consumed by one complete head goal-write/readback.
#[derive(Debug)]
pub struct HeadGazeBaseZeroExclusiveLease {
    shared: Arc<SharedBaseMotionState>,
}

impl Drop for HeadGazeBaseZeroExclusiveLease {
    fn drop(&mut self) {
        if self
            .shared
            .phase
            .compare_exchange(
                HEAD_GAZE_TRANSACTION,
                BASE_ZERO,
                Ordering::SeqCst,
                Ordering::SeqCst,
            )
            .is_err()
        {
            self.shared.fault();
        }
    }
}

/// In-flight base command reservation.
///
/// If this value is dropped without an exact verified commit, gaze remains
/// disabled for the life of this interlock.
#[must_use = "an unfinished physical base transaction faults head-gaze authority"]
#[derive(Debug)]
pub struct HeadGazeBaseCommandTransaction {
    shared: Arc<SharedBaseMotionState>,
    controller_zero_origin: ConfirmedBaseZero,
    completed: bool,
}

impl HeadGazeBaseCommandTransaction {
    /// Commit a freshly acknowledged zero command.
    ///
    /// The new evidence must identify the same controller boot and control
    /// epoch that originated the interlock. Reacquisition must create a new
    /// interlock rather than rebinding an existing head capability.
    pub fn commit_confirmed_zero(
        mut self,
        zero: ConfirmedBaseZero,
    ) -> Result<(), HeadGazeBaseInterlockError> {
        if zero.controller_uid() != self.controller_zero_origin.controller_uid()
            || zero.controller_boot_id() != self.controller_zero_origin.controller_boot_id()
            || zero.control_epoch() != self.controller_zero_origin.control_epoch()
        {
            self.shared.fault();
            self.completed = true;
            return Err(HeadGazeBaseInterlockError::ControllerIdentityChanged);
        }
        self.commit(BASE_ZERO)
    }

    /// Commit an exact verified nonzero application.
    ///
    /// This method must be called only after the physical command client has
    /// verified controller identity, sequence, requested/applied PWM, output
    /// state, faults, and acknowledgement timing.
    pub fn commit_verified_motion_application(mut self) -> Result<(), HeadGazeBaseInterlockError> {
        self.commit(BASE_MAY_MOVE)
    }

    fn commit(&mut self, target: u8) -> Result<(), HeadGazeBaseInterlockError> {
        match self.shared.phase.compare_exchange(
            BASE_TRANSACTION,
            target,
            Ordering::SeqCst,
            Ordering::SeqCst,
        ) {
            Ok(_) => {
                self.completed = true;
                Ok(())
            }
            Err(actual) => {
                self.shared.fault();
                self.completed = true;
                Err(HeadGazeBaseInterlockError::CommitOutsideBaseTransaction {
                    actual: HeadGazeBaseMotionPhase::from_raw(actual),
                })
            }
        }
    }
}

impl Drop for HeadGazeBaseCommandTransaction {
    fn drop(&mut self) {
        if !self.completed {
            self.shared.fault();
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadGazeBaseInterlockError {
    BaseTransactionAlreadyActive,
    BaseMayMove,
    HeadGazeLeaseActive,
    Faulted,
    ControllerIdentityChanged,
    CommitOutsideBaseTransaction { actual: HeadGazeBaseMotionPhase },
    InvalidInternalState(u8),
}

impl fmt::Display for HeadGazeBaseInterlockError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "head/base motion interlock rejected operation: {self:?}"
        )
    }
}

impl std::error::Error for HeadGazeBaseInterlockError {}

#[cfg(test)]
mod tests {
    use super::*;
    use kiko_supervisor_core::MonotonicInstant;
    use robot_protocol::ControllerUptimeMsWrapping;
    use robot_protocol::v2::{
        ControlEpoch, ControllerBootId, ControllerDeadlineMsWrapping, ControllerFaults,
        ControllerUid, HostCommandResult, HostCommandResultCode, OutputState, RemainingLeaseMs,
        TimerPwm, V2CommandSequence,
    };

    fn confirmed_zero(sequence: u32) -> ConfirmedBaseZero {
        let result = HostCommandResult {
            controller_uid: ControllerUid::try_new([7; 12]).expect("nonzero controller UID"),
            boot_id: ControllerBootId::try_new(11).expect("nonzero boot ID"),
            control_epoch: ControlEpoch::try_new(13).expect("nonzero control epoch"),
            sequence: V2CommandSequence::new(sequence),
            result: HostCommandResultCode::AppliedNew,
            requested_timer_pwm: TimerPwm::ZERO,
            controller_timer_pwm: TimerPwm::ZERO,
            output_state: OutputState::ZeroPwm,
            controller_applied_at: ControllerUptimeMsWrapping::new(1),
            controller_expires_at: ControllerDeadlineMsWrapping::new(2),
            remaining_lease: RemainingLeaseMs::ZERO,
            faults: ControllerFaults::NONE,
        };
        ConfirmedBaseZero::try_from_host_command_result(
            result,
            MonotonicInstant::from_nanos_since_process_start(sequence.into()),
        )
        .expect("fixture is exact fresh zero")
    }

    #[test]
    fn head_lease_excludes_base_and_release_restores_zero() {
        let (mut base, head) = HeadGazeBaseMotionInterlock::from_confirmed_zero(confirmed_zero(1));
        let lease = head.try_acquire().expect("zero permits gaze");
        assert_eq!(base.phase(), HeadGazeBaseMotionPhase::HeadGazeTransaction);
        assert!(matches!(
            base.begin_base_transaction(),
            Err(HeadGazeBaseInterlockError::HeadGazeLeaseActive)
        ));
        drop(lease);
        assert_eq!(base.phase(), HeadGazeBaseMotionPhase::ConfirmedZero);
    }

    #[test]
    fn nonzero_application_blocks_gaze_until_fresh_zero() {
        let (mut base, head) = HeadGazeBaseMotionInterlock::from_confirmed_zero(confirmed_zero(1));
        base.begin_base_transaction()
            .expect("base transaction")
            .commit_verified_motion_application()
            .expect("verified motion commit");
        assert_eq!(base.phase(), HeadGazeBaseMotionPhase::BaseMayMove);
        assert!(matches!(
            head.try_acquire(),
            Err(HeadGazeBaseInterlockError::BaseMayMove)
        ));

        base.begin_base_transaction()
            .expect("zero transaction")
            .commit_confirmed_zero(confirmed_zero(2))
            .expect("fresh zero commit");
        let _lease = head.try_acquire().expect("fresh zero re-enables gaze");
    }

    #[test]
    fn dropped_base_transaction_fails_closed_permanently() {
        let (mut base, head) = HeadGazeBaseMotionInterlock::from_confirmed_zero(confirmed_zero(1));
        drop(base.begin_base_transaction().expect("base transaction"));
        assert_eq!(base.phase(), HeadGazeBaseMotionPhase::Faulted);
        assert!(matches!(
            head.try_acquire(),
            Err(HeadGazeBaseInterlockError::Faulted)
        ));
        assert!(matches!(
            base.begin_base_transaction(),
            Err(HeadGazeBaseInterlockError::Faulted)
        ));
    }

    #[test]
    fn controller_identity_change_faults_interlock() {
        let (mut base, head) = HeadGazeBaseMotionInterlock::from_confirmed_zero(confirmed_zero(1));
        let mut wrong = confirmed_zero(2);
        let result = HostCommandResult {
            controller_uid: wrong.controller_uid(),
            boot_id: ControllerBootId::try_new(99).expect("nonzero boot ID"),
            control_epoch: wrong.control_epoch(),
            sequence: wrong.sequence(),
            result: HostCommandResultCode::AppliedNew,
            requested_timer_pwm: TimerPwm::ZERO,
            controller_timer_pwm: TimerPwm::ZERO,
            output_state: OutputState::ZeroPwm,
            controller_applied_at: ControllerUptimeMsWrapping::new(1),
            controller_expires_at: ControllerDeadlineMsWrapping::new(2),
            remaining_lease: RemainingLeaseMs::ZERO,
            faults: ControllerFaults::NONE,
        };
        wrong = ConfirmedBaseZero::try_from_host_command_result(
            result,
            MonotonicInstant::from_nanos_since_process_start(2),
        )
        .expect("wrong-controller fixture is otherwise valid");
        assert!(matches!(
            base.begin_base_transaction()
                .expect("base transaction")
                .commit_confirmed_zero(wrong),
            Err(HeadGazeBaseInterlockError::ControllerIdentityChanged)
        ));
        assert_eq!(head.phase(), HeadGazeBaseMotionPhase::Faulted);
    }

    #[test]
    fn duplicate_head_issuers_cannot_mint_duplicate_leases() {
        let (_base, head) = HeadGazeBaseMotionInterlock::from_confirmed_zero(confirmed_zero(1));
        let duplicate = head.clone();
        let _lease = head.try_acquire().expect("first lease");
        assert!(matches!(
            duplicate.try_acquire(),
            Err(HeadGazeBaseInterlockError::HeadGazeLeaseActive)
        ));
    }

    #[test]
    fn dropping_sole_base_owner_revokes_every_retained_gaze_issuer() {
        let (owner, issuer) = HeadGazeBaseMotionInterlock::from_confirmed_zero(confirmed_zero(1));
        let retained = issuer.clone();
        drop(owner);

        assert_eq!(issuer.phase(), HeadGazeBaseMotionPhase::Faulted);
        assert!(matches!(
            retained.try_acquire(),
            Err(HeadGazeBaseInterlockError::Faulted)
        ));
    }
}
