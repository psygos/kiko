use robot_protocol::v2::{
    ControlEpoch, ControllerBootId, ControllerFaults, ControllerUid, OutputState, RemainingLeaseMs,
    RequestId, TimerPwm, V2CommandLeaseMs, V2CommandSequence,
};
use std::time::Duration;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct MonotonicInstant(u128);

impl MonotonicInstant {
    pub const ZERO: Self = Self(0);

    pub const fn from_nanos_since_clock_start(nanoseconds: u128) -> Self {
        Self(nanoseconds)
    }

    pub const fn nanos_since_clock_start(self) -> u128 {
        self.0
    }

    pub fn checked_add(self, duration: Duration) -> Option<Self> {
        self.0.checked_add(duration.as_nanos()).map(Self)
    }

    pub fn checked_duration_since(self, earlier: Self) -> Option<Duration> {
        let nanoseconds = self.0.checked_sub(earlier.0)?;
        let seconds = u64::try_from(nanoseconds / 1_000_000_000).ok()?;
        let subsecond_nanoseconds = u32::try_from(nanoseconds % 1_000_000_000).ok()?;
        Some(Duration::new(seconds, subsecond_nanoseconds))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ControllerSession {
    controller_uid: ControllerUid,
    boot_id: ControllerBootId,
    control_epoch: ControlEpoch,
}

impl ControllerSession {
    pub(crate) const fn from_verified_acquisition(
        controller_uid: ControllerUid,
        boot_id: ControllerBootId,
        control_epoch: ControlEpoch,
    ) -> Self {
        Self {
            controller_uid,
            boot_id,
            control_epoch,
        }
    }

    pub const fn controller_uid(self) -> ControllerUid {
        self.controller_uid
    }

    pub const fn boot_id(self) -> ControllerBootId {
        self.boot_id
    }

    pub const fn control_epoch(self) -> ControlEpoch {
        self.control_epoch
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct PendingPhysicalCommand {
    requested_timer_pwm: TimerPwm,
    lease: V2CommandLeaseMs,
    valid_through_exclusive: MonotonicInstant,
}

impl PendingPhysicalCommand {
    pub const fn new(
        requested_timer_pwm: TimerPwm,
        lease: V2CommandLeaseMs,
        valid_through_exclusive: MonotonicInstant,
    ) -> Self {
        Self {
            requested_timer_pwm,
            lease,
            valid_through_exclusive,
        }
    }

    pub const fn requested_timer_pwm(&self) -> TimerPwm {
        self.requested_timer_pwm
    }

    pub const fn lease(&self) -> V2CommandLeaseMs {
        self.lease
    }

    pub const fn valid_through_exclusive(&self) -> MonotonicInstant {
        self.valid_through_exclusive
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct AppliedCommandReceipt {
    controller_session: ControllerSession,
    sequence: V2CommandSequence,
    applied_timer_pwm: TimerPwm,
    requested_lease: V2CommandLeaseMs,
    remaining_lease_at_server_emission: RemainingLeaseMs,
    sent_at: MonotonicInstant,
    acknowledged_at: MonotonicInstant,
    known_active_through_exclusive: MonotonicInstant,
}

pub(crate) struct ReceiptTiming {
    pub sent_at: MonotonicInstant,
    pub acknowledged_at: MonotonicInstant,
    pub known_active_through_exclusive: MonotonicInstant,
}

impl AppliedCommandReceipt {
    pub(crate) const fn new(
        controller_session: ControllerSession,
        sequence: V2CommandSequence,
        applied_timer_pwm: TimerPwm,
        requested_lease: V2CommandLeaseMs,
        remaining_lease_at_server_emission: RemainingLeaseMs,
        timing: ReceiptTiming,
    ) -> Self {
        Self {
            controller_session,
            sequence,
            applied_timer_pwm,
            requested_lease,
            remaining_lease_at_server_emission,
            sent_at: timing.sent_at,
            acknowledged_at: timing.acknowledged_at,
            known_active_through_exclusive: timing.known_active_through_exclusive,
        }
    }

    pub const fn controller_session(&self) -> ControllerSession {
        self.controller_session
    }

    pub const fn sequence(&self) -> V2CommandSequence {
        self.sequence
    }

    pub const fn applied_timer_pwm(&self) -> TimerPwm {
        self.applied_timer_pwm
    }

    pub const fn requested_lease(&self) -> V2CommandLeaseMs {
        self.requested_lease
    }

    pub const fn remaining_lease_at_server_emission(&self) -> RemainingLeaseMs {
        self.remaining_lease_at_server_emission
    }

    pub const fn sent_at(&self) -> MonotonicInstant {
        self.sent_at
    }

    pub const fn acknowledged_at(&self) -> MonotonicInstant {
        self.acknowledged_at
    }

    pub const fn known_active_through_exclusive(&self) -> MonotonicInstant {
        self.known_active_through_exclusive
    }

    pub const fn is_confirmed_zero(&self) -> bool {
        self.applied_timer_pwm.is_zero()
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct DisarmReceipt {
    controller_uid: ControllerUid,
    observed_boot_id: ControllerBootId,
    request_id: RequestId,
    output_state: OutputState,
    controller_faults: ControllerFaults,
    acknowledged_at: MonotonicInstant,
}

impl DisarmReceipt {
    pub(crate) const fn new(
        controller_uid: ControllerUid,
        observed_boot_id: ControllerBootId,
        request_id: RequestId,
        output_state: OutputState,
        controller_faults: ControllerFaults,
        acknowledged_at: MonotonicInstant,
    ) -> Self {
        Self {
            controller_uid,
            observed_boot_id,
            request_id,
            output_state,
            controller_faults,
            acknowledged_at,
        }
    }

    pub const fn controller_uid(&self) -> ControllerUid {
        self.controller_uid
    }

    pub const fn observed_boot_id(&self) -> ControllerBootId {
        self.observed_boot_id
    }

    pub const fn request_id(&self) -> RequestId {
        self.request_id
    }

    pub const fn output_state(&self) -> OutputState {
        self.output_state
    }

    pub const fn controller_faults(&self) -> ControllerFaults {
        self.controller_faults
    }

    pub const fn acknowledged_at(&self) -> MonotonicInstant {
        self.acknowledged_at
    }
}
