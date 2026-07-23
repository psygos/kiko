use robot_protocol::ControllerUptimeMsWrapping;
use robot_protocol::v2::{
    ActuatorConfigFingerprint, ControlEpoch, ControllerBootId, ControllerCapabilities,
    ControllerDeadlineMsWrapping, ControllerFaults, ControllerUid, HostCommandResult,
    HostCommandResultCode, OutputState, RemainingLeaseMs, RequestId, TimerPwm, V2CommandLeaseMs,
    V2CommandSequence,
};
use std::num::{NonZeroU16, NonZeroU32};
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

/// Exact controller identity and capabilities retained from one verified
/// `AcquireResult`.
///
/// Callers cannot construct this value. It is minted only after the command
/// client has matched the response to the request and verified UID, boot ID,
/// control epoch, firmware ABI/build, actuator fingerprint, required safety
/// capabilities, and clear controller faults. This is acquisition-time
/// evidence; it does not claim continuing liveness after the response.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VerifiedControllerAcquisition {
    session: ControllerSession,
    firmware_abi: NonZeroU16,
    firmware_build_id: NonZeroU32,
    actuator_config_fingerprint: ActuatorConfigFingerprint,
    capabilities: ControllerCapabilities,
}

impl VerifiedControllerAcquisition {
    pub(crate) const fn new(
        session: ControllerSession,
        firmware_abi: NonZeroU16,
        firmware_build_id: NonZeroU32,
        actuator_config_fingerprint: ActuatorConfigFingerprint,
        capabilities: ControllerCapabilities,
    ) -> Self {
        Self {
            session,
            firmware_abi,
            firmware_build_id,
            actuator_config_fingerprint,
            capabilities,
        }
    }

    pub const fn controller_session(self) -> ControllerSession {
        self.session
    }

    pub const fn controller_uid(self) -> ControllerUid {
        self.session.controller_uid()
    }

    pub const fn boot_id(self) -> ControllerBootId {
        self.session.boot_id()
    }

    pub const fn control_epoch(self) -> ControlEpoch {
        self.session.control_epoch()
    }

    pub const fn firmware_abi(self) -> u16 {
        self.firmware_abi.get()
    }

    pub const fn firmware_build_id(self) -> u32 {
        self.firmware_build_id.get()
    }

    pub const fn actuator_config_fingerprint(self) -> ActuatorConfigFingerprint {
        self.actuator_config_fingerprint
    }

    pub const fn capabilities(self) -> ControllerCapabilities {
        self.capabilities
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct PendingPhysicalCommand {
    requested_timer_pwm: TimerPwm,
    lease: V2CommandLeaseMs,
    acknowledgement_deadline_exclusive: MonotonicInstant,
}

impl PendingPhysicalCommand {
    pub const fn new(
        requested_timer_pwm: TimerPwm,
        lease: V2CommandLeaseMs,
        acknowledgement_deadline_exclusive: MonotonicInstant,
    ) -> Self {
        Self {
            requested_timer_pwm,
            lease,
            acknowledgement_deadline_exclusive,
        }
    }

    pub const fn requested_timer_pwm(&self) -> TimerPwm {
        self.requested_timer_pwm
    }

    pub const fn lease(&self) -> V2CommandLeaseMs {
        self.lease
    }

    /// Latest host time at which an exact application acknowledgement can
    /// authorize this command. This admission deadline does not shorten the
    /// controller lease proven by a successful receipt.
    pub const fn acknowledgement_deadline_exclusive(&self) -> MonotonicInstant {
        self.acknowledgement_deadline_exclusive
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct AppliedCommandReceipt {
    controller_session: ControllerSession,
    verified_host_result: HostCommandResult,
    requested_lease: V2CommandLeaseMs,
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
        requested_lease: V2CommandLeaseMs,
        result: HostCommandResult,
        timing: ReceiptTiming,
    ) -> Self {
        Self {
            controller_session,
            verified_host_result: result,
            requested_lease,
            sent_at: timing.sent_at,
            acknowledged_at: timing.acknowledged_at,
            known_active_through_exclusive: timing.known_active_through_exclusive,
        }
    }

    pub const fn controller_session(&self) -> ControllerSession {
        self.controller_session
    }

    pub const fn sequence(&self) -> V2CommandSequence {
        self.verified_host_result.sequence
    }

    pub const fn result(&self) -> HostCommandResultCode {
        self.verified_host_result.result
    }

    pub const fn applied_timer_pwm(&self) -> TimerPwm {
        self.verified_host_result.controller_timer_pwm
    }

    pub const fn output_state(&self) -> OutputState {
        self.verified_host_result.output_state
    }

    pub const fn controller_applied_at(&self) -> ControllerUptimeMsWrapping {
        self.verified_host_result.controller_applied_at
    }

    pub const fn controller_expires_at(&self) -> ControllerDeadlineMsWrapping {
        self.verified_host_result.controller_expires_at
    }

    pub const fn requested_lease(&self) -> V2CommandLeaseMs {
        self.requested_lease
    }

    pub const fn remaining_lease_at_server_emission(&self) -> RemainingLeaseMs {
        self.verified_host_result.remaining_lease
    }

    pub const fn controller_faults(&self) -> ControllerFaults {
        self.verified_host_result.faults
    }

    /// Exact server result retained after all identity, sequence, PWM, output,
    /// fault, and controller-deadline checks succeeded.
    pub const fn verified_host_result(&self) -> HostCommandResult {
        self.verified_host_result
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
        self.verified_host_result.controller_timer_pwm.is_zero()
            && self.verified_host_result.output_state.is_safe()
            && self.verified_host_result.faults.is_clear()
            && self
                .verified_host_result
                .result
                .proves_controller_application()
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
