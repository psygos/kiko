//! Transport-independent KEP2 host session state machine.

use core::{fmt, num::NonZeroU64};

use kiko_expression_core::MonotonicTimestamp;
use kiko_eye_protocol::{
    AcquireControl, AcquireResult, AcquireResultCode, ApplyIntent, Capabilities, ControlEpoch,
    DeviceBootId, DeviceTimestampMs, DeviceUid, FirmwareBuildId, HandshakeNonce, IdentityReport,
    IntentLeaseMs, IntentResult, IntentResultCode, IntentSequence, Message, ReleaseControl,
    ReleaseReason, RenderedFrameSequence,
};

use crate::PreparedEyeIntent;

/// Capabilities used by every field emitted by this runtime plus autonomous
/// fallback and explicit firmware admission reports.
pub const REQUIRED_EYE_CAPABILITIES: u32 = Capabilities::GAZE
    | Capabilities::LID
    | Capabilities::PUPIL
    | Capabilities::COLOR
    | Capabilities::BRIGHTNESS
    | Capabilities::BLINK
    | Capabilities::AUTONOMOUS_FALLBACK
    | Capabilities::APPLIED_REPORT;

/// Non-zero caller-generated nonce used for one handshake phase.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SessionNonce(HandshakeNonce);

impl SessionNonce {
    pub const fn try_new(value: u64) -> Result<Self, SessionPlanError> {
        let Some(value) = NonZeroU64::new(value) else {
            return Err(SessionPlanError::ZeroNonce);
        };
        Ok(Self(HandshakeNonce::from_nonzero(value)))
    }

    pub const fn get(self) -> u64 {
        self.0.get()
    }

    const fn protocol_nonce(self) -> HandshakeNonce {
        self.0
    }
}

/// Exact identity allow-listed for one physical eye controller.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpectedEyeIdentity {
    device_uid: DeviceUid,
    firmware_build_id: FirmwareBuildId,
    capabilities: Capabilities,
}

impl ExpectedEyeIdentity {
    pub const fn new(
        device_uid: DeviceUid,
        firmware_build_id: FirmwareBuildId,
        capabilities: Capabilities,
    ) -> Self {
        Self {
            device_uid,
            firmware_build_id,
            capabilities,
        }
    }

    pub const fn device_uid(self) -> DeviceUid {
        self.device_uid
    }

    pub const fn firmware_build_id(self) -> FirmwareBuildId {
        self.firmware_build_id
    }

    pub const fn capabilities(self) -> Capabilities {
        self.capabilities
    }
}

/// Immutable identities and nonces for one attempted control session.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EyeSessionPlan {
    expected_identity: ExpectedEyeIdentity,
    identity_nonce: SessionNonce,
    acquire_nonce: SessionNonce,
    control_epoch: ControlEpoch,
}

impl EyeSessionPlan {
    pub fn try_new(
        expected_identity: ExpectedEyeIdentity,
        identity_nonce: SessionNonce,
        acquire_nonce: SessionNonce,
        control_epoch: ControlEpoch,
    ) -> Result<Self, SessionPlanError> {
        if identity_nonce == acquire_nonce {
            return Err(SessionPlanError::ReusedNonce {
                value: identity_nonce.get(),
            });
        }
        let actual = expected_identity.capabilities.bits();
        if actual & REQUIRED_EYE_CAPABILITIES != REQUIRED_EYE_CAPABILITIES {
            return Err(SessionPlanError::MissingRequiredCapabilities {
                required: REQUIRED_EYE_CAPABILITIES,
                actual,
            });
        }
        Ok(Self {
            expected_identity,
            identity_nonce,
            acquire_nonce,
            control_epoch,
        })
    }

    pub const fn expected_identity(self) -> ExpectedEyeIdentity {
        self.expected_identity
    }

    pub const fn identity_nonce(self) -> SessionNonce {
        self.identity_nonce
    }

    pub const fn acquire_nonce(self) -> SessionNonce {
        self.acquire_nonce
    }

    pub const fn control_epoch(self) -> ControlEpoch {
        self.control_epoch
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SessionPlanError {
    ZeroNonce,
    ReusedNonce { value: u64 },
    MissingRequiredCapabilities { required: u32, actual: u32 },
}

impl fmt::Display for SessionPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid eye-session plan: {self:?}")
    }
}

impl core::error::Error for SessionPlanError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ControlBinding {
    boot_id: DeviceBootId,
    control_epoch: ControlEpoch,
}

impl ControlBinding {
    pub const fn boot_id(self) -> DeviceBootId {
        self.boot_id
    }

    pub const fn control_epoch(self) -> ControlEpoch {
        self.control_epoch
    }
}

/// Exact proof returned by KEP2 that firmware admitted an intent.
///
/// It is intentionally named "admission", not "display": neither this report
/// nor this host crate observes the physical LEDs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FirmwareAdmission {
    binding: ControlBinding,
    sequence: IntentSequence,
    lease: IntentLeaseMs,
    applied_at: DeviceTimestampMs,
    expires_at: DeviceTimestampMs,
    rendered_frame_sequence: RenderedFrameSequence,
}

impl FirmwareAdmission {
    pub const fn binding(self) -> ControlBinding {
        self.binding
    }

    pub const fn sequence(self) -> IntentSequence {
        self.sequence
    }

    pub const fn lease(self) -> IntentLeaseMs {
        self.lease
    }

    pub const fn applied_at(self) -> DeviceTimestampMs {
        self.applied_at
    }

    pub const fn expires_at(self) -> DeviceTimestampMs {
        self.expires_at
    }

    pub const fn rendered_frame_sequence(self) -> RenderedFrameSequence {
        self.rendered_frame_sequence
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SessionPhase {
    Ready,
    AwaitingIdentity,
    IdentityVerified,
    AwaitingAcquire,
    Active,
    AwaitingIntentResult,
    AwaitingReleaseResult,
    Released,
    Fallback,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SessionEvent {
    IdentityVerified(IdentityReport),
    ControlAcquired(ControlBinding),
    IntentAdmitted(FirmwareAdmission),
    Released(ControlBinding),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InboundMessageKind {
    IdentityQuery,
    IdentityReport,
    AcquireControl,
    AcquireResult,
    ApplyIntent,
    ReleaseControl,
    IntentResult,
}

impl From<Message> for InboundMessageKind {
    fn from(value: Message) -> Self {
        match value {
            Message::IdentityQuery { .. } => Self::IdentityQuery,
            Message::IdentityReport(_) => Self::IdentityReport,
            Message::AcquireControl(_) => Self::AcquireControl,
            Message::AcquireResult(_) => Self::AcquireResult,
            Message::ApplyIntent(_) => Self::ApplyIntent,
            Message::ReleaseControl(_) => Self::ReleaseControl,
            Message::IntentResult(_) => Self::IntentResult,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ActiveState {
    binding: ControlBinding,
    next_sequence: IntentSequence,
    last_device_event: DeviceTimestampMs,
    last_host_event: MonotonicTimestamp,
    last_admitted_sequence: Option<IntentSequence>,
    last_rendered_frame_sequence: Option<RenderedFrameSequence>,
}

#[derive(Clone, Copy, Debug)]
struct PendingIntent {
    active: ActiveState,
    sequence: IntentSequence,
    successor: IntentSequence,
    lease: IntentLeaseMs,
    prepared: PreparedEyeIntent,
}

#[derive(Clone, Copy, Debug)]
struct PendingRelease {
    active: ActiveState,
    sequence: IntentSequence,
}

#[derive(Clone, Copy, Debug)]
enum State {
    Ready,
    AwaitingIdentity {
        sent_at: MonotonicTimestamp,
    },
    IdentityVerified {
        report: IdentityReport,
        last_host_event: MonotonicTimestamp,
    },
    AwaitingAcquire {
        report: IdentityReport,
        last_host_event: MonotonicTimestamp,
    },
    Active(ActiveState),
    AwaitingIntentResult(PendingIntent),
    AwaitingReleaseResult(PendingRelease),
    Released,
    Fallback,
}

impl State {
    const fn phase(self) -> SessionPhase {
        match self {
            Self::Ready => SessionPhase::Ready,
            Self::AwaitingIdentity { .. } => SessionPhase::AwaitingIdentity,
            Self::IdentityVerified { .. } => SessionPhase::IdentityVerified,
            Self::AwaitingAcquire { .. } => SessionPhase::AwaitingAcquire,
            Self::Active(_) => SessionPhase::Active,
            Self::AwaitingIntentResult(_) => SessionPhase::AwaitingIntentResult,
            Self::AwaitingReleaseResult(_) => SessionPhase::AwaitingReleaseResult,
            Self::Released => SessionPhase::Released,
            Self::Fallback => SessionPhase::Fallback,
        }
    }

    const fn last_host_event(self) -> Option<MonotonicTimestamp> {
        match self {
            Self::AwaitingIdentity { sent_at } => Some(sent_at),
            Self::IdentityVerified {
                last_host_event, ..
            }
            | Self::AwaitingAcquire {
                last_host_event, ..
            } => Some(last_host_event),
            Self::Active(active) => Some(active.last_host_event),
            Self::AwaitingIntentResult(pending) => Some(pending.active.last_host_event),
            Self::AwaitingReleaseResult(pending) => Some(pending.active.last_host_event),
            Self::Ready | Self::Released | Self::Fallback => None,
        }
    }

    const fn trusted_boot(self) -> Option<DeviceBootId> {
        match self {
            Self::IdentityVerified { report, .. } | Self::AwaitingAcquire { report, .. } => {
                Some(report.boot_id)
            }
            Self::Active(active) => Some(active.binding.boot_id),
            Self::AwaitingIntentResult(pending) => Some(pending.active.binding.boot_id),
            Self::AwaitingReleaseResult(pending) => Some(pending.active.binding.boot_id),
            Self::Ready | Self::AwaitingIdentity { .. } | Self::Released | Self::Fallback => None,
        }
    }
}

/// Host ownership state for exactly one KEP2 eye controller.
///
/// Methods return protocol messages for the caller to transport. On a fault,
/// the state always enters [`SessionPhase::Fallback`]. The attached optional
/// release is best-effort cleanup for a still-trusted boot/epoch; the caller is
/// responsible for transmitting it and must not claim that fallback was
/// physically rendered.
pub struct EyeSession {
    plan: EyeSessionPlan,
    state: State,
}

impl EyeSession {
    pub const fn new(plan: EyeSessionPlan) -> Self {
        Self {
            plan,
            state: State::Ready,
        }
    }

    pub const fn phase(&self) -> SessionPhase {
        self.state.phase()
    }

    pub fn begin_identity(&mut self, now: MonotonicTimestamp) -> Result<Message, EyeSessionFault> {
        if !matches!(self.state, State::Ready) {
            return Err(self.fail(
                EyeSessionFaultKind::UnexpectedPhase {
                    operation: "begin identity",
                    actual: self.phase(),
                },
                ReleaseReason::Fault,
                true,
            ));
        }
        self.state = State::AwaitingIdentity { sent_at: now };
        Ok(Message::IdentityQuery {
            nonce: self.plan.identity_nonce.protocol_nonce(),
        })
    }

    pub fn begin_acquire(&mut self, now: MonotonicTimestamp) -> Result<Message, EyeSessionFault> {
        let State::IdentityVerified {
            report,
            last_host_event,
        } = self.state
        else {
            return Err(self.fail(
                EyeSessionFaultKind::UnexpectedPhase {
                    operation: "begin acquire",
                    actual: self.phase(),
                },
                ReleaseReason::Fault,
                true,
            ));
        };
        self.ensure_host_clock(last_host_event, now)?;
        self.state = State::AwaitingAcquire {
            report,
            last_host_event: now,
        };
        Ok(Message::AcquireControl(AcquireControl {
            expected_boot_id: report.boot_id,
            requested_epoch: self.plan.control_epoch,
            nonce: self.plan.acquire_nonce.protocol_nonce(),
        }))
    }

    pub fn submit_intent(
        &mut self,
        prepared: PreparedEyeIntent,
        lease: IntentLeaseMs,
        now: MonotonicTimestamp,
    ) -> Result<Message, EyeSessionFault> {
        let State::Active(mut active) = self.state else {
            return Err(self.fail(
                EyeSessionFaultKind::UnexpectedPhase {
                    operation: "submit intent",
                    actual: self.phase(),
                },
                ReleaseReason::Fault,
                true,
            ));
        };
        self.ensure_host_clock(active.last_host_event, now)?;
        if now < prepared.generated_at() {
            return Err(self.fail(
                EyeSessionFaultKind::IntentSourceFromFuture {
                    generated_at_ns: prepared.generated_at().nanos_since_epoch(),
                    now_ns: now.nanos_since_epoch(),
                },
                ReleaseReason::PerceptionStale,
                true,
            ));
        }
        if !prepared.is_fresh_at(now) {
            return Err(self.fail(
                EyeSessionFaultKind::IntentSourceStale {
                    deadline_ns: prepared
                        .valid_until_exclusive()
                        .map(|deadline| deadline.timestamp().nanos_since_epoch()),
                    now_ns: now.nanos_since_epoch(),
                },
                ReleaseReason::PerceptionStale,
                true,
            ));
        }
        let sequence = active.next_sequence;
        let Some(successor) = sequence.checked_successor() else {
            return Err(self.fail(
                EyeSessionFaultKind::IntentSequenceExhausted {
                    sequence: sequence.get(),
                },
                ReleaseReason::Fault,
                true,
            ));
        };
        active.last_host_event = now;
        self.state = State::AwaitingIntentResult(PendingIntent {
            active,
            sequence,
            successor,
            lease,
            prepared,
        });
        Ok(Message::ApplyIntent(ApplyIntent {
            boot_id: active.binding.boot_id,
            control_epoch: active.binding.control_epoch,
            sequence,
            lease,
            intent: prepared.intent(),
        }))
    }

    pub fn begin_release(
        &mut self,
        reason: ReleaseReason,
        now: MonotonicTimestamp,
    ) -> Result<Message, EyeSessionFault> {
        let State::Active(mut active) = self.state else {
            return Err(self.fail(
                EyeSessionFaultKind::UnexpectedPhase {
                    operation: "begin release",
                    actual: self.phase(),
                },
                ReleaseReason::Fault,
                true,
            ));
        };
        self.ensure_host_clock(active.last_host_event, now)?;
        active.last_host_event = now;
        let sequence = active.next_sequence;
        self.state = State::AwaitingReleaseResult(PendingRelease { active, sequence });
        Ok(Message::ReleaseControl(ReleaseControl {
            boot_id: active.binding.boot_id,
            control_epoch: active.binding.control_epoch,
            sequence,
            reason,
        }))
    }

    pub fn handle_inbound(
        &mut self,
        message: Message,
        now: MonotonicTimestamp,
    ) -> Result<SessionEvent, EyeSessionFault> {
        if let Some(previous) = self.state.last_host_event() {
            self.ensure_host_clock(previous, now)?;
        }
        match (self.state, message) {
            (State::AwaitingIdentity { .. }, Message::IdentityReport(report)) => {
                self.handle_identity_report(report, now)
            }
            (State::AwaitingAcquire { report, .. }, Message::AcquireResult(result)) => {
                self.handle_acquire_result(report, result, now)
            }
            (State::AwaitingIntentResult(pending), Message::IntentResult(result)) => {
                self.handle_intent_result(pending, result, now)
            }
            (State::AwaitingReleaseResult(pending), Message::IntentResult(result)) => {
                self.handle_release_result(pending, result, now)
            }
            (state, Message::IdentityReport(report)) => {
                let (kind, allow_release) = self.classify_unexpected_identity(state, report);
                Err(self.fail(kind, ReleaseReason::Fault, allow_release))
            }
            (state, message) => Err(self.fail(
                EyeSessionFaultKind::UnexpectedInbound {
                    phase: state.phase(),
                    actual: message.into(),
                },
                ReleaseReason::Fault,
                true,
            )),
        }
    }

    /// Enter fallback after a transport failure and return any best-effort
    /// release that remains meaningful for the trusted session binding.
    pub fn transport_fault(&mut self) -> EyeSessionFault {
        self.fail(
            EyeSessionFaultKind::TransportFault,
            ReleaseReason::Fault,
            true,
        )
    }

    fn handle_identity_report(
        &mut self,
        report: IdentityReport,
        now: MonotonicTimestamp,
    ) -> Result<SessionEvent, EyeSessionFault> {
        let expected = self.plan.expected_identity;
        if report.nonce != self.plan.identity_nonce.protocol_nonce() {
            return Err(self.fail(
                EyeSessionFaultKind::IdentityNonceMismatch {
                    expected: self.plan.identity_nonce.get(),
                    actual: report.nonce.get(),
                },
                ReleaseReason::Fault,
                false,
            ));
        }
        if report.device_uid != expected.device_uid {
            return Err(self.fail(
                EyeSessionFaultKind::DeviceUidMismatch {
                    expected: expected.device_uid,
                    actual: report.device_uid,
                },
                ReleaseReason::Fault,
                false,
            ));
        }
        if report.firmware_build_id != expected.firmware_build_id {
            return Err(self.fail(
                EyeSessionFaultKind::FirmwareBuildMismatch {
                    expected: expected.firmware_build_id,
                    actual: report.firmware_build_id,
                },
                ReleaseReason::Fault,
                false,
            ));
        }
        if report.capabilities != expected.capabilities {
            return Err(self.fail(
                EyeSessionFaultKind::CapabilityMismatch {
                    expected: expected.capabilities.bits(),
                    actual: report.capabilities.bits(),
                },
                ReleaseReason::Fault,
                false,
            ));
        }
        self.state = State::IdentityVerified {
            report,
            last_host_event: now,
        };
        Ok(SessionEvent::IdentityVerified(report))
    }

    fn handle_acquire_result(
        &mut self,
        identity: IdentityReport,
        result: AcquireResult,
        now: MonotonicTimestamp,
    ) -> Result<SessionEvent, EyeSessionFault> {
        if result.boot_id != identity.boot_id {
            return Err(self.fail(
                EyeSessionFaultKind::DeviceRebooted {
                    expected_boot_id: identity.boot_id,
                    actual_boot_id: result.boot_id,
                },
                ReleaseReason::Fault,
                false,
            ));
        }
        if result.control_epoch != self.plan.control_epoch {
            return Err(self.fail(
                EyeSessionFaultKind::AcquireEpochMismatch {
                    expected: self.plan.control_epoch,
                    actual: result.control_epoch,
                },
                ReleaseReason::Fault,
                false,
            ));
        }
        if result.nonce != self.plan.acquire_nonce.protocol_nonce() {
            return Err(self.fail(
                EyeSessionFaultKind::AcquireNonceMismatch {
                    expected: self.plan.acquire_nonce.get(),
                    actual: result.nonce.get(),
                },
                ReleaseReason::Fault,
                false,
            ));
        }
        if result.result != AcquireResultCode::Granted {
            return Err(self.fail(
                EyeSessionFaultKind::AcquireRejected {
                    result: result.result,
                },
                ReleaseReason::Fault,
                false,
            ));
        }
        if result.device_uptime < identity.device_uptime {
            return Err(self.fail(
                EyeSessionFaultKind::DeviceClockRegressed {
                    previous_ms: identity.device_uptime.millis_since_boot(),
                    actual_ms: result.device_uptime.millis_since_boot(),
                },
                ReleaseReason::Fault,
                false,
            ));
        }
        let binding = ControlBinding {
            boot_id: result.boot_id,
            control_epoch: result.control_epoch,
        };
        self.state = State::Active(ActiveState {
            binding,
            next_sequence: IntentSequence::FIRST,
            last_device_event: result.device_uptime,
            last_host_event: now,
            last_admitted_sequence: None,
            last_rendered_frame_sequence: None,
        });
        Ok(SessionEvent::ControlAcquired(binding))
    }

    fn handle_intent_result(
        &mut self,
        pending: PendingIntent,
        result: IntentResult,
        now: MonotonicTimestamp,
    ) -> Result<SessionEvent, EyeSessionFault> {
        if !pending.prepared.is_fresh_at(now) {
            return Err(self.fail(
                EyeSessionFaultKind::IntentSourceStale {
                    deadline_ns: pending
                        .prepared
                        .valid_until_exclusive()
                        .map(|deadline| deadline.timestamp().nanos_since_epoch()),
                    now_ns: now.nanos_since_epoch(),
                },
                ReleaseReason::PerceptionStale,
                true,
            ));
        }
        let Some(elapsed_ns) = now
            .nanos_since_epoch()
            .checked_sub(pending.active.last_host_event.nanos_since_epoch())
        else {
            return Err(self.fail(
                EyeSessionFaultKind::HostClockRegressed {
                    previous_ns: pending.active.last_host_event.nanos_since_epoch(),
                    actual_ns: now.nanos_since_epoch(),
                },
                ReleaseReason::Fault,
                true,
            ));
        };
        let lease_ns = u64::from(pending.lease.get()) * 1_000_000;
        if elapsed_ns >= lease_ns {
            return Err(self.fail(
                EyeSessionFaultKind::AcknowledgementExpired {
                    sent_at_ns: pending.active.last_host_event.nanos_since_epoch(),
                    now_ns: now.nanos_since_epoch(),
                    lease_ms: pending.lease.get(),
                },
                ReleaseReason::Fault,
                true,
            ));
        }
        self.validate_result_binding(pending.active, pending.sequence, result)?;
        if result.result() == IntentResultCode::DuplicateCached {
            return Err(self.fail(
                EyeSessionFaultKind::DuplicateIntentAcknowledgement {
                    sequence: result.sequence().get(),
                },
                ReleaseReason::Fault,
                true,
            ));
        }
        if result.result() != IntentResultCode::AppliedNew {
            return Err(self.fail(
                EyeSessionFaultKind::IntentRejected {
                    sequence: result.sequence().get(),
                    result: result.result(),
                },
                ReleaseReason::Fault,
                result.result() != IntentResultCode::RejectedSession,
            ));
        }
        if result.admitted_lease() != Some(pending.lease) {
            return Err(self.fail(
                EyeSessionFaultKind::LeaseMismatch {
                    expected_ms: pending.lease.get(),
                    actual_ms: result.device_interval_ms(),
                },
                ReleaseReason::Fault,
                true,
            ));
        }
        self.validate_device_progress(pending.active, result)?;

        let admission = FirmwareAdmission {
            binding: pending.active.binding,
            sequence: pending.sequence,
            lease: pending.lease,
            applied_at: result.applied_at(),
            expires_at: result.expires_at(),
            rendered_frame_sequence: result.rendered_frame_sequence(),
        };
        self.state = State::Active(ActiveState {
            binding: pending.active.binding,
            next_sequence: pending.successor,
            last_device_event: result.applied_at(),
            last_host_event: now,
            last_admitted_sequence: Some(pending.sequence),
            last_rendered_frame_sequence: Some(result.rendered_frame_sequence()),
        });
        Ok(SessionEvent::IntentAdmitted(admission))
    }

    fn handle_release_result(
        &mut self,
        pending: PendingRelease,
        result: IntentResult,
        _now: MonotonicTimestamp,
    ) -> Result<SessionEvent, EyeSessionFault> {
        self.validate_result_binding(pending.active, pending.sequence, result)?;
        if result.result() != IntentResultCode::Released {
            return Err(self.fail(
                EyeSessionFaultKind::ReleaseRejected {
                    sequence: result.sequence().get(),
                    result: result.result(),
                },
                ReleaseReason::Fault,
                false,
            ));
        }
        self.validate_device_progress(pending.active, result)?;
        self.state = State::Released;
        Ok(SessionEvent::Released(pending.active.binding))
    }

    fn validate_result_binding(
        &mut self,
        active: ActiveState,
        expected_sequence: IntentSequence,
        result: IntentResult,
    ) -> Result<(), EyeSessionFault> {
        if result.boot_id() != active.binding.boot_id {
            return Err(self.fail(
                EyeSessionFaultKind::DeviceRebooted {
                    expected_boot_id: active.binding.boot_id,
                    actual_boot_id: result.boot_id(),
                },
                ReleaseReason::Fault,
                false,
            ));
        }
        if result.control_epoch() != active.binding.control_epoch {
            return Err(self.fail(
                EyeSessionFaultKind::ResultEpochMismatch {
                    expected: active.binding.control_epoch,
                    actual: result.control_epoch(),
                },
                ReleaseReason::Fault,
                false,
            ));
        }
        if result.sequence() != expected_sequence {
            let kind = if active.last_admitted_sequence == Some(result.sequence()) {
                EyeSessionFaultKind::DuplicateResult {
                    sequence: result.sequence().get(),
                }
            } else {
                EyeSessionFaultKind::OutOfOrderResult {
                    expected: expected_sequence.get(),
                    actual: result.sequence().get(),
                }
            };
            return Err(self.fail(kind, ReleaseReason::Fault, true));
        }
        Ok(())
    }

    fn validate_device_progress(
        &mut self,
        active: ActiveState,
        result: IntentResult,
    ) -> Result<(), EyeSessionFault> {
        if result.applied_at() < active.last_device_event {
            return Err(self.fail(
                EyeSessionFaultKind::DeviceClockRegressed {
                    previous_ms: active.last_device_event.millis_since_boot(),
                    actual_ms: result.applied_at().millis_since_boot(),
                },
                ReleaseReason::Fault,
                true,
            ));
        }
        if let Some(previous) = active.last_rendered_frame_sequence
            && !rendered_sequence_is_not_older(previous, result.rendered_frame_sequence())
        {
            return Err(self.fail(
                EyeSessionFaultKind::RenderedFrameSequenceRegressed {
                    previous: previous.get(),
                    actual: result.rendered_frame_sequence().get(),
                },
                ReleaseReason::Fault,
                true,
            ));
        }
        Ok(())
    }

    fn classify_unexpected_identity(
        &self,
        state: State,
        report: IdentityReport,
    ) -> (EyeSessionFaultKind, bool) {
        let expected = self.plan.expected_identity;
        if report.nonce != self.plan.identity_nonce.protocol_nonce() {
            return (
                EyeSessionFaultKind::IdentityNonceMismatch {
                    expected: self.plan.identity_nonce.get(),
                    actual: report.nonce.get(),
                },
                false,
            );
        }
        if report.device_uid != expected.device_uid {
            return (
                EyeSessionFaultKind::DeviceUidMismatch {
                    expected: expected.device_uid,
                    actual: report.device_uid,
                },
                false,
            );
        }
        if report.firmware_build_id != expected.firmware_build_id {
            return (
                EyeSessionFaultKind::FirmwareBuildMismatch {
                    expected: expected.firmware_build_id,
                    actual: report.firmware_build_id,
                },
                false,
            );
        }
        if report.capabilities != expected.capabilities {
            return (
                EyeSessionFaultKind::CapabilityMismatch {
                    expected: expected.capabilities.bits(),
                    actual: report.capabilities.bits(),
                },
                false,
            );
        }
        match state.trusted_boot() {
            Some(expected_boot_id) if expected_boot_id != report.boot_id => (
                EyeSessionFaultKind::DeviceRebooted {
                    expected_boot_id,
                    actual_boot_id: report.boot_id,
                },
                false,
            ),
            _ => (EyeSessionFaultKind::DuplicateIdentityReport, true),
        }
    }

    fn ensure_host_clock(
        &mut self,
        previous: MonotonicTimestamp,
        now: MonotonicTimestamp,
    ) -> Result<(), EyeSessionFault> {
        if now < previous {
            Err(self.fail(
                EyeSessionFaultKind::HostClockRegressed {
                    previous_ns: previous.nanos_since_epoch(),
                    actual_ns: now.nanos_since_epoch(),
                },
                ReleaseReason::Fault,
                true,
            ))
        } else {
            Ok(())
        }
    }

    fn fail(
        &mut self,
        kind: EyeSessionFaultKind,
        reason: ReleaseReason,
        allow_release: bool,
    ) -> EyeSessionFault {
        let release = allow_release
            .then(|| self.release_for_fault(reason))
            .flatten();
        self.state = State::Fallback;
        EyeSessionFault { kind, release }
    }

    fn release_for_fault(&self, reason: ReleaseReason) -> Option<ReleaseControl> {
        let (binding, sequence) = match self.state {
            State::AwaitingAcquire { report, .. } => (
                ControlBinding {
                    boot_id: report.boot_id,
                    control_epoch: self.plan.control_epoch,
                },
                IntentSequence::FIRST,
            ),
            State::Active(active) => (active.binding, active.next_sequence),
            State::AwaitingIntentResult(pending) => (
                pending.active.binding,
                pending.sequence.checked_successor()?,
            ),
            State::Ready
            | State::AwaitingIdentity { .. }
            | State::IdentityVerified { .. }
            | State::AwaitingReleaseResult(_)
            | State::Released
            | State::Fallback => return None,
        };
        Some(ReleaseControl {
            boot_id: binding.boot_id,
            control_epoch: binding.control_epoch,
            sequence,
            reason,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EyeSessionFault {
    kind: EyeSessionFaultKind,
    release: Option<ReleaseControl>,
}

impl EyeSessionFault {
    pub const fn kind(self) -> EyeSessionFaultKind {
        self.kind
    }

    pub const fn release(self) -> Option<ReleaseControl> {
        self.release
    }

    pub const fn release_message(self) -> Option<Message> {
        match self.release {
            Some(release) => Some(Message::ReleaseControl(release)),
            None => None,
        }
    }
}

impl fmt::Display for EyeSessionFault {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "eye session entered fallback: {:?}", self.kind)
    }
}

impl core::error::Error for EyeSessionFault {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EyeSessionFaultKind {
    UnexpectedPhase {
        operation: &'static str,
        actual: SessionPhase,
    },
    UnexpectedInbound {
        phase: SessionPhase,
        actual: InboundMessageKind,
    },
    HostClockRegressed {
        previous_ns: u64,
        actual_ns: u64,
    },
    IdentityNonceMismatch {
        expected: u64,
        actual: u64,
    },
    DeviceUidMismatch {
        expected: DeviceUid,
        actual: DeviceUid,
    },
    FirmwareBuildMismatch {
        expected: FirmwareBuildId,
        actual: FirmwareBuildId,
    },
    CapabilityMismatch {
        expected: u32,
        actual: u32,
    },
    DuplicateIdentityReport,
    DeviceRebooted {
        expected_boot_id: DeviceBootId,
        actual_boot_id: DeviceBootId,
    },
    AcquireEpochMismatch {
        expected: ControlEpoch,
        actual: ControlEpoch,
    },
    AcquireNonceMismatch {
        expected: u64,
        actual: u64,
    },
    AcquireRejected {
        result: AcquireResultCode,
    },
    IntentSourceFromFuture {
        generated_at_ns: u64,
        now_ns: u64,
    },
    IntentSourceStale {
        deadline_ns: Option<u64>,
        now_ns: u64,
    },
    IntentSequenceExhausted {
        sequence: u32,
    },
    AcknowledgementExpired {
        sent_at_ns: u64,
        now_ns: u64,
        lease_ms: u16,
    },
    ResultEpochMismatch {
        expected: ControlEpoch,
        actual: ControlEpoch,
    },
    DuplicateResult {
        sequence: u32,
    },
    OutOfOrderResult {
        expected: u32,
        actual: u32,
    },
    DuplicateIntentAcknowledgement {
        sequence: u32,
    },
    IntentRejected {
        sequence: u32,
        result: IntentResultCode,
    },
    ReleaseRejected {
        sequence: u32,
        result: IntentResultCode,
    },
    LeaseMismatch {
        expected_ms: u16,
        actual_ms: u16,
    },
    DeviceClockRegressed {
        previous_ms: u64,
        actual_ms: u64,
    },
    RenderedFrameSequenceRegressed {
        previous: u32,
        actual: u32,
    },
    TransportFault,
}

/// RFC-1982-style half-range comparison for a wrapping 32-bit device counter.
/// Equality and a forward delta below half the serial space are current; the
/// exactly-half-range case is deliberately treated as ambiguous/old.
const fn rendered_sequence_is_not_older(
    previous: RenderedFrameSequence,
    actual: RenderedFrameSequence,
) -> bool {
    let delta = actual.get().wrapping_sub(previous.get());
    delta == 0 || delta < (1_u32 << 31)
}

#[cfg(test)]
mod tests {
    extern crate std;

    use kiko_expression_core::{
        ExpressionIntent, ExpressionKind, ExpressionPriority, FreshnessWindow, NonZeroDuration,
        PositiveUnitAmount, ReactionInputs, ReactionMixer, UnitAmount as CoreUnitAmount,
    };
    use kiko_eye_protocol::{MAX_ENCODED_FRAME_BYTES, decode, encode};

    use super::*;
    use crate::{EyeRenderStyle, adapt_reaction_output};

    fn timestamp(value: u64) -> MonotonicTimestamp {
        MonotonicTimestamp::from_nanos_since_epoch(value)
    }

    fn capabilities() -> Capabilities {
        Capabilities::try_from_bits(REQUIRED_EYE_CAPABILITIES).expect("capabilities")
    }

    fn uid() -> DeviceUid {
        DeviceUid::try_new([1; 16]).expect("uid")
    }

    fn build() -> FirmwareBuildId {
        FirmwareBuildId::try_new([2; 32]).expect("build")
    }

    fn boot(value: u64) -> DeviceBootId {
        DeviceBootId::try_new(value).expect("boot")
    }

    fn epoch() -> ControlEpoch {
        ControlEpoch::try_new(9).expect("epoch")
    }

    fn protocol_nonce(value: u64) -> HandshakeNonce {
        HandshakeNonce::try_new(value).expect("nonce")
    }

    const fn device_time(value: u64) -> DeviceTimestampMs {
        DeviceTimestampMs::from_millis_since_boot(value)
    }

    const fn rendered_sequence(value: u32) -> RenderedFrameSequence {
        RenderedFrameSequence::new(value)
    }

    fn intent_result(
        binding: ControlBinding,
        sequence: IntentSequence,
        result: IntentResultCode,
        applied_at_ms: u64,
        expires_at_ms: u64,
        rendered_frame_sequence: u32,
    ) -> IntentResult {
        IntentResult::try_new(
            binding.boot_id(),
            binding.control_epoch(),
            sequence,
            result,
            device_time(applied_at_ms),
            device_time(expires_at_ms),
            rendered_sequence(rendered_frame_sequence),
        )
        .expect("valid protocol result")
    }

    fn plan() -> EyeSessionPlan {
        EyeSessionPlan::try_new(
            ExpectedEyeIdentity::new(uid(), build(), capabilities()),
            SessionNonce::try_new(11).expect("nonce"),
            SessionNonce::try_new(12).expect("nonce"),
            epoch(),
        )
        .expect("plan")
    }

    fn identity() -> IdentityReport {
        IdentityReport {
            nonce: protocol_nonce(11),
            device_uid: uid(),
            firmware_build_id: build(),
            boot_id: boot(7),
            device_uptime: device_time(100),
            capabilities: capabilities(),
        }
    }

    fn prepared(now: u64) -> PreparedEyeIntent {
        let output = ReactionMixer::default().mix(timestamp(now), ReactionInputs::empty());
        adapt_reaction_output(
            output,
            ExpressionKind::Neutral,
            EyeRenderStyle::new(
                CoreUnitAmount::try_from_basis_points(5_000).expect("brightness"),
                [3, 4, 5],
                false,
            ),
            timestamp(now),
        )
        .expect("prepared")
    }

    fn expiring_prepared(observed_at: u64, ttl: u64) -> PreparedEyeIntent {
        let freshness = FreshnessWindow::from_ttl(
            timestamp(observed_at),
            NonZeroDuration::try_from_nanos(ttl).expect("ttl"),
        )
        .expect("freshness");
        let intent = ExpressionIntent::new(
            ExpressionKind::Attentive,
            PositiveUnitAmount::ONE,
            ExpressionPriority::Normal,
            None,
            freshness,
        );
        let output = ReactionMixer::default().mix(
            timestamp(observed_at),
            ReactionInputs {
                rgb: None,
                people: &[],
                scene: None,
                intents: &[intent],
            },
        );
        adapt_reaction_output(
            output,
            ExpressionKind::Attentive,
            EyeRenderStyle::new(
                CoreUnitAmount::try_from_basis_points(5_000).expect("brightness"),
                [3, 4, 5],
                false,
            ),
            timestamp(observed_at),
        )
        .expect("prepared")
    }

    fn acquire_active(session: &mut EyeSession) -> ControlBinding {
        assert_eq!(
            session
                .begin_identity(timestamp(1))
                .expect("identity query"),
            Message::IdentityQuery {
                nonce: protocol_nonce(11)
            }
        );
        assert_eq!(
            session
                .handle_inbound(Message::IdentityReport(identity()), timestamp(2))
                .expect("identity report"),
            SessionEvent::IdentityVerified(identity())
        );
        assert_eq!(
            session.begin_acquire(timestamp(3)).expect("acquire"),
            Message::AcquireControl(AcquireControl {
                expected_boot_id: boot(7),
                requested_epoch: epoch(),
                nonce: protocol_nonce(12),
            })
        );
        let event = session
            .handle_inbound(
                Message::AcquireResult(AcquireResult {
                    boot_id: boot(7),
                    control_epoch: epoch(),
                    nonce: protocol_nonce(12),
                    result: AcquireResultCode::Granted,
                    device_uptime: device_time(110),
                }),
                timestamp(4),
            )
            .expect("acquire result");
        let SessionEvent::ControlAcquired(binding) = event else {
            panic!("expected binding");
        };
        binding
    }

    #[test]
    fn plan_requires_distinct_nonces_and_runtime_capabilities() {
        assert_eq!(SessionNonce::try_new(0), Err(SessionPlanError::ZeroNonce));
        let nonce = SessionNonce::try_new(1).expect("nonce");
        assert!(matches!(
            EyeSessionPlan::try_new(
                ExpectedEyeIdentity::new(uid(), build(), capabilities()),
                nonce,
                nonce,
                epoch()
            ),
            Err(SessionPlanError::ReusedNonce { .. })
        ));
        let missing = Capabilities::try_from_bits(Capabilities::GAZE).expect("known caps");
        assert!(matches!(
            EyeSessionPlan::try_new(
                ExpectedEyeIdentity::new(uid(), build(), missing),
                SessionNonce::try_new(1).expect("nonce"),
                SessionNonce::try_new(2).expect("nonce"),
                epoch()
            ),
            Err(SessionPlanError::MissingRequiredCapabilities { .. })
        ));
    }

    #[test]
    fn exact_handshake_intent_admission_and_release_are_sequence_bound() {
        let mut session = EyeSession::new(plan());
        let binding = acquire_active(&mut session);
        let lease = IntentLeaseMs::try_new(200).expect("lease");
        let apply = session
            .submit_intent(prepared(5), lease, timestamp(5))
            .expect("apply");
        let Message::ApplyIntent(apply) = apply else {
            panic!("apply message");
        };
        assert_eq!(apply.boot_id, binding.boot_id());
        assert_eq!(apply.control_epoch, binding.control_epoch());
        assert_eq!(apply.sequence, IntentSequence::FIRST);
        assert_eq!(apply.lease, lease);

        let result = intent_result(
            binding,
            IntentSequence::FIRST,
            IntentResultCode::AppliedNew,
            120,
            320,
            8,
        );
        let SessionEvent::IntentAdmitted(admission) = session
            .handle_inbound(Message::IntentResult(result), timestamp(6))
            .expect("admission")
        else {
            panic!("admission event");
        };
        assert_eq!(admission.sequence(), IntentSequence::FIRST);
        assert_eq!(admission.lease(), lease);
        assert_eq!(admission.rendered_frame_sequence(), rendered_sequence(8));

        let release = session
            .begin_release(ReleaseReason::HostShutdown, timestamp(7))
            .expect("release");
        let Message::ReleaseControl(release) = release else {
            panic!("release message");
        };
        assert_eq!(release.sequence, IntentSequence::new(1));
        let result = intent_result(
            binding,
            release.sequence,
            IntentResultCode::Released,
            330,
            330,
            9,
        );
        assert_eq!(
            session
                .handle_inbound(Message::IntentResult(result), timestamp(8))
                .expect("released"),
            SessionEvent::Released(binding)
        );
        assert_eq!(session.phase(), SessionPhase::Released);
    }

    #[test]
    fn identity_uid_build_capability_and_nonce_must_match_exactly() {
        let mutations = [
            IdentityReport {
                nonce: protocol_nonce(99),
                ..identity()
            },
            IdentityReport {
                device_uid: DeviceUid::try_new([9; 16]).expect("uid"),
                ..identity()
            },
            IdentityReport {
                firmware_build_id: FirmwareBuildId::try_new([9; 32]).expect("build"),
                ..identity()
            },
            IdentityReport {
                capabilities: Capabilities::try_from_bits(
                    REQUIRED_EYE_CAPABILITIES & !Capabilities::BLINK,
                )
                .expect("caps"),
                ..identity()
            },
        ];
        for report in mutations {
            let mut session = EyeSession::new(plan());
            session.begin_identity(timestamp(1)).expect("query");
            let fault = session
                .handle_inbound(Message::IdentityReport(report), timestamp(2))
                .expect_err("identity mismatch");
            assert_eq!(session.phase(), SessionPhase::Fallback);
            assert_eq!(fault.release(), None);
        }
    }

    #[test]
    fn duplicate_out_of_order_and_reboot_results_fail_closed() {
        let lease = IntentLeaseMs::try_new(100).expect("lease");

        let mut session = EyeSession::new(plan());
        let binding = acquire_active(&mut session);
        session
            .submit_intent(prepared(5), lease, timestamp(5))
            .expect("apply");
        let out_of_order = intent_result(
            binding,
            IntentSequence::new(1),
            IntentResultCode::AppliedNew,
            120,
            220,
            1,
        );
        let fault = session
            .handle_inbound(Message::IntentResult(out_of_order), timestamp(6))
            .expect_err("sequence mismatch");
        assert!(matches!(
            fault.kind(),
            EyeSessionFaultKind::OutOfOrderResult { .. }
        ));
        assert_eq!(
            fault.release().expect("cleanup").sequence,
            IntentSequence::new(1)
        );

        let mut session = EyeSession::new(plan());
        let binding = acquire_active(&mut session);
        session
            .submit_intent(prepared(5), lease, timestamp(5))
            .expect("apply");
        let duplicate = intent_result(
            binding,
            IntentSequence::FIRST,
            IntentResultCode::DuplicateCached,
            120,
            220,
            1,
        );
        let fault = session
            .handle_inbound(Message::IntentResult(duplicate), timestamp(6))
            .expect_err("duplicate cached");
        assert!(matches!(
            fault.kind(),
            EyeSessionFaultKind::DuplicateIntentAcknowledgement { .. }
        ));

        let mut session = EyeSession::new(plan());
        let binding = acquire_active(&mut session);
        session
            .submit_intent(prepared(5), lease, timestamp(5))
            .expect("apply");
        let reboot = IntentResult::try_new(
            boot(8),
            binding.control_epoch(),
            IntentSequence::FIRST,
            IntentResultCode::AppliedNew,
            device_time(1),
            device_time(101),
            rendered_sequence(1),
        )
        .expect("result");
        let fault = session
            .handle_inbound(Message::IntentResult(reboot), timestamp(6))
            .expect_err("reboot");
        assert!(matches!(
            fault.kind(),
            EyeSessionFaultKind::DeviceRebooted { .. }
        ));
        assert_eq!(fault.release(), None);
    }

    #[test]
    fn an_old_ack_after_a_new_submission_is_a_duplicate_failure() {
        let lease = IntentLeaseMs::try_new(100).expect("lease");
        let mut session = EyeSession::new(plan());
        let binding = acquire_active(&mut session);
        session
            .submit_intent(prepared(5), lease, timestamp(5))
            .expect("first apply");
        let first = intent_result(
            binding,
            IntentSequence::FIRST,
            IntentResultCode::AppliedNew,
            120,
            220,
            8,
        );
        session
            .handle_inbound(Message::IntentResult(first), timestamp(6))
            .expect("first ack");
        session
            .submit_intent(prepared(7), lease, timestamp(7))
            .expect("second apply");
        let duplicate = intent_result(
            binding,
            IntentSequence::FIRST,
            IntentResultCode::AppliedNew,
            130,
            230,
            9,
        );
        let fault = session
            .handle_inbound(Message::IntentResult(duplicate), timestamp(8))
            .expect_err("duplicate must fail");
        assert_eq!(
            fault.kind(),
            EyeSessionFaultKind::DuplicateResult { sequence: 0 }
        );
    }

    #[test]
    fn prepared_reaction_freshness_is_rechecked_when_dispatched() {
        let mut session = EyeSession::new(plan());
        acquire_active(&mut session);
        let fault = session
            .submit_intent(
                expiring_prepared(5, 5),
                IntentLeaseMs::try_new(100).expect("lease"),
                timestamp(10),
            )
            .expect_err("exclusive deadline is stale");
        assert!(matches!(
            fault.kind(),
            EyeSessionFaultKind::IntentSourceStale {
                deadline_ns: Some(10),
                now_ns: 10
            }
        ));
        assert_eq!(
            fault.release().expect("cleanup").reason,
            ReleaseReason::PerceptionStale
        );
    }

    #[test]
    fn lease_and_host_clock_are_checked() {
        let lease = IntentLeaseMs::try_new(100).expect("lease");
        let mut session = EyeSession::new(plan());
        let binding = acquire_active(&mut session);
        session
            .submit_intent(prepared(5), lease, timestamp(5))
            .expect("apply");
        let wrong_lease = intent_result(
            binding,
            IntentSequence::FIRST,
            IntentResultCode::AppliedNew,
            120,
            140,
            1,
        );
        let fault = session
            .handle_inbound(Message::IntentResult(wrong_lease), timestamp(6))
            .expect_err("lease mismatch");
        assert!(matches!(
            fault.kind(),
            EyeSessionFaultKind::LeaseMismatch { .. }
        ));

        let mut session = EyeSession::new(plan());
        acquire_active(&mut session);
        let fault = session
            .submit_intent(prepared(5), lease, timestamp(3))
            .expect_err("host clock regression");
        assert!(matches!(
            fault.kind(),
            EyeSessionFaultKind::HostClockRegressed { .. }
        ));
        assert_eq!(session.phase(), SessionPhase::Fallback);
    }

    #[test]
    fn invalid_device_intervals_cannot_cross_the_typed_session_boundary() {
        let binding = ControlBinding {
            boot_id: boot(7),
            control_epoch: epoch(),
        };
        assert!(
            IntentResult::try_new(
                binding.boot_id(),
                binding.control_epoch(),
                IntentSequence::FIRST,
                IntentResultCode::AppliedNew,
                device_time(200),
                device_time(199),
                rendered_sequence(1),
            )
            .is_err()
        );
        assert!(
            IntentResult::try_new(
                binding.boot_id(),
                binding.control_epoch(),
                IntentSequence::FIRST,
                IntentResultCode::Released,
                device_time(120),
                device_time(121),
                rendered_sequence(1),
            )
            .is_err()
        );
    }

    #[test]
    fn device_interval_at_u64_upper_boundary_is_admitted_without_overflow() {
        let lease = IntentLeaseMs::try_new(200).expect("lease");
        let mut session = EyeSession::new(plan());
        let binding = acquire_active(&mut session);
        session
            .submit_intent(prepared(5), lease, timestamp(5))
            .expect("apply");
        let boundary = intent_result(
            binding,
            IntentSequence::FIRST,
            IntentResultCode::AppliedNew,
            u64::MAX - 200,
            u64::MAX,
            1,
        );
        let SessionEvent::IntentAdmitted(admission) = session
            .handle_inbound(Message::IntentResult(boundary), timestamp(6))
            .expect("upper boundary is valid")
        else {
            panic!("expected firmware admission");
        };
        assert_eq!(admission.applied_at(), device_time(u64::MAX - 200));
        assert_eq!(admission.expires_at(), device_time(u64::MAX));
        assert_eq!(admission.lease(), lease);
    }

    #[test]
    fn acknowledgement_is_deadline_exclusive_on_the_host_lease_duration() {
        let lease = IntentLeaseMs::try_new(20).expect("lease");
        let mut session = EyeSession::new(plan());
        let binding = acquire_active(&mut session);
        session
            .submit_intent(prepared(5), lease, timestamp(5))
            .expect("apply");
        let result = intent_result(
            binding,
            IntentSequence::FIRST,
            IntentResultCode::AppliedNew,
            120,
            140,
            1,
        );
        let deadline = 5 + 20_000_000;
        let fault = session
            .handle_inbound(Message::IntentResult(result), timestamp(deadline))
            .expect_err("ack at the host lease deadline is expired");
        assert_eq!(
            fault.kind(),
            EyeSessionFaultKind::AcknowledgementExpired {
                sent_at_ns: 5,
                now_ns: deadline,
                lease_ms: 20,
            }
        );
    }

    #[test]
    fn device_and_renderer_progress_cannot_regress() {
        let lease = IntentLeaseMs::try_new(100).expect("lease");
        let mut session = EyeSession::new(plan());
        let binding = acquire_active(&mut session);
        session
            .submit_intent(prepared(5), lease, timestamp(5))
            .expect("first apply");
        let first = intent_result(
            binding,
            IntentSequence::FIRST,
            IntentResultCode::AppliedNew,
            120,
            220,
            8,
        );
        session
            .handle_inbound(Message::IntentResult(first), timestamp(6))
            .expect("first ack");
        session
            .submit_intent(prepared(7), lease, timestamp(7))
            .expect("second apply");
        let clock_regression = intent_result(
            binding,
            IntentSequence::new(1),
            IntentResultCode::AppliedNew,
            119,
            219,
            9,
        );
        let fault = session
            .handle_inbound(Message::IntentResult(clock_regression), timestamp(8))
            .expect_err("device clock regression");
        assert!(matches!(
            fault.kind(),
            EyeSessionFaultKind::DeviceClockRegressed { .. }
        ));

        let mut session = EyeSession::new(plan());
        let binding = acquire_active(&mut session);
        session
            .submit_intent(prepared(5), lease, timestamp(5))
            .expect("first apply");
        session
            .handle_inbound(Message::IntentResult(first), timestamp(6))
            .expect("first ack");
        session
            .submit_intent(prepared(7), lease, timestamp(7))
            .expect("second apply");
        let renderer_regression = intent_result(
            binding,
            IntentSequence::new(1),
            IntentResultCode::AppliedNew,
            130,
            230,
            7,
        );
        let fault = session
            .handle_inbound(Message::IntentResult(renderer_regression), timestamp(8))
            .expect_err("renderer regression");
        assert!(matches!(
            fault.kind(),
            EyeSessionFaultKind::RenderedFrameSequenceRegressed { .. }
        ));
    }

    #[test]
    fn renderer_sequence_comparison_accepts_wrap_and_rejects_ambiguous_age() {
        assert!(rendered_sequence_is_not_older(
            rendered_sequence(8),
            rendered_sequence(8)
        ));
        assert!(rendered_sequence_is_not_older(
            rendered_sequence(8),
            rendered_sequence(9)
        ));
        assert!(rendered_sequence_is_not_older(
            rendered_sequence(u32::MAX),
            rendered_sequence(0)
        ));
        assert!(!rendered_sequence_is_not_older(
            rendered_sequence(8),
            rendered_sequence(7)
        ));
        assert!(!rendered_sequence_is_not_older(
            rendered_sequence(0),
            rendered_sequence(1_u32 << 31)
        ));
    }

    #[test]
    fn transport_fault_returns_best_effort_release_then_falls_back() {
        let mut session = EyeSession::new(plan());
        let binding = acquire_active(&mut session);
        let fault = session.transport_fault();
        let release = fault.release().expect("active binding can be released");
        assert_eq!(release.boot_id, binding.boot_id());
        assert_eq!(release.control_epoch, binding.control_epoch());
        assert_eq!(release.sequence, IntentSequence::FIRST);
        assert_eq!(release.reason, ReleaseReason::Fault);
        assert_eq!(session.phase(), SessionPhase::Fallback);
    }

    #[test]
    fn corrupted_identity_records_are_rejected_before_the_session_boundary() {
        let message = Message::IdentityReport(identity());
        let mut encoded = [0_u8; MAX_ENCODED_FRAME_BYTES];
        let length = encode(message, &mut encoded).expect("encode");
        for index in 0..length - 1 {
            for bit in 0..8 {
                let mut corrupted = encoded;
                corrupted[index] ^= 1 << bit;
                assert_ne!(decode(&corrupted[..length - 1]), Ok(message));
            }
        }
    }
}
