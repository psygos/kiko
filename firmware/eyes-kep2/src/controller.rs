//! Device-side KEP2 ownership, ordering, and finite-lease state machine.

use kiko_eye_protocol::{
    AcquireControl, AcquireResult, AcquireResultCode, ApplyIntent, Capabilities, ControlEpoch,
    DeviceBootId, DeviceTimestampMs, DeviceUid, DomainError, EyeIntent, FirmwareBuildId,
    HandshakeNonce, IdentityReport, IntentResult, IntentResultCode, IntentSequence, Message,
    ReleaseControl, ReleaseReason, RenderedFrameSequence,
};

/// Maximum time acquisition may remain idle before the first intent.
///
/// KEP2 does not put a lease in `AcquireControl`. This local bound prevents an
/// acquired-but-silent connection from owning the renderer forever. It equals
/// the protocol's maximum intent lease and is measured on the device clock.
pub const ACQUIRE_TO_FIRST_INTENT_MS: u64 = 2_000;

/// Every behavior implemented by this image and required by the canonical host.
pub const SUPPORTED_CAPABILITIES_BITS: u32 = Capabilities::GAZE
    | Capabilities::LID
    | Capabilities::PUPIL
    | Capabilities::COLOR
    | Capabilities::BRIGHTNESS
    | Capabilities::BLINK
    | Capabilities::AUTONOMOUS_FALLBACK
    | Capabilities::APPLIED_REPORT;

/// Parsed, nonzero identities for one physical boot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FirmwareIdentity {
    device_uid: DeviceUid,
    firmware_build_id: FirmwareBuildId,
    boot_id: DeviceBootId,
    capabilities: Capabilities,
}

impl FirmwareIdentity {
    /// Parse raw hardware/build inputs exactly once at startup.
    pub fn try_new(
        device_uid: [u8; 16],
        firmware_build_id: [u8; 32],
        boot_id: u64,
    ) -> Result<Self, IdentityError> {
        Ok(Self {
            device_uid: DeviceUid::try_new(device_uid).map_err(IdentityError::ProtocolDomain)?,
            firmware_build_id: FirmwareBuildId::try_new(firmware_build_id)
                .map_err(IdentityError::ProtocolDomain)?,
            boot_id: DeviceBootId::try_new(boot_id).map_err(IdentityError::ProtocolDomain)?,
            capabilities: Capabilities::try_from_bits(SUPPORTED_CAPABILITIES_BITS)
                .map_err(IdentityError::ProtocolDomain)?,
        })
    }

    pub const fn device_uid(self) -> DeviceUid {
        self.device_uid
    }

    pub const fn firmware_build_id(self) -> FirmwareBuildId {
        self.firmware_build_id
    }

    pub const fn boot_id(self) -> DeviceBootId {
        self.boot_id
    }

    pub const fn capabilities(self) -> Capabilities {
        self.capabilities
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IdentityError {
    ProtocolDomain(DomainError),
}

/// Why the renderer is not consuming a host intent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FallbackCause {
    Boot,
    AwaitingFirstIntent,
    LeaseExpired,
    Released(ReleaseReason),
    Disconnected,
    MalformedFrame,
    ProtocolViolation,
    IntentSequenceExhausted,
    InternalFault,
}

/// The only renderer input exposed by the ownership state machine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputState {
    Autonomous {
        cause: FallbackCause,
    },
    Commanded {
        intent: EyeIntent,
        sequence: IntentSequence,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InboundKind {
    IdentityQuery,
    IdentityReport,
    AcquireControl,
    AcquireResult,
    ApplyIntent,
    ReleaseControl,
    IntentResult,
}

impl From<Message> for InboundKind {
    fn from(message: Message) -> Self {
        match message {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControllerError {
    ClockRegressed {
        previous_ms: u64,
        actual_ms: u64,
    },
    ReceiveTimestampFromFuture {
        received_at_ms: u64,
        handled_at_ms: u64,
    },
    DeadlineOverflow {
        now_ms: u64,
        duration_ms: u64,
    },
    ResultInvariant(DomainError),
    UnexpectedInbound(InboundKind),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CachedAdmission {
    command: ApplyIntent,
    result: IntentResult,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ActiveControl {
    epoch: ControlEpoch,
    acquire_nonce: HandshakeNonce,
    next_sequence: IntentSequence,
    deadline_exclusive: DeviceTimestampMs,
    cached_admission: Option<CachedAdmission>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ControlState {
    Idle,
    Active(ActiveControl),
}

/// Allocation-free, single-owner KEP2 device state machine.
pub struct Controller {
    identity: FirmwareIdentity,
    state: ControlState,
    output: OutputState,
    rendered_frame_sequence: u32,
    last_observed_time: DeviceTimestampMs,
    sticky_internal_fault: bool,
}

impl Controller {
    pub const fn new(identity: FirmwareIdentity) -> Self {
        Self {
            identity,
            state: ControlState::Idle,
            output: OutputState::Autonomous {
                cause: FallbackCause::Boot,
            },
            rendered_frame_sequence: 0,
            last_observed_time: DeviceTimestampMs::ZERO,
            sticky_internal_fault: false,
        }
    }

    pub const fn identity(&self) -> FirmwareIdentity {
        self.identity
    }

    pub const fn output(&self) -> OutputState {
        self.output
    }

    pub const fn is_owned(&self) -> bool {
        matches!(self.state, ControlState::Active(_))
    }

    pub const fn is_faulted(&self) -> bool {
        self.sticky_internal_fault
    }

    pub const fn expected_sequence(&self) -> Option<IntentSequence> {
        match self.state {
            ControlState::Idle => None,
            ControlState::Active(active) => Some(active.next_sequence),
        }
    }

    /// Advance lease state and return the renderer selection at `now`.
    pub fn output_at(&mut self, now: DeviceTimestampMs) -> Result<OutputState, ControllerError> {
        self.poll(now)?;
        Ok(self.output)
    }

    /// Advance finite ownership. Deadlines are exclusive: equality is expired.
    pub fn poll(&mut self, now: DeviceTimestampMs) -> Result<(), ControllerError> {
        self.observe_time(now)?;
        if let ControlState::Active(active) = self.state
            && now >= active.deadline_exclusive
        {
            self.enter_fallback(FallbackCause::LeaseExpired);
        }
        Ok(())
    }

    /// Process one already-decoded host request.
    ///
    /// `received_at` is when the complete delimited record arrived. `now` is
    /// when it is handled. Their separation makes queued-command expiry
    /// explicit instead of silently granting a fresh lease after a delay.
    pub fn handle_received(
        &mut self,
        message: Message,
        received_at: DeviceTimestampMs,
        now: DeviceTimestampMs,
    ) -> Result<Message, ControllerError> {
        self.poll(now)?;
        if received_at > now {
            let error = ControllerError::ReceiveTimestampFromFuture {
                received_at_ms: received_at.millis_since_boot(),
                handled_at_ms: now.millis_since_boot(),
            };
            self.enter_internal_fault();
            return Err(error);
        }

        match message {
            Message::IdentityQuery { nonce } => Ok(self.identity_report(nonce, now)),
            Message::AcquireControl(request) => self.acquire(request, now),
            Message::ApplyIntent(command) => self.apply(command, received_at, now),
            Message::ReleaseControl(release) => self.release(release, now),
            unexpected => {
                let kind = unexpected.into();
                self.enter_fallback(FallbackCause::ProtocolViolation);
                Err(ControllerError::UnexpectedInbound(kind))
            }
        }
    }

    /// A malformed/oversized record invalidates any live ownership.
    pub fn on_malformed_frame(&mut self, now: DeviceTimestampMs) -> Result<(), ControllerError> {
        self.poll(now)?;
        self.enter_fallback(FallbackCause::MalformedFrame);
        Ok(())
    }

    /// Link loss immediately relinquishes ownership; it never waits for TTL.
    pub fn on_disconnect(&mut self, now: DeviceTimestampMs) -> Result<(), ControllerError> {
        self.poll(now)?;
        self.enter_fallback(FallbackCause::Disconnected);
        Ok(())
    }

    /// Latch a device-side invariant or renderer failure until reboot.
    pub fn on_internal_fault(&mut self, now: DeviceTimestampMs) -> Result<(), ControllerError> {
        self.observe_time(now)?;
        self.enter_internal_fault();
        Ok(())
    }

    fn identity_report(&self, nonce: HandshakeNonce, now: DeviceTimestampMs) -> Message {
        Message::IdentityReport(IdentityReport {
            nonce,
            device_uid: self.identity.device_uid,
            firmware_build_id: self.identity.firmware_build_id,
            boot_id: self.identity.boot_id,
            device_uptime: now,
            capabilities: self.identity.capabilities,
        })
    }

    fn acquire(
        &mut self,
        request: AcquireControl,
        now: DeviceTimestampMs,
    ) -> Result<Message, ControllerError> {
        let result = if request.expected_boot_id != self.identity.boot_id {
            AcquireResultCode::IdentityMismatch
        } else if self.sticky_internal_fault {
            AcquireResultCode::Faulted
        } else {
            match self.state {
                ControlState::Idle => {
                    let deadline = self.checked_deadline(now, ACQUIRE_TO_FIRST_INTENT_MS)?;
                    self.state = ControlState::Active(ActiveControl {
                        epoch: request.requested_epoch,
                        acquire_nonce: request.nonce,
                        next_sequence: IntentSequence::FIRST,
                        deadline_exclusive: deadline,
                        cached_admission: None,
                    });
                    self.output = OutputState::Autonomous {
                        cause: FallbackCause::AwaitingFirstIntent,
                    };
                    AcquireResultCode::Granted
                }
                ControlState::Active(active)
                    if active.epoch == request.requested_epoch
                        && active.acquire_nonce == request.nonce =>
                {
                    // Exact retransmission is idempotent and never extends the
                    // existing first-intent/intent deadline.
                    AcquireResultCode::Granted
                }
                ControlState::Active(_) => AcquireResultCode::Busy,
            }
        };

        Ok(Message::AcquireResult(AcquireResult {
            boot_id: self.identity.boot_id,
            control_epoch: request.requested_epoch,
            nonce: request.nonce,
            result,
            device_uptime: now,
        }))
    }

    fn apply(
        &mut self,
        command: ApplyIntent,
        received_at: DeviceTimestampMs,
        now: DeviceTimestampMs,
    ) -> Result<Message, ControllerError> {
        if self.sticky_internal_fault {
            return self.instant_result_for_apply(command, IntentResultCode::FaultedFallback, now);
        }

        let Some(elapsed) = now.checked_millis_since(received_at) else {
            let error = ControllerError::ReceiveTimestampFromFuture {
                received_at_ms: received_at.millis_since_boot(),
                handled_at_ms: now.millis_since_boot(),
            };
            self.enter_internal_fault();
            return Err(error);
        };
        if elapsed >= u64::from(command.lease.get()) {
            return self.instant_result_for_apply(command, IntentResultCode::RejectedExpired, now);
        }

        let ControlState::Active(mut active) = self.state else {
            return self.instant_result_for_apply(command, IntentResultCode::RejectedSession, now);
        };
        if command.boot_id != self.identity.boot_id || command.control_epoch != active.epoch {
            return self.instant_result_for_apply(command, IntentResultCode::RejectedSession, now);
        }

        if let Some(cached) = active.cached_admission
            && command.sequence == cached.command.sequence
        {
            if command != cached.command {
                return self.instant_result_for_apply(
                    command,
                    IntentResultCode::RejectedSequence,
                    now,
                );
            }
            let duplicate = IntentResult::try_new(
                cached.result.boot_id(),
                cached.result.control_epoch(),
                cached.result.sequence(),
                IntentResultCode::DuplicateCached,
                cached.result.applied_at(),
                cached.result.expires_at(),
                cached.result.rendered_frame_sequence(),
            )
            .map_err(|error| self.result_invariant(error))?;
            return Ok(Message::IntentResult(duplicate));
        }

        if command.sequence != active.next_sequence {
            return self.instant_result_for_apply(command, IntentResultCode::RejectedSequence, now);
        }
        let Some(successor) = command.sequence.checked_successor() else {
            self.enter_fallback(FallbackCause::IntentSequenceExhausted);
            return self.instant_result_for_apply(command, IntentResultCode::FaultedFallback, now);
        };

        let expires_at = self.checked_deadline(now, u64::from(command.lease.get()))?;
        self.rendered_frame_sequence = self.rendered_frame_sequence.wrapping_add(1);
        let result = IntentResult::try_new(
            command.boot_id,
            command.control_epoch,
            command.sequence,
            IntentResultCode::AppliedNew,
            now,
            expires_at,
            RenderedFrameSequence::new(self.rendered_frame_sequence),
        )
        .map_err(|error| self.result_invariant(error))?;

        active.next_sequence = successor;
        active.deadline_exclusive = expires_at;
        active.cached_admission = Some(CachedAdmission { command, result });
        self.state = ControlState::Active(active);
        self.output = OutputState::Commanded {
            intent: command.intent,
            sequence: command.sequence,
        };
        Ok(Message::IntentResult(result))
    }

    fn release(
        &mut self,
        release: ReleaseControl,
        now: DeviceTimestampMs,
    ) -> Result<Message, ControllerError> {
        if self.sticky_internal_fault {
            return self.instant_result_for_release(
                release,
                IntentResultCode::FaultedFallback,
                now,
            );
        }
        let ControlState::Active(active) = self.state else {
            return self.instant_result_for_release(
                release,
                IntentResultCode::RejectedSession,
                now,
            );
        };
        if release.boot_id != self.identity.boot_id || release.control_epoch != active.epoch {
            return self.instant_result_for_release(
                release,
                IntentResultCode::RejectedSession,
                now,
            );
        }
        if release.sequence != active.next_sequence {
            return self.instant_result_for_release(
                release,
                IntentResultCode::RejectedSequence,
                now,
            );
        }

        let response = self.instant_result_for_release(release, IntentResultCode::Released, now)?;
        self.enter_fallback(FallbackCause::Released(release.reason));
        Ok(response)
    }

    fn instant_result_for_apply(
        &mut self,
        command: ApplyIntent,
        code: IntentResultCode,
        now: DeviceTimestampMs,
    ) -> Result<Message, ControllerError> {
        self.instant_result(
            command.boot_id,
            command.control_epoch,
            command.sequence,
            code,
            now,
        )
    }

    fn instant_result_for_release(
        &mut self,
        release: ReleaseControl,
        code: IntentResultCode,
        now: DeviceTimestampMs,
    ) -> Result<Message, ControllerError> {
        self.instant_result(
            release.boot_id,
            release.control_epoch,
            release.sequence,
            code,
            now,
        )
    }

    fn instant_result(
        &mut self,
        boot_id: DeviceBootId,
        control_epoch: ControlEpoch,
        sequence: IntentSequence,
        code: IntentResultCode,
        now: DeviceTimestampMs,
    ) -> Result<Message, ControllerError> {
        let result = IntentResult::try_new(
            boot_id,
            control_epoch,
            sequence,
            code,
            now,
            now,
            RenderedFrameSequence::new(self.rendered_frame_sequence),
        )
        .map_err(|error| self.result_invariant(error))?;
        Ok(Message::IntentResult(result))
    }

    fn observe_time(&mut self, now: DeviceTimestampMs) -> Result<(), ControllerError> {
        if now < self.last_observed_time {
            let error = ControllerError::ClockRegressed {
                previous_ms: self.last_observed_time.millis_since_boot(),
                actual_ms: now.millis_since_boot(),
            };
            self.enter_internal_fault();
            return Err(error);
        }
        self.last_observed_time = now;
        Ok(())
    }

    fn checked_deadline(
        &mut self,
        now: DeviceTimestampMs,
        duration_ms: u64,
    ) -> Result<DeviceTimestampMs, ControllerError> {
        let Some(deadline) = now.millis_since_boot().checked_add(duration_ms) else {
            let error = ControllerError::DeadlineOverflow {
                now_ms: now.millis_since_boot(),
                duration_ms,
            };
            self.enter_internal_fault();
            return Err(error);
        };
        Ok(DeviceTimestampMs::from_millis_since_boot(deadline))
    }

    fn result_invariant(&mut self, error: DomainError) -> ControllerError {
        self.enter_internal_fault();
        ControllerError::ResultInvariant(error)
    }

    fn enter_fallback(&mut self, cause: FallbackCause) {
        self.state = ControlState::Idle;
        self.output = OutputState::Autonomous {
            cause: if self.sticky_internal_fault {
                FallbackCause::InternalFault
            } else {
                cause
            },
        };
    }

    fn enter_internal_fault(&mut self) {
        self.sticky_internal_fault = true;
        self.enter_fallback(FallbackCause::InternalFault);
    }
}
