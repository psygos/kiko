use std::fmt;
use std::time::Duration;

use kiko_expression_core::MonotonicTimestamp;
use kiko_expression_runtime::{
    ControlBinding, EyeSession, EyeSessionFault, FirmwareAdmission, InboundMessageKind,
    PreparedEyeIntent, SessionEvent,
};
use kiko_eye_protocol::{
    EncodeError, IdentityReport, MAX_ENCODED_FRAME_BYTES, Message, ReleaseControl, ReleaseReason,
    encode,
};
use tokio::runtime::Handle;
use tokio::sync::{mpsc, oneshot};
use tokio::task::{JoinError, JoinHandle};

use crate::config::{EyeRuntimeConfig, OperationTimeout};
use crate::framing::{FrameReadError, FrameReader, ReceivedMessage};
use crate::transport::{
    AsyncByteTransport, ClockError, MonotonicClock, SerialConfigurationEvidence, SerialOpenError,
    SerialTransport, TransportFailure, TransportOperation,
};

/// One command may wait behind the operation currently owned by the actor.
/// The public handle is not cloneable and apply requires `&mut self`, making
/// concurrent submissions unavailable through the safe API.
pub const ACTOR_MAILBOX_CAPACITY: usize = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ProtocolExchange {
    Identity,
    Acquire,
    Intent,
    Release,
    CleanupRelease,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrameWriteEvidence {
    exchange: ProtocolExchange,
    attempts_used: u8,
    recovered_failures: Vec<TransportFailure>,
    encoded_frame_bytes: usize,
    completed_at: MonotonicTimestamp,
}

impl FrameWriteEvidence {
    pub const fn exchange(&self) -> ProtocolExchange {
        self.exchange
    }

    pub const fn attempts_used(&self) -> u8 {
        self.attempts_used
    }

    pub fn recovered_failures(&self) -> impl ExactSizeIterator<Item = &TransportFailure> {
        self.recovered_failures.iter()
    }

    pub const fn encoded_frame_bytes(&self) -> usize {
        self.encoded_frame_bytes
    }

    pub const fn completed_at(&self) -> MonotonicTimestamp {
        self.completed_at
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FrameWriteFailure {
    Encode(EncodeError),
    Clock(ClockError),
    DeadlineOverflow {
        started_at_ns: u64,
        timeout: OperationTimeout,
    },
    Transport(TransportFailure),
    TransportContract {
        expected_operation: TransportOperation,
        source: TransportFailure,
        known_total_progress: usize,
    },
}

impl FrameWriteFailure {
    fn known_progress(&self) -> usize {
        match self {
            Self::Transport(source) => source.bytes_transferred(),
            Self::TransportContract {
                known_total_progress,
                ..
            } => *known_total_progress,
            Self::Encode(_) | Self::Clock(_) | Self::DeadlineOverflow { .. } => 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrameWriteError {
    exchange: ProtocolExchange,
    attempts_used: u8,
    recovered_failures: Vec<TransportFailure>,
    encoded_frame_bytes: Option<usize>,
    source: FrameWriteFailure,
}

impl FrameWriteError {
    pub const fn exchange(&self) -> ProtocolExchange {
        self.exchange
    }

    pub const fn attempts_used(&self) -> u8 {
        self.attempts_used
    }

    pub fn recovered_failures(&self) -> impl ExactSizeIterator<Item = &TransportFailure> {
        self.recovered_failures.iter()
    }

    pub const fn encoded_frame_bytes(&self) -> Option<usize> {
        self.encoded_frame_bytes
    }

    pub const fn source(&self) -> &FrameWriteFailure {
        &self.source
    }

    /// True means at least one frame byte may have reached the driver, or a
    /// flush failed after the complete frame was written. Retransmission is
    /// therefore intentionally forbidden.
    pub fn transmission_uncertain(&self) -> bool {
        self.source.known_progress() != 0
            || matches!(
                self.source,
                FrameWriteFailure::Transport(ref source)
                    if source.operation() == TransportOperation::Flush
            )
    }
}

impl fmt::Display for FrameWriteError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{:?} KEP2 write failed on attempt {}: {:?}",
            self.exchange, self.attempts_used, self.source
        )
    }
}

impl std::error::Error for FrameWriteError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.source {
            FrameWriteFailure::Encode(source) => Some(source),
            FrameWriteFailure::Clock(source) => Some(source),
            FrameWriteFailure::Transport(source) => Some(source),
            FrameWriteFailure::TransportContract { source, .. } => Some(source),
            FrameWriteFailure::DeadlineOverflow { .. } => None,
        }
    }
}

/// Host-side evidence for exact firmware admission. It does not claim that
/// pixels were visible on either panel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FirmwareAdmissionEvidence {
    admission: FirmwareAdmission,
    request_write: FrameWriteEvidence,
    response_received_at: MonotonicTimestamp,
}

impl FirmwareAdmissionEvidence {
    pub const fn admission(&self) -> FirmwareAdmission {
        self.admission
    }

    pub const fn request_write(&self) -> &FrameWriteEvidence {
        &self.request_write
    }

    pub const fn response_received_at(&self) -> MonotonicTimestamp {
        self.response_received_at
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StartupEvidence {
    identity: IdentityReport,
    binding: ControlBinding,
    identity_query_write: FrameWriteEvidence,
    acquire_write: FrameWriteEvidence,
    completed_at: MonotonicTimestamp,
}

impl StartupEvidence {
    pub const fn identity(&self) -> IdentityReport {
        self.identity
    }

    pub const fn binding(&self) -> ControlBinding {
        self.binding
    }

    pub const fn identity_query_write(&self) -> &FrameWriteEvidence {
        &self.identity_query_write
    }

    pub const fn acquire_write(&self) -> &FrameWriteEvidence {
        &self.acquire_write
    }

    pub const fn completed_at(&self) -> MonotonicTimestamp {
        self.completed_at
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReleaseEvidence {
    binding: ControlBinding,
    request_write: FrameWriteEvidence,
    response_received_at: MonotonicTimestamp,
}

impl ReleaseEvidence {
    pub const fn binding(&self) -> ControlBinding {
        self.binding
    }

    pub const fn request_write(&self) -> &FrameWriteEvidence {
        &self.request_write
    }

    pub const fn response_received_at(&self) -> MonotonicTimestamp {
        self.response_received_at
    }
}

/// Result of best-effort fallback cleanup. `WriteCompleted` proves only that
/// the host completed and flushed a release frame; it is not firmware receipt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CleanupOutcome {
    SessionProvidedNoAdditionalRelease,
    WriteCompleted {
        request: ReleaseControl,
        evidence: FrameWriteEvidence,
    },
    WriteFailed {
        request: ReleaseControl,
        source: FrameWriteError,
    },
}

/// A graceful release frame whose host write completed before a later
/// response-side fault. Firmware admission remains unknown unless the normal
/// [`ReleaseEvidence`] path succeeds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PriorReleaseAttempt {
    request: ReleaseControl,
    write: FrameWriteEvidence,
}

impl PriorReleaseAttempt {
    pub const fn request(&self) -> ReleaseControl {
        self.request
    }

    pub const fn write(&self) -> &FrameWriteEvidence {
        &self.write
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RuntimeFaultCause {
    Clock(ClockError),
    Session(EyeSessionFault),
    Write(FrameWriteError),
    Read {
        exchange: ProtocolExchange,
        source: FrameReadError,
    },
    UnexpectedSessionEvent {
        exchange: ProtocolExchange,
        actual: SessionEvent,
    },
    UnexpectedOutboundMessage {
        exchange: ProtocolExchange,
        actual: InboundMessageKind,
    },
    ResponsePredatesRequest {
        exchange: ProtocolExchange,
        request_completed_at_ns: u64,
        response_started_at_ns: u64,
    },
    Cancellation(CancellationCause),
    AdmissionCountExhaustedAfterAdmission(FirmwareAdmissionEvidence),
    ResponseReceiverDroppedAfterAdmission(FirmwareAdmissionEvidence),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EyeRuntimeFault {
    cause: RuntimeFaultCause,
    session_fault: EyeSessionFault,
    cleanup: CleanupOutcome,
    prior_release_attempt: Option<PriorReleaseAttempt>,
}

impl EyeRuntimeFault {
    pub const fn cause(&self) -> &RuntimeFaultCause {
        &self.cause
    }

    pub const fn session_fault(&self) -> EyeSessionFault {
        self.session_fault
    }

    pub const fn cleanup(&self) -> &CleanupOutcome {
        &self.cleanup
    }

    pub const fn prior_release_attempt(&self) -> Option<&PriorReleaseAttempt> {
        self.prior_release_attempt.as_ref()
    }

    fn with_prior_release_attempt(
        mut self,
        request: ReleaseControl,
        write: FrameWriteEvidence,
    ) -> Self {
        self.prior_release_attempt = Some(PriorReleaseAttempt { request, write });
        self
    }
}

impl fmt::Display for EyeRuntimeFault {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "KEP2 eye actor entered fallback: {:?}",
            self.cause
        )
    }
}

impl std::error::Error for EyeRuntimeFault {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.cause {
            RuntimeFaultCause::Clock(source) => Some(source),
            RuntimeFaultCause::Session(source) => Some(source),
            RuntimeFaultCause::Write(source) => Some(source),
            RuntimeFaultCause::Read { source, .. } => Some(source),
            RuntimeFaultCause::UnexpectedSessionEvent { .. }
            | RuntimeFaultCause::UnexpectedOutboundMessage { .. }
            | RuntimeFaultCause::ResponsePredatesRequest { .. }
            | RuntimeFaultCause::Cancellation(_)
            | RuntimeFaultCause::AdmissionCountExhaustedAfterAdmission(_)
            | RuntimeFaultCause::ResponseReceiverDroppedAfterAdmission(_) => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CancellationCause {
    Requested,
    HandleDropped,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReleaseReport {
    Released(ReleaseEvidence),
    Fallback(Box<EyeRuntimeFault>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ActorTermination {
    RequestedShutdown,
    Cancellation(CancellationCause),
    StartupFault,
    RuntimeFault,
    ResponseReceiverDropped,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActorExit {
    startup: Result<StartupEvidence, EyeRuntimeFault>,
    termination: ActorTermination,
    release: Option<ReleaseReport>,
    admitted_intent_count: u64,
    last_admission: Option<FirmwareAdmissionEvidence>,
}

impl ActorExit {
    pub const fn startup(&self) -> &Result<StartupEvidence, EyeRuntimeFault> {
        &self.startup
    }

    pub const fn termination(&self) -> &ActorTermination {
        &self.termination
    }

    pub const fn release(&self) -> Option<&ReleaseReport> {
        self.release.as_ref()
    }

    pub const fn admitted_intent_count(&self) -> u64 {
        self.admitted_intent_count
    }

    pub const fn last_admission(&self) -> Option<&FirmwareAdmissionEvidence> {
        self.last_admission.as_ref()
    }
}

enum EyeCommand {
    Apply {
        prepared: PreparedEyeIntent,
        response: oneshot::Sender<Result<FirmwareAdmissionEvidence, EyeRuntimeFault>>,
    },
    Shutdown {
        response: oneshot::Sender<ReleaseReport>,
    },
    Cancel {
        response: oneshot::Sender<EyeRuntimeFault>,
    },
}

/// Non-cloneable command endpoint for one explicit in-flight request.
pub struct EyeActorHandle {
    commands: mpsc::Sender<EyeCommand>,
}

impl EyeActorHandle {
    pub async fn apply_intent(
        &mut self,
        prepared: PreparedEyeIntent,
    ) -> Result<FirmwareAdmissionEvidence, HandleRequestError> {
        let (response, result) = oneshot::channel();
        self.commands
            .send(EyeCommand::Apply { prepared, response })
            .await
            .map_err(|_| HandleRequestError::ActorAlreadyStopped)?;
        result
            .await
            .map_err(|_| HandleRequestError::ActorStoppedBeforeReporting)?
            .map_err(|source| HandleRequestError::Runtime(Box::new(source)))
    }

    pub async fn shutdown(self) -> Result<ReleaseReport, HandleRequestError> {
        let (response, result) = oneshot::channel();
        self.commands
            .send(EyeCommand::Shutdown { response })
            .await
            .map_err(|_| HandleRequestError::ActorAlreadyStopped)?;
        result
            .await
            .map_err(|_| HandleRequestError::ActorStoppedBeforeReporting)
    }

    pub async fn cancel(self) -> Result<EyeRuntimeFault, HandleRequestError> {
        let (response, result) = oneshot::channel();
        self.commands
            .send(EyeCommand::Cancel { response })
            .await
            .map_err(|_| HandleRequestError::ActorAlreadyStopped)?;
        result
            .await
            .map_err(|_| HandleRequestError::ActorStoppedBeforeReporting)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HandleRequestError {
    ActorAlreadyStopped,
    ActorStoppedBeforeReporting,
    Runtime(Box<EyeRuntimeFault>),
}

impl fmt::Display for HandleRequestError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "KEP2 eye actor request failed: {self:?}")
    }
}

impl std::error::Error for HandleRequestError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Runtime(source) => Some(source),
            Self::ActorAlreadyStopped | Self::ActorStoppedBeforeReporting => None,
        }
    }
}

pub struct StartupReceipt {
    result: oneshot::Receiver<Result<StartupEvidence, EyeRuntimeFault>>,
}

impl StartupReceipt {
    pub async fn wait(
        self,
    ) -> Result<Result<StartupEvidence, EyeRuntimeFault>, StartupReceiptError> {
        self.result
            .await
            .map_err(|_| StartupReceiptError::ActorStoppedBeforeReporting)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StartupReceiptError {
    ActorStoppedBeforeReporting,
}

impl fmt::Display for StartupReceiptError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "KEP2 eye startup receipt failed: {self:?}")
    }
}

impl std::error::Error for StartupReceiptError {}

pub struct EyeActorTask {
    task: JoinHandle<ActorExit>,
}

impl EyeActorTask {
    pub async fn join(self) -> Result<ActorExit, JoinError> {
        self.task.await
    }
}

#[derive(Debug)]
pub enum EyeActorSpawnError {
    NoTokioRuntime {
        source: tokio::runtime::TryCurrentError,
    },
}

impl fmt::Display for EyeActorSpawnError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("could not spawn KEP2 eye actor without an active Tokio runtime")
    }
}

impl std::error::Error for EyeActorSpawnError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::NoTokioRuntime { source } => Some(source),
        }
    }
}

#[derive(Debug)]
pub enum EyeActorStartError {
    NoTokioRuntime {
        source: tokio::runtime::TryCurrentError,
    },
    Serial {
        source: SerialOpenError,
    },
}

impl fmt::Display for EyeActorStartError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("could not start production KEP2 eye actor")
    }
}

impl std::error::Error for EyeActorStartError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::NoTokioRuntime { source } => Some(source),
            Self::Serial { source } => Some(source),
        }
    }
}

pub fn spawn_eye_actor<T, C>(
    transport: T,
    clock: C,
    config: EyeRuntimeConfig,
) -> Result<(EyeActorHandle, StartupReceipt, EyeActorTask), EyeActorSpawnError>
where
    T: AsyncByteTransport,
    C: MonotonicClock,
{
    let runtime =
        Handle::try_current().map_err(|source| EyeActorSpawnError::NoTokioRuntime { source })?;
    Ok(spawn_eye_actor_on(&runtime, transport, clock, config))
}

fn spawn_eye_actor_on<T, C>(
    runtime: &Handle,
    transport: T,
    clock: C,
    config: EyeRuntimeConfig,
) -> (EyeActorHandle, StartupReceipt, EyeActorTask)
where
    T: AsyncByteTransport,
    C: MonotonicClock,
{
    let (commands, receiver) = mpsc::channel(ACTOR_MAILBOX_CAPACITY);
    let (startup_sender, startup_result) = oneshot::channel();
    let actor = EyeActor {
        transport,
        clock,
        session: EyeSession::new(config.session_plan()),
        config,
        reader: FrameReader::new(),
        admitted_intent_count: 0,
        last_admission: None,
    };
    let task = runtime.spawn(actor.run(receiver, startup_sender));
    (
        EyeActorHandle { commands },
        StartupReceipt {
            result: startup_result,
        },
        EyeActorTask { task },
    )
}

pub fn start_serial_eye_actor<C>(
    config: EyeRuntimeConfig,
    clock: C,
) -> Result<
    (
        SerialConfigurationEvidence,
        EyeActorHandle,
        StartupReceipt,
        EyeActorTask,
    ),
    EyeActorStartError,
>
where
    C: MonotonicClock,
{
    // Opening a TTY can change physical serial state. Prove the async owner
    // exists before opening the configured hardware path.
    let runtime =
        Handle::try_current().map_err(|source| EyeActorStartError::NoTokioRuntime { source })?;
    let transport = SerialTransport::open(config.device(), config.baud_rate())
        .map_err(|source| EyeActorStartError::Serial { source })?;
    let serial_evidence = transport.evidence().clone();
    let (handle, startup, task) = spawn_eye_actor_on(&runtime, transport, clock, config);
    Ok((serial_evidence, handle, startup, task))
}

struct EyeActor<T, C> {
    transport: T,
    clock: C,
    session: EyeSession,
    config: EyeRuntimeConfig,
    reader: FrameReader,
    admitted_intent_count: u64,
    last_admission: Option<FirmwareAdmissionEvidence>,
}

impl<T, C> EyeActor<T, C>
where
    T: AsyncByteTransport,
    C: MonotonicClock,
{
    async fn run(
        mut self,
        mut commands: mpsc::Receiver<EyeCommand>,
        startup_sender: oneshot::Sender<Result<StartupEvidence, EyeRuntimeFault>>,
    ) -> ActorExit {
        let startup = self.startup(&commands).await;
        let _receiver_present = startup_sender.send(startup.clone()).is_ok();
        if let Err(fault) = &startup {
            let release_fault = fault.clone();
            let termination = if matches!(
                fault.cause(),
                RuntimeFaultCause::Cancellation(CancellationCause::HandleDropped)
            ) {
                ActorTermination::Cancellation(CancellationCause::HandleDropped)
            } else {
                ActorTermination::StartupFault
            };
            return ActorExit {
                startup,
                termination,
                release: Some(ReleaseReport::Fallback(Box::new(release_fault))),
                admitted_intent_count: self.admitted_intent_count,
                last_admission: self.last_admission,
            };
        }

        loop {
            match commands.recv().await {
                Some(EyeCommand::Apply { prepared, response }) => {
                    match self.apply(prepared).await {
                        Ok(evidence) => {
                            self.last_admission = Some(evidence.clone());
                            self.admitted_intent_count = match self
                                .admitted_intent_count
                                .checked_add(1)
                            {
                                Some(value) => value,
                                None => {
                                    let fault = self
                                        .external_fault(
                                            RuntimeFaultCause::AdmissionCountExhaustedAfterAdmission(
                                                evidence,
                                            ),
                                        )
                                        .await;
                                    let _requester_present =
                                        response.send(Err(fault.clone())).is_ok();
                                    return self.exit(
                                        startup,
                                        ActorTermination::RuntimeFault,
                                        Some(ReleaseReport::Fallback(Box::new(fault))),
                                    );
                                }
                            };
                            let cancellation_evidence = evidence.clone();
                            if response.send(Ok(evidence)).is_err() {
                                let fault = self
                                    .external_fault(
                                        RuntimeFaultCause::ResponseReceiverDroppedAfterAdmission(
                                            cancellation_evidence,
                                        ),
                                    )
                                    .await;
                                return self.exit(
                                    startup,
                                    ActorTermination::ResponseReceiverDropped,
                                    Some(ReleaseReport::Fallback(Box::new(fault))),
                                );
                            }
                        }
                        Err(fault) => {
                            let _requester_present = response.send(Err(fault.clone())).is_ok();
                            return self.exit(
                                startup,
                                ActorTermination::RuntimeFault,
                                Some(ReleaseReport::Fallback(Box::new(fault))),
                            );
                        }
                    }
                }
                Some(EyeCommand::Shutdown { response }) => {
                    let report = self.release_normally().await;
                    let _requester_present = response.send(report.clone()).is_ok();
                    return self.exit(startup, ActorTermination::RequestedShutdown, Some(report));
                }
                Some(EyeCommand::Cancel { response }) => {
                    let session_fault = self.session.transport_fault();
                    let fault = self
                        .fault_with_session(
                            RuntimeFaultCause::Cancellation(CancellationCause::Requested),
                            session_fault,
                        )
                        .await;
                    let _requester_present = response.send(fault.clone()).is_ok();
                    return self.exit(
                        startup,
                        ActorTermination::Cancellation(CancellationCause::Requested),
                        Some(ReleaseReport::Fallback(Box::new(fault))),
                    );
                }
                None => {
                    let session_fault = self.session.transport_fault();
                    let fault = self
                        .fault_with_session(
                            RuntimeFaultCause::Cancellation(CancellationCause::HandleDropped),
                            session_fault,
                        )
                        .await;
                    return self.exit(
                        startup,
                        ActorTermination::Cancellation(CancellationCause::HandleDropped),
                        Some(ReleaseReport::Fallback(Box::new(fault))),
                    );
                }
            }
        }
    }

    fn exit(
        self,
        startup: Result<StartupEvidence, EyeRuntimeFault>,
        termination: ActorTermination,
        release: Option<ReleaseReport>,
    ) -> ActorExit {
        ActorExit {
            startup,
            termination,
            release,
            admitted_intent_count: self.admitted_intent_count,
            last_admission: self.last_admission,
        }
    }

    async fn startup(
        &mut self,
        commands: &mpsc::Receiver<EyeCommand>,
    ) -> Result<StartupEvidence, EyeRuntimeFault> {
        self.ensure_startup_owner(commands).await?;
        let now = self.now_or_fault().await?;
        let identity_query = match self.session.begin_identity(now) {
            Ok(value) => value,
            Err(session_fault) => {
                return Err(self
                    .fault_with_session(RuntimeFaultCause::Session(session_fault), session_fault)
                    .await);
            }
        };
        let identity_query_write = match self
            .write_message(identity_query, ProtocolExchange::Identity)
            .await
        {
            Ok(value) => value,
            Err(source) => {
                return Err(self.external_fault(RuntimeFaultCause::Write(source)).await);
            }
        };
        let inbound = match self.read_message(ProtocolExchange::Identity).await {
            Ok(value) => value,
            Err(source) => {
                return Err(self
                    .external_fault(RuntimeFaultCause::Read {
                        exchange: ProtocolExchange::Identity,
                        source,
                    })
                    .await);
            }
        };
        self.ensure_startup_owner(commands).await?;
        if inbound.started_at <= identity_query_write.completed_at() {
            return Err(self
                .external_fault(RuntimeFaultCause::ResponsePredatesRequest {
                    exchange: ProtocolExchange::Identity,
                    request_completed_at_ns: identity_query_write
                        .completed_at()
                        .nanos_since_epoch(),
                    response_started_at_ns: inbound.started_at.nanos_since_epoch(),
                })
                .await);
        }
        let received_at = self.now_or_fault().await?;
        let identity = match self.session.handle_inbound(inbound.message, received_at) {
            Ok(SessionEvent::IdentityVerified(identity)) => identity,
            Ok(actual) => {
                return Err(self
                    .external_fault(RuntimeFaultCause::UnexpectedSessionEvent {
                        exchange: ProtocolExchange::Identity,
                        actual,
                    })
                    .await);
            }
            Err(session_fault) => {
                return Err(self
                    .fault_with_session(RuntimeFaultCause::Session(session_fault), session_fault)
                    .await);
            }
        };

        self.ensure_startup_owner(commands).await?;
        let now = self.now_or_fault().await?;
        let acquire = match self.session.begin_acquire(now) {
            Ok(value) => value,
            Err(session_fault) => {
                return Err(self
                    .fault_with_session(RuntimeFaultCause::Session(session_fault), session_fault)
                    .await);
            }
        };
        let acquire_write = match self.write_message(acquire, ProtocolExchange::Acquire).await {
            Ok(value) => value,
            Err(source) => {
                return Err(self.external_fault(RuntimeFaultCause::Write(source)).await);
            }
        };
        let inbound = match self.read_message(ProtocolExchange::Acquire).await {
            Ok(value) => value,
            Err(source) => {
                return Err(self
                    .external_fault(RuntimeFaultCause::Read {
                        exchange: ProtocolExchange::Acquire,
                        source,
                    })
                    .await);
            }
        };
        self.ensure_startup_owner(commands).await?;
        if inbound.started_at <= acquire_write.completed_at() {
            return Err(self
                .external_fault(RuntimeFaultCause::ResponsePredatesRequest {
                    exchange: ProtocolExchange::Acquire,
                    request_completed_at_ns: acquire_write.completed_at().nanos_since_epoch(),
                    response_started_at_ns: inbound.started_at.nanos_since_epoch(),
                })
                .await);
        }
        let received_at = self.now_or_fault().await?;
        let binding = match self.session.handle_inbound(inbound.message, received_at) {
            Ok(SessionEvent::ControlAcquired(binding)) => binding,
            Ok(actual) => {
                return Err(self
                    .external_fault(RuntimeFaultCause::UnexpectedSessionEvent {
                        exchange: ProtocolExchange::Acquire,
                        actual,
                    })
                    .await);
            }
            Err(session_fault) => {
                return Err(self
                    .fault_with_session(RuntimeFaultCause::Session(session_fault), session_fault)
                    .await);
            }
        };
        self.ensure_startup_owner(commands).await?;

        Ok(StartupEvidence {
            identity,
            binding,
            identity_query_write,
            acquire_write,
            completed_at: received_at,
        })
    }

    async fn apply(
        &mut self,
        prepared: PreparedEyeIntent,
    ) -> Result<FirmwareAdmissionEvidence, EyeRuntimeFault> {
        let now = self.now_or_fault().await?;
        let message = match self
            .session
            .submit_intent(prepared, self.config.intent_lease(), now)
        {
            Ok(message) => message,
            Err(session_fault) => {
                return Err(self
                    .fault_with_session(RuntimeFaultCause::Session(session_fault), session_fault)
                    .await);
            }
        };
        let request_write = match self.write_message(message, ProtocolExchange::Intent).await {
            Ok(value) => value,
            Err(source) => {
                return Err(self.external_fault(RuntimeFaultCause::Write(source)).await);
            }
        };
        let inbound = match self.read_message(ProtocolExchange::Intent).await {
            Ok(value) => value,
            Err(source) => {
                return Err(self
                    .external_fault(RuntimeFaultCause::Read {
                        exchange: ProtocolExchange::Intent,
                        source,
                    })
                    .await);
            }
        };
        if inbound.started_at <= request_write.completed_at() {
            return Err(self
                .external_fault(RuntimeFaultCause::ResponsePredatesRequest {
                    exchange: ProtocolExchange::Intent,
                    request_completed_at_ns: request_write.completed_at().nanos_since_epoch(),
                    response_started_at_ns: inbound.started_at.nanos_since_epoch(),
                })
                .await);
        }
        let response_received_at = self.now_or_fault().await?;
        let admission = match self
            .session
            .handle_inbound(inbound.message, response_received_at)
        {
            Ok(SessionEvent::IntentAdmitted(admission)) => admission,
            Ok(actual) => {
                return Err(self
                    .external_fault(RuntimeFaultCause::UnexpectedSessionEvent {
                        exchange: ProtocolExchange::Intent,
                        actual,
                    })
                    .await);
            }
            Err(session_fault) => {
                return Err(self
                    .fault_with_session(RuntimeFaultCause::Session(session_fault), session_fault)
                    .await);
            }
        };
        Ok(FirmwareAdmissionEvidence {
            admission,
            request_write,
            response_received_at,
        })
    }

    async fn release_normally(&mut self) -> ReleaseReport {
        match self.release_normally_inner().await {
            Ok(evidence) => ReleaseReport::Released(evidence),
            Err(fault) => ReleaseReport::Fallback(Box::new(fault)),
        }
    }

    async fn release_normally_inner(&mut self) -> Result<ReleaseEvidence, EyeRuntimeFault> {
        let now = self.now_or_fault().await?;
        let message = match self.session.begin_release(ReleaseReason::HostShutdown, now) {
            Ok(value) => value,
            Err(session_fault) => {
                return Err(self
                    .fault_with_session(RuntimeFaultCause::Session(session_fault), session_fault)
                    .await);
            }
        };
        let release_request = match message {
            Message::ReleaseControl(request) => request,
            actual => {
                return Err(self
                    .external_fault(RuntimeFaultCause::UnexpectedOutboundMessage {
                        exchange: ProtocolExchange::Release,
                        actual: actual.into(),
                    })
                    .await);
            }
        };
        let request_write = match self.write_message(message, ProtocolExchange::Release).await {
            Ok(value) => value,
            Err(source) => {
                return Err(self.external_fault(RuntimeFaultCause::Write(source)).await);
            }
        };
        let inbound = match self.read_message(ProtocolExchange::Release).await {
            Ok(value) => value,
            Err(source) => {
                return Err(self
                    .external_fault(RuntimeFaultCause::Read {
                        exchange: ProtocolExchange::Release,
                        source,
                    })
                    .await
                    .with_prior_release_attempt(release_request, request_write));
            }
        };
        if inbound.started_at <= request_write.completed_at() {
            return Err(self
                .external_fault(RuntimeFaultCause::ResponsePredatesRequest {
                    exchange: ProtocolExchange::Release,
                    request_completed_at_ns: request_write.completed_at().nanos_since_epoch(),
                    response_started_at_ns: inbound.started_at.nanos_since_epoch(),
                })
                .await
                .with_prior_release_attempt(release_request, request_write));
        }
        let response_received_at = match self.now_or_fault().await {
            Ok(value) => value,
            Err(fault) => {
                return Err(fault.with_prior_release_attempt(release_request, request_write));
            }
        };
        let binding = match self
            .session
            .handle_inbound(inbound.message, response_received_at)
        {
            Ok(SessionEvent::Released(binding)) => binding,
            Ok(actual) => {
                return Err(self
                    .external_fault(RuntimeFaultCause::UnexpectedSessionEvent {
                        exchange: ProtocolExchange::Release,
                        actual,
                    })
                    .await
                    .with_prior_release_attempt(release_request, request_write));
            }
            Err(session_fault) => {
                return Err(self
                    .fault_with_session(RuntimeFaultCause::Session(session_fault), session_fault)
                    .await
                    .with_prior_release_attempt(release_request, request_write));
            }
        };
        Ok(ReleaseEvidence {
            binding,
            request_write,
            response_received_at,
        })
    }

    async fn now_or_fault(&mut self) -> Result<MonotonicTimestamp, EyeRuntimeFault> {
        match self.clock.now() {
            Ok(value) => Ok(value),
            Err(source) => Err(self.external_fault(RuntimeFaultCause::Clock(source)).await),
        }
    }

    async fn ensure_startup_owner(
        &mut self,
        commands: &mpsc::Receiver<EyeCommand>,
    ) -> Result<(), EyeRuntimeFault> {
        if commands.is_closed() {
            let session_fault = self.session.transport_fault();
            Err(self
                .fault_with_session(
                    RuntimeFaultCause::Cancellation(CancellationCause::HandleDropped),
                    session_fault,
                )
                .await)
        } else {
            Ok(())
        }
    }

    async fn read_message(
        &mut self,
        exchange: ProtocolExchange,
    ) -> Result<ReceivedMessage, FrameReadError> {
        let _exchange = exchange;
        self.reader
            .read_message(
                &mut self.transport,
                &self.clock,
                self.config.response_timeout(),
                self.config.empty_delimiter_budget(),
            )
            .await
    }

    async fn external_fault(&mut self, cause: RuntimeFaultCause) -> EyeRuntimeFault {
        let session_fault = self.session.transport_fault();
        self.fault_with_session(cause, session_fault).await
    }

    async fn fault_with_session(
        &mut self,
        cause: RuntimeFaultCause,
        session_fault: EyeSessionFault,
    ) -> EyeRuntimeFault {
        let cleanup = self.cleanup(session_fault.release()).await;
        EyeRuntimeFault {
            cause,
            session_fault,
            cleanup,
            prior_release_attempt: None,
        }
    }

    async fn cleanup(&mut self, release: Option<ReleaseControl>) -> CleanupOutcome {
        let Some(request) = release else {
            return CleanupOutcome::SessionProvidedNoAdditionalRelease;
        };
        match self
            .write_message(
                Message::ReleaseControl(request),
                ProtocolExchange::CleanupRelease,
            )
            .await
        {
            Ok(evidence) => CleanupOutcome::WriteCompleted { request, evidence },
            Err(source) => CleanupOutcome::WriteFailed { request, source },
        }
    }

    async fn write_message(
        &mut self,
        message: Message,
        exchange: ProtocolExchange,
    ) -> Result<FrameWriteEvidence, FrameWriteError> {
        let mut frame = [0_u8; MAX_ENCODED_FRAME_BYTES];
        let frame_len = encode(message, &mut frame).map_err(|source| FrameWriteError {
            exchange,
            attempts_used: 0,
            recovered_failures: Vec::new(),
            encoded_frame_bytes: None,
            source: FrameWriteFailure::Encode(source),
        })?;
        let attempts_limit = self.config.write_attempts().get();
        let mut recovered = Vec::with_capacity(usize::from(attempts_limit - 1));

        for attempt in 1..=attempts_limit {
            match self.write_one_attempt(&frame[..frame_len]).await {
                Ok(completed_at) => {
                    return Ok(FrameWriteEvidence {
                        exchange,
                        attempts_used: attempt,
                        recovered_failures: recovered,
                        encoded_frame_bytes: frame_len,
                        completed_at,
                    });
                }
                Err(FrameWriteFailure::Transport(source))
                    if attempt < attempts_limit && source.is_retryable_without_progress() =>
                {
                    recovered.push(source);
                }
                Err(source) => {
                    return Err(FrameWriteError {
                        exchange,
                        attempts_used: attempt,
                        recovered_failures: recovered,
                        encoded_frame_bytes: Some(frame_len),
                        source,
                    });
                }
            }
        }
        Err(FrameWriteError {
            exchange,
            attempts_used: 0,
            recovered_failures: recovered,
            encoded_frame_bytes: Some(frame_len),
            source: FrameWriteFailure::Transport(TransportFailure::contract_violation(
                TransportOperation::Write,
                "parsed non-zero write-attempt loop executed no attempt",
                0,
            )),
        })
    }

    async fn write_one_attempt(
        &mut self,
        bytes: &[u8],
    ) -> Result<MonotonicTimestamp, FrameWriteFailure> {
        let started_at = self.clock.now().map_err(FrameWriteFailure::Clock)?;
        let timeout = self.config.write_timeout();
        let timeout_ns = u64::try_from(timeout.get().as_nanos()).map_err(|_| {
            FrameWriteFailure::DeadlineOverflow {
                started_at_ns: started_at.nanos_since_epoch(),
                timeout,
            }
        })?;
        let deadline_ns = started_at
            .nanos_since_epoch()
            .checked_add(timeout_ns)
            .ok_or(FrameWriteFailure::DeadlineOverflow {
                started_at_ns: started_at.nanos_since_epoch(),
                timeout,
            })?;
        let mut transferred = 0_usize;

        while transferred < bytes.len() {
            let remaining = write_remaining(
                &self.clock,
                deadline_ns,
                TransportOperation::Write,
                transferred,
            )?;
            match self
                .transport
                .write_some(&bytes[transferred..], remaining)
                .await
            {
                Ok(0) => {
                    return Err(FrameWriteFailure::Transport(
                        TransportFailure::contract_violation(
                            TransportOperation::Write,
                            "transport reported a zero-progress successful write",
                            transferred,
                        ),
                    ));
                }
                Ok(written) if written <= bytes.len() - transferred => transferred += written,
                Ok(written) => {
                    return Err(FrameWriteFailure::Transport(
                        TransportFailure::contract_violation(
                            TransportOperation::Write,
                            format!(
                                "transport reported {written} bytes for capacity {}",
                                bytes.len() - transferred
                            ),
                            transferred.saturating_add(written),
                        ),
                    ));
                }
                Err(source) => {
                    if source.operation() != TransportOperation::Write {
                        return Err(FrameWriteFailure::TransportContract {
                            expected_operation: TransportOperation::Write,
                            known_total_progress: transferred
                                .saturating_add(source.bytes_transferred()),
                            source,
                        });
                    }
                    let offered = bytes.len() - transferred;
                    if source.bytes_transferred() > offered {
                        return Err(FrameWriteFailure::Transport(
                            TransportFailure::contract_violation(
                                TransportOperation::Write,
                                format!(
                                    "failed write reported {} bytes for capacity {offered}",
                                    source.bytes_transferred()
                                ),
                                transferred,
                            ),
                        ));
                    }
                    let Some(total) = transferred.checked_add(source.bytes_transferred()) else {
                        return Err(FrameWriteFailure::Transport(
                            TransportFailure::contract_violation(
                                TransportOperation::Write,
                                "write progress counter overflowed",
                                usize::MAX,
                            ),
                        ));
                    };
                    return Err(FrameWriteFailure::Transport(
                        source.with_total_progress(total),
                    ));
                }
            }
        }

        let remaining = write_remaining(
            &self.clock,
            deadline_ns,
            TransportOperation::Flush,
            transferred,
        )?;
        if let Err(source) = self.transport.flush(remaining).await {
            if source.operation() != TransportOperation::Flush || source.bytes_transferred() != 0 {
                return Err(FrameWriteFailure::TransportContract {
                    expected_operation: TransportOperation::Flush,
                    known_total_progress: transferred.saturating_add(source.bytes_transferred()),
                    source,
                });
            }
            return Err(FrameWriteFailure::Transport(
                source.with_total_progress(transferred),
            ));
        }
        let completed_at = self.clock.now().map_err(FrameWriteFailure::Clock)?;
        if completed_at.nanos_since_epoch() >= deadline_ns {
            return Err(FrameWriteFailure::Transport(TransportFailure::timed_out(
                TransportOperation::Flush,
                transferred,
            )));
        }
        Ok(completed_at)
    }
}

fn write_remaining<C: MonotonicClock>(
    clock: &C,
    deadline_ns: u64,
    operation: TransportOperation,
    transferred: usize,
) -> Result<Duration, FrameWriteFailure> {
    let now = clock.now().map_err(FrameWriteFailure::Clock)?;
    let Some(remaining_ns) = deadline_ns.checked_sub(now.nanos_since_epoch()) else {
        return Err(FrameWriteFailure::Transport(TransportFailure::timed_out(
            operation,
            transferred,
        )));
    };
    if remaining_ns == 0 {
        return Err(FrameWriteFailure::Transport(TransportFailure::timed_out(
            operation,
            transferred,
        )));
    }
    Ok(Duration::from_nanos(remaining_ns))
}
