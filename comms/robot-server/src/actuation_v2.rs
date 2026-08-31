//! Fail-closed V2 host/controller actuation boundary.
//!
//! One actor exclusively owns the configured serial device and all mutable
//! controller/session state. UDP tasks parse one datagram into a typed V2
//! request, timestamp its first receipt, submit it to that actor, and receive
//! only a terminal typed result. In particular, accepting a host command into
//! the actor is never reported as controller application.

use std::fmt;
use std::io;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[cfg(test)]
use robot_protocol::v2::OPERATOR_SUPERVISED_FOUR_PWM_MAX_COMMAND_STEP_PERCENT;
use robot_protocol::v2::{
    decode_raw_frame, AcquireControl, AcquireResult, AcquireResultCode, ActuatorConfigFingerprint,
    AppliedResult, AppliedResultCode, BeginSession, ControlEpoch, ControllerCapabilities,
    ControllerFaults, ControllerHello, ControllerReady, ControllerSessionAdmission,
    ControllerSessionClass, ControllerUid, DeadlineRelation, ForceStop, ForceStopReason, Heartbeat,
    HeartbeatPeriodMs, HostCommand, HostCommandResult, HostCommandResultCode, HostStop,
    HostStopResult, MaxAbsPwmPercent, Message, MessageKind, NeutralOutput, ObservationalOdometry,
    OutputState, PhysicalStopSemantics, PwmFrequencyHz, RawFrame, RemainingLeaseMs, RequestId,
    StatusCode, StatusQuery, StatusReport, StopResultCode, TargetBootId, TimerPwm, UartEncodeError,
    UartRecord, UartStreamDecoder, UartStreamError, V2CommandSequence, WatchdogNominalPeriodMs,
    MAX_RAW_FRAME_BYTES,
};
use robot_protocol::ControllerUptimeMsWrapping;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::UdpSocket;
use tokio::sync::{mpsc, oneshot, watch, Notify, Semaphore};
use tokio::task::{JoinHandle, JoinSet};
use tokio_serial::{
    ClearBuffer, DataBits, FlowControl, Parity, SerialPort, SerialPortBuilderExt, StopBits,
};

use crate::config::ControllerServerConfig;
#[cfg(test)]
use crate::config::ControllerServerConfigV1;
#[cfg(feature = "qualification-fault-injection")]
use crate::config::ControllerServerConfigV2;
use crate::config::CONTROLLER_SERIAL_BAUD_BPS;
use crate::deadline::{
    conservative_remaining_lease, translate_command_deadline, HeartbeatClockSample,
    TranslatedCommandDeadline,
};

const ACTOR_MAILBOX_CAPACITY: usize = 32;
const MAX_UDP_ORDINARY_EXCHANGES_IN_FLIGHT: usize = 64;
const MAX_UDP_PRIORITY_EXCHANGES_IN_FLIGHT: usize = 16;
const INACTIVE_TIMER_SLEEP: Duration = Duration::from_secs(24 * 60 * 60);
const UDP_EMISSION_QUANTIZATION_MARGIN_MS: u64 = 1;
const SERIAL_READ_TURN_BYTES: usize = robot_protocol::v2::MAX_UART_RECORD_BYTES;
const UART_RECORD_DELIMITER: [u8; 1] = [0];
#[cfg(feature = "qualification-fault-injection")]
const QUALIFICATION_PARTIAL_UART_PREFIX_BYTES: usize = 1;

/// Candidate-only, one-shot serial corruption used by the attended
/// wheels-off qualifier.
///
/// The only variant has no free-form byte count: it writes exactly one
/// non-delimiter byte from the first nonzero translated `ApplyPwm`, then the actor's
/// typed recovery path re-delimits the stream and issues `ForceStop`.
#[cfg(feature = "qualification-fault-injection")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperatorSupervisedCandidateSerialFaultInjection {
    PartialUartRecordOnFirstNonzeroCommand,
}

#[cfg(feature = "qualification-fault-injection")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct QualificationPartialUartRecordPrefix([u8; QUALIFICATION_PARTIAL_UART_PREFIX_BYTES]);

#[cfg(feature = "qualification-fault-injection")]
impl QualificationPartialUartRecordPrefix {
    fn from_encoded_record(
        record: &UartRecord,
    ) -> Result<Self, QualificationPartialUartRecordPrefixError> {
        let bytes = record.as_bytes();
        if bytes.len() <= QUALIFICATION_PARTIAL_UART_PREFIX_BYTES {
            return Err(QualificationPartialUartRecordPrefixError::RecordTooShort {
                actual_bytes: bytes.len(),
            });
        }
        let prefix = [bytes[0]];
        if prefix[0] == UART_RECORD_DELIMITER[0] {
            return Err(QualificationPartialUartRecordPrefixError::StartsWithDelimiter);
        }
        Ok(Self(prefix))
    }

    const fn as_bytes(&self) -> &[u8; QUALIFICATION_PARTIAL_UART_PREFIX_BYTES] {
        &self.0
    }
}

#[cfg(feature = "qualification-fault-injection")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QualificationPartialUartRecordPrefixError {
    RecordTooShort { actual_bytes: usize },
    StartsWithDelimiter,
}

#[cfg(feature = "qualification-fault-injection")]
impl fmt::Display for QualificationPartialUartRecordPrefixError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "encoded qualification UART record cannot supply a strict partial prefix: {self:?}"
        )
    }
}

#[cfg(feature = "qualification-fault-injection")]
impl std::error::Error for QualificationPartialUartRecordPrefixError {}

#[cfg(feature = "qualification-fault-injection")]
fn qualification_partial_uart_record_error(
    logical_record_bytes: usize,
    prefix_result: Result<(), SerialTransmitError>,
) -> SerialTransmitError {
    debug_assert!(logical_record_bytes > QUALIFICATION_PARTIAL_UART_PREFIX_BYTES);
    let prefix_outcome = match prefix_result {
        Ok(()) => QualificationPartialUartPrefixTransmitOutcome::Transmitted,
        Err(source) => QualificationPartialUartPrefixTransmitOutcome::Uncertain(Box::new(source)),
    };
    SerialTransmitError::QualificationPartialRecord {
        prefix_bytes_may_have_reached_transport: QUALIFICATION_PARTIAL_UART_PREFIX_BYTES,
        logical_record_bytes,
        prefix_outcome,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ActuationSnapshot {
    pub status: StatusCode,
    pub startup_phase: ActuationStartupPhase,
    pub fault: Option<ActuationFaultEvidence>,
    pub observed_boot_id: TargetBootId,
    pub control_epoch: Option<ControlEpoch>,
    pub output: ActuationOutputEvidence,
    pub last_sequence: Option<V2CommandSequence>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActuationStartupPhase {
    AwaitingHello,
    AwaitingStartupStopReceipt,
    AwaitingControllerReady,
    AwaitingStoppedHeartbeat,
    ReadyStopped,
    Faulted,
}

/// Typed, retained evidence for the first protocol condition that faulted the
/// current controller admission attempt.
///
/// This is deliberately part of the telemetry snapshot rather than a log-only
/// string: startup callers must be able to report why a controller was not
/// admitted even after their bounded cleanup stop changes the actor state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActuationFaultEvidence {
    ControllerHelloRejected(ControllerHello),
    ControllerReadyRejected(ControllerReady),
    PreSessionHeartbeatRejected(Heartbeat),
    SessionHeartbeatRejected(Heartbeat),
    SessionHeartbeatDidNotProgress(Heartbeat),
    HeartbeatAuthorityConflict(Heartbeat),
    ObservationalOdometryIdentityMismatch(ObservationalOdometry),
    UnexpectedControllerMessage(MessageKind),
    UnexpectedStopResult(HostStopResult),
    StartupStopNotConfirmed(HostStopResult),
    SerialFraming(UartStreamError),
    HeartbeatFreshnessExpired,
    Protocol(ForceStopReason),
}

/// Current controller-output knowledge exposed to telemetry and UI adapters.
///
/// Wire compatibility requires concrete placeholder bits in some negative V2
/// responses. Those bits never enter this API as observations: only a fresh
/// controller heartbeat can construct [`Self::Observed`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActuationOutputEvidence {
    Unknown,
    Observed(ObservedActuationOutput),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ObservedActuationOutput {
    pub controller_uptime: ControllerUptimeMsWrapping,
    pub output_state: OutputState,
    pub controller_timer_pwm: TimerPwm,
    pub faults: ControllerFaults,
}

/// A deliberately small bridge to the legacy HTTP/telemetry state.
///
/// Implementations must return promptly and must not call back into the actor.
/// Odometry remains observational and never affects actuation decisions.
pub trait ActuationTelemetry: Send + Sync + 'static {
    fn update_actuation(&self, snapshot: ActuationSnapshot);

    fn observe_odometry(&self, odometry: ObservationalOdometry, received_at: Instant);
}

#[derive(Debug, Default)]
pub struct NoopActuationTelemetry;

impl ActuationTelemetry for NoopActuationTelemetry {
    fn update_actuation(&self, _snapshot: ActuationSnapshot) {}

    fn observe_odometry(&self, _odometry: ObservationalOdometry, _received_at: Instant) {}
}

#[derive(Clone)]
pub(crate) struct ActuationHandle {
    requests: mpsc::Sender<ActorRequest>,
    priority_stop: Arc<PriorityStopCoordinator>,
    shutdown: Arc<ActuationShutdownSignal>,
}

impl ActuationHandle {
    /// Submit one typed wire message in tests. `first_received_at` must be the
    /// original receive instant and is never replaced for retries.
    #[cfg(test)]
    pub async fn exchange(
        &self,
        source: SocketAddr,
        first_received_at: Instant,
        message: Message,
    ) -> Result<Message, ActuationHandleError> {
        let request = HostRequest::try_from(message)?;
        self.exchange_timed(source, first_received_at, request)
            .await
            .map(|response| response.message)
    }

    async fn exchange_timed(
        &self,
        source: SocketAddr,
        first_received_at: Instant,
        request: HostRequest,
    ) -> Result<TimedActorResponse, ActuationHandleError> {
        if first_received_at > Instant::now() {
            return Err(ActuationHandleError::FutureReceiveInstant);
        }
        if let HostRequest::Stop(request) = request {
            return self
                .priority_stop
                .request(source, request)
                .await
                .map(|result| TimedActorResponse {
                    message: Message::HostStopResult(result),
                    calculated_at: Instant::now(),
                });
        }
        let (response, receiver) = oneshot::channel();
        let (calculated_at, calculated_at_receiver) = oneshot::channel();
        self.requests
            .send(ActorRequest {
                source,
                first_received_at,
                request,
                response: ActorResponseSender {
                    message: response,
                    calculated_at,
                },
            })
            .await
            .map_err(|_| ActuationHandleError::ActorStopped)?;
        let message = receiver
            .await
            .map_err(|_| ActuationHandleError::ResponseDropped)?;
        let calculated_at = calculated_at_receiver
            .await
            .map_err(|_| ActuationHandleError::ResponseDropped)?;
        Ok(TimedActorResponse {
            message,
            calculated_at,
        })
    }

    pub(crate) fn shutdown_handle(&self) -> ActuationShutdownHandle {
        ActuationShutdownHandle {
            signal: Arc::clone(&self.shutdown),
        }
    }

    #[cfg(test)]
    fn enqueue_for_test(
        &self,
        source: SocketAddr,
        first_received_at: Instant,
        message: Message,
    ) -> oneshot::Receiver<Message> {
        let request = HostRequest::try_from(message).expect("supported test request");
        if let HostRequest::Stop(request) = request {
            let priority_stop = Arc::clone(&self.priority_stop);
            let (response, receiver) = oneshot::channel();
            let cache_revision = match priority_stop.prepare_request(source, request) {
                PriorityStopRequestAdmission::Immediate(result) => {
                    let _ = response.send(Message::HostStopResult(result));
                    return receiver;
                }
                PriorityStopRequestAdmission::Pending { cache_revision } => cache_revision,
            };
            let pending = priority_stop
                .latch(cache_revision)
                .expect("test priority-stop generation is available");
            tokio::spawn(async move {
                if let Ok(result) = priority_stop.await_request(source, request, pending).await {
                    let _ = response.send(Message::HostStopResult(result));
                }
            });
            return receiver;
        }
        let (response, receiver) = oneshot::channel();
        let (calculated_at, _calculated_at_receiver) = oneshot::channel();
        if self
            .requests
            .try_send(ActorRequest {
                source,
                first_received_at,
                request,
                response: ActorResponseSender {
                    message: response,
                    calculated_at,
                },
            })
            .is_err()
        {
            panic!("test actor mailbox must have capacity");
        }
        receiver
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ActuationShutdownReason {
    Operator,
    SiblingFailure,
}

impl ActuationShutdownReason {
    const OPERATOR: u8 = 1;
    const SIBLING_FAILURE: u8 = 2;

    const fn encoded(self) -> u8 {
        match self {
            Self::Operator => Self::OPERATOR,
            Self::SiblingFailure => Self::SIBLING_FAILURE,
        }
    }

    const fn force_stop_reason(self) -> ForceStopReason {
        match self {
            Self::Operator => ForceStopReason::Operator,
            Self::SiblingFailure => ForceStopReason::TransportFault,
        }
    }
}

#[derive(Debug)]
struct ActuationShutdownSignal {
    reason: AtomicU8,
    notify: Notify,
}

impl ActuationShutdownSignal {
    fn new() -> Self {
        Self {
            reason: AtomicU8::new(0),
            notify: Notify::new(),
        }
    }

    fn request(&self, reason: ActuationShutdownReason) {
        let _ =
            self.reason
                .compare_exchange(0, reason.encoded(), Ordering::AcqRel, Ordering::Acquire);
        self.notify.notify_waiters();
    }

    fn requested_reason(&self) -> Option<ActuationShutdownReason> {
        match self.reason.load(Ordering::Acquire) {
            0 => None,
            ActuationShutdownReason::OPERATOR => Some(ActuationShutdownReason::Operator),
            ActuationShutdownReason::SIBLING_FAILURE => {
                Some(ActuationShutdownReason::SiblingFailure)
            }
            _ => Some(ActuationShutdownReason::SiblingFailure),
        }
    }

    async fn wait(&self) -> ActuationShutdownReason {
        loop {
            if let Some(reason) = self.requested_reason() {
                return reason;
            }
            let notified = self.notify.notified();
            if let Some(reason) = self.requested_reason() {
                return reason;
            }
            notified.await;
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ActuationShutdownHandle {
    signal: Arc<ActuationShutdownSignal>,
}

impl ActuationShutdownHandle {
    pub(crate) fn request(&self, reason: ActuationShutdownReason) {
        self.signal.request(reason);
    }
}

#[derive(Clone, Copy, Debug)]
enum PriorityStopEvidence {
    Exact(HostStopResult),
    Uncertain(StopResultCode),
}

#[derive(Clone, Copy, Debug)]
struct PriorityStopCompletion {
    through_generation: u64,
    evidence: PriorityStopEvidence,
}

impl PriorityStopCompletion {
    const INITIAL: Self = Self {
        through_generation: 0,
        evidence: PriorityStopEvidence::Uncertain(StopResultCode::ControllerUnavailable),
    };
}

/// Coalesces every concurrently waiting host stop onto one actor-owned
/// ForceStop transaction.
///
/// The actor retains no caller list: callers subscribe to the one bounded
/// shared completion cell and project the exact/uncertain controller evidence
/// back onto their own typed request identity.
#[derive(Debug)]
struct PriorityStopCoordinator {
    controller_uid: ControllerUid,
    requested_generation: AtomicU64,
    actor_running: AtomicBool,
    notify: Notify,
    completion: watch::Sender<PriorityStopCompletion>,
    cache: Mutex<PriorityStopCache>,
}

#[derive(Clone, Copy, Debug)]
struct CachedPriorityStop {
    source: SocketAddr,
    request: HostStop,
    result: HostStopResult,
}

#[derive(Debug)]
struct PriorityStopCache {
    /// `None` permanently disables caching after the revision domain is
    /// exhausted instead of permitting an ABA comparison.
    revision: Option<u64>,
    entry: Option<CachedPriorityStop>,
}

impl PriorityStopCache {
    fn invalidate(&mut self) -> Option<u64> {
        self.entry = None;
        self.revision = self.revision.and_then(|revision| revision.checked_add(1));
        self.revision
    }
}

#[derive(Clone, Copy, Debug)]
enum PriorityStopRequestAdmission {
    Immediate(HostStopResult),
    Pending { cache_revision: Option<u64> },
}

enum PreparedPriorityStopRequest {
    Immediate(HostStopResult),
    Pending(PendingPriorityStopRequest),
}

struct PendingPriorityStopRequest {
    generation: u64,
    completion: watch::Receiver<PriorityStopCompletion>,
    cache_revision: Option<u64>,
}

impl PriorityStopCoordinator {
    fn new(controller_uid: ControllerUid) -> Self {
        let (completion, _initial_receiver) = watch::channel(PriorityStopCompletion::INITIAL);
        Self {
            controller_uid,
            requested_generation: AtomicU64::new(0),
            actor_running: AtomicBool::new(true),
            notify: Notify::new(),
            completion,
            cache: Mutex::new(PriorityStopCache {
                revision: Some(0),
                entry: None,
            }),
        }
    }

    async fn request(
        &self,
        source: SocketAddr,
        request: HostStop,
    ) -> Result<HostStopResult, ActuationHandleError> {
        let prepared = self.prepare_latched_request(source, request)?;
        self.await_prepared_request(source, request, prepared).await
    }

    fn prepare_latched_request(
        &self,
        source: SocketAddr,
        request: HostStop,
    ) -> Result<PreparedPriorityStopRequest, ActuationHandleError> {
        match self.prepare_request(source, request) {
            PriorityStopRequestAdmission::Immediate(result) => {
                Ok(PreparedPriorityStopRequest::Immediate(result))
            }
            PriorityStopRequestAdmission::Pending { cache_revision } => self
                .latch(cache_revision)
                .map(PreparedPriorityStopRequest::Pending),
        }
    }

    async fn await_prepared_request(
        &self,
        source: SocketAddr,
        request: HostStop,
        prepared: PreparedPriorityStopRequest,
    ) -> Result<HostStopResult, ActuationHandleError> {
        let pending = match prepared {
            PreparedPriorityStopRequest::Immediate(result) => return Ok(result),
            PreparedPriorityStopRequest::Pending(pending) => pending,
        };
        self.await_request(source, request, pending).await
    }

    fn prepare_request(
        &self,
        source: SocketAddr,
        request: HostStop,
    ) -> PriorityStopRequestAdmission {
        if request.controller_uid != self.controller_uid {
            return PriorityStopRequestAdmission::Immediate(unproven_stop_result(
                self.controller_uid,
                request,
                StopResultCode::IdentityMismatch,
            ));
        }
        let mut cache = self
            .cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !self.actor_running.load(Ordering::Acquire) {
            return PriorityStopRequestAdmission::Pending {
                cache_revision: cache.invalidate(),
            };
        }
        if let Some(cached) = cache.entry {
            if cached.source == source && cached.request == request {
                return PriorityStopRequestAdmission::Immediate(cached.result);
            }
        }
        PriorityStopRequestAdmission::Pending {
            cache_revision: cache.invalidate(),
        }
    }

    fn latch(
        &self,
        cache_revision: Option<u64>,
    ) -> Result<PendingPriorityStopRequest, ActuationHandleError> {
        if !self.actor_running.load(Ordering::Acquire) {
            return Err(ActuationHandleError::ActorStopped);
        }

        let completion = self.completion.subscribe();
        let generation = self
            .requested_generation
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
                value.checked_add(1)
            })
            .map_err(|_| ActuationHandleError::PriorityStopGenerationExhausted)?
            + 1;
        self.notify.notify_waiters();
        if !self.actor_running.load(Ordering::Acquire) {
            return Err(ActuationHandleError::ActorStopped);
        }
        Ok(PendingPriorityStopRequest {
            generation,
            completion,
            cache_revision,
        })
    }

    async fn await_request(
        &self,
        source: SocketAddr,
        request: HostStop,
        pending: PendingPriorityStopRequest,
    ) -> Result<HostStopResult, ActuationHandleError> {
        let PendingPriorityStopRequest {
            generation,
            mut completion,
            cache_revision,
        } = pending;
        loop {
            let observed = *completion.borrow_and_update();
            if observed.through_generation >= generation {
                let result = self.project(request, observed.evidence);
                if matches!(observed.evidence, PriorityStopEvidence::Exact(_)) {
                    let mut cache = self
                        .cache
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    if cache_revision.is_some() && cache.revision == cache_revision {
                        cache.entry = Some(CachedPriorityStop {
                            source,
                            request,
                            result,
                        });
                    }
                }
                return Ok(result);
            }
            completion
                .changed()
                .await
                .map_err(|_| ActuationHandleError::ActorStopped)?;
        }
    }

    fn invalidate_cache(&self) {
        let mut cache = self
            .cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        cache.invalidate();
    }

    fn project(&self, request: HostStop, evidence: PriorityStopEvidence) -> HostStopResult {
        match evidence {
            PriorityStopEvidence::Exact(result)
                if stop_result_matches_host_request(request, result) =>
            {
                HostStopResult {
                    request_id: request.request_id,
                    ..result
                }
            }
            PriorityStopEvidence::Exact(result) => HostStopResult {
                controller_uid: self.controller_uid,
                observed_boot_id: result.observed_boot_id,
                request_id: request.request_id,
                result: StopResultCode::IdentityMismatch,
                output_state: result.output_state,
                controller_uptime: result.controller_uptime,
                faults: result.faults,
            },
            PriorityStopEvidence::Uncertain(code) => {
                unproven_stop_result(self.controller_uid, request, code)
            }
        }
    }

    fn requested_after(&self, completed_generation: u64) -> Option<u64> {
        let requested = self.requested_generation.load(Ordering::Acquire);
        (requested > completed_generation).then_some(requested)
    }

    async fn wait_after(&self, completed_generation: u64) -> u64 {
        loop {
            if let Some(requested) = self.requested_after(completed_generation) {
                return requested;
            }
            let notified = self.notify.notified();
            if let Some(requested) = self.requested_after(completed_generation) {
                return requested;
            }
            notified.await;
        }
    }

    fn publish(&self, through_generation: u64, evidence: PriorityStopEvidence) {
        if self.completion.borrow().through_generation <= through_generation {
            self.completion.send_replace(PriorityStopCompletion {
                through_generation,
                evidence,
            });
        }
    }

    fn actor_stopped(&self) {
        self.actor_running.store(false, Ordering::Release);
        self.invalidate_cache();
        let requested = self.requested_generation.load(Ordering::Acquire);
        if self.completion.borrow().through_generation < requested {
            self.publish(
                requested,
                PriorityStopEvidence::Uncertain(StopResultCode::ControllerUnavailable),
            );
        }
        self.notify.notify_waiters();
    }
}

fn stop_result_matches_host_request(request: HostStop, result: HostStopResult) -> bool {
    request.controller_uid == result.controller_uid
        && match request.target_boot_id {
            TargetBootId::Any => matches!(result.observed_boot_id, TargetBootId::Exact(_)),
            TargetBootId::Exact(expected) => {
                result.observed_boot_id == TargetBootId::Exact(expected)
            }
        }
}

fn unproven_stop_result(
    controller_uid: ControllerUid,
    request: HostStop,
    result: StopResultCode,
) -> HostStopResult {
    debug_assert!(!result.proves_controller_stop());
    // Frozen-wire placeholders only. `HostStopResult::output_evidence`
    // classifies this response as Unknown.
    HostStopResult {
        controller_uid,
        observed_boot_id: request.target_boot_id,
        request_id: request.request_id,
        result,
        output_state: OutputState::Disabled,
        controller_uptime: ControllerUptimeMsWrapping::new(0),
        faults: ControllerFaults::NONE,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HostRequest {
    Acquire(AcquireControl),
    Command(HostCommand),
    Stop(HostStop),
    Status(StatusQuery),
}

impl TryFrom<Message> for HostRequest {
    type Error = ActuationHandleError;

    fn try_from(value: Message) -> Result<Self, Self::Error> {
        match value {
            Message::AcquireControl(value) => Ok(Self::Acquire(value)),
            Message::HostCommand(value) => Ok(Self::Command(value)),
            Message::HostStop(value) => Ok(Self::Stop(value)),
            Message::StatusQuery(value) => Ok(Self::Status(value)),
            _ => Err(ActuationHandleError::UnsupportedHostMessage(value.kind())),
        }
    }
}

#[derive(Clone, Copy)]
struct TimedActorResponse {
    message: Message,
    calculated_at: Instant,
}

struct ActorResponseSender {
    message: oneshot::Sender<Message>,
    calculated_at: oneshot::Sender<Instant>,
}

impl ActorResponseSender {
    fn send(self, message: Message) -> Result<(), Message> {
        self.send_calculated_at(message, Instant::now())
    }

    fn send_calculated_at(self, message: Message, calculated_at: Instant) -> Result<(), Message> {
        let _ = self.calculated_at.send(calculated_at);
        self.message.send(message)
    }
}

struct ActorRequest {
    source: SocketAddr,
    first_received_at: Instant,
    request: HostRequest,
    response: ActorResponseSender,
}

enum ActorWake {
    Shutdown(ActuationShutdownReason),
    PriorityStop(u64),
    Timer,
    SerialRead(io::Result<usize>),
    HostRequest(Option<ActorRequest>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActuationHandleError {
    UnsupportedHostMessage(MessageKind),
    FutureReceiveInstant,
    PriorityStopGenerationExhausted,
    ActorStopped,
    ResponseDropped,
}

impl fmt::Display for ActuationHandleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedHostMessage(kind) => {
                write!(formatter, "V2 {kind:?} is not a host request")
            }
            Self::FutureReceiveInstant => {
                formatter.write_str("host request receive instant is in the future")
            }
            Self::PriorityStopGenerationExhausted => {
                formatter.write_str("priority HostStop generation space is exhausted")
            }
            Self::ActorStopped => formatter.write_str("V2 serial actor is not running"),
            Self::ResponseDropped => {
                formatter.write_str("V2 serial actor ended without a terminal response")
            }
        }
    }
}

impl std::error::Error for ActuationHandleError {}

#[derive(Debug)]
pub enum ActuationStartError {
    NonUtf8SerialDevice,
    InvalidHeartbeatPeriod,
    OpenSerial(tokio_serial::Error),
    ExclusiveSerial(tokio_serial::Error),
    ClearPendingSerialInput(tokio_serial::Error),
}

impl fmt::Display for ActuationStartError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonUtf8SerialDevice => {
                formatter.write_str("configured serial device is not valid UTF-8")
            }
            Self::InvalidHeartbeatPeriod => formatter
                .write_str("configured heartbeat period is outside the frozen V2 protocol domain"),
            Self::OpenSerial(source) => {
                write!(formatter, "cannot open configured serial device: {source}")
            }
            Self::ExclusiveSerial(source) => write!(
                formatter,
                "cannot acquire exclusive ownership of configured serial device: {source}"
            ),
            Self::ClearPendingSerialInput(source) => write!(
                formatter,
                "cannot clear bytes pending in the host serial input queue: {source}"
            ),
        }
    }
}

impl std::error::Error for ActuationStartError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::OpenSerial(source)
            | Self::ExclusiveSerial(source)
            | Self::ClearPendingSerialInput(source) => Some(source),
            Self::NonUtf8SerialDevice | Self::InvalidHeartbeatPeriod => None,
        }
    }
}

#[derive(Debug)]
pub enum ActuationActorError {
    SerialEof,
    SerialRead(io::Error),
    SerialTransmit(SerialTransmitError),
    ShutdownInterruptedTransmit {
        interrupted: SerialTransmitError,
        recovery: Box<ShutdownInterruptedTransmitRecovery>,
    },
    #[cfg(feature = "qualification-fault-injection")]
    QualificationPartialUartRecordInjected {
        interrupted: SerialTransmitError,
        recovery: Box<QualificationPartialUartRecordInjectionRecovery>,
    },
    #[cfg(feature = "qualification-fault-injection")]
    QualificationPartialUartRecordInvariant(QualificationPartialUartRecordPrefixError),
    ShutdownStopConfirmationTimedOut {
        maximum_wait: Duration,
    },
    ShutdownStopNotConfirmed {
        result: HostStopResult,
    },
    Encode(UartEncodeError),
    InternalRequestIdExhausted,
}

impl fmt::Display for ActuationActorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SerialEof => {
                formatter.write_str("configured controller serial stream reached EOF")
            }
            Self::SerialRead(source) => {
                write!(formatter, "controller serial read failed: {source}")
            }
            Self::SerialTransmit(source) => write!(formatter, "{source}"),
            Self::ShutdownInterruptedTransmit {
                interrupted,
                recovery,
            } => write!(
                formatter,
                "{interrupted}; bounded shutdown recovery preserved that uncertainty and reported {recovery}"
            ),
            #[cfg(feature = "qualification-fault-injection")]
            Self::QualificationPartialUartRecordInjected {
                interrupted,
                recovery,
            } => write!(
                formatter,
                "qualification injected {interrupted}; bounded delimiter/ForceStop recovery reported {recovery}"
            ),
            #[cfg(feature = "qualification-fault-injection")]
            Self::QualificationPartialUartRecordInvariant(source) => source.fmt(formatter),
            Self::ShutdownStopConfirmationTimedOut { maximum_wait } => write!(
                formatter,
                "shutdown ForceStop had no exact controller confirmation within {maximum_wait:?}; physical stop remains uncertain"
            ),
            Self::ShutdownStopNotConfirmed { result } => write!(
                formatter,
                "shutdown ForceStop returned an exact result that did not prove a clean controller stop: {result:?}"
            ),
            Self::Encode(source) => {
                write!(formatter, "cannot encode typed V2 UART record: {source}")
            }
            Self::InternalRequestIdExhausted => {
                formatter.write_str("server internal V2 request-ID space is exhausted")
            }
        }
    }
}

impl std::error::Error for ActuationActorError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::SerialRead(source) => Some(source),
            Self::SerialTransmit(source) => Some(source),
            Self::ShutdownInterruptedTransmit { interrupted, .. } => Some(interrupted),
            #[cfg(feature = "qualification-fault-injection")]
            Self::QualificationPartialUartRecordInjected { interrupted, .. } => Some(interrupted),
            #[cfg(feature = "qualification-fault-injection")]
            Self::QualificationPartialUartRecordInvariant(source) => Some(source),
            Self::Encode(source) => Some(source),
            Self::SerialEof
            | Self::ShutdownStopConfirmationTimedOut { .. }
            | Self::ShutdownStopNotConfirmed { .. }
            | Self::InternalRequestIdExhausted => None,
        }
    }
}

/// Whether a partial UART record was explicitly re-delimited before the
/// shutdown ForceStop was attempted.
#[derive(Debug)]
pub enum SerialResynchronizationOutcome {
    NotRequired,
    DelimiterTransmitted,
    Failed(SerialTransmitError),
}

/// Exact terminal evidence from the ForceStop half of shutdown recovery.
///
/// `Confirmed` requires a matching ForceStop receipt, a stop-proving result,
/// and a safe output state. Controller fault bits are retained in the result:
/// a framing fault caused by re-delimiting a partial record must not be
/// rewritten into a clean-shutdown claim.
#[derive(Debug)]
pub enum ShutdownForceStopOutcome {
    Confirmed(HostStopResult),
    ExactButUnconfirmed(HostStopResult),
    Uncertain(Box<ActuationActorError>),
}

/// Both bounded operations attempted after shutdown interrupted a serial
/// transmission. The original interrupted transmit remains separately
/// present in [`ActuationActorError::ShutdownInterruptedTransmit`].
#[derive(Debug)]
pub struct ShutdownInterruptedTransmitRecovery {
    resynchronization: SerialResynchronizationOutcome,
    force_stop: ShutdownForceStopOutcome,
}

impl ShutdownInterruptedTransmitRecovery {
    pub const fn resynchronization(&self) -> &SerialResynchronizationOutcome {
        &self.resynchronization
    }

    pub const fn force_stop(&self) -> &ShutdownForceStopOutcome {
        &self.force_stop
    }
}

impl fmt::Display for ShutdownInterruptedTransmitRecovery {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "resynchronization={:?}, force_stop={:?}",
            self.resynchronization, self.force_stop
        )
    }
}

/// Exact recovery evidence after the qualification actor intentionally wrote
/// one byte of a nonzero command record.
#[cfg(feature = "qualification-fault-injection")]
#[derive(Debug)]
pub struct QualificationPartialUartRecordInjectionRecovery {
    resynchronization: SerialResynchronizationOutcome,
    force_stop: ShutdownForceStopOutcome,
}

#[cfg(feature = "qualification-fault-injection")]
impl QualificationPartialUartRecordInjectionRecovery {
    pub const fn resynchronization(&self) -> &SerialResynchronizationOutcome {
        &self.resynchronization
    }

    pub const fn force_stop(&self) -> &ShutdownForceStopOutcome {
        &self.force_stop
    }
}

#[cfg(feature = "qualification-fault-injection")]
impl fmt::Display for QualificationPartialUartRecordInjectionRecovery {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "resynchronization={:?}, force_stop={:?}",
            self.resynchronization, self.force_stop
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SerialTransmitPhase {
    Write,
    Flush,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SerialTransmitInterruption {
    DeadlineExceeded,
    ShutdownRequested,
    PriorityStopRequested,
}

/// What the serial API reported after the qualification actor attempted its
/// one-byte unterminated prefix.
///
/// A write error cannot prove that the byte did not cross the host/driver
/// boundary, so every failure is retained as uncertain. Recovery therefore
/// always treats the logical command record as possibly started.
#[cfg(feature = "qualification-fault-injection")]
#[derive(Debug)]
pub enum QualificationPartialUartPrefixTransmitOutcome {
    Transmitted,
    Uncertain(Box<SerialTransmitError>),
}

#[cfg(feature = "qualification-fault-injection")]
impl fmt::Display for QualificationPartialUartPrefixTransmitOutcome {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Transmitted => formatter.write_str("prefix transport completed"),
            Self::Uncertain(source) => write!(formatter, "prefix transport uncertain: {source}"),
        }
    }
}

#[derive(Debug)]
pub enum SerialTransmitError {
    Write {
        source: io::Error,
        written_bytes: usize,
        record_bytes: usize,
    },
    Flush {
        source: io::Error,
        record_bytes: usize,
    },
    Interrupted {
        phase: SerialTransmitPhase,
        cause: SerialTransmitInterruption,
        written_bytes: usize,
        record_bytes: usize,
        maximum_duration: Duration,
    },
    #[cfg(feature = "qualification-fault-injection")]
    QualificationPartialRecord {
        prefix_bytes_may_have_reached_transport: usize,
        logical_record_bytes: usize,
        prefix_outcome: QualificationPartialUartPrefixTransmitOutcome,
    },
}

impl fmt::Display for SerialTransmitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Write {
                source,
                written_bytes,
                record_bytes,
            } => write!(
                formatter,
                "controller serial write failed after {written_bytes}/{record_bytes} record bytes: {source}; controller receipt and stop state are uncertain"
            ),
            Self::Flush {
                source,
                record_bytes,
            } => write!(
                formatter,
                "controller serial flush failed after writing all {record_bytes} record bytes: {source}; controller receipt and stop state are uncertain"
            ),
            Self::Interrupted {
                phase,
                cause,
                written_bytes,
                record_bytes,
                maximum_duration,
            } => write!(
                formatter,
                "controller serial {phase:?} was interrupted by {cause:?} after {written_bytes}/{record_bytes} record bytes (transmit bound {maximum_duration:?}); controller receipt and stop state are uncertain"
            ),
            #[cfg(feature = "qualification-fault-injection")]
            Self::QualificationPartialRecord {
                prefix_bytes_may_have_reached_transport,
                logical_record_bytes,
                prefix_outcome,
            } => write!(
                formatter,
                "qualification UART injection attempted an unterminated {prefix_bytes_may_have_reached_transport}-byte prefix of a {logical_record_bytes}-byte logical command record; the prefix may have reached the transport ({prefix_outcome}), so controller receipt and stop state are uncertain"
            ),
        }
    }
}

impl std::error::Error for SerialTransmitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Write { source, .. } | Self::Flush { source, .. } => Some(source),
            Self::Interrupted { .. } => None,
            #[cfg(feature = "qualification-fault-injection")]
            Self::QualificationPartialRecord {
                prefix_outcome: QualificationPartialUartPrefixTransmitOutcome::Uncertain(source),
                ..
            } => Some(source.as_ref()),
            #[cfg(feature = "qualification-fault-injection")]
            Self::QualificationPartialRecord {
                prefix_outcome: QualificationPartialUartPrefixTransmitOutcome::Transmitted,
                ..
            } => None,
        }
    }
}

impl SerialTransmitError {
    const fn interrupted_by_shutdown(&self) -> bool {
        matches!(
            self,
            Self::Interrupted {
                cause: SerialTransmitInterruption::ShutdownRequested,
                ..
            }
        )
    }

    const fn interrupted_by_priority_stop(&self) -> bool {
        matches!(
            self,
            Self::Interrupted {
                cause: SerialTransmitInterruption::PriorityStopRequested,
                ..
            }
        )
    }

    const fn left_partial_record(&self) -> bool {
        match self {
            Self::Interrupted {
                phase: SerialTransmitPhase::Write,
                written_bytes,
                record_bytes,
                ..
            } => *written_bytes > 0 && *written_bytes < *record_bytes,
            #[cfg(feature = "qualification-fault-injection")]
            Self::QualificationPartialRecord {
                prefix_bytes_may_have_reached_transport,
                logical_record_bytes,
                ..
            } => {
                *prefix_bytes_may_have_reached_transport > 0
                    && *prefix_bytes_may_have_reached_transport < *logical_record_bytes
            }
            Self::Write { .. } | Self::Flush { .. } | Self::Interrupted { .. } => false,
        }
    }
}

/// Open only the configured device, claim OS-level exclusive ownership, and
/// spawn the sole serial/session actor.
pub(crate) type StartedActuationActor = (
    ActuationHandle,
    oneshot::Receiver<()>,
    JoinHandle<Result<(), ActuationActorError>>,
);

pub(crate) async fn start_serial_actor(
    config: ControllerServerConfig,
    telemetry: Arc<dyn ActuationTelemetry>,
) -> Result<StartedActuationActor, ActuationStartError> {
    let actor_config = ActorConfig::from_server_config(&config)?;
    start_serial_actor_inner_with_config(config, telemetry, actor_config).await
}

#[cfg(feature = "qualification-fault-injection")]
pub(crate) async fn start_candidate_serial_actor_with_fault(
    config: ControllerServerConfigV2,
    telemetry: Arc<dyn ActuationTelemetry>,
    fault: OperatorSupervisedCandidateSerialFaultInjection,
) -> Result<StartedActuationActor, ActuationStartError> {
    let config = ControllerServerConfig::from(config);
    let actor_config = ActorConfig::from_server_config_with_fault(&config, Some(fault))?;
    start_serial_actor_inner_with_config(config, telemetry, actor_config).await
}

async fn start_serial_actor_inner_with_config(
    config: ControllerServerConfig,
    telemetry: Arc<dyn ActuationTelemetry>,
    actor_config: ActorConfig,
) -> Result<StartedActuationActor, ActuationStartError> {
    let device = config
        .serial_device()
        .to_str()
        .ok_or(ActuationStartError::NonUtf8SerialDevice)?;
    let mut port = tokio_serial::new(device, CONTROLLER_SERIAL_BAUD_BPS)
        .data_bits(DataBits::Eight)
        .parity(Parity::None)
        .stop_bits(StopBits::One)
        .flow_control(FlowControl::None)
        .open_native_async()
        .map_err(ActuationStartError::OpenSerial)?;
    port.set_exclusive(true)
        .map_err(ActuationStartError::ExclusiveSerial)?;
    port.clear(ClearBuffer::Input)
        .map_err(ActuationStartError::ClearPendingSerialInput)?;
    Ok(spawn_actor(
        port,
        actor_config,
        telemetry,
        UartStreamDecoder::new_at_unknown_record_offset(),
    ))
}

fn spawn_actor<Transport>(
    transport: Transport,
    config: ActorConfig,
    telemetry: Arc<dyn ActuationTelemetry>,
    decoder: UartStreamDecoder,
) -> StartedActuationActor
where
    Transport: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    let (requests, receiver) = mpsc::channel(ACTOR_MAILBOX_CAPACITY);
    let shutdown = Arc::new(ActuationShutdownSignal::new());
    let priority_stop = Arc::new(PriorityStopCoordinator::new(config.controller_uid));
    let handle = ActuationHandle {
        requests,
        priority_stop: Arc::clone(&priority_stop),
        shutdown: Arc::clone(&shutdown),
    };
    let (startup_ready, startup_ready_rx) = oneshot::channel();
    let task = tokio::spawn(async move {
        SerialActor::new(transport, decoder, config, telemetry, startup_ready)
            .run(receiver, priority_stop, shutdown)
            .await
    });
    (handle, startup_ready_rx, task)
}

#[cfg(test)]
pub(crate) fn spawn_actor_for_test<Transport>(
    transport: Transport,
    config: &ControllerServerConfigV1,
    telemetry: Arc<dyn ActuationTelemetry>,
) -> Result<StartedActuationActor, ActuationStartError>
where
    Transport: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    Ok(spawn_actor(
        transport,
        ActorConfig::from_server_config(&ControllerServerConfig::from(config.clone()))?,
        telemetry,
        UartStreamDecoder::new(),
    ))
}

#[derive(Clone, Copy)]
struct ActorConfig {
    controller_uid: ControllerUid,
    firmware_abi: u16,
    firmware_build_id: u32,
    actuator_config_fingerprint: ActuatorConfigFingerprint,
    heartbeat_period: HeartbeatPeriodMs,
    maximum_heartbeat_age: Duration,
    minimum_host_command_interval: Duration,
    serial_transmit_timeout: Duration,
    serial_applied_ack_timeout: Duration,
    controller_clock_abs_error_ppm_bound: std::num::NonZeroU32,
    deadline_quantization_margin_ms: std::num::NonZeroU16,
    expected_max_abs_pwm_percent: MaxAbsPwmPercent,
    expected_pwm_frequency: PwmFrequencyHz,
    expected_watchdog_nominal_period: WatchdogNominalPeriodMs,
    expected_neutral_output: NeutralOutput,
    expected_physical_stop_semantics: PhysicalStopSemantics,
    controller_session_class: ControllerSessionClass,
    maximum_command_step_percent: Option<u8>,
    #[cfg(feature = "qualification-fault-injection")]
    qualification_serial_fault: Option<OperatorSupervisedCandidateSerialFaultInjection>,
}

impl ActorConfig {
    #[cfg(feature = "qualification-fault-injection")]
    fn from_server_config(config: &ControllerServerConfig) -> Result<Self, ActuationStartError> {
        Self::from_server_config_with_fault(config, None)
    }

    #[cfg(feature = "qualification-fault-injection")]
    fn from_server_config_with_fault(
        config: &ControllerServerConfig,
        qualification_serial_fault: Option<OperatorSupervisedCandidateSerialFaultInjection>,
    ) -> Result<Self, ActuationStartError> {
        let heartbeat_ms = u16::try_from(config.heartbeat_period().as_millis())
            .map_err(|_| ActuationStartError::InvalidHeartbeatPeriod)?;
        let heartbeat_period = HeartbeatPeriodMs::try_new(heartbeat_ms)
            .map_err(|_| ActuationStartError::InvalidHeartbeatPeriod)?;
        Ok(Self {
            controller_uid: config.controller_uid(),
            firmware_abi: config.firmware_abi().get(),
            firmware_build_id: config.firmware_build_id().get(),
            actuator_config_fingerprint: config.actuator_config_fingerprint(),
            heartbeat_period,
            maximum_heartbeat_age: config.maximum_heartbeat_age(),
            minimum_host_command_interval: config.minimum_host_command_interval(),
            serial_transmit_timeout: config.serial_transmit_timeout(),
            serial_applied_ack_timeout: config.serial_applied_ack_timeout(),
            controller_clock_abs_error_ppm_bound: config.controller_clock_abs_error_ppm_bound(),
            deadline_quantization_margin_ms: config.deadline_quantization_margin_ms(),
            expected_max_abs_pwm_percent: config.expected_max_abs_pwm_percent(),
            expected_pwm_frequency: config.expected_pwm_frequency(),
            expected_watchdog_nominal_period: config.expected_watchdog_nominal_period(),
            expected_neutral_output: config.expected_neutral_output(),
            expected_physical_stop_semantics: config.expected_physical_stop_semantics(),
            controller_session_class: config.controller_session_class(),
            maximum_command_step_percent: config.maximum_command_step_percent(),
            qualification_serial_fault,
        })
    }

    #[cfg(not(feature = "qualification-fault-injection"))]
    fn from_server_config(config: &ControllerServerConfig) -> Result<Self, ActuationStartError> {
        let heartbeat_ms = u16::try_from(config.heartbeat_period().as_millis())
            .map_err(|_| ActuationStartError::InvalidHeartbeatPeriod)?;
        let heartbeat_period = HeartbeatPeriodMs::try_new(heartbeat_ms)
            .map_err(|_| ActuationStartError::InvalidHeartbeatPeriod)?;
        Ok(Self {
            controller_uid: config.controller_uid(),
            firmware_abi: config.firmware_abi().get(),
            firmware_build_id: config.firmware_build_id().get(),
            actuator_config_fingerprint: config.actuator_config_fingerprint(),
            heartbeat_period,
            maximum_heartbeat_age: config.maximum_heartbeat_age(),
            minimum_host_command_interval: config.minimum_host_command_interval(),
            serial_transmit_timeout: config.serial_transmit_timeout(),
            serial_applied_ack_timeout: config.serial_applied_ack_timeout(),
            controller_clock_abs_error_ppm_bound: config.controller_clock_abs_error_ppm_bound(),
            deadline_quantization_margin_ms: config.deadline_quantization_margin_ms(),
            expected_max_abs_pwm_percent: config.expected_max_abs_pwm_percent(),
            expected_pwm_frequency: config.expected_pwm_frequency(),
            expected_watchdog_nominal_period: config.expected_watchdog_nominal_period(),
            expected_neutral_output: config.expected_neutral_output(),
            expected_physical_stop_semantics: config.expected_physical_stop_semantics(),
            controller_session_class: config.controller_session_class(),
            maximum_command_step_percent: config.maximum_command_step_percent(),
        })
    }

    const fn expected_session_admission(self) -> ControllerSessionAdmission {
        match self.controller_session_class {
            ControllerSessionClass::OperatorSupervisedFourPwmCandidate => {
                ControllerSessionAdmission::OperatorSupervisedFourPwmCandidate
            }
            ControllerSessionClass::AttendedWheelOnCommissioning => {
                ControllerSessionAdmission::AttendedWheelOnCommissioning
            }
            ControllerSessionClass::ProductionExternalInterlocks => {
                ControllerSessionAdmission::ProductionExternalInterlocks
            }
        }
    }
}

fn candidate_command_step_is_admitted(
    previously_applied: TimerPwm,
    requested: TimerPwm,
    maximum_step_percent: Option<u8>,
) -> bool {
    if requested.is_zero() {
        return true;
    }
    let Some(maximum_step_percent) = maximum_step_percent else {
        return true;
    };
    i16::from(previously_applied.left().get()).abs_diff(i16::from(requested.left().get()))
        <= u16::from(maximum_step_percent)
        && i16::from(previously_applied.right().get()).abs_diff(i16::from(requested.right().get()))
            <= u16::from(maximum_step_percent)
}

#[derive(Clone, Copy)]
struct TimedHello {
    message: ControllerHello,
}

#[derive(Clone, Copy)]
struct ReadySession {
    message: ControllerReady,
    received_at: Instant,
}

#[derive(Clone, Copy)]
struct TimedHeartbeat {
    message: Heartbeat,
    received_at: Instant,
}

#[derive(Clone, Copy)]
struct CachedCommand {
    source: SocketAddr,
    command: HostCommand,
    controller_result: AppliedResult,
    host_result: HostCommandResult,
    completed_at: Instant,
    server_deadline_exclusive: Instant,
}

struct Owner {
    source: SocketAddr,
    epoch: ControlEpoch,
    next_sequence: Option<V2CommandSequence>,
    cached: Option<CachedCommand>,
    last_serial_command_at: Option<Instant>,
}

struct PendingCommand {
    source: SocketAddr,
    command: HostCommand,
    translated: TranslatedCommandDeadline,
    controller_uptime_reference: ControllerUptimeMsWrapping,
    serial_sent_at: Instant,
    response: ActorResponseSender,
}

struct PendingStop {
    source: SocketAddr,
    request: HostStop,
    serial_request_id: RequestId,
    serial_sent_at: Instant,
    response: ActorResponseSender,
}

struct PendingPriorityStop {
    force_stop: ForceStop,
    covers_through_generation: u64,
    serial_sent_at: Instant,
}

enum PendingOperation {
    Command(PendingCommand),
    Stop(PendingStop),
    PriorityStop(PendingPriorityStop),
}

#[derive(Clone, Copy)]
struct CachedStop {
    source: SocketAddr,
    request: HostStop,
    result: HostStopResult,
}

struct SerialActor<Transport> {
    transport: Transport,
    decoder: UartStreamDecoder,
    config: ActorConfig,
    telemetry: Arc<dyn ActuationTelemetry>,
    startup_ready: Option<oneshot::Sender<()>>,
    observed_hello: Option<TimedHello>,
    hello: Option<TimedHello>,
    ready: Option<ReadySession>,
    heartbeat: Option<TimedHeartbeat>,
    owner: Option<Owner>,
    pending: Option<PendingOperation>,
    cached_stop: Option<CachedStop>,
    last_internal_stop: Option<ForceStop>,
    startup_begin_after_stop: Option<BeginSession>,
    next_internal_request_id: Option<RequestId>,
    faulted: bool,
    last_fault: Option<ActuationFaultEvidence>,
    priority_stop: Option<Arc<PriorityStopCoordinator>>,
    priority_stop_generation: u64,
    shutdown: Option<Arc<ActuationShutdownSignal>>,
    shutdown_in_progress: bool,
}

impl<Transport> SerialActor<Transport>
where
    Transport: AsyncRead + AsyncWrite + Unpin,
{
    fn new(
        transport: Transport,
        decoder: UartStreamDecoder,
        config: ActorConfig,
        telemetry: Arc<dyn ActuationTelemetry>,
        startup_ready: oneshot::Sender<()>,
    ) -> Self {
        Self {
            transport,
            decoder,
            config,
            telemetry,
            startup_ready: Some(startup_ready),
            observed_hello: None,
            hello: None,
            ready: None,
            heartbeat: None,
            owner: None,
            pending: None,
            cached_stop: None,
            last_internal_stop: None,
            startup_begin_after_stop: None,
            next_internal_request_id: Some(RequestId::new(0)),
            faulted: false,
            last_fault: None,
            priority_stop: None,
            priority_stop_generation: 0,
            shutdown: None,
            shutdown_in_progress: false,
        }
    }

    async fn run(
        &mut self,
        mut requests: mpsc::Receiver<ActorRequest>,
        priority_stop: Arc<PriorityStopCoordinator>,
        shutdown: Arc<ActuationShutdownSignal>,
    ) -> Result<(), ActuationActorError> {
        self.priority_stop = Some(Arc::clone(&priority_stop));
        self.shutdown = Some(Arc::clone(&shutdown));
        self.publish_snapshot(Instant::now());
        let result = loop {
            match self
                .run_until_exit(
                    &mut requests,
                    Arc::clone(&priority_stop),
                    Arc::clone(&shutdown),
                )
                .await
            {
                Err(ActuationActorError::SerialTransmit(interrupted))
                    if interrupted.interrupted_by_priority_stop() =>
                {
                    if let Err(error) = self
                        .recover_priority_stop_interrupted_transmit(interrupted)
                        .await
                    {
                        break Err(error);
                    }
                }
                Err(ActuationActorError::SerialTransmit(interrupted))
                    if interrupted.interrupted_by_shutdown() =>
                {
                    let reason = shutdown
                        .requested_reason()
                        .unwrap_or(ActuationShutdownReason::SiblingFailure);
                    break Err(self
                        .recover_shutdown_interrupted_transmit(interrupted, reason)
                        .await);
                }
                #[cfg(feature = "qualification-fault-injection")]
                Err(ActuationActorError::SerialTransmit(interrupted))
                    if matches!(
                        interrupted,
                        SerialTransmitError::QualificationPartialRecord { .. }
                    ) =>
                {
                    break Err(self
                        .recover_qualification_partial_uart_record(interrupted)
                        .await);
                }
                result => break result,
            }
        };
        priority_stop.actor_stopped();
        result
    }

    async fn run_until_exit(
        &mut self,
        requests: &mut mpsc::Receiver<ActorRequest>,
        priority_stop: Arc<PriorityStopCoordinator>,
        shutdown: Arc<ActuationShutdownSignal>,
    ) -> Result<(), ActuationActorError> {
        let mut read_buffer = [0_u8; SERIAL_READ_TURN_BYTES];
        let mut prefer_host_request = false;

        loop {
            if let Some(reason) = shutdown.requested_reason() {
                return self.handle_shutdown(reason).await;
            }
            if let Some(generation) = priority_stop.requested_after(self.priority_stop_generation) {
                self.handle_priority_stop_generation(generation).await?;
                continue;
            }
            let wake_at = self.next_wake_at().unwrap_or_else(|| {
                Instant::now()
                    .checked_add(INACTIVE_TIMER_SLEEP)
                    .unwrap_or_else(Instant::now)
            });
            if Instant::now() >= wake_at {
                self.handle_timer(Instant::now()).await?;
                continue;
            }
            let sleep = tokio::time::sleep_until(tokio::time::Instant::from_std(wake_at));
            tokio::pin!(sleep);
            let owner_shutdown = shutdown.wait();
            tokio::pin!(owner_shutdown);
            let priority_stop_requested = priority_stop.wait_after(self.priority_stop_generation);
            tokio::pin!(priority_stop_requested);

            let wake = if prefer_host_request {
                tokio::select! {
                    biased;
                    reason = &mut owner_shutdown => ActorWake::Shutdown(reason),
                    generation = &mut priority_stop_requested => ActorWake::PriorityStop(generation),
                    _ = &mut sleep => ActorWake::Timer,
                    request = requests.recv() => ActorWake::HostRequest(request),
                    read = self.transport.read(&mut read_buffer) => ActorWake::SerialRead(read),
                }
            } else {
                tokio::select! {
                    biased;
                    reason = &mut owner_shutdown => ActorWake::Shutdown(reason),
                    generation = &mut priority_stop_requested => ActorWake::PriorityStop(generation),
                    _ = &mut sleep => ActorWake::Timer,
                    read = self.transport.read(&mut read_buffer) => ActorWake::SerialRead(read),
                    request = requests.recv() => ActorWake::HostRequest(request),
                }
            };

            match wake {
                ActorWake::Shutdown(reason) => return self.handle_shutdown(reason).await,
                ActorWake::PriorityStop(generation) => {
                    self.handle_priority_stop_generation(generation).await?;
                    self.publish_snapshot(Instant::now());
                }
                ActorWake::Timer => self.handle_timer(Instant::now()).await?,
                ActorWake::HostRequest(request) => {
                    prefer_host_request = false;
                    let Some(request) = request else {
                        self.fail_all_pending(
                            HostCommandResultCode::ForceStopped,
                            StopResultCode::ControllerUnavailable,
                        );
                        self.issue_internal_stop(ForceStopReason::TransportFault)
                            .await?;
                        self.clear_authority(true);
                        return Ok(());
                    };
                    self.enforce_freshness(Instant::now()).await?;
                    self.handle_host_request(request).await?;
                    self.publish_snapshot(Instant::now());
                }
                ActorWake::SerialRead(read) => {
                    prefer_host_request = true;
                    let count = match read {
                        Ok(value) => value,
                        Err(source) => {
                            self.fail_all_pending(
                                HostCommandResultCode::ForceStopped,
                                StopResultCode::ControllerUnavailable,
                            );
                            if let Err(stop_error) = self
                                .issue_internal_stop(ForceStopReason::TransportFault)
                                .await
                            {
                                log::error!(
                                    "controller serial read failed and the bounded stop attempt also failed: {stop_error}"
                                );
                            }
                            self.clear_authority(true);
                            return Err(ActuationActorError::SerialRead(source));
                        }
                    };
                    if count == 0 {
                        self.fail_all_pending(
                            HostCommandResultCode::ForceStopped,
                            StopResultCode::ControllerUnavailable,
                        );
                        if let Err(stop_error) = self
                            .issue_internal_stop(ForceStopReason::TransportFault)
                            .await
                        {
                            log::error!(
                                "controller serial EOF prevented a confirmed stop; bounded stop attempt failed: {stop_error}"
                            );
                        }
                        self.clear_authority(true);
                        return Err(ActuationActorError::SerialEof);
                    }
                    for &byte in &read_buffer[..count] {
                        if let Some(decoded) = self.decoder.push(byte) {
                            let received_at = Instant::now();
                            match decoded {
                                Ok(message) => {
                                    self.handle_serial_message(message, received_at).await?
                                }
                                Err(error) => {
                                    self.handle_framing_fault(error).await?;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn next_wake_at(&self) -> Option<Instant> {
        let pending = self.pending.as_ref().and_then(|pending| {
            let sent_at = match pending {
                PendingOperation::Command(value) => value.serial_sent_at,
                PendingOperation::Stop(value) => value.serial_sent_at,
                PendingOperation::PriorityStop(value) => value.serial_sent_at,
            };
            sent_at.checked_add(self.config.serial_applied_ack_timeout)
        });
        let heartbeat = self.ready.and_then(|ready| {
            self.heartbeat
                .map_or(ready.received_at, |value| value.received_at)
                .checked_add(self.config.maximum_heartbeat_age)
        });
        match (pending, heartbeat) {
            (Some(left), Some(right)) => Some(left.min(right)),
            (Some(value), None) | (None, Some(value)) => Some(value),
            (None, None) => None,
        }
    }

    async fn handle_timer(&mut self, now: Instant) -> Result<(), ActuationActorError> {
        let pending_expired = self.pending.as_ref().is_some_and(|pending| {
            let sent_at = match pending {
                PendingOperation::Command(value) => value.serial_sent_at,
                PendingOperation::Stop(value) => value.serial_sent_at,
                PendingOperation::PriorityStop(value) => value.serial_sent_at,
            };
            now.checked_duration_since(sent_at)
                .is_none_or(|age| age >= self.config.serial_applied_ack_timeout)
        });
        if pending_expired {
            match self.pending.take() {
                Some(PendingOperation::Command(value)) => {
                    let result = self.failed_command_result(
                        value.command,
                        HostCommandResultCode::AppliedAckTimeout,
                    );
                    let _ = value.response.send(Message::HostCommandResult(result));
                }
                Some(PendingOperation::Stop(value)) => {
                    let result =
                        self.failed_stop_result(value.request, StopResultCode::StopAckTimeout);
                    let _ = value.response.send(Message::HostStopResult(result));
                }
                Some(PendingOperation::PriorityStop(value)) => self.publish_priority_stop(
                    value.covers_through_generation,
                    PriorityStopEvidence::Uncertain(StopResultCode::StopAckTimeout),
                ),
                None => {}
            }
            self.issue_internal_stop(ForceStopReason::TransportFault)
                .await?;
            self.clear_authority(false);
        }
        self.enforce_freshness(now).await
    }

    async fn enforce_freshness(&mut self, now: Instant) -> Result<(), ActuationActorError> {
        let Some(ready) = self.ready else {
            return Ok(());
        };
        let reference = self
            .heartbeat
            .map_or(ready.received_at, |heartbeat| heartbeat.received_at);
        let stale = now
            .checked_duration_since(reference)
            .is_none_or(|age| age >= self.config.maximum_heartbeat_age);
        if stale {
            self.fail_all_pending(
                HostCommandResultCode::ForceStopped,
                StopResultCode::ControllerFaulted,
            );
            self.issue_internal_stop(ForceStopReason::TransportFault)
                .await?;
            self.ready = None;
            self.heartbeat = None;
            self.owner = None;
            self.faulted = true;
            self.last_fault
                .get_or_insert(ActuationFaultEvidence::HeartbeatFreshnessExpired);
            self.publish_snapshot(now);
        }
        Ok(())
    }

    async fn handle_host_request(
        &mut self,
        request: ActorRequest,
    ) -> Result<(), ActuationActorError> {
        match request.request {
            HostRequest::Acquire(value) => {
                let result = self.acquire_result(request.source, value, Instant::now());
                let _ = request.response.send(Message::AcquireResult(result));
            }
            HostRequest::Command(value) => {
                self.handle_host_command(
                    request.source,
                    request.first_received_at,
                    value,
                    request.response,
                )
                .await?;
            }
            HostRequest::Stop(value) => {
                self.handle_host_stop(request.source, value, request.response)
                    .await?;
            }
            HostRequest::Status(value) => {
                let calculated_at = Instant::now();
                let result = self.status_report(value, calculated_at);
                let _ = request
                    .response
                    .send_calculated_at(Message::StatusReport(result), calculated_at);
            }
        }
        Ok(())
    }

    async fn handle_priority_stop_generation(
        &mut self,
        generation: u64,
    ) -> Result<(), ActuationActorError> {
        self.priority_stop_generation = self.priority_stop_generation.max(generation);
        if let Some(PendingOperation::PriorityStop(pending)) = &mut self.pending {
            pending.covers_through_generation = pending
                .covers_through_generation
                .max(self.priority_stop_generation);
            return Ok(());
        }

        self.fail_all_pending(
            HostCommandResultCode::ForceStopped,
            StopResultCode::ControllerUnavailable,
        );
        self.owner = None;
        self.heartbeat = None;
        self.ready = None;
        self.cached_stop = None;

        let target_boot_id = self.observed_hello.map_or(TargetBootId::Any, |hello| {
            TargetBootId::Exact(hello.message.boot_id)
        });
        let force_stop = ForceStop {
            controller_uid: self.config.controller_uid,
            target_boot_id,
            request_id: match self.allocate_internal_request_id() {
                Ok(request_id) => request_id,
                Err(error) => {
                    self.publish_priority_stop(
                        self.priority_stop_generation,
                        PriorityStopEvidence::Uncertain(StopResultCode::ControllerUnavailable),
                    );
                    return Err(error);
                }
            },
            reason: ForceStopReason::Operator,
        };
        let serial_sent_at = match self.send_serial(Message::ForceStop(force_stop)).await {
            Ok(sent_at) => sent_at,
            Err(error) => {
                self.publish_priority_stop(
                    self.priority_stop_generation,
                    PriorityStopEvidence::Uncertain(StopResultCode::ControllerUnavailable),
                );
                self.clear_authority(true);
                return Err(error);
            }
        };
        self.pending = Some(PendingOperation::PriorityStop(PendingPriorityStop {
            force_stop,
            covers_through_generation: self.priority_stop_generation,
            serial_sent_at,
        }));
        Ok(())
    }

    fn acquire_result(
        &mut self,
        source: SocketAddr,
        request: AcquireControl,
        now: Instant,
    ) -> AcquireResult {
        let observed = self.observed_hello.map(|value| value.message);
        let identity_exact = request.expected_controller_uid == self.config.controller_uid
            && request.expected_firmware_abi == self.config.firmware_abi
            && request.expected_firmware_build_id == self.config.firmware_build_id
            && request.expected_actuator_config_fingerprint
                == self.config.actuator_config_fingerprint;
        let current_exact = self
            .hello
            .is_some_and(|hello| hello.message.boot_id == request.expected_boot_id);
        let ready = self.ready;
        let fresh_zero = self.heartbeat.is_some_and(|heartbeat| {
            heartbeat.message.timer_pwm.is_zero()
                && heartbeat.message.output_state.is_safe()
                && heartbeat.message.faults.is_clear()
                && heartbeat.message.control_epoch.is_none()
                && heartbeat.message.last_sequence.is_none()
                && heartbeat
                    .message
                    .readiness
                    .is_stopped_ready_for_session(self.config.controller_session_class)
                && matches!(
                    heartbeat
                        .message
                        .expires_at
                        .relation_to(heartbeat.message.controller_uptime),
                    DeadlineRelation::Expired
                )
                && now
                    .checked_duration_since(heartbeat.received_at)
                    .is_some_and(|age| age < self.config.maximum_heartbeat_age)
        });

        let (result, control_epoch) = if !identity_exact || !current_exact {
            (AcquireResultCode::IdentityMismatch, None)
        } else if self.faulted || !self.current_faults().is_clear() {
            (AcquireResultCode::Faulted, None)
        } else if let Some(owner) = &self.owner {
            if owner.source == source {
                (AcquireResultCode::Granted, Some(owner.epoch))
            } else {
                (AcquireResultCode::Busy, None)
            }
        } else if !fresh_zero {
            (AcquireResultCode::NotReady, None)
        } else if let Some(ready) = ready {
            let epoch = ready.message.control_epoch;
            self.owner = Some(Owner {
                source,
                epoch,
                next_sequence: Some(V2CommandSequence::FIRST),
                cached: None,
                last_serial_command_at: None,
            });
            (AcquireResultCode::Granted, Some(epoch))
        } else {
            (AcquireResultCode::NotReady, None)
        };

        let hello = self.hello.map(|value| value.message).or(observed);
        AcquireResult {
            controller_uid: self.config.controller_uid,
            boot_id: hello.map_or(request.expected_boot_id, |value| value.boot_id),
            request_id: request.request_id,
            control_epoch,
            result,
            capabilities: hello.map_or_else(empty_capabilities, |value| value.capabilities),
            faults: self.current_faults(),
            observed_firmware_abi: hello
                .map_or(request.expected_firmware_abi, |value| value.firmware_abi),
            observed_firmware_build_id: hello.map_or(request.expected_firmware_build_id, |value| {
                value.firmware_build_id
            }),
            observed_actuator_config_fingerprint: hello
                .map_or(request.expected_actuator_config_fingerprint, |value| {
                    value.actuator_config_fingerprint
                }),
        }
    }

    async fn handle_host_command(
        &mut self,
        source: SocketAddr,
        first_received_at: Instant,
        command: HostCommand,
        response: ActorResponseSender,
    ) -> Result<(), ActuationActorError> {
        if matches!(
            self.pending,
            Some(PendingOperation::Stop(_) | PendingOperation::PriorityStop(_))
        ) {
            let result =
                self.failed_command_result(command, HostCommandResultCode::RejectedAtServer);
            let _ = response.send(Message::HostCommandResult(result));
            return Ok(());
        }

        if let Some(PendingOperation::Command(pending)) = &self.pending {
            let code = if pending.source == source && pending.command == command {
                HostCommandResultCode::RejectedAtServer
            } else {
                HostCommandResultCode::ForceStopped
            };
            let result = self.failed_command_result(command, code);
            let _ = response.send(Message::HostCommandResult(result));
            if code == HostCommandResultCode::ForceStopped {
                self.fail_all_pending(code, StopResultCode::ControllerFaulted);
                self.issue_internal_stop(ForceStopReason::SequenceConflict)
                    .await?;
                self.clear_authority(false);
            }
            return Ok(());
        }

        let Some(owner) = &self.owner else {
            let result =
                self.failed_command_result(command, HostCommandResultCode::RejectedAtServer);
            let _ = response.send(Message::HostCommandResult(result));
            return Ok(());
        };
        if owner.source != source {
            let result =
                self.failed_command_result(command, HostCommandResultCode::RejectedAtServer);
            let _ = response.send(Message::HostCommandResult(result));
            return Ok(());
        }
        let identity_matches = self.hello.is_some_and(|hello| {
            command.controller_uid == self.config.controller_uid
                && command.boot_id == hello.message.boot_id
                && command.control_epoch == owner.epoch
        });
        if !identity_matches {
            let result =
                self.failed_command_result(command, HostCommandResultCode::ControllerRestarted);
            let _ = response.send(Message::HostCommandResult(result));
            self.issue_internal_stop(ForceStopReason::SessionReset)
                .await?;
            self.clear_authority(false);
            return Ok(());
        }

        if let Some(cached) = owner.cached {
            if cached.source == source && cached.command == command {
                let calculated_at = Instant::now();
                let result = cached.duplicate_result_at(calculated_at);
                let _ =
                    response.send_calculated_at(Message::HostCommandResult(result), calculated_at);
                return Ok(());
            }
            if cached.command.sequence == command.sequence {
                let result =
                    self.failed_command_result(command, HostCommandResultCode::ForceStopped);
                let _ = response.send(Message::HostCommandResult(result));
                self.issue_internal_stop(ForceStopReason::SequenceConflict)
                    .await?;
                self.clear_authority(false);
                return Ok(());
            }
        }

        let sequence_exact = owner.next_sequence == Some(command.sequence);
        let initial_zero_exact = owner.cached.is_some()
            || (command.sequence == V2CommandSequence::FIRST
                && command.requested_timer_pwm.is_zero());
        if !sequence_exact || !initial_zero_exact {
            let result = self.failed_command_result(command, HostCommandResultCode::ForceStopped);
            let _ = response.send(Message::HostCommandResult(result));
            self.issue_internal_stop(ForceStopReason::SequenceConflict)
                .await?;
            self.clear_authority(false);
            return Ok(());
        }

        let Some(hello) = self.hello.map(|value| value.message) else {
            let result =
                self.failed_command_result(command, HostCommandResultCode::ControllerRestarted);
            let _ = response.send(Message::HostCommandResult(result));
            self.clear_authority(true);
            return Ok(());
        };
        let maximum = hello.max_abs_pwm_percent.get();
        if command.requested_timer_pwm.left().get().unsigned_abs() > maximum
            || command.requested_timer_pwm.right().get().unsigned_abs() > maximum
            || command.lease.get() > hello.max_command_lease.get()
        {
            let result =
                self.failed_command_result(command, HostCommandResultCode::RejectedAtServer);
            let _ = response.send(Message::HostCommandResult(result));
            self.issue_internal_stop(ForceStopReason::ControllerFault)
                .await?;
            self.clear_authority(false);
            return Ok(());
        }
        let previously_applied = owner
            .cached
            .map_or(TimerPwm::ZERO, |cached| cached.controller_result.timer_pwm);
        if !candidate_command_step_is_admitted(
            previously_applied,
            command.requested_timer_pwm,
            self.config.maximum_command_step_percent,
        ) {
            let result =
                self.failed_command_result(command, HostCommandResultCode::RejectedAtServer);
            let _ = response.send(Message::HostCommandResult(result));
            self.issue_internal_stop(ForceStopReason::ControllerFault)
                .await?;
            self.clear_authority(false);
            return Ok(());
        }

        let now = Instant::now();
        let Some(heartbeat) = self.heartbeat else {
            let result =
                self.failed_command_result(command, HostCommandResultCode::RejectedAtServer);
            let _ = response.send(Message::HostCommandResult(result));
            return Ok(());
        };
        if !self.heartbeat_authorizes_command(heartbeat, now, command) {
            let result = self.failed_command_result(command, HostCommandResultCode::ForceStopped);
            let _ = response.send(Message::HostCommandResult(result));
            self.issue_internal_stop(ForceStopReason::TransportFault)
                .await?;
            self.clear_authority(false);
            return Ok(());
        }
        if self.owner.as_ref().is_some_and(|owner| {
            owner.last_serial_command_at.is_some_and(|last| {
                now.checked_duration_since(last)
                    .is_none_or(|elapsed| elapsed < self.config.minimum_host_command_interval)
            })
        }) {
            let result =
                self.failed_command_result(command, HostCommandResultCode::RejectedAtServer);
            let _ = response.send(Message::HostCommandResult(result));
            return Ok(());
        }
        let translated = match translate_command_deadline(
            first_received_at,
            command.lease,
            HeartbeatClockSample::new(heartbeat.message.controller_uptime, heartbeat.received_at),
            now,
            self.config.maximum_heartbeat_age,
            self.config.controller_clock_abs_error_ppm_bound,
            self.config.deadline_quantization_margin_ms,
        ) {
            Ok(value) => value,
            Err(_) => {
                let result =
                    self.failed_command_result(command, HostCommandResultCode::RejectedAtServer);
                let _ = response.send(Message::HostCommandResult(result));
                self.issue_internal_stop(ForceStopReason::LeaseExpired)
                    .await?;
                self.clear_authority(false);
                return Ok(());
            }
        };
        let apply = robot_protocol::v2::ApplyPwm {
            controller_uid: command.controller_uid,
            boot_id: command.boot_id,
            control_epoch: command.control_epoch,
            sequence: command.sequence,
            expires_at: translated.controller_deadline_exclusive(),
            timer_pwm: command.requested_timer_pwm,
        };
        // A new application can change physical output, so an older exact
        // stop receipt is no longer current evidence for a later retry.
        self.invalidate_stop_cache();
        let serial_sent_at = match self.send_serial(Message::ApplyPwm(apply)).await {
            Ok(value) => value,
            Err(error) => {
                let result =
                    self.failed_command_result(command, HostCommandResultCode::ForceStopped);
                let _ = response.send(Message::HostCommandResult(result));
                self.clear_authority(true);
                return Err(error);
            }
        };
        if let Some(owner) = &mut self.owner {
            owner.last_serial_command_at = Some(serial_sent_at);
        }
        self.pending = Some(PendingOperation::Command(PendingCommand {
            source,
            command,
            translated,
            controller_uptime_reference: heartbeat.message.controller_uptime,
            serial_sent_at,
            response,
        }));
        Ok(())
    }

    async fn handle_host_stop(
        &mut self,
        source: SocketAddr,
        request: HostStop,
        response: ActorResponseSender,
    ) -> Result<(), ActuationActorError> {
        if request.controller_uid != self.config.controller_uid {
            let result = self.failed_stop_result(request, StopResultCode::IdentityMismatch);
            let _ = response.send(Message::HostStopResult(result));
            return Ok(());
        }
        if let Some(cached) = self.cached_stop {
            if cached.source == source && cached.request == request {
                let _ = response.send(Message::HostStopResult(cached.result));
                return Ok(());
            }
        }
        if let TargetBootId::Exact(target) = request.target_boot_id {
            if self
                .observed_hello
                .is_some_and(|hello| hello.message.boot_id != target)
            {
                let result = self.failed_stop_result(request, StopResultCode::IdentityMismatch);
                let _ = response.send(Message::HostStopResult(result));
                return Ok(());
            }
        }
        self.invalidate_stop_cache();

        if let Some(pending) = self.pending.take() {
            match pending {
                PendingOperation::Command(value) => {
                    let result = self
                        .failed_command_result(value.command, HostCommandResultCode::ForceStopped);
                    let _ = value.response.send(Message::HostCommandResult(result));
                }
                PendingOperation::Stop(value) => {
                    let result =
                        self.failed_stop_result(value.request, StopResultCode::StopAckTimeout);
                    let _ = value.response.send(Message::HostStopResult(result));
                }
                PendingOperation::PriorityStop(value) => self.publish_priority_stop(
                    value.covers_through_generation,
                    PriorityStopEvidence::Uncertain(StopResultCode::ControllerUnavailable),
                ),
            }
        }
        self.owner = None;
        self.heartbeat = None;
        self.ready = None;

        let serial_request_id = match self.allocate_internal_request_id() {
            Ok(value) => value,
            Err(error) => {
                let result =
                    self.failed_stop_result(request, StopResultCode::ControllerUnavailable);
                let _ = response.send(Message::HostStopResult(result));
                self.clear_authority(true);
                return Err(error);
            }
        };
        let force_stop = ForceStop {
            controller_uid: request.controller_uid,
            target_boot_id: request.target_boot_id,
            request_id: serial_request_id,
            reason: request.reason,
        };
        let serial_sent_at = match self.send_serial(Message::ForceStop(force_stop)).await {
            Ok(value) => value,
            Err(error) => {
                let result =
                    self.failed_stop_result(request, StopResultCode::ControllerUnavailable);
                let _ = response.send(Message::HostStopResult(result));
                self.clear_authority(true);
                return Err(error);
            }
        };
        self.pending = Some(PendingOperation::Stop(PendingStop {
            source,
            request,
            serial_request_id,
            serial_sent_at,
            response,
        }));
        self.publish_snapshot(Instant::now());
        Ok(())
    }

    async fn handle_serial_message(
        &mut self,
        message: Message,
        received_at: Instant,
    ) -> Result<(), ActuationActorError> {
        match message {
            Message::ControllerHello(value) => self.handle_hello(value, received_at).await?,
            Message::ControllerReady(value) => self.handle_ready(value, received_at).await?,
            Message::Heartbeat(value) => self.handle_heartbeat(value, received_at).await?,
            Message::AppliedResult(value) => self.handle_applied_result(value, received_at).await?,
            Message::HostStopResult(value) => self.handle_stop_result(value, received_at).await?,
            Message::ObservationalOdometry(value) => {
                if self.odometry_identity_matches(value) {
                    self.telemetry.observe_odometry(value, received_at);
                } else {
                    self.last_fault =
                        Some(ActuationFaultEvidence::ObservationalOdometryIdentityMismatch(value));
                    self.protocol_fault(ForceStopReason::SessionReset).await?;
                }
            }
            unexpected => {
                self.last_fault = Some(ActuationFaultEvidence::UnexpectedControllerMessage(
                    unexpected.kind(),
                ));
                self.protocol_fault(ForceStopReason::TransportFault).await?;
            }
        }
        self.publish_snapshot(received_at);
        Ok(())
    }

    async fn handle_hello(
        &mut self,
        hello: ControllerHello,
        _received_at: Instant,
    ) -> Result<(), ActuationActorError> {
        // The firmware emits periodic discovery Hello records while its output
        // is safe. An exact same-boot repeat carries no newer session or
        // liveness evidence and must not revoke, renew, or recreate authority.
        if self.hello.is_some_and(|admitted| admitted.message == hello) {
            return Ok(());
        }
        self.fail_all_pending(
            HostCommandResultCode::ControllerRestarted,
            StopResultCode::ControllerUnavailable,
        );
        self.clear_authority(true);
        let timed = TimedHello { message: hello };
        self.observed_hello = Some(timed);

        if !self.hello_is_exact(hello) {
            self.faulted = true;
            self.last_fault = Some(ActuationFaultEvidence::ControllerHelloRejected(hello));
            self.issue_stop_for(
                hello.controller_uid,
                TargetBootId::Exact(hello.boot_id),
                ForceStopReason::ControllerFault,
            )
            .await?;
            return Ok(());
        }

        self.hello = Some(timed);
        self.faulted = false;
        self.last_fault = None;
        self.issue_stop_for(
            hello.controller_uid,
            TargetBootId::Exact(hello.boot_id),
            ForceStopReason::SessionReset,
        )
        .await?;
        let request_id = self.allocate_internal_request_id()?;
        // The STM32 RX queue is deliberately small and fail-closed. Do not
        // burst ForceStop and BeginSession into it: retain the typed begin
        // request until the exact startup stop receipt proves that the first
        // transaction completed. This also prevents a session from starting
        // after an intervening host stop, priority stop, shutdown, or fault.
        self.startup_begin_after_stop = Some(BeginSession {
            controller_uid: hello.controller_uid,
            boot_id: hello.boot_id,
            request_id,
            heartbeat_period: self.config.heartbeat_period,
        });
        Ok(())
    }

    async fn handle_ready(
        &mut self,
        ready: ControllerReady,
        received_at: Instant,
    ) -> Result<(), ActuationActorError> {
        let exact = self.hello.is_some_and(|hello| {
            self.startup_begin_after_stop.is_none()
                && self.last_internal_stop.is_none()
                && ready.controller_uid == hello.message.controller_uid
                && ready.boot_id == hello.message.boot_id
                && ready.capabilities == hello.message.capabilities
                && ready.capabilities.classify_session_admission(
                    self.config.expected_max_abs_pwm_percent,
                    self.config.expected_physical_stop_semantics,
                ) == Ok(self.config.expected_session_admission())
                && ready.output_state.is_safe()
                && ready.faults.is_clear()
        });
        if !exact {
            self.last_fault = Some(ActuationFaultEvidence::ControllerReadyRejected(ready));
            self.protocol_fault(ForceStopReason::SessionReset).await?;
            return Ok(());
        }
        if self.ready.is_some() {
            self.fail_all_pending(
                HostCommandResultCode::ControllerRestarted,
                StopResultCode::ControllerUnavailable,
            );
            self.owner = None;
        }
        self.ready = Some(ReadySession {
            message: ready,
            received_at,
        });
        self.heartbeat = None;
        self.faulted = false;
        self.last_fault = None;
        Ok(())
    }

    async fn handle_heartbeat(
        &mut self,
        heartbeat: Heartbeat,
        received_at: Instant,
    ) -> Result<(), ActuationActorError> {
        let Some(ready) = self.ready else {
            if heartbeat.controller_uid != self.config.controller_uid
                || !heartbeat.timer_pwm.is_zero()
                || !heartbeat.output_state.is_safe()
                || !heartbeat.faults.is_clear()
            {
                self.last_fault = Some(ActuationFaultEvidence::PreSessionHeartbeatRejected(
                    heartbeat,
                ));
                self.protocol_fault(ForceStopReason::ControllerFault)
                    .await?;
            }
            return Ok(());
        };
        if self.heartbeat_is_provably_delayed(heartbeat, ready) {
            return Ok(());
        }
        let exact_session = heartbeat.controller_uid == ready.message.controller_uid
            && heartbeat.boot_id == ready.message.boot_id
            && self.heartbeat_state_is_exact(heartbeat, ready)
            && heartbeat.faults.is_clear();
        if !exact_session {
            self.last_fault = Some(ActuationFaultEvidence::SessionHeartbeatRejected(heartbeat));
            self.protocol_fault(ForceStopReason::SessionReset).await?;
            return Ok(());
        }
        if !self.heartbeat_progresses(heartbeat) {
            self.last_fault = Some(ActuationFaultEvidence::SessionHeartbeatDidNotProgress(
                heartbeat,
            ));
            self.protocol_fault(ForceStopReason::SessionReset).await?;
            return Ok(());
        }
        if !self.heartbeat_matches_authority(heartbeat) {
            self.last_fault = Some(ActuationFaultEvidence::HeartbeatAuthorityConflict(
                heartbeat,
            ));
            self.protocol_fault(ForceStopReason::SequenceConflict)
                .await?;
            return Ok(());
        }
        self.heartbeat = Some(TimedHeartbeat {
            message: heartbeat,
            received_at,
        });
        self.faulted = false;
        self.last_fault = None;
        if let Some(startup_ready) = self.startup_ready.take() {
            let _ = startup_ready.send(());
        }
        Ok(())
    }

    /// Accept record-priority reordering without turning old telemetry into
    /// liveness or authority.
    ///
    /// Applied-control records may overtake a complete best-effort heartbeat
    /// which was already queued by the firmware. Ignore only a same-boot
    /// heartbeat whose controller time and session position are both strictly
    /// before stronger state already accepted by this actor. Stop/fault paths,
    /// ambiguous wrapping comparisons, and same/newer contradictions remain
    /// strict.
    fn heartbeat_is_provably_delayed(&self, heartbeat: Heartbeat, ready: ReadySession) -> bool {
        if self.faulted
            || self.shutdown_in_progress
            || self.last_internal_stop.is_some()
            || matches!(
                self.pending.as_ref(),
                Some(PendingOperation::Stop(_) | PendingOperation::PriorityStop(_))
            )
            || heartbeat.controller_uid != ready.message.controller_uid
            || heartbeat.boot_id != ready.message.boot_id
        {
            return false;
        }

        let Some(owner) = &self.owner else {
            return self.cached_stop.is_none()
                && heartbeat.control_epoch.is_none()
                && heartbeat.last_sequence.is_none()
                && controller_time_strictly_precedes(
                    heartbeat.controller_uptime,
                    ready.message.controller_uptime,
                );
        };
        let Some(cached) = owner.cached else {
            return heartbeat.control_epoch.is_none()
                && heartbeat.last_sequence.is_none()
                && controller_time_strictly_precedes(
                    heartbeat.controller_uptime,
                    ready.message.controller_uptime,
                );
        };
        let state_strictly_precedes = match (heartbeat.control_epoch, heartbeat.last_sequence) {
            (None, None) => true,
            (Some(epoch), Some(sequence)) => {
                epoch == owner.epoch && sequence.get() < cached.command.sequence.get()
            }
            _ => false,
        };
        state_strictly_precedes
            && controller_time_strictly_precedes(
                heartbeat.controller_uptime,
                cached.controller_result.applied_at,
            )
    }

    fn heartbeat_state_is_exact(&self, heartbeat: Heartbeat, ready: ReadySession) -> bool {
        let before_first_application = self
            .owner
            .as_ref()
            .is_none_or(|owner| owner.cached.is_none())
            && heartbeat.control_epoch.is_none()
            && heartbeat.last_sequence.is_none();
        if before_first_application {
            return heartbeat.control_epoch.is_none()
                && heartbeat.last_sequence.is_none()
                && self.heartbeat_is_stopped(heartbeat);
        }

        let established_identity = heartbeat.control_epoch == Some(ready.message.control_epoch)
            && heartbeat.last_sequence.is_some();
        if !established_identity {
            return false;
        }
        if heartbeat.timer_pwm.is_zero() {
            self.heartbeat_is_stopped(heartbeat)
        } else {
            heartbeat.output_state == OutputState::NonzeroPwm
                && heartbeat
                    .readiness
                    .is_deadline_ready_for_session(self.config.controller_session_class)
                && matches!(
                    heartbeat
                        .expires_at
                        .relation_to(heartbeat.controller_uptime),
                    DeadlineRelation::Future { .. }
                )
        }
    }

    fn heartbeat_is_stopped(&self, heartbeat: Heartbeat) -> bool {
        heartbeat.timer_pwm.is_zero()
            && heartbeat.output_state.is_safe()
            && heartbeat
                .readiness
                .is_stopped_ready_for_session(self.config.controller_session_class)
            && matches!(
                heartbeat
                    .expires_at
                    .relation_to(heartbeat.controller_uptime),
                DeadlineRelation::Expired
            )
    }

    fn heartbeat_progresses(&self, heartbeat: Heartbeat) -> bool {
        let Some(previous) = self.heartbeat else {
            return true;
        };
        let delta = heartbeat
            .controller_uptime
            .get()
            .wrapping_sub(previous.message.controller_uptime.get());
        delta != 0 && delta < 0x8000_0000
    }

    fn heartbeat_matches_authority(&self, heartbeat: Heartbeat) -> bool {
        if !heartbeat.timer_pwm.is_zero()
            && !matches!(
                heartbeat
                    .expires_at
                    .relation_to(heartbeat.controller_uptime),
                DeadlineRelation::Future { .. }
            )
        {
            return false;
        }
        let Some(owner) = &self.owner else {
            return heartbeat.timer_pwm.is_zero() && heartbeat.output_state.is_safe();
        };
        if let Some(PendingOperation::Command(pending)) = &self.pending {
            if heartbeat.last_sequence == Some(pending.command.sequence) {
                return if pending.command.requested_timer_pwm.is_zero() {
                    heartbeat.timer_pwm.is_zero() && heartbeat.output_state.is_safe()
                } else {
                    heartbeat.timer_pwm == pending.command.requested_timer_pwm
                        && heartbeat.expires_at
                            == pending.translated.controller_deadline_exclusive()
                };
            }
        }
        if let Some(cached) = owner.cached {
            if heartbeat.last_sequence != Some(cached.command.sequence) {
                return false;
            }
            if cached.controller_result.timer_pwm.is_zero()
                && cached.controller_result.output_state.is_safe()
            {
                return heartbeat.timer_pwm.is_zero() && heartbeat.output_state.is_safe();
            }
            return heartbeat.timer_pwm == cached.controller_result.timer_pwm
                && heartbeat.output_state == cached.controller_result.output_state
                && heartbeat.expires_at == cached.controller_result.expires_at;
        }
        heartbeat.last_sequence.is_none()
            && heartbeat.timer_pwm.is_zero()
            && heartbeat.output_state.is_safe()
    }

    async fn handle_applied_result(
        &mut self,
        applied: AppliedResult,
        received_at: Instant,
    ) -> Result<(), ActuationActorError> {
        let Some(pending) = self.pending.take() else {
            if self
                .owner
                .as_ref()
                .and_then(|owner| owner.cached)
                .is_some_and(|cached| cached.controller_result == applied)
            {
                return Ok(());
            }
            self.protocol_fault(ForceStopReason::SequenceConflict)
                .await?;
            return Ok(());
        };
        let pending = match pending {
            PendingOperation::Command(value) => value,
            other @ (PendingOperation::Stop(_) | PendingOperation::PriorityStop(_)) => {
                self.pending = Some(other);
                return Ok(());
            }
        };

        let session_exact = applied.controller_uid == pending.command.controller_uid
            && applied.boot_id == pending.command.boot_id
            && applied.control_epoch == pending.command.control_epoch;
        let command_exact = applied.sequence == pending.command.sequence
            && applied.expires_at == pending.translated.controller_deadline_exclusive();
        let exact = session_exact && command_exact;
        let (result_code, rejection_reason) = applied_result_disposition(applied.result);
        let requested_pwm = pending.command.requested_timer_pwm;
        let application_shape_exact = if requested_pwm.is_zero() {
            matches!(
                applied.result,
                AppliedResultCode::AppliedNew | AppliedResultCode::Stopped
            ) && applied.output_state.is_safe()
        } else {
            applied.result == AppliedResultCode::AppliedNew
                && applied.output_state == OutputState::NonzeroPwm
        };
        let applied_shape = application_shape_exact
            && applied.timer_pwm == requested_pwm
            && applied.faults.is_clear()
            && applied
                .applied_at
                .wrapping_elapsed_since(pending.controller_uptime_reference)
                < 0x8000_0000;
        if !exact || !applied_shape {
            let code = if exact {
                HostCommandResultCode::RejectedByController
            } else {
                HostCommandResultCode::ForceStopped
            };
            let reason = if !session_exact {
                ForceStopReason::SessionReset
            } else if !command_exact {
                ForceStopReason::SequenceConflict
            } else {
                rejection_reason.unwrap_or(ForceStopReason::ControllerFault)
            };
            let result = self.failed_command_result(pending.command, code);
            let _ = pending.response.send(Message::HostCommandResult(result));
            self.issue_internal_stop(reason).await?;
            self.clear_authority(false);
            return Ok(());
        }
        if received_at >= pending.translated.server_deadline_exclusive() {
            let result = self
                .failed_command_result(pending.command, HostCommandResultCode::AppliedAckTimeout);
            let _ = pending.response.send(Message::HostCommandResult(result));
            self.issue_internal_stop(ForceStopReason::LeaseExpired)
                .await?;
            self.clear_authority(false);
            return Ok(());
        }
        let remaining = conservative_remaining_lease(
            applied.applied_at,
            applied.expires_at,
            pending.serial_sent_at,
            received_at,
            self.config.controller_clock_abs_error_ppm_bound,
            self.config.deadline_quantization_margin_ms,
        )
        .ok()
        .and_then(|controller_remaining| {
            let server_remaining = pending
                .translated
                .server_deadline_exclusive()
                .checked_duration_since(received_at)?;
            let server_ms = u16::try_from(server_remaining.as_millis()).ok()?;
            RemainingLeaseMs::try_new(controller_remaining.get().min(server_ms)).ok()
        });
        let Some(remaining) = remaining.filter(|value| value.get() != 0) else {
            let result = self
                .failed_command_result(pending.command, HostCommandResultCode::AppliedAckTimeout);
            let _ = pending.response.send(Message::HostCommandResult(result));
            self.issue_internal_stop(ForceStopReason::LeaseExpired)
                .await?;
            self.clear_authority(false);
            return Ok(());
        };
        let Some(result_code) = result_code else {
            // The rejected variants are handled by `applied_shape` above.
            // Keep this branch panic-free if those predicates ever diverge.
            let result =
                self.failed_command_result(pending.command, HostCommandResultCode::ForceStopped);
            let _ = pending.response.send(Message::HostCommandResult(result));
            self.issue_internal_stop(ForceStopReason::ControllerFault)
                .await?;
            self.clear_authority(false);
            return Ok(());
        };
        let host_result = HostCommandResult {
            controller_uid: pending.command.controller_uid,
            boot_id: pending.command.boot_id,
            control_epoch: pending.command.control_epoch,
            sequence: pending.command.sequence,
            result: result_code,
            requested_timer_pwm: pending.command.requested_timer_pwm,
            controller_timer_pwm: applied.timer_pwm,
            output_state: applied.output_state,
            controller_applied_at: applied.applied_at,
            controller_expires_at: applied.expires_at,
            remaining_lease: remaining,
            faults: applied.faults,
        };
        let cached = CachedCommand {
            source: pending.source,
            command: pending.command,
            controller_result: applied,
            host_result,
            completed_at: received_at,
            server_deadline_exclusive: pending.translated.server_deadline_exclusive(),
        };
        if let Some(owner) = &mut self.owner {
            owner.next_sequence = pending.command.sequence.checked_successor();
            owner.cached = Some(cached);
        }
        let _ = pending
            .response
            .send_calculated_at(Message::HostCommandResult(host_result), received_at);
        Ok(())
    }

    async fn handle_stop_result(
        &mut self,
        result: HostStopResult,
        _received_at: Instant,
    ) -> Result<(), ActuationActorError> {
        match self.pending.take() {
            Some(PendingOperation::Stop(pending)) => {
                let boot_matches =
                    self.stop_result_boot_matches(pending.request.target_boot_id, result);
                let exact = result.controller_uid == pending.request.controller_uid
                    && result.request_id == pending.serial_request_id
                    && boot_matches;
                if exact {
                    let host_result = HostStopResult {
                        request_id: pending.request.request_id,
                        ..result
                    };
                    self.cached_stop = Some(CachedStop {
                        source: pending.source,
                        request: pending.request,
                        result: host_result,
                    });
                    let _ = pending.response.send(Message::HostStopResult(host_result));
                    self.owner = None;
                    self.heartbeat = None;
                    return Ok(());
                }
                if self
                    .complete_internal_stop_and_maybe_begin(result, false)
                    .await?
                {
                    self.pending = Some(PendingOperation::Stop(pending));
                    return Ok(());
                }
                let failed =
                    self.failed_stop_result(pending.request, StopResultCode::ControllerFaulted);
                let _ = pending.response.send(Message::HostStopResult(failed));
                self.protocol_fault(ForceStopReason::SessionReset).await?;
                return Ok(());
            }
            Some(PendingOperation::PriorityStop(pending)) => {
                let exact = result.controller_uid == pending.force_stop.controller_uid
                    && result.request_id == pending.force_stop.request_id
                    && self.stop_result_boot_matches(pending.force_stop.target_boot_id, result);
                if exact {
                    self.publish_priority_stop(
                        pending.covers_through_generation,
                        PriorityStopEvidence::Exact(result),
                    );
                    self.owner = None;
                    self.heartbeat = None;
                    return Ok(());
                }
                if self
                    .complete_internal_stop_and_maybe_begin(result, false)
                    .await?
                {
                    self.pending = Some(PendingOperation::PriorityStop(pending));
                    return Ok(());
                }
                self.publish_priority_stop(
                    pending.covers_through_generation,
                    PriorityStopEvidence::Uncertain(StopResultCode::ControllerFaulted),
                );
                self.protocol_fault(ForceStopReason::SessionReset).await?;
                return Ok(());
            }
            Some(other @ PendingOperation::Command(_)) => self.pending = Some(other),
            None => {}
        }

        if self
            .complete_internal_stop_and_maybe_begin(result, true)
            .await?
        {
            return Ok(());
        }
        self.last_fault = Some(ActuationFaultEvidence::UnexpectedStopResult(result));
        self.protocol_fault(ForceStopReason::SequenceConflict).await
    }

    async fn complete_internal_stop_and_maybe_begin(
        &mut self,
        result: HostStopResult,
        startup_may_begin: bool,
    ) -> Result<bool, ActuationActorError> {
        if !self.internal_stop_result_matches(result) {
            return Ok(false);
        }
        self.last_internal_stop = None;
        let Some(begin) = self.startup_begin_after_stop.take() else {
            return Ok(true);
        };
        let stop_is_exact = result.result.proves_controller_stop()
            && result.output_state.is_safe()
            && result.faults.is_clear();
        let startup_is_still_authorized = startup_may_begin
            && self.pending.is_none()
            && !self.shutdown_in_progress
            && !self.faulted
            && self.hello.is_some_and(|hello| {
                hello.message.controller_uid == begin.controller_uid
                    && hello.message.boot_id == begin.boot_id
            });
        if stop_is_exact && startup_is_still_authorized {
            self.send_serial(Message::BeginSession(begin)).await?;
        } else if !stop_is_exact {
            self.faulted = true;
            self.last_fault = Some(ActuationFaultEvidence::StartupStopNotConfirmed(result));
        }
        Ok(true)
    }

    async fn handle_framing_fault(
        &mut self,
        error: UartStreamError,
    ) -> Result<(), ActuationActorError> {
        log::error!("V2 controller framing fault: {error}");
        self.last_fault = Some(ActuationFaultEvidence::SerialFraming(error));
        self.fail_all_pending(
            HostCommandResultCode::ForceStopped,
            StopResultCode::ControllerFaulted,
        );
        self.issue_internal_stop(ForceStopReason::TransportFault)
            .await?;
        self.clear_authority(true);
        self.faulted = true;
        self.publish_snapshot(Instant::now());
        Ok(())
    }

    async fn protocol_fault(&mut self, reason: ForceStopReason) -> Result<(), ActuationActorError> {
        self.last_fault
            .get_or_insert(ActuationFaultEvidence::Protocol(reason));
        self.fail_all_pending(
            HostCommandResultCode::ForceStopped,
            StopResultCode::ControllerFaulted,
        );
        self.issue_internal_stop(reason).await?;
        self.clear_authority(false);
        self.faulted = true;
        Ok(())
    }

    fn hello_is_exact(&self, hello: ControllerHello) -> bool {
        hello.controller_uid == self.config.controller_uid
            && hello.firmware_abi == self.config.firmware_abi
            && hello.firmware_build_id == self.config.firmware_build_id
            && hello.actuator_config_fingerprint == self.config.actuator_config_fingerprint
            && hello.pwm_frequency == self.config.expected_pwm_frequency
            && hello.watchdog_nominal_period == self.config.expected_watchdog_nominal_period
            && hello.neutral_output == self.config.expected_neutral_output
            && hello.physical_stop_semantics == self.config.expected_physical_stop_semantics
            && hello.max_abs_pwm_percent == self.config.expected_max_abs_pwm_percent
            && hello.max_abs_pwm_percent.grants_motion_authority()
            && hello.session_admission() == Ok(self.config.expected_session_admission())
            && hello.output_state.is_safe()
    }

    fn heartbeat_authorizes_command(
        &self,
        heartbeat: TimedHeartbeat,
        now: Instant,
        command: HostCommand,
    ) -> bool {
        let bootstrap_stopped = self.owner.as_ref().is_some_and(|owner| {
            (owner.cached.is_none() && command.is_initial_zero_acquisition())
                || owner.cached.is_some_and(|cached| {
                    cached.controller_result.timer_pwm.is_zero()
                        && cached.controller_result.output_state.is_safe()
                })
        });
        let readiness_exact = if bootstrap_stopped {
            let stopped_identity = self.owner.as_ref().is_some_and(|owner| {
                owner.cached.map_or_else(
                    || {
                        heartbeat.message.control_epoch.is_none()
                            && heartbeat.message.last_sequence.is_none()
                    },
                    |cached| {
                        (heartbeat.message.control_epoch.is_none()
                            && heartbeat.message.last_sequence.is_none())
                            || (heartbeat.message.control_epoch == Some(command.control_epoch)
                                && heartbeat.message.last_sequence == Some(cached.command.sequence))
                    },
                )
            });
            stopped_identity && self.heartbeat_is_stopped(heartbeat.message)
        } else {
            heartbeat.message.control_epoch == Some(command.control_epoch)
                && heartbeat.message.last_sequence.is_some()
                && heartbeat
                    .message
                    .readiness
                    .is_deadline_ready_for_session(self.config.controller_session_class)
        };
        now.checked_duration_since(heartbeat.received_at)
            .is_some_and(|age| age < self.config.maximum_heartbeat_age)
            && readiness_exact
            && heartbeat.message.faults.is_clear()
            && self.ready.is_some_and(|ready| {
                heartbeat.message.controller_uid == ready.message.controller_uid
                    && heartbeat.message.boot_id == ready.message.boot_id
                    && command.control_epoch == ready.message.control_epoch
            })
    }

    fn odometry_identity_matches(&self, odometry: ObservationalOdometry) -> bool {
        if odometry.controller_uid != self.config.controller_uid {
            return false;
        }
        match self.ready {
            Some(ready) => {
                odometry.boot_id == ready.message.boot_id
                    && odometry.control_epoch == Some(ready.message.control_epoch)
            }
            None => self
                .hello
                .is_some_and(|hello| odometry.boot_id == hello.message.boot_id),
        }
    }

    async fn send_serial(&mut self, message: Message) -> Result<Instant, ActuationActorError> {
        let is_force_stop = matches!(&message, Message::ForceStop(_));
        #[cfg(feature = "qualification-fault-injection")]
        let inject_partial_uart_record = matches!(
            (&message, self.config.qualification_serial_fault),
            (
                Message::ApplyPwm(command),
                Some(
                    OperatorSupervisedCandidateSerialFaultInjection::PartialUartRecordOnFirstNonzeroCommand
                )
            ) if !command.timer_pwm.is_zero()
        );
        let record = UartRecord::encode(message).map_err(ActuationActorError::Encode)?;
        let sent_at = Instant::now();
        let shutdown = if self.shutdown_in_progress {
            None
        } else {
            self.shutdown.as_deref()
        };
        let priority_stop = if self.shutdown_in_progress || is_force_stop {
            None
        } else {
            self.priority_stop
                .as_deref()
                .map(|priority_stop| (priority_stop, self.priority_stop_generation))
        };
        #[cfg(feature = "qualification-fault-injection")]
        if inject_partial_uart_record {
            self.config.qualification_serial_fault = None;
            let prefix = QualificationPartialUartRecordPrefix::from_encoded_record(&record)
                .map_err(ActuationActorError::QualificationPartialUartRecordInvariant)?;
            let prefix_result = transmit_serial_record(
                &mut self.transport,
                prefix.as_bytes(),
                self.config.serial_transmit_timeout,
                shutdown,
                priority_stop,
            )
            .await;
            return Err(ActuationActorError::SerialTransmit(
                qualification_partial_uart_record_error(record.as_bytes().len(), prefix_result),
            ));
        }
        transmit_serial_record(
            &mut self.transport,
            record.as_bytes(),
            self.config.serial_transmit_timeout,
            shutdown,
            priority_stop,
        )
        .await
        .map_err(ActuationActorError::SerialTransmit)?;
        Ok(sent_at)
    }

    async fn handle_shutdown(
        &mut self,
        reason: ActuationShutdownReason,
    ) -> Result<(), ActuationActorError> {
        self.shutdown_in_progress = true;
        self.fail_all_pending(
            HostCommandResultCode::ForceStopped,
            StopResultCode::ControllerUnavailable,
        );
        self.issue_internal_stop(reason.force_stop_reason()).await?;
        self.clear_authority(true);
        let result = self.await_internal_stop_result().await?;
        if result.result.proves_controller_stop()
            && result.output_state.is_safe()
            && result.faults.is_clear()
        {
            Ok(())
        } else {
            Err(ActuationActorError::ShutdownStopNotConfirmed { result })
        }
    }

    async fn recover_priority_stop_interrupted_transmit(
        &mut self,
        interrupted: SerialTransmitError,
    ) -> Result<(), ActuationActorError> {
        debug_assert!(interrupted.interrupted_by_priority_stop());
        log::warn!(
            "priority HostStop interrupted an ordinary controller record; preserving transmit uncertainty and resynchronizing before ForceStop: {interrupted}"
        );
        self.fail_all_pending(
            HostCommandResultCode::ForceStopped,
            StopResultCode::ControllerUnavailable,
        );

        if interrupted.left_partial_record() {
            if let Err(source) = transmit_serial_record(
                &mut self.transport,
                &UART_RECORD_DELIMITER,
                self.config.serial_transmit_timeout,
                None,
                None,
            )
            .await
            {
                if let Some(priority_stop) = &self.priority_stop {
                    let through_generation = priority_stop
                        .requested_after(self.priority_stop_generation)
                        .unwrap_or(self.priority_stop_generation);
                    priority_stop.publish(
                        through_generation,
                        PriorityStopEvidence::Uncertain(StopResultCode::ControllerUnavailable),
                    );
                }
                self.clear_authority(true);
                return Err(ActuationActorError::SerialTransmit(source));
            }
        }

        // Coordinated shutdown remains the terminal owner. The next actor turn
        // handles it before beginning a host-priority stop transaction.
        if self
            .shutdown
            .as_deref()
            .and_then(ActuationShutdownSignal::requested_reason)
            .is_some()
        {
            return Ok(());
        }

        let generation = self
            .priority_stop
            .as_deref()
            .and_then(|priority_stop| priority_stop.requested_after(self.priority_stop_generation))
            .unwrap_or(self.priority_stop_generation);
        self.handle_priority_stop_generation(generation).await
    }

    async fn recover_shutdown_interrupted_transmit(
        &mut self,
        interrupted: SerialTransmitError,
        reason: ActuationShutdownReason,
    ) -> ActuationActorError {
        debug_assert!(interrupted.interrupted_by_shutdown());
        self.shutdown_in_progress = true;
        self.fail_all_pending(
            HostCommandResultCode::ForceStopped,
            StopResultCode::ControllerUnavailable,
        );

        let resynchronization = if interrupted.left_partial_record() {
            match transmit_serial_record(
                &mut self.transport,
                &UART_RECORD_DELIMITER,
                self.config.serial_transmit_timeout,
                None,
                None,
            )
            .await
            {
                Ok(()) => SerialResynchronizationOutcome::DelimiterTransmitted,
                Err(source) => SerialResynchronizationOutcome::Failed(source),
            }
        } else {
            SerialResynchronizationOutcome::NotRequired
        };

        let force_stop = match self.issue_internal_stop(reason.force_stop_reason()).await {
            Ok(()) => {
                self.clear_authority(true);
                match self.await_internal_stop_result().await {
                    Ok(result)
                        if result.result.proves_controller_stop()
                            && result.output_state.is_safe() =>
                    {
                        ShutdownForceStopOutcome::Confirmed(result)
                    }
                    Ok(result) => ShutdownForceStopOutcome::ExactButUnconfirmed(result),
                    Err(source) => ShutdownForceStopOutcome::Uncertain(Box::new(source)),
                }
            }
            Err(source) => {
                self.clear_authority(true);
                ShutdownForceStopOutcome::Uncertain(Box::new(source))
            }
        };
        ActuationActorError::ShutdownInterruptedTransmit {
            interrupted,
            recovery: Box::new(ShutdownInterruptedTransmitRecovery {
                resynchronization,
                force_stop,
            }),
        }
    }

    #[cfg(feature = "qualification-fault-injection")]
    async fn recover_qualification_partial_uart_record(
        &mut self,
        interrupted: SerialTransmitError,
    ) -> ActuationActorError {
        debug_assert!(matches!(
            interrupted,
            SerialTransmitError::QualificationPartialRecord { .. }
        ));
        self.shutdown_in_progress = true;
        self.fail_all_pending(
            HostCommandResultCode::ForceStopped,
            StopResultCode::ControllerUnavailable,
        );

        debug_assert!(interrupted.left_partial_record());
        let resynchronization = match transmit_serial_record(
            &mut self.transport,
            &UART_RECORD_DELIMITER,
            self.config.serial_transmit_timeout,
            None,
            None,
        )
        .await
        {
            Ok(()) => SerialResynchronizationOutcome::DelimiterTransmitted,
            Err(source) => SerialResynchronizationOutcome::Failed(source),
        };

        let force_stop = match self
            .issue_internal_stop(ForceStopReason::TransportFault)
            .await
        {
            Ok(()) => {
                self.clear_authority(true);
                match self.await_internal_stop_result().await {
                    Ok(result)
                        if result.result.proves_controller_stop()
                            && result.output_state.is_safe() =>
                    {
                        ShutdownForceStopOutcome::Confirmed(result)
                    }
                    Ok(result) => ShutdownForceStopOutcome::ExactButUnconfirmed(result),
                    Err(source) => ShutdownForceStopOutcome::Uncertain(Box::new(source)),
                }
            }
            Err(source) => {
                self.clear_authority(true);
                ShutdownForceStopOutcome::Uncertain(Box::new(source))
            }
        };
        ActuationActorError::QualificationPartialUartRecordInjected {
            interrupted,
            recovery: Box::new(QualificationPartialUartRecordInjectionRecovery {
                resynchronization,
                force_stop,
            }),
        }
    }

    async fn await_internal_stop_result(&mut self) -> Result<HostStopResult, ActuationActorError> {
        let maximum_wait = self.config.serial_applied_ack_timeout;
        let deadline = tokio::time::Instant::now()
            .checked_add(maximum_wait)
            .unwrap_or_else(tokio::time::Instant::now);
        let mut read_buffer = [0_u8; SERIAL_READ_TURN_BYTES];
        loop {
            let count = tokio::select! {
                biased;
                _ = tokio::time::sleep_until(deadline) => {
                    return Err(ActuationActorError::ShutdownStopConfirmationTimedOut {
                        maximum_wait,
                    });
                }
                read = self.transport.read(&mut read_buffer) => {
                    match read {
                        Ok(0) => return Err(ActuationActorError::SerialEof),
                        Ok(count) => count,
                        Err(source) => return Err(ActuationActorError::SerialRead(source)),
                    }
                }
            };
            for &byte in &read_buffer[..count] {
                let Some(decoded) = self.decoder.push(byte) else {
                    continue;
                };
                let message = match decoded {
                    Ok(message) => message,
                    Err(error) => {
                        log::error!(
                            "framing fault while awaiting shutdown stop confirmation: {error}"
                        );
                        continue;
                    }
                };
                if let Message::HostStopResult(result) = message {
                    if self.internal_stop_result_matches(result) {
                        self.last_internal_stop = None;
                        return Ok(result);
                    }
                }
            }
        }
    }

    fn allocate_internal_request_id(&mut self) -> Result<RequestId, ActuationActorError> {
        let request_id = self
            .next_internal_request_id
            .take()
            .ok_or(ActuationActorError::InternalRequestIdExhausted)?;
        self.next_internal_request_id = request_id.get().checked_add(1).map(RequestId::new);
        Ok(request_id)
    }

    async fn issue_internal_stop(
        &mut self,
        reason: ForceStopReason,
    ) -> Result<(), ActuationActorError> {
        self.startup_begin_after_stop = None;
        let (uid, target) =
            self.observed_hello
                .map_or((self.config.controller_uid, TargetBootId::Any), |hello| {
                    (
                        hello.message.controller_uid,
                        TargetBootId::Exact(hello.message.boot_id),
                    )
                });
        self.issue_stop_for(uid, target, reason).await
    }

    async fn issue_stop_for(
        &mut self,
        controller_uid: ControllerUid,
        target_boot_id: TargetBootId,
        reason: ForceStopReason,
    ) -> Result<(), ActuationActorError> {
        // A new controller stop transaction means an older host-correlated
        // receipt no longer proves the result of a request received now. This
        // must happen before allocation/transmit so even an uncertain attempt
        // cannot leave stale evidence available for replay.
        self.invalidate_stop_cache();
        let stop = ForceStop {
            controller_uid,
            target_boot_id,
            request_id: self.allocate_internal_request_id()?,
            reason,
        };
        self.send_serial(Message::ForceStop(stop)).await?;
        self.last_internal_stop = Some(stop);
        Ok(())
    }

    fn clear_authority(&mut self, clear_session: bool) {
        self.owner = None;
        if let Some(PendingOperation::PriorityStop(value)) = &self.pending {
            self.publish_priority_stop(
                value.covers_through_generation,
                PriorityStopEvidence::Uncertain(StopResultCode::ControllerUnavailable),
            );
        }
        self.pending = None;
        self.heartbeat = None;
        if clear_session {
            self.startup_begin_after_stop = None;
            self.hello = None;
            self.ready = None;
            self.invalidate_stop_cache();
        }
    }

    fn invalidate_stop_cache(&mut self) {
        self.cached_stop = None;
        if let Some(priority_stop) = &self.priority_stop {
            priority_stop.invalidate_cache();
        }
    }

    fn stop_result_boot_matches(&self, target: TargetBootId, result: HostStopResult) -> bool {
        match target {
            TargetBootId::Exact(expected) => {
                result.observed_boot_id == TargetBootId::Exact(expected)
            }
            TargetBootId::Any => self.observed_hello.map_or_else(
                || matches!(result.observed_boot_id, TargetBootId::Exact(_)),
                |hello| result.observed_boot_id == TargetBootId::Exact(hello.message.boot_id),
            ),
        }
    }

    fn internal_stop_result_matches(&self, result: HostStopResult) -> bool {
        self.last_internal_stop.is_some_and(|stop| {
            result.controller_uid == stop.controller_uid
                && result.request_id == stop.request_id
                && self.stop_result_boot_matches(stop.target_boot_id, result)
        })
    }

    fn publish_priority_stop(&self, through_generation: u64, evidence: PriorityStopEvidence) {
        if let Some(priority_stop) = &self.priority_stop {
            priority_stop.publish(through_generation, evidence);
        }
    }

    fn fail_all_pending(&mut self, command_code: HostCommandResultCode, stop_code: StopResultCode) {
        let Some(pending) = self.pending.take() else {
            return;
        };
        match pending {
            PendingOperation::Command(value) => {
                let result = self.failed_command_result(value.command, command_code);
                let _ = value.response.send(Message::HostCommandResult(result));
            }
            PendingOperation::Stop(value) => {
                let result = self.failed_stop_result(value.request, stop_code);
                let _ = value.response.send(Message::HostStopResult(result));
            }
            PendingOperation::PriorityStop(value) => self.publish_priority_stop(
                value.covers_through_generation,
                PriorityStopEvidence::Uncertain(stop_code),
            ),
        }
    }

    fn failed_command_result(
        &self,
        command: HostCommand,
        code: HostCommandResultCode,
    ) -> HostCommandResult {
        // Frozen V2 carries fixed-width output fields even when `code` proves
        // no controller observation. These are compatibility placeholders;
        // host/API/UI consumers must use `HostCommandResult::output_evidence`.
        HostCommandResult {
            controller_uid: command.controller_uid,
            boot_id: command.boot_id,
            control_epoch: command.control_epoch,
            sequence: command.sequence,
            result: code,
            requested_timer_pwm: command.requested_timer_pwm,
            controller_timer_pwm: TimerPwm::ZERO,
            output_state: OutputState::Disabled,
            controller_applied_at: self.current_uptime(),
            controller_expires_at: robot_protocol::v2::ControllerDeadlineMsWrapping::new(
                self.current_uptime().get(),
            ),
            remaining_lease: RemainingLeaseMs::ZERO,
            faults: self.current_faults(),
        }
    }

    fn failed_stop_result(&self, request: HostStop, code: StopResultCode) -> HostStopResult {
        // As above, a non-confirming stop code makes all concrete output bits
        // unproven. `HostStopResult::output_evidence` prevents their promotion.
        HostStopResult {
            controller_uid: self.config.controller_uid,
            observed_boot_id: self.observed_hello.map_or(request.target_boot_id, |hello| {
                TargetBootId::Exact(hello.message.boot_id)
            }),
            request_id: request.request_id,
            result: code,
            output_state: OutputState::Disabled,
            controller_uptime: self.current_uptime(),
            faults: self.current_faults(),
        }
    }

    fn status_report(&self, query: StatusQuery, now: Instant) -> StatusReport {
        let heartbeat = self.heartbeat.filter(|heartbeat| {
            now.checked_duration_since(heartbeat.received_at)
                .is_some_and(|age| age < self.config.maximum_heartbeat_age)
        });
        let ready = self.ready.is_some() && heartbeat.is_some() && !self.faulted;
        let stopped = heartbeat.is_some_and(|heartbeat| {
            heartbeat.message.timer_pwm.is_zero()
                && heartbeat.message.output_state.is_safe()
                && heartbeat.message.faults.is_clear()
        });
        let status = if self.faulted {
            StatusCode::Faulted
        } else if self.hello.is_none() {
            StatusCode::Disconnected
        } else if !ready {
            StatusCode::EstablishingSession
        } else if self.owner.is_none() && stopped {
            StatusCode::ReadyStopped
        } else {
            StatusCode::ReadyActive
        };
        let active = status == StatusCode::ReadyActive;
        let cached = self.owner.as_ref().and_then(|owner| owner.cached);
        let (output_state, timer_pwm, uptime, remaining) = if active {
            match (heartbeat, cached) {
                (Some(heartbeat), Some(cached)) if heartbeat.received_at >= cached.completed_at => {
                    let remaining = if heartbeat.message.timer_pwm.is_zero() {
                        RemainingLeaseMs::ZERO
                    } else {
                        cached.remaining_at(now)
                    };
                    (
                        heartbeat.message.output_state,
                        heartbeat.message.timer_pwm,
                        heartbeat.message.controller_uptime,
                        remaining,
                    )
                }
                (Some(heartbeat), None) => (
                    heartbeat.message.output_state,
                    heartbeat.message.timer_pwm,
                    heartbeat.message.controller_uptime,
                    RemainingLeaseMs::ZERO,
                ),
                (_, Some(cached)) => (
                    cached.controller_result.output_state,
                    cached.controller_result.timer_pwm,
                    cached.controller_result.applied_at,
                    cached.remaining_at(now),
                ),
                (None, None) => (
                    OutputState::Disabled,
                    TimerPwm::ZERO,
                    self.current_uptime(),
                    RemainingLeaseMs::ZERO,
                ),
            }
        } else {
            (
                heartbeat.map_or(OutputState::Disabled, |value| value.message.output_state),
                TimerPwm::ZERO,
                self.current_uptime(),
                RemainingLeaseMs::ZERO,
            )
        };
        StatusReport {
            controller_uid: self.config.controller_uid,
            observed_boot_id: self.observed_hello.map_or(TargetBootId::Any, |hello| {
                TargetBootId::Exact(hello.message.boot_id)
            }),
            request_id: query.request_id,
            status,
            control_epoch: self
                .owner
                .as_ref()
                .filter(|_| active)
                .map(|owner| owner.epoch),
            controller_uptime: uptime,
            capabilities: self
                .hello
                .map_or_else(empty_capabilities, |hello| hello.message.capabilities),
            output_state,
            controller_timer_pwm: timer_pwm,
            remaining_lease: remaining,
            faults: self.current_faults(),
        }
    }

    fn current_uptime(&self) -> ControllerUptimeMsWrapping {
        self.heartbeat
            .map_or(ControllerUptimeMsWrapping::new(0), |heartbeat| {
                heartbeat.message.controller_uptime
            })
    }

    fn current_faults(&self) -> ControllerFaults {
        self.heartbeat
            .map_or(ControllerFaults::NONE, |heartbeat| heartbeat.message.faults)
    }

    fn snapshot(&self, now: Instant) -> ActuationSnapshot {
        let query = StatusQuery {
            expected_controller_uid: self.config.controller_uid,
            request_id: RequestId::new(0),
        };
        let status = self.status_report(query, now);
        let output = self
            .heartbeat
            .filter(|heartbeat| {
                now.checked_duration_since(heartbeat.received_at)
                    .is_some_and(|age| age < self.config.maximum_heartbeat_age)
            })
            .map_or(ActuationOutputEvidence::Unknown, |heartbeat| {
                ActuationOutputEvidence::Observed(ObservedActuationOutput {
                    controller_uptime: heartbeat.message.controller_uptime,
                    output_state: heartbeat.message.output_state,
                    controller_timer_pwm: heartbeat.message.timer_pwm,
                    faults: heartbeat.message.faults,
                })
            });
        ActuationSnapshot {
            status: status.status,
            startup_phase: self.startup_phase(),
            fault: self.last_fault,
            observed_boot_id: status.observed_boot_id,
            control_epoch: status.control_epoch,
            output,
            last_sequence: self
                .owner
                .as_ref()
                .and_then(|owner| owner.cached)
                .map(|cached| cached.command.sequence),
        }
    }

    fn startup_phase(&self) -> ActuationStartupPhase {
        if self.faulted {
            ActuationStartupPhase::Faulted
        } else if self.startup_ready.is_none() {
            ActuationStartupPhase::ReadyStopped
        } else if self.hello.is_none() {
            ActuationStartupPhase::AwaitingHello
        } else if self.startup_begin_after_stop.is_some() || self.last_internal_stop.is_some() {
            ActuationStartupPhase::AwaitingStartupStopReceipt
        } else if self.ready.is_none() {
            ActuationStartupPhase::AwaitingControllerReady
        } else {
            ActuationStartupPhase::AwaitingStoppedHeartbeat
        }
    }

    fn publish_snapshot(&self, now: Instant) {
        self.telemetry.update_actuation(self.snapshot(now));
    }
}

const fn controller_time_strictly_precedes(
    candidate: ControllerUptimeMsWrapping,
    accepted: ControllerUptimeMsWrapping,
) -> bool {
    matches!(accepted.wrapping_elapsed_since(candidate), 1..0x8000_0000)
}

async fn transmit_serial_record<Transport>(
    transport: &mut Transport,
    record: &[u8],
    maximum_duration: Duration,
    shutdown: Option<&ActuationShutdownSignal>,
    priority_stop: Option<(&PriorityStopCoordinator, u64)>,
) -> Result<(), SerialTransmitError>
where
    Transport: AsyncWrite + Unpin,
{
    let deadline = tokio::time::Instant::now()
        .checked_add(maximum_duration)
        .unwrap_or_else(tokio::time::Instant::now);
    let mut written_bytes = 0;
    while written_bytes < record.len() {
        let write = transport.write(&record[written_bytes..]);
        tokio::pin!(write);
        let priority_stop_requested = wait_for_priority_stop(priority_stop);
        tokio::pin!(priority_stop_requested);
        let count = match shutdown {
            Some(shutdown) => {
                tokio::select! {
                    biased;
                    _reason = shutdown.wait() => {
                        return Err(SerialTransmitError::Interrupted {
                            phase: SerialTransmitPhase::Write,
                            cause: SerialTransmitInterruption::ShutdownRequested,
                            written_bytes,
                            record_bytes: record.len(),
                            maximum_duration,
                        });
                    }
                    _generation = &mut priority_stop_requested => {
                        return Err(SerialTransmitError::Interrupted {
                            phase: SerialTransmitPhase::Write,
                            cause: SerialTransmitInterruption::PriorityStopRequested,
                            written_bytes,
                            record_bytes: record.len(),
                            maximum_duration,
                        });
                    }
                    _ = tokio::time::sleep_until(deadline) => {
                        return Err(SerialTransmitError::Interrupted {
                            phase: SerialTransmitPhase::Write,
                            cause: SerialTransmitInterruption::DeadlineExceeded,
                            written_bytes,
                            record_bytes: record.len(),
                            maximum_duration,
                        });
                    }
                    result = &mut write => result,
                }
            }
            None => {
                tokio::select! {
                    biased;
                    _generation = &mut priority_stop_requested => {
                        return Err(SerialTransmitError::Interrupted {
                            phase: SerialTransmitPhase::Write,
                            cause: SerialTransmitInterruption::PriorityStopRequested,
                            written_bytes,
                            record_bytes: record.len(),
                            maximum_duration,
                        });
                    }
                    _ = tokio::time::sleep_until(deadline) => {
                        return Err(SerialTransmitError::Interrupted {
                            phase: SerialTransmitPhase::Write,
                            cause: SerialTransmitInterruption::DeadlineExceeded,
                            written_bytes,
                            record_bytes: record.len(),
                            maximum_duration,
                        });
                    }
                    result = &mut write => result,
                }
            }
        }
        .map_err(|source| SerialTransmitError::Write {
            source,
            written_bytes,
            record_bytes: record.len(),
        })?;
        if count == 0 {
            return Err(SerialTransmitError::Write {
                source: io::Error::new(
                    io::ErrorKind::WriteZero,
                    "serial write returned zero bytes",
                ),
                written_bytes,
                record_bytes: record.len(),
            });
        }
        written_bytes = written_bytes.saturating_add(count);
    }

    let flush = transport.flush();
    tokio::pin!(flush);
    let priority_stop_requested = wait_for_priority_stop(priority_stop);
    tokio::pin!(priority_stop_requested);
    let result = match shutdown {
        Some(shutdown) => {
            tokio::select! {
                biased;
                _reason = shutdown.wait() => {
                    return Err(SerialTransmitError::Interrupted {
                        phase: SerialTransmitPhase::Flush,
                        cause: SerialTransmitInterruption::ShutdownRequested,
                        written_bytes,
                        record_bytes: record.len(),
                        maximum_duration,
                    });
                }
                _generation = &mut priority_stop_requested => {
                    return Err(SerialTransmitError::Interrupted {
                        phase: SerialTransmitPhase::Flush,
                        cause: SerialTransmitInterruption::PriorityStopRequested,
                        written_bytes,
                        record_bytes: record.len(),
                        maximum_duration,
                    });
                }
                _ = tokio::time::sleep_until(deadline) => {
                    return Err(SerialTransmitError::Interrupted {
                        phase: SerialTransmitPhase::Flush,
                        cause: SerialTransmitInterruption::DeadlineExceeded,
                        written_bytes,
                        record_bytes: record.len(),
                        maximum_duration,
                    });
                }
                result = &mut flush => result,
            }
        }
        None => {
            tokio::select! {
                biased;
                _generation = &mut priority_stop_requested => {
                    return Err(SerialTransmitError::Interrupted {
                        phase: SerialTransmitPhase::Flush,
                        cause: SerialTransmitInterruption::PriorityStopRequested,
                        written_bytes,
                        record_bytes: record.len(),
                        maximum_duration,
                    });
                }
                _ = tokio::time::sleep_until(deadline) => {
                    return Err(SerialTransmitError::Interrupted {
                        phase: SerialTransmitPhase::Flush,
                        cause: SerialTransmitInterruption::DeadlineExceeded,
                        written_bytes,
                        record_bytes: record.len(),
                        maximum_duration,
                    });
                }
                result = &mut flush => result,
            }
        }
    };
    result.map_err(|source| SerialTransmitError::Flush {
        source,
        record_bytes: record.len(),
    })
}

async fn wait_for_priority_stop(priority_stop: Option<(&PriorityStopCoordinator, u64)>) -> u64 {
    match priority_stop {
        Some((priority_stop, completed_generation)) => {
            priority_stop.wait_after(completed_generation).await
        }
        None => std::future::pending().await,
    }
}

impl CachedCommand {
    fn remaining_at(self, now: Instant) -> RemainingLeaseMs {
        let elapsed_ms = now
            .checked_duration_since(self.completed_at)
            .map(duration_millis_ceil_saturating)
            .unwrap_or(u64::MAX);
        let controller_bound =
            u64::from(self.host_result.remaining_lease.get()).saturating_sub(elapsed_ms);
        let server_bound = self
            .server_deadline_exclusive
            .checked_duration_since(now)
            .map(|duration| u64::try_from(duration.as_millis()).unwrap_or(u64::MAX))
            .unwrap_or(0);
        let bounded = controller_bound.min(server_bound).min(u64::from(u16::MAX));
        let value = u16::try_from(bounded).unwrap_or(u16::MAX);
        RemainingLeaseMs::try_new(value).unwrap_or(RemainingLeaseMs::ZERO)
    }

    fn duplicate_result_at(self, now: Instant) -> HostCommandResult {
        HostCommandResult {
            result: HostCommandResultCode::DuplicateCached,
            remaining_lease: self.remaining_at(now),
            ..self.host_result
        }
    }
}

fn duration_millis_ceil_saturating(duration: Duration) -> u64 {
    let nanos = duration.as_nanos();
    let millis = nanos.saturating_add(999_999) / 1_000_000;
    u64::try_from(millis).unwrap_or(u64::MAX)
}

fn empty_capabilities() -> ControllerCapabilities {
    ControllerCapabilities::try_from_bits(0).expect("zero capability bits are a valid set")
}

const fn applied_result_disposition(
    result: AppliedResultCode,
) -> (Option<HostCommandResultCode>, Option<ForceStopReason>) {
    match result {
        AppliedResultCode::AppliedNew => (Some(HostCommandResultCode::AppliedNew), None),
        AppliedResultCode::DuplicateCached => (Some(HostCommandResultCode::DuplicateCached), None),
        AppliedResultCode::Stopped => (Some(HostCommandResultCode::Stopped), None),
        AppliedResultCode::RejectedExpired => (None, Some(ForceStopReason::LeaseExpired)),
        AppliedResultCode::RejectedSession => (None, Some(ForceStopReason::SessionReset)),
        AppliedResultCode::RejectedSequence => (None, Some(ForceStopReason::SequenceConflict)),
        AppliedResultCode::RejectedDomain | AppliedResultCode::FaultedStop => {
            (None, Some(ForceStopReason::ControllerFault))
        }
    }
}

#[derive(Debug)]
pub enum UdpServiceError {
    BindMustBeLoopback(SocketAddr),
    Bind(io::Error),
    Receive(io::Error),
}

impl fmt::Display for UdpServiceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BindMustBeLoopback(address) => {
                write!(formatter, "V2 command bind {address} is not loopback")
            }
            Self::Bind(source) => write!(formatter, "cannot bind V2 command UDP socket: {source}"),
            Self::Receive(source) => write!(formatter, "V2 command UDP receive failed: {source}"),
        }
    }
}

impl std::error::Error for UdpServiceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Bind(source) | Self::Receive(source) => Some(source),
            Self::BindMustBeLoopback(_) => None,
        }
    }
}

pub(crate) async fn bind_udp_socket(bind: SocketAddr) -> Result<UdpSocket, UdpServiceError> {
    if !bind.ip().is_loopback() || bind.port() == 0 {
        return Err(UdpServiceError::BindMustBeLoopback(bind));
    }
    UdpSocket::bind(bind).await.map_err(UdpServiceError::Bind)
}

#[cfg(test)]
async fn udp_service_on_socket(
    socket: UdpSocket,
    handle: Option<ActuationHandle>,
) -> Result<(), UdpServiceError> {
    udp_service_on_socket_inner(socket, handle, None).await
}

pub(crate) async fn udp_service_on_socket_until(
    socket: UdpSocket,
    handle: ActuationHandle,
    shutdown: oneshot::Receiver<()>,
) -> Result<(), UdpServiceError> {
    udp_service_on_socket_inner(socket, Some(handle), Some(shutdown)).await
}

enum PreparedUdpRequest {
    Ordinary(HostRequest),
    PriorityUnavailable(HostStop),
    PriorityLatched {
        coordinator: Arc<PriorityStopCoordinator>,
        request: HostStop,
        admission: Result<PreparedPriorityStopRequest, ActuationHandleError>,
    },
}

impl PreparedUdpRequest {
    fn new(handle: Option<&ActuationHandle>, request: HostRequest, source: SocketAddr) -> Self {
        match (handle, request) {
            (Some(handle), HostRequest::Stop(request)) => Self::PriorityLatched {
                coordinator: Arc::clone(&handle.priority_stop),
                request,
                admission: handle
                    .priority_stop
                    .prepare_latched_request(source, request),
            },
            (None, HostRequest::Stop(request)) => Self::PriorityUnavailable(request),
            (_, request) => Self::Ordinary(request),
        }
    }

    const fn request(&self) -> HostRequest {
        match self {
            Self::Ordinary(request) => *request,
            Self::PriorityUnavailable(request) | Self::PriorityLatched { request, .. } => {
                HostRequest::Stop(*request)
            }
        }
    }

    const fn is_priority(&self) -> bool {
        !matches!(self, Self::Ordinary(_))
    }
}

async fn udp_service_on_socket_inner(
    socket: UdpSocket,
    handle: Option<ActuationHandle>,
    mut shutdown: Option<oneshot::Receiver<()>>,
) -> Result<(), UdpServiceError> {
    let socket = Arc::new(socket);
    let mut buffer = [0_u8; MAX_RAW_FRAME_BYTES + 1];
    let mut ordinary_exchanges = JoinSet::new();
    let mut priority_exchanges = JoinSet::new();
    let ordinary_response_slots = Arc::new(Semaphore::new(MAX_UDP_ORDINARY_EXCHANGES_IN_FLIGHT));
    let priority_response_slots = Arc::new(Semaphore::new(MAX_UDP_PRIORITY_EXCHANGES_IN_FLIGHT));

    'serving: loop {
        drain_completed_udp_exchanges(&mut ordinary_exchanges);
        drain_completed_udp_exchanges(&mut priority_exchanges);
        let received = tokio::select! {
            biased;
            () = wait_for_udp_shutdown(&mut shutdown) => {
                break 'serving;
            }
            result = socket.recv_from(&mut buffer) => result,
        };
        let (length, source) = received.map_err(UdpServiceError::Receive)?;
        let first_received_at = Instant::now();
        let message = match decode_raw_frame(&buffer[..length]) {
            Ok(message) => message,
            Err(error) => {
                log::warn!("rejected malformed V2 datagram from {source}: {error}");
                continue;
            }
        };
        let request = match HostRequest::try_from(message) {
            Ok(request) => request,
            Err(error) => {
                log::warn!("rejected non-request V2 datagram from {source}: {error}");
                continue;
            }
        };
        // Priority admission is synchronous with datagram receipt. Even when
        // every bounded response slot is occupied, dropping the prepared
        // waiter below cannot undo the generation already latched for the
        // serial actor.
        let prepared_request = PreparedUdpRequest::new(handle.as_ref(), request, source);
        let priority = prepared_request.is_priority();
        let response_slot = if priority {
            Arc::clone(&priority_response_slots).try_acquire_owned()
        } else {
            Arc::clone(&ordinary_response_slots).try_acquire_owned()
        };
        let response_slot = match response_slot {
            Ok(response_slot) => response_slot,
            Err(_) => {
                let request = prepared_request.request();
                let response = match prepared_request {
                    PreparedUdpRequest::PriorityLatched {
                        admission: Ok(PreparedPriorityStopRequest::Immediate(result)),
                        ..
                    } => TimedActorResponse {
                        message: Message::HostStopResult(result),
                        calculated_at: Instant::now(),
                    },
                    PreparedUdpRequest::PriorityLatched {
                        admission: Err(error),
                        ..
                    } => {
                        log::error!("V2 priority stop could not be latched for {source}: {error}");
                        TimedActorResponse {
                            message: unavailable_response(request),
                            calculated_at: Instant::now(),
                        }
                    }
                    PreparedUdpRequest::PriorityLatched {
                        admission: Ok(PreparedPriorityStopRequest::Pending(_)),
                        ..
                    }
                    | PreparedUdpRequest::PriorityUnavailable(_)
                    | PreparedUdpRequest::Ordinary(_) => TimedActorResponse {
                        message: unavailable_response(request),
                        calculated_at: Instant::now(),
                    },
                };
                if let Err(error) =
                    try_send_udp_response(&socket, source, request, first_received_at, response)
                {
                    log::warn!(
                        "bounded V2 UDP overload response to {source} could not be emitted: {error}"
                    );
                }
                continue;
            }
        };

        let socket = Arc::clone(&socket);
        let handle = handle.clone();
        let exchange = async move {
            let _response_slot = response_slot;
            run_udp_exchange(socket, handle, source, first_received_at, prepared_request).await
        };
        if priority {
            priority_exchanges.spawn(exchange);
        } else {
            ordinary_exchanges.spawn(exchange);
        }
    }

    abort_udp_exchanges(&mut priority_exchanges).await;
    abort_udp_exchanges(&mut ordinary_exchanges).await;
    Ok(())
}

async fn run_udp_exchange(
    socket: Arc<UdpSocket>,
    handle: Option<ActuationHandle>,
    source: SocketAddr,
    first_received_at: Instant,
    prepared_request: PreparedUdpRequest,
) -> Result<(), UdpExchangeError> {
    let request = prepared_request.request();
    let response = match prepared_request {
        PreparedUdpRequest::PriorityLatched {
            coordinator,
            request: stop,
            admission: Ok(prepared),
        } => {
            match coordinator
                .await_prepared_request(source, stop, prepared)
                .await
            {
                Ok(result) => TimedActorResponse {
                    message: Message::HostStopResult(result),
                    calculated_at: Instant::now(),
                },
                Err(error) => {
                    log::error!("V2 serial actor could not answer {source}: {error}");
                    TimedActorResponse {
                        message: unavailable_response(request),
                        calculated_at: Instant::now(),
                    }
                }
            }
        }
        PreparedUdpRequest::PriorityLatched {
            admission: Err(error),
            ..
        } => {
            log::error!("V2 priority stop could not be latched for {source}: {error}");
            TimedActorResponse {
                message: unavailable_response(request),
                calculated_at: Instant::now(),
            }
        }
        PreparedUdpRequest::PriorityUnavailable(_) => TimedActorResponse {
            message: unavailable_response(request),
            calculated_at: Instant::now(),
        },
        PreparedUdpRequest::Ordinary(request) => match handle {
            Some(handle) => {
                match handle
                    .exchange_timed(source, first_received_at, request)
                    .await
                {
                    Ok(response) => response,
                    Err(error) => {
                        log::error!("V2 serial actor could not answer {source}: {error}");
                        TimedActorResponse {
                            message: unavailable_response(request),
                            calculated_at: Instant::now(),
                        }
                    }
                }
            }
            None => TimedActorResponse {
                message: unavailable_response(request),
                calculated_at: Instant::now(),
            },
        },
    };
    send_udp_response(&socket, source, request, first_received_at, response).await
}

async fn send_udp_response(
    socket: &UdpSocket,
    source: SocketAddr,
    request: HostRequest,
    first_received_at: Instant,
    response: TimedActorResponse,
) -> Result<(), UdpExchangeError> {
    let frame = udp_response_frame(request, first_received_at, response)?;
    let (sent, expected) = loop {
        match socket.try_send_to(frame.as_bytes(), source) {
            Ok(sent) => break (sent, frame.len()),
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                socket.writable().await.map_err(UdpExchangeError::Send)?;
            }
            Err(error) => return Err(UdpExchangeError::Send(error)),
        }
    };
    if sent != expected {
        return Err(UdpExchangeError::ShortSend {
            expected,
            actual: sent,
        });
    }
    Ok(())
}

fn try_send_udp_response(
    socket: &UdpSocket,
    source: SocketAddr,
    request: HostRequest,
    first_received_at: Instant,
    response: TimedActorResponse,
) -> Result<(), UdpExchangeError> {
    let frame = udp_response_frame(request, first_received_at, response)?;
    let sent = socket
        .try_send_to(frame.as_bytes(), source)
        .map_err(UdpExchangeError::Send)?;
    if sent != frame.len() {
        return Err(UdpExchangeError::ShortSend {
            expected: frame.len(),
            actual: sent,
        });
    }
    Ok(())
}

fn udp_response_frame(
    request: HostRequest,
    first_received_at: Instant,
    response: TimedActorResponse,
) -> Result<RawFrame, UdpExchangeError> {
    let emission_response = response_for_udp_emission(
        request,
        first_received_at,
        response.message,
        response.calculated_at,
        Instant::now(),
    );
    RawFrame::encode(emission_response).map_err(UdpExchangeError::Encode)
}

fn drain_completed_udp_exchanges(exchanges: &mut JoinSet<Result<(), UdpExchangeError>>) {
    while let Some(result) = exchanges.try_join_next() {
        log_udp_task_result(result);
    }
}

async fn abort_udp_exchanges(exchanges: &mut JoinSet<Result<(), UdpExchangeError>>) {
    exchanges.abort_all();
    while let Some(result) = exchanges.join_next().await {
        match result {
            Err(error) if error.is_cancelled() => {}
            result => log_udp_task_result(result),
        }
    }
}

async fn wait_for_udp_shutdown(shutdown: &mut Option<oneshot::Receiver<()>>) {
    match shutdown {
        Some(shutdown) => {
            let _ = shutdown.await;
        }
        None => std::future::pending().await,
    }
}

#[derive(Debug)]
enum UdpExchangeError {
    Encode(robot_protocol::v2::EncodeError),
    Send(io::Error),
    ShortSend { expected: usize, actual: usize },
}

impl fmt::Display for UdpExchangeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Encode(source) => write!(formatter, "cannot encode V2 UDP result: {source}"),
            Self::Send(source) => write!(formatter, "cannot send V2 UDP result: {source}"),
            Self::ShortSend { expected, actual } => write!(
                formatter,
                "V2 UDP result reported {actual} bytes sent instead of {expected}"
            ),
        }
    }
}

fn log_udp_task_result(result: Result<Result<(), UdpExchangeError>, tokio::task::JoinError>) {
    match result {
        Ok(Ok(())) => {}
        Ok(Err(error)) => log::error!("V2 UDP exchange failed: {error}"),
        Err(error) => log::error!("V2 UDP exchange task failed: {error}"),
    }
}

fn unavailable_response(request: HostRequest) -> Message {
    // This boundary has no controller observation. Concrete output fields are
    // frozen-wire placeholders and are deliberately classified as Unknown by
    // each response type's `output_evidence` method.
    match request {
        HostRequest::Acquire(value) => Message::AcquireResult(AcquireResult {
            controller_uid: value.expected_controller_uid,
            boot_id: value.expected_boot_id,
            request_id: value.request_id,
            control_epoch: None,
            result: AcquireResultCode::ControllerUnavailable,
            capabilities: empty_capabilities(),
            faults: ControllerFaults::NONE,
            observed_firmware_abi: value.expected_firmware_abi,
            observed_firmware_build_id: value.expected_firmware_build_id,
            observed_actuator_config_fingerprint: value.expected_actuator_config_fingerprint,
        }),
        HostRequest::Command(value) => Message::HostCommandResult(HostCommandResult {
            controller_uid: value.controller_uid,
            boot_id: value.boot_id,
            control_epoch: value.control_epoch,
            sequence: value.sequence,
            result: HostCommandResultCode::ControllerRestarted,
            requested_timer_pwm: value.requested_timer_pwm,
            controller_timer_pwm: TimerPwm::ZERO,
            output_state: OutputState::Disabled,
            controller_applied_at: ControllerUptimeMsWrapping::new(0),
            controller_expires_at: robot_protocol::v2::ControllerDeadlineMsWrapping::new(0),
            remaining_lease: RemainingLeaseMs::ZERO,
            faults: ControllerFaults::NONE,
        }),
        HostRequest::Stop(value) => Message::HostStopResult(HostStopResult {
            controller_uid: value.controller_uid,
            observed_boot_id: value.target_boot_id,
            request_id: value.request_id,
            result: StopResultCode::ControllerUnavailable,
            output_state: OutputState::Disabled,
            controller_uptime: ControllerUptimeMsWrapping::new(0),
            faults: ControllerFaults::NONE,
        }),
        HostRequest::Status(value) => Message::StatusReport(StatusReport {
            controller_uid: value.expected_controller_uid,
            observed_boot_id: TargetBootId::Any,
            request_id: value.request_id,
            status: StatusCode::Disconnected,
            control_epoch: None,
            controller_uptime: ControllerUptimeMsWrapping::new(0),
            capabilities: empty_capabilities(),
            output_state: OutputState::Disabled,
            controller_timer_pwm: TimerPwm::ZERO,
            remaining_lease: RemainingLeaseMs::ZERO,
            faults: ControllerFaults::NONE,
        }),
    }
}

fn response_for_udp_emission(
    request: HostRequest,
    first_received_at: Instant,
    response: Message,
    remaining_lease_calculated_at: Instant,
    emitted_at: Instant,
) -> Message {
    let post_calculation_elapsed_ms = emitted_at
        .checked_duration_since(remaining_lease_calculated_at)
        .map(duration_millis_ceil_saturating)
        .unwrap_or(u64::MAX)
        .saturating_add(UDP_EMISSION_QUANTIZATION_MARGIN_MS);
    match (request, response) {
        (HostRequest::Command(request), Message::HostCommandResult(mut result)) => {
            let actor_bound =
                u64::from(result.remaining_lease.get()).saturating_sub(post_calculation_elapsed_ms);
            let server_bound = first_received_at
                .checked_add(Duration::from_millis(u64::from(request.lease.get())))
                .and_then(|deadline| deadline.checked_duration_since(emitted_at))
                .map(|duration| u64::try_from(duration.as_millis()).unwrap_or(u64::MAX))
                .unwrap_or(0);
            result.remaining_lease = bounded_remaining_lease(actor_bound.min(server_bound));
            Message::HostCommandResult(result)
        }
        (HostRequest::Status(_), Message::StatusReport(mut result)) => {
            let remaining =
                u64::from(result.remaining_lease.get()).saturating_sub(post_calculation_elapsed_ms);
            result.remaining_lease = bounded_remaining_lease(remaining);
            Message::StatusReport(result)
        }
        (_, response) => response,
    }
}

fn bounded_remaining_lease(value: u64) -> RemainingLeaseMs {
    let value = value.min(u64::from(robot_protocol::v2::MAX_V2_COMMAND_LEASE_MS));
    let value = u16::try_from(value).unwrap_or(robot_protocol::v2::MAX_V2_COMMAND_LEASE_MS);
    RemainingLeaseMs::try_new(value).unwrap_or(RemainingLeaseMs::ZERO)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::num::{NonZeroU16, NonZeroU32};
    use std::pin::Pin;
    #[cfg(feature = "qualification-fault-injection")]
    use std::sync::atomic::AtomicBool;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
    use std::task::{Context, Poll};

    use robot_command_client::{
        ClientConfig, DisarmedCommandClient, MonotonicInstant, PendingPhysicalCommand,
        StopRecoveryPolicy, SystemMonotonicClock, TimeoutNs, UdpEndpoint, UdpV2Transport,
    };
    use robot_protocol::v2::{
        ApplyPwm, ControllerDeadlineMsWrapping, ControllerReady, ReadinessFlags, V2CommandLeaseMs,
    };
    #[cfg(feature = "qualification-fault-injection")]
    use tokio::io::ReadBuf;
    use tokio::io::{AsyncWrite, DuplexStream};
    use tokio::time::{sleep, timeout};

    const IO_TIMEOUT: Duration = Duration::from_millis(250);
    const SHORT_ABSENCE: Duration = Duration::from_millis(5);

    #[derive(Clone, Copy)]
    enum WriteStep {
        Bytes(usize),
        Pending,
    }

    struct ScriptedWriter {
        steps: VecDeque<WriteStep>,
        flush_pending: bool,
        written: Vec<u8>,
        observed_bytes: Arc<AtomicUsize>,
    }

    impl ScriptedWriter {
        fn new(steps: impl IntoIterator<Item = WriteStep>, flush_pending: bool) -> Self {
            Self {
                steps: steps.into_iter().collect(),
                flush_pending,
                written: Vec::new(),
                observed_bytes: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    impl AsyncWrite for ScriptedWriter {
        fn poll_write(
            mut self: Pin<&mut Self>,
            _context: &mut Context<'_>,
            buffer: &[u8],
        ) -> Poll<io::Result<usize>> {
            match self
                .steps
                .pop_front()
                .unwrap_or(WriteStep::Bytes(usize::MAX))
            {
                WriteStep::Pending => {
                    self.steps.push_front(WriteStep::Pending);
                    Poll::Pending
                }
                WriteStep::Bytes(maximum) => {
                    let count = buffer.len().min(maximum);
                    self.written.extend_from_slice(&buffer[..count]);
                    self.observed_bytes
                        .store(self.written.len(), AtomicOrdering::Release);
                    Poll::Ready(Ok(count))
                }
            }
        }

        fn poll_flush(self: Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<io::Result<()>> {
            if self.flush_pending {
                Poll::Pending
            } else {
                Poll::Ready(Ok(()))
            }
        }

        fn poll_shutdown(self: Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<io::Result<()>> {
            Poll::Ready(Ok(()))
        }
    }

    #[cfg(feature = "qualification-fault-injection")]
    #[derive(Clone, Copy)]
    enum QualificationPrefixTransportFault {
        Write,
        Flush,
        Deadline,
        Shutdown,
        PriorityStop,
    }

    #[cfg(feature = "qualification-fault-injection")]
    struct QualificationFaultControl {
        armed: Arc<AtomicBool>,
        triggered: Arc<AtomicBool>,
    }

    #[cfg(feature = "qualification-fault-injection")]
    impl QualificationFaultControl {
        fn arm(&self) {
            self.armed.store(true, AtomicOrdering::Release);
        }

        async fn wait_until_triggered(&self) {
            timeout(IO_TIMEOUT, async {
                while !self.triggered.load(AtomicOrdering::Acquire) {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("qualification transport fault trigger timeout");
        }
    }

    #[cfg(feature = "qualification-fault-injection")]
    struct QualificationFaultingDuplex {
        inner: DuplexStream,
        fault: QualificationPrefixTransportFault,
        armed: Arc<AtomicBool>,
        observed: Arc<AtomicBool>,
        triggered: bool,
        fail_next_flush: bool,
    }

    #[cfg(feature = "qualification-fault-injection")]
    impl QualificationFaultingDuplex {
        fn new(
            inner: DuplexStream,
            fault: QualificationPrefixTransportFault,
        ) -> (Self, QualificationFaultControl) {
            let armed = Arc::new(AtomicBool::new(false));
            let observed = Arc::new(AtomicBool::new(false));
            (
                Self {
                    inner,
                    fault,
                    armed: Arc::clone(&armed),
                    observed: Arc::clone(&observed),
                    triggered: false,
                    fail_next_flush: false,
                },
                QualificationFaultControl {
                    armed,
                    triggered: observed,
                },
            )
        }
    }

    #[cfg(feature = "qualification-fault-injection")]
    impl AsyncRead for QualificationFaultingDuplex {
        fn poll_read(
            mut self: Pin<&mut Self>,
            context: &mut Context<'_>,
            buffer: &mut ReadBuf<'_>,
        ) -> Poll<io::Result<()>> {
            Pin::new(&mut self.inner).poll_read(context, buffer)
        }
    }

    #[cfg(feature = "qualification-fault-injection")]
    impl AsyncWrite for QualificationFaultingDuplex {
        fn poll_write(
            mut self: Pin<&mut Self>,
            context: &mut Context<'_>,
            buffer: &[u8],
        ) -> Poll<io::Result<usize>> {
            if self.armed.load(AtomicOrdering::Acquire)
                && !self.triggered
                && buffer.len() == QUALIFICATION_PARTIAL_UART_PREFIX_BYTES
            {
                let count = match Pin::new(&mut self.inner).poll_write(context, buffer) {
                    Poll::Ready(Ok(count)) => count,
                    Poll::Ready(Err(source)) => return Poll::Ready(Err(source)),
                    Poll::Pending => return Poll::Pending,
                };
                self.triggered = true;
                self.observed.store(true, AtomicOrdering::Release);
                return match self.fault {
                    QualificationPrefixTransportFault::Write => Poll::Ready(Err(io::Error::new(
                        io::ErrorKind::BrokenPipe,
                        "injected write failure after forwarding qualification prefix",
                    ))),
                    QualificationPrefixTransportFault::Flush => {
                        self.fail_next_flush = true;
                        Poll::Ready(Ok(count))
                    }
                    QualificationPrefixTransportFault::Deadline
                    | QualificationPrefixTransportFault::Shutdown
                    | QualificationPrefixTransportFault::PriorityStop => Poll::Pending,
                };
            }
            Pin::new(&mut self.inner).poll_write(context, buffer)
        }

        fn poll_flush(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<io::Result<()>> {
            if self.fail_next_flush {
                self.fail_next_flush = false;
                return Poll::Ready(Err(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "injected flush failure after qualification prefix",
                )));
            }
            Pin::new(&mut self.inner).poll_flush(context)
        }

        fn poll_shutdown(
            mut self: Pin<&mut Self>,
            context: &mut Context<'_>,
        ) -> Poll<io::Result<()>> {
            Pin::new(&mut self.inner).poll_shutdown(context)
        }
    }

    #[tokio::test]
    async fn serial_transmit_completes_exactly_across_partial_writes() {
        let mut writer = ScriptedWriter::new(
            [
                WriteStep::Bytes(1),
                WriteStep::Bytes(2),
                WriteStep::Bytes(3),
            ],
            false,
        );
        let record = b"partial-record";
        transmit_serial_record(&mut writer, record, Duration::from_millis(25), None, None)
            .await
            .expect("bounded partial writes and flush complete");
        assert_eq!(writer.written, record);
    }

    #[tokio::test]
    async fn serial_transmit_reports_a_delayed_flush_without_hanging() {
        let mut writer = ScriptedWriter::new([WriteStep::Bytes(usize::MAX)], true);
        let record = b"complete-before-flush";
        let error =
            transmit_serial_record(&mut writer, record, Duration::from_millis(5), None, None)
                .await
                .expect_err("a flush that never completes reaches the one transmit deadline");
        assert!(matches!(
            error,
            SerialTransmitError::Interrupted {
                phase: SerialTransmitPhase::Flush,
                cause: SerialTransmitInterruption::DeadlineExceeded,
                written_bytes,
                record_bytes,
                ..
            } if written_bytes == record.len() && record_bytes == record.len()
        ));
    }

    #[tokio::test]
    async fn shutdown_cancels_a_partial_serial_write_with_exact_uncertainty() {
        let writer = ScriptedWriter::new([WriteStep::Bytes(3), WriteStep::Pending], false);
        let observed = Arc::clone(&writer.observed_bytes);
        let shutdown = Arc::new(ActuationShutdownSignal::new());
        let request_shutdown = Arc::clone(&shutdown);
        let task = tokio::spawn(async move {
            let mut writer = writer;
            transmit_serial_record(
                &mut writer,
                b"partially-written-record",
                Duration::from_secs(1),
                Some(&shutdown),
                None,
            )
            .await
        });
        timeout(Duration::from_millis(50), async {
            while observed.load(AtomicOrdering::Acquire) != 3 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("first partial write occurs");
        request_shutdown.request(ActuationShutdownReason::Operator);
        let error = timeout(Duration::from_millis(50), task)
            .await
            .expect("shutdown promptly wakes the blocked write")
            .expect("transmit task joins")
            .expect_err("partial record cannot become a success");
        assert!(matches!(
            error,
            SerialTransmitError::Interrupted {
                phase: SerialTransmitPhase::Write,
                cause: SerialTransmitInterruption::ShutdownRequested,
                written_bytes: 3,
                ..
            }
        ));
    }

    #[test]
    fn controller_rejection_stop_reasons_preserve_the_typed_cause() {
        for (result, reason) in [
            (
                AppliedResultCode::RejectedExpired,
                ForceStopReason::LeaseExpired,
            ),
            (
                AppliedResultCode::RejectedSession,
                ForceStopReason::SessionReset,
            ),
            (
                AppliedResultCode::RejectedSequence,
                ForceStopReason::SequenceConflict,
            ),
            (
                AppliedResultCode::RejectedDomain,
                ForceStopReason::ControllerFault,
            ),
            (
                AppliedResultCode::FaultedStop,
                ForceStopReason::ControllerFault,
            ),
        ] {
            assert_eq!(applied_result_disposition(result), (None, Some(reason)));
        }
        for (result, host) in [
            (
                AppliedResultCode::AppliedNew,
                HostCommandResultCode::AppliedNew,
            ),
            (
                AppliedResultCode::DuplicateCached,
                HostCommandResultCode::DuplicateCached,
            ),
            (AppliedResultCode::Stopped, HostCommandResultCode::Stopped),
        ] {
            assert_eq!(applied_result_disposition(result), (Some(host), None));
        }
    }

    fn uid() -> ControllerUid {
        ControllerUid::try_new([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).expect("nonzero UID")
    }

    fn wrong_uid() -> ControllerUid {
        ControllerUid::try_new([12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]).expect("nonzero UID")
    }

    fn boot(value: u64) -> robot_protocol::v2::ControllerBootId {
        robot_protocol::v2::ControllerBootId::try_new(value).expect("nonzero boot ID")
    }

    fn epoch() -> ControlEpoch {
        ControlEpoch::try_new(0x1020_3040).expect("nonzero epoch")
    }

    fn fingerprint() -> ActuatorConfigFingerprint {
        ActuatorConfigFingerprint::try_new([
            0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee,
            0xf0, 0x0f,
        ])
        .expect("nonzero fingerprint")
    }

    fn capabilities() -> ControllerCapabilities {
        ControllerCapabilities::try_from_bits(ControllerCapabilities::REQUIRED_BITS)
            .expect("known capability bits")
    }

    fn candidate_capabilities() -> ControllerCapabilities {
        ControllerCapabilities::try_from_bits(
            ControllerCapabilities::SOFTWARE_GUARD_BITS
                | ControllerCapabilities::OPERATOR_SUPERVISED_FOUR_PWM_CANDIDATE,
        )
        .expect("canonical candidate capability bits")
    }

    fn candidate_fingerprint() -> ActuatorConfigFingerprint {
        ActuatorConfigFingerprint::try_new(*b"KIKO-4PWM-CAND1!")
            .expect("canonical candidate fingerprint")
    }

    fn readiness() -> ReadinessFlags {
        ReadinessFlags::try_from_bits(ReadinessFlags::READY_BITS).expect("known readiness bits")
    }

    fn stopped_readiness() -> ReadinessFlags {
        ReadinessFlags::try_from_bits(ReadinessFlags::STOPPED_READY_BITS)
            .expect("known stopped readiness bits")
    }

    fn actor_config() -> ActorConfig {
        ActorConfig {
            controller_uid: uid(),
            firmware_abi: 2,
            firmware_build_id: 42,
            actuator_config_fingerprint: fingerprint(),
            heartbeat_period: HeartbeatPeriodMs::try_new(20).expect("heartbeat period"),
            maximum_heartbeat_age: Duration::from_millis(80),
            minimum_host_command_interval: Duration::from_millis(10),
            serial_transmit_timeout: Duration::from_millis(10),
            serial_applied_ack_timeout: Duration::from_millis(25),
            controller_clock_abs_error_ppm_bound: NonZeroU32::new(1_000).expect("ppm"),
            deadline_quantization_margin_ms: NonZeroU16::new(1).expect("margin"),
            expected_max_abs_pwm_percent: MaxAbsPwmPercent::try_new(50).expect("PWM maximum"),
            expected_pwm_frequency: PwmFrequencyHz::try_new(20_000).expect("PWM frequency"),
            expected_watchdog_nominal_period: WatchdogNominalPeriodMs::try_new(250)
                .expect("watchdog period"),
            expected_neutral_output: NeutralOutput::BothLow,
            expected_physical_stop_semantics: PhysicalStopSemantics::CoastVerified,
            controller_session_class: ControllerSessionClass::ProductionExternalInterlocks,
            maximum_command_step_percent: None,
            #[cfg(feature = "qualification-fault-injection")]
            qualification_serial_fault: None,
        }
    }

    fn candidate_actor_config() -> ActorConfig {
        ActorConfig {
            controller_uid: uid(),
            firmware_abi: 2,
            firmware_build_id: 0x0002_1001,
            actuator_config_fingerprint: candidate_fingerprint(),
            heartbeat_period: HeartbeatPeriodMs::try_new(20).expect("heartbeat period"),
            maximum_heartbeat_age: Duration::from_millis(80),
            minimum_host_command_interval: Duration::from_millis(10),
            serial_transmit_timeout: Duration::from_millis(10),
            serial_applied_ack_timeout: Duration::from_millis(25),
            controller_clock_abs_error_ppm_bound: NonZeroU32::new(1_000).expect("ppm"),
            deadline_quantization_margin_ms: NonZeroU16::new(1).expect("margin"),
            expected_max_abs_pwm_percent: MaxAbsPwmPercent::try_new(30).expect("candidate cap"),
            expected_pwm_frequency: PwmFrequencyHz::try_new(20_000).expect("PWM frequency"),
            expected_watchdog_nominal_period: WatchdogNominalPeriodMs::try_new(250)
                .expect("watchdog period"),
            expected_neutral_output: NeutralOutput::BothLow,
            expected_physical_stop_semantics: PhysicalStopSemantics::Unverified,
            controller_session_class: ControllerSessionClass::OperatorSupervisedFourPwmCandidate,
            maximum_command_step_percent: Some(
                OPERATOR_SUPERVISED_FOUR_PWM_MAX_COMMAND_STEP_PERCENT,
            ),
            #[cfg(feature = "qualification-fault-injection")]
            qualification_serial_fault: None,
        }
    }

    #[test]
    fn candidate_server_rechecks_firmware_step_from_last_exact_receipt() {
        let maximum_step = candidate_actor_config().maximum_command_step_percent;
        let previous = TimerPwm::try_new(5, -5).expect("bounded previous PWM");

        assert!(candidate_command_step_is_admitted(
            previous,
            TimerPwm::try_new(10, -10).expect("bounded exact-boundary PWM"),
            maximum_step,
        ));
        assert!(!candidate_command_step_is_admitted(
            previous,
            TimerPwm::try_new(11, -10).expect("bounded excessive PWM"),
            maximum_step,
        ));
        assert!(!candidate_command_step_is_admitted(
            previous,
            TimerPwm::try_new(10, -11).expect("bounded excessive PWM"),
            maximum_step,
        ));
        assert!(
            candidate_command_step_is_admitted(
                TimerPwm::try_new(30, -30).expect("bounded previous PWM"),
                TimerPwm::ZERO,
                maximum_step,
            ),
            "full zero must remain the firmware's immediate fail-closed bypass",
        );
        assert!(
            candidate_command_step_is_admitted(
                previous,
                TimerPwm::try_new(100, -100).expect("protocol-domain PWM"),
                None,
            ),
            "the production class has no candidate-only step contract",
        );
    }

    fn client_config(endpoint: SocketAddr) -> ClientConfig {
        let timeout = TimeoutNs::try_new(200_000_000).expect("bounded client timeout");
        ClientConfig::new(
            UdpEndpoint::try_new(endpoint).expect("loopback endpoint"),
            uid(),
            NonZeroU16::new(2).expect("nonzero firmware ABI"),
            NonZeroU32::new(42).expect("nonzero build ID"),
            fingerprint(),
            timeout,
            timeout,
            timeout,
            StopRecoveryPolicy::try_new(1, timeout).expect("bounded stop recovery"),
            V2CommandLeaseMs::try_new(120).expect("zero-acquisition lease"),
        )
    }

    fn hello_with(
        controller_uid: ControllerUid,
        boot_id: robot_protocol::v2::ControllerBootId,
    ) -> ControllerHello {
        ControllerHello {
            controller_uid,
            boot_id,
            firmware_abi: 2,
            firmware_build_id: 42,
            capabilities: capabilities(),
            max_abs_pwm_percent: MaxAbsPwmPercent::try_new(50).expect("PWM maximum"),
            max_command_lease: V2CommandLeaseMs::try_new(250).expect("lease maximum"),
            output_state: OutputState::Disabled,
            actuator_config_fingerprint: fingerprint(),
            watchdog_nominal_period: WatchdogNominalPeriodMs::try_new(250)
                .expect("watchdog period"),
            pwm_frequency: PwmFrequencyHz::try_new(20_000).expect("PWM frequency"),
            neutral_output: NeutralOutput::BothLow,
            physical_stop_semantics: PhysicalStopSemantics::CoastVerified,
        }
    }

    fn candidate_hello() -> ControllerHello {
        ControllerHello {
            controller_uid: uid(),
            boot_id: boot(7),
            firmware_abi: 2,
            firmware_build_id: 0x0002_1001,
            capabilities: candidate_capabilities(),
            max_abs_pwm_percent: MaxAbsPwmPercent::try_new(30).expect("candidate cap"),
            max_command_lease: V2CommandLeaseMs::try_new(250).expect("lease maximum"),
            output_state: OutputState::Disabled,
            actuator_config_fingerprint: candidate_fingerprint(),
            watchdog_nominal_period: WatchdogNominalPeriodMs::try_new(250)
                .expect("watchdog period"),
            pwm_frequency: PwmFrequencyHz::try_new(20_000).expect("PWM frequency"),
            neutral_output: NeutralOutput::BothLow,
            physical_stop_semantics: PhysicalStopSemantics::Unverified,
        }
    }

    fn ready_at(
        boot_id: robot_protocol::v2::ControllerBootId,
        controller_uptime: u32,
    ) -> ControllerReady {
        ControllerReady {
            controller_uid: uid(),
            boot_id,
            control_epoch: epoch(),
            controller_uptime: ControllerUptimeMsWrapping::new(controller_uptime),
            capabilities: capabilities(),
            output_state: OutputState::Disabled,
            faults: ControllerFaults::NONE,
        }
    }

    fn zero_heartbeat(boot_id: robot_protocol::v2::ControllerBootId) -> Heartbeat {
        zero_heartbeat_at(boot_id, 1_000)
    }

    fn zero_heartbeat_at(
        boot_id: robot_protocol::v2::ControllerBootId,
        controller_uptime: u32,
    ) -> Heartbeat {
        Heartbeat {
            controller_uid: uid(),
            boot_id,
            control_epoch: None,
            last_sequence: None,
            controller_uptime: ControllerUptimeMsWrapping::new(controller_uptime),
            expires_at: ControllerDeadlineMsWrapping::new(controller_uptime),
            timer_pwm: TimerPwm::ZERO,
            output_state: OutputState::ZeroPwm,
            readiness: stopped_readiness(),
            faults: ControllerFaults::NONE,
        }
    }

    fn no_session_heartbeat(
        boot_id: robot_protocol::v2::ControllerBootId,
        controller_uptime: u32,
    ) -> Heartbeat {
        Heartbeat {
            readiness: ReadinessFlags::try_from_bits(ReadinessFlags::WATCHDOG_RUNNING)
                .expect("no-session watchdog readiness"),
            ..zero_heartbeat_at(boot_id, controller_uptime)
        }
    }

    fn established_stopped_heartbeat(
        boot_id: robot_protocol::v2::ControllerBootId,
        control_epoch: ControlEpoch,
        sequence: V2CommandSequence,
        uptime: u32,
    ) -> Heartbeat {
        Heartbeat {
            controller_uid: uid(),
            boot_id,
            control_epoch: Some(control_epoch),
            last_sequence: Some(sequence),
            controller_uptime: ControllerUptimeMsWrapping::new(uptime),
            expires_at: ControllerDeadlineMsWrapping::new(uptime),
            timer_pwm: TimerPwm::ZERO,
            output_state: OutputState::Disabled,
            readiness: stopped_readiness(),
            faults: ControllerFaults::NONE,
        }
    }

    fn acquire(boot_id: robot_protocol::v2::ControllerBootId, request_id: u32) -> AcquireControl {
        AcquireControl {
            expected_controller_uid: uid(),
            expected_boot_id: boot_id,
            request_id: RequestId::new(request_id),
            expected_firmware_abi: 2,
            expected_firmware_build_id: 42,
            expected_actuator_config_fingerprint: fingerprint(),
        }
    }

    fn command(
        boot_id: robot_protocol::v2::ControllerBootId,
        sequence: u32,
        pwm: TimerPwm,
    ) -> HostCommand {
        HostCommand {
            controller_uid: uid(),
            boot_id,
            control_epoch: epoch(),
            sequence: V2CommandSequence::new(sequence),
            lease: V2CommandLeaseMs::try_new(120).expect("command lease"),
            requested_timer_pwm: pwm,
        }
    }

    fn applied(apply: ApplyPwm) -> AppliedResult {
        AppliedResult {
            controller_uid: apply.controller_uid,
            boot_id: apply.boot_id,
            control_epoch: apply.control_epoch,
            sequence: apply.sequence,
            result: AppliedResultCode::AppliedNew,
            timer_pwm: apply.timer_pwm,
            output_state: if apply.timer_pwm.is_zero() {
                OutputState::ZeroPwm
            } else {
                OutputState::NonzeroPwm
            },
            applied_at: ControllerUptimeMsWrapping::new(apply.expires_at.get().wrapping_sub(100)),
            expires_at: apply.expires_at,
            faults: ControllerFaults::NONE,
        }
    }

    fn odometry(
        boot_id: robot_protocol::v2::ControllerBootId,
        sample: u32,
    ) -> ObservationalOdometry {
        let sample_delta = i16::try_from(sample).expect("test samples fit i16");
        ObservationalOdometry {
            controller_uid: uid(),
            boot_id,
            control_epoch: Some(epoch()),
            left_estimated_extended_ticks_wrapping:
                robot_protocol::EstimatedWrappingEncoderTicks::new_wrapping(i64::from(sample)),
            right_estimated_extended_ticks_wrapping:
                robot_protocol::EstimatedWrappingEncoderTicks::new_wrapping(-i64::from(sample)),
            left_sample_delta_ticks_modulo: robot_protocol::ModuloEncoderDeltaTicks::new_modulo(
                sample_delta,
            ),
            right_sample_delta_ticks_modulo: robot_protocol::ModuloEncoderDeltaTicks::new_modulo(
                -sample_delta,
            ),
            controller_uptime: ControllerUptimeMsWrapping::new(1_001_u32.wrapping_add(sample)),
        }
    }

    fn confirmed_stop(
        stop: ForceStop,
        boot_id: robot_protocol::v2::ControllerBootId,
    ) -> HostStopResult {
        HostStopResult {
            controller_uid: stop.controller_uid,
            observed_boot_id: TargetBootId::Exact(boot_id),
            request_id: stop.request_id,
            result: StopResultCode::ControllerConfirmed,
            output_state: OutputState::Disabled,
            controller_uptime: ControllerUptimeMsWrapping::new(1_001),
            faults: ControllerFaults::NONE,
        }
    }

    struct FakeController {
        stream: DuplexStream,
        decoder: UartStreamDecoder,
    }

    impl FakeController {
        fn new(stream: DuplexStream) -> Self {
            Self {
                stream,
                decoder: UartStreamDecoder::new(),
            }
        }

        async fn send(&mut self, message: Message) {
            let record = UartRecord::encode(message).expect("fake message encodes");
            self.stream
                .write_all(record.as_bytes())
                .await
                .expect("fake serial write");
        }

        async fn send_corrupted(&mut self, message: Message) {
            let record = UartRecord::encode(message).expect("fake message encodes");
            let mut bytes = record.as_bytes().to_vec();
            let index = bytes[..bytes.len() - 1]
                .iter()
                .position(|byte| *byte > 1)
                .expect("encoded record has a mutable non-code byte");
            bytes[index] ^= 1;
            self.stream
                .write_all(&bytes)
                .await
                .expect("fake corrupt serial write");
        }

        async fn receive(&mut self) -> Message {
            self.receive_result()
                .await
                .expect("actor emits valid V2 only")
        }

        async fn receive_result(&mut self) -> Result<Message, UartStreamError> {
            timeout(IO_TIMEOUT, async {
                loop {
                    let byte = self.stream.read_u8().await.expect("fake serial read");
                    if let Some(decoded) = self.decoder.push(byte) {
                        return decoded;
                    }
                }
            })
            .await
            .expect("actor serial response timeout")
        }

        async fn assert_no_message(&mut self) {
            assert!(timeout(SHORT_ABSENCE, self.stream.read_u8()).await.is_err());
        }
    }

    struct Harness {
        handle: ActuationHandle,
        shutdown: ActuationShutdownHandle,
        controller: FakeController,
        actor: JoinHandle<Result<(), ActuationActorError>>,
        source: SocketAddr,
        boot_id: robot_protocol::v2::ControllerBootId,
    }

    impl Harness {
        async fn ready() -> Self {
            Self::ready_with_serial_capacity(4_096).await
        }

        async fn ready_with_serial_capacity(serial_capacity: usize) -> Self {
            Self::ready_with_actor_config(serial_capacity, actor_config()).await
        }

        async fn ready_with_actor_config(serial_capacity: usize, config: ActorConfig) -> Self {
            let (harness, startup_ready) =
                Self::awaiting_startup_heartbeat_with_config(serial_capacity, 1_000, config).await;
            Self::finish_ready(harness, startup_ready).await
        }

        #[cfg(feature = "qualification-fault-injection")]
        async fn ready_with_qualification_transport_fault(
            fault: QualificationPrefixTransportFault,
        ) -> (Self, QualificationFaultControl) {
            let mut config = actor_config();
            config.qualification_serial_fault = Some(
                OperatorSupervisedCandidateSerialFaultInjection::PartialUartRecordOnFirstNonzeroCommand,
            );
            let (actor_stream, controller_stream) = tokio::io::duplex(4_096);
            let (actor_stream, control) = QualificationFaultingDuplex::new(actor_stream, fault);
            let (harness, startup_ready) = Self::awaiting_startup_heartbeat_with_transport(
                actor_stream,
                controller_stream,
                1_000,
                config,
            )
            .await;
            (Self::finish_ready(harness, startup_ready).await, control)
        }

        async fn finish_ready(mut harness: Self, mut startup_ready: oneshot::Receiver<()>) -> Self {
            assert!(
                timeout(SHORT_ABSENCE, &mut startup_ready).await.is_err(),
                "ControllerReady without an exact stopped heartbeat is not startup evidence"
            );
            harness
                .controller
                .send(Message::Heartbeat(zero_heartbeat(harness.boot_id)))
                .await;
            timeout(IO_TIMEOUT, startup_ready)
                .await
                .expect("startup-ready signal timeout")
                .expect("actor retains startup-ready sender");
            harness.wait_until_ready_stopped().await;
            harness
        }

        async fn awaiting_startup_heartbeat(
            serial_capacity: usize,
            ready_uptime: u32,
        ) -> (Self, oneshot::Receiver<()>) {
            Self::awaiting_startup_heartbeat_with_config(
                serial_capacity,
                ready_uptime,
                actor_config(),
            )
            .await
        }

        async fn awaiting_startup_heartbeat_with_config(
            serial_capacity: usize,
            ready_uptime: u32,
            config: ActorConfig,
        ) -> (Self, oneshot::Receiver<()>) {
            let (actor_stream, controller_stream) = tokio::io::duplex(serial_capacity);
            Self::awaiting_startup_heartbeat_with_transport(
                actor_stream,
                controller_stream,
                ready_uptime,
                config,
            )
            .await
        }

        async fn awaiting_startup_heartbeat_with_transport<Transport>(
            actor_stream: Transport,
            controller_stream: DuplexStream,
            ready_uptime: u32,
            config: ActorConfig,
        ) -> (Self, oneshot::Receiver<()>)
        where
            Transport: AsyncRead + AsyncWrite + Unpin + Send + 'static,
        {
            let boot_id = boot(7);
            let (handle, startup_ready, actor) = spawn_actor(
                actor_stream,
                config,
                Arc::new(NoopActuationTelemetry),
                UartStreamDecoder::new(),
            );
            let mut controller = FakeController::new(controller_stream);
            controller
                .send(Message::ControllerHello(hello_with(uid(), boot_id)))
                .await;
            let stop = match controller.receive().await {
                Message::ForceStop(value) => value,
                other => panic!("expected startup ForceStop, got {:?}", other.kind()),
            };
            controller.assert_no_message().await;
            controller
                .send(Message::HostStopResult(confirmed_stop(stop, boot_id)))
                .await;
            assert!(matches!(
                controller.receive().await,
                Message::BeginSession(_)
            ));
            controller
                .send(Message::ControllerReady(ready_at(boot_id, ready_uptime)))
                .await;
            let source = "127.0.0.1:41000".parse().expect("source address");
            let shutdown = handle.shutdown_handle();
            (
                Self {
                    handle,
                    shutdown,
                    controller,
                    actor,
                    source,
                    boot_id,
                },
                startup_ready,
            )
        }

        async fn wait_until_ready_stopped(&mut self) {
            for request_id in 0..20 {
                let query = StatusQuery {
                    expected_controller_uid: uid(),
                    request_id: RequestId::new(request_id),
                };
                let message = self
                    .handle
                    .exchange(self.source, Instant::now(), Message::StatusQuery(query))
                    .await
                    .expect("status response");
                let Message::StatusReport(report) = message else {
                    panic!("wrong status response")
                };
                if report.status == StatusCode::ReadyStopped {
                    return;
                }
                sleep(Duration::from_millis(1)).await;
            }
            panic!("actor did not become ready-stopped")
        }

        async fn acquire(&self) {
            let message = self
                .handle
                .exchange(
                    self.source,
                    Instant::now(),
                    Message::AcquireControl(acquire(self.boot_id, 100)),
                )
                .await
                .expect("acquire response");
            let Message::AcquireResult(result) = message else {
                panic!("wrong acquire response")
            };
            assert_eq!(result.result, AcquireResultCode::Granted);
            assert_eq!(result.control_epoch, Some(epoch()));
        }

        fn exchange_command(
            &self,
            command: HostCommand,
        ) -> JoinHandle<Result<Message, ActuationHandleError>> {
            let handle = self.handle.clone();
            let source = self.source;
            let first_received_at = Instant::now();
            tokio::spawn(async move {
                handle
                    .exchange(source, first_received_at, Message::HostCommand(command))
                    .await
            })
        }

        fn abort(self) {
            self.actor.abort();
        }
    }

    async fn receive_apply(controller: &mut FakeController) -> ApplyPwm {
        match controller.receive().await {
            Message::ApplyPwm(value) => value,
            other => panic!("expected ApplyPwm, got {other:?}"),
        }
    }

    async fn command_result(
        exchange: JoinHandle<Result<Message, ActuationHandleError>>,
    ) -> HostCommandResult {
        let message = timeout(IO_TIMEOUT, exchange)
            .await
            .expect("host result timeout")
            .expect("host exchange task")
            .expect("actor response");
        let Message::HostCommandResult(result) = message else {
            panic!("wrong host-command response")
        };
        result
    }

    async fn harness_with_cached_zero() -> Harness {
        let mut harness = Harness::ready().await;
        harness.acquire().await;
        let zero = command(harness.boot_id, 0, TimerPwm::ZERO);
        let exchange = harness.exchange_command(zero);
        let apply = receive_apply(&mut harness.controller).await;
        let mut stopped = applied(apply);
        stopped.result = AppliedResultCode::Stopped;
        stopped.output_state = OutputState::Disabled;
        harness
            .controller
            .send(Message::AppliedResult(stopped))
            .await;
        assert_eq!(
            command_result(exchange).await.result,
            HostCommandResultCode::Stopped
        );
        harness
    }

    #[cfg(feature = "qualification-fault-injection")]
    #[test]
    fn qualification_partial_prefix_is_non_delimiter_and_strictly_partial() {
        let message = Message::ApplyPwm(ApplyPwm {
            controller_uid: uid(),
            boot_id: boot(7),
            control_epoch: epoch(),
            sequence: V2CommandSequence::new(1),
            expires_at: ControllerDeadlineMsWrapping::new(1_100),
            timer_pwm: TimerPwm::try_new(1, -1).expect("nonzero candidate PWM"),
        });
        let record = UartRecord::encode(message).expect("ApplyPwm UART record");
        let prefix = QualificationPartialUartRecordPrefix::from_encoded_record(&record)
            .expect("COBS record has a non-delimiter prefix");
        assert_eq!(prefix.as_bytes().len(), 1);
        assert_ne!(prefix.as_bytes()[0], UART_RECORD_DELIMITER[0]);
        assert!(prefix.as_bytes().len() < record.as_bytes().len());
    }

    #[cfg(feature = "qualification-fault-injection")]
    #[test]
    fn qualification_partial_record_wraps_every_uncertain_prefix_transport_outcome() {
        let logical_record_bytes = 37;
        let cases = [
            (
                "write",
                SerialTransmitError::Write {
                    source: io::Error::new(io::ErrorKind::BrokenPipe, "injected write failure"),
                    written_bytes: 0,
                    record_bytes: 1,
                },
            ),
            (
                "flush",
                SerialTransmitError::Flush {
                    source: io::Error::new(io::ErrorKind::BrokenPipe, "injected flush failure"),
                    record_bytes: 1,
                },
            ),
            (
                "deadline",
                SerialTransmitError::Interrupted {
                    phase: SerialTransmitPhase::Write,
                    cause: SerialTransmitInterruption::DeadlineExceeded,
                    written_bytes: 0,
                    record_bytes: 1,
                    maximum_duration: Duration::from_millis(10),
                },
            ),
            (
                "shutdown",
                SerialTransmitError::Interrupted {
                    phase: SerialTransmitPhase::Flush,
                    cause: SerialTransmitInterruption::ShutdownRequested,
                    written_bytes: 1,
                    record_bytes: 1,
                    maximum_duration: Duration::from_millis(10),
                },
            ),
            (
                "priority",
                SerialTransmitError::Interrupted {
                    phase: SerialTransmitPhase::Write,
                    cause: SerialTransmitInterruption::PriorityStopRequested,
                    written_bytes: 0,
                    record_bytes: 1,
                    maximum_duration: Duration::from_millis(10),
                },
            ),
        ];

        for (expected, source) in cases {
            let wrapped =
                qualification_partial_uart_record_error(logical_record_bytes, Err(source));
            assert!(
                wrapped.left_partial_record(),
                "{expected} must retain possible logical-record corruption"
            );
            let SerialTransmitError::QualificationPartialRecord {
                prefix_bytes_may_have_reached_transport,
                logical_record_bytes: observed_logical_record_bytes,
                prefix_outcome: QualificationPartialUartPrefixTransmitOutcome::Uncertain(source),
            } = wrapped
            else {
                panic!("{expected} was not wrapped as qualification uncertainty");
            };
            assert_eq!(prefix_bytes_may_have_reached_transport, 1);
            assert_eq!(observed_logical_record_bytes, logical_record_bytes);
            match expected {
                "write" => assert!(matches!(*source, SerialTransmitError::Write { .. })),
                "flush" => assert!(matches!(*source, SerialTransmitError::Flush { .. })),
                "deadline" => assert!(matches!(
                    *source,
                    SerialTransmitError::Interrupted {
                        cause: SerialTransmitInterruption::DeadlineExceeded,
                        ..
                    }
                )),
                "shutdown" => assert!(matches!(
                    *source,
                    SerialTransmitError::Interrupted {
                        cause: SerialTransmitInterruption::ShutdownRequested,
                        ..
                    }
                )),
                "priority" => assert!(matches!(
                    *source,
                    SerialTransmitError::Interrupted {
                        cause: SerialTransmitInterruption::PriorityStopRequested,
                        ..
                    }
                )),
                _ => unreachable!("closed test table"),
            }
        }
    }

    #[cfg(feature = "qualification-fault-injection")]
    #[derive(Clone, Copy)]
    enum QualificationPrefixExternalInterruption {
        None,
        Shutdown,
        PriorityStop,
    }

    #[cfg(feature = "qualification-fault-injection")]
    async fn exercise_qualification_partial_record_fault(
        mut harness: Harness,
        transport_fault: Option<&QualificationFaultControl>,
        external_interruption: QualificationPrefixExternalInterruption,
    ) -> (
        SerialTransmitError,
        Box<QualificationPartialUartRecordInjectionRecovery>,
    ) {
        harness.acquire().await;

        let zero = command(harness.boot_id, 0, TimerPwm::ZERO);
        let zero_exchange = harness.exchange_command(zero);
        let zero_apply = receive_apply(&mut harness.controller).await;
        harness
            .controller
            .send(Message::AppliedResult(applied(zero_apply)))
            .await;
        assert_eq!(
            command_result(zero_exchange).await.result,
            HostCommandResultCode::AppliedNew,
            "the one-shot declaration must not corrupt acquisition zero"
        );

        if let Some(transport_fault) = transport_fault {
            transport_fault.arm();
        }
        sleep(Duration::from_millis(11)).await;
        let requested = command(
            harness.boot_id,
            1,
            TimerPwm::try_new(1, -1).expect("nonzero candidate PWM"),
        );
        let exchange = harness.exchange_command(requested);
        if let Some(transport_fault) = transport_fault {
            transport_fault.wait_until_triggered().await;
        }
        match external_interruption {
            QualificationPrefixExternalInterruption::None => {}
            QualificationPrefixExternalInterruption::Shutdown => {
                harness
                    .shutdown
                    .request(ActuationShutdownReason::SiblingFailure);
            }
            QualificationPrefixExternalInterruption::PriorityStop => {
                let _priority_response = harness.handle.enqueue_for_test(
                    harness.source,
                    Instant::now(),
                    Message::HostStop(HostStop {
                        controller_uid: uid(),
                        target_boot_id: TargetBootId::Exact(harness.boot_id),
                        request_id: RequestId::new(0x7171),
                        reason: ForceStopReason::Operator,
                    }),
                );
            }
        }
        assert!(
            harness.controller.receive_result().await.is_err(),
            "the explicit delimiter must expose the possibly forwarded prefix as malformed"
        );
        let Message::ForceStop(force_stop) = harness.controller.receive().await else {
            panic!("partial-record recovery must issue ForceStop")
        };
        let mut stop = confirmed_stop(force_stop, harness.boot_id);
        stop.faults = ControllerFaults::try_from_bits(ControllerFaults::SERIAL_INTEGRITY)
            .expect("known serial-integrity bit");
        harness.controller.send(Message::HostStopResult(stop)).await;
        assert_eq!(
            command_result(exchange).await.result,
            HostCommandResultCode::ForceStopped
        );

        let Harness { actor, .. } = harness;
        let error = timeout(IO_TIMEOUT, actor)
            .await
            .expect("actor terminal evidence timeout")
            .expect("actor task join")
            .expect_err("injected fault is a terminal qualification outcome");
        let ActuationActorError::QualificationPartialUartRecordInjected {
            interrupted,
            recovery,
        } = error
        else {
            panic!("wrong qualification terminal evidence: {error}")
        };
        assert!(matches!(
            recovery.resynchronization(),
            SerialResynchronizationOutcome::DelimiterTransmitted
        ));
        assert!(matches!(
            recovery.force_stop(),
            ShutdownForceStopOutcome::Confirmed(result)
                if result.output_state.is_safe()
                    && result.faults.bits() == ControllerFaults::SERIAL_INTEGRITY
        ));
        (interrupted, recovery)
    }

    #[cfg(feature = "qualification-fault-injection")]
    #[tokio::test]
    async fn qualification_partial_record_fault_redelimits_and_force_stops_once() {
        let mut config = actor_config();
        config.qualification_serial_fault = Some(
            OperatorSupervisedCandidateSerialFaultInjection::PartialUartRecordOnFirstNonzeroCommand,
        );
        let harness = Harness::ready_with_actor_config(4_096, config).await;
        let (interrupted, _) = exercise_qualification_partial_record_fault(
            harness,
            None,
            QualificationPrefixExternalInterruption::None,
        )
        .await;
        assert!(matches!(
            interrupted,
            SerialTransmitError::QualificationPartialRecord {
                prefix_bytes_may_have_reached_transport: 1,
                logical_record_bytes,
                prefix_outcome: QualificationPartialUartPrefixTransmitOutcome::Transmitted,
            } if logical_record_bytes > 1
        ));
    }

    #[cfg(feature = "qualification-fault-injection")]
    #[tokio::test]
    async fn qualification_prefix_write_and_flush_errors_still_redelimit_and_force_stop() {
        for (fault, expected) in [
            (QualificationPrefixTransportFault::Write, "write"),
            (QualificationPrefixTransportFault::Flush, "flush"),
        ] {
            let (harness, control) = Harness::ready_with_qualification_transport_fault(fault).await;
            let (interrupted, _) = exercise_qualification_partial_record_fault(
                harness,
                Some(&control),
                QualificationPrefixExternalInterruption::None,
            )
            .await;
            let SerialTransmitError::QualificationPartialRecord {
                prefix_bytes_may_have_reached_transport: 1,
                logical_record_bytes,
                prefix_outcome: QualificationPartialUartPrefixTransmitOutcome::Uncertain(source),
            } = interrupted
            else {
                panic!("{expected} failure lost qualification uncertainty");
            };
            assert!(logical_record_bytes > 1);
            match expected {
                "write" => assert!(matches!(*source, SerialTransmitError::Write { .. })),
                "flush" => assert!(matches!(*source, SerialTransmitError::Flush { .. })),
                _ => unreachable!("closed transport-fault table"),
            }
        }
    }

    #[cfg(feature = "qualification-fault-injection")]
    #[tokio::test]
    async fn qualification_prefix_interruptions_still_redelimit_and_force_stop() {
        for (fault, external_interruption, expected_cause) in [
            (
                QualificationPrefixTransportFault::Deadline,
                QualificationPrefixExternalInterruption::None,
                SerialTransmitInterruption::DeadlineExceeded,
            ),
            (
                QualificationPrefixTransportFault::Shutdown,
                QualificationPrefixExternalInterruption::Shutdown,
                SerialTransmitInterruption::ShutdownRequested,
            ),
            (
                QualificationPrefixTransportFault::PriorityStop,
                QualificationPrefixExternalInterruption::PriorityStop,
                SerialTransmitInterruption::PriorityStopRequested,
            ),
        ] {
            let (harness, control) = Harness::ready_with_qualification_transport_fault(fault).await;
            let (interrupted, _) = exercise_qualification_partial_record_fault(
                harness,
                Some(&control),
                external_interruption,
            )
            .await;
            let SerialTransmitError::QualificationPartialRecord {
                prefix_bytes_may_have_reached_transport: 1,
                logical_record_bytes,
                prefix_outcome: QualificationPartialUartPrefixTransmitOutcome::Uncertain(source),
            } = interrupted
            else {
                panic!("{expected_cause:?} lost qualification uncertainty");
            };
            assert!(logical_record_bytes > 1);
            assert!(matches!(
                *source,
                SerialTransmitError::Interrupted { cause, .. } if cause == expected_cause
            ));
        }
    }

    #[tokio::test]
    async fn fresh_open_excludes_one_unknown_prefix_then_keeps_framing_strict() {
        let boot_id = boot(7);
        let (actor_stream, controller_stream) = tokio::io::duplex(4_096);
        let (_handle, startup_ready, actor) = spawn_actor(
            actor_stream,
            actor_config(),
            Arc::new(NoopActuationTelemetry),
            UartStreamDecoder::new_at_unknown_record_offset(),
        );
        let mut controller = FakeController::new(controller_stream);
        controller
            .stream
            .write_all(&[0x55; MAX_RAW_FRAME_BYTES * 2])
            .await
            .expect("unknown startup prefix");
        controller
            .stream
            .write_all(&[0])
            .await
            .expect("startup alignment delimiter");

        controller
            .send(Message::ControllerHello(hello_with(uid(), boot_id)))
            .await;
        let Message::ForceStop(stop) = controller.receive().await else {
            panic!("aligned Hello must reach the startup stop")
        };
        controller.assert_no_message().await;
        controller
            .send(Message::HostStopResult(confirmed_stop(stop, boot_id)))
            .await;
        assert!(matches!(
            controller.receive().await,
            Message::BeginSession(_)
        ));
        controller
            .send(Message::ControllerReady(ready_at(boot_id, 1_000)))
            .await;
        controller
            .send(Message::Heartbeat(zero_heartbeat(boot_id)))
            .await;
        timeout(IO_TIMEOUT, startup_ready)
            .await
            .expect("aligned startup timeout")
            .expect("aligned startup reaches exact readiness");

        controller
            .send_corrupted(Message::Heartbeat(zero_heartbeat_at(boot_id, 1_002)))
            .await;
        assert!(matches!(controller.receive().await, Message::ForceStop(_)));
        actor.abort();
    }

    #[tokio::test]
    async fn startup_begin_requires_a_confirming_stop_receipt() {
        let boot_id = boot(7);
        let (actor_stream, controller_stream) = tokio::io::duplex(4_096);
        let (_handle, mut startup_ready, actor) = spawn_actor(
            actor_stream,
            actor_config(),
            Arc::new(NoopActuationTelemetry),
            UartStreamDecoder::new(),
        );
        let mut controller = FakeController::new(controller_stream);
        controller
            .send(Message::ControllerHello(hello_with(uid(), boot_id)))
            .await;
        let Message::ForceStop(stop) = controller.receive().await else {
            panic!("exact Hello must start with ForceStop")
        };
        controller.assert_no_message().await;

        let mut nonconfirming = confirmed_stop(stop, boot_id);
        nonconfirming.result = StopResultCode::ControllerUnavailable;
        controller
            .send(Message::HostStopResult(nonconfirming))
            .await;
        controller.assert_no_message().await;
        assert!(
            timeout(SHORT_ABSENCE, &mut startup_ready).await.is_err(),
            "a nonconfirming stop receipt must never reach startup readiness"
        );
        actor.abort();
    }

    #[test]
    fn delayed_controller_time_relation_is_strict_and_wrap_aware() {
        let at = |value| ControllerUptimeMsWrapping::new(value);

        assert!(controller_time_strictly_precedes(at(999), at(1_000)));
        assert!(controller_time_strictly_precedes(at(u32::MAX), at(0)));
        assert!(!controller_time_strictly_precedes(at(1_000), at(1_000)));
        assert!(!controller_time_strictly_precedes(at(1_001), at(1_000)));
        assert!(!controller_time_strictly_precedes(at(0x8000_0000), at(0)));
    }

    #[tokio::test]
    async fn controller_ready_can_overtake_an_older_no_session_heartbeat() {
        let (mut harness, mut startup_ready) =
            Harness::awaiting_startup_heartbeat(4_096, 1_000).await;
        harness
            .controller
            .send(Message::Heartbeat(no_session_heartbeat(
                harness.boot_id,
                999,
            )))
            .await;

        assert!(
            timeout(SHORT_ABSENCE, &mut startup_ready).await.is_err(),
            "a discarded delayed heartbeat must not establish liveness"
        );
        harness.controller.assert_no_message().await;

        harness
            .controller
            .send(Message::Heartbeat(zero_heartbeat_at(
                harness.boot_id,
                1_001,
            )))
            .await;
        timeout(IO_TIMEOUT, startup_ready)
            .await
            .expect("current heartbeat startup timeout")
            .expect("current heartbeat establishes startup");
        harness.controller.assert_no_message().await;
        harness.abort();
    }

    #[tokio::test]
    async fn delayed_startup_heartbeat_wraps_but_half_range_is_never_ignored() {
        let (mut wrapping, startup_ready) = Harness::awaiting_startup_heartbeat(4_096, 0).await;
        wrapping
            .controller
            .send(Message::Heartbeat(no_session_heartbeat(
                wrapping.boot_id,
                u32::MAX,
            )))
            .await;
        wrapping.controller.assert_no_message().await;
        wrapping
            .controller
            .send(Message::Heartbeat(zero_heartbeat_at(wrapping.boot_id, 1)))
            .await;
        timeout(IO_TIMEOUT, startup_ready)
            .await
            .expect("wrapped startup timeout")
            .expect("wrapped current heartbeat establishes startup");
        wrapping.abort();

        let (mut ambiguous, _startup_ready) = Harness::awaiting_startup_heartbeat(4_096, 0).await;
        ambiguous
            .controller
            .send(Message::Heartbeat(no_session_heartbeat(
                ambiguous.boot_id,
                0x8000_0000,
            )))
            .await;
        assert!(matches!(
            ambiguous.controller.receive().await,
            Message::ForceStop(_)
        ));
        ambiguous.abort();
    }

    #[tokio::test]
    async fn wrong_boot_is_never_discarded_as_delayed_startup_telemetry() {
        let (mut harness, _startup_ready) = Harness::awaiting_startup_heartbeat(4_096, 1_000).await;
        harness
            .controller
            .send(Message::Heartbeat(no_session_heartbeat(boot(8), 999)))
            .await;
        assert!(matches!(
            harness.controller.receive().await,
            Message::ForceStop(_)
        ));
        harness.abort();
    }

    #[tokio::test]
    async fn periodic_hello_and_applied_result_can_overtake_older_best_effort_heartbeat() {
        let mut harness = harness_with_cached_zero().await;
        harness
            .controller
            .send(Message::Heartbeat(established_stopped_heartbeat(
                harness.boot_id,
                epoch(),
                V2CommandSequence::FIRST,
                1_001,
            )))
            .await;
        harness.controller.assert_no_message().await;
        sleep(Duration::from_millis(11)).await;

        let pwm = TimerPwm::try_new(10, -10).expect("bounded PWM");
        let moving = command(harness.boot_id, 1, pwm);
        let exchange = harness.exchange_command(moving);
        let apply = receive_apply(&mut harness.controller).await;
        let applied_result = applied(apply);
        harness
            .controller
            .send(Message::AppliedResult(applied_result))
            .await;
        assert_eq!(
            command_result(exchange).await.result,
            HostCommandResultCode::AppliedNew
        );
        harness
            .controller
            .send(Message::ControllerHello(hello_with(uid(), harness.boot_id)))
            .await;
        harness.controller.assert_no_message().await;

        let delayed_uptime = applied_result.applied_at.get().wrapping_sub(1);
        harness
            .controller
            .send(Message::Heartbeat(established_stopped_heartbeat(
                harness.boot_id,
                epoch(),
                V2CommandSequence::FIRST,
                delayed_uptime,
            )))
            .await;
        harness.controller.assert_no_message().await;

        let current_uptime = applied_result.applied_at.get().wrapping_add(1);
        harness
            .controller
            .send(Message::Heartbeat(Heartbeat {
                controller_uid: uid(),
                boot_id: harness.boot_id,
                control_epoch: Some(epoch()),
                last_sequence: Some(V2CommandSequence::new(1)),
                controller_uptime: ControllerUptimeMsWrapping::new(current_uptime),
                expires_at: applied_result.expires_at,
                timer_pwm: pwm,
                output_state: OutputState::NonzeroPwm,
                readiness: readiness(),
                faults: ControllerFaults::NONE,
            }))
            .await;
        harness.controller.assert_no_message().await;
        harness.abort();
    }

    #[tokio::test]
    async fn periodic_same_boot_hello_preserves_a_stopped_owner_and_sequence() {
        let mut harness = harness_with_cached_zero().await;
        harness
            .controller
            .send(Message::ControllerHello(hello_with(uid(), harness.boot_id)))
            .await;
        harness.controller.assert_no_message().await;
        harness
            .controller
            .send(Message::Heartbeat(established_stopped_heartbeat(
                harness.boot_id,
                epoch(),
                V2CommandSequence::FIRST,
                1_001,
            )))
            .await;
        harness.controller.assert_no_message().await;

        sleep(Duration::from_millis(11)).await;
        let next = command(
            harness.boot_id,
            1,
            TimerPwm::try_new(10, -10).expect("bounded PWM"),
        );
        let exchange = harness.exchange_command(next);
        let apply = receive_apply(&mut harness.controller).await;
        assert_eq!(apply.sequence, V2CommandSequence::new(1));
        exchange.abort();
        harness.abort();
    }

    #[tokio::test]
    async fn periodic_same_boot_hello_does_not_cancel_a_pending_command() {
        let mut harness = harness_with_cached_zero().await;
        harness
            .controller
            .send(Message::Heartbeat(established_stopped_heartbeat(
                harness.boot_id,
                epoch(),
                V2CommandSequence::FIRST,
                1_001,
            )))
            .await;
        harness.controller.assert_no_message().await;
        sleep(Duration::from_millis(11)).await;

        let moving = command(
            harness.boot_id,
            1,
            TimerPwm::try_new(10, -10).expect("bounded PWM"),
        );
        let mut exchange = harness.exchange_command(moving);
        let apply = receive_apply(&mut harness.controller).await;
        harness
            .controller
            .send(Message::ControllerHello(hello_with(uid(), harness.boot_id)))
            .await;
        assert!(timeout(SHORT_ABSENCE, &mut exchange).await.is_err());
        harness.controller.assert_no_message().await;

        harness
            .controller
            .send(Message::AppliedResult(applied(apply)))
            .await;
        assert_eq!(
            command_result(exchange).await.result,
            HostCommandResultCode::AppliedNew
        );
        harness.abort();
    }

    #[tokio::test]
    async fn periodic_same_boot_hello_preserves_active_authority() {
        let mut harness = harness_with_cached_zero().await;
        harness
            .controller
            .send(Message::Heartbeat(established_stopped_heartbeat(
                harness.boot_id,
                epoch(),
                V2CommandSequence::FIRST,
                1_001,
            )))
            .await;
        harness.controller.assert_no_message().await;
        sleep(Duration::from_millis(11)).await;

        let pwm = TimerPwm::try_new(10, -10).expect("bounded PWM");
        let moving = command(harness.boot_id, 1, pwm);
        let exchange = harness.exchange_command(moving);
        let apply = receive_apply(&mut harness.controller).await;
        let applied_result = applied(apply);
        harness
            .controller
            .send(Message::AppliedResult(applied_result))
            .await;
        assert_eq!(
            command_result(exchange).await.result,
            HostCommandResultCode::AppliedNew
        );

        harness
            .controller
            .send(Message::ControllerHello(hello_with(uid(), harness.boot_id)))
            .await;
        harness.controller.assert_no_message().await;
        harness
            .controller
            .send(Message::Heartbeat(Heartbeat {
                controller_uid: uid(),
                boot_id: harness.boot_id,
                control_epoch: Some(epoch()),
                last_sequence: Some(V2CommandSequence::new(1)),
                controller_uptime: ControllerUptimeMsWrapping::new(
                    applied_result.applied_at.get().wrapping_add(1),
                ),
                expires_at: applied_result.expires_at,
                timer_pwm: pwm,
                output_state: OutputState::NonzeroPwm,
                readiness: readiness(),
                faults: ControllerFaults::NONE,
            }))
            .await;
        harness.controller.assert_no_message().await;

        sleep(Duration::from_millis(11)).await;
        let next = command(harness.boot_id, 2, pwm);
        let exchange = harness.exchange_command(next);
        let apply = receive_apply(&mut harness.controller).await;
        assert_eq!(apply.sequence, V2CommandSequence::new(2));
        exchange.abort();
        harness.abort();
    }

    #[tokio::test]
    async fn changed_claims_on_the_same_boot_are_not_an_idempotent_hello() {
        let mut harness = harness_with_cached_zero().await;
        let mut conflicting = hello_with(uid(), harness.boot_id);
        conflicting.firmware_build_id = conflicting.firmware_build_id.wrapping_add(1);
        harness
            .controller
            .send(Message::ControllerHello(conflicting))
            .await;
        assert!(matches!(
            harness.controller.receive().await,
            Message::ForceStop(ForceStop {
                reason: ForceStopReason::ControllerFault,
                ..
            })
        ));
        harness.abort();
    }

    #[tokio::test]
    async fn exact_established_stopped_heartbeat_preserves_the_zero_session() {
        let mut harness = harness_with_cached_zero().await;
        harness
            .controller
            .send(Message::Heartbeat(established_stopped_heartbeat(
                harness.boot_id,
                epoch(),
                V2CommandSequence::FIRST,
                1_001,
            )))
            .await;
        harness.controller.assert_no_message().await;

        sleep(Duration::from_millis(11)).await;
        let next = command(
            harness.boot_id,
            1,
            TimerPwm::try_new(10, -10).expect("bounded PWM"),
        );
        let exchange = harness.exchange_command(next);
        let apply = receive_apply(&mut harness.controller).await;
        assert_eq!(apply.sequence, V2CommandSequence::new(1));
        assert_eq!(apply.timer_pwm, next.requested_timer_pwm);
        exchange.abort();
        harness.abort();
    }

    #[tokio::test]
    async fn stopped_heartbeat_with_wrong_epoch_is_force_stopped() {
        let mut harness = harness_with_cached_zero().await;
        let wrong_epoch = ControlEpoch::try_new(epoch().get().wrapping_add(1))
            .expect("fixture epoch remains nonzero");
        harness
            .controller
            .send(Message::Heartbeat(established_stopped_heartbeat(
                harness.boot_id,
                wrong_epoch,
                V2CommandSequence::FIRST,
                1_001,
            )))
            .await;
        assert!(matches!(
            harness.controller.receive().await,
            Message::ForceStop(_)
        ));
        harness.abort();
    }

    #[tokio::test]
    async fn stopped_heartbeat_with_wrong_sequence_is_force_stopped() {
        let mut harness = harness_with_cached_zero().await;
        harness
            .controller
            .send(Message::Heartbeat(established_stopped_heartbeat(
                harness.boot_id,
                epoch(),
                V2CommandSequence::new(1),
                1_001,
            )))
            .await;
        assert!(matches!(
            harness.controller.receive().await,
            Message::ForceStop(_)
        ));
        harness.abort();
    }

    #[tokio::test]
    async fn stopped_heartbeat_from_wrong_boot_session_is_force_stopped() {
        let mut harness = harness_with_cached_zero().await;
        harness
            .controller
            .send(Message::Heartbeat(established_stopped_heartbeat(
                boot(8),
                epoch(),
                V2CommandSequence::FIRST,
                1_001,
            )))
            .await;
        assert!(matches!(
            harness.controller.receive().await,
            Message::ForceStop(_)
        ));
        harness.abort();
    }

    #[tokio::test]
    async fn active_heartbeat_cannot_claim_stopped_readiness() {
        let mut harness = harness_with_cached_zero().await;
        harness
            .controller
            .send(Message::Heartbeat(established_stopped_heartbeat(
                harness.boot_id,
                epoch(),
                V2CommandSequence::FIRST,
                1_001,
            )))
            .await;
        harness.controller.assert_no_message().await;
        sleep(Duration::from_millis(11)).await;
        let pwm = TimerPwm::try_new(10, -10).expect("bounded PWM");
        let moving = command(harness.boot_id, 1, pwm);
        let exchange = harness.exchange_command(moving);
        let apply = receive_apply(&mut harness.controller).await;
        harness
            .controller
            .send(Message::AppliedResult(applied(apply)))
            .await;
        assert_eq!(
            command_result(exchange).await.result,
            HostCommandResultCode::AppliedNew
        );

        let heartbeat = Heartbeat {
            controller_uid: uid(),
            boot_id: harness.boot_id,
            control_epoch: Some(epoch()),
            last_sequence: Some(V2CommandSequence::new(1)),
            controller_uptime: ControllerUptimeMsWrapping::new(
                apply.expires_at.get().wrapping_sub(50),
            ),
            expires_at: apply.expires_at,
            timer_pwm: pwm,
            output_state: OutputState::NonzeroPwm,
            readiness: stopped_readiness(),
            faults: ControllerFaults::NONE,
        };
        harness.controller.send(Message::Heartbeat(heartbeat)).await;
        assert!(matches!(
            harness.controller.receive().await,
            Message::ForceStop(_)
        ));
        harness.abort();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn priority_shutdown_preempts_an_already_queued_nonzero_command() {
        let mut harness = Harness::ready().await;
        harness.acquire().await;
        let requested_pwm = TimerPwm::try_new(20, -20).expect("bounded motion PWM");
        let response = harness.handle.enqueue_for_test(
            harness.source,
            Instant::now(),
            Message::HostCommand(command(harness.boot_id, 0, requested_pwm)),
        );

        // Both the queued request and shutdown are ready before the
        // current-thread executor can poll the actor. The biased owner signal
        // must win, so no ApplyPwm can precede this ForceStop.
        harness.shutdown.request(ActuationShutdownReason::Operator);
        let Message::ForceStop(stop) = harness.controller.receive().await else {
            panic!("priority shutdown must emit ForceStop before queued motion")
        };
        assert_eq!(stop.reason, ForceStopReason::Operator);
        harness
            .controller
            .send(Message::HostStopResult(confirmed_stop(
                stop,
                harness.boot_id,
            )))
            .await;
        assert!(
            response.await.is_err(),
            "the queued command must not receive an application result"
        );
        assert!(
            harness.actor.await.expect("actor task join").is_ok(),
            "priority shutdown is a clean actor exit after best-effort stop"
        );
    }

    #[tokio::test]
    async fn completed_shutdown_write_without_matching_stop_result_remains_uncertain() {
        let mut harness = Harness::ready().await;
        harness.shutdown.request(ActuationShutdownReason::Operator);
        let Message::ForceStop(_) = harness.controller.receive().await else {
            panic!("shutdown emits a bounded ForceStop")
        };
        let result = timeout(IO_TIMEOUT, harness.actor)
            .await
            .expect("actor reaches the stop-confirmation deadline")
            .expect("actor task joins");
        assert!(matches!(
            result,
            Err(ActuationActorError::ShutdownStopConfirmationTimedOut {
                maximum_wait
            }) if maximum_wait == Duration::from_millis(25)
        ));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn shutdown_during_command_write_resynchronizes_then_reports_exact_stop() {
        let mut harness = Harness::ready_with_serial_capacity(1).await;
        harness.acquire().await;
        let response = harness.exchange_command(command(
            harness.boot_id,
            0,
            TimerPwm::try_new(5, -5).expect("bounded first step"),
        ));

        // Consume exactly one byte of the ApplyPwm record, leaving the
        // single-byte serial buffer full again before shutdown is requested.
        // This deterministically interrupts a genuinely partial actor write.
        let first = harness
            .controller
            .stream
            .read_u8()
            .await
            .expect("first partial ApplyPwm byte");
        assert!(
            harness.controller.decoder.push(first).is_none(),
            "one COBS code byte cannot complete an ApplyPwm record"
        );
        harness.shutdown.request(ActuationShutdownReason::Operator);

        let mut framing_fault = None;
        let stop = timeout(IO_TIMEOUT, async {
            loop {
                let byte = harness
                    .controller
                    .stream
                    .read_u8()
                    .await
                    .expect("recovery serial byte");
                let Some(decoded) = harness.controller.decoder.push(byte) else {
                    continue;
                };
                match decoded {
                    Err(source) => framing_fault = Some(source),
                    Ok(Message::ForceStop(stop)) => break stop,
                    Ok(other) => panic!(
                        "partial ApplyPwm must not become a valid message before ForceStop, got {:?}",
                        other.kind()
                    ),
                }
            }
        })
        .await
        .expect("delimiter and ForceStop remain bounded");
        assert!(
            framing_fault.is_some(),
            "the recovery delimiter must expose the interrupted record as malformed"
        );
        harness
            .controller
            .send(Message::HostStopResult(confirmed_stop(
                stop,
                harness.boot_id,
            )))
            .await;

        let message = response
            .await
            .expect("host exchange task")
            .expect("pending command receives a typed terminal result");
        assert!(matches!(
            message,
            Message::HostCommandResult(HostCommandResult {
                result: HostCommandResultCode::ForceStopped,
                ..
            })
        ));

        let result = harness.actor.await.expect("actor task joins");
        let Err(ActuationActorError::ShutdownInterruptedTransmit {
            interrupted:
                SerialTransmitError::Interrupted {
                    phase: SerialTransmitPhase::Write,
                    cause: SerialTransmitInterruption::ShutdownRequested,
                    written_bytes,
                    record_bytes,
                    ..
                },
            recovery,
        }) = result
        else {
            panic!("actor must retain the interrupted transmit and recovery report")
        };
        assert!(written_bytes > 0 && written_bytes < record_bytes);
        assert!(matches!(
            recovery.as_ref(),
            ShutdownInterruptedTransmitRecovery {
                resynchronization: SerialResynchronizationOutcome::DelimiterTransmitted,
                force_stop: ShutdownForceStopOutcome::Confirmed(_),
            }
        ));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn priority_host_stop_preempts_a_saturated_ordinary_mailbox() {
        let mut harness = Harness::ready().await;
        harness.acquire().await;
        let command_response = harness.handle.enqueue_for_test(
            harness.source,
            Instant::now(),
            Message::HostCommand(command(harness.boot_id, 0, TimerPwm::ZERO)),
        );
        let mut queued_status = Vec::new();
        for request_id in 0..(ACTOR_MAILBOX_CAPACITY - 1) {
            queued_status.push(harness.handle.enqueue_for_test(
                harness.source,
                Instant::now(),
                Message::StatusQuery(StatusQuery {
                    expected_controller_uid: uid(),
                    request_id: RequestId::new(
                        u32::try_from(request_id).expect("mailbox index fits request ID"),
                    ),
                }),
            ));
        }
        let host_stop = HostStop {
            controller_uid: uid(),
            target_boot_id: TargetBootId::Exact(harness.boot_id),
            request_id: RequestId::new(0x5a5a),
            reason: ForceStopReason::Operator,
        };
        let stop_response = harness.handle.enqueue_for_test(
            harness.source,
            Instant::now(),
            Message::HostStop(host_stop),
        );

        let Message::ForceStop(force_stop) = harness.controller.receive().await else {
            panic!("latched HostStop must precede every saturated ordinary request")
        };
        assert_eq!(force_stop.reason, ForceStopReason::Operator);
        harness
            .controller
            .send(Message::HostStopResult(confirmed_stop(
                force_stop,
                harness.boot_id,
            )))
            .await;

        let Message::HostStopResult(result) = timeout(IO_TIMEOUT, stop_response)
            .await
            .expect("priority-stop response timeout")
            .expect("priority-stop response sender")
        else {
            panic!("wrong priority-stop response")
        };
        assert_eq!(result.request_id, host_stop.request_id);
        assert_eq!(result.result, StopResultCode::ControllerConfirmed);
        let Message::HostCommandResult(command_result) = timeout(IO_TIMEOUT, command_response)
            .await
            .expect("queued command terminal response timeout")
            .expect("queued command terminal response sender")
        else {
            panic!("wrong queued command response")
        };
        assert!(!command_result.result.proves_controller_application());
        harness.controller.assert_no_message().await;
        drop(queued_status);
        harness.abort();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn udp_ingress_latches_host_stop_while_all_ordinary_response_slots_are_occupied() {
        let (requests, requests_rx) = mpsc::channel(MAX_UDP_ORDINARY_EXCHANGES_IN_FLIGHT);
        let priority_stop = Arc::new(PriorityStopCoordinator::new(uid()));
        let handle = ActuationHandle {
            requests,
            priority_stop: Arc::clone(&priority_stop),
            shutdown: Arc::new(ActuationShutdownSignal::new()),
        };
        let server_socket = UdpSocket::bind("127.0.0.1:0")
            .await
            .expect("bind UDP server");
        let server_address = server_socket.local_addr().expect("server address");
        let server = tokio::spawn(udp_service_on_socket(server_socket, Some(handle)));
        let client = UdpSocket::bind("127.0.0.1:0")
            .await
            .expect("bind UDP client");

        for request_id in 0..MAX_UDP_ORDINARY_EXCHANGES_IN_FLIGHT {
            let request_id = u32::try_from(request_id).expect("bounded request index fits u32");
            let frame = RawFrame::encode(Message::StatusQuery(StatusQuery {
                expected_controller_uid: uid(),
                request_id: RequestId::new(request_id),
            }))
            .expect("status request encodes");
            client
                .send_to(frame.as_bytes(), server_address)
                .await
                .expect("send ordinary request");
        }
        timeout(Duration::from_secs(1), async {
            while requests_rx.len() != MAX_UDP_ORDINARY_EXCHANGES_IN_FLIGHT {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("every ordinary response slot becomes occupied");

        let stop = HostStop {
            controller_uid: uid(),
            target_boot_id: TargetBootId::Any,
            request_id: RequestId::new(0x5a5b),
            reason: ForceStopReason::Operator,
        };
        let frame = RawFrame::encode(Message::HostStop(stop)).expect("stop request encodes");
        client
            .send_to(frame.as_bytes(), server_address)
            .await
            .expect("send priority stop");
        timeout(IO_TIMEOUT, async {
            while priority_stop.requested_after(0).is_none() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("UDP receive latches HostStop without an ordinary response slot");

        server.abort();
        let _ = server.await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn completed_udp_tasks_release_capacity_before_joinset_reaping() {
        let (requests, mut requests_rx) = mpsc::channel(MAX_UDP_ORDINARY_EXCHANGES_IN_FLIGHT + 1);
        let priority_stop = Arc::new(PriorityStopCoordinator::new(uid()));
        let handle = ActuationHandle {
            requests,
            priority_stop,
            shutdown: Arc::new(ActuationShutdownSignal::new()),
        };
        let server_socket = UdpSocket::bind("127.0.0.1:0")
            .await
            .expect("bind UDP server");
        let server_address = server_socket.local_addr().expect("server address");
        let server = tokio::spawn(udp_service_on_socket(server_socket, Some(handle)));
        let client = UdpSocket::bind("127.0.0.1:0")
            .await
            .expect("bind UDP client");

        for request_id in 0..MAX_UDP_ORDINARY_EXCHANGES_IN_FLIGHT {
            let request_id = u32::try_from(request_id).expect("bounded request index fits u32");
            let frame = RawFrame::encode(Message::StatusQuery(StatusQuery {
                expected_controller_uid: uid(),
                request_id: RequestId::new(request_id),
            }))
            .expect("status request encodes");
            client
                .send_to(frame.as_bytes(), server_address)
                .await
                .expect("send ordinary request");
        }

        let mut pending = Vec::with_capacity(MAX_UDP_ORDINARY_EXCHANGES_IN_FLIGHT);
        for _ in 0..MAX_UDP_ORDINARY_EXCHANGES_IN_FLIGHT {
            pending.push(
                timeout(IO_TIMEOUT, requests_rx.recv())
                    .await
                    .expect("actor request timeout")
                    .expect("actor request channel remains open"),
            );
        }
        for request in pending {
            let response = unavailable_response(request.request);
            request
                .response
                .send(response)
                .expect("test receiver remains alive");
        }

        let mut receive_buffer = [0_u8; MAX_RAW_FRAME_BYTES];
        for _ in 0..MAX_UDP_ORDINARY_EXCHANGES_IN_FLIGHT {
            timeout(IO_TIMEOUT, client.recv_from(&mut receive_buffer))
                .await
                .expect("UDP response timeout")
                .expect("receive UDP response");
        }
        tokio::task::yield_now().await;

        let final_query = StatusQuery {
            expected_controller_uid: uid(),
            request_id: RequestId::new(0xf00d),
        };
        let frame =
            RawFrame::encode(Message::StatusQuery(final_query)).expect("final query encodes");
        client
            .send_to(frame.as_bytes(), server_address)
            .await
            .expect("send final ordinary request");
        let final_request = timeout(IO_TIMEOUT, requests_rx.recv())
            .await
            .expect("released response capacity admits the next request")
            .expect("actor request channel remains open");
        assert_eq!(final_request.request, HostRequest::Status(final_query));
        final_request
            .response
            .send(unavailable_response(final_request.request))
            .expect("final test receiver remains alive");

        server.abort();
        let _ = server.await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn concurrent_priority_stops_share_one_transaction_and_keep_request_identity() {
        let mut harness = Harness::ready().await;
        let first = HostStop {
            controller_uid: uid(),
            target_boot_id: TargetBootId::Any,
            request_id: RequestId::new(0x1111),
            reason: ForceStopReason::Operator,
        };
        let second = HostStop {
            request_id: RequestId::new(0x2222),
            ..first
        };
        let first_response = harness.handle.enqueue_for_test(
            harness.source,
            Instant::now(),
            Message::HostStop(first),
        );
        let second_response = harness.handle.enqueue_for_test(
            harness.source,
            Instant::now(),
            Message::HostStop(second),
        );

        let Message::ForceStop(force_stop) = harness.controller.receive().await else {
            panic!("coalesced priority stop must emit one ForceStop")
        };
        harness
            .controller
            .send(Message::HostStopResult(confirmed_stop(
                force_stop,
                harness.boot_id,
            )))
            .await;

        for (response, expected_request_id) in [
            (first_response, first.request_id),
            (second_response, second.request_id),
        ] {
            let Message::HostStopResult(result) = timeout(IO_TIMEOUT, response)
                .await
                .expect("coalesced stop response timeout")
                .expect("coalesced stop response sender")
            else {
                panic!("wrong coalesced stop response")
            };
            assert_eq!(result.request_id, expected_request_id);
            assert_eq!(result.result, StopResultCode::ControllerConfirmed);
        }
        harness.controller.assert_no_message().await;
        harness.abort();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn sequential_duplicate_priority_stop_reuses_the_single_exact_cache_entry() {
        let mut harness = Harness::ready().await;
        let request = HostStop {
            controller_uid: uid(),
            target_boot_id: TargetBootId::Exact(harness.boot_id),
            request_id: RequestId::new(0x2a2a),
            reason: ForceStopReason::Operator,
        };
        let first = harness.handle.enqueue_for_test(
            harness.source,
            Instant::now(),
            Message::HostStop(request),
        );
        let Message::ForceStop(force_stop) = harness.controller.receive().await else {
            panic!("first stop must reach the controller")
        };
        harness
            .controller
            .send(Message::HostStopResult(confirmed_stop(
                force_stop,
                harness.boot_id,
            )))
            .await;
        let first = timeout(IO_TIMEOUT, first)
            .await
            .expect("first stop response timeout")
            .expect("first stop response sender");

        let duplicate = harness.handle.enqueue_for_test(
            harness.source,
            Instant::now(),
            Message::HostStop(request),
        );
        let duplicate = timeout(IO_TIMEOUT, duplicate)
            .await
            .expect("cached duplicate response timeout")
            .expect("cached duplicate response sender");
        assert_eq!(duplicate, first);
        harness.controller.assert_no_message().await;
        harness.abort();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn internal_stop_attempt_invalidates_a_prior_priority_stop_receipt() {
        let mut harness = Harness::ready().await;
        let request = HostStop {
            controller_uid: uid(),
            target_boot_id: TargetBootId::Exact(harness.boot_id),
            request_id: RequestId::new(0x2a2d),
            reason: ForceStopReason::Operator,
        };
        let first = harness.handle.enqueue_for_test(
            harness.source,
            Instant::now(),
            Message::HostStop(request),
        );
        let Message::ForceStop(first_force_stop) = harness.controller.receive().await else {
            panic!("first stop must reach the controller")
        };
        harness
            .controller
            .send(Message::HostStopResult(confirmed_stop(
                first_force_stop,
                harness.boot_id,
            )))
            .await;
        timeout(IO_TIMEOUT, first)
            .await
            .expect("first stop response timeout")
            .expect("first stop response sender");

        let nonzero_pwm = TimerPwm::try_new(1, -1).expect("bounded nonzero PWM");
        harness
            .controller
            .send(Message::Heartbeat(Heartbeat {
                timer_pwm: nonzero_pwm,
                output_state: OutputState::NonzeroPwm,
                ..zero_heartbeat_at(harness.boot_id, 1_100)
            }))
            .await;
        let Message::ForceStop(internal_stop) = harness.controller.receive().await else {
            panic!("post-stop nonzero evidence must cause an internal ForceStop")
        };

        let duplicate = harness.handle.enqueue_for_test(
            harness.source,
            Instant::now(),
            Message::HostStop(request),
        );
        let Message::ForceStop(new_priority_stop) = harness.controller.receive().await else {
            panic!("stale receipt must not answer a later duplicate")
        };
        assert_ne!(new_priority_stop.request_id, internal_stop.request_id);
        assert_ne!(new_priority_stop.request_id, first_force_stop.request_id);
        harness
            .controller
            .send(Message::HostStopResult(confirmed_stop(
                new_priority_stop,
                harness.boot_id,
            )))
            .await;
        let Message::HostStopResult(result) = timeout(IO_TIMEOUT, duplicate)
            .await
            .expect("fresh duplicate stop response timeout")
            .expect("fresh duplicate stop response sender")
        else {
            panic!("wrong fresh duplicate stop response")
        };
        assert_eq!(result.request_id, request.request_id);
        assert_eq!(result.result, StopResultCode::ControllerConfirmed);
        harness.abort();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn a_distinct_priority_stop_invalidates_the_previous_duplicate_cache() {
        let mut harness = Harness::ready().await;
        let first_request = HostStop {
            controller_uid: uid(),
            target_boot_id: TargetBootId::Exact(harness.boot_id),
            request_id: RequestId::new(0x2a2b),
            reason: ForceStopReason::Operator,
        };
        let first = harness.handle.enqueue_for_test(
            harness.source,
            Instant::now(),
            Message::HostStop(first_request),
        );
        let Message::ForceStop(first_force_stop) = harness.controller.receive().await else {
            panic!("first stop must reach the controller")
        };
        harness
            .controller
            .send(Message::HostStopResult(confirmed_stop(
                first_force_stop,
                harness.boot_id,
            )))
            .await;
        timeout(IO_TIMEOUT, first)
            .await
            .expect("first stop response timeout")
            .expect("first stop response sender");

        let second_request = HostStop {
            request_id: RequestId::new(0x2a2c),
            ..first_request
        };
        let mut second = harness.handle.enqueue_for_test(
            harness.source,
            Instant::now(),
            Message::HostStop(second_request),
        );
        let Message::ForceStop(second_force_stop) = harness.controller.receive().await else {
            panic!("distinct stop must begin a new controller transaction")
        };

        let mut old_duplicate = harness.handle.enqueue_for_test(
            harness.source,
            Instant::now(),
            Message::HostStop(first_request),
        );
        assert!(
            timeout(Duration::from_millis(5), &mut old_duplicate)
                .await
                .is_err(),
            "the old exact cache entry must be invalid as soon as a distinct stop is pending"
        );

        harness
            .controller
            .send(Message::HostStopResult(confirmed_stop(
                second_force_stop,
                harness.boot_id,
            )))
            .await;
        let second = timeout(IO_TIMEOUT, &mut second)
            .await
            .expect("second stop response timeout")
            .expect("second stop response sender");
        let old_duplicate = timeout(IO_TIMEOUT, &mut old_duplicate)
            .await
            .expect("coalesced old-identity response timeout")
            .expect("coalesced old-identity response sender");
        let Message::HostStopResult(second) = second else {
            panic!("wrong second response kind")
        };
        let Message::HostStopResult(old_duplicate) = old_duplicate else {
            panic!("wrong duplicate response kind")
        };
        assert_eq!(second.request_id, second_request.request_id);
        assert_eq!(old_duplicate.request_id, first_request.request_id);
        assert_eq!(second.result, StopResultCode::ControllerConfirmed);
        assert_eq!(
            old_duplicate.result,
            StopResultCode::ControllerConfirmed,
            "the in-flight controller evidence is projected onto each caller identity"
        );
        harness.controller.assert_no_message().await;
        harness.abort();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn exact_non_stop_proving_result_preserves_controller_state_without_claiming_stop() {
        let mut harness = Harness::ready().await;
        let request = HostStop {
            controller_uid: uid(),
            target_boot_id: TargetBootId::Exact(harness.boot_id),
            request_id: RequestId::new(0x2b2b),
            reason: ForceStopReason::Operator,
        };
        let response = harness.handle.enqueue_for_test(
            harness.source,
            Instant::now(),
            Message::HostStop(request),
        );
        let Message::ForceStop(force_stop) = harness.controller.receive().await else {
            panic!("priority stop must reach the controller")
        };
        let observed_faults = ControllerFaults::try_from_bits(ControllerFaults::MOTOR_DRIVER)
            .expect("known motor-driver fault");
        harness
            .controller
            .send(Message::HostStopResult(HostStopResult {
                controller_uid: force_stop.controller_uid,
                observed_boot_id: TargetBootId::Exact(harness.boot_id),
                request_id: force_stop.request_id,
                result: StopResultCode::ControllerFaulted,
                output_state: OutputState::NonzeroPwm,
                controller_uptime: ControllerUptimeMsWrapping::new(1_234),
                faults: observed_faults,
            }))
            .await;

        let Message::HostStopResult(result) = timeout(IO_TIMEOUT, response)
            .await
            .expect("exact non-stop response timeout")
            .expect("exact non-stop response sender")
        else {
            panic!("wrong exact non-stop response")
        };
        assert_eq!(result.request_id, request.request_id);
        assert_eq!(result.result, StopResultCode::ControllerFaulted);
        assert_eq!(result.output_state, OutputState::NonzeroPwm);
        assert_eq!(result.controller_uptime.get(), 1_234);
        assert_eq!(result.faults, observed_faults);
        assert!(!result.result.proves_controller_stop());
        assert_eq!(
            result.output_evidence(),
            robot_protocol::v2::OutputEvidence::Unknown
        );
        harness.abort();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn priority_stop_resynchronizes_a_partial_command_then_continues_with_exact_evidence() {
        let mut harness = Harness::ready_with_serial_capacity(1).await;
        harness.acquire().await;
        let command_response =
            harness.exchange_command(command(harness.boot_id, 0, TimerPwm::ZERO));
        let first_byte = harness
            .controller
            .stream
            .read_u8()
            .await
            .expect("first partial ApplyPwm byte");
        assert!(harness.controller.decoder.push(first_byte).is_none());

        let host_stop = HostStop {
            controller_uid: uid(),
            target_boot_id: TargetBootId::Exact(harness.boot_id),
            request_id: RequestId::new(0x3333),
            reason: ForceStopReason::Operator,
        };
        let stop_response = harness.handle.enqueue_for_test(
            harness.source,
            Instant::now(),
            Message::HostStop(host_stop),
        );

        let mut framing_fault = None;
        let force_stop = timeout(IO_TIMEOUT, async {
            loop {
                let byte = harness
                    .controller
                    .stream
                    .read_u8()
                    .await
                    .expect("priority recovery serial byte");
                let Some(decoded) = harness.controller.decoder.push(byte) else {
                    continue;
                };
                match decoded {
                    Err(source) => framing_fault = Some(source),
                    Ok(Message::ForceStop(force_stop)) => break force_stop,
                    Ok(other) => panic!(
                        "partial ApplyPwm must not precede priority ForceStop as {:?}",
                        other.kind()
                    ),
                }
            }
        })
        .await
        .expect("priority delimiter and ForceStop remain bounded");
        assert!(
            framing_fault.is_some(),
            "interrupted partial record is explicitly delimited as malformed"
        );
        harness
            .controller
            .send(Message::HostStopResult(confirmed_stop(
                force_stop,
                harness.boot_id,
            )))
            .await;

        let command_result = command_result(command_response).await;
        assert_eq!(command_result.result, HostCommandResultCode::ForceStopped);
        let Message::HostStopResult(stop_result) = timeout(IO_TIMEOUT, stop_response)
            .await
            .expect("priority-stop exact response timeout")
            .expect("priority-stop exact response sender")
        else {
            panic!("wrong priority-stop exact response")
        };
        assert_eq!(stop_result.request_id, host_stop.request_id);
        assert_eq!(stop_result.result, StopResultCode::ControllerConfirmed);

        let Message::StatusReport(status) = harness
            .handle
            .exchange(
                harness.source,
                Instant::now(),
                Message::StatusQuery(StatusQuery {
                    expected_controller_uid: uid(),
                    request_id: RequestId::new(0x4444),
                }),
            )
            .await
            .expect("actor remains responsive after priority-stop recovery")
        else {
            panic!("wrong post-recovery status response")
        };
        assert_ne!(status.status, StatusCode::ReadyActive);
        harness.abort();
    }

    #[tokio::test]
    async fn exact_applied_result_is_required_and_cached_duplicate_never_reapplies_or_renews() {
        let mut harness = Harness::ready().await;
        harness.acquire().await;
        let command = command(harness.boot_id, 0, TimerPwm::ZERO);
        let mut exchange = harness.exchange_command(command);
        let apply = receive_apply(&mut harness.controller).await;
        assert!(timeout(SHORT_ABSENCE, &mut exchange).await.is_err());

        harness
            .controller
            .send(Message::AppliedResult(applied(apply)))
            .await;
        let first = command_result(exchange).await;
        assert_eq!(first.result, HostCommandResultCode::AppliedNew);
        assert!(first.remaining_lease.get() > 0);

        sleep(Duration::from_millis(2)).await;
        let duplicate = command_result(harness.exchange_command(command)).await;
        assert_eq!(duplicate.result, HostCommandResultCode::DuplicateCached);
        assert!(duplicate.remaining_lease.get() <= first.remaining_lease.get());
        harness.controller.assert_no_message().await;
        harness.abort();
    }

    #[tokio::test]
    async fn declared_command_rate_is_enforced_without_delaying_stop_or_consuming_sequence() {
        let mut harness = Harness::ready().await;
        harness.acquire().await;
        let zero = command(harness.boot_id, 0, TimerPwm::ZERO);
        let zero_exchange = harness.exchange_command(zero);
        let zero_apply = receive_apply(&mut harness.controller).await;
        harness
            .controller
            .send(Message::AppliedResult(applied(zero_apply)))
            .await;
        assert_eq!(
            command_result(zero_exchange).await.result,
            HostCommandResultCode::AppliedNew
        );

        let next = command(
            harness.boot_id,
            1,
            TimerPwm::try_new(10, 10).expect("bounded PWM"),
        );
        assert_eq!(
            command_result(harness.exchange_command(next)).await.result,
            HostCommandResultCode::RejectedAtServer
        );
        harness.controller.assert_no_message().await;

        sleep(Duration::from_millis(11)).await;
        let retry = harness.exchange_command(next);
        let apply = receive_apply(&mut harness.controller).await;
        assert_eq!(apply.sequence, V2CommandSequence::new(1));
        harness
            .controller
            .send(Message::AppliedResult(applied(apply)))
            .await;
        assert_eq!(
            command_result(retry).await.result,
            HostCommandResultCode::AppliedNew
        );

        let stop_request = HostStop {
            controller_uid: uid(),
            target_boot_id: TargetBootId::Exact(harness.boot_id),
            request_id: RequestId::new(0xabcdef),
            reason: ForceStopReason::Operator,
        };
        let stop_exchange = tokio::spawn({
            let handle = harness.handle.clone();
            let source = harness.source;
            async move {
                handle
                    .exchange(source, Instant::now(), Message::HostStop(stop_request))
                    .await
            }
        });
        let Message::ForceStop(stop) = harness.controller.receive().await else {
            panic!("stop bypasses command-rate admission")
        };
        harness
            .controller
            .send(Message::HostStopResult(confirmed_stop(
                stop,
                harness.boot_id,
            )))
            .await;
        let Message::HostStopResult(stop_result) = stop_exchange
            .await
            .expect("stop exchange joins")
            .expect("stop response")
        else {
            panic!("wrong stop response")
        };
        assert_eq!(stop_result.result, StopResultCode::ControllerConfirmed);
        harness.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn concrete_client_completes_loopback_acquire_motion_and_confirmed_disarm() {
        let mut harness = Harness::ready().await;
        let socket = UdpSocket::bind("127.0.0.1:0")
            .await
            .expect("bind loopback test socket");
        let endpoint = socket.local_addr().expect("bound loopback address");
        let udp = tokio::spawn(udp_service_on_socket(socket, Some(harness.handle.clone())));
        let config = client_config(endpoint);

        let acquire = tokio::task::spawn_blocking(move || {
            let transport = UdpV2Transport::connect_canonical(config.endpoint())
                .expect("connect concrete loopback client");
            DisarmedCommandClient::new(transport, SystemMonotonicClock::new(), config)
                .acquire_zero()
        });
        let zero_apply = receive_apply(&mut harness.controller).await;
        assert!(zero_apply.is_initial_zero_acquisition());
        harness
            .controller
            .send(Message::AppliedResult(applied(zero_apply)))
            .await;
        let acquired = timeout(Duration::from_secs(1), acquire)
            .await
            .expect("concrete acquire timeout")
            .expect("concrete acquire task");
        let (armed, zero_receipt) = match acquired {
            Ok(value) => value,
            Err(failure) => panic!("concrete acquire failed: {}", failure.cause()),
        };
        assert_eq!(zero_receipt.sequence(), V2CommandSequence::FIRST);
        assert!(zero_receipt.is_confirmed_zero());

        sleep(Duration::from_millis(11)).await;
        let requested_pwm = TimerPwm::try_new(20, -20).expect("bounded motion PWM");
        let pending = PendingPhysicalCommand::new(
            requested_pwm,
            V2CommandLeaseMs::try_new(120).expect("motion lease"),
            MonotonicInstant::from_nanos_since_clock_start(5_000_000_000),
        );
        let motion = tokio::task::spawn_blocking(move || armed.apply(pending));
        let motion_apply = receive_apply(&mut harness.controller).await;
        assert_eq!(motion_apply.sequence, V2CommandSequence::new(1));
        assert_eq!(motion_apply.timer_pwm, requested_pwm);
        harness
            .controller
            .send(Message::AppliedResult(applied(motion_apply)))
            .await;
        let motion_result = timeout(Duration::from_secs(1), motion)
            .await
            .expect("concrete motion timeout")
            .expect("concrete motion task");
        let (armed, motion_receipt) = match motion_result {
            Ok(value) => value,
            Err(failure) => panic!("concrete motion failed: {}", failure.cause()),
        };
        assert_eq!(motion_receipt.sequence(), V2CommandSequence::new(1));
        assert_eq!(motion_receipt.applied_timer_pwm(), requested_pwm);

        let disarm = tokio::task::spawn_blocking(move || armed.disarm());
        let stop = match harness.controller.receive().await {
            Message::ForceStop(value) => value,
            other => panic!("expected ForceStop, got {:?}", other.kind()),
        };
        harness
            .controller
            .send(Message::HostStopResult(confirmed_stop(
                stop,
                harness.boot_id,
            )))
            .await;
        let disarm_result = timeout(Duration::from_secs(1), disarm)
            .await
            .expect("concrete disarm timeout")
            .expect("concrete disarm task");
        let (_disarmed, stop_receipt) = match disarm_result {
            Ok(value) => value,
            Err(failure) => panic!("concrete disarm failed: {}", failure.cause()),
        };
        assert_eq!(stop_receipt.controller_uid(), uid());
        assert_eq!(stop_receipt.observed_boot_id(), harness.boot_id);
        assert!(stop_receipt.output_state().is_safe());

        udp.abort();
        harness.abort();
    }

    #[tokio::test]
    async fn applied_timestamp_before_deadline_reference_never_proves_application() {
        let mut harness = Harness::ready().await;
        harness.acquire().await;
        let command = command(harness.boot_id, 0, TimerPwm::ZERO);
        let exchange = harness.exchange_command(command);
        let apply = receive_apply(&mut harness.controller).await;
        let mut impossible = applied(apply);
        impossible.applied_at = ControllerUptimeMsWrapping::new(999);

        harness
            .controller
            .send(Message::AppliedResult(impossible))
            .await;

        assert_eq!(
            command_result(exchange).await.result,
            HostCommandResultCode::RejectedByController
        );
        assert!(matches!(
            harness.controller.receive().await,
            Message::ForceStop(_)
        ));
        harness.abort();
    }

    #[tokio::test]
    async fn fresh_zero_result_cannot_be_claimed_as_a_cached_duplicate() {
        let mut harness = Harness::ready().await;
        harness.acquire().await;
        let command = command(harness.boot_id, 0, TimerPwm::ZERO);
        let exchange = harness.exchange_command(command);
        let apply = receive_apply(&mut harness.controller).await;
        let mut contradictory = applied(apply);
        contradictory.result = AppliedResultCode::DuplicateCached;

        harness
            .controller
            .send(Message::AppliedResult(contradictory))
            .await;

        assert_eq!(
            command_result(exchange).await.result,
            HostCommandResultCode::RejectedByController
        );
        assert!(matches!(
            harness.controller.receive().await,
            Message::ForceStop(ForceStop {
                reason: ForceStopReason::ControllerFault,
                ..
            })
        ));
        harness.abort();
    }

    #[tokio::test]
    async fn nonzero_result_requires_fresh_nonzero_application_shape() {
        let mut harness = harness_with_cached_zero().await;
        harness
            .controller
            .send(Message::Heartbeat(established_stopped_heartbeat(
                harness.boot_id,
                epoch(),
                V2CommandSequence::FIRST,
                1_001,
            )))
            .await;
        harness.controller.assert_no_message().await;
        sleep(Duration::from_millis(11)).await;

        let requested = TimerPwm::try_new(10, -10).expect("bounded PWM");
        let command = command(harness.boot_id, 1, requested);
        let exchange = harness.exchange_command(command);
        let apply = receive_apply(&mut harness.controller).await;
        let mut contradictory = applied(apply);
        contradictory.result = AppliedResultCode::Stopped;

        harness
            .controller
            .send(Message::AppliedResult(contradictory))
            .await;

        assert_eq!(
            command_result(exchange).await.result,
            HostCommandResultCode::RejectedByController
        );
        assert!(matches!(
            harness.controller.receive().await,
            Message::ForceStop(ForceStop {
                reason: ForceStopReason::ControllerFault,
                ..
            })
        ));
        harness.abort();
    }

    #[tokio::test]
    async fn dropped_delayed_or_lost_applied_ack_never_becomes_host_success() {
        let mut harness = Harness::ready().await;
        harness.acquire().await;
        let command = command(harness.boot_id, 0, TimerPwm::ZERO);
        let exchange = harness.exchange_command(command);
        let _apply = receive_apply(&mut harness.controller).await;

        let result = command_result(exchange).await;
        assert_eq!(result.result, HostCommandResultCode::AppliedAckTimeout);
        assert_eq!(result.remaining_lease, RemainingLeaseMs::ZERO);
        assert!(matches!(
            harness.controller.receive().await,
            Message::ForceStop(_)
        ));
        harness.abort();
    }

    #[tokio::test]
    async fn continuously_readable_observational_uart_cannot_starve_host_status_or_ack_timer() {
        let mut harness = Harness::ready().await;
        harness.acquire().await;
        for sample in 1..=24 {
            harness
                .controller
                .send(Message::ObservationalOdometry(odometry(
                    harness.boot_id,
                    sample,
                )))
                .await;
        }

        let query = StatusQuery {
            expected_controller_uid: uid(),
            request_id: RequestId::new(0x55aa),
        };
        let status = timeout(
            Duration::from_millis(25),
            harness
                .handle
                .exchange(harness.source, Instant::now(), Message::StatusQuery(query)),
        )
        .await
        .expect("bounded alternating turns service the host mailbox")
        .expect("actor remains live");
        assert!(matches!(status, Message::StatusReport(_)));

        let command = command(harness.boot_id, 0, TimerPwm::ZERO);
        let exchange = harness.exchange_command(command);
        let _apply = receive_apply(&mut harness.controller).await;
        for sample in 25..=48 {
            harness
                .controller
                .send(Message::ObservationalOdometry(odometry(
                    harness.boot_id,
                    sample,
                )))
                .await;
        }
        let result = timeout(Duration::from_millis(75), command_result(exchange))
            .await
            .expect("serial readability cannot starve the applied-ACK deadline");
        assert_eq!(result.result, HostCommandResultCode::AppliedAckTimeout);
        assert!(matches!(
            harness.controller.receive().await,
            Message::ForceStop(_)
        ));
        harness.abort();
    }

    #[tokio::test]
    async fn serial_eof_fails_the_waiter_without_any_application_claim() {
        let mut harness = Harness::ready().await;
        harness.acquire().await;
        let command = command(harness.boot_id, 0, TimerPwm::ZERO);
        let exchange = harness.exchange_command(command);
        let _apply = receive_apply(&mut harness.controller).await;

        drop(harness.controller);
        assert_eq!(
            command_result(exchange).await.result,
            HostCommandResultCode::ForceStopped
        );
        let actor_result = timeout(IO_TIMEOUT, harness.actor)
            .await
            .expect("actor shutdown timeout")
            .expect("actor task join");
        assert!(matches!(actor_result, Err(ActuationActorError::SerialEof)));
    }

    #[tokio::test]
    async fn reordered_old_ack_force_stops_instead_of_confirming_the_new_command() {
        let mut harness = Harness::ready().await;
        harness.acquire().await;
        let zero = command(harness.boot_id, 0, TimerPwm::ZERO);
        let zero_exchange = harness.exchange_command(zero);
        let zero_apply = receive_apply(&mut harness.controller).await;
        let old_result = applied(zero_apply);
        harness
            .controller
            .send(Message::AppliedResult(old_result))
            .await;
        assert_eq!(
            command_result(zero_exchange).await.result,
            HostCommandResultCode::AppliedNew
        );

        sleep(Duration::from_millis(11)).await;
        let moving = command(
            harness.boot_id,
            1,
            TimerPwm::try_new(20, -20).expect("bounded PWM"),
        );
        let moving_exchange = harness.exchange_command(moving);
        let _moving_apply = receive_apply(&mut harness.controller).await;
        harness
            .controller
            .send(Message::AppliedResult(old_result))
            .await;
        assert_eq!(
            command_result(moving_exchange).await.result,
            HostCommandResultCode::ForceStopped
        );
        assert!(matches!(
            harness.controller.receive().await,
            Message::ForceStop(_)
        ));
        harness.abort();
    }

    #[tokio::test]
    async fn sequence_zero_must_be_zero_pwm_before_any_motion_can_be_forwarded() {
        let mut harness = Harness::ready().await;
        harness.acquire().await;
        let invalid_acquisition = command(
            harness.boot_id,
            0,
            TimerPwm::try_new(1, 1).expect("bounded PWM"),
        );
        let exchange = harness.exchange_command(invalid_acquisition);

        assert_eq!(
            command_result(exchange).await.result,
            HostCommandResultCode::ForceStopped
        );
        assert!(matches!(
            harness.controller.receive().await,
            Message::ForceStop(_)
        ));
        harness.controller.assert_no_message().await;
        harness.abort();
    }

    #[tokio::test]
    async fn nonzero_heartbeat_at_its_controller_deadline_is_a_fault() {
        let mut harness = Harness::ready().await;
        harness.acquire().await;

        let zero = command(harness.boot_id, 0, TimerPwm::ZERO);
        let zero_exchange = harness.exchange_command(zero);
        let zero_apply = receive_apply(&mut harness.controller).await;
        harness
            .controller
            .send(Message::AppliedResult(applied(zero_apply)))
            .await;
        assert_eq!(
            command_result(zero_exchange).await.result,
            HostCommandResultCode::AppliedNew
        );

        sleep(Duration::from_millis(11)).await;
        let pwm = TimerPwm::try_new(20, -20).expect("bounded PWM");
        let moving = command(harness.boot_id, 1, pwm);
        let moving_exchange = harness.exchange_command(moving);
        let moving_apply = receive_apply(&mut harness.controller).await;
        harness
            .controller
            .send(Message::AppliedResult(applied(moving_apply)))
            .await;
        assert_eq!(
            command_result(moving_exchange).await.result,
            HostCommandResultCode::AppliedNew
        );

        let expired_uptime = ControllerUptimeMsWrapping::new(moving_apply.expires_at.get());
        let expired_nonzero = Heartbeat {
            controller_uid: uid(),
            boot_id: harness.boot_id,
            control_epoch: Some(epoch()),
            last_sequence: Some(V2CommandSequence::new(1)),
            controller_uptime: expired_uptime,
            expires_at: moving_apply.expires_at,
            timer_pwm: pwm,
            output_state: OutputState::NonzeroPwm,
            readiness: readiness(),
            faults: ControllerFaults::NONE,
        };
        harness
            .controller
            .send(Message::Heartbeat(expired_nonzero))
            .await;
        assert!(matches!(
            harness.controller.receive().await,
            Message::ForceStop(_)
        ));
        harness.abort();
    }

    #[tokio::test]
    async fn corrupted_controller_record_revokes_and_fails_the_waiter() {
        let mut harness = Harness::ready().await;
        harness.acquire().await;
        let command = command(harness.boot_id, 0, TimerPwm::ZERO);
        let exchange = harness.exchange_command(command);
        let apply = receive_apply(&mut harness.controller).await;
        harness
            .controller
            .send_corrupted(Message::AppliedResult(applied(apply)))
            .await;

        assert_eq!(
            command_result(exchange).await.result,
            HostCommandResultCode::ForceStopped
        );
        assert!(matches!(
            harness.controller.receive().await,
            Message::ForceStop(_)
        ));
        harness.abort();
    }

    #[tokio::test]
    async fn controller_reboot_during_command_cannot_confirm_the_old_epoch() {
        let mut harness = Harness::ready().await;
        harness.acquire().await;
        let command = command(harness.boot_id, 0, TimerPwm::ZERO);
        let exchange = harness.exchange_command(command);
        let _apply = receive_apply(&mut harness.controller).await;
        let rebooted = boot(8);
        harness
            .controller
            .send(Message::ControllerHello(hello_with(uid(), rebooted)))
            .await;

        assert_eq!(
            command_result(exchange).await.result,
            HostCommandResultCode::ControllerRestarted
        );
        let stop = harness.controller.receive().await;
        let Message::ForceStop(stop) = stop else {
            panic!("reboot must start with an exact ForceStop")
        };
        harness.controller.assert_no_message().await;
        harness
            .controller
            .send(Message::HostStopResult(confirmed_stop(stop, rebooted)))
            .await;
        assert!(matches!(
            harness.controller.receive().await,
            Message::BeginSession(_)
        ));
        harness.abort();
    }

    #[tokio::test]
    async fn wrong_uid_never_reaches_begin_session() {
        let (actor_stream, controller_stream) = tokio::io::duplex(4_096);
        let (handle, _startup_ready, actor) = spawn_actor(
            actor_stream,
            actor_config(),
            Arc::new(NoopActuationTelemetry),
            UartStreamDecoder::new(),
        );
        let mut controller = FakeController::new(controller_stream);
        controller
            .send(Message::ControllerHello(hello_with(wrong_uid(), boot(9))))
            .await;
        let Message::ForceStop(stop) = controller.receive().await else {
            panic!("wrong-UID hello must be force-stopped")
        };
        assert_eq!(stop.controller_uid, wrong_uid());
        controller.assert_no_message().await;

        let source = "127.0.0.1:41001".parse().expect("source address");
        let query = StatusQuery {
            expected_controller_uid: uid(),
            request_id: RequestId::new(12),
        };
        let Message::StatusReport(report) = handle
            .exchange(source, Instant::now(), Message::StatusQuery(query))
            .await
            .expect("status result")
        else {
            panic!("wrong status result")
        };
        assert_eq!(report.status, StatusCode::Faulted);
        actor.abort();
    }

    #[test]
    fn hello_gate_requires_every_manifest_and_safety_claim() {
        let (actor_stream, _controller_stream) = tokio::io::duplex(256);
        let (startup_ready, _startup_ready_rx) = oneshot::channel();
        let actor = SerialActor::new(
            actor_stream,
            UartStreamDecoder::new(),
            actor_config(),
            Arc::new(NoopActuationTelemetry),
            startup_ready,
        );
        let baseline = hello_with(uid(), boot(7));
        assert!(actor.hello_is_exact(baseline));

        let mutations: [fn(&mut ControllerHello); 10] = [
            |hello| hello.controller_uid = wrong_uid(),
            |hello| hello.firmware_abi += 1,
            |hello| hello.firmware_build_id += 1,
            |hello| {
                hello.actuator_config_fingerprint =
                    ActuatorConfigFingerprint::try_new([9; 16]).expect("alternate fingerprint")
            },
            |hello| {
                hello.max_abs_pwm_percent =
                    MaxAbsPwmPercent::try_new(49).expect("alternate maximum")
            },
            |hello| {
                hello.pwm_frequency = PwmFrequencyHz::try_new(19_999).expect("alternate frequency")
            },
            |hello| {
                hello.watchdog_nominal_period =
                    WatchdogNominalPeriodMs::try_new(249).expect("alternate watchdog")
            },
            |hello| hello.neutral_output = NeutralOutput::HighImpedance,
            |hello| hello.physical_stop_semantics = PhysicalStopSemantics::BrakeVerified,
            |hello| {
                hello.capabilities = ControllerCapabilities::try_from_bits(
                    ControllerCapabilities::REQUIRED_BITS
                        & !ControllerCapabilities::INDEPENDENT_WATCHDOG,
                )
                .expect("known incomplete capabilities")
            },
        ];
        for mutate in mutations {
            let mut candidate = baseline;
            mutate(&mut candidate);
            assert!(!actor.hello_is_exact(candidate));
        }
    }

    #[test]
    fn telemetry_snapshot_exposes_output_only_from_a_fresh_heartbeat() {
        let (actor_stream, _controller_stream) = tokio::io::duplex(256);
        let (startup_ready, _startup_ready_rx) = oneshot::channel();
        let mut actor = SerialActor::new(
            actor_stream,
            UartStreamDecoder::new(),
            actor_config(),
            Arc::new(NoopActuationTelemetry),
            startup_ready,
        );
        let now = Instant::now();
        assert_eq!(actor.snapshot(now).output, ActuationOutputEvidence::Unknown);

        let heartbeat = zero_heartbeat(boot(7));
        actor.heartbeat = Some(TimedHeartbeat {
            message: heartbeat,
            received_at: now,
        });
        assert_eq!(
            actor.snapshot(now).output,
            ActuationOutputEvidence::Observed(ObservedActuationOutput {
                controller_uptime: heartbeat.controller_uptime,
                output_state: heartbeat.output_state,
                controller_timer_pwm: heartbeat.timer_pwm,
                faults: heartbeat.faults,
            })
        );

        let stale_at = now + actor.config.maximum_heartbeat_age;
        assert_eq!(
            actor.snapshot(stale_at).output,
            ActuationOutputEvidence::Unknown
        );
    }

    #[tokio::test]
    async fn rejected_startup_claim_is_retained_as_typed_snapshot_evidence() {
        let (actor_stream, _controller_stream) = tokio::io::duplex(256);
        let (startup_ready, _startup_ready_rx) = oneshot::channel();
        let mut actor = SerialActor::new(
            actor_stream,
            UartStreamDecoder::new(),
            actor_config(),
            Arc::new(NoopActuationTelemetry),
            startup_ready,
        );
        let mut rejected = hello_with(uid(), boot(7));
        rejected.firmware_build_id = rejected.firmware_build_id.wrapping_add(1);

        actor
            .handle_hello(rejected, Instant::now())
            .await
            .expect("fault stop remains serializable");

        let snapshot = actor.snapshot(Instant::now());
        assert_eq!(snapshot.startup_phase, ActuationStartupPhase::Faulted);
        assert_eq!(
            snapshot.fault,
            Some(ActuationFaultEvidence::ControllerHelloRejected(rejected))
        );
        assert_eq!(
            snapshot.observed_boot_id,
            TargetBootId::Exact(rejected.boot_id)
        );
    }

    #[test]
    fn hello_gate_keeps_candidate_and_production_classes_disjoint() {
        let (candidate_stream, _candidate_peer) = tokio::io::duplex(256);
        let (candidate_ready, _candidate_ready_rx) = oneshot::channel();
        let candidate_actor = SerialActor::new(
            candidate_stream,
            UartStreamDecoder::new(),
            candidate_actor_config(),
            Arc::new(NoopActuationTelemetry),
            candidate_ready,
        );
        let candidate = candidate_hello();
        assert!(candidate_actor.hello_is_exact(candidate));
        assert!(!candidate_actor.hello_is_exact(hello_with(uid(), boot(7))));

        let (production_stream, _production_peer) = tokio::io::duplex(256);
        let (production_ready, _production_ready_rx) = oneshot::channel();
        let production_actor = SerialActor::new(
            production_stream,
            UartStreamDecoder::new(),
            actor_config(),
            Arc::new(NoopActuationTelemetry),
            production_ready,
        );
        assert!(!production_actor.hello_is_exact(candidate));

        for mutate in [
            |hello: &mut ControllerHello| {
                hello.capabilities = capabilities();
            },
            |hello: &mut ControllerHello| {
                hello.physical_stop_semantics = PhysicalStopSemantics::CoastVerified;
            },
            |hello: &mut ControllerHello| {
                hello.max_abs_pwm_percent =
                    MaxAbsPwmPercent::try_new(29).expect("alternate candidate cap");
            },
        ] {
            let mut altered = candidate;
            mutate(&mut altered);
            assert!(!candidate_actor.hello_is_exact(altered));
        }
    }

    #[test]
    fn udp_emission_charges_only_time_after_the_actor_remaining_lifetime_calculation() {
        let first_received_at = Instant::now();
        let request = command(boot(7), 0, TimerPwm::ZERO);
        let response = HostCommandResult {
            controller_uid: request.controller_uid,
            boot_id: request.boot_id,
            control_epoch: request.control_epoch,
            sequence: request.sequence,
            result: HostCommandResultCode::AppliedNew,
            requested_timer_pwm: TimerPwm::ZERO,
            controller_timer_pwm: TimerPwm::ZERO,
            output_state: OutputState::ZeroPwm,
            controller_applied_at: ControllerUptimeMsWrapping::new(1_000),
            controller_expires_at: ControllerDeadlineMsWrapping::new(1_120),
            remaining_lease: RemainingLeaseMs::try_new(100).expect("remaining lease"),
            faults: ControllerFaults::NONE,
        };
        let Message::HostCommandResult(emitted) = response_for_udp_emission(
            HostRequest::Command(request),
            first_received_at,
            Message::HostCommandResult(response),
            first_received_at + Duration::from_millis(10),
            first_received_at + Duration::from_millis(14),
        ) else {
            panic!("wrong response kind")
        };
        assert_eq!(emitted.remaining_lease.get(), 95);
    }

    #[tokio::test]
    async fn duplicate_controller_uptime_cannot_refresh_bootstrap_heartbeat_age() {
        let mut harness = Harness::ready().await;
        harness
            .controller
            .send(Message::Heartbeat(zero_heartbeat(harness.boot_id)))
            .await;
        assert!(matches!(
            harness.controller.receive().await,
            Message::ForceStop(_)
        ));
        harness.abort();
    }

    #[tokio::test]
    async fn stale_heartbeat_force_stops_and_blocks_acquisition() {
        let mut harness = Harness::ready().await;
        sleep(Duration::from_millis(85)).await;
        assert!(matches!(
            harness.controller.receive().await,
            Message::ForceStop(_)
        ));
        let Message::AcquireResult(result) = harness
            .handle
            .exchange(
                harness.source,
                Instant::now(),
                Message::AcquireControl(acquire(harness.boot_id, 44)),
            )
            .await
            .expect("acquire terminal result")
        else {
            panic!("wrong acquire result")
        };
        assert_eq!(result.result, AcquireResultCode::Faulted);
        assert_eq!(result.control_epoch, None);
        harness.abort();
    }

    #[tokio::test]
    async fn uid_targeted_stop_works_without_session_and_requires_matching_stop_result() {
        let (actor_stream, controller_stream) = tokio::io::duplex(4_096);
        let (handle, _startup_ready, actor) = spawn_actor(
            actor_stream,
            actor_config(),
            Arc::new(NoopActuationTelemetry),
            UartStreamDecoder::new(),
        );
        let mut controller = FakeController::new(controller_stream);
        let source = "127.0.0.1:41002".parse().expect("source address");
        let request = HostStop {
            controller_uid: uid(),
            target_boot_id: TargetBootId::Any,
            request_id: RequestId::new(90),
            reason: ForceStopReason::Operator,
        };
        let mut exchange = tokio::spawn({
            let handle = handle.clone();
            async move {
                handle
                    .exchange(source, Instant::now(), Message::HostStop(request))
                    .await
            }
        });
        let Message::ForceStop(stop) = controller.receive().await else {
            panic!("host stop was not forwarded")
        };
        assert_eq!(stop.controller_uid, uid());
        assert_ne!(stop.request_id, request.request_id);
        assert!(timeout(SHORT_ABSENCE, &mut exchange).await.is_err());
        controller
            .send(Message::HostStopResult(confirmed_stop(stop, boot(55))))
            .await;
        let message = timeout(IO_TIMEOUT, exchange)
            .await
            .expect("stop result timeout")
            .expect("stop task")
            .expect("stop response");
        let Message::HostStopResult(result) = message else {
            panic!("wrong stop response")
        };
        assert_eq!(result.result, StopResultCode::ControllerConfirmed);
        assert_eq!(result.request_id, request.request_id);
        actor.abort();
    }

    #[tokio::test]
    async fn mismatched_stop_result_is_never_promoted_to_controller_confirmation() {
        let (actor_stream, controller_stream) = tokio::io::duplex(4_096);
        let (handle, _startup_ready, actor) = spawn_actor(
            actor_stream,
            actor_config(),
            Arc::new(NoopActuationTelemetry),
            UartStreamDecoder::new(),
        );
        let mut controller = FakeController::new(controller_stream);
        let source = "127.0.0.1:41003".parse().expect("source address");
        let request = HostStop {
            controller_uid: uid(),
            target_boot_id: TargetBootId::Any,
            request_id: RequestId::new(91),
            reason: ForceStopReason::Operator,
        };
        let exchange = tokio::spawn({
            let handle = handle.clone();
            async move {
                handle
                    .exchange(source, Instant::now(), Message::HostStop(request))
                    .await
            }
        });
        let Message::ForceStop(stop) = controller.receive().await else {
            panic!("host stop was not forwarded")
        };
        let mut mismatched = confirmed_stop(stop, boot(55));
        mismatched.request_id = RequestId::new(stop.request_id.get() + 1);
        controller.send(Message::HostStopResult(mismatched)).await;
        let message = timeout(IO_TIMEOUT, exchange)
            .await
            .expect("stop result timeout")
            .expect("stop task")
            .expect("stop response");
        let Message::HostStopResult(result) = message else {
            panic!("wrong stop response")
        };
        assert_ne!(result.result, StopResultCode::ControllerConfirmed);
        assert!(matches!(controller.receive().await, Message::ForceStop(_)));
        actor.abort();
    }
}
