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
use std::sync::Arc;
use std::time::{Duration, Instant};

use robot_protocol::v2::{
    decode_raw_frame, AcquireControl, AcquireResult, AcquireResultCode, ActuatorConfigFingerprint,
    AppliedResult, AppliedResultCode, BeginSession, ControlEpoch, ControllerCapabilities,
    ControllerFaults, ControllerHello, ControllerReady, ControllerUid, DeadlineRelation, ForceStop,
    ForceStopReason, Heartbeat, HeartbeatPeriodMs, HostCommand, HostCommandResult,
    HostCommandResultCode, HostStop, HostStopResult, MaxAbsPwmPercent, Message, MessageKind,
    NeutralOutput, ObservationalOdometry, OutputState, PhysicalStopSemantics, PwmFrequencyHz,
    RawFrame, RemainingLeaseMs, RequestId, StatusCode, StatusQuery, StatusReport, StopResultCode,
    TargetBootId, TimerPwm, UartEncodeError, UartRecord, UartStreamDecoder, UartStreamError,
    V2CommandSequence, WatchdogNominalPeriodMs, MAX_RAW_FRAME_BYTES,
};
use robot_protocol::ControllerUptimeMsWrapping;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::UdpSocket;
use tokio::sync::{mpsc, oneshot};
use tokio::task::{JoinHandle, JoinSet};
use tokio_serial::SerialPortBuilderExt;

use crate::config::ControllerServerConfigV1;
use crate::deadline::{
    conservative_remaining_lease, translate_command_deadline, HeartbeatClockSample,
    TranslatedCommandDeadline,
};

const SERIAL_BAUD: u32 = 115_200;
const ACTOR_MAILBOX_CAPACITY: usize = 32;
const MAX_UDP_EXCHANGES_IN_FLIGHT: usize = 64;
const INACTIVE_TIMER_SLEEP: Duration = Duration::from_secs(24 * 60 * 60);
const UDP_EMISSION_QUANTIZATION_MARGIN_MS: u64 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ActuationSnapshot {
    pub status: StatusCode,
    pub observed_boot_id: TargetBootId,
    pub control_epoch: Option<ControlEpoch>,
    pub controller_uptime: ControllerUptimeMsWrapping,
    pub output_state: OutputState,
    pub controller_timer_pwm: TimerPwm,
    pub faults: ControllerFaults,
    pub last_sequence: Option<V2CommandSequence>,
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
pub struct ActuationHandle {
    requests: mpsc::Sender<ActorRequest>,
}

impl ActuationHandle {
    /// Submit an already parsed request. `first_received_at` must be the
    /// original UDP receive instant and is never replaced for retries.
    pub async fn exchange(
        &self,
        source: SocketAddr,
        first_received_at: Instant,
        message: Message,
    ) -> Result<Message, ActuationHandleError> {
        let request = HostRequest::try_from(message)?;
        if first_received_at > Instant::now() {
            return Err(ActuationHandleError::FutureReceiveInstant);
        }
        let (response, receiver) = oneshot::channel();
        self.requests
            .send(ActorRequest {
                source,
                first_received_at,
                request,
                response,
            })
            .await
            .map_err(|_| ActuationHandleError::ActorStopped)?;
        receiver
            .await
            .map_err(|_| ActuationHandleError::ResponseDropped)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HostRequest {
    Acquire(AcquireControl),
    Command(HostCommand),
    Stop(HostStop),
    Status(StatusQuery),
}

impl HostRequest {
    const fn message(self) -> Message {
        match self {
            Self::Acquire(value) => Message::AcquireControl(value),
            Self::Command(value) => Message::HostCommand(value),
            Self::Stop(value) => Message::HostStop(value),
            Self::Status(value) => Message::StatusQuery(value),
        }
    }
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

struct ActorRequest {
    source: SocketAddr,
    first_received_at: Instant,
    request: HostRequest,
    response: oneshot::Sender<Message>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActuationHandleError {
    UnsupportedHostMessage(MessageKind),
    FutureReceiveInstant,
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
        }
    }
}

impl std::error::Error for ActuationStartError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::OpenSerial(source) | Self::ExclusiveSerial(source) => Some(source),
            Self::NonUtf8SerialDevice | Self::InvalidHeartbeatPeriod => None,
        }
    }
}

#[derive(Debug)]
pub enum ActuationActorError {
    SerialEof,
    SerialRead(io::Error),
    SerialWrite(io::Error),
    SerialFlush(io::Error),
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
            Self::SerialWrite(source) => {
                write!(formatter, "controller serial write failed: {source}")
            }
            Self::SerialFlush(source) => {
                write!(formatter, "controller serial flush failed: {source}")
            }
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
            Self::SerialRead(source) | Self::SerialWrite(source) | Self::SerialFlush(source) => {
                Some(source)
            }
            Self::Encode(source) => Some(source),
            Self::SerialEof | Self::InternalRequestIdExhausted => None,
        }
    }
}

/// Open only the configured device, claim OS-level exclusive ownership, and
/// spawn the sole serial/session actor.
pub async fn start_serial_actor(
    config: ControllerServerConfigV1,
    telemetry: Arc<dyn ActuationTelemetry>,
) -> Result<(ActuationHandle, JoinHandle<Result<(), ActuationActorError>>), ActuationStartError> {
    let actor_config = ActorConfig::from_server_config(&config)?;
    let device = config
        .serial_device()
        .to_str()
        .ok_or(ActuationStartError::NonUtf8SerialDevice)?;
    let mut port = tokio_serial::new(device, SERIAL_BAUD)
        .open_native_async()
        .map_err(ActuationStartError::OpenSerial)?;
    port.set_exclusive(true)
        .map_err(ActuationStartError::ExclusiveSerial)?;
    Ok(spawn_actor(port, actor_config, telemetry))
}

fn spawn_actor<Transport>(
    transport: Transport,
    config: ActorConfig,
    telemetry: Arc<dyn ActuationTelemetry>,
) -> (ActuationHandle, JoinHandle<Result<(), ActuationActorError>>)
where
    Transport: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    let (requests, receiver) = mpsc::channel(ACTOR_MAILBOX_CAPACITY);
    let handle = ActuationHandle { requests };
    let task = tokio::spawn(async move {
        SerialActor::new(transport, config, telemetry)
            .run(receiver)
            .await
    });
    (handle, task)
}

#[derive(Clone, Copy)]
struct ActorConfig {
    controller_uid: ControllerUid,
    firmware_abi: u16,
    firmware_build_id: u32,
    actuator_config_fingerprint: ActuatorConfigFingerprint,
    heartbeat_period: HeartbeatPeriodMs,
    maximum_heartbeat_age: Duration,
    serial_applied_ack_timeout: Duration,
    controller_clock_abs_error_ppm_bound: std::num::NonZeroU32,
    deadline_quantization_margin_ms: std::num::NonZeroU16,
    expected_max_abs_pwm_percent: MaxAbsPwmPercent,
    expected_pwm_frequency: PwmFrequencyHz,
    expected_watchdog_nominal_period: WatchdogNominalPeriodMs,
    expected_neutral_output: NeutralOutput,
    expected_physical_stop_semantics: PhysicalStopSemantics,
}

impl ActorConfig {
    fn from_server_config(config: &ControllerServerConfigV1) -> Result<Self, ActuationStartError> {
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
            serial_applied_ack_timeout: config.serial_applied_ack_timeout(),
            controller_clock_abs_error_ppm_bound: config.controller_clock_abs_error_ppm_bound(),
            deadline_quantization_margin_ms: config.deadline_quantization_margin_ms(),
            expected_max_abs_pwm_percent: config.expected_max_abs_pwm_percent(),
            expected_pwm_frequency: config.expected_pwm_frequency(),
            expected_watchdog_nominal_period: config.expected_watchdog_nominal_period(),
            expected_neutral_output: config.expected_neutral_output(),
            expected_physical_stop_semantics: config.expected_physical_stop_semantics(),
        })
    }
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
}

struct PendingCommand {
    source: SocketAddr,
    command: HostCommand,
    translated: TranslatedCommandDeadline,
    controller_uptime_reference: ControllerUptimeMsWrapping,
    serial_sent_at: Instant,
    response: oneshot::Sender<Message>,
}

struct PendingStop {
    source: SocketAddr,
    request: HostStop,
    serial_request_id: RequestId,
    serial_sent_at: Instant,
    response: oneshot::Sender<Message>,
}

enum PendingOperation {
    Command(PendingCommand),
    Stop(PendingStop),
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
    observed_hello: Option<TimedHello>,
    hello: Option<TimedHello>,
    ready: Option<ReadySession>,
    heartbeat: Option<TimedHeartbeat>,
    owner: Option<Owner>,
    pending: Option<PendingOperation>,
    cached_stop: Option<CachedStop>,
    last_internal_stop: Option<ForceStop>,
    next_internal_request_id: Option<RequestId>,
    faulted: bool,
}

impl<Transport> SerialActor<Transport>
where
    Transport: AsyncRead + AsyncWrite + Unpin,
{
    fn new(
        transport: Transport,
        config: ActorConfig,
        telemetry: Arc<dyn ActuationTelemetry>,
    ) -> Self {
        Self {
            transport,
            decoder: UartStreamDecoder::new(),
            config,
            telemetry,
            observed_hello: None,
            hello: None,
            ready: None,
            heartbeat: None,
            owner: None,
            pending: None,
            cached_stop: None,
            last_internal_stop: None,
            next_internal_request_id: Some(RequestId::new(0)),
            faulted: false,
        }
    }

    async fn run(
        &mut self,
        mut requests: mpsc::Receiver<ActorRequest>,
    ) -> Result<(), ActuationActorError> {
        let mut read_buffer = [0_u8; 256];
        self.publish_snapshot(Instant::now());

        loop {
            let wake_at = self.next_wake_at().unwrap_or_else(|| {
                Instant::now()
                    .checked_add(INACTIVE_TIMER_SLEEP)
                    .unwrap_or_else(Instant::now)
            });
            let sleep = tokio::time::sleep_until(tokio::time::Instant::from_std(wake_at));
            tokio::pin!(sleep);

            tokio::select! {
                read = self.transport.read(&mut read_buffer) => {
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
                                    "controller serial read failed and the best-effort stop also failed: {stop_error}"
                                );
                            }
                            self.clear_authority(true);
                            return Err(ActuationActorError::SerialRead(source));
                        }
                    };
                    if count == 0 {
                        self.fail_all_pending(HostCommandResultCode::ForceStopped, StopResultCode::ControllerUnavailable);
                        if let Err(stop_error) = self
                            .issue_internal_stop(ForceStopReason::TransportFault)
                            .await
                        {
                            log::error!(
                                "controller serial EOF prevented a confirmed stop; best-effort stop failed: {stop_error}"
                            );
                        }
                        self.clear_authority(true);
                        return Err(ActuationActorError::SerialEof);
                    }
                    for &byte in &read_buffer[..count] {
                        if let Some(decoded) = self.decoder.push(byte) {
                            let received_at = Instant::now();
                            match decoded {
                                Ok(message) => self.handle_serial_message(message, received_at).await?,
                                Err(error) => self.handle_framing_fault(error).await?,
                            }
                        }
                    }
                }
                request = requests.recv() => {
                    let Some(request) = request else {
                        self.fail_all_pending(HostCommandResultCode::ForceStopped, StopResultCode::ControllerUnavailable);
                        self.issue_internal_stop(ForceStopReason::TransportFault).await?;
                        self.clear_authority(true);
                        return Ok(());
                    };
                    self.enforce_freshness(Instant::now()).await?;
                    self.handle_host_request(request).await?;
                    self.publish_snapshot(Instant::now());
                }
                _ = &mut sleep => {
                    self.handle_timer(Instant::now()).await?;
                }
            }
        }
    }

    fn next_wake_at(&self) -> Option<Instant> {
        let pending = self.pending.as_ref().and_then(|pending| {
            let sent_at = match pending {
                PendingOperation::Command(value) => value.serial_sent_at,
                PendingOperation::Stop(value) => value.serial_sent_at,
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
                let result = self.status_report(value, Instant::now());
                let _ = request.response.send(Message::StatusReport(result));
            }
        }
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
                    .is_stopped_ready_for_acquisition()
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
        response: oneshot::Sender<Message>,
    ) -> Result<(), ActuationActorError> {
        if let Some(PendingOperation::Stop(_)) = &self.pending {
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
                let result = cached.duplicate_result_at(Instant::now());
                let _ = response.send(Message::HostCommandResult(result));
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
        response: oneshot::Sender<Message>,
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
                    self.protocol_fault(ForceStopReason::SessionReset).await?;
                }
            }
            _ => self.protocol_fault(ForceStopReason::TransportFault).await?,
        }
        self.publish_snapshot(received_at);
        Ok(())
    }

    async fn handle_hello(
        &mut self,
        hello: ControllerHello,
        _received_at: Instant,
    ) -> Result<(), ActuationActorError> {
        self.fail_all_pending(
            HostCommandResultCode::ControllerRestarted,
            StopResultCode::ControllerUnavailable,
        );
        self.clear_authority(true);
        let timed = TimedHello { message: hello };
        self.observed_hello = Some(timed);

        if !self.hello_is_exact(hello) {
            self.faulted = true;
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
        self.issue_stop_for(
            hello.controller_uid,
            TargetBootId::Exact(hello.boot_id),
            ForceStopReason::SessionReset,
        )
        .await?;
        let request_id = self.allocate_internal_request_id()?;
        self.send_serial(Message::BeginSession(BeginSession {
            controller_uid: hello.controller_uid,
            boot_id: hello.boot_id,
            request_id,
            heartbeat_period: self.config.heartbeat_period,
        }))
        .await?;
        Ok(())
    }

    async fn handle_ready(
        &mut self,
        ready: ControllerReady,
        received_at: Instant,
    ) -> Result<(), ActuationActorError> {
        let exact = self.hello.is_some_and(|hello| {
            ready.controller_uid == hello.message.controller_uid
                && ready.boot_id == hello.message.boot_id
                && ready.capabilities == hello.message.capabilities
                && ready.capabilities.supports_required_safety()
                && ready.output_state.is_safe()
                && ready.faults.is_clear()
        });
        if !exact {
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
                self.protocol_fault(ForceStopReason::ControllerFault)
                    .await?;
            }
            return Ok(());
        };
        let before_first_application = self
            .owner
            .as_ref()
            .is_none_or(|owner| owner.cached.is_none());
        let readiness_exact = if before_first_application {
            heartbeat.control_epoch.is_none()
                && heartbeat.last_sequence.is_none()
                && heartbeat.readiness.is_stopped_ready_for_acquisition()
                && heartbeat.timer_pwm.is_zero()
                && heartbeat.output_state.is_safe()
                && matches!(
                    heartbeat
                        .expires_at
                        .relation_to(heartbeat.controller_uptime),
                    DeadlineRelation::Expired
                )
        } else {
            heartbeat.control_epoch == Some(ready.message.control_epoch)
                && heartbeat.last_sequence.is_some()
                && heartbeat.readiness.is_ready()
        };
        let exact_session = heartbeat.controller_uid == ready.message.controller_uid
            && heartbeat.boot_id == ready.message.boot_id
            && readiness_exact
            && heartbeat.faults.is_clear();
        if !exact_session || !self.heartbeat_progresses(heartbeat) {
            self.protocol_fault(ForceStopReason::SessionReset).await?;
            return Ok(());
        }
        if !self.heartbeat_matches_authority(heartbeat) {
            self.protocol_fault(ForceStopReason::SequenceConflict)
                .await?;
            return Ok(());
        }
        self.heartbeat = Some(TimedHeartbeat {
            message: heartbeat,
            received_at,
        });
        self.faulted = false;
        Ok(())
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
                return heartbeat.timer_pwm == pending.command.requested_timer_pwm
                    && heartbeat.expires_at == pending.translated.controller_deadline_exclusive();
            }
        }
        if let Some(cached) = owner.cached {
            if heartbeat.last_sequence != Some(cached.command.sequence) {
                return false;
            }
            if heartbeat.timer_pwm == cached.controller_result.timer_pwm {
                return heartbeat.expires_at == cached.controller_result.expires_at;
            }
            return heartbeat.timer_pwm.is_zero() && heartbeat.output_state.is_safe();
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
            other @ PendingOperation::Stop(_) => {
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
        let applied_shape = result_code.is_some()
            && applied.timer_pwm == pending.command.requested_timer_pwm
            && applied.faults.is_clear()
            && applied
                .applied_at
                .wrapping_elapsed_since(pending.controller_uptime_reference)
                < 0x8000_0000
            && (pending.command.requested_timer_pwm.is_zero()
                || applied.result != AppliedResultCode::Stopped);
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
            .send(Message::HostCommandResult(host_result));
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
                if self.internal_stop_result_matches(result) {
                    self.last_internal_stop = None;
                    self.pending = Some(PendingOperation::Stop(pending));
                    return Ok(());
                }
                let failed =
                    self.failed_stop_result(pending.request, StopResultCode::ControllerFaulted);
                let _ = pending.response.send(Message::HostStopResult(failed));
                self.protocol_fault(ForceStopReason::SessionReset).await?;
                return Ok(());
            }
            Some(other @ PendingOperation::Command(_)) => self.pending = Some(other),
            None => {}
        }

        if self.internal_stop_result_matches(result) {
            self.last_internal_stop = None;
            return Ok(());
        }
        self.protocol_fault(ForceStopReason::SequenceConflict).await
    }

    async fn handle_framing_fault(
        &mut self,
        error: UartStreamError,
    ) -> Result<(), ActuationActorError> {
        log::error!("V2 controller framing fault: {error}");
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
            && hello.capabilities.supports_required_safety()
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
                    cached.command.sequence == V2CommandSequence::FIRST
                        && cached.controller_result.timer_pwm.is_zero()
                        && cached.controller_result.output_state.is_safe()
                })
        });
        let readiness_exact = if bootstrap_stopped {
            heartbeat.message.control_epoch.is_none()
                && heartbeat.message.last_sequence.is_none()
                && heartbeat
                    .message
                    .readiness
                    .is_stopped_ready_for_acquisition()
        } else {
            heartbeat.message.control_epoch == Some(command.control_epoch)
                && heartbeat.message.last_sequence.is_some()
                && heartbeat.message.readiness.is_ready()
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
        let record = UartRecord::encode(message).map_err(ActuationActorError::Encode)?;
        let sent_at = Instant::now();
        self.transport
            .write_all(record.as_bytes())
            .await
            .map_err(ActuationActorError::SerialWrite)?;
        self.transport
            .flush()
            .await
            .map_err(ActuationActorError::SerialFlush)?;
        Ok(sent_at)
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
        self.pending = None;
        self.heartbeat = None;
        if clear_session {
            self.hello = None;
            self.ready = None;
            self.cached_stop = None;
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
        }
    }

    fn failed_command_result(
        &self,
        command: HostCommand,
        code: HostCommandResultCode,
    ) -> HostCommandResult {
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
        ActuationSnapshot {
            status: status.status,
            observed_boot_id: status.observed_boot_id,
            control_epoch: status.control_epoch,
            controller_uptime: status.controller_uptime,
            output_state: status.output_state,
            controller_timer_pwm: status.controller_timer_pwm,
            faults: status.faults,
            last_sequence: self
                .owner
                .as_ref()
                .and_then(|owner| owner.cached)
                .map(|cached| cached.command.sequence),
        }
    }

    fn publish_snapshot(&self, now: Instant) {
        self.telemetry.update_actuation(self.snapshot(now));
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

/// Serve binary V2 datagrams. Each datagram is parsed once and only the four
/// host request kinds enter the actor. Work is bounded; backpressure reaches
/// the UDP receive queue rather than creating an unbounded task population.
pub async fn udp_service(bind: SocketAddr, handle: ActuationHandle) -> Result<(), UdpServiceError> {
    udp_service_inner(bind, Some(handle)).await
}

/// Keep the loopback V2 endpoint truthful when no controller authority was
/// configured. Status remains queryable and no request can reach actuation.
pub async fn unavailable_udp_service(bind: SocketAddr) -> Result<(), UdpServiceError> {
    udp_service_inner(bind, None).await
}

async fn udp_service_inner(
    bind: SocketAddr,
    handle: Option<ActuationHandle>,
) -> Result<(), UdpServiceError> {
    if !bind.ip().is_loopback() || bind.port() == 0 {
        return Err(UdpServiceError::BindMustBeLoopback(bind));
    }
    let socket = UdpSocket::bind(bind).await.map_err(UdpServiceError::Bind)?;
    udp_service_on_socket(socket, handle).await
}

async fn udp_service_on_socket(
    socket: UdpSocket,
    handle: Option<ActuationHandle>,
) -> Result<(), UdpServiceError> {
    let socket = Arc::new(socket);
    let mut buffer = [0_u8; MAX_RAW_FRAME_BYTES + 1];
    let mut exchanges = JoinSet::new();

    loop {
        while exchanges.len() >= MAX_UDP_EXCHANGES_IN_FLIGHT {
            if let Some(result) = exchanges.join_next().await {
                log_udp_task_result(result);
            }
        }
        let (length, source) = socket
            .recv_from(&mut buffer)
            .await
            .map_err(UdpServiceError::Receive)?;
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
        let socket = Arc::clone(&socket);
        let handle = handle.clone();
        exchanges.spawn(async move {
            let response = match handle {
                Some(handle) => {
                    match handle
                        .exchange(source, first_received_at, request.message())
                        .await
                    {
                        Ok(response) => response,
                        Err(error) => {
                            log::error!("V2 serial actor could not answer {source}: {error}");
                            unavailable_response(request)
                        }
                    }
                }
                None => unavailable_response(request),
            };
            let (sent, expected) = loop {
                let emission_response =
                    response_for_udp_emission(request, first_received_at, response, Instant::now());
                let frame =
                    RawFrame::encode(emission_response).map_err(UdpExchangeError::Encode)?;
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
        });
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
    emitted_at: Instant,
) -> Message {
    let elapsed_ms = emitted_at
        .checked_duration_since(first_received_at)
        .map(duration_millis_ceil_saturating)
        .unwrap_or(u64::MAX)
        .saturating_add(UDP_EMISSION_QUANTIZATION_MARGIN_MS);
    match (request, response) {
        (HostRequest::Command(request), Message::HostCommandResult(mut result)) => {
            let actor_bound = u64::from(result.remaining_lease.get()).saturating_sub(elapsed_ms);
            let server_bound = first_received_at
                .checked_add(Duration::from_millis(u64::from(request.lease.get())))
                .and_then(|deadline| deadline.checked_duration_since(emitted_at))
                .map(|duration| u64::try_from(duration.as_millis()).unwrap_or(u64::MAX))
                .unwrap_or(0);
            result.remaining_lease = bounded_remaining_lease(actor_bound.min(server_bound));
            Message::HostCommandResult(result)
        }
        (HostRequest::Status(_), Message::StatusReport(mut result)) => {
            let remaining = u64::from(result.remaining_lease.get()).saturating_sub(elapsed_ms);
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
    use std::num::{NonZeroU16, NonZeroU32};

    use robot_command_client::{
        ClientConfig, DisarmedCommandClient, MonotonicInstant, PendingPhysicalCommand,
        StopRecoveryPolicy, SystemMonotonicClock, TimeoutNs, UdpEndpoint, UdpV2Transport,
    };
    use robot_protocol::v2::{
        ApplyPwm, ControllerDeadlineMsWrapping, ControllerReady, ReadinessFlags, V2CommandLeaseMs,
    };
    use tokio::io::DuplexStream;
    use tokio::time::{sleep, timeout};

    const IO_TIMEOUT: Duration = Duration::from_millis(250);
    const SHORT_ABSENCE: Duration = Duration::from_millis(5);

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
            heartbeat_period: HeartbeatPeriodMs::try_new(10).expect("heartbeat period"),
            maximum_heartbeat_age: Duration::from_millis(80),
            serial_applied_ack_timeout: Duration::from_millis(25),
            controller_clock_abs_error_ppm_bound: NonZeroU32::new(1_000).expect("ppm"),
            deadline_quantization_margin_ms: NonZeroU16::new(1).expect("margin"),
            expected_max_abs_pwm_percent: MaxAbsPwmPercent::try_new(50).expect("PWM maximum"),
            expected_pwm_frequency: PwmFrequencyHz::try_new(20_000).expect("PWM frequency"),
            expected_watchdog_nominal_period: WatchdogNominalPeriodMs::try_new(250)
                .expect("watchdog period"),
            expected_neutral_output: NeutralOutput::BothLow,
            expected_physical_stop_semantics: PhysicalStopSemantics::CoastVerified,
        }
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

    fn ready(boot_id: robot_protocol::v2::ControllerBootId) -> ControllerReady {
        ControllerReady {
            controller_uid: uid(),
            boot_id,
            control_epoch: epoch(),
            controller_uptime: ControllerUptimeMsWrapping::new(1_000),
            capabilities: capabilities(),
            output_state: OutputState::Disabled,
            faults: ControllerFaults::NONE,
        }
    }

    fn zero_heartbeat(boot_id: robot_protocol::v2::ControllerBootId) -> Heartbeat {
        Heartbeat {
            controller_uid: uid(),
            boot_id,
            control_epoch: None,
            last_sequence: None,
            controller_uptime: ControllerUptimeMsWrapping::new(1_000),
            expires_at: ControllerDeadlineMsWrapping::new(1_000),
            timer_pwm: TimerPwm::ZERO,
            output_state: OutputState::ZeroPwm,
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
            applied_at: ControllerUptimeMsWrapping::new(1_002),
            expires_at: apply.expires_at,
            faults: ControllerFaults::NONE,
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
            timeout(IO_TIMEOUT, async {
                loop {
                    let byte = self.stream.read_u8().await.expect("fake serial read");
                    if let Some(decoded) = self.decoder.push(byte) {
                        return decoded.expect("actor emits valid V2 only");
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
        controller: FakeController,
        actor: JoinHandle<Result<(), ActuationActorError>>,
        source: SocketAddr,
        boot_id: robot_protocol::v2::ControllerBootId,
    }

    impl Harness {
        async fn ready() -> Self {
            let boot_id = boot(7);
            let (actor_stream, controller_stream) = tokio::io::duplex(4_096);
            let (handle, actor) = spawn_actor(
                actor_stream,
                actor_config(),
                Arc::new(NoopActuationTelemetry),
            );
            let mut controller = FakeController::new(controller_stream);
            controller
                .send(Message::ControllerHello(hello_with(uid(), boot_id)))
                .await;
            let stop = match controller.receive().await {
                Message::ForceStop(value) => value,
                other => panic!("expected startup ForceStop, got {:?}", other.kind()),
            };
            assert!(matches!(
                controller.receive().await,
                Message::BeginSession(_)
            ));
            controller
                .send(Message::HostStopResult(confirmed_stop(stop, boot_id)))
                .await;
            controller
                .send(Message::ControllerReady(ready(boot_id)))
                .await;
            controller
                .send(Message::Heartbeat(zero_heartbeat(boot_id)))
                .await;
            let source = "127.0.0.1:41000".parse().expect("source address");
            let mut harness = Self {
                handle,
                controller,
                actor,
                source,
                boot_id,
            };
            harness.wait_until_ready_stopped().await;
            harness
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
            other => panic!("expected ApplyPwm, got {:?}", other.kind()),
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
        assert!(matches!(stop, Message::ForceStop(_)));
        assert!(matches!(
            harness.controller.receive().await,
            Message::BeginSession(_)
        ));
        harness.abort();
    }

    #[tokio::test]
    async fn wrong_uid_never_reaches_begin_session() {
        let (actor_stream, controller_stream) = tokio::io::duplex(4_096);
        let (handle, actor) = spawn_actor(
            actor_stream,
            actor_config(),
            Arc::new(NoopActuationTelemetry),
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
        let actor = SerialActor::new(
            actor_stream,
            actor_config(),
            Arc::new(NoopActuationTelemetry),
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
    fn udp_emission_shortens_actor_remaining_lifetime_instead_of_renewing_it() {
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
        ) else {
            panic!("wrong response kind")
        };
        assert_eq!(emitted.remaining_lease.get(), 89);
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
        let (handle, actor) = spawn_actor(
            actor_stream,
            actor_config(),
            Arc::new(NoopActuationTelemetry),
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
        let (handle, actor) = spawn_actor(
            actor_stream,
            actor_config(),
            Arc::new(NoopActuationTelemetry),
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
