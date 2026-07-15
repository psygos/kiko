#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

use log::{debug, error, info, warn};
use robot_protocol::{
    CommandSequence, LeasedPwmCommand, LeasedPwmCommandError, RobotCommand,
    RobotCommandAcknowledgement, RobotCommandAcknowledgementPacket, RobotCommandPacket,
    RobotOdometryWithServerReceiveAge,
};
use serde::Serialize;
use std::net::{SocketAddr, ToSocketAddrs, UdpSocket};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};
use tauri::{Manager, State};

const SERVER_COMMAND_LEASE_MS: u16 = 150;
const UI_DESIRED_PWM_LEASE: Duration = Duration::from_millis(150);
const MAX_ODOMETRY_RESPONSE_BYTES: usize = 4 * 1_024;

fn error_with_sources(error: &(dyn std::error::Error + 'static)) -> String {
    const MAX_SOURCE_DEPTH: usize = 16;

    let mut message = error.to_string();
    let mut source = error.source();
    let mut depth = 0;
    while let Some(current) = source {
        if depth == MAX_SOURCE_DEPTH {
            message.push_str(": source chain exceeds 16 levels");
            break;
        }
        let source_message = current.to_string();
        if !message.ends_with(&source_message) {
            message.push_str(": ");
            message.push_str(&source_message);
        }
        source = current.source();
        depth += 1;
    }
    message
}

#[derive(Debug, Clone, Serialize)]
pub struct CommandAcknowledgementUpdate {
    pub accepted_sequence: u32,
    pub round_trip_latency_ms: u32,
    pub commanded_left_pwm_percent: i8,
    pub commanded_right_pwm_percent: i8,
}

#[derive(Clone, Copy, Serialize)]
struct CommandAcknowledgementEvent<'a> {
    stream_generation_decimal: &'a str,
    #[serde(flatten)]
    acknowledgement: &'a CommandAcknowledgementUpdate,
}

#[derive(Clone, Copy, Serialize)]
struct ConnectionFailureEvent<'a> {
    stream_generation_decimal: &'a str,
    message: &'a str,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConnectionUpdate {
    pub server_addr: String,
    pub stream_generation_decimal: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct RobotOdometryUpdate {
    pub left_estimated_extended_ticks_wrapping_i64: String,
    pub right_estimated_extended_ticks_wrapping_i64: String,
    pub left_sample_delta_ticks_modulo_i16: i16,
    pub right_sample_delta_ticks_modulo_i16: i16,
    pub controller_uptime_ms_wrapping: u32,
    pub server_receive_age_ms_decimal: String,
}

impl From<RobotOdometryWithServerReceiveAge> for RobotOdometryUpdate {
    fn from(observation: RobotOdometryWithServerReceiveAge) -> Self {
        let odometry = observation.odometry();
        Self {
            // Decimal strings preserve the full i64 domain across JavaScript's
            // narrower exact-integer range.
            left_estimated_extended_ticks_wrapping_i64: odometry
                .left_estimated_extended_ticks_wrapping_i64()
                .get()
                .to_string(),
            right_estimated_extended_ticks_wrapping_i64: odometry
                .right_estimated_extended_ticks_wrapping_i64()
                .get()
                .to_string(),
            left_sample_delta_ticks_modulo_i16: odometry.left_sample_delta_ticks_modulo_i16().get(),
            right_sample_delta_ticks_modulo_i16: odometry
                .right_sample_delta_ticks_modulo_i16()
                .get(),
            controller_uptime_ms_wrapping: odometry.controller_uptime_ms_wrapping().get(),
            server_receive_age_ms_decimal: observation.server_receive_age_ms().to_string(),
        }
    }
}

#[derive(Debug)]
struct AcknowledgementSequenceMismatch {
    sent: u32,
    acknowledged: u32,
}

impl std::fmt::Display for AcknowledgementSequenceMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "robot server acknowledged sequence {}, but the client sent {}",
            self.acknowledged, self.sent
        )
    }
}

impl std::error::Error for AcknowledgementSequenceMismatch {}

#[derive(Debug)]
enum CommandDatagramSendError {
    Io(std::io::Error),
    Incomplete { expected: usize, sent: usize },
}

impl std::fmt::Display for CommandDatagramSendError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(source) => write!(formatter, "failed to send command datagram: {source}"),
            Self::Incomplete { expected, sent } => write!(
                formatter,
                "command datagram send reported {sent} bytes instead of the complete {expected}-byte packet"
            ),
        }
    }
}

impl std::error::Error for CommandDatagramSendError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(source) => Some(source),
            Self::Incomplete { .. } => None,
        }
    }
}

fn send_complete_command_datagram(
    socket: &UdpSocket,
    packet: &[u8],
) -> Result<(), CommandDatagramSendError> {
    let sent = socket.send(packet).map_err(CommandDatagramSendError::Io)?;
    if sent == packet.len() {
        Ok(())
    } else {
        Err(CommandDatagramSendError::Incomplete {
            expected: packet.len(),
            sent,
        })
    }
}

#[derive(Clone, Copy, Debug)]
enum CommandSocketAttemptStage {
    Bind,
    Connect,
}

impl std::fmt::Display for CommandSocketAttemptStage {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bind => formatter.write_str("bind a same-family local socket"),
            Self::Connect => formatter.write_str("connect the UDP socket"),
        }
    }
}

#[derive(Debug)]
struct CommandSocketAttemptError {
    candidate: SocketAddr,
    stage: CommandSocketAttemptStage,
    source: std::io::Error,
}

#[derive(Debug)]
enum CommandSocketError {
    Resolve {
        address: String,
        source: std::io::Error,
    },
    NoResolvedAddress {
        address: String,
    },
    AllCandidatesFailed {
        address: String,
        attempts: Vec<CommandSocketAttemptError>,
    },
}

impl std::fmt::Display for CommandSocketError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Resolve { address, source } => {
                write!(
                    formatter,
                    "failed to resolve command address {address}: {source}"
                )
            }
            Self::NoResolvedAddress { address } => {
                write!(
                    formatter,
                    "command address {address} resolved to no socket addresses"
                )
            }
            Self::AllCandidatesFailed { address, attempts } => {
                write!(formatter, "failed to create a command socket for {address}")?;
                for attempt in attempts {
                    write!(
                        formatter,
                        "; could not {} for {}: {}",
                        attempt.stage, attempt.candidate, attempt.source
                    )?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for CommandSocketError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Resolve { source, .. } => Some(source),
            Self::AllCandidatesFailed { attempts, .. } => attempts
                .last()
                .map(|attempt| &attempt.source as &(dyn std::error::Error + 'static)),
            Self::NoResolvedAddress { .. } => None,
        }
    }
}

fn unspecified_bind_address(candidate: SocketAddr) -> SocketAddr {
    match candidate {
        SocketAddr::V4(_) => SocketAddr::from(([0, 0, 0, 0], 0)),
        SocketAddr::V6(_) => SocketAddr::from(([0, 0, 0, 0, 0, 0, 0, 0], 0)),
    }
}

fn connect_command_socket(address: &str) -> Result<UdpSocket, CommandSocketError> {
    let candidates = address
        .to_socket_addrs()
        .map_err(|source| CommandSocketError::Resolve {
            address: address.to_owned(),
            source,
        })?;
    let mut resolved_any = false;
    let mut attempts = Vec::new();
    for candidate in candidates {
        resolved_any = true;
        let socket = match UdpSocket::bind(unspecified_bind_address(candidate)) {
            Ok(socket) => socket,
            Err(source) => {
                attempts.push(CommandSocketAttemptError {
                    candidate,
                    stage: CommandSocketAttemptStage::Bind,
                    source,
                });
                continue;
            }
        };
        match socket.connect(candidate) {
            Ok(()) => return Ok(socket),
            Err(source) => attempts.push(CommandSocketAttemptError {
                candidate,
                stage: CommandSocketAttemptStage::Connect,
                source,
            }),
        }
    }

    if resolved_any {
        Err(CommandSocketError::AllCandidatesFailed {
            address: address.to_owned(),
            attempts,
        })
    } else {
        Err(CommandSocketError::NoResolvedAddress {
            address: address.to_owned(),
        })
    }
}

fn decode_acknowledgement(
    bytes: &[u8],
    expected_sequence: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let packet = RobotCommandAcknowledgementPacket::try_from_legacy_wire_bytes(bytes)?;
    let acknowledgement = RobotCommandAcknowledgement::try_from(packet)?;
    if acknowledgement.accepted_sequence().get() != expected_sequence {
        return Err(Box::new(AcknowledgementSequenceMismatch {
            sent: expected_sequence,
            acknowledged: acknowledgement.accepted_sequence().get(),
        }));
    }
    Ok(())
}

struct CommandStream {
    socket: UdpSocket,
    sequence: u32,
    server_addr: String,
    desired_pwm: DesiredPwm,
    base_http_url: String,
    http_client: reqwest::Client,
}

impl CommandStream {
    fn new(
        server_addr: String,
        http_port: Option<u16>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        info!("Creating new command stream to {server_addr}");

        let socket = connect_command_socket(&server_addr)?;
        let command_peer_ip = socket.peer_addr()?.ip();
        info!("UDP command socket resolved and connected");

        socket.set_read_timeout(Some(Duration::from_millis(50)))?;
        debug!("Socket acknowledgement timeout configured");

        // Acquire the server-side source session with a zero-PWM command.
        let leased_pwm = LeasedPwmCommand::try_new(0, 0, SERVER_COMMAND_LEASE_MS)?;
        let desired_pwm = DesiredPwm::Stopped(leased_pwm);
        let test_cmd =
            RobotCommand::from_leased_pwm(desired_pwm.command(), CommandSequence::new(0));

        let test_packet = RobotCommandPacket::from(test_cmd).to_legacy_wire_bytes();
        send_complete_command_datagram(&socket, &test_packet)?;
        info!("Zero-PWM acquisition command sent to {server_addr}");

        let mut buf = [0u8; 1024];
        let len = socket.recv(&mut buf)?;
        decode_acknowledgement(&buf[..len], 0)?;
        info!("Server accepted the zero-PWM acquisition sequence");

        let http_port = http_port.unwrap_or(3030);
        let base_http_url = match command_peer_ip {
            std::net::IpAddr::V4(address) => format!("http://{address}:{http_port}"),
            std::net::IpAddr::V6(address) => format!("http://[{address}]:{http_port}"),
        };
        let http_client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()?;

        Ok(CommandStream {
            socket,
            sequence: 0,
            server_addr,
            desired_pwm,
            base_http_url,
            http_client,
        })
    }

    fn send_command(&mut self) -> Result<CommandAcknowledgementUpdate, Box<dyn std::error::Error>> {
        let (leased_pwm, expired) = self.desired_pwm.command_at(Instant::now());
        if expired {
            warn!("UI desired-PWM lease expired; switching the network stream to zero PWM");
        }
        self.sequence = self.sequence.wrapping_add(1);

        let command =
            RobotCommand::from_leased_pwm(leased_pwm, CommandSequence::new(self.sequence));

        let start_time = Instant::now();

        // Send command
        let packet = RobotCommandPacket::from(command).to_legacy_wire_bytes();
        send_complete_command_datagram(&self.socket, &packet)?;
        debug!(
            "Sent {} bytes (seq: {}, L: {}, R: {}) to {}",
            packet.len(),
            self.sequence,
            leased_pwm.left_pwm_percent().get(),
            leased_pwm.right_pwm_percent().get(),
            self.server_addr
        );

        let mut buf = [0u8; 1024];
        let len = self.socket.recv(&mut buf)?;
        decode_acknowledgement(&buf[..len], self.sequence)?;
        let round_trip_latency_ms = u32::try_from(start_time.elapsed().as_millis())?;
        debug!(
            "Received command acknowledgement: sequence={}, round_trip_latency={}ms",
            self.sequence, round_trip_latency_ms,
        );

        Ok(CommandAcknowledgementUpdate {
            accepted_sequence: self.sequence,
            round_trip_latency_ms,
            commanded_left_pwm_percent: leased_pwm.left_pwm_percent().get(),
            commanded_right_pwm_percent: leased_pwm.right_pwm_percent().get(),
        })
    }

    fn set_pwm(&mut self, left: i8, right: i8) -> Result<(), DesiredPwmUpdateError> {
        let old_command = self.desired_pwm.command();
        self.desired_pwm.update_at(left, right, Instant::now())?;
        let command = self.desired_pwm.command();

        if old_command != command {
            info!(
                "PWM command changed: left={} -> {}%, right={} -> {}%",
                old_command.left_pwm_percent().get(),
                command.left_pwm_percent().get(),
                old_command.right_pwm_percent().get(),
                command.right_pwm_percent().get(),
            );
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
enum DesiredPwm {
    Stopped(LeasedPwmCommand),
    Active {
        command: LeasedPwmCommand,
        valid_until: Instant,
    },
}

impl DesiredPwm {
    const fn command(self) -> LeasedPwmCommand {
        match self {
            Self::Stopped(command) | Self::Active { command, .. } => command,
        }
    }

    fn update_at(
        &mut self,
        left: i8,
        right: i8,
        now: Instant,
    ) -> Result<(), DesiredPwmUpdateError> {
        let command = LeasedPwmCommand::try_new(left, right, SERVER_COMMAND_LEASE_MS)
            .map_err(DesiredPwmUpdateError::InvalidCommand)?;
        *self = if command.is_stop() {
            Self::Stopped(command)
        } else {
            Self::Active {
                command,
                valid_until: now
                    .checked_add(UI_DESIRED_PWM_LEASE)
                    .ok_or(DesiredPwmUpdateError::DeadlineOverflow)?,
            }
        };
        Ok(())
    }

    fn command_at(&mut self, now: Instant) -> (LeasedPwmCommand, bool) {
        match *self {
            Self::Active {
                command,
                valid_until,
            } if now >= valid_until => {
                let stopped = LeasedPwmCommand::from_validated(
                    robot_protocol::PwmPercent::ZERO,
                    robot_protocol::PwmPercent::ZERO,
                    command.lease_ms(),
                );
                *self = Self::Stopped(stopped);
                (stopped, true)
            }
            _ => (self.command(), false),
        }
    }
}

#[derive(Debug)]
enum DesiredPwmUpdateError {
    InvalidCommand(LeasedPwmCommandError),
    DeadlineOverflow,
}

impl std::fmt::Display for DesiredPwmUpdateError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidCommand(source) => write!(formatter, "invalid desired PWM: {source}"),
            Self::DeadlineOverflow => {
                formatter.write_str("desired-PWM lease deadline exceeds the monotonic clock domain")
            }
        }
    }
}

impl std::error::Error for DesiredPwmUpdateError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidCommand(source) => Some(source),
            Self::DeadlineOverflow => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct StreamGeneration(u64);

#[derive(Default)]
struct CommandState {
    generation: StreamGeneration,
    stream: Option<CommandStream>,
}

impl CommandState {
    fn invalidate(&mut self) -> Result<StreamGeneration, String> {
        let next = self
            .generation
            .0
            .checked_add(1)
            .ok_or_else(|| "Command stream generation counter is exhausted".to_string())?;
        self.stream = None;
        self.generation = StreamGeneration(next);
        Ok(self.generation)
    }
}

#[derive(Debug)]
enum StreamGenerationParseError {
    InvalidInteger(std::num::ParseIntError),
    NonCanonical,
}

impl std::fmt::Display for StreamGenerationParseError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInteger(source) => write!(
                formatter,
                "stream generation must be an unsigned decimal integer: {source}"
            ),
            Self::NonCanonical => formatter.write_str(
                "stream generation must use canonical unsigned decimal notation without signs or leading zeroes",
            ),
        }
    }
}

impl std::error::Error for StreamGenerationParseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidInteger(source) => Some(source),
            Self::NonCanonical => None,
        }
    }
}

fn parse_stream_generation_decimal(
    stream_generation_decimal: &str,
) -> Result<StreamGeneration, StreamGenerationParseError> {
    let generation = stream_generation_decimal
        .parse()
        .map(StreamGeneration)
        .map_err(StreamGenerationParseError::InvalidInteger)?;
    if generation.0.to_string() != stream_generation_decimal {
        return Err(StreamGenerationParseError::NonCanonical);
    }
    Ok(generation)
}

fn require_current_generation(
    state: &CommandState,
    requested: StreamGeneration,
) -> Result<(), String> {
    if state.generation == requested {
        Ok(())
    } else {
        Err(format!(
            "Command belongs to superseded stream generation {}; current generation is {}",
            requested.0, state.generation.0
        ))
    }
}

enum CommandLoopStep {
    Acknowledged(CommandAcknowledgementUpdate),
    Superseded,
    Failed(String),
}

type StreamState = Arc<Mutex<CommandState>>;

fn lock_stream_state(state: &StreamState) -> Result<MutexGuard<'_, CommandState>, String> {
    state
        .lock()
        .map_err(|_| "Command stream state is unavailable after an internal panic".to_string())
}

fn emit_connection_failure(
    app_handle: &tauri::AppHandle,
    stream_generation_decimal: &str,
    failure: &str,
) {
    error!("Command stream failed: {failure}");
    let event = ConnectionFailureEvent {
        stream_generation_decimal,
        message: failure,
    };
    if let Err(error) = app_handle.emit_all("connection-error", event) {
        error!("Failed to emit connection error: {error}");
    }
    if let Err(error) = app_handle.emit_all("connection-lost", event) {
        error!("Failed to emit connection-lost event: {error}");
    }
}

fn stream_generation_is_current(
    state: &StreamState,
    generation: StreamGeneration,
) -> Result<bool, String> {
    lock_stream_state(state).map(|state| state.generation == generation)
}

#[tauri::command]
async fn connect(
    state: State<'_, StreamState>,
    address: String,
    http_port: Option<u16>,
    app_handle: tauri::AppHandle,
) -> Result<ConnectionUpdate, String> {
    let generation = lock_stream_state(state.inner())?.invalidate()?;

    info!("Attempting to connect to: {address}");

    let server_addr = address;

    info!("Parsed server address: {server_addr}");

    // Create new connection
    let stream = CommandStream::new(server_addr.clone(), http_port).map_err(|e| {
        let detail = error_with_sources(e.as_ref());
        error!("Failed to create command stream: {detail}");
        format!("Connection failed: {detail}")
    })?;

    info!("Command stream created successfully");

    {
        let mut state_guard = lock_stream_state(state.inner())?;
        if state_guard.generation != generation {
            return Err("Connection attempt was superseded by a newer request".to_string());
        }
        state_guard.stream = Some(stream);
    }

    // Start command streaming thread
    let state_clone = state.inner().clone();
    let stream_generation_decimal = generation.0.to_string();
    let event_stream_generation_decimal = stream_generation_decimal.clone();
    let spawn_result = std::thread::Builder::new()
        .name(format!("robot-command-stream-{}", generation.0))
        .spawn(move || loop {
            std::thread::sleep(Duration::from_millis(40));

            let step = {
                let mut state_guard = match lock_stream_state(&state_clone) {
                    Ok(guard) => guard,
                    Err(error) => {
                        emit_connection_failure(
                            &app_handle,
                            &event_stream_generation_decimal,
                            &error,
                        );
                        return;
                    }
                };
                if state_guard.generation != generation {
                    CommandLoopStep::Superseded
                } else if let Some(stream) = state_guard.stream.as_mut() {
                    match stream.send_command() {
                        Ok(update) => CommandLoopStep::Acknowledged(update),
                        Err(e) => {
                            let error = error_with_sources(e.as_ref());
                            state_guard.stream = None;
                            CommandLoopStep::Failed(error)
                        }
                    }
                } else {
                    CommandLoopStep::Superseded
                }
            };

            match step {
                CommandLoopStep::Acknowledged(update) => {
                    match stream_generation_is_current(&state_clone, generation) {
                        Ok(true) => {}
                        Ok(false) => return,
                        Err(error) => {
                            emit_connection_failure(
                                &app_handle,
                                &event_stream_generation_decimal,
                                &error,
                            );
                            return;
                        }
                    }
                    let event = CommandAcknowledgementEvent {
                        stream_generation_decimal: &event_stream_generation_decimal,
                        acknowledgement: &update,
                    };
                    if let Err(error) = app_handle.emit_all("command-acknowledgement", event) {
                        warn!("Failed to emit command acknowledgement: {error}");
                    }
                }
                CommandLoopStep::Superseded => return,
                CommandLoopStep::Failed(error) => {
                    match stream_generation_is_current(&state_clone, generation) {
                        Ok(true) => emit_connection_failure(
                            &app_handle,
                            &event_stream_generation_decimal,
                            &error,
                        ),
                        Ok(false) => {}
                        Err(state_error) => {
                            let combined = format!("{error}; additionally, {state_error}");
                            emit_connection_failure(
                                &app_handle,
                                &event_stream_generation_decimal,
                                &combined,
                            );
                        }
                    }
                    return;
                }
            }
        });
    if let Err(source) = spawn_result {
        let cleanup_result = lock_stream_state(state.inner()).and_then(|mut state| {
            if state.generation == generation {
                state.invalidate().map(|_| ())
            } else {
                Ok(())
            }
        });
        error!("Failed to spawn command-stream thread: {source}");
        return match cleanup_result {
            Ok(()) => Err(format!("Failed to spawn command-stream thread: {source}")),
            Err(cleanup_error) => Err(format!(
                "Failed to spawn command-stream thread: {source}; additionally, {cleanup_error}"
            )),
        };
    }

    Ok(ConnectionUpdate {
        server_addr,
        stream_generation_decimal,
    })
}

#[tauri::command]
async fn disconnect(
    state: State<'_, StreamState>,
    stream_generation_decimal: String,
) -> Result<(), String> {
    info!("Disconnecting from robot server");
    let requested_generation = parse_stream_generation_decimal(&stream_generation_decimal)
        .map_err(|error| error.to_string())?;
    let mut state_guard = lock_stream_state(state.inner())?;
    require_current_generation(&state_guard, requested_generation)?;
    let stop_result = if let Some(stream) = state_guard.stream.as_mut() {
        stream
            .set_pwm(0, 0)
            .map_err(|error| format!("failed to prepare zero-PWM disconnect command: {error}"))
            .and_then(|()| {
                stream.send_command().map(|_| ()).map_err(|error| {
                    format!("zero-PWM disconnect command was not acknowledged: {error}")
                })
            })
    } else {
        Ok(())
    };
    let invalidation_result = state_guard.invalidate();
    drop(state_guard);

    match (stop_result, invalidation_result) {
        (Ok(()), Ok(_)) => {
            info!("Disconnected after the zero-PWM command was acknowledged");
            Ok(())
        }
        (Err(stop_error), Ok(_)) => Err(format!(
            "Disconnected locally, but {stop_error}; command-lease expiry remains the software fallback"
        )),
        (Ok(()), Err(invalidation_error)) => Err(invalidation_error),
        (Err(stop_error), Err(invalidation_error)) => Err(format!(
            "Disconnected locally, but {stop_error}; additionally, {invalidation_error}"
        )),
    }
}

#[tauri::command]
async fn set_motor_pwm(
    state: State<'_, StreamState>,
    left: i8,
    right: i8,
    stream_generation_decimal: String,
) -> Result<(), String> {
    let requested_generation = parse_stream_generation_decimal(&stream_generation_decimal)
        .map_err(|error| error.to_string())?;
    let mut state_guard = lock_stream_state(state.inner())?;
    require_current_generation(&state_guard, requested_generation)?;
    let update_result = if let Some(stream) = state_guard.stream.as_mut() {
        stream
            .set_pwm(left, right)
            .map_err(|error| error.to_string())
    } else {
        warn!("Attempted to set motor PWM while not connected");
        return Err("Not connected".to_string());
    };
    if let Err(update_error) = update_result {
        let fail_safe_stop_result = state_guard
            .stream
            .as_mut()
            .ok_or_else(|| "command stream disappeared before fail-safe stop".to_string())
            .and_then(|stream| {
                stream
                    .set_pwm(0, 0)
                    .map_err(|error| format!("failed to prepare fail-safe zero PWM: {error}"))?;
                stream
                    .send_command()
                    .map(|_| ())
                    .map_err(|error| format!("fail-safe zero PWM was not acknowledged: {error}"))
            });
        let invalidation_result = state_guard.invalidate();
        return Err(match (fail_safe_stop_result, invalidation_result) {
            (Ok(()), Ok(_)) => format!(
                "Rejected desired PWM update ({update_error}); fail-safe zero PWM was acknowledged and the stream was discarded"
            ),
            (Err(stop_error), Ok(_)) => format!(
                "Rejected desired PWM update ({update_error}); {stop_error}; the stream was discarded and lease expiry remains the software fallback"
            ),
            (Ok(()), Err(invalidation_error)) => format!(
                "Rejected desired PWM update ({update_error}); fail-safe zero PWM was acknowledged; additionally, {invalidation_error}"
            ),
            (Err(stop_error), Err(invalidation_error)) => format!(
                "Rejected desired PWM update ({update_error}); {stop_error}; additionally, {invalidation_error}"
            ),
        });
    }
    Ok(())
}

#[tauri::command]
async fn stop_motors(
    state: State<'_, StreamState>,
    stream_generation_decimal: String,
) -> Result<(), String> {
    warn!("ZERO-PWM STOP REQUESTED");
    let requested_generation = parse_stream_generation_decimal(&stream_generation_decimal)
        .map_err(|error| error.to_string())?;
    let mut state_guard = lock_stream_state(state.inner())?;
    require_current_generation(&state_guard, requested_generation)?;
    let stop_result = if let Some(stream) = state_guard.stream.as_mut() {
        stream
            .set_pwm(0, 0)
            .map_err(|error| format!("Failed to prepare stop command: {error}"))
            .and_then(|()| {
                stream
                    .send_command()
                    .map(|_| ())
                    .map_err(|error| format!("Failed to send stop command: {error}"))
            })
    } else {
        warn!("Stop command requested while not connected");
        return Err("Not connected".to_string());
    };

    if let Err(stop_error) = stop_result {
        let invalidation_result = state_guard.invalidate();
        return match invalidation_result {
            Ok(_) => Err(format!(
                "{stop_error}; command stream was discarded and lease expiry remains the software fallback"
            )),
            Err(invalidation_error) => Err(format!(
                "{stop_error}; additionally, {invalidation_error}"
            )),
        };
    }

    info!("Zero-PWM stop command acknowledged");
    Ok(())
}

#[derive(Debug)]
enum OdometryFetchError {
    Request {
        url: String,
        source: reqwest::Error,
    },
    UnexpectedStatus {
        url: String,
        status: reqwest::StatusCode,
    },
    BodyRead {
        url: String,
        source: reqwest::Error,
    },
    BodyTooLarge {
        url: String,
        limit: usize,
        observed_at_least: usize,
    },
    BodyLengthOverflow {
        url: String,
    },
    Decode {
        url: String,
        source: serde_json::Error,
    },
}

impl std::fmt::Display for OdometryFetchError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Request { url, source } => {
                write!(formatter, "odometry request to {url} failed: {source}")
            }
            Self::UnexpectedStatus { url, status } => {
                write!(
                    formatter,
                    "odometry request to {url} returned HTTP {status}"
                )
            }
            Self::BodyRead { url, source } => {
                write!(formatter, "failed to read odometry response from {url}: {source}")
            }
            Self::BodyTooLarge {
                url,
                limit,
                observed_at_least,
            } => write!(
                formatter,
                "odometry response from {url} exceeds the {limit}-byte limit (observed at least {observed_at_least} bytes)"
            ),
            Self::BodyLengthOverflow { url } => write!(
                formatter,
                "odometry response length from {url} overflowed the host size domain"
            ),
            Self::Decode { url, source } => {
                write!(
                    formatter,
                    "odometry response from {url} is invalid: {source}"
                )
            }
        }
    }
}

impl std::error::Error for OdometryFetchError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Request { source, .. } | Self::BodyRead { source, .. } => Some(source),
            Self::Decode { source, .. } => Some(source),
            Self::UnexpectedStatus { .. }
            | Self::BodyTooLarge { .. }
            | Self::BodyLengthOverflow { .. } => None,
        }
    }
}

async fn fetch_odometry(
    client: &reqwest::Client,
    url: String,
) -> Result<Option<RobotOdometryWithServerReceiveAge>, OdometryFetchError> {
    let mut response = client
        .get(&url)
        .timeout(Duration::from_millis(500))
        .send()
        .await
        .map_err(|source| OdometryFetchError::Request {
            url: url.clone(),
            source,
        })?;
    let status = response.status();
    if status == reqwest::StatusCode::SERVICE_UNAVAILABLE {
        return Ok(None);
    }
    if !status.is_success() {
        return Err(OdometryFetchError::UnexpectedStatus { url, status });
    }

    let declared_length = match response.content_length() {
        Some(length) => {
            let length = usize::try_from(length).map_err(|_| OdometryFetchError::BodyTooLarge {
                url: url.clone(),
                limit: MAX_ODOMETRY_RESPONSE_BYTES,
                observed_at_least: MAX_ODOMETRY_RESPONSE_BYTES + 1,
            })?;
            if length > MAX_ODOMETRY_RESPONSE_BYTES {
                return Err(OdometryFetchError::BodyTooLarge {
                    url,
                    limit: MAX_ODOMETRY_RESPONSE_BYTES,
                    observed_at_least: length,
                });
            }
            length
        }
        None => 0,
    };

    let mut body = Vec::with_capacity(declared_length);
    while let Some(chunk) =
        response
            .chunk()
            .await
            .map_err(|source| OdometryFetchError::BodyRead {
                url: url.clone(),
                source,
            })?
    {
        let observed = body
            .len()
            .checked_add(chunk.len())
            .ok_or_else(|| OdometryFetchError::BodyLengthOverflow { url: url.clone() })?;
        if observed > MAX_ODOMETRY_RESPONSE_BYTES {
            return Err(OdometryFetchError::BodyTooLarge {
                url,
                limit: MAX_ODOMETRY_RESPONSE_BYTES,
                observed_at_least: observed,
            });
        }
        body.extend_from_slice(&chunk);
    }

    serde_json::from_slice::<RobotOdometryWithServerReceiveAge>(&body)
        .map(Some)
        .map_err(|source| OdometryFetchError::Decode { url, source })
}

#[tauri::command]
async fn get_odometry(
    state: State<'_, StreamState>,
    stream_generation_decimal: String,
) -> Result<Option<RobotOdometryUpdate>, String> {
    let requested_generation = parse_stream_generation_decimal(&stream_generation_decimal)
        .map_err(|error| error.to_string())?;
    let (client, url) = {
        let state_guard = lock_stream_state(state.inner())?;
        require_current_generation(&state_guard, requested_generation)?;
        if let Some(stream) = state_guard.stream.as_ref() {
            (
                stream.http_client.clone(),
                format!("{}/odometry", stream.base_http_url),
            )
        } else {
            return Err("Not connected".to_string());
        }
    };

    let result = fetch_odometry(&client, url).await;
    {
        let state_guard = lock_stream_state(state.inner())?;
        require_current_generation(&state_guard, requested_generation)?;
        if state_guard.stream.is_none() {
            return Err("Command stream ended while odometry was in flight".to_string());
        }
    }

    match result {
        Ok(Some(observation)) => {
            let odometry = observation.odometry();
            debug!(
                "Fetched odometry: left_estimated_extended_ticks_wrapping_i64={}, right_estimated_extended_ticks_wrapping_i64={}, left_sample_delta_ticks_modulo_i16={}, right_sample_delta_ticks_modulo_i16={}, server_receive_age_ms={}",
                odometry.left_estimated_extended_ticks_wrapping_i64().get(),
                odometry.right_estimated_extended_ticks_wrapping_i64().get(),
                odometry.left_sample_delta_ticks_modulo_i16().get(),
                odometry.right_sample_delta_ticks_modulo_i16().get(),
                observation.server_receive_age_ms(),
            );
            Ok(Some(observation.into()))
        }
        Ok(None) => Ok(None),
        Err(error) => {
            let detail = error_with_sources(&error);
            error!("{detail}");
            Err(detail)
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logger
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .try_init()?;

    info!(
        "Starting Robot Control Client v{}",
        env!("CARGO_PKG_VERSION")
    );

    let stream_state: StreamState = Arc::new(Mutex::new(CommandState::default()));

    tauri::Builder::default()
        .manage(stream_state)
        .invoke_handler(tauri::generate_handler![
            connect,
            disconnect,
            set_motor_pwm,
            stop_motors,
            get_odometry,
        ])
        .run(tauri::generate_context!())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stopped_desired_pwm() -> DesiredPwm {
        DesiredPwm::Stopped(
            LeasedPwmCommand::try_new(0, 0, SERVER_COMMAND_LEASE_MS)
                .expect("test server lease satisfies the shared protocol domain"),
        )
    }

    #[test]
    fn nonzero_desired_pwm_expires_to_zero_at_its_exclusive_deadline() {
        let now = Instant::now();
        let mut desired = stopped_desired_pwm();
        desired
            .update_at(25, -40, now)
            .expect("valid nonzero desired PWM");

        let before_deadline = now
            .checked_add(UI_DESIRED_PWM_LEASE - Duration::from_nanos(1))
            .expect("short test duration fits Instant");
        let (active, expired) = desired.command_at(before_deadline);
        assert!(!expired);
        assert_eq!(active.left_pwm_percent().get(), 25);
        assert_eq!(active.right_pwm_percent().get(), -40);

        let deadline = now
            .checked_add(UI_DESIRED_PWM_LEASE)
            .expect("short test duration fits Instant");
        let (stopped, expired) = desired.command_at(deadline);
        assert!(expired);
        assert!(stopped.is_stop());
        assert!(desired.command().is_stop());

        let (_, expired_again) = desired.command_at(deadline);
        assert!(!expired_again, "expiry transition is reported exactly once");
    }

    #[test]
    fn zero_desired_pwm_has_no_expiring_active_state() {
        let now = Instant::now();
        let mut desired = stopped_desired_pwm();
        desired
            .update_at(0, 0, now)
            .expect("zero PWM is a valid stop");
        let much_later = now
            .checked_add(Duration::from_secs(60))
            .expect("short test duration fits Instant");
        let (stopped, expired) = desired.command_at(much_later);
        assert!(stopped.is_stop());
        assert!(!expired);
    }

    #[test]
    fn stream_generation_parser_accepts_only_canonical_u64_decimal() {
        assert_eq!(
            parse_stream_generation_decimal("0").expect("canonical zero"),
            StreamGeneration(0)
        );
        assert_eq!(
            parse_stream_generation_decimal(&u64::MAX.to_string()).expect("canonical maximum"),
            StreamGeneration(u64::MAX)
        );
        assert!(matches!(
            parse_stream_generation_decimal("01"),
            Err(StreamGenerationParseError::NonCanonical)
        ));
        assert!(parse_stream_generation_decimal("+1").is_err());
        let error = parse_stream_generation_decimal("not-a-number")
            .expect_err("non-integer token must reject");
        assert!(std::error::Error::source(&error).is_some());
    }

    #[test]
    fn superseded_generation_cannot_address_a_replacement_stream() {
        let mut state = CommandState::default();
        let first = state.invalidate().expect("first generation");
        require_current_generation(&state, first)
            .expect("generation is current before replacement");

        let replacement = state.invalidate().expect("replacement generation");
        assert_ne!(first, replacement);
        assert!(require_current_generation(&state, first).is_err());
        require_current_generation(&state, replacement)
            .expect("only the replacement generation remains current");
    }

    #[test]
    fn generation_exhaustion_does_not_partially_invalidate_state() {
        let mut state = CommandState {
            generation: StreamGeneration(u64::MAX),
            stream: Some(CommandStream {
                socket: UdpSocket::bind("127.0.0.1:0").expect("test UDP socket binds"),
                sequence: 0,
                server_addr: "127.0.0.1:8080".to_string(),
                desired_pwm: stopped_desired_pwm(),
                base_http_url: "http://127.0.0.1:3030".to_string(),
                http_client: reqwest::Client::new(),
            }),
        };

        let error = state
            .invalidate()
            .expect_err("the generation domain is exhausted");

        assert_eq!(error, "Command stream generation counter is exhausted");
        assert_eq!(state.generation, StreamGeneration(u64::MAX));
        assert!(state.stream.is_some());
    }

    #[test]
    fn command_socket_bind_address_matches_the_resolved_peer_family() {
        let ipv4_peer = SocketAddr::from(([127, 0, 0, 1], 8080));
        let ipv6_peer = SocketAddr::from(([0, 0, 0, 0, 0, 0, 0, 1], 8080));

        let ipv4_bind = unspecified_bind_address(ipv4_peer);
        assert!(ipv4_bind.is_ipv4());
        assert_eq!(ipv4_bind.port(), 0);
        assert!(ipv4_bind.ip().is_unspecified());

        let ipv6_bind = unspecified_bind_address(ipv6_peer);
        assert!(ipv6_bind.is_ipv6());
        assert_eq!(ipv6_bind.port(), 0);
        assert!(ipv6_bind.ip().is_unspecified());
    }
}
