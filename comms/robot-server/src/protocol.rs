use anyhow::{Context, Result as AnyResult};
use bytes::{Buf, Bytes, BytesMut};
use futures::StreamExt;
use robot_protocol::{
    parse_controller_report, AppliedPwm, CommandSequenceRelation, ControllerError, ControllerEvent,
    ControllerReport, LeasedPwmCommandError, RobotCommand, RobotCommandAcknowledgement,
    RobotCommandAcknowledgementPacket, RobotCommandPacket, RobotOdometry,
    RobotOdometryWithServerReceiveAge, RobotPacketLengthError,
};
#[cfg(test)]
use robot_protocol::{
    ControllerUptimeMsWrapping, EstimatedWrappingEncoderTicks, ModuloEncoderDeltaTicks,
};
use std::convert::Infallible;
use std::fmt::Write as _;
use std::net::SocketAddr;
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::net::UdpSocket;
use tokio::process::Command;
use tokio::sync::{broadcast, RwLock};
use tokio_serial::SerialPortBuilderExt;
use tokio_stream::wrappers::BroadcastStream;
use warp::hyper::Body;
use warp::Filter;

const SERIAL_FORWARD_PERIOD: Duration = Duration::from_millis(20);
const SERIAL_ACTIVE_LEASE_MS: u16 = 50;
const SERIAL_STOP_LEASE_MS: u16 = 1;
const CAMERA_READ_BUFFER_BYTES: usize = 64 * 1_024;
const MAX_JPEG_FRAME_BYTES: usize = 4 * 1_024 * 1_024;

#[derive(Clone, Copy, Debug)]
struct AcceptedRobotCommand {
    command: RobotCommand,
    source: SocketAddr,
    deadline: Instant,
}

#[derive(Clone, Copy, Debug)]
struct ReceivedRobotOdometry {
    odometry: RobotOdometry,
    received_at: Instant,
}

#[derive(Debug)]
enum OdometryAgeError {
    MonotonicClockOrder,
    MillisecondsOutOfRange(std::num::TryFromIntError),
}

impl std::fmt::Display for OdometryAgeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MonotonicClockOrder => formatter.write_str(
                "odometry observation time precedes its server receive time on the monotonic clock",
            ),
            Self::MillisecondsOutOfRange(source) => write!(
                formatter,
                "odometry server-receive age does not fit the u64 millisecond domain: {source}"
            ),
        }
    }
}

impl std::error::Error for OdometryAgeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::MillisecondsOutOfRange(source) => Some(source),
            Self::MonotonicClockOrder => None,
        }
    }
}

impl ReceivedRobotOdometry {
    fn age_at(self, now: Instant) -> Result<Duration, OdometryAgeError> {
        now.checked_duration_since(self.received_at)
            .ok_or(OdometryAgeError::MonotonicClockOrder)
    }

    fn observation_at(
        self,
        now: Instant,
    ) -> Result<RobotOdometryWithServerReceiveAge, OdometryAgeError> {
        let age_ms = u64::try_from(self.age_at(now)?.as_millis())
            .map_err(OdometryAgeError::MillisecondsOutOfRange)?;
        Ok(RobotOdometryWithServerReceiveAge::new(
            self.odometry,
            age_ms,
        ))
    }
}

fn odometry_age_fields(
    received: Option<ReceivedRobotOdometry>,
    now: Instant,
) -> (Option<String>, Option<String>) {
    match received {
        Some(received) => match received.age_at(now) {
            Ok(age) => (Some(age.as_millis().to_string()), None),
            Err(error) => (None, Some(error.to_string())),
        },
        None => (None, None),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ControllerDiagnostic {
    Error(ControllerError),
    Event(ControllerEvent),
}

impl ControllerDiagnostic {
    const fn kind(self) -> &'static str {
        match self {
            Self::Error(_) => "error",
            Self::Event(_) => "event",
        }
    }

    const fn code(self) -> &'static str {
        match self {
            Self::Error(error) => error.code(),
            Self::Event(event) => event.code(),
        }
    }
}

#[derive(Clone, Debug, Default)]
enum VideoStreamStatus {
    #[default]
    NotStarted,
    Initializing,
    Streaming {
        device: &'static str,
    },
    Unavailable {
        detail: String,
    },
}

impl VideoStreamStatus {
    const fn code(&self) -> &'static str {
        match self {
            Self::NotStarted => "not_started",
            Self::Initializing => "initializing",
            Self::Streaming { .. } => "streaming",
            Self::Unavailable { .. } => "unavailable",
        }
    }

    const fn device(&self) -> Option<&'static str> {
        match self {
            Self::Streaming { device } => Some(device),
            Self::NotStarted | Self::Initializing | Self::Unavailable { .. } => None,
        }
    }

    fn detail(&self) -> Option<&str> {
        match self {
            Self::Unavailable { detail } => Some(detail),
            Self::NotStarted | Self::Initializing | Self::Streaming { .. } => None,
        }
    }

    const fn is_streaming(&self) -> bool {
        matches!(self, Self::Streaming { .. })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CommandAcceptanceError {
    LeaseDeadlineOverflow,
    LeaseHeldByAnotherSource {
        active_source: SocketAddr,
    },
    SourceAcquisitionRequiresStop {
        previous_source: SocketAddr,
    },
    SequenceNotNewer {
        previous: u32,
        received: u32,
        relation: CommandSequenceRelation,
    },
}

impl std::fmt::Display for CommandAcceptanceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LeaseDeadlineOverflow => {
                f.write_str("command lease deadline exceeds the monotonic clock domain")
            }
            Self::LeaseHeldByAnotherSource { active_source } => write!(
                f,
                "command source cannot preempt the active lease held by {active_source}"
            ),
            Self::SourceAcquisitionRequiresStop { previous_source } => write!(
                f,
                "a new command source must acquire control with zero PWM after the lease held by {previous_source}"
            ),
            Self::SequenceNotNewer {
                previous,
                received,
                relation,
            } => write!(
                f,
                "command sequence {received} is {} relative to {previous}, not unambiguously newer",
                relation.description()
            ),
        }
    }
}

impl std::error::Error for CommandAcceptanceError {}

#[derive(Debug)]
struct SerialPortDiscoveryError {
    attempts: Vec<(&'static str, tokio_serial::Error)>,
}

impl std::fmt::Display for SerialPortDiscoveryError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("no supported robot-controller serial port could be opened")?;
        for (name, error) in &self.attempts {
            write!(formatter, "; {name}: {error}")?;
        }
        Ok(())
    }
}

impl std::error::Error for SerialPortDiscoveryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.attempts
            .last()
            .map(|(_, source)| source as &(dyn std::error::Error + 'static))
    }
}

#[derive(Default)]
pub struct RobotState {
    accepted_command: Option<AcceptedRobotCommand>,
    last_unsequenced_applied_pwm: Option<AppliedPwm>,
    last_odometry: Option<ReceivedRobotOdometry>,
    last_controller_diagnostic: Option<ControllerDiagnostic>,
    video_tx: Option<broadcast::Sender<Bytes>>,
    video_stream_status: VideoStreamStatus,
}

impl RobotState {
    fn accept_command(
        &mut self,
        command: RobotCommand,
        source: SocketAddr,
        received_at: Instant,
    ) -> std::result::Result<(), CommandAcceptanceError> {
        if let Some(previous) = self.accepted_command {
            if received_at < previous.deadline
                && previous.source != source
                && !command.leased_pwm().is_stop()
            {
                return Err(CommandAcceptanceError::LeaseHeldByAnotherSource {
                    active_source: previous.source,
                });
            }
            if received_at >= previous.deadline
                && previous.source != source
                && !command.leased_pwm().is_stop()
            {
                return Err(CommandAcceptanceError::SourceAcquisitionRequiresStop {
                    previous_source: previous.source,
                });
            }
            let sequence_relation = command.sequence().relation_to(previous.command.sequence());
            if previous.source == source
                && sequence_relation != CommandSequenceRelation::Newer
                && !(received_at >= previous.deadline && command.leased_pwm().is_stop())
            {
                return Err(CommandAcceptanceError::SequenceNotNewer {
                    previous: previous.command.sequence().get(),
                    received: command.sequence().get(),
                    relation: sequence_relation,
                });
            }
        }

        let deadline = received_at
            .checked_add(Duration::from_millis(u64::from(command.lease_ms().get())))
            .ok_or(CommandAcceptanceError::LeaseDeadlineOverflow)?;
        self.accepted_command = Some(AcceptedRobotCommand {
            command,
            source,
            deadline,
        });
        Ok(())
    }

    fn active_command_at(&self, now: Instant) -> Option<AcceptedRobotCommand> {
        self.accepted_command
            .filter(|command| now < command.deadline)
    }

    fn last_accepted_packet(&self) -> Option<RobotCommandPacket> {
        self.accepted_command.map(|value| value.command.into())
    }
}

#[derive(Debug)]
enum CommandDatagramError {
    Decode(RobotPacketLengthError),
    Validation(LeasedPwmCommandError),
}

impl std::fmt::Display for CommandDatagramError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Decode(source) => {
                write!(
                    f,
                    "command datagram does not have the exact legacy layout: {source}"
                )
            }
            Self::Validation(source) => {
                write!(f, "command datagram violates the command domain: {source}")
            }
        }
    }
}

impl std::error::Error for CommandDatagramError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Decode(source) => Some(source),
            Self::Validation(source) => Some(source),
        }
    }
}

fn decode_command_datagram(
    bytes: &[u8],
) -> std::result::Result<RobotCommand, CommandDatagramError> {
    let packet = RobotCommandPacket::try_from_legacy_wire_bytes(bytes)
        .map_err(CommandDatagramError::Decode)?;
    RobotCommand::try_from(packet).map_err(CommandDatagramError::Validation)
}

pub async fn udp_service(state: Arc<RwLock<RobotState>>) -> AnyResult<()> {
    let socket = UdpSocket::bind("0.0.0.0:8080")
        .await
        .context("failed to bind the robot command UDP service to port 8080")?;
    log::info!("UDP service listening on :8080");

    let mut buf = [0_u8; 1024];

    loop {
        let (len, source) = socket
            .recv_from(&mut buf)
            .await
            .context("robot command UDP receive failed")?;
        let command = match decode_command_datagram(&buf[..len]) {
            Ok(command) => command,
            Err(error) => {
                log::warn!("Rejected command datagram from {source}: {error}");
                continue;
            }
        };
        log::debug!(
            "Received command sequence {} from {source}: left_pwm={}%, right_pwm={}%, lease={}ms",
            command.sequence().get(),
            command.left_pwm_percent().get(),
            command.right_pwm_percent().get(),
            command.lease_ms().get(),
        );

        let acknowledgement = {
            let mut state_guard = state.write().await;
            if let Err(error) = state_guard.accept_command(command, source, Instant::now()) {
                log::warn!("Rejected command from {source}: {error}");
                continue;
            }
            RobotCommandAcknowledgement::new(command.sequence())
        };

        let packet = RobotCommandAcknowledgementPacket::from(acknowledgement);
        let data = packet.to_legacy_wire_bytes();
        let sent = socket
            .send_to(&data, source)
            .await
            .with_context(|| format!("failed to send robot command acknowledgement to {source}"))?;
        if sent != data.len() {
            anyhow::bail!(
                "robot command acknowledgement send to {source} reported {sent} bytes instead of the complete {}-byte packet",
                data.len()
            );
        }
    }
}

pub async fn serial_service(state: Arc<RwLock<RobotState>>) -> AnyResult<()> {
    let port_names = [
        "/dev/ttyACM0",
        "/dev/ttyUSB0",
        "/dev/ttyAMA0",
        "/dev/cu.usbmodem1103",
        "/dev/tty.usbmodem1103",
        "/dev/cu.usbmodem",
        "/dev/tty.usbmodem",
        "/dev/cu.usbserial",
        "/dev/tty.usbserial",
    ];
    let mut port = None;
    let mut open_attempts = Vec::with_capacity(port_names.len());

    for name in port_names {
        match tokio_serial::new(name, 115200)
            .timeout(std::time::Duration::from_millis(10))
            .open_native_async()
        {
            Ok(opened) => {
                log::info!("Serial port opened: {name}");
                port = Some(opened);
                break;
            }
            Err(error) => {
                log::debug!("Serial port {name} is unavailable: {error}");
                open_attempts.push((name, error));
            }
        }
    }

    let mut port = port.ok_or(SerialPortDiscoveryError {
        attempts: open_attempts,
    })?;

    let mut serial_buf = [0_u8; 256];
    let mut rx_buffer = Vec::with_capacity(256);
    let mut discard_until_line_end = false;
    let mut serial_packet = String::with_capacity(32);
    let mut last_forwarded_pwm = None;
    let mut forward_interval = tokio::time::interval(SERIAL_FORWARD_PERIOD);
    forward_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    loop {
        tokio::select! {
            _ = forward_interval.tick() => {
                let active = state.read().await.active_command_at(Instant::now());
                let (left_pwm, right_pwm, serial_lease_ms) = match active {
                    Some(accepted) if !accepted.command.leased_pwm().is_stop() => {
                        (
                            accepted.command.left_pwm_percent().get(),
                            accepted.command.right_pwm_percent().get(),
                            SERIAL_ACTIVE_LEASE_MS,
                        )
                    }
                    Some(_) | None => (0, 0, SERIAL_STOP_LEASE_MS),
                };
                let forwarded_pwm = (left_pwm, right_pwm);
                if last_forwarded_pwm != Some(forwarded_pwm) {
                    log::info!(
                        "Forwarding motor PWM command: left={left_pwm}%, right={right_pwm}%, serial_lease={serial_lease_ms}ms"
                    );
                    last_forwarded_pwm = Some(forwarded_pwm);
                }

                serial_packet.clear();
                writeln!(
                    &mut serial_packet,
                    "CMD,{left_pwm},{right_pwm},{serial_lease_ms}"
                )
                .context("failed to format the bounded serial motor command")?;
                port.write_all(serial_packet.as_bytes())
                    .await
                    .context("failed to forward the motor command to the robot controller")?;
            }
            read_result = port.read(&mut serial_buf) => {
                match read_result {
                    Ok(0) => {
                        anyhow::bail!("robot-controller serial port reached EOF");
                    }
                    Ok(n) => {
                        const MAX_SERIAL_BUFFER_BYTES: usize = 4_096;
                        for &byte in &serial_buf[..n] {
                            if discard_until_line_end {
                                if byte == b'\n' {
                                    discard_until_line_end = false;
                                }
                                continue;
                            }

                            if byte != b'\n' {
                                if rx_buffer.len() == MAX_SERIAL_BUFFER_BYTES {
                                    log::error!(
                                        "Discarding robot-controller serial record after it exceeded {MAX_SERIAL_BUFFER_BYTES} bytes"
                                    );
                                    rx_buffer.clear();
                                    discard_until_line_end = true;
                                } else {
                                    rx_buffer.push(byte);
                                }
                                continue;
                            }

                            if rx_buffer.is_empty() || rx_buffer == b"\r" {
                                rx_buffer.clear();
                                continue;
                            }

                            let report = parse_controller_report(&rx_buffer);
                            rx_buffer.clear();
                            match report {
                                Ok(ControllerReport::AppliedPwm(applied_pwm)) => {
                                    state.write().await.last_unsequenced_applied_pwm = Some(applied_pwm);
                                }
                                Ok(ControllerReport::Odometry(odometry)) => {
                                    log::debug!("Robot odometry: {odometry:?}");
                                    state.write().await.last_odometry = Some(ReceivedRobotOdometry {
                                        odometry,
                                        received_at: Instant::now(),
                                    });
                                }
                                Ok(ControllerReport::Error(error)) => {
                                    log::error!("Robot controller reported error: {}", error.code());
                                    state.write().await.last_controller_diagnostic =
                                        Some(ControllerDiagnostic::Error(error));
                                }
                                Ok(ControllerReport::Event(event)) => {
                                    match event {
                                        ControllerEvent::Ready => log::info!(
                                            "Robot controller reported event: {}",
                                            event.code()
                                        ),
                                        ControllerEvent::CommandLeaseExpired => log::warn!(
                                            "Robot controller reported event: {}",
                                            event.code()
                                        ),
                                    }
                                    state.write().await.last_controller_diagnostic =
                                        Some(ControllerDiagnostic::Event(event));
                                }
                                Err(error) => {
                                    log::warn!("Rejected robot-controller serial report: {error}");
                                }
                            }
                        }
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::TimedOut => {}
                    Err(source) => {
                        return Err(source).context("robot-controller serial read failed");
                    }
                }
            }
        }
    }
}

enum JpegBufferOutcome {
    Frame(Bytes),
    NeedMoreData,
    DroppedOversizedFrame,
}

fn retain_possible_jpeg_start_prefix(buffer: &mut BytesMut) {
    let retain_ff = buffer.last() == Some(&0xff);
    buffer.clear();
    if retain_ff {
        buffer.extend_from_slice(&[0xff]);
    }
}

fn take_next_jpeg_frame(buffer: &mut BytesMut) -> JpegBufferOutcome {
    match find_jpeg_start(buffer) {
        Some(0) => {}
        Some(start) => buffer.advance(start),
        None => {
            if buffer.len() > 1 {
                retain_possible_jpeg_start_prefix(buffer);
            }
            return JpegBufferOutcome::NeedMoreData;
        }
    }

    if let Some(relative_end) = find_jpeg_end(&buffer[2..]) {
        let frame_end = 2 + relative_end + 2;
        if frame_end <= MAX_JPEG_FRAME_BYTES {
            return JpegBufferOutcome::Frame(buffer.split_to(frame_end).freeze());
        }
        buffer.advance(2);
        return JpegBufferOutcome::DroppedOversizedFrame;
    }

    if buffer.len() > MAX_JPEG_FRAME_BYTES {
        buffer.advance(2);
        return JpegBufferOutcome::DroppedOversizedFrame;
    }
    JpegBufferOutcome::NeedMoreData
}

async fn forward_camera_frames(
    stdout: tokio::process::ChildStdout,
    video_tx: &broadcast::Sender<Bytes>,
    state: &Arc<RwLock<RobotState>>,
    device: &'static str,
) -> AnyResult<()> {
    use tokio::io::AsyncReadExt;

    let mut reader = tokio::io::BufReader::new(stdout);
    let mut read_buffer = vec![0_u8; CAMERA_READ_BUFFER_BYTES];
    let mut jpeg_buffer = BytesMut::with_capacity(CAMERA_READ_BUFFER_BYTES);
    let mut received_frame = false;

    loop {
        let bytes_read = reader
            .read(&mut read_buffer)
            .await
            .context("failed to read the ffmpeg MJPEG stream")?;
        if bytes_read == 0 {
            return Ok(());
        }
        jpeg_buffer.extend_from_slice(&read_buffer[..bytes_read]);

        loop {
            match take_next_jpeg_frame(&mut jpeg_buffer) {
                JpegBufferOutcome::Frame(frame) => {
                    if !received_frame {
                        received_frame = true;
                        set_video_stream_status(state, VideoStreamStatus::Streaming { device })
                            .await;
                        log::info!("Camera stream active from {device}");
                    }
                    if let Err(unsent) = video_tx.send(frame) {
                        log::trace!(
                            "Discarded a {}-byte camera frame because no HTTP viewer is subscribed",
                            unsent.0.len()
                        );
                    }
                }
                JpegBufferOutcome::NeedMoreData => break,
                JpegBufferOutcome::DroppedOversizedFrame => log::warn!(
                    "Dropped an MJPEG frame that exceeded the {MAX_JPEG_FRAME_BYTES}-byte safety limit"
                ),
            }
        }
    }
}

async fn set_video_stream_status(state: &Arc<RwLock<RobotState>>, status: VideoStreamStatus) {
    state.write().await.video_stream_status = status;
}

async fn camera_service(
    state: Arc<RwLock<RobotState>>,
    video_tx: broadcast::Sender<Bytes>,
) -> AnyResult<()> {
    const DEVICES: [&str; 3] = ["/dev/video0", "/dev/video1", "/dev/video2"];

    set_video_stream_status(&state, VideoStreamStatus::Initializing).await;
    let mut found_device = false;
    let mut last_failure = None;

    for device in DEVICES {
        if !std::path::Path::new(device).exists() {
            continue;
        }
        found_device = true;
        log::info!("Attempting camera stream from {device}");

        let mut command = Command::new("ffmpeg");
        command
            .args([
                "-f",
                "v4l2",
                "-video_size",
                "640x480",
                "-framerate",
                "30",
                "-i",
                device,
                "-f",
                "mjpeg",
                "-q:v",
                "5",
                "-",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true);
        let mut child = match command.spawn() {
            Ok(child) => child,
            Err(error) => {
                log::error!("Failed to start ffmpeg for {device}: {error}");
                last_failure = Some(format!("ffmpeg could not start for {device}: {error}"));
                continue;
            }
        };
        let Some(stdout) = child.stdout.take() else {
            let detail = format!("ffmpeg for {device} did not expose its configured stdout pipe");
            log::error!("{detail}");
            last_failure = Some(detail);
            continue;
        };

        let stream_result = forward_camera_frames(stdout, &video_tx, &state, device).await;
        if stream_result.is_err() {
            if let Err(error) = child.start_kill() {
                log::error!(
                    "Failed to terminate ffmpeg for {device} after a stream error: {error}"
                );
            }
        }
        let exit_result = child.wait().await;
        let detail = match (stream_result, exit_result) {
            (Ok(()), Ok(status)) => format!("ffmpeg for {device} ended with status {status}"),
            (Ok(()), Err(error)) => format!("failed to wait for ffmpeg on {device}: {error}"),
            (Err(stream_error), Ok(status)) => format!(
                "camera stream from {device} failed ({stream_error:#}); ffmpeg ended with status {status}"
            ),
            (Err(stream_error), Err(wait_error)) => format!(
                "camera stream from {device} failed ({stream_error:#}); additionally, waiting for ffmpeg failed: {wait_error}"
            ),
        };
        log::error!("{detail}");
        last_failure = Some(detail);
        set_video_stream_status(&state, VideoStreamStatus::Initializing).await;
    }

    let detail = last_failure.unwrap_or_else(|| {
        if found_device {
            "all discovered camera devices became unavailable".to_string()
        } else {
            "none of /dev/video0, /dev/video1, or /dev/video2 exists".to_string()
        }
    });
    log::warn!("Video streaming unavailable: {detail}");
    set_video_stream_status(&state, VideoStreamStatus::Unavailable { detail }).await;
    Ok(())
}

pub async fn http_service(state: Arc<RwLock<RobotState>>) -> AnyResult<()> {
    let (video_tx, _) = broadcast::channel::<Bytes>(4);
    {
        let mut state_guard = state.write().await;
        state_guard.video_tx = Some(video_tx.clone());
        state_guard.video_stream_status = VideoStreamStatus::Initializing;
    }

    let camera_state = state.clone();
    let state_filter = warp::any().map(move || state.clone());

    let status = warp::path("status")
        .and(warp::path::end())
        .and(state_filter.clone())
        .and_then(
        |state: Arc<RwLock<RobotState>>| async move {
            let state = state.read().await;
            let now = Instant::now();
            let active_command = state.active_command_at(now);
            let (odometry_age_ms_decimal, odometry_age_error) =
                odometry_age_fields(state.last_odometry, now);
            Ok::<_, Infallible>(warp::reply::json(&serde_json::json!({
                "status": "running",
                "command_lease_active": active_command.is_some(),
                "active_command_source": active_command.map(|command| command.source.to_string()),
                "has_unsequenced_applied_pwm_report": state.last_unsequenced_applied_pwm.is_some(),
                "has_odometry": state.last_odometry.is_some(),
                "odometry_server_receive_age_ms_decimal": odometry_age_ms_decimal,
                "odometry_age_error": odometry_age_error,
                "video_stream": {
                    "state": state.video_stream_status.code(),
                    "device": state.video_stream_status.device(),
                    "detail": state.video_stream_status.detail(),
                },
                "last_controller_diagnostic": state.last_controller_diagnostic.map(|diagnostic| serde_json::json!({
                    "kind": diagnostic.kind(),
                    "code": diagnostic.code(),
                })),
            })))
        },
    );

    let video = warp::path("video.mjpeg")
        .and(warp::path::end())
        .and(state_filter.clone())
        .and_then(|state: Arc<RwLock<RobotState>>| async move {
            let (rx, video_status) = {
                let state_guard = state.read().await;
                let rx = if state_guard.video_stream_status.is_streaming() {
                    state_guard.video_tx.as_ref().map(|tx| tx.subscribe())
                } else {
                    None
                };
                (rx, state_guard.video_stream_status.clone())
            };

            if let Some(rx) = rx {
                let body_stream = BroadcastStream::new(rx).map(|result| match result {
                    Ok(frame) => {
                        let mut data = Vec::with_capacity(frame.len() + 96);
                        data.extend_from_slice(b"--frame\r\n");
                        data.extend_from_slice(b"Content-Type: image/jpeg\r\n");
                        data.extend_from_slice(b"Content-Length: ");
                        data.extend_from_slice(frame.len().to_string().as_bytes());
                        data.extend_from_slice(b"\r\n\r\n");
                        data.extend_from_slice(&frame);
                        data.extend_from_slice(b"\r\n");

                        Ok::<_, std::io::Error>(Bytes::from(data))
                    }
                    Err(error) => {
                        log::warn!("Closing a lagged MJPEG viewer stream: {error}");
                        Err(std::io::Error::other(error))
                    }
                });

                let mut response = warp::http::Response::new(Body::wrap_stream(body_stream));
                response.headers_mut().insert(
                    warp::http::header::CONTENT_TYPE,
                    warp::http::HeaderValue::from_static(
                        "multipart/x-mixed-replace; boundary=frame",
                    ),
                );
                response.headers_mut().insert(
                    warp::http::header::CACHE_CONTROL,
                    warp::http::HeaderValue::from_static("no-cache"),
                );

                Ok::<_, std::convert::Infallible>(response)
            } else {
                let detail = video_status
                    .detail()
                    .unwrap_or("camera initialization has not produced a stream");
                let mut response = warp::http::Response::new(Body::from(format!(
                    "Video stream {}: {detail}",
                    video_status.code()
                )));
                *response.status_mut() = warp::http::StatusCode::SERVICE_UNAVAILABLE;
                Ok(response)
            }
        });

    let debug = warp::path("debug")
        .and(warp::path::end())
        .and(state_filter.clone())
        .and_then(
        |state: Arc<RwLock<RobotState>>| async move {
            let s = state.read().await;
            let now = Instant::now();
            let active_command = s.active_command_at(now);
            let (odometry_age_ms_decimal, odometry_age_error) =
                odometry_age_fields(s.last_odometry, now);
            Ok::<_, Infallible>(warp::reply::json(&serde_json::json!({
                "last_accepted_command": s.last_accepted_packet(),
                "command_lease_active": active_command.is_some(),
                "active_command_source": active_command.map(|command| command.source.to_string()),
                "last_unsequenced_applied_pwm": s.last_unsequenced_applied_pwm.map(|applied| serde_json::json!({
                    "left_pwm_percent": applied.left().get(),
                    "right_pwm_percent": applied.right().get(),
                })),
                "last_odometry": s.last_odometry.map(|received| serde_json::json!({
                    "odometry": received.odometry,
                    "server_receive_age_ms_decimal": odometry_age_ms_decimal,
                    "age_error": odometry_age_error,
                })),
                "last_controller_diagnostic": s.last_controller_diagnostic.map(|diagnostic| serde_json::json!({
                    "kind": diagnostic.kind(),
                    "code": diagnostic.code(),
                })),
            })))
        },
    );

    let odometry = warp::path("odometry")
        .and(warp::path::end())
        .and(state_filter.clone())
        .and_then(|state: Arc<RwLock<RobotState>>| async move {
            let received = state.read().await.last_odometry;
            let reply: Box<dyn warp::Reply> = match received {
                Some(received) => match received.observation_at(Instant::now()) {
                    Ok(observation) => Box::new(warp::reply::json(&observation)),
                    Err(error) => {
                        log::error!("Cannot represent odometry server-receive age: {error}");
                        Box::new(warp::reply::with_status(
                            warp::reply::json(&serde_json::json!({
                                "error": "Odometry server-receive age is unavailable"
                            })),
                            warp::http::StatusCode::INTERNAL_SERVER_ERROR,
                        ))
                    }
                },
                None => Box::new(warp::reply::with_status(
                    warp::reply::json(&serde_json::json!({
                        "error": "No odometry data available"
                    })),
                    warp::http::StatusCode::SERVICE_UNAVAILABLE,
                )),
            };
            Ok::<_, Infallible>(reply)
        });

    let routes = status.or(video).or(debug).or(odometry);

    log::info!("HTTP service starting on :3030");
    let http_server = warp::serve(routes).run(([0, 0, 0, 0], 3030));
    let camera = camera_service(camera_state, video_tx);
    tokio::pin!(http_server);
    tokio::pin!(camera);
    tokio::select! {
        camera_result = &mut camera => {
            if let Err(error) = camera_result {
                log::error!("Camera service terminated: {error:#}");
            }
            http_server.as_mut().await;
        }
        () = &mut http_server => {}
    }

    Ok(())
}

// Helper functions for JPEG parsing
fn find_jpeg_start(data: &[u8]) -> Option<usize> {
    data.windows(2)
        .position(|window| window[0] == 0xFF && window[1] == 0xD8)
}

fn find_jpeg_end(data: &[u8]) -> Option<usize> {
    data.windows(2)
        .position(|window| window[0] == 0xFF && window[1] == 0xD9)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bincode::Options;

    fn source(port: u16) -> SocketAddr {
        SocketAddr::from(([127, 0, 0, 1], port))
    }

    fn command(sequence: u32, lease_ms: u16) -> RobotCommand {
        RobotCommand::try_new(-25, 40, lease_ms, sequence).expect("valid command fixture")
    }

    fn stop_command(sequence: u32, lease_ms: u16) -> RobotCommand {
        RobotCommand::try_new(0, 0, lease_ms, sequence).expect("valid stop fixture")
    }

    #[test]
    fn decoder_preserves_legacy_wire_layout_and_rejects_trailing_bytes() {
        #[derive(serde::Serialize)]
        struct LegacyRobotCommand {
            left_speed: i8,
            right_speed: i8,
            timeout_ms: u16,
            sequence: u32,
        }

        let bytes = bincode::serialize(&LegacyRobotCommand {
            left_speed: -25,
            right_speed: 40,
            timeout_ms: 150,
            sequence: 7,
        })
        .expect("legacy packet serialization");
        let decoded = decode_command_datagram(&bytes).expect("legacy packet remains compatible");
        assert_eq!(decoded.left_pwm_percent().get(), -25);
        assert_eq!(decoded.right_pwm_percent().get(), 40);
        assert_eq!(decoded.lease_ms().get(), 150);
        assert_eq!(decoded.sequence().get(), 7);

        let mut with_trailing = bytes;
        with_trailing.push(0);
        assert!(matches!(
            decode_command_datagram(&with_trailing),
            Err(CommandDatagramError::Decode(_))
        ));
    }

    #[test]
    fn acknowledgement_reuses_legacy_eight_byte_layout_without_fake_measurements() {
        #[derive(serde::Serialize)]
        struct LegacyRobotTelemetry {
            left_actual: i8,
            right_actual: i8,
            battery_mv: u16,
            timestamp_ms: u32,
        }

        let bytes = bincode::serialize(&LegacyRobotTelemetry {
            left_actual: 0,
            right_actual: 0,
            battery_mv: 0,
            timestamp_ms: 7,
        })
        .expect("legacy telemetry serialization");
        assert_eq!(bytes.len(), 8);
        let packet = bincode::DefaultOptions::new()
            .with_fixint_encoding()
            .reject_trailing_bytes()
            .deserialize::<RobotCommandAcknowledgementPacket>(&bytes)
            .expect("new acknowledgement preserves the legacy field widths");
        let acknowledgement = RobotCommandAcknowledgement::try_from(packet)
            .expect("reserved legacy fields are all zero");
        assert_eq!(acknowledgement.accepted_sequence().get(), 7);
    }

    #[test]
    fn decoder_rejects_invalid_command_without_clamping_and_preserves_source() {
        let bytes = bincode::serialize(&RobotCommandPacket {
            left_pwm_percent: -101,
            right_pwm_percent: 0,
            lease_ms: 150,
            sequence: 1,
        })
        .expect("raw packet serialization");
        let error = decode_command_datagram(&bytes).expect_err("out-of-domain PWM must reject");

        assert!(matches!(error, CommandDatagramError::Validation(_)));
        let command_error = std::error::Error::source(&error).expect("command-domain source");
        assert!(command_error.to_string().contains("left-wheel"));
        assert!(command_error.source().is_some());
    }

    #[test]
    fn active_lease_rejects_non_newer_sequences_and_nonzero_source_preemption() {
        let now = Instant::now();
        let owner = source(10_001);
        let contender = source(10_002);
        let mut state = RobotState::default();
        state
            .accept_command(command(10, 100), owner, now)
            .expect("first source acquires lease");

        assert!(matches!(
            state.accept_command(command(10, 100), owner, now + Duration::from_millis(1)),
            Err(CommandAcceptanceError::SequenceNotNewer {
                previous: 10,
                received: 10,
                relation: CommandSequenceRelation::Duplicate,
            })
        ));
        assert!(matches!(
            state.accept_command(command(9, 100), owner, now + Duration::from_millis(2)),
            Err(CommandAcceptanceError::SequenceNotNewer {
                previous: 10,
                received: 9,
                relation: CommandSequenceRelation::Older,
            })
        ));
        assert!(matches!(
            state.accept_command(
                command(10_u32.wrapping_add(1_u32 << 31), 100),
                owner,
                now + Duration::from_millis(3),
            ),
            Err(CommandAcceptanceError::SequenceNotNewer {
                previous: 10,
                relation: CommandSequenceRelation::AmbiguousHalfRange,
                ..
            })
        ));
        assert!(matches!(
            state.accept_command(command(1, 100), contender, now + Duration::from_millis(99)),
            Err(CommandAcceptanceError::LeaseHeldByAnotherSource {
                active_source,
            }) if active_source == owner
        ));
        assert_eq!(
            state
                .active_command_at(now + Duration::from_millis(99))
                .expect("lease remains active before deadline")
                .source,
            owner
        );
    }

    #[test]
    fn zero_pwm_from_another_source_preempts_an_active_motion_lease() {
        let now = Instant::now();
        let owner = source(10_101);
        let stopping_source = source(10_102);
        let mut state = RobotState::default();
        state
            .accept_command(command(10, 100), owner, now)
            .expect("first source acquires motion lease");

        state
            .accept_command(
                stop_command(0, 100),
                stopping_source,
                now + Duration::from_millis(1),
            )
            .expect("zero PWM is allowed to preempt motion");
        let accepted = state
            .active_command_at(now + Duration::from_millis(1))
            .expect("preempting stop has a source lease");
        assert_eq!(accepted.source, stopping_source);
        assert!(accepted.command.leased_pwm().is_stop());

        assert!(matches!(
            state.accept_command(command(11, 100), owner, now + Duration::from_millis(2)),
            Err(CommandAcceptanceError::LeaseHeldByAnotherSource {
                active_source,
            }) if active_source == stopping_source
        ));
    }

    #[test]
    fn expired_lease_is_not_forwarded_and_new_source_must_acquire_with_stop() {
        let now = Instant::now();
        let owner = source(11_001);
        let contender = source(11_002);
        let mut state = RobotState::default();
        state
            .accept_command(command(u32::MAX, 100), owner, now)
            .expect("first lease");

        assert!(
            state
                .active_command_at(now + Duration::from_millis(100))
                .is_none(),
            "deadline is exclusive"
        );
        assert!(matches!(
            state.accept_command(command(0, 100), contender, now + Duration::from_millis(100)),
            Err(CommandAcceptanceError::SourceAcquisitionRequiresStop {
                previous_source,
            }) if previous_source == owner
        ));
        state
            .accept_command(
                stop_command(0, 100),
                contender,
                now + Duration::from_millis(100),
            )
            .expect("expired owner permits a zero-PWM acquisition");
        state
            .accept_command(command(1, 100), contender, now + Duration::from_millis(101))
            .expect("new owner advances from its zero-PWM acquisition");
        assert_eq!(
            state
                .active_command_at(now + Duration::from_millis(101))
                .expect("replacement lease")
                .source,
            contender
        );
    }

    #[test]
    fn prior_owner_nonzero_replay_requires_a_new_zero_pwm_acquisition() {
        let now = Instant::now();
        let first = source(11_051);
        let second = source(11_052);
        let mut state = RobotState::default();
        state
            .accept_command(command(10, 100), first, now)
            .expect("first owner");
        state
            .accept_command(
                stop_command(0, 100),
                second,
                now + Duration::from_millis(100),
            )
            .expect("second owner acquires with zero PWM");

        assert!(matches!(
            state.accept_command(command(11, 100), first, now + Duration::from_millis(200)),
            Err(CommandAcceptanceError::SourceAcquisitionRequiresStop {
                previous_source,
            }) if previous_source == second
        ));
    }

    #[test]
    fn expired_lease_does_not_make_an_old_owner_sequence_replayable() {
        let now = Instant::now();
        let owner = source(11_101);
        let mut state = RobotState::default();
        state
            .accept_command(command(10, 100), owner, now)
            .expect("first lease");

        assert!(matches!(
            state.accept_command(command(10, 100), owner, now + Duration::from_millis(100)),
            Err(CommandAcceptanceError::SequenceNotNewer {
                previous: 10,
                received: 10,
                relation: CommandSequenceRelation::Duplicate,
            })
        ));
        state
            .accept_command(command(11, 100), owner, now + Duration::from_millis(100))
            .expect("a newer sequence can renew an expired lease");
    }

    #[test]
    fn expired_owner_can_reset_its_sequence_only_with_a_stop_command() {
        let now = Instant::now();
        let owner = source(11_201);
        let mut state = RobotState::default();
        state
            .accept_command(command(10, 100), owner, now)
            .expect("first lease");

        state
            .accept_command(
                stop_command(0, 100),
                owner,
                now + Duration::from_millis(100),
            )
            .expect("an expired source can begin a new sequence only by stopping");
        state
            .accept_command(command(1, 100), owner, now + Duration::from_millis(101))
            .expect("the reset session advances from the stop sequence");
    }

    #[test]
    fn shared_controller_report_parser_covers_server_input() {
        assert_eq!(
            parse_controller_report(b"PWM,-25,40\r"),
            Ok(ControllerReport::AppliedPwm(
                AppliedPwm::try_new(-25, 40).expect("valid fixture")
            ))
        );
        assert_eq!(
            parse_controller_report(b"ODO,-10,20,-3,4,500"),
            Ok(ControllerReport::Odometry(RobotOdometry::new(
                EstimatedWrappingEncoderTicks::new_wrapping(-10),
                EstimatedWrappingEncoderTicks::new_wrapping(20),
                ModuloEncoderDeltaTicks::new_modulo(-3),
                ModuloEncoderDeltaTicks::new_modulo(4),
                ControllerUptimeMsWrapping::new(500),
            )))
        );
        assert_eq!(
            parse_controller_report(b"ERR,TX_RECORD_DROPPED"),
            Ok(ControllerReport::Error(
                ControllerError::TransmitRecordDropped
            ))
        );
        assert_eq!(
            parse_controller_report(b"EVT,COMMAND_LEASE_EXPIRED"),
            Ok(ControllerReport::Event(
                ControllerEvent::CommandLeaseExpired
            ))
        );
        assert!(matches!(
            parse_controller_report(b"PWM,0,0,extra"),
            Err(robot_protocol::ControllerReportError::TrailingField)
        ));
        let error = parse_controller_report(b"PWM,-101,0")
            .expect_err("invalid applied PWM must reject rather than clamp");
        assert!(matches!(
            error,
            robot_protocol::ControllerReportError::InvalidAppliedPwm(_)
        ));
        assert!(std::error::Error::source(&error).is_some());

        let error = parse_controller_report(b"ODO,not-a-number,0,0,0,0")
            .expect_err("malformed integer must reject the whole report");
        assert!(matches!(
            error,
            robot_protocol::ControllerReportError::InvalidInteger {
                field: "left estimated wrapping extended encoder ticks",
                ..
            }
        ));
        assert!(std::error::Error::source(&error).is_some());
    }

    #[test]
    fn odometry_json_names_expose_estimation_and_wrapping_domains() {
        let odometry = RobotOdometry::new(
            EstimatedWrappingEncoderTicks::new_wrapping(-10),
            EstimatedWrappingEncoderTicks::new_wrapping(20),
            ModuloEncoderDeltaTicks::new_modulo(-3),
            ModuloEncoderDeltaTicks::new_modulo(4),
            ControllerUptimeMsWrapping::new(500),
        );
        let value = serde_json::to_value(odometry).expect("odometry is serializable");
        let object = value.as_object().expect("odometry serializes as an object");
        assert!(object.contains_key("left_estimated_extended_ticks_wrapping_i64"));
        assert!(object.contains_key("right_estimated_extended_ticks_wrapping_i64"));
        assert!(object.contains_key("left_sample_delta_ticks_modulo_i16"));
        assert!(object.contains_key("right_sample_delta_ticks_modulo_i16"));
        assert!(object.contains_key("controller_uptime_ms_wrapping"));
        assert!(!object.contains_key("left_extended_ticks_wrapping_i64"));
        assert!(!object.contains_key("right_extended_ticks_wrapping_i64"));

        let mut with_unknown_field = serde_json::to_value(odometry).expect("serializable fixture");
        with_unknown_field
            .as_object_mut()
            .expect("odometry serializes as an object")
            .insert("unrecognized_revision".to_string(), serde_json::json!(1));
        assert!(
            serde_json::from_value::<RobotOdometry>(with_unknown_field).is_err(),
            "unknown odometry fields must not be silently ignored"
        );

        let observation = RobotOdometryWithServerReceiveAge::new(odometry, 37);
        let encoded = serde_json::to_value(observation).expect("observation is serializable");
        assert_eq!(encoded["server_receive_age_ms"], serde_json::json!(37));
        assert_eq!(
            serde_json::from_value::<RobotOdometryWithServerReceiveAge>(encoded)
                .expect("exact observation schema round-trips"),
            observation
        );

        let now = Instant::now();
        let received = ReceivedRobotOdometry {
            odometry,
            received_at: now,
        };
        assert_eq!(
            received
                .observation_at(now + Duration::from_millis(37))
                .expect("short duration fits the HTTP age domain"),
            observation
        );
        let future_received = ReceivedRobotOdometry {
            odometry,
            received_at: now + Duration::from_millis(1),
        };
        assert!(matches!(
            future_received.observation_at(now),
            Err(OdometryAgeError::MonotonicClockOrder)
        ));
    }

    #[test]
    fn serial_hop_lease_is_bounded_and_outlives_one_forward_period() {
        let lease = robot_protocol::CommandLeaseMs::try_new(SERIAL_ACTIVE_LEASE_MS)
            .expect("serial lease must satisfy the shared command domain");
        assert!(
            Duration::from_millis(u64::from(lease.get())) > SERIAL_FORWARD_PERIOD,
            "one delayed forward may not immediately expire the controller lease"
        );
        assert!(
            robot_protocol::CommandLeaseMs::try_new(SERIAL_STOP_LEASE_MS).is_ok(),
            "stop packets also carry an explicit bounded lease"
        );
    }

    #[test]
    fn jpeg_framing_is_fragment_tolerant_and_discards_unbounded_prefixes() {
        let mut buffer = BytesMut::from(&b"garbage\xff"[..]);
        assert!(matches!(
            take_next_jpeg_frame(&mut buffer),
            JpegBufferOutcome::NeedMoreData
        ));
        assert_eq!(&buffer[..], b"\xff");

        buffer.extend_from_slice(b"\xd8payload\xff\xd9tail");
        let frame = match take_next_jpeg_frame(&mut buffer) {
            JpegBufferOutcome::Frame(frame) => frame,
            JpegBufferOutcome::NeedMoreData => panic!("complete fragmented JPEG was not emitted"),
            JpegBufferOutcome::DroppedOversizedFrame => {
                panic!("small fragmented JPEG was treated as oversized")
            }
        };
        assert_eq!(&frame[..], b"\xff\xd8payload\xff\xd9");
        assert!(matches!(
            take_next_jpeg_frame(&mut buffer),
            JpegBufferOutcome::NeedMoreData
        ));
        assert!(buffer.is_empty());
    }

    #[test]
    fn oversized_jpeg_is_dropped_and_parser_resynchronizes() {
        let mut buffer = BytesMut::with_capacity(MAX_JPEG_FRAME_BYTES + 8);
        buffer.extend_from_slice(b"\xff\xd8");
        buffer.resize(MAX_JPEG_FRAME_BYTES + 1, 0);
        assert!(matches!(
            take_next_jpeg_frame(&mut buffer),
            JpegBufferOutcome::DroppedOversizedFrame
        ));

        buffer.extend_from_slice(b"\xff\xd8ok\xff\xd9");
        let frame = match take_next_jpeg_frame(&mut buffer) {
            JpegBufferOutcome::Frame(frame) => frame,
            JpegBufferOutcome::NeedMoreData => panic!("parser did not find the next JPEG"),
            JpegBufferOutcome::DroppedOversizedFrame => {
                panic!("parser dropped a valid frame after resynchronization")
            }
        };
        assert_eq!(&frame[..], b"\xff\xd8ok\xff\xd9");
    }
}
