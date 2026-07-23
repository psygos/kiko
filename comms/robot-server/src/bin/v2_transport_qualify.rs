//! Motor-inert real-UART qualification for the KRP2 V2 transport.
//!
//! This tool refuses every controller profile that advertises nonzero PWM,
//! never creates a control session, and writes only
//! `TransportDiagnosticProbe` records after exact identity and idle-safe-state
//! admission. Measurements describe this run; they are not performance
//! improvement claims.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use std::num::NonZeroU16;
use std::str::FromStr;
use std::time::Duration;

use clap::Parser;
use robot_protocol::v2::{
    ActuatorConfigFingerprint, ControllerBootId, ControllerCapabilities, ControllerFaults,
    ControllerHello, ControllerUid, Heartbeat, HostElapsedNsToken, Message, NeutralOutput,
    ObservationalOdometry, OutputState, PhysicalStopSemantics, ReadinessFlags, TimerPwm,
    TransportDiagnosticProbe, TransportDiagnosticReport, TransportDiagnosticResultCode,
    TransportDiagnosticRunId, TransportDiagnosticSequence, UartEncodeError, UartRecord,
    UartStreamDecoder, UartStreamError, CANONICAL_CONTROLLER_HELLO_PERIOD_MS,
};
use robot_protocol::ControllerUptimeMsWrapping;
use serde::Serialize;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::sync::mpsc;
use tokio::time::Instant;
use tokio_serial::{DataBits, FlowControl, Parity, SerialPortBuilderExt, StopBits};

const SERIAL_BAUD_BPS: u32 = 115_200;
const UART_BITS_PER_BYTE_8N1: u32 = 10;
const SERIAL_BY_ID_PREFIX: &str = "/dev/serial/by-id/";
const MAX_SERIAL_PATH_BYTES: usize = 512;
const SUPPORTED_BENCHMARK_RATES_HZ: [u16; 4] = [20, 50, 75, 100];
const MIN_DURATION_MS: u64 = 1;
const MAX_DURATION_MS: u64 = 60_000;
const MAX_PROBES: usize = 3_000;
const MAX_IN_FLIGHT: usize = 256;
const WRITER_QUEUE_CAPACITY: usize = 8;
const COMPLETION_QUEUE_CAPACITY: usize = 32;
const MIN_ADMISSION_TIMEOUT_MS: u64 = 1;
const MAX_ADMISSION_TIMEOUT_MS: u64 = 30_000;
const MIN_FINAL_DRAIN_MS: u64 = 100;
const MAX_FINAL_DRAIN_MS: u64 = 10_000;
const MIN_SERIAL_WRITE_TIMEOUT_MS: u64 = 7;
const MAX_SERIAL_WRITE_TIMEOUT_MS: u64 = 100;
const WRITER_JOIN_TIMEOUT: Duration = Duration::from_millis(MAX_SERIAL_WRITE_TIMEOUT_MS);
const MAX_OBSERVED_BYTES: usize = 2 * 1024 * 1024;
const MAX_OBSERVED_RECORDS: usize = 20_000;
const MAX_PENDING_DECODED_MESSAGES: usize = 128;
const MAX_DEFERRED_DIAGNOSTIC_REPORTS: usize = MAX_IN_FLIGHT;
const MAX_ENTROPY_ATTEMPTS: usize = 8;

#[derive(Parser, Debug)]
#[command(
    name = "robot-v2-transport-qualify",
    about = "Measure motor-inert KRP2 UART request/response behavior"
)]
struct Cli {
    /// Exact Linux persistent serial identity. No ttyACM fallback or scan.
    #[arg(long)]
    serial_device: PersistentSerialPath,
    /// Expected 12-byte STM UID as exactly 24 hexadecimal characters.
    #[arg(long)]
    controller_uid_hex: ControllerUidArgument,
    /// Exact nonzero boot ID, decimal or 0x-prefixed hexadecimal.
    #[arg(long, value_parser = parse_boot_id)]
    boot_id: ControllerBootId,
    /// Exact firmware ABI.
    #[arg(long, value_parser = parse_u16)]
    firmware_abi: u16,
    /// Exact firmware build ID, decimal or 0x-prefixed hexadecimal.
    #[arg(long, value_parser = parse_u32)]
    firmware_build_id: u32,
    /// Exact 16-byte actuator profile fingerprint as 32 hexadecimal characters.
    #[arg(long)]
    actuator_config_fingerprint_hex: FingerprintArgument,
    /// Exact known controller capability bits, decimal or 0x-prefixed hexadecimal.
    #[arg(long, value_parser = parse_capabilities)]
    capabilities_bits: ControllerCapabilities,
    /// Exact motor-inert probe rate. 20/50 Hz are baseline measurements;
    /// 75/100 Hz are transport stress only and do not admit motion.
    #[arg(long, default_value = "50")]
    rate_hz: ProbeRateHz,
    /// Scheduled probe-stream duration in milliseconds; it must contain an
    /// integral number of periods at the selected exact rational rate.
    #[arg(long, default_value_t = 10_000, value_parser = parse_duration_ms)]
    duration_ms: u64,
    /// Deadline for exact Hello plus fresh idle-safe Heartbeat admission.
    #[arg(long, default_value_t = 5_000, value_parser = parse_admission_timeout_ms)]
    admission_timeout_ms: u64,
    /// Bounded receive-only drain after the final diagnostic write completes.
    #[arg(long, default_value_t = 1_000, value_parser = parse_final_drain_ms)]
    final_drain_ms: u64,
    /// One deadline covering each complete diagnostic-record write and flush.
    #[arg(long, default_value = "10")]
    serial_write_timeout_ms: SerialWriteTimeout,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PersistentSerialPath(Box<str>);

impl PersistentSerialPath {
    fn as_str(&self) -> &str {
        &self.0
    }
}

impl FromStr for PersistentSerialPath {
    type Err = ArgumentError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.len() > MAX_SERIAL_PATH_BYTES {
            return Err(ArgumentError::SerialPathTooLong {
                actual_bytes: value.len(),
                maximum_bytes: MAX_SERIAL_PATH_BYTES,
            });
        }
        let suffix = value
            .strip_prefix(SERIAL_BY_ID_PREFIX)
            .ok_or(ArgumentError::SerialPathNotPersistent)?;
        if suffix.is_empty() {
            return Err(ArgumentError::SerialPathEmptyIdentity);
        }
        if suffix.contains('/') || matches!(suffix, "." | "..") {
            return Err(ArgumentError::SerialPathNonCanonical);
        }
        if let Some((index, byte)) = value
            .bytes()
            .enumerate()
            .find(|(_, byte)| !byte.is_ascii_graphic())
        {
            return Err(ArgumentError::SerialPathNonGraphic { index, byte });
        }
        Ok(Self(value.into()))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ControllerUidArgument(ControllerUid);

impl FromStr for ControllerUidArgument {
    type Err = ArgumentError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let bytes = parse_fixed_hex::<12>(value, "controller UID")?;
        ControllerUid::try_new(bytes)
            .map(Self)
            .map_err(|_| ArgumentError::ZeroControllerUid)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FingerprintArgument(ActuatorConfigFingerprint);

impl FromStr for FingerprintArgument {
    type Err = ArgumentError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let bytes = parse_fixed_hex::<16>(value, "actuator fingerprint")?;
        ActuatorConfigFingerprint::try_new(bytes)
            .map(Self)
            .map_err(|_| ArgumentError::ZeroFingerprint)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum ArgumentError {
    SerialPathTooLong {
        actual_bytes: usize,
        maximum_bytes: usize,
    },
    SerialPathNotPersistent,
    SerialPathEmptyIdentity,
    SerialPathNonCanonical,
    SerialPathNonGraphic {
        index: usize,
        byte: u8,
    },
    WrongHexLength {
        field: &'static str,
        expected_characters: usize,
        actual_characters: usize,
    },
    InvalidHex {
        field: &'static str,
        index: usize,
    },
    ZeroControllerUid,
    ZeroFingerprint,
    InvalidUnsigned {
        field: &'static str,
    },
    OutsideRange {
        field: &'static str,
        value: u64,
        minimum: u64,
        maximum: u64,
    },
    UnsupportedBenchmarkRate {
        actual_hz: u16,
    },
    ZeroBootId,
    UnknownCapabilityBits {
        bits: u32,
    },
}

impl fmt::Display for ArgumentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid qualification argument: {self:?}")
    }
}

impl std::error::Error for ArgumentError {}

fn parse_fixed_hex<const N: usize>(
    value: &str,
    field: &'static str,
) -> Result<[u8; N], ArgumentError> {
    let expected_characters = N * 2;
    if value.len() != expected_characters {
        return Err(ArgumentError::WrongHexLength {
            field,
            expected_characters,
            actual_characters: value.len(),
        });
    }
    let mut bytes = [0_u8; N];
    for (index, slot) in bytes.iter_mut().enumerate() {
        let offset = index * 2;
        let high = decode_hex(value.as_bytes()[offset]).ok_or(ArgumentError::InvalidHex {
            field,
            index: offset,
        })?;
        let low = decode_hex(value.as_bytes()[offset + 1]).ok_or(ArgumentError::InvalidHex {
            field,
            index: offset + 1,
        })?;
        *slot = (high << 4) | low;
    }
    Ok(bytes)
}

const fn decode_hex(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        b'A'..=b'F' => Some(value - b'A' + 10),
        _ => None,
    }
}

fn parse_integer(value: &str, field: &'static str) -> Result<u64, ArgumentError> {
    if let Some(hex) = value
        .strip_prefix("0x")
        .or_else(|| value.strip_prefix("0X"))
    {
        u64::from_str_radix(hex, 16).map_err(|_| ArgumentError::InvalidUnsigned { field })
    } else {
        value
            .parse::<u64>()
            .map_err(|_| ArgumentError::InvalidUnsigned { field })
    }
}

fn parse_u16(value: &str) -> Result<u16, ArgumentError> {
    let parsed = parse_integer(value, "firmware ABI")?;
    u16::try_from(parsed).map_err(|_| ArgumentError::OutsideRange {
        field: "firmware ABI",
        value: parsed,
        minimum: u64::from(u16::MIN),
        maximum: u64::from(u16::MAX),
    })
}

fn parse_u32(value: &str) -> Result<u32, ArgumentError> {
    let parsed = parse_integer(value, "u32 value")?;
    u32::try_from(parsed).map_err(|_| ArgumentError::OutsideRange {
        field: "u32 value",
        value: parsed,
        minimum: u64::from(u32::MIN),
        maximum: u64::from(u32::MAX),
    })
}

fn parse_boot_id(value: &str) -> Result<ControllerBootId, ArgumentError> {
    let parsed = parse_integer(value, "boot ID")?;
    ControllerBootId::try_new(parsed).map_err(|_| ArgumentError::ZeroBootId)
}

fn parse_capabilities(value: &str) -> Result<ControllerCapabilities, ArgumentError> {
    let parsed = parse_u32(value)?;
    ControllerCapabilities::try_from_bits(parsed)
        .map_err(|_| ArgumentError::UnknownCapabilityBits { bits: parsed })
}

fn parse_bounded_ms(
    value: &str,
    field: &'static str,
    minimum: u64,
    maximum: u64,
) -> Result<u64, ArgumentError> {
    let parsed = parse_integer(value, field)?;
    if !(minimum..=maximum).contains(&parsed) {
        return Err(ArgumentError::OutsideRange {
            field,
            value: parsed,
            minimum,
            maximum,
        });
    }
    Ok(parsed)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ProbeRateHz(NonZeroU16);

impl ProbeRateHz {
    const fn get(self) -> u16 {
        self.0.get()
    }

    fn nominal_period(self) -> Duration {
        Duration::from_nanos(ceiling_div_u64(1_000_000_000, u64::from(self.get())))
    }
}

impl FromStr for ProbeRateHz {
    type Err = ArgumentError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let parsed = parse_integer(value, "rate_hz")?;
        let rate = u16::try_from(parsed).map_err(|_| ArgumentError::OutsideRange {
            field: "rate_hz",
            value: parsed,
            minimum: u64::from(SUPPORTED_BENCHMARK_RATES_HZ[0]),
            maximum: u64::from(SUPPORTED_BENCHMARK_RATES_HZ[3]),
        })?;
        if !SUPPORTED_BENCHMARK_RATES_HZ.contains(&rate) {
            return Err(ArgumentError::UnsupportedBenchmarkRate { actual_hz: rate });
        }
        Ok(Self(
            NonZeroU16::new(rate).expect("supported rates are nonzero"),
        ))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SerialWriteTimeout(Duration);

impl SerialWriteTimeout {
    const fn duration(self) -> Duration {
        self.0
    }

    fn as_millis_u64(self) -> u64 {
        u64::try_from(self.0.as_millis()).expect("parsed serial timeout is within u64")
    }
}

impl FromStr for SerialWriteTimeout {
    type Err = ArgumentError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        parse_bounded_ms(
            value,
            "serial write timeout",
            MIN_SERIAL_WRITE_TIMEOUT_MS,
            MAX_SERIAL_WRITE_TIMEOUT_MS,
        )
        .map(Duration::from_millis)
        .map(Self)
    }
}

fn writer_completion_bound(
    serial_write_timeout: SerialWriteTimeout,
) -> Result<Duration, QualificationError> {
    serial_write_timeout
        .duration()
        .checked_mul(
            u32::try_from(WRITER_QUEUE_CAPACITY + 1)
                .expect("bounded writer queue capacity fits u32"),
        )
        .and_then(|duration| duration.checked_add(WRITER_JOIN_TIMEOUT))
        .ok_or(QualificationError::WriterDeadlineOverflow)
}

fn parse_duration_ms(value: &str) -> Result<u64, ArgumentError> {
    parse_bounded_ms(value, "duration", MIN_DURATION_MS, MAX_DURATION_MS)
}

fn parse_admission_timeout_ms(value: &str) -> Result<u64, ArgumentError> {
    parse_bounded_ms(
        value,
        "admission timeout",
        MIN_ADMISSION_TIMEOUT_MS,
        MAX_ADMISSION_TIMEOUT_MS,
    )
}

fn parse_final_drain_ms(value: &str) -> Result<u64, ArgumentError> {
    parse_bounded_ms(value, "final drain", MIN_FINAL_DRAIN_MS, MAX_FINAL_DRAIN_MS)
}

#[derive(Clone, Copy, Debug)]
struct ExactController {
    uid: ControllerUid,
    boot_id: ControllerBootId,
    firmware_abi: u16,
    firmware_build_id: u32,
    fingerprint: ActuatorConfigFingerprint,
    capabilities: ControllerCapabilities,
}

impl ExactController {
    fn from_cli(cli: &Cli) -> Self {
        Self {
            uid: cli.controller_uid_hex.0,
            boot_id: cli.boot_id,
            firmware_abi: cli.firmware_abi,
            firmware_build_id: cli.firmware_build_id,
            fingerprint: cli.actuator_config_fingerprint_hex.0,
            capabilities: cli.capabilities_bits,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ProbePlan {
    rate: ProbeRateHz,
    nominal_period: Duration,
    duration: Duration,
    final_drain: Duration,
    planned_probes: usize,
}

impl ProbePlan {
    fn parse(cli: &Cli) -> Result<Self, QualificationError> {
        let periods_numerator = cli
            .duration_ms
            .checked_mul(u64::from(cli.rate_hz.get()))
            .ok_or(QualificationError::HostDurationOutsideU64)?;
        if !periods_numerator.is_multiple_of(1_000) {
            return Err(QualificationError::DurationHasFractionalProbeCount {
                duration_ms: cli.duration_ms,
                rate_hz: cli.rate_hz.get(),
            });
        }
        let planned_u64 = periods_numerator / 1_000;
        let planned_probes = usize::try_from(planned_u64)
            .ok()
            .filter(|value| (1..=MAX_PROBES).contains(value))
            .ok_or(QualificationError::ProbeCountOutsideBound {
                actual: planned_u64,
                maximum: MAX_PROBES,
            })?;
        Ok(Self {
            rate: cli.rate_hz,
            nominal_period: cli.rate_hz.nominal_period(),
            duration: Duration::from_millis(cli.duration_ms),
            final_drain: Duration::from_millis(cli.final_drain_ms),
            planned_probes,
        })
    }
}

#[derive(Debug)]
enum QualificationError {
    DurationHasFractionalProbeCount {
        duration_ms: u64,
        rate_hz: u16,
    },
    ProbeCountOutsideBound {
        actual: u64,
        maximum: usize,
    },
    Entropy(getrandom::Error),
    EntropyProducedOnlyZero {
        attempts: usize,
    },
    Open(tokio_serial::Error),
    Exclusive(tokio_serial::Error),
    Read(std::io::Error),
    SerialWrite {
        phase: WriterPhase,
        source: std::io::Error,
    },
    SerialWriteTimeout {
        phase: WriterPhase,
        maximum_ms: u64,
    },
    SerialEof,
    Decode(UartStreamError),
    Encode(UartEncodeError),
    AdmissionTimeout {
        timeout_ms: u64,
        saw_hello: bool,
        saw_idle_heartbeat: bool,
    },
    ByteBudgetExceeded {
        maximum: usize,
    },
    RecordBudgetExceeded {
        maximum: usize,
    },
    PendingDecodeBudgetExceeded {
        maximum: usize,
    },
    IdentityMismatch {
        field: &'static str,
    },
    HelloNotMotorInert {
        detail: &'static str,
    },
    HeartbeatNotIdleSafe {
        detail: &'static str,
    },
    UnexpectedControllerMessage {
        kind: robot_protocol::v2::MessageKind,
    },
    HostDirectionMessageFromController {
        kind: robot_protocol::v2::MessageKind,
    },
    StaleDiagnosticReportBeforeProbe,
    ControllerClockAnomaly {
        stream: &'static str,
        previous_ms: u32,
        current_ms: u32,
    },
    ScheduleInstantOverflow,
    HostDurationOutsideU64,
    SequenceOutsideU32 {
        index: usize,
    },
    WriterQueueClosed,
    WriterCompletionClosed,
    WriterDeadlineOverflow,
    WriterJoin(tokio::task::JoinError),
    WriterJoinTimeout {
        maximum_ms: u64,
    },
    WriterCompletionTimeout {
        outstanding: usize,
        maximum_ms: u64,
    },
    WriterCompletionMissing {
        sequence: u32,
    },
    ReportForUnsentSequence {
        sequence: u32,
    },
    ReportRunMismatch,
    ReportTokenMismatch {
        sequence: u32,
    },
    DiagnosticRecordLengthChanged {
        direction: &'static str,
        expected: usize,
        actual: usize,
    },
    DiagnosticReportNotIdleSafe {
        sequence: u32,
        detail: &'static str,
    },
    DiagnosticDenied {
        sequence: u32,
        result: TransportDiagnosticResultCode,
    },
    DeferredReportBudgetExceeded {
        maximum: usize,
    },
    DeferredReportUnresolved {
        count: usize,
    },
    PeriodicLivenessGapExceeded {
        stream: &'static str,
        observed_gap_ms: u128,
        maximum_gap_ms: u128,
    },
    FinalHeartbeatMissing {
        drain_ms: u64,
    },
    EvidenceFailed,
    Json(serde_json::Error),
}

impl fmt::Display for QualificationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("motor-inert KRP2 transport qualification failed: ")?;
        match self {
            Self::DurationHasFractionalProbeCount {
                duration_ms,
                rate_hz,
            } => write!(
                formatter,
                "duration {duration_ms} ms contains a fractional probe count at {rate_hz} Hz"
            ),
            Self::ProbeCountOutsideBound { actual, maximum } => {
                write!(formatter, "probe count {actual} is outside 1..={maximum}")
            }
            Self::Entropy(source) => write!(formatter, "run-ID entropy failed: {source}"),
            Self::EntropyProducedOnlyZero { attempts } => write!(
                formatter,
                "entropy produced zero run IDs for all {attempts} bounded attempts"
            ),
            Self::Open(source) => write!(formatter, "serial open failed: {source}"),
            Self::Exclusive(source) => {
                write!(formatter, "exclusive serial ownership failed: {source}")
            }
            Self::Read(source) => write!(formatter, "serial read failed: {source}"),
            Self::SerialWrite { phase, source } => {
                write!(formatter, "serial {phase} failed: {source}")
            }
            Self::SerialWriteTimeout { phase, maximum_ms } => {
                write!(
                    formatter,
                    "serial {phase} did not complete within {maximum_ms} ms"
                )
            }
            Self::SerialEof => formatter.write_str("serial stream reached EOF"),
            Self::Decode(source) => write!(formatter, "bounded KRP2 decode failed: {source}"),
            Self::Encode(source) => write!(formatter, "bounded KRP2 encode failed: {source}"),
            Self::AdmissionTimeout {
                timeout_ms,
                saw_hello,
                saw_idle_heartbeat,
            } => write!(
                formatter,
                "admission timed out after {timeout_ms} ms (hello={saw_hello}, idle_safe_heartbeat={saw_idle_heartbeat})"
            ),
            Self::ByteBudgetExceeded { maximum } => {
                write!(formatter, "serial byte budget {maximum} exceeded")
            }
            Self::RecordBudgetExceeded { maximum } => {
                write!(formatter, "decoded-record budget {maximum} exceeded")
            }
            Self::PendingDecodeBudgetExceeded { maximum } => {
                write!(
                    formatter,
                    "pending decoded-message budget {maximum} exceeded"
                )
            }
            Self::IdentityMismatch { field } => {
                write!(formatter, "exact controller identity mismatch in {field}")
            }
            Self::HelloNotMotorInert { detail } => {
                write!(formatter, "ControllerHello is not motor-inert: {detail}")
            }
            Self::HeartbeatNotIdleSafe { detail } => {
                write!(formatter, "Heartbeat is not idle-safe: {detail}")
            }
            Self::UnexpectedControllerMessage { kind } => {
                write!(formatter, "unexpected controller message {kind:?}")
            }
            Self::HostDirectionMessageFromController { kind } => {
                write!(
                    formatter,
                    "controller emitted host-direction message {kind:?}"
                )
            }
            Self::StaleDiagnosticReportBeforeProbe => {
                formatter.write_str("diagnostic report arrived before this run sent any probe")
            }
            Self::ControllerClockAnomaly {
                stream,
                previous_ms,
                current_ms,
            } => write!(
                formatter,
                "{stream} controller clock is non-forward in wrapping half-range ({previous_ms} -> {current_ms})"
            ),
            Self::ScheduleInstantOverflow => formatter.write_str("host schedule instant overflow"),
            Self::HostDurationOutsideU64 => {
                formatter.write_str("host duration exceeds the u64 evidence domain")
            }
            Self::SequenceOutsideU32 { index } => {
                write!(formatter, "probe index {index} exceeds the u32 wire domain")
            }
            Self::WriterQueueClosed => formatter.write_str("writer queue closed unexpectedly"),
            Self::WriterCompletionClosed => {
                formatter.write_str("writer completion queue closed unexpectedly")
            }
            Self::WriterDeadlineOverflow => {
                formatter.write_str("serial writer deadline instant overflow")
            }
            Self::WriterJoin(source) => write!(formatter, "writer task join failed: {source}"),
            Self::WriterJoinTimeout { maximum_ms } => {
                write!(
                    formatter,
                    "writer task did not terminate within {maximum_ms} ms"
                )
            }
            Self::WriterCompletionTimeout {
                outstanding,
                maximum_ms,
            } => write!(
                formatter,
                "{outstanding} dispatched serial writes remained incomplete after {maximum_ms} ms"
            ),
            Self::WriterCompletionMissing { sequence } => {
                write!(
                    formatter,
                    "writer completion missing for sequence {sequence}"
                )
            }
            Self::ReportForUnsentSequence { sequence } => {
                write!(formatter, "report references unsent sequence {sequence}")
            }
            Self::ReportRunMismatch => formatter.write_str("report run ID mismatch"),
            Self::ReportTokenMismatch { sequence } => {
                write!(formatter, "report token mismatch for sequence {sequence}")
            }
            Self::DiagnosticRecordLengthChanged {
                direction,
                expected,
                actual,
            } => write!(
                formatter,
                "{direction} diagnostic record length changed from {expected} to {actual} bytes"
            ),
            Self::DiagnosticReportNotIdleSafe { sequence, detail } => {
                write!(
                    formatter,
                    "diagnostic report {sequence} is not idle-safe: {detail}"
                )
            }
            Self::DiagnosticDenied { sequence, result } => {
                write!(formatter, "probe {sequence} was denied: {result:?}")
            }
            Self::DeferredReportBudgetExceeded { maximum } => write!(
                formatter,
                "deferred diagnostic-report budget {maximum} exceeded"
            ),
            Self::DeferredReportUnresolved { count } => write!(
                formatter,
                "{count} diagnostic reports remained unresolved after all writer completions"
            ),
            Self::PeriodicLivenessGapExceeded {
                stream,
                observed_gap_ms,
                maximum_gap_ms,
            } => write!(
                formatter,
                "{stream} liveness gap {observed_gap_ms} ms exceeds {maximum_gap_ms} ms"
            ),
            Self::FinalHeartbeatMissing { drain_ms } => write!(
                formatter,
                "no fresh idle-safe Heartbeat followed the last write within the {drain_ms} ms drain"
            ),
            Self::EvidenceFailed => {
                formatter.write_str("the emitted measurement report did not pass")
            }
            Self::Json(source) => write!(formatter, "JSON evidence encode failed: {source}"),
        }
    }
}

impl std::error::Error for QualificationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Entropy(source) => Some(source),
            Self::Open(source) | Self::Exclusive(source) => Some(source),
            Self::Read(source) => Some(source),
            Self::SerialWrite { source, .. } => Some(source),
            Self::Decode(source) => Some(source),
            Self::Encode(source) => Some(source),
            Self::WriterJoin(source) => Some(source),
            Self::Json(source) => Some(source),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct TimedMessage {
    message: Message,
    received_at: Instant,
}

struct FramedSerialReader {
    decoder: UartStreamDecoder,
    pending: VecDeque<TimedMessage>,
    buffer: [u8; 512],
    observed_bytes: usize,
    observed_records: usize,
}

impl FramedSerialReader {
    fn new() -> Self {
        Self {
            decoder: UartStreamDecoder::new(),
            pending: VecDeque::new(),
            buffer: [0; 512],
            observed_bytes: 0,
            observed_records: 0,
        }
    }

    async fn next<R>(&mut self, reader: &mut R) -> Result<TimedMessage, QualificationError>
    where
        R: AsyncRead + Unpin,
    {
        loop {
            if let Some(message) = self.pending.pop_front() {
                return Ok(message);
            }
            let count = reader
                .read(&mut self.buffer)
                .await
                .map_err(QualificationError::Read)?;
            if count == 0 {
                return Err(QualificationError::SerialEof);
            }
            self.observed_bytes = self.observed_bytes.checked_add(count).ok_or(
                QualificationError::ByteBudgetExceeded {
                    maximum: MAX_OBSERVED_BYTES,
                },
            )?;
            if self.observed_bytes > MAX_OBSERVED_BYTES {
                return Err(QualificationError::ByteBudgetExceeded {
                    maximum: MAX_OBSERVED_BYTES,
                });
            }
            for &byte in &self.buffer[..count] {
                let Some(decoded) = self.decoder.push(byte) else {
                    continue;
                };
                self.observed_records = self.observed_records.checked_add(1).ok_or(
                    QualificationError::RecordBudgetExceeded {
                        maximum: MAX_OBSERVED_RECORDS,
                    },
                )?;
                if self.observed_records > MAX_OBSERVED_RECORDS {
                    return Err(QualificationError::RecordBudgetExceeded {
                        maximum: MAX_OBSERVED_RECORDS,
                    });
                }
                let message = decoded.map_err(QualificationError::Decode)?;
                if self.pending.len() >= MAX_PENDING_DECODED_MESSAGES {
                    return Err(QualificationError::PendingDecodeBudgetExceeded {
                        maximum: MAX_PENDING_DECODED_MESSAGES,
                    });
                }
                self.pending.push_back(TimedMessage {
                    message,
                    received_at: Instant::now(),
                });
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ControllerClockTracker {
    stream: &'static str,
    previous: Option<u32>,
}

impl ControllerClockTracker {
    const fn new(stream: &'static str) -> Self {
        Self {
            stream,
            previous: None,
        }
    }

    fn observe(&mut self, current_ms: u32) -> Result<(), QualificationError> {
        if let Some(previous_ms) = self.previous {
            let delta = current_ms.wrapping_sub(previous_ms);
            if delta >= 0x8000_0000 {
                return Err(QualificationError::ControllerClockAnomaly {
                    stream: self.stream,
                    previous_ms,
                    current_ms,
                });
            }
        }
        self.previous = Some(current_ms);
        Ok(())
    }
}

struct ControllerClockTrackers {
    heartbeat: ControllerClockTracker,
    odometry_measurement: ControllerClockTracker,
    diagnostic_request_received: ControllerClockTracker,
    diagnostic_response_prepared: ControllerClockTracker,
}

impl ControllerClockTrackers {
    const fn new() -> Self {
        Self {
            heartbeat: ControllerClockTracker::new("Heartbeat uptime"),
            odometry_measurement: ControllerClockTracker::new(
                "ObservationalOdometry measurement uptime",
            ),
            diagnostic_request_received: ControllerClockTracker::new("diagnostic request-received"),
            diagnostic_response_prepared: ControllerClockTracker::new(
                "diagnostic response-prepared",
            ),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Admission {
    hello: ControllerHello,
    heartbeat: Heartbeat,
    hello_received_at: Instant,
    heartbeat_received_at: Instant,
}

async fn admit_controller<R>(
    reader: &mut R,
    framed: &mut FramedSerialReader,
    exact: ExactController,
    timeout_ms: u64,
    clocks: &mut ControllerClockTrackers,
) -> Result<Admission, QualificationError>
where
    R: AsyncRead + Unpin,
{
    let deadline = Instant::now()
        .checked_add(Duration::from_millis(timeout_ms))
        .ok_or(QualificationError::ScheduleInstantOverflow)?;
    let mut hello = None;
    let mut heartbeat = None;
    let mut hello_received_at = None;
    let mut heartbeat_received_at = None;
    let mut saw_idle_heartbeat = false;
    loop {
        if let (
            Some(hello),
            Some(admitted_heartbeat),
            Some(hello_received_at),
            Some(admitted_heartbeat_received_at),
        ) = (hello, heartbeat, hello_received_at, heartbeat_received_at)
        {
            if admission_heartbeat_is_fresh(hello, admitted_heartbeat_received_at, Instant::now()) {
                return Ok(Admission {
                    hello,
                    heartbeat: admitted_heartbeat,
                    hello_received_at,
                    heartbeat_received_at: admitted_heartbeat_received_at,
                });
            }
            heartbeat = None;
            heartbeat_received_at = None;
        }
        let timed = tokio::time::timeout_at(deadline, framed.next(reader))
            .await
            .map_err(|_| QualificationError::AdmissionTimeout {
                timeout_ms,
                saw_hello: hello.is_some(),
                saw_idle_heartbeat,
            })??;
        match timed.message {
            Message::ControllerHello(value) => {
                validate_hello(exact, value)?;
                hello = Some(value);
                hello_received_at = Some(timed.received_at);
            }
            Message::Heartbeat(value) => {
                validate_idle_heartbeat(exact, value)?;
                clocks.heartbeat.observe(value.controller_uptime.get())?;
                saw_idle_heartbeat = true;
                heartbeat = Some(value);
                heartbeat_received_at = Some(timed.received_at);
            }
            Message::ObservationalOdometry(value) => {
                validate_idle_odometry(exact, value)?;
                clocks
                    .odometry_measurement
                    .observe(value.controller_uptime.get())?;
            }
            Message::TransportDiagnosticReport(_) => {
                return Err(QualificationError::StaleDiagnosticReportBeforeProbe);
            }
            Message::ControllerReady(value) => {
                validate_controller_identity(exact, value.controller_uid, value.boot_id)?;
                return Err(QualificationError::UnexpectedControllerMessage {
                    kind: Message::ControllerReady(value).kind(),
                });
            }
            Message::AppliedResult(value) => {
                validate_controller_identity(exact, value.controller_uid, value.boot_id)?;
                return Err(QualificationError::UnexpectedControllerMessage {
                    kind: Message::AppliedResult(value).kind(),
                });
            }
            value @ (Message::HostStopResult(_)
            | Message::AcquireResult(_)
            | Message::HostCommandResult(_)
            | Message::StatusReport(_)) => {
                return Err(QualificationError::UnexpectedControllerMessage { kind: value.kind() });
            }
            value @ (Message::AcquireControl(_)
            | Message::HostCommand(_)
            | Message::HostStop(_)
            | Message::StatusQuery(_)
            | Message::BeginSession(_)
            | Message::ApplyPwm(_)
            | Message::ForceStop(_)
            | Message::TransportDiagnosticProbe(_)) => {
                return Err(QualificationError::HostDirectionMessageFromController {
                    kind: value.kind(),
                });
            }
        }
    }
}

fn admission_heartbeat_is_fresh(
    hello: ControllerHello,
    received_at: Instant,
    now: Instant,
) -> bool {
    let maximum_age = Duration::from_millis(u64::from(hello.watchdog_nominal_period.get()));
    now.checked_duration_since(received_at)
        .is_some_and(|age| age <= maximum_age)
}

fn validate_controller_identity(
    exact: ExactController,
    actual_uid: ControllerUid,
    actual_boot_id: ControllerBootId,
) -> Result<(), QualificationError> {
    if actual_uid != exact.uid {
        return Err(QualificationError::IdentityMismatch {
            field: "controller_uid",
        });
    }
    if actual_boot_id != exact.boot_id {
        return Err(QualificationError::IdentityMismatch { field: "boot_id" });
    }
    Ok(())
}

fn validate_hello(
    exact: ExactController,
    hello: ControllerHello,
) -> Result<(), QualificationError> {
    validate_controller_identity(exact, hello.controller_uid, hello.boot_id)?;
    for (matches, field) in [
        (hello.firmware_abi == exact.firmware_abi, "firmware_abi"),
        (
            hello.firmware_build_id == exact.firmware_build_id,
            "firmware_build_id",
        ),
        (
            hello.actuator_config_fingerprint == exact.fingerprint,
            "actuator_config_fingerprint",
        ),
        (
            hello.capabilities == exact.capabilities,
            "capabilities_bits",
        ),
    ] {
        if !matches {
            return Err(QualificationError::IdentityMismatch { field });
        }
    }
    if !hello
        .capabilities
        .supports_motor_inert_transport_diagnostics()
    {
        return Err(QualificationError::HelloNotMotorInert {
            detail: "transport-diagnostic capability bit is absent",
        });
    }
    if hello.max_abs_pwm_percent.get() != 0 || hello.max_abs_pwm_percent.grants_motion_authority() {
        return Err(QualificationError::HelloNotMotorInert {
            detail: "maximum PWM is not exactly zero",
        });
    }
    if !hello.output_state.is_safe() {
        return Err(QualificationError::HelloNotMotorInert {
            detail: "Hello output state is unsafe",
        });
    }
    Ok(())
}

fn validate_idle_heartbeat(
    exact: ExactController,
    heartbeat: Heartbeat,
) -> Result<(), QualificationError> {
    validate_controller_identity(exact, heartbeat.controller_uid, heartbeat.boot_id)?;
    if heartbeat.control_epoch.is_some() || heartbeat.last_sequence.is_some() {
        return Err(QualificationError::HeartbeatNotIdleSafe {
            detail: "heartbeat carries a control epoch or command sequence",
        });
    }
    if !heartbeat.timer_pwm.is_zero() || !heartbeat.output_state.is_safe() {
        return Err(QualificationError::HeartbeatNotIdleSafe {
            detail: "heartbeat does not report safe zero timer output",
        });
    }
    if !heartbeat.faults.is_clear() {
        return Err(QualificationError::HeartbeatNotIdleSafe {
            detail: "heartbeat reports controller faults",
        });
    }
    let forbidden = ReadinessFlags::SESSION_ESTABLISHED | ReadinessFlags::DEADLINE_ARMED;
    if heartbeat.readiness.bits() & forbidden != 0 {
        return Err(QualificationError::HeartbeatNotIdleSafe {
            detail: "heartbeat reports a session or armed deadline",
        });
    }
    if heartbeat.readiness.bits() & ReadinessFlags::WATCHDOG_RUNNING == 0 {
        return Err(QualificationError::HeartbeatNotIdleSafe {
            detail: "heartbeat does not report the independent watchdog running",
        });
    }
    if heartbeat.expires_at.get() != heartbeat.controller_uptime.get() {
        return Err(QualificationError::HeartbeatNotIdleSafe {
            detail: "idle heartbeat deadline is not equal to current controller uptime",
        });
    }
    Ok(())
}

fn validate_idle_odometry(
    exact: ExactController,
    odometry: ObservationalOdometry,
) -> Result<(), QualificationError> {
    validate_controller_identity(exact, odometry.controller_uid, odometry.boot_id)?;
    if odometry.control_epoch.is_some() {
        return Err(QualificationError::HeartbeatNotIdleSafe {
            detail: "observational odometry unexpectedly carries a control epoch",
        });
    }
    Ok(())
}

fn validate_idle_diagnostic_report(
    report: TransportDiagnosticReport,
) -> Result<(), QualificationError> {
    let sequence = report.sequence.get();
    if !report.output_state.is_safe() || !report.timer_pwm.is_zero() {
        return Err(QualificationError::DiagnosticReportNotIdleSafe {
            sequence,
            detail: "output state or timer PWM is nonzero",
        });
    }
    if !report.faults.is_clear() {
        return Err(QualificationError::DiagnosticReportNotIdleSafe {
            sequence,
            detail: "controller fault bits are nonzero",
        });
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct PeriodicLiveness {
    stream: &'static str,
    maximum_gap: Duration,
    count: usize,
    previous_at: Instant,
    maximum_observed_gap: Duration,
}

impl PeriodicLiveness {
    fn admitted(stream: &'static str, maximum_gap: Duration, received_at: Instant) -> Self {
        Self {
            stream,
            maximum_gap,
            count: 1,
            previous_at: received_at,
            maximum_observed_gap: Duration::ZERO,
        }
    }

    fn observe(&mut self, received_at: Instant) -> Result<(), QualificationError> {
        let gap = received_at
            .checked_duration_since(self.previous_at)
            .ok_or(QualificationError::HostDurationOutsideU64)?;
        self.observe_gap(gap)?;
        self.previous_at = received_at;
        self.count = self
            .count
            .checked_add(1)
            .ok_or(QualificationError::HostDurationOutsideU64)?;
        Ok(())
    }

    fn observe_trailing_gap(&mut self, finished_at: Instant) -> Result<(), QualificationError> {
        let gap = finished_at
            .checked_duration_since(self.previous_at)
            .ok_or(QualificationError::HostDurationOutsideU64)?;
        self.observe_gap(gap)
    }

    fn observe_gap(&mut self, gap: Duration) -> Result<(), QualificationError> {
        self.maximum_observed_gap = self.maximum_observed_gap.max(gap);
        if gap > self.maximum_gap {
            return Err(QualificationError::PeriodicLivenessGapExceeded {
                stream: self.stream,
                observed_gap_ms: gap.as_millis(),
                maximum_gap_ms: self.maximum_gap.as_millis(),
            });
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
struct RuntimeLiveness {
    hello: PeriodicLiveness,
    heartbeat: PeriodicLiveness,
}

impl RuntimeLiveness {
    fn from_admission(admission: Admission) -> Result<Self, QualificationError> {
        let advertised_watchdog =
            Duration::from_millis(u64::from(admission.hello.watchdog_nominal_period.get()));
        let canonical_hello_period =
            Duration::from_millis(u64::from(CANONICAL_CONTROLLER_HELLO_PERIOD_MS));
        let hello_maximum_gap = canonical_hello_period
            .checked_mul(2)
            .ok_or(QualificationError::ScheduleInstantOverflow)?;
        Ok(Self {
            hello: PeriodicLiveness::admitted(
                "ControllerHello",
                hello_maximum_gap,
                admission.hello_received_at,
            ),
            heartbeat: PeriodicLiveness::admitted(
                "Heartbeat",
                advertised_watchdog,
                admission.heartbeat_received_at,
            ),
        })
    }

    fn finish(&mut self, finished_at: Instant) -> Result<(), QualificationError> {
        self.hello.observe_trailing_gap(finished_at)?;
        self.heartbeat.observe_trailing_gap(finished_at)
    }
}

#[derive(Clone, Copy, Debug)]
struct WriteJob {
    sequence: TransportDiagnosticSequence,
    scheduled_at: Instant,
    dispatched_at: Instant,
    record: UartRecord,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WriterPhase {
    Write,
    Flush,
}

impl fmt::Display for WriterPhase {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Write => formatter.write_str("write"),
            Self::Flush => formatter.write_str("flush"),
        }
    }
}

#[derive(Debug)]
enum WriterFailure {
    Io {
        phase: WriterPhase,
        source: std::io::Error,
    },
    Timeout {
        phase: WriterPhase,
        maximum: Duration,
    },
    DeadlineOverflow,
}

impl WriterFailure {
    fn into_qualification(self) -> QualificationError {
        match self {
            Self::Io { phase, source } => QualificationError::SerialWrite { phase, source },
            Self::Timeout { phase, maximum } => QualificationError::SerialWriteTimeout {
                phase,
                maximum_ms: u64::try_from(maximum.as_millis()).unwrap_or(u64::MAX),
            },
            Self::DeadlineOverflow => QualificationError::WriterDeadlineOverflow,
        }
    }
}

#[derive(Debug)]
struct WriteCompletion {
    sequence: TransportDiagnosticSequence,
    scheduled_at: Instant,
    dispatched_at: Instant,
    write_started_at: Instant,
    write_completed_at: Instant,
    result: Result<(), WriterFailure>,
    encoded_bytes: usize,
}

async fn run_writer<W>(
    mut writer: W,
    mut jobs: mpsc::Receiver<WriteJob>,
    completions: mpsc::Sender<WriteCompletion>,
    maximum_write_duration: SerialWriteTimeout,
) where
    W: AsyncWrite + Unpin,
{
    while let Some(job) = jobs.recv().await {
        let write_started_at = Instant::now();
        let result =
            write_and_flush_within(&mut writer, job.record.as_bytes(), maximum_write_duration)
                .await;
        let write_completed_at = Instant::now();
        let failed = result.is_err();
        if completions
            .send(WriteCompletion {
                sequence: job.sequence,
                scheduled_at: job.scheduled_at,
                dispatched_at: job.dispatched_at,
                write_started_at,
                write_completed_at,
                result,
                encoded_bytes: job.record.len(),
            })
            .await
            .is_err()
        {
            return;
        }
        if failed {
            return;
        }
    }
}

async fn write_and_flush_within<W>(
    writer: &mut W,
    bytes: &[u8],
    maximum: SerialWriteTimeout,
) -> Result<(), WriterFailure>
where
    W: AsyncWrite + Unpin,
{
    let maximum_duration = maximum.duration();
    let deadline = Instant::now()
        .checked_add(maximum_duration)
        .ok_or(WriterFailure::DeadlineOverflow)?;
    match tokio::time::timeout_at(deadline, writer.write_all(bytes)).await {
        Ok(Ok(())) => {}
        Ok(Err(source)) => {
            return Err(WriterFailure::Io {
                phase: WriterPhase::Write,
                source,
            });
        }
        Err(_) => {
            return Err(WriterFailure::Timeout {
                phase: WriterPhase::Write,
                maximum: maximum_duration,
            });
        }
    }
    match tokio::time::timeout_at(deadline, writer.flush()).await {
        Ok(Ok(())) => Ok(()),
        Ok(Err(source)) => Err(WriterFailure::Io {
            phase: WriterPhase::Flush,
            source,
        }),
        Err(_) => Err(WriterFailure::Timeout {
            phase: WriterPhase::Flush,
            maximum: maximum_duration,
        }),
    }
}

struct AbortOnDrop<T>(Option<tokio::task::JoinHandle<T>>);

impl<T> AbortOnDrop<T> {
    fn new(handle: tokio::task::JoinHandle<T>) -> Self {
        Self(Some(handle))
    }

    async fn join_within(mut self, maximum: Duration) -> Result<T, QualificationError> {
        let handle = self
            .0
            .as_mut()
            .expect("writer join handle is consumed exactly once");
        match tokio::time::timeout(maximum, handle).await {
            Ok(result) => {
                self.0.take();
                result.map_err(QualificationError::WriterJoin)
            }
            Err(_) => Err(QualificationError::WriterJoinTimeout {
                maximum_ms: u64::try_from(maximum.as_millis()).unwrap_or(u64::MAX),
            }),
        }
    }
}

impl<T> Drop for AbortOnDrop<T> {
    fn drop(&mut self) {
        if let Some(handle) = self.0.take() {
            handle.abort();
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct PendingProbe {
    token: HostElapsedNsToken,
    scheduled_at: Instant,
    dispatched_at: Instant,
    write_started_at: Option<Instant>,
    write_completed_at: Option<Instant>,
    reported: bool,
}

#[derive(Clone, Copy, Debug)]
struct DeferredReportTiming {
    sequence: u32,
    received_at: Instant,
}

struct QualificationTracker {
    run_id: TransportDiagnosticRunId,
    nominal_period: Duration,
    pending: BTreeMap<u32, PendingProbe>,
    received_sequences: BTreeSet<u32>,
    highest_unique_sequence: Option<u32>,
    duplicate_reports: usize,
    reordered_reports: usize,
    scheduler_skipped_periods: usize,
    in_flight_limit_skips: usize,
    writer_queue_skips: usize,
    late_by_at_least_one_period: usize,
    maximum_in_flight: usize,
    sent_wire_bytes: usize,
    diagnostic_report_wire_bytes: usize,
    diagnostic_probe_record_bytes: Option<usize>,
    diagnostic_report_record_bytes: Option<usize>,
    dispatch_lateness_ns: Vec<u64>,
    write_start_lateness_ns: Vec<u64>,
    inter_write_start_jitter_ns: Vec<i64>,
    rtt_write_start_to_decode_ns: Vec<u64>,
    response_after_write_completion_ns: Vec<u64>,
    controller_service_ms: Vec<u64>,
    rx_queue_depth_bytes: Vec<u64>,
    tx_queue_depth_bytes: Vec<u64>,
    previous_write_started_at: Option<Instant>,
    previous_scheduled_at: Option<Instant>,
    last_write_completed_at: Option<Instant>,
}

impl QualificationTracker {
    fn new(run_id: TransportDiagnosticRunId, nominal_period: Duration) -> Self {
        Self {
            run_id,
            nominal_period,
            pending: BTreeMap::new(),
            received_sequences: BTreeSet::new(),
            highest_unique_sequence: None,
            duplicate_reports: 0,
            reordered_reports: 0,
            scheduler_skipped_periods: 0,
            in_flight_limit_skips: 0,
            writer_queue_skips: 0,
            late_by_at_least_one_period: 0,
            maximum_in_flight: 0,
            sent_wire_bytes: 0,
            diagnostic_report_wire_bytes: 0,
            diagnostic_probe_record_bytes: None,
            diagnostic_report_record_bytes: None,
            dispatch_lateness_ns: Vec::with_capacity(MAX_PROBES),
            write_start_lateness_ns: Vec::with_capacity(MAX_PROBES),
            inter_write_start_jitter_ns: Vec::with_capacity(MAX_PROBES.saturating_sub(1)),
            rtt_write_start_to_decode_ns: Vec::with_capacity(MAX_PROBES),
            response_after_write_completion_ns: Vec::with_capacity(MAX_PROBES),
            controller_service_ms: Vec::with_capacity(MAX_PROBES),
            rx_queue_depth_bytes: Vec::with_capacity(MAX_PROBES),
            tx_queue_depth_bytes: Vec::with_capacity(MAX_PROBES),
            previous_write_started_at: None,
            previous_scheduled_at: None,
            last_write_completed_at: None,
        }
    }

    fn outstanding(&self) -> usize {
        self.pending
            .values()
            .filter(|probe| !probe.reported)
            .count()
    }

    fn insert_dispatched(
        &mut self,
        sequence: TransportDiagnosticSequence,
        token: HostElapsedNsToken,
        scheduled_at: Instant,
        dispatched_at: Instant,
        encoded_bytes: usize,
    ) -> Result<(), QualificationError> {
        let dispatch_lateness = dispatched_at
            .checked_duration_since(scheduled_at)
            .ok_or(QualificationError::HostDurationOutsideU64)?;
        self.dispatch_lateness_ns
            .push(duration_ns_u64(dispatch_lateness)?);
        let replaced = self.pending.insert(
            sequence.get(),
            PendingProbe {
                token,
                scheduled_at,
                dispatched_at,
                write_started_at: None,
                write_completed_at: None,
                reported: false,
            },
        );
        debug_assert!(replaced.is_none(), "scheduled sequences are unique");
        record_exact_length(
            &mut self.diagnostic_probe_record_bytes,
            "host-to-controller",
            encoded_bytes,
        )?;
        self.maximum_in_flight = self.maximum_in_flight.max(self.outstanding());
        Ok(())
    }

    fn observe_write_completion(
        &mut self,
        completion: WriteCompletion,
    ) -> Result<(), QualificationError> {
        completion
            .result
            .map_err(WriterFailure::into_qualification)?;
        let sequence = completion.sequence.get();
        let pending = self
            .pending
            .get_mut(&sequence)
            .ok_or(QualificationError::WriterCompletionMissing { sequence })?;
        debug_assert_eq!(pending.scheduled_at, completion.scheduled_at);
        debug_assert_eq!(pending.dispatched_at, completion.dispatched_at);
        let write_lateness = completion
            .write_started_at
            .checked_duration_since(completion.scheduled_at)
            .ok_or(QualificationError::HostDurationOutsideU64)?;
        if write_lateness >= self.nominal_period {
            self.late_by_at_least_one_period = self
                .late_by_at_least_one_period
                .checked_add(1)
                .ok_or(QualificationError::HostDurationOutsideU64)?;
        }
        self.write_start_lateness_ns
            .push(duration_ns_u64(write_lateness)?);
        if let (Some(previous_write), Some(previous_scheduled)) =
            (self.previous_write_started_at, self.previous_scheduled_at)
        {
            let interval = completion
                .write_started_at
                .checked_duration_since(previous_write)
                .ok_or(QualificationError::HostDurationOutsideU64)?;
            let scheduled_interval = completion
                .scheduled_at
                .checked_duration_since(previous_scheduled)
                .ok_or(QualificationError::HostDurationOutsideU64)?;
            let interval_ns = i128::try_from(interval.as_nanos())
                .map_err(|_| QualificationError::HostDurationOutsideU64)?;
            let period_ns = i128::try_from(scheduled_interval.as_nanos())
                .map_err(|_| QualificationError::HostDurationOutsideU64)?;
            self.inter_write_start_jitter_ns.push(
                i64::try_from(interval_ns - period_ns)
                    .map_err(|_| QualificationError::HostDurationOutsideU64)?,
            );
        }
        self.previous_write_started_at = Some(completion.write_started_at);
        self.previous_scheduled_at = Some(completion.scheduled_at);
        self.last_write_completed_at = Some(completion.write_completed_at);
        self.sent_wire_bytes = self
            .sent_wire_bytes
            .checked_add(completion.encoded_bytes)
            .ok_or(QualificationError::HostDurationOutsideU64)?;
        record_exact_length(
            &mut self.diagnostic_probe_record_bytes,
            "host-to-controller",
            completion.encoded_bytes,
        )?;
        pending.write_started_at = Some(completion.write_started_at);
        pending.write_completed_at = Some(completion.write_completed_at);
        Ok(())
    }

    fn observe_report_decode(
        &mut self,
        exact: ExactController,
        report: TransportDiagnosticReport,
        received_at: Instant,
        clocks: &mut ControllerClockTrackers,
    ) -> Result<Option<DeferredReportTiming>, QualificationError> {
        validate_controller_identity(exact, report.controller_uid, report.boot_id)?;
        validate_idle_diagnostic_report(report)?;
        if report.run_id != self.run_id {
            return Err(QualificationError::ReportRunMismatch);
        }
        let sequence = report.sequence.get();
        let Some(pending) = self.pending.get_mut(&sequence) else {
            return Err(QualificationError::ReportForUnsentSequence { sequence });
        };
        if report.host_elapsed_ns_token != pending.token {
            return Err(QualificationError::ReportTokenMismatch { sequence });
        }
        if report.result != TransportDiagnosticResultCode::EchoedMotorInert {
            return Err(QualificationError::DiagnosticDenied {
                sequence,
                result: report.result,
            });
        }
        let service_ms = report
            .response_prepared_at
            .wrapping_elapsed_since(report.request_received_at);
        if service_ms >= 0x8000_0000 {
            return Err(QualificationError::ControllerClockAnomaly {
                stream: "diagnostic request-to-response service",
                previous_ms: report.request_received_at.get(),
                current_ms: report.response_prepared_at.get(),
            });
        }
        let report_record_bytes = UartRecord::encode(Message::TransportDiagnosticReport(report))
            .map_err(QualificationError::Encode)?
            .len();
        record_exact_length(
            &mut self.diagnostic_report_record_bytes,
            "controller-to-host",
            report_record_bytes,
        )?;
        if pending.reported {
            self.duplicate_reports = self
                .duplicate_reports
                .checked_add(1)
                .ok_or(QualificationError::HostDurationOutsideU64)?;
            return Ok(None);
        }
        clocks
            .diagnostic_request_received
            .observe(report.request_received_at.get())?;
        clocks
            .diagnostic_response_prepared
            .observe(report.response_prepared_at.get())?;
        if let Some(previous_highest) = self.highest_unique_sequence {
            if sequence < previous_highest {
                self.reordered_reports = self
                    .reordered_reports
                    .checked_add(1)
                    .ok_or(QualificationError::HostDurationOutsideU64)?;
            }
        }
        self.highest_unique_sequence = Some(
            self.highest_unique_sequence
                .map_or(sequence, |value| value.max(sequence)),
        );
        self.controller_service_ms.push(u64::from(service_ms));
        self.rx_queue_depth_bytes
            .push(u64::from(report.rx_queue_depth_bytes));
        self.tx_queue_depth_bytes
            .push(u64::from(report.tx_queue_depth_bytes));
        self.diagnostic_report_wire_bytes = self
            .diagnostic_report_wire_bytes
            .checked_add(report_record_bytes)
            .ok_or(QualificationError::HostDurationOutsideU64)?;
        pending.reported = true;
        self.received_sequences.insert(sequence);
        if pending.write_completed_at.is_some() {
            self.finalize_report_timing(sequence, received_at)?;
            Ok(None)
        } else {
            Ok(Some(DeferredReportTiming {
                sequence,
                received_at,
            }))
        }
    }

    fn finalize_report_timing(
        &mut self,
        sequence: u32,
        received_at: Instant,
    ) -> Result<(), QualificationError> {
        let pending = self
            .pending
            .get(&sequence)
            .ok_or(QualificationError::ReportForUnsentSequence { sequence })?;
        let write_started_at = pending
            .write_started_at
            .ok_or(QualificationError::WriterCompletionMissing { sequence })?;
        let write_completed_at = pending
            .write_completed_at
            .ok_or(QualificationError::WriterCompletionMissing { sequence })?;
        let rtt = received_at
            .checked_duration_since(write_started_at)
            .ok_or(QualificationError::HostDurationOutsideU64)?;
        let after_completion = received_at
            .checked_duration_since(write_completed_at)
            .ok_or(QualificationError::HostDurationOutsideU64)?;
        self.rtt_write_start_to_decode_ns
            .push(duration_ns_u64(rtt)?);
        self.response_after_write_completion_ns
            .push(duration_ns_u64(after_completion)?);
        Ok(())
    }

    fn write_completed_for(&self, sequence: u32) -> Result<bool, QualificationError> {
        self.pending
            .get(&sequence)
            .map(|pending| pending.write_completed_at.is_some())
            .ok_or(QualificationError::ReportForUnsentSequence { sequence })
    }

    fn missing_sequences(&self) -> Vec<u32> {
        self.pending
            .iter()
            .filter_map(|(sequence, pending)| (!pending.reported).then_some(*sequence))
            .collect()
    }
}

fn record_exact_length(
    recorded: &mut Option<usize>,
    direction: &'static str,
    actual: usize,
) -> Result<(), QualificationError> {
    if let Some(expected) = *recorded {
        if expected != actual {
            return Err(QualificationError::DiagnosticRecordLengthChanged {
                direction,
                expected,
                actual,
            });
        }
    } else {
        *recorded = Some(actual);
    }
    Ok(())
}

fn duration_ns_u64(duration: Duration) -> Result<u64, QualificationError> {
    u64::try_from(duration.as_nanos()).map_err(|_| QualificationError::HostDurationOutsideU64)
}

const fn ceiling_div_u64(numerator: u64, denominator: u64) -> u64 {
    numerator.saturating_add(denominator.saturating_sub(1)) / denominator
}

fn scheduled_instant(
    origin: Instant,
    rate: ProbeRateHz,
    index: usize,
) -> Result<Instant, QualificationError> {
    let index =
        u64::try_from(index).map_err(|_| QualificationError::SequenceOutsideU32 { index })?;
    let numerator = index
        .checked_mul(1_000_000_000)
        .ok_or(QualificationError::ScheduleInstantOverflow)?;
    let offset = Duration::from_nanos(numerator / u64::from(rate.get()));
    origin
        .checked_add(offset)
        .ok_or(QualificationError::ScheduleInstantOverflow)
}

fn fresh_run_id() -> Result<TransportDiagnosticRunId, QualificationError> {
    for _ in 0..MAX_ENTROPY_ATTEMPTS {
        let mut bytes = [0_u8; 8];
        getrandom::fill(&mut bytes).map_err(QualificationError::Entropy)?;
        if let Ok(run_id) = TransportDiagnosticRunId::try_new(u64::from_le_bytes(bytes)) {
            return Ok(run_id);
        }
    }
    Err(QualificationError::EntropyProducedOnlyZero {
        attempts: MAX_ENTROPY_ATTEMPTS,
    })
}

fn encoded_diagnostic_record_lengths(
    exact: ExactController,
    run_id: TransportDiagnosticRunId,
) -> Result<(usize, usize), QualificationError> {
    let sequence = TransportDiagnosticSequence::new(0);
    let token = HostElapsedNsToken::new(0);
    let probe = UartRecord::encode(Message::TransportDiagnosticProbe(
        TransportDiagnosticProbe::new(exact.uid, exact.boot_id, run_id, sequence, token),
    ))
    .map_err(QualificationError::Encode)?;
    let report = UartRecord::encode(Message::TransportDiagnosticReport(
        TransportDiagnosticReport {
            controller_uid: exact.uid,
            boot_id: exact.boot_id,
            run_id,
            sequence,
            host_elapsed_ns_token: token,
            result: TransportDiagnosticResultCode::EchoedMotorInert,
            output_state: OutputState::Disabled,
            timer_pwm: TimerPwm::ZERO,
            faults: ControllerFaults::NONE,
            request_received_at: ControllerUptimeMsWrapping::new(0),
            response_prepared_at: ControllerUptimeMsWrapping::new(0),
            rx_queue_depth_bytes: 0,
            tx_queue_depth_bytes: 0,
        },
    ))
    .map_err(QualificationError::Encode)?;
    Ok((probe.len(), report.len()))
}

enum RuntimeObservation {
    Other,
    Hello(Instant),
    Heartbeat(Instant),
    DeferredReport(DeferredReportTiming),
}

fn observe_runtime_message(
    timed: TimedMessage,
    exact: ExactController,
    tracker: &mut QualificationTracker,
    clocks: &mut ControllerClockTrackers,
) -> Result<RuntimeObservation, QualificationError> {
    match timed.message {
        Message::ControllerHello(value) => {
            validate_hello(exact, value)?;
            Ok(RuntimeObservation::Hello(timed.received_at))
        }
        Message::Heartbeat(value) => {
            validate_idle_heartbeat(exact, value)?;
            clocks.heartbeat.observe(value.controller_uptime.get())?;
            Ok(RuntimeObservation::Heartbeat(timed.received_at))
        }
        Message::ObservationalOdometry(value) => {
            validate_idle_odometry(exact, value)?;
            clocks
                .odometry_measurement
                .observe(value.controller_uptime.get())?;
            Ok(RuntimeObservation::Other)
        }
        Message::TransportDiagnosticReport(value) => Ok(tracker
            .observe_report_decode(exact, value, timed.received_at, clocks)?
            .map_or(
                RuntimeObservation::Other,
                RuntimeObservation::DeferredReport,
            )),
        Message::ControllerReady(value) => {
            validate_controller_identity(exact, value.controller_uid, value.boot_id)?;
            Err(QualificationError::UnexpectedControllerMessage {
                kind: Message::ControllerReady(value).kind(),
            })
        }
        Message::AppliedResult(value) => {
            validate_controller_identity(exact, value.controller_uid, value.boot_id)?;
            Err(QualificationError::UnexpectedControllerMessage {
                kind: Message::AppliedResult(value).kind(),
            })
        }
        value @ (Message::HostStopResult(_)
        | Message::AcquireResult(_)
        | Message::HostCommandResult(_)
        | Message::StatusReport(_)) => {
            Err(QualificationError::UnexpectedControllerMessage { kind: value.kind() })
        }
        value @ (Message::AcquireControl(_)
        | Message::HostCommand(_)
        | Message::HostStop(_)
        | Message::StatusQuery(_)
        | Message::BeginSession(_)
        | Message::ApplyPwm(_)
        | Message::ForceStop(_)
        | Message::TransportDiagnosticProbe(_)) => {
            Err(QualificationError::HostDirectionMessageFromController { kind: value.kind() })
        }
    }
}

fn reconcile_deferred_report_timings(
    tracker: &mut QualificationTracker,
    deferred: &mut VecDeque<DeferredReportTiming>,
) -> Result<(), QualificationError> {
    while let Some(timing) = deferred.front().copied() {
        if !tracker.write_completed_for(timing.sequence)? {
            break;
        }
        deferred.pop_front();
        tracker.finalize_report_timing(timing.sequence, timing.received_at)?;
    }
    Ok(())
}

struct StreamOutcome {
    tracker: QualificationTracker,
    latest_safe_heartbeat_at: Option<Instant>,
    liveness: RuntimeLiveness,
    observation_started_at: Instant,
    drain_started_at: Instant,
    finished_at: Instant,
    stream_controller_wire_bytes_observed: usize,
    stream_controller_records_observed: usize,
    drain_controller_wire_bytes_observed: usize,
    drain_controller_records_observed: usize,
}

#[derive(Clone, Copy)]
struct StreamParameters {
    exact: ExactController,
    plan: ProbePlan,
    run_id: TransportDiagnosticRunId,
    admission: Admission,
    serial_write_timeout: SerialWriteTimeout,
}

async fn run_probe_stream<R, W>(
    read_half: &mut R,
    write_half: W,
    framed: &mut FramedSerialReader,
    parameters: StreamParameters,
    clocks: &mut ControllerClockTrackers,
) -> Result<StreamOutcome, QualificationError>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin + Send + 'static,
{
    let StreamParameters {
        exact,
        plan,
        run_id,
        admission,
        serial_write_timeout,
    } = parameters;
    let observation_started_at = Instant::now();
    let observed_bytes_at_start = framed.observed_bytes;
    let observed_records_at_start = framed.observed_records;
    let (job_tx, job_rx) = mpsc::channel(WRITER_QUEUE_CAPACITY);
    let (completion_tx, mut completion_rx) = mpsc::channel(COMPLETION_QUEUE_CAPACITY);
    let writer = AbortOnDrop::new(tokio::spawn(run_writer(
        write_half,
        job_rx,
        completion_tx,
        serial_write_timeout,
    )));
    let writer_completion_bound = writer_completion_bound(serial_write_timeout)?;
    let mut tracker = QualificationTracker::new(run_id, plan.nominal_period);
    let origin = Instant::now()
        .checked_add(plan.nominal_period)
        .ok_or(QualificationError::ScheduleInstantOverflow)?;
    let mut next_index = 0_usize;
    let mut completed_writes = 0_usize;
    let mut drain_deadline = None;
    let mut writer_completion_deadline = None;
    let mut drain_started_at = None;
    let mut observed_bytes_at_drain_start = None;
    let mut observed_records_at_drain_start = None;
    let mut latest_safe_heartbeat_at = None;
    let mut liveness = RuntimeLiveness::from_admission(admission)?;
    let mut deferred_report_timings = VecDeque::with_capacity(MAX_DEFERRED_DIAGNOSTIC_REPORTS);

    loop {
        if next_index == plan.planned_probes
            && drain_deadline.is_none()
            && writer_completion_deadline.is_none()
        {
            writer_completion_deadline = Some(
                Instant::now()
                    .checked_add(writer_completion_bound)
                    .ok_or(QualificationError::WriterDeadlineOverflow)?,
            );
        }
        let timer_deadline = if next_index < plan.planned_probes {
            scheduled_instant(origin, plan.rate, next_index)?
        } else if let Some(deadline) = drain_deadline {
            deadline
        } else {
            writer_completion_deadline.ok_or(QualificationError::WriterDeadlineOverflow)?
        };

        tokio::select! {
            biased;
            completion = completion_rx.recv(), if completed_writes < tracker.pending.len() => {
                let completion = completion.ok_or(QualificationError::WriterCompletionClosed)?;
                tracker.observe_write_completion(completion)?;
                completed_writes = completed_writes
                    .checked_add(1)
                    .ok_or(QualificationError::HostDurationOutsideU64)?;
                reconcile_deferred_report_timings(
                    &mut tracker,
                    &mut deferred_report_timings,
                )?;
                if next_index == plan.planned_probes
                    && completed_writes == tracker.pending.len()
                    && drain_deadline.is_none()
                {
                    let drain_start = Instant::now();
                    drain_started_at = Some(drain_start);
                    observed_bytes_at_drain_start = Some(framed.observed_bytes);
                    observed_records_at_drain_start = Some(framed.observed_records);
                    drain_deadline = Some(
                        drain_start
                            .checked_add(plan.final_drain)
                            .ok_or(QualificationError::ScheduleInstantOverflow)?
                    );
                }
            }
            _ = tokio::time::sleep_until(timer_deadline) => {
                if next_index < plan.planned_probes {
                    let now = Instant::now();
                    while next_index + 1 < plan.planned_probes {
                        let next_scheduled = scheduled_instant(origin, plan.rate, next_index + 1)?;
                        if next_scheduled > now {
                            break;
                        }
                        tracker.scheduler_skipped_periods = tracker
                            .scheduler_skipped_periods
                            .checked_add(1)
                            .ok_or(QualificationError::HostDurationOutsideU64)?;
                        next_index += 1;
                    }
                    let scheduled_at = scheduled_instant(origin, plan.rate, next_index)?;
                    if tracker.outstanding() >= MAX_IN_FLIGHT {
                        tracker.in_flight_limit_skips = tracker
                            .in_flight_limit_skips
                            .checked_add(1)
                            .ok_or(QualificationError::HostDurationOutsideU64)?;
                    } else {
                        let sequence_value = u32::try_from(next_index)
                            .map_err(|_| QualificationError::SequenceOutsideU32 { index: next_index })?;
                        let sequence = TransportDiagnosticSequence::new(sequence_value);
                        let elapsed = now
                            .checked_duration_since(origin)
                            .ok_or(QualificationError::HostDurationOutsideU64)?;
                        let token = HostElapsedNsToken::new(duration_ns_u64(elapsed)?);
                        let record = UartRecord::encode(Message::TransportDiagnosticProbe(
                            TransportDiagnosticProbe::new(
                                exact.uid,
                                exact.boot_id,
                                run_id,
                                sequence,
                                token,
                            ),
                        ))
                        .map_err(QualificationError::Encode)?;
                        let encoded_bytes = record.len();
                        let job = WriteJob {
                            sequence,
                            scheduled_at,
                            dispatched_at: now,
                            record,
                        };
                        match job_tx.try_send(job) {
                            Ok(()) => {
                                tracker.insert_dispatched(
                                    sequence,
                                    token,
                                    scheduled_at,
                                    now,
                                    encoded_bytes,
                                )?;
                            }
                            Err(mpsc::error::TrySendError::Full(_)) => {
                                tracker.writer_queue_skips = tracker
                                    .writer_queue_skips
                                    .checked_add(1)
                                    .ok_or(QualificationError::HostDurationOutsideU64)?;
                            }
                            Err(mpsc::error::TrySendError::Closed(_)) => {
                                return Err(QualificationError::WriterQueueClosed);
                            }
                        }
                    }
                    next_index += 1;
                    if next_index == plan.planned_probes
                        && completed_writes == tracker.pending.len()
                    {
                        let drain_start = Instant::now();
                        drain_started_at = Some(drain_start);
                        observed_bytes_at_drain_start = Some(framed.observed_bytes);
                        observed_records_at_drain_start = Some(framed.observed_records);
                        drain_deadline = Some(
                            drain_start
                                .checked_add(plan.final_drain)
                                .ok_or(QualificationError::ScheduleInstantOverflow)?
                        );
                    }
                } else if drain_deadline.is_some() {
                    break;
                } else {
                    return Err(QualificationError::WriterCompletionTimeout {
                        outstanding: tracker.pending.len().saturating_sub(completed_writes),
                        maximum_ms: u64::try_from(writer_completion_bound.as_millis())
                            .unwrap_or(u64::MAX),
                    });
                }
            }
            timed = framed.next(read_half) => {
                match observe_runtime_message(timed?, exact, &mut tracker, clocks)? {
                    RuntimeObservation::Other => {}
                    RuntimeObservation::Hello(received_at) => {
                        liveness.hello.observe(received_at)?;
                    }
                    RuntimeObservation::Heartbeat(received_at) => {
                        liveness.heartbeat.observe(received_at)?;
                        latest_safe_heartbeat_at = Some(received_at);
                    }
                    RuntimeObservation::DeferredReport(timing) => {
                        if deferred_report_timings.len() >= MAX_DEFERRED_DIAGNOSTIC_REPORTS {
                            return Err(QualificationError::DeferredReportBudgetExceeded {
                                maximum: MAX_DEFERRED_DIAGNOSTIC_REPORTS,
                            });
                        }
                        deferred_report_timings.push_back(timing);
                    }
                }
            }
        }
    }

    drop(job_tx);
    writer.join_within(WRITER_JOIN_TIMEOUT).await?;
    if completed_writes != tracker.pending.len() {
        return Err(QualificationError::WriterCompletionMissing { sequence: 0 });
    }
    reconcile_deferred_report_timings(&mut tracker, &mut deferred_report_timings)?;
    if !deferred_report_timings.is_empty() {
        return Err(QualificationError::DeferredReportUnresolved {
            count: deferred_report_timings.len(),
        });
    }
    let last_write_completed_at = tracker
        .last_write_completed_at
        .ok_or(QualificationError::WriterCompletionMissing { sequence: 0 })?;
    if !latest_safe_heartbeat_at.is_some_and(|value| value > last_write_completed_at) {
        return Err(QualificationError::FinalHeartbeatMissing {
            drain_ms: u64::try_from(plan.final_drain.as_millis())
                .map_err(|_| QualificationError::HostDurationOutsideU64)?,
        });
    }
    let finished_at = Instant::now();
    liveness.finish(finished_at)?;
    let drain_started_at =
        drain_started_at.ok_or(QualificationError::WriterCompletionMissing { sequence: 0 })?;
    let observed_bytes_at_drain_start = observed_bytes_at_drain_start
        .ok_or(QualificationError::WriterCompletionMissing { sequence: 0 })?;
    let observed_records_at_drain_start = observed_records_at_drain_start
        .ok_or(QualificationError::WriterCompletionMissing { sequence: 0 })?;
    Ok(StreamOutcome {
        tracker,
        latest_safe_heartbeat_at,
        liveness,
        observation_started_at,
        drain_started_at,
        finished_at,
        stream_controller_wire_bytes_observed: observed_bytes_at_drain_start
            .checked_sub(observed_bytes_at_start)
            .ok_or(QualificationError::HostDurationOutsideU64)?,
        stream_controller_records_observed: observed_records_at_drain_start
            .checked_sub(observed_records_at_start)
            .ok_or(QualificationError::HostDurationOutsideU64)?,
        drain_controller_wire_bytes_observed: framed
            .observed_bytes
            .checked_sub(observed_bytes_at_drain_start)
            .ok_or(QualificationError::HostDurationOutsideU64)?,
        drain_controller_records_observed: framed
            .observed_records
            .checked_sub(observed_records_at_drain_start)
            .ok_or(QualificationError::HostDurationOutsideU64)?,
    })
}

#[derive(Serialize)]
struct DistributionU64 {
    samples: usize,
    minimum: u64,
    p50: u64,
    p95: u64,
    p99: u64,
    maximum: u64,
    exact_sum_decimal: String,
    arithmetic_mean: f64,
}

impl DistributionU64 {
    fn from_values(values: &[u64]) -> Option<Self> {
        if values.is_empty() {
            return None;
        }
        let mut sorted = values.to_vec();
        sorted.sort_unstable();
        let sum = sorted.iter().fold(0_u128, |accumulator, value| {
            accumulator + u128::from(*value)
        });
        Some(Self {
            samples: sorted.len(),
            minimum: sorted[0],
            p50: percentile_u64(&sorted, 50),
            p95: percentile_u64(&sorted, 95),
            p99: percentile_u64(&sorted, 99),
            maximum: sorted[sorted.len() - 1],
            exact_sum_decimal: sum.to_string(),
            arithmetic_mean: sum as f64 / sorted.len() as f64,
        })
    }
}

#[derive(Serialize)]
struct DistributionI64 {
    samples: usize,
    minimum: i64,
    p50: i64,
    p95: i64,
    p99: i64,
    maximum: i64,
    exact_sum_decimal: String,
    arithmetic_mean: f64,
}

impl DistributionI64 {
    fn from_values(values: &[i64]) -> Option<Self> {
        if values.is_empty() {
            return None;
        }
        let mut sorted = values.to_vec();
        sorted.sort_unstable();
        let sum = sorted.iter().fold(0_i128, |accumulator, value| {
            accumulator + i128::from(*value)
        });
        Some(Self {
            samples: sorted.len(),
            minimum: sorted[0],
            p50: percentile_i64(&sorted, 50),
            p95: percentile_i64(&sorted, 95),
            p99: percentile_i64(&sorted, 99),
            maximum: sorted[sorted.len() - 1],
            exact_sum_decimal: sum.to_string(),
            arithmetic_mean: sum as f64 / sorted.len() as f64,
        })
    }
}

fn percentile_index(length: usize, percentile: usize) -> usize {
    let rank = percentile
        .checked_mul(length)
        .and_then(|value| value.checked_add(99))
        .map_or(length, |value| value / 100);
    rank.saturating_sub(1).min(length - 1)
}

fn percentile_u64(sorted: &[u64], percentile: usize) -> u64 {
    sorted[percentile_index(sorted.len(), percentile)]
}

fn percentile_i64(sorted: &[i64], percentile: usize) -> i64 {
    sorted[percentile_index(sorted.len(), percentile)]
}

#[derive(Serialize)]
struct IdentityEvidence {
    controller_uid_hex: String,
    boot_id: u64,
    firmware_abi: u16,
    firmware_build_id: u32,
    actuator_config_fingerprint_hex: String,
    capabilities_bits: u32,
    max_abs_pwm_percent: u8,
    output_state: &'static str,
    max_command_lease_ms: u16,
    watchdog_nominal_period_ms: u16,
    pwm_frequency_hz: u16,
    neutral_output: &'static str,
    physical_stop_semantics: &'static str,
    admitted_idle_heartbeat_uptime_ms_wrapping: u32,
}

#[derive(Serialize)]
struct PlanEvidence {
    rate_hz: u16,
    nominal_period_ns: u64,
    duration_ms: u64,
    final_drain_ms: u64,
    serial_write_and_flush_timeout_ms: u64,
    maximum_writer_completion_wait_ms: u64,
    writer_join_timeout_ms: u64,
    planned_periods: usize,
    maximum_in_flight_bound: usize,
    writer_queue_capacity: usize,
}

#[derive(Serialize)]
struct WireLoadEvidence {
    baud_bits_per_second: u32,
    uart_format: &'static str,
    uart_bits_per_encoded_byte: u32,
    diagnostic_probe_encoded_bytes: usize,
    diagnostic_report_encoded_bytes: usize,
    theoretical_diagnostic_pairs_per_second: f64,
    theoretical_host_to_controller_bits_per_second: f64,
    theoretical_controller_to_host_bits_per_second: f64,
    theoretical_host_to_controller_fraction_of_baud: f64,
    theoretical_controller_to_host_fraction_of_baud: f64,
    theoretical_scope: &'static str,
}

#[derive(Serialize)]
struct CountEvidence {
    planned_periods: usize,
    probes_dispatched_to_writer: usize,
    completed_writes: usize,
    unique_reports: usize,
    missing_reports: usize,
    duplicate_reports: usize,
    reordered_reports: usize,
    scheduler_skipped_periods: usize,
    in_flight_limit_skips: usize,
    writer_queue_skips: usize,
    writes_late_by_at_least_one_period: usize,
    maximum_observed_in_flight: usize,
}

#[derive(Serialize)]
struct TimingEvidence {
    planned_schedule_window_ns: u64,
    qualification_stream_observation_elapsed_ns: u64,
    final_receive_only_drain_elapsed_ns: u64,
    total_qualification_observation_elapsed_ns: u64,
    scheduler_dispatch_lateness_ns: Option<DistributionU64>,
    write_start_lateness_ns: Option<DistributionU64>,
    inter_write_start_jitter_ns_signed: Option<DistributionI64>,
    host_write_start_to_report_decode_rtt_ns: Option<DistributionU64>,
    host_write_completion_to_report_decode_ns: Option<DistributionU64>,
    controller_receive_to_prepare_service_ms: Option<DistributionU64>,
}

#[derive(Serialize)]
struct QueueEvidence {
    controller_rx_queue_depth_bytes_at_dispatch: Option<DistributionU64>,
    controller_tx_queue_depth_bytes_before_response_encoding: Option<DistributionU64>,
}

#[derive(Serialize)]
struct ThroughputEvidence {
    diagnostic_probe_wire_bytes_written: usize,
    unique_diagnostic_report_wire_bytes_decoded: usize,
    controller_wire_bytes_observed_during_stream_window: usize,
    controller_records_observed_during_stream_window: usize,
    controller_wire_bytes_observed_during_final_drain_window: usize,
    controller_records_observed_during_final_drain_window: usize,
    unique_reports_per_second_over_total_observation_window: f64,
    unique_diagnostic_8n1_wire_bits_per_second_over_total_observation_window: f64,
    controller_8n1_wire_bits_per_second_during_stream_window: f64,
    controller_8n1_wire_bits_per_second_during_final_drain_window: f64,
    measurement_window_boundary: &'static str,
}

#[derive(Serialize)]
struct LivenessEvidence {
    controller_hello_messages_validated_including_admission: usize,
    idle_safe_heartbeat_messages_validated_including_admission: usize,
    controller_hello_maximum_observed_gap_ns_including_trailing_boundary: u64,
    idle_safe_heartbeat_maximum_observed_gap_ns_including_trailing_boundary: u64,
    controller_hello_maximum_allowed_gap_ms: u64,
    idle_safe_heartbeat_maximum_allowed_gap_ms: u64,
    bound_policy: &'static str,
}

#[derive(Serialize)]
struct QualificationEvidence {
    schema_version: u32,
    evidence_kind: &'static str,
    passed: bool,
    serial_by_id_path: String,
    run_id: u64,
    identity: IdentityEvidence,
    plan: PlanEvidence,
    wire_load: WireLoadEvidence,
    counts: CountEvidence,
    timing: TimingEvidence,
    queues: QueueEvidence,
    throughput: ThroughputEvidence,
    liveness: LivenessEvidence,
    missing_sequences: Vec<u32>,
    final_idle_safe_heartbeat_received_after_last_write: bool,
    integrity_pattern_boundary: &'static str,
    controller_clock_boundary: &'static str,
    evidence_boundary: &'static str,
}

fn build_evidence(
    cli: &Cli,
    plan: ProbePlan,
    run_id: TransportDiagnosticRunId,
    admission: Admission,
    outcome: StreamOutcome,
) -> Result<QualificationEvidence, QualificationError> {
    let stream_elapsed = outcome
        .drain_started_at
        .checked_duration_since(outcome.observation_started_at)
        .ok_or(QualificationError::HostDurationOutsideU64)?;
    let drain_elapsed = outcome
        .finished_at
        .checked_duration_since(outcome.drain_started_at)
        .ok_or(QualificationError::HostDurationOutsideU64)?;
    let total_elapsed = outcome
        .finished_at
        .checked_duration_since(outcome.observation_started_at)
        .ok_or(QualificationError::HostDurationOutsideU64)?;
    let stream_elapsed_ns = duration_ns_u64(stream_elapsed)?;
    let drain_elapsed_ns = duration_ns_u64(drain_elapsed)?;
    let total_elapsed_ns = duration_ns_u64(total_elapsed)?;
    let planned_schedule_window_ns = duration_ns_u64(plan.duration)?;
    let stream_elapsed_seconds = stream_elapsed.as_secs_f64();
    let drain_elapsed_seconds = drain_elapsed.as_secs_f64();
    let total_elapsed_seconds = total_elapsed.as_secs_f64();
    let nominal_period_ns = duration_ns_u64(plan.nominal_period)?;
    let duration_ms = u64::try_from(plan.duration.as_millis())
        .map_err(|_| QualificationError::HostDurationOutsideU64)?;
    let final_drain_ms = u64::try_from(plan.final_drain.as_millis())
        .map_err(|_| QualificationError::HostDurationOutsideU64)?;
    let maximum_writer_completion_wait_ms =
        u64::try_from(writer_completion_bound(cli.serial_write_timeout_ms)?.as_millis())
            .map_err(|_| QualificationError::HostDurationOutsideU64)?;
    let (probe_record_bytes, report_record_bytes) =
        encoded_diagnostic_record_lengths(ExactController::from_cli(cli), run_id)?;
    let tracker = outcome.tracker;
    if let Some(actual) = tracker.diagnostic_probe_record_bytes {
        if actual != probe_record_bytes {
            return Err(QualificationError::DiagnosticRecordLengthChanged {
                direction: "host-to-controller-derived",
                expected: probe_record_bytes,
                actual,
            });
        }
    }
    if let Some(actual) = tracker.diagnostic_report_record_bytes {
        if actual != report_record_bytes {
            return Err(QualificationError::DiagnosticRecordLengthChanged {
                direction: "controller-to-host-derived",
                expected: report_record_bytes,
                actual,
            });
        }
    }
    let missing_sequences = tracker.missing_sequences();
    let theoretical_pairs_per_second = f64::from(plan.rate.get());
    let theoretical_host_to_controller_bits_per_second = probe_record_bytes as f64
        * f64::from(UART_BITS_PER_BYTE_8N1)
        * theoretical_pairs_per_second;
    let theoretical_controller_to_host_bits_per_second = report_record_bytes as f64
        * f64::from(UART_BITS_PER_BYTE_8N1)
        * theoretical_pairs_per_second;
    let measured_diagnostic_bytes = tracker
        .sent_wire_bytes
        .checked_add(tracker.diagnostic_report_wire_bytes)
        .ok_or(QualificationError::HostDurationOutsideU64)?;
    let completed_writes = tracker
        .pending
        .values()
        .filter(|probe| probe.write_completed_at.is_some())
        .count();
    let passed = tracker.pending.len() == plan.planned_probes
        && completed_writes == plan.planned_probes
        && tracker.received_sequences.len() == plan.planned_probes
        && missing_sequences.is_empty()
        && tracker.duplicate_reports == 0
        && tracker.reordered_reports == 0
        && tracker.scheduler_skipped_periods == 0
        && tracker.in_flight_limit_skips == 0
        && tracker.writer_queue_skips == 0
        && tracker.late_by_at_least_one_period == 0;

    Ok(QualificationEvidence {
        schema_version: 1,
        evidence_kind: "motor_inert_krp2_uart_transport_qualification",
        passed,
        serial_by_id_path: cli.serial_device.as_str().to_owned(),
        run_id: run_id.get(),
        identity: IdentityEvidence {
            controller_uid_hex: encode_hex(admission.hello.controller_uid.as_bytes()),
            boot_id: admission.hello.boot_id.get(),
            firmware_abi: admission.hello.firmware_abi,
            firmware_build_id: admission.hello.firmware_build_id,
            actuator_config_fingerprint_hex: encode_hex(
                admission.hello.actuator_config_fingerprint.as_bytes(),
            ),
            capabilities_bits: admission.hello.capabilities.bits(),
            max_abs_pwm_percent: admission.hello.max_abs_pwm_percent.get(),
            output_state: output_state_name(admission.hello.output_state),
            max_command_lease_ms: admission.hello.max_command_lease.get(),
            watchdog_nominal_period_ms: admission.hello.watchdog_nominal_period.get(),
            pwm_frequency_hz: admission.hello.pwm_frequency.get(),
            neutral_output: neutral_output_name(admission.hello.neutral_output),
            physical_stop_semantics: stop_semantics_name(admission.hello.physical_stop_semantics),
            admitted_idle_heartbeat_uptime_ms_wrapping: admission.heartbeat.controller_uptime.get(),
        },
        plan: PlanEvidence {
            rate_hz: plan.rate.get(),
            nominal_period_ns,
            duration_ms,
            final_drain_ms,
            serial_write_and_flush_timeout_ms: cli.serial_write_timeout_ms.as_millis_u64(),
            maximum_writer_completion_wait_ms,
            writer_join_timeout_ms: u64::try_from(WRITER_JOIN_TIMEOUT.as_millis())
                .map_err(|_| QualificationError::HostDurationOutsideU64)?,
            planned_periods: plan.planned_probes,
            maximum_in_flight_bound: MAX_IN_FLIGHT,
            writer_queue_capacity: WRITER_QUEUE_CAPACITY,
        },
        wire_load: WireLoadEvidence {
            baud_bits_per_second: SERIAL_BAUD_BPS,
            uart_format: "8N1_no_flow_control",
            uart_bits_per_encoded_byte: UART_BITS_PER_BYTE_8N1,
            diagnostic_probe_encoded_bytes: probe_record_bytes,
            diagnostic_report_encoded_bytes: report_record_bytes,
            theoretical_diagnostic_pairs_per_second: theoretical_pairs_per_second,
            theoretical_host_to_controller_bits_per_second,
            theoretical_controller_to_host_bits_per_second,
            theoretical_host_to_controller_fraction_of_baud:
                theoretical_host_to_controller_bits_per_second / f64::from(SERIAL_BAUD_BPS),
            theoretical_controller_to_host_fraction_of_baud:
                theoretical_controller_to_host_bits_per_second / f64::from(SERIAL_BAUD_BPS),
            theoretical_scope: "each full-duplex direction is reported separately for diagnostic records only; excludes Hello, Heartbeat, odometry, OS scheduling, and controller service",
        },
        counts: CountEvidence {
            planned_periods: plan.planned_probes,
            probes_dispatched_to_writer: tracker.pending.len(),
            completed_writes,
            unique_reports: tracker.received_sequences.len(),
            missing_reports: missing_sequences.len(),
            duplicate_reports: tracker.duplicate_reports,
            reordered_reports: tracker.reordered_reports,
            scheduler_skipped_periods: tracker.scheduler_skipped_periods,
            in_flight_limit_skips: tracker.in_flight_limit_skips,
            writer_queue_skips: tracker.writer_queue_skips,
            writes_late_by_at_least_one_period: tracker.late_by_at_least_one_period,
            maximum_observed_in_flight: tracker.maximum_in_flight,
        },
        timing: TimingEvidence {
            planned_schedule_window_ns,
            qualification_stream_observation_elapsed_ns: stream_elapsed_ns,
            final_receive_only_drain_elapsed_ns: drain_elapsed_ns,
            total_qualification_observation_elapsed_ns: total_elapsed_ns,
            scheduler_dispatch_lateness_ns: DistributionU64::from_values(
                &tracker.dispatch_lateness_ns,
            ),
            write_start_lateness_ns: DistributionU64::from_values(&tracker.write_start_lateness_ns),
            inter_write_start_jitter_ns_signed: DistributionI64::from_values(
                &tracker.inter_write_start_jitter_ns,
            ),
            host_write_start_to_report_decode_rtt_ns: DistributionU64::from_values(
                &tracker.rtt_write_start_to_decode_ns,
            ),
            host_write_completion_to_report_decode_ns: DistributionU64::from_values(
                &tracker.response_after_write_completion_ns,
            ),
            controller_receive_to_prepare_service_ms: DistributionU64::from_values(
                &tracker.controller_service_ms,
            ),
        },
        queues: QueueEvidence {
            controller_rx_queue_depth_bytes_at_dispatch: DistributionU64::from_values(
                &tracker.rx_queue_depth_bytes,
            ),
            controller_tx_queue_depth_bytes_before_response_encoding: DistributionU64::from_values(
                &tracker.tx_queue_depth_bytes,
            ),
        },
        throughput: ThroughputEvidence {
            diagnostic_probe_wire_bytes_written: tracker.sent_wire_bytes,
            unique_diagnostic_report_wire_bytes_decoded: tracker.diagnostic_report_wire_bytes,
            controller_wire_bytes_observed_during_stream_window: outcome
                .stream_controller_wire_bytes_observed,
            controller_records_observed_during_stream_window: outcome
                .stream_controller_records_observed,
            controller_wire_bytes_observed_during_final_drain_window: outcome
                .drain_controller_wire_bytes_observed,
            controller_records_observed_during_final_drain_window: outcome
                .drain_controller_records_observed,
            unique_reports_per_second_over_total_observation_window: tracker
                .received_sequences
                .len() as f64
                / total_elapsed_seconds,
            unique_diagnostic_8n1_wire_bits_per_second_over_total_observation_window:
                measured_diagnostic_bytes as f64 * f64::from(UART_BITS_PER_BYTE_8N1)
                    / total_elapsed_seconds,
            controller_8n1_wire_bits_per_second_during_stream_window: outcome
                .stream_controller_wire_bytes_observed
                as f64
                * f64::from(UART_BITS_PER_BYTE_8N1)
                / stream_elapsed_seconds,
            controller_8n1_wire_bits_per_second_during_final_drain_window: outcome
                .drain_controller_wire_bytes_observed
                as f64
                * f64::from(UART_BITS_PER_BYTE_8N1)
                / drain_elapsed_seconds,
            measurement_window_boundary: "controller byte/record counters exclude admission; the stream window ends and the receive-only drain window begins when the host processes the final successful write-and-flush completion",
        },
        liveness: LivenessEvidence {
            controller_hello_messages_validated_including_admission: outcome.liveness.hello.count,
            idle_safe_heartbeat_messages_validated_including_admission: outcome
                .liveness
                .heartbeat
                .count,
            controller_hello_maximum_observed_gap_ns_including_trailing_boundary: duration_ns_u64(
                outcome.liveness.hello.maximum_observed_gap,
            )?,
            idle_safe_heartbeat_maximum_observed_gap_ns_including_trailing_boundary:
                duration_ns_u64(outcome.liveness.heartbeat.maximum_observed_gap)?,
            controller_hello_maximum_allowed_gap_ms: u64::try_from(
                outcome.liveness.hello.maximum_gap.as_millis(),
            )
            .map_err(|_| QualificationError::HostDurationOutsideU64)?,
            idle_safe_heartbeat_maximum_allowed_gap_ms: u64::try_from(
                outcome.liveness.heartbeat.maximum_gap.as_millis(),
            )
            .map_err(|_| QualificationError::HostDurationOutsideU64)?,
            bound_policy: "Heartbeat gap <= advertised watchdog_nominal_period; ControllerHello gap <= 2x the canonical protocol Hello period. These are host qualification bounds; only the watchdog period is an on-wire field.",
        },
        missing_sequences,
        final_idle_safe_heartbeat_received_after_last_write: outcome
            .latest_safe_heartbeat_at
            .is_some(),
        integrity_pattern_boundary: "the deterministic 20-byte pattern discriminates intended load and construction only; it is public and is not authentication",
        controller_clock_boundary: "wrapping-forward order is checked independently for Heartbeat uptime, odometry measurement uptime, diagnostic request receipt, and diagnostic response preparation; cross-stream timestamp order is not assumed because measurement and queueing times differ",
        evidence_boundary: "software-observed host UART timing, decoded controller claims, and queue-depth samples for this run; no wheel motion, motor current, physical safety, or performance improvement is claimed",
    })
}

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
}

const fn output_state_name(value: OutputState) -> &'static str {
    match value {
        OutputState::Disabled => "disabled",
        OutputState::ZeroPwm => "zero_pwm",
        OutputState::NonzeroPwm => "nonzero_pwm",
    }
}

const fn neutral_output_name(value: NeutralOutput) -> &'static str {
    match value {
        NeutralOutput::BothLow => "both_low",
        NeutralOutput::BothHigh => "both_high",
        NeutralOutput::HighImpedance => "high_impedance",
    }
}

const fn stop_semantics_name(value: PhysicalStopSemantics) -> &'static str {
    match value {
        PhysicalStopSemantics::Unverified => "unverified",
        PhysicalStopSemantics::CoastVerified => "coast_verified",
        PhysicalStopSemantics::BrakeVerified => "brake_verified",
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), QualificationError> {
    let cli = Cli::parse();
    let plan = ProbePlan::parse(&cli)?;
    let exact = ExactController::from_cli(&cli);
    let run_id = fresh_run_id()?;

    // Entropy and every weak CLI boundary are resolved before opening the TTY.
    let mut port = tokio_serial::new(cli.serial_device.as_str(), SERIAL_BAUD_BPS)
        .data_bits(DataBits::Eight)
        .parity(Parity::None)
        .stop_bits(StopBits::One)
        .flow_control(FlowControl::None)
        .open_native_async()
        .map_err(QualificationError::Open)?;
    port.set_exclusive(true)
        .map_err(QualificationError::Exclusive)?;

    let mut framed = FramedSerialReader::new();
    let mut clocks = ControllerClockTrackers::new();
    let admission = admit_controller(
        &mut port,
        &mut framed,
        exact,
        cli.admission_timeout_ms,
        &mut clocks,
    )
    .await?;
    let (mut read_half, write_half) = tokio::io::split(port);
    let outcome = run_probe_stream(
        &mut read_half,
        write_half,
        &mut framed,
        StreamParameters {
            exact,
            plan,
            run_id,
            admission,
            serial_write_timeout: cli.serial_write_timeout_ms,
        },
        &mut clocks,
    )
    .await?;
    let evidence = build_evidence(&cli, plan, run_id, admission, outcome)?;
    let passed = evidence.passed;
    serde_json::to_writer_pretty(std::io::stdout().lock(), &evidence)
        .map_err(QualificationError::Json)?;
    println!();
    if passed {
        Ok(())
    } else {
        Err(QualificationError::EvidenceFailed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::pin::Pin;
    use std::task::{Context, Poll};

    use robot_protocol::v2::{
        ControllerDeadlineMsWrapping, MaxAbsPwmPercent, PwmFrequencyHz, V2CommandLeaseMs,
        WatchdogNominalPeriodMs, MAX_UART_RECORD_BYTES,
    };

    struct FlushPendingWriter;

    impl AsyncWrite for FlushPendingWriter {
        fn poll_write(
            self: Pin<&mut Self>,
            _context: &mut Context<'_>,
            buffer: &[u8],
        ) -> Poll<std::io::Result<usize>> {
            Poll::Ready(Ok(buffer.len()))
        }

        fn poll_flush(
            self: Pin<&mut Self>,
            _context: &mut Context<'_>,
        ) -> Poll<std::io::Result<()>> {
            Poll::Pending
        }

        fn poll_shutdown(
            self: Pin<&mut Self>,
            _context: &mut Context<'_>,
        ) -> Poll<std::io::Result<()>> {
            Poll::Ready(Ok(()))
        }
    }

    fn controller_uid() -> ControllerUid {
        ControllerUid::try_new([0x11; 12]).expect("nonzero controller UID")
    }

    fn boot_id() -> ControllerBootId {
        ControllerBootId::try_new(0x0102_0304_0506_0708).expect("nonzero boot ID")
    }

    fn fingerprint() -> ActuatorConfigFingerprint {
        ActuatorConfigFingerprint::try_new(*b"KIKO-NO-ACT-V1!!")
            .expect("nonzero actuator fingerprint")
    }

    fn capabilities() -> ControllerCapabilities {
        ControllerCapabilities::try_from_bits(
            ControllerCapabilities::HEARTBEAT
                | ControllerCapabilities::MOTOR_INERT_TRANSPORT_DIAGNOSTICS,
        )
        .expect("known capability bits")
    }

    fn exact_controller() -> ExactController {
        ExactController {
            uid: controller_uid(),
            boot_id: boot_id(),
            firmware_abi: 2,
            firmware_build_id: 0x0002_0002,
            fingerprint: fingerprint(),
            capabilities: capabilities(),
        }
    }

    fn hello() -> ControllerHello {
        ControllerHello {
            controller_uid: controller_uid(),
            boot_id: boot_id(),
            firmware_abi: 2,
            firmware_build_id: 0x0002_0002,
            capabilities: capabilities(),
            max_abs_pwm_percent: MaxAbsPwmPercent::try_new(0).expect("zero is valid"),
            max_command_lease: V2CommandLeaseMs::try_new(100).expect("valid lease"),
            output_state: OutputState::Disabled,
            actuator_config_fingerprint: fingerprint(),
            watchdog_nominal_period: WatchdogNominalPeriodMs::try_new(500)
                .expect("valid watchdog period"),
            pwm_frequency: PwmFrequencyHz::try_new(20_000).expect("nonzero PWM frequency"),
            neutral_output: NeutralOutput::BothLow,
            physical_stop_semantics: PhysicalStopSemantics::Unverified,
        }
    }

    fn heartbeat(uptime: u32) -> Heartbeat {
        Heartbeat {
            controller_uid: controller_uid(),
            boot_id: boot_id(),
            control_epoch: None,
            last_sequence: None,
            controller_uptime: ControllerUptimeMsWrapping::new(uptime),
            expires_at: ControllerDeadlineMsWrapping::new(uptime),
            timer_pwm: TimerPwm::ZERO,
            output_state: OutputState::Disabled,
            readiness: ReadinessFlags::try_from_bits(ReadinessFlags::WATCHDOG_RUNNING)
                .expect("known readiness bit"),
            faults: ControllerFaults::NONE,
        }
    }

    fn report(
        run_id: TransportDiagnosticRunId,
        sequence: TransportDiagnosticSequence,
        token: HostElapsedNsToken,
        received_ms: u32,
    ) -> TransportDiagnosticReport {
        TransportDiagnosticReport {
            controller_uid: controller_uid(),
            boot_id: boot_id(),
            run_id,
            sequence,
            host_elapsed_ns_token: token,
            result: TransportDiagnosticResultCode::EchoedMotorInert,
            output_state: OutputState::Disabled,
            timer_pwm: TimerPwm::ZERO,
            faults: ControllerFaults::NONE,
            request_received_at: ControllerUptimeMsWrapping::new(received_ms),
            response_prepared_at: ControllerUptimeMsWrapping::new(received_ms.wrapping_add(1)),
            rx_queue_depth_bytes: 3,
            tx_queue_depth_bytes: 4,
        }
    }

    fn cli(rate_hz: u16, duration_ms: u64) -> Cli {
        Cli {
            serial_device: PersistentSerialPath::from_str("/dev/serial/by-id/kiko-stm32")
                .expect("persistent path"),
            controller_uid_hex: ControllerUidArgument(controller_uid()),
            boot_id: boot_id(),
            firmware_abi: 2,
            firmware_build_id: 0x0002_0002,
            actuator_config_fingerprint_hex: FingerprintArgument(fingerprint()),
            capabilities_bits: capabilities(),
            rate_hz: ProbeRateHz(NonZeroU16::new(rate_hz).expect("test benchmark rate is nonzero")),
            duration_ms,
            admission_timeout_ms: 5_000,
            final_drain_ms: 1_000,
            serial_write_timeout_ms: SerialWriteTimeout(Duration::from_millis(10)),
        }
    }

    #[test]
    fn weak_cli_boundaries_parse_once_and_plan_is_bounded() {
        assert!(PersistentSerialPath::from_str("/dev/serial/by-id/kiko-stm32").is_ok());
        assert!(matches!(
            PersistentSerialPath::from_str("/dev/ttyACM0"),
            Err(ArgumentError::SerialPathNotPersistent)
        ));
        assert!(matches!(
            PersistentSerialPath::from_str("/dev/serial/by-id/nested/device"),
            Err(ArgumentError::SerialPathNonCanonical)
        ));
        assert_eq!(parse_u16("0x2").expect("hex ABI"), 2);
        assert!(matches!(
            ControllerUidArgument::from_str("000000000000000000000000"),
            Err(ArgumentError::ZeroControllerUid)
        ));
        assert!(matches!(
            parse_capabilities("0x80000000"),
            Err(ArgumentError::UnknownCapabilityBits { .. })
        ));

        for (rate_hz, expected) in [(20, 200), (50, 500), (75, 750), (100, 1_000)] {
            let parsed =
                ProbePlan::parse(&cli(rate_hz, 10_000)).expect("supported exact bounded plan");
            assert_eq!(parsed.planned_probes, expected);
        }
        assert!(ProbeRateHz::from_str("75").is_ok());
        assert!(SerialWriteTimeout::from_str("7").is_ok());
        assert!(SerialWriteTimeout::from_str("100").is_ok());
        assert!(SerialWriteTimeout::from_str("6").is_err());
        assert!(SerialWriteTimeout::from_str("101").is_err());
        assert!(matches!(
            ProbeRateHz::from_str("60"),
            Err(ArgumentError::UnsupportedBenchmarkRate { actual_hz: 60 })
        ));
        assert!(matches!(
            ProbePlan::parse(&cli(75, 1_001)),
            Err(QualificationError::DurationHasFractionalProbeCount { .. })
        ));
    }

    #[test]
    fn seventy_five_hertz_schedule_is_rational_and_has_no_cumulative_rounding_drift() {
        let origin = Instant::now();
        let rate = ProbeRateHz::from_str("75").expect("supported rate");
        let first = scheduled_instant(origin, rate, 1).expect("first schedule");
        let third = scheduled_instant(origin, rate, 3).expect("third schedule");
        assert_eq!(
            first.duration_since(origin),
            Duration::from_nanos(13_333_333)
        );
        assert_eq!(third.duration_since(origin), Duration::from_millis(40));
    }

    #[test]
    fn admission_requires_exact_motor_inert_identity_and_idle_safe_heartbeat() {
        let exact = exact_controller();
        let valid_hello = hello();
        let valid_heartbeat = heartbeat(1_000);
        validate_hello(exact, valid_hello).expect("baseline Hello is admissible");
        validate_idle_heartbeat(exact, valid_heartbeat).expect("baseline Heartbeat is idle-safe");

        let mut motion_capable = valid_hello;
        motion_capable.max_abs_pwm_percent =
            MaxAbsPwmPercent::try_new(1).expect("one percent is representable");
        assert!(matches!(
            validate_hello(exact, motion_capable),
            Err(QualificationError::HelloNotMotorInert { .. })
        ));

        let mut stale_boot = valid_hello;
        stale_boot.boot_id = ControllerBootId::try_new(9).expect("nonzero boot ID");
        assert!(matches!(
            validate_hello(exact, stale_boot),
            Err(QualificationError::IdentityMismatch { field: "boot_id" })
        ));

        let mut active_session = valid_heartbeat;
        active_session.readiness = ReadinessFlags::try_from_bits(
            ReadinessFlags::WATCHDOG_RUNNING | ReadinessFlags::SESSION_ESTABLISHED,
        )
        .expect("known readiness bits");
        assert!(matches!(
            validate_idle_heartbeat(exact, active_session),
            Err(QualificationError::HeartbeatNotIdleSafe { .. })
        ));

        let mut faulted = valid_heartbeat;
        faulted.faults = ControllerFaults::try_from_bits(ControllerFaults::MOTOR_DRIVER)
            .expect("known fault bit");
        assert!(matches!(
            validate_idle_heartbeat(exact, faulted),
            Err(QualificationError::HeartbeatNotIdleSafe { .. })
        ));
    }

    #[test]
    fn admission_heartbeat_freshness_uses_the_advertised_watchdog_bound() {
        let received_at = Instant::now();
        let admitted_hello = hello();
        let bound = Duration::from_millis(u64::from(admitted_hello.watchdog_nominal_period.get()));
        assert!(admission_heartbeat_is_fresh(
            admitted_hello,
            received_at,
            received_at
                .checked_add(bound)
                .expect("small watchdog duration")
        ));
        assert!(!admission_heartbeat_is_fresh(
            admitted_hello,
            received_at,
            received_at
                .checked_add(bound)
                .and_then(|instant| instant.checked_add(Duration::from_nanos(1)))
                .expect("small watchdog duration")
        ));
        assert!(!admission_heartbeat_is_fresh(
            admitted_hello,
            received_at,
            received_at
                .checked_sub(Duration::from_nanos(1))
                .expect("small backwards duration")
        ));
    }

    #[test]
    fn every_diagnostic_report_must_repeat_safe_zero_and_clear_fault_evidence() {
        let run_id = TransportDiagnosticRunId::try_new(1).expect("nonzero run ID");
        let sequence = TransportDiagnosticSequence::new(0);
        let token = HostElapsedNsToken::new(10);
        let valid = report(run_id, sequence, token, 10);
        validate_idle_diagnostic_report(valid).expect("baseline report is idle-safe");

        let mut moving = valid;
        moving.timer_pwm = TimerPwm::try_new(1, 0).expect("valid nonzero PWM");
        moving.output_state = OutputState::NonzeroPwm;
        assert!(matches!(
            validate_idle_diagnostic_report(moving),
            Err(QualificationError::DiagnosticReportNotIdleSafe { .. })
        ));

        let mut faulted = valid;
        faulted.faults =
            ControllerFaults::try_from_bits(ControllerFaults::INTERNAL).expect("known fault bit");
        assert!(matches!(
            validate_idle_diagnostic_report(faulted),
            Err(QualificationError::DiagnosticReportNotIdleSafe { .. })
        ));
    }

    #[test]
    fn controller_clock_accepts_wrap_and_rejects_backward_half_range() {
        let mut clock = ControllerClockTracker::new("test clock");
        clock.observe(u32::MAX - 1).expect("first sample");
        clock.observe(1).expect("small wrapping-forward delta");
        assert!(matches!(
            clock.observe(0),
            Err(QualificationError::ControllerClockAnomaly {
                stream: "test clock",
                previous_ms: 1,
                current_ms: 0
            })
        ));
    }

    #[test]
    fn report_decode_before_completion_is_deferred_without_losing_decode_order() {
        let exact = exact_controller();
        let run_id = TransportDiagnosticRunId::try_new(2).expect("nonzero run ID");
        let sequence = TransportDiagnosticSequence::new(0);
        let token = HostElapsedNsToken::new(20);
        let (probe_bytes, _) =
            encoded_diagnostic_record_lengths(exact, run_id).expect("encodable diagnostics");
        let scheduled_at = Instant::now();
        let dispatched_at = scheduled_at
            .checked_add(Duration::from_millis(1))
            .expect("small duration");
        let mut tracker = QualificationTracker::new(run_id, Duration::from_millis(20));
        tracker
            .insert_dispatched(sequence, token, scheduled_at, dispatched_at, probe_bytes)
            .expect("tracked dispatch");
        let decoded_at = scheduled_at
            .checked_add(Duration::from_millis(5))
            .expect("small duration");
        let diagnostic_report = report(run_id, sequence, token, 100);
        let mut clocks = ControllerClockTrackers::new();
        let deferred = tracker
            .observe_report_decode(exact, diagnostic_report, decoded_at, &mut clocks)
            .expect("valid report")
            .expect("writer completion is not yet observed");
        assert_eq!(tracker.received_sequences.len(), 1);
        assert!(tracker.rtt_write_start_to_decode_ns.is_empty());

        tracker
            .observe_write_completion(WriteCompletion {
                sequence,
                scheduled_at,
                dispatched_at,
                write_started_at: scheduled_at
                    .checked_add(Duration::from_millis(2))
                    .expect("small duration"),
                write_completed_at: scheduled_at
                    .checked_add(Duration::from_millis(3))
                    .expect("small duration"),
                result: Ok(()),
                encoded_bytes: probe_bytes,
            })
            .expect("valid completion");
        let mut deferred_queue = VecDeque::from([deferred]);
        reconcile_deferred_report_timings(&mut tracker, &mut deferred_queue)
            .expect("deferred timing reconciles");
        assert!(deferred_queue.is_empty());
        assert_eq!(tracker.rtt_write_start_to_decode_ns, [3_000_000]);
        assert_eq!(tracker.response_after_write_completion_ns, [2_000_000]);

        tracker
            .observe_report_decode(
                exact,
                diagnostic_report,
                decoded_at
                    .checked_add(Duration::from_millis(1))
                    .expect("small duration"),
                &mut clocks,
            )
            .expect("duplicate remains structurally valid");
        assert_eq!(tracker.duplicate_reports, 1);
    }

    #[test]
    fn tracker_reports_loss_and_decode_order_without_inferring_missing_payloads() {
        let exact = exact_controller();
        let run_id = TransportDiagnosticRunId::try_new(4).expect("nonzero run ID");
        let (probe_bytes, _) =
            encoded_diagnostic_record_lengths(exact, run_id).expect("encodable diagnostics");
        let base = Instant::now();
        let mut tracker = QualificationTracker::new(run_id, Duration::from_millis(20));
        let mut tokens = [HostElapsedNsToken::new(0); 2];
        for (index, token_slot) in tokens.iter_mut().enumerate() {
            let index = u32::try_from(index).expect("two test entries fit u32");
            let sequence = TransportDiagnosticSequence::new(index);
            let token = HostElapsedNsToken::new(u64::from(index) + 100);
            *token_slot = token;
            let scheduled_at = base
                .checked_add(Duration::from_millis(u64::from(index) * 20))
                .expect("small duration");
            tracker
                .insert_dispatched(sequence, token, scheduled_at, scheduled_at, probe_bytes)
                .expect("tracked dispatch");
            tracker
                .observe_write_completion(WriteCompletion {
                    sequence,
                    scheduled_at,
                    dispatched_at: scheduled_at,
                    write_started_at: scheduled_at
                        .checked_add(Duration::from_millis(1))
                        .expect("small duration"),
                    write_completed_at: scheduled_at
                        .checked_add(Duration::from_millis(2))
                        .expect("small duration"),
                    result: Ok(()),
                    encoded_bytes: probe_bytes,
                })
                .expect("valid completion");
        }

        let mut clocks = ControllerClockTrackers::new();
        tracker
            .observe_report_decode(
                exact,
                report(run_id, TransportDiagnosticSequence::new(1), tokens[1], 200),
                base.checked_add(Duration::from_millis(30))
                    .expect("small duration"),
                &mut clocks,
            )
            .expect("sequence one report");
        assert_eq!(tracker.missing_sequences(), [0]);

        tracker
            .observe_report_decode(
                exact,
                report(run_id, TransportDiagnosticSequence::new(0), tokens[0], 202),
                base.checked_add(Duration::from_millis(31))
                    .expect("small duration"),
                &mut clocks,
            )
            .expect("late sequence zero report");
        assert!(tracker.missing_sequences().is_empty());
        assert_eq!(tracker.reordered_reports, 1);
    }

    #[test]
    fn encoded_wire_lengths_are_derived_from_real_records_and_are_fixed_for_this_contract() {
        let lengths = encoded_diagnostic_record_lengths(
            exact_controller(),
            TransportDiagnosticRunId::try_new(3).expect("nonzero run ID"),
        )
        .expect("diagnostic records encode");
        assert_eq!(lengths, (MAX_UART_RECORD_BYTES, MAX_UART_RECORD_BYTES));
    }

    #[test]
    fn percentile_evidence_uses_nearest_rank_for_p50_p95_and_p99() {
        let values: Vec<u64> = (1..=100).collect();
        let distribution = DistributionU64::from_values(&values).expect("nonempty values");
        assert_eq!(distribution.p50, 50);
        assert_eq!(distribution.p95, 95);
        assert_eq!(distribution.p99, 99);
        assert_eq!(distribution.exact_sum_decimal, "5050");

        let signed = DistributionI64::from_values(&[-10, -5, 0, 5, 10]).expect("nonempty values");
        assert_eq!(signed.p50, 0);
        assert_eq!(signed.p95, 10);
        assert_eq!(signed.p99, 10);
    }

    #[test]
    fn periodic_liveness_includes_inter_message_and_trailing_gaps() {
        let admitted_at = Instant::now();
        let mut liveness =
            PeriodicLiveness::admitted("Heartbeat", Duration::from_millis(10), admitted_at);
        liveness
            .observe(
                admitted_at
                    .checked_add(Duration::from_millis(10))
                    .expect("small duration"),
            )
            .expect("gap equal to bound is accepted");
        assert!(matches!(
            liveness.observe_trailing_gap(
                admitted_at
                    .checked_add(Duration::from_millis(21))
                    .expect("small duration")
            ),
            Err(QualificationError::PeriodicLivenessGapExceeded {
                stream: "Heartbeat",
                ..
            })
        ));
    }

    #[test]
    fn runtime_liveness_uses_each_streams_own_period_domain() {
        let admitted_at = Instant::now();
        let mut admitted_hello = hello();
        admitted_hello.watchdog_nominal_period =
            WatchdogNominalPeriodMs::try_new(250).expect("firmware watchdog period");
        let liveness = RuntimeLiveness::from_admission(Admission {
            hello: admitted_hello,
            heartbeat: heartbeat(1),
            hello_received_at: admitted_at,
            heartbeat_received_at: admitted_at,
        })
        .expect("bounded liveness policy");

        assert_eq!(
            liveness.hello.maximum_gap,
            Duration::from_millis(u64::from(CANONICAL_CONTROLLER_HELLO_PERIOD_MS) * 2)
        );
        assert_eq!(liveness.heartbeat.maximum_gap, Duration::from_millis(250));
    }

    #[tokio::test]
    async fn writer_times_out_a_blocked_partial_record_and_reports_the_write_phase() {
        let (_reader, writer) = tokio::io::duplex(1);
        let (job_tx, job_rx) = mpsc::channel(1);
        let (completion_tx, mut completion_rx) = mpsc::channel(1);
        let maximum = SerialWriteTimeout(Duration::from_millis(7));
        let writer_task = tokio::spawn(run_writer(writer, job_rx, completion_tx, maximum));
        let now = Instant::now();
        let record = UartRecord::encode(Message::TransportDiagnosticProbe(
            TransportDiagnosticProbe::new(
                controller_uid(),
                boot_id(),
                TransportDiagnosticRunId::try_new(11).expect("nonzero run ID"),
                TransportDiagnosticSequence::new(0),
                HostElapsedNsToken::new(0),
            ),
        ))
        .expect("diagnostic probe encodes");
        job_tx
            .send(WriteJob {
                sequence: TransportDiagnosticSequence::new(0),
                scheduled_at: now,
                dispatched_at: now,
                record,
            })
            .await
            .expect("writer queue accepts one job");
        let completion = tokio::time::timeout(Duration::from_millis(100), completion_rx.recv())
            .await
            .expect("bounded writer completion")
            .expect("completion channel remains open");
        assert!(matches!(
            completion.result,
            Err(WriterFailure::Timeout {
                phase: WriterPhase::Write,
                maximum,
            }) if maximum == Duration::from_millis(7)
        ));
        drop(job_tx);
        writer_task.await.expect("writer task joins");
    }

    #[tokio::test]
    async fn one_serial_deadline_also_bounds_flush() {
        let error = write_and_flush_within(
            &mut FlushPendingWriter,
            b"complete-record",
            SerialWriteTimeout(Duration::from_millis(7)),
        )
        .await
        .expect_err("a flush that never wakes must time out");
        assert!(matches!(
            error,
            WriterFailure::Timeout {
                phase: WriterPhase::Flush,
                maximum,
            } if maximum == Duration::from_millis(7)
        ));
    }

    #[tokio::test]
    async fn bounded_join_cancels_a_writer_task_that_does_not_terminate() {
        let writer = AbortOnDrop::new(tokio::spawn(std::future::pending::<()>()));
        let error = writer
            .join_within(Duration::from_millis(1))
            .await
            .expect_err("pending writer must reach the join deadline");
        assert!(matches!(
            error,
            QualificationError::WriterJoinTimeout { maximum_ms: 1 }
        ));
    }
}
