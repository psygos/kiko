//! Read-only commissioning probe for one exact KRP2 V2 controller UART.
//!
//! This binary never writes to the serial device. It waits for the controller's
//! periodic `ControllerHello`, parses it through the canonical bounded decoder,
//! and prints the observed software claims as JSON. Those claims are inputs to
//! commissioning; they are not proof of wiring, motor behavior, or safety.

use std::fmt;
use std::io::Write;
use std::str::FromStr;
use std::time::Duration;

use clap::Parser;
use robot_protocol::v2::{
    ControllerHello, Message, MessageKind, NeutralOutput, OutputState, PhysicalStopSemantics,
    UartStreamDecoder, UartStreamError,
};
use serde::Serialize;
use serde_json::json;
use tokio::io::AsyncReadExt;
use tokio_serial::{
    ClearBuffer, DataBits, FlowControl, Parity, SerialPort, SerialPortBuilderExt, StopBits,
};

const SERIAL_BAUD_BPS: u32 = 115_200;
const MAX_SERIAL_PATH_BYTES: usize = 512;
const MAX_PROBE_TIMEOUT_MS: u64 = 30_000;
const MAX_OBSERVED_BYTES: usize = 64 * 1_024;
const MAX_OBSERVED_RECORDS: usize = 1_024;
const FAILURE_TRACE_BYTES: usize = 8 * 1_024;
const FAILURE_TRACE_DELIMITERS: usize = 64;
const FAILURE_COMPLETION_MAX_BYTES: usize = 4 * 1_024;
const FAILURE_COMPLETION_TIMEOUT_MS: u64 = 250;
const SERIAL_BY_ID_PREFIX: &str = "/dev/serial/by-id/";
const FNV1A64_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV1A64_PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Parser, Debug)]
#[command(
    name = "robot-v2-identity-probe",
    about = "Read one KRP2 ControllerHello without transmitting serial bytes"
)]
struct Cli {
    /// Exact Linux persistent serial path. No ttyACM fallback or device scan.
    #[arg(long)]
    serial_device: PersistentSerialPath,
    /// Exclusive total observation deadline in milliseconds.
    #[arg(long, default_value_t = 5_000, value_parser = parse_timeout_ms)]
    timeout_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PersistentSerialPath(Box<str>);

impl PersistentSerialPath {
    fn as_str(&self) -> &str {
        &self.0
    }
}

impl FromStr for PersistentSerialPath {
    type Err = SerialPathError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.len() > MAX_SERIAL_PATH_BYTES {
            return Err(SerialPathError::TooLong {
                actual_bytes: value.len(),
                maximum_bytes: MAX_SERIAL_PATH_BYTES,
            });
        }
        let suffix = value
            .strip_prefix(SERIAL_BY_ID_PREFIX)
            .ok_or(SerialPathError::NotPersistentById)?;
        if suffix.is_empty() {
            return Err(SerialPathError::EmptyIdentity);
        }
        if suffix.contains('/') || matches!(suffix, "." | "..") {
            return Err(SerialPathError::NonCanonicalIdentity);
        }
        if let Some((index, byte)) = value
            .bytes()
            .enumerate()
            .find(|(_, byte)| !byte.is_ascii_graphic())
        {
            return Err(SerialPathError::NonGraphicAscii { index, byte });
        }
        Ok(Self(value.into()))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SerialPathError {
    TooLong {
        actual_bytes: usize,
        maximum_bytes: usize,
    },
    NotPersistentById,
    EmptyIdentity,
    NonCanonicalIdentity,
    NonGraphicAscii {
        index: usize,
        byte: u8,
    },
}

impl fmt::Display for SerialPathError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid persistent controller serial path: {self:?}"
        )
    }
}

impl std::error::Error for SerialPathError {}

fn parse_timeout_ms(value: &str) -> Result<u64, ProbeTimeoutError> {
    let timeout_ms = value
        .parse::<u64>()
        .map_err(|_| ProbeTimeoutError::NotUnsignedInteger)?;
    if timeout_ms == 0 || timeout_ms > MAX_PROBE_TIMEOUT_MS {
        return Err(ProbeTimeoutError::OutsideRange {
            actual_ms: timeout_ms,
            minimum_ms: 1,
            maximum_ms: MAX_PROBE_TIMEOUT_MS,
        });
    }
    Ok(timeout_ms)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProbeTimeoutError {
    NotUnsignedInteger,
    OutsideRange {
        actual_ms: u64,
        minimum_ms: u64,
        maximum_ms: u64,
    },
}

impl fmt::Display for ProbeTimeoutError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid probe timeout: {self:?}")
    }
}

impl std::error::Error for ProbeTimeoutError {}

enum ProbeError {
    Open(tokio_serial::Error),
    Exclusive(tokio_serial::Error),
    ClearPendingInput(tokio_serial::Error),
    Read(std::io::Error),
    SerialEof,
    Timeout {
        timeout_ms: u64,
        observed_bytes: usize,
        observed_records: usize,
        initial_record_boundary_observed: bool,
    },
    ByteBudgetReached {
        maximum_bytes: usize,
    },
    RecordBudgetExceeded {
        maximum_records: usize,
    },
    Decode {
        source: UartStreamError,
        wire: Box<WireFailureEvidence>,
        completion_read_source: Option<std::io::Error>,
    },
    UnexpectedControllerMessage {
        kind: MessageKind,
    },
    HostDirectionMessageFromController {
        kind: MessageKind,
    },
    EncodeObservation(serde_json::Error),
    WriteObservation {
        phase: ObservationWritePhase,
        source: std::io::Error,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ObservationWritePhase {
    Body,
    Terminator,
    Flush,
}

impl fmt::Display for ObservationWritePhase {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Body => formatter.write_str("JSON body"),
            Self::Terminator => formatter.write_str("JSON terminator"),
            Self::Flush => formatter.write_str("flush"),
        }
    }
}

impl fmt::Debug for ProbeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, formatter)
    }
}

impl fmt::Display for ProbeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("read-only V2 controller probe failed: ")?;
        match self {
            Self::Open(source) => write!(formatter, "could not open the exact serial device: {source}"),
            Self::Exclusive(source) => {
                write!(formatter, "could not acquire exclusive serial ownership: {source}")
            }
            Self::ClearPendingInput(source) => write!(
                formatter,
                "could not clear bytes pending in the host serial input queue: {source}"
            ),
            Self::Read(source) => write!(formatter, "serial read failed: {source}"),
            Self::SerialEof => formatter.write_str("serial stream reached EOF"),
            Self::Timeout {
                timeout_ms,
                observed_bytes,
                observed_records,
                initial_record_boundary_observed,
            } => write!(
                formatter,
                "no ControllerHello was accepted before the exclusive {timeout_ms} ms observation deadline after {observed_bytes} accepted bytes and {observed_records} post-boundary parser events (initial_record_boundary_observed={initial_record_boundary_observed})"
            ),
            Self::ByteBudgetReached { maximum_bytes } => write!(
                formatter,
                "serial observation reached its {maximum_bytes}-byte budget"
            ),
            Self::RecordBudgetExceeded { maximum_records } => write!(
                formatter,
                "serial observation exceeded its {maximum_records}-record budget"
            ),
            Self::Decode {
                source,
                wire,
                completion_read_source,
            } => {
                let typed_completion_evidence = completion_read_source
                    .as_ref()
                    .map(WireIoErrorEvidence::from_error);
                if typed_completion_evidence.as_ref()
                    != wire.failure_trace_completion_error.as_ref()
                {
                    return Err(fmt::Error);
                }
                write!(
                    formatter,
                    "canonical UART decode failed: {source}\nfailure_wire_evidence_json="
                )?;
                let encoded = serde_json::to_string(wire).map_err(|_| fmt::Error)?;
                formatter.write_str(&encoded)
            }
            Self::UnexpectedControllerMessage { kind } => {
                write!(formatter, "the device emitted unexpected controller message {kind:?}")
            }
            Self::HostDirectionMessageFromController { kind } => {
                write!(formatter, "the device emitted host-direction message {kind:?}")
            }
            Self::EncodeObservation(source) => {
                write!(formatter, "could not encode the observation: {source}")
            }
            Self::WriteObservation { phase, source } => {
                write!(formatter, "could not write observation {phase}: {source}")
            }
        }
    }
}

impl std::error::Error for ProbeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Open(source) | Self::Exclusive(source) | Self::ClearPendingInput(source) => {
                Some(source)
            }
            Self::Read(source) | Self::WriteObservation { source, .. } => Some(source),
            Self::Decode { source, .. } => Some(source),
            Self::EncodeObservation(source) => Some(source),
            Self::SerialEof
            | Self::Timeout { .. }
            | Self::ByteBudgetReached { .. }
            | Self::RecordBudgetExceeded { .. }
            | Self::UnexpectedControllerMessage { .. }
            | Self::HostDirectionMessageFromController { .. } => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct WireIoErrorEvidence {
    kind_debug: String,
    raw_os_error: Option<i32>,
    message: String,
}

impl WireIoErrorEvidence {
    fn from_error(source: &std::io::Error) -> Self {
        Self {
            kind_debug: format!("{:?}", source.kind()),
            raw_os_error: source.raw_os_error(),
            message: source.to_string(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum FailureTraceStopReason {
    TerminatingDelimiterObserved,
    FailureCompletionByteBudgetReached,
    FailureTraceCounterRegression,
    FailureCompletionDeadlineReached,
    OriginalProbeDeadlineReachedDuringFailureCompletion,
    GlobalObservationByteBudgetReached,
    SerialEofDuringFailureCompletion,
    SerialReadErrorDuringFailureCompletion,
    HostObservationCounterOverflow,
    DecodeFailureByteCompletedRecord,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct WireFailureEvidence {
    schema_version: u32,
    evidence_kind: &'static str,
    total_traced_bytes_after_host_input_clear_through_stop: usize,
    all_traced_bytes_fnv1a64_hex: String,
    initial_synchronization_delimiter_offset_zero_based: Option<usize>,
    retained_delimiter_offsets_zero_based: Vec<usize>,
    first_decode_failure_after_observed_byte_count: usize,
    nonzero_run_bytes_at_first_decode_failure: usize,
    offending_nonzero_run_bytes_if_terminated: Option<usize>,
    failure_trace_stop_reason: FailureTraceStopReason,
    failure_trace_completion_error: Option<WireIoErrorEvidence>,
    current_unterminated_nonzero_run_bytes: usize,
    maximum_completed_nonzero_run_bytes: usize,
    post_boundary_parser_events_including_failure: usize,
    retained_start_offset_zero_based: usize,
    retained_bytes_hex: String,
    evidence_boundary: &'static str,
}

struct WireTrace<const BYTES: usize, const DELIMITERS: usize> {
    bytes: [u8; BYTES],
    next_byte: usize,
    retained_bytes: usize,
    delimiter_offsets: [usize; DELIMITERS],
    next_delimiter: usize,
    retained_delimiters: usize,
    total_bytes: usize,
    current_nonzero_run: usize,
    maximum_completed_nonzero_run: usize,
    initial_synchronization_delimiter_offset: Option<usize>,
    fnv1a64: u64,
}

struct PendingDecodeFailure {
    source: UartStreamError,
    parser_events: usize,
    first_failure_after_observed_byte_count: usize,
    nonzero_run_bytes_at_first_failure: usize,
    completion_deadline: tokio::time::Instant,
    completion_deadline_stop_reason: FailureTraceStopReason,
}

#[derive(Clone, Copy)]
struct ProbeProgress {
    timeout_ms: u64,
    observed_bytes: usize,
    observed_records: usize,
    initial_record_boundary_observed: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FailureTraceAction {
    Continue,
    Finish {
        stop_reason: FailureTraceStopReason,
        offending_nonzero_run_bytes_if_terminated: Option<usize>,
    },
}

enum BoundedParserEvent<T> {
    Valid {
        value: T,
        event_index: usize,
    },
    StrictFailure {
        source: UartStreamError,
        event_index: usize,
    },
}

fn bounded_read_capacity(observed_bytes: usize, buffer_capacity: usize) -> Option<usize> {
    MAX_OBSERVED_BYTES
        .checked_sub(observed_bytes)
        .map(|remaining| remaining.min(buffer_capacity))
        .filter(|capacity| *capacity != 0)
}

fn failure_completion_deadline(
    now: tokio::time::Instant,
    original_deadline: tokio::time::Instant,
) -> (tokio::time::Instant, FailureTraceStopReason) {
    let completion_deadline = now + Duration::from_millis(FAILURE_COMPLETION_TIMEOUT_MS);
    if original_deadline <= completion_deadline {
        (
            original_deadline,
            FailureTraceStopReason::OriginalProbeDeadlineReachedDuringFailureCompletion,
        )
    } else {
        (
            completion_deadline,
            FailureTraceStopReason::FailureCompletionDeadlineReached,
        )
    }
}

fn deadline_has_elapsed(now: tokio::time::Instant, deadline: tokio::time::Instant) -> bool {
    now >= deadline
}

fn active_observation_deadline(
    original_deadline: tokio::time::Instant,
    pending_decode_failure: Option<&PendingDecodeFailure>,
) -> tokio::time::Instant {
    pending_decode_failure.map_or(original_deadline, |failure| failure.completion_deadline)
}

fn classify_parser_event<T>(
    observed_records: usize,
    decoded: Result<T, UartStreamError>,
) -> Result<BoundedParserEvent<T>, ProbeError> {
    let event_index = observed_records
        .checked_add(1)
        .ok_or(ProbeError::RecordBudgetExceeded {
            maximum_records: MAX_OBSERVED_RECORDS,
        })?;
    match decoded {
        Err(source) => Ok(BoundedParserEvent::StrictFailure {
            source,
            event_index,
        }),
        Ok(_) if event_index > MAX_OBSERVED_RECORDS => Err(ProbeError::RecordBudgetExceeded {
            maximum_records: MAX_OBSERVED_RECORDS,
        }),
        Ok(value) => Ok(BoundedParserEvent::Valid { value, event_index }),
    }
}

fn failure_trace_action_after_byte(
    completed_nonzero_run: Option<usize>,
    total_traced_bytes: usize,
    first_failure_after_observed_byte_count: usize,
) -> FailureTraceAction {
    if let Some(run_bytes) = completed_nonzero_run {
        return FailureTraceAction::Finish {
            stop_reason: FailureTraceStopReason::TerminatingDelimiterObserved,
            offending_nonzero_run_bytes_if_terminated: Some(run_bytes),
        };
    }
    let Some(additional_bytes) =
        total_traced_bytes.checked_sub(first_failure_after_observed_byte_count)
    else {
        return FailureTraceAction::Finish {
            stop_reason: FailureTraceStopReason::FailureTraceCounterRegression,
            offending_nonzero_run_bytes_if_terminated: None,
        };
    };
    if additional_bytes >= FAILURE_COMPLETION_MAX_BYTES {
        FailureTraceAction::Finish {
            stop_reason: FailureTraceStopReason::FailureCompletionByteBudgetReached,
            offending_nonzero_run_bytes_if_terminated: None,
        }
    } else {
        FailureTraceAction::Continue
    }
}

impl PendingDecodeFailure {
    fn finish<const BYTES: usize, const DELIMITERS: usize>(
        self,
        wire_trace: &WireTrace<BYTES, DELIMITERS>,
        stop_reason: FailureTraceStopReason,
        offending_nonzero_run_bytes_if_terminated: Option<usize>,
        completion_read_source: Option<std::io::Error>,
    ) -> ProbeError {
        let completion_error = completion_read_source
            .as_ref()
            .map(WireIoErrorEvidence::from_error);
        ProbeError::Decode {
            source: self.source,
            wire: Box::new(wire_trace.failure_evidence(
                self.parser_events,
                self.first_failure_after_observed_byte_count,
                self.nonzero_run_bytes_at_first_failure,
                offending_nonzero_run_bytes_if_terminated,
                stop_reason,
                completion_error,
            )),
            completion_read_source,
        }
    }
}

fn deadline_probe_error<const BYTES: usize, const DELIMITERS: usize>(
    timeout_ms: u64,
    observed_bytes: usize,
    observed_records: usize,
    initial_record_boundary_observed: bool,
    pending_decode_failure: Option<PendingDecodeFailure>,
    wire_trace: &WireTrace<BYTES, DELIMITERS>,
) -> ProbeError {
    if let Some(failure) = pending_decode_failure {
        let stop_reason = failure.completion_deadline_stop_reason;
        failure.finish(wire_trace, stop_reason, None, None)
    } else {
        ProbeError::Timeout {
            timeout_ms,
            observed_bytes,
            observed_records,
            initial_record_boundary_observed,
        }
    }
}

fn enforce_exclusive_deadline<const BYTES: usize, const DELIMITERS: usize>(
    now: tokio::time::Instant,
    deadline: tokio::time::Instant,
    progress: ProbeProgress,
    pending_decode_failure: Option<PendingDecodeFailure>,
    wire_trace: &WireTrace<BYTES, DELIMITERS>,
) -> Result<Option<PendingDecodeFailure>, ProbeError> {
    if deadline_has_elapsed(now, deadline) {
        Err(deadline_probe_error(
            progress.timeout_ms,
            progress.observed_bytes,
            progress.observed_records,
            progress.initial_record_boundary_observed,
            pending_decode_failure,
            wire_trace,
        ))
    } else {
        Ok(pending_decode_failure)
    }
}

impl<const BYTES: usize, const DELIMITERS: usize> WireTrace<BYTES, DELIMITERS> {
    const fn new() -> Self {
        Self {
            bytes: [0; BYTES],
            next_byte: 0,
            retained_bytes: 0,
            delimiter_offsets: [0; DELIMITERS],
            next_delimiter: 0,
            retained_delimiters: 0,
            total_bytes: 0,
            current_nonzero_run: 0,
            maximum_completed_nonzero_run: 0,
            initial_synchronization_delimiter_offset: None,
            fnv1a64: FNV1A64_OFFSET_BASIS,
        }
    }

    fn observe(&mut self, byte: u8) -> Option<usize> {
        let offset = self.total_bytes;
        self.total_bytes += 1;
        self.fnv1a64 ^= u64::from(byte);
        self.fnv1a64 = self.fnv1a64.wrapping_mul(FNV1A64_PRIME);

        if BYTES != 0 {
            self.bytes[self.next_byte] = byte;
            self.next_byte = (self.next_byte + 1) % BYTES;
            if self.retained_bytes < BYTES {
                self.retained_bytes += 1;
            }
        }

        if byte == 0 {
            let completed_nonzero_run = self.current_nonzero_run;
            self.maximum_completed_nonzero_run = self
                .maximum_completed_nonzero_run
                .max(self.current_nonzero_run);
            self.current_nonzero_run = 0;
            if DELIMITERS != 0 {
                self.delimiter_offsets[self.next_delimiter] = offset;
                self.next_delimiter = (self.next_delimiter + 1) % DELIMITERS;
                if self.retained_delimiters < DELIMITERS {
                    self.retained_delimiters += 1;
                }
            }
            Some(completed_nonzero_run)
        } else {
            self.current_nonzero_run += 1;
            None
        }
    }

    fn note_initial_synchronization_delimiter(&mut self) {
        self.initial_synchronization_delimiter_offset = self.total_bytes.checked_sub(1);
    }

    fn failure_evidence(
        &self,
        parser_events: usize,
        first_failure_after_observed_byte_count: usize,
        nonzero_run_bytes_at_first_failure: usize,
        offending_nonzero_run_bytes_if_terminated: Option<usize>,
        stop_reason: FailureTraceStopReason,
        completion_error: Option<WireIoErrorEvidence>,
    ) -> WireFailureEvidence {
        WireFailureEvidence {
            schema_version: 1,
            evidence_kind: "read_only_krp2_decode_failure_wire_trace",
            total_traced_bytes_after_host_input_clear_through_stop: self.total_bytes,
            all_traced_bytes_fnv1a64_hex: format!("{:016x}", self.fnv1a64),
            initial_synchronization_delimiter_offset_zero_based: self
                .initial_synchronization_delimiter_offset,
            retained_delimiter_offsets_zero_based: self.retained_delimiter_offsets(),
            first_decode_failure_after_observed_byte_count:
                first_failure_after_observed_byte_count,
            nonzero_run_bytes_at_first_decode_failure: nonzero_run_bytes_at_first_failure,
            offending_nonzero_run_bytes_if_terminated,
            failure_trace_stop_reason: stop_reason,
            failure_trace_completion_error: completion_error,
            current_unterminated_nonzero_run_bytes: self.current_nonzero_run,
            maximum_completed_nonzero_run_bytes: self.maximum_completed_nonzero_run,
            post_boundary_parser_events_including_failure: parser_events,
            retained_start_offset_zero_based: self.total_bytes - self.retained_bytes,
            retained_bytes_hex: encode_hex(&self.retained_bytes_in_order()),
            evidence_boundary: "failure-only host observation after one input-queue clear; after the first strict decode failure, collection remains read-only and continues only to the earliest of the next zero delimiter, 4096 additional bytes, 250 ms completion deadline, original probe deadline, global 65536-byte observation budget, EOF, read error, or checked counter failure; retained hex is a bounded suffix, FNV-1a is a non-cryptographic fingerprint of bytes traced through that stop, and no serial bytes were transmitted",
        }
    }

    fn retained_bytes_in_order(&self) -> Vec<u8> {
        if BYTES == 0 || self.retained_bytes == 0 {
            return Vec::new();
        }
        let start = if self.retained_bytes == BYTES {
            self.next_byte
        } else {
            0
        };
        (0..self.retained_bytes)
            .map(|index| self.bytes[(start + index) % BYTES])
            .collect()
    }

    fn retained_delimiter_offsets(&self) -> Vec<usize> {
        if DELIMITERS == 0 || self.retained_delimiters == 0 {
            return Vec::new();
        }
        let start = if self.retained_delimiters == DELIMITERS {
            self.next_delimiter
        } else {
            0
        };
        (0..self.retained_delimiters)
            .map(|index| self.delimiter_offsets[(start + index) % DELIMITERS])
            .collect()
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), ProbeError> {
    let cli = Cli::parse();
    let hello = observe_hello(&cli).await?;
    write_observation(&cli.serial_device, hello)
}

async fn observe_hello(cli: &Cli) -> Result<ControllerHello, ProbeError> {
    let mut port = tokio_serial::new(cli.serial_device.as_str(), SERIAL_BAUD_BPS)
        .data_bits(DataBits::Eight)
        .parity(Parity::None)
        .stop_bits(StopBits::One)
        .flow_control(FlowControl::None)
        .open_native_async()
        .map_err(ProbeError::Open)?;
    port.set_exclusive(true).map_err(ProbeError::Exclusive)?;
    port.clear(ClearBuffer::Input)
        .map_err(ProbeError::ClearPendingInput)?;

    let deadline = tokio::time::Instant::now() + Duration::from_millis(cli.timeout_ms);
    let mut decoder = UartStreamDecoder::new_at_unknown_record_offset();
    let mut buffer = [0_u8; 256];
    let mut observed_bytes = 0_usize;
    let mut observed_records = 0_usize;
    let mut wire_trace = WireTrace::<FAILURE_TRACE_BYTES, FAILURE_TRACE_DELIMITERS>::new();
    let mut pending_decode_failure: Option<PendingDecodeFailure> = None;

    loop {
        let read_deadline = active_observation_deadline(deadline, pending_decode_failure.as_ref());
        pending_decode_failure = enforce_exclusive_deadline(
            tokio::time::Instant::now(),
            read_deadline,
            ProbeProgress {
                timeout_ms: cli.timeout_ms,
                observed_bytes,
                observed_records,
                initial_record_boundary_observed: !decoder.is_waiting_for_initial_boundary(),
            },
            pending_decode_failure.take(),
            &wire_trace,
        )?;
        let Some(read_capacity) = bounded_read_capacity(observed_bytes, buffer.len()) else {
            if let Some(failure) = pending_decode_failure.take() {
                return Err(failure.finish(
                    &wire_trace,
                    FailureTraceStopReason::GlobalObservationByteBudgetReached,
                    None,
                    None,
                ));
            }
            return Err(ProbeError::ByteBudgetReached {
                maximum_bytes: MAX_OBSERVED_BYTES,
            });
        };
        let read_result =
            tokio::time::timeout_at(read_deadline, port.read(&mut buffer[..read_capacity])).await;
        pending_decode_failure = enforce_exclusive_deadline(
            tokio::time::Instant::now(),
            read_deadline,
            ProbeProgress {
                timeout_ms: cli.timeout_ms,
                observed_bytes,
                observed_records,
                initial_record_boundary_observed: !decoder.is_waiting_for_initial_boundary(),
            },
            pending_decode_failure.take(),
            &wire_trace,
        )?;
        let count = match read_result {
            Ok(Ok(0)) => {
                if let Some(failure) = pending_decode_failure.take() {
                    return Err(failure.finish(
                        &wire_trace,
                        FailureTraceStopReason::SerialEofDuringFailureCompletion,
                        None,
                        None,
                    ));
                }
                return Err(ProbeError::SerialEof);
            }
            Ok(Ok(count)) => count,
            Ok(Err(source)) => {
                if let Some(failure) = pending_decode_failure.take() {
                    return Err(failure.finish(
                        &wire_trace,
                        FailureTraceStopReason::SerialReadErrorDuringFailureCompletion,
                        None,
                        Some(source),
                    ));
                }
                return Err(ProbeError::Read(source));
            }
            Err(_) => {
                return Err(deadline_probe_error(
                    cli.timeout_ms,
                    observed_bytes,
                    observed_records,
                    !decoder.is_waiting_for_initial_boundary(),
                    pending_decode_failure.take(),
                    &wire_trace,
                ));
            }
        };

        for byte in &buffer[..count] {
            let byte_deadline =
                active_observation_deadline(deadline, pending_decode_failure.as_ref());
            pending_decode_failure = enforce_exclusive_deadline(
                tokio::time::Instant::now(),
                byte_deadline,
                ProbeProgress {
                    timeout_ms: cli.timeout_ms,
                    observed_bytes,
                    observed_records,
                    initial_record_boundary_observed: !decoder.is_waiting_for_initial_boundary(),
                },
                pending_decode_failure.take(),
                &wire_trace,
            )?;
            let Some(next_observed_bytes) = observed_bytes.checked_add(1) else {
                if let Some(failure) = pending_decode_failure.take() {
                    return Err(failure.finish(
                        &wire_trace,
                        FailureTraceStopReason::HostObservationCounterOverflow,
                        None,
                        None,
                    ));
                }
                return Err(ProbeError::ByteBudgetReached {
                    maximum_bytes: MAX_OBSERVED_BYTES,
                });
            };
            observed_bytes = next_observed_bytes;
            let completed_nonzero_run = wire_trace.observe(*byte);
            if pending_decode_failure.is_some() {
                let first_failure_after_observed_byte_count = match pending_decode_failure.as_ref()
                {
                    Some(failure) => failure.first_failure_after_observed_byte_count,
                    None => continue,
                };
                if let FailureTraceAction::Finish {
                    stop_reason,
                    offending_nonzero_run_bytes_if_terminated,
                } = failure_trace_action_after_byte(
                    completed_nonzero_run,
                    wire_trace.total_bytes,
                    first_failure_after_observed_byte_count,
                ) {
                    let commit_deadline =
                        active_observation_deadline(deadline, pending_decode_failure.as_ref());
                    pending_decode_failure = enforce_exclusive_deadline(
                        tokio::time::Instant::now(),
                        commit_deadline,
                        ProbeProgress {
                            timeout_ms: cli.timeout_ms,
                            observed_bytes,
                            observed_records,
                            initial_record_boundary_observed: !decoder
                                .is_waiting_for_initial_boundary(),
                        },
                        pending_decode_failure.take(),
                        &wire_trace,
                    )?;
                    if let Some(failure) = pending_decode_failure.take() {
                        return Err(failure.finish(
                            &wire_trace,
                            stop_reason,
                            offending_nonzero_run_bytes_if_terminated,
                            None,
                        ));
                    }
                }
                continue;
            }

            let was_waiting_for_initial_boundary = decoder.is_waiting_for_initial_boundary();
            let decoded = decoder.push(*byte);
            if was_waiting_for_initial_boundary && !decoder.is_waiting_for_initial_boundary() {
                wire_trace.note_initial_synchronization_delimiter();
            }
            let Some(decoded) = decoded else {
                continue;
            };
            let classified = match classify_parser_event(observed_records, decoded) {
                Ok(classified) => classified,
                Err(source) => {
                    let _no_pending_failure = enforce_exclusive_deadline(
                        tokio::time::Instant::now(),
                        deadline,
                        ProbeProgress {
                            timeout_ms: cli.timeout_ms,
                            observed_bytes,
                            observed_records,
                            initial_record_boundary_observed: !decoder
                                .is_waiting_for_initial_boundary(),
                        },
                        None,
                        &wire_trace,
                    )?;
                    return Err(source);
                }
            };
            let message = match classified {
                BoundedParserEvent::Valid { value, event_index } => {
                    observed_records = event_index;
                    value
                }
                BoundedParserEvent::StrictFailure {
                    source,
                    event_index,
                } => {
                    let first_failure_after_observed_byte_count = wire_trace.total_bytes;
                    let nonzero_run_bytes_at_first_failure =
                        completed_nonzero_run.unwrap_or(wire_trace.current_nonzero_run);
                    if matches!(source, UartStreamError::OversizedRecord { .. }) {
                        let (completion_deadline, completion_deadline_stop_reason) =
                            failure_completion_deadline(tokio::time::Instant::now(), deadline);
                        pending_decode_failure = Some(PendingDecodeFailure {
                            source,
                            parser_events: event_index,
                            first_failure_after_observed_byte_count,
                            nonzero_run_bytes_at_first_failure,
                            completion_deadline,
                            completion_deadline_stop_reason,
                        });
                        pending_decode_failure = enforce_exclusive_deadline(
                            tokio::time::Instant::now(),
                            completion_deadline,
                            ProbeProgress {
                                timeout_ms: cli.timeout_ms,
                                observed_bytes,
                                observed_records,
                                initial_record_boundary_observed: !decoder
                                    .is_waiting_for_initial_boundary(),
                            },
                            pending_decode_failure.take(),
                            &wire_trace,
                        )?;
                        continue;
                    }
                    let _no_pending_failure = enforce_exclusive_deadline(
                        tokio::time::Instant::now(),
                        deadline,
                        ProbeProgress {
                            timeout_ms: cli.timeout_ms,
                            observed_bytes,
                            observed_records,
                            initial_record_boundary_observed: !decoder
                                .is_waiting_for_initial_boundary(),
                        },
                        None,
                        &wire_trace,
                    )?;
                    return Err(ProbeError::Decode {
                        source,
                        wire: Box::new(wire_trace.failure_evidence(
                            event_index,
                            first_failure_after_observed_byte_count,
                            nonzero_run_bytes_at_first_failure,
                            completed_nonzero_run,
                            FailureTraceStopReason::DecodeFailureByteCompletedRecord,
                            None,
                        )),
                        completion_read_source: None,
                    });
                }
            };
            let message_deadline =
                active_observation_deadline(deadline, pending_decode_failure.as_ref());
            let _no_pending_failure = enforce_exclusive_deadline(
                tokio::time::Instant::now(),
                message_deadline,
                ProbeProgress {
                    timeout_ms: cli.timeout_ms,
                    observed_bytes,
                    observed_records,
                    initial_record_boundary_observed: !decoder.is_waiting_for_initial_boundary(),
                },
                None,
                &wire_trace,
            )?;
            match message {
                Message::ControllerHello(hello) => {
                    let _no_pending_failure = enforce_exclusive_deadline(
                        tokio::time::Instant::now(),
                        deadline,
                        ProbeProgress {
                            timeout_ms: cli.timeout_ms,
                            observed_bytes,
                            observed_records,
                            initial_record_boundary_observed: !decoder
                                .is_waiting_for_initial_boundary(),
                        },
                        None,
                        &wire_trace,
                    )?;
                    return Ok(hello);
                }
                Message::ControllerReady(_)
                | Message::AppliedResult(_)
                | Message::HostStopResult(_)
                | Message::AcquireResult(_)
                | Message::HostCommandResult(_)
                | Message::StatusReport(_) => {
                    return Err(ProbeError::UnexpectedControllerMessage {
                        kind: message.kind(),
                    });
                }
                Message::Heartbeat(_)
                | Message::ObservationalOdometry(_)
                | Message::TransportDiagnosticReport(_) => {}
                value @ (Message::AcquireControl(_)
                | Message::HostCommand(_)
                | Message::HostStop(_)
                | Message::StatusQuery(_)
                | Message::BeginSession(_)
                | Message::ApplyPwm(_)
                | Message::ForceStop(_)
                | Message::TransportDiagnosticProbe(_)) => {
                    return Err(ProbeError::HostDirectionMessageFromController {
                        kind: value.kind(),
                    });
                }
            }
        }
        let chunk_deadline = active_observation_deadline(deadline, pending_decode_failure.as_ref());
        pending_decode_failure = enforce_exclusive_deadline(
            tokio::time::Instant::now(),
            chunk_deadline,
            ProbeProgress {
                timeout_ms: cli.timeout_ms,
                observed_bytes,
                observed_records,
                initial_record_boundary_observed: !decoder.is_waiting_for_initial_boundary(),
            },
            pending_decode_failure.take(),
            &wire_trace,
        )?;
        if observed_bytes == MAX_OBSERVED_BYTES {
            if let Some(failure) = pending_decode_failure.take() {
                return Err(failure.finish(
                    &wire_trace,
                    FailureTraceStopReason::GlobalObservationByteBudgetReached,
                    None,
                    None,
                ));
            }
            return Err(ProbeError::ByteBudgetReached {
                maximum_bytes: MAX_OBSERVED_BYTES,
            });
        }
    }
}

fn write_observation(
    serial_device: &PersistentSerialPath,
    hello: ControllerHello,
) -> Result<(), ProbeError> {
    let observation = json!({
        "schema_version": 2,
        "observation_kind": "read_only_krp2_controller_hello",
        "serial_by_id_path": serial_device.as_str(),
        "host_input_queue_cleared_before_observation": true,
        "initial_unknown_record_prefix_excluded": true,
        "controller_uid_hex": encode_hex(hello.controller_uid.as_bytes()),
        "observed_boot_id": hello.boot_id.get(),
        "firmware_abi": hello.firmware_abi,
        "firmware_build_id": hello.firmware_build_id,
        "actuator_config_fingerprint_hex": encode_hex(hello.actuator_config_fingerprint.as_bytes()),
        "capabilities_bits": hello.capabilities.bits(),
        "supports_required_safety_capabilities": hello.capabilities.supports_required_safety(),
        "maximum_absolute_pwm_percent": hello.max_abs_pwm_percent.get(),
        "grants_motion_authority": hello.max_abs_pwm_percent.grants_motion_authority(),
        "maximum_command_lease_ms": hello.max_command_lease.get(),
        "reported_output_state": output_state_name(hello.output_state),
        "reported_output_state_is_safe": hello.output_state.is_safe(),
        "pwm_frequency_hz": hello.pwm_frequency.get(),
        "watchdog_nominal_period_ms": hello.watchdog_nominal_period.get(),
        "neutral_output": neutral_output_name(hello.neutral_output),
        "physical_stop_semantics": stop_semantics_name(hello.physical_stop_semantics),
        "evidence_boundary": "the host input queue was cleared once; subsequently delivered bytes through the first zero delimiter were excluded, including any upstream or in-flight bytes delivered after that clear; the result is one decoded software claim, no serial bytes were transmitted, and no physical behavior was observed"
    });
    let stdout = std::io::stdout();
    let mut stdout = stdout.lock();
    write_json_observation(&mut stdout, &observation)
}

fn write_json_observation<W: Write>(
    writer: &mut W,
    observation: &serde_json::Value,
) -> Result<(), ProbeError> {
    if let Err(source) = serde_json::to_writer_pretty(&mut *writer, observation) {
        if source.classify() == serde_json::error::Category::Io {
            return Err(ProbeError::WriteObservation {
                phase: ObservationWritePhase::Body,
                source: source.into(),
            });
        }
        return Err(ProbeError::EncodeObservation(source));
    }
    writer
        .write_all(b"\n")
        .map_err(|source| ProbeError::WriteObservation {
            phase: ObservationWritePhase::Terminator,
            source,
        })?;
    writer
        .flush()
        .map_err(|source| ProbeError::WriteObservation {
            phase: ObservationWritePhase::Flush,
            source,
        })?;
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct FailAfterJsonWriter {
        bytes: Vec<u8>,
        json_complete: bool,
    }

    struct FailAfterBytesWriter {
        bytes: Vec<u8>,
        remaining: usize,
    }

    #[derive(Default)]
    struct FlushFailWriter {
        bytes: Vec<u8>,
    }

    impl Write for FailAfterJsonWriter {
        fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
            if self.json_complete {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "synthetic closed output",
                ));
            }
            self.bytes.extend_from_slice(buffer);
            if buffer.last() == Some(&b'}') {
                self.json_complete = true;
            }
            Ok(buffer.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl Write for FailAfterBytesWriter {
        fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
            if self.remaining == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "synthetic body failure",
                ));
            }
            let accepted = self.remaining.min(buffer.len());
            self.bytes.extend_from_slice(&buffer[..accepted]);
            self.remaining -= accepted;
            Ok(accepted)
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl Write for FlushFailWriter {
        fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
            self.bytes.extend_from_slice(buffer);
            Ok(buffer.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Err(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "synthetic flush failure",
            ))
        }
    }

    fn pending_oversize_failure(
        first_failure_after_observed_byte_count: usize,
    ) -> PendingDecodeFailure {
        PendingDecodeFailure {
            source: UartStreamError::OversizedRecord { maximum: 73 },
            parser_events: 1,
            first_failure_after_observed_byte_count,
            nonzero_run_bytes_at_first_failure: 74,
            completion_deadline: tokio::time::Instant::now(),
            completion_deadline_stop_reason:
                FailureTraceStopReason::FailureCompletionDeadlineReached,
        }
    }

    #[test]
    fn failure_trace_preserves_bounded_order_and_exact_run_boundaries() {
        let mut trace = WireTrace::<4, 2>::new();
        assert_eq!(trace.observe(0), Some(0));
        trace.note_initial_synchronization_delimiter();
        assert_eq!(trace.observe(1), None);
        assert_eq!(trace.observe(2), None);
        assert_eq!(trace.observe(3), None);
        assert_eq!(trace.observe(0), Some(3));
        for byte in 4..=8 {
            assert_eq!(trace.observe(byte), None);
        }

        let evidence = trace.failure_evidence(
            1,
            10,
            5,
            None,
            FailureTraceStopReason::FailureCompletionByteBudgetReached,
            None,
        );
        assert_eq!(
            evidence.total_traced_bytes_after_host_input_clear_through_stop,
            10
        );
        assert_eq!(
            evidence.initial_synchronization_delimiter_offset_zero_based,
            Some(0)
        );
        assert_eq!(evidence.retained_delimiter_offsets_zero_based, [0, 4]);
        assert_eq!(evidence.first_decode_failure_after_observed_byte_count, 10);
        assert_eq!(evidence.nonzero_run_bytes_at_first_decode_failure, 5);
        assert_eq!(evidence.current_unterminated_nonzero_run_bytes, 5);
        assert_eq!(evidence.maximum_completed_nonzero_run_bytes, 3);
        assert_eq!(evidence.retained_start_offset_zero_based, 6);
        assert_eq!(evidence.retained_bytes_hex, "05060708");
        assert_eq!(evidence.all_traced_bytes_fnv1a64_hex.len(), 16);
    }

    #[test]
    fn failure_trace_retains_only_the_latest_delimiter_offsets() {
        let mut trace = WireTrace::<0, 2>::new();
        for byte in [0, 1, 0, 2, 0] {
            let _completed_run = trace.observe(byte);
        }
        let evidence = trace.failure_evidence(
            0,
            5,
            0,
            Some(1),
            FailureTraceStopReason::TerminatingDelimiterObserved,
            None,
        );
        assert_eq!(evidence.retained_delimiter_offsets_zero_based, [2, 4]);
        assert_eq!(evidence.retained_start_offset_zero_based, 5);
        assert!(evidence.retained_bytes_hex.is_empty());
    }

    #[test]
    fn pending_oversize_failure_retains_the_complete_run_through_delimiter() {
        let mut trace = WireTrace::<FAILURE_TRACE_BYTES, 4>::new();
        assert_eq!(trace.observe(0), Some(0));
        trace.note_initial_synchronization_delimiter();
        for _ in 0..74 {
            assert_eq!(trace.observe(1), None);
        }
        let pending = pending_oversize_failure(75);
        for _ in 0..(FAILURE_COMPLETION_MAX_BYTES - 1) {
            let completed_run = trace.observe(2);
            assert_eq!(
                failure_trace_action_after_byte(completed_run, trace.total_bytes, 75),
                FailureTraceAction::Continue
            );
        }
        let completed_run = trace.observe(0);
        assert_eq!(
            failure_trace_action_after_byte(completed_run, trace.total_bytes, 75),
            FailureTraceAction::Finish {
                stop_reason: FailureTraceStopReason::TerminatingDelimiterObserved,
                offending_nonzero_run_bytes_if_terminated: Some(4_169),
            }
        );
        let error = pending.finish(
            &trace,
            FailureTraceStopReason::TerminatingDelimiterObserved,
            completed_run,
            None,
        );

        let ProbeError::Decode { source, wire, .. } = error else {
            panic!("pending decode failure must remain the primary error");
        };
        assert!(matches!(
            source,
            UartStreamError::OversizedRecord { maximum: 73 }
        ));
        assert_eq!(wire.nonzero_run_bytes_at_first_decode_failure, 74);
        assert_eq!(wire.offending_nonzero_run_bytes_if_terminated, Some(4_169));
        assert_eq!(wire.current_unterminated_nonzero_run_bytes, 0);
        assert_eq!(wire.maximum_completed_nonzero_run_bytes, 4_169);
        assert_eq!(wire.retained_delimiter_offsets_zero_based, [0, 4_170]);
        assert_eq!(wire.retained_start_offset_zero_based, 0);
        assert_eq!(wire.retained_bytes_hex.len(), 2 * 4_171);
        assert_eq!(
            wire.failure_trace_stop_reason,
            FailureTraceStopReason::TerminatingDelimiterObserved
        );
    }

    #[test]
    fn canonical_decoder_oversize_transitions_to_trace_only_until_delimiter() {
        let mut decoder = UartStreamDecoder::new_at_unknown_record_offset();
        let mut trace = WireTrace::<FAILURE_TRACE_BYTES, 4>::new();

        let initial_delimiter = 0;
        let completed_run = trace.observe(initial_delimiter);
        assert_eq!(completed_run, Some(0));
        assert_eq!(decoder.push(initial_delimiter), None);
        trace.note_initial_synchronization_delimiter();

        let mut strict_failure = None;
        for _ in 0..74 {
            let completed_run = trace.observe(1);
            assert_eq!(completed_run, None);
            if let Some(decoded) = decoder.push(1) {
                strict_failure = Some(decoded);
                break;
            }
        }
        let source = strict_failure
            .expect("the 74th nonzero byte must produce a decoder event")
            .expect_err("the event must be the strict oversize failure");
        assert!(matches!(
            source,
            UartStreamError::OversizedRecord { maximum: 73 }
        ));
        let pending = PendingDecodeFailure {
            source,
            parser_events: 1,
            first_failure_after_observed_byte_count: trace.total_bytes,
            nonzero_run_bytes_at_first_failure: trace.current_nonzero_run,
            completion_deadline: tokio::time::Instant::now(),
            completion_deadline_stop_reason:
                FailureTraceStopReason::FailureCompletionDeadlineReached,
        };

        let completed_run = trace.observe(0);
        let action = failure_trace_action_after_byte(
            completed_run,
            trace.total_bytes,
            pending.first_failure_after_observed_byte_count,
        );
        assert_eq!(
            action,
            FailureTraceAction::Finish {
                stop_reason: FailureTraceStopReason::TerminatingDelimiterObserved,
                offending_nonzero_run_bytes_if_terminated: Some(74),
            }
        );
        let error = pending.finish(
            &trace,
            FailureTraceStopReason::TerminatingDelimiterObserved,
            completed_run,
            None,
        );
        assert!(matches!(
            error,
            ProbeError::Decode {
                source: UartStreamError::OversizedRecord { maximum: 73 },
                ..
            }
        ));
    }

    #[test]
    fn failure_completion_bound_is_exact_and_delimiter_has_precedence() {
        let failure_at = 75;
        assert_eq!(
            failure_trace_action_after_byte(None, failure_at, failure_at),
            FailureTraceAction::Continue
        );
        assert_eq!(
            failure_trace_action_after_byte(
                None,
                failure_at + FAILURE_COMPLETION_MAX_BYTES - 1,
                failure_at,
            ),
            FailureTraceAction::Continue
        );
        assert_eq!(
            failure_trace_action_after_byte(
                None,
                failure_at + FAILURE_COMPLETION_MAX_BYTES,
                failure_at,
            ),
            FailureTraceAction::Finish {
                stop_reason: FailureTraceStopReason::FailureCompletionByteBudgetReached,
                offending_nonzero_run_bytes_if_terminated: None,
            }
        );
        assert_eq!(
            failure_trace_action_after_byte(
                Some(4_169),
                failure_at + FAILURE_COMPLETION_MAX_BYTES,
                failure_at,
            ),
            FailureTraceAction::Finish {
                stop_reason: FailureTraceStopReason::TerminatingDelimiterObserved,
                offending_nonzero_run_bytes_if_terminated: Some(4_169),
            }
        );
        assert_eq!(
            failure_trace_action_after_byte(None, failure_at - 1, failure_at),
            FailureTraceAction::Finish {
                stop_reason: FailureTraceStopReason::FailureTraceCounterRegression,
                offending_nonzero_run_bytes_if_terminated: None,
            }
        );
    }

    #[test]
    fn production_trace_ring_retains_the_maximum_unterminated_failure_run() {
        let mut trace = WireTrace::<FAILURE_TRACE_BYTES, 4>::new();
        assert_eq!(trace.observe(0), Some(0));
        for _ in 0..74 {
            assert_eq!(trace.observe(1), None);
        }
        let mut action = FailureTraceAction::Continue;
        for _ in 0..FAILURE_COMPLETION_MAX_BYTES {
            let completed_run = trace.observe(2);
            action = failure_trace_action_after_byte(completed_run, trace.total_bytes, 75);
        }
        assert_eq!(
            action,
            FailureTraceAction::Finish {
                stop_reason: FailureTraceStopReason::FailureCompletionByteBudgetReached,
                offending_nonzero_run_bytes_if_terminated: None,
            }
        );
        let evidence = trace.failure_evidence(
            1,
            75,
            74,
            None,
            FailureTraceStopReason::FailureCompletionByteBudgetReached,
            None,
        );
        assert_eq!(evidence.current_unterminated_nonzero_run_bytes, 4_170);
        assert_eq!(evidence.retained_start_offset_zero_based, 0);
        assert_eq!(evidence.retained_bytes_hex.len(), 2 * 4_171);
    }

    #[test]
    fn reads_are_limited_before_the_global_observation_boundary() {
        assert_eq!(bounded_read_capacity(0, 256), Some(256));
        assert_eq!(bounded_read_capacity(MAX_OBSERVED_BYTES - 1, 256), Some(1));
        assert_eq!(bounded_read_capacity(MAX_OBSERVED_BYTES, 256), None);
        assert_eq!(bounded_read_capacity(MAX_OBSERVED_BYTES + 1, 256), None);
        assert_eq!(bounded_read_capacity(0, 0), None);
    }

    #[test]
    fn failure_completion_deadline_reports_the_bound_that_wins() {
        let now = tokio::time::Instant::now();
        for (original_offset_ms, expected_offset_ms, expected_reason) in [
            (
                249,
                249,
                FailureTraceStopReason::OriginalProbeDeadlineReachedDuringFailureCompletion,
            ),
            (
                250,
                250,
                FailureTraceStopReason::OriginalProbeDeadlineReachedDuringFailureCompletion,
            ),
            (
                251,
                250,
                FailureTraceStopReason::FailureCompletionDeadlineReached,
            ),
        ] {
            let (selected, reason) =
                failure_completion_deadline(now, now + Duration::from_millis(original_offset_ms));
            assert_eq!(selected, now + Duration::from_millis(expected_offset_ms));
            assert_eq!(reason, expected_reason);
        }
        let deadline = now + Duration::from_millis(1);
        assert!(!deadline_has_elapsed(now, deadline));
        assert!(deadline_has_elapsed(deadline, deadline));
        assert!(deadline_has_elapsed(
            deadline + Duration::from_nanos(1),
            deadline
        ));
    }

    #[test]
    fn exclusive_deadline_guard_rejects_terminal_commit_at_equality() {
        let now = tokio::time::Instant::now();
        let deadline = now + Duration::from_millis(1);
        let progress = ProbeProgress {
            timeout_ms: 1,
            observed_bytes: 7,
            observed_records: 2,
            initial_record_boundary_observed: true,
        };
        let trace = WireTrace::<8, 2>::new();
        let before = enforce_exclusive_deadline(now, deadline, progress, None, &trace);
        assert!(matches!(before, Ok(None)));

        let at_deadline = enforce_exclusive_deadline(deadline, deadline, progress, None, &trace);
        assert!(matches!(
            at_deadline,
            Err(ProbeError::Timeout {
                timeout_ms: 1,
                observed_bytes: 7,
                observed_records: 2,
                initial_record_boundary_observed: true,
            })
        ));

        let pending_at_deadline = enforce_exclusive_deadline(
            deadline,
            deadline,
            progress,
            Some(pending_oversize_failure(4)),
            &trace,
        );
        assert!(matches!(
            pending_at_deadline,
            Err(ProbeError::Decode {
                source: UartStreamError::OversizedRecord { maximum: 73 },
                ..
            })
        ));
    }

    #[test]
    fn strict_parser_failure_precedes_the_valid_record_budget() {
        let event = classify_parser_event::<()>(
            MAX_OBSERVED_RECORDS,
            Err(UartStreamError::OversizedRecord { maximum: 73 }),
        )
        .expect("the strict decoder error must remain observable");
        assert!(matches!(
            event,
            BoundedParserEvent::StrictFailure {
                source: UartStreamError::OversizedRecord { maximum: 73 },
                event_index,
            } if event_index == MAX_OBSERVED_RECORDS + 1
        ));
        assert!(matches!(
            classify_parser_event(MAX_OBSERVED_RECORDS, Ok(())),
            Err(ProbeError::RecordBudgetExceeded {
                maximum_records: MAX_OBSERVED_RECORDS
            })
        ));
    }

    #[test]
    fn bounded_completion_stop_conditions_preserve_the_original_decode_error() {
        let mut trace = WireTrace::<128, 4>::new();
        for byte in [0, 1, 2, 3] {
            let _completed_run = trace.observe(byte);
        }
        for stop_reason in [
            FailureTraceStopReason::FailureCompletionDeadlineReached,
            FailureTraceStopReason::SerialEofDuringFailureCompletion,
        ] {
            let error = pending_oversize_failure(4).finish(&trace, stop_reason, None, None);
            let ProbeError::Decode {
                source,
                wire,
                completion_read_source,
            } = error
            else {
                panic!("completion stop must preserve the strict decoder error");
            };
            assert!(matches!(
                source,
                UartStreamError::OversizedRecord { maximum: 73 }
            ));
            assert_eq!(wire.failure_trace_stop_reason, stop_reason);
            assert_eq!(wire.failure_trace_completion_error, None);
            assert!(completion_read_source.is_none());
        }

        let read_source = std::io::Error::new(
            std::io::ErrorKind::ConnectionReset,
            "synthetic read failure",
        );
        let error = pending_oversize_failure(4).finish(
            &trace,
            FailureTraceStopReason::SerialReadErrorDuringFailureCompletion,
            None,
            Some(read_source),
        );
        let ProbeError::Decode {
            source,
            wire,
            completion_read_source: Some(completion_read_source),
        } = error
        else {
            panic!("read completion failure must remain typed");
        };
        assert!(matches!(
            source,
            UartStreamError::OversizedRecord { maximum: 73 }
        ));
        assert_eq!(
            completion_read_source.kind(),
            std::io::ErrorKind::ConnectionReset
        );
        let wire_error = wire
            .failure_trace_completion_error
            .expect("structured read error evidence");
        assert_eq!(wire_error.kind_debug, "ConnectionReset");
        assert_eq!(wire_error.raw_os_error, None);
        assert_eq!(wire_error.message, "synthetic read failure");
    }

    #[test]
    fn decode_failure_debug_output_contains_machine_parseable_wire_json() {
        let mut trace = WireTrace::<8, 2>::new();
        for byte in [0, 1, 2, 3] {
            let _completed_run = trace.observe(byte);
        }
        let error = ProbeError::Decode {
            source: UartStreamError::OversizedRecord { maximum: 73 },
            wire: Box::new(trace.failure_evidence(
                1,
                4,
                3,
                None,
                FailureTraceStopReason::FailureCompletionDeadlineReached,
                None,
            )),
            completion_read_source: None,
        };
        let rendered = format!("{error:?}");
        let (_, json) = rendered
            .split_once("\nfailure_wire_evidence_json=")
            .expect("structured failure marker");
        let value: serde_json::Value =
            serde_json::from_str(json).expect("failure evidence is valid JSON");
        assert_eq!(value["schema_version"], 1);
        assert_eq!(value["first_decode_failure_after_observed_byte_count"], 4);
        assert_eq!(
            value["failure_trace_stop_reason"],
            "failure_completion_deadline_reached"
        );
    }

    #[test]
    fn output_terminator_failure_is_typed_instead_of_panicking() {
        let mut writer = FailAfterJsonWriter::default();
        let error = write_json_observation(&mut writer, &json!({"ok": true}))
            .expect_err("the synthetic writer must reject the final newline");
        assert!(matches!(
            error,
            ProbeError::WriteObservation {
                phase: ObservationWritePhase::Terminator,
                source,
            } if source.kind() == std::io::ErrorKind::BrokenPipe
        ));
        assert!(writer.bytes.ends_with(b"}"));
    }

    #[test]
    fn output_body_io_failure_remains_typed() {
        let mut writer = FailAfterBytesWriter {
            bytes: Vec::new(),
            remaining: 4,
        };
        let error = write_json_observation(&mut writer, &json!({"ok": true}))
            .expect_err("the synthetic writer must fail within the JSON body");
        assert!(matches!(
            error,
            ProbeError::WriteObservation {
                phase: ObservationWritePhase::Body,
                source,
            } if source.kind() == std::io::ErrorKind::BrokenPipe
        ));
    }

    #[test]
    fn output_flush_failure_is_typed() {
        let mut writer = FlushFailWriter::default();
        let error = write_json_observation(&mut writer, &json!({"ok": true}))
            .expect_err("the synthetic writer must reject the explicit flush");
        assert!(matches!(
            error,
            ProbeError::WriteObservation {
                phase: ObservationWritePhase::Flush,
                source,
            } if source.kind() == std::io::ErrorKind::BrokenPipe
        ));
        assert!(writer.bytes.ends_with(b"}\n"));
    }

    #[test]
    fn controller_message_errors_preserve_the_actual_kind() {
        let unexpected = ProbeError::UnexpectedControllerMessage {
            kind: MessageKind::AcquireResult,
        };
        assert!(format!("{unexpected}").contains("AcquireResult"));
        let wrong_direction = ProbeError::HostDirectionMessageFromController {
            kind: MessageKind::BeginSession,
        };
        assert!(format!("{wrong_direction}").contains("BeginSession"));
    }

    #[test]
    fn serial_path_requires_one_persistent_identity_component() {
        assert!(
            "/dev/serial/by-id/usb-STMicroelectronics_STM32_STLink_123-if02"
                .parse::<PersistentSerialPath>()
                .is_ok()
        );
        for invalid in [
            "/dev/ttyACM1",
            "/dev/serial/by-id/",
            "/dev/serial/by-id/../ttyACM1",
            "/dev/serial/by-id/a/b",
            "/dev/serial/by-id/a b",
        ] {
            assert!(
                invalid.parse::<PersistentSerialPath>().is_err(),
                "{invalid}"
            );
        }
    }

    #[test]
    fn timeout_is_strictly_bounded() {
        assert_eq!(parse_timeout_ms("1"), Ok(1));
        assert_eq!(
            parse_timeout_ms(&MAX_PROBE_TIMEOUT_MS.to_string()),
            Ok(MAX_PROBE_TIMEOUT_MS)
        );
        assert!(parse_timeout_ms("0").is_err());
        assert!(parse_timeout_ms(&(MAX_PROBE_TIMEOUT_MS + 1).to_string()).is_err());
        assert!(parse_timeout_ms("-1").is_err());
    }

    #[test]
    fn timeout_text_describes_exclusive_acceptance_not_inferred_arrival() {
        let rendered = ProbeError::Timeout {
            timeout_ms: 5,
            observed_bytes: 7,
            observed_records: 2,
            initial_record_boundary_observed: true,
        }
        .to_string();
        assert!(rendered.contains("was accepted before the exclusive"));
        assert!(!rendered.contains("arrived"));
    }

    #[test]
    fn exact_hex_and_wire_enum_names_match_config_boundaries() {
        assert_eq!(encode_hex(&[0x00, 0xab, 0xff]), "00abff");
        assert_eq!(output_state_name(OutputState::ZeroPwm), "zero_pwm");
        assert_eq!(neutral_output_name(NeutralOutput::BothLow), "both_low");
        assert_eq!(
            stop_semantics_name(PhysicalStopSemantics::CoastVerified),
            "coast_verified"
        );
    }
}
