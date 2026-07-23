//! Read-only commissioning probe for one exact KRP2 V2 controller UART.
//!
//! This binary never writes to the serial device. It waits for the controller's
//! periodic `ControllerHello`, parses it through the canonical bounded decoder,
//! and prints the observed software claims as JSON. Those claims are inputs to
//! commissioning; they are not proof of wiring, motor behavior, or safety.

use std::fmt;
use std::str::FromStr;
use std::time::Duration;

use clap::Parser;
use robot_protocol::v2::{
    ControllerHello, Message, NeutralOutput, OutputState, PhysicalStopSemantics, UartStreamDecoder,
    UartStreamError,
};
use serde_json::json;
use tokio::io::AsyncReadExt;
use tokio_serial::SerialPortBuilderExt;

const SERIAL_BAUD_BPS: u32 = 115_200;
const MAX_SERIAL_PATH_BYTES: usize = 512;
const MAX_PROBE_TIMEOUT_MS: u64 = 30_000;
const MAX_OBSERVED_BYTES: usize = 64 * 1_024;
const MAX_OBSERVED_RECORDS: usize = 1_024;
const SERIAL_BY_ID_PREFIX: &str = "/dev/serial/by-id/";

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

#[derive(Debug)]
enum ProbeError {
    Open(tokio_serial::Error),
    Exclusive(tokio_serial::Error),
    Read(std::io::Error),
    SerialEof,
    Timeout {
        timeout_ms: u64,
        observed_bytes: usize,
        observed_records: usize,
    },
    ByteBudgetExceeded {
        maximum_bytes: usize,
    },
    RecordBudgetExceeded {
        maximum_records: usize,
    },
    Decode(UartStreamError),
    HostDirectionMessageFromController,
    EncodeObservation(serde_json::Error),
}

impl fmt::Display for ProbeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("read-only V2 controller probe failed: ")?;
        match self {
            Self::Open(source) => write!(formatter, "could not open the exact serial device: {source}"),
            Self::Exclusive(source) => {
                write!(formatter, "could not acquire exclusive serial ownership: {source}")
            }
            Self::Read(source) => write!(formatter, "serial read failed: {source}"),
            Self::SerialEof => formatter.write_str("serial stream reached EOF"),
            Self::Timeout {
                timeout_ms,
                observed_bytes,
                observed_records,
            } => write!(
                formatter,
                "no ControllerHello arrived within {timeout_ms} ms after {observed_bytes} bytes and {observed_records} complete records"
            ),
            Self::ByteBudgetExceeded { maximum_bytes } => write!(
                formatter,
                "serial observation exceeded its {maximum_bytes}-byte budget"
            ),
            Self::RecordBudgetExceeded { maximum_records } => write!(
                formatter,
                "serial observation exceeded its {maximum_records}-record budget"
            ),
            Self::Decode(source) => write!(formatter, "canonical UART decode failed: {source}"),
            Self::HostDirectionMessageFromController => formatter.write_str(
                "the device emitted a frame kind reserved for the host direction",
            ),
            Self::EncodeObservation(source) => {
                write!(formatter, "could not encode the observation: {source}")
            }
        }
    }
}

impl std::error::Error for ProbeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Open(source) | Self::Exclusive(source) => Some(source),
            Self::Read(source) => Some(source),
            Self::Decode(source) => Some(source),
            Self::EncodeObservation(source) => Some(source),
            Self::SerialEof
            | Self::Timeout { .. }
            | Self::ByteBudgetExceeded { .. }
            | Self::RecordBudgetExceeded { .. }
            | Self::HostDirectionMessageFromController => None,
        }
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
        .open_native_async()
        .map_err(ProbeError::Open)?;
    port.set_exclusive(true).map_err(ProbeError::Exclusive)?;

    let deadline = tokio::time::Instant::now() + Duration::from_millis(cli.timeout_ms);
    let mut decoder = UartStreamDecoder::new();
    let mut buffer = [0_u8; 256];
    let mut observed_bytes = 0_usize;
    let mut observed_records = 0_usize;

    loop {
        let count = match tokio::time::timeout_at(deadline, port.read(&mut buffer)).await {
            Ok(Ok(0)) => return Err(ProbeError::SerialEof),
            Ok(Ok(count)) => count,
            Ok(Err(source)) => return Err(ProbeError::Read(source)),
            Err(_) => {
                return Err(ProbeError::Timeout {
                    timeout_ms: cli.timeout_ms,
                    observed_bytes,
                    observed_records,
                });
            }
        };
        observed_bytes =
            observed_bytes
                .checked_add(count)
                .ok_or(ProbeError::ByteBudgetExceeded {
                    maximum_bytes: MAX_OBSERVED_BYTES,
                })?;
        if observed_bytes > MAX_OBSERVED_BYTES {
            return Err(ProbeError::ByteBudgetExceeded {
                maximum_bytes: MAX_OBSERVED_BYTES,
            });
        }

        for byte in &buffer[..count] {
            let Some(decoded) = decoder.push(*byte) else {
                continue;
            };
            observed_records =
                observed_records
                    .checked_add(1)
                    .ok_or(ProbeError::RecordBudgetExceeded {
                        maximum_records: MAX_OBSERVED_RECORDS,
                    })?;
            if observed_records > MAX_OBSERVED_RECORDS {
                return Err(ProbeError::RecordBudgetExceeded {
                    maximum_records: MAX_OBSERVED_RECORDS,
                });
            }
            let message = decoded.map_err(ProbeError::Decode)?;
            match message {
                Message::ControllerHello(hello) => return Ok(hello),
                Message::ControllerReady(_)
                | Message::AppliedResult(_)
                | Message::Heartbeat(_)
                | Message::ObservationalOdometry(_)
                | Message::TransportDiagnosticReport(_)
                | Message::HostStopResult(_) => {}
                Message::AcquireControl(_)
                | Message::HostCommand(_)
                | Message::HostStop(_)
                | Message::StatusQuery(_)
                | Message::BeginSession(_)
                | Message::ApplyPwm(_)
                | Message::ForceStop(_)
                | Message::TransportDiagnosticProbe(_)
                | Message::AcquireResult(_)
                | Message::HostCommandResult(_)
                | Message::StatusReport(_) => {
                    return Err(ProbeError::HostDirectionMessageFromController);
                }
            }
        }
    }
}

fn write_observation(
    serial_device: &PersistentSerialPath,
    hello: ControllerHello,
) -> Result<(), ProbeError> {
    let observation = json!({
        "schema_version": 1,
        "observation_kind": "read_only_krp2_controller_hello",
        "serial_by_id_path": serial_device.as_str(),
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
        "evidence_boundary": "decoded software claim only; no serial bytes were transmitted and no physical behavior was observed"
    });
    let stdout = std::io::stdout();
    serde_json::to_writer_pretty(stdout.lock(), &observation)
        .map_err(ProbeError::EncodeObservation)?;
    println!();
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
