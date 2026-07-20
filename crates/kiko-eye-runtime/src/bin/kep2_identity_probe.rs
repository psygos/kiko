//! Identity-only commissioning CLI for one exact KEP2 eye controller.

use std::error::Error;
use std::ffi::OsString;
use std::fmt;
use std::io::Write;
use std::process::ExitCode;

use kiko_expression_runtime::REQUIRED_EYE_CAPABILITIES;
use kiko_eye_protocol::{Capabilities, PROTOCOL_VERSION};
use kiko_eye_runtime::{
    DeviceIdentityKind, EyeIdentityObservation, IdentityProbeConfig, IdentityProbeConfigError,
    IdentityProbeConfigInput, IdentityProbeError, SerialConfigurationEvidence,
    probe_serial_eye_identity,
};
use serde_json::json;

const DEFAULT_OPERATION_TIMEOUT_MS: u64 = 5_000;

#[derive(Debug, PartialEq, Eq)]
struct Cli {
    serial_device: String,
    timeout_ms: u64,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> ExitCode {
    let cli = match parse_cli(std::env::args_os().skip(1)) {
        Ok(cli) => cli,
        Err(CliParseError::HelpRequested) => {
            print_usage();
            return ExitCode::SUCCESS;
        }
        Err(source) => {
            eprintln!("error: {source}");
            print_usage();
            return ExitCode::from(2);
        }
    };

    match run(cli).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(source) => {
            print_error_chain(&source);
            ExitCode::FAILURE
        }
    }
}

async fn run(cli: Cli) -> Result<(), CliError> {
    let config = IdentityProbeConfig::parse(IdentityProbeConfigInput {
        device_path: cli.serial_device,
        operation_timeout_ms: cli.timeout_ms,
    })
    .map_err(CliError::Config)?;
    let (serial, observation) = probe_serial_eye_identity(&config)
        .await
        .map_err(CliError::Probe)?;
    write_observation(&config, &serial, observation)
}

fn parse_cli(arguments: impl IntoIterator<Item = OsString>) -> Result<Cli, CliParseError> {
    let mut arguments = arguments.into_iter();
    let mut serial_device = None;
    let mut timeout_ms = None;
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--help" | "-h") => return Err(CliParseError::HelpRequested),
            Some("--serial-device") => {
                if serial_device.is_some() {
                    return Err(CliParseError::DuplicateSerialDevice);
                }
                let value = arguments
                    .next()
                    .ok_or(CliParseError::MissingSerialDeviceValue)?;
                serial_device = Some(
                    value
                        .into_string()
                        .map_err(CliParseError::NonUnicodeSerialDevice)?,
                );
            }
            Some("--timeout-ms") => {
                if timeout_ms.is_some() {
                    return Err(CliParseError::DuplicateTimeout);
                }
                let value = arguments.next().ok_or(CliParseError::MissingTimeoutValue)?;
                let text = value
                    .to_str()
                    .ok_or_else(|| CliParseError::InvalidTimeout(value.clone()))?;
                if text.is_empty() || !text.bytes().all(|byte| byte.is_ascii_digit()) {
                    return Err(CliParseError::InvalidTimeout(value));
                }
                timeout_ms = Some(
                    text.parse::<u64>()
                        .map_err(|_| CliParseError::InvalidTimeout(value))?,
                );
            }
            _ => return Err(CliParseError::UnknownArgument(argument)),
        }
    }
    Ok(Cli {
        serial_device: serial_device.ok_or(CliParseError::MissingSerialDevice)?,
        timeout_ms: timeout_ms.unwrap_or(DEFAULT_OPERATION_TIMEOUT_MS),
    })
}

#[derive(Debug, PartialEq, Eq)]
enum CliParseError {
    HelpRequested,
    MissingSerialDevice,
    MissingSerialDeviceValue,
    NonUnicodeSerialDevice(OsString),
    DuplicateSerialDevice,
    MissingTimeoutValue,
    InvalidTimeout(OsString),
    DuplicateTimeout,
    UnknownArgument(OsString),
}

impl fmt::Display for CliParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HelpRequested => formatter.write_str("help requested"),
            Self::MissingSerialDevice => formatter.write_str("--serial-device is required"),
            Self::MissingSerialDeviceValue => {
                formatter.write_str("--serial-device requires a path")
            }
            Self::NonUnicodeSerialDevice(value) => {
                write!(formatter, "--serial-device is not valid Unicode: {value:?}")
            }
            Self::DuplicateSerialDevice => {
                formatter.write_str("--serial-device was provided more than once")
            }
            Self::MissingTimeoutValue => formatter.write_str("--timeout-ms requires a value"),
            Self::InvalidTimeout(value) => write!(
                formatter,
                "--timeout-ms must be an unsigned base-10 integer: {value:?}"
            ),
            Self::DuplicateTimeout => {
                formatter.write_str("--timeout-ms was provided more than once")
            }
            Self::UnknownArgument(argument) => write!(formatter, "unknown argument {argument:?}"),
        }
    }
}

impl Error for CliParseError {}

fn write_observation(
    config: &IdentityProbeConfig,
    serial: &SerialConfigurationEvidence,
    observation: EyeIdentityObservation,
) -> Result<(), CliError> {
    let report = observation.report();
    let capabilities_bits = report.capabilities.bits();
    let output = json!({
        "schema_version": 1,
        "observation_kind": "challenged_kep2_identity_report",
        "protocol_version": PROTOCOL_VERSION,
        "operation_timeout_ms": config.operation_timeout().get().as_millis(),
        "empty_delimiter_budget": config.empty_delimiter_budget(),
        "serial_device": {
            "path": config.device().path(),
            "stable_name": config.device().stable_name(),
            "kind": device_kind_name(config.device().kind()),
            "exclusive_owner_claimed": serial.exclusive_owner_claimed(),
            "baud_rate_bps_readback": serial.baud_rate_bps_readback(),
            "data_bits_8_readback": serial.data_bits_8_readback(),
            "parity_none_readback": serial.parity_none_readback(),
            "stop_bits_1_readback": serial.stop_bits_1_readback(),
            "flow_control_none_readback": serial.flow_control_none_readback(),
        },
        "challenge_nonce": observation.challenge().get(),
        "encoded_identity_query_bytes": observation.encoded_query_bytes(),
        "device_uid_hex": encode_hex(report.device_uid.as_bytes()),
        "device_uid_bytes": report.device_uid.as_bytes(),
        "firmware_build_id_hex": encode_hex(report.firmware_build_id.as_bytes()),
        "firmware_build_id_bytes": report.firmware_build_id.as_bytes(),
        "observed_boot_id": report.boot_id.get(),
        "device_uptime_ms": report.device_uptime.millis_since_boot(),
        "capabilities_bits": capabilities_bits,
        "required_host_capabilities_bits": REQUIRED_EYE_CAPABILITIES,
        "supports_required_host_capabilities":
            capabilities_bits & REQUIRED_EYE_CAPABILITIES == REQUIRED_EYE_CAPABILITIES,
        "capabilities": {
            "gaze": has_capability(capabilities_bits, Capabilities::GAZE),
            "lid": has_capability(capabilities_bits, Capabilities::LID),
            "pupil": has_capability(capabilities_bits, Capabilities::PUPIL),
            "color": has_capability(capabilities_bits, Capabilities::COLOR),
            "brightness": has_capability(capabilities_bits, Capabilities::BRIGHTNESS),
            "blink": has_capability(capabilities_bits, Capabilities::BLINK),
            "autonomous_fallback":
                has_capability(capabilities_bits, Capabilities::AUTONOMOUS_FALLBACK),
            "applied_report": has_capability(capabilities_bits, Capabilities::APPLIED_REPORT),
        },
        "evidence_boundary": "one fresh IdentityQuery challenge was transmitted and exactly echoed by one canonical IdentityReport; no control acquisition, expression intent, or release was transmitted; reported identity is a firmware claim, not optical or wiring evidence"
    });
    let stdout = std::io::stdout();
    let mut stdout = stdout.lock();
    serde_json::to_writer_pretty(&mut stdout, &output).map_err(CliError::EncodeOutput)?;
    writeln!(stdout).map_err(CliError::WriteOutput)?;
    Ok(())
}

const fn has_capability(actual: u32, capability: u32) -> bool {
    actual & capability == capability
}

const fn device_kind_name(kind: DeviceIdentityKind) -> &'static str {
    match kind {
        DeviceIdentityKind::LinuxById => "linux_by_id",
        DeviceIdentityKind::MacOsCallout => "macos_callout",
    }
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

#[derive(Debug)]
enum CliError {
    Config(IdentityProbeConfigError),
    Probe(IdentityProbeError),
    EncodeOutput(serde_json::Error),
    WriteOutput(std::io::Error),
}

impl fmt::Display for CliError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(source) => write!(formatter, "{source}"),
            Self::Probe(source) => write!(formatter, "{source}"),
            Self::EncodeOutput(source) => {
                write!(
                    formatter,
                    "could not encode KEP2 identity observation: {source}"
                )
            }
            Self::WriteOutput(source) => {
                write!(formatter, "could not finish KEP2 identity output: {source}")
            }
        }
    }
}

impl std::error::Error for CliError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Config(source) => Some(source),
            Self::Probe(source) => Some(source),
            Self::EncodeOutput(source) => Some(source),
            Self::WriteOutput(source) => Some(source),
        }
    }
}

fn print_usage() {
    eprintln!(
        "usage: kep2_identity_probe --serial-device <exact-stable-path> [--timeout-ms <1..=5000>]"
    );
}

fn print_error_chain(error: &(dyn Error + 'static)) {
    eprintln!("error: {error}");
    let mut source = error.source();
    while let Some(error) = source {
        eprintln!("caused by: {error}");
        source = error.source();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn os(values: &[&str]) -> Vec<OsString> {
        values.iter().map(OsString::from).collect()
    }

    #[test]
    fn cli_requires_one_exact_device_and_defaults_the_timeout() {
        assert_eq!(
            parse_cli(os(&["--serial-device", "/dev/cu.kiko-eyes"])),
            Ok(Cli {
                serial_device: "/dev/cu.kiko-eyes".to_owned(),
                timeout_ms: DEFAULT_OPERATION_TIMEOUT_MS,
            })
        );
        assert!(matches!(
            parse_cli(os(&[])),
            Err(CliParseError::MissingSerialDevice)
        ));
        assert!(matches!(
            parse_cli(os(&["--serial-device"])),
            Err(CliParseError::MissingSerialDeviceValue)
        ));
        assert!(matches!(
            parse_cli(os(&[
                "--serial-device",
                "/dev/cu.a",
                "--serial-device",
                "/dev/cu.b"
            ])),
            Err(CliParseError::DuplicateSerialDevice)
        ));
    }

    #[test]
    fn cli_parses_only_strict_decimal_timeout_and_rejects_unknowns() {
        assert_eq!(
            parse_cli(os(&[
                "--timeout-ms",
                "123",
                "--serial-device",
                "/dev/cu.kiko-eyes"
            ])),
            Ok(Cli {
                serial_device: "/dev/cu.kiko-eyes".to_owned(),
                timeout_ms: 123,
            })
        );
        for invalid in ["", "+1", "-1", "1.0", " 1", "1 "] {
            assert!(matches!(
                parse_cli(os(&[
                    "--serial-device",
                    "/dev/cu.kiko-eyes",
                    "--timeout-ms",
                    invalid
                ])),
                Err(CliParseError::InvalidTimeout(_))
            ));
        }
        assert!(matches!(
            parse_cli(os(&[
                "--serial-device",
                "/dev/cu.kiko-eyes",
                "--timeout-ms",
                "1",
                "--timeout-ms",
                "2"
            ])),
            Err(CliParseError::DuplicateTimeout)
        ));
        assert!(matches!(
            parse_cli(os(&["--serial-device", "/dev/cu.kiko-eyes", "--guess"])),
            Err(CliParseError::UnknownArgument(_))
        ));
    }

    #[test]
    fn output_helpers_are_exact_and_total() {
        assert_eq!(encode_hex(&[0x00, 0xab, 0xff]), "00abff");
        assert_eq!(
            device_kind_name(DeviceIdentityKind::LinuxById),
            "linux_by_id"
        );
        assert!(has_capability(
            Capabilities::KNOWN_BITS,
            Capabilities::BLINK
        ));
        assert!(!has_capability(Capabilities::GAZE, Capabilities::BLINK));
    }
}
