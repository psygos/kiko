//! Eye-only, finite KEP2 expression commissioning CLI.

use std::error::Error;
use std::ffi::OsString;
use std::fmt;
use std::io::Write;
use std::process::ExitCode;

use kiko_expression_runtime::PreparedEyeIntent;
use kiko_eye_protocol::{Expression, EyeIntent, PROTOCOL_VERSION};
use kiko_eye_runtime::{
    ActorExit, ActorTermination, COMMISSIONING_INTENT_LEASE_MS, COMMISSIONING_MAX_HOLD_MS,
    COMMISSIONING_STEP_COUNT, ClockError, CommissioningPrepareError, ConfigParseError,
    EyeActorHandle, EyeActorStartError, EyeActorTask, FirmwareAdmissionEvidence,
    HandleRequestError, MonotonicClock, OsEyeSessionMaterialError, OsEyeSessionMaterialGenerator,
    ReleaseReport, SerialConfigurationEvidence, StartupEvidence, StartupReceiptError,
    StaticEyeRuntimeConfig, StaticEyeRuntimeConfigInput, TokioClock, eye_commissioning_steps,
    start_serial_eye_actor,
};
use serde_json::{Value, json};

const BAUD_RATE_BPS: u32 = 115_200;
const RESPONSE_TIMEOUT_MS: u64 = 500;
const WRITE_TIMEOUT_MS: u64 = 100;
const WRITE_ATTEMPTS: u8 = 2;
const EMPTY_DELIMITER_BUDGET: u8 = 4;

const _: () = assert!(
    COMMISSIONING_MAX_HOLD_MS + RESPONSE_TIMEOUT_MS + WRITE_TIMEOUT_MS * (WRITE_ATTEMPTS as u64)
        < COMMISSIONING_INTENT_LEASE_MS as u64
);

#[derive(Debug, PartialEq, Eq)]
struct Cli {
    serial_device: String,
    expected_device_uid: [u8; 16],
    expected_firmware_build_id: [u8; 32],
    expected_capabilities_bits: u32,
}

#[derive(Debug)]
struct StepAdmission {
    name: &'static str,
    hold_duration_ms: u64,
    requested: PreparedEyeIntent,
    evidence: FirmwareAdmissionEvidence,
}

#[derive(Debug)]
struct RunEvidence {
    interrupted: bool,
    serial: SerialConfigurationEvidence,
    startup: StartupEvidence,
    steps: Vec<StepAdmission>,
    release: ReleaseReport,
    exit: ActorExit,
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

    match run(cli).await.and_then(write_evidence) {
        Ok(interrupted) if interrupted => ExitCode::from(130),
        Ok(_) => ExitCode::SUCCESS,
        Err(source) => {
            print_error_chain(&source);
            ExitCode::FAILURE
        }
    }
}

async fn run(cli: Cli) -> Result<RunEvidence, CliError> {
    let static_config = StaticEyeRuntimeConfig::parse(StaticEyeRuntimeConfigInput {
        device_path: cli.serial_device,
        baud_rate_bps: BAUD_RATE_BPS,
        response_timeout_ms: RESPONSE_TIMEOUT_MS,
        write_timeout_ms: WRITE_TIMEOUT_MS,
        write_attempts: WRITE_ATTEMPTS,
        empty_delimiter_budget: EMPTY_DELIMITER_BUDGET,
        expected_device_uid: cli.expected_device_uid,
        expected_firmware_build_id: cli.expected_firmware_build_id,
        expected_capabilities_bits: cli.expected_capabilities_bits,
        intent_lease_ms: COMMISSIONING_INTENT_LEASE_MS,
    })
    .map_err(CliError::Config)?;
    let config = static_config
        .new_session(&mut OsEyeSessionMaterialGenerator)
        .map_err(CliError::SessionMaterial)?;
    let clock = TokioClock::new();
    let (serial, mut handle, startup_receipt, task) =
        start_serial_eye_actor(config, clock.clone()).map_err(CliError::Start)?;
    let startup = match startup_receipt.wait().await {
        Ok(Ok(startup)) => startup,
        Ok(Err(source)) => {
            drop(handle);
            let exit = task.join().await.map_err(CliError::ActorJoin)?;
            return Err(CliError::StartupFault {
                source: Box::new(source),
                exit: Box::new(exit),
            });
        }
        Err(source) => {
            drop(handle);
            let actor = task.join().await;
            return Err(CliError::StartupReceipt {
                source,
                actor: Box::new(actor),
            });
        }
    };

    let interrupt = tokio::signal::ctrl_c();
    tokio::pin!(interrupt);
    let mut interrupted = false;
    let mut steps = Vec::with_capacity(COMMISSIONING_STEP_COUNT);

    for step in eye_commissioning_steps() {
        let now = match clock.now() {
            Ok(now) => now,
            Err(source) => {
                let cleanup = cleanup_after_local_failure(handle, task).await;
                return Err(CliError::Clock {
                    source,
                    cleanup: Box::new(cleanup),
                });
            }
        };
        let prepared = match step.prepare(now) {
            Ok(prepared) => prepared,
            Err(source) => {
                let cleanup = cleanup_after_local_failure(handle, task).await;
                return Err(CliError::Prepare {
                    source,
                    cleanup: Box::new(cleanup),
                });
            }
        };
        let evidence = match handle.apply_intent(prepared).await {
            Ok(evidence) => evidence,
            Err(source) => {
                drop(handle);
                let actor = task.join().await;
                return Err(CliError::Apply {
                    source,
                    actor: Box::new(actor),
                });
            }
        };
        steps.push(StepAdmission {
            name: step.name(),
            hold_duration_ms: u64::try_from(step.hold_duration().as_millis())
                .expect("fixed millisecond hold fits u64"),
            requested: prepared,
            evidence,
        });

        tokio::select! {
            () = tokio::time::sleep(step.hold_duration()) => {}
            signal = &mut interrupt => {
                if let Err(source) = signal {
                    let cleanup = cleanup_after_local_failure(handle, task).await;
                    return Err(CliError::Interrupt {
                        source,
                        cleanup: Box::new(cleanup),
                    });
                }
                interrupted = true;
                break;
            }
        }
    }

    let release = match handle.shutdown().await {
        Ok(release) => release,
        Err(source) => {
            let actor = task.join().await;
            return Err(CliError::Shutdown {
                source,
                actor: Box::new(actor),
            });
        }
    };
    let exit = task.join().await.map_err(CliError::ActorJoin)?;
    verify_graceful_exit(&startup, &steps, &release, &exit).map_err(CliError::InconsistentExit)?;
    if !matches!(release, ReleaseReport::Released(_)) {
        return Err(CliError::ReleaseFallback {
            release: Box::new(release),
            exit: Box::new(exit),
        });
    }

    Ok(RunEvidence {
        interrupted,
        serial,
        startup,
        steps,
        release,
        exit,
    })
}

async fn cleanup_after_local_failure(handle: EyeActorHandle, task: EyeActorTask) -> LocalCleanup {
    let shutdown = handle.shutdown().await;
    let actor = task.join().await;
    LocalCleanup { shutdown, actor }
}

fn verify_graceful_exit(
    startup: &StartupEvidence,
    steps: &[StepAdmission],
    release: &ReleaseReport,
    exit: &ActorExit,
) -> Result<(), ExitConsistencyError> {
    if exit.startup() != &Ok(startup.clone()) {
        return Err(ExitConsistencyError::Startup);
    }
    if exit.termination() != &ActorTermination::RequestedShutdown {
        return Err(ExitConsistencyError::Termination);
    }
    if exit.release() != Some(release) {
        return Err(ExitConsistencyError::Release);
    }
    if exit.admitted_intent_count()
        != u64::try_from(steps.len()).expect("bounded step count fits u64")
    {
        return Err(ExitConsistencyError::AdmissionCount {
            expected: steps.len(),
            actual: exit.admitted_intent_count(),
        });
    }
    let expected_last = steps.last().map(|step| &step.evidence);
    if exit.last_admission() != expected_last {
        return Err(ExitConsistencyError::LastAdmission);
    }
    Ok(())
}

fn write_evidence(evidence: RunEvidence) -> Result<bool, CliError> {
    let RunEvidence {
        interrupted,
        serial,
        startup,
        steps,
        release,
        exit,
    } = evidence;
    let identity = startup.identity();
    let step_count = steps.len();
    let step_values: Vec<_> = steps.iter().map(step_evidence_json).collect();
    let ReleaseReport::Released(release_evidence) = &release else {
        return Err(CliError::OutputInvariant);
    };
    let output = json!({
        "schema_version": 1,
        "operation": "kep2_eye_only_expression_commissioning",
        "outcome": if interrupted { "operator_interrupted" } else { "complete" },
        "protocol_version": PROTOCOL_VERSION,
        "safety_scope": {
            "eye_serial_only": true,
            "base_access": false,
            "head_access": false,
            "camera_access": false,
            "maximum_firmware_lease_ms": COMMISSIONING_INTENT_LEASE_MS,
            "firmware_fallback_after_host_loss": true,
        },
        "serial": {
            "path": serial.device().path(),
            "stable_name": serial.device().stable_name(),
            "exclusive_owner_claimed": serial.exclusive_owner_claimed(),
            "baud_rate_bps_readback": serial.baud_rate_bps_readback(),
            "data_bits_8_readback": serial.data_bits_8_readback(),
            "parity_none_readback": serial.parity_none_readback(),
            "stop_bits_1_readback": serial.stop_bits_1_readback(),
            "flow_control_none_readback": serial.flow_control_none_readback(),
        },
        "firmware_admission": {
            "device_uid_hex": encode_hex(identity.device_uid.as_bytes()),
            "firmware_build_id_hex": encode_hex(identity.firmware_build_id.as_bytes()),
            "capabilities_bits": identity.capabilities.bits(),
            "boot_id": identity.boot_id.get(),
            "control_epoch": startup.binding().control_epoch().get(),
        },
        "sequence": {
            "fixed_step_count": COMMISSIONING_STEP_COUNT,
            "admitted_step_count": step_count,
            "completed_all_holds": !interrupted && step_count == COMMISSIONING_STEP_COUNT,
            "steps": step_values,
        },
        "release": {
            "firmware_confirmed": true,
            "boot_id": release_evidence.binding().boot_id().get(),
            "control_epoch": release_evidence.binding().control_epoch().get(),
            "request_attempts": release_evidence.request_write().attempts_used(),
            "response_received_at_host_ns": release_evidence.response_received_at().nanos_since_epoch(),
        },
        "actor_exit": {
            "termination": "requested_shutdown",
            "admitted_intent_count": exit.admitted_intent_count(),
            "consistent_with_command_receipts": true,
        },
        "evidence_boundary": "KEP2 identity, acquisition, intent admissions, and release were protocol-confirmed; operator observation is still required to prove that either physical panel displayed the requested sequence"
    });
    let stdout = std::io::stdout();
    let mut stdout = stdout.lock();
    serde_json::to_writer_pretty(&mut stdout, &output).map_err(CliError::EncodeOutput)?;
    writeln!(stdout).map_err(CliError::WriteOutput)?;
    Ok(interrupted)
}

fn step_evidence_json(step: &StepAdmission) -> Value {
    let requested = step.requested.intent();
    let admission = step.evidence.admission();
    json!({
        "name": step.name,
        "hold_duration_ms": step.hold_duration_ms,
        "requested": intent_json(requested),
        "admission": {
            "intent_sequence": admission.sequence().get(),
            "lease_ms": admission.lease().get(),
            "applied_at_device_ms": admission.applied_at().millis_since_boot(),
            "expires_at_device_ms": admission.expires_at().millis_since_boot(),
            "rendered_frame_sequence": admission.rendered_frame_sequence().get(),
            "request_attempts": step.evidence.request_write().attempts_used(),
            "response_received_at_host_ns": step.evidence.response_received_at().nanos_since_epoch(),
        }
    })
}

fn intent_json(intent: EyeIntent) -> Value {
    json!({
        "expression": expression_name(intent.expression()),
        "gaze_x_normalized_1000": intent.gaze_x().get(),
        "gaze_y_normalized_1000": intent.gaze_y().get(),
        "lid_closure_normalized_1000": intent.lid().get(),
        "pupil_normalized_1000": intent.pupil().get(),
        "brightness_normalized_1000": intent.brightness().get(),
        "blink_requested": intent.flags().requests_blink(),
        "color_rgb": intent.color_rgb(),
    })
}

const fn expression_name(expression: Expression) -> &'static str {
    match expression {
        Expression::Neutral => "neutral",
        Expression::Curious => "curious",
        Expression::Greet => "greet",
        Expression::Concerned => "concerned",
        Expression::Sleepy => "sleepy",
    }
}

fn parse_cli(arguments: impl IntoIterator<Item = OsString>) -> Result<Cli, CliParseError> {
    let mut arguments = arguments.into_iter();
    let mut serial_device = None;
    let mut expected_device_uid = None;
    let mut expected_firmware_build_id = None;
    let mut expected_capabilities_bits = None;
    let mut execute = false;

    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--help" | "-h") => return Err(CliParseError::HelpRequested),
            Some("--serial-device") => {
                parse_once_string(&mut serial_device, "--serial-device", arguments.next())?
            }
            Some("--expected-device-uid-hex") => {
                if expected_device_uid.is_some() {
                    return Err(CliParseError::Duplicate("--expected-device-uid-hex"));
                }
                let value = required_value("--expected-device-uid-hex", arguments.next())?;
                expected_device_uid =
                    Some(parse_exact_hex::<16>("--expected-device-uid-hex", value)?);
            }
            Some("--expected-firmware-build-id-hex") => {
                if expected_firmware_build_id.is_some() {
                    return Err(CliParseError::Duplicate("--expected-firmware-build-id-hex"));
                }
                let value = required_value("--expected-firmware-build-id-hex", arguments.next())?;
                expected_firmware_build_id = Some(parse_exact_hex::<32>(
                    "--expected-firmware-build-id-hex",
                    value,
                )?);
            }
            Some("--expected-capabilities-bits") => {
                if expected_capabilities_bits.is_some() {
                    return Err(CliParseError::Duplicate("--expected-capabilities-bits"));
                }
                let value = required_value("--expected-capabilities-bits", arguments.next())?;
                let text = value.to_str().ok_or_else(|| CliParseError::NonUnicode {
                    flag: "--expected-capabilities-bits",
                    value: value.clone(),
                })?;
                if text.is_empty() || !text.bytes().all(|byte| byte.is_ascii_digit()) {
                    return Err(CliParseError::InvalidUnsignedDecimal {
                        flag: "--expected-capabilities-bits",
                        value,
                    });
                }
                expected_capabilities_bits = Some(text.parse::<u32>().map_err(|_| {
                    CliParseError::InvalidUnsignedDecimal {
                        flag: "--expected-capabilities-bits",
                        value,
                    }
                })?);
            }
            Some("--execute-eye-sequence") => {
                if execute {
                    return Err(CliParseError::Duplicate("--execute-eye-sequence"));
                }
                execute = true;
            }
            _ => return Err(CliParseError::UnknownArgument(argument)),
        }
    }

    if !execute {
        return Err(CliParseError::MissingExecuteConsent);
    }
    Ok(Cli {
        serial_device: serial_device.ok_or(CliParseError::Missing("--serial-device"))?,
        expected_device_uid: expected_device_uid
            .ok_or(CliParseError::Missing("--expected-device-uid-hex"))?,
        expected_firmware_build_id: expected_firmware_build_id
            .ok_or(CliParseError::Missing("--expected-firmware-build-id-hex"))?,
        expected_capabilities_bits: expected_capabilities_bits
            .ok_or(CliParseError::Missing("--expected-capabilities-bits"))?,
    })
}

fn parse_once_string(
    destination: &mut Option<String>,
    flag: &'static str,
    value: Option<OsString>,
) -> Result<(), CliParseError> {
    if destination.is_some() {
        return Err(CliParseError::Duplicate(flag));
    }
    let value = required_value(flag, value)?;
    *destination = Some(
        value
            .into_string()
            .map_err(|value| CliParseError::NonUnicode { flag, value })?,
    );
    Ok(())
}

fn required_value(flag: &'static str, value: Option<OsString>) -> Result<OsString, CliParseError> {
    value.ok_or(CliParseError::MissingValue(flag))
}

fn parse_exact_hex<const N: usize>(
    flag: &'static str,
    value: OsString,
) -> Result<[u8; N], CliParseError> {
    let text = value.to_str().ok_or_else(|| CliParseError::NonUnicode {
        flag,
        value: value.clone(),
    })?;
    if text.len() != N * 2 || !text.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(CliParseError::InvalidExactHex {
            flag,
            required_digits: N * 2,
            value,
        });
    }
    let mut output = [0_u8; N];
    for (index, output) in output.iter_mut().enumerate() {
        let high = hex_nibble(text.as_bytes()[index * 2]);
        let low = hex_nibble(text.as_bytes()[index * 2 + 1]);
        *output = (high << 4) | low;
    }
    Ok(output)
}

const fn hex_nibble(byte: u8) -> u8 {
    match byte {
        b'0'..=b'9' => byte - b'0',
        b'a'..=b'f' => byte - b'a' + 10,
        b'A'..=b'F' => byte - b'A' + 10,
        _ => unreachable!(),
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

#[derive(Debug, PartialEq, Eq)]
enum CliParseError {
    HelpRequested,
    Missing(&'static str),
    MissingValue(&'static str),
    MissingExecuteConsent,
    Duplicate(&'static str),
    NonUnicode {
        flag: &'static str,
        value: OsString,
    },
    InvalidExactHex {
        flag: &'static str,
        required_digits: usize,
        value: OsString,
    },
    InvalidUnsignedDecimal {
        flag: &'static str,
        value: OsString,
    },
    UnknownArgument(OsString),
}

impl fmt::Display for CliParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HelpRequested => formatter.write_str("help requested"),
            Self::Missing(flag) => write!(formatter, "{flag} is required"),
            Self::MissingValue(flag) => write!(formatter, "{flag} requires a value"),
            Self::MissingExecuteConsent => formatter.write_str(
                "--execute-eye-sequence is required because this command changes the displays",
            ),
            Self::Duplicate(flag) => write!(formatter, "{flag} was provided more than once"),
            Self::NonUnicode { flag, value } => {
                write!(formatter, "{flag} is not valid Unicode: {value:?}")
            }
            Self::InvalidExactHex {
                flag,
                required_digits,
                value,
            } => write!(
                formatter,
                "{flag} must contain exactly {required_digits} hexadecimal digits: {value:?}"
            ),
            Self::InvalidUnsignedDecimal { flag, value } => write!(
                formatter,
                "{flag} must be an unsigned base-10 integer: {value:?}"
            ),
            Self::UnknownArgument(argument) => write!(formatter, "unknown argument {argument:?}"),
        }
    }
}

impl Error for CliParseError {}

#[derive(Debug)]
struct LocalCleanup {
    shutdown: Result<ReleaseReport, HandleRequestError>,
    actor: Result<ActorExit, tokio::task::JoinError>,
}

impl fmt::Display for LocalCleanup {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "local-error cleanup shutdown={:?}, actor_exit={:?}",
            self.shutdown, self.actor
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExitConsistencyError {
    Startup,
    Termination,
    Release,
    AdmissionCount { expected: usize, actual: u64 },
    LastAdmission,
}

impl fmt::Display for ExitConsistencyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "KEP2 actor exit evidence was inconsistent: {self:?}"
        )
    }
}

impl Error for ExitConsistencyError {}

#[derive(Debug)]
enum CliError {
    Config(ConfigParseError),
    SessionMaterial(OsEyeSessionMaterialError),
    Start(EyeActorStartError),
    StartupReceipt {
        source: StartupReceiptError,
        actor: Box<Result<ActorExit, tokio::task::JoinError>>,
    },
    StartupFault {
        source: Box<kiko_eye_runtime::EyeRuntimeFault>,
        exit: Box<ActorExit>,
    },
    Clock {
        source: ClockError,
        cleanup: Box<LocalCleanup>,
    },
    Prepare {
        source: CommissioningPrepareError,
        cleanup: Box<LocalCleanup>,
    },
    Apply {
        source: HandleRequestError,
        actor: Box<Result<ActorExit, tokio::task::JoinError>>,
    },
    Interrupt {
        source: std::io::Error,
        cleanup: Box<LocalCleanup>,
    },
    Shutdown {
        source: HandleRequestError,
        actor: Box<Result<ActorExit, tokio::task::JoinError>>,
    },
    ActorJoin(tokio::task::JoinError),
    InconsistentExit(ExitConsistencyError),
    ReleaseFallback {
        release: Box<ReleaseReport>,
        exit: Box<ActorExit>,
    },
    OutputInvariant,
    EncodeOutput(serde_json::Error),
    WriteOutput(std::io::Error),
}

impl fmt::Display for CliError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(source) => write!(formatter, "{source}"),
            Self::SessionMaterial(source) => write!(formatter, "{source}"),
            Self::Start(source) => write!(formatter, "{source}"),
            Self::StartupReceipt { source, actor } => {
                write!(
                    formatter,
                    "{source}; actor join after receipt loss: {actor:?}"
                )
            }
            Self::StartupFault { source, exit } => {
                write!(formatter, "{source}; actor exit: {exit:?}")
            }
            Self::Clock { source, cleanup } => write!(formatter, "{source}; {cleanup}"),
            Self::Prepare { source, cleanup } => write!(formatter, "{source}; {cleanup}"),
            Self::Apply { source, actor } => {
                write!(
                    formatter,
                    "{source}; actor exit after apply failure: {actor:?}"
                )
            }
            Self::Interrupt { source, cleanup } => {
                write!(formatter, "could not monitor Ctrl-C: {source}; {cleanup}")
            }
            Self::Shutdown { source, actor } => write!(
                formatter,
                "graceful shutdown failed: {source}; actor join: {actor:?}"
            ),
            Self::ActorJoin(source) => write!(formatter, "KEP2 actor task failed: {source}"),
            Self::InconsistentExit(source) => write!(formatter, "{source}"),
            Self::ReleaseFallback { release, exit } => write!(
                formatter,
                "KEP2 release entered fallback: {release:?}; actor exit: {exit:?}"
            ),
            Self::OutputInvariant => formatter
                .write_str("successful commissioning evidence did not contain a confirmed release"),
            Self::EncodeOutput(source) => {
                write!(
                    formatter,
                    "could not encode commissioning evidence: {source}"
                )
            }
            Self::WriteOutput(source) => {
                write!(
                    formatter,
                    "could not write commissioning evidence: {source}"
                )
            }
        }
    }
}

impl Error for CliError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Config(source) => Some(source),
            Self::SessionMaterial(source) => Some(source),
            Self::Start(source) => Some(source),
            Self::StartupReceipt { source, .. } => Some(source),
            Self::StartupFault { source, .. } => Some(source),
            Self::Clock { source, .. } => Some(source),
            Self::Prepare { source, .. } => Some(source),
            Self::Apply { source, .. } => Some(source),
            Self::Interrupt { source, .. } => Some(source),
            Self::Shutdown { source, .. } => Some(source),
            Self::ActorJoin(source) => Some(source),
            Self::InconsistentExit(source) => Some(source),
            Self::ReleaseFallback { .. } | Self::OutputInvariant => None,
            Self::EncodeOutput(source) => Some(source),
            Self::WriteOutput(source) => Some(source),
        }
    }
}

fn print_usage() {
    eprintln!(
        "Usage: kep2_eye_commission \\\n  --serial-device /dev/serial/by-id/<exact-eye-identity> \\\n  --expected-device-uid-hex <32-hex-digits> \\\n  --expected-firmware-build-id-hex <64-hex-digits> \\\n  --expected-capabilities-bits <u32-decimal> \\\n  --execute-eye-sequence"
    );
}

fn print_error_chain(source: &(dyn Error + 'static)) {
    eprintln!("error: {source}");
    let mut current = source.source();
    while let Some(cause) = current {
        eprintln!("caused by: {cause}");
        current = cause.source();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_arguments() -> Vec<OsString> {
        vec![
            "--serial-device".into(),
            "/dev/serial/by-id/kiko-eye".into(),
            "--expected-device-uid-hex".into(),
            "0102030405060708090a0b0c0d0e0f10".into(),
            "--expected-firmware-build-id-hex".into(),
            "1111111111111111111111111111111111111111111111111111111111111111".into(),
            "--expected-capabilities-bits".into(),
            "255".into(),
            "--execute-eye-sequence".into(),
        ]
    }

    #[test]
    fn strict_cli_parses_manifest_identity_once() {
        let cli = parse_cli(valid_arguments()).expect("valid CLI");
        assert_eq!(cli.serial_device, "/dev/serial/by-id/kiko-eye");
        assert_eq!(cli.expected_device_uid[0], 1);
        assert_eq!(cli.expected_device_uid[15], 16);
        assert_eq!(cli.expected_firmware_build_id, [0x11; 32]);
        assert_eq!(cli.expected_capabilities_bits, 255);
    }

    #[test]
    fn changing_displays_requires_explicit_execution_flag() {
        let mut arguments = valid_arguments();
        arguments.pop();
        assert_eq!(
            parse_cli(arguments),
            Err(CliParseError::MissingExecuteConsent)
        );
    }

    #[test]
    fn duplicate_unknown_and_malformed_values_are_rejected() {
        let mut duplicate = valid_arguments();
        duplicate.extend(["--serial-device".into(), "/dev/cu.other".into()]);
        assert_eq!(
            parse_cli(duplicate),
            Err(CliParseError::Duplicate("--serial-device"))
        );

        let mut unknown = valid_arguments();
        unknown.push("--camera".into());
        assert!(matches!(
            parse_cli(unknown),
            Err(CliParseError::UnknownArgument(_))
        ));

        let mut bad_uid = valid_arguments();
        bad_uid[3] = "01".into();
        assert!(matches!(
            parse_cli(bad_uid),
            Err(CliParseError::InvalidExactHex {
                flag: "--expected-device-uid-hex",
                ..
            })
        ));

        let mut bad_capabilities = valid_arguments();
        bad_capabilities[7] = "0xff".into();
        assert!(matches!(
            parse_cli(bad_capabilities),
            Err(CliParseError::InvalidUnsignedDecimal {
                flag: "--expected-capabilities-bits",
                ..
            })
        ));
    }

    #[test]
    fn exact_hex_parser_is_case_insensitive_but_separator_free() {
        assert_eq!(
            parse_exact_hex::<2>("--value", "aB10".into()).expect("hex"),
            [0xab, 0x10]
        );
        assert!(parse_exact_hex::<2>("--value", "ab:10".into()).is_err());
    }
}
