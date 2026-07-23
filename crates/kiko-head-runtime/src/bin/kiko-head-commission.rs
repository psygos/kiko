use std::error::Error;
use std::ffi::OsString;
use std::fmt;
use std::fs::{self, OpenOptions};
use std::io::Read;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Duration;

use kiko_head_protocol::HeadJoint;
use kiko_head_runtime::{
    ActorExit, HeadProbeConfig, HeadProbeConfigInput, HeadProbeReport, ObservedHoldConfig,
    ObservedHoldConfigInput, PhysicalHeadMotionConsent, PhysicalTorqueEnableConsent,
    ReturnToTargetConfig, ReturnToTargetConfigInput, SerialConfigurationEvidence,
    TorqueDisableReport, VerifiedHeadReturnEvidence, VerifiedNaturalHoldEvidence,
    probe_serial_head, start_serial_head_actor, start_serial_head_return_actor,
};
use serde::Deserialize;

const CONFIG_SCHEMA_VERSION: u8 = 1;
const MAX_CONFIG_BYTES: u64 = 16 * 1024;
const MAX_COMMISSIONING_HOLD_DURATION_MS: u64 = 900_000;
#[cfg(target_os = "linux")]
const O_NOFOLLOW: i32 = 0o400000;
#[cfg(target_os = "macos")]
const O_NOFOLLOW: i32 = 0x100;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Probe,
    HoldObserved,
    ReturnToTarget,
}

#[derive(Debug, PartialEq, Eq)]
struct Cli {
    config_path: PathBuf,
    mode: Mode,
}

#[derive(Debug)]
enum CliError {
    HelpRequested,
    MissingConfig,
    MissingConfigPath,
    DuplicateConfig,
    DuplicateHoldObserved,
    DuplicateReturnToTarget,
    DuplicatePhysicalConsent,
    DuplicatePhysicalMotionConsent,
    ConflictingMotionModes,
    HoldRequiresPhysicalConsent,
    ReturnRequiresPhysicalConsent,
    ReturnRequiresPhysicalMotionConsent,
    TorqueConsentRequiresMotionMode,
    MotionConsentRequiresReturn,
    UnknownArgument(OsString),
}

impl fmt::Display for CliError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HelpRequested => formatter.write_str("help requested"),
            Self::MissingConfig => formatter.write_str("--config is required"),
            Self::MissingConfigPath => formatter.write_str("--config requires a path"),
            Self::DuplicateConfig => formatter.write_str("--config was provided more than once"),
            Self::DuplicateHoldObserved => {
                formatter.write_str("--hold-observed was provided more than once")
            }
            Self::DuplicateReturnToTarget => {
                formatter.write_str("--return-to-target was provided more than once")
            }
            Self::DuplicatePhysicalConsent => {
                formatter.write_str("--physical-torque-consent was provided more than once")
            }
            Self::DuplicatePhysicalMotionConsent => {
                formatter.write_str("--physical-motion-consent was provided more than once")
            }
            Self::ConflictingMotionModes => {
                formatter.write_str("--hold-observed and --return-to-target are mutually exclusive")
            }
            Self::HoldRequiresPhysicalConsent => formatter
                .write_str("--hold-observed requires the separate --physical-torque-consent flag"),
            Self::ReturnRequiresPhysicalConsent => formatter.write_str(
                "--return-to-target requires the separate --physical-torque-consent flag",
            ),
            Self::ReturnRequiresPhysicalMotionConsent => formatter.write_str(
                "--return-to-target requires the separate --physical-motion-consent flag",
            ),
            Self::TorqueConsentRequiresMotionMode => formatter.write_str(
                "--physical-torque-consent requires --hold-observed or --return-to-target",
            ),
            Self::MotionConsentRequiresReturn => {
                formatter.write_str("--physical-motion-consent requires --return-to-target")
            }
            Self::UnknownArgument(argument) => write!(formatter, "unknown argument {argument:?}"),
        }
    }
}

impl Error for CliError {}

#[derive(Debug)]
enum ConfigFileError {
    PathMustBeAbsolute {
        path: PathBuf,
    },
    PathContainsTraversal {
        path: PathBuf,
    },
    Metadata {
        path: PathBuf,
        source: std::io::Error,
    },
    NotRegularFile {
        path: PathBuf,
    },
    SymbolicLink {
        path: PathBuf,
    },
    TooLarge {
        actual_bytes: u64,
        maximum_bytes: u64,
    },
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    Json(serde_json::Error),
    UnsupportedSchema {
        actual: u8,
        supported: u8,
    },
    Probe(kiko_head_runtime::ConfigParseError),
    Hold(kiko_head_runtime::ObservedHoldConfigParseError),
    Return(kiko_head_runtime::ReturnToTargetConfigParseError),
    ReturnHoldDuration(CommissioningHoldDurationError),
    ConflictingConfigurations,
    HoldModeMissingConfiguration,
    ReturnModeMissingConfiguration,
}

impl fmt::Display for ConfigFileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PathMustBeAbsolute { path } => {
                write!(
                    formatter,
                    "configuration path must be absolute: {}",
                    path.display()
                )
            }
            Self::PathContainsTraversal { path } => write!(
                formatter,
                "configuration path must not contain . or .. components: {}",
                path.display()
            ),
            Self::Metadata { path, .. } => {
                write!(
                    formatter,
                    "could not inspect configuration file {}",
                    path.display()
                )
            }
            Self::NotRegularFile { path } => {
                write!(
                    formatter,
                    "configuration is not a regular file: {}",
                    path.display()
                )
            }
            Self::SymbolicLink { path } => write!(
                formatter,
                "configuration path must not be a symbolic link: {}",
                path.display()
            ),
            Self::TooLarge {
                actual_bytes,
                maximum_bytes,
            } => write!(
                formatter,
                "configuration has {actual_bytes} bytes; maximum is {maximum_bytes}"
            ),
            Self::Read { path, .. } => {
                write!(
                    formatter,
                    "could not read configuration file {}",
                    path.display()
                )
            }
            Self::Json(_) => formatter.write_str("configuration is not strict schema-valid JSON"),
            Self::UnsupportedSchema { actual, supported } => write!(
                formatter,
                "configuration schema {actual} is unsupported; expected {supported}"
            ),
            Self::Probe(_) => formatter.write_str("probe configuration is invalid"),
            Self::Hold(_) => formatter.write_str("observed-hold configuration is invalid"),
            Self::Return(_) => formatter.write_str("return-to-target configuration is invalid"),
            Self::ReturnHoldDuration(_) => {
                formatter.write_str("return commissioning hold duration is invalid")
            }
            Self::ConflictingConfigurations => {
                formatter.write_str("hold_observed and return_to_target cannot both be configured")
            }
            Self::HoldModeMissingConfiguration => {
                formatter.write_str("--hold-observed requires a hold_observed configuration object")
            }
            Self::ReturnModeMissingConfiguration => formatter
                .write_str("--return-to-target requires a return_to_target configuration object"),
        }
    }
}

impl Error for ConfigFileError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Metadata { source, .. } | Self::Read { source, .. } => Some(source),
            Self::Json(source) => Some(source),
            Self::Probe(source) => Some(source),
            Self::Hold(source) => Some(source),
            Self::Return(source) => Some(source),
            Self::ReturnHoldDuration(source) => Some(source),
            Self::PathMustBeAbsolute { .. }
            | Self::PathContainsTraversal { .. }
            | Self::NotRegularFile { .. }
            | Self::SymbolicLink { .. }
            | Self::TooLarge { .. }
            | Self::UnsupportedSchema { .. }
            | Self::ConflictingConfigurations
            | Self::HoldModeMissingConfiguration
            | Self::ReturnModeMissingConfiguration => None,
        }
    }
}

/// Bounded lease used only by the commissioning command after a successful
/// return. It is deliberately absent from the reusable return transaction and
/// from the continuous production owner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CommissioningHoldDuration(Duration);

impl CommissioningHoldDuration {
    fn parse(milliseconds: u64) -> Result<Self, CommissioningHoldDurationError> {
        if !(1..=MAX_COMMISSIONING_HOLD_DURATION_MS).contains(&milliseconds) {
            return Err(CommissioningHoldDurationError::OutOfRange {
                milliseconds,
                minimum_ms: 1,
                maximum_ms: MAX_COMMISSIONING_HOLD_DURATION_MS,
            });
        }
        Ok(Self(Duration::from_millis(milliseconds)))
    }

    const fn get(self) -> Duration {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CommissioningHoldDurationError {
    OutOfRange {
        milliseconds: u64,
        minimum_ms: u64,
        maximum_ms: u64,
    },
}

impl fmt::Display for CommissioningHoldDurationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid commissioning hold duration: {self:?}")
    }
}

impl Error for CommissioningHoldDurationError {}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConfigFileInput {
    schema_version: u8,
    probe: ProbeInput,
    hold_observed: Option<HoldInput>,
    return_to_target: Option<ReturnInput>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProbeInput {
    device_path: String,
    response_timeout_ms: u64,
    request_timeout_ms: u64,
    noise_budget_bytes: u16,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct HoldInput {
    write_timeout_ms: u64,
    arming_freshness_ms: u64,
    write_attempts: u8,
    redundant_read_tolerance_ticks: u16,
    readback_tolerance_ticks: u16,
    goal_speed_ticks_per_second: u16,
    torque_limit_permille: [u16; 4],
    minimum_ticks: [u16; 4],
    maximum_ticks: [u16; 4],
    maximum_hold_ms: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReturnInput {
    write_timeout_ms: u64,
    arming_freshness_ms: u64,
    write_attempts: u8,
    redundant_read_tolerance_ticks: u16,
    readback_tolerance_ticks: u16,
    final_target_tolerance_ticks: u16,
    path_corridor_tolerance_ticks: u16,
    direction_regression_tolerance_ticks: u16,
    goal_speed_ticks_per_second: u16,
    torque_limit_permille: [u16; 4],
    minimum_start_ticks: [u16; 4],
    maximum_start_ticks: [u16; 4],
    target_ticks: [u16; 4],
    maximum_travel_ticks: [u16; 4],
    maximum_hold_ms: u64,
    physical_motion_consent: PhysicalMotionConsentInput,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum PhysicalMotionConsentInput {
    ReturnToReviewedTarget,
}

struct CommissionConfig {
    probe: HeadProbeConfig,
    hold: Option<ObservedHoldConfig>,
    head_return: Option<CommissioningReturnConfig>,
}

struct CommissioningReturnConfig {
    transaction: ReturnToTargetConfig,
    hold_duration: CommissioningHoldDuration,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StopReason {
    MaximumDuration,
    Interrupt,
    Terminate,
}

enum WaitOutcome<T> {
    Completed(T),
    Stopped(Result<StopReason, std::io::Error>),
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> ExitCode {
    let cli = match parse_cli(std::env::args_os().skip(1)) {
        Ok(cli) => cli,
        Err(CliError::HelpRequested) => {
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
            print_error_chain(source.as_ref());
            ExitCode::FAILURE
        }
    }
}

async fn run(cli: Cli) -> Result<(), Box<dyn Error>> {
    let config = load_config(&cli.config_path)?;
    let probe = probe_serial_head(&config.probe).await?;
    print_probe(&probe);

    if cli.mode == Mode::HoldObserved {
        let hold = config
            .hold
            .ok_or(ConfigFileError::HoldModeMissingConfiguration)?;
        run_observed_hold(hold).await?;
    } else if cli.mode == Mode::ReturnToTarget {
        let head_return = config
            .head_return
            .ok_or(ConfigFileError::ReturnModeMissingConfiguration)?;
        run_return_to_target(head_return).await?;
    }
    Ok(())
}

fn parse_cli(arguments: impl IntoIterator<Item = OsString>) -> Result<Cli, CliError> {
    let mut arguments = arguments.into_iter();
    let mut config_path = None;
    let mut hold_observed = false;
    let mut return_to_target = false;
    let mut physical_consent = false;
    let mut physical_motion_consent = false;
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--help" | "-h") => return Err(CliError::HelpRequested),
            Some("--config") => {
                if config_path.is_some() {
                    return Err(CliError::DuplicateConfig);
                }
                config_path = Some(arguments.next().ok_or(CliError::MissingConfigPath)?.into());
            }
            Some("--hold-observed") => {
                if hold_observed {
                    return Err(CliError::DuplicateHoldObserved);
                }
                hold_observed = true;
            }
            Some("--return-to-target") => {
                if return_to_target {
                    return Err(CliError::DuplicateReturnToTarget);
                }
                return_to_target = true;
            }
            Some("--physical-torque-consent") => {
                if physical_consent {
                    return Err(CliError::DuplicatePhysicalConsent);
                }
                physical_consent = true;
            }
            Some("--physical-motion-consent") => {
                if physical_motion_consent {
                    return Err(CliError::DuplicatePhysicalMotionConsent);
                }
                physical_motion_consent = true;
            }
            _ => return Err(CliError::UnknownArgument(argument)),
        }
    }
    let mode = match (
        hold_observed,
        return_to_target,
        physical_consent,
        physical_motion_consent,
    ) {
        (false, false, false, false) => Mode::Probe,
        (true, false, true, false) => Mode::HoldObserved,
        (false, true, true, true) => Mode::ReturnToTarget,
        (true, true, _, _) => return Err(CliError::ConflictingMotionModes),
        (true, false, false, _) => return Err(CliError::HoldRequiresPhysicalConsent),
        (true, false, true, true) => return Err(CliError::MotionConsentRequiresReturn),
        (false, true, false, _) => return Err(CliError::ReturnRequiresPhysicalConsent),
        (false, true, true, false) => {
            return Err(CliError::ReturnRequiresPhysicalMotionConsent);
        }
        (false, false, true, _) => return Err(CliError::TorqueConsentRequiresMotionMode),
        (false, false, false, true) => return Err(CliError::MotionConsentRequiresReturn),
    };
    Ok(Cli {
        config_path: config_path.ok_or(CliError::MissingConfig)?,
        mode,
    })
}

fn load_config(path: &Path) -> Result<CommissionConfig, ConfigFileError> {
    if !path.is_absolute() {
        return Err(ConfigFileError::PathMustBeAbsolute {
            path: path.to_owned(),
        });
    }
    if path_contains_dot_segment(path) {
        return Err(ConfigFileError::PathContainsTraversal {
            path: path.to_owned(),
        });
    }
    let path_metadata = fs::symlink_metadata(path).map_err(|source| ConfigFileError::Metadata {
        path: path.to_owned(),
        source,
    })?;
    if path_metadata.file_type().is_symlink() {
        return Err(ConfigFileError::SymbolicLink {
            path: path.to_owned(),
        });
    }
    let mut options = OpenOptions::new();
    options.read(true).custom_flags(O_NOFOLLOW);
    let mut file = options.open(path).map_err(|source| ConfigFileError::Read {
        path: path.to_owned(),
        source,
    })?;
    let metadata = file
        .metadata()
        .map_err(|source| ConfigFileError::Metadata {
            path: path.to_owned(),
            source,
        })?;
    if !metadata.is_file() {
        return Err(ConfigFileError::NotRegularFile {
            path: path.to_owned(),
        });
    }
    if metadata.len() > MAX_CONFIG_BYTES {
        return Err(ConfigFileError::TooLarge {
            actual_bytes: metadata.len(),
            maximum_bytes: MAX_CONFIG_BYTES,
        });
    }
    let mut bytes = Vec::with_capacity(
        usize::try_from(metadata.len()).expect("admitted 16 KiB file length fits usize"),
    );
    file.by_ref()
        .take(MAX_CONFIG_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|source| ConfigFileError::Read {
            path: path.to_owned(),
            source,
        })?;
    let actual_bytes = u64::try_from(bytes.len()).expect("usize fits u64 on supported hosts");
    if actual_bytes > MAX_CONFIG_BYTES {
        return Err(ConfigFileError::TooLarge {
            actual_bytes,
            maximum_bytes: MAX_CONFIG_BYTES,
        });
    }
    let input: ConfigFileInput = serde_json::from_slice(&bytes).map_err(ConfigFileError::Json)?;
    if input.schema_version != CONFIG_SCHEMA_VERSION {
        return Err(ConfigFileError::UnsupportedSchema {
            actual: input.schema_version,
            supported: CONFIG_SCHEMA_VERSION,
        });
    }
    let probe = HeadProbeConfig::parse(HeadProbeConfigInput {
        device_path: input.probe.device_path,
        response_timeout_ms: input.probe.response_timeout_ms,
        request_timeout_ms: input.probe.request_timeout_ms,
        noise_budget_bytes: input.probe.noise_budget_bytes,
    })
    .map_err(ConfigFileError::Probe)?;
    let hold = input
        .hold_observed
        .map(|hold| {
            ObservedHoldConfig::parse(
                &probe,
                ObservedHoldConfigInput {
                    write_timeout_ms: hold.write_timeout_ms,
                    arming_freshness_ms: hold.arming_freshness_ms,
                    write_attempts: hold.write_attempts,
                    redundant_read_tolerance_ticks: hold.redundant_read_tolerance_ticks,
                    readback_tolerance_ticks: hold.readback_tolerance_ticks,
                    goal_speed_ticks_per_second: hold.goal_speed_ticks_per_second,
                    torque_limit_permille: hold.torque_limit_permille,
                    minimum_ticks: hold.minimum_ticks,
                    maximum_ticks: hold.maximum_ticks,
                    maximum_hold_ms: hold.maximum_hold_ms,
                },
            )
        })
        .transpose()
        .map_err(ConfigFileError::Hold)?;
    let head_return = input
        .return_to_target
        .map(|head_return| {
            let PhysicalMotionConsentInput::ReturnToReviewedTarget =
                head_return.physical_motion_consent;
            let hold_duration = CommissioningHoldDuration::parse(head_return.maximum_hold_ms)
                .map_err(ConfigFileError::ReturnHoldDuration)?;
            let transaction = ReturnToTargetConfig::parse(
                &probe,
                ReturnToTargetConfigInput {
                    write_timeout_ms: head_return.write_timeout_ms,
                    arming_freshness_ms: head_return.arming_freshness_ms,
                    write_attempts: head_return.write_attempts,
                    redundant_read_tolerance_ticks: head_return.redundant_read_tolerance_ticks,
                    readback_tolerance_ticks: head_return.readback_tolerance_ticks,
                    final_target_tolerance_ticks: head_return.final_target_tolerance_ticks,
                    path_corridor_tolerance_ticks: head_return.path_corridor_tolerance_ticks,
                    direction_regression_tolerance_ticks: head_return
                        .direction_regression_tolerance_ticks,
                    goal_speed_ticks_per_second: head_return.goal_speed_ticks_per_second,
                    torque_limit_permille: head_return.torque_limit_permille,
                    minimum_start_ticks: head_return.minimum_start_ticks,
                    maximum_start_ticks: head_return.maximum_start_ticks,
                    target_ticks: head_return.target_ticks,
                    maximum_travel_ticks: head_return.maximum_travel_ticks,
                },
            )
            .map_err(ConfigFileError::Return)?;
            Ok(CommissioningReturnConfig {
                transaction,
                hold_duration,
            })
        })
        .transpose()?;
    if hold.is_some() && head_return.is_some() {
        return Err(ConfigFileError::ConflictingConfigurations);
    }
    Ok(CommissionConfig {
        probe,
        hold,
        head_return,
    })
}

fn path_contains_dot_segment(path: &Path) -> bool {
    path.as_os_str()
        .as_encoded_bytes()
        .split(|byte| *byte == b'/')
        .any(|segment| segment == b"." || segment == b"..")
}

async fn run_observed_hold(config: ObservedHoldConfig) -> Result<(), Box<dyn Error>> {
    let maximum_duration = config.maximum_duration();
    // Install both handlers before opening the port or issuing any command
    // that can energise torque. Startup itself is cancellable; a signal does
    // not merely remain pending while further physical writes continue.
    let mut stop_signals = StopSignals::install()?;
    let (serial, handle, startup, task) = start_serial_head_actor(
        config.runtime().clone(),
        config.pose_bounds(),
        PhysicalTorqueEnableConsent::explicitly_granted(),
    )?;
    print_serial("hold_serial", &serial);

    let startup_outcome = {
        let startup_wait = startup.wait();
        tokio::pin!(startup_wait);
        tokio::select! {
            result = &mut startup_wait => WaitOutcome::Completed(result),
            stop = stop_signals.wait_signal() => WaitOutcome::Stopped(stop),
        }
    };
    let startup_result = match startup_outcome {
        WaitOutcome::Completed(Ok(result)) => result,
        WaitOutcome::Stopped(stop) => {
            let shutdown = handle.shutdown().await;
            let exit = task.join().await?;
            print_actor_exit(&exit);
            if !exit.torque_disable().all_writes_completed() {
                return Err(Box::new(CleanupIncomplete));
            }
            println!("hold_stop reason={:?}", stop?);
            shutdown?;
            return Ok(());
        }
        WaitOutcome::Completed(Err(source)) => {
            drop(handle);
            let exit = task.join().await?;
            print_actor_exit(&exit);
            if !exit.torque_disable().all_writes_completed() {
                return Err(Box::new(CleanupIncomplete));
            }
            return Err(Box::new(source));
        }
    };
    let evidence = match startup_result {
        Ok(evidence) => evidence,
        Err(source) => {
            drop(handle);
            let exit = task.join().await?;
            print_actor_exit(&exit);
            if !exit.torque_disable().all_writes_completed() {
                return Err(Box::new(CleanupIncomplete));
            }
            return Err(Box::new(source));
        }
    };
    print_hold_started(&evidence, maximum_duration);

    let stop = stop_signals.wait(maximum_duration).await;
    let shutdown = handle.shutdown().await;
    let exit = task.join().await?;
    print_actor_exit(&exit);

    if !exit.torque_disable().all_writes_completed() {
        return Err(Box::new(CleanupIncomplete));
    }
    let stop = stop?;
    println!("hold_stop reason={stop:?}");
    shutdown?;
    Ok(())
}

async fn run_return_to_target(config: CommissioningReturnConfig) -> Result<(), Box<dyn Error>> {
    let maximum_duration = config.hold_duration.get();
    let mut stop_signals = StopSignals::install()?;
    let (serial, handle, startup, task) = start_serial_head_return_actor(
        config.transaction,
        PhysicalTorqueEnableConsent::explicitly_granted(),
        PhysicalHeadMotionConsent::explicitly_granted(),
    )?;
    print_serial("return_serial", &serial);

    let startup_outcome = {
        let startup_wait = startup.wait();
        tokio::pin!(startup_wait);
        tokio::select! {
            result = &mut startup_wait => WaitOutcome::Completed(result),
            stop = stop_signals.wait_signal() => WaitOutcome::Stopped(stop),
        }
    };
    let startup_result = match startup_outcome {
        WaitOutcome::Completed(Ok(result)) => result,
        WaitOutcome::Stopped(stop) => {
            let shutdown = handle.shutdown().await;
            let exit = task.join().await?;
            print_actor_exit(&exit);
            if !exit.torque_disable().all_writes_completed() {
                return Err(Box::new(CleanupIncomplete));
            }
            println!("hold_stop reason={:?}", stop?);
            shutdown?;
            return Ok(());
        }
        WaitOutcome::Completed(Err(source)) => {
            drop(handle);
            let exit = task.join().await?;
            print_actor_exit(&exit);
            if !exit.torque_disable().all_writes_completed() {
                return Err(Box::new(CleanupIncomplete));
            }
            return Err(Box::new(source));
        }
    };
    let hold = match startup_result {
        Ok(evidence) => evidence,
        Err(source) => {
            drop(handle);
            let exit = task.join().await?;
            print_actor_exit(&exit);
            if !exit.torque_disable().all_writes_completed() {
                return Err(Box::new(CleanupIncomplete));
            }
            return Err(Box::new(source));
        }
    };
    print_hold_started(&hold, maximum_duration);

    let return_outcome = {
        let return_wait = handle.return_to_target();
        tokio::pin!(return_wait);
        tokio::select! {
            result = &mut return_wait => WaitOutcome::Completed(result),
            stop = stop_signals.wait_signal() => WaitOutcome::Stopped(stop),
        }
    };
    let return_result = match return_outcome {
        WaitOutcome::Completed(Ok(result)) => result,
        WaitOutcome::Stopped(stop) => {
            let shutdown = handle.shutdown().await;
            let exit = task.join().await?;
            print_actor_exit(&exit);
            if !exit.torque_disable().all_writes_completed() {
                return Err(Box::new(CleanupIncomplete));
            }
            println!("hold_stop reason={:?}", stop?);
            shutdown?;
            return Ok(());
        }
        WaitOutcome::Completed(Err(source)) => {
            // Losing the command receipt must not let the current-thread Tokio
            // runtime exit before the actor's four-joint cleanup completes.
            drop(handle);
            let exit = task.join().await?;
            print_actor_exit(&exit);
            if !exit.torque_disable().all_writes_completed() {
                return Err(Box::new(CleanupIncomplete));
            }
            return Err(Box::new(source));
        }
    };
    let returned = match return_result {
        Ok(evidence) => evidence,
        Err(source) => {
            // A commissioning recovery write is not continuously supervised.
            // Preserve the typed return error, but end the lease and clean up
            // immediately instead of blindly holding for the commissioning
            // duration.
            eprintln!(
                "return_failed owner_retained={} error={source}",
                source.retains_owner_after_fault()
            );
            let shutdown = handle.shutdown().await;
            let exit = task.join().await?;
            print_actor_exit(&exit);
            if !exit.torque_disable().all_writes_completed() {
                return Err(Box::new(CleanupIncomplete));
            }
            shutdown?;
            return Err(Box::new(source));
        }
    };
    print_return_completed(&returned, maximum_duration);

    let stop = stop_signals.wait(maximum_duration).await;
    let shutdown = handle.shutdown().await;
    let exit = task.join().await?;
    print_actor_exit(&exit);

    if !exit.torque_disable().all_writes_completed() {
        return Err(Box::new(CleanupIncomplete));
    }
    let stop = stop?;
    println!("hold_stop reason={stop:?}");
    shutdown?;
    Ok(())
}

struct StopSignals {
    interrupt: tokio::signal::unix::Signal,
    terminate: tokio::signal::unix::Signal,
}

impl StopSignals {
    fn install() -> Result<Self, std::io::Error> {
        Ok(Self {
            interrupt: tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())?,
            terminate: tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?,
        })
    }

    async fn wait(&mut self, maximum_duration: Duration) -> Result<StopReason, std::io::Error> {
        tokio::select! {
            () = tokio::time::sleep(maximum_duration) => Ok(StopReason::MaximumDuration),
            stop = self.wait_signal() => stop,
        }
    }

    async fn wait_signal(&mut self) -> Result<StopReason, std::io::Error> {
        tokio::select! {
            signal = self.interrupt.recv() => signal
                .map(|()| StopReason::Interrupt)
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::BrokenPipe, "SIGINT stream closed")),
            signal = self.terminate.recv() => signal
                .map(|()| StopReason::Terminate)
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::BrokenPipe, "SIGTERM stream closed")),
        }
    }
}

#[derive(Debug)]
struct CleanupIncomplete;

impl fmt::Display for CleanupIncomplete {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("one or more torque-disable writes did not complete")
    }
}

impl Error for CleanupIncomplete {}

fn print_probe(report: &HeadProbeReport) {
    print_serial("probe_serial", report.serial());
    for servo in report.servos() {
        let telemetry = servo.telemetry();
        println!(
            "probe joint={} id={} torque_switch_raw={} position_ticks={} moving={} speed_raw={} load_raw={} voltage_raw={} temperature_raw={} current_raw={} torque_noise_bytes={} telemetry_noise_bytes={}",
            joint_name(servo.joint()),
            telemetry.id().get(),
            servo.torque_switch().state().raw(),
            telemetry.position().get(),
            telemetry.is_moving(),
            telemetry.speed_raw(),
            telemetry.load_raw(),
            telemetry.voltage_raw(),
            telemetry.temperature_raw(),
            telemetry.current_raw(),
            servo.torque_response().discarded_noise_bytes(),
            servo.telemetry_response().discarded_noise_bytes(),
        );
    }
}

fn print_serial(label: &str, serial: &SerialConfigurationEvidence) {
    println!(
        "{label} device={} exclusive={} baud={} data_bits_8={} parity_none={} stop_bits_1={} flow_control_none={} dtr_false_setter_accepted={} rts_true_setter_accepted={}",
        serial.device.path(),
        serial.exclusive_owner_claimed,
        serial.baud_rate_bps_readback,
        serial.data_bits_8_readback,
        serial.parity_none_readback,
        serial.stop_bits_1_readback,
        serial.flow_control_none_readback,
        serial.dtr_false_setter_accepted,
        serial.rts_true_setter_accepted,
    );
}

fn print_hold_started(evidence: &VerifiedNaturalHoldEvidence, maximum_duration: Duration) {
    let pre_observation_disable_complete = evidence
        .pre_observation_torque_disable()
        .is_some_and(TorqueDisableReport::all_writes_completed);
    println!(
        "hold_started observed_ticks={:?} pre_observation_disable_complete={} maximum_hold_ms={}",
        evidence.observed_pose().positions(),
        pre_observation_disable_complete,
        maximum_duration.as_millis(),
    );
}

fn print_return_completed(evidence: &VerifiedHeadReturnEvidence, maximum_duration: Duration) {
    println!(
        "return_completed start_ticks={:?} target_ticks={:?} waypoint_cycles={} maximum_hold_ms={}",
        evidence
            .start_pose()
            .positions()
            .map(kiko_head_protocol::PositionTicks::get),
        evidence
            .target()
            .positions()
            .map(kiko_head_protocol::PositionTicks::get),
        evidence.waypoint_writes().len(),
        maximum_duration.as_millis(),
    );
    for (index, joint) in HeadJoint::ALL.into_iter().enumerate() {
        let first = evidence.first_stopped()[index];
        let second = evidence.second_stopped()[index];
        println!(
            "return_final joint={} first_ticks={} second_ticks={} first_moving={} second_moving={} load_raw={} voltage_raw={} temperature_raw={} current_raw={} device_status_raw={}",
            joint_name(joint),
            first.position().get(),
            second.position().get(),
            first.is_moving(),
            second.is_moving(),
            second.load_raw(),
            second.voltage_raw(),
            second.temperature_raw(),
            second.current_raw(),
            second.device_status_raw(),
        );
    }
}

fn print_actor_exit(exit: &ActorExit) {
    println!("hold_exit termination={:?}", exit.termination());
    if let Err(source) = exit.startup() {
        eprintln!("hold_exit startup_error={source}");
    }
    if let Some(result) = exit.head_return() {
        match result {
            Ok(evidence) => println!(
                "return_exit completed=true waypoint_batches={}",
                evidence.waypoint_writes().len()
            ),
            Err(source) => eprintln!("return_exit completed=false error={source}"),
        }
    }
    print_cleanup(exit.torque_disable());
}

fn print_cleanup(report: &TorqueDisableReport) {
    println!(
        "torque_disable_complete all_writes_completed={}",
        report.all_writes_completed()
    );
    for outcome in report.outcomes() {
        match outcome.result() {
            Ok(_) => println!(
                "torque_disable joint={} completed=true",
                joint_name(outcome.joint())
            ),
            Err(source) => println!(
                "torque_disable joint={} completed=false error={source}",
                joint_name(outcome.joint())
            ),
        }
    }
}

const fn joint_name(joint: HeadJoint) -> &'static str {
    match joint {
        HeadJoint::Bow => "bow",
        HeadJoint::Curl => "curl",
        HeadJoint::Yaw => "yaw",
        HeadJoint::Roll => "roll",
    }
}

fn print_usage() {
    eprintln!(
        "usage: kiko-head-commission --config /absolute/path.json [--hold-observed --physical-torque-consent | --return-to-target --physical-torque-consent --physical-motion-consent]"
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
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    static TEMP_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    fn os(values: &[&str]) -> Vec<OsString> {
        values.iter().map(OsString::from).collect()
    }

    fn temporary_path(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "kiko-head-commission-{label}-{}-{}",
            std::process::id(),
            TEMP_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ))
    }

    #[test]
    fn probe_is_the_only_ungated_mode() {
        let cli = parse_cli(os(&["--config", "/tmp/head.json"])).expect("probe CLI");
        assert_eq!(cli.mode, Mode::Probe);
        assert!(matches!(
            parse_cli(os(&["--config", "/tmp/head.json", "--hold-observed"])),
            Err(CliError::HoldRequiresPhysicalConsent)
        ));
        assert!(matches!(
            parse_cli(os(&[
                "--config",
                "/tmp/head.json",
                "--physical-torque-consent"
            ])),
            Err(CliError::TorqueConsentRequiresMotionMode)
        ));
    }

    #[test]
    fn hold_requires_both_exact_flags_and_rejects_unknown_arguments() {
        let cli = parse_cli(os(&[
            "--config",
            "/tmp/head.json",
            "--hold-observed",
            "--physical-torque-consent",
        ]))
        .expect("explicitly consented hold CLI");
        assert_eq!(cli.mode, Mode::HoldObserved);
        assert!(matches!(
            parse_cli(os(&["--config", "/tmp/head.json", "--force"])),
            Err(CliError::UnknownArgument(_))
        ));
    }

    #[test]
    fn return_requires_both_physical_consents() {
        let cli = parse_cli(os(&[
            "--config",
            "/tmp/head.json",
            "--return-to-target",
            "--physical-torque-consent",
            "--physical-motion-consent",
        ]))
        .expect("explicitly consented return CLI");
        assert_eq!(cli.mode, Mode::ReturnToTarget);
        assert!(matches!(
            parse_cli(os(&[
                "--config",
                "/tmp/head.json",
                "--return-to-target",
                "--physical-torque-consent",
            ])),
            Err(CliError::ReturnRequiresPhysicalMotionConsent)
        ));
    }

    #[test]
    fn strict_json_rejects_unknown_fields_before_domain_parsing() {
        let json = br#"{
            "schema_version": 1,
            "probe": {
                "device_path": "/dev/serial/by-id/test",
                "response_timeout_ms": 100,
                "request_timeout_ms": 100,
                "noise_budget_bytes": 8,
                "guess_device": true
            },
            "hold_observed": null
        }"#;
        assert!(serde_json::from_slice::<ConfigFileInput>(json).is_err());
    }

    #[test]
    fn strict_json_maps_every_distinct_return_tolerance() {
        let path = temporary_path("valid-return");
        let json = br#"{
            "schema_version": 1,
            "probe": {
                "device_path": "/dev/serial/by-id/usb-Kiko_head_test",
                "response_timeout_ms": 100,
                "request_timeout_ms": 100,
                "noise_budget_bytes": 16
            },
            "hold_observed": null,
            "return_to_target": {
                "write_timeout_ms": 1,
                "arming_freshness_ms": 250,
                "write_attempts": 1,
                "redundant_read_tolerance_ticks": 10,
                "readback_tolerance_ticks": 20,
                "final_target_tolerance_ticks": 0,
                "path_corridor_tolerance_ticks": 0,
                "direction_regression_tolerance_ticks": 0,
                "goal_speed_ticks_per_second": 100,
                "torque_limit_permille": [600, 400, 400, 400],
                "minimum_start_ticks": [2127, 2558, 2925, 2930],
                "maximum_start_ticks": [2127, 2558, 2925, 2930],
                "target_ticks": [2127, 2558, 2925, 2930],
                "maximum_travel_ticks": [1, 1, 1, 1],
                "maximum_hold_ms": 1000,
                "physical_motion_consent": "return_to_reviewed_target"
            }
        }"#;
        fs::write(&path, json).expect("write valid return configuration");

        let config = load_config(&path).expect("strict return configuration");

        fs::remove_file(&path).expect("remove valid return configuration");
        assert!(config.hold.is_none());
        assert_eq!(
            config
                .head_return
                .expect("return configuration")
                .hold_duration
                .get(),
            Duration::from_millis(1_000)
        );
    }

    #[test]
    fn commissioning_return_hold_duration_is_separate_and_bounded() {
        assert_eq!(
            CommissioningHoldDuration::parse(1)
                .expect("minimum duration")
                .get(),
            Duration::from_millis(1)
        );
        assert_eq!(
            CommissioningHoldDuration::parse(MAX_COMMISSIONING_HOLD_DURATION_MS)
                .expect("maximum duration")
                .get(),
            Duration::from_millis(MAX_COMMISSIONING_HOLD_DURATION_MS)
        );
        assert_eq!(
            CommissioningHoldDuration::parse(0),
            Err(CommissioningHoldDurationError::OutOfRange {
                milliseconds: 0,
                minimum_ms: 1,
                maximum_ms: MAX_COMMISSIONING_HOLD_DURATION_MS,
            })
        );
        assert_eq!(
            CommissioningHoldDuration::parse(MAX_COMMISSIONING_HOLD_DURATION_MS + 1),
            Err(CommissioningHoldDurationError::OutOfRange {
                milliseconds: MAX_COMMISSIONING_HOLD_DURATION_MS + 1,
                minimum_ms: 1,
                maximum_ms: MAX_COMMISSIONING_HOLD_DURATION_MS,
            })
        );
    }

    #[test]
    fn config_path_check_rejects_literal_dot_segments() {
        assert!(path_contains_dot_segment(Path::new("/tmp/./head.json")));
        assert!(path_contains_dot_segment(Path::new("/tmp/a/../head.json")));
        assert!(!path_contains_dot_segment(Path::new("/tmp/head.json")));
    }

    #[test]
    fn config_loader_rejects_symlinks_before_reading_the_target() {
        use std::os::unix::fs::symlink;

        let target = temporary_path("target");
        let link = temporary_path("link");
        fs::write(&target, b"{}").expect("write temporary target");
        symlink(&target, &link).expect("create temporary symbolic link");

        let result = load_config(&link);

        fs::remove_file(&link).expect("remove temporary symbolic link");
        fs::remove_file(&target).expect("remove temporary target");
        assert!(matches!(result, Err(ConfigFileError::SymbolicLink { .. })));
    }

    #[test]
    fn config_loader_rejects_an_oversized_regular_file_before_json_parsing() {
        let path = temporary_path("oversized");
        let oversized_bytes =
            usize::try_from(MAX_CONFIG_BYTES + 1).expect("the fixed 16 KiB test bound fits usize");
        fs::write(&path, vec![b' '; oversized_bytes])
            .expect("write oversized temporary configuration");

        let result = load_config(&path);

        fs::remove_file(&path).expect("remove oversized temporary configuration");
        assert!(matches!(
            result,
            Err(ConfigFileError::TooLarge {
                actual_bytes,
                maximum_bytes: MAX_CONFIG_BYTES,
            }) if actual_bytes == MAX_CONFIG_BYTES + 1
        ));
    }
}
