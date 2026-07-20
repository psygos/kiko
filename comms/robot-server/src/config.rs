use std::fmt;
use std::fs::File;
use std::io::Read;
use std::net::SocketAddr;
use std::num::{NonZeroU16, NonZeroU32};
use std::path::{Path, PathBuf};
use std::time::Duration;

use clap::Parser;
use robot_protocol::v2::{
    ActuatorConfigFingerprint, ControllerUid, MaxAbsPwmPercent, NeutralOutput,
    PhysicalStopSemantics, PwmFrequencyHz, WatchdogNominalPeriodMs,
};
use serde::Deserialize;

pub const CONTROLLER_SERVER_CONFIG_V1: u32 = 1;
pub const MAX_CONTROLLER_SERVER_CONFIG_JSON_BYTES: usize = 8 * 1_024;
const MAX_PROFILE_CLAIM_ID_BYTES: usize = 128;

#[derive(Parser, Debug)]
#[command(
    about = "Kiko host services; V2 actuation is disabled unless an exact controller config is supplied"
)]
pub struct ServerArgs {
    /// Loopback V2 command socket. Use an authenticated tunnel for remote clients.
    #[arg(long, default_value = "127.0.0.1:8080")]
    command_bind: SocketAddr,
    /// Strict external controller authority JSON. Absence keeps actuation unavailable.
    #[arg(long, value_name = "CONFIG_JSON")]
    controller_config: Option<PathBuf>,
}

impl ServerArgs {
    pub fn parse_runtime() -> Result<ServerRuntimeConfig, ServerConfigError> {
        Self::parse().into_runtime()
    }

    fn into_runtime(self) -> Result<ServerRuntimeConfig, ServerConfigError> {
        if self.command_bind.port() == 0 {
            return Err(ServerConfigError::CommandPortZero);
        }
        if !self.command_bind.ip().is_loopback() {
            return Err(ServerConfigError::CommandBindIsNotLoopback(
                self.command_bind,
            ));
        }
        let controller = self
            .controller_config
            .as_deref()
            .map(read_controller_config)
            .transpose()?;
        Ok(ServerRuntimeConfig {
            command_bind: self.command_bind,
            controller,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServerRuntimeConfig {
    command_bind: SocketAddr,
    controller: Option<ControllerServerConfigV1>,
}

impl ServerRuntimeConfig {
    pub const fn command_bind(&self) -> SocketAddr {
        self.command_bind
    }

    pub const fn controller(&self) -> Option<&ControllerServerConfigV1> {
        self.controller.as_ref()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ControllerServerConfigV1 {
    serial_device: PathBuf,
    controller_uid: ControllerUid,
    firmware_abi: NonZeroU16,
    firmware_build_id: NonZeroU32,
    actuator_config_fingerprint: ActuatorConfigFingerprint,
    hardware_profile_claim_id: String,
    heartbeat_period: Duration,
    maximum_heartbeat_age: Duration,
    serial_applied_ack_timeout: Duration,
    controller_clock_abs_error_ppm_bound: NonZeroU32,
    deadline_quantization_margin_ms: NonZeroU16,
    expected_max_abs_pwm_percent: MaxAbsPwmPercent,
    expected_pwm_frequency: PwmFrequencyHz,
    expected_watchdog_nominal_period: WatchdogNominalPeriodMs,
    expected_neutral_output: NeutralOutput,
    expected_physical_stop_semantics: PhysicalStopSemantics,
}

impl ControllerServerConfigV1 {
    /// Parse one already bounded controller JSON document into the exact
    /// server-side domain contract.
    ///
    /// This does not open the serial device or prove that the configured
    /// identity is present. A caller that loaded the bytes from an immutable
    /// deployment asset can cross-bind the returned values to device
    /// inventory before either process is allowed to use them.
    pub fn parse_json(bytes: &[u8]) -> Result<Self, ServerConfigError> {
        if bytes.len() > MAX_CONTROLLER_SERVER_CONFIG_JSON_BYTES {
            return Err(ServerConfigError::InputBytesTooLarge {
                actual_bytes: bytes.len(),
                maximum_bytes: MAX_CONTROLLER_SERVER_CONFIG_JSON_BYTES,
            });
        }
        parse_controller_config(bytes)
    }

    pub fn serial_device(&self) -> &Path {
        &self.serial_device
    }

    pub const fn controller_uid(&self) -> ControllerUid {
        self.controller_uid
    }

    pub const fn firmware_abi(&self) -> NonZeroU16 {
        self.firmware_abi
    }

    pub const fn firmware_build_id(&self) -> NonZeroU32 {
        self.firmware_build_id
    }

    pub const fn actuator_config_fingerprint(&self) -> ActuatorConfigFingerprint {
        self.actuator_config_fingerprint
    }

    pub fn hardware_profile_claim_id(&self) -> &str {
        &self.hardware_profile_claim_id
    }

    pub const fn heartbeat_period(&self) -> Duration {
        self.heartbeat_period
    }

    pub const fn maximum_heartbeat_age(&self) -> Duration {
        self.maximum_heartbeat_age
    }

    pub const fn serial_applied_ack_timeout(&self) -> Duration {
        self.serial_applied_ack_timeout
    }

    pub const fn controller_clock_abs_error_ppm_bound(&self) -> NonZeroU32 {
        self.controller_clock_abs_error_ppm_bound
    }

    pub const fn deadline_quantization_margin_ms(&self) -> NonZeroU16 {
        self.deadline_quantization_margin_ms
    }

    pub const fn expected_max_abs_pwm_percent(&self) -> MaxAbsPwmPercent {
        self.expected_max_abs_pwm_percent
    }

    pub const fn expected_pwm_frequency(&self) -> PwmFrequencyHz {
        self.expected_pwm_frequency
    }

    pub const fn expected_watchdog_nominal_period(&self) -> WatchdogNominalPeriodMs {
        self.expected_watchdog_nominal_period
    }

    pub const fn expected_neutral_output(&self) -> NeutralOutput {
        self.expected_neutral_output
    }

    pub const fn expected_physical_stop_semantics(&self) -> PhysicalStopSemantics {
        self.expected_physical_stop_semantics
    }
}

#[derive(Debug)]
pub enum ServerConfigError {
    CommandPortZero,
    CommandBindIsNotLoopback(SocketAddr),
    Open {
        path: PathBuf,
        source: std::io::Error,
    },
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    InputTooLarge {
        path: PathBuf,
        actual_at_least_bytes: usize,
        maximum_bytes: usize,
    },
    InputBytesTooLarge {
        actual_bytes: usize,
        maximum_bytes: usize,
    },
    Json(serde_json::Error),
    UnsupportedSchemaVersion(u32),
    SerialDeviceIsNotAbsolute,
    InvalidHexLength {
        field: &'static str,
        expected_digits: usize,
        actual_digits: usize,
    },
    InvalidHexDigit {
        field: &'static str,
        digit_index: usize,
    },
    ZeroControllerUid,
    ZeroActuatorConfigFingerprint,
    ZeroField(&'static str),
    FieldAboveMaximum {
        field: &'static str,
        value: u64,
        maximum: u64,
    },
    InvalidHardwareProfileClaimId,
    UnverifiedPhysicalStopSemantics,
    HeartbeatAgeTooShort {
        heartbeat_period_ms: u16,
        maximum_heartbeat_age_ms: u16,
    },
    SerialAckTimeoutNotBelowHeartbeatAge {
        serial_ack_timeout_ms: u16,
        maximum_heartbeat_age_ms: u16,
    },
}

impl fmt::Display for ServerConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CommandPortZero => formatter.write_str("command bind port must be nonzero"),
            Self::CommandBindIsNotLoopback(address) => write!(
                formatter,
                "unauthenticated command bind {address} is not loopback; use an authenticated tunnel"
            ),
            Self::Open { path, source } => {
                write!(formatter, "cannot open {}: {source}", path.display())
            }
            Self::Read { path, source } => {
                write!(formatter, "cannot read {}: {source}", path.display())
            }
            Self::InputTooLarge {
                path,
                actual_at_least_bytes,
                maximum_bytes,
            } => write!(
                formatter,
                "controller config {} is at least {actual_at_least_bytes} bytes; maximum is {maximum_bytes}",
                path.display()
            ),
            Self::InputBytesTooLarge {
                actual_bytes,
                maximum_bytes,
            } => write!(
                formatter,
                "controller config input is {actual_bytes} bytes; maximum is {maximum_bytes}"
            ),
            Self::Json(source) => write!(formatter, "invalid controller config JSON: {source}"),
            Self::UnsupportedSchemaVersion(version) => {
                write!(formatter, "unsupported controller server config schema {version}")
            }
            Self::SerialDeviceIsNotAbsolute => {
                formatter.write_str("serial_device must be an absolute path")
            }
            Self::InvalidHexLength {
                field,
                expected_digits,
                actual_digits,
            } => write!(
                formatter,
                "{field} must contain exactly {expected_digits} hex digits, got {actual_digits}"
            ),
            Self::InvalidHexDigit { field, digit_index } => {
                write!(formatter, "{field} has a non-hex digit at index {digit_index}")
            }
            Self::ZeroControllerUid => {
                formatter.write_str("the all-zero controller UID is reserved")
            }
            Self::ZeroActuatorConfigFingerprint => {
                formatter.write_str("the all-zero actuator configuration fingerprint is reserved")
            }
            Self::ZeroField(field) => write!(formatter, "{field} must be nonzero"),
            Self::FieldAboveMaximum {
                field,
                value,
                maximum,
            } => write!(formatter, "{field} value {value} exceeds maximum {maximum}"),
            Self::InvalidHardwareProfileClaimId => formatter.write_str(
                "hardware_profile_claim_id must be bounded canonical ASCII",
            ),
            Self::UnverifiedPhysicalStopSemantics => formatter.write_str(
                "expected_physical_stop_semantics must be coast_verified or brake_verified",
            ),
            Self::HeartbeatAgeTooShort {
                heartbeat_period_ms,
                maximum_heartbeat_age_ms,
            } => write!(
                formatter,
                "maximum heartbeat age {maximum_heartbeat_age_ms} ms must cover at least two {heartbeat_period_ms} ms periods"
            ),
            Self::SerialAckTimeoutNotBelowHeartbeatAge {
                serial_ack_timeout_ms,
                maximum_heartbeat_age_ms,
            } => write!(
                formatter,
                "serial applied-ACK timeout {serial_ack_timeout_ms} ms must be below maximum heartbeat age {maximum_heartbeat_age_ms} ms"
            ),
        }
    }
}

impl std::error::Error for ServerConfigError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Open { source, .. } | Self::Read { source, .. } => Some(source),
            Self::Json(source) => Some(source),
            _ => None,
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields, rename_all = "snake_case")]
struct ControllerServerConfigV1Dto {
    schema_version: u32,
    serial_device: PathBuf,
    controller_uid_hex: String,
    firmware_abi: u16,
    firmware_build_id: u32,
    actuator_config_fingerprint_hex: String,
    hardware_profile_claim_id: String,
    heartbeat_period_ms: u16,
    maximum_heartbeat_age_ms: u16,
    serial_applied_ack_timeout_ms: u16,
    controller_clock_abs_error_ppm_bound: u32,
    deadline_quantization_margin_ms: u16,
    expected_max_abs_pwm_percent: u8,
    expected_pwm_frequency_hz: u32,
    expected_watchdog_nominal_timeout_ms: u16,
    expected_neutral_output: NeutralOutputDto,
    expected_physical_stop_semantics: PhysicalStopSemanticsDto,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum NeutralOutputDto {
    BothLow,
    BothHigh,
    HighImpedance,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum PhysicalStopSemanticsDto {
    Unverified,
    CoastVerified,
    BrakeVerified,
}

fn read_controller_config(path: &Path) -> Result<ControllerServerConfigV1, ServerConfigError> {
    let mut file = File::open(path).map_err(|source| ServerConfigError::Open {
        path: path.to_path_buf(),
        source,
    })?;
    let read_limit = u64::try_from(MAX_CONTROLLER_SERVER_CONFIG_JSON_BYTES + 1)
        .expect("small controller config limit fits u64");
    let mut bytes = Vec::with_capacity(MAX_CONTROLLER_SERVER_CONFIG_JSON_BYTES.min(4_096));
    file.by_ref()
        .take(read_limit)
        .read_to_end(&mut bytes)
        .map_err(|source| ServerConfigError::Read {
            path: path.to_path_buf(),
            source,
        })?;
    if bytes.len() > MAX_CONTROLLER_SERVER_CONFIG_JSON_BYTES {
        return Err(ServerConfigError::InputTooLarge {
            path: path.to_path_buf(),
            actual_at_least_bytes: bytes.len(),
            maximum_bytes: MAX_CONTROLLER_SERVER_CONFIG_JSON_BYTES,
        });
    }
    parse_controller_config(&bytes)
}

fn parse_controller_config(bytes: &[u8]) -> Result<ControllerServerConfigV1, ServerConfigError> {
    let dto: ControllerServerConfigV1Dto =
        serde_json::from_slice(bytes).map_err(ServerConfigError::Json)?;
    if dto.schema_version != CONTROLLER_SERVER_CONFIG_V1 {
        return Err(ServerConfigError::UnsupportedSchemaVersion(
            dto.schema_version,
        ));
    }
    if !dto.serial_device.is_absolute() {
        return Err(ServerConfigError::SerialDeviceIsNotAbsolute);
    }
    let controller_uid = ControllerUid::try_new(parse_hex_exact(
        "controller_uid_hex",
        &dto.controller_uid_hex,
    )?)
    .map_err(|_| ServerConfigError::ZeroControllerUid)?;
    let actuator_config_fingerprint = ActuatorConfigFingerprint::try_new(parse_hex_exact(
        "actuator_config_fingerprint_hex",
        &dto.actuator_config_fingerprint_hex,
    )?)
    .map_err(|_| ServerConfigError::ZeroActuatorConfigFingerprint)?;
    let firmware_abi = nonzero_u16("firmware_abi", dto.firmware_abi)?;
    let firmware_build_id = NonZeroU32::new(dto.firmware_build_id)
        .ok_or(ServerConfigError::ZeroField("firmware_build_id"))?;
    if dto.hardware_profile_claim_id.is_empty()
        || dto.hardware_profile_claim_id.len() > MAX_PROFILE_CLAIM_ID_BYTES
        || !dto.hardware_profile_claim_id.bytes().all(|byte| {
            byte.is_ascii_alphanumeric()
                || matches!(byte, b'-' | b'_' | b'.' | b':' | b'/' | b'@' | b'+')
        })
    {
        return Err(ServerConfigError::InvalidHardwareProfileClaimId);
    }
    let heartbeat_period_ms =
        bounded_nonzero_u16("heartbeat_period_ms", dto.heartbeat_period_ms, 100)?;
    let maximum_heartbeat_age_ms = bounded_nonzero_u16(
        "maximum_heartbeat_age_ms",
        dto.maximum_heartbeat_age_ms,
        1_000,
    )?;
    if u32::from(maximum_heartbeat_age_ms.get())
        < u32::from(heartbeat_period_ms.get()).saturating_mul(2)
    {
        return Err(ServerConfigError::HeartbeatAgeTooShort {
            heartbeat_period_ms: heartbeat_period_ms.get(),
            maximum_heartbeat_age_ms: maximum_heartbeat_age_ms.get(),
        });
    }
    let serial_applied_ack_timeout_ms = bounded_nonzero_u16(
        "serial_applied_ack_timeout_ms",
        dto.serial_applied_ack_timeout_ms,
        100,
    )?;
    if serial_applied_ack_timeout_ms >= maximum_heartbeat_age_ms {
        return Err(ServerConfigError::SerialAckTimeoutNotBelowHeartbeatAge {
            serial_ack_timeout_ms: serial_applied_ack_timeout_ms.get(),
            maximum_heartbeat_age_ms: maximum_heartbeat_age_ms.get(),
        });
    }
    let controller_clock_abs_error_ppm_bound =
        NonZeroU32::new(dto.controller_clock_abs_error_ppm_bound).ok_or(
            ServerConfigError::ZeroField("controller_clock_abs_error_ppm_bound"),
        )?;
    if controller_clock_abs_error_ppm_bound.get() > 100_000 {
        return Err(ServerConfigError::FieldAboveMaximum {
            field: "controller_clock_abs_error_ppm_bound",
            value: u64::from(controller_clock_abs_error_ppm_bound.get()),
            maximum: 100_000,
        });
    }
    let deadline_quantization_margin_ms = bounded_nonzero_u16(
        "deadline_quantization_margin_ms",
        dto.deadline_quantization_margin_ms,
        10,
    )?;
    let expected_max_abs_pwm_percent = MaxAbsPwmPercent::try_new(dto.expected_max_abs_pwm_percent)
        .map_err(|_| ServerConfigError::FieldAboveMaximum {
            field: "expected_max_abs_pwm_percent",
            value: u64::from(dto.expected_max_abs_pwm_percent),
            maximum: 100,
        })?;
    if expected_max_abs_pwm_percent.get() == 0 {
        return Err(ServerConfigError::ZeroField("expected_max_abs_pwm_percent"));
    }
    let expected_pwm_frequency_raw =
        u16::try_from(dto.expected_pwm_frequency_hz).map_err(|_| {
            ServerConfigError::FieldAboveMaximum {
                field: "expected_pwm_frequency_hz",
                value: u64::from(dto.expected_pwm_frequency_hz),
                maximum: u64::from(u16::MAX),
            }
        })?;
    let expected_pwm_frequency = PwmFrequencyHz::try_new(expected_pwm_frequency_raw)
        .map_err(|_| ServerConfigError::ZeroField("expected_pwm_frequency_hz"))?;
    let expected_watchdog_nominal_period = WatchdogNominalPeriodMs::try_new(
        dto.expected_watchdog_nominal_timeout_ms,
    )
    .map_err(|_| {
        if dto.expected_watchdog_nominal_timeout_ms == 0 {
            ServerConfigError::ZeroField("expected_watchdog_nominal_timeout_ms")
        } else {
            ServerConfigError::FieldAboveMaximum {
                field: "expected_watchdog_nominal_timeout_ms",
                value: u64::from(dto.expected_watchdog_nominal_timeout_ms),
                maximum: 1_000,
            }
        }
    })?;
    let expected_neutral_output = match dto.expected_neutral_output {
        NeutralOutputDto::BothLow => NeutralOutput::BothLow,
        NeutralOutputDto::BothHigh => NeutralOutput::BothHigh,
        NeutralOutputDto::HighImpedance => NeutralOutput::HighImpedance,
    };
    let expected_physical_stop_semantics = match dto.expected_physical_stop_semantics {
        PhysicalStopSemanticsDto::Unverified => {
            return Err(ServerConfigError::UnverifiedPhysicalStopSemantics);
        }
        PhysicalStopSemanticsDto::CoastVerified => PhysicalStopSemantics::CoastVerified,
        PhysicalStopSemanticsDto::BrakeVerified => PhysicalStopSemantics::BrakeVerified,
    };

    Ok(ControllerServerConfigV1 {
        serial_device: dto.serial_device,
        controller_uid,
        firmware_abi,
        firmware_build_id,
        actuator_config_fingerprint,
        hardware_profile_claim_id: dto.hardware_profile_claim_id,
        heartbeat_period: Duration::from_millis(u64::from(heartbeat_period_ms.get())),
        maximum_heartbeat_age: Duration::from_millis(u64::from(maximum_heartbeat_age_ms.get())),
        serial_applied_ack_timeout: Duration::from_millis(u64::from(
            serial_applied_ack_timeout_ms.get(),
        )),
        controller_clock_abs_error_ppm_bound,
        deadline_quantization_margin_ms,
        expected_max_abs_pwm_percent,
        expected_pwm_frequency,
        expected_watchdog_nominal_period,
        expected_neutral_output,
        expected_physical_stop_semantics,
    })
}

fn nonzero_u16(field: &'static str, value: u16) -> Result<NonZeroU16, ServerConfigError> {
    NonZeroU16::new(value).ok_or(ServerConfigError::ZeroField(field))
}

fn bounded_nonzero_u16(
    field: &'static str,
    value: u16,
    maximum: u16,
) -> Result<NonZeroU16, ServerConfigError> {
    let value = nonzero_u16(field, value)?;
    if value.get() > maximum {
        Err(ServerConfigError::FieldAboveMaximum {
            field,
            value: u64::from(value.get()),
            maximum: u64::from(maximum),
        })
    } else {
        Ok(value)
    }
}

fn parse_hex_exact<const N: usize>(
    field: &'static str,
    value: &str,
) -> Result<[u8; N], ServerConfigError> {
    let expected_digits = N * 2;
    if value.len() != expected_digits {
        return Err(ServerConfigError::InvalidHexLength {
            field,
            expected_digits,
            actual_digits: value.len(),
        });
    }
    let mut result = [0_u8; N];
    for (index, output) in result.iter_mut().enumerate() {
        let high_index = index * 2;
        let low_index = high_index + 1;
        let high =
            hex_nibble(value.as_bytes()[high_index]).ok_or(ServerConfigError::InvalidHexDigit {
                field,
                digit_index: high_index,
            })?;
        let low =
            hex_nibble(value.as_bytes()[low_index]).ok_or(ServerConfigError::InvalidHexDigit {
                field,
                digit_index: low_index,
            })?;
        *output = (high << 4) | low;
    }
    Ok(result)
}

const fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    fn valid() -> Value {
        json!({
            "schema_version": 1,
            "serial_device": "/dev/ttyACM0",
            "controller_uid_hex": "00112233445566778899aabb",
            "firmware_abi": 2,
            "firmware_build_id": 42,
            "actuator_config_fingerprint_hex": "11223344556677889900aabbccddeeff",
            "hardware_profile_claim_id": "kiko-driver-profile-v1",
            "heartbeat_period_ms": 20,
            "maximum_heartbeat_age_ms": 60,
            "serial_applied_ack_timeout_ms": 30,
            "controller_clock_abs_error_ppm_bound": 50_000,
            "deadline_quantization_margin_ms": 2,
            "expected_max_abs_pwm_percent": 50,
            "expected_pwm_frequency_hz": 20_000,
            "expected_watchdog_nominal_timeout_ms": 250,
            "expected_neutral_output": "both_low",
            "expected_physical_stop_semantics": "coast_verified"
        })
    }

    fn parse(value: &Value) -> Result<ControllerServerConfigV1, ServerConfigError> {
        ControllerServerConfigV1::parse_json(
            &serde_json::to_vec(value).expect("serialize fixture"),
        )
    }

    #[test]
    fn exact_external_controller_authority_parses_once() {
        let config = parse(&valid()).expect("valid controller config");
        assert_eq!(config.serial_device(), Path::new("/dev/ttyACM0"));
        assert_eq!(config.heartbeat_period(), Duration::from_millis(20));
        assert_eq!(
            config.expected_physical_stop_semantics(),
            PhysicalStopSemantics::CoastVerified
        );
    }

    #[test]
    fn unknown_or_unverified_hardware_claims_reject() {
        let mut unknown = valid();
        unknown["unexpected"] = json!(true);
        assert!(matches!(parse(&unknown), Err(ServerConfigError::Json(_))));

        let mut unverified = valid();
        unverified["expected_physical_stop_semantics"] = json!("unverified");
        assert!(matches!(
            parse(&unverified),
            Err(ServerConfigError::UnverifiedPhysicalStopSemantics)
        ));
    }

    #[test]
    fn identity_and_deadline_claims_fail_closed() {
        let mut zero_uid = valid();
        zero_uid["controller_uid_hex"] = json!("00".repeat(12));
        assert!(matches!(
            parse(&zero_uid),
            Err(ServerConfigError::ZeroControllerUid)
        ));

        let mut stale = valid();
        stale["heartbeat_period_ms"] = json!(31);
        assert!(matches!(
            parse(&stale),
            Err(ServerConfigError::HeartbeatAgeTooShort { .. })
        ));

        let mut slow_ack = valid();
        slow_ack["serial_applied_ack_timeout_ms"] = json!(60);
        assert!(matches!(
            parse(&slow_ack),
            Err(ServerConfigError::SerialAckTimeoutNotBelowHeartbeatAge { .. })
        ));
    }

    #[test]
    fn remote_command_bind_is_structurally_rejected() {
        let args = ServerArgs {
            command_bind: "192.168.50.2:8080".parse().expect("fixture address"),
            controller_config: None,
        };
        assert!(matches!(
            args.into_runtime(),
            Err(ServerConfigError::CommandBindIsNotLoopback(_))
        ));
    }

    #[test]
    fn shared_slice_boundary_rejects_oversized_input_before_json_decode() {
        let bytes = vec![b' '; MAX_CONTROLLER_SERVER_CONFIG_JSON_BYTES + 1];
        assert!(matches!(
            ControllerServerConfigV1::parse_json(&bytes),
            Err(ServerConfigError::InputBytesTooLarge {
                actual_bytes,
                maximum_bytes: MAX_CONTROLLER_SERVER_CONFIG_JSON_BYTES,
            }) if actual_bytes == bytes.len()
        ));
    }
}
