use robot_protocol::v2::{ActuatorConfigFingerprint, ControllerUid, DomainError, V2CommandLeaseMs};
use std::fmt;
use std::net::{IpAddr, SocketAddr};
use std::num::{NonZeroU8, NonZeroU16, NonZeroU32, NonZeroU64};
use std::str::FromStr;
use std::time::Duration;

pub const MAX_IO_TIMEOUT_NS: u64 = 5_000_000_000;
pub const MAX_STOP_RECOVERY_ATTEMPTS: u8 = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TimeoutNs(NonZeroU64);

impl TimeoutNs {
    pub fn try_new(nanoseconds: u64) -> Result<Self, ConfigError> {
        let nanoseconds = NonZeroU64::new(nanoseconds).ok_or(ConfigError::ZeroTimeout)?;
        if nanoseconds.get() > MAX_IO_TIMEOUT_NS {
            return Err(ConfigError::TimeoutAboveMaximum {
                nanoseconds: nanoseconds.get(),
                maximum_nanoseconds: MAX_IO_TIMEOUT_NS,
            });
        }
        Ok(Self(nanoseconds))
    }

    pub const fn get(self) -> u64 {
        self.0.get()
    }

    pub const fn as_duration(self) -> Duration {
        Duration::from_nanos(self.0.get())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UdpEndpoint(SocketAddr);

impl UdpEndpoint {
    pub const fn socket_addr(self) -> SocketAddr {
        self.0
    }
}

impl FromStr for UdpEndpoint {
    type Err = ConfigError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        if text.is_empty() || text.trim() != text {
            return Err(ConfigError::EndpointSyntax);
        }
        let address = text
            .parse::<SocketAddr>()
            .map_err(|_| ConfigError::EndpointSyntax)?;
        if address.port() == 0 {
            return Err(ConfigError::EndpointPortZero);
        }
        if !is_literal_loopback(address.ip()) {
            return Err(ConfigError::EndpointNotLoopback(address));
        }
        Ok(Self(address))
    }
}

const fn is_literal_loopback(address: IpAddr) -> bool {
    match address {
        IpAddr::V4(address) => address.is_loopback(),
        IpAddr::V6(address) => address.is_loopback(),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StopRecoveryPolicy {
    attempts: NonZeroU8,
    attempt_timeout: TimeoutNs,
}

impl StopRecoveryPolicy {
    pub fn try_new(attempts: u8, attempt_timeout: TimeoutNs) -> Result<Self, ConfigError> {
        let attempts = NonZeroU8::new(attempts).ok_or(ConfigError::StopRecoveryAttemptsZero)?;
        if attempts.get() > MAX_STOP_RECOVERY_ATTEMPTS {
            return Err(ConfigError::StopRecoveryAttemptsAboveMaximum {
                attempts: attempts.get(),
                maximum: MAX_STOP_RECOVERY_ATTEMPTS,
            });
        }
        Ok(Self {
            attempts,
            attempt_timeout,
        })
    }

    pub const fn attempts(self) -> NonZeroU8 {
        self.attempts
    }

    pub const fn attempt_timeout(self) -> TimeoutNs {
        self.attempt_timeout
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClientConfig {
    endpoint: UdpEndpoint,
    controller_uid: ControllerUid,
    expected_firmware_abi: NonZeroU16,
    expected_firmware_build_id: NonZeroU32,
    expected_actuator_config_fingerprint: ActuatorConfigFingerprint,
    status_timeout: TimeoutNs,
    acquire_timeout: TimeoutNs,
    applied_ack_timeout: TimeoutNs,
    stop_recovery: StopRecoveryPolicy,
    zero_acquisition_lease: V2CommandLeaseMs,
}

impl ClientConfig {
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        endpoint: UdpEndpoint,
        controller_uid: ControllerUid,
        expected_firmware_abi: NonZeroU16,
        expected_firmware_build_id: NonZeroU32,
        expected_actuator_config_fingerprint: ActuatorConfigFingerprint,
        status_timeout: TimeoutNs,
        acquire_timeout: TimeoutNs,
        applied_ack_timeout: TimeoutNs,
        stop_recovery: StopRecoveryPolicy,
        zero_acquisition_lease: V2CommandLeaseMs,
    ) -> Self {
        Self {
            endpoint,
            controller_uid,
            expected_firmware_abi,
            expected_firmware_build_id,
            expected_actuator_config_fingerprint,
            status_timeout,
            acquire_timeout,
            applied_ack_timeout,
            stop_recovery,
            zero_acquisition_lease,
        }
    }

    pub fn parse(input: ClientConfigInput<'_>) -> Result<Self, ConfigError> {
        let endpoint = input.command_endpoint.parse()?;
        let controller_uid = ControllerUid::try_new(parse_hex_exact(
            "controller_uid_hex",
            input.controller_uid_hex,
        )?)
        .map_err(ConfigError::ProtocolDomain)?;
        let expected_firmware_abi = NonZeroU16::new(parse_u16(
            "expected_firmware_abi",
            input.expected_firmware_abi,
        )?)
        .ok_or(ConfigError::ZeroIdentityField("expected_firmware_abi"))?;
        let expected_firmware_build_id = NonZeroU32::new(parse_u32(
            "expected_firmware_build_id",
            input.expected_firmware_build_id,
        )?)
        .ok_or(ConfigError::ZeroIdentityField("expected_firmware_build_id"))?;
        let expected_actuator_config_fingerprint =
            ActuatorConfigFingerprint::try_new(parse_hex_exact(
                "expected_actuator_config_fingerprint_hex",
                input.expected_actuator_config_fingerprint_hex,
            )?)
            .map_err(ConfigError::ProtocolDomain)?;
        let status_timeout = parse_timeout("status_timeout_ns", input.status_timeout_ns)?;
        let acquire_timeout = parse_timeout("acquire_timeout_ns", input.acquire_timeout_ns)?;
        let applied_ack_timeout =
            parse_timeout("applied_ack_timeout_ns", input.applied_ack_timeout_ns)?;
        let stop_attempt_timeout =
            parse_timeout("stop_attempt_timeout_ns", input.stop_attempt_timeout_ns)?;
        let stop_attempts = parse_u8(
            "max_stop_recovery_attempts",
            input.max_stop_recovery_attempts,
        )?;
        let stop_recovery = StopRecoveryPolicy::try_new(stop_attempts, stop_attempt_timeout)?;
        let zero_acquisition_lease = V2CommandLeaseMs::try_new(parse_u16(
            "zero_acquisition_lease_ms",
            input.zero_acquisition_lease_ms,
        )?)
        .map_err(ConfigError::ProtocolDomain)?;

        Ok(Self::new(
            endpoint,
            controller_uid,
            expected_firmware_abi,
            expected_firmware_build_id,
            expected_actuator_config_fingerprint,
            status_timeout,
            acquire_timeout,
            applied_ack_timeout,
            stop_recovery,
            zero_acquisition_lease,
        ))
    }

    pub const fn endpoint(&self) -> UdpEndpoint {
        self.endpoint
    }

    pub const fn controller_uid(&self) -> ControllerUid {
        self.controller_uid
    }

    pub const fn expected_firmware_abi(&self) -> u16 {
        self.expected_firmware_abi.get()
    }

    pub const fn expected_firmware_build_id(&self) -> u32 {
        self.expected_firmware_build_id.get()
    }

    pub const fn expected_actuator_config_fingerprint(&self) -> ActuatorConfigFingerprint {
        self.expected_actuator_config_fingerprint
    }

    pub const fn status_timeout(&self) -> TimeoutNs {
        self.status_timeout
    }

    pub const fn acquire_timeout(&self) -> TimeoutNs {
        self.acquire_timeout
    }

    pub const fn applied_ack_timeout(&self) -> TimeoutNs {
        self.applied_ack_timeout
    }

    pub const fn stop_recovery(&self) -> StopRecoveryPolicy {
        self.stop_recovery
    }

    pub const fn zero_acquisition_lease(&self) -> V2CommandLeaseMs {
        self.zero_acquisition_lease
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClientConfigInput<'a> {
    pub command_endpoint: &'a str,
    pub controller_uid_hex: &'a str,
    pub expected_firmware_abi: &'a str,
    pub expected_firmware_build_id: &'a str,
    pub expected_actuator_config_fingerprint_hex: &'a str,
    pub status_timeout_ns: &'a str,
    pub acquire_timeout_ns: &'a str,
    pub applied_ack_timeout_ns: &'a str,
    pub stop_attempt_timeout_ns: &'a str,
    pub max_stop_recovery_attempts: &'a str,
    pub zero_acquisition_lease_ms: &'a str,
}

fn parse_timeout(field: &'static str, text: &str) -> Result<TimeoutNs, ConfigError> {
    TimeoutNs::try_new(parse_u64(field, text)?)
}

fn parse_u8(field: &'static str, text: &str) -> Result<u8, ConfigError> {
    u8::try_from(parse_u64(field, text)?).map_err(|_| ConfigError::DecimalOutOfRange { field })
}

fn parse_u16(field: &'static str, text: &str) -> Result<u16, ConfigError> {
    u16::try_from(parse_u64(field, text)?).map_err(|_| ConfigError::DecimalOutOfRange { field })
}

fn parse_u32(field: &'static str, text: &str) -> Result<u32, ConfigError> {
    u32::try_from(parse_u64(field, text)?).map_err(|_| ConfigError::DecimalOutOfRange { field })
}

fn parse_u64(field: &'static str, text: &str) -> Result<u64, ConfigError> {
    if text.is_empty()
        || !text.bytes().all(|byte| byte.is_ascii_digit())
        || (text.len() > 1 && text.starts_with('0'))
    {
        return Err(ConfigError::NonCanonicalDecimal { field });
    }
    text.parse::<u64>()
        .map_err(|_| ConfigError::DecimalOutOfRange { field })
}

fn parse_hex_exact<const N: usize>(
    field: &'static str,
    text: &str,
) -> Result<[u8; N], ConfigError> {
    if text.len() != N * 2
        || !text
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(ConfigError::NonCanonicalHex {
            field,
            expected_digits: N * 2,
        });
    }
    let mut bytes = [0_u8; N];
    for (index, byte) in bytes.iter_mut().enumerate() {
        let high = hex_nibble(text.as_bytes()[index * 2]);
        let low = hex_nibble(text.as_bytes()[index * 2 + 1]);
        *byte = (high << 4) | low;
    }
    Ok(bytes)
}

const fn hex_nibble(byte: u8) -> u8 {
    match byte {
        b'0'..=b'9' => byte - b'0',
        b'a'..=b'f' => byte - b'a' + 10,
        _ => 0,
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConfigError {
    EndpointSyntax,
    EndpointPortZero,
    EndpointNotLoopback(SocketAddr),
    NonCanonicalDecimal {
        field: &'static str,
    },
    DecimalOutOfRange {
        field: &'static str,
    },
    NonCanonicalHex {
        field: &'static str,
        expected_digits: usize,
    },
    ZeroIdentityField(&'static str),
    ZeroTimeout,
    TimeoutAboveMaximum {
        nanoseconds: u64,
        maximum_nanoseconds: u64,
    },
    StopRecoveryAttemptsZero,
    StopRecoveryAttemptsAboveMaximum {
        attempts: u8,
        maximum: u8,
    },
    ProtocolDomain(DomainError),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EndpointSyntax => formatter.write_str(
                "command endpoint must be one literal IP socket address without whitespace",
            ),
            Self::EndpointPortZero => formatter.write_str("command endpoint port must be nonzero"),
            Self::EndpointNotLoopback(address) => write!(
                formatter,
                "command endpoint {address} is not loopback; use an authenticated local tunnel"
            ),
            Self::NonCanonicalDecimal { field } => write!(
                formatter,
                "{field} must be canonical unsigned decimal without whitespace, sign, or leading zeroes"
            ),
            Self::DecimalOutOfRange { field } => {
                write!(
                    formatter,
                    "{field} does not fit its supported integer range"
                )
            }
            Self::NonCanonicalHex {
                field,
                expected_digits,
            } => write!(
                formatter,
                "{field} must contain exactly {expected_digits} lowercase hexadecimal digits"
            ),
            Self::ZeroIdentityField(field) => write!(formatter, "{field} must be nonzero"),
            Self::ZeroTimeout => formatter.write_str("I/O timeout must be at least 1 ns"),
            Self::TimeoutAboveMaximum {
                nanoseconds,
                maximum_nanoseconds,
            } => write!(
                formatter,
                "I/O timeout {nanoseconds} ns exceeds the {maximum_nanoseconds} ns bound"
            ),
            Self::StopRecoveryAttemptsZero => {
                formatter.write_str("stop recovery must make at least one bounded attempt")
            }
            Self::StopRecoveryAttemptsAboveMaximum { attempts, maximum } => write!(
                formatter,
                "stop recovery attempt count {attempts} exceeds the bound {maximum}"
            ),
            Self::ProtocolDomain(source) => write!(formatter, "invalid V2 domain value: {source}"),
        }
    }
}

impl std::error::Error for ConfigError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ProtocolDomain(source) => Some(source),
            _ => None,
        }
    }
}
