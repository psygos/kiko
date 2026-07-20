use std::fmt;
use std::time::Duration;

use kiko_expression_runtime::{
    ExpectedEyeIdentity, EyeSessionPlan, SessionNonce, SessionPlanError,
};
use kiko_eye_protocol::{
    Capabilities, ControlEpoch, DeviceUid, DomainError, FirmwareBuildId, IntentLeaseMs,
};

const MAX_DEVICE_PATH_BYTES: usize = 512;
const MAX_DEVICE_ID_BYTES: usize = 255;
const MIN_BAUD_RATE_BPS: u32 = 9_600;
const MAX_BAUD_RATE_BPS: u32 = 3_000_000;
const MAX_OPERATION_TIMEOUT_MS: u64 = 5_000;
const MAX_WRITE_ATTEMPTS: u8 = 8;
const MAX_EMPTY_DELIMITER_BUDGET: u8 = 32;

/// Weak process-boundary values. Parse them once with
/// [`EyeRuntimeConfig::parse`] before constructing an actor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EyeRuntimeConfigInput {
    pub device_path: String,
    pub baud_rate_bps: u32,
    pub response_timeout_ms: u64,
    pub write_timeout_ms: u64,
    pub write_attempts: u8,
    pub empty_delimiter_budget: u8,
    pub expected_device_uid: [u8; 16],
    pub expected_firmware_build_id: [u8; 32],
    pub expected_capabilities_bits: u32,
    pub identity_nonce: u64,
    pub acquire_nonce: u64,
    pub control_epoch: u32,
    pub intent_lease_ms: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeviceIdentityKind {
    LinuxById,
    MacOsCallout,
}

/// One exact stable serial identity. Generic tty names and discovery are not
/// representable.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DeviceIdentity {
    path: Box<str>,
    stable_name: Box<str>,
    kind: DeviceIdentityKind,
}

impl DeviceIdentity {
    fn parse(path: String) -> Result<Self, ConfigParseError> {
        if path.is_empty() {
            return Err(ConfigParseError::EmptyDevicePath);
        }
        if path.len() > MAX_DEVICE_PATH_BYTES {
            return Err(ConfigParseError::DevicePathTooLong {
                actual_bytes: path.len(),
                maximum_bytes: MAX_DEVICE_PATH_BYTES,
            });
        }
        if path.as_bytes().contains(&0) {
            return Err(ConfigParseError::DevicePathContainsNul);
        }

        let (kind, stable_name) = if let Some(name) = path.strip_prefix("/dev/serial/by-id/") {
            (DeviceIdentityKind::LinuxById, name)
        } else if let Some(name) = path.strip_prefix("/dev/cu.") {
            (DeviceIdentityKind::MacOsCallout, name)
        } else {
            return Err(ConfigParseError::UnsupportedDevicePath { path });
        };

        if stable_name.is_empty()
            || stable_name == "."
            || stable_name == ".."
            || stable_name.contains('/')
        {
            return Err(ConfigParseError::InvalidDeviceIdentity {
                identity: stable_name.to_owned(),
            });
        }
        if stable_name.len() > MAX_DEVICE_ID_BYTES {
            return Err(ConfigParseError::DeviceIdentityTooLong {
                actual_bytes: stable_name.len(),
                maximum_bytes: MAX_DEVICE_ID_BYTES,
            });
        }

        Ok(Self {
            stable_name: stable_name.into(),
            path: path.into(),
            kind,
        })
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    pub fn stable_name(&self) -> &str {
        &self.stable_name
    }

    pub const fn kind(&self) -> DeviceIdentityKind {
        self.kind
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BaudRate(u32);

impl BaudRate {
    fn parse(value: u32) -> Result<Self, ConfigParseError> {
        if !(MIN_BAUD_RATE_BPS..=MAX_BAUD_RATE_BPS).contains(&value) {
            return Err(ConfigParseError::BaudRateOutOfRange {
                value,
                minimum: MIN_BAUD_RATE_BPS,
                maximum: MAX_BAUD_RATE_BPS,
            });
        }
        Ok(Self(value))
    }

    pub const fn get(self) -> u32 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OperationTimeout(Duration);

impl OperationTimeout {
    fn parse(field: &'static str, milliseconds: u64) -> Result<Self, ConfigParseError> {
        if !(1..=MAX_OPERATION_TIMEOUT_MS).contains(&milliseconds) {
            return Err(ConfigParseError::OperationTimeoutOutOfRange {
                field,
                milliseconds,
                minimum_ms: 1,
                maximum_ms: MAX_OPERATION_TIMEOUT_MS,
            });
        }
        Ok(Self(Duration::from_millis(milliseconds)))
    }

    pub const fn get(self) -> Duration {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WriteAttemptLimit(u8);

impl WriteAttemptLimit {
    fn parse(value: u8) -> Result<Self, ConfigParseError> {
        if !(1..=MAX_WRITE_ATTEMPTS).contains(&value) {
            return Err(ConfigParseError::WriteAttemptsOutOfRange {
                value,
                minimum: 1,
                maximum: MAX_WRITE_ATTEMPTS,
            });
        }
        Ok(Self(value))
    }

    pub const fn get(self) -> u8 {
        self.0
    }
}

/// Fully parsed, internally consistent actor configuration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EyeRuntimeConfig {
    device: DeviceIdentity,
    baud_rate: BaudRate,
    response_timeout: OperationTimeout,
    write_timeout: OperationTimeout,
    write_attempts: WriteAttemptLimit,
    empty_delimiter_budget: u8,
    session_plan: EyeSessionPlan,
    intent_lease: IntentLeaseMs,
}

impl EyeRuntimeConfig {
    pub fn parse(input: EyeRuntimeConfigInput) -> Result<Self, ConfigParseError> {
        let device = DeviceIdentity::parse(input.device_path)?;
        let baud_rate = BaudRate::parse(input.baud_rate_bps)?;
        let response_timeout =
            OperationTimeout::parse("response_timeout_ms", input.response_timeout_ms)?;
        let write_timeout = OperationTimeout::parse("write_timeout_ms", input.write_timeout_ms)?;
        let write_attempts = WriteAttemptLimit::parse(input.write_attempts)?;
        if input.empty_delimiter_budget > MAX_EMPTY_DELIMITER_BUDGET {
            return Err(ConfigParseError::EmptyDelimiterBudgetOutOfRange {
                value: input.empty_delimiter_budget,
                maximum: MAX_EMPTY_DELIMITER_BUDGET,
            });
        }

        let device_uid = DeviceUid::try_new(input.expected_device_uid)
            .map_err(ConfigParseError::ProtocolDomain)?;
        let firmware_build_id = FirmwareBuildId::try_new(input.expected_firmware_build_id)
            .map_err(ConfigParseError::ProtocolDomain)?;
        let capabilities = Capabilities::try_from_bits(input.expected_capabilities_bits)
            .map_err(ConfigParseError::ProtocolDomain)?;
        let expected_identity =
            ExpectedEyeIdentity::new(device_uid, firmware_build_id, capabilities);
        let identity_nonce =
            SessionNonce::try_new(input.identity_nonce).map_err(ConfigParseError::SessionPlan)?;
        let acquire_nonce =
            SessionNonce::try_new(input.acquire_nonce).map_err(ConfigParseError::SessionPlan)?;
        let control_epoch =
            ControlEpoch::try_new(input.control_epoch).map_err(ConfigParseError::ProtocolDomain)?;
        let session_plan = EyeSessionPlan::try_new(
            expected_identity,
            identity_nonce,
            acquire_nonce,
            control_epoch,
        )
        .map_err(ConfigParseError::SessionPlan)?;
        let intent_lease = IntentLeaseMs::try_new(input.intent_lease_ms)
            .map_err(ConfigParseError::ProtocolDomain)?;

        let worst_case_ms = input
            .write_timeout_ms
            .checked_mul(u64::from(input.write_attempts))
            .and_then(|write_ms| write_ms.checked_add(input.response_timeout_ms))
            .ok_or(ConfigParseError::IntentRoundTripBudgetOverflow)?;
        if worst_case_ms >= u64::from(input.intent_lease_ms) {
            return Err(ConfigParseError::IntentRoundTripNotWithinLease {
                worst_case_ms,
                lease_ms: input.intent_lease_ms,
            });
        }

        Ok(Self {
            device,
            baud_rate,
            response_timeout,
            write_timeout,
            write_attempts,
            empty_delimiter_budget: input.empty_delimiter_budget,
            session_plan,
            intent_lease,
        })
    }

    pub const fn device(&self) -> &DeviceIdentity {
        &self.device
    }

    pub const fn baud_rate(&self) -> BaudRate {
        self.baud_rate
    }

    pub const fn response_timeout(&self) -> OperationTimeout {
        self.response_timeout
    }

    pub const fn write_timeout(&self) -> OperationTimeout {
        self.write_timeout
    }

    pub const fn write_attempts(&self) -> WriteAttemptLimit {
        self.write_attempts
    }

    pub const fn empty_delimiter_budget(&self) -> u8 {
        self.empty_delimiter_budget
    }

    pub const fn session_plan(&self) -> EyeSessionPlan {
        self.session_plan
    }

    pub const fn intent_lease(&self) -> IntentLeaseMs {
        self.intent_lease
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConfigParseError {
    EmptyDevicePath,
    DevicePathTooLong {
        actual_bytes: usize,
        maximum_bytes: usize,
    },
    DevicePathContainsNul,
    UnsupportedDevicePath {
        path: String,
    },
    InvalidDeviceIdentity {
        identity: String,
    },
    DeviceIdentityTooLong {
        actual_bytes: usize,
        maximum_bytes: usize,
    },
    BaudRateOutOfRange {
        value: u32,
        minimum: u32,
        maximum: u32,
    },
    OperationTimeoutOutOfRange {
        field: &'static str,
        milliseconds: u64,
        minimum_ms: u64,
        maximum_ms: u64,
    },
    WriteAttemptsOutOfRange {
        value: u8,
        minimum: u8,
        maximum: u8,
    },
    EmptyDelimiterBudgetOutOfRange {
        value: u8,
        maximum: u8,
    },
    ProtocolDomain(DomainError),
    SessionPlan(SessionPlanError),
    IntentRoundTripBudgetOverflow,
    IntentRoundTripNotWithinLease {
        worst_case_ms: u64,
        lease_ms: u16,
    },
}

impl fmt::Display for ConfigParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid KEP2 eye runtime configuration: {self:?}"
        )
    }
}

impl std::error::Error for ConfigParseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ProtocolDomain(source) => Some(source),
            Self::SessionPlan(source) => Some(source),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use kiko_expression_runtime::REQUIRED_EYE_CAPABILITIES;

    use super::*;

    fn input(path: &str) -> EyeRuntimeConfigInput {
        EyeRuntimeConfigInput {
            device_path: path.to_owned(),
            baud_rate_bps: 115_200,
            response_timeout_ms: 20,
            write_timeout_ms: 5,
            write_attempts: 2,
            empty_delimiter_budget: 2,
            expected_device_uid: [1; 16],
            expected_firmware_build_id: [2; 32],
            expected_capabilities_bits: REQUIRED_EYE_CAPABILITIES,
            identity_nonce: 11,
            acquire_nonce: 12,
            control_epoch: 13,
            intent_lease_ms: 100,
        }
    }

    #[test]
    fn parses_only_exact_stable_platform_paths() {
        let linux =
            EyeRuntimeConfig::parse(input("/dev/serial/by-id/kiko-eye-uid")).expect("Linux by-id");
        assert_eq!(linux.device().kind(), DeviceIdentityKind::LinuxById);
        assert_eq!(linux.device().stable_name(), "kiko-eye-uid");

        let mac =
            EyeRuntimeConfig::parse(input("/dev/cu.usbmodem-kiko-eye")).expect("macOS callout");
        assert_eq!(mac.device().kind(), DeviceIdentityKind::MacOsCallout);

        for invalid in [
            "/dev/ttyACM0",
            "/dev/ttyUSB0",
            "/dev/serial/by-id/",
            "/dev/serial/by-id/a/b",
            "/dev/cu.",
            "/dev/cu.a/b",
        ] {
            assert!(
                EyeRuntimeConfig::parse(input(invalid)).is_err(),
                "accepted {invalid}"
            );
        }
    }

    #[test]
    fn rejects_zero_identity_nonce_epoch_and_incomplete_capabilities() {
        let mut value = input("/dev/serial/by-id/eye");
        value.expected_device_uid = [0; 16];
        assert!(matches!(
            EyeRuntimeConfig::parse(value),
            Err(ConfigParseError::ProtocolDomain(DomainError::ZeroDeviceUid))
        ));

        let mut value = input("/dev/serial/by-id/eye");
        value.identity_nonce = 0;
        assert!(matches!(
            EyeRuntimeConfig::parse(value),
            Err(ConfigParseError::SessionPlan(SessionPlanError::ZeroNonce))
        ));

        let mut value = input("/dev/serial/by-id/eye");
        value.control_epoch = 0;
        assert!(matches!(
            EyeRuntimeConfig::parse(value),
            Err(ConfigParseError::ProtocolDomain(
                DomainError::ZeroControlEpoch
            ))
        ));

        let mut value = input("/dev/serial/by-id/eye");
        value.expected_capabilities_bits &= !Capabilities::GAZE;
        assert!(matches!(
            EyeRuntimeConfig::parse(value),
            Err(ConfigParseError::SessionPlan(
                SessionPlanError::MissingRequiredCapabilities { .. }
            ))
        ));

        let mut value = input("/dev/serial/by-id/eye");
        value.expected_firmware_build_id = [0; 32];
        assert!(matches!(
            EyeRuntimeConfig::parse(value),
            Err(ConfigParseError::ProtocolDomain(
                DomainError::ZeroFirmwareBuildId
            ))
        ));

        let mut value = input("/dev/serial/by-id/eye");
        value.acquire_nonce = value.identity_nonce;
        assert!(matches!(
            EyeRuntimeConfig::parse(value),
            Err(ConfigParseError::SessionPlan(
                SessionPlanError::ReusedNonce { .. }
            ))
        ));

        let mut value = input("/dev/serial/by-id/eye");
        value.expected_capabilities_bits = 1_u32 << 31;
        assert!(matches!(
            EyeRuntimeConfig::parse(value),
            Err(ConfigParseError::ProtocolDomain(
                DomainError::UnknownCapabilityBits { .. }
            ))
        ));

        let mut value = input("/dev/serial/by-id/eye");
        value.intent_lease_ms = 1;
        assert!(matches!(
            EyeRuntimeConfig::parse(value),
            Err(ConfigParseError::ProtocolDomain(
                DomainError::IntentLeaseOutOfRange { .. }
            ))
        ));
    }

    #[test]
    fn round_trip_timeout_budget_must_be_strictly_inside_device_lease() {
        let mut value = input("/dev/serial/by-id/eye");
        value.write_timeout_ms = 20;
        value.write_attempts = 4;
        value.response_timeout_ms = 20;
        value.intent_lease_ms = 100;
        assert_eq!(
            EyeRuntimeConfig::parse(value),
            Err(ConfigParseError::IntentRoundTripNotWithinLease {
                worst_case_ms: 100,
                lease_ms: 100,
            })
        );
    }

    #[test]
    fn rejects_every_unbounded_operational_setting() {
        let mut value = input("/dev/serial/by-id/eye");
        value.baud_rate_bps = 0;
        assert!(matches!(
            EyeRuntimeConfig::parse(value),
            Err(ConfigParseError::BaudRateOutOfRange { .. })
        ));

        let mut value = input("/dev/serial/by-id/eye");
        value.response_timeout_ms = 0;
        assert!(matches!(
            EyeRuntimeConfig::parse(value),
            Err(ConfigParseError::OperationTimeoutOutOfRange { .. })
        ));

        let mut value = input("/dev/serial/by-id/eye");
        value.write_attempts = 9;
        assert!(matches!(
            EyeRuntimeConfig::parse(value),
            Err(ConfigParseError::WriteAttemptsOutOfRange { .. })
        ));

        let mut value = input("/dev/serial/by-id/eye");
        value.empty_delimiter_budget = 33;
        assert!(matches!(
            EyeRuntimeConfig::parse(value),
            Err(ConfigParseError::EmptyDelimiterBudgetOutOfRange { .. })
        ));
    }
}
