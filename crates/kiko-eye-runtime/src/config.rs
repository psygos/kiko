use std::fmt;
use std::time::Duration;

use kiko_expression_runtime::{
    ExpectedEyeIdentity, EyeSessionPlan, REQUIRED_EYE_CAPABILITIES, SessionNonce, SessionPlanError,
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
const SESSION_MATERIAL_BYTES: usize = 20;
const MAX_OS_SESSION_MATERIAL_ATTEMPTS: u8 = 8;

/// Weak deployment values which are safe to retain across process starts.
/// Parse them once with [`StaticEyeRuntimeConfig::parse`]. Per-start nonces and
/// the control epoch are deliberately absent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StaticEyeRuntimeConfigInput {
    pub device_path: String,
    pub baud_rate_bps: u32,
    pub response_timeout_ms: u64,
    pub write_timeout_ms: u64,
    pub write_attempts: u8,
    pub empty_delimiter_budget: u8,
    pub expected_device_uid: [u8; 16],
    pub expected_firmware_build_id: [u8; 32],
    pub expected_capabilities_bits: u32,
    pub intent_lease_ms: u16,
}

/// Weak values returned by a session-material generator. These values are
/// parsed once into [`EyeSessionMaterial`] before they can enter an actor
/// configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EyeSessionMaterialInput {
    pub identity_nonce: u64,
    pub acquire_nonce: u64,
    pub control_epoch: u32,
}

/// Valid material for exactly one attempted KEP2 control session.
///
/// This type is intentionally neither `Clone` nor `Copy`. Validity means the
/// identifiers are non-zero and the two phase nonces differ; freshness is a
/// responsibility of the generator which creates it.
#[derive(Debug, PartialEq, Eq)]
pub struct EyeSessionMaterial {
    identity_nonce: SessionNonce,
    acquire_nonce: SessionNonce,
    control_epoch: ControlEpoch,
}

impl EyeSessionMaterial {
    pub fn parse(input: EyeSessionMaterialInput) -> Result<Self, EyeSessionMaterialError> {
        let identity_nonce = SessionNonce::try_new(input.identity_nonce)
            .map_err(EyeSessionMaterialError::SessionPlan)?;
        let acquire_nonce = SessionNonce::try_new(input.acquire_nonce)
            .map_err(EyeSessionMaterialError::SessionPlan)?;
        if identity_nonce == acquire_nonce {
            return Err(EyeSessionMaterialError::SessionPlan(
                SessionPlanError::ReusedNonce {
                    value: identity_nonce.get(),
                },
            ));
        }
        let control_epoch = ControlEpoch::try_new(input.control_epoch)
            .map_err(EyeSessionMaterialError::ProtocolDomain)?;
        Ok(Self {
            identity_nonce,
            acquire_nonce,
            control_epoch,
        })
    }

    pub const fn identity_nonce(&self) -> SessionNonce {
        self.identity_nonce
    }

    pub const fn acquire_nonce(&self) -> SessionNonce {
        self.acquire_nonce
    }

    pub const fn control_epoch(&self) -> ControlEpoch {
        self.control_epoch
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EyeSessionMaterialError {
    ProtocolDomain(DomainError),
    SessionPlan(SessionPlanError),
}

impl fmt::Display for EyeSessionMaterialError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid one-shot KEP2 session material: {self:?}"
        )
    }
}

impl std::error::Error for EyeSessionMaterialError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ProtocolDomain(source) => Some(source),
            Self::SessionPlan(source) => Some(source),
        }
    }
}

/// Produces material for one new session attempt. Implementations must not
/// reuse material across calls or process starts.
pub trait EyeSessionMaterialGenerator {
    type Error;

    fn generate(&mut self) -> Result<EyeSessionMaterial, Self::Error>;
}

/// Operating-system CSPRNG-backed production generator.
#[derive(Clone, Copy, Debug, Default)]
pub struct OsEyeSessionMaterialGenerator;

#[derive(Debug)]
pub enum OsEyeSessionMaterialError {
    Entropy(getrandom::Error),
    RejectedSamples { attempts: u8 },
}

impl fmt::Display for OsEyeSessionMaterialError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Entropy(source) => write!(formatter, "OS entropy failed: {source}"),
            Self::RejectedSamples { attempts } => write!(
                formatter,
                "OS entropy produced invalid KEP2 session material for {attempts} attempts"
            ),
        }
    }
}

impl std::error::Error for OsEyeSessionMaterialError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Entropy(source) => Some(source),
            Self::RejectedSamples { .. } => None,
        }
    }
}

impl EyeSessionMaterialGenerator for OsEyeSessionMaterialGenerator {
    type Error = OsEyeSessionMaterialError;

    fn generate(&mut self) -> Result<EyeSessionMaterial, Self::Error> {
        for _ in 0..MAX_OS_SESSION_MATERIAL_ATTEMPTS {
            let mut bytes = [0_u8; SESSION_MATERIAL_BYTES];
            getrandom::fill(&mut bytes).map_err(OsEyeSessionMaterialError::Entropy)?;
            let input = EyeSessionMaterialInput {
                identity_nonce: u64::from_le_bytes(
                    bytes[0..8].try_into().expect("fixed identity nonce slice"),
                ),
                acquire_nonce: u64::from_le_bytes(
                    bytes[8..16].try_into().expect("fixed acquire nonce slice"),
                ),
                control_epoch: u32::from_le_bytes(
                    bytes[16..20].try_into().expect("fixed control epoch slice"),
                ),
            };
            if let Ok(material) = EyeSessionMaterial::parse(input) {
                return Ok(material);
            }
        }
        Err(OsEyeSessionMaterialError::RejectedSamples {
            attempts: MAX_OS_SESSION_MATERIAL_ATTEMPTS,
        })
    }
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

/// Fully parsed deployment policy. It cannot be passed to an eye actor until
/// a generator supplies valid material for this process start.
///
/// ```compile_fail
/// use kiko_eye_runtime::{
///     StaticEyeRuntimeConfig, TokioClock, start_serial_eye_actor,
/// };
///
/// fn cannot_start_static_policy(config: StaticEyeRuntimeConfig) {
///     let _ = start_serial_eye_actor(config, TokioClock::new());
/// }
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StaticEyeRuntimeConfig {
    device: DeviceIdentity,
    baud_rate: BaudRate,
    response_timeout: OperationTimeout,
    write_timeout: OperationTimeout,
    write_attempts: WriteAttemptLimit,
    empty_delimiter_budget: u8,
    expected_identity: ExpectedEyeIdentity,
    intent_lease: IntentLeaseMs,
}

impl StaticEyeRuntimeConfig {
    pub fn parse(input: StaticEyeRuntimeConfigInput) -> Result<Self, ConfigParseError> {
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
        let actual_capabilities = capabilities.bits();
        if actual_capabilities & REQUIRED_EYE_CAPABILITIES != REQUIRED_EYE_CAPABILITIES {
            return Err(ConfigParseError::SessionPlan(
                SessionPlanError::MissingRequiredCapabilities {
                    required: REQUIRED_EYE_CAPABILITIES,
                    actual: actual_capabilities,
                },
            ));
        }
        let expected_identity =
            ExpectedEyeIdentity::new(device_uid, firmware_build_id, capabilities);
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
            expected_identity,
            intent_lease,
        })
    }

    /// Build one non-cloneable actor configuration from newly generated
    /// one-shot session material.
    pub fn new_session<G>(&self, generator: &mut G) -> Result<EyeRuntimeConfig, G::Error>
    where
        G: EyeSessionMaterialGenerator,
    {
        let material = generator.generate()?;
        let session_plan = EyeSessionPlan::try_new(
            self.expected_identity,
            material.identity_nonce,
            material.acquire_nonce,
            material.control_epoch,
        )
        .expect("static identity and one-shot material were parsed into compatible domain types");
        Ok(EyeRuntimeConfig {
            static_config: self.clone(),
            session_plan,
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

    pub const fn expected_identity(&self) -> ExpectedEyeIdentity {
        self.expected_identity
    }

    pub const fn intent_lease(&self) -> IntentLeaseMs {
        self.intent_lease
    }
}

/// Actor-ready configuration for one session attempt.
///
/// It is deliberately not cloneable: the only way to obtain another value is
/// to return to [`StaticEyeRuntimeConfig::new_session`] and invoke the
/// generator again.
#[derive(Debug, PartialEq, Eq)]
pub struct EyeRuntimeConfig {
    static_config: StaticEyeRuntimeConfig,
    session_plan: EyeSessionPlan,
}

impl EyeRuntimeConfig {
    pub const fn static_config(&self) -> &StaticEyeRuntimeConfig {
        &self.static_config
    }

    pub const fn device(&self) -> &DeviceIdentity {
        self.static_config.device()
    }

    pub const fn baud_rate(&self) -> BaudRate {
        self.static_config.baud_rate()
    }

    pub const fn response_timeout(&self) -> OperationTimeout {
        self.static_config.response_timeout()
    }

    pub const fn write_timeout(&self) -> OperationTimeout {
        self.static_config.write_timeout()
    }

    pub const fn write_attempts(&self) -> WriteAttemptLimit {
        self.static_config.write_attempts()
    }

    pub const fn empty_delimiter_budget(&self) -> u8 {
        self.static_config.empty_delimiter_budget()
    }

    pub const fn session_plan(&self) -> EyeSessionPlan {
        self.session_plan
    }

    pub const fn intent_lease(&self) -> IntentLeaseMs {
        self.static_config.intent_lease()
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

    fn input(path: &str) -> StaticEyeRuntimeConfigInput {
        StaticEyeRuntimeConfigInput {
            device_path: path.to_owned(),
            baud_rate_bps: 115_200,
            response_timeout_ms: 20,
            write_timeout_ms: 5,
            write_attempts: 2,
            empty_delimiter_budget: 2,
            expected_device_uid: [1; 16],
            expected_firmware_build_id: [2; 32],
            expected_capabilities_bits: REQUIRED_EYE_CAPABILITIES,
            intent_lease_ms: 100,
        }
    }

    #[test]
    fn parses_only_exact_stable_platform_paths() {
        let linux = StaticEyeRuntimeConfig::parse(input("/dev/serial/by-id/kiko-eye-uid"))
            .expect("Linux by-id");
        assert_eq!(linux.device().kind(), DeviceIdentityKind::LinuxById);
        assert_eq!(linux.device().stable_name(), "kiko-eye-uid");

        let mac = StaticEyeRuntimeConfig::parse(input("/dev/cu.usbmodem-kiko-eye"))
            .expect("macOS callout");
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
                StaticEyeRuntimeConfig::parse(input(invalid)).is_err(),
                "accepted {invalid}"
            );
        }
    }

    #[test]
    fn static_policy_rejects_invalid_identity_and_capabilities() {
        let mut value = input("/dev/serial/by-id/eye");
        value.expected_device_uid = [0; 16];
        assert!(matches!(
            StaticEyeRuntimeConfig::parse(value),
            Err(ConfigParseError::ProtocolDomain(DomainError::ZeroDeviceUid))
        ));

        let mut value = input("/dev/serial/by-id/eye");
        value.expected_capabilities_bits &= !Capabilities::GAZE;
        assert!(matches!(
            StaticEyeRuntimeConfig::parse(value),
            Err(ConfigParseError::SessionPlan(
                SessionPlanError::MissingRequiredCapabilities { .. }
            ))
        ));

        let mut value = input("/dev/serial/by-id/eye");
        value.expected_firmware_build_id = [0; 32];
        assert!(matches!(
            StaticEyeRuntimeConfig::parse(value),
            Err(ConfigParseError::ProtocolDomain(
                DomainError::ZeroFirmwareBuildId
            ))
        ));

        let mut value = input("/dev/serial/by-id/eye");
        value.expected_capabilities_bits = 1_u32 << 31;
        assert!(matches!(
            StaticEyeRuntimeConfig::parse(value),
            Err(ConfigParseError::ProtocolDomain(
                DomainError::UnknownCapabilityBits { .. }
            ))
        ));

        let mut value = input("/dev/serial/by-id/eye");
        value.intent_lease_ms = 1;
        assert!(matches!(
            StaticEyeRuntimeConfig::parse(value),
            Err(ConfigParseError::ProtocolDomain(
                DomainError::IntentLeaseOutOfRange { .. }
            ))
        ));
    }

    #[test]
    fn one_shot_material_rejects_zero_and_reused_ids() {
        assert!(matches!(
            EyeSessionMaterial::parse(EyeSessionMaterialInput {
                identity_nonce: 0,
                acquire_nonce: 2,
                control_epoch: 3,
            }),
            Err(EyeSessionMaterialError::SessionPlan(
                SessionPlanError::ZeroNonce
            ))
        ));
        assert!(matches!(
            EyeSessionMaterial::parse(EyeSessionMaterialInput {
                identity_nonce: 1,
                acquire_nonce: 2,
                control_epoch: 0,
            }),
            Err(EyeSessionMaterialError::ProtocolDomain(
                DomainError::ZeroControlEpoch
            ))
        ));
        assert!(matches!(
            EyeSessionMaterial::parse(EyeSessionMaterialInput {
                identity_nonce: 7,
                acquire_nonce: 7,
                control_epoch: 3,
            }),
            Err(EyeSessionMaterialError::SessionPlan(
                SessionPlanError::ReusedNonce { value: 7 }
            ))
        ));
    }

    struct InjectedGenerator(std::array::IntoIter<EyeSessionMaterialInput, 2>);

    impl EyeSessionMaterialGenerator for InjectedGenerator {
        type Error = EyeSessionMaterialError;

        fn generate(&mut self) -> Result<EyeSessionMaterial, Self::Error> {
            EyeSessionMaterial::parse(self.0.next().expect("two configured session attempts"))
        }
    }

    #[test]
    fn injected_generator_gives_two_starts_distinct_nonzero_ids() {
        let policy = StaticEyeRuntimeConfig::parse(input("/dev/serial/by-id/eye"))
            .expect("static eye policy");
        let mut generator = InjectedGenerator(
            [
                EyeSessionMaterialInput {
                    identity_nonce: 11,
                    acquire_nonce: 12,
                    control_epoch: 13,
                },
                EyeSessionMaterialInput {
                    identity_nonce: 21,
                    acquire_nonce: 22,
                    control_epoch: 23,
                },
            ]
            .into_iter(),
        );
        let first = policy.new_session(&mut generator).expect("first session");
        let second = policy.new_session(&mut generator).expect("second session");
        let first = first.session_plan();
        let second = second.session_plan();

        assert_ne!(first.identity_nonce(), second.identity_nonce());
        assert_ne!(first.acquire_nonce(), second.acquire_nonce());
        assert_ne!(first.control_epoch(), second.control_epoch());
        for value in [
            first.identity_nonce().get(),
            first.acquire_nonce().get(),
            second.identity_nonce().get(),
            second.acquire_nonce().get(),
        ] {
            assert_ne!(value, 0);
        }
        assert_ne!(first.control_epoch().get(), 0);
        assert_ne!(second.control_epoch().get(), 0);
    }

    #[test]
    fn round_trip_timeout_budget_must_be_strictly_inside_device_lease() {
        let mut value = input("/dev/serial/by-id/eye");
        value.write_timeout_ms = 20;
        value.write_attempts = 4;
        value.response_timeout_ms = 20;
        value.intent_lease_ms = 100;
        assert_eq!(
            StaticEyeRuntimeConfig::parse(value),
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
            StaticEyeRuntimeConfig::parse(value),
            Err(ConfigParseError::BaudRateOutOfRange { .. })
        ));

        let mut value = input("/dev/serial/by-id/eye");
        value.response_timeout_ms = 0;
        assert!(matches!(
            StaticEyeRuntimeConfig::parse(value),
            Err(ConfigParseError::OperationTimeoutOutOfRange { .. })
        ));

        let mut value = input("/dev/serial/by-id/eye");
        value.write_attempts = 9;
        assert!(matches!(
            StaticEyeRuntimeConfig::parse(value),
            Err(ConfigParseError::WriteAttemptsOutOfRange { .. })
        ));

        let mut value = input("/dev/serial/by-id/eye");
        value.empty_delimiter_budget = 33;
        assert!(matches!(
            StaticEyeRuntimeConfig::parse(value),
            Err(ConfigParseError::EmptyDelimiterBudgetOutOfRange { .. })
        ));
    }
}
