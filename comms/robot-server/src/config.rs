use std::fmt;
use std::num::{NonZeroU16, NonZeroU32};
use std::path::{Path, PathBuf};
use std::time::Duration;

use robot_protocol::v2::{
    ActuatorConfigFingerprint, AppliedResult, ApplyPwm, ControllerHello, ControllerSessionClass,
    ControllerUid, Heartbeat, MaxAbsPwmPercent, NeutralOutput, ObservationalOdometry,
    PhysicalStopSemantics, PwmFrequencyHz, WatchdogNominalPeriodMs,
    CANONICAL_CONTROLLER_HELLO_PERIOD_MS, CANONICAL_ODOMETRY_REPORT_PERIOD_MS, CRC_BYTES,
    HEADER_BYTES, MAX_ATTENDED_WHEEL_ON_COMMISSIONING_PWM_PERCENT,
    MAX_OPERATOR_SUPERVISED_FOUR_PWM_PWM_PERCENT,
    MAX_UART_RECORD_BYTES as PROTOCOL_MAX_UART_RECORD_BYTES, VERSION as ROBOT_PROTOCOL_V2,
};
pub use robot_protocol::v2::{
    ATTENDED_WHEEL_ON_COMMISSIONING_FINGERPRINT_BYTES,
    ATTENDED_WHEEL_ON_COMMISSIONING_FIRMWARE_BUILD_ID,
    ATTENDED_WHEEL_ON_COMMISSIONING_MAX_COMMAND_STEP_PERCENT,
    OPERATOR_SUPERVISED_FOUR_PWM_FINGERPRINT_BYTES, OPERATOR_SUPERVISED_FOUR_PWM_FIRMWARE_BUILD_ID,
    OPERATOR_SUPERVISED_FOUR_PWM_MAX_COMMAND_STEP_PERCENT,
};
use serde::{Deserialize, Deserializer};

pub const CONTROLLER_SERVER_CONFIG_V1: u32 = 1;
pub const CONTROLLER_SERVER_CONFIG_V2: u32 = 2;
pub const CONTROLLER_SERVER_CONFIG_V3: u32 = 3;
pub const MAX_CONTROLLER_SERVER_CONFIG_JSON_BYTES: usize = 8 * 1_024;
const MAX_PROFILE_CLAIM_ID_BYTES: usize = 128;
pub const CONTROLLER_SERIAL_BAUD_BPS: u32 = 115_200;
const UART_8N1_BITS_PER_BYTE: u32 = 10;
const MAX_HOST_COMMAND_RATE_HZ: u16 = 100;
const MAX_UART_RECORD_OVERHEAD_BYTES: usize = HEADER_BYTES + CRC_BYTES + 2;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ControllerServerConfigV1 {
    serial_device: PathBuf,
    controller_uid: ControllerUid,
    firmware_abi: NonZeroU16,
    firmware_build_id: NonZeroU32,
    actuator_config_fingerprint: ActuatorConfigFingerprint,
    hardware_profile_claim_id: String,
    controller_ready_timeout: Duration,
    heartbeat_period: Duration,
    maximum_heartbeat_age: Duration,
    maximum_host_command_rate_hz: NonZeroU16,
    minimum_host_command_interval: Duration,
    serial_transmit_timeout: Duration,
    serial_applied_ack_timeout: Duration,
    controller_clock_abs_error_ppm_bound: NonZeroU32,
    deadline_quantization_margin_ms: NonZeroU16,
    expected_max_abs_pwm_percent: MaxAbsPwmPercent,
    expected_pwm_frequency: PwmFrequencyHz,
    expected_watchdog_nominal_period: WatchdogNominalPeriodMs,
    expected_neutral_output: NeutralOutput,
    expected_physical_stop_semantics: PhysicalStopSemantics,
    controller_session_class: ControllerSessionClass,
}

/// Explicit schema-V2 controller contract for the provisional four-PWM
/// qualification profile.
///
/// This type cannot be constructed from a production V1 document. Its parser
/// requires the exact candidate class, firmware build, fingerprint, protocol
/// ABI, unverified stop semantics, and bounded nonzero PWM claim.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ControllerServerConfigV2(ControllerServerConfigV1);

/// Explicit schema-V3 controller contract for one attended, wheel-on,
/// non-production commissioning session.
///
/// This identity remains disjoint from both production V1 and the wheels-off
/// candidate V2. It truthfully retains unverified physical-stop semantics and
/// the protocol's hard 20% electrical-command ceiling.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ControllerServerConfigV3(ControllerServerConfigV1);

/// One strictly versioned controller contract admitted at the process
/// boundary. V1 remains production-only, V2 remains candidate-only, and V3
/// remains attended-commissioning-only.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ControllerServerConfig {
    ProductionV1(ControllerServerConfigV1),
    OperatorSupervisedFourPwmCandidateV2(ControllerServerConfigV2),
    AttendedWheelOnCommissioningV3(ControllerServerConfigV3),
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
        let dto = parse_controller_config_dto(bytes)?;
        parse_controller_config_v1(dto)
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

    /// Maximum time after exclusive serial acquisition to observe an exact
    /// hello/session/ready/heartbeat sequence.
    pub const fn controller_ready_timeout(&self) -> Duration {
        self.controller_ready_timeout
    }

    pub const fn heartbeat_period(&self) -> Duration {
        self.heartbeat_period
    }

    pub const fn maximum_heartbeat_age(&self) -> Duration {
        self.maximum_heartbeat_age
    }

    pub const fn maximum_host_command_rate_hz(&self) -> NonZeroU16 {
        self.maximum_host_command_rate_hz
    }

    pub const fn minimum_host_command_interval(&self) -> Duration {
        self.minimum_host_command_interval
    }

    pub const fn serial_transmit_timeout(&self) -> Duration {
        self.serial_transmit_timeout
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

    pub const fn controller_session_class(&self) -> ControllerSessionClass {
        self.controller_session_class
    }

    /// Upper bound used to collect the controller actor and UDP task after a
    /// coordinated service shutdown.
    ///
    /// The budget is derived from already admitted controller timing bounds
    /// instead of introducing an unrelated fallback timeout: one
    /// heartbeat-age window, one serial-transmit window, one serial
    /// acknowledgement window, and one heartbeat period. Exceeding it is
    /// reported as an uncertain owner stop.
    pub fn coordinated_shutdown_budget(&self) -> Duration {
        self.maximum_heartbeat_age
            .checked_add(self.serial_transmit_timeout)
            .and_then(|budget| budget.checked_add(self.serial_applied_ack_timeout))
            .and_then(|budget| budget.checked_add(self.heartbeat_period))
            .expect("validated millisecond controller bounds fit Duration")
    }
}

impl ControllerServerConfigV2 {
    pub fn parse_json(bytes: &[u8]) -> Result<Self, ServerConfigError> {
        if bytes.len() > MAX_CONTROLLER_SERVER_CONFIG_JSON_BYTES {
            return Err(ServerConfigError::InputBytesTooLarge {
                actual_bytes: bytes.len(),
                maximum_bytes: MAX_CONTROLLER_SERVER_CONFIG_JSON_BYTES,
            });
        }
        let dto = parse_controller_config_dto(bytes)?;
        parse_controller_config_v2(dto).map(Self)
    }

    const fn contract(&self) -> &ControllerServerConfigV1 {
        &self.0
    }

    pub fn serial_device(&self) -> &Path {
        self.0.serial_device()
    }

    pub const fn controller_uid(&self) -> ControllerUid {
        self.0.controller_uid()
    }

    pub const fn firmware_abi(&self) -> NonZeroU16 {
        self.0.firmware_abi()
    }

    pub const fn controller_session_class(&self) -> ControllerSessionClass {
        self.0.controller_session_class()
    }

    pub const fn firmware_build_id(&self) -> NonZeroU32 {
        self.0.firmware_build_id()
    }

    pub const fn actuator_config_fingerprint(&self) -> ActuatorConfigFingerprint {
        self.0.actuator_config_fingerprint()
    }

    pub fn hardware_profile_claim_id(&self) -> &str {
        self.0.hardware_profile_claim_id()
    }

    pub const fn expected_max_abs_pwm_percent(&self) -> MaxAbsPwmPercent {
        self.0.expected_max_abs_pwm_percent()
    }

    pub const fn expected_physical_stop_semantics(&self) -> PhysicalStopSemantics {
        self.0.expected_physical_stop_semantics()
    }

    pub const fn serial_applied_ack_timeout(&self) -> Duration {
        self.0.serial_applied_ack_timeout()
    }

    pub const fn maximum_heartbeat_age(&self) -> Duration {
        self.0.maximum_heartbeat_age()
    }

    pub const fn minimum_host_command_interval(&self) -> Duration {
        self.0.minimum_host_command_interval()
    }

    pub fn coordinated_shutdown_budget(&self) -> Duration {
        self.0.coordinated_shutdown_budget()
    }

    /// Exact percentage-point delta accepted between consecutive candidate
    /// firmware commands. This is not a physical acceleration claim.
    pub const fn maximum_command_step_percent(&self) -> u8 {
        OPERATOR_SUPERVISED_FOUR_PWM_MAX_COMMAND_STEP_PERCENT
    }
}

impl ControllerServerConfigV3 {
    pub fn parse_json(bytes: &[u8]) -> Result<Self, ServerConfigError> {
        if bytes.len() > MAX_CONTROLLER_SERVER_CONFIG_JSON_BYTES {
            return Err(ServerConfigError::InputBytesTooLarge {
                actual_bytes: bytes.len(),
                maximum_bytes: MAX_CONTROLLER_SERVER_CONFIG_JSON_BYTES,
            });
        }
        let dto = parse_controller_config_dto(bytes)?;
        parse_controller_config_v3(dto).map(Self)
    }

    const fn contract(&self) -> &ControllerServerConfigV1 {
        &self.0
    }

    pub fn serial_device(&self) -> &Path {
        self.0.serial_device()
    }

    pub const fn controller_uid(&self) -> ControllerUid {
        self.0.controller_uid()
    }

    pub const fn firmware_abi(&self) -> NonZeroU16 {
        self.0.firmware_abi()
    }

    pub const fn controller_session_class(&self) -> ControllerSessionClass {
        self.0.controller_session_class()
    }

    pub const fn firmware_build_id(&self) -> NonZeroU32 {
        self.0.firmware_build_id()
    }

    pub const fn actuator_config_fingerprint(&self) -> ActuatorConfigFingerprint {
        self.0.actuator_config_fingerprint()
    }

    pub fn hardware_profile_claim_id(&self) -> &str {
        self.0.hardware_profile_claim_id()
    }

    pub const fn expected_max_abs_pwm_percent(&self) -> MaxAbsPwmPercent {
        self.0.expected_max_abs_pwm_percent()
    }

    pub const fn expected_physical_stop_semantics(&self) -> PhysicalStopSemantics {
        self.0.expected_physical_stop_semantics()
    }

    pub const fn serial_applied_ack_timeout(&self) -> Duration {
        self.0.serial_applied_ack_timeout()
    }

    pub const fn maximum_heartbeat_age(&self) -> Duration {
        self.0.maximum_heartbeat_age()
    }

    pub const fn minimum_host_command_interval(&self) -> Duration {
        self.0.minimum_host_command_interval()
    }

    pub fn coordinated_shutdown_budget(&self) -> Duration {
        self.0.coordinated_shutdown_budget()
    }

    /// Exact percentage-point delta accepted between consecutive attended
    /// commissioning commands. This is not a physical acceleration claim.
    pub const fn maximum_command_step_percent(&self) -> u8 {
        ATTENDED_WHEEL_ON_COMMISSIONING_MAX_COMMAND_STEP_PERCENT
    }
}

impl ControllerServerConfig {
    /// Parse exactly one versioned controller document. Schema V1 admits only
    /// production external-interlock sessions; schema V2 admits only the
    /// operator-supervised four-PWM candidate; schema V3 admits only the
    /// separately identified attended wheel-on commissioning image.
    pub fn parse_json(bytes: &[u8]) -> Result<Self, ServerConfigError> {
        if bytes.len() > MAX_CONTROLLER_SERVER_CONFIG_JSON_BYTES {
            return Err(ServerConfigError::InputBytesTooLarge {
                actual_bytes: bytes.len(),
                maximum_bytes: MAX_CONTROLLER_SERVER_CONFIG_JSON_BYTES,
            });
        }
        let dto = parse_controller_config_dto(bytes)?;
        match dto.schema_version {
            CONTROLLER_SERVER_CONFIG_V1 => parse_controller_config_v1(dto).map(Self::ProductionV1),
            CONTROLLER_SERVER_CONFIG_V2 => parse_controller_config_v2(dto)
                .map(ControllerServerConfigV2)
                .map(Self::OperatorSupervisedFourPwmCandidateV2),
            CONTROLLER_SERVER_CONFIG_V3 => parse_controller_config_v3(dto)
                .map(ControllerServerConfigV3)
                .map(Self::AttendedWheelOnCommissioningV3),
            version => Err(ServerConfigError::UnsupportedSchemaVersion(version)),
        }
    }

    const fn contract(&self) -> &ControllerServerConfigV1 {
        match self {
            Self::ProductionV1(config) => config,
            Self::OperatorSupervisedFourPwmCandidateV2(config) => config.contract(),
            Self::AttendedWheelOnCommissioningV3(config) => config.contract(),
        }
    }

    pub fn serial_device(&self) -> &Path {
        self.contract().serial_device()
    }

    pub const fn controller_uid(&self) -> ControllerUid {
        self.contract().controller_uid()
    }

    pub const fn firmware_abi(&self) -> NonZeroU16 {
        self.contract().firmware_abi()
    }

    pub const fn firmware_build_id(&self) -> NonZeroU32 {
        self.contract().firmware_build_id()
    }

    pub const fn actuator_config_fingerprint(&self) -> ActuatorConfigFingerprint {
        self.contract().actuator_config_fingerprint()
    }

    pub fn hardware_profile_claim_id(&self) -> &str {
        self.contract().hardware_profile_claim_id()
    }

    pub const fn controller_ready_timeout(&self) -> Duration {
        self.contract().controller_ready_timeout()
    }

    pub const fn heartbeat_period(&self) -> Duration {
        self.contract().heartbeat_period()
    }

    pub const fn maximum_heartbeat_age(&self) -> Duration {
        self.contract().maximum_heartbeat_age()
    }

    pub const fn maximum_host_command_rate_hz(&self) -> NonZeroU16 {
        self.contract().maximum_host_command_rate_hz()
    }

    pub const fn minimum_host_command_interval(&self) -> Duration {
        self.contract().minimum_host_command_interval()
    }

    pub const fn serial_transmit_timeout(&self) -> Duration {
        self.contract().serial_transmit_timeout()
    }

    pub const fn serial_applied_ack_timeout(&self) -> Duration {
        self.contract().serial_applied_ack_timeout()
    }

    pub const fn controller_clock_abs_error_ppm_bound(&self) -> NonZeroU32 {
        self.contract().controller_clock_abs_error_ppm_bound()
    }

    pub const fn deadline_quantization_margin_ms(&self) -> NonZeroU16 {
        self.contract().deadline_quantization_margin_ms()
    }

    pub const fn expected_max_abs_pwm_percent(&self) -> MaxAbsPwmPercent {
        self.contract().expected_max_abs_pwm_percent()
    }

    pub const fn expected_pwm_frequency(&self) -> PwmFrequencyHz {
        self.contract().expected_pwm_frequency()
    }

    pub const fn expected_watchdog_nominal_period(&self) -> WatchdogNominalPeriodMs {
        self.contract().expected_watchdog_nominal_period()
    }

    pub const fn expected_neutral_output(&self) -> NeutralOutput {
        self.contract().expected_neutral_output()
    }

    pub const fn expected_physical_stop_semantics(&self) -> PhysicalStopSemantics {
        self.contract().expected_physical_stop_semantics()
    }

    pub const fn controller_session_class(&self) -> ControllerSessionClass {
        self.contract().controller_session_class()
    }

    /// Firmware command-delta invariant when the selected controller profile
    /// declares one. Production V1 has no equivalent schema claim.
    pub const fn maximum_command_step_percent(&self) -> Option<u8> {
        match self {
            Self::ProductionV1(_) => None,
            Self::OperatorSupervisedFourPwmCandidateV2(config) => {
                Some(config.maximum_command_step_percent())
            }
            Self::AttendedWheelOnCommissioningV3(config) => {
                Some(config.maximum_command_step_percent())
            }
        }
    }

    pub fn coordinated_shutdown_budget(&self) -> Duration {
        self.contract().coordinated_shutdown_budget()
    }
}

impl From<ControllerServerConfigV1> for ControllerServerConfig {
    fn from(value: ControllerServerConfigV1) -> Self {
        Self::ProductionV1(value)
    }
}

impl From<ControllerServerConfigV2> for ControllerServerConfig {
    fn from(value: ControllerServerConfigV2) -> Self {
        Self::OperatorSupervisedFourPwmCandidateV2(value)
    }
}

impl From<ControllerServerConfigV3> for ControllerServerConfig {
    fn from(value: ControllerServerConfigV3) -> Self {
        Self::AttendedWheelOnCommissioningV3(value)
    }
}

#[derive(Debug)]
pub enum ServerConfigError {
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
    SessionClassForbiddenInSchemaV1,
    SessionClassMissingInSchemaV2,
    SessionClassMissingInSchemaV3,
    UnsupportedSessionClass,
    UnsupportedCommissioningSessionClass,
    CandidateFirmwareAbiMismatch {
        actual: u16,
        required: u16,
    },
    CandidateFirmwareBuildMismatch {
        actual: u32,
        required: u32,
    },
    CandidateFingerprintMismatch,
    CandidateIdentityRequiresSchemaV2,
    CommissioningFirmwareAbiMismatch {
        actual: u16,
        required: u16,
    },
    CommissioningFirmwareBuildMismatch {
        actual: u32,
        required: u32,
    },
    CommissioningFingerprintMismatch,
    CommissioningIdentityRequiresSchemaV3,
    CommissioningRequiresUnverifiedStop,
    CommissioningPwmCapMismatch {
        actual: u8,
        required: u8,
    },
    CandidateRequiresUnverifiedStop,
    CandidatePwmCapOutOfRange {
        actual: u8,
        minimum: u8,
        maximum: u8,
    },
    HeartbeatAgeTooShort {
        heartbeat_period_ms: u16,
        maximum_heartbeat_age_ms: u16,
    },
    SerialAckTimeoutNotBelowHeartbeatAge {
        serial_ack_timeout_ms: u16,
        maximum_heartbeat_age_ms: u16,
    },
    SerialTransmitTimeoutBelowWireMinimum {
        actual_ms: u16,
        minimum_ms: u16,
        maximum_uart_record_bytes: u32,
        baud_bits_per_second: u32,
    },
    SerialTransmitTimeoutNotBelowAckTimeout {
        serial_transmit_timeout_ms: u16,
        serial_ack_timeout_ms: u16,
    },
    SerialBandwidthExceeded {
        direction: &'static str,
        required_bits_per_second: u64,
        available_bits_per_second: u64,
    },
}

impl fmt::Display for ServerConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputBytesTooLarge {
                actual_bytes,
                maximum_bytes,
            } => write!(
                formatter,
                "controller config input is {actual_bytes} bytes; maximum is {maximum_bytes}"
            ),
            Self::Json(source) => write!(formatter, "invalid controller config JSON: {source}"),
            Self::UnsupportedSchemaVersion(version) => {
                write!(
                    formatter,
                    "unsupported controller server config schema {version}"
                )
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
                write!(
                    formatter,
                    "{field} has a non-hex digit at index {digit_index}"
                )
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
            Self::InvalidHardwareProfileClaimId => {
                formatter.write_str("hardware_profile_claim_id must be bounded canonical ASCII")
            }
            Self::UnverifiedPhysicalStopSemantics => formatter.write_str(
                "expected_physical_stop_semantics must be coast_verified or brake_verified",
            ),
            Self::SessionClassForbiddenInSchemaV1 => formatter.write_str(
                "schema V1 is production-only and must not contain controller_session_class",
            ),
            Self::SessionClassMissingInSchemaV2 => {
                formatter.write_str("schema V2 requires controller_session_class")
            }
            Self::SessionClassMissingInSchemaV3 => {
                formatter.write_str("schema V3 requires controller_session_class")
            }
            Self::UnsupportedSessionClass => formatter.write_str(
                "schema V2 controller_session_class must be operator_supervised_four_pwm_candidate",
            ),
            Self::UnsupportedCommissioningSessionClass => formatter.write_str(
                "schema V3 controller_session_class must be attended_wheel_on_commissioning",
            ),
            Self::CandidateFirmwareAbiMismatch { actual, required } => write!(
                formatter,
                "candidate firmware ABI {actual} does not equal required KRP2 ABI {required}"
            ),
            Self::CandidateFirmwareBuildMismatch { actual, required } => write!(
                formatter,
                "candidate firmware build {actual:#010x} does not equal {required:#010x}"
            ),
            Self::CandidateFingerprintMismatch => formatter
                .write_str("candidate actuator fingerprint does not equal KIKO-4PWM-CAND1!"),
            Self::CandidateIdentityRequiresSchemaV2 => formatter.write_str(
                "the reserved four-PWM candidate identity requires schema V2 and cannot be admitted by the production schema",
            ),
            Self::CommissioningFirmwareAbiMismatch { actual, required } => write!(
                formatter,
                "commissioning firmware ABI {actual} does not equal required KRP2 ABI {required}"
            ),
            Self::CommissioningFirmwareBuildMismatch { actual, required } => write!(
                formatter,
                "commissioning firmware build {actual:#010x} does not equal {required:#010x}"
            ),
            Self::CommissioningFingerprintMismatch => formatter.write_str(
                "commissioning actuator fingerprint does not equal KIKO-WHEELON-CM1",
            ),
            Self::CommissioningIdentityRequiresSchemaV3 => formatter.write_str(
                "the reserved attended wheel-on commissioning identity requires schema V3 and cannot be admitted by the production schema",
            ),
            Self::CommissioningRequiresUnverifiedStop => formatter.write_str(
                "attended wheel-on commissioning physical stop semantics must remain unverified",
            ),
            Self::CommissioningPwmCapMismatch { actual, required } => write!(
                formatter,
                "attended wheel-on commissioning PWM cap {actual}% must equal the fixed {required}% ceiling"
            ),
            Self::CandidateRequiresUnverifiedStop => {
                formatter.write_str("candidate physical stop semantics must remain unverified")
            }
            Self::CandidatePwmCapOutOfRange {
                actual,
                minimum,
                maximum,
            } => write!(
                formatter,
                "candidate PWM cap {actual}% is outside the operator-supervised range {minimum}%..={maximum}%"
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
            Self::SerialTransmitTimeoutBelowWireMinimum {
                actual_ms,
                minimum_ms,
                maximum_uart_record_bytes,
                baud_bits_per_second,
            } => write!(
                formatter,
                "serial transmit timeout {actual_ms} ms is below the {minimum_ms} ms ceiling required to put one {maximum_uart_record_bytes}-byte 8N1 UART record on a {baud_bits_per_second} bit/s link"
            ),
            Self::SerialTransmitTimeoutNotBelowAckTimeout {
                serial_transmit_timeout_ms,
                serial_ack_timeout_ms,
            } => write!(
                formatter,
                "serial transmit timeout {serial_transmit_timeout_ms} ms must be below serial applied-ACK timeout {serial_ack_timeout_ms} ms"
            ),
            Self::SerialBandwidthExceeded {
                direction,
                required_bits_per_second,
                available_bits_per_second,
            } => write!(
                formatter,
                "declared {direction} UART load requires {required_bits_per_second} bit/s but the exact 8N1 link provides only {available_bits_per_second} bit/s"
            ),
        }
    }
}

impl std::error::Error for ServerConfigError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
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
    controller_ready_timeout_ms: u16,
    heartbeat_period_ms: u16,
    maximum_heartbeat_age_ms: u16,
    maximum_host_command_rate_hz: u16,
    serial_transmit_timeout_ms: u16,
    serial_applied_ack_timeout_ms: u16,
    controller_clock_abs_error_ppm_bound: u32,
    deadline_quantization_margin_ms: u16,
    expected_max_abs_pwm_percent: u8,
    expected_pwm_frequency_hz: u32,
    expected_watchdog_nominal_timeout_ms: u16,
    expected_neutral_output: NeutralOutputDto,
    expected_physical_stop_semantics: PhysicalStopSemanticsDto,
    #[serde(default, deserialize_with = "deserialize_session_class_presence")]
    controller_session_class: ControllerSessionClassPresence,
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

#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ControllerSessionClassDto {
    OperatorSupervisedFourPwmCandidate,
    AttendedWheelOnCommissioning,
    ProductionExternalInterlocks,
}

#[derive(Clone, Copy, Default)]
enum ControllerSessionClassPresence {
    #[default]
    Missing,
    Present(ControllerSessionClassDto),
}

fn deserialize_session_class_presence<'de, DeserializerType>(
    deserializer: DeserializerType,
) -> Result<ControllerSessionClassPresence, DeserializerType::Error>
where
    DeserializerType: Deserializer<'de>,
{
    ControllerSessionClassDto::deserialize(deserializer)
        .map(ControllerSessionClassPresence::Present)
}

fn parse_controller_config_dto(
    bytes: &[u8],
) -> Result<ControllerServerConfigV1Dto, ServerConfigError> {
    serde_json::from_slice(bytes).map_err(ServerConfigError::Json)
}

fn parse_controller_config_v1(
    dto: ControllerServerConfigV1Dto,
) -> Result<ControllerServerConfigV1, ServerConfigError> {
    if dto.schema_version != CONTROLLER_SERVER_CONFIG_V1 {
        return Err(ServerConfigError::UnsupportedSchemaVersion(
            dto.schema_version,
        ));
    }
    if !matches!(
        dto.controller_session_class,
        ControllerSessionClassPresence::Missing
    ) {
        return Err(ServerConfigError::SessionClassForbiddenInSchemaV1);
    }
    let config =
        parse_controller_config_fields(dto, ControllerSessionClass::ProductionExternalInterlocks)?;
    if config.firmware_build_id().get() == OPERATOR_SUPERVISED_FOUR_PWM_FIRMWARE_BUILD_ID
        || config.actuator_config_fingerprint().as_bytes()
            == &OPERATOR_SUPERVISED_FOUR_PWM_FINGERPRINT_BYTES
    {
        return Err(ServerConfigError::CandidateIdentityRequiresSchemaV2);
    }
    if config.firmware_build_id().get() == ATTENDED_WHEEL_ON_COMMISSIONING_FIRMWARE_BUILD_ID
        || config.actuator_config_fingerprint().as_bytes()
            == &ATTENDED_WHEEL_ON_COMMISSIONING_FINGERPRINT_BYTES
    {
        return Err(ServerConfigError::CommissioningIdentityRequiresSchemaV3);
    }
    Ok(config)
}

fn parse_controller_config_v2(
    dto: ControllerServerConfigV1Dto,
) -> Result<ControllerServerConfigV1, ServerConfigError> {
    if dto.schema_version != CONTROLLER_SERVER_CONFIG_V2 {
        return Err(ServerConfigError::UnsupportedSchemaVersion(
            dto.schema_version,
        ));
    }
    match dto.controller_session_class {
        ControllerSessionClassPresence::Missing => {
            return Err(ServerConfigError::SessionClassMissingInSchemaV2);
        }
        ControllerSessionClassPresence::Present(
            ControllerSessionClassDto::ProductionExternalInterlocks,
        )
        | ControllerSessionClassPresence::Present(
            ControllerSessionClassDto::AttendedWheelOnCommissioning,
        ) => {
            return Err(ServerConfigError::UnsupportedSessionClass);
        }
        ControllerSessionClassPresence::Present(
            ControllerSessionClassDto::OperatorSupervisedFourPwmCandidate,
        ) => {}
    }
    let config = parse_controller_config_fields(
        dto,
        ControllerSessionClass::OperatorSupervisedFourPwmCandidate,
    )?;
    let required_abi = u16::from(ROBOT_PROTOCOL_V2);
    if config.firmware_abi().get() != required_abi {
        return Err(ServerConfigError::CandidateFirmwareAbiMismatch {
            actual: config.firmware_abi().get(),
            required: required_abi,
        });
    }
    if config.firmware_build_id().get() != OPERATOR_SUPERVISED_FOUR_PWM_FIRMWARE_BUILD_ID {
        return Err(ServerConfigError::CandidateFirmwareBuildMismatch {
            actual: config.firmware_build_id().get(),
            required: OPERATOR_SUPERVISED_FOUR_PWM_FIRMWARE_BUILD_ID,
        });
    }
    if config.actuator_config_fingerprint().as_bytes()
        != &OPERATOR_SUPERVISED_FOUR_PWM_FINGERPRINT_BYTES
    {
        return Err(ServerConfigError::CandidateFingerprintMismatch);
    }
    if config.expected_physical_stop_semantics() != PhysicalStopSemantics::Unverified {
        return Err(ServerConfigError::CandidateRequiresUnverifiedStop);
    }
    let cap = config.expected_max_abs_pwm_percent().get();
    if cap != MAX_OPERATOR_SUPERVISED_FOUR_PWM_PWM_PERCENT {
        return Err(ServerConfigError::CandidatePwmCapOutOfRange {
            actual: cap,
            minimum: MAX_OPERATOR_SUPERVISED_FOUR_PWM_PWM_PERCENT,
            maximum: MAX_OPERATOR_SUPERVISED_FOUR_PWM_PWM_PERCENT,
        });
    }
    Ok(config)
}

fn parse_controller_config_v3(
    dto: ControllerServerConfigV1Dto,
) -> Result<ControllerServerConfigV1, ServerConfigError> {
    if dto.schema_version != CONTROLLER_SERVER_CONFIG_V3 {
        return Err(ServerConfigError::UnsupportedSchemaVersion(
            dto.schema_version,
        ));
    }
    match dto.controller_session_class {
        ControllerSessionClassPresence::Missing => {
            return Err(ServerConfigError::SessionClassMissingInSchemaV3);
        }
        ControllerSessionClassPresence::Present(
            ControllerSessionClassDto::ProductionExternalInterlocks,
        )
        | ControllerSessionClassPresence::Present(
            ControllerSessionClassDto::OperatorSupervisedFourPwmCandidate,
        ) => {
            return Err(ServerConfigError::UnsupportedCommissioningSessionClass);
        }
        ControllerSessionClassPresence::Present(
            ControllerSessionClassDto::AttendedWheelOnCommissioning,
        ) => {}
    }
    let config =
        parse_controller_config_fields(dto, ControllerSessionClass::AttendedWheelOnCommissioning)?;
    let required_abi = u16::from(ROBOT_PROTOCOL_V2);
    if config.firmware_abi().get() != required_abi {
        return Err(ServerConfigError::CommissioningFirmwareAbiMismatch {
            actual: config.firmware_abi().get(),
            required: required_abi,
        });
    }
    if config.firmware_build_id().get() != ATTENDED_WHEEL_ON_COMMISSIONING_FIRMWARE_BUILD_ID {
        return Err(ServerConfigError::CommissioningFirmwareBuildMismatch {
            actual: config.firmware_build_id().get(),
            required: ATTENDED_WHEEL_ON_COMMISSIONING_FIRMWARE_BUILD_ID,
        });
    }
    if config.actuator_config_fingerprint().as_bytes()
        != &ATTENDED_WHEEL_ON_COMMISSIONING_FINGERPRINT_BYTES
    {
        return Err(ServerConfigError::CommissioningFingerprintMismatch);
    }
    if config.expected_physical_stop_semantics() != PhysicalStopSemantics::Unverified {
        return Err(ServerConfigError::CommissioningRequiresUnverifiedStop);
    }
    let cap = config.expected_max_abs_pwm_percent().get();
    if cap != MAX_ATTENDED_WHEEL_ON_COMMISSIONING_PWM_PERCENT {
        return Err(ServerConfigError::CommissioningPwmCapMismatch {
            actual: cap,
            required: MAX_ATTENDED_WHEEL_ON_COMMISSIONING_PWM_PERCENT,
        });
    }
    Ok(config)
}

fn parse_controller_config_fields(
    dto: ControllerServerConfigV1Dto,
    controller_session_class: ControllerSessionClass,
) -> Result<ControllerServerConfigV1, ServerConfigError> {
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
    let controller_ready_timeout_ms = bounded_nonzero_u16(
        "controller_ready_timeout_ms",
        dto.controller_ready_timeout_ms,
        30_000,
    )?;
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
    let maximum_host_command_rate_hz = bounded_nonzero_u16(
        "maximum_host_command_rate_hz",
        dto.maximum_host_command_rate_hz,
        MAX_HOST_COMMAND_RATE_HZ,
    )?;
    let serial_transmit_timeout_ms = bounded_nonzero_u16(
        "serial_transmit_timeout_ms",
        dto.serial_transmit_timeout_ms,
        100,
    )?;
    let minimum_serial_transmit_timeout_ms = minimum_serial_transmit_timeout_ms();
    if serial_transmit_timeout_ms.get() < minimum_serial_transmit_timeout_ms {
        return Err(ServerConfigError::SerialTransmitTimeoutBelowWireMinimum {
            actual_ms: serial_transmit_timeout_ms.get(),
            minimum_ms: minimum_serial_transmit_timeout_ms,
            maximum_uart_record_bytes: u32::try_from(PROTOCOL_MAX_UART_RECORD_BYTES)
                .expect("protocol UART record bound fits u32"),
            baud_bits_per_second: CONTROLLER_SERIAL_BAUD_BPS,
        });
    }
    if serial_transmit_timeout_ms >= serial_applied_ack_timeout_ms {
        return Err(ServerConfigError::SerialTransmitTimeoutNotBelowAckTimeout {
            serial_transmit_timeout_ms: serial_transmit_timeout_ms.get(),
            serial_ack_timeout_ms: serial_applied_ack_timeout_ms.get(),
        });
    }
    admit_serial_bandwidth(heartbeat_period_ms, maximum_host_command_rate_hz)?;
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
        PhysicalStopSemanticsDto::Unverified
            if controller_session_class == ControllerSessionClass::ProductionExternalInterlocks =>
        {
            return Err(ServerConfigError::UnverifiedPhysicalStopSemantics);
        }
        PhysicalStopSemanticsDto::Unverified => PhysicalStopSemantics::Unverified,
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
        controller_ready_timeout: Duration::from_millis(u64::from(
            controller_ready_timeout_ms.get(),
        )),
        heartbeat_period: Duration::from_millis(u64::from(heartbeat_period_ms.get())),
        maximum_heartbeat_age: Duration::from_millis(u64::from(maximum_heartbeat_age_ms.get())),
        maximum_host_command_rate_hz,
        minimum_host_command_interval: minimum_command_interval(maximum_host_command_rate_hz),
        serial_transmit_timeout: Duration::from_millis(u64::from(serial_transmit_timeout_ms.get())),
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
        controller_session_class,
    })
}

fn maximum_uart_record_bytes(payload_bytes: usize) -> u64 {
    u64::try_from(payload_bytes + MAX_UART_RECORD_OVERHEAD_BYTES)
        .expect("bounded protocol record size fits u64")
}

const fn ceiling_div_u64(numerator: u64, denominator: u64) -> u64 {
    numerator.saturating_add(denominator.saturating_sub(1)) / denominator
}

fn minimum_serial_transmit_timeout_ms() -> u16 {
    let maximum_record_bytes =
        u64::try_from(PROTOCOL_MAX_UART_RECORD_BYTES).expect("protocol UART record bound fits u64");
    let wire_bits = maximum_record_bytes.saturating_mul(u64::from(UART_8N1_BITS_PER_BYTE));
    let milliseconds = ceiling_div_u64(
        wire_bits.saturating_mul(1_000),
        u64::from(CONTROLLER_SERIAL_BAUD_BPS),
    );
    u16::try_from(milliseconds).expect("one bounded UART record wire time fits u16 milliseconds")
}

fn minimum_command_interval(rate: NonZeroU16) -> Duration {
    Duration::from_nanos(ceiling_div_u64(1_000_000_000, u64::from(rate.get())))
}

fn admit_serial_bandwidth(
    heartbeat_period_ms: NonZeroU16,
    maximum_host_command_rate_hz: NonZeroU16,
) -> Result<(), ServerConfigError> {
    let heartbeat_rate_hz = ceiling_div_u64(1_000, u64::from(heartbeat_period_ms.get()));
    let controller_tx_bytes_per_second = u64::from(maximum_host_command_rate_hz.get())
        .saturating_mul(maximum_uart_record_bytes(AppliedResult::PAYLOAD_BYTES))
        .saturating_add(
            heartbeat_rate_hz.saturating_mul(maximum_uart_record_bytes(Heartbeat::PAYLOAD_BYTES)),
        )
        .saturating_add(
            ceiling_div_u64(1_000, u64::from(CANONICAL_ODOMETRY_REPORT_PERIOD_MS)).saturating_mul(
                maximum_uart_record_bytes(ObservationalOdometry::PAYLOAD_BYTES),
            ),
        )
        .saturating_add(
            ceiling_div_u64(1_000, u64::from(CANONICAL_CONTROLLER_HELLO_PERIOD_MS))
                .saturating_mul(maximum_uart_record_bytes(ControllerHello::PAYLOAD_BYTES)),
        );
    let host_tx_bytes_per_second = u64::from(maximum_host_command_rate_hz.get())
        .saturating_mul(maximum_uart_record_bytes(ApplyPwm::PAYLOAD_BYTES));
    let available_bits_per_second = u64::from(CONTROLLER_SERIAL_BAUD_BPS);
    for (direction, bytes_per_second) in [
        ("controller-to-host", controller_tx_bytes_per_second),
        ("host-to-controller", host_tx_bytes_per_second),
    ] {
        let required_bits_per_second =
            bytes_per_second.saturating_mul(u64::from(UART_8N1_BITS_PER_BYTE));
        if required_bits_per_second > available_bits_per_second {
            return Err(ServerConfigError::SerialBandwidthExceeded {
                direction,
                required_bits_per_second,
                available_bits_per_second,
            });
        }
    }
    Ok(())
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
            "controller_ready_timeout_ms": 3000,
            "heartbeat_period_ms": 20,
            "maximum_heartbeat_age_ms": 60,
            "maximum_host_command_rate_hz": 100,
            "serial_transmit_timeout_ms": 10,
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

    fn valid_candidate() -> Value {
        json!({
            "schema_version": 2,
            "serial_device": "/dev/serial/by-id/usb-STMicroelectronics_STM32_STLink_candidate-if02",
            "controller_uid_hex": "00112233445566778899aabb",
            "firmware_abi": 2,
            "firmware_build_id": 135169,
            "actuator_config_fingerprint_hex": "4b494b4f2d3450574d2d43414e443121",
            "hardware_profile_claim_id": "kiko-four-pwm-candidate-wheels-off-v1",
            "controller_ready_timeout_ms": 3000,
            "heartbeat_period_ms": 20,
            "maximum_heartbeat_age_ms": 60,
            "maximum_host_command_rate_hz": 100,
            "serial_transmit_timeout_ms": 10,
            "serial_applied_ack_timeout_ms": 30,
            "controller_clock_abs_error_ppm_bound": 50_000,
            "deadline_quantization_margin_ms": 2,
            "expected_max_abs_pwm_percent": 30,
            "expected_pwm_frequency_hz": 20_000,
            "expected_watchdog_nominal_timeout_ms": 250,
            "expected_neutral_output": "both_low",
            "expected_physical_stop_semantics": "unverified",
            "controller_session_class": "operator_supervised_four_pwm_candidate"
        })
    }

    fn valid_commissioning() -> Value {
        json!({
            "schema_version": 3,
            "serial_device": "/dev/serial/by-id/usb-STMicroelectronics_STM32_STLink_commissioning-if02",
            "controller_uid_hex": "00112233445566778899aabb",
            "firmware_abi": 2,
            "firmware_build_id": ATTENDED_WHEEL_ON_COMMISSIONING_FIRMWARE_BUILD_ID,
            "actuator_config_fingerprint_hex": "4b494b4f2d574845454c4f4e2d434d31",
            "hardware_profile_claim_id": "kiko-attended-wheel-on-commissioning-v1",
            "controller_ready_timeout_ms": 3000,
            "heartbeat_period_ms": 20,
            "maximum_heartbeat_age_ms": 60,
            "maximum_host_command_rate_hz": 100,
            "serial_transmit_timeout_ms": 10,
            "serial_applied_ack_timeout_ms": 30,
            "controller_clock_abs_error_ppm_bound": 50_000,
            "deadline_quantization_margin_ms": 2,
            "expected_max_abs_pwm_percent": 20,
            "expected_pwm_frequency_hz": 20_000,
            "expected_watchdog_nominal_timeout_ms": 250,
            "expected_neutral_output": "both_low",
            "expected_physical_stop_semantics": "unverified",
            "controller_session_class": "attended_wheel_on_commissioning"
        })
    }

    fn parse(value: &Value) -> Result<ControllerServerConfigV1, ServerConfigError> {
        ControllerServerConfigV1::parse_json(&serde_json::to_vec(value).expect("serialize fixture"))
    }

    #[test]
    fn exact_external_controller_authority_parses_once() {
        let config = parse(&valid()).expect("valid controller config");
        assert_eq!(config.serial_device(), Path::new("/dev/ttyACM0"));
        assert_eq!(config.controller_ready_timeout(), Duration::from_secs(3));
        assert_eq!(config.heartbeat_period(), Duration::from_millis(20));
        assert_eq!(config.maximum_host_command_rate_hz().get(), 100);
        assert_eq!(
            config.minimum_host_command_interval(),
            Duration::from_millis(10)
        );
        assert_eq!(config.serial_transmit_timeout(), Duration::from_millis(10));
        assert_eq!(
            config.coordinated_shutdown_budget(),
            Duration::from_millis(120)
        );
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
    fn schema_v2_admits_only_the_exact_operator_supervised_candidate_contract() {
        let bytes = serde_json::to_vec(&valid_candidate()).expect("candidate fixture");
        let config = ControllerServerConfigV2::parse_json(&bytes).expect("exact candidate");
        assert_eq!(
            config.controller_session_class(),
            ControllerSessionClass::OperatorSupervisedFourPwmCandidate
        );
        assert_eq!(
            config.firmware_build_id().get(),
            OPERATOR_SUPERVISED_FOUR_PWM_FIRMWARE_BUILD_ID
        );
        assert_eq!(
            config.actuator_config_fingerprint().as_bytes(),
            &OPERATOR_SUPERVISED_FOUR_PWM_FINGERPRINT_BYTES
        );
        assert_eq!(
            config.expected_physical_stop_semantics(),
            PhysicalStopSemantics::Unverified
        );
        assert!(matches!(
            ControllerServerConfig::parse_json(&bytes),
            Ok(ControllerServerConfig::OperatorSupervisedFourPwmCandidateV2(_))
        ));
    }

    #[test]
    fn schema_v3_admits_only_the_exact_attended_wheel_on_contract() {
        let bytes = serde_json::to_vec(&valid_commissioning()).expect("commissioning fixture");
        let config =
            ControllerServerConfigV3::parse_json(&bytes).expect("exact commissioning contract");
        assert_eq!(
            config.controller_session_class(),
            ControllerSessionClass::AttendedWheelOnCommissioning
        );
        assert_eq!(
            config.firmware_build_id().get(),
            ATTENDED_WHEEL_ON_COMMISSIONING_FIRMWARE_BUILD_ID
        );
        assert_eq!(
            config.actuator_config_fingerprint().as_bytes(),
            &ATTENDED_WHEEL_ON_COMMISSIONING_FINGERPRINT_BYTES
        );
        assert_eq!(
            config.expected_max_abs_pwm_percent().get(),
            MAX_ATTENDED_WHEEL_ON_COMMISSIONING_PWM_PERCENT
        );
        assert_eq!(
            config.expected_physical_stop_semantics(),
            PhysicalStopSemantics::Unverified
        );
        assert!(matches!(
            ControllerServerConfig::parse_json(&bytes),
            Ok(ControllerServerConfig::AttendedWheelOnCommissioningV3(_))
        ));
    }

    #[test]
    fn schema_versions_and_controller_classes_cannot_masquerade_as_each_other() {
        let mut v1_with_candidate_class = valid();
        v1_with_candidate_class["controller_session_class"] =
            json!("operator_supervised_four_pwm_candidate");
        assert!(matches!(
            parse(&v1_with_candidate_class),
            Err(ServerConfigError::SessionClassForbiddenInSchemaV1)
        ));

        let mut v1_with_null_class = valid();
        v1_with_null_class["controller_session_class"] = Value::Null;
        assert!(matches!(
            parse(&v1_with_null_class),
            Err(ServerConfigError::Json(_))
        ));
        for (field, value) in [
            (
                "firmware_build_id",
                json!(OPERATOR_SUPERVISED_FOUR_PWM_FIRMWARE_BUILD_ID),
            ),
            (
                "actuator_config_fingerprint_hex",
                json!("4b494b4f2d3450574d2d43414e443121"),
            ),
        ] {
            let mut reserved_candidate_identity = valid();
            reserved_candidate_identity[field] = value;
            assert!(matches!(
                parse(&reserved_candidate_identity),
                Err(ServerConfigError::CandidateIdentityRequiresSchemaV2)
            ));
        }

        for (field, value) in [
            (
                "firmware_build_id",
                json!(ATTENDED_WHEEL_ON_COMMISSIONING_FIRMWARE_BUILD_ID),
            ),
            (
                "actuator_config_fingerprint_hex",
                json!("4b494b4f2d574845454c4f4e2d434d31"),
            ),
        ] {
            let mut reserved_commissioning_identity = valid();
            reserved_commissioning_identity[field] = value;
            assert!(matches!(
                parse(&reserved_commissioning_identity),
                Err(ServerConfigError::CommissioningIdentityRequiresSchemaV3)
            ));
        }

        let mut missing_class = valid_candidate();
        missing_class
            .as_object_mut()
            .expect("object")
            .remove("controller_session_class");
        let parse_v2 = |value: &Value| {
            ControllerServerConfigV2::parse_json(
                &serde_json::to_vec(value).expect("serialize candidate fixture"),
            )
        };
        assert!(matches!(
            parse_v2(&missing_class),
            Err(ServerConfigError::SessionClassMissingInSchemaV2)
        ));

        let mut production_class = valid_candidate();
        production_class["controller_session_class"] = json!("production_external_interlocks");
        assert!(matches!(
            parse_v2(&production_class),
            Err(ServerConfigError::UnsupportedSessionClass)
        ));

        for (field, value) in [
            ("firmware_abi", json!(3)),
            ("firmware_build_id", json!(135170)),
            (
                "actuator_config_fingerprint_hex",
                json!("11223344556677889900aabbccddeeff"),
            ),
            ("expected_physical_stop_semantics", json!("coast_verified")),
            ("expected_max_abs_pwm_percent", json!(0)),
            ("expected_max_abs_pwm_percent", json!(29)),
            ("expected_max_abs_pwm_percent", json!(31)),
        ] {
            let mut altered = valid_candidate();
            altered[field] = value;
            assert!(
                parse_v2(&altered).is_err(),
                "candidate field {field} must be exact"
            );
        }

        let parse_v3 = |value: &Value| {
            ControllerServerConfigV3::parse_json(
                &serde_json::to_vec(value).expect("serialize commissioning fixture"),
            )
        };
        let mut missing_commissioning_class = valid_commissioning();
        missing_commissioning_class
            .as_object_mut()
            .expect("object")
            .remove("controller_session_class");
        assert!(matches!(
            parse_v3(&missing_commissioning_class),
            Err(ServerConfigError::SessionClassMissingInSchemaV3)
        ));
        for class in [
            "production_external_interlocks",
            "operator_supervised_four_pwm_candidate",
        ] {
            let mut wrong_class = valid_commissioning();
            wrong_class["controller_session_class"] = json!(class);
            assert!(matches!(
                parse_v3(&wrong_class),
                Err(ServerConfigError::UnsupportedCommissioningSessionClass)
            ));
        }
        for (field, value) in [
            ("firmware_abi", json!(3)),
            ("firmware_build_id", json!(135169)),
            (
                "actuator_config_fingerprint_hex",
                json!("4b494b4f2d3450574d2d43414e443121"),
            ),
            ("expected_physical_stop_semantics", json!("coast_verified")),
            ("expected_max_abs_pwm_percent", json!(19)),
            ("expected_max_abs_pwm_percent", json!(21)),
        ] {
            let mut altered = valid_commissioning();
            altered[field] = value;
            assert!(
                parse_v3(&altered).is_err(),
                "commissioning field {field} must be exact"
            );
        }
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

        let mut impossible_tx_deadline = valid();
        impossible_tx_deadline["serial_transmit_timeout_ms"] = json!(6);
        assert!(matches!(
            parse(&impossible_tx_deadline),
            Err(ServerConfigError::SerialTransmitTimeoutBelowWireMinimum { .. })
        ));

        let mut tx_deadline_consumes_ack_window = valid();
        tx_deadline_consumes_ack_window["serial_transmit_timeout_ms"] = json!(30);
        assert!(matches!(
            parse(&tx_deadline_consumes_ack_window),
            Err(ServerConfigError::SerialTransmitTimeoutNotBelowAckTimeout { .. })
        ));

        let mut impossible_bandwidth = valid();
        impossible_bandwidth["heartbeat_period_ms"] = json!(1);
        assert!(matches!(
            parse(&impossible_bandwidth),
            Err(ServerConfigError::SerialBandwidthExceeded {
                direction: "controller-to-host",
                ..
            })
        ));
    }

    #[test]
    fn exact_115200_8n1_budget_admits_candidate_twenty_ms_fifty_hertz_baseline() {
        assert_eq!(
            maximum_uart_record_bytes(robot_protocol::v2::MAX_PAYLOAD_BYTES),
            u64::try_from(PROTOCOL_MAX_UART_RECORD_BYTES).expect("protocol bound fits u64")
        );
        assert_eq!(minimum_serial_transmit_timeout_ms(), 7);
        let candidate = valid_candidate();
        let bytes = serde_json::to_vec(&candidate).expect("serialize candidate baseline");
        let config = ControllerServerConfig::parse_json(&bytes)
            .expect("candidate controller contract is within its exact wire ceiling");
        assert!(matches!(
            &config,
            ControllerServerConfig::OperatorSupervisedFourPwmCandidateV2(_)
        ));
        assert_eq!(config.heartbeat_period(), Duration::from_millis(20));
        assert_eq!(config.maximum_host_command_rate_hz().get(), 100);
        assert_eq!(
            config.minimum_host_command_interval(),
            Duration::from_millis(10)
        );
        let baseline_interval = Duration::from_millis(20);
        let scheduling_margin = Duration::from_millis(5);
        assert!(
            config
                .minimum_host_command_interval()
                .checked_add(scheduling_margin)
                .is_some_and(|required| required < baseline_interval),
            "the 50 Hz runtime cadence must retain its 5 ms margin above the server ceiling"
        );

        let heartbeat_records_per_second = 50_u64;
        let expected_controller_tx_bits_per_second = (50_u64
            * maximum_uart_record_bytes(AppliedResult::PAYLOAD_BYTES)
            + heartbeat_records_per_second * maximum_uart_record_bytes(Heartbeat::PAYLOAD_BYTES)
            + 10 * maximum_uart_record_bytes(ObservationalOdometry::PAYLOAD_BYTES)
            + maximum_uart_record_bytes(ControllerHello::PAYLOAD_BYTES))
            * u64::from(UART_8N1_BITS_PER_BYTE);
        assert_eq!(expected_controller_tx_bits_per_second, 65_400);
        assert!(expected_controller_tx_bits_per_second < u64::from(CONTROLLER_SERIAL_BAUD_BPS));
    }

    #[test]
    fn exact_115200_8n1_budget_admits_hundred_hertz_parser_ceiling_as_transport_stress_only() {
        let config =
            parse(&valid()).expect("20 ms heartbeat plus the 100 Hz parser ceiling is admissible");
        assert_eq!(config.maximum_host_command_rate_hz().get(), 100);

        let heartbeat_records_per_second = 50_u64;
        let expected_controller_tx_bits_per_second = (100_u64
            * maximum_uart_record_bytes(AppliedResult::PAYLOAD_BYTES)
            + heartbeat_records_per_second * maximum_uart_record_bytes(Heartbeat::PAYLOAD_BYTES)
            + 10 * maximum_uart_record_bytes(ObservationalOdometry::PAYLOAD_BYTES)
            + maximum_uart_record_bytes(ControllerHello::PAYLOAD_BYTES))
            * u64::from(UART_8N1_BITS_PER_BYTE);
        assert_eq!(expected_controller_tx_bits_per_second, 94_400);
        assert!(expected_controller_tx_bits_per_second < u64::from(CONTROLLER_SERIAL_BAUD_BPS));
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
