use std::fmt;
use std::time::Duration;

use kiko_head_protocol::{
    FrameBuildError, GoalSpeedTicksPerSecond, HeadJoint, HeadPose, HeadTorqueLimits,
    PositionAgreementError, PositionAgreementTicks, PositionTicks, TorqueLimitPermille,
};

const MAX_DEVICE_PATH_BYTES: usize = 512;
const MAX_DEVICE_ID_BYTES: usize = 255;
const MAX_OPERATION_TIMEOUT_MS: u64 = 5_000;
const MAX_ARMING_FRESHNESS_MS: u64 = 5_000;
const MAX_WRITE_ATTEMPTS: u8 = 8;
const MAX_NOISE_BUDGET_BYTES: u16 = 1_024;
/// Maximum admitted width of one raw-encoder startup pose window.
///
/// This is a structural anti-bypass bound, not a physical joint envelope. The
/// assembly-specific minimum and maximum still require independent review.
pub const MAX_CONFIGURED_POSE_WINDOW_SPAN_TICKS: u16 = 256;

/// Weak configuration accepted at the process boundary.
///
/// Parse this exactly once with [`HeadRuntimeConfig::parse`]. Runtime code only
/// accepts the resulting domain type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeadRuntimeConfigInput {
    pub device_path: String,
    pub response_timeout_ms: u64,
    pub write_timeout_ms: u64,
    pub arming_freshness_ms: u64,
    pub write_attempts: u8,
    pub noise_budget_bytes: u16,
    pub redundant_read_tolerance_ticks: u16,
    pub readback_tolerance_ticks: u16,
    pub goal_speed_ticks_per_second: u16,
    /// Bow, curl, yaw, and roll, in that exact order.
    pub torque_limit_permille: [u16; 4],
}

/// Supported stable device-name schemes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeviceIdentityKind {
    LinuxById,
    MacOsCallout,
}

/// Exact configured device identity. No discovery or fallback path exists.
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

/// Strictly positive per-operation timeout, bounded to five seconds.
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

/// Number of permitted write attempts. Only a retryable zero-progress failure
/// can consume an attempt after the first one.
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

/// Maximum permitted age of the oldest admitted joint observation when each
/// servo is armed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ArmingFreshness(Duration);

impl ArmingFreshness {
    fn parse(milliseconds: u64) -> Result<Self, ConfigParseError> {
        if !(1..=MAX_ARMING_FRESHNESS_MS).contains(&milliseconds) {
            return Err(ConfigParseError::ArmingFreshnessOutOfRange {
                milliseconds,
                minimum_ms: 1,
                maximum_ms: MAX_ARMING_FRESHNESS_MS,
            });
        }
        Ok(Self(Duration::from_millis(milliseconds)))
    }

    pub const fn get(self) -> Duration {
        self.0
    }
}

/// Exact caller-reviewed raw-encoder windows in bow/curl/yaw/roll order.
///
/// Construction rejects descending, out-of-domain, and overly broad windows;
/// in particular, the full encoder range is unrepresentable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConfiguredHeadPoseBounds {
    minimum: [PositionTicks; 4],
    maximum: [PositionTicks; 4],
}

impl ConfiguredHeadPoseBounds {
    pub fn try_new(
        minimum_ticks: [u16; 4],
        maximum_ticks: [u16; 4],
    ) -> Result<Self, ConfiguredHeadPoseBoundsError> {
        let mut minimum = [PositionTicks::MIN; 4];
        let mut maximum = [PositionTicks::MIN; 4];
        for (index, joint) in HeadJoint::ALL.into_iter().enumerate() {
            minimum[index] = PositionTicks::try_new(minimum_ticks[index]).map_err(|source| {
                ConfiguredHeadPoseBoundsError::Position {
                    joint,
                    bound: ConfiguredHeadPoseBound::Minimum,
                    source,
                }
            })?;
            maximum[index] = PositionTicks::try_new(maximum_ticks[index]).map_err(|source| {
                ConfiguredHeadPoseBoundsError::Position {
                    joint,
                    bound: ConfiguredHeadPoseBound::Maximum,
                    source,
                }
            })?;
            if minimum[index] > maximum[index] {
                return Err(ConfiguredHeadPoseBoundsError::Descending {
                    joint,
                    minimum: minimum[index],
                    maximum: maximum[index],
                });
            }
            let span_ticks = maximum[index].get() - minimum[index].get();
            if span_ticks > MAX_CONFIGURED_POSE_WINDOW_SPAN_TICKS {
                return Err(ConfiguredHeadPoseBoundsError::SpanAboveMaximum {
                    joint,
                    minimum: minimum[index],
                    maximum: maximum[index],
                    span_ticks,
                    maximum_span_ticks: MAX_CONFIGURED_POSE_WINDOW_SPAN_TICKS,
                });
            }
        }
        Ok(Self { minimum, maximum })
    }

    pub const fn minimum(self, joint: HeadJoint) -> PositionTicks {
        self.minimum[joint as usize]
    }

    pub const fn maximum(self, joint: HeadJoint) -> PositionTicks {
        self.maximum[joint as usize]
    }

    pub fn admit(
        self,
        observed_pose: HeadPose,
    ) -> Result<HeadPoseWithinConfiguredBounds, HeadPoseBoundsAdmissionError> {
        for joint in HeadJoint::ALL {
            let observed = observed_pose.position(joint);
            if observed < self.minimum(joint) || observed > self.maximum(joint) {
                return Err(HeadPoseBoundsAdmissionError::OutsideConfiguredWindow {
                    joint,
                    observed,
                    minimum: self.minimum(joint),
                    maximum: self.maximum(joint),
                });
            }
        }
        Ok(HeadPoseWithinConfiguredBounds {
            observed_pose,
            configured_bounds: self,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConfiguredHeadPoseBound {
    Minimum,
    Maximum,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConfiguredHeadPoseBoundsError {
    Position {
        joint: HeadJoint,
        bound: ConfiguredHeadPoseBound,
        source: FrameBuildError,
    },
    Descending {
        joint: HeadJoint,
        minimum: PositionTicks,
        maximum: PositionTicks,
    },
    SpanAboveMaximum {
        joint: HeadJoint,
        minimum: PositionTicks,
        maximum: PositionTicks,
        span_ticks: u16,
        maximum_span_ticks: u16,
    },
}

impl fmt::Display for ConfiguredHeadPoseBoundsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid configured Kiko head pose bounds: {self:?}"
        )
    }
}

impl std::error::Error for ConfiguredHeadPoseBoundsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Position { source, .. } => Some(source),
            Self::Descending { .. } | Self::SpanAboveMaximum { .. } => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadPoseBoundsAdmissionError {
    OutsideConfiguredWindow {
        joint: HeadJoint,
        observed: PositionTicks,
        minimum: PositionTicks,
        maximum: PositionTicks,
    },
}

impl fmt::Display for HeadPoseBoundsAdmissionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "observed head pose was not admitted: {self:?}")
    }
}

impl std::error::Error for HeadPoseBoundsAdmissionError {}

/// Evidence constructed only when the complete observed pose is inside all
/// four caller-reviewed windows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadPoseWithinConfiguredBounds {
    observed_pose: HeadPose,
    configured_bounds: ConfiguredHeadPoseBounds,
}

impl HeadPoseWithinConfiguredBounds {
    pub const fn observed_pose(self) -> HeadPose {
        self.observed_pose
    }

    pub const fn configured_bounds(self) -> ConfiguredHeadPoseBounds {
        self.configured_bounds
    }
}

/// Fully parsed runtime configuration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeadRuntimeConfig {
    device: DeviceIdentity,
    response_timeout: OperationTimeout,
    write_timeout: OperationTimeout,
    write_attempts: WriteAttemptLimit,
    arming_freshness: ArmingFreshness,
    noise_budget_bytes: u16,
    redundant_read_tolerance: PositionAgreementTicks,
    readback_tolerance: PositionAgreementTicks,
    goal_speed: GoalSpeedTicksPerSecond,
    torque_limits: HeadTorqueLimits,
}

impl HeadRuntimeConfig {
    pub fn parse(input: HeadRuntimeConfigInput) -> Result<Self, ConfigParseError> {
        let device = DeviceIdentity::parse(input.device_path)?;
        let response_timeout =
            OperationTimeout::parse("response_timeout_ms", input.response_timeout_ms)?;
        let write_timeout = OperationTimeout::parse("write_timeout_ms", input.write_timeout_ms)?;
        let write_attempts = WriteAttemptLimit::parse(input.write_attempts)?;
        let arming_freshness = ArmingFreshness::parse(input.arming_freshness_ms)?;
        let maximum_write_budget = write_timeout
            .get()
            .checked_mul(u32::from(write_attempts.get()))
            .expect("bounded write timeout and attempts fit Duration");
        if arming_freshness.get() < maximum_write_budget {
            return Err(ConfigParseError::ArmingFreshnessShorterThanWriteBudget {
                arming_freshness: arming_freshness.get(),
                maximum_write_budget,
            });
        }
        if input.noise_budget_bytes > MAX_NOISE_BUDGET_BYTES {
            return Err(ConfigParseError::NoiseBudgetOutOfRange {
                value: input.noise_budget_bytes,
                maximum: MAX_NOISE_BUDGET_BYTES,
            });
        }
        let redundant_read_tolerance = PositionAgreementTicks::try_new(
            input.redundant_read_tolerance_ticks,
        )
        .map_err(|source| ConfigParseError::InvalidPositionTolerance {
            field: "redundant_read_tolerance_ticks",
            source,
        })?;
        let readback_tolerance = PositionAgreementTicks::try_new(input.readback_tolerance_ticks)
            .map_err(|source| ConfigParseError::InvalidPositionTolerance {
                field: "readback_tolerance_ticks",
                source,
            })?;
        let goal_speed = GoalSpeedTicksPerSecond::try_new(input.goal_speed_ticks_per_second)
            .map_err(|source| ConfigParseError::InvalidGoalSpeed { source })?;
        let [bow, curl, yaw, roll] = input.torque_limit_permille;
        let torque_limits = HeadTorqueLimits::new(
            parse_torque_limit(HeadJoint::Bow, bow)?,
            parse_torque_limit(HeadJoint::Curl, curl)?,
            parse_torque_limit(HeadJoint::Yaw, yaw)?,
            parse_torque_limit(HeadJoint::Roll, roll)?,
        );

        Ok(Self {
            device,
            response_timeout,
            write_timeout,
            write_attempts,
            arming_freshness,
            noise_budget_bytes: input.noise_budget_bytes,
            redundant_read_tolerance,
            readback_tolerance,
            goal_speed,
            torque_limits,
        })
    }

    pub const fn device(&self) -> &DeviceIdentity {
        &self.device
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

    pub const fn arming_freshness(&self) -> ArmingFreshness {
        self.arming_freshness
    }

    pub const fn noise_budget_bytes(&self) -> u16 {
        self.noise_budget_bytes
    }

    pub const fn redundant_read_tolerance(&self) -> PositionAgreementTicks {
        self.redundant_read_tolerance
    }

    pub const fn readback_tolerance(&self) -> PositionAgreementTicks {
        self.readback_tolerance
    }

    pub const fn goal_speed(&self) -> GoalSpeedTicksPerSecond {
        self.goal_speed
    }

    pub const fn torque_limits(&self) -> HeadTorqueLimits {
        self.torque_limits
    }
}

fn parse_torque_limit(
    joint: HeadJoint,
    value: u16,
) -> Result<TorqueLimitPermille, ConfigParseError> {
    TorqueLimitPermille::try_new(value)
        .map_err(|source| ConfigParseError::InvalidTorqueLimit { joint, source })
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
    ArmingFreshnessOutOfRange {
        milliseconds: u64,
        minimum_ms: u64,
        maximum_ms: u64,
    },
    ArmingFreshnessShorterThanWriteBudget {
        arming_freshness: Duration,
        maximum_write_budget: Duration,
    },
    NoiseBudgetOutOfRange {
        value: u16,
        maximum: u16,
    },
    InvalidPositionTolerance {
        field: &'static str,
        source: PositionAgreementError,
    },
    InvalidGoalSpeed {
        source: FrameBuildError,
    },
    InvalidTorqueLimit {
        joint: HeadJoint,
        source: FrameBuildError,
    },
}

impl fmt::Display for ConfigParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid Kiko head runtime configuration: {self:?}"
        )
    }
}

impl std::error::Error for ConfigParseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidPositionTolerance { source, .. } => Some(source),
            Self::InvalidGoalSpeed { source } | Self::InvalidTorqueLimit { source, .. } => {
                Some(source)
            }
            Self::EmptyDevicePath
            | Self::DevicePathTooLong { .. }
            | Self::DevicePathContainsNul
            | Self::UnsupportedDevicePath { .. }
            | Self::InvalidDeviceIdentity { .. }
            | Self::DeviceIdentityTooLong { .. }
            | Self::OperationTimeoutOutOfRange { .. }
            | Self::WriteAttemptsOutOfRange { .. }
            | Self::ArmingFreshnessOutOfRange { .. }
            | Self::ArmingFreshnessShorterThanWriteBudget { .. }
            | Self::NoiseBudgetOutOfRange { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_input() -> HeadRuntimeConfigInput {
        HeadRuntimeConfigInput {
            device_path: "/dev/serial/by-id/usb-Kiko_STS_adapter_0001".to_owned(),
            response_timeout_ms: 100,
            write_timeout_ms: 100,
            write_attempts: 2,
            arming_freshness_ms: 250,
            noise_budget_bytes: 32,
            redundant_read_tolerance_ticks: 10,
            readback_tolerance_ticks: 20,
            goal_speed_ticks_per_second: 100,
            torque_limit_permille: [600, 400, 400, 400],
        }
    }

    #[test]
    fn configured_pose_windows_are_bounded_and_cannot_admit_the_full_encoder_domain() {
        ConfiguredHeadPoseBounds::try_new(
            [1_900; 4],
            [1_900 + MAX_CONFIGURED_POSE_WINDOW_SPAN_TICKS; 4],
        )
        .expect("maximum bounded pose-window span");

        assert!(matches!(
            ConfiguredHeadPoseBounds::try_new(
                [1_900; 4],
                [1_901 + MAX_CONFIGURED_POSE_WINDOW_SPAN_TICKS; 4],
            ),
            Err(ConfiguredHeadPoseBoundsError::SpanAboveMaximum {
                joint: HeadJoint::Bow,
                span_ticks,
                maximum_span_ticks: MAX_CONFIGURED_POSE_WINDOW_SPAN_TICKS,
                ..
            }) if span_ticks == MAX_CONFIGURED_POSE_WINDOW_SPAN_TICKS + 1
        ));
        assert!(matches!(
            ConfiguredHeadPoseBounds::try_new([0; 4], [4_095; 4]),
            Err(ConfiguredHeadPoseBoundsError::SpanAboveMaximum {
                joint: HeadJoint::Bow,
                span_ticks: 4_095,
                ..
            })
        ));
        assert!(matches!(
            ConfiguredHeadPoseBounds::try_new([2_000; 4], [1_999; 4]),
            Err(ConfiguredHeadPoseBoundsError::Descending {
                joint: HeadJoint::Bow,
                ..
            })
        ));
    }

    #[test]
    fn parses_linux_and_macos_exact_device_identities() {
        let linux = HeadRuntimeConfig::parse(valid_input()).expect("Linux by-id config");
        assert_eq!(linux.device().kind(), DeviceIdentityKind::LinuxById);
        assert_eq!(linux.device().stable_name(), "usb-Kiko_STS_adapter_0001");

        let mut mac_input = valid_input();
        mac_input.device_path = "/dev/cu.usbserial-KIKO0001".to_owned();
        let mac = HeadRuntimeConfig::parse(mac_input).expect("macOS callout config");
        assert_eq!(mac.device().kind(), DeviceIdentityKind::MacOsCallout);
    }

    #[test]
    fn rejects_ambiguous_devices_and_every_dangerous_zero() {
        for path in ["/dev/ttyUSB0", "/dev/serial/by-id/", "/dev/cu."] {
            let mut input = valid_input();
            input.device_path = path.to_owned();
            assert!(HeadRuntimeConfig::parse(input).is_err(), "accepted {path}");
        }

        let mut input = valid_input();
        input.response_timeout_ms = 0;
        assert!(HeadRuntimeConfig::parse(input).is_err());
        let mut input = valid_input();
        input.write_attempts = 0;
        assert!(HeadRuntimeConfig::parse(input).is_err());
        let mut input = valid_input();
        input.arming_freshness_ms = 0;
        assert!(HeadRuntimeConfig::parse(input).is_err());
        let mut input = valid_input();
        input.write_timeout_ms = 100;
        input.write_attempts = 2;
        input.arming_freshness_ms = 199;
        assert!(matches!(
            HeadRuntimeConfig::parse(input),
            Err(ConfigParseError::ArmingFreshnessShorterThanWriteBudget { .. })
        ));
        let mut input = valid_input();
        input.goal_speed_ticks_per_second = 0;
        assert!(HeadRuntimeConfig::parse(input).is_err());
        let mut input = valid_input();
        input.torque_limit_permille[2] = 0;
        assert!(matches!(
            HeadRuntimeConfig::parse(input),
            Err(ConfigParseError::InvalidTorqueLimit {
                joint: HeadJoint::Yaw,
                ..
            })
        ));
    }

    #[test]
    fn all_numeric_bounds_are_exact() {
        let mut input = valid_input();
        input.response_timeout_ms = MAX_OPERATION_TIMEOUT_MS;
        input.write_timeout_ms = 1;
        input.write_attempts = MAX_WRITE_ATTEMPTS;
        input.arming_freshness_ms = MAX_ARMING_FRESHNESS_MS;
        input.noise_budget_bytes = MAX_NOISE_BUDGET_BYTES;
        input.redundant_read_tolerance_ticks = 50;
        input.readback_tolerance_ticks = 50;
        assert!(HeadRuntimeConfig::parse(input).is_ok());

        let mut input = valid_input();
        input.response_timeout_ms = MAX_OPERATION_TIMEOUT_MS + 1;
        assert!(HeadRuntimeConfig::parse(input).is_err());
        let mut input = valid_input();
        input.write_attempts = MAX_WRITE_ATTEMPTS + 1;
        assert!(HeadRuntimeConfig::parse(input).is_err());
        let mut input = valid_input();
        input.arming_freshness_ms = MAX_ARMING_FRESHNESS_MS + 1;
        assert!(HeadRuntimeConfig::parse(input).is_err());
        let mut input = valid_input();
        input.noise_budget_bytes = MAX_NOISE_BUDGET_BYTES + 1;
        assert!(HeadRuntimeConfig::parse(input).is_err());
    }
}
