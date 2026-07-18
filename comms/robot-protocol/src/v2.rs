//! Version-two robot-control framing and domain messages.
//!
//! V2 deliberately separates a fixed, integrity-checked wire envelope from
//! parsed domain messages. UDP carries exactly one raw frame per datagram.
//! UART carries the same raw frame encoded with COBS and terminated by one
//! zero byte, which gives a bounded resynchronization point after corruption.
//! No V1 ASCII or legacy packet is accepted by this module.
//!
//! `timer_*_pwm_percent` fields report the signed duty request written to the
//! controller timer channels. They are not encoder velocity, wheel motion,
//! motor current, torque, or proof that the robot moved.

use core::num::{NonZeroU16, NonZeroU32, NonZeroU64};

use crate::{
    ControllerUptimeMsWrapping, EstimatedWrappingEncoderTicks, ModuloEncoderDeltaTicks, PwmPercent,
    PwmPercentError,
};

pub const MAGIC: [u8; 4] = *b"KRP2";
pub const VERSION: u8 = 2;
pub const HEADER_BYTES: usize = 8;
pub const CRC_BYTES: usize = 4;
pub const MAX_PAYLOAD_BYTES: usize = 60;
pub const MAX_RAW_FRAME_BYTES: usize = HEADER_BYTES + MAX_PAYLOAD_BYTES + CRC_BYTES;
pub const MAX_COBS_FRAME_BYTES: usize = MAX_RAW_FRAME_BYTES + 1;
pub const MAX_UART_RECORD_BYTES: usize = MAX_COBS_FRAME_BYTES + 1;

pub const MIN_V2_COMMAND_LEASE_MS: u16 = 50;
pub const MAX_V2_COMMAND_LEASE_MS: u16 = 250;
pub const MAX_HEARTBEAT_PERIOD_MS: u16 = 1_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DomainError {
    ZeroControllerUid,
    ZeroControllerBootId,
    ZeroControlEpoch,
    ZeroActuatorConfigFingerprint,
    CommandLeaseOutOfRange {
        value: u16,
        minimum: u16,
        maximum: u16,
    },
    RemainingLeaseAboveMaximum {
        value: u16,
        maximum: u16,
    },
    HeartbeatPeriodOutOfRange {
        value: u16,
        minimum: u16,
        maximum: u16,
    },
    MaxPwmOutOfRange {
        value: u8,
        minimum: u8,
        maximum: u8,
    },
    WatchdogPeriodOutOfRange {
        value: u16,
        minimum: u16,
        maximum: u16,
    },
    ZeroPwmFrequency,
    UnknownCapabilityBits {
        bits: u32,
    },
    UnknownReadinessBits {
        bits: u16,
    },
    UnknownFaultBits {
        bits: u32,
    },
    LeftPwm(PwmPercentError),
    RightPwm(PwmPercentError),
}

impl core::fmt::Display for DomainError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ZeroControllerUid => formatter.write_str("controller UID must not be all zero"),
            Self::ZeroControllerBootId => formatter.write_str("controller boot ID must be nonzero"),
            Self::ZeroControlEpoch => formatter.write_str("control epoch must be nonzero"),
            Self::ZeroActuatorConfigFingerprint => {
                formatter.write_str("actuator-config fingerprint must not be all zero")
            }
            Self::CommandLeaseOutOfRange {
                value,
                minimum,
                maximum,
            } => write!(
                formatter,
                "V2 command lease {value} ms is outside {minimum}..={maximum} ms"
            ),
            Self::RemainingLeaseAboveMaximum { value, maximum } => write!(
                formatter,
                "remaining command lease {value} ms exceeds {maximum} ms"
            ),
            Self::HeartbeatPeriodOutOfRange {
                value,
                minimum,
                maximum,
            } => write!(
                formatter,
                "heartbeat period {value} ms is outside {minimum}..={maximum} ms"
            ),
            Self::MaxPwmOutOfRange {
                value,
                minimum,
                maximum,
            } => write!(
                formatter,
                "maximum absolute PWM {value}% is outside {minimum}..={maximum}%"
            ),
            Self::WatchdogPeriodOutOfRange {
                value,
                minimum,
                maximum,
            } => write!(
                formatter,
                "watchdog period {value} ms is outside {minimum}..={maximum} ms"
            ),
            Self::ZeroPwmFrequency => formatter.write_str("PWM frequency must be nonzero"),
            Self::UnknownCapabilityBits { bits } => {
                write!(formatter, "unknown controller capability bits 0x{bits:08x}")
            }
            Self::UnknownReadinessBits { bits } => {
                write!(formatter, "unknown controller readiness bits 0x{bits:04x}")
            }
            Self::UnknownFaultBits { bits } => {
                write!(formatter, "unknown controller fault bits 0x{bits:08x}")
            }
            Self::LeftPwm(source) => write!(formatter, "invalid left timer PWM: {source}"),
            Self::RightPwm(source) => write!(formatter, "invalid right timer PWM: {source}"),
        }
    }
}

impl core::error::Error for DomainError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::LeftPwm(source) | Self::RightPwm(source) => Some(source),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ControllerUid([u8; 12]);

impl ControllerUid {
    pub fn try_new(bytes: [u8; 12]) -> Result<Self, DomainError> {
        if bytes == [0; 12] {
            Err(DomainError::ZeroControllerUid)
        } else {
            Ok(Self(bytes))
        }
    }

    pub const fn as_bytes(&self) -> &[u8; 12] {
        &self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ControllerBootId(NonZeroU64);

impl ControllerBootId {
    pub fn try_new(value: u64) -> Result<Self, DomainError> {
        NonZeroU64::new(value)
            .map(Self)
            .ok_or(DomainError::ZeroControllerBootId)
    }

    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ControlEpoch(NonZeroU32);

impl ControlEpoch {
    pub fn try_new(value: u32) -> Result<Self, DomainError> {
        NonZeroU32::new(value)
            .map(Self)
            .ok_or(DomainError::ZeroControlEpoch)
    }

    pub const fn get(self) -> u32 {
        self.0.get()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RequestId(u32);

impl RequestId {
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u32 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct V2CommandSequence(u32);

impl V2CommandSequence {
    pub const FIRST: Self = Self(0);

    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u32 {
        self.0
    }

    pub const fn checked_successor(self) -> Option<Self> {
        match self.0.checked_add(1) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct V2CommandLeaseMs(NonZeroU16);

impl V2CommandLeaseMs {
    pub fn try_new(value: u16) -> Result<Self, DomainError> {
        if !(MIN_V2_COMMAND_LEASE_MS..=MAX_V2_COMMAND_LEASE_MS).contains(&value) {
            return Err(DomainError::CommandLeaseOutOfRange {
                value,
                minimum: MIN_V2_COMMAND_LEASE_MS,
                maximum: MAX_V2_COMMAND_LEASE_MS,
            });
        }
        let value = NonZeroU16::new(value).ok_or(DomainError::CommandLeaseOutOfRange {
            value: 0,
            minimum: MIN_V2_COMMAND_LEASE_MS,
            maximum: MAX_V2_COMMAND_LEASE_MS,
        })?;
        Ok(Self(value))
    }

    pub const fn get(self) -> u16 {
        self.0.get()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RemainingLeaseMs(u16);

impl RemainingLeaseMs {
    pub const ZERO: Self = Self(0);

    pub fn try_new(value: u16) -> Result<Self, DomainError> {
        if value > MAX_V2_COMMAND_LEASE_MS {
            Err(DomainError::RemainingLeaseAboveMaximum {
                value,
                maximum: MAX_V2_COMMAND_LEASE_MS,
            })
        } else {
            Ok(Self(value))
        }
    }

    pub const fn get(self) -> u16 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeartbeatPeriodMs(NonZeroU16);

impl HeartbeatPeriodMs {
    pub fn try_new(value: u16) -> Result<Self, DomainError> {
        let Some(value) = NonZeroU16::new(value) else {
            return Err(DomainError::HeartbeatPeriodOutOfRange {
                value: 0,
                minimum: 1,
                maximum: MAX_HEARTBEAT_PERIOD_MS,
            });
        };
        if value.get() > MAX_HEARTBEAT_PERIOD_MS {
            return Err(DomainError::HeartbeatPeriodOutOfRange {
                value: value.get(),
                minimum: 1,
                maximum: MAX_HEARTBEAT_PERIOD_MS,
            });
        }
        Ok(Self(value))
    }

    pub const fn get(self) -> u16 {
        self.0.get()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MaxAbsPwmPercent(u8);

impl MaxAbsPwmPercent {
    pub fn try_new(value: u8) -> Result<Self, DomainError> {
        if value > 100 {
            Err(DomainError::MaxPwmOutOfRange {
                value,
                minimum: 0,
                maximum: 100,
            })
        } else {
            Ok(Self(value))
        }
    }

    pub const fn get(self) -> u8 {
        self.0
    }

    /// Zero truthfully advertises that the controller profile grants no
    /// motion authority. A server must require a nonzero value before starting
    /// a motion-capable session.
    pub const fn grants_motion_authority(self) -> bool {
        self.0 != 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ActuatorConfigFingerprint([u8; 16]);

impl ActuatorConfigFingerprint {
    pub fn try_new(bytes: [u8; 16]) -> Result<Self, DomainError> {
        if bytes == [0; 16] {
            Err(DomainError::ZeroActuatorConfigFingerprint)
        } else {
            Ok(Self(bytes))
        }
    }

    pub const fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WatchdogNominalPeriodMs(NonZeroU16);

impl WatchdogNominalPeriodMs {
    pub fn try_new(value: u16) -> Result<Self, DomainError> {
        let Some(value) = NonZeroU16::new(value) else {
            return Err(DomainError::WatchdogPeriodOutOfRange {
                value: 0,
                minimum: 1,
                maximum: 1_000,
            });
        };
        if value.get() > 1_000 {
            return Err(DomainError::WatchdogPeriodOutOfRange {
                value: value.get(),
                minimum: 1,
                maximum: 1_000,
            });
        }
        Ok(Self(value))
    }

    pub const fn get(self) -> u16 {
        self.0.get()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PwmFrequencyHz(NonZeroU16);

impl PwmFrequencyHz {
    pub fn try_new(value: u16) -> Result<Self, DomainError> {
        NonZeroU16::new(value)
            .map(Self)
            .ok_or(DomainError::ZeroPwmFrequency)
    }

    pub const fn get(self) -> u16 {
        self.0.get()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum NeutralOutput {
    BothLow = 0,
    BothHigh = 1,
    HighImpedance = 2,
}

impl NeutralOutput {
    fn parse(value: u8) -> Result<Self, PayloadError> {
        match value {
            0 => Ok(Self::BothLow),
            1 => Ok(Self::BothHigh),
            2 => Ok(Self::HighImpedance),
            _ => Err(PayloadError::UnknownEnum {
                field: "neutral-output encoding",
                value,
            }),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum PhysicalStopSemantics {
    Unverified = 0,
    CoastVerified = 1,
    BrakeVerified = 2,
}

impl PhysicalStopSemantics {
    fn parse(value: u8) -> Result<Self, PayloadError> {
        match value {
            0 => Ok(Self::Unverified),
            1 => Ok(Self::CoastVerified),
            2 => Ok(Self::BrakeVerified),
            _ => Err(PayloadError::UnknownEnum {
                field: "physical-stop semantics",
                value,
            }),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TimerPwm {
    left: PwmPercent,
    right: PwmPercent,
}

impl TimerPwm {
    pub const ZERO: Self = Self {
        left: PwmPercent::ZERO,
        right: PwmPercent::ZERO,
    };

    pub fn try_new(left: i8, right: i8) -> Result<Self, DomainError> {
        Ok(Self {
            left: PwmPercent::try_new(left).map_err(DomainError::LeftPwm)?,
            right: PwmPercent::try_new(right).map_err(DomainError::RightPwm)?,
        })
    }

    pub const fn from_validated(left: PwmPercent, right: PwmPercent) -> Self {
        Self { left, right }
    }

    pub const fn left(self) -> PwmPercent {
        self.left
    }

    pub const fn right(self) -> PwmPercent {
        self.right
    }

    pub const fn is_zero(self) -> bool {
        self.left.get() == 0 && self.right.get() == 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum OutputState {
    Disabled = 0,
    ZeroPwm = 1,
    NonzeroPwm = 2,
}

impl OutputState {
    fn parse(value: u8) -> Result<Self, PayloadError> {
        match value {
            0 => Ok(Self::Disabled),
            1 => Ok(Self::ZeroPwm),
            2 => Ok(Self::NonzeroPwm),
            _ => Err(PayloadError::UnknownEnum {
                field: "output state",
                value,
            }),
        }
    }

    pub const fn is_safe(self) -> bool {
        !matches!(self, Self::NonzeroPwm)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ControllerCapabilities(u32);

impl ControllerCapabilities {
    pub const DEADLINE_TIMER_ISR: u32 = 1 << 0;
    pub const INDEPENDENT_WATCHDOG: u32 = 1 << 1;
    pub const BREAK_BEFORE_MAKE: u32 = 1 << 2;
    pub const APPLIED_ACK: u32 = 1 << 3;
    pub const HEARTBEAT: u32 = 1 << 4;
    pub const V2_ONLY: u32 = 1 << 5;
    /// The selected hardware profile claims an external motor-driver enable
    /// gate is configured. This is configuration evidence, not physical proof.
    pub const EXTERNAL_DRIVER_ENABLE_GATE_CONFIGURED: u32 = 1 << 6;
    /// The selected hardware profile claims a driver-fault input is configured.
    /// This is configuration evidence, not physical proof.
    pub const DRIVER_FAULT_INPUT_CONFIGURED: u32 = 1 << 7;
    pub const KNOWN_BITS: u32 = (1 << 8) - 1;
    pub const REQUIRED_BITS: u32 = Self::KNOWN_BITS;

    pub fn try_from_bits(bits: u32) -> Result<Self, DomainError> {
        if bits & !Self::KNOWN_BITS != 0 {
            Err(DomainError::UnknownCapabilityBits { bits })
        } else {
            Ok(Self(bits))
        }
    }

    pub const fn bits(self) -> u32 {
        self.0
    }

    pub const fn supports_required_safety(self) -> bool {
        self.0 & Self::REQUIRED_BITS == Self::REQUIRED_BITS
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReadinessFlags(u16);

impl ReadinessFlags {
    pub const SESSION_ESTABLISHED: u16 = 1 << 0;
    pub const DEADLINE_ARMED: u16 = 1 << 1;
    pub const WATCHDOG_RUNNING: u16 = 1 << 2;
    pub const DRIVER_FAULT_CLEAR: u16 = 1 << 3;
    pub const KNOWN_BITS: u16 = (1 << 4) - 1;
    pub const READY_BITS: u16 = Self::KNOWN_BITS;

    pub fn try_from_bits(bits: u16) -> Result<Self, DomainError> {
        if bits & !Self::KNOWN_BITS != 0 {
            Err(DomainError::UnknownReadinessBits { bits })
        } else {
            Ok(Self(bits))
        }
    }

    pub const fn bits(self) -> u16 {
        self.0
    }

    pub const fn is_ready(self) -> bool {
        self.0 & Self::READY_BITS == Self::READY_BITS
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ControllerFaults(u32);

impl ControllerFaults {
    pub const NONE: Self = Self(0);
    pub const SERIAL_INTEGRITY: u32 = 1 << 0;
    pub const DEADLINE_EXPIRED: u32 = 1 << 1;
    pub const SEQUENCE: u32 = 1 << 2;
    pub const MOTOR_DRIVER: u32 = 1 << 3;
    pub const WATCHDOG: u32 = 1 << 4;
    pub const BOOT_COUNTER: u32 = 1 << 5;
    pub const INTERNAL: u32 = 1 << 6;
    pub const KNOWN_BITS: u32 = (1 << 7) - 1;

    pub fn try_from_bits(bits: u32) -> Result<Self, DomainError> {
        if bits & !Self::KNOWN_BITS != 0 {
            Err(DomainError::UnknownFaultBits { bits })
        } else {
            Ok(Self(bits))
        }
    }

    pub const fn bits(self) -> u32 {
        self.0
    }

    pub const fn is_clear(self) -> bool {
        self.0 == 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetBootId {
    Any,
    Exact(ControllerBootId),
}

impl TargetBootId {
    const fn to_wire(self) -> u64 {
        match self {
            Self::Any => 0,
            Self::Exact(value) => value.get(),
        }
    }

    fn from_wire(value: u64) -> Result<Self, DomainError> {
        if value == 0 {
            Ok(Self::Any)
        } else {
            ControllerBootId::try_new(value).map(Self::Exact)
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ForceStopReason {
    Operator = 0,
    LeaseExpired = 1,
    TransportFault = 2,
    ControllerFault = 3,
    SessionReset = 4,
    SequenceConflict = 5,
}

impl ForceStopReason {
    fn parse(value: u8) -> Result<Self, PayloadError> {
        match value {
            0 => Ok(Self::Operator),
            1 => Ok(Self::LeaseExpired),
            2 => Ok(Self::TransportFault),
            3 => Ok(Self::ControllerFault),
            4 => Ok(Self::SessionReset),
            5 => Ok(Self::SequenceConflict),
            _ => Err(PayloadError::UnknownEnum {
                field: "force-stop reason",
                value,
            }),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum AppliedResultCode {
    AppliedNew = 0,
    DuplicateCached = 1,
    Stopped = 2,
    RejectedExpired = 3,
    RejectedSession = 4,
    RejectedSequence = 5,
    RejectedDomain = 6,
    FaultedStop = 7,
}

impl AppliedResultCode {
    fn parse(value: u8) -> Result<Self, PayloadError> {
        match value {
            0 => Ok(Self::AppliedNew),
            1 => Ok(Self::DuplicateCached),
            2 => Ok(Self::Stopped),
            3 => Ok(Self::RejectedExpired),
            4 => Ok(Self::RejectedSession),
            5 => Ok(Self::RejectedSequence),
            6 => Ok(Self::RejectedDomain),
            7 => Ok(Self::FaultedStop),
            _ => Err(PayloadError::UnknownEnum {
                field: "applied-result code",
                value,
            }),
        }
    }

    pub const fn proves_applied(self) -> bool {
        matches!(
            self,
            Self::AppliedNew | Self::DuplicateCached | Self::Stopped
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum AcquireResultCode {
    Granted = 0,
    ControllerUnavailable = 1,
    NotReady = 2,
    IdentityMismatch = 3,
    Faulted = 4,
    Busy = 5,
}

impl AcquireResultCode {
    fn parse(value: u8) -> Result<Self, PayloadError> {
        match value {
            0 => Ok(Self::Granted),
            1 => Ok(Self::ControllerUnavailable),
            2 => Ok(Self::NotReady),
            3 => Ok(Self::IdentityMismatch),
            4 => Ok(Self::Faulted),
            5 => Ok(Self::Busy),
            _ => Err(PayloadError::UnknownEnum {
                field: "acquire-result code",
                value,
            }),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum HostCommandResultCode {
    AppliedNew = 0,
    DuplicateCached = 1,
    Stopped = 2,
    RejectedAtServer = 3,
    RejectedByController = 4,
    AppliedAckTimeout = 5,
    ControllerRestarted = 6,
    ForceStopped = 7,
}

impl HostCommandResultCode {
    fn parse(value: u8) -> Result<Self, PayloadError> {
        match value {
            0 => Ok(Self::AppliedNew),
            1 => Ok(Self::DuplicateCached),
            2 => Ok(Self::Stopped),
            3 => Ok(Self::RejectedAtServer),
            4 => Ok(Self::RejectedByController),
            5 => Ok(Self::AppliedAckTimeout),
            6 => Ok(Self::ControllerRestarted),
            7 => Ok(Self::ForceStopped),
            _ => Err(PayloadError::UnknownEnum {
                field: "host-command-result code",
                value,
            }),
        }
    }

    pub const fn proves_controller_application(self) -> bool {
        matches!(
            self,
            Self::AppliedNew | Self::DuplicateCached | Self::Stopped
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum StopResultCode {
    ControllerConfirmed = 0,
    ControllerUnavailable = 1,
    IdentityMismatch = 2,
    StopAckTimeout = 3,
    ControllerFaulted = 4,
}

impl StopResultCode {
    fn parse(value: u8) -> Result<Self, PayloadError> {
        match value {
            0 => Ok(Self::ControllerConfirmed),
            1 => Ok(Self::ControllerUnavailable),
            2 => Ok(Self::IdentityMismatch),
            3 => Ok(Self::StopAckTimeout),
            4 => Ok(Self::ControllerFaulted),
            _ => Err(PayloadError::UnknownEnum {
                field: "stop-result code",
                value,
            }),
        }
    }

    pub const fn proves_controller_stop(self) -> bool {
        matches!(self, Self::ControllerConfirmed)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum StatusCode {
    ReadyStopped = 0,
    ReadyActive = 1,
    Disconnected = 2,
    EstablishingSession = 3,
    Faulted = 4,
}

impl StatusCode {
    fn parse(value: u8) -> Result<Self, PayloadError> {
        match value {
            0 => Ok(Self::ReadyStopped),
            1 => Ok(Self::ReadyActive),
            2 => Ok(Self::Disconnected),
            3 => Ok(Self::EstablishingSession),
            4 => Ok(Self::Faulted),
            _ => Err(PayloadError::UnknownEnum {
                field: "status code",
                value,
            }),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum MessageKind {
    AcquireControl = 0x01,
    HostCommand = 0x02,
    HostStop = 0x03,
    StatusQuery = 0x04,
    ControllerHello = 0x10,
    BeginSession = 0x11,
    ControllerReady = 0x12,
    ApplyPwm = 0x20,
    ForceStop = 0x21,
    AppliedResult = 0x30,
    Heartbeat = 0x31,
    ObservationalOdometry = 0x32,
    AcquireResult = 0x81,
    HostCommandResult = 0x82,
    HostStopResult = 0x83,
    StatusReport = 0x84,
}

impl MessageKind {
    fn parse(value: u8) -> Result<Self, FrameError> {
        match value {
            0x01 => Ok(Self::AcquireControl),
            0x02 => Ok(Self::HostCommand),
            0x03 => Ok(Self::HostStop),
            0x04 => Ok(Self::StatusQuery),
            0x10 => Ok(Self::ControllerHello),
            0x11 => Ok(Self::BeginSession),
            0x12 => Ok(Self::ControllerReady),
            0x20 => Ok(Self::ApplyPwm),
            0x21 => Ok(Self::ForceStop),
            0x30 => Ok(Self::AppliedResult),
            0x31 => Ok(Self::Heartbeat),
            0x32 => Ok(Self::ObservationalOdometry),
            0x81 => Ok(Self::AcquireResult),
            0x82 => Ok(Self::HostCommandResult),
            0x83 => Ok(Self::HostStopResult),
            0x84 => Ok(Self::StatusReport),
            _ => Err(FrameError::UnknownMessageKind { value }),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ControllerDeadlineMsWrapping(u32);

impl ControllerDeadlineMsWrapping {
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u32 {
        self.0
    }

    /// Classifies `now` against this deadline in the wrapping half-range.
    ///
    /// A zero delta is expired. Values at exactly half the `u32` range are
    /// deliberately ambiguous and must not authorize motion.
    pub const fn relation_to(self, now: ControllerUptimeMsWrapping) -> DeadlineRelation {
        let delta = self.0.wrapping_sub(now.get());
        match delta {
            0 => DeadlineRelation::Expired,
            0x8000_0000 => DeadlineRelation::AmbiguousHalfRange,
            1..0x8000_0000 => DeadlineRelation::Future {
                remaining_ms: delta,
            },
            _ => DeadlineRelation::Expired,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeadlineRelation {
    Future { remaining_ms: u32 },
    Expired,
    AmbiguousHalfRange,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ControllerHello {
    pub controller_uid: ControllerUid,
    pub boot_id: ControllerBootId,
    pub firmware_abi: u16,
    pub firmware_build_id: u32,
    pub capabilities: ControllerCapabilities,
    pub max_abs_pwm_percent: MaxAbsPwmPercent,
    pub max_command_lease: V2CommandLeaseMs,
    pub output_state: OutputState,
    pub actuator_config_fingerprint: ActuatorConfigFingerprint,
    pub watchdog_nominal_period: WatchdogNominalPeriodMs,
    pub pwm_frequency: PwmFrequencyHz,
    pub neutral_output: NeutralOutput,
    pub physical_stop_semantics: PhysicalStopSemantics,
}

impl ControllerHello {
    pub const PAYLOAD_BYTES: usize = 56;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BeginSession {
    pub controller_uid: ControllerUid,
    pub boot_id: ControllerBootId,
    pub request_id: RequestId,
    pub heartbeat_period: HeartbeatPeriodMs,
}

impl BeginSession {
    pub const PAYLOAD_BYTES: usize = 26;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ControllerReady {
    pub controller_uid: ControllerUid,
    pub boot_id: ControllerBootId,
    pub control_epoch: ControlEpoch,
    pub controller_uptime: ControllerUptimeMsWrapping,
    pub capabilities: ControllerCapabilities,
    pub output_state: OutputState,
    pub faults: ControllerFaults,
}

impl ControllerReady {
    pub const PAYLOAD_BYTES: usize = 37;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ApplyPwm {
    pub controller_uid: ControllerUid,
    pub boot_id: ControllerBootId,
    pub control_epoch: ControlEpoch,
    pub sequence: V2CommandSequence,
    pub expires_at: ControllerDeadlineMsWrapping,
    pub timer_pwm: TimerPwm,
}

impl ApplyPwm {
    pub const PAYLOAD_BYTES: usize = 34;

    pub const fn is_initial_zero_acquisition(self) -> bool {
        self.sequence.get() == V2CommandSequence::FIRST.get() && self.timer_pwm.is_zero()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ForceStop {
    pub controller_uid: ControllerUid,
    pub target_boot_id: TargetBootId,
    pub request_id: RequestId,
    pub reason: ForceStopReason,
}

impl ForceStop {
    pub const PAYLOAD_BYTES: usize = 25;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AppliedResult {
    pub controller_uid: ControllerUid,
    pub boot_id: ControllerBootId,
    pub control_epoch: ControlEpoch,
    pub sequence: V2CommandSequence,
    pub result: AppliedResultCode,
    pub timer_pwm: TimerPwm,
    pub output_state: OutputState,
    pub applied_at: ControllerUptimeMsWrapping,
    pub expires_at: ControllerDeadlineMsWrapping,
    pub faults: ControllerFaults,
}

impl AppliedResult {
    pub const PAYLOAD_BYTES: usize = 44;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Heartbeat {
    pub controller_uid: ControllerUid,
    pub boot_id: ControllerBootId,
    pub control_epoch: Option<ControlEpoch>,
    pub last_sequence: Option<V2CommandSequence>,
    pub controller_uptime: ControllerUptimeMsWrapping,
    pub expires_at: ControllerDeadlineMsWrapping,
    pub timer_pwm: TimerPwm,
    pub output_state: OutputState,
    pub readiness: ReadinessFlags,
    pub faults: ControllerFaults,
}

impl Heartbeat {
    pub const PAYLOAD_BYTES: usize = 45;
}

/// Observational encoder telemetry on the V2 serial stream.
///
/// Extended counts remain explicitly estimated and wrapping; sample deltas
/// remain modulo-`2^16`. This message neither commands nor proves actuation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ObservationalOdometry {
    pub controller_uid: ControllerUid,
    pub boot_id: ControllerBootId,
    pub control_epoch: Option<ControlEpoch>,
    pub left_estimated_extended_ticks_wrapping: EstimatedWrappingEncoderTicks,
    pub right_estimated_extended_ticks_wrapping: EstimatedWrappingEncoderTicks,
    pub left_sample_delta_ticks_modulo: ModuloEncoderDeltaTicks,
    pub right_sample_delta_ticks_modulo: ModuloEncoderDeltaTicks,
    pub controller_uptime: ControllerUptimeMsWrapping,
}

impl ObservationalOdometry {
    pub const PAYLOAD_BYTES: usize = 48;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AcquireControl {
    pub expected_controller_uid: ControllerUid,
    pub expected_boot_id: ControllerBootId,
    pub request_id: RequestId,
    pub expected_firmware_abi: u16,
    pub expected_firmware_build_id: u32,
    pub expected_actuator_config_fingerprint: ActuatorConfigFingerprint,
}

impl AcquireControl {
    pub const PAYLOAD_BYTES: usize = 46;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AcquireResult {
    pub controller_uid: ControllerUid,
    pub boot_id: ControllerBootId,
    pub request_id: RequestId,
    pub control_epoch: Option<ControlEpoch>,
    pub result: AcquireResultCode,
    pub capabilities: ControllerCapabilities,
    pub faults: ControllerFaults,
    pub observed_firmware_abi: u16,
    pub observed_firmware_build_id: u32,
    pub observed_actuator_config_fingerprint: ActuatorConfigFingerprint,
}

impl AcquireResult {
    pub const PAYLOAD_BYTES: usize = 59;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostCommand {
    pub controller_uid: ControllerUid,
    pub boot_id: ControllerBootId,
    pub control_epoch: ControlEpoch,
    pub sequence: V2CommandSequence,
    pub lease: V2CommandLeaseMs,
    pub requested_timer_pwm: TimerPwm,
}

impl HostCommand {
    pub const PAYLOAD_BYTES: usize = 32;

    pub const fn is_initial_zero_acquisition(self) -> bool {
        self.sequence.get() == V2CommandSequence::FIRST.get() && self.requested_timer_pwm.is_zero()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostCommandResult {
    pub controller_uid: ControllerUid,
    pub boot_id: ControllerBootId,
    pub control_epoch: ControlEpoch,
    pub sequence: V2CommandSequence,
    pub result: HostCommandResultCode,
    pub requested_timer_pwm: TimerPwm,
    pub controller_timer_pwm: TimerPwm,
    pub output_state: OutputState,
    pub controller_applied_at: ControllerUptimeMsWrapping,
    pub controller_expires_at: ControllerDeadlineMsWrapping,
    /// Conservative server-emission lifetime. The server must never report a
    /// value later than the controller receipt's remaining absolute lifetime.
    pub remaining_lease: RemainingLeaseMs,
    pub faults: ControllerFaults,
}

impl HostCommandResult {
    pub const PAYLOAD_BYTES: usize = 48;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostStop {
    pub controller_uid: ControllerUid,
    pub target_boot_id: TargetBootId,
    pub request_id: RequestId,
    pub reason: ForceStopReason,
}

impl HostStop {
    pub const PAYLOAD_BYTES: usize = 25;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostStopResult {
    pub controller_uid: ControllerUid,
    pub observed_boot_id: TargetBootId,
    pub request_id: RequestId,
    pub result: StopResultCode,
    pub output_state: OutputState,
    pub controller_uptime: ControllerUptimeMsWrapping,
    pub faults: ControllerFaults,
}

impl HostStopResult {
    pub const PAYLOAD_BYTES: usize = 34;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StatusQuery {
    pub expected_controller_uid: ControllerUid,
    pub request_id: RequestId,
}

impl StatusQuery {
    pub const PAYLOAD_BYTES: usize = 16;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StatusReport {
    pub controller_uid: ControllerUid,
    pub observed_boot_id: TargetBootId,
    pub request_id: RequestId,
    pub status: StatusCode,
    pub control_epoch: Option<ControlEpoch>,
    pub controller_uptime: ControllerUptimeMsWrapping,
    pub capabilities: ControllerCapabilities,
    pub output_state: OutputState,
    pub controller_timer_pwm: TimerPwm,
    pub remaining_lease: RemainingLeaseMs,
    pub faults: ControllerFaults,
}

impl StatusReport {
    pub const PAYLOAD_BYTES: usize = 46;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Message {
    AcquireControl(AcquireControl),
    HostCommand(HostCommand),
    HostStop(HostStop),
    StatusQuery(StatusQuery),
    ControllerHello(ControllerHello),
    BeginSession(BeginSession),
    ControllerReady(ControllerReady),
    ApplyPwm(ApplyPwm),
    ForceStop(ForceStop),
    AppliedResult(AppliedResult),
    Heartbeat(Heartbeat),
    ObservationalOdometry(ObservationalOdometry),
    AcquireResult(AcquireResult),
    HostCommandResult(HostCommandResult),
    HostStopResult(HostStopResult),
    StatusReport(StatusReport),
}

impl Message {
    pub const fn kind(self) -> MessageKind {
        match self {
            Self::AcquireControl(_) => MessageKind::AcquireControl,
            Self::HostCommand(_) => MessageKind::HostCommand,
            Self::HostStop(_) => MessageKind::HostStop,
            Self::StatusQuery(_) => MessageKind::StatusQuery,
            Self::ControllerHello(_) => MessageKind::ControllerHello,
            Self::BeginSession(_) => MessageKind::BeginSession,
            Self::ControllerReady(_) => MessageKind::ControllerReady,
            Self::ApplyPwm(_) => MessageKind::ApplyPwm,
            Self::ForceStop(_) => MessageKind::ForceStop,
            Self::AppliedResult(_) => MessageKind::AppliedResult,
            Self::Heartbeat(_) => MessageKind::Heartbeat,
            Self::ObservationalOdometry(_) => MessageKind::ObservationalOdometry,
            Self::AcquireResult(_) => MessageKind::AcquireResult,
            Self::HostCommandResult(_) => MessageKind::HostCommandResult,
            Self::HostStopResult(_) => MessageKind::HostStopResult,
            Self::StatusReport(_) => MessageKind::StatusReport,
        }
    }

    pub const fn payload_len(self) -> usize {
        match self {
            Self::AcquireControl(_) => AcquireControl::PAYLOAD_BYTES,
            Self::HostCommand(_) => HostCommand::PAYLOAD_BYTES,
            Self::HostStop(_) => HostStop::PAYLOAD_BYTES,
            Self::StatusQuery(_) => StatusQuery::PAYLOAD_BYTES,
            Self::ControllerHello(_) => ControllerHello::PAYLOAD_BYTES,
            Self::BeginSession(_) => BeginSession::PAYLOAD_BYTES,
            Self::ControllerReady(_) => ControllerReady::PAYLOAD_BYTES,
            Self::ApplyPwm(_) => ApplyPwm::PAYLOAD_BYTES,
            Self::ForceStop(_) => ForceStop::PAYLOAD_BYTES,
            Self::AppliedResult(_) => AppliedResult::PAYLOAD_BYTES,
            Self::Heartbeat(_) => Heartbeat::PAYLOAD_BYTES,
            Self::ObservationalOdometry(_) => ObservationalOdometry::PAYLOAD_BYTES,
            Self::AcquireResult(_) => AcquireResult::PAYLOAD_BYTES,
            Self::HostCommandResult(_) => HostCommandResult::PAYLOAD_BYTES,
            Self::HostStopResult(_) => HostStopResult::PAYLOAD_BYTES,
            Self::StatusReport(_) => StatusReport::PAYLOAD_BYTES,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PayloadError {
    WrongLength {
        kind: MessageKind,
        expected: usize,
        actual: usize,
    },
    UnexpectedEnd,
    TrailingBytes {
        remaining: usize,
    },
    UnknownEnum {
        field: &'static str,
        value: u8,
    },
    Domain(DomainError),
    Invariant {
        detail: &'static str,
    },
}

impl core::fmt::Display for PayloadError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::WrongLength {
                kind,
                expected,
                actual,
            } => write!(
                formatter,
                "V2 {kind:?} payload must contain {expected} bytes, got {actual}"
            ),
            Self::UnexpectedEnd => formatter.write_str("V2 payload ended before its typed fields"),
            Self::TrailingBytes { remaining } => {
                write!(formatter, "V2 payload contains {remaining} trailing bytes")
            }
            Self::UnknownEnum { field, value } => {
                write!(formatter, "unknown V2 {field} value {value}")
            }
            Self::Domain(source) => write!(formatter, "invalid V2 payload domain: {source}"),
            Self::Invariant { detail } => {
                write!(formatter, "V2 payload invariant failed: {detail}")
            }
        }
    }
}

impl core::error::Error for PayloadError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Domain(source) => Some(source),
            _ => None,
        }
    }
}

impl From<DomainError> for PayloadError {
    fn from(value: DomainError) -> Self {
        Self::Domain(value)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameError {
    TooShort {
        actual: usize,
        minimum: usize,
    },
    TooLong {
        actual: usize,
        maximum: usize,
    },
    BadMagic {
        actual: [u8; 4],
    },
    UnsupportedVersion {
        actual: u8,
    },
    UnknownMessageKind {
        value: u8,
    },
    NonzeroReserved {
        value: u8,
    },
    PayloadLengthAboveMaximum {
        actual: usize,
        maximum: usize,
    },
    LengthMismatch {
        declared_payload: usize,
        expected_total: usize,
        actual_total: usize,
    },
    CrcMismatch {
        expected: u32,
        actual: u32,
    },
    Payload(PayloadError),
}

impl core::fmt::Display for FrameError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::TooShort { actual, minimum } => {
                write!(
                    formatter,
                    "V2 frame has {actual} bytes, fewer than {minimum}"
                )
            }
            Self::TooLong { actual, maximum } => {
                write!(
                    formatter,
                    "V2 frame has {actual} bytes, exceeding {maximum}"
                )
            }
            Self::BadMagic { actual } => write!(
                formatter,
                "V2 frame magic is {:02x?}, expected {:02x?}",
                actual, MAGIC
            ),
            Self::UnsupportedVersion { actual } => {
                write!(formatter, "unsupported robot-protocol version {actual}")
            }
            Self::UnknownMessageKind { value } => {
                write!(formatter, "unknown V2 message kind 0x{value:02x}")
            }
            Self::NonzeroReserved { value } => {
                write!(
                    formatter,
                    "V2 frame reserved byte must be zero, got {value}"
                )
            }
            Self::PayloadLengthAboveMaximum { actual, maximum } => write!(
                formatter,
                "V2 payload length {actual} exceeds {maximum} bytes"
            ),
            Self::LengthMismatch {
                declared_payload,
                expected_total,
                actual_total,
            } => write!(
                formatter,
                "V2 frame declares {declared_payload} payload bytes and therefore {expected_total} total bytes, got {actual_total}"
            ),
            Self::CrcMismatch { expected, actual } => write!(
                formatter,
                "V2 CRC-32C mismatch: frame has 0x{actual:08x}, computed 0x{expected:08x}"
            ),
            Self::Payload(source) => write!(formatter, "invalid V2 payload: {source}"),
        }
    }
}

impl core::error::Error for FrameError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Payload(source) => Some(source),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EncodeError {
    PayloadOverflow,
    PayloadInvariant(PayloadError),
}

impl core::fmt::Display for EncodeError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::PayloadOverflow => {
                formatter.write_str("typed V2 payload exceeded its fixed buffer")
            }
            Self::PayloadInvariant(source) => {
                write!(formatter, "cannot encode invalid V2 payload: {source}")
            }
        }
    }
}

impl core::error::Error for EncodeError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::PayloadInvariant(source) => Some(source),
            Self::PayloadOverflow => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RawFrame {
    bytes: [u8; MAX_RAW_FRAME_BYTES],
    len: u8,
}

impl RawFrame {
    pub fn encode(message: Message) -> Result<Self, EncodeError> {
        validate_message(message).map_err(EncodeError::PayloadInvariant)?;

        let mut payload = PayloadWriter::new();
        encode_payload(message, &mut payload)?;
        if payload.len() != message.payload_len() {
            return Err(EncodeError::PayloadInvariant(PayloadError::Invariant {
                detail: "encoder length disagrees with the frozen message schema",
            }));
        }

        let mut bytes = [0_u8; MAX_RAW_FRAME_BYTES];
        bytes[..4].copy_from_slice(&MAGIC);
        bytes[4] = VERSION;
        bytes[5] = message.kind() as u8;
        bytes[6] = u8::try_from(payload.len()).map_err(|_| EncodeError::PayloadOverflow)?;
        bytes[7] = 0;
        let payload_end = HEADER_BYTES + payload.len();
        bytes[HEADER_BYTES..payload_end].copy_from_slice(payload.as_bytes());
        let checksum = crc32c(&bytes[..payload_end]).to_le_bytes();
        bytes[payload_end..payload_end + CRC_BYTES].copy_from_slice(&checksum);
        let len = payload_end + CRC_BYTES;
        Ok(Self {
            bytes,
            len: u8::try_from(len).map_err(|_| EncodeError::PayloadOverflow)?,
        })
    }

    pub const fn len(&self) -> usize {
        self.len as usize
    }

    pub const fn is_empty(&self) -> bool {
        false
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes[..self.len()]
    }

    pub fn decode(&self) -> Result<Message, FrameError> {
        decode_raw_frame(self.as_bytes())
    }
}

pub fn decode_raw_frame(bytes: &[u8]) -> Result<Message, FrameError> {
    let minimum = HEADER_BYTES + CRC_BYTES;
    if bytes.len() < minimum {
        return Err(FrameError::TooShort {
            actual: bytes.len(),
            minimum,
        });
    }
    if bytes.len() > MAX_RAW_FRAME_BYTES {
        return Err(FrameError::TooLong {
            actual: bytes.len(),
            maximum: MAX_RAW_FRAME_BYTES,
        });
    }

    let actual_magic = [bytes[0], bytes[1], bytes[2], bytes[3]];
    if actual_magic != MAGIC {
        return Err(FrameError::BadMagic {
            actual: actual_magic,
        });
    }
    if bytes[4] != VERSION {
        return Err(FrameError::UnsupportedVersion { actual: bytes[4] });
    }
    let kind = MessageKind::parse(bytes[5])?;
    let declared_payload = usize::from(bytes[6]);
    if declared_payload > MAX_PAYLOAD_BYTES {
        return Err(FrameError::PayloadLengthAboveMaximum {
            actual: declared_payload,
            maximum: MAX_PAYLOAD_BYTES,
        });
    }
    if bytes[7] != 0 {
        return Err(FrameError::NonzeroReserved { value: bytes[7] });
    }
    let expected_total = HEADER_BYTES + declared_payload + CRC_BYTES;
    if bytes.len() != expected_total {
        return Err(FrameError::LengthMismatch {
            declared_payload,
            expected_total,
            actual_total: bytes.len(),
        });
    }

    let checksum_offset = HEADER_BYTES + declared_payload;
    let expected_crc = crc32c(&bytes[..checksum_offset]);
    let actual_crc = u32::from_le_bytes([
        bytes[checksum_offset],
        bytes[checksum_offset + 1],
        bytes[checksum_offset + 2],
        bytes[checksum_offset + 3],
    ]);
    if actual_crc != expected_crc {
        return Err(FrameError::CrcMismatch {
            expected: expected_crc,
            actual: actual_crc,
        });
    }

    let expected_payload = expected_payload_len(kind);
    if declared_payload != expected_payload {
        return Err(FrameError::Payload(PayloadError::WrongLength {
            kind,
            expected: expected_payload,
            actual: declared_payload,
        }));
    }
    let message =
        decode_payload(kind, &bytes[HEADER_BYTES..checksum_offset]).map_err(FrameError::Payload)?;
    validate_message(message).map_err(FrameError::Payload)?;
    Ok(message)
}

/// Dependency-free CRC-32C (Castagnoli).
///
/// This is the reflected form with polynomial `0x82f63b78`, initial value
/// `0xffffffff`, and final XOR `0xffffffff`.
pub fn crc32c(bytes: &[u8]) -> u32 {
    let mut crc = u32::MAX;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            let mask = 0_u32.wrapping_sub(crc & 1);
            crc = (crc >> 1) ^ (0x82f6_3b78 & mask);
        }
    }
    !crc
}

fn expected_payload_len(kind: MessageKind) -> usize {
    match kind {
        MessageKind::AcquireControl => AcquireControl::PAYLOAD_BYTES,
        MessageKind::HostCommand => HostCommand::PAYLOAD_BYTES,
        MessageKind::HostStop => HostStop::PAYLOAD_BYTES,
        MessageKind::StatusQuery => StatusQuery::PAYLOAD_BYTES,
        MessageKind::ControllerHello => ControllerHello::PAYLOAD_BYTES,
        MessageKind::BeginSession => BeginSession::PAYLOAD_BYTES,
        MessageKind::ControllerReady => ControllerReady::PAYLOAD_BYTES,
        MessageKind::ApplyPwm => ApplyPwm::PAYLOAD_BYTES,
        MessageKind::ForceStop => ForceStop::PAYLOAD_BYTES,
        MessageKind::AppliedResult => AppliedResult::PAYLOAD_BYTES,
        MessageKind::Heartbeat => Heartbeat::PAYLOAD_BYTES,
        MessageKind::ObservationalOdometry => ObservationalOdometry::PAYLOAD_BYTES,
        MessageKind::AcquireResult => AcquireResult::PAYLOAD_BYTES,
        MessageKind::HostCommandResult => HostCommandResult::PAYLOAD_BYTES,
        MessageKind::HostStopResult => HostStopResult::PAYLOAD_BYTES,
        MessageKind::StatusReport => StatusReport::PAYLOAD_BYTES,
    }
}

fn validate_output_matches_pwm(output: OutputState, pwm: TimerPwm) -> Result<(), PayloadError> {
    match (output, pwm.is_zero()) {
        (OutputState::NonzeroPwm, false) | (OutputState::Disabled | OutputState::ZeroPwm, true) => {
            Ok(())
        }
        _ => Err(PayloadError::Invariant {
            detail: "output state and timer PWM disagree",
        }),
    }
}

fn validate_message(message: Message) -> Result<(), PayloadError> {
    match message {
        Message::ControllerHello(value) => {
            if !value.output_state.is_safe() {
                return Err(PayloadError::Invariant {
                    detail: "controller hello must report safe outputs",
                });
            }
        }
        Message::ControllerReady(value) => {
            if !value.output_state.is_safe()
                || !value.capabilities.supports_required_safety()
                || !value.faults.is_clear()
            {
                return Err(PayloadError::Invariant {
                    detail: "controller ready requires safe outputs, all safety capabilities, and no faults",
                });
            }
        }
        Message::Heartbeat(value) => {
            if value.control_epoch.is_some() != value.last_sequence.is_some() {
                return Err(PayloadError::Invariant {
                    detail: "heartbeat epoch and last sequence must be present together",
                });
            }
            validate_output_matches_pwm(value.output_state, value.timer_pwm)?;
        }
        Message::AppliedResult(value) => {
            validate_output_matches_pwm(value.output_state, value.timer_pwm)?;
            if !value.result.proves_applied() && !value.timer_pwm.is_zero() {
                return Err(PayloadError::Invariant {
                    detail: "a rejected applied result must report zero timer PWM",
                });
            }
        }
        Message::AcquireResult(value) => {
            if (value.result == AcquireResultCode::Granted) != value.control_epoch.is_some() {
                return Err(PayloadError::Invariant {
                    detail: "only a granted acquire result carries a control epoch",
                });
            }
        }
        Message::HostCommandResult(value) => {
            validate_output_matches_pwm(value.output_state, value.controller_timer_pwm)?;
            if !value.result.proves_controller_application()
                && !value.controller_timer_pwm.is_zero()
            {
                return Err(PayloadError::Invariant {
                    detail: "an unconfirmed host result must report zero controller timer PWM",
                });
            }
        }
        Message::HostStopResult(value) => {
            if value.result.proves_controller_stop() && !value.output_state.is_safe() {
                return Err(PayloadError::Invariant {
                    detail: "a confirmed stop must report safe outputs",
                });
            }
        }
        Message::StatusReport(value) => {
            validate_output_matches_pwm(value.output_state, value.controller_timer_pwm)?;
        }
        Message::AcquireControl(_)
        | Message::HostCommand(_)
        | Message::HostStop(_)
        | Message::StatusQuery(_)
        | Message::BeginSession(_)
        | Message::ApplyPwm(_)
        | Message::ForceStop(_)
        | Message::ObservationalOdometry(_) => {}
    }
    Ok(())
}

struct PayloadWriter {
    bytes: [u8; MAX_PAYLOAD_BYTES],
    len: usize,
}

impl PayloadWriter {
    const fn new() -> Self {
        Self {
            bytes: [0; MAX_PAYLOAD_BYTES],
            len: 0,
        }
    }

    const fn len(&self) -> usize {
        self.len
    }

    fn as_bytes(&self) -> &[u8] {
        &self.bytes[..self.len]
    }

    fn write_bytes(&mut self, value: &[u8]) -> Result<(), EncodeError> {
        let end = self
            .len
            .checked_add(value.len())
            .ok_or(EncodeError::PayloadOverflow)?;
        let destination = self
            .bytes
            .get_mut(self.len..end)
            .ok_or(EncodeError::PayloadOverflow)?;
        destination.copy_from_slice(value);
        self.len = end;
        Ok(())
    }

    fn u8(&mut self, value: u8) -> Result<(), EncodeError> {
        self.write_bytes(&[value])
    }

    fn i8(&mut self, value: i8) -> Result<(), EncodeError> {
        self.write_bytes(&value.to_le_bytes())
    }

    fn u16(&mut self, value: u16) -> Result<(), EncodeError> {
        self.write_bytes(&value.to_le_bytes())
    }

    fn i16(&mut self, value: i16) -> Result<(), EncodeError> {
        self.write_bytes(&value.to_le_bytes())
    }

    fn u32(&mut self, value: u32) -> Result<(), EncodeError> {
        self.write_bytes(&value.to_le_bytes())
    }

    fn u64(&mut self, value: u64) -> Result<(), EncodeError> {
        self.write_bytes(&value.to_le_bytes())
    }

    fn i64(&mut self, value: i64) -> Result<(), EncodeError> {
        self.write_bytes(&value.to_le_bytes())
    }

    fn uid(&mut self, value: ControllerUid) -> Result<(), EncodeError> {
        self.write_bytes(value.as_bytes())
    }

    fn fingerprint(&mut self, value: ActuatorConfigFingerprint) -> Result<(), EncodeError> {
        self.write_bytes(value.as_bytes())
    }

    fn boot(&mut self, value: ControllerBootId) -> Result<(), EncodeError> {
        self.u64(value.get())
    }

    fn target_boot(&mut self, value: TargetBootId) -> Result<(), EncodeError> {
        self.u64(value.to_wire())
    }

    fn epoch(&mut self, value: ControlEpoch) -> Result<(), EncodeError> {
        self.u32(value.get())
    }

    fn optional_epoch(&mut self, value: Option<ControlEpoch>) -> Result<(), EncodeError> {
        self.u32(value.map_or(0, ControlEpoch::get))
    }

    fn pwm(&mut self, value: TimerPwm) -> Result<(), EncodeError> {
        self.i8(value.left().get())?;
        self.i8(value.right().get())
    }
}

struct PayloadReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> PayloadReader<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take<const N: usize>(&mut self) -> Result<[u8; N], PayloadError> {
        let end = self
            .offset
            .checked_add(N)
            .ok_or(PayloadError::UnexpectedEnd)?;
        let source = self
            .bytes
            .get(self.offset..end)
            .ok_or(PayloadError::UnexpectedEnd)?;
        let mut result = [0; N];
        result.copy_from_slice(source);
        self.offset = end;
        Ok(result)
    }

    fn finish(self) -> Result<(), PayloadError> {
        let remaining = self.bytes.len() - self.offset;
        if remaining == 0 {
            Ok(())
        } else {
            Err(PayloadError::TrailingBytes { remaining })
        }
    }

    fn u8(&mut self) -> Result<u8, PayloadError> {
        Ok(self.take::<1>()?[0])
    }

    fn i8(&mut self) -> Result<i8, PayloadError> {
        Ok(i8::from_le_bytes(self.take()?))
    }

    fn u16(&mut self) -> Result<u16, PayloadError> {
        Ok(u16::from_le_bytes(self.take()?))
    }

    fn i16(&mut self) -> Result<i16, PayloadError> {
        Ok(i16::from_le_bytes(self.take()?))
    }

    fn u32(&mut self) -> Result<u32, PayloadError> {
        Ok(u32::from_le_bytes(self.take()?))
    }

    fn u64(&mut self) -> Result<u64, PayloadError> {
        Ok(u64::from_le_bytes(self.take()?))
    }

    fn i64(&mut self) -> Result<i64, PayloadError> {
        Ok(i64::from_le_bytes(self.take()?))
    }

    fn uid(&mut self) -> Result<ControllerUid, PayloadError> {
        ControllerUid::try_new(self.take()?).map_err(Into::into)
    }

    fn fingerprint(&mut self) -> Result<ActuatorConfigFingerprint, PayloadError> {
        ActuatorConfigFingerprint::try_new(self.take()?).map_err(Into::into)
    }

    fn boot(&mut self) -> Result<ControllerBootId, PayloadError> {
        ControllerBootId::try_new(self.u64()?).map_err(Into::into)
    }

    fn target_boot(&mut self) -> Result<TargetBootId, PayloadError> {
        TargetBootId::from_wire(self.u64()?).map_err(Into::into)
    }

    fn epoch(&mut self) -> Result<ControlEpoch, PayloadError> {
        ControlEpoch::try_new(self.u32()?).map_err(Into::into)
    }

    fn optional_epoch(&mut self) -> Result<Option<ControlEpoch>, PayloadError> {
        let value = self.u32()?;
        if value == 0 {
            Ok(None)
        } else {
            ControlEpoch::try_new(value).map(Some).map_err(Into::into)
        }
    }

    fn pwm(&mut self) -> Result<TimerPwm, PayloadError> {
        TimerPwm::try_new(self.i8()?, self.i8()?).map_err(Into::into)
    }
}

fn encode_payload(message: Message, writer: &mut PayloadWriter) -> Result<(), EncodeError> {
    match message {
        Message::ControllerHello(value) => {
            writer.uid(value.controller_uid)?;
            writer.boot(value.boot_id)?;
            writer.u16(value.firmware_abi)?;
            writer.u32(value.firmware_build_id)?;
            writer.u32(value.capabilities.bits())?;
            writer.u8(value.max_abs_pwm_percent.get())?;
            writer.u16(value.max_command_lease.get())?;
            writer.u8(value.output_state as u8)?;
            writer.fingerprint(value.actuator_config_fingerprint)?;
            writer.u16(value.watchdog_nominal_period.get())?;
            writer.u16(value.pwm_frequency.get())?;
            writer.u8(value.neutral_output as u8)?;
            writer.u8(value.physical_stop_semantics as u8)?;
        }
        Message::BeginSession(value) => {
            writer.uid(value.controller_uid)?;
            writer.boot(value.boot_id)?;
            writer.u32(value.request_id.get())?;
            writer.u16(value.heartbeat_period.get())?;
        }
        Message::ControllerReady(value) => {
            writer.uid(value.controller_uid)?;
            writer.boot(value.boot_id)?;
            writer.epoch(value.control_epoch)?;
            writer.u32(value.controller_uptime.get())?;
            writer.u32(value.capabilities.bits())?;
            writer.u8(value.output_state as u8)?;
            writer.u32(value.faults.bits())?;
        }
        Message::ApplyPwm(value) => {
            writer.uid(value.controller_uid)?;
            writer.boot(value.boot_id)?;
            writer.epoch(value.control_epoch)?;
            writer.u32(value.sequence.get())?;
            writer.u32(value.expires_at.get())?;
            writer.pwm(value.timer_pwm)?;
        }
        Message::ForceStop(value) => {
            writer.uid(value.controller_uid)?;
            writer.target_boot(value.target_boot_id)?;
            writer.u32(value.request_id.get())?;
            writer.u8(value.reason as u8)?;
        }
        Message::AppliedResult(value) => {
            writer.uid(value.controller_uid)?;
            writer.boot(value.boot_id)?;
            writer.epoch(value.control_epoch)?;
            writer.u32(value.sequence.get())?;
            writer.u8(value.result as u8)?;
            writer.pwm(value.timer_pwm)?;
            writer.u8(value.output_state as u8)?;
            writer.u32(value.applied_at.get())?;
            writer.u32(value.expires_at.get())?;
            writer.u32(value.faults.bits())?;
        }
        Message::Heartbeat(value) => {
            writer.uid(value.controller_uid)?;
            writer.boot(value.boot_id)?;
            writer.optional_epoch(value.control_epoch)?;
            writer.u32(value.controller_uptime.get())?;
            writer.u32(value.expires_at.get())?;
            writer.pwm(value.timer_pwm)?;
            writer.u8(value.output_state as u8)?;
            writer.u16(value.readiness.bits())?;
            writer.u32(value.faults.bits())?;
            writer.u32(value.last_sequence.map_or(0, V2CommandSequence::get))?;
        }
        Message::ObservationalOdometry(value) => {
            writer.uid(value.controller_uid)?;
            writer.boot(value.boot_id)?;
            writer.optional_epoch(value.control_epoch)?;
            writer.i64(value.left_estimated_extended_ticks_wrapping.get())?;
            writer.i64(value.right_estimated_extended_ticks_wrapping.get())?;
            writer.i16(value.left_sample_delta_ticks_modulo.get())?;
            writer.i16(value.right_sample_delta_ticks_modulo.get())?;
            writer.u32(value.controller_uptime.get())?;
        }
        Message::AcquireControl(value) => {
            writer.uid(value.expected_controller_uid)?;
            writer.boot(value.expected_boot_id)?;
            writer.u32(value.request_id.get())?;
            writer.u16(value.expected_firmware_abi)?;
            writer.u32(value.expected_firmware_build_id)?;
            writer.fingerprint(value.expected_actuator_config_fingerprint)?;
        }
        Message::AcquireResult(value) => {
            writer.uid(value.controller_uid)?;
            writer.boot(value.boot_id)?;
            writer.u32(value.request_id.get())?;
            writer.optional_epoch(value.control_epoch)?;
            writer.u8(value.result as u8)?;
            writer.u32(value.capabilities.bits())?;
            writer.u32(value.faults.bits())?;
            writer.u16(value.observed_firmware_abi)?;
            writer.u32(value.observed_firmware_build_id)?;
            writer.fingerprint(value.observed_actuator_config_fingerprint)?;
        }
        Message::HostCommand(value) => {
            writer.uid(value.controller_uid)?;
            writer.boot(value.boot_id)?;
            writer.epoch(value.control_epoch)?;
            writer.u32(value.sequence.get())?;
            writer.u16(value.lease.get())?;
            writer.pwm(value.requested_timer_pwm)?;
        }
        Message::HostCommandResult(value) => {
            writer.uid(value.controller_uid)?;
            writer.boot(value.boot_id)?;
            writer.epoch(value.control_epoch)?;
            writer.u32(value.sequence.get())?;
            writer.u8(value.result as u8)?;
            writer.pwm(value.requested_timer_pwm)?;
            writer.pwm(value.controller_timer_pwm)?;
            writer.u8(value.output_state as u8)?;
            writer.u32(value.controller_applied_at.get())?;
            writer.u32(value.controller_expires_at.get())?;
            writer.u16(value.remaining_lease.get())?;
            writer.u32(value.faults.bits())?;
        }
        Message::HostStop(value) => {
            writer.uid(value.controller_uid)?;
            writer.target_boot(value.target_boot_id)?;
            writer.u32(value.request_id.get())?;
            writer.u8(value.reason as u8)?;
        }
        Message::HostStopResult(value) => {
            writer.uid(value.controller_uid)?;
            writer.target_boot(value.observed_boot_id)?;
            writer.u32(value.request_id.get())?;
            writer.u8(value.result as u8)?;
            writer.u8(value.output_state as u8)?;
            writer.u32(value.controller_uptime.get())?;
            writer.u32(value.faults.bits())?;
        }
        Message::StatusQuery(value) => {
            writer.uid(value.expected_controller_uid)?;
            writer.u32(value.request_id.get())?;
        }
        Message::StatusReport(value) => {
            writer.uid(value.controller_uid)?;
            writer.target_boot(value.observed_boot_id)?;
            writer.u32(value.request_id.get())?;
            writer.u8(value.status as u8)?;
            writer.optional_epoch(value.control_epoch)?;
            writer.u32(value.controller_uptime.get())?;
            writer.u32(value.capabilities.bits())?;
            writer.u8(value.output_state as u8)?;
            writer.pwm(value.controller_timer_pwm)?;
            writer.u16(value.remaining_lease.get())?;
            writer.u32(value.faults.bits())?;
        }
    }
    Ok(())
}

fn decode_payload(kind: MessageKind, bytes: &[u8]) -> Result<Message, PayloadError> {
    let mut reader = PayloadReader::new(bytes);
    let message = match kind {
        MessageKind::ControllerHello => Message::ControllerHello(ControllerHello {
            controller_uid: reader.uid()?,
            boot_id: reader.boot()?,
            firmware_abi: reader.u16()?,
            firmware_build_id: reader.u32()?,
            capabilities: ControllerCapabilities::try_from_bits(reader.u32()?)?,
            max_abs_pwm_percent: MaxAbsPwmPercent::try_new(reader.u8()?)?,
            max_command_lease: V2CommandLeaseMs::try_new(reader.u16()?)?,
            output_state: OutputState::parse(reader.u8()?)?,
            actuator_config_fingerprint: reader.fingerprint()?,
            watchdog_nominal_period: WatchdogNominalPeriodMs::try_new(reader.u16()?)?,
            pwm_frequency: PwmFrequencyHz::try_new(reader.u16()?)?,
            neutral_output: NeutralOutput::parse(reader.u8()?)?,
            physical_stop_semantics: PhysicalStopSemantics::parse(reader.u8()?)?,
        }),
        MessageKind::BeginSession => Message::BeginSession(BeginSession {
            controller_uid: reader.uid()?,
            boot_id: reader.boot()?,
            request_id: RequestId::new(reader.u32()?),
            heartbeat_period: HeartbeatPeriodMs::try_new(reader.u16()?)?,
        }),
        MessageKind::ControllerReady => Message::ControllerReady(ControllerReady {
            controller_uid: reader.uid()?,
            boot_id: reader.boot()?,
            control_epoch: reader.epoch()?,
            controller_uptime: ControllerUptimeMsWrapping::new(reader.u32()?),
            capabilities: ControllerCapabilities::try_from_bits(reader.u32()?)?,
            output_state: OutputState::parse(reader.u8()?)?,
            faults: ControllerFaults::try_from_bits(reader.u32()?)?,
        }),
        MessageKind::ApplyPwm => Message::ApplyPwm(ApplyPwm {
            controller_uid: reader.uid()?,
            boot_id: reader.boot()?,
            control_epoch: reader.epoch()?,
            sequence: V2CommandSequence::new(reader.u32()?),
            expires_at: ControllerDeadlineMsWrapping::new(reader.u32()?),
            timer_pwm: reader.pwm()?,
        }),
        MessageKind::ForceStop => Message::ForceStop(ForceStop {
            controller_uid: reader.uid()?,
            target_boot_id: reader.target_boot()?,
            request_id: RequestId::new(reader.u32()?),
            reason: ForceStopReason::parse(reader.u8()?)?,
        }),
        MessageKind::AppliedResult => Message::AppliedResult(AppliedResult {
            controller_uid: reader.uid()?,
            boot_id: reader.boot()?,
            control_epoch: reader.epoch()?,
            sequence: V2CommandSequence::new(reader.u32()?),
            result: AppliedResultCode::parse(reader.u8()?)?,
            timer_pwm: reader.pwm()?,
            output_state: OutputState::parse(reader.u8()?)?,
            applied_at: ControllerUptimeMsWrapping::new(reader.u32()?),
            expires_at: ControllerDeadlineMsWrapping::new(reader.u32()?),
            faults: ControllerFaults::try_from_bits(reader.u32()?)?,
        }),
        MessageKind::Heartbeat => {
            let controller_uid = reader.uid()?;
            let boot_id = reader.boot()?;
            let control_epoch = reader.optional_epoch()?;
            let controller_uptime = ControllerUptimeMsWrapping::new(reader.u32()?);
            let expires_at = ControllerDeadlineMsWrapping::new(reader.u32()?);
            let timer_pwm = reader.pwm()?;
            let output_state = OutputState::parse(reader.u8()?)?;
            let readiness = ReadinessFlags::try_from_bits(reader.u16()?)?;
            let faults = ControllerFaults::try_from_bits(reader.u32()?)?;
            let sequence_wire = reader.u32()?;
            let last_sequence = if control_epoch.is_some() {
                Some(V2CommandSequence::new(sequence_wire))
            } else if sequence_wire == 0 {
                None
            } else {
                return Err(PayloadError::Invariant {
                    detail: "heartbeat without an epoch must encode zero last sequence",
                });
            };
            Message::Heartbeat(Heartbeat {
                controller_uid,
                boot_id,
                control_epoch,
                last_sequence,
                controller_uptime,
                expires_at,
                timer_pwm,
                output_state,
                readiness,
                faults,
            })
        }
        MessageKind::ObservationalOdometry => {
            Message::ObservationalOdometry(ObservationalOdometry {
                controller_uid: reader.uid()?,
                boot_id: reader.boot()?,
                control_epoch: reader.optional_epoch()?,
                left_estimated_extended_ticks_wrapping: EstimatedWrappingEncoderTicks::new_wrapping(
                    reader.i64()?,
                ),
                right_estimated_extended_ticks_wrapping:
                    EstimatedWrappingEncoderTicks::new_wrapping(reader.i64()?),
                left_sample_delta_ticks_modulo: ModuloEncoderDeltaTicks::new_modulo(reader.i16()?),
                right_sample_delta_ticks_modulo: ModuloEncoderDeltaTicks::new_modulo(reader.i16()?),
                controller_uptime: ControllerUptimeMsWrapping::new(reader.u32()?),
            })
        }
        MessageKind::AcquireControl => Message::AcquireControl(AcquireControl {
            expected_controller_uid: reader.uid()?,
            expected_boot_id: reader.boot()?,
            request_id: RequestId::new(reader.u32()?),
            expected_firmware_abi: reader.u16()?,
            expected_firmware_build_id: reader.u32()?,
            expected_actuator_config_fingerprint: reader.fingerprint()?,
        }),
        MessageKind::AcquireResult => Message::AcquireResult(AcquireResult {
            controller_uid: reader.uid()?,
            boot_id: reader.boot()?,
            request_id: RequestId::new(reader.u32()?),
            control_epoch: reader.optional_epoch()?,
            result: AcquireResultCode::parse(reader.u8()?)?,
            capabilities: ControllerCapabilities::try_from_bits(reader.u32()?)?,
            faults: ControllerFaults::try_from_bits(reader.u32()?)?,
            observed_firmware_abi: reader.u16()?,
            observed_firmware_build_id: reader.u32()?,
            observed_actuator_config_fingerprint: reader.fingerprint()?,
        }),
        MessageKind::HostCommand => Message::HostCommand(HostCommand {
            controller_uid: reader.uid()?,
            boot_id: reader.boot()?,
            control_epoch: reader.epoch()?,
            sequence: V2CommandSequence::new(reader.u32()?),
            lease: V2CommandLeaseMs::try_new(reader.u16()?)?,
            requested_timer_pwm: reader.pwm()?,
        }),
        MessageKind::HostCommandResult => Message::HostCommandResult(HostCommandResult {
            controller_uid: reader.uid()?,
            boot_id: reader.boot()?,
            control_epoch: reader.epoch()?,
            sequence: V2CommandSequence::new(reader.u32()?),
            result: HostCommandResultCode::parse(reader.u8()?)?,
            requested_timer_pwm: reader.pwm()?,
            controller_timer_pwm: reader.pwm()?,
            output_state: OutputState::parse(reader.u8()?)?,
            controller_applied_at: ControllerUptimeMsWrapping::new(reader.u32()?),
            controller_expires_at: ControllerDeadlineMsWrapping::new(reader.u32()?),
            remaining_lease: RemainingLeaseMs::try_new(reader.u16()?)?,
            faults: ControllerFaults::try_from_bits(reader.u32()?)?,
        }),
        MessageKind::HostStop => Message::HostStop(HostStop {
            controller_uid: reader.uid()?,
            target_boot_id: reader.target_boot()?,
            request_id: RequestId::new(reader.u32()?),
            reason: ForceStopReason::parse(reader.u8()?)?,
        }),
        MessageKind::HostStopResult => Message::HostStopResult(HostStopResult {
            controller_uid: reader.uid()?,
            observed_boot_id: reader.target_boot()?,
            request_id: RequestId::new(reader.u32()?),
            result: StopResultCode::parse(reader.u8()?)?,
            output_state: OutputState::parse(reader.u8()?)?,
            controller_uptime: ControllerUptimeMsWrapping::new(reader.u32()?),
            faults: ControllerFaults::try_from_bits(reader.u32()?)?,
        }),
        MessageKind::StatusQuery => Message::StatusQuery(StatusQuery {
            expected_controller_uid: reader.uid()?,
            request_id: RequestId::new(reader.u32()?),
        }),
        MessageKind::StatusReport => Message::StatusReport(StatusReport {
            controller_uid: reader.uid()?,
            observed_boot_id: reader.target_boot()?,
            request_id: RequestId::new(reader.u32()?),
            status: StatusCode::parse(reader.u8()?)?,
            control_epoch: reader.optional_epoch()?,
            controller_uptime: ControllerUptimeMsWrapping::new(reader.u32()?),
            capabilities: ControllerCapabilities::try_from_bits(reader.u32()?)?,
            output_state: OutputState::parse(reader.u8()?)?,
            controller_timer_pwm: reader.pwm()?,
            remaining_lease: RemainingLeaseMs::try_new(reader.u16()?)?,
            faults: ControllerFaults::try_from_bits(reader.u32()?)?,
        }),
    };
    reader.finish()?;
    Ok(message)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CobsError {
    OutputTooSmall,
    ZeroCode,
    TruncatedBlock,
}

impl core::fmt::Display for CobsError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::OutputTooSmall => formatter.write_str("COBS output buffer is too small"),
            Self::ZeroCode => formatter.write_str("COBS record contains an illegal zero code"),
            Self::TruncatedBlock => formatter.write_str("COBS record ends inside a block"),
        }
    }
}

impl core::error::Error for CobsError {}

/// Encodes one delimiter-free COBS record and returns its exact length.
pub fn cobs_encode(input: &[u8], output: &mut [u8]) -> Result<usize, CobsError> {
    if output.is_empty() {
        return Err(CobsError::OutputTooSmall);
    }

    let mut read: usize = 0;
    let mut write: usize = 1;
    let mut code_index: usize = 0;
    let mut code = 1_u8;

    while read < input.len() {
        let byte = input[read];
        read += 1;
        if byte == 0 {
            *output
                .get_mut(code_index)
                .ok_or(CobsError::OutputTooSmall)? = code;
            code_index = write;
            write = write.checked_add(1).ok_or(CobsError::OutputTooSmall)?;
            if write > output.len() {
                return Err(CobsError::OutputTooSmall);
            }
            code = 1;
        } else {
            *output.get_mut(write).ok_or(CobsError::OutputTooSmall)? = byte;
            write += 1;
            code = code.wrapping_add(1);
            if code == u8::MAX {
                *output
                    .get_mut(code_index)
                    .ok_or(CobsError::OutputTooSmall)? = code;
                if read == input.len() {
                    return Ok(write);
                }
                code_index = write;
                write = write.checked_add(1).ok_or(CobsError::OutputTooSmall)?;
                if write > output.len() {
                    return Err(CobsError::OutputTooSmall);
                }
                code = 1;
            }
        }
    }

    *output
        .get_mut(code_index)
        .ok_or(CobsError::OutputTooSmall)? = code;
    Ok(write)
}

/// Decodes one COBS record without its zero delimiter.
pub fn cobs_decode(input: &[u8], output: &mut [u8]) -> Result<usize, CobsError> {
    let mut read: usize = 0;
    let mut write: usize = 0;
    while read < input.len() {
        let code = input[read];
        if code == 0 {
            return Err(CobsError::ZeroCode);
        }
        read += 1;
        let block_len = usize::from(code) - 1;
        let block_end = read
            .checked_add(block_len)
            .ok_or(CobsError::TruncatedBlock)?;
        let block = input
            .get(read..block_end)
            .ok_or(CobsError::TruncatedBlock)?;
        let output_end = write
            .checked_add(block_len)
            .ok_or(CobsError::OutputTooSmall)?;
        let destination = output
            .get_mut(write..output_end)
            .ok_or(CobsError::OutputTooSmall)?;
        destination.copy_from_slice(block);
        write = output_end;
        read = block_end;

        if code != u8::MAX && read < input.len() {
            *output.get_mut(write).ok_or(CobsError::OutputTooSmall)? = 0;
            write += 1;
        }
    }
    Ok(write)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UartRecord {
    bytes: [u8; MAX_UART_RECORD_BYTES],
    len: u8,
}

impl UartRecord {
    pub fn encode(message: Message) -> Result<Self, UartEncodeError> {
        let raw = RawFrame::encode(message).map_err(UartEncodeError::Frame)?;
        let mut bytes = [0_u8; MAX_UART_RECORD_BYTES];
        let encoded_len = cobs_encode(raw.as_bytes(), &mut bytes[..MAX_COBS_FRAME_BYTES])
            .map_err(UartEncodeError::Cobs)?;
        bytes[encoded_len] = 0;
        let len = encoded_len + 1;
        Ok(Self {
            bytes,
            len: u8::try_from(len).map_err(|_| UartEncodeError::LengthInvariant)?,
        })
    }

    pub const fn len(&self) -> usize {
        self.len as usize
    }

    pub const fn is_empty(&self) -> bool {
        false
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes[..self.len()]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UartEncodeError {
    Frame(EncodeError),
    Cobs(CobsError),
    LengthInvariant,
}

impl core::fmt::Display for UartEncodeError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Frame(source) => write!(formatter, "cannot encode V2 UART frame: {source}"),
            Self::Cobs(source) => write!(formatter, "cannot COBS-encode V2 UART frame: {source}"),
            Self::LengthInvariant => {
                formatter.write_str("V2 UART record length exceeded its u8 domain")
            }
        }
    }
}

impl core::error::Error for UartEncodeError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Frame(source) => Some(source),
            Self::Cobs(source) => Some(source),
            Self::LengthInvariant => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UartStreamError {
    EmptyRecord,
    OversizedRecord { maximum: usize },
    Cobs(CobsError),
    Frame(FrameError),
}

impl core::fmt::Display for UartStreamError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyRecord => formatter.write_str("empty V2 UART record"),
            Self::OversizedRecord { maximum } => {
                write!(formatter, "V2 UART COBS record exceeds {maximum} bytes")
            }
            Self::Cobs(source) => write!(formatter, "invalid V2 UART COBS record: {source}"),
            Self::Frame(source) => write!(formatter, "invalid V2 UART raw frame: {source}"),
        }
    }
}

impl core::error::Error for UartStreamError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Cobs(source) => Some(source),
            Self::Frame(source) => Some(source),
            Self::EmptyRecord | Self::OversizedRecord { .. } => None,
        }
    }
}

/// Allocation-free UART record decoder.
///
/// An oversize record reports once, discards bytes through the next zero
/// delimiter, and then accepts a fresh record. Every delimiter is therefore a
/// bounded resynchronization point.
#[derive(Clone, Debug)]
pub struct UartStreamDecoder {
    encoded: [u8; MAX_COBS_FRAME_BYTES],
    len: usize,
    discarding_oversized: bool,
}

impl Default for UartStreamDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl UartStreamDecoder {
    pub const fn new() -> Self {
        Self {
            encoded: [0; MAX_COBS_FRAME_BYTES],
            len: 0,
            discarding_oversized: false,
        }
    }

    pub const fn is_discarding_oversized_record(&self) -> bool {
        self.discarding_oversized
    }

    pub fn push(&mut self, byte: u8) -> Option<Result<Message, UartStreamError>> {
        if byte == 0 {
            if self.discarding_oversized {
                self.discarding_oversized = false;
                self.len = 0;
                return None;
            }
            if self.len == 0 {
                return Some(Err(UartStreamError::EmptyRecord));
            }

            let encoded_len = self.len;
            self.len = 0;
            let mut raw = [0_u8; MAX_RAW_FRAME_BYTES];
            let raw_len = match cobs_decode(&self.encoded[..encoded_len], &mut raw) {
                Ok(value) => value,
                Err(source) => return Some(Err(UartStreamError::Cobs(source))),
            };
            return Some(decode_raw_frame(&raw[..raw_len]).map_err(UartStreamError::Frame));
        }

        if self.discarding_oversized {
            return None;
        }
        if self.len == self.encoded.len() {
            self.len = 0;
            self.discarding_oversized = true;
            return Some(Err(UartStreamError::OversizedRecord {
                maximum: MAX_COBS_FRAME_BYTES,
            }));
        }
        self.encoded[self.len] = byte;
        self.len += 1;
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uid() -> ControllerUid {
        ControllerUid::try_new([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
            .expect("nonzero controller UID")
    }

    fn boot() -> ControllerBootId {
        ControllerBootId::try_new(0x0102_0304_0506_0708).expect("nonzero boot ID")
    }

    fn epoch() -> ControlEpoch {
        ControlEpoch::try_new(0x1122_3344).expect("nonzero epoch")
    }

    fn fingerprint() -> ActuatorConfigFingerprint {
        ActuatorConfigFingerprint::try_new([0xa5; 16]).expect("nonzero fingerprint")
    }

    fn capabilities() -> ControllerCapabilities {
        ControllerCapabilities::try_from_bits(ControllerCapabilities::REQUIRED_BITS)
            .expect("known capability bits")
    }

    fn readiness() -> ReadinessFlags {
        ReadinessFlags::try_from_bits(ReadinessFlags::READY_BITS).expect("known readiness bits")
    }

    fn moving_pwm() -> TimerPwm {
        TimerPwm::try_new(-25, 40).expect("valid timer PWM")
    }

    fn hello() -> ControllerHello {
        ControllerHello {
            controller_uid: uid(),
            boot_id: boot(),
            firmware_abi: 2,
            firmware_build_id: 0x89ab_cdef,
            capabilities: capabilities(),
            max_abs_pwm_percent: MaxAbsPwmPercent::try_new(100).expect("valid maximum PWM"),
            max_command_lease: V2CommandLeaseMs::try_new(MAX_V2_COMMAND_LEASE_MS)
                .expect("valid maximum lease"),
            output_state: OutputState::Disabled,
            actuator_config_fingerprint: fingerprint(),
            watchdog_nominal_period: WatchdogNominalPeriodMs::try_new(100)
                .expect("valid watchdog period"),
            pwm_frequency: PwmFrequencyHz::try_new(20_000).expect("valid PWM frequency"),
            neutral_output: NeutralOutput::BothLow,
            physical_stop_semantics: PhysicalStopSemantics::Unverified,
        }
    }

    fn messages() -> [Message; 16] {
        [
            Message::AcquireControl(AcquireControl {
                expected_controller_uid: uid(),
                expected_boot_id: boot(),
                request_id: RequestId::new(3),
                expected_firmware_abi: 2,
                expected_firmware_build_id: 0x89ab_cdef,
                expected_actuator_config_fingerprint: fingerprint(),
            }),
            Message::HostCommand(HostCommand {
                controller_uid: uid(),
                boot_id: boot(),
                control_epoch: epoch(),
                sequence: V2CommandSequence::new(7),
                lease: V2CommandLeaseMs::try_new(150).expect("valid lease"),
                requested_timer_pwm: moving_pwm(),
            }),
            Message::HostStop(HostStop {
                controller_uid: uid(),
                target_boot_id: TargetBootId::Exact(boot()),
                request_id: RequestId::new(4),
                reason: ForceStopReason::Operator,
            }),
            Message::StatusQuery(StatusQuery {
                expected_controller_uid: uid(),
                request_id: RequestId::new(5),
            }),
            Message::ControllerHello(hello()),
            Message::BeginSession(BeginSession {
                controller_uid: uid(),
                boot_id: boot(),
                request_id: RequestId::new(6),
                heartbeat_period: HeartbeatPeriodMs::try_new(20).expect("valid heartbeat period"),
            }),
            Message::ControllerReady(ControllerReady {
                controller_uid: uid(),
                boot_id: boot(),
                control_epoch: epoch(),
                controller_uptime: ControllerUptimeMsWrapping::new(1_000),
                capabilities: capabilities(),
                output_state: OutputState::ZeroPwm,
                faults: ControllerFaults::NONE,
            }),
            Message::ApplyPwm(ApplyPwm {
                controller_uid: uid(),
                boot_id: boot(),
                control_epoch: epoch(),
                sequence: V2CommandSequence::new(7),
                expires_at: ControllerDeadlineMsWrapping::new(2_000),
                timer_pwm: moving_pwm(),
            }),
            Message::ForceStop(ForceStop {
                controller_uid: uid(),
                target_boot_id: TargetBootId::Any,
                request_id: RequestId::new(7),
                reason: ForceStopReason::TransportFault,
            }),
            Message::AppliedResult(AppliedResult {
                controller_uid: uid(),
                boot_id: boot(),
                control_epoch: epoch(),
                sequence: V2CommandSequence::new(7),
                result: AppliedResultCode::AppliedNew,
                timer_pwm: moving_pwm(),
                output_state: OutputState::NonzeroPwm,
                applied_at: ControllerUptimeMsWrapping::new(1_900),
                expires_at: ControllerDeadlineMsWrapping::new(2_000),
                faults: ControllerFaults::NONE,
            }),
            Message::Heartbeat(Heartbeat {
                controller_uid: uid(),
                boot_id: boot(),
                control_epoch: Some(epoch()),
                last_sequence: Some(V2CommandSequence::new(7)),
                controller_uptime: ControllerUptimeMsWrapping::new(1_910),
                expires_at: ControllerDeadlineMsWrapping::new(2_000),
                timer_pwm: moving_pwm(),
                output_state: OutputState::NonzeroPwm,
                readiness: readiness(),
                faults: ControllerFaults::NONE,
            }),
            Message::ObservationalOdometry(ObservationalOdometry {
                controller_uid: uid(),
                boot_id: boot(),
                control_epoch: Some(epoch()),
                left_estimated_extended_ticks_wrapping: EstimatedWrappingEncoderTicks::new_wrapping(
                    -9_000_000_000,
                ),
                right_estimated_extended_ticks_wrapping:
                    EstimatedWrappingEncoderTicks::new_wrapping(9_000_000_000),
                left_sample_delta_ticks_modulo: ModuloEncoderDeltaTicks::new_modulo(-321),
                right_sample_delta_ticks_modulo: ModuloEncoderDeltaTicks::new_modulo(654),
                controller_uptime: ControllerUptimeMsWrapping::new(u32::MAX - 2),
            }),
            Message::AcquireResult(AcquireResult {
                controller_uid: uid(),
                boot_id: boot(),
                request_id: RequestId::new(3),
                control_epoch: Some(epoch()),
                result: AcquireResultCode::Granted,
                capabilities: capabilities(),
                faults: ControllerFaults::NONE,
                observed_firmware_abi: 2,
                observed_firmware_build_id: 0x89ab_cdef,
                observed_actuator_config_fingerprint: fingerprint(),
            }),
            Message::HostCommandResult(HostCommandResult {
                controller_uid: uid(),
                boot_id: boot(),
                control_epoch: epoch(),
                sequence: V2CommandSequence::new(7),
                result: HostCommandResultCode::AppliedNew,
                requested_timer_pwm: moving_pwm(),
                controller_timer_pwm: moving_pwm(),
                output_state: OutputState::NonzeroPwm,
                controller_applied_at: ControllerUptimeMsWrapping::new(1_900),
                controller_expires_at: ControllerDeadlineMsWrapping::new(2_000),
                remaining_lease: RemainingLeaseMs::try_new(80).expect("bounded lifetime"),
                faults: ControllerFaults::NONE,
            }),
            Message::HostStopResult(HostStopResult {
                controller_uid: uid(),
                observed_boot_id: TargetBootId::Exact(boot()),
                request_id: RequestId::new(4),
                result: StopResultCode::ControllerConfirmed,
                output_state: OutputState::ZeroPwm,
                controller_uptime: ControllerUptimeMsWrapping::new(2_010),
                faults: ControllerFaults::NONE,
            }),
            Message::StatusReport(StatusReport {
                controller_uid: uid(),
                observed_boot_id: TargetBootId::Exact(boot()),
                request_id: RequestId::new(5),
                status: StatusCode::ReadyActive,
                control_epoch: Some(epoch()),
                controller_uptime: ControllerUptimeMsWrapping::new(1_910),
                capabilities: capabilities(),
                output_state: OutputState::NonzeroPwm,
                controller_timer_pwm: moving_pwm(),
                remaining_lease: RemainingLeaseMs::try_new(80).expect("bounded lifetime"),
                faults: ControllerFaults::NONE,
            }),
        ]
    }

    fn replace_crc(bytes: &mut [u8]) {
        let checksum_offset = bytes.len() - CRC_BYTES;
        let checksum = crc32c(&bytes[..checksum_offset]).to_le_bytes();
        bytes[checksum_offset..].copy_from_slice(&checksum);
    }

    #[test]
    fn crc32c_matches_the_castagnoli_check_vector() {
        assert_eq!(crc32c(b"123456789"), 0xe306_9283);
        assert_eq!(crc32c(b""), 0);
    }

    #[test]
    fn every_frozen_message_round_trips_with_its_exact_length() {
        for message in messages() {
            assert!(message.payload_len() <= MAX_PAYLOAD_BYTES);
            let frame = RawFrame::encode(message).expect("valid typed message encodes");
            assert_eq!(
                frame.len(),
                HEADER_BYTES + message.payload_len() + CRC_BYTES
            );
            assert!(frame.len() <= MAX_RAW_FRAME_BYTES);
            assert_eq!(
                frame.as_bytes()[6],
                u8::try_from(message.payload_len()).expect("payload length is envelope-bounded")
            );
            assert_eq!(frame.decode(), Ok(message));

            let uart = UartRecord::encode(message).expect("valid UART record");
            assert!(uart.len() <= MAX_UART_RECORD_BYTES);
            assert_eq!(uart.as_bytes().last(), Some(&0));
        }
    }

    #[test]
    fn apply_pwm_wire_layout_is_explicit_little_endian() {
        let message = messages()
            .into_iter()
            .find(|message| message.kind() == MessageKind::ApplyPwm)
            .expect("apply fixture");
        let frame = RawFrame::encode(message).expect("frame");
        let bytes = frame.as_bytes();
        assert_eq!(&bytes[..8], &[b'K', b'R', b'P', b'2', 2, 0x20, 34, 0]);
        assert_eq!(&bytes[8..20], uid().as_bytes());
        assert_eq!(&bytes[20..28], &[8, 7, 6, 5, 4, 3, 2, 1]);
        assert_eq!(&bytes[28..32], &[0x44, 0x33, 0x22, 0x11]);
        assert_eq!(&bytes[32..36], &[7, 0, 0, 0]);
        assert_eq!(&bytes[36..40], &[0xd0, 0x07, 0, 0]);
        assert_eq!(&bytes[40..42], &[231, 40]);
    }

    #[test]
    fn every_single_bit_flip_in_a_valid_frame_is_rejected() {
        for message in messages() {
            let frame = RawFrame::encode(message).expect("frame");
            let mut original = [0_u8; MAX_RAW_FRAME_BYTES];
            original[..frame.len()].copy_from_slice(frame.as_bytes());
            for bit in 0..frame.len() * 8 {
                let mut altered = original;
                altered[bit / 8] ^= 1 << (bit % 8);
                assert!(
                    decode_raw_frame(&altered[..frame.len()]).is_err(),
                    "bit {bit} of {:?} escaped integrity checks",
                    message.kind()
                );
            }
        }
    }

    #[test]
    fn envelope_rejects_malformed_lengths_reserved_fields_and_trailing_bytes() {
        let frame = RawFrame::encode(messages()[7]).expect("apply frame");

        let mut reserved = [0_u8; MAX_RAW_FRAME_BYTES];
        reserved[..frame.len()].copy_from_slice(frame.as_bytes());
        reserved[7] = 1;
        assert!(matches!(
            decode_raw_frame(&reserved[..frame.len()]),
            Err(FrameError::NonzeroReserved { value: 1 })
        ));

        let mut trailing = [0_u8; MAX_RAW_FRAME_BYTES + 1];
        trailing[..frame.len()].copy_from_slice(frame.as_bytes());
        assert!(matches!(
            decode_raw_frame(&trailing[..frame.len() + 1]),
            Err(FrameError::LengthMismatch { .. })
        ));

        let mut wrong_length = [0_u8; MAX_RAW_FRAME_BYTES];
        wrong_length[..HEADER_BYTES + 33].copy_from_slice(&frame.as_bytes()[..HEADER_BYTES + 33]);
        wrong_length[6] = 33;
        let wrong_total = HEADER_BYTES + 33 + CRC_BYTES;
        replace_crc(&mut wrong_length[..wrong_total]);
        assert!(matches!(
            decode_raw_frame(&wrong_length[..wrong_total]),
            Err(FrameError::Payload(PayloadError::WrongLength {
                kind: MessageKind::ApplyPwm,
                expected: 34,
                actual: 33,
            }))
        ));

        assert!(matches!(
            decode_raw_frame(&frame.as_bytes()[..HEADER_BYTES]),
            Err(FrameError::TooShort { .. })
        ));

        let mut unknown_kind = [0_u8; MAX_RAW_FRAME_BYTES];
        unknown_kind[..frame.len()].copy_from_slice(frame.as_bytes());
        unknown_kind[5] = 0xfe;
        assert!(matches!(
            decode_raw_frame(&unknown_kind[..frame.len()]),
            Err(FrameError::UnknownMessageKind { value: 0xfe })
        ));
    }

    #[test]
    fn valid_crc_does_not_bypass_payload_domain_parsing() {
        let frame = RawFrame::encode(messages()[7]).expect("apply frame");
        let mut bytes = [0_u8; MAX_RAW_FRAME_BYTES];
        bytes[..frame.len()].copy_from_slice(frame.as_bytes());
        bytes[HEADER_BYTES..HEADER_BYTES + 12].fill(0);
        replace_crc(&mut bytes[..frame.len()]);
        assert!(matches!(
            decode_raw_frame(&bytes[..frame.len()]),
            Err(FrameError::Payload(PayloadError::Domain(
                DomainError::ZeroControllerUid
            )))
        ));
    }

    #[test]
    fn cobs_round_trips_empty_zero_dense_and_all_byte_values() {
        let mut encoded = [0_u8; 300];
        let mut decoded = [0_u8; 300];

        for input in [&b""[..], &b"\0"[..], &b"\0\0\0"[..], &b"abc\0def"[..]] {
            let encoded_len = cobs_encode(input, &mut encoded).expect("COBS encode");
            assert!(!encoded[..encoded_len].contains(&0));
            let decoded_len =
                cobs_decode(&encoded[..encoded_len], &mut decoded).expect("COBS decode");
            assert_eq!(&decoded[..decoded_len], input);
        }

        let mut all_bytes = [0_u8; 256];
        for (slot, value) in all_bytes.iter_mut().zip(u8::MIN..=u8::MAX) {
            *slot = value;
        }
        let encoded_len = cobs_encode(&all_bytes, &mut encoded).expect("all bytes encode");
        assert!(!encoded[..encoded_len].contains(&0));
        let decoded_len =
            cobs_decode(&encoded[..encoded_len], &mut decoded).expect("all bytes decode");
        assert_eq!(&decoded[..decoded_len], &all_bytes);

        let nonzero = [1_u8; 254];
        let encoded_len = cobs_encode(&nonzero, &mut encoded).expect("full COBS block");
        assert_eq!(encoded_len, 255);
        assert_eq!(encoded[0], u8::MAX);
        let decoded_len = cobs_decode(&encoded[..encoded_len], &mut decoded).expect("decode");
        assert_eq!(&decoded[..decoded_len], &nonzero);
    }

    #[test]
    fn cobs_rejects_zero_codes_truncation_and_small_buffers() {
        let mut output = [0_u8; 8];
        assert_eq!(cobs_decode(&[0], &mut output), Err(CobsError::ZeroCode));
        assert_eq!(
            cobs_decode(&[3, 1], &mut output),
            Err(CobsError::TruncatedBlock)
        );
        assert_eq!(cobs_encode(b"abc", &mut []), Err(CobsError::OutputTooSmall));
        assert_eq!(
            cobs_decode(&[4, 1, 2, 3], &mut [0_u8; 2]),
            Err(CobsError::OutputTooSmall)
        );
    }

    #[test]
    fn uart_stream_decodes_fragmented_records_and_resynchronizes() {
        let first = messages()[8];
        let second = messages()[11];
        let first_record = UartRecord::encode(first).expect("first record");
        let second_record = UartRecord::encode(second).expect("second record");
        let mut decoder = UartStreamDecoder::new();

        let mut event = None;
        for &byte in first_record.as_bytes() {
            if let Some(next) = decoder.push(byte) {
                assert!(event.is_none());
                event = Some(next);
            }
        }
        assert_eq!(event, Some(Ok(first)));

        let mut corrupted = [0_u8; MAX_UART_RECORD_BYTES];
        corrupted[..first_record.len()].copy_from_slice(first_record.as_bytes());
        corrupted[5] ^= 0x40;
        let mut saw_corruption = false;
        for &byte in &corrupted[..first_record.len()] {
            if let Some(result) = decoder.push(byte) {
                assert!(result.is_err());
                saw_corruption = true;
            }
        }
        assert!(saw_corruption);

        let mut second_event = None;
        for &byte in second_record.as_bytes() {
            if let Some(next) = decoder.push(byte) {
                second_event = Some(next);
            }
        }
        assert_eq!(second_event, Some(Ok(second)));
    }

    #[test]
    fn uart_stream_bounds_oversize_and_resumes_at_the_next_delimiter() {
        let mut decoder = UartStreamDecoder::new();
        for _ in 0..MAX_COBS_FRAME_BYTES {
            assert!(decoder.push(1).is_none());
        }
        assert!(matches!(
            decoder.push(1),
            Some(Err(UartStreamError::OversizedRecord {
                maximum: MAX_COBS_FRAME_BYTES
            }))
        ));
        assert!(decoder.is_discarding_oversized_record());
        assert!(decoder.push(2).is_none());
        assert!(decoder.push(0).is_none());
        assert!(!decoder.is_discarding_oversized_record());

        let message = messages()[0];
        let record = UartRecord::encode(message).expect("record");
        let mut recovered = None;
        for &byte in record.as_bytes() {
            if let Some(value) = decoder.push(byte) {
                recovered = Some(value);
            }
        }
        assert_eq!(recovered, Some(Ok(message)));
        assert_eq!(decoder.push(0), Some(Err(UartStreamError::EmptyRecord)));
    }

    #[test]
    fn cross_field_invariants_reject_unsafe_claims_before_encoding() {
        let mut unsafe_hello = hello();
        unsafe_hello.output_state = OutputState::NonzeroPwm;
        assert!(matches!(
            RawFrame::encode(Message::ControllerHello(unsafe_hello)),
            Err(EncodeError::PayloadInvariant(
                PayloadError::Invariant { .. }
            ))
        ));

        let mut invalid_acquire = match messages()[12] {
            Message::AcquireResult(value) => value,
            _ => unreachable!("fixture kind"),
        };
        invalid_acquire.control_epoch = None;
        assert!(RawFrame::encode(Message::AcquireResult(invalid_acquire)).is_err());

        let mut invalid_heartbeat = match messages()[10] {
            Message::Heartbeat(value) => value,
            _ => unreachable!("fixture kind"),
        };
        invalid_heartbeat.last_sequence = None;
        assert!(RawFrame::encode(Message::Heartbeat(invalid_heartbeat)).is_err());

        let mut rejected_nonzero = match messages()[9] {
            Message::AppliedResult(value) => value,
            _ => unreachable!("fixture kind"),
        };
        rejected_nonzero.result = AppliedResultCode::RejectedExpired;
        assert!(RawFrame::encode(Message::AppliedResult(rejected_nonzero)).is_err());
    }

    #[test]
    fn domains_cover_lease_sequence_identity_and_wrapping_deadline_edges() {
        assert!(V2CommandLeaseMs::try_new(MIN_V2_COMMAND_LEASE_MS).is_ok());
        assert!(V2CommandLeaseMs::try_new(MAX_V2_COMMAND_LEASE_MS).is_ok());
        assert!(V2CommandLeaseMs::try_new(MIN_V2_COMMAND_LEASE_MS - 1).is_err());
        assert!(V2CommandLeaseMs::try_new(MAX_V2_COMMAND_LEASE_MS + 1).is_err());
        assert_eq!(RemainingLeaseMs::try_new(0), Ok(RemainingLeaseMs::ZERO));
        assert!(RemainingLeaseMs::try_new(MAX_V2_COMMAND_LEASE_MS + 1).is_err());
        let motion_disabled = MaxAbsPwmPercent::try_new(0).expect("zero is a truthful profile");
        assert!(!motion_disabled.grants_motion_authority());
        assert!(MaxAbsPwmPercent::try_new(1)
            .expect("one percent grants bounded motion authority")
            .grants_motion_authority());
        assert!(MaxAbsPwmPercent::try_new(101).is_err());
        assert_eq!(V2CommandSequence::new(u32::MAX).checked_successor(), None);
        assert!(ControllerUid::try_new([0; 12]).is_err());
        assert!(ActuatorConfigFingerprint::try_new([0; 16]).is_err());
        assert!(ControllerCapabilities::try_from_bits(1 << 31).is_err());

        let before_wrap = ControllerUptimeMsWrapping::new(u32::MAX - 5);
        let after_wrap = ControllerDeadlineMsWrapping::new(3);
        assert_eq!(
            after_wrap.relation_to(before_wrap),
            DeadlineRelation::Future { remaining_ms: 9 }
        );
        assert_eq!(
            ControllerDeadlineMsWrapping::new(0x8000_0000)
                .relation_to(ControllerUptimeMsWrapping::new(0)),
            DeadlineRelation::AmbiguousHalfRange
        );
    }

    #[test]
    fn observational_odometry_preserves_truthful_wrapping_fields_only() {
        let message = messages()[11];
        let decoded = RawFrame::encode(message)
            .expect("odometry frame")
            .decode()
            .expect("typed odometry decode");
        let Message::ObservationalOdometry(value) = decoded else {
            panic!("wrong decoded kind")
        };
        assert_eq!(
            value.left_estimated_extended_ticks_wrapping.get(),
            -9_000_000_000
        );
        assert_eq!(
            value.right_estimated_extended_ticks_wrapping.get(),
            9_000_000_000
        );
        assert_eq!(value.left_sample_delta_ticks_modulo.get(), -321);
        assert_eq!(value.right_sample_delta_ticks_modulo.get(), 654);
        assert_eq!(value.controller_uptime.get(), u32::MAX - 2);
    }
}
