#![no_std]

use core::num::{NonZeroU16, NonZeroU32};

pub const MAX_ABS_PWM_PERCENT: i8 = 100;
pub const MAX_COMMAND_LEASE_MS: u16 = 1_000;
pub const MAX_UNAMBIGUOUS_WRAPPING_TIMER_TICKS: u32 = (1_u32 << 31) - 1;
pub const ROBOT_COMMAND_PACKET_BYTES: usize = 8;
pub const ROBOT_COMMAND_ACKNOWLEDGEMENT_PACKET_BYTES: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WrappingMillisClock {
    ticks_per_second: NonZeroU32,
    previous_ticks: u32,
    fractional_numerator: u64,
    elapsed_ms_wrapping: u32,
}

impl WrappingMillisClock {
    /// Converts wrapping 32-bit timer samples to wrapping milliseconds.
    ///
    /// Callers must sample at least once per complete timer wrap. Multiple
    /// wraps between calls are indistinguishable from a shorter interval.
    pub const fn new(ticks_per_second: NonZeroU32, initial_ticks: u32) -> Self {
        Self {
            ticks_per_second,
            previous_ticks: initial_ticks,
            fractional_numerator: 0,
            elapsed_ms_wrapping: 0,
        }
    }

    pub fn advance_to(&mut self, current_ticks: u32) -> ControllerUptimeMsWrapping {
        let delta_ticks = current_ticks.wrapping_sub(self.previous_ticks);
        self.previous_ticks = current_ticks;

        let numerator = self.fractional_numerator + u64::from(delta_ticks) * 1_000;
        let ticks_per_second = u64::from(self.ticks_per_second.get());
        let whole_ms = numerator / ticks_per_second;
        self.fractional_numerator = numerator % ticks_per_second;
        let whole_ms_bytes = whole_ms.to_le_bytes();
        let whole_ms_wrapping = u32::from_le_bytes([
            whole_ms_bytes[0],
            whole_ms_bytes[1],
            whole_ms_bytes[2],
            whole_ms_bytes[3],
        ]);
        self.elapsed_ms_wrapping = self.elapsed_ms_wrapping.wrapping_add(whole_ms_wrapping);
        ControllerUptimeMsWrapping::new(self.elapsed_ms_wrapping)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
/// Software-extended encoder ticks whose storage wraps in the i64 domain.
///
/// The value is an estimate because firmware may have to infer the direction
/// of an update that is pending while encoder interrupts are masked. It also
/// cannot recover multiple hardware-counter wraps missed between interrupts.
pub struct EstimatedWrappingEncoderTicks(i64);

impl EstimatedWrappingEncoderTicks {
    pub const fn new_wrapping(value: i64) -> Self {
        Self(value)
    }

    pub fn from_extended_16_bit_counter(timer_wraps: i64, raw_count: u16) -> Self {
        Self(
            timer_wraps
                .wrapping_mul(1_i64 << 16)
                .wrapping_add(i64::from(raw_count)),
        )
    }

    pub const fn get(self) -> i64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
pub struct ModuloEncoderDeltaTicks(i16);

impl ModuloEncoderDeltaTicks {
    pub const fn new_modulo(value: i16) -> Self {
        Self(value)
    }

    /// Returns the signed modulo-2^16 difference between two hardware counts.
    ///
    /// The physical displacement is ambiguous when it reaches half a counter
    /// range between samples. This type intentionally exposes only the modulo
    /// observation and does not claim an unambiguous physical displacement.
    pub const fn from_wrapping_counts(previous: u16, current: u16) -> Self {
        Self(i16::from_ne_bytes(
            current.wrapping_sub(previous).to_ne_bytes(),
        ))
    }

    pub const fn get(self) -> i16 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
pub struct ControllerUptimeMsWrapping(u32);

impl ControllerUptimeMsWrapping {
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u32 {
        self.0
    }

    /// Returns only the modulo-2^32 millisecond difference.
    ///
    /// Callers must separately prove that no complete uptime wrap occurred
    /// between the two observations before interpreting this as elapsed time.
    pub const fn wrapping_elapsed_since(self, earlier: Self) -> u32 {
        self.0.wrapping_sub(earlier.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct RobotOdometry {
    left_estimated_extended_ticks_wrapping_i64: EstimatedWrappingEncoderTicks,
    right_estimated_extended_ticks_wrapping_i64: EstimatedWrappingEncoderTicks,
    left_sample_delta_ticks_modulo_i16: ModuloEncoderDeltaTicks,
    right_sample_delta_ticks_modulo_i16: ModuloEncoderDeltaTicks,
    controller_uptime_ms_wrapping: ControllerUptimeMsWrapping,
}

impl RobotOdometry {
    pub const fn new(
        left_estimated_extended_ticks_wrapping_i64: EstimatedWrappingEncoderTicks,
        right_estimated_extended_ticks_wrapping_i64: EstimatedWrappingEncoderTicks,
        left_sample_delta_ticks_modulo_i16: ModuloEncoderDeltaTicks,
        right_sample_delta_ticks_modulo_i16: ModuloEncoderDeltaTicks,
        controller_uptime_ms_wrapping: ControllerUptimeMsWrapping,
    ) -> Self {
        Self {
            left_estimated_extended_ticks_wrapping_i64,
            right_estimated_extended_ticks_wrapping_i64,
            left_sample_delta_ticks_modulo_i16,
            right_sample_delta_ticks_modulo_i16,
            controller_uptime_ms_wrapping,
        }
    }

    pub const fn left_estimated_extended_ticks_wrapping_i64(self) -> EstimatedWrappingEncoderTicks {
        self.left_estimated_extended_ticks_wrapping_i64
    }

    pub const fn right_estimated_extended_ticks_wrapping_i64(
        self,
    ) -> EstimatedWrappingEncoderTicks {
        self.right_estimated_extended_ticks_wrapping_i64
    }

    pub const fn left_sample_delta_ticks_modulo_i16(self) -> ModuloEncoderDeltaTicks {
        self.left_sample_delta_ticks_modulo_i16
    }

    pub const fn right_sample_delta_ticks_modulo_i16(self) -> ModuloEncoderDeltaTicks {
        self.right_sample_delta_ticks_modulo_i16
    }

    pub const fn controller_uptime_ms_wrapping(self) -> ControllerUptimeMsWrapping {
        self.controller_uptime_ms_wrapping
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct RobotOdometryWithServerReceiveAge {
    odometry: RobotOdometry,
    server_receive_age_ms: u64,
}

impl RobotOdometryWithServerReceiveAge {
    pub const fn new(odometry: RobotOdometry, server_receive_age_ms: u64) -> Self {
        Self {
            odometry,
            server_receive_age_ms,
        }
    }

    pub const fn odometry(self) -> RobotOdometry {
        self.odometry
    }

    pub const fn server_receive_age_ms(self) -> u64 {
        self.server_receive_age_ms
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControllerError {
    ReceiveOverrun,
    InvalidCommand,
    LeaseTimerDomain,
    CommandTooLong,
    TransmitRecordDropped,
}

impl ControllerError {
    pub const fn code(self) -> &'static str {
        match self {
            Self::ReceiveOverrun => "RX_OVERRUN",
            Self::InvalidCommand => "CMD",
            Self::LeaseTimerDomain => "LEASE_TIMER",
            Self::CommandTooLong => "CMD_TOO_LONG",
            Self::TransmitRecordDropped => "TX_RECORD_DROPPED",
        }
    }

    const fn from_code(code: &str) -> Option<Self> {
        match code.as_bytes() {
            b"RX_OVERRUN" => Some(Self::ReceiveOverrun),
            b"CMD" => Some(Self::InvalidCommand),
            b"LEASE_TIMER" => Some(Self::LeaseTimerDomain),
            b"CMD_TOO_LONG" => Some(Self::CommandTooLong),
            b"TX_RECORD_DROPPED" => Some(Self::TransmitRecordDropped),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControllerEvent {
    Ready,
    CommandLeaseExpired,
}

impl ControllerEvent {
    pub const fn code(self) -> &'static str {
        match self {
            Self::Ready => "CONTROLLER_READY",
            Self::CommandLeaseExpired => "COMMAND_LEASE_EXPIRED",
        }
    }

    const fn from_code(code: &str) -> Option<Self> {
        match code.as_bytes() {
            b"CONTROLLER_READY" => Some(Self::Ready),
            b"COMMAND_LEASE_EXPIRED" => Some(Self::CommandLeaseExpired),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControllerReport {
    AppliedPwm(AppliedPwm),
    Odometry(RobotOdometry),
    Error(ControllerError),
    Event(ControllerEvent),
}

#[derive(Debug, PartialEq, Eq)]
pub enum ControllerReportError {
    InvalidUtf8(core::str::Utf8Error),
    MissingField {
        field: &'static str,
    },
    InvalidInteger {
        field: &'static str,
        source: core::num::ParseIntError,
    },
    InvalidAppliedPwm(AppliedPwmError),
    TrailingField,
    UnsupportedRecord,
}

impl core::fmt::Display for ControllerReportError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidUtf8(source) => {
                write!(f, "controller report is not valid UTF-8: {source}")
            }
            Self::MissingField { field } => write!(f, "controller report is missing {field}"),
            Self::InvalidInteger { field, source } => {
                write!(
                    f,
                    "controller report {field} is not a valid integer: {source}"
                )
            }
            Self::InvalidAppliedPwm(source) => {
                write!(
                    f,
                    "controller report contains invalid applied PWM: {source}"
                )
            }
            Self::TrailingField => f.write_str("controller report contains trailing fields"),
            Self::UnsupportedRecord => f.write_str("controller report type or code is unsupported"),
        }
    }
}

impl core::error::Error for ControllerReportError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::InvalidUtf8(source) => Some(source),
            Self::InvalidInteger { source, .. } => Some(source),
            Self::InvalidAppliedPwm(source) => Some(source),
            Self::MissingField { .. } | Self::TrailingField | Self::UnsupportedRecord => None,
        }
    }
}

fn parse_controller_report_integer<'a, T>(
    fields: &mut impl Iterator<Item = &'a str>,
    field: &'static str,
) -> Result<T, ControllerReportError>
where
    T: core::str::FromStr<Err = core::num::ParseIntError>,
{
    fields
        .next()
        .ok_or(ControllerReportError::MissingField { field })?
        .parse()
        .map_err(|source| ControllerReportError::InvalidInteger { field, source })
}

fn reject_trailing_controller_report_field<'a>(
    fields: &mut impl Iterator<Item = &'a str>,
) -> Result<(), ControllerReportError> {
    if fields.next().is_some() {
        Err(ControllerReportError::TrailingField)
    } else {
        Ok(())
    }
}

pub fn parse_controller_report(bytes: &[u8]) -> Result<ControllerReport, ControllerReportError> {
    let line = core::str::from_utf8(bytes).map_err(ControllerReportError::InvalidUtf8)?;
    let line = line.strip_suffix('\r').unwrap_or(line);
    let mut fields = line.split(',');
    match fields.next() {
        Some("PWM") => {
            let left = parse_controller_report_integer(&mut fields, "left applied PWM percent")?;
            let right = parse_controller_report_integer(&mut fields, "right applied PWM percent")?;
            reject_trailing_controller_report_field(&mut fields)?;
            AppliedPwm::try_new(left, right)
                .map(ControllerReport::AppliedPwm)
                .map_err(ControllerReportError::InvalidAppliedPwm)
        }
        Some("ODO") => {
            let left_extended = parse_controller_report_integer(
                &mut fields,
                "left estimated wrapping extended encoder ticks",
            )?;
            let right_extended = parse_controller_report_integer(
                &mut fields,
                "right estimated wrapping extended encoder ticks",
            )?;
            let left_delta =
                parse_controller_report_integer(&mut fields, "left modulo sample tick delta")?;
            let right_delta =
                parse_controller_report_integer(&mut fields, "right modulo sample tick delta")?;
            let uptime =
                parse_controller_report_integer(&mut fields, "wrapping controller uptime ms")?;
            reject_trailing_controller_report_field(&mut fields)?;
            Ok(ControllerReport::Odometry(RobotOdometry::new(
                EstimatedWrappingEncoderTicks::new_wrapping(left_extended),
                EstimatedWrappingEncoderTicks::new_wrapping(right_extended),
                ModuloEncoderDeltaTicks::new_modulo(left_delta),
                ModuloEncoderDeltaTicks::new_modulo(right_delta),
                ControllerUptimeMsWrapping::new(uptime),
            )))
        }
        Some("ERR") => {
            let code = fields.next().ok_or(ControllerReportError::MissingField {
                field: "controller error code",
            })?;
            let error =
                ControllerError::from_code(code).ok_or(ControllerReportError::UnsupportedRecord)?;
            reject_trailing_controller_report_field(&mut fields)?;
            Ok(ControllerReport::Error(error))
        }
        Some("EVT") => {
            let code = fields.next().ok_or(ControllerReportError::MissingField {
                field: "controller event code",
            })?;
            let event =
                ControllerEvent::from_code(code).ok_or(ControllerReportError::UnsupportedRecord)?;
            reject_trailing_controller_report_field(&mut fields)?;
            Ok(ControllerReport::Event(event))
        }
        Some(_) | None => Err(ControllerReportError::UnsupportedRecord),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RobotCommandPacket {
    pub left_pwm_percent: i8,
    pub right_pwm_percent: i8,
    pub lease_ms: u16,
    pub sequence: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RobotCommandAcknowledgementPacket {
    pub reserved_zero_i8_a: i8,
    pub reserved_zero_i8_b: i8,
    pub reserved_zero_u16: u16,
    pub accepted_sequence: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RobotPacketLengthError {
    expected: usize,
    actual: usize,
}

impl RobotPacketLengthError {
    pub const fn expected(self) -> usize {
        self.expected
    }

    pub const fn actual(self) -> usize {
        self.actual
    }
}

impl core::fmt::Display for RobotPacketLengthError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            formatter,
            "robot wire packet must contain exactly {} bytes, got {}",
            self.expected, self.actual
        )
    }
}

impl core::error::Error for RobotPacketLengthError {}

impl RobotCommandPacket {
    pub const fn to_legacy_wire_bytes(self) -> [u8; ROBOT_COMMAND_PACKET_BYTES] {
        let lease = self.lease_ms.to_le_bytes();
        let sequence = self.sequence.to_le_bytes();
        [
            self.left_pwm_percent.to_le_bytes()[0],
            self.right_pwm_percent.to_le_bytes()[0],
            lease[0],
            lease[1],
            sequence[0],
            sequence[1],
            sequence[2],
            sequence[3],
        ]
    }

    pub fn try_from_legacy_wire_bytes(bytes: &[u8]) -> Result<Self, RobotPacketLengthError> {
        let bytes: &[u8; ROBOT_COMMAND_PACKET_BYTES] =
            bytes.try_into().map_err(|_| RobotPacketLengthError {
                expected: ROBOT_COMMAND_PACKET_BYTES,
                actual: bytes.len(),
            })?;
        Ok(Self {
            left_pwm_percent: i8::from_le_bytes([bytes[0]]),
            right_pwm_percent: i8::from_le_bytes([bytes[1]]),
            lease_ms: u16::from_le_bytes([bytes[2], bytes[3]]),
            sequence: u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
        })
    }
}

impl RobotCommandAcknowledgementPacket {
    pub const fn to_legacy_wire_bytes(self) -> [u8; ROBOT_COMMAND_ACKNOWLEDGEMENT_PACKET_BYTES] {
        let reserved_u16 = self.reserved_zero_u16.to_le_bytes();
        let sequence = self.accepted_sequence.to_le_bytes();
        [
            self.reserved_zero_i8_a.to_le_bytes()[0],
            self.reserved_zero_i8_b.to_le_bytes()[0],
            reserved_u16[0],
            reserved_u16[1],
            sequence[0],
            sequence[1],
            sequence[2],
            sequence[3],
        ]
    }

    pub fn try_from_legacy_wire_bytes(bytes: &[u8]) -> Result<Self, RobotPacketLengthError> {
        let bytes: &[u8; ROBOT_COMMAND_ACKNOWLEDGEMENT_PACKET_BYTES] =
            bytes.try_into().map_err(|_| RobotPacketLengthError {
                expected: ROBOT_COMMAND_ACKNOWLEDGEMENT_PACKET_BYTES,
                actual: bytes.len(),
            })?;
        Ok(Self {
            reserved_zero_i8_a: i8::from_le_bytes([bytes[0]]),
            reserved_zero_i8_b: i8::from_le_bytes([bytes[1]]),
            reserved_zero_u16: u16::from_le_bytes([bytes[2], bytes[3]]),
            accepted_sequence: u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PwmPercent(i8);

impl PwmPercent {
    pub const ZERO: Self = Self(0);

    pub fn try_new(value: i8) -> Result<Self, PwmPercentError> {
        if (-MAX_ABS_PWM_PERCENT..=MAX_ABS_PWM_PERCENT).contains(&value) {
            Ok(Self(value))
        } else {
            Err(PwmPercentError { value })
        }
    }

    pub const fn get(self) -> i8 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PwmPercentError {
    value: i8,
}

impl PwmPercentError {
    pub const fn value(self) -> i8 {
        self.value
    }
}

impl core::fmt::Display for PwmPercentError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "PWM command must be between -{MAX_ABS_PWM_PERCENT}% and {MAX_ABS_PWM_PERCENT}%, got {}%",
            self.value
        )
    }
}

impl core::error::Error for PwmPercentError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AppliedPwm {
    left: PwmPercent,
    right: PwmPercent,
}

impl AppliedPwm {
    pub fn try_new(left: i8, right: i8) -> Result<Self, AppliedPwmError> {
        Ok(Self {
            left: PwmPercent::try_new(left).map_err(AppliedPwmError::Left)?,
            right: PwmPercent::try_new(right).map_err(AppliedPwmError::Right)?,
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
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AppliedPwmError {
    Left(PwmPercentError),
    Right(PwmPercentError),
}

impl core::fmt::Display for AppliedPwmError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Left(source) => write!(f, "invalid left applied PWM: {source}"),
            Self::Right(source) => write!(f, "invalid right applied PWM: {source}"),
        }
    }
}

impl core::error::Error for AppliedPwmError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Left(source) | Self::Right(source) => Some(source),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommandLeaseMs(NonZeroU16);

impl CommandLeaseMs {
    pub fn try_new(value: u16) -> Result<Self, CommandLeaseError> {
        let value = NonZeroU16::new(value).ok_or(CommandLeaseError::Zero)?;
        if value.get() > MAX_COMMAND_LEASE_MS {
            return Err(CommandLeaseError::AboveMaximum {
                value: value.get(),
                maximum: MAX_COMMAND_LEASE_MS,
            });
        }
        Ok(Self(value))
    }

    pub const fn get(self) -> u16 {
        self.0.get()
    }

    pub fn wrapping_timer_ticks_ceil(
        self,
        ticks_per_second: NonZeroU32,
    ) -> Result<NonZeroU32, CommandLeaseTickError> {
        let numerator = u64::from(self.get()) * u64::from(ticks_per_second.get());
        let ticks = numerator.div_ceil(1_000);
        if ticks > u64::from(MAX_UNAMBIGUOUS_WRAPPING_TIMER_TICKS) {
            return Err(CommandLeaseTickError::ExceedsUnambiguousHalfRange {
                ticks,
                maximum: MAX_UNAMBIGUOUS_WRAPPING_TIMER_TICKS,
            });
        }

        let ticks =
            u32::try_from(ticks).map_err(|_| CommandLeaseTickError::ArithmeticInvariantViolated)?;
        NonZeroU32::new(ticks).ok_or(CommandLeaseTickError::ArithmeticInvariantViolated)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommandLeaseTickError {
    ArithmeticInvariantViolated,
    ExceedsUnambiguousHalfRange { ticks: u64, maximum: u32 },
}

impl core::fmt::Display for CommandLeaseTickError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ArithmeticInvariantViolated => {
                f.write_str("nonzero command lease and timer frequency produced zero timer ticks")
            }
            Self::ExceedsUnambiguousHalfRange { ticks, maximum } => write!(
                f,
                "command lease requires {ticks} timer ticks, exceeding the wrap-safe maximum {maximum}"
            ),
        }
    }
}

impl core::error::Error for CommandLeaseTickError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommandLeaseError {
    Zero,
    AboveMaximum { value: u16, maximum: u16 },
}

impl core::fmt::Display for CommandLeaseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Zero => f.write_str("command lease must be nonzero"),
            Self::AboveMaximum { value, maximum } => write!(
                f,
                "command lease {value} ms exceeds the maximum {maximum} ms"
            ),
        }
    }
}

impl core::error::Error for CommandLeaseError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommandSequence(u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommandSequenceRelation {
    Duplicate,
    Newer,
    Older,
    AmbiguousHalfRange,
}

impl CommandSequenceRelation {
    pub const fn description(self) -> &'static str {
        match self {
            Self::Duplicate => "duplicate",
            Self::Newer => "newer",
            Self::Older => "older",
            Self::AmbiguousHalfRange => "ambiguous at the modular half-range",
        }
    }
}

impl CommandSequence {
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u32 {
        self.0
    }

    pub const fn relation_to(self, previous: Self) -> CommandSequenceRelation {
        let delta = self.0.wrapping_sub(previous.0);
        match delta {
            0 => CommandSequenceRelation::Duplicate,
            0x8000_0000 => CommandSequenceRelation::AmbiguousHalfRange,
            1..0x8000_0000 => CommandSequenceRelation::Newer,
            _ => CommandSequenceRelation::Older,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RobotCommandAcknowledgement {
    accepted_sequence: CommandSequence,
}

impl RobotCommandAcknowledgement {
    pub const fn new(accepted_sequence: CommandSequence) -> Self {
        Self { accepted_sequence }
    }

    pub const fn accepted_sequence(self) -> CommandSequence {
        self.accepted_sequence
    }
}

impl TryFrom<RobotCommandAcknowledgementPacket> for RobotCommandAcknowledgement {
    type Error = RobotCommandAcknowledgementError;

    fn try_from(packet: RobotCommandAcknowledgementPacket) -> Result<Self, Self::Error> {
        if packet.reserved_zero_i8_a != 0
            || packet.reserved_zero_i8_b != 0
            || packet.reserved_zero_u16 != 0
        {
            return Err(RobotCommandAcknowledgementError::NonzeroReservedFields {
                i8_a: packet.reserved_zero_i8_a,
                i8_b: packet.reserved_zero_i8_b,
                u16_value: packet.reserved_zero_u16,
            });
        }
        Ok(Self::new(CommandSequence::new(packet.accepted_sequence)))
    }
}

impl From<RobotCommandAcknowledgement> for RobotCommandAcknowledgementPacket {
    fn from(acknowledgement: RobotCommandAcknowledgement) -> Self {
        Self {
            reserved_zero_i8_a: 0,
            reserved_zero_i8_b: 0,
            reserved_zero_u16: 0,
            accepted_sequence: acknowledgement.accepted_sequence.get(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RobotCommandAcknowledgementError {
    NonzeroReservedFields { i8_a: i8, i8_b: i8, u16_value: u16 },
}

impl core::fmt::Display for RobotCommandAcknowledgementError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NonzeroReservedFields {
                i8_a,
                i8_b,
                u16_value,
            } => write!(
                f,
                "command acknowledgement reserved fields must be zero, got {i8_a}, {i8_b}, {u16_value}"
            ),
        }
    }
}

impl core::error::Error for RobotCommandAcknowledgementError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LeasedPwmCommand {
    left_pwm_percent: PwmPercent,
    right_pwm_percent: PwmPercent,
    lease_ms: CommandLeaseMs,
}

impl LeasedPwmCommand {
    pub const fn from_validated(
        left_pwm_percent: PwmPercent,
        right_pwm_percent: PwmPercent,
        lease_ms: CommandLeaseMs,
    ) -> Self {
        Self {
            left_pwm_percent,
            right_pwm_percent,
            lease_ms,
        }
    }

    pub fn try_new(
        left_pwm_percent: i8,
        right_pwm_percent: i8,
        lease_ms: u16,
    ) -> Result<Self, LeasedPwmCommandError> {
        Ok(Self::from_validated(
            PwmPercent::try_new(left_pwm_percent).map_err(LeasedPwmCommandError::LeftPwm)?,
            PwmPercent::try_new(right_pwm_percent).map_err(LeasedPwmCommandError::RightPwm)?,
            CommandLeaseMs::try_new(lease_ms).map_err(LeasedPwmCommandError::Lease)?,
        ))
    }

    pub const fn left_pwm_percent(self) -> PwmPercent {
        self.left_pwm_percent
    }

    pub const fn right_pwm_percent(self) -> PwmPercent {
        self.right_pwm_percent
    }

    pub const fn lease_ms(self) -> CommandLeaseMs {
        self.lease_ms
    }

    pub const fn is_stop(self) -> bool {
        self.left_pwm_percent.get() == 0 && self.right_pwm_percent.get() == 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RobotCommand {
    leased_pwm: LeasedPwmCommand,
    sequence: CommandSequence,
}

impl RobotCommand {
    pub const fn from_leased_pwm(leased_pwm: LeasedPwmCommand, sequence: CommandSequence) -> Self {
        Self {
            leased_pwm,
            sequence,
        }
    }

    pub fn try_new(
        left_pwm_percent: i8,
        right_pwm_percent: i8,
        lease_ms: u16,
        sequence: u32,
    ) -> Result<Self, LeasedPwmCommandError> {
        Ok(Self::from_leased_pwm(
            LeasedPwmCommand::try_new(left_pwm_percent, right_pwm_percent, lease_ms)?,
            CommandSequence::new(sequence),
        ))
    }

    pub const fn left_pwm_percent(self) -> PwmPercent {
        self.leased_pwm.left_pwm_percent()
    }

    pub const fn right_pwm_percent(self) -> PwmPercent {
        self.leased_pwm.right_pwm_percent()
    }

    pub const fn lease_ms(self) -> CommandLeaseMs {
        self.leased_pwm.lease_ms()
    }

    pub const fn leased_pwm(self) -> LeasedPwmCommand {
        self.leased_pwm
    }

    pub const fn sequence(self) -> CommandSequence {
        self.sequence
    }
}

impl TryFrom<RobotCommandPacket> for RobotCommand {
    type Error = LeasedPwmCommandError;

    fn try_from(value: RobotCommandPacket) -> Result<Self, Self::Error> {
        Self::try_new(
            value.left_pwm_percent,
            value.right_pwm_percent,
            value.lease_ms,
            value.sequence,
        )
    }
}

impl From<RobotCommand> for RobotCommandPacket {
    fn from(value: RobotCommand) -> Self {
        Self {
            left_pwm_percent: value.left_pwm_percent().get(),
            right_pwm_percent: value.right_pwm_percent().get(),
            lease_ms: value.lease_ms().get(),
            sequence: value.sequence().get(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeasedPwmCommandError {
    LeftPwm(PwmPercentError),
    RightPwm(PwmPercentError),
    Lease(CommandLeaseError),
}

impl core::fmt::Display for LeasedPwmCommandError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::LeftPwm(source) => write!(f, "invalid left-wheel command: {source}"),
            Self::RightPwm(source) => write!(f, "invalid right-wheel command: {source}"),
            Self::Lease(source) => write!(f, "invalid command lease: {source}"),
        }
    }
}

impl core::error::Error for LeasedPwmCommandError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::LeftPwm(source) | Self::RightPwm(source) => Some(source),
            Self::Lease(source) => Some(source),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum SerialPwmCommandError {
    InvalidUtf8(core::str::Utf8Error),
    UnsupportedRecord,
    MissingField {
        field: &'static str,
    },
    InvalidInteger {
        field: &'static str,
        source: core::num::ParseIntError,
    },
    TrailingField,
    InvalidCommand(LeasedPwmCommandError),
}

impl core::fmt::Display for SerialPwmCommandError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidUtf8(source) => write!(f, "serial command is not UTF-8: {source}"),
            Self::UnsupportedRecord => f.write_str("serial record is not a CMD command"),
            Self::MissingField { field } => write!(f, "serial command is missing {field}"),
            Self::InvalidInteger { field, source } => {
                write!(f, "serial command {field} is not an integer: {source}")
            }
            Self::TrailingField => f.write_str("serial command contains trailing fields"),
            Self::InvalidCommand(source) => {
                write!(f, "serial command violates the command domain: {source}")
            }
        }
    }
}

impl core::error::Error for SerialPwmCommandError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::InvalidUtf8(source) => Some(source),
            Self::InvalidInteger { source, .. } => Some(source),
            Self::InvalidCommand(source) => Some(source),
            Self::UnsupportedRecord | Self::MissingField { .. } | Self::TrailingField => None,
        }
    }
}

fn parse_serial_integer<'a, T>(
    fields: &mut impl Iterator<Item = &'a str>,
    field: &'static str,
) -> Result<T, SerialPwmCommandError>
where
    T: core::str::FromStr<Err = core::num::ParseIntError>,
{
    fields
        .next()
        .ok_or(SerialPwmCommandError::MissingField { field })?
        .parse()
        .map_err(|source| SerialPwmCommandError::InvalidInteger { field, source })
}

pub fn parse_serial_pwm_command(bytes: &[u8]) -> Result<LeasedPwmCommand, SerialPwmCommandError> {
    let line = core::str::from_utf8(bytes).map_err(SerialPwmCommandError::InvalidUtf8)?;
    let line = line.strip_suffix('\r').unwrap_or(line);
    let mut fields = line.split(',');
    if fields.next() != Some("CMD") {
        return Err(SerialPwmCommandError::UnsupportedRecord);
    }

    let left_pwm_percent = parse_serial_integer(&mut fields, "left PWM percent")?;
    let right_pwm_percent = parse_serial_integer(&mut fields, "right PWM percent")?;
    let lease_ms = parse_serial_integer(&mut fields, "lease milliseconds")?;
    if fields.next().is_some() {
        return Err(SerialPwmCommandError::TrailingField);
    }

    LeasedPwmCommand::try_new(left_pwm_percent, right_pwm_percent, lease_ms)
        .map_err(SerialPwmCommandError::InvalidCommand)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pwm_percent_accepts_exact_domain_without_clamping() {
        for value in i8::MIN..=i8::MAX {
            let parsed = PwmPercent::try_new(value);
            if (-MAX_ABS_PWM_PERCENT..=MAX_ABS_PWM_PERCENT).contains(&value) {
                assert_eq!(parsed.expect("in-domain PWM").get(), value);
            } else {
                assert_eq!(parsed.expect_err("out-of-domain PWM").value(), value);
            }
        }
    }

    #[test]
    fn command_lease_is_nonzero_bounded_and_uses_ceiling_timer_ticks() {
        assert_eq!(CommandLeaseMs::try_new(0), Err(CommandLeaseError::Zero));
        assert_eq!(
            CommandLeaseMs::try_new(MAX_COMMAND_LEASE_MS + 1),
            Err(CommandLeaseError::AboveMaximum {
                value: MAX_COMMAND_LEASE_MS + 1,
                maximum: MAX_COMMAND_LEASE_MS,
            })
        );

        let one_hz = NonZeroU32::new(1).expect("literal is nonzero");
        assert_eq!(
            CommandLeaseMs::try_new(1)
                .expect("one millisecond")
                .wrapping_timer_ticks_ceil(one_hz),
            Ok(NonZeroU32::MIN)
        );
        let core_clock = NonZeroU32::new(168_000_000).expect("literal is nonzero");
        assert_eq!(
            CommandLeaseMs::try_new(150)
                .expect("150 milliseconds")
                .wrapping_timer_ticks_ceil(core_clock)
                .expect("150 ms fits the wrapping timer domain")
                .get(),
            25_200_000
        );
        assert_eq!(
            CommandLeaseMs::try_new(MAX_COMMAND_LEASE_MS)
                .expect("maximum lease")
                .wrapping_timer_ticks_ceil(NonZeroU32::new(u32::MAX).expect("nonzero")),
            Err(CommandLeaseTickError::ExceedsUnambiguousHalfRange {
                ticks: u64::from(u32::MAX),
                maximum: MAX_UNAMBIGUOUS_WRAPPING_TIMER_TICKS,
            })
        );
    }

    #[test]
    fn sequence_relation_distinguishes_duplicate_older_newer_and_half_range() {
        let previous = CommandSequence::new(10);
        assert_eq!(
            previous.relation_to(previous),
            CommandSequenceRelation::Duplicate
        );
        assert_eq!(
            CommandSequence::new(11).relation_to(previous),
            CommandSequenceRelation::Newer
        );
        assert_eq!(
            CommandSequence::new(9).relation_to(previous),
            CommandSequenceRelation::Older
        );
        assert_eq!(
            CommandSequence::new(0).relation_to(CommandSequence::new(u32::MAX)),
            CommandSequenceRelation::Newer
        );
        assert_eq!(
            CommandSequence::new(1_u32 << 31).relation_to(CommandSequence::new(0)),
            CommandSequenceRelation::AmbiguousHalfRange
        );
    }

    #[test]
    fn packet_parses_once_and_round_trips_validated_command() {
        let packet = RobotCommandPacket {
            left_pwm_percent: -25,
            right_pwm_percent: 40,
            lease_ms: 150,
            sequence: u32::MAX,
        };
        let command = RobotCommand::try_from(packet).expect("valid command packet");
        assert_eq!(RobotCommandPacket::from(command), packet);
        let wire = packet.to_legacy_wire_bytes();
        assert_eq!(wire, [231, 40, 150, 0, 255, 255, 255, 255]);
        assert_eq!(
            RobotCommandPacket::try_from_legacy_wire_bytes(&wire),
            Ok(packet)
        );
        let length_error = RobotCommandPacket::try_from_legacy_wire_bytes(&wire[..7])
            .expect_err("truncated packet must reject");
        assert_eq!(length_error.expected(), ROBOT_COMMAND_PACKET_BYTES);
        assert_eq!(length_error.actual(), 7);

        let error = RobotCommand::try_new(-101, 0, 150, 1).expect_err("must reject, not clamp");
        assert!(matches!(error, LeasedPwmCommandError::LeftPwm(_)));
        assert!(core::error::Error::source(&error).is_some());
    }

    #[test]
    fn acknowledgement_proves_only_sequence_acceptance_and_requires_reserved_zeros() {
        let acknowledgement = RobotCommandAcknowledgement::new(CommandSequence::new(8));
        let packet = RobotCommandAcknowledgementPacket::from(acknowledgement);
        let wire = packet.to_legacy_wire_bytes();
        assert_eq!(wire, [0, 0, 0, 0, 8, 0, 0, 0]);
        assert_eq!(
            RobotCommandAcknowledgementPacket::try_from_legacy_wire_bytes(&wire),
            Ok(packet)
        );
        assert_eq!(
            RobotCommandAcknowledgement::try_from(packet),
            Ok(acknowledgement)
        );

        assert_eq!(
            RobotCommandAcknowledgement::try_from(RobotCommandAcknowledgementPacket {
                reserved_zero_i8_a: 1,
                reserved_zero_i8_b: 0,
                reserved_zero_u16: 0,
                accepted_sequence: 9,
            }),
            Err(RobotCommandAcknowledgementError::NonzeroReservedFields {
                i8_a: 1,
                i8_b: 0,
                u16_value: 0,
            })
        );
    }

    #[test]
    fn serial_command_parser_is_exact_and_returns_validated_types() {
        let command = parse_serial_pwm_command(b"CMD,-25,40,150\r")
            .expect("valid command with optional carriage return");
        assert_eq!(command.left_pwm_percent().get(), -25);
        assert_eq!(command.right_pwm_percent().get(), 40);
        assert_eq!(command.lease_ms().get(), 150);

        assert!(matches!(
            parse_serial_pwm_command(b"CMD,0,0,1,extra"),
            Err(SerialPwmCommandError::TrailingField)
        ));
        assert!(matches!(
            parse_serial_pwm_command(b"PWM,0,0"),
            Err(SerialPwmCommandError::UnsupportedRecord)
        ));

        let error = parse_serial_pwm_command(b"CMD,-101,0,150")
            .expect_err("out-of-domain PWM must reject rather than clamp");
        assert!(matches!(
            error,
            SerialPwmCommandError::InvalidCommand(LeasedPwmCommandError::LeftPwm(_))
        ));
        assert!(core::error::Error::source(&error).is_some());

        let error = parse_serial_pwm_command(b"CMD,not-a-number,0,150")
            .expect_err("invalid integers must preserve their source");
        assert!(matches!(
            error,
            SerialPwmCommandError::InvalidInteger {
                field: "left PWM percent",
                ..
            }
        ));
        assert!(core::error::Error::source(&error).is_some());
    }

    #[test]
    fn controller_report_parser_is_exact_typed_and_source_chained() {
        assert_eq!(
            parse_controller_report(b"PWM,-25,40\r"),
            Ok(ControllerReport::AppliedPwm(
                AppliedPwm::try_new(-25, 40).expect("valid fixture")
            ))
        );
        assert_eq!(
            parse_controller_report(b"ODO,-10,20,-3,4,500"),
            Ok(ControllerReport::Odometry(RobotOdometry::new(
                EstimatedWrappingEncoderTicks::new_wrapping(-10),
                EstimatedWrappingEncoderTicks::new_wrapping(20),
                ModuloEncoderDeltaTicks::new_modulo(-3),
                ModuloEncoderDeltaTicks::new_modulo(4),
                ControllerUptimeMsWrapping::new(500),
            )))
        );
        for (record, expected) in [
            (
                &b"ERR,TX_RECORD_DROPPED"[..],
                ControllerReport::Error(ControllerError::TransmitRecordDropped),
            ),
            (
                &b"EVT,CONTROLLER_READY"[..],
                ControllerReport::Event(ControllerEvent::Ready),
            ),
            (
                &b"EVT,COMMAND_LEASE_EXPIRED"[..],
                ControllerReport::Event(ControllerEvent::CommandLeaseExpired),
            ),
        ] {
            assert_eq!(parse_controller_report(record), Ok(expected));
        }

        assert!(matches!(
            parse_controller_report(b"PWM,0,0,extra"),
            Err(ControllerReportError::TrailingField)
        ));
        let error = parse_controller_report(b"PWM,-101,0")
            .expect_err("invalid applied PWM must reject rather than clamp");
        assert!(matches!(error, ControllerReportError::InvalidAppliedPwm(_)));
        assert!(core::error::Error::source(&error).is_some());

        let error = parse_controller_report(b"ODO,not-a-number,0,0,0,0")
            .expect_err("malformed integer must reject the whole report");
        assert!(matches!(
            error,
            ControllerReportError::InvalidInteger {
                field: "left estimated wrapping extended encoder ticks",
                ..
            }
        ));
        assert!(core::error::Error::source(&error).is_some());
        assert!(matches!(
            parse_controller_report(b"EVT,UNKNOWN"),
            Err(ControllerReportError::UnsupportedRecord)
        ));
        assert!(matches!(
            parse_controller_report(b"ERR"),
            Err(ControllerReportError::MissingField {
                field: "controller error code"
            })
        ));
    }

    #[test]
    fn wrapping_millisecond_clock_preserves_fractional_ticks_and_counter_wrap() {
        let mut clock = WrappingMillisClock::new(
            NonZeroU32::new(2_000).expect("literal is nonzero"),
            u32::MAX - 2,
        );
        assert_eq!(clock.advance_to(u32::MAX).get(), 1);
        assert_eq!(clock.advance_to(2).get(), 2);
        assert_eq!(clock.advance_to(3).get(), 3);

        let mut core_clock = WrappingMillisClock::new(
            NonZeroU32::new(168_000_000).expect("literal is nonzero"),
            10,
        );
        assert_eq!(core_clock.advance_to(840_010).get(), 5);

        let mut one_hz_clock = WrappingMillisClock::new(NonZeroU32::MIN, 0);
        assert_eq!(
            one_hz_clock.advance_to(u32::MAX).get(),
            u32::MAX.wrapping_mul(1_000)
        );
    }

    #[test]
    fn encoder_types_expose_wrapping_and_modulo_domains() {
        assert_eq!(
            EstimatedWrappingEncoderTicks::from_extended_16_bit_counter(1, 2).get(),
            65_538
        );
        assert_eq!(
            EstimatedWrappingEncoderTicks::from_extended_16_bit_counter(-1, u16::MAX).get(),
            -1
        );
        assert_eq!(
            EstimatedWrappingEncoderTicks::from_extended_16_bit_counter(i64::MAX, u16::MAX).get(),
            -1
        );

        assert_eq!(
            ModuloEncoderDeltaTicks::from_wrapping_counts(u16::MAX, 0).get(),
            1
        );
        assert_eq!(
            ModuloEncoderDeltaTicks::from_wrapping_counts(0, u16::MAX).get(),
            -1
        );
        assert_eq!(
            ModuloEncoderDeltaTicks::from_wrapping_counts(0, 1_u16 << 15).get(),
            i16::MIN
        );
    }

    #[test]
    fn wrapping_uptime_difference_crosses_u32_wrap() {
        let before = ControllerUptimeMsWrapping::new(u32::MAX - 2);
        let after = ControllerUptimeMsWrapping::new(3);
        assert_eq!(after.wrapping_elapsed_since(before), 6);
    }
}
