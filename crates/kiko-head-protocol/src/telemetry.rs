use core::fmt;

use crate::packet::{
    FULL_TELEMETRY_BYTES, FrameBuildError, PositionTicks, ResponseParseError, ServoId,
    parse_status_response,
};

/// Exact present-position response after envelope and domain parsing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PresentPosition {
    id: ServoId,
    ticks: PositionTicks,
}

impl PresentPosition {
    pub fn parse(bytes: &[u8], expected_id: ServoId) -> Result<Self, TelemetryParseError> {
        let parameters = parse_status_response(bytes, expected_id, 2)?;
        let raw = u16::from_le_bytes([parameters[0], parameters[1]]);
        let ticks = PositionTicks::try_new(raw).map_err(|source| {
            TelemetryParseError::PositionOutOfRange {
                id: expected_id,
                raw,
                source,
            }
        })?;
        Ok(Self {
            id: expected_id,
            ticks,
        })
    }

    pub const fn id(self) -> ServoId {
        self.id
    }

    pub const fn ticks(self) -> PositionTicks {
        self.ticks
    }
}

/// One checksum-valid 15-byte STS register window (`56..=70`).
///
/// Position and the moving flag have stable domain semantics. Other values
/// remain explicitly raw because their sign, scale, and fault conventions
/// have not yet been qualified against the exact installed firmware.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FullTelemetry {
    id: ServoId,
    position: PositionTicks,
    speed_raw: u16,
    load_raw: u16,
    voltage_raw: u8,
    temperature_raw: u8,
    async_write_flag_raw: u8,
    device_status_raw: u8,
    moving: bool,
    registers_67_68_raw: u16,
    current_raw: u16,
}

impl FullTelemetry {
    pub fn parse(bytes: &[u8], expected_id: ServoId) -> Result<Self, TelemetryParseError> {
        let data = parse_status_response(bytes, expected_id, FULL_TELEMETRY_BYTES)?;
        let raw_position = le_u16(data, 0);
        let position = PositionTicks::try_new(raw_position).map_err(|source| {
            TelemetryParseError::PositionOutOfRange {
                id: expected_id,
                raw: raw_position,
                source,
            }
        })?;
        let moving = match data[10] {
            0 => false,
            1 => true,
            value => {
                return Err(TelemetryParseError::InvalidMovingFlag {
                    id: expected_id,
                    value,
                });
            }
        };
        Ok(Self {
            id: expected_id,
            position,
            speed_raw: le_u16(data, 2),
            load_raw: le_u16(data, 4),
            voltage_raw: data[6],
            temperature_raw: data[7],
            async_write_flag_raw: data[8],
            device_status_raw: data[9],
            moving,
            registers_67_68_raw: le_u16(data, 11),
            current_raw: le_u16(data, 13),
        })
    }

    pub const fn id(self) -> ServoId {
        self.id
    }

    pub const fn position(self) -> PositionTicks {
        self.position
    }

    pub const fn speed_raw(self) -> u16 {
        self.speed_raw
    }

    pub const fn load_raw(self) -> u16 {
        self.load_raw
    }

    pub const fn voltage_raw(self) -> u8 {
        self.voltage_raw
    }

    pub const fn temperature_raw(self) -> u8 {
        self.temperature_raw
    }

    pub const fn async_write_flag_raw(self) -> u8 {
        self.async_write_flag_raw
    }

    pub const fn device_status_raw(self) -> u8 {
        self.device_status_raw
    }

    pub const fn is_moving(self) -> bool {
        self.moving
    }

    pub const fn registers_67_68_raw(self) -> u16 {
        self.registers_67_68_raw
    }

    pub const fn current_raw(self) -> u16 {
        self.current_raw
    }
}

fn le_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TelemetryParseError {
    Response(ResponseParseError),
    PositionOutOfRange {
        id: ServoId,
        raw: u16,
        source: FrameBuildError,
    },
    InvalidMovingFlag {
        id: ServoId,
        value: u8,
    },
}

impl From<ResponseParseError> for TelemetryParseError {
    fn from(source: ResponseParseError) -> Self {
        Self::Response(source)
    }
}

impl fmt::Display for TelemetryParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid typed STS telemetry: {self:?}")
    }
}

impl core::error::Error for TelemetryParseError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Response(source) => Some(source),
            Self::PositionOutOfRange { source, .. } => Some(source),
            Self::InvalidMovingFlag { .. } => None,
        }
    }
}

/// Maximum permitted absolute tick difference between two consecutive reads,
/// capped at the 50-tick bound qualified by the source demo.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PositionAgreementTicks(u16);

impl PositionAgreementTicks {
    pub const DEMO_QUALIFIED_MAXIMUM: Self = Self(50);

    pub const fn try_new(value: u16) -> Result<Self, PositionAgreementError> {
        if value <= Self::DEMO_QUALIFIED_MAXIMUM.get() {
            Ok(Self(value))
        } else {
            Err(PositionAgreementError::ToleranceOutOfRange { value })
        }
    }

    pub const fn get(self) -> u16 {
        self.0
    }
}

/// Present pose admitted only after two checksum-valid reads of the same exact
/// servo agree. Absolute rather than modular distance is deliberate: crossing
/// the encoder wrap during a lock attempt is rejected conservatively.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ValidatedPresentPosition {
    id: ServoId,
    first: PositionTicks,
    second: PositionTicks,
    admitted: PositionTicks,
}

impl ValidatedPresentPosition {
    pub fn try_from_pair(
        first: PresentPosition,
        second: PresentPosition,
        tolerance: PositionAgreementTicks,
    ) -> Result<Self, PositionAgreementError> {
        if first.id != second.id {
            return Err(PositionAgreementError::ServoIdMismatch {
                first: first.id,
                second: second.id,
            });
        }
        let difference = first.ticks.get().abs_diff(second.ticks.get());
        if difference > tolerance.get() {
            return Err(PositionAgreementError::ReadingsDisagree {
                id: first.id,
                first: first.ticks,
                second: second.ticks,
                difference,
                tolerance,
            });
        }
        Ok(Self {
            id: first.id,
            first: first.ticks,
            second: second.ticks,
            // The second read is the freshest complete wire observation. No
            // integer averaging can fabricate a position that was never read.
            admitted: second.ticks,
        })
    }

    pub const fn id(self) -> ServoId {
        self.id
    }

    pub const fn first(self) -> PositionTicks {
        self.first
    }

    pub const fn second(self) -> PositionTicks {
        self.second
    }

    pub const fn admitted(self) -> PositionTicks {
        self.admitted
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PositionAgreementError {
    ToleranceOutOfRange {
        value: u16,
    },
    ServoIdMismatch {
        first: ServoId,
        second: ServoId,
    },
    ReadingsDisagree {
        id: ServoId,
        first: PositionTicks,
        second: PositionTicks,
        difference: u16,
        tolerance: PositionAgreementTicks,
    },
}

impl fmt::Display for PositionAgreementError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "present-position qualification failed: {self:?}")
    }
}

impl core::error::Error for PositionAgreementError {}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;

    fn id(value: u8) -> ServoId {
        ServoId::try_new(value).expect("test servo ID")
    }

    fn status(id: ServoId, parameters: &[u8]) -> std::vec::Vec<u8> {
        let mut bytes = std::vec![0xff, 0xff, id.get(), 0, 0];
        bytes[3] = u8::try_from(parameters.len() + 2).expect("parameter count");
        bytes.extend_from_slice(parameters);
        let checksum = !bytes[2..]
            .iter()
            .fold(0_u8, |sum, byte| sum.wrapping_add(*byte));
        bytes.push(checksum);
        bytes
    }

    fn position(id: ServoId, value: u16) -> PresentPosition {
        PresentPosition::parse(&status(id, &value.to_le_bytes()), id).expect("position response")
    }

    #[test]
    fn full_window_is_parsed_once_without_inventing_units() {
        let servo = id(4);
        let data = [
            0x34, 0x02, // position = 564
            0x78, 0x56, // speed raw
            0xbc, 0x9a, // load raw
            119, 31, // voltage/temp raw
            7, 8, 1, // async/status/moving
            0xde, 0xad, // registers 67/68
            0xef, 0xbe, // current raw
        ];
        let telemetry = FullTelemetry::parse(&status(servo, &data), servo).expect("telemetry");
        assert_eq!(telemetry.position().get(), 0x0234);
        assert_eq!(telemetry.speed_raw(), 0x5678);
        assert_eq!(telemetry.load_raw(), 0x9abc);
        assert_eq!(telemetry.voltage_raw(), 119);
        assert_eq!(telemetry.temperature_raw(), 31);
        assert_eq!(telemetry.async_write_flag_raw(), 7);
        assert_eq!(telemetry.device_status_raw(), 8);
        assert!(telemetry.is_moving());
        assert_eq!(telemetry.registers_67_68_raw(), 0xadde);
        assert_eq!(telemetry.current_raw(), 0xbeef);
    }

    #[test]
    fn invalid_position_and_boolean_domain_are_rejected() {
        let servo = id(1);
        assert!(matches!(
            PresentPosition::parse(&status(servo, &4096_u16.to_le_bytes()), servo),
            Err(TelemetryParseError::PositionOutOfRange { raw: 4096, .. })
        ));

        let mut data = [0_u8; 15];
        data[10] = 2;
        assert!(matches!(
            FullTelemetry::parse(&status(servo, &data), servo),
            Err(TelemetryParseError::InvalidMovingFlag { value: 2, .. })
        ));
    }

    #[test]
    fn agreement_uses_two_exact_reads_and_freshest_value() {
        let servo = id(3);
        let qualified = ValidatedPresentPosition::try_from_pair(
            position(servo, 2_900),
            position(servo, 2_925),
            PositionAgreementTicks::DEMO_QUALIFIED_MAXIMUM,
        )
        .expect("agreeing reads");
        assert_eq!(qualified.first().get(), 2_900);
        assert_eq!(qualified.second().get(), 2_925);
        assert_eq!(qualified.admitted().get(), 2_925);

        assert!(matches!(
            ValidatedPresentPosition::try_from_pair(
                position(servo, 4_095),
                position(servo, 0),
                PositionAgreementTicks::DEMO_QUALIFIED_MAXIMUM,
            ),
            Err(PositionAgreementError::ReadingsDisagree {
                difference: 4095,
                ..
            })
        ));
        assert!(PositionAgreementTicks::try_new(51).is_err());
    }
}
