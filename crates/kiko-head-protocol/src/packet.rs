use core::fmt;
use core::num::{NonZeroU8, NonZeroU16};

const HEADER: [u8; 2] = [0xff, 0xff];
const INSTRUCTION_PING: u8 = 0x01;
const INSTRUCTION_READ: u8 = 0x02;
const INSTRUCTION_WRITE: u8 = 0x03;

const PRESENT_POSITION_REGISTER: u8 = 56;
const FULL_TELEMETRY_REGISTER: u8 = 56;
pub(crate) const FULL_TELEMETRY_BYTES: u8 = 15;
const TORQUE_SWITCH_REGISTER: u8 = 40;
const GOAL_POSITION_REGISTER: u8 = 42;
const TORQUE_LIMIT_REGISTER: u8 = 48;

const MAX_EXACT_SERVO_ID: u8 = 253;
const POSITION_TICKS_PER_REVOLUTION: u16 = 4096;
const MAX_GOAL_SPEED_TICKS_PER_SECOND: u16 = 32_766;
const MAX_TORQUE_PERMILLE: u16 = 1_000;
const COMMAND_CAPACITY: usize = 16;

/// One exact, non-broadcast STS servo identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ServoId(NonZeroU8);

impl ServoId {
    pub const fn try_new(value: u8) -> Result<Self, FrameBuildError> {
        match NonZeroU8::new(value) {
            Some(value) if value.get() <= MAX_EXACT_SERVO_ID => Ok(Self(value)),
            _ => Err(FrameBuildError::InvalidServoId { value }),
        }
    }

    pub const fn get(self) -> u8 {
        self.0.get()
    }

    pub(crate) const fn known(value: u8) -> Self {
        match Self::try_new(value) {
            Ok(id) => id,
            Err(_) => panic!("known Kiko servo IDs are exact non-broadcast IDs"),
        }
    }
}

/// Absolute single-turn encoder position (`0..=4095` ticks).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PositionTicks(u16);

impl PositionTicks {
    pub const MIN: Self = Self(0);
    pub const MAX: Self = Self(POSITION_TICKS_PER_REVOLUTION - 1);

    pub const fn try_new(value: u16) -> Result<Self, FrameBuildError> {
        if value < POSITION_TICKS_PER_REVOLUTION {
            Ok(Self(value))
        } else {
            Err(FrameBuildError::PositionOutOfRange { value })
        }
    }

    pub const fn get(self) -> u16 {
        self.0
    }
}

/// Nonzero STS goal-speed magnitude. Zero means maximum speed on this device
/// family and is therefore unrepresentable here.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GoalSpeedTicksPerSecond(NonZeroU16);

impl GoalSpeedTicksPerSecond {
    pub const fn try_new(value: u16) -> Result<Self, FrameBuildError> {
        match NonZeroU16::new(value) {
            Some(value) if value.get() <= MAX_GOAL_SPEED_TICKS_PER_SECOND => Ok(Self(value)),
            _ => Err(FrameBuildError::GoalSpeedOutOfRange { value }),
        }
    }

    pub const fn get(self) -> u16 {
        self.0.get()
    }
}

/// Explicit STS torque-switch value. The dangerous calibration value `128`
/// cannot be constructed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum TorqueSwitch {
    Disabled = 0,
    Enabled = 1,
}

/// Output-torque clamp in permille (`1..=1000`). Disabling torque is a
/// separate, explicit operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TorqueLimitPermille(NonZeroU16);

impl TorqueLimitPermille {
    pub const fn try_new(value: u16) -> Result<Self, FrameBuildError> {
        match NonZeroU16::new(value) {
            Some(value) if value.get() <= MAX_TORQUE_PERMILLE => Ok(Self(value)),
            _ => Err(FrameBuildError::TorqueLimitOutOfRange { value }),
        }
    }

    pub const fn get(self) -> u16 {
        self.0.get()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameBuildError {
    InvalidServoId { value: u8 },
    PositionOutOfRange { value: u16 },
    GoalSpeedOutOfRange { value: u16 },
    TorqueLimitOutOfRange { value: u16 },
}

impl fmt::Display for FrameBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid typed STS request: {self:?}")
    }
}

impl core::error::Error for FrameBuildError {}

/// Allocation-free exact wire frame. Only the initialized prefix is exposed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommandFrame {
    bytes: [u8; COMMAND_CAPACITY],
    len: u8,
}

impl CommandFrame {
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes[..usize::from(self.len)]
    }
}

pub fn build_ping(id: ServoId) -> CommandFrame {
    build_command(id, INSTRUCTION_PING, &[])
}

pub fn build_position_read(id: ServoId) -> CommandFrame {
    build_command(id, INSTRUCTION_READ, &[PRESENT_POSITION_REGISTER, 2])
}

/// Read the exact two-byte goal-position register without changing it.
///
/// Installed Kiko servos do not acknowledge goal writes at their configured
/// response level. A caller which needs stronger evidence than host-write
/// completion must issue this read and parse the result as a goal observation.
pub fn build_goal_position_read(id: ServoId) -> CommandFrame {
    build_command(id, INSTRUCTION_READ, &[GOAL_POSITION_REGISTER, 2])
}

pub fn build_full_telemetry_read(id: ServoId) -> CommandFrame {
    build_command(
        id,
        INSTRUCTION_READ,
        &[FULL_TELEMETRY_REGISTER, FULL_TELEMETRY_BYTES],
    )
}

/// Read the one-byte torque-switch register without changing servo state.
pub fn build_torque_switch_read(id: ServoId) -> CommandFrame {
    build_command(id, INSTRUCTION_READ, &[TORQUE_SWITCH_REGISTER, 1])
}

/// Build the only supported position write: goal, zero time, and a mandatory
/// nonzero speed clamp are sent atomically in one register span.
pub fn build_goal_with_speed_write(
    id: ServoId,
    position: PositionTicks,
    speed: GoalSpeedTicksPerSecond,
) -> CommandFrame {
    let position = position.get().to_le_bytes();
    let speed = speed.get().to_le_bytes();
    build_command(
        id,
        INSTRUCTION_WRITE,
        &[
            GOAL_POSITION_REGISTER,
            position[0],
            position[1],
            0,
            0,
            speed[0],
            speed[1],
        ],
    )
}

pub fn build_torque_switch_write(id: ServoId, switch: TorqueSwitch) -> CommandFrame {
    build_command(
        id,
        INSTRUCTION_WRITE,
        &[TORQUE_SWITCH_REGISTER, switch as u8],
    )
}

pub fn build_torque_limit_write(id: ServoId, limit: TorqueLimitPermille) -> CommandFrame {
    let limit = limit.get().to_le_bytes();
    build_command(
        id,
        INSTRUCTION_WRITE,
        &[TORQUE_LIMIT_REGISTER, limit[0], limit[1]],
    )
}

fn build_command(id: ServoId, instruction: u8, parameters: &[u8]) -> CommandFrame {
    let mut bytes = [0_u8; COMMAND_CAPACITY];
    let wire_len = parameters.len() + 6;
    debug_assert!(wire_len <= COMMAND_CAPACITY);
    let sts_length = u8::try_from(parameters.len() + 2)
        .expect("supported command parameter counts fit the STS length byte");
    bytes[..2].copy_from_slice(&HEADER);
    bytes[2] = id.get();
    bytes[3] = sts_length;
    bytes[4] = instruction;
    bytes[5..5 + parameters.len()].copy_from_slice(parameters);
    bytes[wire_len - 1] = checksum(&bytes[2..wire_len - 1]);
    CommandFrame {
        bytes,
        len: u8::try_from(wire_len).expect("command capacity fits u8"),
    }
}

/// Raw nonzero status bits reported by an STS response. Bit semantics vary by
/// firmware revision, so this crate preserves them without guessing labels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ServoStatus(NonZeroU8);

impl ServoStatus {
    pub const fn bits(self) -> u8 {
        self.0.get()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResponseParseError {
    TooShort {
        actual_bytes: usize,
    },
    HeaderMismatch {
        actual: [u8; 2],
    },
    ServoIdMismatch {
        expected: ServoId,
        actual: u8,
    },
    DeclaredLengthMismatch {
        declared_bytes: usize,
        actual_bytes: usize,
    },
    ParameterCountMismatch {
        expected: u8,
        actual: u8,
    },
    ChecksumMismatch {
        stored: u8,
        computed: u8,
    },
    DeviceStatus(ServoStatus),
}

impl fmt::Display for ResponseParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid STS status response: {self:?}")
    }
}

impl core::error::Error for ResponseParseError {}

/// Parse one already delimited response. Header, identity, declared length,
/// exact parameter count, checksum, and device status are checked before the
/// returned parameter slice becomes trusted boundary data.
pub fn parse_status_response(
    bytes: &[u8],
    expected_id: ServoId,
    expected_parameters: u8,
) -> Result<&[u8], ResponseParseError> {
    if bytes.len() < 6 {
        return Err(ResponseParseError::TooShort {
            actual_bytes: bytes.len(),
        });
    }
    let actual_header = [bytes[0], bytes[1]];
    if actual_header != HEADER {
        return Err(ResponseParseError::HeaderMismatch {
            actual: actual_header,
        });
    }
    if bytes[2] != expected_id.get() {
        return Err(ResponseParseError::ServoIdMismatch {
            expected: expected_id,
            actual: bytes[2],
        });
    }
    let declared_bytes = usize::from(bytes[3]) + 4;
    if declared_bytes != bytes.len() {
        return Err(ResponseParseError::DeclaredLengthMismatch {
            declared_bytes,
            actual_bytes: bytes.len(),
        });
    }
    let actual_parameters = bytes[3].saturating_sub(2);
    if actual_parameters != expected_parameters {
        return Err(ResponseParseError::ParameterCountMismatch {
            expected: expected_parameters,
            actual: actual_parameters,
        });
    }
    let stored = bytes[bytes.len() - 1];
    let computed = checksum(&bytes[2..bytes.len() - 1]);
    if stored != computed {
        return Err(ResponseParseError::ChecksumMismatch { stored, computed });
    }
    if let Some(status) = NonZeroU8::new(bytes[4]) {
        return Err(ResponseParseError::DeviceStatus(ServoStatus(status)));
    }
    Ok(&bytes[5..bytes.len() - 1])
}

fn checksum(payload: &[u8]) -> u8 {
    !payload
        .iter()
        .fold(0_u8, |sum, byte| sum.wrapping_add(*byte))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn torque_switch_read_is_an_exact_checksum_valid_read() {
        let id = ServoId::try_new(4).expect("exact test ID");
        let frame = build_torque_switch_read(id);

        assert_eq!(frame.as_bytes(), &[0xff, 0xff, 4, 4, 2, 40, 1, 204]);
    }

    fn id(value: u8) -> ServoId {
        ServoId::try_new(value).expect("test ID")
    }

    fn status(id: ServoId, device_status: u8, parameters: &[u8]) -> [u8; 21] {
        let mut frame = [0_u8; 21];
        let len = parameters.len() + 6;
        frame[..2].copy_from_slice(&HEADER);
        frame[2] = id.get();
        frame[3] = u8::try_from(parameters.len() + 2).expect("test length");
        frame[4] = device_status;
        frame[5..5 + parameters.len()].copy_from_slice(parameters);
        frame[len - 1] = checksum(&frame[2..len - 1]);
        frame
    }

    #[test]
    fn exact_known_wire_frames_are_stable() {
        assert_eq!(build_ping(id(1)).as_bytes(), &[0xff, 0xff, 1, 2, 1, 0xfb]);
        assert_eq!(
            build_position_read(id(3)).as_bytes(),
            &[0xff, 0xff, 3, 4, 2, 56, 2, 0xbc]
        );
        assert_eq!(
            build_goal_position_read(id(3)).as_bytes(),
            &[0xff, 0xff, 3, 4, 2, 42, 2, 0xca]
        );
        assert_eq!(
            build_full_telemetry_read(id(4)).as_bytes(),
            &[0xff, 0xff, 4, 4, 2, 56, 15, 0xae]
        );
        assert_eq!(
            build_goal_with_speed_write(
                id(2),
                PositionTicks::try_new(0x0abc).expect("position"),
                GoalSpeedTicksPerSecond::try_new(0x0123).expect("speed"),
            )
            .as_bytes(),
            &[0xff, 0xff, 2, 9, 3, 42, 0xbc, 0x0a, 0, 0, 0x23, 0x01, 0xdd]
        );
    }

    #[test]
    fn dangerous_or_ambiguous_values_are_unrepresentable() {
        assert!(ServoId::try_new(0).is_err());
        assert!(ServoId::try_new(254).is_err());
        assert!(PositionTicks::try_new(4096).is_err());
        assert!(GoalSpeedTicksPerSecond::try_new(0).is_err());
        assert!(GoalSpeedTicksPerSecond::try_new(32_767).is_err());
        assert!(TorqueLimitPermille::try_new(0).is_err());
        assert!(TorqueLimitPermille::try_new(1_001).is_err());
        assert_eq!(TorqueSwitch::Disabled as u8, 0);
        assert_eq!(TorqueSwitch::Enabled as u8, 1);
    }

    #[test]
    fn status_parser_rejects_every_untrusted_envelope_field() {
        let expected = id(2);
        let raw = status(expected, 0, &[0x34, 0x12]);
        assert_eq!(
            parse_status_response(&raw[..8], expected, 2).expect("valid status"),
            &[0x34, 0x12]
        );

        let mut corrupt = raw;
        corrupt[7] ^= 1;
        assert!(matches!(
            parse_status_response(&corrupt[..8], expected, 2),
            Err(ResponseParseError::ChecksumMismatch { .. })
        ));

        let wrong_length = status(expected, 0, &[0x34]);
        assert!(matches!(
            parse_status_response(&wrong_length[..7], expected, 2),
            Err(ResponseParseError::ParameterCountMismatch { .. })
        ));

        let device_fault = status(expected, 0x40, &[0x34, 0x12]);
        assert!(matches!(
            parse_status_response(&device_fault[..8], expected, 2),
            Err(ResponseParseError::DeviceStatus(status)) if status.bits() == 0x40
        ));
    }

    #[test]
    fn every_single_bit_corruption_and_nonexact_length_is_rejected() {
        let expected = id(2);
        let valid = status(expected, 0, &[0x34, 0x02]);
        let valid = &valid[..8];
        for byte_index in 0..valid.len() {
            for bit in 0..8 {
                let mut corrupt = [0_u8; 8];
                corrupt.copy_from_slice(valid);
                corrupt[byte_index] ^= 1 << bit;
                assert!(
                    parse_status_response(&corrupt, expected, 2).is_err(),
                    "accepted corruption at byte {byte_index}, bit {bit}"
                );
            }
        }
        for prefix_len in 0..valid.len() {
            assert!(parse_status_response(&valid[..prefix_len], expected, 2).is_err());
        }
        let mut suffixed = [0_u8; 9];
        suffixed[..8].copy_from_slice(valid);
        assert!(matches!(
            parse_status_response(&suffixed, expected, 2),
            Err(ResponseParseError::DeclaredLengthMismatch { .. })
        ));
    }
}
