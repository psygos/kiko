//! Deterministic in-memory STS plant used by head-runtime incident tests.
//!
//! Unlike a byte replayer, this transport parses the real request frames,
//! evolves joint state in explicit tick/second and nanosecond units, and
//! synthesizes checksum-valid register responses. It intentionally remains a
//! test-only model: its constants are fixtures, not claims about Kiko's
//! unmeasured physical dynamics.

use std::collections::VecDeque;
use std::io;
use std::num::NonZeroU16;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use kiko_head_protocol::{
    GoalSpeedTicksPerSecond, HeadJoint, PositionTicks, ServoId, TorqueLimitPermille,
};

use crate::{
    AsyncByteTransport, MonotonicClock, MonotonicTime, TransportFailure, TransportOperation,
};

const JOINT_COUNT: usize = 4;
const NANOS_PER_SECOND: i128 = 1_000_000_000;
const INSTRUCTION_READ: u8 = 0x02;
const INSTRUCTION_WRITE: u8 = 0x03;
const TORQUE_SWITCH_REGISTER: u8 = 40;
const GOAL_POSITION_REGISTER: u8 = 42;
const TORQUE_LIMIT_REGISTER: u8 = 48;
const PRESENT_POSITION_REGISTER: u8 = 56;
const FULL_TELEMETRY_BYTES: u8 = 15;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ServoPlantConfigInput {
    pub initial_ticks: [u16; JOINT_COUNT],
    pub gravity_equilibrium_ticks: [u16; JOINT_COUNT],
    pub hold_floor_permille: [u16; JOINT_COUNT],
    pub initial_torque_limit_permille: [u16; JOINT_COUNT],
    pub initial_torque_enabled: [bool; JOINT_COUNT],
    pub gravity_drift_ticks_per_second: u16,
    pub io_latency_microseconds: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ServoPlantConfig {
    initial: [PositionTicks; JOINT_COUNT],
    gravity_equilibrium: [PositionTicks; JOINT_COUNT],
    hold_floor: [TorqueLimitPermille; JOINT_COUNT],
    initial_torque_limit: [TorqueLimitPermille; JOINT_COUNT],
    initial_torque_enabled: [bool; JOINT_COUNT],
    gravity_drift_ticks_per_second: NonZeroU16,
    io_latency: Duration,
}

impl ServoPlantConfig {
    pub(crate) fn parse(input: ServoPlantConfigInput) -> Result<Self, ServoPlantConfigError> {
        let initial = parse_positions("initial_ticks", input.initial_ticks)?;
        let gravity_equilibrium =
            parse_positions("gravity_equilibrium_ticks", input.gravity_equilibrium_ticks)?;
        let hold_floor = parse_torque("hold_floor_permille", input.hold_floor_permille)?;
        let initial_torque_limit = parse_torque(
            "initial_torque_limit_permille",
            input.initial_torque_limit_permille,
        )?;
        let gravity_drift_ticks_per_second = NonZeroU16::new(input.gravity_drift_ticks_per_second)
            .ok_or(ServoPlantConfigError::ZeroGravityDriftTicksPerSecond)?;
        if input.io_latency_microseconds == 0 {
            return Err(ServoPlantConfigError::ZeroIoLatency);
        }
        let io_latency = Duration::from_micros(input.io_latency_microseconds);
        Ok(Self {
            initial,
            gravity_equilibrium,
            hold_floor,
            initial_torque_limit,
            initial_torque_enabled: input.initial_torque_enabled,
            gravity_drift_ticks_per_second,
            io_latency,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ServoPlantConfigError {
    Position {
        field: &'static str,
        joint: HeadJoint,
        value: u16,
    },
    Torque {
        field: &'static str,
        joint: HeadJoint,
        value: u16,
    },
    ZeroGravityDriftTicksPerSecond,
    ZeroIoLatency,
}

fn parse_positions(
    field: &'static str,
    raw: [u16; JOINT_COUNT],
) -> Result<[PositionTicks; JOINT_COUNT], ServoPlantConfigError> {
    let mut parsed = [PositionTicks::MIN; JOINT_COUNT];
    for (index, joint) in HeadJoint::ALL.into_iter().enumerate() {
        parsed[index] =
            PositionTicks::try_new(raw[index]).map_err(|_| ServoPlantConfigError::Position {
                field,
                joint,
                value: raw[index],
            })?;
    }
    Ok(parsed)
}

fn parse_torque(
    field: &'static str,
    raw: [u16; JOINT_COUNT],
) -> Result<[TorqueLimitPermille; JOINT_COUNT], ServoPlantConfigError> {
    let fallback = TorqueLimitPermille::try_new(1).expect("one permille is representable");
    let mut parsed = [fallback; JOINT_COUNT];
    for (index, joint) in HeadJoint::ALL.into_iter().enumerate() {
        parsed[index] = TorqueLimitPermille::try_new(raw[index]).map_err(|_| {
            ServoPlantConfigError::Torque {
                field,
                joint,
                value: raw[index],
            }
        })?;
    }
    Ok(parsed)
}

#[derive(Clone, Default)]
pub(crate) struct ServoPlantClock {
    elapsed: Arc<Mutex<Duration>>,
}

impl ServoPlantClock {
    pub(crate) fn advance(&self, duration: Duration) {
        let mut elapsed = self.elapsed.lock().expect("servo plant clock");
        *elapsed = elapsed.checked_add(duration).unwrap_or(Duration::MAX);
    }
}

impl MonotonicClock for ServoPlantClock {
    fn now(&self) -> MonotonicTime {
        MonotonicTime::from_duration_since_origin(*self.elapsed.lock().expect("servo plant clock"))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ServoPlantJointSnapshot {
    pub position: PositionTicks,
    pub goal: PositionTicks,
    pub torque_limit: TorqueLimitPermille,
    pub torque_enabled: bool,
    pub jammed: bool,
    pub moving: bool,
}

#[derive(Clone)]
pub(crate) struct ServoPlantProbe {
    shared: Arc<Mutex<ServoPlantState>>,
    clock: ServoPlantClock,
}

impl ServoPlantProbe {
    pub(crate) fn snapshot(&self) -> [ServoPlantJointSnapshot; JOINT_COUNT] {
        let now = self.clock.now();
        let mut state = self.shared.lock().expect("servo plant state");
        state.advance(now);
        state.joints.map(|joint| ServoPlantJointSnapshot {
            position: joint.position(),
            goal: joint.goal,
            torque_limit: joint.torque_limit,
            torque_enabled: joint.torque_enabled,
            jammed: joint.jammed,
            moving: joint.moving,
        })
    }

    pub(crate) fn displace(&self, joint: HeadJoint, position: PositionTicks) {
        let now = self.clock.now();
        let mut state = self.shared.lock().expect("servo plant state");
        state.advance(now);
        state.joints[joint as usize].position_nano_ticks =
            i128::from(position.get()) * NANOS_PER_SECOND;
        state.joints[joint as usize].moving = false;
    }

    pub(crate) fn set_jammed(&self, joint: HeadJoint, jammed: bool) {
        let now = self.clock.now();
        let mut state = self.shared.lock().expect("servo plant state");
        state.advance(now);
        state.joints[joint as usize].jammed = jammed;
    }

    pub(crate) fn set_temperature_raw(&self, joint: HeadJoint, raw: u8) {
        self.shared.lock().expect("servo plant state").joints[joint as usize].temperature_raw = raw;
    }
}

pub(crate) struct ServoPlantTransport {
    shared: Arc<Mutex<ServoPlantState>>,
    clock: ServoPlantClock,
    pending: VecDeque<u8>,
    io_latency: Duration,
}

impl ServoPlantTransport {
    pub(crate) fn new(clock: ServoPlantClock, config: ServoPlantConfig) -> (Self, ServoPlantProbe) {
        let state = Arc::new(Mutex::new(ServoPlantState::new(config, clock.now())));
        let probe = ServoPlantProbe {
            shared: Arc::clone(&state),
            clock: clock.clone(),
        };
        (
            Self {
                shared: state,
                clock,
                pending: VecDeque::new(),
                io_latency: config.io_latency,
            },
            probe,
        )
    }

    fn handle(&mut self, bytes: &[u8]) -> Result<(), PlantCommandParseError> {
        if !self.pending.is_empty() {
            return Err(PlantCommandParseError::ResponseNotConsumed);
        }
        let command = PlantCommand::parse(bytes)?;
        let now = self.clock.now();
        let response = self
            .shared
            .lock()
            .expect("servo plant state")
            .apply(now, command);
        if let Some(response) = response {
            self.pending.extend(response);
        }
        Ok(())
    }
}

impl AsyncByteTransport for ServoPlantTransport {
    async fn write_all(
        &mut self,
        bytes: &[u8],
        _timeout: Duration,
    ) -> Result<(), TransportFailure> {
        self.clock.advance(self.io_latency);
        self.handle(bytes).map_err(invalid_command_transport)
    }

    async fn read_some(
        &mut self,
        bytes: &mut [u8],
        _timeout: Duration,
    ) -> Result<usize, TransportFailure> {
        self.clock.advance(self.io_latency);
        if self.pending.is_empty() {
            return Err(TransportFailure::timed_out(TransportOperation::Read, 0));
        }
        let count = bytes.len().min(self.pending.len());
        for destination in &mut bytes[..count] {
            *destination = self.pending.pop_front().expect("bounded pending response");
        }
        Ok(count)
    }
}

fn invalid_command_transport(source: PlantCommandParseError) -> TransportFailure {
    TransportFailure::from_io(
        TransportOperation::Write,
        &io::Error::new(io::ErrorKind::InvalidData, format!("{source:?}")),
        0,
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PlantRead {
    PresentPosition,
    GoalPosition,
    FullTelemetry,
    TorqueSwitch,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PlantCommand {
    Read {
        joint: HeadJoint,
        register: PlantRead,
    },
    Goal {
        joint: HeadJoint,
        position: PositionTicks,
        speed: GoalSpeedTicksPerSecond,
    },
    TorqueSwitch {
        joint: HeadJoint,
        enabled: bool,
    },
    TorqueLimit {
        joint: HeadJoint,
        limit: TorqueLimitPermille,
    },
}

impl PlantCommand {
    fn parse(bytes: &[u8]) -> Result<Self, PlantCommandParseError> {
        if bytes.len() < 6 {
            return Err(PlantCommandParseError::TooShort(bytes.len()));
        }
        if bytes[..2] != [0xff, 0xff] {
            return Err(PlantCommandParseError::Header([bytes[0], bytes[1]]));
        }
        let declared = usize::from(bytes[3]) + 4;
        if declared != bytes.len() {
            return Err(PlantCommandParseError::Length {
                declared,
                actual: bytes.len(),
            });
        }
        let stored = bytes[bytes.len() - 1];
        let computed = checksum(&bytes[2..bytes.len() - 1]);
        if stored != computed {
            return Err(PlantCommandParseError::Checksum { stored, computed });
        }
        let joint = joint_from_id(bytes[2])?;
        let parameters = &bytes[5..bytes.len() - 1];
        match (bytes[4], parameters) {
            (INSTRUCTION_READ, [PRESENT_POSITION_REGISTER, 2]) => Ok(Self::Read {
                joint,
                register: PlantRead::PresentPosition,
            }),
            (INSTRUCTION_READ, [GOAL_POSITION_REGISTER, 2]) => Ok(Self::Read {
                joint,
                register: PlantRead::GoalPosition,
            }),
            (INSTRUCTION_READ, [PRESENT_POSITION_REGISTER, FULL_TELEMETRY_BYTES]) => {
                Ok(Self::Read {
                    joint,
                    register: PlantRead::FullTelemetry,
                })
            }
            (INSTRUCTION_READ, [TORQUE_SWITCH_REGISTER, 1]) => Ok(Self::Read {
                joint,
                register: PlantRead::TorqueSwitch,
            }),
            (INSTRUCTION_WRITE, [GOAL_POSITION_REGISTER, p0, p1, 0, 0, s0, s1]) => {
                let position_raw = u16::from_le_bytes([*p0, *p1]);
                let speed_raw = u16::from_le_bytes([*s0, *s1]);
                let position = PositionTicks::try_new(position_raw)
                    .map_err(|_| PlantCommandParseError::Position(position_raw))?;
                let speed = GoalSpeedTicksPerSecond::try_new(speed_raw)
                    .map_err(|_| PlantCommandParseError::Speed(speed_raw))?;
                Ok(Self::Goal {
                    joint,
                    position,
                    speed,
                })
            }
            (INSTRUCTION_WRITE, [TORQUE_SWITCH_REGISTER, raw @ (0 | 1)]) => {
                Ok(Self::TorqueSwitch {
                    joint,
                    enabled: *raw == 1,
                })
            }
            (INSTRUCTION_WRITE, [TORQUE_LIMIT_REGISTER, low, high]) => {
                let raw = u16::from_le_bytes([*low, *high]);
                let limit = TorqueLimitPermille::try_new(raw)
                    .map_err(|_| PlantCommandParseError::Torque(raw))?;
                Ok(Self::TorqueLimit { joint, limit })
            }
            (instruction, _) => Err(PlantCommandParseError::Unsupported {
                instruction,
                parameters: parameters.to_vec(),
            }),
        }
    }
}

fn joint_from_id(raw: u8) -> Result<HeadJoint, PlantCommandParseError> {
    HeadJoint::ALL
        .into_iter()
        .find(|joint| joint.servo_id().get() == raw)
        .ok_or(PlantCommandParseError::ServoId(raw))
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum PlantCommandParseError {
    TooShort(usize),
    Header([u8; 2]),
    Length {
        declared: usize,
        actual: usize,
    },
    Checksum {
        stored: u8,
        computed: u8,
    },
    ServoId(u8),
    Position(u16),
    Speed(u16),
    Torque(u16),
    Unsupported {
        instruction: u8,
        parameters: Vec<u8>,
    },
    ResponseNotConsumed,
}

#[derive(Clone, Copy, Debug)]
struct ServoPlantJoint {
    position_nano_ticks: i128,
    goal: PositionTicks,
    goal_speed: GoalSpeedTicksPerSecond,
    gravity_equilibrium: PositionTicks,
    gravity_drift_ticks_per_second: NonZeroU16,
    hold_floor: TorqueLimitPermille,
    torque_limit: TorqueLimitPermille,
    torque_enabled: bool,
    jammed: bool,
    moving: bool,
    speed_raw: u16,
    voltage_raw: u8,
    temperature_raw: u8,
    device_status_raw: u8,
}

impl ServoPlantJoint {
    fn position(self) -> PositionTicks {
        let whole = (self.position_nano_ticks / NANOS_PER_SECOND).clamp(
            i128::from(PositionTicks::MIN.get()),
            i128::from(PositionTicks::MAX.get()),
        );
        PositionTicks::try_new(u16::try_from(whole).expect("clamped encoder position"))
            .expect("clamped encoder position is representable")
    }

    fn advance(&mut self, elapsed: Duration) {
        self.moving = false;
        self.speed_raw = 0;
        if elapsed.is_zero() || self.jammed {
            return;
        }
        let (target, rate_ticks_per_second) =
            if self.torque_enabled && self.torque_limit >= self.hold_floor {
                let scaled =
                    u32::from(self.goal_speed.get()) * u32::from(self.torque_limit.get()) / 1_000;
                (self.goal, scaled.max(1))
            } else {
                let rate = if self.torque_enabled {
                    let deficit = self
                        .hold_floor
                        .get()
                        .saturating_sub(self.torque_limit.get());
                    u32::from(self.gravity_drift_ticks_per_second.get()) * u32::from(deficit)
                        / u32::from(self.hold_floor.get())
                } else {
                    u32::from(self.gravity_drift_ticks_per_second.get())
                };
                (self.gravity_equilibrium, rate)
            };
        if rate_ticks_per_second == 0 {
            return;
        }

        let target_nano_ticks = i128::from(target.get()) * NANOS_PER_SECOND;
        let delta = target_nano_ticks - self.position_nano_ticks;
        if delta == 0 {
            return;
        }
        let elapsed_ns = i128::try_from(elapsed.as_nanos()).unwrap_or(i128::MAX);
        let maximum_step = i128::from(rate_ticks_per_second).saturating_mul(elapsed_ns);
        let step = delta.unsigned_abs().min(maximum_step.unsigned_abs());
        let signed_step = i128::try_from(step).unwrap_or(i128::MAX) * delta.signum();
        self.position_nano_ticks = self.position_nano_ticks.saturating_add(signed_step);
        self.moving = signed_step != 0;
        self.speed_raw = u16::try_from(rate_ticks_per_second).unwrap_or(u16::MAX);
    }

    fn full_telemetry(self) -> [u8; FULL_TELEMETRY_BYTES as usize] {
        let mut raw = [0_u8; FULL_TELEMETRY_BYTES as usize];
        raw[..2].copy_from_slice(&self.position().get().to_le_bytes());
        raw[2..4].copy_from_slice(&self.speed_raw.to_le_bytes());
        let goal_error = self.position().get().abs_diff(self.goal.get()).min(1_023);
        raw[4..6].copy_from_slice(&goal_error.to_le_bytes());
        raw[6] = self.voltage_raw;
        raw[7] = self.temperature_raw;
        raw[8] = 0;
        raw[9] = self.device_status_raw;
        raw[10] = u8::from(self.moving);
        raw[11..13].copy_from_slice(&0_u16.to_le_bytes());
        let current = if self.torque_enabled {
            self.torque_limit.get()
        } else {
            0
        };
        raw[13..15].copy_from_slice(&current.to_le_bytes());
        raw
    }
}

struct ServoPlantState {
    joints: [ServoPlantJoint; JOINT_COUNT],
    advanced_at: MonotonicTime,
}

impl ServoPlantState {
    fn new(config: ServoPlantConfig, started_at: MonotonicTime) -> Self {
        let joints = std::array::from_fn(|index| ServoPlantJoint {
            position_nano_ticks: i128::from(config.initial[index].get()) * NANOS_PER_SECOND,
            goal: config.initial[index],
            goal_speed: GoalSpeedTicksPerSecond::try_new(1).expect("one tick per second"),
            gravity_equilibrium: config.gravity_equilibrium[index],
            gravity_drift_ticks_per_second: config.gravity_drift_ticks_per_second,
            hold_floor: config.hold_floor[index],
            torque_limit: config.initial_torque_limit[index],
            torque_enabled: config.initial_torque_enabled[index],
            jammed: false,
            moving: false,
            speed_raw: 0,
            voltage_raw: 120,
            temperature_raw: 30,
            device_status_raw: 0,
        });
        Self {
            joints,
            advanced_at: started_at,
        }
    }

    fn advance(&mut self, now: MonotonicTime) {
        let elapsed = now
            .checked_duration_since(self.advanced_at)
            .expect("the deterministic plant clock never regresses");
        for joint in &mut self.joints {
            joint.advance(elapsed);
        }
        self.advanced_at = now;
    }

    fn apply(&mut self, now: MonotonicTime, command: PlantCommand) -> Option<Vec<u8>> {
        self.advance(now);
        match command {
            PlantCommand::Read { joint, register } => {
                let servo = self.joints[joint as usize];
                let parameters = match register {
                    PlantRead::PresentPosition => servo.position().get().to_le_bytes().to_vec(),
                    PlantRead::GoalPosition => servo.goal.get().to_le_bytes().to_vec(),
                    PlantRead::FullTelemetry => servo.full_telemetry().to_vec(),
                    PlantRead::TorqueSwitch => vec![u8::from(servo.torque_enabled)],
                };
                Some(status_response(joint.servo_id(), &parameters))
            }
            PlantCommand::Goal {
                joint,
                position,
                speed,
            } => {
                let servo = &mut self.joints[joint as usize];
                servo.goal = position;
                servo.goal_speed = speed;
                None
            }
            PlantCommand::TorqueSwitch { joint, enabled } => {
                self.joints[joint as usize].torque_enabled = enabled;
                None
            }
            PlantCommand::TorqueLimit { joint, limit } => {
                self.joints[joint as usize].torque_limit = limit;
                None
            }
        }
    }
}

fn status_response(id: ServoId, parameters: &[u8]) -> Vec<u8> {
    let mut bytes = vec![0xff, 0xff, id.get(), 0, 0];
    bytes[3] = u8::try_from(parameters.len() + 2).expect("plant response length fits u8");
    bytes.extend_from_slice(parameters);
    let checksum = checksum(&bytes[2..]);
    bytes.push(checksum);
    bytes
}

fn checksum(payload: &[u8]) -> u8 {
    !payload
        .iter()
        .fold(0_u8, |sum, byte| sum.wrapping_add(*byte))
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiko_head_protocol::{
        FullTelemetry, GoalPositionObservation, TorqueSwitch, TorqueSwitchObservation,
        build_full_telemetry_read, build_goal_position_read, build_goal_with_speed_write,
        build_torque_limit_write, build_torque_switch_read, build_torque_switch_write,
    };

    fn config() -> ServoPlantConfig {
        ServoPlantConfig::parse(ServoPlantConfigInput {
            initial_ticks: [1_500, 3_000, 2_000, 3_100],
            gravity_equilibrium_ticks: [1_300, 3_200, 2_000, 3_100],
            hold_floor_permille: [300, 200, 150, 150],
            initial_torque_limit_permille: [650, 550, 400, 400],
            initial_torque_enabled: [true; JOINT_COUNT],
            gravity_drift_ticks_per_second: 40,
            io_latency_microseconds: 100,
        })
        .expect("plant fixture")
    }

    async fn request(
        transport: &mut ServoPlantTransport,
        frame: &[u8],
        response: &mut [u8],
    ) -> usize {
        transport
            .write_all(frame, Duration::from_millis(10))
            .await
            .expect("plant request");
        transport
            .read_some(response, Duration::from_millis(10))
            .await
            .expect("plant response")
    }

    #[test]
    fn config_parses_every_weak_numeric_boundary_once() {
        let mut raw = ServoPlantConfigInput {
            initial_ticks: [1_500; JOINT_COUNT],
            gravity_equilibrium_ticks: [1_500; JOINT_COUNT],
            hold_floor_permille: [300; JOINT_COUNT],
            initial_torque_limit_permille: [400; JOINT_COUNT],
            initial_torque_enabled: [true; JOINT_COUNT],
            gravity_drift_ticks_per_second: 40,
            io_latency_microseconds: 100,
        };
        raw.initial_ticks[2] = 4_096;
        assert!(matches!(
            ServoPlantConfig::parse(raw.clone()),
            Err(ServoPlantConfigError::Position {
                field: "initial_ticks",
                joint: HeadJoint::Yaw,
                value: 4_096,
            })
        ));
        raw.initial_ticks[2] = 1_500;
        raw.hold_floor_permille[1] = 0;
        assert!(matches!(
            ServoPlantConfig::parse(raw.clone()),
            Err(ServoPlantConfigError::Torque {
                field: "hold_floor_permille",
                joint: HeadJoint::Curl,
                value: 0,
            })
        ));
        raw.hold_floor_permille[1] = 300;
        raw.gravity_drift_ticks_per_second = 0;
        assert_eq!(
            ServoPlantConfig::parse(raw.clone()),
            Err(ServoPlantConfigError::ZeroGravityDriftTicksPerSecond)
        );
        raw.gravity_drift_ticks_per_second = 40;
        raw.io_latency_microseconds = 0;
        assert_eq!(
            ServoPlantConfig::parse(raw),
            Err(ServoPlantConfigError::ZeroIoLatency)
        );
    }

    #[tokio::test]
    async fn real_protocol_frames_round_trip_through_typed_registers() {
        let clock = ServoPlantClock::default();
        let (mut transport, _probe) = ServoPlantTransport::new(clock, config());
        let joint = HeadJoint::Bow;

        let goal = PositionTicks::try_new(1_620).unwrap();
        let speed = GoalSpeedTicksPerSecond::try_new(120).unwrap();
        transport
            .write_all(
                build_goal_with_speed_write(joint.servo_id(), goal, speed).as_bytes(),
                Duration::from_millis(10),
            )
            .await
            .unwrap();
        let mut response = [0_u8; 32];
        let read = request(
            &mut transport,
            build_goal_position_read(joint.servo_id()).as_bytes(),
            &mut response,
        )
        .await;
        assert_eq!(
            GoalPositionObservation::parse(&response[..read], joint.servo_id())
                .unwrap()
                .ticks(),
            goal
        );

        transport
            .write_all(
                build_torque_switch_write(joint.servo_id(), TorqueSwitch::Disabled).as_bytes(),
                Duration::from_millis(10),
            )
            .await
            .unwrap();
        let read = request(
            &mut transport,
            build_torque_switch_read(joint.servo_id()).as_bytes(),
            &mut response,
        )
        .await;
        assert_eq!(
            TorqueSwitchObservation::parse(&response[..read], joint.servo_id())
                .unwrap()
                .state()
                .raw(),
            0
        );
    }

    #[tokio::test]
    async fn integer_motion_respects_speed_time_and_never_crosses_goal() {
        let clock = ServoPlantClock::default();
        let (mut transport, probe) = ServoPlantTransport::new(clock.clone(), config());
        let joint = HeadJoint::Bow;
        let goal = PositionTicks::try_new(1_620).unwrap();
        let speed = GoalSpeedTicksPerSecond::try_new(120).unwrap();
        transport
            .write_all(
                build_goal_with_speed_write(joint.servo_id(), goal, speed).as_bytes(),
                Duration::from_millis(10),
            )
            .await
            .unwrap();

        // 650 permille authority deliberately scales 120 t/s to 78 t/s.
        clock.advance(Duration::from_millis(500));
        assert_eq!(probe.snapshot()[joint as usize].position.get(), 1_539);
        clock.advance(Duration::from_secs(10));
        assert_eq!(probe.snapshot()[joint as usize].position, goal);
    }

    #[tokio::test]
    async fn sub_floor_torque_drifts_toward_gravity_but_floor_holds() {
        let clock = ServoPlantClock::default();
        let (mut transport, probe) = ServoPlantTransport::new(clock.clone(), config());
        let joint = HeadJoint::Bow;
        let floor = TorqueLimitPermille::try_new(300).unwrap();
        transport
            .write_all(
                build_torque_limit_write(joint.servo_id(), floor).as_bytes(),
                Duration::from_millis(10),
            )
            .await
            .unwrap();
        clock.advance(Duration::from_secs(1));
        assert_eq!(probe.snapshot()[joint as usize].position.get(), 1_500);

        let below = TorqueLimitPermille::try_new(150).unwrap();
        transport
            .write_all(
                build_torque_limit_write(joint.servo_id(), below).as_bytes(),
                Duration::from_millis(10),
            )
            .await
            .unwrap();
        clock.advance(Duration::from_secs(1));
        assert_eq!(probe.snapshot()[joint as usize].position.get(), 1_480);
    }

    #[tokio::test]
    async fn telemetry_reflects_disturbance_jam_and_temperature_without_fabrication() {
        let clock = ServoPlantClock::default();
        let (mut transport, probe) = ServoPlantTransport::new(clock, config());
        let joint = HeadJoint::Curl;
        probe.displace(joint, PositionTicks::try_new(3_050).unwrap());
        probe.set_jammed(joint, true);
        probe.set_temperature_raw(joint, 66);

        let mut response = [0_u8; 32];
        let read = request(
            &mut transport,
            build_full_telemetry_read(joint.servo_id()).as_bytes(),
            &mut response,
        )
        .await;
        let telemetry = FullTelemetry::parse(&response[..read], joint.servo_id()).unwrap();
        assert_eq!(telemetry.position().get(), 3_050);
        assert_eq!(telemetry.temperature_raw(), 66);
        assert!(!telemetry.is_moving());
    }

    #[test]
    fn parser_rejects_corrupt_checksum_and_unsupported_frames_before_mutation() {
        let joint = HeadJoint::Yaw;
        let mut frame = build_goal_position_read(joint.servo_id())
            .as_bytes()
            .to_vec();
        *frame.last_mut().unwrap() ^= 1;
        assert!(matches!(
            PlantCommand::parse(&frame),
            Err(PlantCommandParseError::Checksum { .. })
        ));

        let unknown = [0xff, 0xff, 3, 4, 2, 99, 1, 0];
        let mut unknown = unknown.to_vec();
        let last = unknown.len() - 1;
        unknown[last] = checksum(&unknown[2..last]);
        assert!(matches!(
            PlantCommand::parse(&unknown),
            Err(PlantCommandParseError::Unsupported { .. })
        ));
    }
}
