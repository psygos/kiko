#![cfg_attr(target_arch = "arm", no_main)]
#![cfg_attr(target_arch = "arm", no_std)]
#![cfg_attr(not(target_arch = "arm"), allow(dead_code))]

//! Wheels-off-only STM32F446 motor commissioning image.
//!
//! This is deliberately a separate binary and Cargo feature from
//! `firmware_v2.rs`. It is not KRP2 motion authority, actuator calibration,
//! velocity calibration, closed-loop control, or MPC validation. Its only
//! physical action is exactly one explicitly selected finite recipe per reset:
//!
//! - `LeftThenRight250Ms`: zero 500 ms, left-forward 8% for 250 ms,
//!   zero 500 ms, right-forward 8% for 250 ms, then lock safe; or
//! - `BothForward10S`: zero 500 ms, both forward outputs at 8% for exactly
//!   10,000 nominal TIM5 milliseconds, then lock safe.
//!
//! A priority-0 TIM5 compare ISR is the nominal cutoff, with a post-enable
//! deadline re-sample and exception-path emergency MMIO. The 250 ms recipe
//! retains its nominal 500 ms independent-watchdog reset backstop. Immediately
//! before the 10,000 ms recipe starts, the watchdog is configured to 15,500 ms
//! under the HAL's nominal 32 kHz LSI model, and is never fed while outputs are
//! active. That conservative configuration keeps the reset later than 10,500
//! ms even at the STM32F446 datasheet's 47 kHz maximum LSI frequency; its real
//! interval can be much longer at lower LSI frequencies. The 500 ms watchdog
//! configuration is restored immediately after outputs are disabled.
//! These layers do **not** prove an unconditional physical cutoff under
//! arbitrary CPU, clock, timer, watchdog, driver, or wiring fault. A watchdog
//! reset returns to disabled boot state and requires a new typed trigger.
//!
//! Trigger frame (little endian, 16 bytes):
//!
//! ```text
//! 0..4   "KMC2"
//! 4      protocol version (2)
//! 5      command (0xa5 = left/right 250 ms; 0xb6 = both forward 10 s)
//! 6      recipe id (1 = left/right 250 ms; 2 = both forward 10 s)
//! 7      expected compiled maximum PWM percent (8)
//! 8..12  non-zero host nonce, u32 LE
//! 12..16 CRC-32/ISO-HDLC of bytes 0..12, u32 LE
//! ```
//!
//! Evidence frame (little endian, 32 bytes):
//!
//! ```text
//! 0..4   "KMR2"
//! 4      protocol version (2)
//! 5      event code
//! 6      recipe id, or zero before admission/rejection
//! 7      zero-based segment index, or 0xff
//! 8..12  echoed host nonce, or zero before admission
//! 12..16 TIM5 uptime in milliseconds, u32 LE
//! 16..20 commissioning build id (0x4b4d_4302), u32 LE
//! 20..24 exact commanded active duration in ms, u32 LE
//! 24     commanded output mask (bit 0 left-forward, bit 1 right-forward)
//! 25     commanded PWM percent
//! 26     event detail (segment count or rejection code)
//! 27     reserved zero
//! 28..32 CRC-32/ISO-HDLC of bytes 0..28, u32 LE
//! ```

use core::num::NonZeroU32;

const REQUEST_MAGIC: [u8; 4] = *b"KMC2";
const RESPONSE_MAGIC: [u8; 4] = *b"KMR2";
const PROTOCOL_VERSION: u8 = 2;
const LEFT_RIGHT_COMMAND: u8 = 0xa5;
const BOTH_FORWARD_10S_COMMAND: u8 = 0xb6;
const REQUEST_BYTES: usize = 16;
const RESPONSE_BYTES: usize = 32;
const COMMISSIONING_BUILD_ID: u32 = 0x4b4d_4302;

const PWM_FREQUENCY_HZ: u16 = 20_000;
const PULSE_PWM_PERCENT: u8 = 8;
const PULSE_DURATION_MS: u32 = 250;
const BOTH_FORWARD_DURATION_MS: u32 = 10_000;
const INITIAL_ZERO_DWELL_MS: u32 = 500;
const INTER_PULSE_ZERO_DWELL_MS: u32 = 500;
const SAFE_WATCHDOG_PERIOD_MS: u32 = 500;
const WATCHDOG_BACKSTOP_MARGIN_MS: u32 = 500;
const BOTH_FORWARD_WATCHDOG_CONFIG_MS: u32 = 15_500;
const LSI_NOMINAL_KHZ: u32 = 32;
const LSI_MAX_KHZ: u32 = 47;
const MAIN_LOOP_DELAY_MS: u32 = 1;
const SERIAL_RESPONSE_TIMEOUT_MS: u32 = 50;
const READY_PERIOD_MS: u32 = 1_000;
const WATCHDOG_REGISTER_UPDATE_TIMEOUT_MS: u32 = 10;

const _: () = assert!(PULSE_PWM_PERCENT > 0 && PULSE_PWM_PERCENT <= 8);
const _: () = assert!(PULSE_DURATION_MS > 0 && PULSE_DURATION_MS <= 300);
const _: () = assert!(BOTH_FORWARD_DURATION_MS == 10_000);
const _: () = assert!(BOTH_FORWARD_WATCHDOG_CONFIG_MS < (1 << 15));
const _: () = assert!(
    BOTH_FORWARD_WATCHDOG_CONFIG_MS * LSI_NOMINAL_KHZ / LSI_MAX_KHZ
        >= BOTH_FORWARD_DURATION_MS + WATCHDOG_BACKSTOP_MARGIN_MS
);
const _: () = assert!(INITIAL_ZERO_DWELL_MS > 0);
const _: () = assert!(INTER_PULSE_ZERO_DWELL_MS > 0);
const _: () = assert!(INITIAL_ZERO_DWELL_MS >= PULSE_DURATION_MS);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Trigger {
    nonce: NonZeroU32,
    recipe: Recipe,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
enum Recipe {
    LeftThenRight250Ms = 1,
    BothForward10S = 2,
}

impl Recipe {
    const fn command(self) -> u8 {
        match self {
            Self::LeftThenRight250Ms => LEFT_RIGHT_COMMAND,
            Self::BothForward10S => BOTH_FORWARD_10S_COMMAND,
        }
    }

    const fn wire(self) -> u8 {
        self as u8
    }

    const fn segment_count(self) -> u8 {
        match self {
            Self::LeftThenRight250Ms => 2,
            Self::BothForward10S => 1,
        }
    }

    const fn segment(self, index: u8) -> Option<Segment> {
        match (self, index) {
            (Self::LeftThenRight250Ms, 0) => Some(Segment {
                outputs: ForwardOutputs::Left,
                duration_ms: PULSE_DURATION_MS,
            }),
            (Self::LeftThenRight250Ms, 1) => Some(Segment {
                outputs: ForwardOutputs::Right,
                duration_ms: PULSE_DURATION_MS,
            }),
            (Self::BothForward10S, 0) => Some(Segment {
                outputs: ForwardOutputs::Both,
                duration_ms: BOTH_FORWARD_DURATION_MS,
            }),
            _ => None,
        }
    }

    const fn total_active_duration_ms(self) -> u32 {
        match self {
            Self::LeftThenRight250Ms => 2 * PULSE_DURATION_MS,
            Self::BothForward10S => BOTH_FORWARD_DURATION_MS,
        }
    }

    const fn output_union(self) -> ForwardOutputs {
        ForwardOutputs::Both
    }

    const fn watchdog_period_ms(self) -> u32 {
        match self {
            Self::LeftThenRight250Ms => SAFE_WATCHDOG_PERIOD_MS,
            Self::BothForward10S => BOTH_FORWARD_WATCHDOG_CONFIG_MS,
        }
    }

    const fn from_wire(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::LeftThenRight250Ms),
            2 => Some(Self::BothForward10S),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Segment {
    outputs: ForwardOutputs,
    duration_ms: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
enum ForwardOutputs {
    Left = 0b01,
    Right = 0b10,
    Both = 0b11,
}

impl ForwardOutputs {
    const fn wire(self) -> u8 {
        self as u8
    }

    #[cfg(any(not(target_arch = "arm"), test))]
    const fn from_wire(value: u8) -> Option<Self> {
        match value {
            0b01 => Some(Self::Left),
            0b10 => Some(Self::Right),
            0b11 => Some(Self::Both),
            _ => None,
        }
    }

    const fn drives_left(self) -> bool {
        self.wire() & Self::Left.wire() != 0
    }

    const fn drives_right(self) -> bool {
        self.wire() & Self::Right.wire() != 0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
enum TriggerError {
    Magic = 1,
    Version = 2,
    Command = 3,
    Recipe = 4,
    ExpectedPwm = 5,
    Nonce = 6,
    Checksum = 7,
}

impl TriggerError {
    const fn code(self) -> u8 {
        self as u8
    }

    #[cfg(any(not(target_arch = "arm"), test))]
    const fn from_code(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::Magic),
            2 => Some(Self::Version),
            3 => Some(Self::Command),
            4 => Some(Self::Recipe),
            5 => Some(Self::ExpectedPwm),
            6 => Some(Self::Nonce),
            7 => Some(Self::Checksum),
            _ => None,
        }
    }
}

fn parse_trigger(frame: &[u8; REQUEST_BYTES]) -> Result<Trigger, TriggerError> {
    if frame[..4] != REQUEST_MAGIC {
        return Err(TriggerError::Magic);
    }
    if frame[4] != PROTOCOL_VERSION {
        return Err(TriggerError::Version);
    }
    let recipe = Recipe::from_wire(frame[6]).ok_or(TriggerError::Recipe)?;
    if frame[5] != recipe.command() {
        return Err(TriggerError::Command);
    }
    if frame[7] != PULSE_PWM_PERCENT {
        return Err(TriggerError::ExpectedPwm);
    }
    let expected_checksum = u32::from_le_bytes([frame[12], frame[13], frame[14], frame[15]]);
    if crc32(&frame[..12]) != expected_checksum {
        return Err(TriggerError::Checksum);
    }
    let nonce = u32::from_le_bytes([frame[8], frame[9], frame[10], frame[11]]);
    let nonce = NonZeroU32::new(nonce).ok_or(TriggerError::Nonce)?;
    Ok(Trigger { nonce, recipe })
}

#[cfg(any(not(target_arch = "arm"), test))]
fn encode_trigger(trigger: Trigger) -> [u8; REQUEST_BYTES] {
    let mut frame = [0_u8; REQUEST_BYTES];
    frame[..4].copy_from_slice(&REQUEST_MAGIC);
    frame[4] = PROTOCOL_VERSION;
    frame[5] = trigger.recipe.command();
    frame[6] = trigger.recipe.wire();
    frame[7] = PULSE_PWM_PERCENT;
    frame[8..12].copy_from_slice(&trigger.nonce.get().to_le_bytes());
    let checksum = crc32(&frame[..12]);
    frame[12..16].copy_from_slice(&checksum.to_le_bytes());
    frame
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DecodeEvent {
    Trigger(Result<Trigger, TriggerError>),
}

#[derive(Debug)]
struct TriggerDecoder {
    frame: [u8; REQUEST_BYTES],
    length: usize,
}

impl TriggerDecoder {
    const fn new() -> Self {
        Self {
            frame: [0; REQUEST_BYTES],
            length: 0,
        }
    }

    fn push(&mut self, byte: u8) -> Option<DecodeEvent> {
        if self.length < REQUEST_MAGIC.len() {
            if byte == REQUEST_MAGIC[self.length] {
                self.frame[self.length] = byte;
                self.length += 1;
            } else if byte == REQUEST_MAGIC[0] {
                self.frame[0] = byte;
                self.length = 1;
            } else {
                self.length = 0;
            }
            return None;
        }

        self.frame[self.length] = byte;
        self.length += 1;
        if self.length != REQUEST_BYTES {
            return None;
        }

        let result = parse_trigger(&self.frame);
        self.length = 0;
        Some(DecodeEvent::Trigger(result))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Phase {
    WaitingForTrigger,
    InitialZeroDwell {
        nonce: NonZeroU32,
        recipe: Recipe,
        deadline_ms: u32,
    },
    PreparedPulse {
        nonce: NonZeroU32,
        recipe: Recipe,
        index: u8,
    },
    Pulse {
        nonce: NonZeroU32,
        recipe: Recipe,
        index: u8,
        deadline_ms: u32,
    },
    InterPulseZeroDwell {
        nonce: NonZeroU32,
        recipe: Recipe,
        next_index: u8,
        deadline_ms: u32,
    },
    LockedComplete {
        nonce: NonZeroU32,
        recipe: Recipe,
    },
    LockedFault,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MachineAction {
    None,
    Accepted {
        nonce: NonZeroU32,
        recipe: Recipe,
    },
    PreparePulse {
        nonce: NonZeroU32,
        recipe: Recipe,
        index: u8,
        outputs: ForwardOutputs,
        duration_ms: u32,
    },
    StartPulse {
        nonce: NonZeroU32,
        recipe: Recipe,
        index: u8,
        outputs: ForwardOutputs,
        duration_ms: u32,
        deadline_ms: u32,
    },
    PulseStopped {
        nonce: NonZeroU32,
        recipe: Recipe,
        index: u8,
        outputs: ForwardOutputs,
        duration_ms: u32,
        complete: bool,
    },
}

#[derive(Debug)]
struct CommissionMachine {
    phase: Phase,
    trigger_consumed: bool,
}

impl CommissionMachine {
    const fn new() -> Self {
        Self {
            phase: Phase::WaitingForTrigger,
            trigger_consumed: false,
        }
    }

    fn accept(&mut self, trigger: Trigger, now_ms: u32) -> MachineAction {
        if self.trigger_consumed || !matches!(self.phase, Phase::WaitingForTrigger) {
            self.phase = Phase::LockedFault;
            return MachineAction::None;
        }
        self.trigger_consumed = true;
        self.phase = Phase::InitialZeroDwell {
            nonce: trigger.nonce,
            recipe: trigger.recipe,
            deadline_ms: now_ms.wrapping_add(INITIAL_ZERO_DWELL_MS),
        };
        MachineAction::Accepted {
            nonce: trigger.nonce,
            recipe: trigger.recipe,
        }
    }

    fn poll_zero_phase(&mut self, now_ms: u32) -> MachineAction {
        let (nonce, recipe, index) = match self.phase {
            Phase::InitialZeroDwell {
                nonce,
                recipe,
                deadline_ms,
            } if deadline_reached(now_ms, deadline_ms) => (nonce, recipe, 0),
            Phase::InterPulseZeroDwell {
                nonce,
                recipe,
                next_index,
                deadline_ms,
            } if deadline_reached(now_ms, deadline_ms) => (nonce, recipe, next_index),
            _ => return MachineAction::None,
        };
        let Some(segment) = recipe.segment(index) else {
            self.phase = Phase::LockedFault;
            return MachineAction::None;
        };
        self.phase = Phase::PreparedPulse {
            nonce,
            recipe,
            index,
        };
        MachineAction::PreparePulse {
            nonce,
            recipe,
            index,
            outputs: segment.outputs,
            duration_ms: segment.duration_ms,
        }
    }

    fn begin_prepared_pulse(&mut self, start_ms: u32) -> MachineAction {
        let Phase::PreparedPulse {
            nonce,
            recipe,
            index,
        } = self.phase
        else {
            self.phase = Phase::LockedFault;
            return MachineAction::None;
        };
        let Some(segment) = recipe.segment(index) else {
            self.phase = Phase::LockedFault;
            return MachineAction::None;
        };
        let deadline_ms = start_ms.wrapping_add(segment.duration_ms);
        self.phase = Phase::Pulse {
            nonce,
            recipe,
            index,
            deadline_ms,
        };
        MachineAction::StartPulse {
            nonce,
            recipe,
            index,
            outputs: segment.outputs,
            duration_ms: segment.duration_ms,
            deadline_ms,
        }
    }

    fn pulse_expired(&mut self, now_ms: u32) -> MachineAction {
        let Phase::Pulse {
            nonce,
            recipe,
            index,
            deadline_ms,
        } = self.phase
        else {
            self.phase = Phase::LockedFault;
            return MachineAction::None;
        };
        if !deadline_reached(now_ms, deadline_ms) {
            self.phase = Phase::LockedFault;
            return MachineAction::None;
        }
        let Some(segment) = recipe.segment(index) else {
            self.phase = Phase::LockedFault;
            return MachineAction::None;
        };
        let Some(next_index) = index.checked_add(1) else {
            self.phase = Phase::LockedFault;
            return MachineAction::None;
        };
        let complete = next_index == recipe.segment_count();
        if complete {
            self.phase = Phase::LockedComplete { nonce, recipe };
        } else {
            self.phase = Phase::InterPulseZeroDwell {
                nonce,
                recipe,
                next_index,
                deadline_ms: now_ms.wrapping_add(INTER_PULSE_ZERO_DWELL_MS),
            };
        }
        MachineAction::PulseStopped {
            nonce,
            recipe,
            index,
            outputs: segment.outputs,
            duration_ms: segment.duration_ms,
            complete,
        }
    }

    const fn pulse_deadline(&self) -> Option<u32> {
        match self.phase {
            Phase::Pulse { deadline_ms, .. } => Some(deadline_ms),
            _ => None,
        }
    }

    const fn outputs_must_be_safe(&self) -> bool {
        !matches!(self.phase, Phase::Pulse { .. })
    }

    fn fault(&mut self) {
        self.trigger_consumed = true;
        self.phase = Phase::LockedFault;
    }
}

const fn deadline_reached(now_ms: u32, deadline_ms: u32) -> bool {
    now_ms.wrapping_sub(deadline_ms) < (1_u32 << 31)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
enum ResponseEvent {
    Ready = 1,
    Accepted = 2,
    PulseCompleted = 3,
    Complete = 4,
    Rejected = 0x7f,
}

#[cfg(any(not(target_arch = "arm"), test))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CommissionResponse {
    Ready {
        uptime_ms: u32,
    },
    Accepted {
        nonce: NonZeroU32,
        recipe: Recipe,
        uptime_ms: u32,
    },
    PulseCompleted {
        nonce: NonZeroU32,
        recipe: Recipe,
        index: u8,
        outputs: ForwardOutputs,
        commanded_duration_ms: u32,
        uptime_ms: u32,
    },
    Complete {
        nonce: NonZeroU32,
        recipe: Recipe,
        uptime_ms: u32,
    },
    Rejected {
        error: TriggerError,
        uptime_ms: u32,
    },
}

#[cfg(any(not(target_arch = "arm"), test))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ResponseParseError {
    Magic,
    Version,
    Checksum,
    BuildId,
    Event,
    Fields,
}

#[cfg(any(not(target_arch = "arm"), test))]
fn parse_response(frame: &[u8; RESPONSE_BYTES]) -> Result<CommissionResponse, ResponseParseError> {
    if frame[..4] != RESPONSE_MAGIC {
        return Err(ResponseParseError::Magic);
    }
    if frame[4] != PROTOCOL_VERSION {
        return Err(ResponseParseError::Version);
    }
    let expected_checksum = u32::from_le_bytes([frame[28], frame[29], frame[30], frame[31]]);
    if crc32(&frame[..28]) != expected_checksum {
        return Err(ResponseParseError::Checksum);
    }
    let build_id = u32::from_le_bytes([frame[16], frame[17], frame[18], frame[19]]);
    if build_id != COMMISSIONING_BUILD_ID {
        return Err(ResponseParseError::BuildId);
    }
    let recipe_wire = frame[6];
    let segment_index = frame[7];
    let nonce = u32::from_le_bytes([frame[8], frame[9], frame[10], frame[11]]);
    let uptime_ms = u32::from_le_bytes([frame[12], frame[13], frame[14], frame[15]]);
    let commanded_duration_ms = u32::from_le_bytes([frame[20], frame[21], frame[22], frame[23]]);
    let outputs_wire = frame[24];
    let pwm_percent = frame[25];
    let detail = frame[26];
    if frame[27] != 0 {
        return Err(ResponseParseError::Fields);
    }
    match frame[5] {
        value if value == ResponseEvent::Ready as u8 => {
            if recipe_wire != 0
                || segment_index != u8::MAX
                || nonce != 0
                || commanded_duration_ms != 0
                || outputs_wire != 0
                || pwm_percent != PULSE_PWM_PERCENT
                || detail != 0
            {
                return Err(ResponseParseError::Fields);
            }
            Ok(CommissionResponse::Ready { uptime_ms })
        }
        value if value == ResponseEvent::Accepted as u8 => {
            let recipe = Recipe::from_wire(recipe_wire).ok_or(ResponseParseError::Fields)?;
            if segment_index != u8::MAX
                || commanded_duration_ms != recipe.total_active_duration_ms()
                || outputs_wire != recipe.output_union().wire()
                || pwm_percent != PULSE_PWM_PERCENT
                || detail != recipe.segment_count()
            {
                return Err(ResponseParseError::Fields);
            }
            let nonce = NonZeroU32::new(nonce).ok_or(ResponseParseError::Fields)?;
            Ok(CommissionResponse::Accepted {
                nonce,
                recipe,
                uptime_ms,
            })
        }
        value if value == ResponseEvent::PulseCompleted as u8 => {
            let recipe = Recipe::from_wire(recipe_wire).ok_or(ResponseParseError::Fields)?;
            let segment = recipe
                .segment(segment_index)
                .ok_or(ResponseParseError::Fields)?;
            let outputs =
                ForwardOutputs::from_wire(outputs_wire).ok_or(ResponseParseError::Fields)?;
            if commanded_duration_ms != segment.duration_ms
                || outputs != segment.outputs
                || pwm_percent != PULSE_PWM_PERCENT
                || detail != 0
            {
                return Err(ResponseParseError::Fields);
            }
            let nonce = NonZeroU32::new(nonce).ok_or(ResponseParseError::Fields)?;
            Ok(CommissionResponse::PulseCompleted {
                nonce,
                recipe,
                index: segment_index,
                outputs,
                commanded_duration_ms,
                uptime_ms,
            })
        }
        value if value == ResponseEvent::Complete as u8 => {
            let recipe = Recipe::from_wire(recipe_wire).ok_or(ResponseParseError::Fields)?;
            if segment_index != u8::MAX
                || commanded_duration_ms != recipe.total_active_duration_ms()
                || outputs_wire != 0
                || pwm_percent != 0
                || detail != 0
            {
                return Err(ResponseParseError::Fields);
            }
            let nonce = NonZeroU32::new(nonce).ok_or(ResponseParseError::Fields)?;
            Ok(CommissionResponse::Complete {
                nonce,
                recipe,
                uptime_ms,
            })
        }
        value if value == ResponseEvent::Rejected as u8 => {
            if recipe_wire != 0
                || segment_index != u8::MAX
                || nonce != 0
                || commanded_duration_ms != 0
                || outputs_wire != 0
                || pwm_percent != 0
            {
                return Err(ResponseParseError::Fields);
            }
            let error = TriggerError::from_code(detail).ok_or(ResponseParseError::Fields)?;
            Ok(CommissionResponse::Rejected { error, uptime_ms })
        }
        _ => Err(ResponseParseError::Event),
    }
}

#[cfg(any(not(target_arch = "arm"), test))]
#[derive(Debug)]
struct ResponseDecoder {
    frame: [u8; RESPONSE_BYTES],
    length: usize,
}

#[cfg(any(not(target_arch = "arm"), test))]
impl ResponseDecoder {
    const fn new() -> Self {
        Self {
            frame: [0; RESPONSE_BYTES],
            length: 0,
        }
    }

    fn push(&mut self, byte: u8) -> Option<Result<CommissionResponse, ResponseParseError>> {
        if self.length < RESPONSE_MAGIC.len() {
            if byte == RESPONSE_MAGIC[self.length] {
                self.frame[self.length] = byte;
                self.length += 1;
            } else if byte == RESPONSE_MAGIC[0] {
                self.frame[0] = byte;
                self.length = 1;
            } else {
                self.length = 0;
            }
            return None;
        }

        self.frame[self.length] = byte;
        self.length += 1;
        if self.length != RESPONSE_BYTES {
            return None;
        }
        let response = parse_response(&self.frame);
        self.length = 0;
        Some(response)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CompletedPulse {
    SequentialLeft,
    SequentialRight,
    Both10S,
}

impl CompletedPulse {
    const fn from_recipe_index(recipe: Recipe, index: u8) -> Option<Self> {
        match (recipe, index) {
            (Recipe::LeftThenRight250Ms, 0) => Some(Self::SequentialLeft),
            (Recipe::LeftThenRight250Ms, 1) => Some(Self::SequentialRight),
            (Recipe::BothForward10S, 0) => Some(Self::Both10S),
            _ => None,
        }
    }

    const fn recipe(self) -> Recipe {
        match self {
            Self::SequentialLeft | Self::SequentialRight => Recipe::LeftThenRight250Ms,
            Self::Both10S => Recipe::BothForward10S,
        }
    }

    const fn index(self) -> u8 {
        match self {
            Self::SequentialLeft | Self::Both10S => 0,
            Self::SequentialRight => 1,
        }
    }

    const fn segment(self) -> Segment {
        match self {
            Self::SequentialLeft => Segment {
                outputs: ForwardOutputs::Left,
                duration_ms: PULSE_DURATION_MS,
            },
            Self::SequentialRight => Segment {
                outputs: ForwardOutputs::Right,
                duration_ms: PULSE_DURATION_MS,
            },
            Self::Both10S => Segment {
                outputs: ForwardOutputs::Both,
                duration_ms: BOTH_FORWARD_DURATION_MS,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OutboundResponse {
    Ready {
        uptime_ms: u32,
    },
    Accepted {
        trigger: Trigger,
        uptime_ms: u32,
    },
    PulseCompleted {
        nonce: NonZeroU32,
        pulse: CompletedPulse,
        uptime_ms: u32,
    },
    Complete {
        trigger: Trigger,
        uptime_ms: u32,
    },
    Rejected {
        error: TriggerError,
        uptime_ms: u32,
    },
}

fn encode_response(response: OutboundResponse) -> [u8; RESPONSE_BYTES] {
    let (
        event,
        recipe,
        segment_index,
        commanded_duration_ms,
        outputs,
        pwm_percent,
        detail,
        nonce,
        uptime_ms,
    ) = match response {
        OutboundResponse::Ready { uptime_ms } => (
            ResponseEvent::Ready,
            None,
            None,
            0,
            None,
            PULSE_PWM_PERCENT,
            0,
            0,
            uptime_ms,
        ),
        OutboundResponse::Accepted { trigger, uptime_ms } => (
            ResponseEvent::Accepted,
            Some(trigger.recipe),
            None,
            trigger.recipe.total_active_duration_ms(),
            Some(trigger.recipe.output_union()),
            PULSE_PWM_PERCENT,
            trigger.recipe.segment_count(),
            trigger.nonce.get(),
            uptime_ms,
        ),
        OutboundResponse::PulseCompleted {
            nonce,
            pulse,
            uptime_ms,
        } => {
            let segment = pulse.segment();
            (
                ResponseEvent::PulseCompleted,
                Some(pulse.recipe()),
                Some(pulse.index()),
                segment.duration_ms,
                Some(segment.outputs),
                PULSE_PWM_PERCENT,
                0,
                nonce.get(),
                uptime_ms,
            )
        }
        OutboundResponse::Complete { trigger, uptime_ms } => (
            ResponseEvent::Complete,
            Some(trigger.recipe),
            None,
            trigger.recipe.total_active_duration_ms(),
            None,
            0,
            0,
            trigger.nonce.get(),
            uptime_ms,
        ),
        OutboundResponse::Rejected { error, uptime_ms } => (
            ResponseEvent::Rejected,
            None,
            None,
            0,
            None,
            0,
            error.code(),
            0,
            uptime_ms,
        ),
    };
    let mut frame = [0_u8; RESPONSE_BYTES];
    frame[..4].copy_from_slice(&RESPONSE_MAGIC);
    frame[4] = PROTOCOL_VERSION;
    frame[5] = event as u8;
    frame[6] = recipe.map_or(0, Recipe::wire);
    frame[7] = segment_index.unwrap_or(u8::MAX);
    frame[8..12].copy_from_slice(&nonce.to_le_bytes());
    frame[12..16].copy_from_slice(&uptime_ms.to_le_bytes());
    frame[16..20].copy_from_slice(&COMMISSIONING_BUILD_ID.to_le_bytes());
    frame[20..24].copy_from_slice(&commanded_duration_ms.to_le_bytes());
    frame[24] = outputs.map_or(0, ForwardOutputs::wire);
    frame[25] = pwm_percent;
    frame[26] = detail;
    frame[27] = 0;
    let checksum = crc32(&frame[..28]);
    frame[28..32].copy_from_slice(&checksum.to_le_bytes());
    frame
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = u32::MAX;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            let mask = 0_u32.wrapping_sub(crc & 1);
            crc = (crc >> 1) ^ (0xedb8_8320 & mask);
        }
    }
    !crc
}

#[cfg(target_arch = "arm")]
mod target {
    use core::{
        cell::RefCell,
        num::NonZeroU16,
        panic::PanicInfo,
        sync::atomic::{AtomicBool, Ordering},
    };

    use cortex_m::{interrupt::Mutex, peripheral::NVIC, register::primask};
    use cortex_m_rt::{ExceptionFrame, entry, exception};
    use heapless::spsc::Queue;
    use stm32f4xx_hal::{
        pac::{self, interrupt},
        prelude::*,
        serial::{Rx, RxISR, RxListen, Serial, Tx, config::Config},
        timer::{Channel1, Channel2, FTimer, PwmChannel, pwm::PwmExt},
        watchdog::IndependentWatchdog,
    };

    use super::{
        CommissionMachine, CompletedPulse, DecodeEvent, ForwardOutputs, MAIN_LOOP_DELAY_MS,
        MachineAction, OutboundResponse, PULSE_PWM_PERCENT, PWM_FREQUENCY_HZ, READY_PERIOD_MS,
        RESPONSE_BYTES, SAFE_WATCHDOG_PERIOD_MS, SERIAL_RESPONSE_TIMEOUT_MS, Trigger,
        TriggerDecoder, WATCHDOG_REGISTER_UPDATE_TIMEOUT_MS, deadline_reached, encode_response,
    };

    static PULSE_ARMED: AtomicBool = AtomicBool::new(false);
    static PULSE_EXPIRED: AtomicBool = AtomicBool::new(false);
    const RX_QUEUE_BYTES: usize = 64;
    const RX_DEQUEUE_BUDGET_BYTES: usize = 32;
    const _: () = assert!(RX_QUEUE_BYTES > (2 * super::REQUEST_BYTES));
    const _: () = assert!(RX_DEQUEUE_BUDGET_BYTES >= super::REQUEST_BYTES);
    static RX_QUEUE: Mutex<RefCell<Queue<u8, RX_QUEUE_BYTES>>> =
        Mutex::new(RefCell::new(Queue::new()));
    static RX_STREAM_INVALIDATED: AtomicBool = AtomicBool::new(false);
    static SERIAL_RX: Mutex<RefCell<Option<Rx<pac::USART2>>>> = Mutex::new(RefCell::new(None));

    type LeftForward = PwmChannel<pac::TIM2, 0>;
    type LeftReverse = PwmChannel<pac::TIM2, 1>;
    type RightForward = PwmChannel<pac::TIM3, 0>;
    type RightReverse = PwmChannel<pac::TIM3, 1>;

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum HardwareState {
        Safe,
        Pulsing,
    }

    struct HardwareMotor {
        left_forward: LeftForward,
        left_reverse: LeftReverse,
        right_forward: RightForward,
        right_reverse: RightReverse,
        left_max_duty: NonZeroU16,
        right_max_duty: NonZeroU16,
        state: HardwareState,
    }

    impl HardwareMotor {
        fn new(
            left_forward: LeftForward,
            left_reverse: LeftReverse,
            right_forward: RightForward,
            right_reverse: RightReverse,
        ) -> Option<Self> {
            let left_max_duty = NonZeroU16::new(left_forward.get_max_duty())?;
            let right_max_duty = NonZeroU16::new(right_forward.get_max_duty())?;
            let mut motor = Self {
                left_forward,
                left_reverse,
                right_forward,
                right_reverse,
                left_max_duty,
                right_max_duty,
                state: HardwareState::Safe,
            };
            motor.disable_and_zero();
            Some(motor)
        }

        fn start_pulse(&mut self, outputs: ForwardOutputs, deadline_ms: u32) -> bool {
            self.disable_and_zero();
            let left_duty = scale_duty(PULSE_PWM_PERCENT, self.left_max_duty);
            let right_duty = scale_duty(PULSE_PWM_PERCENT, self.right_max_duty);
            if (outputs.drives_left() && left_duty == 0)
                || (outputs.drives_right() && right_duty == 0)
            {
                return false;
            }

            cortex_m::interrupt::free(|_| {
                // Break-before-make: every channel is disabled and zeroed
                // before exactly one forward duty is preloaded.
                self.disable_and_zero();
                if outputs.drives_left() {
                    self.left_forward.set_duty(left_duty);
                }
                if outputs.drives_right() {
                    self.right_forward.set_duty(right_duty);
                }
                if !arm_pulse_deadline_unlocked(deadline_ms) {
                    self.disable_and_zero();
                    return false;
                }
                if outputs.drives_left() {
                    self.left_forward.enable();
                }
                if outputs.drives_right() {
                    self.right_forward.enable();
                }
                // Re-sample the complete cutoff evidence after the final PWM
                // enable while PRIMASK still prevents TIM5 from racing this
                // admission. A deadline reached during setup is never allowed
                // to become an unbounded pulse when interrupts resume.
                if !pulse_deadline_is_live_unlocked(deadline_ms) {
                    self.disable_and_zero();
                    cancel_pulse_deadline_unlocked();
                    return false;
                }
                self.state = HardwareState::Pulsing;
                true
            })
        }

        fn disable_and_zero(&mut self) {
            self.left_forward.disable();
            self.left_reverse.disable();
            self.right_forward.disable();
            self.right_reverse.disable();
            self.left_forward.set_duty(0);
            self.left_reverse.set_duty(0);
            self.right_forward.set_duty(0);
            self.right_reverse.set_duty(0);
            self.state = HardwareState::Safe;
        }

        const fn is_safe(&self) -> bool {
            matches!(self.state, HardwareState::Safe)
        }
    }

    #[entry]
    fn main() -> ! {
        let Some(dp) = pac::Peripherals::take() else {
            fatal();
        };

        // Start the independent watchdog before any fallible setup.
        let mut watchdog = IndependentWatchdog::new(dp.IWDG);
        watchdog.start(SAFE_WATCHDOG_PERIOD_MS.millis());

        let Some(mut cp) = cortex_m::peripheral::Peripherals::take() else {
            fatal();
        };
        let rcc = dp.RCC.constrain();
        let clocks = rcc
            .cfgr
            .sysclk(168.MHz())
            .pclk1(21.MHz())
            .pclk2(84.MHz())
            .freeze();
        let mut delay = cp.SYST.delay(&clocks);

        let gpioa = dp.GPIOA.split();
        let gpiob = dp.GPIOB.split();
        let mut led = gpioa.pa5.into_push_pull_output();
        led.set_low();

        // These are exactly the production firmware_v2 motor pins and timer
        // alternate functions. Drive them low before transferring ownership.
        let mut left_forward_pin = gpioa.pa0.into_push_pull_output();
        let mut left_reverse_pin = gpioa.pa1.into_push_pull_output();
        let mut right_forward_pin = gpiob.pb4.into_push_pull_output();
        let mut right_reverse_pin = gpiob.pb5.into_push_pull_output();
        left_forward_pin.set_low();
        left_reverse_pin.set_low();
        right_forward_pin.set_low();
        right_reverse_pin.set_low();
        let left_forward_pin = left_forward_pin.into_alternate::<1>();
        let left_reverse_pin = left_reverse_pin.into_alternate::<1>();
        let right_forward_pin = right_forward_pin.into_alternate::<2>();
        let right_reverse_pin = right_reverse_pin.into_alternate::<2>();

        let tx = gpioa.pa2.into_alternate::<7>();
        let rx = gpioa.pa3.into_alternate::<7>();
        let serial = match Serial::new(
            dp.USART2,
            (tx, rx),
            Config::default()
                .baudrate(115_200.bps())
                .wordlength_8()
                .parity_none(),
            &clocks,
        ) {
            Ok(serial) => serial,
            Err(_) => fatal(),
        };
        let (mut serial_tx, mut serial_rx) = serial.split();
        serial_rx.listen();
        cortex_m::interrupt::free(|cs| SERIAL_RX.borrow(cs).replace(Some(serial_rx)));

        let (left_forward, left_reverse) = dp
            .TIM2
            .pwm_hz(
                (
                    Channel1::new(left_forward_pin),
                    Channel2::new(left_reverse_pin),
                ),
                u32::from(PWM_FREQUENCY_HZ).Hz(),
                &clocks,
            )
            .split();
        let (right_forward, right_reverse) = dp
            .TIM3
            .pwm_hz(
                (
                    Channel1::new(right_forward_pin),
                    Channel2::new(right_reverse_pin),
                ),
                u32::from(PWM_FREQUENCY_HZ).Hz(),
                &clocks,
            )
            .split();
        let Some(mut motor) =
            HardwareMotor::new(left_forward, left_reverse, right_forward, right_reverse)
        else {
            fatal();
        };

        configure_deadline_timer(dp.TIM5, &clocks);
        enable_interrupts(&mut cp.NVIC);

        let mut decoder = TriggerDecoder::new();
        let mut machine = CommissionMachine::new();
        let ready = encode_response(OutboundResponse::Ready {
            uptime_ms: uptime_ms(),
        });
        let mut last_ready_ms = uptime_ms();
        if !send_response(&mut serial_tx, &ready) {
            enter_fault(&mut machine, &mut motor, &mut watchdog);
        }

        loop {
            let now_ms = uptime_ms();

            if PULSE_EXPIRED.swap(false, Ordering::AcqRel) {
                // Re-sample only after observing the ISR flag. TIM5 may fire
                // between the loop's first timestamp and the atomic swap; a
                // pre-deadline timestamp must not turn a correct cutoff into a
                // locked fault or false no-retry result.
                let cutoff_evidence_ms = uptime_ms();
                motor.disable_and_zero();
                disarm_pulse_deadline();
                if !set_watchdog_period(&mut watchdog, SAFE_WATCHDOG_PERIOD_MS) {
                    fatal();
                }
                match machine.pulse_expired(cutoff_evidence_ms) {
                    MachineAction::PulseStopped {
                        nonce,
                        recipe,
                        index,
                        outputs,
                        duration_ms,
                        complete,
                    } => {
                        let Some(pulse) = CompletedPulse::from_recipe_index(recipe, index) else {
                            enter_fault(&mut machine, &mut motor, &mut watchdog);
                            continue;
                        };
                        let expected_segment = pulse.segment();
                        if expected_segment.outputs != outputs
                            || expected_segment.duration_ms != duration_ms
                        {
                            enter_fault(&mut machine, &mut motor, &mut watchdog);
                            continue;
                        }
                        let pulse_evidence = encode_response(OutboundResponse::PulseCompleted {
                            nonce,
                            pulse,
                            uptime_ms: cutoff_evidence_ms,
                        });
                        if !send_response(&mut serial_tx, &pulse_evidence) {
                            enter_fault(&mut machine, &mut motor, &mut watchdog);
                        } else if complete {
                            let complete_evidence = encode_response(OutboundResponse::Complete {
                                trigger: Trigger { nonce, recipe },
                                uptime_ms: uptime_ms(),
                            });
                            if !send_response(&mut serial_tx, &complete_evidence) {
                                enter_fault(&mut machine, &mut motor, &mut watchdog);
                            } else {
                                led.set_high();
                            }
                        }
                    }
                    _ => enter_fault(&mut machine, &mut motor, &mut watchdog),
                }
            }

            if let Some(deadline_ms) = machine.pulse_deadline() {
                if deadline_reached(now_ms, deadline_ms) && !PULSE_EXPIRED.load(Ordering::Acquire) {
                    // The main-loop backstop observed a missed compare ISR.
                    // Stop rather than extending a pulse or attempting again.
                    enter_fault(&mut machine, &mut motor, &mut watchdog);
                }
            } else if !machine.outputs_must_be_safe() || !motor.is_safe() {
                enter_fault(&mut machine, &mut motor, &mut watchdog);
            }

            if matches!(machine.phase, super::Phase::WaitingForTrigger) {
                if deadline_reached(now_ms, last_ready_ms.wrapping_add(READY_PERIOD_MS)) {
                    let ready = encode_response(OutboundResponse::Ready { uptime_ms: now_ms });
                    if send_response(&mut serial_tx, &ready) {
                        last_ready_ms = now_ms;
                    } else {
                        enter_fault(&mut machine, &mut motor, &mut watchdog);
                    }
                }
                for _ in 0..RX_DEQUEUE_BUDGET_BYTES {
                    let byte = match dequeue_rx_event() {
                        RxDequeue::Byte(byte) => byte,
                        RxDequeue::Empty => break,
                        RxDequeue::Invalidated => {
                            enter_fault(&mut machine, &mut motor, &mut watchdog);
                            break;
                        }
                    };
                    let Some(DecodeEvent::Trigger(result)) = decoder.push(byte) else {
                        continue;
                    };
                    match result {
                        Ok(trigger) => {
                            motor.disable_and_zero();
                            let admitted_at_ms = uptime_ms();
                            match machine.accept(trigger, admitted_at_ms) {
                                MachineAction::Accepted { nonce, recipe } => {
                                    let accepted = encode_response(OutboundResponse::Accepted {
                                        trigger: Trigger { nonce, recipe },
                                        uptime_ms: admitted_at_ms,
                                    });
                                    if !send_response(&mut serial_tx, &accepted) {
                                        enter_fault(&mut machine, &mut motor, &mut watchdog);
                                    }
                                }
                                _ => enter_fault(&mut machine, &mut motor, &mut watchdog),
                            }
                        }
                        Err(error) => {
                            enter_fault(&mut machine, &mut motor, &mut watchdog);
                            let rejected = encode_response(OutboundResponse::Rejected {
                                error,
                                uptime_ms: uptime_ms(),
                            });
                            let _reported = send_response(&mut serial_tx, &rejected);
                        }
                    }
                    break;
                }
            } else if !matches!(dequeue_rx_event(), RxDequeue::Empty) {
                // Any bytes, overrun, or queue overflow after the one accepted
                // trigger aborts the sequence rather than changing/retrying it.
                enter_fault(&mut machine, &mut motor, &mut watchdog);
            }

            match machine.poll_zero_phase(now_ms) {
                MachineAction::PreparePulse {
                    recipe,
                    outputs,
                    duration_ms,
                    index,
                    ..
                } => {
                    if !set_watchdog_period(&mut watchdog, recipe.watchdog_period_ms()) {
                        enter_fault(&mut machine, &mut motor, &mut watchdog);
                    } else {
                        // The active deadline is derived only after the new
                        // watchdog shadow registers and reload are confirmed.
                        // This timestamp is immediately before PWM setup; the
                        // hardware path re-samples it after final enable.
                        let start_ms = uptime_ms();
                        match machine.begin_prepared_pulse(start_ms) {
                            MachineAction::StartPulse {
                                recipe: started_recipe,
                                index: started_index,
                                outputs: started_outputs,
                                duration_ms: started_duration_ms,
                                deadline_ms,
                                ..
                            } if started_recipe == recipe
                                && started_index == index
                                && started_outputs == outputs
                                && started_duration_ms == duration_ms =>
                            {
                                if !motor.start_pulse(outputs, deadline_ms) {
                                    enter_fault(&mut machine, &mut motor, &mut watchdog);
                                }
                            }
                            _ => enter_fault(&mut machine, &mut motor, &mut watchdog),
                        }
                    }
                }
                MachineAction::None => {}
                _ => enter_fault(&mut machine, &mut motor, &mut watchdog),
            }

            let outputs_safely_zero = motor.is_safe()
                && !PULSE_ARMED.load(Ordering::Acquire)
                && !PULSE_EXPIRED.load(Ordering::Acquire);
            // Never feed IWDG while outputs are active. Each recipe configured
            // its own pre-enable interval: nominal 500 ms for a 250 ms pulse,
            // or a conservative nominal-32-kHz 15,500 ms configuration for a
            // 10,000 ms pulse. At the datasheet maximum 47 kHz LSI this still
            // leaves at least 500 ms after nominal cutoff; at lower frequency
            // it can be much longer. This is not an arbitrary-fault proof.
            if outputs_safely_zero {
                watchdog.feed();
            }
            delay.delay_ms(MAIN_LOOP_DELAY_MS);
        }
    }

    fn enter_fault(
        machine: &mut CommissionMachine,
        motor: &mut HardwareMotor,
        watchdog: &mut IndependentWatchdog,
    ) {
        motor.disable_and_zero();
        disarm_pulse_deadline();
        machine.fault();
        if !set_watchdog_period(watchdog, SAFE_WATCHDOG_PERIOD_MS) {
            fatal();
        }
    }

    #[allow(unsafe_code)]
    fn set_watchdog_period(watchdog: &mut IndependentWatchdog, requested_ms: u32) -> bool {
        watchdog.start(requested_ms.millis());
        let update_deadline_ms = uptime_ms().wrapping_add(WATCHDOG_REGISTER_UPDATE_TIMEOUT_MS);
        loop {
            // SAFETY: IWDG remains owned by `watchdog`; this read-only access
            // checks both asynchronous shadow-register update flags omitted by
            // HAL 0.20's `interval()` wait. Outputs are disabled at every call.
            let update_pending = unsafe {
                let status = (&*pac::IWDG::ptr()).sr.read();
                status.pvu().bit_is_set() || status.rvu().bit_is_set()
            };
            if !update_pending {
                break;
            }
            if deadline_reached(uptime_ms(), update_deadline_ms) {
                return false;
            }
        }
        // `start()` may have reloaded before RLR reached its active shadow
        // register. Reload again only after both update flags clear and while
        // outputs are still disabled. There are no feeds after PWM enable.
        watchdog.feed();
        let configured_ms = watchdog.interval().ticks();
        configured_ms >= requested_ms && configured_ms <= requested_ms.saturating_add(16)
    }

    enum RxDequeue {
        Byte(u8),
        Empty,
        Invalidated,
    }

    fn dequeue_rx_event() -> RxDequeue {
        cortex_m::interrupt::free(|cs| {
            if RX_STREAM_INVALIDATED.swap(false, Ordering::AcqRel) {
                let mut queue = RX_QUEUE.borrow(cs).borrow_mut();
                while queue.dequeue().is_some() {}
                RxDequeue::Invalidated
            } else {
                RX_QUEUE
                    .borrow(cs)
                    .borrow_mut()
                    .dequeue()
                    .map_or(RxDequeue::Empty, RxDequeue::Byte)
            }
        })
    }

    fn send_response(serial: &mut Tx<pac::USART2>, frame: &[u8; RESPONSE_BYTES]) -> bool {
        let started_at = uptime_ms();
        for &byte in frame {
            loop {
                match serial.write(byte) {
                    Ok(()) => break,
                    Err(nb::Error::WouldBlock) => {
                        if deadline_reached(
                            uptime_ms(),
                            started_at.wrapping_add(SERIAL_RESPONSE_TIMEOUT_MS),
                        ) {
                            return false;
                        }
                    }
                    Err(nb::Error::Other(_)) => return false,
                }
            }
        }
        true
    }

    fn scale_duty(percent: u8, maximum: NonZeroU16) -> u16 {
        let scaled = u32::from(percent) * u32::from(maximum.get()) / 100;
        scaled as u16
    }

    #[allow(unsafe_code)]
    fn configure_deadline_timer(timer: pac::TIM5, clocks: &stm32f4xx_hal::rcc::Clocks) {
        let timer = FTimer::<pac::TIM5, 1_000>::new(timer, clocks).release();
        // SAFETY: this function exclusively owns TIM5 before its interrupt is
        // unmasked. The resulting counter is exactly 1 kHz for the configured
        // 42 MHz APB1 timer clock.
        unsafe {
            timer.cr1.modify(|_, writer| writer.cen().clear_bit());
            timer.dier.write(|writer| writer.bits(0));
            timer.arr.write(|writer| writer.arr().bits(u32::MAX));
            timer.cnt.write(|writer| writer.cnt().bits(0));
            timer.ccr1().write(|writer| writer.ccr().bits(0));
            timer.egr.write(|writer| writer.ug().set_bit());
            timer.sr.write(|writer| writer.bits(0));
            timer.cr1.modify(|_, writer| writer.cen().set_bit());
        }
    }

    #[allow(unsafe_code)]
    fn uptime_ms() -> u32 {
        // SAFETY: TIM5 is configured once as a free-running 1 kHz counter.
        unsafe { (&*pac::TIM5::ptr()).cnt.read().cnt().bits() }
    }

    #[allow(unsafe_code)]
    fn arm_pulse_deadline_unlocked(deadline_ms: u32) -> bool {
        debug_assert!(primask::read().is_active());
        // SAFETY: caller masks interrupts and owns all main-context TIM5
        // compare mutations. The ISR only clears this same compare source.
        unsafe {
            let timer = &*pac::TIM5::ptr();
            if PULSE_ARMED.load(Ordering::Acquire) || PULSE_EXPIRED.load(Ordering::Acquire) {
                return false;
            }
            timer.dier.modify(|_, writer| writer.cc1ie().clear_bit());
            timer.sr.modify(|_, writer| writer.cc1if().clear_bit());
            if !deadline_reached(deadline_ms, timer.cnt.read().cnt().bits().wrapping_add(1)) {
                return false;
            }
            timer.ccr1().write(|writer| writer.ccr().bits(deadline_ms));
            timer.sr.modify(|_, writer| writer.cc1if().clear_bit());
            let after = timer.cnt.read().cnt().bits();
            if deadline_reached(after, deadline_ms) {
                return false;
            }
            PULSE_ARMED.store(true, Ordering::Release);
            timer.dier.modify(|_, writer| writer.cc1ie().set_bit());
            true
        }
    }

    #[allow(unsafe_code)]
    fn pulse_deadline_is_live_unlocked(expected_deadline_ms: u32) -> bool {
        debug_assert!(primask::read().is_active());
        // SAFETY: caller masks interrupts. TIM5 is the initialized 1 kHz
        // commissioning timer and only this binary owns its compare channel.
        unsafe {
            let timer = &*pac::TIM5::ptr();
            let now_ms = timer.cnt.read().cnt().bits();
            PULSE_ARMED.load(Ordering::Acquire)
                && !PULSE_EXPIRED.load(Ordering::Acquire)
                && timer.dier.read().cc1ie().bit_is_set()
                && !timer.sr.read().cc1if().bit_is_set()
                && timer.ccr1().read().ccr().bits() == expected_deadline_ms
                && !deadline_reached(now_ms, expected_deadline_ms)
        }
    }

    #[allow(unsafe_code)]
    fn cancel_pulse_deadline_unlocked() {
        debug_assert!(primask::read().is_active());
        // SAFETY: caller masks interrupts and owns main-context compare state.
        unsafe {
            let timer = &*pac::TIM5::ptr();
            timer.dier.modify(|_, writer| writer.cc1ie().clear_bit());
            timer.sr.modify(|_, writer| writer.cc1if().clear_bit());
        }
        PULSE_ARMED.store(false, Ordering::Release);
        PULSE_EXPIRED.store(false, Ordering::Release);
    }

    #[allow(unsafe_code)]
    fn disarm_pulse_deadline() {
        cortex_m::interrupt::free(|_| {
            cancel_pulse_deadline_unlocked();
        });
    }

    #[interrupt]
    fn USART2() {
        cortex_m::interrupt::free(|cs| {
            let mut receiver = SERIAL_RX.borrow(cs).borrow_mut();
            let Some(receiver) = receiver.as_mut() else {
                RX_STREAM_INVALIDATED.store(true, Ordering::Release);
                return;
            };
            if !receiver.is_rx_not_empty() {
                return;
            }
            match receiver.read() {
                Ok(byte) => {
                    if RX_QUEUE.borrow(cs).borrow_mut().enqueue(byte).is_err() {
                        RX_STREAM_INVALIDATED.store(true, Ordering::Release);
                    }
                }
                Err(nb::Error::WouldBlock) => {}
                Err(nb::Error::Other(_)) => {
                    // This read sequence clears USART framing/noise/overrun
                    // flags after recording permanent stream invalidation.
                    receiver.clear_idle_interrupt();
                    RX_STREAM_INVALIDATED.store(true, Ordering::Release);
                }
            }
        });
    }

    #[interrupt]
    #[allow(unsafe_code)]
    fn TIM5() {
        // SAFETY: TIM5 is initialized before this interrupt is unmasked. The
        // compare handler clears every motor CCER enable before every CCR.
        unsafe {
            let timer = &*pac::TIM5::ptr();
            if timer.sr.read().cc1if().bit_is_set() {
                timer.sr.modify(|_, writer| writer.cc1if().clear_bit());
                timer.dier.modify(|_, writer| writer.cc1ie().clear_bit());
                if PULSE_ARMED.swap(false, Ordering::AcqRel) {
                    emergency_disable_motor_outputs();
                    PULSE_EXPIRED.store(true, Ordering::Release);
                }
            }
        }
    }

    #[allow(unsafe_code)]
    fn enable_interrupts(nvic: &mut NVIC) {
        // SAFETY: TIM5, USART2 RX ownership, and both fixed queues/atomic flags
        // are initialized before their interrupt lines are unmasked.
        unsafe {
            nvic.set_priority(pac::Interrupt::TIM5, 0);
            nvic.set_priority(pac::Interrupt::USART2, 1);
            NVIC::unmask(pac::Interrupt::TIM5);
            NVIC::unmask(pac::Interrupt::USART2);
        }
    }

    #[allow(unsafe_code)]
    fn emergency_disable_motor_outputs() {
        // SAFETY: idempotent fail-safe MMIO. Channel enables are cleared before
        // duty registers so no forward/reverse pair can remain enabled. TIM2
        // and TIM3 clocks are explicitly enabled first: before clock-tree
        // setup the GPIOs and timer enables are already in reset-safe states,
        // while after setup this makes the clearing writes effective even on
        // fatal paths that did not retain HAL ownership.
        unsafe {
            let rcc = &*pac::RCC::ptr();
            rcc.apb1enr
                .modify(|_, writer| writer.tim2en().set_bit().tim3en().set_bit());
            cortex_m::asm::dsb();

            let left = &*pac::TIM2::ptr();
            left.ccer
                .modify(|_, writer| writer.cc1e().clear_bit().cc2e().clear_bit());
            left.ccr1().write(|writer| writer.ccr().bits(0));
            left.ccr2().write(|writer| writer.ccr().bits(0));

            let right = &*pac::TIM3::ptr();
            right
                .ccer
                .modify(|_, writer| writer.cc1e().clear_bit().cc2e().clear_bit());
            right.ccr1().write(|writer| writer.ccr().bits(0));
            right.ccr2().write(|writer| writer.ccr().bits(0));
        }
        PULSE_ARMED.store(false, Ordering::Release);
    }

    #[panic_handler]
    fn panic(_info: &PanicInfo<'_>) -> ! {
        cortex_m::interrupt::disable();
        emergency_disable_motor_outputs();
        loop {
            cortex_m::asm::wfi();
        }
    }

    #[exception]
    #[allow(unsafe_code)]
    unsafe fn HardFault(_frame: &ExceptionFrame) -> ! {
        cortex_m::interrupt::disable();
        emergency_disable_motor_outputs();
        loop {
            cortex_m::asm::wfi();
        }
    }

    fn fatal() -> ! {
        cortex_m::interrupt::disable();
        emergency_disable_motor_outputs();
        loop {
            cortex_m::asm::wfi();
        }
    }
}

#[cfg(not(target_arch = "arm"))]
mod host {
    use std::{fmt, str::FromStr, time::Duration};

    use clap::{ArgAction, Parser};
    use serde_json::json;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio_serial::{DataBits, FlowControl, Parity, SerialPort, SerialPortBuilderExt, StopBits};

    use super::{
        BOTH_FORWARD_DURATION_MS, BOTH_FORWARD_WATCHDOG_CONFIG_MS, COMMISSIONING_BUILD_ID,
        CommissionResponse, INITIAL_ZERO_DWELL_MS, INTER_PULSE_ZERO_DWELL_MS, LSI_MAX_KHZ,
        LSI_NOMINAL_KHZ, PULSE_DURATION_MS, PULSE_PWM_PERCENT, READY_PERIOD_MS, Recipe,
        ResponseDecoder, ResponseParseError, Trigger, WATCHDOG_BACKSTOP_MARGIN_MS, encode_trigger,
    };

    const SERIAL_BAUD_BPS: u32 = 115_200;
    const SERIAL_BY_ID_PREFIX: &str = "/dev/serial/by-id/";
    const STLINK_ID_PREFIX: &str = "usb-STMicroelectronics_STM32_STLink_";
    const STLINK_UART_SUFFIX: &str = "-if02";
    const MAX_SERIAL_PATH_BYTES: usize = 512;
    const MIN_RUN_TIMEOUT_MS: u64 = 5_000;
    const MAX_RUN_TIMEOUT_MS: u64 = 20_000;
    const MAX_OBSERVED_BYTES: usize = 4_096;
    const MAX_TIMER_EVENT_LATENCY_MS: u32 = 100;
    const MAX_FINAL_EVIDENCE_DELAY_MS: u32 = 100;
    const POST_READY_IO_MARGIN_MS: u32 = 250;

    const fn required_post_ready_ms(recipe: Recipe) -> u32 {
        let inter_segment_ms = match recipe {
            Recipe::LeftThenRight250Ms => INTER_PULSE_ZERO_DWELL_MS,
            Recipe::BothForward10S => 0,
        };
        INITIAL_ZERO_DWELL_MS
            + recipe.total_active_duration_ms()
            + inter_segment_ms
            + MAX_TIMER_EVENT_LATENCY_MS
            + MAX_FINAL_EVIDENCE_DELAY_MS
            + POST_READY_IO_MARGIN_MS
    }

    const _: () = assert!(MIN_RUN_TIMEOUT_MS >= READY_PERIOD_MS as u64);
    const _: () =
        assert!(MAX_RUN_TIMEOUT_MS > required_post_ready_ms(Recipe::BothForward10S) as u64);

    #[derive(Parser, Debug)]
    #[command(
        name = "kiko-motor-commission",
        about = "Execute the one-shot, wheels-off-only Kiko STM32 motor pulse sequence"
    )]
    struct Cli {
        /// Exact Linux persistent ST-Link virtual UART path; no tty fallback.
        #[arg(long)]
        serial_device: PersistentSerialPath,

        /// Required physical-action acknowledgement. Wheels must be removed.
        #[arg(
            long,
            action = ArgAction::SetTrue,
            required_unless_present = "execute_both_forward_10s",
            conflicts_with = "execute_both_forward_10s"
        )]
        execute_wheels_off_sequence: bool,

        /// Execute both forward outputs at 8% for 10,000 ms; wheels must be removed.
        #[arg(
            long,
            action = ArgAction::SetTrue,
            required_unless_present = "execute_wheels_off_sequence",
            conflicts_with = "execute_wheels_off_sequence"
        )]
        execute_both_forward_10s: bool,

        /// One exclusive end-to-end deadline, including Ready observation.
        #[arg(long, default_value_t = 15_000, value_parser = parse_timeout_ms)]
        timeout_ms: u64,
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    struct PersistentSerialPath(Box<str>);

    impl PersistentSerialPath {
        fn as_str(&self) -> &str {
            &self.0
        }
    }

    impl FromStr for PersistentSerialPath {
        type Err = SerialPathError;

        fn from_str(value: &str) -> Result<Self, Self::Err> {
            if value.len() > MAX_SERIAL_PATH_BYTES {
                return Err(SerialPathError::TooLong {
                    actual_bytes: value.len(),
                    maximum_bytes: MAX_SERIAL_PATH_BYTES,
                });
            }
            let suffix = value
                .strip_prefix(SERIAL_BY_ID_PREFIX)
                .ok_or(SerialPathError::NotPersistentById)?;
            if suffix.is_empty() {
                return Err(SerialPathError::EmptyIdentity);
            }
            if suffix.contains('/') || matches!(suffix, "." | "..") {
                return Err(SerialPathError::NonCanonicalIdentity);
            }
            let stlink_serial = suffix
                .strip_prefix(STLINK_ID_PREFIX)
                .and_then(|value| value.strip_suffix(STLINK_UART_SUFFIX))
                .ok_or(SerialPathError::NotStLinkVirtualUart)?;
            if stlink_serial.is_empty()
                || !stlink_serial.bytes().all(|byte| byte.is_ascii_hexdigit())
            {
                return Err(SerialPathError::InvalidStLinkSerial);
            }
            if let Some((index, byte)) = value
                .bytes()
                .enumerate()
                .find(|(_, byte)| !byte.is_ascii_graphic())
            {
                return Err(SerialPathError::NonGraphicAscii { index, byte });
            }
            Ok(Self(value.into()))
        }
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum SerialPathError {
        TooLong {
            actual_bytes: usize,
            maximum_bytes: usize,
        },
        NotPersistentById,
        EmptyIdentity,
        NonCanonicalIdentity,
        NotStLinkVirtualUart,
        InvalidStLinkSerial,
        NonGraphicAscii {
            index: usize,
            byte: u8,
        },
    }

    impl fmt::Display for SerialPathError {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(formatter, "invalid persistent serial path: {self:?}")
        }
    }

    impl std::error::Error for SerialPathError {}

    fn parse_timeout_ms(value: &str) -> Result<u64, TimeoutArgumentError> {
        let timeout_ms = value
            .parse::<u64>()
            .map_err(|_| TimeoutArgumentError::NotUnsignedInteger)?;
        if !(MIN_RUN_TIMEOUT_MS..=MAX_RUN_TIMEOUT_MS).contains(&timeout_ms) {
            return Err(TimeoutArgumentError::OutsideRange {
                actual_ms: timeout_ms,
                minimum_ms: MIN_RUN_TIMEOUT_MS,
                maximum_ms: MAX_RUN_TIMEOUT_MS,
            });
        }
        Ok(timeout_ms)
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum TimeoutArgumentError {
        NotUnsignedInteger,
        OutsideRange {
            actual_ms: u64,
            minimum_ms: u64,
            maximum_ms: u64,
        },
    }

    impl fmt::Display for TimeoutArgumentError {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(formatter, "invalid commissioning timeout: {self:?}")
        }
    }

    impl std::error::Error for TimeoutArgumentError {}

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub(super) enum SequenceStage {
        Ready,
        Accepted,
        LeftPulseComplete,
        RightPulseComplete,
        BothPulseComplete,
        Complete,
    }

    #[derive(Debug)]
    pub(super) enum HostError {
        InvalidExecutionFlagSelection,
        Open(tokio_serial::Error),
        Exclusive(tokio_serial::Error),
        Configure {
            setting: &'static str,
            source: tokio_serial::Error,
        },
        ReadSetting {
            setting: &'static str,
            source: tokio_serial::Error,
        },
        LineConfigurationMismatch {
            baud_rate_bps: u32,
            data_bits: DataBits,
            parity: Parity,
            stop_bits: StopBits,
            flow_control: FlowControl,
        },
        Entropy(getrandom::Error),
        InsufficientDeadlineBeforeTrigger {
            remaining_ms: u64,
            required_ms: u32,
        },
        Timeout {
            stage: SequenceStage,
            timeout_ms: u64,
        },
        Read(std::io::Error),
        Write(std::io::Error),
        Flush(std::io::Error),
        SerialEof,
        ByteBudgetExceeded,
        Decode(ResponseParseError),
        DeviceRejected {
            error_code: u8,
        },
        UnexpectedResponse {
            stage: SequenceStage,
            response: CommissionResponse,
        },
        NonceMismatch {
            stage: SequenceStage,
            expected: u32,
            observed: u32,
        },
        AmbiguousControllerTime {
            segment: &'static str,
            from_ms: u32,
            to_ms: u32,
        },
        ControllerTimingOutsideBounds {
            segment: &'static str,
            observed_ms: u32,
            minimum_ms: u32,
            maximum_ms: u32,
        },
        PostTriggerIndeterminate {
            source: Box<HostError>,
        },
        CompletedButEncodeEvidence(serde_json::Error),
        CompletedButWriteEvidence(std::io::Error),
    }

    impl fmt::Display for HostError {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("wheels-off motor commissioning failed: ")?;
            match self {
                Self::InvalidExecutionFlagSelection => formatter.write_str(
                    "exactly one of --execute-wheels-off-sequence or --execute-both-forward-10s is required before any trigger is sent",
                ),
                Self::Open(source) => {
                    write!(formatter, "could not open exact serial device: {source}")
                }
                Self::Exclusive(source) => {
                    write!(
                        formatter,
                        "could not claim exclusive serial ownership: {source}"
                    )
                }
                Self::Configure { setting, source } => {
                    write!(formatter, "could not configure {setting}: {source}")
                }
                Self::ReadSetting { setting, source } => {
                    write!(formatter, "could not read back {setting}: {source}")
                }
                Self::LineConfigurationMismatch {
                    baud_rate_bps,
                    data_bits,
                    parity,
                    stop_bits,
                    flow_control,
                } => write!(
                    formatter,
                    "serial readback mismatch: baud={baud_rate_bps}, data={data_bits:?}, parity={parity:?}, stop={stop_bits:?}, flow={flow_control:?}"
                ),
                Self::Entropy(source) => write!(formatter, "OS nonce generation failed: {source}"),
                Self::InsufficientDeadlineBeforeTrigger {
                    remaining_ms,
                    required_ms,
                } => write!(
                    formatter,
                    "only {remaining_ms} ms remained after Ready; at least {required_ms} ms is required, so no trigger was sent"
                ),
                Self::Timeout { stage, timeout_ms } => {
                    write!(formatter, "timed out at {stage:?} after {timeout_ms} ms")
                }
                Self::Read(source) => write!(formatter, "serial read failed: {source}"),
                Self::Write(source) => write!(formatter, "serial request write failed: {source}"),
                Self::Flush(source) => write!(formatter, "serial request flush failed: {source}"),
                Self::SerialEof => formatter.write_str("serial device returned EOF"),
                Self::ByteBudgetExceeded => write!(
                    formatter,
                    "response observation exceeded {MAX_OBSERVED_BYTES} bytes"
                ),
                Self::Decode(source) => write!(formatter, "invalid evidence frame: {source:?}"),
                Self::DeviceRejected { error_code } => {
                    write!(
                        formatter,
                        "device rejected the trigger with code {error_code}"
                    )
                }
                Self::UnexpectedResponse { stage, response } => {
                    write!(
                        formatter,
                        "unexpected {response:?} while awaiting {stage:?}"
                    )
                }
                Self::NonceMismatch {
                    stage,
                    expected,
                    observed,
                } => write!(
                    formatter,
                    "nonce mismatch at {stage:?}: expected {expected}, observed {observed}"
                ),
                Self::AmbiguousControllerTime {
                    segment,
                    from_ms,
                    to_ms,
                } => write!(
                    formatter,
                    "ambiguous wrapping controller time for {segment}: {from_ms} -> {to_ms}"
                ),
                Self::ControllerTimingOutsideBounds {
                    segment,
                    observed_ms,
                    minimum_ms,
                    maximum_ms,
                } => write!(
                    formatter,
                    "controller timing for {segment} was {observed_ms} ms, outside [{minimum_ms}, {maximum_ms}] ms"
                ),
                Self::PostTriggerIndeterminate { source } => write!(
                    formatter,
                    "{source}; a trigger write was attempted, so the sequence may have executed or still be executing—do not blindly retry or reset; first establish disabled outputs and inspect the shafts"
                ),
                Self::CompletedButEncodeEvidence(source) => write!(
                    formatter,
                    "the device sequence was already verified Complete, but JSON evidence encoding failed: {source}; do not retry the physical sequence"
                ),
                Self::CompletedButWriteEvidence(source) => write!(
                    formatter,
                    "the device sequence was already verified Complete, but JSON evidence output failed: {source}; do not retry the physical sequence"
                ),
            }
        }
    }

    impl std::error::Error for HostError {
        fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
            match self {
                Self::Open(source)
                | Self::Exclusive(source)
                | Self::Configure { source, .. }
                | Self::ReadSetting { source, .. } => Some(source),
                Self::Entropy(source) => Some(source),
                Self::Read(source) | Self::Write(source) | Self::Flush(source) => Some(source),
                Self::PostTriggerIndeterminate { source } => Some(source.as_ref()),
                Self::CompletedButEncodeEvidence(source) => Some(source),
                Self::CompletedButWriteEvidence(source) => Some(source),
                _ => None,
            }
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct SerialEvidence {
        baud_rate_bps: u32,
        data_bits_8: bool,
        parity_none: bool,
        stop_bits_1: bool,
        flow_control_none: bool,
    }

    #[derive(Clone, Copy, Debug)]
    enum SequenceEvidence {
        LeftThenRight {
            accepted_uptime_ms: u32,
            left_uptime_ms: u32,
            right_uptime_ms: u32,
            complete_uptime_ms: u32,
            accepted_to_left_ms: u32,
            left_to_right_ms: u32,
            right_to_complete_ms: u32,
        },
        BothForward10S {
            accepted_uptime_ms: u32,
            both_uptime_ms: u32,
            complete_uptime_ms: u32,
            accepted_to_both_ms: u32,
            both_to_complete_ms: u32,
        },
    }

    const fn select_recipe(
        left_then_right: bool,
        both_forward_10s: bool,
    ) -> Result<Recipe, HostError> {
        match (left_then_right, both_forward_10s) {
            (true, false) => Ok(Recipe::LeftThenRight250Ms),
            (false, true) => Ok(Recipe::BothForward10S),
            _ => Err(HostError::InvalidExecutionFlagSelection),
        }
    }

    pub(super) async fn run() -> Result<(), HostError> {
        let cli = Cli::parse();
        let recipe = select_recipe(
            cli.execute_wheels_off_sequence,
            cli.execute_both_forward_10s,
        )?;
        let (mut port, serial_evidence) = open_exact_serial(&cli.serial_device)?;
        let host_started_at = tokio::time::Instant::now();
        let deadline = host_started_at + Duration::from_millis(cli.timeout_ms);
        let mut responses = ResponseStream::new(deadline, cli.timeout_ms);

        let ready_uptime_ms = match responses.next(&mut port, SequenceStage::Ready).await? {
            CommissionResponse::Ready { uptime_ms } => uptime_ms,
            CommissionResponse::Rejected { error, .. } => {
                return Err(HostError::DeviceRejected {
                    error_code: error.code(),
                });
            }
            response => {
                return Err(HostError::UnexpectedResponse {
                    stage: SequenceStage::Ready,
                    response,
                });
            }
        };

        let remaining_ms = u64::try_from(
            deadline
                .saturating_duration_since(tokio::time::Instant::now())
                .as_millis(),
        )
        .unwrap_or(u64::MAX);
        let required_ms = required_post_ready_ms(recipe);
        if remaining_ms < u64::from(required_ms) {
            return Err(HostError::InsufficientDeadlineBeforeTrigger {
                remaining_ms,
                required_ms,
            });
        }

        let nonce = os_nonce()?;
        let request = encode_trigger(Trigger { nonce, recipe });
        match tokio::time::timeout_at(deadline, port.write_all(&request)).await {
            Ok(Ok(())) => {}
            Ok(Err(source)) => return Err(post_trigger(HostError::Write(source))),
            Err(_) => {
                return Err(post_trigger(HostError::Timeout {
                    stage: SequenceStage::Accepted,
                    timeout_ms: cli.timeout_ms,
                }));
            }
        }
        match tokio::time::timeout_at(deadline, port.flush()).await {
            Ok(Ok(())) => {}
            Ok(Err(source)) => return Err(post_trigger(HostError::Flush(source))),
            Err(_) => {
                return Err(post_trigger(HostError::Timeout {
                    stage: SequenceStage::Accepted,
                    timeout_ms: cli.timeout_ms,
                }));
            }
        }

        let sequence = observe_sequence(&mut port, &mut responses, nonce, recipe)
            .await
            .map_err(post_trigger)?;
        let host_elapsed_ms = u64::try_from(host_started_at.elapsed().as_millis())
            .expect("bounded CLI deadline fits u64 milliseconds");

        let (compiled_recipe, controller_uptime, controller_deltas) = match sequence {
            SequenceEvidence::LeftThenRight {
                accepted_uptime_ms,
                left_uptime_ms,
                right_uptime_ms,
                complete_uptime_ms,
                accepted_to_left_ms,
                left_to_right_ms,
                right_to_complete_ms,
            } => (
                json!({
                    "id": Recipe::LeftThenRight250Ms.wire(),
                    "name": "left_then_right_250ms",
                    "segments": [
                        {"index": 0, "outputs": ["left_forward"], "pwm_percent": PULSE_PWM_PERCENT, "duration_ms": PULSE_DURATION_MS},
                        {"index": 1, "outputs": ["right_forward"], "pwm_percent": PULSE_PWM_PERCENT, "duration_ms": PULSE_DURATION_MS}
                    ],
                    "initial_zero_dwell_ms": INITIAL_ZERO_DWELL_MS,
                    "inter_segment_zero_dwell_ms": INTER_PULSE_ZERO_DWELL_MS,
                    "nominal_active_watchdog_ms": Recipe::LeftThenRight250Ms.watchdog_period_ms(),
                }),
                json!({
                    "ready": ready_uptime_ms,
                    "accepted": accepted_uptime_ms,
                    "left_pulse_complete": left_uptime_ms,
                    "right_pulse_complete": right_uptime_ms,
                    "sequence_complete": complete_uptime_ms,
                }),
                json!({
                    "accepted_to_left_complete": accepted_to_left_ms,
                    "left_complete_to_right_complete": left_to_right_ms,
                    "right_complete_to_sequence_complete": right_to_complete_ms,
                }),
            ),
            SequenceEvidence::BothForward10S {
                accepted_uptime_ms,
                both_uptime_ms,
                complete_uptime_ms,
                accepted_to_both_ms,
                both_to_complete_ms,
            } => (
                json!({
                    "id": Recipe::BothForward10S.wire(),
                    "name": "both_forward_10s",
                    "segments": [
                        {"index": 0, "outputs": ["left_forward", "right_forward"], "pwm_percent": PULSE_PWM_PERCENT, "duration_ms": BOTH_FORWARD_DURATION_MS}
                    ],
                    "initial_zero_dwell_ms": INITIAL_ZERO_DWELL_MS,
                    "inter_segment_zero_dwell_ms": 0,
                    "watchdog_hal_config_ms_at_nominal_32khz_lsi": BOTH_FORWARD_WATCHDOG_CONFIG_MS,
                    "watchdog_lsi_model_nominal_khz": LSI_NOMINAL_KHZ,
                    "watchdog_lsi_datasheet_max_khz": LSI_MAX_KHZ,
                    "watchdog_designed_minimum_margin_at_max_lsi_ms": WATCHDOG_BACKSTOP_MARGIN_MS,
                }),
                json!({
                    "ready": ready_uptime_ms,
                    "accepted": accepted_uptime_ms,
                    "both_forward_complete": both_uptime_ms,
                    "sequence_complete": complete_uptime_ms,
                }),
                json!({
                    "accepted_to_both_complete": accepted_to_both_ms,
                    "both_complete_to_sequence_complete": both_to_complete_ms,
                }),
            ),
        };

        let evidence = json!({
            "schema_version": 2,
            "observation_kind": "wheels_off_stm32_motor_commissioning",
            "serial_by_id_path": cli.serial_device.as_str(),
            "exclusive_lock_requested_and_set": true,
            "serial_readback": {
                "baud_rate_bps": serial_evidence.baud_rate_bps,
                "data_bits_8": serial_evidence.data_bits_8,
                "parity_none": serial_evidence.parity_none,
                "stop_bits_1": serial_evidence.stop_bits_1,
                "flow_control_none": serial_evidence.flow_control_none,
            },
            "commissioning_build_id": COMMISSIONING_BUILD_ID,
            "commissioning_build_id_hex": format!("0x{COMMISSIONING_BUILD_ID:08x}"),
            "protocol_version": super::PROTOCOL_VERSION,
            "host_nonce": nonce.get(),
            "compiled_recipe": compiled_recipe,
            "controller_uptime_ms": controller_uptime,
            "verified_controller_deltas_ms": controller_deltas,
            "host_elapsed_ms": host_elapsed_ms,
            "observed_serial_bytes": responses.observed_bytes,
            "sequence_complete": true,
            "evidence_boundary": "software admission and STM32 nominal timer evidence only; wheels-off state and physical shaft motion require separate operator observation. The 10,000 ms TIM5 cutoff plus a watchdog configuration designed to leave at least 500 ms at the datasheet maximum LSI frequency is not an arbitrary-fault physical bound; the real watchdog interval may be much longer at lower LSI frequency. This is not velocity calibration, KRP2 motion authority, closed-loop control, or MPC validation",
        });
        let stdout = std::io::stdout();
        let mut stdout = stdout.lock();
        serde_json::to_writer_pretty(&mut stdout, &evidence)
            .map_err(HostError::CompletedButEncodeEvidence)?;
        std::io::Write::write_all(&mut stdout, b"\n")
            .map_err(HostError::CompletedButWriteEvidence)?;
        Ok(())
    }

    async fn observe_sequence(
        port: &mut tokio_serial::SerialStream,
        responses: &mut ResponseStream,
        nonce: core::num::NonZeroU32,
        recipe: Recipe,
    ) -> Result<SequenceEvidence, HostError> {
        let accepted_uptime_ms = loop {
            match responses.next(port, SequenceStage::Accepted).await? {
                // A periodic Ready can already be in the UART/USB buffers when
                // the one and only request is written. It is allowed only
                // before Accepted.
                CommissionResponse::Ready { .. } => {}
                CommissionResponse::Accepted {
                    nonce: observed,
                    recipe: observed_recipe,
                    uptime_ms,
                } if observed_recipe == recipe => {
                    require_nonce(SequenceStage::Accepted, nonce, observed)?;
                    break uptime_ms;
                }
                CommissionResponse::Rejected { error, .. } => {
                    return Err(HostError::DeviceRejected {
                        error_code: error.code(),
                    });
                }
                response => {
                    return Err(HostError::UnexpectedResponse {
                        stage: SequenceStage::Accepted,
                        response,
                    });
                }
            }
        };

        match recipe {
            Recipe::LeftThenRight250Ms => {
                let left_uptime_ms = responses
                    .expect_pulse(port, nonce, recipe, 0, SequenceStage::LeftPulseComplete)
                    .await?;
                let right_uptime_ms = responses
                    .expect_pulse(port, nonce, recipe, 1, SequenceStage::RightPulseComplete)
                    .await?;
                let complete_uptime_ms = responses.expect_complete(port, nonce, recipe).await?;
                Ok(SequenceEvidence::LeftThenRight {
                    accepted_uptime_ms,
                    left_uptime_ms,
                    right_uptime_ms,
                    complete_uptime_ms,
                    accepted_to_left_ms: bounded_controller_delta(
                        "accepted_to_left_complete",
                        accepted_uptime_ms,
                        left_uptime_ms,
                        INITIAL_ZERO_DWELL_MS + PULSE_DURATION_MS,
                        INITIAL_ZERO_DWELL_MS + PULSE_DURATION_MS + MAX_TIMER_EVENT_LATENCY_MS,
                    )?,
                    left_to_right_ms: bounded_controller_delta(
                        "left_complete_to_right_complete",
                        left_uptime_ms,
                        right_uptime_ms,
                        INTER_PULSE_ZERO_DWELL_MS + PULSE_DURATION_MS,
                        INTER_PULSE_ZERO_DWELL_MS + PULSE_DURATION_MS + MAX_TIMER_EVENT_LATENCY_MS,
                    )?,
                    right_to_complete_ms: bounded_controller_delta(
                        "right_complete_to_sequence_complete",
                        right_uptime_ms,
                        complete_uptime_ms,
                        0,
                        MAX_FINAL_EVIDENCE_DELAY_MS,
                    )?,
                })
            }
            Recipe::BothForward10S => {
                let both_uptime_ms = responses
                    .expect_pulse(port, nonce, recipe, 0, SequenceStage::BothPulseComplete)
                    .await?;
                let complete_uptime_ms = responses.expect_complete(port, nonce, recipe).await?;
                Ok(SequenceEvidence::BothForward10S {
                    accepted_uptime_ms,
                    both_uptime_ms,
                    complete_uptime_ms,
                    accepted_to_both_ms: bounded_controller_delta(
                        "accepted_to_both_complete",
                        accepted_uptime_ms,
                        both_uptime_ms,
                        INITIAL_ZERO_DWELL_MS + BOTH_FORWARD_DURATION_MS,
                        INITIAL_ZERO_DWELL_MS
                            + BOTH_FORWARD_DURATION_MS
                            + MAX_TIMER_EVENT_LATENCY_MS,
                    )?,
                    both_to_complete_ms: bounded_controller_delta(
                        "both_complete_to_sequence_complete",
                        both_uptime_ms,
                        complete_uptime_ms,
                        0,
                        MAX_FINAL_EVIDENCE_DELAY_MS,
                    )?,
                })
            }
        }
    }

    fn post_trigger(source: HostError) -> HostError {
        HostError::PostTriggerIndeterminate {
            source: Box::new(source),
        }
    }

    fn open_exact_serial(
        device: &PersistentSerialPath,
    ) -> Result<(tokio_serial::SerialStream, SerialEvidence), HostError> {
        let builder = tokio_serial::new(device.as_str(), SERIAL_BAUD_BPS)
            .data_bits(DataBits::Eight)
            .parity(Parity::None)
            .stop_bits(StopBits::One)
            .flow_control(FlowControl::None);
        let mut port = builder.open_native_async().map_err(HostError::Open)?;
        #[cfg(unix)]
        port.set_exclusive(true).map_err(HostError::Exclusive)?;

        apply_setting("data bits", || port.set_data_bits(DataBits::Eight))?;
        apply_setting("parity", || port.set_parity(Parity::None))?;
        apply_setting("stop bits", || port.set_stop_bits(StopBits::One))?;
        apply_setting("flow control", || port.set_flow_control(FlowControl::None))?;

        let baud_rate_bps = read_setting("baud rate", || port.baud_rate())?;
        let data_bits = read_setting("data bits", || port.data_bits())?;
        let parity = read_setting("parity", || port.parity())?;
        let stop_bits = read_setting("stop bits", || port.stop_bits())?;
        let flow_control = read_setting("flow control", || port.flow_control())?;
        if baud_rate_bps != SERIAL_BAUD_BPS
            || data_bits != DataBits::Eight
            || parity != Parity::None
            || stop_bits != StopBits::One
            || flow_control != FlowControl::None
        {
            return Err(HostError::LineConfigurationMismatch {
                baud_rate_bps,
                data_bits,
                parity,
                stop_bits,
                flow_control,
            });
        }
        Ok((
            port,
            SerialEvidence {
                baud_rate_bps,
                data_bits_8: true,
                parity_none: true,
                stop_bits_1: true,
                flow_control_none: true,
            },
        ))
    }

    fn apply_setting(
        setting: &'static str,
        apply: impl FnOnce() -> tokio_serial::Result<()>,
    ) -> Result<(), HostError> {
        apply().map_err(|source| HostError::Configure { setting, source })
    }

    fn read_setting<T>(
        setting: &'static str,
        read: impl FnOnce() -> tokio_serial::Result<T>,
    ) -> Result<T, HostError> {
        read().map_err(|source| HostError::ReadSetting { setting, source })
    }

    struct ResponseStream {
        decoder: ResponseDecoder,
        deadline: tokio::time::Instant,
        timeout_ms: u64,
        observed_bytes: usize,
    }

    impl ResponseStream {
        const fn new(deadline: tokio::time::Instant, timeout_ms: u64) -> Self {
            Self {
                decoder: ResponseDecoder::new(),
                deadline,
                timeout_ms,
                observed_bytes: 0,
            }
        }

        async fn next(
            &mut self,
            port: &mut tokio_serial::SerialStream,
            stage: SequenceStage,
        ) -> Result<CommissionResponse, HostError> {
            loop {
                let mut byte = [0_u8; 1];
                let count = match tokio::time::timeout_at(self.deadline, port.read(&mut byte)).await
                {
                    Ok(Ok(0)) => return Err(HostError::SerialEof),
                    Ok(Ok(count)) => count,
                    Ok(Err(source)) => return Err(HostError::Read(source)),
                    Err(_) => {
                        return Err(HostError::Timeout {
                            stage,
                            timeout_ms: self.timeout_ms,
                        });
                    }
                };
                self.observed_bytes = self
                    .observed_bytes
                    .checked_add(count)
                    .ok_or(HostError::ByteBudgetExceeded)?;
                if self.observed_bytes > MAX_OBSERVED_BYTES {
                    return Err(HostError::ByteBudgetExceeded);
                }
                if let Some(response) = self.decoder.push(byte[0]) {
                    return response.map_err(HostError::Decode);
                }
            }
        }

        async fn expect_pulse(
            &mut self,
            port: &mut tokio_serial::SerialStream,
            expected_nonce: core::num::NonZeroU32,
            expected_recipe: Recipe,
            expected_index: u8,
            stage: SequenceStage,
        ) -> Result<u32, HostError> {
            let expected_segment = expected_recipe
                .segment(expected_index)
                .expect("callers use a compiled recipe segment");
            match self.next(port, stage).await? {
                CommissionResponse::PulseCompleted {
                    nonce,
                    recipe,
                    index,
                    outputs,
                    commanded_duration_ms,
                    uptime_ms,
                } if recipe == expected_recipe
                    && index == expected_index
                    && outputs == expected_segment.outputs
                    && commanded_duration_ms == expected_segment.duration_ms =>
                {
                    require_nonce(stage, expected_nonce, nonce)?;
                    Ok(uptime_ms)
                }
                CommissionResponse::Rejected { error, .. } => Err(HostError::DeviceRejected {
                    error_code: error.code(),
                }),
                response => Err(HostError::UnexpectedResponse { stage, response }),
            }
        }

        async fn expect_complete(
            &mut self,
            port: &mut tokio_serial::SerialStream,
            expected_nonce: core::num::NonZeroU32,
            expected_recipe: Recipe,
        ) -> Result<u32, HostError> {
            match self.next(port, SequenceStage::Complete).await? {
                CommissionResponse::Complete {
                    nonce,
                    recipe,
                    uptime_ms,
                } if recipe == expected_recipe => {
                    require_nonce(SequenceStage::Complete, expected_nonce, nonce)?;
                    Ok(uptime_ms)
                }
                CommissionResponse::Rejected { error, .. } => Err(HostError::DeviceRejected {
                    error_code: error.code(),
                }),
                response => Err(HostError::UnexpectedResponse {
                    stage: SequenceStage::Complete,
                    response,
                }),
            }
        }
    }

    fn os_nonce() -> Result<core::num::NonZeroU32, HostError> {
        let mut bytes = [0_u8; 4];
        getrandom::fill(&mut bytes).map_err(HostError::Entropy)?;
        let value = u32::from_le_bytes(bytes) | 1;
        Ok(core::num::NonZeroU32::new(value).expect("bitwise OR proves nonzero"))
    }

    fn require_nonce(
        stage: SequenceStage,
        expected: core::num::NonZeroU32,
        observed: core::num::NonZeroU32,
    ) -> Result<(), HostError> {
        if expected == observed {
            Ok(())
        } else {
            Err(HostError::NonceMismatch {
                stage,
                expected: expected.get(),
                observed: observed.get(),
            })
        }
    }

    fn bounded_controller_delta(
        segment: &'static str,
        from_ms: u32,
        to_ms: u32,
        minimum_ms: u32,
        maximum_ms: u32,
    ) -> Result<u32, HostError> {
        let observed_ms = to_ms.wrapping_sub(from_ms);
        if observed_ms >= (1_u32 << 31) {
            return Err(HostError::AmbiguousControllerTime {
                segment,
                from_ms,
                to_ms,
            });
        }
        if !(minimum_ms..=maximum_ms).contains(&observed_ms) {
            return Err(HostError::ControllerTimingOutsideBounds {
                segment,
                observed_ms,
                minimum_ms,
                maximum_ms,
            });
        }
        Ok(observed_ms)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        const KNOWN_STLINK: &str =
            "/dev/serial/by-id/usb-STMicroelectronics_STM32_STLink_066EFF313946303143221230-if02";

        #[test]
        fn serial_identity_accepts_only_the_persistent_stlink_uart_interface() {
            let parsed = PersistentSerialPath::from_str(KNOWN_STLINK).expect("known ST-Link UART");
            assert_eq!(parsed.as_str(), KNOWN_STLINK);

            for invalid in [
                "/dev/ttyACM0",
                "/dev/serial/by-id/usb-STMicroelectronics_STM32_STLink_066EFF313946303143221230-if00",
                "/dev/serial/by-id/usb-kiko_controller_066EFF313946303143221230-if02",
                "/dev/serial/by-id/usb-STMicroelectronics_STM32_STLink_not-hex-if02",
            ] {
                assert!(PersistentSerialPath::from_str(invalid).is_err());
            }
        }

        #[test]
        fn timeout_admission_reserves_the_complete_post_ready_evidence_window() {
            assert!(parse_timeout_ms("4999").is_err());
            assert_eq!(parse_timeout_ms("5000"), Ok(5_000));
            assert_eq!(parse_timeout_ms("15000"), Ok(15_000));
            assert_eq!(parse_timeout_ms("20000"), Ok(20_000));
            assert!(parse_timeout_ms("20001").is_err());
            assert!(MAX_RUN_TIMEOUT_MS > u64::from(required_post_ready_ms(Recipe::BothForward10S)));
        }

        #[test]
        fn execution_flags_select_exactly_one_recipe() {
            assert!(matches!(
                select_recipe(true, false),
                Ok(Recipe::LeftThenRight250Ms)
            ));
            assert!(matches!(
                select_recipe(false, true),
                Ok(Recipe::BothForward10S)
            ));
            assert!(select_recipe(false, false).is_err());
            assert!(select_recipe(true, true).is_err());
        }

        #[test]
        fn controller_delta_is_wrapping_aware_and_rejects_ambiguity() {
            assert_eq!(
                bounded_controller_delta("wrap", u32::MAX - 10, 20, 31, 31)
                    .expect("unambiguous wrapping delta"),
                31
            );
            assert!(matches!(
                bounded_controller_delta("ambiguous", 10, 9, 0, u32::MAX),
                Err(HostError::AmbiguousControllerTime { .. })
            ));
        }

        #[test]
        fn post_trigger_errors_forbid_blind_retry() {
            let message = post_trigger(HostError::SerialEof).to_string();
            assert!(message.contains("may have executed"));
            assert!(message.contains("do not blindly retry or reset"));
        }
    }
}

#[cfg(not(target_arch = "arm"))]
#[tokio::main(flavor = "current_thread")]
async fn main() -> std::process::ExitCode {
    match host::run().await {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            let stderr = std::io::stderr();
            let mut stderr = stderr.lock();
            let _reported = std::io::Write::write_fmt(&mut stderr, format_args!("{error}\n"));
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(recipe: Recipe, nonce: u32) -> [u8; REQUEST_BYTES] {
        let mut frame = [0_u8; REQUEST_BYTES];
        frame[..4].copy_from_slice(&REQUEST_MAGIC);
        frame[4] = PROTOCOL_VERSION;
        frame[5] = recipe.command();
        frame[6] = recipe.wire();
        frame[7] = PULSE_PWM_PERCENT;
        frame[8..12].copy_from_slice(&nonce.to_le_bytes());
        let checksum = crc32(&frame[..12]);
        frame[12..16].copy_from_slice(&checksum.to_le_bytes());
        frame
    }

    fn complete_response(recipe: Recipe, nonce: u32, uptime_ms: u32) -> [u8; RESPONSE_BYTES] {
        encode_response(OutboundResponse::Complete {
            trigger: Trigger {
                nonce: NonZeroU32::new(nonce).expect("test response nonce is nonzero"),
                recipe,
            },
            uptime_ms,
        })
    }

    #[test]
    fn crc_matches_standard_check_vector() {
        assert_eq!(crc32(b"123456789"), 0xcbf4_3926);
    }

    #[test]
    fn both_crc_bound_recipes_parse_once_into_typed_triggers() {
        for recipe in [Recipe::LeftThenRight250Ms, Recipe::BothForward10S] {
            let nonce = NonZeroU32::new(0x1234_5678).expect("nonzero");
            let encoded = encode_trigger(Trigger { nonce, recipe });
            assert_eq!(encoded, request(recipe, nonce.get()));
            assert_eq!(parse_trigger(&encoded), Ok(Trigger { nonce, recipe }));
        }
    }

    #[test]
    fn command_recipe_pairs_cannot_be_mixed() {
        for recipe in [Recipe::LeftThenRight250Ms, Recipe::BothForward10S] {
            let other = match recipe {
                Recipe::LeftThenRight250Ms => Recipe::BothForward10S,
                Recipe::BothForward10S => Recipe::LeftThenRight250Ms,
            };
            let mut frame = request(recipe, 1);
            frame[5] = other.command();
            let checksum = crc32(&frame[..12]);
            frame[12..16].copy_from_slice(&checksum.to_le_bytes());
            assert_eq!(parse_trigger(&frame), Err(TriggerError::Command));

            let mut unknown = request(recipe, 1);
            unknown[6] = 0xff;
            let checksum = crc32(&unknown[..12]);
            unknown[12..16].copy_from_slice(&checksum.to_le_bytes());
            assert_eq!(parse_trigger(&unknown), Err(TriggerError::Recipe));
        }
    }

    #[test]
    fn every_single_bit_request_corruption_is_rejected_for_each_recipe() {
        for recipe in [Recipe::LeftThenRight250Ms, Recipe::BothForward10S] {
            let valid = request(recipe, 0x1234_5678);
            for bit in 0..(REQUEST_BYTES * 8) {
                let mut corrupt = valid;
                corrupt[bit / 8] ^= 1 << (bit % 8);
                assert!(
                    parse_trigger(&corrupt).is_err(),
                    "recipe {recipe:?}, bit {bit} retained admission"
                );
            }
        }
    }

    #[test]
    fn stream_decoder_ignores_noise_and_decodes_the_both_recipe_once() {
        let mut decoder = TriggerDecoder::new();
        for byte in [0, b'K', 0, b'K', b'M', b'K', b'M', b'C', b'2'] {
            assert_eq!(decoder.push(byte), None);
        }
        let frame = request(Recipe::BothForward10S, 7);
        let mut event = None;
        for &byte in &frame[4..] {
            event = event.or(decoder.push(byte));
        }
        assert_eq!(
            event,
            Some(DecodeEvent::Trigger(Ok(Trigger {
                nonce: NonZeroU32::new(7).expect("nonzero"),
                recipe: Recipe::BothForward10S,
            })))
        );
    }

    #[test]
    fn sequential_recipe_preserves_left_zero_right_state_outputs() {
        let nonce = NonZeroU32::new(9).expect("nonzero");
        let trigger = Trigger {
            nonce,
            recipe: Recipe::LeftThenRight250Ms,
        };
        let mut machine = CommissionMachine::new();
        assert_eq!(
            machine.accept(trigger, 100),
            MachineAction::Accepted {
                nonce,
                recipe: Recipe::LeftThenRight250Ms,
            }
        );
        assert_eq!(machine.poll_zero_phase(599), MachineAction::None);
        assert_eq!(
            machine.poll_zero_phase(600),
            MachineAction::PreparePulse {
                nonce,
                recipe: Recipe::LeftThenRight250Ms,
                index: 0,
                outputs: ForwardOutputs::Left,
                duration_ms: 250,
            }
        );
        assert!(machine.outputs_must_be_safe());
        assert!(matches!(
            machine.begin_prepared_pulse(605),
            MachineAction::StartPulse {
                deadline_ms: 855,
                ..
            }
        ));
        assert_eq!(
            machine.pulse_expired(855),
            MachineAction::PulseStopped {
                nonce,
                recipe: Recipe::LeftThenRight250Ms,
                index: 0,
                outputs: ForwardOutputs::Left,
                duration_ms: 250,
                complete: false,
            }
        );
        assert!(machine.outputs_must_be_safe());
        assert_eq!(machine.poll_zero_phase(1_354), MachineAction::None);
        assert!(matches!(
            machine.poll_zero_phase(1_355),
            MachineAction::PreparePulse {
                outputs: ForwardOutputs::Right,
                ..
            }
        ));
        assert!(matches!(
            machine.begin_prepared_pulse(1_360),
            MachineAction::StartPulse {
                deadline_ms: 1_610,
                ..
            }
        ));
        assert!(matches!(
            machine.pulse_expired(1_610),
            MachineAction::PulseStopped { complete: true, .. }
        ));
        assert!(matches!(
            machine.phase,
            Phase::LockedComplete {
                nonce: observed,
                recipe: Recipe::LeftThenRight250Ms,
            } if observed == nonce
        ));
    }

    #[test]
    fn both_recipe_commands_both_outputs_for_exactly_ten_thousand_timer_ms() {
        let nonce = NonZeroU32::new(11).expect("nonzero");
        let trigger = Trigger {
            nonce,
            recipe: Recipe::BothForward10S,
        };
        let mut machine = CommissionMachine::new();
        machine.accept(trigger, 1_000);
        assert_eq!(machine.poll_zero_phase(1_499), MachineAction::None);
        assert_eq!(
            machine.poll_zero_phase(1_500),
            MachineAction::PreparePulse {
                nonce,
                recipe: Recipe::BothForward10S,
                index: 0,
                outputs: ForwardOutputs::Both,
                duration_ms: 10_000,
            }
        );
        assert!(machine.outputs_must_be_safe());
        assert_eq!(
            machine.begin_prepared_pulse(1_507),
            MachineAction::StartPulse {
                nonce,
                recipe: Recipe::BothForward10S,
                index: 0,
                outputs: ForwardOutputs::Both,
                duration_ms: 10_000,
                deadline_ms: 11_507,
            }
        );
        assert_eq!(
            machine.pulse_expired(11_507),
            MachineAction::PulseStopped {
                nonce,
                recipe: Recipe::BothForward10S,
                index: 0,
                outputs: ForwardOutputs::Both,
                duration_ms: 10_000,
                complete: true,
            }
        );
        assert!(machine.outputs_must_be_safe());
    }

    #[test]
    fn early_cutoff_evidence_locks_fault_and_never_retries() {
        let nonce = NonZeroU32::new(1).expect("nonzero");
        let mut machine = CommissionMachine::new();
        machine.accept(
            Trigger {
                nonce,
                recipe: Recipe::BothForward10S,
            },
            0,
        );
        machine.poll_zero_phase(INITIAL_ZERO_DWELL_MS);
        machine.begin_prepared_pulse(INITIAL_ZERO_DWELL_MS);
        assert_eq!(
            machine.pulse_expired(INITIAL_ZERO_DWELL_MS + BOTH_FORWARD_DURATION_MS - 1),
            MachineAction::None
        );
        assert!(matches!(machine.phase, Phase::LockedFault));
        assert!(machine.outputs_must_be_safe());
        assert_eq!(
            machine.accept(
                Trigger {
                    nonce,
                    recipe: Recipe::LeftThenRight250Ms
                },
                99
            ),
            MachineAction::None
        );
    }

    #[test]
    fn deadlines_and_ten_second_recipe_are_correct_across_u32_wrap() {
        let nonce = NonZeroU32::new(1).expect("nonzero");
        let mut machine = CommissionMachine::new();
        machine.accept(
            Trigger {
                nonce,
                recipe: Recipe::BothForward10S,
            },
            u32::MAX - 100,
        );
        let start = 399;
        assert!(matches!(
            machine.poll_zero_phase(start),
            MachineAction::PreparePulse { .. }
        ));
        assert!(matches!(
            machine.begin_prepared_pulse(start.wrapping_add(7)),
            MachineAction::StartPulse {
                deadline_ms: 10_406,
                ..
            }
        ));
        assert!(matches!(
            machine.pulse_expired(10_406),
            MachineAction::PulseStopped { complete: true, .. }
        ));
    }

    #[test]
    fn watchdog_profiles_preserve_short_recipe_and_bound_long_recipe_at_max_lsi() {
        assert_eq!(Recipe::LeftThenRight250Ms.watchdog_period_ms(), 500);
        assert_eq!(Recipe::BothForward10S.watchdog_period_ms(), 15_500);
        assert!(
            BOTH_FORWARD_WATCHDOG_CONFIG_MS * LSI_NOMINAL_KHZ / LSI_MAX_KHZ
                >= BOTH_FORWARD_DURATION_MS + WATCHDOG_BACKSTOP_MARGIN_MS
        );
    }

    #[test]
    fn response_is_recipe_timing_output_and_crc_self_identifying() {
        for recipe in [Recipe::LeftThenRight250Ms, Recipe::BothForward10S] {
            let frame = complete_response(recipe, 17, 42);
            assert_eq!(&frame[..4], b"KMR2");
            assert_eq!(frame[4], PROTOCOL_VERSION);
            assert_eq!(frame[6], recipe.wire());
            assert_eq!(
                u32::from_le_bytes(frame[20..24].try_into().expect("duration bytes")),
                recipe.total_active_duration_ms()
            );
            assert_eq!(
                u32::from_le_bytes(frame[28..32].try_into().expect("CRC bytes")),
                crc32(&frame[..28])
            );
            assert_eq!(
                parse_response(&frame),
                Ok(CommissionResponse::Complete {
                    nonce: NonZeroU32::new(17).expect("nonzero"),
                    recipe,
                    uptime_ms: 42,
                })
            );
        }
    }

    #[test]
    fn response_parser_rejects_recipe_timing_and_output_mismatch() {
        let mut frame = encode_response(OutboundResponse::PulseCompleted {
            nonce: NonZeroU32::new(23).expect("nonzero"),
            pulse: CompletedPulse::Both10S,
            uptime_ms: 77,
        });
        assert!(parse_response(&frame).is_ok());
        frame[24] = ForwardOutputs::Left.wire();
        let checksum = crc32(&frame[..28]);
        frame[28..32].copy_from_slice(&checksum.to_le_bytes());
        assert_eq!(parse_response(&frame), Err(ResponseParseError::Fields));
    }

    #[test]
    fn every_single_bit_evidence_corruption_is_rejected_for_each_recipe() {
        for recipe in [Recipe::LeftThenRight250Ms, Recipe::BothForward10S] {
            let valid = complete_response(recipe, 17, 42);
            for bit in 0..(RESPONSE_BYTES * 8) {
                let mut corrupt = valid;
                corrupt[bit / 8] ^= 1 << (bit % 8);
                assert!(
                    parse_response(&corrupt).is_err(),
                    "recipe {recipe:?}, response bit {bit} retained admission"
                );
            }
        }
    }
}
