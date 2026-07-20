#![cfg_attr(target_arch = "arm", no_main)]
#![cfg_attr(target_arch = "arm", no_std)]
#![cfg_attr(not(target_arch = "arm"), allow(dead_code))]

//! Wheels-off-only STM32F446 motor commissioning image.
//!
//! This is deliberately a separate binary and Cargo feature from
//! `firmware_v2.rs`. It is not KRP2 motion authority, actuator calibration,
//! velocity calibration, closed-loop control, or MPC validation. Its only
//! physical action is one finite sequence per reset:
//!
//! 1. wait at disabled/zero for one valid 16-byte `KMC1` trigger;
//! 2. remain at zero for 500 ms;
//! 3. pulse the left-forward output at 8% for 250 ms;
//! 4. remain at zero for 500 ms;
//! 5. pulse the right-forward output at 8% for 250 ms; and
//! 6. disable and zero every output until reset.
//!
//! A highest-priority TIM5 compare ISR normally cuts each pulse off at 250 ms,
//! with a post-enable deadline re-sample, exception-path emergency MMIO, and a
//! 500 ms independent watchdog as layered backstops. This proves the nominal
//! software command bound; it does **not** prove an unconditional sub-300 ms
//! physical cutoff under arbitrary CPU, clock, timer, driver, or wiring fault.
//! A watchdog reset returns to disabled boot state and still requires a new
//! typed trigger before another sequence can run.
//!
//! Trigger frame (little endian, 16 bytes):
//!
//! ```text
//! 0..4   "KMC1"
//! 4      protocol version (1)
//! 5      command (0xa5 = execute the compiled wheels-off sequence)
//! 6      sequence id (1 = left then right)
//! 7      expected compiled maximum PWM percent (8)
//! 8..12  non-zero host nonce, u32 LE
//! 12..16 CRC-32/ISO-HDLC of bytes 0..12, u32 LE
//! ```
//!
//! Evidence frame (little endian, 24 bytes):
//!
//! ```text
//! 0..4   "KMR1"
//! 4      protocol version (1)
//! 5      event code
//! 6      zero-based pulse index, or 0xff
//! 7      event detail
//! 8..12  echoed host nonce, or zero before admission
//! 12..16 TIM5 uptime in milliseconds, u32 LE
//! 16..20 commissioning build id (0x4b4d_4301), u32 LE
//! 20..24 CRC-32/ISO-HDLC of bytes 0..20, u32 LE
//! ```

use core::num::NonZeroU32;

const REQUEST_MAGIC: [u8; 4] = *b"KMC1";
const RESPONSE_MAGIC: [u8; 4] = *b"KMR1";
const PROTOCOL_VERSION: u8 = 1;
const EXECUTE_COMMAND: u8 = 0xa5;
const SEQUENCE_ID: u8 = 1;
const REQUEST_BYTES: usize = 16;
const RESPONSE_BYTES: usize = 24;
const COMMISSIONING_BUILD_ID: u32 = 0x4b4d_4301;

const PWM_FREQUENCY_HZ: u16 = 20_000;
const PULSE_PWM_PERCENT: u8 = 8;
const PULSE_DURATION_MS: u32 = 250;
const INITIAL_ZERO_DWELL_MS: u32 = 500;
const INTER_PULSE_ZERO_DWELL_MS: u32 = 500;
const WATCHDOG_PERIOD_MS: u16 = 500;
const MAIN_LOOP_DELAY_MS: u32 = 1;
const SERIAL_RESPONSE_TIMEOUT_MS: u32 = 50;
const READY_PERIOD_MS: u32 = 1_000;

const _: () = assert!(PULSE_PWM_PERCENT > 0 && PULSE_PWM_PERCENT <= 8);
const _: () = assert!(PULSE_DURATION_MS > 0 && PULSE_DURATION_MS <= 300);
const _: () = assert!(INITIAL_ZERO_DWELL_MS > 0);
const _: () = assert!(INTER_PULSE_ZERO_DWELL_MS > 0);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Trigger {
    nonce: NonZeroU32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
enum TriggerError {
    Magic = 1,
    Version = 2,
    Command = 3,
    Sequence = 4,
    ExpectedPwm = 5,
    Nonce = 6,
    Checksum = 7,
}

impl TriggerError {
    const fn code(self) -> u8 {
        self as u8
    }
}

fn parse_trigger(frame: &[u8; REQUEST_BYTES]) -> Result<Trigger, TriggerError> {
    if frame[..4] != REQUEST_MAGIC {
        return Err(TriggerError::Magic);
    }
    if frame[4] != PROTOCOL_VERSION {
        return Err(TriggerError::Version);
    }
    if frame[5] != EXECUTE_COMMAND {
        return Err(TriggerError::Command);
    }
    if frame[6] != SEQUENCE_ID {
        return Err(TriggerError::Sequence);
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
    Ok(Trigger { nonce })
}

#[cfg(any(not(target_arch = "arm"), test))]
fn encode_trigger(trigger: Trigger) -> [u8; REQUEST_BYTES] {
    let mut frame = [0_u8; REQUEST_BYTES];
    frame[..4].copy_from_slice(&REQUEST_MAGIC);
    frame[4] = PROTOCOL_VERSION;
    frame[5] = EXECUTE_COMMAND;
    frame[6] = SEQUENCE_ID;
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
enum Shaft {
    Left,
    Right,
}

const PULSES: [Shaft; 2] = [Shaft::Left, Shaft::Right];
const _: () = assert!(PULSES.len() == 2);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Phase {
    WaitingForTrigger,
    InitialZeroDwell {
        nonce: NonZeroU32,
        deadline_ms: u32,
    },
    Pulse {
        nonce: NonZeroU32,
        index: u8,
        deadline_ms: u32,
    },
    InterPulseZeroDwell {
        nonce: NonZeroU32,
        next_index: u8,
        deadline_ms: u32,
    },
    LockedComplete {
        nonce: NonZeroU32,
    },
    LockedFault,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MachineAction {
    None,
    Accepted {
        nonce: NonZeroU32,
    },
    StartPulse {
        nonce: NonZeroU32,
        index: u8,
        shaft: Shaft,
        deadline_ms: u32,
    },
    PulseStopped {
        nonce: NonZeroU32,
        index: u8,
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
            deadline_ms: now_ms.wrapping_add(INITIAL_ZERO_DWELL_MS),
        };
        MachineAction::Accepted {
            nonce: trigger.nonce,
        }
    }

    fn poll_zero_phase(&mut self, now_ms: u32) -> MachineAction {
        let (nonce, index) = match self.phase {
            Phase::InitialZeroDwell { nonce, deadline_ms }
                if deadline_reached(now_ms, deadline_ms) =>
            {
                (nonce, 0)
            }
            Phase::InterPulseZeroDwell {
                nonce,
                next_index,
                deadline_ms,
            } if deadline_reached(now_ms, deadline_ms) => (nonce, next_index),
            _ => return MachineAction::None,
        };
        let Some(&shaft) = PULSES.get(usize::from(index)) else {
            self.phase = Phase::LockedFault;
            return MachineAction::None;
        };
        let deadline_ms = now_ms.wrapping_add(PULSE_DURATION_MS);
        self.phase = Phase::Pulse {
            nonce,
            index,
            deadline_ms,
        };
        MachineAction::StartPulse {
            nonce,
            index,
            shaft,
            deadline_ms,
        }
    }

    fn pulse_expired(&mut self, now_ms: u32) -> MachineAction {
        let Phase::Pulse {
            nonce,
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
        let next_index = index.saturating_add(1);
        let complete = usize::from(next_index) == PULSES.len();
        if complete {
            self.phase = Phase::LockedComplete { nonce };
        } else {
            self.phase = Phase::InterPulseZeroDwell {
                nonce,
                next_index,
                deadline_ms: now_ms.wrapping_add(INTER_PULSE_ZERO_DWELL_MS),
            };
        }
        MachineAction::PulseStopped {
            nonce,
            index,
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
enum PulseIndex {
    Left,
    Right,
}

#[cfg(any(not(target_arch = "arm"), test))]
impl PulseIndex {
    const fn wire(self) -> u8 {
        match self {
            Self::Left => 0,
            Self::Right => 1,
        }
    }
}

#[cfg(any(not(target_arch = "arm"), test))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CommissionResponse {
    Ready {
        uptime_ms: u32,
    },
    Accepted {
        nonce: NonZeroU32,
        uptime_ms: u32,
    },
    PulseCompleted {
        nonce: NonZeroU32,
        index: PulseIndex,
        uptime_ms: u32,
    },
    Complete {
        nonce: NonZeroU32,
        uptime_ms: u32,
    },
    Rejected {
        error_code: u8,
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
    let expected_checksum = u32::from_le_bytes([frame[20], frame[21], frame[22], frame[23]]);
    if crc32(&frame[..20]) != expected_checksum {
        return Err(ResponseParseError::Checksum);
    }
    let build_id = u32::from_le_bytes([frame[16], frame[17], frame[18], frame[19]]);
    if build_id != COMMISSIONING_BUILD_ID {
        return Err(ResponseParseError::BuildId);
    }
    let pulse_index = frame[6];
    let detail = frame[7];
    let nonce = u32::from_le_bytes([frame[8], frame[9], frame[10], frame[11]]);
    let uptime_ms = u32::from_le_bytes([frame[12], frame[13], frame[14], frame[15]]);
    match frame[5] {
        value if value == ResponseEvent::Ready as u8 => {
            if pulse_index != u8::MAX || detail != PULSE_PWM_PERCENT || nonce != 0 {
                return Err(ResponseParseError::Fields);
            }
            Ok(CommissionResponse::Ready { uptime_ms })
        }
        value if value == ResponseEvent::Accepted as u8 => {
            if pulse_index != u8::MAX || usize::from(detail) != PULSES.len() {
                return Err(ResponseParseError::Fields);
            }
            let nonce = NonZeroU32::new(nonce).ok_or(ResponseParseError::Fields)?;
            Ok(CommissionResponse::Accepted { nonce, uptime_ms })
        }
        value if value == ResponseEvent::PulseCompleted as u8 => {
            if detail != PULSE_PWM_PERCENT {
                return Err(ResponseParseError::Fields);
            }
            let index = match pulse_index {
                0 => PulseIndex::Left,
                1 => PulseIndex::Right,
                _ => return Err(ResponseParseError::Fields),
            };
            let nonce = NonZeroU32::new(nonce).ok_or(ResponseParseError::Fields)?;
            Ok(CommissionResponse::PulseCompleted {
                nonce,
                index,
                uptime_ms,
            })
        }
        value if value == ResponseEvent::Complete as u8 => {
            if pulse_index != u8::MAX || detail != 0 {
                return Err(ResponseParseError::Fields);
            }
            let nonce = NonZeroU32::new(nonce).ok_or(ResponseParseError::Fields)?;
            Ok(CommissionResponse::Complete { nonce, uptime_ms })
        }
        value if value == ResponseEvent::Rejected as u8 => {
            if pulse_index != u8::MAX
                || nonce != 0
                || !(TriggerError::Magic.code()..=TriggerError::Checksum.code()).contains(&detail)
            {
                return Err(ResponseParseError::Fields);
            }
            Ok(CommissionResponse::Rejected {
                error_code: detail,
                uptime_ms,
            })
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

fn encode_response(
    event: ResponseEvent,
    pulse_index: Option<u8>,
    detail: u8,
    nonce: u32,
    uptime_ms: u32,
) -> [u8; RESPONSE_BYTES] {
    let mut frame = [0_u8; RESPONSE_BYTES];
    frame[..4].copy_from_slice(&RESPONSE_MAGIC);
    frame[4] = PROTOCOL_VERSION;
    frame[5] = event as u8;
    frame[6] = pulse_index.unwrap_or(u8::MAX);
    frame[7] = detail;
    frame[8..12].copy_from_slice(&nonce.to_le_bytes());
    frame[12..16].copy_from_slice(&uptime_ms.to_le_bytes());
    frame[16..20].copy_from_slice(&COMMISSIONING_BUILD_ID.to_le_bytes());
    let checksum = crc32(&frame[..20]);
    frame[20..24].copy_from_slice(&checksum.to_le_bytes());
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
        CommissionMachine, DecodeEvent, INITIAL_ZERO_DWELL_MS, MAIN_LOOP_DELAY_MS, MachineAction,
        PULSE_DURATION_MS, PULSE_PWM_PERCENT, PWM_FREQUENCY_HZ, READY_PERIOD_MS, RESPONSE_BYTES,
        ResponseEvent, SERIAL_RESPONSE_TIMEOUT_MS, Shaft, TriggerDecoder, WATCHDOG_PERIOD_MS,
        deadline_reached, encode_response,
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

        fn start_pulse(&mut self, shaft: Shaft, deadline_ms: u32) -> bool {
            self.disable_and_zero();
            let maximum_duty = match shaft {
                Shaft::Left => self.left_max_duty,
                Shaft::Right => self.right_max_duty,
            };
            let duty = scale_duty(PULSE_PWM_PERCENT, maximum_duty);
            if duty == 0 {
                return false;
            }

            cortex_m::interrupt::free(|_| {
                // Break-before-make: every channel is disabled and zeroed
                // before exactly one forward duty is preloaded.
                self.disable_and_zero();
                match shaft {
                    Shaft::Left => self.left_forward.set_duty(duty),
                    Shaft::Right => self.right_forward.set_duty(duty),
                }
                if !arm_pulse_deadline_unlocked(deadline_ms) {
                    self.disable_and_zero();
                    return false;
                }
                match shaft {
                    Shaft::Left => self.left_forward.enable(),
                    Shaft::Right => self.right_forward.enable(),
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
        watchdog.start(u32::from(WATCHDOG_PERIOD_MS).millis());

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
        let ready = encode_response(
            ResponseEvent::Ready,
            None,
            PULSE_PWM_PERCENT,
            0,
            uptime_ms(),
        );
        let mut last_ready_ms = uptime_ms();
        if !send_response(&mut serial_tx, &ready) {
            enter_fault(&mut machine, &mut motor);
        }

        loop {
            let now_ms = uptime_ms();

            if PULSE_EXPIRED.swap(false, Ordering::AcqRel) {
                motor.disable_and_zero();
                disarm_pulse_deadline();
                match machine.pulse_expired(now_ms) {
                    MachineAction::PulseStopped {
                        nonce,
                        index,
                        complete,
                    } => {
                        let pulse_evidence = encode_response(
                            ResponseEvent::PulseCompleted,
                            Some(index),
                            PULSE_PWM_PERCENT,
                            nonce.get(),
                            now_ms,
                        );
                        if !send_response(&mut serial_tx, &pulse_evidence) {
                            enter_fault(&mut machine, &mut motor);
                        } else if complete {
                            let complete_evidence = encode_response(
                                ResponseEvent::Complete,
                                None,
                                0,
                                nonce.get(),
                                uptime_ms(),
                            );
                            if !send_response(&mut serial_tx, &complete_evidence) {
                                enter_fault(&mut machine, &mut motor);
                            } else {
                                led.set_high();
                            }
                        }
                    }
                    _ => enter_fault(&mut machine, &mut motor),
                }
            }

            if let Some(deadline_ms) = machine.pulse_deadline() {
                if deadline_reached(now_ms, deadline_ms) && !PULSE_EXPIRED.load(Ordering::Acquire) {
                    // The main-loop backstop observed a missed compare ISR.
                    // Stop rather than extending a pulse or attempting again.
                    enter_fault(&mut machine, &mut motor);
                }
            } else if !machine.outputs_must_be_safe() || !motor.is_safe() {
                enter_fault(&mut machine, &mut motor);
            }

            if matches!(machine.phase, super::Phase::WaitingForTrigger) {
                if deadline_reached(now_ms, last_ready_ms.wrapping_add(READY_PERIOD_MS)) {
                    let ready =
                        encode_response(ResponseEvent::Ready, None, PULSE_PWM_PERCENT, 0, now_ms);
                    if send_response(&mut serial_tx, &ready) {
                        last_ready_ms = now_ms;
                    } else {
                        enter_fault(&mut machine, &mut motor);
                    }
                }
                for _ in 0..RX_DEQUEUE_BUDGET_BYTES {
                    let byte = match dequeue_rx_event() {
                        RxDequeue::Byte(byte) => byte,
                        RxDequeue::Empty => break,
                        RxDequeue::Invalidated => {
                            enter_fault(&mut machine, &mut motor);
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
                                MachineAction::Accepted { nonce } => {
                                    let accepted = encode_response(
                                        ResponseEvent::Accepted,
                                        None,
                                        super::PULSES.len() as u8,
                                        nonce.get(),
                                        admitted_at_ms,
                                    );
                                    if !send_response(&mut serial_tx, &accepted) {
                                        enter_fault(&mut machine, &mut motor);
                                    }
                                }
                                _ => enter_fault(&mut machine, &mut motor),
                            }
                        }
                        Err(error) => {
                            enter_fault(&mut machine, &mut motor);
                            let rejected = encode_response(
                                ResponseEvent::Rejected,
                                None,
                                error.code(),
                                0,
                                uptime_ms(),
                            );
                            let _reported = send_response(&mut serial_tx, &rejected);
                        }
                    }
                    break;
                }
            } else if !matches!(dequeue_rx_event(), RxDequeue::Empty) {
                // Any bytes, overrun, or queue overflow after the one accepted
                // trigger aborts the sequence rather than changing/retrying it.
                enter_fault(&mut machine, &mut motor);
            }

            match machine.poll_zero_phase(now_ms) {
                MachineAction::StartPulse {
                    shaft, deadline_ms, ..
                } => {
                    if !motor.start_pulse(shaft, deadline_ms) {
                        enter_fault(&mut machine, &mut motor);
                    }
                }
                MachineAction::None => {}
                _ => enter_fault(&mut machine, &mut motor),
            }

            let outputs_safely_zero = motor.is_safe()
                && !PULSE_ARMED.load(Ordering::Acquire)
                && !PULSE_EXPIRED.load(Ordering::Acquire);
            // Never feed the independent watchdog during a pulse. TIM5 is the
            // normal 250 ms cutoff; if that timer/ISR or the main loop fails,
            // IWDG remains an independent (nominal 500 ms) reset backstop.
            if outputs_safely_zero {
                watchdog.feed();
            }
            delay.delay_ms(MAIN_LOOP_DELAY_MS);
        }
    }

    fn enter_fault(machine: &mut CommissionMachine, motor: &mut HardwareMotor) {
        motor.disable_and_zero();
        disarm_pulse_deadline();
        machine.fault();
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

    const _: () = assert!(INITIAL_ZERO_DWELL_MS >= PULSE_DURATION_MS);
}

#[cfg(not(target_arch = "arm"))]
mod host {
    use std::{fmt, str::FromStr, time::Duration};

    use clap::{ArgAction, Parser};
    use serde_json::json;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio_serial::{DataBits, FlowControl, Parity, SerialPort, SerialPortBuilderExt, StopBits};

    use super::{
        COMMISSIONING_BUILD_ID, CommissionResponse, INITIAL_ZERO_DWELL_MS,
        INTER_PULSE_ZERO_DWELL_MS, PULSE_DURATION_MS, PULSE_PWM_PERCENT, PulseIndex,
        READY_PERIOD_MS, ResponseDecoder, ResponseParseError, Trigger, encode_trigger,
    };

    const SERIAL_BAUD_BPS: u32 = 115_200;
    const SERIAL_BY_ID_PREFIX: &str = "/dev/serial/by-id/";
    const STLINK_ID_PREFIX: &str = "usb-STMicroelectronics_STM32_STLink_";
    const STLINK_UART_SUFFIX: &str = "-if02";
    const MAX_SERIAL_PATH_BYTES: usize = 512;
    const MIN_RUN_TIMEOUT_MS: u64 = 5_000;
    const MAX_RUN_TIMEOUT_MS: u64 = 15_000;
    const MAX_OBSERVED_BYTES: usize = 4_096;
    const MIN_ACCEPTED_TO_PULSE_COMPLETE_MS: u32 = INITIAL_ZERO_DWELL_MS + PULSE_DURATION_MS;
    const MIN_PULSE_TO_PULSE_COMPLETE_MS: u32 = INTER_PULSE_ZERO_DWELL_MS + PULSE_DURATION_MS;
    const MAX_SEQUENCE_SEGMENT_MS: u32 = 1_500;
    const MAX_FINAL_EVIDENCE_DELAY_MS: u32 = 100;
    const POST_READY_IO_MARGIN_MS: u32 = 250;
    const REQUIRED_POST_READY_MS: u32 =
        (2 * MAX_SEQUENCE_SEGMENT_MS) + MAX_FINAL_EVIDENCE_DELAY_MS + POST_READY_IO_MARGIN_MS;
    const _: () = assert!(MIN_RUN_TIMEOUT_MS >= (READY_PERIOD_MS + REQUIRED_POST_READY_MS) as u64);

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
        #[arg(long, action = ArgAction::SetTrue, required = true)]
        execute_wheels_off_sequence: bool,

        /// One exclusive end-to-end deadline, including Ready observation.
        #[arg(long, default_value_t = 5_000, value_parser = parse_timeout_ms)]
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
        Complete,
    }

    #[derive(Debug)]
    pub(super) enum HostError {
        ExplicitExecutionFlagMissing,
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
                Self::ExplicitExecutionFlagMissing => formatter.write_str(
                    "--execute-wheels-off-sequence is required before any trigger is sent",
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
    struct SequenceEvidence {
        accepted_uptime_ms: u32,
        left_uptime_ms: u32,
        right_uptime_ms: u32,
        complete_uptime_ms: u32,
        accepted_to_left_ms: u32,
        left_to_right_ms: u32,
        right_to_complete_ms: u32,
    }

    pub(super) async fn run() -> Result<(), HostError> {
        let cli = Cli::parse();
        if !cli.execute_wheels_off_sequence {
            return Err(HostError::ExplicitExecutionFlagMissing);
        }
        let (mut port, serial_evidence) = open_exact_serial(&cli.serial_device)?;
        let host_started_at = tokio::time::Instant::now();
        let deadline = host_started_at + Duration::from_millis(cli.timeout_ms);
        let mut responses = ResponseStream::new(deadline, cli.timeout_ms);

        let ready_uptime_ms = match responses.next(&mut port, SequenceStage::Ready).await? {
            CommissionResponse::Ready { uptime_ms } => uptime_ms,
            CommissionResponse::Rejected { error_code, .. } => {
                return Err(HostError::DeviceRejected { error_code });
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
        if remaining_ms < u64::from(REQUIRED_POST_READY_MS) {
            return Err(HostError::InsufficientDeadlineBeforeTrigger {
                remaining_ms,
                required_ms: REQUIRED_POST_READY_MS,
            });
        }

        let nonce = os_nonce()?;
        let request = encode_trigger(Trigger { nonce });
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

        let sequence = observe_sequence(&mut port, &mut responses, nonce)
            .await
            .map_err(post_trigger)?;
        let host_elapsed_ms = u64::try_from(host_started_at.elapsed().as_millis())
            .expect("bounded CLI deadline fits u64 milliseconds");

        let evidence = json!({
            "schema_version": 1,
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
            "compiled_sequence": {
                "pulse_pwm_percent": PULSE_PWM_PERCENT,
                "pulse_duration_ms": PULSE_DURATION_MS,
                "initial_zero_dwell_ms": INITIAL_ZERO_DWELL_MS,
                "inter_pulse_zero_dwell_ms": INTER_PULSE_ZERO_DWELL_MS,
                "shaft_order": ["left_forward", "right_forward"],
            },
            "controller_uptime_ms": {
                "ready": ready_uptime_ms,
                "accepted": sequence.accepted_uptime_ms,
                "left_pulse_complete": sequence.left_uptime_ms,
                "right_pulse_complete": sequence.right_uptime_ms,
                "sequence_complete": sequence.complete_uptime_ms,
            },
            "verified_controller_deltas_ms": {
                "accepted_to_left_complete": sequence.accepted_to_left_ms,
                "left_complete_to_right_complete": sequence.left_to_right_ms,
                "right_complete_to_sequence_complete": sequence.right_to_complete_ms,
            },
            "host_elapsed_ms": host_elapsed_ms,
            "observed_serial_bytes": responses.observed_bytes,
            "sequence_complete": true,
            "evidence_boundary": "software and STM32 timer evidence only; wheels-off state and physical shaft motion require separate operator observation, and this is not an unconditional arbitrary-fault pulse bound, velocity calibration, KRP2 motion authority, closed-loop control, or MPC validation",
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
    ) -> Result<SequenceEvidence, HostError> {
        let accepted_uptime_ms = loop {
            match responses.next(port, SequenceStage::Accepted).await? {
                // A periodic Ready can already be in the UART/USB buffers when
                // the one and only request is written. It is allowed only
                // before Accepted.
                CommissionResponse::Ready { .. } => {}
                CommissionResponse::Accepted {
                    nonce: observed,
                    uptime_ms,
                } => {
                    require_nonce(SequenceStage::Accepted, nonce, observed)?;
                    break uptime_ms;
                }
                CommissionResponse::Rejected { error_code, .. } => {
                    return Err(HostError::DeviceRejected { error_code });
                }
                response => {
                    return Err(HostError::UnexpectedResponse {
                        stage: SequenceStage::Accepted,
                        response,
                    });
                }
            }
        };

        let left_uptime_ms = responses
            .expect_pulse(
                port,
                nonce,
                PulseIndex::Left,
                SequenceStage::LeftPulseComplete,
            )
            .await?;
        let right_uptime_ms = responses
            .expect_pulse(
                port,
                nonce,
                PulseIndex::Right,
                SequenceStage::RightPulseComplete,
            )
            .await?;
        let complete_uptime_ms = match responses.next(port, SequenceStage::Complete).await? {
            CommissionResponse::Complete {
                nonce: observed,
                uptime_ms,
            } => {
                require_nonce(SequenceStage::Complete, nonce, observed)?;
                uptime_ms
            }
            CommissionResponse::Rejected { error_code, .. } => {
                return Err(HostError::DeviceRejected { error_code });
            }
            response => {
                return Err(HostError::UnexpectedResponse {
                    stage: SequenceStage::Complete,
                    response,
                });
            }
        };

        Ok(SequenceEvidence {
            accepted_uptime_ms,
            left_uptime_ms,
            right_uptime_ms,
            complete_uptime_ms,
            accepted_to_left_ms: bounded_controller_delta(
                "accepted_to_left_complete",
                accepted_uptime_ms,
                left_uptime_ms,
                MIN_ACCEPTED_TO_PULSE_COMPLETE_MS,
                MAX_SEQUENCE_SEGMENT_MS,
            )?,
            left_to_right_ms: bounded_controller_delta(
                "left_complete_to_right_complete",
                left_uptime_ms,
                right_uptime_ms,
                MIN_PULSE_TO_PULSE_COMPLETE_MS,
                MAX_SEQUENCE_SEGMENT_MS,
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
            expected_index: PulseIndex,
            stage: SequenceStage,
        ) -> Result<u32, HostError> {
            match self.next(port, stage).await? {
                CommissionResponse::PulseCompleted {
                    nonce,
                    index,
                    uptime_ms,
                } if index == expected_index => {
                    require_nonce(stage, expected_nonce, nonce)?;
                    Ok(uptime_ms)
                }
                CommissionResponse::Rejected { error_code, .. } => {
                    Err(HostError::DeviceRejected { error_code })
                }
                response => Err(HostError::UnexpectedResponse { stage, response }),
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
            assert!(parse_timeout_ms("15001").is_err());
            assert!(MIN_RUN_TIMEOUT_MS >= u64::from(READY_PERIOD_MS + REQUIRED_POST_READY_MS));
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
async fn main() -> Result<(), host::HostError> {
    host::run().await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(nonce: u32) -> [u8; REQUEST_BYTES] {
        let mut frame = [0_u8; REQUEST_BYTES];
        frame[..4].copy_from_slice(&REQUEST_MAGIC);
        frame[4] = PROTOCOL_VERSION;
        frame[5] = EXECUTE_COMMAND;
        frame[6] = SEQUENCE_ID;
        frame[7] = PULSE_PWM_PERCENT;
        frame[8..12].copy_from_slice(&nonce.to_le_bytes());
        let checksum = crc32(&frame[..12]);
        frame[12..16].copy_from_slice(&checksum.to_le_bytes());
        frame
    }

    #[test]
    fn crc_matches_standard_check_vector() {
        assert_eq!(crc32(b"123456789"), 0xcbf4_3926);
    }

    #[test]
    fn valid_request_parses_once_into_nonzero_nonce() {
        let nonce = NonZeroU32::new(0x1234_5678).expect("nonzero");
        let encoded = encode_trigger(Trigger { nonce });
        assert_eq!(encoded, request(nonce.get()));
        let parsed = parse_trigger(&encoded).expect("valid request");
        assert_eq!(parsed.nonce.get(), 0x1234_5678);
    }

    #[test]
    fn parser_rejects_every_weakly_typed_contract_mismatch() {
        let mut cases = [
            (0, REQUEST_MAGIC[0].wrapping_add(1), TriggerError::Magic),
            (4, PROTOCOL_VERSION.wrapping_add(1), TriggerError::Version),
            (5, EXECUTE_COMMAND.wrapping_add(1), TriggerError::Command),
            (6, SEQUENCE_ID.wrapping_add(1), TriggerError::Sequence),
            (7, PULSE_PWM_PERCENT - 1, TriggerError::ExpectedPwm),
        ];
        for (index, value, expected) in &mut cases {
            let mut frame = request(1);
            frame[*index] = *value;
            let checksum = crc32(&frame[..12]);
            frame[12..16].copy_from_slice(&checksum.to_le_bytes());
            assert_eq!(parse_trigger(&frame), Err(*expected));
        }

        assert_eq!(parse_trigger(&request(0)), Err(TriggerError::Nonce));

        let mut corrupt = request(1);
        corrupt[11] ^= 0x80;
        assert_eq!(parse_trigger(&corrupt), Err(TriggerError::Checksum));
    }

    #[test]
    fn every_single_bit_wire_corruption_is_rejected() {
        let valid = request(0x1234_5678);
        for bit in 0..(REQUEST_BYTES * 8) {
            let mut corrupt = valid;
            corrupt[bit / 8] ^= 1 << (bit % 8);
            assert!(
                parse_trigger(&corrupt).is_err(),
                "bit {bit} unexpectedly retained admission"
            );
        }
    }

    #[test]
    fn stream_decoder_ignores_noise_and_emits_exactly_one_frame() {
        let mut decoder = TriggerDecoder::new();
        for byte in [0, b'K', 0, b'K', b'M', b'K', b'M', b'C', b'1'] {
            assert_eq!(decoder.push(byte), None);
        }
        let frame = request(7);
        let mut event = None;
        for &byte in &frame[4..] {
            let next = decoder.push(byte);
            assert!(event.is_none() || next.is_none());
            event = event.or(next);
        }
        assert_eq!(
            event,
            Some(DecodeEvent::Trigger(Ok(Trigger {
                nonce: NonZeroU32::new(7).expect("nonzero"),
            })))
        );
    }

    #[test]
    fn state_machine_runs_only_left_then_right_with_bounded_timing() {
        let nonce = NonZeroU32::new(9).expect("nonzero");
        let mut machine = CommissionMachine::new();
        assert_eq!(
            machine.accept(Trigger { nonce }, 100),
            MachineAction::Accepted { nonce }
        );
        assert!(machine.outputs_must_be_safe());
        assert_eq!(machine.poll_zero_phase(599), MachineAction::None);
        assert_eq!(
            machine.poll_zero_phase(600),
            MachineAction::StartPulse {
                nonce,
                index: 0,
                shaft: Shaft::Left,
                deadline_ms: 850,
            }
        );
        assert!(!machine.outputs_must_be_safe());
        assert_eq!(machine.pulse_expired(849), MachineAction::None);
        assert!(machine.outputs_must_be_safe());
        assert!(matches!(machine.phase, Phase::LockedFault));

        let mut machine = CommissionMachine::new();
        machine.accept(Trigger { nonce }, 100);
        machine.poll_zero_phase(600);
        assert_eq!(
            machine.pulse_expired(850),
            MachineAction::PulseStopped {
                nonce,
                index: 0,
                complete: false,
            }
        );
        assert_eq!(machine.poll_zero_phase(1_349), MachineAction::None);
        assert_eq!(
            machine.poll_zero_phase(1_350),
            MachineAction::StartPulse {
                nonce,
                index: 1,
                shaft: Shaft::Right,
                deadline_ms: 1_600,
            }
        );
        assert_eq!(
            machine.pulse_expired(1_600),
            MachineAction::PulseStopped {
                nonce,
                index: 1,
                complete: true,
            }
        );
        assert!(machine.outputs_must_be_safe());
        assert!(matches!(machine.phase, Phase::LockedComplete { nonce: n } if n == nonce));
        assert_eq!(machine.poll_zero_phase(u32::MAX), MachineAction::None);
    }

    #[test]
    fn one_shot_latch_refuses_a_second_trigger_even_after_completion() {
        let nonce = NonZeroU32::new(1).expect("nonzero");
        let mut machine = CommissionMachine::new();
        machine.accept(Trigger { nonce }, 0);
        machine.poll_zero_phase(INITIAL_ZERO_DWELL_MS);
        machine.pulse_expired(INITIAL_ZERO_DWELL_MS + PULSE_DURATION_MS);
        machine
            .poll_zero_phase(INITIAL_ZERO_DWELL_MS + PULSE_DURATION_MS + INTER_PULSE_ZERO_DWELL_MS);
        machine.pulse_expired(
            INITIAL_ZERO_DWELL_MS
                + PULSE_DURATION_MS
                + INTER_PULSE_ZERO_DWELL_MS
                + PULSE_DURATION_MS,
        );
        assert_eq!(
            machine.accept(Trigger { nonce }, 10_000),
            MachineAction::None
        );
        assert!(matches!(machine.phase, Phase::LockedFault));
    }

    #[test]
    fn deadlines_are_correct_across_u32_wrap() {
        assert!(!deadline_reached(u32::MAX - 1, 1));
        assert!(deadline_reached(1, 1));
        assert!(deadline_reached(2, 1));

        let nonce = NonZeroU32::new(1).expect("nonzero");
        let mut machine = CommissionMachine::new();
        machine.accept(Trigger { nonce }, u32::MAX - 100);
        assert_eq!(machine.poll_zero_phase(398), MachineAction::None);
        assert!(matches!(
            machine.poll_zero_phase(399),
            MachineAction::StartPulse {
                shaft: Shaft::Left,
                ..
            }
        ));
    }

    #[test]
    fn response_is_self_identifying_and_checksummed() {
        let frame = encode_response(ResponseEvent::Complete, None, 0, 17, 42);
        assert_eq!(&frame[..4], b"KMR1");
        assert_eq!(frame[4], PROTOCOL_VERSION);
        assert_eq!(frame[5], ResponseEvent::Complete as u8);
        assert_eq!(frame[6], u8::MAX);
        assert_eq!(u32::from_le_bytes(frame[8..12].try_into().unwrap()), 17);
        assert_eq!(
            u32::from_le_bytes(frame[16..20].try_into().unwrap()),
            COMMISSIONING_BUILD_ID
        );
        assert_eq!(
            u32::from_le_bytes(frame[20..24].try_into().unwrap()),
            crc32(&frame[..20])
        );
        assert_eq!(
            parse_response(&frame),
            Ok(CommissionResponse::Complete {
                nonce: NonZeroU32::new(17).expect("nonzero"),
                uptime_ms: 42,
            })
        );
    }

    #[test]
    fn response_parser_makes_event_specific_invalid_states_unrepresentable() {
        let nonce = NonZeroU32::new(23).expect("nonzero");
        let cases = [
            (
                encode_response(ResponseEvent::Ready, None, PULSE_PWM_PERCENT, 0, 1),
                CommissionResponse::Ready { uptime_ms: 1 },
            ),
            (
                encode_response(ResponseEvent::Accepted, None, PULSES.len() as u8, 23, 2),
                CommissionResponse::Accepted {
                    nonce,
                    uptime_ms: 2,
                },
            ),
            (
                encode_response(
                    ResponseEvent::PulseCompleted,
                    Some(PulseIndex::Left.wire()),
                    PULSE_PWM_PERCENT,
                    23,
                    3,
                ),
                CommissionResponse::PulseCompleted {
                    nonce,
                    index: PulseIndex::Left,
                    uptime_ms: 3,
                },
            ),
            (
                encode_response(
                    ResponseEvent::PulseCompleted,
                    Some(PulseIndex::Right.wire()),
                    PULSE_PWM_PERCENT,
                    23,
                    4,
                ),
                CommissionResponse::PulseCompleted {
                    nonce,
                    index: PulseIndex::Right,
                    uptime_ms: 4,
                },
            ),
            (
                encode_response(
                    ResponseEvent::Rejected,
                    None,
                    TriggerError::Checksum.code(),
                    0,
                    5,
                ),
                CommissionResponse::Rejected {
                    error_code: TriggerError::Checksum.code(),
                    uptime_ms: 5,
                },
            ),
        ];
        for (frame, expected) in cases {
            assert_eq!(parse_response(&frame), Ok(expected));
        }

        let mut invalid_ready =
            encode_response(ResponseEvent::Ready, None, PULSE_PWM_PERCENT, 0, 1);
        invalid_ready[8..12].copy_from_slice(&1_u32.to_le_bytes());
        let checksum = crc32(&invalid_ready[..20]);
        invalid_ready[20..24].copy_from_slice(&checksum.to_le_bytes());
        assert_eq!(
            parse_response(&invalid_ready),
            Err(ResponseParseError::Fields)
        );
    }

    #[test]
    fn every_single_bit_evidence_corruption_is_rejected() {
        let valid = encode_response(ResponseEvent::Complete, None, 0, 17, 42);
        for bit in 0..(RESPONSE_BYTES * 8) {
            let mut corrupt = valid;
            corrupt[bit / 8] ^= 1 << (bit % 8);
            assert!(
                parse_response(&corrupt).is_err(),
                "response bit {bit} unexpectedly retained admission"
            );
        }
    }
}
