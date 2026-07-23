#![no_main]
#![no_std]

//! STM32F446 host-control firmware.
//!
//! The compiled default actuator profile deliberately grants no motion
//! authority. A separate feature builds an explicitly provisional,
//! operator-supervised four-PWM candidate or the distinctly identified
//! attended wheel-on commissioning profile. Neither claims production
//! external interlocks or verified physical stop behavior.

#[cfg(all(feature = "external-boot-identity", feature = "flash-boot-journal"))]
compile_error!(
    "select exactly one boot-identity source: external-boot-identity or flash-boot-journal"
);
#[cfg(all(
    feature = "operator-supervised-four-pwm-candidate",
    not(feature = "flash-boot-journal")
))]
compile_error!(
    "operator-supervised-four-pwm-candidate requires the flash-boot-journal identity source"
);
#[cfg(all(
    feature = "operator-supervised-four-pwm-candidate",
    feature = "external-boot-identity"
))]
compile_error!(
    "operator-supervised-four-pwm-candidate is incompatible with external-boot-identity"
);
#[cfg(all(
    feature = "attended-wheel-on-commissioning",
    not(feature = "flash-boot-journal")
))]
compile_error!("attended-wheel-on-commissioning requires the flash-boot-journal identity source");
#[cfg(all(
    feature = "attended-wheel-on-commissioning",
    feature = "external-boot-identity"
))]
compile_error!("attended-wheel-on-commissioning is incompatible with external-boot-identity");
#[cfg(all(
    feature = "attended-wheel-on-commissioning",
    feature = "operator-supervised-four-pwm-candidate"
))]
compile_error!(
    "attended-wheel-on-commissioning and operator-supervised-four-pwm-candidate are distinct images and cannot be combined"
);

use core::{
    cell::{Cell, RefCell},
    num::NonZeroU16,
    panic::PanicInfo,
    sync::atomic::{AtomicBool, Ordering},
};

use cortex_m::{interrupt::Mutex, peripheral::NVIC};
use cortex_m_rt::{ExceptionFrame, entry, exception};
#[cfg(feature = "attended-wheel-on-commissioning")]
use embedded::attended_wheel_on_commissioning::{
    ATTENDED_WHEEL_ON_COMMISSIONING_FIRMWARE_BUILD_ID, AttendedWheelOnCommissioningProfile,
};
#[cfg(feature = "flash-boot-journal")]
use embedded::boot_journal::{plan_next_boot, verify_commit};
#[cfg(not(any(
    feature = "operator-supervised-four-pwm-candidate",
    feature = "attended-wheel-on-commissioning"
)))]
use embedded::motor_inert_profile::{MOTOR_INERT_FIRMWARE_BUILD_ID, MotorInertProfile};
#[cfg(feature = "operator-supervised-four-pwm-candidate")]
use embedded::provisional_four_pwm::{
    OPERATOR_SUPERVISED_FOUR_PWM_FIRMWARE_BUILD_ID, OperatorSupervisedFourPwmProfile,
};
use embedded::{
    FIRMWARE_V2_WATCHDOG_NOMINAL_PERIOD_MS,
    controller::{
        Controller, ControllerCommand, ControllerConfig, ControllerEvent as PureControllerEvent,
        ControllerMode, ControllerStep, ControllerWatchdogStatus, DeadlineTimerSnapshot, FaultCode,
    },
    encoder_wraps_with_pending_direction_assumption, established_session_readiness,
    motor::{
        ActuatorEnvelope, DriveOutput, DurationMs, MotorDirective, MotorTiming,
        MotorTransitionPhase, ObservationalOdometryContract, PwmPair, WheelDrive,
    },
    transport_diagnostic::TransportDiagnosticGateSnapshot,
    transport_scheduler::{PriorityTxScheduler, TxAdmissionError, TxTrafficClass},
    watchdog_gate::{CompletedLoopSafety, LoopIteration, WatchdogDecision, WatchdogGate},
};
use heapless::spsc::Queue;
use robot_protocol::{
    ControllerUptimeMsWrapping, EstimatedWrappingEncoderTicks, ModuloEncoderDeltaTicks,
    v2::{
        ActuatorConfigFingerprint, AppliedResult, AppliedResultCode, ApplyPwm, BeginSession,
        CANONICAL_CONTROLLER_HELLO_PERIOD_MS, CANONICAL_ODOMETRY_REPORT_PERIOD_MS, ControlEpoch,
        ControllerBootId, ControllerCapabilities, ControllerDeadlineMsWrapping, ControllerFaults,
        ControllerHello, ControllerReady, ControllerSessionAdmission, ControllerUid,
        DeadlineRelation, ForceStop, Heartbeat, HostStopResult, MAX_UART_RECORD_BYTES,
        MAX_V2_COMMAND_LEASE_MS, MaxAbsPwmPercent, Message, NeutralOutput, ObservationalOdometry,
        OutputState, PhysicalStopSemantics, PwmFrequencyHz, ReadinessFlags, StopResultCode,
        TargetBootId, TimerPwm, TransportDiagnosticProbe, TransportDiagnosticReport, UartRecord,
        UartStreamDecoder, V2CommandLeaseMs, WatchdogNominalPeriodMs,
    },
};
#[cfg(feature = "flash-boot-journal")]
use stm32f4xx_hal::flash::{FlashExt, LockedFlash};
use stm32f4xx_hal::{
    pac::{self, interrupt},
    prelude::*,
    serial::{Event as SerialEvent, Serial, config::Config},
    timer::{Channel1, Channel2, FTimer, PwmChannel, pwm::PwmExt},
    watchdog::IndependentWatchdog,
};

const MAIN_LOOP_DELAY_MS: u32 = 1;
const DEFAULT_HEARTBEAT_PERIOD_MS: u16 = 250;
const ODOMETRY_SAMPLE_PERIOD_MS: u32 = 10;
const PWM_FREQUENCY_HZ: u16 = 20_000;
const FIRMWARE_ABI: u16 = 2;
#[cfg(not(any(
    feature = "operator-supervised-four-pwm-candidate",
    feature = "attended-wheel-on-commissioning"
)))]
const FIRMWARE_BUILD_ID: u32 = MOTOR_INERT_FIRMWARE_BUILD_ID;
#[cfg(feature = "operator-supervised-four-pwm-candidate")]
const FIRMWARE_BUILD_ID: u32 = OPERATOR_SUPERVISED_FOUR_PWM_FIRMWARE_BUILD_ID;
#[cfg(feature = "attended-wheel-on-commissioning")]
const FIRMWARE_BUILD_ID: u32 = ATTENDED_WHEEL_ON_COMMISSIONING_FIRMWARE_BUILD_ID;
const STM32F446_UID_ADDRESS: usize = 0x1fff_7a10;
#[cfg(feature = "flash-boot-journal")]
const STM32F446_FLASH_BYTES: usize = 512 * 1024;
#[cfg(feature = "flash-boot-journal")]
const BOOT_JOURNAL_FLASH_OFFSET: usize = 384 * 1024;
#[cfg(feature = "flash-boot-journal")]
const BOOT_JOURNAL_FLASH_BYTES: usize = 128 * 1024;

const RX_QUEUE_BYTES: usize = 256;
const MAX_RX_BYTES_PER_LOOP: usize = MAX_UART_RECORD_BYTES;
const MAX_RX_RECORDS_PER_LOOP: usize = 1;
const TX_STOP_QUEUE_BYTES: usize = MAX_UART_RECORD_BYTES * 2;
const TX_APPLIED_QUEUE_BYTES: usize = MAX_UART_RECORD_BYTES * 4;
const TX_BEST_EFFORT_QUEUE_BYTES: usize = MAX_UART_RECORD_BYTES * 4;

static RX_QUEUE: Mutex<RefCell<Queue<u8, RX_QUEUE_BYTES>>> = Mutex::new(RefCell::new(Queue::new()));
static RX_STREAM_INVALIDATED: AtomicBool = AtomicBool::new(false);
type FirmwareTxScheduler =
    PriorityTxScheduler<TX_STOP_QUEUE_BYTES, TX_APPLIED_QUEUE_BYTES, TX_BEST_EFFORT_QUEUE_BYTES>;
static TX_SCHEDULER: Mutex<RefCell<FirmwareTxScheduler>> =
    Mutex::new(RefCell::new(FirmwareTxScheduler::new()));
static TX_PATH_FAILED: AtomicBool = AtomicBool::new(false);
static SERIAL: Mutex<RefCell<Option<Serial<pac::USART2>>>> = Mutex::new(RefCell::new(None));

static LEASE_ARMED: AtomicBool = AtomicBool::new(false);
static LEASE_EXPIRED_IN_ISR: AtomicBool = AtomicBool::new(false);

static LEFT_ENCODER_WRAPS: Mutex<Cell<i64>> = Mutex::new(Cell::new(0));
static RIGHT_ENCODER_WRAPS: Mutex<Cell<i64>> = Mutex::new(Cell::new(0));

type LeftForward = PwmChannel<pac::TIM2, 0>;
type LeftReverse = PwmChannel<pac::TIM2, 1>;
type RightForward = PwmChannel<pac::TIM3, 0>;
type RightReverse = PwmChannel<pac::TIM3, 1>;

#[derive(Clone, Copy)]
struct CompiledActuatorProfile {
    fingerprint: ActuatorConfigFingerprint,
    controller_envelope: ActuatorEnvelope,
    capabilities: ControllerCapabilities,
    max_abs_pwm: MaxAbsPwmPercent,
    maximum_command_lease: V2CommandLeaseMs,
    neutral_output: NeutralOutput,
    physical_stop_semantics: PhysicalStopSemantics,
    per_boot_identity_is_session_unique: bool,
    observational_odometry: ObservationalOdometryContract,
}

impl CompiledActuatorProfile {
    #[cfg(not(any(
        feature = "operator-supervised-four-pwm-candidate",
        feature = "attended-wheel-on-commissioning"
    )))]
    fn load(per_boot_identity_is_session_unique: bool) -> Option<Self> {
        let profile = MotorInertProfile::try_new().ok()?;
        Some(Self {
            fingerprint: profile.fingerprint(),
            controller_envelope: profile.envelope(),
            capabilities: profile.capabilities(),
            max_abs_pwm: profile.max_abs_pwm(),
            maximum_command_lease: V2CommandLeaseMs::try_new(MAX_V2_COMMAND_LEASE_MS).ok()?,
            neutral_output: NeutralOutput::BothLow,
            physical_stop_semantics: profile.physical_stop_semantics(),
            per_boot_identity_is_session_unique,
            observational_odometry: ObservationalOdometryContract::Absent,
        })
    }

    #[cfg(feature = "operator-supervised-four-pwm-candidate")]
    fn load(per_boot_identity_is_session_unique: bool) -> Option<Self> {
        let profile = OperatorSupervisedFourPwmProfile::try_new().ok()?;
        Some(Self {
            fingerprint: profile.fingerprint(),
            controller_envelope: profile.envelope(),
            capabilities: profile.capabilities(),
            max_abs_pwm: profile.max_abs_pwm(),
            maximum_command_lease: V2CommandLeaseMs::try_new(MAX_V2_COMMAND_LEASE_MS).ok()?,
            neutral_output: NeutralOutput::BothLow,
            physical_stop_semantics: profile.physical_stop_semantics(),
            per_boot_identity_is_session_unique,
            observational_odometry: ObservationalOdometryContract::Absent,
        })
    }

    #[cfg(feature = "attended-wheel-on-commissioning")]
    fn load(per_boot_identity_is_session_unique: bool) -> Option<Self> {
        let profile = AttendedWheelOnCommissioningProfile::try_new().ok()?;
        Some(Self {
            fingerprint: profile.fingerprint(),
            controller_envelope: profile.envelope(),
            capabilities: profile.capabilities(),
            max_abs_pwm: profile.max_abs_pwm(),
            maximum_command_lease: V2CommandLeaseMs::try_new(MAX_V2_COMMAND_LEASE_MS).ok()?,
            neutral_output: NeutralOutput::BothLow,
            physical_stop_semantics: profile.physical_stop_semantics(),
            per_boot_identity_is_session_unique,
            observational_odometry: ObservationalOdometryContract::Absent,
        })
    }

    fn grants_motion_authority(self) -> bool {
        if !self.per_boot_identity_is_session_unique {
            return false;
        }
        match self
            .capabilities
            .classify_session_admission(self.max_abs_pwm, self.physical_stop_semantics)
        {
            Ok(ControllerSessionAdmission::OperatorSupervisedFourPwmCandidate) => self
                .controller_envelope
                .is_operator_supervised_four_pwm_candidate(),
            Ok(ControllerSessionAdmission::AttendedWheelOnCommissioning) => self
                .controller_envelope
                .is_attended_wheel_on_commissioning(),
            Ok(ControllerSessionAdmission::ProductionExternalInterlocks) => {
                matches!(self.controller_envelope, ActuatorEnvelope::Validated { .. })
            }
            Ok(ControllerSessionAdmission::MotionDisabled) | Err(_) => false,
        }
    }
}

struct HardwareMotor {
    left_forward: LeftForward,
    left_reverse: LeftReverse,
    right_forward: RightForward,
    right_reverse: RightReverse,
    left_max_duty: NonZeroU16,
    right_max_duty: NonZeroU16,
    timer_pwm: PwmPair,
    output_state: OutputState,
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
            timer_pwm: PwmPair::STOP,
            output_state: OutputState::Disabled,
        };
        motor.disable_and_zero();
        Some(motor)
    }

    fn apply(&mut self, directive: MotorDirective, target: PwmPair) -> bool {
        match directive {
            MotorDirective::Hold => true,
            MotorDirective::DisableAndZero => {
                self.disable_and_zero();
                true
            }
            MotorDirective::PreloadWhileDisabled(output) => {
                if target.is_stop() || DriveOutput::from_pwm(target) != output {
                    self.disable_and_zero();
                    return false;
                }
                self.disable_all();
                self.zero_all_duties();
                if !self.write_output_duties(output) {
                    self.disable_and_zero();
                    return false;
                }
                self.timer_pwm = target;
                self.output_state = OutputState::Disabled;
                true
            }
            MotorDirective::UpdateEnabled(output) => {
                if target.is_stop() || DriveOutput::from_pwm(target) != output {
                    self.disable_and_zero();
                    return false;
                }
                let left_updated = Self::update_enabled_wheel(
                    &mut self.left_forward,
                    &mut self.left_reverse,
                    output.left(),
                    self.left_max_duty,
                );
                let right_updated = Self::update_enabled_wheel(
                    &mut self.right_forward,
                    &mut self.right_reverse,
                    output.right(),
                    self.right_max_duty,
                );
                if !left_updated || !right_updated {
                    self.disable_and_zero();
                    return false;
                }
                self.timer_pwm = target;
                self.output_state = OutputState::NonzeroPwm;
                true
            }
            MotorDirective::EnablePreloaded(output) => {
                if target.is_stop()
                    || DriveOutput::from_pwm(target) != output
                    || self.timer_pwm != target
                {
                    self.disable_and_zero();
                    return false;
                }
                self.disable_all();
                self.enable_output(output);
                self.output_state = OutputState::NonzeroPwm;
                true
            }
        }
    }

    fn disable_and_zero(&mut self) {
        self.disable_all();
        self.zero_all_duties();
        self.timer_pwm = PwmPair::STOP;
        self.output_state = OutputState::Disabled;
    }

    fn disable_all(&mut self) {
        self.left_forward.disable();
        self.left_reverse.disable();
        self.right_forward.disable();
        self.right_reverse.disable();
    }

    fn zero_all_duties(&mut self) {
        self.left_forward.set_duty(0);
        self.left_reverse.set_duty(0);
        self.right_forward.set_duty(0);
        self.right_reverse.set_duty(0);
    }

    fn write_output_duties(&mut self, output: DriveOutput) -> bool {
        let left_written = Self::write_wheel_duties(
            &mut self.left_forward,
            &mut self.left_reverse,
            output.left(),
            self.left_max_duty,
        );
        let right_written = Self::write_wheel_duties(
            &mut self.right_forward,
            &mut self.right_reverse,
            output.right(),
            self.right_max_duty,
        );
        left_written && right_written
    }

    fn write_wheel_duties(
        forward: &mut impl PwmChannelOps,
        reverse: &mut impl PwmChannelOps,
        drive: WheelDrive,
        maximum_duty: NonZeroU16,
    ) -> bool {
        forward.set_duty(0);
        reverse.set_duty(0);
        match drive {
            WheelDrive::Disabled => true,
            WheelDrive::Forward(magnitude) => {
                let Some(duty) = scale_duty(magnitude.get(), maximum_duty) else {
                    return false;
                };
                forward.set_duty(duty);
                true
            }
            WheelDrive::Reverse(magnitude) => {
                let Some(duty) = scale_duty(magnitude.get(), maximum_duty) else {
                    return false;
                };
                reverse.set_duty(duty);
                true
            }
        }
    }

    fn update_enabled_wheel(
        forward: &mut impl PwmChannelOps,
        reverse: &mut impl PwmChannelOps,
        drive: WheelDrive,
        maximum_duty: NonZeroU16,
    ) -> bool {
        match drive {
            WheelDrive::Disabled => {
                forward.disable();
                reverse.disable();
                forward.set_duty(0);
                reverse.set_duty(0);
                true
            }
            WheelDrive::Forward(magnitude) => {
                let Some(duty) = scale_duty(magnitude.get(), maximum_duty) else {
                    return false;
                };
                reverse.disable();
                reverse.set_duty(0);
                forward.set_duty(duty);
                forward.enable();
                true
            }
            WheelDrive::Reverse(magnitude) => {
                let Some(duty) = scale_duty(magnitude.get(), maximum_duty) else {
                    return false;
                };
                forward.disable();
                forward.set_duty(0);
                reverse.set_duty(duty);
                reverse.enable();
                true
            }
        }
    }

    fn enable_output(&mut self, output: DriveOutput) {
        Self::enable_wheel(
            &mut self.left_forward,
            &mut self.left_reverse,
            output.left(),
        );
        Self::enable_wheel(
            &mut self.right_forward,
            &mut self.right_reverse,
            output.right(),
        );
    }

    fn enable_wheel(
        forward: &mut impl PwmChannelOps,
        reverse: &mut impl PwmChannelOps,
        drive: WheelDrive,
    ) {
        match drive {
            WheelDrive::Disabled => {}
            WheelDrive::Forward(_) => forward.enable(),
            WheelDrive::Reverse(_) => reverse.enable(),
        }
    }
}

trait PwmChannelOps {
    fn disable(&mut self);
    fn enable(&mut self);
    fn set_duty(&mut self, duty: u16);
}

impl<const CHANNEL: u8> PwmChannelOps for PwmChannel<pac::TIM2, CHANNEL> {
    fn disable(&mut self) {
        PwmChannel::disable(self);
    }

    fn enable(&mut self) {
        PwmChannel::enable(self);
    }

    fn set_duty(&mut self, duty: u16) {
        PwmChannel::set_duty(self, duty);
    }
}

impl<const CHANNEL: u8> PwmChannelOps for PwmChannel<pac::TIM3, CHANNEL> {
    fn disable(&mut self) {
        PwmChannel::disable(self);
    }

    fn enable(&mut self) {
        PwmChannel::enable(self);
    }

    fn set_duty(&mut self, duty: u16) {
        PwmChannel::set_duty(self, duty);
    }
}

struct ControlSession {
    begin: BeginSession,
    epoch: ControlEpoch,
    controller: Controller,
    pending_application: Option<ApplyPwm>,
    cached_application: Option<(ApplyPwm, AppliedResult)>,
}

#[derive(Clone, Copy)]
struct BootIdentity {
    id: ControllerBootId,
    is_session_unique: bool,
}

#[derive(Clone, Copy)]
struct FirmwareIdentity {
    controller_uid: ControllerUid,
    boot_id: ControllerBootId,
    profile: CompiledActuatorProfile,
}

struct EpochGenerator {
    next: u32,
}

impl EpochGenerator {
    fn from_boot_id(boot_id: ControllerBootId) -> Self {
        let bytes = boot_id.get().to_le_bytes();
        let seed = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        Self {
            next: if seed == 0 { 1 } else { seed },
        }
    }

    fn next(&mut self) -> Option<ControlEpoch> {
        let epoch = ControlEpoch::try_new(self.next).ok()?;
        self.next = self.next.checked_add(1)?;
        Some(epoch)
    }
}

#[entry]
fn main() -> ! {
    let Some(dp) = pac::Peripherals::take() else {
        fatal_reset();
    };

    // IWDG uses its independent LSI clock and is intentionally started before
    // clock-tree, identity, GPIO, timer, or serial initialization can fail.
    let mut watchdog = IndependentWatchdog::new(dp.IWDG);
    watchdog.start(u32::from(FIRMWARE_V2_WATCHDOG_NOMINAL_PERIOD_MS).millis());

    let Some(mut cp) = cortex_m::peripheral::Peripherals::take() else {
        fatal_reset();
    };
    let rcc = dp.RCC.constrain();
    let clocks = rcc
        .cfgr
        .sysclk(168.MHz())
        .pclk1(21.MHz())
        .pclk2(84.MHz())
        .freeze();
    let mut delay = cp.SYST.delay(&clocks);

    let Some(controller_uid) = read_controller_uid() else {
        fatal_reset();
    };
    let Some(boot_identity) = load_boot_identity(controller_uid, dp.FLASH) else {
        fatal_reset();
    };
    let boot_id = boot_identity.id;
    let mut epoch_generator = EpochGenerator::from_boot_id(boot_id);
    let Some(profile) = CompiledActuatorProfile::load(boot_identity.is_session_unique) else {
        fatal_reset();
    };
    let firmware_identity = FirmwareIdentity {
        controller_uid,
        boot_id,
        profile,
    };

    let gpioa = dp.GPIOA.split();
    let gpiob = dp.GPIOB.split();
    let mut led = gpioa.pa5.into_push_pull_output();
    led.set_low();

    // Establish deterministic low levels before handing the motor pins to the
    // timer alternate functions. PWM channels remain disabled after setup.
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
    let mut serial = match Serial::new(
        dp.USART2,
        (tx, rx),
        Config::default()
            .baudrate(115_200.bps())
            .wordlength_8()
            .parity_none(),
        &clocks,
    ) {
        Ok(serial) => serial,
        Err(_) => fatal_reset(),
    };
    serial.listen(SerialEvent::RxNotEmpty);
    cortex_m::interrupt::free(|cs| SERIAL.borrow(cs).replace(Some(serial)));

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
        fatal_reset();
    };

    let quadrature_inputs_configured = profile
        .observational_odometry
        .configures_quadrature_inputs();
    if quadrature_inputs_configured {
        let _left_encoder_pin_a = gpioa.pa8.into_alternate::<1>();
        let _left_encoder_pin_b = gpioa.pa9.into_alternate::<1>();
        let _right_encoder_pin_a = gpiob.pb6.into_alternate::<2>();
        let _right_encoder_pin_b = gpiob.pb7.into_alternate::<2>();
        configure_encoder_tim1(dp.TIM1);
        configure_encoder_tim4(dp.TIM4);
    }
    configure_deadline_timer(dp.TIM5, &clocks);
    enable_interrupts(&mut cp.NVIC, quadrature_inputs_configured);

    let mut decoder = UartStreamDecoder::new();
    let mut session: Option<ControlSession> = None;
    let mut fault_bits = ControllerFaults::NONE.bits();
    let mut watchdog_gate = WatchdogGate::new();
    let mut iteration = LoopIteration::FIRST;
    let initial_now = controller_uptime();
    let mut last_hello = initial_now;
    let mut last_heartbeat = initial_now;
    let mut last_odometry_sample = initial_now;
    let mut last_odometry_report = initial_now;
    let mut previous_left_count = 0_u16;
    let mut previous_right_count = 0_u16;
    let mut latest_odometry = None;

    if !send_hello(controller_uid, boot_id, profile) {
        TX_PATH_FAILED.store(true, Ordering::Release);
    }

    loop {
        let mut rx_stream_valid = true;
        let mut motor_state_synchronized = true;

        if LEASE_EXPIRED_IN_ISR.swap(false, Ordering::AcqRel) {
            fault_bits |= ControllerFaults::DEADLINE_EXPIRED;
            motor_state_synchronized &= stop_and_drop_session(&mut session, &mut motor);
        }

        // Bound receive work so an untrusted continuous stream cannot starve
        // lease progression or the watchdog decision. Queue invalidation and
        // dequeue are one interrupt-masked observation: no byte following a
        // dropped byte can reach the decoder first.
        let mut decoded_records = 0_usize;
        for _ in 0..MAX_RX_BYTES_PER_LOOP {
            let byte = match dequeue_rx_event() {
                RxDequeue::Empty => break,
                RxDequeue::Invalidated => {
                    rx_stream_valid = false;
                    decoder = UartStreamDecoder::new();
                    fault_bits |= ControllerFaults::SERIAL_INTEGRITY;
                    motor_state_synchronized &= stop_and_drop_session(&mut session, &mut motor);
                    break;
                }
                RxDequeue::Byte(byte) => byte,
            };
            let Some(record) = decoder.push(byte) else {
                continue;
            };
            decoded_records += 1;
            match record {
                Ok(message) => {
                    let handled = handle_message(
                        message,
                        firmware_identity,
                        &mut epoch_generator,
                        &mut session,
                        &mut motor,
                        &mut fault_bits,
                        controller_uptime(),
                    );
                    if !handled {
                        let _failure = latch_adapter_failure(&mut fault_bits);
                        motor_state_synchronized = false;
                        let _stopped = stop_and_drop_session(&mut session, &mut motor);
                        break;
                    }
                    if fault_bits != ControllerFaults::NONE.bits() {
                        motor_state_synchronized &= stop_and_drop_session(&mut session, &mut motor);
                        break;
                    }
                }
                Err(_) => {
                    rx_stream_valid = false;
                    fault_bits |= ControllerFaults::SERIAL_INTEGRITY;
                    motor_state_synchronized &= stop_and_drop_session(&mut session, &mut motor);
                    break;
                }
            }
            if decoded_records >= MAX_RX_RECORDS_PER_LOOP {
                break;
            }
        }
        if decoder.is_discarding_oversized_record() {
            rx_stream_valid = false;
            fault_bits |= ControllerFaults::SERIAL_INTEGRITY;
            motor_state_synchronized &= stop_and_drop_session(&mut session, &mut motor);
        }

        if fault_bits != ControllerFaults::NONE.bits() && session.is_some() {
            motor_state_synchronized &= stop_and_drop_session(&mut session, &mut motor);
        }

        // Command admission samples its own current time. Never tick with the
        // older loop-start observation: doing so would look like a wrapping
        // clock regression when an RX record straddles a 1 kHz timer tick.
        let progress_now = controller_uptime();
        if let Some(active) = session.as_mut() {
            let progressed = progress_controller(active, &mut motor, &mut fault_bits, progress_now);
            let controller_faulted = matches!(
                active.controller.mode(),
                ControllerMode::FaultLatched { .. }
            );
            if !progressed {
                let _failure = latch_adapter_failure(&mut fault_bits);
                motor_state_synchronized = false;
            }
            if !progressed || controller_faulted {
                motor_state_synchronized &= stop_and_drop_session(&mut session, &mut motor);
            }
        }

        let now = controller_uptime();
        if quadrature_inputs_configured
            && now.wrapping_elapsed_since(last_odometry_sample) >= ODOMETRY_SAMPLE_PERIOD_MS
        {
            last_odometry_sample = now;
            latest_odometry = Some(sample_odometry(
                now,
                &mut previous_left_count,
                &mut previous_right_count,
            ));
        }
        if now.wrapping_elapsed_since(last_odometry_report) >= CANONICAL_ODOMETRY_REPORT_PERIOD_MS {
            last_odometry_report = now;
            if let Some(odometry) = latest_odometry {
                let epoch = session.as_ref().map(|active| active.epoch);
                if !send_odometry(controller_uid, boot_id, epoch, odometry) {
                    TX_PATH_FAILED.store(true, Ordering::Release);
                }
            }
        }

        if now.wrapping_elapsed_since(last_hello) >= CANONICAL_CONTROLLER_HELLO_PERIOD_MS
            && motor.output_state.is_safe()
        {
            last_hello = now;
            if !send_hello(controller_uid, boot_id, profile) {
                TX_PATH_FAILED.store(true, Ordering::Release);
            }
        }
        let heartbeat_period = session
            .as_ref()
            .map_or(u32::from(DEFAULT_HEARTBEAT_PERIOD_MS), |active| {
                u32::from(active.begin.heartbeat_period.get())
            });
        if now.wrapping_elapsed_since(last_heartbeat) >= heartbeat_period
            && !matches!(
                session.as_ref().map(|active| active.controller.mode()),
                Some(ControllerMode::Transitioning { .. })
            )
        {
            last_heartbeat = now;
            if !send_heartbeat(
                controller_uid,
                boot_id,
                profile,
                session.as_ref(),
                &motor,
                fault_bits,
                now,
            ) {
                TX_PATH_FAILED.store(true, Ordering::Release);
            }
        }

        if TX_PATH_FAILED.load(Ordering::Acquire) {
            fault_bits |= ControllerFaults::SERIAL_INTEGRITY;
            motor_state_synchronized &= stop_and_drop_session(&mut session, &mut motor);
        }

        let watchdog_now = controller_uptime();
        let controller_watchdog = target_watchdog_status(session.as_ref(), &motor, watchdog_now);
        if controller_watchdog == ControllerWatchdogStatus::UnsafeExpiredOrAmbiguous {
            let semantic_deadline_failure = session.as_ref().is_some_and(|active| {
                active.controller.watchdog_status_at(watchdog_now)
                    == ControllerWatchdogStatus::UnsafeExpiredOrAmbiguous
            });
            if semantic_deadline_failure {
                fault_bits |= ControllerFaults::DEADLINE_EXPIRED;
            } else {
                let _failure = latch_adapter_failure(&mut fault_bits);
            }
            motor_state_synchronized = false;
            let _stopped = stop_and_drop_session(&mut session, &mut motor);
        }
        let safety = CompletedLoopSafety::new(
            controller_watchdog,
            motor_state_synchronized,
            rx_stream_valid,
            !TX_PATH_FAILED.load(Ordering::Acquire),
        );
        if let Ok(WatchdogDecision::Feed(permit)) =
            watchdog_gate.complete_iteration(iteration, safety)
        {
            let _completed_iteration = permit.consume();
            watchdog.feed();
        }
        let Ok(next_iteration) = iteration.checked_successor() else {
            fatal_reset();
        };
        iteration = next_iteration;
        delay.delay_ms(MAIN_LOOP_DELAY_MS);
    }
}

fn handle_message(
    message: Message,
    identity: FirmwareIdentity,
    epoch_generator: &mut EpochGenerator,
    session: &mut Option<ControlSession>,
    motor: &mut HardwareMotor,
    fault_bits: &mut u32,
    now: ControllerUptimeMsWrapping,
) -> bool {
    match message {
        Message::ForceStop(request) => {
            handle_force_stop(request, identity, session, motor, fault_bits, now)
        }
        Message::BeginSession(request) => handle_begin_session(
            request,
            identity,
            epoch_generator,
            session,
            motor,
            *fault_bits,
            now,
        ),
        Message::ApplyPwm(request) => {
            handle_apply_pwm(request, identity, session, motor, fault_bits, now)
        }
        Message::TransportDiagnosticProbe(request) => handle_transport_diagnostic_probe(
            request,
            identity,
            session.as_ref(),
            motor,
            *fault_bits,
            now,
        ),
        _ => {
            *fault_bits |= ControllerFaults::SERIAL_INTEGRITY;
            stop_and_drop_session(session, motor)
        }
    }
}

fn handle_transport_diagnostic_probe(
    request: TransportDiagnosticProbe,
    identity: FirmwareIdentity,
    session: Option<&ControlSession>,
    motor: &HardwareMotor,
    fault_bits: u32,
    request_received_at: ControllerUptimeMsWrapping,
) -> bool {
    let result = TransportDiagnosticGateSnapshot {
        identity_matches: request.expected_controller_uid == identity.controller_uid
            && request.expected_boot_id == identity.boot_id,
        capability_available: identity
            .profile
            .capabilities
            .supports_motor_inert_transport_diagnostics(),
        profile_grants_motion_authority: identity.profile.max_abs_pwm.grants_motion_authority(),
        session_active: session.is_some(),
        output_state: motor.output_state,
        timer_pwm: wire_pwm(motor.timer_pwm),
        faults: wire_faults(fault_bits),
    }
    .classify();
    let Some((rx_queue_depth_bytes, tx_queue_depth_bytes)) = serial_queue_depths() else {
        return false;
    };
    let response_prepared_at = controller_uptime();
    queue_message(Message::TransportDiagnosticReport(
        TransportDiagnosticReport {
            controller_uid: identity.controller_uid,
            boot_id: identity.boot_id,
            run_id: request.run_id,
            sequence: request.sequence,
            host_elapsed_ns_token: request.host_elapsed_ns_token,
            result,
            output_state: motor.output_state,
            timer_pwm: wire_pwm(motor.timer_pwm),
            faults: wire_faults(fault_bits),
            request_received_at,
            response_prepared_at,
            rx_queue_depth_bytes,
            tx_queue_depth_bytes,
        },
    ))
}

fn handle_force_stop(
    request: ForceStop,
    identity: FirmwareIdentity,
    session: &mut Option<ControlSession>,
    motor: &mut HardwareMotor,
    fault_bits: &mut u32,
    now: ControllerUptimeMsWrapping,
) -> bool {
    let target_matches = match request.target_boot_id {
        TargetBootId::Any => true,
        TargetBootId::Exact(value) => value == identity.boot_id,
    };
    let identity_matches = request.controller_uid == identity.controller_uid && target_matches;
    if !identity_matches {
        return queue_host_stop_result(HostStopResult {
            controller_uid: identity.controller_uid,
            observed_boot_id: TargetBootId::Exact(identity.boot_id),
            request_id: request.request_id,
            result: StopResultCode::IdentityMismatch,
            output_state: motor.output_state,
            controller_uptime: now,
            faults: wire_faults(*fault_bits),
        });
    }

    // `reason` is the requester's cause label, not controller-side evidence.
    // Stop immediately, but do not fabricate sticky hardware/UART faults from
    // an unauthenticated assertion supplied on the wire.
    let synchronized = stop_and_drop_session(session, motor);
    let stopped_at = controller_uptime();
    let result = if synchronized {
        StopResultCode::ControllerConfirmed
    } else {
        StopResultCode::ControllerFaulted
    };
    let queued = queue_host_stop_result(HostStopResult {
        controller_uid: identity.controller_uid,
        observed_boot_id: TargetBootId::Exact(identity.boot_id),
        request_id: request.request_id,
        result,
        output_state: motor.output_state,
        controller_uptime: stopped_at,
        faults: wire_faults(*fault_bits),
    });
    synchronized && queued
}

fn handle_begin_session(
    request: BeginSession,
    identity: FirmwareIdentity,
    epoch_generator: &mut EpochGenerator,
    session: &mut Option<ControlSession>,
    motor: &mut HardwareMotor,
    fault_bits: u32,
    now: ControllerUptimeMsWrapping,
) -> bool {
    if request.controller_uid != identity.controller_uid || request.boot_id != identity.boot_id {
        return true;
    }

    // The checked-in profile intentionally reaches this branch. It stays
    // stopped, never invents external-gate/fault evidence, and never emits
    // ControllerReady.
    if !identity.profile.grants_motion_authority() || fault_bits != ControllerFaults::NONE.bits() {
        let synchronized = stop_and_drop_session(session, motor);
        return synchronized
            && send_hello(identity.controller_uid, identity.boot_id, identity.profile);
    }

    if let Some(active) = session.as_ref()
        && active.begin == request
    {
        return send_ready(
            identity.controller_uid,
            identity.boot_id,
            identity.profile,
            active.epoch,
            motor.output_state,
            fault_bits,
            now,
        );
    }

    if !stop_and_drop_session(session, motor) {
        return false;
    }
    let Some(epoch) = epoch_generator.next() else {
        return false;
    };
    let Ok(neutral_hold) = DurationMs::try_new(2) else {
        return false;
    };
    let Ok(preload_latch) = DurationMs::try_new(1) else {
        return false;
    };
    let mut controller = Controller::new(ControllerConfig::new(
        epoch,
        identity.profile.controller_envelope,
        identity.profile.maximum_command_lease,
        MotorTiming::new(neutral_hold, preload_latch),
    ));
    let Ok(step) = controller.mark_ready() else {
        return false;
    };
    if !execute_controller_step(&controller, motor, step) {
        return false;
    }
    *session = Some(ControlSession {
        begin: request,
        epoch,
        controller,
        pending_application: None,
        cached_application: None,
    });
    let ready_at = controller_uptime();
    send_ready(
        identity.controller_uid,
        identity.boot_id,
        identity.profile,
        epoch,
        motor.output_state,
        fault_bits,
        ready_at,
    )
}

fn handle_apply_pwm(
    request: ApplyPwm,
    identity: FirmwareIdentity,
    session: &mut Option<ControlSession>,
    motor: &mut HardwareMotor,
    fault_bits: &mut u32,
    now: ControllerUptimeMsWrapping,
) -> bool {
    if request.controller_uid != identity.controller_uid || request.boot_id != identity.boot_id {
        return true;
    }
    if *fault_bits != ControllerFaults::NONE.bits() {
        let synchronized = stop_and_drop_session(session, motor);
        let stopped_at = controller_uptime();
        let result = build_applied_result(
            request,
            AppliedResultCode::FaultedStop,
            motor,
            *fault_bits,
            stopped_at,
        );
        return synchronized && queue_message(Message::AppliedResult(result));
    }
    let epoch_matches = session
        .as_ref()
        .is_some_and(|active| active.epoch == request.control_epoch);
    if !epoch_matches {
        let synchronized = stop_and_drop_session(session, motor);
        let stopped_at = controller_uptime();
        let result = build_applied_result(
            request,
            AppliedResultCode::RejectedSession,
            motor,
            *fault_bits,
            stopped_at,
        );
        return synchronized && queue_message(Message::AppliedResult(result));
    }
    let Some(active) = session.as_mut() else {
        return false;
    };

    let pwm = PwmPair::from_validated(request.timer_pwm.left(), request.timer_pwm.right());
    let command = if request.is_initial_zero_acquisition() {
        ControllerCommand::acquire(
            request.control_epoch,
            pwm,
            request.expires_at,
            identity.profile.controller_envelope.fingerprint(),
        )
    } else {
        ControllerCommand::apply(
            request.control_epoch,
            request.sequence,
            pwm,
            request.expires_at,
        )
    };
    let step = active.controller.accept_command(command, now);
    match step.event() {
        PureControllerEvent::TransitionStarted => {
            if !try_arm_deadline(request.expires_at) {
                motor.disable_and_zero();
                disarm_deadline();
                *fault_bits |= ControllerFaults::DEADLINE_EXPIRED;
                let failure = active.controller.tick(controller_uptime());
                let synchronized = execute_controller_step(&active.controller, motor, failure);
                active.pending_application = None;
                active.cached_application = None;
                let result = build_applied_result(
                    request,
                    AppliedResultCode::RejectedExpired,
                    motor,
                    *fault_bits,
                    controller_uptime(),
                );
                return synchronized && queue_message(Message::AppliedResult(result));
            }
            let synchronized = execute_controller_step(&active.controller, motor, step);
            if !synchronized {
                disarm_deadline();
                let failure = latch_adapter_failure(fault_bits);
                motor.disable_and_zero();
                let result = build_applied_result(
                    request,
                    failure.applied_result_code(),
                    motor,
                    *fault_bits,
                    controller_uptime(),
                );
                return queue_message(Message::AppliedResult(result));
            }
            active.pending_application = Some(request);
            active.cached_application = None;
            true
        }
        PureControllerEvent::ZeroAcquisitionAccepted | PureControllerEvent::StopApplied => {
            let synchronized = execute_controller_step(&active.controller, motor, step);
            disarm_deadline();
            active.pending_application = None;
            let applied_at = controller_uptime();
            let result = build_applied_result(
                request,
                AppliedResultCode::Stopped,
                motor,
                *fault_bits,
                applied_at,
            );
            active.cached_application = Some((request, result));
            synchronized && queue_message(Message::AppliedResult(result))
        }
        PureControllerEvent::DuplicateIgnoredWithoutLeaseRenewal => {
            if let Some((cached_request, cached_result)) = active.cached_application
                && cached_request == request
            {
                let duplicate = AppliedResult {
                    result: AppliedResultCode::DuplicateCached,
                    ..cached_result
                };
                return queue_message(Message::AppliedResult(duplicate));
            }
            // The original effect is still transitioning; its single applied
            // result will be emitted only after hardware-effect completion.
            true
        }
        PureControllerEvent::FaultLatched(fault)
        | PureControllerEvent::AlreadyFaultLatched(fault) => {
            let synchronized = execute_controller_step(&active.controller, motor, step);
            disarm_deadline();
            active.pending_application = None;
            active.cached_application = None;
            *fault_bits |= fault_bits_for(fault);
            let result = build_applied_result(
                request,
                applied_result_for_fault(fault),
                motor,
                *fault_bits,
                controller_uptime(),
            );
            synchronized && queue_message(Message::AppliedResult(result))
        }
        PureControllerEvent::ReadyForZeroAcquisition
        | PureControllerEvent::TransitionAdvanced
        | PureControllerEvent::MotionApplied
        | PureControllerEvent::NoChange => {
            motor.disable_and_zero();
            disarm_deadline();
            *fault_bits |= ControllerFaults::INTERNAL;
            false
        }
    }
}

fn progress_controller(
    active: &mut ControlSession,
    motor: &mut HardwareMotor,
    fault_bits: &mut u32,
    now: ControllerUptimeMsWrapping,
) -> bool {
    let step = active.controller.tick(now);
    match step.event() {
        PureControllerEvent::TransitionAdvanced | PureControllerEvent::NoChange => {
            let synchronized = execute_controller_step(&active.controller, motor, step);
            if !synchronized {
                let _failure = latch_adapter_failure(fault_bits);
            }
            synchronized
        }
        PureControllerEvent::MotionApplied => {
            let synchronized = execute_controller_step(&active.controller, motor, step);
            if !synchronized {
                disarm_deadline();
                motor.disable_and_zero();
                let _failure = latch_adapter_failure(fault_bits);
                active.pending_application = None;
                active.cached_application = None;
                return false;
            }
            let applied_at = controller_uptime();
            let live_after_effect = matches!(
                target_watchdog_status(Some(active), motor, applied_at),
                ControllerWatchdogStatus::SafeDrivingWithLiveLease
            );
            if !live_after_effect {
                let semantic_expiry = matches!(
                    active.controller.watchdog_status_at(applied_at),
                    ControllerWatchdogStatus::UnsafeExpiredOrAmbiguous
                ) || LEASE_EXPIRED_IN_ISR.load(Ordering::Acquire);
                motor.disable_and_zero();
                disarm_deadline();
                if semantic_expiry {
                    *fault_bits |= ControllerFaults::DEADLINE_EXPIRED;
                    let fault_step = active.controller.tick(applied_at);
                    let _synchronized =
                        execute_controller_step(&active.controller, motor, fault_step);
                } else {
                    *fault_bits |= ControllerFaults::INTERNAL;
                }
                let Some(request) = active.pending_application.take() else {
                    active.cached_application = None;
                    return false;
                };
                active.cached_application = None;
                let result = build_applied_result(
                    request,
                    if semantic_expiry {
                        AppliedResultCode::RejectedExpired
                    } else {
                        AppliedResultCode::FaultedStop
                    },
                    motor,
                    *fault_bits,
                    controller_uptime(),
                );
                let queued = queue_message(Message::AppliedResult(result));
                return semantic_expiry && queued;
            }
            let Some(request) = active.pending_application.take() else {
                motor.disable_and_zero();
                disarm_deadline();
                *fault_bits |= ControllerFaults::INTERNAL;
                return false;
            };
            let result = build_applied_result(
                request,
                AppliedResultCode::AppliedNew,
                motor,
                *fault_bits,
                applied_at,
            );
            active.cached_application = Some((request, result));
            queue_message(Message::AppliedResult(result))
        }
        PureControllerEvent::FaultLatched(fault)
        | PureControllerEvent::AlreadyFaultLatched(fault) => {
            let synchronized = execute_controller_step(&active.controller, motor, step);
            disarm_deadline();
            *fault_bits |= fault_bits_for(fault);
            if let Some(request) = active.pending_application.take() {
                let stopped_at = controller_uptime();
                let result = build_applied_result(
                    request,
                    applied_result_for_fault(fault),
                    motor,
                    *fault_bits,
                    stopped_at,
                );
                active.cached_application = None;
                return synchronized && queue_message(Message::AppliedResult(result));
            }
            synchronized
        }
        PureControllerEvent::ReadyForZeroAcquisition
        | PureControllerEvent::ZeroAcquisitionAccepted
        | PureControllerEvent::DuplicateIgnoredWithoutLeaseRenewal
        | PureControllerEvent::StopApplied
        | PureControllerEvent::TransitionStarted => {
            motor.disable_and_zero();
            disarm_deadline();
            *fault_bits |= ControllerFaults::INTERNAL;
            false
        }
    }
}

fn execute_controller_step(
    controller: &Controller,
    motor: &mut HardwareMotor,
    step: ControllerStep,
) -> bool {
    let (target, motion_deadline) = match controller.mode() {
        ControllerMode::Transitioning {
            transition, lease, ..
        } => (transition.target(), Some(lease.expires_at())),
        ControllerMode::Driving { applied, lease, .. } => (applied, Some(lease.expires_at())),
        ControllerMode::BootSafe
        | ControllerMode::AwaitingArm { .. }
        | ControllerMode::ArmedStopped { .. }
        | ControllerMode::FaultLatched { .. } => (PwmPair::STOP, None),
    };
    cortex_m::interrupt::free(|_| {
        let requires_live_deadline = matches!(
            step.motor(),
            MotorDirective::PreloadWhileDisabled(_)
                | MotorDirective::UpdateEnabled(_)
                | MotorDirective::EnablePreloaded(_)
        );
        let expected_deadline = if requires_live_deadline {
            let Some(deadline) = motion_deadline else {
                motor.disable_and_zero();
                return false;
            };
            let before = deadline_timer_snapshot_unlocked();
            if !before.permits_motion_until(deadline) {
                motor.disable_and_zero();
                record_deadline_gate_failure_unlocked(before, deadline);
                return false;
            }
            Some(deadline)
        } else {
            None
        };

        if !motor.apply(step.motor(), target) {
            motor.disable_and_zero();
            return false;
        }

        // TIM5 can reach CCR1 while PRIMASK delays its ISR. Re-sample after
        // the final PWM MMIO and fail closed before leaving the critical
        // section; checking only the software armed flag is insufficient.
        if let Some(deadline) = expected_deadline {
            let after = deadline_timer_snapshot_unlocked();
            if !after.permits_motion_until(deadline) {
                motor.disable_and_zero();
                record_deadline_gate_failure_unlocked(after, deadline);
                return false;
            }
        }
        true
    })
}

fn stop_and_drop_session(session: &mut Option<ControlSession>, motor: &mut HardwareMotor) -> bool {
    motor.disable_and_zero();
    disarm_deadline();
    *session = None;
    true
}

fn build_applied_result(
    request: ApplyPwm,
    result: AppliedResultCode,
    motor: &HardwareMotor,
    fault_bits: u32,
    now: ControllerUptimeMsWrapping,
) -> AppliedResult {
    AppliedResult {
        controller_uid: request.controller_uid,
        boot_id: request.boot_id,
        control_epoch: request.control_epoch,
        sequence: request.sequence,
        result,
        timer_pwm: wire_pwm(motor.timer_pwm),
        output_state: motor.output_state,
        applied_at: now,
        expires_at: request.expires_at,
        faults: wire_faults(fault_bits),
    }
}

const fn applied_result_for_fault(fault: FaultCode) -> AppliedResultCode {
    match fault {
        FaultCode::WrongControlEpoch => AppliedResultCode::RejectedSession,
        FaultCode::CommandLeaseExpired | FaultCode::ClockObservationGap => {
            AppliedResultCode::RejectedExpired
        }
        FaultCode::DuplicateConflict { .. }
        | FaultCode::SequenceGap { .. }
        | FaultCode::SequenceOlder { .. }
        | FaultCode::SequenceAmbiguousHalfRange { .. }
        | FaultCode::SequenceExhausted { .. } => AppliedResultCode::RejectedSequence,
        FaultCode::AcquisitionMustUseSequenceZero
        | FaultCode::AcquisitionMustRequestZero
        | FaultCode::AcquisitionFingerprintMismatch
        | FaultCode::AcquisitionRequired
        | FaultCode::UnexpectedAcquisition
        | FaultCode::LeaseAboveFirmwareMaximum { .. }
        | FaultCode::MotionEnvelope(_)
        | FaultCode::CommandDuringTransition => AppliedResultCode::RejectedDomain,
        FaultCode::CommandBeforeReady | FaultCode::MotorTransition(_) => {
            AppliedResultCode::FaultedStop
        }
    }
}

const fn fault_bits_for(fault: FaultCode) -> u32 {
    match fault {
        FaultCode::CommandLeaseExpired | FaultCode::ClockObservationGap => {
            ControllerFaults::DEADLINE_EXPIRED
        }
        FaultCode::DuplicateConflict { .. }
        | FaultCode::SequenceGap { .. }
        | FaultCode::SequenceOlder { .. }
        | FaultCode::SequenceAmbiguousHalfRange { .. }
        | FaultCode::SequenceExhausted { .. } => ControllerFaults::SEQUENCE,
        FaultCode::CommandBeforeReady
        | FaultCode::WrongControlEpoch
        | FaultCode::AcquisitionMustUseSequenceZero
        | FaultCode::AcquisitionMustRequestZero
        | FaultCode::AcquisitionFingerprintMismatch
        | FaultCode::AcquisitionRequired
        | FaultCode::UnexpectedAcquisition
        | FaultCode::LeaseAboveFirmwareMaximum { .. }
        | FaultCode::MotionEnvelope(_)
        | FaultCode::CommandDuringTransition
        | FaultCode::MotorTransition(_) => ControllerFaults::INTERNAL,
    }
}

#[derive(Clone, Copy)]
enum AdapterFailure {
    DeadlineExpired,
    ReportPathUnavailable,
    Internal,
}

impl AdapterFailure {
    const fn applied_result_code(self) -> AppliedResultCode {
        match self {
            Self::DeadlineExpired => AppliedResultCode::RejectedExpired,
            Self::ReportPathUnavailable | Self::Internal => AppliedResultCode::FaultedStop,
        }
    }
}

fn latch_adapter_failure(fault_bits: &mut u32) -> AdapterFailure {
    if LEASE_EXPIRED_IN_ISR.swap(false, Ordering::AcqRel) {
        *fault_bits |= ControllerFaults::DEADLINE_EXPIRED;
        AdapterFailure::DeadlineExpired
    } else if TX_PATH_FAILED.load(Ordering::Acquire) {
        *fault_bits |= ControllerFaults::SERIAL_INTEGRITY;
        AdapterFailure::ReportPathUnavailable
    } else {
        *fault_bits |= ControllerFaults::INTERNAL;
        AdapterFailure::Internal
    }
}

fn send_hello(
    controller_uid: ControllerUid,
    boot_id: ControllerBootId,
    profile: CompiledActuatorProfile,
) -> bool {
    let Some(watchdog_period) =
        WatchdogNominalPeriodMs::try_new(FIRMWARE_V2_WATCHDOG_NOMINAL_PERIOD_MS).ok()
    else {
        return false;
    };
    let Some(pwm_frequency) = PwmFrequencyHz::try_new(PWM_FREQUENCY_HZ).ok() else {
        return false;
    };
    queue_message(Message::ControllerHello(ControllerHello {
        controller_uid,
        boot_id,
        firmware_abi: FIRMWARE_ABI,
        firmware_build_id: FIRMWARE_BUILD_ID,
        capabilities: profile.capabilities,
        max_abs_pwm_percent: profile.max_abs_pwm,
        max_command_lease: profile.maximum_command_lease,
        output_state: OutputState::Disabled,
        actuator_config_fingerprint: profile.fingerprint,
        watchdog_nominal_period: watchdog_period,
        pwm_frequency,
        neutral_output: profile.neutral_output,
        physical_stop_semantics: profile.physical_stop_semantics,
    }))
}

fn send_ready(
    controller_uid: ControllerUid,
    boot_id: ControllerBootId,
    profile: CompiledActuatorProfile,
    epoch: ControlEpoch,
    output_state: OutputState,
    fault_bits: u32,
    now: ControllerUptimeMsWrapping,
) -> bool {
    queue_message(Message::ControllerReady(ControllerReady {
        controller_uid,
        boot_id,
        control_epoch: epoch,
        controller_uptime: now,
        capabilities: profile.capabilities,
        output_state,
        faults: wire_faults(fault_bits),
    }))
}

fn send_heartbeat(
    controller_uid: ControllerUid,
    boot_id: ControllerBootId,
    profile: CompiledActuatorProfile,
    session: Option<&ControlSession>,
    motor: &HardwareMotor,
    fault_bits: u32,
    now: ControllerUptimeMsWrapping,
) -> bool {
    let deadline_armed = LEASE_ARMED.load(Ordering::Acquire);
    let readiness = if session.is_none() {
        // Neither checked-in profile has a driver-fault input. A no-session
        // heartbeat therefore reports only the independently observed
        // software watchdog state.
        let Some(value) = ReadinessFlags::try_from_bits(ReadinessFlags::WATCHDOG_RUNNING).ok()
        else {
            return false;
        };
        value
    } else {
        // No checked-in production adapter reads a physical driver-fault
        // input. Its profile therefore cannot reach production admission here.
        let Some(readiness) = established_session_readiness(
            profile.capabilities,
            profile.max_abs_pwm,
            profile.physical_stop_semantics,
            deadline_armed,
        ) else {
            return false;
        };
        readiness
    };

    let (control_epoch, last_sequence, expires_at) = match session {
        Some(active) => {
            let last = active.controller.last_command();
            let identity = last.map(|command| (active.epoch, command.sequence()));
            let expires_at = match active.controller.mode() {
                ControllerMode::Transitioning { lease, .. }
                | ControllerMode::Driving { lease, .. } => lease.expires_at(),
                ControllerMode::BootSafe
                | ControllerMode::AwaitingArm { .. }
                | ControllerMode::ArmedStopped { .. }
                | ControllerMode::FaultLatched { .. } => {
                    ControllerDeadlineMsWrapping::new(now.get())
                }
            };
            (
                identity.map(|value| value.0),
                identity.map(|value| value.1),
                expires_at,
            )
        }
        None => (None, None, ControllerDeadlineMsWrapping::new(now.get())),
    };

    queue_message(Message::Heartbeat(Heartbeat {
        controller_uid,
        boot_id,
        control_epoch,
        last_sequence,
        controller_uptime: now,
        expires_at,
        timer_pwm: wire_pwm(motor.timer_pwm),
        output_state: motor.output_state,
        readiness,
        faults: wire_faults(fault_bits),
    }))
}

fn send_odometry(
    controller_uid: ControllerUid,
    boot_id: ControllerBootId,
    control_epoch: Option<ControlEpoch>,
    odometry: EncoderObservation,
) -> bool {
    queue_message(Message::ObservationalOdometry(ObservationalOdometry {
        controller_uid,
        boot_id,
        control_epoch,
        left_estimated_extended_ticks_wrapping: odometry.left_extended,
        right_estimated_extended_ticks_wrapping: odometry.right_extended,
        left_sample_delta_ticks_modulo: odometry.left_delta,
        right_sample_delta_ticks_modulo: odometry.right_delta,
        controller_uptime: odometry.uptime,
    }))
}

fn queue_host_stop_result(result: HostStopResult) -> bool {
    queue_message(Message::HostStopResult(result))
}

fn queue_message(message: Message) -> bool {
    let class = match &message {
        Message::HostStopResult(_) => TxTrafficClass::HostStopResult,
        Message::AppliedResult(_) | Message::ControllerReady(_) => TxTrafficClass::AppliedControl,
        Message::ControllerHello(_)
        | Message::Heartbeat(_)
        | Message::ObservationalOdometry(_)
        | Message::TransportDiagnosticReport(_) => TxTrafficClass::BestEffort,
        _ => {
            TX_PATH_FAILED.store(true, Ordering::Release);
            return false;
        }
    };
    let Ok(record) = UartRecord::encode(message) else {
        TX_PATH_FAILED.store(true, Ordering::Release);
        return false;
    };
    match try_queue_tx_record(class, record.as_bytes()) {
        Ok(()) => true,
        Err(TxQueueError::Admission(TxAdmissionError::QueueFull {
            class: TxTrafficClass::BestEffort,
            ..
        })) if class == TxTrafficClass::BestEffort => true,
        Err(_) => {
            TX_PATH_FAILED.store(true, Ordering::Release);
            false
        }
    }
}

const fn wire_pwm(pair: PwmPair) -> TimerPwm {
    TimerPwm::from_validated(pair.left(), pair.right())
}

fn wire_faults(bits: u32) -> ControllerFaults {
    match ControllerFaults::try_from_bits(bits) {
        Ok(faults) => faults,
        Err(_) => match ControllerFaults::try_from_bits(ControllerFaults::INTERNAL) {
            Ok(internal) => internal,
            Err(_) => fatal_reset(),
        },
    }
}

fn scale_duty(magnitude_percent: u8, maximum_duty: NonZeroU16) -> Option<u16> {
    let scaled = u32::from(magnitude_percent) * u32::from(maximum_duty.get()) / 100;
    u16::try_from(scaled).ok()
}

#[cfg(all(
    feature = "external-boot-identity",
    not(feature = "flash-boot-journal")
))]
#[allow(unsafe_code)]
fn load_boot_identity(_controller_uid: ControllerUid, _flash: pac::FLASH) -> Option<BootIdentity> {
    unsafe extern "C" {
        /// Must return a nonzero identifier that cannot repeat across boots
        /// while any command from an earlier boot could remain in flight.
        fn kiko_external_boot_id() -> u64;
    }
    // SAFETY: enabling `external-boot-identity` is an explicit link-time
    // contract. A reviewed board integration must provide this symbol with the
    // documented C ABI and per-boot uniqueness semantics; omission fails link.
    let value = unsafe { kiko_external_boot_id() };
    Some(BootIdentity {
        id: ControllerBootId::try_new(value).ok()?,
        is_session_unique: true,
    })
}

#[cfg(all(
    feature = "flash-boot-journal",
    not(feature = "external-boot-identity")
))]
fn load_boot_identity(_controller_uid: ControllerUid, flash: pac::FLASH) -> Option<BootIdentity> {
    let mut flash = LockedFlash::new(flash);
    if flash.len() != STM32F446_FLASH_BYTES {
        return None;
    }
    let journal_end = BOOT_JOURNAL_FLASH_OFFSET.checked_add(BOOT_JOURNAL_FLASH_BYTES)?;
    let commit = {
        let journal = flash.read().get(BOOT_JOURNAL_FLASH_OFFSET..journal_end)?;
        plan_next_boot(journal).ok()?
    };
    {
        let absolute_offset = BOOT_JOURNAL_FLASH_OFFSET.checked_add(commit.record_offset())?;
        let mut unlocked = flash.unlocked();
        unlocked
            .program(absolute_offset, commit.record().iter())
            .ok()?;
    }
    let boot_id = {
        let journal = flash.read().get(BOOT_JOURNAL_FLASH_OFFSET..journal_end)?;
        verify_commit(journal, commit).ok()?
    };
    Some(BootIdentity {
        id: boot_id,
        is_session_unique: true,
    })
}

#[cfg(not(any(feature = "external-boot-identity", feature = "flash-boot-journal")))]
fn load_boot_identity(controller_uid: ControllerUid, _flash: pac::FLASH) -> Option<BootIdentity> {
    // STM32F446 has no hardware RNG. This deterministic FNV-1a token exists
    // only so the motion-disabled default can correlate safe diagnostics. It
    // repeats after reset and is explicitly not a boot-session guarantee.
    let mut token = 0xcbf2_9ce4_8422_2325_u64;
    for &byte in controller_uid.as_bytes() {
        token ^= u64::from(byte);
        token = token.wrapping_mul(0x0000_0100_0000_01b3);
    }
    Some(BootIdentity {
        id: ControllerBootId::try_new(token | 1).ok()?,
        is_session_unique: false,
    })
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

fn serial_queue_depths() -> Option<(u16, u16)> {
    cortex_m::interrupt::free(|cs| {
        let rx_depth = u16::try_from(RX_QUEUE.borrow(cs).borrow().len()).ok()?;
        let tx_depth = u16::try_from(TX_SCHEDULER.borrow(cs).borrow().queued_bytes()).ok()?;
        Some((rx_depth, tx_depth))
    })
}

enum TxQueueError {
    SerialUnavailable,
    Admission(TxAdmissionError),
}

fn try_queue_tx_record(class: TxTrafficClass, record: &[u8]) -> Result<(), TxQueueError> {
    cortex_m::interrupt::free(|cs| {
        let mut serial_slot = SERIAL.borrow(cs).borrow_mut();
        let Some(serial) = serial_slot.as_mut() else {
            return Err(TxQueueError::SerialUnavailable);
        };
        TX_SCHEDULER
            .borrow(cs)
            .borrow_mut()
            .try_enqueue_record(class, record)
            .map_err(TxQueueError::Admission)?;
        serial.listen(SerialEvent::TxEmpty);
        Ok(())
    })
}

#[interrupt]
fn USART2() {
    cortex_m::interrupt::free(|cs| {
        if let Some(serial) = SERIAL.borrow(cs).borrow_mut().as_mut() {
            if serial.is_rx_not_empty() {
                match serial.read() {
                    Ok(byte) => {
                        if RX_QUEUE.borrow(cs).borrow_mut().enqueue(byte).is_err() {
                            RX_STREAM_INVALIDATED.store(true, Ordering::Release);
                        }
                    }
                    Err(nb::Error::WouldBlock) => {}
                    Err(nb::Error::Other(_)) => {
                        RX_STREAM_INVALIDATED.store(true, Ordering::Release);
                    }
                }
            }

            if serial.is_tx_empty() {
                let mut scheduler = TX_SCHEDULER.borrow(cs).borrow_mut();
                if let Some(byte) = scheduler.peek_byte() {
                    match serial.write(byte) {
                        Ok(()) => {
                            let consumed = scheduler.consume_byte();
                            if consumed != Some(byte) {
                                TX_PATH_FAILED.store(true, Ordering::Release);
                                serial.unlisten(SerialEvent::TxEmpty);
                            }
                        }
                        Err(nb::Error::WouldBlock) => {}
                        Err(nb::Error::Other(_)) => {
                            TX_PATH_FAILED.store(true, Ordering::Release);
                            serial.unlisten(SerialEvent::TxEmpty);
                        }
                    }
                } else {
                    serial.unlisten(SerialEvent::TxEmpty);
                }
            }
        }
    });
}

#[interrupt]
#[allow(unsafe_code)]
fn TIM5() {
    // SAFETY: TIM5 is configured once before this interrupt is unmasked. The
    // handler is the only context that clears CC1IF; main modifies CC1IE/CCR1
    // only with interrupts masked.
    unsafe {
        let timer = &*pac::TIM5::ptr();
        if timer.sr.read().cc1if().bit_is_set() {
            timer.sr.modify(|_, writer| writer.cc1if().clear_bit());
            timer.dier.modify(|_, writer| writer.cc1ie().clear_bit());
            if LEASE_ARMED.swap(false, Ordering::AcqRel) {
                emergency_disable_motor_outputs();
                LEASE_EXPIRED_IN_ISR.store(true, Ordering::Release);
            }
        }
    }
}

#[interrupt]
#[allow(unsafe_code)]
fn TIM1_UP_TIM10() {
    cortex_m::interrupt::free(|cs| {
        // SAFETY: TIM1 is initialized as the left encoder before its interrupt
        // is unmasked. Register access is serialized by interrupt masking.
        unsafe {
            let timer = &*pac::TIM1::ptr();
            if timer.sr.read().uif().bit_is_set() {
                timer.sr.modify(|_, writer| writer.uif().clear_bit());
                let wraps = LEFT_ENCODER_WRAPS.borrow(cs);
                if timer.cr1.read().dir().bit_is_set() {
                    wraps.set(wraps.get().wrapping_sub(1));
                } else {
                    wraps.set(wraps.get().wrapping_add(1));
                }
            }
        }
    });
}

#[interrupt]
#[allow(unsafe_code)]
fn TIM4() {
    cortex_m::interrupt::free(|cs| {
        // SAFETY: TIM4 is initialized as the right encoder before its
        // interrupt is unmasked. Register access is serialized here.
        unsafe {
            let timer = &*pac::TIM4::ptr();
            if timer.sr.read().uif().bit_is_set() {
                timer.sr.modify(|_, writer| writer.uif().clear_bit());
                let wraps = RIGHT_ENCODER_WRAPS.borrow(cs);
                if timer.cr1.read().dir().bit_is_set() {
                    wraps.set(wraps.get().wrapping_sub(1));
                } else {
                    wraps.set(wraps.get().wrapping_add(1));
                }
            }
        }
    });
}

#[allow(unsafe_code)]
fn configure_deadline_timer(timer: pac::TIM5, clocks: &stm32f4xx_hal::rcc::Clocks) {
    // PCLK1 is configured to 21 MHz, so the doubled APB1 timer clock is
    // 42 MHz and a 41_999 prescaler produces an exact 1 kHz counter.
    let timer = FTimer::<pac::TIM5, 1_000>::new(timer, clocks).release();
    // SAFETY: this function owns TIM5 exclusively and executes before TIM5 is
    // exposed through its interrupt handler.
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
fn controller_uptime() -> ControllerUptimeMsWrapping {
    // SAFETY: TIM5 is a free-running 1 kHz counter after initialization. A
    // volatile peripheral read has no aliasing requirement beyond PAC MMIO.
    let value = unsafe { (&*pac::TIM5::ptr()).cnt.read().cnt().bits() };
    ControllerUptimeMsWrapping::new(value)
}

#[allow(unsafe_code)]
fn deadline_timer_snapshot_unlocked() -> DeadlineTimerSnapshot {
    // SAFETY: callers sample this only while interrupts are masked, making the
    // related CNT/SR/DIER/CCR1 reads one coherent adapter observation with
    // respect to TIM5. The counter itself intentionally continues running.
    unsafe {
        let timer = &*pac::TIM5::ptr();
        DeadlineTimerSnapshot::new(
            LEASE_ARMED.load(Ordering::Acquire),
            timer.dier.read().cc1ie().bit_is_set(),
            timer.sr.read().cc1if().bit_is_set(),
            ControllerDeadlineMsWrapping::new(timer.ccr1().read().ccr().bits()),
            ControllerUptimeMsWrapping::new(timer.cnt.read().cnt().bits()),
        )
    }
}

#[allow(unsafe_code)]
fn record_deadline_gate_failure_unlocked(
    snapshot: DeadlineTimerSnapshot,
    expected_deadline: ControllerDeadlineMsWrapping,
) {
    // SAFETY: called with interrupts masked after outputs have been disabled.
    // Prevent a mismatched compare from retaining motion authority. If the
    // coherent evidence says the expected deadline was reached, preserve that
    // fact for the main loop because disabling CC1IE prevents the ISR from
    // recording it later.
    unsafe {
        (&*pac::TIM5::ptr())
            .dier
            .modify(|_, writer| writer.cc1ie().clear_bit());
    }
    LEASE_ARMED.store(false, Ordering::Release);
    if snapshot.indicates_expiry_of(expected_deadline) {
        LEASE_EXPIRED_IN_ISR.store(true, Ordering::Release);
    }
}

fn target_watchdog_status(
    session: Option<&ControlSession>,
    motor: &HardwareMotor,
    now: ControllerUptimeMsWrapping,
) -> ControllerWatchdogStatus {
    let Some(active) = session else {
        return if motor.output_state.is_safe()
            && motor.timer_pwm.is_stop()
            && !LEASE_ARMED.load(Ordering::Acquire)
        {
            ControllerWatchdogStatus::SafeOutputsDisabled
        } else {
            ControllerWatchdogStatus::UnsafeExpiredOrAmbiguous
        };
    };

    if !motor_matches_controller_mode(&active.controller, motor) {
        return ControllerWatchdogStatus::UnsafeExpiredOrAmbiguous;
    }

    let pure_status = active.controller.watchdog_status_at(now);
    let expected_deadline = match active.controller.mode() {
        ControllerMode::Transitioning { lease, .. } | ControllerMode::Driving { lease, .. } => {
            Some(lease.expires_at())
        }
        ControllerMode::BootSafe
        | ControllerMode::AwaitingArm { .. }
        | ControllerMode::ArmedStopped { .. }
        | ControllerMode::FaultLatched { .. } => None,
    };
    match (pure_status, expected_deadline) {
        (
            ControllerWatchdogStatus::SafeTransitionWithLiveLease
            | ControllerWatchdogStatus::SafeDrivingWithLiveLease,
            Some(deadline),
        ) => {
            let hardware_live = cortex_m::interrupt::free(|_| {
                deadline_timer_snapshot_unlocked().permits_motion_until(deadline)
            });
            if hardware_live && !LEASE_EXPIRED_IN_ISR.load(Ordering::Acquire) {
                pure_status
            } else {
                ControllerWatchdogStatus::UnsafeExpiredOrAmbiguous
            }
        }
        (ControllerWatchdogStatus::SafeOutputsDisabled, None)
            if motor.output_state.is_safe()
                && motor.timer_pwm.is_stop()
                && !LEASE_ARMED.load(Ordering::Acquire) =>
        {
            ControllerWatchdogStatus::SafeOutputsDisabled
        }
        _ => ControllerWatchdogStatus::UnsafeExpiredOrAmbiguous,
    }
}

fn motor_matches_controller_mode(controller: &Controller, motor: &HardwareMotor) -> bool {
    match controller.mode() {
        ControllerMode::BootSafe
        | ControllerMode::AwaitingArm { .. }
        | ControllerMode::ArmedStopped { .. }
        | ControllerMode::FaultLatched { .. } => {
            motor.output_state.is_safe() && motor.timer_pwm.is_stop()
        }
        ControllerMode::Driving { applied, .. } => {
            motor.output_state == OutputState::NonzeroPwm && motor.timer_pwm == applied
        }
        ControllerMode::Transitioning { transition, .. } => match transition.phase() {
            MotorTransitionPhase::Neutralizing => {
                motor.output_state.is_safe() && motor.timer_pwm.is_stop()
            }
            MotorTransitionPhase::PreloadingDisabled => {
                motor.output_state.is_safe() && motor.timer_pwm == transition.target()
            }
            MotorTransitionPhase::UpdatingEnabled => {
                motor.output_state == OutputState::NonzeroPwm
                    && motor.timer_pwm == transition.target()
            }
        },
    }
}

#[allow(unsafe_code)]
fn try_arm_deadline(deadline: ControllerDeadlineMsWrapping) -> bool {
    cortex_m::interrupt::free(|_| {
        // SAFETY: all main-context CCR1/CC1IE changes occur with interrupts
        // masked; TIM5 ISR only clears the same interrupt and never changes
        // the free-running counter.
        unsafe {
            let timer = &*pac::TIM5::ptr();
            if LEASE_EXPIRED_IN_ISR.load(Ordering::Acquire)
                || (LEASE_ARMED.load(Ordering::Acquire) && timer.sr.read().cc1if().bit_is_set())
            {
                timer.dier.modify(|_, writer| writer.cc1ie().clear_bit());
                LEASE_ARMED.store(false, Ordering::Release);
                emergency_disable_motor_outputs();
                LEASE_EXPIRED_IN_ISR.store(true, Ordering::Release);
                return false;
            }
            let before = ControllerUptimeMsWrapping::new(timer.cnt.read().cnt().bits());
            if !matches!(
                deadline.relation_to(before),
                DeadlineRelation::Future { .. }
            ) {
                return false;
            }
            timer.dier.modify(|_, writer| writer.cc1ie().clear_bit());
            timer
                .ccr1()
                .write(|writer| writer.ccr().bits(deadline.get()));
            timer.sr.modify(|_, writer| writer.cc1if().clear_bit());
            let after = ControllerUptimeMsWrapping::new(timer.cnt.read().cnt().bits());
            if !matches!(deadline.relation_to(after), DeadlineRelation::Future { .. }) {
                return false;
            }
            LEASE_ARMED.store(true, Ordering::Release);
            timer.dier.modify(|_, writer| writer.cc1ie().set_bit());
            true
        }
    })
}

#[allow(unsafe_code)]
fn disarm_deadline() {
    cortex_m::interrupt::free(|_| {
        // SAFETY: main owns deadline arming and masks TIM5 while disabling the
        // compare source and clearing a stale flag.
        unsafe {
            let timer = &*pac::TIM5::ptr();
            timer.dier.modify(|_, writer| writer.cc1ie().clear_bit());
            timer.sr.modify(|_, writer| writer.cc1if().clear_bit());
            LEASE_ARMED.store(false, Ordering::Release);
        }
    });
}

#[allow(unsafe_code)]
fn emergency_disable_motor_outputs() {
    // SAFETY: these are idempotent emergency MMIO writes. They may run before
    // timer clocks are enabled (writes are then ineffective) or alongside HAL
    // channel handles only from an interrupt/fault context. CCER is cleared
    // before all compare values, so no direction pair is enabled together.
    unsafe {
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
    LEASE_ARMED.store(false, Ordering::Release);
}

#[allow(unsafe_code)]
fn read_controller_uid() -> Option<ControllerUid> {
    let mut bytes = [0_u8; 12];
    for index in 0..3 {
        // SAFETY: 0x1fff_7a10 is the documented aligned 96-bit unique-device
        // identifier address for STM32F446. Reads are volatile and in range.
        let word =
            unsafe { core::ptr::read_volatile((STM32F446_UID_ADDRESS as *const u32).add(index)) };
        let start = index * 4;
        bytes[start..start + 4].copy_from_slice(&word.to_le_bytes());
    }
    ControllerUid::try_new(bytes).ok()
}

#[allow(unsafe_code)]
fn enable_interrupts(nvic: &mut NVIC, quadrature_inputs_configured: bool) {
    // SAFETY: every peripheral and shared state is fully initialized before
    // these interrupt lines are prioritized and unmasked.
    unsafe {
        nvic.set_priority(pac::Interrupt::TIM5, 0);
        nvic.set_priority(pac::Interrupt::USART2, 2);
        NVIC::unmask(pac::Interrupt::TIM5);
        if quadrature_inputs_configured {
            nvic.set_priority(pac::Interrupt::TIM1_UP_TIM10, 1);
            nvic.set_priority(pac::Interrupt::TIM4, 1);
            NVIC::unmask(pac::Interrupt::TIM1_UP_TIM10);
            NVIC::unmask(pac::Interrupt::TIM4);
        }
        NVIC::unmask(pac::Interrupt::USART2);
    }
}

#[allow(unsafe_code)]
fn configure_encoder_tim1(timer: pac::TIM1) {
    // SAFETY: the caller owns TIM1 and invokes this before the update interrupt
    // is unmasked. Field values are documented encoder-mode settings.
    unsafe {
        (*pac::RCC::ptr())
            .apb2enr
            .modify(|_, writer| writer.tim1en().set_bit());
        timer.cr1.modify(|_, writer| writer.cen().clear_bit());
        timer.smcr.modify(|_, writer| writer.sms().bits(0b011));
        timer.ccmr1_input().modify(|_, writer| {
            writer
                .cc1s()
                .bits(0b01)
                .ic1f()
                .bits(0b0011)
                .cc2s()
                .bits(0b01)
                .ic2f()
                .bits(0b0011)
        });
        timer
            .ccer
            .modify(|_, writer| writer.cc1p().clear_bit().cc2p().clear_bit());
        timer.arr.write(|writer| writer.arr().bits(u16::MAX));
        timer.dier.modify(|_, writer| writer.uie().set_bit());
        timer.sr.modify(|_, writer| writer.uif().clear_bit());
        timer.cr1.modify(|_, writer| writer.cen().set_bit());
    }
}

#[allow(unsafe_code)]
fn configure_encoder_tim4(timer: pac::TIM4) {
    // SAFETY: the caller owns TIM4 and invokes this before the update interrupt
    // is unmasked. Field values are documented encoder-mode settings.
    unsafe {
        (*pac::RCC::ptr())
            .apb1enr
            .modify(|_, writer| writer.tim4en().set_bit());
        timer.cr1.modify(|_, writer| writer.cen().clear_bit());
        timer.smcr.modify(|_, writer| writer.sms().bits(0b011));
        timer.ccmr1_input().modify(|_, writer| {
            writer
                .cc1s()
                .bits(0b01)
                .ic1f()
                .bits(0b0011)
                .cc2s()
                .bits(0b01)
                .ic2f()
                .bits(0b0011)
        });
        timer
            .ccer
            .modify(|_, writer| writer.cc1p().clear_bit().cc2p().clear_bit());
        timer.arr.write(|writer| writer.arr().bits(u16::MAX));
        timer.dier.modify(|_, writer| writer.uie().set_bit());
        timer.sr.modify(|_, writer| writer.uif().clear_bit());
        timer.cr1.modify(|_, writer| writer.cen().set_bit());
    }
}

#[derive(Clone, Copy)]
struct EncoderObservation {
    left_extended: EstimatedWrappingEncoderTicks,
    right_extended: EstimatedWrappingEncoderTicks,
    left_delta: ModuloEncoderDeltaTicks,
    right_delta: ModuloEncoderDeltaTicks,
    uptime: ControllerUptimeMsWrapping,
}

#[allow(unsafe_code)]
fn sample_odometry(
    uptime: ControllerUptimeMsWrapping,
    previous_left: &mut u16,
    previous_right: &mut u16,
) -> EncoderObservation {
    let (left_count, right_count, left_wraps, right_wraps) = cortex_m::interrupt::free(|cs| {
        // SAFETY: both encoder timers are initialized before sampling and
        // interrupt masking makes count/pending-wrap snapshots coherent.
        unsafe {
            let left = &*pac::TIM1::ptr();
            let right = &*pac::TIM4::ptr();
            let left_count = left.cnt.read().cnt().bits();
            let right_count = right.cnt.read().cnt().bits();
            let left_wraps = encoder_wraps_with_pending_direction_assumption(
                LEFT_ENCODER_WRAPS.borrow(cs).get(),
                left.sr.read().uif().bit_is_set(),
                left.cr1.read().dir().bit_is_set(),
            );
            let right_wraps = encoder_wraps_with_pending_direction_assumption(
                RIGHT_ENCODER_WRAPS.borrow(cs).get(),
                right.sr.read().uif().bit_is_set(),
                right.cr1.read().dir().bit_is_set(),
            );
            (left_count, right_count, left_wraps, right_wraps)
        }
    });
    let observation = EncoderObservation {
        left_extended: EstimatedWrappingEncoderTicks::from_extended_16_bit_counter(
            left_wraps, left_count,
        ),
        right_extended: EstimatedWrappingEncoderTicks::from_extended_16_bit_counter(
            right_wraps,
            right_count,
        ),
        left_delta: ModuloEncoderDeltaTicks::from_wrapping_counts(*previous_left, left_count),
        right_delta: ModuloEncoderDeltaTicks::from_wrapping_counts(*previous_right, right_count),
        uptime,
    };
    *previous_left = left_count;
    *previous_right = right_count;
    observation
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

fn fatal_reset() -> ! {
    cortex_m::interrupt::disable();
    emergency_disable_motor_outputs();
    loop {
        cortex_m::asm::wfi();
    }
}
