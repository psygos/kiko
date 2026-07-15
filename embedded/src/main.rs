#![no_std]
#![no_main]

use core::{
    cell::{Cell, RefCell},
    fmt::Write,
    num::NonZeroU32,
};
use cortex_m::interrupt::Mutex;
use cortex_m_rt::entry;
use embedded::{encoder_wraps_with_pending_direction_assumption, pwm_duty};
use heapless::{String, Vec, spsc::Queue};
use panic_halt as _;
use robot_protocol::{
    ControllerError, ControllerEvent, ControllerUptimeMsWrapping, EstimatedWrappingEncoderTicks,
    ModuloEncoderDeltaTicks, PwmPercent, RobotOdometry, WrappingMillisClock,
    parse_serial_pwm_command,
};
use stm32f4xx_hal::{
    dwt::{Instant as CycleInstant, MonoTimer},
    pac,
    prelude::*,
    serial::{Event, Serial, config::Config},
    timer::{Channel1, Channel2, pwm::PwmExt},
};

const MAIN_LOOP_MIN_DELAY_MS: u32 = 5;
const ODOMETRY_SAMPLE_MIN_INTERVAL_MS: u32 = 10;
const ODOMETRY_REPORT_MIN_INTERVAL_MS: u32 = 100;

const RX_BUFFER_SIZE: usize = 128;
static RX_QUEUE: Mutex<RefCell<Queue<u8, RX_BUFFER_SIZE>>> = Mutex::new(RefCell::new(Queue::new()));
static RX_STREAM_INVALIDATED: Mutex<Cell<bool>> = Mutex::new(Cell::new(false));

const TX_BUFFER_SIZE: usize = 128;
static TX_QUEUE: Mutex<RefCell<Queue<u8, TX_BUFFER_SIZE>>> = Mutex::new(RefCell::new(Queue::new()));
static TX_RECORD_DROPPED: Mutex<Cell<bool>> = Mutex::new(Cell::new(false));

static SERIAL: Mutex<RefCell<Option<stm32f4xx_hal::serial::Serial<pac::USART2>>>> =
    Mutex::new(RefCell::new(None));

#[derive(Clone, Copy)]
struct ActiveCommandLease {
    started_at: CycleInstant,
    duration_ticks: NonZeroU32,
}

static LEFT_ENCODER_WRAPS: Mutex<Cell<i64>> = Mutex::new(Cell::new(0));
static RIGHT_ENCODER_WRAPS: Mutex<Cell<i64>> = Mutex::new(Cell::new(0));

#[entry]
fn main() -> ! {
    if let (Some(dp), Some(cp)) = (
        pac::Peripherals::take(),
        cortex_m::peripheral::Peripherals::take(),
    ) {
        // Configure clocks
        let rcc = dp.RCC.constrain();
        let clocks = rcc
            .cfgr
            .sysclk(168.MHz())
            .pclk1(42.MHz())
            .pclk2(84.MHz())
            .freeze();

        let mut delay = cp.SYST.delay(&clocks);
        let monotonic = MonoTimer::new(cp.DWT, cp.DCB, &clocks);
        let gpioa = dp.GPIOA.split();
        let gpiob = dp.GPIOB.split();

        // Status LED
        let mut led = gpioa.pa5.into_push_pull_output();
        let timer_frequency_hz = match NonZeroU32::new(monotonic.frequency().raw()) {
            Some(frequency) => frequency,
            None => {
                led.set_high();
                loop {
                    cortex_m::asm::wfi();
                }
            }
        };

        // Configure encoder pins first
        let _pa8_enc = gpioa.pa8.into_alternate::<1>(); // TIM1 CH1
        let _pa9_enc = gpioa.pa9.into_alternate::<1>(); // TIM1 CH2
        let _pb6_enc = gpiob.pb6.into_alternate::<2>(); // TIM4 CH1
        let _pb7_enc = gpiob.pb7.into_alternate::<2>(); // TIM4 CH2

        // Configure UART with interrupts
        let tx = gpioa.pa2.into_alternate::<7>();
        let rx = gpioa.pa3.into_alternate::<7>();

        let mut serial = match Serial::new(
            dp.USART2,
            (tx, rx),
            Config::default()
                .baudrate(115200.bps())
                .wordlength_8()
                .parity_none(),
            &clocks,
        ) {
            Ok(serial) => serial,
            Err(_) => {
                led.set_high();
                loop {
                    cortex_m::asm::wfi();
                }
            }
        };

        // Enable RXNE interrupt
        serial.listen(Event::RxNotEmpty);

        cortex_m::interrupt::free(|cs| SERIAL.borrow(cs).replace(Some(serial)));

        // Enable USART2 interrupt in NVIC
        unsafe {
            cortex_m::peripheral::NVIC::unmask(pac::Interrupt::USART2);
        }

        // Configure encoders
        configure_encoder_tim1(dp.TIM1);
        configure_encoder_tim4(dp.TIM4);

        // Configure motors
        let left_channels = (
            Channel1::new(gpioa.pa0.into_alternate()),
            Channel2::new(gpioa.pa1.into_alternate()),
        );
        let (mut left_ch1, mut left_ch2) = dp.TIM2.pwm_hz(left_channels, 20.kHz(), &clocks).split();

        let right_channels = (
            Channel1::new(gpiob.pb4.into_alternate()),
            Channel2::new(gpiob.pb5.into_alternate()),
        );
        let (mut right_ch1, mut right_ch2) =
            dp.TIM3.pwm_hz(right_channels, 20.kHz(), &clocks).split();

        let left_max_duty = left_ch1.get_max_duty();
        let right_max_duty = right_ch1.get_max_duty();

        // Enable all PWM channels
        left_ch1.enable();
        left_ch2.enable();
        right_ch1.enable();
        right_ch2.enable();

        // Initialize motors stopped
        left_ch1.set_duty(0);
        left_ch2.set_duty(0);
        right_ch1.set_duty(0);
        right_ch2.set_duty(0);

        // Enable encoder timer interrupts for overflow handling
        unsafe {
            cortex_m::peripheral::NVIC::unmask(pac::Interrupt::TIM1_UP_TIM10);
            cortex_m::peripheral::NVIC::unmask(pac::Interrupt::TIM4);
        }

        send_controller_event(ControllerEvent::Ready);

        // Control state
        let mut left_pwm = PwmPercent::ZERO;
        let mut right_pwm = PwmPercent::ZERO;
        let mut active_lease = None;
        let mut controller_clock =
            WrappingMillisClock::new(timer_frequency_hz, cortex_m::peripheral::DWT::cycle_count());
        let mut last_odometry_sample_ms = ControllerUptimeMsWrapping::new(0);
        let mut last_odometry_report_ms = ControllerUptimeMsWrapping::new(0);
        let mut last_left_count = 0;
        let mut last_right_count = 0;
        let mut odometry = None;

        let mut command_line = Vec::<u8, 32>::new();
        let mut discard_until_line_end = false;

        loop {
            let controller_uptime_ms_wrapping =
                controller_clock.advance_to(cortex_m::peripheral::DWT::cycle_count());

            if take_rx_stream_invalidated() {
                command_line.clear();
                discard_until_line_end = true;
                active_lease = None;
                left_pwm = PwmPercent::ZERO;
                right_pwm = PwmPercent::ZERO;
                stop_motors(&mut left_ch1, &mut left_ch2, &mut right_ch1, &mut right_ch2);
                send_controller_error(ControllerError::ReceiveOverrun);
            }

            while let Some(byte) = dequeue_rx() {
                if byte == b'\n' || byte == b'\r' {
                    if discard_until_line_end {
                        discard_until_line_end = false;
                        command_line.clear();
                        continue;
                    }
                    if command_line.is_empty() {
                        continue;
                    }

                    let parsed = match parse_serial_pwm_command(&command_line) {
                        Ok(command) => command
                            .lease_ms()
                            .wrapping_timer_ticks_ceil(timer_frequency_hz)
                            .map(|duration_ticks| (command, duration_ticks))
                            .map_err(|_| ControllerError::LeaseTimerDomain),
                        Err(_) => Err(ControllerError::InvalidCommand),
                    };
                    command_line.clear();

                    match parsed {
                        Ok((command, duration_ticks)) => {
                            left_pwm = command.left_pwm_percent();
                            right_pwm = command.right_pwm_percent();
                            apply_motor_pwm(&mut left_ch1, &mut left_ch2, left_pwm, left_max_duty);
                            apply_motor_pwm(
                                &mut right_ch1,
                                &mut right_ch2,
                                right_pwm,
                                right_max_duty,
                            );
                            active_lease =
                                if left_pwm == PwmPercent::ZERO && right_pwm == PwmPercent::ZERO {
                                    None
                                } else {
                                    Some(ActiveCommandLease {
                                        started_at: monotonic.now(),
                                        duration_ticks,
                                    })
                                };
                            led.toggle();
                            send_unsequenced_applied_pwm_report(left_pwm, right_pwm);
                        }
                        Err(error) => {
                            active_lease = None;
                            left_pwm = PwmPercent::ZERO;
                            right_pwm = PwmPercent::ZERO;
                            stop_motors(
                                &mut left_ch1,
                                &mut left_ch2,
                                &mut right_ch1,
                                &mut right_ch2,
                            );
                            send_controller_error(error);
                        }
                    }
                } else if !discard_until_line_end && command_line.push(byte).is_err() {
                    command_line.clear();
                    discard_until_line_end = true;
                    active_lease = None;
                    left_pwm = PwmPercent::ZERO;
                    right_pwm = PwmPercent::ZERO;
                    stop_motors(&mut left_ch1, &mut left_ch2, &mut right_ch1, &mut right_ch2);
                    send_controller_error(ControllerError::CommandTooLong);
                }
            }

            if active_lease.is_some_and(|lease: ActiveCommandLease| {
                lease.started_at.elapsed() >= lease.duration_ticks.get()
            }) {
                active_lease = None;
                if left_pwm != PwmPercent::ZERO || right_pwm != PwmPercent::ZERO {
                    left_pwm = PwmPercent::ZERO;
                    right_pwm = PwmPercent::ZERO;
                    stop_motors(&mut left_ch1, &mut left_ch2, &mut right_ch1, &mut right_ch2);
                    send_controller_event(ControllerEvent::CommandLeaseExpired);
                }
            }

            if controller_uptime_ms_wrapping.wrapping_elapsed_since(last_odometry_sample_ms)
                >= ODOMETRY_SAMPLE_MIN_INTERVAL_MS
            {
                last_odometry_sample_ms = controller_uptime_ms_wrapping;
                odometry = Some(sample_odometry(
                    controller_uptime_ms_wrapping,
                    &mut last_left_count,
                    &mut last_right_count,
                ));
            }

            if controller_uptime_ms_wrapping.wrapping_elapsed_since(last_odometry_report_ms)
                >= ODOMETRY_REPORT_MIN_INTERVAL_MS
            {
                last_odometry_report_ms = controller_uptime_ms_wrapping;
                if let Some(odometry) = odometry {
                    send_odometry_report(odometry);
                }
            }

            report_dropped_tx_record();

            delay.delay_ms(MAIN_LOOP_MIN_DELAY_MS);
        }
    }

    loop {
        cortex_m::asm::wfi();
    }
}

// TIM1 Update interrupt handler (left encoder overflow)
#[interrupt]
fn TIM1_UP_TIM10() {
    cortex_m::interrupt::free(|cs| unsafe {
        let tim1 = &*pac::TIM1::ptr();

        if tim1.sr.read().uif().bit_is_set() {
            tim1.sr.modify(|_, w| w.uif().clear_bit());
            let wraps = LEFT_ENCODER_WRAPS.borrow(cs);
            if tim1.cr1.read().dir().bit_is_set() {
                wraps.set(wraps.get().wrapping_sub(1));
            } else {
                wraps.set(wraps.get().wrapping_add(1));
            }
        }
    });
}

// TIM4 interrupt handler (right encoder overflow)
#[interrupt]
fn TIM4() {
    cortex_m::interrupt::free(|cs| unsafe {
        let tim4 = &*pac::TIM4::ptr();

        if tim4.sr.read().uif().bit_is_set() {
            tim4.sr.modify(|_, w| w.uif().clear_bit());
            let wraps = RIGHT_ENCODER_WRAPS.borrow(cs);
            if tim4.cr1.read().dir().bit_is_set() {
                wraps.set(wraps.get().wrapping_sub(1));
            } else {
                wraps.set(wraps.get().wrapping_add(1));
            }
        }
    });
}

// USART2 interrupt handler
#[interrupt]
fn USART2() {
    cortex_m::interrupt::free(|cs| {
        if let Some(serial) = SERIAL.borrow(cs).borrow_mut().as_mut() {
            if serial.is_rx_not_empty() {
                match serial.read() {
                    Ok(byte) => {
                        if RX_QUEUE.borrow(cs).borrow_mut().enqueue(byte).is_err() {
                            RX_STREAM_INVALIDATED.borrow(cs).set(true);
                        }
                    }
                    Err(nb::Error::WouldBlock) => {}
                    Err(nb::Error::Other(_)) => RX_STREAM_INVALIDATED.borrow(cs).set(true),
                }
            }

            if serial.is_tx_empty() {
                let mut tx_queue = TX_QUEUE.borrow(cs).borrow_mut();
                if let Some(byte) = tx_queue.peek().copied() {
                    match serial.write(byte) {
                        Ok(()) => {
                            tx_queue.dequeue();
                        }
                        Err(nb::Error::WouldBlock) => {}
                        Err(nb::Error::Other(_)) => {
                            while tx_queue.dequeue().is_some() {}
                            TX_RECORD_DROPPED.borrow(cs).set(true);
                            serial.unlisten(Event::TxEmpty);
                        }
                    }
                } else {
                    serial.unlisten(Event::TxEmpty);
                }
            }
        }
    });
}

fn dequeue_rx() -> Option<u8> {
    cortex_m::interrupt::free(|cs| RX_QUEUE.borrow(cs).borrow_mut().dequeue())
}

fn take_rx_stream_invalidated() -> bool {
    cortex_m::interrupt::free(|cs| {
        let invalidated = RX_STREAM_INVALIDATED.borrow(cs).replace(false);
        if invalidated {
            let mut queue = RX_QUEUE.borrow(cs).borrow_mut();
            while queue.dequeue().is_some() {}
        }
        invalidated
    })
}

fn try_queue_tx_record(record: &[u8]) -> bool {
    cortex_m::interrupt::free(|cs| {
        let mut serial_slot = SERIAL.borrow(cs).borrow_mut();
        let Some(serial) = serial_slot.as_mut() else {
            return false;
        };
        let mut queue = TX_QUEUE.borrow(cs).borrow_mut();
        if record.len() > queue.capacity() - queue.len() {
            return false;
        }

        for &byte in record {
            if queue.enqueue(byte).is_err() {
                return false;
            }
        }
        serial.listen(Event::TxEmpty);
        true
    })
}

fn queue_tx_record(record: &[u8]) {
    if !try_queue_tx_record(record) {
        cortex_m::interrupt::free(|cs| TX_RECORD_DROPPED.borrow(cs).set(true));
    }
}

fn queue_formatted_tx_record(arguments: core::fmt::Arguments<'_>) {
    let mut record = String::<96>::new();
    if record.write_fmt(arguments).is_err() {
        cortex_m::interrupt::free(|cs| TX_RECORD_DROPPED.borrow(cs).set(true));
        return;
    }
    queue_tx_record(record.as_bytes());
}

fn send_controller_error(error: ControllerError) {
    queue_formatted_tx_record(format_args!("ERR,{}\r\n", error.code()));
}

fn send_controller_event(event: ControllerEvent) {
    queue_formatted_tx_record(format_args!("EVT,{}\r\n", event.code()));
}

fn try_queue_controller_error(error: ControllerError) -> bool {
    let mut record = String::<48>::new();
    record
        .write_fmt(format_args!("\r\nERR,{}\r\n", error.code()))
        .is_ok()
        && try_queue_tx_record(record.as_bytes())
}

fn report_dropped_tx_record() {
    let pending = cortex_m::interrupt::free(|cs| TX_RECORD_DROPPED.borrow(cs).replace(false));
    if pending && !try_queue_controller_error(ControllerError::TransmitRecordDropped) {
        cortex_m::interrupt::free(|cs| TX_RECORD_DROPPED.borrow(cs).set(true));
    }
}

fn send_unsequenced_applied_pwm_report(left: PwmPercent, right: PwmPercent) {
    queue_formatted_tx_record(format_args!("PWM,{},{}\r\n", left.get(), right.get()));
}

fn stop_motors<L1, L2, R1, R2>(
    left_ch1: &mut L1,
    left_ch2: &mut L2,
    right_ch1: &mut R1,
    right_ch2: &mut R2,
) where
    L1: embedded_hal::PwmPin<Duty = u16>,
    L2: embedded_hal::PwmPin<Duty = u16>,
    R1: embedded_hal::PwmPin<Duty = u16>,
    R2: embedded_hal::PwmPin<Duty = u16>,
{
    left_ch1.set_duty(0);
    left_ch2.set_duty(0);
    right_ch1.set_duty(0);
    right_ch2.set_duty(0);
}

fn apply_motor_pwm<T, U>(ch1: &mut T, ch2: &mut U, pwm_percent: PwmPercent, max_duty: u16)
where
    T: embedded_hal::PwmPin<Duty = u16>,
    U: embedded_hal::PwmPin<Duty = u16>,
{
    let value = pwm_percent.get();
    let duty = pwm_duty(pwm_percent, max_duty);
    if value > 0 {
        ch1.set_duty(duty);
        ch2.set_duty(0);
    } else if value < 0 {
        ch1.set_duty(0);
        ch2.set_duty(duty);
    } else {
        ch1.set_duty(0);
        ch2.set_duty(0);
    }
}

// Configure TIM1 as quadrature encoder (left wheel)
fn configure_encoder_tim1(tim1: pac::TIM1) {
    // Enable TIM1 clock
    unsafe {
        (*pac::RCC::ptr())
            .apb2enr
            .modify(|_, w| w.tim1en().set_bit());
    }

    // Configure TIM1 in encoder mode
    unsafe {
        // Reset timer
        tim1.cr1.modify(|_, w| w.cen().clear_bit());

        // Configure encoder interface mode 3 (counts on both TI1 and TI2)
        tim1.smcr.modify(|_, w| w.sms().bits(0b011));

        // Configure input capture channels
        tim1.ccmr1_input().modify(|_, w| {
            w.cc1s()
                .bits(0b01) // IC1 mapped to TI1
                .ic1f()
                .bits(0b0011) // Input filter (noise reduction)
                .cc2s()
                .bits(0b01) // IC2 mapped to TI2
                .ic2f()
                .bits(0b0011) // Input filter
        });

        // Set polarity (non-inverted)
        tim1.ccer
            .modify(|_, w| w.cc1p().clear_bit().cc2p().clear_bit());

        // Set auto-reload to max value for 16-bit timer
        tim1.arr.write(|w| w.arr().bits(0xFFFF));

        // Enable update interrupt for overflow detection
        tim1.dier.modify(|_, w| w.uie().set_bit());

        // Clear update flag
        tim1.sr.modify(|_, w| w.uif().clear_bit());

        // Enable counter
        tim1.cr1.modify(|_, w| w.cen().set_bit());
    }
}

// Configure TIM4 as quadrature encoder (right wheel)
fn configure_encoder_tim4(tim4: pac::TIM4) {
    // Enable TIM4 clock
    unsafe {
        (*pac::RCC::ptr())
            .apb1enr
            .modify(|_, w| w.tim4en().set_bit());
    }

    // Configure TIM4 in encoder mode
    unsafe {
        // Reset timer
        tim4.cr1.modify(|_, w| w.cen().clear_bit());

        // Configure encoder interface mode 3
        tim4.smcr.modify(|_, w| w.sms().bits(0b011));

        // Configure input capture channels
        tim4.ccmr1_input().modify(|_, w| {
            w.cc1s()
                .bits(0b01)
                .ic1f()
                .bits(0b0011)
                .cc2s()
                .bits(0b01)
                .ic2f()
                .bits(0b0011)
        });

        // Set polarity
        tim4.ccer
            .modify(|_, w| w.cc1p().clear_bit().cc2p().clear_bit());

        // Set auto-reload to max value
        tim4.arr.write(|w| w.arr().bits(0xFFFF));

        // Enable update interrupt
        tim4.dier.modify(|_, w| w.uie().set_bit());

        // Clear update flag
        tim4.sr.modify(|_, w| w.uif().clear_bit());

        // Enable counter
        tim4.cr1.modify(|_, w| w.cen().set_bit());
    }
}

fn sample_odometry(
    controller_uptime_ms_wrapping: ControllerUptimeMsWrapping,
    previous_left_count: &mut u16,
    previous_right_count: &mut u16,
) -> RobotOdometry {
    let (left_count, right_count, left_overflows, right_overflows) =
        cortex_m::interrupt::free(|cs| unsafe {
            let left_timer = &*pac::TIM1::ptr();
            let right_timer = &*pac::TIM4::ptr();
            let left_count = left_timer.cnt.read().cnt().bits();
            let right_count = right_timer.cnt.read().cnt().bits();

            // An update can become pending immediately before interrupts are
            // masked for this snapshot. Account for that single pending wrap;
            // the ISR will commit it to the persistent counter after exit.
            let left_overflows = encoder_wraps_with_pending_direction_assumption(
                LEFT_ENCODER_WRAPS.borrow(cs).get(),
                left_timer.sr.read().uif().bit_is_set(),
                left_timer.cr1.read().dir().bit_is_set(),
            );
            let right_overflows = encoder_wraps_with_pending_direction_assumption(
                RIGHT_ENCODER_WRAPS.borrow(cs).get(),
                right_timer.sr.read().uif().bit_is_set(),
                right_timer.cr1.read().dir().bit_is_set(),
            );
            (left_count, right_count, left_overflows, right_overflows)
        });

    let odometry = RobotOdometry::new(
        EstimatedWrappingEncoderTicks::from_extended_16_bit_counter(left_overflows, left_count),
        EstimatedWrappingEncoderTicks::from_extended_16_bit_counter(right_overflows, right_count),
        ModuloEncoderDeltaTicks::from_wrapping_counts(*previous_left_count, left_count),
        ModuloEncoderDeltaTicks::from_wrapping_counts(*previous_right_count, right_count),
        controller_uptime_ms_wrapping,
    );

    *previous_left_count = left_count;
    *previous_right_count = right_count;
    odometry
}

fn send_odometry_report(odometry: RobotOdometry) {
    queue_formatted_tx_record(format_args!(
        "ODO,{},{},{},{},{}\r\n",
        odometry.left_estimated_extended_ticks_wrapping_i64().get(),
        odometry.right_estimated_extended_ticks_wrapping_i64().get(),
        odometry.left_sample_delta_ticks_modulo_i16().get(),
        odometry.right_sample_delta_ticks_modulo_i16().get(),
        odometry.controller_uptime_ms_wrapping().get(),
    ));
}

// Interrupt imports
use stm32f4xx_hal::pac::interrupt;
