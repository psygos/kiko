#![no_std]
#![no_main]

use cortex_m_rt::entry;
use heapless::String;
use panic_halt as _;
use stm32f4xx_hal::{
    pac,
    prelude::*,
    serial::{config::Config, Event, Serial},
    timer::{pwm::PwmExt, Channel1, Channel2},
};

// Ring buffer for incoming UART data
const RX_BUFFER_SIZE: usize = 128;
static mut RX_BUFFER: [u8; RX_BUFFER_SIZE] = [0; RX_BUFFER_SIZE];
static mut RX_HEAD: usize = 0;
static mut RX_TAIL: usize = 0;

// Ring buffer for outgoing UART data
const TX_BUFFER_SIZE: usize = 128;
static mut TX_BUFFER: [u8; TX_BUFFER_SIZE] = [0; TX_BUFFER_SIZE];
static mut TX_HEAD: usize = 0;
static mut TX_TAIL: usize = 0;

// Global serial handle for ISR
static mut SERIAL: Option<stm32f4xx_hal::serial::Serial<pac::USART2>> = None;

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
        let gpioa = dp.GPIOA.split();
        let gpiob = dp.GPIOB.split();

        // Status LED
        let mut led = gpioa.pa5.into_push_pull_output();

        // Configure UART with interrupts
        let tx = gpioa.pa2.into_alternate::<7>();
        let rx = gpioa.pa3.into_alternate::<7>();

        let mut serial = Serial::new(
            dp.USART2,
            (tx, rx),
            Config::default()
                .baudrate(115200.bps())
                .wordlength_8()
                .parity_none(),
            &clocks,
        )
        .unwrap();

        // Enable RXNE interrupt
        serial.listen(Event::RxNotEmpty);

        // Store serial in global for ISR access
        unsafe {
            SERIAL = Some(serial);
        }

        // Enable USART2 interrupt in NVIC
        unsafe {
            cortex_m::peripheral::NVIC::unmask(pac::Interrupt::USART2);
        }

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

        let max_duty = left_ch1.get_max_duty();

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

        // Send startup message
        queue_tx_string(b"STM32 Motor Controller Ready\r\n");

        // Control state
        let mut left_speed: i8 = 0;
        let mut right_speed: i8 = 0;
        let mut timeout_counter: u32 = 0;
        let mut telemetry_counter: u32 = 0;

        // Parser state machine
        enum ParserState {
            Idle,
            GotC,
            GotCM,
            ReadingCommand,
            ReadingLeft,
            ReadingRight,
            ReadingTimeout,
        }

        let mut parser_state = ParserState::Idle;
        let mut parse_buffer = String::<32>::new();

        loop {
            // Process received bytes from ring buffer
            while let Some(byte) = dequeue_rx() {
                let mut reset_parser = false;

                match parser_state {
                    ParserState::Idle => {
                        if byte == b'C' {
                            parser_state = ParserState::GotC;
                            parse_buffer.clear();
                        }
                    }
                    ParserState::GotC => {
                        if byte == b'M' {
                            parser_state = ParserState::GotCM;
                        } else {
                            reset_parser = true;
                        }
                    }
                    ParserState::GotCM => {
                        if byte == b'D' {
                            parser_state = ParserState::ReadingCommand;
                        } else {
                            reset_parser = true;
                        }
                    }
                    ParserState::ReadingCommand => {
                        if byte == b',' {
                            parser_state = ParserState::ReadingLeft;
                            parse_buffer.clear();
                        } else {
                            reset_parser = true;
                        }
                    }
                    ParserState::ReadingLeft => {
                        if byte == b',' {
                            if let Ok(speed) = parse_buffer.parse::<i8>() {
                                left_speed = speed.clamp(-100, 100);
                                parser_state = ParserState::ReadingRight;
                                parse_buffer.clear();
                            } else {
                                reset_parser = true;
                            }
                        } else if byte.is_ascii_digit() || byte == b'-' {
                            let _ = parse_buffer.push(byte as char);
                        } else {
                            reset_parser = true;
                        }
                    }
                    ParserState::ReadingRight => {
                        if byte == b',' {
                            if let Ok(speed) = parse_buffer.parse::<i8>() {
                                right_speed = speed.clamp(-100, 100);
                                parser_state = ParserState::ReadingTimeout;
                                parse_buffer.clear();
                            } else {
                                reset_parser = true;
                            }
                        } else if byte.is_ascii_digit() || byte == b'-' {
                            let _ = parse_buffer.push(byte as char);
                        } else {
                            reset_parser = true;
                        }
                    }
                    ParserState::ReadingTimeout => {
                        if byte == b'\n' || byte == b'\r' {
                            if let Ok(timeout) = parse_buffer.parse::<u32>() {
                                // Command complete! Apply motor speeds
                                timeout_counter = timeout.saturating_mul(200); // ~200Hz loop

                                apply_motor_speed(
                                    &mut left_ch1,
                                    &mut left_ch2,
                                    left_speed,
                                    max_duty,
                                );
                                apply_motor_speed(
                                    &mut right_ch1,
                                    &mut right_ch2,
                                    right_speed,
                                    max_duty,
                                );

                                led.toggle();

                                // Send immediate ACK telemetry
                                send_telemetry(left_speed, right_speed);
                            }
                            reset_parser = true;
                        } else if byte.is_ascii_digit() {
                            let _ = parse_buffer.push(byte as char);
                        } else {
                            reset_parser = true;
                        }
                    }
                }

                if reset_parser {
                    parser_state = ParserState::Idle;
                    parse_buffer.clear();
                }
            }

            // Safety timeout
            if timeout_counter > 0 {
                timeout_counter = timeout_counter.saturating_sub(1);
            } else if left_speed != 0 || right_speed != 0 {
                // Timeout expired, stop motors
                left_speed = 0;
                right_speed = 0;
                left_ch1.set_duty(0);
                left_ch2.set_duty(0);
                right_ch1.set_duty(0);
                right_ch2.set_duty(0);
            }

            // Periodic telemetry (non-blocking)
            telemetry_counter += 1;
            if telemetry_counter >= 1000 {
                telemetry_counter = 0;
                send_telemetry(left_speed, right_speed);
            }

            // Service TX buffer
            service_tx();

            // Small delay for ~200Hz loop rate
            delay.delay_us(5000_u32);
        }
    }

    loop {
        cortex_m::asm::nop();
    }
}

// USART2 interrupt handler
#[interrupt]
fn USART2() {
    unsafe {
        let serial_ptr = &raw mut SERIAL;
        if let Some(serial) = &mut *serial_ptr {
            // Check for RX data
            if serial.is_rx_not_empty() {
                if let Ok(byte) = serial.read() {
                    // Add to ring buffer
                    let next_head = (RX_HEAD + 1) % RX_BUFFER_SIZE;
                    if next_head != RX_TAIL {
                        RX_BUFFER[RX_HEAD] = byte;
                        RX_HEAD = next_head;
                    }
                    // Silently drop on buffer full (could set overflow flag)
                }
            }

            // Check if we can transmit
            if serial.is_tx_empty() && TX_HEAD != TX_TAIL {
                let byte = TX_BUFFER[TX_TAIL];
                TX_TAIL = (TX_TAIL + 1) % TX_BUFFER_SIZE;
                let _ = serial.write(byte);
            }
        }
    }
}

fn dequeue_rx() -> Option<u8> {
    unsafe {
        if RX_HEAD != RX_TAIL {
            let byte = RX_BUFFER[RX_TAIL];
            RX_TAIL = (RX_TAIL + 1) % RX_BUFFER_SIZE;
            Some(byte)
        } else {
            None
        }
    }
}

fn queue_tx(byte: u8) -> bool {
    unsafe {
        let next_head = (TX_HEAD + 1) % TX_BUFFER_SIZE;
        if next_head != TX_TAIL {
            TX_BUFFER[TX_HEAD] = byte;
            TX_HEAD = next_head;
            true
        } else {
            false // Buffer full
        }
    }
}

fn queue_tx_string(s: &[u8]) {
    for &byte in s {
        queue_tx(byte);
    }
}

fn service_tx() {
    unsafe {
        cortex_m::interrupt::free(|_| {
            let serial_ptr = &raw mut SERIAL;
            if let Some(serial) = &mut *serial_ptr {
                if serial.is_tx_empty() && TX_HEAD != TX_TAIL {
                    let byte = TX_BUFFER[TX_TAIL];
                    TX_TAIL = (TX_TAIL + 1) % TX_BUFFER_SIZE;
                    let _ = serial.write(byte);
                }
            }
        });
    }
}

fn send_telemetry(left: i8, right: i8) {
    queue_tx_string(b"TEL,");

    // Simple integer to string conversion
    if left < 0 {
        queue_tx(b'-');
        let pos = -left;
        if pos >= 100 {
            queue_tx(b'1');
            queue_tx(b'0' + (pos % 100 / 10) as u8);
        } else if pos >= 10 {
            queue_tx(b'0' + (pos / 10) as u8);
        }
        queue_tx(b'0' + (pos % 10) as u8);
    } else {
        if left >= 100 {
            queue_tx(b'1');
            queue_tx(b'0' + (left % 100 / 10) as u8);
        } else if left >= 10 {
            queue_tx(b'0' + (left / 10) as u8);
        }
        queue_tx(b'0' + (left % 10) as u8);
    }

    queue_tx(b',');

    if right < 0 {
        queue_tx(b'-');
        let pos = -right;
        if pos >= 100 {
            queue_tx(b'1');
            queue_tx(b'0' + (pos % 100 / 10) as u8);
        } else if pos >= 10 {
            queue_tx(b'0' + (pos / 10) as u8);
        }
        queue_tx(b'0' + (pos % 10) as u8);
    } else {
        if right >= 100 {
            queue_tx(b'1');
            queue_tx(b'0' + (right % 100 / 10) as u8);
        } else if right >= 10 {
            queue_tx(b'0' + (right / 10) as u8);
        }
        queue_tx(b'0' + (right % 10) as u8);
    }

    queue_tx_string(b",0\r\n");
}

fn apply_motor_speed<T, U>(ch1: &mut T, ch2: &mut U, speed: i8, max_duty: u16)
where
    T: embedded_hal::PwmPin<Duty = u16>,
    U: embedded_hal::PwmPin<Duty = u16>,
{
    if speed > 0 {
        let duty = (speed as u32 * max_duty as u32 / 100) as u16;
        ch1.set_duty(duty);
        ch2.set_duty(0);
    } else if speed < 0 {
        let duty = ((-speed) as u32 * max_duty as u32 / 100) as u16;
        ch1.set_duty(0);
        ch2.set_duty(duty);
    } else {
        ch1.set_duty(0);
        ch2.set_duty(0);
    }
}

// Interrupt imports
use stm32f4xx_hal::pac::interrupt;
