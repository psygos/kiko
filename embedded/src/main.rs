#![no_std]
#![no_main]

use panic_halt as _;

use cortex_m_rt::entry;

enum ParserState {
    WaitingForStart,
    ReadingCommand,
    ReadingLeft,
    ReadingRight,
    ReadingTimeout,
}

use heapless::String;
use nb::block;
use stm32f4xx_hal::{
    pac,
    prelude::*,
    serial::{config::Config, Serial},
    timer::{pwm::PwmExt, Channel1, Channel2},
};

#[entry]
fn main() -> ! {
    // dp is device peripherals
    // cp is core peripherals
    // here we are getting access of them
    if let (Some(dp), Some(cp)) = (
        pac::Peripherals::take(),
        cortex_m::peripheral::Peripherals::take(),
    ) {
        // this section is for configuring the clock

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

        // PA5 is the onboard led. We will use it to indicate update in pwm
        let mut led = gpioa.pa5.into_push_pull_output();

        // UART Setup via CN1 VCP: PA2 (USART2_TX) and PA3 (USART2_RX)
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

        // Left motor on TIM2 (PA0, PA1)
        let left_channels = (
            Channel1::new(gpioa.pa0.into_alternate()),
            Channel2::new(gpioa.pa1.into_alternate()),
        );
        let (mut left_ch1, mut left_ch2) = dp.TIM2.pwm_hz(left_channels, 20.kHz(), &clocks).split();

        // Right motor on TIM3 (PB4, PB5)
        let right_channels = (
            Channel1::new(gpiob.pb4.into_alternate()),
            Channel2::new(gpiob.pb5.into_alternate()),
        );
        let (mut right_ch1, mut right_ch2) =
            dp.TIM3.pwm_hz(right_channels, 20.kHz(), &clocks).split();

        let max_duty = left_ch1.get_max_duty();

        left_ch1.enable();
        left_ch2.enable();
        right_ch1.enable();
        right_ch2.enable();

        left_ch1.set_duty(0);
        left_ch2.set_duty(0);
        right_ch1.set_duty(0);
        right_ch2.set_duty(0);

        // Command parsing variables
        let mut parser_state = ParserState::WaitingForStart;
        let mut cmd_buffer = String::<32>::new();
        let mut left_speed: i8 = 0;
        let mut right_speed: i8 = 0;
        let mut timeout_counter: u32 = 0;

        // Send startup message
        for byte in b"STM32 Motor Controller Ready\r\n" {
            block!(serial.write(*byte)).ok();
        }

        // Main control loop
        let mut _loop_counter = 0u32;

        loop {
            // Handle UART reception
            if let Ok(byte) = serial.read() {
                match parser_state {
                    // Ignore carriage return characters
                    _ if byte == b'\r' => {
                        // skip '\r' characters entirely
                    }
                    ParserState::WaitingForStart => {
                        if byte == b'C' {
                            parser_state = ParserState::ReadingCommand;
                            cmd_buffer.clear();
                        }
                    }
                    ParserState::ReadingCommand => {
                        if byte == b',' {
                            if cmd_buffer == "MD" {
                                parser_state = ParserState::ReadingLeft;
                                cmd_buffer.clear();
                            } else {
                                parser_state = ParserState::WaitingForStart;
                            }
                        } else {
                            cmd_buffer.push(byte as char).ok();
                        }
                    }
                    ParserState::ReadingLeft => {
                        if byte == b',' {
                            if let Ok(speed) = cmd_buffer.parse::<i8>() {
                                left_speed = speed.clamp(-100, 100);
                            }
                            parser_state = ParserState::ReadingRight;
                            cmd_buffer.clear();
                        } else {
                            cmd_buffer.push(byte as char).ok();
                        }
                    }
                    ParserState::ReadingRight => {
                        if byte == b',' {
                            if let Ok(speed) = cmd_buffer.parse::<i8>() {
                                right_speed = speed.clamp(-100, 100);
                            }
                            parser_state = ParserState::ReadingTimeout;
                            cmd_buffer.clear();
                        } else {
                            cmd_buffer.push(byte as char).ok();
                        }
                    }
                    ParserState::ReadingTimeout => {
                        if byte == b'\n' {
                            if let Ok(timeout) = cmd_buffer.parse::<u32>() {
                                timeout_counter = timeout * 2; // Convert ms to loop iterations (approx)

                                // Apply motor speeds
                                // Left motor speed control
                                if left_speed > 0 {
                                    // Compute duty in u32 to avoid u16 overflow then cast back
                                    let duty = (left_speed as u32 * max_duty as u32 / 100) as u16;
                                    left_ch1.set_duty(duty);
                                    left_ch2.set_duty(0);
                                } else if left_speed < 0 {
                                    let duty = ((-left_speed) as u32 * max_duty as u32 / 100) as u16;
                                    left_ch1.set_duty(0);
                                    left_ch2.set_duty(duty);
                                } else {
                                    left_ch1.set_duty(0);
                                    left_ch2.set_duty(0);
                                }

                                // Right motor speed control
                                if right_speed > 0 {
                                    let duty = (right_speed as u32 * max_duty as u32 / 100) as u16;
                                    right_ch1.set_duty(duty);
                                    right_ch2.set_duty(0);
                                } else if right_speed < 0 {
                                    let duty = ((-right_speed) as u32 * max_duty as u32 / 100) as u16;
                                    right_ch1.set_duty(0);
                                    right_ch2.set_duty(duty);
                                } else {
                                    right_ch1.set_duty(0);
                                    right_ch2.set_duty(0);
                                }

                                // Toggle LED to show command received
                                led.toggle();

                                // --- Immediate telemetry so host can verify PWM update ---
                                // Send prefix
                                for byte in b"TEL," {
                                    block!(serial.write(*byte)).ok();
                                }

                                // Write left and right speeds as ASCII using heapless String to avoid borrow checker issues
                                use core::fmt::Write as _;
                                let mut buf: heapless::String<8> = heapless::String::new();
                                // guaranteed to fit: "-100,-100" is 8 bytes
                                let _ = write!(buf, "{},{}", left_speed, right_speed);
                                for &b in buf.as_bytes() {
                                    block!(serial.write(b)).ok();
                                }

                                // Placeholder battery value zero and line termination
                                for byte in b",0\r\n" {
                                    block!(serial.write(*byte)).ok();
                                }
                            }
                            parser_state = ParserState::WaitingForStart;
                            cmd_buffer.clear();
                        } else {
                            cmd_buffer.push(byte as char).ok();
                        }
                    }
                }
            }

            // Safety timeout - stop if no commands received
            if timeout_counter > 0 {
                timeout_counter -= 1;
            } else {
                left_speed = 0;
                right_speed = 0;
                // Stop all motors
                left_ch1.set_duty(0);
                left_ch2.set_duty(0);
                right_ch1.set_duty(0);
                right_ch2.set_duty(0);
            }

            // Optional: Simple telemetry (removed for minimal implementation)
            _loop_counter += 1;
            if _loop_counter % 1000 == 0 {
                // Disabled telemetry for minimal implementation
                // Send simple telemetry without format! macro
                // --- Telemetry output ---
                for byte in b"TEL," {
                    block!(serial.write(*byte)).ok();
                }
                // Simple integer to string conversion (basic implementation)
                let left_str = if left_speed >= 0 {
                    if left_speed < 10 {
                        [b'0' + left_speed as u8, 0]
                    } else {
                        [
                            b'0' + (left_speed / 10) as u8,
                            b'0' + (left_speed % 10) as u8,
                        ]
                    }
                } else {
                    let pos = -left_speed;
                    if pos < 10 {
                        [b'-', b'0' + pos as u8]
                    } else {
                        [b'-', b'0' + (pos / 10) as u8]
                    }
                };
                for &byte in &left_str {
                    if byte != 0 {
                        block!(serial.write(byte)).ok();
                    }
                }
                block!(serial.write(b',')).ok();
                let right_str = if right_speed >= 0 {
                    if right_speed < 10 {
                        [b'0' + right_speed as u8, 0]
                    } else {
                        [
                            b'0' + (right_speed / 10) as u8,
                            b'0' + (right_speed % 10) as u8,
                        ]
                    }
                } else {
                    let pos = -right_speed;
                    if pos < 10 {
                        [b'-', b'0' + pos as u8]
                    } else {
                        [b'-', b'0' + (pos / 10) as u8]
                    }
                };
                for &byte in &right_str {
                    if byte != 0 {
                        block!(serial.write(byte)).ok();
                    }
                }

                // Send a placeholder battery value of 0 and terminate with CR LF
                for byte in b",0\r\n" {
                    block!(serial.write(*byte)).ok();
                }
            }

            // Small delay for loop timing
            delay.delay_us(500_u32);
        }
    }

    loop {
        cortex_m::asm::nop();
    }
}
