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

// Encoder state - volatile for ISR access
static mut LEFT_OVERFLOW_COUNT: i32 = 0;
static mut RIGHT_OVERFLOW_COUNT: i32 = 0;
static mut LAST_LEFT_COUNT: u16 = 0;
static mut LAST_RIGHT_COUNT: u16 = 0;

// Odometry state
struct OdometryData {
    left_ticks: i64,
    right_ticks: i64,
    left_velocity: i16,  // ticks/100ms
    right_velocity: i16, // ticks/100ms
    timestamp: u32,      // milliseconds
}

static mut ODOMETRY: OdometryData = OdometryData {
    left_ticks: 0,
    right_ticks: 0,
    left_velocity: 0,
    right_velocity: 0,
    timestamp: 0,
};

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

        // Configure encoder pins first
        let _pa8_enc = gpioa.pa8.into_alternate::<1>(); // TIM1 CH1
        let _pa9_enc = gpioa.pa9.into_alternate::<1>(); // TIM1 CH2
        let _pb6_enc = gpiob.pb6.into_alternate::<2>(); // TIM4 CH1
        let _pb7_enc = gpiob.pb7.into_alternate::<2>(); // TIM4 CH2

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

        // Enable encoder timer interrupts for overflow handling
        unsafe {
            cortex_m::peripheral::NVIC::unmask(pac::Interrupt::TIM1_UP_TIM10);
            cortex_m::peripheral::NVIC::unmask(pac::Interrupt::TIM4);
        }

        // Send startup message
        queue_tx_string(b"STM32 Motor Controller Ready with Encoders\r\n");

        // Control state
        let mut left_speed: i8 = 0;
        let mut right_speed: i8 = 0;
        let mut timeout_counter: u32 = 0;
        let mut telemetry_counter: u32 = 0;
        let mut odometry_counter: u32 = 0;
        let mut system_time_ms: u32 = 0;

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

            // Update odometry at 100Hz (every 10ms)
            odometry_counter += 1;
            if odometry_counter >= 10 {
                odometry_counter = 0;
                update_odometry(system_time_ms);
            }

            // Periodic telemetry (non-blocking)
            telemetry_counter += 1;
            if telemetry_counter >= 200 {  // 10Hz telemetry
                telemetry_counter = 0;
                send_odometry_telemetry();
            }

            // Service TX buffer
            service_tx();

            // Small delay for ~200Hz loop rate
            delay.delay_us(5000_u32);
            system_time_ms = system_time_ms.wrapping_add(5);
        }
    }

    loop {
        cortex_m::asm::nop();
    }
}

// TIM1 Update interrupt handler (left encoder overflow)
#[interrupt]
fn TIM1_UP_TIM10() {
    unsafe {
        let tim1 = &(*pac::TIM1::ptr());
        
        // Check if update interrupt flag is set
        if tim1.sr.read().uif().bit_is_set() {
            // Clear the flag
            tim1.sr.modify(|_, w| w.uif().clear_bit());
            
            // Check direction
            if tim1.cr1.read().dir().bit_is_set() {
                // Counting down
                LEFT_OVERFLOW_COUNT -= 1;
            } else {
                // Counting up
                LEFT_OVERFLOW_COUNT += 1;
            }
        }
    }
}

// TIM4 interrupt handler (right encoder overflow)
#[interrupt]
fn TIM4() {
    unsafe {
        let tim4 = &(*pac::TIM4::ptr());
        
        if tim4.sr.read().uif().bit_is_set() {
            tim4.sr.modify(|_, w| w.uif().clear_bit());
            
            if tim4.cr1.read().dir().bit_is_set() {
                RIGHT_OVERFLOW_COUNT -= 1;
            } else {
                RIGHT_OVERFLOW_COUNT += 1;
            }
        }
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

// Configure TIM1 as quadrature encoder (left wheel)
fn configure_encoder_tim1(tim1: pac::TIM1) {
    // Enable TIM1 clock
    unsafe {
        (*pac::RCC::ptr()).apb2enr.modify(|_, w| w.tim1en().set_bit());
    }

    // Configure TIM1 in encoder mode
    unsafe {
        // Reset timer
        tim1.cr1.modify(|_, w| w.cen().clear_bit());
        
        // Configure encoder interface mode 3 (counts on both TI1 and TI2)
        tim1.smcr.modify(|_, w| w.sms().bits(0b011));
        
        // Configure input capture channels
        tim1.ccmr1_input().modify(|_, w| {
            w.cc1s().bits(0b01)  // IC1 mapped to TI1
             .ic1f().bits(0b0011) // Input filter (noise reduction)
             .cc2s().bits(0b01)  // IC2 mapped to TI2
             .ic2f().bits(0b0011) // Input filter
        });
        
        // Set polarity (non-inverted)
        tim1.ccer.modify(|_, w| {
            w.cc1p().clear_bit()
             .cc2p().clear_bit()
        });
        
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
        (*pac::RCC::ptr()).apb1enr.modify(|_, w| w.tim4en().set_bit());
    }

    // Configure TIM4 in encoder mode
    unsafe {
        // Reset timer
        tim4.cr1.modify(|_, w| w.cen().clear_bit());
        
        // Configure encoder interface mode 3
        tim4.smcr.modify(|_, w| w.sms().bits(0b011));
        
        // Configure input capture channels
        tim4.ccmr1_input().modify(|_, w| {
            w.cc1s().bits(0b01)
             .ic1f().bits(0b0011)
             .cc2s().bits(0b01)
             .ic2f().bits(0b0011)
        });
        
        // Set polarity
        tim4.ccer.modify(|_, w| {
            w.cc1p().clear_bit()
             .cc2p().clear_bit()
        });
        
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

// Update odometry data (called at 100Hz)
fn update_odometry(timestamp_ms: u32) {
    unsafe {
        cortex_m::interrupt::free(|_| {
            // Read current encoder counts
            let left_count = (*pac::TIM1::ptr()).cnt.read().cnt().bits();
            let right_count = (*pac::TIM4::ptr()).cnt.read().cnt().bits();
            
            // Calculate delta ticks (handle wraparound)
            let left_delta = left_count.wrapping_sub(LAST_LEFT_COUNT) as i16;
            let right_delta = right_count.wrapping_sub(LAST_RIGHT_COUNT) as i16;
            
            // Update total position
            let left_total = (LEFT_OVERFLOW_COUNT as i64) * 65536 + left_count as i64;
            let right_total = (RIGHT_OVERFLOW_COUNT as i64) * 65536 + right_count as i64;
            
            ODOMETRY.left_ticks = left_total;
            ODOMETRY.right_ticks = right_total;
            ODOMETRY.left_velocity = left_delta;
            ODOMETRY.right_velocity = right_delta;
            ODOMETRY.timestamp = timestamp_ms;
            
            // Debug: Send raw encoder counts every 50 updates (0.5 seconds)
            static mut DEBUG_COUNTER: u32 = 0;
            DEBUG_COUNTER += 1;
            if DEBUG_COUNTER >= 50 {
                DEBUG_COUNTER = 0;
                // Send debug info outside interrupt context
                queue_tx_string(b"DBG,");
                send_u32(left_count as u32);
                queue_tx(b',');
                send_u32(right_count as u32);
                queue_tx_string(b"\r\n");
            }
            
            // Store for next iteration
            LAST_LEFT_COUNT = left_count;
            LAST_RIGHT_COUNT = right_count;
        });
    }
}

// Send odometry telemetry
fn send_odometry_telemetry() {
    unsafe {
        let (left_ticks, right_ticks, left_vel, right_vel, timestamp) = 
            cortex_m::interrupt::free(|_| {
                (ODOMETRY.left_ticks, ODOMETRY.right_ticks, 
                 ODOMETRY.left_velocity, ODOMETRY.right_velocity,
                 ODOMETRY.timestamp)
            });
        
        queue_tx_string(b"ODO,");
        send_i64(left_ticks);
        queue_tx(b',');
        send_i64(right_ticks);
        queue_tx(b',');
        send_i16(left_vel);
        queue_tx(b',');
        send_i16(right_vel);
        queue_tx(b',');
        send_u32(timestamp);
        queue_tx_string(b"\r\n");
    }
}

// Helper functions for number to string conversion
fn send_i64(mut n: i64) {
    if n < 0 {
        queue_tx(b'-');
        n = -n;
    }
    send_u64(n as u64);
}

fn send_u64(mut n: u64) {
    let mut buf = [0u8; 20];
    let mut i = 0;
    
    if n == 0 {
        queue_tx(b'0');
        return;
    }
    
    while n > 0 {
        buf[i] = b'0' + (n % 10) as u8;
        n /= 10;
        i += 1;
    }
    
    while i > 0 {
        i -= 1;
        queue_tx(buf[i]);
    }
}

fn send_i16(n: i16) {
    if n < 0 {
        queue_tx(b'-');
        send_u32((-n) as u32);
    } else {
        send_u32(n as u32);
    }
}

fn send_u32(mut n: u32) {
    let mut buf = [0u8; 10];
    let mut i = 0;
    
    if n == 0 {
        queue_tx(b'0');
        return;
    }
    
    while n > 0 {
        buf[i] = b'0' + (n % 10) as u8;
        n /= 10;
        i += 1;
    }
    
    while i > 0 {
        i -= 1;
        queue_tx(buf[i]);
    }
}

// Interrupt imports
use stm32f4xx_hal::pac::interrupt;
