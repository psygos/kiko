#![no_std]
#![no_main]
#![forbid(unsafe_code)]

use core::cell::RefCell;

use embassy_executor::Spawner;
use embassy_futures::join::join3;
use embassy_rp::bind_interrupts;
use embassy_rp::peripherals::{DMA_CH0, DMA_CH1, PIO0, TRNG, USB};
use embassy_rp::pio::{InterruptHandler as PioInterruptHandler, Pio};
use embassy_rp::pio_programs::ws2812::{PioWs2812, PioWs2812Program};
use embassy_rp::trng::Trng;
use embassy_rp::usb::{Driver as UsbDriver, InterruptHandler as UsbInterruptHandler};
use embassy_sync::blocking_mutex::Mutex;
use embassy_sync::blocking_mutex::raw::CriticalSectionRawMutex;
use embassy_time::{Duration, Instant, Ticker};
use embassy_usb::class::cdc_acm::{CdcAcmClass, State as CdcState};
use embassy_usb::driver::EndpointError;
use embassy_usb::{Builder, Config as UsbConfig};
use kiko_eye_protocol::DeviceTimestampMs;
use kiko_eyes_kep2_firmware::{
    EndpointEvent, EyeFrame, EyeRenderer, FRAME_RATE_HZ, FallbackCause, FirmwareIdentity,
    Kep2Endpoint, LEDS_PER_EYE, MountingSign, OutputState,
};
use panic_halt as _;
use smart_leds::RGB8;

include!(concat!(env!("OUT_DIR"), "/provisioning.rs"));

bind_interrupts!(struct Irqs {
    PIO0_IRQ_0 => PioInterruptHandler<PIO0>;
    DMA_IRQ_0 => embassy_rp::dma::InterruptHandler<DMA_CH0>, embassy_rp::dma::InterruptHandler<DMA_CH1>;
    USBCTRL_IRQ => UsbInterruptHandler<USB>;
    TRNG_IRQ => embassy_rp::trng::InterruptHandler<TRNG>;
});

static ENDPOINT: Mutex<CriticalSectionRawMutex, RefCell<Option<Kep2Endpoint>>> =
    Mutex::new(RefCell::new(None));

type EyeBank = [EyeFrame; 2];

const USB_SERIAL_HEX_BYTES: usize = 32;
const USB_SERIAL_DESCRIPTOR_BYTES: usize = 2 + USB_SERIAL_HEX_BYTES * 2;
const USB_CONTROL_BUFFER_BYTES: usize = 128;

// embassy-usb 0.6 requires one byte of slack beyond the complete UTF-16LE
// string descriptor. Without this proof, the 32-character OTP serial panics
// the device while the host enumerates it and CDC never becomes available.
const _: () = assert!(USB_CONTROL_BUFFER_BYTES > USB_SERIAL_DESCRIPTOR_BYTES);

fn device_now() -> DeviceTimestampMs {
    DeviceTimestampMs::from_millis_since_boot(Instant::now().as_millis())
}

fn uid_hex(uid: [u8; 16]) -> [u8; USB_SERIAL_HEX_BYTES] {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = [0_u8; USB_SERIAL_HEX_BYTES];
    let mut index = 0;
    while index < uid.len() {
        output[index * 2] = HEX[usize::from(uid[index] >> 4)];
        output[index * 2 + 1] = HEX[usize::from(uid[index] & 0x0f)];
        index += 1;
    }
    output
}

fn endpoint_output(now: DeviceTimestampMs) -> OutputState {
    ENDPOINT.lock(|slot| {
        let mut slot = slot.borrow_mut();
        let Some(endpoint) = slot.as_mut() else {
            return OutputState::Autonomous {
                cause: FallbackCause::InternalFault,
            };
        };
        endpoint
            .controller_mut()
            .output_at(now)
            .unwrap_or(OutputState::Autonomous {
                cause: FallbackCause::InternalFault,
            })
    })
}

fn latch_render_fault(now: DeviceTimestampMs) {
    ENDPOINT.lock(|slot| {
        let mut slot = slot.borrow_mut();
        let disable_endpoint = slot
            .as_mut()
            .is_some_and(|endpoint| endpoint.controller_mut().on_internal_fault(now).is_err());
        if disable_endpoint {
            // A clock regression is already latched internally. Removing the
            // endpoint also prevents any subsequent USB claim on that clock.
            *slot = None;
        }
    });
}

fn render_bank(left: &mut EyeRenderer, right: &mut EyeRenderer, bank: &mut EyeBank) {
    let now = device_now();
    let output = endpoint_output(now);
    let (left_frame, right_frame) = bank.split_at_mut(1);
    if left.render(now, output, &mut left_frame[0]).is_err()
        || right.render(now, output, &mut right_frame[0]).is_err()
    {
        left_frame[0].fill(RGB8::default());
        right_frame[0].fill(RGB8::default());
        latch_render_fault(now);
    }
}

#[embassy_executor::task]
async fn usb_task(driver: UsbDriver<'static, USB>, serial_ascii: [u8; USB_SERIAL_HEX_BYTES]) {
    let mut config = UsbConfig::new(0xc0de, 0xcafe);
    config.manufacturer = Some("kiko");
    config.product = Some("kiko-eyes-kep2");
    config.serial_number = core::str::from_utf8(&serial_ascii).ok();
    config.max_power = 100;

    let mut config_descriptor = [0_u8; 256];
    let mut bos_descriptor = [0_u8; 256];
    let mut msos_descriptor = [0_u8; 256];
    let mut control_buffer = [0_u8; USB_CONTROL_BUFFER_BYTES];
    let mut cdc_state = CdcState::new();
    let mut builder = Builder::new(
        driver,
        config,
        &mut config_descriptor,
        &mut bos_descriptor,
        &mut msos_descriptor,
        &mut control_buffer,
    );
    let mut class = CdcAcmClass::new(&mut builder, &mut cdc_state, 64);
    let mut usb = builder.build();

    let communication = async {
        let mut packet = [0_u8; 64];
        loop {
            class.wait_connection().await;
            'connection: loop {
                let count = match class.read_packet(&mut packet).await {
                    Ok(count) => count,
                    Err(EndpointError::Disabled | EndpointError::BufferOverflow) => {
                        break 'connection;
                    }
                };
                let packet_received_at = device_now();
                for byte in &packet[..count] {
                    let handled_at = device_now();
                    let event = ENDPOINT.lock(|slot| {
                        let mut slot = slot.borrow_mut();
                        match slot.as_mut() {
                            Some(endpoint) => endpoint.push(*byte, packet_received_at, handled_at),
                            None => EndpointEvent::Pending,
                        }
                    });
                    if let EndpointEvent::Response(response) = event {
                        for chunk in response.as_bytes().chunks(64) {
                            if class.write_packet(chunk).await.is_err() {
                                break 'connection;
                            }
                        }
                    }
                }
            }
            let now = device_now();
            ENDPOINT.lock(|slot| {
                let mut slot = slot.borrow_mut();
                let disable_endpoint = slot
                    .as_mut()
                    .is_some_and(|endpoint| endpoint.on_disconnect(now).is_err());
                if disable_endpoint {
                    *slot = None;
                }
            });
        }
    };
    embassy_futures::join::join(usb.run(), communication).await;
}

#[embassy_executor::main]
async fn main(spawner: Spawner) {
    let peripherals = embassy_rp::init(Default::default());
    let mut trng = Trng::new(peripherals.TRNG, Irqs, embassy_rp::trng::Config::default());
    // TRNG provides a per-boot value; setting the low bit makes zero
    // unrepresentable without substituting a deterministic fallback.
    let boot_id = trng.blocking_next_u64() | 1;

    if let Ok(otp_uid) = embassy_rp::otp::get_private_random_number() {
        let uid = otp_uid.to_be_bytes();
        if let Ok(identity) = FirmwareIdentity::try_new(uid, FIRMWARE_BUILD_ID_BYTES, boot_id) {
            ENDPOINT.lock(|slot| *slot.borrow_mut() = Some(Kep2Endpoint::new(identity)));
            match usb_task(UsbDriver::new(peripherals.USB, Irqs), uid_hex(uid)) {
                Ok(task) => spawner.spawn(task),
                Err(_) => latch_render_fault(device_now()),
            }
        }
    }

    let Pio {
        mut common,
        sm0,
        sm1,
        ..
    } = Pio::new(peripherals.PIO0, Irqs);
    let program = PioWs2812Program::new(&mut common);
    let mut left_panel = PioWs2812::new(
        &mut common,
        sm0,
        peripherals.DMA_CH0,
        Irqs,
        peripherals.PIN_15,
        &program,
    );
    let mut right_panel = PioWs2812::new(
        &mut common,
        sm1,
        peripherals.DMA_CH1,
        Irqs,
        peripherals.PIN_16,
        &program,
    );

    let mut left_renderer = EyeRenderer::new(MountingSign::SameDirection);
    let mut right_renderer = EyeRenderer::new(RIGHT_EYE_MOUNTING);
    let black = RGB8::default();
    let mut banks: [EyeBank; 2] = [[[black; LEDS_PER_EYE]; 2]; 2];
    render_bank(&mut left_renderer, &mut right_renderer, &mut banks[0]);

    let mut ticker = Ticker::every(Duration::from_hz(u64::from(FRAME_RATE_HZ)));
    let mut frame_sequence = 1_u32;
    loop {
        let display_index = ((frame_sequence + 1) & 1) as usize;
        let (low, high) = banks.split_at_mut(1);
        let (display, next) = if display_index == 0 {
            (&low[0], &mut high[0])
        } else {
            (&high[0], &mut low[0])
        };
        join3(
            left_panel.write(&display[0]),
            right_panel.write(&display[1]),
            async { render_bank(&mut left_renderer, &mut right_renderer, next) },
        )
        .await;
        frame_sequence = frame_sequence.wrapping_add(1);
        ticker.next().await;
    }
}
