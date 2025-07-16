#![no_std]
#![no_main]

use panic_halt as _;

use cortex_m_rt::entry;

use stm32f4xx_hal::{
    pac,
    prelude::*,
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

        // PA5 is the onboard led. We will use it to indicate update in pwm
        let mut led = gpioa.pa5.into_push_pull_output();

        // HAL 0.20 uses a *ChannelBuilder* wrapper per channel.  Wrap our
        // PA6 pin in a `Channel1` builder, pass it as a 1-tuple, then obtain
        // the split single-channel handle.
        let channels = (
            Channel1::new(gpioa.pa6.into_alternate()),
            Channel2::new(gpioa.pa7.into_alternate()),
        );
        let (mut pwm_channel, _) = dp.TIM3
            .pwm_hz(channels, 20.kHz(), &clocks)
            .split();

        let max_duty = pwm_channel.get_max_duty();
        pwm_channel.enable();
        pwm_channel.set_duty(0);

        let mut duty: i16 = 0;
        let mut sign: i16 = 1;
        const STEP: i16 = 50;
        const MAX: i16 = 250;

        loop {
            duty += STEP * sign;

            if duty >= MAX {
                duty = MAX;
                sign = -1;
            } else if duty <= 0 {
                duty = 0;
                sign = 1;
            }

            let pwm_value = (duty as u32 * max_duty as u32 / 255) as u16;
            pwm_channel.set_duty(pwm_value);

            led.toggle();
            delay.delay_ms(5000_u32);
        }
    }

    loop {
        cortex_m::asm::nop();
    }
}


