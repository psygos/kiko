//! Total, allocation-free renderer for the measured circular eye panels.

use kiko_eye_protocol::{DeviceTimestampMs, Expression, EyeIntent, IntentSequence};
use smart_leds::RGB8;

use crate::controller::OutputState;
use crate::geometry::{EYE_POSITIONS, Position, UNITS_PER_MM};

pub const LEDS_PER_EYE: usize = 56;
pub const FRAME_RATE_HZ: u32 = 60;
pub const BRIGHTNESS_CEILING: u8 = 56;
/// Duration for which a freshly booted application owns the panels before
/// exposing KEP2 USB control.
pub const MATRIX_BOOT_DURATION_MS: u64 = 2_400;

const GAZE_RANGE: i32 = 18 * UNITS_PER_MM;
const GAZE_SLEW_UNITS_PER_SECOND: u32 = 400 * UNITS_PER_MM as u32;
const LID_SLEW_PER_SECOND: u32 = 10_000;
const LID_TOP: i32 = 40 * UNITS_PER_MM;
const PUPIL_MIN_RADIUS: i32 = 7 * UNITS_PER_MM;
const PUPIL_RADIUS_SPAN: i32 = 8 * UNITS_PER_MM;
const PUPIL_HALO: i32 = 4 * UNITS_PER_MM;
const BLINK_HALF_MS: u64 = 100;
const BLINK_TOTAL_MS: u64 = 2 * BLINK_HALF_MS;
const AUTONOMOUS_GAZE_PERIOD_MS: u64 = 8_000;
const AUTONOMOUS_BREATHE_PERIOD_MS: u64 = 4_000;
const KIKO_GREEN: [u8; 3] = [0xd4, 0xff, 0xa2];
const PUPIL_COLOR: [u8; 3] = [0x04, 0x10, 0x02];
const MATRIX_HEAD_HALF_HEIGHT: i32 = 52;
const MATRIX_TAIL_HEIGHT: i32 = 410;
const MATRIX_VERTICAL_SPAN: u64 = 1_500;
const MATRIX_TOP: i32 = -750;

pub type EyeFrame = [RGB8; LEDS_PER_EYE];

/// Fixed phase separation for the two physical panels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatrixPanel {
    Left,
    Right,
}

impl MatrixPanel {
    const fn phase_ms(self) -> u64 {
        match self {
            Self::Left => 0,
            Self::Right => 173,
        }
    }
}

/// Render the boot/update Matrix cue without touching KEP2 controller state.
///
/// The RP2350 application calls this before USB enumeration, so no accepted
/// KEP2 command can be visually ignored by the cue. `elapsed_ms` is relative
/// to this boot animation, not the device/host protocol clock.
pub fn render_matrix_boot(elapsed_ms: u64, panel: MatrixPanel, frame: &mut EyeFrame) {
    let time = elapsed_ms.saturating_add(panel.phase_ms());
    for (index, (pixel, position)) in frame.iter_mut().zip(EYE_POSITIONS).enumerate() {
        let index = u64::try_from(index).expect("the fixed 56-pixel index fits u64");
        let column = matrix_column(position.x);
        let phase = time
            .wrapping_mul(3)
            .wrapping_add(u64::from(column) * 211)
            .wrapping_add(index * 7);
        let head_y = MATRIX_TOP
            + i32::try_from(phase % MATRIX_VERTICAL_SPAN)
                .expect("the 1,500-unit Matrix span fits i32");
        let behind = head_y - position.y;
        let glyph_on = ((time / 92)
            .wrapping_add(u64::from(column) * 5)
            .wrapping_add(index * 3)
            & 0x3)
            != 0;
        let logical = if behind.abs() <= MATRIX_HEAD_HALF_HEIGHT {
            [175, 255, 175]
        } else if (1..=MATRIX_TAIL_HEIGHT).contains(&behind) && glyph_on {
            let green = 220_i32 - behind * 150 / MATRIX_TAIL_HEIGHT;
            [
                0,
                u8::try_from(green.clamp(0, 255)).expect("the clamped Matrix channel fits u8"),
                18,
            ]
        } else {
            [0, 5, 0]
        };
        *pixel = RGB8 {
            r: power_limited_channel(logical[0], 760),
            g: power_limited_channel(logical[1], 760),
            b: power_limited_channel(logical[2], 760),
        };
    }
}

const fn matrix_column(x: i32) -> u8 {
    if x < -416 {
        0
    } else if x < -208 {
        1
    } else if x < 0 {
        2
    } else if x < 208 {
        3
    } else if x < 416 {
        4
    } else {
        5
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MountingSign {
    SameDirection,
    Mirrored,
}

impl MountingSign {
    pub const fn try_from_i8(value: i8) -> Result<Self, MountingSignError> {
        match value {
            1 => Ok(Self::SameDirection),
            -1 => Ok(Self::Mirrored),
            _ => Err(MountingSignError { value }),
        }
    }

    const fn multiplier(self) -> i32 {
        match self {
            Self::SameDirection => 1,
            Self::Mirrored => -1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MountingSignError {
    pub value: i8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderError {
    ClockRegressed { previous_ms: u64, actual_ms: u64 },
}

#[derive(Clone, Copy)]
struct Parameters {
    gaze_x: i16,
    gaze_y: i16,
    lid: u16,
    pupil: u16,
    brightness: u16,
    color: [u8; 3],
    expression: Expression,
}

/// Stateful per-panel renderer with time-based, unit-labelled slew rates.
pub struct EyeRenderer {
    mounting: MountingSign,
    current_gaze: Position,
    current_lid: i32,
    last_render_at: Option<DeviceTimestampMs>,
    last_command_sequence: Option<IntentSequence>,
    blink_started_at: Option<DeviceTimestampMs>,
}

impl EyeRenderer {
    pub const fn new(mounting: MountingSign) -> Self {
        Self {
            mounting,
            current_gaze: Position::new(0, 0),
            current_lid: 0,
            last_render_at: None,
            last_command_sequence: None,
            blink_started_at: None,
        }
    }

    pub fn render(
        &mut self,
        now: DeviceTimestampMs,
        output: OutputState,
        frame: &mut EyeFrame,
    ) -> Result<(), RenderError> {
        let elapsed_ms = match self.last_render_at {
            Some(previous) => {
                now.checked_millis_since(previous)
                    .ok_or(RenderError::ClockRegressed {
                        previous_ms: previous.millis_since_boot(),
                        actual_ms: now.millis_since_boot(),
                    })?
            }
            None => 0,
        };
        self.last_render_at = Some(now);

        let parameters = self.parameters(now, output);
        let target_gaze = Position::new(
            normalized_signed_to_fixed(parameters.gaze_x) * self.mounting.multiplier(),
            normalized_signed_to_fixed(parameters.gaze_y),
        );
        let gaze_step = step_for_rate(GAZE_SLEW_UNITS_PER_SECOND, elapsed_ms);
        self.current_gaze.x = move_toward(self.current_gaze.x, target_gaze.x, gaze_step);
        self.current_gaze.y = move_toward(self.current_gaze.y, target_gaze.y, gaze_step);

        let biased_lid = expression_lid(parameters.lid, parameters.expression);
        let lid_step = step_for_rate(LID_SLEW_PER_SECOND, elapsed_ms);
        self.current_lid = move_toward(self.current_lid, i32::from(biased_lid), lid_step);
        let blink_lid = self.blink_closure(now);
        let closure = self.current_lid.max(i32::from(blink_lid)).clamp(0, 1_000);
        let lid_line = LID_TOP - closure * (2 * LID_TOP) / 1_000;

        let radius = pupil_radius(parameters.pupil);
        let radius_squared = i64::from(radius) * i64::from(radius);
        let halo = radius + PUPIL_HALO;
        let halo_squared = i64::from(halo) * i64::from(halo);
        let iris = parameters.color;

        for (pixel, position) in frame.iter_mut().zip(EYE_POSITIONS) {
            let logical = if position.y > lid_line {
                [0, 0, 0]
            } else {
                let distance_squared = position.squared_distance(self.current_gaze);
                if distance_squared <= radius_squared {
                    PUPIL_COLOR
                } else if distance_squared <= halo_squared {
                    blend_half(iris, PUPIL_COLOR)
                } else {
                    iris
                }
            };
            *pixel = RGB8 {
                r: power_limited_channel(logical[0], parameters.brightness),
                g: power_limited_channel(logical[1], parameters.brightness),
                b: power_limited_channel(logical[2], parameters.brightness),
            };
        }
        Ok(())
    }

    fn parameters(&mut self, now: DeviceTimestampMs, output: OutputState) -> Parameters {
        match output {
            OutputState::Commanded { intent, sequence } => {
                if self.last_command_sequence != Some(sequence) {
                    if intent.flags().requests_blink() {
                        self.blink_started_at = Some(now);
                    }
                    self.last_command_sequence = Some(sequence);
                }
                parameters_from_intent(intent)
            }
            OutputState::Autonomous { .. } => {
                self.last_command_sequence = None;
                self.blink_started_at = None;
                autonomous_parameters(now)
            }
        }
    }

    fn blink_closure(&mut self, now: DeviceTimestampMs) -> u16 {
        let Some(started_at) = self.blink_started_at else {
            return 0;
        };
        let Some(elapsed) = now.checked_millis_since(started_at) else {
            self.blink_started_at = None;
            return 0;
        };
        if elapsed >= BLINK_TOTAL_MS {
            self.blink_started_at = None;
            return 0;
        }
        let linear = if elapsed <= BLINK_HALF_MS {
            elapsed * 1_000 / BLINK_HALF_MS
        } else {
            (BLINK_TOTAL_MS - elapsed) * 1_000 / BLINK_HALF_MS
        };
        smoothstep(u16::try_from(linear).unwrap_or(1_000))
    }
}

const fn parameters_from_intent(intent: EyeIntent) -> Parameters {
    Parameters {
        gaze_x: intent.gaze_x().get(),
        gaze_y: intent.gaze_y().get(),
        lid: intent.lid().get(),
        pupil: intent.pupil().get(),
        brightness: intent.brightness().get(),
        color: intent.color_rgb(),
        expression: intent.expression(),
    }
}

fn autonomous_parameters(now: DeviceTimestampMs) -> Parameters {
    let millis = now.millis_since_boot();
    let gaze_x = triangle_signed(millis, AUTONOMOUS_GAZE_PERIOD_MS, 300);
    let gaze_y_phase = (millis % AUTONOMOUS_GAZE_PERIOD_MS + AUTONOMOUS_GAZE_PERIOD_MS / 4)
        % AUTONOMOUS_GAZE_PERIOD_MS;
    let gaze_y = triangle_signed(gaze_y_phase, AUTONOMOUS_GAZE_PERIOD_MS, 120);
    let breathe = triangle_unsigned(millis, AUTONOMOUS_BREATHE_PERIOD_MS, 550, 750);
    Parameters {
        gaze_x,
        gaze_y,
        lid: 0,
        pupil: 500,
        brightness: breathe,
        color: KIKO_GREEN,
        expression: Expression::Neutral,
    }
}

fn expression_lid(lid: u16, expression: Expression) -> u16 {
    match expression {
        Expression::Neutral => lid,
        Expression::Curious => lid.saturating_sub(50),
        Expression::Greet => lid.saturating_add(30).min(1_000),
        Expression::Concerned => lid.saturating_add(120).min(1_000),
        Expression::Sleepy => lid.max(400),
    }
}

const fn normalized_signed_to_fixed(value: i16) -> i32 {
    let product = value as i32 * GAZE_RANGE;
    if product >= 0 {
        (product + 500) / 1_000
    } else {
        (product - 500) / 1_000
    }
}

const fn pupil_radius(value: u16) -> i32 {
    PUPIL_MIN_RADIUS + (value as i32 * PUPIL_RADIUS_SPAN + 500) / 1_000
}

fn step_for_rate(rate_per_second: u32, elapsed_ms: u64) -> i32 {
    let step = u64::from(rate_per_second)
        .saturating_mul(elapsed_ms)
        .saturating_add(999)
        / 1_000;
    i32::try_from(step).unwrap_or(i32::MAX)
}

fn move_toward(current: i32, target: i32, maximum_step: i32) -> i32 {
    if current < target {
        current.saturating_add(maximum_step).min(target)
    } else {
        current.saturating_sub(maximum_step).max(target)
    }
}

const fn smoothstep(value: u16) -> u16 {
    let value = value as u64;
    ((value * value * (3_000 - 2 * value) + 500_000) / 1_000_000) as u16
}

fn triangle_signed(millis: u64, period_ms: u64, amplitude: i16) -> i16 {
    let phase = millis % period_ms;
    let half = period_ms / 2;
    let amplitude = i64::from(amplitude);
    let value = if phase <= half {
        -amplitude + (2 * amplitude * phase as i64) / half as i64
    } else {
        amplitude - (2 * amplitude * (phase - half) as i64) / half as i64
    };
    i16::try_from(value).unwrap_or(0)
}

fn triangle_unsigned(millis: u64, period_ms: u64, low: u16, high: u16) -> u16 {
    let phase = millis % period_ms;
    let half = period_ms / 2;
    let span = u64::from(high - low);
    let offset = if phase <= half {
        span * phase / half
    } else {
        span * (period_ms - phase) / half
    };
    low + u16::try_from(offset).unwrap_or(0)
}

const fn blend_half(left: [u8; 3], right: [u8; 3]) -> [u8; 3] {
    [
        (left[0] as u16 + right[0] as u16).div_ceil(2) as u8,
        (left[1] as u16 + right[1] as u16).div_ceil(2) as u8,
        (left[2] as u16 + right[2] as u16).div_ceil(2) as u8,
    ]
}

const GAMMA_2: [u8; 256] = gamma_table();

const fn gamma_table() -> [u8; 256] {
    let mut table = [0_u8; 256];
    let mut value = 0_usize;
    while value < table.len() {
        table[value] = ((value * value + 127) / 255) as u8;
        value += 1;
    }
    table
}

const fn power_limited_channel(channel: u8, brightness: u16) -> u8 {
    let numerator =
        GAMMA_2[channel as usize] as u32 * brightness as u32 * BRIGHTNESS_CEILING as u32;
    ((numerator + 127_500) / 255_000) as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::controller::FallbackCause;

    #[test]
    fn signed_gaze_endpoints_and_half_rounding_are_exact() {
        assert_eq!(normalized_signed_to_fixed(-1_000), -GAZE_RANGE);
        assert_eq!(normalized_signed_to_fixed(1_000), GAZE_RANGE);
        assert_eq!(normalized_signed_to_fixed(-50), -14);
        assert_eq!(normalized_signed_to_fixed(50), 14);
    }

    #[test]
    fn pupil_scale_has_documented_endpoints_and_neutral_midpoint() {
        assert_eq!(pupil_radius(0), 7 * UNITS_PER_MM);
        assert_eq!(pupil_radius(500), 11 * UNITS_PER_MM);
        assert_eq!(pupil_radius(1_000), 15 * UNITS_PER_MM);
    }

    #[test]
    fn every_color_and_brightness_stays_under_power_ceiling() {
        for channel in 0..=u8::MAX {
            for brightness in 0..=1_000 {
                assert!(power_limited_channel(channel, brightness) <= BRIGHTNESS_CEILING);
            }
        }
        assert_eq!(power_limited_channel(255, 1_000), BRIGHTNESS_CEILING);
        assert_eq!(power_limited_channel(255, 0), 0);
    }

    #[test]
    fn autonomous_renderer_is_total_and_nonblack() {
        let mut renderer = EyeRenderer::new(MountingSign::SameDirection);
        let mut frame = [RGB8::default(); LEDS_PER_EYE];
        renderer
            .render(
                DeviceTimestampMs::from_millis_since_boot(1_000),
                OutputState::Autonomous {
                    cause: FallbackCause::Boot,
                },
                &mut frame,
            )
            .unwrap();
        assert!(frame.iter().any(|pixel| *pixel != RGB8::default()));
        assert!(frame.iter().all(|pixel| {
            pixel.r <= BRIGHTNESS_CEILING
                && pixel.g <= BRIGHTNESS_CEILING
                && pixel.b <= BRIGHTNESS_CEILING
        }));
    }

    #[test]
    fn matrix_boot_is_green_bounded_dynamic_and_panel_phased() {
        let mut left_early = [RGB8::default(); LEDS_PER_EYE];
        let mut left_later = [RGB8::default(); LEDS_PER_EYE];
        let mut right_early = [RGB8::default(); LEDS_PER_EYE];
        render_matrix_boot(0, MatrixPanel::Left, &mut left_early);
        render_matrix_boot(317, MatrixPanel::Left, &mut left_later);
        render_matrix_boot(0, MatrixPanel::Right, &mut right_early);

        for frame in [&left_early, &left_later, &right_early] {
            assert!(frame.iter().any(|pixel| *pixel != RGB8::default()));
            assert!(frame.iter().all(|pixel| {
                pixel.g >= pixel.r
                    && pixel.g >= pixel.b
                    && pixel.r <= BRIGHTNESS_CEILING
                    && pixel.g <= BRIGHTNESS_CEILING
                    && pixel.b <= BRIGHTNESS_CEILING
            }));
        }
        assert_ne!(left_early, left_later);
        assert_ne!(left_early, right_early);
    }

    #[test]
    fn matrix_boot_time_arithmetic_is_total_at_u64_maximum() {
        let mut frame = [RGB8::default(); LEDS_PER_EYE];
        render_matrix_boot(u64::MAX, MatrixPanel::Right, &mut frame);
        assert!(frame.iter().all(|pixel| {
            pixel.r <= BRIGHTNESS_CEILING
                && pixel.g <= BRIGHTNESS_CEILING
                && pixel.b <= BRIGHTNESS_CEILING
        }));
    }

    #[test]
    fn renderer_rejects_a_regressing_device_clock() {
        let mut renderer = EyeRenderer::new(MountingSign::SameDirection);
        let mut frame = [RGB8::default(); LEDS_PER_EYE];
        let output = OutputState::Autonomous {
            cause: FallbackCause::Boot,
        };
        renderer
            .render(
                DeviceTimestampMs::from_millis_since_boot(2),
                output,
                &mut frame,
            )
            .unwrap();
        assert_eq!(
            renderer.render(DeviceTimestampMs::ZERO, output, &mut frame),
            Err(RenderError::ClockRegressed {
                previous_ms: 2,
                actual_ms: 0
            })
        );
    }

    #[test]
    fn autonomous_phase_is_total_at_maximum_device_time() {
        let mut renderer = EyeRenderer::new(MountingSign::Mirrored);
        let mut frame = [RGB8::default(); LEDS_PER_EYE];
        renderer
            .render(
                DeviceTimestampMs::from_millis_since_boot(u64::MAX),
                OutputState::Autonomous {
                    cause: FallbackCause::Boot,
                },
                &mut frame,
            )
            .unwrap();
        assert!(frame.iter().all(|pixel| {
            pixel.r <= BRIGHTNESS_CEILING
                && pixel.g <= BRIGHTNESS_CEILING
                && pixel.b <= BRIGHTNESS_CEILING
        }));
    }
}
