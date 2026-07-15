#![no_std]

use robot_protocol::PwmPercent;

/// Applies one pending timer update using the timer's currently reported direction.
///
/// The direction is an assumption: a quadrature input can reverse after setting
/// the update flag and before this snapshot reads the direction bit.
pub const fn encoder_wraps_with_pending_direction_assumption(
    committed: i64,
    update_pending: bool,
    counting_down: bool,
) -> i64 {
    if !update_pending {
        committed
    } else if counting_down {
        committed.wrapping_sub(1)
    } else {
        committed.wrapping_add(1)
    }
}

pub fn pwm_duty(pwm_percent: PwmPercent, maximum_duty: u16) -> u16 {
    let scaled = u32::from(pwm_percent.get().unsigned_abs()) * u32::from(maximum_duty) / 100;
    // PwmPercent proves an absolute value at most 100, so scaled cannot exceed
    // maximum_duty and the narrowing conversion preserves the value.
    scaled as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pending_encoder_wrap_assumption_is_directional_and_wraps_explicitly() {
        assert_eq!(
            encoder_wraps_with_pending_direction_assumption(10, false, false),
            10
        );
        assert_eq!(
            encoder_wraps_with_pending_direction_assumption(10, true, false),
            11
        );
        assert_eq!(
            encoder_wraps_with_pending_direction_assumption(10, true, true),
            9
        );
        assert_eq!(
            encoder_wraps_with_pending_direction_assumption(i64::MAX, true, false),
            i64::MIN
        );
        assert_eq!(
            encoder_wraps_with_pending_direction_assumption(i64::MIN, true, true),
            i64::MAX
        );
    }

    #[test]
    fn pwm_scaling_is_bounded_for_every_valid_integer_percent() {
        for raw in -100..=100 {
            let pwm = PwmPercent::try_new(raw).expect("loop covers the complete valid domain");
            for maximum_duty in [0, 1, 99, 100, 1_000, u16::MAX] {
                assert!(pwm_duty(pwm, maximum_duty) <= maximum_duty);
            }
        }
        assert_eq!(pwm_duty(PwmPercent::ZERO, u16::MAX), 0);
        assert_eq!(
            pwm_duty(PwmPercent::try_new(100).expect("valid endpoint"), u16::MAX),
            u16::MAX
        );
        assert_eq!(
            pwm_duty(PwmPercent::try_new(-100).expect("valid endpoint"), u16::MAX),
            u16::MAX
        );
    }
}
