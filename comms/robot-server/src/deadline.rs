use std::fmt;
use std::num::{NonZeroU16, NonZeroU32};
use std::time::{Duration, Instant};

use robot_protocol::v2::{
    ControllerDeadlineMsWrapping, DeadlineRelation, RemainingLeaseMs, V2CommandLeaseMs,
};
use robot_protocol::ControllerUptimeMsWrapping;

const PPM_DENOMINATOR: u128 = 1_000_000;
const MIN_CONTROLLER_EXECUTION_WINDOW_MS: u64 = 10;
const MAX_UNAMBIGUOUS_CONTROLLER_OFFSET_MS: u64 = (1_u64 << 31) - 1;

#[derive(Clone, Copy, Debug)]
pub struct HeartbeatClockSample {
    controller_uptime: ControllerUptimeMsWrapping,
    received_at: Instant,
}

impl HeartbeatClockSample {
    pub const fn new(controller_uptime: ControllerUptimeMsWrapping, received_at: Instant) -> Self {
        Self {
            controller_uptime,
            received_at,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TranslatedCommandDeadline {
    server_deadline_exclusive: Instant,
    controller_deadline_exclusive: ControllerDeadlineMsWrapping,
}

impl TranslatedCommandDeadline {
    pub const fn server_deadline_exclusive(self) -> Instant {
        self.server_deadline_exclusive
    }

    pub const fn controller_deadline_exclusive(self) -> ControllerDeadlineMsWrapping {
        self.controller_deadline_exclusive
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeadlineTranslationError {
    ServerDeadlineOverflow,
    ServerClockOrder,
    HeartbeatStale {
        age_ms_at_least: u128,
        maximum_age_ms: u128,
    },
    HeartbeatAtOrAfterServerDeadline,
    ArithmeticOverflow,
    InsufficientControllerWindow {
        translated_ms: u64,
        minimum_ms: u64,
    },
    ControllerWindowAmbiguous {
        translated_ms: u64,
        maximum_ms: u64,
    },
}

impl fmt::Display for DeadlineTranslationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ServerDeadlineOverflow => {
                formatter.write_str("command lease exceeds the server monotonic clock domain")
            }
            Self::ServerClockOrder => formatter.write_str(
                "server monotonic timestamps are not in the required heartbeat/evaluation order",
            ),
            Self::HeartbeatStale {
                age_ms_at_least,
                maximum_age_ms,
            } => write!(
                formatter,
                "controller heartbeat is at least {age_ms_at_least} ms old; maximum is {maximum_age_ms} ms"
            ),
            Self::HeartbeatAtOrAfterServerDeadline => formatter.write_str(
                "no controller execution window remains after the heartbeat reference",
            ),
            Self::ArithmeticOverflow => {
                formatter.write_str("controller deadline translation arithmetic overflowed")
            }
            Self::InsufficientControllerWindow {
                translated_ms,
                minimum_ms,
            } => write!(
                formatter,
                "translated controller execution window {translated_ms} ms is below minimum {minimum_ms} ms"
            ),
            Self::ControllerWindowAmbiguous {
                translated_ms,
                maximum_ms,
            } => write!(
                formatter,
                "translated controller window {translated_ms} ms exceeds wrapping half-range maximum {maximum_ms} ms"
            ),
        }
    }
}

impl std::error::Error for DeadlineTranslationError {}

/// Translate one host lease into an absolute controller-uptime deadline.
///
/// The first datagram receive instant is authoritative; a duplicate must pass
/// the same value. The absolute clock-error bound and an explicit quantization
/// margin both shorten the controller window, never lengthen it.
pub fn translate_command_deadline(
    first_server_receive: Instant,
    lease: V2CommandLeaseMs,
    heartbeat: HeartbeatClockSample,
    evaluated_at: Instant,
    maximum_heartbeat_age: Duration,
    controller_clock_abs_error_ppm_bound: NonZeroU32,
    quantization_margin_ms: NonZeroU16,
) -> Result<TranslatedCommandDeadline, DeadlineTranslationError> {
    let server_deadline_exclusive = first_server_receive
        .checked_add(Duration::from_millis(u64::from(lease.get())))
        .ok_or(DeadlineTranslationError::ServerDeadlineOverflow)?;
    let heartbeat_age = evaluated_at
        .checked_duration_since(heartbeat.received_at)
        .ok_or(DeadlineTranslationError::ServerClockOrder)?;
    if heartbeat_age > maximum_heartbeat_age {
        return Err(DeadlineTranslationError::HeartbeatStale {
            age_ms_at_least: duration_millis_ceil(heartbeat_age)?,
            maximum_age_ms: maximum_heartbeat_age.as_millis(),
        });
    }
    let span = server_deadline_exclusive
        .checked_duration_since(heartbeat.received_at)
        .ok_or(DeadlineTranslationError::HeartbeatAtOrAfterServerDeadline)?;
    let span_ms = u64::try_from(span.as_millis())
        .map_err(|_| DeadlineTranslationError::ArithmeticOverflow)?;
    if span_ms == 0 {
        return Err(DeadlineTranslationError::HeartbeatAtOrAfterServerDeadline);
    }
    let drift_ms = mul_div_ceil(
        span_ms,
        u64::from(controller_clock_abs_error_ppm_bound.get()),
        PPM_DENOMINATOR,
    )?;
    let translated_ms = span_ms
        .checked_sub(drift_ms)
        .and_then(|value| value.checked_sub(u64::from(quantization_margin_ms.get())))
        .ok_or(DeadlineTranslationError::InsufficientControllerWindow {
            translated_ms: 0,
            minimum_ms: MIN_CONTROLLER_EXECUTION_WINDOW_MS,
        })?;
    if translated_ms < MIN_CONTROLLER_EXECUTION_WINDOW_MS {
        return Err(DeadlineTranslationError::InsufficientControllerWindow {
            translated_ms,
            minimum_ms: MIN_CONTROLLER_EXECUTION_WINDOW_MS,
        });
    }
    if translated_ms > MAX_UNAMBIGUOUS_CONTROLLER_OFFSET_MS {
        return Err(DeadlineTranslationError::ControllerWindowAmbiguous {
            translated_ms,
            maximum_ms: MAX_UNAMBIGUOUS_CONTROLLER_OFFSET_MS,
        });
    }
    let offset =
        u32::try_from(translated_ms).map_err(|_| DeadlineTranslationError::ArithmeticOverflow)?;
    let controller_deadline_exclusive =
        ControllerDeadlineMsWrapping::new(heartbeat.controller_uptime.get().wrapping_add(offset));
    Ok(TranslatedCommandDeadline {
        server_deadline_exclusive,
        controller_deadline_exclusive,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RemainingLeaseError {
    ControllerDeadlineExpiredOrAmbiguous,
    ServerClockOrder,
    ArithmeticOverflow,
    ProtocolDomain,
}

impl fmt::Display for RemainingLeaseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ControllerDeadlineExpiredOrAmbiguous => formatter.write_str(
                "controller applied result has an expired or ambiguous absolute deadline",
            ),
            Self::ServerClockOrder => formatter.write_str(
                "controller result arrived before its serial command was sent on the server clock",
            ),
            Self::ArithmeticOverflow => {
                formatter.write_str("remaining-lease arithmetic overflowed")
            }
            Self::ProtocolDomain => {
                formatter.write_str("remaining lease is outside the V2 protocol domain")
            }
        }
    }
}

impl std::error::Error for RemainingLeaseError {}

/// Compute a conservative controller-active lifetime at server result emit.
///
/// The controller duration is divided by the maximum fast-clock factor, then
/// the complete serial round trip and one explicit timestamp quantization
/// margin are subtracted. The result is a lower bound, not the nominal lease.
pub fn conservative_remaining_lease(
    controller_applied_at: ControllerUptimeMsWrapping,
    controller_deadline_exclusive: ControllerDeadlineMsWrapping,
    serial_command_sent_at: Instant,
    controller_result_received_at: Instant,
    controller_clock_abs_error_ppm_bound: NonZeroU32,
    timestamp_quantization_margin_ms: NonZeroU16,
) -> Result<RemainingLeaseMs, RemainingLeaseError> {
    let controller_remaining_ms =
        match controller_deadline_exclusive.relation_to(controller_applied_at) {
            DeadlineRelation::Future { remaining_ms } => u64::from(remaining_ms),
            DeadlineRelation::Expired | DeadlineRelation::AmbiguousHalfRange => {
                return Err(RemainingLeaseError::ControllerDeadlineExpiredOrAmbiguous);
            }
        };
    let denominator = PPM_DENOMINATOR
        .checked_add(u128::from(controller_clock_abs_error_ppm_bound.get()))
        .ok_or(RemainingLeaseError::ArithmeticOverflow)?;
    let minimum_real_ms = u64::try_from(
        u128::from(controller_remaining_ms)
            .checked_mul(PPM_DENOMINATOR)
            .ok_or(RemainingLeaseError::ArithmeticOverflow)?
            / denominator,
    )
    .map_err(|_| RemainingLeaseError::ArithmeticOverflow)?;
    let serial_round_trip = controller_result_received_at
        .checked_duration_since(serial_command_sent_at)
        .ok_or(RemainingLeaseError::ServerClockOrder)?;
    let elapsed_ms = u64::try_from(
        duration_millis_ceil(serial_round_trip)
            .map_err(|_| RemainingLeaseError::ArithmeticOverflow)?,
    )
    .map_err(|_| RemainingLeaseError::ArithmeticOverflow)?;
    let remaining = minimum_real_ms
        .saturating_sub(elapsed_ms)
        .saturating_sub(u64::from(timestamp_quantization_margin_ms.get()));
    let remaining = u16::try_from(remaining).unwrap_or(u16::MAX);
    RemainingLeaseMs::try_new(remaining).map_err(|_| RemainingLeaseError::ProtocolDomain)
}

fn duration_millis_ceil(duration: Duration) -> Result<u128, DeadlineTranslationError> {
    let nanoseconds = duration.as_nanos();
    nanoseconds
        .checked_add(999_999)
        .map(|value| value / 1_000_000)
        .ok_or(DeadlineTranslationError::ArithmeticOverflow)
}

fn mul_div_ceil(left: u64, right: u64, denominator: u128) -> Result<u64, DeadlineTranslationError> {
    let numerator = u128::from(left)
        .checked_mul(u128::from(right))
        .ok_or(DeadlineTranslationError::ArithmeticOverflow)?;
    let rounded = numerator
        .checked_add(denominator - 1)
        .ok_or(DeadlineTranslationError::ArithmeticOverflow)?
        / denominator;
    u64::try_from(rounded).map_err(|_| DeadlineTranslationError::ArithmeticOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ppm(value: u32) -> NonZeroU32 {
        NonZeroU32::new(value).expect("nonzero ppm fixture")
    }

    fn margin(value: u16) -> NonZeroU16 {
        NonZeroU16::new(value).expect("nonzero margin fixture")
    }

    #[test]
    fn translation_shortens_deadline_for_drift_and_quantization() {
        let base = Instant::now();
        let heartbeat = HeartbeatClockSample::new(ControllerUptimeMsWrapping::new(1_000), base);
        let translated = translate_command_deadline(
            base + Duration::from_millis(10),
            V2CommandLeaseMs::try_new(120).expect("lease"),
            heartbeat,
            base + Duration::from_millis(10),
            Duration::from_millis(60),
            ppm(50_000),
            margin(2),
        )
        .expect("deadline translates");

        // Server deadline is t=130 ms. From heartbeat t=0, 5% ceil drift is
        // 7 ms and the explicit margin is 2 ms: controller offset 121 ms.
        assert_eq!(translated.controller_deadline_exclusive().get(), 1_121);
        assert_eq!(
            translated.server_deadline_exclusive(),
            base + Duration::from_millis(130)
        );
    }

    #[test]
    fn stale_heartbeat_and_equality_fail_closed() {
        let base = Instant::now();
        let heartbeat = HeartbeatClockSample::new(ControllerUptimeMsWrapping::new(0), base);
        assert!(matches!(
            translate_command_deadline(
                base + Duration::from_millis(61),
                V2CommandLeaseMs::try_new(50).expect("lease"),
                heartbeat,
                base + Duration::from_millis(61),
                Duration::from_millis(60),
                ppm(1),
                margin(1),
            ),
            Err(DeadlineTranslationError::HeartbeatStale { .. })
        ));

        let equality_heartbeat = HeartbeatClockSample::new(
            ControllerUptimeMsWrapping::new(0),
            base + Duration::from_millis(50),
        );
        assert!(matches!(
            translate_command_deadline(
                base,
                V2CommandLeaseMs::try_new(50).expect("lease"),
                equality_heartbeat,
                base + Duration::from_millis(50),
                Duration::from_millis(60),
                ppm(1),
                margin(1),
            ),
            Err(DeadlineTranslationError::HeartbeatAtOrAfterServerDeadline)
        ));
    }

    #[test]
    fn same_first_receive_preserves_server_deadline_for_duplicates() {
        let base = Instant::now();
        let lease = V2CommandLeaseMs::try_new(120).expect("lease");
        let first = translate_command_deadline(
            base,
            lease,
            HeartbeatClockSample::new(ControllerUptimeMsWrapping::new(100), base),
            base,
            Duration::from_millis(60),
            ppm(50_000),
            margin(2),
        )
        .expect("first translation");
        let duplicate = translate_command_deadline(
            base,
            lease,
            HeartbeatClockSample::new(
                ControllerUptimeMsWrapping::new(110),
                base + Duration::from_millis(10),
            ),
            base + Duration::from_millis(10),
            Duration::from_millis(60),
            ppm(50_000),
            margin(2),
        )
        .expect("duplicate translation");
        assert_eq!(
            first.server_deadline_exclusive(),
            duplicate.server_deadline_exclusive()
        );
    }

    #[test]
    fn remaining_lease_is_a_lower_bound_after_fast_clock_and_serial_time() {
        let base = Instant::now();
        let remaining = conservative_remaining_lease(
            ControllerUptimeMsWrapping::new(1_000),
            ControllerDeadlineMsWrapping::new(1_120),
            base,
            base + Duration::from_micros(10_100),
            ppm(50_000),
            margin(1),
        )
        .expect("remaining lease");
        // floor(120 / 1.05) = 114, ceil(serial)=11, margin=1.
        assert_eq!(remaining.get(), 102);
    }

    #[test]
    fn wrapping_controller_deadline_remains_unambiguous() {
        let base = Instant::now();
        let remaining = conservative_remaining_lease(
            ControllerUptimeMsWrapping::new(u32::MAX - 20),
            ControllerDeadlineMsWrapping::new(29),
            base,
            base,
            ppm(1),
            margin(1),
        )
        .expect("wrapped deadline");
        assert_eq!(remaining.get(), 48);
    }
}
