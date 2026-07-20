//! Explicit monotonic timestamps and exclusive freshness deadlines.

use core::{fmt, num::NonZeroU64};

/// Nanoseconds since one process-local monotonic clock epoch.
///
/// Values from different clock epochs must never be mixed. The I/O owner is
/// responsible for assigning one epoch and resetting all state on restart.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MonotonicTimestamp(u64);

impl MonotonicTimestamp {
    pub const ZERO: Self = Self(0);

    pub const fn from_nanos_since_epoch(nanoseconds: u64) -> Self {
        Self(nanoseconds)
    }

    pub const fn nanos_since_epoch(self) -> u64 {
        self.0
    }

    pub const fn checked_add(self, duration: NonZeroDuration) -> Result<Self, TimeError> {
        match self.0.checked_add(duration.as_nanos()) {
            Some(value) => Ok(Self(value)),
            None => Err(TimeError::TimestampOverflow {
                timestamp_ns: self.0,
                duration_ns: duration.as_nanos(),
            }),
        }
    }
}

/// A non-zero monotonic duration in nanoseconds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NonZeroDuration(NonZeroU64);

impl NonZeroDuration {
    pub const fn try_from_nanos(nanoseconds: u64) -> Result<Self, TimeError> {
        match NonZeroU64::new(nanoseconds) {
            Some(value) => Ok(Self(value)),
            None => Err(TimeError::ZeroDuration),
        }
    }

    pub const fn as_nanos(self) -> u64 {
        self.0.get()
    }
}

/// An exclusive monotonic deadline: `now < deadline` is alive.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Deadline(MonotonicTimestamp);

impl Deadline {
    pub const fn after(
        start: MonotonicTimestamp,
        duration: NonZeroDuration,
    ) -> Result<Self, TimeError> {
        match start.checked_add(duration) {
            Ok(timestamp) => Ok(Self(timestamp)),
            Err(error) => Err(error),
        }
    }

    pub const fn timestamp(self) -> MonotonicTimestamp {
        self.0
    }

    pub const fn is_alive_at(self, now: MonotonicTimestamp) -> bool {
        now.0 < self.0.0
    }

    pub(crate) const fn earlier(self, other: Self) -> Self {
        if self.0.0 <= other.0.0 { self } else { other }
    }
}

/// A capture time paired with a strictly later exclusive deadline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FreshnessWindow {
    observed_at: MonotonicTimestamp,
    valid_until_exclusive: Deadline,
}

impl FreshnessWindow {
    pub const fn from_ttl(
        observed_at: MonotonicTimestamp,
        ttl: NonZeroDuration,
    ) -> Result<Self, TimeError> {
        match Deadline::after(observed_at, ttl) {
            Ok(valid_until_exclusive) => Ok(Self {
                observed_at,
                valid_until_exclusive,
            }),
            Err(error) => Err(error),
        }
    }

    pub const fn try_new(
        observed_at: MonotonicTimestamp,
        valid_until_exclusive: Deadline,
    ) -> Result<Self, TimeError> {
        if valid_until_exclusive.0.0 <= observed_at.0 {
            Err(TimeError::DeadlineNotAfterObservation {
                observed_at_ns: observed_at.0,
                deadline_ns: valid_until_exclusive.0.0,
            })
        } else {
            Ok(Self {
                observed_at,
                valid_until_exclusive,
            })
        }
    }

    pub const fn observed_at(self) -> MonotonicTimestamp {
        self.observed_at
    }

    pub const fn valid_until_exclusive(self) -> Deadline {
        self.valid_until_exclusive
    }

    /// Future-dated observations are not fresh.
    pub const fn is_fresh_at(self, now: MonotonicTimestamp) -> bool {
        self.observed_at.0 <= now.0 && self.valid_until_exclusive.is_alive_at(now)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimeError {
    ZeroDuration,
    TimestampOverflow {
        timestamp_ns: u64,
        duration_ns: u64,
    },
    DeadlineNotAfterObservation {
        observed_at_ns: u64,
        deadline_ns: u64,
    },
}

impl fmt::Display for TimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDuration => formatter.write_str("duration must be non-zero"),
            Self::TimestampOverflow {
                timestamp_ns,
                duration_ns,
            } => write!(
                formatter,
                "monotonic timestamp {timestamp_ns}ns plus duration {duration_ns}ns overflows"
            ),
            Self::DeadlineNotAfterObservation {
                observed_at_ns,
                deadline_ns,
            } => write!(
                formatter,
                "exclusive deadline {deadline_ns}ns must be after observation {observed_at_ns}ns"
            ),
        }
    }
}

impl core::error::Error for TimeError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn freshness_is_start_inclusive_and_deadline_exclusive() {
        let observed = MonotonicTimestamp::from_nanos_since_epoch(10);
        let ttl = NonZeroDuration::try_from_nanos(5).unwrap();
        let window = FreshnessWindow::from_ttl(observed, ttl).unwrap();

        assert!(!window.is_fresh_at(MonotonicTimestamp::from_nanos_since_epoch(9)));
        assert!(window.is_fresh_at(observed));
        assert!(window.is_fresh_at(MonotonicTimestamp::from_nanos_since_epoch(14)));
        assert!(!window.is_fresh_at(MonotonicTimestamp::from_nanos_since_epoch(15)));
    }

    #[test]
    fn overflow_and_zero_ttl_are_rejected() {
        assert_eq!(
            NonZeroDuration::try_from_nanos(0),
            Err(TimeError::ZeroDuration)
        );
        let ttl = NonZeroDuration::try_from_nanos(2).unwrap();
        assert!(
            FreshnessWindow::from_ttl(
                MonotonicTimestamp::from_nanos_since_epoch(u64::MAX - 1),
                ttl
            )
            .is_err()
        );
    }
}
