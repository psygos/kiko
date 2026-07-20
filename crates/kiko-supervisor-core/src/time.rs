use core::fmt;
use core::num::NonZeroU64;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MonotonicInstant(u64);

impl MonotonicInstant {
    pub const ZERO: Self = Self(0);

    pub const fn from_nanos_since_process_start(nanoseconds: u64) -> Self {
        Self(nanoseconds)
    }

    pub const fn as_nanos(self) -> u64 {
        self.0
    }

    pub(crate) const fn checked_add(self, duration: AuthorityDuration) -> Option<Self> {
        match self.0.checked_add(duration.as_nanos()) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }

    pub(crate) const fn checked_elapsed_since(self, earlier: Self) -> Option<u64> {
        self.0.checked_sub(earlier.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AuthorityDuration(NonZeroU64);

impl AuthorityDuration {
    pub const fn try_from_nanos(nanoseconds: u64) -> Result<Self, TimeValueError> {
        match NonZeroU64::new(nanoseconds) {
            Some(value) => Ok(Self(value)),
            None => Err(TimeValueError::ZeroDuration),
        }
    }

    pub const fn as_nanos(self) -> u64 {
        self.0.get()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimeValueError {
    ZeroDuration,
}

impl fmt::Display for TimeValueError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("authority duration must be non-zero")
    }
}

impl core::error::Error for TimeValueError {}
