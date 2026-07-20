/// Timestamp in one caller-owned monotonic nanosecond domain.
///
/// The type deliberately does not claim wall-clock meaning or cross-process
/// comparability.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MonotonicTimestampNs(u64);

impl MonotonicTimestampNs {
    pub const fn from_nanos(value: u64) -> Self {
        Self(value)
    }

    pub const fn as_nanos(self) -> u64 {
        self.0
    }

    pub(crate) fn checked_age_since(self, observed_at: Self) -> Option<u64> {
        self.0.checked_sub(observed_at.0)
    }
}
