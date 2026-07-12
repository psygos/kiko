use std::collections::VecDeque;
use std::num::{NonZeroU64, NonZeroUsize};

use crate::{Frame, PairError, SensorId, StereoPair};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct PairingWindowNs(NonZeroU64);

impl PairingWindowNs {
    pub fn new(window_ns: i64) -> Result<Self, PairingConfigError> {
        if window_ns <= 0 {
            return Err(PairingConfigError::NonPositiveWindow { window_ns });
        }
        Ok(Self(NonZeroU64::new(window_ns as u64).ok_or(
            PairingConfigError::NonPositiveWindow { window_ns },
        )?))
    }

    pub fn as_ns(&self) -> u64 {
        self.0.get()
    }
}

impl TryFrom<u64> for PairingWindowNs {
    type Error = PairingConfigError;

    fn try_from(window_ns: u64) -> Result<Self, Self::Error> {
        NonZeroU64::new(window_ns)
            .map(Self)
            .ok_or(PairingConfigError::NonPositiveWindow { window_ns: 0 })
    }
}

#[derive(Debug)]
pub enum PairingConfigError {
    NonPositiveWindow { window_ns: i64 },
}

impl std::fmt::Display for PairingConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PairingConfigError::NonPositiveWindow { window_ns } => {
                write!(f, "pairing window must be positive, got {window_ns}")
            }
        }
    }
}

impl std::error::Error for PairingConfigError {}

#[derive(Clone, Copy, Debug, Default)]
pub struct PairingStats {
    pub paired: u64,
    pub dropped_left: u64,
    pub dropped_right: u64,
    pub outside_window: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PendingFramesCapacity(NonZeroUsize);

impl PendingFramesCapacity {
    pub fn get(self) -> usize {
        self.0.get()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PendingFramesCapacityError {
    Zero,
}

impl std::fmt::Display for PendingFramesCapacityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PendingFramesCapacityError::Zero => write!(f, "pending frame capacity must be > 0"),
        }
    }
}

impl std::error::Error for PendingFramesCapacityError {}

impl TryFrom<usize> for PendingFramesCapacity {
    type Error = PendingFramesCapacityError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        NonZeroUsize::new(value)
            .map(PendingFramesCapacity)
            .ok_or(PendingFramesCapacityError::Zero)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PairingDropReason {
    PendingCapacity,
    OutsideWindow,
}

#[derive(Debug)]
pub enum PairingOutcome {
    Produced(StereoPair),
    Dropped {
        sensor: SensorId,
        reason: PairingDropReason,
    },
    Waiting,
}

#[derive(Debug)]
pub struct StereoPairer {
    window: PairingWindowNs,
    left: VecDeque<Frame>,
    right: VecDeque<Frame>,
    max_pending_per_side: PendingFramesCapacity,
    stats: PairingStats,
}

impl StereoPairer {
    pub fn new(window: PairingWindowNs) -> Self {
        Self::new_with_max_pending(
            window,
            PendingFramesCapacity::try_from(64).expect("non-zero default capacity"),
        )
    }

    pub fn new_with_max_pending(
        window: PairingWindowNs,
        max_pending_per_side: PendingFramesCapacity,
    ) -> Self {
        Self {
            window,
            left: VecDeque::new(),
            right: VecDeque::new(),
            max_pending_per_side,
            stats: PairingStats::default(),
        }
    }

    pub fn push_left(&mut self, frame: Frame) -> Option<PairingOutcome> {
        if self.left.len() >= self.max_pending_per_side.get() {
            self.left.pop_front();
            self.stats.dropped_left = self.stats.dropped_left.saturating_add(1);
            self.left.push_back(frame);
            return Some(PairingOutcome::Dropped {
                sensor: SensorId::StereoLeft,
                reason: PairingDropReason::PendingCapacity,
            });
        }
        self.left.push_back(frame);
        None
    }

    pub fn push_right(&mut self, frame: Frame) -> Option<PairingOutcome> {
        if self.right.len() >= self.max_pending_per_side.get() {
            self.right.pop_front();
            self.stats.dropped_right = self.stats.dropped_right.saturating_add(1);
            self.right.push_back(frame);
            return Some(PairingOutcome::Dropped {
                sensor: SensorId::StereoRight,
                reason: PairingDropReason::PendingCapacity,
            });
        }
        self.right.push_back(frame);
        None
    }

    pub fn next_pair(&mut self) -> Result<Option<StereoPair>, PairError> {
        match self.next_outcome()? {
            PairingOutcome::Produced(pair) => Ok(Some(pair)),
            PairingOutcome::Dropped { .. } => self.next_pair(),
            PairingOutcome::Waiting => Ok(None),
        }
    }

    pub fn next_outcome(&mut self) -> Result<PairingOutcome, PairError> {
        let left = match self.left.front() {
            Some(frame) => frame,
            None => return Ok(PairingOutcome::Waiting),
        };
        let left_ts = left.timestamp().as_nanos();

        let (best_idx, best_delta, best_ts) = match self.best_right(left_ts) {
            Some(best) => best,
            None => return Ok(PairingOutcome::Waiting),
        };

        if best_delta <= self.window.as_ns() {
            let Some(left) = self.left.pop_front() else {
                return Ok(PairingOutcome::Waiting);
            };
            let Some(right) = self.right.remove(best_idx) else {
                self.left.push_front(left);
                return Ok(PairingOutcome::Waiting);
            };
            let pair = StereoPair::try_new(left, right, self.window)?;
            self.stats.paired += 1;
            return Ok(PairingOutcome::Produced(pair));
        }

        // No match in window: drop the older frame to advance.
        if best_ts < left_ts {
            self.right.remove(best_idx);
            self.stats.dropped_right += 1;
            self.stats.outside_window += 1;
            Ok(PairingOutcome::Dropped {
                sensor: SensorId::StereoRight,
                reason: PairingDropReason::OutsideWindow,
            })
        } else {
            self.left.pop_front();
            self.stats.dropped_left += 1;
            self.stats.outside_window += 1;
            Ok(PairingOutcome::Dropped {
                sensor: SensorId::StereoLeft,
                reason: PairingDropReason::OutsideWindow,
            })
        }
    }

    pub fn stats(&self) -> PairingStats {
        self.stats
    }

    pub fn window(&self) -> PairingWindowNs {
        self.window
    }

    pub fn max_pending_per_side(&self) -> PendingFramesCapacity {
        self.max_pending_per_side
    }

    fn best_right(&self, left_ts: i64) -> Option<(usize, u64, i64)> {
        if self.right.is_empty() {
            return None;
        }

        let mut best_idx = 0usize;
        let mut best_delta = u64::MAX;
        let mut best_ts = 0i64;

        for (idx, right) in self.right.iter().enumerate() {
            let right_ts = right.timestamp().as_nanos();
            let delta = right_ts.abs_diff(left_ts);
            if delta < best_delta {
                best_delta = delta;
                best_idx = idx;
                best_ts = right_ts;
            }
        }

        Some((best_idx, best_delta, best_ts))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FrameId, SensorId, Timestamp};

    fn frame(sensor: SensorId, ts_ns: i64, id: u64) -> Frame {
        Frame::new(
            sensor,
            FrameId::new(id),
            Timestamp::from_nanos(ts_ns),
            2,
            2,
            vec![0; 4],
        )
        .expect("valid frame")
    }

    #[test]
    fn pending_left_is_capped() {
        let window = PairingWindowNs::new(5_000_000).expect("valid pairing window");
        let mut pairer = StereoPairer::new_with_max_pending(
            window,
            PendingFramesCapacity::try_from(2).expect("capacity"),
        );
        pairer.push_left(frame(SensorId::StereoLeft, 1, 1));
        pairer.push_left(frame(SensorId::StereoLeft, 2, 2));
        pairer.push_left(frame(SensorId::StereoLeft, 3, 3));

        assert_eq!(pairer.stats().dropped_left, 1);
        assert_eq!(pairer.max_pending_per_side().get(), 2);
    }

    #[test]
    fn pending_right_is_capped() {
        let window = PairingWindowNs::new(5_000_000).expect("valid pairing window");
        let mut pairer = StereoPairer::new_with_max_pending(
            window,
            PendingFramesCapacity::try_from(2).expect("capacity"),
        );
        pairer.push_right(frame(SensorId::StereoRight, 1, 1));
        pairer.push_right(frame(SensorId::StereoRight, 2, 2));
        pairer.push_right(frame(SensorId::StereoRight, 3, 3));

        assert_eq!(pairer.stats().dropped_right, 1);
    }

    #[test]
    fn next_pair_returns_none_when_side_becomes_empty() {
        let window = PairingWindowNs::new(5_000_000).expect("valid pairing window");
        let mut pairer = StereoPairer::new_with_max_pending(
            window,
            PendingFramesCapacity::try_from(1).expect("capacity"),
        );
        pairer.push_left(frame(SensorId::StereoLeft, 10, 1));
        assert!(
            pairer
                .next_pair()
                .expect("pairing should not fail")
                .is_none()
        );
    }

    #[test]
    fn next_outcome_reports_outside_window_drop() {
        let window = PairingWindowNs::new(5).expect("valid pairing window");
        let mut pairer = StereoPairer::new_with_max_pending(
            window,
            PendingFramesCapacity::try_from(2).expect("capacity"),
        );
        pairer.push_left(frame(SensorId::StereoLeft, 100, 1));
        pairer.push_right(frame(SensorId::StereoRight, 200, 2));

        let outcome = pairer.next_outcome().expect("pairing should not fail");
        assert!(matches!(
            outcome,
            PairingOutcome::Dropped {
                sensor: SensorId::StereoLeft,
                reason: PairingDropReason::OutsideWindow,
            }
        ));
    }

    #[test]
    fn pairing_handles_timestamps_spanning_the_full_i64_domain() {
        let window = PairingWindowNs::new(i64::MAX).expect("maximum signed window");
        let mut pairer = StereoPairer::new_with_max_pending(
            window,
            PendingFramesCapacity::try_from(2).expect("capacity"),
        );
        pairer.push_left(frame(SensorId::StereoLeft, i64::MIN, 1));
        pairer.push_right(frame(SensorId::StereoRight, i64::MAX, 2));

        assert!(matches!(
            pairer.next_outcome().expect("pairing should not overflow"),
            PairingOutcome::Dropped {
                sensor: SensorId::StereoLeft,
                reason: PairingDropReason::OutsideWindow,
            }
        ));
    }
}
