use std::collections::VecDeque;

use crate::{Frame, PairError, StereoPair};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct PairingWindowNs(i64);

impl PairingWindowNs {
    pub fn new(window_ns: i64) -> Result<Self, PairingConfigError> {
        if window_ns <= 0 {
            return Err(PairingConfigError::NonPositiveWindow { window_ns });
        }
        Ok(Self(window_ns))
    }

    pub fn as_ns(&self) -> i64 {
        self.0
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

#[derive(Debug)]
pub struct StereoPairer {
    window: PairingWindowNs,
    left: VecDeque<Frame>,
    right: VecDeque<Frame>,
    max_pending_per_side: usize,
    stats: PairingStats,
}

impl StereoPairer {
    pub fn new(window: PairingWindowNs) -> Self {
        Self::new_with_max_pending(window, 64)
    }

    pub fn new_with_max_pending(window: PairingWindowNs, max_pending_per_side: usize) -> Self {
        Self {
            window,
            left: VecDeque::new(),
            right: VecDeque::new(),
            max_pending_per_side: max_pending_per_side.max(1),
            stats: PairingStats::default(),
        }
    }

    pub fn push_left(&mut self, frame: Frame) {
        if self.left.len() >= self.max_pending_per_side {
            self.left.pop_front();
            self.stats.dropped_left = self.stats.dropped_left.saturating_add(1);
        }
        self.left.push_back(frame);
    }

    pub fn push_right(&mut self, frame: Frame) {
        if self.right.len() >= self.max_pending_per_side {
            self.right.pop_front();
            self.stats.dropped_right = self.stats.dropped_right.saturating_add(1);
        }
        self.right.push_back(frame);
    }

    pub fn next_pair(&mut self) -> Result<Option<StereoPair>, PairError> {
        loop {
            let left = match self.left.front() {
                Some(frame) => frame,
                None => return Ok(None),
            };
            let left_ts = left.timestamp().as_nanos();

            let (best_idx, best_delta, best_ts) = match self.best_right(left_ts) {
                Some(best) => best,
                None => return Ok(None),
            };

            if best_delta <= self.window.as_ns() {
                let Some(left) = self.left.pop_front() else {
                    return Ok(None);
                };
                let Some(right) = self.right.remove(best_idx) else {
                    self.left.push_front(left);
                    return Ok(None);
                };
                let pair = StereoPair::try_new(left, right, self.window)?;
                self.stats.paired += 1;
                return Ok(Some(pair));
            }

            // No match in window: drop the older frame to advance.
            if best_ts < left_ts {
                self.right.remove(best_idx);
                self.stats.dropped_right += 1;
            } else {
                self.left.pop_front();
                self.stats.dropped_left += 1;
            }
            self.stats.outside_window += 1;
        }
    }

    pub fn stats(&self) -> PairingStats {
        self.stats
    }

    pub fn window(&self) -> PairingWindowNs {
        self.window
    }

    pub fn max_pending_per_side(&self) -> usize {
        self.max_pending_per_side
    }

    fn best_right(&self, left_ts: i64) -> Option<(usize, i64, i64)> {
        if self.right.is_empty() {
            return None;
        }

        let mut best_idx = 0usize;
        let mut best_delta = i64::MAX;
        let mut best_ts = 0i64;

        for (idx, right) in self.right.iter().enumerate() {
            let right_ts = right.timestamp().as_nanos();
            let delta = (right_ts - left_ts).abs();
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
        let mut pairer = StereoPairer::new_with_max_pending(window, 2);
        pairer.push_left(frame(SensorId::StereoLeft, 1, 1));
        pairer.push_left(frame(SensorId::StereoLeft, 2, 2));
        pairer.push_left(frame(SensorId::StereoLeft, 3, 3));

        assert_eq!(pairer.stats().dropped_left, 1);
        assert_eq!(pairer.max_pending_per_side(), 2);
    }

    #[test]
    fn pending_right_is_capped() {
        let window = PairingWindowNs::new(5_000_000).expect("valid pairing window");
        let mut pairer = StereoPairer::new_with_max_pending(window, 2);
        pairer.push_right(frame(SensorId::StereoRight, 1, 1));
        pairer.push_right(frame(SensorId::StereoRight, 2, 2));
        pairer.push_right(frame(SensorId::StereoRight, 3, 3));

        assert_eq!(pairer.stats().dropped_right, 1);
    }

    #[test]
    fn next_pair_returns_none_when_side_becomes_empty() {
        let window = PairingWindowNs::new(5_000_000).expect("valid pairing window");
        let mut pairer = StereoPairer::new_with_max_pending(window, 1);
        pairer.push_left(frame(SensorId::StereoLeft, 10, 1));
        assert!(
            pairer
                .next_pair()
                .expect("pairing should not fail")
                .is_none()
        );
    }
}
