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
    stats: PairingStats,
}

impl StereoPairer {
    pub fn new(window: PairingWindowNs) -> Self {
        Self {
            window,
            left: VecDeque::new(),
            right: VecDeque::new(),
            stats: PairingStats::default(),
        }
    }

    pub fn push_left(&mut self, frame: Frame) {
        self.left.push_back(frame);
    }

    pub fn push_right(&mut self, frame: Frame) {
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
                let left = self.left.pop_front().expect("left frame should exist");
                let right = self
                    .right
                    .remove(best_idx)
                    .expect("right frame should exist");
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
