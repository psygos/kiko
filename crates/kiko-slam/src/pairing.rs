use std::collections::VecDeque;
use std::num::{NonZeroU64, NonZeroUsize};

use crate::{Frame, FrameDimensions, SensorId, StereoPair};

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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PairingStats {
    pub paired: u64,
    pub dropped_left: u64,
    pub dropped_right: u64,
    pub outside_window: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PendingFramesCapacity(NonZeroUsize);

const DEFAULT_PENDING_FRAMES_PER_SIDE: usize = 64;

impl PendingFramesCapacity {
    pub fn get(self) -> usize {
        self.0.get()
    }
}

impl Default for PendingFramesCapacity {
    fn default() -> Self {
        Self(NonZeroUsize::MIN.saturating_add(DEFAULT_PENDING_FRAMES_PER_SIDE - 1))
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
pub enum PairingInputError {
    SensorMismatch {
        expected: SensorId,
        actual: SensorId,
    },
    DimensionMismatch {
        expected: FrameDimensions,
        actual: FrameDimensions,
    },
    NonMonotonicTimestamp {
        sensor: SensorId,
        previous_ns: i64,
        current_ns: i64,
    },
}

impl std::fmt::Display for PairingInputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SensorMismatch { expected, actual } => write!(
                f,
                "pairing input belongs to {actual:?}, expected {expected:?}"
            ),
            Self::DimensionMismatch { expected, actual } => write!(
                f,
                "pairing input dimensions {}x{} do not match stream dimensions {}x{}",
                actual.width(),
                actual.height(),
                expected.width(),
                expected.height()
            ),
            Self::NonMonotonicTimestamp {
                sensor,
                previous_ns,
                current_ns,
            } => write!(
                f,
                "pairing input timestamp for {sensor:?} must increase strictly: previous={previous_ns}ns, current={current_ns}ns"
            ),
        }
    }
}

impl std::error::Error for PairingInputError {}

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
    dimensions: Option<FrameDimensions>,
    last_left_timestamp_ns: Option<i64>,
    last_right_timestamp_ns: Option<i64>,
    max_pending_per_side: PendingFramesCapacity,
    stats: PairingStats,
}

impl StereoPairer {
    pub fn new(window: PairingWindowNs) -> Self {
        Self::new_with_max_pending(window, PendingFramesCapacity::default())
    }

    pub fn new_with_max_pending(
        window: PairingWindowNs,
        max_pending_per_side: PendingFramesCapacity,
    ) -> Self {
        Self {
            window,
            left: VecDeque::new(),
            right: VecDeque::new(),
            dimensions: None,
            last_left_timestamp_ns: None,
            last_right_timestamp_ns: None,
            max_pending_per_side,
            stats: PairingStats::default(),
        }
    }

    /// Parse a left-stream frame before any queue or drop-stat mutation.
    ///
    /// The first accepted frame fixes the dimensions for both streams for the
    /// lifetime of this pairer.
    pub fn push_left(&mut self, frame: Frame) -> Result<Option<PairingOutcome>, PairingInputError> {
        let (dimensions, timestamp_ns) =
            self.parse_input(&frame, SensorId::StereoLeft, self.last_left_timestamp_ns)?;
        self.dimensions.get_or_insert(dimensions);
        self.last_left_timestamp_ns = Some(timestamp_ns);
        if self.left.len() >= self.max_pending_per_side.get() {
            self.left.pop_front();
            self.stats.dropped_left = self.stats.dropped_left.saturating_add(1);
            self.left.push_back(frame);
            return Ok(Some(PairingOutcome::Dropped {
                sensor: SensorId::StereoLeft,
                reason: PairingDropReason::PendingCapacity,
            }));
        }
        self.left.push_back(frame);
        Ok(None)
    }

    /// Parse a right-stream frame before any queue or drop-stat mutation.
    ///
    /// The first accepted frame fixes the dimensions for both streams for the
    /// lifetime of this pairer.
    pub fn push_right(
        &mut self,
        frame: Frame,
    ) -> Result<Option<PairingOutcome>, PairingInputError> {
        let (dimensions, timestamp_ns) =
            self.parse_input(&frame, SensorId::StereoRight, self.last_right_timestamp_ns)?;
        self.dimensions.get_or_insert(dimensions);
        self.last_right_timestamp_ns = Some(timestamp_ns);
        if self.right.len() >= self.max_pending_per_side.get() {
            self.right.pop_front();
            self.stats.dropped_right = self.stats.dropped_right.saturating_add(1);
            self.right.push_back(frame);
            return Ok(Some(PairingOutcome::Dropped {
                sensor: SensorId::StereoRight,
                reason: PairingDropReason::PendingCapacity,
            }));
        }
        self.right.push_back(frame);
        Ok(None)
    }

    pub fn next_pair(&mut self) -> Option<StereoPair> {
        loop {
            match self.next_outcome() {
                PairingOutcome::Produced(pair) => return Some(pair),
                PairingOutcome::Dropped { .. } => {}
                PairingOutcome::Waiting => return None,
            }
        }
    }

    pub fn next_outcome(&mut self) -> PairingOutcome {
        let Some(left) = self.left.front() else {
            return PairingOutcome::Waiting;
        };
        let left_ts = left.timestamp().as_nanos();

        let Some((best_idx, best_delta, best_ts)) = self.best_right(left_ts) else {
            return PairingOutcome::Waiting;
        };

        if best_delta <= self.window.as_ns() {
            let left = self
                .left
                .pop_front()
                .expect("front frame observed immediately before removal");
            let right = self
                .right
                .remove(best_idx)
                .expect("best right index came from the same private queue");
            self.stats.paired = self.stats.paired.saturating_add(1);
            return PairingOutcome::Produced(StereoPair::from_parts(left, right));
        }

        let sensor = if best_ts < left_ts {
            self.right.remove(best_idx);
            self.stats.dropped_right = self.stats.dropped_right.saturating_add(1);
            SensorId::StereoRight
        } else {
            self.left.pop_front();
            self.stats.dropped_left = self.stats.dropped_left.saturating_add(1);
            SensorId::StereoLeft
        };
        self.stats.outside_window = self.stats.outside_window.saturating_add(1);
        PairingOutcome::Dropped {
            sensor,
            reason: PairingDropReason::OutsideWindow,
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

    pub fn dimensions(&self) -> Option<FrameDimensions> {
        self.dimensions
    }

    fn parse_input(
        &self,
        frame: &Frame,
        expected_sensor: SensorId,
        previous_timestamp_ns: Option<i64>,
    ) -> Result<(FrameDimensions, i64), PairingInputError> {
        let actual_sensor = frame.sensor_id();
        if actual_sensor != expected_sensor {
            return Err(PairingInputError::SensorMismatch {
                expected: expected_sensor,
                actual: actual_sensor,
            });
        }
        let actual_dimensions = frame.dimensions();
        if let Some(expected) = self.dimensions
            && actual_dimensions != expected
        {
            return Err(PairingInputError::DimensionMismatch {
                expected,
                actual: actual_dimensions,
            });
        }
        let current_timestamp_ns = frame.timestamp().as_nanos();
        if let Some(previous_ns) = previous_timestamp_ns
            && current_timestamp_ns <= previous_ns
        {
            return Err(PairingInputError::NonMonotonicTimestamp {
                sensor: expected_sensor,
                previous_ns,
                current_ns: current_timestamp_ns,
            });
        }
        Ok((actual_dimensions, current_timestamp_ns))
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
        frame_with_dimensions(sensor, ts_ns, id, 2, 2)
    }

    fn frame_with_dimensions(
        sensor: SensorId,
        ts_ns: i64,
        id: u64,
        width: u32,
        height: u32,
    ) -> Frame {
        Frame::new(
            sensor,
            FrameId::new(id),
            Timestamp::from_nanos(ts_ns),
            width,
            height,
            vec![0; (width * height) as usize],
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
        pairer
            .push_left(frame(SensorId::StereoLeft, 1, 1))
            .expect("left frame");
        pairer
            .push_left(frame(SensorId::StereoLeft, 2, 2))
            .expect("left frame");
        pairer
            .push_left(frame(SensorId::StereoLeft, 3, 3))
            .expect("left frame");

        assert_eq!(pairer.stats().dropped_left, 1);
        assert_eq!(pairer.max_pending_per_side().get(), 2);
    }

    #[test]
    fn default_pending_capacity_matches_the_validated_public_constructor() {
        assert_eq!(
            PendingFramesCapacity::default(),
            PendingFramesCapacity::try_from(DEFAULT_PENDING_FRAMES_PER_SIDE)
                .expect("default capacity must satisfy the public invariant")
        );
    }

    #[test]
    fn pending_right_is_capped() {
        let window = PairingWindowNs::new(5_000_000).expect("valid pairing window");
        let mut pairer = StereoPairer::new_with_max_pending(
            window,
            PendingFramesCapacity::try_from(2).expect("capacity"),
        );
        pairer
            .push_right(frame(SensorId::StereoRight, 1, 1))
            .expect("right frame");
        pairer
            .push_right(frame(SensorId::StereoRight, 2, 2))
            .expect("right frame");
        pairer
            .push_right(frame(SensorId::StereoRight, 3, 3))
            .expect("right frame");

        assert_eq!(pairer.stats().dropped_right, 1);
    }

    #[test]
    fn next_pair_returns_none_when_side_becomes_empty() {
        let window = PairingWindowNs::new(5_000_000).expect("valid pairing window");
        let mut pairer = StereoPairer::new_with_max_pending(
            window,
            PendingFramesCapacity::try_from(1).expect("capacity"),
        );
        pairer
            .push_left(frame(SensorId::StereoLeft, 10, 1))
            .expect("left frame");
        assert!(pairer.next_pair().is_none());
    }

    #[test]
    fn next_outcome_reports_outside_window_drop() {
        let window = PairingWindowNs::new(5).expect("valid pairing window");
        let mut pairer = StereoPairer::new_with_max_pending(
            window,
            PendingFramesCapacity::try_from(2).expect("capacity"),
        );
        pairer
            .push_left(frame(SensorId::StereoLeft, 100, 1))
            .expect("left frame");
        pairer
            .push_right(frame(SensorId::StereoRight, 200, 2))
            .expect("right frame");

        let outcome = pairer.next_outcome();
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
        pairer
            .push_left(frame(SensorId::StereoLeft, i64::MIN, 1))
            .expect("left frame");
        pairer
            .push_right(frame(SensorId::StereoRight, i64::MAX, 2))
            .expect("right frame");

        assert!(matches!(
            pairer.next_outcome(),
            PairingOutcome::Dropped {
                sensor: SensorId::StereoLeft,
                reason: PairingDropReason::OutsideWindow,
            }
        ));
    }

    #[test]
    fn lifetime_statistics_saturate_instead_of_wrapping() {
        let window = PairingWindowNs::new(1).expect("valid narrow window");
        let mut pairer = StereoPairer::new(window);
        pairer.stats = PairingStats {
            paired: u64::MAX,
            dropped_left: u64::MAX,
            dropped_right: u64::MAX,
            outside_window: u64::MAX,
        };

        pairer
            .push_left(frame(SensorId::StereoLeft, 10, 1))
            .expect("left frame");
        pairer
            .push_right(frame(SensorId::StereoRight, 10, 2))
            .expect("right frame");
        assert!(pairer.next_pair().is_some());

        pairer
            .push_left(frame(SensorId::StereoLeft, 20, 3))
            .expect("left frame");
        pairer
            .push_right(frame(SensorId::StereoRight, 30, 4))
            .expect("right frame");
        assert!(pairer.next_pair().is_none());
        pairer
            .push_left(frame(SensorId::StereoLeft, 40, 5))
            .expect("left frame");
        assert!(pairer.next_pair().is_none());

        assert_eq!(pairer.stats().paired, u64::MAX);
        assert_eq!(pairer.stats().dropped_left, u64::MAX);
        assert_eq!(pairer.stats().dropped_right, u64::MAX);
        assert_eq!(pairer.stats().outside_window, u64::MAX);
    }

    #[test]
    fn wrong_side_input_is_rejected_before_capacity_eviction() {
        let window = PairingWindowNs::new(1).expect("narrow window");
        let mut pairer = StereoPairer::new_with_max_pending(
            window,
            PendingFramesCapacity::try_from(1).expect("valid capacity"),
        );
        pairer
            .push_left(frame(SensorId::StereoLeft, 10, 1))
            .expect("valid left frame");
        let before = pairer.stats();

        assert!(matches!(
            pairer.push_left(frame(SensorId::StereoRight, 10, 99)),
            Err(PairingInputError::SensorMismatch {
                expected: SensorId::StereoLeft,
                actual: SensorId::StereoRight,
            })
        ));
        assert_eq!(pairer.stats(), before);

        pairer
            .push_right(frame(SensorId::StereoRight, 10, 2))
            .expect("valid right frame");
        let pair = pairer
            .next_pair()
            .expect("original left frame was retained");
        assert_eq!(pair.left().frame_id(), FrameId::new(1));
    }

    #[test]
    fn dimension_mismatch_is_rejected_without_mutating_queues_or_stats() {
        let window = PairingWindowNs::new(1).expect("narrow window");
        let mut pairer = StereoPairer::new(window);
        pairer
            .push_left(frame_with_dimensions(SensorId::StereoLeft, 10, 1, 2, 2))
            .expect("valid left frame");
        let before = pairer.stats();
        let expected = FrameDimensions::try_new(2, 2).expect("dimensions");
        let actual = FrameDimensions::try_new(3, 2).expect("dimensions");
        assert_eq!(pairer.dimensions(), Some(expected));

        assert!(matches!(
            pairer.push_right(frame_with_dimensions(SensorId::StereoRight, 10, 99, 3, 2,)),
            Err(PairingInputError::DimensionMismatch {
                expected: reported_expected,
                actual: reported_actual,
            }) if reported_expected == expected && reported_actual == actual
        ));
        assert_eq!(pairer.stats(), before);
        assert_eq!(pairer.dimensions(), Some(expected));

        pairer
            .push_right(frame(SensorId::StereoRight, 10, 2))
            .expect("valid right frame");
        assert!(pairer.next_pair().is_some());
    }

    #[test]
    fn regressing_timestamp_is_rejected_before_pairing_state_mutation() {
        let window = PairingWindowNs::new(1).expect("narrow window");
        let mut pairer = StereoPairer::new(window);
        pairer
            .push_left(frame(SensorId::StereoLeft, 10, 1))
            .expect("first left frame");
        pairer
            .push_right(frame(SensorId::StereoRight, 20, 2))
            .expect("first right frame");
        assert!(pairer.next_pair().is_none());
        let before = pairer.stats();

        assert!(matches!(
            pairer.push_right(frame(SensorId::StereoRight, 10, 99)),
            Err(PairingInputError::NonMonotonicTimestamp {
                sensor: SensorId::StereoRight,
                previous_ns: 20,
                current_ns: 10,
            })
        ));
        assert_eq!(pairer.stats(), before);

        pairer
            .push_left(frame(SensorId::StereoLeft, 20, 3))
            .expect("next monotonic left frame");
        let pair = pairer
            .next_pair()
            .expect("retained right frame should still pair");
        assert_eq!(pair.left().frame_id(), FrameId::new(3));
        assert_eq!(pair.right().frame_id(), FrameId::new(2));
    }
}
