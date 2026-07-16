use std::collections::VecDeque;
use std::num::NonZeroUsize;

use crate::{Frame, FrameDimensions, SensorId, StereoPair};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct PairingWindowNs(i64);

impl PairingWindowNs {
    pub fn new(window_ns: i64) -> Result<Self, PairingConfigError> {
        if window_ns < 0 {
            return Err(PairingConfigError::NegativeWindow { window_ns });
        }
        Ok(Self(window_ns))
    }

    pub fn try_from_u64(window_ns: u64) -> Result<Self, PairingConfigError> {
        let window_ns = i64::try_from(window_ns)
            .map_err(|_| PairingConfigError::WindowTooLarge { window_ns })?;
        Ok(Self(window_ns))
    }

    pub fn as_ns(&self) -> i64 {
        self.0
    }

    pub fn as_u64(&self) -> u64 {
        self.0 as u64
    }
}

#[derive(Debug)]
pub enum PairingConfigError {
    NegativeWindow { window_ns: i64 },
    WindowTooLarge { window_ns: u64 },
    ZeroMaxPendingPerSide,
}

impl std::fmt::Display for PairingConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PairingConfigError::NegativeWindow { window_ns } => {
                write!(f, "pairing window must be non-negative, got {window_ns}")
            }
            PairingConfigError::WindowTooLarge { window_ns } => write!(
                f,
                "pairing window must fit the signed timestamp domain, got {window_ns}"
            ),
            PairingConfigError::ZeroMaxPendingPerSide => {
                write!(f, "pairer max pending frames per side must be > 0")
            }
        }
    }
}

impl std::error::Error for PairingConfigError {}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PairingStats {
    /// Stereo pairs emitted by this pairer.
    pub paired: u64,
    /// Left frames discarded by capacity or because no pending right was in the window.
    pub dropped_left: u64,
    /// Right frames discarded by capacity, an out-of-window decision, or supersession by a
    /// closer pending right.
    pub dropped_right: u64,
    /// Frames discarded specifically because the nearest pending cross-stream timestamp was
    /// outside the configured window.
    pub outside_window: u64,
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

#[derive(Debug)]
pub struct StereoPairer {
    window: PairingWindowNs,
    left: VecDeque<Frame>,
    right: VecDeque<Frame>,
    dimensions: Option<FrameDimensions>,
    last_left_timestamp_ns: Option<i64>,
    last_right_timestamp_ns: Option<i64>,
    max_pending_per_side: NonZeroUsize,
    stats: PairingStats,
}

impl StereoPairer {
    pub fn new(window: PairingWindowNs) -> Self {
        Self {
            window,
            left: VecDeque::new(),
            right: VecDeque::new(),
            dimensions: None,
            last_left_timestamp_ns: None,
            last_right_timestamp_ns: None,
            max_pending_per_side: NonZeroUsize::new(64)
                .expect("default pairer capacity is nonzero"),
            stats: PairingStats::default(),
        }
    }

    pub fn new_with_max_pending(
        window: PairingWindowNs,
        max_pending_per_side: usize,
    ) -> Result<Self, PairingConfigError> {
        let max_pending_per_side = NonZeroUsize::new(max_pending_per_side)
            .ok_or(PairingConfigError::ZeroMaxPendingPerSide)?;
        Ok(Self {
            window,
            left: VecDeque::new(),
            right: VecDeque::new(),
            dimensions: None,
            last_left_timestamp_ns: None,
            last_right_timestamp_ns: None,
            max_pending_per_side,
            stats: PairingStats::default(),
        })
    }

    /// Parse a left-stream frame before any queue or drop-stat mutation.
    ///
    /// The first accepted frame fixes the dimensions for both streams for the
    /// lifetime of this pairer.
    pub fn push_left(&mut self, frame: Frame) -> Result<(), PairingInputError> {
        let (dimensions, timestamp_ns) =
            self.parse_input(&frame, SensorId::StereoLeft, self.last_left_timestamp_ns)?;
        self.dimensions.get_or_insert(dimensions);
        self.last_left_timestamp_ns = Some(timestamp_ns);
        if self.left.len() >= self.max_pending_per_side.get() {
            self.left.pop_front();
            self.stats.dropped_left = self.stats.dropped_left.saturating_add(1);
        }
        self.left.push_back(frame);
        Ok(())
    }

    /// Parse a right-stream frame before any queue or drop-stat mutation.
    ///
    /// The first accepted frame fixes the dimensions for both streams for the
    /// lifetime of this pairer.
    pub fn push_right(&mut self, frame: Frame) -> Result<(), PairingInputError> {
        let (dimensions, timestamp_ns) =
            self.parse_input(&frame, SensorId::StereoRight, self.last_right_timestamp_ns)?;
        self.dimensions.get_or_insert(dimensions);
        self.last_right_timestamp_ns = Some(timestamp_ns);
        if self.right.len() >= self.max_pending_per_side.get() {
            self.right.pop_front();
            self.stats.dropped_right = self.stats.dropped_right.saturating_add(1);
        }
        self.right.push_back(frame);
        Ok(())
    }

    /// Match the oldest left frame to its nearest currently pending right frame.
    ///
    /// Ties select the earlier right timestamp. Rights before a selected match are superseded and
    /// discarded so emitted timestamps remain strictly increasing on both streams. This method
    /// cannot promise the globally nearest right because future arrivals are not yet observable.
    pub fn next_pair(&mut self) -> Option<StereoPair> {
        loop {
            let left = self.left.front()?;
            let left_ts = left.timestamp().as_nanos();

            let (best_idx, best_delta, best_ts) = self.best_right(left_ts)?;

            if best_delta <= self.window.as_u64() {
                self.discard_right_prefix(best_idx, false);
                let left = self
                    .left
                    .pop_front()
                    .expect("front frame observed immediately before removal");
                let right = self
                    .right
                    .pop_front()
                    .expect("selected right remained after discarding its strict prefix");
                let pair = StereoPair::from_parts(left, right);
                self.stats.paired = self.stats.paired.saturating_add(1);
                return Some(pair);
            }

            // No match in window: drop the older frame to advance.
            if best_ts < left_ts {
                // Every preceding right is still older and at least as far from this left frame.
                // Future left timestamps can only make that prefix less useful, so discard it in
                // one decision while retaining the existing per-frame diagnostics.
                self.discard_right_prefix(best_idx + 1, true);
            } else {
                self.left.pop_front();
                self.stats.dropped_left = self.stats.dropped_left.saturating_add(1);
                self.stats.outside_window = self.stats.outside_window.saturating_add(1);
            }
        }
    }

    pub fn stats(&self) -> PairingStats {
        self.stats
    }

    pub fn window(&self) -> PairingWindowNs {
        self.window
    }

    pub fn max_pending_per_side(&self) -> usize {
        self.max_pending_per_side.get()
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
        let first = self.right.front()?;
        let mut best_idx = 0usize;
        let mut best_ts = first.timestamp().as_nanos();
        let mut best_delta = best_ts.abs_diff(left_ts);

        // Strictly increasing right timestamps make absolute distance unimodal. Stop once the
        // distance no longer improves; equality deliberately preserves the earlier timestamp.
        for (idx, right) in self.right.iter().enumerate().skip(1) {
            let right_ts = right.timestamp().as_nanos();
            let delta = right_ts.abs_diff(left_ts);
            if delta >= best_delta {
                break;
            }
            best_delta = delta;
            best_idx = idx;
            best_ts = right_ts;
        }

        Some((best_idx, best_delta, best_ts))
    }

    fn discard_right_prefix(&mut self, count: usize, outside_window: bool) {
        debug_assert!(count <= self.right.len());
        for _ in 0..count {
            self.right
                .pop_front()
                .expect("discard count was derived from the same private queue");
            self.stats.dropped_right = self.stats.dropped_right.saturating_add(1);
            if outside_window {
                self.stats.outside_window = self.stats.outside_window.saturating_add(1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FrameId, PairError, Timestamp};

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

    fn reference_pairing(
        left: &[i64],
        right: &[i64],
        window_ns: u64,
    ) -> (Vec<(i64, i64)>, PairingStats, usize, usize) {
        let mut pairs = Vec::new();
        let mut stats = PairingStats::default();
        let mut left_idx = 0usize;
        let mut right_idx = 0usize;

        while left_idx < left.len() && right_idx < right.len() {
            let left_ts = left[left_idx];
            let mut best_offset = 0usize;
            let mut best_ts = right[right_idx];
            let mut best_delta = best_ts.abs_diff(left_ts);
            for (offset, &right_ts) in right[right_idx + 1..].iter().enumerate() {
                let delta = right_ts.abs_diff(left_ts);
                if delta < best_delta {
                    best_offset = offset + 1;
                    best_ts = right_ts;
                    best_delta = delta;
                }
            }

            if best_delta <= window_ns {
                stats.dropped_right += u64::try_from(best_offset).expect("small reference count");
                right_idx += best_offset + 1;
                left_idx += 1;
                stats.paired += 1;
                pairs.push((left_ts, best_ts));
            } else if best_ts < left_ts {
                let discarded = best_offset + 1;
                let discarded = u64::try_from(discarded).expect("small reference count");
                right_idx += best_offset + 1;
                stats.dropped_right += discarded;
                stats.outside_window += discarded;
            } else {
                left_idx += 1;
                stats.dropped_left += 1;
                stats.outside_window += 1;
            }
        }

        (pairs, stats, left.len() - left_idx, right.len() - right_idx)
    }

    #[test]
    fn zero_window_accepts_only_exact_timestamps() {
        let window = PairingWindowNs::new(0).expect("zero is a valid exact-sync window");
        let exact = StereoPair::try_new(
            frame(SensorId::StereoLeft, 42, 1),
            frame(SensorId::StereoRight, 42, 2),
            window,
        );
        assert!(exact.is_ok());

        let offset = StereoPair::try_new(
            frame(SensorId::StereoLeft, 42, 3),
            frame(SensorId::StereoRight, 43, 4),
            window,
        );
        assert!(matches!(offset, Err(PairError::TimestampDelta { .. })));
    }

    #[test]
    fn timestamp_delta_handles_full_i64_range() {
        let window = PairingWindowNs::new(i64::MAX).expect("valid window");
        let result = StereoPair::try_new(
            frame(SensorId::StereoLeft, i64::MIN, 1),
            frame(SensorId::StereoRight, i64::MAX, 2),
            window,
        );

        let Err(PairError::TimestampDelta {
            delta_ns,
            max_delta_ns,
        }) = result
        else {
            panic!("full-range timestamps should exceed the pairing window");
        };
        assert_eq!(delta_ns, u64::MAX);
        assert_eq!(max_delta_ns, i64::MAX as u64);
    }

    #[test]
    fn unsigned_pairing_window_parser_rejects_values_above_i64_max() {
        assert_eq!(
            PairingWindowNs::try_from_u64(i64::MAX as u64)
                .expect("maximum signed window")
                .as_ns(),
            i64::MAX
        );
        assert!(matches!(
            PairingWindowNs::try_from_u64(i64::MAX as u64 + 1),
            Err(PairingConfigError::WindowTooLarge { window_ns })
                if window_ns == i64::MAX as u64 + 1
        ));
    }

    #[test]
    fn pending_left_is_capped() {
        let window = PairingWindowNs::new(5_000_000).expect("valid pairing window");
        let mut pairer =
            StereoPairer::new_with_max_pending(window, 2).expect("valid pairer capacity");
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
        assert_eq!(pairer.max_pending_per_side(), 2);
    }

    #[test]
    fn pending_right_is_capped() {
        let window = PairingWindowNs::new(5_000_000).expect("valid pairing window");
        let mut pairer =
            StereoPairer::new_with_max_pending(window, 2).expect("valid pairer capacity");
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
        let mut pairer =
            StereoPairer::new_with_max_pending(window, 1).expect("valid pairer capacity");
        pairer
            .push_left(frame(SensorId::StereoLeft, 10, 1))
            .expect("left frame");
        assert!(pairer.next_pair().is_none());
    }

    #[test]
    fn selected_later_right_supersedes_earlier_rights_without_crossing() {
        let window = PairingWindowNs::new(10).expect("valid pairing window");
        let mut pairer = StereoPairer::new(window);
        for (timestamp_ns, frame_id) in [(8, 1), (10, 2)] {
            pairer
                .push_left(frame(SensorId::StereoLeft, timestamp_ns, frame_id))
                .expect("monotonic left frame");
        }
        for (timestamp_ns, frame_id) in [(0, 3), (9, 4)] {
            pairer
                .push_right(frame(SensorId::StereoRight, timestamp_ns, frame_id))
                .expect("monotonic right frame");
        }

        let first = pairer.next_pair().expect("nearest pending pair");
        assert_eq!(first.left().timestamp().as_nanos(), 8);
        assert_eq!(first.right().timestamp().as_nanos(), 9);
        assert!(
            pairer.next_pair().is_none(),
            "superseded right must not cross"
        );
        assert_eq!(
            pairer.stats(),
            PairingStats {
                paired: 1,
                dropped_left: 0,
                dropped_right: 1,
                outside_window: 0,
            }
        );

        pairer
            .push_right(frame(SensorId::StereoRight, 11, 5))
            .expect("future monotonic right frame");
        let second = pairer.next_pair().expect("monotonic continuation");
        assert_eq!(second.left().timestamp().as_nanos(), 10);
        assert_eq!(second.right().timestamp().as_nanos(), 11);
    }

    #[test]
    fn old_out_of_window_prefix_is_counted_per_discarded_frame() {
        let window = PairingWindowNs::new(1).expect("valid pairing window");
        let mut pairer = StereoPairer::new(window);
        pairer
            .push_left(frame(SensorId::StereoLeft, 10, 1))
            .expect("left frame");
        for (timestamp_ns, frame_id) in [(0, 2), (1, 3), (2, 4)] {
            pairer
                .push_right(frame(SensorId::StereoRight, timestamp_ns, frame_id))
                .expect("monotonic right frame");
        }

        assert!(pairer.next_pair().is_none());
        assert_eq!(
            pairer.stats(),
            PairingStats {
                paired: 0,
                dropped_left: 0,
                dropped_right: 3,
                outside_window: 3,
            }
        );

        pairer
            .push_right(frame(SensorId::StereoRight, 10, 5))
            .expect("future monotonic right frame");
        assert!(
            pairer.next_pair().is_some(),
            "left frame must remain pending"
        );
    }

    #[test]
    fn nearest_tie_uses_the_earlier_right_timestamp() {
        let window = PairingWindowNs::new(5).expect("valid pairing window");
        let mut pairer = StereoPairer::new(window);
        pairer
            .push_left(frame(SensorId::StereoLeft, 0, 1))
            .expect("left frame");
        pairer
            .push_right(frame(SensorId::StereoRight, -5, 2))
            .expect("earlier right frame");
        pairer
            .push_right(frame(SensorId::StereoRight, 5, 3))
            .expect("later right frame");

        let pair = pairer.next_pair().expect("in-window tie");
        assert_eq!(pair.right().timestamp().as_nanos(), -5);
        assert_eq!(pairer.stats().dropped_right, 0);
    }

    #[test]
    fn full_range_right_prefix_does_not_overflow_distance_ordering() {
        let window = PairingWindowNs::new(0).expect("exact pairing window");
        let mut pairer = StereoPairer::new(window);
        pairer
            .push_left(frame(SensorId::StereoLeft, i64::MAX, 1))
            .expect("left frame");
        pairer
            .push_right(frame(SensorId::StereoRight, i64::MIN, 2))
            .expect("minimum right timestamp");
        pairer
            .push_right(frame(SensorId::StereoRight, i64::MAX, 3))
            .expect("maximum right timestamp");

        let pair = pairer.next_pair().expect("exact maximum timestamp pair");
        assert_eq!(pair.timestamp_delta_ns(), 0);
        assert_eq!(pairer.stats().dropped_right, 1);
        assert_eq!(pairer.stats().outside_window, 0);
    }

    #[test]
    fn nearest_pending_policy_matches_exhaustive_small_reference() {
        const VALUES: [i64; 5] = [-2, -1, 0, 1, 2];
        for left_mask in 0usize..1 << VALUES.len() {
            let left: Vec<i64> = VALUES
                .iter()
                .enumerate()
                .filter_map(|(index, &value)| ((left_mask >> index) & 1 == 1).then_some(value))
                .collect();
            for right_mask in 0usize..1 << VALUES.len() {
                let right: Vec<i64> = VALUES
                    .iter()
                    .enumerate()
                    .filter_map(|(index, &value)| ((right_mask >> index) & 1 == 1).then_some(value))
                    .collect();
                for window_ns in 0..=4 {
                    let window = PairingWindowNs::new(window_ns).expect("small valid window");
                    let mut pairer = StereoPairer::new(window);
                    for (frame_id, &timestamp_ns) in left.iter().enumerate() {
                        pairer
                            .push_left(frame(
                                SensorId::StereoLeft,
                                timestamp_ns,
                                u64::try_from(frame_id).expect("small frame id"),
                            ))
                            .expect("strictly increasing left timestamps");
                    }
                    for (frame_id, &timestamp_ns) in right.iter().enumerate() {
                        pairer
                            .push_right(frame(
                                SensorId::StereoRight,
                                timestamp_ns,
                                u64::try_from(frame_id).expect("small frame id"),
                            ))
                            .expect("strictly increasing right timestamps");
                    }

                    let mut actual = Vec::new();
                    while let Some(pair) = pairer.next_pair() {
                        actual.push((
                            pair.left().timestamp().as_nanos(),
                            pair.right().timestamp().as_nanos(),
                        ));
                        assert!(pair.timestamp_delta_ns() <= window.as_u64());
                    }
                    let (expected, expected_stats, pending_left, pending_right) =
                        reference_pairing(&left, &right, window.as_u64());
                    assert_eq!(
                        actual, expected,
                        "left={left:?}, right={right:?}, window={window_ns}"
                    );
                    assert_eq!(pairer.stats(), expected_stats);
                    assert_eq!(pairer.left.len(), pending_left);
                    assert_eq!(pairer.right.len(), pending_right);
                    assert!(
                        actual
                            .windows(2)
                            .all(|pairs| { pairs[0].0 < pairs[1].0 && pairs[0].1 < pairs[1].1 })
                    );

                    let paired = actual.len();
                    assert_eq!(
                        left.len(),
                        paired
                            + usize::try_from(pairer.stats().dropped_left).expect("small counter")
                            + pairer.left.len()
                    );
                    assert_eq!(
                        right.len(),
                        paired
                            + usize::try_from(pairer.stats().dropped_right).expect("small counter")
                            + pairer.right.len()
                    );
                }
            }
        }
    }

    #[test]
    fn zero_pending_capacity_is_rejected_instead_of_clamped() {
        let window = PairingWindowNs::new(0).expect("valid exact window");
        assert!(matches!(
            StereoPairer::new_with_max_pending(window, 0),
            Err(PairingConfigError::ZeroMaxPendingPerSide)
        ));
    }

    #[test]
    fn lifetime_statistics_saturate_instead_of_wrapping() {
        let window = PairingWindowNs::new(0).expect("valid exact window");
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
        let window = PairingWindowNs::new(0).expect("exact window");
        let mut pairer = StereoPairer::new_with_max_pending(window, 1).expect("valid capacity");
        pairer
            .push_left(frame(SensorId::StereoLeft, 10, 1))
            .expect("valid left frame");
        let before = pairer.stats();

        assert_eq!(
            pairer.push_left(frame(SensorId::StereoRight, 10, 99)),
            Err(PairingInputError::SensorMismatch {
                expected: SensorId::StereoLeft,
                actual: SensorId::StereoRight,
            })
        );
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
        let window = PairingWindowNs::new(0).expect("exact window");
        let mut pairer = StereoPairer::new(window);
        pairer
            .push_left(frame_with_dimensions(SensorId::StereoLeft, 10, 1, 2, 2))
            .expect("valid left frame");
        let before = pairer.stats();
        let expected = FrameDimensions::try_new(2, 2).expect("dimensions");
        let actual = FrameDimensions::try_new(3, 2).expect("dimensions");
        assert_eq!(pairer.dimensions(), Some(expected));

        assert_eq!(
            pairer.push_right(frame_with_dimensions(SensorId::StereoRight, 10, 99, 3, 2,)),
            Err(PairingInputError::DimensionMismatch { expected, actual })
        );
        assert_eq!(pairer.stats(), before);
        assert_eq!(pairer.dimensions(), Some(expected));

        pairer
            .push_right(frame(SensorId::StereoRight, 10, 2))
            .expect("valid right frame");
        assert!(pairer.next_pair().is_some());
    }

    #[test]
    fn regressing_timestamp_is_rejected_before_pairing_state_mutation() {
        let window = PairingWindowNs::new(0).expect("exact window");
        let mut pairer = StereoPairer::new(window);
        pairer
            .push_left(frame(SensorId::StereoLeft, 10, 1))
            .expect("first left frame");
        pairer
            .push_right(frame(SensorId::StereoRight, 20, 2))
            .expect("first right frame");
        assert!(pairer.next_pair().is_none());
        let before = pairer.stats();

        assert_eq!(
            pairer.push_right(frame(SensorId::StereoRight, 10, 99)),
            Err(PairingInputError::NonMonotonicTimestamp {
                sensor: SensorId::StereoRight,
                previous_ns: 20,
                current_ns: 10,
            })
        );
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
