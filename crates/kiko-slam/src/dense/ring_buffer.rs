use std::collections::VecDeque;
use std::num::NonZeroUsize;

use crate::{DepthImage, Timestamp};

/// Fixed-capacity ring buffer of recent depth frames for timestamp association.
///
/// The capture thread pushes depth frames as they arrive. When a keyframe is
/// created, the command mapper queries the buffer for the depth frame closest
/// to the stereo pair timestamp (within a configurable window).
pub struct DepthRingBuffer {
    entries: VecDeque<DepthImage>,
    capacity: NonZeroUsize,
    /// Track severe reorder events for diagnostics.
    reorder_warnings: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DepthRingBufferError {
    ZeroCapacity,
}

impl std::fmt::Display for DepthRingBufferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroCapacity => write!(f, "depth ring buffer capacity must be > 0"),
        }
    }
}

impl std::error::Error for DepthRingBufferError {}

impl DepthRingBuffer {
    pub fn try_new(capacity: usize) -> Result<Self, DepthRingBufferError> {
        let capacity = NonZeroUsize::new(capacity).ok_or(DepthRingBufferError::ZeroCapacity)?;
        Ok(Self {
            entries: VecDeque::with_capacity(capacity.get()),
            capacity,
            reorder_warnings: 0,
        })
    }

    pub fn push(&mut self, depth: DepthImage) {
        // Warn on severe out-of-order delivery but do not reject.
        if let Some(last) = self.entries.back()
            && depth.timestamp().as_nanos() < last.timestamp().as_nanos()
        {
            self.reorder_warnings = self.reorder_warnings.saturating_add(1);
        }

        if self.entries.len() >= self.capacity.get() {
            self.entries.pop_front();
        }
        self.entries.push_back(depth);
    }

    /// Find the depth frame whose timestamp is closest to `query`, provided
    /// the distance is within `max_window_ns` nanoseconds. Returns `None` if
    /// the buffer is empty or no entry falls within the window.
    pub fn find_closest(&self, query: Timestamp, max_window_ns: i64) -> Option<DepthImage> {
        if self.entries.is_empty() {
            return None;
        }
        let max_window = u64::try_from(max_window_ns).ok()?;

        let query_ns = query.as_nanos();
        let mut best: Option<(u64, usize)> = None;

        for (idx, entry) in self.entries.iter().enumerate() {
            let delta = entry.timestamp().as_nanos().abs_diff(query_ns);
            match best {
                Some((best_delta, _)) if delta < best_delta => {
                    best = Some((delta, idx));
                }
                None => {
                    best = Some((delta, idx));
                }
                _ => {}
            }
        }

        let (delta, idx) = best?;
        if delta <= max_window {
            self.entries.get(idx).cloned()
        } else {
            None
        }
    }

    pub fn reorder_warnings(&self) -> u64 {
        self.reorder_warnings
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FrameId;
    use crate::test_helpers::make_depth_image;

    fn ts(ns: i64) -> Timestamp {
        Timestamp::from_nanos(ns)
    }

    fn depth_at(t_ns: i64) -> DepthImage {
        make_depth_image(FrameId::new(0), ts(t_ns), 2, 2, 1.0)
    }

    fn buffer(capacity: usize) -> DepthRingBuffer {
        DepthRingBuffer::try_new(capacity).expect("nonzero test capacity")
    }

    #[test]
    fn empty_returns_none() {
        let buf = buffer(4);
        assert!(buf.find_closest(ts(100), 10).is_none());
    }

    #[test]
    fn single_entry_within_window() {
        let mut buf = buffer(4);
        buf.push(depth_at(100));
        let result = buf.find_closest(ts(105), 10);
        assert!(result.is_some());
        assert_eq!(result.unwrap().timestamp().as_nanos(), 100);
    }

    #[test]
    fn single_entry_outside_window() {
        let mut buf = buffer(4);
        buf.push(depth_at(100));
        assert!(buf.find_closest(ts(200), 10).is_none());
    }

    #[test]
    fn picks_closest_of_two() {
        let mut buf = buffer(4);
        buf.push(depth_at(100));
        buf.push(depth_at(200));
        let result = buf.find_closest(ts(160), 100).unwrap();
        assert_eq!(result.timestamp().as_nanos(), 200);
    }

    #[test]
    fn boundary_exact_match() {
        let mut buf = buffer(4);
        buf.push(depth_at(100));
        let result = buf.find_closest(ts(100), 0).unwrap();
        assert_eq!(result.timestamp().as_nanos(), 100);
    }

    #[test]
    fn boundary_at_window_edge() {
        let mut buf = buffer(4);
        buf.push(depth_at(100));
        // query at 110, window 10 => delta=10, should be found (inclusive)
        assert!(buf.find_closest(ts(110), 10).is_some());
        // query at 111, window 10 => delta=11, should not be found
        assert!(buf.find_closest(ts(111), 10).is_none());
    }

    #[test]
    fn eviction_at_capacity() {
        let mut buf = buffer(3);
        buf.push(depth_at(100));
        buf.push(depth_at(200));
        buf.push(depth_at(300));
        assert_eq!(buf.len(), 3);
        buf.push(depth_at(400));
        assert_eq!(buf.len(), 3);
        // oldest (100) should be evicted
        assert!(buf.find_closest(ts(100), 0).is_none());
        assert!(buf.find_closest(ts(200), 0).is_some());
        assert!(buf.find_closest(ts(400), 0).is_some());
    }

    #[test]
    fn out_of_order_allowed_with_warning() {
        let mut buf = buffer(4);
        buf.push(depth_at(200));
        buf.push(depth_at(100));
        assert_eq!(buf.reorder_warnings(), 1);
        // Both entries are kept and queryable
        assert!(buf.find_closest(ts(100), 0).is_some());
        assert!(buf.find_closest(ts(200), 0).is_some());
    }

    #[test]
    fn large_timestamp_delta_does_not_overflow() {
        let mut buf = buffer(2);
        buf.push(depth_at(i64::MIN + 1));
        assert!(buf.find_closest(ts(i64::MAX), i64::MAX).is_none());
    }

    #[test]
    fn negative_window_is_rejected() {
        let mut buf = buffer(2);
        buf.push(depth_at(100));
        assert!(buf.find_closest(ts(100), -1).is_none());
    }

    #[test]
    fn zero_capacity_is_rejected_instead_of_clamped() {
        assert!(matches!(
            DepthRingBuffer::try_new(0),
            Err(DepthRingBufferError::ZeroCapacity)
        ));
    }
}
