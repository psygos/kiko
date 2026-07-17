use std::collections::VecDeque;
use std::num::NonZeroUsize;

use crate::{DepthImage, Timestamp};

/// Nonnegative inclusive timestamp-distance bound for depth association.
///
/// Nanoseconds are stored as `u64` because the absolute difference between
/// two signed device timestamps can span the full `u64` domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct DepthAssociationWindow(u64);

impl DepthAssociationWindow {
    pub const fn from_nanos(nanoseconds: u64) -> Self {
        Self(nanoseconds)
    }

    pub const fn as_nanos(self) -> u64 {
        self.0
    }
}

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
    /// the distance is within the inclusive `window`. Returns `None` if the
    /// buffer is empty or no entry falls within the window. Equidistant frames
    /// prefer the earlier timestamp; duplicate timestamps retain arrival order.
    pub fn find_closest(
        &self,
        query: Timestamp,
        window: DepthAssociationWindow,
    ) -> Option<DepthImage> {
        if self.entries.is_empty() {
            return None;
        }

        let query_ns = query.as_nanos();
        let mut best: Option<(u64, i64, usize)> = None;

        for (idx, entry) in self.entries.iter().enumerate() {
            let timestamp_ns = entry.timestamp().as_nanos();
            let delta = timestamp_ns.abs_diff(query_ns);
            match best {
                Some((best_delta, best_timestamp_ns, _))
                    if delta < best_delta
                        || (delta == best_delta && timestamp_ns < best_timestamp_ns) =>
                {
                    best = Some((delta, timestamp_ns, idx));
                }
                None => {
                    best = Some((delta, timestamp_ns, idx));
                }
                _ => {}
            }
        }

        let (delta, _, idx) = best?;
        if delta <= window.as_nanos() {
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

    fn identified_depth(frame_id: u64, timestamp_ns: i64) -> DepthImage {
        make_depth_image(FrameId::new(frame_id), ts(timestamp_ns), 2, 2, 1.0)
    }

    fn buffer(capacity: usize) -> DepthRingBuffer {
        DepthRingBuffer::try_new(capacity).expect("nonzero test capacity")
    }

    fn window(nanoseconds: u64) -> DepthAssociationWindow {
        DepthAssociationWindow::from_nanos(nanoseconds)
    }

    #[test]
    fn empty_returns_none() {
        let buf = buffer(4);
        assert!(buf.find_closest(ts(100), window(10)).is_none());
    }

    #[test]
    fn single_entry_within_window() {
        let mut buf = buffer(4);
        buf.push(depth_at(100));
        let result = buf.find_closest(ts(105), window(10));
        assert!(result.is_some());
        assert_eq!(result.unwrap().timestamp().as_nanos(), 100);
    }

    #[test]
    fn single_entry_outside_window() {
        let mut buf = buffer(4);
        buf.push(depth_at(100));
        assert!(buf.find_closest(ts(200), window(10)).is_none());
    }

    #[test]
    fn picks_closest_of_two() {
        let mut buf = buffer(4);
        buf.push(depth_at(100));
        buf.push(depth_at(200));
        let result = buf.find_closest(ts(160), window(100)).unwrap();
        assert_eq!(result.timestamp().as_nanos(), 200);
    }

    #[test]
    fn equidistant_tie_prefers_earlier_timestamp_in_either_arrival_order() {
        for timestamps in [[100, 200], [200, 100]] {
            let mut buf = buffer(4);
            for timestamp in timestamps {
                buf.push(depth_at(timestamp));
            }

            let result = buf
                .find_closest(ts(150), window(100))
                .expect("both candidates are inside the window");
            assert_eq!(result.timestamp().as_nanos(), 100);
        }
    }

    #[test]
    fn duplicate_timestamp_tie_retains_arrival_order() {
        let mut buf = buffer(4);
        buf.push(identified_depth(7, 100));
        buf.push(identified_depth(8, 100));

        let result = buf
            .find_closest(ts(100), window(0))
            .expect("duplicate timestamp is an exact match");
        assert_eq!(result.frame_id(), FrameId::new(7));
    }

    #[test]
    fn boundary_exact_match() {
        let mut buf = buffer(4);
        buf.push(depth_at(100));
        let result = buf.find_closest(ts(100), window(0)).unwrap();
        assert_eq!(result.timestamp().as_nanos(), 100);
    }

    #[test]
    fn boundary_at_window_edge() {
        let mut buf = buffer(4);
        buf.push(depth_at(100));
        // query at 110, window 10 => delta=10, should be found (inclusive)
        assert!(buf.find_closest(ts(110), window(10)).is_some());
        // query at 111, window 10 => delta=11, should not be found
        assert!(buf.find_closest(ts(111), window(10)).is_none());
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
        assert!(buf.find_closest(ts(100), window(0)).is_none());
        assert!(buf.find_closest(ts(200), window(0)).is_some());
        assert!(buf.find_closest(ts(400), window(0)).is_some());
    }

    #[test]
    fn out_of_order_allowed_with_warning() {
        let mut buf = buffer(4);
        buf.push(depth_at(200));
        buf.push(depth_at(100));
        assert_eq!(buf.reorder_warnings(), 1);
        // Both entries are kept and queryable
        assert!(buf.find_closest(ts(100), window(0)).is_some());
        assert!(buf.find_closest(ts(200), window(0)).is_some());
    }

    #[test]
    fn large_timestamp_delta_does_not_overflow() {
        let mut buf = buffer(2);
        buf.push(depth_at(i64::MIN + 1));
        assert!(
            buf.find_closest(ts(i64::MAX), window(i64::MAX as u64))
                .is_none()
        );
    }

    #[test]
    fn association_window_preserves_the_full_nonnegative_domain() {
        assert_eq!(window(0).as_nanos(), 0);
        assert_eq!(window(u64::MAX).as_nanos(), u64::MAX);
    }

    #[test]
    fn zero_capacity_is_rejected_instead_of_clamped() {
        assert!(matches!(
            DepthRingBuffer::try_new(0),
            Err(DepthRingBufferError::ZeroCapacity)
        ));
    }
}
