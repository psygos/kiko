use crate::{ImuBatch, StereoPair, Timestamp};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CaptureId(u64);

impl CaptureId {
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    pub fn as_u64(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CaptureInterval {
    start_exclusive: Option<Timestamp>,
    end_inclusive: Timestamp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureIntervalError {
    NonIncreasing {
        start_exclusive: Timestamp,
        end_inclusive: Timestamp,
    },
}

impl std::fmt::Display for CaptureIntervalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CaptureIntervalError::NonIncreasing {
                start_exclusive,
                end_inclusive,
            } => write!(
                f,
                "capture interval must satisfy start < end: start={} end={}",
                start_exclusive.as_nanos(),
                end_inclusive.as_nanos()
            ),
        }
    }
}

impl std::error::Error for CaptureIntervalError {}

impl CaptureInterval {
    pub fn new(
        start_exclusive: Option<Timestamp>,
        end_inclusive: Timestamp,
    ) -> Result<Self, CaptureIntervalError> {
        if let Some(start_exclusive) = start_exclusive
            && start_exclusive.as_nanos() >= end_inclusive.as_nanos()
        {
            return Err(CaptureIntervalError::NonIncreasing {
                start_exclusive,
                end_inclusive,
            });
        }
        Ok(Self {
            start_exclusive,
            end_inclusive,
        })
    }

    pub fn start_exclusive(self) -> Option<Timestamp> {
        self.start_exclusive
    }

    pub fn end_inclusive(self) -> Timestamp {
        self.end_inclusive
    }

    pub fn contains(self, timestamp: Timestamp) -> bool {
        if timestamp.as_nanos() > self.end_inclusive.as_nanos() {
            return false;
        }
        match self.start_exclusive {
            Some(start) => timestamp.as_nanos() > start.as_nanos(),
            None => true,
        }
    }
}

#[derive(Clone, Debug)]
pub enum CaptureImu {
    Absent,
    Present(ImuBatch),
}

impl CaptureImu {
    pub fn absent() -> Self {
        Self::Absent
    }

    pub fn present(batch: ImuBatch) -> Self {
        Self::Present(batch)
    }

    pub fn batch(&self) -> Option<&ImuBatch> {
        match self {
            CaptureImu::Absent => None,
            CaptureImu::Present(batch) => Some(batch),
        }
    }

    pub fn sample_count(&self) -> usize {
        self.batch().map_or(0, ImuBatch::len)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureBundleError {
    InvalidInterval(CaptureIntervalError),
    IntervalEndMismatch {
        interval_end: Timestamp,
        capture_time: Timestamp,
    },
    ImuOutsideInterval {
        sample_time: Timestamp,
        interval: CaptureInterval,
    },
}

impl std::fmt::Display for CaptureBundleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CaptureBundleError::InvalidInterval(err) => {
                write!(f, "invalid capture interval: {err}")
            }
            CaptureBundleError::IntervalEndMismatch {
                interval_end,
                capture_time,
            } => write!(
                f,
                "capture interval end {} must match pair capture time {}",
                interval_end.as_nanos(),
                capture_time.as_nanos()
            ),
            CaptureBundleError::ImuOutsideInterval {
                sample_time,
                interval,
            } => write!(
                f,
                "imu sample {} lies outside capture interval ({:?}, {}]",
                sample_time.as_nanos(),
                interval.start_exclusive().map(|timestamp| timestamp.as_nanos()),
                interval.end_inclusive().as_nanos()
            ),
        }
    }
}

impl std::error::Error for CaptureBundleError {}

impl From<CaptureIntervalError> for CaptureBundleError {
    fn from(value: CaptureIntervalError) -> Self {
        Self::InvalidInterval(value)
    }
}

#[derive(Clone, Debug)]
pub struct CaptureBundle {
    capture_id: CaptureId,
    pair: StereoPair,
    interval: CaptureInterval,
    imu: CaptureImu,
}

impl CaptureBundle {
    pub fn new(
        capture_id: CaptureId,
        pair: StereoPair,
        interval: CaptureInterval,
        imu: CaptureImu,
    ) -> Result<Self, CaptureBundleError> {
        let capture_time = pair.capture_time();
        if interval.end_inclusive() != capture_time {
            return Err(CaptureBundleError::IntervalEndMismatch {
                interval_end: interval.end_inclusive(),
                capture_time,
            });
        }

        if let Some(batch) = imu.batch() {
            for sample in batch.samples() {
                if !interval.contains(sample.timestamp()) {
                    return Err(CaptureBundleError::ImuOutsideInterval {
                        sample_time: sample.timestamp(),
                        interval,
                    });
                }
            }
        }

        Ok(Self {
            capture_id,
            pair,
            interval,
            imu,
        })
    }

    pub fn visual_only(
        capture_id: CaptureId,
        pair: StereoPair,
        previous_capture_time: Option<Timestamp>,
    ) -> Result<Self, CaptureBundleError> {
        let interval = CaptureInterval::new(previous_capture_time, pair.capture_time())?;
        Self::new(capture_id, pair, interval, CaptureImu::Absent)
    }

    pub fn capture_id(&self) -> CaptureId {
        self.capture_id
    }

    pub fn capture_time(&self) -> Timestamp {
        self.interval.end_inclusive()
    }

    pub fn pair(&self) -> &StereoPair {
        &self.pair
    }

    pub fn interval(&self) -> CaptureInterval {
        self.interval
    }

    pub fn imu(&self) -> &CaptureImu {
        &self.imu
    }

    pub fn into_parts(self) -> (CaptureId, StereoPair, CaptureInterval, CaptureImu) {
        (self.capture_id, self.pair, self.interval, self.imu)
    }
}

pub trait CaptureSource {
    type Error;

    fn next_capture(&mut self) -> Option<Result<CaptureBundle, Self::Error>>;

    #[allow(dead_code)]
    fn captures(self) -> Captures<Self>
    where
        Self: Sized,
    {
        Captures { source: self }
    }
}

pub struct Captures<S> {
    source: S,
}

impl<S: CaptureSource> Iterator for Captures<S> {
    type Item = Result<CaptureBundle, S::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        self.source.next_capture()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Frame, SensorId};

    fn frame(sensor: SensorId, frame_id: u64, timestamp_ns: i64) -> Frame {
        Frame::new(
            sensor,
            crate::FrameId::new(frame_id),
            Timestamp::from_nanos(timestamp_ns),
            2,
            2,
            vec![0; 4],
        )
        .expect("frame")
    }

    fn stereo_pair(left_ts: i64, right_ts: i64) -> StereoPair {
        StereoPair::try_new(
            frame(SensorId::StereoLeft, 1, left_ts),
            frame(SensorId::StereoRight, 2, right_ts),
            crate::PairingWindowNs::new(10).expect("window"),
        )
        .expect("stereo pair")
    }

    #[test]
    fn capture_interval_rejects_non_increasing_bounds() {
        let err = CaptureInterval::new(
            Some(Timestamp::from_nanos(10)),
            Timestamp::from_nanos(10),
        )
        .expect_err("equal bounds should be rejected");
        assert_eq!(
            err,
            CaptureIntervalError::NonIncreasing {
                start_exclusive: Timestamp::from_nanos(10),
                end_inclusive: Timestamp::from_nanos(10),
            }
        );
    }

    #[test]
    fn stereo_pair_capture_time_uses_midpoint() {
        let pair = stereo_pair(100, 106);
        assert_eq!(pair.capture_time(), Timestamp::from_nanos(103));
    }

    #[test]
    fn capture_bundle_accepts_visual_only_interval() {
        let pair = stereo_pair(100, 104);
        let bundle = CaptureBundle::visual_only(CaptureId::new(7), pair, Some(Timestamp::from_nanos(90)))
            .expect("visual-only bundle");
        assert_eq!(bundle.capture_id(), CaptureId::new(7));
        assert_eq!(bundle.capture_time(), Timestamp::from_nanos(102));
        assert!(matches!(bundle.imu(), CaptureImu::Absent));
    }

    #[test]
    fn capture_bundle_rejects_interval_end_mismatch() {
        let pair = stereo_pair(100, 104);
        let interval = CaptureInterval::new(Some(Timestamp::from_nanos(90)), Timestamp::from_nanos(101))
            .expect("interval");
        let err = CaptureBundle::new(CaptureId::new(1), pair, interval, CaptureImu::Absent)
            .expect_err("bundle should reject wrong interval end");
        assert_eq!(
            err,
            CaptureBundleError::IntervalEndMismatch {
                interval_end: Timestamp::from_nanos(101),
                capture_time: Timestamp::from_nanos(102),
            }
        );
    }

    #[test]
    fn capture_bundle_rejects_imu_sample_at_interval_start() {
        let pair = stereo_pair(100, 104);
        let interval =
            CaptureInterval::new(Some(Timestamp::from_nanos(90)), pair.capture_time()).expect("interval");
        let batch = ImuBatch::new(vec![
            crate::ImuSample::new(Timestamp::from_nanos(90), [0.0; 3], [0.0; 3]).expect("sample"),
        ])
        .expect("batch");
        let err = CaptureBundle::new(
            CaptureId::new(2),
            pair,
            interval,
            CaptureImu::present(batch),
        )
        .expect_err("sample at interval start should be rejected");
        assert_eq!(
            err,
            CaptureBundleError::ImuOutsideInterval {
                sample_time: Timestamp::from_nanos(90),
                interval,
            }
        );
    }

    #[test]
    fn capture_bundle_rejects_imu_sample_after_interval_end() {
        let pair = stereo_pair(100, 104);
        let interval =
            CaptureInterval::new(Some(Timestamp::from_nanos(90)), pair.capture_time()).expect("interval");
        let batch = ImuBatch::new(vec![
            crate::ImuSample::new(Timestamp::from_nanos(103), [0.0; 3], [0.0; 3]).expect("sample"),
        ])
        .expect("batch");
        let err = CaptureBundle::new(
            CaptureId::new(3),
            pair,
            interval,
            CaptureImu::present(batch),
        )
        .expect_err("sample after interval end should be rejected");
        assert_eq!(
            err,
            CaptureBundleError::ImuOutsideInterval {
                sample_time: Timestamp::from_nanos(103),
                interval,
            }
        );
    }

    #[test]
    fn capture_bundle_accepts_imu_samples_inside_interval() {
        let pair = stereo_pair(100, 104);
        let interval =
            CaptureInterval::new(Some(Timestamp::from_nanos(90)), pair.capture_time()).expect("interval");
        let batch = ImuBatch::new(vec![
            crate::ImuSample::new(Timestamp::from_nanos(91), [0.0; 3], [0.0; 3]).expect("sample 0"),
            crate::ImuSample::new(Timestamp::from_nanos(102), [1.0; 3], [2.0; 3]).expect("sample 1"),
        ])
        .expect("batch");
        let bundle = CaptureBundle::new(
            CaptureId::new(4),
            pair,
            interval,
            CaptureImu::present(batch),
        )
        .expect("bundle");
        assert_eq!(bundle.imu().sample_count(), 2);
    }

    struct SingleCaptureSource {
        capture: Option<CaptureBundle>,
    }

    impl CaptureSource for SingleCaptureSource {
        type Error = ();

        fn next_capture(&mut self) -> Option<Result<CaptureBundle, Self::Error>> {
            self.capture.take().map(Ok)
        }
    }

    #[test]
    fn capture_source_iterator_yields_captures_in_order() {
        let pair = stereo_pair(100, 104);
        let bundle =
            CaptureBundle::visual_only(CaptureId::new(9), pair, None).expect("bundle");
        let source = SingleCaptureSource {
            capture: Some(bundle),
        };
        let captures: Vec<_> = source.captures().collect();
        assert_eq!(captures.len(), 1);
        let capture = captures[0].as_ref().expect("capture ok");
        assert_eq!(capture.capture_id(), CaptureId::new(9));
    }
}
