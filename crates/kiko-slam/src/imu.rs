use std::sync::Arc;

use crate::{Pose64, Timestamp};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ImuSampleError {
    NonFiniteAccel { axis: usize, value: f64 },
    NonFiniteGyro { axis: usize, value: f64 },
}

impl std::fmt::Display for ImuSampleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImuSampleError::NonFiniteAccel { axis, value } => {
                write!(f, "imu accel axis {axis} must be finite, got {value}")
            }
            ImuSampleError::NonFiniteGyro { axis, value } => {
                write!(f, "imu gyro axis {axis} must be finite, got {value}")
            }
        }
    }
}

impl std::error::Error for ImuSampleError {}

#[derive(Clone, Debug)]
pub struct ImuSample {
    timestamp: Timestamp,
    accel_mps2: [f64; 3],
    gyro_radps: [f64; 3],
}

impl ImuSample {
    pub fn new(
        timestamp: Timestamp,
        accel_mps2: [f64; 3],
        gyro_radps: [f64; 3],
    ) -> Result<Self, ImuSampleError> {
        for (axis, value) in accel_mps2.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(ImuSampleError::NonFiniteAccel { axis, value });
            }
        }
        for (axis, value) in gyro_radps.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(ImuSampleError::NonFiniteGyro { axis, value });
            }
        }
        Ok(Self {
            timestamp,
            accel_mps2,
            gyro_radps,
        })
    }

    pub fn timestamp(&self) -> Timestamp {
        self.timestamp
    }

    pub fn accel_mps2(&self) -> [f64; 3] {
        self.accel_mps2
    }

    pub fn gyro_radps(&self) -> [f64; 3] {
        self.gyro_radps
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ImuBias {
    pub accel: [f64; 3],
    pub gyro: [f64; 3],
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImuNoiseModel {
    pub accel_noise_density: f64,
    pub gyro_noise_density: f64,
    pub accel_random_walk: f64,
    pub gyro_random_walk: f64,
}

#[derive(Clone, Debug)]
pub struct ImuExtrinsics {
    pub t_cam_imu: Pose64,
    pub time_offset_ns: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImuBatchError {
    Empty,
    NonIncreasingTimestamps {
        previous: Timestamp,
        current: Timestamp,
    },
}

impl std::fmt::Display for ImuBatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImuBatchError::Empty => write!(f, "imu batch must contain at least one sample"),
            ImuBatchError::NonIncreasingTimestamps { previous, current } => write!(
                f,
                "imu batch timestamps must be strictly increasing: previous={} current={}",
                previous.as_nanos(),
                current.as_nanos()
            ),
        }
    }
}

impl std::error::Error for ImuBatchError {}

#[derive(Clone, Debug)]
pub struct ImuBatch {
    samples: Arc<[ImuSample]>,
}

impl ImuBatch {
    pub fn new(samples: Vec<ImuSample>) -> Result<Self, ImuBatchError> {
        if samples.is_empty() {
            return Err(ImuBatchError::Empty);
        }
        let mut previous = samples[0].timestamp();
        for sample in samples.iter().skip(1) {
            let current = sample.timestamp();
            if current.as_nanos() <= previous.as_nanos() {
                return Err(ImuBatchError::NonIncreasingTimestamps { previous, current });
            }
            previous = current;
        }
        Ok(Self {
            samples: Arc::from(samples.into_boxed_slice()),
        })
    }

    pub fn samples(&self) -> &[ImuSample] {
        self.samples.as_ref()
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn start_time(&self) -> Timestamp {
        self.samples[0].timestamp()
    }

    pub fn end_time(&self) -> Timestamp {
        self.samples[self.samples.len() - 1].timestamp()
    }

    pub fn dt_seconds(&self) -> f64 {
        self.end_time().seconds_since(self.start_time()).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn imu_sample_rejects_non_finite_accel() {
        let err = ImuSample::new(
            Timestamp::from_nanos(1),
            [f64::NAN, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        )
        .expect_err("nan accel should be rejected");
        match err {
            ImuSampleError::NonFiniteAccel { axis, value } => {
                assert_eq!(axis, 0);
                assert!(value.is_nan());
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn imu_sample_rejects_non_finite_gyro() {
        let err = ImuSample::new(
            Timestamp::from_nanos(1),
            [0.0, 0.0, 0.0],
            [0.0, f64::INFINITY, 0.0],
        )
        .expect_err("inf gyro should be rejected");
        assert_eq!(
            err,
            ImuSampleError::NonFiniteGyro {
                axis: 1,
                value: f64::INFINITY
            }
        );
    }

    #[test]
    fn imu_batch_rejects_empty() {
        let err = ImuBatch::new(Vec::new()).expect_err("empty batch should be rejected");
        assert_eq!(err, ImuBatchError::Empty);
    }

    #[test]
    fn imu_batch_rejects_non_increasing_timestamps() {
        let a = ImuSample::new(Timestamp::from_nanos(10), [0.0; 3], [0.0; 3]).expect("sample a");
        let b = ImuSample::new(Timestamp::from_nanos(10), [0.0; 3], [0.0; 3]).expect("sample b");
        let err = ImuBatch::new(vec![a, b]).expect_err("duplicate timestamps should fail");
        assert_eq!(
            err,
            ImuBatchError::NonIncreasingTimestamps {
                previous: Timestamp::from_nanos(10),
                current: Timestamp::from_nanos(10),
            }
        );
    }

    #[test]
    fn imu_batch_reports_time_span() {
        let samples = vec![
            ImuSample::new(Timestamp::from_nanos(10), [0.0; 3], [0.0; 3]).expect("sample 0"),
            ImuSample::new(Timestamp::from_nanos(30), [1.0; 3], [2.0; 3]).expect("sample 1"),
            ImuSample::new(Timestamp::from_nanos(70), [3.0; 3], [4.0; 3]).expect("sample 2"),
        ];
        let batch = ImuBatch::new(samples).expect("batch");
        assert_eq!(batch.start_time(), Timestamp::from_nanos(10));
        assert_eq!(batch.end_time(), Timestamp::from_nanos(70));
        assert!((batch.dt_seconds() - 60e-9).abs() < 1e-15);
    }

    #[test]
    fn imu_bias_default_is_zero() {
        let bias = ImuBias::default();
        assert_eq!(bias.accel, [0.0; 3]);
        assert_eq!(bias.gyro, [0.0; 3]);
    }
}
