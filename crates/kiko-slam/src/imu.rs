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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImuTimestampShiftError {
    Overflow { timestamp: Timestamp, delta_ns: i64 },
}

impl std::fmt::Display for ImuTimestampShiftError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImuTimestampShiftError::Overflow {
                timestamp,
                delta_ns,
            } => write!(
                f,
                "shifting imu timestamp {} by {}ns overflowed i64 range",
                timestamp.as_nanos(),
                delta_ns
            ),
        }
    }
}

impl std::error::Error for ImuTimestampShiftError {}

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

    pub fn shifted_timestamp_ns(&self, delta_ns: i64) -> Result<Self, ImuTimestampShiftError> {
        let shifted = (self.timestamp.as_nanos() as i128) + (delta_ns as i128);
        if !(i64::MIN as i128..=i64::MAX as i128).contains(&shifted) {
            return Err(ImuTimestampShiftError::Overflow {
                timestamp: self.timestamp,
                delta_ns,
            });
        }
        Ok(Self {
            timestamp: Timestamp::from_nanos(shifted as i64),
            accel_mps2: self.accel_mps2,
            gyro_radps: self.gyro_radps,
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ImuBias {
    pub accel: [f64; 3],
    pub gyro: [f64; 3],
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImuNoiseModel {
    accel_noise_density: f64,
    gyro_noise_density: f64,
    accel_random_walk: f64,
    gyro_random_walk: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ImuNoiseModelError {
    AccelNoiseDensityNonFinite { value: f64 },
    AccelNoiseDensityNonPositive { value: f64 },
    GyroNoiseDensityNonFinite { value: f64 },
    GyroNoiseDensityNonPositive { value: f64 },
    AccelRandomWalkNonFinite { value: f64 },
    AccelRandomWalkNonPositive { value: f64 },
    GyroRandomWalkNonFinite { value: f64 },
    GyroRandomWalkNonPositive { value: f64 },
}

impl std::fmt::Display for ImuNoiseModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImuNoiseModelError::AccelNoiseDensityNonFinite { value } => {
                write!(f, "accelerometer noise density must be finite, got {value}")
            }
            ImuNoiseModelError::AccelNoiseDensityNonPositive { value } => {
                write!(f, "accelerometer noise density must be > 0, got {value}")
            }
            ImuNoiseModelError::GyroNoiseDensityNonFinite { value } => {
                write!(f, "gyroscope noise density must be finite, got {value}")
            }
            ImuNoiseModelError::GyroNoiseDensityNonPositive { value } => {
                write!(f, "gyroscope noise density must be > 0, got {value}")
            }
            ImuNoiseModelError::AccelRandomWalkNonFinite { value } => {
                write!(f, "accelerometer random walk must be finite, got {value}")
            }
            ImuNoiseModelError::AccelRandomWalkNonPositive { value } => {
                write!(f, "accelerometer random walk must be > 0, got {value}")
            }
            ImuNoiseModelError::GyroRandomWalkNonFinite { value } => {
                write!(f, "gyroscope random walk must be finite, got {value}")
            }
            ImuNoiseModelError::GyroRandomWalkNonPositive { value } => {
                write!(f, "gyroscope random walk must be > 0, got {value}")
            }
        }
    }
}

impl std::error::Error for ImuNoiseModelError {}

impl ImuNoiseModel {
    pub fn new(
        accel_noise_density: f64,
        gyro_noise_density: f64,
        accel_random_walk: f64,
        gyro_random_walk: f64,
    ) -> Result<Self, ImuNoiseModelError> {
        validate_positive_finite(
            accel_noise_density,
            ImuNoiseModelError::AccelNoiseDensityNonFinite {
                value: accel_noise_density,
            },
            ImuNoiseModelError::AccelNoiseDensityNonPositive {
                value: accel_noise_density,
            },
        )?;
        validate_positive_finite(
            gyro_noise_density,
            ImuNoiseModelError::GyroNoiseDensityNonFinite {
                value: gyro_noise_density,
            },
            ImuNoiseModelError::GyroNoiseDensityNonPositive {
                value: gyro_noise_density,
            },
        )?;
        validate_positive_finite(
            accel_random_walk,
            ImuNoiseModelError::AccelRandomWalkNonFinite {
                value: accel_random_walk,
            },
            ImuNoiseModelError::AccelRandomWalkNonPositive {
                value: accel_random_walk,
            },
        )?;
        validate_positive_finite(
            gyro_random_walk,
            ImuNoiseModelError::GyroRandomWalkNonFinite {
                value: gyro_random_walk,
            },
            ImuNoiseModelError::GyroRandomWalkNonPositive {
                value: gyro_random_walk,
            },
        )?;
        Ok(Self {
            accel_noise_density,
            gyro_noise_density,
            accel_random_walk,
            gyro_random_walk,
        })
    }

    pub fn accel_noise_density(&self) -> f64 {
        self.accel_noise_density
    }

    pub fn gyro_noise_density(&self) -> f64 {
        self.gyro_noise_density
    }

    pub fn accel_random_walk(&self) -> f64 {
        self.accel_random_walk
    }

    pub fn gyro_random_walk(&self) -> f64 {
        self.gyro_random_walk
    }
}

#[derive(Clone, Debug)]
pub struct ImuExtrinsics {
    t_cam_imu: Pose64,
    time_offset_ns: i64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ImuExtrinsicsError {
    NonFiniteRotation { row: usize, col: usize, value: f64 },
    NonFiniteTranslation { axis: usize, value: f64 },
}

impl std::fmt::Display for ImuExtrinsicsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImuExtrinsicsError::NonFiniteRotation { row, col, value } => write!(
                f,
                "camera-imu rotation[{row}][{col}] must be finite, got {value}"
            ),
            ImuExtrinsicsError::NonFiniteTranslation { axis, value } => {
                write!(
                    f,
                    "camera-imu translation axis {axis} must be finite, got {value}"
                )
            }
        }
    }
}

impl std::error::Error for ImuExtrinsicsError {}

impl ImuExtrinsics {
    pub fn new(t_cam_imu: Pose64, time_offset_ns: i64) -> Result<Self, ImuExtrinsicsError> {
        let rotation = t_cam_imu.rotation();
        for (row_idx, row) in rotation.iter().enumerate() {
            for (col_idx, value) in row.iter().copied().enumerate() {
                if !value.is_finite() {
                    return Err(ImuExtrinsicsError::NonFiniteRotation {
                        row: row_idx,
                        col: col_idx,
                        value,
                    });
                }
            }
        }
        for (axis, value) in t_cam_imu.translation().iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(ImuExtrinsicsError::NonFiniteTranslation { axis, value });
            }
        }
        Ok(Self {
            t_cam_imu,
            time_offset_ns,
        })
    }

    pub fn t_cam_imu(&self) -> Pose64 {
        self.t_cam_imu
    }

    pub fn time_offset_ns(&self) -> i64 {
        self.time_offset_ns
    }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImuAccumulatorError {
    NonIncreasingTimestamps {
        previous: Timestamp,
        current: Timestamp,
    },
}

impl std::fmt::Display for ImuAccumulatorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImuAccumulatorError::NonIncreasingTimestamps { previous, current } => write!(
                f,
                "imu accumulator timestamps must be strictly increasing: previous={} current={}",
                previous.as_nanos(),
                current.as_nanos()
            ),
        }
    }
}

impl std::error::Error for ImuAccumulatorError {}

fn validate_positive_finite(
    value: f64,
    non_finite: ImuNoiseModelError,
    non_positive: ImuNoiseModelError,
) -> Result<(), ImuNoiseModelError> {
    if !value.is_finite() {
        return Err(non_finite);
    }
    if value <= 0.0 {
        return Err(non_positive);
    }
    Ok(())
}

#[derive(Clone, Debug, Default)]
pub struct ImuAccumulator {
    samples: Vec<ImuSample>,
}

impl ImuAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn clear(&mut self) {
        self.samples.clear();
    }

    pub fn extend_batch(&mut self, batch: &ImuBatch) -> Result<(), ImuAccumulatorError> {
        for sample in batch.samples() {
            if let Some(previous) = self.samples.last().map(ImuSample::timestamp)
                && sample.timestamp().as_nanos() <= previous.as_nanos()
            {
                return Err(ImuAccumulatorError::NonIncreasingTimestamps {
                    previous,
                    current: sample.timestamp(),
                });
            }
            self.samples.push(sample.clone());
        }
        Ok(())
    }

    pub fn drain_batch(&mut self) -> Result<Option<ImuBatch>, ImuBatchError> {
        if self.samples.is_empty() {
            return Ok(None);
        }
        let samples = std::mem::take(&mut self.samples);
        ImuBatch::new(samples).map(Some)
    }

    pub fn batch(&self) -> Result<Option<ImuBatch>, ImuBatchError> {
        if self.samples.is_empty() {
            return Ok(None);
        }
        ImuBatch::new(self.samples.clone()).map(Some)
    }
}

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

    pub fn shifted_timestamp_ns(&self, delta_ns: i64) -> Result<Self, ImuTimestampShiftError> {
        if delta_ns == 0 {
            return Ok(self.clone());
        }
        let shifted = self
            .samples()
            .iter()
            .map(|sample| sample.shifted_timestamp_ns(delta_ns))
            .collect::<Result<Vec<_>, _>>()?;
        ImuBatch::new(shifted).map_err(|_| ImuTimestampShiftError::Overflow {
            timestamp: self.end_time(),
            delta_ns,
        })
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
    fn imu_batch_timestamp_shift_preserves_order_and_span() {
        let samples = vec![
            ImuSample::new(Timestamp::from_nanos(10), [0.0; 3], [0.0; 3]).expect("sample 0"),
            ImuSample::new(Timestamp::from_nanos(30), [1.0; 3], [2.0; 3]).expect("sample 1"),
            ImuSample::new(Timestamp::from_nanos(70), [3.0; 3], [4.0; 3]).expect("sample 2"),
        ];
        let batch = ImuBatch::new(samples).expect("batch");
        let shifted = batch.shifted_timestamp_ns(25).expect("shifted batch");
        assert_eq!(shifted.start_time(), Timestamp::from_nanos(35));
        assert_eq!(shifted.end_time(), Timestamp::from_nanos(95));
        assert!((shifted.dt_seconds() - batch.dt_seconds()).abs() < 1e-15);
    }

    #[test]
    fn imu_bias_default_is_zero() {
        let bias = ImuBias::default();
        assert_eq!(bias.accel, [0.0; 3]);
        assert_eq!(bias.gyro, [0.0; 3]);
    }

    #[test]
    fn imu_accumulator_preserves_strict_ordering() {
        let mut accumulator = ImuAccumulator::new();
        let first = ImuBatch::new(vec![
            ImuSample::new(Timestamp::from_nanos(10), [0.0; 3], [0.0; 3]).expect("sample 0"),
            ImuSample::new(Timestamp::from_nanos(20), [0.0; 3], [0.0; 3]).expect("sample 1"),
        ])
        .expect("batch");
        accumulator.extend_batch(&first).expect("extend 1");
        let second = ImuBatch::new(vec![
            ImuSample::new(Timestamp::from_nanos(30), [0.0; 3], [0.0; 3]).expect("sample 2"),
        ])
        .expect("batch");
        accumulator.extend_batch(&second).expect("extend 2");
        assert_eq!(accumulator.len(), 3);
        let drained = accumulator
            .drain_batch()
            .expect("drain")
            .expect("batch should exist");
        assert_eq!(drained.start_time(), Timestamp::from_nanos(10));
        assert_eq!(drained.end_time(), Timestamp::from_nanos(30));
        assert!(accumulator.is_empty());
    }

    #[test]
    fn imu_accumulator_rejects_out_of_order_extension() {
        let mut accumulator = ImuAccumulator::new();
        let batch = ImuBatch::new(vec![
            ImuSample::new(Timestamp::from_nanos(20), [0.0; 3], [0.0; 3]).expect("sample 0"),
        ])
        .expect("batch");
        accumulator.extend_batch(&batch).expect("first extend");
        let err = accumulator
            .extend_batch(
                &ImuBatch::new(vec![
                    ImuSample::new(Timestamp::from_nanos(10), [0.0; 3], [0.0; 3])
                        .expect("sample 1"),
                ])
                .expect("batch"),
            )
            .expect_err("out-of-order extension should fail");
        assert_eq!(
            err,
            ImuAccumulatorError::NonIncreasingTimestamps {
                previous: Timestamp::from_nanos(20),
                current: Timestamp::from_nanos(10),
            }
        );
    }
}
