use crate::{
    DepthImage, DepthImageError, Frame, FrameError, FrameId, ImuBatch, ImuBatchError, ImuSample,
    ImuSampleError, SensorId, Timestamp,
};
use oak_sys::{DepthFrame, ImageFrame};

pub fn oak_to_frame(
    oak_frame: ImageFrame,
    sensor: SensorId,
    frame_id: FrameId,
) -> Result<Frame, FrameError> {
    Frame::new(
        sensor,
        frame_id,
        Timestamp::from_nanos(oak_frame.timestamp.as_nanos()),
        oak_frame.width,
        oak_frame.height,
        oak_frame.into_pixels(),
    )
}

pub fn oak_to_depth_image(oak_frame: DepthFrame) -> Result<DepthImage, DepthImageError> {
    let frame_id = FrameId::new(oak_frame.sequence);
    let timestamp = Timestamp::from_nanos(oak_frame.timestamp.as_nanos());
    let width = oak_frame.width;
    let height = oak_frame.height;
    let depth_mm = oak_frame.into_depth_mm();
    DepthImage::from_depth_mm(frame_id, timestamp, width, height, depth_mm)
}

#[derive(Debug)]
pub enum OakImuError {
    Sample {
        index: usize,
        source: ImuSampleError,
    },
    Batch(ImuBatchError),
}

impl std::fmt::Display for OakImuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OakImuError::Sample { index, source } => {
                write!(f, "invalid oak imu sample at index {index}: {source}")
            }
            OakImuError::Batch(source) => write!(f, "invalid oak imu batch: {source}"),
        }
    }
}

impl std::error::Error for OakImuError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            OakImuError::Sample { source, .. } => Some(source),
            OakImuError::Batch(source) => Some(source),
        }
    }
}

pub fn oak_to_imu_batch(samples: Vec<oak_sys::ImuSample>) -> Result<ImuBatch, OakImuError> {
    let mut converted = Vec::with_capacity(samples.len());
    for (index, sample) in samples.into_iter().enumerate() {
        let converted_sample = ImuSample::new(
            Timestamp::from_nanos(sample.timestamp.as_nanos()),
            [
                f64::from(sample.accel.x),
                f64::from(sample.accel.y),
                f64::from(sample.accel.z),
            ],
            [
                f64::from(sample.gyro.x),
                f64::from(sample.gyro.y),
                f64::from(sample.gyro.z),
            ],
        )
        .map_err(|source| OakImuError::Sample { index, source })?;
        converted.push(converted_sample);
    }
    ImuBatch::new(converted).map_err(OakImuError::Batch)
}

#[cfg(test)]
mod tests {
    use super::OakImuError;
    use crate::ImuBatchError;
    use std::error::Error as _;

    #[test]
    fn oak_imu_error_exposes_batch_source() {
        let error = OakImuError::Batch(ImuBatchError::Empty);
        assert_eq!(
            error.source().expect("batch source").to_string(),
            "imu batch must contain at least one sample"
        );
    }
}
