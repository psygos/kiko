use crate::{
    AccelSample, DepthImage, DepthImageError, DequeueSequence, DeviceSessionId, DeviceTimestamp,
    Frame, FrameError, FrameId, GyroSample, HostMonotonicTimestamp, ImuReport, InertialValueError,
    OakImuAcceleration, OakImuAngularVelocity, SensorAccuracy, SensorId, Timestamp,
};
use oak_sys::{DepthFrame, ImageFrame, ImuAccuracy, ImuSample};

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

/// Parse one OAK bridge sample into hardware-neutral SI domain types.
///
/// Luxonis reports acceleration in m/s^2 and angular velocity in rad/s. The
/// vectors remain explicitly typed in [`crate::OakImuFrame`]; this adapter does
/// not assert an IMU-to-camera or IMU-to-base transform.
pub fn oak_to_imu_report(
    sample: ImuSample,
    session_id: DeviceSessionId,
    host_arrival: HostMonotonicTimestamp,
) -> Result<ImuReport, InertialValueError> {
    let accel_timestamp = DeviceTimestamp::try_from_nanos(sample.accel_timestamp.as_nanos())?;
    let gyro_timestamp = DeviceTimestamp::try_from_nanos(sample.gyro_timestamp.as_nanos())?;
    let accel = sample.accel.as_array();
    let gyro = sample.gyro.as_array();
    let acceleration = OakImuAcceleration::try_new(
        f64::from(accel[0]),
        f64::from(accel[1]),
        f64::from(accel[2]),
    )?;
    let angular_velocity =
        OakImuAngularVelocity::try_new(f64::from(gyro[0]), f64::from(gyro[1]), f64::from(gyro[2]))?;

    Ok(ImuReport::new(
        session_id,
        DequeueSequence::new(sample.sequence),
        host_arrival,
        AccelSample::new(
            accel_timestamp,
            acceleration,
            map_oak_accuracy(sample.accel_accuracy)?,
        ),
        GyroSample::new(
            gyro_timestamp,
            angular_velocity,
            map_oak_accuracy(sample.gyro_accuracy)?,
        ),
    ))
}

fn map_oak_accuracy(accuracy: ImuAccuracy) -> Result<SensorAccuracy, InertialValueError> {
    match accuracy {
        ImuAccuracy::Unreliable => Ok(SensorAccuracy::Unreliable),
        ImuAccuracy::Low => Ok(SensorAccuracy::Low),
        ImuAccuracy::Medium => Ok(SensorAccuracy::Medium),
        ImuAccuracy::High => Ok(SensorAccuracy::High),
        _ => Err(InertialValueError::UnknownSensorAccuracy { raw: accuracy.repr }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oak_sys::{Timestamp as OakTimestamp, Vec3};

    fn session() -> DeviceSessionId {
        DeviceSessionId::try_new(7).expect("nonzero session")
    }

    fn raw_sample() -> ImuSample {
        ImuSample {
            accel_timestamp: OakTimestamp::from_nanos(10),
            gyro_timestamp: OakTimestamp::from_nanos(12),
            sequence: 4,
            accel: Vec3 {
                x: 1.0,
                y: 2.0,
                z: 3.0,
            },
            accel_accuracy: ImuAccuracy::Low,
            gyro: Vec3 {
                x: 4.0,
                y: 5.0,
                z: 6.0,
            },
            gyro_accuracy: ImuAccuracy::High,
        }
    }

    #[test]
    fn oak_adapter_preserves_units_metadata_and_independent_timestamps() {
        let report = oak_to_imu_report(
            raw_sample(),
            session(),
            HostMonotonicTimestamp::from_nanos(99),
        )
        .expect("valid OAK sample");

        assert_eq!(report.session_id(), session());
        assert_eq!(report.sequence(), DequeueSequence::new(4));
        assert_eq!(report.host_arrival().as_nanos(), 99);
        assert_eq!(report.accel().timestamp().as_nanos(), 10);
        assert_eq!(report.gyro().timestamp().as_nanos(), 12);
        assert_eq!(report.accel().acceleration().as_array(), [1.0, 2.0, 3.0]);
        assert_eq!(report.gyro().angular_velocity().as_array(), [4.0, 5.0, 6.0]);
        assert_eq!(report.accel().accuracy(), SensorAccuracy::Low);
        assert_eq!(report.gyro().accuracy(), SensorAccuracy::High);
    }

    #[test]
    fn oak_adapter_rejects_negative_timestamps_and_nonfinite_components() {
        let mut negative = raw_sample();
        negative.accel_timestamp = OakTimestamp::from_nanos(-1);
        assert_eq!(
            oak_to_imu_report(negative, session(), HostMonotonicTimestamp::from_nanos(1)),
            Err(InertialValueError::NegativeDeviceTimestamp { nanos: -1 })
        );

        let mut nonfinite = raw_sample();
        nonfinite.gyro.z = f32::NAN;
        assert!(matches!(
            oak_to_imu_report(nonfinite, session(), HostMonotonicTimestamp::from_nanos(1)),
            Err(InertialValueError::NonFiniteComponent {
                quantity: crate::InertialQuantity::AngularVelocityRadPerSec,
                axis: crate::InertialAxis::Z,
                ..
            })
        ));
    }

    #[test]
    fn oak_accuracy_mapping_is_explicit_and_total_for_supported_values() {
        for (raw, expected) in [
            (ImuAccuracy::Unreliable, SensorAccuracy::Unreliable),
            (ImuAccuracy::Low, SensorAccuracy::Low),
            (ImuAccuracy::Medium, SensorAccuracy::Medium),
            (ImuAccuracy::High, SensorAccuracy::High),
        ] {
            assert_eq!(map_oak_accuracy(raw), Ok(expected));
        }
        assert_eq!(
            map_oak_accuracy(ImuAccuracy { repr: u8::MAX }),
            Err(InertialValueError::UnknownSensorAccuracy { raw: u8::MAX })
        );
    }
}
