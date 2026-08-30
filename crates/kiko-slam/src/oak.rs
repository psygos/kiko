use crate::dataset::{Calibration, CameraIntrinsics, OakEepromCalibrationEvidence};
use crate::{
    AccelSample, DepthImage, DepthImageError, DequeueSequence, DeviceSessionId, DeviceTimestamp,
    Frame, FrameError, FrameId, GyroSample, HostMonotonicTimestamp, ImuReport, InertialValueError,
    OakImuAcceleration, OakImuAngularVelocity, SensorAccuracy, SensorId, Timestamp,
};
use oak_sys::{DepthFrame, ImageFrame, ImuAccuracy, ImuSample, Intrinsics};

/// Build the projection contract for one admitted OAK stereo graph.
///
/// DepthAI RVC2 `StereoDepth` rectified-right frames can retain CAM_C's source
/// intrinsic metadata even though the delivered pixels have been remapped to
/// the common rectified-left projection. A rectified graph therefore binds
/// both delivered images to `left`; a direct, unrectified graph retains each
/// camera's independent projection. Callers must first verify that both
/// frames came from the configured stereo streams and dimensions.
pub fn oak_stereo_calibration_from_frame_metadata(
    left: Intrinsics,
    right: Intrinsics,
    baseline_m: f32,
    oak_eeprom: Option<OakEepromCalibrationEvidence>,
    rectified: bool,
) -> Calibration {
    let right_projection = if rectified { left } else { right };
    Calibration {
        left: camera_intrinsics(left),
        right: camera_intrinsics(right_projection),
        baseline_m,
        rectified,
        oak_eeprom,
    }
}

fn camera_intrinsics(intrinsics: Intrinsics) -> CameraIntrinsics {
    CameraIntrinsics {
        fx: intrinsics.fx(),
        fy: intrinsics.fy(),
        cx: intrinsics.cx(),
        cy: intrinsics.cy(),
        width: intrinsics.width(),
        height: intrinsics.height(),
    }
}

pub fn oak_to_frame(oak_frame: ImageFrame, sensor: SensorId) -> Result<Frame, FrameError> {
    let frame_id = FrameId::new(oak_frame.device_capture_sequence.as_u64());
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
    let frame_id = FrameId::new(oak_frame.device_capture_sequence.as_u64());
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
    use crate::RectifiedStereo;
    use oak_sys::{Timestamp as OakTimestamp, Vec3};

    fn session() -> DeviceSessionId {
        DeviceSessionId::try_new(7).expect("nonzero session")
    }

    fn raw_sample() -> ImuSample {
        ImuSample {
            accel_timestamp: OakTimestamp::try_from_nanos(10).expect("valid timestamp"),
            gyro_timestamp: OakTimestamp::try_from_nanos(12).expect("valid timestamp"),
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

    fn intrinsics(fx: f32, fy: f32, cx: f32, cy: f32) -> Intrinsics {
        Intrinsics::try_from_projection_matrix(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            640,
            400,
        )
        .expect("valid test intrinsics")
    }

    #[test]
    fn rectified_oak_metadata_uses_one_delivered_left_projection() {
        // Exact matrices observed from the qualification OAK-D S2. The
        // right values describe CAM_C's source metadata, not a second
        // projection for the already-remapped rectified-right pixels.
        let left = intrinsics(398.1716, 398.1898, 308.64267, 199.88481);
        let right_source = intrinsics(396.992, 397.00247, 326.84726, 194.88861);

        let calibration =
            oak_stereo_calibration_from_frame_metadata(left, right_source, 0.07503394, None, true);

        assert_eq!(calibration.left.fx.to_bits(), left.fx().to_bits());
        assert_eq!(calibration.left.fy.to_bits(), left.fy().to_bits());
        assert_eq!(calibration.left.cx.to_bits(), left.cx().to_bits());
        assert_eq!(calibration.left.cy.to_bits(), left.cy().to_bits());
        assert_eq!(calibration.right.fx.to_bits(), left.fx().to_bits());
        assert_eq!(calibration.right.fy.to_bits(), left.fy().to_bits());
        assert_eq!(calibration.right.cx.to_bits(), left.cx().to_bits());
        assert_eq!(calibration.right.cy.to_bits(), left.cy().to_bits());
        RectifiedStereo::from_calibration(&calibration)
            .expect("common rectified projection must parse without tolerance inflation");
    }

    #[test]
    fn unrectified_oak_metadata_retains_both_source_projections() {
        let left = intrinsics(398.0, 398.0, 309.0, 200.0);
        let right = intrinsics(397.0, 397.0, 327.0, 195.0);

        let calibration =
            oak_stereo_calibration_from_frame_metadata(left, right, 0.075, None, false);

        assert_eq!(calibration.left.fx.to_bits(), left.fx().to_bits());
        assert_eq!(calibration.right.fx.to_bits(), right.fx().to_bits());
        assert_eq!(calibration.right.cx.to_bits(), right.cx().to_bits());
        assert!(!calibration.rectified);
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
    fn oak_adapter_rejects_nonfinite_components() {
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
