use crate::{
    GeometryError, ImuBias, ImuBiasError, ImuExtrinsics, ImuExtrinsicsError, ImuNoiseModel,
    ImuNoiseModelError, PinholeIntrinsics, Pose64, Pose64Error, PositiveF64, RectifiedStereo,
    RectifiedStereoError,
};

#[derive(Clone, Debug)]
pub struct CalibrationBundle {
    stereo: RectifiedStereo,
    inertial: Option<InertialCalibration>,
}

#[derive(Clone, Debug)]
pub struct InertialCalibration {
    noise: ImuNoiseModel,
    extrinsics: ImuExtrinsics,
    gravity_magnitude_mps2: PositiveF64,
    initial_bias: Option<ImuBias>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CalibrationBundleError {
    InvalidStereo { source: RectifiedStereoError },
    PartialInitialImuBias,
    InvalidImuNoise { source: ImuNoiseModelError },
    InvalidGravity { source: GeometryError },
    InvalidImuPose { source: Pose64Error },
    InvalidImuExtrinsics { source: ImuExtrinsicsError },
    InvalidInitialBias { source: ImuBiasError },
}

impl std::fmt::Display for CalibrationBundleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CalibrationBundleError::InvalidStereo { source } => {
                write!(f, "invalid rectified stereo calibration: {source}")
            }
            CalibrationBundleError::PartialInitialImuBias => {
                write!(
                    f,
                    "imu calibration initial bias must include both accel and gyro blocks"
                )
            }
            CalibrationBundleError::InvalidImuNoise { source } => {
                write!(f, "invalid imu noise model: {source}")
            }
            CalibrationBundleError::InvalidGravity { source } => {
                write!(f, "invalid imu gravity magnitude: {source}")
            }
            CalibrationBundleError::InvalidImuPose { source } => {
                write!(f, "invalid imu pose: {source}")
            }
            CalibrationBundleError::InvalidImuExtrinsics { source } => {
                write!(f, "invalid imu extrinsics: {source}")
            }
            CalibrationBundleError::InvalidInitialBias { source } => {
                write!(f, "invalid initial imu bias: {source}")
            }
        }
    }
}

impl std::error::Error for CalibrationBundleError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            CalibrationBundleError::InvalidStereo { source } => Some(source),
            CalibrationBundleError::InvalidImuNoise { source } => Some(source),
            CalibrationBundleError::InvalidGravity { source } => Some(source),
            CalibrationBundleError::InvalidImuPose { source } => Some(source),
            CalibrationBundleError::InvalidImuExtrinsics { source } => Some(source),
            CalibrationBundleError::InvalidInitialBias { source } => Some(source),
            CalibrationBundleError::PartialInitialImuBias => None,
        }
    }
}

impl CalibrationBundle {
    pub fn visual_only(stereo: RectifiedStereo) -> Self {
        Self {
            stereo,
            inertial: None,
        }
    }

    pub fn with_imu(
        stereo: RectifiedStereo,
        imu_noise: ImuNoiseModel,
        imu_extrinsics: ImuExtrinsics,
        gravity_magnitude_mps2: f64,
    ) -> Result<Self, CalibrationBundleError> {
        Self::with_inertial_parts(
            stereo,
            imu_noise,
            imu_extrinsics,
            gravity_magnitude_mps2,
            None,
        )
    }

    fn with_inertial_parts(
        stereo: RectifiedStereo,
        noise: ImuNoiseModel,
        extrinsics: ImuExtrinsics,
        gravity_magnitude_mps2: f64,
        initial_bias: Option<ImuBias>,
    ) -> Result<Self, CalibrationBundleError> {
        let gravity_magnitude_mps2 = PositiveF64::new(gravity_magnitude_mps2, "gravity magnitude")
            .map_err(|source| CalibrationBundleError::InvalidGravity { source })?;
        Ok(Self {
            stereo,
            inertial: Some(InertialCalibration {
                noise,
                extrinsics,
                gravity_magnitude_mps2,
                initial_bias,
            }),
        })
    }

    pub fn from_dataset_calibration(
        calibration: &crate::dataset::Calibration,
    ) -> Result<Self, CalibrationBundleError> {
        let stereo = RectifiedStereo::from_calibration(calibration)
            .map_err(|source| CalibrationBundleError::InvalidStereo { source })?;
        let Some(imu) = calibration.imu.as_ref() else {
            return Ok(Self::visual_only(stereo));
        };
        let noise = ImuNoiseModel::new(
            imu.noise.accel_noise_density,
            imu.noise.gyro_noise_density,
            imu.noise.accel_random_walk,
            imu.noise.gyro_random_walk,
        )
        .map_err(|source| CalibrationBundleError::InvalidImuNoise { source })?;
        let t_cam_imu = Pose64::try_from_rt(imu.extrinsics.rotation, imu.extrinsics.translation)
            .map_err(|source| CalibrationBundleError::InvalidImuPose { source })?;
        let extrinsics = ImuExtrinsics::new(t_cam_imu, imu.extrinsics.time_offset_ns)
            .map_err(|source| CalibrationBundleError::InvalidImuExtrinsics { source })?;
        let initial_bias = match (imu.initial_accel_bias, imu.initial_gyro_bias) {
            (None, None) => None,
            (Some(accel), Some(gyro)) => Some(
                ImuBias::try_new(accel, gyro)
                    .map_err(|source| CalibrationBundleError::InvalidInitialBias { source })?,
            ),
            _ => return Err(CalibrationBundleError::PartialInitialImuBias),
        };
        Self::with_inertial_parts(
            stereo,
            noise,
            extrinsics,
            imu.gravity_magnitude_mps2,
            initial_bias,
        )
    }

    pub fn intrinsics(&self) -> PinholeIntrinsics {
        PinholeIntrinsics::from_rectified_stereo(&self.stereo)
    }

    pub fn stereo(&self) -> &RectifiedStereo {
        &self.stereo
    }

    pub fn inertial(&self) -> Option<&InertialCalibration> {
        self.inertial.as_ref()
    }
}

impl InertialCalibration {
    pub fn noise(&self) -> &ImuNoiseModel {
        &self.noise
    }

    pub fn extrinsics(&self) -> &ImuExtrinsics {
        &self.extrinsics
    }

    pub fn gravity_magnitude_mps2(&self) -> f64 {
        self.gravity_magnitude_mps2.get()
    }

    pub fn initial_bias(&self) -> Option<&ImuBias> {
        self.initial_bias.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Pose64;
    use crate::dataset::CameraIntrinsics;

    fn stereo() -> RectifiedStereo {
        RectifiedStereo::from_calibration(&crate::dataset::Calibration {
            left: CameraIntrinsics {
                fx: 100.0,
                fy: 100.0,
                cx: 50.0,
                cy: 40.0,
                width: 100,
                height: 80,
            },
            right: CameraIntrinsics {
                fx: 100.0,
                fy: 100.0,
                cx: 50.0,
                cy: 40.0,
                width: 100,
                height: 80,
            },
            baseline_m: 0.1,
            rectified: true,
            imu: None,
        })
        .expect("stereo")
    }

    fn imu_noise() -> ImuNoiseModel {
        ImuNoiseModel::new(0.1, 0.01, 0.001, 0.0001).expect("imu noise")
    }

    fn imu_extrinsics() -> ImuExtrinsics {
        ImuExtrinsics::new(Pose64::identity(), 0).expect("imu extrinsics")
    }

    fn dataset_calibration_with_imu() -> crate::dataset::Calibration {
        crate::dataset::Calibration {
            left: CameraIntrinsics {
                fx: 100.0,
                fy: 100.0,
                cx: 50.0,
                cy: 40.0,
                width: 100,
                height: 80,
            },
            right: CameraIntrinsics {
                fx: 100.0,
                fy: 100.0,
                cx: 50.0,
                cy: 40.0,
                width: 100,
                height: 80,
            },
            baseline_m: 0.1,
            rectified: true,
            imu: Some(crate::dataset::ImuCalibration {
                noise: crate::dataset::ImuNoiseMeta {
                    accel_noise_density: 0.1,
                    gyro_noise_density: 0.01,
                    accel_random_walk: 0.001,
                    gyro_random_walk: 0.0001,
                },
                extrinsics: crate::dataset::ImuExtrinsicsMeta {
                    rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    translation: [0.1, -0.2, 0.3],
                    time_offset_ns: 42,
                },
                gravity_magnitude_mps2: 9.81,
                initial_accel_bias: Some([0.01, -0.02, 0.03]),
                initial_gyro_bias: Some([0.001, -0.002, 0.003]),
            }),
        }
    }

    #[test]
    fn visual_only_bundle_has_no_imu() {
        let bundle = CalibrationBundle::visual_only(stereo());
        assert!(bundle.inertial().is_none());
        assert_eq!(bundle.intrinsics().fx(), 100.0);
    }

    #[test]
    fn imu_noise_model_rejects_non_positive_noise_at_construction() {
        let noise = ImuNoiseModel::new(0.1, 0.0, 0.001, 0.0001).expect_err("zero gyro noise");
        assert_eq!(
            noise,
            crate::imu::ImuNoiseModelError::GyroNoiseDensityNonPositive { value: 0.0 }
        );
    }

    #[test]
    fn with_imu_accepts_complete_calibration() {
        let bundle = CalibrationBundle::with_imu(stereo(), imu_noise(), imu_extrinsics(), 9.81)
            .expect("full imu calibration");
        let inertial = bundle.inertial().expect("inertial calibration");
        assert_eq!(inertial.noise(), &imu_noise());
        assert_eq!(inertial.extrinsics().time_offset_ns(), 0);
        assert_eq!(inertial.extrinsics().t_cam_imu().translation(), [0.0; 3]);
        assert_eq!(inertial.gravity_magnitude_mps2(), 9.81);
    }

    #[test]
    fn from_dataset_calibration_preserves_imu_block() {
        let mut dataset_calibration = dataset_calibration_with_imu();
        let imu = dataset_calibration.imu.as_mut().expect("imu");
        imu.initial_accel_bias = None;
        imu.initial_gyro_bias = None;
        let bundle =
            CalibrationBundle::from_dataset_calibration(&dataset_calibration).expect("bundle");
        let imu_extrinsics = bundle
            .inertial()
            .expect("inertial calibration")
            .extrinsics();
        assert_eq!(imu_extrinsics.time_offset_ns(), 42);
        assert_eq!(imu_extrinsics.t_cam_imu().translation(), [0.1, -0.2, 0.3]);
    }

    #[test]
    fn from_dataset_calibration_preserves_complete_initial_bias() {
        let dataset_calibration = dataset_calibration_with_imu();
        let bundle =
            CalibrationBundle::from_dataset_calibration(&dataset_calibration).expect("bundle");
        let bias = bundle
            .inertial()
            .expect("inertial calibration")
            .initial_bias()
            .expect("initial bias");
        assert_eq!(bias.accel, [0.01, -0.02, 0.03]);
        assert_eq!(bias.gyro, [0.001, -0.002, 0.003]);
    }

    #[test]
    fn from_dataset_calibration_rejects_partial_initial_bias() {
        let mut dataset_calibration = dataset_calibration_with_imu();
        dataset_calibration
            .imu
            .as_mut()
            .expect("imu")
            .initial_gyro_bias = None;
        let err = CalibrationBundle::from_dataset_calibration(&dataset_calibration)
            .expect_err("partial initial bias must fail");
        assert_eq!(err, CalibrationBundleError::PartialInitialImuBias);
    }

    #[test]
    fn dataset_calibration_conversion_preserves_typed_source_chains() {
        let mut invalid_stereo = dataset_calibration_with_imu();
        invalid_stereo.baseline_m = 0.0;
        let error = CalibrationBundle::from_dataset_calibration(&invalid_stereo)
            .expect_err("zero baseline must fail");
        assert!(matches!(
            &error,
            CalibrationBundleError::InvalidStereo {
                source: RectifiedStereoError::InvalidBaseline { baseline_m: 0.0 },
            }
        ));
        assert!(std::error::Error::source(&error).is_some());

        let mut invalid_noise = dataset_calibration_with_imu();
        invalid_noise
            .imu
            .as_mut()
            .expect("imu")
            .noise
            .gyro_noise_density = 0.0;
        let error = CalibrationBundle::from_dataset_calibration(&invalid_noise)
            .expect_err("zero noise must fail");
        assert!(matches!(
            &error,
            CalibrationBundleError::InvalidImuNoise {
                source: ImuNoiseModelError::GyroNoiseDensityNonPositive { value: 0.0 },
            }
        ));
        assert!(std::error::Error::source(&error).is_some());

        let mut invalid_gravity = dataset_calibration_with_imu();
        invalid_gravity
            .imu
            .as_mut()
            .expect("imu")
            .gravity_magnitude_mps2 = f64::NAN;
        let error = CalibrationBundle::from_dataset_calibration(&invalid_gravity)
            .expect_err("nonfinite gravity must fail");
        assert!(matches!(
            &error,
            CalibrationBundleError::InvalidGravity {
                source: GeometryError::NonFiniteScalar { value, .. },
            } if value.is_nan()
        ));
        assert!(std::error::Error::source(&error).is_some());

        let mut invalid_bias = dataset_calibration_with_imu();
        invalid_bias.imu.as_mut().expect("imu").initial_accel_bias = Some([f64::NAN, 0.0, 0.0]);
        let error = CalibrationBundle::from_dataset_calibration(&invalid_bias)
            .expect_err("nonfinite initial bias must fail");
        assert!(matches!(
            &error,
            CalibrationBundleError::InvalidInitialBias {
                source: ImuBiasError::NonFiniteAccel { axis: 0, value },
            } if value.is_nan()
        ));
        assert!(std::error::Error::source(&error).is_some());
    }

    #[test]
    fn calibration_error_preserves_pose_validation_source() {
        let err = CalibrationBundleError::InvalidImuPose {
            source: Pose64Error::ImproperRotation { determinant: -1.0 },
        };
        let source = std::error::Error::source(&err).expect("pose validation source");
        assert!(source.to_string().contains("determinant"));
    }
}
