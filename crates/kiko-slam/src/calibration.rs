use crate::{ImuExtrinsics, ImuNoiseModel, PinholeIntrinsics, Pose64, RectifiedStereo};

#[derive(Clone, Debug)]
pub struct CalibrationBundle {
    intrinsics: PinholeIntrinsics,
    stereo: RectifiedStereo,
    imu_noise: Option<ImuNoiseModel>,
    imu_extrinsics: Option<ImuExtrinsics>,
    gravity_magnitude_mps2: f64,
    initial_bias: Option<crate::ImuBias>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CalibrationBundleError {
    MissingImuNoise,
    MissingImuExtrinsics,
    PartialInitialImuBias,
    NonPositiveGravity { value: f64 },
    NonFiniteGravity { value: f64 },
    NonPositiveAccelNoiseDensity { value: f64 },
    NonFiniteAccelNoiseDensity { value: f64 },
    NonPositiveGyroNoiseDensity { value: f64 },
    NonFiniteGyroNoiseDensity { value: f64 },
    NonPositiveAccelRandomWalk { value: f64 },
    NonFiniteAccelRandomWalk { value: f64 },
    NonPositiveGyroRandomWalk { value: f64 },
    NonFiniteGyroRandomWalk { value: f64 },
    InvalidImuExtrinsics { message: String },
}

impl std::fmt::Display for CalibrationBundleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CalibrationBundleError::MissingImuNoise => {
                write!(f, "imu calibration requires a noise model")
            }
            CalibrationBundleError::MissingImuExtrinsics => {
                write!(f, "imu calibration requires camera-imu extrinsics")
            }
            CalibrationBundleError::PartialInitialImuBias => {
                write!(
                    f,
                    "imu calibration initial bias must include both accel and gyro blocks"
                )
            }
            CalibrationBundleError::NonPositiveGravity { value } => {
                write!(f, "gravity magnitude must be > 0, got {value}")
            }
            CalibrationBundleError::NonFiniteGravity { value } => {
                write!(f, "gravity magnitude must be finite, got {value}")
            }
            CalibrationBundleError::NonPositiveAccelNoiseDensity { value } => {
                write!(f, "accelerometer noise density must be > 0, got {value}")
            }
            CalibrationBundleError::NonFiniteAccelNoiseDensity { value } => {
                write!(f, "accelerometer noise density must be finite, got {value}")
            }
            CalibrationBundleError::NonPositiveGyroNoiseDensity { value } => {
                write!(f, "gyroscope noise density must be > 0, got {value}")
            }
            CalibrationBundleError::NonFiniteGyroNoiseDensity { value } => {
                write!(f, "gyroscope noise density must be finite, got {value}")
            }
            CalibrationBundleError::NonPositiveAccelRandomWalk { value } => {
                write!(f, "accelerometer random walk must be > 0, got {value}")
            }
            CalibrationBundleError::NonFiniteAccelRandomWalk { value } => {
                write!(f, "accelerometer random walk must be finite, got {value}")
            }
            CalibrationBundleError::NonPositiveGyroRandomWalk { value } => {
                write!(f, "gyroscope random walk must be > 0, got {value}")
            }
            CalibrationBundleError::NonFiniteGyroRandomWalk { value } => {
                write!(f, "gyroscope random walk must be finite, got {value}")
            }
            CalibrationBundleError::InvalidImuExtrinsics { message } => {
                write!(f, "invalid imu extrinsics: {message}")
            }
        }
    }
}

impl std::error::Error for CalibrationBundleError {}

impl CalibrationBundle {
    pub fn visual_only(intrinsics: PinholeIntrinsics, stereo: RectifiedStereo) -> Self {
        Self {
            intrinsics,
            stereo,
            imu_noise: None,
            imu_extrinsics: None,
            gravity_magnitude_mps2: 9.81,
            initial_bias: None,
        }
    }

    pub fn with_imu(
        intrinsics: PinholeIntrinsics,
        stereo: RectifiedStereo,
        imu_noise: ImuNoiseModel,
        imu_extrinsics: ImuExtrinsics,
        gravity_magnitude_mps2: f64,
    ) -> Result<Self, CalibrationBundleError> {
        validate_noise(&imu_noise)?;
        if !gravity_magnitude_mps2.is_finite() {
            return Err(CalibrationBundleError::NonFiniteGravity {
                value: gravity_magnitude_mps2,
            });
        }
        if gravity_magnitude_mps2 <= 0.0 {
            return Err(CalibrationBundleError::NonPositiveGravity {
                value: gravity_magnitude_mps2,
            });
        }
        Ok(Self {
            intrinsics,
            stereo,
            imu_noise: Some(imu_noise),
            imu_extrinsics: Some(imu_extrinsics),
            gravity_magnitude_mps2,
            initial_bias: None,
        })
    }

    pub fn from_optional_imu(
        intrinsics: PinholeIntrinsics,
        stereo: RectifiedStereo,
        imu_noise: Option<ImuNoiseModel>,
        imu_extrinsics: Option<ImuExtrinsics>,
        gravity_magnitude_mps2: f64,
    ) -> Result<Self, CalibrationBundleError> {
        match (imu_noise, imu_extrinsics) {
            (None, None) => Ok(Self::visual_only(intrinsics, stereo)),
            (Some(imu_noise), Some(imu_extrinsics)) => Self::with_imu(
                intrinsics,
                stereo,
                imu_noise,
                imu_extrinsics,
                gravity_magnitude_mps2,
            ),
            (None, Some(_)) => Err(CalibrationBundleError::MissingImuNoise),
            (Some(_), None) => Err(CalibrationBundleError::MissingImuExtrinsics),
        }
    }

    pub fn from_dataset_calibration(
        intrinsics: PinholeIntrinsics,
        stereo: RectifiedStereo,
        calibration: &crate::dataset::Calibration,
    ) -> Result<Self, CalibrationBundleError> {
        let Some(imu) = calibration.imu.as_ref() else {
            return Ok(Self::visual_only(intrinsics, stereo));
        };
        let noise = ImuNoiseModel::new(
            imu.noise.accel_noise_density,
            imu.noise.gyro_noise_density,
            imu.noise.accel_random_walk,
            imu.noise.gyro_random_walk,
        )
        .map_err(map_noise_error)?;
        let t_cam_imu = Pose64::from_rt(imu.extrinsics.rotation, imu.extrinsics.translation);
        let extrinsics =
            ImuExtrinsics::new(t_cam_imu, imu.extrinsics.time_offset_ns).map_err(|err| {
                CalibrationBundleError::InvalidImuExtrinsics {
                    message: err.to_string(),
                }
            })?;
        let mut bundle = Self::with_imu(
            intrinsics,
            stereo,
            noise,
            extrinsics,
            imu.gravity_magnitude_mps2,
        )?;
        bundle.initial_bias = match (imu.initial_accel_bias, imu.initial_gyro_bias) {
            (None, None) => None,
            (Some(accel), Some(gyro)) => Some(crate::ImuBias { accel, gyro }),
            _ => return Err(CalibrationBundleError::PartialInitialImuBias),
        };
        Ok(bundle)
    }

    pub fn intrinsics(&self) -> PinholeIntrinsics {
        self.intrinsics
    }

    pub fn stereo(&self) -> &RectifiedStereo {
        &self.stereo
    }

    pub fn imu_noise(&self) -> Option<&ImuNoiseModel> {
        self.imu_noise.as_ref()
    }

    pub fn imu_extrinsics(&self) -> Option<&ImuExtrinsics> {
        self.imu_extrinsics.as_ref()
    }

    pub fn gravity_magnitude_mps2(&self) -> f64 {
        self.gravity_magnitude_mps2
    }

    pub fn has_imu(&self) -> bool {
        self.imu_noise.is_some() && self.imu_extrinsics.is_some()
    }

    pub fn initial_bias(&self) -> Option<&crate::ImuBias> {
        self.initial_bias.as_ref()
    }

    pub fn initial_accel_bias(&self) -> Option<[f64; 3]> {
        self.initial_bias.as_ref().map(|bias| bias.accel)
    }

    pub fn initial_gyro_bias(&self) -> Option<[f64; 3]> {
        self.initial_bias.as_ref().map(|bias| bias.gyro)
    }
}

fn validate_noise(noise: &ImuNoiseModel) -> Result<(), CalibrationBundleError> {
    validate_positive_finite(
        noise.accel_noise_density(),
        CalibrationBundleError::NonPositiveAccelNoiseDensity {
            value: noise.accel_noise_density(),
        },
        CalibrationBundleError::NonFiniteAccelNoiseDensity {
            value: noise.accel_noise_density(),
        },
    )?;
    validate_positive_finite(
        noise.gyro_noise_density(),
        CalibrationBundleError::NonPositiveGyroNoiseDensity {
            value: noise.gyro_noise_density(),
        },
        CalibrationBundleError::NonFiniteGyroNoiseDensity {
            value: noise.gyro_noise_density(),
        },
    )?;
    validate_positive_finite(
        noise.accel_random_walk(),
        CalibrationBundleError::NonPositiveAccelRandomWalk {
            value: noise.accel_random_walk(),
        },
        CalibrationBundleError::NonFiniteAccelRandomWalk {
            value: noise.accel_random_walk(),
        },
    )?;
    validate_positive_finite(
        noise.gyro_random_walk(),
        CalibrationBundleError::NonPositiveGyroRandomWalk {
            value: noise.gyro_random_walk(),
        },
        CalibrationBundleError::NonFiniteGyroRandomWalk {
            value: noise.gyro_random_walk(),
        },
    )
}

fn map_noise_error(err: crate::imu::ImuNoiseModelError) -> CalibrationBundleError {
    match err {
        crate::imu::ImuNoiseModelError::AccelNoiseDensityNonFinite { value } => {
            CalibrationBundleError::NonFiniteAccelNoiseDensity { value }
        }
        crate::imu::ImuNoiseModelError::AccelNoiseDensityNonPositive { value } => {
            CalibrationBundleError::NonPositiveAccelNoiseDensity { value }
        }
        crate::imu::ImuNoiseModelError::GyroNoiseDensityNonFinite { value } => {
            CalibrationBundleError::NonFiniteGyroNoiseDensity { value }
        }
        crate::imu::ImuNoiseModelError::GyroNoiseDensityNonPositive { value } => {
            CalibrationBundleError::NonPositiveGyroNoiseDensity { value }
        }
        crate::imu::ImuNoiseModelError::AccelRandomWalkNonFinite { value } => {
            CalibrationBundleError::NonFiniteAccelRandomWalk { value }
        }
        crate::imu::ImuNoiseModelError::AccelRandomWalkNonPositive { value } => {
            CalibrationBundleError::NonPositiveAccelRandomWalk { value }
        }
        crate::imu::ImuNoiseModelError::GyroRandomWalkNonFinite { value } => {
            CalibrationBundleError::NonFiniteGyroRandomWalk { value }
        }
        crate::imu::ImuNoiseModelError::GyroRandomWalkNonPositive { value } => {
            CalibrationBundleError::NonPositiveGyroRandomWalk { value }
        }
    }
}

fn validate_positive_finite(
    value: f64,
    non_positive: CalibrationBundleError,
    non_finite: CalibrationBundleError,
) -> Result<(), CalibrationBundleError> {
    if !value.is_finite() {
        return Err(non_finite);
    }
    if value <= 0.0 {
        return Err(non_positive);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Pose64;
    use crate::dataset::CameraIntrinsics;

    fn intrinsics() -> PinholeIntrinsics {
        PinholeIntrinsics::try_from(&CameraIntrinsics {
            fx: 100.0,
            fy: 100.0,
            cx: 50.0,
            cy: 40.0,
            width: 100,
            height: 80,
        })
        .expect("intrinsics")
    }

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

    #[test]
    fn visual_only_bundle_has_no_imu() {
        let bundle = CalibrationBundle::visual_only(intrinsics(), stereo());
        assert!(!bundle.has_imu());
        assert!(bundle.imu_noise().is_none());
        assert!(bundle.imu_extrinsics().is_none());
        assert!((bundle.gravity_magnitude_mps2() - 9.81).abs() < 1e-12);
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
        let bundle = CalibrationBundle::with_imu(
            intrinsics(),
            stereo(),
            imu_noise(),
            imu_extrinsics(),
            9.81,
        )
        .expect("full imu calibration");
        assert!(bundle.has_imu());
        assert!(bundle.imu_noise().is_some());
        assert!(bundle.imu_extrinsics().is_some());
    }

    #[test]
    fn from_dataset_calibration_preserves_imu_block() {
        let dataset_calibration = crate::dataset::Calibration {
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
                initial_accel_bias: None,
                initial_gyro_bias: None,
            }),
        };
        let bundle = CalibrationBundle::from_dataset_calibration(
            intrinsics(),
            stereo(),
            &dataset_calibration,
        )
        .expect("bundle");
        let imu_extrinsics = bundle.imu_extrinsics().expect("imu extrinsics");
        assert!(bundle.has_imu());
        assert_eq!(imu_extrinsics.time_offset_ns(), 42);
        assert_eq!(imu_extrinsics.t_cam_imu().translation(), [0.1, -0.2, 0.3]);
    }

    #[test]
    fn from_dataset_calibration_preserves_complete_initial_bias() {
        let dataset_calibration = crate::dataset::Calibration {
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
        };
        let bundle = CalibrationBundle::from_dataset_calibration(
            intrinsics(),
            stereo(),
            &dataset_calibration,
        )
        .expect("bundle");
        let bias = bundle.initial_bias().expect("initial bias");
        assert_eq!(bias.accel, [0.01, -0.02, 0.03]);
        assert_eq!(bias.gyro, [0.001, -0.002, 0.003]);
    }

    #[test]
    fn from_dataset_calibration_rejects_partial_initial_bias() {
        let dataset_calibration = crate::dataset::Calibration {
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
                initial_gyro_bias: None,
            }),
        };
        let err = CalibrationBundle::from_dataset_calibration(
            intrinsics(),
            stereo(),
            &dataset_calibration,
        )
        .expect_err("partial initial bias must fail");
        assert_eq!(err, CalibrationBundleError::PartialInitialImuBias);
    }

    #[test]
    fn from_optional_imu_rejects_partial_imu_configuration() {
        let err = CalibrationBundle::from_optional_imu(
            intrinsics(),
            stereo(),
            Some(imu_noise()),
            None,
            9.81,
        )
        .expect_err("partial imu calibration should fail");
        assert_eq!(err, CalibrationBundleError::MissingImuExtrinsics);
    }
}
