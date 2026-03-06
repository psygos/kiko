use crate::{ImuExtrinsics, ImuNoiseModel, PinholeIntrinsics, RectifiedStereo};

#[derive(Clone, Debug)]
pub struct CalibrationBundle {
    intrinsics: PinholeIntrinsics,
    stereo: RectifiedStereo,
    imu_noise: Option<ImuNoiseModel>,
    imu_extrinsics: Option<ImuExtrinsics>,
    gravity_magnitude_mps2: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CalibrationBundleError {
    MissingImuNoise,
    MissingImuExtrinsics,
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
            (Some(imu_noise), Some(imu_extrinsics)) => {
                Self::with_imu(intrinsics, stereo, imu_noise, imu_extrinsics, gravity_magnitude_mps2)
            }
            (None, Some(_)) => Err(CalibrationBundleError::MissingImuNoise),
            (Some(_), None) => Err(CalibrationBundleError::MissingImuExtrinsics),
        }
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
    use crate::dataset::CameraIntrinsics;
    use crate::Pose64;

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
            crate::imu::ImuNoiseModelError::NonPositiveGyroNoiseDensity { value: 0.0 }
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
