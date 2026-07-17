use crate::{
    GeometryError, ImuBias, ImuBiasError, ImuExtrinsics, ImuExtrinsicsError, ImuNoiseModel,
    ImuNoiseModelError, PinholeIntrinsics, Pose64, Pose64Error, PositiveF64, RectifiedStereo,
    RectifiedStereoCompatibilityError, StereoCalibration, StereoCalibrationError,
};

#[derive(Clone, Debug)]
pub struct CalibrationBundle {
    stereo: StereoCalibration,
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
    InvalidStereo { source: StereoCalibrationError },
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
                write!(f, "invalid structural stereo calibration: {source}")
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
            stereo: *stereo.stereo_calibration(),
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
            *stereo.stereo_calibration(),
            imu_noise,
            imu_extrinsics,
            gravity_magnitude_mps2,
            None,
        )
    }

    fn with_inertial_parts(
        stereo: StereoCalibration,
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
        // Parse the serialized stereo document exactly once into structural
        // types. Rectified-SLAM compatibility is an explicit runtime policy.
        let stereo_calibration = StereoCalibration::try_from(calibration)
            .map_err(|source| CalibrationBundleError::InvalidStereo { source })?;
        let Some(imu) = calibration.imu.as_ref() else {
            return Ok(Self {
                stereo: stereo_calibration,
                inertial: None,
            });
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
            stereo_calibration,
            noise,
            extrinsics,
            imu.gravity_magnitude_mps2,
            initial_bias,
        )
    }

    pub fn intrinsics(&self) -> PinholeIntrinsics {
        self.stereo.left()
    }

    pub fn stereo_calibration(&self) -> &StereoCalibration {
        &self.stereo
    }

    /// Apply the rectified-SLAM policy to the retained structural calibration.
    /// This does not parse serialized data again.
    pub fn rectified_stereo(&self) -> Result<RectifiedStereo, RectifiedStereoCompatibilityError> {
        RectifiedStereo::from_stereo_calibration(&self.stereo)
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
    fn structural_stereo_retains_rectification_and_asymmetric_camera_bits() {
        let mut dataset_calibration = dataset_calibration_with_imu();
        dataset_calibration.imu = None;
        dataset_calibration.left.fx = 400.0;
        dataset_calibration.left.fy = 401.0;
        dataset_calibration.left.cx = 320.0;
        dataset_calibration.left.cy = -0.0;
        dataset_calibration.right.fx = 420.0;
        dataset_calibration.right.fy = 410.0;
        dataset_calibration.right.cx = 316.0;
        dataset_calibration.right.cy = f32::from_bits(1);
        dataset_calibration.baseline_m = f32::from_bits(1);
        dataset_calibration.rectified = false;

        let structural = StereoCalibration::try_from(&dataset_calibration)
            .expect("structurally valid unrectified calibration");
        assert!(!structural.is_rectified());
        assert_eq!(structural.left().fx().to_bits(), 400.0_f32.to_bits());
        assert_eq!(structural.left().cy().to_bits(), (-0.0_f32).to_bits());
        assert_eq!(structural.right().fx().to_bits(), 420.0_f32.to_bits());
        assert_eq!(
            structural.right().cy().to_bits(),
            f32::from_bits(1).to_bits()
        );
        assert_eq!(
            structural.baseline_m().to_bits(),
            f32::from_bits(1).to_bits()
        );
        assert!(matches!(
            RectifiedStereo::from_stereo_calibration(&structural),
            Err(RectifiedStereoCompatibilityError::NotRectified)
        ));

        dataset_calibration.rectified = true;
        let structural = StereoCalibration::try_from(&dataset_calibration)
            .expect("structurally valid rectified calibration");
        let compatible = RectifiedStereo::from_stereo_calibration(&structural)
            .expect("asymmetric calibrated rig is supported");
        assert_eq!(compatible.left(), structural.left());
        assert_eq!(compatible.right(), structural.right());
        assert_eq!(
            compatible.stereo_calibration().baseline_m().to_bits(),
            structural.baseline_m().to_bits()
        );
    }

    #[test]
    fn serialized_camera_projection_and_image_shape_are_separate_parse_domains() {
        let serialized = CameraIntrinsics {
            fx: 400.0,
            fy: 401.0,
            cx: 320.0,
            cy: 240.0,
            width: 0,
            height: 0,
        };
        let intrinsics = PinholeIntrinsics::try_from(&serialized)
            .expect("projection coefficients do not own image dimensions");
        assert_eq!(intrinsics.fx().to_bits(), serialized.fx.to_bits());
        assert_eq!(intrinsics.fy().to_bits(), serialized.fy.to_bits());
        assert!(matches!(
            crate::FrameDimensions::try_new(serialized.width, serialized.height),
            Err(crate::FrameDimensionsError::Zero {
                width: 0,
                height: 0,
            })
        ));
    }

    #[test]
    fn stereo_structure_and_policy_have_stable_error_precedence_and_sources() {
        let mut calibration = dataset_calibration_with_imu();
        calibration.baseline_m = 0.0;
        calibration.left.width = 0;
        calibration.left.fx = 0.0;
        calibration.rectified = false;
        calibration
            .imu
            .as_mut()
            .expect("imu")
            .noise
            .gyro_noise_density = 0.0;

        let baseline_error = CalibrationBundle::from_dataset_calibration(&calibration)
            .expect_err("baseline is the first structural failure");
        assert!(matches!(
            &baseline_error,
            CalibrationBundleError::InvalidStereo {
                source: StereoCalibrationError::InvalidBaseline {
                    source: crate::StereoBaselineError::NonPositive { baseline_m: 0.0 },
                },
            }
        ));
        let stereo_source =
            std::error::Error::source(&baseline_error).expect("structural stereo source");
        assert!(
            stereo_source.source().is_some(),
            "baseline source is retained"
        );

        calibration.baseline_m = 0.1;
        let dimensions_error = CalibrationBundle::from_dataset_calibration(&calibration)
            .expect_err("dimensions precede intrinsics within a camera");
        assert!(matches!(
            dimensions_error,
            CalibrationBundleError::InvalidStereo {
                source: StereoCalibrationError::InvalidDimensions {
                    camera: crate::StereoCameraSide::Left,
                    source: crate::FrameDimensionsError::Zero {
                        width: 0,
                        height: 80,
                    },
                },
            }
        ));

        calibration.left.width = 100;
        let intrinsics_error = CalibrationBundle::from_dataset_calibration(&calibration)
            .expect_err("left intrinsics precede right camera and policy");
        assert!(matches!(
            intrinsics_error,
            CalibrationBundleError::InvalidStereo {
                source: StereoCalibrationError::InvalidIntrinsics {
                    camera: crate::StereoCameraSide::Left,
                    source: crate::IntrinsicsError::NonPositive {
                        field: "fx",
                        value: 0.0,
                    },
                },
            }
        ));

        calibration.left.fx = 100.0;
        let imu_error = CalibrationBundle::from_dataset_calibration(&calibration)
            .expect_err("bundle parses IMU without applying rectification policy");
        assert!(matches!(
            imu_error,
            CalibrationBundleError::InvalidImuNoise {
                source: ImuNoiseModelError::GyroNoiseDensityNonPositive { value: 0.0 },
            }
        ));

        calibration
            .imu
            .as_mut()
            .expect("imu")
            .noise
            .gyro_noise_density = 0.01;
        let bundle = CalibrationBundle::from_dataset_calibration(&calibration)
            .expect("structural bundle retains unrectified declaration");
        assert!(!bundle.stereo_calibration().is_rectified());
        assert!(matches!(
            bundle.rectified_stereo(),
            Err(RectifiedStereoCompatibilityError::NotRectified)
        ));
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
        assert_eq!(bias.accel_mps2(), [0.01, -0.02, 0.03]);
        assert_eq!(bias.gyro_radps(), [0.001, -0.002, 0.003]);
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
                source: StereoCalibrationError::InvalidBaseline {
                    source: crate::StereoBaselineError::NonPositive { baseline_m: 0.0 },
                },
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
                source: ImuBiasError::NonFiniteAccelMps2 { axis: 0, value },
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
