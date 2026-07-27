//! Hardware-free semantic admission for a rendered Nano navigation graph.
//!
//! This boundary uses the same plant, navigation, actuation, and calibration
//! domain parsers as production startup. It proves only that retained bytes
//! form one internally consistent deployment graph; live startup must still
//! bind the calibration to observations from the exclusively opened OAK.

use std::fmt;

use sha2::{Digest, Sha256};

use super::mpc::{PlantModelJsonParseError, PlantModelV1};
use super::{
    ActuationConfigParseError, NanoCalibrationArtifactV1, NanoCalibrationBindingError,
    NavigationActuationConfigV1, ProductionNavigationControllerBindingError,
    ProductionNavigationControllerBindingV1, ProductionNavigationControllerContractV1,
    ShadowNavigationConfigParseError, ShadowNavigationConfigV1,
};
use crate::dense::occupancy::{DepthCameraModel, DepthToTrackingCamera};

/// Parsed shadow-only deployment graph used by wheels-off qualification.
pub struct OfflineShadowNavigationGraphV1 {
    navigation: ShadowNavigationConfigV1,
}

impl OfflineShadowNavigationGraphV1 {
    /// Parse and cross-bind exact retained plant and navigation bytes against
    /// the already parsed assembly calibration.
    pub fn parse(
        calibration: &NanoCalibrationArtifactV1,
        plant_bytes: &[u8],
        navigation_bytes: &[u8],
    ) -> Result<Self, OfflineNavigationGraphParseError> {
        let plant = PlantModelV1::parse_json(plant_bytes)
            .map_err(OfflineNavigationGraphParseError::Plant)?;
        let stereo = calibration.rectified_stereo();
        let depth_camera = DepthCameraModel::new(
            stereo.left(),
            stereo.dimensions(),
            DepthToTrackingCamera::identity(),
        );
        let navigation = ShadowNavigationConfigV1::parse_json_bound_to_plant_artifact(
            navigation_bytes,
            depth_camera,
            plant,
        )
        .map_err(OfflineNavigationGraphParseError::Navigation)?;
        calibration
            .require_navigation(&navigation)
            .map_err(OfflineNavigationGraphParseError::CalibrationNavigationBinding)?;
        Ok(Self { navigation })
    }

    pub const fn navigation(&self) -> &ShadowNavigationConfigV1 {
        &self.navigation
    }
}

/// Parsed production deployment graph. Construction requires one valid
/// shadow graph and an actuation document authorized against that exact graph.
pub struct OfflineProductionNavigationGraphV1 {
    shadow: OfflineShadowNavigationGraphV1,
    actuation: NavigationActuationConfigV1,
    controller_binding: ProductionNavigationControllerBindingV1,
}

impl OfflineProductionNavigationGraphV1 {
    /// Parse and cross-bind exact retained plant, navigation, and generated
    /// actuation bytes without opening hardware or granting motion authority.
    pub fn parse(
        calibration: &NanoCalibrationArtifactV1,
        plant_bytes: &[u8],
        navigation_bytes: &[u8],
        actuation_bytes: &[u8],
        robot_id: &str,
        controller: ProductionNavigationControllerContractV1<'_>,
    ) -> Result<Self, OfflineNavigationGraphParseError> {
        let shadow =
            OfflineShadowNavigationGraphV1::parse(calibration, plant_bytes, navigation_bytes)?;
        let navigation = shadow.navigation();
        let actuation = NavigationActuationConfigV1::parse_and_authorize(
            actuation_bytes,
            robot_id,
            navigation_bytes,
            navigation.mpc_solver().model(),
            navigation.solver_budget(),
            navigation.control_period(),
        )
        .map_err(OfflineNavigationGraphParseError::Actuation)?;
        let observed_plant_sha256: [u8; 32] = Sha256::digest(plant_bytes).into();
        let configured_plant_sha256 = *actuation.plant_artifact_content_sha256().as_bytes();
        if configured_plant_sha256 != observed_plant_sha256 {
            return Err(
                OfflineNavigationGraphParseError::PlantArtifactDigestMismatch {
                    configured: configured_plant_sha256,
                    observed: observed_plant_sha256,
                },
            );
        }
        calibration
            .require_actuation_approval(&actuation)
            .map_err(OfflineNavigationGraphParseError::CalibrationActuationBinding)?;
        let controller_binding = ProductionNavigationControllerBindingV1::bind(
            controller,
            &actuation,
            navigation.mpc_solver().config(),
            navigation.control_period(),
        )
        .map_err(OfflineNavigationGraphParseError::ControllerBinding)?;
        Ok(Self {
            shadow,
            actuation,
            controller_binding,
        })
    }

    pub const fn navigation(&self) -> &ShadowNavigationConfigV1 {
        self.shadow.navigation()
    }

    pub const fn actuation(&self) -> &NavigationActuationConfigV1 {
        &self.actuation
    }

    pub const fn controller_binding(&self) -> ProductionNavigationControllerBindingV1 {
        self.controller_binding
    }
}

#[derive(Debug)]
pub enum OfflineNavigationGraphParseError {
    Plant(PlantModelJsonParseError),
    Navigation(ShadowNavigationConfigParseError),
    CalibrationNavigationBinding(NanoCalibrationBindingError),
    Actuation(ActuationConfigParseError),
    PlantArtifactDigestMismatch {
        configured: [u8; 32],
        observed: [u8; 32],
    },
    CalibrationActuationBinding(NanoCalibrationBindingError),
    ControllerBinding(ProductionNavigationControllerBindingError),
}

impl fmt::Display for OfflineNavigationGraphParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Plant(source) => {
                write!(formatter, "retained plant artifact was rejected: {source}")
            }
            Self::Navigation(source) => {
                write!(
                    formatter,
                    "retained navigation config was rejected: {source}"
                )
            }
            Self::CalibrationNavigationBinding(source) => write!(
                formatter,
                "retained navigation config does not match assembly calibration: {source}"
            ),
            Self::Actuation(source) => {
                write!(
                    formatter,
                    "generated actuation config was rejected: {source}"
                )
            }
            Self::PlantArtifactDigestMismatch {
                configured,
                observed,
            } => write!(
                formatter,
                "generated actuation config plant digest {configured:02x?} does not match retained plant digest {observed:02x?}"
            ),
            Self::CalibrationActuationBinding(source) => write!(
                formatter,
                "generated actuation approval does not match assembly calibration: {source}"
            ),
            Self::ControllerBinding(source) => write!(
                formatter,
                "production navigation graph does not match its controller contract: {source}"
            ),
        }
    }
}

impl std::error::Error for OfflineNavigationGraphParseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Plant(source) => Some(source),
            Self::Navigation(source) => Some(source),
            Self::CalibrationNavigationBinding(source)
            | Self::CalibrationActuationBinding(source) => Some(source),
            Self::Actuation(source) => Some(source),
            Self::PlantArtifactDigestMismatch { .. } => None,
            Self::ControllerBinding(source) => Some(source),
        }
    }
}
