//! Qualification-only Nano bootstrap up to one exactly inventoried, stopped
//! candidate controller.
//!
//! This is deliberately a library boundary rather than a second live-loop
//! binary. It loads every launch-bound file once, binds the schema-V2
//! candidate contracts, probes the connected robot, acquires only an
//! acknowledged zero, performs exact inner-V1 inventory comparison, and
//! returns a linear stopped controller token for a later qualification owner.

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use kiko_device_inventory::{
    ArtifactHashError, ArtifactId, ArtifactKind, ArtifactRelativePath, ArtifactRelativePathError,
    ExactInventoryAdmission, InventoryMismatchReport, LoadedDeploymentAsset,
    LoadedExpectedManifestV2, ManifestArtifactHashes, ManifestLoadError, admit_exact_inventory,
    hash_manifest_artifacts_reusing_loaded_asset, load_expected_manifest_v2_from_slice,
};
use kiko_eye_runtime::{IdentityProbeError, probe_serial_eye_identity};
use kiko_head_runtime::{SerialHeadProbeError, probe_serial_head};
use oak_sys::{ConnectedDeviceIdentity, DepthAiBuildMetadata, Device};
use robot_command_client::{AppliedCommandReceipt, DisarmReceipt};
use robot_server::config::{ControllerServerConfigV2, ServerConfigError};
use robot_server::{
    V2ControllerOwner, V2ControllerOwnerStartError, V2ControllerOwnerTerminationError,
};

use super::actuation::LiveActuationError;
use super::nano_bootstrap::{
    NanoExactDevicePresenceTargets, bootstrap_stereo, derive_required_probe_configs,
    require_running, wait_for_exact_device_presence,
};
use super::{
    AdmittedOakSuperSpeedEvidence, CandidateActuationSessionStartError, CandidateMpcBindingError,
    CandidateRuntimeServiceIntervalError, LoadedNanoWheelsOffQualificationLaunchV1,
    ManifestBoundNanoAgentPolicyConfigV3, NanoAccessoryManifestBindingError,
    NanoAgentPolicyConfigParseError, NanoAgentPolicyConfigV3, NanoBootstrapAccessoryEvidence,
    NanoBootstrapOakCloseDisposition, NanoBootstrapPrimaryError, NanoBootstrapRootError,
    NanoBootstrapRoots, NanoBootstrapStereoEvidence, NanoCalibrationArtifactParseError,
    NanoCalibrationArtifactV1, NanoCalibrationBindingError, NanoLaunchBoundAssetLoadError,
    NanoObservedInventoryBuildError, NanoObservedInventoryBuilder,
    NanoObservedInventoryEvidenceError, NanoWheelsOffQualificationAssetRole,
    NanoWheelsOffQualificationLaunchLoadError, NanoWheelsOffQualificationLaunchV1,
    ParsedNanoLiveConfiguration, ShadowNavigationConfigParseError, ShadowNavigationConfigV1,
    StoppedWheelsOffCandidateController, WheelsOffCandidateActuationSession,
    WheelsOffCandidateControllerBinding, WheelsOffCandidateControllerBindingError,
    WheelsOffCandidateLimits, WheelsOffCandidatePolicyError,
    WheelsOffCandidateRuntimeServiceInterval, load_nano_wheels_off_qualification_launch_v1,
};
use crate::live_runtime::LiveOccupancyHostPolicy;

/// One process-lifetime request for the qualification-only static/hardware
/// boundary.
pub struct QualificationBootstrapRequest<'running> {
    roots: NanoBootstrapRoots,
    launch_relative_path: ArtifactRelativePath,
    controller_clock_origin: Instant,
    running: &'running AtomicBool,
}

impl<'running> QualificationBootstrapRequest<'running> {
    pub fn try_new(
        deployment_root: PathBuf,
        state_root: PathBuf,
        launch_relative_path: String,
        controller_clock_origin: Instant,
        running: &'running AtomicBool,
    ) -> Result<Self, QualificationBootstrapRequestError> {
        Ok(Self {
            roots: NanoBootstrapRoots::try_new(deployment_root, state_root)
                .map_err(QualificationBootstrapRequestError::Roots)?,
            launch_relative_path: ArtifactRelativePath::parse(launch_relative_path)
                .map_err(QualificationBootstrapRequestError::LaunchRelativePath)?,
            controller_clock_origin,
            running,
        })
    }

    pub const fn roots(&self) -> &NanoBootstrapRoots {
        &self.roots
    }

    pub const fn launch_relative_path(&self) -> &ArtifactRelativePath {
        &self.launch_relative_path
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum QualificationBootstrapRequestError {
    Roots(NanoBootstrapRootError),
    LaunchRelativePath(ArtifactRelativePathError),
}

impl fmt::Display for QualificationBootstrapRequestError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid Nano wheels-off qualification bootstrap request: {self:?}"
        )
    }
}

impl std::error::Error for QualificationBootstrapRequestError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Roots(source) => Some(source),
            Self::LaunchRelativePath(source) => Some(source),
        }
    }
}

/// Exact retained bytes for all qualification launch roles.
#[derive(Debug)]
pub struct LoadedNanoWheelsOffQualificationAssets {
    pub agent_policy: LoadedDeploymentAsset,
    pub navigation_shadow_config: LoadedDeploymentAsset,
    pub candidate_inventory_manifest: LoadedDeploymentAsset,
    pub candidate_controller_policy: LoadedDeploymentAsset,
    pub controller_server_contract: LoadedDeploymentAsset,
    pub calibration_artifact: LoadedDeploymentAsset,
    pub plant_artifact: LoadedDeploymentAsset,
    pub onnx_runtime_library: LoadedDeploymentAsset,
    pub superpoint_model: LoadedDeploymentAsset,
    pub lightglue_model: LoadedDeploymentAsset,
}

impl LoadedNanoWheelsOffQualificationAssets {
    fn load(
        deployment_root: &Path,
        launch: &NanoWheelsOffQualificationLaunchV1,
    ) -> Result<Self, QualificationBootstrapPrimaryError> {
        let load = |role| {
            launch
                .asset(role)
                .load_exact(deployment_root)
                .map_err(
                    |source| QualificationBootstrapPrimaryError::BoundAssetLoad { role, source },
                )
        };
        Ok(Self {
            agent_policy: load(NanoWheelsOffQualificationAssetRole::AgentPolicy)?,
            navigation_shadow_config: load(
                NanoWheelsOffQualificationAssetRole::NavigationShadowConfig,
            )?,
            candidate_inventory_manifest: load(
                NanoWheelsOffQualificationAssetRole::CandidateInventoryManifest,
            )?,
            candidate_controller_policy: load(
                NanoWheelsOffQualificationAssetRole::CandidateControllerPolicy,
            )?,
            controller_server_contract: load(
                NanoWheelsOffQualificationAssetRole::ControllerServerContract,
            )?,
            calibration_artifact: load(NanoWheelsOffQualificationAssetRole::CalibrationArtifact)?,
            plant_artifact: load(NanoWheelsOffQualificationAssetRole::PlantArtifact)?,
            onnx_runtime_library: load(NanoWheelsOffQualificationAssetRole::OnnxRuntimeLibrary)?,
            superpoint_model: load(NanoWheelsOffQualificationAssetRole::SuperpointModel)?,
            lightglue_model: load(NanoWheelsOffQualificationAssetRole::LightglueModel)?,
        })
    }
}

/// Exact plant selection jointly proven by launch, policy, manifest, and the
/// retained launch bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QualificationPlantEvidence {
    artifact_id: ArtifactId,
    artifact_relative_path: ArtifactRelativePath,
    deployment_relative_path: ArtifactRelativePath,
    content_sha256: [u8; 32],
}

impl QualificationPlantEvidence {
    pub const fn artifact_id(&self) -> &ArtifactId {
        &self.artifact_id
    }

    pub const fn artifact_relative_path(&self) -> &ArtifactRelativePath {
        &self.artifact_relative_path
    }

    pub const fn deployment_relative_path(&self) -> &ArtifactRelativePath {
        &self.deployment_relative_path
    }

    pub const fn content_sha256(&self) -> &[u8; 32] {
        &self.content_sha256
    }
}

/// Successful qualification handoff. The controller is exactly stopped; no
/// method here can emit motor output.
#[must_use = "the OAK and stopped candidate-controller token need explicit lifecycle ownership"]
pub struct PreparedNanoWheelsOffQualificationBootstrap {
    pub roots: NanoBootstrapRoots,
    pub launch: LoadedNanoWheelsOffQualificationLaunchV1,
    pub assets: LoadedNanoWheelsOffQualificationAssets,
    pub manifest: LoadedExpectedManifestV2,
    pub policy: ManifestBoundNanoAgentPolicyConfigV3,
    pub calibration: NanoCalibrationArtifactV1,
    pub plant: QualificationPlantEvidence,
    pub artifact_hashes: ManifestArtifactHashes,
    pub accessory_evidence: NanoBootstrapAccessoryEvidence,
    pub oak_connected_identity: ConnectedDeviceIdentity,
    pub oak_usb_transport: AdmittedOakSuperSpeedEvidence,
    pub depthai_build_metadata: DepthAiBuildMetadata,
    pub stereo: NanoBootstrapStereoEvidence,
    pub live: ParsedNanoLiveConfiguration,
    pub exact_inventory_admission: ExactInventoryAdmission,
    pub candidate_limits: WheelsOffCandidateLimits,
    pub candidate_runtime_service_interval: WheelsOffCandidateRuntimeServiceInterval,
    pub initial_zero: AppliedCommandReceipt,
    pub initial_stop: DisarmReceipt,
    pub stopped_controller: StoppedWheelsOffCandidateController,
    pub oak: Device,
}

/// Prepared qualification bootstrap plus the sole in-process STM32/UDP owner.
#[must_use = "split the prepared bootstrap and supervise the controller owner"]
pub struct PreparedNanoWheelsOffQualificationOwnedBootstrap {
    bootstrap: PreparedNanoWheelsOffQualificationBootstrap,
    controller_owner: V2ControllerOwner,
    controller_owner_shutdown_timeout: Duration,
}

impl PreparedNanoWheelsOffQualificationOwnedBootstrap {
    pub fn into_parts(
        self,
    ) -> (
        PreparedNanoWheelsOffQualificationBootstrap,
        V2ControllerOwner,
        Duration,
    ) {
        (
            self.bootstrap,
            self.controller_owner,
            self.controller_owner_shutdown_timeout,
        )
    }
}

/// Load, cross-bind, probe, acquire zero, exactly inventory, and return a
/// stopped candidate controller.
pub async fn bootstrap_nano_wheels_off_qualification(
    request: QualificationBootstrapRequest<'_>,
) -> Result<PreparedNanoWheelsOffQualificationOwnedBootstrap, QualificationBootstrapError> {
    let QualificationBootstrapRequest {
        roots,
        launch_relative_path,
        controller_clock_origin,
        running,
    } = request;

    require_running(running)
        .map_err(QualificationBootstrapPrimaryError::common)
        .map_err(QualificationBootstrapError::before_hardware)?;
    let launch =
        load_nano_wheels_off_qualification_launch_v1(roots.deployment_root(), launch_relative_path)
            .map_err(QualificationBootstrapPrimaryError::LaunchLoad)
            .map_err(QualificationBootstrapError::before_hardware)?;
    let assets =
        LoadedNanoWheelsOffQualificationAssets::load(roots.deployment_root(), launch.launch())
            .map_err(QualificationBootstrapError::before_hardware)?;

    let parsed_policy = NanoAgentPolicyConfigV3::parse_json(assets.agent_policy.bytes())
        .map_err(QualificationBootstrapPrimaryError::AgentPolicy)
        .map_err(QualificationBootstrapError::before_hardware)?;
    bind_policy_paths(
        &roots,
        launch.launch(),
        parsed_policy.inventory().manifest_path().as_path(),
        parsed_policy.inventory().artifact_root_path().as_path(),
    )
    .map_err(QualificationBootstrapError::before_hardware)?;
    let manifest =
        load_expected_manifest_v2_from_slice(assets.candidate_inventory_manifest.bytes())
            .map_err(QualificationBootstrapPrimaryError::Manifest)
            .map_err(QualificationBootstrapError::before_hardware)?;
    if launch.launch().robot_id().as_str() != manifest.manifest().as_inventory().robot_id().as_str()
    {
        return Err(QualificationBootstrapError::before_hardware(
            QualificationBootstrapPrimaryError::RobotIdMismatch {
                launch: launch.launch().robot_id().as_str().to_owned(),
                manifest: manifest
                    .manifest()
                    .as_inventory()
                    .robot_id()
                    .as_str()
                    .to_owned(),
            },
        ));
    }
    let policy = parsed_policy
        .bind_accessories_to_manifest(manifest.manifest().as_inventory())
        .map_err(QualificationBootstrapPrimaryError::AccessoryManifestBinding)
        .map_err(QualificationBootstrapError::before_hardware)?;
    let (head_probe_config, eye_probe_config) = derive_required_probe_configs(&policy)
        .map_err(QualificationBootstrapPrimaryError::common)
        .map_err(QualificationBootstrapError::before_hardware)?;

    let artifact_hashes = hash_manifest_artifacts_reusing_loaded_asset(
        manifest.manifest().as_inventory(),
        roots.deployment_root(),
        policy.inventory().artifact_root_path().as_path(),
        policy.inventory().artifact_bindings().clone(),
        &assets.plant_artifact,
    )
    .map_err(QualificationBootstrapPrimaryError::ArtifactHash)
    .map_err(QualificationBootstrapError::before_hardware)?;
    bind_calibration(
        &roots,
        launch.launch(),
        &assets.calibration_artifact,
        &policy,
        manifest.manifest().as_inventory(),
        &artifact_hashes,
    )
    .map_err(QualificationBootstrapError::before_hardware)?;
    let calibration = NanoCalibrationArtifactV1::parse_json(assets.calibration_artifact.bytes())
        .map_err(QualificationBootstrapPrimaryError::CalibrationArtifact)
        .map_err(QualificationBootstrapError::before_hardware)?;
    calibration
        .require_manifest_oak_mxid(manifest.manifest().as_inventory().oak().mxid().as_str())
        .map_err(QualificationBootstrapPrimaryError::CalibrationBinding)
        .map_err(QualificationBootstrapError::before_hardware)?;
    let plant = select_plant(
        &roots,
        launch.launch(),
        &assets.plant_artifact,
        &policy,
        manifest.manifest().as_inventory(),
        &artifact_hashes,
    )
    .map_err(QualificationBootstrapError::before_hardware)?;
    let plant_artifact_model = super::mpc::PlantModelV1::parse_json(assets.plant_artifact.bytes())
        .map_err(NanoBootstrapPrimaryError::PlantArtifactModel)
        .map_err(QualificationBootstrapPrimaryError::common)
        .map_err(QualificationBootstrapError::before_hardware)?;

    let controller_server =
        ControllerServerConfigV2::parse_json(assets.controller_server_contract.bytes())
            .map_err(QualificationBootstrapPrimaryError::ControllerServer)
            .map_err(QualificationBootstrapError::before_hardware)?;
    let candidate_policy =
        WheelsOffCandidateControllerBinding::parse_json(assets.candidate_controller_policy.bytes())
            .map_err(QualificationBootstrapPrimaryError::CandidatePolicy)
            .map_err(QualificationBootstrapError::before_hardware)?;
    let candidate_admission = candidate_policy
        .admit(
            &manifest,
            &controller_server,
            launch.launch().controller_server().command_udp_endpoint(),
        )
        .map_err(QualificationBootstrapPrimaryError::CandidateBinding)
        .map_err(QualificationBootstrapError::before_hardware)?;
    let candidate_limits = candidate_admission.limits();

    // Match production cold-boot behavior exactly: retry only read-only
    // enumeration of the already parsed identities. The head/eye probes, OAK
    // connect, and controller exclusive open below each remain one attempt.
    wait_for_exact_device_presence(
        NanoExactDevicePresenceTargets::new(
            &head_probe_config,
            &eye_probe_config,
            manifest.manifest().as_inventory().stm32(),
            manifest.manifest().as_inventory().oak(),
        ),
        running,
    )
    .map_err(NanoBootstrapPrimaryError::DevicePresenceWait)
    .map_err(QualificationBootstrapPrimaryError::common)
    .map_err(QualificationBootstrapError::before_hardware)?;

    require_running(running)
        .map_err(QualificationBootstrapPrimaryError::common)
        .map_err(QualificationBootstrapError::before_hardware)?;
    let head = probe_serial_head(&head_probe_config)
        .await
        .map_err(|source| QualificationBootstrapPrimaryError::HeadProbe(Box::new(source)))
        .map_err(QualificationBootstrapError::before_hardware)?;
    require_running(running)
        .map_err(QualificationBootstrapPrimaryError::common)
        .map_err(QualificationBootstrapError::before_hardware)?;
    let (eye_serial, eye_identity) = probe_serial_eye_identity(&eye_probe_config)
        .await
        .map_err(QualificationBootstrapPrimaryError::EyeProbe)
        .map_err(QualificationBootstrapError::before_hardware)?;
    require_running(running)
        .map_err(QualificationBootstrapPrimaryError::common)
        .map_err(QualificationBootstrapError::before_hardware)?;

    let mut oak = Device::connect(
        manifest.manifest().as_inventory().oak().mxid().as_str(),
        launch.launch().oak().device_config(),
    )
    .map_err(|source| {
        QualificationBootstrapError::before_hardware(QualificationBootstrapPrimaryError::common(
            NanoBootstrapPrimaryError::OakConnect(source),
        ))
    })?;
    let connected = match prepare_oak(
        &mut oak,
        launch.launch(),
        assets.navigation_shadow_config.bytes(),
        &calibration,
        plant_artifact_model,
        &candidate_admission,
        running,
    ) {
        Ok(connected) => connected,
        Err(primary) => return Err(close_oak_after_failure(primary, oak)),
    };

    let controller_serial_device = controller_server.serial_device().to_path_buf();
    let controller_owner_shutdown_timeout = controller_server.coordinated_shutdown_budget();
    let controller_owner = match V2ControllerOwner::start(
        controller_server,
        launch.launch().controller_server().command_udp_endpoint(),
    )
    .await
    {
        Ok(owner) => owner,
        Err(source) => {
            return Err(close_oak_after_failure(
                QualificationBootstrapPrimaryError::ControllerOwnerStart(source),
                oak,
            ));
        }
    };
    let observed_command_endpoint = controller_owner.command_address();

    let session = match WheelsOffCandidateActuationSession::acquire(
        candidate_admission,
        controller_clock_origin,
    ) {
        Ok(session) => session,
        Err(source) => {
            return Err(cleanup_after_session_start_failure(
                QualificationBootstrapPrimaryError::ControllerAcquire(source),
                controller_owner,
                controller_owner_shutdown_timeout,
                oak,
            )
            .await);
        }
    };
    let acquisition = match session.verified_controller_acquisition() {
        Ok(acquisition) => acquisition,
        Err(source) => {
            return Err(cleanup_after_session_failure(
                QualificationBootstrapPrimaryError::ControllerEvidence(source),
                session,
                controller_owner,
                controller_owner_shutdown_timeout,
                oak,
            )
            .await);
        }
    };

    let mut observed = NanoObservedInventoryBuilder::new();
    if let Err(source) = observed.observe_candidate_deployment(
        launch.launch().robot_id().as_str(),
        observed_command_endpoint,
    ) {
        return Err(cleanup_after_session_failure(
            QualificationBootstrapPrimaryError::ObservedInventoryEvidence(source),
            session,
            controller_owner,
            controller_owner_shutdown_timeout,
            oak,
        )
        .await);
    }
    if let Err(source) = observed.observe_oak(
        &connected.opened_identity,
        &connected.depthai_build_metadata,
        connected.usb_transport,
    ) {
        return Err(cleanup_after_session_failure(
            QualificationBootstrapPrimaryError::ObservedInventoryEvidence(source),
            session,
            controller_owner,
            controller_owner_shutdown_timeout,
            oak,
        )
        .await);
    }
    if let Err(source) = observed.observe_stm32(&controller_serial_device, acquisition) {
        return Err(cleanup_after_session_failure(
            QualificationBootstrapPrimaryError::ObservedInventoryEvidence(source),
            session,
            controller_owner,
            controller_owner_shutdown_timeout,
            oak,
        )
        .await);
    }
    if let Err(source) = observed.observe_head(&head) {
        return Err(cleanup_after_session_failure(
            QualificationBootstrapPrimaryError::ObservedInventoryEvidence(source),
            session,
            controller_owner,
            controller_owner_shutdown_timeout,
            oak,
        )
        .await);
    }
    if let Err(source) = observed.observe_eye(&eye_serial, eye_identity) {
        return Err(cleanup_after_session_failure(
            QualificationBootstrapPrimaryError::ObservedInventoryEvidence(source),
            session,
            controller_owner,
            controller_owner_shutdown_timeout,
            oak,
        )
        .await);
    }
    if let Err(source) = observed.observe_artifacts(&artifact_hashes) {
        return Err(cleanup_after_session_failure(
            QualificationBootstrapPrimaryError::ObservedInventoryEvidence(source),
            session,
            controller_owner,
            controller_owner_shutdown_timeout,
            oak,
        )
        .await);
    }

    let (stopped_controller, initial_zero, initial_stop) =
        match session.stop_now_with_last_applied() {
            Ok(stopped) => stopped,
            Err(source) => {
                return Err(cleanup_after_uncertain_stop(
                    QualificationBootstrapPrimaryError::ControllerStop(source),
                    controller_owner,
                    controller_owner_shutdown_timeout,
                    oak,
                )
                .await);
            }
        };
    if !initial_zero.is_confirmed_zero() {
        return Err(cleanup_after_confirmed_stop(
            QualificationBootstrapPrimaryError::InitialReceiptWasNotConfirmedZero,
            initial_stop,
            controller_owner,
            controller_owner_shutdown_timeout,
            oak,
        )
        .await);
    }

    let observed = match observed.build() {
        Ok(observed) => observed,
        Err(source) => {
            return Err(cleanup_after_confirmed_stop(
                QualificationBootstrapPrimaryError::ObservedInventoryBuild(source),
                initial_stop,
                controller_owner,
                controller_owner_shutdown_timeout,
                oak,
            )
            .await);
        }
    };
    let oak_usb_transport = observed.oak_super_speed();
    let exact_inventory_admission = match admit_exact_inventory(
        manifest.manifest().as_inventory().clone(),
        observed.into_inventory(),
    ) {
        Ok(admission) => admission,
        Err(source) => {
            return Err(cleanup_after_confirmed_stop(
                QualificationBootstrapPrimaryError::ExactInventory(source),
                initial_stop,
                controller_owner,
                controller_owner_shutdown_timeout,
                oak,
            )
            .await);
        }
    };

    Ok(PreparedNanoWheelsOffQualificationOwnedBootstrap {
        bootstrap: PreparedNanoWheelsOffQualificationBootstrap {
            roots,
            launch,
            assets,
            manifest,
            policy,
            calibration,
            plant,
            artifact_hashes,
            accessory_evidence: NanoBootstrapAccessoryEvidence {
                head,
                eye_serial,
                eye_identity,
            },
            oak_connected_identity: connected.opened_identity,
            oak_usb_transport,
            depthai_build_metadata: connected.depthai_build_metadata,
            stereo: connected.stereo,
            live: ParsedNanoLiveConfiguration {
                navigation: connected.navigation,
                occupancy_host_policy: connected.occupancy_host_policy,
            },
            exact_inventory_admission,
            candidate_limits,
            candidate_runtime_service_interval: connected.candidate_runtime_service_interval,
            initial_zero,
            initial_stop,
            stopped_controller,
            oak,
        },
        controller_owner,
        controller_owner_shutdown_timeout,
    })
}

struct ConnectedQualificationOak {
    opened_identity: ConnectedDeviceIdentity,
    usb_transport: oak_sys::UsbTransportAdmissionEvidence,
    depthai_build_metadata: DepthAiBuildMetadata,
    stereo: NanoBootstrapStereoEvidence,
    navigation: ShadowNavigationConfigV1,
    occupancy_host_policy: LiveOccupancyHostPolicy,
    candidate_runtime_service_interval: WheelsOffCandidateRuntimeServiceInterval,
}

fn prepare_oak(
    oak: &mut Device,
    launch: &NanoWheelsOffQualificationLaunchV1,
    navigation_bytes: &[u8],
    calibration: &NanoCalibrationArtifactV1,
    plant_artifact_model: super::mpc::PlantModelV1,
    candidate: &super::AdmittedWheelsOffCandidateController,
    running: &AtomicBool,
) -> Result<ConnectedQualificationOak, QualificationBootstrapPrimaryError> {
    let opened_identity = oak
        .connected_identity()
        .map_err(NanoBootstrapPrimaryError::OakConnectedIdentity)
        .map_err(QualificationBootstrapPrimaryError::common)?
        .clone();
    if opened_identity.mxid().is_empty() {
        return Err(QualificationBootstrapPrimaryError::common(
            NanoBootstrapPrimaryError::EmptyOpenedOakMxid,
        ));
    }
    calibration
        .require_connected_oak_mxid(opened_identity.mxid())
        .map_err(QualificationBootstrapPrimaryError::CalibrationBinding)?;
    let usb_transport = *oak
        .usb_transport_evidence()
        .map_err(NanoBootstrapPrimaryError::OakUsbTransport)
        .map_err(QualificationBootstrapPrimaryError::common)?;
    let depthai_build_metadata = oak_sys::depthai_build_metadata()
        .map_err(NanoBootstrapPrimaryError::DepthAiBuild)
        .map_err(QualificationBootstrapPrimaryError::common)?;
    let stereo = bootstrap_stereo(oak, launch.oak(), running)
        .map_err(QualificationBootstrapPrimaryError::common)?;
    calibration
        .require_observed_stereo(&stereo.calibration)
        .map_err(QualificationBootstrapPrimaryError::CalibrationBinding)?;
    let navigation = ShadowNavigationConfigV1::parse_json_bound_to_plant_artifact(
        navigation_bytes,
        stereo.runtime_depth_camera,
        plant_artifact_model,
    )
    .map_err(QualificationBootstrapPrimaryError::ShadowNavigation)?;
    calibration
        .require_navigation(&navigation)
        .map_err(QualificationBootstrapPrimaryError::CalibrationBinding)?;
    candidate
        .admit_shadow_mpc(navigation.mpc_solver().config())
        .map_err(QualificationBootstrapPrimaryError::CandidateMpc)?;
    let candidate_runtime_service_interval = candidate
        .limits()
        .admit_runtime_service_interval(navigation.control_period().as_duration())
        .map_err(QualificationBootstrapPrimaryError::CandidateRuntimeServiceInterval)?;
    Ok(ConnectedQualificationOak {
        opened_identity,
        usb_transport,
        depthai_build_metadata,
        stereo,
        navigation,
        occupancy_host_policy: launch.occupancy().host_policy(),
        candidate_runtime_service_interval,
    })
}

fn bind_policy_paths(
    roots: &NanoBootstrapRoots,
    launch: &NanoWheelsOffQualificationLaunchV1,
    configured_manifest: &Path,
    artifact_root: &Path,
) -> Result<(), QualificationBootstrapPrimaryError> {
    require_exact_manifest_path(
        roots.deployment_root(),
        launch.candidate_inventory_manifest().relative_path(),
        configured_manifest,
    )?;
    if artifact_root.strip_prefix(roots.deployment_root()).is_err() {
        return Err(
            QualificationBootstrapPrimaryError::ArtifactRootOutsideDeployment {
                deployment_root: roots.deployment_root().to_path_buf(),
                configured: artifact_root.to_path_buf(),
            },
        );
    }
    Ok(())
}

fn require_exact_manifest_path(
    deployment_root: &Path,
    launch_relative: &ArtifactRelativePath,
    configured_manifest: &Path,
) -> Result<(), QualificationBootstrapPrimaryError> {
    let launch_manifest = deployment_root.join(launch_relative.as_path());
    if launch_manifest != configured_manifest {
        return Err(QualificationBootstrapPrimaryError::ManifestPathMismatch {
            launch: launch_manifest,
            policy: configured_manifest.to_path_buf(),
        });
    }
    Ok(())
}

fn bind_calibration(
    roots: &NanoBootstrapRoots,
    launch: &NanoWheelsOffQualificationLaunchV1,
    loaded: &LoadedDeploymentAsset,
    policy: &ManifestBoundNanoAgentPolicyConfigV3,
    manifest: &kiko_device_inventory::DeviceInventoryManifestV1,
    hashes: &ManifestArtifactHashes,
) -> Result<(), QualificationBootstrapPrimaryError> {
    let requested = launch.calibration_artifact().artifact_id().as_str();
    let expected = manifest
        .artifacts()
        .iter()
        .find(|entry| {
            entry.kind() == ArtifactKind::Calibration && entry.artifact_id().as_str() == requested
        })
        .ok_or_else(
            || QualificationBootstrapPrimaryError::CalibrationNotInManifest {
                artifact_id: requested.to_owned(),
            },
        )?;
    if expected.sha256().as_bytes() != loaded.content_sha256().as_bytes() {
        return Err(
            QualificationBootstrapPrimaryError::CalibrationDigestMismatch {
                artifact_id: requested.to_owned(),
                launch: *loaded.content_sha256().as_bytes(),
                manifest: *expected.sha256().as_bytes(),
            },
        );
    }
    let binding = policy
        .inventory()
        .artifact_bindings()
        .iter()
        .find(|binding| {
            binding.kind() == ArtifactKind::Calibration
                && binding.artifact_id().as_str() == requested
        })
        .ok_or_else(
            || QualificationBootstrapPrimaryError::CalibrationPolicyBindingMissing {
                artifact_id: requested.to_owned(),
            },
        )?;
    let hashed = hashes
        .iter()
        .find(|entry| {
            entry.kind() == ArtifactKind::Calibration && entry.artifact_id().as_str() == requested
        })
        .ok_or_else(
            || QualificationBootstrapPrimaryError::CalibrationHashMissing {
                artifact_id: requested.to_owned(),
            },
        )?;
    if hashed.observed_sha256() != loaded.content_sha256().as_bytes() {
        return Err(
            QualificationBootstrapPrimaryError::CalibrationHashMismatch {
                artifact_id: requested.to_owned(),
                retained: *loaded.content_sha256().as_bytes(),
                observed: *hashed.observed_sha256(),
            },
        );
    }
    let deployed = calibration_deployment_relative_artifact_path(
        roots.deployment_root(),
        policy.inventory().artifact_root_path().as_path(),
        binding.relative_path(),
    )?;
    if &deployed != launch.calibration_artifact().asset().relative_path() {
        return Err(
            QualificationBootstrapPrimaryError::CalibrationPathMismatch {
                artifact_id: requested.to_owned(),
                launch: launch
                    .calibration_artifact()
                    .asset()
                    .relative_path()
                    .clone(),
                policy: deployed,
            },
        );
    }
    Ok(())
}

fn calibration_deployment_relative_artifact_path(
    deployment_root: &Path,
    artifact_root: &Path,
    artifact_relative: &ArtifactRelativePath,
) -> Result<ArtifactRelativePath, QualificationBootstrapPrimaryError> {
    let root_relative = artifact_root.strip_prefix(deployment_root).map_err(|_| {
        QualificationBootstrapPrimaryError::ArtifactRootOutsideDeployment {
            deployment_root: deployment_root.to_path_buf(),
            configured: artifact_root.to_path_buf(),
        }
    })?;
    let combined = root_relative.join(artifact_relative.as_path());
    let text = combined.to_str().ok_or_else(|| {
        QualificationBootstrapPrimaryError::CalibrationDeploymentPathNotUtf8 {
            path: combined.clone(),
        }
    })?;
    ArtifactRelativePath::parse(text.to_owned())
        .map_err(QualificationBootstrapPrimaryError::CalibrationDeploymentRelativePath)
}

fn select_plant(
    roots: &NanoBootstrapRoots,
    launch: &NanoWheelsOffQualificationLaunchV1,
    loaded: &LoadedDeploymentAsset,
    policy: &ManifestBoundNanoAgentPolicyConfigV3,
    manifest: &kiko_device_inventory::DeviceInventoryManifestV1,
    hashes: &ManifestArtifactHashes,
) -> Result<QualificationPlantEvidence, QualificationBootstrapPrimaryError> {
    let requested = launch.plant_artifact().artifact_id().as_str();
    let expected = manifest
        .artifacts()
        .iter()
        .find(|entry| {
            entry.kind() == ArtifactKind::Plant && entry.artifact_id().as_str() == requested
        })
        .ok_or_else(|| QualificationBootstrapPrimaryError::PlantNotInManifest {
            artifact_id: requested.to_owned(),
        })?;
    if expected.sha256().as_bytes() != loaded.content_sha256().as_bytes() {
        return Err(QualificationBootstrapPrimaryError::PlantDigestMismatch {
            artifact_id: requested.to_owned(),
            launch: *loaded.content_sha256().as_bytes(),
            manifest: *expected.sha256().as_bytes(),
        });
    }
    let binding = policy
        .inventory()
        .artifact_bindings()
        .iter()
        .find(|binding| {
            binding.kind() == ArtifactKind::Plant && binding.artifact_id().as_str() == requested
        })
        .ok_or_else(|| QualificationBootstrapPrimaryError::PlantBindingMissing {
            artifact_id: requested.to_owned(),
        })?;
    let deployed = deployment_relative_artifact_path(
        roots.deployment_root(),
        policy.inventory().artifact_root_path().as_path(),
        binding.relative_path(),
    )?;
    if &deployed != launch.plant_artifact().asset().relative_path() {
        return Err(QualificationBootstrapPrimaryError::PlantPathMismatch {
            artifact_id: requested.to_owned(),
            launch: launch.plant_artifact().asset().relative_path().clone(),
            policy: deployed,
        });
    }
    let hashed = hashes
        .iter()
        .find(|entry| {
            entry.kind() == ArtifactKind::Plant && entry.artifact_id().as_str() == requested
        })
        .ok_or_else(|| QualificationBootstrapPrimaryError::PlantHashMissing {
            artifact_id: requested.to_owned(),
        })?;
    if hashed.observed_sha256() != loaded.content_sha256().as_bytes() {
        return Err(QualificationBootstrapPrimaryError::PlantHashMismatch {
            artifact_id: requested.to_owned(),
            retained: *loaded.content_sha256().as_bytes(),
            observed: *hashed.observed_sha256(),
        });
    }
    Ok(QualificationPlantEvidence {
        artifact_id: *expected.artifact_id(),
        artifact_relative_path: binding.relative_path().clone(),
        deployment_relative_path: launch.plant_artifact().asset().relative_path().clone(),
        content_sha256: *loaded.content_sha256().as_bytes(),
    })
}

fn deployment_relative_artifact_path(
    deployment_root: &Path,
    artifact_root: &Path,
    artifact_relative: &ArtifactRelativePath,
) -> Result<ArtifactRelativePath, QualificationBootstrapPrimaryError> {
    let root_relative = artifact_root.strip_prefix(deployment_root).map_err(|_| {
        QualificationBootstrapPrimaryError::ArtifactRootOutsideDeployment {
            deployment_root: deployment_root.to_path_buf(),
            configured: artifact_root.to_path_buf(),
        }
    })?;
    let combined = root_relative.join(artifact_relative.as_path());
    let text = combined.to_str().ok_or_else(|| {
        QualificationBootstrapPrimaryError::PlantDeploymentPathNotUtf8 {
            path: combined.clone(),
        }
    })?;
    ArtifactRelativePath::parse(text.to_owned())
        .map_err(QualificationBootstrapPrimaryError::PlantDeploymentRelativePath)
}

async fn cleanup_after_session_failure(
    primary: QualificationBootstrapPrimaryError,
    session: WheelsOffCandidateActuationSession,
    owner: V2ControllerOwner,
    timeout: Duration,
    oak: Device,
) -> QualificationBootstrapError {
    let controller = match session.stop_now() {
        Ok((_stopped, receipt)) => QualificationControllerDisposition::ConfirmedStopped(receipt),
        Err(source) => QualificationControllerDisposition::StopUncertain(source),
    };
    QualificationBootstrapError {
        primary: Box::new(primary),
        controller,
        controller_owner: shutdown_owner(owner, timeout).await,
        oak_close: close_oak(oak),
    }
}

async fn cleanup_after_uncertain_stop(
    primary: QualificationBootstrapPrimaryError,
    owner: V2ControllerOwner,
    timeout: Duration,
    oak: Device,
) -> QualificationBootstrapError {
    QualificationBootstrapError {
        primary: Box::new(primary),
        controller: QualificationControllerDisposition::StopErrorRetainedByPrimary,
        controller_owner: shutdown_owner(owner, timeout).await,
        oak_close: close_oak(oak),
    }
}

async fn cleanup_after_confirmed_stop(
    primary: QualificationBootstrapPrimaryError,
    receipt: DisarmReceipt,
    owner: V2ControllerOwner,
    timeout: Duration,
    oak: Device,
) -> QualificationBootstrapError {
    QualificationBootstrapError {
        primary: Box::new(primary),
        controller: QualificationControllerDisposition::ConfirmedStopped(receipt),
        controller_owner: shutdown_owner(owner, timeout).await,
        oak_close: close_oak(oak),
    }
}

async fn cleanup_after_session_start_failure(
    primary: QualificationBootstrapPrimaryError,
    owner: V2ControllerOwner,
    timeout: Duration,
    oak: Device,
) -> QualificationBootstrapError {
    QualificationBootstrapError {
        primary: Box::new(primary),
        controller: QualificationControllerDisposition::SessionStartErrorRetainsDisposition,
        controller_owner: shutdown_owner(owner, timeout).await,
        oak_close: close_oak(oak),
    }
}

async fn shutdown_owner(
    owner: V2ControllerOwner,
    timeout: Duration,
) -> QualificationControllerOwnerDisposition {
    match owner.shutdown(timeout).await {
        Ok(()) => QualificationControllerOwnerDisposition::ConfirmedStopped,
        Err(source) => QualificationControllerOwnerDisposition::StopUncertain(source),
    }
}

fn close_oak_after_failure(
    primary: QualificationBootstrapPrimaryError,
    oak: Device,
) -> QualificationBootstrapError {
    QualificationBootstrapError {
        primary: Box::new(primary),
        controller: QualificationControllerDisposition::NotAcquired,
        controller_owner: QualificationControllerOwnerDisposition::NotStarted,
        oak_close: close_oak(oak),
    }
}

fn close_oak(oak: Device) -> NanoBootstrapOakCloseDisposition {
    match oak.close() {
        Ok(()) => NanoBootstrapOakCloseDisposition::ConfirmedClosed,
        Err(source) => NanoBootstrapOakCloseDisposition::CloseUncertain(source),
    }
}

pub struct QualificationBootstrapError {
    primary: Box<QualificationBootstrapPrimaryError>,
    controller: QualificationControllerDisposition,
    controller_owner: QualificationControllerOwnerDisposition,
    oak_close: NanoBootstrapOakCloseDisposition,
}

impl QualificationBootstrapError {
    fn before_hardware(primary: QualificationBootstrapPrimaryError) -> Self {
        Self {
            primary: Box::new(primary),
            controller: QualificationControllerDisposition::NotAcquired,
            controller_owner: QualificationControllerOwnerDisposition::NotStarted,
            oak_close: NanoBootstrapOakCloseDisposition::NotOpened,
        }
    }

    pub const fn primary(&self) -> &QualificationBootstrapPrimaryError {
        &self.primary
    }

    pub const fn controller(&self) -> &QualificationControllerDisposition {
        &self.controller
    }

    pub const fn controller_owner(&self) -> &QualificationControllerOwnerDisposition {
        &self.controller_owner
    }

    pub const fn oak_close(&self) -> &NanoBootstrapOakCloseDisposition {
        &self.oak_close
    }
}

impl fmt::Debug for QualificationBootstrapError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, formatter)
    }
}

impl fmt::Display for QualificationBootstrapError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "Nano wheels-off qualification bootstrap failed: {}; {}; {}; {}",
            self.primary, self.controller, self.controller_owner, self.oak_close
        )
    }
}

impl std::error::Error for QualificationBootstrapError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.primary.as_ref())
    }
}

pub enum QualificationControllerDisposition {
    NotAcquired,
    SessionStartErrorRetainsDisposition,
    ConfirmedStopped(DisarmReceipt),
    StopUncertain(LiveActuationError),
    StopErrorRetainedByPrimary,
}

impl fmt::Debug for QualificationControllerDisposition {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, formatter)
    }
}

impl fmt::Display for QualificationControllerDisposition {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotAcquired => formatter.write_str("candidate controller was not acquired"),
            Self::SessionStartErrorRetainsDisposition => formatter
                .write_str("candidate session-start error retains its exact stop disposition"),
            Self::ConfirmedStopped(receipt) => write!(
                formatter,
                "candidate controller stop confirmed at {} ns",
                receipt.acknowledged_at().nanos_since_clock_start()
            ),
            Self::StopUncertain(source) => {
                write!(formatter, "candidate controller stop uncertain: {source}")
            }
            Self::StopErrorRetainedByPrimary => formatter.write_str(
                "candidate controller stop uncertain; exact error retained by primary failure",
            ),
        }
    }
}

pub enum QualificationControllerOwnerDisposition {
    NotStarted,
    ConfirmedStopped,
    StopUncertain(V2ControllerOwnerTerminationError),
}

impl fmt::Debug for QualificationControllerOwnerDisposition {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, formatter)
    }
}

impl fmt::Display for QualificationControllerOwnerDisposition {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotStarted => formatter.write_str("controller owner was not started"),
            Self::ConfirmedStopped => formatter.write_str("controller owner shutdown confirmed"),
            Self::StopUncertain(source) => {
                write!(formatter, "controller owner shutdown uncertain: {source}")
            }
        }
    }
}

#[derive(Debug)]
pub enum QualificationBootstrapPrimaryError {
    CommonBootstrap(Box<NanoBootstrapPrimaryError>),
    LaunchLoad(NanoWheelsOffQualificationLaunchLoadError),
    BoundAssetLoad {
        role: NanoWheelsOffQualificationAssetRole,
        source: NanoLaunchBoundAssetLoadError,
    },
    AgentPolicy(NanoAgentPolicyConfigParseError),
    ManifestPathMismatch {
        launch: PathBuf,
        policy: PathBuf,
    },
    ArtifactRootOutsideDeployment {
        deployment_root: PathBuf,
        configured: PathBuf,
    },
    Manifest(ManifestLoadError),
    RobotIdMismatch {
        launch: String,
        manifest: String,
    },
    AccessoryManifestBinding(NanoAccessoryManifestBindingError),
    ArtifactHash(ArtifactHashError),
    CalibrationArtifact(NanoCalibrationArtifactParseError),
    CalibrationBinding(NanoCalibrationBindingError),
    CalibrationNotInManifest {
        artifact_id: String,
    },
    CalibrationPolicyBindingMissing {
        artifact_id: String,
    },
    CalibrationHashMissing {
        artifact_id: String,
    },
    CalibrationDigestMismatch {
        artifact_id: String,
        launch: [u8; 32],
        manifest: [u8; 32],
    },
    CalibrationHashMismatch {
        artifact_id: String,
        retained: [u8; 32],
        observed: [u8; 32],
    },
    CalibrationPathMismatch {
        artifact_id: String,
        launch: ArtifactRelativePath,
        policy: ArtifactRelativePath,
    },
    CalibrationDeploymentPathNotUtf8 {
        path: PathBuf,
    },
    CalibrationDeploymentRelativePath(ArtifactRelativePathError),
    PlantNotInManifest {
        artifact_id: String,
    },
    PlantBindingMissing {
        artifact_id: String,
    },
    PlantHashMissing {
        artifact_id: String,
    },
    PlantDigestMismatch {
        artifact_id: String,
        launch: [u8; 32],
        manifest: [u8; 32],
    },
    PlantHashMismatch {
        artifact_id: String,
        retained: [u8; 32],
        observed: [u8; 32],
    },
    PlantPathMismatch {
        artifact_id: String,
        launch: ArtifactRelativePath,
        policy: ArtifactRelativePath,
    },
    PlantDeploymentPathNotUtf8 {
        path: PathBuf,
    },
    PlantDeploymentRelativePath(ArtifactRelativePathError),
    ControllerServer(ServerConfigError),
    CandidatePolicy(WheelsOffCandidatePolicyError),
    CandidateBinding(WheelsOffCandidateControllerBindingError),
    CandidateMpc(CandidateMpcBindingError),
    CandidateRuntimeServiceInterval(CandidateRuntimeServiceIntervalError),
    ShadowNavigation(ShadowNavigationConfigParseError),
    HeadProbe(Box<SerialHeadProbeError>),
    EyeProbe(IdentityProbeError),
    ControllerOwnerStart(V2ControllerOwnerStartError),
    ControllerAcquire(CandidateActuationSessionStartError),
    ControllerEvidence(LiveActuationError),
    ControllerStop(LiveActuationError),
    InitialReceiptWasNotConfirmedZero,
    ObservedInventoryEvidence(NanoObservedInventoryEvidenceError),
    ObservedInventoryBuild(NanoObservedInventoryBuildError),
    ExactInventory(InventoryMismatchReport),
}

impl QualificationBootstrapPrimaryError {
    fn common(source: NanoBootstrapPrimaryError) -> Self {
        Self::CommonBootstrap(Box::new(source))
    }
}

impl fmt::Display for QualificationBootstrapPrimaryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "qualification bootstrap boundary rejected input or live evidence: {self:?}"
        )
    }
}

impl std::error::Error for QualificationBootstrapPrimaryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::CommonBootstrap(source) => Some(source),
            Self::LaunchLoad(source) => Some(source),
            Self::BoundAssetLoad { source, .. } => Some(source),
            Self::AgentPolicy(source) => Some(source),
            Self::Manifest(source) => Some(source),
            Self::AccessoryManifestBinding(source) => Some(source),
            Self::ArtifactHash(source) => Some(source),
            Self::CalibrationArtifact(source) => Some(source),
            Self::CalibrationBinding(source) => Some(source),
            Self::CalibrationDeploymentRelativePath(source) => Some(source),
            Self::PlantDeploymentRelativePath(source) => Some(source),
            Self::ControllerServer(source) => Some(source),
            Self::CandidatePolicy(source) => Some(source),
            Self::CandidateBinding(source) => Some(source),
            Self::CandidateMpc(source) => Some(source),
            Self::CandidateRuntimeServiceInterval(source) => Some(source),
            Self::ShadowNavigation(source) => Some(source),
            Self::HeadProbe(source) => Some(source.as_ref()),
            Self::EyeProbe(source) => Some(source),
            Self::ControllerOwnerStart(source) => Some(source),
            Self::ControllerAcquire(source) => Some(source),
            Self::ControllerEvidence(source) | Self::ControllerStop(source) => Some(source),
            Self::ObservedInventoryEvidence(source) => Some(source),
            Self::ObservedInventoryBuild(source) => Some(source),
            Self::ExactInventory(source) => Some(source),
            Self::ManifestPathMismatch { .. }
            | Self::ArtifactRootOutsideDeployment { .. }
            | Self::RobotIdMismatch { .. }
            | Self::CalibrationNotInManifest { .. }
            | Self::CalibrationPolicyBindingMissing { .. }
            | Self::CalibrationHashMissing { .. }
            | Self::CalibrationDigestMismatch { .. }
            | Self::CalibrationHashMismatch { .. }
            | Self::CalibrationPathMismatch { .. }
            | Self::CalibrationDeploymentPathNotUtf8 { .. }
            | Self::PlantNotInManifest { .. }
            | Self::PlantBindingMissing { .. }
            | Self::PlantHashMissing { .. }
            | Self::PlantDigestMismatch { .. }
            | Self::PlantHashMismatch { .. }
            | Self::PlantPathMismatch { .. }
            | Self::PlantDeploymentPathNotUtf8 { .. }
            | Self::InitialReceiptWasNotConfirmedZero => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_manifest_path_must_equal_the_launch_bound_asset() {
        let deployment = Path::new("/opt/kiko/releases/current");
        let relative =
            ArtifactRelativePath::parse("inventory/candidate-v2.json".to_owned()).expect("path");
        require_exact_manifest_path(
            deployment,
            &relative,
            Path::new("/opt/kiko/releases/current/inventory/candidate-v2.json"),
        )
        .expect("exact path");
        assert!(matches!(
            require_exact_manifest_path(
                deployment,
                &relative,
                Path::new("/opt/kiko/releases/current/inventory/other-v2.json"),
            ),
            Err(QualificationBootstrapPrimaryError::ManifestPathMismatch { .. })
        ));
    }

    #[test]
    fn artifact_path_conversion_is_exact_and_rejects_escape() {
        let deployment = Path::new("/opt/kiko/releases/current");
        let relative =
            ArtifactRelativePath::parse("plant/identified.json".to_owned()).expect("path");
        assert_eq!(
            deployment_relative_artifact_path(
                deployment,
                Path::new("/opt/kiko/releases/current/artifacts"),
                &relative,
            )
            .expect("deployment path")
            .as_str(),
            "artifacts/plant/identified.json"
        );
        assert!(matches!(
            deployment_relative_artifact_path(
                deployment,
                Path::new("/opt/kiko/shared-artifacts"),
                &relative,
            ),
            Err(QualificationBootstrapPrimaryError::ArtifactRootOutsideDeployment { .. })
        ));
    }
}
