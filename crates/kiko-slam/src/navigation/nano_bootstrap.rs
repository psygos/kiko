//! Production Nano bootstrap up to one disarmed, fully admitted live runtime.
//!
//! This module owns the cold-start ordering that cannot be expressed by the
//! individual parsers alone:
//!
//! 1. load every launch-bound input without following symlinks;
//! 2. parse policy, manifest, controller, navigation, and actuation contracts
//!    exactly once;
//! 3. issue read-only head and eye identity probes;
//! 4. open one exact OAK at SuperSpeed, retain its first stereo frames, and
//!    derive the runtime projection contract from observed intrinsics;
//! 5. acquire the sole controller session at an acknowledged exact zero;
//! 6. build observed inventory only from retained runtime evidence; and
//! 7. perform production admission, leaving the supervisor disarmed.
//!
//! No accessory actor is started here, no head torque consent is exercised,
//! and no motion-bearing API is exposed before exact admission.

use std::fmt;
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use kiko_device_inventory::{
    ArtifactHashError, ArtifactId, ArtifactKind, ArtifactRelativePath, ArtifactRelativePathError,
    LoadedDeploymentAsset, ManifestArtifactHashes, ManifestLoadError, hash_manifest_artifacts,
    load_expected_manifest_v1_file,
};
use kiko_eye_runtime::{
    EyeIdentityObservation, IdentityProbeConfig, IdentityProbeError,
    SerialConfigurationEvidence as EyeSerialConfigurationEvidence, probe_serial_eye_identity,
};
use kiko_head_runtime::{
    HeadProbeConfig, HeadProbeReport, SerialHeadProbeError, probe_serial_head,
};
use kiko_supervisor_core::{ReadinessEpoch, SupervisorState};
use oak_sys::{
    CalibrationError as OakCalibrationError, CloseError as OakCloseError,
    ConnectedDeviceIdentityError, DepthAiBuildMetadata, DepthAiBuildMetadataError, Device,
    ImageError, ImageFrame as OakImageFrame, StreamId as OakStreamId, UsbTransportEvidenceError,
};
use robot_command_client::DisarmReceipt;
use robot_server::config::{ControllerServerConfigV1, ServerConfigError};

use super::actuation::LiveActuationError;
use super::{
    ManifestBoundNanoAgentPolicyConfigV1, NanoAccessoryManifestBindingError,
    NanoAgentLaunchLoadError, NanoAgentLaunchV1, NanoAgentPolicyConfigParseError,
    NanoAgentPolicyConfigV1, NanoLaunchAssetRole, NanoLaunchBoundAssetLoadError,
    NanoObservedInventoryBuildError, NanoObservedInventoryBuilder,
    NanoObservedInventoryEvidenceError, NanoProductionAdmissionError,
    NanoProductionAdmissionTimeline, NanoProductionAdmissionTimelineError,
    NavigationActuationConfigV1, NavigationClockEpoch, PendingLiveMpcControlDriver,
    PreparedNanoProductionRuntime, ShadowNavigationConfigParseError, ShadowNavigationConfigV1,
    load_nano_agent_launch_v1,
};
use crate::dataset::{Calibration, CameraIntrinsics};
use crate::dense::occupancy::{DepthCameraModel, DepthToTrackingCamera};
use crate::live_runtime::LiveOccupancyHostPolicy;
use crate::{FrameDimensions, HostMonotonicTimestamp, RectifiedStereo, RectifiedStereoError};

const MAX_NANO_BOOTSTRAP_ROOT_BYTES: usize = 1_024;
const STEREO_POLL_TIMEOUT_MS: u32 = 50;
const STEREO_IDLE_SLEEP: Duration = Duration::from_micros(500);
const MAX_STEREO_BOOTSTRAP_WAIT: Duration = Duration::from_secs(15);

/// Canonical service-owned input and output roots.
///
/// This is lexical admission only. Input files are subsequently opened by the
/// no-follow deployment and inventory loaders. The state root is retained for
/// the runtime writer; this module neither creates nor writes it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NanoBootstrapRoots {
    deployment_root: PathBuf,
    state_root: PathBuf,
}

impl NanoBootstrapRoots {
    pub fn try_new(
        deployment_root: PathBuf,
        state_root: PathBuf,
    ) -> Result<Self, NanoBootstrapRootError> {
        validate_absolute_root(NanoBootstrapRootKind::Deployment, &deployment_root)?;
        validate_absolute_root(NanoBootstrapRootKind::State, &state_root)?;
        if deployment_root.starts_with(&state_root) || state_root.starts_with(&deployment_root) {
            return Err(NanoBootstrapRootError::OverlappingRoots {
                deployment_root,
                state_root,
            });
        }
        Ok(Self {
            deployment_root,
            state_root,
        })
    }

    pub fn deployment_root(&self) -> &Path {
        &self.deployment_root
    }

    pub fn state_root(&self) -> &Path {
        &self.state_root
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoBootstrapRootKind {
    Deployment,
    State,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NanoBootstrapRootError {
    Empty {
        kind: NanoBootstrapRootKind,
    },
    TooLong {
        kind: NanoBootstrapRootKind,
        actual_bytes: usize,
        maximum_bytes: usize,
    },
    NotAbsolute {
        kind: NanoBootstrapRootKind,
        path: PathBuf,
    },
    FilesystemRootNotAllowed {
        kind: NanoBootstrapRootKind,
    },
    NonCanonicalComponent {
        kind: NanoBootstrapRootKind,
        path: PathBuf,
    },
    OverlappingRoots {
        deployment_root: PathBuf,
        state_root: PathBuf,
    },
}

impl fmt::Display for NanoBootstrapRootError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid Nano bootstrap root: {self:?}")
    }
}

impl std::error::Error for NanoBootstrapRootError {}

fn validate_absolute_root(
    kind: NanoBootstrapRootKind,
    path: &Path,
) -> Result<(), NanoBootstrapRootError> {
    let bytes = path.as_os_str().as_bytes();
    if bytes.is_empty() {
        return Err(NanoBootstrapRootError::Empty { kind });
    }
    if bytes.len() > MAX_NANO_BOOTSTRAP_ROOT_BYTES {
        return Err(NanoBootstrapRootError::TooLong {
            kind,
            actual_bytes: bytes.len(),
            maximum_bytes: MAX_NANO_BOOTSTRAP_ROOT_BYTES,
        });
    }
    if !path.is_absolute() {
        return Err(NanoBootstrapRootError::NotAbsolute {
            kind,
            path: path.to_path_buf(),
        });
    }
    if bytes == b"/" {
        return Err(NanoBootstrapRootError::FilesystemRootNotAllowed { kind });
    }
    if bytes.last() == Some(&b'/')
        || bytes[1..]
            .split(|byte| *byte == b'/')
            .any(|component| component.is_empty() || component == b"." || component == b"..")
    {
        return Err(NanoBootstrapRootError::NonCanonicalComponent {
            kind,
            path: path.to_path_buf(),
        });
    }
    Ok(())
}

/// One process-lifetime production bootstrap request.
pub struct NanoBootstrapRequest<'running> {
    roots: NanoBootstrapRoots,
    launch_relative_path: ArtifactRelativePath,
    controller_clock_origin: Instant,
    navigation_clock_epoch: NavigationClockEpoch,
    readiness_epoch: ReadinessEpoch,
    running: &'running AtomicBool,
}

impl<'running> NanoBootstrapRequest<'running> {
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        deployment_root: PathBuf,
        state_root: PathBuf,
        launch_relative_path: String,
        controller_clock_origin: Instant,
        navigation_clock_epoch: NavigationClockEpoch,
        readiness_epoch: ReadinessEpoch,
        running: &'running AtomicBool,
    ) -> Result<Self, NanoBootstrapRequestError> {
        Ok(Self {
            roots: NanoBootstrapRoots::try_new(deployment_root, state_root)
                .map_err(NanoBootstrapRequestError::Roots)?,
            launch_relative_path: ArtifactRelativePath::parse(launch_relative_path)
                .map_err(NanoBootstrapRequestError::LaunchRelativePath)?,
            controller_clock_origin,
            navigation_clock_epoch,
            readiness_epoch,
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
pub enum NanoBootstrapRequestError {
    Roots(NanoBootstrapRootError),
    LaunchRelativePath(ArtifactRelativePathError),
}

impl fmt::Display for NanoBootstrapRequestError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid Nano bootstrap request: {self:?}")
    }
}

impl std::error::Error for NanoBootstrapRequestError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Roots(source) => Some(source),
            Self::LaunchRelativePath(source) => Some(source),
        }
    }
}

/// Exact bytes retained for every launch-bound runtime input.
///
/// Model and shared-library consumers must use these identities when they
/// create their own pinned runtime boundary. Reopening a pathname later is not
/// evidence that the same bytes were selected.
#[derive(Debug)]
pub struct LoadedNanoBootstrapAssets {
    pub agent_policy: LoadedDeploymentAsset,
    pub navigation_shadow_config: LoadedDeploymentAsset,
    pub physical_actuation_config: LoadedDeploymentAsset,
    pub controller_server_contract: LoadedDeploymentAsset,
    pub plant_artifact: LoadedDeploymentAsset,
    pub onnx_runtime_library: LoadedDeploymentAsset,
    pub superpoint_model: LoadedDeploymentAsset,
    pub lightglue_model: LoadedDeploymentAsset,
}

impl LoadedNanoBootstrapAssets {
    fn load(
        deployment_root: &Path,
        launch: &NanoAgentLaunchV1,
    ) -> Result<Self, NanoBootstrapPrimaryError> {
        let load = |role| {
            launch
                .asset(role)
                .load_exact(deployment_root)
                .map_err(|source| NanoBootstrapPrimaryError::BoundAssetLoad { role, source })
        };
        Ok(Self {
            agent_policy: load(NanoLaunchAssetRole::AgentPolicy)?,
            navigation_shadow_config: load(NanoLaunchAssetRole::NavigationShadowConfig)?,
            physical_actuation_config: load(NanoLaunchAssetRole::PhysicalActuationConfig)?,
            controller_server_contract: load(NanoLaunchAssetRole::ControllerServerContract)?,
            plant_artifact: load(NanoLaunchAssetRole::PlantArtifact)?,
            onnx_runtime_library: load(NanoLaunchAssetRole::OnnxRuntimeLibrary)?,
            superpoint_model: load(NanoLaunchAssetRole::SuperpointModel)?,
            lightglue_model: load(NanoLaunchAssetRole::LightglueModel)?,
        })
    }
}

/// Read-only accessory evidence collected before any actor or torque consent.
#[derive(Debug)]
pub struct NanoBootstrapAccessoryEvidence {
    pub head: HeadProbeReport,
    pub eye_serial: EyeSerialConfigurationEvidence,
    pub eye_identity: EyeIdentityObservation,
}

/// First observed stereo data and its typed projection contract.
///
/// The frames are returned, not discarded, so the live pairer can retain
/// their native device timestamps and sequences.
#[derive(Debug)]
pub struct NanoBootstrapStereoEvidence {
    pub left: OakImageFrame,
    pub right: OakImageFrame,
    pub calibration: Calibration,
    pub runtime_depth_camera: DepthCameraModel,
}

/// Parsed, mutually bound live configuration. No environment fallback was
/// consulted while constructing this value.
pub struct ParsedNanoLiveConfiguration {
    pub navigation: ShadowNavigationConfigV1,
    pub occupancy_host_policy: LiveOccupancyHostPolicy,
    pub controller_server: ControllerServerConfigV1,
}

/// Successful cold-start handoff to the sole production runtime.
#[must_use = "the returned OAK and controller owners require explicit lifecycle handling"]
pub struct PreparedNanoBootstrap {
    pub roots: NanoBootstrapRoots,
    pub launch: super::LoadedNanoAgentLaunchV1,
    pub assets: LoadedNanoBootstrapAssets,
    pub accessory_evidence: NanoBootstrapAccessoryEvidence,
    pub depthai_build_metadata: DepthAiBuildMetadata,
    pub stereo: NanoBootstrapStereoEvidence,
    pub live: ParsedNanoLiveConfiguration,
    pub runtime: PreparedNanoProductionRuntime,
    pub oak: Device,
}

/// Complete production bootstrap.
///
/// The head and eye probes are finite, identity-only/read-only exchanges. The
/// OAK is then opened once with the exact launch graph. Controller ownership is
/// acquired last; every subsequent failure consumes it through an explicit
/// disarm before attempting to close the OAK.
pub async fn bootstrap_nano_production(
    request: NanoBootstrapRequest<'_>,
) -> Result<PreparedNanoBootstrap, NanoBootstrapError> {
    let NanoBootstrapRequest {
        roots,
        launch_relative_path,
        controller_clock_origin,
        navigation_clock_epoch,
        readiness_epoch,
        running,
    } = request;

    require_running(running).map_err(NanoBootstrapError::before_hardware)?;
    let launch = load_nano_agent_launch_v1(roots.deployment_root(), launch_relative_path).map_err(
        |source| NanoBootstrapError::before_hardware(NanoBootstrapPrimaryError::LaunchLoad(source)),
    )?;
    let assets = LoadedNanoBootstrapAssets::load(roots.deployment_root(), launch.launch())
        .map_err(NanoBootstrapError::before_hardware)?;

    let policy =
        NanoAgentPolicyConfigV1::parse_json(assets.agent_policy.bytes()).map_err(|source| {
            NanoBootstrapError::before_hardware(NanoBootstrapPrimaryError::AgentPolicy(source))
        })?;
    require_policy_paths_within_deployment(&roots, &policy)
        .map_err(NanoBootstrapError::before_hardware)?;
    let loaded_manifest = load_expected_manifest_v1_file(
        policy.inventory().manifest_path().as_path(),
    )
    .map_err(|source| {
        NanoBootstrapError::before_hardware(NanoBootstrapPrimaryError::Manifest(source))
    })?;
    let probe_policy = policy
        .clone()
        .bind_accessories_to_manifest(loaded_manifest.manifest())
        .map_err(|source| {
            NanoBootstrapError::before_hardware(
                NanoBootstrapPrimaryError::AccessoryManifestBinding(source),
            )
        })?;
    let (head_probe_config, eye_probe_config) = derive_required_probe_configs(&probe_policy)
        .map_err(NanoBootstrapError::before_hardware)?;

    let artifact_hashes = hash_manifest_artifacts(
        loaded_manifest.manifest(),
        policy.inventory().artifact_root_path().as_path(),
        policy.inventory().artifact_bindings().clone(),
    )
    .map_err(|source| {
        NanoBootstrapError::before_hardware(NanoBootstrapPrimaryError::ArtifactHash(source))
    })?;
    let selected_plant = select_plant_artifact(
        &roots,
        launch.launch(),
        &assets.plant_artifact,
        &policy,
        loaded_manifest.manifest(),
        &artifact_hashes,
    )
    .map_err(NanoBootstrapError::before_hardware)?;

    let controller_server = ControllerServerConfigV1::parse_json(
        assets.controller_server_contract.bytes(),
    )
    .map_err(|source| {
        NanoBootstrapError::before_hardware(NanoBootstrapPrimaryError::ControllerServerContract(
            source,
        ))
    })?;
    bind_controller_contract_to_manifest(
        launch.launch(),
        &controller_server,
        loaded_manifest.manifest(),
    )
    .map_err(NanoBootstrapError::before_hardware)?;

    let inventory_started_at =
        timestamp_since(controller_clock_origin).map_err(NanoBootstrapError::before_hardware)?;
    require_running(running).map_err(NanoBootstrapError::before_hardware)?;
    let head = probe_serial_head(&head_probe_config)
        .await
        .map_err(|source| {
            NanoBootstrapError::before_hardware(NanoBootstrapPrimaryError::HeadProbe(Box::new(
                source,
            )))
        })?;
    require_running(running).map_err(NanoBootstrapError::before_hardware)?;
    let (eye_serial, eye_identity) =
        probe_serial_eye_identity(&eye_probe_config)
            .await
            .map_err(|source| {
                NanoBootstrapError::before_hardware(NanoBootstrapPrimaryError::EyeProbe(source))
            })?;
    require_running(running).map_err(NanoBootstrapError::before_hardware)?;

    let mut oak = Device::connect(
        loaded_manifest.manifest().oak().mxid().as_str(),
        launch.launch().oak().device_config(),
    )
    .map_err(|source| {
        NanoBootstrapError::before_hardware(NanoBootstrapPrimaryError::OakConnect(source))
    })?;

    let connected = match prepare_connected_oak(
        &mut oak,
        launch.launch(),
        assets.navigation_shadow_config.bytes(),
        assets.physical_actuation_config.bytes(),
        &controller_server,
        loaded_manifest.manifest().robot_id().as_str(),
        running,
    ) {
        Ok(connected) => connected,
        Err(primary) => return Err(close_oak_after_failure(primary, oak)),
    };

    let (pending, initial_zero) =
        match PendingLiveMpcControlDriver::acquire(&connected.actuation, controller_clock_origin) {
            Ok(acquired) => acquired,
            Err(source) => {
                return Err(close_oak_after_failure(
                    NanoBootstrapPrimaryError::ControllerAcquire(source),
                    oak,
                ));
            }
        };
    let acquisition = match pending.verified_controller_acquisition() {
        Ok(acquisition) => acquisition,
        Err(source) => {
            return Err(cleanup_after_pending_failure(
                NanoBootstrapPrimaryError::ControllerEvidence(source),
                pending,
                oak,
            ));
        }
    };

    let mut observed = NanoObservedInventoryBuilder::new();
    if let Err(source) = observed.observe_deployment(&connected.actuation) {
        return Err(cleanup_after_pending_failure(
            NanoBootstrapPrimaryError::ObservedInventoryEvidence(source),
            pending,
            oak,
        ));
    }
    let opened_identity = match oak.connected_identity() {
        Ok(identity) => identity,
        Err(source) => {
            return Err(cleanup_after_pending_failure(
                NanoBootstrapPrimaryError::OakConnectedIdentity(source),
                pending,
                oak,
            ));
        }
    };
    let usb_transport = match oak.usb_transport_evidence() {
        Ok(evidence) => *evidence,
        Err(source) => {
            return Err(cleanup_after_pending_failure(
                NanoBootstrapPrimaryError::OakUsbTransport(source),
                pending,
                oak,
            ));
        }
    };
    if let Err(source) = observed.observe_oak(
        opened_identity,
        &connected.depthai_build_metadata,
        usb_transport,
    ) {
        return Err(cleanup_after_pending_failure(
            NanoBootstrapPrimaryError::ObservedInventoryEvidence(source),
            pending,
            oak,
        ));
    }
    if let Err(source) = observed.observe_stm32(controller_server.serial_device(), acquisition) {
        return Err(cleanup_after_pending_failure(
            NanoBootstrapPrimaryError::ObservedInventoryEvidence(source),
            pending,
            oak,
        ));
    }
    if let Err(source) = observed.observe_head(&head) {
        return Err(cleanup_after_pending_failure(
            NanoBootstrapPrimaryError::ObservedInventoryEvidence(source),
            pending,
            oak,
        ));
    }
    if let Err(source) = observed.observe_eye(&eye_serial, eye_identity) {
        return Err(cleanup_after_pending_failure(
            NanoBootstrapPrimaryError::ObservedInventoryEvidence(source),
            pending,
            oak,
        ));
    }
    if let Err(source) = observed.observe_artifacts(&artifact_hashes) {
        return Err(cleanup_after_pending_failure(
            NanoBootstrapPrimaryError::ObservedInventoryEvidence(source),
            pending,
            oak,
        ));
    }
    let observed = match observed.build() {
        Ok(observed) => observed,
        Err(source) => {
            return Err(cleanup_after_pending_failure(
                NanoBootstrapPrimaryError::ObservedInventoryBuild(source),
                pending,
                oak,
            ));
        }
    };

    let readiness_admitted_at = match timestamp_since(controller_clock_origin) {
        Ok(timestamp) => timestamp,
        Err(primary) => {
            return Err(cleanup_after_pending_failure(primary, pending, oak));
        }
    };
    let timeline = match NanoProductionAdmissionTimeline::try_new(
        readiness_epoch,
        navigation_clock_epoch,
        inventory_started_at,
        readiness_admitted_at,
    ) {
        Ok(timeline) => timeline,
        Err(source) => {
            return Err(cleanup_after_pending_failure(
                NanoBootstrapPrimaryError::AdmissionTimeline(source),
                pending,
                oak,
            ));
        }
    };

    let runtime = match PreparedNanoProductionRuntime::admit(
        policy,
        loaded_manifest,
        artifact_hashes,
        observed,
        connected.actuation,
        selected_plant.id,
        selected_plant.relative_path,
        pending,
        initial_zero,
        timeline,
    ) {
        Ok(runtime) => runtime,
        Err(source) => {
            let oak_close = close_oak(oak);
            return Err(NanoBootstrapError {
                primary: Box::new(NanoBootstrapPrimaryError::ProductionAdmission(source)),
                controller: NanoBootstrapControllerDisposition::AdmissionErrorRetainsStop,
                oak_close,
            });
        }
    };

    debug_assert!(matches!(
        runtime.startup().authority.state(),
        SupervisorState::Disarmed { .. }
    ));
    Ok(PreparedNanoBootstrap {
        roots,
        launch,
        assets,
        accessory_evidence: NanoBootstrapAccessoryEvidence {
            head,
            eye_serial,
            eye_identity,
        },
        depthai_build_metadata: connected.depthai_build_metadata,
        stereo: connected.stereo,
        live: ParsedNanoLiveConfiguration {
            navigation: connected.navigation,
            occupancy_host_policy: connected.occupancy_host_policy,
            controller_server,
        },
        runtime,
        oak,
    })
}

struct ConnectedOakPreparation {
    depthai_build_metadata: DepthAiBuildMetadata,
    stereo: NanoBootstrapStereoEvidence,
    navigation: ShadowNavigationConfigV1,
    occupancy_host_policy: LiveOccupancyHostPolicy,
    actuation: NavigationActuationConfigV1,
}

fn prepare_connected_oak(
    oak: &mut Device,
    launch: &NanoAgentLaunchV1,
    navigation_bytes: &[u8],
    actuation_bytes: &[u8],
    controller_server: &ControllerServerConfigV1,
    robot_id: &str,
    running: &AtomicBool,
) -> Result<ConnectedOakPreparation, NanoBootstrapPrimaryError> {
    let connected_identity = oak
        .connected_identity()
        .map_err(NanoBootstrapPrimaryError::OakConnectedIdentity)?;
    if connected_identity.mxid().is_empty() {
        return Err(NanoBootstrapPrimaryError::EmptyOpenedOakMxid);
    }
    let _usb = oak
        .usb_transport_evidence()
        .map_err(NanoBootstrapPrimaryError::OakUsbTransport)?;
    let depthai_build_metadata =
        oak_sys::depthai_build_metadata().map_err(NanoBootstrapPrimaryError::DepthAiBuild)?;
    let stereo = bootstrap_stereo(oak, launch, running)?;
    let navigation =
        ShadowNavigationConfigV1::parse_json(navigation_bytes, stereo.runtime_depth_camera)
            .map_err(NanoBootstrapPrimaryError::ShadowNavigation)?;
    let occupancy_host_policy = launch.occupancy().host_policy();
    let actuation = NavigationActuationConfigV1::parse_and_authorize(
        actuation_bytes,
        robot_id,
        navigation_bytes,
        navigation.mpc_solver().model(),
        navigation.solver_budget(),
        navigation.control_period(),
    )
    .map_err(NanoBootstrapPrimaryError::Actuation)?;
    bind_controller_contract_to_actuation(launch, controller_server, &actuation)?;
    Ok(ConnectedOakPreparation {
        depthai_build_metadata,
        stereo,
        navigation,
        occupancy_host_policy,
        actuation,
    })
}

fn bootstrap_stereo(
    oak: &mut Device,
    launch: &NanoAgentLaunchV1,
    running: &AtomicBool,
) -> Result<NanoBootstrapStereoEvidence, NanoBootstrapPrimaryError> {
    let started = Instant::now();
    let mut left = None;
    let mut right = None;
    while left.is_none() || right.is_none() {
        require_running(running)?;
        if started.elapsed() >= MAX_STEREO_BOOTSTRAP_WAIT {
            return Err(NanoBootstrapPrimaryError::StereoTimedOut {
                maximum_wait: MAX_STEREO_BOOTSTRAP_WAIT,
                received_left: left.is_some(),
                received_right: right.is_some(),
            });
        }
        let mut received = false;
        if left.is_none() {
            match oak.mono_left(STEREO_POLL_TIMEOUT_MS) {
                Ok(frame) => {
                    left = Some(frame);
                    received = true;
                }
                Err(ImageError::Timeout { .. } | ImageError::QueueEmpty) => {}
                Err(source) => return Err(NanoBootstrapPrimaryError::StereoLeft(source)),
            }
        }
        if right.is_none() {
            match oak.mono_right(STEREO_POLL_TIMEOUT_MS) {
                Ok(frame) => {
                    right = Some(frame);
                    received = true;
                }
                Err(ImageError::Timeout { .. } | ImageError::QueueEmpty) => {}
                Err(source) => return Err(NanoBootstrapPrimaryError::StereoRight(source)),
            }
        }
        if !received {
            thread::sleep(STEREO_IDLE_SLEEP);
        }
    }
    require_running(running)?;

    let left = left.expect("loop exits only with a left frame");
    let right = right.expect("loop exits only with a right frame");
    validate_stereo_frame(
        NanoBootstrapStereoSide::Left,
        &left,
        launch.oak().rectified_stereo(),
    )?;
    validate_stereo_frame(
        NanoBootstrapStereoSide::Right,
        &right,
        launch.oak().rectified_stereo(),
    )?;
    let baseline_m = oak
        .stereo_baseline_m()
        .map_err(NanoBootstrapPrimaryError::StereoCalibration)?;
    let left_intrinsics = left.intrinsics();
    let right_intrinsics = right.intrinsics();
    let calibration = Calibration {
        left: CameraIntrinsics {
            fx: left_intrinsics.fx(),
            fy: left_intrinsics.fy(),
            cx: left_intrinsics.cx(),
            cy: left_intrinsics.cy(),
            width: left_intrinsics.width(),
            height: left_intrinsics.height(),
        },
        right: CameraIntrinsics {
            fx: right_intrinsics.fx(),
            fy: right_intrinsics.fy(),
            cx: right_intrinsics.cx(),
            cy: right_intrinsics.cy(),
            width: right_intrinsics.width(),
            height: right_intrinsics.height(),
        },
        baseline_m,
        rectified: true,
    };
    let rectified = RectifiedStereo::from_calibration(&calibration)
        .map_err(NanoBootstrapPrimaryError::Stereo)?;
    let runtime_depth_camera = DepthCameraModel::new(
        rectified.left(),
        rectified.dimensions(),
        DepthToTrackingCamera::identity(),
    );
    Ok(NanoBootstrapStereoEvidence {
        left,
        right,
        calibration,
        runtime_depth_camera,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoBootstrapStereoSide {
    Left,
    Right,
}

impl NanoBootstrapStereoSide {
    const fn expected_stream(self) -> OakStreamId {
        match self {
            Self::Left => OakStreamId::MonoLeft,
            Self::Right => OakStreamId::MonoRight,
        }
    }
}

fn validate_stereo_frame(
    side: NanoBootstrapStereoSide,
    frame: &OakImageFrame,
    expected: super::NanoOakImageStream,
) -> Result<(), NanoBootstrapPrimaryError> {
    let expected_stream = side.expected_stream();
    if frame.stream != expected_stream {
        return Err(NanoBootstrapPrimaryError::StereoUnexpectedStream {
            side,
            expected: expected_stream,
            actual: frame.stream,
        });
    }
    let expected_dimensions = [expected.width_px(), expected.height_px()];
    let actual_dimensions = [frame.width, frame.height];
    if actual_dimensions != expected_dimensions {
        return Err(NanoBootstrapPrimaryError::StereoUnexpectedDimensions {
            side,
            expected: expected_dimensions,
            actual: actual_dimensions,
        });
    }
    let intrinsics = frame.intrinsics();
    if [intrinsics.width(), intrinsics.height()] != expected_dimensions {
        return Err(
            NanoBootstrapPrimaryError::StereoIntrinsicsDimensionsMismatch {
                side,
                frame: actual_dimensions,
                intrinsics: [intrinsics.width(), intrinsics.height()],
            },
        );
    }
    FrameDimensions::try_new(frame.width, frame.height)
        .map_err(NanoBootstrapPrimaryError::StereoFrameDimensions)?;
    Ok(())
}

fn derive_required_probe_configs(
    policy: &ManifestBoundNanoAgentPolicyConfigV1,
) -> Result<(HeadProbeConfig, IdentityProbeConfig), NanoBootstrapPrimaryError> {
    let head = policy
        .head()
        .natural_hold()
        .ok_or(NanoBootstrapPrimaryError::NaturalHeadHoldRequired)?;
    let eye = policy
        .eye()
        .static_runtime()
        .ok_or(NanoBootstrapPrimaryError::Kep2EyeRequired)?;
    Ok((
        HeadProbeConfig::from_runtime(head.runtime()),
        IdentityProbeConfig::from_static_runtime(eye),
    ))
}

fn require_policy_paths_within_deployment(
    roots: &NanoBootstrapRoots,
    policy: &NanoAgentPolicyConfigV1,
) -> Result<(), NanoBootstrapPrimaryError> {
    require_path_within_deployment(
        NanoBootstrapDeploymentPathKind::Manifest,
        roots.deployment_root(),
        policy.inventory().manifest_path().as_path(),
        false,
    )?;
    require_path_within_deployment(
        NanoBootstrapDeploymentPathKind::ArtifactRoot,
        roots.deployment_root(),
        policy.inventory().artifact_root_path().as_path(),
        true,
    )?;
    Ok(())
}

fn require_path_within_deployment(
    kind: NanoBootstrapDeploymentPathKind,
    deployment_root: &Path,
    path: &Path,
    may_equal_root: bool,
) -> Result<(), NanoBootstrapPrimaryError> {
    let Ok(relative) = path.strip_prefix(deployment_root) else {
        return Err(NanoBootstrapPrimaryError::PolicyPathOutsideDeployment {
            kind,
            deployment_root: deployment_root.to_path_buf(),
            configured: path.to_path_buf(),
        });
    };
    if relative.as_os_str().is_empty() && !may_equal_root {
        return Err(NanoBootstrapPrimaryError::PolicyPathAliasesDeploymentRoot {
            kind,
            deployment_root: deployment_root.to_path_buf(),
        });
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoBootstrapDeploymentPathKind {
    Manifest,
    ArtifactRoot,
}

struct SelectedPlantArtifact {
    id: ArtifactId,
    relative_path: ArtifactRelativePath,
}

fn select_plant_artifact(
    roots: &NanoBootstrapRoots,
    launch: &NanoAgentLaunchV1,
    launch_plant: &LoadedDeploymentAsset,
    policy: &NanoAgentPolicyConfigV1,
    manifest: &kiko_device_inventory::DeviceInventoryManifestV1,
    hashes: &ManifestArtifactHashes,
) -> Result<SelectedPlantArtifact, NanoBootstrapPrimaryError> {
    let requested = launch.plant_artifact().artifact_id().as_str();
    let expected = manifest
        .artifacts()
        .iter()
        .find(|artifact| {
            artifact.kind() == ArtifactKind::Plant && artifact.artifact_id().as_str() == requested
        })
        .ok_or_else(|| NanoBootstrapPrimaryError::PlantArtifactNotInManifest {
            artifact_id: requested.to_owned(),
        })?;
    if expected.sha256().as_bytes() != launch_plant.content_sha256().as_bytes() {
        return Err(
            NanoBootstrapPrimaryError::PlantLaunchManifestDigestMismatch {
                artifact_id: requested.to_owned(),
                launch_sha256: *launch_plant.content_sha256().as_bytes(),
                manifest_sha256: *expected.sha256().as_bytes(),
            },
        );
    }
    let binding = policy
        .inventory()
        .artifact_bindings()
        .iter()
        .find(|binding| {
            binding.kind() == ArtifactKind::Plant && binding.artifact_id().as_str() == requested
        })
        .ok_or_else(|| NanoBootstrapPrimaryError::PlantArtifactBindingMissing {
            artifact_id: requested.to_owned(),
        })?;
    let hashed = hashes
        .iter()
        .find(|artifact| {
            artifact.kind() == ArtifactKind::Plant && artifact.artifact_id().as_str() == requested
        })
        .ok_or_else(|| NanoBootstrapPrimaryError::PlantArtifactHashMissing {
            artifact_id: requested.to_owned(),
        })?;
    if hashed.observed_sha256() != launch_plant.content_sha256().as_bytes() {
        return Err(NanoBootstrapPrimaryError::PlantLaunchHashMismatch {
            artifact_id: requested.to_owned(),
            launch_sha256: *launch_plant.content_sha256().as_bytes(),
            observed_sha256: *hashed.observed_sha256(),
        });
    }
    let deployed_relative = deployment_relative_artifact_path(
        roots.deployment_root(),
        policy.inventory().artifact_root_path().as_path(),
        binding.relative_path(),
    )?;
    if &deployed_relative != launch.plant_artifact().asset().relative_path() {
        return Err(NanoBootstrapPrimaryError::PlantLaunchPathMismatch {
            artifact_id: requested.to_owned(),
            launch: launch
                .plant_artifact()
                .asset()
                .relative_path()
                .as_str()
                .into(),
            inventory: deployed_relative.as_str().into(),
        });
    }
    Ok(SelectedPlantArtifact {
        id: *expected.artifact_id(),
        relative_path: binding.relative_path().clone(),
    })
}

fn deployment_relative_artifact_path(
    deployment_root: &Path,
    artifact_root: &Path,
    artifact_relative_path: &ArtifactRelativePath,
) -> Result<ArtifactRelativePath, NanoBootstrapPrimaryError> {
    let root_relative = artifact_root.strip_prefix(deployment_root).map_err(|_| {
        NanoBootstrapPrimaryError::PolicyPathOutsideDeployment {
            kind: NanoBootstrapDeploymentPathKind::ArtifactRoot,
            deployment_root: deployment_root.to_path_buf(),
            configured: artifact_root.to_path_buf(),
        }
    })?;
    let combined = root_relative.join(artifact_relative_path.as_path());
    let combined =
        combined
            .to_str()
            .ok_or_else(|| NanoBootstrapPrimaryError::PlantDeploymentPathNotUtf8 {
                path: combined.clone(),
            })?;
    ArtifactRelativePath::parse(combined.to_owned())
        .map_err(NanoBootstrapPrimaryError::PlantDeploymentRelativePath)
}

fn bind_controller_contract_to_manifest(
    launch: &NanoAgentLaunchV1,
    server: &ControllerServerConfigV1,
    manifest: &kiko_device_inventory::DeviceInventoryManifestV1,
) -> Result<(), NanoBootstrapPrimaryError> {
    let expected = manifest.stm32();
    if server.serial_device() != Path::new(expected.serial_path().as_str()) {
        return Err(NanoBootstrapPrimaryError::ControllerSerialMismatch {
            server: server.serial_device().to_path_buf(),
            manifest: expected.serial_path().as_str().into(),
        });
    }
    if server.controller_uid() != *expected.controller_uid() {
        return Err(NanoBootstrapPrimaryError::ControllerUidMismatch);
    }
    if server.firmware_abi().get() != expected.firmware_abi() {
        return Err(NanoBootstrapPrimaryError::ControllerFirmwareAbiMismatch {
            server: server.firmware_abi().get(),
            expected: expected.firmware_abi(),
        });
    }
    if server.firmware_build_id().get() != expected.firmware_build_id() {
        return Err(NanoBootstrapPrimaryError::ControllerFirmwareBuildMismatch {
            server: server.firmware_build_id().get(),
            expected: expected.firmware_build_id(),
        });
    }
    if server.actuator_config_fingerprint() != *expected.hardware_profile() {
        return Err(NanoBootstrapPrimaryError::ControllerFingerprintMismatch);
    }
    let launch_endpoint = format!(
        "udp://{}",
        launch.controller_server().command_udp_endpoint()
    );
    if expected.control_endpoint().as_str() != launch_endpoint {
        return Err(NanoBootstrapPrimaryError::ControllerEndpointMismatch {
            launch: launch_endpoint,
            manifest: expected.control_endpoint().as_str().to_owned(),
        });
    }
    Ok(())
}

fn bind_controller_contract_to_actuation(
    launch: &NanoAgentLaunchV1,
    server: &ControllerServerConfigV1,
    actuation: &NavigationActuationConfigV1,
) -> Result<(), NanoBootstrapPrimaryError> {
    let launch_endpoint = launch.controller_server().command_udp_endpoint();
    let actuation_endpoint = actuation.command_endpoint().socket_addr();
    if launch_endpoint != actuation_endpoint {
        return Err(
            NanoBootstrapPrimaryError::ActuationCommandEndpointMismatch {
                launch: launch_endpoint,
                actuation: actuation_endpoint,
            },
        );
    }
    if server.controller_uid() != actuation.controller_uid() {
        return Err(NanoBootstrapPrimaryError::ActuationControllerUidMismatch);
    }
    if server.firmware_abi() != actuation.firmware_abi() {
        return Err(
            NanoBootstrapPrimaryError::ActuationControllerFirmwareAbiMismatch {
                server: server.firmware_abi().get(),
                actuation: actuation.firmware_abi().get(),
            },
        );
    }
    if server.firmware_build_id() != actuation.firmware_build_id() {
        return Err(
            NanoBootstrapPrimaryError::ActuationControllerFirmwareBuildMismatch {
                server: server.firmware_build_id().get(),
                actuation: actuation.firmware_build_id().get(),
            },
        );
    }
    if server.actuator_config_fingerprint() != actuation.actuator_config_fingerprint() {
        return Err(NanoBootstrapPrimaryError::ActuationControllerFingerprintMismatch);
    }
    Ok(())
}

fn require_running(running: &AtomicBool) -> Result<(), NanoBootstrapPrimaryError> {
    if running.load(Ordering::Acquire) {
        Ok(())
    } else {
        Err(NanoBootstrapPrimaryError::Interrupted)
    }
}

fn timestamp_since(origin: Instant) -> Result<HostMonotonicTimestamp, NanoBootstrapPrimaryError> {
    let elapsed = origin.elapsed().as_nanos();
    let nanos = u64::try_from(elapsed)
        .map_err(|_| NanoBootstrapPrimaryError::MonotonicTimestampOverflow { elapsed })?;
    Ok(HostMonotonicTimestamp::from_nanos(nanos))
}

trait ExplicitBootstrapDisarm {
    type Receipt;
    type Error;

    fn explicit_disarm(self) -> Result<Self::Receipt, Self::Error>;
}

impl ExplicitBootstrapDisarm for PendingLiveMpcControlDriver {
    type Receipt = DisarmReceipt;
    type Error = LiveActuationError;

    fn explicit_disarm(self) -> Result<Self::Receipt, Self::Error> {
        self.disarm()
    }
}

trait ExplicitBootstrapClose {
    type Error;

    fn explicit_close(self) -> Result<(), Self::Error>;
}

impl ExplicitBootstrapClose for Device {
    type Error = OakCloseError;

    fn explicit_close(self) -> Result<(), Self::Error> {
        self.close()
    }
}

type ControllerCleanupResult<Controller> = Result<
    <Controller as ExplicitBootstrapDisarm>::Receipt,
    <Controller as ExplicitBootstrapDisarm>::Error,
>;
type OakCleanupResult<Oak> = Result<(), <Oak as ExplicitBootstrapClose>::Error>;
type BootstrapCleanupResult<Controller, Oak> =
    (ControllerCleanupResult<Controller>, OakCleanupResult<Oak>);

fn collect_failure_cleanup<Controller, Oak>(
    controller: Controller,
    oak: Oak,
) -> BootstrapCleanupResult<Controller, Oak>
where
    Controller: ExplicitBootstrapDisarm,
    Oak: ExplicitBootstrapClose,
{
    let stop = controller.explicit_disarm();
    let close = oak.explicit_close();
    (stop, close)
}

fn cleanup_after_pending_failure(
    primary: NanoBootstrapPrimaryError,
    pending: PendingLiveMpcControlDriver,
    oak: Device,
) -> NanoBootstrapError {
    let (stop, close) = collect_failure_cleanup(pending, oak);
    NanoBootstrapError {
        primary: Box::new(primary),
        controller: match stop {
            Ok(receipt) => NanoBootstrapControllerDisposition::ConfirmedStopped(receipt),
            Err(source) => NanoBootstrapControllerDisposition::StopUncertain(source),
        },
        oak_close: match close {
            Ok(()) => NanoBootstrapOakCloseDisposition::ConfirmedClosed,
            Err(source) => NanoBootstrapOakCloseDisposition::CloseUncertain(source),
        },
    }
}

fn close_oak_after_failure(primary: NanoBootstrapPrimaryError, oak: Device) -> NanoBootstrapError {
    NanoBootstrapError {
        primary: Box::new(primary),
        controller: NanoBootstrapControllerDisposition::NotAcquired,
        oak_close: close_oak(oak),
    }
}

fn close_oak(oak: Device) -> NanoBootstrapOakCloseDisposition {
    match oak.close() {
        Ok(()) => NanoBootstrapOakCloseDisposition::ConfirmedClosed,
        Err(source) => NanoBootstrapOakCloseDisposition::CloseUncertain(source),
    }
}

/// Why production bootstrap failed, plus explicit resource disposition.
pub struct NanoBootstrapError {
    primary: Box<NanoBootstrapPrimaryError>,
    controller: NanoBootstrapControllerDisposition,
    oak_close: NanoBootstrapOakCloseDisposition,
}

impl NanoBootstrapError {
    fn before_hardware(primary: NanoBootstrapPrimaryError) -> Self {
        Self {
            primary: Box::new(primary),
            controller: NanoBootstrapControllerDisposition::NotAcquired,
            oak_close: NanoBootstrapOakCloseDisposition::NotOpened,
        }
    }

    pub const fn primary(&self) -> &NanoBootstrapPrimaryError {
        &self.primary
    }

    pub const fn controller(&self) -> &NanoBootstrapControllerDisposition {
        &self.controller
    }

    pub const fn oak_close(&self) -> &NanoBootstrapOakCloseDisposition {
        &self.oak_close
    }
}

impl fmt::Debug for NanoBootstrapError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, formatter)
    }
}

impl fmt::Display for NanoBootstrapError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "Nano bootstrap failed: {}; {}; {}",
            self.primary, self.controller, self.oak_close
        )
    }
}

impl std::error::Error for NanoBootstrapError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.primary.as_ref())
    }
}

pub enum NanoBootstrapControllerDisposition {
    NotAcquired,
    ConfirmedStopped(DisarmReceipt),
    StopUncertain(LiveActuationError),
    /// [`NanoProductionAdmissionError`] in `primary` owns the corresponding
    /// confirmed-versus-uncertain stop evidence.
    AdmissionErrorRetainsStop,
}

impl fmt::Debug for NanoBootstrapControllerDisposition {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, formatter)
    }
}

impl fmt::Display for NanoBootstrapControllerDisposition {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotAcquired => formatter.write_str("controller was not acquired"),
            Self::ConfirmedStopped(receipt) => write!(
                formatter,
                "controller stop confirmed at {} ns",
                receipt.acknowledged_at().nanos_since_clock_start()
            ),
            Self::StopUncertain(source) => write!(formatter, "controller stop uncertain: {source}"),
            Self::AdmissionErrorRetainsStop => {
                formatter.write_str("production-admission error retains controller stop evidence")
            }
        }
    }
}

pub enum NanoBootstrapOakCloseDisposition {
    NotOpened,
    ConfirmedClosed,
    CloseUncertain(OakCloseError),
}

impl fmt::Debug for NanoBootstrapOakCloseDisposition {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, formatter)
    }
}

impl fmt::Display for NanoBootstrapOakCloseDisposition {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotOpened => formatter.write_str("OAK was not opened"),
            Self::ConfirmedClosed => formatter.write_str("OAK close confirmed"),
            Self::CloseUncertain(source) => write!(formatter, "OAK close uncertain: {source}"),
        }
    }
}

#[derive(Debug)]
pub enum NanoBootstrapPrimaryError {
    Interrupted,
    MonotonicTimestampOverflow {
        elapsed: u128,
    },
    LaunchLoad(NanoAgentLaunchLoadError),
    BoundAssetLoad {
        role: NanoLaunchAssetRole,
        source: NanoLaunchBoundAssetLoadError,
    },
    AgentPolicy(NanoAgentPolicyConfigParseError),
    PolicyPathOutsideDeployment {
        kind: NanoBootstrapDeploymentPathKind,
        deployment_root: PathBuf,
        configured: PathBuf,
    },
    PolicyPathAliasesDeploymentRoot {
        kind: NanoBootstrapDeploymentPathKind,
        deployment_root: PathBuf,
    },
    Manifest(ManifestLoadError),
    AccessoryManifestBinding(NanoAccessoryManifestBindingError),
    NaturalHeadHoldRequired,
    Kep2EyeRequired,
    ArtifactHash(ArtifactHashError),
    PlantArtifactNotInManifest {
        artifact_id: String,
    },
    PlantArtifactBindingMissing {
        artifact_id: String,
    },
    PlantArtifactHashMissing {
        artifact_id: String,
    },
    PlantLaunchManifestDigestMismatch {
        artifact_id: String,
        launch_sha256: [u8; 32],
        manifest_sha256: [u8; 32],
    },
    PlantLaunchHashMismatch {
        artifact_id: String,
        launch_sha256: [u8; 32],
        observed_sha256: [u8; 32],
    },
    PlantLaunchPathMismatch {
        artifact_id: String,
        launch: Box<str>,
        inventory: Box<str>,
    },
    PlantDeploymentPathNotUtf8 {
        path: PathBuf,
    },
    PlantDeploymentRelativePath(ArtifactRelativePathError),
    ControllerServerContract(ServerConfigError),
    ControllerSerialMismatch {
        server: PathBuf,
        manifest: PathBuf,
    },
    ControllerUidMismatch,
    ControllerFirmwareAbiMismatch {
        server: u16,
        expected: u16,
    },
    ControllerFirmwareBuildMismatch {
        server: u32,
        expected: u32,
    },
    ControllerFingerprintMismatch,
    ControllerEndpointMismatch {
        launch: String,
        manifest: String,
    },
    HeadProbe(Box<SerialHeadProbeError>),
    EyeProbe(IdentityProbeError),
    OakConnect(oak_sys::ConnectionError),
    OakConnectedIdentity(ConnectedDeviceIdentityError),
    EmptyOpenedOakMxid,
    OakUsbTransport(UsbTransportEvidenceError),
    DepthAiBuild(DepthAiBuildMetadataError),
    StereoLeft(ImageError),
    StereoRight(ImageError),
    StereoTimedOut {
        maximum_wait: Duration,
        received_left: bool,
        received_right: bool,
    },
    StereoUnexpectedStream {
        side: NanoBootstrapStereoSide,
        expected: OakStreamId,
        actual: OakStreamId,
    },
    StereoUnexpectedDimensions {
        side: NanoBootstrapStereoSide,
        expected: [u32; 2],
        actual: [u32; 2],
    },
    StereoIntrinsicsDimensionsMismatch {
        side: NanoBootstrapStereoSide,
        frame: [u32; 2],
        intrinsics: [u32; 2],
    },
    StereoFrameDimensions(crate::FrameDimensionsError),
    StereoCalibration(OakCalibrationError),
    Stereo(RectifiedStereoError),
    ShadowNavigation(ShadowNavigationConfigParseError),
    Actuation(super::ActuationConfigParseError),
    ActuationCommandEndpointMismatch {
        launch: std::net::SocketAddr,
        actuation: std::net::SocketAddr,
    },
    ActuationControllerUidMismatch,
    ActuationControllerFirmwareAbiMismatch {
        server: u16,
        actuation: u16,
    },
    ActuationControllerFirmwareBuildMismatch {
        server: u32,
        actuation: u32,
    },
    ActuationControllerFingerprintMismatch,
    ControllerAcquire(LiveActuationError),
    ControllerEvidence(LiveActuationError),
    ObservedInventoryEvidence(NanoObservedInventoryEvidenceError),
    ObservedInventoryBuild(NanoObservedInventoryBuildError),
    AdmissionTimeline(NanoProductionAdmissionTimelineError),
    ProductionAdmission(NanoProductionAdmissionError),
}

impl fmt::Display for NanoBootstrapPrimaryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "production bootstrap boundary rejected input: {self:?}"
        )
    }
}

impl std::error::Error for NanoBootstrapPrimaryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::LaunchLoad(source) => Some(source),
            Self::BoundAssetLoad { source, .. } => Some(source),
            Self::AgentPolicy(source) => Some(source),
            Self::Manifest(source) => Some(source),
            Self::AccessoryManifestBinding(source) => Some(source),
            Self::ArtifactHash(source) => Some(source),
            Self::PlantDeploymentRelativePath(source) => Some(source),
            Self::ControllerServerContract(source) => Some(source),
            Self::HeadProbe(source) => Some(source.as_ref()),
            Self::EyeProbe(source) => Some(source),
            Self::OakConnect(source) => Some(source),
            Self::OakConnectedIdentity(source) => Some(source),
            Self::OakUsbTransport(source) => Some(source),
            Self::DepthAiBuild(source) => Some(source),
            Self::StereoLeft(source) | Self::StereoRight(source) => Some(source),
            Self::StereoFrameDimensions(source) => Some(source),
            Self::StereoCalibration(source) => Some(source),
            Self::Stereo(source) => Some(source),
            Self::ShadowNavigation(source) => Some(source),
            Self::Actuation(source) => Some(source),
            Self::ControllerAcquire(source) | Self::ControllerEvidence(source) => Some(source),
            Self::ObservedInventoryEvidence(source) => Some(source),
            Self::ObservedInventoryBuild(source) => Some(source),
            Self::AdmissionTimeline(source) => Some(source),
            Self::ProductionAdmission(source) => Some(source),
            Self::Interrupted
            | Self::MonotonicTimestampOverflow { .. }
            | Self::PolicyPathOutsideDeployment { .. }
            | Self::PolicyPathAliasesDeploymentRoot { .. }
            | Self::NaturalHeadHoldRequired
            | Self::Kep2EyeRequired
            | Self::PlantArtifactNotInManifest { .. }
            | Self::PlantArtifactBindingMissing { .. }
            | Self::PlantArtifactHashMissing { .. }
            | Self::PlantLaunchManifestDigestMismatch { .. }
            | Self::PlantLaunchHashMismatch { .. }
            | Self::PlantLaunchPathMismatch { .. }
            | Self::PlantDeploymentPathNotUtf8 { .. }
            | Self::ControllerSerialMismatch { .. }
            | Self::ControllerUidMismatch
            | Self::ControllerFirmwareAbiMismatch { .. }
            | Self::ControllerFirmwareBuildMismatch { .. }
            | Self::ControllerFingerprintMismatch
            | Self::ControllerEndpointMismatch { .. }
            | Self::EmptyOpenedOakMxid
            | Self::StereoTimedOut { .. }
            | Self::StereoUnexpectedStream { .. }
            | Self::StereoUnexpectedDimensions { .. }
            | Self::StereoIntrinsicsDimensionsMismatch { .. }
            | Self::ActuationCommandEndpointMismatch { .. }
            | Self::ActuationControllerUidMismatch
            | Self::ActuationControllerFirmwareAbiMismatch { .. }
            | Self::ActuationControllerFirmwareBuildMismatch { .. }
            | Self::ActuationControllerFingerprintMismatch => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use super::*;

    #[test]
    fn roots_are_absolute_canonical_distinct_and_never_filesystem_root() {
        let valid =
            NanoBootstrapRoots::try_new("/opt/kiko/deployment".into(), "/var/lib/kiko".into())
                .expect("canonical roots");
        assert_eq!(valid.deployment_root(), Path::new("/opt/kiko/deployment"));
        assert_eq!(valid.state_root(), Path::new("/var/lib/kiko"));

        for invalid in [
            PathBuf::from("relative"),
            PathBuf::from("/"),
            PathBuf::from("/opt//kiko"),
            PathBuf::from("/opt/./kiko"),
            PathBuf::from("/opt/../kiko"),
            PathBuf::from("/opt/kiko/"),
        ] {
            assert!(NanoBootstrapRoots::try_new(invalid, "/var/lib/kiko".into()).is_err());
        }
        for state in ["/opt/kiko", "/opt/kiko/state"] {
            assert!(matches!(
                NanoBootstrapRoots::try_new("/opt/kiko".into(), state.into()),
                Err(NanoBootstrapRootError::OverlappingRoots { .. })
            ));
        }
        assert!(matches!(
            NanoBootstrapRoots::try_new("/opt/kiko/deployment".into(), "/opt/kiko".into()),
            Err(NanoBootstrapRootError::OverlappingRoots { .. })
        ));
    }

    #[derive(Clone)]
    struct FakeController {
        events: Rc<RefCell<Vec<&'static str>>>,
        result: Result<u64, &'static str>,
    }

    impl ExplicitBootstrapDisarm for FakeController {
        type Receipt = u64;
        type Error = &'static str;

        fn explicit_disarm(self) -> Result<Self::Receipt, Self::Error> {
            self.events.borrow_mut().push("stop");
            self.result
        }
    }

    struct FakeOak {
        events: Rc<RefCell<Vec<&'static str>>>,
        result: Result<(), &'static str>,
    }

    impl ExplicitBootstrapClose for FakeOak {
        type Error = &'static str;

        fn explicit_close(self) -> Result<(), Self::Error> {
            self.events.borrow_mut().push("close");
            self.result
        }
    }

    #[test]
    fn post_acquisition_cleanup_always_attempts_stop_then_close_and_retains_both() {
        for (stop, close) in [
            (Ok(7), Ok(())),
            (Err("stop uncertain"), Ok(())),
            (Ok(8), Err("close uncertain")),
            (Err("stop uncertain"), Err("close uncertain")),
        ] {
            let events = Rc::new(RefCell::new(Vec::new()));
            let actual = collect_failure_cleanup(
                FakeController {
                    events: Rc::clone(&events),
                    result: stop,
                },
                FakeOak {
                    events: Rc::clone(&events),
                    result: close,
                },
            );
            assert_eq!(&*events.borrow(), &["stop", "close"]);
            assert_eq!(actual.0, stop);
            assert_eq!(actual.1, close);
        }
    }

    #[test]
    fn artifact_binding_is_reexpressed_under_the_single_deployment_root() {
        let relative = ArtifactRelativePath::parse("plant/drive.json".into()).expect("relative");
        assert_eq!(
            deployment_relative_artifact_path(
                Path::new("/opt/kiko/deployment"),
                Path::new("/opt/kiko/deployment/artifacts"),
                &relative,
            )
            .expect("bound deployment path")
            .as_str(),
            "artifacts/plant/drive.json"
        );
        assert!(matches!(
            deployment_relative_artifact_path(
                Path::new("/opt/kiko/deployment"),
                Path::new("/srv/unrelated"),
                &relative,
            ),
            Err(NanoBootstrapPrimaryError::PolicyPathOutsideDeployment {
                kind: NanoBootstrapDeploymentPathKind::ArtifactRoot,
                ..
            })
        ));
    }

    #[test]
    fn stopped_request_is_rejected_without_touching_hardware() {
        let running = AtomicBool::new(false);
        assert!(matches!(
            require_running(&running),
            Err(NanoBootstrapPrimaryError::Interrupted)
        ));
    }
}
