//! Qualification-only Nano bootstrap up to one exactly inventoried, stopped
//! candidate controller.
//!
//! This is deliberately a library boundary rather than a second live-loop
//! binary. It loads every launch-bound file once, binds the schema-V4
//! candidate contracts, probes the connected robot, acquires only an
//! acknowledged zero, performs exact inner-V1 inventory comparison, and
//! returns a linear stopped controller token for a later qualification owner.

use std::fmt;
use std::fs::OpenOptions;
use std::io;
use std::io::Read;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use kiko_device_inventory::{
    ArtifactHashError, ArtifactId, ArtifactKind, ArtifactRelativePath, ArtifactRelativePathError,
    ExactInventoryAdmission, InventoryMismatchReport, LoadedDeploymentAsset,
    LoadedExpectedManifestV2, ManifestArtifactHashes, ManifestLoadError,
    StreamedDeploymentAssetIdentity, UnixFileIdentity, admit_exact_inventory,
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
use sha2::{Digest, Sha256};

use super::actuation::LiveActuationError;
use super::nano_bootstrap::{
    NanoExactDevicePresenceTargets, bootstrap_stereo, derive_required_probe_configs,
    require_running, wait_for_exact_device_presence,
};
use super::{
    AdmittedOakSuperSpeedEvidence, CandidateActuationSessionStartError, CandidateMpcBindingError,
    CandidateRuntimeServiceIntervalError, HeadGazePolicyLifecycleClaim, HeadGazePolicyParseError,
    HeadGazePolicyV1, LoadedNanoWheelsOffQualificationLaunchV4,
    ManifestBoundNanoAgentPolicyConfigV3, NanoAccessoryManifestBindingError,
    NanoAgentPolicyConfigParseError, NanoAgentPolicyConfigV3, NanoBootstrapAccessoryEvidence,
    NanoBootstrapOakCloseDisposition, NanoBootstrapPrimaryError, NanoBootstrapRootError,
    NanoBootstrapRoots, NanoBootstrapStereoEvidence, NanoCalibrationArtifactParseError,
    NanoCalibrationArtifactV1, NanoCalibrationBindingError, NanoFaceCascadeAssetRole,
    NanoLaunchBoundAssetLoadError, NanoObservedInventoryBuildError, NanoObservedInventoryBuilder,
    NanoObservedInventoryEvidenceError, NanoWheelsOffMappedImageError,
    NanoWheelsOffNativeRuntimeBindingError, NanoWheelsOffNativeRuntimeParseError,
    NanoWheelsOffNativeRuntimeV1, NanoWheelsOffNativeRuntimeVerificationError,
    NanoWheelsOffQualificationAssetRole, NanoWheelsOffQualificationLaunchLoadError,
    NanoWheelsOffQualificationLaunchV4, NanoWheelsOffQualificationV4AssetRole,
    ParsedNanoLiveConfiguration, ShadowNavigationConfigParseError, ShadowNavigationConfigV1,
    StoppedWheelsOffCandidateController, VerifiedNanoWheelsOffMappedImages,
    VerifiedNanoWheelsOffNativeRuntimeDependencies, WheelsOffCandidateActuationSession,
    WheelsOffCandidateControllerBinding, WheelsOffCandidateControllerBindingError,
    WheelsOffCandidateLimits, WheelsOffCandidatePolicyError,
    WheelsOffCandidateRuntimeServiceInterval, WheelsOffQualificationFaultInjection,
    load_nano_wheels_off_qualification_launch_v4, verify_linux_mapped_qualification_images,
};
use crate::dense::occupancy::{DepthCameraModel, DepthToTrackingCamera};
use crate::live_runtime::LiveOccupancyHostPolicy;

/// One process-lifetime request for the qualification-only static/hardware
/// boundary.
pub struct QualificationBootstrapRequest<'running> {
    roots: NanoBootstrapRoots,
    launch_relative_path: ArtifactRelativePath,
    controller_clock_origin: Instant,
    fault_injection: Option<WheelsOffQualificationFaultInjection>,
    running: &'running AtomicBool,
}

impl<'running> QualificationBootstrapRequest<'running> {
    pub fn try_new(
        deployment_root: PathBuf,
        state_root: PathBuf,
        launch_relative_path: String,
        controller_clock_origin: Instant,
        fault_injection: Option<WheelsOffQualificationFaultInjection>,
        running: &'running AtomicBool,
    ) -> Result<Self, QualificationBootstrapRequestError> {
        Ok(Self {
            roots: NanoBootstrapRoots::try_new(deployment_root, state_root)
                .map_err(QualificationBootstrapRequestError::Roots)?,
            launch_relative_path: ArtifactRelativePath::parse(launch_relative_path)
                .map_err(QualificationBootstrapRequestError::LaunchRelativePath)?,
            controller_clock_origin,
            fault_injection,
            running,
        })
    }

    pub const fn roots(&self) -> &NanoBootstrapRoots {
        &self.roots
    }

    pub const fn launch_relative_path(&self) -> &ArtifactRelativePath {
        &self.launch_relative_path
    }

    pub const fn fault_injection(&self) -> Option<WheelsOffQualificationFaultInjection> {
        self.fault_injection
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

/// Retained bytes for consumed inputs plus streaming identities for the
/// executable and required native-runtime libraries consumed by the OS loader.
#[derive(Debug)]
pub struct LoadedNanoWheelsOffQualificationAssets {
    pub qualification_executable: StreamedDeploymentAssetIdentity,
    pub native_runtime_manifest: LoadedDeploymentAsset,
    pub native_runtime: NanoWheelsOffNativeRuntimeV1,
    pub native_runtime_dependency_identities: VerifiedNanoWheelsOffNativeRuntimeDependencies,
    pub agent_policy: LoadedDeploymentAsset,
    pub head_gaze_policy: Option<LoadedDeploymentAsset>,
    pub frontal_face_cascade: LoadedDeploymentAsset,
    pub profile_face_cascade: LoadedDeploymentAsset,
    pub navigation_shadow_config: LoadedDeploymentAsset,
    pub candidate_inventory_manifest: LoadedDeploymentAsset,
    pub candidate_controller_policy: LoadedDeploymentAsset,
    pub controller_server_contract: LoadedDeploymentAsset,
    pub calibration_artifact: LoadedDeploymentAsset,
    pub plant_artifact: LoadedDeploymentAsset,
    pub onnx_runtime_library: StreamedDeploymentAssetIdentity,
    pub pinned_onnx_runtime: crate::PinnedOrtRuntime,
    pub mapped_images: VerifiedNanoWheelsOffMappedImages,
    pub superpoint_model: LoadedDeploymentAsset,
    pub lightglue_model: LoadedDeploymentAsset,
}

impl LoadedNanoWheelsOffQualificationAssets {
    fn load(
        deployment_root: &Path,
        launch: &NanoWheelsOffQualificationLaunchV4,
    ) -> Result<Self, QualificationBootstrapPrimaryError> {
        let load = |role| {
            launch
                .asset(role)
                .load_exact(deployment_root)
                .map_err(
                    |source| QualificationBootstrapPrimaryError::BoundAssetLoad { role, source },
                )
        };
        let load_v4 = |role| {
            launch
                .v4_asset(role)
                .map(|asset| {
                    asset.load_exact(deployment_root).map_err(|source| {
                        QualificationBootstrapPrimaryError::V4BoundAssetLoad { role, source }
                    })
                })
                .transpose()
        };
        let load_face = |role| {
            launch
                .face_perception()
                .asset(role)
                .load_exact(deployment_root)
                .map_err(
                    |source| QualificationBootstrapPrimaryError::FaceBoundAssetLoad {
                        role,
                        source,
                    },
                )
        };
        let qualification_executable = launch
            .qualification_executable()
            .verify_exact_streaming(deployment_root)
            .map_err(
                |source| QualificationBootstrapPrimaryError::BoundAssetLoad {
                    role: NanoWheelsOffQualificationAssetRole::QualificationExecutable,
                    source,
                },
            )?;
        require_running_qualification_executable(&qualification_executable)
            .map_err(QualificationBootstrapPrimaryError::QualificationExecutable)?;
        let native_runtime_manifest =
            load(NanoWheelsOffQualificationAssetRole::NativeRuntimeManifest)?;
        let native_runtime =
            NanoWheelsOffNativeRuntimeV1::parse_json(native_runtime_manifest.bytes())
                .map_err(QualificationBootstrapPrimaryError::NativeRuntimeManifest)?;
        let onnx_runtime_library = launch
            .inference()
            .onnx_runtime_library()
            .verify_exact_streaming(deployment_root)
            .map_err(
                |source| QualificationBootstrapPrimaryError::BoundAssetLoad {
                    role: NanoWheelsOffQualificationAssetRole::OnnxRuntimeLibrary,
                    source,
                },
            )?;
        native_runtime
            .bind_onnx_runtime_launch(launch.inference().onnx_runtime_library())
            .map_err(QualificationBootstrapPrimaryError::NativeRuntimeBinding)?;
        native_runtime
            .reject_non_onnx_launch_aliases(
                NanoWheelsOffQualificationAssetRole::ALL
                    .into_iter()
                    .filter(|role| *role != NanoWheelsOffQualificationAssetRole::OnnxRuntimeLibrary)
                    .map(|role| launch.asset(role).relative_path())
                    .chain(
                        NanoWheelsOffQualificationV4AssetRole::ALL
                            .into_iter()
                            .filter_map(|role| {
                                launch.v4_asset(role).map(|asset| asset.relative_path())
                            }),
                    )
                    .chain(
                        NanoFaceCascadeAssetRole::ALL
                            .into_iter()
                            .map(|role| launch.face_perception().asset(role).relative_path()),
                    ),
            )
            .map_err(QualificationBootstrapPrimaryError::NativeRuntimeBinding)?;
        let native_runtime_dependency_identities = native_runtime
            .verify_dependencies_reusing_onnx(deployment_root, &onnx_runtime_library)
            .map_err(QualificationBootstrapPrimaryError::NativeRuntimeVerification)?;
        let onnx_runtime_path =
            deployment_root.join(onnx_runtime_library.relative_path().as_path());
        let pinned_onnx_runtime = crate::pin_ort_runtime_from_path(&onnx_runtime_path)
            .map_err(QualificationBootstrapPrimaryError::OnnxRuntimeInitialization)?;
        let mapped_images = verify_linux_mapped_qualification_images(
            &qualification_executable,
            &native_runtime_dependency_identities,
            &onnx_runtime_library,
        )
        .map_err(QualificationBootstrapPrimaryError::MappedImages)?;

        Ok(Self {
            qualification_executable,
            native_runtime_manifest,
            native_runtime,
            native_runtime_dependency_identities,
            agent_policy: load(NanoWheelsOffQualificationAssetRole::AgentPolicy)?,
            head_gaze_policy: load_v4(NanoWheelsOffQualificationV4AssetRole::HeadGazePolicy)?,
            frontal_face_cascade: load_face(NanoFaceCascadeAssetRole::FrontalFace)?,
            profile_face_cascade: load_face(NanoFaceCascadeAssetRole::ProfileFace)?,
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
            onnx_runtime_library,
            pinned_onnx_runtime,
            mapped_images,
            superpoint_model: load(NanoWheelsOffQualificationAssetRole::SuperpointModel)?,
            lightglue_model: load(NanoWheelsOffQualificationAssetRole::LightglueModel)?,
        })
    }
}

fn require_running_qualification_executable(
    expected: &StreamedDeploymentAssetIdentity,
) -> Result<(), QualificationExecutableIdentityError> {
    let (observed_bytes, observed_sha256, observed_file_identity) = running_executable_identity()?;
    require_executable_identity(
        expected.byte_len(),
        *expected.content_sha256().as_bytes(),
        expected.file_identity(),
        observed_bytes,
        observed_sha256,
        observed_file_identity,
    )
}

fn require_executable_identity(
    expected_bytes: u64,
    expected_sha256: [u8; 32],
    expected_file_identity: UnixFileIdentity,
    observed_bytes: u64,
    observed_sha256: [u8; 32],
    observed_file_identity: UnixFileIdentity,
) -> Result<(), QualificationExecutableIdentityError> {
    if observed_bytes != expected_bytes {
        return Err(QualificationExecutableIdentityError::SizeMismatch {
            expected_bytes,
            observed_bytes,
        });
    }
    if observed_sha256 != expected_sha256 {
        return Err(QualificationExecutableIdentityError::ContentMismatch {
            expected_sha256,
            observed_sha256,
        });
    }
    if observed_file_identity != expected_file_identity {
        return Err(QualificationExecutableIdentityError::FileIdentityMismatch {
            expected: expected_file_identity,
            observed: observed_file_identity,
        });
    }
    Ok(())
}

fn running_executable_identity()
-> Result<(u64, [u8; 32], UnixFileIdentity), QualificationExecutableIdentityError> {
    if !cfg!(target_os = "linux") {
        return Err(QualificationExecutableIdentityError::UnsupportedPlatform {
            target_os: std::env::consts::OS,
        });
    }
    let executable_path = PathBuf::from("/proc/self/exe");

    let mut executable = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC)
        .open(&executable_path)
        .map_err(|source| QualificationExecutableIdentityError::Open {
            path: executable_path.clone(),
            source,
        })?;
    let initial_metadata =
        executable
            .metadata()
            .map_err(|source| QualificationExecutableIdentityError::Metadata {
                path: executable_path.clone(),
                source,
            })?;
    if !initial_metadata.is_file() {
        return Err(QualificationExecutableIdentityError::NotRegularFile {
            path: executable_path,
        });
    }

    let mut observed_bytes = 0_u64;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 16 * 1_024];
    loop {
        let read = executable.read(&mut buffer).map_err(|source| {
            QualificationExecutableIdentityError::Read {
                path: executable_path.clone(),
                observed_bytes,
                source,
            }
        })?;
        if read == 0 {
            break;
        }
        let read_bytes = u64::try_from(read)
            .map_err(|_| QualificationExecutableIdentityError::ReadSizeNotRepresentable)?;
        observed_bytes = observed_bytes
            .checked_add(read_bytes)
            .ok_or(QualificationExecutableIdentityError::ObservedSizeOverflow)?;
        hasher.update(&buffer[..read]);
    }
    let final_metadata =
        executable
            .metadata()
            .map_err(|source| QualificationExecutableIdentityError::Metadata {
                path: executable_path.clone(),
                source,
            })?;
    if initial_metadata.len() != final_metadata.len() || observed_bytes != initial_metadata.len() {
        return Err(
            QualificationExecutableIdentityError::LengthChangedDuringRead {
                path: executable_path,
                initial_bytes: initial_metadata.len(),
                final_bytes: final_metadata.len(),
                observed_bytes,
            },
        );
    }
    let initial_identity = UnixFileIdentity::from_metadata(&initial_metadata);
    let final_identity = UnixFileIdentity::from_metadata(&final_metadata);
    if initial_identity != final_identity {
        return Err(
            QualificationExecutableIdentityError::IdentityChangedDuringRead {
                path: executable_path,
                initial: initial_identity,
                final_identity,
            },
        );
    }
    Ok((observed_bytes, hasher.finalize().into(), initial_identity))
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
    pub launch: LoadedNanoWheelsOffQualificationLaunchV4,
    pub assets: LoadedNanoWheelsOffQualificationAssets,
    pub head_gaze_policy: Option<QualificationHeadGazeProposalOnlyPolicy>,
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

/// A qualification head-gaze policy whose lifecycle was parsed as
/// proposal-only before any hardware was opened.
///
/// This is metadata for a future adapter. Its presence grants no torque,
/// motion, or head-command authority.
#[derive(Debug, PartialEq)]
pub struct QualificationHeadGazeProposalOnlyPolicy {
    policy: HeadGazePolicyV1,
}

impl QualificationHeadGazeProposalOnlyPolicy {
    pub const fn policy(&self) -> &HeadGazePolicyV1 {
        &self.policy
    }
}

impl TryFrom<HeadGazePolicyV1> for QualificationHeadGazeProposalOnlyPolicy {
    type Error = QualificationHeadGazePolicyAdmissionError;

    fn try_from(policy: HeadGazePolicyV1) -> Result<Self, Self::Error> {
        if !matches!(
            policy.lifecycle(),
            HeadGazePolicyLifecycleClaim::ProposalOnly(_)
        ) {
            return Err(QualificationHeadGazePolicyAdmissionError::NotProposalOnly);
        }
        Ok(Self { policy })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QualificationHeadGazePolicyAdmissionError {
    NotProposalOnly,
}

impl fmt::Display for QualificationHeadGazePolicyAdmissionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .write_str("qualification head-gaze policy must retain the proposal-only lifecycle")
    }
}

impl std::error::Error for QualificationHeadGazePolicyAdmissionError {}

fn parse_qualification_head_gaze_policy(
    asset: Option<&LoadedDeploymentAsset>,
) -> Result<Option<QualificationHeadGazeProposalOnlyPolicy>, QualificationBootstrapPrimaryError> {
    let Some(asset) = asset else {
        return Ok(None);
    };
    let policy = HeadGazePolicyV1::parse_json(asset.bytes())
        .map_err(QualificationBootstrapPrimaryError::HeadGazePolicy)?;
    QualificationHeadGazeProposalOnlyPolicy::try_from(policy)
        .map(Some)
        .map_err(QualificationBootstrapPrimaryError::HeadGazePolicyAdmission)
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
        fault_injection,
        running,
    } = request;

    require_linux_qualification_runtime()
        .map_err(QualificationBootstrapPrimaryError::QualificationExecutable)
        .map_err(QualificationBootstrapError::before_hardware)?;
    require_running(running)
        .map_err(QualificationBootstrapPrimaryError::common)
        .map_err(QualificationBootstrapError::before_hardware)?;
    let launch =
        load_nano_wheels_off_qualification_launch_v4(roots.deployment_root(), launch_relative_path)
            .map_err(QualificationBootstrapPrimaryError::LaunchLoad)
            .map_err(QualificationBootstrapError::before_hardware)?;
    let assets =
        LoadedNanoWheelsOffQualificationAssets::load(roots.deployment_root(), launch.launch())
            .map_err(QualificationBootstrapError::before_hardware)?;
    let head_gaze_policy = parse_qualification_head_gaze_policy(assets.head_gaze_policy.as_ref())
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
    let expected_rectified_stereo = calibration.rectified_stereo();
    let launch_rectified_stereo = launch.launch().oak().rectified_stereo();
    if expected_rectified_stereo.width() != launch_rectified_stereo.width_px()
        || expected_rectified_stereo.height() != launch_rectified_stereo.height_px()
    {
        return Err(QualificationBootstrapError::before_hardware(
            QualificationBootstrapPrimaryError::CalibrationLaunchStereoDimensionsMismatch {
                calibration_width_px: expected_rectified_stereo.width(),
                calibration_height_px: expected_rectified_stereo.height(),
                launch_width_px: launch_rectified_stereo.width_px(),
                launch_height_px: launch_rectified_stereo.height_px(),
            },
        ));
    }
    let expected_depth_camera = DepthCameraModel::new(
        expected_rectified_stereo.left(),
        expected_rectified_stereo.dimensions(),
        DepthToTrackingCamera::identity(),
    );
    let navigation = ShadowNavigationConfigV1::parse_json_bound_to_plant_artifact(
        assets.navigation_shadow_config.bytes(),
        expected_depth_camera,
        plant_artifact_model,
    )
    .map_err(QualificationBootstrapPrimaryError::ShadowNavigation)
    .map_err(QualificationBootstrapError::before_hardware)?;
    calibration
        .require_navigation(&navigation)
        .map_err(QualificationBootstrapPrimaryError::CalibrationBinding)
        .map_err(QualificationBootstrapError::before_hardware)?;
    candidate_admission
        .admit_shadow_mpc(navigation.mpc_solver().config())
        .map_err(QualificationBootstrapPrimaryError::CandidateMpc)
        .map_err(QualificationBootstrapError::before_hardware)?;
    let candidate_runtime_service_interval = candidate_limits
        .admit_runtime_service_interval(navigation.control_period().as_duration())
        .map_err(QualificationBootstrapPrimaryError::CandidateRuntimeServiceInterval)
        .map_err(QualificationBootstrapError::before_hardware)?;

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
        &calibration,
        navigation,
        candidate_runtime_service_interval,
        running,
    ) {
        Ok(connected) => connected,
        Err(primary) => return Err(close_oak_after_failure(primary, oak)),
    };

    let controller_serial_device = controller_server.serial_device().to_path_buf();
    let controller_owner_shutdown_timeout = controller_server.coordinated_shutdown_budget();
    let command_endpoint = launch.launch().controller_server().command_udp_endpoint();
    let controller_owner_result = match fault_injection.and_then(|fault| fault.serial_fault()) {
        Some(fault) => {
            V2ControllerOwner::start_operator_supervised_candidate_with_fault(
                controller_server,
                command_endpoint,
                fault,
            )
            .await
        }
        None => V2ControllerOwner::start(controller_server, command_endpoint).await,
    };
    let controller_owner = match controller_owner_result {
        Ok(owner) => owner,
        Err(source) => {
            return Err(close_oak_after_failure(
                QualificationBootstrapPrimaryError::ControllerOwnerStart(source),
                oak,
            ));
        }
    };
    let observed_command_endpoint = controller_owner.command_address();

    let session = match WheelsOffCandidateActuationSession::acquire_with_clock_fault(
        candidate_admission,
        controller_clock_origin,
        fault_injection.and_then(WheelsOffQualificationFaultInjection::host_clock_fault),
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
            head_gaze_policy,
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

fn require_linux_qualification_runtime() -> Result<(), QualificationExecutableIdentityError> {
    #[cfg(target_os = "linux")]
    {
        Ok(())
    }
    #[cfg(not(target_os = "linux"))]
    {
        Err(QualificationExecutableIdentityError::UnsupportedPlatform {
            target_os: std::env::consts::OS,
        })
    }
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
    launch: &NanoWheelsOffQualificationLaunchV4,
    calibration: &NanoCalibrationArtifactV1,
    navigation: ShadowNavigationConfigV1,
    candidate_runtime_service_interval: WheelsOffCandidateRuntimeServiceInterval,
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
    launch: &NanoWheelsOffQualificationLaunchV4,
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
    launch: &NanoWheelsOffQualificationLaunchV4,
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
    launch: &NanoWheelsOffQualificationLaunchV4,
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
pub enum QualificationExecutableIdentityError {
    UnsupportedPlatform {
        target_os: &'static str,
    },
    Open {
        path: PathBuf,
        source: io::Error,
    },
    Metadata {
        path: PathBuf,
        source: io::Error,
    },
    NotRegularFile {
        path: PathBuf,
    },
    Read {
        path: PathBuf,
        observed_bytes: u64,
        source: io::Error,
    },
    ReadSizeNotRepresentable,
    ObservedSizeOverflow,
    LengthChangedDuringRead {
        path: PathBuf,
        initial_bytes: u64,
        final_bytes: u64,
        observed_bytes: u64,
    },
    IdentityChangedDuringRead {
        path: PathBuf,
        initial: UnixFileIdentity,
        final_identity: UnixFileIdentity,
    },
    SizeMismatch {
        expected_bytes: u64,
        observed_bytes: u64,
    },
    ContentMismatch {
        expected_sha256: [u8; 32],
        observed_sha256: [u8; 32],
    },
    FileIdentityMismatch {
        expected: UnixFileIdentity,
        observed: UnixFileIdentity,
    },
}

impl fmt::Display for QualificationExecutableIdentityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "running wheels-off qualification executable identity mismatch: {self:?}"
        )
    }
}

impl std::error::Error for QualificationExecutableIdentityError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Open { source, .. }
            | Self::Metadata { source, .. }
            | Self::Read { source, .. } => Some(source),
            Self::UnsupportedPlatform { .. }
            | Self::NotRegularFile { .. }
            | Self::ReadSizeNotRepresentable
            | Self::ObservedSizeOverflow
            | Self::LengthChangedDuringRead { .. }
            | Self::IdentityChangedDuringRead { .. }
            | Self::SizeMismatch { .. }
            | Self::ContentMismatch { .. }
            | Self::FileIdentityMismatch { .. } => None,
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
    V4BoundAssetLoad {
        role: NanoWheelsOffQualificationV4AssetRole,
        source: NanoLaunchBoundAssetLoadError,
    },
    FaceBoundAssetLoad {
        role: NanoFaceCascadeAssetRole,
        source: NanoLaunchBoundAssetLoadError,
    },
    QualificationExecutable(QualificationExecutableIdentityError),
    NativeRuntimeManifest(NanoWheelsOffNativeRuntimeParseError),
    NativeRuntimeBinding(NanoWheelsOffNativeRuntimeBindingError),
    NativeRuntimeVerification(NanoWheelsOffNativeRuntimeVerificationError),
    OnnxRuntimeInitialization(crate::InferenceError),
    MappedImages(NanoWheelsOffMappedImageError),
    AgentPolicy(NanoAgentPolicyConfigParseError),
    HeadGazePolicy(HeadGazePolicyParseError),
    HeadGazePolicyAdmission(QualificationHeadGazePolicyAdmissionError),
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
    CalibrationLaunchStereoDimensionsMismatch {
        calibration_width_px: u32,
        calibration_height_px: u32,
        launch_width_px: u32,
        launch_height_px: u32,
    },
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
            Self::BoundAssetLoad { source, .. }
            | Self::V4BoundAssetLoad { source, .. }
            | Self::FaceBoundAssetLoad { source, .. } => Some(source),
            Self::QualificationExecutable(source) => Some(source),
            Self::NativeRuntimeManifest(source) => Some(source),
            Self::NativeRuntimeBinding(source) => Some(source),
            Self::NativeRuntimeVerification(source) => Some(source),
            Self::OnnxRuntimeInitialization(source) => Some(source),
            Self::MappedImages(source) => Some(source),
            Self::AgentPolicy(source) => Some(source),
            Self::HeadGazePolicy(source) => Some(source),
            Self::HeadGazePolicyAdmission(source) => Some(source),
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
            | Self::CalibrationLaunchStereoDimensionsMismatch { .. }
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

    #[cfg(target_os = "linux")]
    #[test]
    fn running_executable_identity_is_stable_and_mismatch_is_typed() {
        let (byte_len, sha256, file_identity) =
            running_executable_identity().expect("current test executable identity");
        require_executable_identity(
            byte_len,
            sha256,
            file_identity,
            byte_len,
            sha256,
            file_identity,
        )
        .expect("same exact executable identity");
        assert!(matches!(
            require_executable_identity(
                byte_len + 1,
                sha256,
                file_identity,
                byte_len,
                sha256,
                file_identity,
            ),
            Err(QualificationExecutableIdentityError::SizeMismatch { .. })
        ));
        let mut other_sha256 = sha256;
        other_sha256[0] ^= 1;
        assert!(matches!(
            require_executable_identity(
                byte_len,
                other_sha256,
                file_identity,
                byte_len,
                sha256,
                file_identity,
            ),
            Err(QualificationExecutableIdentityError::ContentMismatch { .. })
        ));
        let other_identity = UnixFileIdentity::from_metadata(
            &std::fs::metadata("/proc/self/maps").expect("maps metadata"),
        );
        assert!(matches!(
            require_executable_identity(
                byte_len,
                sha256,
                other_identity,
                byte_len,
                sha256,
                file_identity,
            ),
            Err(QualificationExecutableIdentityError::FileIdentityMismatch { .. })
        ));
    }

    #[cfg(not(target_os = "linux"))]
    #[test]
    fn qualification_runtime_explicitly_rejects_non_linux_hosts() {
        assert!(matches!(
            require_linux_qualification_runtime(),
            Err(QualificationExecutableIdentityError::UnsupportedPlatform { .. })
        ));
        assert!(matches!(
            running_executable_identity(),
            Err(QualificationExecutableIdentityError::UnsupportedPlatform { .. })
        ));
    }

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
