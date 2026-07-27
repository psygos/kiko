//! Production Nano bootstrap up to one disarmed, fully admitted live runtime.
//!
//! This module owns the cold-start ordering that cannot be expressed by the
//! individual parsers alone:
//!
//! 1. load every launch-bound input without following symlinks;
//! 2. parse and bind policy, manifest, controller, calibration, and plant
//!    contracts exactly once;
//! 3. wait boundedly for read-only presence of every exact parsed device;
//! 4. issue finite read-only head and eye identity probes;
//! 5. start and retain the sole manifest-bound natural-head/eye owner;
//! 6. open one exact OAK at SuperSpeed, retain its first stereo frames, and
//!    derive the runtime projection contract from observed intrinsics, then
//!    parse and bind navigation and actuation exactly once;
//! 7. exclusively start the in-process serial/UDP owner and await an exact
//!    ready-stopped controller heartbeat;
//! 8. acquire the sole controller session at an acknowledged exact zero;
//! 9. build observed inventory only from retained runtime evidence; and
//! 10. perform production admission, leaving the supervisor disarmed.
//!
//! No motion-bearing base API is exposed before exact admission. Once the
//! accessory owner is ready, every later failure keeps it alive through base
//! and OAK cleanup, then performs an explicit ownership release which issues
//! no head torque-switch write. That release is reported without claiming the
//! resulting physical torque state.

use std::fmt;
use std::fs;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::FileTypeExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use kiko_device_inventory::{
    ArtifactHashError, ArtifactId, ArtifactKind, ArtifactRelativePath, ArtifactRelativePathError,
    LoadedDeploymentAsset, ManifestArtifactHashes, ManifestLoadError, OakIdentity,
    Stm32StaticIdentity, hash_manifest_artifacts, load_expected_manifest_v1_file,
};
use kiko_expression_core::StreamEpochId;
use kiko_eye_runtime::{
    EyeIdentityObservation, IdentityProbeConfig, IdentityProbeError,
    SerialConfigurationEvidence as EyeSerialConfigurationEvidence, probe_serial_eye_identity,
};
use kiko_head_runtime::{
    HeadProbeConfig, HeadProbeReport, SerialHeadProbeError, probe_serial_head,
};
use kiko_supervisor_core::{ReadinessEpoch, SupervisorState};
use oak_sys::{
    CalibrationError as OakCalibrationError, CloseError as OakCloseError, ConnectedDeviceIdentity,
    ConnectedDeviceIdentityError, DepthAiBuildMetadata, DepthAiBuildMetadataError, Device,
    DeviceDiscoveryError, DeviceInfo, DeviceState, ImageError, ImageFrame as OakImageFrame,
    StreamId as OakStreamId, UsbTransportEvidenceError,
};
use robot_command_client::DisarmReceipt;
use robot_server::config::{ControllerServerConfigV1, ServerConfigError};
use robot_server::{
    V2ControllerOwner, V2ControllerOwnerStartError, V2ControllerOwnerTerminationError,
};

use super::actuation::LiveActuationError;
use super::mpc::{MpcConfigV1, PlantModelJsonParseError, PlantModelV1, WheelSide};
use super::nano_accessory_worker::NanoFacePerceptionAssets;
use super::{
    AdmittedOakSuperSpeedEvidence, ControlPeriodNs, ManifestBoundNanoAgentPolicyConfigV3,
    NanoAccessoryFaultWaitError, NanoAccessoryHealthPeriod, NanoAccessoryHealthPeriodError,
    NanoAccessoryManifestBindingError, NanoAccessoryPerceptionReadyEvidence,
    NanoAccessoryShutdownEvidence, NanoAccessoryTerminalFault, NanoAccessoryWorker,
    NanoAccessoryWorkerConfig, NanoAccessoryWorkerConfigError, NanoAccessoryWorkerExit,
    NanoAccessoryWorkerJoinError, NanoAccessoryWorkerStartError, NanoAgentLaunchLoadError,
    NanoAgentLaunchV3, NanoAgentPolicyConfigParseError, NanoAgentPolicyConfigV3,
    NanoCalibrationArtifactParseError, NanoCalibrationArtifactV1, NanoCalibrationBindingError,
    NanoFaceCascadeAssetRole, NanoFacePerceptionShutdownClass, NanoFacePerceptionShutdownEvidence,
    NanoLaunchAssetRole, NanoLaunchBoundAssetLoadError, NanoLaunchFacePerception,
    NanoObservedInventoryBuildError, NanoObservedInventoryBuilder,
    NanoObservedInventoryEvidenceError, NanoProductionAdmissionError,
    NanoProductionAdmissionTimeline, NanoProductionAdmissionTimelineError,
    NavigationActuationConfigV1, NavigationClockEpoch, PendingLiveMpcControlDriver,
    PreparedNanoProductionRuntime, ShadowNavigationConfigParseError, ShadowNavigationConfigV1,
    load_nano_agent_launch_v3,
};
use crate::dataset::{Calibration, CameraIntrinsics};
use crate::dense::occupancy::{DepthCameraModel, DepthToTrackingCamera};
use crate::live_runtime::LiveOccupancyHostPolicy;
use crate::{FrameDimensions, HostMonotonicTimestamp, RectifiedStereo, RectifiedStereoError};

const MAX_NANO_BOOTSTRAP_ROOT_BYTES: usize = 1_024;
const STEREO_POLL_TIMEOUT_MS: u32 = 50;
const STEREO_IDLE_SLEEP: Duration = Duration::from_micros(500);
const MAX_STEREO_BOOTSTRAP_WAIT: Duration = Duration::from_secs(15);
const NANO_BOOTSTRAP_ACCESSORY_HEALTH_PERIOD: Duration = Duration::from_secs(1);
const NANO_DEVICE_PRESENCE_POLL_INTERVAL: Duration = Duration::from_millis(100);
/// Fixed poll/sleep budget shared by production and wheels-off bootstrap.
///
/// A synchronous filesystem or DepthAI enumeration already in progress cannot
/// be preempted. Shutdown and deadline are checked after each serial metadata
/// call, before native discovery, and after the composite observation returns.
const NANO_DEVICE_PRESENCE_POLLING_BUDGET: Duration = Duration::from_secs(30);

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
    accessory_stream_epoch: StreamEpochId,
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
        accessory_stream_epoch: StreamEpochId,
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
            accessory_stream_epoch,
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

/// Exact device identities whose enumeration may lag filesystem availability
/// during a cold boot.
///
/// This borrowed value contains only identities which have already crossed
/// their launch, policy, manifest, and controller parsers. Presence is a timing
/// observation, not admission evidence: every downstream probe, connect, and
/// owner-open retains its existing one-shot ownership and admission semantics.
#[derive(Clone, Copy, Debug)]
pub(super) struct NanoExactDevicePresenceTargets<'target> {
    head: &'target HeadProbeConfig,
    eye: &'target IdentityProbeConfig,
    controller: &'target Stm32StaticIdentity,
    oak: &'target OakIdentity,
}

impl<'target> NanoExactDevicePresenceTargets<'target> {
    pub(super) const fn new(
        head: &'target HeadProbeConfig,
        eye: &'target IdentityProbeConfig,
        controller: &'target Stm32StaticIdentity,
        oak: &'target OakIdentity,
    ) -> Self {
        Self {
            head,
            eye,
            controller,
            oak,
        }
    }
}

/// Device role retained by a serial-presence boundary failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoSerialPresenceRole {
    Head,
    Eye,
    Controller,
}

/// One composite, read-only enumeration pass.
///
/// Serial booleans prove only that a character-device target was reported, and
/// the OAK value retains the discovery state for the exact MXID. The snapshot
/// does not prove atomic coexistence, ownership, identity readback, USB speed,
/// firmware, or continued presence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NanoDevicePresenceSnapshot {
    head_serial: bool,
    eye_serial: bool,
    controller_serial: bool,
    oak: NanoOakPresence,
}

impl NanoDevicePresenceSnapshot {
    const fn all_ready_for_acquisition_attempt(self) -> bool {
        self.head_serial
            && self.eye_serial
            && self.controller_serial
            && self.oak.ready_for_connect_attempt()
    }

    pub const fn head_serial_present(self) -> bool {
        self.head_serial
    }

    pub const fn eye_serial_present(self) -> bool {
        self.eye_serial
    }

    pub const fn controller_serial_present(self) -> bool {
        self.controller_serial
    }

    pub const fn oak_presence(self) -> NanoOakPresence {
        self.oak
    }
}

/// Exact OAK MXID observation retained without collapsing transitional states
/// into absence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoOakPresence {
    Missing,
    Available,
    InUse,
    Bootloader,
    Unknown,
}

impl NanoOakPresence {
    const fn ready_for_connect_attempt(self) -> bool {
        matches!(self, Self::Available | Self::InUse)
    }
}

impl From<DeviceState> for NanoOakPresence {
    fn from(state: DeviceState) -> Self {
        match state {
            DeviceState::Available => Self::Available,
            DeviceState::InUse => Self::InUse,
            DeviceState::Bootloader => Self::Bootloader,
            DeviceState::Unknown => Self::Unknown,
            _ => Self::Unknown,
        }
    }
}

/// Failure of one read-only presence observation.
#[derive(Debug)]
pub enum NanoDevicePresenceProbeError {
    SerialMetadata {
        role: NanoSerialPresenceRole,
        path: PathBuf,
        source: std::io::Error,
    },
    SerialTargetIsNotCharacterDevice {
        role: NanoSerialPresenceRole,
        path: PathBuf,
    },
    OakDiscovery(DeviceDiscoveryError),
}

impl fmt::Display for NanoDevicePresenceProbeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "exact-device presence probe failed: {self:?}")
    }
}

impl std::error::Error for NanoDevicePresenceProbeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::SerialMetadata { source, .. } => Some(source),
            Self::OakDiscovery(source) => Some(source),
            Self::SerialTargetIsNotCharacterDevice { .. } => None,
        }
    }
}

/// Why the bounded pre-probe enumeration phase did not complete.
///
/// The fixed budget bounds scheduled boundary calls, polling, and sleeps. A
/// synchronous filesystem or native discovery call already in progress cannot
/// be preempted, but no later component call starts after shutdown or deadline
/// is observed at the preceding boundary. Shutdown takes precedence when a
/// composite observation returns; a successful snapshot is then checked
/// against the deadline, while a probe error remains terminal.
#[derive(Debug)]
pub enum NanoDevicePresenceWaitError {
    Interrupted,
    Probe(NanoDevicePresenceProbeError),
    DeadlineOverflow {
        polling_budget: Duration,
    },
    TimedOut {
        polling_budget: Duration,
        attempts: u32,
        last_observation: Option<NanoDevicePresenceSnapshot>,
    },
}

impl fmt::Display for NanoDevicePresenceWaitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "bounded exact-device presence wait failed: {self:?}"
        )
    }
}

impl std::error::Error for NanoDevicePresenceWaitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Probe(source) => Some(source),
            Self::Interrupted | Self::DeadlineOverflow { .. } | Self::TimedOut { .. } => None,
        }
    }
}

trait NanoDevicePresenceProbe {
    fn observe(
        &mut self,
        checkpoint: &mut dyn FnMut() -> Result<(), NanoDevicePresenceWaitError>,
    ) -> Result<NanoDevicePresenceSnapshot, NanoDevicePresenceWaitError>;
}

trait NanoDevicePresenceWaitRuntime {
    fn now(&self) -> Instant;
    fn sleep(&mut self, duration: Duration);
}

struct SystemNanoDevicePresenceProbe<'target> {
    targets: NanoExactDevicePresenceTargets<'target>,
}

impl NanoDevicePresenceProbe for SystemNanoDevicePresenceProbe<'_> {
    fn observe(
        &mut self,
        checkpoint: &mut dyn FnMut() -> Result<(), NanoDevicePresenceWaitError>,
    ) -> Result<NanoDevicePresenceSnapshot, NanoDevicePresenceWaitError> {
        observe_device_presence_components(
            checkpoint,
            || {
                serial_character_device_is_present(
                    NanoSerialPresenceRole::Head,
                    Path::new(self.targets.head.device().path()),
                )
            },
            || {
                serial_character_device_is_present(
                    NanoSerialPresenceRole::Eye,
                    Path::new(self.targets.eye.device().path()),
                )
            },
            || {
                serial_character_device_is_present(
                    NanoSerialPresenceRole::Controller,
                    Path::new(self.targets.controller.serial_path().as_str()),
                )
            },
            || {
                // Device::list deliberately includes devices already in use.
                // InUse is retained as presence so the one-shot exclusive
                // connect reports the ownership conflict without another poll.
                let devices = Device::list().map_err(NanoDevicePresenceProbeError::OakDiscovery)?;
                Ok(exact_oak_mxid_presence(
                    &devices,
                    self.targets.oak.mxid().as_str(),
                ))
            },
        )
    }
}

fn observe_device_presence_components(
    checkpoint: &mut dyn FnMut() -> Result<(), NanoDevicePresenceWaitError>,
    head: impl FnOnce() -> Result<bool, NanoDevicePresenceProbeError>,
    eye: impl FnOnce() -> Result<bool, NanoDevicePresenceProbeError>,
    controller: impl FnOnce() -> Result<bool, NanoDevicePresenceProbeError>,
    oak: impl FnOnce() -> Result<NanoOakPresence, NanoDevicePresenceProbeError>,
) -> Result<NanoDevicePresenceSnapshot, NanoDevicePresenceWaitError> {
    checkpoint()?;
    let head_serial = head().map_err(NanoDevicePresenceWaitError::Probe)?;
    checkpoint()?;
    let eye_serial = eye().map_err(NanoDevicePresenceWaitError::Probe)?;
    checkpoint()?;
    let controller_serial = controller().map_err(NanoDevicePresenceWaitError::Probe)?;
    checkpoint()?;
    let oak = oak().map_err(NanoDevicePresenceWaitError::Probe)?;
    Ok(NanoDevicePresenceSnapshot {
        head_serial,
        eye_serial,
        controller_serial,
        oak,
    })
}

fn serial_character_device_is_present(
    role: NanoSerialPresenceRole,
    path: &Path,
) -> Result<bool, NanoDevicePresenceProbeError> {
    match fs::metadata(path) {
        Ok(metadata) if metadata.file_type().is_char_device() => Ok(true),
        Ok(_) => Err(
            NanoDevicePresenceProbeError::SerialTargetIsNotCharacterDevice {
                role,
                path: path.to_path_buf(),
            },
        ),
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(source) => Err(NanoDevicePresenceProbeError::SerialMetadata {
            role,
            path: path.to_path_buf(),
            source,
        }),
    }
}

fn exact_oak_mxid_presence(devices: &[DeviceInfo], expected_mxid: &str) -> NanoOakPresence {
    devices
        .iter()
        .find(|device| device.device_id == expected_mxid)
        .map_or(NanoOakPresence::Missing, |device| device.state.into())
}

struct SystemNanoDevicePresenceWaitRuntime;

impl NanoDevicePresenceWaitRuntime for SystemNanoDevicePresenceWaitRuntime {
    fn now(&self) -> Instant {
        Instant::now()
    }

    fn sleep(&mut self, duration: Duration) {
        thread::sleep(duration);
    }
}

pub(super) fn wait_for_exact_device_presence(
    targets: NanoExactDevicePresenceTargets<'_>,
    running: &AtomicBool,
) -> Result<(), NanoDevicePresenceWaitError> {
    let mut probe = SystemNanoDevicePresenceProbe { targets };
    wait_for_exact_device_presence_with(
        NANO_DEVICE_PRESENCE_POLLING_BUDGET,
        running,
        &mut probe,
        &mut SystemNanoDevicePresenceWaitRuntime,
    )
}

fn wait_for_exact_device_presence_with<Probe, Runtime>(
    polling_budget: Duration,
    running: &AtomicBool,
    probe: &mut Probe,
    runtime: &mut Runtime,
) -> Result<(), NanoDevicePresenceWaitError>
where
    Probe: NanoDevicePresenceProbe,
    Runtime: NanoDevicePresenceWaitRuntime,
{
    debug_assert!(!polling_budget.is_zero());
    let started_at = runtime.now();
    let deadline = started_at
        .checked_add(polling_budget)
        .ok_or(NanoDevicePresenceWaitError::DeadlineOverflow { polling_budget })?;
    let mut attempts = 0_u32;
    let mut last_observation = None;

    loop {
        if !running.load(Ordering::Acquire) {
            return Err(NanoDevicePresenceWaitError::Interrupted);
        }
        if let Some(previous_observation) = last_observation {
            let next_observation_at = runtime.now();
            if !running.load(Ordering::Acquire) {
                return Err(NanoDevicePresenceWaitError::Interrupted);
            }
            if next_observation_at >= deadline {
                return Err(NanoDevicePresenceWaitError::TimedOut {
                    polling_budget,
                    attempts,
                    last_observation: Some(previous_observation),
                });
            }
        }

        attempts = attempts.saturating_add(1);
        let mut checkpoint = || {
            if !running.load(Ordering::Acquire) {
                return Err(NanoDevicePresenceWaitError::Interrupted);
            }
            if runtime.now() >= deadline {
                return Err(NanoDevicePresenceWaitError::TimedOut {
                    polling_budget,
                    attempts,
                    last_observation,
                });
            }
            Ok(())
        };
        let observed = probe.observe(&mut checkpoint);
        if !running.load(Ordering::Acquire) {
            return Err(NanoDevicePresenceWaitError::Interrupted);
        }
        let observation = observed?;
        let observed_at = runtime.now();
        if !running.load(Ordering::Acquire) {
            return Err(NanoDevicePresenceWaitError::Interrupted);
        }
        if observation.all_ready_for_acquisition_attempt() && observed_at <= deadline {
            return Ok(());
        }
        if observed_at >= deadline {
            return Err(NanoDevicePresenceWaitError::TimedOut {
                polling_budget,
                attempts,
                last_observation: Some(observation),
            });
        }
        last_observation = Some(observation);
        if !running.load(Ordering::Acquire) {
            return Err(NanoDevicePresenceWaitError::Interrupted);
        }
        let remaining = deadline
            .checked_duration_since(observed_at)
            .expect("the strict deadline comparison proved positive duration");
        runtime.sleep(remaining.min(NANO_DEVICE_PRESENCE_POLL_INTERVAL));
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
    pub calibration_artifact: LoadedDeploymentAsset,
    pub plant_artifact: LoadedDeploymentAsset,
    pub onnx_runtime_library: LoadedDeploymentAsset,
    pub superpoint_model: LoadedDeploymentAsset,
    pub lightglue_model: LoadedDeploymentAsset,
}

impl LoadedNanoBootstrapAssets {
    fn load(
        deployment_root: &Path,
        launch: &NanoAgentLaunchV3,
    ) -> Result<(Self, LoadedNanoBootstrapFacePerceptionAssets), NanoBootstrapPrimaryError> {
        let load = |role| {
            launch
                .asset(role)
                .load_exact(deployment_root)
                .map_err(|source| NanoBootstrapPrimaryError::BoundAssetLoad { role, source })
        };
        let face_perception = LoadedNanoBootstrapFacePerceptionAssets::load(
            deployment_root,
            launch.face_perception(),
        )
        .map_err(NanoBootstrapPrimaryError::FacePerceptionAssetLoad)?;
        Ok((
            Self {
                agent_policy: load(NanoLaunchAssetRole::AgentPolicy)?,
                navigation_shadow_config: load(NanoLaunchAssetRole::NavigationShadowConfig)?,
                physical_actuation_config: load(NanoLaunchAssetRole::PhysicalActuationConfig)?,
                controller_server_contract: load(NanoLaunchAssetRole::ControllerServerContract)?,
                calibration_artifact: load(NanoLaunchAssetRole::CalibrationArtifact)?,
                plant_artifact: load(NanoLaunchAssetRole::PlantArtifact)?,
                onnx_runtime_library: load(NanoLaunchAssetRole::OnnxRuntimeLibrary)?,
                superpoint_model: load(NanoLaunchAssetRole::SuperpointModel)?,
                lightglue_model: load(NanoLaunchAssetRole::LightglueModel)?,
            },
            face_perception,
        ))
    }
}

/// Exact retained V3 face-cascade inputs.
///
/// `LoadedDeploymentAsset` proves that the retained bytes match the V3 launch
/// binding and retains their canonical deployment-relative identity. Startup
/// moves these byte vectors into the named face-perception thread and OpenCV
/// parses them from an in-memory `FileStorage`; no pathname is reopened and no
/// Rust-side duplicate vector is created. The native boundary makes one
/// required owned `std::string` copy per cascade for the startup parse.
#[derive(Debug)]
struct LoadedNanoBootstrapFacePerceptionAssets {
    frontal_face_cascade: LoadedDeploymentAsset,
    profile_face_cascade: LoadedDeploymentAsset,
}

impl LoadedNanoBootstrapFacePerceptionAssets {
    fn load(
        deployment_root: &Path,
        bindings: &NanoLaunchFacePerception,
    ) -> Result<Self, NanoBootstrapFacePerceptionAssetLoadError> {
        let load = |role| {
            bindings
                .asset(role)
                .load_exact(deployment_root)
                .map_err(
                    |source| NanoBootstrapFacePerceptionAssetLoadError::BoundAssetLoad {
                        role,
                        source,
                    },
                )
        };
        Ok(Self {
            frontal_face_cascade: load(NanoFaceCascadeAssetRole::FrontalFace)?,
            profile_face_cascade: load(NanoFaceCascadeAssetRole::ProfileFace)?,
        })
    }

    fn into_parts(self) -> (LoadedDeploymentAsset, LoadedDeploymentAsset) {
        (self.frontal_face_cascade, self.profile_face_cascade)
    }
}

#[derive(Debug)]
pub enum NanoBootstrapFacePerceptionAssetLoadError {
    BoundAssetLoad {
        role: NanoFaceCascadeAssetRole,
        source: NanoLaunchBoundAssetLoadError,
    },
}

impl fmt::Display for NanoBootstrapFacePerceptionAssetLoadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BoundAssetLoad { role, source } => {
                write!(formatter, "{role:?} face-cascade load failed: {source}")
            }
        }
    }
}

impl std::error::Error for NanoBootstrapFacePerceptionAssetLoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::BoundAssetLoad { source, .. } => Some(source),
        }
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
}

/// Successful cold-start handoff to the sole production runtime.
#[must_use = "the returned OAK and controller owners require explicit lifecycle handling"]
pub struct PreparedNanoBootstrap {
    pub roots: NanoBootstrapRoots,
    pub launch: super::LoadedNanoAgentLaunchV3,
    pub assets: LoadedNanoBootstrapAssets,
    pub accessory_evidence: NanoBootstrapAccessoryEvidence,
    pub oak_connected_identity: ConnectedDeviceIdentity,
    pub oak_usb_transport: AdmittedOakSuperSpeedEvidence,
    pub depthai_build_metadata: DepthAiBuildMetadata,
    pub calibration: NanoCalibrationArtifactV1,
    pub stereo: NanoBootstrapStereoEvidence,
    pub live: ParsedNanoLiveConfiguration,
    pub runtime: PreparedNanoProductionRuntime,
    pub accessory: NanoAccessoryWorker,
    pub oak: Device,
}

/// A fully prepared robot session plus the sole in-process STM32/UDP owner.
///
/// The outer state prevents callers from accidentally entering the live loop
/// while still depending on a separately managed `robot-server` process.
#[must_use = "split the bootstrap and supervise its controller owner for the complete live run"]
pub struct PreparedNanoOwnedBootstrap {
    bootstrap: PreparedNanoBootstrap,
    controller_owner: V2ControllerOwner,
    controller_owner_shutdown_timeout: Duration,
}

impl PreparedNanoOwnedBootstrap {
    pub fn into_parts(self) -> (PreparedNanoBootstrap, V2ControllerOwner, Duration) {
        (
            self.bootstrap,
            self.controller_owner,
            self.controller_owner_shutdown_timeout,
        )
    }
}

/// Complete production bootstrap.
///
/// The head and eye probes are finite, identity-only/read-only exchanges. Their
/// ports are released before the manifest-bound accessory owner establishes
/// the reviewed natural hold. The OAK is then opened once with the exact
/// launch graph. The in-process serial/UDP owner and zero-only controller
/// session are acquired last; every subsequent failure explicitly disarms the
/// client, joins the embedded owner, closes OAK, and only then asks the
/// accessory worker for a hold-preserving serial ownership release.
pub async fn bootstrap_nano_production(
    request: NanoBootstrapRequest<'_>,
) -> Result<PreparedNanoOwnedBootstrap, NanoBootstrapError> {
    let NanoBootstrapRequest {
        roots,
        launch_relative_path,
        accessory_stream_epoch,
        controller_clock_origin,
        navigation_clock_epoch,
        readiness_epoch,
        running,
    } = request;

    require_running(running).map_err(NanoBootstrapError::before_hardware)?;
    let launch = load_nano_agent_launch_v3(roots.deployment_root(), launch_relative_path).map_err(
        |source| NanoBootstrapError::before_hardware(NanoBootstrapPrimaryError::LaunchLoad(source)),
    )?;
    let (assets, face_perception_assets) =
        LoadedNanoBootstrapAssets::load(roots.deployment_root(), launch.launch())
            .map_err(NanoBootstrapError::before_hardware)?;

    let policy =
        NanoAgentPolicyConfigV3::parse_json(assets.agent_policy.bytes()).map_err(|source| {
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
    bind_calibration_artifact(
        &roots,
        launch.launch(),
        &assets.calibration_artifact,
        &policy,
        loaded_manifest.manifest(),
        &artifact_hashes,
    )
    .map_err(NanoBootstrapPrimaryError::CalibrationArtifactSelection)
    .map_err(NanoBootstrapError::before_hardware)?;
    let calibration = NanoCalibrationArtifactV1::parse_json(assets.calibration_artifact.bytes())
        .map_err(NanoBootstrapPrimaryError::CalibrationArtifact)
        .map_err(NanoBootstrapError::before_hardware)?;
    calibration
        .require_manifest_oak_mxid(loaded_manifest.manifest().oak().mxid().as_str())
        .map_err(NanoBootstrapPrimaryError::CalibrationBinding)
        .map_err(NanoBootstrapError::before_hardware)?;
    let selected_plant = select_plant_artifact(
        &roots,
        launch.launch(),
        &assets.plant_artifact,
        &policy,
        loaded_manifest.manifest(),
        &artifact_hashes,
    )
    .map_err(NanoBootstrapError::before_hardware)?;
    let plant_artifact_model = PlantModelV1::parse_json(assets.plant_artifact.bytes())
        .map_err(NanoBootstrapPrimaryError::PlantArtifactModel)
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

    // Cold-boot enumeration may lag local-fs without implying a wrong device.
    // Wait only for read-only presence of the exact parsed identities. Once
    // all are reported present in one pass, every probe/connect below remains a
    // single exclusive attempt: busy/in-use and all other open failures return
    // immediately and are never converted into another presence poll.
    wait_for_exact_device_presence(
        NanoExactDevicePresenceTargets::new(
            &head_probe_config,
            &eye_probe_config,
            loaded_manifest.manifest().stm32(),
            loaded_manifest.manifest().oak(),
        ),
        running,
    )
    .map_err(NanoBootstrapPrimaryError::DevicePresenceWait)
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

    // The finite probes above have released their read-only serial sessions.
    // Establish the sole manifest-bound head/eye owner before the potentially
    // slow OAK and STM32 admission below. Readiness proves the reviewed
    // natural return plus an immediate exact-target health transaction.
    let accessory_health_period =
        NanoAccessoryHealthPeriod::try_from_duration(NANO_BOOTSTRAP_ACCESSORY_HEALTH_PERIOD)
            .map_err(|source| {
                NanoBootstrapError::before_hardware(
                    NanoBootstrapPrimaryError::AccessoryHealthPeriod(source),
                )
            })?;
    let accessory_config = NanoAccessoryWorkerConfig::from_manifest_bound_policy(
        &probe_policy,
        accessory_stream_epoch,
        accessory_health_period,
    )
    .map_err(|source| {
        NanoBootstrapError::before_hardware(NanoBootstrapPrimaryError::AccessoryConfig(source))
    })?;
    let (frontal_face_cascade, profile_face_cascade) = face_perception_assets.into_parts();
    let face_perception_assets =
        NanoFacePerceptionAssets::from_v3_loaded_assets(frontal_face_cascade, profile_face_cascade);
    let accessory =
        NanoAccessoryWorker::start_with_face_perception(accessory_config, face_perception_assets)
            .map_err(|source| {
            NanoBootstrapError::before_hardware(NanoBootstrapPrimaryError::AccessoryStart(source))
        })?;
    if !matches!(
        accessory.readiness().perception(),
        NanoAccessoryPerceptionReadyEvidence::Face(_)
    ) {
        return Err(NanoBootstrapError::before_hardware(
            NanoBootstrapPrimaryError::AccessoryFacePerceptionReadinessMissing,
        )
        .with_accessory_shutdown(accessory));
    }
    if let Err(primary) = require_early_accessory_healthy(&accessory, running) {
        return Err(NanoBootstrapError::before_hardware(primary).with_accessory_shutdown(accessory));
    }

    let mut oak = match Device::connect(
        loaded_manifest.manifest().oak().mxid().as_str(),
        launch.launch().oak().device_config(),
    ) {
        Ok(oak) => oak,
        Err(source) => {
            return Err(NanoBootstrapError::before_hardware(
                NanoBootstrapPrimaryError::OakConnect(source),
            )
            .with_accessory_shutdown(accessory));
        }
    };

    let connected_request = ConnectedOakRequest {
        launch: launch.launch(),
        navigation_bytes: assets.navigation_shadow_config.bytes(),
        actuation_bytes: assets.physical_actuation_config.bytes(),
        calibration: &calibration,
        accessory: &accessory,
        plant_artifact_model,
        controller_server: &controller_server,
        robot_id: loaded_manifest.manifest().robot_id().as_str(),
        running,
    };
    let connected = match prepare_connected_oak(&mut oak, connected_request) {
        Ok(connected) => connected,
        Err(primary) => {
            return Err(close_oak_after_failure(primary, oak).with_accessory_shutdown(accessory));
        }
    };
    if let Err(primary) = require_early_accessory_healthy(&accessory, running) {
        return Err(close_oak_after_failure(primary, oak).with_accessory_shutdown(accessory));
    }

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
                NanoBootstrapPrimaryError::ControllerOwnerStart(source),
                oak,
            )
            .with_accessory_shutdown(accessory));
        }
    };
    if let Err(primary) = require_early_accessory_healthy(&accessory, running) {
        return Err(cleanup_after_owner_failure(
            primary,
            controller_owner,
            controller_owner_shutdown_timeout,
            oak,
        )
        .await
        .with_accessory_shutdown(accessory));
    }

    let (pending, initial_zero) =
        match PendingLiveMpcControlDriver::acquire(&connected.actuation, controller_clock_origin) {
            Ok(acquired) => acquired,
            Err(source) => {
                return Err(cleanup_after_owner_failure(
                    NanoBootstrapPrimaryError::ControllerAcquire(source),
                    controller_owner,
                    controller_owner_shutdown_timeout,
                    oak,
                )
                .await
                .with_accessory_shutdown(accessory));
            }
        };
    if let Err(primary) = require_early_accessory_healthy(&accessory, running) {
        return Err(cleanup_after_pending_failure(
            primary,
            pending,
            controller_owner,
            controller_owner_shutdown_timeout,
            oak,
        )
        .await
        .with_accessory_shutdown(accessory));
    }
    let acquisition = match pending.verified_controller_acquisition() {
        Ok(acquisition) => acquisition,
        Err(source) => {
            return Err(cleanup_after_pending_failure(
                NanoBootstrapPrimaryError::ControllerEvidence(source),
                pending,
                controller_owner,
                controller_owner_shutdown_timeout,
                oak,
            )
            .await
            .with_accessory_shutdown(accessory));
        }
    };

    let mut observed = NanoObservedInventoryBuilder::new();
    if let Err(source) = observed.observe_deployment(&connected.actuation) {
        return Err(cleanup_after_pending_failure(
            NanoBootstrapPrimaryError::ObservedInventoryEvidence(source),
            pending,
            controller_owner,
            controller_owner_shutdown_timeout,
            oak,
        )
        .await
        .with_accessory_shutdown(accessory));
    }
    let opened_identity = match oak.connected_identity() {
        Ok(identity) => identity,
        Err(source) => {
            return Err(cleanup_after_pending_failure(
                NanoBootstrapPrimaryError::OakConnectedIdentity(source),
                pending,
                controller_owner,
                controller_owner_shutdown_timeout,
                oak,
            )
            .await
            .with_accessory_shutdown(accessory));
        }
    };
    let retained_opened_identity = opened_identity.clone();
    let usb_transport = match oak.usb_transport_evidence() {
        Ok(evidence) => *evidence,
        Err(source) => {
            return Err(cleanup_after_pending_failure(
                NanoBootstrapPrimaryError::OakUsbTransport(source),
                pending,
                controller_owner,
                controller_owner_shutdown_timeout,
                oak,
            )
            .await
            .with_accessory_shutdown(accessory));
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
            controller_owner,
            controller_owner_shutdown_timeout,
            oak,
        )
        .await
        .with_accessory_shutdown(accessory));
    }
    if let Err(source) = observed.observe_stm32(&controller_serial_device, acquisition) {
        return Err(cleanup_after_pending_failure(
            NanoBootstrapPrimaryError::ObservedInventoryEvidence(source),
            pending,
            controller_owner,
            controller_owner_shutdown_timeout,
            oak,
        )
        .await
        .with_accessory_shutdown(accessory));
    }
    if let Err(source) = observed.observe_head(&head) {
        return Err(cleanup_after_pending_failure(
            NanoBootstrapPrimaryError::ObservedInventoryEvidence(source),
            pending,
            controller_owner,
            controller_owner_shutdown_timeout,
            oak,
        )
        .await
        .with_accessory_shutdown(accessory));
    }
    if let Err(source) = observed.observe_eye(&eye_serial, eye_identity) {
        return Err(cleanup_after_pending_failure(
            NanoBootstrapPrimaryError::ObservedInventoryEvidence(source),
            pending,
            controller_owner,
            controller_owner_shutdown_timeout,
            oak,
        )
        .await
        .with_accessory_shutdown(accessory));
    }
    if let Err(source) = observed.observe_artifacts(&artifact_hashes) {
        return Err(cleanup_after_pending_failure(
            NanoBootstrapPrimaryError::ObservedInventoryEvidence(source),
            pending,
            controller_owner,
            controller_owner_shutdown_timeout,
            oak,
        )
        .await
        .with_accessory_shutdown(accessory));
    }
    let observed = match observed.build() {
        Ok(observed) => observed,
        Err(source) => {
            return Err(cleanup_after_pending_failure(
                NanoBootstrapPrimaryError::ObservedInventoryBuild(source),
                pending,
                controller_owner,
                controller_owner_shutdown_timeout,
                oak,
            )
            .await
            .with_accessory_shutdown(accessory));
        }
    };
    let oak_usb_transport = observed.oak_super_speed();

    let readiness_admitted_at = match timestamp_since(controller_clock_origin) {
        Ok(timestamp) => timestamp,
        Err(primary) => {
            return Err(cleanup_after_pending_failure(
                primary,
                pending,
                controller_owner,
                controller_owner_shutdown_timeout,
                oak,
            )
            .await
            .with_accessory_shutdown(accessory));
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
                controller_owner,
                controller_owner_shutdown_timeout,
                oak,
            )
            .await
            .with_accessory_shutdown(accessory));
        }
    };

    if let Err(primary) = require_early_accessory_healthy(&accessory, running) {
        return Err(cleanup_after_pending_failure(
            primary,
            pending,
            controller_owner,
            controller_owner_shutdown_timeout,
            oak,
        )
        .await
        .with_accessory_shutdown(accessory));
    }

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
            let controller_owner =
                shutdown_controller_owner(controller_owner, controller_owner_shutdown_timeout)
                    .await;
            let oak_close = close_oak(oak);
            return Err(NanoBootstrapError {
                primary: Box::new(NanoBootstrapPrimaryError::ProductionAdmission(source)),
                controller: NanoBootstrapControllerDisposition::AdmissionErrorRetainsStop,
                controller_owner,
                oak_close,
                accessory: NanoBootstrapAccessoryDisposition::NotStarted,
            }
            .with_accessory_shutdown(accessory));
        }
    };

    debug_assert!(matches!(
        runtime.startup().authority.state(),
        SupervisorState::Disarmed { .. }
    ));
    Ok(PreparedNanoOwnedBootstrap {
        bootstrap: PreparedNanoBootstrap {
            roots,
            launch,
            assets,
            accessory_evidence: NanoBootstrapAccessoryEvidence {
                head,
                eye_serial,
                eye_identity,
            },
            oak_connected_identity: retained_opened_identity,
            oak_usb_transport,
            depthai_build_metadata: connected.depthai_build_metadata,
            calibration,
            stereo: connected.stereo,
            live: ParsedNanoLiveConfiguration {
                navigation: connected.navigation,
                occupancy_host_policy: connected.occupancy_host_policy,
            },
            runtime,
            accessory,
            oak,
        },
        controller_owner,
        controller_owner_shutdown_timeout,
    })
}

struct ConnectedOakPreparation {
    depthai_build_metadata: DepthAiBuildMetadata,
    stereo: NanoBootstrapStereoEvidence,
    navigation: ShadowNavigationConfigV1,
    occupancy_host_policy: LiveOccupancyHostPolicy,
    actuation: NavigationActuationConfigV1,
}

struct ConnectedOakRequest<'a> {
    launch: &'a NanoAgentLaunchV3,
    navigation_bytes: &'a [u8],
    actuation_bytes: &'a [u8],
    calibration: &'a NanoCalibrationArtifactV1,
    accessory: &'a NanoAccessoryWorker,
    plant_artifact_model: PlantModelV1,
    controller_server: &'a ControllerServerConfigV1,
    robot_id: &'a str,
    running: &'a AtomicBool,
}

fn prepare_connected_oak(
    oak: &mut Device,
    request: ConnectedOakRequest<'_>,
) -> Result<ConnectedOakPreparation, NanoBootstrapPrimaryError> {
    let ConnectedOakRequest {
        launch,
        navigation_bytes,
        actuation_bytes,
        calibration,
        accessory,
        plant_artifact_model,
        controller_server,
        robot_id,
        running,
    } = request;
    let connected_identity = oak
        .connected_identity()
        .map_err(NanoBootstrapPrimaryError::OakConnectedIdentity)?;
    if connected_identity.mxid().is_empty() {
        return Err(NanoBootstrapPrimaryError::EmptyOpenedOakMxid);
    }
    calibration
        .require_connected_oak_mxid(connected_identity.mxid())
        .map_err(NanoBootstrapPrimaryError::CalibrationBinding)?;
    let _usb = oak
        .usb_transport_evidence()
        .map_err(NanoBootstrapPrimaryError::OakUsbTransport)?;
    let depthai_build_metadata =
        oak_sys::depthai_build_metadata().map_err(NanoBootstrapPrimaryError::DepthAiBuild)?;
    let stereo = bootstrap_stereo_while(oak, launch.oak(), running, || {
        require_early_accessory_healthy(accessory, running)
    })?;
    calibration
        .require_observed_stereo(&stereo.calibration)
        .map_err(NanoBootstrapPrimaryError::CalibrationBinding)?;
    let navigation = ShadowNavigationConfigV1::parse_json_bound_to_plant_artifact(
        navigation_bytes,
        stereo.runtime_depth_camera,
        plant_artifact_model,
    )
    .map_err(NanoBootstrapPrimaryError::ShadowNavigation)?;
    calibration
        .require_navigation(&navigation)
        .map_err(NanoBootstrapPrimaryError::CalibrationBinding)?;
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
    calibration
        .require_actuation_approval(&actuation)
        .map_err(NanoBootstrapPrimaryError::CalibrationBinding)?;
    bind_controller_contract_to_actuation(
        launch,
        controller_server,
        &actuation,
        navigation.mpc_solver().config(),
        navigation.control_period(),
    )?;
    Ok(ConnectedOakPreparation {
        depthai_build_metadata,
        stereo,
        navigation,
        occupancy_host_policy,
        actuation,
    })
}

#[cfg(feature = "nano-wheels-off-qualification")]
pub(super) fn bootstrap_stereo(
    oak: &mut Device,
    oak_graph: &super::NanoOakStreamGraph,
    running: &AtomicBool,
) -> Result<NanoBootstrapStereoEvidence, NanoBootstrapPrimaryError> {
    bootstrap_stereo_while(oak, oak_graph, running, || Ok(()))
}

fn bootstrap_stereo_while(
    oak: &mut Device,
    oak_graph: &super::NanoOakStreamGraph,
    running: &AtomicBool,
    mut require_healthy: impl FnMut() -> Result<(), NanoBootstrapPrimaryError>,
) -> Result<NanoBootstrapStereoEvidence, NanoBootstrapPrimaryError> {
    let started = Instant::now();
    let mut left = None;
    let mut right = None;
    while left.is_none() || right.is_none() {
        require_running(running)?;
        require_healthy()?;
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
    require_healthy()?;

    let left = left.expect("loop exits only with a left frame");
    let right = right.expect("loop exits only with a right frame");
    validate_stereo_frame(
        NanoBootstrapStereoSide::Left,
        &left,
        oak_graph.rectified_stereo(),
    )?;
    validate_stereo_frame(
        NanoBootstrapStereoSide::Right,
        &right,
        oak_graph.rectified_stereo(),
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
        oak_eeprom: None,
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

pub(super) fn derive_required_probe_configs(
    policy: &ManifestBoundNanoAgentPolicyConfigV3,
) -> Result<(HeadProbeConfig, IdentityProbeConfig), NanoBootstrapPrimaryError> {
    let head = policy
        .head()
        .return_to_natural_and_hold_continuously()
        .ok_or(NanoBootstrapPrimaryError::ContinuousNaturalHeadHoldRequired)?;
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
    policy: &NanoAgentPolicyConfigV3,
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

fn bind_calibration_artifact(
    roots: &NanoBootstrapRoots,
    launch: &NanoAgentLaunchV3,
    loaded: &LoadedDeploymentAsset,
    policy: &NanoAgentPolicyConfigV3,
    manifest: &kiko_device_inventory::DeviceInventoryManifestV1,
    hashes: &ManifestArtifactHashes,
) -> Result<(), NanoCalibrationArtifactSelectionError> {
    let requested = launch.calibration_artifact().artifact_id().as_str();
    let expected = manifest
        .artifacts()
        .iter()
        .find(|artifact| {
            artifact.kind() == ArtifactKind::Calibration
                && artifact.artifact_id().as_str() == requested
        })
        .ok_or_else(|| NanoCalibrationArtifactSelectionError::NotInManifest {
            artifact_id: requested.to_owned(),
        })?;
    if expected.sha256().as_bytes() != loaded.content_sha256().as_bytes() {
        return Err(
            NanoCalibrationArtifactSelectionError::LaunchManifestDigestMismatch {
                artifact_id: requested.to_owned(),
                launch_sha256: *loaded.content_sha256().as_bytes(),
                manifest_sha256: *expected.sha256().as_bytes(),
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
            || NanoCalibrationArtifactSelectionError::PolicyBindingMissing {
                artifact_id: requested.to_owned(),
            },
        )?;
    let hashed = hashes
        .iter()
        .find(|artifact| {
            artifact.kind() == ArtifactKind::Calibration
                && artifact.artifact_id().as_str() == requested
        })
        .ok_or_else(
            || NanoCalibrationArtifactSelectionError::ObservedHashMissing {
                artifact_id: requested.to_owned(),
            },
        )?;
    if hashed.observed_sha256() != loaded.content_sha256().as_bytes() {
        return Err(
            NanoCalibrationArtifactSelectionError::LaunchObservedDigestMismatch {
                artifact_id: requested.to_owned(),
                launch_sha256: *loaded.content_sha256().as_bytes(),
                observed_sha256: *hashed.observed_sha256(),
            },
        );
    }
    let deployed_relative = calibration_deployment_relative_path(
        roots.deployment_root(),
        policy.inventory().artifact_root_path().as_path(),
        binding.relative_path(),
    )?;
    if &deployed_relative != launch.calibration_artifact().asset().relative_path() {
        return Err(
            NanoCalibrationArtifactSelectionError::DeploymentPathMismatch {
                artifact_id: requested.to_owned(),
                launch: launch
                    .calibration_artifact()
                    .asset()
                    .relative_path()
                    .clone(),
                policy: deployed_relative,
            },
        );
    }
    Ok(())
}

fn calibration_deployment_relative_path(
    deployment_root: &Path,
    artifact_root: &Path,
    artifact_relative_path: &ArtifactRelativePath,
) -> Result<ArtifactRelativePath, NanoCalibrationArtifactSelectionError> {
    let root_relative = artifact_root.strip_prefix(deployment_root).map_err(|_| {
        NanoCalibrationArtifactSelectionError::ArtifactRootOutsideDeployment {
            deployment_root: deployment_root.to_path_buf(),
            configured: artifact_root.to_path_buf(),
        }
    })?;
    let combined = root_relative.join(artifact_relative_path.as_path());
    let combined = combined.to_str().ok_or_else(|| {
        NanoCalibrationArtifactSelectionError::DeploymentPathNotUtf8 {
            path: combined.clone(),
        }
    })?;
    ArtifactRelativePath::parse(combined.to_owned())
        .map_err(NanoCalibrationArtifactSelectionError::DeploymentRelativePath)
}

#[derive(Debug)]
pub enum NanoCalibrationArtifactSelectionError {
    NotInManifest {
        artifact_id: String,
    },
    PolicyBindingMissing {
        artifact_id: String,
    },
    ObservedHashMissing {
        artifact_id: String,
    },
    LaunchManifestDigestMismatch {
        artifact_id: String,
        launch_sha256: [u8; 32],
        manifest_sha256: [u8; 32],
    },
    LaunchObservedDigestMismatch {
        artifact_id: String,
        launch_sha256: [u8; 32],
        observed_sha256: [u8; 32],
    },
    DeploymentPathMismatch {
        artifact_id: String,
        launch: ArtifactRelativePath,
        policy: ArtifactRelativePath,
    },
    ArtifactRootOutsideDeployment {
        deployment_root: PathBuf,
        configured: PathBuf,
    },
    DeploymentPathNotUtf8 {
        path: PathBuf,
    },
    DeploymentRelativePath(ArtifactRelativePathError),
}

impl fmt::Display for NanoCalibrationArtifactSelectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "calibration artifact launch/manifest/policy binding failed: {self:?}"
        )
    }
}

impl std::error::Error for NanoCalibrationArtifactSelectionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::DeploymentRelativePath(source) => Some(source),
            _ => None,
        }
    }
}

fn select_plant_artifact(
    roots: &NanoBootstrapRoots,
    launch: &NanoAgentLaunchV3,
    launch_plant: &LoadedDeploymentAsset,
    policy: &NanoAgentPolicyConfigV3,
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
    launch: &NanoAgentLaunchV3,
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
    launch: &NanoAgentLaunchV3,
    server: &ControllerServerConfigV1,
    actuation: &NavigationActuationConfigV1,
    mpc: MpcConfigV1,
    control_period: ControlPeriodNs,
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
    bind_mpc_pwm_to_controller_envelope(server.expected_max_abs_pwm_percent().get(), mpc)?;
    bind_navigation_cadence_to_controller(
        control_period.as_duration(),
        server.minimum_host_command_interval(),
        Duration::from_nanos(actuation.scheduling_guard_ns().get()),
    )?;
    Ok(())
}

fn bind_navigation_cadence_to_controller(
    control_period: Duration,
    controller_minimum_interval: Duration,
    scheduling_margin: Duration,
) -> Result<(), NanoBootstrapPrimaryError> {
    let required_exclusive_lower_bound = controller_minimum_interval
        .checked_add(scheduling_margin)
        .ok_or(NanoBootstrapPrimaryError::ActuationCadenceArithmeticOverflow)?;
    if control_period <= required_exclusive_lower_bound {
        return Err(
            NanoBootstrapPrimaryError::ActuationControlPeriodHasNoControllerRateMargin {
                control_period,
                controller_minimum_interval,
                scheduling_margin,
                required_exclusive_lower_bound,
            },
        );
    }
    Ok(())
}

fn bind_mpc_pwm_to_controller_envelope(
    controller_max_abs_percent: u8,
    mpc: MpcConfigV1,
) -> Result<(), NanoBootstrapPrimaryError> {
    for (wheel, (configured_min, configured_max)) in [
        (WheelSide::Left, mpc.left_pwm_bounds_percent()),
        (WheelSide::Right, mpc.right_pwm_bounds_percent()),
    ] {
        bind_one_mpc_pwm_range(
            wheel,
            configured_min,
            configured_max,
            controller_max_abs_percent,
        )?;
    }
    Ok(())
}

fn bind_one_mpc_pwm_range(
    wheel: WheelSide,
    configured_min_percent: i8,
    configured_max_percent: i8,
    controller_max_abs_percent: u8,
) -> Result<(), NanoBootstrapPrimaryError> {
    let controller_max = i16::from(controller_max_abs_percent);
    if i16::from(configured_min_percent) < -controller_max
        || i16::from(configured_max_percent) > controller_max
    {
        return Err(
            NanoBootstrapPrimaryError::ActuationMpcPwmOutsideControllerEnvelope {
                wheel,
                configured_min_percent,
                configured_max_percent,
                controller_max_abs_percent,
            },
        );
    }
    Ok(())
}

pub(super) fn require_running(running: &AtomicBool) -> Result<(), NanoBootstrapPrimaryError> {
    if running.load(Ordering::Acquire) {
        Ok(())
    } else {
        Err(NanoBootstrapPrimaryError::Interrupted)
    }
}

fn require_early_accessory_healthy(
    accessory: &NanoAccessoryWorker,
    running: &AtomicBool,
) -> Result<(), NanoBootstrapPrimaryError> {
    require_running(running)?;
    match accessory.try_terminal_fault() {
        Ok(None) => Ok(()),
        Ok(Some(fault)) => Err(NanoBootstrapPrimaryError::AccessoryTerminalFault(fault)),
        Err(source) => Err(NanoBootstrapPrimaryError::AccessoryFaultMonitor(source)),
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

#[cfg(test)]
type ControllerCleanupResult<Controller> = Result<
    <Controller as ExplicitBootstrapDisarm>::Receipt,
    <Controller as ExplicitBootstrapDisarm>::Error,
>;
#[cfg(test)]
type OakCleanupResult<Oak> = Result<(), <Oak as ExplicitBootstrapClose>::Error>;
#[cfg(test)]
type BootstrapCleanupResult<Controller, Oak> =
    (ControllerCleanupResult<Controller>, OakCleanupResult<Oak>);

#[cfg(test)]
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

async fn cleanup_after_pending_failure(
    primary: NanoBootstrapPrimaryError,
    pending: PendingLiveMpcControlDriver,
    controller_owner: V2ControllerOwner,
    controller_owner_shutdown_timeout: Duration,
    oak: Device,
) -> NanoBootstrapError {
    let stop = pending.explicit_disarm();
    let controller_owner =
        shutdown_controller_owner(controller_owner, controller_owner_shutdown_timeout).await;
    let close = oak.explicit_close();
    NanoBootstrapError {
        primary: Box::new(primary),
        controller: match stop {
            Ok(receipt) => NanoBootstrapControllerDisposition::ConfirmedStopped(receipt),
            Err(source) => NanoBootstrapControllerDisposition::StopUncertain(source),
        },
        controller_owner,
        oak_close: match close {
            Ok(()) => NanoBootstrapOakCloseDisposition::ConfirmedClosed,
            Err(source) => NanoBootstrapOakCloseDisposition::CloseUncertain(source),
        },
        accessory: NanoBootstrapAccessoryDisposition::NotStarted,
    }
}

async fn cleanup_after_owner_failure(
    primary: NanoBootstrapPrimaryError,
    controller_owner: V2ControllerOwner,
    controller_owner_shutdown_timeout: Duration,
    oak: Device,
) -> NanoBootstrapError {
    let controller_owner =
        shutdown_controller_owner(controller_owner, controller_owner_shutdown_timeout).await;
    NanoBootstrapError {
        primary: Box::new(primary),
        controller: NanoBootstrapControllerDisposition::NotAcquired,
        controller_owner,
        oak_close: close_oak(oak),
        accessory: NanoBootstrapAccessoryDisposition::NotStarted,
    }
}

async fn shutdown_controller_owner(
    controller_owner: V2ControllerOwner,
    shutdown_timeout: Duration,
) -> NanoBootstrapControllerOwnerDisposition {
    match controller_owner.shutdown(shutdown_timeout).await {
        Ok(()) => NanoBootstrapControllerOwnerDisposition::ConfirmedStopped,
        Err(source) => NanoBootstrapControllerOwnerDisposition::StopUncertain(source),
    }
}

fn close_oak_after_failure(primary: NanoBootstrapPrimaryError, oak: Device) -> NanoBootstrapError {
    NanoBootstrapError {
        primary: Box::new(primary),
        controller: NanoBootstrapControllerDisposition::NotAcquired,
        controller_owner: NanoBootstrapControllerOwnerDisposition::NotStarted,
        oak_close: close_oak(oak),
        accessory: NanoBootstrapAccessoryDisposition::NotStarted,
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
    controller_owner: NanoBootstrapControllerOwnerDisposition,
    oak_close: NanoBootstrapOakCloseDisposition,
    accessory: NanoBootstrapAccessoryDisposition,
}

impl NanoBootstrapError {
    fn before_hardware(primary: NanoBootstrapPrimaryError) -> Self {
        Self {
            primary: Box::new(primary),
            controller: NanoBootstrapControllerDisposition::NotAcquired,
            controller_owner: NanoBootstrapControllerOwnerDisposition::NotStarted,
            oak_close: NanoBootstrapOakCloseDisposition::NotOpened,
            accessory: NanoBootstrapAccessoryDisposition::NotStarted,
        }
    }

    fn with_accessory_shutdown(mut self, accessory: NanoAccessoryWorker) -> Self {
        debug_assert!(matches!(
            self.accessory,
            NanoBootstrapAccessoryDisposition::NotStarted
        ));
        let primary_terminal_fault = match self.primary.as_ref() {
            NanoBootstrapPrimaryError::AccessoryTerminalFault(fault) => Some(fault),
            _ => None,
        };
        self.accessory =
            NanoBootstrapAccessoryDisposition::shutdown(accessory, primary_terminal_fault);
        self
    }

    pub const fn primary(&self) -> &NanoBootstrapPrimaryError {
        &self.primary
    }

    pub const fn controller(&self) -> &NanoBootstrapControllerDisposition {
        &self.controller
    }

    pub const fn controller_owner(&self) -> &NanoBootstrapControllerOwnerDisposition {
        &self.controller_owner
    }

    pub const fn oak_close(&self) -> &NanoBootstrapOakCloseDisposition {
        &self.oak_close
    }

    pub const fn accessory(&self) -> &NanoBootstrapAccessoryDisposition {
        &self.accessory
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
            "Nano bootstrap failed: {}; {}; {}; {}; {}",
            self.primary, self.controller, self.controller_owner, self.oak_close, self.accessory
        )
    }
}

impl std::error::Error for NanoBootstrapError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.primary.as_ref())
    }
}

pub enum NanoBootstrapAccessoryDisposition {
    NotStarted,
    /// The accessory actor and face lane both reached an internally
    /// consistent terminal state. This does not imply healthy execution:
    /// `terminal_fault` identifies whether the sole first-fault value is
    /// retained here or was already moved into the primary bootstrap error.
    ShutdownCompleted {
        terminal_fault: NanoBootstrapAccessoryTerminalFaultDisposition,
        evidence: Box<NanoAccessoryShutdownEvidence>,
    },
    /// Eye/head cleanup joined, but the bounded face join detached at its
    /// deadline. The raw typed join evidence is retained and no cancellation
    /// or completed face shutdown is claimed.
    FacePerceptionShutdownUncertain {
        terminal_fault: NanoBootstrapAccessoryTerminalFaultDisposition,
        evidence: Box<NanoAccessoryShutdownEvidence>,
    },
    /// Eye/head cleanup joined, but the face exit and retained first terminal
    /// fault are not an internally consistent pair.
    FacePerceptionShutdownUnexpected {
        terminal_fault: NanoBootstrapAccessoryTerminalFaultDisposition,
        evidence: Box<NanoAccessoryShutdownEvidence>,
    },
    UnexpectedExit(Box<NanoAccessoryWorkerExit>),
    JoinUncertain(NanoAccessoryWorkerJoinError),
}

/// Where the accessory actor's retained first terminal fault is owned.
///
/// The worker publishes a clone of its stored first terminal fault to the sole
/// bootstrap observer. If that observer already moved the value into
/// [`NanoBootstrapPrimaryError::AccessoryTerminalFault`], shutdown records
/// only `AlreadyOwnedByPrimary`; it does not retain or print the worker's
/// equivalent clone a second time. Missing or conflicting shutdown evidence
/// remains explicit rather than being guessed equivalent.
#[derive(Debug)]
pub enum NanoBootstrapAccessoryTerminalFaultDisposition {
    NotObserved,
    RetainedByShutdown(NanoAccessoryTerminalFault),
    AlreadyOwnedByPrimary,
    MissingFromShutdownAfterPrimaryObservation,
    ConflictsWithPrimary(NanoAccessoryTerminalFault),
}

impl fmt::Display for NanoBootstrapAccessoryTerminalFaultDisposition {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotObserved => formatter.write_str("no accessory terminal fault was observed"),
            Self::RetainedByShutdown(fault) => {
                write!(formatter, "accessory shutdown retained terminal fault: {fault}")
            }
            Self::AlreadyOwnedByPrimary => formatter.write_str(
                "accessory first terminal fault is already owned by the primary bootstrap error",
            ),
            Self::MissingFromShutdownAfterPrimaryObservation => formatter.write_str(
                "primary bootstrap error owns an accessory terminal fault which shutdown did not retain",
            ),
            Self::ConflictsWithPrimary(fault) => write!(
                formatter,
                "accessory shutdown retained a terminal fault which conflicts with the primary bootstrap error: {fault}"
            ),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NanoBootstrapFaceShutdownDispositionKind {
    Completed,
    Uncertain,
    Unexpected,
}

fn classify_bootstrap_face_shutdown(
    face_perception: &NanoFacePerceptionShutdownEvidence,
    terminal_fault: Option<&NanoAccessoryTerminalFault>,
) -> NanoBootstrapFaceShutdownDispositionKind {
    match face_perception.classify(terminal_fault) {
        NanoFacePerceptionShutdownClass::Disabled
        | NanoFacePerceptionShutdownClass::CoordinatedShutdown
        | NanoFacePerceptionShutdownClass::PublishedRuntimeFault { .. }
        | NanoFacePerceptionShutdownClass::AccessoryFaultFollower { .. } => {
            NanoBootstrapFaceShutdownDispositionKind::Completed
        }
        NanoFacePerceptionShutdownClass::DetachedAfterTimeout { .. } => {
            NanoBootstrapFaceShutdownDispositionKind::Uncertain
        }
        NanoFacePerceptionShutdownClass::UnexpectedDisabledFaceFault { .. }
        | NanoFacePerceptionShutdownClass::UnexpectedJoined { .. } => {
            NanoBootstrapFaceShutdownDispositionKind::Unexpected
        }
    }
}

fn accessory_terminal_faults_are_equivalent(
    primary: &NanoAccessoryTerminalFault,
    shutdown: &NanoAccessoryTerminalFault,
) -> bool {
    match (primary, shutdown) {
        (
            NanoAccessoryTerminalFault::HeadHealth(primary),
            NanoAccessoryTerminalFault::HeadHealth(shutdown),
        ) => primary == shutdown,
        (
            NanoAccessoryTerminalFault::HeadHealthStatusPoisoned,
            NanoAccessoryTerminalFault::HeadHealthStatusPoisoned,
        )
        | (
            NanoAccessoryTerminalFault::RgbHealthStatusPoisoned,
            NanoAccessoryTerminalFault::RgbHealthStatusPoisoned,
        )
        | (
            NanoAccessoryTerminalFault::RgbIngressDisconnected,
            NanoAccessoryTerminalFault::RgbIngressDisconnected,
        )
        | (
            NanoAccessoryTerminalFault::RgbChannelPoisoned,
            NanoAccessoryTerminalFault::RgbChannelPoisoned,
        )
        | (
            NanoAccessoryTerminalFault::ReadinessObserverDropped,
            NanoAccessoryTerminalFault::ReadinessObserverDropped,
        ) => true,
        (
            NanoAccessoryTerminalFault::ExpressionBridge(primary),
            NanoAccessoryTerminalFault::ExpressionBridge(shutdown),
        ) => primary == shutdown,
        (
            NanoAccessoryTerminalFault::FacePerception(primary),
            NanoAccessoryTerminalFault::FacePerception(shutdown),
        ) => primary == shutdown,
        (
            NanoAccessoryTerminalFault::EyeApply(primary),
            NanoAccessoryTerminalFault::EyeApply(shutdown),
        ) => primary == shutdown,
        _ => false,
    }
}

fn reconcile_accessory_terminal_fault(
    primary: Option<&NanoAccessoryTerminalFault>,
    shutdown: Option<NanoAccessoryTerminalFault>,
) -> (NanoBootstrapAccessoryTerminalFaultDisposition, bool) {
    match (primary, shutdown) {
        (None, None) => (
            NanoBootstrapAccessoryTerminalFaultDisposition::NotObserved,
            true,
        ),
        (None, Some(fault)) => (
            NanoBootstrapAccessoryTerminalFaultDisposition::RetainedByShutdown(fault),
            true,
        ),
        (Some(_), None) => (
            NanoBootstrapAccessoryTerminalFaultDisposition::
                MissingFromShutdownAfterPrimaryObservation,
            false,
        ),
        (Some(primary), Some(shutdown))
            if accessory_terminal_faults_are_equivalent(primary, &shutdown) =>
        {
            (
                NanoBootstrapAccessoryTerminalFaultDisposition::AlreadyOwnedByPrimary,
                true,
            )
        }
        (Some(_), Some(shutdown)) => (
            NanoBootstrapAccessoryTerminalFaultDisposition::ConflictsWithPrimary(shutdown),
            false,
        ),
    }
}

impl NanoBootstrapAccessoryDisposition {
    fn shutdown(
        accessory: NanoAccessoryWorker,
        primary_terminal_fault: Option<&NanoAccessoryTerminalFault>,
    ) -> Self {
        match accessory.shutdown() {
            Ok(NanoAccessoryWorkerExit::Shutdown {
                terminal_fault,
                evidence,
            }) => {
                let mut kind = classify_bootstrap_face_shutdown(
                    evidence.face_perception(),
                    terminal_fault.as_ref(),
                );
                let (terminal_fault, terminal_fault_is_consistent) =
                    reconcile_accessory_terminal_fault(primary_terminal_fault, terminal_fault);
                if !terminal_fault_is_consistent {
                    kind = NanoBootstrapFaceShutdownDispositionKind::Unexpected;
                }
                match kind {
                    NanoBootstrapFaceShutdownDispositionKind::Completed => {
                        Self::ShutdownCompleted {
                            terminal_fault,
                            evidence,
                        }
                    }
                    NanoBootstrapFaceShutdownDispositionKind::Uncertain => {
                        Self::FacePerceptionShutdownUncertain {
                            terminal_fault,
                            evidence,
                        }
                    }
                    NanoBootstrapFaceShutdownDispositionKind::Unexpected => {
                        Self::FacePerceptionShutdownUnexpected {
                            terminal_fault,
                            evidence,
                        }
                    }
                }
            }
            Ok(exit) => Self::UnexpectedExit(Box::new(exit)),
            Err(source) => Self::JoinUncertain(source),
        }
    }
}

impl fmt::Debug for NanoBootstrapAccessoryDisposition {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, formatter)
    }
}

impl fmt::Display for NanoBootstrapAccessoryDisposition {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotStarted => formatter.write_str("accessory owner was not started"),
            Self::ShutdownCompleted {
                terminal_fault,
                evidence,
            } => write!(
                formatter,
                "accessory owner shutdown joined with internally consistent face evidence (terminal_fault={terminal_fault}, eye_release_verified={}, head_hold_preserving_release_completed={}, face_perception_classification={:?}, face_perception_evidence={:?})",
                evidence.eye().release_verified(),
                evidence.head().hold_preserving_release_completed(),
                NanoBootstrapFaceShutdownDispositionKind::Completed,
                evidence.face_perception(),
            ),
            Self::FacePerceptionShutdownUncertain {
                terminal_fault,
                evidence,
            } => write!(
                formatter,
                "accessory eye/head shutdown joined but face-perception shutdown is uncertain (terminal_fault={terminal_fault}, eye_release_verified={}, head_hold_preserving_release_completed={}, face_perception_classification={:?}, face_perception_evidence={:?})",
                evidence.eye().release_verified(),
                evidence.head().hold_preserving_release_completed(),
                NanoBootstrapFaceShutdownDispositionKind::Uncertain,
                evidence.face_perception(),
            ),
            Self::FacePerceptionShutdownUnexpected {
                terminal_fault,
                evidence,
            } => write!(
                formatter,
                "accessory eye/head shutdown joined but face-perception shutdown evidence is unexpected (terminal_fault={terminal_fault}, eye_release_verified={}, head_hold_preserving_release_completed={}, face_perception_classification={:?}, face_perception_evidence={:?})",
                evidence.eye().release_verified(),
                evidence.head().hold_preserving_release_completed(),
                NanoBootstrapFaceShutdownDispositionKind::Unexpected,
                evidence.face_perception(),
            ),
            Self::UnexpectedExit(exit) => {
                write!(
                    formatter,
                    "accessory owner returned unexpected exit: {exit:?}"
                )
            }
            Self::JoinUncertain(source) => {
                write!(
                    formatter,
                    "accessory owner shutdown join uncertain: {source}"
                )
            }
        }
    }
}

pub enum NanoBootstrapControllerOwnerDisposition {
    NotStarted,
    ConfirmedStopped,
    StopUncertain(V2ControllerOwnerTerminationError),
}

impl fmt::Debug for NanoBootstrapControllerOwnerDisposition {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, formatter)
    }
}

impl fmt::Display for NanoBootstrapControllerOwnerDisposition {
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
    FacePerceptionAssetLoad(NanoBootstrapFacePerceptionAssetLoadError),
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
    ContinuousNaturalHeadHoldRequired,
    Kep2EyeRequired,
    ArtifactHash(ArtifactHashError),
    CalibrationArtifactSelection(NanoCalibrationArtifactSelectionError),
    CalibrationArtifact(NanoCalibrationArtifactParseError),
    CalibrationBinding(NanoCalibrationBindingError),
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
    PlantArtifactModel(PlantModelJsonParseError),
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
    DevicePresenceWait(NanoDevicePresenceWaitError),
    HeadProbe(Box<SerialHeadProbeError>),
    EyeProbe(IdentityProbeError),
    AccessoryHealthPeriod(NanoAccessoryHealthPeriodError),
    AccessoryConfig(NanoAccessoryWorkerConfigError),
    AccessoryStart(NanoAccessoryWorkerStartError),
    AccessoryFacePerceptionReadinessMissing,
    AccessoryTerminalFault(NanoAccessoryTerminalFault),
    AccessoryFaultMonitor(NanoAccessoryFaultWaitError),
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
    ActuationMpcPwmOutsideControllerEnvelope {
        wheel: WheelSide,
        configured_min_percent: i8,
        configured_max_percent: i8,
        controller_max_abs_percent: u8,
    },
    ActuationCadenceArithmeticOverflow,
    ActuationControlPeriodHasNoControllerRateMargin {
        control_period: Duration,
        controller_minimum_interval: Duration,
        scheduling_margin: Duration,
        required_exclusive_lower_bound: Duration,
    },
    ControllerOwnerStart(V2ControllerOwnerStartError),
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
            Self::FacePerceptionAssetLoad(source) => Some(source),
            Self::AgentPolicy(source) => Some(source),
            Self::Manifest(source) => Some(source),
            Self::AccessoryManifestBinding(source) => Some(source),
            Self::ArtifactHash(source) => Some(source),
            Self::CalibrationArtifactSelection(source) => Some(source),
            Self::CalibrationArtifact(source) => Some(source),
            Self::CalibrationBinding(source) => Some(source),
            Self::PlantDeploymentRelativePath(source) => Some(source),
            Self::PlantArtifactModel(source) => Some(source),
            Self::ControllerServerContract(source) => Some(source),
            Self::DevicePresenceWait(source) => Some(source),
            Self::HeadProbe(source) => Some(source.as_ref()),
            Self::EyeProbe(source) => Some(source),
            Self::AccessoryHealthPeriod(source) => Some(source),
            Self::AccessoryConfig(source) => Some(source),
            Self::AccessoryStart(source) => Some(source),
            Self::AccessoryFaultMonitor(source) => Some(source),
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
            Self::ControllerOwnerStart(source) => Some(source),
            Self::ControllerAcquire(source) | Self::ControllerEvidence(source) => Some(source),
            Self::ObservedInventoryEvidence(source) => Some(source),
            Self::ObservedInventoryBuild(source) => Some(source),
            Self::AdmissionTimeline(source) => Some(source),
            Self::ProductionAdmission(source) => Some(source),
            Self::Interrupted
            | Self::MonotonicTimestampOverflow { .. }
            | Self::PolicyPathOutsideDeployment { .. }
            | Self::PolicyPathAliasesDeploymentRoot { .. }
            | Self::ContinuousNaturalHeadHoldRequired
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
            | Self::AccessoryFacePerceptionReadinessMissing
            | Self::AccessoryTerminalFault(_)
            | Self::EmptyOpenedOakMxid
            | Self::StereoTimedOut { .. }
            | Self::StereoUnexpectedStream { .. }
            | Self::StereoUnexpectedDimensions { .. }
            | Self::StereoIntrinsicsDimensionsMismatch { .. }
            | Self::ActuationCommandEndpointMismatch { .. }
            | Self::ActuationControllerUidMismatch
            | Self::ActuationControllerFirmwareAbiMismatch { .. }
            | Self::ActuationControllerFirmwareBuildMismatch { .. }
            | Self::ActuationControllerFingerprintMismatch
            | Self::ActuationCadenceArithmeticOverflow
            | Self::ActuationControlPeriodHasNoControllerRateMargin { .. }
            | Self::ActuationMpcPwmOutsideControllerEnvelope { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::{Cell, RefCell};
    use std::collections::VecDeque;
    use std::os::unix::fs::symlink;
    use std::rc::Rc;
    use std::sync::atomic::AtomicU64;

    use crate::navigation::{
        NanoFacePerceptionJoinEvidence, NanoFacePerceptionRuntimeError,
        NanoFacePerceptionThreadExit,
    };

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

    #[test]
    fn bootstrap_request_retains_the_exact_accessory_stream_epoch() {
        let running = AtomicBool::new(true);
        let stream_epoch = StreamEpochId::try_new(41).expect("non-zero epoch");
        let request = NanoBootstrapRequest::try_new(
            "/opt/kiko/deployment".into(),
            "/var/lib/kiko".into(),
            "launch.json".into(),
            stream_epoch,
            Instant::now(),
            NavigationClockEpoch::new(HostMonotonicTimestamp::from_nanos(0)),
            ReadinessEpoch::try_new(1).expect("non-zero readiness epoch"),
            &running,
        )
        .expect("valid bootstrap request");

        assert_eq!(request.accessory_stream_epoch, stream_epoch);
    }

    const fn presence(
        head_serial: bool,
        eye_serial: bool,
        controller_serial: bool,
        oak: bool,
    ) -> NanoDevicePresenceSnapshot {
        presence_with_oak(
            head_serial,
            eye_serial,
            controller_serial,
            if oak {
                NanoOakPresence::Available
            } else {
                NanoOakPresence::Missing
            },
        )
    }

    const fn presence_with_oak(
        head_serial: bool,
        eye_serial: bool,
        controller_serial: bool,
        oak: NanoOakPresence,
    ) -> NanoDevicePresenceSnapshot {
        NanoDevicePresenceSnapshot {
            head_serial,
            eye_serial,
            controller_serial,
            oak,
        }
    }

    static NEXT_PRESENCE_PATH_FIXTURE: AtomicU64 = AtomicU64::new(0);

    struct PresencePathFixture {
        root: PathBuf,
    }

    impl PresencePathFixture {
        fn new() -> Self {
            let sequence = NEXT_PRESENCE_PATH_FIXTURE.fetch_add(1, Ordering::Relaxed);
            let root = std::env::temp_dir().join(format!(
                "kiko-device-presence-{}-{sequence}",
                std::process::id()
            ));
            fs::create_dir(&root).expect("create exact-device presence fixture");
            Self { root }
        }

        fn path(&self, name: &str) -> PathBuf {
            self.root.join(name)
        }
    }

    impl Drop for PresencePathFixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    #[test]
    fn serial_presence_distinguishes_character_missing_and_wrong_type() {
        let fixture = PresencePathFixture::new();
        let head = fixture.path("head-by-id");
        symlink("/dev/null", &head).expect("link exact path to character device");
        assert!(
            serial_character_device_is_present(NanoSerialPresenceRole::Head, &head)
                .expect("character-device metadata")
        );

        let missing_path = fixture.path("missing-by-id");
        assert!(
            !serial_character_device_is_present(NanoSerialPresenceRole::Eye, &missing_path)
                .expect("missing serial target is ordinary absence")
        );
        let dangling = fixture.path("eye-by-id");
        symlink(fixture.path("not-enumerated"), &dangling).expect("create dangling serial link");
        assert!(
            !serial_character_device_is_present(NanoSerialPresenceRole::Eye, &dangling)
                .expect("dangling serial target is ordinary absence")
        );

        let wrong_type = fixture.path("controller-by-id");
        fs::write(&wrong_type, b"not a tty").expect("write non-device target");
        assert!(matches!(
            serial_character_device_is_present(
                NanoSerialPresenceRole::Controller,
                &wrong_type,
            ),
            Err(
                NanoDevicePresenceProbeError::SerialTargetIsNotCharacterDevice {
                    role: NanoSerialPresenceRole::Controller,
                    path,
                }
            ) if path == wrong_type
        ));
    }

    fn stop_presence_components_at_checkpoint(
        stop: NanoDevicePresenceWaitError,
        stop_at_checkpoint: u32,
    ) -> (NanoDevicePresenceWaitError, Vec<&'static str>) {
        let calls = Rc::new(RefCell::new(Vec::new()));
        let head_calls = Rc::clone(&calls);
        let eye_calls = Rc::clone(&calls);
        let controller_calls = Rc::clone(&calls);
        let oak_calls = Rc::clone(&calls);
        let mut stop = Some(stop);
        let mut checkpoints = 0_u32;
        let error = observe_device_presence_components(
            &mut || {
                checkpoints = checkpoints.saturating_add(1);
                if checkpoints == stop_at_checkpoint {
                    Err(stop
                        .take()
                        .expect("the selected checkpoint is reached once"))
                } else {
                    Ok(())
                }
            },
            || {
                head_calls.borrow_mut().push("head");
                Ok(true)
            },
            || {
                eye_calls.borrow_mut().push("eye");
                Ok(true)
            },
            || {
                controller_calls.borrow_mut().push("controller");
                Ok(true)
            },
            || {
                oak_calls.borrow_mut().push("oak");
                Ok(NanoOakPresence::Available)
            },
        )
        .expect_err("the first component checkpoint stops the observation");
        let calls = calls.borrow().clone();
        (error, calls)
    }

    #[test]
    fn component_checkpoints_prevent_later_boundaries_after_signal_or_deadline() {
        let (error, calls) =
            stop_presence_components_at_checkpoint(NanoDevicePresenceWaitError::Interrupted, 1);
        assert!(matches!(error, NanoDevicePresenceWaitError::Interrupted));
        assert!(calls.is_empty());

        let polling_budget = Duration::from_secs(1);
        let (error, calls) = stop_presence_components_at_checkpoint(
            NanoDevicePresenceWaitError::TimedOut {
                polling_budget,
                attempts: 1,
                last_observation: None,
            },
            2,
        );
        assert!(matches!(
            error,
            NanoDevicePresenceWaitError::TimedOut {
                polling_budget: actual,
                attempts: 1,
                last_observation: None,
            } if actual == polling_budget
        ));
        assert_eq!(calls, ["head"]);
    }

    struct ScriptedPresenceProbe<'running> {
        observations: VecDeque<Result<NanoDevicePresenceSnapshot, NanoDevicePresenceProbeError>>,
        calls: u32,
        interrupt: Option<&'running AtomicBool>,
    }

    impl ScriptedPresenceProbe<'_> {
        fn new(
            observations: impl IntoIterator<
                Item = Result<NanoDevicePresenceSnapshot, NanoDevicePresenceProbeError>,
            >,
        ) -> Self {
            Self {
                observations: observations.into_iter().collect(),
                calls: 0,
                interrupt: None,
            }
        }
    }

    impl<'running> ScriptedPresenceProbe<'running> {
        fn interrupt(mut self, running: &'running AtomicBool) -> Self {
            self.interrupt = Some(running);
            self
        }
    }

    impl NanoDevicePresenceProbe for ScriptedPresenceProbe<'_> {
        fn observe(
            &mut self,
            _checkpoint: &mut dyn FnMut() -> Result<(), NanoDevicePresenceWaitError>,
        ) -> Result<NanoDevicePresenceSnapshot, NanoDevicePresenceWaitError> {
            self.calls = self.calls.saturating_add(1);
            if let Some(running) = self.interrupt {
                running.store(false, Ordering::Release);
            }
            self.observations
                .pop_front()
                .expect("test supplied one observation per expected attempt")
                .map_err(NanoDevicePresenceWaitError::Probe)
        }
    }

    struct FakePresenceWaitRuntime {
        now: Rc<Cell<Instant>>,
        sleeps: Vec<Duration>,
    }

    impl FakePresenceWaitRuntime {
        fn new(now: Instant) -> Self {
            Self {
                now: Rc::new(Cell::new(now)),
                sleeps: Vec::new(),
            }
        }

        fn clock(&self) -> Rc<Cell<Instant>> {
            Rc::clone(&self.now)
        }
    }

    impl NanoDevicePresenceWaitRuntime for FakePresenceWaitRuntime {
        fn now(&self) -> Instant {
            self.now.get()
        }

        fn sleep(&mut self, duration: Duration) {
            self.sleeps.push(duration);
            self.now.set(self.now.get() + duration);
        }
    }

    fn run_presence_wait(
        polling_budget: Duration,
        running: &AtomicBool,
        probe: &mut impl NanoDevicePresenceProbe,
        runtime: &mut FakePresenceWaitRuntime,
    ) -> Result<(), NanoDevicePresenceWaitError> {
        wait_for_exact_device_presence_with(polling_budget, running, probe, runtime)
    }

    #[test]
    fn presence_wait_returns_without_sleep_after_one_complete_polling_pass() {
        let running = AtomicBool::new(true);
        let mut probe = ScriptedPresenceProbe::new([Ok(presence(true, true, true, true))]);
        let mut runtime = FakePresenceWaitRuntime::new(Instant::now());

        run_presence_wait(Duration::from_secs(1), &running, &mut probe, &mut runtime)
            .expect("all exact targets were present");

        assert_eq!(probe.calls, 1);
        assert!(runtime.sleeps.is_empty());
    }

    #[test]
    fn presence_wait_retries_until_one_pass_reports_all_exact_targets() {
        let running = AtomicBool::new(true);
        let mut probe = ScriptedPresenceProbe::new([
            Ok(presence(true, true, true, false)),
            Ok(presence(true, true, true, true)),
        ]);
        let mut runtime = FakePresenceWaitRuntime::new(Instant::now());

        run_presence_wait(Duration::from_secs(1), &running, &mut probe, &mut runtime)
            .expect("the exact OAK appeared inside the polling window");

        assert_eq!(probe.calls, 2);
        assert_eq!(runtime.sleeps, [NANO_DEVICE_PRESENCE_POLL_INTERVAL]);
    }

    #[test]
    fn presence_wait_stops_exactly_at_deadline_and_retains_missing_roles() {
        let running = AtomicBool::new(true);
        let missing = presence(true, false, true, false);
        let mut probe = ScriptedPresenceProbe::new([Ok(missing), Ok(missing), Ok(missing)]);
        let mut runtime = FakePresenceWaitRuntime::new(Instant::now());
        let polling_budget = Duration::from_millis(250);

        let error = run_presence_wait(polling_budget, &running, &mut probe, &mut runtime)
            .expect_err("eye and OAK never appeared");

        assert!(matches!(
            error,
            NanoDevicePresenceWaitError::TimedOut {
                polling_budget: actual,
                attempts: 3,
                last_observation,
            } if actual == polling_budget && last_observation == Some(missing)
        ));
        assert_eq!(probe.calls, 3);
        assert_eq!(
            runtime.sleeps,
            [
                Duration::from_millis(100),
                Duration::from_millis(100),
                Duration::from_millis(50),
            ]
        );
    }

    #[test]
    fn presence_timeout_retains_transitional_oak_state() {
        let running = AtomicBool::new(true);
        let observation = presence_with_oak(true, true, true, NanoOakPresence::Bootloader);
        let mut probe = ScriptedPresenceProbe::new([Ok(observation)]);
        let mut runtime = FakePresenceWaitRuntime::new(Instant::now());
        let polling_budget = Duration::from_millis(1);

        let error = run_presence_wait(polling_budget, &running, &mut probe, &mut runtime)
            .expect_err("bootloader state is not ready for a connect attempt");

        assert!(matches!(
            error,
            NanoDevicePresenceWaitError::TimedOut {
                attempts: 1,
                last_observation,
                ..
            } if last_observation
                .is_some_and(|snapshot| {
                    snapshot.oak_presence() == NanoOakPresence::Bootloader
                })
        ));
        assert_eq!(probe.calls, 1);
        assert_eq!(runtime.sleeps, [polling_budget]);
    }

    struct ClockAdvancingPresenceProbe {
        clock: Rc<Cell<Instant>>,
        advance: Duration,
        calls: u32,
    }

    impl NanoDevicePresenceProbe for ClockAdvancingPresenceProbe {
        fn observe(
            &mut self,
            _checkpoint: &mut dyn FnMut() -> Result<(), NanoDevicePresenceWaitError>,
        ) -> Result<NanoDevicePresenceSnapshot, NanoDevicePresenceWaitError> {
            self.calls = self.calls.saturating_add(1);
            self.clock.set(self.clock.get() + self.advance);
            Ok(presence(true, true, true, true))
        }
    }

    #[test]
    fn all_present_observation_that_completes_after_deadline_times_out() {
        let running = AtomicBool::new(true);
        let polling_budget = Duration::from_millis(250);
        let mut runtime = FakePresenceWaitRuntime::new(Instant::now());
        let mut probe = ClockAdvancingPresenceProbe {
            clock: runtime.clock(),
            advance: Duration::from_millis(300),
            calls: 0,
        };

        let error = run_presence_wait(polling_budget, &running, &mut probe, &mut runtime)
            .expect_err("a late successful observation cannot cross the deadline");

        assert!(matches!(
            error,
            NanoDevicePresenceWaitError::TimedOut {
                polling_budget: actual,
                attempts: 1,
                last_observation,
            } if actual == polling_budget
                && last_observation
                    .is_some_and(NanoDevicePresenceSnapshot::all_ready_for_acquisition_attempt)
        ));
        assert_eq!(probe.calls, 1);
        assert!(runtime.sleeps.is_empty());
    }

    #[test]
    fn presence_wait_is_signal_aware_before_any_enumeration() {
        let running = AtomicBool::new(false);
        let mut probe = ScriptedPresenceProbe::new([]);
        let mut runtime = FakePresenceWaitRuntime::new(Instant::now());

        assert!(matches!(
            run_presence_wait(Duration::from_secs(1), &running, &mut probe, &mut runtime,),
            Err(NanoDevicePresenceWaitError::Interrupted)
        ));
        assert_eq!(probe.calls, 0);
        assert!(runtime.sleeps.is_empty());
    }

    #[test]
    fn presence_wait_shutdown_during_complete_observation_cannot_report_success() {
        let running = AtomicBool::new(true);
        let mut probe =
            ScriptedPresenceProbe::new([Ok(presence(true, true, true, true))]).interrupt(&running);
        let mut runtime = FakePresenceWaitRuntime::new(Instant::now());

        assert!(matches!(
            run_presence_wait(Duration::from_secs(1), &running, &mut probe, &mut runtime,),
            Err(NanoDevicePresenceWaitError::Interrupted)
        ));
        assert_eq!(probe.calls, 1);
        assert!(runtime.sleeps.is_empty());
    }

    #[test]
    fn presence_wait_shutdown_during_failed_observation_wins_over_probe_error() {
        let running = AtomicBool::new(true);
        let mut probe = ScriptedPresenceProbe::new([Err(
            NanoDevicePresenceProbeError::SerialTargetIsNotCharacterDevice {
                role: NanoSerialPresenceRole::Head,
                path: PathBuf::from("/dev/serial/by-id/head"),
            },
        )])
        .interrupt(&running);
        let mut runtime = FakePresenceWaitRuntime::new(Instant::now());

        assert!(matches!(
            run_presence_wait(Duration::from_secs(1), &running, &mut probe, &mut runtime,),
            Err(NanoDevicePresenceWaitError::Interrupted)
        ));
        assert_eq!(probe.calls, 1);
        assert!(runtime.sleeps.is_empty());
    }

    #[test]
    fn presence_probe_failure_is_not_retried() {
        let running = AtomicBool::new(true);
        let expected_path = PathBuf::from("/dev/serial/by-id/head");
        let mut probe = ScriptedPresenceProbe::new([Err(
            NanoDevicePresenceProbeError::SerialTargetIsNotCharacterDevice {
                role: NanoSerialPresenceRole::Head,
                path: expected_path.clone(),
            },
        )]);
        let mut runtime = FakePresenceWaitRuntime::new(Instant::now());

        let error = run_presence_wait(Duration::from_secs(1), &running, &mut probe, &mut runtime)
            .expect_err("a malformed exact serial target is terminal");

        assert!(matches!(
            error,
            NanoDevicePresenceWaitError::Probe(
                NanoDevicePresenceProbeError::SerialTargetIsNotCharacterDevice {
                    role: NanoSerialPresenceRole::Head,
                    path,
                }
            ) if path == expected_path
        ));
        assert_eq!(probe.calls, 1);
        assert!(runtime.sleeps.is_empty());
    }

    #[test]
    fn oak_presence_accepts_available_and_in_use_but_not_transitional_states() {
        let devices = [
            DeviceInfo {
                device_id: "different".to_owned(),
                name: "unrelated".to_owned(),
                state: oak_sys::DeviceState::Available,
            },
            DeviceInfo {
                device_id: "available".to_owned(),
                name: "exact-and-ready".to_owned(),
                state: oak_sys::DeviceState::Available,
            },
            DeviceInfo {
                device_id: "in-use".to_owned(),
                name: "exact-but-owned".to_owned(),
                state: oak_sys::DeviceState::InUse,
            },
            DeviceInfo {
                device_id: "bootloader".to_owned(),
                name: "exact-but-transitional".to_owned(),
                state: oak_sys::DeviceState::Bootloader,
            },
            DeviceInfo {
                device_id: "unknown".to_owned(),
                name: "exact-but-unclassified".to_owned(),
                state: oak_sys::DeviceState::Unknown,
            },
        ];

        assert_eq!(
            exact_oak_mxid_presence(&devices, "available"),
            NanoOakPresence::Available
        );
        assert_eq!(
            exact_oak_mxid_presence(&devices, "in-use"),
            NanoOakPresence::InUse
        );
        assert_eq!(
            exact_oak_mxid_presence(&devices, "bootloader"),
            NanoOakPresence::Bootloader
        );
        assert_eq!(
            exact_oak_mxid_presence(&devices, "unknown"),
            NanoOakPresence::Unknown
        );
        assert_eq!(
            exact_oak_mxid_presence(&devices, "missing"),
            NanoOakPresence::Missing
        );
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

    #[test]
    fn mpc_pwm_ranges_must_fit_the_exact_admitted_controller_cap() {
        assert!(bind_one_mpc_pwm_range(WheelSide::Left, -30, 30, 30).is_ok());
        for (wheel, minimum, maximum) in [(WheelSide::Left, -31, 30), (WheelSide::Right, -30, 31)] {
            assert!(matches!(
                bind_one_mpc_pwm_range(wheel, minimum, maximum, 30),
                Err(
                    NanoBootstrapPrimaryError::ActuationMpcPwmOutsideControllerEnvelope {
                        wheel: actual_wheel,
                        configured_min_percent,
                        configured_max_percent,
                        controller_max_abs_percent: 30,
                    }
                ) if actual_wheel == wheel
                    && configured_min_percent == minimum
                    && configured_max_percent == maximum
            ));
        }
    }

    #[test]
    fn navigation_cadence_strictly_exceeds_controller_interval_plus_margin() {
        let controller_minimum = Duration::from_millis(10);
        let scheduling_margin = Duration::from_millis(5);
        assert!(
            bind_navigation_cadence_to_controller(
                Duration::from_nanos(15_000_001),
                controller_minimum,
                scheduling_margin,
            )
            .is_ok()
        );
        for control_period in [Duration::from_millis(15), Duration::from_nanos(14_999_999)] {
            assert!(matches!(
                bind_navigation_cadence_to_controller(
                    control_period,
                    controller_minimum,
                    scheduling_margin,
                ),
                Err(
                    NanoBootstrapPrimaryError::ActuationControlPeriodHasNoControllerRateMargin {
                        control_period: actual,
                        required_exclusive_lower_bound,
                        ..
                    }
                ) if actual == control_period
                    && required_exclusive_lower_bound == Duration::from_millis(15)
            ));
        }
    }

    #[test]
    fn bootstrap_face_shutdown_never_calls_a_detached_join_completed() {
        for terminal_fault in [
            None,
            Some(NanoAccessoryTerminalFault::RgbIngressDisconnected),
            Some(NanoAccessoryTerminalFault::FacePerception(
                NanoFacePerceptionRuntimeError::RgbIngressDisconnected,
            )),
        ] {
            let evidence = NanoFacePerceptionShutdownEvidence::Join(
                NanoFacePerceptionJoinEvidence::DetachedAfterTimeout {
                    configured_timeout: Duration::from_secs(2),
                    active_join_budget: Duration::from_millis(375),
                },
            );
            assert_eq!(
                classify_bootstrap_face_shutdown(&evidence, terminal_fault.as_ref()),
                NanoBootstrapFaceShutdownDispositionKind::Uncertain
            );
        }
    }

    #[test]
    fn bootstrap_face_shutdown_preserves_coordinated_first_fault_semantics() {
        let disabled = NanoFacePerceptionShutdownEvidence::Disabled;
        assert_eq!(
            classify_bootstrap_face_shutdown(&disabled, None),
            NanoBootstrapFaceShutdownDispositionKind::Completed
        );

        let coordinated = NanoFacePerceptionShutdownEvidence::Join(
            NanoFacePerceptionJoinEvidence::Joined(NanoFacePerceptionThreadExit::Shutdown),
        );
        assert_eq!(
            classify_bootstrap_face_shutdown(&coordinated, None),
            NanoBootstrapFaceShutdownDispositionKind::Completed
        );

        let published = NanoFacePerceptionShutdownEvidence::Join(
            NanoFacePerceptionJoinEvidence::Joined(NanoFacePerceptionThreadExit::RuntimeFault {
                source: NanoFacePerceptionRuntimeError::RgbIngressDisconnected,
                published_to_accessory: true,
            }),
        );
        let published_fault = NanoAccessoryTerminalFault::FacePerception(
            NanoFacePerceptionRuntimeError::RgbIngressDisconnected,
        );
        assert_eq!(
            classify_bootstrap_face_shutdown(&published, Some(&published_fault)),
            NanoBootstrapFaceShutdownDispositionKind::Completed
        );

        for follower in [
            NanoFacePerceptionThreadExit::AccessoryFaultPendingPublication,
            NanoFacePerceptionThreadExit::AccessoryFaultLatched,
        ] {
            let evidence = NanoFacePerceptionShutdownEvidence::Join(
                NanoFacePerceptionJoinEvidence::Joined(follower),
            );
            let retained_first_fault = NanoAccessoryTerminalFault::RgbIngressDisconnected;
            assert_eq!(
                classify_bootstrap_face_shutdown(&evidence, Some(&retained_first_fault)),
                NanoBootstrapFaceShutdownDispositionKind::Completed
            );
        }
    }

    #[test]
    fn bootstrap_face_shutdown_separates_inconsistent_evidence_from_uncertainty() {
        let disabled = NanoFacePerceptionShutdownEvidence::Disabled;
        let impossible_face_fault = NanoAccessoryTerminalFault::FacePerception(
            NanoFacePerceptionRuntimeError::RgbIngressDisconnected,
        );
        assert_eq!(
            classify_bootstrap_face_shutdown(&disabled, Some(&impossible_face_fault)),
            NanoBootstrapFaceShutdownDispositionKind::Unexpected
        );

        let unpublished = NanoFacePerceptionShutdownEvidence::Join(
            NanoFacePerceptionJoinEvidence::Joined(NanoFacePerceptionThreadExit::RuntimeFault {
                source: NanoFacePerceptionRuntimeError::RgbIngressDisconnected,
                published_to_accessory: false,
            }),
        );
        assert_eq!(
            classify_bootstrap_face_shutdown(&unpublished, None),
            NanoBootstrapFaceShutdownDispositionKind::Unexpected
        );

        let impossible_shutdown = NanoFacePerceptionShutdownEvidence::Join(
            NanoFacePerceptionJoinEvidence::Joined(NanoFacePerceptionThreadExit::Shutdown),
        );
        let retained_first_fault = NanoAccessoryTerminalFault::RgbIngressDisconnected;
        assert_eq!(
            classify_bootstrap_face_shutdown(&impossible_shutdown, Some(&retained_first_fault)),
            NanoBootstrapFaceShutdownDispositionKind::Unexpected
        );
    }

    #[test]
    fn bootstrap_shutdown_does_not_retain_a_terminal_fault_already_owned_by_primary() {
        let primary = NanoAccessoryTerminalFault::FacePerception(
            NanoFacePerceptionRuntimeError::RgbIngressDisconnected,
        );
        let shutdown = NanoAccessoryTerminalFault::FacePerception(
            NanoFacePerceptionRuntimeError::RgbIngressDisconnected,
        );
        let (disposition, consistent) =
            reconcile_accessory_terminal_fault(Some(&primary), Some(shutdown));

        assert!(consistent);
        assert!(matches!(
            disposition,
            NanoBootstrapAccessoryTerminalFaultDisposition::AlreadyOwnedByPrimary
        ));
        assert_eq!(
            disposition.to_string(),
            "accessory first terminal fault is already owned by the primary bootstrap error"
        );
    }

    #[test]
    fn bootstrap_shutdown_retains_distinct_or_missing_terminal_evidence_as_inconsistent() {
        let primary = NanoAccessoryTerminalFault::FacePerception(
            NanoFacePerceptionRuntimeError::RgbIngressDisconnected,
        );
        let conflicting_shutdown = NanoAccessoryTerminalFault::FacePerception(
            NanoFacePerceptionRuntimeError::RgbChannelPoisoned,
        );
        let (conflict, conflict_is_consistent) =
            reconcile_accessory_terminal_fault(Some(&primary), Some(conflicting_shutdown));
        assert!(!conflict_is_consistent);
        assert!(matches!(
            conflict,
            NanoBootstrapAccessoryTerminalFaultDisposition::ConflictsWithPrimary(
                NanoAccessoryTerminalFault::FacePerception(
                    NanoFacePerceptionRuntimeError::RgbChannelPoisoned
                )
            )
        ));

        let (missing, missing_is_consistent) =
            reconcile_accessory_terminal_fault(Some(&primary), None);
        assert!(!missing_is_consistent);
        assert!(matches!(
            missing,
            NanoBootstrapAccessoryTerminalFaultDisposition::
                MissingFromShutdownAfterPrimaryObservation
        ));

        let shutdown_only = NanoAccessoryTerminalFault::RgbIngressDisconnected;
        let (retained, retained_is_consistent) =
            reconcile_accessory_terminal_fault(None, Some(shutdown_only));
        assert!(retained_is_consistent);
        assert!(matches!(
            retained,
            NanoBootstrapAccessoryTerminalFaultDisposition::RetainedByShutdown(
                NanoAccessoryTerminalFault::RgbIngressDisconnected
            )
        ));
    }
}
