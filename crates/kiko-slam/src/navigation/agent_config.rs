//! Strict, bounded policy component for the Nano agent runtime.
//!
//! This document is deliberately separate from the navigation and physical
//! actuation manifests and is not a complete cold-boot configuration. It
//! selects process-owned transports and policies, but it does not prove that a
//! path exists, that an inventory identity matches, that an artifact has the
//! expected digest, or that physical motion is safe. A startup configuration
//! must bind this component by exact content identity and perform those
//! explicit admissions.
//!
//! Every optional subsystem is represented by an explicit tagged state. A
//! missing JSON field is therefore an error rather than an implicit default.
//! RGB scene motion can own only KEP2 eyes and the head policy is fixed to a
//! natural hold at redundantly observed present positions.

use std::fmt;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::time::Duration;

#[cfg(test)]
use kiko_device_inventory::MAX_PLANT_ARTIFACTS;
use kiko_device_inventory::{
    ArtifactFileBindingInput, ArtifactFileBindingParseError, ArtifactFileBindingSet, ArtifactKind,
    DeviceInventoryManifestV1, MAX_ARTIFACT_ROOT_PATH_BYTES, MAX_MANIFEST_PATH_BYTES,
};
use kiko_expression_core::{NonZeroDuration, PositiveUnitAmount, TimeError, UnitAmount};
use kiko_expression_runtime::{
    CameraToHeadGazeExtrinsics, CameraToHeadGazeExtrinsicsInput, EyeRenderStyle,
    GazeExtrinsicsParseError, MotionThresholds, SamplingGeometry, SamplingGeometryError,
    SceneMotionConfig, SceneMotionConfigError,
};
use kiko_eye_protocol::PROTOCOL_VERSION as EYE_PROTOCOL_VERSION;
use kiko_eye_runtime::{
    ConfigParseError as EyeConfigParseError, StaticEyeRuntimeConfig, StaticEyeRuntimeConfigInput,
};
use kiko_head_protocol::{
    ADAPTER_DTR_ASSERTED, ADAPTER_RTS_ASSERTED, BUS_BAUD_RATE_BPS, HeadJoint,
};
use kiko_head_runtime::{
    ConfigParseError as HeadConfigParseError, HeadRuntimeConfig, HeadRuntimeConfigInput,
    PhysicalTorqueEnableConsent,
};
use kiko_supervisor_core::{
    AuthorityDuration, SupervisorConfig, SupervisorConfigError, TimeValueError,
};
use serde::Deserialize;

use super::mpc::{BoundedId, PlantModelV1};
use super::{
    AgentControlRuntimeQueueCapacity, AgentControlRuntimeQueueCapacityError,
    AgentControlSocketConfig, AgentControlSocketPath, AgentControlSocketPathError,
    AgentControlSocketTimeoutError, AgentControlSocketTimeouts, MANUAL_DRIVE_CONFIG_V1,
    ManualDriveConfigParseError, ManualDriveConfigV1, ManualDriveConfigV1Dto,
};

/// The only supported Nano-agent policy-component schema.
pub const NANO_AGENT_POLICY_CONFIG_V1: u32 = 1;

/// Nested camera-to-head gaze-geometry schema retained independently of the
/// surrounding Nano policy schema.
pub const NANO_RGB_GAZE_GEOMETRY_V1: u32 = 1;

/// Hard input bound checked before JSON can allocate caller-sized values.
pub const MAX_NANO_AGENT_POLICY_CONFIG_JSON_BYTES: usize = 64 * 1_024;

/// Maximum UTF-8 bytes in map and dataset paths not bounded downstream.
pub const MAX_NANO_AGENT_DATA_PATH_BYTES: usize = 1_024;

/// Maximum supervisor or per-mode authority lease admitted by this boundary.
pub const MAX_NANO_AGENT_AUTHORITY_LEASE_MS: u64 = 60_000;

/// Maximum admitted manual command/deadman window.
///
/// The live owner ticks this independently of the controller's shorter motion
/// lease. A finite upper bound also prevents a policy typo from turning the
/// manual stream into an effectively unbounded command hold.
pub const MAX_NANO_AGENT_MANUAL_WINDOW_MS: u64 = 60_000;

/// Maximum freshness window for one RGB observation.
pub const MAX_NANO_AGENT_RGB_FRAME_FRESHNESS_MS: u64 = 5_000;

/// Maximum total wall-clock resource budget for one exploration request.
pub const MAX_NANO_AGENT_EXPLORE_RUNTIME_MS: u64 = 24 * 60 * 60 * 1_000;

/// Maximum number of frontier goals admitted for one exploration request.
pub const MAX_NANO_AGENT_EXPLORE_GOALS: u32 = 10_000;

/// Maximum absolute map coordinate admitted for an exploration boundary.
pub const MAX_NANO_AGENT_ABS_EXPLORE_BOUNDARY_M: f64 = 10_000.0;

/// A fully parsed runtime policy component. Construction is possible only
/// through [`Self::parse_json`].
#[derive(Clone, Debug, PartialEq)]
pub struct NanoAgentPolicyConfigV1 {
    control: NanoAgentControlConfig,
    inventory: NanoAgentInventoryConfig,
    map_persistence: NanoMapPersistenceConfig,
    eye: ParsedNanoEyePolicy,
    head: ParsedNanoHeadPolicy,
    rgb_expression: NanoRgbExpressionPolicy,
    supervisor: SupervisorConfig,
    live_mode_policy: NanoLiveModePolicy,
}

impl NanoAgentPolicyConfigV1 {
    /// Parse one exact JSON document into runtime-native domain types.
    ///
    /// This operation does not open the referenced manifest, artifacts, map,
    /// dataset, serial devices, or socket parent. Filesystem and device
    /// admission occur later against these exact retained identities.
    pub fn parse_json(json: &[u8]) -> Result<Self, NanoAgentPolicyConfigParseError> {
        if json.len() > MAX_NANO_AGENT_POLICY_CONFIG_JSON_BYTES {
            return Err(NanoAgentPolicyConfigParseError::InputTooLarge {
                actual_bytes: json.len(),
                maximum_bytes: MAX_NANO_AGENT_POLICY_CONFIG_JSON_BYTES,
            });
        }

        let mut deserializer = serde_json::Deserializer::from_slice(json);
        let dto = NanoAgentPolicyConfigV1Dto::deserialize(&mut deserializer)
            .map_err(NanoAgentPolicyConfigParseError::JsonDecode)?;
        deserializer
            .end()
            .map_err(NanoAgentPolicyConfigParseError::JsonTrailingData)?;
        if dto.schema_version != NANO_AGENT_POLICY_CONFIG_V1 {
            return Err(NanoAgentPolicyConfigParseError::UnsupportedSchemaVersion {
                actual: dto.schema_version,
                supported: NANO_AGENT_POLICY_CONFIG_V1,
            });
        }

        let control = parse_control(dto.control)?;
        let inventory = parse_inventory(dto.inventory)?;
        let map_persistence = parse_map_persistence(dto.map_persistence)?;
        let supervisor = parse_supervisor(dto.supervisor)?;
        let eye = parse_eye(dto.eye)?;
        let head = parse_head(dto.head)?;

        if let (ParsedNanoEyePolicy::Kep2(eye), ParsedNanoHeadPolicy::NaturalHold(head)) =
            (&eye, &head)
            && eye.device().path() == head.runtime().device().path()
        {
            return Err(NanoAgentPolicyConfigParseError::DuplicateAccessorySerialPath);
        }

        let rgb_expression = parse_rgb_expression(dto.rgb_expression, &eye)?;
        let live_mode_policy = parse_live_mode_policy(dto.live_mode_policy, supervisor)?;

        Ok(Self {
            control,
            inventory,
            map_persistence,
            eye,
            head,
            rgb_expression,
            supervisor,
            live_mode_policy,
        })
    }

    pub const fn control(&self) -> &NanoAgentControlConfig {
        &self.control
    }

    pub const fn inventory(&self) -> &NanoAgentInventoryConfig {
        &self.inventory
    }

    pub const fn map_persistence(&self) -> &NanoMapPersistenceConfig {
        &self.map_persistence
    }

    /// Whether this parsed document requests the eye actor. Actor
    /// configuration is intentionally unavailable before manifest binding.
    pub const fn eye_enabled(&self) -> bool {
        matches!(self.eye, ParsedNanoEyePolicy::Kep2(_))
    }

    /// Whether this parsed document requests natural head hold. Actor
    /// configuration is intentionally unavailable before manifest binding.
    pub const fn head_enabled(&self) -> bool {
        matches!(self.head, ParsedNanoHeadPolicy::NaturalHold(_))
    }

    pub const fn rgb_expression(&self) -> &NanoRgbExpressionPolicy {
        &self.rgb_expression
    }

    pub const fn supervisor(&self) -> SupervisorConfig {
        self.supervisor
    }

    pub const fn live_mode_policy(&self) -> &NanoLiveModePolicy {
        &self.live_mode_policy
    }

    /// Consume the parsed document and bind both accessory transports to the
    /// exact already-loaded expected manifest before either actor can open a
    /// serial device.
    pub fn bind_accessories_to_manifest(
        self,
        manifest: &DeviceInventoryManifestV1,
    ) -> Result<ManifestBoundNanoAgentPolicyConfigV1, NanoAccessoryManifestBindingError> {
        let eye = bind_eye_to_manifest(self.eye, manifest)?;
        let head = bind_head_to_manifest(self.head, manifest)?;
        Ok(ManifestBoundNanoAgentPolicyConfigV1 {
            control: self.control,
            inventory: self.inventory,
            map_persistence: self.map_persistence,
            eye,
            head,
            rgb_expression: self.rgb_expression,
            supervisor: self.supervisor,
            live_mode_policy: self.live_mode_policy,
        })
    }
}

/// Runtime policy whose accessory choices exactly match one parsed expected
/// manifest. Only this type exposes actor-owned configurations.
#[derive(Clone, Debug, PartialEq)]
pub struct ManifestBoundNanoAgentPolicyConfigV1 {
    control: NanoAgentControlConfig,
    inventory: NanoAgentInventoryConfig,
    map_persistence: NanoMapPersistenceConfig,
    eye: NanoManifestBoundEyePolicy,
    head: NanoManifestBoundHeadPolicy,
    rgb_expression: NanoRgbExpressionPolicy,
    supervisor: SupervisorConfig,
    live_mode_policy: NanoLiveModePolicy,
}

impl ManifestBoundNanoAgentPolicyConfigV1 {
    pub const fn control(&self) -> &NanoAgentControlConfig {
        &self.control
    }

    pub const fn inventory(&self) -> &NanoAgentInventoryConfig {
        &self.inventory
    }

    pub const fn map_persistence(&self) -> &NanoMapPersistenceConfig {
        &self.map_persistence
    }

    pub const fn eye(&self) -> &NanoManifestBoundEyePolicy {
        &self.eye
    }

    pub const fn head(&self) -> &NanoManifestBoundHeadPolicy {
        &self.head
    }

    pub const fn rgb_expression(&self) -> &NanoRgbExpressionPolicy {
        &self.rgb_expression
    }

    pub const fn supervisor(&self) -> SupervisorConfig {
        self.supervisor
    }

    pub const fn live_mode_policy(&self) -> &NanoLiveModePolicy {
        &self.live_mode_policy
    }

    pub fn into_parts(self) -> NanoAgentPolicyPartsV1 {
        NanoAgentPolicyPartsV1 {
            control: self.control,
            inventory: self.inventory,
            map_persistence: self.map_persistence,
            eye: self.eye,
            head: self.head,
            rgb_expression: self.rgb_expression,
            supervisor: self.supervisor,
            live_mode_policy: self.live_mode_policy,
        }
    }
}

/// Owned policy parts returned only after manifest accessory binding.
#[derive(Clone, Debug, PartialEq)]
pub struct NanoAgentPolicyPartsV1 {
    pub control: NanoAgentControlConfig,
    pub inventory: NanoAgentInventoryConfig,
    pub map_persistence: NanoMapPersistenceConfig,
    pub eye: NanoManifestBoundEyePolicy,
    pub head: NanoManifestBoundHeadPolicy,
    pub rgb_expression: NanoRgbExpressionPolicy,
    pub supervisor: SupervisorConfig,
    pub live_mode_policy: NanoLiveModePolicy,
}

/// Parsed local-control transport and bounded runtime queue.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NanoAgentControlConfig {
    socket: AgentControlSocketConfig,
    runtime_queue_capacity: AgentControlRuntimeQueueCapacity,
}

impl NanoAgentControlConfig {
    pub const fn socket(&self) -> &AgentControlSocketConfig {
        &self.socket
    }

    pub const fn runtime_queue_capacity(&self) -> AgentControlRuntimeQueueCapacity {
        self.runtime_queue_capacity
    }

    pub fn into_parts(self) -> (AgentControlSocketConfig, AgentControlRuntimeQueueCapacity) {
        (self.socket, self.runtime_queue_capacity)
    }
}

/// Lexically canonical absolute Unix path retained without filesystem claims.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NanoConfiguredAbsolutePath(PathBuf);

impl NanoConfiguredAbsolutePath {
    pub fn as_path(&self) -> &Path {
        &self.0
    }

    pub fn into_path_buf(self) -> PathBuf {
        self.0
    }
}

/// Exact manifest location, artifact root, and complete bounded binding set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NanoAgentInventoryConfig {
    manifest_path: NanoConfiguredAbsolutePath,
    artifact_root_path: NanoConfiguredAbsolutePath,
    artifact_bindings: ArtifactFileBindingSet,
}

impl NanoAgentInventoryConfig {
    pub const fn manifest_path(&self) -> &NanoConfiguredAbsolutePath {
        &self.manifest_path
    }

    pub const fn artifact_root_path(&self) -> &NanoConfiguredAbsolutePath {
        &self.artifact_root_path
    }

    pub const fn artifact_bindings(&self) -> &ArtifactFileBindingSet {
        &self.artifact_bindings
    }

    /// Consume the already-parsed binding set for manifest membership checks
    /// and no-follow hashing beneath the exact artifact root.
    pub fn into_parts(self) -> NanoAgentInventoryParts {
        NanoAgentInventoryParts {
            manifest_path: self.manifest_path,
            artifact_root_path: self.artifact_root_path,
            artifact_bindings: self.artifact_bindings,
        }
    }
}

/// Owned inventory inputs ready for manifest loading and content hashing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NanoAgentInventoryParts {
    pub manifest_path: NanoConfiguredAbsolutePath,
    pub artifact_root_path: NanoConfiguredAbsolutePath,
    pub artifact_bindings: ArtifactFileBindingSet,
}

/// Persistence destination plus an optional dataset-bound replay request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NanoMapPersistenceConfig {
    save_snapshot_path: NanoConfiguredAbsolutePath,
    warm_start: NanoMapWarmStart,
}

impl NanoMapPersistenceConfig {
    pub const fn save_snapshot_path(&self) -> &NanoConfiguredAbsolutePath {
        &self.save_snapshot_path
    }

    pub const fn warm_start(&self) -> &NanoMapWarmStart {
        &self.warm_start
    }
}

/// Warm-start policy with no occupancy-only relocalization state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NanoMapWarmStart {
    None,
    /// Replay the bound SLAM dataset before treating the persisted occupancy
    /// snapshot as continued-map context. Parsing these paths is not evidence
    /// that replay succeeded or that live localization was recovered.
    DatasetReplay {
        occupancy_snapshot_path: NanoConfiguredAbsolutePath,
        slam_dataset_directory_path: NanoConfiguredAbsolutePath,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum ParsedNanoEyePolicy {
    Disabled,
    Kep2(StaticEyeRuntimeConfig),
}

/// Manifest-bound KEP2 eye selection. Disabled means no eye transport is
/// opened; the runtime value is exposed only after exact manifest agreement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NanoManifestBoundEyePolicy {
    Disabled,
    Kep2(StaticEyeRuntimeConfig),
}

impl NanoManifestBoundEyePolicy {
    /// Return only restart-safe static policy. The caller must generate fresh
    /// one-shot KEP2 session material before an actor-ready config exists.
    pub const fn static_runtime(&self) -> Option<&StaticEyeRuntimeConfig> {
        match self {
            Self::Disabled => None,
            Self::Kep2(runtime) => Some(runtime),
        }
    }
}

/// Head actor configuration and the explicit consent required to energise it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NanoNaturalHeadHoldConfig {
    runtime: HeadRuntimeConfig,
    torque_consent: PhysicalTorqueEnableConsent,
}

impl NanoNaturalHeadHoldConfig {
    pub const fn runtime(&self) -> &HeadRuntimeConfig {
        &self.runtime
    }

    pub const fn torque_consent(&self) -> PhysicalTorqueEnableConsent {
        self.torque_consent
    }

    pub fn into_parts(self) -> (HeadRuntimeConfig, PhysicalTorqueEnableConsent) {
        (self.runtime, self.torque_consent)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum ParsedNanoHeadPolicy {
    Disabled,
    NaturalHold(NanoNaturalHeadHoldConfig),
}

/// Manifest-bound head selection. There is no expressive-offset variant and
/// the natural-hold actor value is exposed only after exact manifest
/// agreement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NanoManifestBoundHeadPolicy {
    Disabled,
    NaturalHold(NanoNaturalHeadHoldConfig),
}

impl NanoManifestBoundHeadPolicy {
    pub const fn natural_hold(&self) -> Option<&NanoNaturalHeadHoldConfig> {
        match self {
            Self::Disabled => None,
            Self::NaturalHold(config) => Some(config),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoAccessoryKind {
    Eye,
    Head,
}

/// Exact disagreement between parsed runtime accessory selection and the
/// already-parsed expected device manifest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NanoAccessoryManifestBindingError {
    PresenceMismatch {
        accessory: NanoAccessoryKind,
        runtime_enabled: bool,
        manifest_expected: bool,
    },
    SerialPathMismatch {
        accessory: NanoAccessoryKind,
        runtime_path: Box<str>,
        manifest_path: Box<str>,
    },
    EyeProtocolVersionMismatch {
        manifest_version: u8,
        runtime_version: u8,
    },
    EyeDeviceUidMismatch {
        runtime_uid: [u8; 16],
        manifest_uid: [u8; 16],
    },
    EyeFirmwareBuildMismatch {
        runtime_build_id: [u8; 32],
        manifest_build_id: [u8; 32],
    },
    EyeCapabilitiesMismatch {
        runtime_bits: u32,
        manifest_bits: u32,
    },
    HeadElectricalContractMismatch {
        manifest_baud_rate_bps: u32,
        runtime_baud_rate_bps: u32,
        manifest_dtr_asserted: bool,
        runtime_dtr_asserted: bool,
        manifest_rts_asserted: bool,
        runtime_rts_asserted: bool,
    },
    HeadServoContractMismatch {
        manifest_servo_ids: [u8; 4],
        runtime_servo_ids: [u8; 4],
    },
}

impl fmt::Display for NanoAccessoryManifestBindingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "Nano accessory configuration does not match expected manifest: {self:?}"
        )
    }
}

impl std::error::Error for NanoAccessoryManifestBindingError {}

fn bind_eye_to_manifest(
    policy: ParsedNanoEyePolicy,
    manifest: &DeviceInventoryManifestV1,
) -> Result<NanoManifestBoundEyePolicy, NanoAccessoryManifestBindingError> {
    let manifest_eye = manifest.eye();
    match (policy, manifest_eye) {
        (ParsedNanoEyePolicy::Disabled, None) => Ok(NanoManifestBoundEyePolicy::Disabled),
        (ParsedNanoEyePolicy::Disabled, Some(_)) => {
            Err(NanoAccessoryManifestBindingError::PresenceMismatch {
                accessory: NanoAccessoryKind::Eye,
                runtime_enabled: false,
                manifest_expected: true,
            })
        }
        (ParsedNanoEyePolicy::Kep2(_), None) => {
            Err(NanoAccessoryManifestBindingError::PresenceMismatch {
                accessory: NanoAccessoryKind::Eye,
                runtime_enabled: true,
                manifest_expected: false,
            })
        }
        (ParsedNanoEyePolicy::Kep2(runtime), Some(expected)) => {
            if runtime.device().path() != expected.serial_path().as_str() {
                return Err(NanoAccessoryManifestBindingError::SerialPathMismatch {
                    accessory: NanoAccessoryKind::Eye,
                    runtime_path: runtime.device().path().into(),
                    manifest_path: expected.serial_path().as_str().into(),
                });
            }
            if expected.protocol_version() != EYE_PROTOCOL_VERSION {
                return Err(
                    NanoAccessoryManifestBindingError::EyeProtocolVersionMismatch {
                        manifest_version: expected.protocol_version(),
                        runtime_version: EYE_PROTOCOL_VERSION,
                    },
                );
            }

            let runtime_identity = runtime.expected_identity();
            let runtime_uid = *runtime_identity.device_uid().as_bytes();
            let manifest_uid = *expected.device_uid().as_bytes();
            if runtime_uid != manifest_uid {
                return Err(NanoAccessoryManifestBindingError::EyeDeviceUidMismatch {
                    runtime_uid,
                    manifest_uid,
                });
            }
            let runtime_build_id = *runtime_identity.firmware_build_id().as_bytes();
            let manifest_build_id = *expected.firmware_build_id().as_bytes();
            if runtime_build_id != manifest_build_id {
                return Err(
                    NanoAccessoryManifestBindingError::EyeFirmwareBuildMismatch {
                        runtime_build_id,
                        manifest_build_id,
                    },
                );
            }
            let runtime_bits = runtime_identity.capabilities().bits();
            let manifest_bits = expected.capabilities().bits();
            if runtime_bits != manifest_bits {
                return Err(NanoAccessoryManifestBindingError::EyeCapabilitiesMismatch {
                    runtime_bits,
                    manifest_bits,
                });
            }
            Ok(NanoManifestBoundEyePolicy::Kep2(runtime))
        }
    }
}

fn bind_head_to_manifest(
    policy: ParsedNanoHeadPolicy,
    manifest: &DeviceInventoryManifestV1,
) -> Result<NanoManifestBoundHeadPolicy, NanoAccessoryManifestBindingError> {
    let manifest_head = manifest.head();
    match (policy, manifest_head) {
        (ParsedNanoHeadPolicy::Disabled, None) => Ok(NanoManifestBoundHeadPolicy::Disabled),
        (ParsedNanoHeadPolicy::Disabled, Some(_)) => {
            Err(NanoAccessoryManifestBindingError::PresenceMismatch {
                accessory: NanoAccessoryKind::Head,
                runtime_enabled: false,
                manifest_expected: true,
            })
        }
        (ParsedNanoHeadPolicy::NaturalHold(_), None) => {
            Err(NanoAccessoryManifestBindingError::PresenceMismatch {
                accessory: NanoAccessoryKind::Head,
                runtime_enabled: true,
                manifest_expected: false,
            })
        }
        (ParsedNanoHeadPolicy::NaturalHold(runtime), Some(expected)) => {
            if runtime.runtime().device().path() != expected.serial_path().as_str() {
                return Err(NanoAccessoryManifestBindingError::SerialPathMismatch {
                    accessory: NanoAccessoryKind::Head,
                    runtime_path: runtime.runtime().device().path().into(),
                    manifest_path: expected.serial_path().as_str().into(),
                });
            }
            if expected.baud_rate_bps() != BUS_BAUD_RATE_BPS
                || expected.dtr_asserted() != ADAPTER_DTR_ASSERTED
                || expected.rts_asserted() != ADAPTER_RTS_ASSERTED
            {
                return Err(
                    NanoAccessoryManifestBindingError::HeadElectricalContractMismatch {
                        manifest_baud_rate_bps: expected.baud_rate_bps(),
                        runtime_baud_rate_bps: BUS_BAUD_RATE_BPS,
                        manifest_dtr_asserted: expected.dtr_asserted(),
                        runtime_dtr_asserted: ADAPTER_DTR_ASSERTED,
                        manifest_rts_asserted: expected.rts_asserted(),
                        runtime_rts_asserted: ADAPTER_RTS_ASSERTED,
                    },
                );
            }
            let manifest_servo_ids = expected.servo_ids().map(|servo_id| servo_id.get());
            let runtime_servo_ids = HeadJoint::ALL.map(|joint| joint.servo_id().get());
            if manifest_servo_ids != runtime_servo_ids {
                return Err(
                    NanoAccessoryManifestBindingError::HeadServoContractMismatch {
                        manifest_servo_ids,
                        runtime_servo_ids,
                    },
                );
            }
            Ok(NanoManifestBoundHeadPolicy::NaturalHold(runtime))
        }
    }
}

/// Native RGB scene-motion configuration plus an explicit source deadline and
/// fixed render style.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NanoRgbExpressionConfig {
    scene_motion: SceneMotionConfig,
    frame_freshness: NonZeroDuration,
    render_style: EyeRenderStyle,
    gaze_geometry: Option<CameraToHeadGazeExtrinsics>,
}

impl NanoRgbExpressionConfig {
    pub const fn scene_motion(self) -> SceneMotionConfig {
        self.scene_motion
    }

    pub const fn frame_freshness(self) -> NonZeroDuration {
        self.frame_freshness
    }

    pub const fn render_style(self) -> EyeRenderStyle {
        self.render_style
    }

    /// Parsed camera-to-neutral-head geometry, or `None` when this schema-v1
    /// policy deliberately leaves RGB head-gaze projection unavailable.
    pub const fn gaze_geometry(self) -> Option<CameraToHeadGazeExtrinsics> {
        self.gaze_geometry
    }
}

/// RGB reactions are either absent or explicitly routed to an enabled eye.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoRgbExpressionPolicy {
    Disabled,
    SceneMotion(NanoRgbExpressionConfig),
}

impl NanoRgbExpressionPolicy {
    pub const fn scene_motion(self) -> Option<NanoRgbExpressionConfig> {
        match self {
            Self::Disabled => None,
            Self::SceneMotion(config) => Some(config),
        }
    }
}

/// Startup never owns motion authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoAgentStartupMode {
    DisarmedMapOnly,
}

/// Permission for a mode reached only through the local control API.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoMotionModePolicy {
    Disabled,
    ControlApi { authority_lease: AuthorityDuration },
}

/// Manual authority and the only envelope accepted by the manual ingress
/// core. Values are already SI/timing domain types; the live owner must not
/// reparse or reinterpret them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NanoManualControlApiConfig {
    authority_lease: AuthorityDuration,
    drive: ManualDriveConfigV1,
}

impl NanoManualControlApiConfig {
    pub const fn authority_lease(self) -> AuthorityDuration {
        self.authority_lease
    }

    pub const fn drive(self) -> ManualDriveConfigV1 {
        self.drive
    }

    /// Bind the policy envelope to the exact calibrated plant used by MPC.
    pub fn bind_to_plant(
        self,
        plant: PlantModelV1,
    ) -> Result<PlantBoundNanoManualControlApiConfig, NanoManualPlantBindingError> {
        let supported_forward = plant.maximum_symmetric_abs_forward_velocity_mps();
        if self.drive.maximum_abs_forward_velocity_mps() > supported_forward {
            return Err(
                NanoManualPlantBindingError::ForwardVelocityOutsidePlantEnvelope {
                    configured_mps: self.drive.maximum_abs_forward_velocity_mps(),
                    supported_mps: supported_forward,
                },
            );
        }
        let supported_yaw = plant.maximum_abs_yaw_rate_rad_s();
        if self.drive.maximum_abs_yaw_rate_rad_s() > supported_yaw {
            return Err(NanoManualPlantBindingError::YawRateOutsidePlantEnvelope {
                configured_rad_s: self.drive.maximum_abs_yaw_rate_rad_s(),
                supported_rad_s: supported_yaw,
            });
        }
        if !plant.supports_symmetric_body_velocity_box(
            self.drive.maximum_abs_forward_velocity_mps(),
            self.drive.maximum_abs_yaw_rate_rad_s(),
        ) {
            let half_wheelbase_m = 0.5 * plant.wheelbase_m();
            let required_abs_wheel_velocity_mps = half_wheelbase_m.mul_add(
                self.drive.maximum_abs_yaw_rate_rad_s(),
                self.drive.maximum_abs_forward_velocity_mps(),
            );
            return Err(
                NanoManualPlantBindingError::CombinedBodyVelocityOutsidePlantEnvelope {
                    configured_forward_mps: self.drive.maximum_abs_forward_velocity_mps(),
                    configured_yaw_rate_rad_s: self.drive.maximum_abs_yaw_rate_rad_s(),
                    wheelbase_m: plant.wheelbase_m(),
                    required_abs_wheel_velocity_mps,
                    supported_abs_wheel_velocity_mps: plant
                        .maximum_symmetric_abs_wheel_velocity_mps(),
                },
            );
        }
        Ok(PlantBoundNanoManualControlApiConfig {
            authority_lease: self.authority_lease,
            drive: self.drive,
            plant_model_id: plant.model_id(),
            plant_model_version: plant.model_version(),
        })
    }
}

/// Manual policy proven compatible with one exact MPC plant identity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlantBoundNanoManualControlApiConfig {
    authority_lease: AuthorityDuration,
    drive: ManualDriveConfigV1,
    plant_model_id: BoundedId,
    plant_model_version: std::num::NonZeroU32,
}

impl PlantBoundNanoManualControlApiConfig {
    pub const fn authority_lease(self) -> AuthorityDuration {
        self.authority_lease
    }

    pub const fn drive(self) -> ManualDriveConfigV1 {
        self.drive
    }

    pub const fn plant_model_id(self) -> BoundedId {
        self.plant_model_id
    }

    pub const fn plant_model_version(self) -> std::num::NonZeroU32 {
        self.plant_model_version
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NanoManualPlantBindingError {
    ForwardVelocityOutsidePlantEnvelope {
        configured_mps: f64,
        supported_mps: f64,
    },
    YawRateOutsidePlantEnvelope {
        configured_rad_s: f64,
        supported_rad_s: f64,
    },
    CombinedBodyVelocityOutsidePlantEnvelope {
        configured_forward_mps: f64,
        configured_yaw_rate_rad_s: f64,
        wheelbase_m: f64,
        required_abs_wheel_velocity_mps: f64,
        supported_abs_wheel_velocity_mps: f64,
    },
}

impl fmt::Display for NanoManualPlantBindingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "manual policy does not match MPC plant: {self:?}"
        )
    }
}

impl std::error::Error for NanoManualPlantBindingError {}

/// Manual motion is disabled or explicitly bounded and control-API-owned.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NanoManualModePolicy {
    Disabled,
    ControlApi(NanoManualControlApiConfig),
}

impl NanoManualModePolicy {
    pub const fn config(self) -> Option<NanoManualControlApiConfig> {
        match self {
            Self::Disabled => None,
            Self::ControlApi(config) => Some(config),
        }
    }
}

impl NanoMotionModePolicy {
    pub const fn authority_lease(self) -> Option<AuthorityDuration> {
        match self {
            Self::Disabled => None,
            Self::ControlApi { authority_lease } => Some(authority_lease),
        }
    }
}

/// Finite map-frame rectangle supplied by the operator for exploration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NanoExploreBoundaryMeters {
    minimum_x_m: f64,
    minimum_y_m: f64,
    maximum_x_m: f64,
    maximum_y_m: f64,
}

impl NanoExploreBoundaryMeters {
    pub fn try_new(
        minimum_x_m: f64,
        minimum_y_m: f64,
        maximum_x_m: f64,
        maximum_y_m: f64,
    ) -> Result<Self, NanoExploreBoundaryError> {
        for (component, value) in [
            (NanoExploreBoundaryComponent::MinimumX, minimum_x_m),
            (NanoExploreBoundaryComponent::MinimumY, minimum_y_m),
            (NanoExploreBoundaryComponent::MaximumX, maximum_x_m),
            (NanoExploreBoundaryComponent::MaximumY, maximum_y_m),
        ] {
            if !value.is_finite() {
                return Err(NanoExploreBoundaryError::NonFinite { component, value });
            }
            if value.abs() > MAX_NANO_AGENT_ABS_EXPLORE_BOUNDARY_M {
                return Err(NanoExploreBoundaryError::OutsideSupportedRange {
                    component,
                    value_m: value,
                    maximum_absolute_m: MAX_NANO_AGENT_ABS_EXPLORE_BOUNDARY_M,
                });
            }
        }
        if minimum_x_m >= maximum_x_m {
            return Err(NanoExploreBoundaryError::EmptyOrReversedX {
                minimum_x_m,
                maximum_x_m,
            });
        }
        if minimum_y_m >= maximum_y_m {
            return Err(NanoExploreBoundaryError::EmptyOrReversedY {
                minimum_y_m,
                maximum_y_m,
            });
        }
        Ok(Self {
            minimum_x_m,
            minimum_y_m,
            maximum_x_m,
            maximum_y_m,
        })
    }

    pub const fn minimum_x_m(self) -> f64 {
        self.minimum_x_m
    }

    pub const fn minimum_y_m(self) -> f64 {
        self.minimum_y_m
    }

    pub const fn maximum_x_m(self) -> f64 {
        self.maximum_x_m
    }

    pub const fn maximum_y_m(self) -> f64 {
        self.maximum_y_m
    }
}

/// Bounded resources and authority lease for one frontier-exploration request.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NanoFrontierExploreConfig {
    authority_lease: AuthorityDuration,
    boundary_m: NanoExploreBoundaryMeters,
    maximum_runtime: Duration,
    maximum_frontier_goals: NonZeroU32,
}

impl NanoFrontierExploreConfig {
    pub const fn authority_lease(self) -> AuthorityDuration {
        self.authority_lease
    }

    pub const fn boundary_m(self) -> NanoExploreBoundaryMeters {
        self.boundary_m
    }

    pub const fn maximum_runtime(self) -> Duration {
        self.maximum_runtime
    }

    pub const fn maximum_frontier_goals(self) -> NonZeroU32 {
        self.maximum_frontier_goals
    }
}

/// Exploration is disabled or admitted only through the local control API.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NanoFrontierExplorePolicy {
    Disabled,
    ControlApi(NanoFrontierExploreConfig),
}

impl NanoFrontierExplorePolicy {
    pub const fn config(self) -> Option<NanoFrontierExploreConfig> {
        match self {
            Self::Disabled => None,
            Self::ControlApi(config) => Some(config),
        }
    }
}

/// Explicit startup and command-admission policy for every live motion mode.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NanoLiveModePolicy {
    startup: NanoAgentStartupMode,
    manual: NanoManualModePolicy,
    point_goal: NanoMotionModePolicy,
    frontier_explore: NanoFrontierExplorePolicy,
}

impl NanoLiveModePolicy {
    pub const fn startup(self) -> NanoAgentStartupMode {
        self.startup
    }

    pub const fn manual(self) -> NanoManualModePolicy {
        self.manual
    }

    pub const fn point_goal(self) -> NanoMotionModePolicy {
        self.point_goal
    }

    pub const fn frontier_explore(self) -> NanoFrontierExplorePolicy {
        self.frontier_explore
    }
}

#[derive(Debug)]
pub enum NanoAgentPolicyConfigParseError {
    InputTooLarge {
        actual_bytes: usize,
        maximum_bytes: usize,
    },
    JsonDecode(serde_json::Error),
    JsonTrailingData(serde_json::Error),
    UnsupportedSchemaVersion {
        actual: u32,
        supported: u32,
    },
    ControlSocketPath(AgentControlSocketPathError),
    ControlSocketTimeout(AgentControlSocketTimeoutError),
    ControlRuntimeQueue(AgentControlRuntimeQueueCapacityError),
    AbsolutePath {
        field: NanoAgentPathField,
        source: NanoAbsolutePathError,
    },
    ArtifactBindings(ArtifactFileBindingParseError),
    WarmStartPathRoleCollision,
    Eye(EyeConfigParseError),
    Head(HeadConfigParseError),
    DuplicateAccessorySerialPath,
    RgbExpressionRequiresEye,
    RgbSamplingGeometry(SamplingGeometryError),
    RgbActiveFraction(kiko_expression_core::AmountError),
    RgbMotionThreshold(SceneMotionConfigError),
    RgbBrightness(kiko_expression_core::AmountError),
    UnsupportedRgbGazeGeometrySchemaVersion {
        actual: u32,
        supported: u32,
    },
    RgbGazeGeometry(GazeExtrinsicsParseError),
    RgbFrameFreshness {
        source: NanoBoundedMillisecondsError,
    },
    RgbFrameFreshnessDomain(TimeError),
    RgbEyeRoundTripBudgetOverflow {
        write_timeout_ms: u128,
        write_attempts: u8,
        response_timeout_ms: u128,
    },
    RgbFreshnessDoesNotCoverEyeRoundTrip {
        frame_freshness_ms: u64,
        worst_case_eye_round_trip_ms: u64,
    },
    SupervisorDuration {
        field: NanoDurationField,
        source: NanoBoundedMillisecondsError,
    },
    SupervisorTimeDomain {
        field: NanoDurationField,
        source: TimeValueError,
    },
    SupervisorConfig(SupervisorConfigError),
    ModeAuthorityLease {
        mode: NanoConfiguredMotionMode,
        source: NanoBoundedMillisecondsError,
    },
    ModeAuthorityLeaseTimeDomain {
        mode: NanoConfiguredMotionMode,
        source: TimeValueError,
    },
    ModeAuthorityLeaseExceedsSupervisor {
        mode: NanoConfiguredMotionMode,
        lease_ms: u64,
        supervisor_maximum_ms: u64,
    },
    ManualCommandAge {
        source: NanoBoundedMillisecondsError,
    },
    ManualDeadman {
        source: NanoBoundedMillisecondsError,
    },
    ManualDrive(ManualDriveConfigParseError),
    MotionPolicyRequiresActuationFeature {
        mode: NanoConfiguredMotionMode,
    },
    ExploreBoundary {
        source: NanoExploreBoundaryError,
    },
    ExploreRuntime {
        source: NanoBoundedMillisecondsError,
    },
    ExploreGoalCount {
        actual: u32,
        minimum: u32,
        maximum: u32,
    },
}

impl fmt::Display for NanoAgentPolicyConfigParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid Nano agent policy configuration: {self:?}"
        )
    }
}

impl std::error::Error for NanoAgentPolicyConfigParseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::JsonDecode(source) | Self::JsonTrailingData(source) => Some(source),
            Self::ControlSocketPath(source) => Some(source),
            Self::ControlSocketTimeout(source) => Some(source),
            Self::ControlRuntimeQueue(source) => Some(source),
            Self::AbsolutePath { source, .. } => Some(source),
            Self::ArtifactBindings(source) => Some(source),
            Self::Eye(source) => Some(source),
            Self::Head(source) => Some(source),
            Self::RgbSamplingGeometry(source) => Some(source),
            Self::RgbActiveFraction(source) | Self::RgbBrightness(source) => Some(source),
            Self::RgbMotionThreshold(source) => Some(source),
            Self::RgbGazeGeometry(source) => Some(source),
            Self::RgbFrameFreshness { source }
            | Self::SupervisorDuration { source, .. }
            | Self::ModeAuthorityLease { source, .. }
            | Self::ManualCommandAge { source }
            | Self::ManualDeadman { source }
            | Self::ExploreRuntime { source } => Some(source),
            Self::RgbFrameFreshnessDomain(source) => Some(source),
            Self::SupervisorTimeDomain { source, .. }
            | Self::ModeAuthorityLeaseTimeDomain { source, .. } => Some(source),
            Self::SupervisorConfig(source) => Some(source),
            Self::ManualDrive(source) => Some(source),
            Self::ExploreBoundary { source } => Some(source),
            Self::InputTooLarge { .. }
            | Self::UnsupportedSchemaVersion { .. }
            | Self::WarmStartPathRoleCollision
            | Self::DuplicateAccessorySerialPath
            | Self::RgbExpressionRequiresEye
            | Self::UnsupportedRgbGazeGeometrySchemaVersion { .. }
            | Self::RgbEyeRoundTripBudgetOverflow { .. }
            | Self::RgbFreshnessDoesNotCoverEyeRoundTrip { .. }
            | Self::ModeAuthorityLeaseExceedsSupervisor { .. }
            | Self::MotionPolicyRequiresActuationFeature { .. }
            | Self::ExploreGoalCount { .. } => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoAgentPathField {
    Manifest,
    ArtifactRoot,
    MapSaveSnapshot,
    WarmOccupancySnapshot,
    WarmSlamDatasetDirectory,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoAbsolutePathError {
    Empty,
    TooLong {
        actual_bytes: usize,
        maximum_bytes: usize,
    },
    ContainsNul,
    NotAbsolute,
    RootNotAllowed,
    NonCanonicalComponent,
}

impl fmt::Display for NanoAbsolutePathError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid configured absolute path: {self:?}")
    }
}

impl std::error::Error for NanoAbsolutePathError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoDurationField {
    SupervisorMaximumAuthorityLeaseMs,
    SupervisorMaximumZeroAgeMs,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoConfiguredMotionMode {
    Manual,
    PointGoal,
    FrontierExplore,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoBoundedMillisecondsError {
    Zero,
    TooLarge { actual_ms: u64, maximum_ms: u64 },
    NanosecondsOverflow { milliseconds: u64 },
}

impl fmt::Display for NanoBoundedMillisecondsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid bounded millisecond duration: {self:?}")
    }
}

impl std::error::Error for NanoBoundedMillisecondsError {}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NanoExploreBoundaryError {
    NonFinite {
        component: NanoExploreBoundaryComponent,
        value: f64,
    },
    OutsideSupportedRange {
        component: NanoExploreBoundaryComponent,
        value_m: f64,
        maximum_absolute_m: f64,
    },
    EmptyOrReversedX {
        minimum_x_m: f64,
        maximum_x_m: f64,
    },
    EmptyOrReversedY {
        minimum_y_m: f64,
        maximum_y_m: f64,
    },
}

impl fmt::Display for NanoExploreBoundaryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid exploration boundary: {self:?}")
    }
}

impl std::error::Error for NanoExploreBoundaryError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoExploreBoundaryComponent {
    MinimumX,
    MinimumY,
    MaximumX,
    MaximumY,
}

fn parse_control(
    dto: NanoAgentControlConfigDto,
) -> Result<NanoAgentControlConfig, NanoAgentPolicyConfigParseError> {
    let path = AgentControlSocketPath::parse(Path::new(&dto.socket_path))
        .map_err(NanoAgentPolicyConfigParseError::ControlSocketPath)?;
    let timeouts = AgentControlSocketTimeouts::try_new(
        Duration::from_millis(dto.read_timeout_ms),
        Duration::from_millis(dto.write_timeout_ms),
        Duration::from_millis(dto.runtime_response_timeout_ms),
    )
    .map_err(NanoAgentPolicyConfigParseError::ControlSocketTimeout)?;
    let runtime_queue_capacity =
        AgentControlRuntimeQueueCapacity::try_new(usize::from(dto.runtime_queue_capacity))
            .map_err(NanoAgentPolicyConfigParseError::ControlRuntimeQueue)?;
    Ok(NanoAgentControlConfig {
        socket: AgentControlSocketConfig::new(path, timeouts),
        runtime_queue_capacity,
    })
}

fn parse_inventory(
    dto: NanoAgentInventoryConfigDto,
) -> Result<NanoAgentInventoryConfig, NanoAgentPolicyConfigParseError> {
    let manifest_path = parse_absolute_path(
        NanoAgentPathField::Manifest,
        dto.manifest_path,
        MAX_MANIFEST_PATH_BYTES,
    )?;
    let artifact_root_path = parse_absolute_path(
        NanoAgentPathField::ArtifactRoot,
        dto.artifact_root_path,
        MAX_ARTIFACT_ROOT_PATH_BYTES,
    )?;

    let bindings = dto
        .artifact_bindings
        .into_iter()
        .map(|binding| ArtifactFileBindingInput {
            kind: binding.kind.into_domain(),
            artifact_id: binding.artifact_id,
            relative_path: binding.relative_path,
        })
        .collect();
    let artifact_bindings = ArtifactFileBindingSet::parse(bindings)
        .map_err(NanoAgentPolicyConfigParseError::ArtifactBindings)?;

    Ok(NanoAgentInventoryConfig {
        manifest_path,
        artifact_root_path,
        artifact_bindings,
    })
}

fn parse_map_persistence(
    dto: NanoMapPersistenceConfigDto,
) -> Result<NanoMapPersistenceConfig, NanoAgentPolicyConfigParseError> {
    let save_snapshot_path = parse_absolute_path(
        NanoAgentPathField::MapSaveSnapshot,
        dto.save_snapshot_path,
        MAX_NANO_AGENT_DATA_PATH_BYTES,
    )?;
    let warm_start = match dto.warm_start {
        NanoMapWarmStartDto::None => NanoMapWarmStart::None,
        NanoMapWarmStartDto::DatasetReplay {
            occupancy_snapshot_path,
            slam_dataset_directory_path,
        } => {
            let occupancy_snapshot_path = parse_absolute_path(
                NanoAgentPathField::WarmOccupancySnapshot,
                occupancy_snapshot_path,
                MAX_NANO_AGENT_DATA_PATH_BYTES,
            )?;
            let slam_dataset_directory_path = parse_absolute_path(
                NanoAgentPathField::WarmSlamDatasetDirectory,
                slam_dataset_directory_path,
                MAX_NANO_AGENT_DATA_PATH_BYTES,
            )?;
            if occupancy_snapshot_path == slam_dataset_directory_path
                || save_snapshot_path == slam_dataset_directory_path
            {
                return Err(NanoAgentPolicyConfigParseError::WarmStartPathRoleCollision);
            }
            NanoMapWarmStart::DatasetReplay {
                occupancy_snapshot_path,
                slam_dataset_directory_path,
            }
        }
    };
    Ok(NanoMapPersistenceConfig {
        save_snapshot_path,
        warm_start,
    })
}

fn parse_eye(
    dto: NanoEyePolicyDto,
) -> Result<ParsedNanoEyePolicy, NanoAgentPolicyConfigParseError> {
    match dto {
        NanoEyePolicyDto::Disabled => Ok(ParsedNanoEyePolicy::Disabled),
        NanoEyePolicyDto::Kep2 {
            device_path,
            baud_rate_bps,
            response_timeout_ms,
            write_timeout_ms,
            write_attempts,
            empty_delimiter_budget,
            expected_device_uid,
            expected_firmware_build_id,
            expected_capabilities_bits,
            intent_lease_ms,
        } => StaticEyeRuntimeConfig::parse(StaticEyeRuntimeConfigInput {
            device_path,
            baud_rate_bps,
            response_timeout_ms,
            write_timeout_ms,
            write_attempts,
            empty_delimiter_budget,
            expected_device_uid,
            expected_firmware_build_id,
            expected_capabilities_bits,
            intent_lease_ms,
        })
        .map(ParsedNanoEyePolicy::Kep2)
        .map_err(NanoAgentPolicyConfigParseError::Eye),
    }
}

fn parse_head(
    dto: NanoHeadPolicyDto,
) -> Result<ParsedNanoHeadPolicy, NanoAgentPolicyConfigParseError> {
    match dto {
        NanoHeadPolicyDto::Disabled => Ok(ParsedNanoHeadPolicy::Disabled),
        NanoHeadPolicyDto::NaturalHold {
            device_path,
            response_timeout_ms,
            write_timeout_ms,
            arming_freshness_ms,
            write_attempts,
            noise_budget_bytes,
            redundant_read_tolerance_ticks,
            readback_tolerance_ticks,
            goal_speed_ticks_per_second,
            torque_limit_permille,
            physical_torque_consent: NanoPhysicalTorqueConsentDto::NaturalHoldAtObservedPose,
        } => HeadRuntimeConfig::parse(HeadRuntimeConfigInput {
            device_path,
            response_timeout_ms,
            write_timeout_ms,
            arming_freshness_ms,
            write_attempts,
            noise_budget_bytes,
            redundant_read_tolerance_ticks,
            readback_tolerance_ticks,
            goal_speed_ticks_per_second,
            torque_limit_permille,
        })
        .map(|runtime| {
            ParsedNanoHeadPolicy::NaturalHold(NanoNaturalHeadHoldConfig {
                runtime,
                torque_consent: PhysicalTorqueEnableConsent::explicitly_granted(),
            })
        })
        .map_err(NanoAgentPolicyConfigParseError::Head),
    }
}

fn parse_rgb_expression(
    dto: NanoRgbExpressionPolicyDto,
    eye: &ParsedNanoEyePolicy,
) -> Result<NanoRgbExpressionPolicy, NanoAgentPolicyConfigParseError> {
    let NanoRgbExpressionPolicyDto::SceneMotion {
        sampling_columns,
        sampling_rows,
        minimum_residual_luma,
        minimum_active_fraction_basis_points,
        frame_freshness_ms,
        brightness_basis_points,
        color_rgb,
        blink,
        gaze_geometry,
    } = dto
    else {
        return Ok(NanoRgbExpressionPolicy::Disabled);
    };
    let ParsedNanoEyePolicy::Kep2(eye) = eye else {
        return Err(NanoAgentPolicyConfigParseError::RgbExpressionRequiresEye);
    };

    let geometry = SamplingGeometry::try_new(sampling_columns, sampling_rows)
        .map_err(NanoAgentPolicyConfigParseError::RgbSamplingGeometry)?;
    let active_fraction =
        PositiveUnitAmount::try_from_basis_points(minimum_active_fraction_basis_points)
            .map_err(NanoAgentPolicyConfigParseError::RgbActiveFraction)?;
    let thresholds = MotionThresholds::try_new(minimum_residual_luma, active_fraction)
        .map_err(NanoAgentPolicyConfigParseError::RgbMotionThreshold)?;
    let brightness = UnitAmount::try_from_basis_points(brightness_basis_points)
        .map_err(NanoAgentPolicyConfigParseError::RgbBrightness)?;
    let gaze_geometry = match gaze_geometry {
        Some(raw) => {
            if raw.schema_version != NANO_RGB_GAZE_GEOMETRY_V1 {
                return Err(
                    NanoAgentPolicyConfigParseError::UnsupportedRgbGazeGeometrySchemaVersion {
                        actual: raw.schema_version,
                        supported: NANO_RGB_GAZE_GEOMETRY_V1,
                    },
                );
            }
            Some(
                CameraToHeadGazeExtrinsics::parse(CameraToHeadGazeExtrinsicsInput {
                    head_origin_in_camera_m: raw.head_origin_in_camera_m,
                    neutral_head_from_camera_quaternion_xyzw: raw
                        .neutral_head_from_camera_quaternion_xyzw,
                })
                .map_err(NanoAgentPolicyConfigParseError::RgbGazeGeometry)?,
            )
        }
        None => None,
    };
    let frame_freshness_ns =
        parse_bounded_milliseconds(frame_freshness_ms, MAX_NANO_AGENT_RGB_FRAME_FRESHNESS_MS)
            .map_err(|source| NanoAgentPolicyConfigParseError::RgbFrameFreshness { source })?;
    let frame_freshness = NonZeroDuration::try_from_nanos(frame_freshness_ns)
        .map_err(NanoAgentPolicyConfigParseError::RgbFrameFreshnessDomain)?;

    let write_timeout_ms = eye.write_timeout().get().as_millis();
    let write_attempts = eye.write_attempts().get();
    let response_timeout_ms = eye.response_timeout().get().as_millis();
    let worst_case_eye_round_trip_ms = write_timeout_ms
        .checked_mul(u128::from(write_attempts))
        .and_then(|writes| writes.checked_add(response_timeout_ms))
        .and_then(|milliseconds| u64::try_from(milliseconds).ok())
        .ok_or(
            NanoAgentPolicyConfigParseError::RgbEyeRoundTripBudgetOverflow {
                write_timeout_ms,
                write_attempts,
                response_timeout_ms,
            },
        )?;
    if frame_freshness_ms <= worst_case_eye_round_trip_ms {
        return Err(
            NanoAgentPolicyConfigParseError::RgbFreshnessDoesNotCoverEyeRoundTrip {
                frame_freshness_ms,
                worst_case_eye_round_trip_ms,
            },
        );
    }

    Ok(NanoRgbExpressionPolicy::SceneMotion(
        NanoRgbExpressionConfig {
            scene_motion: SceneMotionConfig::new(geometry, thresholds),
            frame_freshness,
            render_style: EyeRenderStyle::new(brightness, color_rgb, blink),
            gaze_geometry,
        },
    ))
}

fn parse_supervisor(
    dto: NanoSupervisorConfigDto,
) -> Result<SupervisorConfig, NanoAgentPolicyConfigParseError> {
    let maximum_authority_lease = parse_supervisor_duration(
        NanoDurationField::SupervisorMaximumAuthorityLeaseMs,
        dto.maximum_authority_lease_ms,
    )?;
    let maximum_zero_age = parse_supervisor_duration(
        NanoDurationField::SupervisorMaximumZeroAgeMs,
        dto.maximum_zero_age_ms,
    )?;
    SupervisorConfig::new(maximum_authority_lease, maximum_zero_age)
        .map_err(NanoAgentPolicyConfigParseError::SupervisorConfig)
}

fn parse_supervisor_duration(
    field: NanoDurationField,
    milliseconds: u64,
) -> Result<AuthorityDuration, NanoAgentPolicyConfigParseError> {
    let nanoseconds = parse_bounded_milliseconds(milliseconds, MAX_NANO_AGENT_AUTHORITY_LEASE_MS)
        .map_err(
        |source| NanoAgentPolicyConfigParseError::SupervisorDuration { field, source },
    )?;
    AuthorityDuration::try_from_nanos(nanoseconds)
        .map_err(|source| NanoAgentPolicyConfigParseError::SupervisorTimeDomain { field, source })
}

fn parse_live_mode_policy(
    dto: NanoLiveModePolicyDto,
    supervisor: SupervisorConfig,
) -> Result<NanoLiveModePolicy, NanoAgentPolicyConfigParseError> {
    let NanoAgentStartupModeDto::DisarmedMapOnly = dto.startup;
    let manual = parse_manual_mode_policy(dto.manual, supervisor)?;
    let point_goal = parse_motion_mode_policy(
        NanoConfiguredMotionMode::PointGoal,
        dto.point_goal,
        supervisor,
    )?;
    let frontier_explore = match dto.frontier_explore {
        NanoFrontierExplorePolicyDto::Disabled => NanoFrontierExplorePolicy::Disabled,
        NanoFrontierExplorePolicyDto::ControlApi {
            authority_lease_ms,
            boundary_minimum_x_m,
            boundary_minimum_y_m,
            boundary_maximum_x_m,
            boundary_maximum_y_m,
            maximum_runtime_ms,
            maximum_frontier_goals,
        } => {
            if !cfg!(feature = "actuation") {
                return Err(
                    NanoAgentPolicyConfigParseError::MotionPolicyRequiresActuationFeature {
                        mode: NanoConfiguredMotionMode::FrontierExplore,
                    },
                );
            }
            let authority_lease = parse_mode_authority_lease(
                NanoConfiguredMotionMode::FrontierExplore,
                authority_lease_ms,
                supervisor,
            )?;
            let boundary_m = parse_explore_boundary(
                boundary_minimum_x_m,
                boundary_minimum_y_m,
                boundary_maximum_x_m,
                boundary_maximum_y_m,
            )?;
            let maximum_runtime_ns =
                parse_bounded_milliseconds(maximum_runtime_ms, MAX_NANO_AGENT_EXPLORE_RUNTIME_MS)
                    .map_err(|source| NanoAgentPolicyConfigParseError::ExploreRuntime { source })?;
            let raw_maximum_frontier_goals = maximum_frontier_goals;
            let maximum_frontier_goals = NonZeroU32::new(raw_maximum_frontier_goals)
                .filter(|value| value.get() <= MAX_NANO_AGENT_EXPLORE_GOALS);
            let Some(maximum_frontier_goals) = maximum_frontier_goals else {
                return Err(NanoAgentPolicyConfigParseError::ExploreGoalCount {
                    actual: raw_maximum_frontier_goals,
                    minimum: 1,
                    maximum: MAX_NANO_AGENT_EXPLORE_GOALS,
                });
            };
            NanoFrontierExplorePolicy::ControlApi(NanoFrontierExploreConfig {
                authority_lease,
                boundary_m,
                maximum_runtime: Duration::from_nanos(maximum_runtime_ns),
                maximum_frontier_goals,
            })
        }
    };

    Ok(NanoLiveModePolicy {
        startup: NanoAgentStartupMode::DisarmedMapOnly,
        manual,
        point_goal,
        frontier_explore,
    })
}

fn parse_manual_mode_policy(
    dto: NanoManualModePolicyDto,
    supervisor: SupervisorConfig,
) -> Result<NanoManualModePolicy, NanoAgentPolicyConfigParseError> {
    let NanoManualModePolicyDto::ControlApi {
        authority_lease_ms,
        maximum_abs_forward_velocity_mps,
        maximum_abs_yaw_rate_rad_s,
        maximum_command_age_ms,
        deadman_timeout_ms,
    } = dto
    else {
        return Ok(NanoManualModePolicy::Disabled);
    };
    if !cfg!(feature = "actuation") {
        return Err(
            NanoAgentPolicyConfigParseError::MotionPolicyRequiresActuationFeature {
                mode: NanoConfiguredMotionMode::Manual,
            },
        );
    }
    let authority_lease = parse_mode_authority_lease(
        NanoConfiguredMotionMode::Manual,
        authority_lease_ms,
        supervisor,
    )?;
    let maximum_command_age_ns =
        parse_bounded_milliseconds(maximum_command_age_ms, MAX_NANO_AGENT_MANUAL_WINDOW_MS)
            .map_err(|source| NanoAgentPolicyConfigParseError::ManualCommandAge { source })?;
    let deadman_timeout_ns =
        parse_bounded_milliseconds(deadman_timeout_ms, MAX_NANO_AGENT_MANUAL_WINDOW_MS)
            .map_err(|source| NanoAgentPolicyConfigParseError::ManualDeadman { source })?;
    let drive = ManualDriveConfigV1::parse(ManualDriveConfigV1Dto {
        schema_version: MANUAL_DRIVE_CONFIG_V1,
        maximum_abs_forward_velocity_mps,
        maximum_abs_yaw_rate_rad_s,
        maximum_command_age_ns,
        deadman_timeout_ns,
    })
    .map_err(NanoAgentPolicyConfigParseError::ManualDrive)?;
    Ok(NanoManualModePolicy::ControlApi(
        NanoManualControlApiConfig {
            authority_lease,
            drive,
        },
    ))
}

fn parse_motion_mode_policy(
    mode: NanoConfiguredMotionMode,
    dto: NanoMotionModePolicyDto,
    supervisor: SupervisorConfig,
) -> Result<NanoMotionModePolicy, NanoAgentPolicyConfigParseError> {
    match dto {
        NanoMotionModePolicyDto::Disabled => Ok(NanoMotionModePolicy::Disabled),
        NanoMotionModePolicyDto::ControlApi { authority_lease_ms } => {
            if !cfg!(feature = "actuation") {
                return Err(
                    NanoAgentPolicyConfigParseError::MotionPolicyRequiresActuationFeature { mode },
                );
            }
            parse_mode_authority_lease(mode, authority_lease_ms, supervisor)
                .map(|authority_lease| NanoMotionModePolicy::ControlApi { authority_lease })
        }
    }
}

fn parse_mode_authority_lease(
    mode: NanoConfiguredMotionMode,
    milliseconds: u64,
    supervisor: SupervisorConfig,
) -> Result<AuthorityDuration, NanoAgentPolicyConfigParseError> {
    let nanoseconds = parse_bounded_milliseconds(milliseconds, MAX_NANO_AGENT_AUTHORITY_LEASE_MS)
        .map_err(
        |source| NanoAgentPolicyConfigParseError::ModeAuthorityLease { mode, source },
    )?;
    let duration = AuthorityDuration::try_from_nanos(nanoseconds).map_err(|source| {
        NanoAgentPolicyConfigParseError::ModeAuthorityLeaseTimeDomain { mode, source }
    })?;
    if duration.as_nanos() > supervisor.maximum_authority_lease().as_nanos() {
        return Err(
            NanoAgentPolicyConfigParseError::ModeAuthorityLeaseExceedsSupervisor {
                mode,
                lease_ms: milliseconds,
                supervisor_maximum_ms: supervisor.maximum_authority_lease().as_nanos() / 1_000_000,
            },
        );
    }
    Ok(duration)
}

fn parse_explore_boundary(
    minimum_x_m: f64,
    minimum_y_m: f64,
    maximum_x_m: f64,
    maximum_y_m: f64,
) -> Result<NanoExploreBoundaryMeters, NanoAgentPolicyConfigParseError> {
    NanoExploreBoundaryMeters::try_new(minimum_x_m, minimum_y_m, maximum_x_m, maximum_y_m)
        .map_err(|source| NanoAgentPolicyConfigParseError::ExploreBoundary { source })
}

fn parse_absolute_path(
    field: NanoAgentPathField,
    value: String,
    maximum_bytes: usize,
) -> Result<NanoConfiguredAbsolutePath, NanoAgentPolicyConfigParseError> {
    let source = if value.is_empty() {
        Some(NanoAbsolutePathError::Empty)
    } else if value.len() > maximum_bytes {
        Some(NanoAbsolutePathError::TooLong {
            actual_bytes: value.len(),
            maximum_bytes,
        })
    } else if value.as_bytes().contains(&0) {
        Some(NanoAbsolutePathError::ContainsNul)
    } else if !value.starts_with('/') {
        Some(NanoAbsolutePathError::NotAbsolute)
    } else if value == "/" {
        Some(NanoAbsolutePathError::RootNotAllowed)
    } else if value[1..]
        .split('/')
        .any(|component| component.is_empty() || matches!(component, "." | ".."))
    {
        Some(NanoAbsolutePathError::NonCanonicalComponent)
    } else {
        None
    };
    match source {
        Some(source) => Err(NanoAgentPolicyConfigParseError::AbsolutePath { field, source }),
        None => Ok(NanoConfiguredAbsolutePath(PathBuf::from(value))),
    }
}

fn parse_bounded_milliseconds(
    milliseconds: u64,
    maximum_ms: u64,
) -> Result<u64, NanoBoundedMillisecondsError> {
    if milliseconds == 0 {
        return Err(NanoBoundedMillisecondsError::Zero);
    }
    if milliseconds > maximum_ms {
        return Err(NanoBoundedMillisecondsError::TooLarge {
            actual_ms: milliseconds,
            maximum_ms,
        });
    }
    milliseconds
        .checked_mul(1_000_000)
        .ok_or(NanoBoundedMillisecondsError::NanosecondsOverflow { milliseconds })
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NanoAgentPolicyConfigV1Dto {
    schema_version: u32,
    control: NanoAgentControlConfigDto,
    inventory: NanoAgentInventoryConfigDto,
    map_persistence: NanoMapPersistenceConfigDto,
    eye: NanoEyePolicyDto,
    head: NanoHeadPolicyDto,
    rgb_expression: NanoRgbExpressionPolicyDto,
    supervisor: NanoSupervisorConfigDto,
    live_mode_policy: NanoLiveModePolicyDto,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NanoAgentControlConfigDto {
    socket_path: String,
    read_timeout_ms: u64,
    write_timeout_ms: u64,
    runtime_response_timeout_ms: u64,
    runtime_queue_capacity: u16,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NanoAgentInventoryConfigDto {
    manifest_path: String,
    artifact_root_path: String,
    artifact_bindings: Vec<NanoArtifactFileBindingDto>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NanoArtifactFileBindingDto {
    kind: ArtifactKindDto,
    artifact_id: String,
    relative_path: String,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum ArtifactKindDto {
    Calibration,
    Plant,
}

impl ArtifactKindDto {
    const fn into_domain(self) -> ArtifactKind {
        match self {
            Self::Calibration => ArtifactKind::Calibration,
            Self::Plant => ArtifactKind::Plant,
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NanoMapPersistenceConfigDto {
    save_snapshot_path: String,
    warm_start: NanoMapWarmStartDto,
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum NanoMapWarmStartDto {
    None,
    DatasetReplay {
        occupancy_snapshot_path: String,
        slam_dataset_directory_path: String,
    },
}

#[derive(Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
enum NanoEyePolicyDto {
    Disabled,
    Kep2 {
        device_path: String,
        baud_rate_bps: u32,
        response_timeout_ms: u64,
        write_timeout_ms: u64,
        write_attempts: u8,
        empty_delimiter_budget: u8,
        expected_device_uid: [u8; 16],
        expected_firmware_build_id: [u8; 32],
        expected_capabilities_bits: u32,
        intent_lease_ms: u16,
    },
}

#[derive(Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
enum NanoHeadPolicyDto {
    Disabled,
    NaturalHold {
        device_path: String,
        response_timeout_ms: u64,
        write_timeout_ms: u64,
        arming_freshness_ms: u64,
        write_attempts: u8,
        noise_budget_bytes: u16,
        redundant_read_tolerance_ticks: u16,
        readback_tolerance_ticks: u16,
        goal_speed_ticks_per_second: u16,
        torque_limit_permille: [u16; 4],
        physical_torque_consent: NanoPhysicalTorqueConsentDto,
    },
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum NanoPhysicalTorqueConsentDto {
    NaturalHoldAtObservedPose,
}

#[derive(Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
enum NanoRgbExpressionPolicyDto {
    Disabled,
    SceneMotion {
        sampling_columns: u16,
        sampling_rows: u16,
        minimum_residual_luma: u16,
        minimum_active_fraction_basis_points: u16,
        frame_freshness_ms: u64,
        brightness_basis_points: u16,
        color_rgb: [u8; 3],
        blink: bool,
        #[serde(default, deserialize_with = "deserialize_present_rgb_gaze_geometry")]
        gaze_geometry: Option<NanoRgbGazeGeometryDto>,
    },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NanoRgbGazeGeometryDto {
    schema_version: u32,
    head_origin_in_camera_m: [f64; 3],
    neutral_head_from_camera_quaternion_xyzw: [f64; 4],
}

/// Missing preserves schema-v1 compatibility; an explicitly present value must
/// be the versioned object, so JSON `null` cannot masquerade as absence.
fn deserialize_present_rgb_gaze_geometry<'de, D>(
    deserializer: D,
) -> Result<Option<NanoRgbGazeGeometryDto>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    NanoRgbGazeGeometryDto::deserialize(deserializer).map(Some)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NanoSupervisorConfigDto {
    maximum_authority_lease_ms: u64,
    maximum_zero_age_ms: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NanoLiveModePolicyDto {
    startup: NanoAgentStartupModeDto,
    manual: NanoManualModePolicyDto,
    point_goal: NanoMotionModePolicyDto,
    frontier_explore: NanoFrontierExplorePolicyDto,
}

#[derive(Deserialize)]
#[serde(tag = "permission", rename_all = "snake_case", deny_unknown_fields)]
enum NanoManualModePolicyDto {
    Disabled,
    ControlApi {
        authority_lease_ms: u64,
        maximum_abs_forward_velocity_mps: f64,
        maximum_abs_yaw_rate_rad_s: f64,
        maximum_command_age_ms: u64,
        deadman_timeout_ms: u64,
    },
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum NanoAgentStartupModeDto {
    DisarmedMapOnly,
}

#[derive(Deserialize)]
#[serde(tag = "permission", rename_all = "snake_case", deny_unknown_fields)]
enum NanoMotionModePolicyDto {
    Disabled,
    ControlApi { authority_lease_ms: u64 },
}

#[derive(Deserialize)]
#[serde(tag = "permission", rename_all = "snake_case", deny_unknown_fields)]
enum NanoFrontierExplorePolicyDto {
    Disabled,
    ControlApi {
        authority_lease_ms: u64,
        boundary_minimum_x_m: f64,
        boundary_minimum_y_m: f64,
        boundary_maximum_x_m: f64,
        boundary_maximum_y_m: f64,
        maximum_runtime_ms: u64,
        maximum_frontier_goals: u32,
    },
}

#[cfg(test)]
mod tests {
    use kiko_expression_runtime::REQUIRED_EYE_CAPABILITIES;
    use serde_json::{Value, json};

    #[cfg(feature = "actuation")]
    use super::super::mpc::{
        PLANT_MODEL_V1, PlantEvidenceV1Dto, PlantModelV1Dto, PlantValidityEnvelopeV1Dto,
        WheelPlantV1Dto,
    };
    use super::*;

    #[cfg(feature = "actuation")]
    fn bounded_manual_plant() -> PlantModelV1 {
        PlantModelV1::parse(PlantModelV1Dto {
            schema_version: PLANT_MODEL_V1,
            model_id: "manual-binding-fixture".to_owned(),
            model_version: 1,
            sample_period_s: 0.1,
            wheelbase_m: 0.4,
            left: WheelPlantV1Dto {
                velocity_gain_mps_per_pwm_percent: 0.01,
                time_constant_s: 0.2,
            },
            right: WheelPlantV1Dto {
                velocity_gain_mps_per_pwm_percent: 0.01,
                time_constant_s: 0.2,
            },
            validity: PlantValidityEnvelopeV1Dto {
                left_pwm_min_percent: -50,
                left_pwm_max_percent: 50,
                right_pwm_min_percent: -50,
                right_pwm_max_percent: 50,
                left_velocity_min_mps: -0.5,
                left_velocity_max_mps: 0.5,
                right_velocity_min_mps: -0.5,
                right_velocity_max_mps: 0.5,
                max_abs_yaw_rate_rad_s: 2.0,
                max_abs_lateral_velocity_mps: 0.0,
            },
            evidence: PlantEvidenceV1Dto::SyntheticFixture {
                fixture_id: "manual-binding".to_owned(),
                generator_id: "unit-test".to_owned(),
            },
        })
        .expect("bounded manual plant")
    }

    fn valid_value() -> Value {
        let value = json!({
            "schema_version": NANO_AGENT_POLICY_CONFIG_V1,
            "control": {
                "socket_path": "/tmp/kiko-agent/control.sock",
                "read_timeout_ms": 100,
                "write_timeout_ms": 100,
                "runtime_response_timeout_ms": 500,
                "runtime_queue_capacity": 8
            },
            "inventory": {
                "manifest_path": "/opt/kiko/config/device-manifest.json",
                "artifact_root_path": "/opt/kiko/artifacts",
                "artifact_bindings": [
                    {
                        "kind": "calibration",
                        "artifact_id": "stereo-v1",
                        "relative_path": "calibration/stereo.json"
                    },
                    {
                        "kind": "plant",
                        "artifact_id": "drive-v1",
                        "relative_path": "plant/drive.json"
                    }
                ]
            },
            "map_persistence": {
                "save_snapshot_path": "/var/lib/kiko/maps/current.kmap",
                "warm_start": {
                    "kind": "dataset_replay",
                    "occupancy_snapshot_path": "/var/lib/kiko/maps/current.kmap",
                    "slam_dataset_directory_path": "/var/lib/kiko/datasets/current"
                }
            },
            "eye": {
                "mode": "kep2",
                "device_path": "/dev/serial/by-id/usb-kiko_kiko-eyes_1-if00",
                "baud_rate_bps": 115200,
                "response_timeout_ms": 20,
                "write_timeout_ms": 5,
                "write_attempts": 2,
                "empty_delimiter_budget": 2,
                "expected_device_uid": vec![1_u8; 16],
                "expected_firmware_build_id": vec![2_u8; 32],
                "expected_capabilities_bits": REQUIRED_EYE_CAPABILITIES,
                "intent_lease_ms": 100
            },
            "head": {
                "mode": "natural_hold",
                "device_path": "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14031114-if00",
                "response_timeout_ms": 100,
                "write_timeout_ms": 100,
                "arming_freshness_ms": 250,
                "write_attempts": 2,
                "noise_budget_bytes": 32,
                "redundant_read_tolerance_ticks": 10,
                "readback_tolerance_ticks": 20,
                "goal_speed_ticks_per_second": 100,
                "torque_limit_permille": [600, 400, 400, 400],
                "physical_torque_consent": "natural_hold_at_observed_pose"
            },
            "rgb_expression": {
                "mode": "scene_motion",
                "sampling_columns": 16,
                "sampling_rows": 12,
                "minimum_residual_luma": 24,
                "minimum_active_fraction_basis_points": 500,
                "frame_freshness_ms": 80,
                "brightness_basis_points": 7000,
                "color_rgb": [32, 128, 255],
                "blink": false,
                "gaze_geometry": {
                    "schema_version": 1,
                    "head_origin_in_camera_m": [0.0, -0.25, -0.20],
                    "neutral_head_from_camera_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]
                }
            },
            "supervisor": {
                "maximum_authority_lease_ms": 1000,
                "maximum_zero_age_ms": 250
            },
            "live_mode_policy": {
                "startup": "disarmed_map_only",
                "manual": {
                    "permission": "control_api",
                    "authority_lease_ms": 500,
                    "maximum_abs_forward_velocity_mps": 0.35,
                    "maximum_abs_yaw_rate_rad_s": 1.0,
                    "maximum_command_age_ms": 100,
                    "deadman_timeout_ms": 250
                },
                "point_goal": {
                    "permission": "control_api",
                    "authority_lease_ms": 1000
                },
                "frontier_explore": {
                    "permission": "control_api",
                    "authority_lease_ms": 1000,
                    "boundary_minimum_x_m": -20.0,
                    "boundary_minimum_y_m": -10.0,
                    "boundary_maximum_x_m": 20.0,
                    "boundary_maximum_y_m": 10.0,
                    "maximum_runtime_ms": 600000,
                    "maximum_frontier_goals": 100
                }
            }
        });
        #[cfg(not(feature = "actuation"))]
        {
            let mut value = value;
            value["live_mode_policy"]["manual"] = json!({"permission": "disabled"});
            value["live_mode_policy"]["point_goal"] = json!({"permission": "disabled"});
            value["live_mode_policy"]["frontier_explore"] = json!({"permission": "disabled"});
            value
        }
        #[cfg(feature = "actuation")]
        {
            value
        }
    }

    fn manifest_value() -> Value {
        let servo_ids = HeadJoint::ALL.map(|joint| joint.servo_id().get());
        json!({
            "schema_version": kiko_device_inventory::DEVICE_INVENTORY_MANIFEST_V1,
            "robot_id": "kiko-production-01",
            "oak": {
                "mxid": "ABCDEF1234567890",
                "compiled_depthai_header_sdk_version": "depthai-sdk-v1",
                "compiled_depthai_header_sdk_commit": "depthai-sdk-commit-v1",
                "compiled_depthai_header_embedded_device_artifact_version": "depthai-device-v1",
                "compiled_depthai_header_embedded_bootloader_artifact_version": "depthai-bootloader-v1"
            },
            "stm32": {
                "serial_by_id_path": "/dev/serial/by-id/usb-kiko-stm32-if00",
                "control_endpoint_identity": "unix:/run/kiko/robot.sock",
                "controller_uid": vec![3_u8; 12],
                "firmware_abi": u16::from(robot_protocol::v2::VERSION),
                "firmware_build_id": 7,
                "hardware_profile_fingerprint": vec![4_u8; 16],
                "capabilities_bits": robot_protocol::v2::ControllerCapabilities::REQUIRED_BITS
            },
            "head": {
                "adapter_serial_by_id_path": "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14031114-if00",
                "bow_servo_id": servo_ids[0],
                "curl_servo_id": servo_ids[1],
                "yaw_servo_id": servo_ids[2],
                "roll_servo_id": servo_ids[3],
                "baud_rate_bps": BUS_BAUD_RATE_BPS,
                "dtr_asserted": ADAPTER_DTR_ASSERTED,
                "rts_asserted": ADAPTER_RTS_ASSERTED
            },
            "eye": {
                "serial_by_id_path": "/dev/serial/by-id/usb-kiko_kiko-eyes_1-if00",
                "kep_protocol_version": EYE_PROTOCOL_VERSION,
                "device_uid": vec![1_u8; 16],
                "firmware_build_id": vec![2_u8; 32],
                "capabilities_bits": REQUIRED_EYE_CAPABILITIES
            },
            "calibration_artifacts": [{
                "artifact_id": "stereo-v1",
                "sha256": vec![5_u8; 32]
            }],
            "plant_artifacts": [{
                "artifact_id": "drive-v1",
                "sha256": vec![6_u8; 32]
            }]
        })
    }

    fn manifest(value: &Value) -> DeviceInventoryManifestV1 {
        kiko_device_inventory::load_expected_manifest_v1_from_slice(
            &serde_json::to_vec(value).expect("serialize manifest fixture"),
        )
        .expect("valid expected manifest fixture")
        .into_manifest()
    }

    fn parse(value: &Value) -> Result<NanoAgentPolicyConfigV1, NanoAgentPolicyConfigParseError> {
        NanoAgentPolicyConfigV1::parse_json(
            &serde_json::to_vec(value).expect("serialize test fixture"),
        )
    }

    #[test]
    fn valid_document_constructs_native_runtime_domains() {
        let parsed = parse(&valid_value()).expect("valid Nano agent config");
        assert_eq!(parsed.control().runtime_queue_capacity().get(), 8);
        assert_eq!(parsed.inventory().artifact_bindings().len(), 2);
        assert!(parsed.eye_enabled());
        assert!(parsed.head_enabled());
        let expression = parsed
            .rgb_expression()
            .scene_motion()
            .expect("scene motion");
        assert_eq!(expression.scene_motion().geometry().sample_count(), 192);
        assert_eq!(expression.frame_freshness().as_nanos(), 80_000_000);
        assert_eq!(
            expression
                .gaze_geometry()
                .expect("explicit gaze geometry")
                .head_origin_in_camera_m(),
            [0.0, -0.25, -0.20]
        );
        assert_eq!(
            parsed.supervisor().maximum_authority_lease().as_nanos(),
            1_000_000_000
        );
        assert_eq!(
            parsed.live_mode_policy().startup(),
            NanoAgentStartupMode::DisarmedMapOnly
        );
        #[cfg(feature = "actuation")]
        {
            let manual = parsed
                .live_mode_policy()
                .manual()
                .config()
                .expect("manual control API policy");
            assert_eq!(manual.authority_lease().as_nanos(), 500_000_000);
            assert_eq!(manual.drive().maximum_abs_forward_velocity_mps(), 0.35);
            assert_eq!(manual.drive().maximum_abs_yaw_rate_rad_s(), 1.0);
            assert_eq!(manual.drive().maximum_command_age_ns(), 100_000_000);
            assert_eq!(manual.drive().deadman_timeout_ns(), 250_000_000);
        }
    }

    #[test]
    fn schema_v1_absence_disables_gaze_projection_without_assuming_extrinsics() {
        let mut value = valid_value();
        value["rgb_expression"]
            .as_object_mut()
            .expect("RGB expression object")
            .remove("gaze_geometry");

        let parsed = parse(&value).expect("legacy schema-v1 RGB policy remains compatible");
        let expression = parsed
            .rgb_expression()
            .scene_motion()
            .expect("scene motion remains enabled");
        assert_eq!(expression.gaze_geometry(), None);

        let mut value = valid_value();
        value["rgb_expression"]["gaze_geometry"] = Value::Null;
        assert!(matches!(
            parse(&value),
            Err(NanoAgentPolicyConfigParseError::JsonDecode(_))
        ));
    }

    #[test]
    fn configured_gaze_geometry_is_parsed_once_and_rejects_invalid_physical_domain() {
        let mut value = valid_value();
        value["rgb_expression"]["gaze_geometry"]["neutral_head_from_camera_quaternion_xyzw"] =
            json!([0.0, 0.0, 0.0, 0.0]);

        assert!(matches!(
            parse(&value),
            Err(NanoAgentPolicyConfigParseError::RgbGazeGeometry(
                GazeExtrinsicsParseError::DegenerateRotationQuaternion
            ))
        ));

        let mut value = valid_value();
        value["rgb_expression"]["gaze_geometry"]["head_origin_in_camera_m"] =
            json!([0.0, -25.0, -20.0]);
        assert!(matches!(
            parse(&value),
            Err(NanoAgentPolicyConfigParseError::RgbGazeGeometry(
                GazeExtrinsicsParseError::HeadOriginDistanceOutOfRange {
                    distance_m,
                    maximum_m,
                }
            )) if distance_m > maximum_m
        ));

        let mut value = valid_value();
        value["rgb_expression"]["gaze_geometry"]["schema_version"] = json!(2);
        assert!(matches!(
            parse(&value),
            Err(
                NanoAgentPolicyConfigParseError::UnsupportedRgbGazeGeometrySchemaVersion {
                    actual: 2,
                    supported: NANO_RGB_GAZE_GEOMETRY_V1,
                }
            )
        ));
    }

    #[test]
    fn exact_manifest_binding_is_the_only_actor_config_exposure() {
        let parsed = parse(&valid_value()).expect("valid Nano agent config");
        let bound = parsed
            .bind_accessories_to_manifest(&manifest(&manifest_value()))
            .expect("exact accessory binding");
        let eye = bound
            .eye()
            .static_runtime()
            .expect("bound static eye runtime");
        assert_eq!(
            eye.device().path(),
            "/dev/serial/by-id/usb-kiko_kiko-eyes_1-if00"
        );
        let head = bound.head().natural_hold().expect("bound head runtime");
        assert_eq!(
            head.runtime().device().path(),
            "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14031114-if00"
        );
        assert_eq!(
            head.torque_consent(),
            PhysicalTorqueEnableConsent::explicitly_granted()
        );
    }

    #[test]
    fn accessory_presence_must_match_the_manifest_in_both_directions() {
        let mut disabled_eye = valid_value();
        disabled_eye["eye"] = json!({"mode": "disabled"});
        disabled_eye["rgb_expression"] = json!({"mode": "disabled"});
        let error = parse(&disabled_eye)
            .expect("internally valid disabled eye")
            .bind_accessories_to_manifest(&manifest(&manifest_value()))
            .expect_err("manifest expects eye");
        assert_eq!(
            error,
            NanoAccessoryManifestBindingError::PresenceMismatch {
                accessory: NanoAccessoryKind::Eye,
                runtime_enabled: false,
                manifest_expected: true,
            }
        );

        let mut no_manifest_eye = manifest_value();
        no_manifest_eye["eye"] = Value::Null;
        let error = parse(&valid_value())
            .expect("enabled eye config")
            .bind_accessories_to_manifest(&manifest(&no_manifest_eye))
            .expect_err("manifest omits eye");
        assert_eq!(
            error,
            NanoAccessoryManifestBindingError::PresenceMismatch {
                accessory: NanoAccessoryKind::Eye,
                runtime_enabled: true,
                manifest_expected: false,
            }
        );

        let mut disabled_head = valid_value();
        disabled_head["head"] = json!({"mode": "disabled"});
        let error = parse(&disabled_head)
            .expect("internally valid disabled head")
            .bind_accessories_to_manifest(&manifest(&manifest_value()))
            .expect_err("manifest expects head");
        assert_eq!(
            error,
            NanoAccessoryManifestBindingError::PresenceMismatch {
                accessory: NanoAccessoryKind::Head,
                runtime_enabled: false,
                manifest_expected: true,
            }
        );
    }

    #[test]
    fn accessory_serial_uid_and_build_mismatches_fail_before_actor_exposure() {
        let expected = manifest(&manifest_value());

        for (accessory, path) in [
            (NanoAccessoryKind::Eye, "/dev/serial/by-id/usb-other-eye"),
            (NanoAccessoryKind::Head, "/dev/serial/by-id/usb-other-head"),
        ] {
            let mut value = valid_value();
            let field = match accessory {
                NanoAccessoryKind::Eye => "eye",
                NanoAccessoryKind::Head => "head",
            };
            value[field]["device_path"] = json!(path);
            assert!(matches!(
                parse(&value)
                    .expect("internally valid alternate serial")
                    .bind_accessories_to_manifest(&expected),
                Err(NanoAccessoryManifestBindingError::SerialPathMismatch {
                    accessory: actual,
                    ..
                }) if actual == accessory
            ));
        }

        let mut wrong_uid = valid_value();
        wrong_uid["eye"]["expected_device_uid"] = json!(vec![9_u8; 16]);
        assert!(matches!(
            parse(&wrong_uid)
                .expect("internally valid alternate UID")
                .bind_accessories_to_manifest(&expected),
            Err(NanoAccessoryManifestBindingError::EyeDeviceUidMismatch { .. })
        ));

        let mut wrong_build = valid_value();
        wrong_build["eye"]["expected_firmware_build_id"] = json!(vec![9_u8; 32]);
        assert!(matches!(
            parse(&wrong_build)
                .expect("internally valid alternate build")
                .bind_accessories_to_manifest(&expected),
            Err(NanoAccessoryManifestBindingError::EyeFirmwareBuildMismatch { .. })
        ));
    }

    #[test]
    fn manifest_parser_prevents_protocol_capability_and_head_contract_drift() {
        let mut wrong_protocol = manifest_value();
        wrong_protocol["eye"]["kep_protocol_version"] = json!(EYE_PROTOCOL_VERSION + 1);
        assert!(
            kiko_device_inventory::load_expected_manifest_v1_from_slice(
                &serde_json::to_vec(&wrong_protocol).expect("protocol fixture")
            )
            .is_err()
        );

        let mut wrong_capabilities = manifest_value();
        wrong_capabilities["eye"]["capabilities_bits"] =
            json!(REQUIRED_EYE_CAPABILITIES & !(1_u32 << 0));
        assert!(
            kiko_device_inventory::load_expected_manifest_v1_from_slice(
                &serde_json::to_vec(&wrong_capabilities).expect("capability fixture")
            )
            .is_err()
        );

        let mut wrong_head_electrical = manifest_value();
        wrong_head_electrical["head"]["baud_rate_bps"] = json!(BUS_BAUD_RATE_BPS / 2);
        assert!(
            kiko_device_inventory::load_expected_manifest_v1_from_slice(
                &serde_json::to_vec(&wrong_head_electrical).expect("head electrical fixture")
            )
            .is_err()
        );

        let mut wrong_head_servos = manifest_value();
        wrong_head_servos["head"]["bow_servo_id"] =
            wrong_head_servos["head"]["curl_servo_id"].clone();
        assert!(
            kiko_device_inventory::load_expected_manifest_v1_from_slice(
                &serde_json::to_vec(&wrong_head_servos).expect("head servo fixture")
            )
            .is_err()
        );
    }

    #[test]
    fn duplicate_unknown_missing_and_trailing_json_are_rejected() {
        let canonical = serde_json::to_string(&valid_value()).expect("fixture JSON");
        let duplicate = canonical.replacen(
            "\"schema_version\":1",
            "\"schema_version\":1,\"schema_version\":1",
            1,
        );
        assert!(matches!(
            NanoAgentPolicyConfigV1::parse_json(duplicate.as_bytes()),
            Err(NanoAgentPolicyConfigParseError::JsonDecode(_))
        ));

        let mut unknown = valid_value();
        unknown
            .as_object_mut()
            .expect("object")
            .insert("surprise".to_owned(), json!(true));
        assert!(matches!(
            parse(&unknown),
            Err(NanoAgentPolicyConfigParseError::JsonDecode(_))
        ));

        let mut missing = valid_value();
        missing.as_object_mut().expect("object").remove("head");
        assert!(matches!(
            parse(&missing),
            Err(NanoAgentPolicyConfigParseError::JsonDecode(_))
        ));

        let mut trailing = serde_json::to_vec(&valid_value()).expect("fixture JSON");
        trailing.extend_from_slice(b" true");
        assert!(matches!(
            NanoAgentPolicyConfigV1::parse_json(&trailing),
            Err(NanoAgentPolicyConfigParseError::JsonTrailingData(_))
        ));
    }

    #[test]
    fn static_eye_policy_rejects_persisted_session_material() {
        for (field, value) in [
            ("identity_nonce", json!(11)),
            ("acquire_nonce", json!(12)),
            ("control_epoch", json!(13)),
        ] {
            let mut document = valid_value();
            document["eye"]
                .as_object_mut()
                .expect("eye policy object")
                .insert(field.to_owned(), value);
            assert!(
                matches!(
                    parse(&document),
                    Err(NanoAgentPolicyConfigParseError::JsonDecode(_))
                ),
                "accepted persisted per-start field {field}"
            );
        }
    }

    #[test]
    fn input_and_all_path_classes_are_bounded_and_canonical() {
        let oversized = vec![b' '; MAX_NANO_AGENT_POLICY_CONFIG_JSON_BYTES + 1];
        assert!(matches!(
            NanoAgentPolicyConfigV1::parse_json(&oversized),
            Err(NanoAgentPolicyConfigParseError::InputTooLarge { .. })
        ));

        for invalid in [
            "relative/manifest.json",
            "/opt/kiko/../manifest.json",
            "/opt//kiko/manifest.json",
            "/opt/kiko/./manifest.json",
            "/",
        ] {
            let mut value = valid_value();
            value["inventory"]["manifest_path"] = json!(invalid);
            assert!(matches!(
                parse(&value),
                Err(NanoAgentPolicyConfigParseError::AbsolutePath {
                    field: NanoAgentPathField::Manifest,
                    ..
                })
            ));
        }

        let mut value = valid_value();
        value["map_persistence"]["save_snapshot_path"] =
            json!(format!("/{}", "a".repeat(MAX_NANO_AGENT_DATA_PATH_BYTES)));
        assert!(matches!(
            parse(&value),
            Err(NanoAgentPolicyConfigParseError::AbsolutePath {
                field: NanoAgentPathField::MapSaveSnapshot,
                source: NanoAbsolutePathError::TooLong { .. }
            })
        ));
    }

    #[test]
    fn artifact_bindings_reject_escape_duplicates_and_excess_counts() {
        for invalid in [
            "/absolute.json",
            "../escape.json",
            "calibration//stereo.json",
            "calibration\\stereo.json",
        ] {
            let mut value = valid_value();
            value["inventory"]["artifact_bindings"][0]["relative_path"] = json!(invalid);
            assert!(matches!(
                parse(&value),
                Err(NanoAgentPolicyConfigParseError::ArtifactBindings(
                    ArtifactFileBindingParseError::InvalidRelativePath { .. }
                ))
            ));
        }

        let mut duplicate = valid_value();
        duplicate["inventory"]["artifact_bindings"][1]["artifact_id"] = json!("stereo-v1");
        assert!(matches!(
            parse(&duplicate),
            Err(NanoAgentPolicyConfigParseError::ArtifactBindings(
                ArtifactFileBindingParseError::DuplicateArtifactId { .. }
            ))
        ));

        let mut too_many = valid_value();
        let bindings = too_many["inventory"]["artifact_bindings"]
            .as_array_mut()
            .expect("bindings");
        for index in 0..=MAX_PLANT_ARTIFACTS {
            bindings.push(json!({
                "kind": "plant",
                "artifact_id": format!("extra-plant-{index}"),
                "relative_path": format!("plant/extra-{index}.json")
            }));
        }
        assert!(matches!(
            parse(&too_many),
            Err(NanoAgentPolicyConfigParseError::ArtifactBindings(
                ArtifactFileBindingParseError::TooManyBindings {
                    kind: ArtifactKind::Plant,
                    ..
                }
            ))
        ));
    }

    #[test]
    fn accessory_and_expression_choices_are_consistent_and_explicit() {
        let mut no_eye = valid_value();
        no_eye["eye"] = json!({"mode": "disabled"});
        assert!(matches!(
            parse(&no_eye),
            Err(NanoAgentPolicyConfigParseError::RgbExpressionRequiresEye)
        ));

        let mut duplicate_serial = valid_value();
        duplicate_serial["head"]["device_path"] = duplicate_serial["eye"]["device_path"].clone();
        assert!(matches!(
            parse(&duplicate_serial),
            Err(NanoAgentPolicyConfigParseError::DuplicateAccessorySerialPath)
        ));

        let mut missing_consent = valid_value();
        missing_consent["head"]
            .as_object_mut()
            .expect("head")
            .remove("physical_torque_consent");
        assert!(matches!(
            parse(&missing_consent),
            Err(NanoAgentPolicyConfigParseError::JsonDecode(_))
        ));

        let mut too_stale = valid_value();
        too_stale["rgb_expression"]["frame_freshness_ms"] = json!(30);
        assert!(matches!(
            parse(&too_stale),
            Err(
                NanoAgentPolicyConfigParseError::RgbFreshnessDoesNotCoverEyeRoundTrip {
                    frame_freshness_ms: 30,
                    worst_case_eye_round_trip_ms: 30
                }
            )
        ));
    }

    #[test]
    fn supervisor_and_mode_leases_are_bounded_and_consistent() {
        let mut zero_age_too_long = valid_value();
        zero_age_too_long["supervisor"]["maximum_zero_age_ms"] = json!(1001);
        assert!(matches!(
            parse(&zero_age_too_long),
            Err(NanoAgentPolicyConfigParseError::SupervisorConfig(_))
        ));

        #[cfg(feature = "actuation")]
        {
            let mut mode_too_long = valid_value();
            mode_too_long["live_mode_policy"]["manual"]["authority_lease_ms"] = json!(1001);
            assert!(matches!(
                parse(&mode_too_long),
                Err(
                    NanoAgentPolicyConfigParseError::ModeAuthorityLeaseExceedsSupervisor {
                        mode: NanoConfiguredMotionMode::Manual,
                        ..
                    }
                )
            ));
        }

        let mut unbounded = valid_value();
        unbounded["supervisor"]["maximum_authority_lease_ms"] =
            json!(MAX_NANO_AGENT_AUTHORITY_LEASE_MS + 1);
        assert!(matches!(
            parse(&unbounded),
            Err(NanoAgentPolicyConfigParseError::SupervisorDuration {
                source: NanoBoundedMillisecondsError::TooLarge { .. },
                ..
            })
        ));
    }

    #[cfg(feature = "actuation")]
    #[test]
    fn manual_envelope_age_and_deadman_are_explicit_and_fail_closed() {
        let mut value = valid_value();
        value["live_mode_policy"]["manual"]["maximum_abs_forward_velocity_mps"] = json!(0.0);
        assert!(matches!(
            parse(&value),
            Err(NanoAgentPolicyConfigParseError::ManualDrive(
                ManualDriveConfigParseError::NonPositiveLimit {
                    field: "maximum_abs_forward_velocity_mps",
                    ..
                }
            ))
        ));

        let mut value = valid_value();
        value["live_mode_policy"]["manual"]["maximum_abs_yaw_rate_rad_s"] =
            json!(super::super::reference::MAX_SUPPORTED_ABS_REFERENCE_YAW_RATE_RAD_S * 2.0);
        assert!(matches!(
            parse(&value),
            Err(NanoAgentPolicyConfigParseError::ManualDrive(
                ManualDriveConfigParseError::AboveSupportedLimit {
                    field: "maximum_abs_yaw_rate_rad_s",
                    ..
                }
            ))
        ));

        let mut value = valid_value();
        value["live_mode_policy"]["manual"]["maximum_command_age_ms"] = json!(251);
        value["live_mode_policy"]["manual"]["deadman_timeout_ms"] = json!(250);
        assert!(matches!(
            parse(&value),
            Err(NanoAgentPolicyConfigParseError::ManualDrive(
                ManualDriveConfigParseError::CommandAgeExceedsDeadman { .. }
            ))
        ));

        for field in ["maximum_command_age_ms", "deadman_timeout_ms"] {
            let mut zero = valid_value();
            zero["live_mode_policy"]["manual"][field] = json!(0);
            assert!(matches!(
                parse(&zero),
                Err(NanoAgentPolicyConfigParseError::ManualCommandAge {
                    source: NanoBoundedMillisecondsError::Zero
                }) | Err(NanoAgentPolicyConfigParseError::ManualDeadman {
                    source: NanoBoundedMillisecondsError::Zero
                })
            ));

            let mut unbounded = valid_value();
            unbounded["live_mode_policy"]["manual"][field] =
                json!(MAX_NANO_AGENT_MANUAL_WINDOW_MS + 1);
            assert!(matches!(
                parse(&unbounded),
                Err(NanoAgentPolicyConfigParseError::ManualCommandAge {
                    source: NanoBoundedMillisecondsError::TooLarge { .. }
                }) | Err(NanoAgentPolicyConfigParseError::ManualDeadman {
                    source: NanoBoundedMillisecondsError::TooLarge { .. }
                })
            ));
        }

        let disabled = json!({"permission": "disabled"});
        let mut value = valid_value();
        value["live_mode_policy"]["manual"] = disabled;
        assert!(matches!(
            parse(&value)
                .expect("disabled manual policy needs no dormant limits")
                .live_mode_policy()
                .manual(),
            NanoManualModePolicy::Disabled
        ));
    }

    #[cfg(feature = "actuation")]
    #[test]
    fn manual_policy_binds_the_whole_body_twist_box_to_wheel_limits() {
        let value = valid_value();
        let manual = parse(&value)
            .expect("manual policy")
            .live_mode_policy()
            .manual()
            .config()
            .expect("manual enabled");
        assert!(matches!(
            manual.bind_to_plant(bounded_manual_plant()),
            Err(
                NanoManualPlantBindingError::CombinedBodyVelocityOutsidePlantEnvelope {
                    configured_forward_mps,
                    configured_yaw_rate_rad_s,
                    wheelbase_m,
                    required_abs_wheel_velocity_mps,
                    supported_abs_wheel_velocity_mps,
                }
            ) if configured_forward_mps == 0.35
                && configured_yaw_rate_rad_s == 1.0
                && wheelbase_m == 0.4
                && required_abs_wheel_velocity_mps > 0.5
                && supported_abs_wheel_velocity_mps == 0.5
        ));

        let mut value = valid_value();
        value["live_mode_policy"]["manual"]["maximum_abs_forward_velocity_mps"] = json!(0.25);
        let manual = parse(&value)
            .expect("reduced manual policy")
            .live_mode_policy()
            .manual()
            .config()
            .expect("manual enabled");
        assert!(manual.bind_to_plant(bounded_manual_plant()).is_ok());
    }

    #[cfg(feature = "actuation")]
    #[test]
    fn exploration_requires_finite_ordered_bounds_and_finite_resources() {
        let mut reversed = valid_value();
        reversed["live_mode_policy"]["frontier_explore"]["boundary_minimum_x_m"] = json!(20.0);
        assert!(matches!(
            parse(&reversed),
            Err(NanoAgentPolicyConfigParseError::ExploreBoundary {
                source: NanoExploreBoundaryError::EmptyOrReversedX { .. }
            })
        ));

        let mut zero_goals = valid_value();
        zero_goals["live_mode_policy"]["frontier_explore"]["maximum_frontier_goals"] = json!(0);
        assert!(matches!(
            parse(&zero_goals),
            Err(NanoAgentPolicyConfigParseError::ExploreGoalCount { actual: 0, .. })
        ));

        let mut too_long = valid_value();
        too_long["live_mode_policy"]["frontier_explore"]["maximum_runtime_ms"] =
            json!(MAX_NANO_AGENT_EXPLORE_RUNTIME_MS + 1);
        assert!(matches!(
            parse(&too_long),
            Err(NanoAgentPolicyConfigParseError::ExploreRuntime { .. })
        ));
    }

    #[test]
    fn exploration_boundary_domain_constructor_rejects_invalid_rectangles_once() {
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(matches!(
                NanoExploreBoundaryMeters::try_new(invalid, -1.0, 1.0, 1.0),
                Err(NanoExploreBoundaryError::NonFinite {
                    component: NanoExploreBoundaryComponent::MinimumX,
                    ..
                })
            ));
        }
        assert!(matches!(
            NanoExploreBoundaryMeters::try_new(-1.0, -1.0, -1.0, 1.0),
            Err(NanoExploreBoundaryError::EmptyOrReversedX { .. })
        ));
        assert!(matches!(
            NanoExploreBoundaryMeters::try_new(-1.0, 1.0, 1.0, 1.0),
            Err(NanoExploreBoundaryError::EmptyOrReversedY { .. })
        ));
        assert!(matches!(
            NanoExploreBoundaryMeters::try_new(
                -1.0,
                -1.0,
                MAX_NANO_AGENT_ABS_EXPLORE_BOUNDARY_M + 1.0,
                1.0,
            ),
            Err(NanoExploreBoundaryError::OutsideSupportedRange {
                component: NanoExploreBoundaryComponent::MaximumX,
                ..
            })
        ));
        let parsed =
            NanoExploreBoundaryMeters::try_new(-2.0, -1.0, 3.0, 4.0).expect("valid rectangle");
        assert_eq!(
            (
                parsed.minimum_x_m(),
                parsed.minimum_y_m(),
                parsed.maximum_x_m(),
                parsed.maximum_y_m(),
            ),
            (-2.0, -1.0, 3.0, 4.0)
        );
    }

    #[cfg(not(feature = "actuation"))]
    #[test]
    fn motion_policies_require_an_actuation_capable_build() {
        for (field, mode) in [("point_goal", NanoConfiguredMotionMode::PointGoal)] {
            let mut value = valid_value();
            value["live_mode_policy"][field] = json!({
                "permission": "control_api",
                "authority_lease_ms": 500
            });
            assert!(matches!(
                parse(&value),
                Err(
                    NanoAgentPolicyConfigParseError::MotionPolicyRequiresActuationFeature {
                        mode: actual
                    }
                ) if actual == mode
            ));
        }

        let mut value = valid_value();
        value["live_mode_policy"]["manual"] = json!({
            "permission": "control_api",
            "authority_lease_ms": 500,
            "maximum_abs_forward_velocity_mps": 0.35,
            "maximum_abs_yaw_rate_rad_s": 1.0,
            "maximum_command_age_ms": 100,
            "deadman_timeout_ms": 250
        });
        assert!(matches!(
            parse(&value),
            Err(
                NanoAgentPolicyConfigParseError::MotionPolicyRequiresActuationFeature {
                    mode: NanoConfiguredMotionMode::Manual
                }
            )
        ));

        let mut value = valid_value();
        value["live_mode_policy"]["frontier_explore"] = json!({
            "permission": "control_api",
            "authority_lease_ms": 500,
            "boundary_minimum_x_m": -1.0,
            "boundary_minimum_y_m": -1.0,
            "boundary_maximum_x_m": 1.0,
            "boundary_maximum_y_m": 1.0,
            "maximum_runtime_ms": 1000,
            "maximum_frontier_goals": 1
        });
        assert!(matches!(
            parse(&value),
            Err(
                NanoAgentPolicyConfigParseError::MotionPolicyRequiresActuationFeature {
                    mode: NanoConfiguredMotionMode::FrontierExplore
                }
            )
        ));
    }

    #[test]
    fn warm_start_cannot_claim_relocalization_from_occupancy_alone() {
        let mut missing_dataset = valid_value();
        missing_dataset["map_persistence"]["warm_start"]
            .as_object_mut()
            .expect("warm start")
            .remove("slam_dataset_directory_path");
        assert!(matches!(
            parse(&missing_dataset),
            Err(NanoAgentPolicyConfigParseError::JsonDecode(_))
        ));

        let mut role_collision = valid_value();
        role_collision["map_persistence"]["warm_start"]["slam_dataset_directory_path"] =
            role_collision["map_persistence"]["save_snapshot_path"].clone();
        assert!(matches!(
            parse(&role_collision),
            Err(NanoAgentPolicyConfigParseError::WarmStartPathRoleCollision)
        ));
    }
}
