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
//! RGB scene motion can own only KEP2 eyes. An enabled head policy is one
//! config-bound return to Kiko's reviewed natural target followed by an active
//! hold against that exact target; an observed startup pose is never
//! re-labelled as natural.

use std::fmt;
use std::net::{AddrParseError, SocketAddr};
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
#[cfg(test)]
use kiko_head_runtime::ConfiguredHeadPoseBoundsError;
use kiko_head_runtime::{
    ConfigParseError as HeadConfigParseError, HeadHoldTarget, HeadProbeConfig,
    HeadProbeConfigInput, HeadRuntimeConfig, PhysicalHeadMotionConsent,
    PhysicalTorqueEnableConsent, ReturnToTargetConfig, ReturnToTargetConfigInput,
    ReturnToTargetConfigParseError,
};
use kiko_supervisor_core::{
    AuthorityDuration, SupervisorConfig, SupervisorConfigError, TimeValueError,
};
use serde::Deserialize;

use super::mpc::{BoundedId, PlantModelV1};
use super::{
    AgentControlRuntimeQueueCapacity, AgentControlRuntimeQueueCapacityError,
    AgentControlSocketConfig, AgentControlSocketPath, AgentControlSocketPathError,
    AgentControlSocketTimeoutError, AgentControlSocketTimeouts, FrontierExplorerConfig,
    FrontierExplorerConfigError, FrontierYawScanBudgetError, FrontierYawScanBudgetV1,
    FrontierYawTurnDirectionV1, MANUAL_DRIVE_CONFIG_V1, MAX_NANO_OCCUPANCY_CELLS,
    ManualDriveConfigParseError, ManualDriveConfigV1, ManualDriveConfigV1Dto,
};

/// The only supported Nano-agent policy-component schema.
///
/// Version 1 described the retired observed-pose hold policy. It is never
/// reinterpreted as this config-bound return-and-continuous-hold contract.
/// Version 2 did not configure the production operator console. It is rejected
/// rather than silently acquiring a network listener, capability-file path, or
/// deadman schedule from runtime defaults.
pub const NANO_AGENT_POLICY_CONFIG_V3: u32 = 3;

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

/// Maximum total wall-clock budget for one selected point-goal request.
pub const MAX_NANO_AGENT_POINT_GOAL_RUNTIME_MS: u64 = 24 * 60 * 60 * 1_000;

/// Maximum freshness window for one RGB observation.
pub const MAX_NANO_AGENT_RGB_FRAME_FRESHNESS_MS: u64 = 5_000;

/// Maximum total wall-clock resource budget for one exploration request.
pub const MAX_NANO_AGENT_EXPLORE_RUNTIME_MS: u64 = 24 * 60 * 60 * 1_000;

/// Maximum number of frontier goals admitted for one exploration request.
pub const MAX_NANO_AGENT_EXPLORE_GOALS: u32 = 10_000;

/// Maximum absolute map coordinate admitted for an exploration boundary.
pub const MAX_NANO_AGENT_ABS_EXPLORE_BOUNDARY_M: f64 = 10_000.0;

/// Maximum UTF-8 bytes in the per-boot operator-console capability path.
pub const MAX_NANO_OPERATOR_CONSOLE_CAPABILITY_PATH_BYTES: usize = 1_024;

/// Tightest admitted operator-console deadman scheduler period.
pub const MIN_NANO_OPERATOR_CONSOLE_DEADMAN_TICK_MS: u64 = 5;

/// Loosest admitted operator-console deadman scheduler period.
pub const MAX_NANO_OPERATOR_CONSOLE_DEADMAN_TICK_MS: u64 = 100;

/// The operator-confirmed neutral target selected by the superseding
/// 2026-08-06 replacement-bow-servo head-policy evidence.
///
/// Values are raw bow/curl/yaw/roll encoder ticks from the operator's
/// hand-placed standing balance after the bow servo transplant. The attended
/// session observed a zero-jump engagement and zero load on all four joints.
/// This does not establish mechanical joint limits or qualify unattended
/// return motion.
pub const KIKO_REVIEWED_NATURAL_HEAD_TARGET_TICKS: [u16; 4] = [1_505, 3_937, 1_551, 3_018];

/// Lower bound of the current attended commissioning start window.
pub const KIKO_REVIEWED_NATURAL_HEAD_START_MINIMUM_TICKS: [u16; 4] = [1_377, 3_809, 1_423, 2_890];

/// Upper bound of the current attended commissioning start window.
pub const KIKO_REVIEWED_NATURAL_HEAD_START_MAXIMUM_TICKS: [u16; 4] = [1_633, 4_065, 1_679, 3_146];

/// Exact software travel caps for the superseding natural-return policy.
///
/// Each cap exactly covers the current symmetric 128-tick start window. It is
/// a software admission bound, not a mechanical travel limit.
pub const KIKO_REVIEWED_NATURAL_HEAD_MAXIMUM_TRAVEL_TICKS: [u16; 4] = [128; 4];

/// Current field holding torque for the replacement-servo assembly.
///
/// Values are bow/curl/yaw/roll permille of each servo's configured maximum.
pub const KIKO_REVIEWED_NATURAL_HEAD_TORQUE_LIMIT_PERMILLE: [u16; 4] = [650, 550, 400, 400];

/// A fully parsed runtime policy component. Construction is possible only
/// through [`Self::parse_json`].
#[derive(Clone, Debug, PartialEq)]
pub struct NanoAgentPolicyConfigV3 {
    control: NanoAgentControlConfig,
    inventory: NanoAgentInventoryConfig,
    map_persistence: NanoMapPersistenceConfig,
    eye: ParsedNanoEyePolicy,
    head: ParsedNanoHeadPolicy,
    rgb_expression: NanoRgbExpressionPolicy,
    supervisor: SupervisorConfig,
    live_mode_policy: NanoLiveModePolicy,
}

impl NanoAgentPolicyConfigV3 {
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
        let dto = NanoAgentPolicyConfigV3Dto::deserialize(&mut deserializer)
            .map_err(NanoAgentPolicyConfigParseError::JsonDecode)?;
        deserializer
            .end()
            .map_err(NanoAgentPolicyConfigParseError::JsonTrailingData)?;
        if dto.schema_version != NANO_AGENT_POLICY_CONFIG_V3 {
            return Err(NanoAgentPolicyConfigParseError::UnsupportedSchemaVersion {
                actual: dto.schema_version,
                supported: NANO_AGENT_POLICY_CONFIG_V3,
            });
        }

        let control = parse_control(dto.control)?;
        let inventory = parse_inventory(dto.inventory)?;
        let map_persistence = parse_map_persistence(dto.map_persistence)?;
        let supervisor = parse_supervisor(dto.supervisor)?;
        let eye = parse_eye(dto.eye)?;
        let head = parse_head(dto.head)?;

        if let (
            ParsedNanoEyePolicy::Kep2(eye),
            ParsedNanoHeadPolicy::ReturnToNaturalAndHoldContinuously(head),
        ) = (&eye, &head)
            && eye.device().path() == head.runtime().device().path()
        {
            return Err(NanoAgentPolicyConfigParseError::DuplicateAccessorySerialPath);
        }

        let rgb_expression = parse_rgb_expression(dto.rgb_expression, &eye)?;
        let live_mode_policy = parse_live_mode_policy(dto.live_mode_policy, supervisor)?;
        if let Some(manual) = live_mode_policy.manual().config() {
            let drive = manual.drive();
            let requested_forward = control
                .operator_console()
                .manual_command_forward_velocity_mps();
            let requested_yaw = control.operator_console().manual_command_yaw_rate_rad_s();
            if requested_forward > drive.maximum_abs_forward_velocity_mps()
                || requested_yaw > drive.maximum_abs_yaw_rate_rad_s()
            {
                return Err(
                    NanoAgentPolicyConfigParseError::OperatorConsoleManualCommandOutsideEnvelope {
                        requested_forward_velocity_mps: requested_forward,
                        maximum_forward_velocity_mps: drive.maximum_abs_forward_velocity_mps(),
                        requested_yaw_rate_rad_s: requested_yaw,
                        maximum_yaw_rate_rad_s: drive.maximum_abs_yaw_rate_rad_s(),
                    },
                );
            }
        }

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
        matches!(
            self.head,
            ParsedNanoHeadPolicy::ReturnToNaturalAndHoldContinuously(_)
        )
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
    ) -> Result<ManifestBoundNanoAgentPolicyConfigV3, NanoAccessoryManifestBindingError> {
        let eye = bind_eye_to_manifest(self.eye, manifest)?;
        let head = bind_head_to_manifest(self.head, manifest)?;
        Ok(ManifestBoundNanoAgentPolicyConfigV3 {
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
pub struct ManifestBoundNanoAgentPolicyConfigV3 {
    control: NanoAgentControlConfig,
    inventory: NanoAgentInventoryConfig,
    map_persistence: NanoMapPersistenceConfig,
    eye: NanoManifestBoundEyePolicy,
    head: NanoManifestBoundHeadPolicy,
    rgb_expression: NanoRgbExpressionPolicy,
    supervisor: SupervisorConfig,
    live_mode_policy: NanoLiveModePolicy,
}

impl ManifestBoundNanoAgentPolicyConfigV3 {
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

    pub fn into_parts(self) -> NanoAgentPolicyPartsV3 {
        NanoAgentPolicyPartsV3 {
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
pub struct NanoAgentPolicyPartsV3 {
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
    operator_console: NanoOperatorConsoleConfig,
}

impl NanoAgentControlConfig {
    pub const fn socket(&self) -> &AgentControlSocketConfig {
        &self.socket
    }

    pub const fn runtime_queue_capacity(&self) -> AgentControlRuntimeQueueCapacity {
        self.runtime_queue_capacity
    }

    pub const fn operator_console(&self) -> &NanoOperatorConsoleConfig {
        &self.operator_console
    }

    pub fn into_parts(
        self,
    ) -> (
        AgentControlSocketConfig,
        AgentControlRuntimeQueueCapacity,
        NanoOperatorConsoleConfig,
    ) {
        (
            self.socket,
            self.runtime_queue_capacity,
            self.operator_console,
        )
    }
}

/// Parsed production operator-console transport and deadman schedule.
///
/// The capability file is constrained to the exact lexical parent of the
/// control socket so startup can admit one private runtime directory for both
/// local-control identities. Parsing does not claim that parent exists, is
/// symlink-free, or has safe ownership/mode; the runtime must prove those
/// filesystem properties before either endpoint becomes live.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NanoOperatorConsoleConfig {
    bind_address: SocketAddr,
    capability_path: NanoConfiguredAbsolutePath,
    deadman_tick: Duration,
    manual_command_forward_mm_per_s: NonZeroU32,
    manual_command_yaw_millirad_per_s: NonZeroU32,
}

impl NanoOperatorConsoleConfig {
    pub const fn bind_address(&self) -> SocketAddr {
        self.bind_address
    }

    pub const fn capability_path(&self) -> &NanoConfiguredAbsolutePath {
        &self.capability_path
    }

    pub const fn deadman_tick(&self) -> Duration {
        self.deadman_tick
    }

    /// Body-forward arrow/WASD command magnitude, converted exactly once from
    /// the integer millimetres-per-second JSON boundary into SI.
    pub fn manual_command_forward_velocity_mps(&self) -> f64 {
        f64::from(self.manual_command_forward_mm_per_s.get()) / 1_000.0
    }

    /// Arrow/WASD yaw command magnitude, converted exactly once from the
    /// integer milliradians-per-second JSON boundary into SI.
    pub fn manual_command_yaw_rate_rad_s(&self) -> f64 {
        f64::from(self.manual_command_yaw_millirad_per_s.get()) / 1_000.0
    }

    #[cfg(all(test, feature = "nano-agent", unix))]
    pub(crate) fn for_test(
        bind_address: SocketAddr,
        capability_path: PathBuf,
        deadman_tick: Duration,
    ) -> Self {
        Self {
            bind_address,
            capability_path: NanoConfiguredAbsolutePath(capability_path),
            deadman_tick,
            manual_command_forward_mm_per_s: NonZeroU32::new(100)
                .expect("static test command is nonzero"),
            manual_command_yaw_millirad_per_s: NonZeroU32::new(100)
                .expect("static test command is nonzero"),
        }
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

/// One config-bound return to Kiko's reviewed natural target followed by
/// continuous actor ownership until coordinated shutdown or fault.
///
/// No hold duration or lease exists in this production type. The two
/// independent physical consents authorize the initial torque and motion.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NanoContinuousNaturalHeadHoldConfig {
    return_to_target: ReturnToTargetConfig,
    torque_consent: PhysicalTorqueEnableConsent,
    motion_consent: PhysicalHeadMotionConsent,
}

impl NanoContinuousNaturalHeadHoldConfig {
    /// Runtime transport/tuning retained inside the inseparable return plan.
    ///
    /// This accessor exists for read-only admission probes; callers cannot
    /// reconstruct a different target, bounds, or travel plan from it.
    pub const fn runtime(&self) -> &HeadRuntimeConfig {
        self.return_to_target.runtime()
    }

    pub const fn return_config(&self) -> &ReturnToTargetConfig {
        &self.return_to_target
    }

    /// Typed target which post-return health evidence must report.
    pub const fn required_hold_target(&self) -> HeadHoldTarget {
        HeadHoldTarget::ReviewedReturn(self.return_to_target.target())
    }

    pub const fn torque_consent(&self) -> PhysicalTorqueEnableConsent {
        self.torque_consent
    }

    pub const fn motion_consent(&self) -> PhysicalHeadMotionConsent {
        self.motion_consent
    }

    pub fn into_parts(
        self,
    ) -> (
        ReturnToTargetConfig,
        PhysicalTorqueEnableConsent,
        PhysicalHeadMotionConsent,
    ) {
        (
            self.return_to_target,
            self.torque_consent,
            self.motion_consent,
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum ParsedNanoHeadPolicy {
    Disabled,
    ReturnToNaturalAndHoldContinuously(NanoContinuousNaturalHeadHoldConfig),
}

/// Manifest-bound head selection. There is no observed-pose-only or
/// expressive-offset variant. The return plan is exposed only after exact
/// manifest agreement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NanoManifestBoundHeadPolicy {
    Disabled,
    ReturnToNaturalAndHoldContinuously(NanoContinuousNaturalHeadHoldConfig),
}

impl NanoManifestBoundHeadPolicy {
    pub const fn return_to_natural_and_hold_continuously(
        &self,
    ) -> Option<&NanoContinuousNaturalHeadHoldConfig> {
        match self {
            Self::Disabled => None,
            Self::ReturnToNaturalAndHoldContinuously(config) => Some(config),
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
        (ParsedNanoHeadPolicy::ReturnToNaturalAndHoldContinuously(_), None) => {
            Err(NanoAccessoryManifestBindingError::PresenceMismatch {
                accessory: NanoAccessoryKind::Head,
                runtime_enabled: true,
                manifest_expected: false,
            })
        }
        (ParsedNanoHeadPolicy::ReturnToNaturalAndHoldContinuously(config), Some(expected)) => {
            if config.runtime().device().path() != expected.serial_path().as_str() {
                return Err(NanoAccessoryManifestBindingError::SerialPathMismatch {
                    accessory: NanoAccessoryKind::Head,
                    runtime_path: config.runtime().device().path().into(),
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
            Ok(NanoManifestBoundHeadPolicy::ReturnToNaturalAndHoldContinuously(config))
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
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NanoMotionModePolicy {
    Disabled,
    ControlApi {
        authority_lease: AuthorityDuration,
        maximum_runtime: Duration,
        arrival_tolerance_m: f64,
    },
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
            Self::ControlApi {
                authority_lease, ..
            } => Some(authority_lease),
        }
    }

    pub const fn maximum_runtime(self) -> Option<Duration> {
        match self {
            Self::Disabled => None,
            Self::ControlApi {
                maximum_runtime, ..
            } => Some(maximum_runtime),
        }
    }

    pub const fn arrival_tolerance_m(self) -> Option<f64> {
        match self {
            Self::Disabled => None,
            Self::ControlApi {
                arrival_tolerance_m,
                ..
            } => Some(arrival_tolerance_m),
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
    arrival_tolerance_m: f64,
    explorer: FrontierExplorerConfig,
    yaw_scan_budget: FrontierYawScanBudgetV1,
    yaw_turn_direction: FrontierYawTurnDirectionV1,
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

    pub const fn arrival_tolerance_m(self) -> f64 {
        self.arrival_tolerance_m
    }

    pub const fn explorer(self) -> FrontierExplorerConfig {
        self.explorer
    }

    pub const fn yaw_scan_budget(self) -> FrontierYawScanBudgetV1 {
        self.yaw_scan_budget
    }

    pub const fn yaw_turn_direction(self) -> FrontierYawTurnDirectionV1 {
        self.yaw_turn_direction
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

    #[cfg(all(test, feature = "actuation"))]
    pub(crate) fn autonomous_for_test(
        authority_lease: AuthorityDuration,
        point_goal_maximum_runtime: Duration,
        arrival_tolerance_m: f64,
        maximum_frontier_goals: NonZeroU32,
    ) -> Self {
        Self {
            startup: NanoAgentStartupMode::DisarmedMapOnly,
            manual: NanoManualModePolicy::Disabled,
            point_goal: NanoMotionModePolicy::ControlApi {
                authority_lease,
                maximum_runtime: point_goal_maximum_runtime,
                arrival_tolerance_m,
            },
            frontier_explore: NanoFrontierExplorePolicy::ControlApi(NanoFrontierExploreConfig {
                authority_lease,
                boundary_m: NanoExploreBoundaryMeters::try_new(-5.0, -5.0, 5.0, 5.0)
                    .expect("test exploration boundary"),
                maximum_runtime: Duration::from_secs(60),
                maximum_frontier_goals,
                arrival_tolerance_m,
                explorer: FrontierExplorerConfig::try_new(0.0, 1_024, 1_024, 8_192)
                    .expect("test frontier resources"),
                yaw_scan_budget: FrontierYawScanBudgetV1::try_new(
                    1.0,
                    std::f64::consts::TAU,
                    0.1,
                    5_000_000_000,
                )
                .expect("test frontier yaw budget"),
                yaw_turn_direction: FrontierYawTurnDirectionV1::CounterClockwise,
            }),
        }
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
    OperatorConsoleBindAddress {
        source: AddrParseError,
    },
    OperatorConsoleBindAddressNotLoopback {
        actual: SocketAddr,
    },
    OperatorConsoleBindPortZero {
        actual: SocketAddr,
    },
    OperatorConsoleDeadmanTick {
        source: NanoOperatorConsoleDeadmanTickError,
    },
    OperatorConsoleManualCommandZero {
        field: &'static str,
    },
    OperatorConsoleManualCommandOutsideEnvelope {
        requested_forward_velocity_mps: f64,
        maximum_forward_velocity_mps: f64,
        requested_yaw_rate_rad_s: f64,
        maximum_yaw_rate_rad_s: f64,
    },
    OperatorConsoleCapabilityPathCollidesWithControlSocket,
    OperatorConsoleCapabilityPathOutsideControlSocketParent,
    AbsolutePath {
        field: NanoAgentPathField,
        source: NanoAbsolutePathError,
    },
    ArtifactBindings(ArtifactFileBindingParseError),
    WarmStartPathRoleCollision,
    Eye(EyeConfigParseError),
    HeadProbe(HeadConfigParseError),
    HeadReturn(ReturnToTargetConfigParseError),
    ReviewedNaturalHeadTargetMismatch {
        configured_ticks: [u16; 4],
        required_ticks: [u16; 4],
    },
    ReviewedNaturalHoldEnvelopeOutsideStartupWindow {
        joint: HeadJoint,
        configured_minimum_ticks: u16,
        configured_maximum_ticks: u16,
        required_minimum_ticks: u16,
        required_maximum_ticks: u16,
    },
    ReviewedNaturalHeadStartBoundsMismatch {
        configured_minimum_ticks: [u16; 4],
        configured_maximum_ticks: [u16; 4],
        required_minimum_ticks: [u16; 4],
        required_maximum_ticks: [u16; 4],
    },
    ReviewedNaturalHeadMaximumTravelMismatch {
        configured_ticks: [u16; 4],
        required_ticks: [u16; 4],
    },
    ReviewedNaturalHeadTorqueMismatch {
        configured_permille: [u16; 4],
        required_permille: [u16; 4],
    },
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
    GoalArrivalTolerance {
        mode: NanoConfiguredMotionMode,
        value_m: f64,
    },
    PointGoalRuntime {
        source: NanoBoundedMillisecondsError,
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
    ExploreGridCellLimit {
        actual: usize,
        maximum: usize,
    },
    ExploreResources(FrontierExplorerConfigError),
    ExploreYawScanDuration {
        source: NanoBoundedMillisecondsError,
    },
    ExploreYawScanBudget(FrontierYawScanBudgetError),
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
            Self::OperatorConsoleBindAddress { source } => Some(source),
            Self::OperatorConsoleDeadmanTick { source } => Some(source),
            Self::AbsolutePath { source, .. } => Some(source),
            Self::ArtifactBindings(source) => Some(source),
            Self::Eye(source) => Some(source),
            Self::HeadProbe(source) => Some(source),
            Self::HeadReturn(source) => Some(source),
            Self::RgbSamplingGeometry(source) => Some(source),
            Self::RgbActiveFraction(source) | Self::RgbBrightness(source) => Some(source),
            Self::RgbMotionThreshold(source) => Some(source),
            Self::RgbGazeGeometry(source) => Some(source),
            Self::RgbFrameFreshness { source }
            | Self::SupervisorDuration { source, .. }
            | Self::ModeAuthorityLease { source, .. }
            | Self::ManualCommandAge { source }
            | Self::ManualDeadman { source }
            | Self::PointGoalRuntime { source }
            | Self::ExploreRuntime { source }
            | Self::ExploreYawScanDuration { source } => Some(source),
            Self::RgbFrameFreshnessDomain(source) => Some(source),
            Self::SupervisorTimeDomain { source, .. }
            | Self::ModeAuthorityLeaseTimeDomain { source, .. } => Some(source),
            Self::SupervisorConfig(source) => Some(source),
            Self::ManualDrive(source) => Some(source),
            Self::ExploreBoundary { source } => Some(source),
            Self::ExploreResources(source) => Some(source),
            Self::ExploreYawScanBudget(source) => Some(source),
            Self::InputTooLarge { .. }
            | Self::UnsupportedSchemaVersion { .. }
            | Self::OperatorConsoleBindAddressNotLoopback { .. }
            | Self::OperatorConsoleBindPortZero { .. }
            | Self::OperatorConsoleManualCommandZero { .. }
            | Self::OperatorConsoleManualCommandOutsideEnvelope { .. }
            | Self::OperatorConsoleCapabilityPathCollidesWithControlSocket
            | Self::OperatorConsoleCapabilityPathOutsideControlSocketParent
            | Self::WarmStartPathRoleCollision
            | Self::ReviewedNaturalHeadTargetMismatch { .. }
            | Self::ReviewedNaturalHoldEnvelopeOutsideStartupWindow { .. }
            | Self::ReviewedNaturalHeadStartBoundsMismatch { .. }
            | Self::ReviewedNaturalHeadMaximumTravelMismatch { .. }
            | Self::ReviewedNaturalHeadTorqueMismatch { .. }
            | Self::DuplicateAccessorySerialPath
            | Self::RgbExpressionRequiresEye
            | Self::UnsupportedRgbGazeGeometrySchemaVersion { .. }
            | Self::RgbEyeRoundTripBudgetOverflow { .. }
            | Self::RgbFreshnessDoesNotCoverEyeRoundTrip { .. }
            | Self::ModeAuthorityLeaseExceedsSupervisor { .. }
            | Self::MotionPolicyRequiresActuationFeature { .. }
            | Self::GoalArrivalTolerance { .. }
            | Self::ExploreGoalCount { .. }
            | Self::ExploreGridCellLimit { .. } => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoAgentPathField {
    OperatorConsoleCapability,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoOperatorConsoleDeadmanTickError {
    OutsideInclusiveRange {
        actual_ms: u64,
        minimum_ms: u64,
        maximum_ms: u64,
    },
}

impl fmt::Display for NanoOperatorConsoleDeadmanTickError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid operator-console deadman tick: {self:?}")
    }
}

impl std::error::Error for NanoOperatorConsoleDeadmanTickError {}

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
        Duration::from_millis(dto.terminal_response_timeout_ms),
    )
    .map_err(NanoAgentPolicyConfigParseError::ControlSocketTimeout)?;
    let runtime_queue_capacity =
        AgentControlRuntimeQueueCapacity::try_new(usize::from(dto.runtime_queue_capacity))
            .map_err(NanoAgentPolicyConfigParseError::ControlRuntimeQueue)?;
    let operator_console = parse_operator_console(dto.operator_console, &path)?;
    Ok(NanoAgentControlConfig {
        socket: AgentControlSocketConfig::new(path, timeouts),
        runtime_queue_capacity,
        operator_console,
    })
}

fn parse_operator_console(
    dto: NanoOperatorConsoleConfigDto,
    control_socket_path: &AgentControlSocketPath,
) -> Result<NanoOperatorConsoleConfig, NanoAgentPolicyConfigParseError> {
    let bind_address = dto
        .bind_address
        .parse::<SocketAddr>()
        .map_err(|source| NanoAgentPolicyConfigParseError::OperatorConsoleBindAddress { source })?;
    if !bind_address.ip().is_loopback() {
        return Err(
            NanoAgentPolicyConfigParseError::OperatorConsoleBindAddressNotLoopback {
                actual: bind_address,
            },
        );
    }
    if bind_address.port() == 0 {
        return Err(
            NanoAgentPolicyConfigParseError::OperatorConsoleBindPortZero {
                actual: bind_address,
            },
        );
    }

    let capability_path = parse_absolute_path(
        NanoAgentPathField::OperatorConsoleCapability,
        dto.capability_path,
        MAX_NANO_OPERATOR_CONSOLE_CAPABILITY_PATH_BYTES,
    )?;
    if capability_path.as_path() == control_socket_path.as_path() {
        return Err(
            NanoAgentPolicyConfigParseError::OperatorConsoleCapabilityPathCollidesWithControlSocket,
        );
    }
    if capability_path.as_path().parent() != control_socket_path.as_path().parent() {
        return Err(
            NanoAgentPolicyConfigParseError::OperatorConsoleCapabilityPathOutsideControlSocketParent,
        );
    }

    if !(MIN_NANO_OPERATOR_CONSOLE_DEADMAN_TICK_MS..=MAX_NANO_OPERATOR_CONSOLE_DEADMAN_TICK_MS)
        .contains(&dto.deadman_tick_ms)
    {
        return Err(
            NanoAgentPolicyConfigParseError::OperatorConsoleDeadmanTick {
                source: NanoOperatorConsoleDeadmanTickError::OutsideInclusiveRange {
                    actual_ms: dto.deadman_tick_ms,
                    minimum_ms: MIN_NANO_OPERATOR_CONSOLE_DEADMAN_TICK_MS,
                    maximum_ms: MAX_NANO_OPERATOR_CONSOLE_DEADMAN_TICK_MS,
                },
            },
        );
    }
    let manual_command_forward_mm_per_s = NonZeroU32::new(dto.manual_command_forward_mm_per_s)
        .ok_or(
            NanoAgentPolicyConfigParseError::OperatorConsoleManualCommandZero {
                field: "control.operator_console.manual_command_forward_mm_per_s",
            },
        )?;
    let manual_command_yaw_millirad_per_s = NonZeroU32::new(dto.manual_command_yaw_millirad_per_s)
        .ok_or(
            NanoAgentPolicyConfigParseError::OperatorConsoleManualCommandZero {
                field: "control.operator_console.manual_command_yaw_millirad_per_s",
            },
        )?;

    Ok(NanoOperatorConsoleConfig {
        bind_address,
        capability_path,
        deadman_tick: Duration::from_millis(dto.deadman_tick_ms),
        manual_command_forward_mm_per_s,
        manual_command_yaw_millirad_per_s,
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
        NanoHeadPolicyDto::ReturnToNaturalAndHoldContinuously {
            device_path,
            response_timeout_ms,
            write_timeout_ms,
            arming_freshness_ms,
            write_attempts,
            noise_budget_bytes,
            redundant_read_tolerance_ticks,
            readback_tolerance_ticks,
            final_target_tolerance_ticks,
            path_corridor_tolerance_ticks,
            direction_regression_tolerance_ticks,
            goal_speed_ticks_per_second,
            torque_limit_permille,
            minimum_start_ticks,
            maximum_start_ticks,
            reviewed_natural_target_ticks,
            maximum_travel_ticks,
            physical_torque_consent:
                NanoPhysicalTorqueConsentDto::EnableForReviewedNaturalReturnAndHold,
            physical_motion_consent: NanoPhysicalHeadMotionConsentDto::ReturnToReviewedNaturalTarget,
        } => {
            let probe = HeadProbeConfig::parse(HeadProbeConfigInput {
                device_path,
                response_timeout_ms,
                request_timeout_ms: write_timeout_ms,
                noise_budget_bytes,
            })
            .map_err(NanoAgentPolicyConfigParseError::HeadProbe)?;
            let return_to_target = ReturnToTargetConfig::parse(
                &probe,
                ReturnToTargetConfigInput {
                    write_timeout_ms,
                    arming_freshness_ms,
                    write_attempts,
                    redundant_read_tolerance_ticks,
                    readback_tolerance_ticks,
                    final_target_tolerance_ticks,
                    path_corridor_tolerance_ticks,
                    direction_regression_tolerance_ticks,
                    goal_speed_ticks_per_second,
                    torque_limit_permille,
                    minimum_start_ticks,
                    maximum_start_ticks,
                    target_ticks: reviewed_natural_target_ticks,
                    maximum_travel_ticks,
                },
            )
            .map_err(NanoAgentPolicyConfigParseError::HeadReturn)?;
            if reviewed_natural_target_ticks != KIKO_REVIEWED_NATURAL_HEAD_TARGET_TICKS {
                return Err(
                    NanoAgentPolicyConfigParseError::ReviewedNaturalHeadTargetMismatch {
                        configured_ticks: reviewed_natural_target_ticks,
                        required_ticks: KIKO_REVIEWED_NATURAL_HEAD_TARGET_TICKS,
                    },
                );
            }
            for joint in HeadJoint::ALL {
                let index = joint as usize;
                let target = reviewed_natural_target_ticks[index];
                let required_minimum_ticks = target.saturating_sub(readback_tolerance_ticks);
                let required_maximum_ticks = target.saturating_add(readback_tolerance_ticks);
                if minimum_start_ticks[index] > required_minimum_ticks
                    || maximum_start_ticks[index] < required_maximum_ticks
                {
                    return Err(
                        NanoAgentPolicyConfigParseError::ReviewedNaturalHoldEnvelopeOutsideStartupWindow {
                            joint,
                            configured_minimum_ticks: minimum_start_ticks[index],
                            configured_maximum_ticks: maximum_start_ticks[index],
                            required_minimum_ticks,
                            required_maximum_ticks,
                        },
                    );
                }
            }
            if minimum_start_ticks != KIKO_REVIEWED_NATURAL_HEAD_START_MINIMUM_TICKS
                || maximum_start_ticks != KIKO_REVIEWED_NATURAL_HEAD_START_MAXIMUM_TICKS
            {
                return Err(
                    NanoAgentPolicyConfigParseError::ReviewedNaturalHeadStartBoundsMismatch {
                        configured_minimum_ticks: minimum_start_ticks,
                        configured_maximum_ticks: maximum_start_ticks,
                        required_minimum_ticks: KIKO_REVIEWED_NATURAL_HEAD_START_MINIMUM_TICKS,
                        required_maximum_ticks: KIKO_REVIEWED_NATURAL_HEAD_START_MAXIMUM_TICKS,
                    },
                );
            }
            if maximum_travel_ticks != KIKO_REVIEWED_NATURAL_HEAD_MAXIMUM_TRAVEL_TICKS {
                return Err(
                    NanoAgentPolicyConfigParseError::ReviewedNaturalHeadMaximumTravelMismatch {
                        configured_ticks: maximum_travel_ticks,
                        required_ticks: KIKO_REVIEWED_NATURAL_HEAD_MAXIMUM_TRAVEL_TICKS,
                    },
                );
            }
            if torque_limit_permille != KIKO_REVIEWED_NATURAL_HEAD_TORQUE_LIMIT_PERMILLE {
                return Err(
                    NanoAgentPolicyConfigParseError::ReviewedNaturalHeadTorqueMismatch {
                        configured_permille: torque_limit_permille,
                        required_permille: KIKO_REVIEWED_NATURAL_HEAD_TORQUE_LIMIT_PERMILLE,
                    },
                );
            }
            Ok(ParsedNanoHeadPolicy::ReturnToNaturalAndHoldContinuously(
                NanoContinuousNaturalHeadHoldConfig {
                    return_to_target,
                    torque_consent: PhysicalTorqueEnableConsent::explicitly_granted(),
                    motion_consent: PhysicalHeadMotionConsent::explicitly_granted(),
                },
            ))
        }
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
            arrival_tolerance_m,
            clearance_from_known_obstacles_m,
            maximum_grid_cells,
            maximum_expanded_cells,
            maximum_open_set_entries,
            maximum_abs_yaw_rate_rad_s,
            yaw_travel_limit_exclusive_rad,
            maximum_scan_origin_displacement_m,
            maximum_scan_duration_ms,
            yaw_turn_direction,
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
            let arrival_tolerance_m = parse_goal_arrival_tolerance(
                NanoConfiguredMotionMode::FrontierExplore,
                arrival_tolerance_m,
            )?;
            let maximum_grid_cells = maximum_grid_cells as usize;
            if maximum_grid_cells > MAX_NANO_OCCUPANCY_CELLS {
                return Err(NanoAgentPolicyConfigParseError::ExploreGridCellLimit {
                    actual: maximum_grid_cells,
                    maximum: MAX_NANO_OCCUPANCY_CELLS,
                });
            }
            let explorer = FrontierExplorerConfig::try_new(
                clearance_from_known_obstacles_m,
                maximum_grid_cells,
                maximum_expanded_cells as usize,
                maximum_open_set_entries as usize,
            )
            .map_err(NanoAgentPolicyConfigParseError::ExploreResources)?;
            let maximum_scan_duration_ns = parse_bounded_milliseconds(
                maximum_scan_duration_ms,
                MAX_NANO_AGENT_EXPLORE_RUNTIME_MS,
            )
            .map_err(|source| NanoAgentPolicyConfigParseError::ExploreYawScanDuration { source })?;
            let yaw_scan_budget = FrontierYawScanBudgetV1::try_new(
                maximum_abs_yaw_rate_rad_s,
                yaw_travel_limit_exclusive_rad,
                maximum_scan_origin_displacement_m,
                maximum_scan_duration_ns,
            )
            .map_err(NanoAgentPolicyConfigParseError::ExploreYawScanBudget)?;
            NanoFrontierExplorePolicy::ControlApi(NanoFrontierExploreConfig {
                authority_lease,
                boundary_m,
                maximum_runtime: Duration::from_nanos(maximum_runtime_ns),
                maximum_frontier_goals,
                arrival_tolerance_m,
                explorer,
                yaw_scan_budget,
                yaw_turn_direction: yaw_turn_direction.into_domain(),
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
        NanoMotionModePolicyDto::ControlApi {
            authority_lease_ms,
            maximum_runtime_ms,
            arrival_tolerance_m,
        } => {
            if !cfg!(feature = "actuation") {
                return Err(
                    NanoAgentPolicyConfigParseError::MotionPolicyRequiresActuationFeature { mode },
                );
            }
            let authority_lease = parse_mode_authority_lease(mode, authority_lease_ms, supervisor)?;
            let maximum_runtime_ns = parse_bounded_milliseconds(
                maximum_runtime_ms,
                MAX_NANO_AGENT_POINT_GOAL_RUNTIME_MS,
            )
            .map_err(|source| NanoAgentPolicyConfigParseError::PointGoalRuntime { source })?;
            let arrival_tolerance_m = parse_goal_arrival_tolerance(mode, arrival_tolerance_m)?;
            Ok(NanoMotionModePolicy::ControlApi {
                authority_lease,
                maximum_runtime: Duration::from_nanos(maximum_runtime_ns),
                arrival_tolerance_m,
            })
        }
    }
}

fn parse_goal_arrival_tolerance(
    mode: NanoConfiguredMotionMode,
    value_m: f64,
) -> Result<f64, NanoAgentPolicyConfigParseError> {
    if !value_m.is_finite() || value_m <= 0.0 {
        return Err(NanoAgentPolicyConfigParseError::GoalArrivalTolerance { mode, value_m });
    }
    Ok(value_m)
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
struct NanoAgentPolicyConfigV3Dto {
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
    terminal_response_timeout_ms: u64,
    runtime_queue_capacity: u16,
    operator_console: NanoOperatorConsoleConfigDto,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NanoOperatorConsoleConfigDto {
    bind_address: String,
    capability_path: String,
    deadman_tick_ms: u64,
    manual_command_forward_mm_per_s: u32,
    manual_command_yaw_millirad_per_s: u32,
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
    ReturnToNaturalAndHoldContinuously {
        device_path: String,
        response_timeout_ms: u64,
        write_timeout_ms: u64,
        arming_freshness_ms: u64,
        write_attempts: u8,
        noise_budget_bytes: u16,
        redundant_read_tolerance_ticks: u16,
        readback_tolerance_ticks: u16,
        final_target_tolerance_ticks: u16,
        path_corridor_tolerance_ticks: u16,
        direction_regression_tolerance_ticks: u16,
        goal_speed_ticks_per_second: u16,
        torque_limit_permille: [u16; 4],
        minimum_start_ticks: [u16; 4],
        maximum_start_ticks: [u16; 4],
        reviewed_natural_target_ticks: [u16; 4],
        maximum_travel_ticks: [u16; 4],
        physical_torque_consent: NanoPhysicalTorqueConsentDto,
        physical_motion_consent: NanoPhysicalHeadMotionConsentDto,
    },
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum NanoPhysicalTorqueConsentDto {
    EnableForReviewedNaturalReturnAndHold,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum NanoPhysicalHeadMotionConsentDto {
    ReturnToReviewedNaturalTarget,
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
    ControlApi {
        authority_lease_ms: u64,
        maximum_runtime_ms: u64,
        arrival_tolerance_m: f64,
    },
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum NanoFrontierYawTurnDirectionDto {
    CounterClockwise,
    Clockwise,
}

impl NanoFrontierYawTurnDirectionDto {
    const fn into_domain(self) -> FrontierYawTurnDirectionV1 {
        match self {
            Self::CounterClockwise => FrontierYawTurnDirectionV1::CounterClockwise,
            Self::Clockwise => FrontierYawTurnDirectionV1::Clockwise,
        }
    }
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
        arrival_tolerance_m: f64,
        clearance_from_known_obstacles_m: f64,
        maximum_grid_cells: u32,
        maximum_expanded_cells: u32,
        maximum_open_set_entries: u32,
        maximum_abs_yaw_rate_rad_s: f64,
        yaw_travel_limit_exclusive_rad: f64,
        maximum_scan_origin_displacement_m: f64,
        maximum_scan_duration_ms: u64,
        yaw_turn_direction: NanoFrontierYawTurnDirectionDto,
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
            "schema_version": NANO_AGENT_POLICY_CONFIG_V3,
            "control": {
                "socket_path": "/tmp/kiko-agent/control.sock",
                "read_timeout_ms": 100,
                "write_timeout_ms": 100,
                "runtime_response_timeout_ms": 500,
                "terminal_response_timeout_ms": 300000,
                "runtime_queue_capacity": 8,
                "operator_console": {
                    "bind_address": "127.0.0.1:9877",
                    "capability_path": "/tmp/kiko-agent/operator-console.capability",
                    "deadman_tick_ms": 20,
                    "manual_command_forward_mm_per_s": 100,
                    "manual_command_yaw_millirad_per_s": 500
                }
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
                "mode": "return_to_natural_and_hold_continuously",
                "device_path": "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14031114-if00",
                "response_timeout_ms": 100,
                "write_timeout_ms": 100,
                "arming_freshness_ms": 250,
                "write_attempts": 2,
                "noise_budget_bytes": 32,
                "redundant_read_tolerance_ticks": 10,
                "readback_tolerance_ticks": 24,
                "final_target_tolerance_ticks": 24,
                "path_corridor_tolerance_ticks": 24,
                "direction_regression_tolerance_ticks": 24,
                "goal_speed_ticks_per_second": 50,
                "torque_limit_permille": KIKO_REVIEWED_NATURAL_HEAD_TORQUE_LIMIT_PERMILLE,
                "minimum_start_ticks": KIKO_REVIEWED_NATURAL_HEAD_START_MINIMUM_TICKS,
                "maximum_start_ticks": KIKO_REVIEWED_NATURAL_HEAD_START_MAXIMUM_TICKS,
                "reviewed_natural_target_ticks": KIKO_REVIEWED_NATURAL_HEAD_TARGET_TICKS,
                "maximum_travel_ticks": KIKO_REVIEWED_NATURAL_HEAD_MAXIMUM_TRAVEL_TICKS,
                "physical_torque_consent": "enable_for_reviewed_natural_return_and_hold",
                "physical_motion_consent": "return_to_reviewed_natural_target"
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
                    "authority_lease_ms": 1000,
                    "maximum_runtime_ms": 600000,
                    "arrival_tolerance_m": 0.1
                },
                "frontier_explore": {
                    "permission": "control_api",
                    "authority_lease_ms": 1000,
                    "boundary_minimum_x_m": -20.0,
                    "boundary_minimum_y_m": -10.0,
                    "boundary_maximum_x_m": 20.0,
                    "boundary_maximum_y_m": 10.0,
                    "maximum_runtime_ms": 600000,
                    "maximum_frontier_goals": 100,
                    "arrival_tolerance_m": 0.1,
                    "clearance_from_known_obstacles_m": 0.2,
                    "maximum_grid_cells": 4000000,
                    "maximum_expanded_cells": 4000000,
                    "maximum_open_set_entries": 32000000,
                    "maximum_abs_yaw_rate_rad_s": 1.0,
                    "yaw_travel_limit_exclusive_rad": std::f64::consts::TAU,
                    "maximum_scan_origin_displacement_m": 0.05,
                    "maximum_scan_duration_ms": 1000,
                    "yaw_turn_direction": "clockwise"
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

    fn parse(value: &Value) -> Result<NanoAgentPolicyConfigV3, NanoAgentPolicyConfigParseError> {
        NanoAgentPolicyConfigV3::parse_json(
            &serde_json::to_vec(value).expect("serialize test fixture"),
        )
    }

    #[test]
    fn valid_document_constructs_native_runtime_domains() {
        let parsed = parse(&valid_value()).expect("valid Nano agent config");
        assert_eq!(parsed.control().runtime_queue_capacity().get(), 8);
        assert_eq!(
            parsed.control().socket().timeouts().runtime_response(),
            Duration::from_millis(500)
        );
        assert_eq!(
            parsed.control().socket().timeouts().terminal_response(),
            Duration::from_secs(300)
        );
        assert_eq!(
            parsed.control().operator_console().bind_address(),
            "127.0.0.1:9877"
                .parse::<SocketAddr>()
                .expect("test address")
        );
        assert_eq!(
            parsed
                .control()
                .operator_console()
                .capability_path()
                .as_path(),
            Path::new("/tmp/kiko-agent/operator-console.capability")
        );
        assert_eq!(
            parsed.control().operator_console().deadman_tick(),
            Duration::from_millis(20)
        );
        assert_eq!(
            parsed
                .control()
                .operator_console()
                .manual_command_forward_velocity_mps(),
            0.1
        );
        assert_eq!(
            parsed
                .control()
                .operator_console()
                .manual_command_yaw_rate_rad_s(),
            0.5
        );
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

            let point_goal = parsed.live_mode_policy().point_goal();
            assert_eq!(
                point_goal
                    .authority_lease()
                    .expect("point-goal authority")
                    .as_nanos(),
                1_000_000_000
            );
            assert_eq!(point_goal.maximum_runtime(), Some(Duration::from_secs(600)));
            assert_eq!(point_goal.arrival_tolerance_m(), Some(0.1));

            let frontier = parsed
                .live_mode_policy()
                .frontier_explore()
                .config()
                .expect("frontier control API policy");
            assert_eq!(frontier.authority_lease().as_nanos(), 1_000_000_000);
            assert_eq!(frontier.maximum_runtime(), Duration::from_secs(600));
            assert_eq!(frontier.maximum_frontier_goals().get(), 100);
            assert_eq!(frontier.arrival_tolerance_m(), 0.1);
            assert_eq!(frontier.explorer().maximum_grid_cells(), 4_000_000);
            assert_eq!(frontier.explorer().maximum_expanded_cells(), 4_000_000);
            assert_eq!(frontier.explorer().maximum_open_set_entries(), 32_000_000);
            assert_eq!(
                frontier.yaw_scan_budget().maximum_duration_ns().get(),
                1_000_000_000
            );
            assert_eq!(
                frontier.yaw_turn_direction(),
                FrontierYawTurnDirectionV1::Clockwise
            );
        }
    }

    #[test]
    fn operator_console_requires_one_complete_unknown_field_free_object() {
        let mut missing_object = valid_value();
        missing_object["control"]
            .as_object_mut()
            .expect("control object")
            .remove("operator_console");
        assert!(matches!(
            parse(&missing_object),
            Err(NanoAgentPolicyConfigParseError::JsonDecode(_))
        ));

        for required_field in [
            "bind_address",
            "capability_path",
            "deadman_tick_ms",
            "manual_command_forward_mm_per_s",
            "manual_command_yaw_millirad_per_s",
        ] {
            let mut missing_field = valid_value();
            missing_field["control"]["operator_console"]
                .as_object_mut()
                .expect("operator-console object")
                .remove(required_field);
            assert!(
                matches!(
                    parse(&missing_field),
                    Err(NanoAgentPolicyConfigParseError::JsonDecode(_))
                ),
                "accepted missing operator-console field {required_field}"
            );
        }

        let mut unknown = valid_value();
        unknown["control"]["operator_console"]
            .as_object_mut()
            .expect("operator-console object")
            .insert("allow_remote".to_owned(), json!(false));
        assert!(matches!(
            parse(&unknown),
            Err(NanoAgentPolicyConfigParseError::JsonDecode(_))
        ));
    }

    #[test]
    fn operator_console_admits_only_explicit_stable_loopback_addresses() {
        for address in ["127.0.0.1:1", "127.255.255.254:65535", "[::1]:9877"] {
            let mut value = valid_value();
            value["control"]["operator_console"]["bind_address"] = json!(address);
            assert_eq!(
                parse(&value)
                    .expect("loopback address")
                    .control()
                    .operator_console()
                    .bind_address(),
                address.parse::<SocketAddr>().expect("fixture address")
            );
        }

        for address in ["0.0.0.0:9877", "192.0.2.1:9877", "[::]:9877"] {
            let mut value = valid_value();
            value["control"]["operator_console"]["bind_address"] = json!(address);
            assert!(matches!(
                parse(&value),
                Err(NanoAgentPolicyConfigParseError::OperatorConsoleBindAddressNotLoopback { .. })
            ));
        }

        for address in ["127.0.0.1:0", "[::1]:0"] {
            let mut value = valid_value();
            value["control"]["operator_console"]["bind_address"] = json!(address);
            assert!(matches!(
                parse(&value),
                Err(NanoAgentPolicyConfigParseError::OperatorConsoleBindPortZero { .. })
            ));
        }

        let mut hostname = valid_value();
        hostname["control"]["operator_console"]["bind_address"] = json!("localhost:9877");
        assert!(matches!(
            parse(&hostname),
            Err(NanoAgentPolicyConfigParseError::OperatorConsoleBindAddress { .. })
        ));
    }

    #[test]
    fn operator_console_capability_is_canonical_and_shares_the_control_parent() {
        for invalid in [
            "operator-console.capability",
            "/tmp/kiko-agent/../operator-console.capability",
            "/tmp//kiko-agent/operator-console.capability",
            "/tmp/kiko-agent/./operator-console.capability",
            "/",
        ] {
            let mut value = valid_value();
            value["control"]["operator_console"]["capability_path"] = json!(invalid);
            assert!(matches!(
                parse(&value),
                Err(NanoAgentPolicyConfigParseError::AbsolutePath {
                    field: NanoAgentPathField::OperatorConsoleCapability,
                    ..
                })
            ));
        }

        let mut other_parent = valid_value();
        other_parent["control"]["operator_console"]["capability_path"] =
            json!("/tmp/other/operator-console.capability");
        assert!(matches!(
            parse(&other_parent),
            Err(
                NanoAgentPolicyConfigParseError::OperatorConsoleCapabilityPathOutsideControlSocketParent
            )
        ));

        let mut collision = valid_value();
        collision["control"]["operator_console"]["capability_path"] =
            collision["control"]["socket_path"].clone();
        assert!(matches!(
            parse(&collision),
            Err(
                NanoAgentPolicyConfigParseError::OperatorConsoleCapabilityPathCollidesWithControlSocket
            )
        ));
    }

    #[test]
    fn operator_console_deadman_tick_is_exact_integer_milliseconds_in_range() {
        for milliseconds in [
            MIN_NANO_OPERATOR_CONSOLE_DEADMAN_TICK_MS,
            MAX_NANO_OPERATOR_CONSOLE_DEADMAN_TICK_MS,
        ] {
            let mut value = valid_value();
            value["control"]["operator_console"]["deadman_tick_ms"] = json!(milliseconds);
            assert_eq!(
                parse(&value)
                    .expect("boundary tick")
                    .control()
                    .operator_console()
                    .deadman_tick(),
                Duration::from_millis(milliseconds)
            );
        }

        for milliseconds in [
            MIN_NANO_OPERATOR_CONSOLE_DEADMAN_TICK_MS - 1,
            MAX_NANO_OPERATOR_CONSOLE_DEADMAN_TICK_MS + 1,
        ] {
            let mut value = valid_value();
            value["control"]["operator_console"]["deadman_tick_ms"] = json!(milliseconds);
            assert!(matches!(
                parse(&value),
                Err(NanoAgentPolicyConfigParseError::OperatorConsoleDeadmanTick {
                    source:
                        NanoOperatorConsoleDeadmanTickError::OutsideInclusiveRange {
                            actual_ms,
                            minimum_ms: MIN_NANO_OPERATOR_CONSOLE_DEADMAN_TICK_MS,
                            maximum_ms: MAX_NANO_OPERATOR_CONSOLE_DEADMAN_TICK_MS,
                        },
                }) if actual_ms == milliseconds
            ));
        }

        let mut fractional = valid_value();
        fractional["control"]["operator_console"]["deadman_tick_ms"] = json!(5.5);
        assert!(matches!(
            parse(&fractional),
            Err(NanoAgentPolicyConfigParseError::JsonDecode(_))
        ));
    }

    #[cfg(feature = "actuation")]
    #[test]
    fn operator_console_manual_step_is_nonzero_and_inside_the_manual_envelope() {
        for field in [
            "manual_command_forward_mm_per_s",
            "manual_command_yaw_millirad_per_s",
        ] {
            let mut zero = valid_value();
            zero["control"]["operator_console"][field] = json!(0);
            assert!(matches!(
                parse(&zero),
                Err(NanoAgentPolicyConfigParseError::OperatorConsoleManualCommandZero {
                    field: actual,
                }) if actual.ends_with(field)
            ));
        }

        let mut excessive_forward = valid_value();
        excessive_forward["control"]["operator_console"]["manual_command_forward_mm_per_s"] =
            json!(351);
        assert!(matches!(
            parse(&excessive_forward),
            Err(
                NanoAgentPolicyConfigParseError::OperatorConsoleManualCommandOutsideEnvelope {
                    requested_forward_velocity_mps,
                    maximum_forward_velocity_mps: 0.35,
                    ..
                }
            ) if (requested_forward_velocity_mps - 0.351).abs() < f64::EPSILON
        ));

        let mut excessive_yaw = valid_value();
        excessive_yaw["control"]["operator_console"]["manual_command_yaw_millirad_per_s"] =
            json!(1_001);
        assert!(matches!(
            parse(&excessive_yaw),
            Err(
                NanoAgentPolicyConfigParseError::OperatorConsoleManualCommandOutsideEnvelope {
                    requested_yaw_rate_rad_s,
                    maximum_yaw_rate_rad_s: 1.0,
                    ..
                }
            ) if (requested_yaw_rate_rad_s - 1.001).abs() < f64::EPSILON
        ));
    }

    #[test]
    fn earlier_policy_versions_are_not_reinterpreted_as_schema_v3() {
        let mut retired_v1 = valid_value();
        retired_v1["schema_version"] = json!(1);
        assert!(matches!(
            parse(&retired_v1),
            Err(NanoAgentPolicyConfigParseError::UnsupportedSchemaVersion {
                actual: 1,
                supported: NANO_AGENT_POLICY_CONFIG_V3,
            })
        ));

        let mut retired_v2 = valid_value();
        retired_v2["schema_version"] = json!(2);
        assert!(matches!(
            parse(&retired_v2),
            Err(NanoAgentPolicyConfigParseError::UnsupportedSchemaVersion {
                actual: 2,
                supported: NANO_AGENT_POLICY_CONFIG_V3,
            })
        ));
    }

    #[test]
    fn retired_version_one_observed_pose_hold_document_is_rejected() {
        let mut retired_v1 = valid_value();
        retired_v1["schema_version"] = json!(1);
        retired_v1["head"] = json!({
            "mode": "natural_hold",
            "device_path": "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14031114-if00",
            "response_timeout_ms": 100,
            "write_timeout_ms": 100,
            "arming_freshness_ms": 250,
            "write_attempts": 2,
            "noise_budget_bytes": 32,
            "redundant_read_tolerance_ticks": 10,
            "readback_tolerance_ticks": 20,
            "goal_speed_ticks_per_second": 50,
            "torque_limit_permille": [100, 100, 100, 100],
            "physical_torque_consent": "natural_hold_at_observed_pose"
        });
        assert!(matches!(
            parse(&retired_v1),
            Err(NanoAgentPolicyConfigParseError::JsonDecode(_))
        ));
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
        let head = bound
            .head()
            .return_to_natural_and_hold_continuously()
            .expect("bound head return");
        assert_eq!(
            head.runtime().device().path(),
            "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14031114-if00"
        );
        assert_eq!(
            head.torque_consent(),
            PhysicalTorqueEnableConsent::explicitly_granted()
        );
        assert_eq!(
            head.motion_consent(),
            PhysicalHeadMotionConsent::explicitly_granted()
        );
        assert_eq!(
            head.required_hold_target(),
            HeadHoldTarget::ReviewedReturn(head.return_config().target())
        );
        assert_eq!(
            head.return_config()
                .target()
                .positions()
                .map(|tick| tick.get()),
            KIKO_REVIEWED_NATURAL_HEAD_TARGET_TICKS
        );
        assert_eq!(
            head.return_config()
                .start_bounds()
                .minimum(HeadJoint::Bow)
                .get(),
            1_377
        );
        assert_eq!(
            head.return_config()
                .start_bounds()
                .maximum(HeadJoint::Roll)
                .get(),
            3_146
        );
        let configured_minimums =
            HeadJoint::ALL.map(|joint| head.return_config().start_bounds().minimum(joint).get());
        let configured_maximums =
            HeadJoint::ALL.map(|joint| head.return_config().start_bounds().maximum(joint).get());
        assert_eq!(
            configured_minimums,
            KIKO_REVIEWED_NATURAL_HEAD_START_MINIMUM_TICKS
        );
        assert_eq!(
            configured_maximums,
            KIKO_REVIEWED_NATURAL_HEAD_START_MAXIMUM_TICKS
        );
        for (index, target) in KIKO_REVIEWED_NATURAL_HEAD_TARGET_TICKS
            .into_iter()
            .enumerate()
        {
            assert_eq!(target.saturating_sub(configured_minimums[index]), 128);
            assert_eq!(configured_maximums[index].saturating_sub(target), 128);
        }
        assert_eq!(
            HeadJoint::ALL.map(|joint| { head.runtime().torque_limits().for_joint(joint).get() }),
            KIKO_REVIEWED_NATURAL_HEAD_TORQUE_LIMIT_PERMILLE
        );
    }

    #[test]
    fn natural_return_requires_bounded_start_and_travel_policy() {
        let mut missing = valid_value();
        missing["head"]
            .as_object_mut()
            .expect("head object")
            .remove("minimum_start_ticks");
        assert!(matches!(
            parse(&missing),
            Err(NanoAgentPolicyConfigParseError::JsonDecode(_))
        ));

        let mut descending = valid_value();
        descending["head"]["minimum_start_ticks"] = json!([1634, 3809, 1423, 2890]);
        descending["head"]["maximum_start_ticks"] = json!([1633, 4065, 1679, 3146]);
        assert!(matches!(
            parse(&descending),
            Err(NanoAgentPolicyConfigParseError::HeadReturn(
                ReturnToTargetConfigParseError::StartBounds(
                    ConfiguredHeadPoseBoundsError::Descending {
                        joint: HeadJoint::Bow,
                        ..
                    }
                )
            ))
        ));

        let mut travel_too_short = valid_value();
        travel_too_short["head"]["maximum_travel_ticks"] = json!([127, 128, 128, 128]);
        assert!(matches!(
            parse(&travel_too_short),
            Err(NanoAgentPolicyConfigParseError::HeadReturn(
                ReturnToTargetConfigParseError::TravelAboveMaximum {
                    joint: HeadJoint::Bow,
                    required_ticks: 128,
                    maximum_ticks: 127,
                }
            ))
        ));

        let mut excludes_reviewed_hold = valid_value();
        excludes_reviewed_hold["head"]["minimum_start_ticks"] = json!([1482, 3809, 1423, 2890]);
        assert!(matches!(
            parse(&excludes_reviewed_hold),
            Err(
                NanoAgentPolicyConfigParseError::ReviewedNaturalHoldEnvelopeOutsideStartupWindow {
                    joint: HeadJoint::Bow,
                    configured_minimum_ticks: 1_482,
                    required_minimum_ticks: 1_481,
                    ..
                }
            )
        ));

        let mut silently_shifted = valid_value();
        silently_shifted["head"]["minimum_start_ticks"] = json!([1378, 3809, 1423, 2890]);
        silently_shifted["head"]["maximum_start_ticks"] = json!([1634, 4065, 1679, 3146]);
        silently_shifted["head"]["maximum_travel_ticks"] = json!([129, 128, 128, 128]);
        let shifted_result = parse(&silently_shifted);
        assert!(
            matches!(
                &shifted_result,
                Err(
                    NanoAgentPolicyConfigParseError::ReviewedNaturalHeadStartBoundsMismatch {
                        configured_minimum_ticks: [1_378, 3_809, 1_423, 2_890],
                        required_minimum_ticks: KIKO_REVIEWED_NATURAL_HEAD_START_MINIMUM_TICKS,
                        required_maximum_ticks: KIKO_REVIEWED_NATURAL_HEAD_START_MAXIMUM_TICKS,
                        ..
                    }
                )
            ),
            "unexpected shifted-start result: {shifted_result:?}"
        );
    }

    #[test]
    fn natural_return_is_bound_to_reviewed_target_torque_and_fixed_motion_policy() {
        let mut wrong_target = valid_value();
        wrong_target["head"]["reviewed_natural_target_ticks"] = json!([1_506, 3_937, 1_551, 3_018]);
        wrong_target["head"]["minimum_start_ticks"] = json!([1_378, 3_809, 1_423, 2_890]);
        assert!(matches!(
            parse(&wrong_target),
            Err(
                NanoAgentPolicyConfigParseError::ReviewedNaturalHeadTargetMismatch {
                    configured_ticks: [1_506, 3_937, 1_551, 3_018],
                    required_ticks: KIKO_REVIEWED_NATURAL_HEAD_TARGET_TICKS,
                }
            )
        ));

        let mut wrong_travel = valid_value();
        wrong_travel["head"]["maximum_travel_ticks"] = json!([129, 128, 128, 128]);
        assert!(matches!(
            parse(&wrong_travel),
            Err(
                NanoAgentPolicyConfigParseError::ReviewedNaturalHeadMaximumTravelMismatch {
                    configured_ticks: [129, 128, 128, 128],
                    required_ticks: KIKO_REVIEWED_NATURAL_HEAD_MAXIMUM_TRAVEL_TICKS,
                }
            )
        ));

        let mut wrong_torque = valid_value();
        wrong_torque["head"]["torque_limit_permille"] = json!([649, 550, 400, 400]);
        assert!(matches!(
            parse(&wrong_torque),
            Err(
                NanoAgentPolicyConfigParseError::ReviewedNaturalHeadTorqueMismatch {
                    configured_permille: [649, 550, 400, 400],
                    required_permille: KIKO_REVIEWED_NATURAL_HEAD_TORQUE_LIMIT_PERMILLE,
                }
            )
        ));

        for fixed_runtime_field in [
            "position_step_ticks",
            "control_period_ms",
            "no_progress_timeout_ms",
            "motion_timeout_ms",
            "telemetry_set_max_age_ms",
            "maximum_hold_ms",
        ] {
            let mut caller_supplied_constant = valid_value();
            caller_supplied_constant["head"][fixed_runtime_field] = json!(1);
            assert!(
                matches!(
                    parse(&caller_supplied_constant),
                    Err(NanoAgentPolicyConfigParseError::JsonDecode(_))
                ),
                "weak policy must not override fixed runtime field {fixed_runtime_field}"
            );
        }

        assert_eq!(kiko_head_runtime::HEAD_RETURN_POSITION_STEP_TICKS, 50);
        assert_eq!(
            kiko_head_runtime::HEAD_RETURN_CONTROL_PERIOD,
            Duration::from_millis(100)
        );
        assert_eq!(
            kiko_head_runtime::HEAD_RETURN_NO_PROGRESS_TIMEOUT,
            Duration::from_secs(2)
        );
        assert_eq!(
            kiko_head_runtime::HEAD_RETURN_MOTION_TIMEOUT,
            Duration::from_secs(20)
        );
        assert_eq!(
            kiko_head_runtime::HEAD_RETURN_TELEMETRY_SET_MAX_AGE,
            Duration::from_millis(100)
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
            "\"schema_version\":3",
            "\"schema_version\":3,\"schema_version\":3",
            1,
        );
        assert!(matches!(
            NanoAgentPolicyConfigV3::parse_json(duplicate.as_bytes()),
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
            NanoAgentPolicyConfigV3::parse_json(&trailing),
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
            NanoAgentPolicyConfigV3::parse_json(&oversized),
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

        let mut missing_motion_consent = valid_value();
        missing_motion_consent["head"]
            .as_object_mut()
            .expect("head")
            .remove("physical_motion_consent");
        assert!(matches!(
            parse(&missing_motion_consent),
            Err(NanoAgentPolicyConfigParseError::JsonDecode(_))
        ));

        let mut legacy_torque_consent = valid_value();
        legacy_torque_consent["head"]["physical_torque_consent"] =
            json!("natural_hold_at_observed_pose");
        assert!(matches!(
            parse(&legacy_torque_consent),
            Err(NanoAgentPolicyConfigParseError::JsonDecode(_))
        ));

        let mut ambiguous_motion_consent = valid_value();
        ambiguous_motion_consent["head"]["physical_motion_consent"] = json!("motion_allowed");
        assert!(matches!(
            parse(&ambiguous_motion_consent),
            Err(NanoAgentPolicyConfigParseError::JsonDecode(_))
        ));

        let mut legacy_observed_hold = valid_value();
        legacy_observed_hold["head"] = json!({
            "mode": "natural_hold",
            "device_path": "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14031114-if00",
            "response_timeout_ms": 100,
            "write_timeout_ms": 100,
            "arming_freshness_ms": 250,
            "write_attempts": 2,
            "noise_budget_bytes": 32,
            "redundant_read_tolerance_ticks": 10,
            "readback_tolerance_ticks": 20,
            "goal_speed_ticks_per_second": 50,
            "torque_limit_permille": [600, 400, 400, 400],
            "minimum_pose_ticks": [2140, 2530, 2920, 2850],
            "maximum_pose_ticks": [2172, 2560, 2970, 2900],
            "physical_torque_consent": "natural_hold_at_observed_pose"
        });
        assert!(matches!(
            parse(&legacy_observed_hold),
            Err(NanoAgentPolicyConfigParseError::JsonDecode(_))
        ));

        let mut finite_hold_mode = valid_value();
        finite_hold_mode["head"]["mode"] = json!("return_to_natural_and_hold");
        assert!(matches!(
            parse(&finite_hold_mode),
            Err(NanoAgentPolicyConfigParseError::JsonDecode(_))
        ));

        let mut ambiguous_legacy_bounds = valid_value();
        ambiguous_legacy_bounds["head"]["minimum_pose_ticks"] =
            ambiguous_legacy_bounds["head"]["minimum_start_ticks"].clone();
        assert!(matches!(
            parse(&ambiguous_legacy_bounds),
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
    fn point_goal_runtime_is_required_positive_and_bounded_without_arithmetic_wrap() {
        let mut missing = valid_value();
        missing["live_mode_policy"]["point_goal"]
            .as_object_mut()
            .expect("point-goal policy object")
            .remove("maximum_runtime_ms");
        assert!(matches!(
            parse(&missing),
            Err(NanoAgentPolicyConfigParseError::JsonDecode(_))
        ));

        let mut zero = valid_value();
        zero["live_mode_policy"]["point_goal"]["maximum_runtime_ms"] = json!(0);
        assert!(matches!(
            parse(&zero),
            Err(NanoAgentPolicyConfigParseError::PointGoalRuntime {
                source: NanoBoundedMillisecondsError::Zero,
            })
        ));

        for invalid_ms in [MAX_NANO_AGENT_POINT_GOAL_RUNTIME_MS + 1, u64::MAX] {
            let mut too_large = valid_value();
            too_large["live_mode_policy"]["point_goal"]["maximum_runtime_ms"] = json!(invalid_ms);
            assert!(matches!(
                parse(&too_large),
                Err(NanoAgentPolicyConfigParseError::PointGoalRuntime {
                    source: NanoBoundedMillisecondsError::TooLarge {
                        actual_ms,
                        maximum_ms: MAX_NANO_AGENT_POINT_GOAL_RUNTIME_MS,
                    },
                }) if actual_ms == invalid_ms
            ));
        }

        let mut exact_maximum = valid_value();
        exact_maximum["live_mode_policy"]["point_goal"]["maximum_runtime_ms"] =
            json!(MAX_NANO_AGENT_POINT_GOAL_RUNTIME_MS);
        assert_eq!(
            parse(&exact_maximum)
                .expect("exact maximum point-goal runtime")
                .live_mode_policy()
                .point_goal()
                .maximum_runtime(),
            Some(Duration::from_millis(MAX_NANO_AGENT_POINT_GOAL_RUNTIME_MS))
        );
    }

    #[cfg(feature = "actuation")]
    #[test]
    fn exploration_requires_finite_ordered_bounds_and_finite_resources() {
        for mode in ["point_goal", "frontier_explore"] {
            let mut zero_tolerance = valid_value();
            zero_tolerance["live_mode_policy"][mode]["arrival_tolerance_m"] = json!(0.0);
            assert!(matches!(
                parse(&zero_tolerance),
                Err(NanoAgentPolicyConfigParseError::GoalArrivalTolerance { value_m: 0.0, .. })
            ));
        }

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

        let mut oversized_grid = valid_value();
        oversized_grid["live_mode_policy"]["frontier_explore"]["maximum_grid_cells"] =
            json!(MAX_NANO_OCCUPANCY_CELLS + 1);
        assert!(matches!(
            parse(&oversized_grid),
            Err(NanoAgentPolicyConfigParseError::ExploreGridCellLimit {
                actual,
                maximum,
            }) if actual == MAX_NANO_OCCUPANCY_CELLS + 1
                && maximum == MAX_NANO_OCCUPANCY_CELLS
        ));

        let mut expanded_exceeds_grid = valid_value();
        expanded_exceeds_grid["live_mode_policy"]["frontier_explore"]["maximum_expanded_cells"] =
            json!(4_000_001);
        assert!(matches!(
            parse(&expanded_exceeds_grid),
            Err(NanoAgentPolicyConfigParseError::ExploreResources(
                FrontierExplorerConfigError::ExpandedCellsExceedGridLimit { .. }
            ))
        ));

        let mut zero_yaw_rate = valid_value();
        zero_yaw_rate["live_mode_policy"]["frontier_explore"]["maximum_abs_yaw_rate_rad_s"] =
            json!(0.0);
        assert!(matches!(
            parse(&zero_yaw_rate),
            Err(NanoAgentPolicyConfigParseError::ExploreYawScanBudget(
                FrontierYawScanBudgetError::NotPositive {
                    field: "maximum_abs_yaw_rate_rad_s",
                    ..
                }
            ))
        ));

        let mut zero_scan_duration = valid_value();
        zero_scan_duration["live_mode_policy"]["frontier_explore"]["maximum_scan_duration_ms"] =
            json!(0);
        assert!(matches!(
            parse(&zero_scan_duration),
            Err(NanoAgentPolicyConfigParseError::ExploreYawScanDuration {
                source: NanoBoundedMillisecondsError::Zero,
            })
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
        let mut value = valid_value();
        value["live_mode_policy"]["point_goal"] = json!({
            "permission": "control_api",
            "authority_lease_ms": 500,
            "maximum_runtime_ms": 1000,
            "arrival_tolerance_m": 0.1
        });
        assert!(matches!(
            parse(&value),
            Err(
                NanoAgentPolicyConfigParseError::MotionPolicyRequiresActuationFeature {
                    mode: NanoConfiguredMotionMode::PointGoal
                }
            )
        ));

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
            "maximum_frontier_goals": 1,
            "arrival_tolerance_m": 0.1,
            "clearance_from_known_obstacles_m": 0.2,
            "maximum_grid_cells": 100,
            "maximum_expanded_cells": 100,
            "maximum_open_set_entries": 800,
            "maximum_abs_yaw_rate_rad_s": 1.0,
            "yaw_travel_limit_exclusive_rad": std::f64::consts::TAU,
            "maximum_scan_origin_displacement_m": 0.05,
            "maximum_scan_duration_ms": 500,
            "yaw_turn_direction": "clockwise"
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
