//! Attended wheel-on commissioning admission and execution seam.
//!
//! Static preparation in this module does not touch hardware. The separate
//! commissioning live adapter opens the one exact OAK graph in the same
//! process and binds its visual velocity, calibrated IMU, RGB expression
//! frames, connected MXID, observed rectified-stereo calibration, clock epoch,
//! and policy stream identities before this module will consume the one-shot
//! attended claims. This module then exclusively owns the commissioning STM32
//! session for the bounded run.
//!
//! Neither production firmware nor the checked-in wheels-off candidate can
//! cross this boundary: the authoritative server and inventory parsers must
//! both admit the distinct `AttendedWheelOnCommissioning` identity with
//! truthful unverified physical-stop semantics. Cross-class documents reject
//! before state mutation, OAK access, or STM32 access.

use std::ffi::OsStr;
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{self, IsTerminal, Read, Write};
use std::net::SocketAddr;
use std::num::NonZeroU64;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use kiko_base_commissioning::{BoundedId, CommissioningEvidenceV1Dto, CommissioningState};
use kiko_device_inventory::{
    ArtifactRelativePath, ArtifactRelativePathError, DeploymentAssetByteLimit,
    DeploymentAssetByteLimitError, DeploymentAssetLoadError, DeviceInventoryManifestV3,
    LoadedDeploymentAsset, MAX_MANIFEST_JSON_BYTES, ManifestLoadError, load_deployment_asset,
    load_expected_manifest_v3_from_slice,
};
#[cfg(feature = "nano-attended-navigation-trial")]
use robot_command_client::AppliedCommandReceipt;
use robot_command_client::{
    ClientConfig, ConfigError as ClientConfigError, StopRecoveryPolicy, TimeoutNs,
};
use robot_protocol::v2::{
    ControllerSafetyClass, ControllerSessionClass, PhysicalStopSemantics, TimerPwm,
    V2CommandLeaseMs,
};
use robot_server::config::{
    ControllerServerConfigV3, MAX_CONTROLLER_SERVER_CONFIG_JSON_BYTES, ServerConfigError,
};
use robot_server::{
    V2ControllerOwner, V2ControllerOwnerStartError, V2ControllerOwnerTerminationError,
};
use rustix::fs::{AtFlags, FileType, Mode, OFlags, fstat, fsync, mkdirat, openat, statat};
use rustix::io::Errno;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[cfg(feature = "nano-attended-navigation-trial")]
use super::LiveMpcControlDriver;
use super::actuation::{LiveActuationError, PhysicalActuationSession};
use super::nano_base_commissioning::{
    AdmittedAttendedCommissioning, CommissioningArtifactDirectory,
    CommissioningArtifactDirectoryError, CommissioningExternalSignal, FileCommissioningJournal,
    FileCommissioningJournalError, NanoBaseCommissioningFailure,
    NanoBaseCommissioningPolicyParseError, NanoBaseCommissioningPolicyV1,
    NanoBaseCommissioningProgress, NanoBaseCommissioningProposal,
    NanoBaseCommissioningPublishFailure, NanoBaseCommissioningSampleV1Dto,
    NanoBaseCommissioningSession, SoleCommissioningActuator,
};
use super::{
    MAX_NANO_AGENT_LAUNCH_JSON_BYTES, MAX_NANO_CALIBRATION_ARTIFACT_JSON_BYTES,
    ManifestBoundNanoAgentPolicyConfigV3, NanoAccessoryManifestBindingError,
    NanoAgentLaunchParseError, NanoAgentLaunchV3, NanoAgentPolicyConfigParseError,
    NanoAgentPolicyConfigV3, NanoCalibrationArtifactParseError, NanoCalibrationArtifactV1,
    NanoCalibrationBindingError, NanoFaceCascadeAssetRole, NanoLaunchAssetRole,
    NanoLaunchBoundAssetLoadError,
};
use crate::dataset::Calibration;

pub const NANO_BASE_COMMISSIONING_LAUNCH_V1: u32 = 1;
pub const NANO_BASE_COMMISSIONING_CONTROLLER_PROFILE_V1: u32 = 1;
pub const MAX_NANO_BASE_COMMISSIONING_LAUNCH_JSON_BYTES: usize = 32 * 1_024;
pub const MAX_NANO_BASE_COMMISSIONING_CONTROLLER_PROFILE_JSON_BYTES: usize = 16 * 1_024;
pub const MAX_WHEEL_ON_COMMISSIONING_PWM_PERCENT: u8 = 20;
pub const MAX_ATTENDED_ATTESTATION_LIFETIME_NS: u64 = 15 * 60 * 1_000_000_000;
pub const MAX_COMMISSIONING_SESSION_ID_BYTES: usize = 64;
const ATTESTATION_SCHEMA_V3: u32 = 3;
const ATTESTATION_FILE_NAME: &str = "attended-wheel-on-attestation-v3.json";
const JOURNAL_FILE_NAME: &str = "commissioning-evidence-v1.ndjson";
const BODY_FRAME_ID: &str = "base_body_flu";
const MAX_STATE_ROOT_PATH_BYTES: usize = 1_024;
const ATTENDED_CONFIRMATION_TTY: &str = "/dev/tty";
const ATTENDED_CONFIRMATION_LINE_MAX_BYTES: usize = 96;
const ATTENDED_CONFIRMATION_RESPONSE_TIMEOUT: Duration = Duration::from_secs(15);
const ATTENDED_CONFIRMATION_POLL_SLICE: Duration = Duration::from_millis(25);
const ATTENDED_CONFIRMATION_MAX_CONSUMPTION_DELAY: Duration = Duration::from_secs(5);
const ATTENDED_CONFIRMATION_CHALLENGE_BYTES: usize = 16;
const ATTENDED_CONFIRMATION_BOUND_ASSET_COUNT: usize = 11;
const ATTENDED_CONFIRMATION_CHANNEL: &str =
    "fresh_controlling_tty_session_bound_nonce_exact_phrases_v2";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommissioningAssetRole {
    Policy,
    ControllerProfile,
    ControllerServerContract,
    DeviceManifest,
    CalibrationArtifact,
    LiveGraphLaunch,
}

impl CommissioningAssetRole {
    const ALL: [Self; 6] = [
        Self::Policy,
        Self::ControllerProfile,
        Self::ControllerServerContract,
        Self::DeviceManifest,
        Self::CalibrationArtifact,
        Self::LiveGraphLaunch,
    ];

    const fn maximum_bytes(self) -> u64 {
        match self {
            Self::Policy => {
                super::nano_base_commissioning::MAX_NANO_BASE_COMMISSIONING_POLICY_JSON_BYTES as u64
            }
            Self::ControllerProfile => {
                MAX_NANO_BASE_COMMISSIONING_CONTROLLER_PROFILE_JSON_BYTES as u64
            }
            Self::ControllerServerContract => MAX_CONTROLLER_SERVER_CONFIG_JSON_BYTES as u64,
            Self::DeviceManifest => MAX_MANIFEST_JSON_BYTES as u64,
            Self::CalibrationArtifact => MAX_NANO_CALIBRATION_ARTIFACT_JSON_BYTES as u64,
            Self::LiveGraphLaunch => MAX_NANO_AGENT_LAUNCH_JSON_BYTES as u64,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CommissioningAssetBinding {
    relative_path: ArtifactRelativePath,
    byte_limit: DeploymentAssetByteLimit,
    expected_sha256: [u8; 32],
}

impl CommissioningAssetBinding {
    fn parse(
        role: CommissioningAssetRole,
        dto: CommissioningAssetBindingDto,
    ) -> Result<Self, NanoBaseCommissioningLaunchParseError> {
        let relative_path = ArtifactRelativePath::parse(dto.relative_path)
            .map_err(|source| NanoBaseCommissioningLaunchParseError::AssetPath { role, source })?;
        let byte_limit =
            DeploymentAssetByteLimit::try_new(dto.maximum_bytes).map_err(|source| {
                NanoBaseCommissioningLaunchParseError::AssetByteLimit { role, source }
            })?;
        if byte_limit.get() > role.maximum_bytes() {
            return Err(
                NanoBaseCommissioningLaunchParseError::AssetByteLimitAboveRoleMaximum {
                    role,
                    actual: byte_limit.get(),
                    maximum: role.maximum_bytes(),
                },
            );
        }
        let expected_sha256 = parse_lower_hex_exact(&dto.sha256_hex)
            .ok_or(NanoBaseCommissioningLaunchParseError::AssetSha256Syntax { role })?;
        Ok(Self {
            relative_path,
            byte_limit,
            expected_sha256,
        })
    }

    fn load_exact(
        &self,
        deployment_root: &Path,
    ) -> Result<LoadedDeploymentAsset, NanoBaseCommissioningAssetLoadError> {
        let loaded =
            load_deployment_asset(deployment_root, self.relative_path.clone(), self.byte_limit)
                .map_err(NanoBaseCommissioningAssetLoadError::Load)?;
        let observed = *loaded.content_sha256().as_bytes();
        if observed != self.expected_sha256 {
            return Err(NanoBaseCommissioningAssetLoadError::ContentMismatch {
                relative_path: self.relative_path.clone(),
                expected: self.expected_sha256,
                observed,
            });
        }
        Ok(loaded)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NanoBaseCommissioningLaunchV1 {
    session_id: CommissioningSessionId,
    policy: CommissioningAssetBinding,
    controller_profile: CommissioningAssetBinding,
    controller_server_contract: CommissioningAssetBinding,
    device_manifest: CommissioningAssetBinding,
    calibration_artifact: CommissioningAssetBinding,
    live_graph_launch: CommissioningAssetBinding,
}

impl NanoBaseCommissioningLaunchV1 {
    pub fn parse_json(json: &[u8]) -> Result<Self, NanoBaseCommissioningLaunchParseError> {
        if json.len() > MAX_NANO_BASE_COMMISSIONING_LAUNCH_JSON_BYTES {
            return Err(NanoBaseCommissioningLaunchParseError::InputTooLarge {
                actual: json.len(),
                maximum: MAX_NANO_BASE_COMMISSIONING_LAUNCH_JSON_BYTES,
            });
        }
        let mut deserializer = serde_json::Deserializer::from_slice(json);
        let dto = NanoBaseCommissioningLaunchV1Dto::deserialize(&mut deserializer)
            .map_err(NanoBaseCommissioningLaunchParseError::JsonDecode)?;
        deserializer
            .end()
            .map_err(NanoBaseCommissioningLaunchParseError::JsonTrailingData)?;
        if dto.schema_version != NANO_BASE_COMMISSIONING_LAUNCH_V1 {
            return Err(NanoBaseCommissioningLaunchParseError::UnsupportedSchema {
                actual: dto.schema_version,
                supported: NANO_BASE_COMMISSIONING_LAUNCH_V1,
            });
        }
        let launch = Self {
            session_id: CommissioningSessionId::parse(dto.session_id)?,
            policy: CommissioningAssetBinding::parse(
                CommissioningAssetRole::Policy,
                dto.commissioning_policy_asset,
            )?,
            controller_profile: CommissioningAssetBinding::parse(
                CommissioningAssetRole::ControllerProfile,
                dto.controller_profile_asset,
            )?,
            controller_server_contract: CommissioningAssetBinding::parse(
                CommissioningAssetRole::ControllerServerContract,
                dto.controller_server_contract_asset,
            )?,
            device_manifest: CommissioningAssetBinding::parse(
                CommissioningAssetRole::DeviceManifest,
                dto.device_manifest_asset,
            )?,
            calibration_artifact: CommissioningAssetBinding::parse(
                CommissioningAssetRole::CalibrationArtifact,
                dto.calibration_artifact_asset,
            )?,
            live_graph_launch: CommissioningAssetBinding::parse(
                CommissioningAssetRole::LiveGraphLaunch,
                dto.live_graph_launch_asset,
            )?,
        };
        launch.require_distinct_assets()?;
        Ok(launch)
    }

    fn asset(&self, role: CommissioningAssetRole) -> &CommissioningAssetBinding {
        match role {
            CommissioningAssetRole::Policy => &self.policy,
            CommissioningAssetRole::ControllerProfile => &self.controller_profile,
            CommissioningAssetRole::ControllerServerContract => &self.controller_server_contract,
            CommissioningAssetRole::DeviceManifest => &self.device_manifest,
            CommissioningAssetRole::CalibrationArtifact => &self.calibration_artifact,
            CommissioningAssetRole::LiveGraphLaunch => &self.live_graph_launch,
        }
    }

    fn require_distinct_assets(&self) -> Result<(), NanoBaseCommissioningLaunchParseError> {
        for (index, left) in CommissioningAssetRole::ALL.iter().copied().enumerate() {
            for right in CommissioningAssetRole::ALL.iter().copied().skip(index + 1) {
                if self.asset(left).relative_path == self.asset(right).relative_path {
                    return Err(NanoBaseCommissioningLaunchParseError::AssetPathAliased {
                        left,
                        right,
                    });
                }
            }
        }
        Ok(())
    }

    pub fn session_id(&self) -> &str {
        self.session_id.as_str()
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NanoBaseCommissioningLaunchV1Dto {
    schema_version: u32,
    session_id: String,
    commissioning_policy_asset: CommissioningAssetBindingDto,
    controller_profile_asset: CommissioningAssetBindingDto,
    controller_server_contract_asset: CommissioningAssetBindingDto,
    device_manifest_asset: CommissioningAssetBindingDto,
    calibration_artifact_asset: CommissioningAssetBindingDto,
    live_graph_launch_asset: CommissioningAssetBindingDto,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CommissioningAssetBindingDto {
    relative_path: String,
    maximum_bytes: u64,
    sha256_hex: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CommissioningSessionId(Box<str>);

impl CommissioningSessionId {
    fn parse(value: String) -> Result<Self, NanoBaseCommissioningLaunchParseError> {
        if value.is_empty()
            || value.len() > MAX_COMMISSIONING_SESSION_ID_BYTES
            || matches!(value.as_str(), "." | "..")
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
        {
            return Err(NanoBaseCommissioningLaunchParseError::InvalidSessionId {
                actual_bytes: value.len(),
                maximum_bytes: MAX_COMMISSIONING_SESSION_ID_BYTES,
            });
        }
        Ok(Self(value.into_boxed_str()))
    }

    fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug)]
pub enum NanoBaseCommissioningLaunchParseError {
    InputTooLarge {
        actual: usize,
        maximum: usize,
    },
    JsonDecode(serde_json::Error),
    JsonTrailingData(serde_json::Error),
    UnsupportedSchema {
        actual: u32,
        supported: u32,
    },
    InvalidSessionId {
        actual_bytes: usize,
        maximum_bytes: usize,
    },
    AssetPath {
        role: CommissioningAssetRole,
        source: ArtifactRelativePathError,
    },
    AssetByteLimit {
        role: CommissioningAssetRole,
        source: DeploymentAssetByteLimitError,
    },
    AssetByteLimitAboveRoleMaximum {
        role: CommissioningAssetRole,
        actual: u64,
        maximum: u64,
    },
    AssetSha256Syntax {
        role: CommissioningAssetRole,
    },
    AssetPathAliased {
        left: CommissioningAssetRole,
        right: CommissioningAssetRole,
    },
}

impl fmt::Display for NanoBaseCommissioningLaunchParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid Nano base-commissioning launch document: {self:?}"
        )
    }
}

impl std::error::Error for NanoBaseCommissioningLaunchParseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::JsonDecode(source) | Self::JsonTrailingData(source) => Some(source),
            Self::AssetPath { source, .. } => Some(source),
            Self::AssetByteLimit { source, .. } => Some(source),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub enum NanoBaseCommissioningAssetLoadError {
    Load(DeploymentAssetLoadError),
    ContentMismatch {
        relative_path: ArtifactRelativePath,
        expected: [u8; 32],
        observed: [u8; 32],
    },
}

impl fmt::Display for NanoBaseCommissioningAssetLoadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Load(source) => write!(formatter, "commissioning asset load failed: {source}"),
            Self::ContentMismatch { relative_path, .. } => write!(
                formatter,
                "commissioning asset {} does not match its launch-bound SHA-256",
                relative_path.as_str()
            ),
        }
    }
}

impl std::error::Error for NanoBaseCommissioningAssetLoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Load(source) => Some(source),
            Self::ContentMismatch { .. } => None,
        }
    }
}

/// Commissioning-only host/client policy. It cannot describe candidate
/// firmware and it does not activate a resulting plant.
#[derive(Clone, Debug)]
pub struct NanoBaseCommissioningControllerProfileV1 {
    session_id: Box<str>,
    controller_session_id: BoundedId,
    command_endpoint: SocketAddr,
    expected_hardware_profile_claim_id: Box<str>,
    maximum_abs_pwm_percent: u8,
    maximum_command_step_percent: u8,
    command_lease: V2CommandLeaseMs,
    status_timeout: TimeoutNs,
    acquire_timeout: TimeoutNs,
    applied_ack_timeout: TimeoutNs,
    stop_recovery: StopRecoveryPolicy,
    attestation_lifetime_ns: NonZeroU64,
    content_sha256: [u8; 32],
}

impl NanoBaseCommissioningControllerProfileV1 {
    pub fn parse_json(
        json: &[u8],
    ) -> Result<Self, NanoBaseCommissioningControllerProfileParseError> {
        if json.len() > MAX_NANO_BASE_COMMISSIONING_CONTROLLER_PROFILE_JSON_BYTES {
            return Err(
                NanoBaseCommissioningControllerProfileParseError::InputTooLarge {
                    actual: json.len(),
                    maximum: MAX_NANO_BASE_COMMISSIONING_CONTROLLER_PROFILE_JSON_BYTES,
                },
            );
        }
        let mut deserializer = serde_json::Deserializer::from_slice(json);
        let dto = NanoBaseCommissioningControllerProfileV1Dto::deserialize(&mut deserializer)
            .map_err(NanoBaseCommissioningControllerProfileParseError::JsonDecode)?;
        deserializer
            .end()
            .map_err(NanoBaseCommissioningControllerProfileParseError::JsonTrailingData)?;
        if dto.schema_version != NANO_BASE_COMMISSIONING_CONTROLLER_PROFILE_V1 {
            return Err(
                NanoBaseCommissioningControllerProfileParseError::UnsupportedSchema {
                    actual: dto.schema_version,
                    supported: NANO_BASE_COMMISSIONING_CONTROLLER_PROFILE_V1,
                },
            );
        }
        let session_id = CommissioningSessionId::parse(dto.session_id)
            .map_err(NanoBaseCommissioningControllerProfileParseError::SessionId)?;
        let controller_session_id =
            parse_bounded_id("controller_session_id", dto.controller_session_id)
                .map_err(NanoBaseCommissioningControllerProfileParseError::ControllerSessionId)?;
        let command_endpoint = dto
            .command_udp_endpoint
            .parse::<SocketAddr>()
            .map_err(NanoBaseCommissioningControllerProfileParseError::CommandEndpointSyntax)?;
        if command_endpoint.port() == 0 || !command_endpoint.ip().is_loopback() {
            return Err(
                NanoBaseCommissioningControllerProfileParseError::CommandEndpointNotLoopback(
                    command_endpoint,
                ),
            );
        }
        let expected_hardware_profile_claim_id =
            parse_profile_claim(dto.expected_hardware_profile_claim_id)?;
        let maximum_abs_pwm_percent = dto.maximum_abs_pwm_percent;
        if !(1..=MAX_WHEEL_ON_COMMISSIONING_PWM_PERCENT).contains(&maximum_abs_pwm_percent) {
            return Err(
                NanoBaseCommissioningControllerProfileParseError::MaximumPwmOutOfRange {
                    actual: maximum_abs_pwm_percent,
                    maximum: MAX_WHEEL_ON_COMMISSIONING_PWM_PERCENT,
                },
            );
        }
        let maximum_command_step_percent = dto.maximum_command_step_percent;
        if maximum_command_step_percent == 0
            || maximum_command_step_percent > maximum_abs_pwm_percent
        {
            return Err(
                NanoBaseCommissioningControllerProfileParseError::MaximumStepOutOfRange {
                    actual: maximum_command_step_percent,
                    maximum: maximum_abs_pwm_percent,
                },
            );
        }
        let command_lease = V2CommandLeaseMs::try_new(dto.command_lease_ms)
            .map_err(NanoBaseCommissioningControllerProfileParseError::CommandLease)?;
        let status_timeout = TimeoutNs::try_new(dto.status_timeout_ns)
            .map_err(NanoBaseCommissioningControllerProfileParseError::Client)?;
        let acquire_timeout = TimeoutNs::try_new(dto.acquire_timeout_ns)
            .map_err(NanoBaseCommissioningControllerProfileParseError::Client)?;
        let applied_ack_timeout = TimeoutNs::try_new(dto.applied_ack_timeout_ns)
            .map_err(NanoBaseCommissioningControllerProfileParseError::Client)?;
        let stop_attempt_timeout = TimeoutNs::try_new(dto.stop_attempt_timeout_ns)
            .map_err(NanoBaseCommissioningControllerProfileParseError::Client)?;
        let stop_recovery =
            StopRecoveryPolicy::try_new(dto.max_stop_recovery_attempts, stop_attempt_timeout)
                .map_err(NanoBaseCommissioningControllerProfileParseError::Client)?;
        let attestation_lifetime_ns = NonZeroU64::new(dto.attestation_lifetime_ns)
            .ok_or(NanoBaseCommissioningControllerProfileParseError::AttestationLifetimeZero)?;
        if attestation_lifetime_ns.get() > MAX_ATTENDED_ATTESTATION_LIFETIME_NS {
            return Err(
                NanoBaseCommissioningControllerProfileParseError::AttestationLifetimeAboveMaximum {
                    actual_ns: attestation_lifetime_ns.get(),
                    maximum_ns: MAX_ATTENDED_ATTESTATION_LIFETIME_NS,
                },
            );
        }
        Ok(Self {
            session_id: session_id.0,
            controller_session_id,
            command_endpoint,
            expected_hardware_profile_claim_id,
            maximum_abs_pwm_percent,
            maximum_command_step_percent,
            command_lease,
            status_timeout,
            acquire_timeout,
            applied_ack_timeout,
            stop_recovery,
            attestation_lifetime_ns,
            content_sha256: Sha256::digest(json).into(),
        })
    }

    fn client_config(
        &self,
        server: &ControllerServerConfigV3,
    ) -> Result<ClientConfig, ClientConfigError> {
        ClientConfig::try_new_for_session(
            robot_command_client::UdpEndpoint::try_new(self.command_endpoint)?,
            server.controller_uid(),
            server.firmware_abi(),
            server.firmware_build_id(),
            server.actuator_config_fingerprint(),
            ControllerSessionClass::AttendedWheelOnCommissioning,
            self.status_timeout,
            self.acquire_timeout,
            self.applied_ack_timeout,
            self.stop_recovery,
            self.command_lease,
        )
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub const fn command_endpoint(&self) -> SocketAddr {
        self.command_endpoint
    }

    pub const fn maximum_abs_pwm_percent(&self) -> u8 {
        self.maximum_abs_pwm_percent
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NanoBaseCommissioningControllerProfileV1Dto {
    schema_version: u32,
    session_id: String,
    controller_session_id: String,
    command_udp_endpoint: String,
    expected_hardware_profile_claim_id: String,
    maximum_abs_pwm_percent: u8,
    maximum_command_step_percent: u8,
    command_lease_ms: u16,
    status_timeout_ns: u64,
    acquire_timeout_ns: u64,
    applied_ack_timeout_ns: u64,
    stop_attempt_timeout_ns: u64,
    max_stop_recovery_attempts: u8,
    attestation_lifetime_ns: u64,
}

#[derive(Debug)]
pub enum NanoBaseCommissioningControllerProfileParseError {
    InputTooLarge { actual: usize, maximum: usize },
    JsonDecode(serde_json::Error),
    JsonTrailingData(serde_json::Error),
    UnsupportedSchema { actual: u32, supported: u32 },
    SessionId(NanoBaseCommissioningLaunchParseError),
    ControllerSessionId(kiko_base_commissioning::IdentifierError),
    CommandEndpointSyntax(std::net::AddrParseError),
    CommandEndpointNotLoopback(SocketAddr),
    InvalidHardwareProfileClaim,
    MaximumPwmOutOfRange { actual: u8, maximum: u8 },
    MaximumStepOutOfRange { actual: u8, maximum: u8 },
    CommandLease(robot_protocol::v2::DomainError),
    Client(ClientConfigError),
    AttestationLifetimeZero,
    AttestationLifetimeAboveMaximum { actual_ns: u64, maximum_ns: u64 },
}

impl fmt::Display for NanoBaseCommissioningControllerProfileParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid Nano base-commissioning controller profile: {self:?}"
        )
    }
}

impl std::error::Error for NanoBaseCommissioningControllerProfileParseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::JsonDecode(source) | Self::JsonTrailingData(source) => Some(source),
            Self::SessionId(source) => Some(source),
            Self::ControllerSessionId(source) => Some(source),
            Self::CommandEndpointSyntax(source) => Some(source),
            Self::CommandLease(source) => Some(source),
            Self::Client(source) => Some(source),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CommissioningClockEpoch([u8; 16]);

impl CommissioningClockEpoch {
    pub fn try_new(bytes: [u8; 16]) -> Result<Self, CommissioningClockEpochError> {
        if bytes == [0; 16] {
            Err(CommissioningClockEpochError)
        } else {
            Ok(Self(bytes))
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommissioningClockEpochError;

impl fmt::Display for CommissioningClockEpochError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("commissioning clock epoch must be nonzero")
    }
}

impl std::error::Error for CommissioningClockEpochError {}

/// Static same-owner stream admission. Runtime samples carry only values and
/// timestamps; these identities cannot vary between samples.
#[derive(Clone, Debug)]
pub struct AdmittedCommissioningObservationStream {
    session_id: Box<str>,
    clock_epoch: CommissioningClockEpoch,
    visual_velocity_source_id: BoundedId,
    imu_calibration_id: BoundedId,
}

impl AdmittedCommissioningObservationStream {
    pub const fn clock_epoch(&self) -> CommissioningClockEpoch {
        self.clock_epoch
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CommissioningAlignedObservation {
    now_ns: u64,
    visual_observed_at_ns: u64,
    imu_observed_at_ns: u64,
    visual_body_forward_velocity_mps: f64,
    visual_body_lateral_velocity_mps: f64,
    calibrated_imu_yaw_rate_rad_s: f64,
}

impl CommissioningAlignedObservation {
    pub fn try_new(
        now_ns: u64,
        visual_observed_at_ns: u64,
        imu_observed_at_ns: u64,
        visual_body_forward_velocity_mps: f64,
        visual_body_lateral_velocity_mps: f64,
        calibrated_imu_yaw_rate_rad_s: f64,
    ) -> Result<Self, CommissioningAlignedObservationError> {
        if now_ns == 0 || visual_observed_at_ns == 0 || imu_observed_at_ns == 0 {
            return Err(CommissioningAlignedObservationError::ZeroTimestamp);
        }
        for (field, value) in [
            (
                CommissioningObservationValue::ForwardVelocityMps,
                visual_body_forward_velocity_mps,
            ),
            (
                CommissioningObservationValue::LateralVelocityMps,
                visual_body_lateral_velocity_mps,
            ),
            (
                CommissioningObservationValue::YawRateRadPerSec,
                calibrated_imu_yaw_rate_rad_s,
            ),
        ] {
            if !value.is_finite() {
                return Err(CommissioningAlignedObservationError::NonFinite { field });
            }
        }
        Ok(Self {
            now_ns,
            visual_observed_at_ns,
            imu_observed_at_ns,
            visual_body_forward_velocity_mps,
            visual_body_lateral_velocity_mps,
            calibrated_imu_yaw_rate_rad_s,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommissioningObservationValue {
    ForwardVelocityMps,
    LateralVelocityMps,
    YawRateRadPerSec,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommissioningAlignedObservationError {
    ZeroTimestamp,
    NonFinite {
        field: CommissioningObservationValue,
    },
}

impl fmt::Display for CommissioningAlignedObservationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid aligned commissioning observation: {self:?}"
        )
    }
}

impl std::error::Error for CommissioningAlignedObservationError {}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CommissioningSamplingRequest {
    pub progress: NanoBaseCommissioningProgress,
    pub expected_receipt: super::nano_base_commissioning::ExactCommissioningControllerReceipt,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum CommissioningObservationEvent {
    Observation(CommissioningAlignedObservation),
    Terminal(CommissioningExternalSignal),
}

/// Internal source implemented only by the canonical same-owner OAK/SLAM/IMU
/// loop. Keeping this trait crate-private prevents another crate from
/// manufacturing commissioning observations.
pub(crate) trait CommissioningObservationSource {
    type Error: std::error::Error + Send + Sync + 'static;

    fn stream_binding(&self) -> &AdmittedCommissioningObservationStream;
    fn next_observation(
        &mut self,
        request: CommissioningSamplingRequest,
    ) -> Result<CommissioningObservationEvent, Self::Error>;

    fn terminal_signal_for_error(&self, _error: &Self::Error) -> CommissioningExternalSignal {
        CommissioningExternalSignal::SupervisorFault
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommissioningObservationKind {
    SampleReady,
    TerminalSignal,
}

/// Read-only progress hook for a console or journal forwarder.
///
/// A reporter observes sampling intent and outcome, but cannot provide or
/// modify controller, visual, or IMU evidence.
pub trait CommissioningProgressReporter {
    type Error: std::error::Error + Send + Sync + 'static;

    fn before_observation(
        &mut self,
        request: CommissioningSamplingRequest,
    ) -> Result<(), Self::Error>;

    fn after_observation(
        &mut self,
        request: CommissioningSamplingRequest,
        outcome: CommissioningObservationKind,
    ) -> Result<(), Self::Error>;
}

#[derive(Debug)]
pub struct LoadedNanoBaseCommissioningInputs {
    pub launch_source: LoadedDeploymentAsset,
    pub policy_source: LoadedDeploymentAsset,
    pub controller_profile_source: LoadedDeploymentAsset,
    pub controller_server_source: LoadedDeploymentAsset,
    pub manifest_source: LoadedDeploymentAsset,
    pub calibration_source: LoadedDeploymentAsset,
    pub live_graph_launch_source: LoadedDeploymentAsset,
    pub accessory_policy_source: LoadedDeploymentAsset,
    pub onnx_runtime_library: LoadedDeploymentAsset,
    pub superpoint_model: LoadedDeploymentAsset,
    pub lightglue_model: LoadedDeploymentAsset,
    pub frontal_face_cascade: LoadedDeploymentAsset,
    pub profile_face_cascade: LoadedDeploymentAsset,
}

/// Read-only, fully parsed commissioning admission. Constructing this value
/// performs no prompt, state mutation, OAK access, or STM32 access.
#[must_use = "preflight does not grant commissioning motion authority"]
pub struct PreparedNanoBaseCommissioning {
    state_root: CommissioningStateRoot,
    launch: NanoBaseCommissioningLaunchV1,
    policy: NanoBaseCommissioningPolicyV1,
    profile: NanoBaseCommissioningControllerProfileV1,
    server: ControllerServerConfigV3,
    calibration: NanoCalibrationArtifactV1,
    live_graph: NanoAgentLaunchV3,
    accessory_policy: ManifestBoundNanoAgentPolicyConfigV3,
    inputs: LoadedNanoBaseCommissioningInputs,
}

impl PreparedNanoBaseCommissioning {
    pub(crate) fn admit_same_owner_stream(
        &self,
        connected_oak_mxid: &str,
        observed_stereo: &Calibration,
        visual_velocity_source_id: &str,
        clock_epoch: CommissioningClockEpoch,
    ) -> Result<AdmittedCommissioningObservationStream, CommissioningStreamAdmissionError> {
        self.calibration
            .require_connected_oak_mxid(connected_oak_mxid)
            .map_err(CommissioningStreamAdmissionError::Calibration)?;
        self.calibration
            .require_observed_stereo(observed_stereo)
            .map_err(CommissioningStreamAdmissionError::Calibration)?;
        let observed_visual = parse_bounded_id(
            "visual_velocity_source_id",
            visual_velocity_source_id.to_owned(),
        )
        .map_err(CommissioningStreamAdmissionError::VisualSourceId)?;
        let expected_visual = self
            .policy
            .commissioning()
            .expected_visual_velocity_source_id();
        if observed_visual != expected_visual {
            return Err(CommissioningStreamAdmissionError::VisualSourceMismatch);
        }
        let imu_calibration_id = parse_bounded_id(
            "imu_calibration_id",
            self.calibration.imu_calibration_id().as_str().to_owned(),
        )
        .map_err(CommissioningStreamAdmissionError::ImuCalibrationId)?;
        if imu_calibration_id != self.policy.commissioning().expected_imu_calibration_id() {
            return Err(CommissioningStreamAdmissionError::ImuCalibrationMismatch);
        }
        Ok(AdmittedCommissioningObservationStream {
            session_id: self.launch.session_id.0.clone(),
            clock_epoch,
            visual_velocity_source_id: observed_visual,
            imu_calibration_id,
        })
    }

    /// Run the attended ceremony for this exact prepared session and stream,
    /// then consume it immediately into one commissioning authority.
    ///
    /// The private confirmation token is session/config/clock-epoch bound,
    /// non-cloneable, timestamped from `clock_origin` inside the ceremony, and
    /// rejected if it is not consumed within the fixed freshness window. No
    /// public API can obtain or date an unbound confirmation token.
    pub fn consume_fresh_attended_attestation(
        self,
        stream: AdmittedCommissioningObservationStream,
        clock_origin: Instant,
        running: &AtomicBool,
    ) -> Result<AdmittedNanoBaseCommissioningRun, FreshAttendedCommissioningAdmissionError> {
        if stream.session_id.as_ref() != self.launch.session_id() {
            return Err(FreshAttendedCommissioningAdmissionError::Attestation(
                Box::new(CommissioningAttestationError::StreamSessionMismatch),
            ));
        }
        let context = self.attended_confirmation_context(&stream);
        let confirmation =
            require_fresh_attended_wheel_on_confirmation(context, clock_origin, running).map_err(
                |source| FreshAttendedCommissioningAdmissionError::Confirmation(Box::new(source)),
            )?;
        self.consume_attended_attestation_at(stream, confirmation, Instant::now(), running)
            .map_err(|source| {
                FreshAttendedCommissioningAdmissionError::Attestation(Box::new(source))
            })
    }

    fn consume_attended_attestation_at(
        self,
        stream: AdmittedCommissioningObservationStream,
        confirmation: AttendedWheelOnConfirmation,
        consumed_at: Instant,
        running: &AtomicBool,
    ) -> Result<AdmittedNanoBaseCommissioningRun, CommissioningAttestationError> {
        if stream.session_id.as_ref() != self.launch.session_id() {
            return Err(CommissioningAttestationError::StreamSessionMismatch);
        }
        let expected_context = self.attended_confirmation_context(&stream);
        let confirmation =
            confirmation.require_bound_fresh(expected_context, consumed_at, running)?;
        let issued_at_ns = confirmation.issued_at_ns;
        let expires_at_ns = issued_at_ns
            .checked_add(self.profile.attestation_lifetime_ns.get())
            .ok_or(CommissioningAttestationError::ExpiryOverflow)?;
        let artifact_directory = self
            .state_root
            .create_session_directory(self.launch.session_id())?;
        let attestation = AttendedAttestationRecordV3 {
            schema_version: ATTESTATION_SCHEMA_V3,
            session_id: self.launch.session_id(),
            issued_at_ns,
            expires_at_ns,
            wheels_attached: true,
            clear_area_confirmed: true,
            independent_power_cut_tested_and_reachable: true,
            operator_attending: true,
            confirmation_channel: ATTENDED_CONFIRMATION_CHANNEL,
            confirmation_context_sha256: canonical_sha256(confirmation.context.0),
            confirmation_challenge_sha256: canonical_sha256(
                confirmation.challenge_transcript_sha256,
            ),
            maximum_response_wait_ms: u64::try_from(
                ATTENDED_CONFIRMATION_RESPONSE_TIMEOUT.as_millis(),
            )
            .expect("confirmation timeout is a small constant"),
            maximum_consumption_delay_ms: u64::try_from(
                ATTENDED_CONFIRMATION_MAX_CONSUMPTION_DELAY.as_millis(),
            )
            .expect("confirmation consumption delay is a small constant"),
            launch_sha256: canonical_sha256(*self.inputs.launch_source.content_sha256().as_bytes()),
            policy_sha256: canonical_sha256(*self.inputs.policy_source.content_sha256().as_bytes()),
            controller_profile_sha256: canonical_sha256(
                *self
                    .inputs
                    .controller_profile_source
                    .content_sha256()
                    .as_bytes(),
            ),
            controller_contract_sha256: canonical_sha256(
                *self
                    .inputs
                    .controller_server_source
                    .content_sha256()
                    .as_bytes(),
            ),
            manifest_sha256: canonical_sha256(
                *self.inputs.manifest_source.content_sha256().as_bytes(),
            ),
            calibration_sha256: canonical_sha256(
                *self.inputs.calibration_source.content_sha256().as_bytes(),
            ),
            live_graph_launch_sha256: canonical_sha256(
                *self
                    .inputs
                    .live_graph_launch_source
                    .content_sha256()
                    .as_bytes(),
            ),
            accessory_policy_sha256: canonical_sha256(
                *self
                    .inputs
                    .accessory_policy_source
                    .content_sha256()
                    .as_bytes(),
            ),
            onnx_runtime_sha256: canonical_sha256(
                *self.inputs.onnx_runtime_library.content_sha256().as_bytes(),
            ),
            superpoint_model_sha256: canonical_sha256(
                *self.inputs.superpoint_model.content_sha256().as_bytes(),
            ),
            lightglue_model_sha256: canonical_sha256(
                *self.inputs.lightglue_model.content_sha256().as_bytes(),
            ),
            consumption_status: "consumed_before_controller_acquisition",
        };
        let mut attestation_bytes =
            serde_json::to_vec(&attestation).map_err(CommissioningAttestationError::Encode)?;
        attestation_bytes.push(b'\n');
        let attestation_sha256: [u8; 32] = Sha256::digest(&attestation_bytes).into();
        write_new_durable(
            &artifact_directory,
            ATTESTATION_FILE_NAME,
            &attestation_bytes,
        )?;
        let journal = FileCommissioningJournal::create_new_at(
            artifact_directory.directory(),
            OsStr::new(JOURNAL_FILE_NAME),
        )
        .map_err(CommissioningAttestationError::Journal)?;
        artifact_directory
            .verify_binding()
            .map_err(CommissioningAttestationError::ArtifactDirectory)?;
        let authority = AdmittedAttendedCommissioning::from_verified_attended_admission(
            self.profile.controller_session_id,
            self.profile.content_sha256,
            attestation_sha256,
            self.profile.maximum_abs_pwm_percent,
            issued_at_ns,
            expires_at_ns,
        )
        .map_err(CommissioningAttestationError::Authority)?;
        Ok(AdmittedNanoBaseCommissioningRun {
            policy: self.policy,
            profile: self.profile,
            server: self.server,
            authority,
            confirmation_issued_at_ns: issued_at_ns,
            stream,
            journal,
            artifact_directory,
        })
    }

    fn attended_confirmation_context(
        &self,
        stream: &AdmittedCommissioningObservationStream,
    ) -> AttendedConfirmationContext {
        let asset_content_sha256 = [
            *self.inputs.launch_source.content_sha256().as_bytes(),
            *self.inputs.policy_source.content_sha256().as_bytes(),
            *self
                .inputs
                .controller_profile_source
                .content_sha256()
                .as_bytes(),
            *self
                .inputs
                .controller_server_source
                .content_sha256()
                .as_bytes(),
            *self.inputs.manifest_source.content_sha256().as_bytes(),
            *self.inputs.calibration_source.content_sha256().as_bytes(),
            *self
                .inputs
                .live_graph_launch_source
                .content_sha256()
                .as_bytes(),
            *self
                .inputs
                .accessory_policy_source
                .content_sha256()
                .as_bytes(),
            *self.inputs.onnx_runtime_library.content_sha256().as_bytes(),
            *self.inputs.superpoint_model.content_sha256().as_bytes(),
            *self.inputs.lightglue_model.content_sha256().as_bytes(),
        ];
        derive_attended_confirmation_context(
            self.launch.session_id(),
            stream.session_id.as_ref(),
            stream.clock_epoch,
            stream.visual_velocity_source_id.as_str(),
            stream.imu_calibration_id.as_str(),
            &asset_content_sha256,
        )
    }

    pub fn launch(&self) -> &NanoBaseCommissioningLaunchV1 {
        &self.launch
    }

    pub fn profile(&self) -> &NanoBaseCommissioningControllerProfileV1 {
        &self.profile
    }

    pub const fn calibration(&self) -> &NanoCalibrationArtifactV1 {
        &self.calibration
    }

    pub const fn live_graph(&self) -> &NanoAgentLaunchV3 {
        &self.live_graph
    }

    pub const fn accessory_policy(&self) -> &ManifestBoundNanoAgentPolicyConfigV3 {
        &self.accessory_policy
    }

    pub const fn loaded_inputs(&self) -> &LoadedNanoBaseCommissioningInputs {
        &self.inputs
    }

    pub fn expected_visual_velocity_source_id(&self) -> BoundedId {
        self.policy
            .commissioning()
            .expected_visual_velocity_source_id()
    }

    pub const fn maximum_sample_gap_ns(&self) -> NonZeroU64 {
        let configured = self.policy.maximum_sample_gap_ns().get();
        let fit = self.policy.fit().max_sample_period_ns().get();
        NonZeroU64::new(if configured < fit { configured } else { fit })
            .expect("both parsed sample periods are nonzero")
    }

    pub const fn minimum_sample_period_ns(&self) -> NonZeroU64 {
        self.policy.fit().min_sample_period_ns()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AttendedConfirmationContext([u8; 32]);

#[derive(Debug)]
struct AttendedWheelOnConfirmation {
    context: AttendedConfirmationContext,
    challenge_transcript_sha256: [u8; 32],
    issued_at_ns: u64,
    completed_at: Instant,
}

impl AttendedWheelOnConfirmation {
    fn require_bound_fresh(
        self,
        expected_context: AttendedConfirmationContext,
        consumed_at: Instant,
        running: &AtomicBool,
    ) -> Result<Self, CommissioningAttestationError> {
        if !running.load(Ordering::Acquire) {
            return Err(CommissioningAttestationError::InterruptedBeforeConsumption);
        }
        if self.context != expected_context {
            return Err(CommissioningAttestationError::ConfirmationContextMismatch);
        }
        let age = consumed_at
            .checked_duration_since(self.completed_at)
            .ok_or(CommissioningAttestationError::ConfirmationClockRegression)?;
        if age > ATTENDED_CONFIRMATION_MAX_CONSUMPTION_DELAY {
            return Err(CommissioningAttestationError::ConfirmationStale {
                age,
                maximum_age: ATTENDED_CONFIRMATION_MAX_CONSUMPTION_DELAY,
            });
        }
        Ok(self)
    }
}

#[derive(Clone, Copy)]
struct AttendedConfirmationClaim {
    explanation: &'static str,
    phrase: &'static str,
}

const ATTENDED_CONFIRMATION_CLAIMS: [AttendedConfirmationClaim; 4] = [
    AttendedConfirmationClaim {
        explanation: "Confirm that both drive wheels are physically attached. Software cannot observe this.",
        phrase: "WHEELS ATTACHED",
    },
    AttendedConfirmationClaim {
        explanation: "Confirm that the complete robot motion envelope is physically clear right now.",
        phrase: "MOTION AREA CLEAR",
    },
    AttendedConfirmationClaim {
        explanation: "Confirm that you will continuously attend this entire calibration attempt.",
        phrase: "OPERATOR ATTENDING",
    },
    AttendedConfirmationClaim {
        explanation: "Confirm that an independent physical power cut was tested and is immediately reachable.",
        phrase: "POWER CUT TESTED AND REACHABLE",
    },
];

/// Parse four fresh, context- and nonce-bound physical claims from the
/// process's controlling terminal. This private seam is reachable only through
/// [`PreparedNanoBaseCommissioning::consume_fresh_attended_attestation`], after
/// the exact same-owner live stream exists. Stdin, stdout, flags, and
/// environment variables are not confirmation channels.
fn require_fresh_attended_wheel_on_confirmation(
    context: AttendedConfirmationContext,
    clock_origin: Instant,
    running: &AtomicBool,
) -> Result<AttendedWheelOnConfirmation, AttendedWheelOnConfirmationError> {
    let mut terminal = RealAttendedConfirmationTerminal::open()?;
    let mut challenges = OsConfirmationChallengeSource;
    run_attended_confirmation_dialog(
        &mut terminal,
        &mut challenges,
        context,
        clock_origin,
        running,
    )
}

#[derive(Debug)]
pub enum AttendedWheelOnConfirmationError {
    OpenControllingTty(io::Error),
    TtyRequired,
    Interrupted,
    DiscardPendingInput(io::Error),
    ChallengeEntropy(getrandom::Error),
    DeadlineOutOfRange,
    ClockOriginAfterConfirmation,
    IssuedAtOutOfRange(u128),
    IssuedAtZero,
    Prompt(io::Error),
    Input(io::Error),
    EndOfInput,
    ResponseTimedOut { maximum_wait: Duration },
    LineTooLong { maximum_bytes: usize },
    InvalidUtf8,
    PhraseMismatch { expected: String },
}

impl fmt::Display for AttendedWheelOnConfirmationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OpenControllingTty(source) => {
                write!(
                    formatter,
                    "could not open controlling TTY /dev/tty: {source}"
                )
            }
            Self::TtyRequired => formatter
                .write_str("attended wheel-on commissioning requires a real controlling terminal"),
            Self::Interrupted => {
                formatter.write_str("attended confirmation was interrupted by the operator")
            }
            Self::DiscardPendingInput(source) => write!(
                formatter,
                "could not discard prebuffered controlling-TTY input: {source}"
            ),
            Self::ChallengeEntropy(source) => {
                write!(
                    formatter,
                    "could not create a fresh confirmation challenge: {source}"
                )
            }
            Self::DeadlineOutOfRange => {
                formatter.write_str("confirmation response deadline exceeds the host clock")
            }
            Self::ClockOriginAfterConfirmation => {
                formatter.write_str("confirmation completed before its host clock origin")
            }
            Self::IssuedAtOutOfRange(value) => {
                write!(
                    formatter,
                    "confirmation host timestamp {value}ns exceeds the u64 evidence domain"
                )
            }
            Self::IssuedAtZero => {
                formatter.write_str("confirmation host timestamp must be nonzero")
            }
            Self::Prompt(source) => {
                write!(
                    formatter,
                    "could not write controlling-TTY prompt: {source}"
                )
            }
            Self::Input(source) => {
                write!(
                    formatter,
                    "could not read controlling-TTY response: {source}"
                )
            }
            Self::EndOfInput => formatter
                .write_str("controlling TTY closed before every physical claim was confirmed"),
            Self::ResponseTimedOut { maximum_wait } => write!(
                formatter,
                "controlling-TTY confirmation exceeded the {:?} per-prompt limit",
                maximum_wait
            ),
            Self::LineTooLong { maximum_bytes } => write!(
                formatter,
                "controlling-TTY confirmation exceeded {maximum_bytes} bytes"
            ),
            Self::InvalidUtf8 => {
                formatter.write_str("controlling-TTY confirmation was not valid UTF-8")
            }
            Self::PhraseMismatch { expected } => write!(
                formatter,
                "physical claim was not confirmed; expected exact response {expected:?}"
            ),
        }
    }
}

impl std::error::Error for AttendedWheelOnConfirmationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::OpenControllingTty(source)
            | Self::DiscardPendingInput(source)
            | Self::Prompt(source)
            | Self::Input(source) => Some(source),
            Self::ChallengeEntropy(source) => Some(source),
            _ => None,
        }
    }
}

trait AttendedConfirmationTerminal {
    fn is_terminal(&self) -> bool;
    fn discard_pending_input(&mut self) -> io::Result<()>;
    fn write_prompt(&mut self, prompt: &str) -> io::Result<()>;
    fn read_bounded_line(
        &mut self,
        deadline: Instant,
        running: &AtomicBool,
    ) -> Result<Vec<u8>, AttendedWheelOnConfirmationError>;
}

trait ConfirmationChallengeSource {
    fn next_challenge(
        &mut self,
    ) -> Result<[u8; ATTENDED_CONFIRMATION_CHALLENGE_BYTES], AttendedWheelOnConfirmationError>;
}

struct OsConfirmationChallengeSource;

impl ConfirmationChallengeSource for OsConfirmationChallengeSource {
    fn next_challenge(
        &mut self,
    ) -> Result<[u8; ATTENDED_CONFIRMATION_CHALLENGE_BYTES], AttendedWheelOnConfirmationError> {
        let mut value = [0_u8; ATTENDED_CONFIRMATION_CHALLENGE_BYTES];
        getrandom::fill(&mut value).map_err(AttendedWheelOnConfirmationError::ChallengeEntropy)?;
        Ok(value)
    }
}

fn run_attended_confirmation_dialog<T, C>(
    terminal: &mut T,
    challenges: &mut C,
    context: AttendedConfirmationContext,
    clock_origin: Instant,
    running: &AtomicBool,
) -> Result<AttendedWheelOnConfirmation, AttendedWheelOnConfirmationError>
where
    T: AttendedConfirmationTerminal,
    C: ConfirmationChallengeSource,
{
    if !terminal.is_terminal() {
        return Err(AttendedWheelOnConfirmationError::TtyRequired);
    }

    let mut transcript = Sha256::new();
    transcript.update(b"kiko-attended-wheel-on-confirmation-transcript-v2\0");
    transcript.update(context.0);
    for claim in ATTENDED_CONFIRMATION_CLAIMS {
        if !running.load(Ordering::Acquire) {
            return Err(AttendedWheelOnConfirmationError::Interrupted);
        }
        terminal
            .discard_pending_input()
            .map_err(AttendedWheelOnConfirmationError::DiscardPendingInput)?;
        let challenge = challenges.next_challenge()?;
        let expected = format!("{} {}", claim.phrase, lower_hex_bytes(&challenge));
        let prompt = format!(
            "{}\nType exactly {:?} then press Enter: ",
            claim.explanation, expected
        );
        terminal
            .write_prompt(&prompt)
            .map_err(AttendedWheelOnConfirmationError::Prompt)?;
        let deadline = Instant::now()
            .checked_add(ATTENDED_CONFIRMATION_RESPONSE_TIMEOUT)
            .ok_or(AttendedWheelOnConfirmationError::DeadlineOutOfRange)?;
        let raw = terminal.read_bounded_line(deadline, running)?;
        let actual =
            String::from_utf8(raw).map_err(|_| AttendedWheelOnConfirmationError::InvalidUtf8)?;
        if actual != expected {
            return Err(AttendedWheelOnConfirmationError::PhraseMismatch { expected });
        }
        transcript.update(expected.as_bytes());
        transcript.update(b"\n");
    }

    let completed_at = Instant::now();
    let issued_at = completed_at
        .checked_duration_since(clock_origin)
        .ok_or(AttendedWheelOnConfirmationError::ClockOriginAfterConfirmation)?;
    let issued_at_ns = u64::try_from(issued_at.as_nanos())
        .map_err(|_| AttendedWheelOnConfirmationError::IssuedAtOutOfRange(issued_at.as_nanos()))?;
    if issued_at_ns == 0 {
        return Err(AttendedWheelOnConfirmationError::IssuedAtZero);
    }
    Ok(AttendedWheelOnConfirmation {
        context,
        challenge_transcript_sha256: transcript.finalize().into(),
        issued_at_ns,
        completed_at,
    })
}

fn derive_attended_confirmation_context(
    launch_session_id: &str,
    stream_session_id: &str,
    clock_epoch: CommissioningClockEpoch,
    visual_velocity_source_id: &str,
    imu_calibration_id: &str,
    asset_content_sha256: &[[u8; 32]; ATTENDED_CONFIRMATION_BOUND_ASSET_COUNT],
) -> AttendedConfirmationContext {
    let mut hasher = Sha256::new();
    hasher.update(b"kiko-attended-wheel-on-confirmation-context-v2\0");
    for value in [
        launch_session_id.as_bytes(),
        stream_session_id.as_bytes(),
        clock_epoch.0.as_slice(),
        visual_velocity_source_id.as_bytes(),
        imu_calibration_id.as_bytes(),
    ] {
        update_confirmation_context_field(&mut hasher, value);
    }
    for digest in asset_content_sha256 {
        update_confirmation_context_field(&mut hasher, digest);
    }
    AttendedConfirmationContext(hasher.finalize().into())
}

fn update_confirmation_context_field(hasher: &mut Sha256, value: &[u8]) {
    let length = u64::try_from(value.len()).expect("bounded confirmation field length fits u64");
    hasher.update(length.to_le_bytes());
    hasher.update(value);
}

struct RealAttendedConfirmationTerminal {
    file: File,
}

impl RealAttendedConfirmationTerminal {
    fn open() -> Result<Self, AttendedWheelOnConfirmationError> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .open(ATTENDED_CONFIRMATION_TTY)
            .map_err(AttendedWheelOnConfirmationError::OpenControllingTty)?;
        Ok(Self { file })
    }
}

impl AttendedConfirmationTerminal for RealAttendedConfirmationTerminal {
    fn is_terminal(&self) -> bool {
        self.file.is_terminal()
    }

    fn discard_pending_input(&mut self) -> io::Result<()> {
        rustix::termios::tcflush(&self.file, rustix::termios::QueueSelector::IFlush)
            .map_err(errno_as_io)
    }

    fn write_prompt(&mut self, prompt: &str) -> io::Result<()> {
        self.file
            .write_all(prompt.as_bytes())
            .and_then(|()| self.file.flush())
    }

    fn read_bounded_line(
        &mut self,
        deadline: Instant,
        running: &AtomicBool,
    ) -> Result<Vec<u8>, AttendedWheelOnConfirmationError> {
        let mut output = Vec::with_capacity(64);
        loop {
            if !running.load(Ordering::Acquire) {
                return Err(AttendedWheelOnConfirmationError::Interrupted);
            }
            let now = Instant::now();
            if now >= deadline {
                return Err(AttendedWheelOnConfirmationError::ResponseTimedOut {
                    maximum_wait: ATTENDED_CONFIRMATION_RESPONSE_TIMEOUT,
                });
            }
            let mut byte = [0_u8; 1];
            match self.file.read(&mut byte) {
                Ok(0) => return Err(AttendedWheelOnConfirmationError::EndOfInput),
                Ok(1) if byte[0] == b'\n' => {
                    if output.last() == Some(&b'\r') {
                        output.pop();
                    }
                    return Ok(output);
                }
                Ok(1) => {
                    if output.len() == ATTENDED_CONFIRMATION_LINE_MAX_BYTES {
                        return Err(AttendedWheelOnConfirmationError::LineTooLong {
                            maximum_bytes: ATTENDED_CONFIRMATION_LINE_MAX_BYTES,
                        });
                    }
                    output.push(byte[0]);
                }
                Ok(_) => unreachable!("one-byte reads return at most one byte"),
                Err(source) if source.kind() == io::ErrorKind::Interrupted => {}
                Err(source) if source.kind() == io::ErrorKind::WouldBlock => {
                    thread::sleep(
                        deadline
                            .saturating_duration_since(now)
                            .min(ATTENDED_CONFIRMATION_POLL_SLICE),
                    );
                }
                Err(source) => return Err(AttendedWheelOnConfirmationError::Input(source)),
            }
        }
    }
}

fn lower_hex_bytes(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
}

#[derive(Serialize)]
struct AttendedAttestationRecordV3<'a> {
    schema_version: u32,
    session_id: &'a str,
    issued_at_ns: u64,
    expires_at_ns: u64,
    wheels_attached: bool,
    clear_area_confirmed: bool,
    independent_power_cut_tested_and_reachable: bool,
    operator_attending: bool,
    confirmation_channel: &'static str,
    confirmation_context_sha256: String,
    confirmation_challenge_sha256: String,
    maximum_response_wait_ms: u64,
    maximum_consumption_delay_ms: u64,
    launch_sha256: String,
    policy_sha256: String,
    controller_profile_sha256: String,
    controller_contract_sha256: String,
    manifest_sha256: String,
    calibration_sha256: String,
    live_graph_launch_sha256: String,
    accessory_policy_sha256: String,
    onnx_runtime_sha256: String,
    superpoint_model_sha256: String,
    lightglue_model_sha256: String,
    consumption_status: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DirectoryIdentity {
    device: u64,
    inode: u64,
}

impl DirectoryIdentity {
    fn from_metadata(metadata: &std::fs::Metadata) -> Self {
        Self {
            device: metadata.dev(),
            inode: metadata.ino(),
        }
    }
}

#[derive(Debug)]
struct CommissioningStateRoot {
    diagnostic_path: PathBuf,
    directory: File,
    identity: DirectoryIdentity,
}

impl CommissioningStateRoot {
    fn inspect(path: &Path) -> Result<Self, CommissioningStateRootError> {
        let bytes = path.as_os_str().as_encoded_bytes();
        if bytes.is_empty()
            || bytes.len() > MAX_STATE_ROOT_PATH_BYTES
            || !path.is_absolute()
            || path == Path::new("/")
        {
            return Err(CommissioningStateRootError::InvalidPath(path.to_path_buf()));
        }
        let directory =
            open_directory_file(path).map_err(CommissioningStateRootError::OpenRetained)?;
        let metadata = directory
            .metadata()
            .map_err(CommissioningStateRootError::Inspect)?;
        if !metadata.file_type().is_dir() {
            return Err(CommissioningStateRootError::NotDirectory);
        }
        let expected_uid = rustix::process::geteuid().as_raw();
        if metadata.uid() != expected_uid {
            return Err(CommissioningStateRootError::OwnerMismatch {
                expected: expected_uid,
                actual: metadata.uid(),
            });
        }
        let mode = metadata.mode() & 0o777;
        if mode != 0o700 {
            return Err(CommissioningStateRootError::PermissionsTooBroad { mode });
        }
        let root = Self {
            diagnostic_path: path.to_path_buf(),
            directory,
            identity: DirectoryIdentity::from_metadata(&metadata),
        };
        root.verify_binding()?;
        Ok(root)
    }

    fn create_session_directory(
        &self,
        session_id: &str,
    ) -> Result<CommissioningArtifactDirectory, CommissioningAttestationError> {
        self.verify_binding()
            .map_err(CommissioningAttestationError::StateRoot)?;
        mkdirat(&self.directory, session_id, Mode::from_raw_mode(0o700)).map_err(|source| {
            CommissioningAttestationError::CreateSessionDirectory(errno_as_io(source))
        })?;
        let session_fd = openat(
            &self.directory,
            session_id,
            OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::empty(),
        )
        .map(File::from)
        .map_err(|source| {
            CommissioningAttestationError::OpenSessionDirectory(errno_as_io(source))
        })?;
        self.directory
            .sync_all()
            .map_err(CommissioningAttestationError::SyncStateRoot)?;
        self.verify_binding()
            .map_err(CommissioningAttestationError::StateRoot)?;
        let session_path = self.diagnostic_path.join(session_id);
        CommissioningArtifactDirectory::from_retained_directory(session_path, session_fd)
            .map_err(CommissioningAttestationError::ArtifactDirectory)
    }

    fn verify_binding(&self) -> Result<(), CommissioningStateRootError> {
        let retained = self
            .directory
            .metadata()
            .map_err(CommissioningStateRootError::Inspect)?;
        if !retained.file_type().is_dir()
            || DirectoryIdentity::from_metadata(&retained) != self.identity
        {
            return Err(CommissioningStateRootError::RetainedIdentityChanged);
        }
        let expected_uid = rustix::process::geteuid().as_raw();
        if retained.uid() != expected_uid {
            return Err(CommissioningStateRootError::OwnerMismatch {
                expected: expected_uid,
                actual: retained.uid(),
            });
        }
        let mode = retained.mode() & 0o777;
        if mode != 0o700 {
            return Err(CommissioningStateRootError::PermissionsTooBroad { mode });
        }
        let rebound = open_directory_file(&self.diagnostic_path)
            .map_err(CommissioningStateRootError::ReopenBinding)?;
        let observed = rebound
            .metadata()
            .map_err(CommissioningStateRootError::ReopenBinding)?;
        let observed_identity = DirectoryIdentity::from_metadata(&observed);
        if observed_identity != self.identity {
            return Err(CommissioningStateRootError::PathBindingChanged {
                expected_device: self.identity.device,
                expected_inode: self.identity.inode,
                observed_device: observed_identity.device,
                observed_inode: observed_identity.inode,
            });
        }
        Ok(())
    }
}

#[derive(Debug)]
pub enum CommissioningStateRootError {
    InvalidPath(PathBuf),
    OpenRetained(io::Error),
    Inspect(io::Error),
    ReopenBinding(io::Error),
    NotDirectory,
    RetainedIdentityChanged,
    PathBindingChanged {
        expected_device: u64,
        expected_inode: u64,
        observed_device: u64,
        observed_inode: u64,
    },
    OwnerMismatch {
        expected: u32,
        actual: u32,
    },
    PermissionsTooBroad {
        mode: u32,
    },
}

impl fmt::Display for CommissioningStateRootError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid commissioning state root: {self:?}")
    }
}

impl std::error::Error for CommissioningStateRootError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::OpenRetained(source) | Self::Inspect(source) | Self::ReopenBinding(source) => {
                Some(source)
            }
            _ => None,
        }
    }
}

#[derive(Debug)]
pub enum CommissioningStreamAdmissionError {
    Calibration(NanoCalibrationBindingError),
    VisualSourceId(kiko_base_commissioning::IdentifierError),
    ImuCalibrationId(kiko_base_commissioning::IdentifierError),
    VisualSourceMismatch,
    ImuCalibrationMismatch,
}

impl fmt::Display for CommissioningStreamAdmissionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "same-owner commissioning stream admission failed: {self:?}"
        )
    }
}

impl std::error::Error for CommissioningStreamAdmissionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Calibration(source) => Some(source),
            Self::VisualSourceId(source) | Self::ImuCalibrationId(source) => Some(source),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub enum CommissioningAttestationError {
    StreamSessionMismatch,
    InterruptedBeforeConsumption,
    ConfirmationContextMismatch,
    ConfirmationClockRegression,
    ConfirmationStale {
        age: Duration,
        maximum_age: Duration,
    },
    ExpiryOverflow,
    StateRoot(CommissioningStateRootError),
    CreateSessionDirectory(io::Error),
    OpenSessionDirectory(io::Error),
    SyncStateRoot(io::Error),
    Encode(serde_json::Error),
    WriteAttestation(io::Error),
    UnsafeAttestationFile,
    AttestationIdentityChanged,
    Journal(FileCommissioningJournalError),
    ArtifactDirectory(CommissioningArtifactDirectoryError),
    Authority(super::nano_base_commissioning::CommissioningAuthorityError),
}

impl fmt::Display for CommissioningAttestationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "attended wheel-on commissioning admission failed: {self:?}"
        )
    }
}

impl std::error::Error for CommissioningAttestationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::StateRoot(source) => Some(source),
            Self::CreateSessionDirectory(source)
            | Self::OpenSessionDirectory(source)
            | Self::SyncStateRoot(source)
            | Self::WriteAttestation(source) => Some(source),
            Self::Encode(source) => Some(source),
            Self::Journal(source) => Some(source),
            Self::ArtifactDirectory(source) => Some(source),
            Self::Authority(source) => Some(source),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub enum FreshAttendedCommissioningAdmissionError {
    Confirmation(Box<AttendedWheelOnConfirmationError>),
    Attestation(Box<CommissioningAttestationError>),
}

impl fmt::Display for FreshAttendedCommissioningAdmissionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Confirmation(source) => source.fmt(formatter),
            Self::Attestation(source) => source.fmt(formatter),
        }
    }
}

impl std::error::Error for FreshAttendedCommissioningAdmissionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Confirmation(source) => Some(source.as_ref()),
            Self::Attestation(source) => Some(source.as_ref()),
        }
    }
}

pub struct AdmittedNanoBaseCommissioningRun {
    policy: NanoBaseCommissioningPolicyV1,
    profile: NanoBaseCommissioningControllerProfileV1,
    server: ControllerServerConfigV3,
    authority: AdmittedAttendedCommissioning,
    confirmation_issued_at_ns: u64,
    stream: AdmittedCommissioningObservationStream,
    journal: FileCommissioningJournal,
    artifact_directory: CommissioningArtifactDirectory,
}

impl AdmittedNanoBaseCommissioningRun {
    pub const fn confirmation_issued_at_ns(&self) -> u64 {
        self.confirmation_issued_at_ns
    }

    pub async fn start_controller(
        self,
        clock_origin: Instant,
    ) -> Result<OwnedNanoBaseCommissioningRun, CommissioningControllerStartError> {
        let shutdown_timeout = self.server.coordinated_shutdown_budget();
        let client_config = self
            .profile
            .client_config(&self.server)
            .map_err(CommissioningControllerStartError::ClientConfig)?;
        let owner = V2ControllerOwner::start(self.server, self.profile.command_endpoint)
            .await
            .map_err(CommissioningControllerStartError::Owner)?;
        let (physical, initial_zero) =
            match PhysicalActuationSession::acquire_commissioning(client_config, clock_origin) {
                Ok(value) => value,
                Err(source) => {
                    let owner_shutdown = owner.shutdown(shutdown_timeout).await.err();
                    return Err(CommissioningControllerStartError::Acquire {
                        source,
                        owner_shutdown: owner_shutdown.map(Box::new),
                    });
                }
            };
        if !initial_zero.is_confirmed_zero() {
            let owner_shutdown = owner.shutdown(shutdown_timeout).await.err();
            return Err(CommissioningControllerStartError::InitialZeroNotExact {
                owner_shutdown: owner_shutdown.map(Box::new),
            });
        }
        Ok(OwnedNanoBaseCommissioningRun {
            policy: self.policy,
            authority: self.authority,
            actuator: NanoCommissioningActuator {
                physical,
                maximum_abs_pwm_percent: self.profile.maximum_abs_pwm_percent,
            },
            stream: self.stream,
            journal: self.journal,
            artifact_directory: self.artifact_directory,
            owner,
            shutdown_timeout,
        })
    }

    /// Start the same attended, commissioning-class controller owner while
    /// transferring the physical session into the guarded live MPC driver.
    /// The normal production controller class remains unreachable.
    #[cfg(feature = "nano-attended-navigation-trial")]
    pub async fn start_attended_navigation_trial_controller(
        self,
        clock_origin: Instant,
    ) -> Result<OwnedNanoAttendedNavigationTrialController, CommissioningControllerStartError> {
        let guard = self
            .authority
            .attended_navigation_trial_guard()
            .map_err(|_| CommissioningControllerStartError::AttendedTrialGuard {
                maximum_abs_pwm_percent: self.authority.maximum_abs_pwm_percent(),
                expires_at_ns: self.authority.expires_at_ns(),
            })?;
        let shutdown_timeout = self.server.coordinated_shutdown_budget();
        let client_config = self
            .profile
            .client_config(&self.server)
            .map_err(CommissioningControllerStartError::ClientConfig)?;
        let owner = V2ControllerOwner::start(self.server, self.profile.command_endpoint)
            .await
            .map_err(CommissioningControllerStartError::Owner)?;
        let (mut physical, initial_zero) =
            match PhysicalActuationSession::acquire_commissioning(client_config, clock_origin) {
                Ok(value) => value,
                Err(source) => {
                    let owner_shutdown = owner.shutdown(shutdown_timeout).await.err();
                    return Err(CommissioningControllerStartError::Acquire {
                        source,
                        owner_shutdown: owner_shutdown.map(Box::new),
                    });
                }
            };
        if !initial_zero.is_confirmed_zero() {
            let owner_shutdown = owner.shutdown(shutdown_timeout).await.err();
            return Err(CommissioningControllerStartError::InitialZeroNotExact {
                owner_shutdown: owner_shutdown.map(Box::new),
            });
        }
        let head_gaze_lease_issuer =
            match physical.install_head_gaze_base_interlock_from_initial_receipt(&initial_zero) {
                Ok(issuer) => issuer,
                Err(source) => {
                    let physical_stop = physical.disarm().err();
                    let owner_shutdown = owner.shutdown(shutdown_timeout).await.err();
                    return Err(CommissioningControllerStartError::AttendedTrialInterlock {
                        source,
                        physical_stop: physical_stop.map(Box::new),
                        owner_shutdown: owner_shutdown.map(Box::new),
                    });
                }
            };
        Ok(OwnedNanoAttendedNavigationTrialController {
            driver: Some(LiveMpcControlDriver::from_attended_commissioning(
                physical, guard,
            )),
            initial_zero: Some(initial_zero),
            head_gaze_lease_issuer: Some(head_gaze_lease_issuer),
            authority: self.authority,
            policy: self.policy,
            stream: self.stream,
            journal: self.journal,
            artifact_directory: self.artifact_directory,
            owner,
            shutdown_timeout,
        })
    }
}

#[derive(Debug)]
pub enum CommissioningControllerStartError {
    ClientConfig(ClientConfigError),
    Owner(V2ControllerOwnerStartError),
    Acquire {
        source: LiveActuationError,
        owner_shutdown: Option<Box<V2ControllerOwnerTerminationError>>,
    },
    InitialZeroNotExact {
        owner_shutdown: Option<Box<V2ControllerOwnerTerminationError>>,
    },
    #[cfg(feature = "nano-attended-navigation-trial")]
    AttendedTrialGuard {
        maximum_abs_pwm_percent: u8,
        expires_at_ns: u64,
    },
    #[cfg(feature = "nano-attended-navigation-trial")]
    AttendedTrialInterlock {
        source: LiveActuationError,
        physical_stop: Option<Box<LiveActuationError>>,
        owner_shutdown: Option<Box<V2ControllerOwnerTerminationError>>,
    },
}

impl fmt::Display for CommissioningControllerStartError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "commissioning controller ownership failed closed: {self:?}"
        )
    }
}

impl std::error::Error for CommissioningControllerStartError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ClientConfig(source) => Some(source),
            Self::Owner(source) => Some(source),
            Self::Acquire { source, .. } => Some(source),
            Self::InitialZeroNotExact { .. } => None,
            #[cfg(feature = "nano-attended-navigation-trial")]
            Self::AttendedTrialGuard { .. } => None,
            #[cfg(feature = "nano-attended-navigation-trial")]
            Self::AttendedTrialInterlock { source, .. } => Some(source),
        }
    }
}

/// Retains the controller server and attended evidence after the guarded MPC
/// driver moves into the sole live motion owner.
#[cfg(feature = "nano-attended-navigation-trial")]
#[must_use = "the attended controller owner must be shut down and inspected"]
pub struct OwnedNanoAttendedNavigationTrialController {
    driver: Option<LiveMpcControlDriver>,
    initial_zero: Option<AppliedCommandReceipt>,
    head_gaze_lease_issuer: Option<kiko_head_runtime::HeadGazeBaseZeroExclusiveLeaseIssuer>,
    authority: AdmittedAttendedCommissioning,
    policy: NanoBaseCommissioningPolicyV1,
    stream: AdmittedCommissioningObservationStream,
    journal: FileCommissioningJournal,
    artifact_directory: CommissioningArtifactDirectory,
    owner: V2ControllerOwner,
    shutdown_timeout: Duration,
}

#[cfg(feature = "nano-attended-navigation-trial")]
impl OwnedNanoAttendedNavigationTrialController {
    /// Single-use transfer into the sole navigation owner. The server and all
    /// attended evidence remain owned here for supervised terminal shutdown.
    pub fn take_motion_driver(
        &mut self,
    ) -> Result<
        (
            LiveMpcControlDriver,
            AppliedCommandReceipt,
            kiko_head_runtime::HeadGazeBaseZeroExclusiveLeaseIssuer,
        ),
        AttendedTrialDriverTransferError,
    > {
        match (
            self.driver.take(),
            self.initial_zero.take(),
            self.head_gaze_lease_issuer.take(),
        ) {
            (Some(driver), Some(initial_zero), Some(head_gaze_lease_issuer)) => {
                Ok((driver, initial_zero, head_gaze_lease_issuer))
            }
            (driver, initial_zero, head_gaze_lease_issuer) => {
                self.driver = driver;
                self.initial_zero = initial_zero;
                self.head_gaze_lease_issuer = head_gaze_lease_issuer;
                Err(AttendedTrialDriverTransferError)
            }
        }
    }

    pub const fn authority(&self) -> AdmittedAttendedCommissioning {
        self.authority
    }

    pub async fn shutdown_controller(mut self) -> Result<(), AttendedTrialControllerShutdownError> {
        let motion_stop = self
            .driver
            .as_mut()
            .map(LiveMpcControlDriver::disarm)
            .transpose()
            .map(|_| ())
            .err();
        let Self {
            driver: _,
            initial_zero: _,
            head_gaze_lease_issuer: _,
            authority: _,
            policy,
            stream,
            journal,
            artifact_directory,
            owner,
            shutdown_timeout,
        } = self;
        // Retain the complete attended evidence ownership until after the
        // physical client has stopped. This slice does not yet publish a
        // navigation-trial terminal record, so dropping these owners makes no
        // stronger durability claim.
        drop((policy, stream, journal, artifact_directory));
        let owner_shutdown = owner.shutdown(shutdown_timeout).await.err();
        match (motion_stop, owner_shutdown) {
            (None, None) => Ok(()),
            (motion_stop, owner_shutdown) => Err(AttendedTrialControllerShutdownError {
                motion_stop,
                owner_shutdown: owner_shutdown.map(Box::new),
            }),
        }
    }
}

#[cfg(feature = "nano-attended-navigation-trial")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttendedTrialDriverTransferError;

#[cfg(feature = "nano-attended-navigation-trial")]
impl fmt::Display for AttendedTrialDriverTransferError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("attended navigation trial driver was already transferred")
    }
}

#[cfg(feature = "nano-attended-navigation-trial")]
impl std::error::Error for AttendedTrialDriverTransferError {}

#[cfg(feature = "nano-attended-navigation-trial")]
#[derive(Debug)]
pub struct AttendedTrialControllerShutdownError {
    pub motion_stop: Option<LiveActuationError>,
    pub owner_shutdown: Option<Box<V2ControllerOwnerTerminationError>>,
}

#[cfg(feature = "nano-attended-navigation-trial")]
impl fmt::Display for AttendedTrialControllerShutdownError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "attended navigation trial controller shutdown failed: motion_stop={:?}; owner_shutdown={:?}",
            self.motion_stop, self.owner_shutdown
        )
    }
}

#[cfg(feature = "nano-attended-navigation-trial")]
impl std::error::Error for AttendedTrialControllerShutdownError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.motion_stop
            .as_ref()
            .map(|source| source as &(dyn std::error::Error + 'static))
            .or_else(|| {
                self.owner_shutdown
                    .as_deref()
                    .map(|source| source as &(dyn std::error::Error + 'static))
            })
    }
}

struct NanoCommissioningActuator {
    physical: PhysicalActuationSession,
    maximum_abs_pwm_percent: u8,
}

#[derive(Debug)]
enum NanoCommissioningActuatorError {
    PwmAboveProfile { left: i8, right: i8, maximum: u8 },
    TimerPwm(robot_protocol::v2::DomainError),
    Receipt(robot_protocol::AppliedPwmError),
    Controller(LiveActuationError),
    HostTimestampOutOfRange(u128),
}

impl fmt::Display for NanoCommissioningActuatorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PwmAboveProfile {
                left,
                right,
                maximum,
            } => write!(
                formatter,
                "commissioning PWM ({left}, {right}) exceeds the profile bound ±{maximum}%"
            ),
            Self::TimerPwm(source) => write!(formatter, "invalid timer PWM: {source}"),
            Self::Receipt(source) => {
                write!(formatter, "invalid applied commissioning receipt: {source}")
            }
            Self::Controller(source) => write!(formatter, "controller request failed: {source}"),
            Self::HostTimestampOutOfRange(value) => write!(
                formatter,
                "controller acknowledgement timestamp {value}ns exceeds the u64 evidence domain"
            ),
        }
    }
}

impl std::error::Error for NanoCommissioningActuatorError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::TimerPwm(source) => Some(source),
            Self::Receipt(source) => Some(source),
            Self::Controller(source) => Some(source),
            _ => None,
        }
    }
}

impl NanoCommissioningActuator {
    fn apply_values(
        &mut self,
        left: i8,
        right: i8,
    ) -> Result<
        super::nano_base_commissioning::ExactCommissioningControllerReceipt,
        NanoCommissioningActuatorError,
    > {
        let maximum = i16::from(self.maximum_abs_pwm_percent);
        if i16::from(left).abs() > maximum || i16::from(right).abs() > maximum {
            return Err(NanoCommissioningActuatorError::PwmAboveProfile {
                left,
                right,
                maximum: self.maximum_abs_pwm_percent,
            });
        }
        let timer =
            TimerPwm::try_new(left, right).map_err(NanoCommissioningActuatorError::TimerPwm)?;
        let receipt = self
            .physical
            .apply_commissioning_pwm(timer)
            .map_err(NanoCommissioningActuatorError::Controller)?;
        let observed_at = receipt.acknowledged_at().nanos_since_clock_start();
        let observed_at_ns = u64::try_from(observed_at)
            .map_err(|_| NanoCommissioningActuatorError::HostTimestampOutOfRange(observed_at))?;
        super::nano_base_commissioning::ExactCommissioningControllerReceipt::try_new(
            observed_at_ns,
            u64::from(receipt.sequence().get()),
            receipt.applied_timer_pwm().left().get(),
            receipt.applied_timer_pwm().right().get(),
        )
        .map_err(NanoCommissioningActuatorError::Receipt)
    }
}

impl SoleCommissioningActuator for NanoCommissioningActuator {
    type Error = NanoCommissioningActuatorError;

    fn apply(
        &mut self,
        command: kiko_base_commissioning::CanonicalPwmCommand,
    ) -> Result<super::nano_base_commissioning::ExactCommissioningControllerReceipt, Self::Error>
    {
        self.apply_values(command.left().get(), command.right().get())
    }

    fn emergency_zero(
        &mut self,
    ) -> Result<super::nano_base_commissioning::ExactCommissioningControllerReceipt, Self::Error>
    {
        self.apply_values(0, 0)
    }
}

#[must_use = "the sole STM32 owner must be shut down and its result inspected"]
pub struct OwnedNanoBaseCommissioningRun {
    policy: NanoBaseCommissioningPolicyV1,
    authority: AdmittedAttendedCommissioning,
    actuator: NanoCommissioningActuator,
    stream: AdmittedCommissioningObservationStream,
    journal: FileCommissioningJournal,
    artifact_directory: CommissioningArtifactDirectory,
    owner: V2ControllerOwner,
    shutdown_timeout: Duration,
}

impl OwnedNanoBaseCommissioningRun {
    /// Fail closed before the sampling loop starts.
    ///
    /// This is used when an external evidence sink fails after controller
    /// ownership was acquired. It creates the normal session, records a
    /// terminal supervisor fault, requests and verifies a fresh exact zero,
    /// and then performs the same bounded controller-owner shutdown as
    /// [`Self::execute`]. The returned compound error retains both the terminal
    /// stop evidence and any owner-shutdown uncertainty.
    pub async fn terminate_before_execution(
        self,
        signal: CommissioningExternalSignal,
    ) -> OwnedCommissioningRunError {
        let Self {
            policy,
            authority,
            actuator,
            stream: _,
            journal,
            artifact_directory: _,
            owner,
            shutdown_timeout,
        } = self;
        let runtime =
            match NanoBaseCommissioningSession::start(policy, authority, actuator, journal) {
                Ok(mut session) => {
                    let terminal = session.terminate(signal).expect_err(
                        "explicit termination always returns compound failure evidence",
                    );
                    CommissioningRuntimeError::Terminal(Box::new(terminal))
                }
                Err(source) => CommissioningRuntimeError::Start(Box::new(source)),
            };
        let owner_shutdown = owner.shutdown(shutdown_timeout).await.err();
        OwnedCommissioningRunError {
            runtime,
            owner_shutdown: owner_shutdown.map(Box::new),
        }
    }

    pub async fn execute<R>(
        self,
        source: &mut super::nano_base_commissioning_live::NanoCommissioningLiveObservationSource,
        reporter: &mut R,
    ) -> Result<NanoBaseCommissioningProposal, OwnedCommissioningRunError>
    where
        R: CommissioningProgressReporter,
    {
        if source.stream_binding().session_id != self.stream.session_id
            || source.stream_binding().clock_epoch != self.stream.clock_epoch
            || source.stream_binding().visual_velocity_source_id
                != self.stream.visual_velocity_source_id
            || source.stream_binding().imu_calibration_id != self.stream.imu_calibration_id
        {
            let owner_shutdown = self.owner.shutdown(self.shutdown_timeout).await.err();
            return Err(OwnedCommissioningRunError {
                runtime: CommissioningRuntimeError::SourceBindingMismatch,
                owner_shutdown: owner_shutdown.map(Box::new),
            });
        }
        let Self {
            policy,
            authority,
            actuator,
            stream,
            journal,
            artifact_directory,
            owner,
            shutdown_timeout,
        } = self;
        let runtime = execute_commissioning_runtime(
            CommissioningRuntimeInputs {
                policy,
                authority,
                actuator,
                journal,
                artifact_directory: &artifact_directory,
                stream: &stream,
            },
            source,
            reporter,
        );
        let owner_shutdown = owner.shutdown(shutdown_timeout).await.err();
        match (runtime, owner_shutdown) {
            (Ok(proposal), None) => Ok(proposal),
            (Ok(_), Some(owner_shutdown)) => Err(OwnedCommissioningRunError {
                runtime: CommissioningRuntimeError::OwnerShutdownAfterProposal,
                owner_shutdown: Some(Box::new(owner_shutdown)),
            }),
            (Err(runtime), owner_shutdown) => Err(OwnedCommissioningRunError {
                runtime,
                owner_shutdown: owner_shutdown.map(Box::new),
            }),
        }
    }
}

#[derive(Debug)]
pub struct OwnedCommissioningRunError {
    pub runtime: CommissioningRuntimeError,
    pub owner_shutdown: Option<Box<V2ControllerOwnerTerminationError>>,
}

impl fmt::Display for OwnedCommissioningRunError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "owned commissioning run failed: {}; owner_shutdown={:?}",
            self.runtime, self.owner_shutdown
        )
    }
}

impl std::error::Error for OwnedCommissioningRunError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.runtime)
    }
}

#[derive(Debug)]
pub enum CommissioningRuntimeError {
    SourceBindingMismatch,
    Start(Box<NanoBaseCommissioningFailure>),
    Source {
        source: Box<dyn std::error::Error + Send + Sync>,
        terminal: Box<NanoBaseCommissioningFailure>,
    },
    Reporter {
        source: Box<dyn std::error::Error + Send + Sync>,
        terminal: Box<NanoBaseCommissioningFailure>,
    },
    Terminal(Box<NanoBaseCommissioningFailure>),
    Session(Box<NanoBaseCommissioningFailure>),
    Publish(Box<NanoBaseCommissioningPublishFailure>),
    OwnerShutdownAfterProposal,
}

impl fmt::Display for CommissioningRuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "commissioning runtime failed closed: {self:?}")
    }
}

impl std::error::Error for CommissioningRuntimeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Start(source) | Self::Terminal(source) | Self::Session(source) => {
                Some(source.as_ref())
            }
            Self::Source { source, .. } | Self::Reporter { source, .. } => Some(source.as_ref()),
            Self::Publish(source) => Some(source.as_ref()),
            Self::SourceBindingMismatch | Self::OwnerShutdownAfterProposal => None,
        }
    }
}

struct CommissioningRuntimeInputs<'a, A, J> {
    policy: NanoBaseCommissioningPolicyV1,
    authority: AdmittedAttendedCommissioning,
    actuator: A,
    journal: J,
    artifact_directory: &'a CommissioningArtifactDirectory,
    stream: &'a AdmittedCommissioningObservationStream,
}

fn execute_commissioning_runtime<A, J, S, R>(
    inputs: CommissioningRuntimeInputs<'_, A, J>,
    source: &mut S,
    reporter: &mut R,
) -> Result<NanoBaseCommissioningProposal, CommissioningRuntimeError>
where
    A: SoleCommissioningActuator,
    J: super::nano_base_commissioning::DurableCommissioningJournal,
    S: CommissioningObservationSource,
    R: CommissioningProgressReporter,
{
    let CommissioningRuntimeInputs {
        policy,
        authority,
        actuator,
        journal,
        artifact_directory,
        stream,
    } = inputs;
    let mut session = NanoBaseCommissioningSession::start(policy, authority, actuator, journal)
        .map_err(|source| CommissioningRuntimeError::Start(Box::new(source)))?;
    loop {
        let progress = session.progress();
        if progress.state == CommissioningState::Completed {
            return session
                .publish_proposal(artifact_directory)
                .map_err(|source| CommissioningRuntimeError::Publish(Box::new(source)));
        }
        let request = CommissioningSamplingRequest {
            progress,
            expected_receipt: session.expected_receipt(),
        };
        if let Err(error) = reporter.before_observation(request) {
            let terminal = session
                .terminate(CommissioningExternalSignal::SupervisorFault)
                .expect_err("explicit termination always returns compound failure evidence");
            return Err(CommissioningRuntimeError::Reporter {
                source: Box::new(error),
                terminal: Box::new(terminal),
            });
        }
        let event = match source.next_observation(request) {
            Ok(event) => event,
            Err(error) => {
                let signal = source.terminal_signal_for_error(&error);
                let terminal = session
                    .terminate(signal)
                    .expect_err("explicit termination always returns compound failure evidence");
                return Err(CommissioningRuntimeError::Source {
                    source: Box::new(error),
                    terminal: Box::new(terminal),
                });
            }
        };
        let outcome = match event {
            CommissioningObservationEvent::Observation(_) => {
                CommissioningObservationKind::SampleReady
            }
            CommissioningObservationEvent::Terminal(_) => {
                CommissioningObservationKind::TerminalSignal
            }
        };
        if let Err(error) = reporter.after_observation(request, outcome) {
            let terminal = session
                .terminate(CommissioningExternalSignal::SupervisorFault)
                .expect_err("explicit termination always returns compound failure evidence");
            return Err(CommissioningRuntimeError::Reporter {
                source: Box::new(error),
                terminal: Box::new(terminal),
            });
        }
        let observation = match event {
            CommissioningObservationEvent::Observation(observation) => observation,
            CommissioningObservationEvent::Terminal(signal) => {
                let terminal = session
                    .terminate(signal)
                    .expect_err("explicit termination always returns compound failure evidence");
                return Err(CommissioningRuntimeError::Terminal(Box::new(terminal)));
            }
        };
        let receipt = session.expected_receipt();
        let sample = NanoBaseCommissioningSampleV1Dto {
            now_ns: observation.now_ns,
            evidence: CommissioningEvidenceV1Dto {
                controller_session_id: policy
                    .commissioning()
                    .expected_controller_session_id()
                    .as_str()
                    .to_owned(),
                visual_velocity_source_id: stream.visual_velocity_source_id.as_str().to_owned(),
                imu_calibration_id: stream.imu_calibration_id.as_str().to_owned(),
                controller_observed_at_ns: receipt.observed_at_ns(),
                visual_observed_at_ns: observation.visual_observed_at_ns,
                imu_observed_at_ns: observation.imu_observed_at_ns,
                applied_command_sequence: receipt.applied_command_sequence(),
                applied_left_pwm_percent: receipt.applied_pwm().left().get(),
                applied_right_pwm_percent: receipt.applied_pwm().right().get(),
                visual_forward_velocity_mps: observation.visual_body_forward_velocity_mps,
                calibrated_imu_yaw_rate_rad_s: observation.calibrated_imu_yaw_rate_rad_s,
            },
            visual_body_lateral_velocity_mps: observation.visual_body_lateral_velocity_mps,
            visual_body_frame_id: BODY_FRAME_ID.to_owned(),
        };
        session
            .advance(sample, CommissioningExternalSignal::Continue)
            .map_err(|source| CommissioningRuntimeError::Session(Box::new(source)))?;
    }
}

pub fn prepare_nano_base_commissioning(
    deployment_root: &Path,
    launch_relative_path: ArtifactRelativePath,
    state_root: &Path,
) -> Result<PreparedNanoBaseCommissioning, NanoBaseCommissioningPreparationError> {
    let state_root = CommissioningStateRoot::inspect(state_root)
        .map_err(NanoBaseCommissioningPreparationError::StateRoot)?;
    let launch_limit = DeploymentAssetByteLimit::try_new(
        u64::try_from(MAX_NANO_BASE_COMMISSIONING_LAUNCH_JSON_BYTES)
            .expect("launch bound fits u64"),
    )
    .expect("launch bound is valid");
    let launch_source = load_deployment_asset(deployment_root, launch_relative_path, launch_limit)
        .map_err(NanoBaseCommissioningPreparationError::LaunchLoad)?;
    let launch = NanoBaseCommissioningLaunchV1::parse_json(launch_source.bytes())
        .map_err(NanoBaseCommissioningPreparationError::LaunchParse)?;
    for role in CommissioningAssetRole::ALL {
        if launch.asset(role).relative_path == *launch_source.relative_path() {
            return Err(NanoBaseCommissioningPreparationError::AssetAliasesLaunch { role });
        }
    }
    let load = |role| {
        launch
            .asset(role)
            .load_exact(deployment_root)
            .map_err(|source| NanoBaseCommissioningPreparationError::AssetLoad { role, source })
    };
    let policy_source = load(CommissioningAssetRole::Policy)?;
    let controller_profile_source = load(CommissioningAssetRole::ControllerProfile)?;
    let controller_server_source = load(CommissioningAssetRole::ControllerServerContract)?;
    let manifest_source = load(CommissioningAssetRole::DeviceManifest)?;
    let calibration_source = load(CommissioningAssetRole::CalibrationArtifact)?;
    let live_graph_launch_source = load(CommissioningAssetRole::LiveGraphLaunch)?;

    let policy = NanoBaseCommissioningPolicyV1::parse_json(policy_source.bytes())
        .map_err(NanoBaseCommissioningPreparationError::Policy)?;
    let profile =
        NanoBaseCommissioningControllerProfileV1::parse_json(controller_profile_source.bytes())
            .map_err(NanoBaseCommissioningPreparationError::ControllerProfile)?;
    let server = ControllerServerConfigV3::parse_json(controller_server_source.bytes())
        .map_err(NanoBaseCommissioningPreparationError::ControllerContract)?;
    let loaded_manifest = load_expected_manifest_v3_from_slice(manifest_source.bytes())
        .map_err(NanoBaseCommissioningPreparationError::Manifest)?;
    let manifest = loaded_manifest.into_manifest();
    let calibration = NanoCalibrationArtifactV1::parse_json(calibration_source.bytes())
        .map_err(NanoBaseCommissioningPreparationError::Calibration)?;
    let live_graph = NanoAgentLaunchV3::parse_json(live_graph_launch_source.bytes())
        .map_err(NanoBaseCommissioningPreparationError::LiveGraphLaunch)?;
    bind_prepared_inputs(
        &launch,
        policy,
        &profile,
        &server,
        &manifest,
        &calibration,
        &live_graph,
    )?;
    let load_live_graph_asset = |role| {
        let binding = live_graph.asset(role);
        if binding.relative_path() == live_graph_launch_source.relative_path() {
            return Err(
                NanoBaseCommissioningPreparationError::LiveGraphAssetAliasesLaunch { role },
            );
        }
        binding.load_exact(deployment_root).map_err(|source| {
            NanoBaseCommissioningPreparationError::LiveGraphAssetLoad { role, source }
        })
    };
    let accessory_policy_source = load_live_graph_asset(NanoLaunchAssetRole::AgentPolicy)?;
    let accessory_policy = NanoAgentPolicyConfigV3::parse_json(accessory_policy_source.bytes())
        .map_err(NanoBaseCommissioningPreparationError::AccessoryPolicy)?;
    let accessory_policy = accessory_policy
        .bind_accessories_to_manifest(manifest.as_inventory())
        .map_err(NanoBaseCommissioningPreparationError::AccessoryManifestBinding)?;
    let onnx_runtime_library = load_live_graph_asset(NanoLaunchAssetRole::OnnxRuntimeLibrary)?;
    let superpoint_model = load_live_graph_asset(NanoLaunchAssetRole::SuperpointModel)?;
    let lightglue_model = load_live_graph_asset(NanoLaunchAssetRole::LightglueModel)?;
    let load_face_asset = |role| {
        let binding = live_graph.face_perception().asset(role);
        if binding.relative_path() == live_graph_launch_source.relative_path() {
            return Err(
                NanoBaseCommissioningPreparationError::LiveGraphFaceAssetAliasesLaunch { role },
            );
        }
        binding.load_exact(deployment_root).map_err(|source| {
            NanoBaseCommissioningPreparationError::LiveGraphFaceAssetLoad { role, source }
        })
    };
    let frontal_face_cascade = load_face_asset(NanoFaceCascadeAssetRole::FrontalFace)?;
    let profile_face_cascade = load_face_asset(NanoFaceCascadeAssetRole::ProfileFace)?;

    Ok(PreparedNanoBaseCommissioning {
        state_root,
        launch,
        policy,
        profile,
        server,
        calibration,
        live_graph,
        accessory_policy,
        inputs: LoadedNanoBaseCommissioningInputs {
            launch_source,
            policy_source,
            controller_profile_source,
            controller_server_source,
            manifest_source,
            calibration_source,
            live_graph_launch_source,
            accessory_policy_source,
            onnx_runtime_library,
            superpoint_model,
            lightglue_model,
            frontal_face_cascade,
            profile_face_cascade,
        },
    })
}

fn bind_prepared_inputs(
    launch: &NanoBaseCommissioningLaunchV1,
    policy: NanoBaseCommissioningPolicyV1,
    profile: &NanoBaseCommissioningControllerProfileV1,
    server: &ControllerServerConfigV3,
    manifest: &DeviceInventoryManifestV3,
    calibration: &NanoCalibrationArtifactV1,
    live_graph: &NanoAgentLaunchV3,
) -> Result<(), NanoBaseCommissioningPreparationError> {
    if launch.session_id() != profile.session_id() {
        return Err(NanoBaseCommissioningPreparationError::SessionIdMismatch);
    }
    if profile.controller_session_id != policy.commissioning().expected_controller_session_id() {
        return Err(NanoBaseCommissioningPreparationError::ControllerSessionIdMismatch);
    }
    if policy.commissioning().max_abs_pwm_percent().get() > profile.maximum_abs_pwm_percent {
        return Err(
            NanoBaseCommissioningPreparationError::PolicyPwmAboveProfile {
                policy: policy.commissioning().max_abs_pwm_percent().get(),
                profile: profile.maximum_abs_pwm_percent,
            },
        );
    }
    let largest_step = policy
        .commissioning()
        .symmetric_pwm_percent()
        .get()
        .max(policy.commissioning().spin_pwm_percent().get());
    if largest_step > profile.maximum_command_step_percent {
        return Err(
            NanoBaseCommissioningPreparationError::PolicyStepAboveProfile {
                policy: largest_step,
                profile: profile.maximum_command_step_percent,
            },
        );
    }
    let maximum_admitted_sample_gap_ns = policy
        .maximum_sample_gap_ns()
        .get()
        .min(policy.fit().max_sample_period_ns().get());
    let required_lease_ns = maximum_admitted_sample_gap_ns
        .checked_add(profile.applied_ack_timeout.get())
        .ok_or(NanoBaseCommissioningPreparationError::LeaseArithmeticOverflow)?;
    let admitted_lease_ns = u64::from(profile.command_lease.get()) * 1_000_000;
    if required_lease_ns >= admitted_lease_ns {
        return Err(
            NanoBaseCommissioningPreparationError::CommandLeaseTooShort {
                required_exclusive_upper_bound_ns: required_lease_ns,
                admitted_ns: admitted_lease_ns,
            },
        );
    }
    if profile.applied_ack_timeout.get() > policy.commissioning().application_timeout_ns().get() {
        return Err(
            NanoBaseCommissioningPreparationError::AppliedAckExceedsApplicationTimeout {
                applied_ack_timeout_ns: profile.applied_ack_timeout.get(),
                application_timeout_ns: policy.commissioning().application_timeout_ns().get(),
            },
        );
    }
    if policy.commissioning().max_total_duration_ns().get() >= profile.attestation_lifetime_ns.get()
    {
        return Err(
            NanoBaseCommissioningPreparationError::AttestationCannotContainRun {
                run_ns: policy.commissioning().max_total_duration_ns().get(),
                attestation_ns: profile.attestation_lifetime_ns.get(),
            },
        );
    }
    if server.controller_session_class() != ControllerSessionClass::AttendedWheelOnCommissioning {
        return Err(NanoBaseCommissioningPreparationError::ControllerClassNotAttended);
    }
    if server.expected_physical_stop_semantics() != PhysicalStopSemantics::Unverified {
        return Err(NanoBaseCommissioningPreparationError::CommissioningClaimedVerifiedStop);
    }
    if server.hardware_profile_claim_id() != profile.expected_hardware_profile_claim_id.as_ref() {
        return Err(NanoBaseCommissioningPreparationError::HardwareProfileClaimMismatch);
    }
    if server.expected_max_abs_pwm_percent().get() < profile.maximum_abs_pwm_percent {
        return Err(
            NanoBaseCommissioningPreparationError::ProfilePwmAboveController {
                profile: profile.maximum_abs_pwm_percent,
                controller: server.expected_max_abs_pwm_percent().get(),
            },
        );
    }
    if profile.maximum_command_step_percent > server.maximum_command_step_percent() {
        return Err(
            NanoBaseCommissioningPreparationError::ProfileStepAboveController {
                profile: profile.maximum_command_step_percent,
                controller: server.maximum_command_step_percent(),
            },
        );
    }
    if manifest.controller_session_class() != ControllerSessionClass::AttendedWheelOnCommissioning
        || manifest.expected_physical_stop_semantics() != PhysicalStopSemantics::Unverified
    {
        return Err(NanoBaseCommissioningPreparationError::ManifestNotAttended);
    }
    let expected_inventory = manifest.as_inventory();
    let expected = expected_inventory.stm32();
    if expected.safety_class() != ControllerSafetyClass::AttendedWheelOnCommissioning {
        return Err(NanoBaseCommissioningPreparationError::ManifestNotAttended);
    }
    if manifest.expected_max_abs_pwm_percent() != server.expected_max_abs_pwm_percent() {
        return Err(
            NanoBaseCommissioningPreparationError::ManifestControllerCapMismatch {
                manifest: manifest.expected_max_abs_pwm_percent().get(),
                controller: server.expected_max_abs_pwm_percent().get(),
            },
        );
    }
    if server.serial_device() != Path::new(expected.serial_path().as_str())
        || server.controller_uid() != *expected.controller_uid()
        || server.firmware_abi().get() != expected.firmware_abi()
        || server.firmware_build_id().get() != expected.firmware_build_id()
        || server.actuator_config_fingerprint() != *expected.hardware_profile()
    {
        return Err(NanoBaseCommissioningPreparationError::ControllerManifestMismatch);
    }
    let expected_endpoint = format!("udp://{}", profile.command_endpoint);
    if expected.control_endpoint().as_str() != expected_endpoint {
        return Err(NanoBaseCommissioningPreparationError::ControllerEndpointMismatch);
    }
    calibration
        .require_manifest_oak_mxid(expected_inventory.oak().mxid().as_str())
        .map_err(NanoBaseCommissioningPreparationError::CalibrationBinding)?;
    let live_calibration = live_graph.calibration_artifact().asset();
    if live_calibration.relative_path() != &launch.calibration_artifact.relative_path
        || live_calibration.expected_sha256() != &launch.calibration_artifact.expected_sha256
    {
        return Err(NanoBaseCommissioningPreparationError::LiveGraphCalibrationMismatch);
    }
    if calibration.imu_calibration_id().as_str()
        != policy
            .commissioning()
            .expected_imu_calibration_id()
            .as_str()
    {
        return Err(NanoBaseCommissioningPreparationError::CalibrationPolicyMismatch);
    }
    profile
        .client_config(server)
        .map_err(NanoBaseCommissioningPreparationError::ClientConfig)?;
    Ok(())
}

#[derive(Debug)]
pub enum NanoBaseCommissioningPreparationError {
    StateRoot(CommissioningStateRootError),
    LaunchLoad(DeploymentAssetLoadError),
    LaunchParse(NanoBaseCommissioningLaunchParseError),
    AssetAliasesLaunch {
        role: CommissioningAssetRole,
    },
    AssetLoad {
        role: CommissioningAssetRole,
        source: NanoBaseCommissioningAssetLoadError,
    },
    Policy(NanoBaseCommissioningPolicyParseError),
    ControllerProfile(NanoBaseCommissioningControllerProfileParseError),
    ControllerContract(ServerConfigError),
    Manifest(ManifestLoadError),
    Calibration(NanoCalibrationArtifactParseError),
    LiveGraphLaunch(NanoAgentLaunchParseError),
    LiveGraphCalibrationMismatch,
    LiveGraphAssetAliasesLaunch {
        role: NanoLaunchAssetRole,
    },
    LiveGraphAssetLoad {
        role: NanoLaunchAssetRole,
        source: NanoLaunchBoundAssetLoadError,
    },
    LiveGraphFaceAssetAliasesLaunch {
        role: NanoFaceCascadeAssetRole,
    },
    LiveGraphFaceAssetLoad {
        role: NanoFaceCascadeAssetRole,
        source: NanoLaunchBoundAssetLoadError,
    },
    AccessoryPolicy(NanoAgentPolicyConfigParseError),
    AccessoryManifestBinding(NanoAccessoryManifestBindingError),
    SessionIdMismatch,
    ControllerSessionIdMismatch,
    PolicyPwmAboveProfile {
        policy: u8,
        profile: u8,
    },
    PolicyStepAboveProfile {
        policy: u8,
        profile: u8,
    },
    LeaseArithmeticOverflow,
    CommandLeaseTooShort {
        required_exclusive_upper_bound_ns: u64,
        admitted_ns: u64,
    },
    AppliedAckExceedsApplicationTimeout {
        applied_ack_timeout_ns: u64,
        application_timeout_ns: u64,
    },
    AttestationCannotContainRun {
        run_ns: u64,
        attestation_ns: u64,
    },
    ControllerClassNotAttended,
    CommissioningClaimedVerifiedStop,
    HardwareProfileClaimMismatch,
    ProfilePwmAboveController {
        profile: u8,
        controller: u8,
    },
    ProfileStepAboveController {
        profile: u8,
        controller: u8,
    },
    ManifestNotAttended,
    ManifestControllerCapMismatch {
        manifest: u8,
        controller: u8,
    },
    ControllerManifestMismatch,
    ControllerEndpointMismatch,
    CalibrationBinding(NanoCalibrationBindingError),
    CalibrationPolicyMismatch,
    ClientConfig(ClientConfigError),
}

impl fmt::Display for NanoBaseCommissioningPreparationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "Nano base-commissioning preflight failed before hardware: {self:?}"
        )
    }
}

impl std::error::Error for NanoBaseCommissioningPreparationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::StateRoot(source) => Some(source),
            Self::LaunchLoad(source) => Some(source),
            Self::LaunchParse(source) => Some(source),
            Self::AssetLoad { source, .. } => Some(source),
            Self::Policy(source) => Some(source),
            Self::ControllerProfile(source) => Some(source),
            Self::ControllerContract(source) => Some(source),
            Self::Manifest(source) => Some(source),
            Self::Calibration(source) => Some(source),
            Self::LiveGraphLaunch(source) => Some(source),
            Self::LiveGraphAssetLoad { source, .. }
            | Self::LiveGraphFaceAssetLoad { source, .. } => Some(source),
            Self::AccessoryPolicy(source) => Some(source),
            Self::AccessoryManifestBinding(source) => Some(source),
            Self::CalibrationBinding(source) => Some(source),
            Self::ClientConfig(source) => Some(source),
            _ => None,
        }
    }
}

fn parse_profile_claim(
    value: String,
) -> Result<Box<str>, NanoBaseCommissioningControllerProfileParseError> {
    if value.is_empty()
        || value.len() > 128
        || !value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric()
                || matches!(byte, b'-' | b'_' | b'.' | b':' | b'/' | b'@' | b'+')
        })
    {
        return Err(NanoBaseCommissioningControllerProfileParseError::InvalidHardwareProfileClaim);
    }
    Ok(value.into_boxed_str())
}

fn parse_bounded_id(
    field: &'static str,
    value: String,
) -> Result<BoundedId, kiko_base_commissioning::IdentifierError> {
    BoundedId::parse_str(field, &value)
}

fn write_new_durable(
    directory: &CommissioningArtifactDirectory,
    name: &str,
    bytes: &[u8],
) -> Result<(), CommissioningAttestationError> {
    directory
        .verify_binding()
        .map_err(CommissioningAttestationError::ArtifactDirectory)?;
    let descriptor = openat(
        directory.directory(),
        name,
        OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::from_raw_mode(0o600),
    )
    .map_err(|source| CommissioningAttestationError::WriteAttestation(errno_as_io(source)))?;
    let initial = fstat(&descriptor)
        .map_err(|source| CommissioningAttestationError::WriteAttestation(errno_as_io(source)))?;
    if FileType::from_raw_mode(initial.st_mode) != FileType::RegularFile
        || initial.st_nlink != 1
        || initial.st_uid != rustix::process::geteuid().as_raw()
        || u32::from(initial.st_mode) & 0o777 != 0o600
    {
        return Err(CommissioningAttestationError::UnsafeAttestationFile);
    }
    let mut file = File::from(descriptor);
    file.write_all(bytes)
        .and_then(|()| file.sync_all())
        .map_err(CommissioningAttestationError::WriteAttestation)?;
    let synchronized = fstat(&file)
        .map_err(|source| CommissioningAttestationError::WriteAttestation(errno_as_io(source)))?;
    let named = statat(directory.directory(), name, AtFlags::SYMLINK_NOFOLLOW)
        .map_err(|source| CommissioningAttestationError::WriteAttestation(errno_as_io(source)))?;
    if synchronized.st_dev != initial.st_dev
        || synchronized.st_ino != initial.st_ino
        || synchronized.st_nlink != 1
        || named.st_dev != initial.st_dev
        || named.st_ino != initial.st_ino
    {
        return Err(CommissioningAttestationError::AttestationIdentityChanged);
    }
    fsync(directory.directory())
        .map_err(|source| CommissioningAttestationError::WriteAttestation(errno_as_io(source)))?;
    let named_after = statat(directory.directory(), name, AtFlags::SYMLINK_NOFOLLOW)
        .map_err(|source| CommissioningAttestationError::WriteAttestation(errno_as_io(source)))?;
    if named_after.st_dev != initial.st_dev || named_after.st_ino != initial.st_ino {
        return Err(CommissioningAttestationError::AttestationIdentityChanged);
    }
    directory
        .verify_binding()
        .map_err(CommissioningAttestationError::ArtifactDirectory)
}

fn open_directory_file(path: &Path) -> io::Result<File> {
    OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_DIRECTORY | libc::O_CLOEXEC)
        .open(path)
}

fn errno_as_io(source: Errno) -> io::Error {
    io::Error::from_raw_os_error(source.raw_os_error())
}

fn parse_lower_hex_exact(value: &str) -> Option<[u8; 32]> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return None;
    }
    let mut output = [0_u8; 32];
    for (index, byte) in output.iter_mut().enumerate() {
        *byte = (hex_nibble(value.as_bytes()[index * 2]) << 4)
            | hex_nibble(value.as_bytes()[index * 2 + 1]);
    }
    Some(output)
}

const fn hex_nibble(byte: u8) -> u8 {
    match byte {
        b'0'..=b'9' => byte - b'0',
        b'a'..=b'f' => byte - b'a' + 10,
        _ => 0,
    }
}

fn canonical_sha256(digest: [u8; 32]) -> String {
    let mut output = String::with_capacity(71);
    output.push_str("sha256:");
    for byte in digest {
        use fmt::Write as _;
        write!(&mut output, "{byte:02x}").expect("String formatting is infallible");
    }
    output
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::fs::Permissions;
    use std::os::unix::fs::PermissionsExt;
    use std::sync::atomic::{AtomicU64, Ordering};

    use serde_json::{Value, json};

    use super::*;

    static NEXT_TEST_DIRECTORY: AtomicU64 = AtomicU64::new(1);

    trait AmbiguousIfClone<A> {
        fn marker() {}
    }

    impl<T: ?Sized> AmbiguousIfClone<()> for T {}

    struct CloneMarker;

    impl<T: Clone> AmbiguousIfClone<CloneMarker> for T {}

    struct FixedChallenges {
        values: VecDeque<[u8; ATTENDED_CONFIRMATION_CHALLENGE_BYTES]>,
    }

    impl ConfirmationChallengeSource for FixedChallenges {
        fn next_challenge(
            &mut self,
        ) -> Result<[u8; ATTENDED_CONFIRMATION_CHALLENGE_BYTES], AttendedWheelOnConfirmationError>
        {
            Ok(self
                .values
                .pop_front()
                .expect("test supplies one challenge per exact prompt"))
        }
    }

    enum ScriptedResponse {
        Line(Vec<u8>),
        EndOfInput,
        TimedOut,
    }

    struct ScriptedTerminal {
        is_terminal: bool,
        responses: VecDeque<ScriptedResponse>,
        output: String,
        actions: Vec<&'static str>,
    }

    impl ScriptedTerminal {
        fn new(is_terminal: bool, responses: Vec<ScriptedResponse>) -> Self {
            Self {
                is_terminal,
                responses: responses.into(),
                output: String::new(),
                actions: Vec::new(),
            }
        }
    }

    impl AttendedConfirmationTerminal for ScriptedTerminal {
        fn is_terminal(&self) -> bool {
            self.is_terminal
        }

        fn discard_pending_input(&mut self) -> io::Result<()> {
            self.actions.push("discard");
            Ok(())
        }

        fn write_prompt(&mut self, prompt: &str) -> io::Result<()> {
            self.actions.push("prompt");
            self.output.push_str(prompt);
            Ok(())
        }

        fn read_bounded_line(
            &mut self,
            _deadline: Instant,
            running: &AtomicBool,
        ) -> Result<Vec<u8>, AttendedWheelOnConfirmationError> {
            self.actions.push("read");
            if !running.load(Ordering::Acquire) {
                return Err(AttendedWheelOnConfirmationError::Interrupted);
            }
            match self.responses.pop_front() {
                Some(ScriptedResponse::Line(line))
                    if line.len() > ATTENDED_CONFIRMATION_LINE_MAX_BYTES =>
                {
                    Err(AttendedWheelOnConfirmationError::LineTooLong {
                        maximum_bytes: ATTENDED_CONFIRMATION_LINE_MAX_BYTES,
                    })
                }
                Some(ScriptedResponse::Line(line)) => Ok(line),
                Some(ScriptedResponse::EndOfInput) | None => {
                    Err(AttendedWheelOnConfirmationError::EndOfInput)
                }
                Some(ScriptedResponse::TimedOut) => {
                    Err(AttendedWheelOnConfirmationError::ResponseTimedOut {
                        maximum_wait: ATTENDED_CONFIRMATION_RESPONSE_TIMEOUT,
                    })
                }
            }
        }
    }

    fn fixed_challenges() -> FixedChallenges {
        FixedChallenges {
            values: (1_u8..=4)
                .map(|byte| [byte; ATTENDED_CONFIRMATION_CHALLENGE_BYTES])
                .collect(),
        }
    }

    fn exact_scripted_responses() -> Vec<ScriptedResponse> {
        ATTENDED_CONFIRMATION_CLAIMS
            .iter()
            .zip(1_u8..=4)
            .map(|(claim, byte)| {
                ScriptedResponse::Line(
                    format!(
                        "{} {}",
                        claim.phrase,
                        lower_hex_bytes(&[byte; ATTENDED_CONFIRMATION_CHALLENGE_BYTES])
                    )
                    .into_bytes(),
                )
            })
            .collect()
    }

    fn test_context(session_id: &str) -> AttendedConfirmationContext {
        derive_attended_confirmation_context(
            session_id,
            session_id,
            CommissioningClockEpoch::try_new([7; 16]).expect("nonzero epoch"),
            "visual-source",
            "imu-calibration",
            &[[11; 32]; ATTENDED_CONFIRMATION_BOUND_ASSET_COUNT],
        )
    }

    fn run_test_attended_confirmation_dialog<T, C>(
        terminal: &mut T,
        challenges: &mut C,
        running: &AtomicBool,
    ) -> Result<AttendedWheelOnConfirmation, AttendedWheelOnConfirmationError>
    where
        T: AttendedConfirmationTerminal,
        C: ConfirmationChallengeSource,
    {
        let clock_origin = Instant::now()
            .checked_sub(Duration::from_secs(1))
            .expect("one-second test clock origin");
        run_attended_confirmation_dialog(
            terminal,
            challenges,
            test_context("session-a"),
            clock_origin,
            running,
        )
    }

    #[test]
    fn attended_confirmation_uses_exact_nonce_bound_prompts() {
        let mut terminal = ScriptedTerminal::new(true, exact_scripted_responses());
        let mut challenges = fixed_challenges();
        let running = AtomicBool::new(true);

        let confirmation =
            run_test_attended_confirmation_dialog(&mut terminal, &mut challenges, &running)
                .expect("four exact fresh responses");
        assert_ne!(confirmation.challenge_transcript_sha256, [0_u8; 32]);
        assert_eq!(confirmation.context, test_context("session-a"));
        assert!(confirmation.issued_at_ns > 0);

        let mut expected_output = String::new();
        for (claim, byte) in ATTENDED_CONFIRMATION_CLAIMS.iter().zip(1_u8..=4) {
            let expected = format!(
                "{} {}",
                claim.phrase,
                lower_hex_bytes(&[byte; ATTENDED_CONFIRMATION_CHALLENGE_BYTES])
            );
            expected_output.push_str(&format!(
                "{}\nType exactly {:?} then press Enter: ",
                claim.explanation, expected
            ));
        }
        assert_eq!(terminal.output, expected_output);
        assert_eq!(
            terminal.actions,
            [
                "discard", "prompt", "read", "discard", "prompt", "read", "discard", "prompt",
                "read", "discard", "prompt", "read"
            ]
        );
    }

    #[test]
    fn attended_confirmation_rejects_non_tty_before_reading() {
        let mut terminal = ScriptedTerminal::new(false, exact_scripted_responses());
        let mut challenges = fixed_challenges();
        let running = AtomicBool::new(true);

        assert!(matches!(
            run_test_attended_confirmation_dialog(&mut terminal, &mut challenges, &running),
            Err(AttendedWheelOnConfirmationError::TtyRequired)
        ));
        assert!(terminal.output.is_empty());
        assert!(terminal.actions.is_empty());
    }

    #[test]
    fn attended_confirmation_rejects_wrong_exact_phrase() {
        let mut terminal =
            ScriptedTerminal::new(true, vec![ScriptedResponse::Line(b"yes".to_vec())]);
        let mut challenges = fixed_challenges();
        let running = AtomicBool::new(true);

        assert!(matches!(
            run_test_attended_confirmation_dialog(&mut terminal, &mut challenges, &running),
            Err(AttendedWheelOnConfirmationError::PhraseMismatch { .. })
        ));
    }

    #[test]
    fn attended_confirmation_rejects_long_input() {
        let mut terminal = ScriptedTerminal::new(
            true,
            vec![ScriptedResponse::Line(vec![
                b'x';
                ATTENDED_CONFIRMATION_LINE_MAX_BYTES
                    + 1
            ])],
        );
        let mut challenges = fixed_challenges();
        let running = AtomicBool::new(true);

        assert!(matches!(
            run_test_attended_confirmation_dialog(&mut terminal, &mut challenges, &running),
            Err(AttendedWheelOnConfirmationError::LineTooLong { .. })
        ));
    }

    #[test]
    fn attended_confirmation_rejects_eof() {
        let mut terminal = ScriptedTerminal::new(true, vec![ScriptedResponse::EndOfInput]);
        let mut challenges = fixed_challenges();
        let running = AtomicBool::new(true);

        assert!(matches!(
            run_test_attended_confirmation_dialog(&mut terminal, &mut challenges, &running),
            Err(AttendedWheelOnConfirmationError::EndOfInput)
        ));
    }

    #[test]
    fn attended_confirmation_rejects_invalid_utf8() {
        let mut terminal = ScriptedTerminal::new(true, vec![ScriptedResponse::Line(vec![0xff])]);
        let mut challenges = fixed_challenges();
        let running = AtomicBool::new(true);

        assert!(matches!(
            run_test_attended_confirmation_dialog(&mut terminal, &mut challenges, &running),
            Err(AttendedWheelOnConfirmationError::InvalidUtf8)
        ));
    }

    #[test]
    fn attended_confirmation_rejects_response_timeout() {
        let mut terminal = ScriptedTerminal::new(true, vec![ScriptedResponse::TimedOut]);
        let mut challenges = fixed_challenges();
        let running = AtomicBool::new(true);

        assert!(matches!(
            run_test_attended_confirmation_dialog(&mut terminal, &mut challenges, &running),
            Err(AttendedWheelOnConfirmationError::ResponseTimedOut { .. })
        ));
    }

    #[test]
    fn attended_confirmation_token_is_not_cloneable_or_copyable() {
        let _ = <AttendedWheelOnConfirmation as AmbiguousIfClone<_>>::marker as fn();
    }

    #[test]
    fn attended_confirmation_cannot_cross_session_context() {
        let completed_at = Instant::now();
        let confirmation = AttendedWheelOnConfirmation {
            context: test_context("session-a"),
            challenge_transcript_sha256: [13; 32],
            issued_at_ns: 1,
            completed_at,
        };
        let running = AtomicBool::new(true);

        assert!(matches!(
            confirmation.require_bound_fresh(test_context("session-b"), completed_at, &running),
            Err(CommissioningAttestationError::ConfirmationContextMismatch)
        ));
    }

    #[test]
    fn attended_confirmation_rejects_stale_consumption() {
        let consumed_at = Instant::now();
        let completed_at = consumed_at
            .checked_sub(ATTENDED_CONFIRMATION_MAX_CONSUMPTION_DELAY + Duration::from_nanos(1))
            .expect("short stale test interval");
        let context = test_context("session-a");
        let confirmation = AttendedWheelOnConfirmation {
            context,
            challenge_transcript_sha256: [17; 32],
            issued_at_ns: 1,
            completed_at,
        };
        let running = AtomicBool::new(true);

        assert!(matches!(
            confirmation.require_bound_fresh(context, consumed_at, &running),
            Err(CommissioningAttestationError::ConfirmationStale { .. })
        ));
    }

    fn asset(path: &str) -> Value {
        json!({
            "relative_path": path,
            "maximum_bytes": 1024,
            "sha256_hex": "1111111111111111111111111111111111111111111111111111111111111111"
        })
    }

    fn launch_value() -> Value {
        json!({
            "schema_version": NANO_BASE_COMMISSIONING_LAUNCH_V1,
            "session_id": "wheel-on-commissioning-001",
            "commissioning_policy_asset": asset("commissioning/policy.json"),
            "controller_profile_asset": asset("commissioning/controller-profile.json"),
            "controller_server_contract_asset": asset("commissioning/controller-v3.json"),
            "device_manifest_asset": asset("commissioning/inventory-v3.json"),
            "calibration_artifact_asset": asset("calibration/oak-base-v1.json"),
            "live_graph_launch_asset": asset("nano-agent-launch-v3.json")
        })
    }

    #[test]
    fn launch_requires_one_distinct_hash_bound_live_graph() {
        let valid = serde_json::to_vec(&launch_value()).expect("launch JSON");
        let parsed = NanoBaseCommissioningLaunchV1::parse_json(&valid).expect("valid launch");
        assert_eq!(parsed.session_id(), "wheel-on-commissioning-001");

        let mut missing = launch_value();
        missing
            .as_object_mut()
            .expect("object")
            .remove("live_graph_launch_asset");
        let bytes = serde_json::to_vec(&missing).expect("missing JSON");
        assert!(matches!(
            NanoBaseCommissioningLaunchV1::parse_json(&bytes),
            Err(NanoBaseCommissioningLaunchParseError::JsonDecode(_))
        ));

        let mut aliased = launch_value();
        aliased["live_graph_launch_asset"]["relative_path"] =
            aliased["calibration_artifact_asset"]["relative_path"].clone();
        let bytes = serde_json::to_vec(&aliased).expect("aliased JSON");
        assert!(matches!(
            NanoBaseCommissioningLaunchV1::parse_json(&bytes),
            Err(NanoBaseCommissioningLaunchParseError::AssetPathAliased {
                left: CommissioningAssetRole::CalibrationArtifact,
                right: CommissioningAssetRole::LiveGraphLaunch,
            })
        ));

        for session_id in [".", ".."] {
            let mut invalid = launch_value();
            invalid["session_id"] = json!(session_id);
            let bytes = serde_json::to_vec(&invalid).expect("invalid session JSON");
            assert!(matches!(
                NanoBaseCommissioningLaunchV1::parse_json(&bytes),
                Err(NanoBaseCommissioningLaunchParseError::InvalidSessionId { .. })
            ));
        }
    }

    #[test]
    fn retained_state_root_rejects_same_user_path_replacement() {
        let container = private_test_container("root-replacement");
        let root_path = container.join("state");
        create_private_directory(&root_path);
        let admitted = CommissioningStateRoot::inspect(&root_path).expect("admit state root");

        std::fs::rename(&root_path, container.join("state-original"))
            .expect("rename admitted root");
        create_private_directory(&root_path);

        assert!(matches!(
            admitted.create_session_directory("session-001"),
            Err(CommissioningAttestationError::StateRoot(
                CommissioningStateRootError::PathBindingChanged { .. }
            ))
        ));
        drop(admitted);
        std::fs::remove_dir_all(container).expect("remove root replacement fixture");
    }

    #[test]
    fn retained_session_rejects_same_user_path_replacement() {
        let container = private_test_container("session-replacement");
        let root_path = container.join("state");
        create_private_directory(&root_path);
        let admitted = CommissioningStateRoot::inspect(&root_path).expect("admit state root");
        let session = admitted
            .create_session_directory("session-001")
            .expect("create retained session");
        let session_path = root_path.join("session-001");

        std::fs::rename(&session_path, root_path.join("session-original"))
            .expect("rename admitted session");
        create_private_directory(&session_path);

        assert!(matches!(
            session.verify_binding(),
            Err(CommissioningArtifactDirectoryError::PathBindingChanged { .. })
        ));
        drop(session);
        drop(admitted);
        std::fs::remove_dir_all(container).expect("remove session replacement fixture");
    }

    fn private_test_container(label: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "kiko-commissioning-bootstrap-{label}-{}-{}",
            std::process::id(),
            NEXT_TEST_DIRECTORY.fetch_add(1, Ordering::Relaxed)
        ));
        create_private_directory(&path);
        path
    }

    fn create_private_directory(path: &Path) {
        std::fs::create_dir(path).expect("create private test directory");
        std::fs::set_permissions(path, Permissions::from_mode(0o700))
            .expect("set private directory mode");
    }
}
