//! Explicit, fail-closed authority for leaving navigation shadow mode.
//!
//! This manifest is deliberately separate from the navigation configuration.
//! It records operator claims and exact content bindings; parsing it does not
//! verify the physical robot, calibration procedure, wiring, or plant dataset.

use std::fmt;
use std::net::SocketAddr;
use std::num::{NonZeroU16, NonZeroU32, NonZeroU64};

use robot_command_client::{
    ClientConfig, ConfigError as CommandClientConfigError, StopRecoveryPolicy, TimeoutNs,
    UdpEndpoint, V2CommandLeaseMs,
};
pub use robot_protocol::v2::{ActuatorConfigFingerprint, ControllerUid};
use serde::Deserialize;
use sha2::{Digest, Sha256};

use super::mpc::{FitResidualsV1, PlantEvidenceV1, PlantModelV1};
use super::{ControlPeriodNs, SolverBudgetNs};

pub const NAVIGATION_ACTUATION_CONFIG_V1: u32 = 1;
pub const MAX_NAVIGATION_ACTUATION_CONFIG_JSON_BYTES: usize = 16 * 1_024;
const MAX_CLAIM_ID_BYTES: usize = 128;
const SHA256_BYTES: usize = 32;
const MIN_CONTROLLER_MOTION_LEASE_MS: u16 = 50;
const MAX_CONTROLLER_MOTION_LEASE_MS: u16 = 250;
const STOP_RECOVERY_ATTEMPTS: u8 = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NavigationConfigSha256([u8; SHA256_BYTES]);

impl NavigationConfigSha256 {
    pub const fn bytes(self) -> [u8; SHA256_BYTES] {
        self.0
    }

    fn digest(bytes: &[u8]) -> Self {
        Self(Sha256::digest(bytes).into())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ClaimId(String);

impl ClaimId {
    fn parse(field: &'static str, value: String) -> Result<Self, ActuationConfigParseError> {
        if value.is_empty()
            || value.len() > MAX_CLAIM_ID_BYTES
            || !value.bytes().all(|byte| {
                byte.is_ascii_alphanumeric()
                    || matches!(byte, b'-' | b'_' | b'.' | b':' | b'/' | b'@' | b'+')
            })
        {
            return Err(ActuationConfigParseError::InvalidClaimId { field });
        }
        Ok(Self(value))
    }

    fn as_str(&self) -> &str {
        &self.0
    }
}

/// Caller-asserted approval metadata. The name intentionally does not say
/// "verified": only exact agreement with the parsed navigation model is
/// established here.
#[derive(Clone, Debug, PartialEq)]
pub struct OperatorClaimedPhysicalApprovalV1 {
    approval_id: ClaimId,
    approver_id: ClaimId,
    plant_dataset_content_id: ClaimId,
    plant_identification_method_id: ClaimId,
    plant_sample_count: NonZeroU64,
    plant_fit_residuals: FitResidualsV1,
    imu_calibration_id: ClaimId,
    stereo_calibration_id: ClaimId,
    tracking_camera_to_base_calibration_id: ClaimId,
}

impl OperatorClaimedPhysicalApprovalV1 {
    pub fn approval_id(&self) -> &str {
        self.approval_id.as_str()
    }

    pub fn approver_id(&self) -> &str {
        self.approver_id.as_str()
    }

    /// Caller-asserted content identity for the physical plant dataset.
    ///
    /// Parsing the V1 config only retains this bounded claim. Production
    /// admission separately requires the canonical `sha256:<lowerhex>` value
    /// to match an exact manifest-bound, no-follow hashed plant artifact.
    pub fn plant_dataset_content_id(&self) -> &str {
        self.plant_dataset_content_id.as_str()
    }

    pub fn imu_calibration_id(&self) -> &str {
        self.imu_calibration_id.as_str()
    }

    pub fn stereo_calibration_id(&self) -> &str {
        self.stereo_calibration_id.as_str()
    }

    pub fn tracking_camera_to_base_calibration_id(&self) -> &str {
        self.tracking_camera_to_base_calibration_id.as_str()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NavigationActuationConfigV1 {
    robot_id: ClaimId,
    command_endpoint: UdpEndpoint,
    navigation_config_sha256: NavigationConfigSha256,
    controller_uid: ControllerUid,
    firmware_abi: NonZeroU16,
    firmware_build_id: NonZeroU32,
    actuator_config_fingerprint: ActuatorConfigFingerprint,
    plant_model_id: ClaimId,
    plant_model_version: NonZeroU32,
    approval: OperatorClaimedPhysicalApprovalV1,
    apply_ack_budget: TimeoutNs,
    stop_ack_budget: TimeoutNs,
    stop_recovery: StopRecoveryPolicy,
    scheduling_guard_ns: NonZeroU64,
    controller_motion_lease: V2CommandLeaseMs,
    controller_deadline_tolerance_ns: NonZeroU64,
    maximum_uncommanded_motion_ns: NonZeroU64,
}

impl NavigationActuationConfigV1 {
    pub fn parse_and_authorize(
        bytes: &[u8],
        requested_robot_id: &str,
        navigation_config_bytes: &[u8],
        plant_model: PlantModelV1,
        solver_budget: SolverBudgetNs,
        control_period: ControlPeriodNs,
    ) -> Result<Self, ActuationConfigParseError> {
        if bytes.len() > MAX_NAVIGATION_ACTUATION_CONFIG_JSON_BYTES {
            return Err(ActuationConfigParseError::InputTooLarge {
                actual_bytes: bytes.len(),
                maximum_bytes: MAX_NAVIGATION_ACTUATION_CONFIG_JSON_BYTES,
            });
        }
        let dto: NavigationActuationConfigV1Dto =
            serde_json::from_slice(bytes).map_err(ActuationConfigParseError::Json)?;
        if dto.schema_version != NAVIGATION_ACTUATION_CONFIG_V1 {
            return Err(ActuationConfigParseError::UnsupportedSchemaVersion(
                dto.schema_version,
            ));
        }

        let robot_id = ClaimId::parse("robot_id", dto.robot_id)?;
        if robot_id.as_str() != requested_robot_id {
            return Err(ActuationConfigParseError::ArmRobotIdMismatch);
        }
        let command_endpoint_address = dto
            .command_endpoint
            .parse::<SocketAddr>()
            .map_err(ActuationConfigParseError::InvalidCommandEndpoint)?;
        if !command_endpoint_address.ip().is_loopback() {
            return Err(ActuationConfigParseError::CommandEndpointIsNotLoopback(
                command_endpoint_address,
            ));
        }
        let command_endpoint = UdpEndpoint::try_new(command_endpoint_address)
            .map_err(ActuationConfigParseError::CommandClientConfig)?;

        let navigation_config_sha256 = NavigationConfigSha256(parse_hex_exact(
            "navigation_config_sha256_hex",
            &dto.navigation_config_sha256_hex,
        )?);
        if navigation_config_sha256 != NavigationConfigSha256::digest(navigation_config_bytes) {
            return Err(ActuationConfigParseError::NavigationConfigHashMismatch);
        }
        let controller_uid = ControllerUid::try_new(parse_hex_exact(
            "controller_uid_hex",
            &dto.controller_uid_hex,
        )?)
        .map_err(|_| ActuationConfigParseError::ZeroControllerUid)?;
        let actuator_config_fingerprint = ActuatorConfigFingerprint::try_new(parse_hex_exact(
            "actuator_config_fingerprint_hex",
            &dto.actuator_config_fingerprint_hex,
        )?)
        .map_err(|_| ActuationConfigParseError::ZeroActuatorConfigFingerprint)?;
        let firmware_abi =
            NonZeroU16::new(dto.firmware_abi).ok_or(ActuationConfigParseError::ZeroFirmwareAbi)?;
        let firmware_build_id = NonZeroU32::new(dto.firmware_build_id)
            .ok_or(ActuationConfigParseError::ZeroFirmwareBuildId)?;
        let plant_model_id = ClaimId::parse("plant_model_id", dto.plant_model_id)?;
        let plant_model_version = NonZeroU32::new(dto.plant_model_version)
            .ok_or(ActuationConfigParseError::ZeroPlantModelVersion)?;
        if plant_model.model_id().as_str() != plant_model_id.as_str()
            || plant_model.model_version() != plant_model_version
        {
            return Err(ActuationConfigParseError::PlantModelIdentityMismatch);
        }

        let approval = parse_approval(dto.operator_claimed_physical_approval)?;
        match plant_model.evidence() {
            PlantEvidenceV1::SyntheticFixture { .. } => {
                return Err(ActuationConfigParseError::SyntheticPlantCannotActuate);
            }
            PlantEvidenceV1::ClaimedPhysicalIdentification {
                dataset_content_id,
                identification_method_id,
                sample_count,
                residuals,
            } => {
                if dataset_content_id.as_str() != approval.plant_dataset_content_id.as_str()
                    || identification_method_id.as_str()
                        != approval.plant_identification_method_id.as_str()
                    || sample_count != approval.plant_sample_count
                    || !residuals_equal(residuals, approval.plant_fit_residuals)
                {
                    return Err(ActuationConfigParseError::PlantEvidenceMismatch);
                }
            }
        }

        let apply_ack_budget = TimeoutNs::try_new(dto.apply_ack_budget_ns)
            .map_err(ActuationConfigParseError::CommandClientConfig)?;
        let stop_ack_budget = TimeoutNs::try_new(dto.stop_ack_budget_ns)
            .map_err(ActuationConfigParseError::CommandClientConfig)?;
        let stop_recovery = StopRecoveryPolicy::try_new(STOP_RECOVERY_ATTEMPTS, stop_ack_budget)
            .map_err(ActuationConfigParseError::CommandClientConfig)?;
        let scheduling_guard_ns = NonZeroU64::new(dto.scheduling_guard_ns).ok_or(
            ActuationConfigParseError::ZeroDuration("scheduling_guard_ns"),
        )?;
        let controller_deadline_tolerance_ns =
            NonZeroU64::new(dto.controller_deadline_tolerance_ns).ok_or(
                ActuationConfigParseError::ZeroDuration("controller_deadline_tolerance_ns"),
            )?;
        let maximum_uncommanded_motion_ns = NonZeroU64::new(dto.maximum_uncommanded_motion_ns)
            .ok_or(ActuationConfigParseError::ZeroDuration(
                "maximum_uncommanded_motion_ns",
            ))?;
        let controller_motion_lease = V2CommandLeaseMs::try_new(dto.controller_motion_lease_ms)
            .map_err(
                |_| ActuationConfigParseError::ControllerMotionLeaseOutOfRange {
                    value_ms: dto.controller_motion_lease_ms,
                    minimum_ms: MIN_CONTROLLER_MOTION_LEASE_MS,
                    maximum_ms: MAX_CONTROLLER_MOTION_LEASE_MS,
                },
            )?;

        let occupied_control_period_ns = solver_budget
            .get()
            .checked_add(apply_ack_budget.get())
            .and_then(|value| value.checked_add(scheduling_guard_ns.get()))
            .ok_or(ActuationConfigParseError::DurationArithmeticOverflow)?;
        if occupied_control_period_ns >= control_period.get() {
            return Err(
                ActuationConfigParseError::ControlPeriodHasNoActuationMargin {
                    solver_budget_ns: solver_budget.get(),
                    apply_ack_budget_ns: apply_ack_budget.get(),
                    scheduling_guard_ns: scheduling_guard_ns.get(),
                    control_period_ns: control_period.get(),
                },
            );
        }
        let lease_ns = u64::from(controller_motion_lease.get())
            .checked_mul(1_000_000)
            .ok_or(ActuationConfigParseError::DurationArithmeticOverflow)?;
        let required_bridge_ns = control_period
            .get()
            .checked_add(solver_budget.get())
            .and_then(|value| value.checked_add(apply_ack_budget.get()))
            .and_then(|value| value.checked_add(scheduling_guard_ns.get()))
            .ok_or(ActuationConfigParseError::DurationArithmeticOverflow)?;
        if lease_ns <= required_bridge_ns {
            return Err(
                ActuationConfigParseError::ControllerLeaseCannotBridgeNextApplication {
                    lease_ns,
                    required_exclusive_lower_bound_ns: required_bridge_ns,
                },
            );
        }
        let derived_uncommanded_motion_bound_ns = apply_ack_budget
            .get()
            .checked_add(lease_ns)
            .and_then(|value| value.checked_add(controller_deadline_tolerance_ns.get()))
            .ok_or(ActuationConfigParseError::DurationArithmeticOverflow)?;
        if derived_uncommanded_motion_bound_ns > maximum_uncommanded_motion_ns.get() {
            return Err(
                ActuationConfigParseError::UncommandedMotionBoundExceedsOperatorLimit {
                    derived_ns: derived_uncommanded_motion_bound_ns,
                    configured_limit_ns: maximum_uncommanded_motion_ns.get(),
                },
            );
        }

        Ok(Self {
            robot_id,
            command_endpoint,
            navigation_config_sha256,
            controller_uid,
            firmware_abi,
            firmware_build_id,
            actuator_config_fingerprint,
            plant_model_id,
            plant_model_version,
            approval,
            apply_ack_budget,
            stop_ack_budget,
            stop_recovery,
            scheduling_guard_ns,
            controller_motion_lease,
            controller_deadline_tolerance_ns,
            maximum_uncommanded_motion_ns,
        })
    }

    pub fn robot_id(&self) -> &str {
        self.robot_id.as_str()
    }

    pub const fn command_endpoint(&self) -> UdpEndpoint {
        self.command_endpoint
    }

    pub const fn navigation_config_sha256(&self) -> NavigationConfigSha256 {
        self.navigation_config_sha256
    }

    pub const fn controller_uid(&self) -> ControllerUid {
        self.controller_uid
    }

    pub const fn firmware_abi(&self) -> NonZeroU16 {
        self.firmware_abi
    }

    pub const fn firmware_build_id(&self) -> NonZeroU32 {
        self.firmware_build_id
    }

    pub const fn actuator_config_fingerprint(&self) -> ActuatorConfigFingerprint {
        self.actuator_config_fingerprint
    }

    pub fn plant_model_id(&self) -> &str {
        self.plant_model_id.as_str()
    }

    pub const fn plant_model_version(&self) -> NonZeroU32 {
        self.plant_model_version
    }

    pub const fn apply_ack_budget(&self) -> TimeoutNs {
        self.apply_ack_budget
    }

    pub const fn stop_ack_budget(&self) -> TimeoutNs {
        self.stop_ack_budget
    }

    pub const fn scheduling_guard_ns(&self) -> NonZeroU64 {
        self.scheduling_guard_ns
    }

    pub const fn controller_motion_lease(&self) -> V2CommandLeaseMs {
        self.controller_motion_lease
    }

    pub const fn controller_deadline_tolerance_ns(&self) -> NonZeroU64 {
        self.controller_deadline_tolerance_ns
    }

    pub const fn maximum_uncommanded_motion_ns(&self) -> NonZeroU64 {
        self.maximum_uncommanded_motion_ns
    }

    pub const fn approval(&self) -> &OperatorClaimedPhysicalApprovalV1 {
        &self.approval
    }

    /// Construct the concrete client without reparsing or stringifying any
    /// value admitted by this authority boundary. Status and acquisition use
    /// the applied-command ACK budget. Stop recovery makes exactly three
    /// attempts, each bounded by the explicit stop ACK budget.
    pub const fn client_config(&self) -> ClientConfig {
        ClientConfig::new(
            self.command_endpoint,
            self.controller_uid,
            self.firmware_abi,
            self.firmware_build_id,
            self.actuator_config_fingerprint,
            self.apply_ack_budget,
            self.apply_ack_budget,
            self.apply_ack_budget,
            self.stop_recovery,
            self.controller_motion_lease,
        )
    }
}

#[derive(Debug)]
pub enum ActuationConfigParseError {
    InputTooLarge {
        actual_bytes: usize,
        maximum_bytes: usize,
    },
    Json(serde_json::Error),
    UnsupportedSchemaVersion(u32),
    InvalidClaimId {
        field: &'static str,
    },
    ArmRobotIdMismatch,
    InvalidCommandEndpoint(std::net::AddrParseError),
    CommandEndpointIsNotLoopback(SocketAddr),
    CommandClientConfig(CommandClientConfigError),
    InvalidHexLength {
        field: &'static str,
        expected_digits: usize,
        actual_digits: usize,
    },
    InvalidHexDigit {
        field: &'static str,
        digit_index: usize,
    },
    NavigationConfigHashMismatch,
    ZeroControllerUid,
    ZeroActuatorConfigFingerprint,
    ZeroFirmwareAbi,
    ZeroFirmwareBuildId,
    ZeroPlantModelVersion,
    PlantModelIdentityMismatch,
    SyntheticPlantCannotActuate,
    ZeroPhysicalSampleCount,
    NonFiniteOrNegativeResidual {
        field: &'static str,
    },
    PlantEvidenceMismatch,
    ZeroDuration(&'static str),
    ControllerMotionLeaseOutOfRange {
        value_ms: u16,
        minimum_ms: u16,
        maximum_ms: u16,
    },
    DurationArithmeticOverflow,
    ControlPeriodHasNoActuationMargin {
        solver_budget_ns: u64,
        apply_ack_budget_ns: u64,
        scheduling_guard_ns: u64,
        control_period_ns: u64,
    },
    ControllerLeaseCannotBridgeNextApplication {
        lease_ns: u64,
        required_exclusive_lower_bound_ns: u64,
    },
    UncommandedMotionBoundExceedsOperatorLimit {
        derived_ns: u64,
        configured_limit_ns: u64,
    },
}

impl fmt::Display for ActuationConfigParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputTooLarge {
                actual_bytes,
                maximum_bytes,
            } => write!(
                formatter,
                "navigation actuation config is {actual_bytes} bytes; maximum is {maximum_bytes}"
            ),
            Self::Json(source) => write!(formatter, "invalid actuation config JSON: {source}"),
            Self::UnsupportedSchemaVersion(version) => {
                write!(
                    formatter,
                    "unsupported navigation actuation schema {version}"
                )
            }
            Self::InvalidClaimId { field } => {
                write!(formatter, "invalid bounded claim ID in {field}")
            }
            Self::ArmRobotIdMismatch => formatter.write_str(
                "--navigation-arm-robot does not exactly match the actuation manifest robot_id",
            ),
            Self::InvalidCommandEndpoint(source) => {
                write!(
                    formatter,
                    "command_endpoint is not an IP socket address: {source}"
                )
            }
            Self::CommandEndpointIsNotLoopback(endpoint) => write!(
                formatter,
                "unauthenticated actuation endpoint {endpoint} is not loopback; use an authenticated tunnel"
            ),
            Self::CommandClientConfig(source) => {
                write!(
                    formatter,
                    "invalid typed command-client configuration: {source}"
                )
            }
            Self::InvalidHexLength {
                field,
                expected_digits,
                actual_digits,
            } => write!(
                formatter,
                "{field} must contain exactly {expected_digits} hexadecimal digits, got {actual_digits}"
            ),
            Self::InvalidHexDigit { field, digit_index } => {
                write!(
                    formatter,
                    "{field} has a non-hexadecimal digit at index {digit_index}"
                )
            }
            Self::NavigationConfigHashMismatch => formatter.write_str(
                "navigation configuration bytes do not match the actuation manifest SHA-256",
            ),
            Self::ZeroControllerUid => {
                formatter.write_str("the all-zero controller UID is reserved")
            }
            Self::ZeroActuatorConfigFingerprint => {
                formatter.write_str("the all-zero actuator configuration fingerprint is reserved")
            }
            Self::ZeroFirmwareAbi => formatter.write_str("firmware_abi must be nonzero"),
            Self::ZeroFirmwareBuildId => formatter.write_str("firmware_build_id must be nonzero"),
            Self::ZeroPlantModelVersion => {
                formatter.write_str("plant_model_version must be nonzero")
            }
            Self::PlantModelIdentityMismatch => formatter.write_str(
                "actuation manifest plant identity does not match the parsed navigation model",
            ),
            Self::SyntheticPlantCannotActuate => formatter.write_str(
                "a synthetic plant fixture can run shadow MPC but cannot authorize actuation",
            ),
            Self::ZeroPhysicalSampleCount => {
                formatter.write_str("physical plant approval sample_count must be nonzero")
            }
            Self::NonFiniteOrNegativeResidual { field } => write!(
                formatter,
                "physical plant approval residual {field} must be finite and nonnegative"
            ),
            Self::PlantEvidenceMismatch => formatter.write_str(
                "operator approval evidence does not exactly match parsed plant evidence",
            ),
            Self::ZeroDuration(field) => write!(formatter, "{field} must be nonzero"),
            Self::ControllerMotionLeaseOutOfRange {
                value_ms,
                minimum_ms,
                maximum_ms,
            } => write!(
                formatter,
                "controller_motion_lease_ms {value_ms} is outside {minimum_ms}..={maximum_ms} ms"
            ),
            Self::DurationArithmeticOverflow => {
                formatter.write_str("actuation duration arithmetic overflowed")
            }
            Self::ControlPeriodHasNoActuationMargin {
                solver_budget_ns,
                apply_ack_budget_ns,
                scheduling_guard_ns,
                control_period_ns,
            } => write!(
                formatter,
                "solver ({solver_budget_ns} ns) + apply ACK ({apply_ack_budget_ns} ns) + scheduling guard ({scheduling_guard_ns} ns) must be less than the control period ({control_period_ns} ns)"
            ),
            Self::ControllerLeaseCannotBridgeNextApplication {
                lease_ns,
                required_exclusive_lower_bound_ns,
            } => write!(
                formatter,
                "controller lease {lease_ns} ns must be greater than the control-period, solver-budget, apply-ACK-budget, and scheduling-guard bridge {required_exclusive_lower_bound_ns} ns"
            ),
            Self::UncommandedMotionBoundExceedsOperatorLimit {
                derived_ns,
                configured_limit_ns,
            } => write!(
                formatter,
                "derived uncommanded-motion bound {derived_ns} ns exceeds operator limit {configured_limit_ns} ns"
            ),
        }
    }
}

impl std::error::Error for ActuationConfigParseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Json(source) => Some(source),
            Self::InvalidCommandEndpoint(source) => Some(source),
            Self::CommandClientConfig(source) => Some(source),
            _ => None,
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NavigationActuationConfigV1Dto {
    schema_version: u32,
    robot_id: String,
    command_endpoint: String,
    navigation_config_sha256_hex: String,
    controller_uid_hex: String,
    firmware_abi: u16,
    firmware_build_id: u32,
    actuator_config_fingerprint_hex: String,
    plant_model_id: String,
    plant_model_version: u32,
    operator_claimed_physical_approval: OperatorClaimedPhysicalApprovalV1Dto,
    apply_ack_budget_ns: u64,
    stop_ack_budget_ns: u64,
    scheduling_guard_ns: u64,
    controller_motion_lease_ms: u16,
    controller_deadline_tolerance_ns: u64,
    maximum_uncommanded_motion_ns: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OperatorClaimedPhysicalApprovalV1Dto {
    approval_id: String,
    approver_id: String,
    plant_dataset_content_id: String,
    plant_identification_method_id: String,
    plant_sample_count: u64,
    plant_fit_residuals: PlantFitResidualsDto,
    imu_calibration_id: String,
    stereo_calibration_id: String,
    tracking_camera_to_base_calibration_id: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PlantFitResidualsDto {
    left_velocity_rmse_mps: f64,
    right_velocity_rmse_mps: f64,
    yaw_rate_rmse_rad_s: f64,
    max_abs_velocity_error_mps: f64,
}

fn parse_approval(
    dto: OperatorClaimedPhysicalApprovalV1Dto,
) -> Result<OperatorClaimedPhysicalApprovalV1, ActuationConfigParseError> {
    let residuals = FitResidualsV1 {
        left_velocity_rmse_mps: dto.plant_fit_residuals.left_velocity_rmse_mps,
        right_velocity_rmse_mps: dto.plant_fit_residuals.right_velocity_rmse_mps,
        yaw_rate_rmse_rad_s: dto.plant_fit_residuals.yaw_rate_rmse_rad_s,
        max_abs_velocity_error_mps: dto.plant_fit_residuals.max_abs_velocity_error_mps,
    };
    for (field, value) in [
        ("left_velocity_rmse_mps", residuals.left_velocity_rmse_mps),
        ("right_velocity_rmse_mps", residuals.right_velocity_rmse_mps),
        ("yaw_rate_rmse_rad_s", residuals.yaw_rate_rmse_rad_s),
        (
            "max_abs_velocity_error_mps",
            residuals.max_abs_velocity_error_mps,
        ),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(ActuationConfigParseError::NonFiniteOrNegativeResidual { field });
        }
    }
    Ok(OperatorClaimedPhysicalApprovalV1 {
        approval_id: ClaimId::parse("approval_id", dto.approval_id)?,
        approver_id: ClaimId::parse("approver_id", dto.approver_id)?,
        plant_dataset_content_id: ClaimId::parse(
            "plant_dataset_content_id",
            dto.plant_dataset_content_id,
        )?,
        plant_identification_method_id: ClaimId::parse(
            "plant_identification_method_id",
            dto.plant_identification_method_id,
        )?,
        plant_sample_count: NonZeroU64::new(dto.plant_sample_count)
            .ok_or(ActuationConfigParseError::ZeroPhysicalSampleCount)?,
        plant_fit_residuals: residuals,
        imu_calibration_id: ClaimId::parse("imu_calibration_id", dto.imu_calibration_id)?,
        stereo_calibration_id: ClaimId::parse("stereo_calibration_id", dto.stereo_calibration_id)?,
        tracking_camera_to_base_calibration_id: ClaimId::parse(
            "tracking_camera_to_base_calibration_id",
            dto.tracking_camera_to_base_calibration_id,
        )?,
    })
}

fn residuals_equal(left: FitResidualsV1, right: FitResidualsV1) -> bool {
    left.left_velocity_rmse_mps.to_bits() == right.left_velocity_rmse_mps.to_bits()
        && left.right_velocity_rmse_mps.to_bits() == right.right_velocity_rmse_mps.to_bits()
        && left.yaw_rate_rmse_rad_s.to_bits() == right.yaw_rate_rmse_rad_s.to_bits()
        && left.max_abs_velocity_error_mps.to_bits() == right.max_abs_velocity_error_mps.to_bits()
}

fn parse_hex_exact<const N: usize>(
    field: &'static str,
    value: &str,
) -> Result<[u8; N], ActuationConfigParseError> {
    let expected_digits = N * 2;
    if value.len() != expected_digits {
        return Err(ActuationConfigParseError::InvalidHexLength {
            field,
            expected_digits,
            actual_digits: value.len(),
        });
    }
    let mut decoded = [0_u8; N];
    let bytes = value.as_bytes();
    let mut index = 0;
    while index < N {
        let high_index = index * 2;
        let low_index = high_index + 1;
        let high =
            hex_nibble(bytes[high_index]).ok_or(ActuationConfigParseError::InvalidHexDigit {
                field,
                digit_index: high_index,
            })?;
        let low =
            hex_nibble(bytes[low_index]).ok_or(ActuationConfigParseError::InvalidHexDigit {
                field,
                digit_index: low_index,
            })?;
        decoded[index] = (high << 4) | low;
        index += 1;
    }
    Ok(decoded)
}

const fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::navigation::mpc::{
        FitResidualsV1Dto, PLANT_MODEL_V1, PlantEvidenceV1Dto, PlantModelV1Dto,
        PlantValidityEnvelopeV1Dto, WheelPlantV1Dto,
    };
    use serde_json::{Value, json};

    const NAVIGATION_BYTES: &[u8] = br#"{"schema":"navigation-fixture"}"#;

    fn physical_model() -> PlantModelV1 {
        PlantModelV1::parse(PlantModelV1Dto {
            schema_version: PLANT_MODEL_V1,
            model_id: "kiko-physical-v1".to_string(),
            model_version: 1,
            sample_period_s: 0.1,
            wheelbase_m: 0.3,
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
                max_abs_yaw_rate_rad_s: 3.0,
                max_abs_lateral_velocity_mps: 0.1,
            },
            evidence: PlantEvidenceV1Dto::ClaimedPhysicalIdentification {
                dataset_content_id: "sha256:plant".to_string(),
                identification_method_id: "method-v1".to_string(),
                sample_count: 100,
                residuals: FitResidualsV1Dto {
                    left_velocity_rmse_mps: 0.01,
                    right_velocity_rmse_mps: 0.02,
                    yaw_rate_rmse_rad_s: 0.03,
                    max_abs_velocity_error_mps: 0.04,
                },
            },
        })
        .expect("physical model fixture")
    }

    fn synthetic_model() -> PlantModelV1 {
        let mut dto = model_dto_from_physical();
        dto.evidence = PlantEvidenceV1Dto::SyntheticFixture {
            fixture_id: "synthetic".to_string(),
            generator_id: "test".to_string(),
        };
        PlantModelV1::parse(dto).expect("synthetic model fixture")
    }

    fn model_dto_from_physical() -> PlantModelV1Dto {
        PlantModelV1Dto {
            schema_version: PLANT_MODEL_V1,
            model_id: "kiko-physical-v1".to_string(),
            model_version: 1,
            sample_period_s: 0.1,
            wheelbase_m: 0.3,
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
                max_abs_yaw_rate_rad_s: 3.0,
                max_abs_lateral_velocity_mps: 0.1,
            },
            evidence: PlantEvidenceV1Dto::ClaimedPhysicalIdentification {
                dataset_content_id: "sha256:plant".to_string(),
                identification_method_id: "method-v1".to_string(),
                sample_count: 100,
                residuals: FitResidualsV1Dto {
                    left_velocity_rmse_mps: 0.01,
                    right_velocity_rmse_mps: 0.02,
                    yaw_rate_rmse_rad_s: 0.03,
                    max_abs_velocity_error_mps: 0.04,
                },
            },
        }
    }

    fn valid_json() -> Value {
        let hash = NavigationConfigSha256::digest(NAVIGATION_BYTES).bytes();
        let hash_hex: String = hash.iter().map(|byte| format!("{byte:02x}")).collect();
        json!({
            "schema_version": 1,
            "robot_id": "kiko-01",
            "command_endpoint": "127.0.0.1:8080",
            "navigation_config_sha256_hex": hash_hex,
            "controller_uid_hex": "00112233445566778899aabb",
            "firmware_abi": 2,
            "firmware_build_id": 42,
            "actuator_config_fingerprint_hex": "11223344556677889900aabbccddeeff",
            "plant_model_id": "kiko-physical-v1",
            "plant_model_version": 1,
            "operator_claimed_physical_approval": {
                "approval_id": "approval-2026-07-18",
                "approver_id": "operator@example.com",
                "plant_dataset_content_id": "sha256:plant",
                "plant_identification_method_id": "method-v1",
                "plant_sample_count": 100,
                "plant_fit_residuals": {
                    "left_velocity_rmse_mps": 0.01,
                    "right_velocity_rmse_mps": 0.02,
                    "yaw_rate_rmse_rad_s": 0.03,
                    "max_abs_velocity_error_mps": 0.04
                },
                "imu_calibration_id": "imu-cal-v1",
                "stereo_calibration_id": "stereo-cal-v1",
                "tracking_camera_to_base_calibration_id": "extrinsic-v1"
            },
            "apply_ack_budget_ns": 20_000_000,
            "stop_ack_budget_ns": 30_000_000,
            "scheduling_guard_ns": 5_000_000,
            "controller_motion_lease_ms": 200,
            "controller_deadline_tolerance_ns": 2_000_000,
            "maximum_uncommanded_motion_ns": 222_000_000
        })
    }

    fn parse(
        value: &Value,
        model: PlantModelV1,
    ) -> Result<NavigationActuationConfigV1, ActuationConfigParseError> {
        let bytes = serde_json::to_vec(value).expect("serialize fixture");
        NavigationActuationConfigV1::parse_and_authorize(
            &bytes,
            "kiko-01",
            NAVIGATION_BYTES,
            model,
            SolverBudgetNs::try_new(50_000_000).expect("solver budget"),
            ControlPeriodNs::from_nonzero(NonZeroU64::new(100_000_000).expect("control period")),
        )
    }

    #[test]
    fn exact_physical_claim_and_content_binding_authorize() {
        let parsed = parse(&valid_json(), physical_model()).expect("valid physical authority");
        assert_eq!(parsed.robot_id(), "kiko-01");
        assert!(parsed.command_endpoint().socket_addr().ip().is_loopback());
        assert_eq!(parsed.controller_motion_lease().get(), 200);
        assert_eq!(parsed.approval().approval_id(), "approval-2026-07-18");
    }

    #[test]
    fn synthetic_model_can_never_authorize_motion() {
        assert!(matches!(
            parse(&valid_json(), synthetic_model()),
            Err(ActuationConfigParseError::SyntheticPlantCannotActuate)
        ));
    }

    #[test]
    fn unknown_fields_hash_mismatch_and_nonloopback_reject() {
        let mut unknown = valid_json();
        unknown["surprise"] = json!(true);
        assert!(matches!(
            parse(&unknown, physical_model()),
            Err(ActuationConfigParseError::Json(_))
        ));

        let mut hash = valid_json();
        hash["navigation_config_sha256_hex"] = json!("00".repeat(32));
        assert!(matches!(
            parse(&hash, physical_model()),
            Err(ActuationConfigParseError::NavigationConfigHashMismatch)
        ));

        let mut remote = valid_json();
        remote["command_endpoint"] = json!("192.168.50.2:8080");
        assert!(matches!(
            parse(&remote, physical_model()),
            Err(ActuationConfigParseError::CommandEndpointIsNotLoopback(_))
        ));
    }

    #[test]
    fn evidence_and_timing_must_match_exactly_with_real_margin() {
        let mut evidence = valid_json();
        evidence["operator_claimed_physical_approval"]["plant_sample_count"] = json!(99);
        assert!(matches!(
            parse(&evidence, physical_model()),
            Err(ActuationConfigParseError::PlantEvidenceMismatch)
        ));

        let mut no_margin = valid_json();
        no_margin["apply_ack_budget_ns"] = json!(45_000_000_u64);
        assert!(matches!(
            parse(&no_margin, physical_model()),
            Err(ActuationConfigParseError::ControlPeriodHasNoActuationMargin { .. })
        ));

        let mut false_bound = valid_json();
        false_bound["maximum_uncommanded_motion_ns"] = json!(141_999_999_u64);
        assert!(matches!(
            parse(&false_bound, physical_model()),
            Err(ActuationConfigParseError::UncommandedMotionBoundExceedsOperatorLimit { .. })
        ));

        let mut cannot_bridge = valid_json();
        // Equality at the complete period + solve + apply-ACK + scheduling
        // horizon is expired. Omitting the ACK budget would admit this value.
        cannot_bridge["controller_motion_lease_ms"] = json!(175_u64);
        assert!(matches!(
            parse(&cannot_bridge, physical_model()),
            Err(ActuationConfigParseError::ControllerLeaseCannotBridgeNextApplication { .. })
        ));
    }

    #[test]
    fn exact_hex_parser_accepts_case_and_rejects_every_bad_position() {
        assert_eq!(
            parse_hex_exact::<2>("test", "aB01").expect("valid mixed-case hex"),
            [0xab, 0x01]
        );
        for index in 0..4 {
            let mut bytes = *b"0000";
            bytes[index] = b'x';
            let text = std::str::from_utf8(&bytes).expect("ASCII fixture");
            assert!(matches!(
                parse_hex_exact::<2>("test", text),
                Err(ActuationConfigParseError::InvalidHexDigit { digit_index, .. }) if digit_index == index
            ));
        }
    }
}
