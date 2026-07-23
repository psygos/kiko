use std::fmt;
use std::time::Duration;

use kiko_head_protocol::{
    ExactHeadTargetPose, ExactHeadTargetPoseError, FrameBuildError, GoalSpeedTicksPerSecond,
    HeadJoint, HeadPose, HeadTorqueLimits, PositionAgreementError, PositionAgreementTicks,
    PositionTicks, TorqueLimitPermille,
};

const MAX_DEVICE_PATH_BYTES: usize = 512;
const MAX_DEVICE_ID_BYTES: usize = 255;
const MAX_OPERATION_TIMEOUT_MS: u64 = 5_000;
const MAX_ARMING_FRESHNESS_MS: u64 = 5_000;
const MAX_WRITE_ATTEMPTS: u8 = 8;
const MAX_NOISE_BUDGET_BYTES: u16 = 1_024;
const MAX_HOLD_DURATION_MS: u64 = 900_000;
pub const HEAD_RETURN_POSITION_STEP_TICKS: u16 = 50;
pub const HEAD_RETURN_CONTROL_PERIOD: Duration = Duration::from_millis(100);
pub const HEAD_RETURN_NO_PROGRESS_TIMEOUT: Duration = Duration::from_secs(2);
pub const HEAD_RETURN_MOTION_TIMEOUT: Duration = Duration::from_secs(20);
/// A four-servo telemetry set older than one control period is not a coherent
/// motion/recovery boundary. The serial requests themselves may have a longer
/// diagnostic timeout, but such a delayed response cannot authorize motion.
pub const HEAD_RETURN_TELEMETRY_SET_MAX_AGE: Duration = HEAD_RETURN_CONTROL_PERIOD;
pub const MAX_HEAD_RETURN_TRAVEL_TICKS: u16 = 512;
/// Maximum admitted width of one raw-encoder startup pose window.
///
/// This is a structural anti-bypass bound, not a physical joint envelope. The
/// assembly-specific minimum and maximum still require independent review.
pub const MAX_CONFIGURED_POSE_WINDOW_SPAN_TICKS: u16 = 256;

/// Weak configuration accepted at the process boundary.
///
/// Parse this exactly once with [`HeadRuntimeConfig::parse`]. Runtime code only
/// accepts the resulting domain type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeadRuntimeConfigInput {
    pub device_path: String,
    pub response_timeout_ms: u64,
    pub write_timeout_ms: u64,
    pub arming_freshness_ms: u64,
    pub write_attempts: u8,
    pub noise_budget_bytes: u16,
    pub redundant_read_tolerance_ticks: u16,
    pub readback_tolerance_ticks: u16,
    pub goal_speed_ticks_per_second: u16,
    /// Bow, curl, yaw, and roll, in that exact order.
    pub torque_limit_permille: [u16; 4],
}

/// Weak configuration for the read-only commissioning probe.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeadProbeConfigInput {
    pub device_path: String,
    pub response_timeout_ms: u64,
    pub request_timeout_ms: u64,
    pub noise_budget_bytes: u16,
}

/// Supported stable device-name schemes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeviceIdentityKind {
    LinuxById,
    MacOsCallout,
}

/// Exact configured device identity. No discovery or fallback path exists.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DeviceIdentity {
    path: Box<str>,
    stable_name: Box<str>,
    kind: DeviceIdentityKind,
}

impl DeviceIdentity {
    fn parse(path: String) -> Result<Self, ConfigParseError> {
        if path.is_empty() {
            return Err(ConfigParseError::EmptyDevicePath);
        }
        if path.len() > MAX_DEVICE_PATH_BYTES {
            return Err(ConfigParseError::DevicePathTooLong {
                actual_bytes: path.len(),
                maximum_bytes: MAX_DEVICE_PATH_BYTES,
            });
        }
        if path.as_bytes().contains(&0) {
            return Err(ConfigParseError::DevicePathContainsNul);
        }

        let (kind, stable_name) = if let Some(name) = path.strip_prefix("/dev/serial/by-id/") {
            (DeviceIdentityKind::LinuxById, name)
        } else if let Some(name) = path.strip_prefix("/dev/cu.") {
            (DeviceIdentityKind::MacOsCallout, name)
        } else {
            return Err(ConfigParseError::UnsupportedDevicePath { path });
        };

        if stable_name.is_empty()
            || stable_name == "."
            || stable_name == ".."
            || stable_name.contains('/')
        {
            return Err(ConfigParseError::InvalidDeviceIdentity {
                identity: stable_name.to_owned(),
            });
        }
        if stable_name.len() > MAX_DEVICE_ID_BYTES {
            return Err(ConfigParseError::DeviceIdentityTooLong {
                actual_bytes: stable_name.len(),
                maximum_bytes: MAX_DEVICE_ID_BYTES,
            });
        }

        Ok(Self {
            stable_name: stable_name.into(),
            path: path.into(),
            kind,
        })
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    pub fn stable_name(&self) -> &str {
        &self.stable_name
    }

    pub const fn kind(&self) -> DeviceIdentityKind {
        self.kind
    }
}

/// Strictly positive per-operation timeout, bounded to five seconds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OperationTimeout(Duration);

impl OperationTimeout {
    fn parse(field: &'static str, milliseconds: u64) -> Result<Self, ConfigParseError> {
        if !(1..=MAX_OPERATION_TIMEOUT_MS).contains(&milliseconds) {
            return Err(ConfigParseError::OperationTimeoutOutOfRange {
                field,
                milliseconds,
                minimum_ms: 1,
                maximum_ms: MAX_OPERATION_TIMEOUT_MS,
            });
        }
        Ok(Self(Duration::from_millis(milliseconds)))
    }

    pub const fn get(self) -> Duration {
        self.0
    }

    pub(crate) fn capped_by(self, maximum: Duration) -> Self {
        debug_assert!(!maximum.is_zero());
        Self(self.0.min(maximum))
    }
}

/// Fully parsed configuration for fixed, read-only head probing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeadProbeConfig {
    device: DeviceIdentity,
    response_timeout: OperationTimeout,
    request_timeout: OperationTimeout,
    noise_budget_bytes: u16,
}

impl HeadProbeConfig {
    pub fn parse(input: HeadProbeConfigInput) -> Result<Self, ConfigParseError> {
        Ok(Self {
            device: DeviceIdentity::parse(input.device_path)?,
            response_timeout: OperationTimeout::parse(
                "response_timeout_ms",
                input.response_timeout_ms,
            )?,
            request_timeout: OperationTimeout::parse(
                "request_timeout_ms",
                input.request_timeout_ms,
            )?,
            noise_budget_bytes: parse_noise_budget(input.noise_budget_bytes)?,
        })
    }

    pub const fn device(&self) -> &DeviceIdentity {
        &self.device
    }

    pub const fn response_timeout(&self) -> OperationTimeout {
        self.response_timeout
    }

    pub const fn request_timeout(&self) -> OperationTimeout {
        self.request_timeout
    }

    pub const fn noise_budget_bytes(&self) -> u16 {
        self.noise_budget_bytes
    }

    /// Derive the read-only probe boundary from an already parsed runtime
    /// policy without reconstructing weak strings or numeric units.
    ///
    /// The fixed READ request uses the runtime's bounded write timeout; the
    /// device identity, response timeout, and framing-noise budget retain the
    /// exact parsed values. This conversion does not open the serial device or
    /// grant torque consent.
    pub fn from_runtime(runtime: &HeadRuntimeConfig) -> Self {
        Self {
            device: runtime.device.clone(),
            response_timeout: runtime.response_timeout,
            request_timeout: runtime.write_timeout,
            noise_budget_bytes: runtime.noise_budget_bytes,
        }
    }
}

/// Number of permitted write attempts. Only a retryable zero-progress failure
/// can consume an attempt after the first one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WriteAttemptLimit(u8);

impl WriteAttemptLimit {
    fn parse(value: u8) -> Result<Self, ConfigParseError> {
        if !(1..=MAX_WRITE_ATTEMPTS).contains(&value) {
            return Err(ConfigParseError::WriteAttemptsOutOfRange {
                value,
                minimum: 1,
                maximum: MAX_WRITE_ATTEMPTS,
            });
        }
        Ok(Self(value))
    }

    pub const fn get(self) -> u8 {
        self.0
    }
}

/// Maximum permitted age of the oldest admitted joint observation when each
/// servo is armed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ArmingFreshness(Duration);

impl ArmingFreshness {
    fn parse(milliseconds: u64) -> Result<Self, ConfigParseError> {
        if !(1..=MAX_ARMING_FRESHNESS_MS).contains(&milliseconds) {
            return Err(ConfigParseError::ArmingFreshnessOutOfRange {
                milliseconds,
                minimum_ms: 1,
                maximum_ms: MAX_ARMING_FRESHNESS_MS,
            });
        }
        Ok(Self(Duration::from_millis(milliseconds)))
    }

    pub const fn get(self) -> Duration {
        self.0
    }
}

/// Exact caller-reviewed raw-encoder windows in bow/curl/yaw/roll order.
///
/// Construction rejects descending, out-of-domain, and overly broad windows;
/// in particular, the full encoder range is unrepresentable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConfiguredHeadPoseBounds {
    minimum: [PositionTicks; 4],
    maximum: [PositionTicks; 4],
}

/// Weak fields required only by the explicitly armed observed-position hold.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObservedHoldConfigInput {
    pub write_timeout_ms: u64,
    pub arming_freshness_ms: u64,
    pub write_attempts: u8,
    pub redundant_read_tolerance_ticks: u16,
    pub readback_tolerance_ticks: u16,
    pub goal_speed_ticks_per_second: u16,
    pub torque_limit_permille: [u16; 4],
    pub minimum_ticks: [u16; 4],
    pub maximum_ticks: [u16; 4],
    pub maximum_hold_ms: u64,
}

/// Weak fields for one explicitly reviewed return-to-target transaction.
/// Start windows, target, and travel limits are all in canonical
/// bow/curl/yaw/roll order. Post-return ownership lifetime is deliberately not
/// part of this transaction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReturnToTargetConfigInput {
    pub write_timeout_ms: u64,
    pub arming_freshness_ms: u64,
    pub write_attempts: u8,
    pub redundant_read_tolerance_ticks: u16,
    /// Startup observed-position hold readback tolerance only.
    pub readback_tolerance_ticks: u16,
    /// Completion distance from the reviewed target.
    pub final_target_tolerance_ticks: u16,
    /// Maximum encoder noise margin around the admitted start-to-target path.
    pub path_corridor_tolerance_ticks: u16,
    /// Maximum regression from the best observed target distance.
    pub direction_regression_tolerance_ticks: u16,
    pub goal_speed_ticks_per_second: u16,
    pub torque_limit_permille: [u16; 4],
    pub minimum_start_ticks: [u16; 4],
    pub maximum_start_ticks: [u16; 4],
    pub target_ticks: [u16; 4],
    pub maximum_travel_ticks: [u16; 4],
}

/// Fully parsed target and structural travel bounds. Timing of the motion
/// transaction is fixed by this crate rather than accepted from weak input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HeadReturnPlan {
    target: ExactHeadTargetPose,
    maximum_travel_ticks: [u16; 4],
    final_target_tolerance: PositionAgreementTicks,
    path_corridor_tolerance: PositionAgreementTicks,
    direction_regression_tolerance: PositionAgreementTicks,
    final_sample_tolerance: PositionAgreementTicks,
}

impl HeadReturnPlan {
    pub const fn target(self) -> ExactHeadTargetPose {
        self.target
    }

    pub const fn maximum_travel_ticks(self, joint: HeadJoint) -> u16 {
        self.maximum_travel_ticks[joint as usize]
    }

    pub const fn position_step_ticks(self) -> u16 {
        HEAD_RETURN_POSITION_STEP_TICKS
    }

    pub const fn control_period(self) -> Duration {
        HEAD_RETURN_CONTROL_PERIOD
    }

    pub const fn no_progress_timeout(self) -> Duration {
        HEAD_RETURN_NO_PROGRESS_TIMEOUT
    }

    pub const fn motion_timeout(self) -> Duration {
        HEAD_RETURN_MOTION_TIMEOUT
    }

    pub const fn telemetry_set_max_age(self) -> Duration {
        HEAD_RETURN_TELEMETRY_SET_MAX_AGE
    }

    pub const fn final_target_tolerance(self) -> PositionAgreementTicks {
        self.final_target_tolerance
    }

    pub const fn path_corridor_tolerance(self) -> PositionAgreementTicks {
        self.path_corridor_tolerance
    }

    pub const fn direction_regression_tolerance(self) -> PositionAgreementTicks {
        self.direction_regression_tolerance
    }

    pub const fn final_sample_tolerance(self) -> PositionAgreementTicks {
        self.final_sample_tolerance
    }

    #[cfg(test)]
    pub(crate) const fn for_test(
        target: ExactHeadTargetPose,
        maximum_travel_ticks: [u16; 4],
        final_target_tolerance: PositionAgreementTicks,
        path_corridor_tolerance: PositionAgreementTicks,
        direction_regression_tolerance: PositionAgreementTicks,
        final_sample_tolerance: PositionAgreementTicks,
    ) -> Self {
        Self {
            target,
            maximum_travel_ticks,
            final_target_tolerance,
            path_corridor_tolerance,
            direction_regression_tolerance,
            final_sample_tolerance,
        }
    }
}

/// Parsed return transaction derived from one already parsed probe boundary.
///
/// This type contains no hold lease or duration. A caller may retain the
/// returned actor continuously or apply its own separately typed lifetime
/// policy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReturnToTargetConfig {
    runtime: HeadRuntimeConfig,
    start_bounds: ConfiguredHeadPoseBounds,
    plan: HeadReturnPlan,
}

impl ReturnToTargetConfig {
    pub fn parse(
        probe: &HeadProbeConfig,
        input: ReturnToTargetConfigInput,
    ) -> Result<Self, ReturnToTargetConfigParseError> {
        let start_bounds =
            ConfiguredHeadPoseBounds::try_new(input.minimum_start_ticks, input.maximum_start_ticks)
                .map_err(ReturnToTargetConfigParseError::StartBounds)?;
        let target = ExactHeadTargetPose::try_from_ticks(input.target_ticks)
            .map_err(ReturnToTargetConfigParseError::Target)?;
        let final_target_tolerance = parse_return_tolerance(
            "final_target_tolerance_ticks",
            input.final_target_tolerance_ticks,
        )?;
        let path_corridor_tolerance = parse_return_tolerance(
            "path_corridor_tolerance_ticks",
            input.path_corridor_tolerance_ticks,
        )?;
        let direction_regression_tolerance = parse_return_tolerance(
            "direction_regression_tolerance_ticks",
            input.direction_regression_tolerance_ticks,
        )?;
        if final_target_tolerance > path_corridor_tolerance {
            return Err(ReturnToTargetConfigParseError::ToleranceOrdering {
                smaller_field: "final_target_tolerance_ticks",
                smaller_value: final_target_tolerance.get(),
                larger_field: "path_corridor_tolerance_ticks",
                larger_value: path_corridor_tolerance.get(),
            });
        }
        if direction_regression_tolerance > path_corridor_tolerance {
            return Err(ReturnToTargetConfigParseError::ToleranceOrdering {
                smaller_field: "direction_regression_tolerance_ticks",
                smaller_value: direction_regression_tolerance.get(),
                larger_field: "path_corridor_tolerance_ticks",
                larger_value: path_corridor_tolerance.get(),
            });
        }
        for joint in HeadJoint::ALL {
            let index = joint as usize;
            let configured_limit = input.maximum_travel_ticks[index];
            if configured_limit == 0 || configured_limit > MAX_HEAD_RETURN_TRAVEL_TICKS {
                return Err(ReturnToTargetConfigParseError::InvalidMaximumTravel {
                    joint,
                    value: configured_limit,
                    maximum: MAX_HEAD_RETURN_TRAVEL_TICKS,
                });
            }
            let target_ticks = target.position(joint).get();
            let required = target_ticks
                .abs_diff(start_bounds.minimum(joint).get())
                .max(target_ticks.abs_diff(start_bounds.maximum(joint).get()));
            if required > configured_limit {
                return Err(ReturnToTargetConfigParseError::TravelAboveMaximum {
                    joint,
                    required_ticks: required,
                    maximum_ticks: configured_limit,
                });
            }
            for (field, tolerance) in [
                ("final_target_tolerance_ticks", final_target_tolerance),
                ("path_corridor_tolerance_ticks", path_corridor_tolerance),
                (
                    "direction_regression_tolerance_ticks",
                    direction_regression_tolerance,
                ),
            ] {
                if tolerance.get() > configured_limit {
                    return Err(ReturnToTargetConfigParseError::ToleranceAboveTravel {
                        joint,
                        field,
                        tolerance_ticks: tolerance.get(),
                        maximum_travel_ticks: configured_limit,
                    });
                }
            }
        }
        let runtime = HeadRuntimeConfig::parse_tuning(
            probe.device.clone(),
            probe.response_timeout,
            probe.noise_budget_bytes,
            RuntimeTuningInput {
                write_timeout_ms: input.write_timeout_ms,
                arming_freshness_ms: input.arming_freshness_ms,
                write_attempts: input.write_attempts,
                redundant_read_tolerance_ticks: input.redundant_read_tolerance_ticks,
                readback_tolerance_ticks: input.readback_tolerance_ticks,
                goal_speed_ticks_per_second: input.goal_speed_ticks_per_second,
                torque_limit_permille: input.torque_limit_permille,
            },
        )
        .map_err(ReturnToTargetConfigParseError::Runtime)?;
        let final_sample_tolerance = runtime.redundant_read_tolerance();
        Ok(Self {
            runtime,
            start_bounds,
            plan: HeadReturnPlan {
                target,
                maximum_travel_ticks: input.maximum_travel_ticks,
                final_target_tolerance,
                path_corridor_tolerance,
                direction_regression_tolerance,
                final_sample_tolerance,
            },
        })
    }

    pub const fn runtime(&self) -> &HeadRuntimeConfig {
        &self.runtime
    }

    /// Exact reviewed target retained by this inseparable return plan.
    pub const fn target(&self) -> ExactHeadTargetPose {
        self.plan.target()
    }

    pub const fn start_bounds(&self) -> ConfiguredHeadPoseBounds {
        self.start_bounds
    }

    #[cfg(test)]
    pub(crate) const fn plan(&self) -> HeadReturnPlan {
        self.plan
    }

    pub(crate) fn into_actor_parts(
        self,
    ) -> (HeadRuntimeConfig, ConfiguredHeadPoseBounds, HeadReturnPlan) {
        (self.runtime, self.start_bounds, self.plan)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReturnToTargetConfigParseError {
    Runtime(ConfigParseError),
    StartBounds(ConfiguredHeadPoseBoundsError),
    Target(ExactHeadTargetPoseError),
    InvalidMaximumTravel {
        joint: HeadJoint,
        value: u16,
        maximum: u16,
    },
    TravelAboveMaximum {
        joint: HeadJoint,
        required_ticks: u16,
        maximum_ticks: u16,
    },
    InvalidTolerance {
        field: &'static str,
        source: PositionAgreementError,
    },
    ToleranceOrdering {
        smaller_field: &'static str,
        smaller_value: u16,
        larger_field: &'static str,
        larger_value: u16,
    },
    ToleranceAboveTravel {
        joint: HeadJoint,
        field: &'static str,
        tolerance_ticks: u16,
        maximum_travel_ticks: u16,
    },
}

impl fmt::Display for ReturnToTargetConfigParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid return-to-target configuration: {self:?}"
        )
    }
}

impl std::error::Error for ReturnToTargetConfigParseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Runtime(source) => Some(source),
            Self::StartBounds(source) => Some(source),
            Self::Target(source) => Some(source),
            Self::InvalidTolerance { source, .. } => Some(source),
            Self::InvalidMaximumTravel { .. }
            | Self::TravelAboveMaximum { .. }
            | Self::ToleranceOrdering { .. }
            | Self::ToleranceAboveTravel { .. } => None,
        }
    }
}

fn parse_return_tolerance(
    field: &'static str,
    value: u16,
) -> Result<PositionAgreementTicks, ReturnToTargetConfigParseError> {
    PositionAgreementTicks::try_new(value)
        .map_err(|source| ReturnToTargetConfigParseError::InvalidTolerance { field, source })
}

/// Typed, bounded observed-position hold configuration derived from an
/// already parsed probe boundary. Device identity and read settings are never
/// parsed or validated a second time.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObservedHoldConfig {
    runtime: HeadRuntimeConfig,
    pose_bounds: ConfiguredHeadPoseBounds,
    maximum_duration: Duration,
}

impl ObservedHoldConfig {
    pub fn parse(
        probe: &HeadProbeConfig,
        input: ObservedHoldConfigInput,
    ) -> Result<Self, ObservedHoldConfigParseError> {
        let pose_bounds =
            ConfiguredHeadPoseBounds::try_new(input.minimum_ticks, input.maximum_ticks)
                .map_err(ObservedHoldConfigParseError::PoseBounds)?;
        if !(1..=MAX_HOLD_DURATION_MS).contains(&input.maximum_hold_ms) {
            return Err(ObservedHoldConfigParseError::DurationOutOfRange {
                milliseconds: input.maximum_hold_ms,
                minimum_ms: 1,
                maximum_ms: MAX_HOLD_DURATION_MS,
            });
        }
        let runtime = HeadRuntimeConfig::parse_tuning(
            probe.device.clone(),
            probe.response_timeout,
            probe.noise_budget_bytes,
            RuntimeTuningInput {
                write_timeout_ms: input.write_timeout_ms,
                arming_freshness_ms: input.arming_freshness_ms,
                write_attempts: input.write_attempts,
                redundant_read_tolerance_ticks: input.redundant_read_tolerance_ticks,
                readback_tolerance_ticks: input.readback_tolerance_ticks,
                goal_speed_ticks_per_second: input.goal_speed_ticks_per_second,
                torque_limit_permille: input.torque_limit_permille,
            },
        )
        .map_err(ObservedHoldConfigParseError::Runtime)?;
        Ok(Self {
            runtime,
            pose_bounds,
            maximum_duration: Duration::from_millis(input.maximum_hold_ms),
        })
    }

    pub const fn runtime(&self) -> &HeadRuntimeConfig {
        &self.runtime
    }

    pub const fn pose_bounds(&self) -> ConfiguredHeadPoseBounds {
        self.pose_bounds
    }

    pub const fn maximum_duration(&self) -> Duration {
        self.maximum_duration
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ObservedHoldConfigParseError {
    Runtime(ConfigParseError),
    PoseBounds(ConfiguredHeadPoseBoundsError),
    DurationOutOfRange {
        milliseconds: u64,
        minimum_ms: u64,
        maximum_ms: u64,
    },
}

impl fmt::Display for ObservedHoldConfigParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid observed-position hold configuration: {self:?}"
        )
    }
}

impl std::error::Error for ObservedHoldConfigParseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Runtime(source) => Some(source),
            Self::PoseBounds(source) => Some(source),
            Self::DurationOutOfRange { .. } => None,
        }
    }
}

impl ConfiguredHeadPoseBounds {
    pub fn try_new(
        minimum_ticks: [u16; 4],
        maximum_ticks: [u16; 4],
    ) -> Result<Self, ConfiguredHeadPoseBoundsError> {
        let mut minimum = [PositionTicks::MIN; 4];
        let mut maximum = [PositionTicks::MIN; 4];
        for (index, joint) in HeadJoint::ALL.into_iter().enumerate() {
            minimum[index] = PositionTicks::try_new(minimum_ticks[index]).map_err(|source| {
                ConfiguredHeadPoseBoundsError::Position {
                    joint,
                    bound: ConfiguredHeadPoseBound::Minimum,
                    source,
                }
            })?;
            maximum[index] = PositionTicks::try_new(maximum_ticks[index]).map_err(|source| {
                ConfiguredHeadPoseBoundsError::Position {
                    joint,
                    bound: ConfiguredHeadPoseBound::Maximum,
                    source,
                }
            })?;
            if minimum[index] > maximum[index] {
                return Err(ConfiguredHeadPoseBoundsError::Descending {
                    joint,
                    minimum: minimum[index],
                    maximum: maximum[index],
                });
            }
            let span_ticks = maximum[index].get() - minimum[index].get();
            if span_ticks > MAX_CONFIGURED_POSE_WINDOW_SPAN_TICKS {
                return Err(ConfiguredHeadPoseBoundsError::SpanAboveMaximum {
                    joint,
                    minimum: minimum[index],
                    maximum: maximum[index],
                    span_ticks,
                    maximum_span_ticks: MAX_CONFIGURED_POSE_WINDOW_SPAN_TICKS,
                });
            }
        }
        Ok(Self { minimum, maximum })
    }

    pub const fn minimum(self, joint: HeadJoint) -> PositionTicks {
        self.minimum[joint as usize]
    }

    pub const fn maximum(self, joint: HeadJoint) -> PositionTicks {
        self.maximum[joint as usize]
    }

    pub fn admit(
        self,
        observed_pose: HeadPose,
    ) -> Result<HeadPoseWithinConfiguredBounds, HeadPoseBoundsAdmissionError> {
        for joint in HeadJoint::ALL {
            let observed = observed_pose.position(joint);
            if observed < self.minimum(joint) || observed > self.maximum(joint) {
                return Err(HeadPoseBoundsAdmissionError::OutsideConfiguredWindow {
                    joint,
                    observed,
                    minimum: self.minimum(joint),
                    maximum: self.maximum(joint),
                });
            }
        }
        Ok(HeadPoseWithinConfiguredBounds {
            observed_pose,
            configured_bounds: self,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConfiguredHeadPoseBound {
    Minimum,
    Maximum,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConfiguredHeadPoseBoundsError {
    Position {
        joint: HeadJoint,
        bound: ConfiguredHeadPoseBound,
        source: FrameBuildError,
    },
    Descending {
        joint: HeadJoint,
        minimum: PositionTicks,
        maximum: PositionTicks,
    },
    SpanAboveMaximum {
        joint: HeadJoint,
        minimum: PositionTicks,
        maximum: PositionTicks,
        span_ticks: u16,
        maximum_span_ticks: u16,
    },
}

impl fmt::Display for ConfiguredHeadPoseBoundsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid configured Kiko head pose bounds: {self:?}"
        )
    }
}

impl std::error::Error for ConfiguredHeadPoseBoundsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Position { source, .. } => Some(source),
            Self::Descending { .. } | Self::SpanAboveMaximum { .. } => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadPoseBoundsAdmissionError {
    OutsideConfiguredWindow {
        joint: HeadJoint,
        observed: PositionTicks,
        minimum: PositionTicks,
        maximum: PositionTicks,
    },
}

impl fmt::Display for HeadPoseBoundsAdmissionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "observed head pose was not admitted: {self:?}")
    }
}

impl std::error::Error for HeadPoseBoundsAdmissionError {}

/// Evidence constructed only when the complete observed pose is inside all
/// four caller-reviewed windows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadPoseWithinConfiguredBounds {
    observed_pose: HeadPose,
    configured_bounds: ConfiguredHeadPoseBounds,
}

impl HeadPoseWithinConfiguredBounds {
    pub const fn observed_pose(self) -> HeadPose {
        self.observed_pose
    }

    pub const fn configured_bounds(self) -> ConfiguredHeadPoseBounds {
        self.configured_bounds
    }
}

/// Fully parsed runtime configuration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeadRuntimeConfig {
    device: DeviceIdentity,
    response_timeout: OperationTimeout,
    write_timeout: OperationTimeout,
    write_attempts: WriteAttemptLimit,
    arming_freshness: ArmingFreshness,
    noise_budget_bytes: u16,
    redundant_read_tolerance: PositionAgreementTicks,
    readback_tolerance: PositionAgreementTicks,
    goal_speed: GoalSpeedTicksPerSecond,
    torque_limits: HeadTorqueLimits,
}

impl HeadRuntimeConfig {
    pub fn parse(input: HeadRuntimeConfigInput) -> Result<Self, ConfigParseError> {
        let device = DeviceIdentity::parse(input.device_path)?;
        let response_timeout =
            OperationTimeout::parse("response_timeout_ms", input.response_timeout_ms)?;
        let noise_budget_bytes = parse_noise_budget(input.noise_budget_bytes)?;
        Self::parse_tuning(
            device,
            response_timeout,
            noise_budget_bytes,
            RuntimeTuningInput {
                write_timeout_ms: input.write_timeout_ms,
                arming_freshness_ms: input.arming_freshness_ms,
                write_attempts: input.write_attempts,
                redundant_read_tolerance_ticks: input.redundant_read_tolerance_ticks,
                readback_tolerance_ticks: input.readback_tolerance_ticks,
                goal_speed_ticks_per_second: input.goal_speed_ticks_per_second,
                torque_limit_permille: input.torque_limit_permille,
            },
        )
    }

    fn parse_tuning(
        device: DeviceIdentity,
        response_timeout: OperationTimeout,
        noise_budget_bytes: u16,
        input: RuntimeTuningInput,
    ) -> Result<Self, ConfigParseError> {
        let write_timeout = OperationTimeout::parse("write_timeout_ms", input.write_timeout_ms)?;
        let write_attempts = WriteAttemptLimit::parse(input.write_attempts)?;
        let arming_freshness = ArmingFreshness::parse(input.arming_freshness_ms)?;
        let maximum_write_budget = write_timeout
            .get()
            .checked_mul(u32::from(write_attempts.get()))
            .expect("bounded write timeout and attempts fit Duration");
        if arming_freshness.get() < maximum_write_budget {
            return Err(ConfigParseError::ArmingFreshnessShorterThanWriteBudget {
                arming_freshness: arming_freshness.get(),
                maximum_write_budget,
            });
        }
        let redundant_read_tolerance = PositionAgreementTicks::try_new(
            input.redundant_read_tolerance_ticks,
        )
        .map_err(|source| ConfigParseError::InvalidPositionTolerance {
            field: "redundant_read_tolerance_ticks",
            source,
        })?;
        let readback_tolerance = PositionAgreementTicks::try_new(input.readback_tolerance_ticks)
            .map_err(|source| ConfigParseError::InvalidPositionTolerance {
                field: "readback_tolerance_ticks",
                source,
            })?;
        let goal_speed = GoalSpeedTicksPerSecond::try_new(input.goal_speed_ticks_per_second)
            .map_err(|source| ConfigParseError::InvalidGoalSpeed { source })?;
        let [bow, curl, yaw, roll] = input.torque_limit_permille;
        let torque_limits = HeadTorqueLimits::new(
            parse_torque_limit(HeadJoint::Bow, bow)?,
            parse_torque_limit(HeadJoint::Curl, curl)?,
            parse_torque_limit(HeadJoint::Yaw, yaw)?,
            parse_torque_limit(HeadJoint::Roll, roll)?,
        );

        Ok(Self {
            device,
            response_timeout,
            write_timeout,
            write_attempts,
            arming_freshness,
            noise_budget_bytes,
            redundant_read_tolerance,
            readback_tolerance,
            goal_speed,
            torque_limits,
        })
    }

    pub const fn device(&self) -> &DeviceIdentity {
        &self.device
    }

    pub const fn response_timeout(&self) -> OperationTimeout {
        self.response_timeout
    }

    pub const fn write_timeout(&self) -> OperationTimeout {
        self.write_timeout
    }

    pub const fn write_attempts(&self) -> WriteAttemptLimit {
        self.write_attempts
    }

    pub const fn arming_freshness(&self) -> ArmingFreshness {
        self.arming_freshness
    }

    pub const fn noise_budget_bytes(&self) -> u16 {
        self.noise_budget_bytes
    }

    pub const fn redundant_read_tolerance(&self) -> PositionAgreementTicks {
        self.redundant_read_tolerance
    }

    pub const fn readback_tolerance(&self) -> PositionAgreementTicks {
        self.readback_tolerance
    }

    pub const fn goal_speed(&self) -> GoalSpeedTicksPerSecond {
        self.goal_speed
    }

    pub const fn torque_limits(&self) -> HeadTorqueLimits {
        self.torque_limits
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RuntimeTuningInput {
    write_timeout_ms: u64,
    arming_freshness_ms: u64,
    write_attempts: u8,
    redundant_read_tolerance_ticks: u16,
    readback_tolerance_ticks: u16,
    goal_speed_ticks_per_second: u16,
    torque_limit_permille: [u16; 4],
}

fn parse_noise_budget(value: u16) -> Result<u16, ConfigParseError> {
    if value > MAX_NOISE_BUDGET_BYTES {
        return Err(ConfigParseError::NoiseBudgetOutOfRange {
            value,
            maximum: MAX_NOISE_BUDGET_BYTES,
        });
    }
    Ok(value)
}

fn parse_torque_limit(
    joint: HeadJoint,
    value: u16,
) -> Result<TorqueLimitPermille, ConfigParseError> {
    TorqueLimitPermille::try_new(value)
        .map_err(|source| ConfigParseError::InvalidTorqueLimit { joint, source })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConfigParseError {
    EmptyDevicePath,
    DevicePathTooLong {
        actual_bytes: usize,
        maximum_bytes: usize,
    },
    DevicePathContainsNul,
    UnsupportedDevicePath {
        path: String,
    },
    InvalidDeviceIdentity {
        identity: String,
    },
    DeviceIdentityTooLong {
        actual_bytes: usize,
        maximum_bytes: usize,
    },
    OperationTimeoutOutOfRange {
        field: &'static str,
        milliseconds: u64,
        minimum_ms: u64,
        maximum_ms: u64,
    },
    WriteAttemptsOutOfRange {
        value: u8,
        minimum: u8,
        maximum: u8,
    },
    ArmingFreshnessOutOfRange {
        milliseconds: u64,
        minimum_ms: u64,
        maximum_ms: u64,
    },
    ArmingFreshnessShorterThanWriteBudget {
        arming_freshness: Duration,
        maximum_write_budget: Duration,
    },
    NoiseBudgetOutOfRange {
        value: u16,
        maximum: u16,
    },
    InvalidPositionTolerance {
        field: &'static str,
        source: PositionAgreementError,
    },
    InvalidGoalSpeed {
        source: FrameBuildError,
    },
    InvalidTorqueLimit {
        joint: HeadJoint,
        source: FrameBuildError,
    },
}

impl fmt::Display for ConfigParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid Kiko head runtime configuration: {self:?}"
        )
    }
}

impl std::error::Error for ConfigParseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidPositionTolerance { source, .. } => Some(source),
            Self::InvalidGoalSpeed { source } | Self::InvalidTorqueLimit { source, .. } => {
                Some(source)
            }
            Self::EmptyDevicePath
            | Self::DevicePathTooLong { .. }
            | Self::DevicePathContainsNul
            | Self::UnsupportedDevicePath { .. }
            | Self::InvalidDeviceIdentity { .. }
            | Self::DeviceIdentityTooLong { .. }
            | Self::OperationTimeoutOutOfRange { .. }
            | Self::WriteAttemptsOutOfRange { .. }
            | Self::ArmingFreshnessOutOfRange { .. }
            | Self::ArmingFreshnessShorterThanWriteBudget { .. }
            | Self::NoiseBudgetOutOfRange { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_input() -> HeadRuntimeConfigInput {
        HeadRuntimeConfigInput {
            device_path: "/dev/serial/by-id/usb-Kiko_STS_adapter_0001".to_owned(),
            response_timeout_ms: 100,
            write_timeout_ms: 100,
            write_attempts: 2,
            arming_freshness_ms: 250,
            noise_budget_bytes: 32,
            redundant_read_tolerance_ticks: 10,
            readback_tolerance_ticks: 20,
            goal_speed_ticks_per_second: 100,
            torque_limit_permille: [600, 400, 400, 400],
        }
    }

    fn valid_probe() -> HeadProbeConfig {
        HeadProbeConfig::parse(HeadProbeConfigInput {
            device_path: "/dev/serial/by-id/usb-Kiko_STS_adapter_0001".to_owned(),
            response_timeout_ms: 100,
            request_timeout_ms: 100,
            noise_budget_bytes: 32,
        })
        .expect("valid probe configuration")
    }

    fn valid_hold_input() -> ObservedHoldConfigInput {
        ObservedHoldConfigInput {
            write_timeout_ms: 100,
            arming_freshness_ms: 250,
            write_attempts: 2,
            redundant_read_tolerance_ticks: 10,
            readback_tolerance_ticks: 20,
            goal_speed_ticks_per_second: 100,
            torque_limit_permille: [600, 400, 400, 400],
            minimum_ticks: [1_900; 4],
            maximum_ticks: [2_100; 4],
            maximum_hold_ms: 60_000,
        }
    }

    fn valid_return_input() -> ReturnToTargetConfigInput {
        ReturnToTargetConfigInput {
            write_timeout_ms: 100,
            arming_freshness_ms: 250,
            write_attempts: 2,
            redundant_read_tolerance_ticks: 10,
            readback_tolerance_ticks: 20,
            final_target_tolerance_ticks: 20,
            path_corridor_tolerance_ticks: 20,
            direction_regression_tolerance_ticks: 20,
            goal_speed_ticks_per_second: 50,
            torque_limit_permille: [600, 400, 400, 400],
            minimum_start_ticks: [2_496, 2_900, 2_887, 2_887],
            maximum_start_ticks: [2_518, 2_932, 2_919, 2_919],
            target_ticks: [2_155, 2_545, 2_943, 2_876],
            maximum_travel_ticks: [400, 400, 64, 64],
        }
    }

    #[test]
    fn probe_and_hold_boundaries_parse_common_identity_once() {
        let probe = valid_probe();
        let hold = ObservedHoldConfig::parse(&probe, valid_hold_input())
            .expect("valid observed-position hold");

        assert_eq!(hold.runtime().device(), probe.device());
        assert_eq!(hold.runtime().response_timeout(), probe.response_timeout());
        assert_eq!(
            hold.runtime().noise_budget_bytes(),
            probe.noise_budget_bytes()
        );
        assert_eq!(hold.maximum_duration(), Duration::from_secs(60));
    }

    #[test]
    fn read_only_probe_is_derived_from_typed_runtime_without_reparsing() {
        let runtime = HeadRuntimeConfig::parse(valid_input()).expect("valid runtime");
        let probe = HeadProbeConfig::from_runtime(&runtime);

        assert_eq!(probe.device(), runtime.device());
        assert_eq!(probe.response_timeout(), runtime.response_timeout());
        assert_eq!(probe.request_timeout(), runtime.write_timeout());
        assert_eq!(probe.noise_budget_bytes(), runtime.noise_budget_bytes());
    }

    #[test]
    fn return_boundary_parses_exact_target_and_fixed_motion_policy_once() {
        let probe = valid_probe();
        let config =
            ReturnToTargetConfig::parse(&probe, valid_return_input()).expect("valid return config");

        assert_eq!(config.runtime().device(), probe.device());
        assert_eq!(
            config.plan().target().positions().map(PositionTicks::get),
            [2_155, 2_545, 2_943, 2_876]
        );
        assert_eq!(config.plan().position_step_ticks(), 50);
        assert_eq!(config.plan().control_period(), Duration::from_millis(100));
        assert_eq!(config.plan().no_progress_timeout(), Duration::from_secs(2));
        assert_eq!(config.plan().motion_timeout(), Duration::from_secs(20));
        assert_eq!(
            config.plan().telemetry_set_max_age(),
            Duration::from_millis(100)
        );
        assert_eq!(config.plan().final_target_tolerance().get(), 20);
        assert_eq!(config.plan().path_corridor_tolerance().get(), 20);
        assert_eq!(config.plan().direction_regression_tolerance().get(), 20);
        assert_eq!(config.plan().final_sample_tolerance().get(), 10);
    }

    #[test]
    fn return_boundary_rejects_targets_outside_declared_travel() {
        let probe = valid_probe();
        let mut input = valid_return_input();
        input.maximum_travel_ticks[0] = 0;
        assert!(matches!(
            ReturnToTargetConfig::parse(&probe, input),
            Err(ReturnToTargetConfigParseError::InvalidMaximumTravel {
                joint: HeadJoint::Bow,
                value: 0,
                maximum: MAX_HEAD_RETURN_TRAVEL_TICKS,
            })
        ));

        let mut input = valid_return_input();
        input.maximum_travel_ticks[3] = MAX_HEAD_RETURN_TRAVEL_TICKS + 1;
        assert!(matches!(
            ReturnToTargetConfig::parse(&probe, input),
            Err(ReturnToTargetConfigParseError::InvalidMaximumTravel {
                joint: HeadJoint::Roll,
                value,
                maximum: MAX_HEAD_RETURN_TRAVEL_TICKS,
            }) if value == MAX_HEAD_RETURN_TRAVEL_TICKS + 1
        ));

        let mut input = valid_return_input();
        input.maximum_travel_ticks[1] = 300;
        assert!(matches!(
            ReturnToTargetConfig::parse(&probe, input),
            Err(ReturnToTargetConfigParseError::TravelAboveMaximum {
                joint: HeadJoint::Curl,
                required_ticks: 387,
                maximum_ticks: 300,
            })
        ));

        let mut input = valid_return_input();
        input.target_ticks[3] = 4_096;
        assert!(matches!(
            ReturnToTargetConfig::parse(&probe, input),
            Err(ReturnToTargetConfigParseError::Target(
                ExactHeadTargetPoseError::Position {
                    joint: HeadJoint::Roll,
                    ..
                }
            ))
        ));

        let mut input = valid_return_input();
        input.final_target_tolerance_ticks = 21;
        assert!(matches!(
            ReturnToTargetConfig::parse(&probe, input),
            Err(ReturnToTargetConfigParseError::ToleranceOrdering {
                smaller_field: "final_target_tolerance_ticks",
                smaller_value: 21,
                larger_field: "path_corridor_tolerance_ticks",
                larger_value: 20,
            })
        ));

        let mut input = valid_return_input();
        input.maximum_travel_ticks[3] = 10;
        input.minimum_start_ticks[3] = 2_876;
        input.maximum_start_ticks[3] = 2_876;
        assert!(matches!(
            ReturnToTargetConfig::parse(&probe, input),
            Err(ReturnToTargetConfigParseError::ToleranceAboveTravel {
                joint: HeadJoint::Roll,
                field: "final_target_tolerance_ticks",
                tolerance_ticks: 20,
                maximum_travel_ticks: 10,
            })
        ));
    }

    #[test]
    fn observed_hold_duration_and_pose_windows_are_structurally_bounded() {
        let probe = valid_probe();
        let mut input = valid_hold_input();
        input.maximum_hold_ms = 0;
        assert!(matches!(
            ObservedHoldConfig::parse(&probe, input),
            Err(ObservedHoldConfigParseError::DurationOutOfRange { .. })
        ));

        let mut input = valid_hold_input();
        input.maximum_hold_ms = MAX_HOLD_DURATION_MS + 1;
        assert!(matches!(
            ObservedHoldConfig::parse(&probe, input),
            Err(ObservedHoldConfigParseError::DurationOutOfRange { .. })
        ));

        let mut input = valid_hold_input();
        input.maximum_ticks[1] = input.minimum_ticks[1] + MAX_CONFIGURED_POSE_WINDOW_SPAN_TICKS + 1;
        assert!(matches!(
            ObservedHoldConfig::parse(&probe, input),
            Err(ObservedHoldConfigParseError::PoseBounds(
                ConfiguredHeadPoseBoundsError::SpanAboveMaximum {
                    joint: HeadJoint::Curl,
                    ..
                }
            ))
        ));
    }

    #[test]
    fn probe_rejects_unbounded_request_settings() {
        let input = HeadProbeConfigInput {
            device_path: "/dev/serial/by-id/usb-Kiko_STS_adapter_0001".to_owned(),
            response_timeout_ms: 100,
            request_timeout_ms: 0,
            noise_budget_bytes: 32,
        };
        assert!(matches!(
            HeadProbeConfig::parse(input),
            Err(ConfigParseError::OperationTimeoutOutOfRange {
                field: "request_timeout_ms",
                ..
            })
        ));
    }

    #[test]
    fn configured_pose_windows_are_bounded_and_cannot_admit_the_full_encoder_domain() {
        ConfiguredHeadPoseBounds::try_new(
            [1_900; 4],
            [1_900 + MAX_CONFIGURED_POSE_WINDOW_SPAN_TICKS; 4],
        )
        .expect("maximum bounded pose-window span");

        assert!(matches!(
            ConfiguredHeadPoseBounds::try_new(
                [1_900; 4],
                [1_901 + MAX_CONFIGURED_POSE_WINDOW_SPAN_TICKS; 4],
            ),
            Err(ConfiguredHeadPoseBoundsError::SpanAboveMaximum {
                joint: HeadJoint::Bow,
                span_ticks,
                maximum_span_ticks: MAX_CONFIGURED_POSE_WINDOW_SPAN_TICKS,
                ..
            }) if span_ticks == MAX_CONFIGURED_POSE_WINDOW_SPAN_TICKS + 1
        ));
        assert!(matches!(
            ConfiguredHeadPoseBounds::try_new([0; 4], [4_095; 4]),
            Err(ConfiguredHeadPoseBoundsError::SpanAboveMaximum {
                joint: HeadJoint::Bow,
                span_ticks: 4_095,
                ..
            })
        ));
        assert!(matches!(
            ConfiguredHeadPoseBounds::try_new([2_000; 4], [1_999; 4]),
            Err(ConfiguredHeadPoseBoundsError::Descending {
                joint: HeadJoint::Bow,
                ..
            })
        ));
    }

    #[test]
    fn parses_linux_and_macos_exact_device_identities() {
        let linux = HeadRuntimeConfig::parse(valid_input()).expect("Linux by-id config");
        assert_eq!(linux.device().kind(), DeviceIdentityKind::LinuxById);
        assert_eq!(linux.device().stable_name(), "usb-Kiko_STS_adapter_0001");

        let mut mac_input = valid_input();
        mac_input.device_path = "/dev/cu.usbserial-KIKO0001".to_owned();
        let mac = HeadRuntimeConfig::parse(mac_input).expect("macOS callout config");
        assert_eq!(mac.device().kind(), DeviceIdentityKind::MacOsCallout);
    }

    #[test]
    fn rejects_ambiguous_devices_and_every_dangerous_zero() {
        for path in ["/dev/ttyUSB0", "/dev/serial/by-id/", "/dev/cu."] {
            let mut input = valid_input();
            input.device_path = path.to_owned();
            assert!(HeadRuntimeConfig::parse(input).is_err(), "accepted {path}");
        }

        let mut input = valid_input();
        input.response_timeout_ms = 0;
        assert!(HeadRuntimeConfig::parse(input).is_err());
        let mut input = valid_input();
        input.write_attempts = 0;
        assert!(HeadRuntimeConfig::parse(input).is_err());
        let mut input = valid_input();
        input.arming_freshness_ms = 0;
        assert!(HeadRuntimeConfig::parse(input).is_err());
        let mut input = valid_input();
        input.write_timeout_ms = 100;
        input.write_attempts = 2;
        input.arming_freshness_ms = 199;
        assert!(matches!(
            HeadRuntimeConfig::parse(input),
            Err(ConfigParseError::ArmingFreshnessShorterThanWriteBudget { .. })
        ));
        let mut input = valid_input();
        input.goal_speed_ticks_per_second = 0;
        assert!(HeadRuntimeConfig::parse(input).is_err());
        let mut input = valid_input();
        input.torque_limit_permille[2] = 0;
        assert!(matches!(
            HeadRuntimeConfig::parse(input),
            Err(ConfigParseError::InvalidTorqueLimit {
                joint: HeadJoint::Yaw,
                ..
            })
        ));
    }

    #[test]
    fn all_numeric_bounds_are_exact() {
        let mut input = valid_input();
        input.response_timeout_ms = MAX_OPERATION_TIMEOUT_MS;
        input.write_timeout_ms = 1;
        input.write_attempts = MAX_WRITE_ATTEMPTS;
        input.arming_freshness_ms = MAX_ARMING_FRESHNESS_MS;
        input.noise_budget_bytes = MAX_NOISE_BUDGET_BYTES;
        input.redundant_read_tolerance_ticks = 50;
        input.readback_tolerance_ticks = 50;
        assert!(HeadRuntimeConfig::parse(input).is_ok());

        let mut input = valid_input();
        input.response_timeout_ms = MAX_OPERATION_TIMEOUT_MS + 1;
        assert!(HeadRuntimeConfig::parse(input).is_err());
        let mut input = valid_input();
        input.write_attempts = MAX_WRITE_ATTEMPTS + 1;
        assert!(HeadRuntimeConfig::parse(input).is_err());
        let mut input = valid_input();
        input.arming_freshness_ms = MAX_ARMING_FRESHNESS_MS + 1;
        assert!(HeadRuntimeConfig::parse(input).is_err());
        let mut input = valid_input();
        input.noise_budget_bytes = MAX_NOISE_BUDGET_BYTES + 1;
        assert!(HeadRuntimeConfig::parse(input).is_err());
    }
}
