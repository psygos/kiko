//! Candidate-only controller admission for a physically wheels-off session.
//!
//! This module is deliberately disjoint from production actuation admission.
//! Its public constructors require schema-V2 inventory and controller types,
//! so a production V1 launch cannot select the provisional four-PWM profile by
//! changing a mode flag. It provides only the lower typed boundary needed by a
//! separately invoked wheels-off qualification runtime.

use std::fmt;
use std::net::SocketAddr;
use std::path::Path;
use std::time::{Duration, Instant};

use kiko_device_inventory::{
    ControlEndpointTransport, LoadedExpectedManifestV2, ManifestContentSha256,
};
use robot_command_client::{
    AppliedCommandReceipt, ClientConfig, ConfigError as ClientConfigError, DisarmReceipt,
    StopRecoveryPolicy, TimeoutNs, UdpEndpoint, VerifiedControllerAcquisition,
};
use robot_protocol::v2::{
    ControllerSessionClass, ControllerUid, DomainError,
    MAX_OPERATOR_SUPERVISED_FOUR_PWM_PWM_PERCENT,
    OPERATOR_SUPERVISED_FOUR_PWM_MAX_COMMAND_STEP_PERCENT, PhysicalStopSemantics, TimerPwm,
    V2CommandLeaseMs,
};
use robot_server::config::ControllerServerConfigV2;
use serde::Deserialize;

use super::actuation::{
    LiveActuationError, PhysicalActuationSession, StoppedCandidateActuationClient,
};
use super::mpc::{MpcConfigV1, WheelSide};

pub const WHEELS_OFF_CANDIDATE_POLICY_V1: u32 = 1;
pub const MAX_WHEELS_OFF_CANDIDATE_POLICY_JSON_BYTES: usize = 4 * 1_024;
pub const MAX_WHEELS_OFF_CANDIDATE_RUNTIME_SERVICE_INTERVAL: Duration =
    Duration::from_nanos(54_999_999);
const MAX_WHEELS_OFF_CANDIDATE_ATTESTATION_AGE: Duration = Duration::from_secs(30);
const MAX_WHEELS_OFF_CANDIDATE_MANUAL_DEADMAN: Duration = Duration::from_secs(1);

/// Parsed host timing and local electrical cap for one candidate-only runtime.
///
/// The command interval is the minimum cadence the owning runtime must enforce
/// between non-stop applications. The lease is an STM32 deadline, not a
/// scheduler period or a physical stopping-distance claim.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WheelsOffCandidateControllerBinding {
    endpoint: UdpEndpoint,
    status_timeout: TimeoutNs,
    acquire_timeout: TimeoutNs,
    applied_ack_timeout: TimeoutNs,
    stop_recovery: StopRecoveryPolicy,
    command_lease: V2CommandLeaseMs,
    command_interval: Duration,
    scheduling_margin: Duration,
    local_max_abs_pwm_percent: u8,
    manual_test_magnitude_timer_pwm_percent: u8,
    manual_deadman: Duration,
    maximum_attestation_age: Duration,
}

impl WheelsOffCandidateControllerBinding {
    pub fn parse_json(bytes: &[u8]) -> Result<Self, WheelsOffCandidatePolicyError> {
        if bytes.len() > MAX_WHEELS_OFF_CANDIDATE_POLICY_JSON_BYTES {
            return Err(WheelsOffCandidatePolicyError::InputTooLarge {
                actual_bytes: bytes.len(),
                maximum_bytes: MAX_WHEELS_OFF_CANDIDATE_POLICY_JSON_BYTES,
            });
        }
        let mut deserializer = serde_json::Deserializer::from_slice(bytes);
        let dto = WheelsOffCandidatePolicyV1Dto::deserialize(&mut deserializer)
            .map_err(WheelsOffCandidatePolicyError::JsonDecode)?;
        deserializer
            .end()
            .map_err(WheelsOffCandidatePolicyError::JsonTrailingData)?;
        if dto.schema_version != WHEELS_OFF_CANDIDATE_POLICY_V1 {
            return Err(WheelsOffCandidatePolicyError::UnsupportedSchemaVersion {
                actual: dto.schema_version,
                supported: WHEELS_OFF_CANDIDATE_POLICY_V1,
            });
        }
        if !(1..=MAX_OPERATOR_SUPERVISED_FOUR_PWM_PWM_PERCENT)
            .contains(&dto.local_max_abs_pwm_percent)
        {
            return Err(WheelsOffCandidatePolicyError::LocalPwmCapOutOfRange {
                actual: dto.local_max_abs_pwm_percent,
                minimum: 1,
                maximum: MAX_OPERATOR_SUPERVISED_FOUR_PWM_PWM_PERCENT,
            });
        }
        if !(1..=dto.local_max_abs_pwm_percent)
            .contains(&dto.manual_test_magnitude_timer_pwm_percent)
        {
            return Err(
                WheelsOffCandidatePolicyError::ManualTestMagnitudeOutOfRange {
                    actual: dto.manual_test_magnitude_timer_pwm_percent,
                    minimum: 1,
                    local_maximum: dto.local_max_abs_pwm_percent,
                },
            );
        }
        if dto.manual_deadman_ms == 0 {
            return Err(WheelsOffCandidatePolicyError::ZeroDuration {
                field: "manual_deadman_ms",
            });
        }
        let manual_deadman = Duration::from_millis(u64::from(dto.manual_deadman_ms));
        if manual_deadman > MAX_WHEELS_OFF_CANDIDATE_MANUAL_DEADMAN {
            return Err(WheelsOffCandidatePolicyError::DurationAboveMaximum {
                field: "manual_deadman_ms",
                actual: manual_deadman,
                maximum: MAX_WHEELS_OFF_CANDIDATE_MANUAL_DEADMAN,
            });
        }
        let endpoint = dto
            .command_endpoint
            .parse()
            .map_err(WheelsOffCandidatePolicyError::ClientConfig)?;
        let status_timeout = TimeoutNs::try_new(dto.status_timeout_ns)
            .map_err(WheelsOffCandidatePolicyError::ClientConfig)?;
        let acquire_timeout = TimeoutNs::try_new(dto.acquire_timeout_ns)
            .map_err(WheelsOffCandidatePolicyError::ClientConfig)?;
        let applied_ack_timeout = TimeoutNs::try_new(dto.applied_ack_timeout_ns)
            .map_err(WheelsOffCandidatePolicyError::ClientConfig)?;
        let stop_attempt_timeout = TimeoutNs::try_new(dto.stop_attempt_timeout_ns)
            .map_err(WheelsOffCandidatePolicyError::ClientConfig)?;
        let stop_recovery =
            StopRecoveryPolicy::try_new(dto.maximum_stop_recovery_attempts, stop_attempt_timeout)
                .map_err(WheelsOffCandidatePolicyError::ClientConfig)?;
        let command_lease = V2CommandLeaseMs::try_new(dto.command_lease_ms)
            .map_err(WheelsOffCandidatePolicyError::ProtocolDomain)?;
        let command_interval = nonzero_duration("command_interval_ns", dto.command_interval_ns)?;
        let scheduling_margin = nonzero_duration("scheduling_margin_ns", dto.scheduling_margin_ns)?;
        let maximum_attestation_age =
            nonzero_duration("maximum_attestation_age_ns", dto.maximum_attestation_age_ns)?;
        if maximum_attestation_age > MAX_WHEELS_OFF_CANDIDATE_ATTESTATION_AGE {
            return Err(WheelsOffCandidatePolicyError::DurationAboveMaximum {
                field: "maximum_attestation_age_ns",
                actual: maximum_attestation_age,
                maximum: MAX_WHEELS_OFF_CANDIDATE_ATTESTATION_AGE,
            });
        }

        Ok(Self {
            endpoint,
            status_timeout,
            acquire_timeout,
            applied_ack_timeout,
            stop_recovery,
            command_lease,
            command_interval,
            scheduling_margin,
            local_max_abs_pwm_percent: dto.local_max_abs_pwm_percent,
            manual_test_magnitude_timer_pwm_percent: dto.manual_test_magnitude_timer_pwm_percent,
            manual_deadman,
            maximum_attestation_age,
        })
    }

    /// Cross-bind one candidate policy to schema-V2 inventory and server
    /// contracts. Production V1 types cannot be passed to this function.
    pub fn admit(
        self,
        inventory: &LoadedExpectedManifestV2,
        server: &ControllerServerConfigV2,
        server_command_bind: SocketAddr,
    ) -> Result<AdmittedWheelsOffCandidateController, WheelsOffCandidateControllerBindingError>
    {
        let required_class = ControllerSessionClass::OperatorSupervisedFourPwmCandidate;
        if inventory.manifest().controller_session_class() != required_class
            || server.controller_session_class() != required_class
        {
            return Err(WheelsOffCandidateControllerBindingError::ControllerSessionClassMismatch);
        }
        if inventory.manifest().expected_physical_stop_semantics()
            != PhysicalStopSemantics::Unverified
            || server.expected_physical_stop_semantics() != PhysicalStopSemantics::Unverified
        {
            return Err(WheelsOffCandidateControllerBindingError::PhysicalStopSemanticsMismatch);
        }

        let server_endpoint = UdpEndpoint::try_new(server_command_bind)
            .map_err(WheelsOffCandidateControllerBindingError::ServerCommandBind)?;
        if self.endpoint != server_endpoint {
            return Err(
                WheelsOffCandidateControllerBindingError::CommandEndpointMismatch {
                    policy: self.endpoint.socket_addr(),
                    server: server_command_bind,
                },
            );
        }

        let expected = inventory.manifest().as_inventory().stm32();
        if expected.control_endpoint().transport() != ControlEndpointTransport::Udp {
            return Err(
                WheelsOffCandidateControllerBindingError::InventoryEndpointScheme {
                    actual: expected.control_endpoint().transport(),
                },
            );
        }
        let inventory_endpoint = expected
            .control_endpoint()
            .socket_addr()
            .expect("parsed UDP endpoint retains its socket address");
        if inventory_endpoint != server_command_bind {
            return Err(
                WheelsOffCandidateControllerBindingError::InventoryEndpointMismatch {
                    inventory: inventory_endpoint,
                    server: server_command_bind,
                },
            );
        }
        if server.serial_device() != Path::new(expected.serial_path().as_str()) {
            return Err(
                WheelsOffCandidateControllerBindingError::SerialDeviceMismatch {
                    inventory: expected.serial_path().as_str().into(),
                    server: server.serial_device().to_path_buf(),
                },
            );
        }
        if server.controller_uid() != *expected.controller_uid() {
            return Err(WheelsOffCandidateControllerBindingError::ControllerUidMismatch);
        }
        if server.firmware_abi().get() != expected.firmware_abi() {
            return Err(
                WheelsOffCandidateControllerBindingError::FirmwareAbiMismatch {
                    inventory: expected.firmware_abi(),
                    server: server.firmware_abi().get(),
                },
            );
        }
        if server.firmware_build_id().get() != expected.firmware_build_id() {
            return Err(
                WheelsOffCandidateControllerBindingError::FirmwareBuildMismatch {
                    inventory: expected.firmware_build_id(),
                    server: server.firmware_build_id().get(),
                },
            );
        }
        if server.actuator_config_fingerprint() != *expected.hardware_profile() {
            return Err(WheelsOffCandidateControllerBindingError::ActuatorFingerprintMismatch);
        }

        let manifest_cap = inventory.manifest().expected_max_abs_pwm_percent().get();
        let server_cap = server.expected_max_abs_pwm_percent().get();
        if manifest_cap != server_cap {
            return Err(
                WheelsOffCandidateControllerBindingError::ControllerPwmCapMismatch {
                    inventory: manifest_cap,
                    server: server_cap,
                },
            );
        }
        let effective_max_abs_pwm_percent = manifest_cap
            .min(server_cap)
            .min(self.local_max_abs_pwm_percent);
        if self.manual_test_magnitude_timer_pwm_percent > effective_max_abs_pwm_percent {
            return Err(
                WheelsOffCandidateControllerBindingError::ManualTestMagnitudeAboveEffectiveCap {
                    requested_percent: self.manual_test_magnitude_timer_pwm_percent,
                    effective_max_abs_percent: effective_max_abs_pwm_percent,
                },
            );
        }

        let serial_ack = server.serial_applied_ack_timeout();
        let required_timeout = serial_ack
            .checked_add(self.scheduling_margin)
            .ok_or(WheelsOffCandidateControllerBindingError::DurationArithmeticOverflow)?;
        for (field, timeout) in [
            ("status_timeout_ns", self.status_timeout),
            ("acquire_timeout_ns", self.acquire_timeout),
            ("applied_ack_timeout_ns", self.applied_ack_timeout),
            (
                "stop_attempt_timeout_ns",
                self.stop_recovery.attempt_timeout(),
            ),
        ] {
            if timeout.as_duration() < required_timeout {
                return Err(
                    WheelsOffCandidateControllerBindingError::ClientTimeoutBelowServerBudget {
                        field,
                        timeout: timeout.as_duration(),
                        serial_ack,
                        scheduling_margin: self.scheduling_margin,
                    },
                );
            }
            if timeout.as_duration() >= server.maximum_heartbeat_age() {
                return Err(
                    WheelsOffCandidateControllerBindingError::ClientTimeoutReachesHeartbeatAge {
                        field,
                        timeout: timeout.as_duration(),
                        heartbeat_age: server.maximum_heartbeat_age(),
                    },
                );
            }
        }

        let required_interval = server
            .minimum_host_command_interval()
            .checked_add(self.scheduling_margin)
            .ok_or(WheelsOffCandidateControllerBindingError::DurationArithmeticOverflow)?;
        if self.command_interval <= required_interval {
            return Err(
                WheelsOffCandidateControllerBindingError::CommandIntervalHasNoControllerMargin {
                    command_interval: self.command_interval,
                    controller_minimum: server.minimum_host_command_interval(),
                    scheduling_margin: self.scheduling_margin,
                    required_exclusive_lower_bound: required_interval,
                },
            );
        }
        let lease = Duration::from_millis(u64::from(self.command_lease.get()));
        let required_lease = self
            .command_interval
            .checked_add(self.applied_ack_timeout.as_duration())
            .and_then(|value| value.checked_add(self.scheduling_margin))
            .ok_or(WheelsOffCandidateControllerBindingError::DurationArithmeticOverflow)?;
        if lease <= required_lease {
            return Err(
                WheelsOffCandidateControllerBindingError::CommandLeaseCannotBridgeNextApplication {
                    lease,
                    required_exclusive_lower_bound: required_lease,
                },
            );
        }
        // Keep the runtime service turn strictly inside the lease after the
        // exact client acknowledgement and scheduling budgets. Duration is an
        // integer nanosecond domain, so subtracting one nanosecond represents
        // the greatest admissible value under the strict inequality.
        let maximum_lease_service_interval = lease
            .checked_sub(self.applied_ack_timeout.as_duration())
            .and_then(|value| value.checked_sub(self.scheduling_margin))
            .and_then(|value| value.checked_sub(Duration::from_nanos(1)))
            .ok_or(WheelsOffCandidateControllerBindingError::DurationArithmeticOverflow)?;
        let maximum_runtime_service_interval =
            maximum_lease_service_interval.min(self.manual_deadman);
        if maximum_runtime_service_interval > MAX_WHEELS_OFF_CANDIDATE_RUNTIME_SERVICE_INTERVAL {
            return Err(
                WheelsOffCandidateControllerBindingError::RuntimeServiceEnvelopeAboveQualificationMaximum {
                    actual: maximum_runtime_service_interval,
                    maximum: MAX_WHEELS_OFF_CANDIDATE_RUNTIME_SERVICE_INTERVAL,
                },
            );
        }

        let client = ClientConfig::try_new_for_session(
            self.endpoint,
            server.controller_uid(),
            server.firmware_abi(),
            server.firmware_build_id(),
            server.actuator_config_fingerprint(),
            required_class,
            self.status_timeout,
            self.acquire_timeout,
            self.applied_ack_timeout,
            self.stop_recovery,
            self.command_lease,
        )
        .map_err(WheelsOffCandidateControllerBindingError::ClientConfig)?;
        Ok(AdmittedWheelsOffCandidateController {
            client,
            inventory_content_sha256: inventory.content_sha256(),
            effective_max_abs_pwm_percent,
            manual_test_magnitude_timer_pwm_percent: self.manual_test_magnitude_timer_pwm_percent,
            manual_deadman: self.manual_deadman,
            maximum_command_step_percent: OPERATOR_SUPERVISED_FOUR_PWM_MAX_COMMAND_STEP_PERCENT,
            command_interval: self.command_interval,
            command_lease: self.command_lease,
            maximum_runtime_service_interval,
            maximum_attestation_age: self.maximum_attestation_age,
        })
    }
}

fn nonzero_duration(
    field: &'static str,
    nanoseconds: u64,
) -> Result<Duration, WheelsOffCandidatePolicyError> {
    if nanoseconds == 0 {
        Err(WheelsOffCandidatePolicyError::ZeroDuration { field })
    } else {
        Ok(Duration::from_nanos(nanoseconds))
    }
}

/// Candidate-only authority after exact inventory/server/client binding.
#[derive(Debug, PartialEq, Eq)]
pub struct AdmittedWheelsOffCandidateController {
    client: ClientConfig,
    inventory_content_sha256: ManifestContentSha256,
    effective_max_abs_pwm_percent: u8,
    manual_test_magnitude_timer_pwm_percent: u8,
    manual_deadman: Duration,
    maximum_command_step_percent: u8,
    command_interval: Duration,
    command_lease: V2CommandLeaseMs,
    maximum_runtime_service_interval: Duration,
    maximum_attestation_age: Duration,
}

/// Immutable limits derived only after candidate policy, schema-V2 inventory,
/// and server contract have been cross-bound.
///
/// Runtime and UI layers consume this value instead of reopening policy JSON
/// or assuming a firmware cap or host cadence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WheelsOffCandidateLimits {
    effective_max_abs_pwm_percent: u8,
    manual_test_magnitude_timer_pwm_percent: u8,
    manual_deadman: Duration,
    maximum_command_step_percent: u8,
    command_interval: Duration,
    command_lease: V2CommandLeaseMs,
    maximum_runtime_service_interval: Duration,
}

impl WheelsOffCandidateLimits {
    pub const fn effective_max_abs_pwm_percent(self) -> u8 {
        self.effective_max_abs_pwm_percent
    }

    pub const fn manual_test_magnitude_timer_pwm_percent(self) -> u8 {
        self.manual_test_magnitude_timer_pwm_percent
    }

    pub const fn manual_deadman(self) -> Duration {
        self.manual_deadman
    }

    pub const fn maximum_command_step_percent(self) -> u8 {
        self.maximum_command_step_percent
    }

    pub const fn command_interval(self) -> Duration {
        self.command_interval
    }

    pub const fn command_lease(self) -> V2CommandLeaseMs {
        self.command_lease
    }

    /// Longest qualification-owner service interval that fits both the
    /// controller lease budget and the attended manual deadman.
    pub const fn maximum_runtime_service_interval(self) -> Duration {
        self.maximum_runtime_service_interval
    }

    pub fn admit_runtime_service_interval(
        self,
        actual: Duration,
    ) -> Result<WheelsOffCandidateRuntimeServiceInterval, CandidateRuntimeServiceIntervalError>
    {
        if actual.is_zero() {
            return Err(CandidateRuntimeServiceIntervalError::Zero);
        }
        if actual > self.maximum_runtime_service_interval {
            return Err(CandidateRuntimeServiceIntervalError::ExceedsMaximum {
                actual,
                maximum: self.maximum_runtime_service_interval,
            });
        }
        Ok(WheelsOffCandidateRuntimeServiceInterval {
            actual,
            maximum: self.maximum_runtime_service_interval,
        })
    }
}

/// Proof that one exact shadow-navigation control period can service the
/// qualification controller before either admitted deadline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WheelsOffCandidateRuntimeServiceInterval {
    actual: Duration,
    maximum: Duration,
}

impl WheelsOffCandidateRuntimeServiceInterval {
    pub const fn actual(self) -> Duration {
        self.actual
    }

    pub const fn maximum(self) -> Duration {
        self.maximum
    }

    pub fn require_exact_runtime_interval(
        self,
        actual: Duration,
    ) -> Result<(), CandidateRuntimeServiceIntervalError> {
        if actual != self.actual {
            return Err(CandidateRuntimeServiceIntervalError::RuntimeMismatch {
                admitted: self.actual,
                actual,
            });
        }
        Ok(())
    }
}

impl AdmittedWheelsOffCandidateController {
    pub const fn controller_session_class(&self) -> ControllerSessionClass {
        ControllerSessionClass::OperatorSupervisedFourPwmCandidate
    }

    pub const fn inventory_content_sha256(&self) -> ManifestContentSha256 {
        self.inventory_content_sha256
    }

    pub const fn effective_max_abs_pwm_percent(&self) -> u8 {
        self.effective_max_abs_pwm_percent
    }

    pub const fn maximum_command_step_percent(&self) -> u8 {
        self.maximum_command_step_percent
    }

    pub const fn manual_test_magnitude_timer_pwm_percent(&self) -> u8 {
        self.manual_test_magnitude_timer_pwm_percent
    }

    pub const fn manual_deadman(&self) -> Duration {
        self.manual_deadman
    }

    pub const fn command_interval(&self) -> Duration {
        self.command_interval
    }

    pub const fn command_lease(&self) -> V2CommandLeaseMs {
        self.command_lease
    }

    pub const fn limits(&self) -> WheelsOffCandidateLimits {
        WheelsOffCandidateLimits {
            effective_max_abs_pwm_percent: self.effective_max_abs_pwm_percent,
            manual_test_magnitude_timer_pwm_percent: self.manual_test_magnitude_timer_pwm_percent,
            manual_deadman: self.manual_deadman,
            maximum_command_step_percent: self.maximum_command_step_percent,
            command_interval: self.command_interval,
            command_lease: self.command_lease,
            maximum_runtime_service_interval: self.maximum_runtime_service_interval,
        }
    }

    /// Cross-bind the already parsed shadow MPC policy to the candidate
    /// firmware's electrical cap and exact five-point transition invariant.
    ///
    /// This proves compatibility only. It does not grant autonomous physical
    /// output; the wheels-off candidate runtime remains manual-output-only.
    pub fn admit_shadow_mpc(&self, mpc: MpcConfigV1) -> Result<(), CandidateMpcBindingError> {
        for (wheel, (minimum, maximum)) in [
            (WheelSide::Left, mpc.left_pwm_bounds_percent()),
            (WheelSide::Right, mpc.right_pwm_bounds_percent()),
        ] {
            if i16::from(minimum) < -i16::from(self.effective_max_abs_pwm_percent)
                || i16::from(maximum) > i16::from(self.effective_max_abs_pwm_percent)
            {
                return Err(CandidateMpcBindingError::PwmOutsideEffectiveCap {
                    wheel,
                    minimum,
                    maximum,
                    effective_max_abs_percent: self.effective_max_abs_pwm_percent,
                });
            }
        }
        let (left_slew, right_slew) = mpc.maximum_slew_percent_per_step();
        bind_candidate_mpc_slew(
            WheelSide::Left,
            left_slew,
            self.maximum_command_step_percent,
        )?;
        bind_candidate_mpc_slew(
            WheelSide::Right,
            right_slew,
            self.maximum_command_step_percent,
        )
    }

    /// Parse one weak PWM request into a cap-proven target.
    ///
    /// A full-zero target needs no operator claim. Every nonzero target carries
    /// the weak wheels/head/power claim which must still be fresh when each
    /// ramp step is derived.
    pub fn admit_target(
        &self,
        requested: CandidatePwmRequest,
        attestation: Option<&OperatorClaimedWheelsOffAttestation>,
        now: Instant,
    ) -> Result<AdmittedCandidatePwmTarget, CandidatePwmAdmissionError> {
        self.target_authority()
            .admit_target(requested, attestation, now)
    }

    fn target_authority(&self) -> CandidateTargetAuthority {
        CandidateTargetAuthority {
            controller_uid: self.client.controller_uid(),
            inventory_content_sha256: self.inventory_content_sha256,
            effective_max_abs_pwm_percent: self.effective_max_abs_pwm_percent,
            maximum_command_step_percent: self.maximum_command_step_percent,
            maximum_attestation_age: self.maximum_attestation_age,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CandidateTargetAuthority {
    controller_uid: ControllerUid,
    inventory_content_sha256: ManifestContentSha256,
    effective_max_abs_pwm_percent: u8,
    maximum_command_step_percent: u8,
    maximum_attestation_age: Duration,
}

impl CandidateTargetAuthority {
    fn admit_target(
        self,
        requested: CandidatePwmRequest,
        attestation: Option<&OperatorClaimedWheelsOffAttestation>,
        now: Instant,
    ) -> Result<AdmittedCandidatePwmTarget, CandidatePwmAdmissionError> {
        let requested = requested.timer_pwm();
        let attestation = if requested.is_zero() {
            None
        } else {
            let attestation = *attestation
                .ok_or(CandidatePwmAdmissionError::FreshWheelsOffAttestationRequired)?;
            attestation.require_fresh(now, self.maximum_attestation_age)?;
            Some(attestation)
        };
        for (wheel, requested_percent) in [
            ("left", requested.left().get()),
            ("right", requested.right().get()),
        ] {
            if requested_percent.unsigned_abs() > self.effective_max_abs_pwm_percent {
                return Err(CandidatePwmAdmissionError::OutsideEffectiveCap {
                    wheel,
                    requested_percent,
                    maximum_abs_percent: self.effective_max_abs_pwm_percent,
                });
            }
        }
        Ok(AdmittedCandidatePwmTarget {
            timer_pwm: requested,
            authority: self,
            attestation,
        })
    }
}

/// Linear owner of one acquired candidate controller session.
///
/// Construction consumes the candidate admission, preventing another session
/// from being acquired from the same token. All keep-session commands obey the
/// admitted cadence. [`Self::stop_now`] uses terminal HostStop recovery and is
/// therefore the emergency/deadman/authority-loss path.
#[must_use = "candidate controller ownership must end in an inspected disarm result"]
pub struct WheelsOffCandidateActuationSession {
    inner: Option<PhysicalActuationSession>,
    target_authority: CandidateTargetAuthority,
    inventory_content_sha256: ManifestContentSha256,
    command_interval: Duration,
    last_applied: AppliedCommandReceipt,
    next_command_not_before: Instant,
}

impl WheelsOffCandidateActuationSession {
    pub fn acquire(
        authority: AdmittedWheelsOffCandidateController,
        clock_origin: Instant,
    ) -> Result<Self, CandidateActuationSessionStartError> {
        let target_authority = authority.target_authority();
        let inventory_content_sha256 = authority.inventory_content_sha256;
        let command_interval = authority.command_interval;
        let (inner, last_applied) =
            PhysicalActuationSession::acquire_candidate(authority.client, clock_origin)
                .map_err(CandidateActuationSessionStartError::Actuation)?;
        Self::from_acquired(
            inner,
            last_applied,
            target_authority,
            inventory_content_sha256,
            command_interval,
        )
    }

    fn from_acquired(
        inner: PhysicalActuationSession,
        last_applied: AppliedCommandReceipt,
        target_authority: CandidateTargetAuthority,
        inventory_content_sha256: ManifestContentSha256,
        command_interval: Duration,
    ) -> Result<Self, CandidateActuationSessionStartError> {
        let acquired_at = Instant::now();
        let next_command_not_before =
            match candidate_command_deadline(acquired_at, command_interval) {
                Ok(deadline) => deadline,
                Err(()) => {
                    let stop = stop_after_cadence_overflow(
                        inner,
                        target_authority,
                        inventory_content_sha256,
                        command_interval,
                    );
                    return Err(
                        CandidateActuationSessionStartError::CadenceDeadlineOverflow { stop },
                    );
                }
            };
        Ok(Self {
            inner: Some(inner),
            target_authority,
            inventory_content_sha256,
            command_interval,
            last_applied,
            next_command_not_before,
        })
    }

    pub const fn inventory_content_sha256(&self) -> ManifestContentSha256 {
        self.inventory_content_sha256
    }

    pub const fn last_applied(&self) -> &AppliedCommandReceipt {
        &self.last_applied
    }

    /// Retain the exact identity and capabilities observed by the zero
    /// acquisition which created this live candidate session.
    pub fn verified_controller_acquisition(
        &self,
    ) -> Result<VerifiedControllerAcquisition, LiveActuationError> {
        self.inner
            .as_ref()
            .ok_or(LiveActuationError::SessionConsumed)?
            .verified_controller_acquisition()
    }

    pub fn admit_target(
        &self,
        requested: CandidatePwmRequest,
        attestation: Option<&OperatorClaimedWheelsOffAttestation>,
    ) -> Result<AdmittedCandidatePwmTarget, CandidatePwmAdmissionError> {
        self.target_authority
            .admit_target(requested, attestation, Instant::now())
    }

    /// Apply one receipt-derived ramp step toward an already admitted target.
    pub fn apply_next_step(
        &mut self,
        target: AdmittedCandidatePwmTarget,
    ) -> Result<&AppliedCommandReceipt, CandidateActuationSessionError> {
        let now = Instant::now();
        if now < self.next_command_not_before {
            return Err(CandidateActuationSessionError::CommandCadenceNotElapsed {
                remaining: self.next_command_not_before.duration_since(now),
                command_interval: self.command_interval,
            });
        }
        if target.authority != self.target_authority {
            return Err(CandidateActuationSessionError::Admission(
                CandidatePwmAdmissionError::TargetAuthorityMismatch,
            ));
        }
        let step = target
            .next_step_from(&self.last_applied, now)
            .map_err(CandidateActuationSessionError::Admission)?;
        let receipt = self
            .inner
            .as_mut()
            .ok_or_else(|| {
                CandidateActuationSessionError::Actuation(LiveActuationError::SessionConsumed)
            })?
            .apply_candidate_pwm(step)
            .map_err(CandidateActuationSessionError::Actuation)?;
        let completed_at = Instant::now();
        self.last_applied = receipt;
        self.next_command_not_before =
            match candidate_command_deadline(completed_at, self.command_interval) {
                Ok(deadline) => deadline,
                Err(()) => {
                    let inner = self
                        .inner
                        .take()
                        .expect("the successful application retained the physical session");
                    let stop = stop_after_cadence_overflow(
                        inner,
                        self.target_authority,
                        self.inventory_content_sha256,
                        self.command_interval,
                    );
                    return Err(CandidateActuationSessionError::CadenceDeadlineOverflow { stop });
                }
            };
        Ok(&self.last_applied)
    }

    /// Consume the armed session and return one linear, exactly stopped owner.
    ///
    /// Unlike a keep-session zero HostCommand, this is not subject to command
    /// cadence. Reacquisition is possible only by consuming the returned
    /// [`StoppedWheelsOffCandidateController`]; admission itself is never
    /// cloned or reconstructed.
    pub fn stop_now(
        self,
    ) -> Result<(StoppedWheelsOffCandidateController, DisarmReceipt), LiveActuationError> {
        let (controller, _last_applied, receipt) = self.stop_now_with_last_applied()?;
        Ok((controller, receipt))
    }

    /// Consume the armed session while retaining its final applied-command
    /// receipt alongside the exact terminal-stop acknowledgement.
    ///
    /// Qualification bootstrap uses this form to prove that its only command
    /// before handing out a stopped controller was the zero acquisition.
    pub fn stop_now_with_last_applied(
        mut self,
    ) -> Result<
        (
            StoppedWheelsOffCandidateController,
            AppliedCommandReceipt,
            DisarmReceipt,
        ),
        LiveActuationError,
    > {
        let inner = self
            .inner
            .take()
            .ok_or(LiveActuationError::SessionConsumed)?;
        let (inner, receipt) = inner.stop_candidate()?;
        Ok((
            StoppedWheelsOffCandidateController {
                inner,
                target_authority: self.target_authority,
                inventory_content_sha256: self.inventory_content_sha256,
                command_interval: self.command_interval,
            },
            self.last_applied,
            receipt,
        ))
    }
}

fn candidate_command_deadline(now: Instant, command_interval: Duration) -> Result<Instant, ()> {
    now.checked_add(command_interval).ok_or(())
}

fn stop_after_cadence_overflow(
    inner: PhysicalActuationSession,
    target_authority: CandidateTargetAuthority,
    inventory_content_sha256: ManifestContentSha256,
    command_interval: Duration,
) -> CandidateCadenceOverflowStop {
    match inner.stop_candidate() {
        Ok((inner, receipt)) => CandidateCadenceOverflowStop::Confirmed {
            controller: Box::new(StoppedWheelsOffCandidateController {
                inner,
                target_authority,
                inventory_content_sha256,
                command_interval,
            }),
            receipt,
        },
        Err(error) => CandidateCadenceOverflowStop::Uncertain(Box::new(error)),
    }
}

/// Linear controller owner retained after an exact HostStop acknowledgement.
///
/// The token has no `Clone` implementation. A zero reacquisition consumes it,
/// preserving a single owner across `armed -> stopped -> armed` transitions.
#[must_use = "a stopped candidate controller token is the sole reacquisition authority"]
pub struct StoppedWheelsOffCandidateController {
    inner: StoppedCandidateActuationClient,
    target_authority: CandidateTargetAuthority,
    inventory_content_sha256: ManifestContentSha256,
    command_interval: Duration,
}

impl StoppedWheelsOffCandidateController {
    pub const fn inventory_content_sha256(&self) -> ManifestContentSha256 {
        self.inventory_content_sha256
    }

    /// Recheck the weak physical-setup claim immediately before this stopped
    /// token is consumed into a runtime which can reacquire output authority.
    pub fn require_fresh_attestation(
        &self,
        attestation: &OperatorClaimedWheelsOffAttestation,
        now: Instant,
    ) -> Result<(), CandidatePwmAdmissionError> {
        (*attestation).require_fresh(now, self.target_authority.maximum_attestation_age)
    }

    pub fn reacquire_zero(
        self,
    ) -> Result<WheelsOffCandidateActuationSession, CandidateActuationSessionStartError> {
        let Self {
            inner,
            target_authority,
            inventory_content_sha256,
            command_interval,
        } = self;
        let (inner, last_applied) = inner
            .reacquire_zero()
            .map_err(CandidateActuationSessionStartError::Actuation)?;
        WheelsOffCandidateActuationSession::from_acquired(
            inner,
            last_applied,
            target_authority,
            inventory_content_sha256,
            command_interval,
        )
    }
}

/// Stop evidence produced when an unrepresentable cadence deadline latches the
/// candidate session before another command can be sent.
pub enum CandidateCadenceOverflowStop {
    Confirmed {
        controller: Box<StoppedWheelsOffCandidateController>,
        receipt: DisarmReceipt,
    },
    Uncertain(Box<LiveActuationError>),
}

impl CandidateCadenceOverflowStop {
    pub fn into_confirmed(self) -> Option<(StoppedWheelsOffCandidateController, DisarmReceipt)> {
        match self {
            Self::Confirmed {
                controller,
                receipt,
            } => Some((*controller, receipt)),
            Self::Uncertain(_) => None,
        }
    }
}

impl fmt::Debug for CandidateCadenceOverflowStop {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, formatter)
    }
}

impl fmt::Display for CandidateCadenceOverflowStop {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Confirmed { receipt, .. } => write!(
                formatter,
                "controller stop confirmed at {} ns",
                receipt.acknowledged_at().nanos_since_clock_start()
            ),
            Self::Uncertain(error) => write!(formatter, "controller stop uncertain: {error}"),
        }
    }
}

pub enum CandidateActuationSessionStartError {
    Actuation(LiveActuationError),
    CadenceDeadlineOverflow { stop: CandidateCadenceOverflowStop },
}

impl fmt::Debug for CandidateActuationSessionStartError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, formatter)
    }
}

impl fmt::Display for CandidateActuationSessionStartError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Actuation(source) => write!(formatter, "{source}"),
            Self::CadenceDeadlineOverflow { stop } => write!(
                formatter,
                "candidate cadence deadline overflowed after zero acquisition; {stop}"
            ),
        }
    }
}

impl std::error::Error for CandidateActuationSessionStartError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Actuation(source) => Some(source),
            Self::CadenceDeadlineOverflow {
                stop: CandidateCadenceOverflowStop::Uncertain(source),
            } => Some(source.as_ref()),
            Self::CadenceDeadlineOverflow {
                stop: CandidateCadenceOverflowStop::Confirmed { .. },
            } => None,
        }
    }
}

fn bind_candidate_mpc_slew(
    wheel: WheelSide,
    configured_percent_per_step: u16,
    firmware_maximum_percent_per_step: u8,
) -> Result<(), CandidateMpcBindingError> {
    if configured_percent_per_step > u16::from(firmware_maximum_percent_per_step) {
        return Err(CandidateMpcBindingError::SlewExceedsFirmwareStep {
            wheel,
            configured_percent_per_step,
            firmware_maximum_percent_per_step,
        });
    }
    Ok(())
}

/// Weak operator claim, timestamped in one monotonic process domain.
///
/// The type records what the operator asserted; software cannot observe wheel
/// removal, external head support, or reachability of the physical power cut.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OperatorClaimedWheelsOffAttestation {
    issued_at: Instant,
}

impl OperatorClaimedWheelsOffAttestation {
    pub fn try_new(
        wheels_removed: bool,
        head_supported: bool,
        power_cut_reachable: bool,
        issued_at: Instant,
    ) -> Result<Self, WheelsOffCandidateAttestationError> {
        if !wheels_removed {
            return Err(WheelsOffCandidateAttestationError::WheelsRemovedNotClaimed);
        }
        if !head_supported {
            return Err(WheelsOffCandidateAttestationError::HeadSupportedNotClaimed);
        }
        if !power_cut_reachable {
            return Err(WheelsOffCandidateAttestationError::PowerCutReachableNotClaimed);
        }
        Ok(Self { issued_at })
    }

    fn require_fresh(
        self,
        now: Instant,
        maximum_age: Duration,
    ) -> Result<(), CandidatePwmAdmissionError> {
        let age = now
            .checked_duration_since(self.issued_at)
            .ok_or(CandidatePwmAdmissionError::AttestationClockRegressed)?;
        if age > maximum_age {
            return Err(CandidatePwmAdmissionError::WheelsOffAttestationExpired {
                age,
                maximum_age,
            });
        }
        Ok(())
    }
}

/// One parsed candidate PWM request. Construction enforces the KRP2 PWM
/// representation but does not yet grant permission to transmit it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CandidatePwmRequest(TimerPwm);

impl CandidatePwmRequest {
    pub fn try_new(left_percent: i8, right_percent: i8) -> Result<Self, DomainError> {
        TimerPwm::try_new(left_percent, right_percent).map(Self)
    }

    pub const fn stop() -> Self {
        Self(TimerPwm::ZERO)
    }

    pub const fn timer_pwm(self) -> TimerPwm {
        self.0
    }
}

/// A request proven to fit the exact local cap and the candidate firmware's
/// command-to-command step invariant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdmittedCandidatePwm {
    timer_pwm: TimerPwm,
}

/// Cap-proven desired PWM plus the exact candidate identity and weak
/// attestation needed to derive bounded steps from controller receipts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdmittedCandidatePwmTarget {
    timer_pwm: TimerPwm,
    authority: CandidateTargetAuthority,
    attestation: Option<OperatorClaimedWheelsOffAttestation>,
}

impl AdmittedCandidatePwmTarget {
    pub const fn timer_pwm(self) -> TimerPwm {
        self.timer_pwm
    }

    /// Recheck the retained weak setup claim without reconstructing or
    /// readmitting the already cap-proven target. Zero remains claim-free.
    pub fn require_fresh(self, now: Instant) -> Result<(), CandidatePwmAdmissionError> {
        match self.attestation {
            Some(attestation) => {
                attestation.require_fresh(now, self.authority.maximum_attestation_age)
            }
            None if self.timer_pwm.is_zero() => Ok(()),
            None => Err(CandidatePwmAdmissionError::FreshWheelsOffAttestationRequired),
        }
    }

    /// Derive one firmware-admissible step from the exact last applied
    /// receipt. Per-wheel reversals first target zero; a full stop is
    /// immediate. This creates a HostCommand value only, so server cadence
    /// still applies.
    pub fn next_step_from(
        self,
        previous_applied: &AppliedCommandReceipt,
        now: Instant,
    ) -> Result<AdmittedCandidatePwm, CandidatePwmAdmissionError> {
        if previous_applied.controller_session().controller_uid() != self.authority.controller_uid {
            return Err(CandidatePwmAdmissionError::PreviousReceiptControllerMismatch);
        }
        if let Some(attestation) = self.attestation {
            attestation.require_fresh(now, self.authority.maximum_attestation_age)?;
        }
        if self.timer_pwm.is_zero() {
            return Ok(AdmittedCandidatePwm {
                timer_pwm: TimerPwm::ZERO,
            });
        }
        let previous = previous_applied.applied_timer_pwm();
        let left = next_candidate_wheel_step(
            previous.left().get(),
            self.timer_pwm.left().get(),
            self.authority.maximum_command_step_percent,
        );
        let right = next_candidate_wheel_step(
            previous.right().get(),
            self.timer_pwm.right().get(),
            self.authority.maximum_command_step_percent,
        );
        let timer_pwm = TimerPwm::try_new(left, right)
            .expect("steps between already parsed PWM values remain in the PWM domain");
        Ok(AdmittedCandidatePwm { timer_pwm })
    }
}

fn next_candidate_wheel_step(previous: i8, target: i8, maximum_step_percent: u8) -> i8 {
    if previous != 0 && target != 0 && previous.signum() != target.signum() {
        return 0;
    }
    let delta = i16::from(target) - i16::from(previous);
    let maximum_step = i16::from(maximum_step_percent);
    let bounded_delta = delta.clamp(-maximum_step, maximum_step);
    i8::try_from(i16::from(previous) + bounded_delta)
        .expect("a step toward an i8 target remains in the i8 domain")
}

impl AdmittedCandidatePwm {
    pub const fn timer_pwm(self) -> TimerPwm {
        self.timer_pwm
    }
}

#[cfg(test)]
fn admit_pwm_transition(
    previous: TimerPwm,
    requested: TimerPwm,
    maximum_abs_percent: u8,
    maximum_step_percent: u8,
) -> Result<AdmittedCandidatePwm, CandidatePwmAdmissionError> {
    if requested.is_zero() {
        return Ok(AdmittedCandidatePwm {
            timer_pwm: requested,
        });
    }
    for (wheel, previous, requested) in [
        ("left", previous.left().get(), requested.left().get()),
        ("right", previous.right().get(), requested.right().get()),
    ] {
        let absolute = requested.unsigned_abs();
        if absolute > maximum_abs_percent {
            return Err(CandidatePwmAdmissionError::OutsideEffectiveCap {
                wheel,
                requested_percent: requested,
                maximum_abs_percent,
            });
        }
        if previous != 0 && requested != 0 && previous.signum() != requested.signum() {
            return Err(
                CandidatePwmAdmissionError::SignChangeRequiresIntermediateZero {
                    wheel,
                    previous_percent: previous,
                    requested_percent: requested,
                },
            );
        }
        let delta = u16::try_from((i16::from(requested) - i16::from(previous)).abs())
            .expect("the absolute difference of two i8 PWM values fits u16");
        if delta > u16::from(maximum_step_percent) {
            return Err(CandidatePwmAdmissionError::StepTooLarge {
                wheel,
                previous_percent: previous,
                requested_percent: requested,
                delta_percent: delta,
                maximum_step_percent,
            });
        }
    }
    Ok(AdmittedCandidatePwm {
        timer_pwm: requested,
    })
}

#[derive(Debug)]
pub enum WheelsOffCandidatePolicyError {
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
    ClientConfig(ClientConfigError),
    ProtocolDomain(DomainError),
    ZeroDuration {
        field: &'static str,
    },
    DurationAboveMaximum {
        field: &'static str,
        actual: Duration,
        maximum: Duration,
    },
    LocalPwmCapOutOfRange {
        actual: u8,
        minimum: u8,
        maximum: u8,
    },
    ManualTestMagnitudeOutOfRange {
        actual: u8,
        minimum: u8,
        local_maximum: u8,
    },
}

impl fmt::Display for WheelsOffCandidatePolicyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid wheels-off candidate policy: {self:?}")
    }
}

impl std::error::Error for WheelsOffCandidatePolicyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::JsonDecode(source) | Self::JsonTrailingData(source) => Some(source),
            Self::ClientConfig(source) => Some(source),
            Self::ProtocolDomain(source) => Some(source),
            Self::InputTooLarge { .. }
            | Self::UnsupportedSchemaVersion { .. }
            | Self::ZeroDuration { .. }
            | Self::DurationAboveMaximum { .. }
            | Self::LocalPwmCapOutOfRange { .. }
            | Self::ManualTestMagnitudeOutOfRange { .. } => None,
        }
    }
}

#[derive(Debug)]
pub enum WheelsOffCandidateControllerBindingError {
    ControllerSessionClassMismatch,
    PhysicalStopSemanticsMismatch,
    ServerCommandBind(ClientConfigError),
    ClientConfig(ClientConfigError),
    CommandEndpointMismatch {
        policy: SocketAddr,
        server: SocketAddr,
    },
    InventoryEndpointScheme {
        actual: ControlEndpointTransport,
    },
    InventoryEndpointMismatch {
        inventory: SocketAddr,
        server: SocketAddr,
    },
    SerialDeviceMismatch {
        inventory: Box<str>,
        server: std::path::PathBuf,
    },
    ControllerUidMismatch,
    FirmwareAbiMismatch {
        inventory: u16,
        server: u16,
    },
    FirmwareBuildMismatch {
        inventory: u32,
        server: u32,
    },
    ActuatorFingerprintMismatch,
    ControllerPwmCapMismatch {
        inventory: u8,
        server: u8,
    },
    ManualTestMagnitudeAboveEffectiveCap {
        requested_percent: u8,
        effective_max_abs_percent: u8,
    },
    DurationArithmeticOverflow,
    ClientTimeoutBelowServerBudget {
        field: &'static str,
        timeout: Duration,
        serial_ack: Duration,
        scheduling_margin: Duration,
    },
    ClientTimeoutReachesHeartbeatAge {
        field: &'static str,
        timeout: Duration,
        heartbeat_age: Duration,
    },
    CommandIntervalHasNoControllerMargin {
        command_interval: Duration,
        controller_minimum: Duration,
        scheduling_margin: Duration,
        required_exclusive_lower_bound: Duration,
    },
    CommandLeaseCannotBridgeNextApplication {
        lease: Duration,
        required_exclusive_lower_bound: Duration,
    },
    RuntimeServiceEnvelopeAboveQualificationMaximum {
        actual: Duration,
        maximum: Duration,
    },
}

impl fmt::Display for WheelsOffCandidateControllerBindingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "wheels-off candidate controller binding failed: {self:?}"
        )
    }
}

impl std::error::Error for WheelsOffCandidateControllerBindingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ServerCommandBind(source) | Self::ClientConfig(source) => Some(source),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WheelsOffCandidateAttestationError {
    WheelsRemovedNotClaimed,
    HeadSupportedNotClaimed,
    PowerCutReachableNotClaimed,
}

impl fmt::Display for WheelsOffCandidateAttestationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid wheels-off candidate operator attestation: {self:?}"
        )
    }
}

impl std::error::Error for WheelsOffCandidateAttestationError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CandidatePwmAdmissionError {
    PreviousReceiptControllerMismatch,
    TargetAuthorityMismatch,
    FreshWheelsOffAttestationRequired,
    AttestationClockRegressed,
    WheelsOffAttestationExpired {
        age: Duration,
        maximum_age: Duration,
    },
    OutsideEffectiveCap {
        wheel: &'static str,
        requested_percent: i8,
        maximum_abs_percent: u8,
    },
    SignChangeRequiresIntermediateZero {
        wheel: &'static str,
        previous_percent: i8,
        requested_percent: i8,
    },
    StepTooLarge {
        wheel: &'static str,
        previous_percent: i8,
        requested_percent: i8,
        delta_percent: u16,
        maximum_step_percent: u8,
    },
}

impl fmt::Display for CandidatePwmAdmissionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "candidate PWM request rejected: {self:?}")
    }
}

impl std::error::Error for CandidatePwmAdmissionError {}

pub enum CandidateActuationSessionError {
    CommandCadenceNotElapsed {
        remaining: Duration,
        command_interval: Duration,
    },
    CadenceDeadlineOverflow {
        stop: CandidateCadenceOverflowStop,
    },
    Admission(CandidatePwmAdmissionError),
    Actuation(LiveActuationError),
}

impl fmt::Debug for CandidateActuationSessionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, formatter)
    }
}

impl fmt::Display for CandidateActuationSessionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CommandCadenceNotElapsed {
                remaining,
                command_interval,
            } => write!(
                formatter,
                "candidate keep-session command arrived {remaining:?} before the admitted {command_interval:?} cadence elapsed"
            ),
            Self::CadenceDeadlineOverflow { stop } => write!(
                formatter,
                "candidate cadence deadline overflowed after an exact application; {stop}"
            ),
            Self::Admission(source) => write!(formatter, "{source}"),
            Self::Actuation(source) => write!(formatter, "{source}"),
        }
    }
}

impl std::error::Error for CandidateActuationSessionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Admission(source) => Some(source),
            Self::Actuation(source) => Some(source),
            Self::CadenceDeadlineOverflow {
                stop: CandidateCadenceOverflowStop::Uncertain(source),
            } => Some(source.as_ref()),
            Self::CommandCadenceNotElapsed { .. }
            | Self::CadenceDeadlineOverflow {
                stop: CandidateCadenceOverflowStop::Confirmed { .. },
            } => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CandidateMpcBindingError {
    PwmOutsideEffectiveCap {
        wheel: WheelSide,
        minimum: i8,
        maximum: i8,
        effective_max_abs_percent: u8,
    },
    SlewExceedsFirmwareStep {
        wheel: WheelSide,
        configured_percent_per_step: u16,
        firmware_maximum_percent_per_step: u8,
    },
}

impl fmt::Display for CandidateMpcBindingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "candidate shadow MPC is incompatible with firmware: {self:?}"
        )
    }
}

impl std::error::Error for CandidateMpcBindingError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CandidateRuntimeServiceIntervalError {
    Zero,
    ExceedsMaximum {
        actual: Duration,
        maximum: Duration,
    },
    RuntimeMismatch {
        admitted: Duration,
        actual: Duration,
    },
}

impl fmt::Display for CandidateRuntimeServiceIntervalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "candidate qualification runtime service interval rejected: {self:?}"
        )
    }
}

impl std::error::Error for CandidateRuntimeServiceIntervalError {}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WheelsOffCandidatePolicyV1Dto {
    schema_version: u32,
    command_endpoint: String,
    status_timeout_ns: u64,
    acquire_timeout_ns: u64,
    applied_ack_timeout_ns: u64,
    stop_attempt_timeout_ns: u64,
    maximum_stop_recovery_attempts: u8,
    command_lease_ms: u16,
    command_interval_ns: u64,
    scheduling_margin_ns: u64,
    local_max_abs_pwm_percent: u8,
    manual_test_magnitude_timer_pwm_percent: u8,
    manual_deadman_ms: u16,
    maximum_attestation_age_ns: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiko_device_inventory::load_expected_manifest_v2_from_slice;
    use robot_protocol::v2::{
        ControllerCapabilities, OPERATOR_SUPERVISED_FOUR_PWM_FINGERPRINT_BYTES,
        OPERATOR_SUPERVISED_FOUR_PWM_FIRMWARE_BUILD_ID,
        OPERATOR_SUPERVISED_FOUR_PWM_MAX_COMMAND_STEP_PERCENT,
    };
    use serde_json::json;

    fn policy() -> Vec<u8> {
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "command_endpoint": "127.0.0.1:8080",
            "status_timeout_ns": 40_000_000,
            "acquire_timeout_ns": 40_000_000,
            "applied_ack_timeout_ns": 40_000_000,
            "stop_attempt_timeout_ns": 40_000_000,
            "maximum_stop_recovery_attempts": robot_command_client::MAX_STOP_RECOVERY_ATTEMPTS,
            "command_lease_ms": 100,
            "command_interval_ns": 20_000_000,
            "scheduling_margin_ns": 5_000_000,
            "local_max_abs_pwm_percent": 20,
            "manual_test_magnitude_timer_pwm_percent": 10,
            "manual_deadman_ms": 150,
            "maximum_attestation_age_ns": 5_000_000_000_u64
        }))
        .expect("policy JSON")
    }

    fn inventory() -> LoadedExpectedManifestV2 {
        let capabilities = ControllerCapabilities::SOFTWARE_GUARD_BITS
            | ControllerCapabilities::OPERATOR_SUPERVISED_FOUR_PWM_CANDIDATE;
        let bytes = serde_json::to_vec(&json!({
            "schema_version": 2,
            "robot_id": "kiko-candidate-01",
            "oak": {
                "mxid": "A1B2C3D4E5F60708",
                "compiled_depthai_header_sdk_version": "3.6.1",
                "compiled_depthai_header_sdk_commit": "abc123",
                "compiled_depthai_header_embedded_device_artifact_version": "device-1",
                "compiled_depthai_header_embedded_bootloader_artifact_version": "bootloader-1"
            },
            "stm32": {
                "serial_by_id_path": "/dev/serial/by-id/usb-STMicroelectronics_STM32_STLink_candidate-if02",
                "control_endpoint_identity": "udp://127.0.0.1:8080",
                "controller_uid": [0, 17, 34, 51, 68, 85, 102, 119, 136, 153, 170, 187],
                "firmware_abi": 2,
                "firmware_build_id": OPERATOR_SUPERVISED_FOUR_PWM_FIRMWARE_BUILD_ID,
                "hardware_profile_fingerprint": OPERATOR_SUPERVISED_FOUR_PWM_FINGERPRINT_BYTES,
                "capabilities_bits": capabilities,
                "controller_session_class": "operator_supervised_four_pwm_candidate",
                "expected_max_abs_pwm_percent": 30,
                "expected_physical_stop_semantics": "unverified"
            },
            "head": null,
            "eye": null,
            "calibration_artifacts": [{
                "artifact_id": "camera",
                "sha256": vec![1_u8; 32]
            }],
            "plant_artifacts": [{
                "artifact_id": "plant",
                "sha256": vec![2_u8; 32]
            }]
        }))
        .expect("candidate inventory JSON");
        load_expected_manifest_v2_from_slice(&bytes).expect("candidate inventory")
    }

    fn server() -> ControllerServerConfigV2 {
        let bytes = serde_json::to_vec(&json!({
            "schema_version": 2,
            "serial_device": "/dev/serial/by-id/usb-STMicroelectronics_STM32_STLink_candidate-if02",
            "controller_uid_hex": "00112233445566778899aabb",
            "firmware_abi": 2,
            "firmware_build_id": OPERATOR_SUPERVISED_FOUR_PWM_FIRMWARE_BUILD_ID,
            "actuator_config_fingerprint_hex": "4b494b4f2d3450574d2d43414e443121",
            "hardware_profile_claim_id": "kiko-four-pwm-candidate-wheels-off-v1",
            "controller_ready_timeout_ms": 3000,
            "heartbeat_period_ms": 20,
            "maximum_heartbeat_age_ms": 60,
            "maximum_host_command_rate_hz": 100,
            "serial_transmit_timeout_ms": 10,
            "serial_applied_ack_timeout_ms": 30,
            "controller_clock_abs_error_ppm_bound": 50_000,
            "deadline_quantization_margin_ms": 2,
            "expected_max_abs_pwm_percent": 30,
            "expected_pwm_frequency_hz": 20_000,
            "expected_watchdog_nominal_timeout_ms": 250,
            "expected_neutral_output": "both_low",
            "expected_physical_stop_semantics": "unverified",
            "controller_session_class": "operator_supervised_four_pwm_candidate"
        }))
        .expect("candidate server JSON");
        ControllerServerConfigV2::parse_json(&bytes).expect("candidate server")
    }

    #[test]
    fn strict_policy_parses_units_once() {
        let parsed =
            WheelsOffCandidateControllerBinding::parse_json(&policy()).expect("candidate policy");
        assert_eq!(parsed.command_interval, Duration::from_millis(20));
        assert_eq!(parsed.scheduling_margin, Duration::from_millis(5));
        assert_eq!(parsed.local_max_abs_pwm_percent, 20);
        assert_eq!(parsed.manual_test_magnitude_timer_pwm_percent, 10);
        assert_eq!(parsed.manual_deadman, Duration::from_millis(150));

        let mut trailing = policy();
        trailing.extend_from_slice(b" {}");
        assert!(matches!(
            WheelsOffCandidateControllerBinding::parse_json(&trailing),
            Err(WheelsOffCandidatePolicyError::JsonTrailingData(_))
        ));

        let mut excessive_attestation_age: serde_json::Value =
            serde_json::from_slice(&policy()).expect("policy fixture");
        excessive_attestation_age["maximum_attestation_age_ns"] =
            json!(MAX_WHEELS_OFF_CANDIDATE_ATTESTATION_AGE.as_nanos() + 1);
        let excessive_attestation_age =
            serde_json::to_vec(&excessive_attestation_age).expect("policy JSON");
        assert!(matches!(
            WheelsOffCandidateControllerBinding::parse_json(&excessive_attestation_age),
            Err(WheelsOffCandidatePolicyError::DurationAboveMaximum {
                field: "maximum_attestation_age_ns",
                maximum: MAX_WHEELS_OFF_CANDIDATE_ATTESTATION_AGE,
                ..
            })
        ));

        let mut excessive_manual: serde_json::Value =
            serde_json::from_slice(&policy()).expect("policy fixture");
        excessive_manual["manual_test_magnitude_timer_pwm_percent"] = json!(21);
        assert!(matches!(
            WheelsOffCandidateControllerBinding::parse_json(
                &serde_json::to_vec(&excessive_manual).expect("policy JSON")
            ),
            Err(
                WheelsOffCandidatePolicyError::ManualTestMagnitudeOutOfRange {
                    actual: 21,
                    local_maximum: 20,
                    ..
                }
            )
        ));

        let mut excessive_deadman: serde_json::Value =
            serde_json::from_slice(&policy()).expect("policy fixture");
        excessive_deadman["manual_deadman_ms"] = json!(1_001);
        assert!(matches!(
            WheelsOffCandidateControllerBinding::parse_json(
                &serde_json::to_vec(&excessive_deadman).expect("policy JSON")
            ),
            Err(WheelsOffCandidatePolicyError::DurationAboveMaximum {
                field: "manual_deadman_ms",
                maximum: MAX_WHEELS_OFF_CANDIDATE_MANUAL_DEADMAN,
                ..
            })
        ));
    }

    #[test]
    fn cadence_deadline_overflow_has_no_now_fallback() {
        assert!(
            candidate_command_deadline(Instant::now(), Duration::MAX).is_err(),
            "an unrepresentable cadence deadline must latch instead of becoming immediately due"
        );
    }

    #[test]
    fn v2_inventory_server_and_client_session_bind_without_a_production_path() {
        let admitted = WheelsOffCandidateControllerBinding::parse_json(&policy())
            .expect("candidate policy")
            .admit(
                &inventory(),
                &server(),
                "127.0.0.1:8080".parse().expect("loopback endpoint"),
            )
            .expect("candidate-only binding");
        assert_eq!(
            admitted.controller_session_class(),
            ControllerSessionClass::OperatorSupervisedFourPwmCandidate
        );
        assert_eq!(admitted.effective_max_abs_pwm_percent(), 20);
        assert_eq!(
            admitted.maximum_command_step_percent(),
            OPERATOR_SUPERVISED_FOUR_PWM_MAX_COMMAND_STEP_PERCENT
        );
        assert_eq!(admitted.command_interval(), Duration::from_millis(20));
        assert_eq!(admitted.manual_test_magnitude_timer_pwm_percent(), 10);
        assert_eq!(admitted.manual_deadman(), Duration::from_millis(150));
        assert_eq!(
            admitted.limits().manual_test_magnitude_timer_pwm_percent(),
            10
        );
        assert_eq!(
            admitted.limits().manual_deadman(),
            Duration::from_millis(150)
        );
        let limits = admitted.limits();
        assert_eq!(
            limits.maximum_runtime_service_interval(),
            Duration::from_nanos(54_999_999)
        );
        let service = limits
            .admit_runtime_service_interval(Duration::from_millis(50))
            .expect("50 ms fits the strict lease and deadman budgets");
        assert_eq!(service.actual(), Duration::from_millis(50));
        service
            .require_exact_runtime_interval(Duration::from_millis(50))
            .expect("the runtime uses the admitted shadow period");
        assert!(matches!(
            limits.admit_runtime_service_interval(Duration::from_millis(55)),
            Err(CandidateRuntimeServiceIntervalError::ExceedsMaximum {
                actual,
                maximum,
            }) if actual == Duration::from_millis(55)
                && maximum == Duration::from_nanos(54_999_999)
        ));
        assert!(matches!(
            service.require_exact_runtime_interval(Duration::from_millis(40)),
            Err(CandidateRuntimeServiceIntervalError::RuntimeMismatch {
                admitted,
                actual,
            }) if admitted == Duration::from_millis(50)
                && actual == Duration::from_millis(40)
        ));
    }

    #[test]
    fn candidate_policy_cannot_widen_the_fixed_qualification_service_envelope() {
        let mut wider: serde_json::Value =
            serde_json::from_slice(&policy()).expect("policy fixture");
        wider["command_lease_ms"] = json!(101);
        let parsed = WheelsOffCandidateControllerBinding::parse_json(
            &serde_json::to_vec(&wider).expect("wider policy fixture"),
        )
        .expect("individual policy fields remain valid");
        assert!(matches!(
            parsed.admit(
                &inventory(),
                &server(),
                "127.0.0.1:8080".parse().expect("loopback endpoint"),
            ),
            Err(
                WheelsOffCandidateControllerBindingError::RuntimeServiceEnvelopeAboveQualificationMaximum {
                    actual,
                    maximum,
                }
            ) if actual == Duration::from_nanos(55_999_999)
                && maximum == MAX_WHEELS_OFF_CANDIDATE_RUNTIME_SERVICE_INTERVAL
        ));
    }

    #[test]
    fn candidate_envelope_uses_previous_application_and_zero_always_bypasses() {
        let previous = TimerPwm::try_new(5, -5).expect("previous");
        let next = TimerPwm::try_new(10, -10).expect("next");
        assert_eq!(
            admit_pwm_transition(
                previous,
                next,
                20,
                OPERATOR_SUPERVISED_FOUR_PWM_MAX_COMMAND_STEP_PERCENT,
            )
            .expect("exact five-point transition")
            .timer_pwm(),
            next
        );
        assert!(matches!(
            admit_pwm_transition(
                previous,
                TimerPwm::try_new(11, -10).expect("six-point request"),
                20,
                OPERATOR_SUPERVISED_FOUR_PWM_MAX_COMMAND_STEP_PERCENT,
            ),
            Err(CandidatePwmAdmissionError::StepTooLarge {
                wheel: "left",
                delta_percent: 6,
                ..
            })
        ));
        assert_eq!(
            admit_pwm_transition(
                TimerPwm::try_new(20, -20).expect("full cap"),
                TimerPwm::ZERO,
                1,
                1,
            )
            .expect("zero is always available")
            .timer_pwm(),
            TimerPwm::ZERO
        );
        assert!(matches!(
            admit_pwm_transition(
                TimerPwm::try_new(2, 0).expect("positive left"),
                TimerPwm::try_new(-2, 0).expect("negative left"),
                20,
                OPERATOR_SUPERVISED_FOUR_PWM_MAX_COMMAND_STEP_PERCENT,
            ),
            Err(
                CandidatePwmAdmissionError::SignChangeRequiresIntermediateZero {
                    wheel: "left",
                    previous_percent: 2,
                    requested_percent: -2,
                }
            )
        ));
        assert_eq!(
            next_candidate_wheel_step(0, 20, OPERATOR_SUPERVISED_FOUR_PWM_MAX_COMMAND_STEP_PERCENT,),
            5
        );
        assert_eq!(
            next_candidate_wheel_step(
                2,
                -20,
                OPERATOR_SUPERVISED_FOUR_PWM_MAX_COMMAND_STEP_PERCENT,
            ),
            0,
            "a reversal must first command the individual wheel to zero"
        );
        assert_eq!(
            next_candidate_wheel_step(
                -5,
                -20,
                OPERATOR_SUPERVISED_FOUR_PWM_MAX_COMMAND_STEP_PERCENT,
            ),
            -10
        );
    }

    #[test]
    fn candidate_mpc_slew_uses_the_same_five_point_firmware_invariant() {
        assert!(
            bind_candidate_mpc_slew(
                WheelSide::Left,
                u16::from(OPERATOR_SUPERVISED_FOUR_PWM_MAX_COMMAND_STEP_PERCENT),
                OPERATOR_SUPERVISED_FOUR_PWM_MAX_COMMAND_STEP_PERCENT,
            )
            .is_ok()
        );
        assert!(matches!(
            bind_candidate_mpc_slew(
                WheelSide::Right,
                u16::from(OPERATOR_SUPERVISED_FOUR_PWM_MAX_COMMAND_STEP_PERCENT) + 1,
                OPERATOR_SUPERVISED_FOUR_PWM_MAX_COMMAND_STEP_PERCENT,
            ),
            Err(CandidateMpcBindingError::SlewExceedsFirmwareStep {
                wheel: WheelSide::Right,
                configured_percent_per_step: 6,
                firmware_maximum_percent_per_step: 5,
            })
        ));
    }

    #[test]
    fn attestation_is_explicit_monotonic_and_bounded() {
        let issued_at = Instant::now();
        let claim = OperatorClaimedWheelsOffAttestation::try_new(true, true, true, issued_at)
            .expect("explicit weak claims");
        assert!(
            claim
                .require_fresh(issued_at + Duration::from_secs(5), Duration::from_secs(5),)
                .is_ok()
        );
        assert!(matches!(
            claim.require_fresh(
                issued_at + Duration::from_secs(5) + Duration::from_nanos(1),
                Duration::from_secs(5),
            ),
            Err(CandidatePwmAdmissionError::WheelsOffAttestationExpired { .. })
        ));
        assert!(matches!(
            OperatorClaimedWheelsOffAttestation::try_new(false, true, true, issued_at),
            Err(WheelsOffCandidateAttestationError::WheelsRemovedNotClaimed)
        ));
        assert!(matches!(
            OperatorClaimedWheelsOffAttestation::try_new(true, false, true, issued_at),
            Err(WheelsOffCandidateAttestationError::HeadSupportedNotClaimed)
        ));
    }

    #[test]
    fn admitted_target_rechecks_its_retained_claim_without_reparsing() {
        let admitted = WheelsOffCandidateControllerBinding::parse_json(&policy())
            .expect("candidate policy")
            .admit(
                &inventory(),
                &server(),
                "127.0.0.1:8080".parse().expect("loopback endpoint"),
            )
            .expect("candidate-only binding");
        let issued_at = Instant::now();
        let claim = OperatorClaimedWheelsOffAttestation::try_new(true, true, true, issued_at)
            .expect("explicit weak claims");
        let target = admitted
            .admit_target(
                CandidatePwmRequest::try_new(10, 0).expect("manual test target"),
                Some(&claim),
                issued_at,
            )
            .expect("fresh admitted target");
        assert!(
            target
                .require_fresh(issued_at + Duration::from_secs(5))
                .is_ok()
        );
        assert!(matches!(
            target.require_fresh(issued_at + Duration::from_secs(5) + Duration::from_nanos(1)),
            Err(CandidatePwmAdmissionError::WheelsOffAttestationExpired { .. })
        ));
        assert!(
            admitted
                .admit_target(CandidatePwmRequest::stop(), None, issued_at)
                .expect("zero target")
                .require_fresh(issued_at + Duration::from_secs(31))
                .is_ok()
        );
    }
}
