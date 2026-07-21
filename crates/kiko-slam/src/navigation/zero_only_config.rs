//! Static authority for acquiring and retaining an applied base zero.
//!
//! This configuration deliberately carries no wheel target, plant model, MPC
//! setting, or motion approval. It is the only physical-controller authority
//! needed by the wheels-off bench runtime. Binding consumes one parsed policy
//! and derives controller identity exclusively from the exact loaded inventory
//! and the exact `robot-server` controller configuration.

use std::fmt;
use std::net::SocketAddr;
use std::path::Path;
use std::time::Duration;

use kiko_device_inventory::{
    ControlEndpointTransport, LoadedExpectedManifestV1, ManifestContentSha256,
};
use robot_command_client::{
    ClientConfig, ConfigError as ClientConfigError, StopRecoveryPolicy, TimeoutNs, UdpEndpoint,
};
use robot_protocol::v2::V2CommandLeaseMs;
use robot_server::config::ControllerServerConfigV1;
use serde::Deserialize;

pub const ZERO_ONLY_ACTUATION_POLICY_V1: u32 = 1;
pub const MAX_ZERO_ONLY_ACTUATION_POLICY_JSON_BYTES: usize = 4 * 1_024;

/// Parsed timing and local-transport policy with no controller identity yet.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZeroOnlyActuationPolicyV1 {
    endpoint: UdpEndpoint,
    status_timeout: TimeoutNs,
    acquire_timeout: TimeoutNs,
    applied_ack_timeout: TimeoutNs,
    stop_recovery: StopRecoveryPolicy,
    zero_acquisition_lease: V2CommandLeaseMs,
    server_dispatch_margin: TimeoutNs,
}

impl ZeroOnlyActuationPolicyV1 {
    pub fn parse_json(bytes: &[u8]) -> Result<Self, ZeroOnlyActuationPolicyError> {
        if bytes.len() > MAX_ZERO_ONLY_ACTUATION_POLICY_JSON_BYTES {
            return Err(ZeroOnlyActuationPolicyError::InputTooLarge {
                actual_bytes: bytes.len(),
                maximum_bytes: MAX_ZERO_ONLY_ACTUATION_POLICY_JSON_BYTES,
            });
        }
        let mut deserializer = serde_json::Deserializer::from_slice(bytes);
        let dto = ZeroOnlyActuationPolicyV1Dto::deserialize(&mut deserializer)
            .map_err(ZeroOnlyActuationPolicyError::JsonDecode)?;
        deserializer
            .end()
            .map_err(ZeroOnlyActuationPolicyError::JsonTrailingData)?;
        if dto.schema_version != ZERO_ONLY_ACTUATION_POLICY_V1 {
            return Err(ZeroOnlyActuationPolicyError::UnsupportedSchemaVersion {
                actual: dto.schema_version,
                supported: ZERO_ONLY_ACTUATION_POLICY_V1,
            });
        }

        let endpoint = dto
            .command_endpoint
            .parse()
            .map_err(ZeroOnlyActuationPolicyError::ClientConfig)?;
        let status_timeout = TimeoutNs::try_new(dto.status_timeout_ns)
            .map_err(ZeroOnlyActuationPolicyError::ClientConfig)?;
        let acquire_timeout = TimeoutNs::try_new(dto.acquire_timeout_ns)
            .map_err(ZeroOnlyActuationPolicyError::ClientConfig)?;
        let applied_ack_timeout = TimeoutNs::try_new(dto.applied_ack_timeout_ns)
            .map_err(ZeroOnlyActuationPolicyError::ClientConfig)?;
        let stop_attempt_timeout = TimeoutNs::try_new(dto.stop_attempt_timeout_ns)
            .map_err(ZeroOnlyActuationPolicyError::ClientConfig)?;
        let stop_recovery =
            StopRecoveryPolicy::try_new(dto.maximum_stop_recovery_attempts, stop_attempt_timeout)
                .map_err(ZeroOnlyActuationPolicyError::ClientConfig)?;
        let zero_acquisition_lease = V2CommandLeaseMs::try_new(dto.zero_acquisition_lease_ms)
            .map_err(ZeroOnlyActuationPolicyError::ProtocolDomain)?;
        let server_dispatch_margin = TimeoutNs::try_new(dto.server_dispatch_margin_ns)
            .map_err(ZeroOnlyActuationPolicyError::ClientConfig)?;
        let lease = Duration::from_millis(u64::from(zero_acquisition_lease.get()));
        let required_exclusive_lower_bound = applied_ack_timeout
            .as_duration()
            .checked_mul(2)
            .ok_or(ZeroOnlyActuationPolicyError::DurationArithmeticOverflow)?;
        if lease <= required_exclusive_lower_bound {
            return Err(
                ZeroOnlyActuationPolicyError::ZeroLeaseCannotCoverProactiveRenewal {
                    lease,
                    applied_ack_timeout: applied_ack_timeout.as_duration(),
                    required_exclusive_lower_bound,
                },
            );
        }

        Ok(Self {
            endpoint,
            status_timeout,
            acquire_timeout,
            applied_ack_timeout,
            stop_recovery,
            zero_acquisition_lease,
            server_dispatch_margin,
        })
    }

    /// Bind every duplicated identity and endpoint before a UDP socket opens.
    pub fn bind(
        self,
        inventory: &LoadedExpectedManifestV1,
        server: &ControllerServerConfigV1,
        server_command_bind: SocketAddr,
    ) -> Result<BoundZeroOnlyActuationConfigV1, ZeroOnlyActuationBindingError> {
        let server_endpoint = UdpEndpoint::try_new(server_command_bind)
            .map_err(ZeroOnlyActuationBindingError::ServerCommandBind)?;
        if self.endpoint != server_endpoint {
            return Err(ZeroOnlyActuationBindingError::CommandEndpointMismatch {
                policy: self.endpoint.socket_addr(),
                server: server_command_bind,
            });
        }

        let expected = inventory.manifest().stm32();
        let inventory_endpoint = *expected.control_endpoint();
        if inventory_endpoint.transport() != ControlEndpointTransport::Udp {
            return Err(ZeroOnlyActuationBindingError::InventoryEndpointScheme {
                actual: inventory_endpoint.transport(),
            });
        }
        let inventory_socket = inventory_endpoint
            .socket_addr()
            .expect("parsed TCP/UDP control endpoints retain a socket address");
        if inventory_socket != server_command_bind {
            return Err(ZeroOnlyActuationBindingError::InventoryEndpointMismatch {
                inventory: inventory_socket,
                server: server_command_bind,
            });
        }
        if server.serial_device() != Path::new(expected.serial_path().as_str()) {
            return Err(ZeroOnlyActuationBindingError::SerialDeviceMismatch {
                inventory: expected.serial_path().as_str().into(),
                server: server.serial_device().to_path_buf(),
            });
        }
        if server.controller_uid() != *expected.controller_uid() {
            return Err(ZeroOnlyActuationBindingError::ControllerUidMismatch);
        }
        if server.firmware_abi().get() != expected.firmware_abi() {
            return Err(ZeroOnlyActuationBindingError::FirmwareAbiMismatch {
                inventory: expected.firmware_abi(),
                server: server.firmware_abi().get(),
            });
        }
        if server.firmware_build_id().get() != expected.firmware_build_id() {
            return Err(ZeroOnlyActuationBindingError::FirmwareBuildMismatch {
                inventory: expected.firmware_build_id(),
                server: server.firmware_build_id().get(),
            });
        }
        if server.actuator_config_fingerprint() != *expected.hardware_profile() {
            return Err(ZeroOnlyActuationBindingError::ActuatorFingerprintMismatch);
        }

        let serial_ack = server.serial_applied_ack_timeout();
        let required_timeout = serial_ack
            .checked_add(self.server_dispatch_margin.as_duration())
            .ok_or(ZeroOnlyActuationBindingError::TimeoutArithmeticOverflow)?;
        let heartbeat_age = server.maximum_heartbeat_age();
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
                    ZeroOnlyActuationBindingError::ClientTimeoutBelowServerBudget {
                        field,
                        timeout: timeout.as_duration(),
                        serial_ack,
                        dispatch_margin: self.server_dispatch_margin.as_duration(),
                    },
                );
            }
            if timeout.as_duration() >= heartbeat_age {
                return Err(
                    ZeroOnlyActuationBindingError::ClientTimeoutReachesHeartbeatAge {
                        field,
                        timeout: timeout.as_duration(),
                        heartbeat_age,
                    },
                );
            }
        }

        let client = ClientConfig::new(
            self.endpoint,
            server.controller_uid(),
            server.firmware_abi(),
            server.firmware_build_id(),
            server.actuator_config_fingerprint(),
            self.status_timeout,
            self.acquire_timeout,
            self.applied_ack_timeout,
            self.stop_recovery,
            self.zero_acquisition_lease,
        );
        Ok(BoundZeroOnlyActuationConfigV1 {
            client,
            inventory_content_sha256: inventory.content_sha256(),
        })
    }
}

/// Zero-only client configuration bound to one exact loaded inventory file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoundZeroOnlyActuationConfigV1 {
    client: ClientConfig,
    inventory_content_sha256: ManifestContentSha256,
}

impl BoundZeroOnlyActuationConfigV1 {
    pub const fn client(&self) -> &ClientConfig {
        &self.client
    }

    pub const fn inventory_content_sha256(&self) -> ManifestContentSha256 {
        self.inventory_content_sha256
    }

    pub fn into_client(self) -> ClientConfig {
        self.client
    }
}

#[derive(Debug)]
pub enum ZeroOnlyActuationPolicyError {
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
    ProtocolDomain(robot_protocol::v2::DomainError),
    DurationArithmeticOverflow,
    ZeroLeaseCannotCoverProactiveRenewal {
        lease: Duration,
        applied_ack_timeout: Duration,
        required_exclusive_lower_bound: Duration,
    },
}

impl fmt::Display for ZeroOnlyActuationPolicyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid zero-only actuation policy: {self:?}")
    }
}

impl std::error::Error for ZeroOnlyActuationPolicyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::JsonDecode(source) | Self::JsonTrailingData(source) => Some(source),
            Self::ClientConfig(source) => Some(source),
            Self::ProtocolDomain(source) => Some(source),
            Self::InputTooLarge { .. }
            | Self::UnsupportedSchemaVersion { .. }
            | Self::DurationArithmeticOverflow
            | Self::ZeroLeaseCannotCoverProactiveRenewal { .. } => None,
        }
    }
}

#[derive(Debug)]
pub enum ZeroOnlyActuationBindingError {
    ServerCommandBind(ClientConfigError),
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
    TimeoutArithmeticOverflow,
    ClientTimeoutBelowServerBudget {
        field: &'static str,
        timeout: Duration,
        serial_ack: Duration,
        dispatch_margin: Duration,
    },
    ClientTimeoutReachesHeartbeatAge {
        field: &'static str,
        timeout: Duration,
        heartbeat_age: Duration,
    },
}

impl fmt::Display for ZeroOnlyActuationBindingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "zero-only actuation binding failed: {self:?}")
    }
}

impl std::error::Error for ZeroOnlyActuationBindingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ServerCommandBind(source) => Some(source),
            _ => None,
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ZeroOnlyActuationPolicyV1Dto {
    schema_version: u32,
    command_endpoint: String,
    status_timeout_ns: u64,
    acquire_timeout_ns: u64,
    applied_ack_timeout_ns: u64,
    stop_attempt_timeout_ns: u64,
    maximum_stop_recovery_attempts: u8,
    zero_acquisition_lease_ms: u16,
    server_dispatch_margin_ns: u64,
}

#[cfg(test)]
mod tests {
    use kiko_device_inventory::load_expected_manifest_v1_from_slice;
    use robot_protocol::v2::{ControllerCapabilities, VERSION as ROBOT_PROTOCOL_VERSION};
    use serde_json::{Value, json};

    use super::*;

    fn policy() -> Value {
        json!({
            "schema_version": 1,
            "command_endpoint": "127.0.0.1:8080",
            "status_timeout_ns": 40_000_000,
            "acquire_timeout_ns": 40_000_000,
            "applied_ack_timeout_ns": 40_000_000,
            "stop_attempt_timeout_ns": 40_000_000,
            "maximum_stop_recovery_attempts": 3,
            "zero_acquisition_lease_ms": 100,
            "server_dispatch_margin_ns": 5_000_000
        })
    }

    fn inventory(endpoint: &str) -> LoadedExpectedManifestV1 {
        let value = json!({
            "schema_version": 1,
            "robot_id": "kiko-production-01",
            "oak": {
                "mxid": "A1B2C3D4E5F60708",
                "linked_depthai_sdk_version": "3.6.1",
                "linked_depthai_sdk_commit": "abc123",
                "linked_depthai_embedded_device_artifact_version": "device-1",
                "linked_depthai_embedded_bootloader_artifact_version": "bootloader-1"
            },
            "stm32": {
                "serial_by_id_path": "/dev/serial/by-id/usb-Kiko_STM32_A1-if00",
                "control_endpoint_identity": endpoint,
                "controller_uid": [0, 17, 34, 51, 68, 85, 102, 119, 136, 153, 170, 187],
                "firmware_abi": ROBOT_PROTOCOL_VERSION,
                "firmware_build_id": 42,
                "hardware_profile_fingerprint": [17, 34, 51, 68, 85, 102, 119, 136, 153, 170, 187, 204, 221, 238, 255, 0],
                "capabilities_bits": ControllerCapabilities::REQUIRED_BITS
            },
            "head": null,
            "eye": null,
            "calibration_artifacts": [{"artifact_id": "camera", "sha256": [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]}],
            "plant_artifacts": [{"artifact_id": "plant", "sha256": [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]}]
        });
        load_expected_manifest_v1_from_slice(&serde_json::to_vec(&value).expect("manifest JSON"))
            .expect("valid inventory")
    }

    fn server() -> ControllerServerConfigV1 {
        let value = json!({
            "schema_version": 1,
            "serial_device": "/dev/serial/by-id/usb-Kiko_STM32_A1-if00",
            "controller_uid_hex": "00112233445566778899aabb",
            "firmware_abi": ROBOT_PROTOCOL_VERSION,
            "firmware_build_id": 42,
            "actuator_config_fingerprint_hex": "112233445566778899aabbccddeeff00",
            "hardware_profile_claim_id": "kiko-driver-profile-v1",
            "heartbeat_period_ms": 20,
            "maximum_heartbeat_age_ms": 60,
            "serial_applied_ack_timeout_ms": 30,
            "controller_clock_abs_error_ppm_bound": 50_000,
            "deadline_quantization_margin_ms": 2,
            "expected_max_abs_pwm_percent": 50,
            "expected_pwm_frequency_hz": 20_000,
            "expected_watchdog_nominal_timeout_ms": 250,
            "expected_neutral_output": "both_low",
            "expected_physical_stop_semantics": "coast_verified"
        });
        ControllerServerConfigV1::parse_json(
            &serde_json::to_vec(&value).expect("server config JSON"),
        )
        .expect("valid server config")
    }

    fn parse(value: &Value) -> Result<ZeroOnlyActuationPolicyV1, ZeroOnlyActuationPolicyError> {
        ZeroOnlyActuationPolicyV1::parse_json(&serde_json::to_vec(value).expect("policy JSON"))
    }

    #[test]
    fn exact_inventory_server_and_udp_endpoint_bind_without_motion_authority() {
        let parsed = parse(&policy()).expect("valid policy");
        let inventory = inventory("udp://127.0.0.1:8080");
        let bound = parsed
            .bind(
                &inventory,
                &server(),
                "127.0.0.1:8080".parse().expect("socket"),
            )
            .expect("exact binding");
        assert_eq!(
            bound.client().controller_uid(),
            *inventory.manifest().stm32().controller_uid()
        );
        assert_eq!(bound.inventory_content_sha256(), inventory.content_sha256());
    }

    #[test]
    fn endpoint_protocol_identity_and_server_budget_mismatches_fail_closed() {
        let wrong_protocol = parse(&policy())
            .expect("policy")
            .bind(
                &inventory("tcp://127.0.0.1:8080"),
                &server(),
                "127.0.0.1:8080".parse().expect("socket"),
            )
            .expect_err("TCP inventory must not bind UDP server");
        assert!(matches!(
            wrong_protocol,
            ZeroOnlyActuationBindingError::InventoryEndpointScheme { .. }
        ));

        let mut short = policy();
        short["applied_ack_timeout_ns"] = json!(34_999_999);
        assert!(matches!(
            parse(&short).expect("structural policy").bind(
                &inventory("udp://127.0.0.1:8080"),
                &server(),
                "127.0.0.1:8080".parse().expect("socket"),
            ),
            Err(
                ZeroOnlyActuationBindingError::ClientTimeoutBelowServerBudget {
                    field: "applied_ack_timeout_ns",
                    ..
                }
            )
        ));
    }

    #[test]
    fn parser_is_bounded_strict_and_rejects_trailing_documents() {
        let mut unknown = policy();
        unknown["surprise"] = json!(true);
        assert!(matches!(
            parse(&unknown),
            Err(ZeroOnlyActuationPolicyError::JsonDecode(_))
        ));

        let mut bytes = serde_json::to_vec(&policy()).expect("policy JSON");
        bytes.extend_from_slice(b" {}");
        assert!(matches!(
            ZeroOnlyActuationPolicyV1::parse_json(&bytes),
            Err(ZeroOnlyActuationPolicyError::JsonTrailingData(_))
        ));

        let oversized = vec![b' '; MAX_ZERO_ONLY_ACTUATION_POLICY_JSON_BYTES + 1];
        assert!(matches!(
            ZeroOnlyActuationPolicyV1::parse_json(&oversized),
            Err(ZeroOnlyActuationPolicyError::InputTooLarge { .. })
        ));
    }

    #[test]
    fn lease_strictly_covers_two_full_ack_budgets() {
        let mut equality = policy();
        equality["zero_acquisition_lease_ms"] = json!(80);
        assert!(matches!(
            parse(&equality),
            Err(ZeroOnlyActuationPolicyError::ZeroLeaseCannotCoverProactiveRenewal {
                lease,
                applied_ack_timeout,
                required_exclusive_lower_bound,
            }) if lease == Duration::from_millis(80)
                && applied_ack_timeout == Duration::from_millis(40)
                && required_exclusive_lower_bound == Duration::from_millis(80)
        ));

        let mut one_millisecond_of_slack = policy();
        one_millisecond_of_slack["zero_acquisition_lease_ms"] = json!(81);
        parse(&one_millisecond_of_slack).expect("strictly positive renewal slack");
    }
}
