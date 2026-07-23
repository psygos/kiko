//! Bounded, transport-independent control messages for the local robot agent.
//!
//! This module owns neither a socket nor authority. A Unix-domain socket
//! adapter may give one [`AgentControlRequestParser`] to one ordered request
//! stream, stamp receipt time itself, and then submit the parsed command to the
//! supervisor/runtime. Parsing a request does **not** arm the robot, acquire an
//! authority lease, prove localization, execute a stop, persist a map, or shut
//! down a process.
//!
//! The wire form is one exact JSON object with no bytes before or after it:
//!
//! ```text
//! {"schema_version":1,"request_id":7,"command":{"kind":"query_status"}}
//! ```
//!
//! Request IDs are nonzero and strictly increase within a parser instance.
//! Invalid requests do not consume an ID. Adapters that accept a new stream
//! must decide separately whether ordering is session-local or restored from a
//! durable identity; this module makes no authentication or replay-prevention
//! claim across parser instances.

use std::fmt;
use std::num::NonZeroU64;

use serde::{Deserialize, Serialize};

use super::{
    FiniteManualVelocityParseError, FiniteManualVelocityV1, ManualDriveParsedCommand,
    ManualDriveSequence, MapPointGoalSelection, MapPointGoalSelectionDto,
    MapPointGoalSelectionParseError, RecordedMapEpochId,
};

/// Wire schema supported by this request and response API.
pub const AGENT_CONTROL_SCHEMA_V1: u32 = 1;

/// Hard limit checked before JSON decoding or allocation.
///
/// Version 1 has no caller-controlled strings, paths, or collections. Four
/// KiB leaves ample room for numeric spellings while bounding parser work and
/// any buffering performed by a transport adapter.
pub const MAX_AGENT_CONTROL_REQUEST_JSON_BYTES: usize = 4 * 1_024;

/// A nonzero request identity, strictly ordered within one parser instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct AgentControlRequestId(NonZeroU64);

impl AgentControlRequestId {
    /// Return the wire value.
    pub const fn get(self) -> u64 {
        self.0.get()
    }

    fn try_new(raw: u64) -> Result<Self, AgentControlRequestParseError> {
        NonZeroU64::new(raw)
            .map(Self)
            .ok_or(AgentControlRequestParseError::ZeroRequestId)
    }
}

/// One fully parsed request. Its command has passed only boundary-level
/// structural and domain parsing; runtime admission remains separate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AgentControlRequestV1 {
    request_id: AgentControlRequestId,
    command: AgentControlCommandV1,
}

impl AgentControlRequestV1 {
    /// Strictly ordered identity of this request.
    pub const fn request_id(self) -> AgentControlRequestId {
        self.request_id
    }

    /// Parsed command intent.
    pub const fn command(self) -> AgentControlCommandV1 {
        self.command
    }
}

/// Stable command discriminator used in responses and diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentControlCommandKindV1 {
    QueryStatus,
    Arm,
    Disarm,
    MapOnly,
    Stop,
    BeginManual,
    ManualVelocity,
    ManualStop,
    FrontierExplore,
    SelectMapPoint,
    SaveMap,
    Shutdown,
}

/// Parsed command intents accepted by the protocol boundary.
///
/// `Arm` and `Disarm` are explicit lifecycle intents; parsing either one does
/// not prove a fresh-zero barrier, change authority, or touch hardware. `Stop`
/// is the global explicit stop/release intent while remaining distinct from a
/// request to disarm. `BeginManual` may acquire manual authority only after
/// separate runtime admission; velocity traffic can never acquire it
/// implicitly. `ManualStop` is an ordered manual-drive command and retains its
/// manual sequence. `SaveMap` deliberately carries no path: the runtime must
/// use its preconfigured, bounded persistence destination rather than
/// accepting filesystem authority through this protocol.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AgentControlCommandV1 {
    QueryStatus,
    Arm,
    Disarm,
    MapOnly,
    Stop,
    BeginManual,
    ManualVelocity(AgentManualVelocityV1),
    ManualStop(AgentManualStopV1),
    FrontierExplore,
    SelectMapPoint(MapPointGoalSelection),
    SaveMap,
    Shutdown,
}

impl AgentControlCommandV1 {
    /// Return the stable command discriminator.
    pub const fn kind(self) -> AgentControlCommandKindV1 {
        match self {
            Self::QueryStatus => AgentControlCommandKindV1::QueryStatus,
            Self::Arm => AgentControlCommandKindV1::Arm,
            Self::Disarm => AgentControlCommandKindV1::Disarm,
            Self::MapOnly => AgentControlCommandKindV1::MapOnly,
            Self::Stop => AgentControlCommandKindV1::Stop,
            Self::BeginManual => AgentControlCommandKindV1::BeginManual,
            Self::ManualVelocity(_) => AgentControlCommandKindV1::ManualVelocity,
            Self::ManualStop(_) => AgentControlCommandKindV1::ManualStop,
            Self::FrontierExplore => AgentControlCommandKindV1::FrontierExplore,
            Self::SelectMapPoint(_) => AgentControlCommandKindV1::SelectMapPoint,
            Self::SaveMap => AgentControlCommandKindV1::SaveMap,
            Self::Shutdown => AgentControlCommandKindV1::Shutdown,
        }
    }
}

/// A finite body-frame manual velocity intent in SI units.
///
/// This is intentionally still unadmitted relative to
/// [`super::ManualDriveCore`]. The control boundary proves finite values once
/// and preserves the existing [`ManualDriveSequence`]. It does not know the
/// active authority lease or configured velocity envelope, and it does not
/// reinterpret `(0, 0)` as a stop. Binding retains the finite-value proof;
/// the core remains responsible for authority, ordering, freshness, limits,
/// ambiguous zero, and deadman semantics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AgentManualVelocityV1 {
    sequence: ManualDriveSequence,
    velocity: FiniteManualVelocityV1,
}

impl AgentManualVelocityV1 {
    /// Ordered identity in the active manual authority lease.
    pub const fn sequence(self) -> ManualDriveSequence {
        self.sequence
    }

    /// Requested body-frame forward velocity in metres per second.
    pub const fn forward_velocity_mps(self) -> f64 {
        self.velocity.forward_velocity_mps()
    }

    /// Requested body-frame yaw rate in radians per second.
    pub const fn yaw_rate_rad_s(self) -> f64 {
        self.velocity.yaw_rate_rad_s()
    }

    /// Bind this finite intent to the supervisor adapter's exact lease type.
    ///
    /// The result retains the finite-value proof. `ManualDriveCore` must still
    /// perform configured-envelope, authority, ordering, receipt-freshness,
    /// ambiguous-zero, and deadman admission; binding alone is not motion
    /// authorization.
    pub const fn bind_to_manual_lease<LeaseId>(
        self,
        authority_lease_id: LeaseId,
    ) -> ManualDriveParsedCommand<LeaseId> {
        ManualDriveParsedCommand::velocity(authority_lease_id, self.sequence, self.velocity)
    }

    fn parse(
        request_id: AgentControlRequestId,
        sequence: u64,
        forward_velocity_mps: f64,
        yaw_rate_rad_s: f64,
    ) -> Result<Self, AgentControlRequestParseError> {
        let velocity = FiniteManualVelocityV1::parse(forward_velocity_mps, yaw_rate_rad_s)
            .map_err(
                |source| AgentControlRequestParseError::NonFiniteManualVelocity {
                    request_id,
                    source,
                },
            )?;
        Ok(Self {
            sequence: ManualDriveSequence::from_raw(sequence),
            velocity,
        })
    }
}

/// An explicit ordered stop within the manual-drive stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AgentManualStopV1 {
    sequence: ManualDriveSequence,
}

impl AgentManualStopV1 {
    /// Ordered identity in the active manual authority lease.
    pub const fn sequence(self) -> ManualDriveSequence {
        self.sequence
    }

    /// Bind this explicit stop to the supervisor adapter's exact lease type.
    ///
    /// The returned command still requires normal `ManualDriveCore` admission;
    /// in particular, parsing this request alone does not prove a stop was sent
    /// or applied by the base controller.
    pub const fn bind_to_manual_lease<LeaseId>(
        self,
        authority_lease_id: LeaseId,
    ) -> ManualDriveParsedCommand<LeaseId> {
        ManualDriveParsedCommand::stop(authority_lease_id, self.sequence)
    }
}

/// Stateful parser for one ordered control request stream.
///
/// The parser stores only the last successfully parsed request ID. JSON,
/// version, domain, duplicate, and regression failures leave it unchanged, so
/// a corrected retry may reuse the rejected ID.
#[derive(Debug, Default)]
pub struct AgentControlRequestParser {
    last_request_id: Option<AgentControlRequestId>,
}

impl AgentControlRequestParser {
    /// Create a fresh stream parser with no accepted request ID.
    pub const fn new() -> Self {
        Self {
            last_request_id: None,
        }
    }

    /// Return the most recent fully parsed request ID.
    pub const fn last_request_id(&self) -> Option<AgentControlRequestId> {
        self.last_request_id
    }

    /// Parse one exact bounded JSON document and enforce stream ordering.
    pub fn parse_next(
        &mut self,
        bytes: &[u8],
    ) -> Result<AgentControlRequestV1, AgentControlRequestParseError> {
        if bytes.is_empty() {
            return Err(AgentControlRequestParseError::EmptyInput);
        }
        if bytes.len() > MAX_AGENT_CONTROL_REQUEST_JSON_BYTES {
            return Err(AgentControlRequestParseError::InputTooLarge {
                actual_bytes: bytes.len(),
                maximum_bytes: MAX_AGENT_CONTROL_REQUEST_JSON_BYTES,
            });
        }
        if bytes.first() != Some(&b'{') {
            return Err(AgentControlRequestParseError::UnexpectedLeadingByte { byte: bytes[0] });
        }
        if bytes.last() != Some(&b'}') {
            return Err(AgentControlRequestParseError::UnexpectedTrailingByte {
                byte: bytes[bytes.len() - 1],
            });
        }

        let mut stream =
            serde_json::Deserializer::from_slice(bytes).into_iter::<AgentControlRequestV1Dto>();
        let dto = stream
            .next()
            .ok_or(AgentControlRequestParseError::EmptyInput)?
            .map_err(AgentControlRequestParseError::Json)?;
        let parsed_bytes = stream.byte_offset();
        if parsed_bytes != bytes.len() {
            return Err(AgentControlRequestParseError::TrailingBytes {
                parsed_bytes,
                total_bytes: bytes.len(),
            });
        }
        if dto.schema_version != AGENT_CONTROL_SCHEMA_V1 {
            return Err(AgentControlRequestParseError::UnsupportedSchemaVersion {
                actual: dto.schema_version,
                supported: AGENT_CONTROL_SCHEMA_V1,
            });
        }

        let request_id = AgentControlRequestId::try_new(dto.request_id)?;
        let command = dto.command.parse(request_id)?;
        if let Some(previous) = self.last_request_id {
            if request_id == previous {
                return Err(AgentControlRequestParseError::DuplicateRequestId { request_id });
            }
            if request_id < previous {
                return Err(AgentControlRequestParseError::RequestIdRegression {
                    previous,
                    current: request_id,
                });
            }
        }

        self.last_request_id = Some(request_id);
        Ok(AgentControlRequestV1 {
            request_id,
            command,
        })
    }
}

/// Exact parse and stream-order failure.
#[derive(Debug)]
pub enum AgentControlRequestParseError {
    EmptyInput,
    InputTooLarge {
        actual_bytes: usize,
        maximum_bytes: usize,
    },
    UnexpectedLeadingByte {
        byte: u8,
    },
    UnexpectedTrailingByte {
        byte: u8,
    },
    TrailingBytes {
        parsed_bytes: usize,
        total_bytes: usize,
    },
    Json(serde_json::Error),
    UnsupportedSchemaVersion {
        actual: u32,
        supported: u32,
    },
    ZeroRequestId,
    DuplicateRequestId {
        request_id: AgentControlRequestId,
    },
    RequestIdRegression {
        previous: AgentControlRequestId,
        current: AgentControlRequestId,
    },
    NonFiniteManualVelocity {
        request_id: AgentControlRequestId,
        source: FiniteManualVelocityParseError,
    },
    MapPoint {
        request_id: AgentControlRequestId,
        source: MapPointGoalSelectionParseError,
    },
}

impl AgentControlRequestParseError {
    /// Return a trustworthy request ID when structural parsing reached one.
    ///
    /// This lets an adapter correlate a domain/order rejection without parsing
    /// the weak bytes again. JSON and unsupported-schema failures deliberately
    /// return `None`.
    pub const fn request_id(&self) -> Option<AgentControlRequestId> {
        match self {
            Self::DuplicateRequestId { request_id }
            | Self::NonFiniteManualVelocity { request_id, .. }
            | Self::MapPoint { request_id, .. } => Some(*request_id),
            Self::RequestIdRegression { current, .. } => Some(*current),
            Self::EmptyInput
            | Self::InputTooLarge { .. }
            | Self::UnexpectedLeadingByte { .. }
            | Self::UnexpectedTrailingByte { .. }
            | Self::TrailingBytes { .. }
            | Self::Json(_)
            | Self::UnsupportedSchemaVersion { .. }
            | Self::ZeroRequestId => None,
        }
    }
}

impl fmt::Display for AgentControlRequestParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => formatter.write_str("agent-control request is empty"),
            Self::InputTooLarge {
                actual_bytes,
                maximum_bytes,
            } => write!(
                formatter,
                "agent-control request is {actual_bytes} bytes; maximum is {maximum_bytes} bytes"
            ),
            Self::UnexpectedLeadingByte { byte } => write!(
                formatter,
                "agent-control request must start with '{{', got byte 0x{byte:02x}"
            ),
            Self::UnexpectedTrailingByte { byte } => write!(
                formatter,
                "agent-control request must end with '}}', got byte 0x{byte:02x}"
            ),
            Self::TrailingBytes {
                parsed_bytes,
                total_bytes,
            } => write!(
                formatter,
                "agent-control request has trailing bytes after offset {parsed_bytes} of {total_bytes}"
            ),
            Self::Json(source) => write!(formatter, "invalid agent-control JSON: {source}"),
            Self::UnsupportedSchemaVersion { actual, supported } => write!(
                formatter,
                "unsupported agent-control schema {actual}; supported schema is {supported}"
            ),
            Self::ZeroRequestId => formatter.write_str("agent-control request ID must be nonzero"),
            Self::DuplicateRequestId { request_id } => write!(
                formatter,
                "duplicate agent-control request ID {}",
                request_id.get()
            ),
            Self::RequestIdRegression { previous, current } => write!(
                formatter,
                "agent-control request ID regressed from {} to {}",
                previous.get(),
                current.get()
            ),
            Self::NonFiniteManualVelocity { source, .. } => {
                write!(formatter, "invalid manual velocity: {source}")
            }
            Self::MapPoint { source, .. } => {
                write!(formatter, "invalid map-point command: {source}")
            }
        }
    }
}

impl std::error::Error for AgentControlRequestParseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Json(source) => Some(source),
            Self::NonFiniteManualVelocity { source, .. } => Some(source),
            Self::MapPoint { source, .. } => Some(source),
            _ => None,
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AgentControlRequestV1Dto {
    schema_version: u32,
    request_id: u64,
    command: AgentControlCommandV1Dto,
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum AgentControlCommandV1Dto {
    QueryStatus {},
    Arm {},
    Disarm {},
    MapOnly {},
    Stop {},
    BeginManual {},
    ManualVelocity {
        sequence: u64,
        forward_velocity_mps: f64,
        yaw_rate_rad_s: f64,
    },
    ManualStop {
        sequence: u64,
    },
    FrontierExplore {},
    SelectMapPoint {
        map_epoch_id: u64,
        displayed_revision: u64,
        x_m: f64,
        y_m: f64,
    },
    SaveMap {},
    Shutdown {},
}

impl AgentControlCommandV1Dto {
    fn parse(
        self,
        request_id: AgentControlRequestId,
    ) -> Result<AgentControlCommandV1, AgentControlRequestParseError> {
        match self {
            Self::QueryStatus {} => Ok(AgentControlCommandV1::QueryStatus),
            Self::Arm {} => Ok(AgentControlCommandV1::Arm),
            Self::Disarm {} => Ok(AgentControlCommandV1::Disarm),
            Self::MapOnly {} => Ok(AgentControlCommandV1::MapOnly),
            Self::Stop {} => Ok(AgentControlCommandV1::Stop),
            Self::BeginManual {} => Ok(AgentControlCommandV1::BeginManual),
            Self::ManualVelocity {
                sequence,
                forward_velocity_mps,
                yaw_rate_rad_s,
            } => AgentManualVelocityV1::parse(
                request_id,
                sequence,
                forward_velocity_mps,
                yaw_rate_rad_s,
            )
            .map(AgentControlCommandV1::ManualVelocity),
            Self::ManualStop { sequence } => {
                Ok(AgentControlCommandV1::ManualStop(AgentManualStopV1 {
                    sequence: ManualDriveSequence::from_raw(sequence),
                }))
            }
            Self::FrontierExplore {} => Ok(AgentControlCommandV1::FrontierExplore),
            Self::SelectMapPoint {
                map_epoch_id,
                displayed_revision,
                x_m,
                y_m,
            } => MapPointGoalSelection::parse(MapPointGoalSelectionDto {
                map_epoch_id,
                displayed_revision,
                x_m,
                y_m,
            })
            .map(AgentControlCommandV1::SelectMapPoint)
            .map_err(|source| AgentControlRequestParseError::MapPoint { request_id, source }),
            Self::SaveMap {} => Ok(AgentControlCommandV1::SaveMap),
            Self::Shutdown {} => Ok(AgentControlCommandV1::Shutdown),
        }
    }
}

/// Runtime phase reported by the stable status response.
///
/// This is observational state, not permission to command motion.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AgentRuntimeStateV1 {
    Booting,
    Inventory,
    Disarmed,
    AwaitingZero,
    ReadyStopped,
    Active { mode: AgentOperatingModeV1 },
    Faulted,
    ShuttingDown,
}

/// Mutually exclusive runtime mode reported while active.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentOperatingModeV1 {
    MapOnly,
    Commissioning,
    Manual,
    FrontierExplore,
    PointGoal,
}

/// Best known base command state. It deliberately does not claim measured
/// physical motion when wheel encoders are absent.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentBaseCommandStateV1 {
    Unknown,
    ConfirmedStopped,
    CommandOutstanding,
}

/// Current localization relationship to an available map.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentLocalizationStateV1 {
    Unavailable,
    Localized,
    Lost,
}

/// Exact map identity reported in a status snapshot.
///
/// The representation is private so an available map cannot carry a zero
/// epoch ID. Use [`Self::UNAVAILABLE`] or [`Self::available`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct AgentMapStateV1(AgentMapStateV1Repr);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum AgentMapStateV1Repr {
    Unavailable,
    Available {
        map_epoch_id: u64,
        revision: u64,
        localization: AgentLocalizationStateV1,
    },
}

impl AgentMapStateV1 {
    /// No occupancy map is currently available.
    pub const UNAVAILABLE: Self = Self(AgentMapStateV1Repr::Unavailable);

    /// Build a status map binding from the coordinator's typed epoch.
    pub fn available(
        map_epoch_id: RecordedMapEpochId,
        revision: u64,
        localization: AgentLocalizationStateV1,
    ) -> Self {
        Self(AgentMapStateV1Repr::Available {
            map_epoch_id: map_epoch_id.as_u64(),
            revision,
            localization,
        })
    }

    /// Wire-stable map epoch when a map is available.
    pub const fn map_epoch_id(self) -> Option<u64> {
        match self.0 {
            AgentMapStateV1Repr::Unavailable => None,
            AgentMapStateV1Repr::Available { map_epoch_id, .. } => Some(map_epoch_id),
        }
    }

    /// Mapper revision when a map is available.
    pub const fn revision(self) -> Option<u64> {
        match self.0 {
            AgentMapStateV1Repr::Unavailable => None,
            AgentMapStateV1Repr::Available { revision, .. } => Some(revision),
        }
    }

    /// Localization state when a map is available.
    pub const fn localization(self) -> Option<AgentLocalizationStateV1> {
        match self.0 {
            AgentMapStateV1Repr::Unavailable => None,
            AgentMapStateV1Repr::Available { localization, .. } => Some(localization),
        }
    }
}

/// Fixed-shape status payload. No field grants authority or promises that a
/// requested command has reached hardware.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct AgentControlStatusV1 {
    runtime: AgentRuntimeStateV1,
    base_command: AgentBaseCommandStateV1,
    map: AgentMapStateV1,
}

impl AgentControlStatusV1 {
    /// Construct an observational status snapshot from already typed runtime
    /// state.
    pub const fn new(
        runtime: AgentRuntimeStateV1,
        base_command: AgentBaseCommandStateV1,
        map: AgentMapStateV1,
    ) -> Self {
        Self {
            runtime,
            base_command,
            map,
        }
    }

    pub const fn runtime(self) -> AgentRuntimeStateV1 {
        self.runtime
    }

    pub const fn base_command(self) -> AgentBaseCommandStateV1 {
        self.base_command
    }

    pub const fn map(self) -> AgentMapStateV1 {
        self.map
    }
}

/// Whether a successful response means queue admission or completed handling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentControlCompletionV1 {
    AcceptedForProcessing,
    Completed,
}

/// Stable machine-readable rejection categories for a transport adapter.
///
/// The closed enum avoids unbounded caller-controlled response strings. A
/// runtime should log its richer typed source error separately.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentControlRejectionCodeV1 {
    MalformedRequest,
    UnsupportedSchema,
    RequestOrder,
    NotReady,
    AuthorityDenied,
    ModeConflict,
    StaleMapSelection,
    MapUnavailable,
    LocalizationUnavailable,
    SafetyStopped,
    PersistenceFailed,
    ShutdownInProgress,
    InternalFault,
}

/// Stable response payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AgentControlResponseKindV1 {
    Accepted {
        command: AgentControlCommandKindV1,
        completion: AgentControlCompletionV1,
    },
    Status {
        status: AgentControlStatusV1,
    },
    Rejected {
        code: AgentControlRejectionCodeV1,
        retryable: bool,
    },
}

/// Fixed, serializable version-1 response envelope.
///
/// Serialization performs no transport I/O by itself. `request_id` is `null`
/// only when malformed input did not yield a trustworthy nonzero ID.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct AgentControlResponseV1 {
    schema_version: u32,
    request_id: Option<AgentControlRequestId>,
    response: AgentControlResponseKindV1,
}

impl AgentControlResponseV1 {
    /// Report admission or completion without claiming more than `completion`.
    pub const fn accepted(
        request_id: AgentControlRequestId,
        command: AgentControlCommandKindV1,
        completion: AgentControlCompletionV1,
    ) -> Self {
        Self {
            schema_version: AGENT_CONTROL_SCHEMA_V1,
            request_id: Some(request_id),
            response: AgentControlResponseKindV1::Accepted {
                command,
                completion,
            },
        }
    }

    /// Respond to `query_status` with an observational snapshot.
    pub const fn status(request_id: AgentControlRequestId, status: AgentControlStatusV1) -> Self {
        Self {
            schema_version: AGENT_CONTROL_SCHEMA_V1,
            request_id: Some(request_id),
            response: AgentControlResponseKindV1::Status { status },
        }
    }

    /// Report a closed, machine-readable rejection.
    pub const fn rejected(
        request_id: Option<AgentControlRequestId>,
        code: AgentControlRejectionCodeV1,
        retryable: bool,
    ) -> Self {
        Self {
            schema_version: AGENT_CONTROL_SCHEMA_V1,
            request_id,
            response: AgentControlResponseKindV1::Rejected { code, retryable },
        }
    }

    pub const fn request_id(self) -> Option<AgentControlRequestId> {
        self.request_id
    }

    pub const fn response(self) -> AgentControlResponseKindV1 {
        self.response
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::*;
    use crate::navigation::ManualVelocityComponentV1;

    fn request(id: u64, command: Value) -> Vec<u8> {
        serde_json::to_vec(&json!({
            "schema_version": AGENT_CONTROL_SCHEMA_V1,
            "request_id": id,
            "command": command,
        }))
        .expect("serialize request fixture")
    }

    #[test]
    fn parses_every_v1_command_and_preserves_existing_domain_types() {
        let commands = [
            json!({"kind": "query_status"}),
            json!({"kind": "arm"}),
            json!({"kind": "disarm"}),
            json!({"kind": "map_only"}),
            json!({"kind": "stop"}),
            json!({"kind": "begin_manual"}),
            json!({
                "kind": "manual_velocity",
                "sequence": 0,
                "forward_velocity_mps": 0.25,
                "yaw_rate_rad_s": -0.5
            }),
            json!({"kind": "manual_stop", "sequence": 1}),
            json!({"kind": "frontier_explore"}),
            json!({
                "kind": "select_map_point",
                "map_epoch_id": 9,
                "displayed_revision": 42,
                "x_m": -1.25,
                "y_m": 2.5
            }),
            json!({"kind": "save_map"}),
            json!({"kind": "shutdown"}),
        ];
        let expected = [
            AgentControlCommandKindV1::QueryStatus,
            AgentControlCommandKindV1::Arm,
            AgentControlCommandKindV1::Disarm,
            AgentControlCommandKindV1::MapOnly,
            AgentControlCommandKindV1::Stop,
            AgentControlCommandKindV1::BeginManual,
            AgentControlCommandKindV1::ManualVelocity,
            AgentControlCommandKindV1::ManualStop,
            AgentControlCommandKindV1::FrontierExplore,
            AgentControlCommandKindV1::SelectMapPoint,
            AgentControlCommandKindV1::SaveMap,
            AgentControlCommandKindV1::Shutdown,
        ];
        let mut parser = AgentControlRequestParser::new();
        for (index, (command, expected_kind)) in commands.into_iter().zip(expected).enumerate() {
            let id = u64::try_from(index + 1).expect("small fixture ID");
            let parsed = parser
                .parse_next(&request(id, command))
                .expect("valid command");
            assert_eq!(parsed.request_id().get(), id);
            assert_eq!(parsed.command().kind(), expected_kind);
        }

        let selected = parser
            .parse_next(&request(
                13,
                json!({
                    "kind": "select_map_point",
                    "map_epoch_id": 11,
                    "displayed_revision": 91,
                    "x_m": 3.0,
                    "y_m": -4.0
                }),
            ))
            .expect("typed map selection");
        let AgentControlCommandV1::SelectMapPoint(selected) = selected.command() else {
            panic!("expected map-point command");
        };
        assert_eq!(selected.map_epoch_id().as_u64(), 11);
        assert_eq!(selected.displayed_revision(), 91);
        assert_eq!(selected.point().as_array(), [3.0, -4.0]);
    }

    #[test]
    fn arm_and_disarm_have_exact_zero_payload_wire_forms() {
        let mut parser = AgentControlRequestParser::new();
        let arm = parser
            .parse_next(&request(1, json!({"kind": "arm"})))
            .expect("exact arm intent");
        assert_eq!(arm.command(), AgentControlCommandV1::Arm);
        assert_eq!(arm.command().kind(), AgentControlCommandKindV1::Arm);

        let disarm = parser
            .parse_next(&request(2, json!({"kind": "disarm"})))
            .expect("exact disarm intent");
        assert_eq!(disarm.command(), AgentControlCommandV1::Disarm);
        assert_eq!(disarm.command().kind(), AgentControlCommandKindV1::Disarm);

        for (request, kind, expected, request_id) in [
            (arm, AgentControlCommandKindV1::Arm, "arm", 1),
            (disarm, AgentControlCommandKindV1::Disarm, "disarm", 2),
        ] {
            assert_eq!(
                serde_json::to_value(kind).expect("serialize lifecycle command kind"),
                json!(expected)
            );
            assert_eq!(
                serde_json::to_value(AgentControlResponseV1::accepted(
                    request.request_id(),
                    kind,
                    AgentControlCompletionV1::Completed,
                ))
                .expect("serialize lifecycle response"),
                json!({
                    "schema_version": AGENT_CONTROL_SCHEMA_V1,
                    "request_id": request_id,
                    "response": {
                        "kind": "accepted",
                        "command": expected,
                        "completion": "completed"
                    }
                })
            );
        }
    }

    #[test]
    fn manual_commands_bind_once_parsed_values_to_the_exact_manual_lease() {
        let mut parser = AgentControlRequestParser::new();
        let velocity = parser
            .parse_next(&request(
                1,
                json!({
                    "kind": "manual_velocity",
                    "sequence": 17,
                    "forward_velocity_mps": 1.0e100,
                    "yaw_rate_rad_s": -2.0e100
                }),
            ))
            .expect("finite values remain unadmitted until configured admission");
        let AgentControlCommandV1::ManualVelocity(velocity) = velocity.command() else {
            panic!("expected velocity");
        };
        let parsed =
            velocity.bind_to_manual_lease(NonZeroU64::new(7).expect("nonzero lease identity"));
        assert_eq!(parsed.sequence().get(), 17);
        assert_eq!(parsed.authority_lease_id().get(), 7);
        let finite = parsed
            .finite_velocity()
            .expect("velocity command retains finite proof");
        assert_eq!(finite.forward_velocity_mps(), 1.0e100);
        assert_eq!(finite.yaw_rate_rad_s(), -2.0e100);

        let stop = parser
            .parse_next(&request(2, json!({"kind": "manual_stop", "sequence": 18})))
            .expect("explicit manual stop");
        let AgentControlCommandV1::ManualStop(stop) = stop.command() else {
            panic!("expected stop");
        };
        let parsed = stop.bind_to_manual_lease(7_u64);
        assert_eq!(parsed.sequence().get(), 18);
        assert!(parsed.is_explicit_stop());
    }

    #[test]
    fn manual_zero_is_not_silently_reinterpreted_or_envelope_admitted() {
        let mut parser = AgentControlRequestParser::new();
        let parsed = parser
            .parse_next(&request(
                1,
                json!({
                    "kind": "manual_velocity",
                    "sequence": 0,
                    "forward_velocity_mps": 0.0,
                    "yaw_rate_rad_s": -0.0
                }),
            ))
            .expect("finite velocity boundary");
        let AgentControlCommandV1::ManualVelocity(velocity) = parsed.command() else {
            panic!("zero-valued velocity must remain a velocity intent");
        };
        assert_eq!(velocity.sequence(), ManualDriveSequence::from_raw(0));
        assert_eq!(velocity.forward_velocity_mps(), 0.0);
        assert_eq!(velocity.yaw_rate_rad_s(), 0.0);
        assert!(
            velocity
                .bind_to_manual_lease(1_u64)
                .finite_velocity()
                .is_some()
        );
    }

    #[test]
    fn request_ids_are_nonzero_strictly_increasing_and_failures_do_not_advance() {
        let mut parser = AgentControlRequestParser::new();
        assert!(matches!(
            parser.parse_next(&request(0, json!({"kind": "query_status"}))),
            Err(AgentControlRequestParseError::ZeroRequestId)
        ));
        assert_eq!(parser.last_request_id(), None);

        parser
            .parse_next(&request(5, json!({"kind": "query_status"})))
            .expect("initial nonzero ID may start above one");
        assert!(matches!(
            parser.parse_next(&request(5, json!({"kind": "stop"}))),
            Err(AgentControlRequestParseError::DuplicateRequestId { request_id })
                if request_id.get() == 5
        ));
        assert!(matches!(
            parser.parse_next(&request(4, json!({"kind": "stop"}))),
            Err(AgentControlRequestParseError::RequestIdRegression { previous, current })
                if previous.get() == 5 && current.get() == 4
        ));
        assert_eq!(
            parser.last_request_id().map(AgentControlRequestId::get),
            Some(5)
        );

        assert!(matches!(
            parser.parse_next(&request(
                6,
                json!({
                    "kind": "select_map_point",
                    "map_epoch_id": 0,
                    "displayed_revision": 1,
                    "x_m": 0.0,
                    "y_m": 0.0
                }),
            )),
            Err(AgentControlRequestParseError::MapPoint { request_id, .. })
                if request_id.get() == 6
        ));
        assert_eq!(
            parser.last_request_id().map(AgentControlRequestId::get),
            Some(5)
        );
        parser
            .parse_next(&request(6, json!({"kind": "stop"})))
            .expect("corrected request reuses unconsumed ID");
    }

    #[test]
    fn rejects_unknown_missing_duplicate_and_unsupported_schema_fields() {
        let cases: &[&[u8]] = &[
            br#"{"schema_version":1,"request_id":1,"command":{"kind":"query_status"},"extra":0}"#,
            br#"{"schema_version":1,"request_id":1,"command":{"kind":"stop","extra":0}}"#,
            br#"{"schema_version":1,"request_id":1,"command":{"kind":"arm","extra":0}}"#,
            br#"{"schema_version":1,"request_id":1,"command":{"kind":"disarm","reason":"operator"}}"#,
            br#"{"schema_version":1,"request_id":1,"command":{"kind":"unknown"}}"#,
            br#"{"schema_version":1,"request_id":1}"#,
            br#"{"schema_version":1,"schema_version":1,"request_id":1,"command":{"kind":"stop"}}"#,
            br#"{"schema_version":1,"request_id":1,"command":{"kind":"stop","kind":"shutdown"}}"#,
        ];
        for bytes in cases {
            assert!(
                matches!(
                    AgentControlRequestParser::new().parse_next(bytes),
                    Err(AgentControlRequestParseError::Json(_))
                ),
                "case must fail: {}",
                String::from_utf8_lossy(bytes)
            );
        }

        assert!(matches!(
            AgentControlRequestParser::new()
                .parse_next(br#"{"schema_version":2,"request_id":1,"command":{"kind":"stop"}}"#),
            Err(AgentControlRequestParseError::UnsupportedSchemaVersion {
                actual: 2,
                supported: AGENT_CONTROL_SCHEMA_V1
            })
        ));
    }

    #[test]
    fn exact_framing_rejects_surrounding_and_trailing_bytes() {
        for bytes in [
            b"".as_slice(),
            br#" {"schema_version":1,"request_id":1,"command":{"kind":"stop"}}"#,
            b"\n{\"schema_version\":1,\"request_id\":1,\"command\":{\"kind\":\"stop\"}}",
        ] {
            assert!(matches!(
                AgentControlRequestParser::new().parse_next(bytes),
                Err(AgentControlRequestParseError::EmptyInput
                    | AgentControlRequestParseError::UnexpectedLeadingByte { .. })
            ));
        }
        for bytes in [
            br#"{"schema_version":1,"request_id":1,"command":{"kind":"stop"}} "#.as_slice(),
            b"{\"schema_version\":1,\"request_id\":1,\"command\":{\"kind\":\"stop\"}}\0",
        ] {
            assert!(matches!(
                AgentControlRequestParser::new().parse_next(bytes),
                Err(AgentControlRequestParseError::UnexpectedTrailingByte { .. })
            ));
        }

        let two_documents = br#"{"schema_version":1,"request_id":1,"command":{"kind":"stop"}}{"schema_version":1,"request_id":2,"command":{"kind":"stop"}}"#;
        assert!(matches!(
            AgentControlRequestParser::new().parse_next(two_documents),
            Err(AgentControlRequestParseError::TrailingBytes { .. })
        ));
    }

    #[test]
    fn bounded_input_is_checked_before_decoding() {
        let oversized = vec![b'{'; MAX_AGENT_CONTROL_REQUEST_JSON_BYTES + 1];
        assert!(matches!(
            AgentControlRequestParser::new().parse_next(&oversized),
            Err(AgentControlRequestParseError::InputTooLarge {
                actual_bytes,
                maximum_bytes: MAX_AGENT_CONTROL_REQUEST_JSON_BYTES,
            }) if actual_bytes == MAX_AGENT_CONTROL_REQUEST_JSON_BYTES + 1
        ));
    }

    #[test]
    fn every_strict_prefix_of_a_valid_request_is_rejected() {
        let valid = request(
            1,
            json!({
                "kind": "select_map_point",
                "map_epoch_id": 1,
                "displayed_revision": 0,
                "x_m": 0.125,
                "y_m": -0.25
            }),
        );
        for end in 0..valid.len() {
            assert!(
                AgentControlRequestParser::new()
                    .parse_next(&valid[..end])
                    .is_err(),
                "truncated at byte {end}"
            );
        }
        assert!(AgentControlRequestParser::new().parse_next(&valid).is_ok());
    }

    #[test]
    fn invalid_numbers_and_map_domains_are_rejected_without_order_effects() {
        let mut parser = AgentControlRequestParser::new();
        for bytes in [
            br#"{"schema_version":1,"request_id":1,"command":{"kind":"manual_velocity","sequence":0,"forward_velocity_mps":1e400,"yaw_rate_rad_s":0}}"#.as_slice(),
            br#"{"schema_version":1,"request_id":1,"command":{"kind":"manual_velocity","sequence":0,"forward_velocity_mps":0,"yaw_rate_rad_s":NaN}}"#,
            br#"{"schema_version":1,"request_id":1,"command":{"kind":"select_map_point","map_epoch_id":1,"displayed_revision":0,"x_m":1e400,"y_m":0}}"#,
        ] {
            assert!(matches!(
                parser.parse_next(bytes),
                Err(AgentControlRequestParseError::Json(_))
            ));
            assert_eq!(parser.last_request_id(), None);
        }
        let request_id = AgentControlRequestId::try_new(1).expect("request ID");
        assert!(matches!(
            AgentManualVelocityV1::parse(request_id, 0, f64::INFINITY, 0.0),
            Err(AgentControlRequestParseError::NonFiniteManualVelocity {
                request_id: actual,
                source,
            }) if actual == request_id
                && source.component() == ManualVelocityComponentV1::ForwardVelocityMps
                && source.value() == f64::INFINITY
        ));
        assert!(matches!(
            AgentManualVelocityV1::parse(request_id, 0, 0.0, f64::NAN),
            Err(AgentControlRequestParseError::NonFiniteManualVelocity {
                request_id: actual,
                source,
            }) if actual == request_id
                && source.component() == ManualVelocityComponentV1::YawRateRadS
                && source.value().is_nan()
        ));
    }

    #[test]
    fn response_envelope_is_stable_bounded_and_truthful() {
        let mut parser = AgentControlRequestParser::new();
        let id = parser
            .parse_next(&request(7, json!({"kind": "query_status"})))
            .expect("request")
            .request_id();
        let map_epoch = RecordedMapEpochId::try_new(3).expect("map epoch");
        let status = AgentControlStatusV1::new(
            AgentRuntimeStateV1::Active {
                mode: AgentOperatingModeV1::MapOnly,
            },
            AgentBaseCommandStateV1::ConfirmedStopped,
            AgentMapStateV1::available(map_epoch, 12, AgentLocalizationStateV1::Localized),
        );
        assert_eq!(status.map().map_epoch_id(), Some(3));
        assert_eq!(status.map().revision(), Some(12));
        assert_eq!(
            status.map().localization(),
            Some(AgentLocalizationStateV1::Localized)
        );
        assert_eq!(AgentMapStateV1::UNAVAILABLE.map_epoch_id(), None);
        let encoded = serde_json::to_value(AgentControlResponseV1::status(id, status))
            .expect("serialize status response");
        assert_eq!(
            encoded,
            json!({
                "schema_version": 1,
                "request_id": 7,
                "response": {
                    "kind": "status",
                    "status": {
                        "runtime": {"kind": "active", "mode": "map_only"},
                        "base_command": "confirmed_stopped",
                        "map": {
                            "kind": "available",
                            "map_epoch_id": 3,
                            "revision": 12,
                            "localization": "localized"
                        }
                    }
                }
            })
        );

        let accepted = serde_json::to_value(AgentControlResponseV1::accepted(
            id,
            AgentControlCommandKindV1::Stop,
            AgentControlCompletionV1::AcceptedForProcessing,
        ))
        .expect("serialize accepted response");
        assert_eq!(
            accepted["response"]["completion"],
            "accepted_for_processing"
        );

        let rejected = serde_json::to_value(AgentControlResponseV1::rejected(
            None,
            AgentControlRejectionCodeV1::MalformedRequest,
            false,
        ))
        .expect("serialize rejection");
        assert_eq!(rejected["request_id"], Value::Null);
        assert_eq!(rejected["response"]["code"], "malformed_request");
        assert_eq!(rejected["response"]["retryable"], Value::Bool(false));
    }

    #[test]
    fn every_lifecycle_and_authority_mode_has_an_exact_wire_name() {
        let lifecycle = [
            (AgentRuntimeStateV1::Booting, "booting"),
            (AgentRuntimeStateV1::Inventory, "inventory"),
            (AgentRuntimeStateV1::Disarmed, "disarmed"),
            (AgentRuntimeStateV1::AwaitingZero, "awaiting_zero"),
            (AgentRuntimeStateV1::ReadyStopped, "ready_stopped"),
            (AgentRuntimeStateV1::Faulted, "faulted"),
            (AgentRuntimeStateV1::ShuttingDown, "shutting_down"),
        ];
        for (state, expected) in lifecycle {
            let encoded = serde_json::to_value(state).expect("serialize lifecycle state");
            assert_eq!(encoded, json!({"kind": expected}));
        }

        let modes = [
            (AgentOperatingModeV1::MapOnly, "map_only"),
            (AgentOperatingModeV1::Commissioning, "commissioning"),
            (AgentOperatingModeV1::Manual, "manual"),
            (AgentOperatingModeV1::FrontierExplore, "frontier_explore"),
            (AgentOperatingModeV1::PointGoal, "point_goal"),
        ];
        for (mode, expected) in modes {
            let encoded = serde_json::to_value(AgentRuntimeStateV1::Active { mode })
                .expect("serialize active mode");
            assert_eq!(encoded, json!({"kind": "active", "mode": expected}));
        }
    }
}
