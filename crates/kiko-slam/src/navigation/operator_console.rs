//! Host-testable operator/agent console boundary.
//!
//! The console is intentionally upstream of the existing
//! [`super::AgentControlDispatcher`]. Browser and agent sessions have their own
//! ordered identities, but only this module allocates the one process-lifetime
//! downstream request sequence. The bounded receiver is non-clone and is the
//! sole adapter seam into the existing runtime owner.
//!
//! This module owns no camera, serial port, STM32 connection, motor authority,
//! or Rerun stream. Queue acceptance is never represented as physical
//! application. Only an exact receipt supplied by the runtime may populate the
//! applied-receipt field.

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::net::{IpAddr, Ipv4Addr, SocketAddr, SocketAddrV4};
use std::num::{NonZeroU16, NonZeroU32, NonZeroU64, NonZeroUsize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, Weak};

use crossbeam_channel::{Receiver, Sender, TryRecvError, TrySendError};
use serde::Serializer;
use serde::{Deserialize, Serialize};

use super::{
    AgentControlCommandV1, AgentControlRequestId, AgentControlRequestV1, AgentManualVelocityV1,
    AgentRuntimeStateV1, FiniteManualVelocityParseError, FiniteManualVelocityV1,
    ManualDriveSequence, MapPointGoalSelection, MapPointGoalSelectionDto,
    MapPointGoalSelectionParseError,
};
use crate::HostMonotonicTimestamp;

fn recover_lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn serialize_u64_as_decimal_string<S>(value: &u64, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    serializer.serialize_str(&value.to_string())
}

fn serialize_nonzero_u64_as_decimal_string<S>(
    value: &NonZeroU64,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    serializer.serialize_str(&value.get().to_string())
}

fn serialize_optional_u64_as_decimal_string<S>(
    value: &Option<u64>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    match value {
        Some(value) => serializer.serialize_some(&value.to_string()),
        None => serializer.serialize_none(),
    }
}

pub const OPERATOR_CONSOLE_SCHEMA_V1: u32 = 1;
/// Legacy snapshot schema. Its optional diagnostics field was never populated
/// by the server and the bundled client treated a value as a browser URL.
pub const OPERATOR_CONSOLE_SNAPSHOT_SCHEMA_V2: u32 = 2;
/// Snapshot schema whose diagnostics field is a canonical Rerun proxy URI.
pub const OPERATOR_CONSOLE_SNAPSHOT_SCHEMA_V3: u32 = 3;
/// Snapshot schema with an explicit, deny-by-default runtime authority class.
///
/// V4 prevents a client from inferring production authority from the absence
/// of a qualification-only field.
pub const OPERATOR_CONSOLE_SNAPSHOT_SCHEMA_V4: u32 = 4;
/// Snapshot schema with separate OAK transport and sparse-SLAM evidence.
///
/// V5 removes the misleading implication that fresh camera streams alone are
/// evidence of a healthy tracker. It also carries the requested and actually
/// selected inference providers plus an exact integer throughput window.
pub const OPERATOR_CONSOLE_SNAPSHOT_SCHEMA_V5: u32 = 5;
pub const MAX_OPERATOR_CONSOLE_REQUEST_BYTES: usize = 8 * 1_024;
pub const MAX_OPERATOR_CONSOLE_SESSIONS: usize = 32;
pub const MAX_OPERATOR_CONSOLE_QUEUE_CAPACITY: usize = 256;
pub const MAX_OPERATOR_CONSOLE_IDEMPOTENCY_RECORDS: usize = 64;
pub const MAX_OPERATOR_CONSOLE_RESPONSE_RECORDS: usize = 1_024;
pub const OPERATOR_CONSOLE_CRITICAL_RESPONSE_RESERVE: usize = 16;
pub const MAX_OPERATOR_CONSOLE_SUBSCRIBERS: usize = 32;
pub const MAX_OPERATOR_CONSOLE_GRID_CELLS: usize = 2_000_000;
pub const MAX_OPERATOR_CONSOLE_PATH_POINTS: usize = 16_384;

/// A socket address parsed once and proven to be loopback-only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OperatorConsoleBind(SocketAddr);

impl OperatorConsoleBind {
    pub fn parse(address: SocketAddr) -> Result<Self, OperatorConsoleBindError> {
        if !address.ip().is_loopback() {
            return Err(OperatorConsoleBindError::NotLoopback(address.ip()));
        }
        Ok(Self(address))
    }

    pub const fn address(self) -> SocketAddr {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperatorConsoleBindError {
    NotLoopback(IpAddr),
}

impl fmt::Display for OperatorConsoleBindError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotLoopback(ip) => {
                write!(formatter, "operator console bind IP {ip} is not loopback")
            }
        }
    }
}

impl std::error::Error for OperatorConsoleBindError {}

/// Rerun proxy URI exposed to an operator through a same-port SSH loopback
/// forward.
///
/// The Nano listener address is parsed once by launch admission. This type
/// retains only its nonzero port and deliberately renders the operator-side
/// endpoint as `127.0.0.1`; it never advertises the Nano address or a public
/// listener. The URI is diagnostic metadata and grants no control authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConsoleRerunDiagnosticsUrl {
    forwarded_port: NonZeroU16,
}

impl ConsoleRerunDiagnosticsUrl {
    /// Construct from the nonzero port already admitted by the Nano launch
    /// parser. The listener address is fixed here, so a diagnostics endpoint
    /// cannot retain an address which disagrees with the documented tunnel.
    pub const fn from_admitted_forwarded_port(forwarded_port: NonZeroU16) -> Self {
        Self { forwarded_port }
    }

    pub const fn forwarded_port(self) -> NonZeroU16 {
        self.forwarded_port
    }

    pub fn serve_loopback_bind(self) -> SocketAddr {
        SocketAddr::V4(SocketAddrV4::new(
            Ipv4Addr::LOCALHOST,
            self.forwarded_port.get(),
        ))
    }
}

impl fmt::Display for ConsoleRerunDiagnosticsUrl {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "rerun+http://127.0.0.1:{}/proxy",
            self.forwarded_port
        )
    }
}

impl Serialize for ConsoleRerunDiagnosticsUrl {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.collect_str(self)
    }
}

/// Exact nanoseconds in the runtime's injected host-monotonic epoch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct ConsoleHostTimestampNs(u64);

impl ConsoleHostTimestampNs {
    pub fn from_host(value: HostMonotonicTimestamp) -> Self {
        Self(value.as_nanos())
    }

    pub const fn as_nanos(self) -> u64 {
        self.0
    }

    fn checked_add_millis(self, duration_ms: u64) -> Option<Self> {
        duration_ms
            .checked_mul(1_000_000)
            .and_then(|duration_ns| self.0.checked_add(duration_ns))
            .map(Self)
    }
}

impl Serialize for ConsoleHostTimestampNs {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.0.to_string())
    }
}

/// A finite scalar used in observational geometry.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct ConsoleFiniteF64(f64);

impl ConsoleFiniteF64 {
    pub fn parse(value: f64) -> Result<Self, ConsoleFiniteF64Error> {
        if value.is_finite() {
            Ok(Self(value))
        } else {
            Err(ConsoleFiniteF64Error(value))
        }
    }

    pub const fn get(self) -> f64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConsoleFiniteF64Error(f64);

impl fmt::Display for ConsoleFiniteF64Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "console scalar must be finite, got {}", self.0)
    }
}

impl std::error::Error for ConsoleFiniteF64Error {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ConsoleSessionId(NonZeroU64);

impl ConsoleSessionId {
    pub fn parse(value: u64) -> Result<Self, ConsoleIdentityError> {
        NonZeroU64::new(value)
            .map(Self)
            .ok_or(ConsoleIdentityError::Zero)
    }

    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

/// Per-session 256-bit capability. It is issued under the per-boot HTTP
/// capability and compared in constant time before sequence state is read.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct ConsoleSessionCapability([u8; 32]);

impl ConsoleSessionCapability {
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    fn constant_time_matches(self, candidate: Self) -> bool {
        let mut difference = 0_u8;
        let mut index = 0;
        while index < self.0.len() {
            difference |= self.0[index] ^ candidate.0[index];
            index += 1;
        }
        difference == 0
    }
}

impl fmt::Debug for ConsoleSessionCapability {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ConsoleSessionCapability([REDACTED])")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ConsoleSourceSequence(NonZeroU64);

impl ConsoleSourceSequence {
    pub fn parse(value: u64) -> Result<Self, ConsoleIdentityError> {
        NonZeroU64::new(value)
            .map(Self)
            .ok_or(ConsoleIdentityError::Zero)
    }

    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ConsoleIdempotencyKey(NonZeroU64);

impl ConsoleIdempotencyKey {
    pub fn parse(value: u64) -> Result<Self, ConsoleIdentityError> {
        NonZeroU64::new(value)
            .map(Self)
            .ok_or(ConsoleIdentityError::Zero)
    }

    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConsoleIdentityError {
    Zero,
}

impl fmt::Display for ConsoleIdentityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("console identity must be nonzero")
    }
}

impl std::error::Error for ConsoleIdentityError {}

/// Private downstream identity. Clients never deserialize or choose it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ConsoleDownstreamRequestId(NonZeroU64);

impl ConsoleDownstreamRequestId {
    pub const fn get(self) -> u64 {
        self.0.get()
    }

    fn as_agent_id(self) -> AgentControlRequestId {
        AgentControlRequestId::from_console_sequence(self.0)
    }

    pub(crate) const fn from_nonzero_for_http(value: NonZeroU64) -> Self {
        Self(value)
    }
}

macro_rules! serialize_nonzero_u64_as_decimal_string {
    ($type:ty) => {
        impl Serialize for $type {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: Serializer,
            {
                serializer.serialize_str(&self.get().to_string())
            }
        }
    };
}

serialize_nonzero_u64_as_decimal_string!(ConsoleSessionId);
serialize_nonzero_u64_as_decimal_string!(ConsoleSourceSequence);
serialize_nonzero_u64_as_decimal_string!(ConsoleIdempotencyKey);
serialize_nonzero_u64_as_decimal_string!(ConsoleDownstreamRequestId);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConsoleSourceKind {
    Operator,
    Agent,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConsoleAutonomousMode {
    FrontierExplore,
    PointGoal,
}

/// One parsed intent. Raw wheel PWM and camera operations are deliberately
/// absent.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OperatorConsoleIntent {
    Arm,
    Disarm,
    BeginManual,
    ManualVelocity(FiniteManualVelocityV1),
    ReleaseManual,
    AutonomousMapOnly,
    AutonomousFrontierExplore,
    AutonomousPointGoal(MapPointGoalSelection),
    Stop,
    SaveMap,
    SoftwareSafetyStop,
}

impl OperatorConsoleIntent {
    pub const fn kind(self) -> OperatorConsoleIntentKind {
        match self {
            Self::Arm => OperatorConsoleIntentKind::Arm,
            Self::Disarm => OperatorConsoleIntentKind::Disarm,
            Self::BeginManual => OperatorConsoleIntentKind::BeginManual,
            Self::ManualVelocity(_) => OperatorConsoleIntentKind::ManualVelocity,
            Self::ReleaseManual => OperatorConsoleIntentKind::ReleaseManual,
            Self::AutonomousMapOnly => OperatorConsoleIntentKind::AutonomousMapOnly,
            Self::AutonomousFrontierExplore => OperatorConsoleIntentKind::AutonomousFrontierExplore,
            Self::AutonomousPointGoal(_) => OperatorConsoleIntentKind::AutonomousPointGoal,
            Self::Stop => OperatorConsoleIntentKind::Stop,
            Self::SaveMap => OperatorConsoleIntentKind::SaveMap,
            Self::SoftwareSafetyStop => OperatorConsoleIntentKind::SoftwareSafetyStop,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OperatorConsoleIntentKind {
    Arm,
    Disarm,
    BeginManual,
    ManualVelocity,
    ReleaseManual,
    AutonomousMapOnly,
    AutonomousFrontierExplore,
    AutonomousPointGoal,
    Stop,
    SaveMap,
    SoftwareSafetyStop,
    ManualDeadmanStop,
}

/// Command delivered to the sole runtime adapter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OperatorConsoleCommand {
    Arm,
    Disarm,
    BeginManual,
    ManualVelocity {
        sequence: ManualDriveSequence,
        velocity: FiniteManualVelocityV1,
    },
    AutonomousMapOnly,
    AutonomousFrontierExplore,
    AutonomousPointGoal(MapPointGoalSelection),
    Stop {
        cause: ConsoleStopCause,
    },
    SaveMap,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConsoleStopCause {
    ManualRelease,
    ManualDeadman,
    ExplicitGlobalStop,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ConsoleExpectedLifecycleZero {
    ArmAdmission,
    ManualAdmission,
    AutonomousAdmission,
    GlobalStopRequest,
    MappingOnlyRequest,
    DisarmRequest,
}

impl ConsoleExpectedLifecycleZero {
    fn for_intent(intent: OperatorConsoleIntentKind) -> Option<Self> {
        match intent {
            OperatorConsoleIntentKind::Arm => Some(Self::ArmAdmission),
            OperatorConsoleIntentKind::Disarm => Some(Self::DisarmRequest),
            OperatorConsoleIntentKind::BeginManual => Some(Self::ManualAdmission),
            OperatorConsoleIntentKind::AutonomousMapOnly => Some(Self::MappingOnlyRequest),
            OperatorConsoleIntentKind::AutonomousFrontierExplore
            | OperatorConsoleIntentKind::AutonomousPointGoal => Some(Self::AutonomousAdmission),
            OperatorConsoleIntentKind::ReleaseManual
            | OperatorConsoleIntentKind::Stop
            | OperatorConsoleIntentKind::ManualDeadmanStop => Some(Self::GlobalStopRequest),
            OperatorConsoleIntentKind::ManualVelocity
            | OperatorConsoleIntentKind::SaveMap
            | OperatorConsoleIntentKind::SoftwareSafetyStop => None,
        }
    }

    #[cfg(all(feature = "actuation", feature = "agent-runtime", unix))]
    fn matches_owner_reason(
        self,
        actual: super::live_motion_owner::LiveLifecycleZeroReason,
    ) -> bool {
        use super::live_motion_owner::LiveLifecycleZeroReason;

        matches!(
            (self, actual),
            (Self::ArmAdmission, LiveLifecycleZeroReason::ArmAdmission)
                | (
                    Self::ManualAdmission,
                    LiveLifecycleZeroReason::ManualAdmission
                )
                | (
                    Self::AutonomousAdmission,
                    LiveLifecycleZeroReason::AutonomousAdmission
                )
                | (
                    Self::GlobalStopRequest,
                    LiveLifecycleZeroReason::GlobalStopRequest
                )
                | (
                    Self::MappingOnlyRequest,
                    LiveLifecycleZeroReason::MappingOnlyRequest
                )
                | (Self::DisarmRequest, LiveLifecycleZeroReason::DisarmRequest)
        )
    }
}

impl OperatorConsoleCommand {
    fn to_agent_command(self) -> AgentControlCommandV1 {
        match self {
            Self::Arm => AgentControlCommandV1::Arm,
            Self::Disarm => AgentControlCommandV1::Disarm,
            Self::BeginManual => AgentControlCommandV1::BeginManual,
            Self::ManualVelocity { sequence, velocity } => AgentControlCommandV1::ManualVelocity(
                AgentManualVelocityV1::from_console_parts(sequence, velocity),
            ),
            Self::AutonomousMapOnly => AgentControlCommandV1::MapOnly,
            Self::AutonomousFrontierExplore => AgentControlCommandV1::FrontierExplore,
            Self::AutonomousPointGoal(selection) => {
                AgentControlCommandV1::SelectMapPoint(selection)
            }
            Self::Stop { .. } => AgentControlCommandV1::Stop,
            Self::SaveMap => AgentControlCommandV1::SaveMap,
        }
    }

    pub const fn kind(self) -> OperatorConsoleIntentKind {
        match self {
            Self::Arm => OperatorConsoleIntentKind::Arm,
            Self::Disarm => OperatorConsoleIntentKind::Disarm,
            Self::BeginManual => OperatorConsoleIntentKind::BeginManual,
            Self::ManualVelocity { .. } => OperatorConsoleIntentKind::ManualVelocity,
            Self::AutonomousMapOnly => OperatorConsoleIntentKind::AutonomousMapOnly,
            Self::AutonomousFrontierExplore => OperatorConsoleIntentKind::AutonomousFrontierExplore,
            Self::AutonomousPointGoal(_) => OperatorConsoleIntentKind::AutonomousPointGoal,
            Self::Stop {
                cause: ConsoleStopCause::ManualRelease,
            } => OperatorConsoleIntentKind::ReleaseManual,
            Self::Stop {
                cause: ConsoleStopCause::ManualDeadman,
            } => OperatorConsoleIntentKind::ManualDeadmanStop,
            Self::Stop {
                cause: ConsoleStopCause::ExplicitGlobalStop,
            } => OperatorConsoleIntentKind::Stop,
            Self::SaveMap => OperatorConsoleIntentKind::SaveMap,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConsoleDispatchSource {
    Session {
        session_id: ConsoleSessionId,
        source_sequence: ConsoleSourceSequence,
    },
    ManualDeadman {
        session_id: ConsoleSessionId,
    },
    InternalFailClosed,
}

/// A normal or deadman dispatch plus a token for reporting the runtime result.
#[derive(Debug)]
pub struct OperatorConsoleDispatch {
    downstream_request_id: ConsoleDownstreamRequestId,
    source: ConsoleDispatchSource,
    received_at: HostMonotonicTimestamp,
    command: OperatorConsoleCommand,
    response: OperatorConsoleResponseToken,
}

impl OperatorConsoleDispatch {
    pub const fn downstream_request_id(&self) -> ConsoleDownstreamRequestId {
        self.downstream_request_id
    }

    pub const fn source(&self) -> ConsoleDispatchSource {
        self.source
    }

    pub const fn received_at(&self) -> HostMonotonicTimestamp {
        self.received_at
    }

    pub const fn command(&self) -> OperatorConsoleCommand {
        self.command
    }

    /// Convert at the in-process adapter boundary without reparsing JSON.
    pub fn agent_request(&self) -> AgentControlRequestV1 {
        AgentControlRequestV1::from_console_parts(
            self.downstream_request_id.as_agent_id(),
            self.command.to_agent_command(),
        )
    }

    pub fn into_parts(
        self,
    ) -> (
        AgentControlRequestV1,
        ConsoleDispatchSource,
        HostMonotonicTimestamp,
        OperatorConsoleResponseToken,
    ) {
        let request = self.agent_request();
        (request, self.source, self.received_at, self.response)
    }
}

/// Highest-priority, one-way process-lifetime software stop signal.
///
/// This is not the independent physical emergency stop.
#[derive(Debug)]
pub struct OperatorConsoleSoftwareSafetyStop {
    downstream_request_id: Option<ConsoleDownstreamRequestId>,
    source: ConsoleDispatchSource,
    received_at: Option<HostMonotonicTimestamp>,
    response: Option<OperatorConsoleResponseToken>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConsoleRequiredSafetyLatch {
    /// Adapter requirement: latch the sole supervisor with
    /// `kiko_supervisor_core::FaultKind::EmergencyStop`, then satisfy its
    /// physical stop/zero obligation. A normal transient `Stop` alone is not
    /// a valid completion.
    SupervisorEmergencyStop,
}

impl OperatorConsoleSoftwareSafetyStop {
    pub const fn downstream_request_id(&self) -> Option<ConsoleDownstreamRequestId> {
        self.downstream_request_id
    }

    pub const fn source(&self) -> ConsoleDispatchSource {
        self.source
    }

    pub const fn received_at(&self) -> Option<HostMonotonicTimestamp> {
        self.received_at
    }

    pub const fn required_latch(&self) -> ConsoleRequiredSafetyLatch {
        ConsoleRequiredSafetyLatch::SupervisorEmergencyStop
    }

    /// Consume the signal into the dedicated emergency-stop adapter boundary.
    /// No ordinary `AgentControlCommandV1::Stop` conversion exists.
    pub fn into_emergency_parts(
        self,
    ) -> (
        Option<ConsoleDownstreamRequestId>,
        ConsoleDispatchSource,
        Option<HostMonotonicTimestamp>,
        ConsoleRequiredSafetyLatch,
        Option<OperatorConsoleResponseToken>,
    ) {
        (
            self.downstream_request_id,
            self.source,
            self.received_at,
            ConsoleRequiredSafetyLatch::SupervisorEmergencyStop,
            self.response,
        )
    }
}

#[derive(Debug)]
pub enum OperatorConsoleIngressItem {
    SoftwareSafetyStop(OperatorConsoleSoftwareSafetyStop),
    Dispatch(OperatorConsoleDispatch),
}

/// Non-clone sole receiver. Drain one item, claim/submit its typed request in
/// the existing dispatcher, then complete the attached response token.
#[derive(Debug)]
pub struct OperatorConsoleIngressReceiver {
    safety_rx: Receiver<OperatorConsoleSoftwareSafetyStop>,
    deadman_rx: Receiver<OperatorConsoleDispatch>,
    urgent_rx: Receiver<OperatorConsoleDispatch>,
    begin_manual_rx: Receiver<OperatorConsoleDispatch>,
    normal_rx: Receiver<OperatorConsoleDispatch>,
    manual_latest_rx: Receiver<OperatorConsoleDispatch>,
    nonpriority_pending: Mutex<ConsoleNonpriorityPending>,
    safety_latched: Arc<AtomicBool>,
    safety_drained: Arc<AtomicBool>,
}

#[derive(Debug, Default)]
struct ConsoleNonpriorityPending {
    begin_manual: Option<OperatorConsoleDispatch>,
    manual_latest: Option<OperatorConsoleDispatch>,
    normal: Option<OperatorConsoleDispatch>,
}

impl OperatorConsoleIngressReceiver {
    pub fn try_next(&self) -> Result<OperatorConsoleIngressItem, TryRecvError> {
        match self.safety_rx.try_recv() {
            Ok(stop) => {
                self.safety_drained.store(true, Ordering::Release);
                self.discard_pending_motion();
                return Ok(OperatorConsoleIngressItem::SoftwareSafetyStop(stop));
            }
            Err(TryRecvError::Disconnected) | Err(TryRecvError::Empty) => {}
        }
        if self.safety_latched.load(Ordering::Acquire) {
            self.discard_pending_motion();
            return Err(TryRecvError::Empty);
        }
        match self.deadman_rx.try_recv() {
            Ok(stop) => {
                self.discard_normal(ConsoleResponseRejectionCode::CancelledByManualDeadman);
                self.discard_begin_manual(ConsoleResponseRejectionCode::CancelledByManualDeadman);
                self.discard_manual(ConsoleResponseRejectionCode::CancelledByManualDeadman);
                return Ok(OperatorConsoleIngressItem::Dispatch(stop));
            }
            Err(TryRecvError::Disconnected) | Err(TryRecvError::Empty) => {}
        }
        match self.urgent_rx.try_recv() {
            Ok(dispatch) => {
                if matches!(
                    dispatch.command,
                    OperatorConsoleCommand::Stop { .. }
                        | OperatorConsoleCommand::Disarm
                        | OperatorConsoleCommand::AutonomousMapOnly
                ) {
                    self.discard_normal(ConsoleResponseRejectionCode::CancelledByPriorityStop);
                    self.discard_begin_manual(
                        ConsoleResponseRejectionCode::CancelledByPriorityStop,
                    );
                    self.discard_manual(ConsoleResponseRejectionCode::CancelledByPriorityStop);
                }
                return Ok(OperatorConsoleIngressItem::Dispatch(dispatch));
            }
            Err(TryRecvError::Disconnected) | Err(TryRecvError::Empty) => {}
        }
        let mut pending = recover_lock(&self.nonpriority_pending);
        if pending.begin_manual.is_none() {
            pending.begin_manual = self.begin_manual_rx.try_recv().ok();
        }
        while let Ok(newest) = self.manual_latest_rx.try_recv() {
            if let Some(superseded) = pending.manual_latest.replace(newest) {
                superseded
                    .response
                    .reject(ConsoleResponseRejectionCode::SupersededByNewerManualDesiredState);
            }
        }
        if pending.normal.is_none() {
            pending.normal = self.normal_rx.try_recv().ok();
        }
        let minimum = [
            pending
                .begin_manual
                .as_ref()
                .map(|dispatch| (dispatch.downstream_request_id.get(), 0_u8)),
            pending
                .manual_latest
                .as_ref()
                .map(|dispatch| (dispatch.downstream_request_id.get(), 1_u8)),
            pending
                .normal
                .as_ref()
                .map(|dispatch| (dispatch.downstream_request_id.get(), 2_u8)),
        ]
        .into_iter()
        .flatten()
        .min_by_key(|(id, _)| *id);
        match minimum {
            Some((_, 0)) => pending
                .begin_manual
                .take()
                .map(OperatorConsoleIngressItem::Dispatch)
                .ok_or(TryRecvError::Empty),
            Some((_, 1)) => pending
                .manual_latest
                .take()
                .map(OperatorConsoleIngressItem::Dispatch)
                .ok_or(TryRecvError::Empty),
            Some((_, 2)) => pending
                .normal
                .take()
                .map(OperatorConsoleIngressItem::Dispatch)
                .ok_or(TryRecvError::Empty),
            Some(_) | None => Err(TryRecvError::Empty),
        }
    }

    pub(crate) fn discard_queued_for_external_priority_stop(
        &self,
        reason: ConsoleResponseRejectionCode,
    ) {
        // Match the conservative console-originated urgent barrier: normal
        // commands, pending manual acquisition, and desired manual state may
        // already be cached inside this receiver rather than the adapter's
        // submitted-correlation table.
        self.discard_normal(reason);
        self.discard_begin_manual(reason);
        self.discard_manual(reason);
    }

    fn discard_pending_motion(&self) {
        self.discard_normal(ConsoleResponseRejectionCode::CancelledBySoftwareSafetyStop);
        self.discard_begin_manual(ConsoleResponseRejectionCode::CancelledBySoftwareSafetyStop);
        self.discard_manual(ConsoleResponseRejectionCode::CancelledBySoftwareSafetyStop);
        while let Ok(dispatch) = self.urgent_rx.try_recv() {
            dispatch
                .response
                .reject(ConsoleResponseRejectionCode::CancelledBySoftwareSafetyStop);
        }
        while let Ok(dispatch) = self.deadman_rx.try_recv() {
            dispatch
                .response
                .reject(ConsoleResponseRejectionCode::CancelledBySoftwareSafetyStop);
        }
    }

    fn discard_normal(&self, reason: ConsoleResponseRejectionCode) {
        if let Some(dispatch) = recover_lock(&self.nonpriority_pending).normal.take() {
            dispatch.response.reject(reason);
        }
        while let Ok(dispatch) = self.normal_rx.try_recv() {
            dispatch.response.reject(reason);
        }
    }

    fn discard_manual(&self, reason: ConsoleResponseRejectionCode) {
        if let Some(dispatch) = recover_lock(&self.nonpriority_pending).manual_latest.take() {
            dispatch.response.reject(reason);
        }
        while let Ok(dispatch) = self.manual_latest_rx.try_recv() {
            dispatch.response.reject(reason);
        }
    }

    fn discard_begin_manual(&self, reason: ConsoleResponseRejectionCode) {
        if let Some(dispatch) = recover_lock(&self.nonpriority_pending).begin_manual.take() {
            dispatch.response.reject(reason);
        }
        while let Ok(dispatch) = self.begin_manual_rx.try_recv() {
            dispatch.response.reject(reason);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ConsoleRequestedOwner {
    Manual {
        session_id: ConsoleSessionId,
        authority_generation: ConsoleDownstreamRequestId,
        deadman_deadline_host_monotonic_ns: ConsoleHostTimestampNs,
    },
    Autonomous {
        session_id: ConsoleSessionId,
        authority_generation: ConsoleDownstreamRequestId,
        mode: ConsoleAutonomousMode,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConsoleActualAuthoritySource {
    Operator,
    Agent,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConsoleActualAuthorityMode {
    Manual,
    FrontierExplore,
    PointGoal,
}

/// Exact supervisor authority token retained by the sole motion owner.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct ConsoleActualAuthority {
    pub source: ConsoleActualAuthoritySource,
    pub mode: ConsoleActualAuthorityMode,
    #[serde(serialize_with = "serialize_u64_as_decimal_string")]
    pub authority_lease_id: u64,
    #[serde(serialize_with = "serialize_optional_u64_as_decimal_string")]
    pub console_downstream_request_id: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OperatorConsoleLimits {
    maximum_sessions: NonZeroUsize,
    normal_queue_capacity: NonZeroUsize,
    idempotency_records_per_session: NonZeroUsize,
    response_records: NonZeroUsize,
    snapshot_subscribers: NonZeroUsize,
    manual_deadman_ms: NonZeroU64,
}

impl OperatorConsoleLimits {
    pub fn parse(
        maximum_sessions: usize,
        normal_queue_capacity: usize,
        idempotency_records_per_session: usize,
        response_records: usize,
        snapshot_subscribers: usize,
        manual_deadman_ms: u64,
    ) -> Result<Self, OperatorConsoleLimitsError> {
        let bounded_usize =
            |field, value, maximum| -> Result<NonZeroUsize, OperatorConsoleLimitsError> {
                let value =
                    NonZeroUsize::new(value).ok_or(OperatorConsoleLimitsError::Zero { field })?;
                if value.get() > maximum {
                    return Err(OperatorConsoleLimitsError::TooLarge {
                        field,
                        actual: value.get() as u64,
                        maximum: maximum as u64,
                    });
                }
                Ok(value)
            };
        let manual_deadman_ms =
            NonZeroU64::new(manual_deadman_ms).ok_or(OperatorConsoleLimitsError::Zero {
                field: OperatorConsoleLimitField::ManualDeadmanMs,
            })?;
        if manual_deadman_ms.get() > 5_000 {
            return Err(OperatorConsoleLimitsError::TooLarge {
                field: OperatorConsoleLimitField::ManualDeadmanMs,
                actual: manual_deadman_ms.get(),
                maximum: 5_000,
            });
        }
        Ok(Self {
            maximum_sessions: bounded_usize(
                OperatorConsoleLimitField::Sessions,
                maximum_sessions,
                MAX_OPERATOR_CONSOLE_SESSIONS,
            )?,
            normal_queue_capacity: bounded_usize(
                OperatorConsoleLimitField::NormalQueue,
                normal_queue_capacity,
                MAX_OPERATOR_CONSOLE_QUEUE_CAPACITY,
            )?,
            idempotency_records_per_session: bounded_usize(
                OperatorConsoleLimitField::IdempotencyRecords,
                idempotency_records_per_session,
                MAX_OPERATOR_CONSOLE_IDEMPOTENCY_RECORDS,
            )?,
            response_records: bounded_usize(
                OperatorConsoleLimitField::ResponseRecords,
                response_records,
                MAX_OPERATOR_CONSOLE_RESPONSE_RECORDS,
            )?,
            snapshot_subscribers: bounded_usize(
                OperatorConsoleLimitField::SnapshotSubscribers,
                snapshot_subscribers,
                MAX_OPERATOR_CONSOLE_SUBSCRIBERS,
            )?,
            manual_deadman_ms,
        })
    }

    pub fn production_default() -> Self {
        Self::parse(16, 64, 32, 256, 16, 250).expect("static console limits are valid")
    }
}

impl Default for OperatorConsoleLimits {
    fn default() -> Self {
        Self::production_default()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperatorConsoleLimitField {
    Sessions,
    NormalQueue,
    IdempotencyRecords,
    ResponseRecords,
    SnapshotSubscribers,
    ManualDeadmanMs,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperatorConsoleLimitsError {
    Zero {
        field: OperatorConsoleLimitField,
    },
    TooLarge {
        field: OperatorConsoleLimitField,
        actual: u64,
        maximum: u64,
    },
}

impl fmt::Display for OperatorConsoleLimitsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid operator-console limit: {self:?}")
    }
}

impl std::error::Error for OperatorConsoleLimitsError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum OperatorConsoleSubmitOutcome {
    AcceptedForProcessing {
        downstream_request_id: ConsoleDownstreamRequestId,
    },
    IdempotentReplay {
        downstream_request_id: ConsoleDownstreamRequestId,
    },
    SoftwareSafetyStopLatched {
        downstream_request_id: ConsoleDownstreamRequestId,
    },
}

impl OperatorConsoleSubmitOutcome {
    pub const fn downstream_request_id(self) -> ConsoleDownstreamRequestId {
        match self {
            Self::AcceptedForProcessing {
                downstream_request_id,
            }
            | Self::IdempotentReplay {
                downstream_request_id,
            }
            | Self::SoftwareSafetyStopLatched {
                downstream_request_id,
            } => downstream_request_id,
        }
    }

    /// Queue/latch acceptance is never physical application.
    pub const fn is_applied(self) -> bool {
        false
    }
}

#[derive(Debug)]
pub enum OperatorConsoleSubmitError {
    UnknownSession(ConsoleSessionId),
    SessionCapabilityMismatch,
    SessionCapacityReached {
        maximum: usize,
    },
    SourceSequenceNotIncreasing {
        previous: ConsoleSourceSequence,
        current: ConsoleSourceSequence,
    },
    IdempotencyConflict(ConsoleIdempotencyKey),
    SoftwareSafetyStopLatched,
    AuthorityConflict {
        requested_by: ConsoleSessionId,
        held_by: ConsoleSessionId,
    },
    ManualAuthorityRequired(ConsoleSessionId),
    NormalQueueFull,
    RuntimeAdapterDisconnected,
    DownstreamSequenceExhausted,
    ManualSequenceExhausted,
    DeadmanDeadlineOverflow,
    ResponseCapacityReached,
    StopPending,
}

impl fmt::Display for OperatorConsoleSubmitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "operator-console submission failed: {self:?}")
    }
}

impl std::error::Error for OperatorConsoleSubmitError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConsoleRuntimeResponseState {
    AcceptedForProcessing,
    ActiveAuthority,
    Completed,
    Cancelled,
    Rejected,
}

impl ConsoleRuntimeResponseState {
    const fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Cancelled | Self::Rejected)
    }
}

/// Exact physical evidence supplied by the sole runtime/controller path.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct ConsoleAppliedReceipt {
    controller_uid: [u8; 12],
    #[serde(serialize_with = "serialize_u64_as_decimal_string")]
    boot_id: u64,
    control_epoch: u32,
    sequence: u32,
    result_code: ConsoleCommandResultCode,
    requested_lease_ms: u16,
    requested_left_timer_pwm_percent: i8,
    requested_right_timer_pwm_percent: i8,
    applied_left_timer_pwm_percent: i8,
    applied_right_timer_pwm_percent: i8,
    output_state: ConsoleControllerOutputState,
    controller_applied_at_wrapping_ms: u32,
    controller_expires_at_wrapping_ms: u32,
    remaining_lease_ms: u16,
    controller_fault_bits: u32,
    sent_at_host_monotonic_ns: ConsoleHostMonotonicNs,
    acknowledged_at_host_monotonic_ns: ConsoleHostMonotonicNs,
    known_active_through_exclusive_host_monotonic_ns: ConsoleHostMonotonicNs,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConsoleControllerOutputState {
    Disabled,
    ZeroPwm,
    NonzeroPwm,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConsoleCommandResultCode {
    AppliedNew,
    DuplicateCached,
    Stopped,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConsoleHostMonotonicNs(u128);

impl ConsoleHostMonotonicNs {
    pub const fn as_u128(self) -> u128 {
        self.0
    }
}

impl Serialize for ConsoleHostMonotonicNs {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // JSON numbers cannot preserve arbitrary u128 host-clock values in a
        // browser. A decimal string is exact.
        serializer.serialize_str(&self.0.to_string())
    }
}

#[cfg(feature = "actuation")]
impl ConsoleAppliedReceipt {
    /// Project only an already sealed, verified command-client receipt. There
    /// is intentionally no public primitive-field constructor.
    pub fn from_verified(
        receipt: &robot_command_client::AppliedCommandReceipt,
    ) -> Result<Self, ConsoleReceiptProjectionError> {
        use robot_protocol::v2::{HostCommandResultCode, OutputState};

        let session = receipt.controller_session();
        let result = receipt.verified_host_result();
        let requested = result.requested_timer_pwm;
        let applied = receipt.applied_timer_pwm();
        let output_state = match receipt.output_state() {
            OutputState::Disabled => ConsoleControllerOutputState::Disabled,
            OutputState::ZeroPwm => ConsoleControllerOutputState::ZeroPwm,
            OutputState::NonzeroPwm => ConsoleControllerOutputState::NonzeroPwm,
        };
        let result_code = match receipt.result() {
            HostCommandResultCode::AppliedNew => ConsoleCommandResultCode::AppliedNew,
            HostCommandResultCode::DuplicateCached => ConsoleCommandResultCode::DuplicateCached,
            HostCommandResultCode::Stopped => ConsoleCommandResultCode::Stopped,
            other => {
                return Err(ConsoleReceiptProjectionError::NonApplicationResult {
                    debug_code: other as u8,
                });
            }
        };
        Ok(Self {
            controller_uid: *session.controller_uid().as_bytes(),
            boot_id: session.boot_id().get(),
            control_epoch: session.control_epoch().get(),
            sequence: receipt.sequence().get(),
            result_code,
            requested_lease_ms: receipt.requested_lease().get(),
            requested_left_timer_pwm_percent: requested.left().get(),
            requested_right_timer_pwm_percent: requested.right().get(),
            applied_left_timer_pwm_percent: applied.left().get(),
            applied_right_timer_pwm_percent: applied.right().get(),
            output_state,
            controller_applied_at_wrapping_ms: receipt.controller_applied_at().get(),
            controller_expires_at_wrapping_ms: receipt.controller_expires_at().get(),
            remaining_lease_ms: receipt.remaining_lease_at_server_emission().get(),
            controller_fault_bits: receipt.controller_faults().bits(),
            sent_at_host_monotonic_ns: ConsoleHostMonotonicNs(
                receipt.sent_at().nanos_since_clock_start(),
            ),
            acknowledged_at_host_monotonic_ns: ConsoleHostMonotonicNs(
                receipt.acknowledged_at().nanos_since_clock_start(),
            ),
            known_active_through_exclusive_host_monotonic_ns: ConsoleHostMonotonicNs(
                receipt
                    .known_active_through_exclusive()
                    .nanos_since_clock_start(),
            ),
        })
    }
}

#[cfg(feature = "actuation")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConsoleReceiptProjectionError {
    NonApplicationResult { debug_code: u8 },
}

#[cfg(feature = "actuation")]
impl fmt::Display for ConsoleReceiptProjectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "cannot project sealed applied receipt: {self:?}")
    }
}

#[cfg(feature = "actuation")]
impl std::error::Error for ConsoleReceiptProjectionError {}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct OperatorConsoleResponseRecord {
    pub downstream_request_id: ConsoleDownstreamRequestId,
    pub intent: OperatorConsoleIntentKind,
    pub state: ConsoleRuntimeResponseState,
    pub applied: bool,
    pub exact_receipt: Option<ConsoleExactReceipt>,
    pub rejection_code: Option<ConsoleResponseRejectionCode>,
    pub source_session_id: Option<ConsoleSessionId>,
    #[serde(skip)]
    idempotency_pinned: bool,
    #[serde(skip)]
    http_observed: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ConsoleExactReceipt {
    AppliedCommand {
        receipt: ConsoleAppliedReceipt,
    },
    Disarm {
        receipt: ConsoleDisarmReceipt,
    },
    SoftwareEmergencyStop {
        receipt: ConsoleSoftwareEmergencyStopReceipt,
    },
}

/// Exact subset available from the sole live owner's unforgeable software
/// emergency-stop outcome. It deliberately does not invent command-client
/// send/ack timing that the owner outcome does not retain.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct ConsoleSoftwareEmergencyStopReceipt {
    controller_uid: [u8; 12],
    #[serde(serialize_with = "serialize_u64_as_decimal_string")]
    boot_id: u64,
    control_epoch: u32,
    sequence: u32,
    result_code: ConsoleCommandResultCode,
    requested_left_timer_pwm_percent: i8,
    requested_right_timer_pwm_percent: i8,
    applied_left_timer_pwm_percent: i8,
    applied_right_timer_pwm_percent: i8,
    output_state: ConsoleControllerOutputState,
    controller_applied_at_wrapping_ms: u32,
    controller_expires_at_wrapping_ms: u32,
    remaining_lease_ms: u16,
    controller_fault_bits: u32,
    observed_at_host_monotonic_ns: ConsoleHostTimestampNs,
}

#[cfg(all(feature = "actuation", feature = "agent-runtime", unix))]
impl ConsoleSoftwareEmergencyStopReceipt {
    fn from_verified(
        evidence: &super::live_motion_owner::LiveSoftwareEmergencyStopApplied,
    ) -> Result<Self, ConsoleVerifiedCompletionError> {
        use kiko_supervisor_core::FaultKind;
        use robot_protocol::v2::{HostCommandResultCode, OutputState};

        if evidence.fault() != FaultKind::EmergencyStop {
            return Err(ConsoleVerifiedCompletionError::EmergencyStopEvidenceRequired);
        }
        let result = evidence.result();
        if !result.requested_timer_pwm.is_zero()
            || !result.controller_timer_pwm.is_zero()
            || !result.output_state.is_safe()
            || !result.faults.is_clear()
            || !result.result.proves_controller_application()
        {
            return Err(ConsoleVerifiedCompletionError::ExactAppliedZeroRequired);
        }
        let result_code = match result.result {
            HostCommandResultCode::AppliedNew => ConsoleCommandResultCode::AppliedNew,
            HostCommandResultCode::DuplicateCached => ConsoleCommandResultCode::DuplicateCached,
            HostCommandResultCode::Stopped => ConsoleCommandResultCode::Stopped,
            other => {
                return Err(ConsoleVerifiedCompletionError::Projection(
                    ConsoleReceiptProjectionError::NonApplicationResult {
                        debug_code: other as u8,
                    },
                ));
            }
        };
        let output_state = match result.output_state {
            OutputState::Disabled => ConsoleControllerOutputState::Disabled,
            OutputState::ZeroPwm => ConsoleControllerOutputState::ZeroPwm,
            OutputState::NonzeroPwm => ConsoleControllerOutputState::NonzeroPwm,
        };
        Ok(Self {
            controller_uid: *result.controller_uid.as_bytes(),
            boot_id: result.boot_id.get(),
            control_epoch: result.control_epoch.get(),
            sequence: result.sequence.get(),
            result_code,
            requested_left_timer_pwm_percent: result.requested_timer_pwm.left().get(),
            requested_right_timer_pwm_percent: result.requested_timer_pwm.right().get(),
            applied_left_timer_pwm_percent: result.controller_timer_pwm.left().get(),
            applied_right_timer_pwm_percent: result.controller_timer_pwm.right().get(),
            output_state,
            controller_applied_at_wrapping_ms: result.controller_applied_at.get(),
            controller_expires_at_wrapping_ms: result.controller_expires_at.get(),
            remaining_lease_ms: result.remaining_lease.get(),
            controller_fault_bits: result.faults.bits(),
            observed_at_host_monotonic_ns: ConsoleHostTimestampNs::from_host(
                evidence.observed_at(),
            ),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct ConsoleDisarmReceipt {
    controller_uid: [u8; 12],
    #[serde(serialize_with = "serialize_u64_as_decimal_string")]
    observed_boot_id: u64,
    request_id: u32,
    output_state: ConsoleControllerOutputState,
    controller_fault_bits: u32,
    acknowledged_at_host_monotonic_ns: ConsoleHostMonotonicNs,
}

#[cfg(feature = "actuation")]
impl ConsoleDisarmReceipt {
    pub fn from_verified(receipt: &robot_command_client::DisarmReceipt) -> Self {
        use robot_protocol::v2::OutputState;

        let output_state = match receipt.output_state() {
            OutputState::Disabled => ConsoleControllerOutputState::Disabled,
            OutputState::ZeroPwm => ConsoleControllerOutputState::ZeroPwm,
            OutputState::NonzeroPwm => ConsoleControllerOutputState::NonzeroPwm,
        };
        Self {
            controller_uid: *receipt.controller_uid().as_bytes(),
            observed_boot_id: receipt.observed_boot_id().get(),
            request_id: receipt.request_id().get(),
            output_state,
            controller_fault_bits: receipt.controller_faults().bits(),
            acknowledged_at_host_monotonic_ns: ConsoleHostMonotonicNs(
                receipt.acknowledged_at().nanos_since_clock_start(),
            ),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConsoleResponseRejectionCode {
    RuntimeRejected,
    SupersededByNewerManualDesiredState,
    CancelledByPriorityStop,
    CancelledByManualDeadman,
    CancelledBySoftwareSafetyStop,
    AdapterDropped,
    InternalFault,
}

#[derive(Debug)]
struct ResponseLedger {
    maximum: usize,
    records: VecDeque<OperatorConsoleResponseRecord>,
}

impl ResponseLedger {
    fn insert_pending(
        &mut self,
        downstream_request_id: ConsoleDownstreamRequestId,
        intent: OperatorConsoleIntentKind,
        idempotency_pinned: bool,
        critical: bool,
        source_session_id: Option<ConsoleSessionId>,
    ) -> Result<Option<ConsoleDownstreamRequestId>, OperatorConsoleSubmitError> {
        let hard_maximum = self
            .maximum
            .saturating_add(OPERATOR_CONSOLE_CRITICAL_RESPONSE_RESERVE);
        let mut evicted = None;
        if self.records.len() >= self.maximum && !critical {
            let removable = self.records.iter().position(|record| {
                record.state.is_terminal()
                    && !record.idempotency_pinned
                    && record.intent != OperatorConsoleIntentKind::SoftwareSafetyStop
            });
            let Some(removable) = removable else {
                return Err(OperatorConsoleSubmitError::ResponseCapacityReached);
            };
            evicted = self
                .records
                .remove(removable)
                .map(|record| record.downstream_request_id);
        } else if self.records.len() >= hard_maximum {
            let removable = self
                .records
                .iter()
                .position(|record| record.state.is_terminal());
            if let Some(removable) = removable {
                evicted = self
                    .records
                    .remove(removable)
                    .map(|record| record.downstream_request_id);
            } else {
                return Err(OperatorConsoleSubmitError::ResponseCapacityReached);
            }
        }
        self.records.push_back(OperatorConsoleResponseRecord {
            downstream_request_id,
            intent,
            state: ConsoleRuntimeResponseState::AcceptedForProcessing,
            applied: false,
            exact_receipt: None,
            rejection_code: None,
            source_session_id,
            idempotency_pinned,
            http_observed: false,
        });
        Ok(evicted)
    }

    fn update(
        &mut self,
        id: ConsoleDownstreamRequestId,
        state: ConsoleRuntimeResponseState,
        receipt: Option<ConsoleExactReceipt>,
        rejection_code: Option<ConsoleResponseRejectionCode>,
    ) {
        if let Some(record) = self
            .records
            .iter_mut()
            .find(|record| record.downstream_request_id == id)
        {
            record.state = state;
            record.applied = receipt.is_some();
            record.exact_receipt = receipt;
            record.rejection_code = rejection_code;
            record.http_observed = false;
        }
    }

    fn transition_preserving_receipt(
        &mut self,
        id: ConsoleDownstreamRequestId,
        state: ConsoleRuntimeResponseState,
        rejection_code: Option<ConsoleResponseRejectionCode>,
    ) {
        if let Some(record) = self
            .records
            .iter_mut()
            .find(|record| record.downstream_request_id == id)
        {
            record.state = state;
            record.rejection_code = rejection_code;
            record.http_observed = false;
        }
    }

    fn unpin(&mut self, id: ConsoleDownstreamRequestId) {
        if let Some(record) = self
            .records
            .iter_mut()
            .find(|record| record.downstream_request_id == id)
        {
            record.idempotency_pinned = false;
        }
    }

    fn observe_for_http_session(
        &mut self,
        id: ConsoleDownstreamRequestId,
        source_session_id: ConsoleSessionId,
    ) -> Option<OperatorConsoleResponseRecord> {
        let record = self.records.iter_mut().find(|record| {
            record.downstream_request_id == id
                && record.source_session_id == Some(source_session_id)
        })?;
        record.http_observed = true;
        Some(record.clone())
    }

    fn current_record_was_http_observed(&self, id: ConsoleDownstreamRequestId) -> bool {
        self.records
            .iter()
            .find(|record| record.downstream_request_id == id)
            .is_some_and(|record| record.http_observed)
    }
}

/// Runtime-owned completion capability for exactly one downstream request.
#[derive(Debug)]
pub struct OperatorConsoleResponseToken {
    downstream_request_id: ConsoleDownstreamRequestId,
    ledger: Arc<Mutex<ResponseLedger>>,
    shared: Weak<ConsoleShared>,
    owner_acquisition: Option<ConsoleSessionId>,
    requires_applied_zero: bool,
    expected_lifecycle_zero: Option<ConsoleExpectedLifecycleZero>,
    typed_request_key: Option<super::control_socket::AgentControlTypedRequestKey>,
    stop_generation: Option<ConsoleDownstreamRequestId>,
    #[cfg(feature = "actuation")]
    software_safety_stop: bool,
    terminal: bool,
}

#[derive(Debug)]
pub enum OperatorConsoleCompletion {
    Completed,
    Authority(OperatorConsoleAuthorityGeneration),
}

/// Linear lifetime guard for one exact admitted authority generation.
#[derive(Debug)]
pub struct OperatorConsoleAuthorityGeneration {
    shared: Weak<ConsoleShared>,
    ledger: Arc<Mutex<ResponseLedger>>,
    session_id: ConsoleSessionId,
    generation: ConsoleDownstreamRequestId,
    terminal: bool,
}

impl OperatorConsoleAuthorityGeneration {
    pub const fn session_id(&self) -> ConsoleSessionId {
        self.session_id
    }

    pub const fn generation(&self) -> ConsoleDownstreamRequestId {
        self.generation
    }

    pub fn completed(mut self) -> bool {
        let cleared = self.clear_exact_generation();
        recover_lock(&self.ledger).transition_preserving_receipt(
            self.generation,
            ConsoleRuntimeResponseState::Completed,
            None,
        );
        self.terminal = true;
        cleared
    }

    pub fn cancelled(mut self) -> bool {
        let cleared = self.clear_exact_generation();
        recover_lock(&self.ledger).transition_preserving_receipt(
            self.generation,
            ConsoleRuntimeResponseState::Cancelled,
            None,
        );
        self.terminal = true;
        cleared
    }

    fn clear_exact_generation(&self) -> bool {
        let Some(shared) = self.shared.upgrade() else {
            return false;
        };
        let mut state = match shared.state.lock() {
            Ok(state) => state,
            Err(poisoned) => {
                latch_shared_internal_fail_closed(&shared);
                poisoned.into_inner()
            }
        };
        let matches = state.requested_owner.is_some_and(|owner| match owner {
            ConsoleRequestedOwner::Manual {
                session_id,
                authority_generation,
                ..
            }
            | ConsoleRequestedOwner::Autonomous {
                session_id,
                authority_generation,
                ..
            } => session_id == self.session_id && authority_generation == self.generation,
        });
        if matches {
            state.requested_owner = None;
        }
        matches
    }
}

impl Drop for OperatorConsoleAuthorityGeneration {
    fn drop(&mut self) {
        if !self.terminal
            && let Some(shared) = self.shared.upgrade()
        {
            recover_lock(&self.ledger).transition_preserving_receipt(
                self.generation,
                ConsoleRuntimeResponseState::Rejected,
                Some(ConsoleResponseRejectionCode::InternalFault),
            );
            latch_shared_internal_fail_closed(&shared);
        }
    }
}

impl OperatorConsoleResponseToken {
    pub const fn downstream_request_id(&self) -> ConsoleDownstreamRequestId {
        self.downstream_request_id
    }

    /// Consume a response which never crossed the runtime ingress boundary.
    ///
    /// The caller removes the provisional ledger entry separately. This is not
    /// an adapter drop after admission and therefore must not fabricate an
    /// internal safety fault or try to clear an authority generation that was
    /// never installed.
    fn abort_before_runtime_delivery(mut self) {
        self.terminal = true;
    }

    pub fn accepted_for_processing(&self) {
        recover_lock(&self.ledger).update(
            self.downstream_request_id,
            ConsoleRuntimeResponseState::AcceptedForProcessing,
            None,
            None,
        );
    }

    pub(crate) fn bind_typed_request_key(
        mut self,
        key: super::control_socket::AgentControlTypedRequestKey,
    ) -> Result<Self, ConsoleResponseBindError> {
        if self.typed_request_key.is_some() {
            return Err(ConsoleResponseBindError::AlreadyBound);
        }
        self.typed_request_key = Some(key);
        Ok(self)
    }

    /// Record completed handling without inventing physical evidence.
    pub fn completed(
        mut self,
    ) -> Result<OperatorConsoleCompletion, ConsoleResponseCompletionError> {
        if self.requires_applied_zero {
            return Err(ConsoleResponseCompletionError::ExactAppliedZeroRequired);
        }
        recover_lock(&self.ledger).update(
            self.downstream_request_id,
            ConsoleRuntimeResponseState::Completed,
            None,
            None,
        );
        let completion = self.completion_outcome();
        self.terminal = true;
        Ok(completion)
    }

    #[cfg(all(feature = "actuation", feature = "agent-runtime", unix))]
    pub(super) fn completed_with_verified_motion<D>(
        mut self,
        evidence: super::live_motion_owner::LiveMotionApplied<
            robot_command_client::AppliedCommandReceipt,
            D,
        >,
    ) -> Result<(OperatorConsoleCompletion, D), ConsoleVerifiedCompletionError> {
        if !matches!(
            self.expected_lifecycle_zero,
            None | Some(ConsoleExpectedLifecycleZero::AutonomousAdmission)
        ) || self.software_safety_stop
        {
            return Err(ConsoleVerifiedCompletionError::LifecycleEvidenceRequired);
        }
        let (actual_key, receipt, diagnostic) = evidence.into_correlated_parts();
        self.verify_typed_request_key(actual_key)?;
        let receipt = ConsoleAppliedReceipt::from_verified(&receipt)
            .map_err(ConsoleVerifiedCompletionError::Projection)?;
        let state = if self.owner_acquisition.is_some() {
            ConsoleRuntimeResponseState::ActiveAuthority
        } else {
            ConsoleRuntimeResponseState::Completed
        };
        recover_lock(&self.ledger).update(
            self.downstream_request_id,
            state,
            Some(ConsoleExactReceipt::AppliedCommand { receipt }),
            None,
        );
        let completion = self.completion_outcome();
        self.terminal = true;
        Ok((completion, diagnostic))
    }

    /// Complete an exploration request that proved there is no reachable
    /// frontier before acquiring motion authority or applying a command.
    ///
    /// This path deliberately records no physical receipt. It is narrower
    /// than [`Self::completed`]: only an exact typed autonomous-acquisition
    /// token may use it, and the matching requested-owner generation must be
    /// cleared atomically.
    #[cfg(all(feature = "actuation", feature = "agent-runtime", unix))]
    pub(super) fn completed_exploration_without_authority(
        mut self,
        key: super::control_socket::AgentControlTypedRequestKey,
    ) -> Result<(), ConsoleVerifiedCompletionError> {
        if self.expected_lifecycle_zero != Some(ConsoleExpectedLifecycleZero::AutonomousAdmission)
            || self.owner_acquisition.is_none()
            || self.software_safety_stop
        {
            return Err(ConsoleVerifiedCompletionError::UnexpectedAutonomousNoMotionCompletion);
        }
        self.verify_typed_request_key(Some(key))?;
        if !self.clear_owner_acquisition_after_terminal() {
            return Err(ConsoleVerifiedCompletionError::AuthorityGenerationMismatch);
        }
        recover_lock(&self.ledger).update(
            self.downstream_request_id,
            ConsoleRuntimeResponseState::Completed,
            None,
            None,
        );
        self.terminal = true;
        Ok(())
    }

    #[cfg(all(feature = "actuation", feature = "agent-runtime", unix))]
    pub(super) fn completed_with_verified_lifecycle_zero(
        mut self,
        evidence: super::live_motion_owner::LiveLifecycleZeroApplied<
            robot_command_client::AppliedCommandReceipt,
        >,
    ) -> Result<OperatorConsoleCompletion, ConsoleVerifiedCompletionError> {
        let expected = self
            .expected_lifecycle_zero
            .ok_or(ConsoleVerifiedCompletionError::UnexpectedLifecycleEvidence)?;
        let (actual_key, _requested_at, receipt, actual_reason) = evidence.into_correlated_parts();
        self.verify_typed_request_key(actual_key)?;
        let autonomous_completed_immediately = expected
            == ConsoleExpectedLifecycleZero::AutonomousAdmission
            && actual_reason
                == super::live_motion_owner::LiveLifecycleZeroReason::AutonomousRelease;
        if !expected.matches_owner_reason(actual_reason) && !autonomous_completed_immediately {
            return Err(ConsoleVerifiedCompletionError::LifecycleReasonMismatch);
        }
        if !is_verified_exact_safe_zero(&receipt) {
            return Err(ConsoleVerifiedCompletionError::ExactAppliedZeroRequired);
        }
        let receipt = ConsoleAppliedReceipt::from_verified(&receipt)
            .map_err(ConsoleVerifiedCompletionError::Projection)?;
        let state = if self.owner_acquisition.is_some() && !autonomous_completed_immediately {
            ConsoleRuntimeResponseState::ActiveAuthority
        } else {
            ConsoleRuntimeResponseState::Completed
        };
        recover_lock(&self.ledger).update(
            self.downstream_request_id,
            state,
            Some(ConsoleExactReceipt::AppliedCommand { receipt }),
            None,
        );
        self.clear_stop_pending_after_exact_zero();
        let completion = if autonomous_completed_immediately {
            if !self.clear_owner_acquisition_after_terminal() {
                return Err(ConsoleVerifiedCompletionError::AuthorityGenerationMismatch);
            }
            OperatorConsoleCompletion::Completed
        } else {
            self.completion_outcome()
        };
        self.terminal = true;
        Ok(completion)
    }

    #[cfg(all(feature = "actuation", feature = "agent-runtime", unix))]
    pub(super) fn completed_with_verified_emergency_stop(
        mut self,
        evidence: super::live_motion_owner::LiveSoftwareEmergencyStopApplied,
    ) -> Result<(), ConsoleVerifiedCompletionError> {
        if !self.software_safety_stop {
            return Err(ConsoleVerifiedCompletionError::UnexpectedEmergencyStopEvidence);
        }
        self.verify_typed_request_key(Some(evidence.typed_request_key()))?;
        let receipt = ConsoleSoftwareEmergencyStopReceipt::from_verified(&evidence)?;
        recover_lock(&self.ledger).update(
            self.downstream_request_id,
            ConsoleRuntimeResponseState::Completed,
            Some(ConsoleExactReceipt::SoftwareEmergencyStop { receipt }),
            None,
        );
        self.clear_stop_pending_after_exact_zero();
        self.mark_safety_complete();
        self.terminal = true;
        Ok(())
    }

    pub fn reject(mut self, code: ConsoleResponseRejectionCode) {
        recover_lock(&self.ledger).update(
            self.downstream_request_id,
            ConsoleRuntimeResponseState::Rejected,
            None,
            Some(code),
        );
        if let (Some(session_id), Some(shared)) = (self.owner_acquisition, self.shared.upgrade()) {
            let mut state = match shared.state.lock() {
                Ok(state) => state,
                Err(poisoned) => {
                    latch_shared_internal_fail_closed(&shared);
                    poisoned.into_inner()
                }
            };
            let owner_session = state.requested_owner.map(|owner| match owner {
                ConsoleRequestedOwner::Manual { session_id, .. }
                | ConsoleRequestedOwner::Autonomous { session_id, .. } => session_id,
            });
            let owner_generation = state.requested_owner.map(|owner| match owner {
                ConsoleRequestedOwner::Manual {
                    authority_generation,
                    ..
                }
                | ConsoleRequestedOwner::Autonomous {
                    authority_generation,
                    ..
                } => authority_generation,
            });
            if owner_session == Some(session_id)
                && owner_generation == Some(self.downstream_request_id)
            {
                state.requested_owner = None;
            }
        }
        self.terminal = true;
    }

    #[cfg(feature = "actuation")]
    fn clear_stop_pending_after_exact_zero(&self) {
        let Some(stop_generation) = self.stop_generation else {
            return;
        };
        if let Some(shared) = self.shared.upgrade() {
            match shared.state.lock() {
                Ok(mut state) => {
                    if state.stop_pending == Some(stop_generation) {
                        state.stop_pending = None;
                    }
                }
                Err(poisoned) => {
                    latch_shared_internal_fail_closed(&shared);
                    poisoned.into_inner().stop_pending = Some(stop_generation);
                }
            }
        }
    }

    #[cfg(feature = "actuation")]
    fn mark_safety_complete(&self) {
        if self.software_safety_stop
            && let Some(shared) = self.shared.upgrade()
        {
            shared.safety_completed.store(true, Ordering::Release);
        }
    }

    fn completion_outcome(&mut self) -> OperatorConsoleCompletion {
        match self.owner_acquisition.take() {
            Some(session_id) => {
                OperatorConsoleCompletion::Authority(OperatorConsoleAuthorityGeneration {
                    shared: Weak::clone(&self.shared),
                    ledger: Arc::clone(&self.ledger),
                    session_id,
                    generation: self.downstream_request_id,
                    terminal: false,
                })
            }
            None => OperatorConsoleCompletion::Completed,
        }
    }

    fn clear_owner_acquisition_after_terminal(&mut self) -> bool {
        let Some(session_id) = self.owner_acquisition else {
            return false;
        };
        let Some(shared) = self.shared.upgrade() else {
            return false;
        };
        let mut state = match shared.state.lock() {
            Ok(state) => state,
            Err(poisoned) => {
                latch_shared_internal_fail_closed(&shared);
                poisoned.into_inner()
            }
        };
        let exact_generation = state.requested_owner.is_some_and(|owner| match owner {
            ConsoleRequestedOwner::Manual {
                session_id: owner,
                authority_generation,
                ..
            }
            | ConsoleRequestedOwner::Autonomous {
                session_id: owner,
                authority_generation,
                ..
            } => owner == session_id && authority_generation == self.downstream_request_id,
        });
        if exact_generation {
            state.requested_owner = None;
            self.owner_acquisition = None;
        }
        exact_generation
    }

    #[cfg(all(feature = "actuation", feature = "agent-runtime", unix))]
    fn verify_typed_request_key(
        &self,
        actual: Option<super::control_socket::AgentControlTypedRequestKey>,
    ) -> Result<(), ConsoleVerifiedCompletionError> {
        let expected = self
            .typed_request_key
            .ok_or(ConsoleVerifiedCompletionError::TypedRequestKeyRequired)?;
        if actual != Some(expected) {
            return Err(ConsoleVerifiedCompletionError::TypedRequestKeyMismatch);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConsoleResponseBindError {
    AlreadyBound,
}

impl fmt::Display for ConsoleResponseBindError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("console response token is already bound to a typed request")
    }
}

impl std::error::Error for ConsoleResponseBindError {}

#[cfg(feature = "actuation")]
#[derive(Debug)]
pub enum ConsoleVerifiedCompletionError {
    Projection(ConsoleReceiptProjectionError),
    ExactAppliedZeroRequired,
    LifecycleEvidenceRequired,
    UnexpectedLifecycleEvidence,
    LifecycleReasonMismatch,
    TypedRequestKeyRequired,
    TypedRequestKeyMismatch,
    UnexpectedAutonomousNoMotionCompletion,
    AuthorityGenerationMismatch,
    EmergencyStopEvidenceRequired,
    UnexpectedEmergencyStopEvidence,
}

#[cfg(feature = "actuation")]
impl fmt::Display for ConsoleVerifiedCompletionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid console completion evidence: {self:?}")
    }
}

#[cfg(feature = "actuation")]
impl std::error::Error for ConsoleVerifiedCompletionError {}

#[cfg(feature = "actuation")]
fn is_verified_exact_safe_zero(receipt: &robot_command_client::AppliedCommandReceipt) -> bool {
    receipt.verified_host_result().requested_timer_pwm.is_zero() && receipt.is_confirmed_zero()
}

impl Drop for OperatorConsoleResponseToken {
    fn drop(&mut self) {
        if !self.terminal {
            recover_lock(&self.ledger).update(
                self.downstream_request_id,
                ConsoleRuntimeResponseState::Rejected,
                None,
                Some(ConsoleResponseRejectionCode::AdapterDropped),
            );
            if let Some(shared) = self.shared.upgrade() {
                if let Some(session_id) = self.owner_acquisition {
                    let mut state = match shared.state.lock() {
                        Ok(state) => state,
                        Err(poisoned) => {
                            latch_shared_internal_fail_closed(&shared);
                            poisoned.into_inner()
                        }
                    };
                    let exact_generation = state.requested_owner.is_some_and(|owner| match owner {
                        ConsoleRequestedOwner::Manual {
                            session_id: owner,
                            authority_generation,
                            ..
                        }
                        | ConsoleRequestedOwner::Autonomous {
                            session_id: owner,
                            authority_generation,
                            ..
                        } => {
                            owner == session_id
                                && authority_generation == self.downstream_request_id
                        }
                    });
                    if exact_generation {
                        state.requested_owner = None;
                    }
                }
                if self.requires_applied_zero || self.owner_acquisition.is_some() {
                    latch_shared_internal_fail_closed(&shared);
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConsoleResponseCompletionError {
    ExactAppliedZeroRequired,
}

impl fmt::Display for ConsoleResponseCompletionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("this console response requires an exact applied-zero receipt")
    }
}

impl std::error::Error for ConsoleResponseCompletionError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ConsoleSnapshotRevision(NonZeroU64);

impl ConsoleSnapshotRevision {
    pub fn parse(value: u64) -> Result<Self, ConsoleIdentityError> {
        NonZeroU64::new(value)
            .map(Self)
            .ok_or(ConsoleIdentityError::Zero)
    }

    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

serialize_nonzero_u64_as_decimal_string!(ConsoleSnapshotRevision);

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct ConsolePose2 {
    pub x_m: ConsoleFiniteF64,
    pub y_m: ConsoleFiniteF64,
    pub yaw_rad: ConsoleFiniteF64,
}

impl ConsolePose2 {
    pub fn parse(x_m: f64, y_m: f64, yaw_rad: f64) -> Result<Self, ConsoleFiniteF64Error> {
        Ok(Self {
            x_m: ConsoleFiniteF64::parse(x_m)?,
            y_m: ConsoleFiniteF64::parse(y_m)?,
            yaw_rad: ConsoleFiniteF64::parse(yaw_rad)?,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct ConsolePoint2 {
    pub x_m: ConsoleFiniteF64,
    pub y_m: ConsoleFiniteF64,
}

impl ConsolePoint2 {
    pub fn parse(x_m: f64, y_m: f64) -> Result<Self, ConsoleFiniteF64Error> {
        Ok(Self {
            x_m: ConsoleFiniteF64::parse(x_m)?,
            y_m: ConsoleFiniteF64::parse(y_m)?,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct ConsoleGridMetadata {
    pub width: NonZeroU32,
    pub height: NonZeroU32,
    pub resolution_m_per_cell: ConsoleFiniteF64,
    pub origin_x_m: ConsoleFiniteF64,
    pub origin_y_m: ConsoleFiniteF64,
    pub cell_encoding: ConsoleGridCellEncoding,
    pub linearization: ConsoleGridLinearization,
    pub origin_convention: ConsoleGridOriginConvention,
    pub map_axes: ConsoleMapAxes,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConsoleGridCellEncoding {
    Unknown0Free1Occupied2,
}

/// `index = y * width + x`; row zero is the minimum map-Y row and X increases
/// within a row. Canvas renderers must therefore invert their display Y axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConsoleGridLinearization {
    RowMajorXFastRowsIncreasePositiveMapY,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConsoleGridOriginConvention {
    MinimumXYCornerOfCell00,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConsoleMapAxes {
    RightHandedXRightYUp,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ConsoleOccupancyGrid {
    pub map_epoch_id: NonZeroU64,
    pub revision: u64,
    pub metadata: ConsoleGridMetadata,
    /// Stable values: 0 unknown, 1 free, 2 occupied.
    pub cells: Vec<u8>,
}

impl ConsoleOccupancyGrid {
    /// Project one already-validated live occupancy snapshot into the wire
    /// representation. This performs exactly one cell-buffer allocation and
    /// copy, while preserving the snapshot's row order and class IDs.
    pub fn from_snapshot(
        binding: super::CurrentMapEpochBinding,
        snapshot: &crate::dense::occupancy::OccupancyGridSnapshot,
    ) -> Result<Self, ConsoleGridProjectionError> {
        use crate::dense::occupancy::OccupancyRowOrder;

        let observed_map_instance_id = snapshot.map_instance_id();
        if observed_map_instance_id != Some(binding.map_instance_id()) {
            return Err(ConsoleGridProjectionError::MapBindingMismatch {
                expected_map_instance_id: binding.map_instance_id().as_u64(),
                observed_map_instance_id: observed_map_instance_id.map(|id| id.as_u64()),
            });
        }
        match snapshot.row_order() {
            OccupancyRowOrder::IncreasingOccupancyY => {}
        }
        let geometry = snapshot.geometry();
        let expected_cells = geometry.cell_count();
        let source_cells = snapshot.class_ids();
        if source_cells.len() != expected_cells {
            return Err(ConsoleGridProjectionError::SnapshotInvariant {
                expected_cells,
                actual_cells: source_cells.len(),
            });
        }
        if expected_cells > MAX_OPERATOR_CONSOLE_GRID_CELLS {
            return Err(ConsoleGridProjectionError::TooManyCells {
                actual: expected_cells,
                maximum: MAX_OPERATOR_CONSOLE_GRID_CELLS,
            });
        }
        let mut cells = Vec::new();
        cells.try_reserve_exact(expected_cells).map_err(|source| {
            ConsoleGridProjectionError::Allocation {
                cells: expected_cells,
                source,
            }
        })?;
        for (index, class_id) in source_cells.iter().copied().enumerate() {
            if class_id > 2 {
                return Err(ConsoleGridProjectionError::InvalidClassId { index, class_id });
            }
            cells.push(class_id);
        }
        let [origin_x_m, origin_y_m] = geometry.lower_bound_m();
        let width = NonZeroU32::new(geometry.width()).ok_or(
            ConsoleGridProjectionError::SnapshotInvariant {
                expected_cells,
                actual_cells: source_cells.len(),
            },
        )?;
        let height = NonZeroU32::new(geometry.height()).ok_or(
            ConsoleGridProjectionError::SnapshotInvariant {
                expected_cells,
                actual_cells: source_cells.len(),
            },
        )?;
        Ok(Self {
            map_epoch_id: NonZeroU64::new(binding.map_epoch_id().as_u64())
                .expect("recorded map epoch IDs are nonzero"),
            revision: snapshot.revision(),
            metadata: ConsoleGridMetadata {
                width,
                height,
                resolution_m_per_cell: ConsoleFiniteF64::parse(geometry.resolution_m())
                    .map_err(ConsoleGridProjectionError::NumericInvariant)?,
                origin_x_m: ConsoleFiniteF64::parse(origin_x_m)
                    .map_err(ConsoleGridProjectionError::NumericInvariant)?,
                origin_y_m: ConsoleFiniteF64::parse(origin_y_m)
                    .map_err(ConsoleGridProjectionError::NumericInvariant)?,
                cell_encoding: ConsoleGridCellEncoding::Unknown0Free1Occupied2,
                linearization: ConsoleGridLinearization::RowMajorXFastRowsIncreasePositiveMapY,
                origin_convention: ConsoleGridOriginConvention::MinimumXYCornerOfCell00,
                map_axes: ConsoleMapAxes::RightHandedXRightYUp,
            },
            cells,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn parse(
        map_epoch_id: u64,
        revision: u64,
        width: u32,
        height: u32,
        resolution_m_per_cell: f64,
        origin_x_m: f64,
        origin_y_m: f64,
        cells: Vec<u8>,
    ) -> Result<Self, ConsoleGridError> {
        let map_epoch_id = NonZeroU64::new(map_epoch_id).ok_or(ConsoleGridError::ZeroMapEpoch)?;
        let width = NonZeroU32::new(width).ok_or(ConsoleGridError::ZeroDimension)?;
        let height = NonZeroU32::new(height).ok_or(ConsoleGridError::ZeroDimension)?;
        let expected = usize::try_from(width.get())
            .ok()
            .and_then(|width| {
                usize::try_from(height.get())
                    .ok()
                    .and_then(|height| width.checked_mul(height))
            })
            .ok_or(ConsoleGridError::TooManyCells)?;
        if expected > MAX_OPERATOR_CONSOLE_GRID_CELLS {
            return Err(ConsoleGridError::TooManyCells);
        }
        if cells.len() != expected {
            return Err(ConsoleGridError::CellCount {
                expected,
                actual: cells.len(),
            });
        }
        if cells.iter().any(|cell| *cell > 2) {
            return Err(ConsoleGridError::InvalidCell);
        }
        let resolution_m_per_cell =
            ConsoleFiniteF64::parse(resolution_m_per_cell).map_err(ConsoleGridError::Numeric)?;
        if resolution_m_per_cell.get() <= 0.0 {
            return Err(ConsoleGridError::NonPositiveResolution);
        }
        let origin_x_m = ConsoleFiniteF64::parse(origin_x_m).map_err(ConsoleGridError::Numeric)?;
        let origin_y_m = ConsoleFiniteF64::parse(origin_y_m).map_err(ConsoleGridError::Numeric)?;
        let maximum_x_m = resolution_m_per_cell
            .get()
            .mul_add(f64::from(width.get()), origin_x_m.get());
        let maximum_y_m = resolution_m_per_cell
            .get()
            .mul_add(f64::from(height.get()), origin_y_m.get());
        if !maximum_x_m.is_finite() || !maximum_y_m.is_finite() {
            return Err(ConsoleGridError::NonFiniteExtent);
        }
        if maximum_x_m <= origin_x_m.get() || maximum_y_m <= origin_y_m.get() {
            return Err(ConsoleGridError::UnrepresentableExtent);
        }
        Ok(Self {
            map_epoch_id,
            revision,
            metadata: ConsoleGridMetadata {
                width,
                height,
                resolution_m_per_cell,
                origin_x_m,
                origin_y_m,
                cell_encoding: ConsoleGridCellEncoding::Unknown0Free1Occupied2,
                linearization: ConsoleGridLinearization::RowMajorXFastRowsIncreasePositiveMapY,
                origin_convention: ConsoleGridOriginConvention::MinimumXYCornerOfCell00,
                map_axes: ConsoleMapAxes::RightHandedXRightYUp,
            },
            cells,
        })
    }
}

#[derive(Debug, PartialEq)]
pub enum ConsoleGridError {
    ZeroMapEpoch,
    ZeroDimension,
    TooManyCells,
    CellCount { expected: usize, actual: usize },
    InvalidCell,
    NonPositiveResolution,
    NonFiniteExtent,
    UnrepresentableExtent,
    Numeric(ConsoleFiniteF64Error),
}

impl fmt::Display for ConsoleGridError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid console occupancy grid: {self:?}")
    }
}

impl std::error::Error for ConsoleGridError {}

#[derive(Debug)]
pub enum ConsoleGridProjectionError {
    MapBindingMismatch {
        expected_map_instance_id: u64,
        observed_map_instance_id: Option<u64>,
    },
    SnapshotInvariant {
        expected_cells: usize,
        actual_cells: usize,
    },
    TooManyCells {
        actual: usize,
        maximum: usize,
    },
    InvalidClassId {
        index: usize,
        class_id: u8,
    },
    NumericInvariant(ConsoleFiniteF64Error),
    Allocation {
        cells: usize,
        source: std::collections::TryReserveError,
    },
}

impl fmt::Display for ConsoleGridProjectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "could not project occupancy snapshot into operator console: {self:?}"
        )
    }
}

impl std::error::Error for ConsoleGridProjectionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::NumericInvariant(source) => Some(source),
            Self::Allocation { source, .. } => Some(source),
            Self::MapBindingMismatch { .. }
            | Self::SnapshotInvariant { .. }
            | Self::TooManyCells { .. }
            | Self::InvalidClassId { .. } => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConsoleLocalization {
    Localized,
    Lost,
    Unavailable,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ConsoleMapSnapshot {
    #[serde(serialize_with = "serialize_nonzero_u64_as_decimal_string")]
    pub map_epoch_id: NonZeroU64,
    #[serde(serialize_with = "serialize_u64_as_decimal_string")]
    pub revision: u64,
    pub localization: ConsoleLocalization,
    pub grid: Option<ConsoleGridMetadata>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ConsoleNavigationSnapshot {
    pub pose: Option<ConsolePose2>,
    pub path: Option<Vec<ConsolePoint2>>,
    pub goal: Option<ConsolePoint2>,
    pub mpc_predicted_path: Option<Vec<ConsolePoint2>>,
    #[serde(serialize_with = "serialize_optional_u64_as_decimal_string")]
    pub solver_duration_ns: Option<u64>,
    #[serde(serialize_with = "serialize_optional_u64_as_decimal_string")]
    pub control_tick_lateness_ns: Option<u64>,
}

impl ConsoleNavigationSnapshot {
    pub fn parse_path(points: Vec<ConsolePoint2>) -> Result<Vec<ConsolePoint2>, ConsolePathError> {
        if points.len() > MAX_OPERATOR_CONSOLE_PATH_POINTS {
            return Err(ConsolePathError {
                actual: points.len(),
                maximum: MAX_OPERATOR_CONSOLE_PATH_POINTS,
            });
        }
        Ok(points)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConsolePathError {
    pub actual: usize,
    pub maximum: usize,
}

impl fmt::Display for ConsolePathError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "console path has {} points; maximum is {}",
            self.actual, self.maximum
        )
    }
}

impl std::error::Error for ConsolePathError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct ConsoleRequestedCommand {
    pub downstream_request_id: ConsoleDownstreamRequestId,
    pub kind: OperatorConsoleIntentKind,
}

/// Exact controller-domain request observed after MPC/safety conversion. It is
/// not inferred from the body-frame console intent.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct ConsoleRequestedActuation {
    downstream_request_id: Option<ConsoleDownstreamRequestId>,
    #[serde(serialize_with = "serialize_optional_u64_as_decimal_string")]
    decision_id: Option<u64>,
    left_timer_pwm_percent: i8,
    right_timer_pwm_percent: i8,
}

impl ConsoleRequestedActuation {
    /// Project only an already checked shadow command record.
    pub fn from_checked_record(
        downstream_request_id: Option<ConsoleDownstreamRequestId>,
        record: super::ShadowCommandRecord,
    ) -> Self {
        let pwm = record.pwm();
        Self {
            downstream_request_id,
            decision_id: Some(record.decision_id().as_u64()),
            left_timer_pwm_percent: pwm.left().get(),
            right_timer_pwm_percent: pwm.right().get(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct ConsoleStopCertainty(ConsoleStopCertaintyKind);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum ConsoleStopCertaintyKind {
    ConfirmedAppliedZero,
    ControllerReportedSafe,
    Uncertain,
}

impl ConsoleStopCertainty {
    pub const fn uncertain() -> Self {
        Self(ConsoleStopCertaintyKind::Uncertain)
    }

    #[cfg(feature = "actuation")]
    pub fn from_verified_applied(receipt: &robot_command_client::AppliedCommandReceipt) -> Self {
        let result = receipt.verified_host_result();
        if result.requested_timer_pwm.is_zero() && receipt.is_confirmed_zero() {
            Self(ConsoleStopCertaintyKind::ConfirmedAppliedZero)
        } else if result.controller_timer_pwm.is_zero()
            && result.output_state.is_safe()
            && result.faults.is_clear()
            && result.result.proves_controller_application()
        {
            Self(ConsoleStopCertaintyKind::ControllerReportedSafe)
        } else {
            Self::uncertain()
        }
    }

    #[cfg(feature = "actuation")]
    pub fn from_verified_disarm(receipt: &robot_command_client::DisarmReceipt) -> Self {
        if receipt.output_state().is_safe() && receipt.controller_faults().is_clear() {
            Self(ConsoleStopCertaintyKind::ControllerReportedSafe)
        } else {
            Self::uncertain()
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConsoleSafetySignalState {
    NotLatched,
    PendingRuntimeDrain,
    RuntimeDrainedAwaitingCompletion,
    CompletedFaultLatched,
    RuntimeAdapterDisconnected,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConsoleHealth {
    Ready,
    Degraded,
    Faulted,
    Unavailable,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConsolePhysicalEmergencyStopState {
    Released,
    Engaged,
    Unavailable,
    Faulted,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct ConsoleSubsystemHealth {
    pub stm32: Option<ConsoleHealth>,
    pub head: Option<ConsoleHealth>,
    pub eyes: Option<ConsoleHealth>,
    pub oak: Option<ConsoleHealth>,
    pub slam: Option<ConsoleHealth>,
}

/// Requested provider names retain `auto`; selected provider names do not.
/// This separation prevents an unresolved request from being serialized as
/// runtime evidence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConsoleRequestedInferenceBackend {
    Auto,
    Cpu,
    CoremlGpu,
    Cuda,
    #[serde(rename = "tensorrt")]
    TensorRt,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConsoleSelectedInferenceBackend {
    Cpu,
    CoremlGpu,
    Cuda,
    #[serde(rename = "tensorrt")]
    TensorRt,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct ConsoleInferenceSelection {
    pub requested: ConsoleRequestedInferenceBackend,
    pub selected: ConsoleSelectedInferenceBackend,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct ConsoleInferenceRuntime {
    pub superpoint: ConsoleInferenceSelection,
    pub lightglue: ConsoleInferenceSelection,
}

/// Exact measurement window. Consumers derive hertz as
/// `(successful_completions - 1) * 1e9 / span_ns`; neither endpoint is a
/// nominal camera rate or a benchmark claim.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct ConsoleSlamRateWindow {
    pub successful_completions: u8,
    #[serde(serialize_with = "serialize_u64_as_decimal_string")]
    pub span_ns: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct ConsoleSlamSnapshot {
    pub inference: ConsoleInferenceRuntime,
    #[serde(serialize_with = "serialize_u64_as_decimal_string")]
    pub started_pairs: u64,
    #[serde(serialize_with = "serialize_u64_as_decimal_string")]
    pub successful_pairs: u64,
    #[serde(serialize_with = "serialize_u64_as_decimal_string")]
    pub recoverable_failures: u64,
    #[serde(serialize_with = "serialize_u64_as_decimal_string")]
    pub fatal_failures: u64,
    #[serde(serialize_with = "serialize_optional_u64_as_decimal_string")]
    pub last_successful_source_arrival_host_monotonic_ns: Option<u64>,
    #[serde(serialize_with = "serialize_optional_u64_as_decimal_string")]
    pub last_successful_completion_host_monotonic_ns: Option<u64>,
    pub rate_window: Option<ConsoleSlamRateWindow>,
}

/// UI command magnitudes projected from the already admitted manual-control
/// envelope. The browser must not invent drive limits.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct ConsoleManualCommandEnvelope {
    pub max_abs_forward_velocity_mps: ConsoleFiniteF64,
    pub max_abs_yaw_rate_rad_s: ConsoleFiniteF64,
    pub command_forward_velocity_mps: ConsoleFiniteF64,
    pub command_yaw_rate_rad_s: ConsoleFiniteF64,
}

impl ConsoleManualCommandEnvelope {
    pub fn parse(
        max_abs_forward_velocity_mps: f64,
        max_abs_yaw_rate_rad_s: f64,
        command_forward_velocity_mps: f64,
        command_yaw_rate_rad_s: f64,
    ) -> Result<Self, ConsoleManualCommandEnvelopeError> {
        let parsed = Self {
            max_abs_forward_velocity_mps: ConsoleFiniteF64::parse(max_abs_forward_velocity_mps)
                .map_err(ConsoleManualCommandEnvelopeError::Numeric)?,
            max_abs_yaw_rate_rad_s: ConsoleFiniteF64::parse(max_abs_yaw_rate_rad_s)
                .map_err(ConsoleManualCommandEnvelopeError::Numeric)?,
            command_forward_velocity_mps: ConsoleFiniteF64::parse(command_forward_velocity_mps)
                .map_err(ConsoleManualCommandEnvelopeError::Numeric)?,
            command_yaw_rate_rad_s: ConsoleFiniteF64::parse(command_yaw_rate_rad_s)
                .map_err(ConsoleManualCommandEnvelopeError::Numeric)?,
        };
        if parsed.max_abs_forward_velocity_mps.get() <= 0.0
            || parsed.max_abs_yaw_rate_rad_s.get() <= 0.0
            || parsed.command_forward_velocity_mps.get() <= 0.0
            || parsed.command_yaw_rate_rad_s.get() <= 0.0
        {
            return Err(ConsoleManualCommandEnvelopeError::NonPositive);
        }
        if parsed.command_forward_velocity_mps > parsed.max_abs_forward_velocity_mps
            || parsed.command_yaw_rate_rad_s > parsed.max_abs_yaw_rate_rad_s
        {
            return Err(ConsoleManualCommandEnvelopeError::CommandExceedsMaximum);
        }
        Ok(parsed)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ConsoleManualCommandEnvelopeError {
    Numeric(ConsoleFiniteF64Error),
    NonPositive,
    CommandExceedsMaximum,
}

impl fmt::Display for ConsoleManualCommandEnvelopeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid console manual command envelope: {self:?}"
        )
    }
}

impl std::error::Error for ConsoleManualCommandEnvelopeError {}

/// Explicit reason why the live dashboard is terminal and must not invite
/// further authority requests.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConsoleTerminalReason {
    /// Capture is closed and the causal pipeline is draining so one exact map
    /// revision can be bound to its finalized restart dataset.
    FinalizingWarmRestartCheckpoint,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConsoleCheckpointLocalizationEvidence {
    NotClaimed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ConsoleTerminalState {
    ControlEnding {
        reason: ConsoleTerminalReason,
        /// A restart checkpoint proves replay inputs only. Fresh-camera
        /// localization must be established independently on the next run.
        current_camera_localization: ConsoleCheckpointLocalizationEvidence,
    },
}

/// The only motion-authority policy under which a console snapshot was
/// produced. Clients must match this field explicitly; the absence of a
/// mode-specific extension is never evidence of production authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConsoleRuntimeAuthorityKind {
    ProductionExternalInterlocks,
    AttendedNavigationTrial,
    WheelsOffQualification,
}

/// Immutable latest-only observational state. `None` means unknown, never a
/// guessed default.
///
/// `telemetry_observed_at_host_monotonic_ns` and `revision` cover the
/// navigation/actuation telemetry projection only. Arbitration ownership and
/// software-safety fields are live overlays and deliberately make no claim to
/// have been observed at that telemetry timestamp.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct OperatorConsoleSnapshot {
    pub schema_version: u32,
    pub authority_kind: ConsoleRuntimeAuthorityKind,
    pub revision: ConsoleSnapshotRevision,
    pub telemetry_observed_at_host_monotonic_ns: Option<ConsoleHostTimestampNs>,
    pub runtime: Option<AgentRuntimeStateV1>,
    pub terminal: Option<ConsoleTerminalState>,
    /// Requested/pending authority from console arbitration. This is not a
    /// claim that the live runtime has admitted or currently holds authority.
    pub requested_owner: Option<ConsoleRequestedOwner>,
    /// Actual supervisor-backed motion authority retained by the sole owner.
    pub actual_authority: Option<ConsoleActualAuthority>,
    pub manual_command_envelope: Option<ConsoleManualCommandEnvelope>,
    pub map: Option<ConsoleMapSnapshot>,
    pub navigation: Option<ConsoleNavigationSnapshot>,
    pub slam: Option<ConsoleSlamSnapshot>,
    pub last_requested: Option<ConsoleRequestedCommand>,
    pub last_requested_actuation: Option<ConsoleRequestedActuation>,
    pub last_applied: Option<ConsoleAppliedReceipt>,
    pub stop_certainty: Option<ConsoleStopCertainty>,
    pub health: ConsoleSubsystemHealth,
    pub software_safety_stop_latched: bool,
    pub software_safety_signal_state: ConsoleSafetySignalState,
    pub physical_emergency_stop_state: ConsolePhysicalEmergencyStopState,
    /// Configured operator-side Rerun proxy URI. Presence proves only that a
    /// loopback serve target was admitted, not that the diagnostic worker is
    /// healthy, and never conveys motion or safety authority.
    pub rerun_diagnostics_url: Option<ConsoleRerunDiagnosticsUrl>,
}

impl OperatorConsoleSnapshot {
    pub fn unknown(
        revision: ConsoleSnapshotRevision,
        authority_kind: ConsoleRuntimeAuthorityKind,
    ) -> Self {
        Self {
            schema_version: OPERATOR_CONSOLE_SNAPSHOT_SCHEMA_V5,
            authority_kind,
            revision,
            telemetry_observed_at_host_monotonic_ns: None,
            runtime: None,
            terminal: None,
            requested_owner: None,
            actual_authority: None,
            manual_command_envelope: None,
            map: None,
            navigation: None,
            slam: None,
            last_requested: None,
            last_requested_actuation: None,
            last_applied: None,
            stop_certainty: None,
            health: ConsoleSubsystemHealth {
                stm32: None,
                head: None,
                eyes: None,
                oak: None,
                slam: None,
            },
            software_safety_stop_latched: false,
            software_safety_signal_state: ConsoleSafetySignalState::NotLatched,
            physical_emergency_stop_state: ConsolePhysicalEmergencyStopState::Unavailable,
            rerun_diagnostics_url: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct OperatorConsoleSnapshotEvent {
    snapshot: Arc<OperatorConsoleSnapshot>,
}

impl OperatorConsoleSnapshotEvent {
    pub fn snapshot(&self) -> Arc<OperatorConsoleSnapshot> {
        Arc::clone(&self.snapshot)
    }
}

#[derive(Debug)]
struct SnapshotSubscriber {
    sender: Sender<OperatorConsoleSnapshotEvent>,
    eviction_receiver: Receiver<OperatorConsoleSnapshotEvent>,
    liveness: Weak<()>,
}

#[derive(Debug)]
struct SnapshotState {
    latest: Arc<OperatorConsoleSnapshot>,
    latest_grid: Option<Arc<ConsoleOccupancyGrid>>,
    maximum_subscribers: usize,
    subscribers: Vec<SnapshotSubscriber>,
}

#[derive(Clone, Debug)]
pub struct OperatorConsoleSnapshotSubscriber {
    receiver: Receiver<OperatorConsoleSnapshotEvent>,
    _liveness: Arc<()>,
}

impl OperatorConsoleSnapshotSubscriber {
    pub fn try_latest(&self) -> Result<OperatorConsoleSnapshotEvent, TryRecvError> {
        let first = self.receiver.try_recv()?;
        let mut latest = first;
        while let Ok(next) = self.receiver.try_recv() {
            latest = next;
        }
        Ok(latest)
    }
}

#[derive(Debug)]
pub enum OperatorConsoleSnapshotError {
    SubscriberCapacityReached {
        maximum: usize,
    },
    RevisionNotIncreasing {
        previous: ConsoleSnapshotRevision,
        current: ConsoleSnapshotRevision,
    },
}

impl fmt::Display for OperatorConsoleSnapshotError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "operator-console snapshot error: {self:?}")
    }
}

impl std::error::Error for OperatorConsoleSnapshotError {}

#[derive(Clone, Debug)]
struct CachedSubmission {
    key: ConsoleIdempotencyKey,
    intent: OperatorConsoleIntent,
    outcome: OperatorConsoleSubmitOutcome,
}

#[derive(Debug)]
struct SessionState {
    source: ConsoleSourceKind,
    capability: ConsoleSessionCapability,
    last_source_sequence: Option<ConsoleSourceSequence>,
    next_manual_sequence: u64,
    idempotency: VecDeque<CachedSubmission>,
    last_activity: HostMonotonicTimestamp,
}

#[derive(Debug)]
struct ConsoleState {
    limits: OperatorConsoleLimits,
    next_session_id: Option<NonZeroU64>,
    next_downstream_id: Option<NonZeroU64>,
    sessions: HashMap<ConsoleSessionId, SessionState>,
    requested_owner: Option<ConsoleRequestedOwner>,
    latched_safety_stop_id: Option<ConsoleDownstreamRequestId>,
    stop_pending: Option<ConsoleDownstreamRequestId>,
}

#[derive(Clone, Copy, Debug)]
struct ConsoleResponseTokenSpec {
    intent: OperatorConsoleIntentKind,
    owner_acquisition: Option<ConsoleSessionId>,
    idempotency_pinned: bool,
    critical: bool,
    stop_barrier: bool,
    source_session_id: Option<ConsoleSessionId>,
}

impl ConsoleState {
    fn forget_idempotency_for(&mut self, downstream_request_id: ConsoleDownstreamRequestId) {
        for session in self.sessions.values_mut() {
            session
                .idempotency
                .retain(|cached| cached.outcome.downstream_request_id() != downstream_request_id);
        }
    }
}

#[derive(Debug)]
struct ConsoleShared {
    state: Mutex<ConsoleState>,
    safety_latched: Arc<AtomicBool>,
    safety_delivery_failed: AtomicBool,
    safety_drained: Arc<AtomicBool>,
    safety_completed: AtomicBool,
    safety_tx: Sender<OperatorConsoleSoftwareSafetyStop>,
    deadman_tx: Sender<OperatorConsoleDispatch>,
    urgent_tx: Sender<OperatorConsoleDispatch>,
    begin_manual_tx: Sender<OperatorConsoleDispatch>,
    normal_tx: Sender<OperatorConsoleDispatch>,
    manual_latest_tx: Sender<OperatorConsoleDispatch>,
    manual_eviction_rx: Receiver<OperatorConsoleDispatch>,
    responses: Arc<Mutex<ResponseLedger>>,
    snapshots: Mutex<SnapshotState>,
}

fn latch_shared_internal_fail_closed(shared: &ConsoleShared) {
    let was_latched = shared.safety_latched.swap(true, Ordering::AcqRel);
    if was_latched {
        return;
    }
    let signal = OperatorConsoleSoftwareSafetyStop {
        downstream_request_id: None,
        source: ConsoleDispatchSource::InternalFailClosed,
        received_at: None,
        response: None,
    };
    if shared.safety_tx.try_send(signal).is_err() {
        shared.safety_delivery_failed.store(true, Ordering::Release);
    }
}

/// Cloneable frontend/backend handle. It can submit typed intentions and
/// publish immutable observations, but cannot receive or execute commands.
#[derive(Clone, Debug)]
pub struct OperatorConsoleHandle {
    shared: Arc<ConsoleShared>,
}

pub fn operator_console(
    limits: OperatorConsoleLimits,
    initial_snapshot: OperatorConsoleSnapshot,
) -> (OperatorConsoleHandle, OperatorConsoleIngressReceiver) {
    let (safety_tx, safety_rx) = crossbeam_channel::bounded(1);
    let (deadman_tx, deadman_rx) = crossbeam_channel::bounded(1);
    let (urgent_tx, urgent_rx) = crossbeam_channel::bounded(8);
    let (begin_manual_tx, begin_manual_rx) = crossbeam_channel::bounded(1);
    let (normal_tx, normal_rx) = crossbeam_channel::bounded(limits.normal_queue_capacity.get());
    let (manual_latest_tx, manual_latest_rx) = crossbeam_channel::bounded(1);
    let safety_latched = Arc::new(AtomicBool::new(false));
    let safety_drained = Arc::new(AtomicBool::new(false));
    let responses = Arc::new(Mutex::new(ResponseLedger {
        maximum: limits.response_records.get(),
        records: VecDeque::new(),
    }));
    let handle = OperatorConsoleHandle {
        shared: Arc::new(ConsoleShared {
            state: Mutex::new(ConsoleState {
                limits,
                next_session_id: NonZeroU64::new(1),
                next_downstream_id: NonZeroU64::new(1),
                sessions: HashMap::new(),
                requested_owner: None,
                latched_safety_stop_id: None,
                stop_pending: None,
            }),
            safety_latched: Arc::clone(&safety_latched),
            safety_delivery_failed: AtomicBool::new(false),
            safety_drained: Arc::clone(&safety_drained),
            safety_completed: AtomicBool::new(false),
            safety_tx,
            deadman_tx,
            urgent_tx,
            begin_manual_tx,
            normal_tx,
            manual_latest_tx,
            manual_eviction_rx: manual_latest_rx.clone(),
            responses,
            snapshots: Mutex::new(SnapshotState {
                latest: Arc::new(initial_snapshot),
                latest_grid: None,
                maximum_subscribers: limits.snapshot_subscribers.get(),
                subscribers: Vec::new(),
            }),
        }),
    };
    let receiver = OperatorConsoleIngressReceiver {
        safety_rx,
        deadman_rx,
        urgent_rx,
        begin_manual_rx,
        normal_rx,
        manual_latest_rx,
        nonpriority_pending: Mutex::new(ConsoleNonpriorityPending::default()),
        safety_latched,
        safety_drained,
    };
    (handle, receiver)
}

impl OperatorConsoleHandle {
    fn lock_state(&self) -> MutexGuard<'_, ConsoleState> {
        match self.shared.state.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                // A panic while mutating arbitration state makes continued
                // motion authority unknowable. Preserve the data only for
                // diagnosis and fail closed for the process lifetime.
                self.latch_internal_fail_closed();
                poisoned.into_inner()
            }
        }
    }

    fn latch_internal_fail_closed(&self) {
        latch_shared_internal_fail_closed(&self.shared);
    }

    /// Fail-closed hook for the HTTP clock/task boundary.
    pub fn signal_internal_fail_closed(&self) {
        self.latch_internal_fail_closed();
    }

    pub fn open_session(
        &self,
        source: ConsoleSourceKind,
        capability: ConsoleSessionCapability,
        opened_at: HostMonotonicTimestamp,
    ) -> Result<ConsoleSessionId, OperatorConsoleSubmitError> {
        let mut state = self.lock_state();
        if state.sessions.len() == state.limits.maximum_sessions.get() {
            let owner = state.requested_owner.map(|owner| match owner {
                ConsoleRequestedOwner::Manual { session_id, .. }
                | ConsoleRequestedOwner::Autonomous { session_id, .. } => session_id,
            });
            let reclaim = state
                .sessions
                .iter()
                .filter(|(session_id, _)| Some(**session_id) != owner)
                .min_by_key(|(_, session)| session.last_activity)
                .map(|(session_id, _)| *session_id);
            let Some(reclaim) = reclaim else {
                return Err(OperatorConsoleSubmitError::SessionCapacityReached {
                    maximum: state.limits.maximum_sessions.get(),
                });
            };
            if let Some(reclaimed) = state.sessions.remove(&reclaim) {
                let mut ledger = recover_lock(&self.shared.responses);
                for cached in reclaimed.idempotency {
                    ledger.unpin(cached.outcome.downstream_request_id());
                }
            }
        }
        let raw = state
            .next_session_id
            .take()
            .ok_or(OperatorConsoleSubmitError::DownstreamSequenceExhausted)?;
        state.next_session_id = raw.get().checked_add(1).and_then(NonZeroU64::new);
        let id = ConsoleSessionId(raw);
        state.sessions.insert(
            id,
            SessionState {
                source,
                capability,
                last_source_sequence: None,
                next_manual_sequence: 0,
                idempotency: VecDeque::new(),
                last_activity: opened_at,
            },
        );
        Ok(id)
    }

    pub fn source_kind(
        &self,
        session_id: ConsoleSessionId,
    ) -> Result<ConsoleSourceKind, OperatorConsoleSubmitError> {
        self.lock_state()
            .sessions
            .get(&session_id)
            .map(|session| session.source)
            .ok_or(OperatorConsoleSubmitError::UnknownSession(session_id))
    }

    pub fn session_capability_matches(
        &self,
        session_id: ConsoleSessionId,
        capability: ConsoleSessionCapability,
    ) -> bool {
        self.lock_state()
            .sessions
            .get(&session_id)
            .is_some_and(|session| session.capability.constant_time_matches(capability))
    }

    pub fn requested_owner(&self) -> Option<ConsoleRequestedOwner> {
        self.lock_state().requested_owner
    }

    pub fn software_safety_stop_latched(&self) -> bool {
        self.shared.safety_latched.load(Ordering::Acquire)
    }

    pub fn submit(
        &self,
        session_id: ConsoleSessionId,
        capability: ConsoleSessionCapability,
        source_sequence: ConsoleSourceSequence,
        idempotency_key: ConsoleIdempotencyKey,
        intent: OperatorConsoleIntent,
        received_at: HostMonotonicTimestamp,
    ) -> Result<OperatorConsoleSubmitOutcome, OperatorConsoleSubmitError> {
        let mut state = self.lock_state();
        let session = state
            .sessions
            .get(&session_id)
            .ok_or(OperatorConsoleSubmitError::UnknownSession(session_id))?;
        if !session.capability.constant_time_matches(capability) {
            return Err(OperatorConsoleSubmitError::SessionCapabilityMismatch);
        }
        if let Some(cached) = session
            .idempotency
            .iter()
            .find(|cached| cached.key == idempotency_key)
        {
            if cached.intent == intent {
                return Ok(OperatorConsoleSubmitOutcome::IdempotentReplay {
                    downstream_request_id: cached.outcome.downstream_request_id(),
                });
            }
            return Err(OperatorConsoleSubmitError::IdempotencyConflict(
                idempotency_key,
            ));
        }
        if let Some(previous) = session.last_source_sequence
            && source_sequence <= previous
        {
            return Err(OperatorConsoleSubmitError::SourceSequenceNotIncreasing {
                previous,
                current: source_sequence,
            });
        }
        if self.shared.safety_latched.load(Ordering::Acquire) {
            // A response identity is a session-scoped observation capability.
            // The process-wide latch may be observed by every session, but its
            // originating response ID must never cross that boundary.
            if intent == OperatorConsoleIntent::SoftwareSafetyStop
                && let Some(downstream_request_id) = state.latched_safety_stop_id
                && recover_lock(&self.shared.responses)
                    .records
                    .iter()
                    .any(|record| {
                        record.downstream_request_id == downstream_request_id
                            && record.source_session_id == Some(session_id)
                    })
            {
                return Ok(OperatorConsoleSubmitOutcome::SoftwareSafetyStopLatched {
                    downstream_request_id,
                });
            }
            return Err(OperatorConsoleSubmitError::SoftwareSafetyStopLatched);
        }
        if state.stop_pending.is_some()
            && !matches!(intent, OperatorConsoleIntent::SoftwareSafetyStop)
        {
            return Err(OperatorConsoleSubmitError::StopPending);
        }

        let held_by = state.requested_owner.map(|owner| match owner {
            ConsoleRequestedOwner::Manual { session_id, .. }
            | ConsoleRequestedOwner::Autonomous { session_id, .. } => session_id,
        });
        match intent {
            OperatorConsoleIntent::BeginManual
            | OperatorConsoleIntent::AutonomousMapOnly
            | OperatorConsoleIntent::AutonomousFrontierExplore
            | OperatorConsoleIntent::AutonomousPointGoal(_) => {
                if let Some(held_by) = held_by {
                    return Err(OperatorConsoleSubmitError::AuthorityConflict {
                        requested_by: session_id,
                        held_by,
                    });
                }
            }
            OperatorConsoleIntent::ManualVelocity(_) | OperatorConsoleIntent::ReleaseManual => {
                if !matches!(
                    state.requested_owner,
                    Some(ConsoleRequestedOwner::Manual {
                        session_id: owner,
                        ..
                    }) if owner == session_id
                ) {
                    return Err(OperatorConsoleSubmitError::ManualAuthorityRequired(
                        session_id,
                    ));
                }
            }
            OperatorConsoleIntent::Arm
            | OperatorConsoleIntent::Disarm
            | OperatorConsoleIntent::Stop
            | OperatorConsoleIntent::SaveMap
            | OperatorConsoleIntent::SoftwareSafetyStop => {}
        }
        let manual_deadline = if matches!(
            intent,
            OperatorConsoleIntent::BeginManual | OperatorConsoleIntent::ManualVelocity(_)
        ) {
            Some(
                ConsoleHostTimestampNs::from_host(received_at)
                    .checked_add_millis(state.limits.manual_deadman_ms.get())
                    .ok_or(OperatorConsoleSubmitError::DeadmanDeadlineOverflow)?,
            )
        } else {
            None
        };

        let downstream_raw = state
            .next_downstream_id
            .ok_or(OperatorConsoleSubmitError::DownstreamSequenceExhausted)?;
        let downstream_request_id = ConsoleDownstreamRequestId(downstream_raw);
        let manual_sequence = state
            .sessions
            .get(&session_id)
            .ok_or(OperatorConsoleSubmitError::UnknownSession(session_id))?
            .next_manual_sequence;
        let next_manual_sequence = if matches!(intent, OperatorConsoleIntent::ManualVelocity(_)) {
            Some(
                manual_sequence
                    .checked_add(1)
                    .ok_or(OperatorConsoleSubmitError::ManualSequenceExhausted)?,
            )
        } else {
            None
        };
        let command = match intent {
            OperatorConsoleIntent::Arm => OperatorConsoleCommand::Arm,
            OperatorConsoleIntent::Disarm => OperatorConsoleCommand::Disarm,
            OperatorConsoleIntent::BeginManual => OperatorConsoleCommand::BeginManual,
            OperatorConsoleIntent::ManualVelocity(velocity) => {
                OperatorConsoleCommand::ManualVelocity {
                    sequence: ManualDriveSequence::from_raw(manual_sequence),
                    velocity,
                }
            }
            OperatorConsoleIntent::ReleaseManual => OperatorConsoleCommand::Stop {
                cause: ConsoleStopCause::ManualRelease,
            },
            OperatorConsoleIntent::AutonomousMapOnly => OperatorConsoleCommand::AutonomousMapOnly,
            OperatorConsoleIntent::AutonomousFrontierExplore => {
                OperatorConsoleCommand::AutonomousFrontierExplore
            }
            OperatorConsoleIntent::AutonomousPointGoal(selection) => {
                OperatorConsoleCommand::AutonomousPointGoal(selection)
            }
            OperatorConsoleIntent::Stop | OperatorConsoleIntent::SoftwareSafetyStop => {
                OperatorConsoleCommand::Stop {
                    cause: ConsoleStopCause::ExplicitGlobalStop,
                }
            }
            OperatorConsoleIntent::SaveMap => OperatorConsoleCommand::SaveMap,
        };
        let source = ConsoleDispatchSource::Session {
            session_id,
            source_sequence,
        };
        let stop_barrier = matches!(
            intent,
            OperatorConsoleIntent::ReleaseManual
                | OperatorConsoleIntent::Stop
                | OperatorConsoleIntent::Disarm
                | OperatorConsoleIntent::AutonomousMapOnly
                | OperatorConsoleIntent::SoftwareSafetyStop
        );
        let response = self.response_token(
            &mut state,
            downstream_request_id,
            ConsoleResponseTokenSpec {
                intent: intent.kind(),
                owner_acquisition: if matches!(
                    intent,
                    OperatorConsoleIntent::BeginManual
                        | OperatorConsoleIntent::AutonomousFrontierExplore
                        | OperatorConsoleIntent::AutonomousPointGoal(_)
                ) {
                    Some(session_id)
                } else {
                    None
                },
                idempotency_pinned: true,
                critical: stop_barrier,
                stop_barrier,
                source_session_id: Some(session_id),
            },
        )?;
        let active_console_authority = state.requested_owner.is_some_and(|owner| {
            let generation = match owner {
                ConsoleRequestedOwner::Manual {
                    authority_generation,
                    ..
                }
                | ConsoleRequestedOwner::Autonomous {
                    authority_generation,
                    ..
                } => authority_generation,
            };
            recover_lock(&self.shared.responses)
                .records
                .iter()
                .any(|record| {
                    record.downstream_request_id == generation
                        && record.state == ConsoleRuntimeResponseState::ActiveAuthority
                })
        });
        let fail_closed_on_delivery_failure = active_console_authority
            || matches!(
                intent,
                OperatorConsoleIntent::Disarm
                    | OperatorConsoleIntent::ReleaseManual
                    | OperatorConsoleIntent::AutonomousMapOnly
                    | OperatorConsoleIntent::Stop
            );

        let outcome = if intent == OperatorConsoleIntent::SoftwareSafetyStop {
            self.shared.safety_latched.store(true, Ordering::Release);
            let signal = OperatorConsoleSoftwareSafetyStop {
                downstream_request_id: Some(downstream_request_id),
                source,
                received_at: Some(received_at),
                response: Some(response),
            };
            match self.shared.safety_tx.try_send(signal) {
                Ok(()) => {
                    state.latched_safety_stop_id = Some(downstream_request_id);
                    OperatorConsoleSubmitOutcome::SoftwareSafetyStopLatched {
                        downstream_request_id,
                    }
                }
                Err(TrySendError::Full(signal)) => {
                    // An independently latched fail-closed signal won the
                    // single safety lane after this request's initial atomic
                    // check. This request was never delivered and cannot
                    // truthfully adopt that anonymous stop's completion.
                    let (_, _, _, _, response) = signal.into_emergency_parts();
                    drop(state);
                    if let Some(response) = response {
                        response.abort_before_runtime_delivery();
                    }
                    self.remove_response(downstream_request_id);
                    return Err(OperatorConsoleSubmitError::SoftwareSafetyStopLatched);
                }
                Err(TrySendError::Disconnected(signal)) => {
                    self.shared
                        .safety_delivery_failed
                        .store(true, Ordering::Release);
                    drop(state);
                    let (_, _, _, _, response) = signal.into_emergency_parts();
                    if let Some(response) = response {
                        response.abort_before_runtime_delivery();
                    }
                    self.remove_response(downstream_request_id);
                    return Err(OperatorConsoleSubmitError::RuntimeAdapterDisconnected);
                }
            }
        } else {
            let dispatch = OperatorConsoleDispatch {
                downstream_request_id,
                source,
                received_at,
                command,
                response,
            };
            let send_result = if matches!(intent, OperatorConsoleIntent::ManualVelocity(_)) {
                match self.shared.manual_latest_tx.try_send(dispatch) {
                    Ok(()) => Ok(()),
                    Err(TrySendError::Full(dispatch)) => {
                        if let Ok(superseded) = self.shared.manual_eviction_rx.try_recv() {
                            superseded.response.reject(
                                ConsoleResponseRejectionCode::SupersededByNewerManualDesiredState,
                            );
                        }
                        self.shared.manual_latest_tx.try_send(dispatch)
                    }
                    Err(TrySendError::Disconnected(dispatch)) => {
                        Err(TrySendError::Disconnected(dispatch))
                    }
                }
            } else if intent == OperatorConsoleIntent::BeginManual {
                self.shared.begin_manual_tx.try_send(dispatch)
            } else if matches!(
                intent,
                OperatorConsoleIntent::ReleaseManual
                    | OperatorConsoleIntent::Stop
                    | OperatorConsoleIntent::Disarm
                    | OperatorConsoleIntent::AutonomousMapOnly
            ) {
                self.shared.urgent_tx.try_send(dispatch)
            } else {
                self.shared.normal_tx.try_send(dispatch)
            };
            match send_result {
                Ok(()) => OperatorConsoleSubmitOutcome::AcceptedForProcessing {
                    downstream_request_id,
                },
                Err(TrySendError::Full(dispatch)) => {
                    let (_, _, _, response) = dispatch.into_parts();
                    drop(state);
                    response.abort_before_runtime_delivery();
                    self.remove_response(downstream_request_id);
                    if fail_closed_on_delivery_failure {
                        latch_shared_internal_fail_closed(&self.shared);
                    }
                    return Err(OperatorConsoleSubmitError::NormalQueueFull);
                }
                Err(TrySendError::Disconnected(dispatch)) => {
                    let (_, _, _, response) = dispatch.into_parts();
                    drop(state);
                    response.abort_before_runtime_delivery();
                    self.remove_response(downstream_request_id);
                    if fail_closed_on_delivery_failure {
                        latch_shared_internal_fail_closed(&self.shared);
                    }
                    return Err(OperatorConsoleSubmitError::RuntimeAdapterDisconnected);
                }
            }
        };

        state.next_downstream_id = downstream_raw
            .get()
            .checked_add(1)
            .and_then(NonZeroU64::new);
        let idempotency_limit = state.limits.idempotency_records_per_session.get();
        let session = state
            .sessions
            .get_mut(&session_id)
            .ok_or(OperatorConsoleSubmitError::UnknownSession(session_id))?;
        session.last_source_sequence = Some(source_sequence);
        if matches!(intent, OperatorConsoleIntent::ManualVelocity(_)) {
            session.next_manual_sequence =
                next_manual_sequence.ok_or(OperatorConsoleSubmitError::ManualSequenceExhausted)?;
        }
        if session.idempotency.len() == idempotency_limit
            && let Some(evicted) = session.idempotency.pop_front()
        {
            recover_lock(&self.shared.responses).unpin(evicted.outcome.downstream_request_id());
        }
        session.idempotency.push_back(CachedSubmission {
            key: idempotency_key,
            intent,
            outcome,
        });
        session.last_activity = received_at;
        state.requested_owner = match intent {
            OperatorConsoleIntent::BeginManual => Some(ConsoleRequestedOwner::Manual {
                session_id,
                authority_generation: downstream_request_id,
                deadman_deadline_host_monotonic_ns: manual_deadline
                    .ok_or(OperatorConsoleSubmitError::DeadmanDeadlineOverflow)?,
            }),
            OperatorConsoleIntent::ManualVelocity(_) => {
                let authority_generation = match state.requested_owner {
                    Some(ConsoleRequestedOwner::Manual {
                        session_id: owner,
                        authority_generation,
                        ..
                    }) if owner == session_id => authority_generation,
                    _ => {
                        return Err(OperatorConsoleSubmitError::ManualAuthorityRequired(
                            session_id,
                        ));
                    }
                };
                Some(ConsoleRequestedOwner::Manual {
                    session_id,
                    authority_generation,
                    deadman_deadline_host_monotonic_ns: manual_deadline
                        .ok_or(OperatorConsoleSubmitError::DeadmanDeadlineOverflow)?,
                })
            }
            // Map-only is an explicitly stopped observation mode, not an
            // acquired autonomous motion owner.
            OperatorConsoleIntent::AutonomousMapOnly => None,
            OperatorConsoleIntent::AutonomousFrontierExplore => {
                Some(ConsoleRequestedOwner::Autonomous {
                    session_id,
                    authority_generation: downstream_request_id,
                    mode: ConsoleAutonomousMode::FrontierExplore,
                })
            }
            OperatorConsoleIntent::AutonomousPointGoal(_) => {
                Some(ConsoleRequestedOwner::Autonomous {
                    session_id,
                    authority_generation: downstream_request_id,
                    mode: ConsoleAutonomousMode::PointGoal,
                })
            }
            OperatorConsoleIntent::ReleaseManual
            | OperatorConsoleIntent::Stop
            | OperatorConsoleIntent::Disarm
            | OperatorConsoleIntent::SoftwareSafetyStop => None,
            OperatorConsoleIntent::Arm | OperatorConsoleIntent::SaveMap => state.requested_owner,
        };
        if matches!(
            intent,
            OperatorConsoleIntent::ReleaseManual
                | OperatorConsoleIntent::Stop
                | OperatorConsoleIntent::Disarm
                | OperatorConsoleIntent::AutonomousMapOnly
                | OperatorConsoleIntent::SoftwareSafetyStop
        ) {
            state.stop_pending = Some(downstream_request_id);
        }
        Ok(outcome)
    }

    pub fn close_session(
        &self,
        session_id: ConsoleSessionId,
        capability: ConsoleSessionCapability,
    ) -> Result<bool, OperatorConsoleSubmitError> {
        let mut state = self.lock_state();
        let session = state
            .sessions
            .get(&session_id)
            .ok_or(OperatorConsoleSubmitError::UnknownSession(session_id))?;
        if !session.capability.constant_time_matches(capability) {
            return Err(OperatorConsoleSubmitError::SessionCapabilityMismatch);
        }
        let owns = state.requested_owner.is_some_and(|owner| match owner {
            ConsoleRequestedOwner::Manual {
                session_id: owner, ..
            }
            | ConsoleRequestedOwner::Autonomous {
                session_id: owner, ..
            } => owner == session_id,
        });
        if owns || state.stop_pending.is_some() {
            return Ok(false);
        }
        let removed = state.sessions.remove(&session_id);
        drop(state);
        if let Some(removed) = removed {
            let mut ledger = recover_lock(&self.shared.responses);
            for cached in removed.idempotency {
                ledger.unpin(cached.outcome.downstream_request_id());
            }
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Advance the server-owned manual deadman. The generated stop has a
    /// priority queue independent of ordinary request saturation.
    pub fn tick_deadman(
        &self,
        now: HostMonotonicTimestamp,
    ) -> Result<bool, OperatorConsoleSubmitError> {
        if self.shared.safety_latched.load(Ordering::Acquire) {
            return Ok(false);
        }
        let mut state = self.lock_state();
        let Some(ConsoleRequestedOwner::Manual {
            session_id,
            deadman_deadline_host_monotonic_ns,
            ..
        }) = state.requested_owner
        else {
            return Ok(false);
        };
        if ConsoleHostTimestampNs::from_host(now) < deadman_deadline_host_monotonic_ns {
            return Ok(false);
        }
        let downstream_raw = state
            .next_downstream_id
            .ok_or(OperatorConsoleSubmitError::DownstreamSequenceExhausted)?;
        let downstream_request_id = ConsoleDownstreamRequestId(downstream_raw);
        let response = self.response_token(
            &mut state,
            downstream_request_id,
            ConsoleResponseTokenSpec {
                intent: OperatorConsoleIntentKind::ManualDeadmanStop,
                owner_acquisition: None,
                idempotency_pinned: false,
                critical: true,
                stop_barrier: true,
                source_session_id: Some(session_id),
            },
        )?;
        let dispatch = OperatorConsoleDispatch {
            downstream_request_id,
            source: ConsoleDispatchSource::ManualDeadman { session_id },
            received_at: now,
            command: OperatorConsoleCommand::Stop {
                cause: ConsoleStopCause::ManualDeadman,
            },
            response,
        };
        match self.shared.deadman_tx.try_send(dispatch) {
            Ok(()) => {}
            Err(TrySendError::Full(dispatch)) => {
                let (_, _, _, response) = dispatch.into_parts();
                drop(state);
                response.abort_before_runtime_delivery();
                self.remove_response(downstream_request_id);
                latch_shared_internal_fail_closed(&self.shared);
                return Ok(false);
            }
            Err(TrySendError::Disconnected(dispatch)) => {
                let (_, _, _, response) = dispatch.into_parts();
                drop(state);
                response.abort_before_runtime_delivery();
                self.remove_response(downstream_request_id);
                latch_shared_internal_fail_closed(&self.shared);
                return Err(OperatorConsoleSubmitError::RuntimeAdapterDisconnected);
            }
        }
        state.next_downstream_id = downstream_raw
            .get()
            .checked_add(1)
            .and_then(NonZeroU64::new);
        state.requested_owner = None;
        state.stop_pending = Some(downstream_request_id);
        Ok(true)
    }

    pub fn response_record(
        &self,
        id: ConsoleDownstreamRequestId,
    ) -> Option<OperatorConsoleResponseRecord> {
        recover_lock(&self.shared.responses)
            .records
            .iter()
            .find(|record| record.downstream_request_id == id)
            .cloned()
    }

    /// Whether an authorized HTTP request from the owning session has read the
    /// currently retained version of this exact response record.
    ///
    /// Runtime state transitions clear the evidence, so observing an earlier
    /// pending version cannot stand in for observing its terminal completion.
    /// Eviction removes the evidence with the record.
    pub fn response_record_was_http_observed(&self, id: ConsoleDownstreamRequestId) -> bool {
        recover_lock(&self.shared.responses).current_record_was_http_observed(id)
    }

    pub(super) fn observe_response_record_for_http(
        &self,
        id: ConsoleDownstreamRequestId,
        source_session_id: ConsoleSessionId,
    ) -> Option<OperatorConsoleResponseRecord> {
        recover_lock(&self.shared.responses).observe_for_http_session(id, source_session_id)
    }

    pub fn latest_requested_command(&self) -> Option<ConsoleRequestedCommand> {
        recover_lock(&self.shared.responses)
            .records
            .back()
            .map(|record| ConsoleRequestedCommand {
                downstream_request_id: record.downstream_request_id,
                kind: record.intent,
            })
    }

    pub fn latest_snapshot(&self) -> Arc<OperatorConsoleSnapshot> {
        let (latched, signal_state) = self.current_safety_observation();
        let requested_owner = self.requested_owner();
        let mut state = recover_lock(&self.shared.snapshots);
        if state.latest.software_safety_stop_latched != latched
            || state.latest.software_safety_signal_state != signal_state
            || state.latest.requested_owner != requested_owner
        {
            let mut overlaid = (*state.latest).clone();
            overlaid.software_safety_stop_latched = latched;
            overlaid.software_safety_signal_state = signal_state;
            overlaid.requested_owner = requested_owner;
            state.latest = Arc::new(overlaid);
        }
        Arc::clone(&state.latest)
    }

    /// Publish one immutable grid. Only the newest epoch/revision is retained;
    /// status polling therefore never clones or serializes cell storage.
    pub fn publish_grid(&self, grid: ConsoleOccupancyGrid) -> bool {
        let mut state = recover_lock(&self.shared.snapshots);
        if state.latest_grid.as_ref().is_some_and(|latest| {
            (latest.map_epoch_id.get(), latest.revision) >= (grid.map_epoch_id.get(), grid.revision)
        }) {
            return false;
        }
        state.latest_grid = Some(Arc::new(grid));
        true
    }

    /// Fetch only an exact current grid binding. A stale click/view can never
    /// silently receive cells from a different epoch or revision.
    pub fn exact_grid(
        &self,
        map_epoch_id: NonZeroU64,
        revision: u64,
    ) -> Option<Arc<ConsoleOccupancyGrid>> {
        recover_lock(&self.shared.snapshots)
            .latest_grid
            .as_ref()
            .filter(|grid| grid.map_epoch_id == map_epoch_id && grid.revision == revision)
            .map(Arc::clone)
    }

    pub fn publish_snapshot(
        &self,
        mut snapshot: OperatorConsoleSnapshot,
    ) -> Result<(), OperatorConsoleSnapshotError> {
        let (latched, signal_state) = self.current_safety_observation();
        snapshot.software_safety_stop_latched = latched;
        snapshot.software_safety_signal_state = signal_state;
        snapshot.requested_owner = self.requested_owner();
        let snapshot = Arc::new(snapshot);
        let mut state = recover_lock(&self.shared.snapshots);
        if snapshot.revision <= state.latest.revision {
            return Err(OperatorConsoleSnapshotError::RevisionNotIncreasing {
                previous: state.latest.revision,
                current: snapshot.revision,
            });
        }
        state.latest = Arc::clone(&snapshot);
        state.subscribers.retain(|subscriber| {
            if subscriber.liveness.upgrade().is_none() {
                return false;
            }
            let event = OperatorConsoleSnapshotEvent {
                snapshot: Arc::clone(&snapshot),
            };
            match subscriber.sender.try_send(event) {
                Ok(()) => true,
                Err(TrySendError::Full(event)) => {
                    let _ = subscriber.eviction_receiver.try_recv();
                    subscriber.sender.try_send(event).is_ok()
                }
                Err(TrySendError::Disconnected(_)) => false,
            }
        });
        Ok(())
    }

    fn current_safety_observation(&self) -> (bool, ConsoleSafetySignalState) {
        let latched = self.shared.safety_latched.load(Ordering::Acquire);
        let signal_state = if !latched {
            ConsoleSafetySignalState::NotLatched
        } else if self.shared.safety_delivery_failed.load(Ordering::Acquire) {
            ConsoleSafetySignalState::RuntimeAdapterDisconnected
        } else if self.shared.safety_completed.load(Ordering::Acquire) {
            ConsoleSafetySignalState::CompletedFaultLatched
        } else if self.shared.safety_drained.load(Ordering::Acquire) {
            ConsoleSafetySignalState::RuntimeDrainedAwaitingCompletion
        } else {
            ConsoleSafetySignalState::PendingRuntimeDrain
        };
        (latched, signal_state)
    }

    pub fn subscribe_snapshots(
        &self,
    ) -> Result<OperatorConsoleSnapshotSubscriber, OperatorConsoleSnapshotError> {
        let mut state = recover_lock(&self.shared.snapshots);
        if state.subscribers.len() == state.maximum_subscribers {
            return Err(OperatorConsoleSnapshotError::SubscriberCapacityReached {
                maximum: state.maximum_subscribers,
            });
        }
        let (sender, receiver) = crossbeam_channel::bounded(1);
        let liveness = Arc::new(());
        let initial = OperatorConsoleSnapshotEvent {
            snapshot: Arc::clone(&state.latest),
        };
        if sender.try_send(initial).is_err() {
            return Err(OperatorConsoleSnapshotError::SubscriberCapacityReached {
                maximum: state.maximum_subscribers,
            });
        }
        state.subscribers.push(SnapshotSubscriber {
            sender,
            eviction_receiver: receiver.clone(),
            liveness: Arc::downgrade(&liveness),
        });
        Ok(OperatorConsoleSnapshotSubscriber {
            receiver,
            _liveness: liveness,
        })
    }

    fn response_token(
        &self,
        state: &mut ConsoleState,
        downstream_request_id: ConsoleDownstreamRequestId,
        spec: ConsoleResponseTokenSpec,
    ) -> Result<OperatorConsoleResponseToken, OperatorConsoleSubmitError> {
        let evicted = recover_lock(&self.shared.responses).insert_pending(
            downstream_request_id,
            spec.intent,
            spec.idempotency_pinned,
            spec.critical,
            spec.source_session_id,
        )?;
        if let Some(evicted) = evicted {
            state.forget_idempotency_for(evicted);
        }
        let expected_lifecycle_zero = ConsoleExpectedLifecycleZero::for_intent(spec.intent);
        Ok(OperatorConsoleResponseToken {
            downstream_request_id,
            ledger: Arc::clone(&self.shared.responses),
            shared: Arc::downgrade(&self.shared),
            owner_acquisition: spec.owner_acquisition,
            requires_applied_zero: expected_lifecycle_zero.is_some()
                || spec.intent == OperatorConsoleIntentKind::SoftwareSafetyStop,
            expected_lifecycle_zero,
            typed_request_key: None,
            stop_generation: spec.stop_barrier.then_some(downstream_request_id),
            #[cfg(feature = "actuation")]
            software_safety_stop: spec.intent == OperatorConsoleIntentKind::SoftwareSafetyStop,
            terminal: false,
        })
    }

    fn remove_response(&self, id: ConsoleDownstreamRequestId) {
        recover_lock(&self.shared.responses)
            .records
            .retain(|record| record.downstream_request_id != id);
    }

    #[cfg(all(feature = "actuation", feature = "agent-runtime", unix))]
    pub(super) fn complete_internal_fail_closed_with_verified_emergency_stop(
        &self,
        expected_key: super::control_socket::AgentControlTypedRequestKey,
        evidence: super::live_motion_owner::LiveSoftwareEmergencyStopApplied,
    ) -> Result<bool, ConsoleVerifiedCompletionError> {
        if evidence.typed_request_key() != expected_key {
            return Err(ConsoleVerifiedCompletionError::TypedRequestKeyMismatch);
        }
        let _receipt = ConsoleSoftwareEmergencyStopReceipt::from_verified(&evidence)?;
        if !self.shared.safety_latched.load(Ordering::Acquire) {
            return Ok(false);
        }
        self.shared.safety_completed.store(true, Ordering::Release);
        Ok(true)
    }
}

/// Strict weak HTTP DTO. The HTTP adapter decodes it once and immediately
/// converts it to domain types.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ConsoleIntentRequestDto {
    pub schema_version: u32,
    pub session_id: String,
    pub source_sequence: String,
    pub idempotency_key: String,
    pub intent: ConsoleIntentDto,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum ConsoleIntentDto {
    Arm {},
    Disarm {},
    BeginManual {},
    ManualVelocity {
        forward_velocity_mps: f64,
        yaw_rate_rad_s: f64,
    },
    ReleaseManual {},
    AutonomousMapOnly {},
    AutonomousFrontierExplore {},
    AutonomousPointGoal {
        map_epoch_id: String,
        displayed_revision: String,
        x_m: f64,
        y_m: f64,
    },
    Stop {},
    SaveMap {},
    SoftwareSafetyStop {},
}

#[derive(Debug)]
pub(crate) enum ConsoleIntentRequestParseError {
    UnsupportedSchema(u32),
    Identity(ConsoleIdentityError),
    Velocity(FiniteManualVelocityParseError),
    MapPoint(MapPointGoalSelectionParseError),
    InvalidDecimalIdentity,
}

impl fmt::Display for ConsoleIntentRequestParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedSchema(version) => {
                write!(formatter, "unsupported console intent schema {version}")
            }
            Self::Identity(source) => write!(formatter, "invalid console identity: {source}"),
            Self::Velocity(source) => write!(formatter, "invalid body velocity: {source}"),
            Self::MapPoint(source) => write!(formatter, "invalid bound map point: {source}"),
            Self::InvalidDecimalIdentity => {
                formatter.write_str("console identity is not an exact decimal u64 string")
            }
        }
    }
}

impl std::error::Error for ConsoleIntentRequestParseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Identity(source) => Some(source),
            Self::Velocity(source) => Some(source),
            Self::MapPoint(source) => Some(source),
            Self::UnsupportedSchema(_) | Self::InvalidDecimalIdentity => None,
        }
    }
}

impl ConsoleIntentRequestDto {
    pub(crate) fn parse(
        self,
    ) -> Result<
        (
            ConsoleSessionId,
            ConsoleSourceSequence,
            ConsoleIdempotencyKey,
            OperatorConsoleIntent,
        ),
        ConsoleIntentRequestParseError,
    > {
        if self.schema_version != OPERATOR_CONSOLE_SCHEMA_V1 {
            return Err(ConsoleIntentRequestParseError::UnsupportedSchema(
                self.schema_version,
            ));
        }
        let session = ConsoleSessionId::parse(
            self.session_id
                .parse()
                .map_err(|_| ConsoleIntentRequestParseError::InvalidDecimalIdentity)?,
        )
        .map_err(ConsoleIntentRequestParseError::Identity)?;
        let sequence = ConsoleSourceSequence::parse(
            self.source_sequence
                .parse()
                .map_err(|_| ConsoleIntentRequestParseError::InvalidDecimalIdentity)?,
        )
        .map_err(ConsoleIntentRequestParseError::Identity)?;
        let idempotency = ConsoleIdempotencyKey::parse(
            self.idempotency_key
                .parse()
                .map_err(|_| ConsoleIntentRequestParseError::InvalidDecimalIdentity)?,
        )
        .map_err(ConsoleIntentRequestParseError::Identity)?;
        let intent = match self.intent {
            ConsoleIntentDto::Arm {} => OperatorConsoleIntent::Arm,
            ConsoleIntentDto::Disarm {} => OperatorConsoleIntent::Disarm,
            ConsoleIntentDto::BeginManual {} => OperatorConsoleIntent::BeginManual,
            ConsoleIntentDto::ManualVelocity {
                forward_velocity_mps,
                yaw_rate_rad_s,
            } => OperatorConsoleIntent::ManualVelocity(
                FiniteManualVelocityV1::parse(forward_velocity_mps, yaw_rate_rad_s)
                    .map_err(ConsoleIntentRequestParseError::Velocity)?,
            ),
            ConsoleIntentDto::ReleaseManual {} => OperatorConsoleIntent::ReleaseManual,
            ConsoleIntentDto::AutonomousMapOnly {} => OperatorConsoleIntent::AutonomousMapOnly,
            ConsoleIntentDto::AutonomousFrontierExplore {} => {
                OperatorConsoleIntent::AutonomousFrontierExplore
            }
            ConsoleIntentDto::AutonomousPointGoal {
                map_epoch_id,
                displayed_revision,
                x_m,
                y_m,
            } => OperatorConsoleIntent::AutonomousPointGoal(
                MapPointGoalSelection::parse(MapPointGoalSelectionDto {
                    map_epoch_id: map_epoch_id
                        .parse()
                        .map_err(|_| ConsoleIntentRequestParseError::InvalidDecimalIdentity)?,
                    displayed_revision: displayed_revision
                        .parse()
                        .map_err(|_| ConsoleIntentRequestParseError::InvalidDecimalIdentity)?,
                    x_m,
                    y_m,
                })
                .map_err(ConsoleIntentRequestParseError::MapPoint)?,
            ),
            ConsoleIntentDto::Stop {} => OperatorConsoleIntent::Stop,
            ConsoleIntentDto::SaveMap {} => OperatorConsoleIntent::SaveMap,
            ConsoleIntentDto::SoftwareSafetyStop {} => OperatorConsoleIntent::SoftwareSafetyStop,
        };
        Ok((session, sequence, idempotency, intent))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    use super::*;

    fn fixture_with_limits(
        limits: OperatorConsoleLimits,
    ) -> (OperatorConsoleHandle, OperatorConsoleIngressReceiver) {
        operator_console(
            limits,
            OperatorConsoleSnapshot::unknown(
                ConsoleSnapshotRevision::parse(1).unwrap(),
                ConsoleRuntimeAuthorityKind::ProductionExternalInterlocks,
            ),
        )
    }

    fn fixture() -> (OperatorConsoleHandle, OperatorConsoleIngressReceiver) {
        fixture_with_limits(OperatorConsoleLimits::default())
    }

    fn seq(value: u64) -> ConsoleSourceSequence {
        ConsoleSourceSequence::parse(value).unwrap()
    }

    fn key(value: u64) -> ConsoleIdempotencyKey {
        ConsoleIdempotencyKey::parse(value).unwrap()
    }

    fn at_ms(value: u64) -> HostMonotonicTimestamp {
        HostMonotonicTimestamp::from_nanos(value * 1_000_000)
    }

    fn cap() -> ConsoleSessionCapability {
        ConsoleSessionCapability::from_bytes([0x5a; 32])
    }

    #[test]
    fn bind_rejects_wildcard_and_non_loopback() {
        for address in ["0.0.0.0:0", "[::]:0", "192.168.50.2:8080"] {
            assert!(OperatorConsoleBind::parse(address.parse().unwrap()).is_err());
        }
        for address in ["127.0.0.1:0", "[::1]:8080"] {
            assert!(OperatorConsoleBind::parse(address.parse().unwrap()).is_ok());
        }
    }

    #[test]
    fn rerun_diagnostics_url_is_an_exact_operator_side_loopback_proxy() {
        for port in [1, 9_876, u16::MAX] {
            let url = ConsoleRerunDiagnosticsUrl::from_admitted_forwarded_port(
                NonZeroU16::new(port).unwrap(),
            );
            assert_eq!(url.forwarded_port().get(), port);
            assert_eq!(
                url.serve_loopback_bind(),
                format!("127.0.0.1:{port}").parse().unwrap()
            );
            assert_eq!(
                url.to_string(),
                format!("rerun+http://127.0.0.1:{port}/proxy"),
                "the displayed endpoint is the same-port operator-side SSH forward"
            );
            assert_eq!(
                serde_json::to_value(url).unwrap(),
                serde_json::json!(format!("rerun+http://127.0.0.1:{port}/proxy"))
            );
        }
    }

    #[test]
    fn snapshot_exposes_rerun_only_when_a_serve_target_is_supplied() {
        let mut snapshot = OperatorConsoleSnapshot::unknown(
            ConsoleSnapshotRevision::parse(1).unwrap(),
            ConsoleRuntimeAuthorityKind::ProductionExternalInterlocks,
        );
        assert_eq!(snapshot.schema_version, OPERATOR_CONSOLE_SNAPSHOT_SCHEMA_V5);
        assert_eq!(
            snapshot.authority_kind,
            ConsoleRuntimeAuthorityKind::ProductionExternalInterlocks
        );
        assert_ne!(
            snapshot.schema_version, OPERATOR_CONSOLE_SNAPSHOT_SCHEMA_V2,
            "a canonical Rerun proxy URI must not silently reuse V2 browser-URL semantics"
        );
        let absent = serde_json::to_value(&snapshot).unwrap();
        assert!(absent["rerun_diagnostics_url"].is_null());

        snapshot.rerun_diagnostics_url =
            Some(ConsoleRerunDiagnosticsUrl::from_admitted_forwarded_port(
                NonZeroU16::new(9_876).unwrap(),
            ));
        let configured = serde_json::to_value(snapshot).unwrap();
        assert_eq!(
            configured["rerun_diagnostics_url"],
            serde_json::json!("rerun+http://127.0.0.1:9876/proxy")
        );
    }

    #[test]
    fn slam_snapshot_serializes_exact_counters_clocks_and_runtime_providers() {
        let mut snapshot = OperatorConsoleSnapshot::unknown(
            ConsoleSnapshotRevision::parse(1).unwrap(),
            ConsoleRuntimeAuthorityKind::ProductionExternalInterlocks,
        );
        snapshot.health.slam = Some(ConsoleHealth::Ready);
        snapshot.slam = Some(ConsoleSlamSnapshot {
            inference: ConsoleInferenceRuntime {
                superpoint: ConsoleInferenceSelection {
                    requested: ConsoleRequestedInferenceBackend::Auto,
                    selected: ConsoleSelectedInferenceBackend::Cuda,
                },
                lightglue: ConsoleInferenceSelection {
                    requested: ConsoleRequestedInferenceBackend::TensorRt,
                    selected: ConsoleSelectedInferenceBackend::TensorRt,
                },
            },
            started_pairs: u64::MAX,
            successful_pairs: u64::MAX - 1,
            recoverable_failures: 1,
            fatal_failures: 0,
            last_successful_source_arrival_host_monotonic_ns: Some(u64::MAX - 1),
            last_successful_completion_host_monotonic_ns: Some(u64::MAX),
            rate_window: Some(ConsoleSlamRateWindow {
                successful_completions: 64,
                span_ns: u64::MAX,
            }),
        });

        let json = serde_json::to_value(snapshot).unwrap();
        assert_eq!(json["schema_version"], serde_json::json!(5));
        assert_eq!(json["health"]["slam"], serde_json::json!("ready"));
        assert_eq!(json["slam"]["started_pairs"], u64::MAX.to_string());
        assert_eq!(
            json["slam"]["last_successful_completion_host_monotonic_ns"],
            u64::MAX.to_string()
        );
        assert_eq!(json["slam"]["rate_window"]["span_ns"], u64::MAX.to_string());
        assert_eq!(
            json["slam"]["inference"]["superpoint"],
            serde_json::json!({"requested": "auto", "selected": "cuda"})
        );
        assert_eq!(
            json["slam"]["inference"]["lightglue"],
            serde_json::json!({"requested": "tensorrt", "selected": "tensorrt"})
        );
    }

    #[test]
    fn independent_source_sequences_share_one_private_downstream_sequence() {
        let (handle, receiver) = fixture();
        let user = handle
            .open_session(ConsoleSourceKind::Operator, cap(), at_ms(0))
            .unwrap();
        let agent = handle
            .open_session(ConsoleSourceKind::Agent, cap(), at_ms(0))
            .unwrap();
        let first = handle
            .submit(
                user,
                cap(),
                seq(1),
                key(1),
                OperatorConsoleIntent::Arm,
                at_ms(0),
            )
            .unwrap();
        let second = handle
            .submit(
                agent,
                cap(),
                seq(1),
                key(1),
                OperatorConsoleIntent::SaveMap,
                at_ms(0),
            )
            .unwrap();
        assert_eq!(first.downstream_request_id().get(), 1);
        assert_eq!(second.downstream_request_id().get(), 2);
        let one = match receiver.try_next().unwrap() {
            OperatorConsoleIngressItem::Dispatch(value) => value,
            OperatorConsoleIngressItem::SoftwareSafetyStop(_) => panic!("unexpected safety stop"),
        };
        let two = match receiver.try_next().unwrap() {
            OperatorConsoleIngressItem::Dispatch(value) => value,
            OperatorConsoleIngressItem::SoftwareSafetyStop(_) => panic!("unexpected safety stop"),
        };
        assert_eq!(one.agent_request().request_id().get(), 1);
        assert_eq!(two.agent_request().request_id().get(), 2);
    }

    #[test]
    fn idempotent_replay_does_not_requeue_or_advance_global_sequence() {
        let (handle, receiver) = fixture();
        let user = handle
            .open_session(ConsoleSourceKind::Operator, cap(), at_ms(0))
            .unwrap();
        let first = handle
            .submit(
                user,
                cap(),
                seq(1),
                key(9),
                OperatorConsoleIntent::Arm,
                at_ms(0),
            )
            .unwrap();
        let replay = handle
            .submit(
                user,
                cap(),
                seq(1),
                key(9),
                OperatorConsoleIntent::Arm,
                at_ms(0),
            )
            .unwrap();
        assert!(matches!(
            replay,
            OperatorConsoleSubmitOutcome::IdempotentReplay { .. }
        ));
        assert_eq!(
            first.downstream_request_id(),
            replay.downstream_request_id()
        );
        let dispatch = match receiver.try_next().unwrap() {
            OperatorConsoleIngressItem::Dispatch(dispatch) => dispatch,
            OperatorConsoleIngressItem::SoftwareSafetyStop(_) => panic!("unexpected safety stop"),
        };
        dispatch
            .response
            .reject(ConsoleResponseRejectionCode::RuntimeRejected);
        assert!(matches!(receiver.try_next(), Err(TryRecvError::Empty)));
        let next = handle
            .submit(
                user,
                cap(),
                seq(2),
                key(10),
                OperatorConsoleIntent::SaveMap,
                at_ms(0),
            )
            .unwrap();
        assert_eq!(next.downstream_request_id().get(), 2);
    }

    #[test]
    fn session_and_queue_bounds_fail_closed_without_consuming_sequence() {
        let limits = OperatorConsoleLimits::parse(1, 1, 1, 4, 1, 100).unwrap();
        let (handle, receiver) = fixture_with_limits(limits);
        let user = handle
            .open_session(ConsoleSourceKind::Operator, cap(), at_ms(0))
            .unwrap();
        handle
            .submit(
                user,
                cap(),
                seq(1),
                key(1),
                OperatorConsoleIntent::BeginManual,
                at_ms(0),
            )
            .unwrap();
        assert!(matches!(
            handle.open_session(ConsoleSourceKind::Agent, cap(), at_ms(0)),
            Err(OperatorConsoleSubmitError::SessionCapacityReached { .. })
        ));
        handle
            .submit(
                user,
                cap(),
                seq(2),
                key(2),
                OperatorConsoleIntent::SaveMap,
                at_ms(0),
            )
            .unwrap();
        assert!(matches!(
            handle.submit(
                user,
                cap(),
                seq(3),
                key(3),
                OperatorConsoleIntent::SaveMap,
                at_ms(0)
            ),
            Err(OperatorConsoleSubmitError::NormalQueueFull)
        ));
        for _ in 0..2 {
            let dispatch = match receiver.try_next().unwrap() {
                OperatorConsoleIngressItem::Dispatch(dispatch) => dispatch,
                OperatorConsoleIngressItem::SoftwareSafetyStop(_) => {
                    panic!("unexpected safety stop")
                }
            };
            dispatch
                .response
                .reject(ConsoleResponseRejectionCode::RuntimeRejected);
        }
        let retry = handle
            .submit(
                user,
                cap(),
                seq(3),
                key(3),
                OperatorConsoleIntent::SaveMap,
                at_ms(0),
            )
            .unwrap();
        assert_eq!(retry.downstream_request_id().get(), 3);
    }

    #[test]
    fn pre_admission_arm_backpressure_is_retryable_without_false_safety_latch() {
        let limits = OperatorConsoleLimits::parse(1, 1, 1, 4, 1, 100).unwrap();
        let (handle, receiver) = fixture_with_limits(limits);
        let user = handle
            .open_session(ConsoleSourceKind::Operator, cap(), at_ms(0))
            .unwrap();
        handle
            .submit(
                user,
                cap(),
                seq(1),
                key(1),
                OperatorConsoleIntent::SaveMap,
                at_ms(0),
            )
            .expect("fill normal queue");

        assert!(matches!(
            handle.submit(
                user,
                cap(),
                seq(2),
                key(2),
                OperatorConsoleIntent::Arm,
                at_ms(1),
            ),
            Err(OperatorConsoleSubmitError::NormalQueueFull)
        ));
        assert!(!handle.software_safety_stop_latched());
        assert!(handle.requested_owner().is_none());

        let queued = match receiver.try_next().expect("queued save-map") {
            OperatorConsoleIngressItem::Dispatch(dispatch) => dispatch,
            OperatorConsoleIngressItem::SoftwareSafetyStop(_) => {
                panic!("unexpected safety stop")
            }
        };
        queued
            .response
            .reject(ConsoleResponseRejectionCode::RuntimeRejected);
        let retry = handle
            .submit(
                user,
                cap(),
                seq(2),
                key(2),
                OperatorConsoleIntent::Arm,
                at_ms(1),
            )
            .expect("same source sequence remains retryable");
        assert_eq!(retry.downstream_request_id().get(), 2);
        assert!(!handle.software_safety_stop_latched());
    }

    #[test]
    fn disconnected_first_authority_submission_returns_without_deadlock_or_false_latch() {
        let (handle, receiver) = fixture();
        let user = handle
            .open_session(ConsoleSourceKind::Operator, cap(), at_ms(0))
            .unwrap();
        drop(receiver);

        let submitter = handle.clone();
        let (result_tx, result_rx) = mpsc::sync_channel(1);
        let worker = thread::spawn(move || {
            let result = submitter.submit(
                user,
                cap(),
                seq(1),
                key(1),
                OperatorConsoleIntent::BeginManual,
                at_ms(0),
            );
            result_tx.send(result).expect("test result receiver");
        });
        let result = result_rx
            .recv_timeout(Duration::from_millis(250))
            .expect("disconnected submit must not deadlock");
        assert!(matches!(
            result,
            Err(OperatorConsoleSubmitError::RuntimeAdapterDisconnected)
        ));
        worker.join().expect("bounded submit worker");
        assert!(!handle.software_safety_stop_latched());
        assert!(handle.requested_owner().is_none());
        assert!(handle.latest_requested_command().is_none());
    }

    #[test]
    fn software_stop_is_priority_latched_and_irreversible() {
        let (handle, receiver) = fixture();
        let user = handle
            .open_session(ConsoleSourceKind::Operator, cap(), at_ms(0))
            .unwrap();
        handle
            .submit(
                user,
                cap(),
                seq(1),
                key(1),
                OperatorConsoleIntent::Arm,
                at_ms(0),
            )
            .unwrap();
        let stop = handle
            .submit(
                user,
                cap(),
                seq(2),
                key(2),
                OperatorConsoleIntent::SoftwareSafetyStop,
                at_ms(1),
            )
            .unwrap();
        assert!(matches!(
            stop,
            OperatorConsoleSubmitOutcome::SoftwareSafetyStopLatched { .. }
        ));
        assert!(handle.software_safety_stop_latched());
        assert!(matches!(
            handle.submit(
                user,
                cap(),
                seq(3),
                key(3),
                OperatorConsoleIntent::Arm,
                at_ms(2)
            ),
            Err(OperatorConsoleSubmitError::SoftwareSafetyStopLatched)
        ));
        assert!(matches!(
            receiver.try_next().unwrap(),
            OperatorConsoleIngressItem::SoftwareSafetyStop(_)
        ));
        assert!(matches!(receiver.try_next(), Err(TryRecvError::Empty)));
    }

    #[test]
    fn latched_software_stop_response_identity_never_crosses_sessions() {
        let (handle, receiver) = fixture();
        let owner = handle
            .open_session(ConsoleSourceKind::Operator, cap(), at_ms(0))
            .unwrap();
        let observer = handle
            .open_session(ConsoleSourceKind::Agent, cap(), at_ms(0))
            .unwrap();
        let original = handle
            .submit(
                owner,
                cap(),
                seq(1),
                key(1),
                OperatorConsoleIntent::SoftwareSafetyStop,
                at_ms(1),
            )
            .expect("first stop owns one session-scoped response");
        let original_id = original.downstream_request_id();

        let same_session = handle
            .submit(
                owner,
                cap(),
                seq(2),
                key(2),
                OperatorConsoleIntent::SoftwareSafetyStop,
                at_ms(2),
            )
            .expect("the owning session may observe its existing stop response");
        assert_eq!(same_session.downstream_request_id(), original_id);
        assert!(matches!(
            handle.submit(
                observer,
                cap(),
                seq(1),
                key(1),
                OperatorConsoleIntent::SoftwareSafetyStop,
                at_ms(2),
            ),
            Err(OperatorConsoleSubmitError::SoftwareSafetyStopLatched)
        ));
        assert_eq!(
            handle
                .response_record(original_id)
                .expect("latched response remains retained")
                .source_session_id,
            Some(owner)
        );
        assert!(matches!(
            receiver.try_next().expect("one process-wide safety signal"),
            OperatorConsoleIngressItem::SoftwareSafetyStop(_)
        ));
        assert!(matches!(receiver.try_next(), Err(TryRecvError::Empty)));
    }

    #[test]
    fn software_stop_does_not_adopt_an_anonymous_fail_closed_signal_that_won_the_lane() {
        let (handle, receiver) = fixture();
        let user = handle
            .open_session(ConsoleSourceKind::Operator, cap(), at_ms(0))
            .unwrap();
        // Model the exact interleaving after submit's initial atomic read:
        // another thread latches and fills the one-slot lane before this
        // request enqueues its response-bearing signal.
        handle
            .shared
            .safety_tx
            .try_send(OperatorConsoleSoftwareSafetyStop {
                downstream_request_id: None,
                source: ConsoleDispatchSource::InternalFailClosed,
                received_at: None,
                response: None,
            })
            .expect("anonymous fail-closed signal");
        handle.shared.safety_latched.store(false, Ordering::Release);

        assert!(matches!(
            handle.submit(
                user,
                cap(),
                seq(1),
                key(1),
                OperatorConsoleIntent::SoftwareSafetyStop,
                at_ms(1),
            ),
            Err(OperatorConsoleSubmitError::SoftwareSafetyStopLatched)
        ));
        let state = handle.lock_state();
        assert!(state.latched_safety_stop_id.is_none());
        assert!(state.stop_pending.is_none());
        assert_eq!(state.next_downstream_id.map(NonZeroU64::get), Some(1));
        drop(state);
        assert!(handle.latest_requested_command().is_none());
        assert!(matches!(
            receiver
                .try_next()
                .expect("winning anonymous safety signal"),
            OperatorConsoleIngressItem::SoftwareSafetyStop(OperatorConsoleSoftwareSafetyStop {
                downstream_request_id: None,
                source: ConsoleDispatchSource::InternalFailClosed,
                ..
            })
        ));
    }

    #[test]
    fn body_si_is_finite_and_manual_deadman_has_priority() {
        let limits = OperatorConsoleLimits::parse(2, 4, 4, 8, 2, 100).unwrap();
        let (handle, receiver) = fixture_with_limits(limits);
        let user = handle
            .open_session(ConsoleSourceKind::Operator, cap(), at_ms(0))
            .unwrap();
        let velocity = FiniteManualVelocityV1::parse(0.2, -0.3).unwrap();
        assert!(FiniteManualVelocityV1::parse(f64::NAN, 0.0).is_err());
        handle
            .submit(
                user,
                cap(),
                seq(1),
                key(1),
                OperatorConsoleIntent::BeginManual,
                at_ms(0),
            )
            .unwrap();
        handle
            .submit(
                user,
                cap(),
                seq(2),
                key(2),
                OperatorConsoleIntent::ManualVelocity(velocity),
                at_ms(10),
            )
            .unwrap();
        assert!(!handle.tick_deadman(at_ms(109)).unwrap());
        assert!(handle.tick_deadman(at_ms(110)).unwrap());
        let first = receiver.try_next().unwrap();
        assert!(matches!(
            first,
            OperatorConsoleIngressItem::Dispatch(OperatorConsoleDispatch {
                command: OperatorConsoleCommand::Stop {
                    cause: ConsoleStopCause::ManualDeadman,
                },
                ..
            })
        ));
        assert!(matches!(receiver.try_next(), Err(TryRecvError::Empty)));
    }

    #[test]
    fn immediate_manual_release_cancels_queued_begin_before_it_can_reacquire() {
        let (handle, receiver) = fixture();
        let user = handle
            .open_session(ConsoleSourceKind::Operator, cap(), at_ms(0))
            .unwrap();
        handle
            .submit(
                user,
                cap(),
                seq(1),
                key(1),
                OperatorConsoleIntent::BeginManual,
                at_ms(0),
            )
            .unwrap();
        handle
            .submit(
                user,
                cap(),
                seq(2),
                key(2),
                OperatorConsoleIntent::ReleaseManual,
                at_ms(1),
            )
            .unwrap();
        let dispatch = match receiver.try_next().unwrap() {
            OperatorConsoleIngressItem::Dispatch(dispatch) => dispatch,
            OperatorConsoleIngressItem::SoftwareSafetyStop(_) => panic!("unexpected safety stop"),
        };
        assert!(matches!(
            dispatch.command(),
            OperatorConsoleCommand::Stop {
                cause: ConsoleStopCause::ManualRelease,
            }
        ));
        dispatch
            .response
            .reject(ConsoleResponseRejectionCode::RuntimeRejected);
        assert!(matches!(receiver.try_next(), Err(TryRecvError::Empty)));
        assert!(matches!(
            handle.submit(
                user,
                cap(),
                seq(3),
                key(3),
                OperatorConsoleIntent::BeginManual,
                at_ms(2),
            ),
            Err(OperatorConsoleSubmitError::StopPending)
        ));
    }

    #[test]
    fn non_safety_ingress_preserves_global_source_order_across_lanes() {
        let (handle, receiver) = fixture();
        let user = handle
            .open_session(ConsoleSourceKind::Operator, cap(), at_ms(0))
            .unwrap();
        handle
            .submit(
                user,
                cap(),
                seq(1),
                key(1),
                OperatorConsoleIntent::Arm,
                at_ms(0),
            )
            .unwrap();
        handle
            .submit(
                user,
                cap(),
                seq(2),
                key(2),
                OperatorConsoleIntent::BeginManual,
                at_ms(1),
            )
            .unwrap();
        let first = match receiver.try_next().unwrap() {
            OperatorConsoleIngressItem::Dispatch(dispatch) => dispatch,
            OperatorConsoleIngressItem::SoftwareSafetyStop(_) => panic!("unexpected safety stop"),
        };
        assert_eq!(first.downstream_request_id().get(), 1);
        assert!(matches!(first.command(), OperatorConsoleCommand::Arm));
        first
            .response
            .reject(ConsoleResponseRejectionCode::RuntimeRejected);
        let second = match receiver.try_next().unwrap() {
            OperatorConsoleIngressItem::Dispatch(dispatch) => dispatch,
            OperatorConsoleIngressItem::SoftwareSafetyStop(_) => panic!("unexpected safety stop"),
        };
        assert_eq!(second.downstream_request_id().get(), 2);
        assert!(matches!(
            second.command(),
            OperatorConsoleCommand::BeginManual
        ));
        second
            .response
            .reject(ConsoleResponseRejectionCode::RuntimeRejected);
    }

    #[test]
    fn mapping_only_stop_barrier_cancels_an_older_queued_arm() {
        let (handle, receiver) = fixture();
        let user = handle
            .open_session(ConsoleSourceKind::Operator, cap(), at_ms(0))
            .unwrap();
        let arm = handle
            .submit(
                user,
                cap(),
                seq(1),
                key(1),
                OperatorConsoleIntent::Arm,
                at_ms(0),
            )
            .unwrap();
        handle
            .submit(
                user,
                cap(),
                seq(2),
                key(2),
                OperatorConsoleIntent::AutonomousMapOnly,
                at_ms(1),
            )
            .unwrap();
        let map_only = match receiver.try_next().unwrap() {
            OperatorConsoleIngressItem::Dispatch(dispatch) => dispatch,
            OperatorConsoleIngressItem::SoftwareSafetyStop(_) => panic!("unexpected safety stop"),
        };
        assert!(matches!(
            map_only.command(),
            OperatorConsoleCommand::AutonomousMapOnly
        ));
        map_only
            .response
            .reject(ConsoleResponseRejectionCode::RuntimeRejected);
        assert!(matches!(receiver.try_next(), Err(TryRecvError::Empty)));
        let arm_record = handle
            .response_record(arm.downstream_request_id())
            .expect("queued arm response remains tracked");
        assert_eq!(
            arm_record.rejection_code,
            Some(ConsoleResponseRejectionCode::CancelledByPriorityStop)
        );
    }

    #[test]
    fn map_click_preserves_epoch_revision_and_finite_coordinates() {
        let (handle, receiver) = fixture();
        let user = handle
            .open_session(ConsoleSourceKind::Operator, cap(), at_ms(0))
            .unwrap();
        let selection = MapPointGoalSelection::parse(MapPointGoalSelectionDto {
            map_epoch_id: 7,
            displayed_revision: 42,
            x_m: -1.25,
            y_m: 2.5,
        })
        .unwrap();
        handle
            .submit(
                user,
                cap(),
                seq(1),
                key(1),
                OperatorConsoleIntent::AutonomousPointGoal(selection),
                at_ms(0),
            )
            .unwrap();
        let dispatch = match receiver.try_next().unwrap() {
            OperatorConsoleIngressItem::Dispatch(value) => value,
            OperatorConsoleIngressItem::SoftwareSafetyStop(_) => panic!("unexpected stop"),
        };
        match dispatch.command() {
            OperatorConsoleCommand::AutonomousPointGoal(actual) => {
                assert_eq!(actual.map_epoch_id().as_u64(), 7);
                assert_eq!(actual.displayed_revision(), 42);
                assert_eq!(actual.point().as_array(), [-1.25, 2.5]);
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn slow_snapshot_subscriber_gets_only_latest_without_blocking_publisher() {
        let (handle, _receiver) = fixture();
        let subscriber = handle.subscribe_snapshots().unwrap();
        for revision in 2..=100 {
            handle
                .publish_snapshot(OperatorConsoleSnapshot::unknown(
                    ConsoleSnapshotRevision::parse(revision).unwrap(),
                    ConsoleRuntimeAuthorityKind::ProductionExternalInterlocks,
                ))
                .unwrap();
        }
        let latest = subscriber.try_latest().unwrap().snapshot();
        assert_eq!(latest.revision.get(), 100);
        assert!(matches!(subscriber.try_latest(), Err(TryRecvError::Empty)));
    }

    #[test]
    fn telemetry_timestamp_does_not_claim_later_live_arbitration_overlays() {
        let (handle, _receiver) = fixture();
        let mut telemetry = OperatorConsoleSnapshot::unknown(
            ConsoleSnapshotRevision::parse(2).unwrap(),
            ConsoleRuntimeAuthorityKind::ProductionExternalInterlocks,
        );
        telemetry.telemetry_observed_at_host_monotonic_ns =
            Some(ConsoleHostTimestampNs::from_host(at_ms(10)));
        handle.publish_snapshot(telemetry).unwrap();

        let user = handle
            .open_session(ConsoleSourceKind::Operator, cap(), at_ms(20))
            .unwrap();
        handle
            .submit(
                user,
                cap(),
                seq(1),
                key(1),
                OperatorConsoleIntent::BeginManual,
                at_ms(20),
            )
            .unwrap();
        let overlaid = handle.latest_snapshot();
        assert_eq!(overlaid.revision.get(), 2);
        assert_eq!(
            overlaid.telemetry_observed_at_host_monotonic_ns,
            Some(ConsoleHostTimestampNs::from_host(at_ms(10)))
        );
        assert!(matches!(
            overlaid.requested_owner,
            Some(ConsoleRequestedOwner::Manual { session_id, .. }) if session_id == user
        ));
        let json = serde_json::to_value(overlaid.as_ref()).unwrap();
        assert_eq!(
            json["schema_version"],
            serde_json::Value::from(OPERATOR_CONSOLE_SNAPSHOT_SCHEMA_V5)
        );
        assert_eq!(
            json["authority_kind"],
            serde_json::json!("production_external_interlocks")
        );
        assert!(
            json.get("telemetry_observed_at_host_monotonic_ns")
                .is_some()
        );
        assert!(
            json.get("observed_at_host_monotonic_ns").is_none(),
            "the current schema must not expose a false aggregate timestamp"
        );
    }

    #[test]
    fn terminal_checkpoint_wire_state_cannot_claim_current_camera_localization() {
        let mut snapshot = OperatorConsoleSnapshot::unknown(
            ConsoleSnapshotRevision::parse(1).unwrap(),
            ConsoleRuntimeAuthorityKind::ProductionExternalInterlocks,
        );
        snapshot.runtime = Some(AgentRuntimeStateV1::ShuttingDown);
        snapshot.terminal = Some(ConsoleTerminalState::ControlEnding {
            reason: ConsoleTerminalReason::FinalizingWarmRestartCheckpoint,
            current_camera_localization: ConsoleCheckpointLocalizationEvidence::NotClaimed,
        });
        let json = serde_json::to_value(snapshot).expect("terminal snapshot JSON");
        assert_eq!(json["runtime"]["kind"], "shutting_down");
        assert_eq!(json["terminal"]["kind"], "control_ending");
        assert_eq!(
            json["terminal"]["reason"],
            "finalizing_warm_restart_checkpoint"
        );
        assert_eq!(
            json["terminal"]["current_camera_localization"],
            "not_claimed"
        );
    }

    #[test]
    fn accepted_and_completed_without_receipt_are_never_applied() {
        let (handle, receiver) = fixture();
        let user = handle
            .open_session(ConsoleSourceKind::Operator, cap(), at_ms(0))
            .unwrap();
        let outcome = handle
            .submit(
                user,
                cap(),
                seq(1),
                key(1),
                OperatorConsoleIntent::Arm,
                at_ms(0),
            )
            .unwrap();
        assert!(!outcome.is_applied());
        let dispatch = match receiver.try_next().unwrap() {
            OperatorConsoleIngressItem::Dispatch(value) => value,
            OperatorConsoleIngressItem::SoftwareSafetyStop(_) => panic!("unexpected stop"),
        };
        let id = dispatch.downstream_request_id();
        dispatch
            .response
            .reject(ConsoleResponseRejectionCode::RuntimeRejected);
        let record = handle.response_record(id).unwrap();
        assert_eq!(record.state, ConsoleRuntimeResponseState::Rejected);
        assert!(!record.applied);
        assert!(record.exact_receipt.is_none());

        handle
            .submit(
                user,
                cap(),
                seq(2),
                key(2),
                OperatorConsoleIntent::SaveMap,
                at_ms(1),
            )
            .unwrap();
        let dispatch = match receiver.try_next().unwrap() {
            OperatorConsoleIngressItem::Dispatch(value) => value,
            OperatorConsoleIngressItem::SoftwareSafetyStop(_) => panic!("unexpected stop"),
        };
        let id = dispatch.downstream_request_id();
        dispatch.response.completed().unwrap();
        let record = handle.response_record(id).unwrap();
        assert_eq!(record.state, ConsoleRuntimeResponseState::Completed);
        assert!(!record.applied);
        assert!(record.exact_receipt.is_none());
    }

    #[test]
    fn finite_grid_rejects_extent_collapsed_by_floating_point_rounding() {
        assert_eq!(
            ConsoleOccupancyGrid::parse(
                1,
                1,
                1,
                1,
                f64::MIN_POSITIVE,
                f64::MAX / 2.0,
                f64::MAX / 2.0,
                vec![0],
            ),
            Err(ConsoleGridError::UnrepresentableExtent)
        );
    }

    #[test]
    fn occupancy_snapshot_projection_binds_identity_and_copies_cells_once() {
        use crate::dense::occupancy::{
            OccupancyCell, OccupancyGridGeometry, OccupancyGridSnapshot,
        };
        use crate::map::SlamMap;
        use crate::navigation::{NavigationClockEpoch, NavigationMapEpochCoordinator};

        let map = SlamMap::new();
        let now = at_ms(0);
        let mut epochs = NavigationMapEpochCoordinator::new();
        let map_instance_id = map.snapshot().instance_id();
        let binding = epochs
            .start_epoch(NavigationClockEpoch::new(now), now, map_instance_id)
            .unwrap()
            .binding();
        let geometry = OccupancyGridGeometry::try_new(0.5, [-1.0, 2.0], 2, 2, 4).unwrap();
        let snapshot = OccupancyGridSnapshot::from_test_cells(
            geometry,
            &[
                OccupancyCell::Unknown,
                OccupancyCell::Free,
                OccupancyCell::Occupied,
                OccupancyCell::Free,
            ],
            map_instance_id,
            7,
        );
        let source_cells = snapshot.class_ids().as_ptr();
        let projected = ConsoleOccupancyGrid::from_snapshot(binding, &snapshot).unwrap();
        assert_eq!(projected.map_epoch_id.get(), 1);
        assert_eq!(projected.revision, 7);
        assert_eq!(projected.metadata.width.get(), 2);
        assert_eq!(projected.metadata.height.get(), 2);
        assert_eq!(projected.metadata.resolution_m_per_cell.get(), 0.5);
        assert_eq!(projected.metadata.origin_x_m.get(), -1.0);
        assert_eq!(projected.metadata.origin_y_m.get(), 2.0);
        assert_eq!(projected.cells, [0, 1, 2, 1]);
        assert_ne!(projected.cells.as_ptr(), source_cells);

        let replacement_map = SlamMap::new();
        let replacement_map_instance_id = replacement_map.snapshot().instance_id();
        let replacement_binding = epochs
            .start_epoch(
                NavigationClockEpoch::new(now),
                now,
                replacement_map_instance_id,
            )
            .unwrap()
            .binding();
        assert!(matches!(
            ConsoleOccupancyGrid::from_snapshot(replacement_binding, &snapshot),
            Err(ConsoleGridProjectionError::MapBindingMismatch { .. })
        ));
    }

    #[test]
    fn poisoned_arbitration_state_enqueues_a_real_fail_closed_signal() {
        let (handle, receiver) = fixture();
        let poisoner = handle.clone();
        assert!(
            std::thread::spawn(move || {
                let _guard = poisoner.shared.state.lock().unwrap();
                panic!("intentional state poison");
            })
            .join()
            .is_err()
        );

        let _ = handle.open_session(ConsoleSourceKind::Operator, cap(), at_ms(0));
        let signal = receiver
            .try_next()
            .expect("poison recovery must signal the sole owner");
        assert!(matches!(
            signal,
            OperatorConsoleIngressItem::SoftwareSafetyStop(OperatorConsoleSoftwareSafetyStop {
                source: ConsoleDispatchSource::InternalFailClosed,
                ..
            })
        ));
        assert!(handle.software_safety_stop_latched());
    }

    #[test]
    fn critical_reserve_never_evicts_an_outstanding_response() {
        let mut ledger = ResponseLedger {
            maximum: 1,
            records: VecDeque::new(),
        };
        for raw in 1..=1 + OPERATOR_CONSOLE_CRITICAL_RESPONSE_RESERVE as u64 {
            let id = ConsoleDownstreamRequestId(NonZeroU64::new(raw).unwrap());
            assert_eq!(
                ledger
                    .insert_pending(id, OperatorConsoleIntentKind::Stop, true, raw != 1, None,)
                    .unwrap(),
                None
            );
        }
        let overflow = ConsoleDownstreamRequestId(
            NonZeroU64::new(2 + OPERATOR_CONSOLE_CRITICAL_RESPONSE_RESERVE as u64).unwrap(),
        );
        assert!(matches!(
            ledger.insert_pending(overflow, OperatorConsoleIntentKind::Stop, true, true, None,),
            Err(OperatorConsoleSubmitError::ResponseCapacityReached)
        ));

        let oldest = ConsoleDownstreamRequestId(NonZeroU64::new(1).unwrap());
        ledger.update(
            oldest,
            ConsoleRuntimeResponseState::Rejected,
            None,
            Some(ConsoleResponseRejectionCode::RuntimeRejected),
        );
        assert_eq!(
            ledger
                .insert_pending(overflow, OperatorConsoleIntentKind::Stop, true, true, None,)
                .unwrap(),
            Some(oldest)
        );
        assert_eq!(
            ledger.records.len(),
            1 + OPERATOR_CONSOLE_CRITICAL_RESPONSE_RESERVE
        );
    }

    #[test]
    fn http_observation_tracks_only_the_current_retained_record_version() {
        let mut ledger = ResponseLedger {
            maximum: 1,
            records: VecDeque::new(),
        };
        let owner = ConsoleSessionId::parse(1).unwrap();
        let foreign = ConsoleSessionId::parse(2).unwrap();
        let first = ConsoleDownstreamRequestId(NonZeroU64::new(1).unwrap());
        assert_eq!(
            ledger
                .insert_pending(
                    first,
                    OperatorConsoleIntentKind::SaveMap,
                    false,
                    false,
                    Some(owner),
                )
                .unwrap(),
            None
        );

        assert!(ledger.observe_for_http_session(first, foreign).is_none());
        assert!(!ledger.current_record_was_http_observed(first));
        assert!(ledger.observe_for_http_session(first, owner).is_some());
        assert!(ledger.current_record_was_http_observed(first));

        ledger.update(first, ConsoleRuntimeResponseState::Completed, None, None);
        assert!(
            !ledger.current_record_was_http_observed(first),
            "observing a pending response must not count as observing its completion"
        );
        assert!(ledger.observe_for_http_session(first, owner).is_some());
        assert!(ledger.current_record_was_http_observed(first));

        let second = ConsoleDownstreamRequestId(NonZeroU64::new(2).unwrap());
        assert_eq!(
            ledger
                .insert_pending(
                    second,
                    OperatorConsoleIntentKind::SaveMap,
                    false,
                    false,
                    Some(owner),
                )
                .unwrap(),
            Some(first)
        );
        assert!(
            !ledger.current_record_was_http_observed(first),
            "observation evidence must be evicted with its response record"
        );
    }

    #[test]
    fn forced_critical_eviction_removes_the_matching_idempotency_replay() {
        let limits = OperatorConsoleLimits::parse(1, 1, 64, 1, 1, 100).unwrap();
        let (handle, _receiver) = fixture_with_limits(limits);
        let session_id = handle
            .open_session(ConsoleSourceKind::Operator, cap(), at_ms(0))
            .unwrap();
        let mut state = handle.lock_state();
        for raw in 1..=1 + OPERATOR_CONSOLE_CRITICAL_RESPONSE_RESERVE as u64 {
            let id = ConsoleDownstreamRequestId(NonZeroU64::new(raw).unwrap());
            recover_lock(&handle.shared.responses)
                .insert_pending(
                    id,
                    OperatorConsoleIntentKind::Stop,
                    true,
                    true,
                    Some(session_id),
                )
                .unwrap();
            recover_lock(&handle.shared.responses).update(
                id,
                ConsoleRuntimeResponseState::Rejected,
                None,
                Some(ConsoleResponseRejectionCode::RuntimeRejected),
            );
            state
                .sessions
                .get_mut(&session_id)
                .unwrap()
                .idempotency
                .push_back(CachedSubmission {
                    key: key(raw),
                    intent: OperatorConsoleIntent::Stop,
                    outcome: OperatorConsoleSubmitOutcome::AcceptedForProcessing {
                        downstream_request_id: id,
                    },
                });
        }
        let next = ConsoleDownstreamRequestId(
            NonZeroU64::new(2 + OPERATOR_CONSOLE_CRITICAL_RESPONSE_RESERVE as u64).unwrap(),
        );
        let response = handle
            .response_token(
                &mut state,
                next,
                ConsoleResponseTokenSpec {
                    intent: OperatorConsoleIntentKind::Stop,
                    owner_acquisition: None,
                    idempotency_pinned: false,
                    critical: true,
                    stop_barrier: true,
                    source_session_id: Some(session_id),
                },
            )
            .unwrap();
        let session = state.sessions.get(&session_id).unwrap();
        assert!(
            session
                .idempotency
                .iter()
                .all(|cached| cached.outcome.downstream_request_id().get() != 1)
        );
        assert!(
            session
                .idempotency
                .iter()
                .any(|cached| cached.outcome.downstream_request_id().get() == 2)
        );
        drop(state);
        response.reject(ConsoleResponseRejectionCode::RuntimeRejected);
        assert!(
            handle
                .response_record(ConsoleDownstreamRequestId(NonZeroU64::new(1).unwrap()))
                .is_none()
        );
    }
}
