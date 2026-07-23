//! Loopback-only HTTP owner for wheels-off qualification.
//!
//! This module is compiled only into the explicit
//! `nano-wheels-off-qualification` build. It shares hardened capability-file
//! primitives and static assets with the production console, but it has a
//! distinct backend, DTO parser, session arbiter, intent endpoint, telemetry
//! projection, deadman task, and lifecycle owner. The production
//! `/api/v1/intents` endpoint is intentionally absent.

use std::convert::Infallible;
use std::fmt;
use std::net::SocketAddr;
use std::num::NonZeroU64;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, mpsc};
use std::thread;
use std::time::{Duration, Instant};

use bytes::{Bytes, BytesMut};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use tokio::sync::{Notify, Semaphore, oneshot};
use warp::http::{HeaderMap, Method, StatusCode};
use warp::hyper::body::HttpBody;
use warp::hyper::service::{make_service_fn, service_fn};
use warp::hyper::{Body, Request, Response, Server};

use super::operator_console_http::{
    APP_JS, INDEX_HTML, STYLES_CSS, encode_hex, error_response, json_response, secure_response,
    text_response, valid_host,
};
use super::{
    AgentControlMonotonicOrigin, ConsoleActualAuthority, ConsoleAppliedReceipt,
    ConsoleHostTimestampNs, ConsoleManualCommandEnvelope, ConsoleMapSnapshot,
    ConsoleNavigationSnapshot, ConsoleOccupancyGrid, ConsolePhysicalEmergencyStopState,
    ConsoleRequestedActuation, ConsoleRequestedCommand, ConsoleRequestedOwner,
    ConsoleSafetySignalState, ConsoleSnapshotRevision, ConsoleSourceKind, ConsoleStopCertainty,
    ConsoleSubsystemHealth, OperatorConsoleAccessCapability, OperatorConsoleBind,
    OperatorConsoleBindError, OperatorConsoleCapabilityCleanupEvidence,
    OperatorConsoleCapabilityPersistError, OperatorConsolePersistedAccessCapability,
    OperatorConsoleSnapshot, WheelsOffQualificationConsoleHandle,
    WheelsOffQualificationControlProfile, WheelsOffQualificationIntentRequestDto,
    WheelsOffQualificationRequestParseError, WheelsOffQualificationRuntimeIngressState,
    WheelsOffQualificationSnapshot, WheelsOffQualificationSubmitError,
};

const ACCESS_CAPABILITY_BYTES: usize = 32;
const MAX_QUALIFICATION_HTTP_REQUEST_BYTES: usize = 8 * 1_024;
const MAX_CONCURRENT_QUALIFICATION_HTTP_REQUESTS: usize = 32;
const QUALIFICATION_HTTP_REQUEST_TIMEOUT: Duration = Duration::from_millis(750);
const QUALIFICATION_HTTP_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(3);
const QUALIFICATION_HTTP_FORCE_SHUTDOWN_AFTER: Duration = Duration::from_secs(1);
const MIN_DEADMAN_TICK_MS: u64 = 5;
const MAX_DEADMAN_TICK_MS: u64 = 100;

#[derive(Clone, Debug)]
pub struct WheelsOffQualificationFrontendConfig {
    bind: OperatorConsoleBind,
    capability_path: PathBuf,
    clock: AgentControlMonotonicOrigin,
    deadman_tick: Duration,
}

impl WheelsOffQualificationFrontendConfig {
    pub fn parse(
        bind_address: SocketAddr,
        capability_path: PathBuf,
        clock: AgentControlMonotonicOrigin,
        deadman_tick: Duration,
    ) -> Result<Self, WheelsOffQualificationFrontendConfigError> {
        let bind = OperatorConsoleBind::parse(bind_address)
            .map_err(WheelsOffQualificationFrontendConfigError::Bind)?;
        if !capability_path.is_absolute() || capability_path.file_name().is_none() {
            return Err(
                WheelsOffQualificationFrontendConfigError::InvalidCapabilityPath(capability_path),
            );
        }
        let millis = u64::try_from(deadman_tick.as_millis())
            .map_err(|_| WheelsOffQualificationFrontendConfigError::DeadmanTickOutOfRange)?;
        if !(MIN_DEADMAN_TICK_MS..=MAX_DEADMAN_TICK_MS).contains(&millis)
            || Duration::from_millis(millis) != deadman_tick
        {
            return Err(WheelsOffQualificationFrontendConfigError::DeadmanTickOutOfRange);
        }
        Ok(Self {
            bind,
            capability_path,
            clock,
            deadman_tick,
        })
    }

    pub const fn bind(&self) -> OperatorConsoleBind {
        self.bind
    }

    pub fn capability_path(&self) -> &Path {
        &self.capability_path
    }

    pub const fn deadman_tick(&self) -> Duration {
        self.deadman_tick
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WheelsOffQualificationFrontendConfigError {
    Bind(OperatorConsoleBindError),
    InvalidCapabilityPath(PathBuf),
    DeadmanTickOutOfRange,
}

impl fmt::Display for WheelsOffQualificationFrontendConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bind(source) => source.fmt(formatter),
            Self::InvalidCapabilityPath(path) => write!(
                formatter,
                "qualification capability path must be an absolute file path: {}",
                path.display()
            ),
            Self::DeadmanTickOutOfRange => formatter
                .write_str("qualification HTTP deadman tick must be an exact 5..=100 milliseconds"),
        }
    }
}

impl std::error::Error for WheelsOffQualificationFrontendConfigError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Bind(source) => Some(source),
            Self::InvalidCapabilityPath(_) | Self::DeadmanTickOutOfRange => None,
        }
    }
}

#[derive(Debug)]
struct QualificationTelemetryState {
    profile: WheelsOffQualificationControlProfile,
    base: OperatorConsoleSnapshot,
    base_source_revision: ConsoleSnapshotRevision,
    projection_revision: NonZeroU64,
    observed_qualification_revision: NonZeroU64,
    serialized_projection: Option<Arc<Vec<u8>>>,
    exact_grid: Option<Arc<ConsoleOccupancyGrid>>,
}

#[derive(Clone, Debug)]
pub struct WheelsOffQualificationTelemetryStore {
    state: Arc<Mutex<QualificationTelemetryState>>,
    poison_latched: Arc<AtomicBool>,
}

impl WheelsOffQualificationTelemetryStore {
    pub fn parse(
        profile: WheelsOffQualificationControlProfile,
        initial_base: OperatorConsoleSnapshot,
        initial_qualification: WheelsOffQualificationSnapshot,
    ) -> Result<Self, WheelsOffQualificationTelemetryError> {
        validate_observational_base(&initial_base)?;
        if initial_qualification.schema_version != super::WHEELS_OFF_QUALIFICATION_SCHEMA_V1 {
            return Err(
                WheelsOffQualificationTelemetryError::UnsupportedQualificationSchema(
                    initial_qualification.schema_version,
                ),
            );
        }
        if initial_qualification.control_profile != profile {
            return Err(WheelsOffQualificationTelemetryError::ProfileMismatch);
        }
        Ok(Self {
            state: Arc::new(Mutex::new(QualificationTelemetryState {
                profile,
                base_source_revision: initial_base.revision,
                base: initial_base,
                projection_revision: NonZeroU64::new(1).expect("one is nonzero"),
                observed_qualification_revision: initial_qualification.revision,
                serialized_projection: None,
                exact_grid: None,
            })),
            poison_latched: Arc::new(AtomicBool::new(false)),
        })
    }

    fn lock_state(
        &self,
    ) -> Result<MutexGuard<'_, QualificationTelemetryState>, WheelsOffQualificationTelemetryError>
    {
        if self.poison_latched.load(Ordering::Acquire) {
            return Err(WheelsOffQualificationTelemetryError::Poisoned);
        }
        match self.state.lock() {
            Ok(state) => {
                if self.poison_latched.load(Ordering::Acquire) {
                    Err(WheelsOffQualificationTelemetryError::Poisoned)
                } else {
                    Ok(state)
                }
            }
            Err(_) => {
                self.poison_latched.store(true, Ordering::Release);
                Err(WheelsOffQualificationTelemetryError::Poisoned)
            }
        }
    }

    fn runtime_known(&self) -> Result<bool, WheelsOffQualificationTelemetryError> {
        Ok(self.lock_state()?.base.runtime.is_some())
    }

    pub fn control_profile(
        &self,
    ) -> Result<WheelsOffQualificationControlProfile, WheelsOffQualificationTelemetryError> {
        Ok(self.lock_state()?.profile)
    }

    /// Publish navigation/SLAM observations only. Production authority,
    /// manual-envelope, request, and software-latch fields are rejected rather
    /// than silently hidden.
    pub fn publish_observational_base(
        &self,
        snapshot: OperatorConsoleSnapshot,
    ) -> Result<(), WheelsOffQualificationTelemetryError> {
        validate_observational_base(&snapshot)?;
        let mut state = self.lock_state()?;
        if snapshot.revision <= state.base_source_revision {
            return Err(
                WheelsOffQualificationTelemetryError::BaseRevisionNotIncreasing {
                    previous: state.base_source_revision,
                    current: snapshot.revision,
                },
            );
        }
        bump_projection_revision(&mut state)?;
        state.base_source_revision = snapshot.revision;
        state.base = snapshot;
        state.serialized_projection = None;
        Ok(())
    }

    /// Publish one exact already-validated grid without cloning it on GET.
    pub fn publish_grid(
        &self,
        grid: ConsoleOccupancyGrid,
    ) -> Result<bool, WheelsOffQualificationTelemetryError> {
        let mut state = self.lock_state()?;
        if let Some(current) = state.exact_grid.as_ref() {
            let current_identity = (current.map_epoch_id.get(), current.revision);
            let incoming_identity = (grid.map_epoch_id.get(), grid.revision);
            if current_identity == incoming_identity {
                if current.as_ref() == &grid {
                    return Ok(false);
                }
                return Err(WheelsOffQualificationTelemetryError::GridIdentityConflict {
                    map_epoch_id: grid.map_epoch_id,
                    revision: grid.revision,
                });
            }
            if current_identity > incoming_identity {
                return Ok(false);
            }
        }
        state.exact_grid = Some(Arc::new(grid));
        Ok(true)
    }

    pub fn exact_grid(
        &self,
        map_epoch_id: NonZeroU64,
        revision: u64,
    ) -> Result<Option<Arc<ConsoleOccupancyGrid>>, WheelsOffQualificationTelemetryError> {
        Ok(self
            .lock_state()?
            .exact_grid
            .as_ref()
            .filter(|grid| grid.map_epoch_id == map_epoch_id && grid.revision == revision)
            .map(Arc::clone))
    }

    fn serialized_projection(
        &self,
        qualification: WheelsOffQualificationSnapshot,
    ) -> Result<Arc<Vec<u8>>, WheelsOffQualificationTelemetryError> {
        let mut state = self.lock_state()?;
        if qualification.control_profile != state.profile {
            return Err(WheelsOffQualificationTelemetryError::ProfileMismatch);
        }
        if qualification.revision < state.observed_qualification_revision {
            return Err(
                WheelsOffQualificationTelemetryError::QualificationRevisionRegressed {
                    previous: state.observed_qualification_revision,
                    current: qualification.revision,
                },
            );
        }
        if qualification.revision > state.observed_qualification_revision {
            bump_projection_revision(&mut state)?;
            state.observed_qualification_revision = qualification.revision;
            state.serialized_projection = None;
        }
        if let Some(serialized) = state.serialized_projection.as_ref() {
            return Ok(Arc::clone(serialized));
        }
        let projection_revision =
            ConsoleSnapshotRevision::parse(state.projection_revision.get())
                .map_err(|_| WheelsOffQualificationTelemetryError::ProjectionRevisionExhausted)?;
        let signal_state = qualification_signal_state(&qualification);
        let projection = QualificationSnapshotProjection {
            schema_version: state.base.schema_version,
            revision: projection_revision,
            telemetry_observed_at_host_monotonic_ns: &state
                .base
                .telemetry_observed_at_host_monotonic_ns,
            runtime: &state.base.runtime,
            requested_owner: None,
            actual_authority: None,
            manual_command_envelope: None,
            map: &state.base.map,
            navigation: &state.base.navigation,
            last_requested: None,
            last_requested_actuation: &state.base.last_requested_actuation,
            last_applied: &state.base.last_applied,
            stop_certainty: &state.base.stop_certainty,
            health: &state.base.health,
            software_safety_stop_latched: qualification.software_safety_stop_latched,
            software_safety_signal_state: signal_state,
            physical_emergency_stop_state: state.base.physical_emergency_stop_state,
            rerun_diagnostics_url: &state.base.rerun_diagnostics_url,
            control_profile: state.profile,
            wheels_off_qualification: &qualification,
        };
        let serialized = Arc::new(
            serde_json::to_vec(&projection)
                .map_err(WheelsOffQualificationTelemetryError::Serialize)?,
        );
        state.serialized_projection = Some(Arc::clone(&serialized));
        Ok(serialized)
    }
}

fn bump_projection_revision(
    state: &mut QualificationTelemetryState,
) -> Result<(), WheelsOffQualificationTelemetryError> {
    state.projection_revision = state
        .projection_revision
        .get()
        .checked_add(1)
        .and_then(NonZeroU64::new)
        .ok_or(WheelsOffQualificationTelemetryError::ProjectionRevisionExhausted)?;
    Ok(())
}

fn validate_observational_base(
    snapshot: &OperatorConsoleSnapshot,
) -> Result<(), WheelsOffQualificationTelemetryError> {
    if snapshot.schema_version != super::OPERATOR_CONSOLE_SNAPSHOT_SCHEMA_V2 {
        return Err(WheelsOffQualificationTelemetryError::UnsupportedBaseSchema(
            snapshot.schema_version,
        ));
    }
    if snapshot.requested_owner.is_some()
        || snapshot.actual_authority.is_some()
        || snapshot.manual_command_envelope.is_some()
        || snapshot.last_requested.is_some()
    {
        return Err(WheelsOffQualificationTelemetryError::ProductionControlStatePresent);
    }
    if snapshot.software_safety_stop_latched
        || snapshot.software_safety_signal_state != ConsoleSafetySignalState::NotLatched
    {
        return Err(WheelsOffQualificationTelemetryError::ProductionSafetyLatchPresent);
    }
    Ok(())
}

fn qualification_signal_state(
    qualification: &WheelsOffQualificationSnapshot,
) -> ConsoleSafetySignalState {
    if !qualification.software_safety_stop_latched {
        return ConsoleSafetySignalState::NotLatched;
    }
    if qualification.runtime_ingress_state
        == WheelsOffQualificationRuntimeIngressState::DisconnectedStopUnconfirmed
    {
        return ConsoleSafetySignalState::RuntimeAdapterDisconnected;
    }
    if qualification.stop_barrier_pending {
        ConsoleSafetySignalState::PendingRuntimeDrain
    } else {
        ConsoleSafetySignalState::CompletedFaultLatched
    }
}

#[derive(Debug)]
pub enum WheelsOffQualificationTelemetryError {
    Poisoned,
    UnsupportedBaseSchema(u32),
    UnsupportedQualificationSchema(u32),
    ProductionControlStatePresent,
    ProductionSafetyLatchPresent,
    ProfileMismatch,
    BaseRevisionNotIncreasing {
        previous: ConsoleSnapshotRevision,
        current: ConsoleSnapshotRevision,
    },
    GridIdentityConflict {
        map_epoch_id: NonZeroU64,
        revision: u64,
    },
    QualificationRevisionRegressed {
        previous: NonZeroU64,
        current: NonZeroU64,
    },
    ProjectionRevisionExhausted,
    Serialize(serde_json::Error),
}

impl fmt::Display for WheelsOffQualificationTelemetryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid wheels-off qualification telemetry projection: {self:?}"
        )
    }
}

impl std::error::Error for WheelsOffQualificationTelemetryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Serialize(source) => Some(source),
            Self::Poisoned
            | Self::UnsupportedBaseSchema(_)
            | Self::UnsupportedQualificationSchema(_)
            | Self::ProductionControlStatePresent
            | Self::ProductionSafetyLatchPresent
            | Self::ProfileMismatch
            | Self::BaseRevisionNotIncreasing { .. }
            | Self::GridIdentityConflict { .. }
            | Self::QualificationRevisionRegressed { .. }
            | Self::ProjectionRevisionExhausted => None,
        }
    }
}

#[derive(Serialize)]
struct QualificationSnapshotProjection<'a> {
    schema_version: u32,
    revision: ConsoleSnapshotRevision,
    telemetry_observed_at_host_monotonic_ns: &'a Option<ConsoleHostTimestampNs>,
    runtime: &'a Option<super::AgentRuntimeStateV1>,
    requested_owner: Option<ConsoleRequestedOwner>,
    actual_authority: Option<ConsoleActualAuthority>,
    manual_command_envelope: Option<ConsoleManualCommandEnvelope>,
    map: &'a Option<ConsoleMapSnapshot>,
    navigation: &'a Option<ConsoleNavigationSnapshot>,
    last_requested: Option<ConsoleRequestedCommand>,
    last_requested_actuation: &'a Option<ConsoleRequestedActuation>,
    last_applied: &'a Option<ConsoleAppliedReceipt>,
    stop_certainty: &'a Option<ConsoleStopCertainty>,
    health: &'a ConsoleSubsystemHealth,
    software_safety_stop_latched: bool,
    software_safety_signal_state: ConsoleSafetySignalState,
    physical_emergency_stop_state: ConsolePhysicalEmergencyStopState,
    rerun_diagnostics_url: &'a Option<String>,
    control_profile: WheelsOffQualificationControlProfile,
    wheels_off_qualification: &'a WheelsOffQualificationSnapshot,
}

#[derive(Debug)]
struct QualificationHttpContext {
    console: WheelsOffQualificationConsoleHandle,
    telemetry: WheelsOffQualificationTelemetryStore,
    profile: WheelsOffQualificationControlProfile,
    access_capability: OperatorConsoleAccessCapability,
    clock: AgentControlMonotonicOrigin,
    bound_port: u16,
    request_permits: Arc<Semaphore>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OpenSessionDto {
    schema_version: u32,
    source: ConsoleSourceKind,
}

#[derive(Debug, Serialize)]
struct OpenSessionResponse {
    schema_version: u32,
    session_id: super::ConsoleSessionId,
    session_capability: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CloseSessionDto {
    schema_version: u32,
    session_id: String,
}

#[derive(Debug, Serialize)]
struct CloseSessionResponse {
    schema_version: u32,
    closed: bool,
    stop_queued: bool,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    schema_version: u32,
    http_ready: bool,
    runtime_known: bool,
    qualification_profile: super::WheelsOffQualificationProfileKind,
}

#[derive(Debug)]
struct GridBytesOwner(Arc<ConsoleOccupancyGrid>);

impl AsRef<[u8]> for GridBytesOwner {
    fn as_ref(&self) -> &[u8] {
        &self.0.cells
    }
}

#[derive(Debug)]
struct SerializedBytesOwner(Arc<Vec<u8>>);

impl AsRef<[u8]> for SerializedBytesOwner {
    fn as_ref(&self) -> &[u8] {
        self.0.as_slice()
    }
}

#[derive(Debug)]
enum RequestBodyError {
    TimedOut,
    TooLarge,
    InvalidLength,
    Transport,
}

async fn handle_qualification_request(
    request: Request<Body>,
    context: Arc<QualificationHttpContext>,
) -> Result<Response<Body>, Infallible> {
    Ok(handle_qualification_request_inner(request, context).await)
}

async fn handle_qualification_request_inner(
    request: Request<Body>,
    context: Arc<QualificationHttpContext>,
) -> Response<Body> {
    let method = request.method().clone();
    let path = request.uri().path().to_owned();
    if request.uri().query().is_some() {
        return error_response(StatusCode::BAD_REQUEST, "query_not_allowed");
    }
    let Some(host) = single_header(request.headers(), "host") else {
        return error_response(StatusCode::BAD_REQUEST, "invalid_loopback_host");
    };
    if !valid_host(host, context.bound_port) {
        return error_response(StatusCode::BAD_REQUEST, "invalid_loopback_host");
    }
    if method == Method::GET
        && let Some(response) = static_get(&path)
    {
        return response;
    }
    if !path.starts_with("/api/v1/") {
        return error_response(StatusCode::NOT_FOUND, "not_found");
    }
    if !access_capability_matches(request.headers(), context.access_capability) {
        return error_response(StatusCode::UNAUTHORIZED, "unauthorized");
    }
    if method != Method::GET && method != Method::POST {
        return error_response(StatusCode::METHOD_NOT_ALLOWED, "method_not_allowed");
    }
    let expected_origin = format!("http://{host}");
    let origin = optional_single_header(request.headers(), "origin");
    let origin_valid = match method {
        Method::GET => origin
            .is_ok_and(|origin| origin.is_none_or(|origin| origin == expected_origin.as_str())),
        Method::POST => origin == Ok(Some(expected_origin.as_str())),
        _ => false,
    };
    if !origin_valid {
        return error_response(StatusCode::FORBIDDEN, "invalid_origin");
    }
    let Ok(_permit) = Arc::clone(&context.request_permits).try_acquire_owned() else {
        return error_response(StatusCode::TOO_MANY_REQUESTS, "too_many_requests");
    };
    match method {
        Method::GET => handle_qualification_get(&path, request.headers(), &context),
        Method::POST => handle_qualification_post(path, request, &context).await,
        _ => error_response(StatusCode::METHOD_NOT_ALLOWED, "method_not_allowed"),
    }
}

fn static_get(path: &str) -> Option<Response<Body>> {
    match path {
        "/" | "/index.html" => Some(text_response(
            StatusCode::OK,
            "text/html; charset=utf-8",
            INDEX_HTML,
        )),
        "/assets/styles.css" => Some(text_response(
            StatusCode::OK,
            "text/css; charset=utf-8",
            STYLES_CSS,
        )),
        "/assets/app.js" => Some(text_response(
            StatusCode::OK,
            "text/javascript; charset=utf-8",
            APP_JS,
        )),
        _ => None,
    }
}

fn handle_qualification_get(
    path: &str,
    headers: &HeaderMap,
    context: &QualificationHttpContext,
) -> Response<Body> {
    match path {
        "/api/v1/health" => {
            let base_runtime_known = match context.telemetry.runtime_known() {
                Ok(runtime_known) => runtime_known,
                Err(_) => {
                    context.console.signal_internal_fail_closed(None);
                    return error_response(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "telemetry_store_fault",
                    );
                }
            };
            json_response(
                StatusCode::OK,
                &HealthResponse {
                    schema_version: 1,
                    http_ready: true,
                    runtime_known: base_runtime_known,
                    qualification_profile: context.profile.kind(),
                },
            )
        }
        "/api/v1/snapshot" => match context
            .telemetry
            .serialized_projection(context.console.snapshot())
        {
            Ok(serialized) => secure_response(
                StatusCode::OK,
                Some("application/json"),
                Bytes::from_owner(SerializedBytesOwner(serialized)),
                &[],
            ),
            Err(_) => {
                context.console.signal_internal_fail_closed(None);
                error_response(StatusCode::SERVICE_UNAVAILABLE, "snapshot_projection_fault")
            }
        },
        _ if path.starts_with("/api/v1/maps/") => {
            qualification_grid_response(path, headers, context)
        }
        _ => error_response(StatusCode::NOT_FOUND, "not_found"),
    }
}

async fn handle_qualification_post(
    path: String,
    request: Request<Body>,
    context: &QualificationHttpContext,
) -> Response<Body> {
    if !matches!(
        path.as_str(),
        "/api/v1/sessions"
            | "/api/v1/sessions/close"
            | super::WHEELS_OFF_QUALIFICATION_INTENT_ENDPOINT
    ) {
        return error_response(StatusCode::NOT_FOUND, "not_found");
    }
    if !content_type_is_exact_json(request.headers()) {
        return error_response(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "application_json_required",
        );
    }
    let authorized_session = if path == "/api/v1/sessions" {
        None
    } else {
        match authenticated_session(request.headers(), &context.console) {
            Some(session) => Some(session),
            None => return error_response(StatusCode::UNAUTHORIZED, "invalid_session"),
        }
    };
    let body = match read_bounded_body(request.into_body()).await {
        Ok(body) => body,
        Err(RequestBodyError::TimedOut) => {
            return error_response(StatusCode::REQUEST_TIMEOUT, "request_body_timeout");
        }
        Err(RequestBodyError::TooLarge) => {
            return error_response(StatusCode::PAYLOAD_TOO_LARGE, "request_body_too_large");
        }
        Err(RequestBodyError::InvalidLength | RequestBodyError::Transport) => {
            return error_response(StatusCode::BAD_REQUEST, "invalid_http_body");
        }
    };
    match path.as_str() {
        "/api/v1/sessions" => open_session_response(context, &body),
        "/api/v1/sessions/close" => close_session_response(context, authorized_session, &body),
        super::WHEELS_OFF_QUALIFICATION_INTENT_ENDPOINT => {
            qualification_intent_response(context, authorized_session, &body)
        }
        _ => error_response(StatusCode::NOT_FOUND, "not_found"),
    }
}

fn open_session_response(context: &QualificationHttpContext, body: &[u8]) -> Response<Body> {
    let dto = match parse_exact_json::<OpenSessionDto>(body) {
        Ok(dto) if dto.schema_version == 1 => dto,
        _ => return error_response(StatusCode::BAD_REQUEST, "invalid_session_request"),
    };
    let mut bytes = [0_u8; ACCESS_CAPABILITY_BYTES];
    if getrandom::fill(&mut bytes).is_err() || bytes == [0; ACCESS_CAPABILITY_BYTES] {
        context.console.signal_internal_fail_closed(None);
        return error_response(StatusCode::INTERNAL_SERVER_ERROR, "entropy_unavailable");
    }
    let capability = super::ConsoleSessionCapability::from_bytes(bytes);
    match context.console.open_session(dto.source, capability) {
        Ok(session_id) => json_response(
            StatusCode::CREATED,
            &OpenSessionResponse {
                schema_version: 1,
                session_id,
                session_capability: encode_hex(capability.as_bytes()),
            },
        ),
        Err(WheelsOffQualificationSubmitError::SessionCapacityReached) => {
            error_response(StatusCode::TOO_MANY_REQUESTS, "session_capacity_reached")
        }
        Err(_) => error_response(StatusCode::CONFLICT, "session_unavailable"),
    }
}

fn close_session_response(
    context: &QualificationHttpContext,
    authorized_session: Option<(super::ConsoleSessionId, super::ConsoleSessionCapability)>,
    body: &[u8],
) -> Response<Body> {
    let Some((authorized_session_id, capability)) = authorized_session else {
        return error_response(StatusCode::UNAUTHORIZED, "invalid_session");
    };
    let dto = match parse_exact_json::<CloseSessionDto>(body) {
        Ok(dto) if dto.schema_version == 1 => dto,
        _ => return error_response(StatusCode::BAD_REQUEST, "invalid_close_request"),
    };
    let Some(session_id) = parse_session_id(&dto.session_id) else {
        return error_response(StatusCode::BAD_REQUEST, "invalid_session_id");
    };
    if session_id != authorized_session_id {
        return error_response(StatusCode::UNAUTHORIZED, "invalid_session");
    }
    let now = match context.clock.try_now() {
        Ok(now) => now,
        Err(_) => {
            context.console.signal_internal_fail_closed(None);
            return error_response(StatusCode::SERVICE_UNAVAILABLE, "host_clock_fault");
        }
    };
    match context.console.close_session(session_id, capability, now) {
        Ok(stop_queued) => json_response(
            StatusCode::OK,
            &CloseSessionResponse {
                schema_version: 1,
                closed: true,
                stop_queued,
            },
        ),
        Err(
            WheelsOffQualificationSubmitError::UnknownSession(_)
            | WheelsOffQualificationSubmitError::SessionCapabilityMismatch,
        ) => error_response(StatusCode::UNAUTHORIZED, "invalid_session"),
        Err(_) => error_response(StatusCode::CONFLICT, "close_rejected"),
    }
}

fn qualification_intent_response(
    context: &QualificationHttpContext,
    authorized_session: Option<(super::ConsoleSessionId, super::ConsoleSessionCapability)>,
    body: &[u8],
) -> Response<Body> {
    let Some((authorized_session_id, capability)) = authorized_session else {
        return error_response(StatusCode::UNAUTHORIZED, "invalid_session");
    };
    let dto = match parse_exact_json::<WheelsOffQualificationIntentRequestDto>(body) {
        Ok(dto) => dto,
        Err(()) => return error_response(StatusCode::BAD_REQUEST, "invalid_intent_json"),
    };
    let request = match dto.parse(context.profile) {
        Ok(request) => request,
        Err(error) => {
            return qualification_parse_error_response(error);
        }
    };
    if request.session_id != authorized_session_id {
        return error_response(StatusCode::UNAUTHORIZED, "invalid_session");
    }
    let now = match context.clock.try_now() {
        Ok(now) => now,
        Err(_) => {
            context.console.signal_internal_fail_closed(None);
            return error_response(StatusCode::SERVICE_UNAVAILABLE, "host_clock_fault");
        }
    };
    match context.console.submit(request, capability, now) {
        Ok(outcome) => json_response(StatusCode::ACCEPTED, &outcome),
        Err(
            WheelsOffQualificationSubmitError::UnknownSession(_)
            | WheelsOffQualificationSubmitError::SessionCapabilityMismatch,
        ) => error_response(StatusCode::UNAUTHORIZED, "invalid_session"),
        Err(WheelsOffQualificationSubmitError::SoftwareSafetyStopLatched) => {
            error_response(StatusCode::LOCKED, "software_safety_stop_latched")
        }
        Err(WheelsOffQualificationSubmitError::RuntimeReceiverDisconnected) => error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "runtime_receiver_disconnected",
        ),
        Err(WheelsOffQualificationSubmitError::SessionCapacityReached) => {
            error_response(StatusCode::TOO_MANY_REQUESTS, "session_capacity_reached")
        }
        Err(_) => error_response(StatusCode::CONFLICT, "qualification_intent_rejected"),
    }
}

fn qualification_parse_error_response(
    error: WheelsOffQualificationRequestParseError,
) -> Response<Body> {
    let code = match error {
        WheelsOffQualificationRequestParseError::UnsupportedSchema(_) => {
            "unsupported_qualification_schema"
        }
        WheelsOffQualificationRequestParseError::WrongControlProfile => "wrong_control_profile",
        WheelsOffQualificationRequestParseError::NonCanonicalDecimalIdentity
        | WheelsOffQualificationRequestParseError::ZeroIdentity => "invalid_request_identity",
        WheelsOffQualificationRequestParseError::TimerPwm(_) => "invalid_timer_pwm_percent",
    };
    error_response(StatusCode::BAD_REQUEST, code)
}

fn qualification_grid_response(
    path: &str,
    headers: &HeaderMap,
    context: &QualificationHttpContext,
) -> Response<Body> {
    let fields: Vec<_> = path.trim_matches('/').split('/').collect();
    if fields.len() != 7
        || fields[0..3] != ["api", "v1", "maps"]
        || fields[4] != "revisions"
        || fields[6] != "grid"
    {
        return error_response(StatusCode::NOT_FOUND, "not_found");
    }
    let Some(epoch) = fields[3].parse::<u64>().ok().and_then(NonZeroU64::new) else {
        return error_response(StatusCode::BAD_REQUEST, "invalid_map_epoch");
    };
    let Ok(revision) = fields[5].parse::<u64>() else {
        return error_response(StatusCode::BAD_REQUEST, "invalid_map_revision");
    };
    let grid = match context.telemetry.exact_grid(epoch, revision) {
        Ok(Some(grid)) => grid,
        Ok(None) => {
            return error_response(StatusCode::NOT_FOUND, "exact_grid_not_available");
        }
        Err(_) => {
            context.console.signal_internal_fail_closed(None);
            return error_response(StatusCode::SERVICE_UNAVAILABLE, "telemetry_store_fault");
        }
    };
    let etag = format!("\"{}:{}\"", epoch.get(), revision);
    if optional_single_header(headers, "if-none-match") == Ok(Some(etag.as_str())) {
        return secure_response(StatusCode::NOT_MODIFIED, None, Vec::new(), &[]);
    }
    let metadata = grid.metadata;
    let extra_headers = [
        ("etag", etag),
        ("x-kiko-map-epoch", epoch.get().to_string()),
        ("x-kiko-map-revision", revision.to_string()),
        ("x-kiko-grid-width", metadata.width.get().to_string()),
        ("x-kiko-grid-height", metadata.height.get().to_string()),
        (
            "x-kiko-grid-encoding",
            "u8_unknown0_free1_occupied2".to_string(),
        ),
        (
            "x-kiko-grid-row-order",
            "row_major_x_fast_rows_increase_positive_map_y".to_string(),
        ),
        (
            "x-kiko-grid-origin",
            "minimum_xy_corner_of_cell_0_0".to_string(),
        ),
    ];
    secure_response(
        StatusCode::OK,
        Some("application/octet-stream"),
        Bytes::from_owner(GridBytesOwner(grid)),
        &extra_headers,
    )
}

async fn read_bounded_body(mut body: Body) -> Result<Bytes, RequestBodyError> {
    if body
        .size_hint()
        .upper()
        .is_some_and(|upper| upper > MAX_QUALIFICATION_HTTP_REQUEST_BYTES as u64)
    {
        return Err(RequestBodyError::TooLarge);
    }
    let deadline = tokio::time::Instant::now() + QUALIFICATION_HTTP_REQUEST_TIMEOUT;
    let mut output = BytesMut::with_capacity(
        body.size_hint()
            .upper()
            .and_then(|upper| usize::try_from(upper).ok())
            .unwrap_or(0)
            .min(MAX_QUALIFICATION_HTTP_REQUEST_BYTES),
    );
    loop {
        let next = tokio::time::timeout_at(deadline, body.data())
            .await
            .map_err(|_| RequestBodyError::TimedOut)?;
        let Some(chunk) = next else {
            break;
        };
        let chunk = chunk.map_err(|_| RequestBodyError::Transport)?;
        let new_length = output
            .len()
            .checked_add(chunk.len())
            .ok_or(RequestBodyError::TooLarge)?;
        if new_length > MAX_QUALIFICATION_HTTP_REQUEST_BYTES {
            return Err(RequestBodyError::TooLarge);
        }
        output.extend_from_slice(&chunk);
    }
    if output.is_empty() {
        return Err(RequestBodyError::InvalidLength);
    }
    Ok(output.freeze())
}

fn parse_exact_json<T: DeserializeOwned>(body: &[u8]) -> Result<T, ()> {
    if body.is_empty()
        || body.len() > MAX_QUALIFICATION_HTTP_REQUEST_BYTES
        || body.first() != Some(&b'{')
        || body.last() != Some(&b'}')
    {
        return Err(());
    }
    serde_json::from_slice(body).map_err(|_| ())
}

fn single_header<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    let mut values = headers.get_all(name).iter();
    let value = values.next()?.to_str().ok()?;
    if values.next().is_some() {
        return None;
    }
    Some(value)
}

fn optional_single_header<'a>(headers: &'a HeaderMap, name: &str) -> Result<Option<&'a str>, ()> {
    let mut values = headers.get_all(name).iter();
    let Some(value) = values.next() else {
        return Ok(None);
    };
    let value = value.to_str().map_err(|_| ())?;
    if values.next().is_some() {
        return Err(());
    }
    Ok(Some(value))
}

fn content_type_is_exact_json(headers: &HeaderMap) -> bool {
    optional_single_header(headers, "content-type").is_ok_and(|value| {
        value.is_some_and(|value| {
            let value = value.trim();
            value.eq_ignore_ascii_case("application/json")
                || value
                    .split_once(';')
                    .is_some_and(|(media_type, parameter)| {
                        media_type.trim().eq_ignore_ascii_case("application/json")
                            && parameter.trim().eq_ignore_ascii_case("charset=utf-8")
                    })
        })
    })
}

fn access_capability_matches(
    headers: &HeaderMap,
    expected: OperatorConsoleAccessCapability,
) -> bool {
    single_header(headers, "x-kiko-console-capability")
        .and_then(OperatorConsoleAccessCapability::parse_hex)
        .is_some_and(|candidate| expected.constant_time_matches(candidate))
}

fn authenticated_session(
    headers: &HeaderMap,
    console: &WheelsOffQualificationConsoleHandle,
) -> Option<(super::ConsoleSessionId, super::ConsoleSessionCapability)> {
    let session_id = single_header(headers, "x-kiko-session-id").and_then(parse_session_id)?;
    let capability = single_header(headers, "x-kiko-session-capability")
        .and_then(OperatorConsoleAccessCapability::parse_hex)
        .map(|capability| {
            super::ConsoleSessionCapability::from_bytes(*capability.as_bytes_for_session())
        })?;
    console
        .session_capability_matches(session_id, capability)
        .then_some((session_id, capability))
}

fn parse_session_id(raw: &str) -> Option<super::ConsoleSessionId> {
    if raw.is_empty()
        || !raw.bytes().all(|byte| byte.is_ascii_digit())
        || (raw.len() > 1 && raw.starts_with('0'))
    {
        return None;
    }
    raw.parse::<u64>()
        .ok()
        .and_then(|value| super::ConsoleSessionId::parse(value).ok())
}

#[derive(Clone, Copy, Debug)]
struct QualificationHttpServerConfig {
    bind: OperatorConsoleBind,
    access_capability: OperatorConsoleAccessCapability,
    clock: AgentControlMonotonicOrigin,
    deadman_tick: Duration,
}

#[derive(Debug)]
struct WheelsOffQualificationHttpServer {
    bound_address: SocketAddr,
    shutdown: Option<oneshot::Sender<()>>,
    join: Option<thread::JoinHandle<WheelsOffQualificationHttpServerExit>>,
}

impl WheelsOffQualificationHttpServer {
    fn start(
        config: QualificationHttpServerConfig,
        console: WheelsOffQualificationConsoleHandle,
        telemetry: WheelsOffQualificationTelemetryStore,
        profile: WheelsOffQualificationControlProfile,
    ) -> Result<Self, WheelsOffQualificationHttpServerStartError> {
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let thread = thread::Builder::new()
            .name("kiko-wheels-off-qualification-http".to_string())
            .spawn(move || {
                run_qualification_http_server(
                    config,
                    console,
                    telemetry,
                    profile,
                    shutdown_rx,
                    ready_tx,
                )
            })
            .map_err(WheelsOffQualificationHttpServerStartError::Spawn)?;
        let ready = match ready_rx.recv_timeout(Duration::from_secs(5)) {
            Ok(ready) => ready,
            Err(_) => {
                let _ = shutdown_tx.send(());
                let _ = thread.join();
                return Err(WheelsOffQualificationHttpServerStartError::ReadinessLost);
            }
        };
        let bound_address = match ready {
            Ok(address) => address,
            Err(message) => {
                let _ = thread.join();
                return Err(WheelsOffQualificationHttpServerStartError::Bind(message));
            }
        };
        if !bound_address.ip().is_loopback() {
            let _ = shutdown_tx.send(());
            let _ = thread.join();
            return Err(
                WheelsOffQualificationHttpServerStartError::NonLoopbackBound(bound_address),
            );
        }
        Ok(Self {
            bound_address,
            shutdown: Some(shutdown_tx),
            join: Some(thread),
        })
    }

    const fn bound_address(&self) -> SocketAddr {
        self.bound_address
    }

    fn request_shutdown(&mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
    }

    fn try_join(
        &mut self,
    ) -> Result<Option<WheelsOffQualificationHttpServerExit>, WheelsOffQualificationHttpJoinError>
    {
        let join = self
            .join
            .as_ref()
            .ok_or(WheelsOffQualificationHttpJoinError::AlreadyJoined)?;
        if !join.is_finished() {
            return Ok(None);
        }
        self.join
            .take()
            .ok_or(WheelsOffQualificationHttpJoinError::AlreadyJoined)?
            .join()
            .map(Some)
            .map_err(|_| WheelsOffQualificationHttpJoinError::Panicked)
    }

    fn shutdown(
        &mut self,
    ) -> Result<WheelsOffQualificationHttpServerExit, WheelsOffQualificationHttpJoinError> {
        self.request_shutdown();
        let join = self
            .join
            .as_ref()
            .ok_or(WheelsOffQualificationHttpJoinError::AlreadyJoined)?;
        let deadline = Instant::now() + QUALIFICATION_HTTP_SHUTDOWN_TIMEOUT;
        while !join.is_finished() {
            if Instant::now() >= deadline {
                return Err(WheelsOffQualificationHttpJoinError::TimedOut);
            }
            thread::sleep(Duration::from_millis(10));
        }
        self.join
            .take()
            .ok_or(WheelsOffQualificationHttpJoinError::AlreadyJoined)?
            .join()
            .map_err(|_| WheelsOffQualificationHttpJoinError::Panicked)
    }
}

impl Drop for WheelsOffQualificationHttpServer {
    fn drop(&mut self) {
        self.request_shutdown();
    }
}

#[derive(Debug)]
pub enum WheelsOffQualificationHttpServerStartError {
    Spawn(std::io::Error),
    ReadinessLost,
    Bind(String),
    NonLoopbackBound(SocketAddr),
}

impl fmt::Display for WheelsOffQualificationHttpServerStartError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "could not start wheels-off qualification HTTP server: {self:?}"
        )
    }
}

impl std::error::Error for WheelsOffQualificationHttpServerStartError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Spawn(source) => Some(source),
            Self::ReadinessLost | Self::Bind(_) | Self::NonLoopbackBound(_) => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WheelsOffQualificationHttpJoinError {
    AlreadyJoined,
    TimedOut,
    Panicked,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WheelsOffQualificationHttpServerExit {
    pub bound_address: SocketAddr,
    pub graceful_shutdown: bool,
    pub forced_shutdown: bool,
    pub server_error: bool,
    pub deadman_ticks: u64,
    pub deadman_stops_enqueued: u64,
    pub clock_faulted: bool,
    pub frontend_loss_stop_enqueued: bool,
}

fn run_qualification_http_server(
    config: QualificationHttpServerConfig,
    console: WheelsOffQualificationConsoleHandle,
    telemetry: WheelsOffQualificationTelemetryStore,
    profile: WheelsOffQualificationControlProfile,
    shutdown_rx: oneshot::Receiver<()>,
    ready_tx: mpsc::SyncSender<Result<SocketAddr, String>>,
) -> WheelsOffQualificationHttpServerExit {
    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            let _ = ready_tx.send(Err(error.to_string()));
            console.signal_internal_fail_closed(None);
            return failed_server_exit(config.bind.address(), true);
        }
    };
    let builder = {
        let _guard = runtime.enter();
        match Server::try_bind(&config.bind.address()) {
            Ok(builder) => builder,
            Err(error) => {
                let _ = ready_tx.send(Err(error.to_string()));
                return failed_server_exit(config.bind.address(), false);
            }
        }
    };
    let bound_address = builder.local_addr();
    let context = Arc::new(QualificationHttpContext {
        console: console.clone(),
        telemetry,
        profile,
        access_capability: config.access_capability,
        clock: config.clock,
        bound_port: bound_address.port(),
        request_permits: Arc::new(Semaphore::new(MAX_CONCURRENT_QUALIFICATION_HTTP_REQUESTS)),
    });
    let make_service = make_service_fn(move |_| {
        let context = Arc::clone(&context);
        async move {
            Ok::<_, Infallible>(service_fn(move |request| {
                handle_qualification_request(request, Arc::clone(&context))
            }))
        }
    });
    let shutdown_observed = Arc::new(AtomicBool::new(false));
    let shutdown_notify = Arc::new(Notify::new());
    let signal_observed = Arc::clone(&shutdown_observed);
    let signal_notify = Arc::clone(&shutdown_notify);
    let graceful_signal = async move {
        let _ = shutdown_rx.await;
        signal_observed.store(true, Ordering::Release);
        signal_notify.notify_one();
    };
    let server = builder
        .serve(make_service)
        .with_graceful_shutdown(graceful_signal);
    if ready_tx.send(Ok(bound_address)).is_err() {
        console.signal_internal_fail_closed(None);
        return failed_server_exit(bound_address, false);
    }
    let ticks = Arc::new(AtomicU64::new(0));
    let stops = Arc::new(AtomicU64::new(0));
    let clock_faulted = Arc::new(AtomicBool::new(false));
    let deadman_console = console.clone();
    let deadman_ticks = Arc::clone(&ticks);
    let deadman_stops = Arc::clone(&stops);
    let deadman_clock_faulted = Arc::clone(&clock_faulted);
    let deadman_task = runtime.spawn(async move {
        let mut interval = tokio::time::interval(config.deadman_tick);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            interval.tick().await;
            deadman_ticks.fetch_add(1, Ordering::Relaxed);
            match config.clock.try_now() {
                Ok(now) => match deadman_console.tick_deadman(now) {
                    Ok(true) => {
                        deadman_stops.fetch_add(1, Ordering::Relaxed);
                    }
                    Ok(false) => {}
                    Err(_) => deadman_console.signal_internal_fail_closed(Some(now)),
                },
                Err(_) => {
                    deadman_clock_faulted.store(true, Ordering::Release);
                    deadman_console.signal_internal_fail_closed(None);
                }
            }
        }
    });
    let force_notify = Arc::clone(&shutdown_notify);
    let (server_error, forced_shutdown) = runtime.block_on(async {
        tokio::select! {
            result = server => (result.is_err(), false),
            () = async move {
                force_notify.notified().await;
                tokio::time::sleep(QUALIFICATION_HTTP_FORCE_SHUTDOWN_AFTER).await;
            } => (false, true),
        }
    });
    deadman_task.abort();
    let frontend_loss_stop_enqueued = match config.clock.try_now() {
        Ok(now) => console
            .report_frontend_connection_lost(now)
            .unwrap_or_else(|_| {
                console.signal_internal_fail_closed(Some(now));
                false
            }),
        Err(_) => {
            clock_faulted.store(true, Ordering::Release);
            console.signal_internal_fail_closed(None);
            false
        }
    };
    WheelsOffQualificationHttpServerExit {
        bound_address,
        graceful_shutdown: shutdown_observed.load(Ordering::Acquire) && !forced_shutdown,
        forced_shutdown,
        server_error,
        deadman_ticks: ticks.load(Ordering::Relaxed),
        deadman_stops_enqueued: stops.load(Ordering::Relaxed),
        clock_faulted: clock_faulted.load(Ordering::Acquire),
        frontend_loss_stop_enqueued,
    }
}

fn failed_server_exit(
    bound_address: SocketAddr,
    clock_faulted: bool,
) -> WheelsOffQualificationHttpServerExit {
    WheelsOffQualificationHttpServerExit {
        bound_address,
        graceful_shutdown: false,
        forced_shutdown: false,
        server_error: true,
        deadman_ticks: 0,
        deadman_stops_enqueued: 0,
        clock_faulted,
        frontend_loss_stop_enqueued: false,
    }
}

#[must_use = "qualification frontend shutdown evidence must be inspected"]
#[derive(Debug)]
pub struct WheelsOffQualificationFrontend {
    bound_address: SocketAddr,
    http: Option<WheelsOffQualificationHttpServer>,
    capability: Option<OperatorConsolePersistedAccessCapability>,
    terminal_evidence: Option<WheelsOffQualificationFrontendShutdownEvidence>,
}

impl WheelsOffQualificationFrontend {
    pub fn start(
        config: &WheelsOffQualificationFrontendConfig,
        console: WheelsOffQualificationConsoleHandle,
        telemetry: WheelsOffQualificationTelemetryStore,
        profile: WheelsOffQualificationControlProfile,
    ) -> Result<Self, WheelsOffQualificationFrontendStartError> {
        if console.snapshot().control_profile != profile {
            return Err(WheelsOffQualificationFrontendStartError::ProfileMismatch);
        }
        match telemetry.control_profile() {
            Ok(telemetry_profile) if telemetry_profile == profile => {}
            Ok(_) => return Err(WheelsOffQualificationFrontendStartError::ProfileMismatch),
            Err(source) => {
                console.signal_internal_fail_closed(None);
                return Err(WheelsOffQualificationFrontendStartError::Telemetry(source));
            }
        }
        let capability =
            OperatorConsoleAccessCapability::generate_and_persist_new(config.capability_path())
                .map_err(WheelsOffQualificationFrontendStartError::Capability)?;
        let http_config = QualificationHttpServerConfig {
            bind: config.bind,
            access_capability: capability.access_capability(),
            clock: config.clock,
            deadman_tick: config.deadman_tick,
        };
        let http =
            match WheelsOffQualificationHttpServer::start(http_config, console, telemetry, profile)
            {
                Ok(http) => http,
                Err(source) => {
                    return Err(WheelsOffQualificationFrontendStartError::Http {
                        source,
                        capability_cleanup: capability.cleanup(),
                    });
                }
            };
        Ok(Self {
            bound_address: http.bound_address(),
            http: Some(http),
            capability: Some(capability),
            terminal_evidence: None,
        })
    }

    pub const fn bound_address(&self) -> SocketAddr {
        self.bound_address
    }

    pub fn request_shutdown(&mut self) {
        if let Some(http) = self.http.as_mut() {
            http.request_shutdown();
        }
    }

    pub fn poll_unexpected_exit(
        &mut self,
    ) -> Option<WheelsOffQualificationFrontendShutdownEvidence> {
        if let Some(evidence) = self.terminal_evidence {
            return Some(evidence);
        }
        let result = match self.http.as_mut()?.try_join() {
            Ok(None) => return None,
            Ok(Some(exit)) => Ok(exit),
            Err(WheelsOffQualificationHttpJoinError::TimedOut) => return None,
            Err(source) => Err(source),
        };
        self.http.take();
        let capability = QualificationCapabilityShutdownEvidence::Cleaned(
            self.capability
                .take()
                .expect("live qualification HTTP owner retains capability")
                .cleanup(),
        );
        let evidence = WheelsOffQualificationFrontendShutdownEvidence {
            http: result,
            capability,
        };
        self.terminal_evidence = Some(evidence);
        Some(evidence)
    }

    pub fn shutdown(&mut self) -> WheelsOffQualificationFrontendShutdownEvidence {
        if let Some(evidence) = self.terminal_evidence {
            return evidence;
        }
        self.request_shutdown();
        let http = self
            .http
            .as_mut()
            .expect("qualification HTTP owner is consumed exactly once")
            .shutdown();
        let capability = if matches!(http, Err(WheelsOffQualificationHttpJoinError::TimedOut)) {
            QualificationCapabilityShutdownEvidence::RetainedWhileHttpOwnerLive
        } else {
            self.http.take();
            QualificationCapabilityShutdownEvidence::Cleaned(
                self.capability
                    .take()
                    .expect("qualification capability is consumed exactly once")
                    .cleanup(),
            )
        };
        let evidence = WheelsOffQualificationFrontendShutdownEvidence { http, capability };
        if !evidence.retains_live_http_owner() {
            self.terminal_evidence = Some(evidence);
        }
        evidence
    }
}

impl Drop for WheelsOffQualificationFrontend {
    fn drop(&mut self) {
        if self.terminal_evidence.is_some() {
            return;
        }
        self.request_shutdown();
        let stopped = match self
            .http
            .as_mut()
            .map(WheelsOffQualificationHttpServer::shutdown)
        {
            None | Some(Ok(_)) => true,
            Some(Err(WheelsOffQualificationHttpJoinError::TimedOut)) => false,
            Some(Err(
                WheelsOffQualificationHttpJoinError::AlreadyJoined
                | WheelsOffQualificationHttpJoinError::Panicked,
            )) => true,
        };
        if stopped {
            self.http.take();
            if let Some(capability) = self.capability.take() {
                let _ = capability.cleanup();
            }
        } else {
            if let Some(http) = self.http.take() {
                std::mem::forget(http);
            }
            if let Some(capability) = self.capability.take() {
                std::mem::forget(capability);
            }
        }
    }
}

#[derive(Debug)]
pub enum WheelsOffQualificationFrontendStartError {
    ProfileMismatch,
    Telemetry(WheelsOffQualificationTelemetryError),
    Capability(OperatorConsoleCapabilityPersistError),
    Http {
        source: WheelsOffQualificationHttpServerStartError,
        capability_cleanup: OperatorConsoleCapabilityCleanupEvidence,
    },
}

impl fmt::Display for WheelsOffQualificationFrontendStartError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ProfileMismatch => {
                formatter.write_str("qualification frontend profile does not match console")
            }
            Self::Telemetry(source) => source.fmt(formatter),
            Self::Capability(source) => source.fmt(formatter),
            Self::Http {
                source,
                capability_cleanup,
            } => write!(
                formatter,
                "qualification HTTP startup failed: {source}; capability cleanup: {capability_cleanup:?}"
            ),
        }
    }
}

impl std::error::Error for WheelsOffQualificationFrontendStartError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Telemetry(source) => Some(source),
            Self::Capability(source) => Some(source),
            Self::Http { source, .. } => Some(source),
            Self::ProfileMismatch => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QualificationCapabilityShutdownEvidence {
    RetainedWhileHttpOwnerLive,
    Cleaned(OperatorConsoleCapabilityCleanupEvidence),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WheelsOffQualificationFrontendShutdownEvidence {
    http: Result<WheelsOffQualificationHttpServerExit, WheelsOffQualificationHttpJoinError>,
    capability: QualificationCapabilityShutdownEvidence,
}

impl WheelsOffQualificationFrontendShutdownEvidence {
    pub const fn http(
        self,
    ) -> Result<WheelsOffQualificationHttpServerExit, WheelsOffQualificationHttpJoinError> {
        self.http
    }

    pub const fn capability(self) -> QualificationCapabilityShutdownEvidence {
        self.capability
    }

    pub const fn retains_live_http_owner(self) -> bool {
        matches!(
            self.http,
            Err(WheelsOffQualificationHttpJoinError::TimedOut)
        )
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::{Read, Write};
    use std::net::TcpStream;
    use std::os::unix::fs::{MetadataExt, PermissionsExt};
    use std::sync::atomic::{AtomicU64, Ordering};

    use warp::hyper::body::to_bytes;

    use super::*;
    use crate::HostMonotonicTimestamp;

    const TEST_PORT: u16 = 18_321;
    const ACCESS_BYTES: [u8; ACCESS_CAPABILITY_BYTES] = [0x33; ACCESS_CAPABILITY_BYTES];

    fn profile() -> WheelsOffQualificationControlProfile {
        WheelsOffQualificationControlProfile::parse(30, 10, 250).unwrap()
    }

    fn test_context() -> (
        Arc<QualificationHttpContext>,
        super::super::WheelsOffQualificationIngressReceiver,
    ) {
        let profile = profile();
        let (console, receiver) = super::super::wheels_off_qualification_console(profile);
        let telemetry = WheelsOffQualificationTelemetryStore::parse(
            profile,
            OperatorConsoleSnapshot::unknown(ConsoleSnapshotRevision::parse(1).unwrap()),
            console.snapshot(),
        )
        .unwrap();
        (
            Arc::new(QualificationHttpContext {
                console,
                telemetry,
                profile,
                access_capability: OperatorConsoleAccessCapability::parse(ACCESS_BYTES).unwrap(),
                clock: AgentControlMonotonicOrigin::new(
                    Instant::now(),
                    HostMonotonicTimestamp::from_nanos(1_000_000_000),
                ),
                bound_port: TEST_PORT,
                request_permits: Arc::new(Semaphore::new(
                    MAX_CONCURRENT_QUALIFICATION_HTTP_REQUESTS,
                )),
            }),
            receiver,
        )
    }

    fn api_request(method: Method, path: &str, body: Body) -> Request<Body> {
        Request::builder()
            .method(method)
            .uri(path)
            .header("host", format!("127.0.0.1:{TEST_PORT}"))
            .header("origin", format!("http://127.0.0.1:{TEST_PORT}"))
            .header("x-kiko-console-capability", encode_hex(&ACCESS_BYTES))
            .header("content-type", "application/json")
            .body(body)
            .unwrap()
    }

    async fn response_json(response: Response<Body>) -> serde_json::Value {
        serde_json::from_slice(&to_bytes(response.into_body()).await.unwrap()).unwrap()
    }

    async fn open_session(
        context: Arc<QualificationHttpContext>,
        source: ConsoleSourceKind,
    ) -> (String, String) {
        let source = match source {
            ConsoleSourceKind::Operator => "operator",
            ConsoleSourceKind::Agent => "agent",
        };
        let response = handle_qualification_request_inner(
            api_request(
                Method::POST,
                "/api/v1/sessions",
                Body::from(format!("{{\"schema_version\":1,\"source\":\"{source}\"}}")),
            ),
            context,
        )
        .await;
        assert_eq!(response.status(), StatusCode::CREATED);
        let body = response_json(response).await;
        (
            body["session_id"].as_str().unwrap().to_owned(),
            body["session_capability"].as_str().unwrap().to_owned(),
        )
    }

    fn authenticated_intent_request(
        session_id: &str,
        session_capability: &str,
        sequence: u64,
        intent: &str,
    ) -> Request<Body> {
        let body = format!(
            "{{\"schema_version\":1,\"control_profile\":\"wheels_off_raw_timer_pwm_qualification\",\"session_id\":\"{session_id}\",\"source_sequence\":\"{sequence}\",\"idempotency_key\":\"{sequence}\",\"intent\":{intent}}}"
        );
        let mut request = api_request(
            Method::POST,
            super::super::WHEELS_OFF_QUALIFICATION_INTENT_ENDPOINT,
            Body::from(body),
        );
        request
            .headers_mut()
            .insert("x-kiko-session-id", session_id.parse().unwrap());
        request.headers_mut().insert(
            "x-kiko-session-capability",
            session_capability.parse().unwrap(),
        );
        request
    }

    #[tokio::test(flavor = "current_thread")]
    async fn distinct_endpoint_parses_once_and_emits_exact_raw_pwm() {
        let (context, mut receiver) = test_context();

        let static_response = handle_qualification_request_inner(
            Request::builder()
                .method(Method::GET)
                .uri("/")
                .header("host", format!("127.0.0.1:{TEST_PORT}"))
                .body(Body::empty())
                .unwrap(),
            Arc::clone(&context),
        )
        .await;
        assert_eq!(static_response.status(), StatusCode::OK);

        let unauthorized = handle_qualification_request_inner(
            Request::builder()
                .method(Method::GET)
                .uri("/api/v1/snapshot")
                .header("host", format!("127.0.0.1:{TEST_PORT}"))
                .body(Body::empty())
                .unwrap(),
            Arc::clone(&context),
        )
        .await;
        assert_eq!(unauthorized.status(), StatusCode::UNAUTHORIZED);

        let production_endpoint = handle_qualification_request_inner(
            api_request(Method::POST, "/api/v1/intents", Body::from("{}")),
            Arc::clone(&context),
        )
        .await;
        assert_eq!(production_endpoint.status(), StatusCode::NOT_FOUND);

        let (session_id, session_capability) =
            open_session(Arc::clone(&context), ConsoleSourceKind::Operator).await;
        let begin = handle_qualification_request_inner(
            authenticated_intent_request(
                &session_id,
                &session_capability,
                1,
                "{\"kind\":\"begin_manual\"}",
            ),
            Arc::clone(&context),
        )
        .await;
        assert_eq!(begin.status(), StatusCode::ACCEPTED);

        let pwm = handle_qualification_request_inner(
            authenticated_intent_request(
                &session_id,
                &session_capability,
                2,
                "{\"kind\":\"manual_pwm\",\"left_timer_pwm_percent\":-10,\"right_timer_pwm_percent\":10}",
            ),
            Arc::clone(&context),
        )
        .await;
        assert_eq!(pwm.status(), StatusCode::ACCEPTED);
        let event = match receiver.try_recv().unwrap() {
            super::super::WheelsOffQualificationIngressEvent::CandidatePwm(event) => event,
            super::super::WheelsOffQualificationIngressEvent::TerminalStop(_) => {
                panic!("raw candidate expected")
            }
        };
        assert_eq!(event.requested_pwm().left_timer_pwm_percent.get(), -10);
        assert_eq!(event.requested_pwm().right_timer_pwm_percent.get(), 10);

        let release = handle_qualification_request_inner(
            authenticated_intent_request(
                &session_id,
                &session_capability,
                3,
                "{\"kind\":\"release_manual\"}",
            ),
            Arc::clone(&context),
        )
        .await;
        assert_eq!(release.status(), StatusCode::ACCEPTED);
        assert!(matches!(
            receiver.try_recv(),
            Ok(super::super::WheelsOffQualificationIngressEvent::TerminalStop(_))
        ));

        let snapshot = handle_qualification_request_inner(
            api_request(Method::GET, "/api/v1/snapshot", Body::empty()),
            context,
        )
        .await;
        assert_eq!(snapshot.status(), StatusCode::OK);
        let snapshot = response_json(snapshot).await;
        assert_eq!(
            snapshot["control_profile"]["command_units"],
            "signed_timer_duty_percent"
        );
        assert_eq!(
            snapshot["wheels_off_qualification"]["control_profile"]["required_wheel_state"],
            "removed"
        );
        assert!(snapshot["requested_owner"].is_null());
        assert!(snapshot["actual_authority"].is_null());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn adversarial_headers_content_types_and_bodies_are_rejected() {
        let (context, _receiver) = test_context();

        let wrong_host = handle_qualification_request_inner(
            Request::builder()
                .method(Method::GET)
                .uri("/api/v1/health")
                .header("host", "127.0.0.1:9")
                .header("x-kiko-console-capability", encode_hex(&ACCESS_BYTES))
                .body(Body::empty())
                .unwrap(),
            Arc::clone(&context),
        )
        .await;
        assert_eq!(wrong_host.status(), StatusCode::BAD_REQUEST);

        let mut duplicate_capability = api_request(Method::GET, "/api/v1/health", Body::empty());
        duplicate_capability.headers_mut().append(
            "x-kiko-console-capability",
            encode_hex(&ACCESS_BYTES).parse().unwrap(),
        );
        let duplicate_capability =
            handle_qualification_request_inner(duplicate_capability, Arc::clone(&context)).await;
        assert_eq!(duplicate_capability.status(), StatusCode::UNAUTHORIZED);

        let mut missing_origin = api_request(Method::POST, "/api/v1/sessions", Body::from("{}"));
        missing_origin.headers_mut().remove("origin");
        let missing_origin =
            handle_qualification_request_inner(missing_origin, Arc::clone(&context)).await;
        assert_eq!(missing_origin.status(), StatusCode::FORBIDDEN);

        let query = handle_qualification_request_inner(
            api_request(Method::GET, "/api/v1/health?cap=secret", Body::empty()),
            Arc::clone(&context),
        )
        .await;
        assert_eq!(query.status(), StatusCode::BAD_REQUEST);

        let unsupported_method = handle_qualification_request_inner(
            api_request(Method::PUT, "/api/v1/health", Body::empty()),
            Arc::clone(&context),
        )
        .await;
        assert_eq!(unsupported_method.status(), StatusCode::METHOD_NOT_ALLOWED);

        let mut invented_parameter = api_request(
            Method::POST,
            "/api/v1/sessions",
            Body::from("{\"schema_version\":1,\"source\":\"operator\"}"),
        );
        invented_parameter.headers_mut().insert(
            "content-type",
            "application/json; profile=unsafe".parse().unwrap(),
        );
        let invented_parameter =
            handle_qualification_request_inner(invented_parameter, Arc::clone(&context)).await;
        assert_eq!(
            invented_parameter.status(),
            StatusCode::UNSUPPORTED_MEDIA_TYPE
        );

        let whitespace_wrapped = handle_qualification_request_inner(
            api_request(
                Method::POST,
                "/api/v1/sessions",
                Body::from(" {\"schema_version\":1,\"source\":\"operator\"}"),
            ),
            Arc::clone(&context),
        )
        .await;
        assert_eq!(whitespace_wrapped.status(), StatusCode::BAD_REQUEST);

        let oversized = handle_qualification_request_inner(
            api_request(
                Method::POST,
                "/api/v1/sessions",
                Body::from(vec![b'x'; MAX_QUALIFICATION_HTTP_REQUEST_BYTES + 1]),
            ),
            context,
        )
        .await;
        assert_eq!(oversized.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn absolute_body_timeout_and_concurrency_limit_are_enforced() {
        let (context, _receiver) = test_context();
        let (_sender, stalled_body) = Body::channel();
        let stalled = handle_qualification_request_inner(
            api_request(Method::POST, "/api/v1/sessions", stalled_body),
            Arc::clone(&context),
        )
        .await;
        assert_eq!(stalled.status(), StatusCode::REQUEST_TIMEOUT);

        let permits = Arc::clone(&context.request_permits)
            .acquire_many_owned(u32::try_from(MAX_CONCURRENT_QUALIFICATION_HTTP_REQUESTS).unwrap())
            .await
            .unwrap();
        let overloaded = handle_qualification_request_inner(
            api_request(Method::GET, "/api/v1/health", Body::empty()),
            context,
        )
        .await;
        assert_eq!(overloaded.status(), StatusCode::TOO_MANY_REQUESTS);
        drop(permits);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn exact_grid_and_poisoned_telemetry_fail_closed_deterministically() {
        let (context, _receiver) = test_context();
        let mut wrong_schema = context.console.snapshot();
        wrong_schema.schema_version = 99;
        assert!(matches!(
            WheelsOffQualificationTelemetryStore::parse(
                profile(),
                OperatorConsoleSnapshot::unknown(ConsoleSnapshotRevision::parse(1).unwrap()),
                wrong_schema,
            ),
            Err(WheelsOffQualificationTelemetryError::UnsupportedQualificationSchema(99))
        ));
        let grid =
            ConsoleOccupancyGrid::parse(7, 9, 2, 2, 0.05, -1.0, 2.0, vec![0, 1, 2, 2]).unwrap();
        assert!(context.telemetry.publish_grid(grid.clone()).unwrap());
        assert!(!context.telemetry.publish_grid(grid).unwrap());
        let conflicting_grid =
            ConsoleOccupancyGrid::parse(7, 9, 2, 2, 0.05, -1.0, 2.0, vec![2, 1, 0, 2]).unwrap();
        assert!(matches!(
            context.telemetry.publish_grid(conflicting_grid),
            Err(WheelsOffQualificationTelemetryError::GridIdentityConflict {
                map_epoch_id,
                revision: 9,
            }) if map_epoch_id.get() == 7
        ));

        let grid_response = handle_qualification_request_inner(
            api_request(
                Method::GET,
                "/api/v1/maps/7/revisions/9/grid",
                Body::empty(),
            ),
            Arc::clone(&context),
        )
        .await;
        assert_eq!(grid_response.status(), StatusCode::OK);
        assert_eq!(grid_response.headers()["etag"], "\"7:9\"");
        assert_eq!(
            to_bytes(grid_response.into_body()).await.unwrap().as_ref(),
            &[0, 1, 2, 2]
        );

        let mut unsafe_base =
            OperatorConsoleSnapshot::unknown(ConsoleSnapshotRevision::parse(2).unwrap());
        unsafe_base.software_safety_stop_latched = true;
        unsafe_base.software_safety_signal_state = ConsoleSafetySignalState::CompletedFaultLatched;
        assert!(matches!(
            context.telemetry.publish_observational_base(unsafe_base),
            Err(WheelsOffQualificationTelemetryError::ProductionSafetyLatchPresent)
        ));

        let poison_store = context.telemetry.clone();
        assert!(
            thread::spawn(move || {
                let _guard = poison_store.state.lock().unwrap();
                panic!("deliberately poison qualification telemetry");
            })
            .join()
            .is_err()
        );
        assert!(matches!(
            context.telemetry.exact_grid(NonZeroU64::new(7).unwrap(), 9),
            Err(WheelsOffQualificationTelemetryError::Poisoned)
        ));
        let poisoned_snapshot = handle_qualification_request_inner(
            api_request(Method::GET, "/api/v1/snapshot", Body::empty()),
            Arc::clone(&context),
        )
        .await;
        assert_eq!(poisoned_snapshot.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(context.console.snapshot().software_safety_stop_latched);
    }

    static NEXT_TEST_DIRECTORY_ID: AtomicU64 = AtomicU64::new(1);

    struct PrivateTestDirectory(PathBuf);

    impl PrivateTestDirectory {
        fn create() -> Self {
            let id = NEXT_TEST_DIRECTORY_ID.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "kiko-qualification-http-{}-{id}",
                std::process::id()
            ));
            fs::create_dir(&path).unwrap();
            fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).unwrap();
            Self(path)
        }
    }

    impl Drop for PrivateTestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn raw_http_request(address: SocketAddr, request: &str) -> String {
        let mut stream = TcpStream::connect(address).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        stream
            .set_write_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        stream.write_all(request.as_bytes()).unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        response
    }

    #[test]
    fn real_loopback_frontend_publishes_and_cleans_capability_with_shutdown_evidence() {
        let directory = PrivateTestDirectory::create();
        let capability_path = directory.0.join("qualification.cap");
        let profile = profile();
        let (console, _receiver) = super::super::wheels_off_qualification_console(profile);
        let telemetry = WheelsOffQualificationTelemetryStore::parse(
            profile,
            OperatorConsoleSnapshot::unknown(ConsoleSnapshotRevision::parse(1).unwrap()),
            console.snapshot(),
        )
        .unwrap();
        let config = WheelsOffQualificationFrontendConfig::parse(
            "127.0.0.1:0".parse().unwrap(),
            capability_path.clone(),
            AgentControlMonotonicOrigin::new(
                Instant::now(),
                HostMonotonicTimestamp::from_nanos(1_000_000_000),
            ),
            Duration::from_millis(20),
        )
        .unwrap();
        let mut frontend =
            WheelsOffQualificationFrontend::start(&config, console, telemetry, profile).unwrap();
        assert!(frontend.bound_address().ip().is_loopback());
        assert_eq!(
            fs::metadata(&capability_path).unwrap().mode() & 0o7777,
            0o600
        );
        let capability = fs::read_to_string(&capability_path).unwrap();
        let address = frontend.bound_address();
        let host = format!("127.0.0.1:{}", address.port());
        let response = raw_http_request(
            address,
            &format!(
                "GET /api/v1/health HTTP/1.1\r\nHost: {host}\r\nX-Kiko-Console-Capability: {capability}\r\nConnection: close\r\n\r\n"
            ),
        );
        assert!(response.starts_with("HTTP/1.1 200"), "{response}");
        assert!(response.contains("\"http_ready\":true"));

        let evidence = frontend.shutdown();
        let exit = evidence.http().unwrap();
        assert!(exit.graceful_shutdown);
        assert!(!exit.forced_shutdown);
        assert!(!exit.server_error);
        assert!(matches!(
            evidence.capability(),
            QualificationCapabilityShutdownEvidence::Cleaned(
                OperatorConsoleCapabilityCleanupEvidence::ExactEntryRemovedAndParentSynced
            )
        ));
        assert!(!capability_path.exists());
    }
}
