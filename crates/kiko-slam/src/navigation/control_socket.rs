//! Local Unix-domain transport for the bounded agent-control protocol.
//!
//! One [`AgentControlSocketServer`] owns the listener and one process-lifetime
//! request parser. It handles one connection at a time and one request per
//! connection; it creates no client threads and never allocates a
//! caller-sized buffer. The surrounding runtime may place this sequential
//! owner on one dedicated thread.
//!
//! Filesystem permissions are a local access boundary, not authentication.
//! The socket must live in an existing canonical directory owned by the
//! effective user with mode `0700`; the socket inode is changed to `0600`
//! before `accept` is exposed. Startup rejects every existing destination and
//! never unlinks it.
//!
//! Cleanup compares socket type, device, and inode before unlinking. This
//! preserves replacements made before the check. POSIX has no atomic
//! inode-conditioned unlink, so the final check-to-unlink interval assumes the
//! verified private directory is modified only by this process or cooperating
//! same-UID code. This is not protection against a hostile same-UID process.

use std::fmt;
use std::fs;
use std::io::{self, Read, Write};
#[cfg(feature = "operator-console")]
use std::num::NonZeroU64;
use std::num::NonZeroUsize;
use std::os::fd::AsRawFd;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{FileTypeExt, MetadataExt, PermissionsExt};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Component, Path, PathBuf};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender, TryRecvError, TrySendError};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use super::{
    AgentControlCommandV1, AgentControlCompletionV1, AgentControlRejectionCodeV1,
    AgentControlRequestId, AgentControlRequestParseError, AgentControlRequestParser,
    AgentControlRequestV1, AgentControlResponseV1, AgentControlStatusV1,
    MAX_AGENT_CONTROL_REQUEST_JSON_BYTES,
};
use crate::HostMonotonicTimestamp;

/// Conservative pathname limit valid for pathname Unix sockets on Linux and
/// macOS, including the required terminating NUL in the platform structure.
pub const MAX_AGENT_CONTROL_SOCKET_PATH_BYTES: usize = 103;

/// Fixed upper bound for the version-1 response JSON payload.
pub const MAX_AGENT_CONTROL_RESPONSE_JSON_BYTES: usize = 1_024;

/// Maximum number of parsed requests waiting for the runtime.
pub const MAX_AGENT_CONTROL_RUNTIME_QUEUE_CAPACITY: usize = 64;

const FRAME_LENGTH_BYTES: usize = 4;
const MIN_SOCKET_TIMEOUT: Duration = Duration::from_millis(1);
const MAX_SOCKET_TIMEOUT: Duration = Duration::from_secs(30);
const MIN_TERMINAL_SOCKET_TIMEOUT: Duration = Duration::from_secs(4);
const MAX_TERMINAL_SOCKET_TIMEOUT: Duration = Duration::from_secs(10 * 60);
const AGENT_CONTROL_SOCKET_THREAD_NAME: &str = "kiko-agent-control";
const AGENT_CONTROL_SOCKET_STARTUP_TIMEOUT: Duration = Duration::from_secs(5);
const AGENT_CONTROL_SOCKET_CANCELLATION_POLL: Duration = Duration::from_millis(5);
const AGENT_CONTROL_RUNTIME_PRIORITY_POLL: Duration = Duration::from_millis(1);

/// Exact absolute pathname for one local control socket.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AgentControlSocketPath {
    path: PathBuf,
    parent: PathBuf,
}

impl AgentControlSocketPath {
    /// Parse a lexical absolute path without canonicalizing or substituting it.
    ///
    /// Every component after the root must be normal: `.`, `..`, prefixes,
    /// repeated-root forms, and a missing filename are rejected. Parent
    /// existence, ownership, canonical identity, and mode are checked at bind.
    pub fn parse(path: &Path) -> Result<Self, AgentControlSocketPathError> {
        let bytes = path.as_os_str().as_bytes();
        if bytes.is_empty() {
            return Err(AgentControlSocketPathError::Empty);
        }
        if bytes.len() > MAX_AGENT_CONTROL_SOCKET_PATH_BYTES {
            return Err(AgentControlSocketPathError::TooLong {
                actual_bytes: bytes.len(),
                maximum_bytes: MAX_AGENT_CONTROL_SOCKET_PATH_BYTES,
            });
        }
        if bytes.contains(&0) {
            return Err(AgentControlSocketPathError::InteriorNul);
        }
        if !path.is_absolute() {
            return Err(AgentControlSocketPathError::NotAbsolute);
        }
        let normalized = path.components().collect::<PathBuf>();
        if normalized.as_os_str().as_bytes() != bytes {
            return Err(AgentControlSocketPathError::NonCanonicalComponent);
        }

        let mut components = path.components();
        if !matches!(components.next(), Some(Component::RootDir)) {
            return Err(AgentControlSocketPathError::NotAbsolute);
        }
        let mut normal_components = 0_usize;
        for component in components {
            match component {
                Component::Normal(_) => normal_components += 1,
                Component::RootDir
                | Component::CurDir
                | Component::ParentDir
                | Component::Prefix(_) => {
                    return Err(AgentControlSocketPathError::NonCanonicalComponent);
                }
            }
        }
        if normal_components == 0 || path.file_name().is_none() {
            return Err(AgentControlSocketPathError::MissingFileName);
        }
        let parent = path
            .parent()
            .ok_or(AgentControlSocketPathError::MissingFileName)?;
        Ok(Self {
            path: path.to_path_buf(),
            parent: parent.to_path_buf(),
        })
    }

    /// Exact pathname supplied at parsing.
    pub fn as_path(&self) -> &Path {
        &self.path
    }

    fn parent(&self) -> &Path {
        &self.parent
    }
}

/// Socket pathname parse failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AgentControlSocketPathError {
    Empty,
    NotAbsolute,
    MissingFileName,
    NonCanonicalComponent,
    InteriorNul,
    TooLong {
        actual_bytes: usize,
        maximum_bytes: usize,
    },
}

impl fmt::Display for AgentControlSocketPathError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => formatter.write_str("agent-control socket path is empty"),
            Self::NotAbsolute => formatter.write_str("agent-control socket path is not absolute"),
            Self::MissingFileName => {
                formatter.write_str("agent-control socket path has no socket filename")
            }
            Self::NonCanonicalComponent => formatter.write_str(
                "agent-control socket path must contain only one root and normal components",
            ),
            Self::InteriorNul => {
                formatter.write_str("agent-control socket path contains a NUL byte")
            }
            Self::TooLong {
                actual_bytes,
                maximum_bytes,
            } => write!(
                formatter,
                "agent-control socket path is {actual_bytes} bytes; portable maximum is {maximum_bytes} bytes"
            ),
        }
    }
}

impl std::error::Error for AgentControlSocketPathError {}

/// Bounded blocking deadlines for one accepted connection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AgentControlSocketTimeouts {
    read: Duration,
    write: Duration,
    runtime_response: Duration,
    terminal_response: Duration,
}

impl AgentControlSocketTimeouts {
    /// Parse the ordinary bounded deadlines plus the distinct terminal
    /// `save_map` deadline. Terminal persistence may drain capture, finalize
    /// and synchronize a dataset, and hash the selected replay inputs, so it
    /// must not inherit the short ordinary command budget.
    pub fn try_new(
        read: Duration,
        write: Duration,
        runtime_response: Duration,
        terminal_response: Duration,
    ) -> Result<Self, AgentControlSocketTimeoutError> {
        validate_timeout(AgentControlTimeoutKind::Read, read)?;
        validate_timeout(AgentControlTimeoutKind::Write, write)?;
        validate_timeout(AgentControlTimeoutKind::RuntimeResponse, runtime_response)?;
        validate_terminal_timeout(terminal_response)?;
        Ok(Self {
            read,
            write,
            runtime_response,
            terminal_response,
        })
    }

    pub const fn read(self) -> Duration {
        self.read
    }

    pub const fn write(self) -> Duration {
        self.write
    }

    pub const fn runtime_response(self) -> Duration {
        self.runtime_response
    }

    pub const fn terminal_response(self) -> Duration {
        self.terminal_response
    }

    const fn response_timeout_for(self, command: AgentControlCommandV1) -> Duration {
        if matches!(command, AgentControlCommandV1::SaveMap) {
            self.terminal_response
        } else {
            self.runtime_response
        }
    }
}

fn validate_timeout(
    kind: AgentControlTimeoutKind,
    value: Duration,
) -> Result<(), AgentControlSocketTimeoutError> {
    if !(MIN_SOCKET_TIMEOUT..=MAX_SOCKET_TIMEOUT).contains(&value) {
        return Err(AgentControlSocketTimeoutError {
            kind,
            actual: value,
            minimum: MIN_SOCKET_TIMEOUT,
            maximum: MAX_SOCKET_TIMEOUT,
        });
    }
    Ok(())
}

fn validate_terminal_timeout(value: Duration) -> Result<(), AgentControlSocketTimeoutError> {
    if !(MIN_TERMINAL_SOCKET_TIMEOUT..=MAX_TERMINAL_SOCKET_TIMEOUT).contains(&value) {
        return Err(AgentControlSocketTimeoutError {
            kind: AgentControlTimeoutKind::TerminalResponse,
            actual: value,
            minimum: MIN_TERMINAL_SOCKET_TIMEOUT,
            maximum: MAX_TERMINAL_SOCKET_TIMEOUT,
        });
    }
    Ok(())
}

/// Deadline named by a timeout configuration error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AgentControlTimeoutKind {
    Read,
    Write,
    RuntimeResponse,
    TerminalResponse,
}

/// Invalid connection deadline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AgentControlSocketTimeoutError {
    pub kind: AgentControlTimeoutKind,
    pub actual: Duration,
    pub minimum: Duration,
    pub maximum: Duration,
}

impl fmt::Display for AgentControlSocketTimeoutError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "agent-control {:?} timeout {:?} is outside {:?}..={:?}",
            self.kind, self.actual, self.minimum, self.maximum
        )
    }
}

impl std::error::Error for AgentControlSocketTimeoutError {}

/// Fully parsed server configuration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AgentControlSocketConfig {
    path: AgentControlSocketPath,
    timeouts: AgentControlSocketTimeouts,
}

impl AgentControlSocketConfig {
    pub const fn new(path: AgentControlSocketPath, timeouts: AgentControlSocketTimeouts) -> Self {
        Self { path, timeouts }
    }

    pub const fn path(&self) -> &AgentControlSocketPath {
        &self.path
    }

    pub const fn timeouts(&self) -> AgentControlSocketTimeouts {
        self.timeouts
    }
}

/// Shared mapping between `Instant` and the process host-monotonic timebase.
///
/// Copying this value preserves the exact origin; callers should create it
/// once alongside the other host clock adapters and inject copies.
#[derive(Clone, Copy, Debug)]
pub struct AgentControlMonotonicOrigin {
    instant: Instant,
    host: HostMonotonicTimestamp,
}

impl AgentControlMonotonicOrigin {
    pub const fn new(instant: Instant, host: HostMonotonicTimestamp) -> Self {
        Self { instant, host }
    }

    /// Stamp the current instant into the shared host-monotonic epoch.
    pub fn try_now(self) -> Result<HostMonotonicTimestamp, AgentControlClockError> {
        let now = Instant::now();
        let elapsed = now.checked_duration_since(self.instant).ok_or(
            AgentControlClockError::OriginInFuture {
                origin: self.instant,
                observed: now,
            },
        )?;
        let elapsed_ns = u64::try_from(elapsed.as_nanos()).map_err(|_| {
            AgentControlClockError::ElapsedOutOfRange {
                elapsed_ns: elapsed.as_nanos(),
            }
        })?;
        let timestamp_ns = self.host.as_nanos().checked_add(elapsed_ns).ok_or(
            AgentControlClockError::TimestampOverflow {
                origin_ns: self.host.as_nanos(),
                elapsed_ns,
            },
        )?;
        Ok(HostMonotonicTimestamp::from_nanos(timestamp_ns))
    }
}

/// Failure to project `Instant::now()` into the injected host timebase.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AgentControlClockError {
    OriginInFuture { origin: Instant, observed: Instant },
    ElapsedOutOfRange { elapsed_ns: u128 },
    TimestampOverflow { origin_ns: u64, elapsed_ns: u64 },
}

impl fmt::Display for AgentControlClockError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "agent-control host clock failed: {self:?}")
    }
}

impl std::error::Error for AgentControlClockError {}

/// Validated capacity of the runtime handoff queue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AgentControlRuntimeQueueCapacity(NonZeroUsize);

impl AgentControlRuntimeQueueCapacity {
    pub fn try_new(raw: usize) -> Result<Self, AgentControlRuntimeQueueCapacityError> {
        let value = NonZeroUsize::new(raw).ok_or(AgentControlRuntimeQueueCapacityError::Zero)?;
        if value.get() > MAX_AGENT_CONTROL_RUNTIME_QUEUE_CAPACITY {
            return Err(AgentControlRuntimeQueueCapacityError::TooLarge {
                actual: value.get(),
                maximum: MAX_AGENT_CONTROL_RUNTIME_QUEUE_CAPACITY,
            });
        }
        Ok(Self(value))
    }

    pub const fn get(self) -> usize {
        self.0.get()
    }
}

/// Invalid runtime queue capacity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AgentControlRuntimeQueueCapacityError {
    Zero,
    TooLarge { actual: usize, maximum: usize },
}

impl fmt::Display for AgentControlRuntimeQueueCapacityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid agent-control runtime queue: {self:?}")
    }
}

impl std::error::Error for AgentControlRuntimeQueueCapacityError {}

/// Sole server-side handle for a bounded runtime queue.
///
/// This type is intentionally not `Clone`, preventing accidental creation of
/// multiple socket producers through the public API.
#[derive(Debug)]
pub struct AgentControlRuntimeSender {
    inner: SyncSender<AgentControlDispatch>,
    priority: Option<SyncSender<AgentControlDispatch>>,
}

/// Runtime-side receiver for parsed requests.
#[derive(Debug)]
pub struct AgentControlRuntimeReceiver {
    inner: Receiver<AgentControlDispatch>,
    priority: Option<Receiver<AgentControlDispatch>>,
}

/// Opaque, non-clone capability for the unified console's already-typed
/// in-process lane.
///
/// The only public constructor is
/// [`AgentControlSocketTask::bind_and_spawn_with_typed_ingress`], which creates
/// this capability beside the sole socket owner and runtime receiver. Its
/// submission and correlation operations remain navigation-private inside the
/// production console adapter.
#[cfg(all(
    feature = "agent-runtime",
    feature = "actuation",
    feature = "operator-console"
))]
#[derive(Debug)]
pub struct AgentControlTypedIngress {
    normal: SyncSender<AgentControlDispatch>,
    priority: SyncSender<AgentControlDispatch>,
    next_key: Option<NonZeroU64>,
}

/// Process-local identity for one typed in-process submission.
///
/// Socket request IDs are caller-controlled and may numerically collide with
/// console request IDs. This key comes only from the sole non-clone typed
/// ingress and therefore remains a distinct correlation namespace through
/// claim, physical application, and response completion.
#[cfg(feature = "operator-console")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct AgentControlTypedRequestKey(NonZeroU64);

#[cfg(feature = "operator-console")]
impl AgentControlTypedRequestKey {
    #[cfg(test)]
    pub(crate) fn for_test(raw: u64) -> Self {
        Self(NonZeroU64::new(raw).expect("typed request test key must be nonzero"))
    }
}

/// Completion rendezvous for one already-typed in-process submission.
///
/// The claim and response channels are capacity one, so the single-threaded
/// live owner never blocks waiting for the submitting adapter to run. The
/// adapter must still observe the exact response before completing its own
/// request token.
#[cfg(all(
    feature = "agent-runtime",
    feature = "actuation",
    feature = "operator-console"
))]
#[derive(Debug)]
pub(crate) struct AgentControlTypedSubmission {
    request_id: AgentControlRequestId,
    typed_request_key: AgentControlTypedRequestKey,
    claim: Receiver<()>,
    response: Receiver<AgentControlResponseV1>,
}

#[cfg(all(
    feature = "agent-runtime",
    feature = "actuation",
    feature = "operator-console"
))]
impl AgentControlTypedSubmission {
    pub const fn request_id(&self) -> AgentControlRequestId {
        self.request_id
    }

    pub const fn typed_request_key(&self) -> AgentControlTypedRequestKey {
        self.typed_request_key
    }

    pub fn try_take_claim(&self) -> Result<(), AgentControlTypedSubmissionPollError> {
        self.claim.try_recv().map_err(|source| match source {
            TryRecvError::Empty => AgentControlTypedSubmissionPollError::Pending,
            TryRecvError::Disconnected => AgentControlTypedSubmissionPollError::ClaimDisconnected,
        })
    }

    pub fn try_take_response(
        &self,
    ) -> Result<AgentControlResponseV1, AgentControlTypedSubmissionPollError> {
        self.response.try_recv().map_err(|source| match source {
            TryRecvError::Empty => AgentControlTypedSubmissionPollError::Pending,
            TryRecvError::Disconnected => {
                AgentControlTypedSubmissionPollError::ResponseDisconnected
            }
        })
    }
}

#[cfg(all(
    feature = "agent-runtime",
    feature = "actuation",
    feature = "operator-console"
))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AgentControlTypedSubmissionPollError {
    Pending,
    ClaimDisconnected,
    ResponseDisconnected,
}

#[cfg(all(
    feature = "agent-runtime",
    feature = "actuation",
    feature = "operator-console"
))]
#[derive(Debug)]
pub(crate) enum AgentControlTypedSubmitError {
    CorrelationExhausted,
    QueueFull {
        submission: AgentControlTypedSubmission,
    },
    RuntimeDisconnected {
        submission: AgentControlTypedSubmission,
    },
}

#[cfg(all(
    feature = "agent-runtime",
    feature = "actuation",
    feature = "operator-console"
))]
impl AgentControlTypedIngress {
    fn allocate_key(
        &mut self,
    ) -> Result<AgentControlTypedRequestKey, AgentControlTypedSubmitError> {
        let raw = self
            .next_key
            .take()
            .ok_or(AgentControlTypedSubmitError::CorrelationExhausted)?;
        self.next_key = raw.get().checked_add(1).and_then(NonZeroU64::new);
        Ok(AgentControlTypedRequestKey(raw))
    }

    /// Reserve one process-local identity for a direct safety operation that
    /// bypasses every bounded dispatch queue.
    pub(crate) fn reserve_direct_safety_key(
        &mut self,
    ) -> Result<AgentControlTypedRequestKey, AgentControlTypedSubmitError> {
        self.allocate_key()
    }

    /// Submit one domain request that was already parsed by a trusted
    /// in-process boundary. No JSON serialization or second parser is used.
    pub(crate) fn try_submit(
        &mut self,
        request: AgentControlRequestV1,
        received_at: HostMonotonicTimestamp,
    ) -> Result<AgentControlTypedSubmission, AgentControlTypedSubmitError> {
        let request_id = request.request_id();
        let typed_request_key = self.allocate_key()?;
        let (claim_sender, claim_receiver) = mpsc::sync_channel(1);
        let (response_sender, response_receiver) = mpsc::sync_channel(1);
        let (_wire_delivery_sender, wire_delivery_receiver) = mpsc::sync_channel(0);
        let dispatch = AgentControlDispatch {
            request,
            received_at,
            typed_request_key: Some(typed_request_key),
            terminal_response_deadline: None,
            claim: claim_sender,
            response: response_sender,
            wire_delivery: wire_delivery_receiver,
        };
        let submission = AgentControlTypedSubmission {
            request_id,
            typed_request_key,
            claim: claim_receiver,
            response: response_receiver,
        };
        let sender = if request_is_priority(request.command()) {
            &self.priority
        } else {
            &self.normal
        };
        match sender.try_send(dispatch) {
            Ok(()) => Ok(submission),
            Err(TrySendError::Full(_)) => {
                Err(AgentControlTypedSubmitError::QueueFull { submission })
            }
            Err(TrySendError::Disconnected(_)) => {
                Err(AgentControlTypedSubmitError::RuntimeDisconnected { submission })
            }
        }
    }
}

/// Construct the bounded handoff queue used by exactly one socket server.
pub fn agent_control_runtime_queue(
    capacity: AgentControlRuntimeQueueCapacity,
) -> (AgentControlRuntimeSender, AgentControlRuntimeReceiver) {
    let (sender, receiver) = mpsc::sync_channel(capacity.get());
    (
        AgentControlRuntimeSender {
            inner: sender,
            priority: None,
        },
        AgentControlRuntimeReceiver {
            inner: receiver,
            priority: None,
        },
    )
}

#[cfg(all(
    test,
    feature = "agent-runtime",
    feature = "actuation",
    feature = "operator-console"
))]
pub(crate) fn agent_control_test_runtime_with_typed_ingress(
    capacity: AgentControlRuntimeQueueCapacity,
) -> (
    AgentControlRuntimeSender,
    AgentControlRuntimeReceiver,
    AgentControlTypedIngress,
) {
    let (mut sender, mut receiver) = agent_control_runtime_queue(capacity);
    let (priority, priority_receiver) = mpsc::sync_channel(capacity.get());
    receiver.priority = Some(priority_receiver);
    sender.priority = Some(priority.clone());
    let ingress = AgentControlTypedIngress {
        normal: sender.inner.clone(),
        priority,
        next_key: NonZeroU64::new(1),
    };
    (sender, receiver, ingress)
}

fn request_is_priority(command: AgentControlCommandV1) -> bool {
    matches!(
        command,
        AgentControlCommandV1::Stop
            | AgentControlCommandV1::Disarm
            | AgentControlCommandV1::MapOnly
            | AgentControlCommandV1::ManualStop(_)
            | AgentControlCommandV1::Shutdown
    )
}

impl AgentControlRuntimeSender {
    fn try_send(
        &self,
        dispatch: AgentControlDispatch,
    ) -> Result<(), AgentControlRuntimeTrySendError> {
        if request_is_priority(dispatch.request().command())
            && let Some(priority) = self.priority.as_ref()
        {
            return priority
                .try_send(dispatch)
                .map_err(AgentControlRuntimeTrySendError::from);
        }
        self.inner
            .try_send(dispatch)
            .map_err(AgentControlRuntimeTrySendError::from)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AgentControlRuntimeTrySendError {
    Full,
    Disconnected,
}

impl From<TrySendError<AgentControlDispatch>> for AgentControlRuntimeTrySendError {
    fn from(source: TrySendError<AgentControlDispatch>) -> Self {
        match source {
            TrySendError::Full(_) => Self::Full,
            TrySendError::Disconnected(_) => Self::Disconnected,
        }
    }
}

#[cfg(all(test, feature = "agent-runtime", feature = "actuation"))]
pub(crate) fn enqueue_agent_control_test_request(
    sender: &AgentControlRuntimeSender,
    request: AgentControlRequestV1,
    received_at: HostMonotonicTimestamp,
) -> JoinHandle<Option<AgentControlResponseV1>> {
    let (claim_sender, claim_receiver) = mpsc::sync_channel(0);
    let (response_sender, response_receiver) = mpsc::sync_channel(0);
    let (wire_delivery_sender, wire_delivery_receiver) = mpsc::sync_channel(1);
    sender
        .inner
        .try_send(AgentControlDispatch {
            request,
            received_at,
            #[cfg(feature = "operator-console")]
            typed_request_key: None,
            terminal_response_deadline: None,
            claim: claim_sender,
            response: response_sender,
            wire_delivery: wire_delivery_receiver,
        })
        .expect("test runtime queue has capacity");
    thread::spawn(move || {
        claim_receiver.recv().ok()?;
        let response = response_receiver.recv().ok()?;
        // Most test callers exercise the ordinary response API, which is
        // allowed to drop the delivery receiver. A failed acknowledgement is
        // therefore not a failed response for this general helper.
        let _ = wire_delivery_sender.send(());
        Some(response)
    })
}

#[cfg(all(
    test,
    feature = "agent-runtime",
    feature = "actuation",
    feature = "operator-console"
))]
pub(crate) fn enqueue_agent_control_test_request_through_runtime_lanes(
    sender: &AgentControlRuntimeSender,
    request: AgentControlRequestV1,
    received_at: HostMonotonicTimestamp,
) -> JoinHandle<Option<AgentControlResponseV1>> {
    let (claim_sender, claim_receiver) = mpsc::sync_channel(0);
    let (response_sender, response_receiver) = mpsc::sync_channel(0);
    let (wire_delivery_sender, wire_delivery_receiver) = mpsc::sync_channel(1);
    sender
        .try_send(AgentControlDispatch {
            request,
            received_at,
            #[cfg(feature = "operator-console")]
            typed_request_key: None,
            terminal_response_deadline: None,
            claim: claim_sender,
            response: response_sender,
            wire_delivery: wire_delivery_receiver,
        })
        .expect("test runtime lane has capacity");
    thread::spawn(move || {
        claim_receiver.recv().ok()?;
        let response = response_receiver.recv().ok()?;
        let _ = wire_delivery_sender.send(());
        Some(response)
    })
}

#[cfg(all(test, feature = "agent-runtime", feature = "actuation"))]
pub(crate) fn enqueue_agent_control_test_request_with_expired_response(
    sender: &AgentControlRuntimeSender,
    request: AgentControlRequestV1,
    received_at: HostMonotonicTimestamp,
) -> JoinHandle<()> {
    let (claim_sender, claim_receiver) = mpsc::sync_channel(0);
    let (response_sender, response_receiver) = mpsc::sync_channel(0);
    let (_wire_delivery_sender, wire_delivery_receiver) = mpsc::sync_channel(1);
    sender
        .inner
        .try_send(AgentControlDispatch {
            request,
            received_at,
            #[cfg(feature = "operator-console")]
            typed_request_key: None,
            terminal_response_deadline: None,
            claim: claim_sender,
            response: response_sender,
            wire_delivery: wire_delivery_receiver,
        })
        .expect("test runtime queue has capacity");
    thread::spawn(move || {
        claim_receiver.recv().expect("dispatcher claims request");
        drop(response_receiver);
    })
}

#[cfg(all(test, feature = "agent-runtime", feature = "actuation"))]
pub(crate) fn enqueue_agent_control_test_request_with_failed_wire_delivery(
    sender: &AgentControlRuntimeSender,
    request: AgentControlRequestV1,
    received_at: HostMonotonicTimestamp,
) -> JoinHandle<Option<AgentControlResponseV1>> {
    let (claim_sender, claim_receiver) = mpsc::sync_channel(0);
    let (response_sender, response_receiver) = mpsc::sync_channel(0);
    let (wire_delivery_sender, wire_delivery_receiver) = mpsc::sync_channel(1);
    sender
        .inner
        .try_send(AgentControlDispatch {
            request,
            received_at,
            #[cfg(feature = "operator-console")]
            typed_request_key: None,
            terminal_response_deadline: None,
            claim: claim_sender,
            response: response_sender,
            wire_delivery: wire_delivery_receiver,
        })
        .expect("test runtime queue has capacity");
    thread::spawn(move || {
        claim_receiver.recv().ok()?;
        let response = response_receiver.recv().ok()?;
        drop(wire_delivery_sender);
        Some(response)
    })
}

impl AgentControlRuntimeReceiver {
    /// Wait until one request is available or all producers are gone.
    pub fn recv(&self) -> Result<AgentControlDispatch, AgentControlRuntimeReceiveError> {
        loop {
            match self.try_recv() {
                Ok(dispatch) => return Ok(dispatch),
                Err(AgentControlRuntimeReceiveError::Empty) => {
                    thread::park_timeout(AGENT_CONTROL_RUNTIME_PRIORITY_POLL);
                }
                Err(source) => return Err(source),
            }
        }
    }

    /// Wait for one bounded duration.
    pub fn recv_timeout(
        &self,
        timeout: Duration,
    ) -> Result<AgentControlDispatch, AgentControlRuntimeReceiveError> {
        let started = Instant::now();
        loop {
            match self.try_recv() {
                Ok(dispatch) => return Ok(dispatch),
                Err(AgentControlRuntimeReceiveError::Empty) => {}
                Err(source) => return Err(source),
            }
            let remaining = timeout.saturating_sub(started.elapsed());
            if remaining.is_zero() {
                return Err(AgentControlRuntimeReceiveError::Timeout);
            }
            thread::park_timeout(remaining.min(AGENT_CONTROL_RUNTIME_PRIORITY_POLL));
        }
    }

    /// Poll without blocking.
    pub fn try_recv(&self) -> Result<AgentControlDispatch, AgentControlRuntimeReceiveError> {
        let priority_state = match self.priority.as_ref().map(Receiver::try_recv) {
            Some(Ok(dispatch)) => return Ok(dispatch),
            Some(Err(TryRecvError::Empty)) => Some(false),
            Some(Err(TryRecvError::Disconnected)) => Some(true),
            None => None,
        };
        match self.inner.try_recv() {
            Ok(dispatch) => Ok(dispatch),
            Err(TryRecvError::Empty) => Err(AgentControlRuntimeReceiveError::Empty),
            Err(TryRecvError::Disconnected) => {
                if matches!(priority_state, Some(false)) {
                    Err(AgentControlRuntimeReceiveError::Empty)
                } else {
                    Err(AgentControlRuntimeReceiveError::Disconnected)
                }
            }
        }
    }
}

/// Runtime queue receive outcome.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AgentControlRuntimeReceiveError {
    Empty,
    Timeout,
    Disconnected,
}

impl fmt::Display for AgentControlRuntimeReceiveError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "agent-control runtime receive failed: {self:?}")
    }
}

impl std::error::Error for AgentControlRuntimeReceiveError {}

/// Parsed request waiting for a runtime claim.
///
/// Claim and response are separate zero-capacity one-shot rendezvous. A
/// runtime must call [`claim`](Self::claim) before causing command side
/// effects. If the server's total runtime deadline expires before that
/// rendezvous, no executable token can be produced.
#[derive(Debug)]
pub struct AgentControlDispatch {
    request: AgentControlRequestV1,
    received_at: HostMonotonicTimestamp,
    #[cfg(feature = "operator-console")]
    typed_request_key: Option<AgentControlTypedRequestKey>,
    terminal_response_deadline: Option<Instant>,
    claim: SyncSender<()>,
    response: SyncSender<AgentControlResponseV1>,
    wire_delivery: Receiver<()>,
}

impl AgentControlDispatch {
    pub const fn request(&self) -> AgentControlRequestV1 {
        self.request
    }

    pub const fn received_at(&self) -> HostMonotonicTimestamp {
        self.received_at
    }

    #[cfg(feature = "operator-console")]
    pub(crate) const fn typed_request_key(&self) -> Option<AgentControlTypedRequestKey> {
        self.typed_request_key
    }

    /// Rendezvous with the server and obtain the only token from which command
    /// completion can be reported. This is not supervisor authority or
    /// hardware admission.
    pub fn claim(self) -> Result<AgentControlClaimedRequest, AgentControlDispatchResponseError> {
        self.claim
            .send(())
            .map_err(|_| AgentControlDispatchResponseError::ClientUnavailable)?;
        Ok(AgentControlClaimedRequest {
            request: self.request,
            received_at: self.received_at,
            terminal_response_deadline: self.terminal_response_deadline,
            response: self.response,
            wire_delivery: self.wire_delivery,
        })
    }
}

/// Request released only after an internal claim rendezvous. It still carries
/// no authority proof. Exactly one consuming response method must follow.
#[derive(Debug)]
pub struct AgentControlClaimedRequest {
    request: AgentControlRequestV1,
    received_at: HostMonotonicTimestamp,
    terminal_response_deadline: Option<Instant>,
    response: SyncSender<AgentControlResponseV1>,
    wire_delivery: Receiver<()>,
}

impl AgentControlClaimedRequest {
    pub const fn request(&self) -> AgentControlRequestV1 {
        self.request
    }

    pub const fn received_at(&self) -> HostMonotonicTimestamp {
        self.received_at
    }

    /// Exact absolute response deadline supplied by the direct socket owner.
    /// Trusted in-process submissions have no socket rendezvous and therefore
    /// return `None`; their terminal owner must derive a deadline from the
    /// same parsed terminal policy.
    pub const fn terminal_response_deadline(&self) -> Option<Instant> {
        self.terminal_response_deadline
    }

    /// Report that a long-running command has passed runtime admission, then
    /// release a token that may be processed after the wire acknowledgement.
    /// If the client/server rendezvous has expired, no token is returned.
    pub fn respond_accepted_for_processing(
        self,
    ) -> Result<AgentControlAcceptedRequest, AgentControlDispatchResponseError> {
        if matches!(self.request.command(), AgentControlCommandV1::QueryStatus) {
            return Err(AgentControlDispatchResponseError::StatusQueryRequiresStatus);
        }
        self.response
            .send(AgentControlResponseV1::accepted(
                self.request.request_id(),
                self.request.command().kind(),
                AgentControlCompletionV1::AcceptedForProcessing,
            ))
            .map_err(|_| AgentControlDispatchResponseError::ClientUnavailable)?;
        Ok(AgentControlAcceptedRequest {
            request: self.request,
            received_at: self.received_at,
        })
    }

    /// Report exact completion after the runtime has obtained the evidence
    /// required by this command. The command discriminator is derived from the
    /// claimed request and cannot be mismatched by the caller.
    pub fn respond_completed(self) -> Result<(), AgentControlDispatchResponseError> {
        if matches!(self.request.command(), AgentControlCommandV1::QueryStatus) {
            return Err(AgentControlDispatchResponseError::StatusQueryRequiresStatus);
        }
        self.response
            .send(AgentControlResponseV1::accepted(
                self.request.request_id(),
                self.request.command().kind(),
                AgentControlCompletionV1::Completed,
            ))
            .map_err(|_| AgentControlDispatchResponseError::ClientUnavailable)
    }

    /// Report exact completion and wait until the server confirms that the
    /// complete response frame reached the connected socket.
    ///
    /// Shutdown uses this stronger boundary before clearing the shared run
    /// flag. A write failure, cancellation, or socket-thread exit drops the
    /// acknowledgement sender and is reported as delivery uncertainty.
    pub fn respond_completed_after_wire_delivery(
        self,
    ) -> Result<(), AgentControlDispatchResponseError> {
        if matches!(self.request.command(), AgentControlCommandV1::QueryStatus) {
            return Err(AgentControlDispatchResponseError::StatusQueryRequiresStatus);
        }
        self.response
            .send(AgentControlResponseV1::accepted(
                self.request.request_id(),
                self.request.command().kind(),
                AgentControlCompletionV1::Completed,
            ))
            .map_err(|_| AgentControlDispatchResponseError::ClientUnavailable)?;
        self.wire_delivery
            .recv()
            .map_err(|_| AgentControlDispatchResponseError::WireDeliveryUncertain)
    }

    /// Return the exact status snapshot for a claimed query.
    pub fn respond_status(
        self,
        status: AgentControlStatusV1,
    ) -> Result<(), AgentControlDispatchResponseError> {
        if !matches!(self.request.command(), AgentControlCommandV1::QueryStatus) {
            return Err(AgentControlDispatchResponseError::StatusForNonQuery);
        }
        self.response
            .send(AgentControlResponseV1::status(
                self.request.request_id(),
                status,
            ))
            .map_err(|_| AgentControlDispatchResponseError::ClientUnavailable)
    }

    /// Report a final runtime rejection after claim.
    pub fn reject(
        self,
        code: AgentControlRejectionCodeV1,
        retryable: bool,
    ) -> Result<(), AgentControlDispatchResponseError> {
        self.response
            .send(AgentControlResponseV1::rejected(
                Some(self.request.request_id()),
                code,
                retryable,
            ))
            .map_err(|_| AgentControlDispatchResponseError::ClientUnavailable)
    }

    /// Send a final rejection and retain the socket owner until the complete
    /// frame reaches the connected client. Terminal operations use this
    /// before clearing their shared run flag.
    pub fn reject_after_wire_delivery(
        self,
        code: AgentControlRejectionCodeV1,
        retryable: bool,
    ) -> Result<(), AgentControlDispatchResponseError> {
        self.response
            .send(AgentControlResponseV1::rejected(
                Some(self.request.request_id()),
                code,
                retryable,
            ))
            .map_err(|_| AgentControlDispatchResponseError::ClientUnavailable)?;
        self.wire_delivery
            .recv()
            .map_err(|_| AgentControlDispatchResponseError::WireDeliveryUncertain)
    }
}

/// Long-running request released only after its truthful
/// `accepted_for_processing` response reached the server.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AgentControlAcceptedRequest {
    request: AgentControlRequestV1,
    received_at: HostMonotonicTimestamp,
}

impl AgentControlAcceptedRequest {
    pub const fn request(self) -> AgentControlRequestV1 {
        self.request
    }

    pub const fn received_at(self) -> HostMonotonicTimestamp {
        self.received_at
    }
}

/// Invalid or expired runtime response attempt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AgentControlDispatchResponseError {
    StatusQueryRequiresStatus,
    StatusForNonQuery,
    ClientUnavailable,
    WireDeliveryUncertain,
}

impl fmt::Display for AgentControlDispatchResponseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "agent-control dispatch response failed: {self:?}"
        )
    }
}

impl std::error::Error for AgentControlDispatchResponseError {}

/// Sequential owner of the local control listener and process-lifetime parser.
pub struct AgentControlSocketServer {
    listener: UnixListener,
    path_guard: SocketPathGuard,
    timeouts: AgentControlSocketTimeouts,
    clock: AgentControlMonotonicOrigin,
    runtime: AgentControlRuntimeSender,
    parser: AgentControlRequestParser,
}

type AgentControlPayloadOutcome = (
    AgentControlResponseV1,
    Option<AgentControlConnectionIssue>,
    Option<SyncSender<()>>,
);

impl AgentControlSocketServer {
    /// Verify the private directory, reject any existing destination, bind the
    /// exact path, and set the created socket inode to mode `0600`.
    pub fn bind(
        config: AgentControlSocketConfig,
        clock: AgentControlMonotonicOrigin,
        runtime: AgentControlRuntimeSender,
    ) -> Result<Self, AgentControlSocketBindError> {
        verify_private_parent(config.path.parent())?;
        match fs::symlink_metadata(config.path.as_path()) {
            Ok(_) => return Err(AgentControlSocketBindError::DestinationExists),
            Err(source) if source.kind() == io::ErrorKind::NotFound => {}
            Err(source) => {
                return Err(AgentControlSocketBindError::InspectDestination { source });
            }
        }

        let listener = UnixListener::bind(config.path.as_path())
            .map_err(|source| AgentControlSocketBindError::Bind { source })?;
        let descriptor = socket_fd_properties(&listener)
            .map_err(|source| AgentControlSocketBindError::InspectCreatedSocket { source })?;
        if descriptor.owner_uid != effective_uid() {
            return Err(AgentControlSocketBindError::CreatedSocketOwnerMismatch {
                expected_uid: effective_uid(),
                actual_uid: descriptor.owner_uid,
            });
        }
        if !descriptor.is_socket {
            return Err(AgentControlSocketBindError::CreatedObjectIsNotSocket);
        }

        let identity = path_socket_identity(config.path.as_path())
            .map_err(|source| AgentControlSocketBindError::InspectCreatedSocketPath { source })?;
        if identity.owner_uid != effective_uid() {
            return Err(AgentControlSocketBindError::CreatedSocketOwnerMismatch {
                expected_uid: effective_uid(),
                actual_uid: identity.owner_uid,
            });
        }
        if !identity.is_socket {
            return Err(AgentControlSocketBindError::CreatedObjectIsNotSocket);
        }
        let path_guard = SocketPathGuard::new(config.path.path.clone(), identity);
        set_socket_path_mode(config.path.as_path(), 0o600)
            .map_err(|source| AgentControlSocketBindError::SetSocketPermissions { source })?;
        let path_after = path_socket_identity(config.path.as_path())
            .map_err(|source| AgentControlSocketBindError::InspectCreatedSocketPath { source })?;
        if !identity.same_inode(path_after)
            || !path_after.is_socket
            || path_after.permission_bits != 0o600
        {
            return Err(AgentControlSocketBindError::SocketPathChanged {
                expected_device: identity.device,
                expected_inode: identity.inode,
                actual_device: path_after.device,
                actual_inode: path_after.inode,
                actual_mode: path_after.permission_bits,
                actual_is_socket: path_after.is_socket,
            });
        }
        listener
            .set_nonblocking(true)
            .map_err(|source| AgentControlSocketBindError::ConfigureNonblocking { source })?;

        Ok(Self {
            listener,
            path_guard,
            timeouts: config.timeouts,
            clock,
            runtime,
            parser: AgentControlRequestParser::new(),
        })
    }

    /// Exact socket pathname currently owned by this server.
    pub fn socket_path(&self) -> &Path {
        &self.path_guard.path
    }

    /// Last request ID accepted by the process-lifetime parser.
    pub const fn last_request_id(&self) -> Option<AgentControlRequestId> {
        self.parser.last_request_id()
    }

    /// Poll for and finish at most one client connection.
    ///
    /// An idle listener returns [`AgentControlServeOutcome::Idle`] immediately,
    /// allowing the sole owner to observe cancellation without a wakeup client.
    /// Every operation after a connection is accepted has an absolute deadline.
    /// The connection is dropped after one response regardless of extra frames.
    pub fn poll_one(&mut self) -> Result<AgentControlServeOutcome, AgentControlServeError> {
        self.poll_one_with_cancellation(None)
    }

    fn poll_one_for_task(
        &mut self,
        running: &AtomicBool,
    ) -> Result<AgentControlServeOutcome, AgentControlServeError> {
        self.poll_one_with_cancellation(Some(running))
    }

    fn poll_one_with_cancellation(
        &mut self,
        cancellation: Option<&AtomicBool>,
    ) -> Result<AgentControlServeOutcome, AgentControlServeError> {
        require_not_cancelled(cancellation)?;
        self.path_guard.verify_current().map_err(|observed| {
            AgentControlServeError::SocketPathNoLongerOwned {
                expected_device: self.path_guard.identity.device,
                expected_inode: self.path_guard.identity.inode,
                observed,
            }
        })?;
        let (mut stream, _) = match self.listener.accept() {
            Ok(accepted) => accepted,
            Err(source) if source.kind() == io::ErrorKind::WouldBlock => {
                return Ok(AgentControlServeOutcome::Idle);
            }
            Err(source) => return Err(AgentControlServeError::Accept { source }),
        };
        stream
            .set_nonblocking(false)
            .map_err(|source| AgentControlServeError::ConfigureClient { source })?;
        let mut request_payload = [0_u8; MAX_AGENT_CONTROL_REQUEST_JSON_BYTES];
        let (response, connection_issue, wire_delivery) = match read_request_frame_into(
            &mut stream,
            self.timeouts.read,
            &mut request_payload,
            cancellation,
        ) {
            Ok(payload) => self.handle_payload(payload, cancellation)?,
            Err(ReadRequestFrameError::InvalidLength { declared_bytes }) => (
                AgentControlResponseV1::rejected(
                    None,
                    AgentControlRejectionCodeV1::MalformedRequest,
                    false,
                ),
                Some(AgentControlConnectionIssue::InvalidFrameLength { declared_bytes }),
                None,
            ),
            Err(ReadRequestFrameError::Io { source, timeout }) => (
                AgentControlResponseV1::rejected(
                    None,
                    AgentControlRejectionCodeV1::MalformedRequest,
                    timeout,
                ),
                Some(AgentControlConnectionIssue::Read { source, timeout }),
                None,
            ),
            Err(ReadRequestFrameError::ShutdownRequested) => {
                return Err(AgentControlServeError::ShutdownRequested);
            }
        };
        write_response_frame(&mut stream, response, self.timeouts.write, cancellation)?;
        if let Some(wire_delivery) = wire_delivery {
            // The runtime may use the ordinary response API and deliberately
            // drop its receiver. Wire success remains successful either way.
            let _ = wire_delivery.send(());
        }
        Ok(AgentControlServeOutcome::Responded {
            response,
            connection_issue,
        })
    }

    fn handle_payload(
        &mut self,
        payload: &[u8],
        cancellation: Option<&AtomicBool>,
    ) -> Result<AgentControlPayloadOutcome, AgentControlServeError> {
        require_not_cancelled(cancellation)?;
        let request = match self.parser.parse_next(payload) {
            Ok(request) => request,
            Err(source) => {
                return Ok((
                    parse_rejection(&source),
                    Some(AgentControlConnectionIssue::RequestParse { source }),
                    None,
                ));
            }
        };
        let received_at = match self.clock.try_now() {
            Ok(received_at) => received_at,
            Err(source) => {
                return Ok((
                    AgentControlResponseV1::rejected(
                        Some(request.request_id()),
                        AgentControlRejectionCodeV1::InternalFault,
                        false,
                    ),
                    Some(AgentControlConnectionIssue::Clock { source }),
                    None,
                ));
            }
        };
        let runtime_timeout = self.timeouts.response_timeout_for(request.command());
        let runtime_deadline = Instant::now()
            .checked_add(runtime_timeout)
            .ok_or(AgentControlServeError::DeadlineOverflow)?;
        let (claim_sender, claim_receiver) = mpsc::sync_channel(0);
        let (response_sender, response_receiver) = mpsc::sync_channel(0);
        let (wire_delivery_sender, wire_delivery_receiver) = mpsc::sync_channel(1);
        let dispatch = AgentControlDispatch {
            request,
            received_at,
            #[cfg(feature = "operator-console")]
            typed_request_key: None,
            terminal_response_deadline: matches!(request.command(), AgentControlCommandV1::SaveMap)
                .then_some(runtime_deadline),
            claim: claim_sender,
            response: response_sender,
            wire_delivery: wire_delivery_receiver,
        };
        require_not_cancelled(cancellation)?;
        match self.runtime.try_send(dispatch) {
            Ok(()) => {}
            Err(AgentControlRuntimeTrySendError::Full) => {
                return Ok((
                    AgentControlResponseV1::rejected(
                        Some(request.request_id()),
                        AgentControlRejectionCodeV1::NotReady,
                        true,
                    ),
                    Some(AgentControlConnectionIssue::RuntimeQueueFull),
                    None,
                ));
            }
            Err(AgentControlRuntimeTrySendError::Disconnected) => {
                return Ok((
                    AgentControlResponseV1::rejected(
                        Some(request.request_id()),
                        AgentControlRejectionCodeV1::ShutdownInProgress,
                        false,
                    ),
                    Some(AgentControlConnectionIssue::RuntimeQueueDisconnected),
                    None,
                ));
            }
        }

        match recv_until(&claim_receiver, runtime_deadline, cancellation) {
            Ok(()) => {}
            Err(CancellableReceiveError::Deadline) => {
                return Ok((
                    AgentControlResponseV1::rejected(
                        Some(request.request_id()),
                        AgentControlRejectionCodeV1::NotReady,
                        true,
                    ),
                    Some(AgentControlConnectionIssue::RuntimeClaimTimeout),
                    None,
                ));
            }
            Err(CancellableReceiveError::Disconnected) => {
                return Ok((
                    AgentControlResponseV1::rejected(
                        Some(request.request_id()),
                        AgentControlRejectionCodeV1::InternalFault,
                        false,
                    ),
                    Some(AgentControlConnectionIssue::RuntimeClaimChannelClosed),
                    None,
                ));
            }
            Err(CancellableReceiveError::ShutdownRequested) => {
                return Err(AgentControlServeError::ShutdownRequested);
            }
        }

        match recv_until(&response_receiver, runtime_deadline, cancellation) {
            Ok(response) => Ok((response, None, Some(wire_delivery_sender))),
            Err(CancellableReceiveError::Deadline) => {
                Err(AgentControlServeError::RuntimeResponseDeadlineAfterClaim {
                    request_id: request.request_id(),
                })
            }
            Err(CancellableReceiveError::Disconnected) => Err(
                AgentControlServeError::RuntimeResponseChannelClosedAfterClaim {
                    request_id: request.request_id(),
                },
            ),
            Err(CancellableReceiveError::ShutdownRequested) => {
                Err(AgentControlServeError::ShutdownRequested)
            }
        }
    }

    /// Stop serving and conditionally remove only the created socket inode.
    pub fn shutdown(mut self) -> AgentControlSocketCleanupOutcome {
        self.path_guard.cleanup()
    }
}

impl Drop for AgentControlSocketServer {
    fn drop(&mut self) {
        let _ = self.path_guard.cleanup();
    }
}

/// Dedicated sequential socket owner used by the live agent.
///
/// The runtime receiver remains in the navigation owner. If this task exits,
/// its sole sender is dropped and the receiver observes disconnection as a
/// fail-closed runtime fault.
#[must_use = "the live owner must join the control socket task and inspect cleanup"]
pub struct AgentControlSocketTask {
    running: Arc<AtomicBool>,
    handle: Option<JoinHandle<AgentControlSocketTaskExit>>,
}

struct AgentControlSocketTaskParts {
    task: AgentControlSocketTask,
    receiver: AgentControlRuntimeReceiver,
    #[cfg(all(
        feature = "agent-runtime",
        feature = "actuation",
        feature = "operator-console"
    ))]
    typed_ingress: AgentControlTypedIngress,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AgentControlSocketChildStart {
    Ready,
    ShutdownRequested,
    StartStateUnavailable,
}

impl AgentControlSocketTask {
    /// Bind before spawning, so startup cannot report readiness until the
    /// private path and permissions have been established. Thread creation is
    /// fallible and reported without panicking; when creation fails, the
    /// created socket is synchronously cleaned and its exact outcome is
    /// retained in the error.
    pub fn bind_and_spawn(
        config: AgentControlSocketConfig,
        clock: AgentControlMonotonicOrigin,
        capacity: AgentControlRuntimeQueueCapacity,
        running: Arc<AtomicBool>,
    ) -> Result<(Self, AgentControlRuntimeReceiver), AgentControlSocketTaskStartError> {
        Self::bind_and_spawn_with(config, clock, capacity, running, |task_main| {
            thread::Builder::new()
                .name(AGENT_CONTROL_SOCKET_THREAD_NAME.to_owned())
                .spawn(task_main)
        })
    }

    /// Start the socket owner and release exactly one additional, typed
    /// in-process ingress for the authenticated unified console.
    ///
    /// Socket and console requests still converge on the same receiver and
    /// therefore the same supervisor/motion owner. The extra ingress is not
    /// cloneable and accepts only an existing [`AgentControlRequestV1`].
    #[cfg(all(
        feature = "agent-runtime",
        feature = "actuation",
        feature = "operator-console"
    ))]
    pub fn bind_and_spawn_with_typed_ingress(
        config: AgentControlSocketConfig,
        clock: AgentControlMonotonicOrigin,
        capacity: AgentControlRuntimeQueueCapacity,
        running: Arc<AtomicBool>,
    ) -> Result<
        (Self, AgentControlRuntimeReceiver, AgentControlTypedIngress),
        AgentControlSocketTaskStartError,
    > {
        Self::bind_and_spawn_components(config, clock, capacity, running, |task_main| {
            thread::Builder::new()
                .name(AGENT_CONTROL_SOCKET_THREAD_NAME.to_owned())
                .spawn(task_main)
        })
        .map(|parts| (parts.task, parts.receiver, parts.typed_ingress))
    }

    fn bind_and_spawn_with<Spawn>(
        config: AgentControlSocketConfig,
        clock: AgentControlMonotonicOrigin,
        capacity: AgentControlRuntimeQueueCapacity,
        running: Arc<AtomicBool>,
        spawn: Spawn,
    ) -> Result<(Self, AgentControlRuntimeReceiver), AgentControlSocketTaskStartError>
    where
        Spawn: FnOnce(
            Box<dyn FnOnce() -> AgentControlSocketTaskExit + Send + 'static>,
        ) -> io::Result<JoinHandle<AgentControlSocketTaskExit>>,
    {
        Self::bind_and_spawn_components(config, clock, capacity, running, spawn)
            .map(|parts| (parts.task, parts.receiver))
    }

    fn bind_and_spawn_components<Spawn>(
        config: AgentControlSocketConfig,
        clock: AgentControlMonotonicOrigin,
        capacity: AgentControlRuntimeQueueCapacity,
        running: Arc<AtomicBool>,
        spawn: Spawn,
    ) -> Result<AgentControlSocketTaskParts, AgentControlSocketTaskStartError>
    where
        Spawn: FnOnce(
            Box<dyn FnOnce() -> AgentControlSocketTaskExit + Send + 'static>,
        ) -> io::Result<JoinHandle<AgentControlSocketTaskExit>>,
    {
        if !running.load(Ordering::Acquire) {
            return Err(AgentControlSocketTaskStartError::ShutdownAlreadyRequested);
        }
        let (mut sender, mut receiver) = agent_control_runtime_queue(capacity);
        let (priority, priority_receiver) = mpsc::sync_channel(capacity.get());
        receiver.priority = Some(priority_receiver);
        sender.priority = Some(priority.clone());
        #[cfg(all(
            feature = "agent-runtime",
            feature = "actuation",
            feature = "operator-console"
        ))]
        let typed_ingress = AgentControlTypedIngress {
            normal: sender.inner.clone(),
            priority,
            next_key: NonZeroU64::new(1),
        };
        let server = match AgentControlSocketServer::bind(config, clock, sender) {
            Ok(server) => server,
            Err(source) => {
                running.store(false, Ordering::Release);
                return Err(AgentControlSocketTaskStartError::Bind { source });
            }
        };

        // Keep ownership outside the task closure until the spawn operation
        // succeeds. If the OS refuses the thread, the caller can explicitly
        // shut down the still-owned server and report its cleanup outcome.
        let pending_server = Arc::new(Mutex::new(Some(server)));
        let task_server = Arc::clone(&pending_server);
        let task_running = Arc::clone(&running);
        let (start_sender, start_receiver) = mpsc::sync_channel(1);
        let task_main = Box::new(move || {
            let mut server_slot = task_server
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let Some(mut server) = server_slot.take() else {
                task_running.store(false, Ordering::Release);
                let _ = start_sender.send(AgentControlSocketChildStart::StartStateUnavailable);
                return AgentControlSocketTaskExit::StartStateUnavailable;
            };
            drop(server_slot);

            if !task_running.load(Ordering::Acquire) {
                let _ = start_sender.send(AgentControlSocketChildStart::ShutdownRequested);
                return AgentControlSocketTaskExit::Shutdown {
                    cleanup: server.shutdown(),
                };
            }
            if start_sender
                .send(AgentControlSocketChildStart::Ready)
                .is_err()
            {
                task_running.store(false, Ordering::Release);
                return AgentControlSocketTaskExit::StartupObserverUnavailable {
                    cleanup: server.shutdown(),
                };
            }

            while task_running.load(Ordering::Acquire) {
                match server.poll_one_for_task(&task_running) {
                    Ok(AgentControlServeOutcome::Idle) => {
                        thread::park_timeout(Duration::from_millis(1));
                    }
                    Ok(AgentControlServeOutcome::Responded { .. }) => {}
                    Err(AgentControlServeError::ShutdownRequested) => {
                        return AgentControlSocketTaskExit::Shutdown {
                            cleanup: server.shutdown(),
                        };
                    }
                    Err(source) => {
                        task_running.store(false, Ordering::Release);
                        return AgentControlSocketTaskExit::ServeFailed {
                            source,
                            cleanup: server.shutdown(),
                        };
                    }
                }
            }
            AgentControlSocketTaskExit::Shutdown {
                cleanup: server.shutdown(),
            }
        });
        let handle = match spawn(task_main) {
            Ok(handle) => handle,
            Err(source) => {
                running.store(false, Ordering::Release);
                let mut server_slot = pending_server
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                return match server_slot.take() {
                    Some(server) => Err(AgentControlSocketTaskStartError::ThreadSpawn {
                        source,
                        cleanup: server.shutdown(),
                    }),
                    None => {
                        Err(AgentControlSocketTaskStartError::ThreadSpawnOwnershipLost { source })
                    }
                };
            }
        };
        drop(pending_server);
        let mut task = Self {
            running,
            handle: Some(handle),
        };
        match start_receiver.recv_timeout(AGENT_CONTROL_SOCKET_STARTUP_TIMEOUT) {
            Ok(AgentControlSocketChildStart::Ready) => Ok(AgentControlSocketTaskParts {
                task,
                receiver,
                #[cfg(all(
                    feature = "agent-runtime",
                    feature = "actuation",
                    feature = "operator-console"
                ))]
                typed_ingress,
            }),
            Ok(AgentControlSocketChildStart::ShutdownRequested) => {
                task.request_shutdown();
                let task_exit = task.join_handle();
                Err(AgentControlSocketTaskStartError::ChildNotReady {
                    reason: AgentControlSocketTaskStartFailure::ShutdownRequested,
                    task_exit: Box::new(task_exit),
                })
            }
            Ok(AgentControlSocketChildStart::StartStateUnavailable) => {
                task.request_shutdown();
                let task_exit = task.join_handle();
                Err(AgentControlSocketTaskStartError::ChildNotReady {
                    reason: AgentControlSocketTaskStartFailure::StartStateUnavailable,
                    task_exit: Box::new(task_exit),
                })
            }
            Err(RecvTimeoutError::Timeout) => {
                drop(start_receiver);
                task.request_shutdown();
                let task_exit = task.join_handle();
                Err(AgentControlSocketTaskStartError::ChildNotReady {
                    reason: AgentControlSocketTaskStartFailure::ReadyTimeout {
                        timeout: AGENT_CONTROL_SOCKET_STARTUP_TIMEOUT,
                    },
                    task_exit: Box::new(task_exit),
                })
            }
            Err(RecvTimeoutError::Disconnected) => {
                task.request_shutdown();
                let task_exit = task.join_handle();
                Err(AgentControlSocketTaskStartError::ChildNotReady {
                    reason: AgentControlSocketTaskStartFailure::ReadyChannelClosed,
                    task_exit: Box::new(task_exit),
                })
            }
        }
    }

    /// Signal this task and the shared fail-closed runtime flag to stop.
    ///
    /// This is nonblocking. Call [`shutdown`](Self::shutdown) when cleanup
    /// evidence is required before proceeding.
    pub fn request_shutdown(&self) {
        self.running.store(false, Ordering::Release);
        if let Some(handle) = &self.handle {
            handle.thread().unpark();
        }
    }

    /// Explicitly signal, wake, and join the socket owner.
    ///
    /// The returned exit contains the conditional socket cleanup outcome.
    /// In-flight read, runtime-rendezvous, and write waits cooperatively poll
    /// the same run flag, so shutdown does not wait for their longer configured
    /// deadlines.
    pub fn shutdown(
        mut self,
    ) -> Result<AgentControlSocketTaskExit, AgentControlSocketTaskJoinError> {
        self.request_shutdown();
        self.join_handle()
    }

    /// Backwards-compatible stop-and-join operation.
    ///
    /// There is deliberately no public wait-only join: waiting on a healthy
    /// owner without first clearing its run flag would never complete.
    pub fn join(self) -> Result<AgentControlSocketTaskExit, AgentControlSocketTaskJoinError> {
        self.shutdown()
    }

    fn join_handle(
        &mut self,
    ) -> Result<AgentControlSocketTaskExit, AgentControlSocketTaskJoinError> {
        let handle = self
            .handle
            .take()
            .ok_or(AgentControlSocketTaskJoinError::HandleUnavailable)?;
        handle
            .join()
            .map_err(|_| AgentControlSocketTaskJoinError::Panicked)
    }
}

impl Drop for AgentControlSocketTask {
    fn drop(&mut self) {
        if self.handle.is_none() {
            return;
        }
        // A dropped owner is a fail-closed shutdown, never a detached control
        // thread. Explicit shutdown should normally be used so the caller can
        // inspect the exit and cleanup evidence.
        self.request_shutdown();
        let _ = self.join_handle();
    }
}

/// Failure before a control-socket task becomes an owned live thread.
#[derive(Debug)]
pub enum AgentControlSocketTaskStartError {
    ShutdownAlreadyRequested,
    Bind {
        source: AgentControlSocketBindError,
    },
    ThreadSpawn {
        source: io::Error,
        cleanup: AgentControlSocketCleanupOutcome,
    },
    /// Internal ownership invariant failure retained instead of guessing that
    /// cleanup occurred. The production thread spawner cannot start a closure
    /// while returning an error.
    ThreadSpawnOwnershipLost {
        source: io::Error,
    },
    ChildNotReady {
        reason: AgentControlSocketTaskStartFailure,
        task_exit: Box<Result<AgentControlSocketTaskExit, AgentControlSocketTaskJoinError>>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AgentControlSocketTaskStartFailure {
    ShutdownRequested,
    StartStateUnavailable,
    ReadyTimeout { timeout: Duration },
    ReadyChannelClosed,
}

impl fmt::Display for AgentControlSocketTaskStartError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShutdownAlreadyRequested => formatter.write_str(
                "agent-control task startup rejected because shutdown was already requested",
            ),
            Self::Bind { source } => write!(formatter, "{source}"),
            Self::ThreadSpawn { source, cleanup } => write!(
                formatter,
                "agent-control thread creation failed: {source}; socket cleanup: {cleanup:?}"
            ),
            Self::ThreadSpawnOwnershipLost { source } => write!(
                formatter,
                "agent-control thread creation failed and bound-server ownership was unexpectedly lost: {source}"
            ),
            Self::ChildNotReady { reason, task_exit } => write!(
                formatter,
                "agent-control child did not establish readiness: {reason:?}; task exit: {task_exit:?}"
            ),
        }
    }
}

impl std::error::Error for AgentControlSocketTaskStartError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Bind { source } => Some(source),
            Self::ThreadSpawn { source, .. } | Self::ThreadSpawnOwnershipLost { source } => {
                Some(source)
            }
            Self::ShutdownAlreadyRequested | Self::ChildNotReady { .. } => None,
        }
    }
}

#[derive(Debug)]
pub enum AgentControlSocketTaskExit {
    Shutdown {
        cleanup: AgentControlSocketCleanupOutcome,
    },
    ServeFailed {
        source: AgentControlServeError,
        cleanup: AgentControlSocketCleanupOutcome,
    },
    StartupObserverUnavailable {
        cleanup: AgentControlSocketCleanupOutcome,
    },
    /// The spawned closure could not acquire the server placed there before
    /// thread creation. This is an internal ownership invariant failure.
    StartStateUnavailable,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AgentControlSocketTaskJoinError {
    Panicked,
    HandleUnavailable,
}

impl fmt::Display for AgentControlSocketTaskJoinError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Panicked => formatter.write_str("agent-control socket task panicked"),
            Self::HandleUnavailable => {
                formatter.write_str("agent-control socket task join handle is unavailable")
            }
        }
    }
}

impl std::error::Error for AgentControlSocketTaskJoinError {}

/// Successful completion of one client connection.
#[derive(Debug)]
pub enum AgentControlServeOutcome {
    Idle,
    Responded {
        response: AgentControlResponseV1,
        connection_issue: Option<AgentControlConnectionIssue>,
    },
}

/// Transport-level reason retained when a fixed malformed response was sent.
#[derive(Debug)]
pub enum AgentControlConnectionIssue {
    InvalidFrameLength {
        declared_bytes: u32,
    },
    Read {
        source: io::Error,
        timeout: bool,
    },
    RequestParse {
        source: AgentControlRequestParseError,
    },
    Clock {
        source: AgentControlClockError,
    },
    RuntimeQueueFull,
    RuntimeQueueDisconnected,
    RuntimeClaimTimeout,
    RuntimeClaimChannelClosed,
}

/// Listener or response-write failure.
#[derive(Debug)]
pub enum AgentControlServeError {
    /// Internal cooperative cancellation used by the owned socket task. The
    /// public standalone poll path never produces this variant.
    ShutdownRequested,
    Accept {
        source: io::Error,
    },
    ConfigureClient {
        source: io::Error,
    },
    ResponseSerialization {
        source: serde_json::Error,
    },
    ResponseTooLarge {
        maximum_bytes: usize,
    },
    ResponseDeadline,
    ResponseWrite {
        source: io::Error,
    },
    DeadlineOverflow,
    RuntimeResponseDeadlineAfterClaim {
        request_id: AgentControlRequestId,
    },
    RuntimeResponseChannelClosedAfterClaim {
        request_id: AgentControlRequestId,
    },
    SocketPathNoLongerOwned {
        expected_device: u64,
        expected_inode: u64,
        observed: AgentControlObservedSocketPath,
    },
}

impl fmt::Display for AgentControlServeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShutdownRequested => {
                formatter.write_str("agent-control socket shutdown was requested")
            }
            Self::Accept { source } => write!(formatter, "agent-control accept failed: {source}"),
            Self::ConfigureClient { source } => {
                write!(formatter, "agent-control client setup failed: {source}")
            }
            Self::ResponseSerialization { source } => {
                write!(
                    formatter,
                    "agent-control response serialization failed: {source}"
                )
            }
            Self::ResponseTooLarge { maximum_bytes } => write!(
                formatter,
                "agent-control response exceeds fixed {maximum_bytes}-byte buffer"
            ),
            Self::ResponseDeadline => {
                formatter.write_str("agent-control response write deadline expired")
            }
            Self::ResponseWrite { source } => {
                write!(formatter, "agent-control response write failed: {source}")
            }
            Self::DeadlineOverflow => {
                formatter.write_str("agent-control response deadline overflowed Instant")
            }
            Self::RuntimeResponseDeadlineAfterClaim { request_id } => write!(
                formatter,
                "agent-control runtime response deadline expired after request {} was claimed; no completion response was sent",
                request_id.get()
            ),
            Self::RuntimeResponseChannelClosedAfterClaim { request_id } => write!(
                formatter,
                "agent-control runtime dropped the final response after request {} was claimed; no completion response was sent",
                request_id.get()
            ),
            Self::SocketPathNoLongerOwned {
                expected_device,
                expected_inode,
                observed,
            } => write!(
                formatter,
                "agent-control socket path no longer names created inode {expected_device}:{expected_inode}; observed {observed:?}"
            ),
        }
    }
}

impl std::error::Error for AgentControlServeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Accept { source }
            | Self::ConfigureClient { source }
            | Self::ResponseWrite { source } => Some(source),
            Self::ResponseSerialization { source } => Some(source),
            Self::ShutdownRequested
            | Self::ResponseTooLarge { .. }
            | Self::ResponseDeadline
            | Self::DeadlineOverflow
            | Self::RuntimeResponseDeadlineAfterClaim { .. }
            | Self::RuntimeResponseChannelClosedAfterClaim { .. }
            | Self::SocketPathNoLongerOwned { .. } => None,
        }
    }
}

/// Fail-closed bind error. No variant authorizes removal of a preexisting path.
#[derive(Debug)]
pub enum AgentControlSocketBindError {
    InspectParent {
        source: io::Error,
    },
    ParentIsSymlink,
    ParentIsNotDirectory,
    ParentIsNotCanonical,
    ParentOwnerMismatch {
        expected_uid: u32,
        actual_uid: u32,
    },
    ParentModeNotPrivate {
        actual_mode: u32,
    },
    InspectDestination {
        source: io::Error,
    },
    DestinationExists,
    Bind {
        source: io::Error,
    },
    InspectCreatedSocket {
        source: io::Error,
    },
    InspectCreatedSocketPath {
        source: io::Error,
    },
    CreatedSocketOwnerMismatch {
        expected_uid: u32,
        actual_uid: u32,
    },
    CreatedObjectIsNotSocket,
    SetSocketPermissions {
        source: io::Error,
    },
    ConfigureNonblocking {
        source: io::Error,
    },
    SocketPathChanged {
        expected_device: u64,
        expected_inode: u64,
        actual_device: u64,
        actual_inode: u64,
        actual_mode: u32,
        actual_is_socket: bool,
    },
}

impl fmt::Display for AgentControlSocketBindError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "agent-control socket bind failed: {self:?}")
    }
}

impl std::error::Error for AgentControlSocketBindError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InspectParent { source }
            | Self::InspectDestination { source }
            | Self::Bind { source }
            | Self::InspectCreatedSocket { source }
            | Self::InspectCreatedSocketPath { source }
            | Self::SetSocketPermissions { source }
            | Self::ConfigureNonblocking { source } => Some(source),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SocketIdentity {
    device: u64,
    inode: u64,
    owner_uid: u32,
    permission_bits: u32,
    is_socket: bool,
}

impl SocketIdentity {
    const fn same_inode(self, other: Self) -> bool {
        self.device == other.device && self.inode == other.inode
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SocketDescriptorProperties {
    owner_uid: u32,
    is_socket: bool,
}

#[derive(Debug)]
struct SocketPathGuard {
    path: PathBuf,
    identity: SocketIdentity,
    armed: bool,
}

impl SocketPathGuard {
    fn new(path: PathBuf, identity: SocketIdentity) -> Self {
        Self {
            path,
            identity,
            armed: true,
        }
    }

    fn cleanup(&mut self) -> AgentControlSocketCleanupOutcome {
        if !self.armed {
            return AgentControlSocketCleanupOutcome::NotArmed;
        }
        self.armed = false;
        let observed = match path_socket_identity(&self.path) {
            Ok(observed) => observed,
            Err(source) if source.kind() == io::ErrorKind::NotFound => {
                return AgentControlSocketCleanupOutcome::AlreadyAbsent;
            }
            Err(source) => {
                return AgentControlSocketCleanupOutcome::InspectionFailed { source };
            }
        };
        if !observed.is_socket || !self.identity.same_inode(observed) {
            return AgentControlSocketCleanupOutcome::ReplacementPreserved {
                observed_device: observed.device,
                observed_inode: observed.inode,
                observed_is_socket: observed.is_socket,
            };
        }
        match fs::remove_file(&self.path) {
            Ok(()) => AgentControlSocketCleanupOutcome::RemovedCreatedSocket,
            Err(source) => AgentControlSocketCleanupOutcome::RemoveFailed { source },
        }
    }

    fn verify_current(&self) -> Result<(), AgentControlObservedSocketPath> {
        match path_socket_identity(&self.path) {
            Ok(observed) if observed.is_socket && self.identity.same_inode(observed) => Ok(()),
            Ok(observed) => Err(AgentControlObservedSocketPath::Present {
                device: observed.device,
                inode: observed.inode,
                is_socket: observed.is_socket,
            }),
            Err(source) if source.kind() == io::ErrorKind::NotFound => {
                Err(AgentControlObservedSocketPath::Absent)
            }
            Err(source) => Err(AgentControlObservedSocketPath::InspectionFailed {
                kind: source.kind(),
                raw_os_error: source.raw_os_error(),
            }),
        }
    }
}

impl Drop for SocketPathGuard {
    fn drop(&mut self) {
        let _ = self.cleanup();
    }
}

/// Bounded evidence retained when the serving pathname loses its identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AgentControlObservedSocketPath {
    Absent,
    Present {
        device: u64,
        inode: u64,
        is_socket: bool,
    },
    InspectionFailed {
        kind: io::ErrorKind,
        raw_os_error: Option<i32>,
    },
}

/// Explicit socket-path cleanup outcome.
#[derive(Debug)]
pub enum AgentControlSocketCleanupOutcome {
    RemovedCreatedSocket,
    AlreadyAbsent,
    ReplacementPreserved {
        observed_device: u64,
        observed_inode: u64,
        observed_is_socket: bool,
    },
    InspectionFailed {
        source: io::Error,
    },
    RemoveFailed {
        source: io::Error,
    },
    NotArmed,
}

#[allow(unsafe_code)]
fn effective_uid() -> u32 {
    // SAFETY: `geteuid` has no arguments and no memory-safety preconditions.
    unsafe { libc::geteuid() }
}

fn verify_private_parent(parent: &Path) -> Result<(), AgentControlSocketBindError> {
    let metadata = fs::symlink_metadata(parent)
        .map_err(|source| AgentControlSocketBindError::InspectParent { source })?;
    if metadata.file_type().is_symlink() {
        return Err(AgentControlSocketBindError::ParentIsSymlink);
    }
    if !metadata.is_dir() {
        return Err(AgentControlSocketBindError::ParentIsNotDirectory);
    }
    let canonical = fs::canonicalize(parent)
        .map_err(|source| AgentControlSocketBindError::InspectParent { source })?;
    if canonical.as_os_str().as_bytes() != parent.as_os_str().as_bytes() {
        return Err(AgentControlSocketBindError::ParentIsNotCanonical);
    }
    let expected_uid = effective_uid();
    if metadata.uid() != expected_uid {
        return Err(AgentControlSocketBindError::ParentOwnerMismatch {
            expected_uid,
            actual_uid: metadata.uid(),
        });
    }
    let mode = metadata.mode() & 0o777;
    if mode != 0o700 {
        return Err(AgentControlSocketBindError::ParentModeNotPrivate { actual_mode: mode });
    }
    Ok(())
}

#[allow(unsafe_code, clippy::unnecessary_cast)]
fn socket_fd_properties(listener: &UnixListener) -> io::Result<SocketDescriptorProperties> {
    let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
    // SAFETY: `stat` points to writable storage for one `libc::stat`, and the
    // listener owns a valid file descriptor for the duration of this call.
    let result = unsafe { libc::fstat(listener.as_raw_fd(), stat.as_mut_ptr()) };
    if result != 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: successful `fstat` initialized the complete structure.
    let stat = unsafe { stat.assume_init() };
    let mode = stat.st_mode as u32;
    Ok(SocketDescriptorProperties {
        owner_uid: stat.st_uid,
        is_socket: mode & libc::S_IFMT as u32 == libc::S_IFSOCK as u32,
    })
}

fn path_socket_identity(path: &Path) -> io::Result<SocketIdentity> {
    let metadata = fs::symlink_metadata(path)?;
    Ok(SocketIdentity {
        device: metadata.dev(),
        inode: metadata.ino(),
        owner_uid: metadata.uid(),
        permission_bits: metadata.mode() & 0o777,
        is_socket: metadata.file_type().is_socket(),
    })
}

fn set_socket_path_mode(path: &Path, mode: u32) -> io::Result<()> {
    fs::set_permissions(path, fs::Permissions::from_mode(mode))
}

fn parse_rejection(source: &AgentControlRequestParseError) -> AgentControlResponseV1 {
    let (code, retryable) = match source {
        AgentControlRequestParseError::UnsupportedSchemaVersion { .. } => {
            (AgentControlRejectionCodeV1::UnsupportedSchema, false)
        }
        AgentControlRequestParseError::DuplicateRequestId { .. }
        | AgentControlRequestParseError::RequestIdRegression { .. } => {
            (AgentControlRejectionCodeV1::RequestOrder, false)
        }
        AgentControlRequestParseError::EmptyInput
        | AgentControlRequestParseError::InputTooLarge { .. }
        | AgentControlRequestParseError::UnexpectedLeadingByte { .. }
        | AgentControlRequestParseError::UnexpectedTrailingByte { .. }
        | AgentControlRequestParseError::TrailingBytes { .. }
        | AgentControlRequestParseError::Json(_)
        | AgentControlRequestParseError::ZeroRequestId
        | AgentControlRequestParseError::NonFiniteManualVelocity { .. }
        | AgentControlRequestParseError::MapPoint { .. } => {
            (AgentControlRejectionCodeV1::MalformedRequest, false)
        }
    };
    AgentControlResponseV1::rejected(source.request_id(), code, retryable)
}

#[derive(Debug)]
enum ReadRequestFrameError {
    InvalidLength { declared_bytes: u32 },
    Io { source: io::Error, timeout: bool },
    ShutdownRequested,
}

fn read_request_frame_into<'buffer>(
    stream: &mut UnixStream,
    timeout: Duration,
    payload: &'buffer mut [u8; MAX_AGENT_CONTROL_REQUEST_JSON_BYTES],
    cancellation: Option<&AtomicBool>,
) -> Result<&'buffer [u8], ReadRequestFrameError> {
    let deadline =
        Instant::now()
            .checked_add(timeout)
            .ok_or_else(|| ReadRequestFrameError::Io {
                source: io::Error::new(io::ErrorKind::InvalidInput, "read deadline overflow"),
                timeout: false,
            })?;
    let mut length = [0_u8; FRAME_LENGTH_BYTES];
    read_exact_until_cancellable(stream, &mut length, deadline, cancellation)
        .map_err(map_request_read_error)?;
    let declared_wire_bytes = u32::from_be_bytes(length);
    let declared_bytes =
        usize::try_from(declared_wire_bytes).map_err(|_| ReadRequestFrameError::InvalidLength {
            declared_bytes: declared_wire_bytes,
        })?;
    if declared_bytes == 0 || declared_bytes > MAX_AGENT_CONTROL_REQUEST_JSON_BYTES {
        return Err(ReadRequestFrameError::InvalidLength {
            declared_bytes: declared_wire_bytes,
        });
    }
    read_exact_until_cancellable(
        stream,
        &mut payload[..declared_bytes],
        deadline,
        cancellation,
    )
    .map_err(map_request_read_error)?;
    Ok(&payload[..declared_bytes])
}

fn map_request_read_error(source: CancellableIoError) -> ReadRequestFrameError {
    match source {
        CancellableIoError::Io(source) => {
            let timeout = is_timeout_error(&source);
            ReadRequestFrameError::Io { source, timeout }
        }
        CancellableIoError::ShutdownRequested => ReadRequestFrameError::ShutdownRequested,
    }
}

fn write_response_frame(
    stream: &mut UnixStream,
    response: AgentControlResponseV1,
    timeout: Duration,
    cancellation: Option<&AtomicBool>,
) -> Result<(), AgentControlServeError> {
    let mut payload = FixedResponseBuffer::new();
    serde_json::to_writer(&mut payload, &response).map_err(|source| {
        if payload.overflowed {
            AgentControlServeError::ResponseTooLarge {
                maximum_bytes: MAX_AGENT_CONTROL_RESPONSE_JSON_BYTES,
            }
        } else {
            AgentControlServeError::ResponseSerialization { source }
        }
    })?;
    let length = u32::try_from(payload.len)
        .map_err(|_| AgentControlServeError::ResponseTooLarge {
            maximum_bytes: MAX_AGENT_CONTROL_RESPONSE_JSON_BYTES,
        })?
        .to_be_bytes();
    let deadline = Instant::now()
        .checked_add(timeout)
        .ok_or(AgentControlServeError::DeadlineOverflow)?;
    write_all_until_cancellable(stream, &length, deadline, cancellation)
        .map_err(map_response_write_error)?;
    write_all_until_cancellable(stream, payload.as_slice(), deadline, cancellation)
        .map_err(map_response_write_error)
}

fn map_response_write_error(source: CancellableIoError) -> AgentControlServeError {
    match source {
        CancellableIoError::Io(source) if is_timeout_error(&source) => {
            AgentControlServeError::ResponseDeadline
        }
        CancellableIoError::Io(source) => AgentControlServeError::ResponseWrite { source },
        CancellableIoError::ShutdownRequested => AgentControlServeError::ShutdownRequested,
    }
}

fn require_not_cancelled(cancellation: Option<&AtomicBool>) -> Result<(), AgentControlServeError> {
    if cancellation.is_some_and(|running| !running.load(Ordering::Acquire)) {
        Err(AgentControlServeError::ShutdownRequested)
    } else {
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CancellableReceiveError {
    Deadline,
    Disconnected,
    ShutdownRequested,
}

fn recv_until<T>(
    receiver: &Receiver<T>,
    deadline: Instant,
    cancellation: Option<&AtomicBool>,
) -> Result<T, CancellableReceiveError> {
    loop {
        if cancellation.is_some_and(|running| !running.load(Ordering::Acquire)) {
            return Err(CancellableReceiveError::ShutdownRequested);
        }
        let remaining = deadline
            .checked_duration_since(Instant::now())
            .filter(|remaining| !remaining.is_zero())
            .ok_or(CancellableReceiveError::Deadline)?;
        let wait = if cancellation.is_some() {
            remaining.min(AGENT_CONTROL_SOCKET_CANCELLATION_POLL)
        } else {
            remaining
        };
        match receiver.recv_timeout(wait) {
            Ok(value) => return Ok(value),
            Err(RecvTimeoutError::Disconnected) => {
                return Err(CancellableReceiveError::Disconnected);
            }
            Err(RecvTimeoutError::Timeout) if cancellation.is_some() => {}
            Err(RecvTimeoutError::Timeout) => return Err(CancellableReceiveError::Deadline),
        }
    }
}

#[derive(Debug)]
enum CancellableIoError {
    Io(io::Error),
    ShutdownRequested,
}

#[cfg(test)]
fn read_exact_until(
    stream: &mut UnixStream,
    destination: &mut [u8],
    deadline: Instant,
) -> io::Result<()> {
    match read_exact_until_cancellable(stream, destination, deadline, None) {
        Ok(()) => Ok(()),
        Err(CancellableIoError::Io(source)) => Err(source),
        Err(CancellableIoError::ShutdownRequested) => {
            unreachable!("uncancellable read cannot observe shutdown")
        }
    }
}

fn read_exact_until_cancellable(
    stream: &mut UnixStream,
    mut destination: &mut [u8],
    deadline: Instant,
    cancellation: Option<&AtomicBool>,
) -> Result<(), CancellableIoError> {
    while !destination.is_empty() {
        if cancellation.is_some_and(|running| !running.load(Ordering::Acquire)) {
            return Err(CancellableIoError::ShutdownRequested);
        }
        let remaining = deadline
            .checked_duration_since(Instant::now())
            .filter(|remaining| !remaining.is_zero())
            .ok_or_else(|| CancellableIoError::Io(deadline_io_error()))?;
        let wait = if cancellation.is_some() {
            remaining.min(AGENT_CONTROL_SOCKET_CANCELLATION_POLL)
        } else {
            remaining
        };
        stream
            .set_read_timeout(Some(wait))
            .map_err(CancellableIoError::Io)?;
        match stream.read(destination) {
            Ok(0) => {
                return Err(CancellableIoError::Io(io::Error::from(
                    io::ErrorKind::UnexpectedEof,
                )));
            }
            Ok(read) => destination = &mut destination[read..],
            Err(source) if source.kind() == io::ErrorKind::Interrupted => {}
            Err(source) if cancellation.is_some() && is_timeout_error(&source) => {}
            Err(source) => return Err(CancellableIoError::Io(source)),
        }
    }
    Ok(())
}

#[cfg(test)]
fn write_all_until(stream: &mut UnixStream, source: &[u8], deadline: Instant) -> io::Result<()> {
    match write_all_until_cancellable(stream, source, deadline, None) {
        Ok(()) => Ok(()),
        Err(CancellableIoError::Io(source)) => Err(source),
        Err(CancellableIoError::ShutdownRequested) => {
            unreachable!("uncancellable write cannot observe shutdown")
        }
    }
}

fn write_all_until_cancellable(
    stream: &mut UnixStream,
    mut source: &[u8],
    deadline: Instant,
    cancellation: Option<&AtomicBool>,
) -> Result<(), CancellableIoError> {
    while !source.is_empty() {
        if cancellation.is_some_and(|running| !running.load(Ordering::Acquire)) {
            return Err(CancellableIoError::ShutdownRequested);
        }
        let remaining = deadline
            .checked_duration_since(Instant::now())
            .filter(|remaining| !remaining.is_zero())
            .ok_or_else(|| CancellableIoError::Io(deadline_io_error()))?;
        let wait = if cancellation.is_some() {
            remaining.min(AGENT_CONTROL_SOCKET_CANCELLATION_POLL)
        } else {
            remaining
        };
        stream
            .set_write_timeout(Some(wait))
            .map_err(CancellableIoError::Io)?;
        match stream.write(source) {
            Ok(0) => {
                return Err(CancellableIoError::Io(io::Error::from(
                    io::ErrorKind::WriteZero,
                )));
            }
            Ok(written) => source = &source[written..],
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) if cancellation.is_some() && is_timeout_error(&error) => {}
            Err(error) => return Err(CancellableIoError::Io(error)),
        }
    }
    Ok(())
}

fn deadline_io_error() -> io::Error {
    io::Error::new(io::ErrorKind::TimedOut, "absolute socket deadline expired")
}

fn is_timeout_error(source: &io::Error) -> bool {
    matches!(
        source.kind(),
        io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock
    )
}

struct FixedResponseBuffer {
    bytes: [u8; MAX_AGENT_CONTROL_RESPONSE_JSON_BYTES],
    len: usize,
    overflowed: bool,
}

impl FixedResponseBuffer {
    const fn new() -> Self {
        Self {
            bytes: [0; MAX_AGENT_CONTROL_RESPONSE_JSON_BYTES],
            len: 0,
            overflowed: false,
        }
    }

    fn as_slice(&self) -> &[u8] {
        &self.bytes[..self.len]
    }
}

impl Write for FixedResponseBuffer {
    fn write(&mut self, source: &[u8]) -> io::Result<usize> {
        let remaining = self.bytes.len() - self.len;
        if source.len() > remaining {
            self.overflowed = true;
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "fixed response buffer exhausted",
            ));
        }
        self.bytes[self.len..self.len + source.len()].copy_from_slice(source);
        self.len += source.len();
        Ok(source.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::{PermissionsExt, symlink};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::thread;

    use serde_json::{Value, json};

    use super::*;
    use crate::navigation::{
        AgentBaseCommandStateV1, AgentControlCommandKindV1, AgentControlResponseKindV1,
        AgentMapStateV1, AgentRuntimeStateV1,
    };

    static NEXT_TEST_DIRECTORY: AtomicU64 = AtomicU64::new(1);

    struct TestDirectory {
        path: PathBuf,
    }

    impl TestDirectory {
        fn new(mode: u32) -> Self {
            let root = fs::canonicalize(std::env::temp_dir()).expect("canonical temp root");
            for _ in 0..100 {
                let serial = NEXT_TEST_DIRECTORY.fetch_add(1, Ordering::Relaxed);
                let path = root.join(format!("kc-{}-{serial}", std::process::id()));
                match fs::create_dir(&path) {
                    Ok(()) => {
                        fs::set_permissions(&path, fs::Permissions::from_mode(mode))
                            .expect("set test-directory mode");
                        return Self { path };
                    }
                    Err(source) if source.kind() == io::ErrorKind::AlreadyExists => {}
                    Err(source) => panic!("create test directory: {source}"),
                }
            }
            panic!("could not allocate unique test directory");
        }

        fn socket_path(&self) -> PathBuf {
            self.path.join("c.sock")
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    fn timeouts(runtime: Duration) -> AgentControlSocketTimeouts {
        AgentControlSocketTimeouts::try_new(
            Duration::from_millis(150),
            Duration::from_secs(2),
            runtime,
            Duration::from_secs(5),
        )
        .expect("test timeouts")
    }

    fn long_cancellable_timeouts() -> AgentControlSocketTimeouts {
        AgentControlSocketTimeouts::try_new(
            Duration::from_secs(30),
            Duration::from_secs(30),
            Duration::from_secs(30),
            Duration::from_secs(10 * 60),
        )
        .expect("maximum bounded cancellation test timeouts")
    }

    fn bind_server(
        directory: &TestDirectory,
        capacity: usize,
        runtime_timeout: Duration,
    ) -> (AgentControlSocketServer, AgentControlRuntimeReceiver) {
        let path = AgentControlSocketPath::parse(&directory.socket_path()).expect("socket path");
        let config = AgentControlSocketConfig::new(path, timeouts(runtime_timeout));
        let capacity = AgentControlRuntimeQueueCapacity::try_new(capacity).expect("queue capacity");
        let (sender, receiver) = agent_control_runtime_queue(capacity);
        let origin = AgentControlMonotonicOrigin::new(
            Instant::now(),
            HostMonotonicTimestamp::from_nanos(10_000),
        );
        let server = AgentControlSocketServer::bind(config, origin, sender).expect("bind server");
        (server, receiver)
    }

    fn request(id: u64, command: Value) -> Vec<u8> {
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "request_id": id,
            "command": command,
        }))
        .expect("serialize request")
    }

    #[cfg(all(
        feature = "agent-runtime",
        feature = "actuation",
        feature = "operator-console"
    ))]
    #[test]
    fn typed_ingress_converges_on_the_same_claimed_dispatch_without_reparsing() {
        let capacity = AgentControlRuntimeQueueCapacity::try_new(1).expect("queue capacity");
        let (sender, mut receiver) = agent_control_runtime_queue(capacity);
        let (priority, priority_receiver) = mpsc::sync_channel(capacity.get());
        receiver.priority = Some(priority_receiver);
        let mut ingress = AgentControlTypedIngress {
            normal: sender.inner.clone(),
            priority,
            next_key: NonZeroU64::new(1),
        };
        drop(sender);
        let parsed = AgentControlRequestParser::new()
            .parse_next(&request(41, json!({"kind":"stop"})))
            .expect("parse once at the test boundary");
        let submission = ingress
            .try_submit(parsed, HostMonotonicTimestamp::from_nanos(77))
            .expect("typed submission");
        assert_eq!(submission.request_id().get(), 41);
        assert_eq!(
            submission.try_take_claim(),
            Err(AgentControlTypedSubmissionPollError::Pending)
        );

        let dispatch = receiver.try_recv().expect("shared runtime dispatch");
        assert_eq!(dispatch.request(), parsed);
        assert_eq!(
            dispatch.received_at(),
            HostMonotonicTimestamp::from_nanos(77)
        );
        let claimed = dispatch.claim().expect("buffered local claim");
        submission.try_take_claim().expect("exact claim observed");
        claimed
            .respond_completed()
            .expect("buffered local response");
        let response = submission
            .try_take_response()
            .expect("exact runtime response");
        assert_eq!(response.request_id(), Some(parsed.request_id()));
        assert!(matches!(
            response.response(),
            AgentControlResponseKindV1::Accepted {
                command: AgentControlCommandKindV1::Stop,
                completion: AgentControlCompletionV1::Completed,
            }
        ));
    }

    #[cfg(all(
        feature = "agent-runtime",
        feature = "actuation",
        feature = "operator-console"
    ))]
    #[test]
    fn typed_stop_has_a_reserved_priority_lane_and_distinct_correlation_namespace() {
        let capacity = AgentControlRuntimeQueueCapacity::try_new(1).expect("queue capacity");
        let (sender, mut receiver) = agent_control_runtime_queue(capacity);
        let (priority, priority_receiver) = mpsc::sync_channel(capacity.get());
        receiver.priority = Some(priority_receiver);
        let mut ingress = AgentControlTypedIngress {
            normal: sender.inner.clone(),
            priority,
            next_key: NonZeroU64::new(1),
        };

        let socket_request = AgentControlRequestParser::new()
            .parse_next(&request(1, json!({"kind":"arm"})))
            .expect("socket request");
        let (claim, _claim_rx) = mpsc::sync_channel(0);
        let (response, _response_rx) = mpsc::sync_channel(0);
        let (_delivery, delivery_rx) = mpsc::sync_channel(0);
        sender
            .inner
            .try_send(AgentControlDispatch {
                request: socket_request,
                received_at: HostMonotonicTimestamp::from_nanos(10),
                #[cfg(feature = "operator-console")]
                typed_request_key: None,
                terminal_response_deadline: None,
                claim,
                response,
                wire_delivery: delivery_rx,
            })
            .expect("normal lane filled");

        // A separate boundary may legitimately reuse the same wire request ID.
        // The process-local typed key—not that caller number—correlates its
        // eventual physical evidence.
        let console_request = AgentControlRequestParser::new()
            .parse_next(&request(1, json!({"kind":"stop"})))
            .expect("typed request");
        let submission = ingress
            .try_submit(console_request, HostMonotonicTimestamp::from_nanos(20))
            .expect("reserved priority lane accepts stop");

        let priority_dispatch = receiver.try_recv().expect("priority dispatch first");
        assert!(matches!(
            priority_dispatch.request().command(),
            AgentControlCommandV1::Stop
        ));
        assert_eq!(
            priority_dispatch.typed_request_key(),
            Some(submission.typed_request_key())
        );
        assert_eq!(
            priority_dispatch.request().request_id(),
            socket_request.request_id()
        );

        let normal_dispatch = receiver.try_recv().expect("normal dispatch retained");
        assert!(matches!(
            normal_dispatch.request().command(),
            AgentControlCommandV1::Arm
        ));
        assert_eq!(normal_dispatch.typed_request_key(), None);
    }

    fn write_request(stream: &mut UnixStream, payload: &[u8]) {
        let length = u32::try_from(payload.len()).expect("small test request");
        stream
            .write_all(&length.to_be_bytes())
            .expect("write frame length");
        stream.write_all(payload).expect("write frame payload");
    }

    fn read_response(stream: &mut UnixStream) -> (Value, usize) {
        stream
            .set_read_timeout(Some(Duration::from_secs(3)))
            .expect("client read timeout");
        let mut length = [0_u8; FRAME_LENGTH_BYTES];
        stream
            .read_exact(&mut length)
            .expect("read response length");
        let length = usize::try_from(u32::from_be_bytes(length)).expect("response length");
        assert!(length <= MAX_AGENT_CONTROL_RESPONSE_JSON_BYTES);
        let mut payload = vec![0_u8; length];
        stream
            .read_exact(&mut payload)
            .expect("read response payload");
        (
            serde_json::from_slice(&payload).expect("response JSON"),
            length,
        )
    }

    fn round_trip(path: &Path, payload: &[u8]) -> (Value, usize) {
        let mut stream = UnixStream::connect(path).expect("connect client");
        write_request(&mut stream, payload);
        read_response(&mut stream)
    }

    fn poll_until_connection(
        server: &mut AgentControlSocketServer,
    ) -> Result<AgentControlServeOutcome, AgentControlServeError> {
        let deadline = Instant::now() + Duration::from_secs(3);
        loop {
            match server.poll_one()? {
                AgentControlServeOutcome::Idle if Instant::now() < deadline => {
                    thread::sleep(Duration::from_millis(1));
                }
                AgentControlServeOutcome::Idle => panic!("test accept deadline"),
                outcome @ AgentControlServeOutcome::Responded { .. } => return Ok(outcome),
            }
        }
    }

    #[test]
    fn path_and_timeout_parsing_are_exact_and_bounded() {
        assert!(matches!(
            AgentControlSocketPath::parse(Path::new("relative.sock")),
            Err(AgentControlSocketPathError::NotAbsolute)
        ));
        for raw in ["/tmp/./c.sock", "/tmp/a/../c.sock", "/tmp/c.sock/"] {
            assert!(matches!(
                AgentControlSocketPath::parse(Path::new(raw)),
                Err(AgentControlSocketPathError::NonCanonicalComponent)
            ));
        }
        let nul = Path::new(OsStr::from_bytes(b"/tmp/c\0.sock"));
        assert!(matches!(
            AgentControlSocketPath::parse(nul),
            Err(AgentControlSocketPathError::InteriorNul)
        ));
        let long = PathBuf::from(format!(
            "/{}",
            "x".repeat(MAX_AGENT_CONTROL_SOCKET_PATH_BYTES)
        ));
        assert!(matches!(
            AgentControlSocketPath::parse(&long),
            Err(AgentControlSocketPathError::TooLong { .. })
        ));
        assert!(
            AgentControlSocketTimeouts::try_new(
                Duration::ZERO,
                Duration::from_secs(1),
                Duration::from_secs(1),
                Duration::from_secs(1),
            )
            .is_err()
        );
        assert!(
            AgentControlSocketTimeouts::try_new(
                Duration::from_secs(1),
                Duration::from_secs(31),
                Duration::from_secs(1),
                Duration::from_secs(1),
            )
            .is_err()
        );
        assert!(
            AgentControlSocketTimeouts::try_new(
                Duration::from_secs(1),
                Duration::from_secs(1),
                Duration::from_secs(1),
                Duration::from_secs(3),
            )
            .is_err()
        );
        let parsed_timeouts = timeouts(Duration::from_millis(40));
        assert_eq!(
            parsed_timeouts.response_timeout_for(AgentControlCommandV1::Arm),
            Duration::from_millis(40)
        );
        assert_eq!(
            parsed_timeouts.response_timeout_for(AgentControlCommandV1::SaveMap),
            Duration::from_secs(5),
            "terminal SaveMap must not inherit the ordinary runtime deadline"
        );
        assert!(matches!(
            AgentControlRuntimeQueueCapacity::try_new(0),
            Err(AgentControlRuntimeQueueCapacityError::Zero)
        ));
        assert!(matches!(
            AgentControlRuntimeQueueCapacity::try_new(MAX_AGENT_CONTROL_RUNTIME_QUEUE_CAPACITY + 1),
            Err(AgentControlRuntimeQueueCapacityError::TooLarge { .. })
        ));
        let future = AgentControlMonotonicOrigin::new(
            Instant::now() + Duration::from_secs(1),
            HostMonotonicTimestamp::from_nanos(0),
        );
        assert!(matches!(
            future.try_now(),
            Err(AgentControlClockError::OriginInFuture { .. })
        ));
    }

    #[test]
    fn bind_requires_private_canonical_parent_and_preserves_existing_path() {
        let public = TestDirectory::new(0o755);
        let path = AgentControlSocketPath::parse(&public.socket_path()).expect("absolute path");
        let (sender, _) = agent_control_runtime_queue(
            AgentControlRuntimeQueueCapacity::try_new(1).expect("capacity"),
        );
        let error = AgentControlSocketServer::bind(
            AgentControlSocketConfig::new(path, timeouts(Duration::from_secs(1))),
            AgentControlMonotonicOrigin::new(Instant::now(), HostMonotonicTimestamp::from_nanos(0)),
            sender,
        )
        .err()
        .expect("public directory rejected");
        assert!(matches!(
            error,
            AgentControlSocketBindError::ParentModeNotPrivate { actual_mode: 0o755 }
        ));

        let private = TestDirectory::new(0o700);
        let existing = private.socket_path();
        fs::write(&existing, b"keep").expect("existing sentinel");
        let path = AgentControlSocketPath::parse(&existing).expect("path");
        let (sender, _) = agent_control_runtime_queue(
            AgentControlRuntimeQueueCapacity::try_new(1).expect("capacity"),
        );
        assert!(matches!(
            AgentControlSocketServer::bind(
                AgentControlSocketConfig::new(path, timeouts(Duration::from_secs(1))),
                AgentControlMonotonicOrigin::new(
                    Instant::now(),
                    HostMonotonicTimestamp::from_nanos(0)
                ),
                sender,
            ),
            Err(AgentControlSocketBindError::DestinationExists)
        ));
        assert_eq!(fs::read(existing).expect("sentinel remains"), b"keep");

        let symlink_root = TestDirectory::new(0o700);
        let link = symlink_root.path.join("linked");
        symlink(&private.path, &link).expect("directory symlink");
        let linked_path = AgentControlSocketPath::parse(&link.join("c.sock")).expect("link path");
        let (sender, _) = agent_control_runtime_queue(
            AgentControlRuntimeQueueCapacity::try_new(1).expect("capacity"),
        );
        assert!(matches!(
            AgentControlSocketServer::bind(
                AgentControlSocketConfig::new(linked_path, timeouts(Duration::from_secs(1))),
                AgentControlMonotonicOrigin::new(
                    Instant::now(),
                    HostMonotonicTimestamp::from_nanos(0)
                ),
                sender,
            ),
            Err(AgentControlSocketBindError::ParentIsSymlink)
        ));
    }

    #[test]
    fn shutdown_completion_waits_for_a_complete_framed_wire_write() {
        let directory = TestDirectory::new(0o700);
        let socket_path = directory.socket_path();
        let (server, receiver) = bind_server(&directory, 2, Duration::from_secs(1));
        let metadata = fs::symlink_metadata(&socket_path).expect("socket metadata");
        assert!(metadata.file_type().is_socket());
        assert_eq!(metadata.permissions().mode() & 0o777, 0o600);

        let runtime = thread::spawn(move || {
            let dispatch = receiver
                .recv_timeout(Duration::from_secs(2))
                .expect("runtime request");
            assert_eq!(dispatch.request().request_id().get(), 1);
            assert!(dispatch.received_at().as_nanos() >= 10_000);
            let claimed = dispatch.claim().expect("claim rendezvous");
            assert!(matches!(
                claimed.request().command(),
                AgentControlCommandV1::Shutdown
            ));
            claimed
                .respond_completed_after_wire_delivery()
                .expect("wire-delivered completion");
        });
        let server_thread = thread::spawn(move || {
            let mut server = server;
            let outcome = poll_until_connection(&mut server);
            (server, outcome)
        });
        let (response, encoded_bytes) =
            round_trip(&socket_path, &request(1, json!({"kind":"shutdown"})));
        assert!(encoded_bytes <= MAX_AGENT_CONTROL_RESPONSE_JSON_BYTES);
        assert_eq!(response["request_id"], 1);
        assert_eq!(response["response"]["kind"], "accepted");
        assert_eq!(response["response"]["command"], "shutdown");
        assert_eq!(response["response"]["completion"], "completed");
        runtime.join().expect("runtime thread");
        let (server, outcome) = server_thread.join().expect("server thread");
        assert!(matches!(
            outcome.expect("served"),
            AgentControlServeOutcome::Responded {
                connection_issue: None,
                ..
            }
        ));
        assert_eq!(
            server.last_request_id().map(AgentControlRequestId::get),
            Some(1)
        );
        assert!(matches!(
            server.shutdown(),
            AgentControlSocketCleanupOutcome::RemovedCreatedSocket
        ));
        assert!(!socket_path.exists());
    }

    #[test]
    fn shutdown_completion_reports_wire_delivery_uncertainty_after_response_rendezvous() {
        let (claim_sender, claim_receiver) = mpsc::sync_channel(0);
        let (response_sender, response_receiver) = mpsc::sync_channel(0);
        let (wire_delivery_sender, wire_delivery_receiver) = mpsc::sync_channel(1);
        let request = AgentControlRequestParser::new()
            .parse_next(&request(1, json!({"kind":"shutdown"})))
            .expect("parsed shutdown");
        let dispatch = AgentControlDispatch {
            request,
            received_at: HostMonotonicTimestamp::from_nanos(10_000),
            #[cfg(feature = "operator-console")]
            typed_request_key: None,
            terminal_response_deadline: None,
            claim: claim_sender,
            response: response_sender,
            wire_delivery: wire_delivery_receiver,
        };
        let peer = thread::spawn(move || {
            claim_receiver.recv().expect("claim");
            let response = response_receiver.recv().expect("runtime response");
            drop(wire_delivery_sender);
            response
        });

        let error = dispatch
            .claim()
            .expect("claimed shutdown")
            .respond_completed_after_wire_delivery()
            .expect_err("a dropped write acknowledgement is not completion");
        assert_eq!(
            error,
            AgentControlDispatchResponseError::WireDeliveryUncertain
        );
        let response = peer.join().expect("delivery peer");
        assert_eq!(
            response.response(),
            AgentControlResponseKindV1::Accepted {
                command: AgentControlCommandKindV1::Shutdown,
                completion: AgentControlCompletionV1::Completed,
            }
        );
    }

    #[test]
    fn terminal_rejection_requires_wire_delivery_acknowledgement() {
        let (claim_sender, claim_receiver) = mpsc::sync_channel(0);
        let (response_sender, response_receiver) = mpsc::sync_channel(0);
        let (wire_delivery_sender, wire_delivery_receiver) = mpsc::sync_channel(1);
        let request = AgentControlRequestParser::new()
            .parse_next(&request(1, json!({"kind":"save_map"})))
            .expect("parsed save-map request");
        let terminal_response_deadline = Instant::now() + Duration::from_secs(5);
        let dispatch = AgentControlDispatch {
            request,
            received_at: HostMonotonicTimestamp::from_nanos(10_000),
            #[cfg(feature = "operator-console")]
            typed_request_key: None,
            terminal_response_deadline: Some(terminal_response_deadline),
            claim: claim_sender,
            response: response_sender,
            wire_delivery: wire_delivery_receiver,
        };
        let runtime = thread::spawn(move || {
            let claimed = dispatch.claim().expect("claimed save-map request");
            assert_eq!(
                claimed.terminal_response_deadline(),
                Some(terminal_response_deadline)
            );
            claimed
                .reject_after_wire_delivery(AgentControlRejectionCodeV1::PersistenceFailed, false)
        });
        claim_receiver.recv().expect("claim");
        let response = response_receiver.recv().expect("runtime response");
        assert!(matches!(
            response.response(),
            AgentControlResponseKindV1::Rejected {
                code: AgentControlRejectionCodeV1::PersistenceFailed,
                retryable: false,
            }
        ));
        assert!(!runtime.is_finished());
        wire_delivery_sender
            .send(())
            .expect("wire-delivery evidence");
        runtime
            .join()
            .expect("runtime thread")
            .expect("wire-delivered rejection");
    }

    #[test]
    fn status_response_uses_claim_then_one_final_wire_response() {
        let directory = TestDirectory::new(0o700);
        let socket_path = directory.socket_path();
        let (server, receiver) = bind_server(&directory, 1, Duration::from_secs(1));
        let runtime = thread::spawn(move || {
            let dispatch = receiver.recv().expect("status dispatch");
            let claimed = dispatch.claim().expect("claim status");
            claimed
                .respond_status(AgentControlStatusV1::new(
                    AgentRuntimeStateV1::ReadyStopped,
                    AgentBaseCommandStateV1::ConfirmedStopped,
                    AgentMapStateV1::UNAVAILABLE,
                ))
                .expect("status response");
        });
        let server_thread = thread::spawn(move || {
            let mut server = server;
            let outcome = poll_until_connection(&mut server);
            (server, outcome)
        });
        let (response, _) = round_trip(&socket_path, &request(1, json!({"kind":"query_status"})));
        assert_eq!(response["response"]["kind"], "status");
        assert_eq!(
            response["response"]["status"]["runtime"]["kind"],
            "ready_stopped"
        );
        runtime.join().expect("runtime");
        let (server, outcome) = server_thread.join().expect("server");
        outcome.expect("status served");
        drop(server);
        assert!(!socket_path.exists());
    }

    #[test]
    fn request_order_is_global_across_connections_and_never_redispatched() {
        let directory = TestDirectory::new(0o700);
        let socket_path = directory.socket_path();
        let (server, receiver) = bind_server(&directory, 2, Duration::from_secs(1));
        let runtime = thread::spawn(move || {
            let dispatch = receiver.recv().expect("first request");
            assert_eq!(dispatch.request().request_id().get(), 5);
            let accepted = dispatch
                .claim()
                .expect("claim first")
                .respond_accepted_for_processing()
                .expect("accept first");
            assert!(matches!(
                accepted.request().command(),
                AgentControlCommandV1::Stop
            ));
            receiver
        });
        let server_thread = thread::spawn(move || {
            let mut server = server;
            let outcomes = (0..3)
                .map(|_| poll_until_connection(&mut server))
                .collect::<Vec<_>>();
            (server, outcomes)
        });

        let (first, _) = round_trip(&socket_path, &request(5, json!({"kind":"stop"})));
        let (duplicate, _) = round_trip(&socket_path, &request(5, json!({"kind":"shutdown"})));
        let (regression, _) = round_trip(&socket_path, &request(4, json!({"kind":"shutdown"})));
        assert_eq!(first["response"]["completion"], "accepted_for_processing");
        assert_eq!(duplicate["response"]["code"], "request_order");
        assert_eq!(regression["response"]["code"], "request_order");

        let receiver = runtime.join().expect("runtime");
        assert!(matches!(
            receiver.try_recv(),
            Err(AgentControlRuntimeReceiveError::Empty)
        ));
        let (server, outcomes) = server_thread.join().expect("server");
        assert!(outcomes.into_iter().all(|outcome| outcome.is_ok()));
        assert_eq!(
            server.last_request_id().map(AgentControlRequestId::get),
            Some(5)
        );
        drop(server);
    }

    #[test]
    fn queue_backpressure_and_unclaimed_timeout_are_retryable_rejections() {
        let directory = TestDirectory::new(0o700);
        let socket_path = directory.socket_path();
        let (server, receiver) = bind_server(&directory, 1, Duration::from_millis(60));
        let server_thread = thread::spawn(move || {
            let mut server = server;
            let first = poll_until_connection(&mut server);
            let second = poll_until_connection(&mut server);
            (server, first, second)
        });

        let (unclaimed, _) = round_trip(&socket_path, &request(1, json!({"kind":"stop"})));
        let (full, _) = round_trip(&socket_path, &request(2, json!({"kind":"shutdown"})));
        for response in [&unclaimed, &full] {
            assert_eq!(response["response"]["kind"], "rejected");
            assert_eq!(response["response"]["code"], "not_ready");
            assert_eq!(response["response"]["retryable"], true);
        }

        let (server, first, second) = server_thread.join().expect("server");
        first.expect("unclaimed response sent");
        second.expect("backpressure response sent");
        let expired = receiver.try_recv().expect("first dispatch remains bounded");
        assert!(matches!(
            expired.claim(),
            Err(AgentControlDispatchResponseError::ClientUnavailable)
        ));
        assert!(matches!(
            receiver.try_recv(),
            Err(AgentControlRuntimeReceiveError::Empty)
        ));
        assert_eq!(
            server.last_request_id().map(AgentControlRequestId::get),
            Some(2)
        );
        drop(server);
    }

    #[test]
    fn post_claim_timeout_sends_no_false_completion_and_token_expires() {
        let directory = TestDirectory::new(0o700);
        let socket_path = directory.socket_path();
        let (server, receiver) = bind_server(&directory, 1, Duration::from_millis(80));
        let (claimed_sender, claimed_receiver) = mpsc::sync_channel(0);
        let runtime = thread::spawn(move || {
            let dispatch = receiver.recv().expect("dispatch");
            let claimed = dispatch.claim().expect("claim before deadline");
            claimed_sender.send(()).expect("claimed evidence");
            thread::sleep(Duration::from_millis(160));
            claimed.respond_completed()
        });
        let server_thread = thread::spawn(move || {
            let mut server = server;
            let outcome = poll_until_connection(&mut server);
            (server, outcome)
        });

        let mut stream = UnixStream::connect(&socket_path).expect("connect");
        write_request(&mut stream, &request(1, json!({"kind":"stop"})));
        claimed_receiver
            .recv_timeout(Duration::from_secs(1))
            .expect("runtime claimed");
        stream
            .set_read_timeout(Some(Duration::from_secs(1)))
            .expect("client timeout");
        let mut header = [0_u8; FRAME_LENGTH_BYTES];
        let read_error = stream
            .read_exact(&mut header)
            .expect_err("server must close without a false completion");
        assert!(matches!(
            read_error.kind(),
            io::ErrorKind::UnexpectedEof | io::ErrorKind::ConnectionReset
        ));
        assert!(matches!(
            runtime.join().expect("runtime"),
            Err(AgentControlDispatchResponseError::ClientUnavailable)
        ));
        let (server, outcome) = server_thread.join().expect("server");
        assert!(matches!(
            outcome,
            Err(AgentControlServeError::RuntimeResponseDeadlineAfterClaim { request_id })
                if request_id.get() == 1
        ));
        drop(server);
    }

    #[test]
    fn partial_read_timeout_and_oversized_frame_receive_fixed_rejections() {
        let directory = TestDirectory::new(0o700);
        let socket_path = directory.socket_path();
        let (server, receiver) = bind_server(&directory, 1, Duration::from_secs(1));
        let server_thread = thread::spawn(move || {
            let mut server = server;
            let first = poll_until_connection(&mut server);
            let second = poll_until_connection(&mut server);
            (server, first, second)
        });

        let mut partial = UnixStream::connect(&socket_path).expect("partial client");
        partial.write_all(&[0, 0]).expect("partial header");
        let (timed_out, _) = read_response(&mut partial);
        assert_eq!(timed_out["response"]["code"], "malformed_request");
        assert_eq!(timed_out["response"]["retryable"], true);

        let mut oversized = UnixStream::connect(&socket_path).expect("oversized client");
        let oversized_length =
            u32::try_from(MAX_AGENT_CONTROL_REQUEST_JSON_BYTES + 1).expect("small bound");
        oversized
            .write_all(&oversized_length.to_be_bytes())
            .expect("oversized header");
        let (rejected, _) = read_response(&mut oversized);
        assert_eq!(rejected["response"]["code"], "malformed_request");
        assert_eq!(rejected["response"]["retryable"], false);

        let (server, first, second) = server_thread.join().expect("server");
        assert!(matches!(
            first.expect("timeout response"),
            AgentControlServeOutcome::Responded {
                connection_issue: Some(AgentControlConnectionIssue::Read { timeout: true, .. }),
                ..
            }
        ));
        assert!(matches!(
            second.expect("length response"),
            AgentControlServeOutcome::Responded {
                connection_issue: Some(AgentControlConnectionIssue::InvalidFrameLength {
                    declared_bytes
                }),
                ..
            } if declared_bytes == oversized_length
        ));
        assert!(matches!(
            receiver.try_recv(),
            Err(AgentControlRuntimeReceiveError::Empty)
        ));
        drop(server);
    }

    #[test]
    fn malformed_schema_and_closed_runtime_have_fixed_non_dispatch_responses() {
        let directory = TestDirectory::new(0o700);
        let socket_path = directory.socket_path();
        let (server, receiver) = bind_server(&directory, 1, Duration::from_secs(1));
        drop(receiver);
        let server_thread = thread::spawn(move || {
            let mut server = server;
            let outcomes = (0..3)
                .map(|_| poll_until_connection(&mut server))
                .collect::<Vec<_>>();
            (server, outcomes)
        });

        let malformed =
            br#"{"schema_version":1,"request_id":1,"command":{"kind":"stop","extra":1}}"#;
        let (malformed_response, _) = round_trip(&socket_path, malformed);
        assert_eq!(malformed_response["response"]["code"], "malformed_request");

        let unsupported = br#"{"schema_version":2,"request_id":2,"command":{"kind":"stop"}}"#;
        let (unsupported_response, _) = round_trip(&socket_path, unsupported);
        assert_eq!(
            unsupported_response["response"]["code"],
            "unsupported_schema"
        );

        let (closed_response, _) = round_trip(&socket_path, &request(3, json!({"kind":"stop"})));
        assert_eq!(closed_response["response"]["code"], "shutdown_in_progress");
        assert_eq!(closed_response["response"]["retryable"], false);

        let (server, outcomes) = server_thread.join().expect("server");
        assert!(outcomes.into_iter().all(|outcome| outcome.is_ok()));
        assert_eq!(
            server.last_request_id().map(AgentControlRequestId::get),
            Some(3)
        );
        drop(server);
    }

    #[test]
    fn absolute_read_and_write_helpers_refuse_expired_deadlines() {
        let (mut first, mut second) = UnixStream::pair().expect("socket pair");
        let expired = Instant::now()
            .checked_sub(Duration::from_millis(1))
            .expect("past instant");
        let mut byte = [0_u8; 1];
        let read_error =
            read_exact_until(&mut first, &mut byte, expired).expect_err("expired read deadline");
        let write_error =
            write_all_until(&mut second, &[1], expired).expect_err("expired write deadline");
        assert!(is_timeout_error(&read_error));
        assert!(is_timeout_error(&write_error));
    }

    #[test]
    fn cancellable_write_observes_shutdown_while_peer_is_not_reading() {
        let (mut writer, _reader) = UnixStream::pair().expect("socket pair");
        writer
            .set_nonblocking(true)
            .expect("make writer nonblocking while filling its kernel buffer");
        let fill = [0_u8; 8 * 1_024];
        loop {
            match writer.write(&fill) {
                Ok(0) => panic!("socket write unexpectedly returned zero"),
                Ok(_) => {}
                Err(source) if source.kind() == io::ErrorKind::WouldBlock => break,
                Err(source) => panic!("fill socket write buffer: {source}"),
            }
        }
        writer
            .set_nonblocking(false)
            .expect("restore blocking writer");

        let running = Arc::new(AtomicBool::new(true));
        let writer_running = Arc::clone(&running);
        let write_thread = thread::spawn(move || {
            let deadline = Instant::now()
                .checked_add(Duration::from_secs(30))
                .expect("bounded future deadline");
            write_all_until_cancellable(&mut writer, &[1], deadline, Some(&writer_running))
        });
        thread::sleep(Duration::from_millis(25));
        assert!(
            !write_thread.is_finished(),
            "the unread peer must keep the write in flight"
        );

        let started = Instant::now();
        running.store(false, Ordering::Release);
        assert!(matches!(
            write_thread.join().expect("join cancellable writer"),
            Err(CancellableIoError::ShutdownRequested)
        ));
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "write cancellation must poll shutdown instead of the 30-second deadline"
        );
    }

    #[test]
    fn idle_poll_is_bounded_and_shutdown_removes_only_created_socket() {
        let directory = TestDirectory::new(0o700);
        let socket_path = directory.socket_path();
        let (mut server, _receiver) = bind_server(&directory, 1, Duration::from_secs(1));
        let started = Instant::now();
        assert!(matches!(
            server.poll_one().expect("idle poll"),
            AgentControlServeOutcome::Idle
        ));
        assert!(started.elapsed() < Duration::from_millis(100));
        assert!(matches!(
            server.shutdown(),
            AgentControlSocketCleanupOutcome::RemovedCreatedSocket
        ));
        assert!(!socket_path.exists());
    }

    #[test]
    fn task_thread_creation_failure_is_typed_fail_closed_and_cleans_socket() {
        let directory = TestDirectory::new(0o700);
        let socket_path = directory.socket_path();
        let path = AgentControlSocketPath::parse(&socket_path).expect("socket path");
        let config = AgentControlSocketConfig::new(path, timeouts(Duration::from_millis(100)));
        let capacity = AgentControlRuntimeQueueCapacity::try_new(1).expect("queue capacity");
        let running = Arc::new(AtomicBool::new(true));

        let result = AgentControlSocketTask::bind_and_spawn_with(
            config,
            AgentControlMonotonicOrigin::new(Instant::now(), HostMonotonicTimestamp::from_nanos(0)),
            capacity,
            Arc::clone(&running),
            |_task_main| {
                Err(io::Error::new(
                    io::ErrorKind::ResourceBusy,
                    "forced thread creation failure",
                ))
            },
        );
        let error = match result {
            Ok(_) => panic!("forced thread creation must fail"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            AgentControlSocketTaskStartError::ThreadSpawn {
                source,
                cleanup: AgentControlSocketCleanupOutcome::RemovedCreatedSocket,
            } if source.kind() == io::ErrorKind::ResourceBusy
        ));
        assert!(!running.load(Ordering::Acquire));
        assert!(!socket_path.exists());
    }

    #[test]
    fn task_start_rejects_an_already_cleared_run_flag_before_binding() {
        let directory = TestDirectory::new(0o700);
        let socket_path = directory.socket_path();
        let path = AgentControlSocketPath::parse(&socket_path).expect("socket path");
        let config = AgentControlSocketConfig::new(path, long_cancellable_timeouts());
        let capacity = AgentControlRuntimeQueueCapacity::try_new(1).expect("queue capacity");
        let running = Arc::new(AtomicBool::new(false));

        assert!(matches!(
            AgentControlSocketTask::bind_and_spawn(
                config,
                AgentControlMonotonicOrigin::new(
                    Instant::now(),
                    HostMonotonicTimestamp::from_nanos(0),
                ),
                capacity,
                Arc::clone(&running),
            ),
            Err(AgentControlSocketTaskStartError::ShutdownAlreadyRequested)
        ));
        assert!(!running.load(Ordering::Acquire));
        assert!(!socket_path.exists());
    }

    #[test]
    fn task_start_waits_until_the_child_has_taken_ownership_and_reported_ready() {
        let directory = TestDirectory::new(0o700);
        let socket_path = directory.socket_path();
        let path = AgentControlSocketPath::parse(&socket_path).expect("socket path");
        let config = AgentControlSocketConfig::new(path, long_cancellable_timeouts());
        let capacity = AgentControlRuntimeQueueCapacity::try_new(1).expect("queue capacity");
        let running = Arc::new(AtomicBool::new(true));
        let task_running = Arc::clone(&running);
        let (child_entered_sender, child_entered_receiver) = mpsc::sync_channel(0);
        let (release_sender, release_receiver) = mpsc::sync_channel(0);

        let starter = thread::spawn(move || {
            AgentControlSocketTask::bind_and_spawn_with(
                config,
                AgentControlMonotonicOrigin::new(
                    Instant::now(),
                    HostMonotonicTimestamp::from_nanos(0),
                ),
                capacity,
                task_running,
                move |task_main| {
                    thread::Builder::new().spawn(move || {
                        child_entered_sender
                            .send(())
                            .expect("startup observer remains live");
                        release_receiver
                            .recv()
                            .expect("test releases child startup");
                        task_main()
                    })
                },
            )
        });

        child_entered_receiver
            .recv_timeout(Duration::from_secs(1))
            .expect("child thread entered");
        assert!(
            !starter.is_finished(),
            "bind_and_spawn must wait for the child readiness handshake"
        );
        release_sender.send(()).expect("release child startup");
        let (task, receiver) = starter
            .join()
            .expect("join startup caller")
            .expect("child reports ready");
        assert!(socket_path.exists());
        assert!(!task.handle.as_ref().expect("live handle").is_finished());
        assert!(matches!(
            task.shutdown().expect("shutdown ready task"),
            AgentControlSocketTaskExit::Shutdown {
                cleanup: AgentControlSocketCleanupOutcome::RemovedCreatedSocket,
            }
        ));
        assert!(matches!(
            receiver.recv_timeout(Duration::from_millis(50)),
            Err(AgentControlRuntimeReceiveError::Disconnected)
        ));
        assert!(!running.load(Ordering::Acquire));
    }

    #[test]
    fn explicit_task_shutdown_joins_and_returns_cleanup_evidence() {
        let directory = TestDirectory::new(0o700);
        let socket_path = directory.socket_path();
        let path = AgentControlSocketPath::parse(&socket_path).expect("socket path");
        let config = AgentControlSocketConfig::new(path, timeouts(Duration::from_millis(100)));
        let capacity = AgentControlRuntimeQueueCapacity::try_new(1).expect("queue capacity");
        let running = Arc::new(AtomicBool::new(true));
        let (task, receiver) = AgentControlSocketTask::bind_and_spawn(
            config,
            AgentControlMonotonicOrigin::new(Instant::now(), HostMonotonicTimestamp::from_nanos(0)),
            capacity,
            Arc::clone(&running),
        )
        .expect("spawn socket task");

        assert!(socket_path.exists());
        assert!(matches!(
            task.shutdown().expect("join socket task"),
            AgentControlSocketTaskExit::Shutdown {
                cleanup: AgentControlSocketCleanupOutcome::RemovedCreatedSocket,
            }
        ));
        assert!(!running.load(Ordering::Acquire));
        assert!(!socket_path.exists());
        assert!(matches!(
            receiver.recv_timeout(Duration::from_millis(50)),
            Err(AgentControlRuntimeReceiveError::Disconnected)
        ));
    }

    #[test]
    fn public_join_initiates_shutdown_instead_of_waiting_forever() {
        let directory = TestDirectory::new(0o700);
        let socket_path = directory.socket_path();
        let path = AgentControlSocketPath::parse(&socket_path).expect("socket path");
        let config = AgentControlSocketConfig::new(path, long_cancellable_timeouts());
        let capacity = AgentControlRuntimeQueueCapacity::try_new(1).expect("queue capacity");
        let running = Arc::new(AtomicBool::new(true));
        let (task, _receiver) = AgentControlSocketTask::bind_and_spawn(
            config,
            AgentControlMonotonicOrigin::new(Instant::now(), HostMonotonicTimestamp::from_nanos(0)),
            capacity,
            Arc::clone(&running),
        )
        .expect("spawn socket task");

        let started = Instant::now();
        assert!(matches!(
            task.join().expect("stop and join socket task"),
            AgentControlSocketTaskExit::Shutdown {
                cleanup: AgentControlSocketCleanupOutcome::RemovedCreatedSocket,
            }
        ));
        assert!(started.elapsed() < Duration::from_secs(1));
        assert!(!running.load(Ordering::Acquire));
        assert!(!socket_path.exists());
    }

    #[test]
    fn shutdown_cancels_an_in_flight_partial_read_without_waiting_for_io_timeout() {
        let directory = TestDirectory::new(0o700);
        let socket_path = directory.socket_path();
        let path = AgentControlSocketPath::parse(&socket_path).expect("socket path");
        let config = AgentControlSocketConfig::new(path, long_cancellable_timeouts());
        let capacity = AgentControlRuntimeQueueCapacity::try_new(1).expect("queue capacity");
        let running = Arc::new(AtomicBool::new(true));
        let (task, _receiver) = AgentControlSocketTask::bind_and_spawn(
            config,
            AgentControlMonotonicOrigin::new(Instant::now(), HostMonotonicTimestamp::from_nanos(0)),
            capacity,
            Arc::clone(&running),
        )
        .expect("spawn socket task");
        let mut client = UnixStream::connect(&socket_path).expect("connect partial client");
        client.write_all(&[0]).expect("write partial frame header");
        thread::sleep(Duration::from_millis(25));

        let started = Instant::now();
        assert!(matches!(
            task.shutdown().expect("cancel partial read"),
            AgentControlSocketTaskExit::Shutdown {
                cleanup: AgentControlSocketCleanupOutcome::RemovedCreatedSocket,
            }
        ));
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "shutdown must poll cancellation instead of the 30-second read deadline"
        );
        assert!(!running.load(Ordering::Acquire));
        assert!(!socket_path.exists());
    }

    #[test]
    fn dropping_task_cancels_an_in_flight_read_without_detaching_or_waiting() {
        let directory = TestDirectory::new(0o700);
        let socket_path = directory.socket_path();
        let path = AgentControlSocketPath::parse(&socket_path).expect("socket path");
        let config = AgentControlSocketConfig::new(path, long_cancellable_timeouts());
        let capacity = AgentControlRuntimeQueueCapacity::try_new(1).expect("queue capacity");
        let running = Arc::new(AtomicBool::new(true));
        let (task, receiver) = AgentControlSocketTask::bind_and_spawn(
            config,
            AgentControlMonotonicOrigin::new(Instant::now(), HostMonotonicTimestamp::from_nanos(0)),
            capacity,
            Arc::clone(&running),
        )
        .expect("spawn socket task");
        let mut client = UnixStream::connect(&socket_path).expect("connect partial client");
        client.write_all(&[0]).expect("write partial frame header");
        thread::sleep(Duration::from_millis(25));

        let started = Instant::now();
        drop(task);
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "Drop must join through cooperative cancellation, not the 30-second read deadline"
        );
        assert!(!running.load(Ordering::Acquire));
        assert!(!socket_path.exists());
        assert!(matches!(
            receiver.recv_timeout(Duration::from_millis(50)),
            Err(AgentControlRuntimeReceiveError::Disconnected)
        ));
    }

    #[test]
    fn shutdown_while_runtime_holds_claim_does_not_wait_for_the_claimed_response() {
        let directory = TestDirectory::new(0o700);
        let socket_path = directory.socket_path();
        let path = AgentControlSocketPath::parse(&socket_path).expect("socket path");
        let config = AgentControlSocketConfig::new(path, long_cancellable_timeouts());
        let capacity = AgentControlRuntimeQueueCapacity::try_new(1).expect("queue capacity");
        let running = Arc::new(AtomicBool::new(true));
        let (task, receiver) = AgentControlSocketTask::bind_and_spawn(
            config,
            AgentControlMonotonicOrigin::new(Instant::now(), HostMonotonicTimestamp::from_nanos(0)),
            capacity,
            Arc::clone(&running),
        )
        .expect("spawn socket task");
        let mut client = UnixStream::connect(&socket_path).expect("connect control client");
        write_request(&mut client, &request(1, json!({"kind": "query_status"})));
        let dispatch = receiver
            .recv_timeout(Duration::from_secs(1))
            .expect("runtime receives request");
        let claimed = dispatch.claim().expect("claim rendezvous");

        let started = Instant::now();
        assert!(matches!(
            task.shutdown().expect("cancel claimed response wait"),
            AgentControlSocketTaskExit::Shutdown {
                cleanup: AgentControlSocketCleanupOutcome::RemovedCreatedSocket,
            }
        ));
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "shutdown must not wait for the held response or its 30-second deadline"
        );
        assert_eq!(
            claimed.reject(AgentControlRejectionCodeV1::ShutdownInProgress, false),
            Err(AgentControlDispatchResponseError::ClientUnavailable)
        );
        assert!(!running.load(Ordering::Acquire));
        assert!(!socket_path.exists());
    }

    #[test]
    fn dropping_task_never_detaches_and_completes_idle_cleanup_synchronously() {
        let directory = TestDirectory::new(0o700);
        let socket_path = directory.socket_path();
        let path = AgentControlSocketPath::parse(&socket_path).expect("socket path");
        let config = AgentControlSocketConfig::new(path, timeouts(Duration::from_millis(100)));
        let capacity = AgentControlRuntimeQueueCapacity::try_new(1).expect("queue capacity");
        let running = Arc::new(AtomicBool::new(true));
        let (task, receiver) = AgentControlSocketTask::bind_and_spawn(
            config,
            AgentControlMonotonicOrigin::new(Instant::now(), HostMonotonicTimestamp::from_nanos(0)),
            capacity,
            Arc::clone(&running),
        )
        .expect("spawn socket task");

        let started = Instant::now();
        drop(task);
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "idle cancellation must wake the parked task"
        );
        assert!(!running.load(Ordering::Acquire));
        assert!(!socket_path.exists());
        assert!(matches!(
            receiver.recv_timeout(Duration::from_millis(50)),
            Err(AgentControlRuntimeReceiveError::Disconnected)
        ));
    }

    #[test]
    fn replacement_is_detected_and_deterministically_preserved() {
        let directory = TestDirectory::new(0o700);
        let socket_path = directory.socket_path();
        let (mut server, _receiver) = bind_server(&directory, 1, Duration::from_secs(1));
        fs::remove_file(&socket_path).expect("remove created pathname");
        fs::write(&socket_path, b"replacement").expect("replacement file");

        assert!(matches!(
            server.poll_one(),
            Err(AgentControlServeError::SocketPathNoLongerOwned {
                observed: AgentControlObservedSocketPath::Present {
                    is_socket: false,
                    ..
                },
                ..
            })
        ));
        assert!(matches!(
            server.shutdown(),
            AgentControlSocketCleanupOutcome::ReplacementPreserved {
                observed_is_socket: false,
                ..
            }
        ));
        assert_eq!(
            fs::read(&socket_path).expect("replacement preserved"),
            b"replacement"
        );
    }
}
