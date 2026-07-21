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
use std::num::NonZeroUsize;
use std::os::fd::AsRawFd;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{FileTypeExt, MetadataExt, PermissionsExt};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Component, Path, PathBuf};
use std::sync::mpsc::{
    self, Receiver, RecvError, RecvTimeoutError, SyncSender, TryRecvError, TrySendError,
};
use std::sync::{
    Arc,
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
}

impl AgentControlSocketTimeouts {
    /// Parse three nonzero deadlines, each in the inclusive range 1 ms..=30 s.
    pub fn try_new(
        read: Duration,
        write: Duration,
        runtime_response: Duration,
    ) -> Result<Self, AgentControlSocketTimeoutError> {
        validate_timeout(AgentControlTimeoutKind::Read, read)?;
        validate_timeout(AgentControlTimeoutKind::Write, write)?;
        validate_timeout(AgentControlTimeoutKind::RuntimeResponse, runtime_response)?;
        Ok(Self {
            read,
            write,
            runtime_response,
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

/// Deadline named by a timeout configuration error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AgentControlTimeoutKind {
    Read,
    Write,
    RuntimeResponse,
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
}

/// Runtime-side receiver for parsed requests.
#[derive(Debug)]
pub struct AgentControlRuntimeReceiver {
    inner: Receiver<AgentControlDispatch>,
}

/// Construct the bounded handoff queue used by exactly one socket server.
pub fn agent_control_runtime_queue(
    capacity: AgentControlRuntimeQueueCapacity,
) -> (AgentControlRuntimeSender, AgentControlRuntimeReceiver) {
    let (sender, receiver) = mpsc::sync_channel(capacity.get());
    (
        AgentControlRuntimeSender { inner: sender },
        AgentControlRuntimeReceiver { inner: receiver },
    )
}

#[cfg(all(test, feature = "agent-runtime", feature = "actuation"))]
pub(crate) fn enqueue_agent_control_test_request(
    sender: &AgentControlRuntimeSender,
    request: AgentControlRequestV1,
    received_at: HostMonotonicTimestamp,
) -> JoinHandle<Option<AgentControlResponseV1>> {
    let (claim_sender, claim_receiver) = mpsc::sync_channel(0);
    let (response_sender, response_receiver) = mpsc::sync_channel(0);
    sender
        .inner
        .try_send(AgentControlDispatch {
            request,
            received_at,
            claim: claim_sender,
            response: response_sender,
        })
        .expect("test runtime queue has capacity");
    thread::spawn(move || {
        claim_receiver.recv().ok()?;
        response_receiver.recv().ok()
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
    sender
        .inner
        .try_send(AgentControlDispatch {
            request,
            received_at,
            claim: claim_sender,
            response: response_sender,
        })
        .expect("test runtime queue has capacity");
    thread::spawn(move || {
        claim_receiver.recv().expect("dispatcher claims request");
        drop(response_receiver);
    })
}

impl AgentControlRuntimeReceiver {
    /// Wait until one request is available or all producers are gone.
    pub fn recv(&self) -> Result<AgentControlDispatch, AgentControlRuntimeReceiveError> {
        self.inner
            .recv()
            .map_err(|_: RecvError| AgentControlRuntimeReceiveError::Disconnected)
    }

    /// Wait for one bounded duration.
    pub fn recv_timeout(
        &self,
        timeout: Duration,
    ) -> Result<AgentControlDispatch, AgentControlRuntimeReceiveError> {
        self.inner
            .recv_timeout(timeout)
            .map_err(|error| match error {
                RecvTimeoutError::Timeout => AgentControlRuntimeReceiveError::Timeout,
                RecvTimeoutError::Disconnected => AgentControlRuntimeReceiveError::Disconnected,
            })
    }

    /// Poll without blocking.
    pub fn try_recv(&self) -> Result<AgentControlDispatch, AgentControlRuntimeReceiveError> {
        self.inner.try_recv().map_err(|error| match error {
            TryRecvError::Empty => AgentControlRuntimeReceiveError::Empty,
            TryRecvError::Disconnected => AgentControlRuntimeReceiveError::Disconnected,
        })
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
    claim: SyncSender<()>,
    response: SyncSender<AgentControlResponseV1>,
}

impl AgentControlDispatch {
    pub const fn request(&self) -> AgentControlRequestV1 {
        self.request
    }

    pub const fn received_at(&self) -> HostMonotonicTimestamp {
        self.received_at
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
            response: self.response,
        })
    }
}

/// Request released only after an internal claim rendezvous. It still carries
/// no authority proof. Exactly one consuming response method must follow.
#[derive(Debug)]
pub struct AgentControlClaimedRequest {
    request: AgentControlRequestV1,
    received_at: HostMonotonicTimestamp,
    response: SyncSender<AgentControlResponseV1>,
}

impl AgentControlClaimedRequest {
    pub const fn request(&self) -> AgentControlRequestV1 {
        self.request
    }

    pub const fn received_at(&self) -> HostMonotonicTimestamp {
        self.received_at
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
        let (response, connection_issue) =
            match read_request_frame_into(&mut stream, self.timeouts.read, &mut request_payload) {
                Ok(payload) => self.handle_payload(payload)?,
                Err(ReadRequestFrameError::InvalidLength { declared_bytes }) => (
                    AgentControlResponseV1::rejected(
                        None,
                        AgentControlRejectionCodeV1::MalformedRequest,
                        false,
                    ),
                    Some(AgentControlConnectionIssue::InvalidFrameLength { declared_bytes }),
                ),
                Err(ReadRequestFrameError::Io { source, timeout }) => (
                    AgentControlResponseV1::rejected(
                        None,
                        AgentControlRejectionCodeV1::MalformedRequest,
                        timeout,
                    ),
                    Some(AgentControlConnectionIssue::Read { source, timeout }),
                ),
            };
        write_response_frame(&mut stream, response, self.timeouts.write)?;
        Ok(AgentControlServeOutcome::Responded {
            response,
            connection_issue,
        })
    }

    fn handle_payload(
        &mut self,
        payload: &[u8],
    ) -> Result<(AgentControlResponseV1, Option<AgentControlConnectionIssue>), AgentControlServeError>
    {
        let request = match self.parser.parse_next(payload) {
            Ok(request) => request,
            Err(source) => {
                return Ok((
                    parse_rejection(&source),
                    Some(AgentControlConnectionIssue::RequestParse { source }),
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
                ));
            }
        };
        let runtime_deadline = Instant::now()
            .checked_add(self.timeouts.runtime_response)
            .ok_or(AgentControlServeError::DeadlineOverflow)?;
        let (claim_sender, claim_receiver) = mpsc::sync_channel(0);
        let (response_sender, response_receiver) = mpsc::sync_channel(0);
        let dispatch = AgentControlDispatch {
            request,
            received_at,
            claim: claim_sender,
            response: response_sender,
        };
        match self.runtime.inner.try_send(dispatch) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {
                return Ok((
                    AgentControlResponseV1::rejected(
                        Some(request.request_id()),
                        AgentControlRejectionCodeV1::NotReady,
                        true,
                    ),
                    Some(AgentControlConnectionIssue::RuntimeQueueFull),
                ));
            }
            Err(TrySendError::Disconnected(_)) => {
                return Ok((
                    AgentControlResponseV1::rejected(
                        Some(request.request_id()),
                        AgentControlRejectionCodeV1::ShutdownInProgress,
                        false,
                    ),
                    Some(AgentControlConnectionIssue::RuntimeQueueDisconnected),
                ));
            }
        }

        let claim_wait = match remaining_until(runtime_deadline) {
            Ok(remaining) => remaining,
            Err(AgentControlServeError::RuntimeDeadline) => {
                return Ok((
                    AgentControlResponseV1::rejected(
                        Some(request.request_id()),
                        AgentControlRejectionCodeV1::NotReady,
                        true,
                    ),
                    Some(AgentControlConnectionIssue::RuntimeClaimTimeout),
                ));
            }
            Err(other) => return Err(other),
        };
        match claim_receiver.recv_timeout(claim_wait) {
            Ok(()) => {}
            Err(RecvTimeoutError::Timeout) => {
                return Ok((
                    AgentControlResponseV1::rejected(
                        Some(request.request_id()),
                        AgentControlRejectionCodeV1::NotReady,
                        true,
                    ),
                    Some(AgentControlConnectionIssue::RuntimeClaimTimeout),
                ));
            }
            Err(RecvTimeoutError::Disconnected) => {
                return Ok((
                    AgentControlResponseV1::rejected(
                        Some(request.request_id()),
                        AgentControlRejectionCodeV1::InternalFault,
                        false,
                    ),
                    Some(AgentControlConnectionIssue::RuntimeClaimChannelClosed),
                ));
            }
        }

        let response_wait = remaining_until(runtime_deadline).map_err(|error| match error {
            AgentControlServeError::RuntimeDeadline => {
                AgentControlServeError::RuntimeResponseDeadlineAfterClaim {
                    request_id: request.request_id(),
                }
            }
            other => other,
        })?;
        match response_receiver.recv_timeout(response_wait) {
            Ok(response) => Ok((response, None)),
            Err(RecvTimeoutError::Timeout) => {
                Err(AgentControlServeError::RuntimeResponseDeadlineAfterClaim {
                    request_id: request.request_id(),
                })
            }
            Err(RecvTimeoutError::Disconnected) => Err(
                AgentControlServeError::RuntimeResponseChannelClosedAfterClaim {
                    request_id: request.request_id(),
                },
            ),
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
    handle: JoinHandle<AgentControlSocketTaskExit>,
}

impl AgentControlSocketTask {
    /// Bind before spawning, so startup cannot report readiness until the
    /// private path and permissions have been established.
    pub fn bind_and_spawn(
        config: AgentControlSocketConfig,
        clock: AgentControlMonotonicOrigin,
        capacity: AgentControlRuntimeQueueCapacity,
        running: Arc<AtomicBool>,
    ) -> Result<(Self, AgentControlRuntimeReceiver), AgentControlSocketBindError> {
        let (sender, receiver) = agent_control_runtime_queue(capacity);
        let mut server = AgentControlSocketServer::bind(config, clock, sender)?;
        let handle = thread::spawn(move || {
            while running.load(Ordering::Acquire) {
                match server.poll_one() {
                    Ok(AgentControlServeOutcome::Idle) => {
                        thread::park_timeout(Duration::from_millis(1));
                    }
                    Ok(AgentControlServeOutcome::Responded { .. }) => {}
                    Err(source) => {
                        running.store(false, Ordering::Release);
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
        Ok((Self { handle }, receiver))
    }

    pub fn join(self) -> Result<AgentControlSocketTaskExit, AgentControlSocketTaskJoinError> {
        self.handle
            .join()
            .map_err(|_| AgentControlSocketTaskJoinError::Panicked)
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
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AgentControlSocketTaskJoinError {
    Panicked,
}

impl fmt::Display for AgentControlSocketTaskJoinError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("agent-control socket task panicked")
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
    RuntimeDeadline,
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
            Self::RuntimeDeadline => {
                formatter.write_str("agent-control runtime claim deadline expired")
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
            Self::ResponseTooLarge { .. }
            | Self::ResponseDeadline
            | Self::DeadlineOverflow
            | Self::RuntimeDeadline
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
}

fn read_request_frame_into<'buffer>(
    stream: &mut UnixStream,
    timeout: Duration,
    payload: &'buffer mut [u8; MAX_AGENT_CONTROL_REQUEST_JSON_BYTES],
) -> Result<&'buffer [u8], ReadRequestFrameError> {
    let deadline =
        Instant::now()
            .checked_add(timeout)
            .ok_or_else(|| ReadRequestFrameError::Io {
                source: io::Error::new(io::ErrorKind::InvalidInput, "read deadline overflow"),
                timeout: false,
            })?;
    let mut length = [0_u8; FRAME_LENGTH_BYTES];
    read_exact_until(stream, &mut length, deadline).map_err(|source| {
        let timeout = is_timeout_error(&source);
        ReadRequestFrameError::Io { source, timeout }
    })?;
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
    read_exact_until(stream, &mut payload[..declared_bytes], deadline).map_err(|source| {
        let timeout = is_timeout_error(&source);
        ReadRequestFrameError::Io { source, timeout }
    })?;
    Ok(&payload[..declared_bytes])
}

fn write_response_frame(
    stream: &mut UnixStream,
    response: AgentControlResponseV1,
    timeout: Duration,
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
    write_all_until(stream, &length, deadline).map_err(map_response_write_error)?;
    write_all_until(stream, payload.as_slice(), deadline).map_err(map_response_write_error)
}

fn map_response_write_error(source: io::Error) -> AgentControlServeError {
    if is_timeout_error(&source) {
        AgentControlServeError::ResponseDeadline
    } else {
        AgentControlServeError::ResponseWrite { source }
    }
}

fn remaining_until(deadline: Instant) -> Result<Duration, AgentControlServeError> {
    deadline
        .checked_duration_since(Instant::now())
        .filter(|remaining| !remaining.is_zero())
        .ok_or(AgentControlServeError::RuntimeDeadline)
}

fn read_exact_until(
    stream: &mut UnixStream,
    mut destination: &mut [u8],
    deadline: Instant,
) -> io::Result<()> {
    while !destination.is_empty() {
        let remaining = deadline
            .checked_duration_since(Instant::now())
            .filter(|remaining| !remaining.is_zero())
            .ok_or_else(deadline_io_error)?;
        stream.set_read_timeout(Some(remaining))?;
        match stream.read(destination) {
            Ok(0) => return Err(io::Error::from(io::ErrorKind::UnexpectedEof)),
            Ok(read) => destination = &mut destination[read..],
            Err(source) if source.kind() == io::ErrorKind::Interrupted => {}
            Err(source) => return Err(source),
        }
    }
    Ok(())
}

fn write_all_until(
    stream: &mut UnixStream,
    mut source: &[u8],
    deadline: Instant,
) -> io::Result<()> {
    while !source.is_empty() {
        let remaining = deadline
            .checked_duration_since(Instant::now())
            .filter(|remaining| !remaining.is_zero())
            .ok_or_else(deadline_io_error)?;
        stream.set_write_timeout(Some(remaining))?;
        match stream.write(source) {
            Ok(0) => return Err(io::Error::from(io::ErrorKind::WriteZero)),
            Ok(written) => source = &source[written..],
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
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
    use crate::navigation::{AgentBaseCommandStateV1, AgentMapStateV1, AgentRuntimeStateV1};

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
        )
        .expect("test timeouts")
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
                Duration::from_secs(1)
            )
            .is_err()
        );
        assert!(
            AgentControlSocketTimeouts::try_new(
                Duration::from_secs(1),
                Duration::from_secs(31),
                Duration::from_secs(1)
            )
            .is_err()
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
    fn completed_response_is_framed_bounded_and_stamped_from_shared_origin() {
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
                AgentControlCommandV1::Stop
            ));
            claimed.respond_completed().expect("completion rendezvous");
        });
        let server_thread = thread::spawn(move || {
            let mut server = server;
            let outcome = poll_until_connection(&mut server);
            (server, outcome)
        });
        let (response, encoded_bytes) =
            round_trip(&socket_path, &request(1, json!({"kind":"stop"})));
        assert!(encoded_bytes <= MAX_AGENT_CONTROL_RESPONSE_JSON_BYTES);
        assert_eq!(response["request_id"], 1);
        assert_eq!(response["response"]["kind"], "accepted");
        assert_eq!(response["response"]["command"], "stop");
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
