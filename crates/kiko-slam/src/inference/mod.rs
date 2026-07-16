use ort::Error as OrtError;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use std::cell::Cell;
use std::ffi::CStr;
use std::future::Future;
use std::num::NonZeroUsize;
use std::path::Path;
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex};
use std::task::{Context, Poll, Wake, Waker};
use std::time::{Duration, Instant};

mod backend;
mod eigenplaces;
mod lightglue;
mod place;
mod superpoint;

use crate::env::{EnvError, env_bool, env_string, env_u64, env_usize};

pub use backend::InferenceBackend;
pub use eigenplaces::EigenPlaces;
pub use place::PlaceDescriptorExtractor;

#[derive(Debug)]
pub enum InferenceError {
    RuntimeLibraryUnavailable {
        path: PathBuf,
        message: String,
    },
    RuntimeLoadFailed {
        path: PathBuf,
        source: OrtError,
    },
    LoadFailed {
        path: PathBuf,
        source: OrtError,
    },
    ModelFileUnavailable {
        path: PathBuf,
        source: std::io::Error,
    },

    Execution(OrtError),

    SessionConfiguration {
        source: OrtError,
    },
    Environment(EnvError),
    InvalidOptimizationLevel {
        key: String,
        value: String,
    },
    ThreadCountOutOfRange {
        key: String,
        value: usize,
        maximum: usize,
    },
    AsyncIntraThreadCountTooSmall {
        key: String,
        value: usize,
        minimum: usize,
    },
    HostParallelism {
        source: std::io::Error,
    },
    InvalidWatchdog(WatchdogConfigError),

    UnexpectedOutput {
        name: String,
        expected: String,
        actual: String,
    },
    OutputDecode {
        name: &'static str,
        source: OrtError,
    },
    InputDimensionsTooSmall {
        model: &'static str,
        width: u32,
        height: u32,
        minimum: u32,
    },
    KeypointCoordinateUnrepresentable {
        model: &'static str,
        index: usize,
        axis: &'static str,
        coordinate: u32,
    },
    Downscale(crate::DownscaleError),
    Detection(crate::DetectionError),
    Match(crate::MatchError),
    GlobalDescriptor(crate::loop_closure::GlobalDescriptorError),
    BackendUnavailable {
        requested: InferenceBackend,
    },
    WatchdogTimeout {
        model: &'static str,
        timeout_ms: u64,
    },
    WatchdogDeadlineExceeded {
        model: &'static str,
        timeout_ms: u64,
    },
    WatchdogDeadlineUnavailable {
        model: &'static str,
        timeout_ms: u64,
    },
    SessionQuarantined {
        model: &'static str,
    },
    ThreadPanic {
        stage: &'static str,
    },
    InvariantViolation {
        context: &'static str,
    },
}
impl std::error::Error for InferenceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::RuntimeLoadFailed { source, .. }
            | Self::LoadFailed { source, .. }
            | Self::Execution(source)
            | Self::SessionConfiguration { source }
            | Self::OutputDecode { source, .. } => Some(source),
            Self::Environment(source) => Some(source),
            Self::InvalidWatchdog(source) => Some(source),
            Self::HostParallelism { source } => Some(source),
            Self::ModelFileUnavailable { source, .. } => Some(source),
            Self::Downscale(source) => Some(source),
            Self::Detection(source) => Some(source),
            Self::Match(source) => Some(source),
            Self::GlobalDescriptor(source) => Some(source),
            Self::RuntimeLibraryUnavailable { .. }
            | Self::UnexpectedOutput { .. }
            | Self::InputDimensionsTooSmall { .. }
            | Self::KeypointCoordinateUnrepresentable { .. }
            | Self::InvalidOptimizationLevel { .. }
            | Self::ThreadCountOutOfRange { .. }
            | Self::AsyncIntraThreadCountTooSmall { .. }
            | Self::BackendUnavailable { .. }
            | Self::WatchdogTimeout { .. }
            | Self::WatchdogDeadlineExceeded { .. }
            | Self::WatchdogDeadlineUnavailable { .. }
            | Self::SessionQuarantined { .. }
            | Self::ThreadPanic { .. }
            | Self::InvariantViolation { .. } => None,
        }
    }
}

impl From<OrtError> for InferenceError {
    fn from(e: OrtError) -> Self {
        InferenceError::Execution(e)
    }
}

impl From<EnvError> for InferenceError {
    fn from(error: EnvError) -> Self {
        Self::Environment(error)
    }
}

impl From<crate::DownscaleError> for InferenceError {
    fn from(err: crate::DownscaleError) -> Self {
        InferenceError::Downscale(err)
    }
}

impl From<crate::DetectionError> for InferenceError {
    fn from(err: crate::DetectionError) -> Self {
        InferenceError::Detection(err)
    }
}

impl From<crate::MatchError> for InferenceError {
    fn from(err: crate::MatchError) -> Self {
        InferenceError::Match(err)
    }
}

impl From<crate::loop_closure::GlobalDescriptorError> for InferenceError {
    fn from(err: crate::loop_closure::GlobalDescriptorError) -> Self {
        InferenceError::GlobalDescriptor(err)
    }
}

impl std::fmt::Display for InferenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InferenceError::RuntimeLibraryUnavailable { path, message } => write!(
                f,
                "ONNX Runtime library is unavailable at {}: {message}",
                path.display()
            ),
            InferenceError::RuntimeLoadFailed { path, source } => write!(
                f,
                "failed to load ONNX Runtime at {}: {source}",
                path.display()
            ),
            InferenceError::LoadFailed { path, source } => {
                write!(f, "failed to load model at {}: {source}", path.display())
            }
            InferenceError::ModelFileUnavailable { path, source } => write!(
                f,
                "model file is unavailable at {}: {source}",
                path.display()
            ),
            InferenceError::Execution(e) => write!(f, "execution error: {e}"),
            InferenceError::SessionConfiguration { source } => {
                write!(f, "session configuration error: {source}")
            }
            InferenceError::Environment(source) => {
                write!(f, "invalid inference environment: {source}")
            }
            InferenceError::InvalidOptimizationLevel { key, value } => {
                write!(
                    f,
                    "environment variable {key} has unsupported ONNX optimization level {value:?}"
                )
            }
            InferenceError::ThreadCountOutOfRange {
                key,
                value,
                maximum,
            } => {
                write!(
                    f,
                    "environment variable {key} sets ONNX thread count {value}, maximum is {maximum}"
                )
            }
            InferenceError::AsyncIntraThreadCountTooSmall {
                key,
                value,
                minimum,
            } => write!(
                f,
                "environment variable {key} sets ONNX intra-op thread count {value}; asynchronous inference requires zero (automatic) or at least {minimum}"
            ),
            InferenceError::HostParallelism { source } => write!(
                f,
                "failed to determine host parallelism for the ONNX CPU session: {source}"
            ),
            InferenceError::InvalidWatchdog(source) => {
                write!(f, "invalid inference watchdog configuration: {source}")
            }
            InferenceError::UnexpectedOutput {
                name,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "unexpected output '{name}': expected {expected}, got {actual}"
                )
            }
            InferenceError::OutputDecode { name, source } => {
                write!(f, "failed to decode model output '{name}': {source}")
            }
            InferenceError::InputDimensionsTooSmall {
                model,
                width,
                height,
                minimum,
            } => write!(
                f,
                "{model} input dimensions {width}x{height} are smaller than the {minimum}x{minimum} model minimum"
            ),
            InferenceError::KeypointCoordinateUnrepresentable {
                model,
                index,
                axis,
                coordinate,
            } => write!(
                f,
                "{model} keypoint {index} scaled {axis} pixel {coordinate} cannot be represented exactly by the host f32 coordinate domain"
            ),
            InferenceError::Downscale(err) => write!(f, "downscale error: {err}"),
            InferenceError::Detection(err) => write!(f, "detection error: {err}"),
            InferenceError::Match(err) => write!(f, "match error: {err}"),
            InferenceError::GlobalDescriptor(err) => {
                write!(f, "global descriptor error: {err}")
            }
            InferenceError::BackendUnavailable { requested } => {
                write!(
                    f,
                    "requested inference backend {requested:?} is unavailable"
                )
            }
            InferenceError::WatchdogTimeout { model, timeout_ms } => {
                write!(
                    f,
                    "ONNX inference timed out: model={model} timeout_ms={timeout_ms}"
                )
            }
            InferenceError::WatchdogDeadlineExceeded { model, timeout_ms } => write!(
                f,
                "ONNX inference completed at or after its deadline: model={model} timeout_ms={timeout_ms}"
            ),
            InferenceError::WatchdogDeadlineUnavailable { model, timeout_ms } => write!(
                f,
                "cannot represent ONNX inference deadline: model={model} timeout_ms={timeout_ms}"
            ),
            InferenceError::SessionQuarantined { model } => {
                write!(
                    f,
                    "ONNX session is unavailable after a timed-out run: model={model}"
                )
            }
            InferenceError::ThreadPanic { stage } => {
                write!(f, "inference worker thread panicked ({stage})")
            }
            InferenceError::InvariantViolation { context } => {
                write!(f, "inference invariant violated: {context}")
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WatchdogConfigError {
    ZeroTimeout {
        key: String,
    },
    WarningExceedsTimeout {
        warning_key: String,
        warning_ms: u64,
        timeout_key: String,
        timeout_ms: u64,
    },
    DurationOutOfRange {
        key: String,
        milliseconds: u64,
    },
}

impl std::fmt::Display for WatchdogConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroTimeout { key } => {
                write!(f, "environment variable {key} must be greater than zero")
            }
            Self::WarningExceedsTimeout {
                warning_key,
                warning_ms,
                timeout_key,
                timeout_ms,
            } => write!(
                f,
                "environment variable {warning_key} ({warning_ms} ms) exceeds {timeout_key} ({timeout_ms} ms)"
            ),
            Self::DurationOutOfRange { key, milliseconds } => write!(
                f,
                "environment variable {key} duration {milliseconds} ms cannot be represented by the monotonic clock"
            ),
        }
    }
}

impl std::error::Error for WatchdogConfigError {}

impl From<WatchdogConfigError> for InferenceError {
    fn from(error: WatchdogConfigError) -> Self {
        Self::InvalidWatchdog(error)
    }
}
pub use lightglue::LightGlue;
pub use superpoint::SuperPoint;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WatchdogDuration(u64);

impl WatchdogDuration {
    fn duration(self) -> Duration {
        Duration::from_millis(self.0)
    }

    fn milliseconds(self) -> u64 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WatchdogLimits {
    warn_after: WatchdogDuration,
    timeout: WatchdogDuration,
}

impl WatchdogLimits {
    fn try_from_millis(
        warning_key: &str,
        warning_ms: u64,
        timeout_key: &str,
        timeout_ms: u64,
    ) -> Result<Self, WatchdogConfigError> {
        Self::try_from_millis_at(
            Instant::now(),
            warning_key,
            warning_ms,
            timeout_key,
            timeout_ms,
        )
    }

    fn try_from_millis_at(
        now: Instant,
        warning_key: &str,
        warning_ms: u64,
        timeout_key: &str,
        timeout_ms: u64,
    ) -> Result<Self, WatchdogConfigError> {
        if timeout_ms == 0 {
            return Err(WatchdogConfigError::ZeroTimeout {
                key: timeout_key.to_owned(),
            });
        }
        if warning_ms > timeout_ms {
            return Err(WatchdogConfigError::WarningExceedsTimeout {
                warning_key: warning_key.to_owned(),
                warning_ms,
                timeout_key: timeout_key.to_owned(),
                timeout_ms,
            });
        }
        let warn_after = checked_watchdog_duration(now, warning_key, warning_ms)?;
        let timeout = checked_watchdog_duration(now, timeout_key, timeout_ms)?;
        Ok(Self {
            warn_after,
            timeout,
        })
    }
}

fn checked_watchdog_duration(
    now: Instant,
    key: &str,
    milliseconds: u64,
) -> Result<WatchdogDuration, WatchdogConfigError> {
    let duration = Duration::from_millis(milliseconds);
    if now.checked_add(duration).is_none() {
        return Err(WatchdogConfigError::DurationOutOfRange {
            key: key.to_owned(),
            milliseconds,
        });
    }
    Ok(WatchdogDuration(milliseconds))
}

std::thread_local! {
    static ACTIVE_WATCHDOG_LIMITS: Cell<Option<WatchdogLimits>> = const { Cell::new(None) };
}

struct WatchdogScope {
    previous: Option<WatchdogLimits>,
}

impl WatchdogScope {
    fn enter(limits: WatchdogLimits) -> Self {
        let previous = ACTIVE_WATCHDOG_LIMITS.with(|active| active.replace(Some(limits)));
        Self { previous }
    }
}

impl Drop for WatchdogScope {
    fn drop(&mut self) {
        ACTIVE_WATCHDOG_LIMITS.with(|active| active.set(self.previous));
    }
}

pub(super) struct ManagedSession {
    inner: Option<Box<Session>>,
    watchdog: WatchdogLimits,
}

impl ManagedSession {
    fn new(session: Session, watchdog: WatchdogLimits) -> Self {
        Self {
            inner: Some(Box::new(session)),
            watchdog,
        }
    }

    fn run<T>(
        &mut self,
        model: &'static str,
        operation: impl FnOnce(&mut Session) -> Result<T, InferenceError>,
    ) -> Result<T, InferenceError> {
        let _watchdog_scope = WatchdogScope::enter(self.watchdog);
        let result = match self.inner.as_deref_mut() {
            Some(session) => operation(session),
            None => Err(InferenceError::SessionQuarantined { model }),
        };

        if matches!(result, Err(InferenceError::WatchdogTimeout { .. })) {
            // ort's async cancellation is nonblocking. Its callback retains a
            // reference into Session, so keep the boxed allocation alive and
            // make the model fail-stop rather than risk reuse or destruction.
            if let Some(session) = self.inner.take() {
                let _ = Box::leak(session);
            }
        }

        result
    }
}

pub(super) fn run_with_watchdog<T, F>(
    model: &'static str,
    start_inference: impl FnOnce() -> Result<F, OrtError>,
) -> Result<T, InferenceError>
where
    F: Future<Output = Result<T, OrtError>>,
{
    let limits =
        ACTIVE_WATCHDOG_LIMITS
            .with(Cell::get)
            .ok_or(InferenceError::InvariantViolation {
                context: "watchdog limits are unavailable outside ManagedSession::run",
            })?;
    run_with_limits(model, limits, start_inference)
}

struct WatchdogWake {
    ready: Mutex<bool>,
    wake: Condvar,
}

enum WatchdogRunOutcome<T> {
    CompletedOnTime(Result<T, OrtError>),
    CompletedLate(Result<T, OrtError>),
    PendingAtDeadline,
}

impl Wake for WatchdogWake {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        *self
            .ready
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = true;
        self.wake.notify_one();
    }
}

fn run_with_limits<T, F>(
    model: &'static str,
    limits: WatchdogLimits,
    start_inference: impl FnOnce() -> Result<F, OrtError>,
) -> Result<T, InferenceError>
where
    F: Future<Output = Result<T, OrtError>>,
{
    let start = Instant::now();
    run_with_limits_at(model, limits, start, Instant::now, start_inference)
}

fn run_with_limits_at<T, F>(
    model: &'static str,
    limits: WatchdogLimits,
    start: Instant,
    mut now: impl FnMut() -> Instant,
    start_inference: impl FnOnce() -> Result<F, OrtError>,
) -> Result<T, InferenceError>
where
    F: Future<Output = Result<T, OrtError>>,
{
    const REPOLL_INTERVAL: Duration = Duration::from_millis(10);

    let deadline = start.checked_add(limits.timeout.duration()).ok_or(
        InferenceError::WatchdogDeadlineUnavailable {
            model,
            timeout_ms: limits.timeout.milliseconds(),
        },
    )?;
    let notify = Arc::new(WatchdogWake {
        ready: Mutex::new(false),
        wake: Condvar::new(),
    });
    let waker = Waker::from(Arc::clone(&notify));
    let mut context = Context::from_waker(&waker);
    // No fallible watchdog setup remains after this call. Once an async future
    // exists, every path either observes completion or returns WatchdogTimeout
    // while still pending, causing ManagedSession to quarantine that
    // non-synchronously cancelled run.
    let mut future = Box::pin(start_inference().map_err(InferenceError::Execution)?);

    let outcome = loop {
        *notify
            .ready
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = false;
        if let Poll::Ready(result) = future.as_mut().poll(&mut context) {
            break if now() < deadline {
                WatchdogRunOutcome::CompletedOnTime(result)
            } else {
                WatchdogRunOutcome::CompletedLate(result)
            };
        }

        let observed_at = now();
        if observed_at >= deadline {
            break match future.as_mut().poll(&mut context) {
                Poll::Ready(result) => WatchdogRunOutcome::CompletedLate(result),
                Poll::Pending => WatchdogRunOutcome::PendingAtDeadline,
            };
        }
        let remaining = deadline.saturating_duration_since(observed_at);
        let wait_for = remaining.min(REPOLL_INTERVAL);
        let ready = notify
            .ready
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let (ready, _) = notify
            .wake
            .wait_timeout_while(ready, wait_for, |ready| !*ready)
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        drop(ready);
    };

    // Dropping an incomplete ort future requests cancellation. Do it before
    // logging and before exposing the timeout to the session quarantine layer.
    drop(future);

    let elapsed = now().saturating_duration_since(start);
    if elapsed > limits.warn_after.duration() {
        eprintln!(
            "slow ONNX inference: model={model} elapsed_ms={:.1} threshold_ms={}",
            elapsed.as_secs_f64() * 1000.0,
            limits.warn_after.milliseconds()
        );
    }

    match outcome {
        WatchdogRunOutcome::CompletedOnTime(result) => result.map_err(InferenceError::Execution),
        WatchdogRunOutcome::CompletedLate(Err(source)) => Err(InferenceError::Execution(source)),
        WatchdogRunOutcome::CompletedLate(Ok(_)) => Err(InferenceError::WatchdogDeadlineExceeded {
            model,
            timeout_ms: limits.timeout.milliseconds(),
        }),
        WatchdogRunOutcome::PendingAtDeadline => Err(InferenceError::WatchdogTimeout {
            model,
            timeout_ms: limits.timeout.milliseconds(),
        }),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OrtThreadCount(usize);

impl OrtThreadCount {
    const DEFAULT_INTER: Self = Self(1);

    fn try_from_environment(key: &str, value: usize) -> Result<Self, InferenceError> {
        let maximum = Self::maximum();
        if value > maximum {
            return Err(InferenceError::ThreadCountOutOfRange {
                key: key.to_owned(),
                value,
                maximum,
            });
        }
        Ok(Self(value))
    }

    fn maximum() -> usize {
        usize::try_from(i32::MAX).expect("i32::MAX must fit usize")
    }

    fn get(self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AsyncIntraThreadCount(OrtThreadCount);

impl AsyncIntraThreadCount {
    const MINIMUM: usize = 2;

    fn try_from_environment(key: &str, value: usize) -> Result<Self, InferenceError> {
        if value < Self::MINIMUM {
            return Err(InferenceError::AsyncIntraThreadCountTooSmall {
                key: key.to_owned(),
                value,
                minimum: Self::MINIMUM,
            });
        }
        OrtThreadCount::try_from_environment(key, value).map(Self)
    }

    fn cpu_default(parallelism: NonZeroUsize) -> Self {
        let desired = (parallelism.get() / 2).max(Self::MINIMUM);
        Self(OrtThreadCount(desired.min(OrtThreadCount::maximum())))
    }

    fn get(self) -> usize {
        self.0.get()
    }
}

// `ort::Session::run_async` requires multiple intra-op threads. Parse the
// environment into a policy whose resolved count cannot violate that contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AsyncIntraThreadPolicy {
    Automatic,
    Exact(AsyncIntraThreadCount),
}

impl AsyncIntraThreadPolicy {
    fn try_from_environment(key: &str, value: Option<usize>) -> Result<Self, InferenceError> {
        match value {
            None | Some(0) => Ok(Self::Automatic),
            Some(value) => AsyncIntraThreadCount::try_from_environment(key, value).map(Self::Exact),
        }
    }

    fn resolve(self, selected: InferenceBackend) -> Result<AsyncIntraThreadCount, InferenceError> {
        self.resolve_with(selected, std::thread::available_parallelism)
    }

    fn resolve_with(
        self,
        selected: InferenceBackend,
        available_parallelism: impl FnOnce() -> std::io::Result<NonZeroUsize>,
    ) -> Result<AsyncIntraThreadCount, InferenceError> {
        match (self, selected) {
            (Self::Exact(count), _) => Ok(count),
            (Self::Automatic, InferenceBackend::Cpu) => available_parallelism()
                .map(AsyncIntraThreadCount::cpu_default)
                .map_err(|source| InferenceError::HostParallelism { source }),
            (
                Self::Automatic,
                InferenceBackend::Auto
                | InferenceBackend::CoreMLGpu
                | InferenceBackend::Cuda
                | InferenceBackend::TensorRT,
            ) => Ok(AsyncIntraThreadCount(OrtThreadCount(
                AsyncIntraThreadCount::MINIMUM,
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct SessionSettings {
    intra_thread_policy: AsyncIntraThreadPolicy,
    inter_threads: OrtThreadCount,
    optimization_level: GraphOptimizationLevel,
    memory_pattern: bool,
    parallel_execution: bool,
    cpu_arena: bool,
    watchdog: WatchdogLimits,
}

impl SessionSettings {
    fn from_environment(requested_backend: InferenceBackend) -> Result<Self, InferenceError> {
        const INTRA_THREADS: &str = "KIKO_ORT_INTRA_THREADS";
        const INTER_THREADS: &str = "KIKO_ORT_INTER_THREADS";
        const WATCHDOG_WARNING: &str = "KIKO_ORT_RUN_WARN_MS";
        const WATCHDOG_TIMEOUT: &str = "KIKO_ORT_RUN_TIMEOUT_MS";

        let intra_thread_policy =
            AsyncIntraThreadPolicy::try_from_environment(INTRA_THREADS, env_usize(INTRA_THREADS)?)?;
        let inter_threads = env_usize(INTER_THREADS)?
            .map(|value| OrtThreadCount::try_from_environment(INTER_THREADS, value))
            .transpose()?
            .unwrap_or(OrtThreadCount::DEFAULT_INTER);
        let optimization_level =
            env_opt_level("KIKO_ORT_OPT_LEVEL")?.unwrap_or(GraphOptimizationLevel::Level3);
        let memory_pattern = env_bool("KIKO_ORT_MEM_PATTERN")?.unwrap_or(true);
        let parallel_execution = env_bool("KIKO_ORT_PARALLEL_EXEC")?.unwrap_or(false);
        let cpu_arena = if cpu_provider_may_be_configured(requested_backend) {
            env_bool("KIKO_ORT_CPU_ARENA")?.unwrap_or(true)
        } else {
            true
        };
        let warning_ms = env_u64(WATCHDOG_WARNING)?.unwrap_or(200);
        let timeout_ms = env_u64(WATCHDOG_TIMEOUT)?.unwrap_or(5_000);
        let watchdog = WatchdogLimits::try_from_millis(
            WATCHDOG_WARNING,
            warning_ms,
            WATCHDOG_TIMEOUT,
            timeout_ms,
        )?;

        Ok(Self {
            intra_thread_policy,
            inter_threads,
            optimization_level,
            memory_pattern,
            parallel_execution,
            cpu_arena,
            watchdog,
        })
    }
}

fn cpu_provider_may_be_configured(requested_backend: InferenceBackend) -> bool {
    matches!(
        requested_backend,
        InferenceBackend::Auto | InferenceBackend::Cpu
    )
}

fn build_session(
    path: &std::path::Path,
    backend: InferenceBackend,
) -> Result<(ManagedSession, InferenceBackend), InferenceError> {
    let settings = SessionSettings::from_environment(backend)?;
    ensure_ort_runtime()?;
    let mut builder = Session::builder().map_err(|e| InferenceError::LoadFailed {
        path: path.to_path_buf(),
        source: e,
    })?;

    let selection = backend::select_backend(backend, settings.cpu_arena)?;
    builder = apply_session_config(builder, selection.selected(), &settings)?;
    if selection.strict_accelerator() {
        builder = builder
            .with_disable_cpu_fallback()
            .map_err(session_configuration_error)?;
    }
    if !selection.providers().is_empty() {
        builder = builder
            .with_execution_providers(selection.providers())
            .map_err(session_configuration_error)?;
    }

    let session = builder
        .commit_from_file(path)
        .map_err(|e| InferenceError::LoadFailed {
            path: path.to_path_buf(),
            source: e,
        })?;

    Ok((
        ManagedSession::new(session, settings.watchdog),
        selection.selected(),
    ))
}

fn session_configuration_error(
    error: ort::Error<ort::session::builder::SessionBuilder>,
) -> InferenceError {
    InferenceError::SessionConfiguration {
        source: error.into(),
    }
}

fn ensure_ort_runtime() -> Result<(), InferenceError> {
    let path = std::env::var_os("ORT_DYLIB_PATH")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(default_ort_runtime_path);
    let resolved_path = resolve_runtime_path(&path);
    let preflight = preflight_ort_runtime(&resolved_path)?;
    let environment =
        ort::init_from(&resolved_path).map_err(|source| InferenceError::RuntimeLoadFailed {
            path: resolved_path.clone(),
            source,
        })?;
    let _ = environment.commit();
    drop(preflight);
    Ok(())
}

fn resolve_runtime_path(path: &Path) -> PathBuf {
    if path.is_absolute() {
        return path.to_path_buf();
    }
    let beside_executable = std::env::current_exe()
        .ok()
        .and_then(|executable| executable.parent().map(|parent| parent.join(path)));
    beside_executable
        .filter(|candidate| candidate.is_file())
        .unwrap_or_else(|| path.to_path_buf())
}

#[allow(unsafe_code)]
fn preflight_ort_runtime(path: &Path) -> Result<libloading::Library, InferenceError> {
    let unavailable = |message: String| InferenceError::RuntimeLibraryUnavailable {
        path: path.to_path_buf(),
        message,
    };
    // Loading and querying a native library necessarily crosses an unsafe FFI
    // boundary. Keep the handle alive through ort::init_from, and call only the
    // stable ONNX Runtime entry point after checking every returned pointer.
    let library =
        unsafe { libloading::Library::new(path) }.map_err(|err| unavailable(err.to_string()))?;

    unsafe {
        type ApiBaseGetter = unsafe extern "C" fn() -> *const ort::sys::OrtApiBase;
        let getter: libloading::Symbol<'_, ApiBaseGetter> = library
            .get(b"OrtGetApiBase")
            .map_err(|err| unavailable(err.to_string()))?;
        let base = getter();
        if base.is_null() {
            return Err(unavailable("OrtGetApiBase returned null".to_string()));
        }
        let version_ptr = ((*base).GetVersionString)();
        if version_ptr.is_null() {
            return Err(unavailable(
                "ONNX Runtime returned a null version string".to_string(),
            ));
        }
        let version = CStr::from_ptr(version_ptr).to_string_lossy();
        let minor = version
            .split('.')
            .nth(1)
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or(0);
        if minor < ort::MINOR_VERSION {
            return Err(unavailable(format!(
                "expected ONNX Runtime 1.{} or newer, found {version}",
                ort::MINOR_VERSION
            )));
        }
        if ((*base).GetApi)(ort::sys::ORT_API_VERSION).is_null() {
            return Err(unavailable(format!(
                "runtime {version} does not expose ONNX Runtime API {}",
                ort::sys::ORT_API_VERSION
            )));
        }
    }

    Ok(library)
}

fn default_ort_runtime_path() -> PathBuf {
    #[cfg(target_os = "windows")]
    const LIBRARY: &str = "onnxruntime.dll";
    #[cfg(any(target_os = "linux", target_os = "android"))]
    const LIBRARY: &str = "libonnxruntime.so";
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    const LIBRARY: &str = "libonnxruntime.dylib";
    #[cfg(not(any(
        target_os = "windows",
        target_os = "linux",
        target_os = "android",
        target_os = "macos",
        target_os = "ios"
    )))]
    const LIBRARY: &str = "libonnxruntime.so";
    PathBuf::from(LIBRARY)
}

fn apply_session_config(
    builder: ort::session::builder::SessionBuilder,
    selected: InferenceBackend,
    settings: &SessionSettings,
) -> Result<ort::session::builder::SessionBuilder, InferenceError> {
    let intra = settings.intra_thread_policy.resolve(selected)?;
    let builder = builder
        .with_optimization_level(settings.optimization_level)
        .map_err(session_configuration_error)?;
    let builder = builder
        .with_memory_pattern(settings.memory_pattern)
        .map_err(session_configuration_error)?;
    let builder = builder
        .with_intra_threads(intra.get())
        .map_err(session_configuration_error)?;
    let builder = builder
        .with_inter_threads(settings.inter_threads.get())
        .map_err(session_configuration_error)?;
    builder
        .with_parallel_execution(settings.parallel_execution)
        .map_err(session_configuration_error)
}

fn env_opt_level(key: &str) -> Result<Option<GraphOptimizationLevel>, InferenceError> {
    env_string(key)?
        .map(|raw| parse_opt_level(key, raw))
        .transpose()
}

fn parse_opt_level(key: &str, raw: String) -> Result<GraphOptimizationLevel, InferenceError> {
    let value = raw.trim();
    if value == "0" || value.eq_ignore_ascii_case("disable") {
        return Ok(GraphOptimizationLevel::Disable);
    }
    if value == "1" || value.eq_ignore_ascii_case("level1") || value.eq_ignore_ascii_case("basic") {
        return Ok(GraphOptimizationLevel::Level1);
    }
    if value == "2"
        || value.eq_ignore_ascii_case("level2")
        || value.eq_ignore_ascii_case("extended")
    {
        return Ok(GraphOptimizationLevel::Level2);
    }
    if value == "3" || value.eq_ignore_ascii_case("level3") || value.eq_ignore_ascii_case("all") {
        return Ok(GraphOptimizationLevel::Level3);
    }
    Err(InferenceError::InvalidOptimizationLevel {
        key: key.to_owned(),
        value: raw,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        ACTIVE_WATCHDOG_LIMITS, AsyncIntraThreadPolicy, InferenceBackend, InferenceError,
        OrtThreadCount, WatchdogConfigError, WatchdogDuration, WatchdogLimits, WatchdogScope,
        cpu_provider_may_be_configured, parse_opt_level, preflight_ort_runtime, run_with_limits,
        run_with_limits_at, run_with_watchdog,
    };
    use ort::session::builder::GraphOptimizationLevel;
    use std::cell::Cell;
    use std::num::NonZeroUsize;
    use std::time::{Duration, Instant};

    fn watchdog_limits(warning_ms: u64, timeout_ms: u64) -> WatchdogLimits {
        WatchdogLimits::try_from_millis("WARN", warning_ms, "TIMEOUT", timeout_ms)
            .expect("valid watchdog fixture")
    }

    fn instant_that_cannot_add(duration: Duration) -> Instant {
        let mut base = Instant::now();
        for _ in 0..4_096 {
            let Some(next) = base.checked_add(duration) else {
                return base;
            };
            base = next;
        }
        panic!("monotonic clock accepted more than 4096 maximum-duration steps");
    }

    #[test]
    fn optimization_level_parser_accepts_aliases_without_normalizing() {
        assert_eq!(
            parse_opt_level("TEST_OPT", " BASIC ".to_owned()).expect("basic level"),
            GraphOptimizationLevel::Level1
        );
        assert_eq!(
            parse_opt_level("TEST_OPT", "Extended".to_owned()).expect("extended level"),
            GraphOptimizationLevel::Level2
        );
        assert_eq!(
            parse_opt_level("TEST_OPT", "ALL".to_owned()).expect("all level"),
            GraphOptimizationLevel::Level3
        );
    }

    #[test]
    fn optimization_level_parser_rejects_unknown_values() {
        assert!(matches!(
            parse_opt_level("TEST_OPT", "fastest".to_owned()),
            Err(InferenceError::InvalidOptimizationLevel { key, value })
                if key == "TEST_OPT" && value == "fastest"
        ));
    }

    #[test]
    fn cpu_arena_is_parsed_only_when_a_cpu_provider_may_be_configured() {
        assert!(cpu_provider_may_be_configured(InferenceBackend::Auto));
        assert!(cpu_provider_may_be_configured(InferenceBackend::Cpu));
        assert!(!cpu_provider_may_be_configured(InferenceBackend::CoreMLGpu));
        assert!(!cpu_provider_may_be_configured(InferenceBackend::Cuda));
        assert!(!cpu_provider_may_be_configured(InferenceBackend::TensorRT));
    }

    #[test]
    fn ort_thread_count_preserves_ort_default_and_rejects_c_int_overflow() {
        assert_eq!(
            OrtThreadCount::try_from_environment("TEST_THREADS", 0)
                .expect("zero means ORT default")
                .get(),
            0
        );
        let maximum = OrtThreadCount::maximum();
        assert_eq!(
            OrtThreadCount::try_from_environment("TEST_THREADS", maximum)
                .expect("maximum c_int count")
                .get(),
            maximum
        );
        assert!(matches!(
            OrtThreadCount::try_from_environment("TEST_THREADS", maximum + 1),
            Err(InferenceError::ThreadCountOutOfRange {
                key,
                value,
                maximum: error_maximum,
            }) if key == "TEST_THREADS"
                && value == maximum + 1
                && error_maximum == maximum
        ));
    }

    #[test]
    fn async_intra_thread_policy_parses_automatic_or_multiple_threads() {
        assert_eq!(
            AsyncIntraThreadPolicy::try_from_environment("TEST_THREADS", None)
                .expect("unset selects the automatic policy"),
            AsyncIntraThreadPolicy::Automatic
        );
        assert_eq!(
            AsyncIntraThreadPolicy::try_from_environment("TEST_THREADS", Some(0))
                .expect("zero selects the automatic policy"),
            AsyncIntraThreadPolicy::Automatic
        );
        assert!(matches!(
            AsyncIntraThreadPolicy::try_from_environment("TEST_THREADS", Some(1)),
            Err(InferenceError::AsyncIntraThreadCountTooSmall {
                key,
                value: 1,
                minimum: 2,
            }) if key == "TEST_THREADS"
        ));
        assert!(matches!(
            AsyncIntraThreadPolicy::try_from_environment("TEST_THREADS", Some(2)),
            Ok(AsyncIntraThreadPolicy::Exact(count)) if count.get() == 2
        ));

        let maximum = OrtThreadCount::maximum();
        assert!(matches!(
            AsyncIntraThreadPolicy::try_from_environment("TEST_THREADS", Some(maximum + 1)),
            Err(InferenceError::ThreadCountOutOfRange {
                key,
                value,
                maximum: error_maximum,
            }) if key == "TEST_THREADS"
                && value == maximum + 1
                && error_maximum == maximum
        ));
    }

    #[test]
    fn automatic_async_thread_policy_is_multiple_on_every_backend() {
        let expected = [(1, 2), (2, 2), (3, 2), (4, 2), (8, 4)];
        for (parallelism, thread_count) in expected {
            let parallelism = NonZeroUsize::new(parallelism).expect("positive fixture");
            assert_eq!(
                AsyncIntraThreadPolicy::Automatic
                    .resolve_with(InferenceBackend::Cpu, || Ok(parallelism))
                    .expect("fixture parallelism is available")
                    .get(),
                thread_count
            );
        }

        for backend in [
            InferenceBackend::Auto,
            InferenceBackend::CoreMLGpu,
            InferenceBackend::Cuda,
            InferenceBackend::TensorRT,
        ] {
            assert_eq!(
                AsyncIntraThreadPolicy::Automatic
                    .resolve_with(backend, || panic!(
                        "accelerators do not query CPU parallelism"
                    ))
                    .expect("automatic accelerator count")
                    .get(),
                2
            );
        }
    }

    #[test]
    fn async_thread_policy_queries_parallelism_only_for_automatic_cpu() {
        let explicit = AsyncIntraThreadPolicy::try_from_environment("TEST_THREADS", Some(3))
            .expect("valid exact count");
        assert_eq!(
            explicit
                .resolve_with(InferenceBackend::Cpu, || {
                    panic!("exact counts do not query host parallelism")
                })
                .expect("exact count")
                .get(),
            3
        );

        let error = AsyncIntraThreadPolicy::Automatic
            .resolve_with(InferenceBackend::Cpu, || {
                Err(std::io::Error::other("parallelism unavailable"))
            })
            .expect_err("automatic CPU policy preserves discovery failure");
        assert!(matches!(
            error,
            InferenceError::HostParallelism { source }
                if source.to_string() == "parallelism unavailable"
        ));
    }

    #[test]
    fn watchdog_domain_rejects_zero_timeout_and_reversed_thresholds() {
        assert_eq!(
            WatchdogLimits::try_from_millis("WARN", 0, "TIMEOUT", 0),
            Err(WatchdogConfigError::ZeroTimeout {
                key: "TIMEOUT".to_owned(),
            })
        );
        assert_eq!(
            WatchdogLimits::try_from_millis("WARN", 11, "TIMEOUT", 10),
            Err(WatchdogConfigError::WarningExceedsTimeout {
                warning_key: "WARN".to_owned(),
                warning_ms: 11,
                timeout_key: "TIMEOUT".to_owned(),
                timeout_ms: 10,
            })
        );
        let limits = WatchdogLimits::try_from_millis("WARN", 10, "TIMEOUT", 10)
            .expect("equal warning and timeout are valid");
        assert_eq!(limits.warn_after.milliseconds(), 10);
        assert_eq!(limits.warn_after.duration(), Duration::from_millis(10));
        assert_eq!(limits.timeout.milliseconds(), 10);
        assert_eq!(limits.timeout.duration(), Duration::from_millis(10));
    }

    #[test]
    fn watchdog_domain_rejects_unrepresentable_deadlines() {
        let step = Duration::from_millis(u64::MAX);
        let base = instant_that_cannot_add(step);
        assert_eq!(
            WatchdogLimits::try_from_millis_at(base, "WARN", 0, "TIMEOUT", u64::MAX,),
            Err(WatchdogConfigError::DurationOutOfRange {
                key: "TIMEOUT".to_owned(),
                milliseconds: u64::MAX,
            })
        );
    }

    #[test]
    fn watchdog_limits_are_scoped_to_a_managed_run() {
        let limits = watchdog_limits(10, 20);
        assert_eq!(ACTIVE_WATCHDOG_LIMITS.with(Cell::get), None);
        {
            let _scope = WatchdogScope::enter(limits);
            assert_eq!(ACTIVE_WATCHDOG_LIMITS.with(Cell::get), Some(limits));
        }
        assert_eq!(ACTIVE_WATCHDOG_LIMITS.with(Cell::get), None);
        let started = Cell::new(false);
        assert!(matches!(
            run_with_watchdog("unmanaged", || {
                started.set(true);
                Ok(std::future::ready(Ok(())))
            }),
            Err(InferenceError::InvariantViolation { .. })
        ));
        assert!(!started.get());
    }

    #[test]
    fn watchdog_runtime_deadline_failure_does_not_start_inference() {
        let timeout = Duration::from_millis(u64::MAX);
        let start = instant_that_cannot_add(timeout);
        let limits = WatchdogLimits {
            warn_after: WatchdogDuration(0),
            timeout: WatchdogDuration(u64::MAX),
        };
        let started = Cell::new(false);
        let result = run_with_limits_at(
            "deadline-test",
            limits,
            start,
            || panic!("clock is not observed when deadline setup fails"),
            || {
                started.set(true);
                Ok(std::future::ready(Ok(())))
            },
        );
        assert!(matches!(
            result,
            Err(InferenceError::WatchdogDeadlineUnavailable {
                model: "deadline-test",
                timeout_ms: u64::MAX,
            })
        ));
        assert!(!started.get());
    }

    #[test]
    fn watchdog_deadline_is_strict() {
        let start = Instant::now();
        let limits = watchdog_limits(1, 1);
        let deadline = start
            .checked_add(limits.timeout.duration())
            .expect("small test deadline");
        let at_deadline = run_with_limits_at(
            "tie-test",
            limits,
            start,
            || deadline,
            || Ok(std::future::ready(Ok(()))),
        );
        assert!(matches!(
            at_deadline,
            Err(InferenceError::WatchdogDeadlineExceeded {
                model: "tie-test",
                timeout_ms: 1,
            })
        ));

        let before_deadline = deadline
            .checked_sub(Duration::from_nanos(1))
            .expect("deadline follows start");
        let completed = run_with_limits_at(
            "before-deadline-test",
            limits,
            start,
            || before_deadline,
            || Ok(std::future::ready(Ok(()))),
        );
        assert!(completed.is_ok());
    }

    #[test]
    fn watchdog_repolls_before_classifying_a_wait_timeout() {
        let start = Instant::now();
        let limits = watchdog_limits(1, 1);
        let deadline = start
            .checked_add(limits.timeout.duration())
            .expect("small test deadline");
        let polls = Cell::new(0_usize);
        let inference = std::future::poll_fn(|_| {
            let poll = polls.get();
            polls.set(poll + 1);
            if poll == 0 {
                std::task::Poll::Pending
            } else {
                std::task::Poll::Ready(Ok(()))
            }
        });
        let observations = Cell::new(0_usize);
        let result = run_with_limits_at(
            "repoll-test",
            limits,
            start,
            || {
                let observation = observations.get();
                observations.set(observation + 1);
                if observation == 0 { start } else { deadline }
            },
            || Ok(inference),
        );

        assert_eq!(polls.get(), 2);
        assert!(matches!(
            result,
            Err(InferenceError::WatchdogDeadlineExceeded {
                model: "repoll-test",
                timeout_ms: 1,
            })
        ));
    }

    #[test]
    fn watchdog_repolls_before_classifying_the_first_deadline_observation() {
        let start = Instant::now();
        let limits = watchdog_limits(1, 1);
        let deadline = start
            .checked_add(limits.timeout.duration())
            .expect("small test deadline");
        let polls = Cell::new(0_usize);
        let inference = std::future::poll_fn(|_| {
            let poll = polls.get();
            polls.set(poll + 1);
            if poll == 0 {
                std::task::Poll::Pending
            } else {
                std::task::Poll::Ready(Ok(()))
            }
        });
        let result = run_with_limits_at(
            "first-deadline-observation-test",
            limits,
            start,
            || deadline,
            || Ok(inference),
        );

        assert_eq!(polls.get(), 2);
        assert!(matches!(
            result,
            Err(InferenceError::WatchdogDeadlineExceeded {
                model: "first-deadline-observation-test",
                timeout_ms: 1,
            })
        ));
    }

    #[test]
    fn watchdog_timeout_does_not_affect_the_next_independent_run() {
        let timed_out = run_with_limits("pending-test", watchdog_limits(1, 1), || {
            Ok(std::future::pending::<Result<(), ort::Error>>())
        });
        assert!(matches!(
            timed_out,
            Err(InferenceError::WatchdogTimeout { .. })
        ));

        let completed = run_with_limits("ready-test", watchdog_limits(50, 50), || {
            Ok(std::future::ready(Ok(())))
        });
        assert!(completed.is_ok());
    }

    #[test]
    fn runtime_preflight_rejects_a_missing_library_without_entering_ort() {
        let missing = std::env::temp_dir().join(format!(
            "kiko-missing-onnxruntime-{}-{}.so",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        assert!(!missing.exists());
        assert!(matches!(
            preflight_ort_runtime(&missing),
            Err(InferenceError::RuntimeLibraryUnavailable { .. })
        ));
    }
}
