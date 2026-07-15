use ort::Error as OrtError;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use std::ffi::CStr;
use std::future::Future;
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

use crate::env::{env_bool, env_usize};

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

    Execution(OrtError),

    SessionConfiguration {
        message: String,
    },

    UnexpectedOutput {
        name: String,
        expected: String,
        actual: String,
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
            | Self::Execution(source) => Some(source),
            Self::Downscale(source) => Some(source),
            Self::Detection(source) => Some(source),
            Self::Match(source) => Some(source),
            Self::GlobalDescriptor(source) => Some(source),
            Self::RuntimeLibraryUnavailable { .. }
            | Self::UnexpectedOutput { .. }
            | Self::SessionConfiguration { .. }
            | Self::BackendUnavailable { .. }
            | Self::WatchdogTimeout { .. }
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
            InferenceError::Execution(e) => write!(f, "execution error: {e}"),
            InferenceError::SessionConfiguration { message } => {
                write!(f, "session configuration error: {message}")
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
pub use lightglue::LightGlue;
pub use superpoint::SuperPoint;

pub(super) struct ManagedSession {
    inner: Option<Box<Session>>,
}

impl ManagedSession {
    fn new(session: Session) -> Self {
        Self {
            inner: Some(Box::new(session)),
        }
    }

    fn run<T>(
        &mut self,
        model: &'static str,
        operation: impl FnOnce(&mut Session) -> Result<T, InferenceError>,
    ) -> Result<T, InferenceError> {
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

pub(super) fn run_with_watchdog<T, F>(model: &'static str, future: F) -> Result<T, InferenceError>
where
    F: Future<Output = Result<T, OrtError>>,
{
    let warn_ms = env_usize("KIKO_ORT_RUN_WARN_MS").unwrap_or(200) as u64;
    let timeout_ms = env_usize("KIKO_ORT_RUN_TIMEOUT_MS").unwrap_or(5_000) as u64;
    run_with_limits(
        model,
        Duration::from_millis(warn_ms),
        Duration::from_millis(timeout_ms),
        future,
    )
}

struct WatchdogWake {
    ready: Mutex<bool>,
    wake: Condvar,
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
    warn_after: Duration,
    timeout: Duration,
    future: F,
) -> Result<T, InferenceError>
where
    F: Future<Output = Result<T, OrtError>>,
{
    const REPOLL_INTERVAL: Duration = Duration::from_millis(10);

    let start = Instant::now();
    let deadline = start.checked_add(timeout).unwrap_or(start);
    let notify = Arc::new(WatchdogWake {
        ready: Mutex::new(false),
        wake: Condvar::new(),
    });
    let waker = Waker::from(Arc::clone(&notify));
    let mut context = Context::from_waker(&waker);
    let mut future = Box::pin(future);

    let result = loop {
        *notify
            .ready
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = false;
        if let Poll::Ready(result) = future.as_mut().poll(&mut context) {
            break Some(result);
        }

        let now = Instant::now();
        if now >= deadline {
            break None;
        }
        let remaining = deadline.saturating_duration_since(now);
        let wait_for = remaining.min(REPOLL_INTERVAL);
        let ready = notify
            .ready
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let (ready, wait) = notify
            .wake
            .wait_timeout_while(ready, wait_for, |ready| !*ready)
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if wait.timed_out() && !*ready {
            if Instant::now() >= deadline {
                break None;
            }
            continue;
        }
    };

    let elapsed = start.elapsed();
    if elapsed > warn_after {
        eprintln!(
            "slow ONNX inference: model={model} elapsed_ms={:.1} threshold_ms={}",
            elapsed.as_secs_f64() * 1000.0,
            warn_after.as_millis()
        );
    }

    match result {
        Some(result) => result.map_err(InferenceError::Execution),
        None => Err(InferenceError::WatchdogTimeout {
            model,
            timeout_ms: timeout.as_millis().min(u128::from(u64::MAX)) as u64,
        }),
    }
}

fn build_session(
    path: &std::path::Path,
    backend: InferenceBackend,
) -> Result<(ManagedSession, InferenceBackend), InferenceError> {
    ensure_ort_runtime()?;
    let mut builder = Session::builder().map_err(|e| InferenceError::LoadFailed {
        path: path.to_path_buf(),
        source: e,
    })?;

    let selection = backend::select_backend(backend)?;
    builder = apply_session_config(builder, selection.selected())?;
    if selection.strict_accelerator() {
        builder = builder.with_disable_cpu_fallback().map_err(|err| {
            InferenceError::SessionConfiguration {
                message: err.to_string(),
            }
        })?;
    }
    if !selection.providers().is_empty() {
        builder = builder
            .with_execution_providers(selection.providers())
            .map_err(|err| InferenceError::SessionConfiguration {
                message: err.to_string(),
            })?;
    }

    let session = builder
        .commit_from_file(path)
        .map_err(|e| InferenceError::LoadFailed {
            path: path.to_path_buf(),
            source: e,
        })?;

    Ok((ManagedSession::new(session), selection.selected()))
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
) -> Result<ort::session::builder::SessionBuilder, InferenceError> {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let default_intra = match selected {
        InferenceBackend::Cpu => (cores / 2).max(1),
        _ => 1,
    };
    let intra = env_usize("KIKO_ORT_INTRA_THREADS").unwrap_or(default_intra);
    let inter = env_usize("KIKO_ORT_INTER_THREADS").unwrap_or(1);
    let opt_level = env_opt_level("KIKO_ORT_OPT_LEVEL").unwrap_or(GraphOptimizationLevel::Level3);
    let mem_pattern = env_bool("KIKO_ORT_MEM_PATTERN").unwrap_or(true);
    let parallel_exec = env_bool("KIKO_ORT_PARALLEL_EXEC").unwrap_or(false);

    let configure = |err: ort::Error<_>| InferenceError::SessionConfiguration {
        message: err.to_string(),
    };
    let builder = builder
        .with_optimization_level(opt_level)
        .map_err(configure)?;
    let builder = builder
        .with_memory_pattern(mem_pattern)
        .map_err(configure)?;
    let builder = builder.with_intra_threads(intra).map_err(configure)?;
    let builder = builder.with_inter_threads(inter).map_err(configure)?;
    builder
        .with_parallel_execution(parallel_exec)
        .map_err(configure)
}

fn env_opt_level(key: &str) -> Option<GraphOptimizationLevel> {
    let raw = std::env::var(key).ok()?;
    match raw.trim().to_lowercase().as_str() {
        "disable" | "0" => Some(GraphOptimizationLevel::Disable),
        "1" | "level1" | "basic" => Some(GraphOptimizationLevel::Level1),
        "2" | "level2" | "extended" => Some(GraphOptimizationLevel::Level2),
        "3" | "level3" | "all" => Some(GraphOptimizationLevel::Level3),
        _ => {
            eprintln!("invalid {key}={raw}, ignoring");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{InferenceError, preflight_ort_runtime, run_with_limits};
    use std::time::Duration;

    #[test]
    fn watchdog_timeout_does_not_poison_the_next_run() {
        let timed_out = run_with_limits(
            "pending-test",
            Duration::ZERO,
            Duration::from_millis(1),
            std::future::pending::<Result<(), ort::Error>>(),
        );
        assert!(matches!(
            timed_out,
            Err(InferenceError::WatchdogTimeout { .. })
        ));

        let completed = run_with_limits(
            "ready-test",
            Duration::ZERO,
            Duration::from_millis(50),
            std::future::ready(Ok(())),
        );
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
