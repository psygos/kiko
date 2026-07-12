use ort::Error as OrtError;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use std::path::PathBuf;
use std::time::{Duration, Instant};

mod backend;
mod eigenplaces;
mod end2end;
mod lightglue;
mod place;
mod superpoint;

pub use backend::InferenceBackend;
pub use eigenplaces::EigenPlaces;
pub use end2end::{End2EndPipeline, End2EndTimings};
pub use place::PlaceDescriptorExtractor;

#[derive(Debug)]
pub enum InferenceError {
    LoadFailed {
        path: PathBuf,
        source: OrtError,
    },

    Execution(OrtError),

    UnexpectedOutput {
        name: String,
        expected: String,
        actual: String,
    },
    BackendUnavailable {
        requested: InferenceBackend,
        selected: InferenceBackend,
    },
    Environment(crate::env::EnvError),
    InvalidSetting {
        key: &'static str,
        value: String,
        expected: &'static str,
    },
    CacheDirectory {
        path: PathBuf,
        source: std::io::Error,
    },
    Frame(crate::FrameError),
    Downscale(crate::DownscaleError),
    Detection(crate::DetectionError),
    Match(crate::MatchError),
    GlobalDescriptor(crate::loop_closure::GlobalDescriptorError),
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
            InferenceError::LoadFailed { source, .. } | InferenceError::Execution(source) => {
                Some(source)
            }
            InferenceError::Frame(source) => Some(source),
            InferenceError::Downscale(source) => Some(source),
            InferenceError::Detection(source) => Some(source),
            InferenceError::Match(source) => Some(source),
            InferenceError::GlobalDescriptor(source) => Some(source),
            InferenceError::Environment(source) => Some(source),
            InferenceError::CacheDirectory { source, .. } => Some(source),
            InferenceError::UnexpectedOutput { .. }
            | InferenceError::BackendUnavailable { .. }
            | InferenceError::InvalidSetting { .. }
            | InferenceError::ThreadPanic { .. }
            | InferenceError::InvariantViolation { .. } => None,
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

impl From<crate::FrameError> for InferenceError {
    fn from(err: crate::FrameError) -> Self {
        InferenceError::Frame(err)
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
            InferenceError::LoadFailed { path, source } => {
                write!(f, "failed to load model at {}: {source}", path.display())
            }
            InferenceError::Execution(e) => write!(f, "execution error: {e}"),
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
            InferenceError::BackendUnavailable {
                requested,
                selected,
            } => write!(
                f,
                "requested inference backend {requested:?} but selected {selected:?}"
            ),
            InferenceError::Environment(err) => {
                write!(f, "inference environment error: {err}")
            }
            InferenceError::InvalidSetting {
                key,
                value,
                expected,
            } => write!(
                f,
                "invalid inference setting {key}={value:?}; expected {expected}"
            ),
            InferenceError::CacheDirectory { path, source } => write!(
                f,
                "failed to create TensorRT cache directory {}: {source}",
                path.display()
            ),
            InferenceError::Frame(err) => write!(f, "frame error: {err}"),
            InferenceError::Downscale(err) => write!(f, "downscale error: {err}"),
            InferenceError::Detection(err) => write!(f, "detection error: {err}"),
            InferenceError::Match(err) => write!(f, "match error: {err}"),
            InferenceError::GlobalDescriptor(err) => {
                write!(f, "global descriptor error: {err}")
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

pub(super) fn require_output_elements(
    name: &str,
    actual: usize,
    required: usize,
) -> Result<(), InferenceError> {
    if actual < required {
        return Err(InferenceError::UnexpectedOutput {
            name: name.to_string(),
            expected: format!("at least {required} elements"),
            actual: format!("{actual} elements"),
        });
    }
    Ok(())
}

pub(super) fn output_record_count(
    name: &str,
    scalar_count: usize,
    record_width: usize,
) -> Result<usize, InferenceError> {
    if record_width == 0 || scalar_count % record_width != 0 {
        return Err(InferenceError::UnexpectedOutput {
            name: name.to_string(),
            expected: format!("records of {record_width} scalar values"),
            actual: format!("{scalar_count} scalar values"),
        });
    }
    Ok(scalar_count / record_width)
}

pub use lightglue::LightGlue;
pub use superpoint::SuperPoint;

pub(super) fn run_with_slow_call_diagnostics<T>(
    diagnostics: InferenceRunDiagnostics,
    model: &'static str,
    run: impl FnOnce() -> Result<T, InferenceError>,
) -> Result<T, InferenceError> {
    let start = Instant::now();
    let result = run();
    let elapsed = start.elapsed();
    let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
    if elapsed > diagnostics.warn_after {
        eprintln!(
            "slow ONNX inference: model={model} elapsed_ms={elapsed_ms:.1} threshold_ms={}",
            diagnostics.warn_after.as_millis(),
        );
    }
    if diagnostics.timing_enabled {
        eprintln!("inference: model={model} elapsed_ms={elapsed_ms:.1}");
    }
    result
}

#[derive(Clone, Copy, Debug)]
pub(super) struct InferenceRunDiagnostics {
    warn_after: Duration,
    timing_enabled: bool,
}

impl InferenceRunDiagnostics {
    fn try_from_env() -> Result<Self, InferenceError> {
        let default_warn_ms = if cfg!(target_vendor = "apple") {
            300
        } else {
            200
        };
        let warn_ms = inference_env(crate::env::try_env_usize("KIKO_ORT_RUN_WARN_MS"))?
            .unwrap_or(default_warn_ms);
        let warn_ms = u64::try_from(warn_ms).map_err(|_| InferenceError::InvalidSetting {
            key: "KIKO_ORT_RUN_WARN_MS",
            value: warn_ms.to_string(),
            expected: "a millisecond duration within u64 range",
        })?;
        let timing_enabled =
            inference_env(crate::env::try_env_bool("KIKO_INFERENCE_TIMING"))?.unwrap_or(false);
        Ok(Self {
            warn_after: Duration::from_millis(warn_ms),
            timing_enabled,
        })
    }
}

fn build_session(
    path: &std::path::Path,
    backend: InferenceBackend,
) -> Result<(Session, InferenceBackend, InferenceRunDiagnostics), InferenceError> {
    let diagnostics = InferenceRunDiagnostics::try_from_env()?;
    let mut builder = Session::builder().map_err(|e| InferenceError::LoadFailed {
        path: path.to_path_buf(),
        source: e,
    })?;

    let selection = backend::select_backend(backend)?;
    builder = apply_session_config(builder, selection.selected())?;
    if !selection.providers().is_empty() {
        builder = builder
            .with_execution_providers(selection.providers())
            .map_err(|err| InferenceError::Execution(err.into()))?;
    }

    let session = builder
        .commit_from_file(path)
        .map_err(|e| InferenceError::LoadFailed {
            path: path.to_path_buf(),
            source: e,
        })?;

    Ok((session, selection.selected(), diagnostics))
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
    let intra = inference_env(crate::env::try_env_usize("KIKO_ORT_INTRA_THREADS"))?
        .unwrap_or(default_intra);
    let inter = inference_env(crate::env::try_env_usize("KIKO_ORT_INTER_THREADS"))?.unwrap_or(1);
    if intra == 0 {
        return Err(invalid_setting(
            "KIKO_ORT_INTRA_THREADS",
            intra,
            "an integer greater than zero",
        ));
    }
    if inter == 0 {
        return Err(invalid_setting(
            "KIKO_ORT_INTER_THREADS",
            inter,
            "an integer greater than zero",
        ));
    }
    let opt_level = env_opt_level("KIKO_ORT_OPT_LEVEL")?.unwrap_or(GraphOptimizationLevel::Level3);
    let mem_pattern =
        inference_env(crate::env::try_env_bool("KIKO_ORT_MEM_PATTERN"))?.unwrap_or(true);
    let parallel_exec =
        inference_env(crate::env::try_env_bool("KIKO_ORT_PARALLEL_EXEC"))?.unwrap_or(false);

    builder
        .with_optimization_level(opt_level)
        .and_then(|b| b.with_memory_pattern(mem_pattern))
        .and_then(|b| b.with_intra_threads(intra))
        .and_then(|b| b.with_inter_threads(inter))
        .and_then(|b| b.with_parallel_execution(parallel_exec))
        .map_err(|err| InferenceError::Execution(err.into()))
}

fn env_opt_level(key: &'static str) -> Result<Option<GraphOptimizationLevel>, InferenceError> {
    let Some(raw) = inference_env(crate::env::try_env_string(key))? else {
        return Ok(None);
    };
    parse_opt_level(key, raw).map(Some)
}

fn parse_opt_level(
    key: &'static str,
    raw: String,
) -> Result<GraphOptimizationLevel, InferenceError> {
    match raw.trim().to_lowercase().as_str() {
        "disable" | "0" => Ok(GraphOptimizationLevel::Disable),
        "1" | "level1" | "basic" => Ok(GraphOptimizationLevel::Level1),
        "2" | "level2" | "extended" => Ok(GraphOptimizationLevel::Level2),
        "3" | "level3" | "all" => Ok(GraphOptimizationLevel::Level3),
        _ => Err(InferenceError::InvalidSetting {
            key,
            value: raw,
            expected: "disable, 0, 1/level1/basic, 2/level2/extended, or 3/level3/all",
        }),
    }
}

pub(super) fn inference_env<T>(
    result: Result<Option<T>, crate::env::EnvError>,
) -> Result<Option<T>, InferenceError> {
    result.map_err(InferenceError::Environment)
}

fn invalid_setting(
    key: &'static str,
    value: impl ToString,
    expected: &'static str,
) -> InferenceError {
    InferenceError::InvalidSetting {
        key,
        value: value.to_string(),
        expected,
    }
}

#[cfg(test)]
mod tests {
    use super::{InferenceError, output_record_count, parse_opt_level, require_output_elements};

    #[test]
    fn required_output_elements_rejects_truncated_tensors() {
        assert!(matches!(
            require_output_elements("mscores0", 1, 2),
            Err(InferenceError::UnexpectedOutput {
                name,
                expected,
                actual,
            }) if name == "mscores0"
                && expected == "at least 2 elements"
                && actual == "1 elements"
        ));
        assert!(require_output_elements("mscores0", 2, 2).is_ok());
    }

    #[test]
    fn optimization_level_parser_rejects_unknown_values() {
        assert!(matches!(
            parse_opt_level("KIKO_ORT_OPT_LEVEL", "fastest".to_string()),
            Err(InferenceError::InvalidSetting {
                key: "KIKO_ORT_OPT_LEVEL",
                value,
                ..
            }) if value == "fastest"
        ));
    }

    #[test]
    fn output_record_count_rejects_partial_records() {
        assert!(matches!(
            output_record_count("matches0", 3, 2),
            Err(InferenceError::UnexpectedOutput { name, .. }) if name == "matches0"
        ));
        assert_eq!(output_record_count("matches0", 4, 2).expect("pairs"), 2);
    }
}
