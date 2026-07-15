use ort::Error as OrtError;
use ort::logging::LogLevel;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::{RunOptions, Session};
use std::path::PathBuf;
use std::sync::Arc;
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
    SessionBuilder {
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
    BackendProbe {
        backend: InferenceBackend,
        source: OrtError,
    },
    SessionConfiguration {
        backend: InferenceBackend,
        source: OrtError,
    },
    ExecutionProviderRegistration {
        backend: InferenceBackend,
        source: OrtError,
    },
    InputTensor {
        name: &'static str,
        source: OrtError,
    },
    SessionRun {
        model: &'static str,
        source: OrtError,
    },
    OutputTensor {
        name: String,
        source: OrtError,
    },
    UnsupportedModelInterface {
        model: &'static str,
        expected: &'static str,
        actual: String,
    },
    StereoInputDimensionsMismatch {
        left: crate::FrameDimensions,
        right: crate::FrameDimensions,
    },
    InputBatchSizeOverflow {
        dimensions: crate::FrameDimensions,
        batch_size: usize,
    },
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
    DescriptorOutput {
        name: String,
        descriptor_index: usize,
        source: crate::DescriptorError,
    },
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
            InferenceError::SessionBuilder { source }
            | InferenceError::LoadFailed { source, .. }
            | InferenceError::BackendProbe { source, .. }
            | InferenceError::SessionConfiguration { source, .. }
            | InferenceError::ExecutionProviderRegistration { source, .. }
            | InferenceError::InputTensor { source, .. }
            | InferenceError::SessionRun { source, .. }
            | InferenceError::OutputTensor { source, .. } => Some(source),
            InferenceError::ModelFileUnavailable { source, .. } => Some(source),
            InferenceError::Frame(source) => Some(source),
            InferenceError::Downscale(source) => Some(source),
            InferenceError::Detection(source) => Some(source),
            InferenceError::DescriptorOutput { source, .. } => Some(source),
            InferenceError::Match(source) => Some(source),
            InferenceError::GlobalDescriptor(source) => Some(source),
            InferenceError::Environment(source) => Some(source),
            InferenceError::CacheDirectory { source, .. } => Some(source),
            InferenceError::UnexpectedOutput { .. }
            | InferenceError::UnsupportedModelInterface { .. }
            | InferenceError::StereoInputDimensionsMismatch { .. }
            | InferenceError::InputBatchSizeOverflow { .. }
            | InferenceError::BackendUnavailable { .. }
            | InferenceError::InvalidSetting { .. }
            | InferenceError::ThreadPanic { .. }
            | InferenceError::InvariantViolation { .. } => None,
        }
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
            InferenceError::SessionBuilder { source } => {
                write!(f, "failed to create ONNX Runtime session builder: {source}")
            }
            InferenceError::LoadFailed { path, source } => {
                write!(f, "failed to load model at {}: {source}", path.display())
            }
            InferenceError::ModelFileUnavailable { path, source } => write!(
                f,
                "model file is unavailable at {}: {source}",
                path.display()
            ),
            InferenceError::BackendProbe { backend, source } => write!(
                f,
                "failed to query {backend:?} ONNX Runtime provider availability: {source}"
            ),
            InferenceError::SessionConfiguration { backend, source } => write!(
                f,
                "failed to configure ONNX Runtime session for {backend:?}: {source}"
            ),
            InferenceError::ExecutionProviderRegistration { backend, source } => write!(
                f,
                "failed to register the ONNX Runtime provider stack for selected {backend:?} backend: {source}"
            ),
            InferenceError::InputTensor { name, source } => write!(
                f,
                "failed to construct ONNX Runtime input tensor '{name}': {source}"
            ),
            InferenceError::SessionRun { model, source } => {
                write!(f, "ONNX Runtime model '{model}' execution failed: {source}")
            }
            InferenceError::OutputTensor { name, source } => write!(
                f,
                "failed to extract ONNX Runtime output tensor '{name}': {source}"
            ),
            InferenceError::UnsupportedModelInterface {
                model,
                expected,
                actual,
            } => write!(
                f,
                "unsupported {model} model interface: expected {expected}, got {actual}"
            ),
            InferenceError::StereoInputDimensionsMismatch { left, right } => write!(
                f,
                "stereo inference requires equal image dimensions, got left={}x{} and right={}x{}",
                left.width(),
                left.height(),
                right.width(),
                right.height()
            ),
            InferenceError::InputBatchSizeOverflow {
                dimensions,
                batch_size,
            } => write!(
                f,
                "inference batch of {batch_size} images at {}x{} exceeds addressable memory",
                dimensions.width(),
                dimensions.height()
            ),
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
            InferenceError::DescriptorOutput {
                name,
                descriptor_index,
                source,
            } => write!(
                f,
                "invalid descriptor {descriptor_index} in inference output '{name}': {source}"
            ),
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

pub(super) fn extract_tensor<'value, T: ort::value::PrimitiveTensorElementType>(
    value: &'value ort::value::DynValue,
    name: &str,
) -> Result<(&'value ort::value::Shape, &'value [T]), InferenceError> {
    value
        .try_extract_tensor::<T>()
        .map_err(|source| InferenceError::OutputTensor {
            name: name.to_string(),
            source,
        })
}

pub(super) fn exact_i64_output_f32(
    name: &str,
    index: usize,
    value: i64,
) -> Result<f32, InferenceError> {
    let converted = value as f32;
    if converted as i128 == i128::from(value) {
        Ok(converted)
    } else {
        Err(InferenceError::UnexpectedOutput {
            name: name.to_string(),
            expected: "integer values exactly representable as f32".to_string(),
            actual: format!("element {index} is {value}"),
        })
    }
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
    let mut builder =
        Session::builder().map_err(|source| InferenceError::SessionBuilder { source })?;

    let selection = backend::select_backend(backend)?;
    builder = apply_session_config(builder, selection.selected())?;
    let disable_cpu_ep_fallback =
        inference_env(crate::env::try_env_bool("KIKO_ORT_DISABLE_CPU_EP_FALLBACK"))?
            .unwrap_or(false);
    if disable_cpu_ep_fallback {
        builder = builder.with_disable_cpu_fallback().map_err(|source| {
            InferenceError::SessionConfiguration {
                backend: selection.selected(),
                source: source.into(),
            }
        })?;
    }
    if !selection.providers().is_empty() {
        builder = builder
            .with_execution_providers(selection.providers())
            .map_err(|source| InferenceError::ExecutionProviderRegistration {
                backend: selection.selected(),
                source: source.into(),
            })?;
    }
    let session = builder
        .commit_from_file(path)
        .map_err(|e| InferenceError::LoadFailed {
            path: path.to_path_buf(),
            source: e,
        })?;

    eprintln!(
        "ort session policy: model={} requested_backend={backend:?} configured_primary_backend={:?} configured_providers=[{}] strict_backend_registration={} ort_cpu_ep_fallback_disabled={} session_committed=true",
        path.display(),
        selection.selected(),
        selection.provider_names().join(","),
        selection.strict_accelerator(),
        disable_cpu_ep_fallback
    );

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
    let log_level = env_log_level("KIKO_ORT_LOG_LEVEL")?;
    let log_verbosity = env_log_verbosity("KIKO_ORT_LOG_VERBOSITY")?;

    let mut builder = builder
        .with_optimization_level(opt_level)
        .and_then(|b| b.with_memory_pattern(mem_pattern))
        .and_then(|b| b.with_intra_threads(intra))
        .and_then(|b| b.with_inter_threads(inter))
        .and_then(|b| b.with_parallel_execution(parallel_exec))
        .map_err(|source| InferenceError::SessionConfiguration {
            backend: selected,
            source: source.into(),
        })?;
    if let Some(level) = log_level {
        builder = builder
            .with_logger(Arc::new(
                |level, category, id, code_location, message| {
                    let placement_summary = message.contains("Node placements")
                        || message.contains("Node(s) placed on")
                        || message.contains("All nodes placed on");
                    if placement_summary
                        || matches!(
                            level,
                            LogLevel::Warning | LogLevel::Error | LogLevel::Fatal
                        )
                    {
                        eprintln!(
                            "ort[{level:?}] category={category} id={id} location={code_location} {message}"
                        );
                    }
                },
            ))
            .and_then(|builder| builder.with_log_level(level))
            .map_err(|source| InferenceError::SessionConfiguration {
                backend: selected,
                source: source.into(),
            })?;
    }
    if let Some(verbosity) = log_verbosity {
        builder = builder.with_log_verbosity(verbosity).map_err(|source| {
            InferenceError::SessionConfiguration {
                backend: selected,
                source: source.into(),
            }
        })?;
    }
    Ok(builder)
}

pub(super) fn build_run_options(selected: InferenceBackend) -> Result<RunOptions, InferenceError> {
    let mut options = RunOptions::new().map_err(|source| InferenceError::SessionConfiguration {
        backend: selected,
        source,
    })?;
    if let Some(level) = env_log_level("KIKO_ORT_RUN_LOG_LEVEL")? {
        options
            .set_log_level(level)
            .map_err(|source| InferenceError::SessionConfiguration {
                backend: selected,
                source,
            })?;
    }
    if let Some(verbosity) = env_log_verbosity("KIKO_ORT_RUN_LOG_VERBOSITY")? {
        options.set_log_verbosity(verbosity).map_err(|source| {
            InferenceError::SessionConfiguration {
                backend: selected,
                source,
            }
        })?;
    }
    Ok(options)
}

fn env_log_level(key: &'static str) -> Result<Option<LogLevel>, InferenceError> {
    let Some(raw) = inference_env(crate::env::try_env_string(key))? else {
        return Ok(None);
    };
    parse_log_level(key, raw).map(Some)
}

fn env_log_verbosity(key: &'static str) -> Result<Option<i32>, InferenceError> {
    inference_env(crate::env::try_env_usize(key))?
        .map(|value| {
            i32::try_from(value).map_err(|_| InferenceError::InvalidSetting {
                key,
                value: value.to_string(),
                expected: "a non-negative integer within i32 range",
            })
        })
        .transpose()
}

fn parse_log_level(key: &'static str, raw: String) -> Result<LogLevel, InferenceError> {
    let level = match raw.trim().to_ascii_lowercase().as_str() {
        "verbose" => LogLevel::Verbose,
        "info" => LogLevel::Info,
        "warning" | "warn" => LogLevel::Warning,
        "error" => LogLevel::Error,
        "fatal" => LogLevel::Fatal,
        _ => {
            return Err(InferenceError::InvalidSetting {
                key,
                value: raw,
                expected: "verbose, info, warning/warn, error, or fatal",
            });
        }
    };
    Ok(level)
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
    use super::{
        InferenceError, exact_i64_output_f32, output_record_count, parse_log_level,
        parse_opt_level, require_output_elements,
    };
    use ort::logging::LogLevel;

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
    fn session_log_level_parser_is_explicit_and_fail_closed() {
        assert_eq!(
            parse_log_level("KIKO_ORT_LOG_LEVEL", "verbose".to_string()).expect("verbose level"),
            LogLevel::Verbose
        );
        assert!(matches!(
            parse_log_level("KIKO_ORT_LOG_LEVEL", "debug".to_string()),
            Err(InferenceError::InvalidSetting {
                key: "KIKO_ORT_LOG_LEVEL",
                value,
                ..
            }) if value == "debug"
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

    #[test]
    fn ort_operation_errors_preserve_source_and_truthful_context() {
        let errors = [
            InferenceError::InputTensor {
                name: "image",
                source: ort::Error::new("input failure"),
            },
            InferenceError::SessionRun {
                model: "superpoint",
                source: ort::Error::new("run failure"),
            },
            InferenceError::OutputTensor {
                name: "keypoints".to_string(),
                source: ort::Error::new("extraction failure"),
            },
        ];

        for error in &errors {
            assert!(std::error::Error::source(&error).is_some());
        }
        assert!(errors[0].to_string().contains("input tensor 'image'"));
        assert!(
            errors[1]
                .to_string()
                .contains("model 'superpoint' execution")
        );
        assert!(errors[2].to_string().contains("output tensor 'keypoints'"));
    }

    #[test]
    fn i64_to_f32_output_conversion_is_exact_or_rejected() {
        for value in -10_000_i64..=10_000 {
            assert_eq!(
                exact_i64_output_f32("keypoints", 0, value).expect("small integer"),
                value as f32
            );
        }
        assert_eq!(
            exact_i64_output_f32("keypoints", 0, 16_777_218).expect("representable even value"),
            16_777_218.0
        );
        assert!(matches!(
            exact_i64_output_f32("keypoints", 3, 16_777_217),
            Err(InferenceError::UnexpectedOutput {
                name,
                expected,
                actual,
            }) if name == "keypoints"
                && expected.contains("exactly representable")
                && actual == "element 3 is 16777217"
        ));
        assert!(exact_i64_output_f32("keypoints", 0, i64::MAX).is_err());
    }
}
