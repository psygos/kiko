use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::Error as OrtError;
use std::path::PathBuf;

mod backend;
mod lightglue;
mod superpoint;

use crate::env::{env_bool, env_usize};

pub use backend::InferenceBackend;

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

    Domain(String),
}
impl std::error::Error for InferenceError {}

impl From<OrtError> for InferenceError {
    fn from(e: OrtError) -> Self {
        InferenceError::Execution(e)
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
            InferenceError::Domain(msg) => write!(f, "domain error: {msg}"),
        }
    }
}
pub use lightglue::LightGlue;
pub use superpoint::SuperPoint;

fn build_session(
    path: &std::path::Path,
    backend: InferenceBackend,
) -> Result<(Session, InferenceBackend), InferenceError> {
    let mut builder = Session::builder().map_err(|e| InferenceError::LoadFailed {
        path: path.to_path_buf(),
        source: e,
    })?;

    let selection = backend::select_backend(backend)?;
    builder = apply_session_config(builder, selection.selected())?;
    if !selection.providers().is_empty() {
        builder = builder
            .with_execution_providers(selection.providers())
            .map_err(InferenceError::Execution)?;
    }

    let session = builder
        .commit_from_file(path)
        .map_err(|e| InferenceError::LoadFailed {
            path: path.to_path_buf(),
            source: e,
        })?;

    Ok((session, selection.selected()))
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

    builder
        .with_optimization_level(opt_level)
        .and_then(|b| b.with_memory_pattern(mem_pattern))
        .and_then(|b| b.with_intra_threads(intra))
        .and_then(|b| b.with_inter_threads(inter))
        .and_then(|b| b.with_parallel_execution(parallel_exec))
        .map_err(InferenceError::Execution)
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
