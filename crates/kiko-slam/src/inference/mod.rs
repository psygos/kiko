use ort::Error as OrtError;
use std::path::PathBuf;

mod lightglue;
mod superpoint;

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
