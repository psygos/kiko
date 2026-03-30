use std::path::PathBuf;

use clap::{Args, ValueEnum};

use kiko_slam::{
    DownscaleFactor, End2EndPipeline, InferenceBackend, InferencePipeline, KeypointLimit,
    LightGlue, SuperPoint, VizDecimation,
};

use std::path::Path;

pub const DEFAULT_MAX_KEYPOINTS: usize = 1024;
const DEFAULT_MAC_CPU_MAX_KEYPOINTS: usize = 512;
const DEFAULT_MAC_CPU_DOWNSCALE: usize = 2;

#[derive(Args, Clone, Debug)]
pub struct InferenceArgs {
    /// Downscale factor for input images (1 = full resolution)
    #[arg(long, env = "KIKO_DOWNSCALE", default_value = "1")]
    pub downscale: DownscaleFactor,
    /// Maximum number of keypoints to extract per image
    #[arg(long, env = "KIKO_MAX_KEYPOINTS", default_value_t = KeypointLimit::try_from(DEFAULT_MAX_KEYPOINTS).unwrap())]
    pub max_keypoints: KeypointLimit,
    /// Inference backend for all models (overridden by per-model flags)
    #[arg(long, env = "KIKO_BACKEND", value_enum)]
    pub backend: Option<BackendArg>,
    /// Override inference backend for SuperPoint only
    #[arg(long, env = "KIKO_SUPERPOINT_BACKEND", value_enum)]
    pub superpoint_backend: Option<BackendArg>,
    /// Override inference backend for LightGlue only
    #[arg(long, env = "KIKO_LIGHTGLUE_BACKEND", value_enum)]
    pub lightglue_backend: Option<BackendArg>,
    /// Path to SuperPoint ONNX model
    #[arg(long, env = "KIKO_SUPERPOINT_MODEL")]
    pub superpoint_model: Option<PathBuf>,
    /// Path to LightGlue ONNX model
    #[arg(long, env = "KIKO_LIGHTGLUE_MODEL")]
    pub lightglue_model: Option<PathBuf>,
    /// Path to end-to-end pipeline ONNX model (SP+LG fused, replaces separate models)
    #[arg(long, env = "KIKO_PIPELINE_MODEL")]
    pub pipeline_model: Option<PathBuf>,
}

#[derive(Args, Clone, Debug)]
pub struct DatasetArgs {
    /// Path to the dataset directory
    #[arg(value_name = "DATASET_PATH")]
    pub path: PathBuf,
    /// Maximum number of stereo pairs to attempt from the dataset
    #[arg(long, env = "KIKO_MAX_PAIRS")]
    pub max_pairs: Option<usize>,
    /// Skip the first N frames (camera/IMU settling time)
    #[arg(long, env = "KIKO_SKIP_FRAMES", default_value_t = 0)]
    pub skip_frames: usize,
}

#[derive(Args, Clone, Debug)]
pub struct RerunArgs {
    /// Log every Nth frame to Rerun (1 = every frame)
    #[arg(long, env = "KIKO_RERUN_DECIMATION", default_value = "1")]
    pub rerun_decimation: VizDecimation,
    /// Save Rerun data to .rrd file instead of streaming
    #[arg(long, env = "KIKO_RERUN_SAVE")]
    pub save_rrd: Option<PathBuf>,
    /// Stream Rerun data to a remote viewer, e.g. rerun+http://192.168.50.1:9876/proxy
    #[arg(long, env = "KIKO_RERUN_URL", value_name = "URL")]
    pub rerun_url: Option<String>,
    /// Host a gRPC server on 0.0.0.0 so remote Rerun viewers can connect to this machine
    #[arg(long, env = "KIKO_RERUN_SERVE", default_value_t = false)]
    pub rerun_serve: bool,
    /// Port for the gRPC server when using --rerun-serve (default: 9876)
    #[arg(long, env = "KIKO_RERUN_PORT", default_value_t = 9876)]
    pub rerun_port: u16,
}

#[derive(Args, Clone, Debug)]
pub struct RectifyArgs {
    /// Maximum principal point delta tolerance (pixels)
    #[arg(long, env = "KIKO_RECTIFY_TOLERANCE")]
    pub rectify_tolerance: Option<f32>,
    /// Allow unrectified calibration data
    #[arg(long, env = "KIKO_ALLOW_UNRECTIFIED", default_value_t = false)]
    pub allow_unrectified: bool,
}

#[derive(Args, Clone, Debug)]
#[cfg(feature = "record")]
pub struct CameraArgs {
    #[arg(long, default_value_t = 640)]
    pub width: u32,
    #[arg(long, default_value_t = 480)]
    pub height: u32,
    #[arg(long, default_value_t = 30)]
    pub fps: u32,
    /// Disable stereo rectification (enabled by default)
    #[arg(long, default_value_t = false)]
    pub no_rectify: bool,
}

#[cfg(feature = "record")]
impl CameraArgs {
    pub fn rectified(&self) -> bool {
        !self.no_rectify
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum BackendArg {
    #[value(name = "auto")]
    Auto,
    #[value(name = "cpu")]
    Cpu,
    #[value(name = "coreml-gpu", alias = "coreml")]
    CoremlGpu,
    #[value(name = "cuda")]
    Cuda,
    #[value(name = "tensorrt", alias = "trt")]
    TensorRt,
}

impl From<BackendArg> for InferenceBackend {
    fn from(value: BackendArg) -> Self {
        match value {
            BackendArg::Auto => InferenceBackend::Auto,
            BackendArg::Cpu => InferenceBackend::Cpu,
            BackendArg::CoremlGpu => InferenceBackend::CoreMLGpu,
            BackendArg::Cuda => InferenceBackend::Cuda,
            BackendArg::TensorRt => InferenceBackend::TensorRT,
        }
    }
}

pub struct InferenceConfig {
    pub superpoint: SuperPoint,
    pub superpoint_right: Option<SuperPoint>,
    pub lightglue: LightGlue,
    pub end2end: Option<End2EndPipeline>,
    pub key_limit: KeypointLimit,
    pub downscale: DownscaleFactor,
}

impl InferenceConfig {
    pub fn from_args(args: &InferenceArgs) -> Result<Self, Box<dyn std::error::Error>> {
        let default_backend = args
            .backend
            .map(InferenceBackend::from)
            .unwrap_or(InferenceBackend::auto());
        let superpoint_backend = args
            .superpoint_backend
            .map(InferenceBackend::from)
            .unwrap_or(default_backend);
        let lightglue_backend = args
            .lightglue_backend
            .map(InferenceBackend::from)
            .unwrap_or(default_backend);

        let model_dir = resolve_model_dir();

        // Check for end-to-end pipeline model
        let end2end = if let Some(ref pipeline_path) = args.pipeline_model {
            eprintln!("pipeline model: {}", pipeline_path.display());
            let pipeline = End2EndPipeline::new(pipeline_path, default_backend)?;
            eprintln!("pipeline backend: {:?}", pipeline.backend());
            Some(pipeline)
        } else {
            None
        };

        let sp_path = resolve_model_path(&model_dir, args.superpoint_model.as_ref(), "sp.onnx");
        let lg_path = resolve_model_path(&model_dir, args.lightglue_model.as_ref(), "lg.onnx");
        eprintln!(
            "models: superpoint={} lightglue={}",
            sp_path.display(),
            lg_path.display()
        );

        let superpoint = SuperPoint::new_with_backend(&sp_path, superpoint_backend)?;
        let superpoint_right = if end2end.is_none() {
            SuperPoint::new_with_backend(&sp_path, superpoint_backend).ok()
        } else {
            None
        };
        let lightglue = LightGlue::new_with_backend(&lg_path, lightglue_backend)?;

        eprintln!(
            "inference backend: superpoint={:?}, lightglue={:?}",
            superpoint.backend(),
            lightglue.backend()
        );
        if end2end.is_some() {
            eprintln!("end-to-end pipeline: enabled (SP+LG fused, single call)");
        } else if superpoint_right.is_some() {
            eprintln!("parallel stereo superpoint: enabled");
        }

        let (downscale, key_limit) =
            tuned_mac_inference_settings(args, superpoint.backend(), lightglue.backend())?;
        eprintln!("downscale: {downscale}");
        eprintln!("max_keypoints: {key_limit}");

        Ok(Self {
            superpoint,
            superpoint_right,
            lightglue,
            end2end,
            key_limit,
            downscale,
        })
    }

    pub fn into_pipeline(self) -> InferencePipeline {
        let mut pipeline = InferencePipeline::new(self.superpoint, self.lightglue, self.key_limit)
            .with_downscale(self.downscale);
        if let Some(sp_right) = self.superpoint_right {
            pipeline = pipeline.with_stereo_superpoint(sp_right);
        }
        pipeline
    }
}

fn tuned_mac_inference_settings(
    args: &InferenceArgs,
    superpoint_backend: InferenceBackend,
    lightglue_backend: InferenceBackend,
) -> Result<(DownscaleFactor, KeypointLimit), Box<dyn std::error::Error>> {
    let mut downscale = args.downscale;
    let mut key_limit = args.max_keypoints;
    if !cfg!(target_vendor = "apple") {
        return Ok((downscale, key_limit));
    }

    let cpu_fallback =
        superpoint_backend == InferenceBackend::Cpu || lightglue_backend == InferenceBackend::Cpu;
    if !cpu_fallback {
        return Ok((downscale, key_limit));
    }

    eprintln!(
        "warning: Apple inference backend fell back to CPU (superpoint={superpoint_backend:?}, lightglue={lightglue_backend:?})"
    );

    let downscale_explicit = setting_explicit("--downscale", "KIKO_DOWNSCALE")
        || args.downscale != DownscaleFactor::identity();
    let key_limit_explicit = setting_explicit("--max-keypoints", "KIKO_MAX_KEYPOINTS")
        || args.max_keypoints != KeypointLimit::try_from(DEFAULT_MAX_KEYPOINTS).expect("default");

    if !downscale_explicit {
        downscale = DownscaleFactor::try_from(DEFAULT_MAC_CPU_DOWNSCALE)?;
    }
    if !key_limit_explicit {
        key_limit = KeypointLimit::try_from(DEFAULT_MAC_CPU_MAX_KEYPOINTS)?;
    }

    if downscale != args.downscale || key_limit != args.max_keypoints {
        eprintln!(
            "mac CPU fallback: adjusting defaults to downscale={downscale} max_keypoints={key_limit}"
        );
    }

    Ok((downscale, key_limit))
}

fn setting_explicit(flag: &str, env_key: &str) -> bool {
    if std::env::var_os(env_key).is_some() {
        return true;
    }
    std::env::args_os().any(|arg| {
        let value = arg.to_string_lossy();
        value == flag || value.starts_with(&format!("{flag}="))
    })
}

/// Resolve the default model directory deterministically.
///
/// Preference order:
/// 1. Compile-time `$CARGO_MANIFEST_DIR/models/`
/// 2. Adjacent to the executable
fn resolve_model_dir() -> PathBuf {
    let manifest_models = Path::new(env!("CARGO_MANIFEST_DIR")).join("models");
    if manifest_models.is_dir() {
        return manifest_models;
    }

    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            let exe_models = exe_dir.join("models");
            if exe_models.is_dir() {
                return exe_models;
            }
        }
    }

    manifest_models
}

fn resolve_model_path(
    model_dir: &Path,
    override_path: Option<&PathBuf>,
    default_name: &str,
) -> PathBuf {
    match override_path {
        Some(candidate) => {
            if candidate.is_absolute() {
                candidate.clone()
            } else {
                model_dir.join(candidate)
            }
        }
        None => model_dir.join(default_name),
    }
}
