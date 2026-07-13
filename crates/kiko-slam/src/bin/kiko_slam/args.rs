use std::path::{Path, PathBuf};

use clap::{Args, ValueEnum};

use kiko_slam::{
    DownscaleFactor, End2EndPipeline, InferenceBackend, InferencePipeline, KeypointLimit,
    LightGlue, SuperPoint, VizDecimation,
};

pub const DEFAULT_MAX_KEYPOINTS: usize = 1024;
const DEFAULT_MAC_CPU_MAX_KEYPOINTS: usize = 512;
const DEFAULT_MAC_CPU_DOWNSCALE: usize = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum RunProfileArg {
    /// Preserve explicit flags and environment defaults.
    #[value(name = "default")]
    Default,
    /// Jetson Orin dataset SLAM: CUDA, FP16 LightGlue, VIO, and realtime BA defaults.
    #[value(name = "jetson")]
    Jetson,
}

#[derive(Args, Clone, Debug)]
pub struct InferenceArgs {
    /// Downscale factor for input images (1 = full resolution)
    #[arg(long, env = "KIKO_DOWNSCALE", default_value = "1")]
    pub downscale: DownscaleFactor,
    /// Maximum number of keypoints to extract per image
    #[arg(long, visible_alias = "keypoints", env = "KIKO_MAX_KEYPOINTS", default_value_t = KeypointLimit::try_from(DEFAULT_MAX_KEYPOINTS).unwrap())]
    pub max_keypoints: KeypointLimit,
    /// Inference backend for all models (overridden by per-model flags)
    #[arg(long, env = "KIKO_BACKEND", value_enum)]
    pub backend: Option<BackendArg>,
    /// Override inference backend for SuperPoint only
    #[arg(
        long,
        visible_alias = "sp-backend",
        env = "KIKO_SUPERPOINT_BACKEND",
        value_enum
    )]
    pub superpoint_backend: Option<BackendArg>,
    /// Override inference backend for LightGlue only
    #[arg(
        long,
        visible_alias = "lg-backend",
        env = "KIKO_LIGHTGLUE_BACKEND",
        value_enum
    )]
    pub lightglue_backend: Option<BackendArg>,
    /// Path to SuperPoint ONNX model
    #[arg(long, visible_alias = "sp-model", env = "KIKO_SUPERPOINT_MODEL")]
    pub superpoint_model: Option<PathBuf>,
    /// Path to LightGlue ONNX model
    #[arg(long, visible_alias = "lg-model", env = "KIKO_LIGHTGLUE_MODEL")]
    pub lightglue_model: Option<PathBuf>,
    /// Path to end-to-end pipeline ONNX model (SP+LG fused, replaces separate models)
    #[arg(long, visible_alias = "pipeline", env = "KIKO_PIPELINE_MODEL")]
    pub pipeline_model: Option<PathBuf>,
}

impl InferenceArgs {
    pub fn with_profile_defaults(
        &self,
        profile: RunProfileArg,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut effective = self.clone();
        match profile {
            RunProfileArg::Default => {}
            RunProfileArg::Jetson => {
                if !any_setting_explicit(&["--backend"], "KIKO_BACKEND") {
                    effective.backend = Some(BackendArg::Cuda);
                }
                if !any_setting_explicit(&["--max-keypoints", "--keypoints"], "KIKO_MAX_KEYPOINTS")
                {
                    effective.max_keypoints = KeypointLimit::try_from(2048)?;
                }
                if !any_setting_explicit(
                    &["--superpoint-model", "--sp-model"],
                    "KIKO_SUPERPOINT_MODEL",
                ) {
                    effective.superpoint_model = Some(PathBuf::from("sp_topk2048.onnx"));
                }
                if !any_setting_explicit(
                    &["--lightglue-model", "--lg-model"],
                    "KIKO_LIGHTGLUE_MODEL",
                ) {
                    effective.lightglue_model =
                        Some(PathBuf::from("superpoint_lightglue_fused_fp16.onnx"));
                }
            }
        }
        Ok(effective)
    }
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
    #[arg(
        long,
        env = "KIKO_RERUN_SAVE",
        visible_alias = "rerun-save",
        group = "rerun_destination"
    )]
    pub save_rrd: Option<PathBuf>,
    /// Stream Rerun data to a remote viewer, e.g. rerun+http://192.168.50.1:9876/proxy
    #[arg(
        long,
        env = "KIKO_RERUN_URL",
        value_name = "URL",
        group = "rerun_destination"
    )]
    pub rerun_url: Option<String>,
    /// Stream to the default laptop Rerun viewer endpoint.
    #[arg(
        long,
        env = "KIKO_RERUN_LAPTOP",
        default_value_t = false,
        group = "rerun_destination"
    )]
    pub rerun_laptop: bool,
    /// Default laptop Rerun endpoint used by --rerun-laptop.
    #[arg(
        long,
        env = "KIKO_RERUN_LAPTOP_URL",
        default_value = "rerun+http://192.168.50.1:9876/proxy",
        hide = true
    )]
    pub rerun_laptop_url: String,
    /// Host a gRPC server on 0.0.0.0 so remote Rerun viewers can connect to this machine
    #[arg(
        long,
        env = "KIKO_RERUN_SERVE",
        default_value_t = false,
        group = "rerun_destination"
    )]
    pub rerun_serve: bool,
    /// Port for the gRPC server when using --rerun-serve (default: 9876)
    #[arg(long, env = "KIKO_RERUN_PORT", default_value_t = 9876)]
    pub rerun_port: u16,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RerunDestination {
    Save(PathBuf),
    Serve { port: u16 },
    Connect(String),
    ImplicitLocalViewer,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RerunOutput {
    destination: RerunDestination,
    decimation: VizDecimation,
}

impl RerunOutput {
    pub fn try_from_args(args: &RerunArgs) -> Result<Self, RerunArgsError> {
        let destination_count = usize::from(args.save_rrd.is_some())
            + usize::from(args.rerun_url.is_some())
            + usize::from(args.rerun_laptop)
            + usize::from(args.rerun_serve);
        if destination_count > 1 {
            return Err(RerunArgsError::ConflictingDestinations { destination_count });
        }

        let destination = if let Some(path) = args.save_rrd.as_ref() {
            RerunDestination::Save(path.clone())
        } else if args.rerun_serve {
            RerunDestination::Serve {
                port: args.rerun_port,
            }
        } else if let Some(url) = args.rerun_url.as_ref() {
            RerunDestination::Connect(url.clone())
        } else if args.rerun_laptop {
            RerunDestination::Connect(args.rerun_laptop_url.clone())
        } else {
            RerunDestination::ImplicitLocalViewer
        };
        Ok(Self {
            destination,
            decimation: args.rerun_decimation,
        })
    }

    pub fn destination(&self) -> &RerunDestination {
        &self.destination
    }

    pub fn decimation(&self) -> VizDecimation {
        self.decimation
    }

    pub fn has_explicit_destination(&self) -> bool {
        !matches!(self.destination, RerunDestination::ImplicitLocalViewer)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RerunArgsError {
    ConflictingDestinations { destination_count: usize },
}

impl std::fmt::Display for RerunArgsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConflictingDestinations { destination_count } => write!(
                f,
                "rerun output has {destination_count} destinations; configure at most one of --save-rrd, --rerun-url, --rerun-laptop, or --rerun-serve"
            ),
        }
    }
}

impl std::error::Error for RerunArgsError {}

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
    pub superpoint_right: PrefetchSession<SuperPoint>,
    pub lightglue: LightGlue,
    pub lightglue_prefetch: PrefetchSession<LightGlue>,
    pub end2end: Option<End2EndPipeline>,
    pub key_limit: KeypointLimit,
    pub downscale: DownscaleFactor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InferencePurpose {
    Benchmark,
    Slam,
    Visualization,
}

impl InferencePurpose {
    fn allows_end_to_end(self) -> bool {
        matches!(self, Self::Benchmark)
    }

    fn uses_speculative_lightglue(self) -> bool {
        matches!(self, Self::Slam)
    }
}

pub enum PrefetchSession<T> {
    Ready(T),
    NotApplicable,
    Unavailable {
        component: &'static str,
        source: kiko_slam::InferenceError,
    },
}

impl<T> PrefetchSession<T> {
    fn from_result(component: &'static str, result: Result<T, kiko_slam::InferenceError>) -> Self {
        match result {
            Ok(session) => Self::Ready(session),
            Err(source) => Self::Unavailable { component, source },
        }
    }

    pub fn is_ready(&self) -> bool {
        matches!(self, Self::Ready(_))
    }

    pub fn into_option(self) -> Option<T> {
        match self {
            Self::Ready(session) => Some(session),
            Self::NotApplicable => None,
            Self::Unavailable { component, source } => {
                eprintln!(
                    "inference prefetch unavailable: component={component}; continuing sequentially: {source}"
                );
                let mut nested = std::error::Error::source(&source);
                while let Some(cause) = nested {
                    eprintln!("  caused by: {cause}");
                    nested = cause.source();
                }
                None
            }
        }
    }
}

impl InferenceConfig {
    pub fn from_args(
        args: &InferenceArgs,
        purpose: InferencePurpose,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if args.pipeline_model.is_some() && !purpose.allows_end_to_end() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "--pipeline-model is supported only by the benchmark command",
            )
            .into());
        }
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
            let pipeline_path = resolve_model_path(&model_dir, Some(pipeline_path), "");
            eprintln!("pipeline model: {}", pipeline_path.display());
            let pipeline = End2EndPipeline::new(&pipeline_path, default_backend)?;
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
            PrefetchSession::from_result(
                "stereo_superpoint",
                SuperPoint::new_with_backend(&sp_path, superpoint_backend),
            )
        } else {
            PrefetchSession::NotApplicable
        };
        let lightglue = LightGlue::new_with_backend(&lg_path, lightglue_backend)?;
        let lightglue_prefetch = if end2end.is_none() && purpose.uses_speculative_lightglue() {
            PrefetchSession::from_result(
                "speculative_lightglue",
                LightGlue::new_with_backend(&lg_path, lightglue_backend),
            )
        } else {
            PrefetchSession::NotApplicable
        };

        eprintln!(
            "inference backend: superpoint={:?}, lightglue={:?}",
            superpoint.backend(),
            lightglue.backend()
        );
        if end2end.is_some() {
            eprintln!("end-to-end pipeline: enabled (SP+LG fused, single call)");
        } else if superpoint_right.is_ready() {
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
            lightglue_prefetch,
            end2end,
            key_limit,
            downscale,
        })
    }

    pub fn into_pipeline(self) -> Result<InferencePipeline, Box<dyn std::error::Error>> {
        if self.end2end.is_some() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "end-to-end inference cannot be converted to the standard inference pipeline",
            )
            .into());
        }
        let mut pipeline = InferencePipeline::new(self.superpoint, self.lightglue, self.key_limit)
            .with_downscale(self.downscale);
        if let Some(sp_right) = self.superpoint_right.into_option() {
            pipeline = pipeline.with_stereo_superpoint(sp_right);
        }
        Ok(pipeline)
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
    any_setting_explicit(&[flag], env_key)
}

fn any_setting_explicit(flags: &[&str], env_key: &str) -> bool {
    if std::env::var_os(env_key).is_some() {
        return true;
    }
    std::env::args_os().any(|arg| {
        let value = arg.to_string_lossy();
        flags
            .iter()
            .any(|flag| value == *flag || value.starts_with(&format!("{flag}=")))
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
