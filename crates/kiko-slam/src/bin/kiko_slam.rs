use std::num::{NonZeroU16, NonZeroUsize};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use clap::{Args, Parser, Subcommand, ValueEnum};

use kiko_slam::dataset::{DatasetDepthCursor, DatasetError, DatasetReader};
use kiko_slam::dense::{self, DenseConfig, command_mapper, ring_buffer::DepthRingBuffer};
use kiko_slam::{
    BackendConfig, DenseCommand, DepthImage, DownscaleFactor, FrameId, GlobalDescriptorConfig,
    InferenceBackend, InferencePipeline, KeyframePolicy, KeypointLimit, LightGlue, LmConfig,
    LocalBaConfig, LoopClosureConfig, LoopSubsystemConfig, PipelineError, PipelineTimingError,
    PipelineWallBreakdown, RansacConfig, RectifiedStereo, RectifiedStereoConfig,
    RectifiedStereoError, RedundancyPolicy, RelocalizationConfig, RerunSink, RerunSinkConfig,
    SlamTracker, SuperPoint, TrackerConfig, TriangulationConfig, TriangulationError, Triangulator,
    VizDecimation, VizError, VizFlushError, VizLogError, VizPacket,
};

use kiko_slam::env::{env_bool, env_f32, env_usize};

#[cfg(any(feature = "record", test))]
use kiko_slam::ChannelCapacity;

#[cfg(feature = "record")]
use kiko_slam::env::{EnvError, env_u64};

#[cfg(feature = "record")]
use kiko_slam::{CameraPoint3, DenseStats, Frame, Raw, ReconState, WorldToCamera};

#[cfg(feature = "record")]
use kiko_slam::dataset::{
    Calibration, CameraIntrinsics, DatasetWriteError, DatasetWriter, DepthMeta, ImuMeta, Meta,
    MonoMeta, WriteOutcome,
};
#[cfg(feature = "record")]
use kiko_slam::{
    DenseCommandQueueStatsHandle, DenseCommandReceiver, DenseCommandSendOutcome,
    DenseCommandSender, DiagnosticEvent, DropPolicy, DropReceiver, FrameDiagnostics,
    PairingConfigError, PairingInputError, PairingWindowNs, SendOutcome, SensorId, StereoPair,
    StereoPairer, SystemHealth, TrackerInitError, VizConfigError, bounded_channel,
    dense_command_channel, oak_to_depth_image, oak_to_frame,
};
#[cfg(feature = "record")]
use oak_sys::{
    DepthConfig, DepthError, Device, DeviceConfig, ImageError, ImuConfig, MonoConfig, QueueConfig,
};
#[cfg(feature = "record")]
use std::sync::Arc;
#[cfg(feature = "record")]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "record")]
use std::thread;

const DEFAULT_MAX_KEYPOINTS: usize = 1024;
const DEFAULT_RERUN_PORT: NonZeroU16 =
    NonZeroU16::new(9876).expect("the default Rerun port is nonzero");

// BA defaults (overridable via KIKO_BA_* / KIKO_LM_* env vars)
const DEFAULT_BA_WINDOW: usize = 10;
const DEFAULT_BA_ITERS: usize = 6;
const DEFAULT_BA_MIN_OBS: usize = 8;
const DEFAULT_BA_HUBER_PX: f32 = 3.0;
const DEFAULT_BA_DAMPING: f32 = 1e-3;
const DEFAULT_LM_FACTOR: f32 = 10.0;
const DEFAULT_LM_MIN: f32 = 1e-8;
const DEFAULT_LM_MAX: f32 = 1e4;
const DEFAULT_BA_MOTION_WEIGHT: f32 = 0.0;

// Keyframe policy defaults (overridable via KIKO_KEYFRAME_* env vars)
const DEFAULT_KEYFRAME_PARALLAX_PX: f32 = 40.0;
const DEFAULT_KEYFRAME_COVISIBILITY: f32 = 0.6;
const DEFAULT_KEYFRAME_REDUNDANT_COVISIBILITY: f32 = 0.9;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DepthRingCapacityError {
    key: &'static str,
    value: usize,
}

impl std::fmt::Display for DepthRingCapacityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "environment variable {} must be at least {}, got {}",
            self.key,
            DepthRingCapacity::MINIMUM,
            self.value
        )
    }
}

impl std::error::Error for DepthRingCapacityError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DepthRingCapacity(NonZeroUsize);

impl DepthRingCapacity {
    const MINIMUM: usize = 4;

    fn try_new(key: &'static str, value: usize) -> Result<Self, DepthRingCapacityError> {
        if value < Self::MINIMUM {
            return Err(DepthRingCapacityError { key, value });
        }
        Ok(Self(
            NonZeroUsize::new(value).expect("capacity at or above four is nonzero"),
        ))
    }

    #[cfg(feature = "record")]
    fn minimum() -> Self {
        Self(NonZeroUsize::new(Self::MINIMUM).expect("minimum depth ring capacity is nonzero"))
    }

    fn get(self) -> usize {
        self.0.get()
    }

    #[cfg(any(feature = "record", test))]
    fn from_queue_capacity(capacity: ChannelCapacity) -> Self {
        Self(
            NonZeroUsize::new(capacity.get().max(Self::MINIMUM))
                .expect("typed queue capacity is nonzero"),
        )
    }
}

#[derive(Parser, Debug)]
#[command(name = "kiko-slam", about = "Kiko SLAM tools")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    #[cfg(feature = "record")]
    Record(RecordArgs),
    #[cfg(feature = "record")]
    Live(LiveArgs),
    Viz(VizArgs),
    Bench(BenchArgs),
}

#[derive(Args, Clone, Debug)]
struct InferenceArgs {
    #[arg(long, env = "KIKO_DOWNSCALE", default_value_t = DownscaleArg::default())]
    downscale: DownscaleArg,
    #[arg(long, env = "KIKO_MAX_KEYPOINTS", default_value_t = KeypointLimitArg::default())]
    max_keypoints: KeypointLimitArg,
    #[arg(long, env = "KIKO_BACKEND", value_enum)]
    backend: Option<BackendArg>,
    #[arg(long, env = "KIKO_SUPERPOINT_BACKEND", value_enum)]
    superpoint_backend: Option<BackendArg>,
    #[arg(long, env = "KIKO_LIGHTGLUE_BACKEND", value_enum)]
    lightglue_backend: Option<BackendArg>,
    #[arg(long, env = "KIKO_SUPERPOINT_MODEL")]
    superpoint_model: Option<PathBuf>,
    #[arg(long, env = "KIKO_LIGHTGLUE_MODEL")]
    lightglue_model: Option<PathBuf>,
}

#[derive(Args, Clone, Debug)]
struct DatasetArgs {
    #[arg(value_name = "DATASET_PATH")]
    path: PathBuf,
    #[arg(value_name = "MAX_PAIRS")]
    max_pairs: Option<PairLimitArg>,
}

#[derive(Args, Clone, Debug)]
struct VizArgs {
    #[command(flatten)]
    inference: InferenceArgs,
    #[arg(long, env = "KIKO_RERUN_DECIMATION", default_value_t = VizDecimationArg::default())]
    rerun_decimation: VizDecimationArg,
    #[arg(long, env = "KIKO_RERUN_SAVE")]
    save_rrd: Option<PathBuf>,
    /// Start a gRPC server on 0.0.0.0:<port> so remote Rerun viewers can connect.
    #[arg(long, env = "KIKO_RERUN_SERVE", default_value_t = false)]
    rerun_serve: bool,
    /// Port for gRPC server (used with --rerun-serve). Default: 9876.
    #[arg(long, env = "KIKO_RERUN_PORT")]
    rerun_port: Option<NonZeroU16>,
    /// Timeout passed to Rerun for the configured sink flush, in milliseconds.
    #[arg(
        long,
        env = "KIKO_RERUN_FINISH_TIMEOUT_MS",
        default_value_t = RerunFinishTimeout::default()
    )]
    rerun_finish_timeout_ms: RerunFinishTimeout,
    #[arg(long, env = "KIKO_VIZ_ODOMETRY", default_value_t = false)]
    odometry: bool,
    #[arg(long, env = "KIKO_RECTIFY_TOLERANCE")]
    rectify_tolerance: Option<f32>,
    #[command(flatten)]
    dataset: DatasetArgs,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RerunDestination<'a> {
    Save(&'a Path),
    Serve { port: NonZeroU16 },
    Connect,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RerunDestinationError {
    SaveAndServe,
    PortWithoutServer,
}

impl std::fmt::Display for RerunDestinationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SaveAndServe => write!(
                f,
                "Rerun output cannot save a recording and serve it at the same time"
            ),
            Self::PortWithoutServer => {
                write!(f, "a Rerun port requires Rerun serving to be enabled")
            }
        }
    }
}

impl std::error::Error for RerunDestinationError {}

#[derive(Debug)]
enum RerunSessionError<P> {
    Processing(P),
    Finalization(VizFlushError),
    ProcessingAndFinalization {
        processing: P,
        finalization: VizFlushError,
    },
}

impl<P> RerunSessionError<P> {
    fn processing_error(&self) -> Option<&P> {
        match self {
            Self::Processing(source) => Some(source),
            Self::Finalization(_) => None,
            Self::ProcessingAndFinalization { processing, .. } => Some(processing),
        }
    }

    fn finalization_error(&self) -> Option<&VizFlushError> {
        match self {
            Self::Processing(_) => None,
            Self::Finalization(source) => Some(source),
            Self::ProcessingAndFinalization { finalization, .. } => Some(finalization),
        }
    }
}

impl<P: std::fmt::Display> std::fmt::Display for RerunSessionError<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Processing(source) => write!(f, "Rerun session processing failed: {source}"),
            Self::Finalization(source) => write!(f, "Rerun session finalization failed: {source}"),
            Self::ProcessingAndFinalization {
                processing,
                finalization,
            } => write!(
                f,
                "Rerun session processing failed: {processing}; finalization also failed: {finalization}"
            ),
        }
    }
}

impl<P: std::error::Error + 'static> std::error::Error for RerunSessionError<P> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        if let Some(source) = self.processing_error() {
            Some(source)
        } else {
            self.finalization_error()
                .map(|source| source as &(dyn std::error::Error + 'static))
        }
    }
}

fn combine_rerun_results<T, P>(
    processing: Result<T, P>,
    finalization: Result<(), VizFlushError>,
) -> Result<T, RerunSessionError<P>> {
    match (processing, finalization) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(source), Ok(())) => Err(RerunSessionError::Processing(source)),
        (Ok(_), Err(source)) => Err(RerunSessionError::Finalization(source)),
        (Err(processing), Err(finalization)) => Err(RerunSessionError::ProcessingAndFinalization {
            processing,
            finalization,
        }),
    }
}

fn run_rerun_session<T, P>(
    mut sink: RerunSink,
    timeout: RerunFinishTimeout,
    process: impl FnOnce(&mut RerunSink) -> Result<T, P>,
) -> Result<T, RerunSessionError<P>> {
    let processing = process(&mut sink);
    let finalization = sink.finish_with_timeout(timeout.get());
    combine_rerun_results(processing, finalization)
}

#[derive(Debug)]
enum OdometryVizProcessingError {
    Dataset(DatasetError),
    Packet(VizError),
    Log(VizLogError),
}

impl std::fmt::Display for OdometryVizProcessingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dataset(source) => write!(f, "offline depth replay failed: {source}"),
            Self::Packet(source) => write!(f, "visualization packet creation failed: {source}"),
            Self::Log(source) => write!(f, "visualization logging failed: {source}"),
        }
    }
}

impl std::error::Error for OdometryVizProcessingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Dataset(source) => Some(source),
            Self::Packet(source) => Some(source),
            Self::Log(source) => Some(source),
        }
    }
}

impl From<DatasetError> for OdometryVizProcessingError {
    fn from(source: DatasetError) -> Self {
        Self::Dataset(source)
    }
}

impl From<VizError> for OdometryVizProcessingError {
    fn from(source: VizError) -> Self {
        Self::Packet(source)
    }
}

impl From<VizLogError> for OdometryVizProcessingError {
    fn from(source: VizLogError) -> Self {
        Self::Log(source)
    }
}

impl<'a> RerunDestination<'a> {
    fn parse(
        save_rrd: Option<&'a Path>,
        rerun_serve: bool,
        rerun_port: Option<NonZeroU16>,
    ) -> Result<Self, RerunDestinationError> {
        match (save_rrd, rerun_serve, rerun_port) {
            (Some(_), true, _) => Err(RerunDestinationError::SaveAndServe),
            (Some(path), false, None) => Ok(Self::Save(path)),
            (Some(_), false, Some(_)) | (None, false, Some(_)) => {
                Err(RerunDestinationError::PortWithoutServer)
            }
            (None, true, port) => Ok(Self::Serve {
                port: port.unwrap_or(DEFAULT_RERUN_PORT),
            }),
            (None, false, None) => Ok(Self::Connect),
        }
    }
}

#[derive(Args, Clone, Debug)]
struct BenchArgs {
    #[command(flatten)]
    inference: InferenceArgs,
    #[command(flatten)]
    dataset: DatasetArgs,
}

#[derive(Debug)]
enum BenchError {
    Dataset(DatasetError),
    Pipeline(PipelineError),
    Timing(PipelineTimingError),
    NoPairsProcessed,
    NoNonzeroMatches,
}

impl std::fmt::Display for BenchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dataset(source) => write!(f, "benchmark dataset failure: {source}"),
            Self::Pipeline(source) => write!(f, "benchmark pipeline failure: {source}"),
            Self::Timing(source) => write!(f, "benchmark timing failure: {source}"),
            Self::NoPairsProcessed => write!(f, "benchmark processed no stereo pairs"),
            Self::NoNonzeroMatches => write!(
                f,
                "benchmark produced no nonzero matches; check the models and dataset"
            ),
        }
    }
}

impl std::error::Error for BenchError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Dataset(source) => Some(source),
            Self::Pipeline(source) => Some(source),
            Self::Timing(source) => Some(source),
            Self::NoPairsProcessed | Self::NoNonzeroMatches => None,
        }
    }
}

impl From<DatasetError> for BenchError {
    fn from(source: DatasetError) -> Self {
        Self::Dataset(source)
    }
}

impl From<PipelineError> for BenchError {
    fn from(source: PipelineError) -> Self {
        Self::Pipeline(source)
    }
}

impl From<PipelineTimingError> for BenchError {
    fn from(source: PipelineTimingError) -> Self {
        Self::Timing(source)
    }
}

#[derive(Args, Clone, Debug)]
#[cfg(feature = "record")]
struct CameraArgs {
    #[arg(long, default_value_t = 640)]
    width: u32,
    #[arg(long, default_value_t = 480)]
    height: u32,
    #[arg(long, default_value_t = 30)]
    fps: u32,
    #[arg(long, default_value_t = true)]
    rectified: bool,
}

#[derive(Args, Clone, Debug)]
#[cfg(feature = "record")]
struct RecordArgs {
    #[arg(value_name = "OUTPUT_PATH")]
    output_path: PathBuf,
    #[command(flatten)]
    camera: CameraArgs,
}

#[derive(Args, Clone, Debug)]
#[cfg(feature = "record")]
struct LiveArgs {
    #[command(flatten)]
    camera: CameraArgs,
    #[command(flatten)]
    inference: InferenceArgs,
    #[arg(long, env = "KIKO_RERUN_DECIMATION", default_value_t = VizDecimationArg::default())]
    rerun_decimation: VizDecimationArg,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum BackendArg {
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

#[derive(Clone, Copy, Debug)]
struct DownscaleArg(DownscaleFactor);

impl Default for DownscaleArg {
    fn default() -> Self {
        Self(DownscaleFactor::identity())
    }
}

impl std::str::FromStr for DownscaleArg {
    type Err = String;

    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        let value = raw
            .trim()
            .parse::<usize>()
            .map_err(|_| format!("invalid downscale factor: {raw}"))?;
        DownscaleFactor::try_from(value)
            .map(DownscaleArg)
            .map_err(|err| err.to_string())
    }
}

impl DownscaleArg {
    fn get(self) -> DownscaleFactor {
        self.0
    }
}

impl std::fmt::Display for DownscaleArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.get())
    }
}

#[derive(Clone, Copy, Debug)]
struct KeypointLimitArg(KeypointLimit);

impl Default for KeypointLimitArg {
    fn default() -> Self {
        Self(
            KeypointLimit::try_from(DEFAULT_MAX_KEYPOINTS).unwrap_or_else(|_| KeypointLimit::min()),
        )
    }
}

impl std::str::FromStr for KeypointLimitArg {
    type Err = String;

    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        let value = raw
            .trim()
            .parse::<usize>()
            .map_err(|_| format!("invalid max keypoints: {raw}"))?;
        KeypointLimit::try_from(value)
            .map(KeypointLimitArg)
            .map_err(|err| err.to_string())
    }
}

impl KeypointLimitArg {
    fn limit(self) -> KeypointLimit {
        self.0
    }

    fn value(self) -> usize {
        self.0.get()
    }
}

impl std::fmt::Display for KeypointLimitArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.get())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PairLimitArg(NonZeroUsize);

impl std::str::FromStr for PairLimitArg {
    type Err = String;

    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        let value = raw
            .trim()
            .parse::<usize>()
            .map_err(|_| format!("invalid maximum pair count: {raw}"))?;
        NonZeroUsize::new(value)
            .map(PairLimitArg)
            .ok_or_else(|| "maximum pair count must be nonzero".to_owned())
    }
}

impl PairLimitArg {
    fn get(self) -> usize {
        self.0.get()
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct VizDecimationArg(VizDecimation);

impl std::str::FromStr for VizDecimationArg {
    type Err = String;

    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        let value = raw
            .trim()
            .parse::<usize>()
            .map_err(|_| format!("invalid rerun decimation: {raw}"))?;
        VizDecimation::try_from(value)
            .map(VizDecimationArg)
            .map_err(|err| err.to_string())
    }
}

impl VizDecimationArg {
    fn get(self) -> VizDecimation {
        self.0
    }
}

impl std::fmt::Display for VizDecimationArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.get())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RerunFinishTimeout(Duration);

impl Default for RerunFinishTimeout {
    fn default() -> Self {
        Self(Duration::from_secs(5))
    }
}

impl std::str::FromStr for RerunFinishTimeout {
    type Err = String;

    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        raw.trim()
            .parse::<u64>()
            .map(Duration::from_millis)
            .map(Self)
            .map_err(|_| format!("invalid Rerun finish timeout in milliseconds: {raw}"))
    }
}

impl RerunFinishTimeout {
    fn get(self) -> Duration {
        self.0
    }
}

impl std::fmt::Display for RerunFinishTimeout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.as_millis())
    }
}

struct InferenceConfig {
    superpoint_left: SuperPoint,
    superpoint_right: SuperPoint,
    lightglue: LightGlue,
    key_limit: KeypointLimit,
    downscale: DownscaleFactor,
}

impl InferenceConfig {
    fn from_args(args: &InferenceArgs) -> Result<Self, Box<dyn std::error::Error>> {
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

        let model_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("models");
        let sp_path = resolve_model_path(&model_dir, args.superpoint_model.as_ref(), "sp.onnx");
        let lg_path = resolve_model_path(&model_dir, args.lightglue_model.as_ref(), "lg.onnx");
        eprintln!(
            "models: superpoint={} lightglue={}",
            sp_path.display(),
            lg_path.display()
        );

        let superpoint_left = SuperPoint::new_with_backend(&sp_path, superpoint_backend)?;
        let superpoint_right = SuperPoint::new_with_backend(&sp_path, superpoint_backend)?;
        let lightglue = LightGlue::new_with_backend(&lg_path, lightglue_backend)?;

        eprintln!(
            "inference backend: superpoint={:?}, lightglue={:?}",
            superpoint_left.backend(),
            lightglue.backend()
        );

        let downscale = args.downscale.get();
        let key_limit = args.max_keypoints.limit();
        eprintln!("downscale: {}", downscale.get());
        eprintln!("max_keypoints: {}", args.max_keypoints.value());

        Ok(Self {
            superpoint_left,
            superpoint_right,
            lightglue,
            key_limit,
            downscale,
        })
    }

    fn into_pipeline(self) -> InferencePipeline {
        InferencePipeline::new(
            self.superpoint_left,
            self.superpoint_right,
            self.lightglue,
            self.key_limit,
        )
        .with_downscale(self.downscale)
    }
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.command {
        #[cfg(feature = "record")]
        Command::Record(args) => run_record(args),
        #[cfg(feature = "record")]
        Command::Live(args) => run_live(args),
        Command::Viz(args) => run_viz(args),
        Command::Bench(args) => run_bench(args),
    }
}

fn run_viz(args: VizArgs) -> Result<(), Box<dyn std::error::Error>> {
    let destination =
        RerunDestination::parse(args.save_rrd.as_deref(), args.rerun_serve, args.rerun_port)?;
    let sink_config = RerunSinkConfig::from_environment()?;
    if args.odometry {
        return run_viz_odometry(&args, destination, sink_config);
    }
    run_viz_matches(&args, destination, sink_config)
}

fn build_recording(
    destination: RerunDestination<'_>,
    name: &str,
) -> Result<rerun::RecordingStream, Box<dyn std::error::Error>> {
    match destination {
        RerunDestination::Save(path) => {
            let path = if path.is_dir() {
                path.join(format!("{name}.rrd"))
            } else {
                path.to_path_buf()
            };
            if let Some(parent) = path
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty())
            {
                std::fs::create_dir_all(parent)?;
            }
            eprintln!("rerun: saving to {}", path.display());
            let rec = rerun::RecordingStreamBuilder::new(name).save(&path)?;
            Ok(rec)
        }
        RerunDestination::Serve { port } => {
            let port = port.get();
            eprintln!("rerun: serving gRPC on 0.0.0.0:{port}");
            eprintln!(
                "rerun: connect from laptop with:  rerun --connect rerun+http://192.168.50.2:{port}/proxy"
            );
            let rec = rerun::RecordingStreamBuilder::new(name).serve_grpc_opts(
                "0.0.0.0",
                port,
                Default::default(),
            )?;
            Ok(rec)
        }
        RerunDestination::Connect => Ok(rerun::RecordingStreamBuilder::new(name).connect_grpc()?),
    }
}

fn build_rectified_stereo_config(
    args: &VizArgs,
) -> Result<RectifiedStereoConfig, RectifiedStereoError> {
    let defaults = RectifiedStereoConfig::default();
    RectifiedStereoConfig::try_new(
        args.rectify_tolerance
            .unwrap_or(defaults.max_principal_delta_px()),
        defaults.max_focal_delta_px(),
    )
}

fn run_viz_matches(
    args: &VizArgs,
    destination: RerunDestination<'_>,
    sink_config: RerunSinkConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut reader = DatasetReader::open(&args.dataset.path)?;
    let stats = reader.stats();

    eprintln!("dataset: {}", args.dataset.path.display());
    eprintln!(
        "camera fps: left={:.2?} right={:.2?} paired={:.2?} (left={}, right={}, paired={}, left_orphans={}, right_orphans={})",
        stats.left_fps,
        stats.right_fps,
        stats.paired_fps,
        stats.left_count,
        stats.right_count,
        stats.paired_count,
        stats.left_orphan_count,
        stats.right_orphan_count
    );
    let inference = InferenceConfig::from_args(&args.inference)?;
    let decimation = args.rerun_decimation.get();

    let rectified = RectifiedStereo::from_calibration_with_config(
        reader.calibration(),
        build_rectified_stereo_config(args)?,
    )?;
    let triangulator = Triangulator::new(rectified, TriangulationConfig::default());

    let rec = build_recording(destination, "kiko-slam-dataset")?;
    let sink = RerunSink::from_config(rec, decimation, sink_config);

    let mut pipeline = inference.into_pipeline();

    let start = Instant::now();
    let mut processed = 0usize;
    let mut inference_errors = 0usize;
    let mut read_errors = 0usize;
    let mut triangulation_empty = 0usize;
    let mut triangulation_errors = 0usize;
    let mut triangulated_points = 0usize;
    let mut total_matches = 0usize;

    run_rerun_session(
        sink,
        args.rerun_finish_timeout_ms,
        |sink| -> Result<(), VizLogError> {
            for pair in reader.pairs() {
                let pair = match pair {
                    Ok(pair) => pair,
                    Err(err) => {
                        read_errors += 1;
                        eprintln!("read error: {err}");
                        continue;
                    }
                };

                match pipeline.process_pair(pair) {
                    Ok(packet) => {
                        total_matches += packet.matches().len();
                        let mut keyframe = None;
                        match triangulator.triangulate(packet.matches()) {
                            Ok(result) => {
                                triangulated_points += result.keyframe.landmarks().len();
                                keyframe = Some(result.keyframe);
                            }
                            Err(TriangulationError::NoLandmarks { .. }) => {
                                triangulation_empty += 1;
                            }
                            Err(err) => {
                                triangulation_errors += 1;
                                eprintln!("triangulation error: {err}");
                            }
                        };

                        let points = keyframe.as_ref().map(|kf| kf.landmarks());
                        sink.log_with_points(&packet, points)?;
                        processed += 1;
                    }
                    Err(err) => {
                        inference_errors += 1;
                        eprintln!("inference error: {err}");
                    }
                }

                if let Some(limit) = args.dataset.max_pairs
                    && processed >= limit.get()
                {
                    break;
                }
            }
            Ok(())
        },
    )?;

    let elapsed = start.elapsed().as_secs_f64();
    let fps = if elapsed > 0.0 {
        processed as f64 / elapsed
    } else {
        0.0
    };
    let avg_matches = if processed > 0 {
        total_matches as f64 / processed as f64
    } else {
        0.0
    };
    let avg_triangulated = if processed > 0 {
        triangulated_points as f64 / processed as f64
    } else {
        0.0
    };

    eprintln!(
        "done: processed={processed}, elapsed={elapsed:.2}s, fps={fps:.2}, read_errors={read_errors}, inference_errors={inference_errors}, triangulation_empty={triangulation_empty}, triangulation_errors={triangulation_errors}, triangulated_points={triangulated_points}"
    );
    eprintln!("summary: avg_matches={avg_matches:.1}, avg_triangulated={avg_triangulated:.1}");

    Ok(())
}

#[derive(Debug, Default)]
struct OfflineDepthSelector {
    previous: Option<DepthImage>,
    lookahead: Option<DepthImage>,
}

impl OfflineDepthSelector {
    fn select(
        &mut self,
        timestamp: kiko_slam::Timestamp,
        mut next_at_or_before: impl FnMut(
            kiko_slam::Timestamp,
        ) -> Result<Option<DepthImage>, DatasetError>,
    ) -> Result<Option<DepthImage>, DatasetError> {
        // DatasetReader parses left timestamps as strictly increasing, so the
        // nearest depth can only be the latest predecessor or first successor.
        if self
            .lookahead
            .as_ref()
            .is_some_and(|depth| depth.timestamp() <= timestamp)
        {
            self.previous = self.lookahead.take();
        }

        let cutoff_ns = timestamp
            .as_nanos()
            .checked_add(command_mapper::MAX_ASSOCIATION_WINDOW_NS)
            .unwrap_or(i64::MAX);
        let cutoff = kiko_slam::Timestamp::from_nanos(cutoff_ns);
        while self.lookahead.is_none() {
            let Some(depth) = next_at_or_before(cutoff)? else {
                break;
            };
            if depth.timestamp() <= timestamp {
                self.previous = Some(depth);
            } else {
                self.lookahead = Some(depth);
            }
        }

        let max_delta = u64::try_from(command_mapper::MAX_ASSOCIATION_WINDOW_NS)
            .expect("the depth association window is nonnegative");
        let candidate = match (&self.previous, &self.lookahead) {
            (Some(previous), Some(lookahead)) => {
                let previous_delta = previous
                    .timestamp()
                    .as_nanos()
                    .abs_diff(timestamp.as_nanos());
                let lookahead_delta = lookahead
                    .timestamp()
                    .as_nanos()
                    .abs_diff(timestamp.as_nanos());
                if previous_delta <= lookahead_delta {
                    Some((previous, previous_delta))
                } else {
                    Some((lookahead, lookahead_delta))
                }
            }
            (Some(previous), None) => Some((
                previous,
                previous
                    .timestamp()
                    .as_nanos()
                    .abs_diff(timestamp.as_nanos()),
            )),
            (None, Some(lookahead)) => Some((
                lookahead,
                lookahead
                    .timestamp()
                    .as_nanos()
                    .abs_diff(timestamp.as_nanos()),
            )),
            (None, None) => None,
        };
        Ok(candidate
            .filter(|(_, delta)| *delta <= max_delta)
            .map(|(depth, _)| depth.clone()))
    }
}

struct OfflineDenseState {
    cursor: DatasetDepthCursor,
    selector: OfflineDepthSelector,
    ring: DepthRingBuffer,
    state: dense::DenseState,
    generation: u64,
    last_buffered_depth: Option<FrameId>,
}

enum OfflineDenseReplay {
    Disabled,
    Enabled(Box<OfflineDenseState>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct EmptyOfflineDepthStream;

impl std::fmt::Display for EmptyOfflineDepthStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "offline dense reconstruction requires at least one manifest-indexed depth frame"
        )
    }
}

impl std::error::Error for EmptyOfflineDepthStream {}

struct TrackerDefaults {
    min_keyframe_points: usize,
    refresh_inliers: usize,
    min_inliers: usize,
}

fn build_tracker_config(
    defaults: TrackerDefaults,
    key_limit: KeypointLimit,
    downscale: DownscaleFactor,
) -> Result<TrackerConfig, Box<dyn std::error::Error>> {
    let min_keyframe_points =
        env_usize("KIKO_KEYFRAME_MIN_POINTS")?.unwrap_or(defaults.min_keyframe_points);
    let refresh_inliers =
        env_usize("KIKO_KEYFRAME_REFRESH_INLIERS")?.unwrap_or(defaults.refresh_inliers);
    let parallax_px = env_f32("KIKO_KEYFRAME_PARALLAX_PX")?.unwrap_or(DEFAULT_KEYFRAME_PARALLAX_PX);
    let min_covisibility =
        env_f32("KIKO_KEYFRAME_COVISIBILITY")?.unwrap_or(DEFAULT_KEYFRAME_COVISIBILITY);
    let redundant_covisibility = env_f32("KIKO_KEYFRAME_REDUNDANT_COVISIBILITY")?
        .unwrap_or(DEFAULT_KEYFRAME_REDUNDANT_COVISIBILITY);
    let min_inliers = env_usize("KIKO_TRACK_MIN_INLIERS")?.unwrap_or(defaults.min_inliers);
    let ransac_defaults = RansacConfig::default();
    let ransac = RansacConfig::try_new(
        ransac_defaults.max_iterations(),
        ransac_defaults.reprojection_threshold_px(),
        min_inliers,
        ransac_defaults.seed(),
    )?;
    let ba_config = build_ba_config()?;
    let keyframe_policy = KeyframePolicy::new(refresh_inliers, parallax_px, min_covisibility)?;
    let redundancy = Some(RedundancyPolicy::new(redundant_covisibility)?);
    let backend = if env_bool("KIKO_BACKEND_ASYNC")?.unwrap_or(true) {
        Some(BackendConfig::new(
            env_usize("KIKO_BACKEND_QUEUE_DEPTH")?.unwrap_or(2),
        )?)
    } else {
        None
    };
    let loop_closure_enabled = env_bool("KIKO_LOOP_CLOSURE")?.unwrap_or(true);
    let learned_descriptors_enabled = if loop_closure_enabled {
        env_bool("KIKO_LEARNED_DESCRIPTORS")?.unwrap_or(true)
    } else {
        false
    };
    let relocalization_enabled = if loop_closure_enabled {
        env_bool("KIKO_RELOCALIZATION")?.unwrap_or(true)
    } else {
        false
    };
    let loop_subsystem = if loop_closure_enabled {
        if !learned_descriptors_enabled {
            return Err("invalid tracker config: loop closure requires learned descriptors".into());
        }
        let loop_cfg = LoopClosureConfig::default();
        let descriptor_cfg =
            GlobalDescriptorConfig::new(env_usize("KIKO_DESCRIPTOR_QUEUE_DEPTH")?.unwrap_or(2))?;
        let relocalization = relocalization_enabled.then_some(RelocalizationConfig::default());
        LoopSubsystemConfig::enabled(loop_cfg, descriptor_cfg, relocalization)
    } else {
        LoopSubsystemConfig::Disabled
    };

    eprintln!(
        "tracker requested: keyframe_min_points={min_keyframe_points} refresh_inliers={refresh_inliers} parallax_px={parallax_px:.1} min_covisibility={min_covisibility:.2} redundant_covisibility={redundant_covisibility:.2} min_inliers={min_inliers} downscale={} max_keypoints={} loop_closure_requested={} learned_descriptors_requested={} relocalization_requested={}",
        downscale.get(),
        key_limit.get(),
        loop_closure_enabled,
        learned_descriptors_enabled,
        relocalization_enabled,
    );

    Ok(TrackerConfig {
        max_keypoints: key_limit,
        downscale,
        min_keyframe_points,
        ransac,
        triangulation: TriangulationConfig::default(),
        keyframe_policy,
        ba: ba_config,
        redundancy,
        backend,
        loop_subsystem,
    })
}

fn report_tracker_runtime(config: &TrackerConfig, tracker: &SlamTracker) {
    eprintln!(
        "tracker runtime: loop_closure_enabled={} learned_descriptors_enabled={} relocalization_enabled={}",
        config.loop_subsystem.is_enabled(),
        tracker.system_health().descriptor.is_alive(),
        config.loop_subsystem.relocalization().is_some(),
    );
}

fn run_viz_odometry(
    args: &VizArgs,
    destination: RerunDestination<'_>,
    sink_config: RerunSinkConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut reader = DatasetReader::open(&args.dataset.path)?;
    let stats = reader.stats();

    eprintln!("dataset: {}", args.dataset.path.display());
    eprintln!(
        "camera fps: left={:.2?} right={:.2?} paired={:.2?} (left={}, right={}, paired={}, left_orphans={}, right_orphans={})",
        stats.left_fps,
        stats.right_fps,
        stats.paired_fps,
        stats.left_count,
        stats.right_count,
        stats.paired_count,
        stats.left_orphan_count,
        stats.right_orphan_count
    );
    let mut offline_dense = if env_bool("KIKO_DENSE")?.unwrap_or(false) {
        let depth_ring_capacity = DepthRingCapacity::try_new(
            "KIKO_OFFLINE_DEPTH_RING_CAPACITY",
            env_usize("KIKO_OFFLINE_DEPTH_RING_CAPACITY")?.unwrap_or(8),
        )?;
        let cursor = reader.depth_cursor()?;
        if cursor.is_empty() {
            return Err(EmptyOfflineDepthStream.into());
        }
        eprintln!(
            "offline dense enabled: manifest_depth_frames={} ring_capacity={}",
            cursor.len(),
            depth_ring_capacity.get()
        );
        OfflineDenseReplay::Enabled(Box::new(OfflineDenseState {
            cursor,
            selector: OfflineDepthSelector::default(),
            ring: DepthRingBuffer::try_new(depth_ring_capacity.get())?,
            state: dense::DenseState::new(&DenseConfig::default()),
            generation: 0,
            last_buffered_depth: None,
        }))
    } else {
        OfflineDenseReplay::Disabled
    };

    let inference = InferenceConfig::from_args(&args.inference)?;
    let decimation = args.rerun_decimation.get();

    let rectified = RectifiedStereo::from_calibration_with_config(
        reader.calibration(),
        build_rectified_stereo_config(args)?,
    )?;
    let InferenceConfig {
        superpoint_left,
        superpoint_right,
        lightglue,
        key_limit,
        downscale,
    } = inference;

    let tracker_config = build_tracker_config(
        TrackerDefaults {
            min_keyframe_points: 12,
            refresh_inliers: 12,
            min_inliers: 8,
        },
        key_limit,
        downscale,
    )?;

    let mut tracker = SlamTracker::try_new(
        superpoint_left,
        superpoint_right,
        lightglue,
        rectified,
        tracker_config,
    )?;
    report_tracker_runtime(&tracker_config, &tracker);
    let rec = build_recording(destination, "kiko-slam-dataset-odometry")?;
    let sink = RerunSink::from_config(rec, decimation, sink_config);

    let start = Instant::now();
    let mut processed = 0usize;
    let mut inference_errors = 0usize;
    let mut read_errors = 0usize;
    let mut poses_logged = 0usize;
    let mut keyframes = 0usize;

    run_rerun_session(
        sink,
        args.rerun_finish_timeout_ms,
        |sink| -> Result<(), OdometryVizProcessingError> {
            for pair in reader.pairs() {
                let pair = match pair {
                    Ok(pair) => pair,
                    Err(err) => {
                        read_errors += 1;
                        eprintln!("read error: {err}");
                        continue;
                    }
                };

                let left = pair.left().clone();
                let right = pair.right().clone();
                let selected_depth = match &mut offline_dense {
                    OfflineDenseReplay::Disabled => None,
                    OfflineDenseReplay::Enabled(dense) => {
                        let OfflineDenseState {
                            cursor,
                            selector,
                            ring,
                            last_buffered_depth,
                            ..
                        } = dense.as_mut();
                        let depth = selector
                            .select(left.timestamp(), |cutoff| cursor.next_at_or_before(cutoff))?;
                        if let Some(depth) = depth.as_ref()
                            && *last_buffered_depth != Some(depth.frame_id())
                        {
                            ring.push(depth.clone());
                            *last_buffered_depth = Some(depth.frame_id());
                        }
                        depth
                    }
                };

                match tracker.process(pair) {
                    Ok(mut output) => {
                        let timestamp = left.timestamp();
                        let dense_stats = match &mut offline_dense {
                            OfflineDenseReplay::Disabled => None,
                            OfflineDenseReplay::Enabled(dense) => {
                                let OfflineDenseState {
                                    ring,
                                    state,
                                    generation,
                                    ..
                                } = dense.as_mut();
                                output.diagnostics_mut().depth_reorder_warnings =
                                    Some(ring.reorder_warnings());
                                let correction = tracker.take_pending_loop_correction();
                                let cmds = command_mapper::map_output_to_dense_commands(
                                    &output,
                                    correction.as_deref(),
                                    ring,
                                    timestamp,
                                    generation,
                                );
                                cmds.into_iter()
                                    .map(|cmd| dense::process_dense_command(state, cmd))
                                    .last()
                            }
                        };
                        if let Some(depth) = selected_depth.as_ref() {
                            sink.log_depth(depth)?;
                        }
                        if let Some(matches) = output.take_stereo_matches() {
                            let points = output
                                .keyframe()
                                .map(|kf| kf.landmarks())
                                .filter(|pts| !pts.is_empty());
                            let packet = VizPacket::try_new(left.clone(), right.clone(), matches)?;
                            sink.log_with_points(&packet, points)?;
                            if output.keyframe().is_some() {
                                keyframes += 1;
                                let snapshot = tracker.covisibility_snapshot();
                                sink.log_covisibility_graph(left.timestamp(), &snapshot)?;
                            }
                        } else {
                            sink.log_frames(&left, &right)?;
                        }

                        if let Some(pose) = output.pose() {
                            sink.log_pose(timestamp, &pose)?;
                            poses_logged += 1;
                        }
                        sink.log_system_health(timestamp, output.health())?;
                        sink.log_diagnostics(timestamp, output.diagnostics())?;
                        for event in output.events() {
                            sink.log_event(timestamp, event)?;
                        }
                        if let Some(stats) = dense_stats.as_ref() {
                            sink.log_dense_stats(timestamp, stats)?;
                        }
                        processed += 1;
                    }
                    Err(err) => {
                        inference_errors += 1;
                        if let OfflineDenseReplay::Enabled(dense) = &mut offline_dense
                            && let Some(correction) = tracker.take_pending_loop_correction()
                        {
                            let state = &mut dense.state;
                            let generation = &mut dense.generation;
                            *generation = generation
                                .checked_add(1)
                                .expect("dense rebuild generation space exhausted");
                            let stats = dense::process_dense_command(
                                state,
                                DenseCommand::RebuildFromSnapshot {
                                    corrected_poses: correction,
                                    generation: *generation,
                                },
                            );
                            sink.log_dense_stats(left.timestamp(), &stats)?;
                        }
                        eprintln!("tracker error: {err}");
                    }
                }

                if let Some(limit) = args.dataset.max_pairs
                    && processed >= limit.get()
                {
                    break;
                }
            }
            Ok(())
        },
    )?;

    let elapsed = start.elapsed().as_secs_f64();
    let fps = if elapsed > 0.0 {
        processed as f64 / elapsed
    } else {
        0.0
    };

    eprintln!(
        "done: processed={processed}, elapsed={elapsed:.2}s, fps={fps:.2}, read_errors={read_errors}, tracker_errors={inference_errors}, poses_logged={poses_logged}, keyframes={keyframes}"
    );

    Ok(())
}

fn run_bench(args: BenchArgs) -> Result<(), Box<dyn std::error::Error>> {
    let dataset_path = &args.dataset.path;
    let open_start = Instant::now();
    let mut reader = DatasetReader::open(dataset_path).map_err(BenchError::from)?;
    let open_time = open_start.elapsed();

    let stats_start = Instant::now();
    let stats = reader.stats();
    let stats_time = stats_start.elapsed();

    eprintln!("dataset: {}", dataset_path.display());
    eprintln!("dataset open: {:.2}ms", open_time.as_secs_f64() * 1000.0);
    eprintln!("dataset stats: {:.2}ms", stats_time.as_secs_f64() * 1000.0);
    eprintln!(
        "camera fps: left={:.2?} right={:.2?} paired={:.2?} (left={}, right={}, paired={}, left_orphans={}, right_orphans={})",
        stats.left_fps,
        stats.right_fps,
        stats.paired_fps,
        stats.left_count,
        stats.right_count,
        stats.paired_count,
        stats.left_orphan_count,
        stats.right_orphan_count
    );

    let inference = InferenceConfig::from_args(&args.inference)?;
    let mut pipeline = inference.into_pipeline();

    let cpu_start = process_usage();
    let mut processed = 0usize;
    let mut matches_nonzero = 0usize;
    let mut total_matches = 0usize;
    let mut sum_read_left = Duration::ZERO;
    let mut sum_read_right = Duration::ZERO;
    let mut sum_pairing = Duration::ZERO;
    let mut sum_read_bytes = 0usize;
    let mut sum_sp_left = Duration::ZERO;
    let mut sum_sp_right = Duration::ZERO;
    let mut sum_detector_wall = Duration::ZERO;
    let mut sum_lightglue = Duration::ZERO;
    let mut sum_total = Duration::ZERO;

    let start = Instant::now();
    for sample in reader.timed_pairs() {
        let sample = sample.map_err(BenchError::from)?;
        let pair = sample.pair;
        sum_read_left += sample.timings.left_read;
        sum_read_right += sample.timings.right_read;
        sum_pairing += sample.timings.pairing;
        sum_read_bytes += sample.timings.left_bytes + sample.timings.right_bytes;

        let (packet, timings) = pipeline
            .process_pair_timed(pair)
            .map_err(BenchError::from)?;
        let matches = packet.matches();
        if !matches.is_empty() {
            matches_nonzero += 1;
            total_matches += matches.len();
        }
        sum_sp_left += timings.superpoint_left;
        sum_sp_right += timings.superpoint_right;
        sum_detector_wall += timings.detector_wall();
        sum_lightglue += timings.lightglue;
        sum_total += timings.total;
        processed += 1;

        if let Some(limit) = args.dataset.max_pairs
            && processed >= limit.get()
        {
            break;
        }
    }
    let elapsed = start.elapsed();
    let cpu_end = process_usage();
    let elapsed_s = elapsed.as_secs_f64();
    let fps = if elapsed_s > 0.0 {
        processed as f64 / elapsed_s
    } else {
        0.0
    };
    let infer_s = sum_total.as_secs_f64();
    let infer_fps = if infer_s > 0.0 {
        processed as f64 / infer_s
    } else {
        0.0
    };

    let match_rate = if processed > 0 {
        matches_nonzero as f64 / processed as f64
    } else {
        0.0
    };
    let avg_matches = if matches_nonzero > 0 {
        total_matches as f64 / matches_nonzero as f64
    } else {
        0.0
    };

    let read_total = sum_read_left + sum_read_right + sum_pairing;
    let read_s = read_total.as_secs_f64();
    let read_fps = if read_s > 0.0 {
        processed as f64 / read_s
    } else {
        0.0
    };
    let read_mb_s = if read_s > 0.0 {
        (sum_read_bytes as f64 / (1024.0 * 1024.0)) / read_s
    } else {
        0.0
    };

    eprintln!("pipeline fps: {fps:.2} (processed={processed}, elapsed={elapsed_s:.2}s)");
    eprintln!("reader fps: {read_fps:.2} (read_time={read_s:.2}s, throughput={read_mb_s:.2} MB/s)");
    eprintln!("inference fps: {infer_fps:.2} (sum_infer_time={infer_s:.2}s)");
    eprintln!(
        "matching: nonzero_pairs={matches_nonzero}, match_rate={match_rate:.2} avg_matches={avg_matches:.1}"
    );
    if processed > 0 {
        let denom = processed as f64;
        let avg_sp_left_ms = (sum_sp_left.as_secs_f64() * 1000.0) / denom;
        let avg_sp_right_ms = (sum_sp_right.as_secs_f64() * 1000.0) / denom;
        let breakdown =
            PipelineWallBreakdown::try_from_totals(sum_detector_wall, sum_lightglue, sum_total)
                .map_err(BenchError::from)?;
        let avg_detector_ms = (breakdown.detector().as_secs_f64() * 1000.0) / denom;
        let avg_lightglue_ms = (breakdown.lightglue().as_secs_f64() * 1000.0) / denom;
        let avg_overhead_ms = (breakdown.overhead().as_secs_f64() * 1000.0) / denom;
        let avg_total_ms = (breakdown.total().as_secs_f64() * 1000.0) / denom;
        let total_seconds = breakdown.total().as_secs_f64().max(f64::MIN_POSITIVE);
        let pct_detector = (breakdown.detector().as_secs_f64() / total_seconds) * 100.0;
        let pct_lightglue = (breakdown.lightglue().as_secs_f64() / total_seconds) * 100.0;
        let pct_overhead = (breakdown.overhead().as_secs_f64() / total_seconds) * 100.0;

        eprintln!(
            "timings avg ms: sp_left_worker={avg_sp_left_ms:.2} sp_right_worker={avg_sp_right_ms:.2} detector_wall={avg_detector_ms:.2} lightglue={avg_lightglue_ms:.2} overhead={avg_overhead_ms:.2} total={avg_total_ms:.2}"
        );
        eprintln!(
            "timings wall pct: detector={pct_detector:.1}% lightglue={pct_lightglue:.1}% overhead={pct_overhead:.1}%"
        );
    }

    if let (Some(start_usage), Some(end_usage)) = (cpu_start, cpu_end) {
        let cpu_time = end_usage.cpu_time.saturating_sub(start_usage.cpu_time);
        let cpu_s = cpu_time.user.as_secs_f64() + cpu_time.sys.as_secs_f64();
        let cpu_pct = if elapsed_s > 0.0 {
            (cpu_s / elapsed_s) * 100.0
        } else {
            0.0
        };
        eprintln!(
            "cpu: user={:.2}ms sys={:.2}ms total={:.2}ms cpu%={:.1}",
            cpu_time.user.as_secs_f64() * 1000.0,
            cpu_time.sys.as_secs_f64() * 1000.0,
            cpu_s * 1000.0,
            cpu_pct
        );
        if let Some(rss) = end_usage.max_rss_bytes {
            eprintln!("memory: max_rss={:.2} MB", (rss as f64) / (1024.0 * 1024.0));
        }
    }

    if processed == 0 {
        return Err(BenchError::NoPairsProcessed.into());
    }
    if matches_nonzero == 0 {
        return Err(BenchError::NoNonzeroMatches.into());
    }

    Ok(())
}

#[cfg(feature = "record")]
const DEFAULT_PAIRING_WINDOW_NS: u64 = 5_000_000;
#[cfg(feature = "record")]
const DEFAULT_PAIRER_MAX_PENDING_PER_SIDE: usize = 64;

#[cfg(feature = "record")]
#[derive(Debug)]
enum PairingWindowLoadError {
    Environment(EnvError),
    InvalidWindow(PairingConfigError),
}

#[cfg(feature = "record")]
impl std::fmt::Display for PairingWindowLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Environment(source) => write!(f, "invalid pairing environment: {source}"),
            Self::InvalidWindow(source) => write!(f, "invalid pairing window: {source}"),
        }
    }
}

#[cfg(feature = "record")]
impl std::error::Error for PairingWindowLoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Environment(source) => Some(source),
            Self::InvalidWindow(source) => Some(source),
        }
    }
}

#[cfg(feature = "record")]
impl From<EnvError> for PairingWindowLoadError {
    fn from(source: EnvError) -> Self {
        Self::Environment(source)
    }
}

#[cfg(feature = "record")]
impl From<PairingConfigError> for PairingWindowLoadError {
    fn from(source: PairingConfigError) -> Self {
        Self::InvalidWindow(source)
    }
}

#[cfg(feature = "record")]
fn load_pairing_window() -> Result<PairingWindowNs, PairingWindowLoadError> {
    let window_ns = env_u64("KIKO_PAIRING_WINDOW_NS")?.unwrap_or(DEFAULT_PAIRING_WINDOW_NS);
    Ok(PairingWindowNs::try_from_u64(window_ns)?)
}

#[cfg(feature = "record")]
fn load_pairer_max_pending_per_side() -> Result<usize, EnvError> {
    Ok(env_usize("KIKO_PAIRER_MAX_PENDING_PER_SIDE")?
        .unwrap_or(DEFAULT_PAIRER_MAX_PENDING_PER_SIDE))
}

#[cfg(feature = "record")]
#[derive(Clone, Copy, Debug)]
enum RecordItem {
    DepthFrame,
    StereoPair,
}

#[cfg(feature = "record")]
impl std::fmt::Display for RecordItem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DepthFrame => write!(f, "depth frame"),
            Self::StereoPair => write!(f, "stereo pair"),
        }
    }
}

#[cfg(feature = "record")]
#[derive(Debug)]
enum RecordCaptureError {
    LeftImage {
        source: ImageError,
    },
    RightImage {
        source: ImageError,
    },
    Depth {
        source: DepthError,
    },
    PairingInput {
        source: PairingInputError,
    },
    DatasetWrite {
        item: RecordItem,
        source: DatasetWriteError,
    },
    DatasetDropped {
        item: RecordItem,
    },
    DatasetWriterFailed {
        item: RecordItem,
    },
}

#[cfg(feature = "record")]
impl std::fmt::Display for RecordCaptureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LeftImage { source } => write!(f, "left camera capture failed: {source}"),
            Self::RightImage { source } => write!(f, "right camera capture failed: {source}"),
            Self::Depth { source } => write!(f, "depth camera capture failed: {source}"),
            Self::PairingInput { source } => {
                write!(f, "stereo pairing input failed: {source}")
            }
            Self::DatasetWrite { item, source } => {
                write!(f, "dataset writer rejected {item}: {source}")
            }
            Self::DatasetDropped { item } => write!(f, "dataset writer dropped {item}"),
            Self::DatasetWriterFailed { item } => {
                write!(f, "dataset writer failed while enqueueing {item}")
            }
        }
    }
}

#[cfg(feature = "record")]
impl std::error::Error for RecordCaptureError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::LeftImage { source } | Self::RightImage { source } => Some(source),
            Self::Depth { source } => Some(source),
            Self::PairingInput { source } => Some(source),
            Self::DatasetWrite { source, .. } => Some(source),
            Self::DatasetDropped { .. } | Self::DatasetWriterFailed { .. } => None,
        }
    }
}

#[cfg(feature = "record")]
#[derive(Debug)]
enum RecordError {
    Capture {
        source: RecordCaptureError,
    },
    Finalization {
        source: Box<DatasetError>,
    },
    CaptureAndFinalization {
        capture: RecordCaptureError,
        finalization: Box<DatasetError>,
    },
}

#[cfg(feature = "record")]
impl std::fmt::Display for RecordError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Capture { source } => write!(f, "recording failed: {source}"),
            Self::Finalization { source } => {
                write!(f, "dataset finalization failed: {source}")
            }
            Self::CaptureAndFinalization {
                capture,
                finalization,
            } => write!(
                f,
                "recording failed ({capture}); dataset finalization also failed: {finalization}"
            ),
        }
    }
}

#[cfg(feature = "record")]
impl std::error::Error for RecordError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Capture { source } => Some(source),
            Self::Finalization { source } => Some(source.as_ref()),
            Self::CaptureAndFinalization { capture, .. } => Some(capture),
        }
    }
}

#[cfg(feature = "record")]
fn require_record_write(
    outcome: Result<WriteOutcome, DatasetWriteError>,
    item: RecordItem,
) -> Result<(), RecordCaptureError> {
    let outcome = outcome.map_err(|source| RecordCaptureError::DatasetWrite { item, source })?;
    match outcome {
        WriteOutcome::Enqueued => Ok(()),
        WriteOutcome::Dropped => Err(RecordCaptureError::DatasetDropped { item }),
        WriteOutcome::WriterFailed => Err(RecordCaptureError::DatasetWriterFailed { item }),
    }
}

#[cfg(feature = "record")]
fn run_record(args: RecordArgs) -> Result<(), Box<dyn std::error::Error>> {
    let output_path = &args.output_path;

    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        eprintln!("\nreceived ctrl+c, stopping...");
        r.store(false, Ordering::SeqCst);
    })?;

    let mono_config = MonoConfig {
        width: args.camera.width,
        height: args.camera.height,
        fps: args.camera.fps,
        rectified: args.camera.rectified,
    };
    let depth_enabled = env_bool("KIKO_RECORD_DEPTH")?.unwrap_or(false);
    let depth_config = depth_enabled.then_some(DepthConfig {
        width: mono_config.width,
        height: mono_config.height,
        fps: mono_config.fps,
        align_to_rgb: false,
    });

    let config = DeviceConfig {
        rgb: None,
        mono: Some(mono_config),
        depth: depth_config,
        imu: None,
        queue: QueueConfig {
            size: 8,
            blocking: false,
        },
    };

    eprintln!("connecting to oak-d...");
    let mut device = Device::connect("", config)?;
    let baseline_m = device.stereo_baseline_m();

    let meta = build_meta(&mono_config, depth_config.as_ref(), None);
    let calibration = build_calibration(&device, baseline_m, &mono_config);
    let pairing_window = load_pairing_window()?;

    eprintln!("creating dataset at {}", output_path.display());
    let (writer, writer_handle) =
        DatasetWriter::create_paired(output_path, &meta, &calibration, pairing_window)?;

    let mut pair_count = 0u64;
    let mut left_count = 0u64;
    let mut right_count = 0u64;
    let mut depth_count = 0u64;
    let mut left_seq = 0u64;
    let mut right_seq = 0u64;
    let pairer_max_pending = load_pairer_max_pending_per_side()?;
    let mut pairer = StereoPairer::new_with_max_pending(pairing_window, pairer_max_pending)?;
    let start = Instant::now();
    let mut capture_error = None;

    eprintln!("recording... press ctrl+c to stop");

    'capture: while running.load(Ordering::Relaxed) {
        let mut got_any = false;

        match device.mono_left(0) {
            Ok(frame) => match oak_to_frame(frame, SensorId::StereoLeft, FrameId::new(left_seq)) {
                Ok(frame) => {
                    if let Err(source) = pairer.push_left(frame) {
                        capture_error = Some(RecordCaptureError::PairingInput { source });
                        break 'capture;
                    }
                    left_count += 1;
                    left_seq += 1;
                    got_any = true;
                }
                Err(err) => {
                    eprintln!("left frame dropped (invalid dimensions): {err}");
                }
            },
            Err(ImageError::Timeout { .. } | ImageError::QueueEmpty) => {}
            Err(source) => {
                capture_error = Some(RecordCaptureError::LeftImage { source });
                break 'capture;
            }
        }

        match device.mono_right(0) {
            Ok(frame) => {
                match oak_to_frame(frame, SensorId::StereoRight, FrameId::new(right_seq)) {
                    Ok(frame) => {
                        if let Err(source) = pairer.push_right(frame) {
                            capture_error = Some(RecordCaptureError::PairingInput { source });
                            break 'capture;
                        }
                        right_count += 1;
                        right_seq += 1;
                        got_any = true;
                    }
                    Err(err) => {
                        eprintln!("right frame dropped (invalid dimensions): {err}");
                    }
                }
            }
            Err(ImageError::Timeout { .. } | ImageError::QueueEmpty) => {}
            Err(source) => {
                capture_error = Some(RecordCaptureError::RightImage { source });
                break 'capture;
            }
        }

        if depth_enabled {
            match device.depth(0) {
                Ok(depth_frame) => match oak_to_depth_image(depth_frame) {
                    Ok(depth) => {
                        if let Err(err) =
                            require_record_write(writer.write_depth(&depth), RecordItem::DepthFrame)
                        {
                            capture_error = Some(err);
                            break 'capture;
                        }
                        depth_count = depth_count.saturating_add(1);
                        got_any = true;
                    }
                    Err(err) => {
                        eprintln!("depth frame dropped (invalid dimensions): {err}");
                    }
                },
                Err(DepthError::Timeout { .. } | DepthError::QueueEmpty) => {}
                Err(source) => {
                    capture_error = Some(RecordCaptureError::Depth { source });
                    break 'capture;
                }
            }
        }

        loop {
            let pair = match pairer.next_pair() {
                Some(pair) => pair,
                None => break,
            };
            if let Err(err) = require_record_write(writer.write_pair(pair), RecordItem::StereoPair)
            {
                capture_error = Some(err);
                break 'capture;
            }
            pair_count += 1;

            if pair_count.is_multiple_of(30) {
                eprintln!("captured {pair_count} stereo pairs");
            }
        }

        if !got_any {
            thread::sleep(Duration::from_micros(500));
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    let pairer_stats = pairer.stats();
    drop(writer);
    let finalization = writer_handle.finish();
    if let Ok(stats) = &finalization {
        eprintln!(
            "finished in {:.1}s: pairs={}, left={} ({:.1}fps), right={} ({:.1}fps), depth={} ({:.1}fps), written={}, dropped={}",
            elapsed,
            pair_count,
            left_count,
            left_count as f64 / elapsed,
            right_count,
            right_count as f64 / elapsed,
            depth_count,
            depth_count as f64 / elapsed,
            stats.frames_written,
            stats.frames_dropped
        );
    }
    eprintln!(
        "pairer stats: window_ns={} max_pending_per_side={} paired={} dropped_left={} dropped_right={} outside_window={}",
        pairer.window().as_ns(),
        pairer.max_pending_per_side(),
        pairer_stats.paired,
        pairer_stats.dropped_left,
        pairer_stats.dropped_right,
        pairer_stats.outside_window
    );
    match (capture_error, finalization) {
        (None, Ok(_)) => Ok(()),
        (Some(source), Ok(_)) => Err(RecordError::Capture { source }.into()),
        (None, Err(source)) => Err(RecordError::Finalization {
            source: Box::new(source),
        }
        .into()),
        (Some(capture), Err(finalization)) => Err(RecordError::CaptureAndFinalization {
            capture,
            finalization: Box::new(finalization),
        }
        .into()),
    }
}

#[cfg(feature = "record")]
struct LiveVizMsg {
    left: Frame,
    right: Frame,
    depth: Option<DepthImage>,
    pose: Option<WorldToCamera>,
    packet: Option<VizPacket<Raw>>,
    points: Option<Vec<CameraPoint3>>,
    health: SystemHealth,
    diagnostics: FrameDiagnostics,
    events: Vec<DiagnosticEvent>,
    dense_stats: Option<DenseStats>,
}

#[cfg(feature = "record")]
#[derive(Debug)]
enum LiveThreadError {
    VizChannelDisconnected,
    RerunConnect { detail: String },
    VisualizationConfiguration { source: VizConfigError },
    InferenceUnavailable { detail: String },
    TrackerInitialization { source: TrackerInitError },
    FrameProcessingPanic { detail: String },
}

#[cfg(feature = "record")]
impl std::fmt::Display for LiveThreadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LiveThreadError::VizChannelDisconnected => write!(f, "viz channel disconnected"),
            LiveThreadError::RerunConnect { detail } => {
                write!(f, "failed to connect to rerun viewer: {detail}")
            }
            LiveThreadError::VisualizationConfiguration { source } => {
                write!(f, "invalid live visualization configuration: {source}")
            }
            LiveThreadError::InferenceUnavailable { detail } => {
                write!(f, "inference pipeline is unavailable: {detail}")
            }
            LiveThreadError::TrackerInitialization { source } => {
                write!(f, "tracker initialization failed: {source}")
            }
            LiveThreadError::FrameProcessingPanic { detail } => {
                write!(f, "inference panic while processing frame: {detail}")
            }
        }
    }
}

#[cfg(feature = "record")]
impl std::error::Error for LiveThreadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::VisualizationConfiguration { source } => Some(source),
            Self::TrackerInitialization { source } => Some(source),
            Self::VizChannelDisconnected
            | Self::RerunConnect { .. }
            | Self::InferenceUnavailable { .. }
            | Self::FrameProcessingPanic { .. } => None,
        }
    }
}

#[cfg(feature = "record")]
struct LiveThreadExitGuard(Arc<AtomicBool>);

#[cfg(feature = "record")]
impl Drop for LiveThreadExitGuard {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

#[cfg(feature = "record")]
fn drain_depth_batch(rx: &DropReceiver<DepthImage>) -> Vec<DepthImage> {
    std::iter::from_fn(|| rx.try_recv().ok()).collect()
}

#[cfg(feature = "record")]
fn run_live(args: LiveArgs) -> Result<(), Box<dyn std::error::Error>> {
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        eprintln!("\nreceived ctrl+c, stopping...");
        r.store(false, Ordering::SeqCst);
    })?;

    let mono_config = MonoConfig {
        width: args.camera.width,
        height: args.camera.height,
        fps: args.camera.fps,
        rectified: args.camera.rectified,
    };
    let depth_enabled = env_bool("KIKO_LIVE_DEPTH")?.unwrap_or(false);
    let depth_queue_capacity = if depth_enabled {
        Some(ChannelCapacity::try_from(
            env_usize("KIKO_LIVE_DEPTH_QUEUE_DEPTH")?.unwrap_or(8),
        )?)
    } else {
        None
    };
    let depth_ring_capacity = depth_queue_capacity
        .map(DepthRingCapacity::from_queue_capacity)
        .unwrap_or_else(DepthRingCapacity::minimum);

    let config = DeviceConfig {
        rgb: None,
        mono: Some(mono_config),
        depth: depth_enabled.then_some(DepthConfig {
            width: mono_config.width,
            height: mono_config.height,
            fps: mono_config.fps,
            align_to_rgb: false,
        }),
        imu: None,
        queue: QueueConfig {
            size: 8,
            blocking: false,
        },
    };

    eprintln!("connecting to oak-d...");
    let mut device = Device::connect("", config)?;

    let pairing_window = load_pairing_window()?;
    let pairer_max_pending = load_pairer_max_pending_per_side()?;
    let mut pairer = StereoPairer::new_with_max_pending(pairing_window, pairer_max_pending)?;

    let pair_queue_depth = env_usize("KIKO_LIVE_PAIR_QUEUE_DEPTH")?.unwrap_or(12);
    let pair_capacity = ChannelCapacity::try_from(pair_queue_depth)?;
    let (pair_tx, pair_rx, pair_stats) =
        bounded_channel::<StereoPair>(pair_capacity, DropPolicy::DropOldest);

    let viz_queue_depth = env_usize("KIKO_LIVE_VIZ_QUEUE_DEPTH")?.unwrap_or(12);
    let viz_capacity = ChannelCapacity::try_from(viz_queue_depth)?;
    let (viz_tx, viz_rx, viz_stats) = bounded_channel(viz_capacity, DropPolicy::DropNewest);
    let (depth_tx, depth_rx, depth_stats_handle) =
        if let Some(depth_capacity) = depth_queue_capacity {
            let (depth_tx, depth_rx, depth_stats) =
                bounded_channel::<DepthImage>(depth_capacity, DropPolicy::DropOldest);
            (Some(depth_tx), Some(depth_rx), Some(depth_stats))
        } else {
            (None, None, None)
        };

    let inference = InferenceConfig::from_args(&args.inference)?;
    let InferenceConfig {
        superpoint_left,
        superpoint_right,
        lightglue,
        key_limit,
        downscale,
    } = inference;

    let calibration = build_calibration(&device, device.stereo_baseline_m(), &mono_config);
    let rectified = RectifiedStereo::from_calibration(&calibration)?;
    let tracker_config = build_tracker_config(
        TrackerDefaults {
            min_keyframe_points: 80,
            refresh_inliers: 20,
            min_inliers: 15,
        },
        key_limit,
        downscale,
    )?;

    eprintln!(
        "live: pair_queue_depth={} viz_queue_depth={} depth_enabled={} depth_queue_depth={} pairing_window_ns={} pairer_max_pending_per_side={}",
        pair_queue_depth,
        viz_queue_depth,
        depth_enabled,
        depth_queue_capacity.map_or(0, ChannelCapacity::get),
        pairer.window().as_ns(),
        pairer.max_pending_per_side()
    );

    // Dense reconstruction channels and worker thread.
    let dense_enabled = if depth_enabled {
        env_bool("KIKO_DENSE")?.unwrap_or(false)
    } else {
        false
    };
    let dense_capacities = if dense_enabled {
        Some((
            ChannelCapacity::try_from(env_usize("KIKO_DENSE_DATA_QUEUE_DEPTH")?.unwrap_or(4))?,
            ChannelCapacity::try_from(env_usize("KIKO_DENSE_CTRL_QUEUE_DEPTH")?.unwrap_or(64))?,
        ))
    } else {
        None
    };

    // Use one FIFO so reset/rebuild commands cannot overtake or be overtaken
    // by causally adjacent integrations and removals. The data quota reserves
    // the configured control headroom within the bounded queue.
    let mut dense_command_tx: Option<DenseCommandSender> = None;
    let mut dense_command_rx_for_worker: Option<DenseCommandReceiver> = None;
    let mut dense_command_stats_handle: Option<DenseCommandQueueStatsHandle> = None;
    let mut dense_stats_tx_for_worker: Option<kiko_slam::DropSender<DenseStats>> = None;
    let mut dense_stats_rx: Option<kiko_slam::DropReceiver<DenseStats>> = None;

    if let Some((data_cap, ctrl_cap)) = dense_capacities {
        let (command_tx, command_rx, command_stats) =
            dense_command_channel(data_cap, ctrl_cap, Duration::from_millis(5))?;
        let stats_cap = ChannelCapacity::try_from(1_usize)?;
        let (stats_tx, stats_rx_inner, _stats_handle) =
            bounded_channel(stats_cap, DropPolicy::DropNewest);
        dense_command_tx = Some(command_tx);
        dense_command_rx_for_worker = Some(command_rx);
        dense_command_stats_handle = Some(command_stats);
        dense_stats_tx_for_worker = Some(stats_tx);
        dense_stats_rx = Some(stats_rx_inner);
    }

    let dense_handle = if let (Some(command_rx), stats_tx) = (
        dense_command_rx_for_worker.take(),
        dense_stats_tx_for_worker.take(),
    ) {
        let cfg = DenseConfig::default();
        Some(thread::spawn(move || {
            kiko_slam::dense::run_dense_worker(&cfg, &command_rx, None, stats_tx.as_ref());
        }))
    } else {
        None
    };

    let mut depth_ring = DepthRingBuffer::try_new(depth_ring_capacity.get())?;
    let inference_running = Arc::clone(&running);
    let inference_handle = thread::spawn(move || -> Result<(), LiveThreadError> {
        let _exit_guard = LiveThreadExitGuard(inference_running);
        let mut tracker = SlamTracker::try_new(
            superpoint_left,
            superpoint_right,
            lightglue,
            rectified,
            tracker_config,
        )
        .map_err(|source| LiveThreadError::TrackerInitialization { source })?;
        report_tracker_runtime(&tracker_config, &tracker);
        let depth_rx = depth_rx;
        let depth_enabled_for_diagnostics = depth_rx.is_some();
        let mut dense_generation: u64 = 0;
        let mut dense_command_tx = dense_command_tx;
        let dense_stats_rx = dense_stats_rx;
        let mut dense_active = dense_enabled;
        let mut dense_integrations_dropped_newest: u64 = 0;
        let mut depth_reorder_warnings_seen: u64 = 0;

        for pair in pair_rx.iter() {
            let left = pair.left().clone();
            let right = pair.right().clone();
            let timestamp = left.timestamp();
            let depth_batch = depth_rx.as_ref().map(drain_depth_batch).unwrap_or_default();
            let depth = depth_batch.last().cloned();
            for depth_image in depth_batch {
                depth_ring.push(depth_image);
            }
            let reorder_warnings = depth_ring.reorder_warnings();
            if reorder_warnings > depth_reorder_warnings_seen {
                depth_reorder_warnings_seen = reorder_warnings;
                eprintln!(
                    "depth ring observed out-of-order timestamps (count={depth_reorder_warnings_seen})"
                );
            }
            let process_result =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| tracker.process(pair)));
            match process_result {
                Ok(Ok(mut output)) => {
                    // Map tracker output to dense commands.
                    let correction = tracker.take_pending_loop_correction();
                    let dense_stats = if dense_active {
                        let cmds = command_mapper::map_output_to_dense_commands(
                            &output,
                            correction.as_deref(),
                            &depth_ring,
                            timestamp,
                            &mut dense_generation,
                        );
                        for cmd in cmds {
                            if let Some(ref tx) = dense_command_tx {
                                match tx.route(cmd) {
                                    DenseCommandSendOutcome::Enqueued => {}
                                    DenseCommandSendOutcome::IntegrationDroppedNewest => {
                                        dense_integrations_dropped_newest =
                                            dense_integrations_dropped_newest.saturating_add(1);
                                    }
                                    DenseCommandSendOutcome::ControlTimedOut => {
                                        dense_active = false;
                                        dense_command_tx = None;
                                        eprintln!(
                                            "dense ordered command queue timed out accepting control; disabling dense"
                                        );
                                        break;
                                    }
                                    DenseCommandSendOutcome::Disconnected => {
                                        dense_active = false;
                                        dense_command_tx = None;
                                        eprintln!(
                                            "dense ordered command queue disconnected; disabling dense"
                                        );
                                        break;
                                    }
                                }
                            }
                        }
                        // Drain latest dense stats for viz.
                        dense_stats_rx
                            .as_ref()
                            .and_then(|rx| std::iter::from_fn(|| rx.try_recv().ok()).last())
                    } else {
                        None
                    };
                    if let Some(ref stats) = dense_stats
                        && stats.state == ReconState::Down
                    {
                        dense_active = false;
                        dense_command_tx = None;
                        eprintln!("dense worker entered Down state; disabling dense");
                    }

                    if depth_enabled_for_diagnostics {
                        output.diagnostics_mut().depth_reorder_warnings =
                            Some(depth_reorder_warnings_seen);
                    }
                    let mut packet = None;
                    let mut points = None;
                    if let Some(matches) = output.take_stereo_matches() {
                        if let Some(keyframe) = output.keyframe() {
                            points = Some(keyframe.landmarks().to_vec());
                        }
                        if let Ok(viz_packet) =
                            VizPacket::try_new(left.clone(), right.clone(), matches)
                        {
                            packet = Some(viz_packet);
                        }
                    }
                    let (pose, health, diagnostics, events) = output.into_status_parts();
                    let msg = LiveVizMsg {
                        left,
                        right,
                        depth,
                        pose,
                        packet,
                        points,
                        health,
                        diagnostics,
                        events,
                        dense_stats,
                    };
                    if matches!(viz_tx.try_send(msg), SendOutcome::Disconnected) {
                        return Err(LiveThreadError::VizChannelDisconnected);
                    }
                }
                Ok(Err(err)) => {
                    if err.requires_pipeline_shutdown() {
                        return Err(LiveThreadError::InferenceUnavailable {
                            detail: err.to_string(),
                        });
                    }
                    if dense_active && let Some(correction) = tracker.take_pending_loop_correction()
                    {
                        dense_generation = dense_generation
                            .checked_add(1)
                            .expect("dense rebuild generation space exhausted");
                        let rebuild_cmd = DenseCommand::RebuildFromSnapshot {
                            corrected_poses: correction,
                            generation: dense_generation,
                        };
                        if let Some(ref tx) = dense_command_tx {
                            match tx.route(rebuild_cmd) {
                                DenseCommandSendOutcome::Enqueued => {}
                                DenseCommandSendOutcome::ControlTimedOut => {
                                    dense_active = false;
                                    dense_command_tx = None;
                                    eprintln!(
                                        "dense ordered command queue timed out accepting rebuild after tracker error; disabling dense"
                                    );
                                }
                                DenseCommandSendOutcome::Disconnected => {
                                    dense_active = false;
                                    dense_command_tx = None;
                                    eprintln!(
                                        "dense ordered command queue disconnected after tracker error; disabling dense"
                                    );
                                }
                                DenseCommandSendOutcome::IntegrationDroppedNewest => {
                                    dense_active = false;
                                    dense_command_tx = None;
                                    eprintln!(
                                        "dense command router rejected a rebuild as integration data; disabling dense"
                                    );
                                }
                            }
                        }
                    }
                    eprintln!("tracker error: {err}");
                }
                Err(payload) => {
                    return Err(LiveThreadError::FrameProcessingPanic {
                        detail: kiko_slam::panic_payload_to_string(payload.as_ref()),
                    });
                }
            }
        }
        if dense_integrations_dropped_newest > 0 {
            eprintln!(
                "dense integrations dropped_newest (inference view): {dense_integrations_dropped_newest}"
            );
        }
        if depth_reorder_warnings_seen > 0 {
            eprintln!("depth reorder warnings observed: {depth_reorder_warnings_seen}");
        }
        Ok(())
    });

    let decimation = args.rerun_decimation.get();
    let viz_running = Arc::clone(&running);
    let viz_handle = thread::spawn(move || -> Result<(), LiveThreadError> {
        let _exit_guard = LiveThreadExitGuard(viz_running);
        let rec = match rerun::RecordingStreamBuilder::new("kiko-slam-live").connect_grpc() {
            Ok(rec) => rec,
            Err(err) => {
                eprintln!("failed to connect to rerun viewer: {err}");
                return Err(LiveThreadError::RerunConnect {
                    detail: err.to_string(),
                });
            }
        };

        let mut sink = RerunSink::new(rec, decimation)
            .map_err(|source| LiveThreadError::VisualizationConfiguration { source })?;
        for msg in viz_rx.iter() {
            if let Some(packet) = msg.packet.as_ref() {
                if let Err(err) = sink.log_with_points(packet, msg.points.as_deref()) {
                    eprintln!("rerun log error: {err}");
                }
            } else if let Err(err) = sink.log_frames(&msg.left, &msg.right) {
                eprintln!("rerun log error: {err}");
            }
            if let Some(depth) = msg.depth.as_ref()
                && let Err(err) = sink.log_depth(depth)
            {
                eprintln!("rerun log error: {err}");
            }

            if let Some(pose) = msg.pose.as_ref()
                && let Err(err) = sink.log_pose(msg.left.timestamp(), pose)
            {
                eprintln!("rerun log error: {err}");
            }
            if let Err(err) = sink.log_system_health(msg.left.timestamp(), &msg.health) {
                eprintln!("rerun health error: {err}");
            }
            if let Err(err) = sink.log_diagnostics(msg.left.timestamp(), &msg.diagnostics) {
                eprintln!("rerun diagnostics error: {err}");
            }
            for event in &msg.events {
                if let Err(err) = sink.log_event(msg.left.timestamp(), event) {
                    eprintln!("rerun event error: {err}");
                }
            }
            if let Some(ref dense_stats) = msg.dense_stats
                && let Err(err) = sink.log_dense_stats(msg.left.timestamp(), dense_stats)
            {
                eprintln!("rerun dense stats error: {err}");
            }
        }
        Ok(())
    });

    let mut left_seq = 0u64;
    let mut right_seq = 0u64;
    let mut capture_error = None;

    eprintln!("streaming matches... press ctrl+c to stop");

    'capture: while running.load(Ordering::Relaxed) {
        let mut got_any = false;

        match device.mono_left(0) {
            Ok(frame) => match oak_to_frame(frame, SensorId::StereoLeft, FrameId::new(left_seq)) {
                Ok(frame) => {
                    if let Err(error) = pairer.push_left(frame) {
                        capture_error = Some(std::io::Error::other(error));
                        break 'capture;
                    }
                    left_seq += 1;
                    got_any = true;
                }
                Err(err) => {
                    eprintln!("left frame dropped (invalid dimensions): {err}");
                }
            },
            Err(ImageError::Timeout { .. } | ImageError::QueueEmpty) => {}
            Err(e) => {
                capture_error = Some(std::io::Error::other(format!(
                    "left camera capture failed: {e:?}"
                )));
                break 'capture;
            }
        }

        match device.mono_right(0) {
            Ok(frame) => {
                match oak_to_frame(frame, SensorId::StereoRight, FrameId::new(right_seq)) {
                    Ok(frame) => {
                        if let Err(error) = pairer.push_right(frame) {
                            capture_error = Some(std::io::Error::other(error));
                            break 'capture;
                        }
                        right_seq += 1;
                        got_any = true;
                    }
                    Err(err) => {
                        eprintln!("right frame dropped (invalid dimensions): {err}");
                    }
                }
            }
            Err(ImageError::Timeout { .. } | ImageError::QueueEmpty) => {}
            Err(e) => {
                capture_error = Some(std::io::Error::other(format!(
                    "right camera capture failed: {e:?}"
                )));
                break 'capture;
            }
        }

        if depth_enabled {
            match device.depth(0) {
                Ok(depth_frame) => match oak_to_depth_image(depth_frame) {
                    Ok(depth_image) => {
                        got_any = true;
                        if let Some(depth_tx) = depth_tx.as_ref()
                            && matches!(depth_tx.try_send(depth_image), SendOutcome::Disconnected)
                        {
                            break;
                        }
                    }
                    Err(err) => {
                        eprintln!("depth frame dropped (invalid dimensions): {err}");
                    }
                },
                Err(DepthError::Timeout { .. } | DepthError::QueueEmpty) => {}
                Err(e) => {
                    capture_error = Some(std::io::Error::other(format!(
                        "depth camera capture failed: {e:?}"
                    )));
                    break 'capture;
                }
            }
        }

        loop {
            let pair = match pairer.next_pair() {
                Some(pair) => pair,
                None => break,
            };
            if matches!(pair_tx.try_send(pair), SendOutcome::Disconnected) {
                running.store(false, Ordering::SeqCst);
                break 'capture;
            }
        }

        if !got_any {
            thread::sleep(Duration::from_micros(500));
        }
    }

    drop(pair_tx);
    drop(depth_tx);
    let inference_result = inference_handle.join().map_err(|payload| {
        std::io::Error::other(format!(
            "inference thread panicked: {}",
            kiko_slam::panic_payload_to_string(payload.as_ref())
        ))
    })?;
    if let Err(err) = inference_result {
        return Err(Box::new(err));
    }

    let viz_result = viz_handle.join().map_err(|payload| {
        std::io::Error::other(format!(
            "viz thread panicked: {}",
            kiko_slam::panic_payload_to_string(payload.as_ref())
        ))
    })?;
    if let Err(err) = viz_result {
        return Err(std::io::Error::other(err).into());
    }

    if let Some(handle) = dense_handle {
        // Channels are dropped when inference thread exits, causing worker to return.
        handle.join().map_err(|payload| {
            std::io::Error::other(format!(
                "dense thread panicked: {}",
                kiko_slam::panic_payload_to_string(payload.as_ref())
            ))
        })?;
    }

    let pair_snapshot = pair_stats.snapshot();
    let viz_snapshot = viz_stats.snapshot();
    eprintln!(
        "pair queue stats: enqueued={}, dropped_oldest={}, dropped_newest={}, disconnected={}",
        pair_snapshot.enqueued,
        pair_snapshot.dropped_oldest,
        pair_snapshot.dropped_newest,
        pair_snapshot.disconnected
    );
    eprintln!(
        "viz queue stats: enqueued={}, dropped_oldest={}, dropped_newest={}, disconnected={}",
        viz_snapshot.enqueued,
        viz_snapshot.dropped_oldest,
        viz_snapshot.dropped_newest,
        viz_snapshot.disconnected
    );
    if let Some(depth_stats_handle) = depth_stats_handle {
        let depth_snapshot = depth_stats_handle.snapshot();
        eprintln!(
            "depth queue stats: enqueued={}, dropped_oldest={}, dropped_newest={}, disconnected={}",
            depth_snapshot.enqueued,
            depth_snapshot.dropped_oldest,
            depth_snapshot.dropped_newest,
            depth_snapshot.disconnected
        );
    }
    if let Some(dense_command_stats_handle) = dense_command_stats_handle {
        let dense_command_snapshot = dense_command_stats_handle.snapshot();
        eprintln!(
            "dense ordered command queue stats: commands_enqueued={}, integrations_dropped_newest={}, controls_timed_out={}, disconnected={}",
            dense_command_snapshot.commands_enqueued,
            dense_command_snapshot.integrations_dropped_newest,
            dense_command_snapshot.controls_timed_out,
            dense_command_snapshot.disconnected
        );
    }
    let pairer_stats = pairer.stats();
    eprintln!(
        "pairer stats: paired={} dropped_left={} dropped_right={} outside_window={}",
        pairer_stats.paired,
        pairer_stats.dropped_left,
        pairer_stats.dropped_right,
        pairer_stats.outside_window
    );

    if let Some(err) = capture_error {
        return Err(err.into());
    }

    Ok(())
}

#[cfg(feature = "record")]
fn build_meta(
    config: &MonoConfig,
    depth_config: Option<&DepthConfig>,
    imu_config: Option<&ImuConfig>,
) -> Meta {
    Meta {
        created: chrono::Utc::now().to_rfc3339(),
        device: "OAK-D".to_string(),
        mono: Some(MonoMeta {
            width: config.width,
            height: config.height,
            fps: config.fps,
        }),
        depth: depth_config.map(|c| DepthMeta {
            width: c.width,
            height: c.height,
            fps: c.fps,
            encoding: "f32_meters_le".to_string(),
        }),
        imu: imu_config.map(|c| ImuMeta { rate_hz: c.rate_hz }),
    }
}

#[cfg(feature = "record")]
fn build_calibration(device: &Device, baseline_m: f32, config: &MonoConfig) -> Calibration {
    let left = device.left_intrinsics();
    let right = device.right_intrinsics();

    Calibration {
        left: CameraIntrinsics {
            fx: left.fx,
            fy: left.fy,
            cx: left.cx,
            cy: left.cy,
            width: left.width,
            height: left.height,
        },
        right: CameraIntrinsics {
            fx: right.fx,
            fy: right.fy,
            cx: right.cx,
            cy: right.cy,
            width: right.width,
            height: right.height,
        },
        baseline_m,
        rectified: config.rectified,
    }
}

#[derive(Clone, Copy, Debug)]
struct BaConfigValues {
    window: usize,
    iterations: usize,
    min_observations: usize,
    huber_delta_px: f32,
    initial_lambda: f32,
    lambda_factor: f32,
    min_lambda: f32,
    max_lambda: f32,
    motion_prior_weight: f32,
}

fn build_ba_config_from_values(
    values: BaConfigValues,
) -> Result<LocalBaConfig, Box<dyn std::error::Error>> {
    let default_lm = LmConfig::default();
    let lm = LmConfig::new(
        values.initial_lambda,
        values.lambda_factor,
        values.min_lambda,
        values.max_lambda,
        default_lm.rho_accept(),
        default_lm.rho_good(),
    )?;
    Ok(LocalBaConfig::new(
        values.window,
        values.iterations,
        values.min_observations,
        values.huber_delta_px,
        lm,
        values.motion_prior_weight,
    )?)
}

fn build_ba_config() -> Result<LocalBaConfig, Box<dyn std::error::Error>> {
    let config = build_ba_config_from_values(BaConfigValues {
        window: env_usize("KIKO_BA_WINDOW")?.unwrap_or(DEFAULT_BA_WINDOW),
        iterations: env_usize("KIKO_BA_ITERS")?.unwrap_or(DEFAULT_BA_ITERS),
        min_observations: env_usize("KIKO_BA_MIN_OBS")?.unwrap_or(DEFAULT_BA_MIN_OBS),
        huber_delta_px: env_f32("KIKO_BA_HUBER_PX")?.unwrap_or(DEFAULT_BA_HUBER_PX),
        initial_lambda: env_f32("KIKO_BA_DAMPING")?.unwrap_or(DEFAULT_BA_DAMPING),
        lambda_factor: env_f32("KIKO_LM_FACTOR")?.unwrap_or(DEFAULT_LM_FACTOR),
        min_lambda: env_f32("KIKO_LM_MIN")?.unwrap_or(DEFAULT_LM_MIN),
        max_lambda: env_f32("KIKO_LM_MAX")?.unwrap_or(DEFAULT_LM_MAX),
        motion_prior_weight: env_f32("KIKO_BA_MOTION_WEIGHT")?.unwrap_or(DEFAULT_BA_MOTION_WEIGHT),
    })?;
    eprintln!(
        "local BA: window={} iters={} min_obs={} huber_px={} lm_init={} lm_factor={} lm_min={} lm_max={} motion_weight={}",
        config.window(),
        config.max_iterations(),
        config.min_observations(),
        config.huber_delta_px(),
        config.lm().initial_lambda(),
        config.lm().lambda_factor(),
        config.lm().min_lambda(),
        config.lm().max_lambda(),
        config.motion_prior_weight()
    );
    Ok(config)
}

#[derive(Clone, Copy, Debug)]
struct CpuSnapshot {
    cpu_time: CpuTime,
    max_rss_bytes: Option<u64>,
}

#[derive(Clone, Copy, Debug)]
struct CpuTime {
    user: Duration,
    sys: Duration,
}

impl CpuTime {
    fn saturating_sub(self, other: CpuTime) -> CpuTime {
        CpuTime {
            user: self.user.saturating_sub(other.user),
            sys: self.sys.saturating_sub(other.sys),
        }
    }
}

#[cfg(unix)]
#[allow(unsafe_code)]
fn process_usage() -> Option<CpuSnapshot> {
    // SAFETY: `libc::rusage` is a plain-old-data C struct; zeroed is a valid
    // representation. `getrusage` writes into the provided pointer.
    unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        if libc::getrusage(libc::RUSAGE_SELF, &mut usage) != 0 {
            return None;
        }
        let user = timeval_to_duration(usage.ru_utime);
        let sys = timeval_to_duration(usage.ru_stime);
        let max_rss_bytes = max_rss_bytes(usage.ru_maxrss);
        Some(CpuSnapshot {
            cpu_time: CpuTime { user, sys },
            max_rss_bytes,
        })
    }
}

#[cfg(not(unix))]
fn process_usage() -> Option<CpuSnapshot> {
    None
}

#[cfg(unix)]
fn timeval_to_duration(tv: libc::timeval) -> Duration {
    let secs = tv.tv_sec.max(0) as u64;
    let micros = tv.tv_usec.max(0) as u32;
    Duration::new(secs, micros * 1000)
}

#[cfg(unix)]
fn max_rss_bytes(raw: libc::c_long) -> Option<u64> {
    if raw <= 0 {
        return None;
    }
    let rss = raw as u64;
    if cfg!(target_os = "macos") {
        Some(rss)
    } else {
        Some(rss * 1024)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BaConfigValues, BenchError, Cli, Command, DepthRingCapacity, OfflineDepthSelector,
        RerunDestination, RerunDestinationError, RerunFinishTimeout, RerunSessionError,
        build_ba_config_from_values, combine_rerun_results,
    };
    use clap::{Parser as _, error::ErrorKind};
    use kiko_slam::dataset::DatasetError;
    use kiko_slam::{
        DepthImage, FrameId, InferenceError, PipelineError, PipelineTimingError, Timestamp,
        VizFlushError,
    };
    use std::collections::VecDeque;
    use std::num::NonZeroU16;
    use std::path::Path;
    use std::time::Duration;

    #[cfg(feature = "record")]
    use super::LiveThreadExitGuard;
    #[cfg(feature = "record")]
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

    #[test]
    fn benchmark_errors_preserve_dataset_and_pipeline_sources() {
        let dataset = BenchError::from(DatasetError::InvalidManifest {
            reason: "test manifest failure",
        });
        assert!(matches!(&dataset, BenchError::Dataset(_)));
        assert!(
            std::error::Error::source(&dataset)
                .and_then(|source| source.downcast_ref::<DatasetError>())
                .is_some()
        );

        let pipeline = BenchError::from(PipelineError::Inference(
            InferenceError::InvariantViolation {
                context: "test pipeline failure",
            },
        ));
        assert!(matches!(&pipeline, BenchError::Pipeline(_)));
        assert!(
            std::error::Error::source(&pipeline)
                .and_then(|source| source.downcast_ref::<PipelineError>())
                .is_some()
        );

        let timing_source = PipelineTimingError::ComponentsExceedTotal {
            accounted: Duration::from_millis(2),
            total: Duration::from_millis(1),
        };
        let timing = BenchError::from(timing_source);
        assert!(matches!(&timing, BenchError::Timing(source) if *source == timing_source));
        assert!(
            std::error::Error::source(&timing)
                .and_then(|source| source.downcast_ref::<PipelineTimingError>())
                .is_some()
        );
    }

    #[test]
    fn capacity_minimum_is_rejected_instead_of_clamped() {
        let error = DepthRingCapacity::try_new("TEST_CAPACITY", 3)
            .expect_err("undersized capacity must fail");
        assert_eq!(error.key, "TEST_CAPACITY");
        assert_eq!(error.value, 3);
        assert_eq!(
            DepthRingCapacity::try_new("TEST_CAPACITY", 4).map(DepthRingCapacity::get),
            Ok(4)
        );
        assert_eq!(
            DepthRingCapacity::from_queue_capacity(
                kiko_slam::ChannelCapacity::try_from(1).expect("nonzero queue capacity")
            )
            .get(),
            4
        );
        assert_eq!(
            DepthRingCapacity::from_queue_capacity(
                kiko_slam::ChannelCapacity::try_from(8).expect("nonzero queue capacity")
            )
            .get(),
            8
        );
    }

    #[test]
    fn cli_rejects_zero_pair_limits_at_the_boundary() {
        let error = Cli::try_parse_from(["kiko-slam", "bench", "/tmp/dataset", "0"])
            .expect_err("zero pair limit must be rejected");
        assert_eq!(error.kind(), ErrorKind::ValueValidation);
    }

    #[test]
    fn rerun_destination_rejects_contradictory_weak_fields() {
        let save_path = Path::new("output.rrd");
        let port = NonZeroU16::new(9877).expect("nonzero test port");

        assert_eq!(
            RerunDestination::parse(Some(save_path), true, None),
            Err(RerunDestinationError::SaveAndServe)
        );
        assert_eq!(
            RerunDestination::parse(None, false, Some(port)),
            Err(RerunDestinationError::PortWithoutServer)
        );
        assert_eq!(
            RerunDestination::parse(Some(save_path), false, Some(port)),
            Err(RerunDestinationError::PortWithoutServer)
        );
    }

    #[test]
    fn rerun_destination_resolves_each_valid_mode_once() {
        let save_path = Path::new("output.rrd");
        let port = NonZeroU16::new(9877).expect("nonzero test port");

        assert_eq!(
            RerunDestination::parse(Some(save_path), false, None),
            Ok(RerunDestination::Save(save_path))
        );
        assert_eq!(
            RerunDestination::parse(None, true, Some(port)),
            Ok(RerunDestination::Serve { port })
        );
        assert!(matches!(
            RerunDestination::parse(None, true, None),
            Ok(RerunDestination::Serve { port }) if port.get() == 9876
        ));
        assert_eq!(
            RerunDestination::parse(None, false, None),
            Ok(RerunDestination::Connect)
        );
    }

    #[test]
    fn rerun_finish_timeout_parses_milliseconds_once_at_the_cli_boundary() {
        let cli = Cli::try_parse_from([
            "kiko-slam",
            "viz",
            "--rerun-finish-timeout-ms",
            "17",
            "/tmp/dataset",
        ])
        .expect("an exact millisecond timeout is valid");
        let Command::Viz(args) = cli.command else {
            panic!("expected visualization command");
        };
        assert_eq!(
            args.rerun_finish_timeout_ms.get(),
            Duration::from_millis(17)
        );

        let zero = "0"
            .parse::<RerunFinishTimeout>()
            .expect("zero is a valid immediate sink-flush timeout");
        assert_eq!(zero.get(), Duration::ZERO);
        assert!("not-a-timeout".parse::<RerunFinishTimeout>().is_err());
    }

    #[test]
    fn rerun_result_combiner_preserves_each_failure_outcome() {
        assert_eq!(
            combine_rerun_results::<_, DatasetError>(Ok(7_u8), Ok(())).expect("both succeeded"),
            7
        );

        let processing =
            combine_rerun_results::<(), _>(Err(DatasetError::DepthStreamNotConfigured), Ok(()))
                .expect_err("processing failure must be returned");
        assert!(matches!(
            processing,
            RerunSessionError::Processing(DatasetError::DepthStreamNotConfigured)
        ));

        let finalization = combine_rerun_results::<(), DatasetError>(
            Ok(()),
            Err(VizFlushError::from(rerun::sink::SinkFlushError::Timeout)),
        )
        .expect_err("finalization failure must be returned");
        assert!(matches!(
            finalization,
            RerunSessionError::Finalization(VizFlushError::Rerun(
                rerun::sink::SinkFlushError::Timeout
            ))
        ));

        let combined = combine_rerun_results::<(), _>(
            Err(DatasetError::DepthStreamNotConfigured),
            Err(VizFlushError::from(rerun::sink::SinkFlushError::Timeout)),
        )
        .expect_err("neither failure may hide the other");
        let display = combined.to_string();
        assert!(display.contains("dataset metadata does not configure a depth stream"));
        assert!(display.contains("finalization also failed"));
        assert!(matches!(
            combined.processing_error(),
            Some(DatasetError::DepthStreamNotConfigured)
        ));
        assert!(matches!(
            combined.finalization_error(),
            Some(VizFlushError::Rerun(rerun::sink::SinkFlushError::Timeout))
        ));
        assert!(matches!(
            combined,
            RerunSessionError::ProcessingAndFinalization {
                processing: DatasetError::DepthStreamNotConfigured,
                finalization: VizFlushError::Rerun(rerun::sink::SinkFlushError::Timeout),
            }
        ));
    }

    #[test]
    fn cli_rejects_ephemeral_rerun_port_zero() {
        let error = Cli::try_parse_from([
            "kiko-slam",
            "viz",
            "--rerun-serve",
            "--rerun-port",
            "0",
            "/tmp/dataset",
        ])
        .expect_err("port zero would make the announced endpoint untruthful");
        assert_eq!(error.kind(), ErrorKind::ValueValidation);
    }

    fn test_depth(frame_id: u64, timestamp_ns: i64) -> DepthImage {
        DepthImage::new(
            FrameId::new(frame_id),
            Timestamp::from_nanos(timestamp_ns),
            1,
            1,
            vec![1.0],
        )
        .expect("valid test depth")
    }

    fn select_test_depth(
        selector: &mut OfflineDepthSelector,
        entries: &mut VecDeque<DepthImage>,
        timestamp_ns: i64,
    ) -> Option<DepthImage> {
        selector
            .select(Timestamp::from_nanos(timestamp_ns), |cutoff| {
                if entries
                    .front()
                    .is_some_and(|depth| depth.timestamp() <= cutoff)
                {
                    Ok(entries.pop_front())
                } else {
                    Ok(None)
                }
            })
            .expect("in-memory depth source")
    }

    #[test]
    fn offline_depth_selector_considers_the_first_future_frame() {
        let mut selector = OfflineDepthSelector::default();
        let mut entries = VecDeque::from([
            test_depth(0, -10_000_000),
            test_depth(1, 5_000_000),
            test_depth(2, 6_000_000),
        ]);

        let selected = select_test_depth(&mut selector, &mut entries, 0)
            .expect("a future frame is closer than the previous frame");
        assert_eq!(selected.frame_id(), FrameId::new(1));
        assert_eq!(entries.len(), 1, "only one lookahead frame is decoded");

        let selected = select_test_depth(&mut selector, &mut entries, 5_500_000)
            .expect("retained lookahead remains a candidate");
        assert_eq!(selected.frame_id(), FrameId::new(1));
        let selected = select_test_depth(&mut selector, &mut entries, 6_000_000)
            .expect("the next query advances to the retained successor");
        assert_eq!(selected.frame_id(), FrameId::new(2));
    }

    #[test]
    fn offline_depth_selector_prefers_the_earlier_frame_on_a_tie() {
        let mut selector = OfflineDepthSelector::default();
        let mut entries = VecDeque::from([test_depth(0, -5_000_000), test_depth(1, 5_000_000)]);

        let selected = select_test_depth(&mut selector, &mut entries, 0)
            .expect("both frames are inside the association window");
        assert_eq!(selected.frame_id(), FrameId::new(0));
    }

    #[test]
    fn offline_depth_selector_handles_the_maximum_timestamp_cutoff() {
        let mut selector = OfflineDepthSelector::default();
        let mut entries = VecDeque::from([test_depth(0, i64::MAX)]);

        let selected = select_test_depth(&mut selector, &mut entries, i64::MAX - 1)
            .expect("the representable upper timestamp remains selectable");
        assert_eq!(selected.timestamp(), Timestamp::from_nanos(i64::MAX));
    }

    #[test]
    fn offline_depth_selector_uses_an_inclusive_association_window() {
        let window = kiko_slam::dense::command_mapper::MAX_ASSOCIATION_WINDOW_NS;
        let mut selector = OfflineDepthSelector::default();
        let mut entries = VecDeque::from([
            test_depth(0, window),
            test_depth(1, window.checked_add(1).expect("test timestamp")),
        ]);

        let selected = select_test_depth(&mut selector, &mut entries, 0)
            .expect("a frame exactly at the association bound is valid");
        assert_eq!(selected.frame_id(), FrameId::new(0));
        assert_eq!(entries.len(), 1, "the out-of-window frame stays unread");

        let mut selector = OfflineDepthSelector::default();
        let mut entries = VecDeque::from([test_depth(
            1,
            window.checked_add(1).expect("test timestamp"),
        )]);
        assert!(select_test_depth(&mut selector, &mut entries, 0).is_none());
        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn offline_depth_selector_propagates_cursor_errors() {
        let mut selector = OfflineDepthSelector::default();
        let error = selector
            .select(Timestamp::from_nanos(0), |_| {
                Err(DatasetError::DepthStreamNotConfigured)
            })
            .expect_err("cursor errors must not disable dense replay silently");
        assert!(matches!(error, DatasetError::DepthStreamNotConfigured));
    }

    #[test]
    fn offline_depth_selector_matches_nearest_timestamp_oracle_for_all_small_subsets() {
        let window = kiko_slam::dense::command_mapper::MAX_ASSOCIATION_WINDOW_NS;
        let candidates = [
            -30_000_000,
            -20_000_000,
            -10_000_000,
            -1,
            0,
            2_000_000,
            20_000_000,
            30_000_000,
        ];
        let queries = [
            -25_000_000,
            -10_000_000,
            0,
            5_000_000,
            20_000_000,
            35_000_000,
        ];
        let max_delta = u64::try_from(window).expect("nonnegative association window");

        for mask in 0_u16..(1_u16 << candidates.len()) {
            let selected_timestamps: Vec<i64> = candidates
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(index, timestamp)| (mask & (1 << index) != 0).then_some(timestamp))
                .collect();
            let mut entries: VecDeque<_> = selected_timestamps
                .iter()
                .copied()
                .enumerate()
                .map(|(index, timestamp)| {
                    test_depth(u64::try_from(index).expect("small test index"), timestamp)
                })
                .collect();
            let mut selector = OfflineDepthSelector::default();

            for query in queries {
                let actual = select_test_depth(&mut selector, &mut entries, query)
                    .map(|depth| depth.timestamp().as_nanos());
                let expected = selected_timestamps
                    .iter()
                    .copied()
                    .filter_map(|timestamp| {
                        let delta = timestamp.abs_diff(query);
                        (delta <= max_delta).then_some((delta, timestamp))
                    })
                    .min_by_key(|&(delta, timestamp)| (delta, timestamp))
                    .map(|(_, timestamp)| timestamp);
                assert_eq!(
                    actual, expected,
                    "mask={mask:#010b}, query={query}, depths={selected_timestamps:?}"
                );
                if let (Some(lookahead), Some(unread)) =
                    (selector.lookahead.as_ref(), entries.front())
                {
                    assert!(lookahead.timestamp() < unread.timestamp());
                }
            }
        }
    }

    #[test]
    fn build_ba_config_from_parsed_values_preserves_lm_settings() {
        let config = build_ba_config_from_values(BaConfigValues {
            window: 12,
            iterations: 7,
            min_observations: 9,
            huber_delta_px: 2.5,
            initial_lambda: 0.002,
            lambda_factor: 12.0,
            min_lambda: 0.000_001,
            max_lambda: 5000.0,
            motion_prior_weight: 0.25,
        })
        .expect("build config");
        assert_eq!(config.window(), 12);
        assert_eq!(config.max_iterations(), 7);
        assert_eq!(config.min_observations(), 9);
        assert!((config.huber_delta_px() - 2.5).abs() < 1e-6);
        assert!((config.lm().initial_lambda() - 0.002).abs() < 1e-9);
        assert!((config.lm().lambda_factor() - 12.0).abs() < 1e-9);
        assert!((config.lm().min_lambda() - 1e-6).abs() < 1e-12);
        assert!((config.lm().max_lambda() - 5000.0).abs() < 1e-6);
        assert!((config.motion_prior_weight() - 0.25).abs() < 1e-6);
    }

    #[cfg(feature = "record")]
    #[test]
    fn live_thread_exit_guard_stops_capture() {
        let running = Arc::new(AtomicBool::new(true));
        {
            let _guard = LiveThreadExitGuard(Arc::clone(&running));
            assert!(running.load(Ordering::SeqCst));
        }
        assert!(!running.load(Ordering::SeqCst));
    }
}
