use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use clap::{Args, Parser, Subcommand, ValueEnum};

use kiko_slam::dataset::DatasetReader;
use kiko_slam::{
    BackendConfig, DownscaleFactor, GlobalDescriptorConfig, InferenceBackend, InferencePipeline,
    KeyframePolicy, KeypointLimit, LightGlue, LmConfig, LocalBaConfig, LoopClosureConfig,
    LoopClosureConfigInput, LoopSubsystemConfig, PinholeIntrinsics, RansacConfig,
    RectificationMode, RectifiedStereo, RectifiedStereoConfig, RedundancyPolicy,
    RelocalizationConfig, RerunSink, SlamTracker, SuperPoint, TrackerConfig,
    TriangulationConfig, TriangulationError, Triangulator, VizDecimation, VizPacket,
};

use kiko_slam::env::{env_bool, env_f32, env_usize};

#[cfg(feature = "record")]
use kiko_slam::{Frame, Point3, Pose, Raw};
#[cfg(feature = "record")]
use kiko_slam::DepthImage;

#[cfg(feature = "record")]
use kiko_slam::dataset::{
    Calibration, CameraIntrinsics, DatasetWriter, DepthMeta, ImuMeta, Meta, MonoMeta,
};
#[cfg(feature = "record")]
use kiko_slam::{
    bounded_channel, oak_to_depth_image, oak_to_frame, ChannelCapacity, DiagnosticEvent,
    DropPolicy, DropReceiver, FrameDiagnostics, FrameId, PairingConfigError, PairingOutcome,
    PairingWindowNs, PendingFramesCapacity, SendOutcome, SensorId, StereoPair, StereoPairer,
    SystemHealth,
};
#[cfg(feature = "record")]
use oak_sys::{
    DepthConfig, DepthError, Device, DeviceConfig, ImageError, ImuConfig, MonoConfig, QueueConfig,
};
#[cfg(feature = "record")]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "record")]
use std::sync::Arc;
#[cfg(feature = "record")]
use std::thread;

const DEFAULT_MAX_KEYPOINTS: usize = 1024;

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
    max_pairs: Option<usize>,
}

#[derive(Args, Clone, Debug)]
struct VizArgs {
    #[command(flatten)]
    inference: InferenceArgs,
    #[arg(long, env = "KIKO_RERUN_DECIMATION", default_value_t = VizDecimationArg::default())]
    rerun_decimation: VizDecimationArg,
    #[arg(long, env = "KIKO_RERUN_SAVE")]
    save_rrd: Option<PathBuf>,
    #[arg(long, env = "KIKO_VIZ_ODOMETRY", default_value_t = false)]
    odometry: bool,
    #[arg(long, env = "KIKO_RECTIFY_TOLERANCE")]
    rectify_tolerance: Option<f32>,
    #[arg(long, env = "KIKO_ALLOW_UNRECTIFIED", default_value_t = false)]
    allow_unrectified: bool,
    #[command(flatten)]
    dataset: DatasetArgs,
}

#[derive(Args, Clone, Debug)]
struct BenchArgs {
    #[command(flatten)]
    inference: InferenceArgs,
    #[command(flatten)]
    dataset: DatasetArgs,
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

struct InferenceConfig {
    superpoint: SuperPoint,
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

        let superpoint = SuperPoint::new_with_backend(&sp_path, superpoint_backend)?;
        let lightglue = LightGlue::new_with_backend(&lg_path, lightglue_backend)?;

        eprintln!(
            "inference backend: superpoint={:?}, lightglue={:?}",
            superpoint.backend(),
            lightglue.backend()
        );

        let downscale = args.downscale.get();
        let key_limit = args.max_keypoints.limit();
        eprintln!("downscale: {}", downscale.get());
        eprintln!("max_keypoints: {}", args.max_keypoints.value());

        Ok(Self {
            superpoint,
            lightglue,
            key_limit,
            downscale,
        })
    }

    fn into_pipeline(self) -> InferencePipeline {
        InferencePipeline::new(
            self.superpoint,
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
    if args.odometry {
        return run_viz_odometry(&args);
    }
    run_viz_matches(&args)
}

fn build_recording(
    args: &VizArgs,
    name: &str,
) -> Result<rerun::RecordingStream, Box<dyn std::error::Error>> {
    if let Some(path) = &args.save_rrd {
        let path = if path.is_dir() {
            path.join(format!("{name}.rrd"))
        } else {
            path.clone()
        };
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        eprintln!("rerun: saving to {}", path.display());
        let rec = rerun::RecordingStreamBuilder::new(name).save(&path)?;
        Ok(rec)
    } else {
        Ok(rerun::RecordingStreamBuilder::new(name).connect_grpc()?)
    }
}

fn build_rectified_stereo_config(args: &VizArgs) -> RectifiedStereoConfig {
    RectifiedStereoConfig {
        max_principal_delta_px: args.rectify_tolerance,
        rectification: if args.allow_unrectified {
            RectificationMode::AllowUnrectified
        } else {
            RectificationMode::RequireRectified
        },
    }
}

fn run_viz_matches(args: &VizArgs) -> Result<(), Box<dyn std::error::Error>> {
    let mut reader = DatasetReader::open(&args.dataset.path)?;
    let stats = reader.stats()?;

    eprintln!("dataset: {}", args.dataset.path.display());
    eprintln!(
        "camera fps: left={:.2?} right={:.2?} paired={:.2?} (left={}, right={})",
        stats.left_fps, stats.right_fps, stats.paired_fps, stats.left_count, stats.right_count
    );

    let inference = InferenceConfig::from_args(&args.inference)?;
    let decimation = args.rerun_decimation.get();

    let rectified = RectifiedStereo::from_calibration_with_config(
        reader.calibration(),
        build_rectified_stereo_config(args),
    )?;
    let triangulator = Triangulator::new(rectified, TriangulationConfig::default());

    let rec = build_recording(args, "kiko-slam-dataset")?;
    let mut sink = RerunSink::new(rec, decimation);

    let mut pipeline = inference.into_pipeline();

    let start = Instant::now();
    let mut processed = 0usize;
    let mut inference_errors = 0usize;
    let mut read_errors = 0usize;
    let mut triangulation_empty = 0usize;
    let mut triangulation_errors = 0usize;
    let mut triangulated_points = 0usize;
    let mut total_matches = 0usize;

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
                if let Err(err) = sink.log_with_points(&packet, points) {
                    eprintln!("rerun log error: {err}");
                }
                processed += 1;
            }
            Err(err) => {
                inference_errors += 1;
                eprintln!("inference error: {err}");
            }
        }

        if let Some(limit) = args.dataset.max_pairs {
            if processed >= limit {
                break;
            }
        }
    }

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
        env_usize("KIKO_KEYFRAME_MIN_POINTS").unwrap_or(defaults.min_keyframe_points);
    let refresh_inliers =
        env_usize("KIKO_KEYFRAME_REFRESH_INLIERS").unwrap_or(defaults.refresh_inliers);
    let parallax_px = env_f32("KIKO_KEYFRAME_PARALLAX_PX").unwrap_or(DEFAULT_KEYFRAME_PARALLAX_PX);
    let min_covisibility =
        env_f32("KIKO_KEYFRAME_COVISIBILITY").unwrap_or(DEFAULT_KEYFRAME_COVISIBILITY);
    let redundant_covisibility = env_f32("KIKO_KEYFRAME_REDUNDANT_COVISIBILITY")
        .unwrap_or(DEFAULT_KEYFRAME_REDUNDANT_COVISIBILITY);
    let min_inliers = env_usize("KIKO_TRACK_MIN_INLIERS").unwrap_or(defaults.min_inliers);
    let ransac = RansacConfig {
        min_inliers,
        ..RansacConfig::default()
    };
    let ba_config = build_ba_config()?;
    let keyframe_policy = KeyframePolicy::new(refresh_inliers, parallax_px, min_covisibility)?;
    let redundancy = Some(RedundancyPolicy::new(redundant_covisibility)?);
    let backend = if env_bool("KIKO_BACKEND_ASYNC").unwrap_or(true) {
        Some(BackendConfig::new(
            env_usize("KIKO_BACKEND_QUEUE_DEPTH").unwrap_or(2),
        )?)
    } else {
        None
    };
    let loop_closure_enabled = env_bool("KIKO_LOOP_CLOSURE").unwrap_or(true);
    let learned_descriptors_enabled = env_bool("KIKO_LEARNED_DESCRIPTORS").unwrap_or(true);
    let relocalization_enabled = env_bool("KIKO_RELOCALIZATION").unwrap_or(true);
    let loop_subsystem = if loop_closure_enabled {
        if !learned_descriptors_enabled {
            return Err("invalid tracker config: loop closure requires learned descriptors".into());
        }
        let loop_cfg = build_loop_closure_config_from_env()?;
        let descriptor_cfg =
            GlobalDescriptorConfig::new(env_usize("KIKO_DESCRIPTOR_QUEUE_DEPTH").unwrap_or(2))?;
        eprintln!(
            "loop config: similarity={:.3} descriptor_match={:.3} min_inliers={} max_candidates={} temporal_gap={} min_streak={} max_translation_m={:.3} max_rotation_deg={:.3} ransac_iters={} ransac_px={:.3} ransac_min_inliers={}",
            loop_cfg.similarity_threshold(),
            loop_cfg.descriptor_match_threshold(),
            loop_cfg.min_inliers(),
            loop_cfg.max_candidates(),
            loop_cfg.temporal_gap(),
            loop_cfg.min_streak(),
            loop_cfg.max_correction_translation(),
            loop_cfg.max_correction_rotation_deg(),
            loop_cfg.ransac().max_iterations,
            loop_cfg.ransac().reprojection_threshold_px,
            loop_cfg.ransac().min_inliers,
        );
        if relocalization_enabled {
            LoopSubsystemConfig::with_relocalization(
                loop_cfg,
                descriptor_cfg,
                RelocalizationConfig::default(),
            )
        } else {
            LoopSubsystemConfig::loop_closure_only(loop_cfg, descriptor_cfg)
        }
    } else {
        if relocalization_enabled {
            eprintln!(
                "relocalization requested but loop closure is disabled; disabling relocalization"
            );
        }
        LoopSubsystemConfig::Disabled
    };

    eprintln!(
        "tracker: keyframe_min_points={min_keyframe_points} refresh_inliers={refresh_inliers} parallax_px={parallax_px:.1} min_covisibility={min_covisibility:.2} redundant_covisibility={redundant_covisibility:.2} min_inliers={min_inliers} downscale={} max_keypoints={} loop_closure={} learned_descriptors={} relocalization={}",
        downscale.get(),
        key_limit.get(),
        loop_closure_enabled,
        learned_descriptors_enabled && loop_closure_enabled,
        relocalization_enabled && loop_closure_enabled,
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

fn build_loop_closure_config_from_env() -> Result<LoopClosureConfig, Box<dyn std::error::Error>> {
    let mut input = LoopClosureConfigInput::default();
    if let Some(v) = env_f32("KIKO_LOOP_SIMILARITY_THRESHOLD") {
        input.similarity_threshold = v;
    }
    if let Some(v) = env_f32("KIKO_LOOP_DESCRIPTOR_MATCH_THRESHOLD") {
        input.descriptor_match_threshold = v;
    }
    if let Some(v) = env_usize("KIKO_LOOP_MIN_INLIERS") {
        input.min_inliers = v;
    }
    if let Some(v) = env_usize("KIKO_LOOP_MAX_CANDIDATES") {
        input.max_candidates = v;
    }
    if let Some(v) = env_usize("KIKO_LOOP_TEMPORAL_GAP") {
        input.temporal_gap = v;
    }
    if let Some(v) = env_usize("KIKO_LOOP_MIN_STREAK") {
        input.min_streak = v;
    }
    if let Some(v) = env_f32("KIKO_LOOP_MAX_CORRECTION_TRANSLATION_M") {
        input.max_correction_translation = v;
    }
    if let Some(v) = env_f32("KIKO_LOOP_MAX_CORRECTION_ROTATION_DEG") {
        input.max_correction_rotation_deg = v;
    }

    let mut ransac = input.ransac;
    if let Some(v) = env_usize("KIKO_LOOP_RANSAC_MAX_ITERATIONS") {
        ransac.max_iterations = v;
    }
    if let Some(v) = env_f32("KIKO_LOOP_RANSAC_THRESHOLD_PX") {
        ransac.reprojection_threshold_px = v;
    }
    if let Some(v) = env_usize("KIKO_LOOP_RANSAC_MIN_INLIERS") {
        ransac.min_inliers = v;
    }
    input.ransac = ransac;

    LoopClosureConfig::new(input).map_err(Into::into)
}

fn run_viz_odometry(args: &VizArgs) -> Result<(), Box<dyn std::error::Error>> {
    let mut reader = DatasetReader::open(&args.dataset.path)?;
    let stats = reader.stats()?;

    eprintln!("dataset: {}", args.dataset.path.display());
    eprintln!(
        "camera fps: left={:.2?} right={:.2?} paired={:.2?} (left={}, right={})",
        stats.left_fps, stats.right_fps, stats.paired_fps, stats.left_count, stats.right_count
    );

    let inference = InferenceConfig::from_args(&args.inference)?;
    let decimation = args.rerun_decimation.get();

    let rectified = RectifiedStereo::from_calibration_with_config(
        reader.calibration(),
        build_rectified_stereo_config(args),
    )?;
    let intrinsics = PinholeIntrinsics::try_from(&reader.calibration().left)?;

    let InferenceConfig {
        superpoint,
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

    let rec = build_recording(args, "kiko-slam-dataset-odometry")?;
    let mut sink = RerunSink::new(rec, decimation);
    let mut tracker = SlamTracker::try_new(
        superpoint,
        lightglue,
        rectified,
        intrinsics,
        tracker_config,
    )?;

    let start = Instant::now();
    let mut processed = 0usize;
    let mut inference_errors = 0usize;
    let mut read_errors = 0usize;
    let mut poses_logged = 0usize;
    let mut keyframes = 0usize;

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

        match tracker.process(pair) {
            Ok(output) => {
                let timestamp = left.timestamp();
                let loop_applied = output.events.iter().any(|event| {
                    matches!(
                        event,
                        kiko_slam::DiagnosticEvent::LoopClosureDetected { .. }
                    )
                });
                if let Some(matches) = output.stereo_matches {
                    let points = output
                        .keyframe
                        .as_ref()
                        .map(|kf| kf.landmarks())
                        .filter(|pts| !pts.is_empty());
                    if let Ok(packet) = VizPacket::try_new(left.clone(), right.clone(), matches) {
                        if let Err(err) = sink.log_with_points(&packet, points) {
                            eprintln!("rerun log error: {err}");
                        }
                    }
                    if output.keyframe.is_some() {
                        keyframes += 1;
                    }
                    if output.keyframe.is_some() || loop_applied {
                        let snapshot = tracker.covisibility_snapshot();
                        if let Err(err) = sink.log_covisibility_graph(left.timestamp(), &snapshot) {
                            eprintln!("rerun log error: {err}");
                        }
                    }
                } else if let Err(err) = sink.log_frames(&left, &right) {
                    eprintln!("rerun log error: {err}");
                }

                if let Some(pose) = output.pose.as_ref() {
                    if let Err(err) = sink.log_pose(timestamp, pose) {
                        eprintln!("rerun log error: {err}");
                    } else {
                        poses_logged += 1;
                    }
                }
                if let Err(err) = sink.log_system_health(timestamp, &output.health) {
                    eprintln!("rerun health error: {err}");
                }
                if let Err(err) = sink.log_diagnostics(timestamp, &output.diagnostics) {
                    eprintln!("rerun diagnostics error: {err}");
                }
                for event in &output.events {
                    if let Err(err) = sink.log_event(timestamp, event) {
                        eprintln!("rerun event error: {err}");
                    }
                }
                processed += 1;
            }
            Err(err) => {
                inference_errors += 1;
                eprintln!("tracker error: {err}");
            }
        }

        if let Some(limit) = args.dataset.max_pairs {
            if processed >= limit {
                break;
            }
        }
    }

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

#[derive(Default)]
struct BenchAccum {
    read_samples: usize,
    processed: usize,
    matches_nonzero: usize,
    total_matches: usize,
    read_errors: usize,
    pairing_errors: usize,
    inference_errors: usize,
    sum_read_left: Duration,
    sum_read_right: Duration,
    sum_pairing: Duration,
    sum_read_bytes: usize,
    sum_sp_left: Duration,
    sum_sp_right: Duration,
    sum_lightglue: Duration,
    sum_total_success: Duration,
    sum_inference_attempt: Duration,
}

struct BenchSummary {
    read_samples: usize,
    processed: usize,
    wall_seconds: f64,
    wall_fps: f64,
    reader_stage_seconds: f64,
    reader_stage_fps: f64,
    reader_throughput_mb_s: f64,
    inference_attempt_seconds: f64,
    inference_attempt_fps: f64,
    successful_inference_seconds: f64,
    successful_inference_fps: f64,
    match_rate: f64,
    avg_matches_per_processed_pair: f64,
    avg_matches_per_nonzero_pair: f64,
    avg_sp_left_ms: f64,
    avg_sp_right_ms: f64,
    avg_lightglue_ms: f64,
    avg_total_success_ms: f64,
    avg_overhead_ms: f64,
    pct_sp_left: f64,
    pct_sp_right: f64,
    pct_lightglue: f64,
    pct_overhead: f64,
}

fn summarize_bench(accum: &BenchAccum, elapsed: Duration) -> BenchSummary {
    let wall_seconds = elapsed.as_secs_f64();
    let wall_fps = if wall_seconds > 0.0 {
        accum.processed as f64 / wall_seconds
    } else {
        0.0
    };

    let reader_stage = accum.sum_read_left + accum.sum_read_right + accum.sum_pairing;
    let reader_stage_seconds = reader_stage.as_secs_f64();
    let reader_stage_fps = if reader_stage_seconds > 0.0 {
        accum.read_samples as f64 / reader_stage_seconds
    } else {
        0.0
    };
    let reader_throughput_mb_s = if reader_stage_seconds > 0.0 {
        (accum.sum_read_bytes as f64 / (1024.0 * 1024.0)) / reader_stage_seconds
    } else {
        0.0
    };

    let inference_attempt_seconds = accum.sum_inference_attempt.as_secs_f64();
    let inference_attempt_fps = if inference_attempt_seconds > 0.0 {
        accum.read_samples as f64 / inference_attempt_seconds
    } else {
        0.0
    };

    let successful_inference_seconds = accum.sum_total_success.as_secs_f64();
    let successful_inference_fps = if successful_inference_seconds > 0.0 {
        accum.processed as f64 / successful_inference_seconds
    } else {
        0.0
    };

    let match_rate = if accum.processed > 0 {
        accum.matches_nonzero as f64 / accum.processed as f64
    } else {
        0.0
    };
    let avg_matches_per_processed_pair = if accum.processed > 0 {
        accum.total_matches as f64 / accum.processed as f64
    } else {
        0.0
    };
    let avg_matches_per_nonzero_pair = if accum.matches_nonzero > 0 {
        accum.total_matches as f64 / accum.matches_nonzero as f64
    } else {
        0.0
    };

    let denom = accum.processed as f64;
    let avg_sp_left_ms = if accum.processed > 0 {
        (accum.sum_sp_left.as_secs_f64() * 1000.0) / denom
    } else {
        0.0
    };
    let avg_sp_right_ms = if accum.processed > 0 {
        (accum.sum_sp_right.as_secs_f64() * 1000.0) / denom
    } else {
        0.0
    };
    let avg_lightglue_ms = if accum.processed > 0 {
        (accum.sum_lightglue.as_secs_f64() * 1000.0) / denom
    } else {
        0.0
    };
    let avg_total_success_ms = if accum.processed > 0 {
        (accum.sum_total_success.as_secs_f64() * 1000.0) / denom
    } else {
        0.0
    };
    let overhead = accum
        .sum_total_success
        .saturating_sub(accum.sum_sp_left + accum.sum_sp_right + accum.sum_lightglue);
    let avg_overhead_ms = if accum.processed > 0 {
        (overhead.as_secs_f64() * 1000.0) / denom
    } else {
        0.0
    };
    let total_ms = accum.sum_total_success.as_secs_f64().max(1e-9);
    let pct_sp_left = (accum.sum_sp_left.as_secs_f64() / total_ms) * 100.0;
    let pct_sp_right = (accum.sum_sp_right.as_secs_f64() / total_ms) * 100.0;
    let pct_lightglue = (accum.sum_lightglue.as_secs_f64() / total_ms) * 100.0;
    let pct_overhead = (overhead.as_secs_f64() / total_ms) * 100.0;

    BenchSummary {
        read_samples: accum.read_samples,
        processed: accum.processed,
        wall_seconds,
        wall_fps,
        reader_stage_seconds,
        reader_stage_fps,
        reader_throughput_mb_s,
        inference_attempt_seconds,
        inference_attempt_fps,
        successful_inference_seconds,
        successful_inference_fps,
        match_rate,
        avg_matches_per_processed_pair,
        avg_matches_per_nonzero_pair,
        avg_sp_left_ms,
        avg_sp_right_ms,
        avg_lightglue_ms,
        avg_total_success_ms,
        avg_overhead_ms,
        pct_sp_left,
        pct_sp_right,
        pct_lightglue,
        pct_overhead,
    }
}

fn run_bench(args: BenchArgs) -> Result<(), Box<dyn std::error::Error>> {
    let dataset_path = &args.dataset.path;
    let open_start = Instant::now();
    let mut reader = DatasetReader::open(dataset_path)?;
    let open_time = open_start.elapsed();

    let stats_start = Instant::now();
    let stats = reader.stats()?;
    let stats_time = stats_start.elapsed();

    eprintln!("dataset: {}", dataset_path.display());
    eprintln!("dataset open: {:.2}ms", open_time.as_secs_f64() * 1000.0);
    eprintln!("scan frames: {:.2}ms", stats_time.as_secs_f64() * 1000.0);
    eprintln!(
        "camera fps: left={:.2?} right={:.2?} paired={:.2?} (left={}, right={})",
        stats.left_fps, stats.right_fps, stats.paired_fps, stats.left_count, stats.right_count
    );

    let inference = InferenceConfig::from_args(&args.inference)?;
    let mut pipeline = inference.into_pipeline();

    let cpu_start = process_usage();
    let mut accum = BenchAccum::default();

    let start = Instant::now();
    for sample in reader.timed_pairs() {
        let sample = match sample {
            Ok(sample) => sample,
            Err(err) => {
                match err {
                    kiko_slam::dataset::DatasetError::PairingFailed { .. } => {
                        accum.pairing_errors += 1;
                    }
                    _ => accum.read_errors += 1,
                }
                eprintln!("read error: {err}");
                continue;
            }
        };
        accum.read_samples += 1;
        let pair = sample.pair;
        accum.sum_read_left += sample.timings.left_read;
        accum.sum_read_right += sample.timings.right_read;
        accum.sum_pairing += sample.timings.pairing;
        accum.sum_read_bytes += sample.timings.left_bytes + sample.timings.right_bytes;

        let inference_attempt_start = Instant::now();
        match pipeline.process_pair_timed(pair) {
            Ok((packet, timings)) => {
                let matches = packet.matches();
                accum.total_matches += matches.len();
                if !matches.is_empty() {
                    accum.matches_nonzero += 1;
                }
                accum.sum_sp_left += timings.superpoint_left;
                accum.sum_sp_right += timings.superpoint_right;
                accum.sum_lightglue += timings.lightglue;
                accum.sum_total_success += timings.total;
                accum.processed += 1;
            }
            Err(err) => {
                accum.inference_errors += 1;
                eprintln!("inference error: {err}");
            }
        }
        accum.sum_inference_attempt += inference_attempt_start.elapsed();

        if let Some(limit) = args.dataset.max_pairs {
            if accum.read_samples >= limit {
                break;
            }
        }
    }
    let elapsed = start.elapsed();
    let cpu_end = process_usage();
    let summary = summarize_bench(&accum, elapsed);

    eprintln!(
        "pipeline wall fps: {:.2} (processed={}, elapsed={:.2}s)",
        summary.wall_fps,
        summary.processed,
        summary.wall_seconds
    );
    eprintln!(
        "reader stage fps: {:.2} (read_samples={}, read_stage_time={:.2}s, throughput={:.2} MB/s)",
        summary.reader_stage_fps,
        summary.read_samples,
        summary.reader_stage_seconds,
        summary.reader_throughput_mb_s
    );
    eprintln!(
        "inference attempt fps: {:.2} (attempts={}, attempt_time={:.2}s)",
        summary.inference_attempt_fps,
        summary.read_samples,
        summary.inference_attempt_seconds
    );
    eprintln!(
        "successful inference fps: {:.2} (processed={}, successful_infer_time={:.2}s)",
        summary.successful_inference_fps,
        summary.processed,
        summary.successful_inference_seconds
    );
    eprintln!(
        "matching: nonzero_pairs={}, match_rate={:.2} avg_matches_processed={:.1} avg_matches_nonzero={:.1}",
        accum.matches_nonzero,
        summary.match_rate,
        summary.avg_matches_per_processed_pair,
        summary.avg_matches_per_nonzero_pair
    );
    eprintln!(
        "errors: read={} pairing={} inference={}",
        accum.read_errors,
        accum.pairing_errors,
        accum.inference_errors
    );

    if accum.processed > 0 {
        eprintln!(
            "timings avg ms: sp_left={:.2} sp_right={:.2} lightglue={:.2} overhead={:.2} total_success={:.2}",
            summary.avg_sp_left_ms,
            summary.avg_sp_right_ms,
            summary.avg_lightglue_ms,
            summary.avg_overhead_ms,
            summary.avg_total_success_ms
        );
        eprintln!(
            "timings pct of successful inference time: sp_left={:.1}% sp_right={:.1}% lightglue={:.1}% overhead={:.1}%",
            summary.pct_sp_left,
            summary.pct_sp_right,
            summary.pct_lightglue,
            summary.pct_overhead
        );
    }

    if let (Some(start_usage), Some(end_usage)) = (cpu_start, cpu_end) {
        let cpu_time = end_usage.cpu_time.saturating_sub(start_usage.cpu_time);
        let cpu_s = cpu_time.user.as_secs_f64() + cpu_time.sys.as_secs_f64();
        let core_equiv = if summary.wall_seconds > 0.0 {
            cpu_s / summary.wall_seconds
        } else {
            0.0
        };
        eprintln!(
            "cpu: user={:.2}ms sys={:.2}ms total={:.2}ms cpu_time_over_wall_pct={:.1} core_equiv={:.2}",
            cpu_time.user.as_secs_f64() * 1000.0,
            cpu_time.sys.as_secs_f64() * 1000.0,
            cpu_s * 1000.0,
            core_equiv * 100.0,
            core_equiv
        );
        if let Some(rss) = end_usage.max_rss_bytes {
            eprintln!("memory: max_rss={:.2} MB", (rss as f64) / (1024.0 * 1024.0));
        }
    }

    if accum.processed == 0 {
        return Err("no paired frames processed".into());
    }
    if accum.matches_nonzero == 0 {
        return Err("no nonzero matches; check models/data".into());
    }
    if accum.inference_errors > 0 {
        return Err("inference errors encountered during run".into());
    }

    Ok(())
}

#[cfg(feature = "record")]
const DEFAULT_PAIRING_WINDOW_NS: i64 = 5_000_000;
#[cfg(feature = "record")]
const DEFAULT_PAIRER_MAX_PENDING_PER_SIDE: usize = 64;

#[cfg(feature = "record")]
fn load_pairing_window() -> Result<PairingWindowNs, PairingConfigError> {
    let window_ns = match env_usize("KIKO_PAIRING_WINDOW_NS") {
        Some(raw) => match i64::try_from(raw) {
            Ok(value) => value,
            Err(_) => {
                eprintln!(
                    "invalid KIKO_PAIRING_WINDOW_NS={raw}, exceeds i64::MAX, using default {DEFAULT_PAIRING_WINDOW_NS}"
                );
                DEFAULT_PAIRING_WINDOW_NS
            }
        },
        None => DEFAULT_PAIRING_WINDOW_NS,
    };
    match PairingWindowNs::new(window_ns) {
        Ok(window) => Ok(window),
        Err(err) => {
            eprintln!("invalid pairing window from env ({err}); using default");
            PairingWindowNs::new(DEFAULT_PAIRING_WINDOW_NS)
        }
    }
}

#[cfg(feature = "record")]
fn load_pairer_max_pending_per_side() -> PendingFramesCapacity {
    let raw = env_usize("KIKO_PAIRER_MAX_PENDING_PER_SIDE")
        .unwrap_or(DEFAULT_PAIRER_MAX_PENDING_PER_SIDE);
    match PendingFramesCapacity::try_from(raw) {
        Ok(capacity) => capacity,
        Err(err) => {
            eprintln!("invalid pairer capacity from env ({err}); using default");
            PendingFramesCapacity::try_from(DEFAULT_PAIRER_MAX_PENDING_PER_SIDE)
                .expect("default pairer capacity")
        }
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
    let depth_enabled = env_bool("KIKO_RECORD_DEPTH").unwrap_or(false);
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

    eprintln!("creating dataset at {}", output_path.display());
    let (writer, writer_handle) = DatasetWriter::create(output_path, &meta, &calibration)?;

    let mut pair_count = 0u64;
    let mut left_count = 0u64;
    let mut right_count = 0u64;
    let mut depth_count = 0u64;
    let mut left_seq = 0u64;
    let mut right_seq = 0u64;
    let pairing_window = load_pairing_window()?;
    let pairer_max_pending = load_pairer_max_pending_per_side();
    let mut pairer = StereoPairer::new_with_max_pending(pairing_window, pairer_max_pending);
    let read_timeout_ms = load_oak_read_timeout_ms();
    let start = Instant::now();

    eprintln!("recording... press ctrl+c to stop");

    while running.load(Ordering::Relaxed) {
        let mut got_any = false;

        match device.mono_left(read_timeout_ms) {
            Ok(frame) => match oak_to_frame(frame, SensorId::StereoLeft, FrameId::new(left_seq)) {
                Ok(frame) => {
                    pairer.push_left(frame);
                    left_count += 1;
                    left_seq += 1;
                    got_any = true;
                }
                Err(err) => {
                    eprintln!("left frame dropped (invalid dimensions): {err}");
                }
            },
            Err(ImageError::Timeout { .. } | ImageError::QueueEmpty) => {}
            Err(e) => {
                eprintln!("left error: {e:?}");
                break;
            }
        }

        match device.mono_right(read_timeout_ms) {
            Ok(frame) => {
                match oak_to_frame(frame, SensorId::StereoRight, FrameId::new(right_seq)) {
                    Ok(frame) => {
                        pairer.push_right(frame);
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
            Err(e) => {
                eprintln!("right error: {e:?}");
                break;
            }
        }

        if depth_enabled {
            match device.depth(read_timeout_ms) {
                Ok(depth_frame) => match oak_to_depth_image(depth_frame) {
                    Ok(depth) => {
                        writer.write_depth(&depth);
                        depth_count = depth_count.saturating_add(1);
                        got_any = true;
                    }
                    Err(err) => {
                        eprintln!("depth frame dropped (invalid dimensions): {err}");
                    }
                },
                Err(DepthError::Timeout { .. } | DepthError::QueueEmpty) => {}
                Err(e) => {
                    eprintln!("depth error: {e:?}");
                    break;
                }
            }
        }

        loop {
            match pairer.next_outcome()? {
                PairingOutcome::Produced(pair) => {
                    writer.write_frame(pair.left());
                    writer.write_frame(pair.right());
                    pair_count += 1;

                    if pair_count % 30 == 0 {
                        eprintln!("captured {pair_count} stereo pairs");
                    }
                }
                PairingOutcome::Dropped { .. } => continue,
                PairingOutcome::Waiting => break,
            }
        }

        if !got_any {
            thread::sleep(Duration::from_micros(500));
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    let pairer_stats = pairer.stats();
    drop(writer);
    let stats = writer_handle.finish()?;
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
    eprintln!(
        "pairer stats: window_ns={} max_pending_per_side={} paired={} dropped_left={} dropped_right={} outside_window={}",
        pairer.window().as_ns(),
        pairer.max_pending_per_side().get(),
        pairer_stats.paired,
        pairer_stats.dropped_left,
        pairer_stats.dropped_right,
        pairer_stats.outside_window
    );
    Ok(())
}

#[cfg(feature = "record")]
struct LiveVizMsg {
    left: Frame,
    right: Frame,
    depth: Option<DepthImage>,
    pose: Option<Pose>,
    packet: Option<VizPacket<Raw>>,
    points: Option<Vec<Point3>>,
    covisibility_snapshot: Option<kiko_slam::CovisibilitySnapshot>,
    health: SystemHealth,
    diagnostics: FrameDiagnostics,
    events: Vec<DiagnosticEvent>,
}

#[cfg(feature = "record")]
#[derive(Debug)]
enum LiveThreadError {
    TrackerInit { detail: String },
    VizChannelDisconnected,
    FrameProcessingPanic { detail: String },
}

#[cfg(feature = "record")]
impl std::fmt::Display for LiveThreadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LiveThreadError::TrackerInit { detail } => {
                write!(f, "failed to initialize tracker: {detail}")
            }
            LiveThreadError::VizChannelDisconnected => write!(f, "viz channel disconnected"),
            LiveThreadError::FrameProcessingPanic { detail } => {
                write!(f, "inference panic while processing frame: {detail}")
            }
        }
    }
}

#[cfg(feature = "record")]
impl std::error::Error for LiveThreadError {}

#[cfg(feature = "record")]
fn drain_latest_depth(rx: &DropReceiver<DepthImage>) -> Option<DepthImage> {
    let mut latest = None;
    while let Ok(depth) = rx.try_recv() {
        latest = Some(depth);
    }
    latest
}

#[cfg(feature = "record")]
fn loop_closure_applied(events: &[DiagnosticEvent]) -> bool {
    events
        .iter()
        .any(|event| matches!(event, DiagnosticEvent::LoopClosureDetected { .. }))
}

#[cfg(feature = "record")]
fn load_oak_read_timeout_ms() -> u32 {
    env_usize("KIKO_OAK_READ_TIMEOUT_MS")
        .unwrap_or(2)
        .min(u32::MAX as usize) as u32
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
    let depth_enabled = env_bool("KIKO_LIVE_DEPTH").unwrap_or(false);
    let depth_queue_depth = env_usize("KIKO_LIVE_DEPTH_QUEUE_DEPTH").unwrap_or(8);

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
    let pairer_max_pending = load_pairer_max_pending_per_side();
    let mut pairer = StereoPairer::new_with_max_pending(pairing_window, pairer_max_pending);
    let read_timeout_ms = load_oak_read_timeout_ms();

    let pair_queue_depth = env_usize("KIKO_LIVE_PAIR_QUEUE_DEPTH").unwrap_or(12);
    let pair_capacity = ChannelCapacity::try_from(pair_queue_depth)?;
    let (pair_tx, pair_rx, pair_stats) =
        bounded_channel::<StereoPair>(pair_capacity, DropPolicy::DropOldest);

    let viz_queue_depth = env_usize("KIKO_LIVE_VIZ_QUEUE_DEPTH").unwrap_or(12);
    let viz_capacity = ChannelCapacity::try_from(viz_queue_depth)?;
    let (viz_tx, viz_rx, viz_stats) = bounded_channel(viz_capacity, DropPolicy::DropNewest);
    let (depth_tx, depth_rx, depth_stats_handle) = if depth_enabled {
        let depth_capacity = ChannelCapacity::try_from(depth_queue_depth)?;
        let (depth_tx, depth_rx, depth_stats) =
            bounded_channel::<DepthImage>(depth_capacity, DropPolicy::DropOldest);
        (Some(depth_tx), Some(depth_rx), Some(depth_stats))
    } else {
        (None, None, None)
    };

    let inference = InferenceConfig::from_args(&args.inference)?;
    let InferenceConfig {
        superpoint,
        lightglue,
        key_limit,
        downscale,
    } = inference;

    let calibration = build_calibration(&device, device.stereo_baseline_m(), &mono_config);
    let rectified = RectifiedStereo::from_calibration(&calibration)?;
    let intrinsics = PinholeIntrinsics::try_from(&calibration.left)?;

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
        depth_queue_depth,
        pairer.window().as_ns(),
        pairer.max_pending_per_side().get()
    );

    let inference_handle = thread::spawn(move || -> Result<(), LiveThreadError> {
        let mut tracker = SlamTracker::try_new(
            superpoint,
            lightglue,
            rectified,
            intrinsics,
            tracker_config,
        )
        .map_err(|err| LiveThreadError::TrackerInit {
            detail: err.to_string(),
        })?;
        let depth_rx = depth_rx;

        for pair in pair_rx.iter() {
            let left = pair.left().clone();
            let right = pair.right().clone();
            let depth = depth_rx.as_ref().and_then(drain_latest_depth);
            let process_result =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| tracker.process(pair)));
            match process_result {
                Ok(Ok(output)) => {
                    let health = output.health.clone();
                    let mut packet = None;
                    let mut points = None;
                    let log_covisibility =
                        output.keyframe.is_some() || loop_closure_applied(&output.events);
                    let covisibility_snapshot = if log_covisibility {
                        Some(tracker.covisibility_snapshot())
                    } else {
                        None
                    };
                    if let Some(matches) = output.stereo_matches {
                        if let Some(keyframe) = output.keyframe.as_ref() {
                            points = Some(keyframe.landmarks().to_vec());
                        }
                        if let Ok(viz_packet) =
                            VizPacket::try_new(left.clone(), right.clone(), matches)
                        {
                            packet = Some(viz_packet);
                        }
                    }
                    let msg = LiveVizMsg {
                        left,
                        right,
                        depth,
                        pose: output.pose,
                        packet,
                        points,
                        covisibility_snapshot,
                        health,
                        diagnostics: output.diagnostics,
                        events: output.events,
                    };
                    if matches!(viz_tx.try_send(msg), SendOutcome::Disconnected) {
                        return Err(LiveThreadError::VizChannelDisconnected);
                    }
                }
                Ok(Err(err)) => {
                    eprintln!("tracker error: {err}");
                }
                Err(payload) => {
                    return Err(LiveThreadError::FrameProcessingPanic {
                        detail: kiko_slam::panic_payload_to_string(payload.as_ref()),
                    });
                }
            }
        }
        Ok(())
    });

    let decimation = args.rerun_decimation.get();
    let live_viz_enabled = env_bool("KIKO_LIVE_VIZ").unwrap_or(true);
    let viz_handle = thread::spawn(move || -> Result<(), LiveThreadError> {
        let mut sink = if live_viz_enabled {
            match rerun::RecordingStreamBuilder::new("kiko-slam-live").connect_grpc() {
                Ok(rec) => Some(RerunSink::new(rec, decimation)),
                Err(err) => {
                    eprintln!("failed to connect to rerun viewer; continuing headless: {err}");
                    None
                }
            }
        } else {
            eprintln!("live viz disabled; continuing headless");
            None
        };
        for msg in viz_rx.iter() {
            if let Some(sink) = sink.as_mut() {
                if let Some(packet) = msg.packet.as_ref() {
                    if let Err(err) = sink.log_with_points(packet, msg.points.as_deref()) {
                        eprintln!("rerun log error: {err}");
                    }
                } else if let Err(err) = sink.log_frames(&msg.left, &msg.right) {
                    eprintln!("rerun log error: {err}");
                }
                if let Some(depth) = msg.depth.as_ref() {
                    if let Err(err) = sink.log_depth(depth) {
                        eprintln!("rerun log error: {err}");
                    }
                }

                if let Some(pose) = msg.pose.as_ref() {
                    if let Err(err) = sink.log_pose(msg.left.timestamp(), pose) {
                        eprintln!("rerun log error: {err}");
                    }
                }
                if let Some(snapshot) = msg.covisibility_snapshot.as_ref() {
                    if let Err(err) = sink.log_covisibility_graph(msg.left.timestamp(), snapshot) {
                        eprintln!("rerun log error: {err}");
                    }
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
            }
        }
        Ok(())
    });

    let mut left_seq = 0u64;
    let mut right_seq = 0u64;

    eprintln!("streaming matches... press ctrl+c to stop");

    'capture: while running.load(Ordering::Relaxed) {
        let mut got_any = false;

        match device.mono_left(read_timeout_ms) {
            Ok(frame) => match oak_to_frame(frame, SensorId::StereoLeft, FrameId::new(left_seq)) {
                Ok(frame) => {
                    pairer.push_left(frame);
                    left_seq += 1;
                    got_any = true;
                }
                Err(err) => {
                    eprintln!("left frame dropped (invalid dimensions): {err}");
                }
            },
            Err(ImageError::Timeout { .. } | ImageError::QueueEmpty) => {}
            Err(e) => {
                eprintln!("left error: {e:?}");
                break;
            }
        }

        match device.mono_right(read_timeout_ms) {
            Ok(frame) => {
                match oak_to_frame(frame, SensorId::StereoRight, FrameId::new(right_seq)) {
                    Ok(frame) => {
                        pairer.push_right(frame);
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
                eprintln!("right error: {e:?}");
                break;
            }
        }

        if depth_enabled {
            match device.depth(read_timeout_ms) {
                Ok(depth_frame) => match oak_to_depth_image(depth_frame) {
                    Ok(depth_image) => {
                        got_any = true;
                        if let Some(depth_tx) = depth_tx.as_ref() {
                            if matches!(depth_tx.try_send(depth_image), SendOutcome::Disconnected) {
                                break;
                            }
                        }
                    }
                    Err(err) => {
                        eprintln!("depth frame dropped (invalid dimensions): {err}");
                    }
                },
                Err(DepthError::Timeout { .. } | DepthError::QueueEmpty) => {}
                Err(e) => {
                    eprintln!("depth error: {e:?}");
                    break;
                }
            }
        }

        loop {
            match pairer.next_outcome()? {
                PairingOutcome::Produced(pair) => {
                    if matches!(pair_tx.try_send(pair), SendOutcome::Disconnected) {
                        running.store(false, Ordering::SeqCst);
                        break 'capture;
                    }
                }
                PairingOutcome::Dropped { .. } => continue,
                PairingOutcome::Waiting => break,
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
        return Err(std::io::Error::other(err).into());
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

    let pair_snapshot = pair_stats.snapshot();
    let viz_snapshot = viz_stats.snapshot();
    eprintln!(
        "pair queue stats: enqueued={}, dropped_oldest={}, dropped_newest={}, disconnected={}, current_depth={}, max_depth={}",
        pair_snapshot.enqueued,
        pair_snapshot.dropped_oldest,
        pair_snapshot.dropped_newest,
        pair_snapshot.disconnected,
        pair_snapshot.current_depth,
        pair_snapshot.max_depth
    );
    eprintln!(
        "viz queue stats: enqueued={}, dropped_oldest={}, dropped_newest={}, disconnected={}, current_depth={}, max_depth={}",
        viz_snapshot.enqueued,
        viz_snapshot.dropped_oldest,
        viz_snapshot.dropped_newest,
        viz_snapshot.disconnected,
        viz_snapshot.current_depth,
        viz_snapshot.max_depth
    );
    if let Some(depth_stats_handle) = depth_stats_handle {
        let depth_snapshot = depth_stats_handle.snapshot();
        eprintln!(
            "depth queue stats: enqueued={}, dropped_oldest={}, dropped_newest={}, disconnected={}, current_depth={}, max_depth={}",
            depth_snapshot.enqueued,
            depth_snapshot.dropped_oldest,
            depth_snapshot.dropped_newest,
            depth_snapshot.disconnected,
            depth_snapshot.current_depth,
            depth_snapshot.max_depth
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

fn build_ba_config() -> Result<LocalBaConfig, Box<dyn std::error::Error>> {
    let window = env_usize("KIKO_BA_WINDOW").unwrap_or(DEFAULT_BA_WINDOW);
    let iters = env_usize("KIKO_BA_ITERS").unwrap_or(DEFAULT_BA_ITERS);
    let min_obs = env_usize("KIKO_BA_MIN_OBS").unwrap_or(DEFAULT_BA_MIN_OBS);
    let huber = env_f32("KIKO_BA_HUBER_PX").unwrap_or(DEFAULT_BA_HUBER_PX);
    let initial_lambda = env_f32("KIKO_BA_DAMPING").unwrap_or(DEFAULT_BA_DAMPING);
    let lambda_factor = env_f32("KIKO_LM_FACTOR").unwrap_or(DEFAULT_LM_FACTOR);
    let min_lambda = env_f32("KIKO_LM_MIN").unwrap_or(DEFAULT_LM_MIN);
    let max_lambda = env_f32("KIKO_LM_MAX").unwrap_or(DEFAULT_LM_MAX);
    let motion = env_f32("KIKO_BA_MOTION_WEIGHT").unwrap_or(DEFAULT_BA_MOTION_WEIGHT);
    let default_lm = LmConfig::default();
    let lm = LmConfig::new(
        initial_lambda,
        lambda_factor,
        min_lambda,
        max_lambda,
        default_lm.rho_accept(),
        default_lm.rho_good(),
    )?;
    let config = LocalBaConfig::new(window, iters, min_obs, huber, lm, motion)?;
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
    use super::{build_ba_config, build_tracker_config, summarize_bench, BenchAccum, TrackerDefaults};
    use kiko_slam::{DownscaleFactor, KeypointLimit, LoopSubsystemConfig};
    use std::ffi::OsString;
    use std::sync::{Mutex, OnceLock};
    use std::time::Duration;

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn set_env(key: &str, value: &str) {
        // Safety: tests hold a process-wide lock while mutating environment vars.
        #[allow(unsafe_code)]
        unsafe {
            std::env::set_var(key, value);
        }
    }

    fn restore_env(key: &str, value: Option<OsString>) {
        // Safety: tests hold a process-wide lock while mutating environment vars.
        #[allow(unsafe_code)]
        unsafe {
            match value {
                Some(v) => std::env::set_var(key, v),
                None => std::env::remove_var(key),
            }
        }
    }

    #[test]
    fn build_ba_config_reads_lm_env_settings() {
        let _guard = env_lock().lock().expect("env lock");
        let keys = [
            "KIKO_BA_WINDOW",
            "KIKO_BA_ITERS",
            "KIKO_BA_MIN_OBS",
            "KIKO_BA_HUBER_PX",
            "KIKO_BA_DAMPING",
            "KIKO_LM_FACTOR",
            "KIKO_LM_MIN",
            "KIKO_LM_MAX",
            "KIKO_BA_MOTION_WEIGHT",
        ];
        let saved: Vec<(String, Option<OsString>)> = keys
            .iter()
            .map(|&key| (key.to_string(), std::env::var_os(key)))
            .collect();

        set_env("KIKO_BA_WINDOW", "12");
        set_env("KIKO_BA_ITERS", "7");
        set_env("KIKO_BA_MIN_OBS", "9");
        set_env("KIKO_BA_HUBER_PX", "2.5");
        set_env("KIKO_BA_DAMPING", "0.002");
        set_env("KIKO_LM_FACTOR", "12.0");
        set_env("KIKO_LM_MIN", "0.000001");
        set_env("KIKO_LM_MAX", "5000");
        set_env("KIKO_BA_MOTION_WEIGHT", "0.25");

        let config = build_ba_config().expect("build config");
        assert_eq!(config.window(), 12);
        assert_eq!(config.max_iterations(), 7);
        assert_eq!(config.min_observations(), 9);
        assert!((config.huber_delta_px() - 2.5).abs() < 1e-6);
        assert!((config.lm().initial_lambda() - 0.002).abs() < 1e-9);
        assert!((config.lm().lambda_factor() - 12.0).abs() < 1e-9);
        assert!((config.lm().min_lambda() - 1e-6).abs() < 1e-12);
        assert!((config.lm().max_lambda() - 5000.0).abs() < 1e-6);
        assert!((config.motion_prior_weight() - 0.25).abs() < 1e-6);

        for (key, value) in saved {
            restore_env(&key, value);
        }
    }

    #[test]
    fn build_tracker_config_reads_loop_env_settings() {
        let _guard = env_lock().lock().expect("env lock");
        let keys = [
            "KIKO_LOOP_SIMILARITY_THRESHOLD",
            "KIKO_LOOP_DESCRIPTOR_MATCH_THRESHOLD",
            "KIKO_LOOP_MIN_INLIERS",
            "KIKO_LOOP_MAX_CANDIDATES",
            "KIKO_LOOP_TEMPORAL_GAP",
            "KIKO_LOOP_MIN_STREAK",
            "KIKO_LOOP_MAX_CORRECTION_TRANSLATION_M",
            "KIKO_LOOP_MAX_CORRECTION_ROTATION_DEG",
            "KIKO_LOOP_RANSAC_MAX_ITERATIONS",
            "KIKO_LOOP_RANSAC_THRESHOLD_PX",
            "KIKO_LOOP_RANSAC_MIN_INLIERS",
        ];
        let saved: Vec<(String, Option<OsString>)> = keys
            .iter()
            .map(|&key| (key.to_string(), std::env::var_os(key)))
            .collect();

        set_env("KIKO_LOOP_SIMILARITY_THRESHOLD", "0.80");
        set_env("KIKO_LOOP_DESCRIPTOR_MATCH_THRESHOLD", "0.72");
        set_env("KIKO_LOOP_MIN_INLIERS", "18");
        set_env("KIKO_LOOP_MAX_CANDIDATES", "5");
        set_env("KIKO_LOOP_TEMPORAL_GAP", "25");
        set_env("KIKO_LOOP_MIN_STREAK", "2");
        set_env("KIKO_LOOP_MAX_CORRECTION_TRANSLATION_M", "4.5");
        set_env("KIKO_LOOP_MAX_CORRECTION_ROTATION_DEG", "25");
        set_env("KIKO_LOOP_RANSAC_MAX_ITERATIONS", "150");
        set_env("KIKO_LOOP_RANSAC_THRESHOLD_PX", "1.75");
        set_env("KIKO_LOOP_RANSAC_MIN_INLIERS", "18");

        let config = build_tracker_config(
            TrackerDefaults {
                min_keyframe_points: 12,
                refresh_inliers: 12,
                min_inliers: 8,
            },
            KeypointLimit::try_from(1024).expect("keypoint limit"),
            DownscaleFactor::try_from(1).expect("downscale"),
        )
        .expect("tracker config");

        let loop_cfg = match config.loop_subsystem {
            LoopSubsystemConfig::LoopClosureOnly { loop_closure, .. }
            | LoopSubsystemConfig::LoopClosureAndRelocalization { loop_closure, .. } => {
                loop_closure
            }
            LoopSubsystemConfig::Disabled => panic!("loop subsystem should be enabled"),
        };
        assert!((loop_cfg.similarity_threshold() - 0.80).abs() < 1e-6);
        assert!((loop_cfg.descriptor_match_threshold() - 0.72).abs() < 1e-6);
        assert_eq!(loop_cfg.min_inliers(), 18);
        assert_eq!(loop_cfg.max_candidates(), 5);
        assert_eq!(loop_cfg.temporal_gap(), 25);
        assert_eq!(loop_cfg.min_streak(), 2);
        assert!((loop_cfg.max_correction_translation() - 4.5).abs() < 1e-6);
        assert!((loop_cfg.max_correction_rotation_deg() - 25.0).abs() < 1e-6);
        assert_eq!(loop_cfg.ransac().max_iterations, 150);
        assert!((loop_cfg.ransac().reprojection_threshold_px - 1.75).abs() < 1e-6);
        assert_eq!(loop_cfg.ransac().min_inliers, 18);

        for (key, value) in saved {
            restore_env(&key, value);
        }
    }

    #[test]
    fn build_tracker_config_rejects_invalid_loop_env() {
        let _guard = env_lock().lock().expect("env lock");
        let key = "KIKO_LOOP_MAX_CORRECTION_ROTATION_DEG";
        let saved = std::env::var_os(key);
        set_env(key, "181.0");

        let result = build_tracker_config(
            TrackerDefaults {
                min_keyframe_points: 12,
                refresh_inliers: 12,
                min_inliers: 8,
            },
            KeypointLimit::try_from(1024).expect("keypoint limit"),
            DownscaleFactor::try_from(1).expect("downscale"),
        );
        assert!(
            result.is_err(),
            "invalid loop config should return an error"
        );

        restore_env(key, saved);
    }

    #[test]
    fn summarize_bench_reports_exact_stage_metrics() {
        let accum = BenchAccum {
            read_samples: 4,
            processed: 3,
            matches_nonzero: 2,
            total_matches: 12,
            sum_read_left: Duration::from_millis(20),
            sum_read_right: Duration::from_millis(20),
            sum_pairing: Duration::from_millis(10),
            sum_read_bytes: 8 * 1024 * 1024,
            sum_sp_left: Duration::from_millis(9),
            sum_sp_right: Duration::from_millis(12),
            sum_lightglue: Duration::from_millis(15),
            sum_total_success: Duration::from_millis(45),
            sum_inference_attempt: Duration::from_millis(60),
            ..BenchAccum::default()
        };
        let summary = summarize_bench(&accum, Duration::from_secs(2));

        assert!((summary.wall_fps - 1.5).abs() < 1e-9);
        assert!((summary.reader_stage_fps - 80.0).abs() < 1e-9);
        assert!((summary.inference_attempt_fps - (4.0 / 0.06)).abs() < 1e-9);
        assert!((summary.successful_inference_fps - (3.0 / 0.045)).abs() < 1e-9);
        assert!((summary.match_rate - (2.0 / 3.0)).abs() < 1e-9);
        assert!((summary.avg_matches_per_processed_pair - 4.0).abs() < 1e-9);
        assert!((summary.avg_matches_per_nonzero_pair - 6.0).abs() < 1e-9);
    }
}
