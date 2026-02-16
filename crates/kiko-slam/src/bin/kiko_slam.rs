use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use clap::{Args, Parser, Subcommand, ValueEnum};

use kiko_slam::dataset::DatasetReader;
use kiko_slam::dense::{self, DenseConfig, command_mapper, ring_buffer::DepthRingBuffer};
use kiko_slam::{
    BackendConfig, DenseCommand, DepthImage, DownscaleFactor, GlobalDescriptorConfig,
    InferenceBackend, InferencePipeline, KeyframePolicy, KeypointLimit, LightGlue, LmConfig,
    LocalBaConfig, LoopClosureConfig, PinholeIntrinsics, RansacConfig, RectifiedStereo,
    RectifiedStereoConfig, RedundancyPolicy, RelocalizationConfig, RerunSink, SlamTracker,
    SuperPoint, TrackerConfig, TriangulationConfig, TriangulationError, Triangulator,
    VizDecimation, VizPacket,
};

use kiko_slam::env::{env_bool, env_f32, env_usize};

#[cfg(feature = "record")]
use kiko_slam::{DenseStats, Frame, Point3, Pose, Raw, ReconState};

#[cfg(feature = "record")]
use kiko_slam::dataset::{
    Calibration, CameraIntrinsics, DatasetWriter, DepthMeta, ImuMeta, Meta, MonoMeta,
};
#[cfg(feature = "record")]
use kiko_slam::{
    ChannelCapacity, DiagnosticEvent, DropPolicy, DropReceiver, FrameDiagnostics, FrameId,
    PairingWindowNs, SendOutcome, SensorId, StereoPair, StereoPairer, SystemHealth,
    bounded_channel, oak_to_depth_image, oak_to_frame,
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
        Self(KeypointLimit::try_from(DEFAULT_MAX_KEYPOINTS).expect("default max keypoints"))
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

    fn into_models(
        self,
    ) -> (
        SuperPoint,
        SuperPoint,
        LightGlue,
        KeypointLimit,
        DownscaleFactor,
    ) {
        (
            self.superpoint_left,
            self.superpoint_right,
            self.lightglue,
            self.key_limit,
            self.downscale,
        )
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
        RectifiedStereoConfig {
            max_principal_delta_px: args.rectify_tolerance,
            allow_unrectified: args.allow_unrectified,
        },
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

struct OfflineDepthCursor {
    entries: Vec<(i64, PathBuf)>,
    next: usize,
    width: u32,
    height: u32,
    frame_seq: u64,
}

impl OfflineDepthCursor {
    fn new(
        dataset_root: &Path,
        depth_meta: Option<&kiko_slam::dataset::DepthMeta>,
    ) -> Result<Option<Self>, std::io::Error> {
        let Some(depth_meta) = depth_meta else {
            return Ok(None);
        };
        let frames_dir = dataset_root.join(kiko_slam::dataset::format::FRAMES_DIR);
        let mut entries = Vec::new();
        for dir_entry in std::fs::read_dir(&frames_dir)? {
            let dir_entry = dir_entry?;
            let file_name = dir_entry.file_name();
            let file_name = file_name.to_string_lossy();
            let Some((timestamp_ns, sensor)) =
                kiko_slam::dataset::format::parse_frame_filename(file_name.as_ref())
            else {
                continue;
            };
            if sensor == "depth" {
                entries.push((timestamp_ns, dir_entry.path()));
            }
        }
        entries.sort_by_key(|(timestamp_ns, _)| *timestamp_ns);
        if entries.is_empty() {
            return Ok(None);
        }
        Ok(Some(Self {
            entries,
            next: 0,
            width: depth_meta.width,
            height: depth_meta.height,
            frame_seq: 0,
        }))
    }

    fn push_until(
        &mut self,
        timestamp: kiko_slam::Timestamp,
        ring: &mut DepthRingBuffer,
    ) -> Result<(), std::io::Error> {
        let cutoff = timestamp.as_nanos();
        while let Some((depth_ts, path)) = self.entries.get(self.next) {
            if *depth_ts > cutoff {
                break;
            }
            let depth = read_depth_image_file(
                path,
                self.width,
                self.height,
                kiko_slam::FrameId::new(self.frame_seq),
                kiko_slam::Timestamp::from_nanos(*depth_ts),
            )?;
            self.frame_seq = self.frame_seq.saturating_add(1);
            ring.push(depth);
            self.next = self.next.saturating_add(1);
        }
        Ok(())
    }
}

fn read_depth_image_file(
    path: &Path,
    width: u32,
    height: u32,
    frame_id: kiko_slam::FrameId,
    timestamp: kiko_slam::Timestamp,
) -> Result<DepthImage, std::io::Error> {
    let bytes = std::fs::read(path)?;
    let pixel_count = (width as usize).saturating_mul(height as usize);
    let expected_len = pixel_count.saturating_mul(std::mem::size_of::<f32>());
    if bytes.len() != expected_len {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "depth file {} length mismatch: expected {expected_len}, got {}",
                path.display(),
                bytes.len()
            ),
        ));
    }

    let mut depth_m = Vec::with_capacity(pixel_count);
    for chunk in bytes.chunks_exact(4) {
        depth_m.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }

    DepthImage::new(frame_id, timestamp, width, height, depth_m).map_err(|err| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("invalid depth image at {}: {err}", path.display()),
        )
    })
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
        RectifiedStereoConfig {
            max_principal_delta_px: args.rectify_tolerance,
            allow_unrectified: args.allow_unrectified,
        },
    )?;
    let intrinsics = PinholeIntrinsics::try_from(&reader.calibration().left)?;

    let (superpoint_left, superpoint_right, lightglue, key_limit, downscale) =
        inference.into_models();

    let min_keyframe_points = env_usize("KIKO_KEYFRAME_MIN_POINTS").unwrap_or(12);
    let refresh_inliers = env_usize("KIKO_KEYFRAME_REFRESH_INLIERS").unwrap_or(12);
    let parallax_px = env_f32("KIKO_KEYFRAME_PARALLAX_PX").unwrap_or(40.0);
    let min_covisibility = env_f32("KIKO_KEYFRAME_COVISIBILITY").unwrap_or(0.6);
    let redundant_covisibility = env_f32("KIKO_KEYFRAME_REDUNDANT_COVISIBILITY").unwrap_or(0.9);
    let min_inliers = env_usize("KIKO_TRACK_MIN_INLIERS").unwrap_or(8);
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
    let loop_closure = if env_bool("KIKO_LOOP_CLOSURE").unwrap_or(true) {
        Some(LoopClosureConfig::default())
    } else {
        None
    };
    let global_descriptor = if env_bool("KIKO_LEARNED_DESCRIPTORS").unwrap_or(true) {
        Some(GlobalDescriptorConfig::new(
            env_usize("KIKO_DESCRIPTOR_QUEUE_DEPTH").unwrap_or(2),
        )?)
    } else {
        None
    };
    let relocalization = if env_bool("KIKO_RELOCALIZATION").unwrap_or(true) {
        Some(RelocalizationConfig::default())
    } else {
        None
    };
    let loop_closure_enabled = loop_closure.is_some();
    let learned_descriptors_enabled = global_descriptor.is_some();
    let relocalization_enabled = relocalization.is_some();
    let tracker_config = TrackerConfig {
        max_keypoints: key_limit,
        downscale,
        min_keyframe_points,
        ransac,
        triangulation: TriangulationConfig::default(),
        keyframe_policy,
        ba: ba_config,
        redundancy,
        backend,
        loop_closure,
        global_descriptor,
        relocalization,
    };

    eprintln!(
        "tracker: keyframe_min_points={} refresh_inliers={} parallax_px={:.1} min_covisibility={:.2} redundant_covisibility={:.2} min_inliers={} downscale={} max_keypoints={} loop_closure={} learned_descriptors={} relocalization={}",
        min_keyframe_points,
        refresh_inliers,
        parallax_px,
        min_covisibility,
        redundant_covisibility,
        min_inliers,
        downscale.get(),
        key_limit.get(),
        loop_closure_enabled,
        learned_descriptors_enabled,
        relocalization_enabled
    );

    let rec = build_recording(args, "kiko-slam-dataset-odometry")?;
    let mut sink = RerunSink::new(rec, decimation);
    let mut tracker = SlamTracker::new(
        superpoint_left,
        superpoint_right,
        lightglue,
        rectified,
        intrinsics,
        tracker_config,
    );
    let dense_enabled = env_bool("KIKO_DENSE").unwrap_or(false);
    let depth_ring_capacity = env_usize("KIKO_OFFLINE_DEPTH_RING_CAPACITY")
        .unwrap_or(8)
        .max(4);
    let mut depth_ring = DepthRingBuffer::new(depth_ring_capacity);
    let mut depth_cursor = if dense_enabled {
        match OfflineDepthCursor::new(&args.dataset.path, reader.meta().depth.as_ref()) {
            Ok(cursor) => cursor,
            Err(err) => {
                eprintln!("failed to initialize offline depth stream: {err}");
                None
            }
        }
    } else {
        None
    };
    let mut dense_state = dense_enabled.then(|| dense::DenseState::new(&DenseConfig::default()));
    let mut dense_generation = 0_u64;
    if dense_enabled {
        eprintln!(
            "offline dense enabled: depth_stream={} ring_capacity={}",
            depth_cursor.is_some(),
            depth_ring_capacity
        );
    }

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

        let left = pair.left.clone();
        let right = pair.right.clone();
        if let Some(cursor) = depth_cursor.as_mut() {
            if let Err(err) = cursor.push_until(left.timestamp(), &mut depth_ring) {
                eprintln!("offline depth decode failed; disabling dense: {err}");
                depth_cursor = None;
                dense_state = None;
            }
        }

        match tracker.process(pair) {
            Ok(output) => {
                let timestamp = left.timestamp();
                let dense_stats = if let Some(state) = dense_state.as_mut() {
                    let correction = tracker.take_pending_loop_correction();
                    let cmds = command_mapper::map_output_to_dense_commands(
                        &output,
                        correction.as_deref(),
                        &depth_ring,
                        timestamp,
                        &mut dense_generation,
                    );
                    let mut latest = None;
                    for cmd in cmds {
                        latest = Some(dense::process_dense_command(state, cmd));
                    }
                    latest
                } else {
                    None
                };
                if let Some(depth) =
                    depth_ring.find_closest(timestamp, command_mapper::MAX_ASSOCIATION_WINDOW_NS)
                {
                    if let Err(err) = sink.log_depth(&depth) {
                        eprintln!("rerun depth error: {err}");
                    }
                }
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
                if let Some(stats) = dense_stats.as_ref() {
                    if let Err(err) = sink.log_dense_stats(timestamp, stats) {
                        eprintln!("rerun dense stats error: {err}");
                    }
                }
                processed += 1;
            }
            Err(err) => {
                inference_errors += 1;
                if let Some(state) = dense_state.as_mut() {
                    if let Some(correction) = tracker.take_pending_loop_correction() {
                        dense_generation = dense_generation.saturating_add(1);
                        let stats = dense::process_dense_command(
                            state,
                            DenseCommand::RebuildFromSnapshot {
                                corrected_poses: correction,
                                generation: dense_generation,
                            },
                        );
                        if let Err(log_err) = sink.log_dense_stats(left.timestamp(), &stats) {
                            eprintln!("rerun dense stats error: {log_err}");
                        }
                    }
                }
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
    let mut processed = 0usize;
    let mut matches_nonzero = 0usize;
    let mut total_matches = 0usize;
    let mut read_errors = 0usize;
    let mut pairing_errors = 0usize;
    let mut inference_errors = 0usize;
    let mut sum_read_left = Duration::ZERO;
    let mut sum_read_right = Duration::ZERO;
    let mut sum_pairing = Duration::ZERO;
    let mut sum_read_bytes = 0usize;
    let mut sum_sp_left = Duration::ZERO;
    let mut sum_sp_right = Duration::ZERO;
    let mut sum_lightglue = Duration::ZERO;
    let mut sum_total = Duration::ZERO;

    let start = Instant::now();
    for sample in reader.timed_pairs() {
        let sample = match sample {
            Ok(sample) => sample,
            Err(err) => {
                match err {
                    kiko_slam::dataset::DatasetError::PairingFailed { .. } => {
                        pairing_errors += 1;
                    }
                    _ => read_errors += 1,
                }
                eprintln!("read error: {err}");
                continue;
            }
        };
        let pair = sample.pair;
        sum_read_left += sample.timings.left_read;
        sum_read_right += sample.timings.right_read;
        sum_pairing += sample.timings.pairing;
        sum_read_bytes += sample.timings.left_bytes + sample.timings.right_bytes;

        match pipeline.process_pair_timed(pair) {
            Ok((packet, timings)) => {
                let matches = packet.matches();
                if !matches.is_empty() {
                    matches_nonzero += 1;
                    total_matches += matches.len();
                }
                sum_sp_left += timings.superpoint_left;
                sum_sp_right += timings.superpoint_right;
                sum_lightglue += timings.lightglue;
                sum_total += timings.total;
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
    eprintln!("errors: read={read_errors} pairing={pairing_errors} inference={inference_errors}");

    if processed > 0 {
        let denom = processed as f64;
        let avg_sp_left_ms = (sum_sp_left.as_secs_f64() * 1000.0) / denom;
        let avg_sp_right_ms = (sum_sp_right.as_secs_f64() * 1000.0) / denom;
        let avg_lightglue_ms = (sum_lightglue.as_secs_f64() * 1000.0) / denom;
        let avg_total_ms = (sum_total.as_secs_f64() * 1000.0) / denom;
        let overhead = sum_total.saturating_sub(sum_sp_left + sum_sp_right + sum_lightglue);
        let avg_overhead_ms = (overhead.as_secs_f64() * 1000.0) / denom;
        let total_ms = sum_total.as_secs_f64().max(1e-9);
        let pct_sp_left = (sum_sp_left.as_secs_f64() / total_ms) * 100.0;
        let pct_sp_right = (sum_sp_right.as_secs_f64() / total_ms) * 100.0;
        let pct_lightglue = (sum_lightglue.as_secs_f64() / total_ms) * 100.0;
        let pct_overhead = (overhead.as_secs_f64() / total_ms) * 100.0;

        eprintln!(
            "timings avg ms: sp_left={avg_sp_left_ms:.2} sp_right={avg_sp_right_ms:.2} lightglue={avg_lightglue_ms:.2} overhead={avg_overhead_ms:.2} total={avg_total_ms:.2}"
        );
        eprintln!(
            "timings pct: sp_left={pct_sp_left:.1}% sp_right={pct_sp_right:.1}% lightglue={pct_lightglue:.1}% overhead={pct_overhead:.1}%"
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
        return Err("no paired frames processed".into());
    }
    if matches_nonzero == 0 {
        return Err("no nonzero matches; check models/data".into());
    }
    if inference_errors > 0 {
        return Err("inference errors encountered during run".into());
    }

    Ok(())
}

#[cfg(feature = "record")]
const DEFAULT_PAIRING_WINDOW_NS: i64 = 5_000_000;
#[cfg(feature = "record")]
const DEFAULT_PAIRER_MAX_PENDING_PER_SIDE: usize = 64;

#[cfg(feature = "record")]
fn load_pairing_window() -> PairingWindowNs {
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
    PairingWindowNs::new(window_ns).unwrap_or_else(|err| {
        eprintln!("invalid pairing window from env ({err}); using default");
        PairingWindowNs::new(DEFAULT_PAIRING_WINDOW_NS).expect("default pairing window is valid")
    })
}

#[cfg(feature = "record")]
fn load_pairer_max_pending_per_side() -> usize {
    env_usize("KIKO_PAIRER_MAX_PENDING_PER_SIDE")
        .unwrap_or(DEFAULT_PAIRER_MAX_PENDING_PER_SIDE)
        .max(1)
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
    let pairing_window = load_pairing_window();
    let pairer_max_pending = load_pairer_max_pending_per_side();
    let mut pairer = StereoPairer::new_with_max_pending(pairing_window, pairer_max_pending);
    let start = Instant::now();

    eprintln!("recording... press ctrl+c to stop");

    while running.load(Ordering::Relaxed) {
        let mut got_any = false;

        match device.mono_left(0) {
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

        match device.mono_right(0) {
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
            match device.depth(0) {
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

        while let Some(pair) = pairer.next_pair()? {
            writer.write_frame(&pair.left);
            writer.write_frame(&pair.right);
            pair_count += 1;

            if pair_count % 30 == 0 {
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
        pairer.max_pending_per_side(),
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
    health: SystemHealth,
    diagnostics: FrameDiagnostics,
    events: Vec<DiagnosticEvent>,
    dense_stats: Option<DenseStats>,
}

#[cfg(feature = "record")]
fn drain_depth_batch(rx: &DropReceiver<DepthImage>) -> Vec<DepthImage> {
    let mut depths = Vec::new();
    loop {
        match rx.try_recv() {
            Ok(depth) => depths.push(depth),
            Err(crossbeam_channel::TryRecvError::Empty) => break,
            Err(crossbeam_channel::TryRecvError::Disconnected) => break,
        }
    }
    depths
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
    let depth_ring_capacity = depth_queue_depth.max(4);

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

    let pairing_window = load_pairing_window();
    let pairer_max_pending = load_pairer_max_pending_per_side();
    let mut pairer = StereoPairer::new_with_max_pending(pairing_window, pairer_max_pending);

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
    let (superpoint_left, superpoint_right, lightglue, key_limit, downscale) =
        inference.into_models();

    let calibration = build_calibration(&device, device.stereo_baseline_m(), &mono_config);
    let rectified = RectifiedStereo::from_calibration(&calibration)?;
    let intrinsics = PinholeIntrinsics::try_from(&calibration.left)?;

    let min_keyframe_points = env_usize("KIKO_KEYFRAME_MIN_POINTS").unwrap_or(80);
    let refresh_inliers = env_usize("KIKO_KEYFRAME_REFRESH_INLIERS").unwrap_or(20);
    let parallax_px = env_f32("KIKO_KEYFRAME_PARALLAX_PX").unwrap_or(40.0);
    let min_covisibility = env_f32("KIKO_KEYFRAME_COVISIBILITY").unwrap_or(0.6);
    let redundant_covisibility = env_f32("KIKO_KEYFRAME_REDUNDANT_COVISIBILITY").unwrap_or(0.9);
    let min_inliers = env_usize("KIKO_TRACK_MIN_INLIERS").unwrap_or(15);
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
    let loop_closure = if env_bool("KIKO_LOOP_CLOSURE").unwrap_or(true) {
        Some(LoopClosureConfig::default())
    } else {
        None
    };
    let global_descriptor = if env_bool("KIKO_LEARNED_DESCRIPTORS").unwrap_or(true) {
        Some(GlobalDescriptorConfig::new(
            env_usize("KIKO_DESCRIPTOR_QUEUE_DEPTH").unwrap_or(2),
        )?)
    } else {
        None
    };
    let relocalization = if env_bool("KIKO_RELOCALIZATION").unwrap_or(true) {
        Some(RelocalizationConfig::default())
    } else {
        None
    };
    let loop_closure_enabled = loop_closure.is_some();
    let learned_descriptors_enabled = global_descriptor.is_some();
    let relocalization_enabled = relocalization.is_some();
    let tracker_config = TrackerConfig {
        max_keypoints: key_limit,
        downscale,
        min_keyframe_points,
        ransac,
        triangulation: TriangulationConfig::default(),
        keyframe_policy,
        ba: ba_config,
        redundancy,
        backend,
        loop_closure,
        global_descriptor,
        relocalization,
    };

    eprintln!(
        "tracker: keyframe_min_points={} refresh_inliers={} parallax_px={:.1} min_covisibility={:.2} redundant_covisibility={:.2} min_inliers={} downscale={} max_keypoints={} loop_closure={} learned_descriptors={} relocalization={} pair_queue_depth={} viz_queue_depth={} depth_enabled={} depth_queue_depth={} pairing_window_ns={} pairer_max_pending_per_side={}",
        min_keyframe_points,
        refresh_inliers,
        parallax_px,
        min_covisibility,
        redundant_covisibility,
        min_inliers,
        downscale.get(),
        key_limit.get(),
        loop_closure_enabled,
        learned_descriptors_enabled,
        relocalization_enabled,
        pair_queue_depth,
        viz_queue_depth,
        depth_enabled,
        depth_queue_depth,
        pairer.window().as_ns(),
        pairer.max_pending_per_side()
    );

    // Dense reconstruction channels and worker thread.
    let dense_enabled = depth_enabled && env_bool("KIKO_DENSE").unwrap_or(false);
    let dense_data_queue_depth = env_usize("KIKO_DENSE_DATA_QUEUE_DEPTH").unwrap_or(4);
    let dense_ctrl_queue_depth = env_usize("KIKO_DENSE_CTRL_QUEUE_DEPTH")
        .unwrap_or(64)
        .max(1);

    // Create dense channels. Control is bounded (to avoid unbounded memory growth),
    // data uses DropNewest backpressure for IntegrateKeyframe.
    let mut dense_ctrl_tx: Option<crossbeam_channel::Sender<kiko_slam::DenseCommand>> = None;
    let mut dense_ctrl_rx_for_worker: Option<crossbeam_channel::Receiver<kiko_slam::DenseCommand>> =
        None;
    let mut dense_data_tx: Option<kiko_slam::DropSender<kiko_slam::DenseCommand>> = None;
    let mut dense_data_rx_for_worker: Option<kiko_slam::DropReceiver<kiko_slam::DenseCommand>> =
        None;
    let mut dense_data_stats_handle: Option<kiko_slam::ChannelStatsHandle> = None;
    let mut dense_stats_tx_for_worker: Option<kiko_slam::DropSender<DenseStats>> = None;
    let mut dense_stats_rx: Option<kiko_slam::DropReceiver<DenseStats>> = None;

    if dense_enabled {
        let (ctrl_tx, ctrl_rx) = crossbeam_channel::bounded(dense_ctrl_queue_depth);
        let data_cap = ChannelCapacity::try_from(dense_data_queue_depth)?;
        let (data_tx, data_rx, data_stats) = bounded_channel(data_cap, DropPolicy::DropNewest);
        let stats_cap = ChannelCapacity::try_from(1_usize)?;
        let (stats_tx, stats_rx_inner, _stats_handle) =
            bounded_channel(stats_cap, DropPolicy::DropNewest);
        dense_ctrl_tx = Some(ctrl_tx);
        dense_ctrl_rx_for_worker = Some(ctrl_rx);
        dense_data_tx = Some(data_tx);
        dense_data_rx_for_worker = Some(data_rx);
        dense_data_stats_handle = Some(data_stats);
        dense_stats_tx_for_worker = Some(stats_tx);
        dense_stats_rx = Some(stats_rx_inner);
    }

    let dense_handle = if let (Some(ctrl_rx), Some(data_rx), stats_tx) = (
        dense_ctrl_rx_for_worker.take(),
        dense_data_rx_for_worker.take(),
        dense_stats_tx_for_worker.take(),
    ) {
        let cfg = DenseConfig::default();
        Some(thread::spawn(move || {
            kiko_slam::dense::run_dense_worker(
                &cfg,
                &ctrl_rx,
                data_rx.as_receiver(),
                stats_tx.as_ref(),
            );
        }))
    } else {
        None
    };

    let inference_handle = thread::spawn(move || -> Result<(), String> {
        let mut tracker = SlamTracker::new(
            superpoint_left,
            superpoint_right,
            lightglue,
            rectified,
            intrinsics,
            tracker_config,
        );
        let depth_rx = depth_rx;
        let mut depth_ring = DepthRingBuffer::new(depth_ring_capacity);
        let mut dense_generation: u64 = 0;
        let mut dense_ctrl_tx = dense_ctrl_tx;
        let mut dense_data_tx = dense_data_tx;
        let dense_stats_rx = dense_stats_rx;
        let mut dense_active = dense_enabled;
        let mut dense_data_dropped_newest: u64 = 0;
        let mut depth_reorder_warnings_seen: u64 = 0;

        for pair in pair_rx.iter() {
            let left = pair.left.clone();
            let right = pair.right.clone();
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
                Ok(Ok(output)) => {
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
                            if cmd.is_control() {
                                if let Some(ref tx) = dense_ctrl_tx {
                                    match tx.send_timeout(cmd, Duration::from_millis(5)) {
                                        Ok(()) => {}
                                        Err(crossbeam_channel::SendTimeoutError::Timeout(_)) => {
                                            dense_active = false;
                                            dense_ctrl_tx = None;
                                            dense_data_tx = None;
                                            eprintln!(
                                                "dense control queue saturated; disabling dense"
                                            );
                                            break;
                                        }
                                        Err(crossbeam_channel::SendTimeoutError::Disconnected(
                                            _,
                                        )) => {
                                            dense_active = false;
                                            dense_ctrl_tx = None;
                                            dense_data_tx = None;
                                            eprintln!(
                                                "dense control channel disconnected; disabling dense"
                                            );
                                            break;
                                        }
                                    }
                                }
                            } else if let Some(ref tx) = dense_data_tx {
                                match tx.try_send(cmd) {
                                    SendOutcome::Enqueued | SendOutcome::DroppedOldest => {}
                                    SendOutcome::DroppedNewest => {
                                        dense_data_dropped_newest =
                                            dense_data_dropped_newest.saturating_add(1);
                                    }
                                    SendOutcome::Disconnected => {
                                        dense_active = false;
                                        dense_ctrl_tx = None;
                                        dense_data_tx = None;
                                        eprintln!(
                                            "dense data channel disconnected; disabling dense"
                                        );
                                        break;
                                    }
                                }
                            }
                        }
                        // Drain latest dense stats for viz.
                        dense_stats_rx.as_ref().and_then(|rx| {
                            let mut latest = None;
                            while let Ok(s) = rx.try_recv() {
                                latest = Some(s);
                            }
                            latest
                        })
                    } else {
                        None
                    };
                    if let Some(ref stats) = dense_stats {
                        if stats.state == ReconState::Down {
                            dense_active = false;
                            dense_ctrl_tx = None;
                            dense_data_tx = None;
                            eprintln!("dense worker entered Down state; disabling dense");
                        }
                    }

                    let health = output.health.clone();
                    let mut packet = None;
                    let mut points = None;
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
                        health,
                        diagnostics: output.diagnostics,
                        events: output.events,
                        dense_stats,
                    };
                    if matches!(viz_tx.try_send(msg), SendOutcome::Disconnected) {
                        return Err("viz channel disconnected".to_string());
                    }
                }
                Ok(Err(err)) => {
                    if dense_active {
                        if let Some(correction) = tracker.take_pending_loop_correction() {
                            dense_generation = dense_generation.saturating_add(1);
                            let rebuild_cmd = DenseCommand::RebuildFromSnapshot {
                                corrected_poses: correction,
                                generation: dense_generation,
                            };
                            if let Some(ref tx) = dense_ctrl_tx {
                                match tx.send_timeout(rebuild_cmd, Duration::from_millis(5)) {
                                    Ok(()) => {}
                                    Err(crossbeam_channel::SendTimeoutError::Timeout(_)) => {
                                        dense_active = false;
                                        dense_ctrl_tx = None;
                                        dense_data_tx = None;
                                        eprintln!(
                                            "dense control queue saturated after tracker error; disabling dense"
                                        );
                                    }
                                    Err(crossbeam_channel::SendTimeoutError::Disconnected(_)) => {
                                        dense_active = false;
                                        dense_ctrl_tx = None;
                                        dense_data_tx = None;
                                        eprintln!(
                                            "dense control disconnected after tracker error; disabling dense"
                                        );
                                    }
                                }
                            }
                        }
                    }
                    eprintln!("tracker error: {err}");
                }
                Err(payload) => {
                    return Err(format!(
                        "inference panic while processing frame: {}",
                        panic_payload_to_string(payload.as_ref())
                    ));
                }
            }
        }
        if dense_data_dropped_newest > 0 {
            eprintln!("dense data dropped_newest (inference view): {dense_data_dropped_newest}");
        }
        if depth_reorder_warnings_seen > 0 {
            eprintln!("depth reorder warnings observed: {depth_reorder_warnings_seen}");
        }
        Ok(())
    });

    let decimation = args.rerun_decimation.get();
    let viz_running = Arc::clone(&running);
    let viz_handle = thread::spawn(move || -> Result<(), String> {
        let rec = match rerun::RecordingStreamBuilder::new("kiko-slam-live").connect_grpc() {
            Ok(rec) => rec,
            Err(err) => {
                viz_running.store(false, Ordering::SeqCst);
                eprintln!("failed to connect to rerun viewer: {err}");
                return Err(format!("failed to connect to rerun viewer: {err}"));
            }
        };

        let mut sink = RerunSink::new(rec, decimation);
        for msg in viz_rx.iter() {
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
            if let Some(ref dense_stats) = msg.dense_stats {
                if let Err(err) = sink.log_dense_stats(msg.left.timestamp(), dense_stats) {
                    eprintln!("rerun dense stats error: {err}");
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

        match device.mono_left(0) {
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

        match device.mono_right(0) {
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
            match device.depth(0) {
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

        while let Some(pair) = pairer.next_pair()? {
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
            panic_payload_to_string(payload.as_ref())
        ))
    })?;
    if let Err(err) = inference_result {
        return Err(std::io::Error::other(err).into());
    }

    let viz_result = viz_handle.join().map_err(|payload| {
        std::io::Error::other(format!(
            "viz thread panicked: {}",
            panic_payload_to_string(payload.as_ref())
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
                panic_payload_to_string(payload.as_ref())
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
    if let Some(dense_data_stats_handle) = dense_data_stats_handle {
        let dense_data_snapshot = dense_data_stats_handle.snapshot();
        eprintln!(
            "dense data queue stats: enqueued={}, dropped_oldest={}, dropped_newest={}, disconnected={}",
            dense_data_snapshot.enqueued,
            dense_data_snapshot.dropped_oldest,
            dense_data_snapshot.dropped_newest,
            dense_data_snapshot.disconnected
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
fn panic_payload_to_string(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        return (*message).to_string();
    }
    if let Some(message) = payload.downcast_ref::<String>() {
        return message.clone();
    }
    "unknown panic payload".to_string()
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
    let window = env_usize("KIKO_BA_WINDOW").unwrap_or(10);
    let iters = env_usize("KIKO_BA_ITERS").unwrap_or(6);
    let min_obs = env_usize("KIKO_BA_MIN_OBS").unwrap_or(8);
    let huber = env_f32("KIKO_BA_HUBER_PX").unwrap_or(3.0);
    let initial_lambda = env_f32("KIKO_BA_DAMPING").unwrap_or(1e-3);
    let lambda_factor = env_f32("KIKO_LM_FACTOR").unwrap_or(10.0);
    let min_lambda = env_f32("KIKO_LM_MIN").unwrap_or(1e-8);
    let max_lambda = env_f32("KIKO_LM_MAX").unwrap_or(1e4);
    let motion = env_f32("KIKO_BA_MOTION_WEIGHT").unwrap_or(0.0);
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
    use super::build_ba_config;
    use std::ffi::OsString;
    use std::sync::{Mutex, OnceLock};

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
}
