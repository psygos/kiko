use std::num::{NonZeroU16, NonZeroUsize};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use clap::{Args, Parser, Subcommand, ValueEnum};

use kiko_slam::dataset::{
    DatasetDepthCursor, DatasetError, DatasetReader, DepthOpticalFrame, DepthProjectionContract,
};
use kiko_slam::dense::{
    self, command_mapper,
    occupancy::{
        DepthCameraModel, DepthRangeMeters, DepthToTrackingCamera, HeightRangeMeters,
        OccupancyConfig, OccupancyError, OccupancyEvidenceModel, OccupancyGridGeometry,
        WorldToOccupancy,
    },
    occupancy_runtime::{
        OccupancyRuntime, OccupancyRuntimeConfig, OccupancyRuntimeError, OccupancySnapshotCadence,
        TimedOccupancySnapshot,
    },
    ring_buffer::DepthRingBuffer,
};
use kiko_slam::{
    BackendConfig, DenseStats, DepthImage, DownscaleFactor, FrameDimensions, FrameId,
    GlobalDescriptorConfig, InferenceBackend, InferencePipeline, KeyframePolicy, KeypointLimit,
    LightGlue, LmConfig, LocalBaConfig, LoopClosureConfig, LoopSubsystemConfig, PinholeIntrinsics,
    PipelineError, PipelineTimingError, PipelineWallBreakdown, RansacConfig, RectifiedStereo,
    RectifiedStereoConfig, RectifiedStereoConfigError, RedundancyPolicy, RelocalizationConfig,
    RerunSink, RerunSinkConfig, SlamTracker, SuperPoint, TrackerConfig, TrackerError,
    TriangulationConfig, TriangulationError, Triangulator, VizDecimation, VizError, VizFlushError,
    VizLogError, VizPacket,
};

use kiko_slam::env::{env_bool, env_f32, env_f64, env_string, env_u32, env_usize};

#[cfg(any(feature = "record", test))]
use kiko_slam::{ChannelCapacity, DenseCommandSendOutcome};

#[cfg(feature = "record")]
use kiko_slam::env::{EnvError, env_u64};

#[cfg(feature = "record")]
use kiko_slam::{CameraPoint3, DepthImageError, Frame, FrameError, Raw, ReconState};

#[cfg(all(test, not(feature = "record")))]
use kiko_slam::HostMonotonicTimestamp;
#[cfg(feature = "record")]
use kiko_slam::dataset::{
    Backpressure, Calibration, CameraIntrinsics, DatasetWriteError, DatasetWriter,
    DatasetWriterConfig, DatasetWriterHandle, DepthMeta, ImuExtrinsicProvenance, ImuMeta,
    ImuStreamMetadata, Meta, MonoMeta, PairedDatasetWriter, WriteOutcome,
};
#[cfg(any(feature = "record", test))]
use kiko_slam::navigation::NavigationGoalArg;
#[cfg(all(feature = "record", feature = "actuation"))]
use kiko_slam::navigation::actuation::{LiveActuationError, PhysicalActuationSession};
#[cfg(any(feature = "record", test))]
use kiko_slam::navigation::mpc::HostMonotonicClockReadError;
#[cfg(feature = "record")]
use kiko_slam::navigation::mpc::{HostMonotonicClock, MpcConfigV1};
#[cfg(feature = "record")]
use kiko_slam::navigation::{
    ControlPeriodNs, CoordinatorAdmissionError, CoordinatorTickError, CoordinatorTickOutcome,
    GlobalPlannerConfig, LocalCostmap, MAX_SHADOW_NAVIGATION_CONFIG_JSON_BYTES,
    NAVIGATION_INGRESS_STREAM_FILE, NavigationClockEpoch, NavigationIngressBoundaryError,
    NavigationIngressCapacity, NavigationIngressSidecarDescriptor,
    NavigationIngressStreamWriteError, NavigationIngressWriter, NavigationRecordingId,
    NavigationRecordingIdError, PathReferenceBuilderV1, PendingVisualAttemptIngress,
    PlanarOdometry, SafetyDecisionOutcome, ShadowNavigationConfigV1, ShadowNavigationCoordinator,
    ShadowSafetySupervisor, SolverBudgetNs, VisualAdmission, VisualAdmissionError,
    VisualAttemptOutcome,
};
#[cfg(all(feature = "record", feature = "actuation"))]
use kiko_slam::navigation::{
    MAX_NAVIGATION_ACTUATION_CONFIG_JSON_BYTES, NavigationActuationConfigV1,
};
#[cfg(feature = "record")]
use kiko_slam::{
    DenseCommandQueueStatsHandle, DenseCommandReceiver, DenseCommandSender, DepthObservation,
    DepthObservationError, DeviceSessionId, DropPolicy, DropReceiver, DropSender,
    HostMonotonicTimestamp, ImuReport, InertialOrderingError, InertialValueError,
    PairingConfigError, PairingInputError, PairingWindowNs, SendOutcome, SensorId,
    StereoObservation, StereoObservationError, StereoPairer, TrackerInitError, TrackerOutput,
    VizConfigError, bounded_channel, dense_command_channel, depth_router, imu_report_router,
    oak_to_depth_image, oak_to_frame, oak_to_imu_report,
};
#[cfg(feature = "record")]
use oak_sys::{
    CalibrationError as OakCalibrationError, CloseError as OakCloseError,
    ConnectedDeviceIdentityError, DepthAiBuildMetadataError, DepthAlignment, DepthConfig,
    DepthError, DepthFrame as OakDepthFrame, Device, DeviceConfig, ImageError,
    ImageFrame as OakImageFrame, ImuConfig, ImuError, Intrinsics as OakIntrinsics, MonoConfig,
    QueueConfig, StreamId as OakStreamId,
};
#[cfg(all(feature = "record", feature = "actuation"))]
use robot_command_client::AppliedCommandReceipt;
#[cfg(feature = "record")]
use std::num::{NonZeroU32, NonZeroU64};
#[cfg(any(feature = "record", test))]
use std::sync::Arc;
#[cfg(any(feature = "record", test))]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "record")]
use std::thread;
#[cfg(feature = "record")]
use std::{
    fs::{File, OpenOptions},
    io::Read,
};

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
enum OfflineFatalDenseError {
    CommandGeneration(command_mapper::DenseCommandGenerationError),
    Occupancy(OccupancyRuntimeError),
}

impl std::fmt::Display for OfflineFatalDenseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CommandGeneration(source) => {
                write!(f, "final dense command sequencing failed: {source}")
            }
            Self::Occupancy(source) => write!(f, "final occupancy update failed: {source}"),
        }
    }
}

impl std::error::Error for OfflineFatalDenseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::CommandGeneration(source) => Some(source),
            Self::Occupancy(source) => Some(source),
        }
    }
}

#[derive(Debug)]
struct OfflineFatalTrackerError {
    source: TrackerError,
    dense_update: Option<OfflineFatalDenseError>,
    publication: Option<VizLogError>,
    occupancy_finalization: Option<OccupancyRuntimeError>,
}

impl std::fmt::Display for OfflineFatalTrackerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "offline tracker failed: {}", self.source)?;
        if let Some(dense_update) = self.dense_update.as_ref() {
            write!(f, "; {dense_update}")?;
        }
        if let Some(publication) = self.publication.as_ref() {
            write!(
                f,
                "; publishing its final authoritative dense update also failed: {publication}"
            )?;
        }
        if let Some(finalization) = self.occupancy_finalization.as_ref() {
            write!(
                f,
                "; offline occupancy finalization also failed: {finalization}"
            )?;
        }
        Ok(())
    }
}

impl std::error::Error for OfflineFatalTrackerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

#[derive(Debug)]
enum OdometryVizProcessingError {
    Dataset(DatasetError),
    DenseCommandGeneration(command_mapper::DenseCommandGenerationError),
    DenseCommandMapping(command_mapper::DenseCommandMappingError),
    Occupancy(OccupancyRuntimeError),
    Tracker(Box<OfflineFatalTrackerError>),
    Packet(VizError),
    Log(VizLogError),
}

impl std::fmt::Display for OdometryVizProcessingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dataset(source) => write!(f, "offline depth replay failed: {source}"),
            Self::DenseCommandGeneration(source) => {
                write!(f, "offline dense command sequencing failed: {source}")
            }
            Self::DenseCommandMapping(source) => {
                write!(f, "offline dense command mapping failed: {source}")
            }
            Self::Occupancy(source) => write!(f, "offline occupancy mapping failed: {source}"),
            Self::Tracker(source) => std::fmt::Display::fmt(source, f),
            Self::Packet(source) => write!(f, "visualization packet creation failed: {source}"),
            Self::Log(source) => write!(f, "visualization logging failed: {source}"),
        }
    }
}

impl std::error::Error for OdometryVizProcessingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Dataset(source) => Some(source),
            Self::DenseCommandGeneration(source) => Some(source),
            Self::DenseCommandMapping(source) => Some(source),
            Self::Occupancy(source) => Some(source),
            Self::Tracker(source) => Some(&source.source),
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

impl From<command_mapper::DenseCommandGenerationError> for OdometryVizProcessingError {
    fn from(source: command_mapper::DenseCommandGenerationError) -> Self {
        Self::DenseCommandGeneration(source)
    }
}

impl From<command_mapper::DenseCommandMappingError> for OdometryVizProcessingError {
    fn from(source: command_mapper::DenseCommandMappingError) -> Self {
        Self::DenseCommandMapping(source)
    }
}

impl From<OccupancyRuntimeError> for OdometryVizProcessingError {
    fn from(source: OccupancyRuntimeError) -> Self {
        Self::Occupancy(source)
    }
}

impl From<TrackerError> for OdometryVizProcessingError {
    fn from(source: TrackerError) -> Self {
        Self::Tracker(Box::new(OfflineFatalTrackerError {
            source,
            dense_update: None,
            publication: None,
            occupancy_finalization: None,
        }))
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
    /// Exact DepthAI MXID. No first-device fallback is permitted.
    #[arg(long, env = "KIKO_OAK_DEVICE_ID", value_name = "EXACT_MXID")]
    oak_device_id: OakMxidArg,
    #[arg(long, default_value_t = 640)]
    width: u32,
    #[arg(long, default_value_t = 480)]
    height: u32,
    #[arg(long, default_value_t = 30)]
    fps: u32,
    #[arg(long, default_value_t = true)]
    rectified: bool,
    /// Enable raw accelerometer and gyroscope capture at this nominal rate.
    #[arg(long, env = "KIKO_IMU_RATE_HZ")]
    imu_rate_hz: Option<NonZeroU32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg(feature = "record")]
struct OakMxidArg(String);

#[cfg(feature = "record")]
impl OakMxidArg {
    fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg(feature = "record")]
struct OakMxidArgError;

#[cfg(feature = "record")]
impl std::fmt::Display for OakMxidArgError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("OAK MXID must be nonempty")
    }
}

#[cfg(feature = "record")]
impl std::error::Error for OakMxidArgError {}

#[cfg(feature = "record")]
impl std::str::FromStr for OakMxidArg {
    type Err = OakMxidArgError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.trim().is_empty() {
            return Err(OakMxidArgError);
        }
        Ok(Self(value.to_owned()))
    }
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
    /// Timeout used to prove final live Rerun delivery, in milliseconds.
    #[arg(
        long,
        env = "KIKO_RERUN_FINISH_TIMEOUT_MS",
        default_value_t = RerunFinishTimeout::default()
    )]
    rerun_finish_timeout_ms: RerunFinishTimeout,
    /// Strict V1 host shadow-navigation configuration JSON.
    #[arg(long, value_name = "CONFIG_JSON")]
    navigation_config: Option<PathBuf>,
    /// Map-frame navigation target in the exact form `X_M,Y_M`.
    #[arg(long, value_name = "X_M,Y_M")]
    navigation_goal: Option<NavigationGoalArg>,
    /// Dataset directory that atomically binds captured payloads to navigation ingress.
    #[arg(long, value_name = "DATASET_PATH")]
    navigation_record: Option<PathBuf>,
    /// Separate physical-authority manifest. Requires --navigation-arm-robot.
    #[cfg(feature = "actuation")]
    #[arg(long, value_name = "ACTUATION_JSON")]
    navigation_actuation_config: Option<PathBuf>,
    /// Exact robot ID named by the physical-authority manifest. No environment alias exists.
    #[cfg(feature = "actuation")]
    #[arg(long, value_name = "EXACT_ROBOT_ID")]
    navigation_arm_robot: Option<String>,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OccupancyProjectionContractError {
    CameraHeightNotConfigured,
    LevelOpticalWorldNotDeclared,
    LegacyOpticalFrameNotDeclared,
    UnsupportedOpticalFrame(DepthOpticalFrame),
    DepthCalibrationDimensionsMismatch {
        depth: FrameDimensions,
        tracking: FrameDimensions,
    },
}

impl std::fmt::Display for OccupancyProjectionContractError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CameraHeightNotConfigured => write!(
                f,
                "2D occupancy requires explicit KIKO_OCCUPANCY_CAMERA_HEIGHT_M because visual SLAM does not establish gravity or a floor"
            ),
            Self::LevelOpticalWorldNotDeclared => write!(
                f,
                "2D occupancy requires KIKO_OCCUPANCY_ASSUME_LEVEL_OPTICAL_WORLD=true; camera height alone does not establish gravity, floor orientation, pitch, or roll"
            ),
            Self::LegacyOpticalFrameNotDeclared => write!(
                f,
                "legacy depth metadata does not declare its optical frame; set KIKO_OCCUPANCY_ASSUME_RECTIFIED_LEFT=true only when that physical assumption is known to be correct"
            ),
            Self::UnsupportedOpticalFrame(frame) => write!(
                f,
                "2D occupancy currently requires rectified_left depth aligned to Kiko's tracking camera, got {frame:?} without a calibrated extrinsic"
            ),
            Self::DepthCalibrationDimensionsMismatch { depth, tracking } => write!(
                f,
                "depth projection dimensions {}x{} differ from tracking-camera calibration {}x{}; depth-specific scaled intrinsics are not recorded",
                depth.width(),
                depth.height(),
                tracking.width(),
                tracking.height()
            ),
        }
    }
}

impl std::error::Error for OccupancyProjectionContractError {}

fn require_level_optical_world(
    assumption_declared: bool,
    camera_height_m: Option<f64>,
) -> Result<f64, OccupancyProjectionContractError> {
    if !assumption_declared {
        return Err(OccupancyProjectionContractError::LevelOpticalWorldNotDeclared);
    }
    camera_height_m.ok_or(OccupancyProjectionContractError::CameraHeightNotConfigured)
}

fn occupancy_depth_camera(
    tracking_intrinsics: PinholeIntrinsics,
    tracking_dimensions: FrameDimensions,
    depth: DepthProjectionContract,
    assume_rectified_left: bool,
) -> Result<DepthCameraModel, OccupancyProjectionContractError> {
    match depth.optical_frame() {
        Some(DepthOpticalFrame::RectifiedLeft) => {}
        None if assume_rectified_left => {}
        None => return Err(OccupancyProjectionContractError::LegacyOpticalFrameNotDeclared),
        Some(frame) => {
            return Err(OccupancyProjectionContractError::UnsupportedOpticalFrame(
                frame,
            ));
        }
    }

    let depth_dimensions = depth.dimensions();
    if depth_dimensions != tracking_dimensions {
        return Err(
            OccupancyProjectionContractError::DepthCalibrationDimensionsMismatch {
                depth: depth_dimensions,
                tracking: tracking_dimensions,
            },
        );
    }
    Ok(DepthCameraModel::new(
        tracking_intrinsics,
        depth_dimensions,
        DepthToTrackingCamera::identity(),
    ))
}

/// Parse occupancy policy once at the process boundary.
///
/// This is deliberately geometric and deterministic; no learned occupancy
/// model or device-specific accelerator is involved.
fn build_occupancy_runtime_config(
    tracking_intrinsics: PinholeIntrinsics,
    tracking_dimensions: FrameDimensions,
    depth: DepthProjectionContract,
) -> Result<OccupancyRuntimeConfig, Box<dyn std::error::Error>> {
    let assume_rectified_left = env_bool("KIKO_OCCUPANCY_ASSUME_RECTIFIED_LEFT")?.unwrap_or(false);
    let camera = occupancy_depth_camera(
        tracking_intrinsics,
        tracking_dimensions,
        depth,
        assume_rectified_left,
    )?;
    let camera_height_m = require_level_optical_world(
        env_bool("KIKO_OCCUPANCY_ASSUME_LEVEL_OPTICAL_WORLD")?.unwrap_or(false),
        env_f64("KIKO_OCCUPANCY_CAMERA_HEIGHT_M")?,
    )?;
    let resolution_m = env_f64("KIKO_OCCUPANCY_RESOLUTION_M")?.unwrap_or(0.05);
    let lower_x_m = env_f64("KIKO_OCCUPANCY_LOWER_X_M")?.unwrap_or(-10.0);
    let lower_y_m = env_f64("KIKO_OCCUPANCY_LOWER_Y_M")?.unwrap_or(-5.0);
    let width = env_u32("KIKO_OCCUPANCY_WIDTH_CELLS")?.unwrap_or(400);
    let height = env_u32("KIKO_OCCUPANCY_HEIGHT_CELLS")?.unwrap_or(400);
    let maximum_cells = env_usize("KIKO_OCCUPANCY_MAX_CELLS")?.unwrap_or(4_000_000);
    let minimum_height_m = env_f64("KIKO_OCCUPANCY_MIN_HEIGHT_M")?.unwrap_or(0.05);
    let maximum_height_m = env_f64("KIKO_OCCUPANCY_MAX_HEIGHT_M")?.unwrap_or(1.8);
    let minimum_depth_m = env_f64("KIKO_OCCUPANCY_MIN_DEPTH_M")?.unwrap_or(0.2);
    let maximum_depth_m = env_f64("KIKO_OCCUPANCY_MAX_DEPTH_M")?.unwrap_or(10.0);
    let sampling_block = env_u32("KIKO_OCCUPANCY_SAMPLE_BLOCK_PX")?.unwrap_or(4);
    let maximum_keyframes = env_usize("KIKO_OCCUPANCY_MAX_KEYFRAMES")?.unwrap_or(300);
    let snapshot_cadence = OccupancySnapshotCadence::try_new(
        env_usize("KIKO_OCCUPANCY_RERUN_EVERY_KEYFRAMES")?.unwrap_or(5),
    )?;

    let geometry = OccupancyGridGeometry::try_new(
        resolution_m,
        [lower_x_m, lower_y_m],
        width,
        height,
        maximum_cells,
    )?;
    let world_to_occupancy = WorldToOccupancy::level_optical_world(camera_height_m)?;
    let height_range = HeightRangeMeters::try_new(minimum_height_m, maximum_height_m)?;
    let depth_range = DepthRangeMeters::try_new(minimum_depth_m, maximum_depth_m)?;
    let evidence = OccupancyEvidenceModel::try_new(-1, 3, -2, 2)?;
    let mapper = OccupancyConfig::try_new(
        geometry,
        world_to_occupancy,
        camera,
        height_range,
        depth_range,
        sampling_block,
        evidence,
        maximum_keyframes,
    )?;

    eprintln!(
        "occupancy requested: geometric=true learned=false level_optical_world_assumed=true world_axes=[x:right,y:down,z:forward] occupancy_axes=[x:world_x,y:world_z,height:camera_height-world_y] grid={}x{} resolution_m={} lower_xy_m=[{},{}] height_m=[{},{}] depth_m=[{},{}] sample_block_px={} max_keyframes={} rerun_every_keyframes={} camera_height_m={}",
        width,
        height,
        resolution_m,
        lower_x_m,
        lower_y_m,
        minimum_height_m,
        maximum_height_m,
        minimum_depth_m,
        maximum_depth_m,
        sampling_block,
        maximum_keyframes,
        snapshot_cadence.get(),
        camera_height_m,
    );

    Ok(OccupancyRuntimeConfig::new(mapper, snapshot_cadence))
}

/// Build the global dense map from the already-parsed navigation contract.
///
/// Global map extent, evidence retention, and publication cadence remain host
/// resource policy at their existing environment boundary. Coordinate frame,
/// runtime camera, height/depth eligibility, and pixel sampling come from the
/// one strict navigation JSON parse and are not independently reconstructed.
#[cfg(feature = "record")]
fn build_navigation_occupancy_runtime_config(
    navigation: &ShadowNavigationConfigV1,
) -> Result<OccupancyRuntimeConfig, Box<dyn std::error::Error>> {
    let resolution_m = env_f64("KIKO_OCCUPANCY_RESOLUTION_M")?.unwrap_or(0.05);
    let lower_x_m = env_f64("KIKO_OCCUPANCY_LOWER_X_M")?.unwrap_or(-10.0);
    let lower_y_m = env_f64("KIKO_OCCUPANCY_LOWER_Y_M")?.unwrap_or(-5.0);
    let width = env_u32("KIKO_OCCUPANCY_WIDTH_CELLS")?.unwrap_or(400);
    let height = env_u32("KIKO_OCCUPANCY_HEIGHT_CELLS")?.unwrap_or(400);
    let maximum_cells = env_usize("KIKO_OCCUPANCY_MAX_CELLS")?.unwrap_or(4_000_000);
    let maximum_keyframes = env_usize("KIKO_OCCUPANCY_MAX_KEYFRAMES")?.unwrap_or(300);
    let snapshot_cadence = OccupancySnapshotCadence::try_new(
        env_usize("KIKO_OCCUPANCY_RERUN_EVERY_KEYFRAMES")?.unwrap_or(5),
    )?;
    let geometry = OccupancyGridGeometry::try_new(
        resolution_m,
        [lower_x_m, lower_y_m],
        width,
        height,
        maximum_cells,
    )?;
    let local = navigation.local_costmap();
    let world_to_occupancy = navigation.world_to_occupancy();
    let evidence = OccupancyEvidenceModel::try_new(-1, 3, -2, 2)?;
    let mapper = OccupancyConfig::try_new(
        geometry,
        world_to_occupancy,
        local.camera(),
        local.obstacle_height_range(),
        local.depth_range(),
        local.sampling_block(),
        evidence,
        maximum_keyframes,
    )?;
    eprintln!(
        "navigation occupancy: geometric=true learned=false grid={}x{} resolution_m={} lower_xy_m=[{},{}] max_keyframes={} snapshot_every_keyframes={} world_to_occupancy_rotation={:?} world_to_occupancy_translation_m={:?}",
        width,
        height,
        resolution_m,
        lower_x_m,
        lower_y_m,
        maximum_keyframes,
        snapshot_cadence.get(),
        world_to_occupancy.rotation(),
        world_to_occupancy.translation_m(),
    );
    Ok(OccupancyRuntimeConfig::new(mapper, snapshot_cadence))
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
) -> Result<RectifiedStereoConfig, RectifiedStereoConfigError> {
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

    let rectified = RectifiedStereo::from_stereo_calibration_with_config(
        reader.stereo_calibration(),
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

        let cutoff_delta = i64::try_from(command_mapper::DEPTH_ASSOCIATION_WINDOW.as_nanos())
            .expect("the 20 ms depth-association policy fits in i64");
        let cutoff_ns = timestamp
            .as_nanos()
            .checked_add(cutoff_delta)
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

        let max_delta = command_mapper::DEPTH_ASSOCIATION_WINDOW.as_nanos();
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
    runtime: OccupancyRuntime,
    snapshots_enabled: bool,
    deferred_snapshot_error: Option<OccupancyError>,
    generation: command_mapper::DenseCommandGeneration,
    last_buffered_depth: Option<FrameId>,
}

enum OfflineDenseReplay {
    Disabled,
    Enabled(Box<OfflineDenseState>),
}

fn process_offline_occupancy_commands(
    runtime: &mut OccupancyRuntime,
    snapshots_enabled: &mut bool,
    deferred_snapshot_error: &mut Option<OccupancyError>,
    commands: impl IntoIterator<Item = dense::DenseCommand>,
) -> Result<(Option<DenseStats>, Option<TimedOccupancySnapshot>), OccupancyRuntimeError> {
    let mut latest_stats = None;
    let mut latest_snapshot = None;
    for command in commands {
        match runtime.process(command, *snapshots_enabled) {
            Ok(outcome) => {
                let (stats, snapshot) = outcome.into_parts();
                latest_stats = Some(stats);
                if snapshot.is_some() {
                    latest_snapshot = snapshot;
                }
            }
            Err(OccupancyRuntimeError::Snapshot(error)) => {
                eprintln!(
                    "offline occupancy snapshot publication failed; mapping will drain before the failure is returned: {error}"
                );
                deferred_snapshot_error.get_or_insert(error);
                *snapshots_enabled = false;
                latest_stats = Some(runtime.stats());
            }
            Err(error @ OccupancyRuntimeError::Mapping(_)) => {
                return Err(error.with_deferred_snapshot(deferred_snapshot_error));
            }
            Err(error @ OccupancyRuntimeError::MappingAndSnapshot { .. }) => return Err(error),
        }
    }
    Ok((latest_stats, latest_snapshot))
}

fn take_deferred_offline_snapshot_error(
    deferred_snapshot_error: &mut Option<OccupancyError>,
) -> Result<(), OccupancyRuntimeError> {
    deferred_snapshot_error
        .take()
        .map_or(Ok(()), |error| Err(OccupancyRuntimeError::Snapshot(error)))
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
        let depth_projection = reader
            .depth_projection_contract()
            .ok_or(DatasetError::DepthStreamNotConfigured)?;
        let stereo_calibration = reader.stereo_calibration();
        let occupancy_config = build_occupancy_runtime_config(
            stereo_calibration.left(),
            stereo_calibration.dimensions(),
            depth_projection,
        )?;
        eprintln!(
            "offline dense enabled: manifest_depth_frames={} ring_capacity={}",
            cursor.len(),
            depth_ring_capacity.get()
        );
        OfflineDenseReplay::Enabled(Box::new(OfflineDenseState {
            cursor,
            selector: OfflineDepthSelector::default(),
            ring: DepthRingBuffer::try_new(depth_ring_capacity.get())?,
            runtime: OccupancyRuntime::try_new(occupancy_config)?,
            snapshots_enabled: true,
            deferred_snapshot_error: None,
            generation: command_mapper::DenseCommandGeneration::default(),
            last_buffered_depth: None,
        }))
    } else {
        OfflineDenseReplay::Disabled
    };

    let inference = InferenceConfig::from_args(&args.inference)?;
    let decimation = args.rerun_decimation.get();

    let rectified = RectifiedStereo::from_stereo_calibration_with_config(
        reader.stereo_calibration(),
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
                        let (dense_stats, occupancy_snapshot) = match &mut offline_dense {
                            OfflineDenseReplay::Disabled => (None, None),
                            OfflineDenseReplay::Enabled(dense) => {
                                let OfflineDenseState {
                                    ring,
                                    runtime,
                                    snapshots_enabled,
                                    deferred_snapshot_error,
                                    generation,
                                    ..
                                } = dense.as_mut();
                                output.diagnostics_mut().depth_reorder_warnings =
                                    Some(ring.reorder_warnings());
                                let pose_updates = tracker.take_pending_dense_pose_updates();
                                let cmds = command_mapper::map_output_to_dense_commands(
                                    &output,
                                    pose_updates,
                                    |keyframe_id| tracker.keyframe_pose(keyframe_id),
                                    ring,
                                    timestamp,
                                    generation,
                                )?;
                                process_offline_occupancy_commands(
                                    runtime,
                                    snapshots_enabled,
                                    deferred_snapshot_error,
                                    cmds,
                                )?
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
                        if let Some(snapshot) = occupancy_snapshot {
                            let (snapshot_timestamp, snapshot) = snapshot.into_parts();
                            sink.log_occupancy(snapshot_timestamp, snapshot)?;
                        }
                        processed += 1;
                    }
                    Err(err) => {
                        inference_errors += 1;
                        let requires_pipeline_shutdown = err.requires_pipeline_shutdown();
                        let mut dense_update = None;
                        let mut dense_update_failure = None;
                        if let OfflineDenseReplay::Enabled(dense) = &mut offline_dense {
                            let pose_updates = tracker.take_pending_dense_pose_updates();
                            let generation = &mut dense.generation;
                            match command_mapper::apply_pose_updates_command(
                                pose_updates,
                                left.timestamp(),
                                generation,
                            ) {
                                Ok(Some(command)) => match process_offline_occupancy_commands(
                                    &mut dense.runtime,
                                    &mut dense.snapshots_enabled,
                                    &mut dense.deferred_snapshot_error,
                                    [command],
                                ) {
                                    Ok(update) => dense_update = Some(update),
                                    Err(source) if requires_pipeline_shutdown => {
                                        dense_update_failure =
                                            Some(OfflineFatalDenseError::Occupancy(source));
                                    }
                                    Err(source) => return Err(source.into()),
                                },
                                Ok(None) => {}
                                Err(source) if requires_pipeline_shutdown => {
                                    dense_update_failure =
                                        Some(OfflineFatalDenseError::CommandGeneration(source));
                                }
                                Err(source) => return Err(source.into()),
                            }
                        }

                        if requires_pipeline_shutdown {
                            // A tracker failure can follow a committed BA correction. The
                            // correction above is authoritative, so finish and publish its final
                            // occupancy state before ending the session. Preserve every bounded
                            // related failure alongside the typed tracker source.
                            let mut final_snapshot = None;
                            let mut occupancy_finalization = None;
                            if let OfflineDenseReplay::Enabled(dense) = &mut offline_dense {
                                match dense.runtime.finish(dense.snapshots_enabled) {
                                    Ok(snapshot) => {
                                        final_snapshot = snapshot;
                                        occupancy_finalization =
                                            take_deferred_offline_snapshot_error(
                                                &mut dense.deferred_snapshot_error,
                                            )
                                            .err();
                                    }
                                    Err(error) => {
                                        occupancy_finalization =
                                            Some(error.with_deferred_snapshot(
                                                &mut dense.deferred_snapshot_error,
                                            ));
                                    }
                                }
                            }

                            let (stats, command_snapshot) = dense_update.unwrap_or_default();
                            debug_assert!(
                                command_snapshot.is_none() || final_snapshot.is_none(),
                                "a forced command snapshot must clear occupancy dirtiness"
                            );
                            // A finish snapshot, if present, is the latest authoritative revision
                            // and supersedes an earlier command snapshot.
                            let snapshot = final_snapshot.or(command_snapshot);
                            let publication = (|| -> Result<(), VizLogError> {
                                if let Some(snapshot) = snapshot {
                                    let (snapshot_timestamp, snapshot) = snapshot.into_parts();
                                    sink.log_occupancy(snapshot_timestamp, snapshot)?;
                                }
                                if let Some(stats) = stats.as_ref() {
                                    sink.log_dense_stats(left.timestamp(), stats)?;
                                }
                                Ok(())
                            })()
                            .err();
                            return Err(OdometryVizProcessingError::Tracker(Box::new(
                                OfflineFatalTrackerError {
                                    source: err,
                                    dense_update: dense_update_failure,
                                    publication,
                                    occupancy_finalization,
                                },
                            )));
                        }
                        if let Some((stats, snapshot)) = dense_update {
                            if let Some(stats) = stats.as_ref() {
                                sink.log_dense_stats(left.timestamp(), stats)?;
                            }
                            if let Some(snapshot) = snapshot {
                                let (snapshot_timestamp, snapshot) = snapshot.into_parts();
                                sink.log_occupancy(snapshot_timestamp, snapshot)?;
                            }
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
            if let OfflineDenseReplay::Enabled(dense) = &mut offline_dense {
                match dense.runtime.finish(dense.snapshots_enabled) {
                    Ok(Some(snapshot)) => {
                        let (timestamp, snapshot) = snapshot.into_parts();
                        sink.log_occupancy(timestamp, snapshot)?;
                    }
                    Ok(None) => {}
                    Err(OccupancyRuntimeError::Snapshot(error)) => {
                        dense.deferred_snapshot_error.get_or_insert(error);
                    }
                    Err(error @ OccupancyRuntimeError::Mapping(_)) => {
                        return Err(error
                            .with_deferred_snapshot(&mut dense.deferred_snapshot_error)
                            .into());
                    }
                    Err(error @ OccupancyRuntimeError::MappingAndSnapshot { .. }) => {
                        return Err(error.into());
                    }
                }
                take_deferred_offline_snapshot_error(&mut dense.deferred_snapshot_error)?;
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
const STEREO_BOOTSTRAP_POLL_TIMEOUT_MS: u32 = 10;

#[cfg(feature = "record")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StereoSide {
    Left,
    Right,
}

#[cfg(feature = "record")]
impl StereoSide {
    fn expected_stream(self) -> OakStreamId {
        match self {
            Self::Left => OakStreamId::MonoLeft,
            Self::Right => OakStreamId::MonoRight,
        }
    }
}

#[cfg(feature = "record")]
impl std::fmt::Display for StereoSide {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Left => f.write_str("left"),
            Self::Right => f.write_str("right"),
        }
    }
}

#[cfg(feature = "record")]
#[derive(Debug)]
enum StereoBootstrapError {
    Interrupted,
    LeftImage {
        source: ImageError,
    },
    RightImage {
        source: ImageError,
    },
    Calibration {
        source: OakCalibrationError,
    },
    UnexpectedStream {
        side: StereoSide,
        expected: OakStreamId,
        actual: OakStreamId,
    },
    UnexpectedDimensions {
        side: StereoSide,
        expected_width: u32,
        expected_height: u32,
        actual_width: u32,
        actual_height: u32,
    },
    LeftFrame {
        source: FrameError,
    },
    RightFrame {
        source: FrameError,
    },
    PairingInput {
        side: StereoSide,
        source: PairingInputError,
    },
}

#[cfg(feature = "record")]
impl std::fmt::Display for StereoBootstrapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Interrupted => {
                f.write_str("stereo bootstrap was interrupted before both frames arrived")
            }
            Self::LeftImage { source } => {
                write!(f, "left camera bootstrap capture failed: {source}")
            }
            Self::RightImage { source } => {
                write!(f, "right camera bootstrap capture failed: {source}")
            }
            Self::Calibration { source } => {
                write!(f, "stereo bootstrap calibration failed: {source}")
            }
            Self::UnexpectedStream {
                side,
                expected,
                actual,
            } => write!(
                f,
                "{side} camera bootstrap returned stream {actual:?}, expected {expected:?}"
            ),
            Self::UnexpectedDimensions {
                side,
                expected_width,
                expected_height,
                actual_width,
                actual_height,
            } => write!(
                f,
                "{side} camera bootstrap returned {actual_width}x{actual_height}, expected configured {expected_width}x{expected_height}"
            ),
            Self::LeftFrame { source } => {
                write!(f, "left bootstrap frame conversion failed: {source}")
            }
            Self::RightFrame { source } => {
                write!(f, "right bootstrap frame conversion failed: {source}")
            }
            Self::PairingInput { side, source } => {
                write!(f, "{side} bootstrap pairing input failed: {source}")
            }
        }
    }
}

#[cfg(feature = "record")]
impl std::error::Error for StereoBootstrapError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::LeftImage { source } | Self::RightImage { source } => Some(source),
            Self::Calibration { source } => Some(source),
            Self::LeftFrame { source } | Self::RightFrame { source } => Some(source),
            Self::PairingInput { source, .. } => Some(source),
            Self::Interrupted
            | Self::UnexpectedStream { .. }
            | Self::UnexpectedDimensions { .. } => None,
        }
    }
}

#[cfg(feature = "record")]
#[derive(Debug)]
enum OakRuntimeProvenanceError {
    ConnectedIdentity {
        source: ConnectedDeviceIdentityError,
    },
    LinkedDepthAiBuildMetadata {
        source: DepthAiBuildMetadataError,
    },
}

#[cfg(feature = "record")]
impl std::fmt::Display for OakRuntimeProvenanceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConnectedIdentity { source } => {
                write!(f, "could not read actual connected OAK identity: {source}")
            }
            Self::LinkedDepthAiBuildMetadata { source } => {
                write!(f, "could not read linked DepthAI build metadata: {source}")
            }
        }
    }
}

#[cfg(feature = "record")]
impl std::error::Error for OakRuntimeProvenanceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ConnectedIdentity { source } => Some(source),
            Self::LinkedDepthAiBuildMetadata { source } => Some(source),
        }
    }
}

#[cfg(feature = "record")]
#[derive(Clone, Debug, PartialEq, Eq)]
struct OakRuntimeProvenance {
    connected_mxid: String,
    depthai_sdk_version: String,
    depthai_sdk_commit: String,
    embedded_device_artifact_version: String,
    embedded_bootloader_artifact_version: String,
}

#[cfg(feature = "record")]
impl OakRuntimeProvenance {
    fn dataset_device_label(&self) -> String {
        format!(
            "OAK-D mxid={} depthai_sdk={} depthai_commit={} embedded_device={} embedded_bootloader={} timestamp=device_exposure_midpoint",
            self.connected_mxid,
            self.depthai_sdk_version,
            self.depthai_sdk_commit,
            self.embedded_device_artifact_version,
            self.embedded_bootloader_artifact_version,
        )
    }
}

#[cfg(feature = "record")]
fn inspect_oak_runtime(
    device: &Device,
    context: &str,
) -> Result<OakRuntimeProvenance, OakRuntimeProvenanceError> {
    let connected = device
        .connected_identity()
        .map_err(|source| OakRuntimeProvenanceError::ConnectedIdentity { source })?;
    let build = oak_sys::depthai_build_metadata()
        .map_err(|source| OakRuntimeProvenanceError::LinkedDepthAiBuildMetadata { source })?;

    eprintln!(
        "{context} OAK connected identity: mxid={:?} xlink_name={:?} eeprom_name={:?} product_name={:?}",
        connected.mxid(),
        connected.discovery_transport_name(),
        connected.eeprom_device_name(),
        connected.product_name(),
    );
    eprintln!(
        "{context} DepthAI build provenance: sdk_version={:?} sdk_commit={:?} embedded_device_artifact={:?} embedded_bootloader_artifact={:?} camera_timestamp=device_exposure_midpoint",
        build.sdk_version(),
        build.sdk_commit(),
        build.embedded_device_artifact_version(),
        build.embedded_bootloader_artifact_version(),
    );

    Ok(OakRuntimeProvenance {
        connected_mxid: connected.mxid().to_owned(),
        depthai_sdk_version: build.sdk_version().to_owned(),
        depthai_sdk_commit: build.sdk_commit().to_owned(),
        embedded_device_artifact_version: build.embedded_device_artifact_version().to_owned(),
        embedded_bootloader_artifact_version: build
            .embedded_bootloader_artifact_version()
            .to_owned(),
    })
}

#[cfg(feature = "record")]
struct StereoBootstrap {
    calibration: Calibration,
    rectified_left_intrinsics: OakIntrinsics,
}

#[cfg(feature = "record")]
fn require_bootstrap_frame_contract(
    side: StereoSide,
    frame: &OakImageFrame,
    config: &MonoConfig,
) -> Result<(), StereoBootstrapError> {
    let expected = side.expected_stream();
    if frame.stream != expected {
        return Err(StereoBootstrapError::UnexpectedStream {
            side,
            expected,
            actual: frame.stream,
        });
    }
    if (frame.width, frame.height) != (config.width, config.height) {
        return Err(StereoBootstrapError::UnexpectedDimensions {
            side,
            expected_width: config.width,
            expected_height: config.height,
            actual_width: frame.width,
            actual_height: frame.height,
        });
    }
    Ok(())
}

/// Establish the runtime stereo contract from the first delivered projections.
///
/// Both boundary frames retain their native DepthAI capture identities and are
/// inserted into the caller's pairer, so calibration discovery does not
/// silently consume data or replace device provenance with host counters.
#[cfg(feature = "record")]
fn bootstrap_stereo(
    device: &mut Device,
    config: &MonoConfig,
    running: &AtomicBool,
    pairer: &mut StereoPairer,
) -> Result<StereoBootstrap, StereoBootstrapError> {
    let mut left = None;
    let mut right = None;

    while left.is_none() || right.is_none() {
        if !running.load(Ordering::Relaxed) {
            return Err(StereoBootstrapError::Interrupted);
        }
        let mut received_frame = false;
        if left.is_none() {
            match device.mono_left(STEREO_BOOTSTRAP_POLL_TIMEOUT_MS) {
                Ok(frame) => {
                    left = Some(frame);
                    received_frame = true;
                }
                Err(ImageError::Timeout { .. } | ImageError::QueueEmpty) => {}
                Err(source) => return Err(StereoBootstrapError::LeftImage { source }),
            }
        }
        if right.is_none() {
            match device.mono_right(STEREO_BOOTSTRAP_POLL_TIMEOUT_MS) {
                Ok(frame) => {
                    right = Some(frame);
                    received_frame = true;
                }
                Err(ImageError::Timeout { .. } | ImageError::QueueEmpty) => {}
                Err(source) => return Err(StereoBootstrapError::RightImage { source }),
            }
        }
        if !received_frame {
            thread::sleep(Duration::from_micros(500));
        }
    }

    if !running.load(Ordering::Relaxed) {
        return Err(StereoBootstrapError::Interrupted);
    }
    let left = left.expect("loop exits only after receiving a left frame");
    let right = right.expect("loop exits only after receiving a right frame");
    require_bootstrap_frame_contract(StereoSide::Left, &left, config)?;
    require_bootstrap_frame_contract(StereoSide::Right, &right, config)?;

    let left_intrinsics = left.intrinsics();
    let right_intrinsics = right.intrinsics();
    let baseline_m = device
        .stereo_baseline_m()
        .map_err(|source| StereoBootstrapError::Calibration { source })?;
    let calibration = build_calibration(
        left_intrinsics,
        right_intrinsics,
        baseline_m,
        config.rectified,
    );

    let left = oak_to_frame(left, SensorId::StereoLeft)
        .map_err(|source| StereoBootstrapError::LeftFrame { source })?;
    let right = oak_to_frame(right, SensorId::StereoRight)
        .map_err(|source| StereoBootstrapError::RightFrame { source })?;
    pairer
        .push_left(left)
        .map_err(|source| StereoBootstrapError::PairingInput {
            side: StereoSide::Left,
            source,
        })?;
    pairer
        .push_right(right)
        .map_err(|source| StereoBootstrapError::PairingInput {
            side: StereoSide::Right,
            source,
        })?;

    Ok(StereoBootstrap {
        calibration,
        rectified_left_intrinsics: left_intrinsics,
    })
}

#[cfg(feature = "record")]
#[derive(Debug)]
enum RectifiedLeftDepthError {
    DimensionMismatch {
        expected_width: u32,
        expected_height: u32,
        actual_width: u32,
        actual_height: u32,
    },
    ProjectionMismatch {
        expected: [[f32; 3]; 3],
        actual: [[f32; 3]; 3],
    },
    MissingConnectedAlignment,
    UnexpectedConnectedAlignment {
        actual: DepthAlignment,
    },
    Conversion {
        source: DepthImageError,
    },
}

#[cfg(feature = "record")]
impl std::fmt::Display for RectifiedLeftDepthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionMismatch {
                expected_width,
                expected_height,
                actual_width,
                actual_height,
            } => write!(
                f,
                "depth projection grid {actual_width}x{actual_height} does not match calibrated rectified-left grid {expected_width}x{expected_height}"
            ),
            Self::ProjectionMismatch { expected, actual } => write!(
                f,
                "depth projection intrinsics {actual:?} do not match calibrated rectified-left intrinsics {expected:?}"
            ),
            Self::MissingConnectedAlignment => {
                f.write_str("depth frame lacks a connected-device alignment stamp")
            }
            Self::UnexpectedConnectedAlignment { actual } => write!(
                f,
                "depth frame is aligned to {actual:?}, not the required RectifiedLeft optical frame"
            ),
            Self::Conversion { source } => write!(f, "invalid delivered depth frame: {source}"),
        }
    }
}

#[cfg(feature = "record")]
impl std::error::Error for RectifiedLeftDepthError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Conversion { source } => Some(source),
            Self::DimensionMismatch { .. }
            | Self::ProjectionMismatch { .. }
            | Self::MissingConnectedAlignment
            | Self::UnexpectedConnectedAlignment { .. } => None,
        }
    }
}

#[cfg(feature = "record")]
fn require_rectified_left_depth_projection(
    expected: OakIntrinsics,
    actual: OakIntrinsics,
) -> Result<(), RectifiedLeftDepthError> {
    if (actual.width(), actual.height()) != (expected.width(), expected.height()) {
        return Err(RectifiedLeftDepthError::DimensionMismatch {
            expected_width: expected.width(),
            expected_height: expected.height(),
            actual_width: actual.width(),
            actual_height: actual.height(),
        });
    }
    if actual.projection_matrix() != expected.projection_matrix() {
        return Err(RectifiedLeftDepthError::ProjectionMismatch {
            expected: expected.projection_matrix(),
            actual: actual.projection_matrix(),
        });
    }
    Ok(())
}

#[cfg(feature = "record")]
fn require_rectified_left_depth_alignment(
    actual: Option<DepthAlignment>,
) -> Result<(), RectifiedLeftDepthError> {
    match actual {
        Some(DepthAlignment::RectifiedLeft) => Ok(()),
        Some(actual) => Err(RectifiedLeftDepthError::UnexpectedConnectedAlignment { actual }),
        None => Err(RectifiedLeftDepthError::MissingConnectedAlignment),
    }
}

#[cfg(feature = "record")]
fn require_rectified_left_depth_contract(
    expected: OakIntrinsics,
    actual: OakIntrinsics,
    alignment: Option<DepthAlignment>,
) -> Result<(), RectifiedLeftDepthError> {
    require_rectified_left_depth_projection(expected, actual)?;
    require_rectified_left_depth_alignment(alignment)
}

#[cfg(feature = "record")]
fn parse_rectified_left_depth(
    frame: OakDepthFrame,
    expected: OakIntrinsics,
) -> Result<DepthImage, RectifiedLeftDepthError> {
    require_rectified_left_depth_contract(
        expected,
        frame.intrinsics(),
        frame.connected_alignment(),
    )?;
    oak_to_depth_image(frame).map_err(|source| RectifiedLeftDepthError::Conversion { source })
}

#[cfg(feature = "record")]
#[derive(Debug)]
struct DeviceCloseFailure {
    source: Box<dyn std::error::Error>,
}

#[cfg(feature = "record")]
impl DeviceCloseFailure {
    fn new(source: impl std::error::Error + 'static) -> Self {
        Self {
            source: Box::new(source),
        }
    }
}

#[cfg(feature = "record")]
impl std::fmt::Display for DeviceCloseFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.source.fmt(f)
    }
}

#[cfg(feature = "record")]
impl std::error::Error for DeviceCloseFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.source.as_ref())
    }
}

#[cfg(feature = "record")]
#[derive(Debug)]
struct OperationAndDeviceCloseError {
    operation: Box<dyn std::error::Error>,
    close: DeviceCloseFailure,
}

#[cfg(feature = "record")]
impl std::fmt::Display for OperationAndDeviceCloseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "operation failed ({}); OAK device close also failed: {}",
            self.operation, self.close
        )
    }
}

#[cfg(feature = "record")]
impl std::error::Error for OperationAndDeviceCloseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.operation.as_ref())
    }
}

#[cfg(feature = "record")]
#[derive(Clone, Copy, Debug)]
enum RecordItem {
    DepthFrame,
    ImuReport,
    StereoPair,
}

#[cfg(feature = "record")]
impl std::fmt::Display for RecordItem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DepthFrame => write!(f, "depth frame"),
            Self::ImuReport => write!(f, "IMU report"),
            Self::StereoPair => write!(f, "stereo pair"),
        }
    }
}

#[cfg(any(feature = "record", test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct HostMonotonicRangeError {
    elapsed_ns: u128,
}

#[cfg(any(feature = "record", test))]
impl std::fmt::Display for HostMonotonicRangeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "host monotonic elapsed time {} ns exceeds the u64 recording timebase",
            self.elapsed_ns
        )
    }
}

#[cfg(any(feature = "record", test))]
impl std::error::Error for HostMonotonicRangeError {}

#[cfg(feature = "record")]
fn host_monotonic_since(
    origin: Instant,
) -> Result<HostMonotonicTimestamp, HostMonotonicRangeError> {
    host_monotonic_from_elapsed_nanos(origin.elapsed().as_nanos())
}

#[cfg(any(feature = "record", test))]
fn host_monotonic_from_elapsed_nanos(
    elapsed_ns: u128,
) -> Result<HostMonotonicTimestamp, HostMonotonicRangeError> {
    let elapsed_ns =
        u64::try_from(elapsed_ns).map_err(|_| HostMonotonicRangeError { elapsed_ns })?;
    Ok(HostMonotonicTimestamp::from_nanos(elapsed_ns))
}

#[cfg(any(feature = "record", test))]
fn navigation_clock_read_error(source: HostMonotonicRangeError) -> HostMonotonicClockReadError {
    HostMonotonicClockReadError::ElapsedNanosecondsOutOfRange {
        elapsed_nanoseconds: source.elapsed_ns,
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
    LeftFrame {
        source: FrameError,
    },
    RightFrame {
        source: FrameError,
    },
    Depth {
        source: DepthError,
    },
    DepthFrame {
        source: RectifiedLeftDepthError,
    },
    Imu {
        source: ImuError,
    },
    ImuSample {
        source: InertialValueError,
    },
    HostTimestamp {
        source: HostMonotonicRangeError,
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
    DeviceClose {
        source: DeviceCloseFailure,
    },
    CaptureAndDeviceClose {
        capture: Box<RecordCaptureError>,
        close: DeviceCloseFailure,
    },
}

#[cfg(feature = "record")]
impl std::fmt::Display for RecordCaptureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LeftImage { source } => write!(f, "left camera capture failed: {source}"),
            Self::RightImage { source } => write!(f, "right camera capture failed: {source}"),
            Self::LeftFrame { source } => {
                write!(f, "left camera returned an invalid frame: {source}")
            }
            Self::RightFrame { source } => {
                write!(f, "right camera returned an invalid frame: {source}")
            }
            Self::Depth { source } => write!(f, "depth camera capture failed: {source}"),
            Self::DepthFrame { source } => {
                write!(f, "depth camera contract failed: {source}")
            }
            Self::Imu { source } => write!(f, "IMU capture failed: {source}"),
            Self::ImuSample { source } => write!(f, "IMU sample contract failed: {source}"),
            Self::HostTimestamp { source } => {
                write!(f, "IMU host-arrival timestamp failed: {source}")
            }
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
            Self::DeviceClose { source } => write!(f, "OAK device close failed: {source}"),
            Self::CaptureAndDeviceClose { capture, close } => {
                write!(f, "{capture}; OAK device close also failed: {close}")
            }
        }
    }
}

#[cfg(feature = "record")]
impl std::error::Error for RecordCaptureError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::LeftImage { source } | Self::RightImage { source } => Some(source),
            Self::LeftFrame { source } | Self::RightFrame { source } => Some(source),
            Self::Depth { source } => Some(source),
            Self::DepthFrame { source } => Some(source),
            Self::Imu { source } => Some(source),
            Self::ImuSample { source } => Some(source),
            Self::HostTimestamp { source } => Some(source),
            Self::PairingInput { source } => Some(source),
            Self::DatasetWrite { source, .. } => Some(source),
            Self::DeviceClose { source } => Some(source),
            Self::CaptureAndDeviceClose { capture, .. } => Some(capture.as_ref()),
            Self::DatasetDropped { .. } | Self::DatasetWriterFailed { .. } => None,
        }
    }
}

#[cfg(feature = "record")]
fn record_device_close_error(
    capture: Option<RecordCaptureError>,
    close: DeviceCloseFailure,
) -> RecordCaptureError {
    match capture {
        None => RecordCaptureError::DeviceClose { source: close },
        Some(capture) => RecordCaptureError::CaptureAndDeviceClose {
            capture: Box::new(capture),
            close,
        },
    }
}

#[cfg(feature = "record")]
fn finite_rate_per_second(count: u64, elapsed_seconds: f64) -> f64 {
    if !elapsed_seconds.is_finite() || elapsed_seconds <= 0.0 {
        return 0.0;
    }
    let rate = count as f64 / elapsed_seconds;
    if rate.is_finite() { rate } else { 0.0 }
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
impl RecordError {
    fn with_device_close(self, close: DeviceCloseFailure) -> Self {
        match self {
            Self::Capture { source } => Self::Capture {
                source: record_device_close_error(Some(source), close),
            },
            Self::Finalization { source } => Self::CaptureAndFinalization {
                capture: record_device_close_error(None, close),
                finalization: source,
            },
            Self::CaptureAndFinalization {
                capture,
                finalization,
            } => Self::CaptureAndFinalization {
                capture: record_device_close_error(Some(capture), close),
                finalization,
            },
        }
    }
}

#[cfg(feature = "record")]
fn compose_record_errors(
    capture: Option<RecordCaptureError>,
    finalization: Option<Box<DatasetError>>,
) -> Option<RecordError> {
    match (capture, finalization) {
        (None, None) => None,
        (Some(source), None) => Some(RecordError::Capture { source }),
        (None, Some(source)) => Some(RecordError::Finalization { source }),
        (Some(capture), Some(finalization)) => Some(RecordError::CaptureAndFinalization {
            capture,
            finalization,
        }),
    }
}

#[cfg(feature = "record")]
fn finish_record_device_session(
    operation: Result<(), Box<dyn std::error::Error>>,
    close: Result<(), OakCloseError>,
) -> Result<(), Box<dyn std::error::Error>> {
    match (operation, close.map_err(DeviceCloseFailure::new)) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(operation), Ok(())) => Err(operation),
        (Ok(()), Err(close)) => Err(RecordError::Capture {
            source: record_device_close_error(None, close),
        }
        .into()),
        (Err(operation), Err(close)) => match operation.downcast::<RecordError>() {
            Ok(record) => Err(Box::new((*record).with_device_close(close))),
            Err(operation) => Err(Box::new(OperationAndDeviceCloseError { operation, close })),
        },
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
        alignment: DepthAlignment::RectifiedLeft,
    });
    let imu_config = args.camera.imu_rate_hz.map(|rate_hz| ImuConfig {
        rate_hz: rate_hz.get(),
    });
    // Device reconnect is not implemented. One invocation therefore contains
    // exactly one dataset-local device-clock session.
    let imu_session = imu_config
        .map(|_| DeviceSessionId::try_new(1))
        .transpose()?;

    let config = DeviceConfig {
        rgb: None,
        mono: Some(mono_config),
        depth: depth_config,
        imu: imu_config,
        queue: QueueConfig {
            size: 8,
            blocking: false,
        },
    };

    eprintln!(
        "connecting to OAK MXID {:?}...",
        args.camera.oak_device_id.as_str()
    );
    let mut device = Device::connect(args.camera.oak_device_id.as_str(), config)?;
    let oak_provenance = inspect_oak_runtime(&device, "record")?;
    let operation = (|| -> Result<(), Box<dyn std::error::Error>> {
        let pairing_window = load_pairing_window()?;
        let pairer_max_pending = load_pairer_max_pending_per_side()?;
        let mut pairer = StereoPairer::new_with_max_pending(pairing_window, pairer_max_pending)?;
        let StereoBootstrap {
            calibration,
            rectified_left_intrinsics,
        } = bootstrap_stereo(&mut device, &mono_config, running.as_ref(), &mut pairer)?;

        let meta = build_meta(
            &mono_config,
            depth_config.as_ref(),
            imu_config.as_ref(),
            &oak_provenance,
        );
        eprintln!("creating dataset at {}", output_path.display());
        let (writer, writer_handle) = if let Some(session_id) = imu_session {
            let stream_metadata =
                ImuStreamMetadata::new(session_id, ImuExtrinsicProvenance::uncalibrated_unknown());
            DatasetWriter::create_paired_with_imu_config(
                output_path,
                &meta,
                &calibration,
                pairing_window,
                stream_metadata,
                DatasetWriterConfig::default(),
            )?
        } else {
            DatasetWriter::create_paired(output_path, &meta, &calibration, pairing_window)?
        };

        let start = Instant::now();
        let mut pair_count = 0u64;
        let mut left_count = 1u64;
        let mut right_count = 1u64;
        let mut depth_count = 0u64;
        let mut imu_count = 0u64;
        let mut capture_error = None;

        eprintln!("recording... press ctrl+c to stop");

        'capture: while running.load(Ordering::Relaxed) {
            let mut got_any = false;

            match device.mono_left(0) {
                Ok(frame) => match oak_to_frame(frame, SensorId::StereoLeft) {
                    Ok(frame) => {
                        if let Err(source) = pairer.push_left(frame) {
                            capture_error = Some(RecordCaptureError::PairingInput { source });
                            break 'capture;
                        }
                        left_count += 1;
                        got_any = true;
                    }
                    Err(source) => {
                        capture_error = Some(RecordCaptureError::LeftFrame { source });
                        break 'capture;
                    }
                },
                Err(ImageError::Timeout { .. } | ImageError::QueueEmpty) => {}
                Err(source) => {
                    capture_error = Some(RecordCaptureError::LeftImage { source });
                    break 'capture;
                }
            }

            match device.mono_right(0) {
                Ok(frame) => match oak_to_frame(frame, SensorId::StereoRight) {
                    Ok(frame) => {
                        if let Err(source) = pairer.push_right(frame) {
                            capture_error = Some(RecordCaptureError::PairingInput { source });
                            break 'capture;
                        }
                        right_count += 1;
                        got_any = true;
                    }
                    Err(source) => {
                        capture_error = Some(RecordCaptureError::RightFrame { source });
                        break 'capture;
                    }
                },
                Err(ImageError::Timeout { .. } | ImageError::QueueEmpty) => {}
                Err(source) => {
                    capture_error = Some(RecordCaptureError::RightImage { source });
                    break 'capture;
                }
            }

            if depth_enabled {
                match device.depth(0) {
                    Ok(depth_frame) => {
                        match parse_rectified_left_depth(depth_frame, rectified_left_intrinsics) {
                            Ok(depth) => {
                                if let Err(err) = require_record_write(
                                    writer.write_depth(&depth),
                                    RecordItem::DepthFrame,
                                ) {
                                    capture_error = Some(err);
                                    break 'capture;
                                }
                                depth_count = depth_count.saturating_add(1);
                                got_any = true;
                            }
                            Err(source) => {
                                capture_error = Some(RecordCaptureError::DepthFrame { source });
                                break 'capture;
                            }
                        }
                    }
                    Err(DepthError::Timeout { .. } | DepthError::QueueEmpty) => {}
                    Err(source) => {
                        capture_error = Some(RecordCaptureError::Depth { source });
                        break 'capture;
                    }
                }
            }

            if let Some(session_id) = imu_session {
                match device.imu() {
                    Ok(samples) => {
                        let host_arrival = match host_monotonic_since(start) {
                            Ok(timestamp) => timestamp,
                            Err(source) => {
                                capture_error = Some(RecordCaptureError::HostTimestamp { source });
                                break 'capture;
                            }
                        };
                        for sample in samples {
                            let report = match oak_to_imu_report(sample, session_id, host_arrival) {
                                Ok(report) => report,
                                Err(source) => {
                                    capture_error = Some(RecordCaptureError::ImuSample { source });
                                    break 'capture;
                                }
                            };
                            if let Err(error) = require_record_write(
                                writer.write_imu(report),
                                RecordItem::ImuReport,
                            ) {
                                capture_error = Some(error);
                                break 'capture;
                            }
                            imu_count = imu_count.saturating_add(1);
                        }
                        got_any = true;
                    }
                    Err(ImuError::Empty) => {}
                    Err(source) => {
                        capture_error = Some(RecordCaptureError::Imu { source });
                        break 'capture;
                    }
                }
            }

            while let Some(pair) = pairer.next_pair() {
                if let Err(err) =
                    require_record_write(writer.write_pair(pair), RecordItem::StereoPair)
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
            let timed_left_count = left_count.saturating_sub(1);
            let timed_right_count = right_count.saturating_sub(1);
            eprintln!(
                "finished timed capture in {:.1}s: pairs={}, left={} (1 bootstrap + {} timed, {:.1} timed fps), right={} (1 bootstrap + {} timed, {:.1} timed fps), depth={} ({:.1}fps), imu_reports={} ({:.1}Hz), logical_payload_units_written={}, logical_payload_units_dropped={}",
                elapsed,
                pair_count,
                left_count,
                timed_left_count,
                finite_rate_per_second(timed_left_count, elapsed),
                right_count,
                timed_right_count,
                finite_rate_per_second(timed_right_count, elapsed),
                depth_count,
                finite_rate_per_second(depth_count, elapsed),
                imu_count,
                finite_rate_per_second(imu_count, elapsed),
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
        match compose_record_errors(capture_error, finalization.err().map(Box::new)) {
            None => Ok(()),
            Some(error) => Err(error.into()),
        }
    })();
    finish_record_device_session(operation, device.close())
}

#[cfg(any(feature = "record", test))]
#[derive(Clone, Debug, PartialEq)]
enum LiveNavigationRequest {
    Disabled,
    Enabled {
        config_path: PathBuf,
        goal: Option<NavigationGoalArg>,
        dataset_path: PathBuf,
        actuation: LiveActuationRequest,
    },
}

#[cfg(any(feature = "record", test))]
#[derive(Clone, Debug, PartialEq, Eq)]
enum LiveActuationRequest {
    ShadowOnly,
    Physical {
        config_path: PathBuf,
        exact_robot_id: String,
    },
}

#[cfg(any(feature = "record", test))]
impl LiveNavigationRequest {
    fn parse(
        config_path: Option<PathBuf>,
        goal: Option<NavigationGoalArg>,
        dataset_path: Option<PathBuf>,
        actuation_config_path: Option<PathBuf>,
        exact_robot_id: Option<String>,
    ) -> Result<Self, LiveNavigationBoundaryError> {
        match (config_path, dataset_path) {
            (None, None) if goal.is_none() => {
                if actuation_config_path.is_some() || exact_robot_id.is_some() {
                    Err(LiveNavigationBoundaryError::ActuationRequiresNavigation)
                } else {
                    Ok(Self::Disabled)
                }
            }
            (Some(config_path), Some(dataset_path)) => {
                let actuation = match (actuation_config_path, exact_robot_id) {
                    (None, None) => LiveActuationRequest::ShadowOnly,
                    (Some(config_path), Some(exact_robot_id)) => LiveActuationRequest::Physical {
                        config_path,
                        exact_robot_id,
                    },
                    (config_path, exact_robot_id) => {
                        return Err(LiveNavigationBoundaryError::IncompleteActuation {
                            missing_config: config_path.is_none(),
                            missing_robot_id: exact_robot_id.is_none(),
                        });
                    }
                };
                Ok(Self::Enabled {
                    config_path,
                    goal,
                    dataset_path,
                    actuation,
                })
            }
            (config_path, dataset_path) => Err(LiveNavigationBoundaryError::IncompleteNavigation {
                missing_config: config_path.is_none(),
                missing_dataset: dataset_path.is_none(),
            }),
        }
    }

    fn is_enabled(&self) -> bool {
        matches!(self, Self::Enabled { .. })
    }
}

#[cfg(any(feature = "record", test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LiveNavigationBoundaryError {
    IncompleteNavigation {
        missing_config: bool,
        missing_dataset: bool,
    },
    IncompleteActuation {
        missing_config: bool,
        missing_robot_id: bool,
    },
    ActuationRequiresNavigation,
}

#[cfg(any(feature = "record", test))]
impl std::fmt::Display for LiveNavigationBoundaryError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::IncompleteNavigation {
                missing_config,
                missing_dataset,
            } => {
                formatter.write_str(
                    "live navigation requires --navigation-config and --navigation-record together; --navigation-goal is optional; missing",
                )?;
                if missing_config {
                    formatter.write_str(" --navigation-config")?;
                }
                if missing_dataset {
                    formatter.write_str(" --navigation-record")?;
                }
                Ok(())
            }
            Self::IncompleteActuation {
                missing_config,
                missing_robot_id,
            } => {
                formatter.write_str(
                    "physical navigation actuation requires --navigation-actuation-config and --navigation-arm-robot together; missing",
                )?;
                if missing_config {
                    formatter.write_str(" --navigation-actuation-config")?;
                }
                if missing_robot_id {
                    formatter.write_str(" --navigation-arm-robot")?;
                }
                Ok(())
            }
            Self::ActuationRequiresNavigation => formatter
                .write_str("physical actuation options require a complete live navigation request"),
        }
    }
}

#[cfg(any(feature = "record", test))]
impl std::error::Error for LiveNavigationBoundaryError {}

#[cfg(any(feature = "record", test))]
fn navigation_dataset_may_publish(
    authoritative_failure: bool,
    journal_descriptor_finalized: bool,
    _visualization_failure: bool,
) -> bool {
    !authoritative_failure && journal_descriptor_finalized
}

#[cfg(feature = "record")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LiveNavigationPrerequisiteError {
    DepthDisabled,
    ImuDisabled,
    DenseOccupancyDisabled,
    UnrectifiedStereo,
}

#[cfg(feature = "record")]
impl std::fmt::Display for LiveNavigationPrerequisiteError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::DepthDisabled => {
                "live navigation requires KIKO_LIVE_DEPTH=true and rectified-left metric depth"
            }
            Self::ImuDisabled => "live navigation requires --imu-rate-hz (or KIKO_IMU_RATE_HZ)",
            Self::DenseOccupancyDisabled => {
                "live navigation requires KIKO_DENSE=true for global occupancy snapshots"
            }
            Self::UnrectifiedStereo => {
                "live navigation requires --rectified=true for the rectified-left depth contract"
            }
        })
    }
}

#[cfg(feature = "record")]
impl std::error::Error for LiveNavigationPrerequisiteError {}

#[cfg(feature = "record")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LiveNavigationPreparationError {
    MissingNavigationPolicy,
    NavigationPolicyWhileDisabled,
    #[cfg(feature = "actuation")]
    MissingPhysicalAuthority,
    PhysicalAuthorityInShadowMode,
    #[cfg(not(feature = "actuation"))]
    PhysicalActuationFeatureDisabled,
}

#[cfg(feature = "record")]
impl std::fmt::Display for LiveNavigationPreparationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::MissingNavigationPolicy => {
                "enabled live navigation has no bounded navigation-config input"
            }
            Self::NavigationPolicyWhileDisabled => {
                "disabled live navigation received unexpected navigation-config input"
            }
            #[cfg(feature = "actuation")]
            Self::MissingPhysicalAuthority => {
                "physical live navigation has no bounded actuation-config input"
            }
            Self::PhysicalAuthorityInShadowMode => {
                "shadow-only live navigation received unexpected actuation-config input"
            }
            #[cfg(not(feature = "actuation"))]
            Self::PhysicalActuationFeatureDisabled => {
                "physical live navigation was requested without the actuation build feature"
            }
        })
    }
}

#[cfg(feature = "record")]
impl std::error::Error for LiveNavigationPreparationError {}

#[cfg(feature = "record")]
#[derive(Debug)]
enum LiveNavigationConfigReadError {
    Open {
        kind: &'static str,
        path: PathBuf,
        source: std::io::Error,
    },
    Read {
        kind: &'static str,
        path: PathBuf,
        source: std::io::Error,
    },
    InputTooLarge {
        kind: &'static str,
        path: PathBuf,
        actual_bytes_at_least: usize,
        maximum_bytes: usize,
    },
}

#[cfg(feature = "record")]
impl std::fmt::Display for LiveNavigationConfigReadError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Open { kind, path, source } => write!(
                formatter,
                "cannot open {kind} config {}: {source}",
                path.display()
            ),
            Self::Read { kind, path, source } => write!(
                formatter,
                "cannot read {kind} config {}: {source}",
                path.display()
            ),
            Self::InputTooLarge {
                kind,
                path,
                actual_bytes_at_least,
                maximum_bytes,
            } => write!(
                formatter,
                "{kind} config {} is at least {actual_bytes_at_least} bytes; maximum is {maximum_bytes} bytes",
                path.display()
            ),
        }
    }
}

#[cfg(feature = "record")]
impl std::error::Error for LiveNavigationConfigReadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Open { source, .. } | Self::Read { source, .. } => Some(source),
            Self::InputTooLarge { .. } => None,
        }
    }
}

#[cfg(feature = "record")]
fn read_navigation_config_bounded(path: &Path) -> Result<Vec<u8>, LiveNavigationConfigReadError> {
    read_live_config_bounded(path, "navigation", MAX_SHADOW_NAVIGATION_CONFIG_JSON_BYTES)
}

#[cfg(all(feature = "record", feature = "actuation"))]
fn read_actuation_config_bounded(path: &Path) -> Result<Vec<u8>, LiveNavigationConfigReadError> {
    read_live_config_bounded(
        path,
        "navigation actuation",
        MAX_NAVIGATION_ACTUATION_CONFIG_JSON_BYTES,
    )
}

#[cfg(feature = "record")]
fn read_live_config_bounded(
    path: &Path,
    kind: &'static str,
    maximum_bytes: usize,
) -> Result<Vec<u8>, LiveNavigationConfigReadError> {
    let file = File::open(path).map_err(|source| LiveNavigationConfigReadError::Open {
        kind,
        path: path.to_path_buf(),
        source,
    })?;
    let read_bound = u64::try_from(maximum_bytes).expect("live config byte bound fits u64") + 1;
    let mut bytes = Vec::with_capacity(maximum_bytes.min(16 * 1024));
    file.take(read_bound)
        .read_to_end(&mut bytes)
        .map_err(|source| LiveNavigationConfigReadError::Read {
            kind,
            path: path.to_path_buf(),
            source,
        })?;
    if bytes.len() > maximum_bytes {
        return Err(LiveNavigationConfigReadError::InputTooLarge {
            kind,
            path: path.to_path_buf(),
            actual_bytes_at_least: bytes.len(),
            maximum_bytes,
        });
    }
    Ok(bytes)
}

#[cfg(feature = "record")]
#[derive(Debug)]
enum NavigationEntropyError {
    Open(std::io::Error),
    Read(std::io::Error),
    RepeatedAllZero,
    Invalid(NavigationRecordingIdError),
}

#[cfg(feature = "record")]
impl std::fmt::Display for NavigationEntropyError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Open(source) => write!(formatter, "cannot open OS entropy source: {source}"),
            Self::Read(source) => write!(formatter, "cannot read OS entropy source: {source}"),
            Self::RepeatedAllZero => {
                formatter.write_str("OS entropy repeatedly returned an all-zero recording ID")
            }
            Self::Invalid(source) => source.fmt(formatter),
        }
    }
}

#[cfg(feature = "record")]
impl std::error::Error for NavigationEntropyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Open(source) | Self::Read(source) => Some(source),
            Self::Invalid(source) => Some(source),
            Self::RepeatedAllZero => None,
        }
    }
}

/// `/dev/urandom` is a kernel entropy interface on both supported host families.
#[cfg(feature = "record")]
fn generate_navigation_recording_id() -> Result<NavigationRecordingId, NavigationEntropyError> {
    let mut source = File::open("/dev/urandom").map_err(NavigationEntropyError::Open)?;
    for _ in 0..4 {
        let mut bytes = [0_u8; 16];
        source
            .read_exact(&mut bytes)
            .map_err(NavigationEntropyError::Read)?;
        if bytes != [0; 16] {
            return NavigationRecordingId::try_new(bytes).map_err(NavigationEntropyError::Invalid);
        }
    }
    Err(NavigationEntropyError::RepeatedAllZero)
}

#[cfg(feature = "record")]
#[derive(Clone, Copy, Debug)]
struct InstantHostClock {
    origin: Instant,
}

#[cfg(feature = "record")]
impl InstantHostClock {
    fn new(origin: Instant) -> Self {
        Self { origin }
    }

    fn checked_now(&self) -> Result<HostMonotonicTimestamp, HostMonotonicRangeError> {
        host_monotonic_since(self.origin)
    }
}

#[cfg(feature = "record")]
impl HostMonotonicClock for InstantHostClock {
    fn try_now(&mut self) -> Result<HostMonotonicTimestamp, HostMonotonicClockReadError> {
        self.checked_now().map_err(navigation_clock_read_error)
    }
}

#[cfg(any(feature = "record", test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LiveVisualShape {
    IncrementAndLocalization,
    LocalizationOnly,
    NoLocalization,
}

#[cfg(any(feature = "record", test))]
fn classify_live_visual_shape(has_increment: bool, has_localization: bool) -> LiveVisualShape {
    match (has_increment, has_localization) {
        (true, true) => LiveVisualShape::IncrementAndLocalization,
        (false, true) => LiveVisualShape::LocalizationOnly,
        // A correction-safe increment without a current map localization cannot
        // update map->odom. Journal the exact attempt as NoLocalization and
        // deliberately discard that unusable increment.
        (false, false) | (true, false) => LiveVisualShape::NoLocalization,
    }
}

#[cfg(feature = "record")]
#[derive(Debug)]
enum LiveVisualAdmissionBuildError {
    Admission(VisualAdmissionError),
}

#[cfg(feature = "record")]
impl std::fmt::Display for LiveVisualAdmissionBuildError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Admission(source) => source.fmt(formatter),
        }
    }
}

#[cfg(feature = "record")]
impl std::error::Error for LiveVisualAdmissionBuildError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Admission(source) => Some(source),
        }
    }
}

#[cfg(feature = "record")]
fn visual_admission_from_output(
    pending: PendingVisualAttemptIngress,
    output: &TrackerOutput,
) -> Result<VisualAdmission, LiveVisualAdmissionBuildError> {
    let increment = output.visual_increment();
    let localization = output.current_map_localization();
    match classify_live_visual_shape(increment.is_some(), localization.is_some()) {
        LiveVisualShape::IncrementAndLocalization => VisualAdmission::increment_and_localization(
            pending.complete(VisualAttemptOutcome::IncrementAndLocalization),
            increment.expect("shape proves a visual increment"),
            localization.expect("shape proves a map localization"),
        ),
        LiveVisualShape::LocalizationOnly => VisualAdmission::localization_only(
            pending.complete(VisualAttemptOutcome::LocalizationOnly),
            localization.expect("shape proves a map localization"),
        ),
        LiveVisualShape::NoLocalization => {
            VisualAdmission::no_localization(pending.complete(VisualAttemptOutcome::NoLocalization))
        }
    }
    .map_err(LiveVisualAdmissionBuildError::Admission)
}

#[cfg(any(feature = "record", test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LiveLosslessRouteError {
    TimedOut,
    Disconnected,
}

#[cfg(any(feature = "record", test))]
impl std::fmt::Display for LiveLosslessRouteError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::TimedOut => "lossless live navigation route timed out",
            Self::Disconnected => "lossless live navigation route disconnected",
        })
    }
}

#[cfg(any(feature = "record", test))]
impl std::error::Error for LiveLosslessRouteError {}

#[cfg(any(feature = "record", test))]
fn classify_lossless_send<T>(
    outcome: Result<(), crossbeam_channel::SendTimeoutError<T>>,
) -> Result<(), LiveLosslessRouteError> {
    match outcome {
        Ok(()) => Ok(()),
        Err(crossbeam_channel::SendTimeoutError::Timeout(_)) => {
            Err(LiveLosslessRouteError::TimedOut)
        }
        Err(crossbeam_channel::SendTimeoutError::Disconnected(_)) => {
            Err(LiveLosslessRouteError::Disconnected)
        }
    }
}

#[cfg(feature = "record")]
const LIVE_NAVIGATION_VISUAL_QUEUE_CAPACITY: usize = 64;
#[cfg(feature = "record")]
const LIVE_NAVIGATION_VISUAL_SEND_TIMEOUT: Duration = Duration::from_millis(50);

#[cfg(feature = "record")]
fn route_visual_admission(
    sender: &crossbeam_channel::Sender<VisualAdmission>,
    admission: VisualAdmission,
) -> Result<(), LiveLosslessRouteError> {
    classify_lossless_send(sender.send_timeout(admission, LIVE_NAVIGATION_VISUAL_SEND_TIMEOUT))
}

#[cfg(feature = "record")]
struct LiveVizMsg {
    left: Frame,
    right: Frame,
    depth: Option<DepthImage>,
    packet: Option<VizPacket<Raw>>,
    points: Option<Vec<CameraPoint3>>,
    output: TrackerOutput,
    dense_stats: Option<DenseStats>,
}

#[cfg(feature = "record")]
#[derive(Debug)]
struct LiveLocalCostmapViz {
    width: u32,
    height: u32,
    lower_bound_m: [f32; 2],
    resolution_m: f32,
    class_ids: Vec<u8>,
    local_costmap_to_odom: Option<[f64; 3]>,
    evidence: String,
}

#[cfg(feature = "record")]
#[derive(Debug)]
struct LiveAppliedActuationViz {
    controller_boot_id: u64,
    control_epoch: u32,
    sequence: u32,
    applied_pwm: [i8; 2],
    remaining_lease_at_server_emission_ms: u16,
    acknowledged_at_ns_decimal: String,
    known_active_through_ns_decimal: String,
}

#[cfg(all(feature = "record", feature = "actuation"))]
fn live_applied_actuation_viz(receipt: &AppliedCommandReceipt) -> LiveAppliedActuationViz {
    let pwm = receipt.applied_timer_pwm();
    LiveAppliedActuationViz {
        controller_boot_id: receipt.controller_session().boot_id().get(),
        control_epoch: receipt.controller_session().control_epoch().get(),
        sequence: receipt.sequence().get(),
        applied_pwm: [pwm.left().get(), pwm.right().get()],
        remaining_lease_at_server_emission_ms: receipt.remaining_lease_at_server_emission().get(),
        acknowledged_at_ns_decimal: receipt
            .acknowledged_at()
            .nanos_since_clock_start()
            .to_string(),
        known_active_through_ns_decimal: receipt
            .known_active_through_exclusive()
            .nanos_since_clock_start()
            .to_string(),
    }
}

#[cfg(any(feature = "record", test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LiveDecisionVizKind {
    Controller,
    Stopped,
}

#[cfg(any(feature = "record", test))]
const fn live_decision_viz_status(kind: LiveDecisionVizKind, applied: bool) -> &'static str {
    match (kind, applied) {
        (LiveDecisionVizKind::Controller, false) => "controller_request",
        (LiveDecisionVizKind::Controller, true) => "controller_applied",
        (LiveDecisionVizKind::Stopped, false) => "fail_closed_stop",
        (LiveDecisionVizKind::Stopped, true) => "fail_closed_stop_applied",
    }
}

#[cfg(feature = "record")]
#[derive(Debug)]
struct LiveNavigationVizMsg {
    tick_sequence: i64,
    host_timestamp_ns: u64,
    goal: Option<[f32; 2]>,
    goal_state: String,
    odometry_state: Option<String>,
    path: Option<Vec<[f32; 2]>>,
    local_costmap: Option<LiveLocalCostmapViz>,
    base_to_odom: Option<[f64; 3]>,
    odom_to_map: Option<[f64; 3]>,
    predicted_odom: Option<Vec<[f32; 2]>>,
    decision_id: u64,
    request_id: Option<u64>,
    status: &'static str,
    reason: String,
    requested_pwm: [i8; 2],
    objective_cost: Option<f64>,
    shadow_record_motor_packets_sent: u64,
    applied_actuation: Option<LiveAppliedActuationViz>,
    diagnostic_warning: Option<String>,
}

#[cfg(feature = "record")]
fn rerun_point2(value: [f64; 2]) -> Option<[f32; 2]> {
    let x = value[0] as f32;
    let y = value[1] as f32;
    (x.is_finite() && y.is_finite()).then_some([x, y])
}

#[cfg(feature = "record")]
fn build_live_navigation_viz_message(
    coordinator: &ShadowNavigationCoordinator<NavigationIngressWriter<File>>,
    tick: HostMonotonicTimestamp,
    tick_sequence: i64,
    outcome: &CoordinatorTickOutcome<NavigationIngressStreamWriteError>,
    applied_actuation: Option<LiveAppliedActuationViz>,
) -> LiveNavigationVizMsg {
    let mut warnings = Vec::new();
    let goal = coordinator.current_goal().and_then(|goal| {
        rerun_point2(goal.point().as_array()).or_else(|| {
            warnings.push("goal is outside Rerun's finite f32 coordinate domain".to_owned());
            None
        })
    });
    let path = coordinator.global_path().and_then(|path| {
        let points = path
            .points()
            .iter()
            .map(|point| rerun_point2(point.as_array()))
            .collect::<Option<Vec<_>>>();
        if points.is_none() {
            warnings.push("global path is outside Rerun's finite f32 coordinate domain".to_owned());
        }
        points
    });

    let local_costmap = match coordinator.local_costmap().view_at(tick) {
        Ok(view) => {
            let lower_bound_m = rerun_point2(view.lower_bound_m());
            let resolution_m = view.resolution_m() as f32;
            let provenance = view.provenance();
            let local_costmap_to_odom = provenance.map(|provenance| {
                let transform = provenance.local_costmap_to_odom();
                [
                    transform.source_origin_x_in_destination_m(),
                    transform.source_origin_y_in_destination_m(),
                    transform.source_yaw_in_destination_rad(),
                ]
            });
            if let Some(lower_bound_m) = lower_bound_m
                && resolution_m.is_finite()
                && resolution_m > 0.0
            {
                Some(LiveLocalCostmapViz {
                    width: view.width(),
                    height: view.height(),
                    lower_bound_m,
                    resolution_m,
                    class_ids: view.class_ids().to_vec(),
                    local_costmap_to_odom,
                    evidence: format!("freshness={:?} provenance={provenance:?}", view.freshness()),
                })
            } else {
                warnings
                    .push("local costmap geometry is outside Rerun's finite f32 domain".to_owned());
                None
            }
        }
        Err(source) => {
            warnings.push(format!("local costmap diagnostic unavailable: {source}"));
            None
        }
    };

    let current = coordinator.odometry().current();
    let base_to_odom = current.map(|state| {
        let transform = state.base_to_odom();
        [
            transform.source_origin_x_in_destination_m(),
            transform.source_origin_y_in_destination_m(),
            transform.source_yaw_in_destination_rad(),
        ]
    });
    let odom_to_map = current.map(|state| {
        let transform = state.odom_to_map();
        [
            transform.source_origin_x_in_destination_m(),
            transform.source_origin_y_in_destination_m(),
            transform.source_yaw_in_destination_rad(),
        ]
    });
    let odometry_state = current.map(|state| {
        format!(
            "session_id={} segment_id={} device_timestamp_ns={} map_snapshot={:?} quality={:?}",
            state.session_id().as_u64(),
            state.segment_id().as_u64(),
            state.timestamp().as_nanos(),
            state.map_snapshot(),
            state.quality(),
        )
    });

    let decision = outcome.decision();
    let applied = applied_actuation.is_some();
    let (kind, reason, objective_cost) = match decision.outcome() {
        SafetyDecisionOutcome::Controller(controller) => (
            LiveDecisionVizKind::Controller,
            if applied {
                "exact V2 controller application receipt matched; physical wheel motion remains unobserved".to_owned()
            } else {
                "safety-approved shadow MPC request; no transport exists".to_owned()
            },
            Some(controller.objective_cost()),
        ),
        SafetyDecisionOutcome::Stopped(stopped) => (
            LiveDecisionVizKind::Stopped,
            if applied {
                format!(
                    "{}; exact V2 zero application receipt matched; physical wheel motion remains unobserved",
                    stopped.cause()
                )
            } else {
                stopped.cause().to_string()
            },
            None,
        ),
    };
    let status = live_decision_viz_status(kind, applied);
    let predicted_odom = coordinator
        .safety()
        .last_success_trajectory()
        .and_then(|trajectory| {
            let points = trajectory
                .points()
                .iter()
                .map(|point| rerun_point2(point.pose().position().as_array()))
                .collect::<Option<Vec<_>>>();
            if points.is_none() {
                warnings
                    .push("predicted trajectory is outside Rerun's finite f32 domain".to_owned());
            }
            points
        });
    let record = decision.record();
    let pwm = record.pwm();
    LiveNavigationVizMsg {
        tick_sequence,
        host_timestamp_ns: tick.as_nanos(),
        goal,
        goal_state: format!("{:?}", coordinator.goal_state()),
        odometry_state,
        path,
        local_costmap,
        base_to_odom,
        odom_to_map,
        predicted_odom,
        decision_id: record.decision_id().as_u64(),
        request_id: decision.request_id().map(NonZeroU64::get),
        status,
        reason,
        requested_pwm: [pwm.left().get(), pwm.right().get()],
        objective_cost,
        shadow_record_motor_packets_sent: decision.motor_packets_sent().get(),
        applied_actuation,
        diagnostic_warning: (!warnings.is_empty()).then(|| warnings.join("; ")),
    }
}

#[cfg(feature = "record")]
fn log_live_navigation_viz_message(
    recording: &rerun::RecordingStream,
    message: LiveNavigationVizMsg,
    context_logged: &mut bool,
) -> Result<(), VizLogError> {
    let local_costmap_to_odom = message
        .local_costmap
        .as_ref()
        .and_then(|local| local.local_costmap_to_odom);
    let local_costmap_evidence = message
        .local_costmap
        .as_ref()
        .map(|local| local.evidence.clone());
    recording.set_time_sequence("navigation_tick", message.tick_sequence);
    if let Ok(host_timestamp_ns) = i64::try_from(message.host_timestamp_ns) {
        recording.set_time(
            "navigation_host_ns",
            rerun::TimeCell::from_duration_nanos(host_timestamp_ns),
        );
    }
    if !*context_logged {
        recording.log_static(
            "navigation/local_costmap_at_capture",
            &rerun::AnnotationContext::new([
                (0_u16, "unknown", rerun::Rgba32::from_rgb(96, 96, 96)),
                (1_u16, "free", rerun::Rgba32::from_rgb(238, 238, 238)),
                (2_u16, "inflated", rerun::Rgba32::from_rgb(255, 183, 3)),
                (3_u16, "occupied", rerun::Rgba32::from_rgb(230, 57, 70)),
            ]),
        )?;
        recording.log_static(
            "navigation/local_costmap_at_capture/frame_contract",
            &rerun::TextLog::new(
                "unlinked planar LocalCostmapFrame, base-aligned at the admitted depth capture: +x forward, +y left; it is not the moving current BaseFrame and is not silently placed in map coordinates",
            ),
        )?;
        *context_logged = true;
    }

    if let Some(goal) = message.goal {
        recording.log(
            "world/map2d/navigation/goal",
            &rerun::Points2D::new([goal])
                .with_colors([rerun::Color::from_rgb(131, 56, 236)])
                .with_radii([0.12]),
        )?;
    } else {
        recording.log("world/map2d/navigation/goal", &rerun::Clear::flat())?;
    }
    if let Some(path) = message.path {
        recording.log(
            "world/map2d/navigation/path",
            &rerun::LineStrips2D::new([path])
                .with_colors([rerun::Color::from_rgb(69, 123, 157)])
                .with_radii([0.035]),
        )?;
    } else {
        recording.log("world/map2d/navigation/path", &rerun::Clear::flat())?;
    }
    if let Some(predicted) = message.predicted_odom {
        recording.log(
            "navigation/odom_frame/predicted_trajectory",
            &rerun::LineStrips2D::new([predicted])
                .with_colors([rerun::Color::from_rgb(251, 86, 7)])
                .with_radii([0.025]),
        )?;
    } else {
        recording.log(
            "navigation/odom_frame/predicted_trajectory",
            &rerun::Clear::flat(),
        )?;
    }
    if let Some(local) = message.local_costmap {
        recording.log(
            "navigation/local_costmap_at_capture/grid",
            &rerun::Transform3D::from_translation_scale(
                [local.lower_bound_m[0], local.lower_bound_m[1], 0.0],
                [local.resolution_m, local.resolution_m, 1.0],
            )
            .with_relation(rerun::components::TransformRelation::ParentFromChild),
        )?;
        recording.log(
            "navigation/local_costmap_at_capture/grid",
            &rerun::SegmentationImage::new(
                local.class_ids,
                rerun::components::ImageFormat::segmentation(
                    [local.width, local.height],
                    rerun::ChannelDatatype::U8,
                ),
            ),
        )?;
    } else {
        recording.log(
            "navigation/local_costmap_at_capture/grid",
            &rerun::Clear::flat(),
        )?;
    }

    for (path, value) in [
        (
            "navigation/decision/left_pwm_percent",
            f64::from(message.requested_pwm[0]),
        ),
        (
            "navigation/decision/right_pwm_percent",
            f64::from(message.requested_pwm[1]),
        ),
    ] {
        recording.log(path, &rerun::Scalars::single(value))?;
    }
    if let Some(applied) = message.applied_actuation.as_ref() {
        for (path, value) in [
            (
                "navigation/actuation/applied_left_pwm_percent",
                f64::from(applied.applied_pwm[0]),
            ),
            (
                "navigation/actuation/applied_right_pwm_percent",
                f64::from(applied.applied_pwm[1]),
            ),
            (
                "navigation/actuation/remaining_lease_at_server_emission_ms",
                f64::from(applied.remaining_lease_at_server_emission_ms),
            ),
        ] {
            recording.log(path, &rerun::Scalars::single(value))?;
        }
        recording.log(
            "navigation/actuation/applied_receipt",
            &rerun::TextLog::new(format!(
                "boot_id={} control_epoch={} sequence={} acknowledged_at_host_ns={} known_active_through_host_ns={}",
                applied.controller_boot_id,
                applied.control_epoch,
                applied.sequence,
                applied.acknowledged_at_ns_decimal,
                applied.known_active_through_ns_decimal,
            )),
        )?;
    } else {
        recording.log("navigation/actuation", &rerun::Clear::recursive())?;
    }
    if let Ok(motor_packets_sent) = u32::try_from(message.shadow_record_motor_packets_sent) {
        recording.log(
            "navigation/decision/shadow_record_motor_packets_sent",
            &rerun::Scalars::single(f64::from(motor_packets_sent)),
        )?;
    } else {
        recording.log(
            "navigation/decision/shadow_record_motor_packets_sent",
            &rerun::Clear::flat(),
        )?;
    }
    if let Some(cost) = message.objective_cost {
        recording.log(
            "navigation/decision/objective_cost",
            &rerun::Scalars::single(cost),
        )?;
    } else {
        recording.log("navigation/decision/objective_cost", &rerun::Clear::flat())?;
    }
    if let Some(transform) = message.base_to_odom {
        for (suffix, value) in [
            ("x_m", transform[0]),
            ("y_m", transform[1]),
            ("yaw_rad", transform[2]),
        ] {
            recording.log(
                format!("navigation/transforms/base_to_odom/{suffix}"),
                &rerun::Scalars::single(value),
            )?;
        }
    } else {
        recording.log(
            "navigation/transforms/base_to_odom",
            &rerun::Clear::recursive(),
        )?;
    }
    if let Some(transform) = message.odom_to_map {
        for (suffix, value) in [
            ("x_m", transform[0]),
            ("y_m", transform[1]),
            ("yaw_rad", transform[2]),
        ] {
            recording.log(
                format!("navigation/transforms/odom_to_map/{suffix}"),
                &rerun::Scalars::single(value),
            )?;
        }
    } else {
        recording.log(
            "navigation/transforms/odom_to_map",
            &rerun::Clear::recursive(),
        )?;
    }
    if let Some(transform) = local_costmap_to_odom {
        for (suffix, value) in [
            ("x_m", transform[0]),
            ("y_m", transform[1]),
            ("yaw_rad", transform[2]),
        ] {
            recording.log(
                format!("navigation/transforms/local_costmap_to_odom/{suffix}"),
                &rerun::Scalars::single(value),
            )?;
        }
    } else {
        recording.log(
            "navigation/transforms/local_costmap_to_odom",
            &rerun::Clear::recursive(),
        )?;
    }
    let mut status = format!(
        "host_timestamp_ns={} decision_id={} request_id={} status={} goal_state={} shadow_record_motor_packets_sent={} reason={}",
        message.host_timestamp_ns,
        message.decision_id,
        message
            .request_id
            .map_or_else(|| "none".to_owned(), |id| id.to_string()),
        message.status,
        message.goal_state,
        message.shadow_record_motor_packets_sent,
        message.reason
    );
    if let Some(local_costmap_evidence) = local_costmap_evidence {
        status.push_str(" local_costmap=");
        status.push_str(&local_costmap_evidence);
    }
    if let Some(odometry_state) = message.odometry_state {
        status.push_str(" odometry=");
        status.push_str(&odometry_state);
    }
    if let Some(warning) = message.diagnostic_warning {
        status.push_str(" diagnostic_warning=");
        status.push_str(&warning);
    }
    recording.log("navigation/decision/status", &rerun::TextLog::new(status))?;
    Ok(())
}

#[cfg(feature = "record")]
fn log_live_viz_message(sink: &mut RerunSink, msg: LiveVizMsg) -> Result<(), VizLogError> {
    if let Some(packet) = msg.packet.as_ref() {
        sink.log_with_points(packet, msg.points.as_deref())?;
    } else {
        sink.log_frames(&msg.left, &msg.right)?;
    }
    if let Some(depth) = msg.depth.as_ref() {
        sink.log_depth(depth)?;
    }
    if let Some(pose) = msg.output.pose().as_ref() {
        sink.log_pose(msg.left.timestamp(), pose)?;
    }
    sink.log_system_health(msg.left.timestamp(), msg.output.health())?;
    sink.log_diagnostics(msg.left.timestamp(), msg.output.diagnostics())?;
    for event in msg.output.events() {
        sink.log_event(msg.left.timestamp(), event)?;
    }
    if let Some(ref dense_stats) = msg.dense_stats {
        sink.log_dense_stats(msg.left.timestamp(), dense_stats)?;
    }
    Ok(())
}

#[cfg(any(feature = "record", test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LiveDenseCommandClass {
    IntegrationData,
    OrderedControl,
}

#[cfg(any(feature = "record", test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LiveDenseRouteContext {
    TrackerOutput,
    PoseUpdateAfterTrackerError,
}

#[cfg(any(feature = "record", test))]
impl std::fmt::Display for LiveDenseRouteContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TrackerOutput => f.write_str("tracker output"),
            Self::PoseUpdateAfterTrackerError => f.write_str("pose update after tracker error"),
        }
    }
}

#[cfg(any(feature = "record", test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LiveDenseRouteDisposition {
    Enqueued,
    IntegrationDroppedNewest,
    Disconnected,
}

#[cfg(any(feature = "record", test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LiveDenseRouteError {
    ControlTimedOut { context: LiveDenseRouteContext },
    ControlMisclassifiedAsIntegration { context: LiveDenseRouteContext },
}

#[cfg(any(feature = "record", test))]
impl std::fmt::Display for LiveDenseRouteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ControlTimedOut { context } => {
                write!(f, "dense ordered control timed out while routing {context}")
            }
            Self::ControlMisclassifiedAsIntegration { context } => write!(
                f,
                "dense router misclassified ordered control as integration data while routing {context}"
            ),
        }
    }
}

#[cfg(any(feature = "record", test))]
impl std::error::Error for LiveDenseRouteError {}

#[cfg(any(feature = "record", test))]
fn classify_live_dense_route(
    outcome: DenseCommandSendOutcome,
    command_class: LiveDenseCommandClass,
    context: LiveDenseRouteContext,
) -> Result<LiveDenseRouteDisposition, LiveDenseRouteError> {
    match outcome {
        DenseCommandSendOutcome::Enqueued => Ok(LiveDenseRouteDisposition::Enqueued),
        DenseCommandSendOutcome::IntegrationDroppedNewest => match command_class {
            LiveDenseCommandClass::IntegrationData => {
                Ok(LiveDenseRouteDisposition::IntegrationDroppedNewest)
            }
            LiveDenseCommandClass::OrderedControl => {
                Err(LiveDenseRouteError::ControlMisclassifiedAsIntegration { context })
            }
        },
        DenseCommandSendOutcome::ControlTimedOut => {
            Err(LiveDenseRouteError::ControlTimedOut { context })
        }
        DenseCommandSendOutcome::Disconnected => Ok(LiveDenseRouteDisposition::Disconnected),
    }
}

#[cfg(feature = "record")]
type LiveCoordinatorAdmissionError = CoordinatorAdmissionError<NavigationIngressStreamWriteError>;

#[cfg(feature = "record")]
#[derive(Debug)]
enum LiveNavigationWorkerError {
    HostClock(HostMonotonicRangeError),
    VisualAdmission {
        source: LiveCoordinatorAdmissionError,
    },
    ImuAdmission {
        source: LiveCoordinatorAdmissionError,
    },
    DepthAdmission {
        source: LiveCoordinatorAdmissionError,
    },
    MapAdmission {
        source: LiveCoordinatorAdmissionError,
    },
    Tick {
        source: CoordinatorTickError,
    },
    TickSequenceExhausted,
    #[cfg(feature = "actuation")]
    Actuation {
        phase: &'static str,
        source: LiveActuationError,
    },
    #[cfg(feature = "actuation")]
    OperationAndDisarm {
        operation: Box<Self>,
        disarm: LiveActuationError,
    },
    JournalFinalization {
        source: NavigationIngressStreamWriteError,
    },
    JournalSync {
        source: std::io::Error,
    },
}

#[cfg(feature = "record")]
impl std::fmt::Display for LiveNavigationWorkerError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HostClock(source) => {
                write!(formatter, "navigation host clock failed: {source}")
            }
            Self::VisualAdmission { source } => {
                write!(formatter, "visual navigation admission failed: {source}")
            }
            Self::ImuAdmission { source } => {
                write!(formatter, "IMU navigation admission failed: {source}")
            }
            Self::DepthAdmission { source } => {
                write!(formatter, "depth navigation admission failed: {source}")
            }
            Self::MapAdmission { source } => {
                write!(
                    formatter,
                    "global-map navigation admission failed: {source}"
                )
            }
            Self::Tick { source } => write!(formatter, "navigation tick failed: {source}"),
            Self::TickSequenceExhausted => {
                formatter.write_str("navigation diagnostic tick sequence exhausted i64")
            }
            #[cfg(feature = "actuation")]
            Self::Actuation { phase, source } => {
                write!(
                    formatter,
                    "physical actuation failed during {phase}: {source}"
                )
            }
            #[cfg(feature = "actuation")]
            Self::OperationAndDisarm { operation, disarm } => write!(
                formatter,
                "navigation failed ({operation}); shutdown also could not prove controller stop ({disarm})"
            ),
            Self::JournalFinalization { source } => {
                write!(
                    formatter,
                    "navigation journal finalization failed: {source}"
                )
            }
            Self::JournalSync { source } => {
                write!(formatter, "navigation journal sync failed: {source}")
            }
        }
    }
}

#[cfg(feature = "record")]
impl std::error::Error for LiveNavigationWorkerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::HostClock(source) => Some(source),
            Self::VisualAdmission { source }
            | Self::ImuAdmission { source }
            | Self::DepthAdmission { source }
            | Self::MapAdmission { source } => Some(source),
            Self::Tick { source } => Some(source),
            Self::JournalFinalization { source } => Some(source),
            Self::JournalSync { source } => Some(source),
            #[cfg(feature = "actuation")]
            Self::Actuation { source, .. } => Some(source),
            #[cfg(feature = "actuation")]
            Self::OperationAndDisarm { operation, .. } => Some(operation.as_ref()),
            Self::TickSequenceExhausted => None,
        }
    }
}

#[cfg(feature = "record")]
struct LiveNavigationWorkerSuccess {
    descriptor: NavigationIngressSidecarDescriptor,
}

#[cfg(feature = "record")]
#[allow(clippy::too_many_arguments)]
fn run_live_navigation_worker(
    mut coordinator: ShadowNavigationCoordinator<NavigationIngressWriter<File>>,
    control_period: ControlPeriodNs,
    clock_origin: Instant,
    #[cfg(feature = "actuation")] actuation_config: Option<NavigationActuationConfigV1>,
    running: Arc<AtomicBool>,
    visual_rx: crossbeam_channel::Receiver<VisualAdmission>,
    depth_rx: DropReceiver<DepthObservation>,
    imu_rx: DropReceiver<ImuReport>,
    map_rx: DropReceiver<TimedOccupancySnapshot>,
    mut map_viz_tx: Option<DropSender<TimedOccupancySnapshot>>,
    mut navigation_viz_tx: Option<DropSender<LiveNavigationVizMsg>>,
) -> Result<LiveNavigationWorkerSuccess, LiveNavigationWorkerError> {
    let _exit_guard = LiveThreadExitGuard(Arc::clone(&running));
    #[cfg(feature = "actuation")]
    let mut physical_actuation = match actuation_config.as_ref() {
        Some(config) => {
            let (session, initial_zero) = PhysicalActuationSession::acquire(config, clock_origin)
                .map_err(|source| {
                LiveNavigationWorkerError::Actuation {
                    phase: "zero acquisition",
                    source,
                }
            })?;
            eprintln!(
                "physical actuation acquired: robot_id={} boot_id={} epoch={} sequence={} applied_pwm=[{},{}] known_active_through_host_ns={}",
                config.robot_id(),
                initial_zero.controller_session().boot_id().get(),
                initial_zero.controller_session().control_epoch().get(),
                initial_zero.sequence().get(),
                initial_zero.applied_timer_pwm().left().get(),
                initial_zero.applied_timer_pwm().right().get(),
                initial_zero
                    .known_active_through_exclusive()
                    .nanos_since_clock_start(),
            );
            Some(session)
        }
        None => None,
    };
    let operation_result = (|| -> Result<(), LiveNavigationWorkerError> {
        let mut clock = InstantHostClock::new(clock_origin);
        let tick_rx = crossbeam_channel::tick(control_period.as_duration());
        let never_visual = crossbeam_channel::never::<VisualAdmission>();
        let never_depth = crossbeam_channel::never::<DepthObservation>();
        let never_imu = crossbeam_channel::never::<ImuReport>();
        let never_map = crossbeam_channel::never::<TimedOccupancySnapshot>();
        let mut visual_open = true;
        let mut depth_open = true;
        let mut imu_open = true;
        let mut map_open = true;
        let mut tick_sequence = 0_i64;

        while visual_open || depth_open || imu_open || map_open {
            if !running.load(Ordering::SeqCst) {
                break;
            }
            let visual_receiver = if visual_open {
                &visual_rx
            } else {
                &never_visual
            };
            let depth_receiver = if depth_open {
                depth_rx.as_receiver()
            } else {
                &never_depth
            };
            let imu_receiver = if imu_open {
                imu_rx.as_receiver()
            } else {
                &never_imu
            };
            let map_receiver = if map_open {
                map_rx.as_receiver()
            } else {
                &never_map
            };
            crossbeam_channel::select! {
                recv(visual_receiver) -> message => match message {
                    Ok(admission) => {
                        let now = clock.checked_now().map_err(LiveNavigationWorkerError::HostClock)?;
                        coordinator.accept_visual(admission, now).map_err(|source| {
                            LiveNavigationWorkerError::VisualAdmission { source }
                        })?;
                    }
                    Err(_) => visual_open = false,
                },
                recv(depth_receiver) -> message => match message {
                    Ok(observation) => {
                        let now = clock.checked_now().map_err(LiveNavigationWorkerError::HostClock)?;
                        coordinator.accept_depth(observation, now).map_err(|source| {
                            LiveNavigationWorkerError::DepthAdmission { source }
                        })?;
                    }
                    Err(_) => depth_open = false,
                },
                recv(imu_receiver) -> message => match message {
                    Ok(report) => {
                        let now = clock.checked_now().map_err(LiveNavigationWorkerError::HostClock)?;
                        coordinator.accept_imu(report, now).map_err(|source| {
                            LiveNavigationWorkerError::ImuAdmission { source }
                        })?;
                    }
                    Err(_) => imu_open = false,
                },
                recv(map_receiver) -> message => match message {
                    Ok(snapshot) => {
                        // The inference producer routes each visual outcome before it can
                        // issue that attempt's dense-map command. Because visual and map
                        // traffic use distinct bounded queues, `select!` may nevertheless
                        // observe the derived map first. Admit every visual outcome already
                        // ready on its queue before admitting this map; the journal remains
                        // the authority for the actual cross-channel admission order.
                        loop {
                            match visual_rx.try_recv() {
                                Ok(admission) => {
                                    let now = clock
                                        .checked_now()
                                        .map_err(LiveNavigationWorkerError::HostClock)?;
                                    coordinator.accept_visual(admission, now).map_err(|source| {
                                        LiveNavigationWorkerError::VisualAdmission { source }
                                    })?;
                                }
                                Err(crossbeam_channel::TryRecvError::Empty) => break,
                                Err(crossbeam_channel::TryRecvError::Disconnected) => {
                                    visual_open = false;
                                    break;
                                }
                            }
                        }
                        let now = clock.checked_now().map_err(LiveNavigationWorkerError::HostClock)?;
                        coordinator
                            .accept_global_map(now, snapshot.timestamp(), snapshot.snapshot())
                            .map_err(|source| LiveNavigationWorkerError::MapAdmission { source })?;
                        if let Some(sender) = map_viz_tx.as_ref()
                            && matches!(sender.try_send(snapshot), SendOutcome::Disconnected)
                        {
                            map_viz_tx = None;
                        }
                    }
                    Err(_) => map_open = false,
                },
                recv(&tick_rx) -> _ => {
                    if !running.load(Ordering::SeqCst) {
                        break;
                    }
                    #[cfg(feature = "actuation")]
                    if let Some(session) = physical_actuation.as_mut() {
                        session.require_current_before_solve().map_err(|source| {
                            LiveNavigationWorkerError::Actuation {
                                phase: "pre-solve applied-evidence gate",
                                source,
                            }
                        })?;
                    }
                    let tick = clock.checked_now().map_err(LiveNavigationWorkerError::HostClock)?;
                    let outcome = coordinator
                        .tick(tick, &mut clock)
                        .map_err(|source| LiveNavigationWorkerError::Tick { source })?;
                    tick_sequence = tick_sequence
                        .checked_add(1)
                        .ok_or(LiveNavigationWorkerError::TickSequenceExhausted)?;
                    #[cfg(feature = "actuation")]
                    let applied_actuation = if let Some(session) = physical_actuation.as_mut() {
                        let receipt = session.apply(outcome.decision()).map_err(|source| {
                            LiveNavigationWorkerError::Actuation {
                                phase: "synchronous command application",
                                source,
                            }
                        })?;
                        Some(live_applied_actuation_viz(&receipt))
                    } else {
                        None
                    };
                    #[cfg(not(feature = "actuation"))]
                    let applied_actuation = None;
                    // Diagnostic copies happen only after the authoritative decision and never
                    // feed back into planning, MPC, journal admission, applied evidence, or shadow evidence.
                    if let Some(sender) = navigation_viz_tx.as_ref() {
                        let message = build_live_navigation_viz_message(
                            &coordinator,
                            tick,
                            tick_sequence,
                            &outcome,
                            applied_actuation,
                        );
                        if matches!(sender.try_send(message), SendOutcome::Disconnected) {
                            navigation_viz_tx = None;
                        }
                    }
                },
            }
        }

        Ok(())
    })();

    #[cfg(feature = "actuation")]
    let disarm_result = match physical_actuation.as_mut() {
        Some(session) if !session.is_consumed() => session.disarm().map(|receipt| {
            eprintln!(
                "physical actuation disarmed: boot_id={} request_id={} acknowledged_at_host_ns={}",
                receipt.observed_boot_id().get(),
                receipt.request_id().get(),
                receipt.acknowledged_at().nanos_since_clock_start(),
            );
        }),
        Some(_) | None => Ok(()),
    };
    #[cfg(feature = "actuation")]
    match (operation_result, disarm_result) {
        (Ok(()), Ok(())) => {}
        (Err(operation), Ok(())) => return Err(operation),
        (Ok(()), Err(source)) => {
            return Err(LiveNavigationWorkerError::Actuation {
                phase: "shutdown disarm",
                source,
            });
        }
        (Err(operation), Err(disarm)) => {
            return Err(LiveNavigationWorkerError::OperationAndDisarm {
                operation: Box::new(operation),
                disarm,
            });
        }
    }
    #[cfg(not(feature = "actuation"))]
    operation_result?;

    let finalized = coordinator
        .into_journal()
        .finish_with_descriptor()
        .map_err(|source| LiveNavigationWorkerError::JournalFinalization { source })?;
    let (file, descriptor) = finalized.into_parts();
    file.sync_all()
        .map_err(|source| LiveNavigationWorkerError::JournalSync { source })?;
    Ok(LiveNavigationWorkerSuccess { descriptor })
}

#[cfg(feature = "record")]
#[derive(Debug)]
enum LiveThreadError {
    RerunConnect {
        source: rerun::RecordingStreamError,
    },
    VisualizationConfiguration {
        source: VizConfigError,
    },
    VisualizationLog {
        source: VizLogError,
    },
    VisualizationFinalization {
        source: VizFlushError,
    },
    VisualizationLogAndFinalization {
        logging: VizLogError,
        finalization: VizFlushError,
    },
    VisualizationPacket {
        source: VizError,
    },
    DenseCommandGeneration(command_mapper::DenseCommandGenerationError),
    DenseCommandMapping(command_mapper::DenseCommandMappingError),
    DenseCommandRoute(LiveDenseRouteError),
    VisualIngressBoundary {
        source: NavigationIngressBoundaryError,
    },
    VisualAdmissionBuild {
        source: LiveVisualAdmissionBuildError,
    },
    VisualAdmissionRoute {
        source: LiveLosslessRouteError,
    },
    RequiredDenseUnavailable {
        reason: &'static str,
    },
    RequiredDenseAndInferenceUnavailable {
        reason: &'static str,
        inference: TrackerError,
    },
    InferenceUnavailable {
        source: TrackerError,
    },
    DenseCommandRouteAndInferenceUnavailable {
        routing: LiveDenseRouteError,
        inference: TrackerError,
    },
    DenseCommandGenerationAndInferenceUnavailable {
        generation: command_mapper::DenseCommandGenerationError,
        inference: TrackerError,
    },
    TrackerInitialization {
        source: TrackerInitError,
    },
    FrameProcessingPanic {
        detail: String,
    },
}

#[cfg(feature = "record")]
impl std::fmt::Display for LiveThreadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LiveThreadError::RerunConnect { source } => {
                write!(f, "failed to connect to rerun viewer: {source}")
            }
            LiveThreadError::VisualizationConfiguration { source } => {
                write!(f, "invalid live visualization configuration: {source}")
            }
            LiveThreadError::VisualizationLog { source } => {
                write!(f, "live visualization logging failed: {source}")
            }
            LiveThreadError::VisualizationFinalization { source } => {
                write!(f, "live visualization finalization failed: {source}")
            }
            LiveThreadError::VisualizationLogAndFinalization {
                logging,
                finalization,
            } => write!(
                f,
                "live visualization logging failed: {logging}; finalization also failed: {finalization}"
            ),
            LiveThreadError::VisualizationPacket { source } => {
                write!(f, "invalid live visualization packet: {source}")
            }
            LiveThreadError::DenseCommandGeneration(source) => {
                write!(f, "live dense command sequencing failed: {source}")
            }
            LiveThreadError::DenseCommandMapping(source) => {
                write!(f, "live dense command mapping failed: {source}")
            }
            LiveThreadError::DenseCommandRoute(source) => {
                write!(f, "live dense command routing failed: {source}")
            }
            LiveThreadError::VisualIngressBoundary { source } => {
                write!(f, "visual ingress identity failed: {source}")
            }
            LiveThreadError::VisualAdmissionBuild { source } => {
                write!(f, "visual admission classification failed: {source}")
            }
            LiveThreadError::VisualAdmissionRoute { source } => {
                write!(f, "visual admission routing failed: {source}")
            }
            LiveThreadError::RequiredDenseUnavailable { reason } => {
                write!(
                    f,
                    "required navigation occupancy became unavailable: {reason}"
                )
            }
            LiveThreadError::RequiredDenseAndInferenceUnavailable { reason, inference } => write!(
                f,
                "required navigation occupancy became unavailable: {reason}; inference pipeline is also unavailable: {inference}"
            ),
            LiveThreadError::InferenceUnavailable { source } => {
                write!(f, "inference pipeline is unavailable: {source}")
            }
            LiveThreadError::DenseCommandRouteAndInferenceUnavailable { routing, inference } => {
                write!(
                    f,
                    "live dense command routing failed: {routing}; inference pipeline is also unavailable: {inference}"
                )
            }
            LiveThreadError::DenseCommandGenerationAndInferenceUnavailable {
                generation,
                inference,
            } => write!(
                f,
                "live dense command sequencing failed: {generation}; inference pipeline is also unavailable: {inference}"
            ),
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
            Self::VisualizationLog { source } => Some(source),
            Self::VisualizationFinalization { source } => Some(source),
            Self::VisualizationLogAndFinalization { logging, .. } => Some(logging),
            Self::VisualizationPacket { source } => Some(source),
            Self::DenseCommandGeneration(source) => Some(source),
            Self::DenseCommandMapping(source) => Some(source),
            Self::DenseCommandRoute(source) => Some(source),
            Self::VisualIngressBoundary { source } => Some(source),
            Self::VisualAdmissionBuild { source } => Some(source),
            Self::VisualAdmissionRoute { source } => Some(source),
            Self::RequiredDenseUnavailable { .. } => None,
            Self::RequiredDenseAndInferenceUnavailable { inference, .. } => Some(inference),
            Self::InferenceUnavailable { source } => Some(source),
            Self::DenseCommandRouteAndInferenceUnavailable { routing, .. } => Some(routing),
            Self::DenseCommandGenerationAndInferenceUnavailable { generation, .. } => {
                Some(generation)
            }
            Self::TrackerInitialization { source } => Some(source),
            Self::RerunConnect { source } => Some(source),
            Self::FrameProcessingPanic { .. } => None,
        }
    }
}

#[cfg(feature = "record")]
impl From<command_mapper::DenseCommandGenerationError> for LiveThreadError {
    fn from(source: command_mapper::DenseCommandGenerationError) -> Self {
        Self::DenseCommandGeneration(source)
    }
}

#[cfg(feature = "record")]
impl From<command_mapper::DenseCommandMappingError> for LiveThreadError {
    fn from(source: command_mapper::DenseCommandMappingError) -> Self {
        Self::DenseCommandMapping(source)
    }
}

#[cfg(feature = "record")]
impl From<LiveDenseRouteError> for LiveThreadError {
    fn from(source: LiveDenseRouteError) -> Self {
        Self::DenseCommandRoute(source)
    }
}

#[cfg(feature = "record")]
#[derive(Debug)]
enum LiveWorkerFailure {
    Capture(LiveCaptureError),
    Inference(LiveThreadError),
    InferencePanic { detail: String },
    Occupancy(OccupancyRuntimeError),
    OccupancyPanic { detail: String },
    Navigation(LiveNavigationWorkerError),
    NavigationPanic { detail: String },
    DatasetFinalization(DatasetError),
    DatasetAbort(DatasetError),
    Visualization(LiveThreadError),
    VisualizationPanic { detail: String },
}

#[cfg(feature = "record")]
impl std::fmt::Display for LiveWorkerFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Capture(source) => write!(f, "live capture failed: {source}"),
            Self::Inference(source) => write!(f, "live inference worker failed: {source}"),
            Self::InferencePanic { detail } => {
                write!(f, "live inference worker panicked: {detail}")
            }
            Self::Occupancy(source) => write!(f, "live occupancy worker failed: {source}"),
            Self::OccupancyPanic { detail } => {
                write!(f, "live occupancy worker panicked: {detail}")
            }
            Self::Navigation(source) => write!(f, "live navigation worker failed: {source}"),
            Self::NavigationPanic { detail } => {
                write!(f, "live navigation worker panicked: {detail}")
            }
            Self::DatasetFinalization(source) => {
                write!(f, "live navigation dataset finalization failed: {source}")
            }
            Self::DatasetAbort(source) => {
                write!(
                    f,
                    "unpublished live navigation dataset abort failed: {source}"
                )
            }
            Self::Visualization(source) => {
                write!(f, "live visualization worker failed: {source}")
            }
            Self::VisualizationPanic { detail } => {
                write!(f, "live visualization worker panicked: {detail}")
            }
        }
    }
}

#[cfg(feature = "record")]
impl std::error::Error for LiveWorkerFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Capture(source) => Some(source),
            Self::Inference(source) | Self::Visualization(source) => Some(source),
            Self::Occupancy(source) => Some(source),
            Self::Navigation(source) => Some(source),
            Self::DatasetFinalization(source) | Self::DatasetAbort(source) => Some(source),
            Self::InferencePanic { .. }
            | Self::OccupancyPanic { .. }
            | Self::NavigationPanic { .. }
            | Self::VisualizationPanic { .. } => None,
        }
    }
}

#[cfg(feature = "record")]
#[derive(Debug)]
struct LiveRunError {
    failures: Vec<LiveWorkerFailure>,
}

#[cfg(feature = "record")]
impl std::fmt::Display for LiveRunError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "live session failed")?;
        for (index, failure) in self.failures.iter().enumerate() {
            write!(f, "; failure {}: {failure}", index + 1)?;
        }
        Ok(())
    }
}

#[cfg(feature = "record")]
impl std::error::Error for LiveRunError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.failures
            .first()
            .map(|failure| failure as &(dyn std::error::Error + 'static))
    }
}

#[cfg(any(feature = "record", test))]
struct LiveThreadExitGuard(Arc<AtomicBool>);

#[cfg(any(feature = "record", test))]
impl Drop for LiveThreadExitGuard {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

#[cfg(feature = "record")]
#[derive(Debug)]
enum LiveCaptureError {
    LeftImage { source: ImageError },
    RightImage { source: ImageError },
    LeftFrame { source: FrameError },
    RightFrame { source: FrameError },
    PairingInput { source: PairingInputError },
    StereoObservation { source: StereoObservationError },
    DatasetWrite { source: RecordCaptureError },
    Depth { source: DepthError },
    DepthFrame { source: RectifiedLeftDepthError },
    DepthObservation { source: DepthObservationError },
    Imu { source: ImuError },
    ImuSample { source: InertialValueError },
    ImuOrdering { source: InertialOrderingError },
    ImuRouteDisconnected,
    HostTimestamp { source: HostMonotonicRangeError },
    DeviceClose { source: DeviceCloseFailure },
}

#[cfg(feature = "record")]
impl std::fmt::Display for LiveCaptureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LeftImage { source } => write!(f, "left camera capture failed: {source}"),
            Self::RightImage { source } => write!(f, "right camera capture failed: {source}"),
            Self::LeftFrame { source } => {
                write!(f, "left camera returned an invalid frame: {source}")
            }
            Self::RightFrame { source } => {
                write!(f, "right camera returned an invalid frame: {source}")
            }
            Self::PairingInput { source } => write!(f, "stereo pairing input failed: {source}"),
            Self::StereoObservation { source } => {
                write!(f, "stereo observation contract failed: {source}")
            }
            Self::DatasetWrite { source } => {
                write!(f, "navigation dataset capture failed: {source}")
            }
            Self::Depth { source } => write!(f, "depth camera capture failed: {source}"),
            Self::DepthFrame { source } => write!(f, "depth camera contract failed: {source}"),
            Self::DepthObservation { source } => {
                write!(f, "navigation depth observation contract failed: {source}")
            }
            Self::Imu { source } => write!(f, "IMU capture failed: {source}"),
            Self::ImuSample { source } => write!(f, "IMU sample contract failed: {source}"),
            Self::ImuOrdering { source } => write!(f, "IMU ordering contract failed: {source}"),
            Self::ImuRouteDisconnected => {
                write!(f, "IMU estimator route disconnected during capture")
            }
            Self::HostTimestamp { source } => {
                write!(f, "capture host-arrival timestamp failed: {source}")
            }
            Self::DeviceClose { source } => write!(f, "OAK device close failed: {source}"),
        }
    }
}

#[cfg(feature = "record")]
impl std::error::Error for LiveCaptureError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::LeftImage { source } | Self::RightImage { source } => Some(source),
            Self::LeftFrame { source } | Self::RightFrame { source } => Some(source),
            Self::PairingInput { source } => Some(source),
            Self::StereoObservation { source } => Some(source),
            Self::DatasetWrite { source } => Some(source),
            Self::Depth { source } => Some(source),
            Self::DepthFrame { source } => Some(source),
            Self::DepthObservation { source } => Some(source),
            Self::Imu { source } => Some(source),
            Self::ImuSample { source } => Some(source),
            Self::ImuOrdering { source } => Some(source),
            Self::ImuRouteDisconnected => None,
            Self::HostTimestamp { source } => Some(source),
            Self::DeviceClose { source } => Some(source),
        }
    }
}

#[cfg(feature = "record")]
fn finish_live_device_session(
    operation: Result<(), Box<dyn std::error::Error>>,
    close: Result<(), OakCloseError>,
) -> Result<(), Box<dyn std::error::Error>> {
    match (operation, close.map_err(DeviceCloseFailure::new)) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(operation), Ok(())) => Err(operation),
        (Ok(()), Err(source)) => Err(LiveRunError {
            failures: vec![LiveWorkerFailure::Capture(LiveCaptureError::DeviceClose {
                source,
            })],
        }
        .into()),
        (Err(operation), Err(source)) => match operation.downcast::<LiveRunError>() {
            Ok(mut live) => {
                live.failures
                    .push(LiveWorkerFailure::Capture(LiveCaptureError::DeviceClose {
                        source,
                    }));
                Err(live)
            }
            Err(operation) => Err(Box::new(OperationAndDeviceCloseError {
                operation,
                close: source,
            })),
        },
    }
}

#[cfg(feature = "record")]
fn drain_depth_batch(rx: &DropReceiver<DepthImage>) -> Vec<DepthImage> {
    std::iter::from_fn(|| rx.try_recv().ok()).collect()
}

#[cfg(feature = "record")]
struct PreparedLiveNavigationRuntime {
    goal: Option<NavigationGoalArg>,
    dataset_path: PathBuf,
    control_period: ControlPeriodNs,
    occupancy_config: Option<OccupancyRuntimeConfig>,
    ingress_capacity: NavigationIngressCapacity,
    odometry: PlanarOdometry,
    local_costmap: LocalCostmap,
    global_planner: GlobalPlannerConfig,
    reference_builder: PathReferenceBuilderV1,
    mpc_config: MpcConfigV1,
    solver_budget: SolverBudgetNs,
    safety: ShadowSafetySupervisor,
    #[cfg(feature = "actuation")]
    actuation: Option<NavigationActuationConfigV1>,
}

#[cfg(feature = "record")]
struct ActiveLiveNavigation {
    coordinator: ShadowNavigationCoordinator<NavigationIngressWriter<File>>,
    control_period: ControlPeriodNs,
    dataset_writer: PairedDatasetWriter,
    dataset_handle: DatasetWriterHandle,
    #[cfg(feature = "actuation")]
    actuation: Option<NavigationActuationConfigV1>,
}

#[cfg(feature = "record")]
#[derive(Debug)]
struct NavigationSetupAndDatasetAbortError {
    setup: Box<dyn std::error::Error>,
    abort: DatasetError,
}

#[cfg(feature = "record")]
impl std::fmt::Display for NavigationSetupAndDatasetAbortError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "live navigation setup failed ({}); unpublished dataset abort also failed: {}",
            self.setup, self.abort
        )
    }
}

#[cfg(feature = "record")]
impl std::error::Error for NavigationSetupAndDatasetAbortError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.setup.as_ref())
    }
}

#[cfg(feature = "record")]
fn fail_navigation_setup_after_dataset<T>(
    writer: PairedDatasetWriter,
    handle: DatasetWriterHandle,
    setup: impl std::error::Error + 'static,
) -> Result<T, Box<dyn std::error::Error>> {
    drop(writer);
    match handle.abort_without_manifest() {
        Ok(_) => Err(Box::new(setup)),
        Err(abort) => Err(Box::new(NavigationSetupAndDatasetAbortError {
            setup: Box::new(setup),
            abort,
        })),
    }
}

#[cfg(feature = "record")]
#[allow(clippy::too_many_arguments)]
fn prepare_live_navigation_runtime(
    request: &LiveNavigationRequest,
    config_bytes: Option<&[u8]>,
    actuation_config_bytes: Option<&[u8]>,
    runtime_depth_camera: DepthCameraModel,
    device_session: DeviceSessionId,
) -> Result<Option<PreparedLiveNavigationRuntime>, Box<dyn std::error::Error>> {
    let LiveNavigationRequest::Enabled {
        goal,
        dataset_path,
        actuation,
        ..
    } = request
    else {
        if config_bytes.is_some() {
            return Err(Box::new(
                LiveNavigationPreparationError::NavigationPolicyWhileDisabled,
            ));
        }
        if actuation_config_bytes.is_some() {
            return Err(Box::new(
                LiveNavigationPreparationError::PhysicalAuthorityInShadowMode,
            ));
        }
        return Ok(None);
    };
    let Some(bytes) = config_bytes else {
        return Err(Box::new(
            LiveNavigationPreparationError::MissingNavigationPolicy,
        ));
    };
    let parsed = ShadowNavigationConfigV1::parse_json(bytes, runtime_depth_camera)?;
    let occupancy_config = build_navigation_occupancy_runtime_config(&parsed)?;
    let parts = parsed.into_runtime_parts();
    #[cfg(feature = "actuation")]
    let actuation = match actuation {
        LiveActuationRequest::ShadowOnly => {
            if actuation_config_bytes.is_some() {
                return Err(Box::new(
                    LiveNavigationPreparationError::PhysicalAuthorityInShadowMode,
                ));
            }
            None
        }
        LiveActuationRequest::Physical { exact_robot_id, .. } => {
            let Some(actuation_bytes) = actuation_config_bytes else {
                return Err(Box::new(
                    LiveNavigationPreparationError::MissingPhysicalAuthority,
                ));
            };
            Some(NavigationActuationConfigV1::parse_and_authorize(
                actuation_bytes,
                exact_robot_id,
                bytes,
                parts.mpc_solver.model(),
                parts.solver_budget,
                parts.control_period,
            )?)
        }
    };
    #[cfg(not(feature = "actuation"))]
    match actuation {
        LiveActuationRequest::ShadowOnly if actuation_config_bytes.is_none() => {}
        LiveActuationRequest::ShadowOnly => {
            return Err(Box::new(
                LiveNavigationPreparationError::PhysicalAuthorityInShadowMode,
            ));
        }
        LiveActuationRequest::Physical { .. } => {
            return Err(Box::new(
                LiveNavigationPreparationError::PhysicalActuationFeatureDisabled,
            ));
        }
    }
    let mpc_config = parts.mpc_solver.config();
    let odometry = PlanarOdometry::new(parts.odometry);
    let local_costmap = LocalCostmap::try_new(parts.local_costmap, device_session)?;
    let reference_builder = PathReferenceBuilderV1::new(parts.path_reference);
    let safety = ShadowSafetySupervisor::try_new(parts.mpc_solver, parts.shadow_command)?;
    Ok(Some(PreparedLiveNavigationRuntime {
        goal: *goal,
        dataset_path: dataset_path.clone(),
        control_period: parts.control_period,
        occupancy_config: Some(occupancy_config),
        ingress_capacity: parts.ingress_capacity,
        odometry,
        local_costmap,
        global_planner: parts.global_planner,
        reference_builder,
        mpc_config,
        solver_budget: parts.solver_budget,
        safety,
        #[cfg(feature = "actuation")]
        actuation,
    }))
}

#[cfg(feature = "record")]
#[allow(clippy::too_many_arguments)]
fn activate_live_navigation(
    runtime: PreparedLiveNavigationRuntime,
    mono_config: &MonoConfig,
    depth_config: Option<&DepthConfig>,
    imu_config: Option<&ImuConfig>,
    calibration: &Calibration,
    pairing_window: PairingWindowNs,
    device_session: DeviceSessionId,
    clock_epoch: NavigationClockEpoch,
    oak_provenance: &OakRuntimeProvenance,
) -> Result<ActiveLiveNavigation, Box<dyn std::error::Error>> {
    let meta = build_meta(mono_config, depth_config, imu_config, oak_provenance);
    let imu_metadata = ImuStreamMetadata::new(
        device_session,
        ImuExtrinsicProvenance::uncalibrated_unknown(),
    );
    let writer_config = DatasetWriterConfig {
        backpressure: Backpressure::Block,
        ..DatasetWriterConfig::default()
    };
    let (dataset_writer, dataset_handle) = DatasetWriter::create_paired_with_imu_config(
        &runtime.dataset_path,
        &meta,
        calibration,
        pairing_window,
        imu_metadata,
        writer_config,
    )?;
    let journal_path = runtime.dataset_path.join(NAVIGATION_INGRESS_STREAM_FILE);
    let journal_file = match OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&journal_path)
    {
        Ok(file) => file,
        Err(source) => {
            return fail_navigation_setup_after_dataset(dataset_writer, dataset_handle, source);
        }
    };
    let recording_id = match generate_navigation_recording_id() {
        Ok(recording_id) => recording_id,
        Err(source) => {
            return fail_navigation_setup_after_dataset(dataset_writer, dataset_handle, source);
        }
    };
    let journal =
        match NavigationIngressWriter::new(journal_file, recording_id, runtime.ingress_capacity) {
            Ok(journal) => journal,
            Err(source) => {
                return fail_navigation_setup_after_dataset(dataset_writer, dataset_handle, source);
            }
        };
    let coordinator = match runtime.goal {
        Some(goal) => ShadowNavigationCoordinator::new(
            clock_epoch,
            journal,
            goal.point(),
            runtime.odometry,
            runtime.local_costmap,
            runtime.global_planner,
            runtime.reference_builder,
            runtime.mpc_config,
            runtime.solver_budget,
            runtime.safety,
        ),
        None => ShadowNavigationCoordinator::new_without_goal(
            clock_epoch,
            journal,
            runtime.odometry,
            runtime.local_costmap,
            runtime.global_planner,
            runtime.reference_builder,
            runtime.mpc_config,
            runtime.solver_budget,
            runtime.safety,
        ),
    };
    Ok(ActiveLiveNavigation {
        coordinator,
        control_period: runtime.control_period,
        dataset_writer,
        dataset_handle,
        #[cfg(feature = "actuation")]
        actuation: runtime.actuation,
    })
}

#[cfg(feature = "record")]
fn run_live(args: LiveArgs) -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "actuation")]
    let (navigation_actuation_config, navigation_arm_robot) = (
        args.navigation_actuation_config.clone(),
        args.navigation_arm_robot.clone(),
    );
    #[cfg(not(feature = "actuation"))]
    let (navigation_actuation_config, navigation_arm_robot) = (None, None);
    let navigation_request = LiveNavigationRequest::parse(
        args.navigation_config.clone(),
        args.navigation_goal,
        args.navigation_record.clone(),
        navigation_actuation_config,
        navigation_arm_robot,
    )?;
    let navigation_config_bytes = match &navigation_request {
        LiveNavigationRequest::Disabled => None,
        LiveNavigationRequest::Enabled { config_path, .. } => {
            Some(read_navigation_config_bounded(config_path)?)
        }
    };
    #[cfg(feature = "actuation")]
    let navigation_actuation_config_bytes = match &navigation_request {
        LiveNavigationRequest::Enabled {
            actuation: LiveActuationRequest::Physical { config_path, .. },
            ..
        } => Some(read_actuation_config_bounded(config_path)?),
        LiveNavigationRequest::Disabled
        | LiveNavigationRequest::Enabled {
            actuation: LiveActuationRequest::ShadowOnly,
            ..
        } => None,
    };
    #[cfg(not(feature = "actuation"))]
    let navigation_actuation_config_bytes: Option<Vec<u8>> = None;
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
    let dense_requested = if depth_enabled {
        env_bool("KIKO_DENSE")?.unwrap_or(false)
    } else {
        false
    };
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
    let imu_config = args.camera.imu_rate_hz.map(|rate_hz| ImuConfig {
        rate_hz: rate_hz.get(),
    });
    if navigation_request.is_enabled() {
        if !depth_enabled {
            return Err(LiveNavigationPrerequisiteError::DepthDisabled.into());
        }
        if imu_config.is_none() {
            return Err(LiveNavigationPrerequisiteError::ImuDisabled.into());
        }
        if !dense_requested {
            return Err(LiveNavigationPrerequisiteError::DenseOccupancyDisabled.into());
        }
        if !mono_config.rectified {
            return Err(LiveNavigationPrerequisiteError::UnrectifiedStereo.into());
        }
    }
    // This command does not reconnect. One invocation is therefore one
    // explicitly delimited device-clock session.
    let device_session = DeviceSessionId::try_new(1)?;
    let imu_session = imu_config.map(|_| device_session);
    let imu_queue_capacity = if imu_config.is_some() {
        Some(ChannelCapacity::try_from(
            env_usize("KIKO_LIVE_IMU_QUEUE_DEPTH")?.unwrap_or(256),
        )?)
    } else {
        None
    };

    let depth_config = depth_enabled.then_some(DepthConfig {
        width: mono_config.width,
        height: mono_config.height,
        fps: mono_config.fps,
        alignment: DepthAlignment::RectifiedLeft,
    });
    let config = DeviceConfig {
        rgb: None,
        mono: Some(mono_config),
        depth: depth_config,
        imu: imu_config,
        queue: QueueConfig {
            size: 8,
            blocking: false,
        },
    };

    eprintln!(
        "connecting to OAK MXID {:?}...",
        args.camera.oak_device_id.as_str()
    );
    let mut device = Device::connect(args.camera.oak_device_id.as_str(), config)?;
    let oak_provenance = inspect_oak_runtime(&device, "live")?;
    let operation = (|| -> Result<(), Box<dyn std::error::Error>> {
        let pairing_window = load_pairing_window()?;
        let pairer_max_pending = load_pairer_max_pending_per_side()?;
        let mut pairer = StereoPairer::new_with_max_pending(pairing_window, pairer_max_pending)?;
        let StereoBootstrap {
            calibration,
            rectified_left_intrinsics,
        } = bootstrap_stereo(&mut device, &mono_config, running.as_ref(), &mut pairer)?;
        let capture_clock_origin = Instant::now();
        let navigation_clock_epoch =
            NavigationClockEpoch::new(HostMonotonicTimestamp::from_nanos(0));
        let rectified = RectifiedStereo::from_calibration(&calibration)?;
        let runtime_depth_camera = DepthCameraModel::new(
            rectified.left(),
            rectified.dimensions(),
            DepthToTrackingCamera::identity(),
        );
        let mut prepared_navigation_runtime = prepare_live_navigation_runtime(
            &navigation_request,
            navigation_config_bytes.as_deref(),
            navigation_actuation_config_bytes.as_deref(),
            runtime_depth_camera,
            device_session,
        )?;
        let navigation_enabled = prepared_navigation_runtime.is_some();

        let pair_queue_depth = env_usize("KIKO_LIVE_PAIR_QUEUE_DEPTH")?.unwrap_or(12);
        let pair_capacity = ChannelCapacity::try_from(pair_queue_depth)?;
        let (pair_tx, pair_rx, pair_stats) =
            bounded_channel::<StereoObservation>(pair_capacity, DropPolicy::DropOldest);

        let viz_queue_depth = env_usize("KIKO_LIVE_VIZ_QUEUE_DEPTH")?.unwrap_or(12);
        let viz_capacity = ChannelCapacity::try_from(viz_queue_depth)?;
        let (viz_tx, viz_rx, viz_stats) = bounded_channel(viz_capacity, DropPolicy::DropNewest);
        let (depth_tx, depth_rx, mut navigation_depth_rx, depth_stats_handle) =
            if let Some(depth_capacity) = depth_queue_capacity {
                let (depth_tx, depth_routes, depth_stats) =
                    depth_router(depth_capacity, DropPolicy::DropOldest);
                (
                    Some(depth_tx),
                    Some(depth_routes.slam),
                    Some(depth_routes.navigation),
                    Some(depth_stats),
                )
            } else {
                (None, None, None, None)
            };
        let (mut imu_tx, mut navigation_imu_rx, imu_stats_handle) =
            match (imu_session, imu_queue_capacity) {
                (Some(session_id), Some(capacity)) => {
                    let (tx, rx, stats) = imu_report_router(session_id, capacity);
                    (Some(tx), Some(rx), Some(stats))
                }
                (None, None) => (None, None, None),
                _ => unreachable!("IMU session and queue capacity are derived together"),
            };

        let inference = InferenceConfig::from_args(&args.inference)?;
        let InferenceConfig {
            superpoint_left,
            superpoint_right,
            lightglue,
            key_limit,
            downscale,
        } = inference;

        let tracker_config = build_tracker_config(
            TrackerDefaults {
                min_keyframe_points: 80,
                refresh_inliers: 20,
                min_inliers: 15,
            },
            key_limit,
            downscale,
        )?;
        let mut navigation_occupancy_config = prepared_navigation_runtime
            .as_mut()
            .and_then(|runtime| runtime.occupancy_config.take());
        let (navigation_visual_tx, navigation_visual_rx) = if navigation_enabled {
            let (tx, rx) = crossbeam_channel::bounded(LIVE_NAVIGATION_VISUAL_QUEUE_CAPACITY);
            (Some(tx), Some(rx))
        } else {
            (None, None)
        };

        eprintln!(
            "live: pair_queue_depth={} viz_queue_depth={} depth_enabled={} depth_queue_depth={} imu_enabled={} imu_rate_hz={} imu_queue_depth={} pairing_window_ns={} pairer_max_pending_per_side={}",
            pair_queue_depth,
            viz_queue_depth,
            depth_enabled,
            depth_queue_capacity.map_or(0, ChannelCapacity::get),
            imu_config.is_some(),
            imu_config.map_or(0, |config| config.rate_hz),
            imu_queue_capacity.map_or(0, ChannelCapacity::get),
            pairer.window().as_ns(),
            pairer.max_pending_per_side()
        );

        // Dense reconstruction channels and worker thread.
        let dense_enabled = depth_enabled && dense_requested;
        let dense_capacities = if dense_enabled {
            Some((
                ChannelCapacity::try_from(env_usize("KIKO_DENSE_DATA_QUEUE_DEPTH")?.unwrap_or(4))?,
                ChannelCapacity::try_from(env_usize("KIKO_DENSE_CTRL_QUEUE_DEPTH")?.unwrap_or(64))?,
            ))
        } else {
            None
        };
        let occupancy_config = if let Some(config) = navigation_occupancy_config.take() {
            Some(config)
        } else if dense_enabled {
            let depth_projection = DepthProjectionContract::new(
                rectified.dimensions(),
                DepthOpticalFrame::RectifiedLeft,
            );
            Some(build_occupancy_runtime_config(
                rectified.left(),
                rectified.dimensions(),
                depth_projection,
            )?)
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
        let mut occupancy_snapshot_tx_for_worker = None;
        let mut occupancy_snapshot_rx = None;
        let mut occupancy_snapshot_stats_handle = None;
        let mut navigation_snapshot_rx = None;
        let mut navigation_map_viz_tx = None;
        let mut occupancy_viz_forward_stats_handle = None;
        let (navigation_viz_tx, navigation_viz_rx, navigation_viz_stats_handle) =
            if navigation_enabled {
                let (tx, rx, stats) =
                    bounded_channel(ChannelCapacity::try_from(4_usize)?, DropPolicy::DropNewest);
                (Some(tx), Some(rx), Some(stats))
            } else {
                (None, None, None)
            };

        if let Some((data_cap, ctrl_cap)) = dense_capacities {
            let (command_tx, command_rx, command_stats) =
                dense_command_channel(data_cap, ctrl_cap, Duration::from_millis(5))?;
            let stats_cap = ChannelCapacity::try_from(1_usize)?;
            let (stats_tx, stats_rx_inner, _stats_handle) =
                bounded_channel(stats_cap, DropPolicy::DropOldest);
            let (snapshot_tx, snapshot_rx, snapshot_stats) =
                bounded_channel(ChannelCapacity::try_from(1_usize)?, DropPolicy::DropOldest);
            dense_command_tx = Some(command_tx);
            dense_command_rx_for_worker = Some(command_rx);
            dense_command_stats_handle = Some(command_stats);
            dense_stats_tx_for_worker = Some(stats_tx);
            dense_stats_rx = Some(stats_rx_inner);
            occupancy_snapshot_tx_for_worker = Some(snapshot_tx);
            if navigation_enabled {
                let (viz_map_tx, viz_map_rx, viz_map_stats) =
                    bounded_channel(ChannelCapacity::try_from(1_usize)?, DropPolicy::DropOldest);
                navigation_snapshot_rx = Some(snapshot_rx);
                navigation_map_viz_tx = Some(viz_map_tx);
                occupancy_snapshot_rx = Some(viz_map_rx);
                occupancy_viz_forward_stats_handle = Some(viz_map_stats);
            } else {
                occupancy_snapshot_rx = Some(snapshot_rx);
            }
            occupancy_snapshot_stats_handle = Some(snapshot_stats);
        }

        let mut depth_ring = DepthRingBuffer::try_new(depth_ring_capacity.get())?;
        // Dataset creation is intentionally the final fallible setup boundary.
        // From this point every exit path consumes its handle through either
        // bound finalization or abort_without_manifest.
        let active_navigation = prepared_navigation_runtime
            .take()
            .map(|runtime| {
                activate_live_navigation(
                    runtime,
                    &mono_config,
                    depth_config.as_ref(),
                    imu_config.as_ref(),
                    &calibration,
                    pairing_window,
                    device_session,
                    navigation_clock_epoch,
                    &oak_provenance,
                )
            })
            .transpose()?;
        #[cfg(feature = "actuation")]
        let navigation_actuation_config = active_navigation
            .as_ref()
            .and_then(|active| active.actuation.clone());
        let (
            navigation_coordinator,
            navigation_control_period,
            navigation_dataset_writer,
            navigation_dataset_handle,
        ) = match active_navigation {
            Some(active) => (
                Some(active.coordinator),
                Some(active.control_period),
                Some(active.dataset_writer),
                Some(active.dataset_handle),
            ),
            None => (None, None, None, None),
        };

        let dense_handle = if let (Some(config), Some(command_rx), stats_tx, snapshot_tx) = (
            occupancy_config,
            dense_command_rx_for_worker.take(),
            dense_stats_tx_for_worker.take(),
            occupancy_snapshot_tx_for_worker.take(),
        ) {
            let dense_running = Arc::clone(&running);
            Some(thread::spawn(move || {
                let _exit_guard = LiveThreadExitGuard(dense_running);
                kiko_slam::dense::occupancy_runtime::run_occupancy_worker(
                    config,
                    &command_rx,
                    stats_tx.as_ref(),
                    snapshot_tx,
                )
            }))
        } else {
            None
        };

        let navigation_handle = match navigation_coordinator {
            Some(coordinator) => {
                let control_period = navigation_control_period
                    .expect("enabled navigation has a parsed control period");
                let visual_rx = navigation_visual_rx
                    .expect("enabled navigation has a lossless visual receiver");
                let depth_rx = navigation_depth_rx
                    .take()
                    .expect("enabled navigation requires a depth route");
                let imu_rx = navigation_imu_rx
                    .take()
                    .expect("enabled navigation requires an IMU route")
                    .reports;
                let map_rx = navigation_snapshot_rx
                    .take()
                    .expect("enabled navigation requires a dense map route");
                let navigation_running = Arc::clone(&running);
                Some(thread::spawn(move || {
                    run_live_navigation_worker(
                        coordinator,
                        control_period,
                        capture_clock_origin,
                        #[cfg(feature = "actuation")]
                        navigation_actuation_config,
                        navigation_running,
                        visual_rx,
                        depth_rx,
                        imu_rx,
                        map_rx,
                        navigation_map_viz_tx,
                        navigation_viz_tx,
                    )
                }))
            }
            None => None,
        };

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
            let mut dense_generation = command_mapper::DenseCommandGeneration::default();
            let mut dense_command_tx = dense_command_tx;
            let dense_stats_rx = dense_stats_rx;
            let mut dense_active = dense_enabled;
            let mut dense_integrations_dropped_newest: u64 = 0;
            let mut depth_reorder_warnings_seen: u64 = 0;
            let mut viz_tx = Some(viz_tx);
            let navigation_visual_tx = navigation_visual_tx;

            for observation in pair_rx.iter() {
                let pending_visual = if navigation_visual_tx.is_some() {
                    Some(
                        PendingVisualAttemptIngress::from_observation(
                            navigation_clock_epoch,
                            &observation,
                        )
                        .map_err(|source| LiveThreadError::VisualIngressBoundary { source })?,
                    )
                } else {
                    None
                };
                let left = observation.pair().left().clone();
                let right = observation.pair().right().clone();
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
                let process_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    tracker.process(observation.into_pair())
                }));
                match process_result {
                    Ok(Ok(mut output)) => {
                        if let (Some(sender), Some(pending)) =
                            (navigation_visual_tx.as_ref(), pending_visual)
                        {
                            let admission = visual_admission_from_output(pending, &output)
                                .map_err(|source| LiveThreadError::VisualAdmissionBuild {
                                    source,
                                })?;
                            route_visual_admission(sender, admission).map_err(|source| {
                                LiveThreadError::VisualAdmissionRoute { source }
                            })?;
                        }
                        // Map tracker output to dense commands.
                        let pose_updates = tracker.take_pending_dense_pose_updates();
                        let dense_stats = if dense_active {
                            let cmds = command_mapper::map_output_to_dense_commands(
                                &output,
                                pose_updates,
                                |keyframe_id| tracker.keyframe_pose(keyframe_id),
                                &depth_ring,
                                timestamp,
                                &mut dense_generation,
                            )?;
                            for cmd in cmds {
                                if let Some(ref tx) = dense_command_tx {
                                    let command_class = if matches!(
                                        &cmd,
                                        dense::DenseCommand::IntegrateKeyframe { .. }
                                    ) {
                                        LiveDenseCommandClass::IntegrationData
                                    } else {
                                        LiveDenseCommandClass::OrderedControl
                                    };
                                    match classify_live_dense_route(
                                        tx.route(cmd),
                                        command_class,
                                        LiveDenseRouteContext::TrackerOutput,
                                    )? {
                                        LiveDenseRouteDisposition::Enqueued => {}
                                        LiveDenseRouteDisposition::IntegrationDroppedNewest => {
                                            dense_integrations_dropped_newest =
                                                dense_integrations_dropped_newest.saturating_add(1);
                                        }
                                        LiveDenseRouteDisposition::Disconnected => {
                                            if navigation_enabled {
                                                return Err(
                                                    LiveThreadError::RequiredDenseUnavailable {
                                                        reason: "dense command consumer disconnected",
                                                    },
                                                );
                                            }
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
                            if navigation_enabled {
                                return Err(LiveThreadError::RequiredDenseUnavailable {
                                    reason: "dense runtime entered Down state",
                                });
                            }
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
                            packet = Some(
                                VizPacket::try_new(left.clone(), right.clone(), matches).map_err(
                                    |source| LiveThreadError::VisualizationPacket { source },
                                )?,
                            );
                        }
                        let msg = LiveVizMsg {
                            left,
                            right,
                            depth,
                            packet,
                            points,
                            output,
                            dense_stats,
                        };
                        if let Some(sender) = viz_tx.as_ref()
                            && matches!(sender.try_send(msg), SendOutcome::Disconnected)
                        {
                            eprintln!(
                                "live visualization consumer disconnected; continuing authoritative tracking and occupancy"
                            );
                            viz_tx = None;
                        }
                    }
                    Ok(Err(err)) => {
                        let requires_pipeline_shutdown = err.requires_pipeline_shutdown();
                        if let (Some(sender), Some(pending)) =
                            (navigation_visual_tx.as_ref(), pending_visual)
                        {
                            let admission = if requires_pipeline_shutdown {
                                VisualAdmission::fatal_failure(
                                    pending.complete(VisualAttemptOutcome::FatalFailure),
                                )
                            } else {
                                VisualAdmission::recoverable_failure(
                                    pending.complete(VisualAttemptOutcome::RecoverableFailure),
                                )
                            }
                            .map_err(|source| {
                                LiveThreadError::VisualAdmissionBuild {
                                    source: LiveVisualAdmissionBuildError::Admission(source),
                                }
                            })?;
                            route_visual_admission(sender, admission).map_err(|source| {
                                LiveThreadError::VisualAdmissionRoute { source }
                            })?;
                        }
                        if dense_active {
                            let pose_updates = tracker.take_pending_dense_pose_updates();
                            let pose_update_command =
                                match command_mapper::apply_pose_updates_command(
                                    pose_updates,
                                    timestamp,
                                    &mut dense_generation,
                                ) {
                                    Ok(command) => command,
                                    Err(generation) if requires_pipeline_shutdown => {
                                        return Err(LiveThreadError::DenseCommandGenerationAndInferenceUnavailable {
                                            generation,
                                            inference: err,
                                        });
                                    }
                                    Err(generation) => return Err(generation.into()),
                                };
                            if let Some(pose_update_command) = pose_update_command
                                && let Some(ref tx) = dense_command_tx
                            {
                                let route = classify_live_dense_route(
                                    tx.route(pose_update_command),
                                    LiveDenseCommandClass::OrderedControl,
                                    LiveDenseRouteContext::PoseUpdateAfterTrackerError,
                                );
                                let disposition = match route {
                                    Ok(disposition) => disposition,
                                    Err(routing) if requires_pipeline_shutdown => {
                                        return Err(
                                            LiveThreadError::DenseCommandRouteAndInferenceUnavailable {
                                                routing,
                                                inference: err,
                                            },
                                        );
                                    }
                                    Err(routing) => return Err(routing.into()),
                                };
                                match disposition {
                                    LiveDenseRouteDisposition::Enqueued => {}
                                    LiveDenseRouteDisposition::Disconnected => {
                                        if navigation_enabled {
                                            let reason = "dense command consumer disconnected after tracker failure";
                                            return if requires_pipeline_shutdown {
                                                Err(LiveThreadError::RequiredDenseAndInferenceUnavailable {
                                                    reason,
                                                    inference: err,
                                                })
                                            } else {
                                                Err(LiveThreadError::RequiredDenseUnavailable {
                                                    reason,
                                                })
                                            };
                                        }
                                        dense_active = false;
                                        dense_command_tx = None;
                                        eprintln!(
                                            "dense ordered command queue disconnected after tracker error; disabling dense"
                                        );
                                    }
                                    LiveDenseRouteDisposition::IntegrationDroppedNewest => {
                                        unreachable!(
                                            "ordered controls cannot be reported as integration data"
                                        )
                                    }
                                }
                            }
                        }
                        if requires_pipeline_shutdown {
                            return Err(LiveThreadError::InferenceUnavailable { source: err });
                        }
                        eprintln!("tracker error: {err}");
                    }
                    Err(payload) => {
                        if let (Some(sender), Some(pending)) =
                            (navigation_visual_tx.as_ref(), pending_visual)
                        {
                            let admission = VisualAdmission::fatal_failure(
                                pending.complete(VisualAttemptOutcome::FatalFailure),
                            )
                            .map_err(|source| {
                                LiveThreadError::VisualAdmissionBuild {
                                    source: LiveVisualAdmissionBuildError::Admission(source),
                                }
                            })?;
                            route_visual_admission(sender, admission).map_err(|source| {
                                LiveThreadError::VisualAdmissionRoute { source }
                            })?;
                        }
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
        let rerun_finish_timeout = args.rerun_finish_timeout_ms.get();
        let viz_handle = thread::spawn(move || -> Result<(), LiveThreadError> {
            let mut initialization_error = None;
            let mut navigation_recording = None;
            let mut sink = match rerun::RecordingStreamBuilder::new("kiko-slam-live").connect_grpc()
            {
                Ok(rec) => {
                    navigation_recording = Some(rec.clone());
                    match RerunSink::new(rec, decimation) {
                        Ok(sink) => Some(sink),
                        Err(source) => {
                            eprintln!("invalid live Rerun configuration: {source}");
                            initialization_error =
                                Some(LiveThreadError::VisualizationConfiguration { source });
                            None
                        }
                    }
                }
                Err(err) => {
                    eprintln!("failed to connect to rerun viewer: {err}");
                    initialization_error = Some(LiveThreadError::RerunConnect { source: err });
                    None
                }
            };
            let mut logging_error = None;
            let never_frames = crossbeam_channel::never::<LiveVizMsg>();
            let never_occupancy = crossbeam_channel::never::<TimedOccupancySnapshot>();
            let never_navigation = crossbeam_channel::never::<LiveNavigationVizMsg>();
            let mut frame_rx = Some(viz_rx);
            let mut map_rx = occupancy_snapshot_rx;
            let mut navigation_rx = navigation_viz_rx;
            let mut navigation_context_logged = false;
            if initialization_error.is_some() {
                // Stop upstream visualization work immediately. Tracking and
                // occupancy remain authoritative and shut down through their own
                // channels; this worker still reports the typed initialization
                // failure after any applicable Rerun finalization.
                frame_rx = None;
                map_rx = None;
                navigation_rx = None;
            }
            while frame_rx.is_some() || map_rx.is_some() || navigation_rx.is_some() {
                let mut close_frames = false;
                let mut close_maps = false;
                let mut close_navigation = false;
                {
                    let frame_receiver = frame_rx
                        .as_ref()
                        .map_or(&never_frames, kiko_slam::DropReceiver::as_receiver);
                    let map_receiver = map_rx
                        .as_ref()
                        .map_or(&never_occupancy, kiko_slam::DropReceiver::as_receiver);
                    let navigation_receiver = navigation_rx
                        .as_ref()
                        .map_or(&never_navigation, kiko_slam::DropReceiver::as_receiver);
                    crossbeam_channel::select! {
                        recv(frame_receiver) -> message => match message {
                            Ok(message) => {
                                if logging_error.is_none()
                                    && let Some(sink) = sink.as_mut()
                                    && let Err(error) = log_live_viz_message(sink, message)
                                {
                                    eprintln!(
                                        "live Rerun logging failed; disconnecting visualization producers: {error}"
                                    );
                                    logging_error = Some(error);
                                    close_frames = true;
                                    close_maps = true;
                                    close_navigation = true;
                                }
                            }
                            Err(_) => close_frames = true,
                        },
                        recv(map_receiver) -> snapshot => match snapshot {
                            Ok(snapshot) => {
                                if logging_error.is_none()
                                    && let Some(sink) = sink.as_mut()
                                {
                                    let (timestamp, snapshot) = snapshot.into_parts();
                                    if let Err(error) = sink.log_occupancy(timestamp, snapshot) {
                                        eprintln!(
                                            "live Rerun occupancy logging failed; disconnecting visualization producers: {error}"
                                        );
                                        logging_error = Some(error);
                                        close_frames = true;
                                        close_maps = true;
                                        close_navigation = true;
                                    }
                                }
                            }
                            Err(_) => close_maps = true,
                        },
                        recv(navigation_receiver) -> message => match message {
                            Ok(message) => {
                                if logging_error.is_none()
                                    && let Some(recording) = navigation_recording.as_ref()
                                    && let Err(error) = log_live_navigation_viz_message(
                                        recording,
                                        message,
                                        &mut navigation_context_logged,
                                    )
                                {
                                    eprintln!(
                                        "live Rerun navigation logging failed; disconnecting visualization producers: {error}"
                                    );
                                    logging_error = Some(error);
                                    close_frames = true;
                                    close_maps = true;
                                    close_navigation = true;
                                }
                            }
                            Err(_) => close_navigation = true,
                        },
                    }
                }
                if close_frames {
                    frame_rx = None;
                }
                if close_maps {
                    map_rx = None;
                }
                if close_navigation {
                    navigation_rx = None;
                }
            }
            drop(navigation_recording);
            let finalization_error = sink
                .map(|sink| sink.finish_with_timeout(rerun_finish_timeout))
                .and_then(Result::err);
            match (initialization_error, logging_error, finalization_error) {
                (Some(error), _, _) => Err(error),
                (None, Some(logging), Some(finalization)) => {
                    Err(LiveThreadError::VisualizationLogAndFinalization {
                        logging,
                        finalization,
                    })
                }
                (None, Some(source), None) => Err(LiveThreadError::VisualizationLog { source }),
                (None, None, Some(source)) => {
                    Err(LiveThreadError::VisualizationFinalization { source })
                }
                (None, None, None) => Ok(()),
            }
        });

        let mut capture_error = None;

        eprintln!("streaming matches... press ctrl+c to stop");

        let capture_exit_guard = LiveThreadExitGuard(Arc::clone(&running));
        'capture: while running.load(Ordering::Relaxed) {
            let mut got_any = false;

            match device.mono_left(0) {
                Ok(frame) => match oak_to_frame(frame, SensorId::StereoLeft) {
                    Ok(frame) => {
                        if let Err(source) = pairer.push_left(frame) {
                            capture_error = Some(LiveCaptureError::PairingInput { source });
                            break 'capture;
                        }
                        got_any = true;
                    }
                    Err(source) => {
                        capture_error = Some(LiveCaptureError::LeftFrame { source });
                        break 'capture;
                    }
                },
                Err(ImageError::Timeout { .. } | ImageError::QueueEmpty) => {}
                Err(source) => {
                    capture_error = Some(LiveCaptureError::LeftImage { source });
                    break 'capture;
                }
            }

            match device.mono_right(0) {
                Ok(frame) => match oak_to_frame(frame, SensorId::StereoRight) {
                    Ok(frame) => {
                        if let Err(source) = pairer.push_right(frame) {
                            capture_error = Some(LiveCaptureError::PairingInput { source });
                            break 'capture;
                        }
                        got_any = true;
                    }
                    Err(source) => {
                        capture_error = Some(LiveCaptureError::RightFrame { source });
                        break 'capture;
                    }
                },
                Err(ImageError::Timeout { .. } | ImageError::QueueEmpty) => {}
                Err(source) => {
                    capture_error = Some(LiveCaptureError::RightImage { source });
                    break 'capture;
                }
            }

            if depth_enabled {
                match device.depth(0) {
                    Ok(depth_frame) => {
                        let host_arrival = match host_monotonic_since(capture_clock_origin) {
                            Ok(timestamp) => timestamp,
                            Err(source) => {
                                capture_error = Some(LiveCaptureError::HostTimestamp { source });
                                break 'capture;
                            }
                        };
                        match parse_rectified_left_depth(depth_frame, rectified_left_intrinsics) {
                            Ok(depth_image) => {
                                let observation = match DepthObservation::parse(
                                    device_session,
                                    host_arrival,
                                    depth_image,
                                ) {
                                    Ok(observation) => observation,
                                    Err(source) => {
                                        capture_error =
                                            Some(LiveCaptureError::DepthObservation { source });
                                        break 'capture;
                                    }
                                };
                                if let Some(writer) = navigation_dataset_writer.as_ref()
                                    && let Err(source) = require_record_write(
                                        writer.write_depth(observation.depth()),
                                        RecordItem::DepthFrame,
                                    )
                                {
                                    capture_error = Some(LiveCaptureError::DatasetWrite { source });
                                    break 'capture;
                                }
                                got_any = true;
                                if let Some(depth_tx) = depth_tx.as_ref()
                                    && matches!(
                                        depth_tx.route(observation).slam,
                                        SendOutcome::Disconnected
                                    )
                                {
                                    break;
                                }
                            }
                            Err(source) => {
                                capture_error = Some(LiveCaptureError::DepthFrame { source });
                                break 'capture;
                            }
                        }
                    }
                    Err(DepthError::Timeout { .. } | DepthError::QueueEmpty) => {}
                    Err(source) => {
                        capture_error = Some(LiveCaptureError::Depth { source });
                        break 'capture;
                    }
                }
            }

            if let (Some(session_id), Some(imu_tx)) = (imu_session, imu_tx.as_mut()) {
                match device.imu() {
                    Ok(samples) => {
                        let host_arrival = match host_monotonic_since(capture_clock_origin) {
                            Ok(timestamp) => timestamp,
                            Err(source) => {
                                capture_error = Some(LiveCaptureError::HostTimestamp { source });
                                break 'capture;
                            }
                        };
                        for sample in samples {
                            let report = match oak_to_imu_report(sample, session_id, host_arrival) {
                                Ok(report) => report,
                                Err(source) => {
                                    capture_error = Some(LiveCaptureError::ImuSample { source });
                                    break 'capture;
                                }
                            };
                            if let Some(writer) = navigation_dataset_writer.as_ref()
                                && let Err(source) = require_record_write(
                                    writer.write_imu(report),
                                    RecordItem::ImuReport,
                                )
                            {
                                capture_error = Some(LiveCaptureError::DatasetWrite { source });
                                break 'capture;
                            }
                            let outcome = match imu_tx.route(report) {
                                Ok(outcome) => outcome,
                                Err(source) => {
                                    capture_error = Some(LiveCaptureError::ImuOrdering { source });
                                    break 'capture;
                                }
                            };
                            if matches!(outcome.delivery, SendOutcome::Disconnected) {
                                capture_error = Some(LiveCaptureError::ImuRouteDisconnected);
                                break 'capture;
                            }
                        }
                        got_any = true;
                    }
                    Err(ImuError::Empty) => {}
                    Err(source) => {
                        capture_error = Some(LiveCaptureError::Imu { source });
                        break 'capture;
                    }
                }
            }

            while let Some(pair) = pairer.next_pair() {
                let host_arrival = match host_monotonic_since(capture_clock_origin) {
                    Ok(timestamp) => timestamp,
                    Err(source) => {
                        capture_error = Some(LiveCaptureError::HostTimestamp { source });
                        break 'capture;
                    }
                };
                let observation = match StereoObservation::parse(device_session, host_arrival, pair)
                {
                    Ok(observation) => observation,
                    Err(source) => {
                        capture_error = Some(LiveCaptureError::StereoObservation { source });
                        break 'capture;
                    }
                };
                if let Some(writer) = navigation_dataset_writer.as_ref()
                    && let Err(source) = require_record_write(
                        writer.write_pair(observation.pair().clone()),
                        RecordItem::StereoPair,
                    )
                {
                    capture_error = Some(LiveCaptureError::DatasetWrite { source });
                    break 'capture;
                }
                if matches!(pair_tx.try_send(observation), SendOutcome::Disconnected) {
                    running.store(false, Ordering::SeqCst);
                    break 'capture;
                }
            }

            if !got_any {
                thread::sleep(Duration::from_micros(500));
            }
        }

        // Dropping the capture exit guard propagates every normal/error exit
        // before the remaining bounded queues are joined and drained. It also
        // covers unwinding through this scope.
        drop(capture_exit_guard);
        drop(pair_tx);
        drop(depth_tx);
        drop(imu_tx);
        drop(navigation_dataset_writer);
        let mut live_failures = capture_error
            .into_iter()
            .map(LiveWorkerFailure::Capture)
            .collect::<Vec<_>>();
        match inference_handle.join() {
            Ok(Ok(())) => {}
            Ok(Err(error)) => live_failures.push(LiveWorkerFailure::Inference(error)),
            Err(payload) => live_failures.push(LiveWorkerFailure::InferencePanic {
                detail: kiko_slam::panic_payload_to_string(payload.as_ref()),
            }),
        }

        // Drain the causal chain in ownership order. Inference closes the dense
        // command stream; dense publishes its last dirty map; navigation admits
        // that map and finalizes its journal; visualization then drains only
        // diagnostic copies.
        if let Some(handle) = dense_handle {
            match handle.join() {
                Ok(Ok(())) => {}
                Ok(Err(error)) => live_failures.push(LiveWorkerFailure::Occupancy(error)),
                Err(payload) => live_failures.push(LiveWorkerFailure::OccupancyPanic {
                    detail: kiko_slam::panic_payload_to_string(payload.as_ref()),
                }),
            }
        }

        let mut navigation_descriptor = None;
        if let Some(handle) = navigation_handle {
            match handle.join() {
                Ok(Ok(success)) => navigation_descriptor = Some(success.descriptor),
                Ok(Err(error)) => live_failures.push(LiveWorkerFailure::Navigation(error)),
                Err(payload) => live_failures.push(LiveWorkerFailure::NavigationPanic {
                    detail: kiko_slam::panic_payload_to_string(payload.as_ref()),
                }),
            }
        }
        // Rerun is non-authoritative: its failure is reported, but cannot change
        // whether an otherwise complete payload+journal recording is publishable.
        let authoritative_failure = !live_failures.is_empty();

        let visualization_failure = match viz_handle.join() {
            Ok(Ok(())) => false,
            Ok(Err(error)) => {
                live_failures.push(LiveWorkerFailure::Visualization(error));
                true
            }
            Err(payload) => {
                live_failures.push(LiveWorkerFailure::VisualizationPanic {
                    detail: kiko_slam::panic_payload_to_string(payload.as_ref()),
                });
                true
            }
        };
        let navigation_dataset_publishable = navigation_dataset_may_publish(
            authoritative_failure,
            navigation_descriptor.is_some(),
            visualization_failure,
        );

        if let Some(handle) = navigation_dataset_handle {
            if navigation_dataset_publishable {
                let descriptor = navigation_descriptor
                    .expect("publishable navigation dataset has a finalized descriptor");
                if let Err(source) = handle.finish_with_navigation_ingress(descriptor) {
                    live_failures.push(LiveWorkerFailure::DatasetFinalization(source));
                }
            } else if let Err(source) = handle.abort_without_manifest() {
                live_failures.push(LiveWorkerFailure::DatasetAbort(source));
            }
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
                "depth SLAM queue stats: enqueued={}, dropped_oldest={}, dropped_newest={}, disconnected={}",
                depth_snapshot.slam.enqueued,
                depth_snapshot.slam.dropped_oldest,
                depth_snapshot.slam.dropped_newest,
                depth_snapshot.slam.disconnected
            );
            eprintln!(
                "depth navigation queue stats: enqueued={}, dropped_oldest={}, dropped_newest={}, disconnected={}",
                depth_snapshot.navigation.enqueued,
                depth_snapshot.navigation.dropped_oldest,
                depth_snapshot.navigation.dropped_newest,
                depth_snapshot.navigation.disconnected
            );
        }
        if let Some(imu_stats_handle) = imu_stats_handle {
            let imu_snapshot = imu_stats_handle.snapshot();
            eprintln!(
                "IMU route stats: enqueued={}, dropped_oldest={}, dropped_newest={}, disconnected={}, source_gap_events={}, source_missing_reports={}, ordering_rejected={}",
                imu_snapshot.reports.enqueued,
                imu_snapshot.reports.dropped_oldest,
                imu_snapshot.reports.dropped_newest,
                imu_snapshot.reports.disconnected,
                imu_snapshot.source_gap_events,
                imu_snapshot.source_missing_reports,
                imu_snapshot.ordering_rejected
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
        if let Some(snapshot_stats_handle) = occupancy_snapshot_stats_handle {
            let snapshot = snapshot_stats_handle.snapshot();
            eprintln!(
                "occupancy snapshot queue stats: enqueued={}, dropped_oldest={}, dropped_newest={}, disconnected={}",
                snapshot.enqueued,
                snapshot.dropped_oldest,
                snapshot.dropped_newest,
                snapshot.disconnected
            );
        }
        if let Some(stats_handle) = occupancy_viz_forward_stats_handle {
            let snapshot = stats_handle.snapshot();
            eprintln!(
                "occupancy visualization forward queue stats: enqueued={}, dropped_oldest={}, dropped_newest={}, disconnected={}",
                snapshot.enqueued,
                snapshot.dropped_oldest,
                snapshot.dropped_newest,
                snapshot.disconnected
            );
        }
        if let Some(stats_handle) = navigation_viz_stats_handle {
            let snapshot = stats_handle.snapshot();
            eprintln!(
                "navigation visualization queue stats: enqueued={}, dropped_oldest={}, dropped_newest={}, disconnected={}",
                snapshot.enqueued,
                snapshot.dropped_oldest,
                snapshot.dropped_newest,
                snapshot.disconnected
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

        if !live_failures.is_empty() {
            return Err(LiveRunError {
                failures: live_failures,
            }
            .into());
        }

        Ok(())
    })();
    finish_live_device_session(operation, device.close())
}

#[cfg(feature = "record")]
fn build_meta(
    config: &MonoConfig,
    depth_config: Option<&DepthConfig>,
    imu_config: Option<&ImuConfig>,
    oak_provenance: &OakRuntimeProvenance,
) -> Meta {
    Meta {
        created: chrono::Utc::now().to_rfc3339(),
        device: oak_provenance.dataset_device_label(),
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
            optical_frame: Some(match c.alignment {
                DepthAlignment::RectifiedLeft => DepthOpticalFrame::RectifiedLeft,
                DepthAlignment::RectifiedRight => DepthOpticalFrame::RectifiedRight,
                DepthAlignment::Rgb => DepthOpticalFrame::Rgb,
            }),
        }),
        imu: imu_config.map(|c| ImuMeta { rate_hz: c.rate_hz }),
    }
}

#[cfg(feature = "record")]
fn build_calibration(
    left: OakIntrinsics,
    right: OakIntrinsics,
    baseline_m: f32,
    rectified: bool,
) -> Calibration {
    Calibration {
        left: CameraIntrinsics {
            fx: left.fx(),
            fy: left.fy(),
            cx: left.cx(),
            cy: left.cy(),
            width: left.width(),
            height: left.height(),
        },
        right: CameraIntrinsics {
            fx: right.fx(),
            fy: right.fy(),
            cx: right.cx(),
            cy: right.cy(),
            width: right.width(),
            height: right.height(),
        },
        baseline_m,
        rectified,
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
}

#[derive(Debug)]
struct RemovedBaMotionPriorSetting;

impl std::fmt::Display for RemovedBaMotionPriorSetting {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "KIKO_BA_MOTION_WEIGHT is no longer supported because its absolute pose-parameter penalty was not a frame-invariant SE(3) objective; remove the environment setting"
        )
    }
}

impl std::error::Error for RemovedBaMotionPriorSetting {}

fn reject_removed_ba_motion_prior(
    value: Option<String>,
) -> Result<(), RemovedBaMotionPriorSetting> {
    match value {
        Some(_) => Err(RemovedBaMotionPriorSetting),
        None => Ok(()),
    }
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
    )?)
}

fn build_ba_config() -> Result<LocalBaConfig, Box<dyn std::error::Error>> {
    reject_removed_ba_motion_prior(env_string("KIKO_BA_MOTION_WEIGHT")?)?;
    let config = build_ba_config_from_values(BaConfigValues {
        window: env_usize("KIKO_BA_WINDOW")?.unwrap_or(DEFAULT_BA_WINDOW),
        iterations: env_usize("KIKO_BA_ITERS")?.unwrap_or(DEFAULT_BA_ITERS),
        min_observations: env_usize("KIKO_BA_MIN_OBS")?.unwrap_or(DEFAULT_BA_MIN_OBS),
        huber_delta_px: env_f32("KIKO_BA_HUBER_PX")?.unwrap_or(DEFAULT_BA_HUBER_PX),
        initial_lambda: env_f32("KIKO_BA_DAMPING")?.unwrap_or(DEFAULT_BA_DAMPING),
        lambda_factor: env_f32("KIKO_LM_FACTOR")?.unwrap_or(DEFAULT_LM_FACTOR),
        min_lambda: env_f32("KIKO_LM_MIN")?.unwrap_or(DEFAULT_LM_MIN),
        max_lambda: env_f32("KIKO_LM_MAX")?.unwrap_or(DEFAULT_LM_MAX),
    })?;
    eprintln!(
        "local BA: window={} iters={} min_obs={} huber_px={} lm_init={} lm_factor={} lm_min={} lm_max={}",
        config.window(),
        config.max_iterations(),
        config.min_observations(),
        config.huber_delta_px(),
        config.lm().initial_lambda(),
        config.lm().lambda_factor(),
        config.lm().min_lambda(),
        config.lm().max_lambda()
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
        BaConfigValues, BenchError, Cli, Command, DepthRingCapacity, LiveActuationRequest,
        LiveDecisionVizKind, LiveDenseCommandClass, LiveDenseRouteContext,
        LiveDenseRouteDisposition, LiveDenseRouteError, LiveLosslessRouteError,
        LiveNavigationBoundaryError, LiveNavigationRequest, LiveThreadExitGuard, LiveVisualShape,
        OccupancyProjectionContractError, OdometryVizProcessingError, OfflineDepthSelector,
        OfflineFatalDenseError, OfflineFatalTrackerError, RerunDestination, RerunDestinationError,
        RerunFinishTimeout, RerunSessionError, build_ba_config_from_values,
        classify_live_dense_route, classify_live_visual_shape, classify_lossless_send,
        combine_rerun_results, live_decision_viz_status, navigation_dataset_may_publish,
        occupancy_depth_camera, reject_removed_ba_motion_prior, require_level_optical_world,
        take_deferred_offline_snapshot_error,
    };
    use clap::{Parser as _, error::ErrorKind};
    use kiko_slam::dataset::{DatasetError, DepthOpticalFrame, DepthProjectionContract};
    use kiko_slam::dense::{occupancy::OccupancyError, occupancy_runtime::OccupancyRuntimeError};
    use kiko_slam::{
        DenseCommandSendOutcome, DepthImage, FrameDimensions, FrameId, InferenceError,
        PinholeIntrinsics, PipelineError, PipelineTimingError, Timestamp, TrackerError,
        VizFlushError, VizLogError,
    };
    use std::collections::VecDeque;
    use std::num::NonZeroU16;
    use std::path::{Path, PathBuf};
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };
    use std::time::Duration;

    #[test]
    fn live_navigation_clock_rejects_elapsed_time_outside_u64_without_timestamp_fallback() {
        let elapsed_nanoseconds = 18_446_744_073_709_551_616_u128;
        let source = super::host_monotonic_from_elapsed_nanos(elapsed_nanoseconds)
            .expect_err("u64 navigation timebase must reject the next value");
        let error = super::navigation_clock_read_error(source);

        assert!(matches!(
            error,
            kiko_slam::navigation::mpc::HostMonotonicClockReadError::ElapsedNanosecondsOutOfRange {
                elapsed_nanoseconds: actual,
            } if actual == elapsed_nanoseconds
        ));
    }

    #[cfg(feature = "record")]
    use super::{
        DeviceCloseFailure, LiveThreadError, OakMxidArg, OakRuntimeProvenance, RecordCaptureError,
        RecordError, RecordItem, RectifiedLeftDepthError, build_calibration, compose_record_errors,
        finite_rate_per_second, record_device_close_error, require_rectified_left_depth_contract,
        require_rectified_left_depth_projection,
    };
    #[cfg(feature = "record")]
    use oak_sys::{DepthAlignment, Intrinsics as OakIntrinsics};
    #[cfg(feature = "record")]
    fn oak_intrinsics(fx: f32, width: u32, height: u32) -> OakIntrinsics {
        OakIntrinsics::try_from_projection_matrix(
            [
                [fx, 0.0, width as f32 * 0.5],
                [0.0, fx + 1.0, height as f32 * 0.5],
                [0.0, 0.0, 1.0],
            ],
            width,
            height,
        )
        .expect("valid test projection")
    }

    #[test]
    fn live_navigation_boundary_disables_only_when_all_options_are_absent() {
        let request =
            LiveNavigationRequest::parse(None, None, None, None, None).expect("all options absent");
        assert_eq!(request, LiveNavigationRequest::Disabled);
        assert!(!request.is_enabled());
    }

    #[test]
    fn live_navigation_boundary_preserves_a_complete_typed_request() {
        let config_path = PathBuf::from("navigation.json");
        let goal = "1.25,-2.5"
            .parse::<kiko_slam::navigation::NavigationGoalArg>()
            .expect("finite map-frame goal");
        let dataset_path = PathBuf::from("capture");

        let request = LiveNavigationRequest::parse(
            Some(config_path.clone()),
            Some(goal),
            Some(dataset_path.clone()),
            None,
            None,
        )
        .expect("complete navigation request");
        assert!(request.is_enabled());
        assert_eq!(
            request,
            LiveNavigationRequest::Enabled {
                config_path,
                goal: Some(goal),
                dataset_path,
                actuation: LiveActuationRequest::ShadowOnly,
            }
        );
    }

    #[test]
    fn live_navigation_boundary_accepts_safe_mapping_without_an_initial_goal() {
        let request = LiveNavigationRequest::parse(
            Some(PathBuf::from("navigation.json")),
            None,
            Some(PathBuf::from("capture")),
            None,
            None,
        )
        .expect("mapping-only navigation request");
        assert_eq!(
            request,
            LiveNavigationRequest::Enabled {
                config_path: PathBuf::from("navigation.json"),
                goal: None,
                dataset_path: PathBuf::from("capture"),
                actuation: LiveActuationRequest::ShadowOnly,
            }
        );
    }

    #[test]
    fn live_actuation_requires_an_explicit_paired_manifest_and_robot_id() {
        let goal = "1.25,-2.5"
            .parse::<kiko_slam::navigation::NavigationGoalArg>()
            .expect("finite map-frame goal");
        let request = LiveNavigationRequest::parse(
            Some(PathBuf::from("navigation.json")),
            Some(goal),
            Some(PathBuf::from("capture")),
            Some(PathBuf::from("actuation.json")),
            Some("kiko-01".to_string()),
        )
        .expect("complete physical request");
        assert_eq!(
            request,
            LiveNavigationRequest::Enabled {
                config_path: PathBuf::from("navigation.json"),
                goal: Some(goal),
                dataset_path: PathBuf::from("capture"),
                actuation: LiveActuationRequest::Physical {
                    config_path: PathBuf::from("actuation.json"),
                    exact_robot_id: "kiko-01".to_string(),
                },
            }
        );

        for (config, robot, expected) in [
            (
                Some(PathBuf::from("actuation.json")),
                None,
                LiveNavigationBoundaryError::IncompleteActuation {
                    missing_config: false,
                    missing_robot_id: true,
                },
            ),
            (
                None,
                Some("kiko-01".to_string()),
                LiveNavigationBoundaryError::IncompleteActuation {
                    missing_config: true,
                    missing_robot_id: false,
                },
            ),
        ] {
            assert_eq!(
                LiveNavigationRequest::parse(
                    Some(PathBuf::from("navigation.json")),
                    Some(goal),
                    Some(PathBuf::from("capture")),
                    config,
                    robot,
                )
                .expect_err("partial actuation authority must reject"),
                expected
            );
        }

        assert_eq!(
            LiveNavigationRequest::parse(
                None,
                None,
                None,
                Some(PathBuf::from("actuation.json")),
                Some("kiko-01".to_string()),
            )
            .expect_err("actuation without navigation must reject"),
            LiveNavigationBoundaryError::ActuationRequiresNavigation
        );
    }

    #[test]
    fn live_navigation_boundary_reports_each_partial_option_set_exactly() {
        let goal = "1,2"
            .parse::<kiko_slam::navigation::NavigationGoalArg>()
            .expect("finite map-frame goal");
        let cases = [
            (
                0b001,
                LiveNavigationBoundaryError::IncompleteNavigation {
                    missing_config: false,
                    missing_dataset: true,
                },
                " --navigation-record",
            ),
            (
                0b010,
                LiveNavigationBoundaryError::IncompleteNavigation {
                    missing_config: true,
                    missing_dataset: true,
                },
                " --navigation-config --navigation-record",
            ),
            (
                0b100,
                LiveNavigationBoundaryError::IncompleteNavigation {
                    missing_config: true,
                    missing_dataset: false,
                },
                " --navigation-config",
            ),
            (
                0b011,
                LiveNavigationBoundaryError::IncompleteNavigation {
                    missing_config: false,
                    missing_dataset: true,
                },
                " --navigation-record",
            ),
            (
                0b110,
                LiveNavigationBoundaryError::IncompleteNavigation {
                    missing_config: true,
                    missing_dataset: false,
                },
                " --navigation-config",
            ),
        ];

        for (present, expected, missing_flags) in cases {
            let actual = LiveNavigationRequest::parse(
                (present & 0b001 != 0).then(|| PathBuf::from("navigation.json")),
                (present & 0b010 != 0).then_some(goal),
                (present & 0b100 != 0).then(|| PathBuf::from("capture")),
                None,
                None,
            )
            .expect_err("every nonempty partial request must fail");
            assert_eq!(actual, expected, "present mask {present:#05b}");
            assert_eq!(
                actual.to_string(),
                format!(
                    "live navigation requires --navigation-config and --navigation-record together; --navigation-goal is optional; missing{missing_flags}"
                ),
                "present mask {present:#05b}"
            );
        }
    }

    #[test]
    fn live_visual_shape_classification_covers_every_optional_output_shape() {
        assert_eq!(
            classify_live_visual_shape(true, true),
            LiveVisualShape::IncrementAndLocalization
        );
        assert_eq!(
            classify_live_visual_shape(false, true),
            LiveVisualShape::LocalizationOnly
        );
        assert_eq!(
            classify_live_visual_shape(false, false),
            LiveVisualShape::NoLocalization
        );
        assert_eq!(
            classify_live_visual_shape(true, false),
            LiveVisualShape::NoLocalization,
            "an increment without map localization cannot update map-to-odom"
        );
    }

    #[test]
    fn lossless_live_route_preserves_success_timeout_and_disconnect() {
        assert_eq!(classify_lossless_send::<u8>(Ok(())), Ok(()));
        assert_eq!(
            classify_lossless_send(Err(crossbeam_channel::SendTimeoutError::Timeout(1_u8))),
            Err(LiveLosslessRouteError::TimedOut)
        );
        assert_eq!(
            classify_lossless_send(Err(crossbeam_channel::SendTimeoutError::Disconnected(1_u8))),
            Err(LiveLosslessRouteError::Disconnected)
        );
    }

    #[test]
    fn navigation_dataset_publication_ignores_only_visualization_failure() {
        assert!(navigation_dataset_may_publish(false, true, false));
        assert!(navigation_dataset_may_publish(false, true, true));
        assert!(!navigation_dataset_may_publish(true, true, false));
        assert!(!navigation_dataset_may_publish(true, true, true));
        assert!(!navigation_dataset_may_publish(false, false, false));
        assert!(!navigation_dataset_may_publish(false, false, true));
    }

    #[cfg(feature = "record")]
    #[test]
    fn delivered_depth_must_match_the_exact_rectified_left_projection() {
        let expected = oak_intrinsics(400.0, 640, 480);
        assert!(require_rectified_left_depth_projection(expected, expected).is_ok());

        let wrong_dimensions = oak_intrinsics(400.0, 320, 240);
        assert!(matches!(
            require_rectified_left_depth_projection(expected, wrong_dimensions),
            Err(RectifiedLeftDepthError::DimensionMismatch {
                expected_width: 640,
                expected_height: 480,
                actual_width: 320,
                actual_height: 240,
            })
        ));

        let wrong_projection = oak_intrinsics(401.0, 640, 480);
        assert!(matches!(
            require_rectified_left_depth_projection(expected, wrong_projection),
            Err(RectifiedLeftDepthError::ProjectionMismatch { .. })
        ));
    }

    #[cfg(feature = "record")]
    #[test]
    fn oak_mxid_boundary_rejects_empty_without_normalizing_exact_values() {
        assert!("".parse::<OakMxidArg>().is_err());
        assert!("   ".parse::<OakMxidArg>().is_err());
        let exact = "18443010C1A34AF500"
            .parse::<OakMxidArg>()
            .expect("nonempty exact MXID");
        assert_eq!(exact.as_str(), "18443010C1A34AF500");
    }

    #[cfg(feature = "record")]
    #[test]
    fn dataset_device_label_retains_runtime_identity_and_build_provenance() {
        let provenance = OakRuntimeProvenance {
            connected_mxid: "mxid-123".to_owned(),
            depthai_sdk_version: "3.6.1".to_owned(),
            depthai_sdk_commit: "commit-abc".to_owned(),
            embedded_device_artifact_version: "device-1".to_owned(),
            embedded_bootloader_artifact_version: "bootloader-1".to_owned(),
        };
        assert_eq!(
            provenance.dataset_device_label(),
            "OAK-D mxid=mxid-123 depthai_sdk=3.6.1 depthai_commit=commit-abc embedded_device=device-1 embedded_bootloader=bootloader-1 timestamp=device_exposure_midpoint"
        );
    }

    #[cfg(feature = "record")]
    #[test]
    fn matching_intrinsics_cannot_relabel_non_left_or_missing_depth() {
        let expected = oak_intrinsics(400.0, 640, 480);
        assert!(matches!(
            require_rectified_left_depth_contract(
                expected,
                expected,
                Some(DepthAlignment::RectifiedRight),
            ),
            Err(RectifiedLeftDepthError::UnexpectedConnectedAlignment {
                actual: DepthAlignment::RectifiedRight
            })
        ));
        assert!(matches!(
            require_rectified_left_depth_contract(expected, expected, Some(DepthAlignment::Rgb),),
            Err(RectifiedLeftDepthError::UnexpectedConnectedAlignment {
                actual: DepthAlignment::Rgb
            })
        ));
        assert!(matches!(
            require_rectified_left_depth_contract(expected, expected, None),
            Err(RectifiedLeftDepthError::MissingConnectedAlignment)
        ));
        assert!(matches!(
            require_rectified_left_depth_contract(
                expected,
                expected,
                Some(DepthAlignment::RectifiedLeft),
            ),
            Ok(())
        ));
    }

    #[cfg(feature = "record")]
    #[test]
    fn projection_failure_precedes_alignment_failure() {
        let expected = oak_intrinsics(400.0, 640, 480);
        let wrong_projection = oak_intrinsics(401.0, 640, 480);
        assert!(matches!(
            require_rectified_left_depth_contract(
                expected,
                wrong_projection,
                Some(DepthAlignment::Rgb),
            ),
            Err(RectifiedLeftDepthError::ProjectionMismatch { .. })
        ));
    }

    #[cfg(feature = "record")]
    #[test]
    fn calibration_is_derived_only_from_delivered_projection_types() {
        let left = oak_intrinsics(400.0, 640, 480);
        let right = oak_intrinsics(402.0, 640, 480);
        let calibration = build_calibration(left, right, 0.075, true);

        assert_eq!(calibration.left.fx, left.fx());
        assert_eq!(calibration.left.fy, left.fy());
        assert_eq!(calibration.left.cx, left.cx());
        assert_eq!(calibration.left.cy, left.cy());
        assert_eq!(
            (calibration.left.width, calibration.left.height),
            (left.width(), left.height())
        );
        assert_eq!(calibration.right.fx, right.fx());
        assert_eq!(calibration.baseline_m, 0.075);
        assert!(calibration.rectified);
    }

    #[cfg(feature = "record")]
    #[test]
    fn record_capture_and_close_composition_preserves_both_typed_failures() {
        let capture = RecordCaptureError::DatasetDropped {
            item: RecordItem::StereoPair,
        };
        let close = DeviceCloseFailure::new(std::io::Error::other("test close failure"));
        let combined = record_device_close_error(Some(capture), close);

        assert!(matches!(
            &combined,
            RecordCaptureError::CaptureAndDeviceClose {
                capture,
                close,
            } if matches!(capture.as_ref(), RecordCaptureError::DatasetDropped {
                item: RecordItem::StereoPair,
            }) && std::error::Error::source(close)
                .and_then(|source| source.downcast_ref::<std::io::Error>())
                .is_some()
        ));
        assert!(combined.to_string().contains("test close failure"));
        assert!(
            std::error::Error::source(&combined)
                .and_then(|source| source.downcast_ref::<RecordCaptureError>())
                .is_some()
        );
    }

    #[cfg(feature = "record")]
    #[test]
    fn record_error_composition_preserves_capture_close_and_finalization() {
        let capture = RecordCaptureError::DatasetDropped {
            item: RecordItem::DepthFrame,
        };
        let finalization = Box::new(DatasetError::InvalidManifest {
            reason: "test finalization failure",
        });
        let initial = compose_record_errors(Some(capture), Some(finalization))
            .expect("capture and finalization must be retained");
        let combined = initial.with_device_close(DeviceCloseFailure::new(std::io::Error::other(
            "test close failure",
        )));

        assert!(matches!(
            combined,
            RecordError::CaptureAndFinalization {
                capture: RecordCaptureError::CaptureAndDeviceClose {
                    capture,
                    close,
                },
                finalization,
            } if matches!(capture.as_ref(), RecordCaptureError::DatasetDropped {
                item: RecordItem::DepthFrame,
            }) && std::error::Error::source(&close)
                .and_then(|source| source.downcast_ref::<std::io::Error>())
                .is_some()
                && matches!(finalization.as_ref(), DatasetError::InvalidManifest {
                    reason: "test finalization failure",
                })
        ));
    }

    #[cfg(feature = "record")]
    #[test]
    fn record_summary_rate_is_finite_for_zero_or_invalid_elapsed_time() {
        assert_eq!(finite_rate_per_second(30, 2.0), 15.0);
        for elapsed in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let rate = finite_rate_per_second(30, elapsed);
            assert_eq!(rate, 0.0);
            assert!(rate.is_finite());
        }

        let overflow_rate = finite_rate_per_second(u64::MAX, f64::MIN_POSITIVE);
        assert_eq!(overflow_rate, 0.0);
        assert!(overflow_rate.is_finite());
    }

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
    fn odometry_error_preserves_dense_generation_source() {
        let mut generation =
            kiko_slam::dense::command_mapper::DenseCommandGeneration::from_current(u64::MAX);
        let source = kiko_slam::dense::command_mapper::apply_pose_updates_command(
            vec![kiko_slam::KeyframePoseUpdate::new(
                kiko_slam::map::KeyframeId::default(),
                kiko_slam::WorldToCamera::identity(),
            )],
            kiko_slam::Timestamp::from_nanos(0),
            &mut generation,
        )
        .expect_err("exhausted generation");
        let error = OdometryVizProcessingError::from(source);

        assert!(matches!(
            &error,
            OdometryVizProcessingError::DenseCommandGeneration(actual)
                if actual.current() == u64::MAX
        ));
        assert!(
            std::error::Error::source(&error)
                .and_then(|source| {
                    source.downcast_ref::<
                        kiko_slam::dense::command_mapper::DenseCommandGenerationError,
                    >()
                })
                .is_some_and(|source| source.current() == u64::MAX)
        );
    }

    #[test]
    fn odometry_error_preserves_fatal_tracker_source() {
        let error = OdometryVizProcessingError::from(TrackerError::Inference(
            InferenceError::SessionQuarantined {
                model: "test-offline-model",
            },
        ));

        let OdometryVizProcessingError::Tracker(failure) = &error else {
            panic!("typed tracker failure expected");
        };
        assert!(matches!(
            &failure.source,
            TrackerError::Inference(InferenceError::SessionQuarantined {
                model: "test-offline-model"
            })
        ));
        assert!(failure.dense_update.is_none());
        assert!(failure.publication.is_none());
        assert!(failure.occupancy_finalization.is_none());
        assert!(
            std::error::Error::source(&error)
                .and_then(|source| source.downcast_ref::<TrackerError>())
                .is_some_and(TrackerError::requires_pipeline_shutdown)
        );
    }

    #[test]
    fn fatal_odometry_error_preserves_publication_and_occupancy_failures() {
        let publication = VizLogError::TimestampUnrepresentable {
            timestamp_ns: 1,
            encoded_ns: 2,
        };
        let occupancy_finalization =
            OccupancyRuntimeError::Snapshot(OccupancyError::RevisionExhausted);
        let error = OdometryVizProcessingError::Tracker(Box::new(OfflineFatalTrackerError {
            source: TrackerError::Inference(InferenceError::WatchdogTimeout {
                model: "test-offline-model",
                timeout_ms: 5,
            }),
            dense_update: Some(OfflineFatalDenseError::Occupancy(
                OccupancyRuntimeError::Mapping(OccupancyError::RevisionExhausted),
            )),
            publication: Some(publication),
            occupancy_finalization: Some(occupancy_finalization),
        }));

        let OdometryVizProcessingError::Tracker(failure) = &error else {
            panic!("typed tracker failure expected");
        };
        assert!(matches!(
            &failure.source,
            TrackerError::Inference(InferenceError::WatchdogTimeout {
                model: "test-offline-model",
                timeout_ms: 5,
            })
        ));
        assert!(matches!(
            &failure.dense_update,
            Some(OfflineFatalDenseError::Occupancy(
                OccupancyRuntimeError::Mapping(OccupancyError::RevisionExhausted)
            ))
        ));
        assert!(matches!(
            &failure.publication,
            Some(VizLogError::TimestampUnrepresentable {
                timestamp_ns: 1,
                encoded_ns: 2,
            })
        ));
        assert!(matches!(
            &failure.occupancy_finalization,
            Some(OccupancyRuntimeError::Snapshot(
                OccupancyError::RevisionExhausted
            ))
        ));
        assert!(
            std::error::Error::source(&error)
                .and_then(|source| source.downcast_ref::<TrackerError>())
                .is_some_and(TrackerError::requires_pipeline_shutdown)
        );
    }

    #[test]
    fn live_dense_route_keeps_integration_drop_as_data_loss() {
        assert_eq!(
            classify_live_dense_route(
                DenseCommandSendOutcome::IntegrationDroppedNewest,
                LiveDenseCommandClass::IntegrationData,
                LiveDenseRouteContext::TrackerOutput,
            ),
            Ok(LiveDenseRouteDisposition::IntegrationDroppedNewest)
        );
    }

    #[test]
    fn live_dense_route_surfaces_control_timeout_in_every_context() {
        for context in [
            LiveDenseRouteContext::TrackerOutput,
            LiveDenseRouteContext::PoseUpdateAfterTrackerError,
        ] {
            assert_eq!(
                classify_live_dense_route(
                    DenseCommandSendOutcome::ControlTimedOut,
                    LiveDenseCommandClass::OrderedControl,
                    context,
                ),
                Err(LiveDenseRouteError::ControlTimedOut { context })
            );
        }
    }

    #[test]
    fn live_dense_route_rejects_control_reported_as_integration_drop() {
        let context = LiveDenseRouteContext::PoseUpdateAfterTrackerError;
        assert_eq!(
            classify_live_dense_route(
                DenseCommandSendOutcome::IntegrationDroppedNewest,
                LiveDenseCommandClass::OrderedControl,
                context,
            ),
            Err(LiveDenseRouteError::ControlMisclassifiedAsIntegration { context })
        );
    }

    #[cfg(feature = "record")]
    #[test]
    fn live_rerun_connect_error_preserves_typed_source() {
        let failure = LiveThreadError::RerunConnect {
            source: rerun::RecordingStreamError::NotAProxyEndpoint,
        };

        assert!(matches!(
            &failure,
            LiveThreadError::RerunConnect {
                source: rerun::RecordingStreamError::NotAProxyEndpoint,
            }
        ));
        assert!(
            std::error::Error::source(&failure)
                .and_then(|source| source.downcast_ref::<rerun::RecordingStreamError>())
                .is_some_and(|source| {
                    matches!(source, rerun::RecordingStreamError::NotAProxyEndpoint)
                })
        );
    }

    #[cfg(feature = "record")]
    #[test]
    fn live_failure_preserves_route_and_fatal_tracker_sources() {
        let routing = LiveDenseRouteError::ControlTimedOut {
            context: LiveDenseRouteContext::PoseUpdateAfterTrackerError,
        };
        let failure = LiveThreadError::DenseCommandRouteAndInferenceUnavailable {
            routing,
            inference: TrackerError::Inference(InferenceError::SessionQuarantined {
                model: "test-live-model",
            }),
        };

        assert!(matches!(
            &failure,
            LiveThreadError::DenseCommandRouteAndInferenceUnavailable {
                routing: actual_routing,
                inference: TrackerError::Inference(InferenceError::SessionQuarantined {
                    model: "test-live-model"
                }),
            } if *actual_routing == routing
        ));
        assert_eq!(
            std::error::Error::source(&failure)
                .and_then(|source| source.downcast_ref::<LiveDenseRouteError>()),
            Some(&routing)
        );
    }

    #[cfg(feature = "record")]
    #[test]
    fn live_failure_preserves_generation_and_fatal_tracker_sources() {
        let mut sequence =
            kiko_slam::dense::command_mapper::DenseCommandGeneration::from_current(u64::MAX);
        let generation = kiko_slam::dense::command_mapper::apply_pose_updates_command(
            vec![kiko_slam::KeyframePoseUpdate::new(
                kiko_slam::map::KeyframeId::default(),
                kiko_slam::WorldToCamera::identity(),
            )],
            Timestamp::from_nanos(1),
            &mut sequence,
        )
        .expect_err("exhausted generation");
        let failure = LiveThreadError::DenseCommandGenerationAndInferenceUnavailable {
            generation,
            inference: TrackerError::Inference(InferenceError::SessionQuarantined {
                model: "test-live-model",
            }),
        };

        assert!(matches!(
            &failure,
            LiveThreadError::DenseCommandGenerationAndInferenceUnavailable {
                generation,
                inference: TrackerError::Inference(InferenceError::SessionQuarantined {
                    model: "test-live-model"
                }),
            } if generation.current() == u64::MAX
        ));
        assert!(
            std::error::Error::source(&failure)
                .and_then(|source| {
                    source.downcast_ref::<
                        kiko_slam::dense::command_mapper::DenseCommandGenerationError,
                    >()
                })
                .is_some_and(|source| source.current() == u64::MAX)
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
    fn occupancy_requires_an_explicit_level_world_and_camera_height() {
        assert_eq!(
            require_level_optical_world(false, Some(0.5)),
            Err(OccupancyProjectionContractError::LevelOpticalWorldNotDeclared)
        );
        assert_eq!(
            require_level_optical_world(true, None),
            Err(OccupancyProjectionContractError::CameraHeightNotConfigured)
        );
        assert_eq!(require_level_optical_world(true, Some(0.5)), Ok(0.5));
    }

    #[test]
    fn occupancy_depth_projection_accepts_only_the_tracking_optical_frame_and_shape() {
        let tracking_dimensions = FrameDimensions::try_new(640, 480).expect("dimensions");
        let tracking_intrinsics =
            PinholeIntrinsics::try_new(500.0, 500.0, 319.5, 239.5).expect("intrinsics");
        let valid = occupancy_depth_camera(
            tracking_intrinsics,
            tracking_dimensions,
            DepthProjectionContract::new(tracking_dimensions, DepthOpticalFrame::RectifiedLeft),
            false,
        )
        .expect("rectified-left tracking projection");
        assert_eq!(valid.dimensions(), tracking_dimensions);

        assert!(matches!(
            occupancy_depth_camera(
                tracking_intrinsics,
                tracking_dimensions,
                DepthProjectionContract::new(
                    tracking_dimensions,
                    DepthOpticalFrame::RectifiedRight,
                ),
                false,
            ),
            Err(OccupancyProjectionContractError::UnsupportedOpticalFrame(
                DepthOpticalFrame::RectifiedRight
            ))
        ));
        assert!(matches!(
            occupancy_depth_camera(
                tracking_intrinsics,
                tracking_dimensions,
                DepthProjectionContract::new(
                    FrameDimensions::try_new(320, 240).expect("different dimensions"),
                    DepthOpticalFrame::RectifiedLeft,
                ),
                false,
            ),
            Err(OccupancyProjectionContractError::DepthCalibrationDimensionsMismatch { .. })
        ));
    }

    #[test]
    fn offline_deferred_snapshot_failure_preserves_its_typed_source() {
        let source = OccupancyError::AllocationFailed {
            context: "test offline snapshot",
            requested: 42,
        };
        let mut deferred = Some(source);

        assert!(matches!(
            take_deferred_offline_snapshot_error(&mut deferred),
            Err(OccupancyRuntimeError::Snapshot(error)) if error == source
        ));
        assert!(deferred.is_none());
        assert!(take_deferred_offline_snapshot_error(&mut deferred).is_ok());
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
        let window =
            i64::try_from(kiko_slam::dense::command_mapper::DEPTH_ASSOCIATION_WINDOW.as_nanos())
                .expect("test association window fits in i64");
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
        let window = kiko_slam::dense::command_mapper::DEPTH_ASSOCIATION_WINDOW.as_nanos();
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
        let max_delta = window;

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
    }

    #[test]
    fn removed_ba_motion_prior_setting_is_never_silently_ignored() {
        assert!(reject_removed_ba_motion_prior(None).is_ok());
        for value in ["0", "1", "not-a-number"] {
            assert!(reject_removed_ba_motion_prior(Some(value.to_owned())).is_err());
        }
    }

    #[cfg(feature = "record")]
    #[test]
    fn live_error_preserves_dense_generation_source() {
        let mut generation =
            kiko_slam::dense::command_mapper::DenseCommandGeneration::from_current(u64::MAX);
        let source = kiko_slam::dense::command_mapper::apply_pose_updates_command(
            vec![kiko_slam::KeyframePoseUpdate::new(
                kiko_slam::map::KeyframeId::default(),
                kiko_slam::WorldToCamera::identity(),
            )],
            kiko_slam::Timestamp::from_nanos(0),
            &mut generation,
        )
        .expect_err("exhausted generation");
        let error = LiveThreadError::from(source);

        assert!(matches!(
            &error,
            LiveThreadError::DenseCommandGeneration(source)
                if source.current() == u64::MAX
        ));
        assert!(
            std::error::Error::source(&error)
                .and_then(|source| {
                    source.downcast_ref::<
                        kiko_slam::dense::command_mapper::DenseCommandGenerationError,
                    >()
                })
                .is_some_and(|source| source.current() == u64::MAX)
        );
    }

    #[test]
    fn live_thread_exit_guard_stops_capture() {
        let running = Arc::new(AtomicBool::new(true));
        {
            let _guard = LiveThreadExitGuard(Arc::clone(&running));
            assert!(running.load(Ordering::SeqCst));
        }
        assert!(!running.load(Ordering::SeqCst));
    }

    #[test]
    fn live_decision_viz_keeps_stop_outcome_separate_from_applied_evidence() {
        assert_eq!(
            live_decision_viz_status(LiveDecisionVizKind::Controller, false),
            "controller_request"
        );
        assert_eq!(
            live_decision_viz_status(LiveDecisionVizKind::Controller, true),
            "controller_applied"
        );
        assert_eq!(
            live_decision_viz_status(LiveDecisionVizKind::Stopped, false),
            "fail_closed_stop"
        );
        assert_eq!(
            live_decision_viz_status(LiveDecisionVizKind::Stopped, true),
            "fail_closed_stop_applied"
        );
    }
}
