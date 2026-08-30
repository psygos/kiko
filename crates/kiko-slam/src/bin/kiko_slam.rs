use std::num::{NonZeroU16, NonZeroUsize};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[cfg(all(feature = "nano-agent", unix))]
#[path = "kiko_slam_systemd.rs"]
mod nano_systemd;

use clap::{Args, Parser, Subcommand, ValueEnum};

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
use std::io::{BufRead, IsTerminal, Write};
#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
use std::os::unix::fs::OpenOptionsExt;

#[cfg(all(feature = "nano-attended-navigation-trial", unix))]
use kiko_device_inventory::ArtifactRelativePath;
#[cfg(all(feature = "nano-agent", unix))]
use kiko_expression_core::{ChannelOrder, ImagePoint, StreamEpochId};
#[cfg(all(feature = "nano-agent", unix))]
use kiko_expression_runtime::{
    FaceDetection, FaceDetectorSource, FaceResultAdmission, FaceTargetState, FaceTrackingUpdate,
};
#[cfg(all(feature = "nano-agent", unix))]
use kiko_slam::TrackerRuntimePolicy;
use kiko_slam::dataset::{
    DatasetDepthCursor, DatasetError, DatasetReader, DepthOpticalFrame, DepthProjectionContract,
};
#[cfg(all(feature = "nano-agent", unix))]
use kiko_slam::dense::occupancy_persistence::OccupancyMapLimits;
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
#[cfg(all(feature = "nano-attended-navigation-trial", unix))]
use kiko_slam::navigation::nano_base_commissioning_bootstrap::{
    AttendedNavigationTrialMotionAdmission, CommissioningClockEpoch,
    OwnedNanoAttendedNavigationTrialController, prepare_nano_base_commissioning,
};
#[cfg(all(feature = "nano-attended-navigation-trial", unix))]
use kiko_slam::navigation::nano_base_commissioning_live::{
    PreparedAttendedNavigationTrialLiveHardware, prepare_attended_navigation_trial_live_hardware,
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
#[cfg(all(feature = "nano-agent", unix))]
use kiko_supervisor_core::ReadinessEpoch;

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
    ImuStreamMetadata, Meta, MonoMeta, OakCalibrationCameraSocket,
    OakEepromCalibrationEvidence as DatasetOakEepromCalibrationEvidence, PairedDatasetWriter,
    WriteOutcome,
};
#[cfg(all(feature = "nano-agent", unix))]
use kiko_slam::dataset::{DatasetStorageLimits, MAX_PRODUCTION_DATASET_MANIFEST_BYTES};
#[cfg(feature = "record")]
use kiko_slam::live_runtime::LiveNavigationRequest;
#[cfg(all(feature = "nano-agent", unix))]
use kiko_slam::live_runtime::prepare_live_navigation_runtime_from_parsed;
#[cfg(feature = "record")]
use kiko_slam::live_runtime::{
    LiveNavigationPrerequisites, PreparedLiveNavigationRuntime, PreparedLiveNavigationRuntimeParts,
    prepare_live_navigation_runtime,
};
#[cfg(feature = "record")]
use kiko_slam::navigation::NavigationGoalArg;
#[cfg(all(feature = "record", feature = "actuation"))]
use kiko_slam::navigation::actuation::LiveActuationError;
#[cfg(feature = "record")]
use kiko_slam::navigation::mpc::HostMonotonicClock;
#[cfg(any(feature = "record", test))]
use kiko_slam::navigation::mpc::HostMonotonicClockReadError;
#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    unix
))]
use kiko_slam::navigation::{
    AgentAuthoritySupervisor, AgentControlDispatchResponseError, AgentControlDispatcher,
    AgentControlMonotonicOrigin, AgentControlRejectionCodeV1, AgentControlSocketCleanupOutcome,
    AgentControlSocketTask, AgentControlSocketTaskExit, AgentControlSocketTaskJoinError,
    AgentControlSocketTaskStartError, AgentControllerStopKnowledge, AgentLiveActuationDisposition,
    AgentLocalizationStateV1, AgentManualControlCore, AgentManualRuntimePolicy, AgentMapStateV1,
    CoordinatorMotionModeV1, LiveLifecycleZeroApplied, LiveMotionActuationFaultEvidence,
    LiveMotionMapAdmissionError, LiveMotionOperationError, LiveMotionOwner, LiveMotionOwnerError,
    LiveMotionOwnerOutcome, LiveMotionTerminalStop, LivePhysicalStateEvent,
    ManifestBoundNanoAgentPolicyConfigV3, NanoAccessoryHealthObserver, NanoLiveModePolicy,
    NanoManualPlantBindingError, PreparedNanoProductionRuntime, PreparedNanoProductionRuntimeParts,
    VisualAdmissionOutcome, classify_live_actuation_error,
};
#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    feature = "nano-agent",
    feature = "operator-console",
    unix
))]
use kiko_slam::navigation::{
    AgentRuntimeStateV1, ConsoleActualAuthority, ConsoleActualAuthorityMode,
    ConsoleActualAuthoritySource, ConsoleAppliedReceipt, ConsoleCheckpointLocalizationEvidence,
    ConsoleFiniteF64Error, ConsoleGridProjectionError, ConsoleHealth, ConsoleHostTimestampNs,
    ConsoleInferenceRuntime, ConsoleInferenceSelection, ConsoleLocalization,
    ConsoleManualCommandEnvelope, ConsoleManualCommandEnvelopeError, ConsoleMapSnapshot,
    ConsoleNavigationSnapshot, ConsoleOccupancyGrid, ConsolePathError, ConsolePoint2, ConsolePose2,
    ConsoleReceiptProjectionError, ConsoleRequestedActuation, ConsoleRequestedInferenceBackend,
    ConsoleRerunDiagnosticsUrl, ConsoleRuntimeAuthorityKind, ConsoleSelectedInferenceBackend,
    ConsoleSlamRateWindow, ConsoleSlamSnapshot, ConsoleSnapshotRevision, ConsoleSourceKind,
    ConsoleStopCertainty, ConsoleSubsystemHealth, ConsoleTerminalReason, ConsoleTerminalState,
    LiveMotionAuthorityState, LiveMotionAuthorityStateError, NanoAccessoryComponentHealth,
    NanoAccessoryHealthStatusError, NanoAccessoryRuntimeHealth, NanoOperatorConsoleFrontend,
    NanoOperatorConsoleFrontendShutdownEvidence, NanoOperatorConsoleFrontendStartError,
    OperatorConsoleIngressDisposition, OperatorConsoleLimits, OperatorConsoleProcessDisposition,
    OperatorConsoleRetainedAuthorityKind, OperatorConsoleRuntimeAdapter,
    OperatorConsoleRuntimeAdapterError, OperatorConsoleRuntimeIngressError,
    OperatorConsoleSnapshot, OperatorConsoleSnapshotError, operator_console,
};
#[cfg(feature = "record")]
use kiko_slam::navigation::{
    ControlPeriodNs, CoordinatorAdmissionError, CoordinatorTickError, CoordinatorTickOutcome,
    NAVIGATION_INGRESS_STREAM_FILE, NavigationClockEpoch, NavigationIngressBoundaryError,
    NavigationIngressCapacity, NavigationIngressCapacityError, NavigationIngressEvent,
    NavigationIngressReader, NavigationIngressSidecarDescriptor, NavigationIngressStreamReadError,
    NavigationIngressStreamWriteError, NavigationIngressWriter, NavigationRecordingId,
    NavigationRecordingIdError, PendingVisualAttemptIngress, RecordedMapEpochId,
    SafetyDecisionOutcome, ShadowNavigationCoordinator, VisualAdmission, VisualAdmissionError,
    VisualAttemptOutcome,
};
#[cfg(all(feature = "nano-agent", unix))]
use kiko_slam::navigation::{
    HeadGazeFaceProposalOutcome, MAX_NANO_WARM_SELECTION_BYTES,
    NANO_ACCESSORY_TERMINAL_PUBLICATION_TIMEOUT, NanoAccessoryFaultWaitError,
    NanoAccessoryFrameSubmitOutcome, NanoAccessoryShutdownEvidence, NanoAccessoryTerminalFault,
    NanoAccessoryWorker, NanoAccessoryWorkerExit, NanoAccessoryWorkerJoinError,
    NanoBootstrapRequest, NanoBootstrapRoots, NanoBootstrapStereoEvidence,
    NanoDatasetReplayRequired, NanoFaceDiagnosticFrame, NanoFaceDiagnosticReceiver,
    NanoFaceDiagnosticStatsHandle, NanoFacePerceptionShutdownClass,
    NanoFacePerceptionShutdownEvidence, NanoFacePerceptionStageStatsHandle,
    NanoFaultRecoveryPresentationEvidence, NanoFaultRecoveryPresentationFault,
    NanoFinalizedJournalMapIdentity, NanoHeadGazeActuationAvailability, NanoHeadGazeDiagnostic,
    NanoLaunchInference, NanoLaunchOccupancy, NanoLaunchRerun, NanoLaunchStorage,
    NanoMapPersistenceConfig, NanoMapPersistenceOwner, NanoMapPersistencePathError,
    NanoMapSaveCommandError, NanoMapSnapshotRetentionError, NanoMapWarmStartLoad,
    NanoOakStreamGraph, NanoStateQuotaAdmissionError, NanoStateQuotaOwner,
    NanoWarmCheckpointCommandError, NanoWarmStartRelocalizationError,
    NanoWarmStartRelocalizationTransition, NanoWarmStartReplayConfig, ParsedNanoLiveConfiguration,
    PreparedNanoBootstrap, bootstrap_nano_production, replay_nano_warm_start,
};
#[cfg(all(feature = "record", feature = "actuation"))]
use kiko_slam::navigation::{
    LiveMpcControlDriver, LiveMpcControlError, NavigationActuationConfigV1,
};
#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
use kiko_slam::navigation::{
    NanoAccessoryHealthPeriod, NanoAccessoryWorkerConfig, NanoAccessoryWorkerStartError,
};
#[cfg(feature = "record")]
use kiko_slam::{
    DenseCommandQueueStatsHandle, DenseCommandReceiver, DenseCommandSender, DepthObservation,
    DepthObservationError, DeviceSessionId, DropPolicy, DropReceiver, DropSender,
    HostMonotonicTimestamp, ImuReport, InertialOrderingError, InertialValueError,
    PairingConfigError, PairingInputError, PairingWindowNs, SendOutcome, SensorId,
    StereoObservation, StereoObservationError, StereoPairer, TrackerOutput, VizConfigError,
    bounded_channel, dense_command_channel, depth_router, imu_report_router, oak_to_depth_image,
    oak_to_frame, oak_to_imu_report,
};
#[cfg(all(feature = "nano-agent", unix))]
use nano_systemd::{
    NanoSystemdRuntimeSupervision, NanoSystemdServiceSupervision, NanoSystemdSupervisionError,
};
#[cfg(feature = "record")]
use oak_sys::{
    CalibrationError as OakCalibrationError, CameraTimestampReference, CloseError as OakCloseError,
    ConnectedDeviceIdentityError, DepthAiBuildMetadataError, DepthAlignment, DepthConfig,
    DepthError, DepthFrame as OakDepthFrame, Device, DeviceConfig, ImageError,
    ImageFrame as OakImageFrame, ImuConfig, ImuError, Intrinsics as OakIntrinsics, MonoConfig,
    OakCameraSocket, OakEepromCalibrationEvidence, QueueConfig, StreamId as OakStreamId,
    UsbTransportEvidenceError, UsbTransportPolicy, UsbTransportSpeed,
};
#[cfg(all(feature = "record", feature = "actuation"))]
use robot_command_client::AppliedCommandReceipt;
#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    unix
))]
use robot_command_client::DisarmReceipt;
#[cfg(all(feature = "nano-agent", unix))]
use robot_server::V2ControllerOwnerTerminationError;
#[cfg(feature = "record")]
use std::num::{NonZeroU32, NonZeroU64};
#[cfg(any(feature = "record", test))]
use std::sync::Arc;
#[cfg(feature = "record")]
use std::sync::Mutex;
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
    #[cfg(all(feature = "nano-agent", unix))]
    NanoAgent(NanoAgentArgs),
    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    NanoWheelsOffQualification(NanoWheelsOffQualificationArgs),
    #[cfg(all(feature = "nano-attended-navigation-trial", unix))]
    NanoAttendedNavigationTrial(NanoAttendedNavigationTrialArgs),
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

#[derive(Args, Clone, Debug)]
#[cfg(all(feature = "nano-agent", unix))]
struct NanoAgentArgs {
    /// Root-owned directory containing the launch document and every bound asset.
    #[arg(long, value_name = "ABSOLUTE_DIRECTORY")]
    deployment_root: PathBuf,
    /// Canonical deployment-relative production launch document.
    #[arg(long, value_name = "RELATIVE_JSON")]
    launch_config: String,
    /// Persistent systemd-managed state root for maps, records, and evidence.
    #[arg(long, value_name = "ABSOLUTE_DIRECTORY")]
    state_root: PathBuf,
}

/// Manually invoked qualification surface compiled out of the production
/// `nano-agent` feature. Physical preconditions are accepted only from an
/// attended TTY; there are deliberately no flags or environment aliases for
/// them.
#[derive(Args, Clone, Debug)]
#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
struct NanoWheelsOffQualificationArgs {
    /// Root-owned directory containing the qualification launch and every bound asset.
    #[arg(long, value_name = "ABSOLUTE_DIRECTORY")]
    deployment_root: PathBuf,
    /// Canonical deployment-relative wheels-off qualification launch document.
    #[arg(long, value_name = "RELATIVE_JSON")]
    launch_config: String,
    /// Persistent state root for the qualification dataset and diagnostics.
    #[arg(long, value_name = "ABSOLUTE_DIRECTORY")]
    state_root: PathBuf,
    /// One typed, one-shot fault on the first nonzero candidate command.
    #[arg(long, value_name = "QUALIFICATION_FAULT")]
    fault_injection: Option<kiko_slam::navigation::WheelsOffQualificationFaultInjection>,
}

/// Foreground-only wheel-on trial. The commissioning launch binds the
/// candidate controller, OAK graph, accessories, navigation policy, and plant.
/// Physical claims are accepted only through the fresh controlling-TTY
/// ceremony; no flag or environment value can bypass it.
#[derive(Args, Clone, Debug)]
#[cfg(all(feature = "nano-attended-navigation-trial", unix))]
struct NanoAttendedNavigationTrialArgs {
    /// Root-owned directory containing the commissioning launch and all bound assets.
    #[arg(long, value_name = "ABSOLUTE_DIRECTORY")]
    deployment_root: PathBuf,
    /// Canonical deployment-relative attended commissioning launch document.
    #[arg(long, value_name = "RELATIVE_JSON")]
    launch_config: String,
    /// Persistent state root for live maps, datasets, and attended evidence.
    #[arg(long, value_name = "ABSOLUTE_DIRECTORY")]
    state_root: PathBuf,
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

#[cfg(feature = "record")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LiveInferenceRuntimeEvidence {
    superpoint_requested: InferenceBackend,
    superpoint_selected: LiveSelectedInferenceBackend,
    lightglue_requested: InferenceBackend,
    lightglue_selected: LiveSelectedInferenceBackend,
}

#[cfg(feature = "record")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LiveSelectedInferenceBackend {
    Cpu,
    CoremlGpu,
    Cuda,
    TensorRt,
}

#[cfg(feature = "record")]
impl LiveSelectedInferenceBackend {
    fn parse(
        component: &'static str,
        backend: InferenceBackend,
    ) -> Result<Self, LiveInferenceRuntimeEvidenceError> {
        match backend {
            InferenceBackend::Auto => {
                Err(LiveInferenceRuntimeEvidenceError::UnresolvedAuto { component })
            }
            InferenceBackend::Cpu => Ok(Self::Cpu),
            InferenceBackend::CoreMLGpu => Ok(Self::CoremlGpu),
            InferenceBackend::Cuda => Ok(Self::Cuda),
            InferenceBackend::TensorRT => Ok(Self::TensorRt),
        }
    }
}

#[cfg(feature = "record")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LiveInferenceRuntimeEvidenceError {
    UnresolvedAuto {
        component: &'static str,
    },
    SuperpointReplicaBackendMismatch {
        left: InferenceBackend,
        right: InferenceBackend,
    },
}

#[cfg(feature = "record")]
impl std::fmt::Display for LiveInferenceRuntimeEvidenceError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnresolvedAuto { component } => write!(
                formatter,
                "{component} reported unresolved auto as its selected inference backend"
            ),
            Self::SuperpointReplicaBackendMismatch { left, right } => write!(
                formatter,
                "SuperPoint replicas selected different inference backends: left={left:?} right={right:?}"
            ),
        }
    }
}

#[cfg(feature = "record")]
impl std::error::Error for LiveInferenceRuntimeEvidenceError {}

#[cfg(feature = "record")]
const LIVE_SLAM_RATE_WINDOW_CAPACITY: usize = 64;

#[cfg(feature = "record")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LiveSlamPipelineState {
    Running,
    Faulted,
    Closed,
}

#[cfg(feature = "record")]
#[derive(Debug, PartialEq, Eq)]
struct LiveSlamAttempt {
    source_arrival: HostMonotonicTimestamp,
}

#[cfg(feature = "record")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LiveSlamRateWindowEvidence {
    successful_completions: u8,
    span_ns: u64,
}

#[cfg(feature = "record")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LiveSlamTelemetrySnapshot {
    inference: LiveInferenceRuntimeEvidence,
    pipeline_state: LiveSlamPipelineState,
    started_pairs: u64,
    successful_pairs: u64,
    recoverable_failures: u64,
    fatal_failures: u64,
    last_successful_source_arrival: Option<HostMonotonicTimestamp>,
    last_successful_completion: Option<HostMonotonicTimestamp>,
    rate_window: Option<LiveSlamRateWindowEvidence>,
}

#[cfg(feature = "record")]
#[derive(Debug)]
struct LiveSlamTelemetryState {
    inference: LiveInferenceRuntimeEvidence,
    pipeline_state: LiveSlamPipelineState,
    started_pairs: u64,
    successful_pairs: u64,
    recoverable_failures: u64,
    fatal_failures: u64,
    in_flight: bool,
    last_started_source_arrival: Option<HostMonotonicTimestamp>,
    last_completion: Option<HostMonotonicTimestamp>,
    last_successful_source_arrival: Option<HostMonotonicTimestamp>,
    last_successful_completion: Option<HostMonotonicTimestamp>,
    successful_completion_history_ns: [u64; LIVE_SLAM_RATE_WINDOW_CAPACITY],
    successful_completion_history_len: usize,
    successful_completion_history_next: usize,
}

#[cfg(feature = "record")]
impl LiveSlamTelemetryState {
    fn snapshot(&self) -> LiveSlamTelemetrySnapshot {
        let rate_window = if self.successful_completion_history_len < 2 {
            None
        } else {
            let oldest_index =
                if self.successful_completion_history_len < LIVE_SLAM_RATE_WINDOW_CAPACITY {
                    0
                } else {
                    self.successful_completion_history_next
                };
            let newest_index =
                (self.successful_completion_history_next + LIVE_SLAM_RATE_WINDOW_CAPACITY - 1)
                    % LIVE_SLAM_RATE_WINDOW_CAPACITY;
            let oldest = self.successful_completion_history_ns[oldest_index];
            let newest = self.successful_completion_history_ns[newest_index];
            newest.checked_sub(oldest).and_then(|span_ns| {
                (span_ns > 0).then_some(LiveSlamRateWindowEvidence {
                    successful_completions: u8::try_from(self.successful_completion_history_len)
                        .expect("the fixed SLAM rate window capacity fits u8"),
                    span_ns,
                })
            })
        };
        LiveSlamTelemetrySnapshot {
            inference: self.inference,
            pipeline_state: self.pipeline_state,
            started_pairs: self.started_pairs,
            successful_pairs: self.successful_pairs,
            recoverable_failures: self.recoverable_failures,
            fatal_failures: self.fatal_failures,
            last_successful_source_arrival: self.last_successful_source_arrival,
            last_successful_completion: self.last_successful_completion,
            rate_window,
        }
    }
}

#[cfg(feature = "record")]
#[derive(Clone, Debug)]
struct LiveSlamTelemetry {
    state: Arc<Mutex<LiveSlamTelemetryState>>,
}

#[cfg(feature = "record")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LiveSlamTelemetryError {
    LockPoisoned,
    PipelineNotRunning {
        state: LiveSlamPipelineState,
    },
    AttemptAlreadyInFlight,
    NoAttemptInFlight,
    AttemptSourceMismatch {
        expected: HostMonotonicTimestamp,
        actual: HostMonotonicTimestamp,
    },
    SourceArrivalRegressed {
        previous: HostMonotonicTimestamp,
        actual: HostMonotonicTimestamp,
    },
    CompletionBeforeSource {
        source: HostMonotonicTimestamp,
        completion: HostMonotonicTimestamp,
    },
    CompletionRegressed {
        previous: HostMonotonicTimestamp,
        actual: HostMonotonicTimestamp,
    },
    CounterExhausted {
        field: &'static str,
    },
}

#[cfg(feature = "record")]
impl std::fmt::Display for LiveSlamTelemetryError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "invalid live SLAM telemetry transition: {self:?}"
        )
    }
}

#[cfg(feature = "record")]
impl std::error::Error for LiveSlamTelemetryError {}

#[cfg(feature = "record")]
impl LiveSlamTelemetry {
    fn new(inference: LiveInferenceRuntimeEvidence) -> Self {
        Self {
            state: Arc::new(Mutex::new(LiveSlamTelemetryState {
                inference,
                pipeline_state: LiveSlamPipelineState::Running,
                started_pairs: 0,
                successful_pairs: 0,
                recoverable_failures: 0,
                fatal_failures: 0,
                in_flight: false,
                last_started_source_arrival: None,
                last_completion: None,
                last_successful_source_arrival: None,
                last_successful_completion: None,
                successful_completion_history_ns: [0; LIVE_SLAM_RATE_WINDOW_CAPACITY],
                successful_completion_history_len: 0,
                successful_completion_history_next: 0,
            })),
        }
    }

    fn begin(
        &self,
        source_arrival: HostMonotonicTimestamp,
    ) -> Result<LiveSlamAttempt, LiveSlamTelemetryError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| LiveSlamTelemetryError::LockPoisoned)?;
        if state.pipeline_state != LiveSlamPipelineState::Running {
            return Err(LiveSlamTelemetryError::PipelineNotRunning {
                state: state.pipeline_state,
            });
        }
        if state.in_flight {
            return Err(LiveSlamTelemetryError::AttemptAlreadyInFlight);
        }
        if let Some(previous) = state.last_started_source_arrival
            && source_arrival < previous
        {
            return Err(LiveSlamTelemetryError::SourceArrivalRegressed {
                previous,
                actual: source_arrival,
            });
        }
        state.started_pairs =
            state
                .started_pairs
                .checked_add(1)
                .ok_or(LiveSlamTelemetryError::CounterExhausted {
                    field: "started_pairs",
                })?;
        state.in_flight = true;
        state.last_started_source_arrival = Some(source_arrival);
        Ok(LiveSlamAttempt { source_arrival })
    }

    fn complete_success(
        &self,
        attempt: LiveSlamAttempt,
        completed_at: HostMonotonicTimestamp,
    ) -> Result<LiveSlamTelemetrySnapshot, LiveSlamTelemetryError> {
        let mut state = self.lock_for_completion(&attempt, completed_at)?;
        state.successful_pairs = state.successful_pairs.checked_add(1).ok_or(
            LiveSlamTelemetryError::CounterExhausted {
                field: "successful_pairs",
            },
        )?;
        state.in_flight = false;
        state.last_completion = Some(completed_at);
        state.last_successful_source_arrival = Some(attempt.source_arrival);
        state.last_successful_completion = Some(completed_at);
        let next = state.successful_completion_history_next;
        state.successful_completion_history_ns[next] = completed_at.as_nanos();
        state.successful_completion_history_next = (next + 1) % LIVE_SLAM_RATE_WINDOW_CAPACITY;
        state.successful_completion_history_len = state
            .successful_completion_history_len
            .saturating_add(1)
            .min(LIVE_SLAM_RATE_WINDOW_CAPACITY);
        Ok(state.snapshot())
    }

    fn complete_failure(
        &self,
        attempt: LiveSlamAttempt,
        completed_at: HostMonotonicTimestamp,
        fatal: bool,
    ) -> Result<LiveSlamTelemetrySnapshot, LiveSlamTelemetryError> {
        let mut state = self.lock_for_completion(&attempt, completed_at)?;
        if fatal {
            state.fatal_failures = state.fatal_failures.checked_add(1).ok_or(
                LiveSlamTelemetryError::CounterExhausted {
                    field: "fatal_failures",
                },
            )?;
            state.in_flight = false;
            state.last_completion = Some(completed_at);
            state.pipeline_state = LiveSlamPipelineState::Faulted;
        } else {
            state.recoverable_failures = state.recoverable_failures.checked_add(1).ok_or(
                LiveSlamTelemetryError::CounterExhausted {
                    field: "recoverable_failures",
                },
            )?;
            state.in_flight = false;
            state.last_completion = Some(completed_at);
        }
        Ok(state.snapshot())
    }

    fn lock_for_completion(
        &self,
        attempt: &LiveSlamAttempt,
        completed_at: HostMonotonicTimestamp,
    ) -> Result<std::sync::MutexGuard<'_, LiveSlamTelemetryState>, LiveSlamTelemetryError> {
        let state = self
            .state
            .lock()
            .map_err(|_| LiveSlamTelemetryError::LockPoisoned)?;
        if !state.in_flight {
            return Err(LiveSlamTelemetryError::NoAttemptInFlight);
        }
        if let Some(expected) = state.last_started_source_arrival
            && expected != attempt.source_arrival
        {
            return Err(LiveSlamTelemetryError::AttemptSourceMismatch {
                expected,
                actual: attempt.source_arrival,
            });
        }
        if completed_at < attempt.source_arrival {
            return Err(LiveSlamTelemetryError::CompletionBeforeSource {
                source: attempt.source_arrival,
                completion: completed_at,
            });
        }
        if let Some(previous) = state.last_completion
            && completed_at < previous
        {
            return Err(LiveSlamTelemetryError::CompletionRegressed {
                previous,
                actual: completed_at,
            });
        }
        Ok(state)
    }

    fn close(&self) -> Result<LiveSlamTelemetrySnapshot, LiveSlamTelemetryError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| LiveSlamTelemetryError::LockPoisoned)?;
        if state.in_flight {
            return Err(LiveSlamTelemetryError::AttemptAlreadyInFlight);
        }
        if state.pipeline_state == LiveSlamPipelineState::Running {
            state.pipeline_state = LiveSlamPipelineState::Closed;
        }
        Ok(state.snapshot())
    }

    fn fault(&self) -> Result<LiveSlamTelemetrySnapshot, LiveSlamTelemetryError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| LiveSlamTelemetryError::LockPoisoned)?;
        state.pipeline_state = LiveSlamPipelineState::Faulted;
        Ok(state.snapshot())
    }

    fn snapshot(&self) -> Result<LiveSlamTelemetrySnapshot, LiveSlamTelemetryError> {
        self.state
            .lock()
            .map(|state| state.snapshot())
            .map_err(|_| LiveSlamTelemetryError::LockPoisoned)
    }
}

struct InferenceConfig {
    superpoint_left: SuperPoint,
    superpoint_right: SuperPoint,
    lightglue: LightGlue,
    #[cfg(feature = "record")]
    superpoint_requested_backend: InferenceBackend,
    #[cfg(feature = "record")]
    lightglue_requested_backend: InferenceBackend,
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

        let model_dir = Path::new(kiko_slam::WORKSPACE_MODEL_DIRECTORY);
        let sp_path = resolve_model_path(model_dir, args.superpoint_model.as_ref(), "sp.onnx");
        let lg_path = resolve_model_path(model_dir, args.lightglue_model.as_ref(), "lg.onnx");
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
            #[cfg(feature = "record")]
            superpoint_requested_backend: superpoint_backend,
            #[cfg(feature = "record")]
            lightglue_requested_backend: lightglue_backend,
            key_limit,
            downscale,
        })
    }

    #[cfg(feature = "record")]
    fn runtime_evidence(
        &self,
    ) -> Result<LiveInferenceRuntimeEvidence, LiveInferenceRuntimeEvidenceError> {
        let superpoint_left = self.superpoint_left.backend();
        let superpoint_right = self.superpoint_right.backend();
        if superpoint_left != superpoint_right {
            return Err(
                LiveInferenceRuntimeEvidenceError::SuperpointReplicaBackendMismatch {
                    left: superpoint_left,
                    right: superpoint_right,
                },
            );
        }
        let superpoint_selected =
            LiveSelectedInferenceBackend::parse("SuperPoint", superpoint_left)?;
        let lightglue_selected =
            LiveSelectedInferenceBackend::parse("LightGlue", self.lightglue.backend())?;
        Ok(LiveInferenceRuntimeEvidence {
            superpoint_requested: self.superpoint_requested_backend,
            superpoint_selected,
            lightglue_requested: self.lightglue_requested_backend,
            lightglue_selected,
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
        #[cfg(all(feature = "nano-agent", unix))]
        Command::NanoAgent(args) => run_nano_agent(args),
        #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
        Command::NanoWheelsOffQualification(args) => run_nano_wheels_off_qualification(args),
        #[cfg(all(feature = "nano-attended-navigation-trial", unix))]
        Command::NanoAttendedNavigationTrial(args) => run_nano_attended_navigation_trial(args),
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

/// Build the canonical robot's geometric/worker tracker policy without
/// consulting process environment.
///
/// The production and attended wheels-off launch documents already bind the
/// inference models, backends, and keypoint/downscale limits. The remaining
/// tracker policy is fixed here until a future versioned launch schema owns
/// it; accepting ambient `KIKO_*` overrides in a system service would create a
/// second, unaudited configuration authority. ONNX Runtime session tuning is a
/// separate, pre-existing inference boundary and is not represented by this
/// builder.
///
/// The canonical graph deliberately uses the tracker's existing aggregate
/// SuperPoint descriptors for loop closure and relocalization. It therefore
/// opens no ambient EigenPlaces path. This is functionally available loop
/// closure, not evidence that its place-recognition quality equals a learned
/// EigenPlaces model; representative-map qualification remains required.
#[cfg(all(feature = "nano-agent", unix))]
fn build_canonical_nano_tracker_config(
    defaults: TrackerDefaults,
    key_limit: KeypointLimit,
    downscale: DownscaleFactor,
) -> Result<TrackerConfig, Box<dyn std::error::Error>> {
    let ransac_defaults = RansacConfig::default();
    let ransac = RansacConfig::try_new(
        ransac_defaults.max_iterations(),
        ransac_defaults.reprojection_threshold_px(),
        defaults.min_inliers,
        ransac_defaults.seed(),
    )?;
    let ba = build_ba_config_from_values(BaConfigValues {
        window: DEFAULT_BA_WINDOW,
        iterations: DEFAULT_BA_ITERS,
        min_observations: DEFAULT_BA_MIN_OBS,
        huber_delta_px: DEFAULT_BA_HUBER_PX,
        initial_lambda: DEFAULT_BA_DAMPING,
        lambda_factor: DEFAULT_LM_FACTOR,
        min_lambda: DEFAULT_LM_MIN,
        max_lambda: DEFAULT_LM_MAX,
    })?;
    let keyframe_policy = KeyframePolicy::new(
        defaults.refresh_inliers,
        DEFAULT_KEYFRAME_PARALLAX_PX,
        DEFAULT_KEYFRAME_COVISIBILITY,
    )?;
    let redundancy = Some(RedundancyPolicy::new(
        DEFAULT_KEYFRAME_REDUNDANT_COVISIBILITY,
    )?);
    let backend = Some(BackendConfig::new(2)?);
    let loop_subsystem = LoopSubsystemConfig::bootstrap_descriptors(
        LoopClosureConfig::default(),
        RelocalizationConfig::default(),
    );

    eprintln!(
        "canonical Nano tracker requested: keyframe_min_points={} refresh_inliers={} parallax_px={:.1} min_covisibility={:.2} redundant_covisibility={:.2} min_inliers={} downscale={} max_keypoints={} async_backend=true backend_queue_depth=2 loop_closure_requested=true descriptor_mode=bootstrap descriptor_model_source=none relocalization_requested=true geometric_worker_policy_source=canonical-fixed ort_session_policy_source=environment-compatibility",
        defaults.min_keyframe_points,
        defaults.refresh_inliers,
        DEFAULT_KEYFRAME_PARALLAX_PX,
        DEFAULT_KEYFRAME_COVISIBILITY,
        DEFAULT_KEYFRAME_REDUNDANT_COVISIBILITY,
        defaults.min_inliers,
        downscale.get(),
        key_limit.get(),
    );

    Ok(TrackerConfig {
        max_keypoints: key_limit,
        downscale,
        min_keyframe_points: defaults.min_keyframe_points,
        ransac,
        triangulation: TriangulationConfig::default(),
        keyframe_policy,
        ba,
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

#[cfg(feature = "record")]
const fn live_inference_backend_name(backend: InferenceBackend) -> &'static str {
    match backend {
        InferenceBackend::Auto => "auto",
        InferenceBackend::Cpu => "cpu",
        InferenceBackend::CoreMLGpu => "coreml_gpu",
        InferenceBackend::Cuda => "cuda",
        InferenceBackend::TensorRT => "tensorrt",
    }
}

#[cfg(feature = "record")]
const fn live_selected_inference_backend_name(
    backend: LiveSelectedInferenceBackend,
) -> &'static str {
    match backend {
        LiveSelectedInferenceBackend::Cpu => "cpu",
        LiveSelectedInferenceBackend::CoremlGpu => "coreml_gpu",
        LiveSelectedInferenceBackend::Cuda => "cuda",
        LiveSelectedInferenceBackend::TensorRt => "tensorrt",
    }
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
        ..
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
    UsbTransport {
        source: UsbTransportEvidenceError,
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
            Self::UsbTransport { source } => {
                write!(f, "could not read admitted OAK USB transport: {source}")
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
            Self::UsbTransport { source } => Some(source),
            Self::LinkedDepthAiBuildMetadata { source } => Some(source),
        }
    }
}

#[cfg(feature = "record")]
#[derive(Clone, Debug, PartialEq, Eq)]
struct OakRuntimeProvenance {
    connected_mxid: String,
    usb_requested_maximum: UsbTransportSpeed,
    usb_required_minimum: UsbTransportSpeed,
    usb_observed: UsbTransportSpeed,
    depthai_sdk_version: String,
    depthai_sdk_commit: String,
    embedded_device_artifact_version: String,
    embedded_bootloader_artifact_version: String,
}

#[cfg(feature = "record")]
impl OakRuntimeProvenance {
    fn dataset_device_label(&self) -> String {
        format!(
            "OAK-D mxid={} usb_requested_maximum={} usb_required_minimum={} usb_observed={} depthai_sdk={} depthai_commit={} embedded_device={} embedded_bootloader={} timestamp=device_exposure_midpoint",
            self.connected_mxid,
            self.usb_requested_maximum,
            self.usb_required_minimum,
            self.usb_observed,
            self.depthai_sdk_version,
            self.depthai_sdk_commit,
            self.embedded_device_artifact_version,
            self.embedded_bootloader_artifact_version,
        )
    }

    #[cfg(all(feature = "nano-agent", unix))]
    fn from_nano_bootstrap(bootstrap: &PreparedNanoBootstrap) -> Self {
        Self::from_admitted_nano_parts(
            &bootstrap.oak_connected_identity,
            bootstrap.oak_usb_transport,
            &bootstrap.depthai_build_metadata,
        )
    }

    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    fn from_nano_wheels_off_qualification_bootstrap(
        bootstrap: &kiko_slam::navigation::PreparedNanoWheelsOffQualificationBootstrap,
    ) -> Self {
        Self::from_admitted_nano_parts(
            &bootstrap.oak_connected_identity,
            bootstrap.oak_usb_transport,
            &bootstrap.depthai_build_metadata,
        )
    }

    #[cfg(all(feature = "nano-agent", unix))]
    fn from_admitted_nano_parts(
        connected_identity: &oak_sys::ConnectedDeviceIdentity,
        usb_transport: kiko_slam::navigation::AdmittedOakSuperSpeedEvidence,
        build_metadata: &oak_sys::DepthAiBuildMetadata,
    ) -> Self {
        Self {
            connected_mxid: connected_identity.mxid().to_owned(),
            usb_requested_maximum: usb_transport.requested_maximum(),
            usb_required_minimum: usb_transport.required_minimum(),
            usb_observed: usb_transport.observed(),
            depthai_sdk_version: build_metadata.sdk_version().to_owned(),
            depthai_sdk_commit: build_metadata.sdk_commit().to_owned(),
            embedded_device_artifact_version: build_metadata
                .embedded_device_artifact_version()
                .to_owned(),
            embedded_bootloader_artifact_version: build_metadata
                .embedded_bootloader_artifact_version()
                .to_owned(),
        }
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
    let usb = device
        .usb_transport_evidence()
        .map_err(|source| OakRuntimeProvenanceError::UsbTransport { source })?;
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
        "{context} OAK USB transport: requested_maximum={} required_minimum={} observed={}",
        usb.requested_maximum(),
        usb.required_minimum(),
        usb.observed(),
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
        usb_requested_maximum: usb.requested_maximum(),
        usb_required_minimum: usb.required_minimum(),
        usb_observed: usb.observed(),
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

/// Whether stereo bootstrap must acquire the optional raw EEPROM matrices.
///
/// Raw IMU recording deliberately omits this evidence: its stream metadata is
/// `uncalibrated_unknown`, and the navigation calibration assembler separately
/// requires source-bound IMU-to-base calibration. Requiring a vendor EEPROM
/// IMU extrinsic there would both overstate its meaning and make raw capture
/// depend on data that some supported OAK devices do not expose.
#[cfg(feature = "record")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OakEepromEvidencePolicy {
    Omit,
    Require,
}

#[cfg(feature = "record")]
impl OakEepromEvidencePolicy {
    fn acquire(
        self,
        device: &Device,
    ) -> Result<Option<OakEepromCalibrationEvidence>, OakCalibrationError> {
        match self {
            Self::Omit => Ok(None),
            Self::Require => device.eeprom_calibration_evidence().map(Some),
        }
    }
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
    oak_eeprom_evidence_policy: OakEepromEvidencePolicy,
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
    let oak_eeprom = oak_eeprom_evidence_policy
        .acquire(device)
        .map_err(|source| StereoBootstrapError::Calibration { source })?;
    let calibration = build_calibration(
        left_intrinsics,
        right_intrinsics,
        baseline_m,
        oak_eeprom,
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
        usb_transport: UsbTransportPolicy::super_speed_required(),
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
        } = bootstrap_stereo(
            &mut device,
            &mono_config,
            OakEepromEvidencePolicy::Omit,
            running.as_ref(),
            &mut pairer,
        )?;

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
fn navigation_dataset_may_publish(
    authoritative_failure: bool,
    journal_descriptor_finalized: bool,
) -> bool {
    !authoritative_failure && journal_descriptor_finalized
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
    slam: LiveSlamTelemetrySnapshot,
}

#[cfg(feature = "record")]
#[derive(Debug)]
struct LiveRgbVizMsg {
    device_capture_sequence: i64,
    host_delivery_sequence: i64,
    device_timestamp_ns: i64,
    timestamp_reference: CameraTimestampReference,
    stream_epoch: u64,
    width: u32,
    height: u32,
    pixels_bgr8: Vec<u8>,
}

#[cfg(feature = "record")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LiveRgbFrameKey {
    device_capture_sequence: i64,
    host_delivery_sequence: i64,
    device_timestamp_ns: i64,
    timestamp_reference: CameraTimestampReference,
    stream_epoch: u64,
    width: u32,
    height: u32,
}

#[cfg(all(feature = "record", feature = "nano-agent", unix))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LiveRgbVizBuildError {
    WrongStream { actual: OakStreamId },
    HostDeliverySequenceUnrepresentable { actual: u64 },
    RowBytesOverflow { width: u32 },
    PixelBytesOverflow { width: u32, height: u32 },
    StrideMismatch { expected: u32, actual: u32 },
    PixelLengthMismatch { expected: usize, actual: usize },
}

#[cfg(all(feature = "record", feature = "nano-agent", unix))]
impl std::fmt::Display for LiveRgbVizBuildError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "live RGB frame cannot be represented as an exact diagnostic: {self:?}"
        )
    }
}

#[cfg(all(feature = "record", feature = "nano-agent", unix))]
impl std::error::Error for LiveRgbVizBuildError {}

#[cfg(all(feature = "record", feature = "nano-agent", unix))]
fn validate_live_rgb_viz_layout(
    width: u32,
    height: u32,
    stride_bytes: u32,
    pixel_length: usize,
) -> Result<(), LiveRgbVizBuildError> {
    let expected_stride = width
        .checked_mul(3)
        .ok_or(LiveRgbVizBuildError::RowBytesOverflow { width })?;
    if stride_bytes != expected_stride {
        return Err(LiveRgbVizBuildError::StrideMismatch {
            expected: expected_stride,
            actual: stride_bytes,
        });
    }
    let expected = usize::try_from(
        expected_stride
            .checked_mul(height)
            .ok_or(LiveRgbVizBuildError::PixelBytesOverflow { width, height })?,
    )
    .map_err(|_| LiveRgbVizBuildError::PixelBytesOverflow { width, height })?;
    if pixel_length != expected {
        return Err(LiveRgbVizBuildError::PixelLengthMismatch {
            expected,
            actual: pixel_length,
        });
    }
    Ok(())
}

#[cfg(all(feature = "record", feature = "nano-agent", unix))]
impl LiveRgbVizMsg {
    fn try_from_oak(
        frame: &OakImageFrame,
        stream_epoch: StreamEpochId,
    ) -> Result<Self, LiveRgbVizBuildError> {
        if frame.stream != OakStreamId::Rgb {
            return Err(LiveRgbVizBuildError::WrongStream {
                actual: frame.stream,
            });
        }
        validate_live_rgb_viz_layout(
            frame.width,
            frame.height,
            frame.stride_bytes,
            frame.pixels().len(),
        )?;
        let host_delivery_sequence =
            i64::try_from(frame.host_delivery_sequence.as_u64()).map_err(|_| {
                LiveRgbVizBuildError::HostDeliverySequenceUnrepresentable {
                    actual: frame.host_delivery_sequence.as_u64(),
                }
            })?;
        Ok(Self {
            device_capture_sequence: frame.device_capture_sequence.as_i64(),
            host_delivery_sequence,
            device_timestamp_ns: frame.timestamp.as_nanos(),
            timestamp_reference: frame.timestamp_reference,
            stream_epoch: stream_epoch.get(),
            width: frame.width,
            height: frame.height,
            pixels_bgr8: frame.pixels().to_vec(),
        })
    }
}

#[cfg(feature = "record")]
impl LiveRgbVizMsg {
    const fn frame_key(&self) -> LiveRgbFrameKey {
        LiveRgbFrameKey {
            device_capture_sequence: self.device_capture_sequence,
            host_delivery_sequence: self.host_delivery_sequence,
            device_timestamp_ns: self.device_timestamp_ns,
            timestamp_reference: self.timestamp_reference,
            stream_epoch: self.stream_epoch,
            width: self.width,
            height: self.height,
        }
    }
}

#[cfg(all(feature = "record", feature = "nano-agent", unix))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LiveFaceVizBuildError {
    WrongStream {
        actual: OakStreamId,
    },
    ObservationMismatch,
    ResultSequenceMismatch {
        batch: u64,
        tracking: u64,
    },
    TruncatedCountMismatch {
        batch: u32,
        tracking: u32,
    },
    CaptureSequenceMismatch {
        observation: u64,
        oak: u64,
    },
    DimensionsMismatch {
        observation: [u32; 2],
        oak: [u32; 2],
    },
    LayoutSizeOverflow,
    LayoutNotTightlyPackedBgr8 {
        channel_order: ChannelOrder,
        stride_bytes: u32,
        byte_len: usize,
    },
    HostDeliverySequenceUnrepresentable {
        actual: u64,
    },
    DetectorResultSequenceUnrepresentable {
        actual: u64,
    },
    PixelCoordinateUnrepresentable {
        field: &'static str,
        value: u32,
    },
}

#[cfg(all(feature = "record", feature = "nano-agent", unix))]
impl std::fmt::Display for LiveFaceVizBuildError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "face metadata cannot be represented as an exact live diagnostic: {self:?}"
        )
    }
}

#[cfg(all(feature = "record", feature = "nano-agent", unix))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ValidatedLiveFaceViz {
    frame_key: LiveRgbFrameKey,
    detector_result_sequence: i64,
}

#[cfg(all(feature = "record", feature = "nano-agent", unix))]
fn exact_u32_as_f32(field: &'static str, value: u32) -> Result<f32, LiveFaceVizBuildError> {
    let converted = value as f32;
    if f64::from(converted) == f64::from(value) {
        Ok(converted)
    } else {
        Err(LiveFaceVizBuildError::PixelCoordinateUnrepresentable { field, value })
    }
}

#[cfg(all(feature = "record", feature = "nano-agent", unix))]
fn validate_live_face_viz(
    message: NanoFaceDiagnosticFrame,
) -> Result<ValidatedLiveFaceViz, LiveFaceVizBuildError> {
    let provenance = message.provenance();
    if provenance.stream() != OakStreamId::Rgb {
        return Err(LiveFaceVizBuildError::WrongStream {
            actual: provenance.stream(),
        });
    }
    let output = message.output();
    let batch = output.batch();
    let tracking = output.tracking();
    let observation = batch.observation();
    if observation != tracking.observation() {
        return Err(LiveFaceVizBuildError::ObservationMismatch);
    }
    let batch_result_sequence = batch.detector_result_sequence().get();
    let tracking_result_sequence = tracking.detector_result_sequence().get();
    if batch_result_sequence != tracking_result_sequence {
        return Err(LiveFaceVizBuildError::ResultSequenceMismatch {
            batch: batch_result_sequence,
            tracking: tracking_result_sequence,
        });
    }
    if batch.detector_truncated_count() != tracking.detector_truncated_count() {
        return Err(LiveFaceVizBuildError::TruncatedCountMismatch {
            batch: batch.detector_truncated_count(),
            tracking: tracking.detector_truncated_count(),
        });
    }
    let frame_id = observation.frame_id();
    let oak_capture_sequence = provenance.device_capture_sequence().as_u64();
    if frame_id.sequence() != oak_capture_sequence {
        return Err(LiveFaceVizBuildError::CaptureSequenceMismatch {
            observation: frame_id.sequence(),
            oak: oak_capture_sequence,
        });
    }
    let layout = observation.layout();
    if layout.width_px() != provenance.width_px() || layout.height_px() != provenance.height_px() {
        return Err(LiveFaceVizBuildError::DimensionsMismatch {
            observation: [layout.width_px(), layout.height_px()],
            oak: [provenance.width_px(), provenance.height_px()],
        });
    }
    let expected_stride = layout
        .width_px()
        .checked_mul(3)
        .ok_or(LiveFaceVizBuildError::LayoutSizeOverflow)?;
    let expected_byte_len = usize::try_from(
        u64::from(expected_stride)
            .checked_mul(u64::from(layout.height_px()))
            .ok_or(LiveFaceVizBuildError::LayoutSizeOverflow)?,
    )
    .map_err(|_| LiveFaceVizBuildError::LayoutSizeOverflow)?;
    if layout.channel_order() != ChannelOrder::Bgr
        || layout.stride_bytes() != expected_stride
        || layout.byte_len() != expected_byte_len
    {
        return Err(LiveFaceVizBuildError::LayoutNotTightlyPackedBgr8 {
            channel_order: layout.channel_order(),
            stride_bytes: layout.stride_bytes(),
            byte_len: layout.byte_len(),
        });
    }
    // Rerun's 2D geometry is f32. Reject dimensions that cannot be converted
    // exactly instead of silently shifting integer detector rectangles.
    let _ = exact_u32_as_f32("frame width", layout.width_px())?;
    let _ = exact_u32_as_f32("frame height", layout.height_px())?;
    for detection in batch.iter() {
        let rectangle = detection.rectangle();
        let _ = exact_u32_as_f32("face left", rectangle.left_px())?;
        let _ = exact_u32_as_f32("face top", rectangle.top_px())?;
        let _ = exact_u32_as_f32("face width", rectangle.width_px())?;
        let _ = exact_u32_as_f32("face height", rectangle.height_px())?;
    }
    let host_delivery_sequence = i64::try_from(provenance.host_delivery_sequence().as_u64())
        .map_err(
            |_| LiveFaceVizBuildError::HostDeliverySequenceUnrepresentable {
                actual: provenance.host_delivery_sequence().as_u64(),
            },
        )?;
    let detector_result_sequence = i64::try_from(batch_result_sequence).map_err(|_| {
        LiveFaceVizBuildError::DetectorResultSequenceUnrepresentable {
            actual: batch_result_sequence,
        }
    })?;
    Ok(ValidatedLiveFaceViz {
        frame_key: LiveRgbFrameKey {
            device_capture_sequence: provenance.device_capture_sequence().as_i64(),
            host_delivery_sequence,
            device_timestamp_ns: provenance.timestamp().as_nanos(),
            timestamp_reference: provenance.timestamp_reference(),
            stream_epoch: frame_id.stream_epoch().get(),
            width: provenance.width_px(),
            height: provenance.height_px(),
        },
        detector_result_sequence,
    })
}

#[cfg(all(feature = "record", feature = "nano-agent", unix))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct LiveFaceVizStats {
    received: u64,
    logged: u64,
    overlay_matched: u64,
    overlay_unmatched: u64,
    invalid: u64,
    consumer_cancelled: u64,
    pending_abandoned: u64,
}

#[cfg(all(feature = "record", feature = "nano-agent", unix))]
#[derive(Debug, Default)]
struct LiveFaceVizCounters {
    received: std::sync::atomic::AtomicU64,
    logged: std::sync::atomic::AtomicU64,
    overlay_matched: std::sync::atomic::AtomicU64,
    overlay_unmatched: std::sync::atomic::AtomicU64,
    invalid: std::sync::atomic::AtomicU64,
    consumer_cancelled: std::sync::atomic::AtomicU64,
    pending_abandoned: std::sync::atomic::AtomicU64,
}

#[cfg(all(feature = "record", feature = "nano-agent", unix))]
fn increment_live_face_counter(counter: &std::sync::atomic::AtomicU64) {
    let _ = counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
        value.checked_add(1)
    });
}

#[cfg(all(feature = "record", feature = "nano-agent", unix))]
impl LiveFaceVizCounters {
    fn record_received(&self) {
        increment_live_face_counter(&self.received);
    }

    fn record_logged(&self, overlay_matched: bool) {
        increment_live_face_counter(&self.logged);
        if overlay_matched {
            increment_live_face_counter(&self.overlay_matched);
        } else {
            increment_live_face_counter(&self.overlay_unmatched);
        }
    }

    fn record_invalid(&self) {
        increment_live_face_counter(&self.invalid);
    }

    fn record_cancelled(&self, pending_abandoned: bool) {
        increment_live_face_counter(&self.consumer_cancelled);
        if pending_abandoned {
            increment_live_face_counter(&self.pending_abandoned);
        }
    }

    fn snapshot(&self) -> LiveFaceVizStats {
        LiveFaceVizStats {
            received: self.received.load(Ordering::Relaxed),
            logged: self.logged.load(Ordering::Relaxed),
            overlay_matched: self.overlay_matched.load(Ordering::Relaxed),
            overlay_unmatched: self.overlay_unmatched.load(Ordering::Relaxed),
            invalid: self.invalid.load(Ordering::Relaxed),
            consumer_cancelled: self.consumer_cancelled.load(Ordering::Relaxed),
            pending_abandoned: self.pending_abandoned.load(Ordering::Relaxed),
        }
    }
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
    conservative_decision_to_send_ns: Option<u64>,
    command_send_to_ack_ns: Option<u64>,
    conservative_decision_to_ack_ns: Option<u64>,
    acknowledged_at_ns_decimal: String,
    known_active_through_ns_decimal: String,
}

#[cfg(feature = "record")]
#[derive(Debug)]
struct LiveFaultActuationViz {
    kind: String,
    controller_stop_confirmed: bool,
}

#[cfg(feature = "record")]
fn checked_monotonic_duration_ns(start_ns: u128, end_ns: u128) -> Option<u64> {
    end_ns
        .checked_sub(start_ns)
        .and_then(|duration_ns| u64::try_from(duration_ns).ok())
}

#[cfg(all(feature = "record", feature = "actuation"))]
fn live_applied_actuation_viz(
    receipt: &AppliedCommandReceipt,
    decision_started_at: Option<HostMonotonicTimestamp>,
) -> LiveAppliedActuationViz {
    let pwm = receipt.applied_timer_pwm();
    let sent_at_ns = receipt.sent_at().nanos_since_clock_start();
    let acknowledged_at_ns = receipt.acknowledged_at().nanos_since_clock_start();
    let conservative_decision_to_send_ns = decision_started_at.and_then(|started_at| {
        checked_monotonic_duration_ns(u128::from(started_at.as_nanos()), sent_at_ns)
    });
    let command_send_to_ack_ns = checked_monotonic_duration_ns(sent_at_ns, acknowledged_at_ns);
    let conservative_decision_to_ack_ns = decision_started_at.and_then(|started_at| {
        checked_monotonic_duration_ns(u128::from(started_at.as_nanos()), acknowledged_at_ns)
    });
    LiveAppliedActuationViz {
        controller_boot_id: receipt.controller_session().boot_id().get(),
        control_epoch: receipt.controller_session().control_epoch().get(),
        sequence: receipt.sequence().get(),
        applied_pwm: [pwm.left().get(), pwm.right().get()],
        remaining_lease_at_server_emission_ms: receipt.remaining_lease_at_server_emission().get(),
        conservative_decision_to_send_ns,
        command_send_to_ack_ns,
        conservative_decision_to_ack_ns,
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LiveControlTickTiming {
    /// Delay from crossbeam's scheduled delivery instant until this worker
    /// actually selected the tick. This excludes MPC and controller I/O.
    current_lateness_ns: u64,
    /// Process-lifetime high-water mark. Later packets retain a peak even when
    /// an intermediate visualization packet was evicted by DropOldest.
    maximum_lateness_ns: u64,
}

#[cfg(feature = "record")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LiveControlTickTimingError {
    ScheduledAfterSelection,
    LatenessOutsideU64 { lateness_ns: u128 },
}

#[cfg(feature = "record")]
impl std::fmt::Display for LiveControlTickTimingError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ScheduledAfterSelection => {
                formatter.write_str("the selected tick instant preceded the scheduled tick instant")
            }
            Self::LatenessOutsideU64 { lateness_ns } => write!(
                formatter,
                "tick lateness {lateness_ns} ns exceeds the u64 telemetry domain"
            ),
        }
    }
}

#[cfg(feature = "record")]
impl std::error::Error for LiveControlTickTimingError {}

#[cfg(feature = "record")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LiveNavigationVizMessageKind {
    State {
        control_tick_timing: Option<LiveControlTickTiming>,
    },
    ControlTickTimingOnly {
        control_tick_timing: LiveControlTickTiming,
    },
}

#[cfg(feature = "record")]
impl LiveNavigationVizMessageKind {
    const fn updates_navigation_state(self) -> bool {
        matches!(self, Self::State { .. })
    }

    const fn control_tick_timing(self) -> Option<LiveControlTickTiming> {
        match self {
            Self::State {
                control_tick_timing,
            } => control_tick_timing,
            Self::ControlTickTimingOnly {
                control_tick_timing,
            } => Some(control_tick_timing),
        }
    }
}

#[cfg(feature = "record")]
fn measure_live_control_tick_timing(
    scheduled_at: Instant,
    selected_at: Instant,
    previous_maximum_lateness_ns: u64,
) -> Result<LiveControlTickTiming, LiveControlTickTimingError> {
    let current_lateness = selected_at
        .checked_duration_since(scheduled_at)
        .ok_or(LiveControlTickTimingError::ScheduledAfterSelection)?
        .as_nanos();
    let current_lateness_ns = u64::try_from(current_lateness).map_err(|_| {
        LiveControlTickTimingError::LatenessOutsideU64 {
            lateness_ns: current_lateness,
        }
    })?;
    Ok(LiveControlTickTiming {
        current_lateness_ns,
        maximum_lateness_ns: previous_maximum_lateness_ns.max(current_lateness_ns),
    })
}

#[cfg(feature = "record")]
#[derive(Debug)]
struct LiveNavigationVizMsg {
    tick_sequence: i64,
    host_timestamp_ns: Option<u64>,
    goal: Option<[f32; 2]>,
    goal_state: String,
    odometry_state: Option<String>,
    path: Option<Vec<[f32; 2]>>,
    local_costmap: Option<LiveLocalCostmapViz>,
    base_to_odom: Option<[f64; 3]>,
    odom_to_map: Option<[f64; 3]>,
    predicted_odom: Option<Vec<[f32; 2]>>,
    decision_id: Option<u64>,
    request_id: Option<u64>,
    status: &'static str,
    reason: String,
    requested_pwm: Option<[i8; 2]>,
    objective_cost: Option<f64>,
    shadow_record_motor_packets_sent: Option<u64>,
    applied_actuation: Option<LiveAppliedActuationViz>,
    fault_actuation: Option<LiveFaultActuationViz>,
    diagnostic_warning: Option<String>,
    successful_solver_duration_ns: Option<u64>,
    kind: LiveNavigationVizMessageKind,
}

#[cfg(feature = "record")]
impl LiveNavigationVizMsg {
    fn with_control_tick_timing(mut self, timing: LiveControlTickTiming) -> Self {
        self.kind = match self.kind {
            LiveNavigationVizMessageKind::State { .. } => LiveNavigationVizMessageKind::State {
                control_tick_timing: Some(timing),
            },
            LiveNavigationVizMessageKind::ControlTickTimingOnly { .. } => {
                LiveNavigationVizMessageKind::ControlTickTimingOnly {
                    control_tick_timing: timing,
                }
            }
        };
        self
    }

    fn control_tick_timing_only(
        tick_sequence: i64,
        host_timestamp_ns: u64,
        timing: LiveControlTickTiming,
    ) -> Self {
        Self {
            tick_sequence,
            host_timestamp_ns: Some(host_timestamp_ns),
            goal: None,
            goal_state: String::new(),
            odometry_state: None,
            path: None,
            local_costmap: None,
            base_to_odom: None,
            odom_to_map: None,
            predicted_odom: None,
            decision_id: None,
            request_id: None,
            status: "control_tick_timing",
            reason: String::new(),
            requested_pwm: None,
            objective_cost: None,
            shadow_record_motor_packets_sent: None,
            applied_actuation: None,
            fault_actuation: None,
            diagnostic_warning: None,
            successful_solver_duration_ns: None,
            kind: LiveNavigationVizMessageKind::ControlTickTimingOnly {
                control_tick_timing: timing,
            },
        }
    }
}

#[cfg(feature = "record")]
struct LiveOdometryViz {
    state: Option<String>,
    base_to_odom: Option<[f64; 3]>,
    odom_to_map: Option<[f64; 3]>,
}

#[cfg(feature = "record")]
fn live_odometry_viz(
    coordinator: &ShadowNavigationCoordinator<NavigationIngressWriter<File>>,
) -> LiveOdometryViz {
    let current = coordinator.odometry().current();
    LiveOdometryViz {
        base_to_odom: current.map(|state| {
            let transform = state.base_to_odom();
            [
                transform.source_origin_x_in_destination_m(),
                transform.source_origin_y_in_destination_m(),
                transform.source_yaw_in_destination_rad(),
            ]
        }),
        odom_to_map: current.map(|state| {
            let transform = state.odom_to_map();
            [
                transform.source_origin_x_in_destination_m(),
                transform.source_origin_y_in_destination_m(),
                transform.source_yaw_in_destination_rad(),
            ]
        }),
        state: current.map(|state| {
            format!(
                "session_id={} segment_id={} device_timestamp_ns={} map_snapshot={:?} quality={:?}",
                state.session_id().as_u64(),
                state.segment_id().as_u64(),
                state.timestamp().as_nanos(),
                state.map_snapshot(),
                state.quality(),
            )
        }),
    }
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

    let odometry = live_odometry_viz(coordinator);

    let decision = outcome.decision();
    let applied = applied_actuation.is_some();
    let (kind, reason, objective_cost, successful_solver_duration_ns) = match decision.outcome() {
        SafetyDecisionOutcome::Controller(controller) => {
            let solve_status = controller.solve_status();
            (
                LiveDecisionVizKind::Controller,
                if applied {
                    "exact V2 controller application receipt matched; physical wheel motion remains unobserved".to_owned()
                } else {
                    "safety-approved shadow MPC request; no transport exists".to_owned()
                },
                Some(controller.objective_cost()),
                checked_monotonic_duration_ns(
                    u128::from(solve_status.started_at().as_nanos()),
                    u128::from(solve_status.observed_at().as_nanos()),
                ),
            )
        }
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
        host_timestamp_ns: Some(tick.as_nanos()),
        goal,
        goal_state: format!("{:?}", coordinator.goal_state()),
        odometry_state: odometry.state,
        path,
        local_costmap,
        base_to_odom: odometry.base_to_odom,
        odom_to_map: odometry.odom_to_map,
        predicted_odom,
        decision_id: Some(record.decision_id().as_u64()),
        request_id: decision.request_id().map(NonZeroU64::get),
        status,
        reason,
        requested_pwm: Some([pwm.left().get(), pwm.right().get()]),
        objective_cost,
        shadow_record_motor_packets_sent: Some(decision.motor_packets_sent().get()),
        applied_actuation,
        fault_actuation: None,
        diagnostic_warning: (!warnings.is_empty()).then(|| warnings.join("; ")),
        successful_solver_duration_ns,
        kind: LiveNavigationVizMessageKind::State {
            control_tick_timing: None,
        },
    }
}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    unix
))]
fn build_live_lifecycle_zero_viz_message(
    coordinator: &ShadowNavigationCoordinator<NavigationIngressWriter<File>>,
    tick_sequence: i64,
    applied: &LiveLifecycleZeroApplied<AppliedCommandReceipt>,
) -> LiveNavigationVizMsg {
    let odometry = live_odometry_viz(coordinator);
    LiveNavigationVizMsg {
        tick_sequence,
        host_timestamp_ns: Some(applied.requested_at().as_nanos()),
        goal: None,
        goal_state: format!("{:?}", coordinator.goal_state()),
        odometry_state: odometry.state,
        path: None,
        local_costmap: None,
        base_to_odom: odometry.base_to_odom,
        odom_to_map: odometry.odom_to_map,
        predicted_odom: None,
        decision_id: None,
        request_id: None,
        status: "lifecycle_zero_applied",
        reason: format!(
            "{:?}; exact V2 zero application receipt matched; no MPC outcome is attributed to this lifecycle transition",
            applied.reason()
        ),
        requested_pwm: Some([0, 0]),
        objective_cost: None,
        shadow_record_motor_packets_sent: None,
        applied_actuation: Some(live_applied_actuation_viz(applied.receipt(), None)),
        fault_actuation: None,
        diagnostic_warning: None,
        successful_solver_duration_ns: None,
        kind: LiveNavigationVizMessageKind::State {
            control_tick_timing: None,
        },
    }
}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    unix
))]
fn build_live_actuation_fault_viz_message(
    coordinator: &ShadowNavigationCoordinator<NavigationIngressWriter<File>>,
    tick_sequence: i64,
    observed_at: HostMonotonicTimestamp,
    evidence: LiveMotionActuationFaultEvidence,
) -> LiveNavigationVizMsg {
    let odometry = live_odometry_viz(coordinator);
    let controller_stop_confirmed =
        evidence.controller_stop() == AgentControllerStopKnowledge::Confirmed;
    LiveNavigationVizMsg {
        tick_sequence,
        host_timestamp_ns: Some(observed_at.as_nanos()),
        goal: None,
        goal_state: format!("{:?}", coordinator.goal_state()),
        odometry_state: odometry.state,
        path: None,
        local_costmap: None,
        base_to_odom: odometry.base_to_odom,
        odom_to_map: odometry.odom_to_map,
        predicted_odom: None,
        decision_id: None,
        request_id: None,
        status: if controller_stop_confirmed {
            "actuation_fault_stop_confirmed"
        } else {
            "actuation_fault_stop_uncertain"
        },
        reason: format!(
            "physical actuation fault kind={:?}; controller_stop={:?}; no MPC outcome or receipt is fabricated",
            evidence.kind(),
            evidence.controller_stop()
        ),
        requested_pwm: None,
        objective_cost: None,
        shadow_record_motor_packets_sent: None,
        applied_actuation: None,
        fault_actuation: Some(LiveFaultActuationViz {
            kind: format!("{:?}", evidence.kind()),
            controller_stop_confirmed,
        }),
        diagnostic_warning: (!controller_stop_confirmed).then(|| {
            "controller stop is uncertain; applied PWM is intentionally cleared".to_owned()
        }),
        successful_solver_duration_ns: None,
        kind: LiveNavigationVizMessageKind::State {
            control_tick_timing: None,
        },
    }
}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    unix
))]
fn build_live_terminal_controller_stop_viz_message(
    coordinator: &ShadowNavigationCoordinator<NavigationIngressWriter<File>>,
    tick_sequence: i64,
    timeline_at_ns: Option<u64>,
    controller_stop_confirmed: bool,
    status: &'static str,
    kind: String,
    reason: String,
) -> LiveNavigationVizMsg {
    let odometry = live_odometry_viz(coordinator);
    LiveNavigationVizMsg {
        tick_sequence,
        host_timestamp_ns: timeline_at_ns,
        goal: None,
        goal_state: format!("{:?}", coordinator.goal_state()),
        odometry_state: odometry.state,
        path: None,
        local_costmap: None,
        base_to_odom: odometry.base_to_odom,
        odom_to_map: odometry.odom_to_map,
        predicted_odom: None,
        decision_id: None,
        request_id: None,
        status,
        reason,
        requested_pwm: None,
        objective_cost: None,
        shadow_record_motor_packets_sent: None,
        applied_actuation: None,
        fault_actuation: Some(LiveFaultActuationViz {
            kind,
            controller_stop_confirmed,
        }),
        diagnostic_warning: (!controller_stop_confirmed).then(|| {
            "terminal controller stop is uncertain; applied PWM is intentionally cleared".to_owned()
        }),
        successful_solver_duration_ns: None,
        kind: LiveNavigationVizMessageKind::State {
            control_tick_timing: None,
        },
    }
}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    unix
))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LivePhysicalStateVizPublishError {
    DroppedNewest,
}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    unix
))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LiveNavigationVizPublishOutcome {
    AcceptedByBoundedQueue,
    ConsumerUnavailable,
}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    unix
))]
impl std::fmt::Display for LivePhysicalStateVizPublishError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(
            "the newest live-navigation visualization was not enqueued; timing or actuation diagnostics may remain stale",
        )
    }
}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    unix
))]
impl std::error::Error for LivePhysicalStateVizPublishError {}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    unix
))]
fn publish_live_navigation_viz_message(
    sender: &mut Option<DropSender<LiveNavigationVizMsg>>,
    message: LiveNavigationVizMsg,
) -> Result<LiveNavigationVizPublishOutcome, LivePhysicalStateVizPublishError> {
    let Some(active_sender) = sender.as_ref() else {
        return Ok(LiveNavigationVizPublishOutcome::ConsumerUnavailable);
    };
    match active_sender.try_send(message) {
        SendOutcome::Enqueued | SendOutcome::DroppedOldest => {
            Ok(LiveNavigationVizPublishOutcome::AcceptedByBoundedQueue)
        }
        SendOutcome::Disconnected => {
            *sender = None;
            Ok(LiveNavigationVizPublishOutcome::ConsumerUnavailable)
        }
        SendOutcome::DroppedNewest => Err(LivePhysicalStateVizPublishError::DroppedNewest),
    }
}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    unix
))]
fn publish_live_physical_state_viz(
    sender: &mut Option<DropSender<LiveNavigationVizMsg>>,
    coordinator: &ShadowNavigationCoordinator<NavigationIngressWriter<File>>,
    tick_sequence: i64,
    timing: LiveControlTickTiming,
    event: &LivePhysicalStateEvent<
        AppliedCommandReceipt,
        CoordinatorTickOutcome<NavigationIngressStreamWriteError>,
    >,
) -> Result<(), LivePhysicalStateVizPublishError> {
    let message = match event {
        LivePhysicalStateEvent::CoordinatorTick(applied) => build_live_navigation_viz_message(
            coordinator,
            applied.tick(),
            tick_sequence,
            applied.diagnostic(),
            Some(live_applied_actuation_viz(
                applied.receipt(),
                Some(applied.tick()),
            )),
        ),
        LivePhysicalStateEvent::LifecycleZero(applied) => {
            build_live_lifecycle_zero_viz_message(coordinator, tick_sequence, applied)
        }
        LivePhysicalStateEvent::ActuationFault {
            observed_at,
            evidence,
        } => build_live_actuation_fault_viz_message(
            coordinator,
            tick_sequence,
            *observed_at,
            *evidence,
        ),
    };
    publish_live_navigation_viz_message(sender, message.with_control_tick_timing(timing))
        .map(|_| ())
}

#[cfg(feature = "record")]
fn log_live_navigation_viz_message(
    recording: &rerun::RecordingStream,
    message: LiveNavigationVizMsg,
    context_logged: &mut bool,
) -> Result<(), VizLogError> {
    let host_timestamp_ns = message
        .host_timestamp_ns
        .and_then(|value| i64::try_from(value).ok());
    apply_live_rerun_timeline_domain(
        recording,
        LiveRerunTimelineDomain::Navigation {
            tick_sequence: message.tick_sequence,
            host_timestamp_ns,
        },
    )?;
    if let Some(timing) = message.kind.control_tick_timing() {
        recording.log(
            "navigation/control_loop/tick_lateness_ns",
            &rerun::Scalars::single(timing.current_lateness_ns as f64),
        )?;
        recording.log(
            "navigation/control_loop/maximum_tick_lateness_ns",
            &rerun::Scalars::single(timing.maximum_lateness_ns as f64),
        )?;
        recording.log(
            "navigation/control_loop/timing_evidence",
            &rerun::TextLog::new(format!(
                "current_tick_lateness_ns={} maximum_tick_lateness_ns={}",
                timing.current_lateness_ns, timing.maximum_lateness_ns
            )),
        )?;
    } else {
        for path in [
            "navigation/control_loop/tick_lateness_ns",
            "navigation/control_loop/maximum_tick_lateness_ns",
            "navigation/control_loop/timing_evidence",
        ] {
            recording.log(path, &rerun::Clear::flat())?;
        }
    }
    if !message.kind.updates_navigation_state() {
        return Ok(());
    }
    if let Some(duration_ns) = message.successful_solver_duration_ns {
        recording.log(
            "navigation/control_loop/successful_solver_duration_ns",
            &rerun::Scalars::single(duration_ns as f64),
        )?;
    } else {
        recording.log(
            "navigation/control_loop/successful_solver_duration_ns",
            &rerun::Clear::flat(),
        )?;
    }

    let local_costmap_to_odom = message
        .local_costmap
        .as_ref()
        .and_then(|local| local.local_costmap_to_odom);
    let local_costmap_evidence = message
        .local_costmap
        .as_ref()
        .map(|local| local.evidence.clone());
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

    if let Some(requested_pwm) = message.requested_pwm {
        for (path, value) in [
            (
                "navigation/decision/left_pwm_percent",
                f64::from(requested_pwm[0]),
            ),
            (
                "navigation/decision/right_pwm_percent",
                f64::from(requested_pwm[1]),
            ),
        ] {
            recording.log(path, &rerun::Scalars::single(value))?;
        }
    } else {
        for path in [
            "navigation/decision/left_pwm_percent",
            "navigation/decision/right_pwm_percent",
        ] {
            recording.log(path, &rerun::Clear::flat())?;
        }
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
        for (path, duration_ns) in [
            (
                "navigation/actuation/conservative_decision_to_send_ns",
                applied.conservative_decision_to_send_ns,
            ),
            (
                "navigation/actuation/command_send_to_ack_ns",
                applied.command_send_to_ack_ns,
            ),
            (
                "navigation/actuation/conservative_decision_to_ack_ns",
                applied.conservative_decision_to_ack_ns,
            ),
        ] {
            if let Some(duration_ns) = duration_ns {
                recording.log(path, &rerun::Scalars::single(duration_ns as f64))?;
            } else {
                recording.log(path, &rerun::Clear::flat())?;
            }
        }
        recording.log(
            "navigation/actuation/applied_receipt",
            &rerun::TextLog::new(format!(
                "boot_id={} control_epoch={} sequence={} conservative_decision_to_send_ns={} command_send_to_ack_ns={} conservative_decision_to_ack_ns={} acknowledged_at_host_ns={} known_active_through_host_ns={}",
                applied.controller_boot_id,
                applied.control_epoch,
                applied.sequence,
                applied
                    .conservative_decision_to_send_ns
                    .map_or_else(|| "unavailable".to_owned(), |value| value.to_string()),
                applied
                    .command_send_to_ack_ns
                    .map_or_else(|| "unavailable".to_owned(), |value| value.to_string()),
                applied
                    .conservative_decision_to_ack_ns
                    .map_or_else(|| "unavailable".to_owned(), |value| value.to_string()),
                applied.acknowledged_at_ns_decimal,
                applied.known_active_through_ns_decimal,
            )),
        )?;
        recording.log(
            "navigation/actuation/fault_stop_evidence",
            &rerun::Clear::flat(),
        )?;
    } else if let Some(fault) = message.fault_actuation.as_ref() {
        recording.log("navigation/actuation", &rerun::Clear::recursive())?;
        if fault.controller_stop_confirmed {
            for path in [
                "navigation/actuation/applied_left_pwm_percent",
                "navigation/actuation/applied_right_pwm_percent",
            ] {
                recording.log(path, &rerun::Scalars::single(0.0))?;
            }
        }
        recording.log(
            "navigation/actuation/fault_stop_evidence",
            &rerun::TextLog::new(format!(
                "kind={} controller_stop_confirmed={}",
                fault.kind, fault.controller_stop_confirmed
            )),
        )?;
    } else {
        recording.log("navigation/actuation", &rerun::Clear::recursive())?;
    }
    if let Some(motor_packets_sent) = message
        .shadow_record_motor_packets_sent
        .and_then(|value| u32::try_from(value).ok())
    {
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
        message
            .host_timestamp_ns
            .map_or_else(|| "unavailable".to_owned(), |value| value.to_string()),
        message
            .decision_id
            .map_or_else(|| "none".to_owned(), |id| id.to_string()),
        message
            .request_id
            .map_or_else(|| "none".to_owned(), |id| id.to_string()),
        message.status,
        message.goal_state,
        message
            .shadow_record_motor_packets_sent
            .map_or_else(|| "none".to_owned(), |count| count.to_string()),
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
fn log_live_rgb_viz_message(
    recording: &rerun::RecordingStream,
    message: LiveRgbVizMsg,
) -> Result<LiveRgbFrameKey, VizLogError> {
    let frame_key = message.frame_key();
    apply_live_rerun_timeline_domain(
        recording,
        LiveRerunTimelineDomain::Rgb {
            capture_timestamp_ns: message.device_timestamp_ns,
            device_capture_sequence: message.device_capture_sequence,
            host_delivery_sequence: message.host_delivery_sequence,
        },
    )?;
    let image = rerun::Image::from_color_model_and_bytes(
        message.pixels_bgr8,
        [message.width, message.height],
        rerun::ColorModel::BGR,
        rerun::ChannelDatatype::U8,
    );
    // A face overlay is valid only for one exact RGB key. Clear the prior
    // subtree before replacing the image so a dropped/disconnected face
    // diagnostic cannot remain painted over a newer capture.
    recording.log("view/rgb/face", &rerun::Clear::recursive())?;
    recording.log("view/rgb", &image)?;
    let timestamp_reference = match message.timestamp_reference {
        CameraTimestampReference::ExposureMidpoint => "exposure_midpoint",
    };
    recording.log(
        "view/rgb/provenance",
        &rerun::TextLog::new(format!(
            "stream=rgb stream_epoch={} device_capture_sequence={} host_delivery_sequence={} capture_timestamp_ns={} timestamp_reference={} width_px={} height_px={} layout=tightly_packed_bgr8",
            message.stream_epoch,
            message.device_capture_sequence,
            message.host_delivery_sequence,
            message.device_timestamp_ns,
            timestamp_reference,
            message.width,
            message.height,
        )),
    )?;
    Ok(frame_key)
}

#[cfg(feature = "record")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LiveRerunTimelineDomain {
    Capture,
    Rgb {
        capture_timestamp_ns: i64,
        device_capture_sequence: i64,
        host_delivery_sequence: i64,
    },
    Navigation {
        tick_sequence: i64,
        host_timestamp_ns: Option<i64>,
    },
    #[cfg(all(feature = "nano-agent", unix))]
    Face {
        capture_timestamp_ns: i64,
        device_capture_sequence: i64,
        host_delivery_sequence: i64,
        detector_result_sequence: i64,
    },
}

#[cfg(feature = "record")]
trait LiveRerunTimelineTarget {
    fn reset_live_time(&self);
    fn set_live_time(&self, timeline: &'static str, time: rerun::TimeCell);
    fn set_live_sequence(&self, timeline: &'static str, sequence: i64);
}

#[cfg(feature = "record")]
impl LiveRerunTimelineTarget for rerun::RecordingStream {
    fn reset_live_time(&self) {
        self.reset_time();
    }

    fn set_live_time(&self, timeline: &'static str, time: rerun::TimeCell) {
        self.set_time(timeline, time);
    }

    fn set_live_sequence(&self, timeline: &'static str, sequence: i64) {
        self.set_time_sequence(timeline, sequence);
    }
}

#[cfg(feature = "record")]
fn checked_live_rerun_time(timestamp_ns: i64) -> Result<rerun::TimeCell, VizLogError> {
    let time = rerun::TimeCell::from_duration_nanos(timestamp_ns);
    if time.as_i64() != timestamp_ns {
        return Err(VizLogError::TimestampUnrepresentable {
            timestamp_ns,
            encoded_ns: time.as_i64(),
        });
    }
    Ok(time)
}

/// Selects one live diagnostic clock domain without retaining time columns
/// from the preceding message handled by the shared Rerun thread.
///
/// Rerun timeline state is sticky and thread-local. Every live message family
/// therefore resets the complete thread-local timepoint before setting exactly
/// the clocks that describe its own evidence.
#[cfg(feature = "record")]
fn apply_live_rerun_timeline_domain(
    target: &impl LiveRerunTimelineTarget,
    domain: LiveRerunTimelineDomain,
) -> Result<(), VizLogError> {
    let capture_time = match domain {
        LiveRerunTimelineDomain::Rgb {
            capture_timestamp_ns,
            ..
        } => Some(checked_live_rerun_time(capture_timestamp_ns)?),
        #[cfg(all(feature = "nano-agent", unix))]
        LiveRerunTimelineDomain::Face {
            capture_timestamp_ns,
            ..
        } => Some(checked_live_rerun_time(capture_timestamp_ns)?),
        LiveRerunTimelineDomain::Navigation {
            host_timestamp_ns: Some(host_timestamp_ns),
            ..
        } => Some(checked_live_rerun_time(host_timestamp_ns)?),
        LiveRerunTimelineDomain::Capture
        | LiveRerunTimelineDomain::Navigation {
            host_timestamp_ns: None,
            ..
        } => None,
    };

    target.reset_live_time();
    match domain {
        LiveRerunTimelineDomain::Capture => {}
        LiveRerunTimelineDomain::Rgb {
            device_capture_sequence,
            host_delivery_sequence,
            ..
        } => {
            target.set_live_time(
                "capture_ns",
                capture_time.expect("RGB domain validates one capture timestamp"),
            );
            target.set_live_sequence("oak_rgb_capture_sequence", device_capture_sequence);
            target.set_live_sequence("oak_rgb_host_delivery_sequence", host_delivery_sequence);
        }
        LiveRerunTimelineDomain::Navigation {
            tick_sequence,
            host_timestamp_ns,
        } => {
            target.set_live_sequence("navigation_tick", tick_sequence);
            if host_timestamp_ns.is_some() {
                target.set_live_time(
                    "navigation_host_ns",
                    capture_time.expect("present navigation timestamp was validated"),
                );
            }
        }
        #[cfg(all(feature = "nano-agent", unix))]
        LiveRerunTimelineDomain::Face {
            device_capture_sequence,
            host_delivery_sequence,
            detector_result_sequence,
            ..
        } => {
            target.set_live_time(
                "capture_ns",
                capture_time.expect("face domain validates one capture timestamp"),
            );
            target.set_live_sequence("oak_rgb_capture_sequence", device_capture_sequence);
            target.set_live_sequence("oak_rgb_host_delivery_sequence", host_delivery_sequence);
            target.set_live_sequence("face_detector_result_sequence", detector_result_sequence);
        }
    }
    Ok(())
}

#[cfg(all(feature = "record", feature = "nano-agent", unix))]
fn live_face_source_name(source: FaceDetectorSource) -> &'static str {
    match source {
        FaceDetectorSource::Frontal => "frontal",
        FaceDetectorSource::Profile => "profile",
        FaceDetectorSource::MirroredProfile => "mirrored_profile",
    }
}

#[cfg(all(feature = "record", feature = "nano-agent", unix))]
fn live_face_source_color(source: FaceDetectorSource) -> rerun::Color {
    match source {
        FaceDetectorSource::Frontal => rerun::Color::from_rgb(46, 196, 182),
        FaceDetectorSource::Profile => rerun::Color::from_rgb(255, 159, 28),
        FaceDetectorSource::MirroredProfile => rerun::Color::from_rgb(131, 56, 236),
    }
}

#[cfg(all(feature = "record", feature = "nano-agent", unix))]
fn live_face_admission_text(admission: FaceResultAdmission) -> String {
    match admission {
        FaceResultAdmission::ColdStart => "ColdStart".to_owned(),
        FaceResultAdmission::Consecutive { previous, actual } => format!(
            "Consecutive(previous={},actual={})",
            previous.get(),
            actual.get()
        ),
        FaceResultAdmission::ForwardGap {
            previous,
            actual,
            skipped_result_count,
        } => format!(
            "ForwardGap(previous={},actual={},skipped_detector_results={})",
            previous.get(),
            actual.get(),
            skipped_result_count.get()
        ),
    }
}

#[cfg(all(feature = "record", feature = "nano-agent", unix))]
fn live_face_state_text(state: FaceTargetState) -> String {
    match state {
        FaceTargetState::NoTarget => "NoTarget".to_owned(),
        FaceTargetState::Acquiring(target) => format!(
            "Acquiring(frame_epoch=0x{:016x},frame_sequence={},consecutive_results={},required_results={})",
            target.frame_id().stream_epoch().get(),
            target.frame_id().sequence(),
            target.consecutive_results().get(),
            target.required_results().get(),
        ),
        FaceTargetState::Tracked(observation) => format!(
            "Tracked(track_id=0x{:016x},frame_epoch=0x{:016x},frame_sequence={})",
            observation.track_id().get(),
            observation.frame_id().stream_epoch().get(),
            observation.frame_id().sequence(),
        ),
        FaceTargetState::Coasting(target) => {
            let last = target.last_observation();
            format!(
                "Coasting(track_id=0x{:016x},last_frame_epoch=0x{:016x},last_frame_sequence={},evaluated_frame_epoch=0x{:016x},evaluated_frame_sequence={},loss_deadline_accessory_ns={})",
                last.track_id().get(),
                last.frame_id().stream_epoch().get(),
                last.frame_id().sequence(),
                target.evaluated_frame_id().stream_epoch().get(),
                target.evaluated_frame_id().sequence(),
                target.loss_deadline().timestamp().nanos_since_epoch(),
            )
        }
        FaceTargetState::Lost(target) => {
            let last = target.last_observation();
            format!(
                "Lost(track_id=0x{:016x},last_frame_epoch=0x{:016x},last_frame_sequence={},evaluated_frame_epoch=0x{:016x},evaluated_frame_sequence={},loss_deadline_accessory_ns={})",
                last.track_id().get(),
                last.frame_id().stream_epoch().get(),
                last.frame_id().sequence(),
                target.evaluated_frame_id().stream_epoch().get(),
                target.evaluated_frame_id().sequence(),
                target.loss_deadline().timestamp().nanos_since_epoch(),
            )
        }
        FaceTargetState::Switched(target) => {
            let observation = target.observation();
            format!(
                "Switched(previous_track_id=0x{:016x},track_id=0x{:016x},frame_epoch=0x{:016x},frame_sequence={})",
                target.previous_track_id().get(),
                observation.track_id().get(),
                observation.frame_id().stream_epoch().get(),
                observation.frame_id().sequence(),
            )
        }
    }
}

#[cfg(all(feature = "record", feature = "nano-agent", unix))]
fn live_face_current_target(
    update: FaceTrackingUpdate,
) -> Option<(ImagePoint, &'static str, rerun::Color)> {
    match update.state() {
        FaceTargetState::NoTarget | FaceTargetState::Coasting(_) | FaceTargetState::Lost(_) => None,
        FaceTargetState::Acquiring(target) => Some((
            target.detection().center(),
            "acquiring",
            rerun::Color::from_rgb(255, 214, 10),
        )),
        FaceTargetState::Tracked(observation) => Some((
            observation.center(),
            "tracked",
            rerun::Color::from_rgb(42, 157, 143),
        )),
        FaceTargetState::Switched(target) => Some((
            target.observation().center(),
            "switched",
            rerun::Color::from_rgb(0, 180, 216),
        )),
    }
}

#[cfg(all(feature = "record", feature = "nano-agent", unix))]
fn live_face_point_pixels(point: ImagePoint, frame_key: LiveRgbFrameKey) -> [f32; 2] {
    let width = frame_key.width as f32;
    let height = frame_key.height as f32;
    [
        f32::from(point.x_right().basis_points()) * width / 10_000.0,
        f32::from(point.y_down().basis_points()) * height / 10_000.0,
    ]
}

#[cfg(all(feature = "record", feature = "nano-agent", unix))]
fn live_face_detection_label(detection: FaceDetection) -> String {
    format!(
        "{} opaque_rank_bits=0x{:016x}",
        live_face_source_name(detection.source()),
        detection.detector_level_weight().to_bits(),
    )
}

#[cfg(all(feature = "record", feature = "nano-agent", unix))]
fn log_live_head_gaze_diagnostic(
    recording: &rerun::RecordingStream,
    diagnostic: NanoHeadGazeDiagnostic,
) -> Result<(), VizLogError> {
    let actuation = NanoHeadGazeActuationAvailability::UnavailableWithoutBaseZeroExclusiveLease;
    debug_assert_eq!(diagnostic.actuation(), actuation);
    match diagnostic {
        NanoHeadGazeDiagnostic::DisabledNoPolicy { actuation } => {
            recording.log(
                "diagnostics/head_gaze/status",
                &rerun::TextLog::new(format!(
                    "policy=absent proposal=disabled actuation={actuation:?}"
                )),
            )?;
            recording.log(
                "diagnostics/head_gaze/proposal_target_ticks",
                &rerun::Clear::flat(),
            )?;
            recording.log(
                "diagnostics/head_gaze/camera_ray_unit_direction",
                &rerun::Clear::flat(),
            )?;
        }
        NanoHeadGazeDiagnostic::ProposalOnly {
            evaluated_at,
            projection,
            outcome,
            actuation,
        } => {
            let [fx, fy, cx, cy] = projection.coefficients_px();
            let [width, height] = projection.dimensions_px();
            recording.log(
                "diagnostics/head_gaze/rgb_intrinsics_px",
                &rerun::Scalars::new([f64::from(fx), f64::from(fy), f64::from(cx), f64::from(cy)]),
            )?;
            match outcome {
                Ok(HeadGazeFaceProposalOutcome::Proposed(proposal)) => {
                    let positions = proposal.target().positions();
                    recording.log(
                        "diagnostics/head_gaze/status",
                        &rerun::TextLog::new(format!(
                            "policy=proposal_only outcome=proposed evaluated_at_accessory_ns={} rgb_grid={}x{} track_id=0x{:016x} transition={:?} actuation={actuation:?}",
                            evaluated_at.nanos_since_epoch(),
                            width,
                            height,
                            proposal.face().track_id().get(),
                            proposal.face().transition(),
                        )),
                    )?;
                    recording.log(
                        "diagnostics/head_gaze/proposal_target_ticks",
                        &rerun::Scalars::new(positions.map(|position| f64::from(position.get()))),
                    )?;
                    recording.log(
                        "diagnostics/head_gaze/camera_ray_unit_direction",
                        &rerun::Scalars::new(proposal.camera_ray().unit_direction()),
                    )?;
                }
                Ok(HeadGazeFaceProposalOutcome::Withheld(reason)) => {
                    recording.log(
                        "diagnostics/head_gaze/status",
                        &rerun::TextLog::new(format!(
                            "policy=proposal_only outcome=withheld reason={reason:?} evaluated_at_accessory_ns={} rgb_grid={}x{} actuation={actuation:?}",
                            evaluated_at.nanos_since_epoch(),
                            width,
                            height,
                        )),
                    )?;
                    recording.log(
                        "diagnostics/head_gaze/proposal_target_ticks",
                        &rerun::Clear::flat(),
                    )?;
                    recording.log(
                        "diagnostics/head_gaze/camera_ray_unit_direction",
                        &rerun::Clear::flat(),
                    )?;
                }
                Err(source) => {
                    recording.log(
                        "diagnostics/head_gaze/status",
                        &rerun::TextLog::new(format!(
                            "policy=proposal_only outcome=rejected source={source:?} evaluated_at_accessory_ns={} rgb_grid={}x{} actuation={actuation:?}",
                            evaluated_at.nanos_since_epoch(),
                            width,
                            height,
                        )),
                    )?;
                    recording.log(
                        "diagnostics/head_gaze/proposal_target_ticks",
                        &rerun::Clear::flat(),
                    )?;
                    recording.log(
                        "diagnostics/head_gaze/camera_ray_unit_direction",
                        &rerun::Clear::flat(),
                    )?;
                }
            }
        }
    }
    Ok(())
}

#[cfg(all(feature = "record", feature = "nano-agent", unix))]
fn log_live_face_viz_message(
    recording: &rerun::RecordingStream,
    message: NanoFaceDiagnosticFrame,
    last_logged_rgb: Option<LiveRgbFrameKey>,
    context_logged: &mut bool,
) -> Result<Result<bool, LiveFaceVizBuildError>, VizLogError> {
    let validated = match validate_live_face_viz(message) {
        Ok(validated) => validated,
        Err(source) => return Ok(Err(source)),
    };
    apply_live_rerun_timeline_domain(
        recording,
        LiveRerunTimelineDomain::Face {
            capture_timestamp_ns: validated.frame_key.device_timestamp_ns,
            device_capture_sequence: validated.frame_key.device_capture_sequence,
            host_delivery_sequence: validated.frame_key.host_delivery_sequence,
            detector_result_sequence: validated.detector_result_sequence,
        },
    )?;
    if !*context_logged {
        recording.log_static(
            "diagnostics/face/contract",
            &rerun::TextLog::new(
                "Face rectangles and head-gaze proposals are best-effort diagnostics from a classical Haar detector. A proposal-only policy never actuates the head: physical servicing remains unavailable without the sole-base-owner exact-zero exclusive lease. opaque_rank is not confidence, probability, identity, range, occupancy, collision evidence, or navigation authority. Accessory monotonic timestamps use a distinct process-local clock origin and are never subtracted from OAK or navigation clocks.",
            ),
        )?;
        *context_logged = true;
    }

    let output = message.output();
    let batch = output.batch();
    let tracking = output.tracking();
    let timestamp_reference = match validated.frame_key.timestamp_reference {
        CameraTimestampReference::ExposureMidpoint => "exposure_midpoint",
    };
    recording.log(
        "diagnostics/face/status",
        &rerun::TextLog::new(format!(
            "stream=rgb stream_epoch=0x{:016x} device_capture_sequence={} host_delivery_sequence={} capture_timestamp_ns={} timestamp_reference={} layout=tightly_packed_bgr8 width_px={} height_px={} detector_result_sequence={} retained_count={} detector_truncated_count={} admission={} state={} accessory_clock_domain=tokio_process_local_monotonic accessory_observed_at_ns={} accessory_source_deadline_exclusive_ns={}",
            validated.frame_key.stream_epoch,
            validated.frame_key.device_capture_sequence,
            validated.frame_key.host_delivery_sequence,
            validated.frame_key.device_timestamp_ns,
            timestamp_reference,
            validated.frame_key.width,
            validated.frame_key.height,
            validated.detector_result_sequence,
            batch.retained_count(),
            batch.detector_truncated_count(),
            live_face_admission_text(tracking.admission()),
            live_face_state_text(tracking.state()),
            message
                .accessory_observed_at()
                .nanos_since_epoch(),
            message
                .accessory_source_deadline()
                .timestamp()
                .nanos_since_epoch(),
        )),
    )?;
    recording.log(
        "diagnostics/face/retained_count",
        &rerun::Scalars::single(batch.retained_count() as f64),
    )?;
    recording.log(
        "diagnostics/face/detector_truncated_count",
        &rerun::Scalars::single(f64::from(batch.detector_truncated_count())),
    )?;
    log_live_head_gaze_diagnostic(recording, message.head_gaze())?;

    let overlay_matched = last_logged_rgb == Some(validated.frame_key);
    recording.log(
        "diagnostics/face/overlay_exact_rgb_match",
        &rerun::Scalars::single(if overlay_matched { 1.0 } else { 0.0 }),
    )?;
    if overlay_matched {
        if batch.is_empty() {
            recording.log("view/rgb/face/detections", &rerun::Clear::flat())?;
        } else {
            let mut minimums = Vec::with_capacity(batch.retained_count());
            let mut sizes = Vec::with_capacity(batch.retained_count());
            let mut colors = Vec::with_capacity(batch.retained_count());
            let mut labels = Vec::with_capacity(batch.retained_count());
            for detection in batch.iter() {
                let rectangle = detection.rectangle();
                minimums.push([rectangle.left_px() as f32, rectangle.top_px() as f32]);
                sizes.push([rectangle.width_px() as f32, rectangle.height_px() as f32]);
                colors.push(live_face_source_color(detection.source()));
                labels.push(live_face_detection_label(detection));
            }
            recording.log(
                "view/rgb/face/detections",
                &rerun::Boxes2D::from_mins_and_sizes(minimums, sizes)
                    .with_colors(colors)
                    .with_labels(labels)
                    .with_draw_order(10.0),
            )?;
        }
        if let Some((point, label, color)) = live_face_current_target(tracking) {
            recording.log(
                "view/rgb/face/current_target",
                &rerun::Points2D::new([live_face_point_pixels(point, validated.frame_key)])
                    .with_colors([color])
                    .with_labels([label])
                    .with_radii([5.0])
                    .with_draw_order(11.0),
            )?;
        } else {
            // Coasting and Lost refer to prior-frame evidence and must not be
            // drawn as a target on the current frame.
            recording.log("view/rgb/face/current_target", &rerun::Clear::flat())?;
        }
    } else {
        // Rerun images are decimated and independently queued. Never draw a
        // face batch over the most recently displayed but different capture.
        recording.log("view/rgb/face", &rerun::Clear::recursive())?;
    }
    Ok(Ok(overlay_matched))
}

#[cfg(feature = "record")]
fn log_live_viz_message(
    recording: &rerun::RecordingStream,
    sink: &mut RerunSink,
    msg: LiveVizMsg,
    slam_context_logged: &mut bool,
) -> Result<(), VizLogError> {
    apply_live_rerun_timeline_domain(recording, LiveRerunTimelineDomain::Capture)?;
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
    if !*slam_context_logged {
        recording.log_static(
            "diagnostics/slam/inference_runtime",
            &rerun::TextLog::new(format!(
                "superpoint_requested={} superpoint_selected={} lightglue_requested={} lightglue_selected={}; selected is runtime session evidence, not a speed or utilization claim",
                live_inference_backend_name(msg.slam.inference.superpoint_requested),
                live_selected_inference_backend_name(msg.slam.inference.superpoint_selected),
                live_inference_backend_name(msg.slam.inference.lightglue_requested),
                live_selected_inference_backend_name(msg.slam.inference.lightglue_selected),
            )),
        )?;
        recording.log_static(
            "diagnostics/slam/rate_contract",
            &rerun::TextLog::new(
                "successful_rate_window_hz is derived from at most 64 actual successful tracker completion timestamps: (count-1)*1e9/span_ns. It is neither the configured camera FPS nor a benchmark claim.",
            ),
        )?;
        *slam_context_logged = true;
    }
    for (path, value) in [
        ("diagnostics/slam/started_pairs", msg.slam.started_pairs),
        (
            "diagnostics/slam/successful_pairs",
            msg.slam.successful_pairs,
        ),
        (
            "diagnostics/slam/recoverable_failures",
            msg.slam.recoverable_failures,
        ),
        ("diagnostics/slam/fatal_failures", msg.slam.fatal_failures),
    ] {
        recording.log(path, &rerun::Scalars::single(value as f64))?;
    }
    recording.log(
        "diagnostics/slam/status",
        &rerun::TextLog::new(format!(
            "pipeline_state={:?} started_pairs={} successful_pairs={} recoverable_failures={} fatal_failures={} last_successful_source_arrival_host_monotonic_ns={} last_successful_completion_host_monotonic_ns={}",
            msg.slam.pipeline_state,
            msg.slam.started_pairs,
            msg.slam.successful_pairs,
            msg.slam.recoverable_failures,
            msg.slam.fatal_failures,
            msg.slam
                .last_successful_source_arrival
                .map_or_else(|| "unavailable".to_owned(), |value| value.as_nanos().to_string()),
            msg.slam
                .last_successful_completion
                .map_or_else(|| "unavailable".to_owned(), |value| value.as_nanos().to_string()),
        )),
    )?;
    if let Some(window) = msg.slam.rate_window {
        let rate_hz = f64::from(window.successful_completions.saturating_sub(1)) * 1e9
            / window.span_ns as f64;
        recording.log(
            "diagnostics/slam/successful_rate_window_hz",
            &rerun::Scalars::single(rate_hz),
        )?;
        recording.log(
            "diagnostics/slam/rate_window_span_ns",
            &rerun::Scalars::single(window.span_ns as f64),
        )?;
        recording.log(
            "diagnostics/slam/rate_window_successful_completions",
            &rerun::Scalars::single(f64::from(window.successful_completions)),
        )?;
    } else {
        for path in [
            "diagnostics/slam/successful_rate_window_hz",
            "diagnostics/slam/rate_window_span_ns",
            "diagnostics/slam/rate_window_successful_completions",
        ] {
            recording.log(path, &rerun::Clear::flat())?;
        }
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

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    unix
))]
type ProductionLiveMotionOwner =
    LiveMotionOwner<NavigationIngressWriter<File>, LiveMpcControlDriver, InstantHostClock>;

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    unix
))]
type LiveProductionMotionStartFailure = Box<(
    ShadowNavigationCoordinator<NavigationIngressWriter<File>>,
    LiveProductionOwnerStartError,
)>;

/// Motion ownership selected before the navigation worker starts.
///
/// The compatibility variant retains the standalone CLI behavior. The
/// production variant accepts only the already-admitted bundle; this binary
/// does not reconstruct admission from paths or environment values.
#[cfg(all(feature = "record", feature = "actuation"))]
enum LiveNavigationWorkerMotion {
    Compatibility(Box<Option<NavigationActuationConfigV1>>),
    #[cfg(all(feature = "agent-runtime", unix))]
    #[allow(
        dead_code,
        reason = "constructed by the Nano launch owner once launch admission is wired"
    )]
    Production(Box<LiveProductionMotionInput>),
    #[cfg(all(feature = "nano-attended-navigation-trial", unix))]
    AttendedNavigationTrial(Box<LiveAttendedNavigationTrialMotionInput>),
    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    WheelsOffQualification(Box<LiveWheelsOffQualificationMotionInput>),
}

#[cfg(all(feature = "nano-attended-navigation-trial", unix))]
struct LiveAttendedNavigationTrialMotionInput {
    admission: Option<AttendedNavigationTrialMotionAdmission>,
    accessory_health: NanoAccessoryHealthObserver,
    rerun_diagnostics_url: ConsoleRerunDiagnosticsUrl,
}

#[cfg(all(feature = "nano-attended-navigation-trial", unix))]
impl LiveAttendedNavigationTrialMotionInput {
    fn new(
        admission: AttendedNavigationTrialMotionAdmission,
        accessory_health: NanoAccessoryHealthObserver,
        rerun_diagnostics_url: ConsoleRerunDiagnosticsUrl,
    ) -> Self {
        Self {
            admission: Some(admission),
            accessory_health,
            rerun_diagnostics_url,
        }
    }

    fn head_gaze_lease_issuer(
        &self,
    ) -> Option<&kiko_head_runtime::HeadGazeBaseZeroExclusiveLeaseIssuer> {
        self.admission
            .as_ref()
            .map(|admission| &admission.head_gaze_lease_issuer)
    }

    fn take_for_owner(
        &mut self,
    ) -> (
        AttendedNavigationTrialMotionAdmission,
        NanoAccessoryHealthObserver,
        ConsoleRerunDiagnosticsUrl,
    ) {
        (
            self.admission
                .take()
                .expect("attended motion admission transfers exactly once"),
            self.accessory_health.clone(),
            self.rerun_diagnostics_url,
        )
    }
}

#[cfg(all(feature = "nano-attended-navigation-trial", unix))]
impl Drop for LiveAttendedNavigationTrialMotionInput {
    fn drop(&mut self) {
        let Some(mut admission) = self.admission.take() else {
            return;
        };
        match admission.driver.disarm() {
            Ok(receipt) => eprintln!(
                "unused attended navigation admission explicitly disarmed: boot_id={} request_id={}",
                receipt.observed_boot_id().get(),
                receipt.request_id().get(),
            ),
            Err(source) => eprintln!(
                "unused attended navigation admission could not prove controller stop: {source}"
            ),
        }
    }
}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    unix
))]
struct LiveProductionMotionInput {
    prepared: Option<PreparedNanoProductionRuntime>,
    coordinator_actuation_config: NavigationActuationConfigV1,
    accessory_health: NanoAccessoryHealthObserver,
    #[cfg(all(feature = "nano-agent", feature = "operator-console"))]
    rerun_diagnostics_url: ConsoleRerunDiagnosticsUrl,
}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    unix
))]
impl LiveProductionMotionInput {
    #[cfg(feature = "nano-agent")]
    fn from_admitted(
        prepared: PreparedNanoProductionRuntime,
        accessory_health: NanoAccessoryHealthObserver,
        rerun_diagnostics_url: ConsoleRerunDiagnosticsUrl,
    ) -> Self {
        let coordinator_actuation_config = prepared.actuation().config().clone();
        Self {
            prepared: Some(prepared),
            coordinator_actuation_config,
            accessory_health,
            rerun_diagnostics_url,
        }
    }

    #[cfg(all(feature = "nano-agent", feature = "operator-console"))]
    const fn rerun_diagnostics_url(&self) -> ConsoleRerunDiagnosticsUrl {
        self.rerun_diagnostics_url
    }

    #[cfg(feature = "nano-agent")]
    fn head_gaze_lease_issuer(
        &self,
    ) -> Option<&kiko_head_runtime::HeadGazeBaseZeroExclusiveLeaseIssuer> {
        self.prepared
            .as_ref()
            .map(PreparedNanoProductionRuntime::head_gaze_lease_issuer)
    }

    fn take_for_owner(&mut self) -> (LiveAgentMotionStartInput, NanoAccessoryHealthObserver) {
        let PreparedNanoProductionRuntimeParts {
            startup,
            actuation,
            physical_driver,
            initial_zero: _,
            head_gaze_lease_issuer: _,
        } = self
            .prepared
            .take()
            .expect("production admission is transferred exactly once")
            .into_parts();
        let admitted_actuation_config = actuation.config().clone();
        (
            LiveAgentMotionStartInput {
                policy: startup.policy,
                authority: startup.authority,
                physical_driver,
                admitted_actuation_config: Some(admitted_actuation_config),
                coordinator_actuation_config: Some(self.coordinator_actuation_config.clone()),
                kind: LiveAgentAuthorityKind::Production,
            },
            self.accessory_health.clone(),
        )
    }
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
struct LiveWheelsOffQualificationMotionInput {
    stopped_controller: Option<kiko_slam::navigation::StoppedWheelsOffCandidateController>,
    initial_zero: Option<AppliedCommandReceipt>,
    initial_stop: Option<DisarmReceipt>,
    limits: kiko_slam::navigation::WheelsOffCandidateLimits,
    runtime_service_interval: kiko_slam::navigation::WheelsOffCandidateRuntimeServiceInterval,
    preflight: Option<AttendedWheelsOffPreflight>,
    profile: kiko_slam::navigation::WheelsOffQualificationControlProfile,
    frontend_config: kiko_slam::navigation::WheelsOffQualificationFrontendConfig,
    initial_health: ConsoleSubsystemHealth,
    accessory_health: NanoAccessoryHealthObserver,
    rerun_diagnostics_url: ConsoleRerunDiagnosticsUrl,
    fault_injection: Option<kiko_slam::navigation::WheelsOffQualificationFaultInjection>,
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
impl LiveWheelsOffQualificationMotionInput {
    #[allow(clippy::too_many_arguments)]
    fn new(
        stopped_controller: kiko_slam::navigation::StoppedWheelsOffCandidateController,
        initial_zero: AppliedCommandReceipt,
        initial_stop: DisarmReceipt,
        limits: kiko_slam::navigation::WheelsOffCandidateLimits,
        runtime_service_interval: kiko_slam::navigation::WheelsOffCandidateRuntimeServiceInterval,
        preflight: AttendedWheelsOffPreflight,
        profile: kiko_slam::navigation::WheelsOffQualificationControlProfile,
        frontend_config: kiko_slam::navigation::WheelsOffQualificationFrontendConfig,
        initial_health: ConsoleSubsystemHealth,
        accessory_health: NanoAccessoryHealthObserver,
        rerun_diagnostics_url: ConsoleRerunDiagnosticsUrl,
        fault_injection: Option<kiko_slam::navigation::WheelsOffQualificationFaultInjection>,
    ) -> Self {
        Self {
            stopped_controller: Some(stopped_controller),
            initial_zero: Some(initial_zero),
            initial_stop: Some(initial_stop),
            limits,
            runtime_service_interval,
            preflight: Some(preflight),
            profile,
            frontend_config,
            initial_health,
            accessory_health,
            rerun_diagnostics_url,
            fault_injection,
        }
    }

    #[allow(clippy::type_complexity)]
    fn take_for_owner(
        &mut self,
    ) -> (
        kiko_slam::navigation::StoppedWheelsOffCandidateController,
        AppliedCommandReceipt,
        DisarmReceipt,
        kiko_slam::navigation::WheelsOffCandidateLimits,
        kiko_slam::navigation::WheelsOffCandidateRuntimeServiceInterval,
        AttendedWheelsOffPreflight,
        kiko_slam::navigation::WheelsOffQualificationControlProfile,
        kiko_slam::navigation::WheelsOffQualificationFrontendConfig,
        ConsoleSubsystemHealth,
        NanoAccessoryHealthObserver,
        ConsoleRerunDiagnosticsUrl,
        Option<kiko_slam::navigation::WheelsOffQualificationFaultInjection>,
    ) {
        (
            self.stopped_controller
                .take()
                .expect("qualification stopped token transfers once"),
            self.initial_zero
                .take()
                .expect("qualification initial zero transfers once"),
            self.initial_stop
                .take()
                .expect("qualification initial stop transfers once"),
            self.limits,
            self.runtime_service_interval,
            self.preflight
                .take()
                .expect("qualification motion starts only after attended preflight"),
            self.profile,
            self.frontend_config.clone(),
            self.initial_health,
            self.accessory_health.clone(),
            self.rerun_diagnostics_url,
            self.fault_injection,
        )
    }
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
impl Drop for LiveWheelsOffQualificationMotionInput {
    fn drop(&mut self) {
        if let Some(initial_stop) = self.initial_stop.as_ref()
            && self.stopped_controller.is_some()
        {
            eprintln!(
                "unused wheels-off qualification input remained stopped: boot_id={} request_id={}",
                initial_stop.observed_boot_id().get(),
                initial_stop.request_id().get(),
            );
        }
    }
}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    unix
))]
impl Drop for LiveProductionMotionInput {
    fn drop(&mut self) {
        let Some(prepared) = self.prepared.take() else {
            return;
        };
        match prepared.abort_before_owner() {
            Ok(receipt) => eprintln!(
                "unused production admission explicitly disarmed: boot_id={} request_id={} acknowledged_at_host_ns={}",
                receipt.observed_boot_id().get(),
                receipt.request_id().get(),
                receipt.acknowledged_at().nanos_since_clock_start(),
            ),
            Err(source) => {
                eprintln!("unused production admission could not prove controller stop: {source}")
            }
        }
    }
}

#[cfg(all(feature = "record", feature = "actuation"))]
impl LiveNavigationWorkerMotion {
    fn compatibility(actuation_config: Option<NavigationActuationConfigV1>) -> Self {
        Self::Compatibility(Box::new(actuation_config))
    }
}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    unix
))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LiveAgentAuthorityKind {
    Production,
    #[cfg(feature = "nano-attended-navigation-trial")]
    AttendedNavigationTrial,
}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    unix
))]
impl LiveAgentAuthorityKind {
    #[cfg(all(feature = "operator-console", feature = "nano-agent"))]
    const fn console(self) -> ConsoleRuntimeAuthorityKind {
        match self {
            Self::Production => ConsoleRuntimeAuthorityKind::ProductionExternalInterlocks,
            #[cfg(feature = "nano-attended-navigation-trial")]
            Self::AttendedNavigationTrial => ConsoleRuntimeAuthorityKind::AttendedNavigationTrial,
        }
    }
}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    unix
))]
struct LiveAgentMotionStartInput {
    policy: ManifestBoundNanoAgentPolicyConfigV3,
    authority: AgentAuthoritySupervisor,
    physical_driver: LiveMpcControlDriver,
    admitted_actuation_config: Option<NavigationActuationConfigV1>,
    coordinator_actuation_config: Option<NavigationActuationConfigV1>,
    kind: LiveAgentAuthorityKind,
}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    unix
))]
#[derive(Debug)]
enum LiveProductionOwnerStartPrimary {
    ClockEpochMismatch {
        coordinator_origin_ns: u64,
        supervisor_origin_ns: u64,
    },
    ActuationConfigMismatch,
    CoordinatorNotMappingOnly {
        actual: CoordinatorMotionModeV1,
    },
    ManualPlantBinding(NanoManualPlantBindingError),
    ControlSocket(AgentControlSocketTaskStartError),
    #[cfg(all(feature = "operator-console", feature = "nano-agent"))]
    ConsoleManualEnvelope(ConsoleManualCommandEnvelopeError),
    #[cfg(all(feature = "operator-console", feature = "nano-agent"))]
    AccessoryHealth(NanoAccessoryHealthStatusError),
    #[cfg(all(feature = "operator-console", feature = "nano-agent"))]
    ConsoleFrontend(NanoOperatorConsoleFrontendStartError),
}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    unix
))]
impl std::fmt::Display for LiveProductionOwnerStartPrimary {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ClockEpochMismatch {
                coordinator_origin_ns,
                supervisor_origin_ns,
            } => write!(
                formatter,
                "live-agent coordinator clock origin {coordinator_origin_ns} ns differs from supervisor origin {supervisor_origin_ns} ns"
            ),
            Self::ActuationConfigMismatch => formatter.write_str(
                "live-agent controller admission does not equal the actuation config bound to this coordinator",
            ),
            Self::CoordinatorNotMappingOnly { actual } => write!(
                formatter,
                "live-agent owner must start mapping-only, but the coordinator starts in {actual:?}"
            ),
            Self::ManualPlantBinding(source) => source.fmt(formatter),
            Self::ControlSocket(source) => source.fmt(formatter),
            #[cfg(all(feature = "operator-console", feature = "nano-agent"))]
            Self::ConsoleManualEnvelope(source) => source.fmt(formatter),
            #[cfg(all(feature = "operator-console", feature = "nano-agent"))]
            Self::AccessoryHealth(source) => source.fmt(formatter),
            #[cfg(all(feature = "operator-console", feature = "nano-agent"))]
            Self::ConsoleFrontend(source) => source.fmt(formatter),
        }
    }
}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    unix
))]
impl std::error::Error for LiveProductionOwnerStartPrimary {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ManualPlantBinding(source) => Some(source),
            Self::ControlSocket(source) => Some(source),
            #[cfg(all(feature = "operator-console", feature = "nano-agent"))]
            Self::ConsoleManualEnvelope(source) => Some(source),
            #[cfg(all(feature = "operator-console", feature = "nano-agent"))]
            Self::AccessoryHealth(source) => Some(source),
            #[cfg(all(feature = "operator-console", feature = "nano-agent"))]
            Self::ConsoleFrontend(source) => Some(source),
            Self::ClockEpochMismatch { .. }
            | Self::ActuationConfigMismatch
            | Self::CoordinatorNotMappingOnly { .. } => None,
        }
    }
}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    unix
))]
#[derive(Debug)]
enum LiveProductionControllerStop {
    Confirmed(DisarmReceipt),
    DisarmFailedStopConfirmed(LiveActuationError),
    Uncertain(LiveActuationError),
}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    unix
))]
#[derive(Debug)]
struct LiveProductionOwnerStartError {
    primary: LiveProductionOwnerStartPrimary,
    controller_stop: LiveProductionControllerStop,
    lifecycle_cleanup:
        Option<LiveMotionOperationError<LiveActuationError, NavigationIngressStreamWriteError>>,
    socket_shutdown: Option<Result<AgentControlSocketTaskExit, AgentControlSocketTaskJoinError>>,
}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    unix
))]
impl std::fmt::Display for LiveProductionOwnerStartError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "live-agent motion-owner startup failed: {}",
            self.primary
        )?;
        match &self.controller_stop {
            LiveProductionControllerStop::Confirmed(receipt) => write!(
                formatter,
                "; controller stop confirmed at {} ns",
                receipt.acknowledged_at().nanos_since_clock_start()
            ),
            LiveProductionControllerStop::DisarmFailedStopConfirmed(source) => write!(
                formatter,
                "; controller disarm failed, but recovery proved stop: {source}"
            ),
            LiveProductionControllerStop::Uncertain(source) => {
                write!(formatter, "; controller stop uncertain: {source}")
            }
        }?;
        if let Some(source) = &self.lifecycle_cleanup {
            write!(formatter, "; owner lifecycle cleanup also failed: {source}")?;
        }
        if let Some(socket_shutdown) = &self.socket_shutdown {
            write!(formatter, "; control socket cleanup: {socket_shutdown:?}")?;
        }
        Ok(())
    }
}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    unix
))]
impl std::error::Error for LiveProductionOwnerStartError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.primary)
    }
}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    feature = "nano-agent",
    feature = "operator-console",
    unix
))]
struct LiveProductionConsoleRuntime {
    adapter: Option<OperatorConsoleRuntimeAdapter>,
    frontend: Option<NanoOperatorConsoleFrontend>,
    observation: LiveConsoleNavigationObservation,
}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    feature = "nano-agent",
    feature = "operator-console",
    unix
))]
struct LiveConsoleNavigationObservation {
    next_snapshot_revision: Option<u64>,
    map: Option<ConsoleMapSnapshot>,
    last_requested_actuation: Option<ConsoleRequestedActuation>,
    last_applied: Option<ConsoleAppliedReceipt>,
    stop_certainty: Option<ConsoleStopCertainty>,
    successful_solver_duration_ns: Option<u64>,
    rerun_diagnostics_url: ConsoleRerunDiagnosticsUrl,
}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    feature = "nano-agent",
    feature = "operator-console",
    unix
))]
#[derive(Debug)]
enum LiveProductionConsoleProjectionError {
    Grid(ConsoleGridProjectionError),
    GridPublicationRejected {
        map_epoch_id: u64,
        revision: u64,
    },
    Receipt(ConsoleReceiptProjectionError),
    Numeric(ConsoleFiniteF64Error),
    Path(ConsolePathError),
    HostClock(HostMonotonicRangeError),
    AuthorityState(LiveMotionAuthorityStateError),
    AuthorityAdapter(OperatorConsoleRuntimeAdapterError),
    AccessoryHealth(NanoAccessoryHealthStatusError),
    SlamTelemetry(LiveSlamTelemetryError),
    OwnerAuthorityWithoutConsole {
        owner: LiveMotionAuthorityState,
    },
    ConsoleAuthorityWithoutOwner {
        console: OperatorConsoleRetainedAuthorityKind,
    },
    ConsoleAuthorityModeMismatch {
        owner: LiveMotionAuthorityState,
        console: OperatorConsoleRetainedAuthorityKind,
    },
    MapEpochZero,
    SnapshotRevisionExhausted,
    Snapshot(OperatorConsoleSnapshotError),
}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    feature = "nano-agent",
    feature = "operator-console",
    unix
))]
impl std::fmt::Display for LiveProductionConsoleProjectionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("live operator-console telemetry projection failed: ")?;
        match self {
            Self::Grid(source) => write!(formatter, "{source}"),
            Self::GridPublicationRejected {
                map_epoch_id,
                revision,
            } => write!(
                formatter,
                "grid publication rejected for map epoch {map_epoch_id}, revision {revision}"
            ),
            Self::Receipt(source) => write!(formatter, "{source}"),
            Self::Numeric(source) => write!(formatter, "{source}"),
            Self::Path(source) => write!(formatter, "{source}"),
            Self::HostClock(source) => write!(formatter, "{source}"),
            Self::AuthorityState(source) => write!(formatter, "{source}"),
            Self::AuthorityAdapter(source) => write!(formatter, "{source}"),
            Self::AccessoryHealth(source) => write!(formatter, "{source}"),
            Self::SlamTelemetry(source) => write!(formatter, "{source}"),
            Self::OwnerAuthorityWithoutConsole { owner } => write!(
                formatter,
                "sole owner retains {owner:?} authority without its unified-console linear guard"
            ),
            Self::ConsoleAuthorityWithoutOwner { console } => write!(
                formatter,
                "console retains {console:?} authority without a sole-owner supervisor token"
            ),
            Self::ConsoleAuthorityModeMismatch { owner, console } => write!(
                formatter,
                "console authority {console:?} contradicts sole-owner authority {owner:?}"
            ),
            Self::MapEpochZero => {
                formatter.write_str("map state is incomplete or has a zero epoch")
            }
            Self::SnapshotRevisionExhausted => {
                formatter.write_str("snapshot revision exhausted its nonzero u64 domain")
            }
            Self::Snapshot(source) => write!(formatter, "{source}"),
        }
    }
}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    feature = "nano-agent",
    feature = "operator-console",
    unix
))]
impl std::error::Error for LiveProductionConsoleProjectionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Grid(source) => Some(source),
            Self::Receipt(source) => Some(source),
            Self::Numeric(source) => Some(source),
            Self::Path(source) => Some(source),
            Self::HostClock(source) => Some(source),
            Self::AuthorityState(source) => Some(source),
            Self::AuthorityAdapter(source) => Some(source),
            Self::AccessoryHealth(source) => Some(source),
            Self::SlamTelemetry(source) => Some(source),
            Self::Snapshot(source) => Some(source),
            Self::GridPublicationRejected { .. }
            | Self::OwnerAuthorityWithoutConsole { .. }
            | Self::ConsoleAuthorityWithoutOwner { .. }
            | Self::ConsoleAuthorityModeMismatch { .. }
            | Self::MapEpochZero
            | Self::SnapshotRevisionExhausted => None,
        }
    }
}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    unix
))]
struct LiveProductionMotionRuntime {
    owner: Option<ProductionLiveMotionOwner>,
    socket_task: Option<AgentControlSocketTask>,
    #[cfg(all(feature = "operator-console", feature = "nano-agent"))]
    terminal_response_timeout: Duration,
    #[cfg(all(feature = "operator-console", feature = "nano-agent"))]
    console: Option<LiveProductionConsoleRuntime>,
    #[cfg(all(feature = "operator-console", feature = "nano-agent"))]
    accessory_health: NanoAccessoryHealthObserver,
    #[cfg(all(feature = "operator-console", feature = "nano-agent"))]
    sensor_health: LiveSensorStreamHealth,
    map_revision: Option<u64>,
    localized: bool,
    #[cfg(all(feature = "operator-console", feature = "nano-agent"))]
    terminal_checkpoint_pending: bool,
}

#[cfg(all(feature = "nano-agent", feature = "operator-console", unix))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LiveSensorStream {
    Visual,
    Depth,
    Imu,
}

#[cfg(all(feature = "nano-agent", feature = "operator-console", unix))]
impl LiveSensorStream {
    const fn bit(self) -> u8 {
        1 << (self as u8)
    }
}

#[cfg(all(feature = "nano-agent", feature = "operator-console", unix))]
const LIVE_SENSOR_CONSOLE_MAX_SAMPLE_AGE_NS: u64 = 1_000_000_000;

#[cfg(all(feature = "nano-agent", feature = "operator-console", unix))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LiveSensorStreamHealth {
    visual_observed_at: Option<HostMonotonicTimestamp>,
    depth_observed_at: Option<HostMonotonicTimestamp>,
    imu_observed_at: Option<HostMonotonicTimestamp>,
    visual_open: bool,
    depth_open: bool,
    imu_open: bool,
    latched_stale_streams: u8,
}

#[cfg(all(feature = "nano-agent", feature = "operator-console", unix))]
impl LiveSensorStreamHealth {
    const fn awaiting_first_samples() -> Self {
        Self {
            visual_observed_at: None,
            depth_observed_at: None,
            imu_observed_at: None,
            visual_open: true,
            depth_open: true,
            imu_open: true,
            latched_stale_streams: 0,
        }
    }

    fn observe(&mut self, stream: LiveSensorStream, observed_at: HostMonotonicTimestamp) {
        if self.latched_stale_streams & stream.bit() != 0 {
            return;
        }
        match stream {
            LiveSensorStream::Visual => self.visual_observed_at = Some(observed_at),
            LiveSensorStream::Depth => self.depth_observed_at = Some(observed_at),
            LiveSensorStream::Imu => self.imu_observed_at = Some(observed_at),
        }
    }

    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    fn latch_stale(&mut self, stream: LiveSensorStream) {
        self.latched_stale_streams |= stream.bit();
        match stream {
            LiveSensorStream::Visual => self.visual_observed_at = None,
            LiveSensorStream::Depth => self.depth_observed_at = None,
            LiveSensorStream::Imu => self.imu_observed_at = None,
        }
    }

    fn mark_closed(&mut self, stream: LiveSensorStream) {
        match stream {
            LiveSensorStream::Visual => self.visual_open = false,
            LiveSensorStream::Depth => self.depth_open = false,
            LiveSensorStream::Imu => self.imu_open = false,
        }
    }

    fn console_health(self, observed_at: HostMonotonicTimestamp) -> ConsoleHealth {
        if !self.visual_open || !self.depth_open || !self.imu_open {
            ConsoleHealth::Faulted
        } else if [
            self.visual_observed_at,
            self.depth_observed_at,
            self.imu_observed_at,
        ]
        .into_iter()
        .all(|sample| {
            sample.is_some_and(|sample| {
                observed_at
                    .as_nanos()
                    .checked_sub(sample.as_nanos())
                    .is_some_and(|age| age <= LIVE_SENSOR_CONSOLE_MAX_SAMPLE_AGE_NS)
            })
        }) {
            ConsoleHealth::Ready
        } else {
            ConsoleHealth::Degraded
        }
    }
}

#[cfg(all(feature = "nano-agent", feature = "operator-console", unix))]
fn console_oak_stream_health(
    sensor_health: LiveSensorStreamHealth,
    observed_at: HostMonotonicTimestamp,
) -> ConsoleHealth {
    sensor_health.console_health(observed_at)
}

#[cfg(all(feature = "nano-agent", feature = "operator-console", unix))]
const fn console_requested_inference_backend(
    backend: InferenceBackend,
) -> ConsoleRequestedInferenceBackend {
    match backend {
        InferenceBackend::Auto => ConsoleRequestedInferenceBackend::Auto,
        InferenceBackend::Cpu => ConsoleRequestedInferenceBackend::Cpu,
        InferenceBackend::CoreMLGpu => ConsoleRequestedInferenceBackend::CoremlGpu,
        InferenceBackend::Cuda => ConsoleRequestedInferenceBackend::Cuda,
        InferenceBackend::TensorRT => ConsoleRequestedInferenceBackend::TensorRt,
    }
}

#[cfg(all(feature = "nano-agent", feature = "operator-console", unix))]
const fn console_selected_inference_backend(
    backend: LiveSelectedInferenceBackend,
) -> ConsoleSelectedInferenceBackend {
    match backend {
        LiveSelectedInferenceBackend::Cpu => ConsoleSelectedInferenceBackend::Cpu,
        LiveSelectedInferenceBackend::CoremlGpu => ConsoleSelectedInferenceBackend::CoremlGpu,
        LiveSelectedInferenceBackend::Cuda => ConsoleSelectedInferenceBackend::Cuda,
        LiveSelectedInferenceBackend::TensorRt => ConsoleSelectedInferenceBackend::TensorRt,
    }
}

#[cfg(all(feature = "nano-agent", feature = "operator-console", unix))]
fn live_slam_sample_is_fresh(
    sample: Option<HostMonotonicTimestamp>,
    observed_at: HostMonotonicTimestamp,
) -> bool {
    sample.is_some_and(|sample| {
        observed_at
            .as_nanos()
            .checked_sub(sample.as_nanos())
            .is_some_and(|age| age <= LIVE_SENSOR_CONSOLE_MAX_SAMPLE_AGE_NS)
    })
}

#[cfg(all(feature = "nano-agent", feature = "operator-console", unix))]
fn project_live_slam_console(
    telemetry: &LiveSlamTelemetry,
    observed_at: HostMonotonicTimestamp,
) -> Result<(ConsoleSlamSnapshot, ConsoleHealth), LiveSlamTelemetryError> {
    let snapshot = telemetry.snapshot()?;
    let health = match snapshot.pipeline_state {
        LiveSlamPipelineState::Faulted => ConsoleHealth::Faulted,
        LiveSlamPipelineState::Closed => ConsoleHealth::Unavailable,
        LiveSlamPipelineState::Running => {
            if live_slam_sample_is_fresh(snapshot.last_successful_source_arrival, observed_at)
                && live_slam_sample_is_fresh(snapshot.last_successful_completion, observed_at)
            {
                ConsoleHealth::Ready
            } else {
                ConsoleHealth::Degraded
            }
        }
    };
    Ok((
        ConsoleSlamSnapshot {
            inference: ConsoleInferenceRuntime {
                superpoint: ConsoleInferenceSelection {
                    requested: console_requested_inference_backend(
                        snapshot.inference.superpoint_requested,
                    ),
                    selected: console_selected_inference_backend(
                        snapshot.inference.superpoint_selected,
                    ),
                },
                lightglue: ConsoleInferenceSelection {
                    requested: console_requested_inference_backend(
                        snapshot.inference.lightglue_requested,
                    ),
                    selected: console_selected_inference_backend(
                        snapshot.inference.lightglue_selected,
                    ),
                },
            },
            started_pairs: snapshot.started_pairs,
            successful_pairs: snapshot.successful_pairs,
            recoverable_failures: snapshot.recoverable_failures,
            fatal_failures: snapshot.fatal_failures,
            last_successful_source_arrival_host_monotonic_ns: snapshot
                .last_successful_source_arrival
                .map(HostMonotonicTimestamp::as_nanos),
            last_successful_completion_host_monotonic_ns: snapshot
                .last_successful_completion
                .map(HostMonotonicTimestamp::as_nanos),
            rate_window: snapshot.rate_window.map(|window| ConsoleSlamRateWindow {
                successful_completions: window.successful_completions,
                span_ns: window.span_ns,
            }),
        },
        health,
    ))
}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    unix
))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LiveProductionMapStateError {
    BindingWithoutRevision,
    RevisionWithoutBinding,
}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    unix
))]
impl std::fmt::Display for LiveProductionMapStateError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BindingWithoutRevision => formatter.write_str(
                "live-agent map state has a coordinator binding without its admitted revision",
            ),
            Self::RevisionWithoutBinding => formatter.write_str(
                "live-agent map state has an admitted revision without a coordinator binding",
            ),
        }
    }
}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    unix
))]
impl std::error::Error for LiveProductionMapStateError {}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    unix
))]
impl LiveProductionMotionRuntime {
    fn owner(&self) -> &ProductionLiveMotionOwner {
        self.owner
            .as_ref()
            .expect("production owner exists until terminal shutdown")
    }

    fn owner_mut(&mut self) -> &mut ProductionLiveMotionOwner {
        self.owner
            .as_mut()
            .expect("production owner exists until terminal shutdown")
    }

    fn take_terminal_parts(&mut self) -> (ProductionLiveMotionOwner, AgentControlSocketTask) {
        let owner = self
            .owner
            .take()
            .expect("production owner is consumed exactly once");
        let socket_task = self
            .socket_task
            .take()
            .expect("production socket task is consumed exactly once");
        (owner, socket_task)
    }

    #[cfg(all(feature = "operator-console", feature = "nano-agent"))]
    fn console_mut(&mut self) -> &mut LiveProductionConsoleRuntime {
        self.console
            .as_mut()
            .expect("production console exists until terminal shutdown")
    }

    #[cfg(all(feature = "operator-console", feature = "nano-agent"))]
    fn take_console(&mut self) -> LiveProductionConsoleRuntime {
        self.console
            .take()
            .expect("production console is consumed exactly once")
    }

    fn map_state(&self) -> Result<AgentMapStateV1, LiveProductionMapStateError> {
        match (
            self.owner().coordinator().current_map_binding(),
            self.map_revision,
        ) {
            (Some(binding), Some(revision)) => Ok(AgentMapStateV1::available(
                binding.map_epoch_id(),
                revision,
                if self.localized {
                    AgentLocalizationStateV1::Localized
                } else {
                    AgentLocalizationStateV1::Lost
                },
            )),
            (None, None) => Ok(AgentMapStateV1::UNAVAILABLE),
            (Some(_), None) => Err(LiveProductionMapStateError::BindingWithoutRevision),
            (None, Some(_)) => Err(LiveProductionMapStateError::RevisionWithoutBinding),
        }
    }
}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    unix
))]
impl Drop for LiveProductionMotionRuntime {
    fn drop(&mut self) {
        if let Some(socket_task) = self.socket_task.as_ref() {
            socket_task.request_shutdown();
        }
        #[cfg(all(feature = "operator-console", feature = "nano-agent"))]
        let mut console = self.console.take();
        #[cfg(all(feature = "operator-console", feature = "nano-agent"))]
        if let Some(frontend) = console
            .as_mut()
            .and_then(|console| console.frontend.as_mut())
        {
            frontend.request_shutdown();
        }
        if let Some(owner) = self.owner.take() {
            let report = owner.shutdown();
            #[cfg(all(feature = "operator-console", feature = "nano-agent"))]
            if let Some(adapter) = console.as_mut().and_then(|console| console.adapter.take()) {
                let outcome = adapter.shutdown(&report);
                if !outcome.controller_stop_confirmed || outcome.lifecycle_cleanup_failed {
                    eprintln!("live-agent console unwind terminalized fail-closed: {outcome:?}");
                }
            }
            if let Some(source) = report.lifecycle_cleanup() {
                eprintln!(
                    "live-agent motion owner unwound with lifecycle cleanup failure: {source}"
                );
            }
            match report.controller_stop() {
                LiveMotionTerminalStop::Confirmed(_) => {}
                LiveMotionTerminalStop::DisarmFailedStopConfirmed(source) => eprintln!(
                    "live-agent motion owner unwind disarm failed, but recovery proved controller stop: {source}"
                ),
                LiveMotionTerminalStop::Uncertain(source) => eprintln!(
                    "live-agent motion owner unwind could not prove controller stop: {source}"
                ),
            }
        }
        #[cfg(all(feature = "operator-console", feature = "nano-agent"))]
        if let Some(frontend) = console
            .as_mut()
            .and_then(|console| console.frontend.as_mut())
        {
            let evidence = frontend.shutdown();
            let retains_live_http_owner = evidence.retains_live_http_owner();
            if !evidence.is_clean() {
                eprintln!("live-agent console unwind cleanup was not clean: {evidence}");
            }
            if !retains_live_http_owner && let Some(console) = console.as_mut() {
                console.frontend.take();
            }
        }
        #[cfg(all(feature = "operator-console", feature = "nano-agent"))]
        drop(console);
        if let Some(socket_task) = self.socket_task.take()
            && let Err(source) = socket_task.shutdown()
        {
            eprintln!("live-agent control socket unwind cleanup failed: {source}");
        }
    }
}

#[cfg(feature = "record")]
#[derive(Debug)]
enum LiveNavigationWorkerError {
    HostClock(HostMonotonicRangeError),
    TickTiming(LiveControlTickTimingError),
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
    #[cfg(all(feature = "actuation", feature = "agent-runtime", unix))]
    ProductionMapAdmission {
        source: LiveMotionMapAdmissionError<NavigationIngressStreamWriteError>,
    },
    #[cfg(all(feature = "actuation", feature = "agent-runtime", unix))]
    ProductionMapState {
        source: LiveProductionMapStateError,
    },
    #[cfg(all(feature = "actuation", feature = "agent-runtime", unix))]
    PhysicalStateVisualization {
        source: LivePhysicalStateVizPublishError,
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
    MpcControl {
        source: LiveMpcControlError,
    },
    #[cfg(all(feature = "actuation", feature = "agent-runtime", unix))]
    ProductionStart {
        source: Box<LiveProductionOwnerStartError>,
    },
    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    WheelsOffQualificationStart {
        source: Box<LiveWheelsOffQualificationMotionStartError>,
    },
    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    WheelsOffQualificationRuntime {
        source: Box<kiko_slam::navigation::WheelsOffQualificationRuntimeError>,
    },
    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    WheelsOffQualificationAttestation {
        source: FreshAttendedMotionAttestationWorkerError,
    },
    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    WheelsOffQualificationAttestationCleanup {
        source: FreshAttendedMotionAttestationWorkerError,
    },
    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    WheelsOffQualificationMotionAuthorityEnable {
        source: kiko_slam::navigation::WheelsOffQualificationMotionAuthorityEnableFailure,
    },
    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    WheelsOffQualificationAppliedStepBoundary {
        source: NavigationIngressBoundaryError,
    },
    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    WheelsOffQualificationAppliedStepJournal {
        source: NavigationIngressStreamWriteError,
    },
    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    WheelsOffQualificationAppliedStepCorrelation {
        source: kiko_slam::navigation::WheelsOffQualificationAppliedStepJournalError,
    },
    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    WheelsOffQualificationFaultTrigger {
        selected: kiko_slam::navigation::WheelsOffQualificationFaultInjection,
        source: kiko_slam::navigation::WheelsOffQualificationLiveFaultTriggerError,
    },
    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    WheelsOffQualificationFaultNotExercised {
        selected: kiko_slam::navigation::WheelsOffQualificationFaultInjection,
    },
    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    WheelsOffQualificationFrontendExited {
        evidence: Box<kiko_slam::navigation::WheelsOffQualificationFrontendShutdownEvidence>,
    },
    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    WheelsOffQualificationFrontendShutdown {
        evidence: Box<kiko_slam::navigation::WheelsOffQualificationFrontendShutdownEvidence>,
    },
    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    WheelsOffQualificationTelemetry {
        source: kiko_slam::navigation::WheelsOffQualificationTelemetryError,
    },
    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    WheelsOffQualificationProjection {
        source: LiveProductionConsoleProjectionError,
    },
    #[cfg(all(feature = "actuation", feature = "agent-runtime", unix))]
    ProductionOwner {
        source: Box<LiveMotionOwnerError<LiveActuationError, NavigationIngressStreamWriteError>>,
    },
    #[cfg(all(feature = "nano-agent", feature = "operator-console", unix))]
    ProductionConsoleIngress {
        source: Box<
            OperatorConsoleRuntimeIngressError<
                LiveActuationError,
                NavigationIngressStreamWriteError,
            >,
        >,
    },
    #[cfg(all(feature = "nano-agent", feature = "operator-console", unix))]
    ProductionConsoleAdapter {
        source: OperatorConsoleRuntimeAdapterError,
    },
    #[cfg(all(feature = "nano-agent", feature = "operator-console", unix))]
    ProductionConsoleFrontendShutdown {
        evidence: Box<NanoOperatorConsoleFrontendShutdownEvidence>,
    },
    #[cfg(all(feature = "nano-agent", feature = "operator-console", unix))]
    ProductionConsoleFrontendExited {
        evidence: Box<NanoOperatorConsoleFrontendShutdownEvidence>,
    },
    #[cfg(all(feature = "nano-agent", feature = "operator-console", unix))]
    ProductionConsoleProjection {
        source: LiveProductionConsoleProjectionError,
    },
    #[cfg(all(feature = "nano-agent", unix))]
    ProductionSaveMap {
        source: Box<NanoMapSaveCommandError>,
    },
    #[cfg(all(feature = "nano-agent", unix))]
    ProductionWarmCheckpoint {
        source: Box<NanoWarmCheckpointCommandError>,
    },
    #[cfg(all(feature = "nano-agent", unix))]
    WarmCheckpointRequestChannelDisconnected,
    #[cfg(all(feature = "nano-agent", unix))]
    WarmCheckpointFinalizationChannelDisconnected,
    #[cfg(all(feature = "nano-agent", unix))]
    WarmCheckpointFinalizationTimedOut,
    #[cfg(all(feature = "nano-agent", unix))]
    WarmCheckpointDeadlineOverflow {
        response: Option<AgentControlDispatchResponseError>,
    },
    #[cfg(all(feature = "nano-agent", unix))]
    WarmCheckpointDeadlineUnavailable {
        response: Option<AgentControlDispatchResponseError>,
    },
    #[cfg(all(feature = "nano-agent", unix))]
    WarmCheckpointDatasetNotPublishedResponse {
        source: Option<AgentControlDispatchResponseError>,
    },
    #[cfg(all(
        feature = "actuation",
        feature = "agent-runtime",
        unix,
        not(feature = "nano-agent")
    ))]
    ProductionSaveMapResponse {
        source: AgentControlDispatchResponseError,
    },
    #[cfg(all(feature = "nano-agent", unix))]
    MapPersistenceRetention {
        source: NanoMapSnapshotRetentionError,
    },
    #[cfg(all(feature = "nano-agent", unix))]
    MapPersistenceBindingUnavailable,
    #[cfg(all(feature = "nano-agent", unix))]
    ProductionMapPersistenceUnavailable {
        response: Option<AgentControlDispatchResponseError>,
    },
    #[cfg(all(feature = "actuation", feature = "agent-runtime", unix))]
    ProductionLifecycleCleanup {
        source:
            Box<LiveMotionOperationError<LiveActuationError, NavigationIngressStreamWriteError>>,
    },
    #[cfg(all(feature = "actuation", feature = "agent-runtime", unix))]
    ProductionControllerStop {
        source: LiveActuationError,
    },
    #[cfg(all(feature = "actuation", feature = "agent-runtime", unix))]
    ProductionSocketJoin {
        source: AgentControlSocketTaskJoinError,
    },
    #[cfg(all(feature = "actuation", feature = "agent-runtime", unix))]
    ProductionSocketExit {
        exit: Box<AgentControlSocketTaskExit>,
    },
    Multiple {
        failures: Vec<Self>,
    },
    JournalFinalization {
        source: NavigationIngressStreamWriteError,
    },
    JournalSync {
        source: std::io::Error,
    },
    JournalSeek {
        source: std::io::Error,
    },
    JournalRecordCountOutOfRange {
        record_count: u64,
    },
    JournalCapacity {
        source: NavigationIngressCapacityError,
    },
    JournalVerification {
        source: NavigationIngressStreamReadError,
    },
}

#[cfg(feature = "record")]
impl std::fmt::Display for LiveNavigationWorkerError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HostClock(source) => {
                write!(formatter, "navigation host clock failed: {source}")
            }
            Self::TickTiming(source) => {
                write!(formatter, "navigation control-tick timing failed: {source}")
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
            #[cfg(all(feature = "actuation", feature = "agent-runtime", unix))]
            Self::ProductionMapAdmission { source } => {
                write!(
                    formatter,
                    "live-agent global-map navigation admission failed: {source}"
                )
            }
            #[cfg(all(feature = "actuation", feature = "agent-runtime", unix))]
            Self::ProductionMapState { source } => source.fmt(formatter),
            #[cfg(all(feature = "actuation", feature = "agent-runtime", unix))]
            Self::PhysicalStateVisualization { source } => source.fmt(formatter),
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
            Self::MpcControl { source } => write!(formatter, "{source}"),
            #[cfg(all(feature = "actuation", feature = "agent-runtime", unix))]
            Self::ProductionStart { source } => source.fmt(formatter),
            #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
            Self::WheelsOffQualificationStart { source } => source.fmt(formatter),
            #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
            Self::WheelsOffQualificationRuntime { source } => source.fmt(formatter),
            #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
            Self::WheelsOffQualificationAttestation { source } => {
                write!(formatter, "wheels-off qualification attestation failed: {source}")
            }
            #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
            Self::WheelsOffQualificationAttestationCleanup { source } => write!(
                formatter,
                "wheels-off qualification attestation cleanup failed: {source}"
            ),
            #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
            Self::WheelsOffQualificationMotionAuthorityEnable { source } => source.fmt(formatter),
            #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
            Self::WheelsOffQualificationAppliedStepBoundary { source } => write!(
                formatter,
                "wheels-off qualification applied-step timestamp is invalid: {source}"
            ),
            #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
            Self::WheelsOffQualificationAppliedStepJournal { source } => write!(
                formatter,
                "wheels-off qualification applied-step journal append failed: {source}"
            ),
            #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
            Self::WheelsOffQualificationAppliedStepCorrelation { source } => source.fmt(formatter),
            #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
            Self::WheelsOffQualificationFaultTrigger { selected, source } => write!(
                formatter,
                "wheels-off qualification fault {selected} was not exercised: {source}"
            ),
            #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
            Self::WheelsOffQualificationFaultNotExercised { selected } => write!(
                formatter,
                "wheels-off qualification fault {selected} was selected but the process reached normal teardown without exercising its exact injected-fault path"
            ),
            #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
            Self::WheelsOffQualificationFrontendExited { evidence } => write!(
                formatter,
                "wheels-off qualification frontend exited before shutdown; controller is being stopped: {evidence:?}"
            ),
            #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
            Self::WheelsOffQualificationFrontendShutdown { evidence } => write!(
                formatter,
                "wheels-off qualification frontend cleanup was not clean: {evidence:?}"
            ),
            #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
            Self::WheelsOffQualificationTelemetry { source } => source.fmt(formatter),
            #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
            Self::WheelsOffQualificationProjection { source } => source.fmt(formatter),
            #[cfg(all(feature = "actuation", feature = "agent-runtime", unix))]
            Self::ProductionOwner { source } => source.fmt(formatter),
            #[cfg(all(feature = "nano-agent", feature = "operator-console", unix))]
            Self::ProductionConsoleIngress { source } => source.fmt(formatter),
            #[cfg(all(feature = "nano-agent", feature = "operator-console", unix))]
            Self::ProductionConsoleAdapter { source } => source.fmt(formatter),
            #[cfg(all(feature = "nano-agent", feature = "operator-console", unix))]
            Self::ProductionConsoleFrontendShutdown { evidence } => evidence.fmt(formatter),
            #[cfg(all(feature = "nano-agent", feature = "operator-console", unix))]
            Self::ProductionConsoleFrontendExited { evidence } => write!(
                formatter,
                "operator-console HTTP owner exited before live-agent shutdown; motion is being stopped: {evidence}"
            ),
            #[cfg(all(feature = "nano-agent", feature = "operator-console", unix))]
            Self::ProductionConsoleProjection { source } => source.fmt(formatter),
            #[cfg(all(feature = "nano-agent", unix))]
            Self::ProductionSaveMap { source } => source.fmt(formatter),
            #[cfg(all(feature = "nano-agent", unix))]
            Self::ProductionWarmCheckpoint { source } => source.fmt(formatter),
            #[cfg(all(feature = "nano-agent", unix))]
            Self::WarmCheckpointRequestChannelDisconnected => formatter.write_str(
                "terminal warm checkpoint could not hand its finalized journal to the dataset owner",
            ),
            #[cfg(all(feature = "nano-agent", unix))]
            Self::WarmCheckpointFinalizationChannelDisconnected => formatter.write_str(
                "terminal warm checkpoint dataset owner disappeared before reporting publication",
            ),
            #[cfg(all(feature = "nano-agent", unix))]
            Self::WarmCheckpointFinalizationTimedOut => formatter.write_str(
                "terminal warm checkpoint dataset publication exceeded its parsed terminal deadline",
            ),
            #[cfg(all(feature = "nano-agent", unix))]
            Self::WarmCheckpointDeadlineOverflow { response: None } => formatter.write_str(
                "terminal warm checkpoint deadline overflowed and the request was rejected",
            ),
            #[cfg(all(feature = "nano-agent", unix))]
            Self::WarmCheckpointDeadlineOverflow {
                response: Some(source),
            } => write!(
                formatter,
                "terminal warm checkpoint deadline overflowed and its rejection response failed: {source}"
            ),
            #[cfg(all(feature = "nano-agent", unix))]
            Self::WarmCheckpointDeadlineUnavailable { response: None } => formatter.write_str(
                "terminal warm checkpoint had no remaining admitted deadline or another terminal checkpoint already owned it",
            ),
            #[cfg(all(feature = "nano-agent", unix))]
            Self::WarmCheckpointDeadlineUnavailable {
                response: Some(source),
            } => write!(
                formatter,
                "terminal warm checkpoint had no available deadline and its rejection response failed: {source}"
            ),
            #[cfg(all(feature = "nano-agent", unix))]
            Self::WarmCheckpointDatasetNotPublishedResponse { source: None } => formatter
                .write_str("terminal warm checkpoint was rejected because its exact dataset was not published"),
            #[cfg(all(feature = "nano-agent", unix))]
            Self::WarmCheckpointDatasetNotPublishedResponse {
                source: Some(source),
            } => write!(
                formatter,
                "terminal warm checkpoint dataset was not published and the truthful rejection response also failed: {source}"
            ),
            #[cfg(all(
                feature = "actuation",
                feature = "agent-runtime",
                unix,
                not(feature = "nano-agent")
            ))]
            Self::ProductionSaveMapResponse { source } => write!(
                formatter,
                "unwired live-agent map persistence was rejected but the response failed: {source}"
            ),
            #[cfg(all(feature = "nano-agent", unix))]
            Self::MapPersistenceRetention { source } => source.fmt(formatter),
            #[cfg(all(feature = "nano-agent", unix))]
            Self::MapPersistenceBindingUnavailable => formatter
                .write_str("an admitted occupancy snapshot has no current coordinator map binding"),
            #[cfg(all(feature = "nano-agent", unix))]
            Self::ProductionMapPersistenceUnavailable { response: None } => formatter.write_str(
                "live-agent runtime accepted a save-map request without a map persistence owner",
            ),
            #[cfg(all(feature = "nano-agent", unix))]
            Self::ProductionMapPersistenceUnavailable {
                response: Some(source),
            } => write!(
                formatter,
                "live-agent runtime accepted a save-map request without a map persistence owner; rejection response also failed: {source}"
            ),
            #[cfg(all(feature = "actuation", feature = "agent-runtime", unix))]
            Self::ProductionLifecycleCleanup { source } => {
                write!(formatter, "live-agent lifecycle cleanup failed: {source}")
            }
            #[cfg(all(feature = "actuation", feature = "agent-runtime", unix))]
            Self::ProductionControllerStop { source } => write!(
                formatter,
                "live-agent shutdown could not prove controller stop: {source}"
            ),
            #[cfg(all(feature = "actuation", feature = "agent-runtime", unix))]
            Self::ProductionSocketJoin { source } => source.fmt(formatter),
            #[cfg(all(feature = "actuation", feature = "agent-runtime", unix))]
            Self::ProductionSocketExit { exit } => {
                write!(
                    formatter,
                    "agent-control socket task exited abnormally: {exit:?}"
                )
            }
            Self::Multiple { failures } => {
                formatter.write_str("multiple live navigation failures")?;
                for (index, failure) in failures.iter().enumerate() {
                    write!(formatter, "; failure {}: {failure}", index + 1)?;
                }
                Ok(())
            }
            Self::JournalFinalization { source } => {
                write!(
                    formatter,
                    "navigation journal finalization failed: {source}"
                )
            }
            Self::JournalSync { source } => {
                write!(formatter, "navigation journal sync failed: {source}")
            }
            Self::JournalSeek { source } => {
                write!(formatter, "navigation journal rewind failed: {source}")
            }
            Self::JournalRecordCountOutOfRange { record_count } => write!(
                formatter,
                "navigation journal record count {record_count} is not representable in memory"
            ),
            Self::JournalCapacity { source } => {
                write!(formatter, "navigation journal verification bound is invalid: {source}")
            }
            Self::JournalVerification { source } => {
                write!(formatter, "synchronized navigation journal failed verification: {source}")
            }
        }
    }
}

#[cfg(feature = "record")]
impl std::error::Error for LiveNavigationWorkerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::HostClock(source) => Some(source),
            Self::TickTiming(source) => Some(source),
            Self::VisualAdmission { source }
            | Self::ImuAdmission { source }
            | Self::DepthAdmission { source }
            | Self::MapAdmission { source } => Some(source),
            #[cfg(all(feature = "actuation", feature = "agent-runtime", unix))]
            Self::ProductionMapAdmission { source } => Some(source),
            #[cfg(all(feature = "actuation", feature = "agent-runtime", unix))]
            Self::ProductionMapState { source } => Some(source),
            #[cfg(all(feature = "actuation", feature = "agent-runtime", unix))]
            Self::PhysicalStateVisualization { source } => Some(source),
            Self::Tick { source } => Some(source),
            Self::JournalFinalization { source } => Some(source),
            Self::JournalSync { source } | Self::JournalSeek { source } => Some(source),
            Self::JournalCapacity { source } => Some(source),
            Self::JournalVerification { source } => Some(source),
            Self::JournalRecordCountOutOfRange { .. } => None,
            #[cfg(feature = "actuation")]
            Self::Actuation { source, .. } => Some(source),
            #[cfg(feature = "actuation")]
            Self::MpcControl { source } => Some(source),
            #[cfg(all(feature = "actuation", feature = "agent-runtime", unix))]
            Self::ProductionStart { source } => Some(source),
            #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
            Self::WheelsOffQualificationStart { source } => Some(source.as_ref()),
            #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
            Self::WheelsOffQualificationRuntime { source } => Some(source.as_ref()),
            #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
            Self::WheelsOffQualificationAttestation { source }
            | Self::WheelsOffQualificationAttestationCleanup { source } => Some(source),
            #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
            Self::WheelsOffQualificationMotionAuthorityEnable { source } => Some(source),
            #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
            Self::WheelsOffQualificationAppliedStepBoundary { source } => Some(source),
            #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
            Self::WheelsOffQualificationAppliedStepJournal { source } => Some(source),
            #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
            Self::WheelsOffQualificationAppliedStepCorrelation { source } => Some(source),
            #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
            Self::WheelsOffQualificationFaultTrigger { source, .. } => Some(source),
            #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
            Self::WheelsOffQualificationFaultNotExercised { .. }
            | Self::WheelsOffQualificationFrontendExited { .. }
            | Self::WheelsOffQualificationFrontendShutdown { .. } => None,
            #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
            Self::WheelsOffQualificationTelemetry { source } => Some(source),
            #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
            Self::WheelsOffQualificationProjection { source } => Some(source),
            #[cfg(all(feature = "actuation", feature = "agent-runtime", unix))]
            Self::ProductionOwner { source } => Some(source.as_ref()),
            #[cfg(all(feature = "nano-agent", feature = "operator-console", unix))]
            Self::ProductionConsoleIngress { source } => Some(source.as_ref()),
            #[cfg(all(feature = "nano-agent", feature = "operator-console", unix))]
            Self::ProductionConsoleAdapter { source } => Some(source),
            #[cfg(all(feature = "nano-agent", feature = "operator-console", unix))]
            Self::ProductionConsoleFrontendShutdown { .. }
            | Self::ProductionConsoleFrontendExited { .. } => None,
            #[cfg(all(feature = "nano-agent", feature = "operator-console", unix))]
            Self::ProductionConsoleProjection { source } => Some(source),
            #[cfg(all(feature = "nano-agent", unix))]
            Self::ProductionSaveMap { source } => Some(source.as_ref()),
            #[cfg(all(feature = "nano-agent", unix))]
            Self::ProductionWarmCheckpoint { source } => Some(source.as_ref()),
            #[cfg(all(feature = "nano-agent", unix))]
            Self::WarmCheckpointDatasetNotPublishedResponse {
                source: Some(source),
            }
            | Self::WarmCheckpointDeadlineOverflow {
                response: Some(source),
            }
            | Self::WarmCheckpointDeadlineUnavailable {
                response: Some(source),
            } => Some(source),
            #[cfg(all(
                feature = "actuation",
                feature = "agent-runtime",
                unix,
                not(feature = "nano-agent")
            ))]
            Self::ProductionSaveMapResponse { source } => Some(source),
            #[cfg(all(feature = "nano-agent", unix))]
            Self::MapPersistenceRetention { source } => Some(source),
            #[cfg(all(feature = "nano-agent", unix))]
            Self::MapPersistenceBindingUnavailable => None,
            #[cfg(all(feature = "nano-agent", unix))]
            Self::ProductionMapPersistenceUnavailable {
                response: Some(source),
            } => Some(source),
            #[cfg(all(feature = "nano-agent", unix))]
            Self::ProductionMapPersistenceUnavailable { response: None } => None,
            #[cfg(all(feature = "nano-agent", unix))]
            Self::WarmCheckpointRequestChannelDisconnected
            | Self::WarmCheckpointFinalizationChannelDisconnected
            | Self::WarmCheckpointFinalizationTimedOut
            | Self::WarmCheckpointDatasetNotPublishedResponse { source: None }
            | Self::WarmCheckpointDeadlineOverflow { response: None }
            | Self::WarmCheckpointDeadlineUnavailable { response: None } => None,
            #[cfg(all(feature = "actuation", feature = "agent-runtime", unix))]
            Self::ProductionLifecycleCleanup { source } => Some(source.as_ref()),
            #[cfg(all(feature = "actuation", feature = "agent-runtime", unix))]
            Self::ProductionControllerStop { source } => Some(source),
            #[cfg(all(feature = "actuation", feature = "agent-runtime", unix))]
            Self::ProductionSocketJoin { source } => Some(source),
            #[cfg(all(feature = "actuation", feature = "agent-runtime", unix))]
            Self::ProductionSocketExit { .. } => None,
            Self::Multiple { failures } => failures
                .first()
                .map(|failure| failure as &(dyn std::error::Error + 'static)),
            Self::TickSequenceExhausted => None,
        }
    }
}

#[cfg(feature = "record")]
fn combine_live_navigation_failures(
    mut failures: Vec<LiveNavigationWorkerError>,
) -> Result<(), LiveNavigationWorkerError> {
    match failures.len() {
        0 => Ok(()),
        1 => Err(failures.pop().expect("one retained failure")),
        _ => Err(LiveNavigationWorkerError::Multiple { failures }),
    }
}

#[cfg(feature = "record")]
struct LiveNavigationWorkerSuccess {
    descriptor: NavigationIngressSidecarDescriptor,
}

#[cfg(feature = "record")]
struct FinalizedLiveNavigationJournal {
    descriptor: NavigationIngressSidecarDescriptor,
    final_map_identity: Option<FinalizedJournalMapIdentity>,
}

#[cfg(feature = "record")]
impl FinalizedLiveNavigationJournal {
    fn into_descriptor(self) -> NavigationIngressSidecarDescriptor {
        let Self {
            descriptor,
            final_map_identity: _final_map_identity,
        } = self;
        descriptor
    }
}

#[cfg(feature = "record")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FinalizedJournalMapIdentity {
    map_epoch_id: RecordedMapEpochId,
    revision: u64,
}

#[cfg(all(feature = "record", feature = "nano-agent", unix))]
struct NanoDatasetCheckpointRequest {
    descriptor: Option<NavigationIngressSidecarDescriptor>,
    navigation_publishable: bool,
}

#[cfg(all(feature = "record", feature = "nano-agent", unix))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NanoDatasetCheckpointFinalization {
    Published,
    Rejected,
}

#[cfg(all(feature = "record", feature = "nano-agent", unix))]
struct NanoDatasetCheckpointWorkerBridge {
    requested: Arc<AtomicBool>,
    checkpoint_deadline: Arc<std::sync::OnceLock<Instant>>,
    dataset_directory: PathBuf,
    request: std::sync::mpsc::SyncSender<NanoDatasetCheckpointRequest>,
    finalization: std::sync::mpsc::Receiver<NanoDatasetCheckpointFinalization>,
}

#[cfg(all(feature = "record", feature = "nano-agent", unix))]
struct NanoDatasetCheckpointMainBridge {
    checkpoint_deadline: Arc<std::sync::OnceLock<Instant>>,
    request: std::sync::mpsc::Receiver<NanoDatasetCheckpointRequest>,
    finalization: std::sync::mpsc::SyncSender<NanoDatasetCheckpointFinalization>,
}

#[cfg(all(feature = "record", feature = "nano-agent", unix))]
const NANO_TERMINAL_RESPONSE_COMPLETION_RESERVE: Duration = Duration::from_secs(3);
#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    feature = "nano-agent",
    feature = "operator-console",
    unix
))]
// Browser polling waits 300 ms and bounds each snapshot/response request body
// at 750 ms. Two seconds covers one already-started snapshot request, the
// inter-poll delay, and the final response-record request.
const NANO_OPERATOR_CONSOLE_RESPONSE_OBSERVATION_GRACE: Duration = Duration::from_secs(2);

#[cfg(all(feature = "record", feature = "nano-agent", unix))]
fn nano_dataset_checkpoint_bridge(
    requested: Arc<AtomicBool>,
    dataset_directory: PathBuf,
) -> (
    NanoDatasetCheckpointWorkerBridge,
    NanoDatasetCheckpointMainBridge,
) {
    let (request_tx, request_rx) = std::sync::mpsc::sync_channel(1);
    let (finalization_tx, finalization_rx) = std::sync::mpsc::sync_channel(1);
    let checkpoint_deadline = Arc::new(std::sync::OnceLock::new());
    (
        NanoDatasetCheckpointWorkerBridge {
            requested: Arc::clone(&requested),
            checkpoint_deadline: Arc::clone(&checkpoint_deadline),
            dataset_directory,
            request: request_tx,
            finalization: finalization_rx,
        },
        NanoDatasetCheckpointMainBridge {
            checkpoint_deadline,
            request: request_rx,
            finalization: finalization_tx,
        },
    )
}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    feature = "nano-agent",
    feature = "operator-console",
    unix
))]
struct PendingNanoWarmCheckpoint {
    claimed: kiko_slam::navigation::AgentControlClaimedRequest,
    console_response_pending: bool,
}

#[cfg(feature = "record")]
struct LiveCompatibilityNavigationRuntime {
    coordinator: ShadowNavigationCoordinator<NavigationIngressWriter<File>>,
    #[cfg(feature = "actuation")]
    physical_actuation: Option<LiveMpcControlDriver>,
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
struct LiveWheelsOffQualificationMotionRuntime {
    coordinator: ShadowNavigationCoordinator<NavigationIngressWriter<File>>,
    controller: Option<kiko_slam::navigation::WheelsOffQualificationRuntime>,
    frontend: Option<kiko_slam::navigation::WheelsOffQualificationFrontend>,
    attestation_gate: Option<FreshAttendedMotionAttestationGate>,
    process_running: Arc<AtomicBool>,
    telemetry: kiko_slam::navigation::WheelsOffQualificationTelemetryStore,
    observation: LiveConsoleNavigationObservation,
    initial_health: ConsoleSubsystemHealth,
    accessory_health: NanoAccessoryHealthObserver,
    sensor_health: LiveSensorStreamHealth,
    map_revision: Option<u64>,
    localized: bool,
    fault_injection: kiko_slam::navigation::WheelsOffQualificationLiveFaultState,
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
impl LiveWheelsOffQualificationMotionRuntime {
    fn controller(&self) -> &kiko_slam::navigation::WheelsOffQualificationRuntime {
        self.controller
            .as_ref()
            .expect("qualification controller owner exists until terminal shutdown")
    }

    fn controller_mut(&mut self) -> &mut kiko_slam::navigation::WheelsOffQualificationRuntime {
        self.controller
            .as_mut()
            .expect("qualification controller owner exists until terminal shutdown")
    }

    fn frontend_mut(&mut self) -> &mut kiko_slam::navigation::WheelsOffQualificationFrontend {
        self.frontend
            .as_mut()
            .expect("qualification frontend exists until terminal shutdown")
    }

    fn attestation_gate_mut(&mut self) -> &mut FreshAttendedMotionAttestationGate {
        self.attestation_gate
            .as_mut()
            .expect("qualification attestation gate exists until terminal shutdown")
    }

    fn take_terminal_parts(
        &mut self,
    ) -> (
        kiko_slam::navigation::WheelsOffQualificationRuntime,
        kiko_slam::navigation::WheelsOffQualificationFrontend,
        FreshAttendedMotionAttestationGate,
    ) {
        (
            self.controller
                .take()
                .expect("qualification controller transfers once"),
            self.frontend
                .take()
                .expect("qualification frontend transfers once"),
            self.attestation_gate
                .take()
                .expect("qualification attestation gate transfers once"),
        )
    }
}

#[cfg(feature = "record")]
enum LiveNavigationRuntime {
    Compatibility(Box<LiveCompatibilityNavigationRuntime>),
    #[cfg(all(feature = "actuation", feature = "agent-runtime", unix))]
    Production(Box<LiveProductionMotionRuntime>),
    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    WheelsOffQualification(Box<LiveWheelsOffQualificationMotionRuntime>),
}

#[cfg(feature = "record")]
impl LiveNavigationRuntime {
    #[cfg(all(feature = "nano-agent", feature = "operator-console", unix))]
    fn mark_sensor_closed(&mut self, stream: LiveSensorStream) {
        match self {
            Self::Compatibility(_) => {}
            #[cfg(all(feature = "actuation", feature = "agent-runtime", unix))]
            Self::Production(runtime) => runtime.sensor_health.mark_closed(stream),
            #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
            Self::WheelsOffQualification(runtime) => runtime.sensor_health.mark_closed(stream),
        }
    }

    #[cfg(all(feature = "nano-agent", unix))]
    fn current_map_binding(&self) -> Option<kiko_slam::navigation::CurrentMapEpochBinding> {
        match self {
            Self::Compatibility(runtime) => runtime.coordinator.current_map_binding(),
            #[cfg(all(feature = "actuation", feature = "agent-runtime", unix))]
            Self::Production(runtime) => runtime.owner().coordinator().current_map_binding(),
            #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
            Self::WheelsOffQualification(runtime) => runtime.coordinator.current_map_binding(),
        }
    }

    fn accept_visual(
        &mut self,
        admission: VisualAdmission,
        now: HostMonotonicTimestamp,
    ) -> Result<(), LiveCoordinatorAdmissionError> {
        match self {
            Self::Compatibility(runtime) => runtime
                .coordinator
                .accept_visual(admission, now)
                .map(|_| ()),
            #[cfg(all(feature = "actuation", feature = "agent-runtime", unix))]
            Self::Production(runtime) => {
                let outcome = runtime.owner_mut().accept_visual(admission, now)?;
                #[cfg(all(feature = "nano-agent", feature = "operator-console"))]
                runtime.sensor_health.observe(LiveSensorStream::Visual, now);
                runtime.localized = match &outcome {
                    VisualAdmissionOutcome::Reanchored(state)
                    | VisualAdmissionOutcome::Updated(state) => runtime
                        .owner()
                        .coordinator()
                        .current_map_binding()
                        .is_some_and(|binding| {
                            binding.map_instance_id() == state.map_snapshot().instance_id()
                        }),
                    VisualAdmissionOutcome::ChainBroken(_)
                    | VisualAdmissionOutcome::Rejected(_) => false,
                };
                Ok(())
            }
            #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
            Self::WheelsOffQualification(runtime) => {
                if runtime.fault_injection.forces_localization_lost() {
                    runtime.sensor_health.observe(LiveSensorStream::Visual, now);
                    return Ok(());
                }
                let outcome = runtime.coordinator.accept_visual(admission, now)?;
                runtime.sensor_health.observe(LiveSensorStream::Visual, now);
                let observed_localized = match &outcome {
                    VisualAdmissionOutcome::Reanchored(state)
                    | VisualAdmissionOutcome::Updated(state) => runtime
                        .coordinator
                        .current_map_binding()
                        .is_some_and(|binding| {
                            binding.map_instance_id() == state.map_snapshot().instance_id()
                        }),
                    VisualAdmissionOutcome::ChainBroken(_)
                    | VisualAdmissionOutcome::Rejected(_) => false,
                };
                runtime.localized =
                    observed_localized && !runtime.fault_injection.forces_localization_lost();
                Ok(())
            }
        }
    }

    fn accept_depth(
        &mut self,
        observation: DepthObservation,
        now: HostMonotonicTimestamp,
    ) -> Result<(), LiveCoordinatorAdmissionError> {
        match self {
            Self::Compatibility(runtime) => runtime
                .coordinator
                .accept_depth(observation, now)
                .map(|_| ()),
            #[cfg(all(feature = "actuation", feature = "agent-runtime", unix))]
            Self::Production(runtime) => {
                runtime.owner_mut().accept_depth(observation, now)?;
                #[cfg(all(feature = "nano-agent", feature = "operator-console"))]
                runtime.sensor_health.observe(LiveSensorStream::Depth, now);
                Ok(())
            }
            #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
            Self::WheelsOffQualification(runtime) => {
                if runtime.fault_injection.suppresses_depth_admission() {
                    return Ok(());
                }
                runtime.coordinator.accept_depth(observation, now)?;
                runtime.sensor_health.observe(LiveSensorStream::Depth, now);
                Ok(())
            }
        }
    }

    fn accept_imu(
        &mut self,
        report: ImuReport,
        now: HostMonotonicTimestamp,
    ) -> Result<(), LiveCoordinatorAdmissionError> {
        match self {
            Self::Compatibility(runtime) => runtime.coordinator.accept_imu(report, now).map(|_| ()),
            #[cfg(all(feature = "actuation", feature = "agent-runtime", unix))]
            Self::Production(runtime) => {
                runtime.owner_mut().accept_imu(report, now)?;
                #[cfg(all(feature = "nano-agent", feature = "operator-console"))]
                runtime.sensor_health.observe(LiveSensorStream::Imu, now);
                Ok(())
            }
            #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
            Self::WheelsOffQualification(runtime) => {
                runtime.coordinator.accept_imu(report, now)?;
                runtime.sensor_health.observe(LiveSensorStream::Imu, now);
                Ok(())
            }
        }
    }

    fn accept_global_map(
        &mut self,
        host_arrival: HostMonotonicTimestamp,
        snapshot: &TimedOccupancySnapshot,
    ) -> Result<(), LiveNavigationWorkerError> {
        match self {
            Self::Compatibility(runtime) => runtime
                .coordinator
                .accept_global_map(host_arrival, snapshot.timestamp(), snapshot.snapshot())
                .map(|_| ())
                .map_err(|source| LiveNavigationWorkerError::MapAdmission { source }),
            #[cfg(all(feature = "actuation", feature = "agent-runtime", unix))]
            Self::Production(runtime) => {
                let outcome = runtime
                    .owner_mut()
                    .accept_global_map(host_arrival, snapshot.timestamp(), snapshot.snapshot())
                    .map_err(|source| LiveNavigationWorkerError::ProductionMapAdmission {
                        source,
                    })?;
                runtime.map_revision = Some(outcome.revision());
                if outcome.started_new_epoch() {
                    runtime.localized = false;
                }
                Ok(())
            }
            #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
            Self::WheelsOffQualification(runtime) => {
                let outcome = runtime
                    .coordinator
                    .accept_global_map(host_arrival, snapshot.timestamp(), snapshot.snapshot())
                    .map_err(|source| LiveNavigationWorkerError::MapAdmission { source })?;
                runtime.map_revision = Some(outcome.revision());
                if outcome.started_new_epoch() {
                    runtime.localized = false;
                }
                Ok(())
            }
        }
    }
}

#[cfg(feature = "record")]
fn finalize_live_navigation_coordinator(
    coordinator: ShadowNavigationCoordinator<NavigationIngressWriter<File>>,
) -> Result<FinalizedLiveNavigationJournal, LiveNavigationWorkerError> {
    let finalized = coordinator
        .into_journal()
        .finish_with_descriptor()
        .map_err(|source| LiveNavigationWorkerError::JournalFinalization { source })?;
    let (mut file, descriptor) = finalized.into_parts();
    file.sync_all()
        .map_err(|source| LiveNavigationWorkerError::JournalSync { source })?;
    std::io::Seek::seek(&mut file, std::io::SeekFrom::Start(0))
        .map_err(|source| LiveNavigationWorkerError::JournalSeek { source })?;
    let declared_count = usize::try_from(descriptor.record_count()).map_err(|_| {
        LiveNavigationWorkerError::JournalRecordCountOutOfRange {
            record_count: descriptor.record_count(),
        }
    })?;
    let capacity = NavigationIngressCapacity::try_new(declared_count.max(1))
        .map_err(|source| LiveNavigationWorkerError::JournalCapacity { source })?;
    let mut reader = NavigationIngressReader::new(file, descriptor.recording_id(), capacity)
        .map_err(|source| LiveNavigationWorkerError::JournalVerification { source })?;
    let mut final_map_identity = None;
    while let Some(record) = reader
        .next_record()
        .map_err(|source| LiveNavigationWorkerError::JournalVerification { source })?
    {
        match record.event() {
            NavigationIngressEvent::MapEpochStarted(_) => final_map_identity = None,
            NavigationIngressEvent::AcceptedGlobalMap(map) => {
                final_map_identity = Some(FinalizedJournalMapIdentity {
                    map_epoch_id: map.map_epoch_id(),
                    revision: map.revision(),
                });
            }
            NavigationIngressEvent::VisualAttempt(_)
            | NavigationIngressEvent::ImuReport(_)
            | NavigationIngressEvent::AcceptedDepth(_)
            | NavigationIngressEvent::PointGoal(_)
            | NavigationIngressEvent::ControlTick(_)
            | NavigationIngressEvent::QualificationAppliedStep(_) => {}
        }
    }
    Ok(FinalizedLiveNavigationJournal {
        descriptor,
        final_map_identity,
    })
}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    unix
))]
fn fail_production_owner_start(
    primary: LiveProductionOwnerStartPrimary,
    physical_driver: &mut LiveMpcControlDriver,
) -> LiveProductionOwnerStartError {
    let controller_stop = match physical_driver.disarm() {
        Ok(receipt) => LiveProductionControllerStop::Confirmed(receipt),
        Err(source) => {
            let AgentLiveActuationDisposition::LatchFault(fault) =
                classify_live_actuation_error(&source);
            if fault.controller_stop() == AgentControllerStopKnowledge::Confirmed {
                LiveProductionControllerStop::DisarmFailedStopConfirmed(source)
            } else {
                LiveProductionControllerStop::Uncertain(source)
            }
        }
    };
    LiveProductionOwnerStartError {
        primary,
        controller_stop,
        lifecycle_cleanup: None,
        socket_shutdown: None,
    }
}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    feature = "nano-agent",
    unix
))]
fn fail_started_production_owner(
    primary: LiveProductionOwnerStartPrimary,
    owner: ProductionLiveMotionOwner,
    socket_task: AgentControlSocketTask,
) -> LiveProductionMotionStartFailure {
    socket_task.request_shutdown();
    let terminal = owner.shutdown();
    let (coordinator, lifecycle_cleanup, controller_stop, _last_physical_state) =
        terminal.into_parts();
    let controller_stop = match controller_stop {
        LiveMotionTerminalStop::Confirmed(receipt) => {
            LiveProductionControllerStop::Confirmed(receipt)
        }
        LiveMotionTerminalStop::DisarmFailedStopConfirmed(source) => {
            LiveProductionControllerStop::DisarmFailedStopConfirmed(source)
        }
        LiveMotionTerminalStop::Uncertain(source) => {
            LiveProductionControllerStop::Uncertain(source)
        }
    };
    let socket_shutdown = Some(socket_task.shutdown());
    Box::new((
        coordinator,
        LiveProductionOwnerStartError {
            primary,
            controller_stop,
            lifecycle_cleanup,
            socket_shutdown,
        },
    ))
}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    unix
))]
#[allow(clippy::too_many_arguments)]
fn start_production_motion_runtime(
    coordinator: ShadowNavigationCoordinator<NavigationIngressWriter<File>>,
    coordinator_clock_epoch: NavigationClockEpoch,
    clock_origin: Instant,
    input: LiveAgentMotionStartInput,
    accessory_health: NanoAccessoryHealthObserver,
    #[cfg(all(feature = "operator-console", feature = "nano-agent"))]
    rerun_diagnostics_url: ConsoleRerunDiagnosticsUrl,
    running: Arc<AtomicBool>,
) -> Result<LiveProductionMotionRuntime, LiveProductionMotionStartFailure> {
    let LiveAgentMotionStartInput {
        policy,
        authority,
        mut physical_driver,
        admitted_actuation_config,
        coordinator_actuation_config,
        kind,
    } = input;
    #[cfg(not(all(feature = "operator-console", feature = "nano-agent")))]
    let _ = kind;
    let control = policy.control().clone();
    let live_mode_policy: NanoLiveModePolicy = *policy.live_mode_policy();

    let supervisor_clock_epoch = authority.clock_epoch();
    if supervisor_clock_epoch != coordinator_clock_epoch {
        let source = fail_production_owner_start(
            LiveProductionOwnerStartPrimary::ClockEpochMismatch {
                coordinator_origin_ns: coordinator_clock_epoch.origin().as_nanos(),
                supervisor_origin_ns: supervisor_clock_epoch.origin().as_nanos(),
            },
            &mut physical_driver,
        );
        return Err(Box::new((coordinator, source)));
    }
    if admitted_actuation_config != coordinator_actuation_config {
        let source = fail_production_owner_start(
            LiveProductionOwnerStartPrimary::ActuationConfigMismatch,
            &mut physical_driver,
        );
        return Err(Box::new((coordinator, source)));
    }
    if coordinator.motion_mode() != CoordinatorMotionModeV1::MappingOnly {
        let source = fail_production_owner_start(
            LiveProductionOwnerStartPrimary::CoordinatorNotMappingOnly {
                actual: coordinator.motion_mode(),
            },
            &mut physical_driver,
        );
        return Err(Box::new((coordinator, source)));
    }

    let plant = coordinator.safety().solver().model();
    let manual_policy = match live_mode_policy
        .manual()
        .config()
        .map(|manual| {
            manual
                .bind_to_plant(plant)
                .map(AgentManualRuntimePolicy::from)
        })
        .transpose()
    {
        Ok(policy) => policy,
        Err(source) => {
            let source = fail_production_owner_start(
                LiveProductionOwnerStartPrimary::ManualPlantBinding(source),
                &mut physical_driver,
            );
            return Err(Box::new((coordinator, source)));
        }
    };

    let (socket_config, runtime_queue_capacity, operator_console_config) = control.into_parts();
    #[cfg(all(feature = "operator-console", feature = "nano-agent"))]
    let terminal_response_timeout = socket_config.timeouts().terminal_response();
    #[cfg(all(feature = "operator-console", feature = "nano-agent"))]
    let manual_command_envelope = match manual_policy
        .map(|policy| {
            let drive = policy.drive();
            ConsoleManualCommandEnvelope::parse(
                drive.maximum_abs_forward_velocity_mps(),
                drive.maximum_abs_yaw_rate_rad_s(),
                operator_console_config.manual_command_forward_velocity_mps(),
                operator_console_config.manual_command_yaw_rate_rad_s(),
            )
        })
        .transpose()
    {
        Ok(envelope) => envelope,
        Err(source) => {
            let source = fail_production_owner_start(
                LiveProductionOwnerStartPrimary::ConsoleManualEnvelope(source),
                &mut physical_driver,
            );
            return Err(Box::new((coordinator, source)));
        }
    };
    #[cfg(not(all(feature = "operator-console", feature = "nano-agent")))]
    let _ = operator_console_config;
    #[cfg(not(all(feature = "operator-console", feature = "nano-agent")))]
    let _ = accessory_health;
    let socket_clock =
        AgentControlMonotonicOrigin::new(clock_origin, coordinator_clock_epoch.origin());
    #[cfg(all(feature = "operator-console", feature = "nano-agent"))]
    let (socket_task, receiver, typed_ingress) =
        match AgentControlSocketTask::bind_and_spawn_with_typed_ingress(
            socket_config,
            socket_clock,
            runtime_queue_capacity,
            Arc::clone(&running),
        ) {
            Ok(runtime) => runtime,
            Err(source) => {
                let source = fail_production_owner_start(
                    LiveProductionOwnerStartPrimary::ControlSocket(source),
                    &mut physical_driver,
                );
                return Err(Box::new((coordinator, source)));
            }
        };
    #[cfg(not(all(feature = "operator-console", feature = "nano-agent")))]
    let (socket_task, receiver) = match AgentControlSocketTask::bind_and_spawn(
        socket_config,
        socket_clock,
        runtime_queue_capacity,
        Arc::clone(&running),
    ) {
        Ok(runtime) => runtime,
        Err(source) => {
            let source = fail_production_owner_start(
                LiveProductionOwnerStartPrimary::ControlSocket(source),
                &mut physical_driver,
            );
            return Err(Box::new((coordinator, source)));
        }
    };

    #[cfg(all(feature = "operator-console", feature = "nano-agent"))]
    let dispatcher = AgentControlDispatcher::new_with_unified_console_authority(
        receiver,
        socket_clock,
        AgentManualControlCore::new(authority, manual_policy),
    );
    #[cfg(not(all(feature = "operator-console", feature = "nano-agent")))]
    let dispatcher = AgentControlDispatcher::new(
        receiver,
        socket_clock,
        AgentManualControlCore::new(authority, manual_policy),
    );
    let owner = LiveMotionOwner::new(
        dispatcher,
        coordinator,
        physical_driver,
        InstantHostClock::new(clock_origin),
        live_mode_policy,
    );
    #[cfg(all(feature = "operator-console", feature = "nano-agent"))]
    let console = {
        let mut initial_snapshot = OperatorConsoleSnapshot::unknown(
            ConsoleSnapshotRevision::parse(1)
                .expect("the static initial console revision is nonzero"),
            kind.console(),
        );
        initial_snapshot.rerun_diagnostics_url = Some(rerun_diagnostics_url);
        initial_snapshot.manual_command_envelope = manual_command_envelope;
        initial_snapshot.runtime = Some(
            owner
                .dispatcher()
                .control_status(AgentMapStateV1::UNAVAILABLE)
                .runtime(),
        );
        initial_snapshot.health = ConsoleSubsystemHealth {
            stm32: Some(ConsoleHealth::Ready),
            head: Some(
                if policy
                    .head()
                    .return_to_natural_and_hold_continuously()
                    .is_some()
                {
                    ConsoleHealth::Ready
                } else {
                    ConsoleHealth::Unavailable
                },
            ),
            eyes: Some(if policy.eye().static_runtime().is_some() {
                ConsoleHealth::Ready
            } else {
                ConsoleHealth::Unavailable
            }),
            oak: Some(ConsoleHealth::Degraded),
            slam: Some(ConsoleHealth::Degraded),
        };
        initial_snapshot.health =
            match refresh_console_accessory_health(initial_snapshot.health, &accessory_health) {
                Ok(health) => health,
                Err(source) => {
                    return Err(fail_started_production_owner(
                        LiveProductionOwnerStartPrimary::AccessoryHealth(source),
                        owner,
                        socket_task,
                    ));
                }
            };
        let (console_handle, console_receiver) = operator_console(
            OperatorConsoleLimits::production_default(),
            initial_snapshot,
        );
        let frontend = match NanoOperatorConsoleFrontend::start(
            &operator_console_config,
            socket_clock,
            console_handle.clone(),
        ) {
            Ok(frontend) => frontend,
            Err(source) => {
                return Err(fail_started_production_owner(
                    LiveProductionOwnerStartPrimary::ConsoleFrontend(source),
                    owner,
                    socket_task,
                ));
            }
        };
        eprintln!(
            "operator console ready on {}; read the per-boot capability from {} through the private runtime directory",
            frontend.bound_address(),
            operator_console_config
                .capability_path()
                .as_path()
                .display(),
        );
        LiveProductionConsoleRuntime {
            adapter: Some(OperatorConsoleRuntimeAdapter::new(
                console_handle,
                console_receiver,
                typed_ingress,
            )),
            frontend: Some(frontend),
            observation: LiveConsoleNavigationObservation {
                next_snapshot_revision: Some(2),
                map: None,
                last_requested_actuation: None,
                last_applied: None,
                stop_certainty: None,
                successful_solver_duration_ns: None,
                rerun_diagnostics_url,
            },
        }
    };
    Ok(LiveProductionMotionRuntime {
        owner: Some(owner),
        socket_task: Some(socket_task),
        #[cfg(all(feature = "operator-console", feature = "nano-agent"))]
        terminal_response_timeout,
        #[cfg(all(feature = "operator-console", feature = "nano-agent"))]
        console: Some(console),
        #[cfg(all(feature = "operator-console", feature = "nano-agent"))]
        accessory_health,
        #[cfg(all(feature = "operator-console", feature = "nano-agent"))]
        sensor_health: LiveSensorStreamHealth::awaiting_first_samples(),
        map_revision: None,
        localized: false,
        #[cfg(all(feature = "operator-console", feature = "nano-agent"))]
        terminal_checkpoint_pending: false,
    })
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
type LiveWheelsOffQualificationMotionStartFailure = Box<(
    ShadowNavigationCoordinator<NavigationIngressWriter<File>>,
    LiveWheelsOffQualificationMotionStartError,
)>;

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
#[derive(Debug)]
enum LiveWheelsOffQualificationMotionStartPrimary {
    Receipt(ConsoleReceiptProjectionError),
    Telemetry(kiko_slam::navigation::WheelsOffQualificationTelemetryError),
    Runtime {
        source: kiko_slam::navigation::WheelsOffQualificationRuntimeStartError,
        frontend_shutdown: kiko_slam::navigation::WheelsOffQualificationFrontendShutdownEvidence,
    },
    FrontendExited {
        frontend_shutdown: kiko_slam::navigation::WheelsOffQualificationFrontendShutdownEvidence,
    },
    Frontend(kiko_slam::navigation::WheelsOffQualificationFrontendStartError),
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
impl std::fmt::Display for LiveWheelsOffQualificationMotionStartPrimary {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Receipt(source) => source.fmt(formatter),
            Self::Telemetry(source) => source.fmt(formatter),
            Self::Runtime {
                source,
                frontend_shutdown,
            } => write!(
                formatter,
                "{source}; already-bound frontend shutdown evidence: {frontend_shutdown:?}"
            ),
            Self::FrontendExited { frontend_shutdown } => write!(
                formatter,
                "qualification frontend exited before motion enablement; shutdown evidence: {frontend_shutdown:?}"
            ),
            Self::Frontend(source) => source.fmt(formatter),
        }
    }
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
impl std::error::Error for LiveWheelsOffQualificationMotionStartPrimary {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Receipt(source) => Some(source),
            Self::Telemetry(source) => Some(source),
            Self::Runtime { source, .. } => Some(source),
            Self::FrontendExited { .. } => None,
            Self::Frontend(source) => Some(source),
        }
    }
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
#[derive(Debug)]
enum LiveWheelsOffQualificationStartStop {
    BootstrapExact {
        boot_id: u64,
        stop_request_id: u32,
    },
    RuntimeShutdown(
        Result<
            kiko_slam::navigation::WheelsOffQualificationRuntimeShutdown,
            Box<kiko_slam::navigation::WheelsOffQualificationRuntimeError>,
        >,
    ),
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
impl std::fmt::Display for LiveWheelsOffQualificationStartStop {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BootstrapExact {
                boot_id,
                stop_request_id,
            } => write!(
                formatter,
                "bootstrap exact stop (boot_id={boot_id}, request_id={stop_request_id})"
            ),
            Self::RuntimeShutdown(Ok(shutdown)) => write!(
                formatter,
                "runtime exact stop ({:?})",
                shutdown.terminal_completion
            ),
            Self::RuntimeShutdown(Err(source)) => {
                write!(formatter, "runtime shutdown failed: {source}")
            }
        }
    }
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
#[derive(Debug)]
struct LiveWheelsOffQualificationMotionStartError {
    primary: LiveWheelsOffQualificationMotionStartPrimary,
    controller_stop: LiveWheelsOffQualificationStartStop,
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
impl std::fmt::Display for LiveWheelsOffQualificationMotionStartError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "wheels-off qualification owner startup failed: {}; controller stop evidence: {}",
            self.primary, self.controller_stop,
        )
    }
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
impl std::error::Error for LiveWheelsOffQualificationMotionStartError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.primary)
    }
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
fn fail_wheels_off_qualification_before_runtime(
    coordinator: ShadowNavigationCoordinator<NavigationIngressWriter<File>>,
    primary: LiveWheelsOffQualificationMotionStartPrimary,
    boot_id: u64,
    stop_request_id: u32,
) -> LiveWheelsOffQualificationMotionStartFailure {
    Box::new((
        coordinator,
        LiveWheelsOffQualificationMotionStartError {
            primary,
            controller_stop: LiveWheelsOffQualificationStartStop::BootstrapExact {
                boot_id,
                stop_request_id,
            },
        },
    ))
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
fn start_wheels_off_qualification_motion_runtime(
    coordinator: ShadowNavigationCoordinator<NavigationIngressWriter<File>>,
    mut input: LiveWheelsOffQualificationMotionInput,
    actual_runtime_service_interval: Duration,
    process_running: Arc<AtomicBool>,
) -> Result<LiveWheelsOffQualificationMotionRuntime, LiveWheelsOffQualificationMotionStartFailure> {
    let (
        stopped_controller,
        initial_zero,
        initial_stop,
        limits,
        admitted_runtime_service_interval,
        preflight,
        profile,
        frontend_config,
        initial_health,
        accessory_health,
        rerun_diagnostics_url,
        fault_injection,
    ) = input.take_for_owner();
    let boot_id = initial_stop.observed_boot_id().get();
    let stop_request_id = initial_stop.request_id().get();
    let last_applied = match ConsoleAppliedReceipt::from_verified(&initial_zero) {
        Ok(receipt) => receipt,
        Err(source) => {
            return Err(fail_wheels_off_qualification_before_runtime(
                coordinator,
                LiveWheelsOffQualificationMotionStartPrimary::Receipt(source),
                boot_id,
                stop_request_id,
            ));
        }
    };
    let stop_certainty = ConsoleStopCertainty::from_verified_disarm(&initial_stop);
    let mut initial_base = OperatorConsoleSnapshot::unknown(
        ConsoleSnapshotRevision::parse(1).expect("static qualification revision is nonzero"),
        ConsoleRuntimeAuthorityKind::WheelsOffQualification,
    );
    initial_base.runtime = Some(AgentRuntimeStateV1::ReadyStopped);
    initial_base.health = initial_health;
    initial_base.last_applied = Some(last_applied);
    initial_base.stop_certainty = Some(stop_certainty);
    initial_base.rerun_diagnostics_url = Some(rerun_diagnostics_url);
    let (console, receiver) = kiko_slam::navigation::wheels_off_qualification_console(profile);
    let telemetry = match kiko_slam::navigation::WheelsOffQualificationTelemetryStore::parse(
        profile,
        initial_base,
        console.snapshot(),
    ) {
        Ok(telemetry) => telemetry,
        Err(source) => {
            return Err(fail_wheels_off_qualification_before_runtime(
                coordinator,
                LiveWheelsOffQualificationMotionStartPrimary::Telemetry(source),
                boot_id,
                stop_request_id,
            ));
        }
    };
    // Bind the loopback UI while the console's motion boundary is still
    // closed. Motion-capable requests fail with `motion_attestation_pending`
    // until the stopped runtime owner and fresh attended attestation are both
    // ready; ordinary Stop and the one-way safety stop remain available.
    let mut frontend = match kiko_slam::navigation::WheelsOffQualificationFrontend::start(
        &frontend_config,
        console.clone(),
        telemetry.clone(),
        profile,
    ) {
        Ok(frontend) => frontend,
        Err(source) => {
            return Err(fail_wheels_off_qualification_before_runtime(
                coordinator,
                LiveWheelsOffQualificationMotionStartPrimary::Frontend(source),
                boot_id,
                stop_request_id,
            ));
        }
    };
    eprintln!(
        "wheels-off qualification console ready but motion-attestation-pending on {}; capability={}; raw_timer_pwm_cap={} test_magnitude={} deadman_ms={}; autonomous actuation disabled (SLAM/MPC shadow only)",
        frontend.bound_address(),
        frontend_config.capability_path().display(),
        limits.effective_max_abs_pwm_percent(),
        limits.manual_test_magnitude_timer_pwm_percent(),
        limits.manual_deadman().as_millis(),
    );
    if let Some(frontend_shutdown) = frontend.poll_unexpected_exit() {
        return Err(fail_wheels_off_qualification_before_runtime(
            coordinator,
            LiveWheelsOffQualificationMotionStartPrimary::FrontendExited { frontend_shutdown },
            boot_id,
            stop_request_id,
        ));
    }
    let mut controller = match kiko_slam::navigation::WheelsOffQualificationRuntime::try_new_pending(
        stopped_controller,
        initial_zero,
        initial_stop,
        limits,
        admitted_runtime_service_interval,
        actual_runtime_service_interval,
        console.clone(),
        receiver,
    ) {
        Ok(runtime) => runtime,
        Err(source) => {
            let frontend_shutdown = frontend.shutdown();
            return Err(fail_wheels_off_qualification_before_runtime(
                coordinator,
                LiveWheelsOffQualificationMotionStartPrimary::Runtime {
                    source,
                    frontend_shutdown,
                },
                boot_id,
                stop_request_id,
            ));
        }
    };
    if let Some(frontend_shutdown) = frontend.poll_unexpected_exit() {
        let controller_stop = controller.shutdown().map_err(Box::new);
        return Err(Box::new((
            coordinator,
            LiveWheelsOffQualificationMotionStartError {
                primary: LiveWheelsOffQualificationMotionStartPrimary::FrontendExited {
                    frontend_shutdown,
                },
                controller_stop: LiveWheelsOffQualificationStartStop::RuntimeShutdown(
                    controller_stop,
                ),
            },
        )));
    }
    let attestation_gate = FreshAttendedMotionAttestationGate::AwaitingReadOnlyCycle(
        FreshAttendedMotionAttestationInput {
            preflight,
            console: console.clone(),
            process_running: Arc::clone(&process_running),
        },
    );
    eprintln!(
        "wheels-off qualification stopped runtime is live; OAK SLAM, occupancy, Rerun, accessories, console, and exact applied-zero STM32 evidence are starting while fresh attended motion attestation remains locked"
    );
    Ok(LiveWheelsOffQualificationMotionRuntime {
        coordinator,
        controller: Some(controller),
        frontend: Some(frontend),
        attestation_gate: Some(attestation_gate),
        process_running,
        telemetry,
        observation: LiveConsoleNavigationObservation {
            next_snapshot_revision: Some(2),
            map: None,
            last_requested_actuation: None,
            last_applied: Some(last_applied),
            stop_certainty: Some(stop_certainty),
            successful_solver_duration_ns: None,
            rerun_diagnostics_url,
        },
        initial_health,
        accessory_health,
        sensor_health: LiveSensorStreamHealth::awaiting_first_samples(),
        map_revision: None,
        localized: false,
        fault_injection: kiko_slam::navigation::WheelsOffQualificationLiveFaultState::new(
            fault_injection,
        ),
    })
}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    unix
))]
fn abnormal_production_socket_exit(
    exit: AgentControlSocketTaskExit,
) -> Option<LiveNavigationWorkerError> {
    match exit {
        AgentControlSocketTaskExit::Shutdown {
            cleanup: AgentControlSocketCleanupOutcome::RemovedCreatedSocket,
        } => None,
        exit => Some(LiveNavigationWorkerError::ProductionSocketExit {
            exit: Box::new(exit),
        }),
    }
}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    unix
))]
const fn production_period_requires_motion_tick(
    terminal_transition_requested: bool,
    periodic_tick_deferred: bool,
    already_applied: bool,
) -> bool {
    !terminal_transition_requested && !periodic_tick_deferred && !already_applied
}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    feature = "nano-agent",
    feature = "operator-console",
    unix
))]
fn observe_production_console_physical_state(
    observation: &mut LiveConsoleNavigationObservation,
    downstream_request_id: Option<kiko_slam::navigation::ConsoleDownstreamRequestId>,
    event: &LivePhysicalStateEvent<
        AppliedCommandReceipt,
        CoordinatorTickOutcome<NavigationIngressStreamWriteError>,
    >,
) -> Result<(), LiveProductionConsoleProjectionError> {
    match event {
        LivePhysicalStateEvent::CoordinatorTick(applied) => {
            observation.last_requested_actuation =
                Some(ConsoleRequestedActuation::from_checked_record(
                    downstream_request_id,
                    applied.diagnostic().decision().record(),
                ));
            observation.last_applied = Some(
                ConsoleAppliedReceipt::from_verified(applied.receipt())
                    .map_err(LiveProductionConsoleProjectionError::Receipt)?,
            );
            observation.stop_certainty = Some(ConsoleStopCertainty::from_verified_applied(
                applied.receipt(),
            ));
            observation.successful_solver_duration_ns =
                match applied.diagnostic().decision().outcome() {
                    SafetyDecisionOutcome::Controller(controller) => {
                        let status = controller.solve_status();
                        checked_monotonic_duration_ns(
                            u128::from(status.started_at().as_nanos()),
                            u128::from(status.observed_at().as_nanos()),
                        )
                    }
                    SafetyDecisionOutcome::Stopped(_) => None,
                };
        }
        LivePhysicalStateEvent::LifecycleZero(applied) => {
            observation.last_requested_actuation = None;
            observation.last_applied = Some(
                ConsoleAppliedReceipt::from_verified(applied.receipt())
                    .map_err(LiveProductionConsoleProjectionError::Receipt)?,
            );
            observation.stop_certainty = Some(ConsoleStopCertainty::from_verified_applied(
                applied.receipt(),
            ));
            observation.successful_solver_duration_ns = None;
        }
        LivePhysicalStateEvent::ActuationFault { .. } => {
            observation.last_requested_actuation = None;
            observation.stop_certainty = Some(ConsoleStopCertainty::uncertain());
            observation.successful_solver_duration_ns = None;
        }
    }
    Ok(())
}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    feature = "nano-agent",
    feature = "operator-console",
    unix
))]
fn compose_console_map_pose(
    coordinator: &ShadowNavigationCoordinator<NavigationIngressWriter<File>>,
) -> Result<Option<ConsolePose2>, LiveProductionConsoleProjectionError> {
    let Some(current) = coordinator.odometry().current() else {
        return Ok(None);
    };
    let base_to_odom = current.base_to_odom();
    let odom_to_map = current.odom_to_map();
    let base_x_odom = base_to_odom.source_origin_x_in_destination_m();
    let base_y_odom = base_to_odom.source_origin_y_in_destination_m();
    let base_yaw_odom = base_to_odom.source_yaw_in_destination_rad();
    let odom_x_map = odom_to_map.source_origin_x_in_destination_m();
    let odom_y_map = odom_to_map.source_origin_y_in_destination_m();
    let odom_yaw_map = odom_to_map.source_yaw_in_destination_rad();
    compose_console_pose_components(
        base_x_odom,
        base_y_odom,
        base_yaw_odom,
        odom_x_map,
        odom_y_map,
        odom_yaw_map,
    )
    .map(Some)
}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    feature = "nano-agent",
    feature = "operator-console",
    unix
))]
fn compose_console_pose_components(
    base_x_odom: f64,
    base_y_odom: f64,
    base_yaw_odom: f64,
    odom_x_map: f64,
    odom_y_map: f64,
    odom_yaw_map: f64,
) -> Result<ConsolePose2, LiveProductionConsoleProjectionError> {
    let (sin_yaw, cos_yaw) = odom_yaw_map.sin_cos();
    let base_x_map = cos_yaw.mul_add(base_x_odom, (-sin_yaw).mul_add(base_y_odom, odom_x_map));
    let base_y_map = sin_yaw.mul_add(base_x_odom, cos_yaw.mul_add(base_y_odom, odom_y_map));
    let raw_yaw = odom_yaw_map + base_yaw_odom;
    let base_yaw_map = raw_yaw.sin().atan2(raw_yaw.cos());
    ConsolePose2::parse(base_x_map, base_y_map, base_yaw_map)
        .map_err(LiveProductionConsoleProjectionError::Numeric)
}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    feature = "nano-agent",
    feature = "operator-console",
    unix
))]
fn console_points_from_map_points(
    points: impl IntoIterator<Item = [f64; 2]>,
) -> Result<Vec<ConsolePoint2>, LiveProductionConsoleProjectionError> {
    let points = points
        .into_iter()
        .map(|[x_m, y_m]| {
            ConsolePoint2::parse(x_m, y_m).map_err(LiveProductionConsoleProjectionError::Numeric)
        })
        .collect::<Result<Vec<_>, _>>()?;
    ConsoleNavigationSnapshot::parse_path(points)
        .map_err(LiveProductionConsoleProjectionError::Path)
}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    feature = "nano-agent",
    feature = "operator-console",
    unix
))]
fn transform_console_odom_point_components(
    x_odom_m: f64,
    y_odom_m: f64,
    odom_x_map_m: f64,
    odom_y_map_m: f64,
    odom_yaw_map_rad: f64,
) -> Result<ConsolePoint2, LiveProductionConsoleProjectionError> {
    let (sin_yaw, cos_yaw) = odom_yaw_map_rad.sin_cos();
    let x_map_m = cos_yaw.mul_add(x_odom_m, (-sin_yaw).mul_add(y_odom_m, odom_x_map_m));
    let y_map_m = sin_yaw.mul_add(x_odom_m, cos_yaw.mul_add(y_odom_m, odom_y_map_m));
    ConsolePoint2::parse(x_map_m, y_map_m).map_err(LiveProductionConsoleProjectionError::Numeric)
}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    feature = "nano-agent",
    feature = "operator-console",
    unix
))]
const fn console_localized_navigation_visible(
    localization: Option<AgentLocalizationStateV1>,
) -> bool {
    matches!(localization, Some(AgentLocalizationStateV1::Localized))
}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    feature = "nano-agent",
    feature = "operator-console",
    unix
))]
const fn console_current_solver_path_visible(
    localization: Option<AgentLocalizationStateV1>,
    successful_solver_duration_ns: Option<u64>,
) -> bool {
    console_localized_navigation_visible(localization) && successful_solver_duration_ns.is_some()
}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    feature = "nano-agent",
    feature = "operator-console",
    unix
))]
fn build_production_console_navigation_snapshot(
    coordinator: &ShadowNavigationCoordinator<NavigationIngressWriter<File>>,
    observation: &LiveConsoleNavigationObservation,
    localization: Option<AgentLocalizationStateV1>,
    tick_timing: LiveControlTickTiming,
) -> Result<ConsoleNavigationSnapshot, LiveProductionConsoleProjectionError> {
    let path = coordinator
        .global_path()
        .map(|path| {
            console_points_from_map_points(path.points().iter().map(|point| point.as_array()))
        })
        .transpose()?;
    let goal = coordinator
        .current_goal()
        .map(|goal| {
            let [x_m, y_m] = goal.point().as_array();
            ConsolePoint2::parse(x_m, y_m).map_err(LiveProductionConsoleProjectionError::Numeric)
        })
        .transpose()?;
    let localized_navigation_visible = console_localized_navigation_visible(localization);
    let current_solver_path_visible = console_current_solver_path_visible(
        localization,
        observation.successful_solver_duration_ns,
    );
    let current_odometry = localized_navigation_visible
        .then(|| coordinator.odometry().current())
        .flatten();
    let mpc_predicted_path = match (
        current_solver_path_visible,
        current_odometry,
        coordinator.safety().last_success_trajectory(),
    ) {
        (true, Some(current), Some(trajectory)) => {
            let odom_to_map = current.odom_to_map();
            let odom_x_map_m = odom_to_map.source_origin_x_in_destination_m();
            let odom_y_map_m = odom_to_map.source_origin_y_in_destination_m();
            let odom_yaw_map_rad = odom_to_map.source_yaw_in_destination_rad();
            let points = trajectory
                .points()
                .iter()
                .map(|point| {
                    let [x_odom_m, y_odom_m] = point.pose().position().as_array();
                    transform_console_odom_point_components(
                        x_odom_m,
                        y_odom_m,
                        odom_x_map_m,
                        odom_y_map_m,
                        odom_yaw_map_rad,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            Some(
                ConsoleNavigationSnapshot::parse_path(points)
                    .map_err(LiveProductionConsoleProjectionError::Path)?,
            )
        }
        (false, _, _) | (_, None, _) | (_, _, None) => None,
    };
    Ok(ConsoleNavigationSnapshot {
        pose: if localized_navigation_visible {
            compose_console_map_pose(coordinator)?
        } else {
            None
        },
        path,
        goal,
        mpc_predicted_path,
        solver_duration_ns: observation.successful_solver_duration_ns,
        control_tick_lateness_ns: Some(tick_timing.current_lateness_ns),
    })
}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    feature = "nano-agent",
    feature = "operator-console",
    unix
))]
fn project_console_actual_authority(
    owner: &ProductionLiveMotionOwner,
    adapter: &OperatorConsoleRuntimeAdapter,
) -> Result<Option<ConsoleActualAuthority>, LiveProductionConsoleProjectionError> {
    let owner_authority = owner
        .active_motion_authority()
        .map_err(LiveProductionConsoleProjectionError::AuthorityState)?;
    let console_authority = adapter
        .retained_authority()
        .map_err(LiveProductionConsoleProjectionError::AuthorityAdapter)?
        .map(|authority| {
            (
                authority.kind(),
                authority.downstream_request_id().get(),
                authority.source(),
            )
        });
    project_console_actual_authority_state(owner_authority, console_authority)
}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    feature = "nano-agent",
    feature = "operator-console",
    unix
))]
fn project_console_actual_authority_state(
    owner_authority: Option<LiveMotionAuthorityState>,
    console_authority: Option<(OperatorConsoleRetainedAuthorityKind, u64, ConsoleSourceKind)>,
) -> Result<Option<ConsoleActualAuthority>, LiveProductionConsoleProjectionError> {
    let Some(owner_authority) = owner_authority else {
        return match console_authority {
            None => Ok(None),
            Some((console, _, _)) => {
                Err(LiveProductionConsoleProjectionError::ConsoleAuthorityWithoutOwner { console })
            }
        };
    };
    let Some(console_authority) = console_authority else {
        return Err(
            LiveProductionConsoleProjectionError::OwnerAuthorityWithoutConsole {
                owner: owner_authority,
            },
        );
    };
    let (authority_lease_id, mode) = match owner_authority {
        LiveMotionAuthorityState::Manual { lease_id } => {
            (lease_id.get(), ConsoleActualAuthorityMode::Manual)
        }
        LiveMotionAuthorityState::Autonomous { lease_id, mode } => (
            lease_id.get(),
            match mode {
                kiko_slam::navigation::AgentAutonomousMode::Explore => {
                    ConsoleActualAuthorityMode::FrontierExplore
                }
                kiko_slam::navigation::AgentAutonomousMode::PointGoal => {
                    ConsoleActualAuthorityMode::PointGoal
                }
            },
        ),
    };
    let matching_console_authority = matches!(
        (owner_authority, console_authority.0),
        (
            LiveMotionAuthorityState::Manual { .. },
            OperatorConsoleRetainedAuthorityKind::Manual
        ) | (
            LiveMotionAuthorityState::Autonomous {
                mode: kiko_slam::navigation::AgentAutonomousMode::Explore,
                ..
            },
            OperatorConsoleRetainedAuthorityKind::Autonomous(
                kiko_slam::navigation::AgentAutonomousMode::Explore
            )
        ) | (
            LiveMotionAuthorityState::Autonomous {
                mode: kiko_slam::navigation::AgentAutonomousMode::PointGoal,
                ..
            },
            OperatorConsoleRetainedAuthorityKind::Autonomous(
                kiko_slam::navigation::AgentAutonomousMode::PointGoal
            )
        )
    );
    if !matching_console_authority {
        return Err(
            LiveProductionConsoleProjectionError::ConsoleAuthorityModeMismatch {
                owner: owner_authority,
                console: console_authority.0,
            },
        );
    }
    Ok(Some(ConsoleActualAuthority {
        source: match console_authority.2 {
            ConsoleSourceKind::Operator => ConsoleActualAuthoritySource::Operator,
            ConsoleSourceKind::Agent => ConsoleActualAuthoritySource::Agent,
        },
        mode,
        authority_lease_id,
        console_downstream_request_id: Some(console_authority.1),
    }))
}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    feature = "nano-agent",
    feature = "operator-console",
    unix
))]
const fn console_health_from_accessory(health: NanoAccessoryComponentHealth) -> ConsoleHealth {
    match health {
        NanoAccessoryComponentHealth::Ready => ConsoleHealth::Ready,
        NanoAccessoryComponentHealth::Degraded => ConsoleHealth::Degraded,
        NanoAccessoryComponentHealth::Faulted => ConsoleHealth::Faulted,
    }
}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    feature = "nano-agent",
    feature = "operator-console",
    unix
))]
fn refresh_console_accessory_health(
    health: ConsoleSubsystemHealth,
    observer: &NanoAccessoryHealthObserver,
) -> Result<ConsoleSubsystemHealth, NanoAccessoryHealthStatusError> {
    let accessory = observer.snapshot()?;
    Ok(project_console_accessory_health(health, accessory))
}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    feature = "nano-agent",
    feature = "operator-console",
    unix
))]
fn project_console_accessory_health(
    mut health: ConsoleSubsystemHealth,
    accessory: NanoAccessoryRuntimeHealth,
) -> ConsoleSubsystemHealth {
    health.head = Some(console_health_from_accessory(accessory.head));
    health.eyes = Some(console_health_from_accessory(
        match (accessory.eyes, accessory.rgb_expression) {
            (NanoAccessoryComponentHealth::Faulted, _)
            | (_, NanoAccessoryComponentHealth::Faulted) => NanoAccessoryComponentHealth::Faulted,
            (NanoAccessoryComponentHealth::Degraded, _)
            | (_, NanoAccessoryComponentHealth::Degraded) => NanoAccessoryComponentHealth::Degraded,
            (NanoAccessoryComponentHealth::Ready, NanoAccessoryComponentHealth::Ready) => {
                NanoAccessoryComponentHealth::Ready
            }
        },
    ));
    health
}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    feature = "nano-agent",
    feature = "operator-console",
    unix
))]
struct ProductionConsoleSnapshotContext<'runtime> {
    terminal_checkpoint_pending: bool,
    accessory_health: &'runtime NanoAccessoryHealthObserver,
    sensor_health: LiveSensorStreamHealth,
    slam_telemetry: &'runtime LiveSlamTelemetry,
    map_state: AgentMapStateV1,
    snapshot_clock: &'runtime InstantHostClock,
    tick_timing: LiveControlTickTiming,
}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    feature = "nano-agent",
    feature = "operator-console",
    unix
))]
fn publish_production_console_snapshot(
    owner: &ProductionLiveMotionOwner,
    console: &mut LiveProductionConsoleRuntime,
    context: ProductionConsoleSnapshotContext<'_>,
) -> Result<(), LiveProductionConsoleProjectionError> {
    let ProductionConsoleSnapshotContext {
        terminal_checkpoint_pending,
        accessory_health,
        sensor_health,
        slam_telemetry,
        map_state,
        snapshot_clock,
        tick_timing,
    } = context;
    let observation = &mut console.observation;
    let revision = observation
        .next_snapshot_revision
        .take()
        .ok_or(LiveProductionConsoleProjectionError::SnapshotRevisionExhausted)?;
    observation.next_snapshot_revision = revision.checked_add(1);
    let revision = ConsoleSnapshotRevision::parse(revision)
        .map_err(|_| LiveProductionConsoleProjectionError::SnapshotRevisionExhausted)?;
    let adapter = console
        .adapter
        .as_ref()
        .expect("production console adapter exists until terminal shutdown");
    let mut snapshot = adapter.handle().latest_snapshot().as_ref().clone();
    snapshot.revision = revision;
    snapshot.runtime = Some(if terminal_checkpoint_pending {
        AgentRuntimeStateV1::ShuttingDown
    } else {
        owner.dispatcher().control_status(map_state).runtime()
    });
    snapshot.terminal =
        terminal_checkpoint_pending.then_some(ConsoleTerminalState::ControlEnding {
            reason: ConsoleTerminalReason::FinalizingWarmRestartCheckpoint,
            current_camera_localization: ConsoleCheckpointLocalizationEvidence::NotClaimed,
        });
    snapshot.map = match (
        map_state.map_epoch_id(),
        map_state.revision(),
        map_state.localization(),
    ) {
        (None, None, None) => None,
        (Some(map_epoch_id), Some(revision), Some(localization)) => {
            let map_epoch_id = NonZeroU64::new(map_epoch_id)
                .ok_or(LiveProductionConsoleProjectionError::MapEpochZero)?;
            let grid = observation
                .map
                .as_ref()
                .filter(|map| map.map_epoch_id == map_epoch_id && map.revision == revision)
                .and_then(|map| map.grid);
            Some(ConsoleMapSnapshot {
                map_epoch_id,
                revision,
                localization: match localization {
                    AgentLocalizationStateV1::Localized => ConsoleLocalization::Localized,
                    AgentLocalizationStateV1::Lost => ConsoleLocalization::Lost,
                    AgentLocalizationStateV1::Unavailable => ConsoleLocalization::Unavailable,
                },
                grid,
            })
        }
        _ => return Err(LiveProductionConsoleProjectionError::MapEpochZero),
    };
    snapshot.navigation = Some(build_production_console_navigation_snapshot(
        owner.coordinator(),
        observation,
        map_state.localization(),
        tick_timing,
    )?);
    snapshot.actual_authority = project_console_actual_authority(owner, adapter)?;
    snapshot.last_requested = adapter.handle().latest_requested_command();
    snapshot.last_requested_actuation = observation.last_requested_actuation;
    snapshot.last_applied = observation.last_applied;
    snapshot.stop_certainty = observation.stop_certainty;
    snapshot.rerun_diagnostics_url = Some(observation.rerun_diagnostics_url);
    snapshot.health = refresh_console_accessory_health(snapshot.health, accessory_health)
        .map_err(LiveProductionConsoleProjectionError::AccessoryHealth)?;
    if matches!(snapshot.runtime, Some(AgentRuntimeStateV1::Faulted)) {
        snapshot.health.stm32 = Some(ConsoleHealth::Faulted);
    }
    // Sample only after every navigation/actuation telemetry field and
    // correlated receipt has been observed. Arbitration and safety are
    // separately documented live overlays, not timestamped by this value.
    let telemetry_observed_at = snapshot_clock
        .checked_now()
        .map_err(LiveProductionConsoleProjectionError::HostClock)?;
    snapshot.health.oak = Some(console_oak_stream_health(
        sensor_health,
        telemetry_observed_at,
    ));
    let (slam, slam_health) = project_live_slam_console(slam_telemetry, telemetry_observed_at)
        .map_err(LiveProductionConsoleProjectionError::SlamTelemetry)?;
    snapshot.slam = Some(slam);
    snapshot.health.slam = Some(slam_health);
    snapshot.telemetry_observed_at_host_monotonic_ns =
        Some(ConsoleHostTimestampNs::from_host(telemetry_observed_at));
    adapter
        .handle()
        .publish_snapshot(snapshot)
        .map_err(LiveProductionConsoleProjectionError::Snapshot)
}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    feature = "nano-agent",
    feature = "operator-console",
    unix
))]
fn publish_production_console_grid(
    production: &mut LiveProductionMotionRuntime,
    snapshot: &TimedOccupancySnapshot,
) -> Result<(), LiveProductionConsoleProjectionError> {
    let binding = production
        .owner()
        .coordinator()
        .current_map_binding()
        .ok_or(LiveProductionConsoleProjectionError::MapEpochZero)?;
    let map_epoch_id = NonZeroU64::new(binding.map_epoch_id().as_u64())
        .ok_or(LiveProductionConsoleProjectionError::MapEpochZero)?;
    let revision = snapshot.snapshot().revision();
    let localized = production.localized;
    let console = production.console_mut();
    if console
        .observation
        .map
        .as_ref()
        .is_some_and(|map| map.map_epoch_id == map_epoch_id && map.revision >= revision)
    {
        return Ok(());
    }
    let grid = ConsoleOccupancyGrid::from_snapshot(binding, snapshot.snapshot())
        .map_err(LiveProductionConsoleProjectionError::Grid)?;
    let metadata = grid.metadata;
    let adapter = console
        .adapter
        .as_ref()
        .expect("production console adapter exists until terminal shutdown");
    if !adapter.handle().publish_grid(grid) {
        return Err(
            LiveProductionConsoleProjectionError::GridPublicationRejected {
                map_epoch_id: map_epoch_id.get(),
                revision,
            },
        );
    }
    console.observation.map = Some(ConsoleMapSnapshot {
        map_epoch_id,
        revision,
        localization: if localized {
            ConsoleLocalization::Localized
        } else {
            ConsoleLocalization::Lost
        },
        grid: Some(metadata),
    });
    Ok(())
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
fn publish_wheels_off_qualification_grid(
    runtime: &mut LiveWheelsOffQualificationMotionRuntime,
    snapshot: &TimedOccupancySnapshot,
) -> Result<(), LiveNavigationWorkerError> {
    let binding = runtime.coordinator.current_map_binding().ok_or(
        LiveNavigationWorkerError::WheelsOffQualificationProjection {
            source: LiveProductionConsoleProjectionError::MapEpochZero,
        },
    )?;
    let map_epoch_id = NonZeroU64::new(binding.map_epoch_id().as_u64()).ok_or(
        LiveNavigationWorkerError::WheelsOffQualificationProjection {
            source: LiveProductionConsoleProjectionError::MapEpochZero,
        },
    )?;
    let revision = snapshot.snapshot().revision();
    if runtime
        .observation
        .map
        .as_ref()
        .is_some_and(|map| map.map_epoch_id == map_epoch_id && map.revision >= revision)
    {
        return Ok(());
    }
    let grid =
        ConsoleOccupancyGrid::from_snapshot(binding, snapshot.snapshot()).map_err(|source| {
            LiveNavigationWorkerError::WheelsOffQualificationProjection {
                source: LiveProductionConsoleProjectionError::Grid(source),
            }
        })?;
    let metadata = grid.metadata;
    runtime
        .telemetry
        .publish_grid(grid)
        .map_err(|source| LiveNavigationWorkerError::WheelsOffQualificationTelemetry { source })?;
    runtime.observation.map = Some(ConsoleMapSnapshot {
        map_epoch_id,
        revision,
        localization: if runtime.localized {
            ConsoleLocalization::Localized
        } else {
            ConsoleLocalization::Lost
        },
        grid: Some(metadata),
    });
    Ok(())
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
fn observe_wheels_off_qualification_controller(
    runtime: &mut LiveWheelsOffQualificationMotionRuntime,
) -> Result<(), LiveNavigationWorkerError> {
    runtime.observation.last_requested_actuation = None;
    runtime.observation.last_applied = runtime
        .controller()
        .last_applied()
        .map(ConsoleAppliedReceipt::from_verified)
        .transpose()
        .map_err(
            |source| LiveNavigationWorkerError::WheelsOffQualificationProjection {
                source: LiveProductionConsoleProjectionError::Receipt(source),
            },
        )?;
    runtime.observation.stop_certainty = match runtime.controller().controller_state() {
        kiko_slam::navigation::WheelsOffQualificationControllerState::StoppedWithExactReceipt => {
            runtime
                .controller()
                .last_stop()
                .map(ConsoleStopCertainty::from_verified_disarm)
        }
        kiko_slam::navigation::WheelsOffQualificationControllerState::Active => runtime
            .controller()
            .last_applied()
            .map(ConsoleStopCertainty::from_verified_applied),
        kiko_slam::navigation::WheelsOffQualificationControllerState::StopConfirmedWithoutRetainedReceipt
        | kiko_slam::navigation::WheelsOffQualificationControllerState::StopUncertain => {
            Some(ConsoleStopCertainty::uncertain())
        }
    };
    Ok(())
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
fn trigger_wheels_off_qualification_live_fault(
    runtime: &mut LiveWheelsOffQualificationMotionRuntime,
    actual_applied: robot_protocol::v2::TimerPwm,
    applied_observed_at: HostMonotonicTimestamp,
) -> Result<(), LiveNavigationWorkerError> {
    let trigger = match runtime
        .fault_injection
        .observe_controller_confirmed_applied_step(actual_applied, runtime.localized)
    {
        Ok(trigger) => trigger,
        Err(source) => {
            runtime
                .controller()
                .signal_internal_fail_closed(Some(applied_observed_at));
            return Err(
                LiveNavigationWorkerError::WheelsOffQualificationFaultTrigger {
                    selected: source.selected(),
                    source,
                },
            );
        }
    };
    let Some(trigger) = trigger else {
        return Ok(());
    };
    match trigger {
        kiko_slam::navigation::WheelsOffQualificationLiveFaultTrigger::StaleDepthOnFirstNonzeroCommand => {
            runtime
                .coordinator
                .inject_wheels_off_qualification_stale_depth();
            runtime.sensor_health.latch_stale(LiveSensorStream::Depth);
        }
        kiko_slam::navigation::WheelsOffQualificationLiveFaultTrigger::LocalizationLossOnFirstNonzeroCommand => {
            runtime
                .coordinator
                .inject_wheels_off_qualification_localization_loss();
            runtime.localized = false;
        }
    }
    runtime
        .controller()
        .signal_internal_fail_closed(Some(applied_observed_at));
    eprintln!(
        "wheels-off qualification synthetic fault triggered: selected={} triggered={}; basis=controller-confirmed nonzero applied step; process-lifetime terminal stop queued; this qualifier seam does not claim a physical sensor disconnect",
        trigger.declaration(),
        trigger.as_str(),
    );
    Ok(())
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
fn publish_wheels_off_qualification_snapshot(
    runtime: &mut LiveWheelsOffQualificationMotionRuntime,
    snapshot_clock: &InstantHostClock,
    tick_timing: LiveControlTickTiming,
    slam_telemetry: &LiveSlamTelemetry,
) -> Result<NanoAccessoryRuntimeHealth, LiveNavigationWorkerError> {
    observe_wheels_off_qualification_controller(runtime)?;
    let revision = runtime.observation.next_snapshot_revision.take().ok_or(
        LiveNavigationWorkerError::WheelsOffQualificationProjection {
            source: LiveProductionConsoleProjectionError::SnapshotRevisionExhausted,
        },
    )?;
    runtime.observation.next_snapshot_revision = revision.checked_add(1);
    let revision = ConsoleSnapshotRevision::parse(revision).map_err(|_| {
        LiveNavigationWorkerError::WheelsOffQualificationProjection {
            source: LiveProductionConsoleProjectionError::SnapshotRevisionExhausted,
        }
    })?;
    let map_binding = runtime.coordinator.current_map_binding();
    let localization = map_binding.map(|_| {
        if runtime.localized {
            AgentLocalizationStateV1::Localized
        } else {
            AgentLocalizationStateV1::Lost
        }
    });
    let map = match (map_binding, runtime.map_revision, localization) {
        (None, None, None) => None,
        (Some(binding), Some(revision), Some(localization)) => {
            let map_epoch_id = NonZeroU64::new(binding.map_epoch_id().as_u64()).ok_or(
                LiveNavigationWorkerError::WheelsOffQualificationProjection {
                    source: LiveProductionConsoleProjectionError::MapEpochZero,
                },
            )?;
            let grid = runtime
                .observation
                .map
                .as_ref()
                .filter(|map| map.map_epoch_id == map_epoch_id && map.revision == revision)
                .and_then(|map| map.grid);
            Some(ConsoleMapSnapshot {
                map_epoch_id,
                revision,
                localization: match localization {
                    AgentLocalizationStateV1::Localized => ConsoleLocalization::Localized,
                    AgentLocalizationStateV1::Lost => ConsoleLocalization::Lost,
                    AgentLocalizationStateV1::Unavailable => ConsoleLocalization::Unavailable,
                },
                grid,
            })
        }
        _ => {
            return Err(
                LiveNavigationWorkerError::WheelsOffQualificationProjection {
                    source: LiveProductionConsoleProjectionError::MapEpochZero,
                },
            );
        }
    };
    let mut snapshot = OperatorConsoleSnapshot::unknown(
        revision,
        ConsoleRuntimeAuthorityKind::WheelsOffQualification,
    );
    snapshot.runtime = Some(match (
        runtime.controller().state(),
        runtime.controller().controller_state(),
    ) {
        (
            kiko_slam::navigation::WheelsOffQualificationRuntimeState::Running,
            kiko_slam::navigation::WheelsOffQualificationControllerState::Active,
        ) => AgentRuntimeStateV1::Active {
            mode: kiko_slam::navigation::AgentOperatingModeV1::Commissioning,
        },
        (
            kiko_slam::navigation::WheelsOffQualificationRuntimeState::Running,
            kiko_slam::navigation::WheelsOffQualificationControllerState::StoppedWithExactReceipt,
        ) => AgentRuntimeStateV1::ReadyStopped,
        (
            kiko_slam::navigation::WheelsOffQualificationRuntimeState::Shutdown,
            _,
        ) => AgentRuntimeStateV1::ShuttingDown,
        _ => AgentRuntimeStateV1::Faulted,
    });
    snapshot.map = map;
    snapshot.navigation = Some(
        build_production_console_navigation_snapshot(
            &runtime.coordinator,
            &runtime.observation,
            localization,
            tick_timing,
        )
        .map_err(|source| LiveNavigationWorkerError::WheelsOffQualificationProjection { source })?,
    );
    snapshot.last_requested_actuation = runtime.observation.last_requested_actuation;
    snapshot.last_applied = runtime.observation.last_applied;
    snapshot.stop_certainty = runtime.observation.stop_certainty;
    snapshot.rerun_diagnostics_url = Some(runtime.observation.rerun_diagnostics_url);
    let accessory_health = runtime.accessory_health.snapshot().map_err(|source| {
        LiveNavigationWorkerError::WheelsOffQualificationProjection {
            source: LiveProductionConsoleProjectionError::AccessoryHealth(source),
        }
    })?;
    snapshot.health = project_console_accessory_health(runtime.initial_health, accessory_health);
    if matches!(snapshot.runtime, Some(AgentRuntimeStateV1::Faulted)) {
        snapshot.health.stm32 = Some(ConsoleHealth::Faulted);
    }
    let telemetry_observed_at = snapshot_clock
        .checked_now()
        .map_err(LiveNavigationWorkerError::HostClock)?;
    snapshot.health.oak = Some(console_oak_stream_health(
        runtime.sensor_health,
        telemetry_observed_at,
    ));
    let (slam, slam_health) = project_live_slam_console(slam_telemetry, telemetry_observed_at)
        .map_err(
            |source| LiveNavigationWorkerError::WheelsOffQualificationProjection {
                source: LiveProductionConsoleProjectionError::SlamTelemetry(source),
            },
        )?;
    snapshot.slam = Some(slam);
    snapshot.health.slam = Some(slam_health);
    snapshot.telemetry_observed_at_host_monotonic_ns =
        Some(ConsoleHostTimestampNs::from_host(telemetry_observed_at));
    runtime
        .telemetry
        .publish_observational_base(snapshot)
        .map_err(|source| LiveNavigationWorkerError::WheelsOffQualificationTelemetry { source })?;
    Ok(accessory_health)
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WheelsOffQualificationAttestationReadinessBlocker {
    FreshVisualDepthImuUnavailable,
    AccessoryNotReady,
    PublishedOccupancyRevisionUnavailable,
    CoordinatorMotionStartNotReady,
    NavigationVisualizationNotAccepted,
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
fn classify_wheels_off_qualification_attestation_readiness(
    fresh_visual_depth_imu: bool,
    accessory_ready: bool,
    published_occupancy_revision: bool,
    coordinator_motion_start_ready: bool,
    navigation_visualization_accepted: bool,
) -> Result<(), WheelsOffQualificationAttestationReadinessBlocker> {
    if !fresh_visual_depth_imu {
        return Err(
            WheelsOffQualificationAttestationReadinessBlocker::FreshVisualDepthImuUnavailable,
        );
    }
    if !accessory_ready {
        return Err(WheelsOffQualificationAttestationReadinessBlocker::AccessoryNotReady);
    }
    if !published_occupancy_revision {
        return Err(
            WheelsOffQualificationAttestationReadinessBlocker::PublishedOccupancyRevisionUnavailable,
        );
    }
    if !coordinator_motion_start_ready {
        return Err(
            WheelsOffQualificationAttestationReadinessBlocker::CoordinatorMotionStartNotReady,
        );
    }
    if !navigation_visualization_accepted {
        return Err(
            WheelsOffQualificationAttestationReadinessBlocker::NavigationVisualizationNotAccepted,
        );
    }
    Ok(())
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
fn wheels_off_qualification_accessory_is_ready(health: NanoAccessoryRuntimeHealth) -> bool {
    health.head == NanoAccessoryComponentHealth::Ready
        && health.eyes == NanoAccessoryComponentHealth::Ready
        && health.rgb_expression == NanoAccessoryComponentHealth::Ready
        && health.successful_rgb_expression_frames > 0
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
fn current_wheels_off_qualification_attestation_readiness(
    runtime: &LiveWheelsOffQualificationMotionRuntime,
    observed_at: HostMonotonicTimestamp,
    accessory_health: NanoAccessoryRuntimeHealth,
    navigation_viz: LiveNavigationVizPublishOutcome,
) -> Result<(), WheelsOffQualificationAttestationReadinessBlocker> {
    let published_occupancy_revision = runtime.map_revision.is_some_and(|revision| {
        runtime
            .observation
            .map
            .as_ref()
            .is_some_and(|map| map.revision == revision && map.grid.is_some())
    });
    classify_wheels_off_qualification_attestation_readiness(
        runtime.sensor_health.console_health(observed_at) == ConsoleHealth::Ready,
        wheels_off_qualification_accessory_is_ready(accessory_health),
        published_occupancy_revision,
        runtime
            .coordinator
            .motion_start_readiness_at(observed_at)
            .is_ok(),
        navigation_viz == LiveNavigationVizPublishOutcome::AcceptedByBoundedQueue,
    )
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
fn advance_wheels_off_qualification_motion_attestation_after_read_only_tick(
    runtime: &mut LiveWheelsOffQualificationMotionRuntime,
    snapshot_clock: &InstantHostClock,
    accessory_health: NanoAccessoryRuntimeHealth,
    navigation_viz: LiveNavigationVizPublishOutcome,
) -> Result<(), LiveNavigationWorkerError> {
    if runtime.controller().motion_authority_state()
        == kiko_slam::navigation::WheelsOffQualificationMotionAuthorityState::Enabled
    {
        return Ok(());
    }
    let process_running = Arc::clone(&runtime.process_running);
    if !process_running.load(Ordering::SeqCst) {
        runtime
            .attestation_gate_mut()
            .close_without_enablement(FreshAttendedMotionAttestationClosure::ProcessNotRunning)
            .map_err(|source| {
                LiveNavigationWorkerError::WheelsOffQualificationAttestationCleanup { source }
            })?;
        return Ok(());
    }
    if runtime
        .controller()
        .console_snapshot()
        .software_safety_stop_latched
    {
        let already_closed = runtime.attestation_gate_mut().is_closed();
        runtime
            .attestation_gate_mut()
            .close_without_enablement(
                FreshAttendedMotionAttestationClosure::SoftwareSafetyStopLatched,
            )
            .map_err(|source| {
                LiveNavigationWorkerError::WheelsOffQualificationAttestationCleanup { source }
            })?;
        if !already_closed {
            eprintln!(
                "wheels-off qualification motion-attestation gate closed without enablement because the process-lifetime software safety stop is latched"
            );
        }
        return Ok(());
    }
    let prompt_observed_at = snapshot_clock
        .checked_now()
        .map_err(LiveNavigationWorkerError::HostClock)?;
    if let Err(blocker) = current_wheels_off_qualification_attestation_readiness(
        runtime,
        prompt_observed_at,
        accessory_health,
        navigation_viz,
    ) {
        if runtime.attestation_gate_mut().has_started_prompt() {
            runtime
                .attestation_gate_mut()
                .close_without_enablement(FreshAttendedMotionAttestationClosure::ReadinessLost(
                    blocker,
                ))
                .map_err(|source| {
                    LiveNavigationWorkerError::WheelsOffQualificationAttestationCleanup { source }
                })?;
            eprintln!(
                "wheels-off qualification motion-attestation gate closed without enablement because integrated readiness was lost: {blocker:?}"
            );
        }
        return Ok(());
    }
    let poll = runtime
        .attestation_gate_mut()
        .advance_after_read_only_runtime_tick(process_running.as_ref())
        .map_err(
            |source| LiveNavigationWorkerError::WheelsOffQualificationAttestation { source },
        )?;
    let FreshAttendedMotionAttestationWorkerPoll::Ready(attestation) = poll else {
        return Ok(());
    };
    if !process_running.load(Ordering::SeqCst) {
        runtime
            .attestation_gate_mut()
            .close_without_enablement(FreshAttendedMotionAttestationClosure::ProcessNotRunning)
            .map_err(|source| {
                LiveNavigationWorkerError::WheelsOffQualificationAttestationCleanup { source }
            })?;
        return Ok(());
    }
    let final_accessory_health = runtime.accessory_health.snapshot().map_err(|source| {
        LiveNavigationWorkerError::WheelsOffQualificationProjection {
            source: LiveProductionConsoleProjectionError::AccessoryHealth(source),
        }
    })?;
    let enable_observed_at = snapshot_clock
        .checked_now()
        .map_err(LiveNavigationWorkerError::HostClock)?;
    if let Err(blocker) = current_wheels_off_qualification_attestation_readiness(
        runtime,
        enable_observed_at,
        final_accessory_health,
        navigation_viz,
    ) {
        runtime
            .attestation_gate_mut()
            .close_without_enablement(FreshAttendedMotionAttestationClosure::ReadinessLost(
                blocker,
            ))
            .map_err(|source| {
                LiveNavigationWorkerError::WheelsOffQualificationAttestationCleanup { source }
            })?;
        return Ok(());
    }
    if !process_running.load(Ordering::SeqCst) {
        runtime
            .attestation_gate_mut()
            .close_without_enablement(FreshAttendedMotionAttestationClosure::ProcessNotRunning)
            .map_err(|source| {
                LiveNavigationWorkerError::WheelsOffQualificationAttestationCleanup { source }
            })?;
        return Ok(());
    }
    runtime
        .controller_mut()
        .enable_motion_authority(attestation)
        .map_err(|source| {
            LiveNavigationWorkerError::WheelsOffQualificationMotionAuthorityEnable { source }
        })?;
    eprintln!(
        "wheels-off qualification manual motion boundary enabled exactly once from the fresh attended attestation"
    );
    Ok(())
}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "agent-runtime",
    feature = "nano-agent",
    feature = "operator-console",
    unix
))]
#[allow(clippy::too_many_arguments)]
fn run_production_console_control_period(
    production: &mut LiveProductionMotionRuntime,
    production_state: &mut Option<NanoProductionStateOwners>,
    pending_warm_checkpoint: &mut Option<PendingNanoWarmCheckpoint>,
    checkpoint_bridge: &NanoDatasetCheckpointWorkerBridge,
    running: &AtomicBool,
    slam_telemetry: &LiveSlamTelemetry,
    map_state: AgentMapStateV1,
    tick: HostMonotonicTimestamp,
    snapshot_clock: &InstantHostClock,
    tick_sequence: i64,
    tick_timing: LiveControlTickTiming,
    navigation_viz_tx: &mut Option<DropSender<LiveNavigationVizMsg>>,
) -> Result<(), LiveNavigationWorkerError> {
    let LiveProductionMotionRuntime {
        owner,
        console,
        terminal_response_timeout,
        accessory_health,
        sensor_health,
        map_revision: _,
        localized: _,
        terminal_checkpoint_pending,
        socket_task: _,
    } = production;
    let owner = owner
        .as_mut()
        .expect("production owner exists until terminal shutdown");
    let console = console
        .as_mut()
        .expect("production console exists until terminal shutdown");
    if let Some(evidence) = console
        .frontend
        .as_mut()
        .and_then(NanoOperatorConsoleFrontend::poll_unexpected_exit)
    {
        return Err(LiveNavigationWorkerError::ProductionConsoleFrontendExited {
            evidence: Box::new(evidence),
        });
    }
    let adapter = console
        .adapter
        .as_mut()
        .expect("production console adapter exists until terminal shutdown");

    let ingress = adapter.drain_one_before_owner(owner).map_err(|source| {
        LiveNavigationWorkerError::ProductionConsoleIngress {
            source: Box::new(source),
        }
    })?;
    if ingress == OperatorConsoleIngressDisposition::SoftwareEmergencyStopApplied {
        let event = owner.take_last_physical_state().ok_or(
            LiveNavigationWorkerError::ProductionConsoleAdapter {
                source: OperatorConsoleRuntimeAdapterError::PhysicalEvidenceRequired,
            },
        )?;
        let downstream_request_id = adapter
            .correlated_downstream_request_id(&event)
            .map_err(|source| LiveNavigationWorkerError::ProductionConsoleAdapter { source })?;
        observe_production_console_physical_state(
            &mut console.observation,
            downstream_request_id,
            &event,
        )
        .map_err(|source| LiveNavigationWorkerError::ProductionConsoleProjection { source })?;
        publish_live_physical_state_viz(
            navigation_viz_tx,
            owner.coordinator(),
            tick_sequence,
            tick_timing,
            &event,
        )
        .map_err(|source| LiveNavigationWorkerError::PhysicalStateVisualization { source })?;
        drop(event);
        adapter
            .complete_software_emergency_stop()
            .map_err(|source| LiveNavigationWorkerError::ProductionConsoleAdapter { source })?;
        publish_production_console_snapshot(
            owner,
            console,
            ProductionConsoleSnapshotContext {
                terminal_checkpoint_pending: *terminal_checkpoint_pending,
                accessory_health,
                sensor_health: *sensor_health,
                slam_telemetry,
                map_state,
                snapshot_clock,
                tick_timing,
            },
        )
        .map_err(|source| LiveNavigationWorkerError::ProductionConsoleProjection { source })?;
        return Ok(());
    }

    let command_outcome = match owner.process_one_with_motion_start_readiness(map_state) {
        Ok(outcome) => outcome,
        Err(source) => {
            let mut failures = vec![LiveNavigationWorkerError::ProductionOwner {
                source: Box::new(source),
            }];
            if let Some(event) = owner.take_last_physical_state()
                && let Err(source) = publish_live_physical_state_viz(
                    navigation_viz_tx,
                    owner.coordinator(),
                    tick_sequence,
                    tick_timing,
                    &event,
                )
            {
                failures.push(LiveNavigationWorkerError::PhysicalStateVisualization { source });
            }
            if let Err(source) = adapter.fail_processed_owner_operation(owner) {
                failures.push(LiveNavigationWorkerError::ProductionConsoleAdapter { source });
            }
            return Err(if failures.len() == 1 {
                failures
                    .pop()
                    .expect("one retained production-owner failure")
            } else {
                LiveNavigationWorkerError::Multiple { failures }
            });
        }
    };
    let periodic_tick_deferred = command_outcome.defers_periodic_motion_tick();
    let command_physical_state = owner.take_last_physical_state();
    let command_applied = command_physical_state.is_some();
    if let Some(event) = command_physical_state.as_ref() {
        let downstream_request_id = adapter
            .correlated_downstream_request_id(event)
            .map_err(|source| LiveNavigationWorkerError::ProductionConsoleAdapter { source })?;
        observe_production_console_physical_state(
            &mut console.observation,
            downstream_request_id,
            event,
        )
        .map_err(|source| LiveNavigationWorkerError::ProductionConsoleProjection { source })?;
        publish_live_physical_state_viz(
            navigation_viz_tx,
            owner.coordinator(),
            tick_sequence,
            tick_timing,
            event,
        )
        .map_err(|source| LiveNavigationWorkerError::PhysicalStateVisualization { source })?;
    }
    let adapter_disposition = adapter
        .complete_processed_owner_outcome(owner, &command_outcome, command_physical_state)
        .map_err(|source| LiveNavigationWorkerError::ProductionConsoleAdapter { source })?;

    let shutdown_requested = match command_outcome {
        LiveMotionOwnerOutcome::SaveMapRequested { claimed } => {
            let console_response_pending = matches!(
                adapter_disposition,
                OperatorConsoleProcessDisposition::SaveMapPersistenceRequired { .. }
            );
            if !console_response_pending
                && !matches!(
                    adapter_disposition,
                    OperatorConsoleProcessDisposition::UnrelatedRuntimeOutcome
                )
            {
                return Err(LiveNavigationWorkerError::ProductionConsoleAdapter {
                    source: OperatorConsoleRuntimeAdapterError::OwnerOutcomeMismatch,
                });
            }
            match production_state.as_mut() {
                Some(state) => {
                    if state.map_persistence.requires_quiescent_warm_checkpoint() {
                        if pending_warm_checkpoint.is_some() {
                            return Err(LiveNavigationWorkerError::ProductionConsoleAdapter {
                                source:
                                    OperatorConsoleRuntimeAdapterError::SaveMapCompletionPending,
                            });
                        }
                        let terminal_response_deadline = match claimed
                            .terminal_response_deadline()
                            .or_else(|| Instant::now().checked_add(*terminal_response_timeout))
                        {
                            Some(deadline) => deadline,
                            None => {
                                let response = claimed
                                    .reject(AgentControlRejectionCodeV1::InternalFault, false)
                                    .err();
                                return Err(
                                    LiveNavigationWorkerError::WarmCheckpointDeadlineOverflow {
                                        response,
                                    },
                                );
                            }
                        };
                        let Some(checkpoint_deadline) = terminal_response_deadline
                            .checked_sub(NANO_TERMINAL_RESPONSE_COMPLETION_RESERVE)
                        else {
                            let response = claimed
                                .reject(AgentControlRejectionCodeV1::InternalFault, false)
                                .err();
                            return Err(
                                LiveNavigationWorkerError::WarmCheckpointDeadlineOverflow {
                                    response,
                                },
                            );
                        };
                        if checkpoint_deadline <= Instant::now()
                            || checkpoint_bridge
                                .checkpoint_deadline
                                .set(checkpoint_deadline)
                                .is_err()
                        {
                            let response = claimed
                                .reject(AgentControlRejectionCodeV1::ShutdownInProgress, false)
                                .err();
                            return Err(
                                LiveNavigationWorkerError::WarmCheckpointDeadlineUnavailable {
                                    response,
                                },
                            );
                        }
                        *pending_warm_checkpoint = Some(PendingNanoWarmCheckpoint {
                            claimed,
                            console_response_pending,
                        });
                        *terminal_checkpoint_pending = true;
                        checkpoint_bridge.requested.store(true, Ordering::Release);
                    } else {
                        state
                            .map_persistence
                            .respond_to_claimed_save_map_with_quota(claimed, &mut state.quota)
                            .map_err(|source| LiveNavigationWorkerError::ProductionSaveMap {
                                source: Box::new(source),
                            })?;
                        if console_response_pending {
                            adapter.complete_save_map_response().map_err(|source| {
                                LiveNavigationWorkerError::ProductionConsoleAdapter { source }
                            })?;
                        }
                    }
                }
                None => {
                    let response = claimed
                        .reject(AgentControlRejectionCodeV1::PersistenceFailed, false)
                        .err();
                    if response.is_none() {
                        adapter.complete_save_map_response().map_err(|source| {
                            LiveNavigationWorkerError::ProductionConsoleAdapter { source }
                        })?;
                    }
                    return Err(
                        LiveNavigationWorkerError::ProductionMapPersistenceUnavailable { response },
                    );
                }
            }
            false
        }
        LiveMotionOwnerOutcome::ShutdownRequested => true,
        LiveMotionOwnerOutcome::Idle
        | LiveMotionOwnerOutcome::ClientUnavailableBeforeClaim
        | LiveMotionOwnerOutcome::StatusReplied(_)
        | LiveMotionOwnerOutcome::Rejected { .. }
        | LiveMotionOwnerOutcome::Completed(_)
        | LiveMotionOwnerOutcome::PeriodicManualApplied
        | LiveMotionOwnerOutcome::PeriodicManualStopped
        | LiveMotionOwnerOutcome::AutonomousAccepted { .. }
        | LiveMotionOwnerOutcome::PeriodicAutonomousApplied { .. }
        | LiveMotionOwnerOutcome::PeriodicAutonomousStopped { .. }
        | LiveMotionOwnerOutcome::AutonomousCompleted { .. } => {
            if matches!(
                adapter_disposition,
                OperatorConsoleProcessDisposition::SaveMapPersistenceRequired { .. }
            ) {
                return Err(LiveNavigationWorkerError::ProductionConsoleAdapter {
                    source: OperatorConsoleRuntimeAdapterError::OwnerOutcomeMismatch,
                });
            }
            false
        }
    };
    if shutdown_requested {
        running.store(false, Ordering::SeqCst);
    }

    let mut published_physical_state = command_applied;
    if production_period_requires_motion_tick(
        shutdown_requested || *terminal_checkpoint_pending,
        periodic_tick_deferred,
        command_applied,
    ) {
        let periodic_outcome = match owner.tick_motion() {
            Ok(outcome) => outcome,
            Err(source) => {
                let owner_failure = LiveNavigationWorkerError::ProductionOwner {
                    source: Box::new(source),
                };
                if let Some(event) = owner.take_last_physical_state()
                    && let Err(source) = publish_live_physical_state_viz(
                        navigation_viz_tx,
                        owner.coordinator(),
                        tick_sequence,
                        tick_timing,
                        &event,
                    )
                {
                    return Err(LiveNavigationWorkerError::Multiple {
                        failures: vec![
                            owner_failure,
                            LiveNavigationWorkerError::PhysicalStateVisualization { source },
                        ],
                    });
                }
                return Err(owner_failure);
            }
        };
        let periodic_physical_state = owner.take_last_physical_state();
        published_physical_state = periodic_physical_state.is_some();
        if let Some(event) = periodic_physical_state.as_ref() {
            let downstream_request_id = adapter
                .correlated_downstream_request_id(event)
                .map_err(|source| LiveNavigationWorkerError::ProductionConsoleAdapter { source })?;
            observe_production_console_physical_state(
                &mut console.observation,
                downstream_request_id,
                event,
            )
            .map_err(|source| LiveNavigationWorkerError::ProductionConsoleProjection { source })?;
            publish_live_physical_state_viz(
                navigation_viz_tx,
                owner.coordinator(),
                tick_sequence,
                tick_timing,
                event,
            )
            .map_err(|source| LiveNavigationWorkerError::PhysicalStateVisualization { source })?;
        }
        adapter
            .complete_periodic_owner_outcome(&periodic_outcome, periodic_physical_state)
            .map_err(|source| LiveNavigationWorkerError::ProductionConsoleAdapter { source })?;
    }
    if !published_physical_state {
        publish_live_navigation_viz_message(
            navigation_viz_tx,
            LiveNavigationVizMsg::control_tick_timing_only(
                tick_sequence,
                tick.as_nanos(),
                tick_timing,
            ),
        )
        .map_err(|source| LiveNavigationWorkerError::PhysicalStateVisualization { source })?;
    }
    publish_production_console_snapshot(
        owner,
        console,
        ProductionConsoleSnapshotContext {
            terminal_checkpoint_pending: *terminal_checkpoint_pending,
            accessory_health,
            sensor_health: *sensor_health,
            slam_telemetry,
            map_state,
            snapshot_clock,
            tick_timing,
        },
    )
    .map_err(|source| LiveNavigationWorkerError::ProductionConsoleProjection { source })?;
    Ok(())
}

#[cfg(feature = "record")]
enum LiveNavigationWorkerInput {
    Tick(Instant),
    Visual(Result<VisualAdmission, crossbeam_channel::RecvError>),
    Depth(Result<DepthObservation, crossbeam_channel::RecvError>),
    Imu(Result<ImuReport, crossbeam_channel::RecvError>),
    Map(Result<TimedOccupancySnapshot, crossbeam_channel::RecvError>),
}

#[cfg(feature = "record")]
fn select_live_navigation_worker_input(
    tick: &crossbeam_channel::Receiver<Instant>,
    visual: &crossbeam_channel::Receiver<VisualAdmission>,
    depth: &crossbeam_channel::Receiver<DepthObservation>,
    imu: &crossbeam_channel::Receiver<ImuReport>,
    map: &crossbeam_channel::Receiver<TimedOccupancySnapshot>,
) -> LiveNavigationWorkerInput {
    crossbeam_channel::select_biased! {
        recv(tick) -> message => LiveNavigationWorkerInput::Tick(
            message.expect("crossbeam periodic tick receivers do not disconnect"),
        ),
        recv(visual) -> message => LiveNavigationWorkerInput::Visual(message),
        recv(depth) -> message => LiveNavigationWorkerInput::Depth(message),
        recv(imu) -> message => LiveNavigationWorkerInput::Imu(message),
        recv(map) -> message => LiveNavigationWorkerInput::Map(message),
    }
}

#[cfg(feature = "record")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct EntrySnapshotDrain {
    drained: usize,
    disconnected: bool,
}

#[cfg(feature = "record")]
fn drain_entry_snapshot<T, E>(
    receiver: &crossbeam_channel::Receiver<T>,
    mut admit: impl FnMut(T) -> Result<(), E>,
) -> Result<EntrySnapshotDrain, E> {
    let ready_on_entry = receiver.len();
    let mut drained = 0;
    let mut disconnected = false;
    for _ in 0..ready_on_entry {
        match receiver.try_recv() {
            Ok(value) => {
                admit(value)?;
                drained += 1;
            }
            Err(crossbeam_channel::TryRecvError::Empty) => break,
            Err(crossbeam_channel::TryRecvError::Disconnected) => {
                disconnected = true;
                break;
            }
        }
    }
    Ok(EntrySnapshotDrain {
        drained,
        disconnected,
    })
}

#[cfg(feature = "record")]
#[allow(clippy::too_many_arguments)]
fn run_live_navigation_worker(
    coordinator: ShadowNavigationCoordinator<NavigationIngressWriter<File>>,
    control_period: ControlPeriodNs,
    coordinator_clock_epoch: NavigationClockEpoch,
    clock_origin: Instant,
    #[cfg(feature = "actuation")] motion: LiveNavigationWorkerMotion,
    #[cfg(all(feature = "nano-agent", unix))] mut production_state: Option<
        NanoProductionStateOwners,
    >,
    #[cfg(all(feature = "nano-agent", unix))] checkpoint_bridge: Option<
        NanoDatasetCheckpointWorkerBridge,
    >,
    running: Arc<AtomicBool>,
    slam_telemetry: LiveSlamTelemetry,
    visual_rx: crossbeam_channel::Receiver<VisualAdmission>,
    depth_rx: DropReceiver<DepthObservation>,
    imu_rx: DropReceiver<ImuReport>,
    map_rx: DropReceiver<TimedOccupancySnapshot>,
    mut map_viz_tx: Option<DropSender<TimedOccupancySnapshot>>,
    mut navigation_viz_tx: Option<DropSender<LiveNavigationVizMsg>>,
) -> Result<LiveNavigationWorkerSuccess, LiveNavigationWorkerError> {
    #[cfg(all(feature = "nano-agent", unix))]
    let _exit_guard = match checkpoint_bridge.as_ref() {
        Some(bridge) => LiveThreadExitGuard::checkpoint_aware(
            Arc::clone(&running),
            Arc::clone(&bridge.requested),
        ),
        None => LiveThreadExitGuard::new(Arc::clone(&running)),
    };
    #[cfg(not(all(feature = "nano-agent", unix)))]
    let _exit_guard = LiveThreadExitGuard::new(Arc::clone(&running));
    #[cfg(not(all(feature = "actuation", feature = "agent-runtime", unix)))]
    let _ = coordinator_clock_epoch;
    #[cfg(feature = "actuation")]
    let mut runtime = match motion {
        LiveNavigationWorkerMotion::Compatibility(actuation_config) => {
            let physical_actuation = match actuation_config.as_ref() {
                Some(config) => {
                    let (driver, initial_zero) =
                        LiveMpcControlDriver::acquire(config, clock_origin).map_err(|source| {
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
                    Some(driver)
                }
                None => None,
            };
            LiveNavigationRuntime::Compatibility(Box::new(LiveCompatibilityNavigationRuntime {
                coordinator,
                physical_actuation,
            }))
        }
        #[cfg(all(feature = "agent-runtime", unix))]
        LiveNavigationWorkerMotion::Production(input) => {
            let mut input = *input;
            #[cfg(all(feature = "nano-agent", feature = "operator-console"))]
            let rerun_diagnostics_url = input.rerun_diagnostics_url();
            let (start_input, accessory_health) = input.take_for_owner();
            match start_production_motion_runtime(
                coordinator,
                coordinator_clock_epoch,
                clock_origin,
                start_input,
                accessory_health,
                #[cfg(all(feature = "nano-agent", feature = "operator-console"))]
                rerun_diagnostics_url,
                Arc::clone(&running),
            ) {
                Ok(production) => LiveNavigationRuntime::Production(Box::new(production)),
                Err(failure) => {
                    let (coordinator, source) = *failure;
                    let mut failures = vec![LiveNavigationWorkerError::ProductionStart {
                        source: Box::new(source),
                    }];
                    if let Err(source) = finalize_live_navigation_coordinator(coordinator) {
                        failures.push(source);
                    }
                    combine_live_navigation_failures(failures)?;
                    unreachable!("production startup retained at least one failure")
                }
            }
        }
        #[cfg(all(feature = "nano-attended-navigation-trial", unix))]
        LiveNavigationWorkerMotion::AttendedNavigationTrial(input) => {
            let mut input = *input;
            let (admission, accessory_health, rerun_diagnostics_url) = input.take_for_owner();
            let AttendedNavigationTrialMotionAdmission {
                startup,
                driver,
                initial_zero: _,
                head_gaze_lease_issuer: _,
            } = admission;
            let start_input = LiveAgentMotionStartInput {
                policy: startup.policy,
                authority: startup.authority,
                physical_driver: driver,
                admitted_actuation_config: None,
                coordinator_actuation_config: None,
                kind: LiveAgentAuthorityKind::AttendedNavigationTrial,
            };
            match start_production_motion_runtime(
                coordinator,
                coordinator_clock_epoch,
                clock_origin,
                start_input,
                accessory_health,
                rerun_diagnostics_url,
                Arc::clone(&running),
            ) {
                Ok(runtime) => LiveNavigationRuntime::Production(Box::new(runtime)),
                Err(failure) => {
                    let (coordinator, source) = *failure;
                    let mut failures = vec![LiveNavigationWorkerError::ProductionStart {
                        source: Box::new(source),
                    }];
                    if let Err(source) = finalize_live_navigation_coordinator(coordinator) {
                        failures.push(source);
                    }
                    combine_live_navigation_failures(failures)?;
                    unreachable!("attended navigation startup retained at least one failure")
                }
            }
        }
        #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
        LiveNavigationWorkerMotion::WheelsOffQualification(input) => {
            match start_wheels_off_qualification_motion_runtime(
                coordinator,
                *input,
                control_period.as_duration(),
                Arc::clone(&running),
            ) {
                Ok(runtime) => LiveNavigationRuntime::WheelsOffQualification(Box::new(runtime)),
                Err(failure) => {
                    let (coordinator, source) = *failure;
                    let mut failures =
                        vec![LiveNavigationWorkerError::WheelsOffQualificationStart {
                            source: Box::new(source),
                        }];
                    if let Err(source) = finalize_live_navigation_coordinator(coordinator) {
                        failures.push(source);
                    }
                    combine_live_navigation_failures(failures)?;
                    unreachable!("qualification startup retained at least one failure")
                }
            }
        }
    };
    #[cfg(not(feature = "actuation"))]
    let mut runtime =
        LiveNavigationRuntime::Compatibility(Box::new(LiveCompatibilityNavigationRuntime {
            coordinator,
        }));
    let mut tick_sequence = 0_i64;
    #[cfg(all(feature = "nano-agent", feature = "operator-console", unix))]
    let mut pending_warm_checkpoint = None;
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
        let mut maximum_tick_lateness_ns = 0_u64;
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
            match select_live_navigation_worker_input(
                &tick_rx,
                visual_receiver,
                depth_receiver,
                imu_receiver,
                map_receiver,
            ) {
                LiveNavigationWorkerInput::Visual(message) => match message {
                    Ok(admission) => {
                        let now = clock
                            .checked_now()
                            .map_err(LiveNavigationWorkerError::HostClock)?;
                        runtime.accept_visual(admission, now).map_err(|source| {
                            LiveNavigationWorkerError::VisualAdmission { source }
                        })?;
                    }
                    Err(_) => {
                        visual_open = false;
                        #[cfg(all(feature = "nano-agent", feature = "operator-console", unix))]
                        runtime.mark_sensor_closed(LiveSensorStream::Visual);
                    }
                },
                LiveNavigationWorkerInput::Depth(message) => match message {
                    Ok(observation) => {
                        let now = clock
                            .checked_now()
                            .map_err(LiveNavigationWorkerError::HostClock)?;
                        runtime.accept_depth(observation, now).map_err(|source| {
                            LiveNavigationWorkerError::DepthAdmission { source }
                        })?;
                    }
                    Err(_) => {
                        depth_open = false;
                        #[cfg(all(feature = "nano-agent", feature = "operator-console", unix))]
                        runtime.mark_sensor_closed(LiveSensorStream::Depth);
                    }
                },
                LiveNavigationWorkerInput::Imu(message) => match message {
                    Ok(report) => {
                        let now = clock
                            .checked_now()
                            .map_err(LiveNavigationWorkerError::HostClock)?;
                        runtime
                            .accept_imu(report, now)
                            .map_err(|source| LiveNavigationWorkerError::ImuAdmission { source })?;
                    }
                    Err(_) => {
                        imu_open = false;
                        #[cfg(all(feature = "nano-agent", feature = "operator-console", unix))]
                        runtime.mark_sensor_closed(LiveSensorStream::Imu);
                    }
                },
                LiveNavigationWorkerInput::Map(message) => match message {
                    Ok(snapshot) => {
                        // The inference producer routes each visual outcome before it can
                        // issue that attempt's dense-map command. Because visual and map
                        // traffic use distinct bounded queues, selection may nevertheless
                        // observe the derived map first. Admit every visual outcome already
                        // ready on entry before admitting this map. Snapshotting the ready
                        // count preserves that causal prefix without allowing a producer to
                        // extend this drain and starve the control tick. The journal remains
                        // the authority for the actual cross-channel admission order.
                        let drained = drain_entry_snapshot(&visual_rx, |admission| {
                            let now = clock
                                .checked_now()
                                .map_err(LiveNavigationWorkerError::HostClock)?;
                            runtime.accept_visual(admission, now).map_err(|source| {
                                LiveNavigationWorkerError::VisualAdmission { source }
                            })
                        })?;
                        debug_assert!(
                            drained.drained <= LIVE_NAVIGATION_VISUAL_QUEUE_CAPACITY,
                            "the production visual receiver has one fixed bounded capacity"
                        );
                        if drained.disconnected {
                            visual_open = false;
                            #[cfg(all(
                                feature = "nano-agent",
                                feature = "operator-console",
                                unix
                            ))]
                            runtime.mark_sensor_closed(LiveSensorStream::Visual);
                        }
                        let now = clock
                            .checked_now()
                            .map_err(LiveNavigationWorkerError::HostClock)?;
                        runtime.accept_global_map(now, &snapshot)?;
                        #[cfg(all(feature = "nano-agent", feature = "operator-console", unix))]
                        if let LiveNavigationRuntime::Production(production) = &mut runtime {
                            publish_production_console_grid(production, &snapshot).map_err(
                                |source| LiveNavigationWorkerError::ProductionConsoleProjection {
                                    source,
                                },
                            )?;
                        }
                        #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
                        if let LiveNavigationRuntime::WheelsOffQualification(qualification) =
                            &mut runtime
                        {
                            publish_wheels_off_qualification_grid(qualification, &snapshot)?;
                        }
                        #[cfg(all(feature = "nano-agent", unix))]
                        if let Some(state) = production_state.as_mut() {
                            let viz_snapshot = if map_viz_tx.is_some() {
                                match snapshot.try_duplicate() {
                                    Ok(snapshot) => Some(snapshot),
                                    Err(source) => {
                                        eprintln!(
                                            "non-authoritative occupancy visualization copy failed; disabling map visualization: {source}"
                                        );
                                        map_viz_tx = None;
                                        None
                                    }
                                }
                            } else {
                                None
                            };
                            let binding = runtime.current_map_binding().ok_or(
                                LiveNavigationWorkerError::MapPersistenceBindingUnavailable,
                            )?;
                            let (_, snapshot) = snapshot.into_parts();
                            state
                                .map_persistence
                                .retain_latest(binding, snapshot)
                                .map_err(|source| {
                                    LiveNavigationWorkerError::MapPersistenceRetention { source }
                                })?;
                            if let (Some(sender), Some(snapshot)) =
                                (map_viz_tx.as_ref(), viz_snapshot)
                                && matches!(sender.try_send(snapshot), SendOutcome::Disconnected)
                            {
                                map_viz_tx = None;
                            }
                        } else if let Some(sender) = map_viz_tx.as_ref()
                            && matches!(sender.try_send(snapshot), SendOutcome::Disconnected)
                        {
                            map_viz_tx = None;
                        }
                        #[cfg(not(all(feature = "nano-agent", unix)))]
                        if let Some(sender) = map_viz_tx.as_ref()
                            && matches!(sender.try_send(snapshot), SendOutcome::Disconnected)
                        {
                            map_viz_tx = None;
                        }
                    }
                    Err(_) => map_open = false,
                },
                LiveNavigationWorkerInput::Tick(scheduled_at) => {
                    if !running.load(Ordering::SeqCst) {
                        break;
                    }
                    let tick_timing = measure_live_control_tick_timing(
                        scheduled_at,
                        Instant::now(),
                        maximum_tick_lateness_ns,
                    )
                    .map_err(LiveNavigationWorkerError::TickTiming)?;
                    maximum_tick_lateness_ns = tick_timing.maximum_lateness_ns;
                    let tick = clock
                        .checked_now()
                        .map_err(LiveNavigationWorkerError::HostClock)?;
                    tick_sequence = tick_sequence
                        .checked_add(1)
                        .ok_or(LiveNavigationWorkerError::TickSequenceExhausted)?;
                    #[cfg(all(feature = "nano-agent", feature = "operator-console", unix))]
                    if pending_warm_checkpoint.is_some() {
                        // Capture has been asked to stop and all sensor/map
                        // channels remain authoritative until drained. No new
                        // control request or periodic motion tick may overtake
                        // the terminal checkpoint boundary.
                        continue;
                    }
                    match &mut runtime {
                        LiveNavigationRuntime::Compatibility(runtime) => {
                            #[cfg(feature = "actuation")]
                            let (outcome, applied_actuation) =
                                if let Some(driver) = runtime.physical_actuation.as_mut() {
                                    let applied = driver
                                        .tick_point_goal(&mut runtime.coordinator, tick, &mut clock)
                                        .map_err(|source| {
                                            LiveNavigationWorkerError::MpcControl { source }
                                        })?;
                                    let (outcome, receipt) = applied.into_parts();
                                    (
                                        outcome,
                                        Some(live_applied_actuation_viz(&receipt, Some(tick))),
                                    )
                                } else {
                                    let outcome =
                                        runtime.coordinator.tick(tick, &mut clock).map_err(
                                            |source| LiveNavigationWorkerError::Tick { source },
                                        )?;
                                    (outcome, None)
                                };
                            #[cfg(not(feature = "actuation"))]
                            let outcome = runtime
                                .coordinator
                                .tick(tick, &mut clock)
                                .map_err(|source| LiveNavigationWorkerError::Tick { source })?;
                            #[cfg(not(feature = "actuation"))]
                            let applied_actuation = None;
                            // Diagnostic copies happen only after the authoritative decision and never
                            // feed back into planning, MPC, journal admission, applied evidence, or shadow evidence.
                            if let Some(sender) = navigation_viz_tx.as_ref() {
                                let message = build_live_navigation_viz_message(
                                    &runtime.coordinator,
                                    tick,
                                    tick_sequence,
                                    &outcome,
                                    applied_actuation,
                                )
                                .with_control_tick_timing(tick_timing);
                                if matches!(sender.try_send(message), SendOutcome::Disconnected) {
                                    navigation_viz_tx = None;
                                }
                            }
                        }
                        #[cfg(all(feature = "actuation", feature = "agent-runtime", unix))]
                        LiveNavigationRuntime::Production(production) => {
                            #[cfg(all(feature = "nano-agent", feature = "operator-console", unix))]
                            {
                                let map_state = production.map_state().map_err(|source| {
                                    LiveNavigationWorkerError::ProductionMapState { source }
                                })?;
                                run_production_console_control_period(
                                    production,
                                    &mut production_state,
                                    &mut pending_warm_checkpoint,
                                    checkpoint_bridge.as_ref().expect(
                                        "production Nano navigation has one checkpoint bridge",
                                    ),
                                    running.as_ref(),
                                    &slam_telemetry,
                                    map_state,
                                    tick,
                                    &clock,
                                    tick_sequence,
                                    tick_timing,
                                    &mut navigation_viz_tx,
                                )?;
                            }
                            #[cfg(not(all(
                                feature = "nano-agent",
                                feature = "operator-console",
                                unix
                            )))]
                            {
                                let map_state = production.map_state().map_err(|source| {
                                    LiveNavigationWorkerError::ProductionMapState { source }
                                })?;
                                let command_outcome = match production
                                    .owner_mut()
                                    .process_one_with_motion_start_readiness(map_state)
                                {
                                    Ok(outcome) => outcome,
                                    Err(source) => {
                                        let owner_failure =
                                            LiveNavigationWorkerError::ProductionOwner {
                                                source: Box::new(source),
                                            };
                                        if let Some(event) =
                                            production.owner_mut().take_last_physical_state()
                                            && let Err(source) = publish_live_physical_state_viz(
                                                &mut navigation_viz_tx,
                                                production.owner().coordinator(),
                                                tick_sequence,
                                                tick_timing,
                                                &event,
                                            )
                                        {
                                            return Err(LiveNavigationWorkerError::Multiple {
                                                failures: vec![
                                                    owner_failure,
                                                    LiveNavigationWorkerError::PhysicalStateVisualization {
                                                        source,
                                                    },
                                                ],
                                            });
                                        }
                                        return Err(owner_failure);
                                    }
                                };
                                let periodic_tick_deferred =
                                    command_outcome.defers_periodic_motion_tick();
                                let shutdown_requested = match command_outcome {
                                    LiveMotionOwnerOutcome::SaveMapRequested { claimed } => {
                                        #[cfg(all(feature = "nano-agent", unix))]
                                        match production_state.as_mut() {
                                            Some(state) => {
                                                state
                                                .map_persistence
                                                .respond_to_claimed_save_map_with_quota(
                                                    claimed,
                                                    &mut state.quota,
                                                )
                                                .map_err(|source| {
                                                    LiveNavigationWorkerError::ProductionSaveMap {
                                                        source: Box::new(source),
                                                    }
                                                })?;
                                            }
                                            None => {
                                                let response = claimed
                                                .reject(
                                                    AgentControlRejectionCodeV1::PersistenceFailed,
                                                    false,
                                                )
                                                .err();
                                                return Err(
                                                LiveNavigationWorkerError::ProductionMapPersistenceUnavailable {
                                                    response,
                                                },
                                            );
                                            }
                                        }
                                        #[cfg(not(all(feature = "nano-agent", unix)))]
                                    claimed
                                        .reject(
                                            AgentControlRejectionCodeV1::PersistenceFailed,
                                            true,
                                        )
                                        .map_err(|source| {
                                            LiveNavigationWorkerError::ProductionSaveMapResponse {
                                                source,
                                            }
                                        })?;
                                        false
                                    }
                                    LiveMotionOwnerOutcome::ShutdownRequested => true,
                                    LiveMotionOwnerOutcome::Idle
                                    | LiveMotionOwnerOutcome::ClientUnavailableBeforeClaim
                                    | LiveMotionOwnerOutcome::StatusReplied(_)
                                    | LiveMotionOwnerOutcome::Rejected { .. }
                                    | LiveMotionOwnerOutcome::Completed(_)
                                    | LiveMotionOwnerOutcome::PeriodicManualApplied
                                    | LiveMotionOwnerOutcome::PeriodicManualStopped
                                    | LiveMotionOwnerOutcome::AutonomousAccepted { .. }
                                    | LiveMotionOwnerOutcome::PeriodicAutonomousApplied {
                                        ..
                                    }
                                    | LiveMotionOwnerOutcome::PeriodicAutonomousStopped {
                                        ..
                                    }
                                    | LiveMotionOwnerOutcome::AutonomousCompleted { .. } => false,
                                };
                                if shutdown_requested {
                                    running.store(false, Ordering::SeqCst);
                                }

                                // A command can itself perform a receipt-gated
                                // manual or autonomous tick. Transfer that evidence
                                // before deciding whether this period still needs a
                                // periodic tick; this prevents duplicate physical
                                // applications in one host control period.
                                let mut physical_state =
                                    production.owner_mut().take_last_physical_state();
                                if production_period_requires_motion_tick(
                                    shutdown_requested,
                                    periodic_tick_deferred,
                                    physical_state.is_some(),
                                ) {
                                    if let Err(source) = production.owner_mut().tick_motion() {
                                        let owner_failure =
                                            LiveNavigationWorkerError::ProductionOwner {
                                                source: Box::new(source),
                                            };
                                        if let Some(event) =
                                            production.owner_mut().take_last_physical_state()
                                            && let Err(source) = publish_live_physical_state_viz(
                                                &mut navigation_viz_tx,
                                                production.owner().coordinator(),
                                                tick_sequence,
                                                tick_timing,
                                                &event,
                                            )
                                        {
                                            return Err(LiveNavigationWorkerError::Multiple {
                                            failures: vec![
                                                owner_failure,
                                                LiveNavigationWorkerError::PhysicalStateVisualization {
                                                    source,
                                                },
                                            ],
                                        });
                                        }
                                        return Err(owner_failure);
                                    }
                                    physical_state =
                                        production.owner_mut().take_last_physical_state();
                                }
                                if let Some(event) = physical_state.as_ref() {
                                    publish_live_physical_state_viz(
                                        &mut navigation_viz_tx,
                                        production.owner().coordinator(),
                                        tick_sequence,
                                        tick_timing,
                                        event,
                                    )
                                    .map_err(|source| {
                                        LiveNavigationWorkerError::PhysicalStateVisualization {
                                            source,
                                        }
                                    })?;
                                } else {
                                    publish_live_navigation_viz_message(
                                        &mut navigation_viz_tx,
                                        LiveNavigationVizMsg::control_tick_timing_only(
                                            tick_sequence,
                                            tick.as_nanos(),
                                            tick_timing,
                                        ),
                                    )
                                    .map_err(|source| {
                                        LiveNavigationWorkerError::PhysicalStateVisualization {
                                            source,
                                        }
                                    })?;
                                }
                            }
                        }
                        #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
                        LiveNavigationRuntime::WheelsOffQualification(qualification) => {
                            if let Some(evidence) =
                                qualification.frontend_mut().poll_unexpected_exit()
                            {
                                return Err(
                                    LiveNavigationWorkerError::WheelsOffQualificationFrontendExited {
                                        evidence: Box::new(evidence),
                                    },
                                );
                            }
                            let controller_tick =
                                qualification.controller_mut().tick().map_err(|source| {
                                    LiveNavigationWorkerError::WheelsOffQualificationRuntime {
                                        source: Box::new(source),
                                    }
                                })?;
                            if let kiko_slam::navigation::WheelsOffQualificationRuntimeTick::CandidateStepApplied {
                                pending,
                            } = controller_tick
                            {
                                let actual_applied = pending.actual_applied();
                                let applied_observed_at = clock
                                    .checked_now()
                                    .map_err(LiveNavigationWorkerError::HostClock)?;
                                let ingress = pending
                                    .journal_event(
                                        coordinator_clock_epoch,
                                        applied_observed_at,
                                    )
                                    .map_err(|source| {
                                        LiveNavigationWorkerError::WheelsOffQualificationAppliedStepBoundary {
                                            source,
                                        }
                                    })?;
                                let record = qualification
                                    .coordinator
                                    .journal_mut()
                                    .append(kiko_slam::navigation::NavigationIngressEvent::QualificationAppliedStep(
                                        ingress,
                                    ))
                                    .map_err(|source| {
                                        LiveNavigationWorkerError::WheelsOffQualificationAppliedStepJournal {
                                            source,
                                        }
                                    })?;
                                let journaled = pending
                                    .bind_journal_record(ingress, record)
                                    .map_err(|source| {
                                        LiveNavigationWorkerError::WheelsOffQualificationAppliedStepCorrelation {
                                            source,
                                        }
                                    })?;
                                qualification
                                    .controller_mut()
                                    .record_journaled_applied_step(journaled)
                                    .map_err(|source| {
                                        LiveNavigationWorkerError::WheelsOffQualificationRuntime {
                                            source: Box::new(source),
                                        }
                                    })?;
                                trigger_wheels_off_qualification_live_fault(
                                    qualification,
                                    actual_applied,
                                    applied_observed_at,
                                )?;
                            }
                            let outcome = qualification
                                .coordinator
                                .tick(tick, &mut clock)
                                .map_err(|source| LiveNavigationWorkerError::Tick { source })?;
                            qualification.observation.successful_solver_duration_ns =
                                match outcome.decision().outcome() {
                                    SafetyDecisionOutcome::Controller(controller) => {
                                        let status = controller.solve_status();
                                        checked_monotonic_duration_ns(
                                            u128::from(status.started_at().as_nanos()),
                                            u128::from(status.observed_at().as_nanos()),
                                        )
                                    }
                                    SafetyDecisionOutcome::Stopped(_) => None,
                                };
                            let accessory_health = publish_wheels_off_qualification_snapshot(
                                qualification,
                                &clock,
                                tick_timing,
                                &slam_telemetry,
                            )?;
                            // The candidate controller is intentionally absent
                            // from this diagnostic builder: this is the MPC
                            // shadow decision, never a physical-output claim.
                            let navigation_viz = publish_live_navigation_viz_message(
                                &mut navigation_viz_tx,
                                build_live_navigation_viz_message(
                                    &qualification.coordinator,
                                    tick,
                                    tick_sequence,
                                    &outcome,
                                    None,
                                )
                                .with_control_tick_timing(tick_timing),
                            )
                            .map_err(|source| {
                                LiveNavigationWorkerError::PhysicalStateVisualization { source }
                            })?;
                            advance_wheels_off_qualification_motion_attestation_after_read_only_tick(
                                qualification,
                                &clock,
                                accessory_health,
                                navigation_viz,
                            )?;
                        }
                    }
                }
            }
        }

        Ok(())
    })();

    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    let operation_completed_without_error = operation_result.is_ok();
    let mut failures = Vec::new();
    if let Err(source) = operation_result {
        failures.push(source);
    }

    #[cfg(all(feature = "nano-agent", unix))]
    let mut prefinalized_descriptor = None;
    #[cfg(not(all(feature = "nano-agent", unix)))]
    let prefinalized_descriptor: Option<NavigationIngressSidecarDescriptor> = None;
    let coordinator = match runtime {
        LiveNavigationRuntime::Compatibility(runtime) => {
            let LiveCompatibilityNavigationRuntime {
                coordinator,
                #[cfg(feature = "actuation")]
                mut physical_actuation,
            } = *runtime;
            #[cfg(feature = "actuation")]
            if let Some(session) = physical_actuation.as_mut()
                && !session.is_consumed()
            {
                match session.disarm() {
                    Ok(receipt) => {
                        eprintln!(
                            "physical actuation disarmed: boot_id={} request_id={} acknowledged_at_host_ns={}",
                            receipt.observed_boot_id().get(),
                            receipt.request_id().get(),
                            receipt.acknowledged_at().nanos_since_clock_start(),
                        );
                    }
                    Err(source) => failures.push(LiveNavigationWorkerError::Actuation {
                        phase: "shutdown disarm",
                        source,
                    }),
                }
            }
            Some(coordinator)
        }
        #[cfg(all(feature = "actuation", feature = "agent-runtime", unix))]
        LiveNavigationRuntime::Production(mut production) => {
            #[cfg(all(feature = "nano-agent", feature = "operator-console", unix))]
            let mut console = production.take_console();
            let (owner, socket_task) = production.take_terminal_parts();
            #[cfg(not(all(feature = "nano-agent", unix)))]
            socket_task.request_shutdown();
            let terminal_clock = InstantHostClock::new(clock_origin);
            let terminal_requested_at = match terminal_clock.checked_now() {
                Ok(timestamp) => Some(timestamp),
                Err(source) => {
                    failures.push(LiveNavigationWorkerError::HostClock(source));
                    None
                }
            };
            let terminal = owner.shutdown();
            #[cfg(all(feature = "nano-agent", feature = "operator-console", unix))]
            let terminal_shutdown_evidence = terminal.shutdown_evidence();
            let (coordinator, lifecycle_cleanup, controller_stop, _last_physical_state) =
                terminal.into_parts();
            tick_sequence = match tick_sequence.checked_add(1) {
                Some(sequence) => sequence,
                None => {
                    failures.push(LiveNavigationWorkerError::TickSequenceExhausted);
                    i64::MAX
                }
            };
            let terminal_message = match &controller_stop {
                LiveMotionTerminalStop::Confirmed(receipt) => {
                    build_live_terminal_controller_stop_viz_message(
                        &coordinator,
                        tick_sequence,
                        u64::try_from(receipt.acknowledged_at().nanos_since_clock_start()).ok(),
                        true,
                        "terminal_controller_disarm_confirmed",
                        "controller_session_disarm".to_owned(),
                        format!(
                            "exact disarm receipt: boot_id={} request_id={} output_state={:?} controller_faults={:?}",
                            receipt.observed_boot_id().get(),
                            receipt.request_id().get(),
                            receipt.output_state(),
                            receipt.controller_faults(),
                        ),
                    )
                }
                LiveMotionTerminalStop::DisarmFailedStopConfirmed(source) => {
                    let AgentLiveActuationDisposition::LatchFault(fault) =
                        classify_live_actuation_error(source);
                    debug_assert_eq!(
                        fault.controller_stop(),
                        AgentControllerStopKnowledge::Confirmed
                    );
                    build_live_terminal_controller_stop_viz_message(
                        &coordinator,
                        tick_sequence,
                        terminal_requested_at.map(HostMonotonicTimestamp::as_nanos),
                        true,
                        "terminal_controller_disarm_error_stop_confirmed",
                        format!("terminal_disarm_error:{:?}", fault.kind()),
                        format!(
                            "terminal disarm failed after the recorded shutdown-request boundary: {source}; classified_kind={:?} controller_stop={:?}; no receipt is fabricated",
                            fault.kind(),
                            fault.controller_stop(),
                        ),
                    )
                }
                LiveMotionTerminalStop::Uncertain(source) => {
                    let AgentLiveActuationDisposition::LatchFault(fault) =
                        classify_live_actuation_error(source);
                    debug_assert_eq!(
                        fault.controller_stop(),
                        AgentControllerStopKnowledge::Uncertain
                    );
                    build_live_terminal_controller_stop_viz_message(
                        &coordinator,
                        tick_sequence,
                        terminal_requested_at.map(HostMonotonicTimestamp::as_nanos),
                        false,
                        "terminal_controller_disarm_stop_uncertain",
                        format!("terminal_disarm_error:{:?}", fault.kind()),
                        format!(
                            "terminal disarm failed after the recorded shutdown-request boundary: {source}; classified_kind={:?} controller_stop={:?}; no receipt is fabricated",
                            fault.kind(),
                            fault.controller_stop(),
                        ),
                    )
                }
            };
            if let Err(source) =
                publish_live_navigation_viz_message(&mut navigation_viz_tx, terminal_message)
            {
                failures.push(LiveNavigationWorkerError::PhysicalStateVisualization { source });
            }
            if let Some(source) = lifecycle_cleanup {
                failures.push(LiveNavigationWorkerError::ProductionLifecycleCleanup {
                    source: Box::new(source),
                });
            }
            match controller_stop {
                LiveMotionTerminalStop::Confirmed(receipt) => {
                    eprintln!(
                        "live-agent physical owner disarmed: boot_id={} request_id={} acknowledged_at_host_ns={}",
                        receipt.observed_boot_id().get(),
                        receipt.request_id().get(),
                        receipt.acknowledged_at().nanos_since_clock_start(),
                    );
                }
                LiveMotionTerminalStop::DisarmFailedStopConfirmed(source)
                | LiveMotionTerminalStop::Uncertain(source) => {
                    failures.push(LiveNavigationWorkerError::ProductionControllerStop { source });
                }
            }
            #[cfg(all(feature = "nano-agent", unix))]
            {
                let (descriptor, finalized_map_identity) =
                    match finalize_live_navigation_coordinator(coordinator) {
                        Ok(finalized) => (
                            Some(finalized.descriptor),
                            finalized.final_map_identity.map(|identity| {
                                NanoFinalizedJournalMapIdentity::new(
                                    identity.map_epoch_id,
                                    identity.revision,
                                )
                            }),
                        ),
                        Err(source) => {
                            failures.push(source);
                            (None, None)
                        }
                    };
                if let Some(pending) = pending_warm_checkpoint.take() {
                    let bridge = checkpoint_bridge
                        .as_ref()
                        .expect("production Nano checkpoint request retains its bridge");
                    let navigation_publishable = failures.is_empty() && descriptor.is_some();
                    if bridge
                        .request
                        .send(NanoDatasetCheckpointRequest {
                            descriptor,
                            navigation_publishable,
                        })
                        .is_err()
                    {
                        let response = if pending.console_response_pending {
                            pending
                                .claimed
                                .reject(AgentControlRejectionCodeV1::PersistenceFailed, false)
                        } else {
                            pending.claimed.reject_after_wire_delivery(
                                AgentControlRejectionCodeV1::PersistenceFailed,
                                false,
                            )
                        };
                        failures.push(
                            LiveNavigationWorkerError::WarmCheckpointRequestChannelDisconnected,
                        );
                        if let Err(source) = response {
                            failures.push(
                                LiveNavigationWorkerError::WarmCheckpointDatasetNotPublishedResponse {
                                    source: Some(source),
                                },
                            );
                        }
                    } else {
                        let remaining = bridge
                            .checkpoint_deadline
                            .get()
                            .copied()
                            .map(|deadline| deadline.saturating_duration_since(Instant::now()))
                            .unwrap_or(Duration::ZERO);
                        match bridge.finalization.recv_timeout(remaining) {
                            Ok(NanoDatasetCheckpointFinalization::Published) => {
                                match production_state.as_mut() {
                                    Some(state) => {
                                        let response = state
                                            .map_persistence
                                            .respond_to_claimed_quiescent_warm_checkpoint_with_quota(
                                                pending.claimed,
                                                &bridge.dataset_directory,
                                                finalized_map_identity,
                                                &mut state.quota,
                                                !pending.console_response_pending,
                                            );
                                        match response {
                                            Ok(receipt) => {
                                                eprintln!(
                                                    "terminal warm checkpoint selected: dataset={} occupancy={} selection={} map_epoch={} revision={} content_binding={}; current_camera_localized=false",
                                                    receipt.dataset_directory().display(),
                                                    receipt.occupancy_snapshot_path().display(),
                                                    receipt.selection_path().display(),
                                                    receipt
                                                        .map_identity()
                                                        .map_epoch_id()
                                                        .as_u64(),
                                                    receipt.map_identity().revision(),
                                                    receipt.dataset_content_binding_status(),
                                                );
                                            }
                                            Err(source) => failures.push(
                                                LiveNavigationWorkerError::ProductionWarmCheckpoint {
                                                    source: Box::new(source),
                                                },
                                            ),
                                        }
                                    }
                                    None => failures.push(
                                        LiveNavigationWorkerError::ProductionMapPersistenceUnavailable {
                                            response: pending
                                                .claimed
                                                .reject(
                                                    AgentControlRejectionCodeV1::PersistenceFailed,
                                                    false,
                                                )
                                                .err(),
                                        },
                                    ),
                                }
                            }
                            Ok(NanoDatasetCheckpointFinalization::Rejected) => {
                                let response = if pending.console_response_pending {
                                    pending.claimed.reject(
                                        AgentControlRejectionCodeV1::PersistenceFailed,
                                        false,
                                    )
                                } else {
                                    pending.claimed.reject_after_wire_delivery(
                                        AgentControlRejectionCodeV1::PersistenceFailed,
                                        false,
                                    )
                                };
                                if let Err(source) = response {
                                    failures.push(
                                        LiveNavigationWorkerError::WarmCheckpointDatasetNotPublishedResponse {
                                            source: Some(source),
                                        },
                                    );
                                }
                            }
                            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                                let response = if pending.console_response_pending {
                                    pending.claimed.reject(
                                        AgentControlRejectionCodeV1::PersistenceFailed,
                                        false,
                                    )
                                } else {
                                    pending.claimed.reject_after_wire_delivery(
                                        AgentControlRejectionCodeV1::PersistenceFailed,
                                        false,
                                    )
                                };
                                failures.push(
                                    LiveNavigationWorkerError::WarmCheckpointFinalizationTimedOut,
                                );
                                if let Err(source) = response {
                                    failures.push(
                                        LiveNavigationWorkerError::WarmCheckpointDatasetNotPublishedResponse {
                                            source: Some(source),
                                        },
                                    );
                                }
                            }
                            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                                let response = if pending.console_response_pending {
                                    pending.claimed.reject(
                                        AgentControlRejectionCodeV1::PersistenceFailed,
                                        false,
                                    )
                                } else {
                                    pending.claimed.reject_after_wire_delivery(
                                        AgentControlRejectionCodeV1::PersistenceFailed,
                                        false,
                                    )
                                };
                                failures.push(
                                    LiveNavigationWorkerError::WarmCheckpointFinalizationChannelDisconnected,
                                );
                                if let Err(source) = response {
                                    failures.push(
                                        LiveNavigationWorkerError::WarmCheckpointDatasetNotPublishedResponse {
                                            source: Some(source),
                                        },
                                    );
                                }
                            }
                        }
                    }
                    if pending.console_response_pending {
                        let terminal_response_id = match console.adapter.as_mut() {
                            Some(adapter) => match adapter.complete_save_map_response() {
                                Ok(
                                    OperatorConsoleProcessDisposition::ResponseCompleted {
                                        downstream_request_id,
                                    }
                                    | OperatorConsoleProcessDisposition::ResponseRejected {
                                        downstream_request_id,
                                    },
                                ) => Some(downstream_request_id),
                                Ok(_) => {
                                    failures.push(
                                        LiveNavigationWorkerError::ProductionConsoleAdapter {
                                            source: OperatorConsoleRuntimeAdapterError::OwnerOutcomeMismatch,
                                        },
                                    );
                                    None
                                }
                                Err(source) => {
                                    failures.push(
                                        LiveNavigationWorkerError::ProductionConsoleAdapter {
                                            source,
                                        },
                                    );
                                    None
                                }
                            },
                            None => None,
                        };
                        if let Some(response_id) = terminal_response_id {
                            let observation_started = Instant::now();
                            let mut response_observed = false;
                            let mut frontend_remained_live = true;
                            while observation_started.elapsed()
                                < NANO_OPERATOR_CONSOLE_RESPONSE_OBSERVATION_GRACE
                            {
                                let observed = console.adapter.as_ref().is_some_and(|adapter| {
                                    adapter
                                        .handle()
                                        .response_record_was_http_observed(response_id)
                                });
                                if observed {
                                    response_observed = true;
                                    break;
                                }
                                if let Some(evidence) = console
                                    .frontend
                                    .as_mut()
                                    .and_then(NanoOperatorConsoleFrontend::poll_unexpected_exit)
                                {
                                    frontend_remained_live = false;
                                    failures.push(
                                        LiveNavigationWorkerError::ProductionConsoleFrontendExited {
                                            evidence: Box::new(evidence),
                                        },
                                    );
                                    break;
                                }
                                std::thread::sleep(Duration::from_millis(10));
                            }
                            if !response_observed && frontend_remained_live {
                                eprintln!(
                                    "operator-console terminal response {} remained retrievable for the bounded {} ms poll grace; client observation is unproven",
                                    response_id.get(),
                                    NANO_OPERATOR_CONSOLE_RESPONSE_OBSERVATION_GRACE.as_millis(),
                                );
                            } else if !response_observed {
                                eprintln!(
                                    "operator-console terminal response {} client observation is unproven because the HTTP owner exited during the poll grace",
                                    response_id.get(),
                                );
                            }
                        }
                    }
                }

                if let Some(adapter) = console.adapter.take() {
                    let outcome =
                        adapter.shutdown_with_terminal_evidence(terminal_shutdown_evidence);
                    eprintln!("operator console terminalization: {outcome:?}");
                }
                if let Some(frontend) = console.frontend.as_mut() {
                    let evidence = frontend.shutdown();
                    let retains_live_http_owner = evidence.retains_live_http_owner();
                    if !evidence.is_clean() {
                        failures.push(
                            LiveNavigationWorkerError::ProductionConsoleFrontendShutdown {
                                evidence: Box::new(evidence),
                            },
                        );
                    }
                    if !retains_live_http_owner {
                        console.frontend.take();
                    }
                }
                socket_task.request_shutdown();
                match socket_task.shutdown() {
                    Ok(exit) => {
                        if let Some(source) = abnormal_production_socket_exit(exit) {
                            failures.push(source);
                        }
                    }
                    Err(source) => {
                        failures.push(LiveNavigationWorkerError::ProductionSocketJoin { source });
                    }
                }
                prefinalized_descriptor = descriptor;
                None
            }
            #[cfg(not(all(feature = "nano-agent", unix)))]
            {
                match socket_task.shutdown() {
                    Ok(exit) => {
                        if let Some(source) = abnormal_production_socket_exit(exit) {
                            failures.push(source);
                        }
                    }
                    Err(source) => {
                        failures.push(LiveNavigationWorkerError::ProductionSocketJoin { source });
                    }
                }
                Some(coordinator)
            }
        }
        #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
        LiveNavigationRuntime::WheelsOffQualification(mut qualification) => {
            if operation_completed_without_error
                && let Some(selected) = qualification.fault_injection.unexercised_on_normal_exit()
            {
                failures.push(
                    LiveNavigationWorkerError::WheelsOffQualificationFaultNotExercised { selected },
                );
            }
            qualification.frontend_mut().request_shutdown();
            let (mut controller, mut frontend, mut attestation_gate) =
                qualification.take_terminal_parts();
            if let Err(source) = attestation_gate.cancel_and_join() {
                failures.push(
                    LiveNavigationWorkerError::WheelsOffQualificationAttestationCleanup { source },
                );
            }
            if let Err(source) = controller.shutdown() {
                failures.push(LiveNavigationWorkerError::WheelsOffQualificationRuntime {
                    source: Box::new(source),
                });
            }
            let evidence = frontend.shutdown();
            let http_clean = evidence.http().is_ok_and(|exit| {
                exit.graceful_shutdown
                    && !exit.forced_shutdown
                    && !exit.server_error
                    && !exit.clock_faulted
            });
            let capability_clean = matches!(
                evidence.capability(),
                kiko_slam::navigation::QualificationCapabilityShutdownEvidence::Cleaned(
                    kiko_slam::navigation::OperatorConsoleCapabilityCleanupEvidence::ExactEntryRemovedAndParentSynced
                )
            );
            if !http_clean || !capability_clean {
                failures.push(
                    LiveNavigationWorkerError::WheelsOffQualificationFrontendShutdown {
                        evidence: Box::new(evidence),
                    },
                );
            }
            Some(qualification.coordinator)
        }
    };

    let descriptor = match coordinator {
        Some(coordinator) => match finalize_live_navigation_coordinator(coordinator) {
            Ok(finalized) => Some(finalized.into_descriptor()),
            Err(source) => {
                failures.push(source);
                None
            }
        },
        None => prefinalized_descriptor,
    };
    combine_live_navigation_failures(failures)?;
    Ok(LiveNavigationWorkerSuccess {
        descriptor: descriptor.expect("a failure-free finalization returns a descriptor"),
    })
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
    SlamClock {
        source: HostMonotonicRangeError,
    },
    SlamTelemetry {
        source: LiveSlamTelemetryError,
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
    #[cfg(all(feature = "nano-agent", unix))]
    WarmStartRelocalization {
        source: NanoWarmStartRelocalizationError,
    },
    #[cfg(all(feature = "nano-agent", unix))]
    WarmStartRelocalizationIncomplete,
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
            LiveThreadError::SlamClock { source } => {
                write!(f, "live SLAM completion clock failed: {source}")
            }
            LiveThreadError::SlamTelemetry { source } => {
                write!(f, "live SLAM telemetry failed: {source}")
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
            #[cfg(all(feature = "nano-agent", unix))]
            LiveThreadError::WarmStartRelocalization { source } => {
                write!(f, "warm-start live relocalization failed: {source}")
            }
            #[cfg(all(feature = "nano-agent", unix))]
            LiveThreadError::WarmStartRelocalizationIncomplete => f.write_str(
                "fresh-camera relocalization did not complete before the inference input closed",
            ),
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
            Self::SlamClock { source } => Some(source),
            Self::SlamTelemetry { source } => Some(source),
            Self::RequiredDenseUnavailable { .. } => None,
            Self::RequiredDenseAndInferenceUnavailable { inference, .. } => Some(inference),
            Self::InferenceUnavailable { source } => Some(source),
            Self::DenseCommandRouteAndInferenceUnavailable { routing, .. } => Some(routing),
            Self::DenseCommandGenerationAndInferenceUnavailable { generation, .. } => {
                Some(generation)
            }
            #[cfg(all(feature = "nano-agent", unix))]
            Self::WarmStartRelocalization { source } => Some(source),
            #[cfg(all(feature = "nano-agent", unix))]
            Self::WarmStartRelocalizationIncomplete => None,
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

#[cfg(all(feature = "nano-agent", unix))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NanoFaceShutdownProblemKind {
    UnexpectedEvidence,
    DetachedUncertain,
}

#[cfg(all(feature = "nano-agent", unix))]
#[derive(Debug)]
struct NanoAccessoryShutdownSummary {
    terminal_fault: Option<NanoAccessoryTerminalFault>,
    eye_release_verified: bool,
    head_hold_preserving_release_completed: bool,
    fault_recovery_presentation: Box<NanoFaultRecoveryPresentationEvidence>,
    pet_evidence_clean: bool,
    face_perception: NanoFacePerceptionShutdownEvidence,
}

#[cfg(all(feature = "nano-agent", unix))]
impl NanoAccessoryShutdownSummary {
    fn from_evidence(
        terminal_fault: Option<NanoAccessoryTerminalFault>,
        evidence: NanoAccessoryShutdownEvidence,
    ) -> Self {
        let (eye, head, fault_recovery_presentation, pet_evidence, face_perception) =
            evidence.into_parts();
        Self {
            terminal_fault,
            eye_release_verified: eye.release_verified(),
            head_hold_preserving_release_completed: head.hold_preserving_release_completed(),
            fault_recovery_presentation: Box::new(fault_recovery_presentation),
            pet_evidence_clean: pet_evidence.clean(),
            face_perception,
        }
    }

    fn face_classification(&self) -> NanoFacePerceptionShutdownClass<'_> {
        self.face_perception.classify(self.terminal_fault.as_ref())
    }

    fn face_problem_kind(&self) -> Option<NanoFaceShutdownProblemKind> {
        match self.face_classification() {
            NanoFacePerceptionShutdownClass::Disabled
            | NanoFacePerceptionShutdownClass::CoordinatedShutdown
            | NanoFacePerceptionShutdownClass::PublishedRuntimeFault { .. }
            | NanoFacePerceptionShutdownClass::AccessoryFaultFollower { .. } => None,
            NanoFacePerceptionShutdownClass::UnexpectedDisabledFaceFault { .. }
            | NanoFacePerceptionShutdownClass::UnexpectedJoined { .. } => {
                Some(NanoFaceShutdownProblemKind::UnexpectedEvidence)
            }
            NanoFacePerceptionShutdownClass::DetachedAfterTimeout { .. } => {
                Some(NanoFaceShutdownProblemKind::DetachedUncertain)
            }
        }
    }

    fn face_stage_stats_are_final(&self) -> bool {
        !matches!(
            self.face_classification(),
            NanoFacePerceptionShutdownClass::DetachedAfterTimeout { .. }
        )
    }

    fn is_fully_healthy(&self) -> bool {
        self.terminal_fault.is_none()
            && self.eye_release_verified
            && self.head_hold_preserving_release_completed
            && matches!(
                self.fault_recovery_presentation.as_ref(),
                NanoFaultRecoveryPresentationEvidence::NotRequired
            )
            && self.pet_evidence_clean
            && self.face_classification().is_healthy()
    }
}

#[cfg(all(feature = "nano-agent", unix))]
#[derive(Debug)]
enum LiveAccessoryError {
    PreparationInterrupted(NanoLivePreparationInterrupted),
    #[cfg(feature = "nano-wheels-off-qualification")]
    Start(NanoAccessoryWorkerStartError),
    TerminalFault(NanoAccessoryTerminalFault),
    FaultMonitor(NanoAccessoryFaultWaitError),
    FrameIngress(NanoAccessoryFrameSubmitOutcome),
    ShutdownJoin(NanoAccessoryWorkerJoinError),
    UnexpectedExit(Box<NanoAccessoryWorkerExit>),
    EyeReleaseUnverified,
    HeadHoldPreservingReleaseUnverified,
    FaultRecoveryPresentationFailed {
        frames_applied: u64,
        source: Box<NanoFaultRecoveryPresentationFault>,
    },
    FaultRecoveryPresentationMissing,
    FaultRecoveryPresentationUnexpected(Box<NanoFaultRecoveryPresentationEvidence>),
    PetEvidenceShutdownUnclean,
    FaceShutdown {
        kind: NanoFaceShutdownProblemKind,
        evidence: NanoFacePerceptionShutdownEvidence,
    },
}

#[cfg(all(feature = "nano-agent", unix))]
fn stop_live_before_waiting_for_accessory_fault<T>(
    running: &AtomicBool,
    wait: impl FnOnce() -> T,
) -> T {
    running.store(false, Ordering::SeqCst);
    wait()
}

#[cfg(all(feature = "nano-agent", unix))]
const fn accessory_submission_requires_exact_fault_wait(
    outcome: NanoAccessoryFrameSubmitOutcome,
) -> bool {
    matches!(
        outcome,
        NanoAccessoryFrameSubmitOutcome::TerminalFaultPendingPublication
            | NanoAccessoryFrameSubmitOutcome::TerminalFaultLatched
    )
}

#[cfg(all(feature = "nano-agent", unix))]
impl std::fmt::Display for LiveAccessoryError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PreparationInterrupted(source) => source.fmt(formatter),
            #[cfg(feature = "nano-wheels-off-qualification")]
            Self::Start(source) => write!(formatter, "accessory startup failed: {source}"),
            Self::TerminalFault(source) => write!(formatter, "{source}"),
            Self::FaultMonitor(source) => {
                write!(formatter, "accessory fault monitor failed: {source}")
            }
            Self::FrameIngress(outcome) => {
                write!(formatter, "RGB accessory ingress failed: {outcome:?}")
            }
            Self::ShutdownJoin(source) => {
                write!(formatter, "accessory shutdown join failed: {source}")
            }
            Self::UnexpectedExit(exit) => {
                write!(formatter, "accessory worker exited unexpectedly: {exit:?}")
            }
            Self::EyeReleaseUnverified => {
                formatter.write_str("eye release could not be verified during accessory shutdown")
            }
            Self::HeadHoldPreservingReleaseUnverified => formatter.write_str(
                "the head hold-preserving ownership release could not be verified during accessory shutdown",
            ),
            Self::FaultRecoveryPresentationFailed {
                frames_applied,
                source,
            } => write!(
                formatter,
                "fault recovery applied {frames_applied} fresh eye frames after the primary stop latch, then failed: {source}"
            ),
            Self::FaultRecoveryPresentationMissing => formatter.write_str(
                "an accessory terminal fault was retained without a post-latch fault-recovery presentation attempt",
            ),
            Self::FaultRecoveryPresentationUnexpected(evidence) => write!(
                formatter,
                "fault-recovery presentation evidence existed without an accessory terminal fault: {evidence:?}"
            ),
            Self::PetEvidenceShutdownUnclean => formatter.write_str(
                "the pet-evidence writer did not complete a clean coordinated shutdown and join",
            ),
            Self::FaceShutdown { kind, evidence } => write!(
                formatter,
                "face-perception shutdown was {kind:?}; retained evidence: {evidence:?}"
            ),
        }
    }
}

#[cfg(all(feature = "nano-agent", unix))]
impl std::error::Error for LiveAccessoryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::PreparationInterrupted(source) => Some(source),
            #[cfg(feature = "nano-wheels-off-qualification")]
            Self::Start(source) => Some(source),
            Self::TerminalFault(source) => Some(source),
            Self::FaultRecoveryPresentationFailed { source, .. } => Some(source.as_ref()),
            Self::FaultMonitor(source) => Some(source),
            Self::ShutdownJoin(source) => Some(source),
            Self::FrameIngress(_)
            | Self::UnexpectedExit(_)
            | Self::EyeReleaseUnverified
            | Self::HeadHoldPreservingReleaseUnverified
            | Self::FaultRecoveryPresentationMissing
            | Self::FaultRecoveryPresentationUnexpected(_)
            | Self::PetEvidenceShutdownUnclean
            | Self::FaceShutdown { .. } => None,
        }
    }
}

#[cfg(all(feature = "nano-agent", unix))]
fn classify_fault_recovery_presentation(
    terminal_fault_present: bool,
    evidence: NanoFaultRecoveryPresentationEvidence,
) -> Option<LiveAccessoryError> {
    match (terminal_fault_present, evidence) {
        (false, NanoFaultRecoveryPresentationEvidence::NotRequired)
        | (true, NanoFaultRecoveryPresentationEvidence::Presented { .. }) => None,
        (
            true,
            NanoFaultRecoveryPresentationEvidence::Failed {
                frames_applied,
                source,
            },
        ) => Some(LiveAccessoryError::FaultRecoveryPresentationFailed {
            frames_applied,
            source: Box::new(source),
        }),
        (true, NanoFaultRecoveryPresentationEvidence::NotRequired) => {
            Some(LiveAccessoryError::FaultRecoveryPresentationMissing)
        }
        (false, evidence) => Some(LiveAccessoryError::FaultRecoveryPresentationUnexpected(
            Box::new(evidence),
        )),
    }
}

#[cfg(feature = "record")]
#[derive(Debug)]
enum LiveWorkerFailure {
    Capture(LiveCaptureError),
    Inference(LiveThreadError),
    InferencePanic {
        detail: String,
    },
    Occupancy(OccupancyRuntimeError),
    OccupancyPanic {
        detail: String,
    },
    Navigation(LiveNavigationWorkerError),
    NavigationPanic {
        detail: String,
    },
    SlamTelemetry(LiveSlamTelemetryError),
    DatasetFinalization(DatasetError),
    DatasetAbort(DatasetError),
    #[cfg(all(feature = "nano-agent", unix))]
    WarmCheckpointCoordination(NanoDatasetCheckpointCoordinationError),
    #[cfg(all(feature = "nano-agent", unix))]
    Accessory(LiveAccessoryError),
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
            Self::SlamTelemetry(source) => {
                write!(f, "live SLAM telemetry finalization failed: {source}")
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
            #[cfg(all(feature = "nano-agent", unix))]
            Self::WarmCheckpointCoordination(source) => source.fmt(f),
            #[cfg(all(feature = "nano-agent", unix))]
            Self::Accessory(source) => write!(f, "live accessory owner failed: {source}"),
        }
    }
}

#[cfg(feature = "record")]
impl std::error::Error for LiveWorkerFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Capture(source) => Some(source),
            Self::Inference(source) => Some(source),
            Self::Occupancy(source) => Some(source),
            Self::Navigation(source) => Some(source),
            Self::SlamTelemetry(source) => Some(source),
            Self::DatasetFinalization(source) | Self::DatasetAbort(source) => Some(source),
            #[cfg(all(feature = "nano-agent", unix))]
            Self::WarmCheckpointCoordination(source) => Some(source),
            #[cfg(all(feature = "nano-agent", unix))]
            Self::Accessory(source) => Some(source),
            Self::InferencePanic { .. }
            | Self::OccupancyPanic { .. }
            | Self::NavigationPanic { .. } => None,
        }
    }
}

#[cfg(all(feature = "record", feature = "nano-agent", unix))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NanoDatasetCheckpointCoordinationError {
    MissingBridge,
    RequestTimedOut,
    RequestChannelDisconnected,
    MissingDatasetOwner,
    FinalizationResponseChannelDisconnected,
}

#[cfg(all(feature = "record", feature = "nano-agent", unix))]
impl std::fmt::Display for NanoDatasetCheckpointCoordinationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::MissingBridge => {
                "terminal warm checkpoint was requested without a dataset coordination bridge"
            }
            Self::RequestTimedOut => {
                "terminal warm checkpoint timed out waiting for the finalized navigation journal"
            }
            Self::RequestChannelDisconnected => {
                "terminal warm checkpoint navigation worker exited before transferring its finalized journal"
            }
            Self::MissingDatasetOwner => {
                "terminal warm checkpoint has no live session dataset owner"
            }
            Self::FinalizationResponseChannelDisconnected => {
                "terminal warm checkpoint navigation worker exited before receiving dataset publication evidence"
            }
        })
    }
}

#[cfg(all(feature = "record", feature = "nano-agent", unix))]
impl std::error::Error for NanoDatasetCheckpointCoordinationError {}

/// A terminal Rerun worker problem is diagnostic evidence, never an
/// authoritative robot-owner failure. Keeping it outside [`LiveWorkerFailure`]
/// makes it impossible to append one to the service-failure ledger by mistake.
#[cfg(feature = "record")]
#[derive(Debug)]
enum LiveVisualizationFailure {
    Worker(LiveThreadError),
    Panic { detail: String },
}

#[cfg(feature = "record")]
impl std::fmt::Display for LiveVisualizationFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Worker(source) => write!(formatter, "live visualization worker failed: {source}"),
            Self::Panic { detail } => {
                write!(formatter, "live visualization worker panicked: {detail}")
            }
        }
    }
}

#[cfg(feature = "record")]
impl std::error::Error for LiveVisualizationFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Worker(source) => Some(source),
            Self::Panic { .. } => None,
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

#[cfg(feature = "record")]
#[derive(Debug)]
struct LiveThreadSpawnError {
    name: &'static str,
    source: std::io::Error,
}

#[cfg(feature = "record")]
impl std::fmt::Display for LiveThreadSpawnError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "failed to spawn required live worker thread {:?}: {}",
            self.name, self.source
        )
    }
}

#[cfg(feature = "record")]
impl std::error::Error for LiveThreadSpawnError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

#[cfg(feature = "record")]
fn spawn_live_thread<T, F>(
    name: &'static str,
    worker: F,
) -> Result<std::thread::JoinHandle<T>, LiveThreadSpawnError>
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    thread::Builder::new()
        .name(name.to_owned())
        .spawn(worker)
        .map_err(|source| LiveThreadSpawnError { name, source })
}

#[cfg(any(feature = "record", test))]
struct LiveThreadExitGuard {
    running: Arc<AtomicBool>,
    #[cfg(all(feature = "nano-agent", unix))]
    quiescent_checkpoint_requested: Option<Arc<AtomicBool>>,
}

#[cfg(any(feature = "record", test))]
impl LiveThreadExitGuard {
    fn new(running: Arc<AtomicBool>) -> Self {
        Self {
            running,
            #[cfg(all(feature = "nano-agent", unix))]
            quiescent_checkpoint_requested: None,
        }
    }

    #[cfg(all(feature = "record", feature = "nano-agent", unix))]
    fn checkpoint_aware(
        running: Arc<AtomicBool>,
        quiescent_checkpoint_requested: Arc<AtomicBool>,
    ) -> Self {
        Self {
            running,
            quiescent_checkpoint_requested: Some(quiescent_checkpoint_requested),
        }
    }
}

#[cfg(any(feature = "record", test))]
impl Drop for LiveThreadExitGuard {
    fn drop(&mut self) {
        #[cfg(all(feature = "nano-agent", unix))]
        if self
            .quiescent_checkpoint_requested
            .as_ref()
            .is_some_and(|requested| requested.load(Ordering::Acquire))
        {
            return;
        }
        self.running.store(false, Ordering::SeqCst);
    }
}

#[cfg(feature = "record")]
#[derive(Debug)]
enum LiveCaptureError {
    #[cfg(all(feature = "nano-agent", unix))]
    SystemdSupervision {
        source: NanoSystemdSupervisionError,
    },
    #[cfg(all(feature = "nano-agent", unix))]
    RgbImage {
        source: ImageError,
    },
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
    PairingInput {
        source: PairingInputError,
    },
    StereoObservation {
        source: StereoObservationError,
    },
    DatasetWrite {
        source: RecordCaptureError,
    },
    Depth {
        source: DepthError,
    },
    DepthFrame {
        source: RectifiedLeftDepthError,
    },
    DepthObservation {
        source: DepthObservationError,
    },
    Imu {
        source: ImuError,
    },
    ImuSample {
        source: InertialValueError,
    },
    ImuOrdering {
        source: InertialOrderingError,
    },
    ImuRouteDisconnected,
    HostTimestamp {
        source: HostMonotonicRangeError,
    },
    DeviceClose {
        source: DeviceCloseFailure,
    },
}

#[cfg(feature = "record")]
impl std::fmt::Display for LiveCaptureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            #[cfg(all(feature = "nano-agent", unix))]
            Self::SystemdSupervision { source } => {
                write!(f, "integrated service supervision failed: {source}")
            }
            #[cfg(all(feature = "nano-agent", unix))]
            Self::RgbImage { source } => write!(f, "RGB camera capture failed: {source}"),
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
            #[cfg(all(feature = "nano-agent", unix))]
            Self::SystemdSupervision { source } => Some(source),
            #[cfg(all(feature = "nano-agent", unix))]
            Self::RgbImage { source } => Some(source),
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
struct ActiveLiveNavigation {
    coordinator: ShadowNavigationCoordinator<NavigationIngressWriter<File>>,
    control_period: ControlPeriodNs,
    dataset_directory: PathBuf,
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
    let PreparedLiveNavigationRuntimeParts {
        goal,
        dataset_path,
        control_period,
        ingress_capacity,
        odometry,
        local_costmap,
        global_planner,
        reference_builder,
        mpc_config,
        solver_budget,
        safety,
        #[cfg(unix)]
        dataset_storage_limits,
        #[cfg(feature = "actuation")]
        actuation,
    } = runtime.into_parts();
    let meta = build_meta(mono_config, depth_config, imu_config, oak_provenance);
    let dataset_directory = dataset_path.clone();
    let imu_metadata = ImuStreamMetadata::new(
        device_session,
        ImuExtrinsicProvenance::uncalibrated_unknown(),
    );
    let writer_config = DatasetWriterConfig {
        backpressure: Backpressure::Block,
        ..DatasetWriterConfig::default()
    };
    #[cfg(unix)]
    let (dataset_writer, dataset_handle) = match dataset_storage_limits {
        Some(storage_limits) => DatasetWriter::create_paired_with_imu_config_and_storage_limits(
            &dataset_path,
            &meta,
            calibration,
            pairing_window,
            imu_metadata,
            writer_config,
            storage_limits,
        )?,
        None => DatasetWriter::create_paired_with_imu_config(
            &dataset_path,
            &meta,
            calibration,
            pairing_window,
            imu_metadata,
            writer_config,
        )?,
    };
    #[cfg(not(unix))]
    let (dataset_writer, dataset_handle) = DatasetWriter::create_paired_with_imu_config(
        &dataset_path,
        &meta,
        calibration,
        pairing_window,
        imu_metadata,
        writer_config,
    )?;
    let journal_path = dataset_path.join(NAVIGATION_INGRESS_STREAM_FILE);
    #[cfg(unix)]
    let quota_bound_journal = match dataset_writer.storage_quota() {
        Some(_) => match dataset_writer.create_quota_bound_navigation_ingress_file() {
            Ok((file, quota)) => Some((file, quota)),
            Err(source) => {
                return fail_navigation_setup_after_dataset(dataset_writer, dataset_handle, source);
            }
        },
        None => None,
    };
    #[cfg(unix)]
    let journal_file_result = match quota_bound_journal.as_ref() {
        Some((file, _)) => file.try_clone(),
        None => OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&journal_path),
    };
    #[cfg(not(unix))]
    let journal_file_result = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&journal_path);
    let journal_file = match journal_file_result {
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
    #[cfg(unix)]
    let journal_result = match quota_bound_journal {
        Some((_original_file, quota)) => NavigationIngressWriter::new_with_dataset_storage_quota(
            journal_file,
            recording_id,
            ingress_capacity,
            quota,
        ),
        None => NavigationIngressWriter::new(journal_file, recording_id, ingress_capacity),
    };
    #[cfg(not(unix))]
    let journal_result = NavigationIngressWriter::new(journal_file, recording_id, ingress_capacity);
    let journal = match journal_result {
        Ok(journal) => journal,
        Err(source) => {
            return fail_navigation_setup_after_dataset(dataset_writer, dataset_handle, source);
        }
    };
    let coordinator = match goal {
        Some(goal) => ShadowNavigationCoordinator::new(
            clock_epoch,
            journal,
            goal.point(),
            odometry,
            local_costmap,
            global_planner,
            reference_builder,
            mpc_config,
            solver_budget,
            safety,
        ),
        None => ShadowNavigationCoordinator::new_without_goal(
            clock_epoch,
            journal,
            odometry,
            local_costmap,
            global_planner,
            reference_builder,
            mpc_config,
            solver_budget,
            safety,
        ),
    };
    Ok(ActiveLiveNavigation {
        coordinator,
        control_period,
        dataset_directory,
        dataset_writer,
        dataset_handle,
        #[cfg(feature = "actuation")]
        actuation,
    })
}

#[cfg(feature = "record")]
enum LiveRerunTarget {
    Connect,
    #[cfg(all(feature = "nano-agent", unix))]
    ServeLoopback {
        bind: std::net::SocketAddr,
        memory_limit_bytes: u64,
    },
}

#[cfg(feature = "record")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PreparedTrackerInitialization {
    /// Compatibility/offline live mode retains its documented environment
    /// tuning boundary.
    Environment,
    /// Canonical robot modes use one fixed typed configuration and explicit
    /// runtime policy; they never inspect ambient tracker variables.
    #[cfg(all(feature = "nano-agent", unix))]
    CanonicalNano,
}

#[cfg(all(feature = "record", feature = "actuation"))]
enum PreparedLiveMotionSelection {
    Compatibility,
    #[cfg(all(feature = "nano-agent", unix))]
    Production(Box<LiveProductionMotionInput>),
    #[cfg(all(feature = "nano-attended-navigation-trial", unix))]
    AttendedNavigationTrial(Box<LiveAttendedNavigationTrialMotionInput>),
    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    WheelsOffQualification(Box<LiveWheelsOffQualificationMotionInput>),
}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "nano-agent",
    unix
))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NanoLiveMotionKind {
    Compatibility,
    Production,
    #[cfg(feature = "nano-attended-navigation-trial")]
    AttendedNavigationTrial,
    #[cfg(feature = "nano-wheels-off-qualification")]
    WheelsOffQualification,
}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "nano-agent",
    unix
))]
impl NanoLiveMotionKind {
    const fn requires_navigation_stop_before_accessory_release(self) -> bool {
        match self {
            #[cfg(feature = "nano-attended-navigation-trial")]
            Self::AttendedNavigationTrial => true,
            #[cfg(feature = "nano-wheels-off-qualification")]
            Self::WheelsOffQualification => true,
            Self::Compatibility | Self::Production => false,
        }
    }
}

#[cfg(all(
    feature = "record",
    feature = "actuation",
    feature = "nano-agent",
    unix
))]
impl PreparedLiveMotionSelection {
    const fn nano_live_motion_kind(&self) -> NanoLiveMotionKind {
        match self {
            Self::Compatibility => NanoLiveMotionKind::Compatibility,
            Self::Production(_) => NanoLiveMotionKind::Production,
            #[cfg(feature = "nano-attended-navigation-trial")]
            Self::AttendedNavigationTrial(_) => NanoLiveMotionKind::AttendedNavigationTrial,
            #[cfg(feature = "nano-wheels-off-qualification")]
            Self::WheelsOffQualification(_) => NanoLiveMotionKind::WheelsOffQualification,
        }
    }
}

#[cfg(feature = "record")]
enum LiveOccupancyWorkerStartup {
    Fresh(OccupancyRuntimeConfig),
    #[cfg(all(feature = "nano-agent", unix))]
    ContinuedReplay {
        runtime: Box<OccupancyRuntime>,
        initial_snapshot: TimedOccupancySnapshot,
    },
}

#[cfg(feature = "record")]
struct PreparedLiveSession {
    device: Device,
    device_session: DeviceSessionId,
    mono_config: MonoConfig,
    depth_config: Option<DepthConfig>,
    imu_config: Option<ImuConfig>,
    depth_queue_capacity: Option<ChannelCapacity>,
    depth_ring_capacity: DepthRingCapacity,
    imu_session: Option<DeviceSessionId>,
    imu_queue_capacity: Option<ChannelCapacity>,
    dense_requested: bool,
    dense_data_capacity: ChannelCapacity,
    dense_control_capacity: ChannelCapacity,
    pairing_window: PairingWindowNs,
    pairer: StereoPairer,
    calibration: Calibration,
    rectified_left_intrinsics: OakIntrinsics,
    prepared_navigation_runtime: Option<PreparedLiveNavigationRuntime>,
    inference: InferenceConfig,
    tracker_initialization: PreparedTrackerInitialization,
    pair_queue_depth: usize,
    viz_queue_depth: usize,
    rerun_decimation: VizDecimation,
    rerun_finish_timeout: Duration,
    rerun_target: LiveRerunTarget,
    oak_provenance: OakRuntimeProvenance,
    #[cfg(feature = "actuation")]
    motion: PreparedLiveMotionSelection,
    #[cfg(all(feature = "nano-agent", unix))]
    accessory: Option<NanoAccessoryWorker>,
    #[cfg(all(feature = "nano-agent", unix))]
    production_state: Option<NanoProductionStateOwners>,
    #[cfg(all(feature = "nano-agent", unix))]
    warm_start_replay: Option<Box<NanoDatasetReplayRequired>>,
    #[cfg(all(feature = "nano-agent", unix))]
    systemd_supervision: Option<NanoSystemdRuntimeSupervision>,
}

/// Owns the admitted zero-only controller and already-held head while the
/// common live runtime still has fallible setup to perform. On any early
/// return, controller stop is attempted before accessory shutdown.
#[cfg(all(feature = "nano-agent", unix))]
struct NanoLiveSetupGuard {
    motion: Option<PreparedLiveMotionSelection>,
    accessory: Option<NanoAccessoryWorker>,
    production_state: Option<NanoProductionStateOwners>,
    accessory_terminal_fault_reported: bool,
}

#[cfg(all(feature = "nano-agent", unix))]
impl NanoLiveSetupGuard {
    fn new(
        motion: PreparedLiveMotionSelection,
        accessory: Option<NanoAccessoryWorker>,
        production_state: Option<NanoProductionStateOwners>,
    ) -> Self {
        Self {
            motion: Some(motion),
            accessory,
            production_state,
            accessory_terminal_fault_reported: false,
        }
    }

    fn take_motion(&mut self) -> PreparedLiveMotionSelection {
        self.motion
            .take()
            .expect("live setup transfers one motion selection")
    }

    fn bind_head_gaze_lease_if_configured(
        &self,
    ) -> Result<(), kiko_slam::navigation::NanoHeadGazeLeaseBindError> {
        let issuer = match self.motion.as_ref() {
            Some(PreparedLiveMotionSelection::Production(input)) => input.head_gaze_lease_issuer(),
            #[cfg(feature = "nano-attended-navigation-trial")]
            Some(PreparedLiveMotionSelection::AttendedNavigationTrial(input)) => {
                input.head_gaze_lease_issuer()
            }
            _ => None,
        };
        let (Some(accessory), Some(issuer)) = (self.accessory.as_ref(), issuer) else {
            return Ok(());
        };
        match accessory.bind_head_gaze_base_zero_lease_issuer(issuer) {
            Ok(())
            | Err(kiko_slam::navigation::NanoHeadGazeLeaseBindError::PhysicalGazeNotConfigured) => {
                Ok(())
            }
            Err(source) => Err(source),
        }
    }

    fn take_production_state(&mut self) -> Option<NanoProductionStateOwners> {
        self.production_state.take()
    }

    fn take_accessory(&mut self) -> Option<NanoAccessoryWorker> {
        self.accessory.take()
    }

    fn take_face_diagnostics(
        &mut self,
    ) -> Option<(NanoFaceDiagnosticReceiver, NanoFaceDiagnosticStatsHandle)> {
        self.accessory
            .as_mut()
            .and_then(NanoAccessoryWorker::take_face_diagnostics)
    }

    fn face_perception_stage_stats_handle(&self) -> Option<NanoFacePerceptionStageStatsHandle> {
        self.accessory
            .as_ref()
            .and_then(NanoAccessoryWorker::face_perception_stage_stats_handle)
    }

    fn require_accessory_healthy_if_present(
        &mut self,
        running: &AtomicBool,
    ) -> Result<(), LiveAccessoryError> {
        if !running.load(Ordering::Acquire) {
            return Err(LiveAccessoryError::PreparationInterrupted(
                NanoLivePreparationInterrupted,
            ));
        }
        let Some(accessory) = self.accessory.as_ref() else {
            return Ok(());
        };
        match accessory.try_terminal_fault() {
            Ok(None) => Ok(()),
            Ok(Some(fault)) => {
                self.accessory_terminal_fault_reported = true;
                Err(LiveAccessoryError::TerminalFault(fault))
            }
            Err(source) => Err(LiveAccessoryError::FaultMonitor(source)),
        }
    }
}

#[cfg(all(feature = "nano-agent", unix))]
impl Drop for NanoLiveSetupGuard {
    fn drop(&mut self) {
        // Dropping an unused production motion input invokes its explicit
        // abort-before-owner path. Do this before removing natural hold.
        drop(self.motion.take());
        let Some(accessory) = self.accessory.take() else {
            return;
        };
        match accessory.shutdown() {
            Ok(NanoAccessoryWorkerExit::Shutdown {
                terminal_fault,
                evidence,
            }) => {
                let summary =
                    NanoAccessoryShutdownSummary::from_evidence(terminal_fault, *evidence);
                if !summary.is_fully_healthy() {
                    if self.accessory_terminal_fault_reported && summary.terminal_fault.is_some() {
                        eprintln!(
                            "early live setup accessory shutdown was not fully healthy: terminal_fault=retained_for_consistency_and_already_reported_by_primary eye_release_verified={} head_hold_preserving_release_completed={} fault_recovery_presentation={:?} pet_evidence_clean={} face_perception={}",
                            summary.eye_release_verified,
                            summary.head_hold_preserving_release_completed,
                            summary.fault_recovery_presentation,
                            summary.pet_evidence_clean,
                            summary.face_classification(),
                        );
                    } else {
                        eprintln!(
                            "early live setup accessory shutdown was not fully healthy: terminal_fault={:?} eye_release_verified={} head_hold_preserving_release_completed={} fault_recovery_presentation={:?} pet_evidence_clean={} face_perception={}",
                            summary.terminal_fault,
                            summary.eye_release_verified,
                            summary.head_hold_preserving_release_completed,
                            summary.fault_recovery_presentation,
                            summary.pet_evidence_clean,
                            summary.face_classification(),
                        );
                    }
                }
            }
            Ok(exit) => {
                eprintln!("early live setup accessory worker exited unexpectedly: {exit:?}");
            }
            Err(source) => {
                eprintln!("early live setup accessory shutdown join failed: {source}");
            }
        }
    }
}

/// Keeps the natural hold coupled to the production base owner until every
/// remaining worker thread has been created. If a later thread spawn unwinds,
/// the shared stop is asserted, the navigation owner is joined (and therefore
/// stops the controller), and only then are the accessories shut down.
#[cfg(all(feature = "nano-agent", unix))]
struct NanoPostNavigationSetupGuard {
    running: Arc<AtomicBool>,
    navigation:
        Option<thread::JoinHandle<Result<LiveNavigationWorkerSuccess, LiveNavigationWorkerError>>>,
    accessory: Option<NanoAccessoryWorker>,
}

#[cfg(all(feature = "nano-agent", unix))]
impl NanoPostNavigationSetupGuard {
    fn new(
        running: Arc<AtomicBool>,
        navigation: Option<
            thread::JoinHandle<Result<LiveNavigationWorkerSuccess, LiveNavigationWorkerError>>,
        >,
        accessory: Option<NanoAccessoryWorker>,
    ) -> Self {
        Self {
            running,
            navigation,
            accessory,
        }
    }

    fn into_parts(
        mut self,
    ) -> (
        Option<thread::JoinHandle<Result<LiveNavigationWorkerSuccess, LiveNavigationWorkerError>>>,
        Option<NanoAccessoryWorker>,
    ) {
        let navigation = self.navigation.take();
        let accessory = self.accessory.take();
        (navigation, accessory)
    }
}

#[cfg(all(feature = "nano-agent", unix))]
impl Drop for NanoPostNavigationSetupGuard {
    fn drop(&mut self) {
        self.running.store(false, Ordering::SeqCst);
        if let Some(handle) = self.navigation.take() {
            match handle.join() {
                Ok(Ok(_)) => {}
                Ok(Err(source)) => {
                    eprintln!("post-navigation setup cleanup observed owner failure: {source}");
                }
                Err(payload) => eprintln!(
                    "post-navigation setup cleanup observed owner panic: {}",
                    kiko_slam::panic_payload_to_string(payload.as_ref())
                ),
            }
        }
        let Some(accessory) = self.accessory.take() else {
            return;
        };
        match accessory.shutdown() {
            Ok(NanoAccessoryWorkerExit::Shutdown {
                terminal_fault,
                evidence,
            }) => {
                let summary =
                    NanoAccessoryShutdownSummary::from_evidence(terminal_fault, *evidence);
                if !summary.is_fully_healthy() {
                    eprintln!(
                        "post-navigation setup accessory shutdown was not fully healthy: terminal_fault={:?} eye_release_verified={} head_hold_preserving_release_completed={} fault_recovery_presentation={:?} pet_evidence_clean={} face_perception={}",
                        summary.terminal_fault,
                        summary.eye_release_verified,
                        summary.head_hold_preserving_release_completed,
                        summary.fault_recovery_presentation,
                        summary.pet_evidence_clean,
                        summary.face_classification(),
                    );
                }
            }
            Ok(exit) => {
                eprintln!("post-navigation setup accessory worker exited unexpectedly: {exit:?}");
            }
            Err(source) => {
                eprintln!("post-navigation setup accessory shutdown join failed: {source}");
            }
        }
    }
}

#[cfg(all(feature = "nano-agent", unix))]
#[derive(Debug)]
enum NanoPreOwnerControllerStop {
    Confirmed(DisarmReceipt),
    Uncertain(LiveActuationError),
}

#[cfg(all(feature = "nano-agent", unix))]
#[derive(Debug)]
enum NanoPreOwnerOakClose {
    Confirmed,
    Uncertain(OakCloseError),
}

#[cfg(all(feature = "nano-agent", unix))]
#[derive(Debug)]
enum NanoPreOwnerAccessoryShutdown {
    NotStarted,
    Evidence {
        summary: NanoAccessoryShutdownSummary,
        terminal_fault_already_reported: bool,
    },
    UnexpectedExit(Box<NanoAccessoryWorkerExit>),
    JoinUncertain(NanoAccessoryWorkerJoinError),
}

#[cfg(all(feature = "nano-agent", unix))]
impl std::fmt::Display for NanoPreOwnerAccessoryShutdown {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotStarted => formatter.write_str("accessory shutdown not started"),
            Self::Evidence {
                summary,
                terminal_fault_already_reported,
            } => {
                if *terminal_fault_already_reported && summary.terminal_fault.is_some() {
                    write!(
                        formatter,
                        "accessory shutdown evidence (terminal_fault=retained_for_consistency_and_already_reported_by_primary, eye_release_verified={}, head_hold_preserving_release_completed={}, fault_recovery_presentation={:?}, pet_evidence_clean={}, face_perception={})",
                        summary.eye_release_verified,
                        summary.head_hold_preserving_release_completed,
                        summary.fault_recovery_presentation,
                        summary.pet_evidence_clean,
                        summary.face_classification(),
                    )
                } else {
                    write!(
                        formatter,
                        "accessory shutdown evidence (terminal_fault={:?}, eye_release_verified={}, head_hold_preserving_release_completed={}, fault_recovery_presentation={:?}, pet_evidence_clean={}, face_perception={})",
                        summary.terminal_fault,
                        summary.eye_release_verified,
                        summary.head_hold_preserving_release_completed,
                        summary.fault_recovery_presentation,
                        summary.pet_evidence_clean,
                        summary.face_classification(),
                    )
                }
            }
            Self::UnexpectedExit(exit) => {
                write!(
                    formatter,
                    "accessory shutdown returned unexpected exit: {exit:?}"
                )
            }
            Self::JoinUncertain(source) => {
                write!(formatter, "accessory shutdown join uncertain: {source}")
            }
        }
    }
}

#[cfg(all(feature = "nano-agent", unix))]
fn shutdown_nano_pre_owner_accessory(
    accessory: NanoAccessoryWorker,
    terminal_fault_already_reported: bool,
) -> NanoPreOwnerAccessoryShutdown {
    match accessory.shutdown() {
        Ok(NanoAccessoryWorkerExit::Shutdown {
            terminal_fault,
            evidence,
        }) => NanoPreOwnerAccessoryShutdown::Evidence {
            summary: NanoAccessoryShutdownSummary::from_evidence(terminal_fault, *evidence),
            terminal_fault_already_reported,
        },
        Ok(exit) => NanoPreOwnerAccessoryShutdown::UnexpectedExit(Box::new(exit)),
        Err(source) => NanoPreOwnerAccessoryShutdown::JoinUncertain(source),
    }
}

#[cfg(all(feature = "nano-agent", unix))]
fn live_accessory_error_reports_terminal(source: &(dyn std::error::Error + 'static)) -> bool {
    matches!(
        source.downcast_ref::<LiveAccessoryError>(),
        Some(LiveAccessoryError::TerminalFault(_))
    )
}

#[cfg(all(feature = "nano-agent", unix))]
#[derive(Debug)]
struct NanoLivePreparationError {
    primary: Box<dyn std::error::Error>,
    controller_stop: NanoPreOwnerControllerStop,
    accessory_shutdown: NanoPreOwnerAccessoryShutdown,
    oak_close: NanoPreOwnerOakClose,
}

#[cfg(all(feature = "nano-agent", unix))]
impl std::fmt::Display for NanoLivePreparationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "Nano live preparation failed: {}", self.primary)?;
        match &self.controller_stop {
            NanoPreOwnerControllerStop::Confirmed(receipt) => write!(
                formatter,
                "; controller stop confirmed at {} ns",
                receipt.acknowledged_at().nanos_since_clock_start()
            )?,
            NanoPreOwnerControllerStop::Uncertain(source) => {
                write!(formatter, "; controller stop uncertain: {source}")?
            }
        }
        match &self.accessory_shutdown {
            NanoPreOwnerAccessoryShutdown::NotStarted => {}
            shutdown => write!(formatter, "; {shutdown}")?,
        }
        match &self.oak_close {
            NanoPreOwnerOakClose::Confirmed => formatter.write_str("; OAK close confirmed"),
            NanoPreOwnerOakClose::Uncertain(source) => {
                write!(formatter, "; OAK close uncertain: {source}")
            }
        }
    }
}

#[cfg(all(feature = "nano-agent", unix))]
impl std::error::Error for NanoLivePreparationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.primary.as_ref())
    }
}

#[cfg(all(feature = "nano-agent", unix))]
struct NanoPreOwnerResources {
    runtime: Option<PreparedNanoProductionRuntime>,
    accessory: Option<NanoAccessoryWorker>,
    oak: Option<Device>,
}

#[cfg(all(feature = "nano-agent", unix))]
impl NanoPreOwnerResources {
    fn new(
        runtime: PreparedNanoProductionRuntime,
        accessory: NanoAccessoryWorker,
        oak: Device,
    ) -> Self {
        Self {
            runtime: Some(runtime),
            accessory: Some(accessory),
            oak: Some(oak),
        }
    }

    fn cleanup(
        &mut self,
        terminal_fault_already_reported: bool,
    ) -> (
        NanoPreOwnerControllerStop,
        NanoPreOwnerAccessoryShutdown,
        NanoPreOwnerOakClose,
    ) {
        let controller_stop = match self
            .runtime
            .take()
            .expect("pre-owner runtime is cleaned exactly once")
            .abort_before_owner()
        {
            Ok(receipt) => NanoPreOwnerControllerStop::Confirmed(receipt),
            Err(source) => NanoPreOwnerControllerStop::Uncertain(source),
        };
        let accessory_shutdown = match self.accessory.take() {
            None => NanoPreOwnerAccessoryShutdown::NotStarted,
            Some(accessory) => {
                shutdown_nano_pre_owner_accessory(accessory, terminal_fault_already_reported)
            }
        };
        let oak_close = match self
            .oak
            .take()
            .expect("pre-owner OAK is closed exactly once")
            .close()
        {
            Ok(()) => NanoPreOwnerOakClose::Confirmed,
            Err(source) => NanoPreOwnerOakClose::Uncertain(source),
        };
        (controller_stop, accessory_shutdown, oak_close)
    }

    fn fail_box<T>(
        mut self,
        primary: Box<dyn std::error::Error>,
    ) -> Result<T, Box<dyn std::error::Error>> {
        let terminal_fault_already_reported =
            live_accessory_error_reports_terminal(primary.as_ref());
        let (controller_stop, accessory_shutdown, oak_close) =
            self.cleanup(terminal_fault_already_reported);
        Err(Box::new(NanoLivePreparationError {
            primary,
            controller_stop,
            accessory_shutdown,
            oak_close,
        }))
    }

    fn into_parts(mut self) -> (PreparedNanoProductionRuntime, NanoAccessoryWorker, Device) {
        let runtime = self
            .runtime
            .take()
            .expect("pre-owner runtime transfers exactly once");
        let accessory = self
            .accessory
            .take()
            .expect("started accessory owner transfers exactly once");
        let oak = self
            .oak
            .take()
            .expect("pre-owner OAK transfers exactly once");
        (runtime, accessory, oak)
    }
}

#[cfg(all(feature = "nano-agent", unix))]
impl Drop for NanoPreOwnerResources {
    fn drop(&mut self) {
        if self.runtime.is_none() && self.accessory.is_none() && self.oak.is_none() {
            return;
        }
        let (controller_stop, accessory_shutdown, oak_close) = self.cleanup(false);
        eprintln!(
            "Nano pre-owner resources unwound: controller_stop={controller_stop:?} accessory_shutdown={accessory_shutdown:?} oak_close={oak_close:?}"
        );
    }
}

/// Owns a bootstrap-confirmed stopped candidate controller while the common
/// Nano software stack is still being prepared. Unlike production admission,
/// this owner never contains an armed session: failure drops the sole
/// reacquisition token, shuts down accessories, and closes OAK while retaining
/// the exact bootstrap stop receipt in the returned error.
#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
struct NanoQualificationPreOwnerResources {
    stopped_controller: Option<kiko_slam::navigation::StoppedWheelsOffCandidateController>,
    initial_zero: Option<AppliedCommandReceipt>,
    initial_stop: Option<DisarmReceipt>,
    limits: kiko_slam::navigation::WheelsOffCandidateLimits,
    runtime_service_interval: kiko_slam::navigation::WheelsOffCandidateRuntimeServiceInterval,
    accessory: Option<NanoAccessoryWorker>,
    oak: Option<Device>,
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
impl NanoQualificationPreOwnerResources {
    fn new(
        stopped_controller: kiko_slam::navigation::StoppedWheelsOffCandidateController,
        initial_zero: AppliedCommandReceipt,
        initial_stop: DisarmReceipt,
        limits: kiko_slam::navigation::WheelsOffCandidateLimits,
        runtime_service_interval: kiko_slam::navigation::WheelsOffCandidateRuntimeServiceInterval,
        oak: Device,
    ) -> Self {
        Self {
            stopped_controller: Some(stopped_controller),
            initial_zero: Some(initial_zero),
            initial_stop: Some(initial_stop),
            limits,
            runtime_service_interval,
            accessory: None,
            oak: Some(oak),
        }
    }

    fn fail_box<T>(
        mut self,
        primary: Box<dyn std::error::Error>,
    ) -> Result<T, Box<dyn std::error::Error>> {
        let initial_stop = self
            .initial_stop
            .take()
            .expect("qualification bootstrap stop receipt is consumed once");
        let _ = self.initial_zero.take();
        drop(self.stopped_controller.take());
        let terminal_fault_already_reported =
            live_accessory_error_reports_terminal(primary.as_ref());
        let accessory_shutdown = match self.accessory.take() {
            None => NanoPreOwnerAccessoryShutdown::NotStarted,
            Some(accessory) => {
                shutdown_nano_pre_owner_accessory(accessory, terminal_fault_already_reported)
            }
        };
        let oak_close = match self
            .oak
            .take()
            .expect("qualification pre-owner OAK is closed exactly once")
            .close()
        {
            Ok(()) => NanoPreOwnerOakClose::Confirmed,
            Err(source) => NanoPreOwnerOakClose::Uncertain(source),
        };
        Err(Box::new(NanoQualificationLivePreparationError {
            primary,
            initial_stop,
            accessory_shutdown,
            oak_close,
        }))
    }

    #[allow(clippy::type_complexity)]
    fn into_parts(
        mut self,
    ) -> (
        kiko_slam::navigation::StoppedWheelsOffCandidateController,
        AppliedCommandReceipt,
        DisarmReceipt,
        kiko_slam::navigation::WheelsOffCandidateLimits,
        kiko_slam::navigation::WheelsOffCandidateRuntimeServiceInterval,
        NanoAccessoryWorker,
        Device,
    ) {
        (
            self.stopped_controller
                .take()
                .expect("qualification stopped token transfers exactly once"),
            self.initial_zero
                .take()
                .expect("qualification initial zero transfers exactly once"),
            self.initial_stop
                .take()
                .expect("qualification initial stop transfers exactly once"),
            self.limits,
            self.runtime_service_interval,
            self.accessory
                .take()
                .expect("qualification accessory transfers exactly once"),
            self.oak
                .take()
                .expect("qualification OAK transfers exactly once"),
        )
    }
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
impl Drop for NanoQualificationPreOwnerResources {
    fn drop(&mut self) {
        if self.stopped_controller.is_none()
            && self.initial_zero.is_none()
            && self.initial_stop.is_none()
            && self.accessory.is_none()
            && self.oak.is_none()
        {
            return;
        }
        let initial_stop = self.initial_stop.take();
        let _ = self.initial_zero.take();
        drop(self.stopped_controller.take());
        let accessory_shutdown = self
            .accessory
            .take()
            .map(|accessory| shutdown_nano_pre_owner_accessory(accessory, false));
        let oak_close = self.oak.take().map(Device::close);
        eprintln!(
            "qualification pre-owner resources unwound from an exact bootstrap stop: stop_request_id={:?} accessory_shutdown={accessory_shutdown:?} oak_close={oak_close:?}",
            initial_stop
                .as_ref()
                .map(|receipt| receipt.request_id().get()),
        );
    }
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
#[derive(Debug)]
struct NanoQualificationLivePreparationError {
    primary: Box<dyn std::error::Error>,
    initial_stop: DisarmReceipt,
    accessory_shutdown: NanoPreOwnerAccessoryShutdown,
    oak_close: NanoPreOwnerOakClose,
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
impl std::fmt::Display for NanoQualificationLivePreparationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "Nano wheels-off qualification preparation failed: {}; controller remained stopped by exact request {} on boot {}",
            self.primary,
            self.initial_stop.request_id().get(),
            self.initial_stop.observed_boot_id().get(),
        )?;
        write!(
            formatter,
            "; {}; OAK close: {:?}",
            self.accessory_shutdown, self.oak_close,
        )
    }
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
impl std::error::Error for NanoQualificationLivePreparationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.primary.as_ref())
    }
}

/// Production admits map publication and its exact map byte/headroom policy
/// as one process-lifetime capability. Keeping these owners inseparable makes
/// a quota-free production map save path unrepresentable after startup.
#[cfg(all(feature = "nano-agent", unix))]
struct NanoProductionStateOwners {
    map_persistence: NanoMapPersistenceOwner,
    quota: NanoStateQuotaOwner,
}

#[cfg(all(feature = "nano-agent", unix))]
impl NanoProductionStateOwners {
    fn admit(
        roots: &kiko_slam::navigation::NanoBootstrapRoots,
        storage: &NanoLaunchStorage,
        map_config: &NanoMapPersistenceConfig,
        map_limits: OccupancyMapLimits,
    ) -> Result<Self, NanoProductionStateAdmissionError> {
        let launch_destination = roots.state_root().join(storage.map_snapshot().as_path());
        let policy_destination = map_config.save_snapshot_path().as_path();
        if launch_destination != policy_destination {
            return Err(NanoProductionStateAdmissionError::MapDestinationMismatch {
                launch_destination,
                policy_destination: policy_destination.to_path_buf(),
            });
        }

        // Map quota admission creates only the parsed map parent. Persistence
        // then requires that exact parent to exist and independently admits it.
        let quota = NanoStateQuotaOwner::admit(roots, storage)
            .map_err(NanoProductionStateAdmissionError::Quota)?;
        let map_persistence = NanoMapPersistenceOwner::try_new(roots, map_config, map_limits)
            .map_err(NanoProductionStateAdmissionError::MapPersistence)?;
        Ok(Self {
            map_persistence,
            quota,
        })
    }
}

#[cfg(all(feature = "nano-agent", unix))]
#[derive(Debug)]
enum NanoProductionStateAdmissionError {
    MapDestinationMismatch {
        launch_destination: PathBuf,
        policy_destination: PathBuf,
    },
    Quota(NanoStateQuotaAdmissionError),
    MapPersistence(NanoMapPersistencePathError),
}

#[cfg(all(feature = "nano-agent", unix))]
impl std::fmt::Display for NanoProductionStateAdmissionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MapDestinationMismatch {
                launch_destination,
                policy_destination,
            } => write!(
                formatter,
                "launch map destination {} does not equal admitted agent-policy destination {}",
                launch_destination.display(),
                policy_destination.display()
            ),
            Self::Quota(source) => {
                write!(formatter, "Nano map-storage admission failed: {source}")
            }
            Self::MapPersistence(source) => {
                write!(formatter, "Nano map persistence admission failed: {source}")
            }
        }
    }
}

#[cfg(all(feature = "nano-agent", unix))]
impl std::error::Error for NanoProductionStateAdmissionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::MapDestinationMismatch { .. } => None,
            Self::Quota(source) => Some(source),
            Self::MapPersistence(source) => Some(source),
        }
    }
}

#[cfg(all(feature = "nano-agent", unix))]
struct NanoLiveSoftwarePreparation {
    device_session: DeviceSessionId,
    queue_size: usize,
    queue_capacity: ChannelCapacity,
    dense_control_capacity: ChannelCapacity,
    mono_config: MonoConfig,
    depth_config: DepthConfig,
    imu_config: ImuConfig,
    pairing_window: PairingWindowNs,
    pairer: StereoPairer,
    calibration: Calibration,
    rectified_left_intrinsics: OakIntrinsics,
    navigation: PreparedLiveNavigationRuntime,
    inference: InferenceConfig,
    rerun_decimation: VizDecimation,
    rerun_finish_timeout: Duration,
    rerun_target: LiveRerunTarget,
    rerun_diagnostics_url: ConsoleRerunDiagnosticsUrl,
    production_state: NanoProductionStateOwners,
    warm_start_replay: Option<Box<NanoDatasetReplayRequired>>,
}

#[cfg(all(feature = "nano-agent", unix))]
enum NanoOrtRuntimeInput<'config> {
    RetainedBytes(&'config [u8]),
    #[cfg(feature = "nano-wheels-off-qualification")]
    Pinned(kiko_slam::PinnedOrtRuntime),
}

#[cfg(all(feature = "nano-agent", unix))]
struct NanoCommonLiveSoftwareInput<'config> {
    roots: &'config NanoBootstrapRoots,
    graph: &'config NanoOakStreamGraph,
    storage: &'config NanoLaunchStorage,
    occupancy: &'config NanoLaunchOccupancy,
    inference_policy: &'config NanoLaunchInference,
    rerun: NanoLaunchRerun,
    onnx_runtime: NanoOrtRuntimeInput<'config>,
    superpoint_model: &'config [u8],
    lightglue_model: &'config [u8],
    stereo: NanoBootstrapStereoEvidence,
    live: ParsedNanoLiveConfiguration,
    map_persistence: &'config NanoMapPersistenceConfig,
    stream_epoch: StreamEpochId,
}

/// Construct the OAK/SLAM/occupancy/inference half shared by production and
/// the separately compiled qualifier. This helper has no physical-actuation
/// input. Callers retain their own linear controller token throughout every
/// fallible model and persistence operation.
#[cfg(all(feature = "nano-agent", unix))]
fn prepare_nano_common_live_software(
    input: NanoCommonLiveSoftwareInput<'_>,
    mut require_accessory_healthy: impl FnMut() -> Result<(), Box<dyn std::error::Error>>,
) -> Result<NanoLiveSoftwarePreparation, Box<dyn std::error::Error>> {
    require_accessory_healthy()?;
    let graph_config = input.graph.device_config();
    let mono_config = graph_config.mono.ok_or(NanoLiveGraphInvariantError::Mono)?;
    let depth_config = graph_config
        .depth
        .ok_or(NanoLiveGraphInvariantError::Depth)?;
    let imu_config = graph_config.imu.ok_or(NanoLiveGraphInvariantError::Imu)?;
    let queue_size = usize::try_from(input.graph.queue_size())?;
    let queue_capacity = ChannelCapacity::try_from(queue_size)?;
    let dense_control_capacity = ChannelCapacity::try_from(64_usize)?;
    let device_session = DeviceSessionId::try_new(1)?;

    let pairing_window = PairingWindowNs::try_from_u64(DEFAULT_PAIRING_WINDOW_NS)?;
    let mut pairer = StereoPairer::new_with_max_pending(pairing_window, queue_size)?;
    let rectified_left_intrinsics = input.stereo.left.intrinsics();
    pairer.push_left(oak_to_frame(input.stereo.left, SensorId::StereoLeft)?)?;
    pairer.push_right(oak_to_frame(input.stereo.right, SensorId::StereoRight)?)?;

    let dataset_path = input
        .roots
        .state_root()
        .join(input.storage.navigation_dataset_directory().as_path())
        .join(format!("session-{:016x}", input.stream_epoch.get()));
    let dataset_limits = input.storage.navigation_dataset_limits();
    let storage_limits = DatasetStorageLimits::try_new(
        dataset_limits.maximum_bytes(),
        dataset_limits.maximum_files(),
        dataset_limits.minimum_free_bytes_after_write(),
        dataset_limits.terminal_reserve_bytes(),
        input.storage.maximum_map_snapshot_bytes(),
        MAX_PRODUCTION_DATASET_MANIFEST_BYTES,
        MAX_NANO_WARM_SELECTION_BYTES,
    )?;
    let mut navigation = prepare_live_navigation_runtime_from_parsed(
        input.live.navigation,
        None,
        dataset_path,
        input.live.occupancy_host_policy,
        device_session,
    )?;
    navigation.bind_production_dataset_storage(
        storage_limits,
        dataset_limits.maximum_ingress_records(),
    )?;

    let map_limits = OccupancyMapLimits::try_new(input.occupancy.geometry().cell_count())?;
    let production_state = NanoProductionStateOwners::admit(
        input.roots,
        input.storage,
        input.map_persistence,
        map_limits,
    )?;
    let warm_start_replay = match production_state.map_persistence.load_warm_start()? {
        NanoMapWarmStartLoad::Disabled => None,
        NanoMapWarmStartLoad::DatasetReplayRequired(replay) => Some(replay),
    };

    require_accessory_healthy()?;
    let runtime = match input.onnx_runtime {
        NanoOrtRuntimeInput::RetainedBytes(runtime_bytes) => {
            kiko_slam::pin_ort_runtime_from_memory(runtime_bytes)?
        }
        #[cfg(feature = "nano-wheels-off-qualification")]
        NanoOrtRuntimeInput::Pinned(runtime) => runtime,
    };
    let superpoint_backend = input.inference_policy.superpoint_backend().runtime();
    let lightglue_backend = input.inference_policy.lightglue_backend().runtime();
    require_accessory_healthy()?;
    let superpoint_left = SuperPoint::new_from_memory_with_backend(
        input.superpoint_model,
        runtime,
        superpoint_backend,
    )?;
    require_accessory_healthy()?;
    let superpoint_right = SuperPoint::new_from_memory_with_backend(
        input.superpoint_model,
        runtime,
        superpoint_backend,
    )?;
    require_accessory_healthy()?;
    let lightglue =
        LightGlue::new_from_memory_with_backend(input.lightglue_model, runtime, lightglue_backend)?;
    require_accessory_healthy()?;
    let inference = InferenceConfig {
        superpoint_left,
        superpoint_right,
        lightglue,
        #[cfg(feature = "record")]
        superpoint_requested_backend: superpoint_backend,
        #[cfg(feature = "record")]
        lightglue_requested_backend: lightglue_backend,
        key_limit: KeypointLimit::try_from(usize::try_from(
            input.inference_policy.maximum_keypoints(),
        )?)?,
        downscale: DownscaleFactor::try_from(usize::try_from(
            input.inference_policy.downscale_factor(),
        )?)?,
    };

    Ok(NanoLiveSoftwarePreparation {
        device_session,
        queue_size,
        queue_capacity,
        dense_control_capacity,
        mono_config,
        depth_config,
        imu_config,
        pairing_window,
        pairer,
        calibration: input.stereo.calibration,
        rectified_left_intrinsics,
        navigation,
        inference,
        rerun_decimation: VizDecimation::try_from(usize::try_from(input.rerun.decimation())?)?,
        rerun_finish_timeout: Duration::from_millis(input.rerun.flush_timeout_ms()),
        rerun_target: LiveRerunTarget::ServeLoopback {
            bind: input.rerun.bind(),
            memory_limit_bytes: input.rerun.memory_limit_bytes(),
        },
        rerun_diagnostics_url: input.rerun.diagnostics_url(),
        production_state,
        warm_start_replay,
    })
}

#[cfg(all(feature = "nano-agent", unix))]
#[derive(Debug)]
enum NanoStreamEpochError {
    EntropyOpen(std::io::Error),
    EntropyRead(std::io::Error),
    NonzeroCandidateExhausted { attempts: usize },
}

#[cfg(all(feature = "nano-agent", unix))]
impl std::fmt::Display for NanoStreamEpochError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EntropyOpen(source) => {
                write!(formatter, "could not open the OS random source: {source}")
            }
            Self::EntropyRead(source) => {
                write!(formatter, "could not read the OS random source: {source}")
            }
            Self::NonzeroCandidateExhausted { attempts } => write!(
                formatter,
                "OS randomness produced only the reserved zero stream epoch in {attempts} attempts"
            ),
        }
    }
}

#[cfg(all(feature = "nano-agent", unix))]
impl std::error::Error for NanoStreamEpochError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::EntropyOpen(source) | Self::EntropyRead(source) => Some(source),
            Self::NonzeroCandidateExhausted { .. } => None,
        }
    }
}

#[cfg(all(feature = "nano-agent", unix))]
const MAX_NANO_STREAM_EPOCH_ATTEMPTS: usize = 8;

#[cfg(all(feature = "nano-agent", unix))]
fn fresh_nano_stream_epoch_from(
    entropy: &mut impl Read,
    maximum_attempts: usize,
) -> Result<StreamEpochId, NanoStreamEpochError> {
    for _ in 0..maximum_attempts {
        let mut bytes = [0_u8; std::mem::size_of::<u64>()];
        entropy
            .read_exact(&mut bytes)
            .map_err(NanoStreamEpochError::EntropyRead)?;
        if let Ok(epoch) = StreamEpochId::try_new(u64::from_ne_bytes(bytes)) {
            return Ok(epoch);
        }
    }
    Err(NanoStreamEpochError::NonzeroCandidateExhausted {
        attempts: maximum_attempts,
    })
}

#[cfg(all(feature = "nano-agent", unix))]
fn fresh_nano_stream_epoch() -> Result<StreamEpochId, NanoStreamEpochError> {
    let mut entropy = File::open("/dev/urandom").map_err(NanoStreamEpochError::EntropyOpen)?;
    fresh_nano_stream_epoch_from(&mut entropy, MAX_NANO_STREAM_EPOCH_ATTEMPTS)
}

#[cfg(all(feature = "nano-agent", unix))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NanoLiveGraphInvariantError {
    Mono,
    Depth,
    Imu,
}

#[cfg(all(feature = "nano-agent", unix))]
impl std::fmt::Display for NanoLiveGraphInvariantError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "parsed Nano OAK graph violated its mandatory-stream invariant: {self:?}"
        )
    }
}

#[cfg(all(feature = "nano-agent", unix))]
impl std::error::Error for NanoLiveGraphInvariantError {}

#[cfg(all(feature = "nano-agent", unix))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct NanoWarmStartOccupancyUnavailable;

#[cfg(all(feature = "nano-agent", unix))]
impl std::fmt::Display for NanoWarmStartOccupancyUnavailable {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(
            "admitted Nano warm start has no live occupancy configuration to replay into",
        )
    }
}

#[cfg(all(feature = "nano-agent", unix))]
impl std::error::Error for NanoWarmStartOccupancyUnavailable {}

#[cfg(all(feature = "nano-agent", unix))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct NanoLivePreparationInterrupted;

#[cfg(all(feature = "nano-agent", unix))]
impl std::fmt::Display for NanoLivePreparationInterrupted {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(
            "Nano live preparation was interrupted by shutdown or controller-owner failure",
        )
    }
}

#[cfg(all(feature = "nano-agent", unix))]
impl std::error::Error for NanoLivePreparationInterrupted {}

#[cfg(all(feature = "nano-agent", unix))]
fn require_pre_owner_accessory_healthy(
    resources: &NanoPreOwnerResources,
    running: &AtomicBool,
) -> Result<(), LiveAccessoryError> {
    if !running.load(Ordering::Acquire) {
        return Err(LiveAccessoryError::PreparationInterrupted(
            NanoLivePreparationInterrupted,
        ));
    }
    let accessory = resources
        .accessory
        .as_ref()
        .expect("software preparation begins only after accessory readiness");
    match accessory.try_terminal_fault() {
        Ok(None) => Ok(()),
        Ok(Some(fault)) => Err(LiveAccessoryError::TerminalFault(fault)),
        Err(source) => Err(LiveAccessoryError::FaultMonitor(source)),
    }
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
fn require_qualification_pre_owner_accessory_healthy(
    resources: &NanoQualificationPreOwnerResources,
    running: &AtomicBool,
) -> Result<(), LiveAccessoryError> {
    if !running.load(Ordering::Acquire) {
        return Err(LiveAccessoryError::PreparationInterrupted(
            NanoLivePreparationInterrupted,
        ));
    }
    let accessory = resources
        .accessory
        .as_ref()
        .expect("qualification software preparation starts after accessory readiness");
    match accessory.try_terminal_fault() {
        Ok(None) => Ok(()),
        Ok(Some(fault)) => Err(LiveAccessoryError::TerminalFault(fault)),
        Err(source) => Err(LiveAccessoryError::FaultMonitor(source)),
    }
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
const MAX_QUALIFICATION_ATTESTATION_LINE_BYTES: usize = 128;

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
const QUALIFICATION_ATTESTATION_CHALLENGE_BYTES: usize = 16;

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
const QUALIFICATION_ATTESTATION_TTY: &str = "/dev/tty";

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
const QUALIFICATION_ATTESTATION_POLL_SLICE: Duration = Duration::from_millis(10);

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
#[derive(Debug)]
struct InitialMotorPowerDisconnectedClaim;

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
#[derive(Debug)]
struct AttendedWheelsOffPreflight {
    motor_power_disconnected: InitialMotorPowerDisconnectedClaim,
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
#[derive(Debug)]
enum AttendedWheelsOffAttestationError {
    TtyRequired,
    Interrupted,
    OpenControllingTty(std::io::Error),
    DiscardPendingInput(std::io::Error),
    ChallengeEntropy(getrandom::Error),
    Input(std::io::Error),
    Output(std::io::Error),
    EndOfInput,
    LineTooLong { maximum_bytes: usize },
    InvalidUtf8,
    PhraseMismatch { expected: String },
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
impl std::fmt::Display for AttendedWheelsOffAttestationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TtyRequired => formatter
                .write_str("wheels-off qualification requires attended terminal input and output"),
            Self::Interrupted => {
                formatter.write_str("wheels-off qualification attestation was cancelled")
            }
            Self::OpenControllingTty(source) => {
                write!(formatter, "could not open the controlling TTY: {source}")
            }
            Self::DiscardPendingInput(source) => {
                write!(
                    formatter,
                    "could not discard pending controlling-TTY input: {source}"
                )
            }
            Self::ChallengeEntropy(source) => write!(
                formatter,
                "could not create a fresh qualification confirmation challenge: {source}"
            ),
            Self::Input(source) => write!(formatter, "could not read qualification TTY: {source}"),
            Self::Output(source) => {
                write!(
                    formatter,
                    "could not write qualification TTY prompt: {source}"
                )
            }
            Self::EndOfInput => formatter.write_str(
                "qualification TTY closed before every physical precondition was confirmed",
            ),
            Self::LineTooLong { maximum_bytes } => write!(
                formatter,
                "qualification TTY response exceeded {maximum_bytes} bytes"
            ),
            Self::InvalidUtf8 => {
                formatter.write_str("qualification TTY response was not valid UTF-8")
            }
            Self::PhraseMismatch { expected } => write!(
                formatter,
                "qualification physical precondition was not confirmed; expected exact phrase {expected:?}"
            ),
        }
    }
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
impl std::error::Error for AttendedWheelsOffAttestationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::OpenControllingTty(source)
            | Self::DiscardPendingInput(source)
            | Self::Input(source)
            | Self::Output(source) => Some(source),
            Self::ChallengeEntropy(source) => Some(source),
            Self::TtyRequired
            | Self::Interrupted
            | Self::EndOfInput
            | Self::LineTooLong { .. }
            | Self::InvalidUtf8
            | Self::PhraseMismatch { .. } => None,
        }
    }
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
#[derive(Debug)]
enum FreshAttendedMotionAttestationError {
    Terminal(AttendedWheelsOffAttestationError),
    Domain(kiko_slam::navigation::WheelsOffCandidateAttestationError),
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
impl std::fmt::Display for FreshAttendedMotionAttestationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Terminal(source) => source.fmt(formatter),
            Self::Domain(source) => source.fmt(formatter),
        }
    }
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
impl std::error::Error for FreshAttendedMotionAttestationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Terminal(source) => Some(source),
            Self::Domain(source) => Some(source),
        }
    }
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
#[derive(Debug)]
enum FreshAttendedMotionAttestationWorkerError {
    Spawn(std::io::Error),
    Dialog(FreshAttendedMotionAttestationError),
    Panicked { detail: String },
    JoinedWithoutGateTransition,
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
impl std::fmt::Display for FreshAttendedMotionAttestationWorkerError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Spawn(source) => {
                write!(
                    formatter,
                    "could not start attended motion-attestation worker: {source}"
                )
            }
            Self::Dialog(source) => source.fmt(formatter),
            Self::Panicked { detail } => {
                write!(
                    formatter,
                    "attended motion-attestation worker panicked: {detail}"
                )
            }
            Self::JoinedWithoutGateTransition => formatter.write_str(
                "attended motion-attestation worker was already joined without enabling its gate",
            ),
        }
    }
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
impl std::error::Error for FreshAttendedMotionAttestationWorkerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Spawn(source) => Some(source),
            Self::Dialog(source) => Some(source),
            Self::Panicked { .. } | Self::JoinedWithoutGateTransition => None,
        }
    }
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FreshAttendedMotionAttestationWorkerPoll {
    Pending,
    Ready(kiko_slam::navigation::OperatorClaimedWheelsOffAttestation),
    Completed,
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FreshAttendedMotionAttestationWorkerShutdown {
    AlreadyJoined,
    Cancelled,
    CompletedBeforeCancellation,
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
struct FreshAttendedMotionAttestationInput {
    preflight: AttendedWheelsOffPreflight,
    console: kiko_slam::navigation::WheelsOffQualificationConsoleHandle,
    process_running: Arc<AtomicBool>,
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FreshAttendedMotionAttestationClosure {
    SoftwareSafetyStopLatched,
    ProcessNotRunning,
    ReadinessLost(WheelsOffQualificationAttestationReadinessBlocker),
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
enum FreshAttendedMotionAttestationGate {
    AwaitingReadOnlyCycle(FreshAttendedMotionAttestationInput),
    WaitingForOperator(FreshAttendedMotionAttestationWorker),
    Completed,
    Closed(FreshAttendedMotionAttestationClosure),
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
impl FreshAttendedMotionAttestationGate {
    const fn is_closed(&self) -> bool {
        self.closure().is_some()
    }

    const fn closure(&self) -> Option<FreshAttendedMotionAttestationClosure> {
        match self {
            Self::Closed(reason) => Some(*reason),
            Self::AwaitingReadOnlyCycle(_) | Self::WaitingForOperator(_) | Self::Completed => None,
        }
    }

    const fn has_started_prompt(&self) -> bool {
        matches!(self, Self::WaitingForOperator(_))
    }

    fn advance_after_read_only_runtime_tick(
        &mut self,
        process_running: &AtomicBool,
    ) -> Result<FreshAttendedMotionAttestationWorkerPoll, FreshAttendedMotionAttestationWorkerError>
    {
        if !process_running.load(Ordering::SeqCst) {
            self.close_without_enablement(
                FreshAttendedMotionAttestationClosure::ProcessNotRunning,
            )?;
            return Ok(FreshAttendedMotionAttestationWorkerPoll::Completed);
        }
        if matches!(self, Self::AwaitingReadOnlyCycle(_)) {
            let previous = std::mem::replace(self, Self::Completed);
            let Self::AwaitingReadOnlyCycle(input) = previous else {
                unreachable!("matching gate phase is transferred exactly once");
            };
            let worker = FreshAttendedMotionAttestationWorker::spawn(input)
                .map_err(FreshAttendedMotionAttestationWorkerError::Spawn)?;
            *self = Self::WaitingForOperator(worker);
            return Ok(FreshAttendedMotionAttestationWorkerPoll::Pending);
        }
        match self {
            Self::WaitingForOperator(worker) => match worker.poll()? {
                FreshAttendedMotionAttestationWorkerPoll::Pending => {
                    Ok(FreshAttendedMotionAttestationWorkerPoll::Pending)
                }
                FreshAttendedMotionAttestationWorkerPoll::Ready(attestation) => {
                    if !process_running.load(Ordering::SeqCst) {
                        *self =
                            Self::Closed(FreshAttendedMotionAttestationClosure::ProcessNotRunning);
                        return Ok(FreshAttendedMotionAttestationWorkerPoll::Completed);
                    }
                    *self = Self::Completed;
                    Ok(FreshAttendedMotionAttestationWorkerPoll::Ready(attestation))
                }
                FreshAttendedMotionAttestationWorkerPoll::Completed => {
                    unreachable!("a worker cannot report a gate-owned completed state")
                }
            },
            Self::Completed => Ok(FreshAttendedMotionAttestationWorkerPoll::Completed),
            Self::Closed(_) => Ok(FreshAttendedMotionAttestationWorkerPoll::Completed),
            Self::AwaitingReadOnlyCycle(_) => {
                unreachable!("awaiting phase returns after spawning its worker")
            }
        }
    }

    fn close_without_enablement(
        &mut self,
        reason: FreshAttendedMotionAttestationClosure,
    ) -> Result<
        FreshAttendedMotionAttestationWorkerShutdown,
        FreshAttendedMotionAttestationWorkerError,
    > {
        let shutdown = match self {
            Self::WaitingForOperator(worker) => worker.cancel_and_join()?,
            Self::AwaitingReadOnlyCycle(_) | Self::Completed | Self::Closed(_) => {
                FreshAttendedMotionAttestationWorkerShutdown::AlreadyJoined
            }
        };
        *self = Self::Closed(reason);
        Ok(shutdown)
    }

    fn cancel_and_join(
        &mut self,
    ) -> Result<
        FreshAttendedMotionAttestationWorkerShutdown,
        FreshAttendedMotionAttestationWorkerError,
    > {
        match self {
            Self::AwaitingReadOnlyCycle(_) | Self::Completed | Self::Closed(_) => {
                Ok(FreshAttendedMotionAttestationWorkerShutdown::AlreadyJoined)
            }
            Self::WaitingForOperator(worker) => worker.cancel_and_join(),
        }
    }
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
struct FreshAttendedMotionAttestationWorker {
    cancellation_requested: Arc<AtomicBool>,
    handle: Option<
        std::thread::JoinHandle<
            Result<
                kiko_slam::navigation::OperatorClaimedWheelsOffAttestation,
                FreshAttendedMotionAttestationError,
            >,
        >,
    >,
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
impl FreshAttendedMotionAttestationWorker {
    fn spawn(input: FreshAttendedMotionAttestationInput) -> Result<Self, std::io::Error> {
        let cancellation_requested = Arc::new(AtomicBool::new(false));
        let worker_cancellation = Arc::clone(&cancellation_requested);
        let handle = std::thread::Builder::new()
            .name("kiko-motion-attestation".to_owned())
            .spawn(move || {
                let FreshAttendedMotionAttestationInput {
                    preflight,
                    console,
                    process_running,
                } = input;
                let cancellation = ConsoleAwareFreshAttendedMotionCancellation {
                    local: worker_cancellation.as_ref(),
                    console: &console,
                    process_running: process_running.as_ref(),
                };
                run_real_fresh_attended_motion_attestation(preflight, &cancellation)
            })?;
        Ok(Self {
            cancellation_requested,
            handle: Some(handle),
        })
    }

    #[cfg(test)]
    fn spawn_with(
        preflight: AttendedWheelsOffPreflight,
        worker: impl FnOnce(
            AttendedWheelsOffPreflight,
            &AtomicBool,
        ) -> Result<
            kiko_slam::navigation::OperatorClaimedWheelsOffAttestation,
            FreshAttendedMotionAttestationError,
        > + Send
        + 'static,
    ) -> Result<Self, std::io::Error> {
        let cancellation_requested = Arc::new(AtomicBool::new(false));
        let worker_cancellation = Arc::clone(&cancellation_requested);
        let handle = std::thread::Builder::new()
            .name("kiko-motion-attestation".to_owned())
            .spawn(move || worker(preflight, worker_cancellation.as_ref()))?;
        Ok(Self {
            cancellation_requested,
            handle: Some(handle),
        })
    }

    fn poll(
        &mut self,
    ) -> Result<FreshAttendedMotionAttestationWorkerPoll, FreshAttendedMotionAttestationWorkerError>
    {
        let Some(handle) = self.handle.as_ref() else {
            return Err(FreshAttendedMotionAttestationWorkerError::JoinedWithoutGateTransition);
        };
        if !handle.is_finished() {
            return Ok(FreshAttendedMotionAttestationWorkerPoll::Pending);
        }
        let handle = self
            .handle
            .take()
            .expect("finished attestation handle is still retained");
        match handle.join() {
            Ok(Ok(attestation)) => Ok(FreshAttendedMotionAttestationWorkerPoll::Ready(attestation)),
            Ok(Err(source)) => Err(FreshAttendedMotionAttestationWorkerError::Dialog(source)),
            Err(payload) => Err(FreshAttendedMotionAttestationWorkerError::Panicked {
                detail: kiko_slam::panic_payload_to_string(payload.as_ref()),
            }),
        }
    }

    fn cancel_and_join(
        &mut self,
    ) -> Result<
        FreshAttendedMotionAttestationWorkerShutdown,
        FreshAttendedMotionAttestationWorkerError,
    > {
        self.cancellation_requested.store(true, Ordering::Release);
        let Some(handle) = self.handle.take() else {
            return Ok(FreshAttendedMotionAttestationWorkerShutdown::AlreadyJoined);
        };
        match handle.join() {
            Ok(Ok(_attestation)) => {
                Ok(FreshAttendedMotionAttestationWorkerShutdown::CompletedBeforeCancellation)
            }
            Ok(Err(FreshAttendedMotionAttestationError::Terminal(
                AttendedWheelsOffAttestationError::Interrupted,
            ))) => Ok(FreshAttendedMotionAttestationWorkerShutdown::Cancelled),
            Ok(Err(source)) => Err(FreshAttendedMotionAttestationWorkerError::Dialog(source)),
            Err(payload) => Err(FreshAttendedMotionAttestationWorkerError::Panicked {
                detail: kiko_slam::panic_payload_to_string(payload.as_ref()),
            }),
        }
    }
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
impl Drop for FreshAttendedMotionAttestationWorker {
    fn drop(&mut self) {
        self.cancellation_requested.store(true, Ordering::Release);
        if let Some(handle) = self.handle.take()
            && let Err(payload) = handle.join()
        {
            eprintln!(
                "attended motion-attestation worker panicked during drop: {}",
                kiko_slam::panic_payload_to_string(payload.as_ref())
            );
        }
    }
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
#[derive(Debug)]
struct WheelsOffQualificationAndMotorPowerDisconnectError {
    operation: Box<dyn std::error::Error>,
    motor_power_disconnect: AttendedWheelsOffAttestationError,
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
impl std::fmt::Display for WheelsOffQualificationAndMotorPowerDisconnectError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "wheels-off qualification failed: {}; mandatory post-run motor-power disconnect confirmation also failed: {}",
            self.operation, self.motor_power_disconnect,
        )
    }
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
impl std::error::Error for WheelsOffQualificationAndMotorPowerDisconnectError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.operation.as_ref())
    }
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
trait WheelsOffAttestationChallengeSource {
    fn next_challenge(
        &mut self,
    ) -> Result<[u8; QUALIFICATION_ATTESTATION_CHALLENGE_BYTES], AttendedWheelsOffAttestationError>;
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
struct OsWheelsOffAttestationChallengeSource;

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
impl WheelsOffAttestationChallengeSource for OsWheelsOffAttestationChallengeSource {
    fn next_challenge(
        &mut self,
    ) -> Result<[u8; QUALIFICATION_ATTESTATION_CHALLENGE_BYTES], AttendedWheelsOffAttestationError>
    {
        let mut challenge = [0_u8; QUALIFICATION_ATTESTATION_CHALLENGE_BYTES];
        getrandom::fill(&mut challenge)
            .map_err(AttendedWheelsOffAttestationError::ChallengeEntropy)?;
        Ok(challenge)
    }
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
fn lower_hex_qualification_challenge(
    bytes: &[u8; QUALIFICATION_ATTESTATION_CHALLENGE_BYTES],
) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(QUALIFICATION_ATTESTATION_CHALLENGE_BYTES * 2);
    for byte in bytes {
        encoded.push(char::from(HEX[usize::from(byte >> 4)]));
        encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    encoded
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
fn read_bounded_tty_line(
    input: &mut impl BufRead,
) -> Result<String, AttendedWheelsOffAttestationError> {
    let mut output = Vec::with_capacity(48);
    loop {
        let available = input
            .fill_buf()
            .map_err(AttendedWheelsOffAttestationError::Input)?;
        if available.is_empty() {
            return Err(AttendedWheelsOffAttestationError::EndOfInput);
        }
        if let Some(newline) = available.iter().position(|byte| *byte == b'\n') {
            if output.len().saturating_add(newline) > MAX_QUALIFICATION_ATTESTATION_LINE_BYTES {
                return Err(AttendedWheelsOffAttestationError::LineTooLong {
                    maximum_bytes: MAX_QUALIFICATION_ATTESTATION_LINE_BYTES,
                });
            }
            output.extend_from_slice(&available[..newline]);
            input.consume(newline + 1);
            if output.last() == Some(&b'\r') {
                output.pop();
            }
            return String::from_utf8(output)
                .map_err(|_| AttendedWheelsOffAttestationError::InvalidUtf8);
        }
        if output.len().saturating_add(available.len()) > MAX_QUALIFICATION_ATTESTATION_LINE_BYTES {
            return Err(AttendedWheelsOffAttestationError::LineTooLong {
                maximum_bytes: MAX_QUALIFICATION_ATTESTATION_LINE_BYTES,
            });
        }
        output.extend_from_slice(available);
        let consumed = available.len();
        input.consume(consumed);
    }
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
fn prompt_fresh_attended_phrase(
    input: &mut impl BufRead,
    output: &mut impl Write,
    challenges: &mut impl WheelsOffAttestationChallengeSource,
    explanation: &str,
    phrase: &'static str,
) -> Result<(), AttendedWheelsOffAttestationError> {
    // The challenge is created only after this physical boundary is reached.
    // Input queued for an earlier prompt therefore cannot reuse a known
    // response; it would have to predict this fresh 128-bit value.
    let challenge = challenges.next_challenge()?;
    let expected = format!("{phrase} {}", lower_hex_qualification_challenge(&challenge));
    writeln!(output, "{explanation}")
        .and_then(|()| write!(output, "Type {expected:?}: "))
        .and_then(|()| output.flush())
        .map_err(AttendedWheelsOffAttestationError::Output)?;
    let actual = read_bounded_tty_line(input)?;
    if actual != expected {
        return Err(AttendedWheelsOffAttestationError::PhraseMismatch { expected });
    }
    Ok(())
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
fn read_attended_wheels_off_preflight(
    input: &mut impl BufRead,
    output: &mut impl Write,
    challenges: &mut impl WheelsOffAttestationChallengeSource,
) -> Result<AttendedWheelsOffPreflight, AttendedWheelsOffAttestationError> {
    prompt_fresh_attended_phrase(
        input,
        output,
        challenges,
        "Confirm that both drive wheels are physically removed. Software cannot observe this.",
        "WHEELS REMOVED",
    )?;
    prompt_fresh_attended_phrase(
        input,
        output,
        challenges,
        "Confirm that the head is physically supported before the natural-hold actor starts.",
        "HEAD SUPPORTED",
    )?;
    prompt_fresh_attended_phrase(
        input,
        output,
        challenges,
        "Physically disconnect the motor output power supply while leaving only the controller logic/serial path available for stopped-device qualification. Confirm that motor power is disconnected before any device is opened.",
        "MOTOR POWER PHYSICALLY DISCONNECTED",
    )?;
    prompt_fresh_attended_phrase(
        input,
        output,
        challenges,
        "Confirm that an independent physical power cut is immediately reachable.",
        "POWER CUT REACHABLE",
    )?;
    Ok(AttendedWheelsOffPreflight {
        motor_power_disconnected: InitialMotorPowerDisconnectedClaim,
    })
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
fn require_attended_wheels_off_preflight()
-> Result<AttendedWheelsOffPreflight, AttendedWheelsOffAttestationError> {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    if !stdin.is_terminal() || !stdout.is_terminal() {
        return Err(AttendedWheelsOffAttestationError::TtyRequired);
    }
    let mut challenges = OsWheelsOffAttestationChallengeSource;
    read_attended_wheels_off_preflight(&mut stdin.lock(), &mut stdout.lock(), &mut challenges)
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
trait FreshAttendedMotionCancellation {
    fn is_cancelled(&self) -> bool;
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
impl FreshAttendedMotionCancellation for AtomicBool {
    fn is_cancelled(&self) -> bool {
        self.load(Ordering::Acquire)
    }
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
struct ConsoleAwareFreshAttendedMotionCancellation<'a> {
    local: &'a AtomicBool,
    console: &'a kiko_slam::navigation::WheelsOffQualificationConsoleHandle,
    process_running: &'a AtomicBool,
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
fn fresh_motion_attestation_must_cancel(
    local_cancellation_requested: bool,
    process_running: bool,
    snapshot: &kiko_slam::navigation::WheelsOffQualificationSnapshot,
) -> bool {
    local_cancellation_requested
        || !process_running
        || snapshot.software_safety_stop_latched
        || snapshot.frontend_state
            != kiko_slam::navigation::WheelsOffQualificationFrontendState::Connected
        || snapshot.runtime_ingress_state
            != kiko_slam::navigation::WheelsOffQualificationRuntimeIngressState::Connected
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
impl FreshAttendedMotionCancellation for ConsoleAwareFreshAttendedMotionCancellation<'_> {
    fn is_cancelled(&self) -> bool {
        let snapshot = self.console.snapshot();
        fresh_motion_attestation_must_cancel(
            self.local.load(Ordering::Acquire),
            self.process_running.load(Ordering::SeqCst),
            &snapshot,
        )
    }
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
trait FreshAttendedMotionTerminal {
    fn is_terminal(&self) -> bool;
    fn discard_pending_input(&mut self) -> std::io::Result<()>;
    fn write_prompt(&mut self, prompt: &str) -> std::io::Result<()>;
    fn read_bounded_line(
        &mut self,
        cancellation: &dyn FreshAttendedMotionCancellation,
    ) -> Result<String, AttendedWheelsOffAttestationError>;
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix, test))]
struct BufferedFreshAttendedMotionTerminal<'a, R, W> {
    input: &'a mut R,
    output: &'a mut W,
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix, test))]
impl<R: BufRead, W: Write> FreshAttendedMotionTerminal
    for BufferedFreshAttendedMotionTerminal<'_, R, W>
{
    fn is_terminal(&self) -> bool {
        true
    }

    fn discard_pending_input(&mut self) -> std::io::Result<()> {
        Ok(())
    }

    fn write_prompt(&mut self, prompt: &str) -> std::io::Result<()> {
        self.output
            .write_all(prompt.as_bytes())
            .and_then(|()| self.output.flush())
    }

    fn read_bounded_line(
        &mut self,
        cancellation: &dyn FreshAttendedMotionCancellation,
    ) -> Result<String, AttendedWheelsOffAttestationError> {
        if cancellation.is_cancelled() {
            return Err(AttendedWheelsOffAttestationError::Interrupted);
        }
        read_bounded_tty_line(self.input)
    }
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
struct RealFreshAttendedMotionTerminal {
    file: File,
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
impl RealFreshAttendedMotionTerminal {
    fn open() -> Result<Self, AttendedWheelsOffAttestationError> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .open(QUALIFICATION_ATTESTATION_TTY)
            .map_err(AttendedWheelsOffAttestationError::OpenControllingTty)?;
        Ok(Self { file })
    }
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
impl FreshAttendedMotionTerminal for RealFreshAttendedMotionTerminal {
    fn is_terminal(&self) -> bool {
        self.file.is_terminal()
    }

    fn discard_pending_input(&mut self) -> std::io::Result<()> {
        rustix::termios::tcflush(&self.file, rustix::termios::QueueSelector::IFlush)
            .map_err(|source| std::io::Error::from_raw_os_error(source.raw_os_error()))
    }

    fn write_prompt(&mut self, prompt: &str) -> std::io::Result<()> {
        self.file
            .write_all(prompt.as_bytes())
            .and_then(|()| self.file.flush())
    }

    fn read_bounded_line(
        &mut self,
        cancellation: &dyn FreshAttendedMotionCancellation,
    ) -> Result<String, AttendedWheelsOffAttestationError> {
        let mut output = Vec::with_capacity(64);
        loop {
            if cancellation.is_cancelled() {
                return Err(AttendedWheelsOffAttestationError::Interrupted);
            }
            let mut byte = [0_u8; 1];
            match self.file.read(&mut byte) {
                Ok(0) => return Err(AttendedWheelsOffAttestationError::EndOfInput),
                Ok(1) if byte[0] == b'\n' => {
                    if output.last() == Some(&b'\r') {
                        output.pop();
                    }
                    return String::from_utf8(output)
                        .map_err(|_| AttendedWheelsOffAttestationError::InvalidUtf8);
                }
                Ok(1) => {
                    if output.len() == MAX_QUALIFICATION_ATTESTATION_LINE_BYTES {
                        return Err(AttendedWheelsOffAttestationError::LineTooLong {
                            maximum_bytes: MAX_QUALIFICATION_ATTESTATION_LINE_BYTES,
                        });
                    }
                    output.push(byte[0]);
                }
                Ok(_) => unreachable!("one-byte reads return at most one byte"),
                Err(source) if source.kind() == std::io::ErrorKind::Interrupted => {}
                Err(source) if source.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(QUALIFICATION_ATTESTATION_POLL_SLICE);
                }
                Err(source) => return Err(AttendedWheelsOffAttestationError::Input(source)),
            }
        }
    }
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
fn prompt_fresh_attended_motion_phrase(
    terminal: &mut impl FreshAttendedMotionTerminal,
    challenges: &mut impl WheelsOffAttestationChallengeSource,
    cancellation: &dyn FreshAttendedMotionCancellation,
    explanation: &str,
    phrase: &'static str,
) -> Result<(), AttendedWheelsOffAttestationError> {
    if cancellation.is_cancelled() {
        return Err(AttendedWheelsOffAttestationError::Interrupted);
    }
    terminal
        .discard_pending_input()
        .map_err(AttendedWheelsOffAttestationError::DiscardPendingInput)?;
    let challenge = challenges.next_challenge()?;
    if cancellation.is_cancelled() {
        return Err(AttendedWheelsOffAttestationError::Interrupted);
    }
    let expected = format!("{phrase} {}", lower_hex_qualification_challenge(&challenge));
    let prompt = format!("{explanation}\nType {expected:?}: ");
    if cancellation.is_cancelled() {
        return Err(AttendedWheelsOffAttestationError::Interrupted);
    }
    terminal
        .write_prompt(&prompt)
        .map_err(AttendedWheelsOffAttestationError::Output)?;
    let actual = terminal.read_bounded_line(cancellation)?;
    if actual != expected {
        return Err(AttendedWheelsOffAttestationError::PhraseMismatch { expected });
    }
    Ok(())
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
fn run_fresh_attended_motion_attestation(
    preflight: AttendedWheelsOffPreflight,
    terminal: &mut impl FreshAttendedMotionTerminal,
    challenges: &mut impl WheelsOffAttestationChallengeSource,
    cancellation: &dyn FreshAttendedMotionCancellation,
) -> Result<
    kiko_slam::navigation::OperatorClaimedWheelsOffAttestation,
    FreshAttendedMotionAttestationError,
> {
    if !terminal.is_terminal() {
        return Err(FreshAttendedMotionAttestationError::Terminal(
            AttendedWheelsOffAttestationError::TtyRequired,
        ));
    }
    let AttendedWheelsOffPreflight {
        motor_power_disconnected: _initial_motor_power_disconnected,
    } = preflight;
    prompt_fresh_attended_motion_phrase(
        terminal,
        challenges,
        cancellation,
        "The drive controller has exact applied-zero and disarm receipts. With nonzero authority still locked, one stopped runtime tick had fresh visual/depth/IMU observations, ready accessory health, a published occupancy revision, coordinator motion-start readiness, and a navigation diagnostic accepted by the bounded visualization queue. This does not claim that Rerun consumed or displayed that diagnostic. Confirm that motor power remained physically disconnected throughout device acquisition, zero/disarm, and this stopped runtime-readiness boundary.",
        "MOTOR POWER REMAINED PHYSICALLY DISCONNECTED THROUGH SETUP",
    )
    .map_err(FreshAttendedMotionAttestationError::Terminal)?;
    prompt_fresh_attended_motion_phrase(
        terminal,
        challenges,
        cancellation,
        "Only now physically reconnect motor power. Keep both wheels removed, keep the head supported, and keep the independent power cut immediately reachable throughout the bounded motion window.",
        "MOTOR POWER RECONNECTED WHEELS OFF HEAD SUPPORTED POWER CUT READY",
    )
    .map_err(FreshAttendedMotionAttestationError::Terminal)?;
    kiko_slam::navigation::OperatorClaimedWheelsOffAttestation::try_new(
        true,
        true,
        true,
        true,
        true,
        Instant::now(),
    )
    .map_err(FreshAttendedMotionAttestationError::Domain)
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix, test))]
fn read_fresh_attended_motion_attestation(
    preflight: AttendedWheelsOffPreflight,
    input: &mut impl BufRead,
    output: &mut impl Write,
    challenges: &mut impl WheelsOffAttestationChallengeSource,
) -> Result<
    kiko_slam::navigation::OperatorClaimedWheelsOffAttestation,
    FreshAttendedMotionAttestationError,
> {
    let mut terminal = BufferedFreshAttendedMotionTerminal { input, output };
    let cancellation = AtomicBool::new(false);
    run_fresh_attended_motion_attestation(preflight, &mut terminal, challenges, &cancellation)
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
fn run_real_fresh_attended_motion_attestation(
    preflight: AttendedWheelsOffPreflight,
    cancellation: &dyn FreshAttendedMotionCancellation,
) -> Result<
    kiko_slam::navigation::OperatorClaimedWheelsOffAttestation,
    FreshAttendedMotionAttestationError,
> {
    let mut terminal = RealFreshAttendedMotionTerminal::open()
        .map_err(FreshAttendedMotionAttestationError::Terminal)?;
    let mut challenges = OsWheelsOffAttestationChallengeSource;
    run_fresh_attended_motion_attestation(preflight, &mut terminal, &mut challenges, cancellation)
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
fn read_post_run_motor_power_disconnected(
    input: &mut impl BufRead,
    output: &mut impl Write,
    challenges: &mut impl WheelsOffAttestationChallengeSource,
) -> Result<(), AttendedWheelsOffAttestationError> {
    prompt_fresh_attended_phrase(
        input,
        output,
        challenges,
        "Software qualification has ended. Regardless of whether a controller owner started or its cleanup proved a stop, physically disconnect motor power now, or confirm that it was never reconnected. Do not leave this foreground qualification until motor power is physically disconnected.",
        "MOTOR POWER PHYSICALLY DISCONNECTED",
    )
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
fn require_post_run_motor_power_disconnected() -> Result<(), AttendedWheelsOffAttestationError> {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    if !stdin.is_terminal() || !stdout.is_terminal() {
        return Err(AttendedWheelsOffAttestationError::TtyRequired);
    }
    let mut challenges = OsWheelsOffAttestationChallengeSource;
    read_post_run_motor_power_disconnected(&mut stdin.lock(), &mut stdout.lock(), &mut challenges)
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
fn finish_attended_wheels_off_qualification(
    operation: Result<(), Box<dyn std::error::Error>>,
    motor_power_disconnect: Result<(), AttendedWheelsOffAttestationError>,
) -> Result<(), Box<dyn std::error::Error>> {
    match (operation, motor_power_disconnect) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(operation), Ok(())) => Err(operation),
        (Ok(()), Err(motor_power_disconnect)) => Err(Box::new(motor_power_disconnect)),
        (Err(operation), Err(motor_power_disconnect)) => Err(Box::new(
            WheelsOffQualificationAndMotorPowerDisconnectError {
                operation,
                motor_power_disconnect,
            },
        )),
    }
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
fn prepare_nano_wheels_off_qualification_live_session(
    bootstrap: kiko_slam::navigation::PreparedNanoWheelsOffQualificationBootstrap,
    preflight: AttendedWheelsOffPreflight,
    stream_epoch: StreamEpochId,
    capture_clock_origin: Instant,
    running: &AtomicBool,
) -> Result<PreparedLiveSession, Box<dyn std::error::Error>> {
    let oak_provenance =
        OakRuntimeProvenance::from_nano_wheels_off_qualification_bootstrap(&bootstrap);
    let kiko_slam::navigation::PreparedNanoWheelsOffQualificationBootstrap {
        roots,
        launch,
        assets,
        manifest: _,
        policy,
        head_gaze_policy,
        calibration: _,
        plant: _,
        artifact_hashes: _,
        accessory_evidence: _,
        oak_connected_identity: _,
        oak_usb_transport: _,
        depthai_build_metadata: _,
        stereo,
        live,
        exact_inventory_admission: _,
        candidate_limits,
        candidate_runtime_service_interval,
        fault_injection,
        initial_zero,
        initial_stop,
        stopped_controller,
        oak,
    } = bootstrap;
    let mut resources = NanoQualificationPreOwnerResources::new(
        stopped_controller,
        initial_zero,
        initial_stop,
        candidate_limits,
        candidate_runtime_service_interval,
        oak,
    );

    // The candidate controller is already exactly stopped. Natural head hold
    // and expression ownership are established before the expensive common
    // model stack; every later failure retains the bootstrap stop evidence.
    let accessory_config = (|| -> Result<NanoAccessoryWorkerConfig, Box<dyn std::error::Error>> {
        let health_period = NanoAccessoryHealthPeriod::try_from_duration(Duration::from_secs(1))?;
        let config = NanoAccessoryWorkerConfig::from_manifest_bound_policy(
            &policy,
            stream_epoch,
            health_period,
        )?;
        match head_gaze_policy {
            Some(policy) => {
                Ok(config.with_proposal_only_head_gaze_diagnostics(policy.into_policy())?)
            }
            None => Ok(config),
        }
    })();
    let accessory_config = match accessory_config {
        Ok(config) => config,
        Err(primary) => return resources.fail_box(primary),
    };
    resources.accessory = match NanoAccessoryWorker::start_with_loaded_face_perception(
        accessory_config,
        assets.frontal_face_cascade,
        assets.profile_face_cascade,
    ) {
        Ok(accessory) => Some(accessory),
        Err(source) => {
            return resources.fail_box(Box::new(LiveAccessoryError::Start(source)));
        }
    };

    let software = prepare_nano_common_live_software(
        NanoCommonLiveSoftwareInput {
            roots: &roots,
            graph: launch.launch().oak(),
            storage: launch.launch().storage(),
            occupancy: launch.launch().occupancy(),
            inference_policy: launch.launch().inference(),
            rerun: launch.launch().rerun(),
            onnx_runtime: NanoOrtRuntimeInput::Pinned(assets.pinned_onnx_runtime),
            superpoint_model: assets.superpoint_model.bytes(),
            lightglue_model: assets.lightglue_model.bytes(),
            stereo,
            live,
            map_persistence: policy.map_persistence(),
            stream_epoch,
        },
        || {
            require_qualification_pre_owner_accessory_healthy(&resources, running)
                .map_err(|source| Box::new(source) as Box<dyn std::error::Error>)
        },
    );
    let software = match software {
        Ok(software) => software,
        Err(primary) => return resources.fail_box(primary),
    };

    let operator_console = policy.control().operator_console();
    let profile = kiko_slam::navigation::WheelsOffQualificationControlProfile::parse(
        candidate_limits.effective_max_abs_pwm_percent(),
        candidate_limits.manual_test_magnitude_timer_pwm_percent(),
        u64::try_from(candidate_limits.manual_deadman().as_millis())?,
    )?;
    let frontend_config = kiko_slam::navigation::WheelsOffQualificationFrontendConfig::parse(
        operator_console.bind_address(),
        operator_console.capability_path().as_path().to_path_buf(),
        AgentControlMonotonicOrigin::new(
            capture_clock_origin,
            HostMonotonicTimestamp::from_nanos(0),
        ),
        operator_console.deadman_tick(),
    )?;
    let initial_health = ConsoleSubsystemHealth {
        stm32: Some(ConsoleHealth::Ready),
        head: Some(
            if policy
                .head()
                .return_to_natural_and_hold_continuously()
                .is_some()
            {
                ConsoleHealth::Ready
            } else {
                ConsoleHealth::Unavailable
            },
        ),
        eyes: Some(if policy.eye().static_runtime().is_some() {
            ConsoleHealth::Ready
        } else {
            ConsoleHealth::Unavailable
        }),
        oak: Some(ConsoleHealth::Degraded),
        slam: Some(ConsoleHealth::Degraded),
    };
    let accessory_health = resources
        .accessory
        .as_ref()
        .expect("successful accessory startup retains its sole owner")
        .health_observer();
    let (
        stopped_controller,
        initial_zero,
        initial_stop,
        limits,
        runtime_service_interval,
        accessory,
        device,
    ) = resources.into_parts();
    debug_assert_eq!(limits, candidate_limits);

    Ok(PreparedLiveSession {
        device,
        device_session: software.device_session,
        mono_config: software.mono_config,
        depth_config: Some(software.depth_config),
        imu_config: Some(software.imu_config),
        depth_queue_capacity: Some(software.queue_capacity),
        depth_ring_capacity: DepthRingCapacity::from_queue_capacity(software.queue_capacity),
        imu_session: Some(software.device_session),
        imu_queue_capacity: Some(software.queue_capacity),
        dense_requested: true,
        dense_data_capacity: software.queue_capacity,
        dense_control_capacity: software.dense_control_capacity,
        pairing_window: software.pairing_window,
        pairer: software.pairer,
        calibration: software.calibration,
        rectified_left_intrinsics: software.rectified_left_intrinsics,
        prepared_navigation_runtime: Some(software.navigation),
        inference: software.inference,
        tracker_initialization: PreparedTrackerInitialization::CanonicalNano,
        pair_queue_depth: software.queue_size,
        viz_queue_depth: software.queue_size,
        rerun_decimation: software.rerun_decimation,
        rerun_finish_timeout: software.rerun_finish_timeout,
        rerun_target: software.rerun_target,
        oak_provenance,
        motion: PreparedLiveMotionSelection::WheelsOffQualification(Box::new(
            LiveWheelsOffQualificationMotionInput::new(
                stopped_controller,
                initial_zero,
                initial_stop,
                limits,
                runtime_service_interval,
                preflight,
                profile,
                frontend_config,
                initial_health,
                accessory_health,
                software.rerun_diagnostics_url,
                fault_injection,
            ),
        )),
        accessory: Some(accessory),
        production_state: Some(software.production_state),
        warm_start_replay: software.warm_start_replay,
        systemd_supervision: None,
    })
}

#[cfg(all(feature = "nano-attended-navigation-trial", unix))]
#[derive(Debug)]
struct NanoAttendedTrialLivePreparationError {
    primary: Box<dyn std::error::Error>,
    controller_shutdown: Option<Box<dyn std::error::Error>>,
    accessory_shutdown: NanoPreOwnerAccessoryShutdown,
    oak_close: NanoPreOwnerOakClose,
}

#[cfg(all(feature = "nano-attended-navigation-trial", unix))]
impl std::fmt::Display for NanoAttendedTrialLivePreparationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "Nano attended navigation preparation failed: {}",
            self.primary
        )?;
        if let Some(source) = self.controller_shutdown.as_ref() {
            write!(
                formatter,
                "; controller-owner shutdown also failed: {source}"
            )?;
        }
        write!(
            formatter,
            "; {}; OAK close: {:?}",
            self.accessory_shutdown, self.oak_close
        )
    }
}

#[cfg(all(feature = "nano-attended-navigation-trial", unix))]
impl std::error::Error for NanoAttendedTrialLivePreparationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.primary.as_ref())
    }
}

#[cfg(all(feature = "nano-attended-navigation-trial", unix))]
fn fail_attended_trial_hardware<T>(
    primary: Box<dyn std::error::Error>,
    controller_shutdown: Option<Box<dyn std::error::Error>>,
    accessory: NanoAccessoryWorker,
    oak: Device,
) -> Result<T, Box<dyn std::error::Error>> {
    // No controller is active on the ordinary preparation failures. When a
    // controller owner is supplied, its shutdown has already completed before
    // this function is called. Close OAK before releasing the held head/eyes.
    let oak_close = match oak.close() {
        Ok(()) => NanoPreOwnerOakClose::Confirmed,
        Err(source) => NanoPreOwnerOakClose::Uncertain(source),
    };
    let accessory_shutdown = shutdown_nano_pre_owner_accessory(accessory, false);
    Err(Box::new(NanoAttendedTrialLivePreparationError {
        primary,
        controller_shutdown,
        accessory_shutdown,
        oak_close,
    }))
}

#[cfg(all(feature = "nano-attended-navigation-trial", unix))]
fn fail_attended_trial_after_controller<T>(
    primary: Box<dyn std::error::Error>,
    controller: OwnedNanoAttendedNavigationTrialController,
    async_runtime: &tokio::runtime::Runtime,
    accessory: NanoAccessoryWorker,
    oak: Device,
) -> Result<T, Box<dyn std::error::Error>> {
    let controller_shutdown = async_runtime
        .block_on(controller.shutdown_controller())
        .err()
        .map(|source| Box::new(source) as Box<dyn std::error::Error>);
    fail_attended_trial_hardware(primary, controller_shutdown, accessory, oak)
}

#[cfg(all(feature = "nano-attended-navigation-trial", unix))]
#[derive(Clone, Copy)]
struct NanoAttendedTrialEpochs {
    stream: StreamEpochId,
    capture_origin: Instant,
    navigation: NavigationClockEpoch,
    readiness: ReadinessEpoch,
}

#[cfg(all(feature = "nano-attended-navigation-trial", unix))]
fn prepare_nano_attended_navigation_trial_live_session(
    roots: NanoBootstrapRoots,
    prepared: kiko_slam::navigation::nano_base_commissioning_bootstrap::PreparedNanoBaseCommissioning,
    hardware: PreparedAttendedNavigationTrialLiveHardware,
    epochs: NanoAttendedTrialEpochs,
    running: &Arc<AtomicBool>,
    async_runtime: &tokio::runtime::Runtime,
) -> Result<
    (
        PreparedLiveSession,
        OwnedNanoAttendedNavigationTrialController,
    ),
    Box<dyn std::error::Error>,
> {
    let PreparedAttendedNavigationTrialLiveHardware {
        stream,
        stereo,
        live,
        accessory,
        oak,
    } = hardware;
    let oak_provenance = match inspect_oak_runtime(&oak, "attended navigation") {
        Ok(value) => value,
        Err(source) => {
            return fail_attended_trial_hardware(Box::new(source), None, accessory, oak);
        }
    };
    let inputs = prepared.loaded_inputs();
    let live_graph = prepared.live_graph();
    let software = prepare_nano_common_live_software(
        NanoCommonLiveSoftwareInput {
            roots: &roots,
            graph: live_graph.oak(),
            storage: live_graph.storage(),
            occupancy: live_graph.occupancy(),
            inference_policy: live_graph.inference(),
            rerun: live_graph.rerun(),
            onnx_runtime: NanoOrtRuntimeInput::RetainedBytes(inputs.onnx_runtime_library.bytes()),
            superpoint_model: inputs.superpoint_model.bytes(),
            lightglue_model: inputs.lightglue_model.bytes(),
            stereo,
            live,
            map_persistence: prepared.accessory_policy().map_persistence(),
            stream_epoch: epochs.stream,
        },
        || {
            if !running.load(Ordering::Acquire) {
                return Err(Box::new(NanoLivePreparationInterrupted) as Box<dyn std::error::Error>);
            }
            match accessory.try_terminal_fault() {
                Ok(None) => Ok(()),
                Ok(Some(fault)) => Err(Box::new(LiveAccessoryError::TerminalFault(fault))
                    as Box<dyn std::error::Error>),
                Err(source) => Err(Box::new(LiveAccessoryError::FaultMonitor(source))
                    as Box<dyn std::error::Error>),
            }
        },
    );
    let software = match software {
        Ok(value) => value,
        Err(primary) => {
            return fail_attended_trial_hardware(primary, None, accessory, oak);
        }
    };

    let admitted = match prepared.consume_fresh_attended_attestation(
        stream,
        epochs.capture_origin,
        running.as_ref(),
    ) {
        Ok(value) => value,
        Err(source) => {
            return fail_attended_trial_hardware(Box::new(source), None, accessory, oak);
        }
    };
    let mut controller = match async_runtime
        .block_on(admitted.start_attended_navigation_trial_controller(epochs.capture_origin))
    {
        Ok(value) => value,
        Err(source) => {
            return fail_attended_trial_hardware(Box::new(source), None, accessory, oak);
        }
    };
    let inventory_transition_at = match host_monotonic_since(epochs.capture_origin) {
        Ok(timestamp) => timestamp,
        Err(source) => {
            return fail_attended_trial_after_controller(
                Box::new(source),
                controller,
                async_runtime,
                accessory,
                oak,
            );
        }
    };
    let readiness_transition_at = match host_monotonic_since(epochs.capture_origin) {
        Ok(timestamp) => timestamp,
        Err(source) => {
            return fail_attended_trial_after_controller(
                Box::new(source),
                controller,
                async_runtime,
                accessory,
                oak,
            );
        }
    };
    let admission = match controller.take_live_motion_admission(
        epochs.navigation,
        epochs.readiness,
        inventory_transition_at,
        readiness_transition_at,
    ) {
        Ok(value) => value,
        Err(source) => {
            return fail_attended_trial_after_controller(
                Box::new(source),
                controller,
                async_runtime,
                accessory,
                oak,
            );
        }
    };

    let accessory_health = accessory.health_observer();
    Ok((
        PreparedLiveSession {
            device: oak,
            device_session: software.device_session,
            mono_config: software.mono_config,
            depth_config: Some(software.depth_config),
            imu_config: Some(software.imu_config),
            depth_queue_capacity: Some(software.queue_capacity),
            depth_ring_capacity: DepthRingCapacity::from_queue_capacity(software.queue_capacity),
            imu_session: Some(software.device_session),
            imu_queue_capacity: Some(software.queue_capacity),
            dense_requested: true,
            dense_data_capacity: software.queue_capacity,
            dense_control_capacity: software.dense_control_capacity,
            pairing_window: software.pairing_window,
            pairer: software.pairer,
            calibration: software.calibration,
            rectified_left_intrinsics: software.rectified_left_intrinsics,
            prepared_navigation_runtime: Some(software.navigation),
            inference: software.inference,
            tracker_initialization: PreparedTrackerInitialization::CanonicalNano,
            pair_queue_depth: software.queue_size,
            viz_queue_depth: software.queue_size,
            rerun_decimation: software.rerun_decimation,
            rerun_finish_timeout: software.rerun_finish_timeout,
            rerun_target: software.rerun_target,
            oak_provenance,
            motion: PreparedLiveMotionSelection::AttendedNavigationTrial(Box::new(
                LiveAttendedNavigationTrialMotionInput::new(
                    admission,
                    accessory_health,
                    software.rerun_diagnostics_url,
                ),
            )),
            accessory: Some(accessory),
            production_state: Some(software.production_state),
            warm_start_replay: software.warm_start_replay,
            systemd_supervision: None,
        },
        controller,
    ))
}

#[cfg(all(feature = "nano-agent", unix))]
fn prepare_nano_live_session(
    bootstrap: PreparedNanoBootstrap,
    stream_epoch: StreamEpochId,
    systemd_supervision: NanoSystemdServiceSupervision,
    running: &AtomicBool,
) -> Result<PreparedLiveSession, Box<dyn std::error::Error>> {
    let oak_provenance = OakRuntimeProvenance::from_nano_bootstrap(&bootstrap);
    let PreparedNanoBootstrap {
        roots,
        launch,
        assets,
        accessory_evidence: _,
        oak_connected_identity: _,
        oak_usb_transport: _,
        depthai_build_metadata: _,
        calibration: _,
        stereo,
        live,
        runtime,
        accessory,
        oak,
    } = bootstrap;
    let resources = NanoPreOwnerResources::new(runtime, accessory, oak);

    let map_persistence = resources
        .runtime
        .as_ref()
        .expect("pre-owner runtime remains present during map admission")
        .startup()
        .policy
        .map_persistence();
    let software = prepare_nano_common_live_software(
        NanoCommonLiveSoftwareInput {
            roots: &roots,
            graph: launch.launch().oak(),
            storage: launch.launch().storage(),
            occupancy: launch.launch().occupancy(),
            inference_policy: launch.launch().inference(),
            rerun: launch.launch().rerun(),
            onnx_runtime: NanoOrtRuntimeInput::RetainedBytes(assets.onnx_runtime_library.bytes()),
            superpoint_model: assets.superpoint_model.bytes(),
            lightglue_model: assets.lightglue_model.bytes(),
            stereo,
            live,
            map_persistence,
            stream_epoch,
        },
        || {
            require_pre_owner_accessory_healthy(&resources, running)
                .map_err(|source| Box::new(source) as Box<dyn std::error::Error>)
        },
    );
    let software = match software {
        Ok(software) => software,
        Err(primary) => return resources.fail_box(primary),
    };

    let accessory_health = resources
        .accessory
        .as_ref()
        .expect("successful accessory startup retains its sole owner")
        .health_observer();
    let accessory_liveness = resources
        .accessory
        .as_ref()
        .expect("successful accessory startup retains its sole owner")
        .loop_liveness_observer();
    let (runtime, accessory, device) = resources.into_parts();
    Ok(PreparedLiveSession {
        device,
        device_session: software.device_session,
        mono_config: software.mono_config,
        depth_config: Some(software.depth_config),
        imu_config: Some(software.imu_config),
        depth_queue_capacity: Some(software.queue_capacity),
        depth_ring_capacity: DepthRingCapacity::from_queue_capacity(software.queue_capacity),
        imu_session: Some(software.device_session),
        imu_queue_capacity: Some(software.queue_capacity),
        dense_requested: true,
        dense_data_capacity: software.queue_capacity,
        dense_control_capacity: software.dense_control_capacity,
        pairing_window: software.pairing_window,
        pairer: software.pairer,
        calibration: software.calibration,
        rectified_left_intrinsics: software.rectified_left_intrinsics,
        prepared_navigation_runtime: Some(software.navigation),
        inference: software.inference,
        tracker_initialization: PreparedTrackerInitialization::CanonicalNano,
        pair_queue_depth: software.queue_size,
        viz_queue_depth: software.queue_size,
        rerun_decimation: software.rerun_decimation,
        rerun_finish_timeout: software.rerun_finish_timeout,
        rerun_target: software.rerun_target,
        oak_provenance,
        motion: PreparedLiveMotionSelection::Production(Box::new(
            LiveProductionMotionInput::from_admitted(
                runtime,
                accessory_health,
                software.rerun_diagnostics_url,
            ),
        )),
        accessory: Some(accessory),
        production_state: Some(software.production_state),
        warm_start_replay: software.warm_start_replay,
        systemd_supervision: Some(systemd_supervision.bind(accessory_liveness)),
    })
}

#[cfg(all(feature = "nano-agent", unix))]
struct NanoControllerOwnerExitGuard {
    running: Arc<AtomicBool>,
}

#[cfg(all(feature = "nano-agent", unix))]
impl NanoControllerOwnerExitGuard {
    fn new(running: Arc<AtomicBool>) -> Self {
        Self { running }
    }
}

#[cfg(all(feature = "nano-agent", unix))]
impl Drop for NanoControllerOwnerExitGuard {
    fn drop(&mut self) {
        // This guard also runs while unwinding a task panic. No controller
        // owner exit—clean, failed, or panicked—may leave the rest of the
        // production process believing physical ownership is still healthy.
        self.running.store(false, Ordering::SeqCst);
    }
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
fn run_nano_wheels_off_qualification(
    args: NanoWheelsOffQualificationArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    let running = install_live_shutdown_handler()?;
    if let Some(fault) = args.fault_injection {
        eprintln!(
            "wheels-off qualification fault session selected: {fault}; one-shot trigger is the first nonzero candidate command"
        );
    }
    // No device, serial port, listener, or controller owner is opened before
    // the attended physical preflight succeeds.
    let preflight = require_attended_wheels_off_preflight()?;
    // Every return after this point—successful or failed—reaches the final
    // attended physical-disconnect confirmation. The closure scope also makes
    // all bootstrap/runtime owners drop before that final prompt is issued.
    let operation = (|| -> Result<(), Box<dyn std::error::Error>> {
        let capture_clock_origin = Instant::now();
        let navigation_clock_epoch =
            NavigationClockEpoch::new(HostMonotonicTimestamp::from_nanos(0));
        let stream_epoch = fresh_nano_stream_epoch()?;
        let request = kiko_slam::navigation::QualificationBootstrapRequest::try_new(
            args.deployment_root,
            args.state_root,
            args.launch_config,
            capture_clock_origin,
            args.fault_injection,
            running.as_ref(),
        )?;
        let async_runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()?;
        let owned_bootstrap = async_runtime
            .block_on(kiko_slam::navigation::bootstrap_nano_wheels_off_qualification(request))?;
        let (bootstrap, controller_owner, controller_owner_shutdown_timeout) =
            owned_bootstrap.into_parts();
        let (request_controller_shutdown, controller_shutdown) = tokio::sync::oneshot::channel();
        let controller_running = Arc::clone(&running);
        let controller_task = async_runtime.spawn(async move {
            let _exit_guard = NanoControllerOwnerExitGuard::new(controller_running);
            controller_owner
                .run_until_shutdown(controller_shutdown, controller_owner_shutdown_timeout)
                .await
        });

        let operation = prepare_nano_wheels_off_qualification_live_session(
            bootstrap,
            preflight,
            stream_epoch,
            capture_clock_origin,
            running.as_ref(),
        )
        .and_then(|prepared| {
            run_prepared_live_session(
                prepared,
                running,
                capture_clock_origin,
                navigation_clock_epoch,
            )
        });
        let _ = request_controller_shutdown.send(());
        let controller = async_runtime.block_on(controller_task);
        drop(async_runtime);
        finish_nano_controller_owner(operation, controller)
    })();
    finish_attended_wheels_off_qualification(operation, require_post_run_motor_power_disconnected())
}

#[cfg(all(feature = "nano-attended-navigation-trial", unix))]
#[derive(Debug)]
struct NanoAttendedTrialOperationAndShutdownError {
    operation: Box<dyn std::error::Error>,
    shutdown: kiko_slam::navigation::nano_base_commissioning_bootstrap::AttendedTrialControllerShutdownError,
}

#[cfg(all(feature = "nano-attended-navigation-trial", unix))]
impl std::fmt::Display for NanoAttendedTrialOperationAndShutdownError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "attended navigation runtime failed: {}; controller-owner shutdown also failed: {}",
            self.operation, self.shutdown
        )
    }
}

#[cfg(all(feature = "nano-attended-navigation-trial", unix))]
impl std::error::Error for NanoAttendedTrialOperationAndShutdownError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.operation.as_ref())
    }
}

#[cfg(all(feature = "nano-attended-navigation-trial", unix))]
fn run_nano_attended_navigation_trial(
    args: NanoAttendedNavigationTrialArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    let running = install_live_shutdown_handler()?;
    let capture_clock_origin = Instant::now();
    let navigation_clock_epoch = NavigationClockEpoch::new(HostMonotonicTimestamp::from_nanos(0));
    let readiness_epoch = ReadinessEpoch::try_new(1)?;
    let stream_epoch = fresh_nano_stream_epoch()?;
    let roots = NanoBootstrapRoots::try_new(args.deployment_root.clone(), args.state_root.clone())?;
    let launch = ArtifactRelativePath::parse(args.launch_config)?;
    let mut prepared =
        prepare_nano_base_commissioning(&args.deployment_root, launch, &args.state_root)?;
    let mut clock_epoch_bytes = [0_u8; 16];
    getrandom::fill(&mut clock_epoch_bytes)?;
    let commissioning_clock_epoch = CommissioningClockEpoch::try_new(clock_epoch_bytes)?;
    let async_runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()?;
    let hardware = prepare_attended_navigation_trial_live_hardware(
        &mut prepared,
        Arc::clone(&running),
        commissioning_clock_epoch,
        stream_epoch,
    )?;
    let (session, controller) = prepare_nano_attended_navigation_trial_live_session(
        roots,
        prepared,
        hardware,
        NanoAttendedTrialEpochs {
            stream: stream_epoch,
            capture_origin: capture_clock_origin,
            navigation: navigation_clock_epoch,
            readiness: readiness_epoch,
        },
        &running,
        &async_runtime,
    )?;
    eprintln!(
        "attended navigation trial ready: one OAK/accessory/STM32 owner, full SLAM and occupancy enabled, loopback console enabled, controller disarmed; Arm still requires a fresh applied zero and every command remains capped by the attended profile and attestation deadline"
    );
    let operation = run_prepared_live_session(
        session,
        running,
        capture_clock_origin,
        navigation_clock_epoch,
    );
    let shutdown = async_runtime.block_on(controller.shutdown_controller());
    drop(async_runtime);
    match (operation, shutdown) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(operation), Ok(())) => Err(operation),
        (Ok(()), Err(shutdown)) => Err(Box::new(shutdown)),
        (Err(operation), Err(shutdown)) => {
            Err(Box::new(NanoAttendedTrialOperationAndShutdownError {
                operation,
                shutdown,
            }))
        }
    }
}

#[cfg(all(feature = "nano-agent", unix))]
fn run_nano_agent(args: NanoAgentArgs) -> Result<(), Box<dyn std::error::Error>> {
    // Parse the weak process environment exactly once before any worker thread
    // or hardware owner exists. Manual foreground runs have all three systemd
    // variables absent and remain intentionally unsupervised.
    let systemd_supervision = NanoSystemdServiceSupervision::from_process_environment()?;
    let running = install_live_shutdown_handler()?;
    let capture_clock_origin = Instant::now();
    let navigation_clock_epoch = NavigationClockEpoch::new(HostMonotonicTimestamp::from_nanos(0));
    let readiness_epoch = ReadinessEpoch::try_new(1)?;
    let stream_epoch = fresh_nano_stream_epoch()?;
    let request = NanoBootstrapRequest::try_new(
        args.deployment_root,
        args.state_root,
        args.launch_config,
        stream_epoch,
        capture_clock_origin,
        navigation_clock_epoch,
        readiness_epoch,
        running.as_ref(),
    )?;
    // One dedicated async worker keeps the sole serial/UDP controller owner
    // progressing while the host SLAM pipeline runs on its bounded native
    // worker threads.
    let async_runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()?;
    let owned_bootstrap = async_runtime.block_on(bootstrap_nano_production(request))?;
    let (bootstrap, controller_owner, controller_owner_shutdown_timeout) =
        owned_bootstrap.into_parts();
    let (request_controller_shutdown, controller_shutdown) = tokio::sync::oneshot::channel();
    let controller_running = Arc::clone(&running);
    let controller_task = async_runtime.spawn(async move {
        let _exit_guard = NanoControllerOwnerExitGuard::new(controller_running);
        controller_owner
            .run_until_shutdown(controller_shutdown, controller_owner_shutdown_timeout)
            .await
    });

    let operation = prepare_nano_live_session(
        bootstrap,
        stream_epoch,
        systemd_supervision,
        running.as_ref(),
    )
    .and_then(|prepared| {
        run_prepared_live_session(
            prepared,
            running,
            capture_clock_origin,
            navigation_clock_epoch,
        )
    });
    let _ = request_controller_shutdown.send(());
    let controller = async_runtime.block_on(controller_task);
    drop(async_runtime);
    finish_nano_controller_owner(operation, controller)
}

#[cfg(all(feature = "nano-agent", unix))]
#[derive(Debug)]
enum NanoControllerOwnerRunError {
    Terminated(V2ControllerOwnerTerminationError),
    TaskJoin(tokio::task::JoinError),
}

#[cfg(all(feature = "nano-agent", unix))]
impl std::fmt::Display for NanoControllerOwnerRunError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Terminated(source) => source.fmt(formatter),
            Self::TaskJoin(source) => {
                write!(formatter, "controller-owner task join failed: {source}")
            }
        }
    }
}

#[cfg(all(feature = "nano-agent", unix))]
impl std::error::Error for NanoControllerOwnerRunError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Terminated(source) => Some(source),
            Self::TaskJoin(source) => Some(source),
        }
    }
}

#[cfg(all(feature = "nano-agent", unix))]
#[derive(Debug)]
struct NanoOperationAndControllerOwnerError {
    operation: Box<dyn std::error::Error>,
    controller: NanoControllerOwnerRunError,
}

#[cfg(all(feature = "nano-agent", unix))]
impl std::fmt::Display for NanoOperationAndControllerOwnerError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "Nano operation failed: {}; controller-owner termination also failed: {}",
            self.operation, self.controller
        )
    }
}

#[cfg(all(feature = "nano-agent", unix))]
impl std::error::Error for NanoOperationAndControllerOwnerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.operation.as_ref())
    }
}

#[cfg(all(feature = "nano-agent", unix))]
fn finish_nano_controller_owner(
    operation: Result<(), Box<dyn std::error::Error>>,
    controller: Result<Result<(), V2ControllerOwnerTerminationError>, tokio::task::JoinError>,
) -> Result<(), Box<dyn std::error::Error>> {
    let controller = match controller {
        Ok(Ok(())) => None,
        Ok(Err(source)) => Some(NanoControllerOwnerRunError::Terminated(source)),
        Err(source) => Some(NanoControllerOwnerRunError::TaskJoin(source)),
    };
    match (operation, controller) {
        (Ok(()), None) => Ok(()),
        (Err(operation), None) => Err(operation),
        (Ok(()), Some(controller)) => Err(Box::new(controller)),
        (Err(operation), Some(controller)) => Err(Box::new(NanoOperationAndControllerOwnerError {
            operation,
            controller,
        })),
    }
}

#[cfg(feature = "record")]
fn install_live_shutdown_handler() -> Result<Arc<AtomicBool>, ctrlc::Error> {
    let running = Arc::new(AtomicBool::new(true));
    let signal_running = Arc::clone(&running);
    ctrlc::set_handler(move || {
        eprintln!("\nreceived shutdown signal, stopping...");
        signal_running.store(false, Ordering::SeqCst);
    })?;
    Ok(running)
}

#[cfg(feature = "record")]
fn prepare_compatibility_live_session(
    args: LiveArgs,
    running: &AtomicBool,
) -> Result<PreparedLiveSession, Box<dyn std::error::Error>> {
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
    )?
    .load()?;

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
    LiveNavigationPrerequisites::new(
        depth_enabled,
        imu_config.is_some(),
        dense_requested,
        mono_config.rectified,
    )
    .require_for(navigation_request.request())?;
    // EEPROM IMU-to-camera evidence is persisted by an enabled navigation
    // dataset. A compatibility live session with navigation disabled neither
    // consumes nor persists it, so it must not gain a new EEPROM failure mode.
    let oak_eeprom_evidence_policy = if imu_config.is_some() && navigation_request.is_enabled() {
        OakEepromEvidencePolicy::Require
    } else {
        OakEepromEvidencePolicy::Omit
    };
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
        usb_transport: UsbTransportPolicy::super_speed_required(),
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
    let pairing_window = load_pairing_window()?;
    let pairer_max_pending = load_pairer_max_pending_per_side()?;
    let mut pairer = StereoPairer::new_with_max_pending(pairing_window, pairer_max_pending)?;
    let StereoBootstrap {
        calibration,
        rectified_left_intrinsics,
    } = bootstrap_stereo(
        &mut device,
        &mono_config,
        oak_eeprom_evidence_policy,
        running,
        &mut pairer,
    )?;
    let rectified = RectifiedStereo::from_calibration(&calibration)?;
    let runtime_depth_camera = DepthCameraModel::new(
        rectified.left(),
        rectified.dimensions(),
        DepthToTrackingCamera::identity(),
    );
    let prepared_navigation_runtime =
        prepare_live_navigation_runtime(navigation_request, runtime_depth_camera, device_session)?;
    let inference = InferenceConfig::from_args(&args.inference)?;
    let pair_queue_depth = env_usize("KIKO_LIVE_PAIR_QUEUE_DEPTH")?.unwrap_or(12);
    let viz_queue_depth = env_usize("KIKO_LIVE_VIZ_QUEUE_DEPTH")?.unwrap_or(12);
    let dense_data_capacity =
        ChannelCapacity::try_from(env_usize("KIKO_DENSE_DATA_QUEUE_DEPTH")?.unwrap_or(4))?;
    let dense_control_capacity =
        ChannelCapacity::try_from(env_usize("KIKO_DENSE_CTRL_QUEUE_DEPTH")?.unwrap_or(64))?;

    Ok(PreparedLiveSession {
        device,
        device_session,
        mono_config,
        depth_config,
        imu_config,
        depth_queue_capacity,
        depth_ring_capacity,
        imu_session,
        imu_queue_capacity,
        dense_requested,
        dense_data_capacity,
        dense_control_capacity,
        pairing_window,
        pairer,
        calibration,
        rectified_left_intrinsics,
        prepared_navigation_runtime,
        inference,
        tracker_initialization: PreparedTrackerInitialization::Environment,
        pair_queue_depth,
        viz_queue_depth,
        rerun_decimation: args.rerun_decimation.get(),
        rerun_finish_timeout: args.rerun_finish_timeout_ms.get(),
        rerun_target: LiveRerunTarget::Connect,
        oak_provenance,
        #[cfg(feature = "actuation")]
        motion: PreparedLiveMotionSelection::Compatibility,
        #[cfg(all(feature = "nano-agent", unix))]
        accessory: None,
        #[cfg(all(feature = "nano-agent", unix))]
        production_state: None,
        #[cfg(all(feature = "nano-agent", unix))]
        warm_start_replay: None,
        #[cfg(all(feature = "nano-agent", unix))]
        systemd_supervision: None,
    })
}

#[cfg(feature = "record")]
fn run_live(args: LiveArgs) -> Result<(), Box<dyn std::error::Error>> {
    let running = install_live_shutdown_handler()?;
    let prepared = prepare_compatibility_live_session(args, running.as_ref())?;
    let capture_clock_origin = Instant::now();
    let navigation_clock_epoch = NavigationClockEpoch::new(HostMonotonicTimestamp::from_nanos(0));
    run_prepared_live_session(
        prepared,
        running,
        capture_clock_origin,
        navigation_clock_epoch,
    )
}

#[cfg(feature = "record")]
fn run_prepared_live_session(
    prepared: PreparedLiveSession,
    running: Arc<AtomicBool>,
    capture_clock_origin: Instant,
    navigation_clock_epoch: NavigationClockEpoch,
) -> Result<(), Box<dyn std::error::Error>> {
    let PreparedLiveSession {
        mut device,
        device_session,
        mono_config,
        depth_config,
        imu_config,
        depth_queue_capacity,
        depth_ring_capacity,
        imu_session,
        imu_queue_capacity,
        dense_requested,
        dense_data_capacity,
        dense_control_capacity,
        pairing_window,
        mut pairer,
        calibration,
        rectified_left_intrinsics,
        mut prepared_navigation_runtime,
        inference,
        tracker_initialization,
        pair_queue_depth,
        viz_queue_depth,
        rerun_decimation,
        rerun_finish_timeout,
        rerun_target,
        oak_provenance,
        #[cfg(feature = "actuation")]
        motion,
        #[cfg(all(feature = "nano-agent", unix))]
        accessory,
        #[cfg(all(feature = "nano-agent", unix))]
        production_state,
        #[cfg(all(feature = "nano-agent", unix))]
        warm_start_replay,
        #[cfg(all(feature = "nano-agent", unix))]
        mut systemd_supervision,
    } = prepared;
    let depth_enabled = depth_config.is_some();
    let operation = (|| -> Result<(), Box<dyn std::error::Error>> {
        #[cfg(all(feature = "nano-agent", unix))]
        let mut nano_setup_guard = NanoLiveSetupGuard::new(motion, accessory, production_state);
        #[cfg(all(feature = "nano-agent", unix))]
        nano_setup_guard.bind_head_gaze_lease_if_configured()?;
        #[cfg(all(feature = "nano-agent", unix))]
        let face_stage_stats_handle = nano_setup_guard.face_perception_stage_stats_handle();
        #[cfg(all(feature = "nano-agent", unix))]
        let (face_viz_rx, face_viz_channel_stats) = match nano_setup_guard.take_face_diagnostics() {
            Some((receiver, stats)) => (Some(receiver), Some(stats)),
            None => (None, None),
        };
        #[cfg(all(feature = "nano-agent", unix))]
        let (face_viz_cancel_tx, face_viz_cancel_rx) = crossbeam_channel::bounded::<()>(0);
        #[cfg(all(feature = "nano-agent", unix))]
        let face_viz_counters = Arc::new(LiveFaceVizCounters::default());
        let rectified = RectifiedStereo::from_calibration(&calibration)?;
        let navigation_enabled = prepared_navigation_runtime.is_some();
        let pair_capacity = ChannelCapacity::try_from(pair_queue_depth)?;
        let (pair_tx, pair_rx, pair_stats) =
            bounded_channel::<StereoObservation>(pair_capacity, DropPolicy::DropOldest);

        let viz_capacity = ChannelCapacity::try_from(viz_queue_depth)?;
        let (viz_tx, viz_rx, viz_stats) = bounded_channel(viz_capacity, DropPolicy::DropNewest);
        let (rgb_viz_tx, rgb_viz_rx, rgb_viz_stats) =
            bounded_channel(ChannelCapacity::try_from(1_usize)?, DropPolicy::DropOldest);
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

        let inference_runtime = inference.runtime_evidence()?;
        eprintln!(
            "live inference providers: superpoint requested={:?} selected={:?}; lightglue requested={:?} selected={:?}",
            inference_runtime.superpoint_requested,
            inference_runtime.superpoint_selected,
            inference_runtime.lightglue_requested,
            inference_runtime.lightglue_selected,
        );
        let slam_telemetry = LiveSlamTelemetry::new(inference_runtime);
        let InferenceConfig {
            superpoint_left,
            superpoint_right,
            lightglue,
            key_limit,
            downscale,
            ..
        } = inference;

        let tracker_defaults = TrackerDefaults {
            min_keyframe_points: 80,
            refresh_inliers: 20,
            min_inliers: 15,
        };
        let tracker_config = match tracker_initialization {
            PreparedTrackerInitialization::Environment => {
                build_tracker_config(tracker_defaults, key_limit, downscale)?
            }
            #[cfg(all(feature = "nano-agent", unix))]
            PreparedTrackerInitialization::CanonicalNano => {
                build_canonical_nano_tracker_config(tracker_defaults, key_limit, downscale)?
            }
        };
        let mut navigation_occupancy_config = prepared_navigation_runtime
            .as_mut()
            .and_then(PreparedLiveNavigationRuntime::take_occupancy_config);
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
            Some((dense_data_capacity, dense_control_capacity))
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
        #[cfg(all(feature = "nano-agent", unix))]
        nano_setup_guard.require_accessory_healthy_if_present(running.as_ref())?;
        let tracker = match tracker_initialization {
            PreparedTrackerInitialization::Environment => SlamTracker::try_new(
                superpoint_left,
                superpoint_right,
                lightglue,
                rectified,
                tracker_config,
            )?,
            #[cfg(all(feature = "nano-agent", unix))]
            PreparedTrackerInitialization::CanonicalNano => {
                SlamTracker::try_new_with_runtime_policy(
                    superpoint_left,
                    superpoint_right,
                    lightglue,
                    rectified,
                    tracker_config,
                    TrackerRuntimePolicy::canonical_nano(),
                )?
            }
        };
        report_tracker_runtime(&tracker_config, &tracker);
        #[cfg(all(feature = "nano-agent", unix))]
        nano_setup_guard.require_accessory_healthy_if_present(running.as_ref())?;

        #[cfg(all(feature = "nano-agent", unix))]
        let (tracker, occupancy_startup, dense_generation, warm_relocalization_gate) =
            match warm_start_replay {
                Some(required) => {
                    let occupancy_config =
                        occupancy_config.ok_or(NanoWarmStartOccupancyUnavailable)?;
                    let parts = replay_nano_warm_start(
                        *required,
                        tracker,
                        occupancy_config,
                        NanoWarmStartReplayConfig::default(),
                    )?
                    .into_parts();
                    let kiko_slam::navigation::NanoWarmStartReplayRuntimeParts {
                        tracker,
                        occupancy,
                        dense_generation,
                        initial_snapshot,
                        relocalization_gate,
                        receipt,
                    } = parts;
                    eprintln!(
                        "warm start: exact replay matched occupancy={} dataset={} selected_map_epoch={} selected_map_revision={} stereo_pairs={} replay_events={} replay_map_corrections={} replay_map={:?}; live_localized=false; dataset_content_binding={}",
                        receipt.occupancy_snapshot_path().display(),
                        receipt.slam_dataset_directory_path().display(),
                        receipt.selected_map_epoch_id().as_u64(),
                        receipt.selected_map_revision(),
                        receipt.processed_stereo_pairs(),
                        receipt.replay_diagnostic_events(),
                        receipt.replay_map_corrections(),
                        receipt.final_replay_map(),
                        receipt.dataset_content_binding_status(),
                    );
                    (
                        tracker,
                        Some(LiveOccupancyWorkerStartup::ContinuedReplay {
                            runtime: Box::new(occupancy),
                            initial_snapshot,
                        }),
                        dense_generation,
                        Some(relocalization_gate),
                    )
                }
                None => (
                    tracker,
                    occupancy_config.map(LiveOccupancyWorkerStartup::Fresh),
                    command_mapper::DenseCommandGeneration::default(),
                    None,
                ),
            };
        #[cfg(all(feature = "nano-agent", unix))]
        nano_setup_guard.require_accessory_healthy_if_present(running.as_ref())?;
        #[cfg(not(all(feature = "nano-agent", unix)))]
        let (tracker, occupancy_startup, dense_generation) = (
            tracker,
            occupancy_config.map(LiveOccupancyWorkerStartup::Fresh),
            command_mapper::DenseCommandGeneration::default(),
        );

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
                    bounded_channel(ChannelCapacity::try_from(4_usize)?, DropPolicy::DropOldest);
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
        #[cfg(feature = "actuation")]
        #[cfg(all(feature = "nano-agent", unix))]
        let selected_motion = nano_setup_guard.take_motion();
        #[cfg(all(feature = "actuation", feature = "nano-agent", unix))]
        let navigation_stop_precedes_accessory_release = selected_motion
            .nano_live_motion_kind()
            .requires_navigation_stop_before_accessory_release();
        #[cfg(all(feature = "actuation", not(all(feature = "nano-agent", unix))))]
        let selected_motion = motion;
        #[cfg(feature = "actuation")]
        let mut navigation_worker_motion = Some(match selected_motion {
            PreparedLiveMotionSelection::Compatibility => {
                LiveNavigationWorkerMotion::compatibility(navigation_actuation_config)
            }
            #[cfg(all(feature = "nano-agent", unix))]
            PreparedLiveMotionSelection::Production(input) => {
                LiveNavigationWorkerMotion::Production(input)
            }
            #[cfg(all(feature = "nano-attended-navigation-trial", unix))]
            PreparedLiveMotionSelection::AttendedNavigationTrial(input) => {
                LiveNavigationWorkerMotion::AttendedNavigationTrial(input)
            }
            #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
            PreparedLiveMotionSelection::WheelsOffQualification(input) => {
                LiveNavigationWorkerMotion::WheelsOffQualification(input)
            }
        });
        let (
            navigation_coordinator,
            navigation_control_period,
            _navigation_dataset_directory,
            navigation_dataset_writer,
            navigation_dataset_handle,
        ) = match active_navigation {
            Some(active) => (
                Some(active.coordinator),
                Some(active.control_period),
                Some(active.dataset_directory),
                Some(active.dataset_writer),
                Some(active.dataset_handle),
            ),
            None => (None, None, None, None, None),
        };
        #[cfg(all(feature = "nano-agent", unix))]
        let mut navigation_dataset_handle = navigation_dataset_handle;

        #[cfg(all(feature = "nano-agent", unix))]
        let quiescent_checkpoint_requested = Arc::new(AtomicBool::new(false));
        #[cfg(all(feature = "nano-agent", unix))]
        let (mut checkpoint_worker_bridge, mut checkpoint_main_bridge) =
            match _navigation_dataset_directory.as_ref() {
                Some(dataset_directory) => {
                    let (worker, main) = nano_dataset_checkpoint_bridge(
                        Arc::clone(&quiescent_checkpoint_requested),
                        dataset_directory.clone(),
                    );
                    (Some(worker), Some(main))
                }
                None => (None, None),
            };

        let dense_handle = if let (Some(startup), Some(command_rx), stats_tx, snapshot_tx) = (
            occupancy_startup,
            dense_command_rx_for_worker.take(),
            dense_stats_tx_for_worker.take(),
            occupancy_snapshot_tx_for_worker.take(),
        ) {
            let dense_running = Arc::clone(&running);
            #[cfg(all(feature = "nano-agent", unix))]
            let dense_checkpoint_requested = Arc::clone(&quiescent_checkpoint_requested);
            Some(spawn_live_thread("kiko-occupancy", move || {
                #[cfg(all(feature = "nano-agent", unix))]
                let _exit_guard = LiveThreadExitGuard::checkpoint_aware(
                    dense_running,
                    dense_checkpoint_requested,
                );
                #[cfg(not(all(feature = "nano-agent", unix)))]
                let _exit_guard = LiveThreadExitGuard::new(dense_running);
                match startup {
                    LiveOccupancyWorkerStartup::Fresh(config) => {
                        kiko_slam::dense::occupancy_runtime::run_occupancy_worker(
                            config,
                            &command_rx,
                            stats_tx.as_ref(),
                            snapshot_tx,
                        )
                    }
                    #[cfg(all(feature = "nano-agent", unix))]
                    LiveOccupancyWorkerStartup::ContinuedReplay {
                        runtime,
                        initial_snapshot,
                    } => kiko_slam::dense::occupancy_runtime::run_occupancy_worker_from_runtime(
                        *runtime,
                        Some(initial_snapshot),
                        &command_rx,
                        stats_tx.as_ref(),
                        snapshot_tx,
                    ),
                }
            })?)
        } else {
            None
        };

        #[cfg(all(feature = "nano-agent", unix))]
        let navigation_production_state = nano_setup_guard.take_production_state();
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
                let navigation_slam_telemetry = slam_telemetry.clone();
                Some(spawn_live_thread("kiko-navigation", move || {
                    run_live_navigation_worker(
                        coordinator,
                        control_period,
                        navigation_clock_epoch,
                        capture_clock_origin,
                        #[cfg(feature = "actuation")]
                        navigation_worker_motion
                            .take()
                            .expect("enabled navigation consumes one motion selection"),
                        #[cfg(all(feature = "nano-agent", unix))]
                        navigation_production_state,
                        #[cfg(all(feature = "nano-agent", unix))]
                        checkpoint_worker_bridge.take(),
                        navigation_running,
                        navigation_slam_telemetry,
                        visual_rx,
                        depth_rx,
                        imu_rx,
                        map_rx,
                        navigation_map_viz_tx,
                        navigation_viz_tx,
                    )
                })?)
            }
            None => None,
        };
        #[cfg(all(feature = "nano-agent", unix))]
        let post_navigation_setup_guard = NanoPostNavigationSetupGuard::new(
            Arc::clone(&running),
            navigation_handle,
            nano_setup_guard.take_accessory(),
        );

        let inference_running = Arc::clone(&running);
        let inference_slam_telemetry = slam_telemetry.clone();
        #[cfg(all(feature = "nano-agent", unix))]
        let inference_checkpoint_requested = Arc::clone(&quiescent_checkpoint_requested);
        let inference_handle = spawn_live_thread(
            "kiko-inference",
            move || -> Result<(), LiveThreadError> {
                #[cfg(all(feature = "nano-agent", unix))]
                let _exit_guard = LiveThreadExitGuard::checkpoint_aware(
                    Arc::clone(&inference_running),
                    inference_checkpoint_requested,
                );
                #[cfg(not(all(feature = "nano-agent", unix)))]
                let _exit_guard = LiveThreadExitGuard::new(Arc::clone(&inference_running));
                let mut tracker = tracker;
                let inference_clock = InstantHostClock::new(capture_clock_origin);
                let depth_rx = depth_rx;
                let depth_enabled_for_diagnostics = depth_rx.is_some();
                let mut dense_generation = dense_generation;
                #[cfg(all(feature = "nano-agent", unix))]
                let mut warm_relocalization_gate = warm_relocalization_gate;
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
                    let slam_attempt =
                        inference_slam_telemetry
                            .begin(observation.host_arrival())
                            .map_err(|source| LiveThreadError::SlamTelemetry { source })?;
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
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            tracker.process(observation.into_pair())
                        }));
                    let slam_completed_at = inference_clock
                        .checked_now()
                        .map_err(|source| LiveThreadError::SlamClock { source })?;
                    match process_result {
                        Ok(Ok(mut output)) => {
                            let slam_snapshot = inference_slam_telemetry
                                .complete_success(slam_attempt, slam_completed_at)
                                .map_err(|source| LiveThreadError::SlamTelemetry { source })?;
                            #[cfg(all(feature = "nano-agent", unix))]
                            let warm_localized = match warm_relocalization_gate.take() {
                                None => true,
                                Some(gate) => match gate.observe(&output).map_err(|source| {
                                    LiveThreadError::WarmStartRelocalization { source }
                                })? {
                                    NanoWarmStartRelocalizationTransition::Awaiting(next) => {
                                        warm_relocalization_gate = Some(next);
                                        false
                                    }
                                    NanoWarmStartRelocalizationTransition::Localized(evidence) => {
                                        eprintln!(
                                            "warm start: fresh-camera relocalization proven against replay_map={:?} by candidate={:?}, current_map={:?}",
                                            evidence.replay_map(),
                                            evidence.candidate(),
                                            evidence.localization().map_snapshot(),
                                        );
                                        true
                                    }
                                },
                            };
                            #[cfg(not(all(feature = "nano-agent", unix)))]
                            let warm_localized = true;
                            if let (Some(sender), Some(pending)) =
                                (navigation_visual_tx.as_ref(), pending_visual)
                            {
                                let admission = if warm_localized {
                                    visual_admission_from_output(pending, &output)
                                } else {
                                    VisualAdmission::no_localization(
                                        pending.complete(VisualAttemptOutcome::NoLocalization),
                                    )
                                    .map_err(LiveVisualAdmissionBuildError::Admission)
                                }
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
                                                    dense_integrations_dropped_newest
                                                        .saturating_add(1);
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
                                    VizPacket::try_new(left.clone(), right.clone(), matches)
                                        .map_err(|source| LiveThreadError::VisualizationPacket {
                                            source,
                                        })?,
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
                                slam: slam_snapshot,
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
                            inference_slam_telemetry
                                .complete_failure(
                                    slam_attempt,
                                    slam_completed_at,
                                    requires_pipeline_shutdown,
                                )
                                .map_err(|source| LiveThreadError::SlamTelemetry { source })?;
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
                                .map_err(|source| LiveThreadError::VisualAdmissionBuild {
                                    source: LiveVisualAdmissionBuildError::Admission(source),
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
                            inference_slam_telemetry
                                .complete_failure(slam_attempt, slam_completed_at, true)
                                .map_err(|source| LiveThreadError::SlamTelemetry { source })?;
                            if let (Some(sender), Some(pending)) =
                                (navigation_visual_tx.as_ref(), pending_visual)
                            {
                                let admission = VisualAdmission::fatal_failure(
                                    pending.complete(VisualAttemptOutcome::FatalFailure),
                                )
                                .map_err(|source| LiveThreadError::VisualAdmissionBuild {
                                    source: LiveVisualAdmissionBuildError::Admission(source),
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
                #[cfg(all(feature = "nano-agent", unix))]
                if warm_relocalization_gate.is_some() && inference_running.load(Ordering::Acquire) {
                    return Err(LiveThreadError::WarmStartRelocalizationIncomplete);
                }
                if dense_integrations_dropped_newest > 0 {
                    eprintln!(
                        "dense integrations dropped_newest (inference view): {dense_integrations_dropped_newest}"
                    );
                }
                if depth_reorder_warnings_seen > 0 {
                    eprintln!("depth reorder warnings observed: {depth_reorder_warnings_seen}");
                }
                inference_slam_telemetry
                    .close()
                    .map_err(|source| LiveThreadError::SlamTelemetry { source })?;
                Ok(())
            },
        )?;

        let decimation = rerun_decimation;
        #[cfg(all(feature = "nano-agent", unix))]
        let face_viz_thread_counters = Arc::clone(&face_viz_counters);
        let viz_handle = spawn_live_thread(
            "kiko-rerun",
            move || -> Result<(), LiveThreadError> {
                let mut initialization_error = None;
                let mut navigation_recording = None;
                let recording = match rerun_target {
                    LiveRerunTarget::Connect => {
                        rerun::RecordingStreamBuilder::new("kiko-slam-live").connect_grpc()
                    }
                    #[cfg(all(feature = "nano-agent", unix))]
                    LiveRerunTarget::ServeLoopback {
                        bind,
                        memory_limit_bytes,
                    } => {
                        let address = bind.ip().to_string();
                        eprintln!(
                            "rerun: serving live-agent diagnostics on {bind} with memory_limit_bytes={memory_limit_bytes}"
                        );
                        rerun::RecordingStreamBuilder::new("kiko-nano-agent").serve_grpc_opts(
                            &address,
                            bind.port(),
                            rerun::ServerOptions {
                                memory_limit: rerun::MemoryLimit::from_bytes(memory_limit_bytes),
                                ..Default::default()
                            },
                        )
                    }
                };
                let mut sink = match recording {
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
                        eprintln!("failed to initialize live Rerun diagnostics: {err}");
                        initialization_error = Some(LiveThreadError::RerunConnect { source: err });
                        None
                    }
                };
                let mut logging_error = None;
                let mut frame_rx = Some(viz_rx);
                let mut rgb_rx = Some(rgb_viz_rx);
                let mut map_rx = occupancy_snapshot_rx;
                let mut navigation_rx = navigation_viz_rx;
                let mut navigation_context_logged = false;
                let mut slam_context_logged = false;
                #[cfg(all(feature = "nano-agent", unix))]
                let mut face_rx = face_viz_rx;
                #[cfg(all(feature = "nano-agent", unix))]
                let mut face_cancel_rx = Some(face_viz_cancel_rx);
                #[cfg(all(feature = "nano-agent", unix))]
                let mut last_logged_rgb = None;
                #[cfg(all(feature = "nano-agent", unix))]
                let mut face_context_logged = false;
                if initialization_error.is_some() {
                    // Stop upstream visualization work immediately. Tracking and
                    // occupancy remain authoritative and shut down through their own
                    // channels; this worker still reports the typed initialization
                    // failure after any applicable Rerun finalization.
                    frame_rx = None;
                    rgb_rx = None;
                    map_rx = None;
                    navigation_rx = None;
                    #[cfg(all(feature = "nano-agent", unix))]
                    {
                        face_rx = None;
                        face_cancel_rx = None;
                    }
                }
                while frame_rx.is_some()
                    || rgb_rx.is_some()
                    || map_rx.is_some()
                    || navigation_rx.is_some()
                    || {
                        #[cfg(all(feature = "nano-agent", unix))]
                        {
                            face_rx.is_some()
                        }
                        #[cfg(not(all(feature = "nano-agent", unix)))]
                        {
                            false
                        }
                    }
                {
                    let mut close_frames = false;
                    let mut close_rgb = false;
                    let mut close_maps = false;
                    let mut close_navigation = false;
                    #[cfg(all(feature = "nano-agent", unix))]
                    let mut close_face = false;
                    {
                        let mut selector = crossbeam_channel::Select::new();
                        let frame_operation = frame_rx
                            .as_ref()
                            .map(|receiver| selector.recv(receiver.as_receiver()));
                        let rgb_operation = rgb_rx
                            .as_ref()
                            .map(|receiver| selector.recv(receiver.as_receiver()));
                        let map_operation = map_rx
                            .as_ref()
                            .map(|receiver| selector.recv(receiver.as_receiver()));
                        let navigation_operation = navigation_rx
                            .as_ref()
                            .map(|receiver| selector.recv(receiver.as_receiver()));
                        #[cfg(all(feature = "nano-agent", unix))]
                        let face_operation = face_rx
                            .as_ref()
                            .map(|receiver| selector.recv(receiver.as_receiver()));
                        #[cfg(all(feature = "nano-agent", unix))]
                        let face_cancel_operation = if face_rx.is_some() {
                            face_cancel_rx
                                .as_ref()
                                .map(|receiver| selector.recv(receiver))
                        } else {
                            None
                        };
                        let selected = selector.select();
                        let selected_index = selected.index();
                        if frame_operation == Some(selected_index) {
                            let receiver = frame_rx
                                .as_ref()
                                .expect("registered frame receiver")
                                .as_receiver();
                            match selected.recv(receiver) {
                                Ok(message) => {
                                    if logging_error.is_none()
                                        && let Some(sink) = sink.as_mut()
                                        && let Some(recording) = navigation_recording.as_ref()
                                        && let Err(error) = log_live_viz_message(
                                            recording,
                                            sink,
                                            message,
                                            &mut slam_context_logged,
                                        )
                                    {
                                        eprintln!(
                                            "live Rerun logging failed; disconnecting visualization producers: {error}"
                                        );
                                        logging_error = Some(error);
                                        close_frames = true;
                                        close_rgb = true;
                                        close_maps = true;
                                        close_navigation = true;
                                        #[cfg(all(feature = "nano-agent", unix))]
                                        {
                                            close_face = true;
                                        }
                                    }
                                }
                                Err(_) => close_frames = true,
                            }
                        } else if rgb_operation == Some(selected_index) {
                            let receiver = rgb_rx
                                .as_ref()
                                .expect("registered RGB receiver")
                                .as_receiver();
                            match selected.recv(receiver) {
                                Ok(message) => {
                                    if logging_error.is_none()
                                        && let Some(recording) = navigation_recording.as_ref()
                                    {
                                        match log_live_rgb_viz_message(recording, message) {
                                            Ok(frame_key) => {
                                                #[cfg(all(feature = "nano-agent", unix))]
                                                {
                                                    last_logged_rgb = Some(frame_key);
                                                }
                                                #[cfg(not(all(feature = "nano-agent", unix)))]
                                                {
                                                    let _ = frame_key;
                                                }
                                            }
                                            Err(error) => {
                                                eprintln!(
                                                    "live Rerun RGB logging failed; disconnecting visualization producers: {error}"
                                                );
                                                logging_error = Some(error);
                                                close_frames = true;
                                                close_rgb = true;
                                                close_maps = true;
                                                close_navigation = true;
                                                #[cfg(all(feature = "nano-agent", unix))]
                                                {
                                                    close_face = true;
                                                }
                                            }
                                        }
                                    }
                                }
                                Err(_) => close_rgb = true,
                            }
                        } else if map_operation == Some(selected_index) {
                            let receiver = map_rx
                                .as_ref()
                                .expect("registered occupancy receiver")
                                .as_receiver();
                            match selected.recv(receiver) {
                                Ok(snapshot) => {
                                    if logging_error.is_none()
                                        && let Some(sink) = sink.as_mut()
                                        && let Some(recording) = navigation_recording.as_ref()
                                    {
                                        let (timestamp, snapshot) = snapshot.into_parts();
                                        let result = apply_live_rerun_timeline_domain(
                                            recording,
                                            LiveRerunTimelineDomain::Capture,
                                        )
                                        .and_then(|()| sink.log_occupancy(timestamp, snapshot));
                                        if let Err(error) = result {
                                            eprintln!(
                                                "live Rerun occupancy logging failed; disconnecting visualization producers: {error}"
                                            );
                                            logging_error = Some(error);
                                            close_frames = true;
                                            close_rgb = true;
                                            close_maps = true;
                                            close_navigation = true;
                                            #[cfg(all(feature = "nano-agent", unix))]
                                            {
                                                close_face = true;
                                            }
                                        }
                                    }
                                }
                                Err(_) => close_maps = true,
                            }
                        } else if navigation_operation == Some(selected_index) {
                            let receiver = navigation_rx
                                .as_ref()
                                .expect("registered navigation receiver")
                                .as_receiver();
                            match selected.recv(receiver) {
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
                                        close_rgb = true;
                                        close_maps = true;
                                        close_navigation = true;
                                        #[cfg(all(feature = "nano-agent", unix))]
                                        {
                                            close_face = true;
                                        }
                                    }
                                }
                                Err(_) => close_navigation = true,
                            }
                        } else {
                            #[cfg(all(feature = "nano-agent", unix))]
                            if face_operation == Some(selected_index) {
                                let receiver = face_rx
                                    .as_ref()
                                    .expect("registered face receiver")
                                    .as_receiver();
                                match selected.recv(receiver) {
                                    Ok(message) => {
                                        face_viz_thread_counters.record_received();
                                        if logging_error.is_none()
                                            && let Some(recording) = navigation_recording.as_ref()
                                        {
                                            match log_live_face_viz_message(
                                                recording,
                                                message,
                                                last_logged_rgb,
                                                &mut face_context_logged,
                                            ) {
                                                Ok(Ok(overlay_matched)) => {
                                                    face_viz_thread_counters
                                                        .record_logged(overlay_matched);
                                                }
                                                Ok(Err(source)) => {
                                                    face_viz_thread_counters.record_invalid();
                                                    eprintln!(
                                                        "live face Rerun diagnostics disabled after invalid provenance: {source}"
                                                    );
                                                    close_face = true;
                                                }
                                                Err(error) => {
                                                    eprintln!(
                                                        "live face Rerun logging failed; disconnecting visualization producers: {error}"
                                                    );
                                                    logging_error = Some(error);
                                                    close_frames = true;
                                                    close_rgb = true;
                                                    close_maps = true;
                                                    close_navigation = true;
                                                    close_face = true;
                                                }
                                            }
                                        }
                                    }
                                    Err(_) => close_face = true,
                                }
                            } else if face_cancel_operation == Some(selected_index) {
                                let receiver = face_cancel_rx
                                    .as_ref()
                                    .expect("registered face cancellation receiver");
                                let _ = selected.recv(receiver);
                                let pending_abandoned = face_rx
                                    .as_ref()
                                    .is_some_and(|receiver| receiver.try_recv().is_ok());
                                face_viz_thread_counters.record_cancelled(pending_abandoned);
                                close_face = true;
                            } else {
                                unreachable!("selected one registered live Rerun operation");
                            }
                            #[cfg(not(all(feature = "nano-agent", unix)))]
                            unreachable!("selected one registered live Rerun operation");
                        }
                    }
                    if close_frames {
                        frame_rx = None;
                    }
                    if close_rgb {
                        rgb_rx = None;
                    }
                    if close_maps {
                        map_rx = None;
                    }
                    if close_navigation {
                        navigation_rx = None;
                    }
                    #[cfg(all(feature = "nano-agent", unix))]
                    if close_face {
                        face_rx = None;
                        face_cancel_rx = None;
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
            },
        )?;

        #[cfg(all(feature = "nano-agent", unix))]
        let (mut navigation_handle, accessory) = post_navigation_setup_guard.into_parts();
        #[cfg(all(feature = "nano-agent", unix))]
        let mut accessory_failures = Vec::new();
        #[cfg(all(feature = "nano-agent", unix))]
        let mut accessory_terminal_fault_recorded = false;
        #[cfg(all(feature = "nano-agent", unix))]
        let mut face_stage_stats_final = None;
        #[cfg(all(feature = "nano-agent", unix))]
        let mut accessory_worker = accessory;
        let mut capture_error = None;
        #[cfg(all(feature = "nano-agent", unix))]
        let mut supervision_errors = Vec::new();
        let rgb_viz_tx = Some(rgb_viz_tx);
        #[cfg(all(feature = "nano-agent", unix))]
        let mut rgb_viz_tx = rgb_viz_tx;
        #[cfg(all(feature = "nano-agent", unix))]
        let mut rgb_viz_frame_index = 0_usize;

        #[cfg(all(feature = "nano-agent", unix))]
        if let Some(supervision) = systemd_supervision.as_mut() {
            supervision.notify_ready(Instant::now())?;
        }
        eprintln!("streaming matches... press ctrl+c to stop");

        #[cfg(all(feature = "nano-agent", unix))]
        let capture_exit_guard = LiveThreadExitGuard::checkpoint_aware(
            Arc::clone(&running),
            Arc::clone(&quiescent_checkpoint_requested),
        );
        #[cfg(not(all(feature = "nano-agent", unix)))]
        let capture_exit_guard = LiveThreadExitGuard::new(Arc::clone(&running));
        'capture: while running.load(Ordering::Relaxed) {
            #[cfg(all(feature = "nano-agent", unix))]
            if quiescent_checkpoint_requested.load(Ordering::Acquire) {
                break 'capture;
            }
            let mut got_any = false;

            #[cfg(all(feature = "nano-agent", unix))]
            if let Some(worker) = accessory_worker.as_ref() {
                match worker.try_terminal_fault() {
                    Ok(Some(fault)) => {
                        accessory_terminal_fault_recorded = true;
                        accessory_failures.push(LiveAccessoryError::TerminalFault(fault));
                        running.store(false, Ordering::SeqCst);
                        break 'capture;
                    }
                    Ok(None) => {}
                    Err(source) => {
                        accessory_failures.push(LiveAccessoryError::FaultMonitor(source));
                        running.store(false, Ordering::SeqCst);
                        break 'capture;
                    }
                }
            }

            #[cfg(all(feature = "nano-agent", unix))]
            if let Some(supervision) = systemd_supervision.as_mut()
                && let Err(source) = supervision.poll_watchdog(Instant::now())
            {
                supervision_errors.push(LiveCaptureError::SystemdSupervision { source });
                running.store(false, Ordering::SeqCst);
                break 'capture;
            }

            #[cfg(all(feature = "nano-agent", unix))]
            if let Some(worker) = accessory_worker.as_mut() {
                match device.rgb(0) {
                    Ok(frame) => {
                        let accessory_stream_epoch = worker.readiness().stream_epoch();
                        let publish_rgb_viz =
                            rgb_viz_frame_index.is_multiple_of(rerun_decimation.get());
                        rgb_viz_frame_index = match rgb_viz_frame_index.checked_add(1) {
                            Some(next) => next,
                            None => {
                                eprintln!(
                                    "live RGB Rerun diagnostic frame index exhausted; disabling RGB visualization"
                                );
                                rgb_viz_tx = None;
                                0
                            }
                        };
                        let submit_outcome =
                            worker.submit_rgb_after_observation(frame, |frame| {
                                if publish_rgb_viz && let Some(sender) = rgb_viz_tx.as_ref() {
                                    match LiveRgbVizMsg::try_from_oak(
                                        frame,
                                        accessory_stream_epoch,
                                    ) {
                                        Ok(message) => {
                                            if matches!(
                                                sender.try_send(message),
                                                SendOutcome::Disconnected
                                            ) {
                                                rgb_viz_tx = None;
                                            }
                                        }
                                        Err(source) => {
                                            eprintln!(
                                                "live RGB Rerun diagnostic disabled after an invalid frame: {source}"
                                            );
                                            rgb_viz_tx = None;
                                        }
                                    }
                                }
                            });
                        match submit_outcome {
                            NanoAccessoryFrameSubmitOutcome::Enqueued
                            | NanoAccessoryFrameSubmitOutcome::ReplacedOlderFrame => {}
                            outcome @ (NanoAccessoryFrameSubmitOutcome::TerminalFaultPendingPublication
                            | NanoAccessoryFrameSubmitOutcome::TerminalFaultLatched) => {
                                debug_assert!(
                                    accessory_submission_requires_exact_fault_wait(outcome)
                                );
                                // Both outcomes can precede publication to the
                                // sole fault receiver: the face lane commits
                                // its terminal output and closes raw admission
                                // before the accessory actor consumes that
                                // output. Stop every controller/navigation
                                // owner, then wait boundedly for the exact first
                                // cause instead of inventing a generic ingress
                                // failure. The accessory stays alive during the
                                // wait, but prior base authority cannot.
                                match stop_live_before_waiting_for_accessory_fault(
                                    running.as_ref(),
                                    || {
                                        worker.wait_for_terminal_fault(
                                            NANO_ACCESSORY_TERMINAL_PUBLICATION_TIMEOUT,
                                        )
                                    },
                                ) {
                                    Ok(fault) => {
                                        accessory_terminal_fault_recorded = true;
                                        accessory_failures
                                            .push(LiveAccessoryError::TerminalFault(fault));
                                    }
                                    Err(source) => accessory_failures
                                        .push(LiveAccessoryError::FaultMonitor(source)),
                                }
                                break 'capture;
                            }
                            outcome @ (NanoAccessoryFrameSubmitOutcome::IngressDisconnected
                            | NanoAccessoryFrameSubmitOutcome::ChannelPoisoned) => {
                                accessory_failures.push(LiveAccessoryError::FrameIngress(outcome));
                                running.store(false, Ordering::SeqCst);
                                break 'capture;
                            }
                        }
                        got_any = true;
                    }
                    Err(ImageError::Timeout { .. } | ImageError::QueueEmpty) => {}
                    Err(source) => {
                        capture_error = Some(LiveCaptureError::RgbImage { source });
                        break 'capture;
                    }
                }
            }

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

        #[cfg(all(feature = "nano-agent", unix))]
        if let Some(supervision) = systemd_supervision.as_mut()
            && let Err(source) = supervision.notify_stopping()
        {
            supervision_errors.push(LiveCaptureError::SystemdSupervision { source });
        }

        // Dropping the capture exit guard propagates every normal/error exit
        // before the remaining bounded queues are joined and drained. It also
        // covers unwinding through this scope.
        drop(capture_exit_guard);
        drop(pair_tx);
        drop(depth_tx);
        drop(imu_tx);
        drop(rgb_viz_tx);
        #[cfg(all(feature = "nano-agent", unix))]
        drop(face_viz_cancel_tx);
        drop(navigation_dataset_writer);
        let mut live_failures = capture_error
            .into_iter()
            .map(LiveWorkerFailure::Capture)
            .collect::<Vec<_>>();
        #[cfg(all(feature = "nano-agent", unix))]
        live_failures.extend(supervision_errors.drain(..).map(LiveWorkerFailure::Capture));
        #[cfg(all(feature = "nano-agent", unix))]
        live_failures.extend(
            accessory_failures
                .drain(..)
                .map(LiveWorkerFailure::Accessory),
        );
        let inference_failed = match inference_handle.join() {
            Ok(Ok(())) => false,
            Ok(Err(error)) => {
                live_failures.push(LiveWorkerFailure::Inference(error));
                true
            }
            Err(payload) => {
                live_failures.push(LiveWorkerFailure::InferencePanic {
                    detail: kiko_slam::panic_payload_to_string(payload.as_ref()),
                });
                true
            }
        };
        if inference_failed && let Err(source) = slam_telemetry.fault() {
            live_failures.push(LiveWorkerFailure::SlamTelemetry(source));
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
        #[cfg(all(feature = "nano-agent", unix))]
        if navigation_stop_precedes_accessory_release && let Some(handle) = navigation_handle.take()
        {
            match handle.join() {
                Ok(Ok(success)) => navigation_descriptor = Some(success.descriptor),
                Ok(Err(error)) => live_failures.push(LiveWorkerFailure::Navigation(error)),
                Err(payload) => live_failures.push(LiveWorkerFailure::NavigationPanic {
                    detail: kiko_slam::panic_payload_to_string(payload.as_ref()),
                }),
            }
        }

        // Accessory terminal evidence is authoritative for a restart-ready
        // checkpoint too. Collect it before deciding whether the drained
        // dataset may be published. Wheels-off qualification cannot request a
        // warm checkpoint, so its navigation owner has already joined above:
        // the controller-shutdown outcome is therefore collected before
        // release of supervised natural head hold. Production retains its
        // checkpoint handshake.
        #[cfg(all(feature = "nano-agent", unix))]
        if let Some(worker) = accessory_worker.take() {
            match worker.shutdown() {
                Ok(NanoAccessoryWorkerExit::Shutdown {
                    terminal_fault,
                    evidence,
                }) => {
                    let summary =
                        NanoAccessoryShutdownSummary::from_evidence(terminal_fault, *evidence);
                    face_stage_stats_final = Some(summary.face_stage_stats_are_final());
                    let face_problem_kind = summary.face_problem_kind();
                    let NanoAccessoryShutdownSummary {
                        terminal_fault,
                        eye_release_verified,
                        head_hold_preserving_release_completed,
                        fault_recovery_presentation,
                        pet_evidence_clean,
                        face_perception,
                    } = summary;
                    let terminal_fault_present = terminal_fault.is_some();
                    if let Some(fault) = terminal_fault
                        && !accessory_terminal_fault_recorded
                    {
                        live_failures.push(LiveWorkerFailure::Accessory(
                            LiveAccessoryError::TerminalFault(fault),
                        ));
                    }
                    if !eye_release_verified {
                        live_failures.push(LiveWorkerFailure::Accessory(
                            LiveAccessoryError::EyeReleaseUnverified,
                        ));
                    }
                    if !head_hold_preserving_release_completed {
                        live_failures.push(LiveWorkerFailure::Accessory(
                            LiveAccessoryError::HeadHoldPreservingReleaseUnverified,
                        ));
                    }
                    if let Some(problem) = classify_fault_recovery_presentation(
                        terminal_fault_present,
                        *fault_recovery_presentation,
                    ) {
                        live_failures.push(LiveWorkerFailure::Accessory(problem));
                    }
                    if !pet_evidence_clean {
                        live_failures.push(LiveWorkerFailure::Accessory(
                            LiveAccessoryError::PetEvidenceShutdownUnclean,
                        ));
                    }
                    if let Some(kind) = face_problem_kind {
                        live_failures.push(LiveWorkerFailure::Accessory(
                            LiveAccessoryError::FaceShutdown {
                                kind,
                                evidence: face_perception,
                            },
                        ));
                    }
                }
                Ok(exit) => {
                    face_stage_stats_final = Some(false);
                    live_failures.push(LiveWorkerFailure::Accessory(
                        LiveAccessoryError::UnexpectedExit(Box::new(exit)),
                    ));
                }
                Err(source) => {
                    face_stage_stats_final = Some(false);
                    live_failures.push(LiveWorkerFailure::Accessory(
                        LiveAccessoryError::ShutdownJoin(source),
                    ));
                }
            }
        }

        // A warm SaveMap is terminal: navigation has already stopped the
        // controller and finalized the journal, but waits here while the sole
        // dataset owner either publishes that exact descriptor or aborts the
        // session. This handshake occurs only after capture, inference,
        // occupancy, and accessory owners have drained.
        #[cfg(all(feature = "nano-agent", unix))]
        if quiescent_checkpoint_requested.load(Ordering::Acquire) {
            match checkpoint_main_bridge.take() {
                Some(bridge) => {
                    let remaining = bridge
                        .checkpoint_deadline
                        .get()
                        .copied()
                        .map(|deadline| deadline.saturating_duration_since(Instant::now()))
                        .unwrap_or(Duration::ZERO);
                    match bridge.request.recv_timeout(remaining) {
                        Ok(request) => {
                            let mut finalization = NanoDatasetCheckpointFinalization::Rejected;
                            let publishable = live_failures.is_empty()
                                && request.navigation_publishable
                                && request.descriptor.is_some();
                            if publishable {
                                match navigation_dataset_handle.take() {
                                    Some(handle) => {
                                        let descriptor = request.descriptor.expect(
                                        "publishable checkpoint request has a finalized descriptor",
                                    );
                                        match handle.finish_with_navigation_ingress(descriptor) {
                                            Ok(_) => {
                                                finalization =
                                                    NanoDatasetCheckpointFinalization::Published;
                                            }
                                            Err(source) => live_failures.push(
                                                LiveWorkerFailure::DatasetFinalization(source),
                                            ),
                                        }
                                    }
                                    None => live_failures
                                        .push(LiveWorkerFailure::WarmCheckpointCoordination(
                                        NanoDatasetCheckpointCoordinationError::MissingDatasetOwner,
                                    )),
                                }
                            }
                            if finalization == NanoDatasetCheckpointFinalization::Rejected
                                && let Some(handle) = navigation_dataset_handle.take()
                                && let Err(source) = handle.abort_without_manifest()
                            {
                                live_failures.push(LiveWorkerFailure::DatasetAbort(source));
                            }
                            if bridge.finalization.send(finalization).is_err() {
                                live_failures.push(LiveWorkerFailure::WarmCheckpointCoordination(
                                NanoDatasetCheckpointCoordinationError::FinalizationResponseChannelDisconnected,
                            ));
                            }
                        }
                        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                            live_failures.push(LiveWorkerFailure::WarmCheckpointCoordination(
                                NanoDatasetCheckpointCoordinationError::RequestTimedOut,
                            ));
                            if let Some(handle) = navigation_dataset_handle.take()
                                && let Err(source) = handle.abort_without_manifest()
                            {
                                live_failures.push(LiveWorkerFailure::DatasetAbort(source));
                            }
                        }
                        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                            live_failures.push(LiveWorkerFailure::WarmCheckpointCoordination(
                                NanoDatasetCheckpointCoordinationError::RequestChannelDisconnected,
                            ));
                            if let Some(handle) = navigation_dataset_handle.take()
                                && let Err(source) = handle.abort_without_manifest()
                            {
                                live_failures.push(LiveWorkerFailure::DatasetAbort(source));
                            }
                        }
                    }
                }
                None => {
                    live_failures.push(LiveWorkerFailure::WarmCheckpointCoordination(
                        NanoDatasetCheckpointCoordinationError::MissingBridge,
                    ));
                    if let Some(handle) = navigation_dataset_handle.take()
                        && let Err(source) = handle.abort_without_manifest()
                    {
                        live_failures.push(LiveWorkerFailure::DatasetAbort(source));
                    }
                }
            }
        }

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

        let visualization_diagnostic = match viz_handle.join() {
            Ok(Ok(())) => None,
            Ok(Err(error)) => Some(LiveVisualizationFailure::Worker(error)),
            Err(payload) => Some(LiveVisualizationFailure::Panic {
                detail: kiko_slam::panic_payload_to_string(payload.as_ref()),
            }),
        };
        let navigation_dataset_publishable =
            navigation_dataset_may_publish(authoritative_failure, navigation_descriptor.is_some());

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
        if let Some(diagnostic) = visualization_diagnostic {
            // Rerun receives diagnostic copies only. Reporting its terminal
            // failure must not turn an otherwise healthy robot owner into a
            // failed service that an external manager could restart.
            eprintln!("non-authoritative live visualization failure: {diagnostic}");
        }

        let pair_snapshot = pair_stats.snapshot();
        let viz_snapshot = viz_stats.snapshot();
        let rgb_viz_snapshot = rgb_viz_stats.snapshot();
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
        eprintln!(
            "RGB viz queue stats: enqueued={}, dropped_oldest={}, dropped_newest={}, disconnected={}",
            rgb_viz_snapshot.enqueued,
            rgb_viz_snapshot.dropped_oldest,
            rgb_viz_snapshot.dropped_newest,
            rgb_viz_snapshot.disconnected
        );
        #[cfg(all(feature = "nano-agent", unix))]
        if let Some(stats) = face_viz_channel_stats.as_ref() {
            let channel = stats.snapshot();
            let rerun = face_viz_counters.snapshot();
            eprintln!(
                "face viz stats: channel_enqueued={} channel_dropped_oldest={} channel_dropped_newest={} channel_disconnected={} rerun_received={} rerun_logged={} overlay_matched={} overlay_unmatched={} invalid={} consumer_cancelled={} pending_abandoned={}",
                channel.enqueued,
                channel.dropped_oldest,
                channel.dropped_newest,
                channel.disconnected,
                rerun.received,
                rerun.logged,
                rerun.overlay_matched,
                rerun.overlay_unmatched,
                rerun.invalid,
                rerun.consumer_cancelled,
                rerun.pending_abandoned,
            );
        }
        #[cfg(all(feature = "nano-agent", unix))]
        if let Some(handle) = face_stage_stats_handle.as_ref() {
            let stats = handle.snapshot();
            eprintln!(
                "face perception stage stats: final={} results_produced={} head_gaze_disabled_no_policy={} head_gaze_proposed={} head_gaze_withheld={} head_gaze_rejected={} handoff_enqueued={} handoff_replaced_older={} handoff_terminal_pending={} handoff_terminal_fault_latched={} handoff_disconnected={} handoff_channel_poisoned={}",
                face_stage_stats_final.unwrap_or(false),
                stats.results_produced,
                stats.head_gaze_disabled_no_policy,
                stats.head_gaze_proposed,
                stats.head_gaze_withheld,
                stats.head_gaze_rejected,
                stats.handoff_enqueued,
                stats.handoff_replaced_older,
                stats.handoff_terminal_pending,
                stats.handoff_terminal_fault_latched,
                stats.handoff_disconnected,
                stats.handoff_channel_poisoned,
            );
        }
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
        match slam_telemetry.snapshot() {
            Ok(snapshot) => {
                let rate = snapshot.rate_window.map_or_else(
                    || "unavailable".to_owned(),
                    |window| {
                        let hz = f64::from(window.successful_completions.saturating_sub(1)) * 1e9
                            / window.span_ns as f64;
                        format!(
                            "{hz:.6}Hz from {} successful completions over {}ns",
                            window.successful_completions, window.span_ns
                        )
                    },
                );
                eprintln!(
                    "live SLAM evidence: pipeline_state={:?} started_pairs={} successful_pairs={} recoverable_failures={} fatal_failures={} measured_success_rate_window={}; superpoint requested={} selected={}; lightglue requested={} selected={}",
                    snapshot.pipeline_state,
                    snapshot.started_pairs,
                    snapshot.successful_pairs,
                    snapshot.recoverable_failures,
                    snapshot.fatal_failures,
                    rate,
                    live_inference_backend_name(snapshot.inference.superpoint_requested),
                    live_selected_inference_backend_name(snapshot.inference.superpoint_selected),
                    live_inference_backend_name(snapshot.inference.lightglue_requested),
                    live_selected_inference_backend_name(snapshot.inference.lightglue_selected),
                );
            }
            Err(source) => live_failures.push(LiveWorkerFailure::SlamTelemetry(source)),
        }

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
    oak_eeprom: Option<OakEepromCalibrationEvidence>,
    rectified: bool,
) -> Calibration {
    // The OAK graph fixes StereoDepth alignment to rectified-left. On RVC2,
    // `rectifiedRight` can retain CAM_C's source intrinsic metadata even
    // though its pixels have been remapped into the common rectified-left
    // projection. Persist the projection of the delivered pixels, not that
    // stale source-camera metadata. Unrectified camera outputs keep their
    // independent projections.
    let right_projection = if rectified { left } else { right };
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
            fx: right_projection.fx(),
            fy: right_projection.fy(),
            cx: right_projection.cx(),
            cy: right_projection.cy(),
            width: right_projection.width(),
            height: right_projection.height(),
        },
        baseline_m,
        rectified,
        oak_eeprom: oak_eeprom.map(|evidence| DatasetOakEepromCalibrationEvidence {
            stereo_left_camera_socket: dataset_oak_camera_socket(
                evidence.stereo_left_camera_socket(),
            ),
            stereo_right_camera_socket: dataset_oak_camera_socket(
                evidence.stereo_right_camera_socket(),
            ),
            imu_to_camera_b_m: evidence.imu_to_camera_b_m(),
            stereo_left_rectification_rotation_raw: evidence
                .stereo_left_rectification_rotation_raw(),
        }),
    }
}

#[cfg(feature = "record")]
fn dataset_oak_camera_socket(socket: OakCameraSocket) -> OakCalibrationCameraSocket {
    match socket {
        OakCameraSocket::CameraA => OakCalibrationCameraSocket::CameraA,
        OakCameraSocket::CameraB => OakCalibrationCameraSocket::CameraB,
        OakCameraSocket::CameraC => OakCalibrationCameraSocket::CameraC,
        OakCameraSocket::Unrecognized => {
            unreachable!("admitted EEPROM evidence cannot retain an unrecognized camera socket")
        }
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
    #[test]
    fn generic_cli_model_defaults_are_workspace_relative() {
        let model_dir = Path::new(kiko_slam::WORKSPACE_MODEL_DIRECTORY);
        assert!(model_dir.is_relative());
        assert_eq!(
            resolve_model_path(model_dir, None, "sp.onnx"),
            PathBuf::from("crates/kiko-slam/models/sp.onnx")
        );
        assert_eq!(
            resolve_model_path(model_dir, Some(&PathBuf::from("alternate.onnx")), "sp.onnx"),
            PathBuf::from("crates/kiko-slam/models/alternate.onnx")
        );
        assert_eq!(
            resolve_model_path(
                model_dir,
                Some(&PathBuf::from("/opt/models/sp.onnx")),
                "sp.onnx"
            ),
            PathBuf::from("/opt/models/sp.onnx")
        );
    }

    #[cfg(all(feature = "record", feature = "actuation"))]
    use super::LiveNavigationWorkerMotion;
    #[cfg(all(
        unix,
        any(
            feature = "nano-attended-navigation-trial",
            feature = "nano-wheels-off-qualification"
        )
    ))]
    use super::NanoLiveMotionKind;
    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    use super::{
        AttendedWheelsOffAttestationError, AttendedWheelsOffPreflight,
        BufferedFreshAttendedMotionTerminal, FreshAttendedMotionAttestationClosure,
        FreshAttendedMotionAttestationError, FreshAttendedMotionAttestationGate,
        FreshAttendedMotionAttestationInput, FreshAttendedMotionAttestationWorker,
        FreshAttendedMotionAttestationWorkerError, FreshAttendedMotionAttestationWorkerPoll,
        FreshAttendedMotionAttestationWorkerShutdown, InitialMotorPowerDisconnectedClaim,
        MAX_QUALIFICATION_ATTESTATION_LINE_BYTES, QUALIFICATION_ATTESTATION_CHALLENGE_BYTES,
        WheelsOffAttestationChallengeSource, WheelsOffQualificationAndMotorPowerDisconnectError,
        WheelsOffQualificationAttestationReadinessBlocker,
        classify_wheels_off_qualification_attestation_readiness,
        finish_attended_wheels_off_qualification, fresh_motion_attestation_must_cancel,
        lower_hex_qualification_challenge, prompt_fresh_attended_motion_phrase,
        read_attended_wheels_off_preflight, read_bounded_tty_line,
        read_fresh_attended_motion_attestation, read_post_run_motor_power_disconnected,
    };
    use super::{
        BaConfigValues, BenchError, Cli, Command, DepthRingCapacity, LiveDecisionVizKind,
        LiveDenseCommandClass, LiveDenseRouteContext, LiveDenseRouteDisposition,
        LiveDenseRouteError, LiveLosslessRouteError, LiveThreadExitGuard, LiveVisualShape,
        OccupancyProjectionContractError, OdometryVizProcessingError, OfflineDepthSelector,
        OfflineFatalDenseError, OfflineFatalTrackerError, RerunDestination, RerunDestinationError,
        RerunFinishTimeout, RerunSessionError, build_ba_config_from_values,
        classify_live_dense_route, classify_live_visual_shape, classify_lossless_send,
        combine_rerun_results, live_decision_viz_status, navigation_dataset_may_publish,
        occupancy_depth_camera, reject_removed_ba_motion_prior, require_level_optical_world,
        resolve_model_path, take_deferred_offline_snapshot_error,
    };
    #[cfg(feature = "record")]
    use super::{
        HostMonotonicRangeError, LiveInferenceRuntimeEvidence, LiveNavigationWorkerError,
        LiveNavigationWorkerInput, LiveSelectedInferenceBackend, LiveSlamPipelineState,
        LiveSlamTelemetry, LiveSlamTelemetryError, checked_monotonic_duration_ns,
        combine_live_navigation_failures, drain_entry_snapshot, measure_live_control_tick_timing,
        select_live_navigation_worker_input,
    };
    #[cfg(all(
        feature = "record",
        feature = "actuation",
        feature = "agent-runtime",
        unix
    ))]
    use super::{
        LiveNavigationVizMsg, LivePhysicalStateVizPublishError, abnormal_production_socket_exit,
        production_period_requires_motion_tick, publish_live_navigation_viz_message,
    };
    #[cfg(all(feature = "nano-agent", feature = "operator-console", unix))]
    use super::{LiveSensorStream, LiveSensorStreamHealth};
    #[cfg(all(feature = "nano-agent", unix))]
    use super::{
        MAX_NANO_STREAM_EPOCH_ATTEMPTS, NanoOperationAndControllerOwnerError, NanoStreamEpochError,
        TrackerDefaults, V2ControllerOwnerTerminationError, build_canonical_nano_tracker_config,
        finish_nano_controller_owner, fresh_nano_stream_epoch_from,
    };
    use clap::{Parser as _, error::ErrorKind};
    use kiko_slam::dataset::{DatasetError, DepthOpticalFrame, DepthProjectionContract};
    #[cfg(feature = "record")]
    use kiko_slam::dataset::{
        OakCalibrationCameraSocket,
        OakEepromCalibrationEvidence as DatasetOakEepromCalibrationEvidence,
    };
    use kiko_slam::dense::{occupancy::OccupancyError, occupancy_runtime::OccupancyRuntimeError};
    #[cfg(all(
        feature = "record",
        feature = "actuation",
        feature = "agent-runtime",
        unix
    ))]
    use kiko_slam::navigation::{AgentControlSocketCleanupOutcome, AgentControlSocketTaskExit};
    #[cfg(feature = "record")]
    use kiko_slam::{ChannelCapacity, DropPolicy, SendOutcome, bounded_channel};
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

    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    struct FixedWheelsOffChallenges {
        values: VecDeque<[u8; QUALIFICATION_ATTESTATION_CHALLENGE_BYTES]>,
    }

    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    impl FixedWheelsOffChallenges {
        fn from_bytes(values: impl IntoIterator<Item = u8>) -> Self {
            Self {
                values: values
                    .into_iter()
                    .map(|value| [value; QUALIFICATION_ATTESTATION_CHALLENGE_BYTES])
                    .collect(),
            }
        }
    }

    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    impl WheelsOffAttestationChallengeSource for FixedWheelsOffChallenges {
        fn next_challenge(
            &mut self,
        ) -> Result<
            [u8; QUALIFICATION_ATTESTATION_CHALLENGE_BYTES],
            AttendedWheelsOffAttestationError,
        > {
            Ok(self.values.pop_front().expect("scripted challenge"))
        }
    }

    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    struct CancellingWheelsOffChallenge<'a> {
        cancellation: &'a AtomicBool,
    }

    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    impl WheelsOffAttestationChallengeSource for CancellingWheelsOffChallenge<'_> {
        fn next_challenge(
            &mut self,
        ) -> Result<
            [u8; QUALIFICATION_ATTESTATION_CHALLENGE_BYTES],
            AttendedWheelsOffAttestationError,
        > {
            self.cancellation.store(true, Ordering::Release);
            Ok([0x5a; QUALIFICATION_ATTESTATION_CHALLENGE_BYTES])
        }
    }

    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    fn challenged_wheels_off_phrase(phrase: &str, byte: u8) -> String {
        format!(
            "{phrase} {}",
            lower_hex_qualification_challenge(&[byte; QUALIFICATION_ATTESTATION_CHALLENGE_BYTES])
        )
    }

    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    fn test_wheels_off_preflight() -> AttendedWheelsOffPreflight {
        AttendedWheelsOffPreflight {
            motor_power_disconnected: InitialMotorPowerDisconnectedClaim,
        }
    }

    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    fn test_wheels_off_attestation() -> kiko_slam::navigation::OperatorClaimedWheelsOffAttestation {
        kiko_slam::navigation::OperatorClaimedWheelsOffAttestation::try_new(
            true,
            true,
            true,
            true,
            true,
            Instant::now(),
        )
        .expect("explicit test attestation")
    }

    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    fn poll_test_attestation_gate(
        gate: &mut FreshAttendedMotionAttestationGate,
    ) -> Result<FreshAttendedMotionAttestationWorkerPoll, FreshAttendedMotionAttestationWorkerError>
    {
        let process_running = AtomicBool::new(true);
        for _ in 0..100 {
            match gate.advance_after_read_only_runtime_tick(&process_running) {
                Ok(FreshAttendedMotionAttestationWorkerPoll::Pending) => {
                    std::thread::sleep(Duration::from_millis(1));
                }
                outcome => return outcome,
            }
        }
        panic!("test attestation worker did not complete");
    }
    #[cfg(feature = "record")]
    use std::time::Instant;

    #[cfg(feature = "record")]
    fn test_slam_telemetry() -> LiveSlamTelemetry {
        LiveSlamTelemetry::new(LiveInferenceRuntimeEvidence {
            superpoint_requested: kiko_slam::InferenceBackend::Auto,
            superpoint_selected: LiveSelectedInferenceBackend::Cpu,
            lightglue_requested: kiko_slam::InferenceBackend::Cpu,
            lightglue_selected: LiveSelectedInferenceBackend::Cpu,
        })
    }

    #[cfg(feature = "record")]
    #[test]
    fn live_slam_telemetry_preserves_exact_outcomes_and_measured_rate_window() {
        let telemetry = test_slam_telemetry();
        let first = telemetry
            .begin(kiko_slam::HostMonotonicTimestamp::from_nanos(100))
            .expect("first pair starts");
        assert!(matches!(
            telemetry.begin(kiko_slam::HostMonotonicTimestamp::from_nanos(101)),
            Err(LiveSlamTelemetryError::AttemptAlreadyInFlight)
        ));
        let first_snapshot = telemetry
            .complete_success(first, kiko_slam::HostMonotonicTimestamp::from_nanos(200))
            .expect("first success completes");
        assert_eq!(first_snapshot.started_pairs, 1);
        assert_eq!(first_snapshot.successful_pairs, 1);
        assert_eq!(first_snapshot.rate_window, None);

        let recoverable = telemetry
            .begin(kiko_slam::HostMonotonicTimestamp::from_nanos(250))
            .expect("second pair starts");
        telemetry
            .complete_failure(
                recoverable,
                kiko_slam::HostMonotonicTimestamp::from_nanos(300),
                false,
            )
            .expect("recoverable result completes");
        let second_success = telemetry
            .begin(kiko_slam::HostMonotonicTimestamp::from_nanos(350))
            .expect("third pair starts");
        let snapshot = telemetry
            .complete_success(
                second_success,
                kiko_slam::HostMonotonicTimestamp::from_nanos(500),
            )
            .expect("second success completes");
        assert_eq!(snapshot.started_pairs, 3);
        assert_eq!(snapshot.successful_pairs, 2);
        assert_eq!(snapshot.recoverable_failures, 1);
        assert_eq!(snapshot.fatal_failures, 0);
        assert_eq!(
            snapshot.rate_window,
            Some(super::LiveSlamRateWindowEvidence {
                successful_completions: 2,
                span_ns: 300,
            })
        );
        assert_eq!(
            snapshot.last_successful_source_arrival,
            Some(kiko_slam::HostMonotonicTimestamp::from_nanos(350))
        );

        let fatal = telemetry
            .begin(kiko_slam::HostMonotonicTimestamp::from_nanos(600))
            .expect("fatal pair starts");
        let faulted = telemetry
            .complete_failure(
                fatal,
                kiko_slam::HostMonotonicTimestamp::from_nanos(700),
                true,
            )
            .expect("fatal outcome remains observable");
        assert_eq!(faulted.pipeline_state, LiveSlamPipelineState::Faulted);
        assert_eq!(faulted.fatal_failures, 1);
        assert!(matches!(
            telemetry.begin(kiko_slam::HostMonotonicTimestamp::from_nanos(800)),
            Err(LiveSlamTelemetryError::PipelineNotRunning {
                state: LiveSlamPipelineState::Faulted
            })
        ));
    }

    #[cfg(feature = "record")]
    #[test]
    fn live_slam_telemetry_rejects_clock_regression_without_rounding_it_valid() {
        let telemetry = test_slam_telemetry();
        let attempt = telemetry
            .begin(kiko_slam::HostMonotonicTimestamp::from_nanos(200))
            .expect("pair starts");
        assert!(matches!(
            telemetry.complete_success(attempt, kiko_slam::HostMonotonicTimestamp::from_nanos(199)),
            Err(LiveSlamTelemetryError::CompletionBeforeSource { .. })
        ));

        let ordered = test_slam_telemetry();
        let first = ordered
            .begin(kiko_slam::HostMonotonicTimestamp::from_nanos(300))
            .expect("pair starts");
        ordered
            .complete_success(first, kiko_slam::HostMonotonicTimestamp::from_nanos(400))
            .expect("pair completes");
        assert!(matches!(
            ordered.begin(kiko_slam::HostMonotonicTimestamp::from_nanos(299)),
            Err(LiveSlamTelemetryError::SourceArrivalRegressed { .. })
        ));
    }

    #[cfg(feature = "record")]
    #[test]
    fn live_slam_rate_window_retains_the_latest_sixty_four_completions() {
        let telemetry = test_slam_telemetry();
        let mut snapshot = None;
        for index in 0_u64..65 {
            let source = kiko_slam::HostMonotonicTimestamp::from_nanos(index * 10);
            let completion = kiko_slam::HostMonotonicTimestamp::from_nanos(index * 10 + 1);
            let attempt = telemetry.begin(source).expect("ordered pair starts");
            snapshot = Some(
                telemetry
                    .complete_success(attempt, completion)
                    .expect("ordered pair completes"),
            );
        }
        let snapshot = snapshot.expect("the loop produced a snapshot");
        assert_eq!(snapshot.started_pairs, 65);
        assert_eq!(snapshot.successful_pairs, 65);
        assert_eq!(
            snapshot.rate_window,
            Some(super::LiveSlamRateWindowEvidence {
                successful_completions: 64,
                span_ns: 630,
            })
        );
    }

    #[cfg(all(feature = "nano-agent", unix))]
    #[test]
    fn canonical_nano_tracker_config_keeps_loop_relocalization_without_ambient_model() {
        let config = build_canonical_nano_tracker_config(
            TrackerDefaults {
                min_keyframe_points: 80,
                refresh_inliers: 20,
                min_inliers: 15,
            },
            kiko_slam::KeypointLimit::try_from(1_024_usize).expect("bounded keypoint limit"),
            kiko_slam::DownscaleFactor::try_from(2_usize).expect("bounded downscale"),
        )
        .expect("fixed canonical tracker policy");

        assert!(config.loop_subsystem.is_enabled());
        assert!(config.loop_subsystem.uses_bootstrap_descriptors_only());
        assert!(config.loop_subsystem.loop_closure().is_some());
        assert!(config.loop_subsystem.global_descriptor().is_none());
        assert!(config.loop_subsystem.relocalization().is_some());
        assert_eq!(
            config
                .backend
                .expect("canonical async backend")
                .queue_depth(),
            2
        );
        assert_eq!(
            kiko_slam::TrackerRuntimePolicy::canonical_nano().descriptor_max_respawns(),
            3
        );
    }

    #[cfg(all(feature = "nano-agent", feature = "operator-console", unix))]
    #[test]
    fn live_oak_health_requires_fresh_samples_from_all_streams_and_faults_on_disconnect() {
        let mut health = LiveSensorStreamHealth::awaiting_first_samples();
        let first = kiko_slam::HostMonotonicTimestamp::from_nanos(100);
        assert_eq!(
            health.console_health(first),
            kiko_slam::navigation::ConsoleHealth::Degraded
        );
        health.observe(LiveSensorStream::Visual, first);
        health.observe(LiveSensorStream::Depth, first);
        assert_eq!(
            health.console_health(first),
            kiko_slam::navigation::ConsoleHealth::Degraded
        );
        health.observe(LiveSensorStream::Imu, first);
        assert_eq!(
            health.console_health(first),
            kiko_slam::navigation::ConsoleHealth::Ready
        );
        let stale = kiko_slam::HostMonotonicTimestamp::from_nanos(
            first.as_nanos() + super::LIVE_SENSOR_CONSOLE_MAX_SAMPLE_AGE_NS + 1,
        );
        assert_eq!(
            health.console_health(stale),
            kiko_slam::navigation::ConsoleHealth::Degraded
        );
        let before_first = kiko_slam::HostMonotonicTimestamp::from_nanos(first.as_nanos() - 1);
        assert_eq!(
            health.console_health(before_first),
            kiko_slam::navigation::ConsoleHealth::Degraded
        );
        #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
        {
            health.latch_stale(LiveSensorStream::Depth);
            let after_latch =
                kiko_slam::HostMonotonicTimestamp::from_nanos(first.as_nanos().saturating_add(1));
            health.observe(LiveSensorStream::Visual, after_latch);
            health.observe(LiveSensorStream::Depth, after_latch);
            health.observe(LiveSensorStream::Imu, after_latch);
            assert_eq!(health.depth_observed_at, None);
            assert_eq!(
                health.console_health(after_latch),
                kiko_slam::navigation::ConsoleHealth::Degraded,
                "a latched synthetic stale stream cannot recover from later frames"
            );
        }
        health.mark_closed(LiveSensorStream::Visual);
        assert_eq!(
            health.console_health(first),
            kiko_slam::navigation::ConsoleHealth::Faulted
        );
    }

    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    #[test]
    fn attestation_enable_boundary_rejects_samples_that_expired_during_a_delayed_tick() {
        let mut health = LiveSensorStreamHealth::awaiting_first_samples();
        let sample = kiko_slam::HostMonotonicTimestamp::from_nanos(100);
        for stream in [
            LiveSensorStream::Visual,
            LiveSensorStream::Depth,
            LiveSensorStream::Imu,
        ] {
            health.observe(stream, sample);
        }
        let tick_started = kiko_slam::HostMonotonicTimestamp::from_nanos(
            sample.as_nanos() + super::LIVE_SENSOR_CONSOLE_MAX_SAMPLE_AGE_NS,
        );
        assert_eq!(
            health.console_health(tick_started),
            kiko_slam::navigation::ConsoleHealth::Ready
        );

        let enable_boundary =
            kiko_slam::HostMonotonicTimestamp::from_nanos(tick_started.as_nanos() + 1);
        assert_eq!(
            classify_wheels_off_qualification_attestation_readiness(
                health.console_health(enable_boundary)
                    == kiko_slam::navigation::ConsoleHealth::Ready,
                true,
                true,
                true,
                true,
            ),
            Err(WheelsOffQualificationAttestationReadinessBlocker::FreshVisualDepthImuUnavailable)
        );
    }

    #[cfg(all(feature = "nano-agent", unix))]
    #[test]
    fn pending_or_latched_accessory_fault_stops_before_waiting_for_exact_publication() {
        for outcome in [
            kiko_slam::navigation::NanoAccessoryFrameSubmitOutcome::TerminalFaultPendingPublication,
            kiko_slam::navigation::NanoAccessoryFrameSubmitOutcome::TerminalFaultLatched,
        ] {
            assert!(super::accessory_submission_requires_exact_fault_wait(
                outcome
            ));
            let running = AtomicBool::new(true);
            let observed = super::stop_live_before_waiting_for_accessory_fault(&running, || {
                running.load(Ordering::SeqCst)
            });
            assert!(
                !observed,
                "{outcome:?} publication wait must observe live authority stopped"
            );
            assert!(!running.load(Ordering::SeqCst));
        }
        assert!(!super::accessory_submission_requires_exact_fault_wait(
            kiko_slam::navigation::NanoAccessoryFrameSubmitOutcome::Enqueued
        ));
    }

    #[cfg(all(feature = "nano-agent", unix))]
    #[test]
    fn pre_owner_cleanup_marks_a_terminal_already_reported_by_the_primary_error() {
        let terminal = super::LiveAccessoryError::TerminalFault(
            kiko_slam::navigation::NanoAccessoryTerminalFault::ReadinessObserverDropped,
        );
        assert!(super::live_accessory_error_reports_terminal(&terminal));

        let ingress = super::LiveAccessoryError::FrameIngress(
            kiko_slam::navigation::NanoAccessoryFrameSubmitOutcome::IngressDisconnected,
        );
        assert!(!super::live_accessory_error_reports_terminal(&ingress));

        let shutdown = super::NanoPreOwnerAccessoryShutdown::Evidence {
            summary: super::NanoAccessoryShutdownSummary {
                terminal_fault: Some(
                    kiko_slam::navigation::NanoAccessoryTerminalFault::ReadinessObserverDropped,
                ),
                eye_release_verified: true,
                head_hold_preserving_release_completed: true,
                fault_recovery_presentation: Box::new(
                    kiko_slam::navigation::NanoFaultRecoveryPresentationEvidence::Presented {
                        frames_applied: std::num::NonZeroU64::MIN,
                    },
                ),
                pet_evidence_clean: true,
                face_perception: super::NanoFacePerceptionShutdownEvidence::Disabled,
            },
            terminal_fault_already_reported: true,
        };
        let displayed = shutdown.to_string();
        assert!(displayed.contains("already_reported_by_primary"));
        assert!(!displayed.contains("ReadinessObserverDropped"));
    }

    #[cfg(all(feature = "nano-agent", unix))]
    #[test]
    fn fault_recovery_classification_never_replaces_the_primary_fault() {
        let secondary = super::classify_fault_recovery_presentation(
            true,
            kiko_slam::navigation::NanoFaultRecoveryPresentationEvidence::Failed {
                frames_applied: 3,
                source: kiko_slam::navigation::NanoFaultRecoveryPresentationFault::MissingPresentationDuration,
            },
        )
        .expect("secondary presentation failure remains reportable");
        let super::LiveAccessoryError::FaultRecoveryPresentationFailed {
            frames_applied,
            source,
        } = secondary
        else {
            panic!("secondary failure must retain its dedicated classification");
        };
        assert_eq!(frames_applied, 3);
        assert!(matches!(
            source.as_ref(),
            kiko_slam::navigation::NanoFaultRecoveryPresentationFault::MissingPresentationDuration
        ));

        assert!(matches!(
            super::classify_fault_recovery_presentation(
                true,
                kiko_slam::navigation::NanoFaultRecoveryPresentationEvidence::NotRequired,
            ),
            Some(super::LiveAccessoryError::FaultRecoveryPresentationMissing)
        ));
        assert!(matches!(
            super::classify_fault_recovery_presentation(
                false,
                kiko_slam::navigation::NanoFaultRecoveryPresentationEvidence::Presented {
                    frames_applied: std::num::NonZeroU64::MIN,
                },
            ),
            Some(super::LiveAccessoryError::FaultRecoveryPresentationUnexpected(_))
        ));
        assert!(
            super::classify_fault_recovery_presentation(
                false,
                kiko_slam::navigation::NanoFaultRecoveryPresentationEvidence::NotRequired,
            )
            .is_none()
        );
    }

    #[cfg(all(feature = "nano-agent", unix))]
    #[test]
    fn live_face_stage_stats_are_final_only_after_disabled_or_joined_shutdown() {
        let disabled = super::NanoAccessoryShutdownSummary {
            terminal_fault: None,
            eye_release_verified: true,
            head_hold_preserving_release_completed: true,
            fault_recovery_presentation: Box::new(
                kiko_slam::navigation::NanoFaultRecoveryPresentationEvidence::NotRequired,
            ),
            pet_evidence_clean: true,
            face_perception: super::NanoFacePerceptionShutdownEvidence::Disabled,
        };
        assert!(disabled.is_fully_healthy());
        assert_eq!(disabled.face_problem_kind(), None);
        assert!(disabled.face_stage_stats_are_final());

        let detached = super::NanoAccessoryShutdownSummary {
            terminal_fault: None,
            eye_release_verified: true,
            head_hold_preserving_release_completed: true,
            fault_recovery_presentation: Box::new(
                kiko_slam::navigation::NanoFaultRecoveryPresentationEvidence::NotRequired,
            ),
            pet_evidence_clean: true,
            face_perception: super::NanoFacePerceptionShutdownEvidence::Join(
                kiko_slam::navigation::NanoFacePerceptionJoinEvidence::DetachedAfterTimeout {
                    configured_timeout: std::time::Duration::from_secs(2),
                    active_join_budget: std::time::Duration::from_millis(250),
                },
            ),
        };
        assert!(!detached.is_fully_healthy());
        assert_eq!(
            detached.face_problem_kind(),
            Some(super::NanoFaceShutdownProblemKind::DetachedUncertain)
        );
        assert!(!detached.face_stage_stats_are_final());
    }

    #[cfg(all(feature = "record", feature = "nano-agent", unix))]
    #[test]
    fn live_rgb_diagnostics_require_exact_tightly_packed_bgr8() {
        assert_eq!(
            super::validate_live_rgb_viz_layout(640, 400, 1_920, 768_000),
            Ok(())
        );
        assert_eq!(
            super::validate_live_rgb_viz_layout(640, 400, 1_921, 768_400),
            Err(super::LiveRgbVizBuildError::StrideMismatch {
                expected: 1_920,
                actual: 1_921,
            })
        );
        assert_eq!(
            super::validate_live_rgb_viz_layout(640, 400, 1_920, 767_999),
            Err(super::LiveRgbVizBuildError::PixelLengthMismatch {
                expected: 768_000,
                actual: 767_999,
            })
        );
        assert_eq!(
            super::validate_live_rgb_viz_layout(u32::MAX, 1, u32::MAX, 0),
            Err(super::LiveRgbVizBuildError::RowBytesOverflow { width: u32::MAX })
        );
    }

    #[cfg(all(feature = "record", feature = "nano-agent", unix))]
    #[test]
    fn live_face_geometry_rejects_every_inexact_u32_to_f32_boundary() {
        assert_eq!(
            super::exact_u32_as_f32("coordinate", (1_u32 << 24) - 1),
            Ok(((1_u32 << 24) - 1) as f32)
        );
        assert_eq!(
            super::exact_u32_as_f32("coordinate", 1_u32 << 24),
            Ok((1_u32 << 24) as f32)
        );
        assert_eq!(
            super::exact_u32_as_f32("coordinate", (1_u32 << 24) + 1),
            Err(
                super::LiveFaceVizBuildError::PixelCoordinateUnrepresentable {
                    field: "coordinate",
                    value: (1_u32 << 24) + 1,
                }
            )
        );
        assert_eq!(
            super::exact_u32_as_f32("coordinate", u32::MAX),
            Err(
                super::LiveFaceVizBuildError::PixelCoordinateUnrepresentable {
                    field: "coordinate",
                    value: u32::MAX,
                }
            )
        );
    }

    #[cfg(all(feature = "record", feature = "nano-agent", unix))]
    #[test]
    fn live_face_overlay_join_requires_the_complete_rgb_provenance_key() {
        let expected = super::LiveRgbFrameKey {
            device_capture_sequence: 11,
            host_delivery_sequence: 13,
            device_timestamp_ns: 17,
            timestamp_reference: super::CameraTimestampReference::ExposureMidpoint,
            stream_epoch: 19,
            width: 640,
            height: 400,
        };
        assert_eq!(Some(expected), Some(expected));

        let distinct = [
            super::LiveRgbFrameKey {
                device_capture_sequence: 12,
                ..expected
            },
            super::LiveRgbFrameKey {
                host_delivery_sequence: 14,
                ..expected
            },
            super::LiveRgbFrameKey {
                device_timestamp_ns: 18,
                ..expected
            },
            super::LiveRgbFrameKey {
                stream_epoch: 20,
                ..expected
            },
            super::LiveRgbFrameKey {
                width: 641,
                ..expected
            },
            super::LiveRgbFrameKey {
                height: 401,
                ..expected
            },
        ];
        for actual in distinct {
            assert_ne!(
                Some(expected),
                Some(actual),
                "a face result must never overlay a different RGB provenance key"
            );
        }
    }

    #[cfg(feature = "record")]
    #[test]
    fn live_rerun_domain_switches_remove_every_sticky_foreign_timeline() {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        enum Value {
            Time(i64),
            Sequence(i64),
        }

        #[derive(Default)]
        struct FakeTimelineTarget {
            active: std::cell::RefCell<std::collections::BTreeMap<&'static str, Value>>,
        }

        impl super::LiveRerunTimelineTarget for FakeTimelineTarget {
            fn reset_live_time(&self) {
                self.active.borrow_mut().clear();
            }

            fn set_live_time(&self, timeline: &'static str, time: rerun::TimeCell) {
                self.active
                    .borrow_mut()
                    .insert(timeline, Value::Time(time.as_i64()));
            }

            fn set_live_sequence(&self, timeline: &'static str, sequence: i64) {
                self.active
                    .borrow_mut()
                    .insert(timeline, Value::Sequence(sequence));
            }
        }

        fn active(target: &FakeTimelineTarget) -> std::collections::BTreeMap<&'static str, Value> {
            target.active.borrow().clone()
        }

        let target = FakeTimelineTarget::default();
        target
            .active
            .borrow_mut()
            .insert("future_timeline", Value::Sequence(1));
        super::apply_live_rerun_timeline_domain(
            &target,
            super::LiveRerunTimelineDomain::Navigation {
                tick_sequence: 7,
                host_timestamp_ns: Some(11),
            },
        )
        .expect("navigation timeline is representable");
        assert_eq!(
            active(&target),
            std::collections::BTreeMap::from([
                ("navigation_host_ns", Value::Time(11)),
                ("navigation_tick", Value::Sequence(7)),
            ])
        );

        super::apply_live_rerun_timeline_domain(&target, super::LiveRerunTimelineDomain::Capture)
            .expect("capture domain has no additional weak fields");
        assert!(active(&target).is_empty());

        super::apply_live_rerun_timeline_domain(
            &target,
            super::LiveRerunTimelineDomain::Rgb {
                capture_timestamp_ns: 13,
                device_capture_sequence: 17,
                host_delivery_sequence: 19,
            },
        )
        .expect("RGB timeline is representable");
        assert_eq!(
            active(&target),
            std::collections::BTreeMap::from([
                ("capture_ns", Value::Time(13)),
                ("oak_rgb_capture_sequence", Value::Sequence(17)),
                ("oak_rgb_host_delivery_sequence", Value::Sequence(19),),
            ])
        );

        #[cfg(all(feature = "nano-agent", unix))]
        {
            super::apply_live_rerun_timeline_domain(
                &target,
                super::LiveRerunTimelineDomain::Face {
                    capture_timestamp_ns: 19,
                    device_capture_sequence: 23,
                    host_delivery_sequence: 29,
                    detector_result_sequence: 31,
                },
            )
            .expect("face timeline is representable");
            assert_eq!(
                active(&target),
                std::collections::BTreeMap::from([
                    ("capture_ns", Value::Time(19)),
                    ("face_detector_result_sequence", Value::Sequence(31),),
                    ("oak_rgb_capture_sequence", Value::Sequence(23),),
                    ("oak_rgb_host_delivery_sequence", Value::Sequence(29),),
                ])
            );
        }

        super::apply_live_rerun_timeline_domain(
            &target,
            super::LiveRerunTimelineDomain::Navigation {
                tick_sequence: 37,
                host_timestamp_ns: None,
            },
        )
        .expect("navigation sequence alone is valid");
        assert_eq!(
            active(&target),
            std::collections::BTreeMap::from([("navigation_tick", Value::Sequence(37),)])
        );
    }

    #[cfg(feature = "record")]
    #[test]
    fn due_navigation_tick_precedes_every_ready_sensor_input() {
        fn disconnected<T>() -> crossbeam_channel::Receiver<T> {
            let (sender, receiver) = crossbeam_channel::bounded(1);
            drop(sender);
            receiver
        }

        let scheduled_at = Instant::now();
        let (tick_sender, tick_receiver) = crossbeam_channel::bounded(1);
        tick_sender
            .send(scheduled_at)
            .expect("tick fixture receiver");
        let visual = disconnected::<super::VisualAdmission>();
        let depth = disconnected::<super::DepthObservation>();
        let imu = disconnected::<super::ImuReport>();
        let map = disconnected::<super::TimedOccupancySnapshot>();

        assert!(matches!(
            select_live_navigation_worker_input(
                &tick_receiver,
                &visual,
                &depth,
                &imu,
                &map,
            ),
            LiveNavigationWorkerInput::Tick(actual) if actual == scheduled_at
        ));
    }

    #[cfg(feature = "record")]
    #[test]
    fn entry_snapshot_drain_is_not_extended_by_concurrent_replenishment() {
        let (sender, receiver) = crossbeam_channel::bounded(3);
        for value in [1_u8, 2, 3] {
            sender.send(value).expect("bounded fixture capacity");
        }
        let mut admitted = Vec::new();

        let outcome = drain_entry_snapshot(&receiver, |value| {
            admitted.push(value);
            sender
                .try_send(value + 10)
                .expect("each receive frees one replacement slot");
            Ok::<(), std::convert::Infallible>(())
        })
        .expect("infallible admission");

        assert_eq!(outcome.drained, 3);
        assert!(!outcome.disconnected);
        assert_eq!(admitted, [1, 2, 3]);
        assert_eq!(
            receiver.try_iter().collect::<Vec<_>>(),
            [11, 12, 13],
            "arrivals after entry remain for the next outer-loop selection"
        );
    }

    #[cfg(feature = "record")]
    #[test]
    fn control_tick_timing_retains_current_and_monotonic_maximum_lateness() {
        let scheduled_at = Instant::now();
        let first = measure_live_control_tick_timing(
            scheduled_at,
            scheduled_at + Duration::from_nanos(7),
            10,
        )
        .expect("bounded timing fixture");
        assert_eq!(first.current_lateness_ns, 7);
        assert_eq!(first.maximum_lateness_ns, 10);

        let second = measure_live_control_tick_timing(
            scheduled_at,
            scheduled_at + Duration::from_nanos(12),
            first.maximum_lateness_ns,
        )
        .expect("bounded timing fixture");
        assert_eq!(second.current_lateness_ns, 12);
        assert_eq!(second.maximum_lateness_ns, 12);

        assert!(matches!(
            measure_live_control_tick_timing(
                scheduled_at + Duration::from_nanos(1),
                scheduled_at,
                second.maximum_lateness_ns,
            ),
            Err(super::LiveControlTickTimingError::ScheduledAfterSelection)
        ));
    }

    #[cfg(feature = "record")]
    #[test]
    fn monotonic_duration_is_checked_for_order_and_u64_range() {
        assert_eq!(checked_monotonic_duration_ns(7, 12), Some(5));
        assert_eq!(checked_monotonic_duration_ns(12, 7), None);
        assert_eq!(
            checked_monotonic_duration_ns(0, u128::from(u64::MAX) + 1),
            None
        );
    }

    #[cfg(all(
        feature = "record",
        feature = "actuation",
        feature = "agent-runtime",
        feature = "nano-agent",
        feature = "operator-console",
        unix
    ))]
    #[test]
    fn console_pose_composition_uses_map_from_odom_and_normalizes_yaw() {
        let pose = super::compose_console_pose_components(
            2.0,
            3.0,
            std::f64::consts::FRAC_PI_4,
            10.0,
            20.0,
            std::f64::consts::FRAC_PI_2,
        )
        .expect("finite rigid transform composition");

        assert!((pose.x_m.get() - 7.0).abs() < 1.0e-12);
        assert!((pose.y_m.get() - 22.0).abs() < 1.0e-12);
        assert!((pose.yaw_rad.get() - 3.0 * std::f64::consts::FRAC_PI_4).abs() < 1.0e-12);

        let wrapped = super::compose_console_pose_components(
            0.0,
            0.0,
            3.0 * std::f64::consts::FRAC_PI_4,
            0.0,
            0.0,
            3.0 * std::f64::consts::FRAC_PI_4,
        )
        .expect("finite wrapped yaw");
        assert!((wrapped.yaw_rad.get() + std::f64::consts::FRAC_PI_2).abs() < 1.0e-12);
    }

    #[cfg(all(
        feature = "record",
        feature = "actuation",
        feature = "agent-runtime",
        feature = "nano-agent",
        feature = "operator-console",
        unix
    ))]
    #[test]
    fn console_pose_composition_rejects_nonfinite_boundary_values() {
        assert!(matches!(
            super::compose_console_pose_components(f64::INFINITY, 0.0, 0.0, 0.0, 0.0, 0.0,),
            Err(super::LiveProductionConsoleProjectionError::Numeric(_))
        ));
    }

    #[cfg(all(
        feature = "record",
        feature = "actuation",
        feature = "agent-runtime",
        feature = "nano-agent",
        feature = "operator-console",
        unix
    ))]
    #[test]
    fn console_mpc_odom_points_use_the_same_map_from_odom_transform() {
        let point = super::transform_console_odom_point_components(
            2.0,
            3.0,
            10.0,
            20.0,
            std::f64::consts::FRAC_PI_2,
        )
        .expect("finite odom-to-map point");

        assert!((point.x_m.get() - 7.0).abs() < 1.0e-12);
        assert!((point.y_m.get() - 22.0).abs() < 1.0e-12);
    }

    #[cfg(all(
        feature = "record",
        feature = "actuation",
        feature = "agent-runtime",
        feature = "nano-agent",
        feature = "operator-console",
        unix
    ))]
    #[test]
    fn console_hides_current_pose_and_stale_solver_path_when_localization_is_not_current() {
        use kiko_slam::navigation::AgentLocalizationStateV1;

        assert!(!super::console_localized_navigation_visible(Some(
            AgentLocalizationStateV1::Lost,
        )));
        assert!(!super::console_localized_navigation_visible(Some(
            AgentLocalizationStateV1::Unavailable,
        )));
        assert!(!super::console_localized_navigation_visible(None));
        assert!(super::console_localized_navigation_visible(Some(
            AgentLocalizationStateV1::Localized,
        )));

        assert!(!super::console_current_solver_path_visible(
            Some(AgentLocalizationStateV1::Lost),
            Some(10),
        ));
        assert!(!super::console_current_solver_path_visible(
            Some(AgentLocalizationStateV1::Localized),
            None,
        ));
        assert!(super::console_current_solver_path_visible(
            Some(AgentLocalizationStateV1::Localized),
            Some(10),
        ));
    }

    #[cfg(all(
        feature = "record",
        feature = "actuation",
        feature = "agent-runtime",
        feature = "nano-agent",
        feature = "operator-console",
        unix
    ))]
    #[test]
    fn console_projects_actual_operator_and_agent_authority_from_the_linear_guard() {
        use kiko_slam::navigation::{
            ConsoleActualAuthoritySource, ConsoleSourceKind, LiveMotionAuthorityState,
            OperatorConsoleRetainedAuthorityKind,
        };
        use kiko_supervisor_core::AuthorityLeaseId;

        let owner = Some(LiveMotionAuthorityState::Manual {
            lease_id: AuthorityLeaseId::try_new(7).unwrap(),
        });
        let console = super::project_console_actual_authority_state(
            owner,
            Some((
                OperatorConsoleRetainedAuthorityKind::Manual,
                11,
                ConsoleSourceKind::Operator,
            )),
        )
        .expect("console authority projection")
        .expect("actual console authority");
        assert_eq!(console.source, ConsoleActualAuthoritySource::Operator);
        assert_eq!(console.console_downstream_request_id, Some(11));

        let agent = super::project_console_actual_authority_state(
            owner,
            Some((
                OperatorConsoleRetainedAuthorityKind::Manual,
                12,
                ConsoleSourceKind::Agent,
            )),
        )
        .expect("agent authority projection")
        .expect("actual agent authority");
        assert_eq!(agent.source, ConsoleActualAuthoritySource::Agent);
        assert_eq!(agent.console_downstream_request_id, Some(12));
    }

    #[cfg(all(
        feature = "record",
        feature = "actuation",
        feature = "agent-runtime",
        feature = "nano-agent",
        feature = "operator-console",
        unix
    ))]
    #[test]
    fn console_rejects_retained_authority_that_contradicts_the_sole_owner() {
        use kiko_slam::navigation::{
            AgentAutonomousMode, ConsoleSourceKind, LiveMotionAuthorityState,
            OperatorConsoleRetainedAuthorityKind,
        };
        use kiko_supervisor_core::AuthorityLeaseId;

        let error = super::project_console_actual_authority_state(
            Some(LiveMotionAuthorityState::Manual {
                lease_id: AuthorityLeaseId::try_new(7).unwrap(),
            }),
            Some((
                OperatorConsoleRetainedAuthorityKind::Autonomous(AgentAutonomousMode::Explore),
                11,
                ConsoleSourceKind::Operator,
            )),
        )
        .expect_err("mismatched authority must fail closed");
        assert!(matches!(
            error,
            super::LiveProductionConsoleProjectionError::ConsoleAuthorityModeMismatch { .. }
        ));

        let error = super::project_console_actual_authority_state(
            Some(LiveMotionAuthorityState::Manual {
                lease_id: AuthorityLeaseId::try_new(8).unwrap(),
            }),
            None,
        )
        .expect_err("an owner token cannot outlive its unified-console guard");
        assert!(matches!(
            error,
            super::LiveProductionConsoleProjectionError::OwnerAuthorityWithoutConsole { .. }
        ));
    }

    #[cfg(feature = "record")]
    #[test]
    fn timing_only_packet_cannot_update_or_clear_physical_navigation_state() {
        let message = super::LiveNavigationVizMsg::control_tick_timing_only(
            7,
            11,
            super::LiveControlTickTiming {
                current_lateness_ns: 3,
                maximum_lateness_ns: 5,
            },
        );

        assert!(matches!(
            message.kind,
            super::LiveNavigationVizMessageKind::ControlTickTimingOnly { .. }
        ));
        assert!(!message.kind.updates_navigation_state());
        assert!(message.kind.control_tick_timing().is_some());
        assert!(message.requested_pwm.is_none());
        assert!(message.applied_actuation.is_none());
        assert!(message.fault_actuation.is_none());
        assert!(message.successful_solver_duration_ns.is_none());
    }

    #[cfg(all(
        feature = "record",
        feature = "actuation",
        feature = "agent-runtime",
        unix
    ))]
    #[test]
    fn production_period_defers_manual_begin_without_weakening_later_deadman_ticks() {
        assert!(
            !production_period_requires_motion_tick(false, true, false),
            "manual begin owns exactly its transition period"
        );
        assert!(
            production_period_requires_motion_tick(false, false, false),
            "the following period runs the normal missing-command deadman path"
        );
        assert!(
            !production_period_requires_motion_tick(false, false, true),
            "a command-applied tick is never duplicated"
        );
    }

    #[cfg(all(
        feature = "record",
        feature = "actuation",
        feature = "agent-runtime",
        unix
    ))]
    #[test]
    fn terminal_transition_suppresses_the_same_period_motion_tick() {
        assert!(
            !production_period_requires_motion_tick(true, false, false),
            "shutdown and warm-checkpoint transitions never start another periodic tick"
        );
    }

    #[cfg(all(
        feature = "record",
        feature = "actuation",
        feature = "agent-runtime",
        unix
    ))]
    fn physical_state_viz_message(
        status: &'static str,
        requested_pwm: Option<[i8; 2]>,
        objective_cost: Option<f64>,
    ) -> LiveNavigationVizMsg {
        LiveNavigationVizMsg {
            tick_sequence: 1,
            host_timestamp_ns: Some(1),
            goal: None,
            goal_state: "MappingOnly".to_owned(),
            odometry_state: None,
            path: None,
            local_costmap: None,
            base_to_odom: None,
            odom_to_map: None,
            predicted_odom: None,
            decision_id: None,
            request_id: None,
            status,
            reason: status.to_owned(),
            requested_pwm,
            objective_cost,
            shadow_record_motor_packets_sent: None,
            applied_actuation: None,
            fault_actuation: None,
            diagnostic_warning: None,
            successful_solver_duration_ns: None,
            kind: super::LiveNavigationVizMessageKind::State {
                control_tick_timing: None,
            },
        }
    }

    #[cfg(all(
        feature = "record",
        feature = "actuation",
        feature = "agent-runtime",
        unix
    ))]
    #[test]
    fn newest_physical_stop_replaces_stale_nonzero_visualization_without_fallback() {
        let (sender, receiver, _) = bounded_channel(
            ChannelCapacity::try_from(1_usize).expect("nonzero capacity"),
            DropPolicy::DropOldest,
        );
        assert_eq!(
            sender.try_send(physical_state_viz_message(
                "stale_nonzero",
                Some([40, 40]),
                Some(3.0),
            )),
            SendOutcome::Enqueued
        );
        let mut sender = Some(sender);

        publish_live_navigation_viz_message(
            &mut sender,
            physical_state_viz_message("lifecycle_zero_applied", Some([0, 0]), None),
        )
        .expect("DropOldest retains the newest physical state");

        let newest = receiver
            .as_receiver()
            .try_recv()
            .expect("newest physical state");
        assert_eq!(newest.status, "lifecycle_zero_applied");
        assert_eq!(newest.requested_pwm, Some([0, 0]));
        assert_eq!(newest.objective_cost, None);
        assert!(receiver.as_receiver().try_recv().is_err());
    }

    #[cfg(all(
        feature = "record",
        feature = "actuation",
        feature = "agent-runtime",
        unix
    ))]
    #[test]
    fn dropped_newest_physical_state_is_a_typed_error() {
        let (sender, _receiver, _) = bounded_channel(
            ChannelCapacity::try_from(1_usize).expect("nonzero capacity"),
            DropPolicy::DropNewest,
        );
        assert_eq!(
            sender.try_send(physical_state_viz_message(
                "stale_nonzero",
                Some([40, 40]),
                Some(3.0),
            )),
            SendOutcome::Enqueued
        );
        let mut sender = Some(sender);

        assert_eq!(
            publish_live_navigation_viz_message(
                &mut sender,
                physical_state_viz_message("lifecycle_zero_applied", Some([0, 0]), None),
            ),
            Err(LivePhysicalStateVizPublishError::DroppedNewest)
        );
    }

    #[cfg(all(feature = "nano-agent", unix))]
    #[tokio::test]
    async fn nano_controller_owner_join_failure_is_never_hidden_by_operation_failure() {
        let task = tokio::spawn(async {
            panic!("synthetic controller-owner task panic");
            #[allow(unreachable_code)]
            Ok::<(), V2ControllerOwnerTerminationError>(())
        });
        let controller = task.await;
        let operation: Result<(), Box<dyn std::error::Error>> =
            Err(std::io::Error::other("synthetic operation failure").into());

        let error = finish_nano_controller_owner(operation, controller)
            .expect_err("both failures must be returned");
        let combined = error
            .downcast_ref::<NanoOperationAndControllerOwnerError>()
            .expect("typed combined error");
        assert_eq!(
            combined.operation.to_string(),
            "synthetic operation failure"
        );
        assert!(combined.controller.to_string().contains("task join failed"));
    }

    #[cfg(all(feature = "nano-agent", unix))]
    #[tokio::test]
    async fn nano_controller_owner_exit_guard_clears_running_while_unwinding_a_task_panic() {
        let running = Arc::new(AtomicBool::new(true));
        let task_running = Arc::clone(&running);
        let task = tokio::spawn(async move {
            let _guard = super::NanoControllerOwnerExitGuard::new(task_running);
            panic!("synthetic guarded controller-owner panic");
        });

        assert!(task.await.expect_err("task must panic").is_panic());
        assert!(
            !running.load(Ordering::SeqCst),
            "controller-owner panic must synchronously revoke the shared run flag"
        );
    }

    #[cfg(all(feature = "nano-agent", unix))]
    #[test]
    fn nano_controller_owner_clean_join_preserves_the_operation_result() {
        assert!(
            finish_nano_controller_owner(Ok(()), Ok(Ok(()))).is_ok(),
            "clean operation and owner are clean"
        );
        let error = finish_nano_controller_owner(
            Err(std::io::Error::other("operation").into()),
            Ok(Ok(())),
        )
        .expect_err("operation error remains authoritative");
        assert_eq!(error.to_string(), "operation");
    }

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
    use oak_sys::{
        DepthAlignment, Intrinsics as OakIntrinsics, OakCameraSocket, OakEepromCalibrationEvidence,
    };
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

    #[cfg(feature = "record")]
    fn oak_eeprom_calibration() -> OakEepromCalibrationEvidence {
        OakEepromCalibrationEvidence::try_new(
            OakCameraSocket::CameraB,
            OakCameraSocket::CameraC,
            [
                [1.0, 0.0, 0.0, 0.01],
                [0.0, 1.0, 0.0, -0.02],
                [0.0, 0.0, 1.0, 0.03],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        )
        .expect("valid test EEPROM calibration")
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
    fn navigation_dataset_publication_uses_only_authoritative_inputs() {
        assert!(navigation_dataset_may_publish(false, true));
        assert!(!navigation_dataset_may_publish(true, true));
        assert!(!navigation_dataset_may_publish(false, false));
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
            usb_requested_maximum: oak_sys::UsbTransportSpeed::SuperPlus,
            usb_required_minimum: oak_sys::UsbTransportSpeed::Super,
            usb_observed: oak_sys::UsbTransportSpeed::Super,
            depthai_sdk_version: "3.6.1".to_owned(),
            depthai_sdk_commit: "commit-abc".to_owned(),
            embedded_device_artifact_version: "device-1".to_owned(),
            embedded_bootloader_artifact_version: "bootloader-1".to_owned(),
        };
        assert_eq!(
            provenance.dataset_device_label(),
            "OAK-D mxid=mxid-123 usb_requested_maximum=SUPER_PLUS usb_required_minimum=SUPER usb_observed=SUPER depthai_sdk=3.6.1 depthai_commit=commit-abc embedded_device=device-1 embedded_bootloader=bootloader-1 timestamp=device_exposure_midpoint"
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
    fn rectified_calibration_uses_the_common_rectified_left_projection() {
        let left = oak_intrinsics(400.0, 640, 480);
        let right = oak_intrinsics(402.0, 640, 480);
        let oak_eeprom = oak_eeprom_calibration();
        let stereo_only = build_calibration(left, right, 0.075, None, true);
        assert!(
            stereo_only.oak_eeprom.is_none(),
            "stereo-only recording must not require or invent IMU EEPROM evidence"
        );
        let calibration = build_calibration(left, right, 0.075, Some(oak_eeprom), true);

        assert_eq!(calibration.left.fx, left.fx());
        assert_eq!(calibration.left.fy, left.fy());
        assert_eq!(calibration.left.cx, left.cx());
        assert_eq!(calibration.left.cy, left.cy());
        assert_eq!(
            (calibration.left.width, calibration.left.height),
            (left.width(), left.height())
        );
        assert_eq!(calibration.right.fx, left.fx());
        assert_eq!(calibration.right.fy, left.fy());
        assert_eq!(calibration.right.cx, left.cx());
        assert_eq!(calibration.right.cy, left.cy());
        assert_eq!(calibration.right.width, left.width());
        assert_eq!(calibration.right.height, left.height());
        assert_eq!(calibration.baseline_m, 0.075);
        assert!(calibration.rectified);
        assert_eq!(
            calibration.oak_eeprom,
            Some(DatasetOakEepromCalibrationEvidence {
                stereo_left_camera_socket: OakCalibrationCameraSocket::CameraB,
                stereo_right_camera_socket: OakCalibrationCameraSocket::CameraC,
                imu_to_camera_b_m: oak_eeprom.imu_to_camera_b_m(),
                stereo_left_rectification_rotation_raw: oak_eeprom
                    .stereo_left_rectification_rotation_raw(),
            })
        );
    }

    #[cfg(feature = "record")]
    #[test]
    fn unrectified_calibration_retains_each_camera_projection() {
        let left = oak_intrinsics(400.0, 640, 480);
        let right = oak_intrinsics(402.0, 640, 480);
        let calibration = build_calibration(left, right, 0.075, None, false);

        assert_eq!(calibration.left.fx, left.fx());
        assert_eq!(calibration.right.fx, right.fx());
        assert_ne!(calibration.left.fx, calibration.right.fx);
        assert!(!calibration.rectified);
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
            let _guard = LiveThreadExitGuard::new(Arc::clone(&running));
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

    #[cfg(all(feature = "record", feature = "actuation"))]
    #[test]
    fn live_worker_compatibility_selection_cannot_create_a_production_owner() {
        assert!(matches!(
            LiveNavigationWorkerMotion::compatibility(None),
            LiveNavigationWorkerMotion::Compatibility(actuation_config)
                if actuation_config.is_none()
        ));
    }

    #[cfg(feature = "record")]
    #[test]
    fn live_navigation_cleanup_retains_every_independent_failure() {
        let result = combine_live_navigation_failures(vec![
            LiveNavigationWorkerError::TickSequenceExhausted,
            LiveNavigationWorkerError::HostClock(HostMonotonicRangeError {
                elapsed_ns: u128::from(u64::MAX) + 1,
            }),
        ]);

        assert!(matches!(
            result,
            Err(LiveNavigationWorkerError::Multiple { failures })
                if failures.len() == 2
                    && matches!(
                        failures[0],
                        LiveNavigationWorkerError::TickSequenceExhausted
                    )
                    && matches!(
                        failures[1],
                        LiveNavigationWorkerError::HostClock(_)
                    )
        ));
    }

    #[cfg(all(
        feature = "record",
        feature = "actuation",
        feature = "agent-runtime",
        unix
    ))]
    #[test]
    fn production_socket_cleanup_requires_removal_of_the_created_inode() {
        assert!(
            abnormal_production_socket_exit(AgentControlSocketTaskExit::Shutdown {
                cleanup: AgentControlSocketCleanupOutcome::RemovedCreatedSocket,
            })
            .is_none()
        );
        assert!(matches!(
            abnormal_production_socket_exit(AgentControlSocketTaskExit::Shutdown {
                cleanup: AgentControlSocketCleanupOutcome::AlreadyAbsent,
            }),
            Some(LiveNavigationWorkerError::ProductionSocketExit { .. })
        ));
    }

    #[cfg(all(feature = "nano-agent", unix))]
    #[test]
    fn nano_stream_epoch_uses_bounded_nonzero_entropy() {
        let expected = 42_u64;
        let mut bytes = 0_u64.to_ne_bytes().to_vec();
        bytes.extend_from_slice(&expected.to_ne_bytes());
        let mut entropy = std::io::Cursor::new(bytes);
        assert_eq!(
            fresh_nano_stream_epoch_from(&mut entropy, 2)
                .expect("second OS-random candidate is nonzero")
                .get(),
            expected
        );

        let mut all_zero = std::io::Cursor::new(vec![
            0_u8;
            MAX_NANO_STREAM_EPOCH_ATTEMPTS
                * std::mem::size_of::<u64>()
        ]);
        assert!(matches!(
            fresh_nano_stream_epoch_from(&mut all_zero, MAX_NANO_STREAM_EPOCH_ATTEMPTS),
            Err(NanoStreamEpochError::NonzeroCandidateExhausted {
                attempts: MAX_NANO_STREAM_EPOCH_ATTEMPTS
            })
        ));
    }

    #[cfg(all(feature = "nano-agent", unix))]
    #[test]
    fn nano_agent_cli_owns_the_three_explicit_launch_roots() {
        let cli = Cli::try_parse_from([
            "kiko-slam",
            "nano-agent",
            "--deployment-root",
            "/opt/kiko/deployment",
            "--launch-config",
            "nano-agent-launch-v4.json",
            "--state-root",
            "/var/lib/kiko-nano-agent",
        ])
        .expect("production command has every explicit launch boundary");
        let Command::NanoAgent(args) = cli.command else {
            panic!("expected Nano agent command");
        };
        assert_eq!(args.deployment_root, Path::new("/opt/kiko/deployment"));
        assert_eq!(args.launch_config, "nano-agent-launch-v4.json");
        assert_eq!(args.state_root, Path::new("/var/lib/kiko-nano-agent"));
    }

    #[cfg(all(feature = "nano-attended-navigation-trial", unix))]
    #[test]
    fn attended_navigation_cli_has_no_flag_bypass_for_physical_claims() {
        let cli = Cli::try_parse_from([
            "kiko-slam",
            "nano-attended-navigation-trial",
            "--deployment-root",
            "/opt/kiko/deployment",
            "--launch-config",
            "commissioning/attended-navigation-launch-v1.json",
            "--state-root",
            "/var/lib/kiko-nano-attended-trial",
        ])
        .expect("attended command has only immutable deployment and state boundaries");
        let Command::NanoAttendedNavigationTrial(args) = cli.command else {
            panic!("expected attended navigation command");
        };
        assert_eq!(args.deployment_root, Path::new("/opt/kiko/deployment"));
        assert_eq!(
            args.launch_config,
            "commissioning/attended-navigation-launch-v1.json"
        );
        assert_eq!(
            args.state_root,
            Path::new("/var/lib/kiko-nano-attended-trial")
        );

        for forbidden in [
            "--wheels-attached",
            "--motion-area-clear",
            "--operator-attending",
            "--power-cut-reachable",
            "--maximum-pwm-percent",
        ] {
            assert!(
                Cli::try_parse_from([
                    "kiko-slam",
                    "nano-attended-navigation-trial",
                    "--deployment-root",
                    "/opt/kiko/deployment",
                    "--launch-config",
                    "commissioning/attended-navigation-launch-v1.json",
                    "--state-root",
                    "/var/lib/kiko-nano-attended-trial",
                    forbidden,
                ])
                .is_err(),
                "physical claim or authority override {forbidden} must not exist"
            );
        }
    }

    #[cfg(all(feature = "nano-attended-navigation-trial", unix))]
    #[test]
    fn attended_navigation_stops_base_before_releasing_accessories() {
        assert!(
            NanoLiveMotionKind::AttendedNavigationTrial
                .requires_navigation_stop_before_accessory_release()
        );
    }

    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    #[test]
    fn qualification_keeps_accessory_hold_until_navigation_stop() {
        assert!(
            NanoLiveMotionKind::WheelsOffQualification
                .requires_navigation_stop_before_accessory_release()
        );
        assert!(
            !NanoLiveMotionKind::Compatibility.requires_navigation_stop_before_accessory_release()
        );
        assert!(
            !NanoLiveMotionKind::Production.requires_navigation_stop_before_accessory_release()
        );
    }

    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    #[test]
    fn qualification_cli_has_no_physical_attestation_flag_or_environment_bypass() {
        let cli = Cli::try_parse_from([
            "kiko-slam",
            "nano-wheels-off-qualification",
            "--deployment-root",
            "/opt/kiko/deployment",
            "--launch-config",
            "nano-wheels-off-qualification-launch-v3.json",
            "--state-root",
            "/var/lib/kiko-nano-qualification",
        ])
        .expect("qualification command has only deployment and state boundaries");
        let Command::NanoWheelsOffQualification(args) = cli.command else {
            panic!("expected wheels-off qualification command");
        };
        assert_eq!(args.deployment_root, Path::new("/opt/kiko/deployment"));
        assert_eq!(
            args.launch_config,
            "nano-wheels-off-qualification-launch-v3.json"
        );
        assert_eq!(
            args.state_root,
            Path::new("/var/lib/kiko-nano-qualification")
        );
        assert_eq!(args.fault_injection, None);

        assert!(
            Cli::try_parse_from([
                "kiko-slam",
                "nano-wheels-off-qualification",
                "--deployment-root",
                "/opt/kiko/deployment",
                "--launch-config",
                "nano-wheels-off-qualification-launch-v3.json",
                "--state-root",
                "/var/lib/kiko-nano-qualification",
                "--wheels-removed",
            ])
            .is_err()
        );
        for forbidden in [
            "--motor-power-disconnected",
            "--motor-power-reconnected",
            "--power-cut-reachable",
        ] {
            assert!(
                Cli::try_parse_from([
                    "kiko-slam",
                    "nano-wheels-off-qualification",
                    "--deployment-root",
                    "/opt/kiko/deployment",
                    "--launch-config",
                    "nano-wheels-off-qualification-launch-v4.json",
                    "--state-root",
                    "/var/lib/kiko-nano-qualification",
                    forbidden,
                ])
                .is_err(),
                "physical claim flag {forbidden} must not exist"
            );
        }

        let injected = Cli::try_parse_from([
            "kiko-slam",
            "nano-wheels-off-qualification",
            "--deployment-root",
            "/opt/kiko/qualification",
            "--launch-config",
            "nano-wheels-off-qualification-launch-v4.json",
            "--state-root",
            "/var/lib/kiko-nano-qualification",
            "--fault-injection",
            "partial-uart-record-on-first-nonzero-command",
        ])
        .expect("closed qualification fault declaration");
        let Command::NanoWheelsOffQualification(injected) = injected.command else {
            panic!("expected wheels-off qualification command");
        };
        assert_eq!(
            injected.fault_injection,
            Some(
                kiko_slam::navigation::WheelsOffQualificationFaultInjection::PartialUartRecordOnFirstNonzeroCommand
            )
        );

        assert!(
            Cli::try_parse_from([
                "kiko-slam",
                "nano-wheels-off-qualification",
                "--deployment-root",
                "/opt/kiko/qualification",
                "--launch-config",
                "nano-wheels-off-qualification-launch-v4.json",
                "--state-root",
                "/var/lib/kiko-nano-qualification",
                "--fault-injection",
                "partial-uart-record-on-first-nonzero-command=3",
            ])
            .is_err()
        );
        assert!(
            Cli::try_parse_from([
                "kiko-slam",
                "nano-agent",
                "--deployment-root",
                "/opt/kiko/deployment",
                "--launch-config",
                "nano-agent-launch-v4.json",
                "--state-root",
                "/var/lib/kiko-nano-agent",
                "--fault-injection",
                "partial-uart-record-on-first-nonzero-command",
            ])
            .is_err()
        );
    }

    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    #[test]
    fn qualification_tty_line_parser_is_exact_bounded_and_crlf_aware() {
        let mut exact = std::io::Cursor::new(b"WHEELS REMOVED\r\nnext\n".to_vec());
        assert_eq!(
            read_bounded_tty_line(&mut exact).expect("bounded CRLF line"),
            "WHEELS REMOVED"
        );
        assert_eq!(
            read_bounded_tty_line(&mut exact).expect("second buffered line"),
            "next"
        );

        let mut too_long =
            std::io::Cursor::new(vec![b'x'; MAX_QUALIFICATION_ATTESTATION_LINE_BYTES + 1]);
        assert!(matches!(
            read_bounded_tty_line(&mut too_long),
            Err(AttendedWheelsOffAttestationError::LineTooLong {
                maximum_bytes: MAX_QUALIFICATION_ATTESTATION_LINE_BYTES
            })
        ));

        let mut invalid = std::io::Cursor::new(vec![0xff, b'\n']);
        assert!(matches!(
            read_bounded_tty_line(&mut invalid),
            Err(AttendedWheelsOffAttestationError::InvalidUtf8)
        ));
    }

    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    #[test]
    fn integrated_attestation_readiness_requires_every_current_evidence_class() {
        let cases = [
            (
                [false, true, true, true, true],
                WheelsOffQualificationAttestationReadinessBlocker::FreshVisualDepthImuUnavailable,
            ),
            (
                [true, false, true, true, true],
                WheelsOffQualificationAttestationReadinessBlocker::AccessoryNotReady,
            ),
            (
                [true, true, false, true, true],
                WheelsOffQualificationAttestationReadinessBlocker::PublishedOccupancyRevisionUnavailable,
            ),
            (
                [true, true, true, false, true],
                WheelsOffQualificationAttestationReadinessBlocker::CoordinatorMotionStartNotReady,
            ),
            (
                [true, true, true, true, false],
                WheelsOffQualificationAttestationReadinessBlocker::NavigationVisualizationNotAccepted,
            ),
        ];
        for (evidence, expected) in cases {
            assert_eq!(
                classify_wheels_off_qualification_attestation_readiness(
                    evidence[0],
                    evidence[1],
                    evidence[2],
                    evidence[3],
                    evidence[4],
                ),
                Err(expected)
            );
        }
        assert_eq!(
            classify_wheels_off_qualification_attestation_readiness(true, true, true, true, true,),
            Ok(())
        );
    }

    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    #[test]
    fn cancellation_after_fresh_challenge_prevents_prompt_output() {
        let cancellation = AtomicBool::new(false);
        let mut challenges = CancellingWheelsOffChallenge {
            cancellation: &cancellation,
        };
        let mut input = std::io::Cursor::new(Vec::<u8>::new());
        let mut output = Vec::new();
        let mut terminal = BufferedFreshAttendedMotionTerminal {
            input: &mut input,
            output: &mut output,
        };

        assert!(matches!(
            prompt_fresh_attended_motion_phrase(
                &mut terminal,
                &mut challenges,
                &cancellation,
                "synthetic boundary",
                "SYNTHETIC PHRASE",
            ),
            Err(AttendedWheelsOffAttestationError::Interrupted)
        ));
        assert!(output.is_empty());
    }

    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    #[test]
    fn process_shutdown_after_valid_reply_discards_attestation_before_enablement() {
        let attestation = test_wheels_off_attestation();
        let (reply_ready_tx, reply_ready_rx) = std::sync::mpsc::sync_channel(1);
        let worker = FreshAttendedMotionAttestationWorker::spawn_with(
            test_wheels_off_preflight(),
            move |_preflight, _cancellation| {
                reply_ready_tx.send(()).expect("reply readiness receiver");
                Ok(attestation)
            },
        )
        .expect("test attestation worker");
        reply_ready_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("valid reply constructed");
        let mut gate = FreshAttendedMotionAttestationGate::WaitingForOperator(worker);
        let process_running = AtomicBool::new(false);

        assert_eq!(
            gate.advance_after_read_only_runtime_tick(&process_running)
                .expect("shutdown closes the gate"),
            FreshAttendedMotionAttestationWorkerPoll::Completed
        );
        assert_eq!(
            gate.closure(),
            Some(FreshAttendedMotionAttestationClosure::ProcessNotRunning)
        );
    }

    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    #[test]
    fn pending_attestation_poll_is_nonblocking_and_shutdown_joins_its_worker() {
        let (started_tx, started_rx) = std::sync::mpsc::sync_channel(1);
        let worker = FreshAttendedMotionAttestationWorker::spawn_with(
            test_wheels_off_preflight(),
            move |_preflight, cancellation| {
                started_tx.send(()).expect("test start receiver");
                while !cancellation.load(Ordering::Acquire) {
                    std::thread::sleep(Duration::from_millis(1));
                }
                Err(FreshAttendedMotionAttestationError::Terminal(
                    AttendedWheelsOffAttestationError::Interrupted,
                ))
            },
        )
        .expect("test attestation worker");
        started_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("worker entered its pending dialog");
        let mut gate = FreshAttendedMotionAttestationGate::WaitingForOperator(worker);
        let process_running = AtomicBool::new(true);

        assert_eq!(
            gate.advance_after_read_only_runtime_tick(&process_running)
                .expect("pending poll"),
            FreshAttendedMotionAttestationWorkerPoll::Pending
        );
        let (sensor_tx, sensor_rx) = crossbeam_channel::bounded(4);
        for stream in ["visual", "depth", "imu", "map"] {
            sensor_tx.send(stream).expect("bounded sensor fixture");
        }
        assert_eq!(
            sensor_rx.try_iter().collect::<Vec<_>>(),
            ["visual", "depth", "imu", "map"],
            "pending operator input does not block the caller's sensor work"
        );
        assert_eq!(
            gate.cancel_and_join().expect("bounded cancellation join"),
            FreshAttendedMotionAttestationWorkerShutdown::Cancelled
        );
    }

    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    #[test]
    fn readiness_loss_during_operator_delay_cancels_and_closes_the_gate() {
        let (started_tx, started_rx) = std::sync::mpsc::sync_channel(1);
        let worker = FreshAttendedMotionAttestationWorker::spawn_with(
            test_wheels_off_preflight(),
            move |_preflight, cancellation| {
                started_tx.send(()).expect("test start receiver");
                while !cancellation.load(Ordering::Acquire) {
                    std::thread::sleep(Duration::from_millis(1));
                }
                Err(FreshAttendedMotionAttestationError::Terminal(
                    AttendedWheelsOffAttestationError::Interrupted,
                ))
            },
        )
        .expect("test attestation worker");
        started_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("worker entered its pending dialog");
        let mut gate = FreshAttendedMotionAttestationGate::WaitingForOperator(worker);
        let blocker =
            WheelsOffQualificationAttestationReadinessBlocker::CoordinatorMotionStartNotReady;

        assert_eq!(
            gate.close_without_enablement(FreshAttendedMotionAttestationClosure::ReadinessLost(
                blocker
            ),)
                .expect("readiness loss joins the worker"),
            FreshAttendedMotionAttestationWorkerShutdown::Cancelled
        );
        assert_eq!(
            gate.closure(),
            Some(FreshAttendedMotionAttestationClosure::ReadinessLost(
                blocker
            ))
        );
    }

    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    #[test]
    fn shutdown_or_safety_stop_before_the_first_read_only_cycle_spawns_no_worker() {
        let profile =
            kiko_slam::navigation::WheelsOffQualificationControlProfile::parse(30, 10, 250)
                .expect("test profile");
        let (console, _receiver) = kiko_slam::navigation::wheels_off_qualification_console(profile);
        let input = FreshAttendedMotionAttestationInput {
            preflight: test_wheels_off_preflight(),
            console,
            process_running: Arc::new(AtomicBool::new(true)),
        };
        let mut gate = FreshAttendedMotionAttestationGate::AwaitingReadOnlyCycle(input);

        assert_eq!(
            gate.close_without_enablement(
                FreshAttendedMotionAttestationClosure::SoftwareSafetyStopLatched,
            )
            .expect("no worker exists to cancel"),
            FreshAttendedMotionAttestationWorkerShutdown::AlreadyJoined
        );
        assert!(gate.is_closed());
        let process_running = AtomicBool::new(true);
        assert_eq!(
            gate.advance_after_read_only_runtime_tick(&process_running)
                .expect("closed gate remains terminal"),
            FreshAttendedMotionAttestationWorkerPoll::Completed
        );
    }

    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    #[test]
    fn software_stop_or_owner_loss_cancels_a_waiting_attestation_without_a_tick() {
        let profile =
            kiko_slam::navigation::WheelsOffQualificationControlProfile::parse(30, 10, 250)
                .expect("test profile");
        let (console, _receiver) = kiko_slam::navigation::wheels_off_qualification_console(profile);
        let mut snapshot = console.snapshot();
        snapshot.frontend_state =
            kiko_slam::navigation::WheelsOffQualificationFrontendState::Connected;
        snapshot.runtime_ingress_state =
            kiko_slam::navigation::WheelsOffQualificationRuntimeIngressState::Connected;

        assert!(!fresh_motion_attestation_must_cancel(
            false, true, &snapshot
        ));
        snapshot.software_safety_stop_latched = true;
        assert!(fresh_motion_attestation_must_cancel(false, true, &snapshot));
        snapshot.software_safety_stop_latched = false;
        snapshot.frontend_state =
            kiko_slam::navigation::WheelsOffQualificationFrontendState::Disconnected;
        assert!(fresh_motion_attestation_must_cancel(false, true, &snapshot));
        snapshot.frontend_state =
            kiko_slam::navigation::WheelsOffQualificationFrontendState::Connected;
        assert!(fresh_motion_attestation_must_cancel(true, true, &snapshot));
        assert!(fresh_motion_attestation_must_cancel(
            false, false, &snapshot
        ));
    }

    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    #[test]
    fn completed_attestation_gate_is_one_way_and_cannot_replay_its_token() {
        let attestation = test_wheels_off_attestation();
        let worker = FreshAttendedMotionAttestationWorker::spawn_with(
            test_wheels_off_preflight(),
            move |_preflight, _cancellation| Ok(attestation),
        )
        .expect("test attestation worker");
        let mut gate = FreshAttendedMotionAttestationGate::WaitingForOperator(worker);

        assert_eq!(
            poll_test_attestation_gate(&mut gate).expect("successful worker"),
            FreshAttendedMotionAttestationWorkerPoll::Ready(attestation)
        );
        assert_eq!(
            gate.advance_after_read_only_runtime_tick(&AtomicBool::new(true))
                .expect("completed gate remains terminal"),
            FreshAttendedMotionAttestationWorkerPoll::Completed
        );
    }

    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    #[test]
    fn attestation_eof_remains_a_typed_worker_failure() {
        let worker = FreshAttendedMotionAttestationWorker::spawn_with(
            test_wheels_off_preflight(),
            |_preflight, _cancellation| {
                Err(FreshAttendedMotionAttestationError::Terminal(
                    AttendedWheelsOffAttestationError::EndOfInput,
                ))
            },
        )
        .expect("test attestation worker");
        let mut gate = FreshAttendedMotionAttestationGate::WaitingForOperator(worker);

        assert!(matches!(
            poll_test_attestation_gate(&mut gate),
            Err(FreshAttendedMotionAttestationWorkerError::Dialog(
                FreshAttendedMotionAttestationError::Terminal(
                    AttendedWheelsOffAttestationError::EndOfInput
                )
            ))
        ));
    }

    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    #[test]
    fn attestation_worker_drop_cancels_and_joins_before_returning() {
        let joined = Arc::new(AtomicBool::new(false));
        let worker_joined = Arc::clone(&joined);
        let (started_tx, started_rx) = std::sync::mpsc::sync_channel(1);
        let worker = FreshAttendedMotionAttestationWorker::spawn_with(
            test_wheels_off_preflight(),
            move |_preflight, cancellation| {
                started_tx.send(()).expect("test start receiver");
                while !cancellation.load(Ordering::Acquire) {
                    std::thread::sleep(Duration::from_millis(1));
                }
                worker_joined.store(true, Ordering::Release);
                Err(FreshAttendedMotionAttestationError::Terminal(
                    AttendedWheelsOffAttestationError::Interrupted,
                ))
            },
        )
        .expect("test attestation worker");
        started_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("worker entered its pending dialog");

        drop(worker);
        assert!(joined.load(Ordering::Acquire));
    }

    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    #[test]
    fn qualification_motor_power_transition_requires_every_exact_tty_phrase_in_order() {
        let mut preflight_challenges = FixedWheelsOffChallenges::from_bytes([1, 2, 3, 4]);
        let mut preflight_input = std::io::Cursor::new(
            [
                challenged_wheels_off_phrase("WHEELS REMOVED", 1),
                challenged_wheels_off_phrase("HEAD SUPPORTED", 2),
                challenged_wheels_off_phrase("MOTOR POWER PHYSICALLY DISCONNECTED", 3),
                challenged_wheels_off_phrase("POWER CUT REACHABLE", 4),
                String::new(),
            ]
            .join("\n")
            .into_bytes(),
        );
        let mut preflight_output = Vec::new();
        let preflight = read_attended_wheels_off_preflight(
            &mut preflight_input,
            &mut preflight_output,
            &mut preflight_challenges,
        )
        .expect("all pre-device physical claims are fresh and exact");
        let preflight_prompt =
            String::from_utf8(preflight_output).expect("static prompts are UTF-8");
        let disconnected_prompt = preflight_prompt
            .find("MOTOR POWER PHYSICALLY DISCONNECTED")
            .expect("preflight asks for a physical disconnect");
        let cut_prompt = preflight_prompt
            .find("POWER CUT REACHABLE")
            .expect("preflight asks for an independent cut");
        assert!(
            disconnected_prompt < cut_prompt,
            "motor power is disconnected before the final pre-device precondition"
        );

        let mut readiness_challenges = FixedWheelsOffChallenges::from_bytes([5, 6]);
        let mut readiness_input = std::io::Cursor::new(
            [
                challenged_wheels_off_phrase(
                    "MOTOR POWER REMAINED PHYSICALLY DISCONNECTED THROUGH SETUP",
                    5,
                ),
                challenged_wheels_off_phrase(
                    "MOTOR POWER RECONNECTED WHEELS OFF HEAD SUPPORTED POWER CUT READY",
                    6,
                ),
                String::new(),
            ]
            .join("\n")
            .into_bytes(),
        );
        let mut readiness_output = Vec::new();
        read_fresh_attended_motion_attestation(
            preflight,
            &mut readiness_input,
            &mut readiness_output,
            &mut readiness_challenges,
        )
        .expect("fresh disconnected-through-setup and reconnection claims are exact");
        let readiness_prompt =
            String::from_utf8(readiness_output).expect("static prompts are UTF-8");
        let remained_disconnected = readiness_prompt
            .find("MOTOR POWER REMAINED PHYSICALLY DISCONNECTED THROUGH SETUP")
            .expect("readiness first reconfirms the setup state");
        let reconnected = readiness_prompt
            .find("MOTOR POWER RECONNECTED WHEELS OFF HEAD SUPPORTED POWER CUT READY")
            .expect("readiness then authorizes physical reconnection");
        assert!(
            remained_disconnected < reconnected,
            "reconnection cannot precede the through-setup confirmation"
        );

        let mut post_run_challenges = FixedWheelsOffChallenges::from_bytes([7]);
        let mut post_run_input = std::io::Cursor::new(
            format!(
                "{}\n",
                challenged_wheels_off_phrase("MOTOR POWER PHYSICALLY DISCONNECTED", 7)
            )
            .into_bytes(),
        );
        let mut post_run_output = Vec::new();
        read_post_run_motor_power_disconnected(
            &mut post_run_input,
            &mut post_run_output,
            &mut post_run_challenges,
        )
        .expect("post-run physical disconnect is fresh and exact");
    }

    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    #[test]
    fn qualification_rejects_the_old_motion_phrase_and_missing_power_disconnect() {
        let mut missing_challenges = FixedWheelsOffChallenges::from_bytes([1, 2, 3]);
        let mut missing_disconnect = std::io::Cursor::new(
            [
                challenged_wheels_off_phrase("WHEELS REMOVED", 1),
                challenged_wheels_off_phrase("HEAD SUPPORTED", 2),
                challenged_wheels_off_phrase("POWER CUT REACHABLE", 3),
                String::new(),
            ]
            .join("\n")
            .into_bytes(),
        );
        assert!(matches!(
            read_attended_wheels_off_preflight(
                &mut missing_disconnect,
                &mut Vec::new(),
                &mut missing_challenges,
            ),
            Err(AttendedWheelsOffAttestationError::PhraseMismatch {
                expected
            }) if expected
                == challenged_wheels_off_phrase("MOTOR POWER PHYSICALLY DISCONNECTED", 3)
        ));

        let mut preflight_challenges = FixedWheelsOffChallenges::from_bytes([1, 2, 3, 4]);
        let mut preflight_input = std::io::Cursor::new(
            [
                challenged_wheels_off_phrase("WHEELS REMOVED", 1),
                challenged_wheels_off_phrase("HEAD SUPPORTED", 2),
                challenged_wheels_off_phrase("MOTOR POWER PHYSICALLY DISCONNECTED", 3),
                challenged_wheels_off_phrase("POWER CUT REACHABLE", 4),
                String::new(),
            ]
            .join("\n")
            .into_bytes(),
        );
        let preflight = read_attended_wheels_off_preflight(
            &mut preflight_input,
            &mut Vec::new(),
            &mut preflight_challenges,
        )
        .expect("fresh exact preflight");
        let mut readiness_challenges = FixedWheelsOffChallenges::from_bytes([5]);
        let mut old_motion_phrase =
            std::io::Cursor::new(b"WHEELS OFF HEAD SUPPORTED POWER CUT READY\n".as_slice());
        assert!(matches!(
            read_fresh_attended_motion_attestation(
                preflight,
                &mut old_motion_phrase,
                &mut Vec::new(),
                &mut readiness_challenges,
            ),
            Err(FreshAttendedMotionAttestationError::Terminal(
                AttendedWheelsOffAttestationError::PhraseMismatch {
                    expected
                }
            )) if expected
                == challenged_wheels_off_phrase(
                    "MOTOR POWER REMAINED PHYSICALLY DISCONNECTED THROUGH SETUP",
                    5,
                )
        ));
    }

    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    #[test]
    fn queued_or_replayed_physical_claims_cannot_cross_a_fresh_challenge_boundary() {
        let mut preflight_challenges = FixedWheelsOffChallenges::from_bytes([1, 2, 3, 4]);
        let mut preflight_input = std::io::Cursor::new(
            [
                challenged_wheels_off_phrase("WHEELS REMOVED", 1),
                challenged_wheels_off_phrase("HEAD SUPPORTED", 2),
                challenged_wheels_off_phrase("MOTOR POWER PHYSICALLY DISCONNECTED", 3),
                challenged_wheels_off_phrase("POWER CUT REACHABLE", 4),
                String::new(),
            ]
            .join("\n")
            .into_bytes(),
        );
        let preflight = read_attended_wheels_off_preflight(
            &mut preflight_input,
            &mut Vec::new(),
            &mut preflight_challenges,
        )
        .expect("fresh exact preflight");

        let mut transition_challenges = FixedWheelsOffChallenges::from_bytes([5, 6]);
        let mut replayed_transition = std::io::Cursor::new(
            [
                challenged_wheels_off_phrase(
                    "MOTOR POWER REMAINED PHYSICALLY DISCONNECTED THROUGH SETUP",
                    5,
                ),
                challenged_wheels_off_phrase(
                    "MOTOR POWER RECONNECTED WHEELS OFF HEAD SUPPORTED POWER CUT READY",
                    5,
                ),
                String::new(),
            ]
            .join("\n")
            .into_bytes(),
        );
        assert!(matches!(
            read_fresh_attended_motion_attestation(
                preflight,
                &mut replayed_transition,
                &mut Vec::new(),
                &mut transition_challenges,
            ),
            Err(FreshAttendedMotionAttestationError::Terminal(
                AttendedWheelsOffAttestationError::PhraseMismatch { expected }
            )) if expected
                == challenged_wheels_off_phrase(
                    "MOTOR POWER RECONNECTED WHEELS OFF HEAD SUPPORTED POWER CUT READY",
                    6,
                )
        ));

        let mut final_challenges = FixedWheelsOffChallenges::from_bytes([7]);
        let mut static_final =
            std::io::Cursor::new(b"MOTOR POWER PHYSICALLY DISCONNECTED\n".as_slice());
        assert!(matches!(
            read_post_run_motor_power_disconnected(
                &mut static_final,
                &mut Vec::new(),
                &mut final_challenges,
            ),
            Err(AttendedWheelsOffAttestationError::PhraseMismatch { expected })
                if expected
                    == challenged_wheels_off_phrase(
                        "MOTOR POWER PHYSICALLY DISCONNECTED",
                        7,
                    )
        ));
    }

    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    #[test]
    fn qualification_preserves_operation_and_post_run_disconnect_failures() {
        let operation: Result<(), Box<dyn std::error::Error>> =
            Err(std::io::Error::other("synthetic operation failure").into());
        let disconnect = Err(AttendedWheelsOffAttestationError::PhraseMismatch {
            expected: "MOTOR POWER PHYSICALLY DISCONNECTED challenge".to_owned(),
        });

        let error = finish_attended_wheels_off_qualification(operation, disconnect)
            .expect_err("both failures remain visible");
        let combined = error
            .downcast_ref::<WheelsOffQualificationAndMotorPowerDisconnectError>()
            .expect("combined failure remains typed");
        assert_eq!(
            combined.operation.to_string(),
            "synthetic operation failure"
        );
        assert!(matches!(
            &combined.motor_power_disconnect,
            AttendedWheelsOffAttestationError::PhraseMismatch {
                expected
            }
            if expected == "MOTOR POWER PHYSICALLY DISCONNECTED challenge"
        ));
    }
}
