use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::ffi::OsString;
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

/// Minimum 3D-2D correspondences needed for PnP pose estimation.
const MIN_PNP_CORRESPONDENCES: usize = 4;
/// Default maximum respawn attempts for backend and descriptor workers.
const DEFAULT_MAX_RESPAWNS: u32 = 3;
/// Minimum keyframes required for multi-frame optimization (BA or pose graph).
const MIN_OPTIMIZATION_KEYFRAMES: usize = 2;
/// Default minimum observations per map point to survive culling.
const DEFAULT_CULL_MIN_OBSERVATIONS: NonZeroUsize = NonZeroUsize::MIN;
const BACKEND_MAX_RESPAWNS_ENV: &str = "KIKO_BACKEND_MAX_RESPAWNS";
const DESCRIPTOR_MAX_RESPAWNS_ENV: &str = "KIKO_DESCRIPTOR_MAX_RESPAWNS";
const MAP_CULL_MIN_OBSERVATIONS_ENV: &str = "KIKO_MAP_CULL_MIN_OBSERVATIONS";
const TRACK_TRACE_ENV: &str = "KIKO_TRACK_TRACE";
const EIGENPLACES_MODEL_ENV: &str = "KIKO_EIGENPLACES_MODEL";

use crate::loop_closure::{
    DescriptorSource, GlobalDescriptorError, KeyframeDatabase, KeyframeDatabaseError,
    LoopApplyError, LoopCandidate, LoopClosureConfig, LoopDetectError, LoopVerificationError,
    PlaceMatch, RelocalizationCandidate, RelocalizationConfig, VerifiedLoop,
    aggregate_global_descriptor, try_match_descriptors_for_loop,
};
use crate::pose_graph::{
    EssentialEdge, EssentialEdgeKind, EssentialGraph, EssentialGraphError, PoseGraphConfig,
    PoseGraphError, PoseGraphOptimizer,
};
use crate::{
    BaResult, Detections, DiagnosticEvent, DownscaleFactor, EigenPlaces, Frame, FrameDiagnostics,
    FrameId, Keyframe, KeyframeRemovalReason, KeypointLimit, LightGlue, LocalBaConfig,
    LocalBaError, LocalBundleAdjuster, LoopClosureRejectReason, MapObservation,
    MappingSessionTransition, Matches, ObservationSet, PinholeIntrinsics, PlaceDescriptorExtractor,
    Pose, RansacConfig, Raw, RectifiedStereo, StereoPair, SuperPoint, Timestamp,
    TriangulationConfig, TriangulationError, Triangulator, Verified,
    map::{KeyframeId, MapPointId, MapSnapshot, SlamMap},
    solve_pnp_ransac,
};

use crate::inference::InferenceError;
use crossbeam_channel::{Receiver, Sender, TryRecvError, TrySendError};

#[derive(Clone, Copy, Debug)]
pub struct TrackerConfig {
    pub max_keypoints: KeypointLimit,
    pub downscale: DownscaleFactor,
    pub min_keyframe_points: usize,
    pub ransac: RansacConfig,
    pub triangulation: TriangulationConfig,
    pub keyframe_policy: KeyframePolicy,
    pub ba: LocalBaConfig,
    pub redundancy: Option<RedundancyPolicy>,
    pub backend: Option<BackendConfig>,
    pub loop_subsystem: LoopSubsystemConfig,
}

impl TrackerConfig {
    pub fn max_keypoints(&self) -> usize {
        self.max_keypoints.get()
    }

    fn loop_closure_config(&self) -> Option<LoopClosureConfig> {
        self.loop_subsystem.loop_closure()
    }

    fn global_descriptor_config(&self) -> Option<GlobalDescriptorConfig> {
        self.loop_subsystem.global_descriptor()
    }

    fn relocalization_config(&self) -> Option<RelocalizationConfig> {
        self.loop_subsystem.relocalization()
    }
}

#[derive(Clone, Copy, Debug)]
pub enum LoopSubsystemConfig {
    Disabled,
    Enabled {
        loop_closure: LoopClosureConfig,
        global_descriptor: GlobalDescriptorConfig,
        relocalization: Option<RelocalizationConfig>,
    },
}

impl LoopSubsystemConfig {
    pub fn enabled(
        loop_closure: LoopClosureConfig,
        global_descriptor: GlobalDescriptorConfig,
        relocalization: Option<RelocalizationConfig>,
    ) -> Self {
        Self::Enabled {
            loop_closure,
            global_descriptor,
            relocalization,
        }
    }

    pub fn loop_closure(self) -> Option<LoopClosureConfig> {
        match self {
            Self::Disabled => None,
            Self::Enabled { loop_closure, .. } => Some(loop_closure),
        }
    }

    pub fn global_descriptor(self) -> Option<GlobalDescriptorConfig> {
        match self {
            Self::Disabled => None,
            Self::Enabled {
                global_descriptor, ..
            } => Some(global_descriptor),
        }
    }

    pub fn relocalization(self) -> Option<RelocalizationConfig> {
        match self {
            Self::Disabled => None,
            Self::Enabled { relocalization, .. } => relocalization,
        }
    }

    pub fn is_enabled(self) -> bool {
        matches!(self, Self::Enabled { .. })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GlobalDescriptorConfig {
    queue_depth: NonZeroUsize,
}

#[derive(Debug)]
pub enum GlobalDescriptorConfigError {
    ZeroQueueDepth,
}

impl std::fmt::Display for GlobalDescriptorConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GlobalDescriptorConfigError::ZeroQueueDepth => {
                write!(f, "global descriptor queue depth must be > 0")
            }
        }
    }
}

impl std::error::Error for GlobalDescriptorConfigError {}

impl GlobalDescriptorConfig {
    pub fn new(queue_depth: usize) -> Result<Self, GlobalDescriptorConfigError> {
        let queue_depth =
            NonZeroUsize::new(queue_depth).ok_or(GlobalDescriptorConfigError::ZeroQueueDepth)?;
        Ok(Self { queue_depth })
    }

    pub fn queue_depth(&self) -> usize {
        self.queue_depth.get()
    }
}

impl Default for GlobalDescriptorConfig {
    fn default() -> Self {
        Self {
            queue_depth: NonZeroUsize::new(2).unwrap_or(NonZeroUsize::MIN),
        }
    }
}

#[derive(Debug)]
pub enum TrackerInitError {
    Environment(crate::env::EnvError),
    ZeroMapCullMinObservations { variable: &'static str },
    EmptyDescriptorModelPath { variable: &'static str },
    BackendWorkerSpawn(std::io::Error),
    DescriptorModelLoad(InferenceError),
    DescriptorWorkerSpawn(std::io::Error),
}

impl std::fmt::Display for TrackerInitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Environment(err) => write!(f, "invalid tracker environment: {err}"),
            Self::ZeroMapCullMinObservations { variable } => {
                write!(f, "{variable} must be greater than zero")
            }
            Self::EmptyDescriptorModelPath { variable } => {
                write!(
                    f,
                    "{variable} must not be empty when learned descriptors are enabled"
                )
            }
            Self::BackendWorkerSpawn(err) => {
                write!(f, "failed to spawn backend worker: {err}")
            }
            Self::DescriptorModelLoad(err) => {
                write!(f, "failed to initialize learned descriptor model: {err}")
            }
            Self::DescriptorWorkerSpawn(err) => {
                write!(f, "failed to spawn learned descriptor worker: {err}")
            }
        }
    }
}

impl std::error::Error for TrackerInitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Environment(err) => Some(err),
            Self::BackendWorkerSpawn(err) => Some(err),
            Self::DescriptorModelLoad(err) => Some(err),
            Self::DescriptorWorkerSpawn(err) => Some(err),
            Self::ZeroMapCullMinObservations { .. } | Self::EmptyDescriptorModelPath { .. } => None,
        }
    }
}

impl From<crate::env::EnvError> for TrackerInitError {
    fn from(value: crate::env::EnvError) -> Self {
        Self::Environment(value)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BackendConfig {
    queue_depth: NonZeroUsize,
}

#[derive(Debug)]
pub enum BackendConfigError {
    ZeroQueueDepth,
}

impl std::fmt::Display for BackendConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendConfigError::ZeroQueueDepth => {
                write!(f, "backend queue depth must be > 0")
            }
        }
    }
}

impl std::error::Error for BackendConfigError {}

impl BackendConfig {
    pub fn new(queue_depth: usize) -> Result<Self, BackendConfigError> {
        let queue_depth =
            NonZeroUsize::new(queue_depth).ok_or(BackendConfigError::ZeroQueueDepth)?;
        Ok(Self { queue_depth })
    }

    pub fn queue_depth(&self) -> usize {
        self.queue_depth.get()
    }
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            queue_depth: NonZeroUsize::new(2).unwrap_or(NonZeroUsize::MIN),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct KeyframePolicy {
    min_inliers: NonZeroUsize,
    parallax_px: ParallaxPx,
    min_covisibility: CovisibilityRatio,
}

#[derive(Clone, Copy, Debug)]
pub struct ParallaxPx(f32);

#[derive(Clone, Copy, Debug)]
pub struct CovisibilityRatio(f32);

#[derive(Clone, Copy, Debug)]
pub struct RedundancyPolicy {
    max_covisibility: CovisibilityRatio,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct BackendRequestId(NonZeroU64);

impl BackendRequestId {
    const FIRST: Self = Self(NonZeroU64::MIN);

    fn as_u64(self) -> u64 {
        self.0.get()
    }

    fn checked_successor(self) -> Option<Self> {
        self.as_u64()
            .checked_add(1)
            .and_then(NonZeroU64::new)
            .map(Self)
    }
}

/// Monotonic backend request identifiers for one supervisor lifetime.
///
/// Failed submissions may leave gaps. Identifiers are never reused across worker respawns, but
/// they are not globally unique across tracker instances or processes.
#[derive(Debug)]
struct BackendRequestIds {
    next: Option<BackendRequestId>,
}

impl BackendRequestIds {
    fn new() -> Self {
        Self {
            next: Some(BackendRequestId::FIRST),
        }
    }

    #[cfg(test)]
    fn from_next(next: BackendRequestId) -> Self {
        Self { next: Some(next) }
    }

    fn take_next(&mut self) -> Result<BackendRequestId, BackendRequestIdExhausted> {
        let current = self.next.ok_or(BackendRequestIdExhausted)?;
        self.next = current.checked_successor();
        Ok(current)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BackendRequestIdExhausted;

impl std::fmt::Display for BackendRequestIdExhausted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "backend request identifier domain is exhausted")
    }
}

impl std::error::Error for BackendRequestIdExhausted {}

#[derive(Debug)]
struct BackendWindow {
    keyframes: Vec<KeyframeId>,
}

#[derive(Debug)]
enum BackendWindowError {
    TooFewKeyframes { required: usize, actual: usize },
    DuplicateKeyframe { keyframe_id: KeyframeId },
}

impl std::fmt::Display for BackendWindowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendWindowError::TooFewKeyframes { required, actual } => write!(
                f,
                "backend window requires at least {required} keyframes, got {actual}"
            ),
            BackendWindowError::DuplicateKeyframe { keyframe_id } => {
                write!(f, "backend window has duplicate keyframe {keyframe_id:?}")
            }
        }
    }
}

impl std::error::Error for BackendWindowError {}

impl BackendWindow {
    fn try_new(keyframes: Vec<KeyframeId>) -> Result<Self, BackendWindowError> {
        if keyframes.len() < MIN_OPTIMIZATION_KEYFRAMES {
            return Err(BackendWindowError::TooFewKeyframes {
                required: MIN_OPTIMIZATION_KEYFRAMES,
                actual: keyframes.len(),
            });
        }
        let mut seen = HashSet::new();
        for &keyframe_id in &keyframes {
            if !seen.insert(keyframe_id) {
                return Err(BackendWindowError::DuplicateKeyframe { keyframe_id });
            }
        }
        Ok(Self { keyframes })
    }

    fn as_slice(&self) -> &[KeyframeId] {
        &self.keyframes
    }
}

#[derive(Debug, Clone)]
struct DescriptorRequest {
    keyframe_id: KeyframeId,
    source_snapshot: MapSnapshot,
    frame: Frame,
}

#[derive(Debug, Clone)]
struct DescriptorResponse {
    keyframe_id: KeyframeId,
    source_snapshot: MapSnapshot,
    descriptor: crate::loop_closure::GlobalDescriptor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DescriptorApplyDisposition {
    Applied,
    Stale,
}

#[derive(Debug, Clone)]
enum DescriptorWorkerResponse {
    Descriptor(Box<DescriptorResponse>),
    Failure {
        keyframe_id: KeyframeId,
        source_snapshot: MapSnapshot,
        error: String,
    },
    WorkerPanic {
        keyframe_id: KeyframeId,
        source_snapshot: MapSnapshot,
        message: String,
    },
}

#[derive(Debug)]
enum SubmitDescriptorError {
    QueueFull,
    Disconnected,
}

type DescriptorExtractorFactory =
    Arc<dyn Fn() -> Result<Box<dyn PlaceDescriptorExtractor>, InferenceError> + Send + Sync>;

struct DescriptorWorker {
    tx: Sender<DescriptorRequest>,
    rx: Receiver<DescriptorWorkerResponse>,
    _thread: thread::JoinHandle<()>,
}

impl DescriptorWorker {
    fn model_path_from_override(
        override_path: Option<OsString>,
    ) -> Result<PathBuf, TrackerInitError> {
        if let Some(path) = override_path {
            if path.is_empty() {
                return Err(TrackerInitError::EmptyDescriptorModelPath {
                    variable: EIGENPLACES_MODEL_ENV,
                });
            }
            return Ok(PathBuf::from(path));
        }
        Ok(PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("models")
            .join("eigenplaces.onnx"))
    }

    fn model_path() -> Result<PathBuf, TrackerInitError> {
        Self::model_path_from_override(std::env::var_os(EIGENPLACES_MODEL_ENV))
    }

    fn spawn(
        config: GlobalDescriptorConfig,
        mut extractor: Box<dyn PlaceDescriptorExtractor>,
    ) -> Result<Self, std::io::Error> {
        let (tx, req_rx) = crossbeam_channel::bounded::<DescriptorRequest>(config.queue_depth());
        let (resp_tx, rx) =
            crossbeam_channel::bounded::<DescriptorWorkerResponse>(config.queue_depth());
        let thread = thread::Builder::new()
            .name("kiko-descriptor-worker".to_string())
            .spawn(move || {
                while let Ok(request) = req_rx.recv() {
                    let processing = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        extractor.compute_descriptor(&request.frame)
                    }));
                    let response = match processing {
                        Ok(Ok(descriptor)) => {
                            DescriptorWorkerResponse::Descriptor(Box::new(DescriptorResponse {
                                keyframe_id: request.keyframe_id,
                                source_snapshot: request.source_snapshot,
                                descriptor,
                            }))
                        }
                        Ok(Err(err)) => DescriptorWorkerResponse::Failure {
                            keyframe_id: request.keyframe_id,
                            source_snapshot: request.source_snapshot,
                            error: err.to_string(),
                        },
                        Err(payload) => DescriptorWorkerResponse::WorkerPanic {
                            keyframe_id: request.keyframe_id,
                            source_snapshot: request.source_snapshot,
                            message: crate::panic_payload_to_string(payload.as_ref()),
                        },
                    };
                    let should_stop =
                        matches!(response, DescriptorWorkerResponse::WorkerPanic { .. });
                    if resp_tx.send(response).is_err() {
                        break;
                    }
                    if should_stop {
                        break;
                    }
                }
            })?;
        Ok(Self {
            tx,
            rx,
            _thread: thread,
        })
    }

    #[cfg(test)]
    fn spawn_with_extractor(
        config: GlobalDescriptorConfig,
        extractor: Box<dyn PlaceDescriptorExtractor>,
    ) -> Result<Self, std::io::Error> {
        Self::spawn(config, extractor)
    }

    fn submit(&self, request: DescriptorRequest) -> Result<(), SubmitDescriptorError> {
        match self.tx.try_send(request) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(_)) => Err(SubmitDescriptorError::QueueFull),
            Err(TrySendError::Disconnected(_)) => Err(SubmitDescriptorError::Disconnected),
        }
    }

    fn try_recv(&self) -> Result<Option<DescriptorWorkerResponse>, ()> {
        match self.rx.try_recv() {
            Ok(value) => Ok(Some(value)),
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => Err(()),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DescriptorStats {
    pub submitted: u64,
    pub dropped_full: u64,
    pub dropped_disconnected: u64,
    pub applied: u64,
    pub worker_failures: u64,
    pub respawn_count: u32,
    pub panics: u64,
}

struct DescriptorSupervisor {
    worker: Option<DescriptorWorker>,
    config: GlobalDescriptorConfig,
    factory: DescriptorExtractorFactory,
    respawn_count: u32,
    max_respawns: u32,
    spawn_exhausted: bool,
}

impl DescriptorSupervisor {
    fn default_factory() -> Result<DescriptorExtractorFactory, TrackerInitError> {
        let path = DescriptorWorker::model_path()?;
        let backend = crate::InferenceBackend::auto();
        Ok(Arc::new(move || {
            EigenPlaces::new_with_backend(&path, backend)
                .map(|extractor| Box::new(extractor) as Box<dyn PlaceDescriptorExtractor>)
        }))
    }

    fn spawn_with_max_respawns(
        config: GlobalDescriptorConfig,
        max_respawns: u32,
    ) -> Result<Self, TrackerInitError> {
        Self::try_with_factory_and_max_respawns(config, Self::default_factory()?, max_respawns)
    }

    fn try_with_factory_and_max_respawns(
        config: GlobalDescriptorConfig,
        factory: DescriptorExtractorFactory,
        max_respawns: u32,
    ) -> Result<Self, TrackerInitError> {
        let worker = Self::spawn_worker(config, &factory)?;
        Ok(Self {
            worker: Some(worker),
            config,
            factory,
            respawn_count: 0,
            max_respawns,
            spawn_exhausted: false,
        })
    }

    fn spawn_worker(
        config: GlobalDescriptorConfig,
        factory: &DescriptorExtractorFactory,
    ) -> Result<DescriptorWorker, TrackerInitError> {
        let extractor = factory().map_err(TrackerInitError::DescriptorModelLoad)?;
        DescriptorWorker::spawn(config, extractor).map_err(TrackerInitError::DescriptorWorkerSpawn)
    }

    fn check_health(&mut self) {
        if self.worker.is_some() || self.spawn_exhausted {
            return;
        }
        if self.respawn_count >= self.max_respawns {
            self.spawn_exhausted = true;
            eprintln!(
                "descriptor worker reached max respawns ({}) ; using bootstrap descriptors",
                self.max_respawns
            );
            return;
        }

        eprintln!(
            "descriptor worker disconnected; respawning ({}/{})",
            self.respawn_count + 1,
            self.max_respawns
        );
        self.worker = match Self::spawn_worker(self.config, &self.factory) {
            Ok(worker) => Some(worker),
            Err(err) => {
                eprintln!("descriptor worker respawn failed: {err}");
                None
            }
        };
        self.respawn_count = self.respawn_count.saturating_add(1);
        if self.worker.is_none() && self.respawn_count >= self.max_respawns {
            self.spawn_exhausted = true;
            eprintln!("descriptor worker respawn exhausted; using bootstrap descriptors");
        }
    }

    fn submit(&mut self, request: DescriptorRequest) -> Result<(), SubmitDescriptorError> {
        if self.worker.is_none() {
            self.check_health();
        }
        let Some(worker) = self.worker.as_ref() else {
            return Err(SubmitDescriptorError::Disconnected);
        };
        let result = worker.submit(request);
        if matches!(result, Err(SubmitDescriptorError::Disconnected)) {
            self.worker = None;
            self.check_health();
        }
        result
    }

    fn try_recv(&mut self) -> Option<DescriptorWorkerResponse> {
        let worker = self.worker.as_ref()?;
        match worker.try_recv() {
            Ok(Some(response)) => {
                if matches!(response, DescriptorWorkerResponse::WorkerPanic { .. }) {
                    self.worker = None;
                    self.check_health();
                }
                Some(response)
            }
            Ok(None) => None,
            Err(()) => {
                self.worker = None;
                self.check_health();
                None
            }
        }
    }

    fn respawn_count(&self) -> u32 {
        self.respawn_count
    }

    fn has_worker(&self) -> bool {
        self.worker.is_some()
    }
}

#[derive(Debug)]
struct KeyframeEvent {
    request_id: BackendRequestId,
    source_snapshot: MapSnapshot,
    trigger_keyframe: KeyframeId,
    window: BackendWindow,
    map_snapshot: SlamMap,
    #[cfg(test)]
    force_panic: bool,
}

#[derive(Debug)]
enum KeyframeEventError {
    TriggerMissingFromWindow { keyframe_id: KeyframeId },
    MissingKeyframeInSnapshot { keyframe_id: KeyframeId },
}

impl std::fmt::Display for KeyframeEventError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KeyframeEventError::TriggerMissingFromWindow { keyframe_id } => write!(
                f,
                "backend keyframe event window does not contain trigger keyframe {keyframe_id:?}"
            ),
            KeyframeEventError::MissingKeyframeInSnapshot { keyframe_id } => write!(
                f,
                "backend keyframe event references missing snapshot keyframe {keyframe_id:?}"
            ),
        }
    }
}

impl std::error::Error for KeyframeEventError {}

impl KeyframeEvent {
    fn try_new(
        request_id: BackendRequestId,
        trigger_keyframe: KeyframeId,
        window: BackendWindow,
        map_snapshot: SlamMap,
    ) -> Result<Self, KeyframeEventError> {
        if !window.as_slice().contains(&trigger_keyframe) {
            return Err(KeyframeEventError::TriggerMissingFromWindow {
                keyframe_id: trigger_keyframe,
            });
        }
        for &keyframe_id in window.as_slice() {
            if map_snapshot.keyframe(keyframe_id).is_none() {
                return Err(KeyframeEventError::MissingKeyframeInSnapshot { keyframe_id });
            }
        }
        let source_snapshot = map_snapshot.snapshot();
        Ok(Self {
            request_id,
            source_snapshot,
            trigger_keyframe,
            window,
            map_snapshot,
            #[cfg(test)]
            force_panic: false,
        })
    }
}

#[derive(Debug)]
struct CorrectionEvent {
    request_id: BackendRequestId,
    source_snapshot: MapSnapshot,
    trigger_keyframe: KeyframeId,
    correction: BackendCorrection,
}

#[derive(Debug)]
struct BackendCorrection {
    corrected_poses: Vec<(KeyframeId, crate::WorldToCamera)>,
    corrected_landmarks: Vec<(MapPointId, crate::WorldPoint3)>,
    result: BaResult,
}

#[derive(Debug)]
enum CorrectionBuildError {
    MissingKeyframe { keyframe_id: KeyframeId },
    MissingMapPoint { point_id: MapPointId },
    Map(crate::map::MapError),
}

impl std::fmt::Display for CorrectionBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CorrectionBuildError::MissingKeyframe { keyframe_id } => {
                write!(f, "optimized map missing keyframe {keyframe_id:?}")
            }
            CorrectionBuildError::MissingMapPoint { point_id } => {
                write!(f, "optimized map missing map point {point_id:?}")
            }
            CorrectionBuildError::Map(err) => {
                write!(
                    f,
                    "optimized map lookup failed while building correction: {err}"
                )
            }
        }
    }
}

impl std::error::Error for CorrectionBuildError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Map(err) => Some(err),
            Self::MissingKeyframe { .. } | Self::MissingMapPoint { .. } => None,
        }
    }
}

impl From<crate::map::MapError> for CorrectionBuildError {
    fn from(err: crate::map::MapError) -> Self {
        Self::Map(err)
    }
}

impl CorrectionEvent {
    fn from_optimized_map(
        event: &KeyframeEvent,
        optimized_map: &SlamMap,
        result: BaResult,
    ) -> Result<Self, CorrectionBuildError> {
        let mut correction = BackendCorrection {
            corrected_poses: Vec::new(),
            corrected_landmarks: Vec::new(),
            result: result.clone(),
        };

        if matches!(
            result,
            BaResult::Converged { .. } | BaResult::MaxIterations { .. }
        ) {
            correction.corrected_poses = Vec::with_capacity(event.window.as_slice().len());
            for &keyframe_id in event.window.as_slice() {
                event
                    .map_snapshot
                    .keyframe(keyframe_id)
                    .ok_or(CorrectionBuildError::MissingKeyframe { keyframe_id })?;
                let after = optimized_map
                    .keyframe(keyframe_id)
                    .ok_or(CorrectionBuildError::MissingKeyframe { keyframe_id })?;
                correction.corrected_poses.push((keyframe_id, after.pose()));
            }

            let point_ids = collect_window_points(optimized_map, &event.window)?;
            correction.corrected_landmarks = Vec::with_capacity(point_ids.len());
            for point_id in point_ids {
                event
                    .map_snapshot
                    .point(point_id)
                    .ok_or(CorrectionBuildError::MissingMapPoint { point_id })?;
                let after = optimized_map
                    .point(point_id)
                    .ok_or(CorrectionBuildError::MissingMapPoint { point_id })?;
                correction
                    .corrected_landmarks
                    .push((point_id, after.position()));
            }
        }

        Ok(Self {
            request_id: event.request_id,
            source_snapshot: event.source_snapshot,
            trigger_keyframe: event.trigger_keyframe,
            correction,
        })
    }
}

#[derive(Debug)]
enum BackendWorkerError {
    BuildCorrection(CorrectionBuildError),
    BundleAdjustment(LocalBaError),
}

impl std::fmt::Display for BackendWorkerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendWorkerError::BuildCorrection(err) => {
                write!(f, "backend correction build failed: {err}")
            }
            BackendWorkerError::BundleAdjustment(err) => {
                write!(f, "backend bundle adjustment failed: {err}")
            }
        }
    }
}

impl std::error::Error for BackendWorkerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::BuildCorrection(err) => Some(err),
            Self::BundleAdjustment(err) => Some(err),
        }
    }
}

#[derive(Debug)]
enum BackendResponse {
    Correction(CorrectionEvent),
    WorkerPanic {
        request_id: BackendRequestId,
        source_snapshot: MapSnapshot,
    },
    Failure {
        request_id: BackendRequestId,
        source_snapshot: MapSnapshot,
        error: BackendWorkerError,
    },
}

#[derive(Clone, Copy, Debug, Default)]
pub struct BackendStats {
    pub submitted: u64,
    pub dropped_full: u64,
    pub dropped_disconnected: u64,
    pub applied: u64,
    pub stale: u64,
    pub rejected: u64,
    pub worker_failures: u64,
    pub respawn_count: u32,
    pub panics: u64,
}

#[derive(Debug)]
enum SubmitEventError {
    InvalidWindow(BackendWindowError),
    InvalidEvent(KeyframeEventError),
    RequestIdExhausted(BackendRequestIdExhausted),
    QueueFull,
    Disconnected,
}

impl std::fmt::Display for SubmitEventError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SubmitEventError::InvalidWindow(err) => write!(f, "invalid backend window: {err}"),
            SubmitEventError::InvalidEvent(err) => write!(f, "invalid backend event: {err}"),
            SubmitEventError::RequestIdExhausted(err) => write!(f, "{err}"),
            SubmitEventError::QueueFull => write!(f, "backend event queue is full"),
            SubmitEventError::Disconnected => write!(f, "backend worker is disconnected"),
        }
    }
}

impl std::error::Error for SubmitEventError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidWindow(err) => Some(err),
            Self::InvalidEvent(err) => Some(err),
            Self::RequestIdExhausted(err) => Some(err),
            Self::QueueFull | Self::Disconnected => None,
        }
    }
}

#[derive(Debug)]
enum ApplyCorrectionError {
    StaleSnapshot {
        current: MapSnapshot,
        correction: MapSnapshot,
    },
    MissingKeyframe {
        keyframe_id: KeyframeId,
    },
    MissingMapPoint {
        point_id: MapPointId,
    },
    Map(crate::map::MapError),
}

impl std::fmt::Display for ApplyCorrectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ApplyCorrectionError::StaleSnapshot {
                current,
                correction,
            } => write!(
                f,
                "stale correction: correction snapshot={correction:?}, current snapshot={current:?}"
            ),
            ApplyCorrectionError::MissingKeyframe { keyframe_id } => {
                write!(f, "correction references missing keyframe {keyframe_id:?}")
            }
            ApplyCorrectionError::MissingMapPoint { point_id } => {
                write!(f, "correction references missing map point {point_id:?}")
            }
            ApplyCorrectionError::Map(err) => write!(f, "map correction apply error: {err}"),
        }
    }
}

impl std::error::Error for ApplyCorrectionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Map(err) => Some(err),
            Self::StaleSnapshot { .. }
            | Self::MissingKeyframe { .. }
            | Self::MissingMapPoint { .. } => None,
        }
    }
}

impl From<crate::map::MapError> for ApplyCorrectionError {
    fn from(value: crate::map::MapError) -> Self {
        Self::Map(value)
    }
}

#[derive(Debug)]
struct BackendWorker {
    tx: Sender<KeyframeEvent>,
    rx: Receiver<BackendResponse>,
}

impl BackendWorker {
    fn spawn(
        config: BackendConfig,
        intrinsics: PinholeIntrinsics,
        ba_config: LocalBaConfig,
    ) -> Result<Self, std::io::Error> {
        let queue_depth = config.queue_depth();
        let (tx_req, rx_req) = crossbeam_channel::bounded::<KeyframeEvent>(queue_depth);
        // Keep backend responses bounded to prevent unbounded memory growth if
        // the tracking thread falls behind response draining.
        let (tx_resp, rx_resp) = crossbeam_channel::bounded::<BackendResponse>(queue_depth);

        thread::Builder::new()
            .name("kiko-backend".to_string())
            .spawn(move || {
                let mut ba = LocalBundleAdjuster::new(intrinsics, ba_config);
                while let Ok(event) = rx_req.recv() {
                    let request_id = event.request_id;
                    let source_snapshot = event.source_snapshot;
                    let processing = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        #[cfg(test)]
                        if event.force_panic {
                            panic!("forced backend worker panic");
                        }
                        let mut optimized_map = event.map_snapshot.clone();
                        let result = ba
                            .optimize_keyframe_window(&mut optimized_map, event.window.as_slice())
                            .map_err(BackendWorkerError::BundleAdjustment)?;
                        CorrectionEvent::from_optimized_map(&event, &optimized_map, result)
                            .map_err(BackendWorkerError::BuildCorrection)
                    }));

                    match processing {
                        Ok(Ok(correction)) => {
                            if tx_resp
                                .send(BackendResponse::Correction(correction))
                                .is_err()
                            {
                                eprintln!(
                                    "backend response channel disconnected during correction send"
                                );
                                break;
                            }
                        }
                        Ok(Err(err)) => {
                            if tx_resp
                                .send(BackendResponse::Failure {
                                    request_id,
                                    source_snapshot,
                                    error: err,
                                })
                                .is_err()
                            {
                                eprintln!(
                                    "backend response channel disconnected during failure send"
                                );
                                break;
                            }
                        }
                        Err(_) => {
                            if tx_resp
                                .send(BackendResponse::WorkerPanic {
                                    request_id,
                                    source_snapshot,
                                })
                                .is_err()
                            {
                                eprintln!(
                                    "backend response channel disconnected during panic send"
                                );
                                break;
                            }
                            break;
                        }
                    }
                }
            })?;

        Ok(Self {
            tx: tx_req,
            rx: rx_resp,
        })
    }

    fn try_submit(&self, event: KeyframeEvent) -> Result<(), SubmitEventError> {
        match self.tx.try_send(event) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(_)) => Err(SubmitEventError::QueueFull),
            Err(TrySendError::Disconnected(_)) => Err(SubmitEventError::Disconnected),
        }
    }

    fn try_recv(&self) -> Result<Option<BackendResponse>, ()> {
        match self.rx.try_recv() {
            Ok(response) => Ok(Some(response)),
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => Err(()),
        }
    }
}

#[derive(Debug)]
struct BackendSupervisor {
    worker: Option<BackendWorker>,
    config: BackendConfig,
    intrinsics: PinholeIntrinsics,
    ba_config: LocalBaConfig,
    respawn_count: u32,
    max_respawns: u32,
    spawn_exhausted: bool,
    request_ids: BackendRequestIds,
}

impl BackendSupervisor {
    fn respawn_worker(
        config: BackendConfig,
        intrinsics: PinholeIntrinsics,
        ba_config: LocalBaConfig,
    ) -> Option<BackendWorker> {
        match BackendWorker::spawn(config, intrinsics, ba_config) {
            Ok(worker) => Some(worker),
            Err(err) => {
                eprintln!("failed to spawn backend worker thread: {err}");
                None
            }
        }
    }

    fn spawn_with_max_respawns(
        config: BackendConfig,
        intrinsics: PinholeIntrinsics,
        ba_config: LocalBaConfig,
        max_respawns: u32,
    ) -> Result<Self, TrackerInitError> {
        Self::spawn_initial_with(
            config,
            intrinsics,
            ba_config,
            max_respawns,
            BackendWorker::spawn,
        )
    }

    fn spawn_initial_with<F>(
        config: BackendConfig,
        intrinsics: PinholeIntrinsics,
        ba_config: LocalBaConfig,
        max_respawns: u32,
        spawn: F,
    ) -> Result<Self, TrackerInitError>
    where
        F: FnOnce(
            BackendConfig,
            PinholeIntrinsics,
            LocalBaConfig,
        ) -> Result<BackendWorker, std::io::Error>,
    {
        let worker =
            spawn(config, intrinsics, ba_config).map_err(TrackerInitError::BackendWorkerSpawn)?;
        Ok(Self {
            worker: Some(worker),
            config,
            intrinsics,
            ba_config,
            respawn_count: 0,
            max_respawns,
            spawn_exhausted: false,
            request_ids: BackendRequestIds::new(),
        })
    }

    #[cfg(test)]
    fn with_max_respawns(
        config: BackendConfig,
        intrinsics: PinholeIntrinsics,
        ba_config: LocalBaConfig,
        max_respawns: u32,
    ) -> Self {
        Self::spawn_with_max_respawns(config, intrinsics, ba_config, max_respawns)
            .expect("spawn initial backend worker")
    }

    fn check_health(&mut self) {
        if self.worker.is_some() || self.spawn_exhausted {
            return;
        }

        if self.respawn_count >= self.max_respawns {
            self.spawn_exhausted = true;
            eprintln!(
                "backend worker reached max respawns ({}) ; disabling backend optimization",
                self.max_respawns
            );
            return;
        }

        eprintln!(
            "backend worker disconnected; respawning ({}/{})",
            self.respawn_count + 1,
            self.max_respawns
        );
        self.worker = Self::respawn_worker(self.config, self.intrinsics, self.ba_config);
        self.respawn_count = self.respawn_count.saturating_add(1);
        if self.worker.is_none() && self.respawn_count >= self.max_respawns {
            self.spawn_exhausted = true;
            eprintln!(
                "backend worker respawn exhausted after {} attempts",
                self.max_respawns
            );
        }
    }

    fn submit(&mut self, event: KeyframeEvent) -> Result<(), SubmitEventError> {
        if self.worker.is_none() {
            self.check_health();
        }
        let Some(worker) = self.worker.as_ref() else {
            return Err(SubmitEventError::Disconnected);
        };
        let result = worker.try_submit(event);
        if matches!(result, Err(SubmitEventError::Disconnected)) {
            self.worker = None;
            self.check_health();
        }
        result
    }

    fn try_recv(&mut self) -> Option<BackendResponse> {
        let response = {
            let worker = self.worker.as_ref()?;
            worker.try_recv()
        };
        match response {
            Ok(Some(BackendResponse::WorkerPanic {
                request_id,
                source_snapshot,
            })) => {
                self.worker = None;
                self.check_health();
                Some(BackendResponse::WorkerPanic {
                    request_id,
                    source_snapshot,
                })
            }
            Ok(response) => response,
            Err(()) => {
                self.worker = None;
                self.check_health();
                None
            }
        }
    }

    fn next_request_id(&mut self) -> Result<BackendRequestId, SubmitEventError> {
        if self.worker.is_none() {
            self.check_health();
        }
        if self.worker.is_none() {
            return Err(SubmitEventError::Disconnected);
        }
        self.request_ids
            .take_next()
            .map_err(SubmitEventError::RequestIdExhausted)
    }

    #[cfg(test)]
    fn shutdown(&mut self) {
        self.worker = None;
        self.spawn_exhausted = true;
    }

    fn respawn_count(&self) -> u32 {
        self.respawn_count
    }

    fn has_worker(&self) -> bool {
        self.worker.is_some()
    }
}

#[derive(Debug)]
pub enum KeyframePolicyError {
    ZeroInliers,
    NonPositiveParallax { value: f32 },
    CovisibilityOutOfRange { value: f32 },
}

#[derive(Debug)]
pub enum RedundancyPolicyError {
    CovisibilityOutOfRange { value: f32 },
}

impl std::fmt::Display for KeyframePolicyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KeyframePolicyError::ZeroInliers => write!(f, "keyframe inlier threshold must be > 0"),
            KeyframePolicyError::NonPositiveParallax { value } => {
                write!(f, "parallax threshold must be > 0 (got {value})")
            }
            KeyframePolicyError::CovisibilityOutOfRange { value } => {
                write!(f, "covisibility ratio must be within [0, 1] (got {value})")
            }
        }
    }
}

impl std::error::Error for KeyframePolicyError {}

impl std::fmt::Display for RedundancyPolicyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RedundancyPolicyError::CovisibilityOutOfRange { value } => write!(
                f,
                "redundancy covisibility must be within [0, 1] (got {value})"
            ),
        }
    }
}

impl std::error::Error for RedundancyPolicyError {}

impl KeyframePolicy {
    pub fn new(
        min_inliers: usize,
        parallax_px: f32,
        min_covisibility: f32,
    ) -> Result<Self, KeyframePolicyError> {
        let min_inliers = NonZeroUsize::new(min_inliers).ok_or(KeyframePolicyError::ZeroInliers)?;
        if !parallax_px.is_finite() || parallax_px <= 0.0 {
            return Err(KeyframePolicyError::NonPositiveParallax { value: parallax_px });
        }
        if !min_covisibility.is_finite() || !(0.0..=1.0).contains(&min_covisibility) {
            return Err(KeyframePolicyError::CovisibilityOutOfRange {
                value: min_covisibility,
            });
        }
        Ok(Self {
            min_inliers,
            parallax_px: ParallaxPx(parallax_px),
            min_covisibility: CovisibilityRatio(min_covisibility),
        })
    }

    pub fn min_inliers(&self) -> usize {
        self.min_inliers.get()
    }

    pub fn parallax_px(&self) -> f32 {
        self.parallax_px.0
    }

    pub fn min_covisibility(&self) -> f32 {
        self.min_covisibility.0
    }

    pub fn should_refresh(
        &self,
        inliers: usize,
        parallax_px: Option<f32>,
        covisibility: f32,
    ) -> bool {
        if inliers < self.min_inliers.get() {
            return true;
        }
        if let Some(parallax) = parallax_px
            && parallax > self.parallax_px.0
        {
            return true;
        }
        if covisibility < self.min_covisibility.0 {
            return true;
        }
        false
    }
}

impl RedundancyPolicy {
    pub fn new(max_covisibility: f32) -> Result<Self, RedundancyPolicyError> {
        if !max_covisibility.is_finite() || !(0.0..=1.0).contains(&max_covisibility) {
            return Err(RedundancyPolicyError::CovisibilityOutOfRange {
                value: max_covisibility,
            });
        }
        Ok(Self {
            max_covisibility: CovisibilityRatio(max_covisibility),
        })
    }

    pub fn max_covisibility(&self) -> f32 {
        self.max_covisibility.0
    }
}

#[derive(Debug)]
pub enum TrackerError {
    Inference(InferenceError),
    GlobalDescriptor(GlobalDescriptorError),
    KeyframeDatabase(KeyframeDatabaseError),
    Triangulation(TriangulationError),
    LocalBa(LocalBaError),
    Pnp(crate::PnpError),
    Map(crate::map::MapError),
    EssentialGraph(EssentialGraphError),
    PoseGraph(PoseGraphError),
    Pose(crate::PoseError),
    Transform(crate::TransformError),
    Pose64(crate::Pose64Error),
    PoseNarrowing(crate::PoseNarrowingError),
    LoopMapSnapshotMismatch {
        verified: crate::map::MapSnapshot,
        current: crate::map::MapSnapshot,
    },
    RelocalizationMapSnapshotMismatch {
        verified: crate::map::MapSnapshot,
        current: crate::map::MapSnapshot,
    },
    KeyframeRejected {
        landmarks: usize,
    },
    InvariantViolation(&'static str),
}

impl std::fmt::Display for TrackerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrackerError::Inference(err) => write!(f, "inference error: {err}"),
            TrackerError::GlobalDescriptor(err) => {
                write!(f, "global descriptor error: {err}")
            }
            TrackerError::KeyframeDatabase(err) => {
                write!(f, "keyframe database error: {err}")
            }
            TrackerError::Triangulation(err) => write!(f, "triangulation error: {err}"),
            TrackerError::LocalBa(err) => write!(f, "local BA error: {err}"),
            TrackerError::Pnp(err) => write!(f, "pnp error: {err}"),
            TrackerError::Map(err) => write!(f, "map error: {err}"),
            TrackerError::EssentialGraph(err) => write!(f, "essential graph error: {err}"),
            TrackerError::PoseGraph(err) => write!(f, "pose graph error: {err}"),
            TrackerError::Pose(err) => write!(f, "pose error: {err}"),
            TrackerError::Transform(err) => write!(f, "coordinate transform error: {err}"),
            TrackerError::Pose64(err) => write!(f, "64-bit pose error: {err}"),
            TrackerError::PoseNarrowing(err) => write!(f, "pose narrowing error: {err}"),
            TrackerError::LoopMapSnapshotMismatch { verified, current } => write!(
                f,
                "verified loop map snapshot mismatch: verified={verified:?}, current={current:?}"
            ),
            TrackerError::RelocalizationMapSnapshotMismatch { verified, current } => write!(
                f,
                "verified relocalization map snapshot mismatch: verified={verified:?}, current={current:?}"
            ),
            TrackerError::KeyframeRejected { landmarks } => {
                write!(f, "keyframe rejected: only {landmarks} landmarks")
            }
            TrackerError::InvariantViolation(message) => {
                write!(f, "tracker invariant violation: {message}")
            }
        }
    }
}

impl std::error::Error for TrackerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Inference(err) => Some(err),
            Self::GlobalDescriptor(err) => Some(err),
            Self::KeyframeDatabase(err) => Some(err),
            Self::Triangulation(err) => Some(err),
            Self::LocalBa(err) => Some(err),
            Self::Pnp(err) => Some(err),
            Self::Map(err) => Some(err),
            Self::EssentialGraph(err) => Some(err),
            Self::PoseGraph(err) => Some(err),
            Self::Pose(err) => Some(err),
            Self::Transform(err) => Some(err),
            Self::Pose64(err) => Some(err),
            Self::PoseNarrowing(err) => Some(err),
            Self::LoopMapSnapshotMismatch { .. }
            | Self::RelocalizationMapSnapshotMismatch { .. }
            | Self::KeyframeRejected { .. }
            | Self::InvariantViolation(_) => None,
        }
    }
}

impl TrackerError {
    pub fn requires_pipeline_shutdown(&self) -> bool {
        matches!(
            self,
            Self::Inference(
                InferenceError::WatchdogTimeout { .. } | InferenceError::SessionQuarantined { .. }
            )
        )
    }
}

impl From<InferenceError> for TrackerError {
    fn from(err: InferenceError) -> Self {
        TrackerError::Inference(err)
    }
}

impl From<GlobalDescriptorError> for TrackerError {
    fn from(err: GlobalDescriptorError) -> Self {
        TrackerError::GlobalDescriptor(err)
    }
}

impl From<KeyframeDatabaseError> for TrackerError {
    fn from(err: KeyframeDatabaseError) -> Self {
        TrackerError::KeyframeDatabase(err)
    }
}

impl From<TriangulationError> for TrackerError {
    fn from(err: TriangulationError) -> Self {
        TrackerError::Triangulation(err)
    }
}

impl From<LocalBaError> for TrackerError {
    fn from(err: LocalBaError) -> Self {
        TrackerError::LocalBa(err)
    }
}

impl From<crate::PnpError> for TrackerError {
    fn from(err: crate::PnpError) -> Self {
        TrackerError::Pnp(err)
    }
}

/// Publishable post-BA metrics either cover the complete claimed-inlier set or are absent.
#[derive(Clone, Copy, Debug, PartialEq)]
enum PostBaReprojectionDiagnostics {
    NotAllProjectable,
    Complete { rmse_px: f32, max_px: f32 },
}

fn post_ba_reprojection_diagnostics<'a>(
    pose: Pose,
    observations: impl ExactSizeIterator<Item = &'a crate::Observation>,
    intrinsics: PinholeIntrinsics,
) -> Result<PostBaReprojectionDiagnostics, TrackerError> {
    let claimed_inliers = observations.len();
    if claimed_inliers == 0 {
        return Err(TrackerError::InvariantViolation(
            "post-BA reprojection diagnostics require at least one claimed PnP inlier",
        ));
    }
    let metrics = crate::pnp::reprojection_metrics(&pose, observations, intrinsics);
    if metrics.projected_count() + metrics.not_in_front_count() != claimed_inliers {
        return Err(TrackerError::InvariantViolation(
            "post-BA reprojection accounting did not cover every claimed PnP inlier",
        ));
    }
    if metrics.not_in_front_count() != 0 {
        return Ok(PostBaReprojectionDiagnostics::NotAllProjectable);
    }

    let complete = metrics
        .complete_px()?
        .ok_or(TrackerError::InvariantViolation(
            "nonempty projectable PnP inlier set produced no reprojection metrics",
        ))?;
    Ok(PostBaReprojectionDiagnostics::Complete {
        rmse_px: complete.rmse_px(),
        max_px: complete.max_px(),
    })
}

impl From<crate::map::MapError> for TrackerError {
    fn from(err: crate::map::MapError) -> Self {
        TrackerError::Map(err)
    }
}

impl From<EssentialGraphError> for TrackerError {
    fn from(err: EssentialGraphError) -> Self {
        TrackerError::EssentialGraph(err)
    }
}

impl From<PoseGraphError> for TrackerError {
    fn from(err: PoseGraphError) -> Self {
        TrackerError::PoseGraph(err)
    }
}

impl From<crate::PoseError> for TrackerError {
    fn from(err: crate::PoseError) -> Self {
        TrackerError::Pose(err)
    }
}

impl From<crate::TransformError> for TrackerError {
    fn from(err: crate::TransformError) -> Self {
        TrackerError::Transform(err)
    }
}

impl From<crate::Pose64Error> for TrackerError {
    fn from(err: crate::Pose64Error) -> Self {
        TrackerError::Pose64(err)
    }
}

impl From<crate::PoseNarrowingError> for TrackerError {
    fn from(err: crate::PoseNarrowingError) -> Self {
        TrackerError::PoseNarrowing(err)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrackingHealth {
    Good,
    Degraded,
    Lost,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoseStatus {
    Current,
    Stale,
    Unavailable,
}

impl PoseStatus {
    fn is_consistent_with<T>(self, pose: Option<T>) -> bool {
        match self {
            Self::Current | Self::Stale => pose.is_some(),
            Self::Unavailable => pose.is_none(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DegradationLevel {
    Nominal,
    TrackingDegraded,
    DescriptorDown,
    BackendDown,
    Lost,
}

impl DegradationLevel {
    fn rank(self) -> u8 {
        match self {
            DegradationLevel::Nominal => 0,
            DegradationLevel::TrackingDegraded => 1,
            DegradationLevel::DescriptorDown => 2,
            DegradationLevel::BackendDown => 3,
            DegradationLevel::Lost => 4,
        }
    }

    pub fn worst(a: Self, b: Self) -> Self {
        if a.rank() >= b.rank() { a } else { b }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComponentHealth {
    Disabled,
    Alive,
    Down,
}

impl ComponentHealth {
    fn from_expected_and_alive(expected: bool, alive: bool) -> Self {
        if !expected {
            Self::Disabled
        } else if alive {
            Self::Alive
        } else {
            Self::Down
        }
    }

    pub fn is_alive(self) -> bool {
        matches!(self, Self::Alive)
    }
}

#[derive(Clone, Debug)]
pub struct SystemHealth {
    pub tracking: TrackingHealth,
    pub backend: ComponentHealth,
    pub descriptor: ComponentHealth,
    pub backend_stats: BackendStats,
    pub degradation: DegradationLevel,
}

impl SystemHealth {
    fn from_components(
        tracking: TrackingHealth,
        backend_expected: bool,
        backend_alive: bool,
        descriptor_expected: bool,
        descriptor_alive: bool,
        backend_stats: BackendStats,
    ) -> Self {
        let tracking_degradation = match tracking {
            TrackingHealth::Good => DegradationLevel::Nominal,
            TrackingHealth::Degraded => DegradationLevel::TrackingDegraded,
            TrackingHealth::Lost => DegradationLevel::Lost,
        };
        let descriptor =
            ComponentHealth::from_expected_and_alive(descriptor_expected, descriptor_alive);
        let descriptor_degradation = if descriptor == ComponentHealth::Down {
            DegradationLevel::DescriptorDown
        } else {
            DegradationLevel::Nominal
        };
        let backend = ComponentHealth::from_expected_and_alive(backend_expected, backend_alive);
        let backend_degradation = if backend == ComponentHealth::Down {
            DegradationLevel::BackendDown
        } else {
            DegradationLevel::Nominal
        };
        let degradation = DegradationLevel::worst(
            DegradationLevel::worst(tracking_degradation, descriptor_degradation),
            backend_degradation,
        );
        Self {
            tracking,
            backend,
            descriptor,
            backend_stats,
            degradation,
        }
    }
}

#[derive(Debug)]
pub struct TrackerOutput {
    /// World-to-camera pose; inspect `pose_status` before use.
    pub(crate) pose: Option<crate::WorldToCamera>,
    pub(crate) pose_status: PoseStatus,
    pub(crate) inliers: usize,
    pub(crate) keyframe: Option<Arc<Keyframe>>,
    pub(crate) stereo_matches: Option<Matches<Raw>>,
    pub(crate) frame_id: FrameId,
    pub(crate) health: SystemHealth,
    pub(crate) diagnostics: FrameDiagnostics,
    pub(crate) events: Vec<DiagnosticEvent>,
}

impl TrackerOutput {
    pub fn pose(&self) -> Option<crate::WorldToCamera> {
        self.pose
    }

    pub fn pose_status(&self) -> PoseStatus {
        self.pose_status
    }

    pub fn inliers(&self) -> usize {
        self.inliers
    }

    pub fn keyframe(&self) -> Option<&Arc<Keyframe>> {
        self.keyframe.as_ref()
    }

    pub fn stereo_matches(&self) -> Option<&Matches<Raw>> {
        self.stereo_matches.as_ref()
    }

    pub fn take_stereo_matches(&mut self) -> Option<Matches<Raw>> {
        self.stereo_matches.take()
    }

    pub fn frame_id(&self) -> FrameId {
        self.frame_id
    }

    pub fn health(&self) -> &SystemHealth {
        &self.health
    }

    pub fn diagnostics(&self) -> &FrameDiagnostics {
        &self.diagnostics
    }

    pub fn diagnostics_mut(&mut self) -> &mut FrameDiagnostics {
        &mut self.diagnostics
    }

    pub fn events(&self) -> &[DiagnosticEvent] {
        &self.events
    }

    pub fn into_status_parts(
        self,
    ) -> (
        Option<crate::WorldToCamera>,
        SystemHealth,
        FrameDiagnostics,
        Vec<DiagnosticEvent>,
    ) {
        (self.pose, self.health, self.diagnostics, self.events)
    }
}

#[derive(Debug)]
enum TrackerState {
    NeedKeyframe,
    Tracking {
        keyframe: Arc<Keyframe>,
        keyframe_id: KeyframeId,
    },
    Relocalizing(RelocalizationSession),
}

#[derive(Debug, Clone)]
enum RelocalizationPhase {
    Searching,
    Confirming {
        candidate: KeyframeId,
        confirmations: NonZeroUsize,
        pose_world: Pose,
    },
}

#[derive(Debug, Clone)]
struct RelocalizationSession {
    attempts: usize,
    phase: RelocalizationPhase,
}

#[derive(Debug)]
struct RelocalizationAttempt {
    session: RelocalizationSession,
    is_final: bool,
}

#[derive(Clone, Copy, Debug)]
enum RelocalizationEvidence {
    NoCandidate,
    Verified {
        attachment: RelocalizationAttachment,
        pose_world: Pose,
    },
}

#[derive(Debug)]
enum RelocalizationStep {
    Continue(RelocalizationSession),
    Recovered {
        attachment: RelocalizationAttachment,
        pose_world: Pose,
    },
    Exhausted,
}

#[derive(Clone, Copy, Debug)]
struct RelocalizationAttachment {
    candidate: KeyframeId,
    verified_snapshot: MapSnapshot,
    inlier_count: NonZeroUsize,
}

#[derive(Debug)]
struct SharedMatches {
    keyframe_id: KeyframeId,
    pairs: Vec<(usize, usize)>,
}

#[derive(Debug)]
enum KeyframeConnection {
    Bootstrap,
    Covisibility(SharedMatches),
    Relocalization(RelocalizationAttachment),
}

impl KeyframeConnection {
    fn shared_matches(&self) -> Option<&SharedMatches> {
        match self {
            Self::Covisibility(shared) => Some(shared),
            Self::Bootstrap | Self::Relocalization(_) => None,
        }
    }
}

#[derive(Debug)]
struct PendingLoopCandidate {
    query_kf: KeyframeId,
    detections: Arc<Detections>,
    candidates: Vec<PlaceMatch>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TrackerEnvironment {
    backend_max_respawns: u32,
    descriptor_max_respawns: u32,
    cull_min_observations: NonZeroUsize,
    trace_transitions: bool,
}

fn load_when_enabled<T, E>(
    enabled: bool,
    default: T,
    load: impl FnOnce() -> Result<Option<T>, E>,
) -> Result<T, E> {
    if enabled {
        Ok(load()?.unwrap_or(default))
    } else {
        Ok(default)
    }
}

impl TrackerEnvironment {
    fn parse(backend_enabled: bool, descriptor_enabled: bool) -> Result<Self, TrackerInitError> {
        let backend_max_respawns =
            load_when_enabled(backend_enabled, DEFAULT_MAX_RESPAWNS, || {
                crate::env::env_u32(BACKEND_MAX_RESPAWNS_ENV)
            })?;
        let descriptor_max_respawns =
            load_when_enabled(descriptor_enabled, DEFAULT_MAX_RESPAWNS, || {
                crate::env::env_u32(DESCRIPTOR_MAX_RESPAWNS_ENV)
            })?;
        Self::from_parsed(
            backend_max_respawns,
            descriptor_max_respawns,
            crate::env::env_usize(MAP_CULL_MIN_OBSERVATIONS_ENV)?,
            crate::env::env_bool(TRACK_TRACE_ENV)?,
        )
    }

    fn from_parsed(
        backend_max_respawns: u32,
        descriptor_max_respawns: u32,
        cull_min_observations: Option<usize>,
        trace_transitions: Option<bool>,
    ) -> Result<Self, TrackerInitError> {
        let cull_min_observations = match cull_min_observations {
            Some(value) => {
                NonZeroUsize::new(value).ok_or(TrackerInitError::ZeroMapCullMinObservations {
                    variable: MAP_CULL_MIN_OBSERVATIONS_ENV,
                })?
            }
            None => DEFAULT_CULL_MIN_OBSERVATIONS,
        };

        Ok(Self {
            backend_max_respawns,
            descriptor_max_respawns,
            cull_min_observations,
            trace_transitions: trace_transitions.unwrap_or(false),
        })
    }
}

pub struct SlamTracker {
    superpoint_left: SuperPoint,
    superpoint_right: SuperPoint,
    lightglue: LightGlue,
    triangulator: Triangulator,
    intrinsics: PinholeIntrinsics,
    config: TrackerConfig,
    state: TrackerState,
    ba: LocalBundleAdjuster,
    map: SlamMap,
    essential_graph: EssentialGraph,
    pose_graph_optimizer: PoseGraphOptimizer,
    backend: Option<BackendSupervisor>,
    backend_stats: BackendStats,
    descriptor_worker: Option<DescriptorSupervisor>,
    descriptor_stats: DescriptorStats,
    loop_db: Option<KeyframeDatabase>,
    loop_config: Option<LoopClosureConfig>,
    pending_loop: Option<PendingLoopCandidate>,
    loop_streak: HashMap<KeyframeId, usize>,
    pending_events: Vec<DiagnosticEvent>,
    pending_loop_correction: Option<Vec<(KeyframeId, Pose)>>,
    tracking_health: TrackingHealth,
    consecutive_tracking_failures: usize,
    last_pose_world: Option<Pose>,
    cull_min_observations: NonZeroUsize,
    trace_transitions: bool,
}

impl SlamTracker {
    const DEFAULT_ESSENTIAL_GRAPH_STRONG_THRESHOLD: u32 = 15;

    /// Construct a tracker whose triangulation, PnP, bundle adjustment, loop,
    /// and backend paths all share `stereo`'s validated left-camera model.
    pub fn try_new(
        superpoint_left: SuperPoint,
        superpoint_right: SuperPoint,
        lightglue: LightGlue,
        stereo: RectifiedStereo,
        config: TrackerConfig,
    ) -> Result<Self, TrackerInitError> {
        let environment = TrackerEnvironment::parse(
            config.backend.is_some(),
            config.global_descriptor_config().is_some(),
        )?;
        // The tracker has one camera-calibration authority. Derive the PnP,
        // BA, loop, and backend projection from the same parsed left camera
        // that the triangulator consumes.
        let intrinsics = stereo.left();
        let triangulator = Triangulator::new(stereo, config.triangulation);
        let ba = LocalBundleAdjuster::new(intrinsics, config.ba);
        let backend = match config.backend {
            Some(backend_cfg) => Some(BackendSupervisor::spawn_with_max_respawns(
                backend_cfg,
                intrinsics,
                config.ba,
                environment.backend_max_respawns,
            )?),
            None => None,
        };
        let loop_config = config.loop_closure_config();
        let loop_db = loop_config.map(|cfg| KeyframeDatabase::new(cfg.temporal_gap()));
        let descriptor_worker = match (loop_config, config.global_descriptor_config()) {
            (Some(_), Some(cfg)) => Some(DescriptorSupervisor::spawn_with_max_respawns(
                cfg,
                environment.descriptor_max_respawns,
            )?),
            _ => None,
        };
        Ok(Self {
            superpoint_left,
            superpoint_right,
            lightglue,
            triangulator,
            intrinsics,
            config,
            state: TrackerState::NeedKeyframe,
            ba,
            map: SlamMap::new(),
            essential_graph: EssentialGraph::new(Self::DEFAULT_ESSENTIAL_GRAPH_STRONG_THRESHOLD),
            pose_graph_optimizer: PoseGraphOptimizer::new(PoseGraphConfig::default()),
            backend,
            backend_stats: BackendStats::default(),
            descriptor_worker,
            descriptor_stats: DescriptorStats::default(),
            loop_db,
            loop_config,
            pending_loop: None,
            loop_streak: HashMap::new(),
            pending_events: Vec::new(),
            pending_loop_correction: None,
            tracking_health: TrackingHealth::Good,
            consecutive_tracking_failures: 0,
            last_pose_world: None,
            cull_min_observations: environment.cull_min_observations,
            trace_transitions: environment.trace_transitions,
        })
    }

    pub fn process(&mut self, pair: StereoPair) -> Result<TrackerOutput, TrackerError> {
        // Keep drained diagnostics queued until a TrackerOutput can carry them.
        self.drain_backend_responses();
        self.drain_descriptor_responses()?;
        if let Err(err) = self.process_pending_loop_closure() {
            match err {
                LoopDetectError::DescriptorMatchFailed(source)
                | LoopDetectError::VerificationFailed(LoopVerificationError::Map(source)) => {
                    return Err(source.into());
                }
                LoopDetectError::VerificationFailed(LoopVerificationError::InvariantViolation(
                    message,
                )) => return Err(TrackerError::InvariantViolation(message)),
                LoopDetectError::Pose(source) => return Err(source.into()),
                rejection => eprintln!("loop closure: {rejection}"),
            }
        }
        let tracking = match &self.state {
            TrackerState::NeedKeyframe => None,
            TrackerState::Tracking {
                keyframe,
                keyframe_id,
            } => Some((Arc::clone(keyframe), *keyframe_id)),
            TrackerState::Relocalizing(_) => None,
        };
        let relocalization_session = match &self.state {
            TrackerState::Relocalizing(session) => Some(session.clone()),
            _ => None,
        };
        if let Some(session) = relocalization_session {
            return self.relocalize(pair, session);
        }

        if let Some((keyframe, keyframe_id)) = tracking {
            self.track(pair, &keyframe, keyframe_id)
        } else {
            let bootstrap_pose = self.last_pose_world.unwrap_or(Pose::identity());
            if self.trace_transitions {
                eprintln!(
                    "tracker bootstrap keyframe: pose_source={} tx={:.3} ty={:.3} tz={:.3}",
                    if self.last_pose_world.is_some() {
                        "last_pose"
                    } else {
                        "identity"
                    },
                    bootstrap_pose.translation()[0],
                    bootstrap_pose.translation()[1],
                    bootstrap_pose.translation()[2]
                );
            }
            self.create_keyframe(pair, bootstrap_pose)
        }
    }

    pub fn covisibility_snapshot(&self) -> crate::map::CovisibilitySnapshot {
        self.map.covisibility_snapshot()
    }

    pub fn backend_stats(&self) -> BackendStats {
        self.backend_stats
    }

    pub fn descriptor_stats(&self) -> DescriptorStats {
        self.descriptor_stats
    }

    pub fn system_health(&self) -> SystemHealth {
        let backend_expected = self.config.backend.is_some();
        let backend_alive = self
            .backend
            .as_ref()
            .is_some_and(BackendSupervisor::has_worker);
        let descriptor_expected = self.config.global_descriptor_config().is_some();
        let descriptor_alive = self
            .descriptor_worker
            .as_ref()
            .is_some_and(DescriptorSupervisor::has_worker);
        SystemHealth::from_components(
            self.tracking_health,
            backend_expected,
            backend_alive,
            descriptor_expected,
            descriptor_alive,
            self.backend_stats,
        )
    }

    pub fn apply_loop_closure(&mut self, verified: VerifiedLoop) -> Result<(), TrackerError> {
        let corrected = apply_loop_closure_correction(
            &mut self.map,
            &mut self.essential_graph,
            &self.pose_graph_optimizer,
            &verified,
        )?;
        self.pending_loop_correction = if corrected.is_empty() {
            None
        } else {
            Some(corrected)
        };
        Ok(())
    }

    /// Take the pending loop closure correction, if any.
    ///
    /// Returns the corrected poses produced by the most recent
    /// `apply_loop_closure` call. The caller (pipeline) uses this to
    /// send a `RebuildFromSnapshot` command to the dense worker.
    pub fn take_pending_loop_correction(&mut self) -> Option<Vec<(KeyframeId, Pose)>> {
        self.pending_loop_correction.take()
    }

    fn emit_health(&mut self, tracking: TrackingHealth) -> SystemHealth {
        self.tracking_health = tracking;
        self.system_health()
    }

    fn emit_event(&mut self, event: DiagnosticEvent) {
        self.pending_events.push(event);
    }

    fn drain_events(&mut self) -> Vec<DiagnosticEvent> {
        std::mem::take(&mut self.pending_events)
    }

    fn empty_diagnostics(&self) -> FrameDiagnostics {
        let mut diagnostics =
            FrameDiagnostics::empty(self.map.num_keyframes(), self.map.num_points());
        diagnostics.loop_candidate_count = self
            .pending_loop
            .as_ref()
            .map_or(0, |pending| pending.candidates.len());
        diagnostics.loop_closure_applied = self
            .pending_events
            .iter()
            .any(|event| matches!(event, DiagnosticEvent::LoopClosureDetected { .. }));
        diagnostics
    }

    #[allow(clippy::too_many_arguments)]
    fn output_with_diagnostics(
        &mut self,
        pose: Option<Pose>,
        pose_status: PoseStatus,
        inliers: usize,
        keyframe: Option<Arc<Keyframe>>,
        stereo_matches: Option<Matches<Raw>>,
        frame_id: FrameId,
        tracking: TrackingHealth,
        diagnostics: FrameDiagnostics,
    ) -> TrackerOutput {
        debug_assert!(pose_status.is_consistent_with(pose));
        if let Some(pose_world) = pose {
            self.last_pose_world = Some(pose_world);
        }
        TrackerOutput {
            pose: pose.map(crate::WorldToCamera::from_legacy_pose),
            pose_status,
            inliers,
            keyframe,
            stereo_matches,
            frame_id,
            health: self.emit_health(tracking),
            diagnostics,
            events: self.drain_events(),
        }
    }

    /// Build a tracking-failure output with an explicitly stale fallback pose when available.
    fn tracking_failure_output(
        &mut self,
        frame_id: FrameId,
        health: TrackingHealth,
        diagnostics: FrameDiagnostics,
    ) -> TrackerOutput {
        let pose = self.last_pose_world;
        let pose_status = if pose.is_some() {
            PoseStatus::Stale
        } else {
            PoseStatus::Unavailable
        };
        self.output_with_diagnostics(
            pose,
            pose_status,
            0,
            None,
            None,
            frame_id,
            health,
            diagnostics,
        )
    }

    fn enqueue_loop_candidates(
        &mut self,
        keyframe_id: KeyframeId,
        detections: &Arc<Detections>,
    ) -> Result<(), TrackerError> {
        let Some((config, mut candidates)) = register_bootstrap_loop_descriptor(
            self.loop_config,
            self.loop_db.as_mut(),
            keyframe_id,
            detections,
        )?
        else {
            return Ok(());
        };
        candidates.retain(|candidate| candidate.similarity >= config.similarity_threshold());

        if candidates.is_empty() {
            self.loop_streak.clear();
            return Ok(());
        }

        let present: HashSet<KeyframeId> = candidates.iter().map(|m| m.candidate).collect();
        self.loop_streak
            .retain(|candidate, _| present.contains(candidate));
        for candidate in &candidates {
            let streak = self.loop_streak.entry(candidate.candidate).or_insert(0);
            *streak = streak.saturating_add(1);
        }

        if self.pending_loop.is_some() {
            return Ok(());
        }

        let promoted: Vec<PlaceMatch> = candidates
            .into_iter()
            .filter(|candidate| {
                self.loop_streak
                    .get(&candidate.candidate)
                    .copied()
                    .unwrap_or(0)
                    >= config.min_streak()
            })
            .collect();

        if promoted.is_empty() {
            return Ok(());
        }

        self.pending_loop = Some(PendingLoopCandidate {
            query_kf: keyframe_id,
            detections: Arc::clone(detections),
            candidates: promoted,
        });
        Ok(())
    }

    fn enqueue_descriptor_request(&mut self, keyframe_id: KeyframeId, frame: &Frame) {
        let Some(supervisor) = self.descriptor_worker.as_mut() else {
            return;
        };
        let request = DescriptorRequest {
            keyframe_id,
            source_snapshot: self.map.snapshot(),
            frame: frame.clone(),
        };
        match supervisor.submit(request) {
            Ok(()) => {
                self.descriptor_stats.submitted = self.descriptor_stats.submitted.saturating_add(1);
            }
            Err(SubmitDescriptorError::QueueFull) => {
                self.descriptor_stats.dropped_full =
                    self.descriptor_stats.dropped_full.saturating_add(1);
                eprintln!("descriptor worker queue full; keeping bootstrap descriptor");
            }
            Err(SubmitDescriptorError::Disconnected) => {
                self.descriptor_stats.dropped_disconnected =
                    self.descriptor_stats.dropped_disconnected.saturating_add(1);
                self.descriptor_stats.respawn_count = supervisor.respawn_count();
                eprintln!("descriptor worker disconnected; retrying with supervisor");
            }
        }
    }

    fn drain_descriptor_responses(&mut self) -> Result<(), TrackerError> {
        loop {
            let response = {
                let Some(supervisor) = self.descriptor_worker.as_mut() else {
                    return Ok(());
                };
                let response = supervisor.try_recv();
                self.descriptor_stats.respawn_count = supervisor.respawn_count();
                response
            };
            let Some(response) = response else {
                break;
            };
            match response {
                DescriptorWorkerResponse::Descriptor(response) => {
                    let Some(loop_db) = self.loop_db.as_mut() else {
                        return Err(TrackerError::InvariantViolation(
                            "descriptor worker requires an enabled keyframe database",
                        ));
                    };
                    match apply_descriptor_response(&self.map, loop_db, *response)? {
                        DescriptorApplyDisposition::Applied => {
                            self.descriptor_stats.applied =
                                self.descriptor_stats.applied.saturating_add(1);
                        }
                        DescriptorApplyDisposition::Stale => {}
                    }
                }
                DescriptorWorkerResponse::Failure {
                    keyframe_id,
                    source_snapshot,
                    error,
                } => {
                    self.descriptor_stats.worker_failures =
                        self.descriptor_stats.worker_failures.saturating_add(1);
                    eprintln!(
                        "descriptor worker failure (keyframe={keyframe_id:?}, snapshot={source_snapshot:?}): {error}"
                    );
                }
                DescriptorWorkerResponse::WorkerPanic {
                    keyframe_id,
                    source_snapshot,
                    message,
                } => {
                    self.descriptor_stats.panics = self.descriptor_stats.panics.saturating_add(1);
                    self.descriptor_stats.worker_failures =
                        self.descriptor_stats.worker_failures.saturating_add(1);
                    eprintln!(
                        "descriptor worker panic (keyframe={keyframe_id:?}, snapshot={source_snapshot:?}): {message}"
                    );
                    self.emit_event(DiagnosticEvent::DescriptorWorkerDied {
                        respawn_count: self.descriptor_stats.respawn_count,
                    });
                }
            }
        }
        Ok(())
    }

    fn process_pending_loop_closure(&mut self) -> Result<Option<VerifiedLoop>, LoopDetectError> {
        let Some(config) = self.loop_config else {
            self.pending_loop = None;
            self.loop_streak.clear();
            return Ok(None);
        };
        let Some(pending) = self.pending_loop.take() else {
            return Ok(None);
        };

        let mut first_error: Option<LoopDetectError> = None;
        for candidate in pending.candidates {
            let correspondences = try_match_descriptors_for_loop(
                pending.detections.descriptors(),
                candidate.candidate,
                &self.map,
                config.descriptor_match_threshold(),
            )
            .map_err(LoopDetectError::DescriptorMatchFailed)?;

            if correspondences.len() < MIN_PNP_CORRESPONDENCES {
                if first_error.is_none() {
                    first_error = Some(LoopDetectError::TooFewCorrespondences {
                        count: correspondences.len(),
                    });
                }
                continue;
            }

            let loop_candidate = LoopCandidate {
                query_kf: pending.query_kf,
                match_kf: candidate.candidate,
            };

            let verified = match loop_candidate.verify(
                pending.detections.keypoints(),
                &correspondences,
                &self.map,
                self.intrinsics,
                config.ransac(),
                config.min_inliers(),
            ) {
                Ok(value) => value,
                Err(
                    err @ (LoopVerificationError::Map(_)
                    | LoopVerificationError::InvariantViolation(_)),
                ) => return Err(LoopDetectError::VerificationFailed(err)),
                Err(err) => {
                    if first_error.is_none() {
                        first_error = Some(LoopDetectError::VerificationFailed(err));
                    }
                    continue;
                }
            };

            let Some(query_keyframe) = self.map.keyframe(verified.query_kf()) else {
                if first_error.is_none() {
                    first_error = Some(LoopDetectError::ApplyFailed(
                        LoopApplyError::MissingKeyframe,
                    ));
                }
                continue;
            };
            let correction = loop_pose_correction(
                query_keyframe.pose().into_legacy_pose(),
                verified.query_pose_world(),
            )
            .map_err(LoopDetectError::Pose)?;
            let translation = loop_translation_norm(correction);
            let rotation_deg = loop_rotation_angle_deg(correction);
            if translation > config.max_correction_translation()
                || rotation_deg > config.max_correction_rotation_deg()
            {
                if first_error.is_none() {
                    first_error = Some(LoopDetectError::CorrectionTooLarge {
                        translation,
                        rotation_deg,
                    });
                }
                continue;
            }

            if let Err(err) = self.apply_loop_closure(verified.clone()) {
                let detect_err = LoopDetectError::ApplyFailed(loop_apply_error_kind(&err));
                self.emit_event(DiagnosticEvent::LoopClosureRejected {
                    reason: loop_reject_reason(&detect_err),
                });
                return Err(detect_err);
            }
            self.emit_event(DiagnosticEvent::LoopClosureDetected {
                query: pending.query_kf,
                match_kf: candidate.candidate,
                similarity: candidate.similarity,
            });
            self.loop_streak.remove(&candidate.candidate);
            return Ok(Some(verified));
        }

        if let Some(err) = first_error {
            self.emit_event(DiagnosticEvent::LoopClosureRejected {
                reason: loop_reject_reason(&err),
            });
            Err(err)
        } else {
            Ok(None)
        }
    }

    fn submit_backend_event(
        &mut self,
        trigger_keyframe: KeyframeId,
        window_ids: Vec<KeyframeId>,
    ) -> Result<(), SubmitEventError> {
        let Some(supervisor) = self.backend.as_mut() else {
            return Err(SubmitEventError::Disconnected);
        };

        let window = BackendWindow::try_new(window_ids).map_err(SubmitEventError::InvalidWindow)?;
        let request_id = supervisor.next_request_id()?;
        let event = KeyframeEvent::try_new(request_id, trigger_keyframe, window, self.map.clone())
            .map_err(SubmitEventError::InvalidEvent)?;
        supervisor.submit(event)?;
        self.backend_stats.respawn_count = supervisor.respawn_count();
        self.backend_stats.submitted = self.backend_stats.submitted.saturating_add(1);
        Ok(())
    }

    fn drain_backend_responses(&mut self) {
        loop {
            let response = {
                let Some(supervisor) = self.backend.as_mut() else {
                    return;
                };
                let response = supervisor.try_recv();
                self.backend_stats.respawn_count = supervisor.respawn_count();
                response
            };

            let Some(response) = response else {
                break;
            };
            match response {
                BackendResponse::Correction(correction) => {
                    if correction.source_snapshot != self.map.snapshot() {
                        self.backend_stats.stale = self.backend_stats.stale.saturating_add(1);
                        continue;
                    }
                    match &correction.correction.result {
                        BaResult::Converged { .. } | BaResult::MaxIterations { .. } => {
                            match apply_correction_event(&mut self.map, &correction) {
                                Ok(()) => {
                                    self.backend_stats.applied =
                                        self.backend_stats.applied.saturating_add(1);
                                }
                                Err(ApplyCorrectionError::StaleSnapshot { .. }) => {
                                    self.backend_stats.stale =
                                        self.backend_stats.stale.saturating_add(1);
                                }
                                Err(err) => {
                                    self.backend_stats.rejected =
                                        self.backend_stats.rejected.saturating_add(1);
                                    eprintln!(
                                        "backend correction rejected (req={}, keyframe={:?}): {err}",
                                        correction.request_id.as_u64(),
                                        correction.trigger_keyframe
                                    );
                                }
                            }
                        }
                        BaResult::Degenerate { reason } => {
                            self.backend_stats.rejected =
                                self.backend_stats.rejected.saturating_add(1);
                            self.emit_event(DiagnosticEvent::BaDegenerate { reason: *reason });
                            eprintln!(
                                "backend BA degenerate (req={}, keyframe={:?}): {reason:?}",
                                correction.request_id.as_u64(),
                                correction.trigger_keyframe
                            );
                        }
                    }
                }
                BackendResponse::Failure {
                    request_id,
                    source_snapshot,
                    error,
                } => {
                    self.backend_stats.worker_failures =
                        self.backend_stats.worker_failures.saturating_add(1);
                    eprintln!(
                        "backend worker failure (req={}, snapshot={source_snapshot:?}): {error}",
                        request_id.as_u64(),
                    );
                }
                BackendResponse::WorkerPanic {
                    request_id,
                    source_snapshot,
                } => {
                    self.backend_stats.panics = self.backend_stats.panics.saturating_add(1);
                    self.backend_stats.worker_failures =
                        self.backend_stats.worker_failures.saturating_add(1);
                    if let Some(supervisor) = self.backend.as_mut() {
                        supervisor.check_health();
                        self.backend_stats.respawn_count = supervisor.respawn_count();
                    }
                    self.emit_event(DiagnosticEvent::BackendWorkerDied {
                        respawn_count: self.backend_stats.respawn_count,
                    });
                    eprintln!(
                        "backend worker panic (req={}, snapshot={source_snapshot:?})",
                        request_id.as_u64(),
                    );
                }
            }
        }
    }

    fn tracking_failure_health(&mut self) -> TrackingHealth {
        const LOST_AFTER_CONSECUTIVE_FAILURES: usize = 5;
        self.consecutive_tracking_failures = self.consecutive_tracking_failures.saturating_add(1);
        let health = if self.consecutive_tracking_failures >= LOST_AFTER_CONSECUTIVE_FAILURES {
            if self.tracking_health != TrackingHealth::Lost {
                self.emit_event(DiagnosticEvent::TrackingLost {
                    consecutive_failures: self.consecutive_tracking_failures,
                });
            }
            TrackingHealth::Lost
        } else {
            TrackingHealth::Degraded
        };
        if self.trace_transitions {
            eprintln!(
                "tracking failure count={} health={health:?}",
                self.consecutive_tracking_failures
            );
        }
        health
    }

    fn maybe_enter_relocalization(&mut self, tracking_health: TrackingHealth) {
        if let Some(session) = Self::initial_relocalization_session(
            tracking_health,
            self.config.relocalization_config().is_some(),
        ) {
            self.emit_event(DiagnosticEvent::RelocalizationStarted);
            self.pending_loop = None;
            self.loop_streak.clear();
            self.state = TrackerState::Relocalizing(session);
            if self.trace_transitions {
                eprintln!("entering relocalization after tracking loss");
            }
        }
    }

    fn initial_relocalization_session(
        tracking_health: TrackingHealth,
        relocalization_enabled: bool,
    ) -> Option<RelocalizationSession> {
        if tracking_health != TrackingHealth::Lost || !relocalization_enabled {
            return None;
        }
        Some(RelocalizationSession {
            attempts: 0,
            phase: RelocalizationPhase::Searching,
        })
    }

    fn relocalization_output(
        &mut self,
        frame_id: FrameId,
        health: TrackingHealth,
    ) -> TrackerOutput {
        let diagnostics = self.empty_diagnostics();
        let pose = self.last_pose_world;
        let pose_status = if pose.is_some() {
            PoseStatus::Stale
        } else {
            PoseStatus::Unavailable
        };
        self.output_with_diagnostics(
            pose,
            pose_status,
            0,
            None,
            None,
            frame_id,
            health,
            diagnostics,
        )
    }

    fn relocalization_pose_consistent(
        previous_pose: Pose,
        current_pose: Pose,
        cfg: RelocalizationConfig,
    ) -> Result<bool, crate::PoseError> {
        let delta = crate::local_ba::se3_delta_between(previous_pose, current_pose)?;
        let translation_delta = delta[0].hypot(delta[1]).hypot(delta[2]);
        let rotation_delta_deg = delta[3].hypot(delta[4]).hypot(delta[5]).to_degrees();
        Ok(translation_delta <= cfg.max_translation_delta_m()
            && rotation_delta_deg <= cfg.max_rotation_delta_deg())
    }

    fn relocalization_candidate(
        current: &Detections,
        cfg: RelocalizationConfig,
        loop_db: &KeyframeDatabase,
        map: &SlamMap,
        intrinsics: PinholeIntrinsics,
        ransac: RansacConfig,
    ) -> Result<Option<crate::loop_closure::VerifiedRelocalization>, TrackerError> {
        let global_descriptor = aggregate_global_descriptor(current.descriptors())?;
        let candidates = loop_db.query_for_relocalization(&global_descriptor, cfg.max_candidates());
        for candidate in candidates {
            let correspondences = try_match_descriptors_for_loop(
                current.descriptors(),
                candidate.candidate,
                map,
                cfg.descriptor_match_threshold(),
            )?;
            if correspondences.len() < MIN_PNP_CORRESPONDENCES {
                continue;
            }
            let relocalization_candidate = RelocalizationCandidate {
                match_kf: candidate.candidate,
            };
            let verified = match relocalization_candidate.verify(
                current.keypoints(),
                &correspondences,
                map,
                intrinsics,
                ransac,
                cfg.min_inliers(),
            ) {
                Ok(value) => value,
                Err(error) => {
                    Self::classify_relocalization_verification_failure(error)?;
                    continue;
                }
            };
            return Ok(Some(verified));
        }
        Ok(None)
    }

    fn classify_relocalization_verification_failure(
        error: LoopVerificationError,
    ) -> Result<(), TrackerError> {
        match error {
            LoopVerificationError::TooFewMatches { .. }
            | LoopVerificationError::PnpFailed(crate::PnpError::NoSolution)
            | LoopVerificationError::InsufficientInliers { .. } => Ok(()),
            LoopVerificationError::PnpFailed(source) => Err(source.into()),
            LoopVerificationError::Map(source) => Err(source.into()),
            LoopVerificationError::InvariantViolation(message) => {
                Err(TrackerError::InvariantViolation(message))
            }
        }
    }

    fn begin_relocalization_attempt(
        mut session: RelocalizationSession,
        cfg: RelocalizationConfig,
    ) -> Option<RelocalizationAttempt> {
        if session.attempts >= cfg.max_attempts() {
            return None;
        }
        session.attempts += 1;
        Some(RelocalizationAttempt {
            is_final: session.attempts == cfg.max_attempts(),
            session,
        })
    }

    fn relocalization_fallback_state(attempt: &RelocalizationAttempt) -> TrackerState {
        let mut session = attempt.session.clone();
        session.phase = RelocalizationPhase::Searching;
        TrackerState::Relocalizing(session)
    }

    fn relocalization_step(
        attempt: RelocalizationAttempt,
        evidence: RelocalizationEvidence,
        cfg: RelocalizationConfig,
    ) -> Result<RelocalizationStep, crate::PoseError> {
        let RelocalizationAttempt {
            mut session,
            is_final,
        } = attempt;
        let next_phase = match evidence {
            RelocalizationEvidence::NoCandidate => RelocalizationPhase::Searching,
            RelocalizationEvidence::Verified {
                attachment,
                pose_world,
            } => {
                let candidate_id = attachment.candidate;
                let required_confirmations = cfg.min_confirmations();
                let consistent_with_previous = match &session.phase {
                    RelocalizationPhase::Confirming {
                        candidate,
                        pose_world: previous_pose,
                        ..
                    } if *candidate == candidate_id => {
                        Self::relocalization_pose_consistent(*previous_pose, pose_world, cfg)?
                    }
                    _ => false,
                };
                match session.phase {
                    RelocalizationPhase::Confirming {
                        candidate,
                        confirmations,
                        pose_world: _,
                    } if candidate == candidate_id && consistent_with_previous => {
                        let next_confirmations = confirmations.get().saturating_add(1);
                        if next_confirmations >= required_confirmations {
                            return Ok(RelocalizationStep::Recovered {
                                attachment,
                                pose_world,
                            });
                        }
                        RelocalizationPhase::Confirming {
                            candidate,
                            confirmations: NonZeroUsize::new(next_confirmations)
                                .expect("incrementing a nonzero confirmation count stays nonzero"),
                            pose_world,
                        }
                    }
                    _ if required_confirmations <= 1 => {
                        return Ok(RelocalizationStep::Recovered {
                            attachment,
                            pose_world,
                        });
                    }
                    _ => RelocalizationPhase::Confirming {
                        candidate: candidate_id,
                        confirmations: NonZeroUsize::MIN,
                        pose_world,
                    },
                }
            }
        };
        if is_final {
            Ok(RelocalizationStep::Exhausted)
        } else {
            session.phase = next_phase;
            Ok(RelocalizationStep::Continue(session))
        }
    }

    fn reset_mapping_session(&mut self) {
        let transition = replace_mapping_session(
            &mut self.map,
            &mut self.essential_graph,
            self.loop_db.as_mut(),
            Self::DEFAULT_ESSENTIAL_GRAPH_STRONG_THRESHOLD,
        );
        // Publish the boundary before any subsequent frame can create a
        // keyframe in the fresh map. Downstream state must never mix map
        // instances merely because sparse relocalization was exhausted.
        self.emit_event(DiagnosticEvent::MappingSessionReset { transition });
        self.state = TrackerState::NeedKeyframe;
        self.ba.reset();
        self.pending_loop = None;
        self.loop_streak.clear();
        self.pending_loop_correction = None;
        self.consecutive_tracking_failures = 0;
        self.last_pose_world = None;
    }

    fn relocalize(
        &mut self,
        pair: StereoPair,
        session: RelocalizationSession,
    ) -> Result<TrackerOutput, TrackerError> {
        let (left, right) = pair.into_parts();
        let frame_id = left.frame_id();
        let Some(cfg) = self.config.relocalization_config() else {
            self.reset_mapping_session();
            return Ok(self.relocalization_output(frame_id, TrackingHealth::Lost));
        };
        let Some(attempt) = Self::begin_relocalization_attempt(session, cfg) else {
            self.reset_mapping_session();
            return Ok(self.relocalization_output(frame_id, TrackingHealth::Lost));
        };
        let attempt_number = attempt.session.attempts;
        self.state = Self::relocalization_fallback_state(&attempt);

        let current = self
            .superpoint_left
            .detect_with_downscale(&left, self.config.downscale)?
            .top_k(self.config.max_keypoints());

        let evidence = if current.is_empty() {
            RelocalizationEvidence::NoCandidate
        } else {
            let Some(loop_db) = self.loop_db.as_ref() else {
                self.reset_mapping_session();
                return Ok(self.relocalization_output(frame_id, TrackingHealth::Lost));
            };
            let verified_snapshot = self.map.snapshot();
            match Self::relocalization_candidate(
                &current,
                cfg,
                loop_db,
                &self.map,
                self.intrinsics,
                self.config.ransac,
            )? {
                Some(verified) => {
                    let candidate = verified.match_kf();
                    let inlier_count = NonZeroUsize::new(verified.inlier_count()).ok_or(
                        TrackerError::InvariantViolation(
                            "verified relocalization reported zero inliers",
                        ),
                    )?;
                    if self.trace_transitions {
                        eprintln!(
                            "relocalization candidate frame={} candidate={candidate:?} inliers={}",
                            frame_id.as_u64(),
                            verified.inlier_count()
                        );
                    }
                    RelocalizationEvidence::Verified {
                        attachment: RelocalizationAttachment {
                            candidate,
                            verified_snapshot,
                            inlier_count,
                        },
                        pose_world: verified.pose_world(),
                    }
                }
                None => RelocalizationEvidence::NoCandidate,
            }
        };
        let pending_health = match evidence {
            RelocalizationEvidence::NoCandidate => TrackingHealth::Lost,
            RelocalizationEvidence::Verified { .. } => TrackingHealth::Degraded,
        };

        match Self::relocalization_step(attempt, evidence, cfg)? {
            RelocalizationStep::Recovered {
                attachment,
                pose_world,
            } => {
                let candidate = attachment.candidate;
                self.pending_loop = None;
                self.loop_streak.clear();
                if self.trace_transitions {
                    eprintln!(
                        "relocalization recovered frame={} candidate={candidate:?}; attaching keyframe at recovered pose",
                        frame_id.as_u64()
                    );
                }
                // Relocalization already ran the left detector for this frame. Transfer those
                // detections only on recovery so keyframe creation needs only the right detector.
                let mut output = self.create_keyframe_with_left_detections(
                    StereoPair::from_parts(left, right),
                    pose_world,
                    Some(Arc::new(current)),
                    Some(attachment),
                )?;
                if output.diagnostics.keyframe_created {
                    output
                        .events
                        .push(DiagnosticEvent::RelocalizationSucceeded {
                            keyframe_id: candidate,
                        });
                }
                return Ok(output);
            }
            RelocalizationStep::Continue(next_session) => {
                self.state = TrackerState::Relocalizing(next_session);
                if self.trace_transitions {
                    match evidence {
                        RelocalizationEvidence::NoCandidate => eprintln!(
                            "relocalization failure frame={} attempt={}/{}",
                            frame_id.as_u64(),
                            attempt_number,
                            cfg.max_attempts()
                        ),
                        RelocalizationEvidence::Verified { .. } => eprintln!(
                            "relocalization confirmation pending frame={} attempt={}/{}",
                            frame_id.as_u64(),
                            attempt_number,
                            cfg.max_attempts()
                        ),
                    }
                }
            }
            RelocalizationStep::Exhausted => {
                if self.trace_transitions {
                    eprintln!(
                        "relocalization exhausted frame={} attempt={}/{}",
                        frame_id.as_u64(),
                        cfg.max_attempts(),
                        cfg.max_attempts()
                    );
                }
                self.reset_mapping_session();
                return Ok(self.relocalization_output(frame_id, TrackingHealth::Lost));
            }
        }
        Ok(self.relocalization_output(frame_id, pending_health))
    }

    fn track(
        &mut self,
        pair: StereoPair,
        keyframe: &Arc<Keyframe>,
        keyframe_id: KeyframeId,
    ) -> Result<TrackerOutput, TrackerError> {
        let tracking_start = Instant::now();
        let (left, right) = pair.into_parts();
        let frame_id = left.frame_id();

        let current = self
            .superpoint_left
            .detect_with_downscale(&left, self.config.downscale)?
            .top_k(self.config.max_keypoints());
        let current = Arc::new(current);

        let matches = if current.is_empty() || keyframe.detections().is_empty() {
            if self.trace_transitions {
                eprintln!(
                    "tracking failure frame={} reason=empty_features current={} keyframe={}",
                    frame_id.as_u64(),
                    current.len(),
                    keyframe.detections().len()
                );
            }
            let tracking_health = self.tracking_failure_health();
            self.maybe_enter_relocalization(tracking_health);
            let mut diagnostics = self.empty_diagnostics();
            diagnostics.features_detected = Some(current.len());
            diagnostics.features_matched = Some(0);
            diagnostics.tracking_time = Some(tracking_start.elapsed());
            return Ok(self.tracking_failure_output(frame_id, tracking_health, diagnostics));
        } else {
            self.lightglue
                .match_these(current.clone(), keyframe.detections().clone())?
        };

        let verified = match matches.with_landmarks(keyframe) {
            Ok(verified) => verified,
            Err(err) => {
                return Err(TrackerError::Inference(InferenceError::Match(err)));
            }
        };

        let observation_batch = match build_map_observations(
            &self.map,
            keyframe_id,
            &verified,
            current.as_ref(),
        ) {
            Ok(obs) => obs,
            Err(crate::PnpError::NotEnoughPoints { .. }) => {
                if self.trace_transitions {
                    eprintln!(
                        "tracking failure frame={} reason=not_enough_map_points matches={} verified={} current={}",
                        frame_id.as_u64(),
                        matches.len(),
                        verified.len(),
                        current.len()
                    );
                }
                let tracking_health = self.tracking_failure_health();
                self.maybe_enter_relocalization(tracking_health);
                let mut diagnostics = self.empty_diagnostics();
                diagnostics.features_detected = Some(current.len());
                diagnostics.features_matched = Some(matches.len());
                diagnostics.tracking_time = Some(tracking_start.elapsed());
                return Ok(self.tracking_failure_output(frame_id, tracking_health, diagnostics));
            }
            Err(err) => return Err(TrackerError::Pnp(err)),
        };

        let observations = &observation_batch.observations;
        let result = match solve_pnp_ransac(observations, self.intrinsics, self.config.ransac) {
            Ok(result) => result,
            Err(crate::PnpError::NotEnoughPoints { .. } | crate::PnpError::NoSolution) => {
                if self.trace_transitions {
                    eprintln!(
                        "tracking failure frame={} reason=pnp_failed observations={} matches={} verified={}",
                        frame_id.as_u64(),
                        observations.len(),
                        matches.len(),
                        verified.len()
                    );
                }
                let tracking_health = self.tracking_failure_health();
                self.maybe_enter_relocalization(tracking_health);
                let mut diagnostics = self.empty_diagnostics();
                diagnostics.features_detected = Some(current.len());
                diagnostics.features_matched = Some(matches.len());
                diagnostics.pnp_observations = Some(observations.len());
                diagnostics.tracking_time = Some(tracking_start.elapsed());
                return Ok(self.tracking_failure_output(frame_id, tracking_health, diagnostics));
            }
            Err(err) => return Err(TrackerError::Pnp(err)),
        };

        let inlier_match_indices: Vec<usize> = result
            .inliers
            .iter()
            .map(|&observation_idx| {
                let match_index = observation_batch
                    .match_indices
                    .get(observation_idx)
                    .copied()
                    .ok_or(TrackerError::InvariantViolation(
                        "PnP inlier index out of observation batch bounds",
                    ))?;
                observations
                    .get(observation_idx)
                    .ok_or(TrackerError::InvariantViolation(
                        "PnP inlier index out of observation bounds",
                    ))?;
                Ok(match_index)
            })
            .collect::<Result<_, TrackerError>>()?;

        let mut map_observations = Vec::with_capacity(result.inliers.len());
        for &match_idx in &inlier_match_indices {
            let (ci, ki) = *verified
                .indices()
                .get(match_idx)
                .ok_or(TrackerError::Inference(
                    InferenceError::InvariantViolation {
                        context: "verified match index out of bounds",
                    },
                ))?;
            let pixel = *current.keypoints().get(ci).ok_or(TrackerError::Inference(
                InferenceError::InvariantViolation {
                    context: "current keypoint index out of bounds",
                },
            ))?;
            let keypoint_ref = self.map.keyframe_keypoint(keyframe_id, ki)?;
            map_observations.push(MapObservation::new(keypoint_ref, pixel));
        }

        let parallax_px = median_parallax_px(&verified, &inlier_match_indices, keyframe);
        let covisibility = if keyframe.landmarks().is_empty() {
            0.0
        } else {
            result.inliers.len() as f32 / keyframe.landmarks().len() as f32
        };

        let pose_world = result.pose;
        let pose_world_legacy = pose_world.into_legacy_pose();
        let observation_set = ObservationSet::new(map_observations, self.ba.min_observations())
            .map_err(LocalBaError::from)?;
        let refined_world = self
            .ba
            .push_frame(&self.map, pose_world_legacy, observation_set)?;

        let pose_world = crate::WorldToCamera::from_legacy_pose(refined_world);
        // This is the last fallible acceptance check before recovery events or tracker/map/backend
        // mutations. If it rejects the refined frame, discard the already-mutated BA window just
        // as LocalBundleAdjuster::push_frame does when its own optimization fails.
        let reprojection_diagnostics = match post_ba_reprojection_diagnostics(
            refined_world,
            result.inliers.iter().map(|&idx| &observations[idx]),
            self.intrinsics,
        ) {
            Ok(diagnostics) => diagnostics,
            Err(err) => {
                self.ba.reset();
                return Err(err);
            }
        };
        if self.consecutive_tracking_failures > 0 {
            self.emit_event(DiagnosticEvent::TrackingRecovered);
        }
        self.consecutive_tracking_failures = 0;
        let mut output_keyframe = None;
        let mut output_matches = None;
        let mut keyframe_created = false;
        let mut triangulation_stats = None;
        let mut ba_result = None;

        let should_refresh = self.config.keyframe_policy.should_refresh(
            result.inliers.len(),
            parallax_px,
            covisibility,
        );

        if should_refresh {
            let new_pose = pose_world;
            let shared = build_shared_matches(keyframe_id, &verified, &inlier_match_indices);
            let created = match self.create_keyframe_internal(
                left,
                right,
                new_pose,
                Some(current.clone()),
                KeyframeConnection::Covisibility(shared),
            ) {
                Ok(value) => Some(value),
                Err(TrackerError::KeyframeRejected { .. }) => None,
                Err(err) => return Err(err),
            };
            if let Some((keyframe_output, keyframe_id)) = created {
                keyframe_created = true;
                triangulation_stats = keyframe_output.diagnostics.triangulation;
                ba_result = keyframe_output.diagnostics.ba_result.clone();
                if let Some(keyframe) = keyframe_output.keyframe {
                    let redundant = self
                        .config
                        .redundancy
                        .map(|policy| {
                            is_redundant(&self.map, keyframe_id, policy.max_covisibility())
                        })
                        .transpose()?
                        .unwrap_or(false);
                    if redundant {
                        remove_keyframe_from_graph_and_db(
                            &mut self.map,
                            &mut self.essential_graph,
                            self.loop_db.as_mut(),
                            keyframe_id,
                        )?;
                        self.emit_event(DiagnosticEvent::KeyframeRemoved {
                            keyframe_id,
                            reason: KeyframeRemovalReason::Redundant,
                        });
                        self.loop_streak.remove(&keyframe_id);
                        if let Some(pending) = self.pending_loop.as_mut() {
                            if pending.query_kf == keyframe_id {
                                self.pending_loop = None;
                            } else {
                                pending
                                    .candidates
                                    .retain(|candidate| candidate.candidate != keyframe_id);
                                if pending.candidates.is_empty() {
                                    self.pending_loop = None;
                                }
                            }
                        }
                    } else {
                        let window = self
                            .map
                            .covisible_window(keyframe_id, self.ba.window_size())?;
                        if window.len() >= 2 {
                            if self.backend.is_some() {
                                if let Err(err) =
                                    self.submit_backend_event(keyframe_id, window.clone())
                                {
                                    match err {
                                        SubmitEventError::QueueFull => {
                                            self.backend_stats.dropped_full =
                                                self.backend_stats.dropped_full.saturating_add(1);
                                        }
                                        SubmitEventError::Disconnected => {
                                            self.backend_stats.dropped_disconnected = self
                                                .backend_stats
                                                .dropped_disconnected
                                                .saturating_add(1);
                                            if let Some(supervisor) = self.backend.as_ref() {
                                                self.backend_stats.respawn_count =
                                                    supervisor.respawn_count();
                                            }
                                        }
                                        SubmitEventError::InvalidWindow(_)
                                        | SubmitEventError::InvalidEvent(_)
                                        | SubmitEventError::RequestIdExhausted(_) => {
                                            self.backend_stats.rejected =
                                                self.backend_stats.rejected.saturating_add(1);
                                        }
                                    }
                                    eprintln!(
                                        "backend submit failed for keyframe {keyframe_id:?}: {err}"
                                    );
                                    let result =
                                        self.ba.optimize_keyframe_window(&mut self.map, &window)?;
                                    ba_result = Some(result);
                                }
                            } else {
                                let result =
                                    self.ba.optimize_keyframe_window(&mut self.map, &window)?;
                                ba_result = Some(result);
                            }
                        }
                        self.state = TrackerState::Tracking {
                            keyframe: keyframe.clone(),
                            keyframe_id,
                        };
                        self.ba.reset();
                        output_keyframe = Some(keyframe);
                        output_matches = keyframe_output.stereo_matches;
                    }
                }
            }
        }

        let mut diagnostics = self.empty_diagnostics();
        diagnostics.inlier_ratio =
            Some(result.inliers.len() as f32 / observations.len().max(1) as f32);
        diagnostics.pnp_observations = Some(observations.len());
        diagnostics.ransac_iterations = Some(result.iterations);
        if let PostBaReprojectionDiagnostics::Complete { rmse_px, max_px } =
            reprojection_diagnostics
        {
            diagnostics.reprojection_rmse_px = Some(rmse_px);
            diagnostics.reprojection_max_px = Some(max_px);
        }
        diagnostics.parallax_px = parallax_px;
        diagnostics.covisibility = Some(covisibility);
        diagnostics.keyframe_created = keyframe_created;
        diagnostics.triangulation = triangulation_stats;
        diagnostics.ba_result = ba_result;
        diagnostics.tracking_time = Some(tracking_start.elapsed());
        diagnostics.features_detected = Some(current.len());
        diagnostics.features_matched = Some(matches.len());

        Ok(self.output_with_diagnostics(
            Some(pose_world.into_legacy_pose()),
            PoseStatus::Current,
            result.inliers.len(),
            output_keyframe,
            output_matches,
            frame_id,
            TrackingHealth::Good,
            diagnostics,
        ))
    }

    fn create_keyframe(
        &mut self,
        pair: StereoPair,
        pose_world: Pose,
    ) -> Result<TrackerOutput, TrackerError> {
        self.create_keyframe_with_left_detections(pair, pose_world, None, None)
    }

    fn create_keyframe_with_left_detections(
        &mut self,
        pair: StereoPair,
        pose_world: Pose,
        left_det: Option<Arc<Detections>>,
        relocalization: Option<RelocalizationAttachment>,
    ) -> Result<TrackerOutput, TrackerError> {
        let (left, right) = pair.into_parts();
        let frame_id = left.frame_id();
        let is_relocalization = relocalization.is_some();
        let connection = relocalization.map_or(
            KeyframeConnection::Bootstrap,
            KeyframeConnection::Relocalization,
        );
        let (output, keyframe_id) = match self.create_keyframe_internal(
            left,
            right,
            crate::WorldToCamera::from_legacy_pose(pose_world),
            left_det,
            connection,
        ) {
            Ok(value) => value,
            Err(TrackerError::KeyframeRejected { landmarks }) => {
                if is_relocalization {
                    // This stereo pair did not form a keyframe, but the old map
                    // remains valid. Re-verify a later frame instead of entering
                    // generic bootstrap against a non-empty graph.
                    self.state = TrackerState::Relocalizing(RelocalizationSession {
                        attempts: 0,
                        phase: RelocalizationPhase::Searching,
                    });
                }
                if self.trace_transitions {
                    if is_relocalization {
                        eprintln!(
                            "relocalization keyframe rejected frame={} landmarks={}; retrying verification",
                            frame_id.as_u64(),
                            landmarks
                        );
                    } else {
                        eprintln!(
                            "keyframe bootstrap rejected frame={} landmarks={} -> staying in NeedKeyframe",
                            frame_id.as_u64(),
                            landmarks
                        );
                    }
                }
                let mut diagnostics = self.empty_diagnostics();
                diagnostics.keyframe_created = false;
                return Ok(self.tracking_failure_output(
                    frame_id,
                    TrackingHealth::Degraded,
                    diagnostics,
                ));
            }
            Err(err) => {
                if self.trace_transitions {
                    if is_relocalization {
                        eprintln!(
                            "relocalization keyframe failed frame={} error={err}",
                            frame_id.as_u64()
                        );
                    } else {
                        eprintln!(
                            "keyframe bootstrap rejected frame={} error={err}",
                            frame_id.as_u64()
                        );
                    }
                }
                return Err(err);
            }
        };
        let Some(keyframe) = output.keyframe.clone() else {
            return Err(TrackerError::InvariantViolation(
                "create_keyframe_internal returned TrackerOutput without keyframe",
            ));
        };
        self.state = TrackerState::Tracking {
            keyframe,
            keyframe_id,
        };
        self.ba.reset();
        self.consecutive_tracking_failures = 0;
        let diagnostics = output.diagnostics;
        Ok(self.output_with_diagnostics(
            Some(pose_world),
            PoseStatus::Current,
            0,
            output.keyframe,
            output.stereo_matches,
            frame_id,
            TrackingHealth::Good,
            diagnostics,
        ))
    }

    fn create_keyframe_internal(
        &mut self,
        left: Frame,
        right: Frame,
        pose_world: crate::WorldToCamera,
        left_det: Option<Arc<Detections>>,
        connection: KeyframeConnection,
    ) -> Result<(TrackerOutput, KeyframeId), TrackerError> {
        let frame_id = left.frame_id();
        let max_keypoints = self.config.max_keypoints();

        let (left_arc, right_arc) = match left_det {
            Some(left_arc) => {
                let right_det = self
                    .superpoint_right
                    .detect_with_downscale(&right, self.config.downscale)?
                    .top_k(max_keypoints);
                (left_arc, Arc::new(right_det))
            }
            None => {
                let (left_det, right_det) = std::thread::scope(|scope| {
                    let left_sp = &mut self.superpoint_left;
                    let right_sp = &mut self.superpoint_right;
                    let left_ref = &left;
                    let right_ref = &right;
                    let downscale = self.config.downscale;

                    let left_handle = scope.spawn(move || {
                        left_sp
                            .detect_with_downscale(left_ref, downscale)
                            .map(|d| d.top_k(max_keypoints))
                    });
                    let right_handle = scope.spawn(move || {
                        right_sp
                            .detect_with_downscale(right_ref, downscale)
                            .map(|d| d.top_k(max_keypoints))
                    });

                    (left_handle.join(), right_handle.join())
                });

                let left_det = left_det.map_err(|_| InferenceError::ThreadPanic {
                    stage: "left superpoint",
                })??;
                let right_det = right_det.map_err(|_| InferenceError::ThreadPanic {
                    stage: "right superpoint",
                })??;

                (Arc::new(left_det), Arc::new(right_det))
            }
        };

        let matches = if left_arc.is_empty() || right_arc.is_empty() {
            return Err(TrackerError::KeyframeRejected { landmarks: 0 });
        } else {
            self.lightglue
                .match_these(left_arc.clone(), right_arc.clone())?
        };

        let result = self.triangulator.triangulate(&matches)?;
        let triangulation_stats = result.stats;
        let landmarks = result.keyframe.landmarks().len();
        if landmarks < self.config.min_keyframe_points {
            if self.trace_transitions {
                eprintln!(
                    "keyframe rejected frame={} landmarks={} min_required={}",
                    frame_id.as_u64(),
                    landmarks,
                    self.config.min_keyframe_points
                );
            }
            return Err(TrackerError::KeyframeRejected { landmarks });
        }

        let keyframe = Arc::new(result.keyframe);
        let (staged_map, staged_graph, keyframe_id) = stage_keyframe_in_map_and_graph(
            &self.map,
            &self.essential_graph,
            &keyframe,
            left.timestamp(),
            pose_world,
            &connection,
            self.cull_min_observations,
        )?;
        self.enqueue_loop_candidates(keyframe_id, keyframe.detections())?;
        self.map = staged_map;
        self.essential_graph = staged_graph;
        self.emit_event(DiagnosticEvent::KeyframeCreated {
            keyframe_id,
            landmarks,
        });
        self.enqueue_descriptor_request(keyframe_id, &left);

        let mut diagnostics = self.empty_diagnostics();
        diagnostics.keyframe_created = true;
        diagnostics.triangulation = Some(triangulation_stats);
        diagnostics.features_detected = Some(left_arc.len());
        diagnostics.features_matched = Some(matches.len());

        Ok((
            TrackerOutput {
                pose: None,
                pose_status: PoseStatus::Unavailable,
                inliers: 0,
                keyframe: Some(keyframe),
                stereo_matches: Some(matches),
                frame_id,
                health: self.system_health(),
                diagnostics,
                events: Vec::new(),
            },
            keyframe_id,
        ))
    }
}

fn register_bootstrap_loop_descriptor(
    config: Option<LoopClosureConfig>,
    loop_db: Option<&mut KeyframeDatabase>,
    keyframe_id: KeyframeId,
    detections: &Detections,
) -> Result<Option<(LoopClosureConfig, Vec<PlaceMatch>)>, TrackerError> {
    let (config, loop_db) = match (config, loop_db) {
        (None, None) => return Ok(None),
        (Some(config), Some(loop_db)) => (config, loop_db),
        _ => {
            return Err(TrackerError::InvariantViolation(
                "loop configuration and keyframe database must be enabled together",
            ));
        }
    };
    let descriptor = aggregate_global_descriptor(detections.descriptors())?;
    let candidates = loop_db.insert_with_source_and_query(
        keyframe_id,
        descriptor,
        DescriptorSource::Bootstrap,
        config.max_candidates(),
    )?;
    Ok(Some((config, candidates)))
}

fn replace_mapping_session(
    map: &mut SlamMap,
    essential_graph: &mut EssentialGraph,
    loop_db: Option<&mut KeyframeDatabase>,
    strong_threshold: u32,
) -> MappingSessionTransition {
    let old_map = map.snapshot().instance_id();
    *map = SlamMap::new();
    *essential_graph = EssentialGraph::new(strong_threshold);
    if let Some(loop_db) = loop_db {
        *loop_db = KeyframeDatabase::new(loop_db.temporal_gap());
    }
    MappingSessionTransition::try_new(old_map, map.snapshot().instance_id())
        .expect("a fresh SlamMap has a distinct map instance ID")
}

#[cfg(test)]
fn insert_keyframe_into_map(
    map: &mut SlamMap,
    keyframe: &Arc<Keyframe>,
    timestamp: Timestamp,
    pose_world: crate::WorldToCamera,
    shared: Option<&SharedMatches>,
    cull_min_observations: NonZeroUsize,
) -> Result<KeyframeId, TrackerError> {
    let (staged, keyframe_id) = stage_keyframe_in_map(
        map,
        keyframe,
        timestamp,
        pose_world,
        shared,
        cull_min_observations,
    )?;
    *map = staged;
    Ok(keyframe_id)
}

#[cfg(test)]
fn insert_keyframe_into_map_and_graph(
    map: &mut SlamMap,
    essential_graph: &mut EssentialGraph,
    keyframe: &Arc<Keyframe>,
    timestamp: Timestamp,
    pose_world: crate::WorldToCamera,
    connection: &KeyframeConnection,
    cull_min_observations: NonZeroUsize,
) -> Result<KeyframeId, TrackerError> {
    let (staged_map, staged_graph, keyframe_id) = stage_keyframe_in_map_and_graph(
        map,
        essential_graph,
        keyframe,
        timestamp,
        pose_world,
        connection,
        cull_min_observations,
    )?;
    *map = staged_map;
    *essential_graph = staged_graph;
    Ok(keyframe_id)
}

fn stage_keyframe_in_map_and_graph(
    map: &SlamMap,
    essential_graph: &EssentialGraph,
    keyframe: &Arc<Keyframe>,
    timestamp: Timestamp,
    pose_world: crate::WorldToCamera,
    connection: &KeyframeConnection,
    cull_min_observations: NonZeroUsize,
) -> Result<(SlamMap, EssentialGraph, KeyframeId), TrackerError> {
    if let KeyframeConnection::Relocalization(attachment) = connection {
        let current = map.snapshot();
        if attachment.verified_snapshot != current {
            return Err(TrackerError::RelocalizationMapSnapshotMismatch {
                verified: attachment.verified_snapshot,
                current,
            });
        }
    }
    let (staged_map, keyframe_id) = stage_keyframe_in_map(
        map,
        keyframe,
        timestamp,
        pose_world,
        connection.shared_matches(),
        cull_min_observations,
    )?;
    let mut staged_graph = essential_graph.clone();
    match connection {
        KeyframeConnection::Relocalization(attachment) => {
            staged_graph.add_keyframe_with_verified_parent(
                keyframe_id,
                attachment.candidate,
                loop_information_matrix(attachment.inlier_count.get()),
                &staged_map,
            )?;
        }
        KeyframeConnection::Bootstrap | KeyframeConnection::Covisibility(_) => {
            staged_graph.add_keyframe(
                keyframe_id,
                staged_map.covisibility().neighbors(keyframe_id),
                &staged_map,
            )?;
        }
    }

    Ok((staged_map, staged_graph, keyframe_id))
}

fn stage_keyframe_in_map(
    map: &SlamMap,
    keyframe: &Arc<Keyframe>,
    timestamp: Timestamp,
    pose_world: crate::WorldToCamera,
    shared: Option<&SharedMatches>,
    cull_min_observations: NonZeroUsize,
) -> Result<(SlamMap, KeyframeId), TrackerError> {
    let mut staged = map.clone();
    let keyframe_id = staged.add_keyframe_from_detections(
        keyframe.detections().as_ref(),
        timestamp,
        pose_world,
    )?;

    if let Some(shared) = shared {
        for &(current_idx, old_idx) in &shared.pairs {
            let old_kp = staged.keyframe_keypoint(shared.keyframe_id, old_idx)?;
            let Some(point_id) = staged.map_point_for_keypoint(old_kp)? else {
                continue;
            };
            let new_kp = staged.keyframe_keypoint(keyframe_id, current_idx)?;
            if staged.map_point_for_keypoint(new_kp)?.is_none() {
                staged.add_observation(point_id, new_kp)?;
            }
        }
    }

    // Keep singleton points by default so active keyframes retain enough
    // point associations for robust PnP.
    if cull_min_observations.get() > 1 && staged.num_points() > 0 {
        let points_before = staged.num_points();
        let culled_points = staged.cull_points(cull_min_observations.get());
        debug_assert!(
            culled_points <= points_before,
            "culled more points than existed"
        );
    }

    for (landmark, &det_idx) in keyframe
        .landmarks()
        .iter()
        .zip(keyframe.landmark_indices().iter())
    {
        let keypoint_ref = staged.keyframe_keypoint(keyframe_id, det_idx)?;
        if staged.map_point_for_keypoint(keypoint_ref)?.is_some() {
            continue;
        }
        let descriptor = keyframe.detections().descriptors()[det_idx].quantize();
        let world = camera_to_world(pose_world, *landmark)?;
        staged.add_map_point(world, descriptor, keypoint_ref)?;
    }
    Ok((staged, keyframe_id))
}

fn remove_keyframe_from_graph_and_db(
    map: &mut SlamMap,
    essential_graph: &mut EssentialGraph,
    loop_db: Option<&mut KeyframeDatabase>,
    keyframe_id: KeyframeId,
) -> Result<(), TrackerError> {
    let mut staged_map = map.clone();
    let mut staged_graph = essential_graph.clone();
    let staged_db = match loop_db.as_deref() {
        Some(database) => {
            let mut staged = database.clone();
            staged.remove(keyframe_id)?;
            Some(staged)
        }
        None => None,
    };

    staged_graph.remove_keyframe(keyframe_id, &staged_map)?;
    staged_map.remove_keyframe(keyframe_id)?;

    *map = staged_map;
    *essential_graph = staged_graph;
    if let (Some(database), Some(staged)) = (loop_db, staged_db) {
        *database = staged;
    }
    Ok(())
}

fn camera_to_world(
    pose_world: crate::WorldToCamera,
    point: crate::CameraPoint3,
) -> Result<crate::WorldPoint3, TrackerError> {
    Ok(pose_world.try_inverse()?.try_transform_point(point)?)
}

fn build_shared_matches(
    keyframe_id: KeyframeId,
    matches: &Matches<Verified>,
    inliers: &[usize],
) -> SharedMatches {
    let mut pairs = Vec::with_capacity(inliers.len());
    for &idx in inliers {
        if let Some(&(ci, ki)) = matches.indices().get(idx) {
            pairs.push((ci, ki));
        }
    }
    SharedMatches { keyframe_id, pairs }
}

fn is_redundant(
    map: &SlamMap,
    keyframe_id: KeyframeId,
    max_covisibility: f32,
) -> Result<bool, TrackerError> {
    let Some(neighbors) = map.covisibility().neighbors(keyframe_id) else {
        return Ok(false);
    };
    for &neighbor in neighbors.keys() {
        let ratio = map.covisibility_ratio(keyframe_id, neighbor)?;
        if ratio >= max_covisibility {
            return Ok(true);
        }
    }
    Ok(false)
}

fn collect_window_points(
    map: &SlamMap,
    window: &BackendWindow,
) -> Result<Vec<MapPointId>, CorrectionBuildError> {
    let mut points = Vec::new();
    let mut seen = HashSet::new();
    for &keyframe_id in window.as_slice() {
        let keyframe = map
            .keyframe(keyframe_id)
            .ok_or(CorrectionBuildError::MissingKeyframe { keyframe_id })?;
        for index in 0..keyframe.len() {
            let keypoint_ref = map.keyframe_keypoint(keyframe_id, index)?;
            let Some(point_id) = map.map_point_for_keypoint(keypoint_ref)? else {
                continue;
            };
            if seen.insert(point_id) {
                points.push(point_id);
            }
        }
    }
    Ok(points)
}

fn apply_correction_event(
    map: &mut SlamMap,
    correction: &CorrectionEvent,
) -> Result<(), ApplyCorrectionError> {
    let current_snapshot = map.snapshot();
    if correction.source_snapshot != current_snapshot {
        return Err(ApplyCorrectionError::StaleSnapshot {
            current: current_snapshot,
            correction: correction.source_snapshot,
        });
    }

    for (keyframe_id, _) in &correction.correction.corrected_poses {
        if map.keyframe(*keyframe_id).is_none() {
            return Err(ApplyCorrectionError::MissingKeyframe {
                keyframe_id: *keyframe_id,
            });
        }
    }
    for (point_id, _) in &correction.correction.corrected_landmarks {
        if map.point(*point_id).is_none() {
            return Err(ApplyCorrectionError::MissingMapPoint {
                point_id: *point_id,
            });
        }
    }

    let mut staged = map.clone();
    for (keyframe_id, corrected_pose) in &correction.correction.corrected_poses {
        staged.set_keyframe_pose(*keyframe_id, *corrected_pose)?;
    }
    for (point_id, corrected_position) in &correction.correction.corrected_landmarks {
        staged.set_map_point_position(*point_id, *corrected_position)?;
    }
    *map = staged;
    Ok(())
}

fn apply_descriptor_response(
    map: &SlamMap,
    loop_db: &mut KeyframeDatabase,
    response: DescriptorResponse,
) -> Result<DescriptorApplyDisposition, TrackerError> {
    if response.source_snapshot != map.snapshot() {
        return Ok(DescriptorApplyDisposition::Stale);
    }
    map.keyframe(response.keyframe_id)
        .ok_or(crate::map::MapError::KeyframeNotFound(response.keyframe_id))?;
    loop_db.replace_descriptor(
        response.keyframe_id,
        response.descriptor,
        DescriptorSource::Learned,
    )?;
    Ok(DescriptorApplyDisposition::Applied)
}

fn apply_loop_closure_correction(
    map: &mut SlamMap,
    essential_graph: &mut EssentialGraph,
    optimizer: &PoseGraphOptimizer,
    verified: &VerifiedLoop,
) -> Result<Vec<(KeyframeId, Pose)>, TrackerError> {
    let current_snapshot = map.snapshot();
    if verified.map_snapshot() != current_snapshot {
        return Err(TrackerError::LoopMapSnapshotMismatch {
            verified: verified.map_snapshot(),
            current: current_snapshot,
        });
    }
    let query_kf = verified.query_kf();
    let match_kf = verified.match_kf();
    let match_pose = map
        .keyframe(match_kf)
        .ok_or(TrackerError::Map(crate::map::MapError::KeyframeNotFound(
            match_kf,
        )))?
        .pose();
    let query_pose_estimate = verified.query_pose_world();
    let loop_relative = crate::Pose64::try_from_pose32(query_pose_estimate)?.try_compose(
        crate::Pose64::try_from_pose32(match_pose.into_legacy_pose())?.try_inverse()?,
    )?;

    let mut staged_graph = essential_graph.clone();
    staged_graph.add_loop_edge(
        EssentialEdge::try_new(
            match_kf,
            query_kf,
            EssentialEdgeKind::Loop,
            loop_relative,
            loop_information_matrix(verified.inlier_count()),
        )?,
        map,
    )?;

    let input = staged_graph.pose_graph_input(map)?;
    if input.keyframe_ids.len() < MIN_OPTIMIZATION_KEYFRAMES || input.edges.is_empty() {
        *essential_graph = staged_graph;
        return Ok(Vec::new());
    }

    let mut old_poses = HashMap::with_capacity(input.keyframe_ids.len());
    let mut initial_poses = Vec::with_capacity(input.keyframe_ids.len());
    for &keyframe_id in &input.keyframe_ids {
        let pose = map
            .keyframe(keyframe_id)
            .ok_or(TrackerError::Map(crate::map::MapError::KeyframeNotFound(
                keyframe_id,
            )))?
            .pose();
        let legacy_pose = pose.into_legacy_pose();
        old_poses.insert(keyframe_id, legacy_pose);
        initial_poses.push(crate::Pose64::try_from_pose32(legacy_pose)?);
    }

    let result = optimizer.optimize(&input.edges, &mut initial_poses)?;
    if !result.converged {
        return Err(PoseGraphError::OptimizationDidNotConverge {
            iterations: result.iterations,
        }
        .into());
    }
    let corrected_poses: HashMap<KeyframeId, Pose> = input
        .keyframe_ids
        .iter()
        .copied()
        .zip(result.corrected_poses)
        .map(|(keyframe_id, pose)| pose.try_to_pose32().map(|pose| (keyframe_id, pose)))
        .collect::<Result<_, _>>()?;

    let mut staged_map = map.clone();
    for (keyframe_id, corrected_pose) in &corrected_poses {
        staged_map.set_keyframe_pose(
            *keyframe_id,
            crate::WorldToCamera::from_legacy_pose(*corrected_pose),
        )?;
    }

    let mut point_updates = Vec::new();
    for (point_id, point) in staged_map.points() {
        let world = point.position();
        let mut accum = [0.0_f64; 3];
        let mut count = 0usize;

        for observation in point.observations() {
            let keyframe_id = observation.keyframe_id();
            let Some(old_pose) = old_poses.get(&keyframe_id).copied() else {
                continue;
            };
            let Some(new_pose) = corrected_poses.get(&keyframe_id).copied() else {
                continue;
            };

            let camera =
                crate::WorldToCamera::from_legacy_pose(old_pose).try_transform_point(world)?;
            let corrected_world =
                camera_to_world(crate::WorldToCamera::from_legacy_pose(new_pose), camera)?;
            accum[0] += f64::from(corrected_world.x);
            accum[1] += f64::from(corrected_world.y);
            accum[2] += f64::from(corrected_world.z);
            count = count
                .checked_add(1)
                .ok_or(TrackerError::InvariantViolation(
                    "map-point observation count overflow",
                ))?;
        }

        if count > 0 {
            let count = u32::try_from(count).map_err(|_| {
                TrackerError::InvariantViolation(
                    "map-point observation count exceeds the exact averaging domain",
                )
            })?;
            let inv_count = 1.0_f64 / f64::from(count);
            let corrected = crate::WorldPoint3::try_from_f64([
                accum[0] * inv_count,
                accum[1] * inv_count,
                accum[2] * inv_count,
            ])
            .map_err(crate::map::MapError::InvalidMapPointPosition)?;
            point_updates.push((point_id, corrected));
        }
    }

    for (point_id, corrected_world) in point_updates {
        staged_map.set_map_point_position(point_id, corrected_world)?;
    }

    *map = staged_map;
    *essential_graph = staged_graph;
    Ok(corrected_poses.into_iter().collect())
}

fn loop_pose_correction(
    current_query_pose: Pose,
    estimated_query_pose: Pose,
) -> Result<Pose, crate::PoseError> {
    estimated_query_pose.try_compose(current_query_pose.try_inverse()?)
}

fn loop_translation_norm(pose: Pose) -> f32 {
    let t = pose.translation();
    t[0].hypot(t[1]).hypot(t[2])
}

fn loop_apply_error_kind(error: &TrackerError) -> LoopApplyError {
    match error {
        TrackerError::LoopMapSnapshotMismatch { .. } => LoopApplyError::StaleCorrection,
        TrackerError::Map(crate::map::MapError::KeyframeNotFound(_)) => {
            LoopApplyError::MissingKeyframe
        }
        TrackerError::Map(crate::map::MapError::MapPointNotFound(_)) => {
            LoopApplyError::MissingMapPoint
        }
        TrackerError::Map(_) => LoopApplyError::MapMutation,
        TrackerError::EssentialGraph(_) => LoopApplyError::EssentialGraph,
        TrackerError::PoseGraph(_) => LoopApplyError::PoseOptimization,
        TrackerError::Pose(_)
        | TrackerError::Transform(_)
        | TrackerError::Pose64(_)
        | TrackerError::PoseNarrowing(_) => LoopApplyError::PoseConversion,
        TrackerError::InvariantViolation(_) => LoopApplyError::InvariantViolation,
        _ => LoopApplyError::UnexpectedFailure,
    }
}

fn loop_reject_reason(error: &LoopDetectError) -> LoopClosureRejectReason {
    match error {
        LoopDetectError::TooFewCorrespondences { count } => {
            LoopClosureRejectReason::TooFewCorrespondences { count: *count }
        }
        LoopDetectError::DescriptorMatchFailed(_) => LoopClosureRejectReason::VerificationFailed,
        LoopDetectError::VerificationFailed(_) => LoopClosureRejectReason::VerificationFailed,
        LoopDetectError::Pose(_) => LoopClosureRejectReason::ApplyFailed,
        LoopDetectError::CorrectionTooLarge {
            translation,
            rotation_deg,
        } => LoopClosureRejectReason::CorrectionTooLarge {
            translation_m: *translation,
            rotation_deg: *rotation_deg,
        },
        LoopDetectError::ApplyFailed(_) => LoopClosureRejectReason::ApplyFailed,
    }
}

fn loop_rotation_angle_deg(pose: Pose) -> f32 {
    let r = pose.rotation();
    let trace = r[0][0] + r[1][1] + r[2][2];
    let cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0);
    cos_theta.acos().to_degrees()
}

fn loop_information_matrix(inlier_count: usize) -> [[f64; 6]; 6] {
    let weight = inlier_count.max(1) as f64;
    let mut info = [[0.0_f64; 6]; 6];
    for (axis, row) in info.iter_mut().enumerate() {
        row[axis] = weight;
    }
    info
}

struct ResolvedMapObservations {
    observations: Vec<crate::Observation>,
    match_indices: Vec<usize>,
}

fn build_map_observations(
    map: &SlamMap,
    keyframe_id: KeyframeId,
    matches: &Matches<Verified>,
    current: &Detections,
) -> Result<ResolvedMapObservations, crate::PnpError> {
    let mut observations = Vec::with_capacity(matches.len());
    let mut match_indices = Vec::with_capacity(matches.len());
    let current_len = current.len();

    for (match_index, &(ci, ki)) in matches.indices().iter().enumerate() {
        if ci >= current_len {
            return Err(crate::PnpError::IndexOutOfBounds {
                current_len,
                keyframe_len: 0,
                current_index: ci,
                keyframe_index: ki,
            });
        }
        let keypoint_ref = map.keyframe_keypoint(keyframe_id, ki)?;
        let Some(point_id) = map.map_point_for_keypoint(keypoint_ref)? else {
            continue;
        };
        let point = map
            .point(point_id)
            .ok_or(crate::map::MapError::MapPointNotFound(point_id))?;
        let pixel = current.keypoints()[ci];
        let obs = crate::Observation::try_new(point.position(), pixel)?;
        observations.push(obs);
        match_indices.push(match_index);
    }

    if observations.len() < MIN_PNP_CORRESPONDENCES {
        return Err(crate::PnpError::NotEnoughPoints {
            required: MIN_PNP_CORRESPONDENCES,
            actual: observations.len(),
        });
    }
    Ok(ResolvedMapObservations {
        observations,
        match_indices,
    })
}

fn median_parallax_px(
    matches: &Matches<Verified>,
    inliers: &[usize],
    keyframe: &Keyframe,
) -> Option<f32> {
    if inliers.is_empty() {
        return None;
    }

    let left_kps = matches.source_a().keypoints();
    let key_kps = keyframe.detections().keypoints();
    let mut parallax = Vec::with_capacity(inliers.len());

    for &idx in inliers {
        let Some(&(li, ki)) = matches.indices().get(idx) else {
            continue;
        };
        let (Some(left), Some(key)) = (left_kps.get(li), key_kps.get(ki)) else {
            continue;
        };
        let dx = left.x - key.x;
        let dy = left.y - key.y;
        parallax.push((dx * dx + dy * dy).sqrt());
    }

    if parallax.is_empty() {
        return None;
    }

    parallax.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let mid = parallax.len() / 2;
    let median = if parallax.len() % 2 == 0 {
        (parallax[mid - 1] + parallax[mid]) * 0.5
    } else {
        parallax[mid]
    };

    Some(median)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::assert_map_invariants;
    use crate::{CompactDescriptor, Descriptor, Detections, Keypoint, Point3, SensorId, Timestamp};
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
    use std::time::Duration;

    #[test]
    fn tracker_environment_defaults_only_absent_values() {
        assert_eq!(
            TrackerEnvironment::from_parsed(
                DEFAULT_MAX_RESPAWNS,
                DEFAULT_MAX_RESPAWNS,
                None,
                None,
            )
            .expect("default tracker environment"),
            TrackerEnvironment {
                backend_max_respawns: DEFAULT_MAX_RESPAWNS,
                descriptor_max_respawns: DEFAULT_MAX_RESPAWNS,
                cull_min_observations: DEFAULT_CULL_MIN_OBSERVATIONS,
                trace_transitions: false,
            }
        );
    }

    #[test]
    fn tracker_environment_preserves_typed_values() {
        assert_eq!(
            TrackerEnvironment::from_parsed(u32::MAX, 0, Some(7), Some(true))
                .expect("explicit tracker environment"),
            TrackerEnvironment {
                backend_max_respawns: u32::MAX,
                descriptor_max_respawns: 0,
                cull_min_observations: NonZeroUsize::new(7).expect("non-zero threshold"),
                trace_transitions: true,
            }
        );
    }

    #[test]
    fn disabled_setting_does_not_invoke_its_parser() {
        let value = load_when_enabled(
            false,
            DEFAULT_MAX_RESPAWNS,
            || -> Result<Option<u32>, std::convert::Infallible> {
                panic!("disabled setting parser must not run")
            },
        )
        .expect("disabled setting uses its typed default");
        assert_eq!(value, DEFAULT_MAX_RESPAWNS);
    }

    #[test]
    fn tracker_environment_rejects_zero_cull_threshold() {
        assert!(matches!(
            TrackerEnvironment::from_parsed(
                DEFAULT_MAX_RESPAWNS,
                DEFAULT_MAX_RESPAWNS,
                Some(0),
                None
            ),
            Err(TrackerInitError::ZeroMapCullMinObservations {
                variable: MAP_CULL_MIN_OBSERVATIONS_ENV
            })
        ));
    }

    fn make_descriptor() -> Descriptor {
        Descriptor([0.0; 256])
    }

    fn make_single_landmark_keyframe(frame_id: u64) -> Arc<Keyframe> {
        make_single_landmark_keyframe_with_descriptor(frame_id, make_descriptor())
    }

    fn make_single_landmark_keyframe_with_descriptor(
        frame_id: u64,
        descriptor: Descriptor,
    ) -> Arc<Keyframe> {
        let detections = Arc::new(
            Detections::new(
                SensorId::StereoLeft,
                FrameId::new(frame_id),
                320,
                240,
                vec![Keypoint { x: 100.0, y: 80.0 }],
                vec![1.0],
                vec![descriptor],
            )
            .expect("single-landmark detections"),
        );
        Arc::new(
            Keyframe::from_arc(
                detections,
                vec![Point3 {
                    x: 0.0,
                    y: 0.0,
                    z: 1.0,
                }],
                vec![0],
            )
            .expect("single-landmark keyframe"),
        )
    }

    fn make_relocalization_detections(descriptor: Descriptor) -> Detections {
        Detections::new(
            SensorId::StereoLeft,
            FrameId::new(1),
            320,
            240,
            vec![Keypoint { x: 160.0, y: 120.0 }],
            vec![1.0],
            vec![descriptor],
        )
        .expect("valid relocalization detections")
    }

    fn make_relocalization_attachment(candidate: KeyframeId) -> RelocalizationAttachment {
        RelocalizationAttachment {
            candidate,
            verified_snapshot: SlamMap::new().snapshot(),
            inlier_count: NonZeroUsize::new(MIN_PNP_CORRESPONDENCES)
                .expect("PnP correspondence minimum is nonzero"),
        }
    }

    fn make_global_descriptor_basis(idx: usize) -> crate::loop_closure::GlobalDescriptor {
        let mut data = [0.0_f32; 512];
        data[idx % 512] = 1.0;
        crate::loop_closure::GlobalDescriptor::try_new(data).expect("basis descriptor")
    }

    struct StubDescriptorExtractor {
        descriptor: crate::loop_closure::GlobalDescriptor,
        calls: Arc<Mutex<usize>>,
    }

    impl PlaceDescriptorExtractor for StubDescriptorExtractor {
        fn backend_name(&self) -> &'static str {
            "stub"
        }

        fn compute_descriptor(
            &mut self,
            _frame: &Frame,
        ) -> Result<crate::loop_closure::GlobalDescriptor, InferenceError> {
            let mut calls = self.calls.lock().expect("calls lock");
            *calls = calls.saturating_add(1);
            Ok(self.descriptor.clone())
        }
    }

    struct PanicDescriptorExtractor;

    impl PlaceDescriptorExtractor for PanicDescriptorExtractor {
        fn backend_name(&self) -> &'static str {
            "panic"
        }

        fn compute_descriptor(
            &mut self,
            _frame: &Frame,
        ) -> Result<crate::loop_closure::GlobalDescriptor, InferenceError> {
            panic!("forced descriptor panic");
        }
    }

    fn make_map_with_single_point() -> (SlamMap, KeyframeId, MapPointId) {
        let detections = Detections::new(
            SensorId::StereoLeft,
            FrameId::new(1),
            320,
            240,
            vec![Keypoint { x: 100.0, y: 80.0 }],
            vec![1.0],
            vec![make_descriptor()],
        )
        .expect("detections");
        let mut map = SlamMap::new();
        let keyframe_id = map
            .add_keyframe_from_detections(
                &detections,
                Timestamp::from_nanos(1),
                crate::WorldToCamera::identity(),
            )
            .expect("keyframe");
        let keypoint = map.keyframe_keypoint(keyframe_id, 0).expect("keypoint ref");
        let point_id = map
            .add_map_point(
                Point3 {
                    x: 0.0,
                    y: 0.0,
                    z: 1.0,
                },
                make_descriptor().quantize(),
                keypoint,
            )
            .expect("map point");
        (map, keyframe_id, point_id)
    }

    fn make_map_with_two_keyframes_one_shared_point() -> (SlamMap, KeyframeId, KeyframeId) {
        let detections_a = Detections::new(
            SensorId::StereoLeft,
            FrameId::new(10),
            320,
            240,
            vec![Keypoint { x: 100.0, y: 80.0 }],
            vec![1.0],
            vec![make_descriptor()],
        )
        .expect("detections a");
        let detections_b = Detections::new(
            SensorId::StereoLeft,
            FrameId::new(11),
            320,
            240,
            vec![Keypoint { x: 110.0, y: 82.0 }],
            vec![1.0],
            vec![make_descriptor()],
        )
        .expect("detections b");

        let mut map = SlamMap::new();
        let kf_a = map
            .add_keyframe_from_detections(
                &detections_a,
                Timestamp::from_nanos(10),
                crate::WorldToCamera::identity(),
            )
            .expect("kf a");
        let kf_b = map
            .add_keyframe_from_detections(
                &detections_b,
                Timestamp::from_nanos(11),
                crate::WorldToCamera::identity(),
            )
            .expect("kf b");
        let kp_a = map.keyframe_keypoint(kf_a, 0).expect("kp a");
        let point_id = map
            .add_map_point(
                Point3 {
                    x: 0.0,
                    y: 0.0,
                    z: 1.0,
                },
                make_descriptor().quantize(),
                kp_a,
            )
            .expect("point");
        let kp_b = map.keyframe_keypoint(kf_b, 0).expect("kp b");
        map.add_observation(point_id, kp_b).expect("shared obs");
        (map, kf_a, kf_b)
    }

    fn make_forced_panic_event(
        request_id: BackendRequestId,
        map: SlamMap,
        kf_a: KeyframeId,
        kf_b: KeyframeId,
    ) -> KeyframeEvent {
        let window = BackendWindow::try_new(vec![kf_a, kf_b]).expect("window");
        let mut event = KeyframeEvent::try_new(request_id, kf_b, window, map).expect("event");
        event.force_panic = true;
        event
    }

    #[test]
    fn keyframe_insertion_populates_map_points() {
        let keypoints = vec![
            Keypoint { x: 1.0, y: 2.0 },
            Keypoint { x: 3.0, y: 4.0 },
            Keypoint { x: 5.0, y: 6.0 },
        ];
        let scores = vec![1.0, 1.0, 1.0];
        let descriptors = vec![make_descriptor(), make_descriptor(), make_descriptor()];

        let detections = Detections::new(
            SensorId::StereoLeft,
            FrameId::new(10),
            640,
            480,
            keypoints,
            scores,
            descriptors,
        )
        .expect("detections");

        let landmarks = vec![
            Point3 {
                x: 0.0,
                y: 0.0,
                z: 1.0,
            },
            Point3 {
                x: 1.0,
                y: 0.0,
                z: 1.5,
            },
        ];
        let landmark_indices = vec![0, 2];
        let keyframe = Arc::new(
            Keyframe::from_arc(Arc::new(detections), landmarks, landmark_indices)
                .expect("keyframe"),
        );

        let mut map = SlamMap::new();
        assert_map_invariants(&map).expect("empty map invariants");
        let keyframe_id = insert_keyframe_into_map(
            &mut map,
            &keyframe,
            Timestamp::from_nanos(42),
            crate::WorldToCamera::identity(),
            None,
            DEFAULT_CULL_MIN_OBSERVATIONS,
        )
        .expect("insert keyframe");
        assert_map_invariants(&map).expect("post-insertion invariants");

        assert_eq!(map.num_keyframes(), 1);
        assert_eq!(map.num_points(), keyframe.landmarks().len());

        for &det_idx in keyframe.landmark_indices() {
            let kp_ref = map.keyframe_keypoint(keyframe_id, det_idx).expect("kp ref");
            let point_id = map
                .map_point_for_keypoint(kp_ref)
                .expect("map lookup")
                .expect("point id");
            let point = map.point(point_id).expect("point");
            let landmark = keyframe.landmark_for_detection(det_idx).expect("landmark");
            let Point3 { x, y, z } = point.position();
            assert_eq!(x, landmark.x);
            assert_eq!(y, landmark.y);
            assert_eq!(z, landmark.z);
        }
        assert_map_invariants(&map).expect("final invariants");
    }

    #[test]
    fn keyframe_insertion_rolls_back_after_shared_match_failure() {
        let detections = Arc::new(
            Detections::new(
                SensorId::StereoLeft,
                FrameId::new(20),
                640,
                480,
                vec![Keypoint { x: 10.0, y: 20.0 }],
                vec![1.0],
                vec![make_descriptor()],
            )
            .expect("detections"),
        );
        let keyframe = Arc::new(
            Keyframe::from_arc(
                detections,
                vec![Point3 {
                    x: 0.0,
                    y: 0.0,
                    z: 1.0,
                }],
                vec![0],
            )
            .expect("keyframe"),
        );
        let mut map = SlamMap::new();
        let generation_before = map.generation();
        let invalid_shared = SharedMatches {
            keyframe_id: KeyframeId::default(),
            pairs: vec![(0, 0)],
        };

        let error = insert_keyframe_into_map(
            &mut map,
            &keyframe,
            Timestamp::from_nanos(20),
            crate::WorldToCamera::identity(),
            Some(&invalid_shared),
            DEFAULT_CULL_MIN_OBSERVATIONS,
        )
        .expect_err("invalid shared keyframe must fail insertion");

        assert!(matches!(
            error,
            TrackerError::Map(crate::map::MapError::KeyframeNotFound(_))
        ));
        assert_eq!(map.num_keyframes(), 0);
        assert_eq!(map.num_points(), 0);
        assert_eq!(map.generation(), generation_before);
        assert_map_invariants(&map).expect("rolled-back map invariants");
    }

    #[test]
    fn keyframe_insertion_rolls_back_map_and_graph_when_topology_is_disconnected() {
        let mut map = SlamMap::new();
        let mut graph = EssentialGraph::new(2);
        insert_keyframe_into_map_and_graph(
            &mut map,
            &mut graph,
            &make_single_landmark_keyframe(30),
            Timestamp::from_nanos(30),
            crate::WorldToCamera::identity(),
            &KeyframeConnection::Bootstrap,
            DEFAULT_CULL_MIN_OBSERVATIONS,
        )
        .expect("first keyframe establishes the root");
        let map_before = map.snapshot();
        let points_before = map.num_points();
        let graph_before = graph.snapshot();

        let error = insert_keyframe_into_map_and_graph(
            &mut map,
            &mut graph,
            &make_single_landmark_keyframe(31),
            Timestamp::from_nanos(31),
            crate::WorldToCamera::identity(),
            &KeyframeConnection::Bootstrap,
            DEFAULT_CULL_MIN_OBSERVATIONS,
        )
        .expect_err("an isolated second keyframe must abort both staged updates");

        assert!(matches!(
            error,
            TrackerError::EssentialGraph(EssentialGraphError::DisconnectedKeyframe { .. })
        ));
        assert_eq!(map.snapshot(), map_before);
        assert_eq!(map.num_keyframes(), 1);
        assert_eq!(map.num_points(), points_before);
        assert_eq!(graph.snapshot().order, graph_before.order);
        assert_eq!(graph.snapshot().parent, graph_before.parent);
        assert_map_invariants(&map).expect("rolled-back map invariants");
    }

    #[test]
    fn bootstrap_descriptor_failure_leaves_map_graph_and_database_uncommitted() {
        let map = SlamMap::new();
        let graph = EssentialGraph::new(2);
        let keyframe = make_single_landmark_keyframe(32);
        let map_before = map.snapshot();
        let graph_before = graph.snapshot();
        let (staged_map, staged_graph, keyframe_id) = stage_keyframe_in_map_and_graph(
            &map,
            &graph,
            &keyframe,
            Timestamp::from_nanos(32),
            crate::WorldToCamera::identity(),
            &KeyframeConnection::Bootstrap,
            DEFAULT_CULL_MIN_OBSERVATIONS,
        )
        .expect("map and graph staging succeeds");
        assert_eq!(staged_map.num_keyframes(), 1);
        assert_eq!(staged_graph.parent_of(keyframe_id), Some(keyframe_id));
        let mut loop_db = KeyframeDatabase::new(0);

        let error = register_bootstrap_loop_descriptor(
            Some(LoopClosureConfig::default()),
            Some(&mut loop_db),
            keyframe_id,
            keyframe.detections(),
        )
        .expect_err("zero-norm bootstrap descriptor must propagate");

        assert!(matches!(
            error,
            TrackerError::GlobalDescriptor(GlobalDescriptorError::ZeroNorm)
        ));
        assert_eq!(map.snapshot(), map_before);
        assert_eq!(graph.snapshot().order, graph_before.order);
        assert!(loop_db.is_empty());
    }

    #[test]
    fn bootstrap_database_failure_leaves_map_and_graph_uncommitted() {
        let map = SlamMap::new();
        let graph = EssentialGraph::new(2);
        let mut descriptor = [0.0; 256];
        descriptor[0] = 1.0;
        let keyframe = make_single_landmark_keyframe_with_descriptor(33, Descriptor(descriptor));
        let map_before = map.snapshot();
        let graph_before = graph.snapshot();
        let (_staged_map, _staged_graph, keyframe_id) = stage_keyframe_in_map_and_graph(
            &map,
            &graph,
            &keyframe,
            Timestamp::from_nanos(33),
            crate::WorldToCamera::identity(),
            &KeyframeConnection::Bootstrap,
            DEFAULT_CULL_MIN_OBSERVATIONS,
        )
        .expect("map and graph staging succeeds");
        let mut loop_db = KeyframeDatabase::new(0);
        loop_db
            .insert(keyframe_id, make_global_descriptor_basis(7))
            .expect("inject existing registration");

        let error = register_bootstrap_loop_descriptor(
            Some(LoopClosureConfig::default()),
            Some(&mut loop_db),
            keyframe_id,
            keyframe.detections(),
        )
        .expect_err("duplicate database registration must propagate");

        assert!(matches!(
            error,
            TrackerError::KeyframeDatabase(KeyframeDatabaseError::DuplicateKeyframe {
                keyframe_id: duplicate
            }) if duplicate == keyframe_id
        ));
        assert_eq!(map.snapshot(), map_before);
        assert_eq!(graph.snapshot().order, graph_before.order);
        assert_eq!(loop_db.len(), 1);
    }

    #[test]
    fn relocalized_keyframe_transaction_attaches_to_verified_candidate_without_covisibility() {
        let mut map = SlamMap::new();
        let mut graph = EssentialGraph::new(2);
        let root = insert_keyframe_into_map_and_graph(
            &mut map,
            &mut graph,
            &make_single_landmark_keyframe(40),
            Timestamp::from_nanos(40),
            crate::WorldToCamera::identity(),
            &KeyframeConnection::Bootstrap,
            DEFAULT_CULL_MIN_OBSERVATIONS,
        )
        .expect("root keyframe");
        let verified_snapshot = map.snapshot();
        let inlier_count = NonZeroUsize::new(9).expect("non-zero inlier count");
        let attachment = RelocalizationAttachment {
            candidate: root,
            verified_snapshot,
            inlier_count,
        };
        let recovered_pose = crate::WorldToCamera::from_legacy_pose(Pose::from_rt(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [0.5, -0.1, 0.2],
        ));

        let recovered = insert_keyframe_into_map_and_graph(
            &mut map,
            &mut graph,
            &make_single_landmark_keyframe(41),
            Timestamp::from_nanos(41),
            recovered_pose,
            &KeyframeConnection::Relocalization(attachment),
            DEFAULT_CULL_MIN_OBSERVATIONS,
        )
        .expect("verified relocalization must attach without invented observations");

        assert!(map.covisibility().neighbors(recovered).is_none());
        assert_eq!(graph.parent_of(recovered), Some(root));
        let snapshot = graph.snapshot();
        let edge = snapshot
            .spanning_edges
            .iter()
            .find(|edge| edge.b() == recovered)
            .expect("relocalization spanning edge");
        assert_eq!(edge.a(), root);
        assert_eq!(
            edge.information(),
            loop_information_matrix(inlier_count.get())
        );
        graph
            .pose_graph_input(&map)
            .expect("relocalized graph is connected");

        let map_before_stale_attempt = map.snapshot();
        let graph_before_stale_attempt = graph.snapshot();
        let error = insert_keyframe_into_map_and_graph(
            &mut map,
            &mut graph,
            &make_single_landmark_keyframe(42),
            Timestamp::from_nanos(42),
            recovered_pose,
            &KeyframeConnection::Relocalization(attachment),
            DEFAULT_CULL_MIN_OBSERVATIONS,
        )
        .expect_err("attachment snapshot cannot cross a map generation");
        assert!(matches!(
            error,
            TrackerError::RelocalizationMapSnapshotMismatch { .. }
        ));
        assert_eq!(map.snapshot(), map_before_stale_attempt);
        assert_eq!(graph.snapshot().order, graph_before_stale_attempt.order);
    }

    #[test]
    fn exhausted_relocalization_starts_fresh_session_and_rejects_old_async_results() {
        let (mut map, old_keyframe, old_point) = make_map_with_single_point();
        let mut graph = EssentialGraph::new(2);
        graph
            .add_keyframe(old_keyframe, None, &map)
            .expect("old graph root");
        let descriptor = make_global_descriptor_basis(5);
        let mut loop_db = KeyframeDatabase::new(17);
        loop_db
            .insert(old_keyframe, descriptor.clone())
            .expect("unique keyframe");
        let old_snapshot = map.snapshot();
        let correction = CorrectionEvent {
            request_id: BackendRequestId(NonZeroU64::new(99).expect("non-zero")),
            source_snapshot: old_snapshot,
            trigger_keyframe: old_keyframe,
            correction: BackendCorrection {
                corrected_poses: vec![(old_keyframe, crate::WorldToCamera::identity())],
                corrected_landmarks: vec![(old_point, crate::WorldPoint3::new(0.0, 0.0, 1.0))],
                result: BaResult::Converged {
                    iterations: 1,
                    final_cost: 0.0,
                },
            },
        };

        let transition = replace_mapping_session(&mut map, &mut graph, Some(&mut loop_db), 2);

        let fresh_snapshot = map.snapshot();
        assert_eq!(transition.old_map(), old_snapshot.instance_id());
        assert_eq!(transition.new_map(), fresh_snapshot.instance_id());
        assert_ne!(fresh_snapshot.instance_id(), old_snapshot.instance_id());
        assert_eq!(map.num_keyframes(), 0);
        assert_eq!(map.num_points(), 0);
        assert!(graph.snapshot().order.is_empty());
        assert!(loop_db.is_empty());
        assert_eq!(loop_db.temporal_gap(), 17);
        assert!(matches!(
            apply_correction_event(&mut map, &correction),
            Err(ApplyCorrectionError::StaleSnapshot { .. })
        ));
        assert_eq!(
            apply_descriptor_response(
                &map,
                &mut loop_db,
                DescriptorResponse {
                    keyframe_id: old_keyframe,
                    source_snapshot: old_snapshot,
                    descriptor,
                },
            )
            .expect("stale response is nonfatal"),
            DescriptorApplyDisposition::Stale
        );

        let fresh_root = insert_keyframe_into_map_and_graph(
            &mut map,
            &mut graph,
            &make_single_landmark_keyframe(43),
            Timestamp::from_nanos(43),
            crate::WorldToCamera::identity(),
            &KeyframeConnection::Bootstrap,
            DEFAULT_CULL_MIN_OBSERVATIONS,
        )
        .expect("fresh session must permit a new identity-frame root");
        assert_eq!(graph.parent_of(fresh_root), Some(fresh_root));
        assert_ne!(fresh_root.map_instance_id(), old_keyframe.map_instance_id());
    }

    #[test]
    fn map_observation_batch_preserves_original_match_indices() {
        let keypoints: Vec<Keypoint> = (0..5)
            .map(|idx| Keypoint {
                x: 100.0 + idx as f32 * 10.0,
                y: 80.0,
            })
            .collect();
        let reference = Arc::new(
            Detections::new(
                SensorId::StereoLeft,
                FrameId::new(30),
                320,
                240,
                keypoints.clone(),
                vec![1.0; 5],
                vec![make_descriptor(); 5],
            )
            .expect("reference detections"),
        );
        let current = Arc::new(
            Detections::new(
                SensorId::StereoLeft,
                FrameId::new(31),
                320,
                240,
                keypoints,
                vec![1.0; 5],
                vec![make_descriptor(); 5],
            )
            .expect("current detections"),
        );
        let mut map = SlamMap::new();
        let keyframe_id = map
            .add_keyframe_from_detections(
                reference.as_ref(),
                Timestamp::from_nanos(30),
                crate::WorldToCamera::identity(),
            )
            .expect("map keyframe");
        for idx in 1..5 {
            let keypoint = map
                .keyframe_keypoint(keyframe_id, idx)
                .expect("keypoint ref");
            map.add_map_point(
                Point3 {
                    x: idx as f32,
                    y: 0.0,
                    z: 5.0,
                },
                make_descriptor().quantize(),
                keypoint,
            )
            .expect("map point");
        }
        let matches = Matches::new_verified(
            Arc::clone(&current),
            Arc::clone(&reference),
            (0..5).map(|idx| (idx, idx)).collect(),
            vec![1.0; 5],
        )
        .expect("verified matches");
        let batch = build_map_observations(&map, keyframe_id, &matches, current.as_ref())
            .expect("resolved observations");

        assert_eq!(batch.observations.len(), 4);
        assert_eq!(batch.match_indices, vec![1, 2, 3, 4]);
        assert_eq!(batch.observations[0].world().x, 1.0);

        let mut foreign_map = SlamMap::new();
        let foreign_keyframe = foreign_map
            .add_keyframe_from_detections(
                reference.as_ref(),
                Timestamp::from_nanos(32),
                crate::WorldToCamera::identity(),
            )
            .expect("foreign keyframe");
        assert!(matches!(
            build_map_observations(
                &map,
                foreign_keyframe,
                &matches,
                current.as_ref(),
            ),
            Err(crate::PnpError::Map(
                crate::map::MapError::KeyframeNotFound(id)
            )) if id == foreign_keyframe
        ));
    }

    #[test]
    fn backend_window_enforces_non_empty_unique_keyframes() {
        let duplicate = KeyframeId::default();
        assert!(matches!(
            BackendWindow::try_new(vec![duplicate]),
            Err(BackendWindowError::TooFewKeyframes { .. })
        ));
        assert!(matches!(
            BackendWindow::try_new(vec![duplicate, duplicate]),
            Err(BackendWindowError::DuplicateKeyframe { .. })
        ));
    }

    #[test]
    fn backend_request_ids_issue_maximum_once_then_exhaust_stickily() {
        let penultimate = BackendRequestId(
            NonZeroU64::new(u64::MAX - 1).expect("the penultimate u64 is nonzero"),
        );
        let mut request_ids = BackendRequestIds::from_next(penultimate);

        assert_eq!(
            request_ids.take_next().expect("penultimate id").as_u64(),
            u64::MAX - 1
        );
        assert_eq!(
            request_ids.take_next().expect("maximum id").as_u64(),
            u64::MAX
        );
        assert_eq!(request_ids.take_next(), Err(BackendRequestIdExhausted));
        let error = SubmitEventError::RequestIdExhausted(
            request_ids
                .take_next()
                .expect_err("exhaustion must remain sticky"),
        );
        assert!(
            std::error::Error::source(&error)
                .and_then(|source| source.downcast_ref::<BackendRequestIdExhausted>())
                .is_some(),
            "submission errors must retain the typed exhaustion source"
        );
    }

    #[test]
    fn correction_apply_rejects_stale_snapshot() {
        let (mut map, keyframe_id, point_id) = make_map_with_single_point();
        let correction = CorrectionEvent {
            request_id: BackendRequestId(NonZeroU64::new(1).expect("non-zero")),
            source_snapshot: map.snapshot(),
            trigger_keyframe: keyframe_id,
            correction: BackendCorrection {
                corrected_poses: vec![(keyframe_id, crate::WorldToCamera::identity())],
                corrected_landmarks: vec![(point_id, crate::WorldPoint3::new(1.0, 2.0, 3.0))],
                result: BaResult::Converged {
                    iterations: 1,
                    final_cost: 0.0,
                },
            },
        };
        let position = map.point(point_id).expect("map point").position();
        map.set_map_point_position(point_id, position)
            .expect("advance map generation");
        assert!(matches!(
            apply_correction_event(&mut map, &correction),
            Err(ApplyCorrectionError::StaleSnapshot { .. })
        ));
    }

    #[test]
    fn correction_apply_rejects_divergent_clone_at_same_generation() {
        let (base, first_keyframe, second_keyframe) =
            make_map_with_two_keyframes_one_shared_point();
        let keypoint = base.keyframe_keypoint(first_keyframe, 0).expect("keypoint");
        let point_id = base
            .map_point_for_keypoint(keypoint)
            .expect("map lookup")
            .expect("shared point");
        let mut correction_source = base.clone();
        let mut current = base.clone();
        correction_source
            .set_map_point_position(point_id, crate::WorldPoint3::new(1.0, 2.0, 3.0))
            .expect("mutate correction branch");
        current
            .set_map_point_position(point_id, crate::WorldPoint3::new(-1.0, -2.0, 4.0))
            .expect("mutate current branch");

        assert_eq!(correction_source.generation(), current.generation());
        assert_eq!(
            correction_source.snapshot().instance_id(),
            current.snapshot().instance_id()
        );
        assert_ne!(correction_source.snapshot(), current.snapshot());

        let event = KeyframeEvent::try_new(
            BackendRequestId(NonZeroU64::new(2).expect("non-zero")),
            second_keyframe,
            BackendWindow::try_new(vec![first_keyframe, second_keyframe]).expect("window"),
            correction_source.clone(),
        )
        .expect("backend event");
        assert_eq!(event.source_snapshot, correction_source.snapshot());
        let mut optimized = correction_source.clone();
        optimized
            .set_map_point_position(point_id, crate::WorldPoint3::new(5.0, 6.0, 7.0))
            .expect("optimize correction branch");
        let correction = CorrectionEvent::from_optimized_map(
            &event,
            &optimized,
            BaResult::Converged {
                iterations: 1,
                final_cost: 0.0,
            },
        )
        .expect("build correction");
        let snapshot_before = current.snapshot();
        let position_before = current.point(point_id).expect("current point").position();
        let first_pose_before = current
            .keyframe(first_keyframe)
            .expect("first keyframe")
            .pose()
            .translation();
        let second_pose_before = current
            .keyframe(second_keyframe)
            .expect("second keyframe")
            .pose()
            .translation();

        assert!(matches!(
            apply_correction_event(&mut current, &correction),
            Err(ApplyCorrectionError::StaleSnapshot { .. })
        ));
        assert_eq!(current.snapshot(), snapshot_before);
        assert_eq!(
            current.point(point_id).expect("current point").position(),
            position_before
        );
        assert_eq!(
            current
                .keyframe(first_keyframe)
                .expect("first keyframe")
                .pose()
                .translation(),
            first_pose_before
        );
        assert_eq!(
            current
                .keyframe(second_keyframe)
                .expect("second keyframe")
                .pose()
                .translation(),
            second_pose_before
        );
    }

    #[test]
    fn correction_apply_rejects_different_map_instance() {
        let (map, keyframe_id, point_id) = make_map_with_single_point();
        let correction = CorrectionEvent {
            request_id: BackendRequestId(NonZeroU64::new(2).expect("non-zero")),
            source_snapshot: map.snapshot(),
            trigger_keyframe: keyframe_id,
            correction: BackendCorrection {
                corrected_poses: vec![(keyframe_id, crate::WorldToCamera::identity())],
                corrected_landmarks: vec![(point_id, crate::WorldPoint3::new(0.0, 0.0, 1.0))],
                result: BaResult::Converged {
                    iterations: 1,
                    final_cost: 0.0,
                },
            },
        };
        let mut other_map = SlamMap::new();

        assert!(matches!(
            apply_correction_event(&mut other_map, &correction),
            Err(ApplyCorrectionError::StaleSnapshot { .. })
        ));
    }

    #[test]
    fn correction_apply_updates_pose_and_landmark_atomically() {
        let (mut map, keyframe_id, point_id) = make_map_with_single_point();
        let corrected_pose = crate::WorldToCamera::from_legacy_pose(
            crate::test_helpers::axis_angle_pose([0.2, -0.1, 0.05], [0.01, 0.0, 0.0]),
        );
        let corrected_point: crate::WorldPoint3 = Point3 {
            x: 0.4,
            y: -0.3,
            z: 2.1,
        };
        let correction = CorrectionEvent {
            request_id: BackendRequestId(NonZeroU64::new(3).expect("non-zero")),
            source_snapshot: map.snapshot(),
            trigger_keyframe: keyframe_id,
            correction: BackendCorrection {
                corrected_poses: vec![(keyframe_id, corrected_pose)],
                corrected_landmarks: vec![(point_id, corrected_point)],
                result: BaResult::Converged {
                    iterations: 2,
                    final_cost: 0.1,
                },
            },
        };

        apply_correction_event(&mut map, &correction).expect("correction apply");
        assert_map_invariants(&map).expect("post-correction invariants");

        let stored_pose = map.keyframe(keyframe_id).expect("keyframe").pose();
        for i in 0..3 {
            let a = stored_pose.translation()[i];
            let b = corrected_pose.translation()[i];
            assert!(
                (a - b).abs() < 1e-4,
                "translation mismatch at {i}: {a} vs {b}"
            );
        }
        let stored_rot = stored_pose.rotation();
        let corrected_rot = corrected_pose.rotation();
        for row in 0..3 {
            for col in 0..3 {
                let a = stored_rot[row][col];
                let b = corrected_rot[row][col];
                assert!(
                    (a - b).abs() < 2e-3,
                    "rotation mismatch at ({row},{col}): {a} vs {b}"
                );
            }
        }

        let stored_point = map.point(point_id).expect("map point").position();
        assert!((stored_point.x - corrected_point.x).abs() < 1e-6);
        assert!((stored_point.y - corrected_point.y).abs() < 1e-6);
        assert!((stored_point.z - corrected_point.z).abs() < 1e-6);
    }

    #[test]
    fn correction_apply_preserves_small_absolute_pose_after_large_source_pose() {
        let (mut map, keyframe_id, _) = make_map_with_single_point();
        let large_source_pose = crate::WorldToCamera::from_legacy_pose(
            Pose::try_from_rt(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [1.0e20, -1.0e20, 1.0e20],
            )
            .expect("large finite source pose"),
        );
        map.set_keyframe_pose(keyframe_id, large_source_pose)
            .expect("store source pose");
        let corrected_pose = crate::WorldToCamera::from_legacy_pose(
            Pose::try_from_rt(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [1.0, -2.0, 3.0],
            )
            .expect("small finite corrected pose"),
        );
        let correction = CorrectionEvent {
            request_id: BackendRequestId(NonZeroU64::new(4).expect("non-zero")),
            source_snapshot: map.snapshot(),
            trigger_keyframe: keyframe_id,
            correction: BackendCorrection {
                corrected_poses: vec![(keyframe_id, corrected_pose)],
                corrected_landmarks: Vec::new(),
                result: BaResult::Converged {
                    iterations: 1,
                    final_cost: 0.0,
                },
            },
        };

        apply_correction_event(&mut map, &correction).expect("apply exact corrected pose");
        assert_eq!(
            map.keyframe(keyframe_id)
                .expect("keyframe")
                .pose()
                .translation(),
            [1.0, -2.0, 3.0]
        );
    }

    #[test]
    fn correction_apply_rejects_all_updates_when_a_landmark_is_missing() {
        let (mut map, keyframe_id, point_id) = make_map_with_single_point();
        map.remove_map_point(point_id).expect("remove map point");
        let before_snapshot = map.snapshot();
        let before_pose = map.keyframe(keyframe_id).expect("keyframe").pose();
        let corrected_pose = crate::WorldToCamera::from_legacy_pose(
            Pose::try_from_rt(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [0.1, 0.0, 0.0],
            )
            .expect("finite corrected pose"),
        );
        let correction = CorrectionEvent {
            request_id: BackendRequestId(NonZeroU64::new(4).expect("non-zero")),
            source_snapshot: before_snapshot,
            trigger_keyframe: keyframe_id,
            correction: BackendCorrection {
                corrected_poses: vec![(keyframe_id, corrected_pose)],
                corrected_landmarks: vec![(point_id, crate::WorldPoint3::new(1.0, 2.0, 3.0))],
                result: BaResult::Converged {
                    iterations: 1,
                    final_cost: 0.0,
                },
            },
        };

        assert!(matches!(
            apply_correction_event(&mut map, &correction),
            Err(ApplyCorrectionError::MissingMapPoint { point_id: missing })
                if missing == point_id
        ));
        assert_eq!(map.snapshot(), before_snapshot);
        assert_eq!(
            map.keyframe(keyframe_id)
                .expect("keyframe")
                .pose()
                .rotation(),
            before_pose.rotation()
        );
        assert_eq!(
            map.keyframe(keyframe_id)
                .expect("keyframe")
                .pose()
                .translation(),
            before_pose.translation()
        );
        assert!(map.point(point_id).is_none());
    }

    #[test]
    fn correction_build_preserves_large_finite_landmark_exactly() {
        let (mut map, kf_a, kf_b) = make_map_with_two_keyframes_one_shared_point();
        let keypoint = map.keyframe_keypoint(kf_a, 0).expect("keypoint");
        let point_id = map
            .map_point_for_keypoint(keypoint)
            .expect("map lookup")
            .expect("shared point");
        map.set_map_point_position(point_id, crate::WorldPoint3::new(f32::MAX, 0.0, 1.0))
            .expect("maximum finite point");

        let window = BackendWindow::try_new(vec![kf_a, kf_b]).expect("window");
        let event = KeyframeEvent::try_new(
            BackendRequestId(NonZeroU64::new(5).expect("non-zero")),
            kf_b,
            window,
            map.clone(),
        )
        .expect("event");
        let mut optimized_map = map;
        optimized_map
            .set_map_point_position(point_id, crate::WorldPoint3::new(-f32::MAX, 0.0, 1.0))
            .expect("negative maximum finite point");

        let correction = CorrectionEvent::from_optimized_map(
            &event,
            &optimized_map,
            BaResult::Converged {
                iterations: 1,
                final_cost: 0.0,
            },
        )
        .expect("finite optimized state is directly representable");
        let mut source_map = event.map_snapshot.clone();
        apply_correction_event(&mut source_map, &correction).expect("apply exact correction");
        assert_eq!(
            source_map.point(point_id).expect("map point").position(),
            crate::WorldPoint3::new(-f32::MAX, 0.0, 1.0)
        );
    }

    #[test]
    fn backend_roundtrip_carries_typed_ba_result() {
        let (map, kf_a, kf_b) = make_map_with_two_keyframes_one_shared_point();
        let backend_cfg = BackendConfig::new(1).expect("backend config");
        let intrinsics =
            crate::test_helpers::make_pinhole_intrinsics(320, 240, 200.0, 200.0, 160.0, 120.0)
                .expect("intrinsics");
        let ba_cfg = LocalBaConfig::new(5, 5, 4, 1.0, crate::local_ba::LmConfig::default())
            .expect("ba config");
        let worker =
            BackendWorker::spawn(backend_cfg, intrinsics, ba_cfg).expect("spawn backend worker");

        let window = BackendWindow::try_new(vec![kf_a, kf_b]).expect("window");
        let source_snapshot = map.snapshot();
        let event =
            KeyframeEvent::try_new(BackendRequestId::FIRST, kf_b, window, map).expect("event");
        worker.try_submit(event).expect("submit");

        let mut response = None;
        for _ in 0..50 {
            match worker.try_recv() {
                Ok(Some(msg)) => {
                    response = Some(msg);
                    break;
                }
                Ok(None) => {}
                Err(()) => break,
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        let Some(response) = response else {
            panic!("backend did not produce a response in time");
        };
        match response {
            BackendResponse::Correction(correction) => {
                assert_eq!(correction.source_snapshot, source_snapshot);
                assert!(matches!(
                    correction.correction.result,
                    BaResult::Degenerate { .. }
                        | BaResult::Converged { .. }
                        | BaResult::MaxIterations { .. }
                ));
            }
            BackendResponse::Failure { error, .. } => {
                panic!("unexpected backend failure: {error}");
            }
            BackendResponse::WorkerPanic { .. } => {
                panic!("unexpected worker panic");
            }
        }
    }

    #[test]
    fn backend_supervisor_propagates_initial_spawn_failure() {
        let intrinsics =
            crate::test_helpers::make_pinhole_intrinsics(320, 240, 200.0, 200.0, 160.0, 120.0)
                .expect("intrinsics");
        let error = BackendSupervisor::spawn_initial_with(
            BackendConfig::new(1).expect("backend config"),
            intrinsics,
            LocalBaConfig::new(5, 5, 4, 1.0, crate::local_ba::LmConfig::default())
                .expect("ba config"),
            3,
            |_, _, _| -> Result<BackendWorker, std::io::Error> {
                Err(std::io::Error::other("forced initial spawn failure"))
            },
        )
        .expect_err("initial spawn failure must abort tracker initialization");

        assert!(matches!(error, TrackerInitError::BackendWorkerSpawn(_)));
        assert!(
            std::error::Error::source(&error)
                .and_then(|source| source.downcast_ref::<std::io::Error>())
                .is_some()
        );
    }

    #[test]
    fn backend_supervisor_respawns_after_worker_panic() {
        let intrinsics =
            crate::test_helpers::make_pinhole_intrinsics(320, 240, 200.0, 200.0, 160.0, 120.0)
                .expect("intrinsics");
        let mut supervisor = BackendSupervisor::with_max_respawns(
            BackendConfig::new(1).expect("backend config"),
            intrinsics,
            LocalBaConfig::new(5, 5, 4, 1.0, crate::local_ba::LmConfig::default())
                .expect("ba config"),
            3,
        );

        let (map, kf_a, kf_b) = make_map_with_two_keyframes_one_shared_point();
        let request_id = supervisor.next_request_id().expect("first request id");
        assert_eq!(request_id.as_u64(), 1);
        let event = make_forced_panic_event(request_id, map, kf_a, kf_b);
        supervisor.submit(event).expect("submit");

        let mut saw_panic = false;
        for _ in 0..100 {
            if matches!(
                supervisor.try_recv(),
                Some(BackendResponse::WorkerPanic { .. })
            ) {
                saw_panic = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(saw_panic, "expected worker panic response");

        for _ in 0..100 {
            supervisor.check_health();
            if supervisor.respawn_count() > 0 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }

        assert_eq!(supervisor.respawn_count(), 1);
        assert!(supervisor.has_worker());
        assert_eq!(
            supervisor
                .next_request_id()
                .expect("request id after respawn")
                .as_u64(),
            2,
            "a replacement worker must not reset the supervisor-owned sequence"
        );
    }

    #[test]
    fn backend_supervisor_enforces_max_respawns() {
        let intrinsics =
            crate::test_helpers::make_pinhole_intrinsics(320, 240, 200.0, 200.0, 160.0, 120.0)
                .expect("intrinsics");
        let mut supervisor = BackendSupervisor::with_max_respawns(
            BackendConfig::new(1).expect("backend config"),
            intrinsics,
            LocalBaConfig::new(5, 5, 4, 1.0, crate::local_ba::LmConfig::default())
                .expect("ba config"),
            1,
        );
        let (map1, kf_a1, kf_b1) = make_map_with_two_keyframes_one_shared_point();
        let panic1 = make_forced_panic_event(
            supervisor.next_request_id().expect("first request id"),
            map1,
            kf_a1,
            kf_b1,
        );
        supervisor.submit(panic1).expect("submit panic1");
        for _ in 0..100 {
            if matches!(
                supervisor.try_recv(),
                Some(BackendResponse::WorkerPanic { .. })
            ) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        for _ in 0..100 {
            supervisor.check_health();
            if supervisor.respawn_count() == 1 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(supervisor.respawn_count(), 1);
        assert!(supervisor.has_worker());

        let (map2, kf_a2, kf_b2) = make_map_with_two_keyframes_one_shared_point();
        let panic2 = make_forced_panic_event(
            supervisor.next_request_id().expect("second request id"),
            map2,
            kf_a2,
            kf_b2,
        );
        supervisor.submit(panic2).expect("submit panic2");
        for _ in 0..100 {
            if matches!(
                supervisor.try_recv(),
                Some(BackendResponse::WorkerPanic { .. })
            ) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        for _ in 0..100 {
            supervisor.check_health();
            if !supervisor.has_worker() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }

        assert_eq!(supervisor.respawn_count(), 1);
        assert!(!supervisor.has_worker());
    }

    #[test]
    fn backend_supervisor_shutdown_does_not_respawn() {
        let intrinsics =
            crate::test_helpers::make_pinhole_intrinsics(320, 240, 200.0, 200.0, 160.0, 120.0)
                .expect("intrinsics");
        let mut supervisor = BackendSupervisor::with_max_respawns(
            BackendConfig::new(1).expect("backend config"),
            intrinsics,
            LocalBaConfig::new(5, 5, 4, 1.0, crate::local_ba::LmConfig::default())
                .expect("ba config"),
            3,
        );
        supervisor.shutdown();
        supervisor.check_health();
        assert_eq!(supervisor.respawn_count(), 0);
        assert!(!supervisor.has_worker());
    }

    #[test]
    fn backend_supervisor_continues_after_panic_respawn() {
        let intrinsics =
            crate::test_helpers::make_pinhole_intrinsics(320, 240, 200.0, 200.0, 160.0, 120.0)
                .expect("intrinsics");
        let mut supervisor = BackendSupervisor::with_max_respawns(
            BackendConfig::new(1).expect("backend config"),
            intrinsics,
            LocalBaConfig::new(5, 5, 4, 1.0, crate::local_ba::LmConfig::default())
                .expect("ba config"),
            2,
        );

        let (map_panic, kf_a, kf_b) = make_map_with_two_keyframes_one_shared_point();
        let first_request = supervisor.next_request_id().expect("first request id");
        assert_eq!(first_request.as_u64(), 1);
        let panic_event = make_forced_panic_event(first_request, map_panic, kf_a, kf_b);
        supervisor.submit(panic_event).expect("submit panic");

        let mut saw_panic = false;
        for _ in 0..100 {
            if matches!(
                supervisor.try_recv(),
                Some(BackendResponse::WorkerPanic { .. })
            ) {
                saw_panic = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(saw_panic, "expected worker panic");

        for _ in 0..100 {
            supervisor.check_health();
            if supervisor.respawn_count() >= 1 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(supervisor.has_worker(), "supervisor should respawn worker");

        let (map_ok, kf_a2, kf_b2) = make_map_with_two_keyframes_one_shared_point();
        let window = BackendWindow::try_new(vec![kf_a2, kf_b2]).expect("window");
        let second_request = supervisor.next_request_id().expect("second request id");
        assert_eq!(second_request.as_u64(), 2);
        let ok_event =
            KeyframeEvent::try_new(second_request, kf_b2, window, map_ok).expect("event");
        supervisor
            .submit(ok_event)
            .expect("submit event after respawn");

        let mut got_non_panic = false;
        for _ in 0..100 {
            match supervisor.try_recv() {
                Some(BackendResponse::Correction(_)) | Some(BackendResponse::Failure { .. }) => {
                    got_non_panic = true;
                    break;
                }
                Some(BackendResponse::WorkerPanic { .. }) => {
                    panic!("worker panicked again on normal event");
                }
                None => {}
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(got_non_panic, "expected non-panic response after respawn");
    }

    #[test]
    fn descriptor_worker_processes_requests() {
        let descriptor = make_global_descriptor_basis(42);
        let calls = Arc::new(Mutex::new(0_usize));
        let worker = DescriptorWorker::spawn_with_extractor(
            GlobalDescriptorConfig::new(2).expect("config"),
            Box::new(StubDescriptorExtractor {
                descriptor: descriptor.clone(),
                calls: Arc::clone(&calls),
            }),
        )
        .expect("spawn descriptor worker");

        let frame = Frame::new(
            SensorId::StereoLeft,
            FrameId::new(77),
            Timestamp::from_nanos(77),
            16,
            12,
            vec![128_u8; 16 * 12],
        )
        .expect("frame");
        let source_snapshot = SlamMap::new().snapshot();
        worker
            .submit(DescriptorRequest {
                keyframe_id: KeyframeId::default(),
                source_snapshot,
                frame,
            })
            .expect("submit descriptor request");

        let mut response = None;
        for _ in 0..50 {
            match worker.try_recv() {
                Ok(Some(DescriptorWorkerResponse::Descriptor(value))) => {
                    response = Some(value);
                    break;
                }
                Ok(Some(other)) => panic!("unexpected descriptor worker response: {other:?}"),
                Ok(None) => {}
                Err(()) => panic!("descriptor worker disconnected"),
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        let response = response.expect("descriptor response");
        assert_eq!(response.source_snapshot, source_snapshot);
        assert_eq!(response.descriptor, descriptor);
        assert_eq!(*calls.lock().expect("calls lock"), 1);
    }

    #[test]
    fn descriptor_response_requires_exact_map_snapshot() {
        let (mut map, keyframe_id, _) = make_map_with_single_point();
        let bootstrap = make_global_descriptor_basis(1);
        let learned = make_global_descriptor_basis(2);
        let mut loop_db = KeyframeDatabase::new(0);
        loop_db
            .insert_with_source(
                keyframe_id,
                bootstrap.clone(),
                crate::loop_closure::DescriptorSource::Bootstrap,
            )
            .expect("unique keyframe");

        assert_eq!(
            apply_descriptor_response(
                &map,
                &mut loop_db,
                DescriptorResponse {
                    keyframe_id,
                    source_snapshot: map.snapshot(),
                    descriptor: learned.clone(),
                },
            )
            .expect("current response"),
            DescriptorApplyDisposition::Applied
        );
        assert_eq!(
            loop_db.descriptor_source(keyframe_id),
            Some(crate::loop_closure::DescriptorSource::Learned)
        );

        loop_db
            .replace_descriptor(
                keyframe_id,
                bootstrap,
                crate::loop_closure::DescriptorSource::Bootstrap,
            )
            .expect("registered keyframe");
        let stale_snapshot = map.snapshot();
        let pose = map.keyframe(keyframe_id).expect("keyframe").pose();
        map.set_keyframe_pose(keyframe_id, pose)
            .expect("advance map generation");
        assert_eq!(
            apply_descriptor_response(
                &map,
                &mut loop_db,
                DescriptorResponse {
                    keyframe_id,
                    source_snapshot: stale_snapshot,
                    descriptor: learned,
                },
            )
            .expect("stale response is nonfatal"),
            DescriptorApplyDisposition::Stale
        );
        assert_eq!(
            loop_db.descriptor_source(keyframe_id),
            Some(crate::loop_closure::DescriptorSource::Bootstrap)
        );
    }

    #[test]
    fn current_descriptor_response_propagates_missing_database_entry() {
        let (map, keyframe_id, _) = make_map_with_single_point();
        let mut loop_db = KeyframeDatabase::new(0);

        let error = apply_descriptor_response(
            &map,
            &mut loop_db,
            DescriptorResponse {
                keyframe_id,
                source_snapshot: map.snapshot(),
                descriptor: make_global_descriptor_basis(3),
            },
        )
        .expect_err("current response requires a registered keyframe descriptor");

        assert!(matches!(
            error,
            TrackerError::KeyframeDatabase(KeyframeDatabaseError::KeyframeNotFound {
                keyframe_id: missing
            }) if missing == keyframe_id
        ));
        assert!(loop_db.is_empty());
    }

    #[test]
    fn descriptor_model_path_rejects_explicit_empty_override() {
        let err = DescriptorWorker::model_path_from_override(Some(OsString::new()))
            .expect_err("an explicitly empty model path must be rejected");
        assert!(matches!(
            err,
            TrackerInitError::EmptyDescriptorModelPath {
                variable: EIGENPLACES_MODEL_ENV
            }
        ));

        let override_path = OsString::from("custom-eigenplaces.onnx");
        assert_eq!(
            DescriptorWorker::model_path_from_override(Some(override_path))
                .expect("nonempty model path"),
            PathBuf::from("custom-eigenplaces.onnx")
        );
    }

    #[test]
    fn descriptor_supervisor_propagates_initial_model_error() {
        let config = GlobalDescriptorConfig::new(2).expect("config");
        let factory: DescriptorExtractorFactory = Arc::new(|| {
            Err(InferenceError::InvariantViolation {
                context: "forced descriptor initialization failure",
            })
        });

        let err = match DescriptorSupervisor::try_with_factory_and_max_respawns(config, factory, 2)
        {
            Ok(_) => panic!("initial descriptor failure must abort construction"),
            Err(err) => err,
        };

        assert!(matches!(
            err,
            TrackerInitError::DescriptorModelLoad(InferenceError::InvariantViolation {
                context: "forced descriptor initialization failure"
            })
        ));
    }

    #[test]
    fn descriptor_supervisor_recovers_after_worker_panic() {
        let config = GlobalDescriptorConfig::new(2).expect("config");
        let spawn_count = Arc::new(AtomicUsize::new(0));
        let calls = Arc::new(Mutex::new(0_usize));
        let descriptor = make_global_descriptor_basis(17);

        let factory: DescriptorExtractorFactory = {
            let spawn_count = Arc::clone(&spawn_count);
            let calls = Arc::clone(&calls);
            let descriptor = descriptor.clone();
            Arc::new(move || {
                let spawn_idx = spawn_count.fetch_add(1, AtomicOrdering::SeqCst);
                if spawn_idx == 0 {
                    Ok(Box::new(PanicDescriptorExtractor) as Box<dyn PlaceDescriptorExtractor>)
                } else {
                    Ok(Box::new(StubDescriptorExtractor {
                        descriptor: descriptor.clone(),
                        calls: Arc::clone(&calls),
                    }) as Box<dyn PlaceDescriptorExtractor>)
                }
            })
        };

        let mut supervisor =
            DescriptorSupervisor::try_with_factory_and_max_respawns(config, factory, 2)
                .expect("initial descriptor worker");
        let frame = Frame::new(
            SensorId::StereoLeft,
            FrameId::new(78),
            Timestamp::from_nanos(78),
            16,
            12,
            vec![128_u8; 16 * 12],
        )
        .expect("frame");

        supervisor
            .submit(DescriptorRequest {
                keyframe_id: KeyframeId::default(),
                source_snapshot: SlamMap::new().snapshot(),
                frame: frame.clone(),
            })
            .expect("submit panic request");

        let mut saw_panic = false;
        for _ in 0..50 {
            match supervisor.try_recv() {
                Some(DescriptorWorkerResponse::WorkerPanic { .. }) => {
                    saw_panic = true;
                    break;
                }
                Some(_) => {}
                None => {}
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(saw_panic, "expected worker panic event");
        assert_eq!(supervisor.respawn_count(), 1);
        assert!(supervisor.has_worker());

        supervisor
            .submit(DescriptorRequest {
                keyframe_id: KeyframeId::default(),
                source_snapshot: SlamMap::new().snapshot(),
                frame,
            })
            .expect("submit recovered request");

        let mut recovered = None;
        for _ in 0..50 {
            match supervisor.try_recv() {
                Some(DescriptorWorkerResponse::Descriptor(value)) => {
                    recovered = Some(value);
                    break;
                }
                Some(DescriptorWorkerResponse::Failure { error, .. }) => {
                    panic!("unexpected descriptor failure after respawn: {error}");
                }
                Some(DescriptorWorkerResponse::WorkerPanic { .. }) => {
                    panic!("unexpected second panic");
                }
                None => {}
            }
            std::thread::sleep(Duration::from_millis(5));
        }

        let recovered = recovered.expect("descriptor response after respawn");
        assert_eq!(recovered.descriptor, descriptor);
        assert_eq!(*calls.lock().expect("calls lock"), 1);
    }

    #[test]
    fn descriptor_supervisor_uses_all_configured_respawn_attempts() {
        let config = GlobalDescriptorConfig::new(2).expect("config");
        let spawn_count = Arc::new(AtomicUsize::new(0));
        let descriptor = make_global_descriptor_basis(23);
        let factory: DescriptorExtractorFactory = {
            let spawn_count = Arc::clone(&spawn_count);
            let descriptor = descriptor.clone();
            Arc::new(move || {
                let spawn_idx = spawn_count.fetch_add(1, AtomicOrdering::SeqCst);
                if spawn_idx == 0 {
                    Ok(Box::new(PanicDescriptorExtractor) as Box<dyn PlaceDescriptorExtractor>)
                } else if spawn_idx == 1 {
                    Err(InferenceError::InvariantViolation {
                        context: "forced descriptor respawn failure",
                    })
                } else {
                    Ok(Box::new(StubDescriptorExtractor {
                        descriptor: descriptor.clone(),
                        calls: Arc::new(Mutex::new(0)),
                    }) as Box<dyn PlaceDescriptorExtractor>)
                }
            })
        };
        let mut supervisor =
            DescriptorSupervisor::try_with_factory_and_max_respawns(config, factory, 2)
                .expect("initial descriptor worker");
        let frame = Frame::new(
            SensorId::StereoLeft,
            FrameId::new(79),
            Timestamp::from_nanos(79),
            16,
            12,
            vec![128_u8; 16 * 12],
        )
        .expect("frame");

        supervisor
            .submit(DescriptorRequest {
                keyframe_id: KeyframeId::default(),
                source_snapshot: SlamMap::new().snapshot(),
                frame: frame.clone(),
            })
            .expect("submit panic request");
        let mut saw_panic = false;
        for _ in 0..50 {
            if matches!(
                supervisor.try_recv(),
                Some(DescriptorWorkerResponse::WorkerPanic { .. })
            ) {
                saw_panic = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(saw_panic, "expected initial worker panic");
        assert_eq!(supervisor.respawn_count(), 1);
        assert!(!supervisor.has_worker());
        assert!(!supervisor.spawn_exhausted);

        supervisor
            .submit(DescriptorRequest {
                keyframe_id: KeyframeId::default(),
                source_snapshot: SlamMap::new().snapshot(),
                frame,
            })
            .expect("second configured respawn should recover");
        assert_eq!(supervisor.respawn_count(), 2);
        assert_eq!(spawn_count.load(AtomicOrdering::SeqCst), 3);
        assert!(supervisor.has_worker());
        assert!(!supervisor.spawn_exhausted);
    }

    fn make_loop_closure_apply_fixture() -> (
        SlamMap,
        EssentialGraph,
        crate::loop_closure::VerifiedLoop,
        KeyframeId,
        Vec<(MapPointId, Point3)>,
    ) {
        let mut map = SlamMap::new();
        let image_size = crate::map::ImageSize::try_new(640, 480).expect("image size");
        let keypoints = vec![
            Keypoint { x: 120.0, y: 100.0 },
            Keypoint { x: 220.0, y: 110.0 },
            Keypoint { x: 320.0, y: 120.0 },
            Keypoint { x: 420.0, y: 130.0 },
        ];
        let kf0 = map
            .add_keyframe(
                FrameId::new(100),
                Timestamp::from_nanos(100),
                crate::WorldToCamera::identity(),
                image_size,
                keypoints.clone(),
            )
            .expect("kf0");
        let kf1 = map
            .add_keyframe(
                FrameId::new(101),
                Timestamp::from_nanos(101),
                crate::WorldToCamera::from_legacy_pose(Pose::from_rt(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    [1.0, 0.0, 0.0],
                )),
                image_size,
                keypoints.clone(),
            )
            .expect("kf1");
        let kf2 = map
            .add_keyframe(
                FrameId::new(102),
                Timestamp::from_nanos(102),
                crate::WorldToCamera::from_legacy_pose(Pose::from_rt(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    [2.4, 0.2, 0.0],
                )),
                image_size,
                keypoints.clone(),
            )
            .expect("kf2");

        let world_points = [
            Point3 {
                x: -0.4,
                y: -0.2,
                z: 3.0,
            },
            Point3 {
                x: -0.1,
                y: -0.1,
                z: 3.2,
            },
            Point3 {
                x: 0.2,
                y: 0.0,
                z: 3.4,
            },
            Point3 {
                x: 0.5,
                y: 0.1,
                z: 3.6,
            },
        ];

        for (idx, &world) in world_points.iter().enumerate() {
            let kp0 = map.keyframe_keypoint(kf0, idx).expect("kp0");
            let point_id = map
                .add_map_point(world, CompactDescriptor([128; 256]), kp0)
                .expect("point");
            let kp1 = map.keyframe_keypoint(kf1, idx).expect("kp1");
            map.add_observation(point_id, kp1).expect("obs1");
            let kp2 = map.keyframe_keypoint(kf2, idx).expect("kp2");
            map.add_observation(point_id, kp2).expect("obs2");
        }

        let before_points: Vec<(MapPointId, Point3)> = map
            .points()
            .map(|(id, point)| (id, point.position()))
            .collect();

        let mut essential_graph = EssentialGraph::new(1);
        essential_graph
            .add_keyframe(kf0, map.covisibility().neighbors(kf0), &map)
            .expect("register kf0");
        essential_graph
            .add_keyframe(kf1, map.covisibility().neighbors(kf1), &map)
            .expect("register kf1");
        essential_graph
            .add_keyframe(kf2, map.covisibility().neighbors(kf2), &map)
            .expect("register kf2");

        let verified = crate::loop_closure::VerifiedLoop::from_parts(
            kf2,
            kf0,
            map.snapshot(),
            Pose::from_rt(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [2.0, 0.0, 0.0],
            ),
            60,
        );

        (map, essential_graph, verified, kf2, before_points)
    }

    #[test]
    fn loop_closure_correction_reduces_synthetic_drift_ring() {
        let (mut map, mut essential_graph, verified, query_kf, _) =
            make_loop_closure_apply_fixture();
        let optimizer = PoseGraphOptimizer::new(PoseGraphConfig::default());

        let before = map
            .keyframe(query_kf)
            .expect("query pose")
            .pose()
            .translation();
        let before_error =
            ((before[0] - 2.0).powi(2) + (before[1]).powi(2) + (before[2]).powi(2)).sqrt();

        apply_loop_closure_correction(&mut map, &mut essential_graph, &optimizer, &verified)
            .expect("apply loop closure");

        let after = map
            .keyframe(query_kf)
            .expect("corrected query")
            .pose()
            .translation();
        let after_error =
            ((after[0] - 2.0).powi(2) + (after[1]).powi(2) + (after[2]).powi(2)).sqrt();
        assert!(
            after_error < before_error,
            "loop closure should reduce drift: before={before_error}, after={after_error}"
        );
    }

    #[test]
    fn loop_closure_reprojects_map_points_with_pose_correction() {
        let (mut map, mut essential_graph, verified, _query_kf, before_points) =
            make_loop_closure_apply_fixture();
        let optimizer = PoseGraphOptimizer::new(PoseGraphConfig::default());

        apply_loop_closure_correction(&mut map, &mut essential_graph, &optimizer, &verified)
            .expect("apply loop closure");

        let moved_points = before_points
            .iter()
            .filter(|(point_id, before)| {
                let after = map.point(*point_id).expect("point").position();
                let dx = after.x - before.x;
                let dy = after.y - before.y;
                let dz = after.z - before.z;
                (dx * dx + dy * dy + dz * dz).sqrt() > 1e-5
            })
            .count();
        assert!(
            moved_points > 0,
            "expected map points to move after loop correction"
        );
    }

    #[test]
    fn loop_closure_adds_loop_edge_to_essential_graph() {
        let (mut map, mut essential_graph, verified, _query_kf, _before_points) =
            make_loop_closure_apply_fixture();
        let optimizer = PoseGraphOptimizer::new(PoseGraphConfig::default());

        assert_eq!(essential_graph.snapshot().loop_edges.len(), 0);
        apply_loop_closure_correction(&mut map, &mut essential_graph, &optimizer, &verified)
            .expect("apply loop closure");
        let snapshot = essential_graph.snapshot();
        assert_eq!(snapshot.loop_edges.len(), 1);
        assert_eq!(snapshot.loop_edges[0].kind(), EssentialEdgeKind::Loop);
    }

    #[test]
    fn loop_closure_rejects_unregistered_endpoint_before_optimizer_input() {
        let (mut map, mut essential_graph, verified, query_kf, _before_points) =
            make_loop_closure_apply_fixture();
        essential_graph
            .remove_keyframe(query_kf, &map)
            .expect("inject graph/map topology mismatch");
        let map_before = map.snapshot();
        let graph_before = essential_graph.snapshot();
        let optimizer = PoseGraphOptimizer::new(PoseGraphConfig::default());

        let error =
            apply_loop_closure_correction(&mut map, &mut essential_graph, &optimizer, &verified)
                .expect_err("loop endpoint must already be registered");

        assert!(matches!(
            error,
            TrackerError::EssentialGraph(EssentialGraphError::KeyframeNotRegistered {
                keyframe_id
            }) if keyframe_id == query_kf
        ));
        assert_eq!(map.snapshot(), map_before);
        assert_eq!(essential_graph.snapshot().order, graph_before.order);
        assert!(essential_graph.snapshot().loop_edges.is_empty());
    }

    #[test]
    fn loop_correction_threshold_uses_pose_delta_not_absolute_pose() {
        let current = Pose::from_rt(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [1_000.0, -500.0, 25.0],
        );
        let estimate = Pose::from_rt(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [1_000.1, -500.0, 25.0],
        );

        let correction = loop_pose_correction(current, estimate).expect("finite loop correction");

        assert!((loop_translation_norm(correction) - 0.1).abs() < 1e-4);
        assert!(loop_rotation_angle_deg(correction) < 1e-4);
    }

    #[test]
    fn loop_translation_norm_avoids_intermediate_square_overflow() {
        let component = f32::MAX / 2.0;
        let pose = Pose::from_rt(Pose::identity().rotation(), [component; 3]);
        let norm = loop_translation_norm(pose);
        let expected = (f64::from(component) * 3.0_f64.sqrt()) as f32;
        assert!(norm.is_finite());
        assert!((norm - expected).abs() <= expected * 2.0 * f32::EPSILON);
    }

    #[test]
    fn loop_pose_correction_maps_current_camera_point_to_estimated_camera() {
        let current = crate::math::se3_exp_f64([0.8, -0.3, 0.2, 0.25, -0.15, 0.1])
            .try_to_pose32()
            .expect("test pose should fit in f32");
        let estimate = crate::math::se3_exp_f64([-0.4, 0.7, 0.5, -0.2, 0.12, 0.3])
            .try_to_pose32()
            .expect("test pose should fit in f32");
        let correction = loop_pose_correction(current, estimate).expect("finite loop correction");
        let world_point = [1.3, -0.6, 4.2];
        let point_in_current =
            crate::math::transform_point(current.rotation(), current.translation(), world_point);
        let expected_in_estimate =
            crate::math::transform_point(estimate.rotation(), estimate.translation(), world_point);
        let corrected = crate::math::transform_point(
            correction.rotation(),
            correction.translation(),
            point_in_current,
        );

        for axis in 0..3 {
            assert!(
                (corrected[axis] - expected_in_estimate[axis]).abs() < 2e-5,
                "camera-point correction mismatch on axis {axis}"
            );
        }
    }

    #[test]
    fn loop_closure_rolls_back_graph_and_map_on_pcg_failure() {
        let (mut map, mut essential_graph, verified, query_kf, before_points) =
            make_loop_closure_apply_fixture();
        let defaults = PoseGraphConfig::default();
        let optimizer = PoseGraphOptimizer::new(PoseGraphConfig::new_unchecked_for_test(
            defaults.max_iterations(),
            0,
            defaults.pcg_tol(),
            defaults.huber_delta(),
        ));
        let pose_before = map.keyframe(query_kf).expect("query keyframe").pose();
        let generation_before = map.generation();
        let loop_edges_before = essential_graph.snapshot().loop_edges.len();

        let error =
            apply_loop_closure_correction(&mut map, &mut essential_graph, &optimizer, &verified)
                .expect_err("PCG failure must reject loop correction");

        assert!(matches!(
            error,
            TrackerError::PoseGraph(PoseGraphError::PcgDidNotConverge { .. })
        ));
        assert_eq!(
            essential_graph.snapshot().loop_edges.len(),
            loop_edges_before
        );
        assert_eq!(map.generation(), generation_before);
        assert_eq!(
            map.keyframe(query_kf)
                .expect("query keyframe")
                .pose()
                .translation(),
            pose_before.translation()
        );
        for (point_id, position) in before_points {
            let actual = map.point(point_id).expect("map point").position();
            assert_eq!(
                [actual.x, actual.y, actual.z],
                [position.x, position.y, position.z]
            );
        }
    }

    #[test]
    fn loop_closure_rejects_stale_verified_map_snapshot() {
        let (mut map, mut essential_graph, verified, query_kf, _) =
            make_loop_closure_apply_fixture();
        let current_pose = map.keyframe(query_kf).expect("query keyframe").pose();
        map.set_keyframe_pose(query_kf, current_pose)
            .expect("advance map generation");
        let loop_edges_before = essential_graph.snapshot().loop_edges.len();
        let optimizer = PoseGraphOptimizer::new(PoseGraphConfig::default());

        let error =
            apply_loop_closure_correction(&mut map, &mut essential_graph, &optimizer, &verified)
                .expect_err("stale verified loop must be rejected");

        assert!(matches!(
            error,
            TrackerError::LoopMapSnapshotMismatch { .. }
        ));
        assert_eq!(
            essential_graph.snapshot().loop_edges.len(),
            loop_edges_before
        );
    }

    #[test]
    fn remove_keyframe_from_graph_and_db_cleans_all_structures() {
        let (mut map, mut essential_graph, _verified, removed_kf, _before_points) =
            make_loop_closure_apply_fixture();
        let mut loop_db = KeyframeDatabase::new(0);
        for (idx, (keyframe_id, _)) in map.keyframes().enumerate() {
            loop_db
                .insert_with_source(
                    keyframe_id,
                    make_global_descriptor_basis(idx),
                    crate::loop_closure::DescriptorSource::Bootstrap,
                )
                .expect("unique keyframe");
        }

        remove_keyframe_from_graph_and_db(
            &mut map,
            &mut essential_graph,
            Some(&mut loop_db),
            removed_kf,
        )
        .expect("remove keyframe");

        assert!(map.keyframe(removed_kf).is_none());
        assert!(essential_graph.parent_of(removed_kf).is_none());
        assert!(loop_db.descriptor_source(removed_kf).is_none());
        let input = essential_graph
            .pose_graph_input(&map)
            .expect("pose graph input");
        assert!(input.keyframe_ids.iter().all(|&id| id != removed_kf));
    }

    #[test]
    fn remove_keyframe_propagates_missing_database_entry_without_mutation() {
        let (mut map, mut essential_graph, _verified, removed_kf, _before_points) =
            make_loop_closure_apply_fixture();
        let mut loop_db = KeyframeDatabase::new(0);
        let map_before = map.snapshot();
        let graph_before = essential_graph.snapshot();

        let error = remove_keyframe_from_graph_and_db(
            &mut map,
            &mut essential_graph,
            Some(&mut loop_db),
            removed_kf,
        )
        .expect_err("registered map keyframe requires a database entry");

        assert!(matches!(
            error,
            TrackerError::KeyframeDatabase(KeyframeDatabaseError::KeyframeNotFound {
                keyframe_id
            }) if keyframe_id == removed_kf
        ));
        assert_eq!(map.snapshot(), map_before);
        assert_eq!(essential_graph.snapshot().order, graph_before.order);
        assert!(loop_db.is_empty());
    }

    #[test]
    fn remove_keyframe_rejects_stale_map_without_mutating_graph_or_db() {
        let (mut map, mut essential_graph, _verified, removed_kf, _before_points) =
            make_loop_closure_apply_fixture();
        let mut loop_db = KeyframeDatabase::new(0);
        loop_db
            .insert_with_source(
                removed_kf,
                make_global_descriptor_basis(0),
                crate::loop_closure::DescriptorSource::Bootstrap,
            )
            .expect("unique keyframe");
        map.remove_keyframe(removed_kf)
            .expect("remove map entry to inject late failure");
        let parent_before = essential_graph.parent_of(removed_kf);

        let error = remove_keyframe_from_graph_and_db(
            &mut map,
            &mut essential_graph,
            Some(&mut loop_db),
            removed_kf,
        )
        .expect_err("missing map keyframe must fail graph preflight");

        assert!(matches!(
            error,
            TrackerError::EssentialGraph(EssentialGraphError::KeyframeNotFound {
                keyframe_id
            }) if keyframe_id == removed_kf
        ));
        assert_eq!(essential_graph.parent_of(removed_kf), parent_before);
        assert!(loop_db.descriptor_source(removed_kf).is_some());
    }

    #[test]
    fn inference_timeout_and_quarantine_require_pipeline_shutdown() {
        for error in [
            TrackerError::Inference(InferenceError::WatchdogTimeout {
                model: "test",
                timeout_ms: 1,
            }),
            TrackerError::Inference(InferenceError::SessionQuarantined { model: "test" }),
        ] {
            assert!(error.requires_pipeline_shutdown());
        }
        assert!(
            !TrackerError::Inference(InferenceError::WatchdogDeadlineExceeded {
                model: "test",
                timeout_ms: 1,
            })
            .requires_pipeline_shutdown()
        );
        assert!(!TrackerError::KeyframeRejected { landmarks: 0 }.requires_pipeline_shutdown());
    }

    #[test]
    fn tracker_error_preserves_pose_narrowing_error() {
        let source = crate::PoseNarrowingError::TranslationNotRepresentable {
            axis: 2,
            value: f64::MAX,
        };

        let error = TrackerError::from(source);

        assert!(matches!(error, TrackerError::PoseNarrowing(actual) if actual == source));
    }

    #[test]
    fn tracker_error_preserves_pose64_error() {
        let source = crate::Pose64Error::ComposeTranslationNonFinite { axis: 1 };

        let error = TrackerError::from(source);

        assert!(matches!(error, TrackerError::Pose64(actual) if actual == source));
    }

    #[test]
    fn post_ba_diagnostics_accept_partially_unprojectable_inliers_without_subset_metrics() {
        let intrinsics = crate::test_helpers::make_pinhole_intrinsics(1, 1, 1.0, 1.0, 0.0, 0.0)
            .expect("intrinsics");
        let observations: Vec<_> = [1.0, 1.0, 1.0, 1.0, -1.0]
            .into_iter()
            .map(|depth| {
                crate::Observation::try_new(
                    Point3::new(0.0, 0.0, depth),
                    Keypoint { x: 0.0, y: 0.0 },
                )
                .expect("finite observation")
            })
            .collect();

        let diagnostics =
            post_ba_reprojection_diagnostics(Pose::identity(), observations.iter(), intrinsics)
                .expect("individual hidden factors are valid BA output");

        assert_eq!(
            diagnostics,
            PostBaReprojectionDiagnostics::NotAllProjectable
        );

        let complete = post_ba_reprojection_diagnostics(
            Pose::identity(),
            observations[..4].iter(),
            intrinsics,
        )
        .expect("complete projectable set");
        assert_eq!(
            complete,
            PostBaReprojectionDiagnostics::Complete {
                rmse_px: 0.0,
                max_px: 0.0,
            }
        );

        let extreme_intrinsics =
            crate::test_helpers::make_pinhole_intrinsics(1, 1, f32::MAX, 1.0, 0.0, 0.0)
                .expect("extreme intrinsics");
        let extreme_visible = crate::Observation::try_new(
            Point3::new(f32::MAX, 0.0, f32::from_bits(1)),
            Keypoint { x: 0.0, y: 0.0 },
        )
        .expect("visible extreme observation");
        let hidden =
            crate::Observation::try_new(Point3::new(0.0, 0.0, -1.0), Keypoint { x: 0.0, y: 0.0 })
                .expect("hidden observation");
        let partial = [
            extreme_visible,
            extreme_visible,
            extreme_visible,
            extreme_visible,
            hidden,
        ];

        assert_eq!(
            post_ba_reprojection_diagnostics(Pose::identity(), partial.iter(), extreme_intrinsics,)
                .expect("discarded subset metrics must not be narrowed"),
            PostBaReprojectionDiagnostics::NotAllProjectable
        );
    }

    #[test]
    fn loop_apply_error_classification_does_not_alias_pose_failures_to_map_mutation() {
        assert_eq!(
            loop_apply_error_kind(&TrackerError::Pose64(
                crate::Pose64Error::ComposeTranslationNonFinite { axis: 0 }
            )),
            LoopApplyError::PoseConversion
        );
        assert_eq!(
            loop_apply_error_kind(&TrackerError::PoseGraph(
                PoseGraphError::PcgNonFiniteResidual
            )),
            LoopApplyError::PoseOptimization
        );
        assert_eq!(
            loop_apply_error_kind(&TrackerError::InvariantViolation("test")),
            LoopApplyError::InvariantViolation
        );
    }

    #[test]
    fn degradation_level_worst_returns_more_severe_variant() {
        assert_eq!(
            DegradationLevel::worst(
                DegradationLevel::Nominal,
                DegradationLevel::TrackingDegraded
            ),
            DegradationLevel::TrackingDegraded
        );
        assert_eq!(
            DegradationLevel::worst(
                DegradationLevel::TrackingDegraded,
                DegradationLevel::DescriptorDown
            ),
            DegradationLevel::DescriptorDown
        );
        assert_eq!(
            DegradationLevel::worst(
                DegradationLevel::DescriptorDown,
                DegradationLevel::BackendDown
            ),
            DegradationLevel::BackendDown
        );
        assert_eq!(
            DegradationLevel::worst(DegradationLevel::BackendDown, DegradationLevel::Lost),
            DegradationLevel::Lost
        );
    }

    #[test]
    fn system_health_aggregation_combines_tracking_and_backend_state() {
        let stats = BackendStats {
            submitted: 7,
            ..BackendStats::default()
        };
        let nominal =
            SystemHealth::from_components(TrackingHealth::Good, true, true, true, true, stats);
        assert_eq!(nominal.degradation, DegradationLevel::Nominal);
        assert_eq!(nominal.backend, ComponentHealth::Alive);
        assert_eq!(nominal.descriptor, ComponentHealth::Alive);
        assert_eq!(nominal.backend_stats.submitted, 7);

        let degraded =
            SystemHealth::from_components(TrackingHealth::Degraded, true, true, true, true, stats);
        assert_eq!(degraded.degradation, DegradationLevel::TrackingDegraded);

        let descriptor_down =
            SystemHealth::from_components(TrackingHealth::Good, true, true, true, false, stats);
        assert_eq!(
            descriptor_down.degradation,
            DegradationLevel::DescriptorDown
        );
        assert_eq!(descriptor_down.descriptor, ComponentHealth::Down);

        let backend_down =
            SystemHealth::from_components(TrackingHealth::Good, true, false, true, true, stats);
        assert_eq!(backend_down.degradation, DegradationLevel::BackendDown);
        assert_eq!(backend_down.backend, ComponentHealth::Down);

        let lost =
            SystemHealth::from_components(TrackingHealth::Lost, true, false, true, false, stats);
        assert_eq!(lost.degradation, DegradationLevel::Lost);

        let backend_optional =
            SystemHealth::from_components(TrackingHealth::Good, false, true, false, true, stats);
        assert_eq!(backend_optional.degradation, DegradationLevel::Nominal);
        assert_eq!(backend_optional.backend, ComponentHealth::Disabled);
        assert_eq!(backend_optional.descriptor, ComponentHealth::Disabled);
    }

    #[test]
    fn relocalization_initial_session_requires_lost_tracking_and_enabled_config() {
        assert!(SlamTracker::initial_relocalization_session(TrackingHealth::Good, true).is_none());
        assert!(SlamTracker::initial_relocalization_session(TrackingHealth::Lost, false).is_none());

        let session = SlamTracker::initial_relocalization_session(TrackingHealth::Lost, true)
            .expect("lost tracking should create relocalization session");
        assert_eq!(session.attempts, 0);
        assert!(matches!(session.phase, RelocalizationPhase::Searching));
    }

    #[test]
    fn relocalization_candidate_propagates_global_descriptor_error() {
        let current = make_relocalization_detections(Descriptor([0.0; 256]));
        let intrinsics =
            crate::test_helpers::make_pinhole_intrinsics(320, 240, 200.0, 200.0, 160.0, 120.0)
                .expect("intrinsics");

        let error = SlamTracker::relocalization_candidate(
            &current,
            RelocalizationConfig::default(),
            &KeyframeDatabase::new(0),
            &SlamMap::new(),
            intrinsics,
            RansacConfig::default(),
        )
        .expect_err("zero-norm aggregate descriptor must propagate");

        assert!(matches!(
            error,
            TrackerError::GlobalDescriptor(GlobalDescriptorError::ZeroNorm)
        ));
    }

    #[test]
    fn relocalization_candidate_propagates_descriptor_map_error() {
        let mut descriptor = [0.0; 256];
        descriptor[0] = 1.0;
        let descriptor = Descriptor(descriptor);
        let current = make_relocalization_detections(descriptor);
        let global = aggregate_global_descriptor(current.descriptors()).expect("global descriptor");
        let mut loop_db = KeyframeDatabase::new(0);
        loop_db
            .insert(KeyframeId::default(), global)
            .expect("unique keyframe");
        let intrinsics =
            crate::test_helpers::make_pinhole_intrinsics(320, 240, 200.0, 200.0, 160.0, 120.0)
                .expect("intrinsics");

        let error = SlamTracker::relocalization_candidate(
            &current,
            RelocalizationConfig::default(),
            &loop_db,
            &SlamMap::new(),
            intrinsics,
            RansacConfig::default(),
        )
        .expect_err("missing descriptor keyframe must propagate");

        assert!(matches!(
            error,
            TrackerError::Map(crate::map::MapError::KeyframeNotFound(_))
        ));
    }

    #[test]
    fn relocalization_verification_classifies_only_geometric_rejections_as_expected() {
        for expected in [
            LoopVerificationError::TooFewMatches { count: 3 },
            LoopVerificationError::PnpFailed(crate::PnpError::NoSolution),
            LoopVerificationError::InsufficientInliers {
                inliers: 3,
                required: 4,
            },
        ] {
            SlamTracker::classify_relocalization_verification_failure(expected)
                .expect("expected geometric rejection");
        }

        let error = SlamTracker::classify_relocalization_verification_failure(
            LoopVerificationError::PnpFailed(crate::PnpError::Numerical {
                operation: "testing relocalization classification",
                value: f64::MAX,
            }),
        )
        .expect_err("non-solver PnP errors must propagate");

        assert!(matches!(
            error,
            TrackerError::Pnp(crate::PnpError::Numerical {
                operation: "testing relocalization classification",
                value: f64::MAX,
            })
        ));
    }

    #[test]
    fn relocalization_failure_transitions_respect_max_attempts() {
        let cfg = RelocalizationConfig::new(crate::loop_closure::RelocalizationConfigInput {
            max_attempts: 2,
            ..crate::loop_closure::RelocalizationConfigInput::default()
        })
        .expect("relocalization config");
        let first_attempt = SlamTracker::begin_relocalization_attempt(
            RelocalizationSession {
                attempts: 0,
                phase: RelocalizationPhase::Searching,
            },
            cfg,
        )
        .expect("first attempt should be available");
        assert_eq!(first_attempt.session.attempts, 1);
        assert!(!first_attempt.is_final);
        assert!(matches!(
            SlamTracker::relocalization_fallback_state(&first_attempt),
            TrackerState::Relocalizing(RelocalizationSession {
                attempts: 1,
                phase: RelocalizationPhase::Searching,
            })
        ));
        let keep_trying = SlamTracker::relocalization_step(
            first_attempt,
            RelocalizationEvidence::NoCandidate,
            cfg,
        )
        .expect("finite relocalization poses");
        let RelocalizationStep::Continue(updated) = keep_trying else {
            panic!("expected relocalization to continue")
        };
        assert_eq!(updated.attempts, 1);
        assert!(matches!(updated.phase, RelocalizationPhase::Searching));

        let final_attempt = SlamTracker::begin_relocalization_attempt(
            RelocalizationSession {
                attempts: 1,
                phase: RelocalizationPhase::Searching,
            },
            cfg,
        )
        .expect("final attempt should be available");
        assert!(final_attempt.is_final);
        assert!(matches!(
            SlamTracker::relocalization_fallback_state(&final_attempt),
            TrackerState::Relocalizing(RelocalizationSession {
                attempts: 2,
                phase: RelocalizationPhase::Searching,
            })
        ));
        let give_up = SlamTracker::relocalization_step(
            final_attempt,
            RelocalizationEvidence::NoCandidate,
            cfg,
        )
        .expect("finite relocalization poses");
        assert!(matches!(give_up, RelocalizationStep::Exhausted));
        assert!(
            SlamTracker::begin_relocalization_attempt(
                RelocalizationSession {
                    attempts: 2,
                    phase: RelocalizationPhase::Searching,
                },
                cfg,
            )
            .is_none()
        );
    }

    #[test]
    fn relocalization_step_requires_confirmation_before_recovery() {
        let cfg = RelocalizationConfig::default();
        let candidate = KeyframeId::default();
        let pose = Pose::identity();

        let attempt = SlamTracker::begin_relocalization_attempt(
            RelocalizationSession {
                attempts: 0,
                phase: RelocalizationPhase::Searching,
            },
            cfg,
        )
        .expect("first attempt should be available");
        let step = SlamTracker::relocalization_step(
            attempt,
            RelocalizationEvidence::Verified {
                attachment: make_relocalization_attachment(candidate),
                pose_world: pose,
            },
            cfg,
        )
        .expect("finite relocalization poses");
        let RelocalizationStep::Continue(session) = step else {
            panic!("first successful relocalization should begin confirmation")
        };
        let RelocalizationPhase::Confirming {
            candidate: confirmed_candidate,
            confirmations,
            ..
        } = session.phase
        else {
            panic!("expected confirming phase")
        };
        assert_eq!(confirmed_candidate, candidate);
        assert_eq!(confirmations.get(), 1);
        assert_eq!(session.attempts, 1);
    }

    #[test]
    fn relocalization_step_recovers_on_final_allowed_confirmation() {
        let cfg = RelocalizationConfig::new(crate::loop_closure::RelocalizationConfigInput {
            max_attempts: 2,
            min_confirmations: 2,
            ..crate::loop_closure::RelocalizationConfigInput::default()
        })
        .expect("relocalization config");
        let candidate = KeyframeId::default();
        let pose = Pose::identity();

        let attempt = SlamTracker::begin_relocalization_attempt(
            RelocalizationSession {
                attempts: 1,
                phase: RelocalizationPhase::Confirming {
                    candidate,
                    confirmations: NonZeroUsize::new(1).expect("non-zero"),
                    pose_world: pose,
                },
            },
            cfg,
        )
        .expect("final attempt should be available");
        let step = SlamTracker::relocalization_step(
            attempt,
            RelocalizationEvidence::Verified {
                attachment: make_relocalization_attachment(candidate),
                pose_world: pose,
            },
            cfg,
        )
        .expect("finite relocalization poses");

        assert!(matches!(step, RelocalizationStep::Recovered { .. }));
    }

    #[test]
    fn relocalization_step_exhausts_on_inconsistent_final_candidate() {
        let cfg = RelocalizationConfig::new(crate::loop_closure::RelocalizationConfigInput {
            max_attempts: 2,
            min_confirmations: 2,
            ..crate::loop_closure::RelocalizationConfigInput::default()
        })
        .expect("relocalization config");
        let first_candidate = KeyframeId::for_test(0);
        let second_candidate = KeyframeId::for_test(1);
        let pose = Pose::identity();
        let session = RelocalizationSession {
            attempts: 1,
            phase: RelocalizationPhase::Confirming {
                candidate: first_candidate,
                confirmations: NonZeroUsize::MIN,
                pose_world: pose,
            },
        };

        let different_candidate_attempt =
            SlamTracker::begin_relocalization_attempt(session.clone(), cfg)
                .expect("final attempt should be available");
        let different_candidate = SlamTracker::relocalization_step(
            different_candidate_attempt,
            RelocalizationEvidence::Verified {
                attachment: make_relocalization_attachment(second_candidate),
                pose_world: pose,
            },
            cfg,
        )
        .expect("finite relocalization poses");
        assert!(matches!(different_candidate, RelocalizationStep::Exhausted));

        let inconsistent_pose = Pose::from_rt(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [cfg.max_translation_delta_m() * 2.0, 0.0, 0.0],
        );
        let inconsistent_pose_attempt = SlamTracker::begin_relocalization_attempt(session, cfg)
            .expect("final attempt should be available");
        let inconsistent_pose = SlamTracker::relocalization_step(
            inconsistent_pose_attempt,
            RelocalizationEvidence::Verified {
                attachment: make_relocalization_attachment(first_candidate),
                pose_world: inconsistent_pose,
            },
            cfg,
        )
        .expect("finite relocalization poses");
        assert!(matches!(inconsistent_pose, RelocalizationStep::Exhausted));
    }

    #[test]
    fn relocalization_pose_consistency_enforces_translation_and_rotation_limits() {
        let cfg = RelocalizationConfig::default();
        let identity = Pose::identity();

        let within_translation = Pose::from_rt(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [cfg.max_translation_delta_m() * 0.5, 0.0, 0.0],
        );
        assert!(
            SlamTracker::relocalization_pose_consistent(identity, within_translation, cfg)
                .expect("finite pose delta")
        );

        let beyond_translation = Pose::from_rt(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [cfg.max_translation_delta_m() * 1.5, 0.0, 0.0],
        );
        assert!(
            !SlamTracker::relocalization_pose_consistent(identity, beyond_translation, cfg)
                .expect("finite pose delta")
        );

        let half_angle = (cfg.max_rotation_delta_deg() * 0.5).to_radians();
        let within_rotation = Pose::from_rt(
            [
                [half_angle.cos(), -half_angle.sin(), 0.0],
                [half_angle.sin(), half_angle.cos(), 0.0],
                [0.0, 0.0, 1.0],
            ],
            [0.0, 0.0, 0.0],
        );
        assert!(
            SlamTracker::relocalization_pose_consistent(identity, within_rotation, cfg)
                .expect("finite pose delta")
        );

        let over_angle = (cfg.max_rotation_delta_deg() * 1.5).to_radians();
        let beyond_rotation = Pose::from_rt(
            [
                [over_angle.cos(), -over_angle.sin(), 0.0],
                [over_angle.sin(), over_angle.cos(), 0.0],
                [0.0, 0.0, 1.0],
            ],
            [0.0, 0.0, 0.0],
        );
        assert!(
            !SlamTracker::relocalization_pose_consistent(identity, beyond_rotation, cfg)
                .expect("finite pose delta")
        );
    }
}
