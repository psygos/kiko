use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use crate::loop_closure::{
    aggregate_global_descriptor, match_descriptors_for_loop, DescriptorSource, KeyframeDatabase,
    LoopCandidate, LoopClosureConfig, LoopDetectError, PlaceMatch, RelocalizationCandidate,
    RelocalizationConfig, VerifiedLoop,
};
use crate::pose_graph::{
    EssentialEdge, EssentialEdgeKind, EssentialGraph, EssentialGraphError, PoseGraphConfig,
    PoseGraphOptimizer,
};
use crate::{
    map::{KeyframeId, MapPointId, SlamMap},
    solve_pnp_ransac, BaCorrection, BaResult, Detections, DiagnosticEvent, DownscaleFactor,
    EigenPlaces, Frame, FrameDiagnostics, FrameId, Keyframe, KeyframeRemovalReason, KeypointLimit,
    LightGlue, LocalBaConfig, LocalBundleAdjuster, LoopClosureRejectReason, MapObservation,
    Matches, ObservationSet, PinholeIntrinsics, PlaceDescriptorExtractor, Point3, Pose,
    RansacConfig, Raw, RectifiedStereo, StereoPair, SuperPoint, Timestamp, TriangulationConfig,
    TriangulationError, Triangulator, Verified,
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
    pub loop_closure: Option<LoopClosureConfig>,
    pub global_descriptor: Option<GlobalDescriptorConfig>,
    pub relocalization: Option<RelocalizationConfig>,
}

impl TrackerConfig {
    pub fn max_keypoints(&self) -> usize {
        self.max_keypoints.get()
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
            queue_depth: NonZeroUsize::new(2).expect("non-zero"),
        }
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
            queue_depth: NonZeroUsize::new(2).expect("non-zero"),
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct MapVersion(NonZeroU64);

impl MapVersion {
    fn initial() -> Self {
        Self(NonZeroU64::new(1).expect("non-zero"))
    }

    fn next(self) -> Self {
        let next = self.0.get().saturating_add(1).max(1);
        Self(NonZeroU64::new(next).expect("non-zero"))
    }

    fn as_u64(self) -> u64 {
        self.0.get()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct BackendRequestId(NonZeroU64);

impl BackendRequestId {
    fn from_counter(counter: &mut u64) -> Self {
        *counter = counter.saturating_add(1).max(1);
        Self(NonZeroU64::new(*counter).expect("non-zero"))
    }

    fn as_u64(self) -> u64 {
        self.0.get()
    }
}

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
        if keyframes.len() < 2 {
            return Err(BackendWindowError::TooFewKeyframes {
                required: 2,
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
    map_version: MapVersion,
    frame: Frame,
}

#[derive(Debug, Clone)]
struct DescriptorResponse {
    keyframe_id: KeyframeId,
    map_version: MapVersion,
    descriptor: crate::loop_closure::GlobalDescriptor,
}

#[derive(Debug, Clone)]
enum DescriptorWorkerResponse {
    Descriptor(Box<DescriptorResponse>),
    Failure {
        keyframe_id: KeyframeId,
        map_version: MapVersion,
        error: String,
    },
    WorkerPanic {
        keyframe_id: KeyframeId,
        map_version: MapVersion,
        message: String,
    },
}

#[derive(Debug)]
enum SubmitDescriptorError {
    QueueFull,
    Disconnected,
}

type DescriptorExtractorFactory =
    Arc<dyn Fn() -> Option<Box<dyn PlaceDescriptorExtractor>> + Send + Sync>;

struct DescriptorWorker {
    tx: Sender<DescriptorRequest>,
    rx: Receiver<DescriptorWorkerResponse>,
    _thread: thread::JoinHandle<()>,
}

impl DescriptorWorker {
    fn model_path() -> PathBuf {
        if let Ok(path) = std::env::var("KIKO_EIGENPLACES_MODEL") {
            return PathBuf::from(path);
        }
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("models")
            .join("eigenplaces.onnx")
    }

    fn spawn(
        config: GlobalDescriptorConfig,
        mut extractor: Box<dyn PlaceDescriptorExtractor>,
    ) -> Self {
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
                                map_version: request.map_version,
                                descriptor,
                            }))
                        }
                        Ok(Err(err)) => DescriptorWorkerResponse::Failure {
                            keyframe_id: request.keyframe_id,
                            map_version: request.map_version,
                            error: err.to_string(),
                        },
                        Err(payload) => DescriptorWorkerResponse::WorkerPanic {
                            keyframe_id: request.keyframe_id,
                            map_version: request.map_version,
                            message: panic_payload_to_string(payload.as_ref()),
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
            })
            .expect("descriptor worker thread");
        Self {
            tx,
            rx,
            _thread: thread,
        }
    }

    #[cfg(test)]
    fn spawn_with_extractor(
        config: GlobalDescriptorConfig,
        extractor: Box<dyn PlaceDescriptorExtractor>,
    ) -> Self {
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

fn panic_payload_to_string(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(msg) = payload.downcast_ref::<&'static str>() {
        return (*msg).to_string();
    }
    if let Some(msg) = payload.downcast_ref::<String>() {
        return msg.clone();
    }
    "unknown panic payload".to_string()
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
    fn default_factory() -> DescriptorExtractorFactory {
        Arc::new(|| {
            let path = DescriptorWorker::model_path();
            EigenPlaces::try_load(path, crate::InferenceBackend::auto())
                .map(|extractor| Box::new(extractor) as Box<dyn PlaceDescriptorExtractor>)
        })
    }

    fn spawn(config: GlobalDescriptorConfig) -> Self {
        Self::with_factory_and_max_respawns(config, Self::default_factory(), 3)
    }

    fn with_factory_and_max_respawns(
        config: GlobalDescriptorConfig,
        factory: DescriptorExtractorFactory,
        max_respawns: u32,
    ) -> Self {
        let worker = Self::spawn_worker(config, &factory);
        let spawn_exhausted = worker.is_none();
        if worker.is_none() {
            eprintln!("descriptor model unavailable; using bootstrap descriptors only");
        }
        Self {
            worker,
            config,
            factory,
            respawn_count: 0,
            max_respawns,
            spawn_exhausted,
        }
    }

    fn spawn_worker(
        config: GlobalDescriptorConfig,
        factory: &DescriptorExtractorFactory,
    ) -> Option<DescriptorWorker> {
        let extractor = factory()?;
        Some(DescriptorWorker::spawn(config, extractor))
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
        let Some(worker) = Self::spawn_worker(self.config, &self.factory) else {
            self.spawn_exhausted = true;
            eprintln!("descriptor worker respawn failed; using bootstrap descriptors");
            return;
        };
        self.worker = Some(worker);
        self.respawn_count = self.respawn_count.saturating_add(1);
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
    map_version: MapVersion,
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
        map_version: MapVersion,
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
        Ok(Self {
            request_id,
            map_version,
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
    map_version: MapVersion,
    trigger_keyframe: KeyframeId,
    correction: BaCorrection,
}

#[derive(Debug)]
enum CorrectionBuildError {
    MissingKeyframe { keyframe_id: KeyframeId },
    MissingMapPoint { point_id: MapPointId },
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
        }
    }
}

impl std::error::Error for CorrectionBuildError {}

impl CorrectionEvent {
    fn from_optimized_map(
        event: &KeyframeEvent,
        optimized_map: &SlamMap,
        result: BaResult,
    ) -> Result<Self, CorrectionBuildError> {
        let mut correction = BaCorrection {
            pose_deltas: Vec::new(),
            landmark_deltas: Vec::new(),
            result: result.clone(),
        };

        if matches!(
            result,
            BaResult::Converged { .. } | BaResult::MaxIterations { .. }
        ) {
            correction.pose_deltas = Vec::with_capacity(event.window.as_slice().len());
            for &keyframe_id in event.window.as_slice() {
                let before = event
                    .map_snapshot
                    .keyframe(keyframe_id)
                    .ok_or(CorrectionBuildError::MissingKeyframe { keyframe_id })?;
                let after = optimized_map
                    .keyframe(keyframe_id)
                    .ok_or(CorrectionBuildError::MissingKeyframe { keyframe_id })?;
                let delta = crate::local_ba::se3_delta_between(before.pose(), after.pose());
                correction.pose_deltas.push((keyframe_id, delta));
            }

            let point_ids = collect_window_points(optimized_map, &event.window)?;
            correction.landmark_deltas = Vec::with_capacity(point_ids.len());
            for point_id in point_ids {
                let before = event
                    .map_snapshot
                    .point(point_id)
                    .ok_or(CorrectionBuildError::MissingMapPoint { point_id })?;
                let after = optimized_map
                    .point(point_id)
                    .ok_or(CorrectionBuildError::MissingMapPoint { point_id })?;
                let before_pos = before.position();
                let after_pos = after.position();
                correction.landmark_deltas.push((
                    point_id,
                    [
                        after_pos.x - before_pos.x,
                        after_pos.y - before_pos.y,
                        after_pos.z - before_pos.z,
                    ],
                ));
            }
        }

        Ok(Self {
            request_id: event.request_id,
            map_version: event.map_version,
            trigger_keyframe: event.trigger_keyframe,
            correction,
        })
    }
}

#[derive(Debug)]
enum BackendWorkerError {
    BuildCorrection(CorrectionBuildError),
}

impl std::fmt::Display for BackendWorkerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendWorkerError::BuildCorrection(err) => {
                write!(f, "backend correction build failed: {err}")
            }
        }
    }
}

impl std::error::Error for BackendWorkerError {}

#[derive(Debug)]
enum BackendResponse {
    Correction(CorrectionEvent),
    WorkerPanic {
        request_id: BackendRequestId,
        map_version: MapVersion,
    },
    Failure {
        request_id: BackendRequestId,
        map_version: MapVersion,
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
    QueueFull,
    Disconnected,
}

impl std::fmt::Display for SubmitEventError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SubmitEventError::InvalidWindow(err) => write!(f, "invalid backend window: {err}"),
            SubmitEventError::InvalidEvent(err) => write!(f, "invalid backend event: {err}"),
            SubmitEventError::QueueFull => write!(f, "backend event queue is full"),
            SubmitEventError::Disconnected => write!(f, "backend worker is disconnected"),
        }
    }
}

impl std::error::Error for SubmitEventError {}

#[derive(Debug)]
enum ApplyCorrectionError {
    StaleVersion {
        current: MapVersion,
        correction: MapVersion,
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
            ApplyCorrectionError::StaleVersion {
                current,
                correction,
            } => write!(
                f,
                "stale correction: correction version={}, current version={}",
                correction.as_u64(),
                current.as_u64()
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

impl std::error::Error for ApplyCorrectionError {}

impl From<crate::map::MapError> for ApplyCorrectionError {
    fn from(value: crate::map::MapError) -> Self {
        Self::Map(value)
    }
}

#[derive(Debug)]
struct BackendWorker {
    tx: Sender<KeyframeEvent>,
    rx: Receiver<BackendResponse>,
    next_request_id: u64,
}

impl BackendWorker {
    fn spawn(
        config: BackendConfig,
        intrinsics: PinholeIntrinsics,
        ba_config: LocalBaConfig,
    ) -> Self {
        let (tx_req, rx_req) = crossbeam_channel::bounded::<KeyframeEvent>(config.queue_depth());
        let (tx_resp, rx_resp) = crossbeam_channel::unbounded::<BackendResponse>();

        thread::Builder::new()
            .name("kiko-backend".to_string())
            .spawn(move || {
                let mut ba = LocalBundleAdjuster::new(intrinsics, ba_config);
                while let Ok(event) = rx_req.recv() {
                    let request_id = event.request_id;
                    let map_version = event.map_version;
                    let processing = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        #[cfg(test)]
                        if event.force_panic {
                            panic!("forced backend worker panic");
                        }
                        let mut optimized_map = event.map_snapshot.clone();
                        let result = ba
                            .optimize_keyframe_window(&mut optimized_map, event.window.as_slice());
                        CorrectionEvent::from_optimized_map(&event, &optimized_map, result)
                    }));

                    match processing {
                        Ok(Ok(correction)) => {
                            let _ = tx_resp.send(BackendResponse::Correction(correction));
                        }
                        Ok(Err(err)) => {
                            let _ = tx_resp.send(BackendResponse::Failure {
                                request_id,
                                map_version,
                                error: BackendWorkerError::BuildCorrection(err),
                            });
                        }
                        Err(_) => {
                            let _ = tx_resp.send(BackendResponse::WorkerPanic {
                                request_id,
                                map_version,
                            });
                            break;
                        }
                    }
                }
            })
            .expect("spawn backend worker");

        Self {
            tx: tx_req,
            rx: rx_resp,
            next_request_id: 0,
        }
    }

    fn next_request_id(&mut self) -> BackendRequestId {
        BackendRequestId::from_counter(&mut self.next_request_id)
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
}

impl BackendSupervisor {
    fn spawn(
        config: BackendConfig,
        intrinsics: PinholeIntrinsics,
        ba_config: LocalBaConfig,
    ) -> Self {
        Self {
            worker: Some(BackendWorker::spawn(config, intrinsics, ba_config)),
            config,
            intrinsics,
            ba_config,
            respawn_count: 0,
            max_respawns: 3,
        }
    }

    #[cfg(test)]
    fn with_max_respawns(
        config: BackendConfig,
        intrinsics: PinholeIntrinsics,
        ba_config: LocalBaConfig,
        max_respawns: u32,
    ) -> Self {
        let mut supervisor = Self::spawn(config, intrinsics, ba_config);
        supervisor.max_respawns = max_respawns;
        supervisor
    }

    fn check_health(&mut self) {
        if self.worker.is_none() {
            return;
        }

        if self.respawn_count >= self.max_respawns {
            self.worker = None;
            return;
        }

        eprintln!(
            "backend worker disconnected; respawning ({}/{})",
            self.respawn_count + 1,
            self.max_respawns
        );
        self.worker = Some(BackendWorker::spawn(
            self.config,
            self.intrinsics,
            self.ba_config,
        ));
        self.respawn_count = self.respawn_count.saturating_add(1);
    }

    fn submit(&mut self, event: KeyframeEvent) -> Result<(), SubmitEventError> {
        let Some(worker) = self.worker.as_ref() else {
            return Err(SubmitEventError::Disconnected);
        };
        let result = worker.try_submit(event);
        if matches!(result, Err(SubmitEventError::Disconnected)) {
            self.check_health();
        }
        result
    }

    fn try_recv(&mut self) -> Option<BackendResponse> {
        let worker = self.worker.as_ref()?;
        match worker.try_recv() {
            Ok(response) => response,
            Err(()) => {
                self.check_health();
                None
            }
        }
    }

    fn next_request_id(&mut self) -> Option<BackendRequestId> {
        self.worker.as_mut().map(BackendWorker::next_request_id)
    }

    #[cfg(test)]
    fn shutdown(&mut self) {
        self.worker = None;
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
        if let Some(parallax) = parallax_px {
            if parallax > self.parallax_px.0 {
                return true;
            }
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
    Triangulation(TriangulationError),
    Pnp(crate::PnpError),
    Map(crate::map::MapError),
    EssentialGraph(EssentialGraphError),
    KeyframeRejected { landmarks: usize },
}

impl std::fmt::Display for TrackerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrackerError::Inference(err) => write!(f, "inference error: {err}"),
            TrackerError::Triangulation(err) => write!(f, "triangulation error: {err}"),
            TrackerError::Pnp(err) => write!(f, "pnp error: {err}"),
            TrackerError::Map(err) => write!(f, "map error: {err}"),
            TrackerError::EssentialGraph(err) => write!(f, "essential graph error: {err}"),
            TrackerError::KeyframeRejected { landmarks } => {
                write!(f, "keyframe rejected: only {landmarks} landmarks")
            }
        }
    }
}

impl std::error::Error for TrackerError {}

impl From<InferenceError> for TrackerError {
    fn from(err: InferenceError) -> Self {
        TrackerError::Inference(err)
    }
}

impl From<TriangulationError> for TrackerError {
    fn from(err: TriangulationError) -> Self {
        TrackerError::Triangulation(err)
    }
}

impl From<crate::PnpError> for TrackerError {
    fn from(err: crate::PnpError) -> Self {
        TrackerError::Pnp(err)
    }
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrackingHealth {
    Good,
    Degraded,
    Lost,
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
        if a.rank() >= b.rank() {
            a
        } else {
            b
        }
    }
}

#[derive(Clone, Debug)]
pub struct SystemHealth {
    pub tracking: TrackingHealth,
    pub backend_alive: bool,
    pub descriptor_alive: bool,
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
        let descriptor_degradation = if descriptor_expected && !descriptor_alive {
            DegradationLevel::DescriptorDown
        } else {
            DegradationLevel::Nominal
        };
        let backend_degradation = if backend_expected && !backend_alive {
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
            backend_alive,
            descriptor_alive,
            backend_stats,
            degradation,
        }
    }
}

#[derive(Debug)]
pub struct TrackerOutput {
    pub pose: Option<Pose>,
    pub inliers: usize,
    pub keyframe: Option<Arc<Keyframe>>,
    pub stereo_matches: Option<Matches<Raw>>,
    pub frame_id: FrameId,
    pub health: SystemHealth,
    pub diagnostics: FrameDiagnostics,
    pub events: Vec<DiagnosticEvent>,
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
    last_detections: Arc<Detections>,
}

#[derive(Debug)]
enum RelocalizationStep {
    Continue(RelocalizationSession),
    Recovered { pose_world: Pose },
}

#[derive(Debug)]
struct SharedMatches {
    keyframe_id: KeyframeId,
    pairs: Vec<(usize, usize)>,
}

#[derive(Debug)]
struct PendingLoopCandidate {
    query_kf: KeyframeId,
    detections: Arc<Detections>,
    candidates: Vec<PlaceMatch>,
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
    map_version: MapVersion,
    backend: Option<BackendSupervisor>,
    backend_stats: BackendStats,
    descriptor_worker: Option<DescriptorSupervisor>,
    descriptor_stats: DescriptorStats,
    loop_db: Option<KeyframeDatabase>,
    loop_config: Option<LoopClosureConfig>,
    pending_loop: Option<PendingLoopCandidate>,
    loop_streak: HashMap<KeyframeId, usize>,
    pending_events: Vec<DiagnosticEvent>,
    tracking_health: TrackingHealth,
    consecutive_tracking_failures: usize,
}

impl SlamTracker {
    const DEFAULT_ESSENTIAL_GRAPH_STRONG_THRESHOLD: u32 = 15;

    pub fn new(
        superpoint_left: SuperPoint,
        superpoint_right: SuperPoint,
        lightglue: LightGlue,
        stereo: RectifiedStereo,
        intrinsics: PinholeIntrinsics,
        config: TrackerConfig,
    ) -> Self {
        let triangulator = Triangulator::new(stereo, config.triangulation);
        let ba = LocalBundleAdjuster::new(intrinsics, config.ba);
        let backend = config
            .backend
            .map(|backend_cfg| BackendSupervisor::spawn(backend_cfg, intrinsics, config.ba));
        let loop_config = config.loop_closure;
        let loop_db = loop_config.map(|cfg| KeyframeDatabase::new(cfg.temporal_gap()));
        let descriptor_worker = if loop_config.is_some() {
            config.global_descriptor.map(DescriptorSupervisor::spawn)
        } else {
            None
        };
        Self {
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
            map_version: MapVersion::initial(),
            backend,
            backend_stats: BackendStats::default(),
            descriptor_worker,
            descriptor_stats: DescriptorStats::default(),
            loop_db,
            loop_config,
            pending_loop: None,
            loop_streak: HashMap::new(),
            pending_events: Vec::new(),
            tracking_health: TrackingHealth::Good,
            consecutive_tracking_failures: 0,
        }
    }

    pub fn process(&mut self, pair: StereoPair) -> Result<TrackerOutput, TrackerError> {
        self.drain_backend_responses();
        self.drain_descriptor_responses();
        if let Err(err) = self.process_pending_loop_closure() {
            eprintln!("loop closure: {err}");
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
            let result = self.relocalize(pair, session);
            if result.is_err() {
                self.clear_events();
            }
            return result;
        }

        let result = if let Some((keyframe, keyframe_id)) = tracking {
            self.track(pair, &keyframe, keyframe_id)
        } else {
            self.create_keyframe(pair, Pose::identity())
        };
        if result.is_err() {
            self.clear_events();
        }
        result
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
        let backend_expected = self.backend.is_some();
        let backend_alive = self
            .backend
            .as_ref()
            .is_none_or(BackendSupervisor::has_worker);
        let descriptor_expected = self.descriptor_worker.is_some();
        let descriptor_alive = self
            .descriptor_worker
            .as_ref()
            .is_none_or(DescriptorSupervisor::has_worker);
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
        apply_loop_closure_correction(
            &mut self.map,
            &mut self.essential_graph,
            &self.pose_graph_optimizer,
            &verified,
        )?;
        self.bump_map_version();
        Ok(())
    }

    fn bump_map_version(&mut self) {
        self.map_version = self.map_version.next();
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

    fn clear_events(&mut self) {
        self.pending_events.clear();
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
        inliers: usize,
        keyframe: Option<Arc<Keyframe>>,
        stereo_matches: Option<Matches<Raw>>,
        frame_id: FrameId,
        tracking: TrackingHealth,
        diagnostics: FrameDiagnostics,
    ) -> TrackerOutput {
        TrackerOutput {
            pose,
            inliers,
            keyframe,
            stereo_matches,
            frame_id,
            health: self.emit_health(tracking),
            diagnostics,
            events: self.drain_events(),
        }
    }

    fn enqueue_loop_candidates(&mut self, keyframe_id: KeyframeId, detections: &Arc<Detections>) {
        let (Some(config), Some(loop_db)) = (self.loop_config, self.loop_db.as_mut()) else {
            return;
        };

        let Ok(global_descriptor) = aggregate_global_descriptor(detections.descriptors()) else {
            return;
        };
        loop_db.insert_with_source(
            keyframe_id,
            global_descriptor.clone(),
            DescriptorSource::Bootstrap,
        );

        let mut candidates = loop_db.query(&global_descriptor, config.max_candidates());
        candidates.retain(|candidate| candidate.similarity >= config.similarity_threshold());

        if candidates.is_empty() {
            self.loop_streak.clear();
            return;
        }

        let present: HashSet<KeyframeId> = candidates.iter().map(|m| m.candidate).collect();
        self.loop_streak
            .retain(|candidate, _| present.contains(candidate));
        for candidate in &candidates {
            let streak = self.loop_streak.entry(candidate.candidate).or_insert(0);
            *streak = streak.saturating_add(1);
        }

        if self.pending_loop.is_some() {
            return;
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
            return;
        }

        self.pending_loop = Some(PendingLoopCandidate {
            query_kf: keyframe_id,
            detections: Arc::clone(detections),
            candidates: promoted,
        });
    }

    fn enqueue_descriptor_request(&mut self, keyframe_id: KeyframeId, frame: &Frame) {
        let Some(supervisor) = self.descriptor_worker.as_mut() else {
            return;
        };
        let request = DescriptorRequest {
            keyframe_id,
            map_version: self.map_version,
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

    fn drain_descriptor_responses(&mut self) {
        loop {
            let response = {
                let Some(supervisor) = self.descriptor_worker.as_mut() else {
                    return;
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
                    if response.map_version.as_u64() > self.map_version.as_u64() {
                        continue;
                    }
                    if self.map.keyframe(response.keyframe_id).is_none() {
                        continue;
                    }
                    let Some(loop_db) = self.loop_db.as_mut() else {
                        continue;
                    };
                    if loop_db.replace_descriptor(
                        response.keyframe_id,
                        response.descriptor,
                        DescriptorSource::Learned,
                    ) {
                        self.descriptor_stats.applied =
                            self.descriptor_stats.applied.saturating_add(1);
                    }
                }
                DescriptorWorkerResponse::Failure {
                    keyframe_id,
                    map_version,
                    error,
                } => {
                    self.descriptor_stats.worker_failures =
                        self.descriptor_stats.worker_failures.saturating_add(1);
                    eprintln!(
                        "descriptor worker failure (keyframe={keyframe_id:?}, version={}): {error}",
                        map_version.as_u64()
                    );
                }
                DescriptorWorkerResponse::WorkerPanic {
                    keyframe_id,
                    map_version,
                    message,
                } => {
                    self.descriptor_stats.panics = self.descriptor_stats.panics.saturating_add(1);
                    self.descriptor_stats.worker_failures =
                        self.descriptor_stats.worker_failures.saturating_add(1);
                    eprintln!(
                        "descriptor worker panic (keyframe={keyframe_id:?}, version={}): {message}",
                        map_version.as_u64()
                    );
                    self.emit_event(DiagnosticEvent::DescriptorWorkerDied {
                        respawn_count: self.descriptor_stats.respawn_count,
                    });
                }
            }
        }
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
            let correspondences = match_descriptors_for_loop(
                pending.detections.descriptors(),
                candidate.candidate,
                &self.map,
                config.descriptor_match_threshold(),
            );

            if correspondences.len() < 4 {
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
                similarity: candidate.similarity,
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
                Err(err) => {
                    if first_error.is_none() {
                        first_error = Some(LoopDetectError::VerificationFailed(err));
                    }
                    continue;
                }
            };

            let translation = loop_translation_norm(verified.relative_pose());
            let rotation_deg = loop_rotation_angle_deg(verified.relative_pose());
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
                let detect_err = LoopDetectError::ApplyFailed(err.to_string());
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
        let request_id = supervisor
            .next_request_id()
            .ok_or(SubmitEventError::Disconnected)?;
        let event = KeyframeEvent::try_new(
            request_id,
            self.map_version,
            trigger_keyframe,
            window,
            self.map.clone(),
        )
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
                BackendResponse::Correction(correction) => match &correction.correction.result {
                    BaResult::Converged { .. } | BaResult::MaxIterations { .. } => {
                        match apply_correction_event(&mut self.map, self.map_version, &correction) {
                            Ok(()) => {
                                self.bump_map_version();
                                self.backend_stats.applied =
                                    self.backend_stats.applied.saturating_add(1);
                            }
                            Err(ApplyCorrectionError::StaleVersion { .. }) => {
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
                        self.backend_stats.rejected = self.backend_stats.rejected.saturating_add(1);
                        self.emit_event(DiagnosticEvent::BaDegenerate { reason: *reason });
                        eprintln!(
                            "backend BA degenerate (req={}, keyframe={:?}): {reason:?}",
                            correction.request_id.as_u64(),
                            correction.trigger_keyframe
                        );
                    }
                },
                BackendResponse::Failure {
                    request_id,
                    map_version,
                    error,
                } => {
                    self.backend_stats.worker_failures =
                        self.backend_stats.worker_failures.saturating_add(1);
                    eprintln!(
                        "backend worker failure (req={}, version={}): {error}",
                        request_id.as_u64(),
                        map_version.as_u64()
                    );
                }
                BackendResponse::WorkerPanic {
                    request_id,
                    map_version,
                } => {
                    self.backend_stats.panics = self.backend_stats.panics.saturating_add(1);
                    self.backend_stats.worker_failures =
                        self.backend_stats.worker_failures.saturating_add(1);
                    self.map_version = self.map_version.next();
                    if let Some(supervisor) = self.backend.as_mut() {
                        supervisor.check_health();
                        self.backend_stats.respawn_count = supervisor.respawn_count();
                    }
                    self.emit_event(DiagnosticEvent::BackendWorkerDied {
                        respawn_count: self.backend_stats.respawn_count,
                    });
                    eprintln!(
                        "backend worker panic (req={}, version={}); map version bumped to invalidate in-flight",
                        request_id.as_u64(),
                        map_version.as_u64()
                    );
                }
            }
        }
    }

    fn tracking_failure_health(&mut self) -> TrackingHealth {
        const LOST_AFTER_CONSECUTIVE_FAILURES: usize = 3;
        self.consecutive_tracking_failures = self.consecutive_tracking_failures.saturating_add(1);
        if self.consecutive_tracking_failures >= LOST_AFTER_CONSECUTIVE_FAILURES {
            if self.tracking_health != TrackingHealth::Lost {
                self.emit_event(DiagnosticEvent::TrackingLost {
                    consecutive_failures: self.consecutive_tracking_failures,
                });
            }
            TrackingHealth::Lost
        } else {
            TrackingHealth::Degraded
        }
    }

    fn maybe_enter_relocalization(
        &mut self,
        tracking_health: TrackingHealth,
        detections: Arc<Detections>,
    ) {
        if let Some(session) = Self::initial_relocalization_session(
            tracking_health,
            self.config.relocalization.is_some(),
            detections,
        ) {
            self.emit_event(DiagnosticEvent::RelocalizationStarted);
            self.pending_loop = None;
            self.loop_streak.clear();
            self.state = TrackerState::Relocalizing(session);
        }
    }

    fn initial_relocalization_session(
        tracking_health: TrackingHealth,
        relocalization_enabled: bool,
        detections: Arc<Detections>,
    ) -> Option<RelocalizationSession> {
        if tracking_health != TrackingHealth::Lost || !relocalization_enabled {
            return None;
        }
        Some(RelocalizationSession {
            attempts: 0,
            phase: RelocalizationPhase::Searching,
            last_detections: detections,
        })
    }

    fn relocalization_output(
        &mut self,
        frame_id: FrameId,
        health: TrackingHealth,
    ) -> TrackerOutput {
        let diagnostics = self.empty_diagnostics();
        self.output_with_diagnostics(None, 0, None, None, frame_id, health, diagnostics)
    }

    fn relocalization_pose_consistent(
        previous_pose: Pose,
        current_pose: Pose,
        cfg: RelocalizationConfig,
    ) -> bool {
        let delta = crate::local_ba::se3_delta_between(previous_pose, current_pose);
        let translation_delta =
            (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
        let rotation_delta_deg = (delta[3] * delta[3] + delta[4] * delta[4] + delta[5] * delta[5])
            .sqrt()
            .to_degrees();
        translation_delta <= cfg.max_translation_delta_m()
            && rotation_delta_deg <= cfg.max_rotation_delta_deg()
    }

    fn fail_relocalization(
        &mut self,
        frame_id: FrameId,
        cfg: RelocalizationConfig,
        session: RelocalizationSession,
        current: Arc<Detections>,
    ) -> TrackerOutput {
        self.state = Self::next_state_after_relocalization_failure(cfg, session, current);
        self.relocalization_output(frame_id, TrackingHealth::Lost)
    }

    fn next_state_after_relocalization_failure(
        cfg: RelocalizationConfig,
        mut session: RelocalizationSession,
        current: Arc<Detections>,
    ) -> TrackerState {
        session.attempts = session.attempts.saturating_add(1);
        session.phase = RelocalizationPhase::Searching;
        session.last_detections = current;
        if session.attempts >= cfg.max_attempts() {
            TrackerState::NeedKeyframe
        } else {
            TrackerState::Relocalizing(session)
        }
    }

    fn relocalization_candidate(
        &self,
        current: &Detections,
        cfg: RelocalizationConfig,
        loop_db: &KeyframeDatabase,
    ) -> Option<crate::loop_closure::VerifiedRelocalization> {
        let global_descriptor = aggregate_global_descriptor(current.descriptors()).ok()?;
        let candidates = loop_db.query_for_relocalization(&global_descriptor, cfg.max_candidates());
        for candidate in candidates {
            let correspondences = match_descriptors_for_loop(
                current.descriptors(),
                candidate.candidate,
                &self.map,
                cfg.descriptor_match_threshold(),
            );
            if correspondences.len() < 4 {
                continue;
            }
            let relocalization_candidate = RelocalizationCandidate {
                match_kf: candidate.candidate,
                similarity: candidate.similarity,
            };
            let verified = match relocalization_candidate.verify(
                current.keypoints(),
                &correspondences,
                &self.map,
                self.intrinsics,
                self.config.ransac,
                cfg.min_inliers(),
            ) {
                Ok(value) => value,
                Err(_) => continue,
            };
            return Some(verified);
        }
        None
    }

    fn relocalization_step(
        session: RelocalizationSession,
        candidate_id: KeyframeId,
        pose_world: Pose,
        cfg: RelocalizationConfig,
    ) -> RelocalizationStep {
        let required_confirmations = cfg.min_confirmations();
        match session.phase {
            RelocalizationPhase::Confirming {
                candidate,
                confirmations,
                pose_world: previous_pose,
            } if candidate == candidate_id
                && Self::relocalization_pose_consistent(previous_pose, pose_world, cfg) =>
            {
                let next_confirmations = confirmations.get().saturating_add(1);
                if next_confirmations >= required_confirmations {
                    return RelocalizationStep::Recovered { pose_world };
                }
                RelocalizationStep::Continue(RelocalizationSession {
                    attempts: session.attempts,
                    phase: RelocalizationPhase::Confirming {
                        candidate,
                        confirmations: NonZeroUsize::new(next_confirmations).expect("non-zero"),
                        pose_world,
                    },
                    last_detections: session.last_detections,
                })
            }
            _ if required_confirmations <= 1 => RelocalizationStep::Recovered { pose_world },
            _ => RelocalizationStep::Continue(RelocalizationSession {
                attempts: session.attempts,
                phase: RelocalizationPhase::Confirming {
                    candidate: candidate_id,
                    confirmations: NonZeroUsize::new(1).expect("non-zero"),
                    pose_world,
                },
                last_detections: session.last_detections,
            }),
        }
    }

    fn relocalize(
        &mut self,
        pair: StereoPair,
        mut session: RelocalizationSession,
    ) -> Result<TrackerOutput, TrackerError> {
        let StereoPair { left, right } = pair;
        let frame_id = left.frame_id();
        let Some(cfg) = self.config.relocalization else {
            self.state = TrackerState::NeedKeyframe;
            return Ok(self.relocalization_output(frame_id, TrackingHealth::Lost));
        };

        let current = self
            .superpoint_left
            .detect_with_downscale(&left, self.config.downscale)?
            .top_k(self.config.max_keypoints());
        let current = Arc::new(current);

        if current.is_empty() {
            return Ok(self.fail_relocalization(frame_id, cfg, session, Arc::clone(&current)));
        }

        let Some(loop_db) = self.loop_db.as_ref() else {
            self.state = TrackerState::NeedKeyframe;
            return Ok(self.relocalization_output(frame_id, TrackingHealth::Lost));
        };

        let Some(verified) = self.relocalization_candidate(current.as_ref(), cfg, loop_db) else {
            return Ok(self.fail_relocalization(frame_id, cfg, session, current));
        };
        let candidate_id = verified.match_kf();
        let pose_world = verified.pose_world();

        session.last_detections = current;
        match Self::relocalization_step(session, candidate_id, pose_world, cfg) {
            RelocalizationStep::Recovered { pose_world } => {
                self.emit_event(DiagnosticEvent::RelocalizationSucceeded {
                    keyframe_id: candidate_id,
                });
                self.pending_loop = None;
                self.loop_streak.clear();
                self.state = TrackerState::NeedKeyframe;
                return self.create_keyframe(StereoPair { left, right }, pose_world);
            }
            RelocalizationStep::Continue(next_session) => {
                self.state = TrackerState::Relocalizing(next_session);
            }
        }
        Ok(self.relocalization_output(frame_id, TrackingHealth::Degraded))
    }

    fn track(
        &mut self,
        pair: StereoPair,
        keyframe: &Arc<Keyframe>,
        keyframe_id: KeyframeId,
    ) -> Result<TrackerOutput, TrackerError> {
        let tracking_start = Instant::now();
        let StereoPair { left, right } = pair;
        let frame_id = left.frame_id();

        let current = self
            .superpoint_left
            .detect_with_downscale(&left, self.config.downscale)?
            .top_k(self.config.max_keypoints());
        let current = Arc::new(current);

        let matches = if current.is_empty() || keyframe.detections().is_empty() {
            let tracking_health = self.tracking_failure_health();
            self.maybe_enter_relocalization(tracking_health, Arc::clone(&current));
            let mut diagnostics = self.empty_diagnostics();
            diagnostics.features_detected = Some(current.len());
            diagnostics.features_matched = Some(0);
            diagnostics.tracking_time = Some(tracking_start.elapsed());
            diagnostics.loop_candidate_count = self
                .pending_loop
                .as_ref()
                .map_or(0, |pending| pending.candidates.len());
            diagnostics.loop_closure_applied = self
                .pending_events
                .iter()
                .any(|event| matches!(event, DiagnosticEvent::LoopClosureDetected { .. }));
            return Ok(self.output_with_diagnostics(
                None,
                0,
                None,
                None,
                frame_id,
                tracking_health,
                diagnostics,
            ));
        } else {
            self.lightglue
                .match_these(current.clone(), keyframe.detections().clone())?
        };

        let verified = match matches.with_landmarks(keyframe) {
            Ok(verified) => verified,
            Err(err) => {
                return Err(TrackerError::Inference(InferenceError::Domain(format!(
                    "{err:?}"
                ))));
            }
        };

        let observations = match build_map_observations(
            &self.map,
            keyframe_id,
            &verified,
            current.as_ref(),
            self.intrinsics,
        ) {
            Ok(obs) => obs,
            Err(crate::PnpError::NotEnoughPoints { .. }) => {
                let tracking_health = self.tracking_failure_health();
                self.maybe_enter_relocalization(tracking_health, Arc::clone(&current));
                let mut diagnostics = self.empty_diagnostics();
                diagnostics.features_detected = Some(current.len());
                diagnostics.features_matched = Some(matches.len());
                diagnostics.tracking_time = Some(tracking_start.elapsed());
                diagnostics.loop_candidate_count = self
                    .pending_loop
                    .as_ref()
                    .map_or(0, |pending| pending.candidates.len());
                diagnostics.loop_closure_applied = self
                    .pending_events
                    .iter()
                    .any(|event| matches!(event, DiagnosticEvent::LoopClosureDetected { .. }));
                return Ok(self.output_with_diagnostics(
                    None,
                    0,
                    None,
                    None,
                    frame_id,
                    tracking_health,
                    diagnostics,
                ));
            }
            Err(err) => return Err(TrackerError::Pnp(err)),
        };

        let result = match solve_pnp_ransac(&observations, self.intrinsics, self.config.ransac) {
            Ok(result) => result,
            Err(crate::PnpError::NotEnoughPoints { .. } | crate::PnpError::NoSolution) => {
                let tracking_health = self.tracking_failure_health();
                self.maybe_enter_relocalization(tracking_health, Arc::clone(&current));
                let mut diagnostics = self.empty_diagnostics();
                diagnostics.features_detected = Some(current.len());
                diagnostics.features_matched = Some(matches.len());
                diagnostics.pnp_observations = Some(observations.len());
                diagnostics.tracking_time = Some(tracking_start.elapsed());
                diagnostics.loop_candidate_count = self
                    .pending_loop
                    .as_ref()
                    .map_or(0, |pending| pending.candidates.len());
                diagnostics.loop_closure_applied = self
                    .pending_events
                    .iter()
                    .any(|event| matches!(event, DiagnosticEvent::LoopClosureDetected { .. }));
                return Ok(self.output_with_diagnostics(
                    None,
                    0,
                    None,
                    None,
                    frame_id,
                    tracking_health,
                    diagnostics,
                ));
            }
            Err(err) => return Err(TrackerError::Pnp(err)),
        };

        let mut map_observations = Vec::with_capacity(result.inliers.len());
        for &idx in &result.inliers {
            let (ci, ki) = *verified.indices().get(idx).ok_or_else(|| {
                TrackerError::Inference(InferenceError::Domain(
                    "verified match index out of bounds".to_string(),
                ))
            })?;
            let pixel = *current.keypoints().get(ci).ok_or_else(|| {
                TrackerError::Inference(InferenceError::Domain(
                    "current keypoint index out of bounds".to_string(),
                ))
            })?;
            let keypoint_ref = self.map.keyframe_keypoint(keyframe_id, ki)?;
            map_observations.push(MapObservation::new(keypoint_ref, pixel));
        }

        let parallax_px = median_parallax_px(&verified, &result.inliers, keyframe);
        let covisibility = if keyframe.landmarks().is_empty() {
            0.0
        } else {
            result.inliers.len() as f32 / keyframe.landmarks().len() as f32
        };

        let pose_world = result.pose;
        let refined_world = ObservationSet::new(map_observations, self.ba.min_observations())
            .ok()
            .and_then(|set| self.ba.push_frame(&self.map, pose_world, set));

        let pose_world = refined_world.unwrap_or(pose_world);
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
            let shared = build_shared_matches(keyframe_id, &verified, &result.inliers);
            if let Ok((keyframe_output, keyframe_id)) = self.create_keyframe_internal(
                left,
                right,
                new_pose,
                Some(current.clone()),
                Some(shared),
            ) {
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
                        if let Err(err) = remove_keyframe_from_graph_and_db(
                            &mut self.map,
                            &mut self.essential_graph,
                            self.loop_db.as_mut(),
                            keyframe_id,
                        ) {
                            eprintln!("failed to remove redundant keyframe {keyframe_id:?}: {err}");
                        } else {
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
                            self.bump_map_version();
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
                                                if !supervisor.has_worker() {
                                                    self.backend = None;
                                                }
                                            } else {
                                                self.backend = None;
                                            }
                                        }
                                        SubmitEventError::InvalidWindow(_)
                                        | SubmitEventError::InvalidEvent(_) => {
                                            self.backend_stats.rejected =
                                                self.backend_stats.rejected.saturating_add(1);
                                        }
                                    }
                                    eprintln!(
                                        "backend submit failed for keyframe {keyframe_id:?}: {err}"
                                    );
                                    let result =
                                        self.ba.optimize_keyframe_window(&mut self.map, &window);
                                    ba_result = Some(result.clone());
                                    if matches!(
                                        result,
                                        BaResult::Converged { .. } | BaResult::MaxIterations { .. }
                                    ) {
                                        self.bump_map_version();
                                    }
                                }
                            } else if matches!(
                                {
                                    let result =
                                        self.ba.optimize_keyframe_window(&mut self.map, &window);
                                    ba_result = Some(result.clone());
                                    result
                                },
                                BaResult::Converged { .. } | BaResult::MaxIterations { .. }
                            ) {
                                self.bump_map_version();
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

        let inlier_observations: Vec<_> = result
            .inliers
            .iter()
            .filter_map(|&idx| observations.get(idx).copied())
            .collect();
        let inlier_errors =
            crate::pnp::reprojection_errors(&pose_world, &inlier_observations, self.intrinsics);

        let mut diagnostics = self.empty_diagnostics();
        diagnostics.inlier_ratio =
            Some(result.inliers.len() as f32 / observations.len().max(1) as f32);
        diagnostics.pnp_observations = Some(observations.len());
        diagnostics.ransac_iterations = Some(result.iterations);
        diagnostics.reprojection_rmse_px = crate::pnp::reprojection_rmse(&inlier_errors);
        diagnostics.reprojection_max_px = crate::pnp::reprojection_max(&inlier_errors);
        diagnostics.parallax_px = parallax_px;
        diagnostics.covisibility = Some(covisibility);
        diagnostics.keyframe_created = keyframe_created;
        diagnostics.triangulation = triangulation_stats;
        diagnostics.ba_result = ba_result;
        diagnostics.loop_candidate_count = self
            .pending_loop
            .as_ref()
            .map_or(0, |pending| pending.candidates.len());
        diagnostics.loop_closure_applied = self
            .pending_events
            .iter()
            .any(|event| matches!(event, DiagnosticEvent::LoopClosureDetected { .. }));
        diagnostics.tracking_time = Some(tracking_start.elapsed());
        diagnostics.features_detected = Some(current.len());
        diagnostics.features_matched = Some(matches.len());

        Ok(self.output_with_diagnostics(
            Some(pose_world),
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
        let StereoPair { left, right } = pair;
        let frame_id = left.frame_id();
        let (output, keyframe_id) =
            self.create_keyframe_internal(left, right, pose_world, None, None)?;
        let keyframe = output.keyframe.clone().expect("keyframe");
        self.state = TrackerState::Tracking {
            keyframe,
            keyframe_id,
        };
        self.ba.reset();
        self.consecutive_tracking_failures = 0;
        let diagnostics = output.diagnostics;
        Ok(self.output_with_diagnostics(
            Some(pose_world),
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
        pose_world: Pose,
        left_det: Option<Arc<Detections>>,
        shared: Option<SharedMatches>,
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

                let left_det = left_det.map_err(|_| {
                    InferenceError::Domain("left superpoint thread panicked".to_string())
                })??;
                let right_det = right_det.map_err(|_| {
                    InferenceError::Domain("right superpoint thread panicked".to_string())
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
            return Err(TrackerError::KeyframeRejected { landmarks });
        }

        let keyframe = Arc::new(result.keyframe);
        let keyframe_id = insert_keyframe_into_map(
            &mut self.map,
            &keyframe,
            left.timestamp(),
            pose_world,
            shared.as_ref(),
        )?;
        self.emit_event(DiagnosticEvent::KeyframeCreated {
            keyframe_id,
            landmarks,
        });
        self.essential_graph.add_keyframe(
            keyframe_id,
            self.map.covisibility().neighbors(keyframe_id),
            &self.map,
        );
        self.bump_map_version();
        self.enqueue_loop_candidates(keyframe_id, keyframe.detections());
        self.enqueue_descriptor_request(keyframe_id, &left);

        let mut diagnostics = self.empty_diagnostics();
        diagnostics.keyframe_created = true;
        diagnostics.triangulation = Some(triangulation_stats);
        diagnostics.features_detected = Some(left_arc.len());
        diagnostics.features_matched = Some(matches.len());
        diagnostics.loop_candidate_count = self
            .pending_loop
            .as_ref()
            .map_or(0, |pending| pending.candidates.len());
        diagnostics.loop_closure_applied = self
            .pending_events
            .iter()
            .any(|event| matches!(event, DiagnosticEvent::LoopClosureDetected { .. }));

        Ok((
            TrackerOutput {
                pose: None,
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

fn insert_keyframe_into_map(
    map: &mut SlamMap,
    keyframe: &Arc<Keyframe>,
    timestamp: Timestamp,
    pose_world: Pose,
    shared: Option<&SharedMatches>,
) -> Result<KeyframeId, TrackerError> {
    let keyframe_id =
        map.add_keyframe_from_detections(keyframe.detections().as_ref(), timestamp, pose_world)?;

    if let Some(shared) = shared {
        for &(current_idx, old_idx) in &shared.pairs {
            let old_kp = map.keyframe_keypoint(shared.keyframe_id, old_idx)?;
            let Some(point_id) = map.map_point_for_keypoint(old_kp)? else {
                continue;
            };
            let new_kp = map.keyframe_keypoint(keyframe_id, current_idx)?;
            if map.map_point_for_keypoint(new_kp)?.is_none() {
                map.add_observation(point_id, new_kp)?;
            }
        }
    }

    // Cull stale singleton landmarks from previous keyframes before adding
    // new landmarks from this keyframe.
    if map.num_points() > 0 {
        let _ = map.cull_points(2);
    }

    for (landmark, &det_idx) in keyframe
        .landmarks()
        .iter()
        .zip(keyframe.landmark_indices().iter())
    {
        let keypoint_ref = map.keyframe_keypoint(keyframe_id, det_idx)?;
        if map.map_point_for_keypoint(keypoint_ref)?.is_some() {
            continue;
        }
        let descriptor = keyframe.detections().descriptors()[det_idx].quantize();
        let world = camera_to_world(pose_world, *landmark);
        map.add_map_point(world, descriptor, keypoint_ref)?;
    }
    Ok(keyframe_id)
}

fn remove_keyframe_from_graph_and_db(
    map: &mut SlamMap,
    essential_graph: &mut EssentialGraph,
    loop_db: Option<&mut KeyframeDatabase>,
    keyframe_id: KeyframeId,
) -> Result<(), TrackerError> {
    essential_graph.remove_keyframe(keyframe_id, map)?;
    if let Some(loop_db) = loop_db {
        loop_db.remove(keyframe_id);
    }
    map.remove_keyframe(keyframe_id)?;
    Ok(())
}

fn camera_to_world(pose_world: Pose, point: Point3) -> Point3 {
    let inv = pose_world.inverse();
    let v = crate::math::transform_point(
        inv.rotation(),
        inv.translation(),
        [point.x, point.y, point.z],
    );
    Point3 {
        x: v[0],
        y: v[1],
        z: v[2],
    }
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
            let keypoint_ref = map
                .keyframe_keypoint(keyframe_id, index)
                .map_err(|_| CorrectionBuildError::MissingKeyframe { keyframe_id })?;
            let Some(point_id) = map.map_point_for_keypoint(keypoint_ref).ok().flatten() else {
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
    current_version: MapVersion,
    correction: &CorrectionEvent,
) -> Result<(), ApplyCorrectionError> {
    if correction.map_version != current_version {
        return Err(ApplyCorrectionError::StaleVersion {
            current: current_version,
            correction: correction.map_version,
        });
    }

    for (keyframe_id, _) in &correction.correction.pose_deltas {
        if map.keyframe(*keyframe_id).is_none() {
            return Err(ApplyCorrectionError::MissingKeyframe {
                keyframe_id: *keyframe_id,
            });
        }
    }
    for (point_id, _) in &correction.correction.landmark_deltas {
        if map.point(*point_id).is_none() {
            return Err(ApplyCorrectionError::MissingMapPoint {
                point_id: *point_id,
            });
        }
    }

    for (keyframe_id, delta) in &correction.correction.pose_deltas {
        let current_pose = map
            .keyframe(*keyframe_id)
            .ok_or(ApplyCorrectionError::MissingKeyframe {
                keyframe_id: *keyframe_id,
            })?
            .pose();
        let corrected = crate::local_ba::apply_se3_delta(current_pose, *delta);
        map.set_keyframe_pose(*keyframe_id, corrected)?;
    }
    for (point_id, delta) in &correction.correction.landmark_deltas {
        let current = map
            .point(*point_id)
            .ok_or(ApplyCorrectionError::MissingMapPoint {
                point_id: *point_id,
            })?
            .position();
        let corrected = Point3 {
            x: current.x + delta[0],
            y: current.y + delta[1],
            z: current.z + delta[2],
        };
        map.set_map_point_position(*point_id, corrected)?;
    }
    Ok(())
}

fn apply_loop_closure_correction(
    map: &mut SlamMap,
    essential_graph: &mut EssentialGraph,
    optimizer: &PoseGraphOptimizer,
    verified: &VerifiedLoop,
) -> Result<(), TrackerError> {
    let query_kf = verified.query_kf();
    let match_kf = verified.match_kf();
    let match_pose = map
        .keyframe(match_kf)
        .ok_or(TrackerError::Map(crate::map::MapError::KeyframeNotFound(
            match_kf,
        )))?
        .pose();
    let query_pose_estimate = verified.relative_pose();
    let loop_relative = crate::Pose64::from_pose32(match_pose)
        .inverse()
        .compose(crate::Pose64::from_pose32(query_pose_estimate));

    essential_graph.add_loop_edge(EssentialEdge {
        a: match_kf,
        b: query_kf,
        kind: EssentialEdgeKind::Loop,
        relative_pose: loop_relative,
        information: loop_information_matrix(verified.inlier_count()),
    });

    let input = essential_graph.pose_graph_input();
    if input.keyframe_ids.len() < 2 || input.edges.is_empty() {
        return Ok(());
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
        old_poses.insert(keyframe_id, pose);
        initial_poses.push(crate::Pose64::from_pose32(pose));
    }

    let result = optimizer.optimize(&input.edges, &mut initial_poses);
    let corrected_poses: HashMap<KeyframeId, Pose> = input
        .keyframe_ids
        .iter()
        .copied()
        .zip(
            result
                .corrected_poses
                .into_iter()
                .map(|pose| pose.to_pose32()),
        )
        .collect();

    for (keyframe_id, corrected_pose) in &corrected_poses {
        map.set_keyframe_pose(*keyframe_id, *corrected_pose)?;
    }

    let mut point_updates = Vec::new();
    for (point_id, point) in map.points() {
        let world = point.position();
        let world_vec = [world.x, world.y, world.z];
        let mut accum = [0.0_f32; 3];
        let mut count = 0usize;

        for observation in point.observations() {
            let keyframe_id = observation.keyframe_id();
            let Some(old_pose) = old_poses.get(&keyframe_id).copied() else {
                continue;
            };
            let Some(new_pose) = corrected_poses.get(&keyframe_id).copied() else {
                continue;
            };

            let camera = crate::math::transform_point(
                old_pose.rotation(),
                old_pose.translation(),
                world_vec,
            );
            let corrected_world = camera_to_world(
                new_pose,
                Point3 {
                    x: camera[0],
                    y: camera[1],
                    z: camera[2],
                },
            );
            accum[0] += corrected_world.x;
            accum[1] += corrected_world.y;
            accum[2] += corrected_world.z;
            count = count.saturating_add(1);
        }

        if count > 0 {
            let inv_count = 1.0_f32 / count as f32;
            point_updates.push((
                point_id,
                Point3 {
                    x: accum[0] * inv_count,
                    y: accum[1] * inv_count,
                    z: accum[2] * inv_count,
                },
            ));
        }
    }

    for (point_id, corrected_world) in point_updates {
        map.set_map_point_position(point_id, corrected_world)?;
    }

    Ok(())
}

fn loop_translation_norm(pose: Pose) -> f32 {
    let t = pose.translation();
    (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt()
}

fn loop_reject_reason(error: &LoopDetectError) -> LoopClosureRejectReason {
    match error {
        LoopDetectError::TooFewCorrespondences { count } => {
            LoopClosureRejectReason::TooFewCorrespondences { count: *count }
        }
        LoopDetectError::VerificationFailed(_) => LoopClosureRejectReason::VerificationFailed,
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

fn build_map_observations(
    map: &SlamMap,
    keyframe_id: KeyframeId,
    matches: &Matches<Verified>,
    current: &Detections,
    intrinsics: PinholeIntrinsics,
) -> Result<Vec<crate::Observation>, crate::PnpError> {
    let mut observations = Vec::with_capacity(matches.len());
    let current_len = current.len();

    for &(ci, ki) in matches.indices() {
        if ci >= current_len {
            return Err(crate::PnpError::IndexOutOfBounds {
                current_len,
                keyframe_len: 0,
                current_index: ci,
                keyframe_index: ki,
            });
        }
        let keypoint_ref = match map.keyframe_keypoint(keyframe_id, ki) {
            Ok(kp) => kp,
            Err(_) => continue,
        };
        let Some(point_id) = map.map_point_for_keypoint(keypoint_ref).ok().flatten() else {
            continue;
        };
        let Some(point) = map.point(point_id) else {
            continue;
        };
        let pixel = current.keypoints()[ci];
        let obs = crate::Observation::try_new(point.position(), pixel, intrinsics)?;
        observations.push(obs);
    }

    if observations.len() < 4 {
        return Err(crate::PnpError::NotEnoughPoints {
            required: 4,
            actual: observations.len(),
        });
    }
    Ok(observations)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::assert_map_invariants;
    use crate::{CompactDescriptor, Descriptor, Detections, Keypoint, Point3, SensorId, Timestamp};
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
    use std::sync::Mutex;
    use std::time::Duration;

    fn make_descriptor() -> Descriptor {
        Descriptor([0.0; 256])
    }

    fn make_test_detections(frame_id: u64) -> Arc<Detections> {
        Arc::new(
            Detections::new(
                SensorId::StereoLeft,
                FrameId::new(frame_id),
                320,
                240,
                vec![Keypoint { x: 100.0, y: 80.0 }],
                vec![1.0],
                vec![make_descriptor()],
            )
            .expect("detections"),
        )
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
            .add_keyframe_from_detections(&detections, Timestamp::from_nanos(1), Pose::identity())
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
                Pose::identity(),
            )
            .expect("kf a");
        let kf_b = map
            .add_keyframe_from_detections(
                &detections_b,
                Timestamp::from_nanos(11),
                Pose::identity(),
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
        let mut event =
            KeyframeEvent::try_new(request_id, MapVersion::initial(), kf_b, window, map)
                .expect("event");
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
            Pose::identity(),
            None,
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
    fn correction_apply_rejects_stale_version() {
        let (mut map, keyframe_id, point_id) = make_map_with_single_point();
        let correction = CorrectionEvent {
            request_id: BackendRequestId(NonZeroU64::new(1).expect("non-zero")),
            map_version: MapVersion::initial(),
            trigger_keyframe: keyframe_id,
            correction: BaCorrection {
                pose_deltas: vec![(keyframe_id, [0.0; 6])],
                landmark_deltas: vec![(point_id, [1.0, 2.0, 3.0])],
                result: BaResult::Converged {
                    iterations: 1,
                    final_cost: 0.0,
                },
            },
        };
        let stale = MapVersion::initial().next();
        assert!(matches!(
            apply_correction_event(&mut map, stale, &correction),
            Err(ApplyCorrectionError::StaleVersion { .. })
        ));
    }

    #[test]
    fn correction_apply_updates_pose_and_landmark_atomically() {
        let (mut map, keyframe_id, point_id) = make_map_with_single_point();
        let corrected_pose = Pose::from_rt(
            [[1.0, 0.0, 0.0], [0.0, 0.999, -0.01], [0.0, 0.01, 0.999]],
            [0.2, -0.1, 0.05],
        );
        let corrected_point = Point3 {
            x: 0.4,
            y: -0.3,
            z: 2.1,
        };
        let initial_pose = map.keyframe(keyframe_id).expect("keyframe").pose();
        let pose_delta = crate::local_ba::se3_delta_between(initial_pose, corrected_pose);
        let correction = CorrectionEvent {
            request_id: BackendRequestId(NonZeroU64::new(2).expect("non-zero")),
            map_version: MapVersion::initial(),
            trigger_keyframe: keyframe_id,
            correction: BaCorrection {
                pose_deltas: vec![(keyframe_id, pose_delta)],
                landmark_deltas: vec![(
                    point_id,
                    [
                        corrected_point.x,
                        corrected_point.y,
                        corrected_point.z - 1.0,
                    ],
                )],
                result: BaResult::Converged {
                    iterations: 2,
                    final_cost: 0.1,
                },
            },
        };

        apply_correction_event(&mut map, MapVersion::initial(), &correction)
            .expect("correction apply");
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
    fn backend_roundtrip_carries_typed_ba_result() {
        let (map, kf_a, kf_b) = make_map_with_two_keyframes_one_shared_point();
        let backend_cfg = BackendConfig::new(1).expect("backend config");
        let intrinsics =
            crate::test_helpers::make_pinhole_intrinsics(320, 240, 200.0, 200.0, 160.0, 120.0)
                .expect("intrinsics");
        let ba_cfg = LocalBaConfig::new(5, 5, 4, 1.0, crate::local_ba::LmConfig::default(), 0.0)
            .expect("ba config");
        let mut worker = BackendWorker::spawn(backend_cfg, intrinsics, ba_cfg);

        let window = BackendWindow::try_new(vec![kf_a, kf_b]).expect("window");
        let event = KeyframeEvent::try_new(
            worker.next_request_id(),
            MapVersion::initial(),
            kf_b,
            window,
            map,
        )
        .expect("event");
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
    fn backend_supervisor_respawns_after_worker_panic() {
        let intrinsics =
            crate::test_helpers::make_pinhole_intrinsics(320, 240, 200.0, 200.0, 160.0, 120.0)
                .expect("intrinsics");
        let mut supervisor = BackendSupervisor::with_max_respawns(
            BackendConfig::new(1).expect("backend config"),
            intrinsics,
            LocalBaConfig::new(5, 5, 4, 1.0, crate::local_ba::LmConfig::default(), 0.0)
                .expect("ba config"),
            3,
        );

        let (map, kf_a, kf_b) = make_map_with_two_keyframes_one_shared_point();
        let mut req_counter = 0;
        let event = make_forced_panic_event(
            BackendRequestId::from_counter(&mut req_counter),
            map,
            kf_a,
            kf_b,
        );
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
    }

    #[test]
    fn backend_supervisor_enforces_max_respawns() {
        let intrinsics =
            crate::test_helpers::make_pinhole_intrinsics(320, 240, 200.0, 200.0, 160.0, 120.0)
                .expect("intrinsics");
        let mut supervisor = BackendSupervisor::with_max_respawns(
            BackendConfig::new(1).expect("backend config"),
            intrinsics,
            LocalBaConfig::new(5, 5, 4, 1.0, crate::local_ba::LmConfig::default(), 0.0)
                .expect("ba config"),
            1,
        );

        let mut req_counter = 0;

        let (map1, kf_a1, kf_b1) = make_map_with_two_keyframes_one_shared_point();
        let panic1 = make_forced_panic_event(
            BackendRequestId::from_counter(&mut req_counter),
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
            BackendRequestId::from_counter(&mut req_counter),
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
            LocalBaConfig::new(5, 5, 4, 1.0, crate::local_ba::LmConfig::default(), 0.0)
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
            LocalBaConfig::new(5, 5, 4, 1.0, crate::local_ba::LmConfig::default(), 0.0)
                .expect("ba config"),
            2,
        );
        let mut req_counter = 0;

        let (map_panic, kf_a, kf_b) = make_map_with_two_keyframes_one_shared_point();
        let panic_event = make_forced_panic_event(
            BackendRequestId::from_counter(&mut req_counter),
            map_panic,
            kf_a,
            kf_b,
        );
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
        let ok_event = KeyframeEvent::try_new(
            BackendRequestId::from_counter(&mut req_counter),
            MapVersion::initial(),
            kf_b2,
            window,
            map_ok,
        )
        .expect("event");
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
        );

        let frame = Frame::new(
            SensorId::StereoLeft,
            FrameId::new(77),
            Timestamp::from_nanos(77),
            16,
            12,
            vec![128_u8; 16 * 12],
        )
        .expect("frame");
        worker
            .submit(DescriptorRequest {
                keyframe_id: KeyframeId::default(),
                map_version: MapVersion::initial(),
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
        assert_eq!(response.map_version, MapVersion::initial());
        assert_eq!(response.descriptor, descriptor);
        assert_eq!(*calls.lock().expect("calls lock"), 1);
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
                    Some(Box::new(PanicDescriptorExtractor) as Box<dyn PlaceDescriptorExtractor>)
                } else {
                    Some(Box::new(StubDescriptorExtractor {
                        descriptor: descriptor.clone(),
                        calls: Arc::clone(&calls),
                    }) as Box<dyn PlaceDescriptorExtractor>)
                }
            })
        };

        let mut supervisor =
            DescriptorSupervisor::with_factory_and_max_respawns(config, factory, 2);
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
                map_version: MapVersion::initial(),
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
                map_version: MapVersion::initial(),
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
                Pose::identity(),
                image_size,
                keypoints.clone(),
            )
            .expect("kf0");
        let kf1 = map
            .add_keyframe(
                FrameId::new(101),
                Timestamp::from_nanos(101),
                Pose::from_rt(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    [1.0, 0.0, 0.0],
                ),
                image_size,
                keypoints.clone(),
            )
            .expect("kf1");
        let kf2 = map
            .add_keyframe(
                FrameId::new(102),
                Timestamp::from_nanos(102),
                Pose::from_rt(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    [2.4, 0.2, 0.0],
                ),
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
        essential_graph.add_keyframe(kf0, map.covisibility().neighbors(kf0), &map);
        essential_graph.add_keyframe(kf1, map.covisibility().neighbors(kf1), &map);
        essential_graph.add_keyframe(kf2, map.covisibility().neighbors(kf2), &map);

        let verified = crate::loop_closure::VerifiedLoop::from_parts(
            kf2,
            kf0,
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
        assert_eq!(snapshot.loop_edges[0].kind, EssentialEdgeKind::Loop);
    }

    #[test]
    fn remove_keyframe_from_graph_and_db_cleans_all_structures() {
        let (mut map, mut essential_graph, _verified, removed_kf, _before_points) =
            make_loop_closure_apply_fixture();
        let mut loop_db = KeyframeDatabase::new(0);
        for (idx, (keyframe_id, _)) in map.keyframes().enumerate() {
            loop_db.insert_with_source(
                keyframe_id,
                make_global_descriptor_basis(idx),
                crate::loop_closure::DescriptorSource::Bootstrap,
            );
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
        let input = essential_graph.pose_graph_input();
        assert!(input.keyframe_ids.iter().all(|&id| id != removed_kf));
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
        assert!(nominal.backend_alive);
        assert!(nominal.descriptor_alive);
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
        assert!(!descriptor_down.descriptor_alive);

        let backend_down =
            SystemHealth::from_components(TrackingHealth::Good, true, false, true, true, stats);
        assert_eq!(backend_down.degradation, DegradationLevel::BackendDown);
        assert!(!backend_down.backend_alive);

        let lost =
            SystemHealth::from_components(TrackingHealth::Lost, true, false, true, false, stats);
        assert_eq!(lost.degradation, DegradationLevel::Lost);

        let backend_optional =
            SystemHealth::from_components(TrackingHealth::Good, false, true, false, true, stats);
        assert_eq!(backend_optional.degradation, DegradationLevel::Nominal);
    }

    #[test]
    fn relocalization_initial_session_requires_lost_tracking_and_enabled_config() {
        let detections = make_test_detections(900);
        assert!(SlamTracker::initial_relocalization_session(
            TrackingHealth::Good,
            true,
            Arc::clone(&detections)
        )
        .is_none());
        assert!(SlamTracker::initial_relocalization_session(
            TrackingHealth::Lost,
            false,
            Arc::clone(&detections)
        )
        .is_none());

        let session = SlamTracker::initial_relocalization_session(
            TrackingHealth::Lost,
            true,
            Arc::clone(&detections),
        )
        .expect("lost tracking should create relocalization session");
        assert_eq!(session.attempts, 0);
        assert!(matches!(session.phase, RelocalizationPhase::Searching));
    }

    #[test]
    fn relocalization_failure_transitions_respect_max_attempts() {
        let cfg = RelocalizationConfig::new(crate::loop_closure::RelocalizationConfigInput {
            max_attempts: 2,
            ..crate::loop_closure::RelocalizationConfigInput::default()
        })
        .expect("relocalization config");
        let detections = make_test_detections(901);

        let keep_trying = SlamTracker::next_state_after_relocalization_failure(
            cfg,
            RelocalizationSession {
                attempts: 0,
                phase: RelocalizationPhase::Searching,
                last_detections: Arc::clone(&detections),
            },
            Arc::clone(&detections),
        );
        assert!(matches!(keep_trying, TrackerState::Relocalizing(_)));
        let TrackerState::Relocalizing(updated) = keep_trying else {
            panic!("expected relocalizing state")
        };
        assert_eq!(updated.attempts, 1);
        assert!(matches!(updated.phase, RelocalizationPhase::Searching));

        let give_up = SlamTracker::next_state_after_relocalization_failure(
            cfg,
            RelocalizationSession {
                attempts: 1,
                phase: RelocalizationPhase::Searching,
                last_detections: Arc::clone(&detections),
            },
            detections,
        );
        assert!(matches!(give_up, TrackerState::NeedKeyframe));
    }

    #[test]
    fn relocalization_step_requires_confirmation_before_recovery() {
        let cfg = RelocalizationConfig::default();
        let candidate = KeyframeId::default();
        let detections = make_test_detections(902);
        let pose = Pose::identity();

        let step = SlamTracker::relocalization_step(
            RelocalizationSession {
                attempts: 0,
                phase: RelocalizationPhase::Searching,
                last_detections: detections,
            },
            candidate,
            pose,
            cfg,
        );
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
    }

    #[test]
    fn relocalization_step_recovers_after_consistent_confirmation() {
        let cfg = RelocalizationConfig::default();
        let candidate = KeyframeId::default();
        let detections = make_test_detections(903);
        let pose = Pose::identity();

        let step = SlamTracker::relocalization_step(
            RelocalizationSession {
                attempts: 2,
                phase: RelocalizationPhase::Confirming {
                    candidate,
                    confirmations: NonZeroUsize::new(1).expect("non-zero"),
                    pose_world: pose,
                },
                last_detections: detections,
            },
            candidate,
            pose,
            cfg,
        );

        assert!(matches!(step, RelocalizationStep::Recovered { .. }));
    }

    #[test]
    fn relocalization_pose_consistency_enforces_translation_and_rotation_limits() {
        let cfg = RelocalizationConfig::default();
        let identity = Pose::identity();

        let within_translation = Pose::from_rt(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [cfg.max_translation_delta_m() * 0.5, 0.0, 0.0],
        );
        assert!(SlamTracker::relocalization_pose_consistent(
            identity,
            within_translation,
            cfg
        ));

        let beyond_translation = Pose::from_rt(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [cfg.max_translation_delta_m() * 1.5, 0.0, 0.0],
        );
        assert!(!SlamTracker::relocalization_pose_consistent(
            identity,
            beyond_translation,
            cfg
        ));

        let half_angle = (cfg.max_rotation_delta_deg() * 0.5).to_radians();
        let within_rotation = Pose::from_rt(
            [
                [half_angle.cos(), -half_angle.sin(), 0.0],
                [half_angle.sin(), half_angle.cos(), 0.0],
                [0.0, 0.0, 1.0],
            ],
            [0.0, 0.0, 0.0],
        );
        assert!(SlamTracker::relocalization_pose_consistent(
            identity,
            within_rotation,
            cfg
        ));

        let over_angle = (cfg.max_rotation_delta_deg() * 1.5).to_radians();
        let beyond_rotation = Pose::from_rt(
            [
                [over_angle.cos(), -over_angle.sin(), 0.0],
                [over_angle.sin(), over_angle.cos(), 0.0],
                [0.0, 0.0, 1.0],
            ],
            [0.0, 0.0, 0.0],
        );
        assert!(!SlamTracker::relocalization_pose_consistent(
            identity,
            beyond_rotation,
            cfg
        ));
    }
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
