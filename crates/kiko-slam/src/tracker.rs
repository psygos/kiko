use std::collections::HashSet;
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
const DEFAULT_CULL_MIN_OBSERVATIONS: usize = 1;

use crate::frontend::{StereoFrontend, median_parallax_px};
use crate::global_map::GlobalMap;
use crate::loop_closure::{
    LoopApplyError, LoopCandidate, LoopClosureConfig, LoopDetectError, RelocalizationCandidate,
    RelocalizationConfig, VerifiedLoop, aggregate_global_descriptor,
    match_quantized_descriptors_for_loop,
};
use crate::loop_manager::LoopManager;
use crate::place_recognition::{
    DescriptorStats, PlaceRecognition, PlaceRecognitionEvent, PlaceRecognitionInitError,
};
#[cfg(feature = "vio")]
use crate::pose_graph::{EssentialEdge, EssentialEdgeKind};
use crate::pose_graph::{EssentialGraphError, PoseGraphConfig, PoseGraphError};
use crate::{
    BaCorrection, BaResult, CalibrationBundle, CaptureBundle, CaptureBundleError, CaptureId,
    Detections, DiagnosticEvent, DownscaleFactor, Frame, FrameDiagnostics, FrameId, Keyframe,
    KeyframeRemovalReason, KeyframeStatus, KeypointLimit, LightGlue, LocalBaConfig,
    LocalBundleAdjuster, LoopClosureStatus, MapFromOdom, MapObservation, Matches, Observation,
    ObservationSet, PinholeIntrinsics, Point3, Pose, Pose64, RansacConfig, Raw, StereoPair,
    SuperPoint, Timestamp, TriangulationConfig, TriangulationError, Triangulator, Verified,
    map::{KeyframeId, MapPointId, SlamMap},
};
#[cfg(feature = "vio")]
use crate::{Gravity, ImuAccumulator, LocalVio, PreintegratedImu, VioConfig, VioObservation};

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
    #[cfg(feature = "vio")]
    pub vio: Option<VioConfig>,
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
    LoopClosureOnly {
        loop_closure: LoopClosureConfig,
        global_descriptor: GlobalDescriptorConfig,
    },
    LoopClosureAndRelocalization {
        loop_closure: LoopClosureConfig,
        global_descriptor: GlobalDescriptorConfig,
        relocalization: RelocalizationConfig,
    },
}

impl LoopSubsystemConfig {
    pub fn loop_closure_only(
        loop_closure: LoopClosureConfig,
        global_descriptor: GlobalDescriptorConfig,
    ) -> Self {
        Self::LoopClosureOnly {
            loop_closure,
            global_descriptor,
        }
    }

    pub fn with_relocalization(
        loop_closure: LoopClosureConfig,
        global_descriptor: GlobalDescriptorConfig,
        relocalization: RelocalizationConfig,
    ) -> Self {
        Self::LoopClosureAndRelocalization {
            loop_closure,
            global_descriptor,
            relocalization,
        }
    }

    pub fn loop_closure(self) -> Option<LoopClosureConfig> {
        match self {
            Self::Disabled => None,
            Self::LoopClosureOnly { loop_closure, .. }
            | Self::LoopClosureAndRelocalization { loop_closure, .. } => Some(loop_closure),
        }
    }

    pub fn global_descriptor(self) -> Option<GlobalDescriptorConfig> {
        match self {
            Self::Disabled => None,
            Self::LoopClosureOnly {
                global_descriptor, ..
            }
            | Self::LoopClosureAndRelocalization {
                global_descriptor, ..
            } => Some(global_descriptor),
        }
    }

    pub fn relocalization(self) -> Option<RelocalizationConfig> {
        match self {
            Self::Disabled => None,
            Self::LoopClosureOnly { .. } => None,
            Self::LoopClosureAndRelocalization { relocalization, .. } => Some(relocalization),
        }
    }

    pub fn is_enabled(self) -> bool {
        !matches!(self, Self::Disabled)
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum KeyframeInsertReason {
    TooFewInliers { inliers: usize, min_required: usize },
    HighParallax { parallax_px: f32, threshold_px: f32 },
    LowCovisibility { covisibility: f32, threshold: f32 },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum KeyframeDecision {
    KeepTracking,
    Insert(KeyframeInsertReason),
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
pub(crate) struct MapVersion(NonZeroU64);

impl MapVersion {
    fn initial() -> Self {
        Self(NonZeroU64::MIN)
    }

    fn next(self) -> Self {
        let next = self.0.get().saturating_add(1).max(1);
        Self(NonZeroU64::new(next).unwrap_or(NonZeroU64::MIN))
    }

    pub(crate) fn as_u64(self) -> u64 {
        self.0.get()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct BackendRequestId(NonZeroU64);

impl BackendRequestId {
    fn from_counter(counter: &mut u64) -> Self {
        *counter = counter.saturating_add(1).max(1);
        Self(NonZeroU64::new(*counter).unwrap_or(NonZeroU64::MIN))
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
                                    map_version,
                                    error: BackendWorkerError::BuildCorrection(err),
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
                                    map_version,
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
            next_request_id: 0,
        })
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
    spawn_exhausted: bool,
}

impl BackendSupervisor {
    fn spawn_worker(
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
    ) -> Self {
        let worker = Self::spawn_worker(config, intrinsics, ba_config);
        let spawn_exhausted = worker.is_none() && max_respawns == 0;
        if worker.is_none() {
            eprintln!("backend worker unavailable at startup; backend optimization disabled");
        }
        Self {
            worker,
            config,
            intrinsics,
            ba_config,
            respawn_count: 0,
            max_respawns,
            spawn_exhausted,
        }
    }

    #[cfg(test)]
    fn with_max_respawns(
        config: BackendConfig,
        intrinsics: PinholeIntrinsics,
        ba_config: LocalBaConfig,
        max_respawns: u32,
    ) -> Self {
        Self::spawn_with_max_respawns(config, intrinsics, ba_config, max_respawns)
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
        self.worker = Self::spawn_worker(self.config, self.intrinsics, self.ba_config);
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
                map_version,
            })) => {
                self.worker = None;
                self.check_health();
                Some(BackendResponse::WorkerPanic {
                    request_id,
                    map_version,
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

    fn next_request_id(&mut self) -> Option<BackendRequestId> {
        if self.worker.is_none() {
            self.check_health();
        }
        self.worker.as_mut().map(BackendWorker::next_request_id)
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

    pub fn decide(
        &self,
        inliers: usize,
        parallax_px: Option<f32>,
        covisibility: f32,
    ) -> KeyframeDecision {
        if inliers < self.min_inliers.get() {
            return KeyframeDecision::Insert(KeyframeInsertReason::TooFewInliers {
                inliers,
                min_required: self.min_inliers.get(),
            });
        }
        if let Some(parallax) = parallax_px {
            if parallax > self.parallax_px.0 {
                return KeyframeDecision::Insert(KeyframeInsertReason::HighParallax {
                    parallax_px: parallax,
                    threshold_px: self.parallax_px.0,
                });
            }
        }
        if covisibility < self.min_covisibility.0 {
            return KeyframeDecision::Insert(KeyframeInsertReason::LowCovisibility {
                covisibility,
                threshold: self.min_covisibility.0,
            });
        }
        KeyframeDecision::KeepTracking
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
pub enum TrackerInitError {
    DescriptorUnavailable {
        model_path: PathBuf,
    },
    #[cfg(feature = "vio")]
    VioInvalidGravity {
        message: String,
    },
}

impl std::fmt::Display for TrackerInitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrackerInitError::DescriptorUnavailable { model_path } => write!(
                f,
                "loop closure requires learned descriptors but descriptor worker failed to start (model: {})",
                model_path.display()
            ),
            #[cfg(feature = "vio")]
            TrackerInitError::VioInvalidGravity { message } => {
                write!(f, "invalid vio gravity configuration: {message}")
            }
        }
    }
}

impl std::error::Error for TrackerInitError {}

#[derive(Debug)]
pub enum TrackerError {
    Capture(CaptureBundleError),
    #[cfg(feature = "vio")]
    Vio(String),
    Inference(InferenceError),
    Triangulation(TriangulationError),
    Pnp(crate::PnpError),
    Map(crate::map::MapError),
    EssentialGraph(EssentialGraphError),
    PoseGraph(PoseGraphError),
    KeyframeRejected {
        landmarks: usize,
    },
    InvariantViolation(&'static str),
}

impl std::fmt::Display for TrackerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrackerError::Capture(err) => write!(f, "capture error: {err}"),
            #[cfg(feature = "vio")]
            TrackerError::Vio(err) => write!(f, "vio error: {err}"),
            TrackerError::Inference(err) => write!(f, "inference error: {err}"),
            TrackerError::Triangulation(err) => write!(f, "triangulation error: {err}"),
            TrackerError::Pnp(err) => write!(f, "pnp error: {err}"),
            TrackerError::Map(err) => write!(f, "map error: {err}"),
            TrackerError::EssentialGraph(err) => write!(f, "essential graph error: {err}"),
            TrackerError::PoseGraph(err) => write!(f, "pose graph error: {err}"),
            TrackerError::KeyframeRejected { landmarks } => {
                write!(f, "keyframe rejected: only {landmarks} landmarks")
            }
            TrackerError::InvariantViolation(message) => {
                write!(f, "tracker invariant violation: {message}")
            }
        }
    }
}

impl std::error::Error for TrackerError {}

impl From<CaptureBundleError> for TrackerError {
    fn from(err: CaptureBundleError) -> Self {
        TrackerError::Capture(err)
    }
}

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

impl From<PoseGraphError> for TrackerError {
    fn from(err: PoseGraphError) -> Self {
        TrackerError::PoseGraph(err)
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
pub struct TrackingPose {
    cam_from_odom: Pose64,
    cam_from_map_corrected: Pose64,
    cam_from_map_visual_measurement: Option<Pose64>,
}

impl TrackingPose {
    pub fn new(
        cam_from_odom: Pose64,
        cam_from_map_corrected: Pose64,
        cam_from_map_visual_measurement: Option<Pose64>,
    ) -> Self {
        Self {
            cam_from_odom,
            cam_from_map_corrected,
            cam_from_map_visual_measurement,
        }
    }

    pub fn cam_from_odom(&self) -> Pose64 {
        self.cam_from_odom
    }

    pub fn cam_from_map(&self) -> Pose64 {
        self.cam_from_map_corrected
    }

    pub fn cam_from_map_visual_measurement(&self) -> Option<Pose64> {
        self.cam_from_map_visual_measurement
    }

    pub fn cam_from_odom_pose32(&self) -> Pose {
        self.cam_from_odom.to_pose32()
    }

    pub fn cam_from_map_pose32(&self) -> Pose {
        self.cam_from_map_corrected.to_pose32()
    }

    pub fn cam_from_map_visual_measurement_pose32(&self) -> Option<Pose> {
        self.cam_from_map_visual_measurement.map(Pose64::to_pose32)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VioTelemetry {
    velocity_odom_mps: [f64; 3],
    accel_bias_mps2: [f64; 3],
    gyro_bias_radps: [f64; 3],
}

impl VioTelemetry {
    fn from_nav_state(state: &crate::NavState) -> Self {
        let bias = state.bias();
        Self {
            velocity_odom_mps: state.velocity_odom_mps(),
            accel_bias_mps2: bias.accel,
            gyro_bias_radps: bias.gyro,
        }
    }

    pub fn velocity_odom_mps(&self) -> [f64; 3] {
        self.velocity_odom_mps
    }

    pub fn accel_bias_mps2(&self) -> [f64; 3] {
        self.accel_bias_mps2
    }

    pub fn gyro_bias_radps(&self) -> [f64; 3] {
        self.gyro_bias_radps
    }
}

#[derive(Debug)]
pub struct TrackerOutput {
    pub pose: Option<TrackingPose>,
    pub inliers: usize,
    pub keyframe: Option<Arc<Keyframe>>,
    pub stereo_matches: Option<Matches<Raw>>,
    pub frame_id: FrameId,
    pub health: SystemHealth,
    pub diagnostics: FrameDiagnostics,
    pub events: Vec<DiagnosticEvent>,
    pub vio_telemetry: Option<VioTelemetry>,
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

#[cfg(feature = "vio")]
enum LocalEstimator {
    VisualOnly,
    Inertial(Box<VioRuntime>),
}

#[cfg(feature = "vio")]
struct VioRuntime {
    local_vio: LocalVio,
    noise: crate::ImuNoiseModel,
    pending_imu: ImuAccumulator,
    predicted_preintegration: Option<PreintegratedImu>,
    predicted_state: Option<crate::NavState>,
}

pub struct SlamTracker {
    frontend: StereoFrontend,
    config: TrackerConfig,
    state: TrackerState,
    #[cfg(feature = "vio")]
    local_estimator: LocalEstimator,
    ba: LocalBundleAdjuster,
    global_map: GlobalMap,
    loop_manager: LoopManager,
    map_version: MapVersion,
    backend: Option<BackendSupervisor>,
    backend_stats: BackendStats,
    place_recognition: Option<PlaceRecognition>,
    pending_events: Vec<DiagnosticEvent>,
    tracking_health: TrackingHealth,
    consecutive_tracking_failures: usize,
    last_pose_world: Option<Pose>,
    next_capture_id: u64,
    previous_capture_time: Option<Timestamp>,
    map_from_odom: MapFromOdom,
    trace_transitions: bool,
}

impl SlamTracker {
    const DEFAULT_ESSENTIAL_GRAPH_STRONG_THRESHOLD: u32 = 15;

    pub fn try_new(
        superpoint: SuperPoint,
        lightglue: LightGlue,
        calibration: CalibrationBundle,
        config: TrackerConfig,
    ) -> Result<Self, TrackerInitError> {
        let stereo = calibration.stereo().clone();
        let intrinsics = calibration.intrinsics();
        let triangulator = Triangulator::new(stereo, config.triangulation);
        let frontend = StereoFrontend::new(superpoint, lightglue, triangulator, intrinsics);
        let ba = LocalBundleAdjuster::new(intrinsics, config.ba);
        #[cfg(feature = "vio")]
        let local_estimator = match (config.vio, calibration.imu_noise(), calibration.has_imu()) {
            (Some(vio_config), Some(noise), true) => {
                let gravity = Gravity::try_new([0.0, 0.0, -calibration.gravity_magnitude_mps2()])
                    .map_err(|err| TrackerInitError::VioInvalidGravity {
                    message: err.to_string(),
                })?;
                let camera_from_body = calibration
                    .imu_extrinsics()
                    .map(|extrinsics| extrinsics.t_cam_imu())
                    .unwrap_or_else(Pose64::identity);
                LocalEstimator::Inertial(Box::new(VioRuntime {
                    local_vio: LocalVio::new(vio_config, gravity, camera_from_body, intrinsics),
                    noise: noise.clone(),
                    pending_imu: ImuAccumulator::new(),
                    predicted_preintegration: None,
                    predicted_state: None,
                }))
            }
            _ => LocalEstimator::VisualOnly,
        };
        let backend_max_respawns = crate::env::env_usize("KIKO_BACKEND_MAX_RESPAWNS")
            .and_then(|value| u32::try_from(value).ok())
            .unwrap_or(DEFAULT_MAX_RESPAWNS);
        let backend = config.backend.map(|backend_cfg| {
            BackendSupervisor::spawn_with_max_respawns(
                backend_cfg,
                intrinsics,
                config.ba,
                backend_max_respawns,
            )
        });
        let loop_config = config.loop_closure_config();
        let descriptor_max_respawns = crate::env::env_usize("KIKO_DESCRIPTOR_MAX_RESPAWNS")
            .and_then(|value| u32::try_from(value).ok())
            .unwrap_or(DEFAULT_MAX_RESPAWNS);
        let place_recognition = match (loop_config, config.global_descriptor_config()) {
            (Some(loop_config), Some(descriptor_config)) => Some(
                PlaceRecognition::new(loop_config, descriptor_config, descriptor_max_respawns)
                    .map_err(|err| match err {
                        PlaceRecognitionInitError::DescriptorUnavailable { model_path } => {
                            TrackerInitError::DescriptorUnavailable { model_path }
                        }
                    })?,
            ),
            _ => None,
        };
        let trace_transitions = crate::env::env_bool("KIKO_TRACK_TRACE").unwrap_or(false);
        Ok(Self {
            frontend,
            config,
            state: TrackerState::NeedKeyframe,
            #[cfg(feature = "vio")]
            local_estimator,
            ba,
            global_map: GlobalMap::new(Self::DEFAULT_ESSENTIAL_GRAPH_STRONG_THRESHOLD),
            loop_manager: LoopManager::new(PoseGraphConfig::default()),
            map_version: MapVersion::initial(),
            backend,
            backend_stats: BackendStats::default(),
            place_recognition,
            pending_events: Vec::new(),
            tracking_health: TrackingHealth::Good,
            consecutive_tracking_failures: 0,
            last_pose_world: None,
            next_capture_id: 0,
            previous_capture_time: None,
            map_from_odom: MapFromOdom::identity(),
            trace_transitions,
        })
    }

    pub fn process(&mut self, pair: StereoPair) -> Result<TrackerOutput, TrackerError> {
        let capture_id = CaptureId::new(self.next_capture_id);
        self.next_capture_id = self.next_capture_id.saturating_add(1);
        let capture = CaptureBundle::visual_only(capture_id, pair, self.previous_capture_time)?;
        self.process_capture(capture)
    }

    pub fn process_capture(
        &mut self,
        capture: CaptureBundle,
    ) -> Result<TrackerOutput, TrackerError> {
        #[cfg(feature = "vio")]
        {
            if let Some(batch) = capture.imu().batch() {
                self.ingest_imu_batch(batch)?;
            }
            self.refresh_predicted_pose_from_vio()?;
        }
        self.previous_capture_time = Some(capture.capture_time());
        let (_, pair, _, _) = capture.into_parts();
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
        };
        if result.is_err() {
            self.clear_events();
        }
        result
    }

    #[cfg(feature = "vio")]
    fn ingest_imu_batch(&mut self, batch: &crate::ImuBatch) -> Result<(), TrackerError> {
        let LocalEstimator::Inertial(vio_runtime) = &mut self.local_estimator else {
            return Ok(());
        };
        vio_runtime
            .pending_imu
            .extend_batch(batch)
            .map_err(|err| TrackerError::Vio(err.to_string()))
    }

    #[cfg(feature = "vio")]
    fn refresh_predicted_pose_from_vio(&mut self) -> Result<(), TrackerError> {
        let LocalEstimator::Inertial(vio_runtime) = &mut self.local_estimator else {
            return Ok(());
        };
        vio_runtime
            .local_vio
            .set_map_from_odom(self.map_from_odom.clone());
        let Some(batch) = vio_runtime
            .pending_imu
            .batch()
            .map_err(|err| TrackerError::Vio(err.to_string()))?
        else {
            vio_runtime.predicted_preintegration = None;
            return Ok(());
        };
        if batch.len() < 2 {
            vio_runtime.predicted_preintegration = None;
            return Ok(());
        }
        let Some(latest) = vio_runtime.local_vio.latest_estimate() else {
            return Ok(());
        };
        let preintegrated =
            PreintegratedImu::integrate(&batch, latest.state().bias(), &vio_runtime.noise)
                .map_err(|err| TrackerError::Vio(err.to_string()))?;
        let predicted = vio_runtime
            .local_vio
            .predict_from_latest(&preintegrated)
            .map_err(|err| TrackerError::Vio(err.to_string()))?;
        vio_runtime.predicted_preintegration = Some(preintegrated);
        vio_runtime.predicted_state = Some(predicted.state().clone());
        self.last_pose_world = Some(
            self.map_from_odom
                .odom_to_map(predicted.state().pose_odom_from_body())
                .to_pose32(),
        );
        Ok(())
    }

    #[cfg(feature = "vio")]
    fn on_keyframe_for_vio(
        &mut self,
        keyframe_id: KeyframeId,
        pose_world: Pose,
        visual_observations: Vec<Observation>,
    ) -> Result<(), TrackerError> {
        let visual_observations = self.map_observations_for_vio(visual_observations)?;
        let LocalEstimator::Inertial(vio_runtime) = &mut self.local_estimator else {
            return Ok(());
        };
        vio_runtime
            .local_vio
            .set_map_from_odom(self.map_from_odom.clone());
        if vio_runtime.local_vio.is_empty() {
            let pose_measurement_odom = self
                .map_from_odom
                .map_to_odom(Pose64::from_pose32(pose_world));
            let state = crate::NavState::try_new(
                pose_measurement_odom,
                [0.0; 3],
                crate::ImuBias::default(),
            )
            .map_err(|err| TrackerError::Vio(err.to_string()))?;
            let predicted_state = state.clone();
            vio_runtime
                .local_vio
                .initialize(
                    keyframe_id,
                    state,
                    pose_measurement_odom,
                    visual_observations,
                )
                .map_err(|err| TrackerError::Vio(err.to_string()))?;
            vio_runtime.predicted_state = Some(predicted_state);
            vio_runtime.predicted_preintegration = None;
            vio_runtime.pending_imu.clear();
            return Ok(());
        }

        let Some(batch) = vio_runtime
            .pending_imu
            .drain_batch()
            .map_err(|err| TrackerError::Vio(err.to_string()))?
        else {
            return Ok(());
        };
        if batch.len() < 2 {
            return Ok(());
        }

        let latest = vio_runtime
            .local_vio
            .latest_estimate()
            .ok_or_else(|| TrackerError::Vio("local vio lost latest state".to_string()))?;
        let preintegrated =
            PreintegratedImu::integrate(&batch, latest.state().bias(), &vio_runtime.noise)
                .map_err(|err| TrackerError::Vio(err.to_string()))?;
        let estimate = vio_runtime
            .local_vio
            .push_preintegrated(
                keyframe_id,
                preintegrated,
                self.map_from_odom
                    .map_to_odom(Pose64::from_pose32(pose_world)),
                visual_observations,
            )
            .map_err(|err| TrackerError::Vio(err.to_string()))?;
        vio_runtime.predicted_preintegration = None;
        vio_runtime.predicted_state = Some(estimate.state().clone());
        self.last_pose_world = Some(
            self.map_from_odom
                .odom_to_map(estimate.state().pose_odom_from_body())
                .to_pose32(),
        );
        for odometry in vio_runtime.local_vio.drain_exported_odometry() {
            self.global_map.add_odometry_edge(EssentialEdge {
                a: odometry.from(),
                b: odometry.to(),
                kind: EssentialEdgeKind::Odometry,
                relative_pose: odometry.relative_pose(),
                information: odometry.information(),
            });
        }
        Ok(())
    }

    pub fn covisibility_snapshot(&self) -> crate::map::CovisibilitySnapshot {
        self.global_map.covisibility_snapshot()
    }

    pub fn backend_stats(&self) -> BackendStats {
        self.backend_stats
    }

    pub fn descriptor_stats(&self) -> DescriptorStats {
        self.place_recognition
            .as_ref()
            .map_or_else(DescriptorStats::default, PlaceRecognition::descriptor_stats)
    }

    pub fn system_health(&self) -> SystemHealth {
        let backend_expected = self.backend.is_some();
        let backend_alive = self
            .backend
            .as_ref()
            .is_none_or(BackendSupervisor::has_worker);
        let descriptor_expected = self.place_recognition.is_some();
        let descriptor_alive = self
            .place_recognition
            .as_ref()
            .is_none_or(PlaceRecognition::has_worker);
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
        #[cfg(feature = "vio")]
        let corrected = self
            .loop_manager
            .apply_verified_loop(&mut self.global_map, &verified)?;
        #[cfg(not(feature = "vio"))]
        self.loop_manager
            .apply_verified_loop(&mut self.global_map, &verified)?;
        #[cfg(feature = "vio")]
        self.realign_map_from_odom(&corrected);
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

    #[cfg(feature = "vio")]
    fn map_observations_for_vio(
        &self,
        observations_map: Vec<Observation>,
    ) -> Result<Vec<VioObservation>, TrackerError> {
        observations_map
            .into_iter()
            .map(|observation_map| {
                VioObservation::from_map_observation(observation_map)
                    .map_err(|err| TrackerError::Vio(err.to_string()))
            })
            .collect()
    }

    #[cfg(feature = "vio")]
    fn align_map_from_odom_to_pose(&mut self, pose_map: Pose) {
        let Some(cam_from_odom) = self.current_odom_pose() else {
            return;
        };
        self.map_from_odom
            .align_to_pose(Pose64::from_pose32(pose_map), cam_from_odom);
        if let LocalEstimator::Inertial(vio_runtime) = &mut self.local_estimator {
            vio_runtime
                .local_vio
                .set_map_from_odom(self.map_from_odom.clone());
        }
    }

    #[cfg(feature = "vio")]
    fn realign_map_from_odom(&mut self, corrected_poses: &[(KeyframeId, Pose)]) {
        let LocalEstimator::Inertial(vio_runtime) = &mut self.local_estimator else {
            return;
        };
        for (keyframe_id, corrected_pose_map) in corrected_poses.iter().rev() {
            let Some(estimate) = vio_runtime.local_vio.estimate_for(*keyframe_id) else {
                continue;
            };
            self.map_from_odom.align_to_pose(
                Pose64::from_pose32(*corrected_pose_map),
                estimate.state().pose_odom_from_body(),
            );
            vio_runtime
                .local_vio
                .set_map_from_odom(self.map_from_odom.clone());
            break;
        }
    }

    #[cfg(feature = "vio")]
    fn current_odom_pose(&self) -> Option<Pose64> {
        match &self.local_estimator {
            LocalEstimator::VisualOnly => None,
            LocalEstimator::Inertial(vio_runtime) => vio_runtime
                .predicted_state
                .as_ref()
                .map(crate::NavState::pose_odom_from_body),
        }
    }

    #[cfg(feature = "vio")]
    fn current_vio_telemetry(&self) -> Option<VioTelemetry> {
        match &self.local_estimator {
            LocalEstimator::VisualOnly => None,
            LocalEstimator::Inertial(vio_runtime) => {
                vio_runtime.predicted_state.as_ref().map(VioTelemetry::from_nav_state)
            }
        }
    }

    #[cfg(feature = "vio")]
    fn correct_tracking_pose_from_vio(
        &mut self,
        pose_map: Pose,
        inlier_observations_map: Vec<Observation>,
    ) -> Result<(), TrackerError> {
        let visual_observations = self.map_observations_for_vio(inlier_observations_map)?;
        let LocalEstimator::Inertial(vio_runtime) = &mut self.local_estimator else {
            return Ok(());
        };
        vio_runtime
            .local_vio
            .set_map_from_odom(self.map_from_odom.clone());
        let Some(preintegrated) = vio_runtime.predicted_preintegration.as_ref() else {
            return Ok(());
        };
        let estimate = vio_runtime
            .local_vio
            .correct_prediction(
                preintegrated,
                self.map_from_odom
                    .map_to_odom(Pose64::from_pose32(pose_map)),
                visual_observations,
            )
            .map_err(|err| TrackerError::Vio(err.to_string()))?;
        vio_runtime.predicted_state = Some(estimate.state().clone());
        Ok(())
    }

    fn drain_events(&mut self) -> Vec<DiagnosticEvent> {
        std::mem::take(&mut self.pending_events)
    }

    fn clear_events(&mut self) {
        self.pending_events.clear();
    }

    fn empty_diagnostics(&self) -> FrameDiagnostics {
        let mut diagnostics = FrameDiagnostics::empty(
            self.global_map.num_keyframes(),
            self.global_map.num_points(),
        );
        diagnostics.loop_candidate_count = self
            .place_recognition
            .as_ref()
            .map_or(0, PlaceRecognition::pending_candidate_count);
        diagnostics.loop_closure_status = self.pending_events.iter().find_map(|event| {
            matches!(event, DiagnosticEvent::LoopClosureDetected { .. })
                .then_some(LoopClosureStatus::Applied)
        });
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
        #[cfg(feature = "vio")]
        let pose = match (self.current_odom_pose(), pose) {
            (Some(cam_from_odom), maybe_pose_map) => Some(tracking_pose_from_vio_output(
                &self.map_from_odom,
                cam_from_odom,
                maybe_pose_map,
            )),
            (None, maybe_pose) => maybe_pose.map(|pose_world| {
                let cam_from_odom = Pose64::from_pose32(pose_world);
                let cam_from_map = self.map_from_odom.odom_to_map(cam_from_odom);
                TrackingPose::new(cam_from_odom, cam_from_map, None)
            }),
        };
        #[cfg(not(feature = "vio"))]
        let pose = pose.map(|pose_world| {
            let cam_from_odom = Pose64::from_pose32(pose_world);
            let cam_from_map = self.map_from_odom.odom_to_map(cam_from_odom);
            TrackingPose::new(cam_from_odom, cam_from_map, None)
        });
        #[cfg(feature = "vio")]
        let vio_telemetry = self.current_vio_telemetry();
        #[cfg(not(feature = "vio"))]
        let vio_telemetry = None;
        if let Some(pose_world) = pose.as_ref() {
            self.last_pose_world = Some(pose_world.cam_from_map_pose32());
        }
        TrackerOutput {
            pose,
            inliers,
            keyframe,
            stereo_matches,
            frame_id,
            health: self.emit_health(tracking),
            diagnostics,
            events: self.drain_events(),
            vio_telemetry,
        }
    }

    /// Build a `TrackerOutput` for a tracking failure (no pose, no keyframe, no matches).
    fn tracking_failure_output(
        &mut self,
        frame_id: FrameId,
        health: TrackingHealth,
        diagnostics: FrameDiagnostics,
    ) -> TrackerOutput {
        self.output_with_diagnostics(
            self.last_pose_world,
            0,
            None,
            None,
            frame_id,
            health,
            diagnostics,
        )
    }

    fn drain_descriptor_responses(&mut self) {
        let Some(place_recognition) = self.place_recognition.as_mut() else {
            return;
        };
        let events = place_recognition.drain_responses(self.map_version, |keyframe_id| {
            self.global_map.keyframe(keyframe_id).is_some()
        });
        for event in events {
            match event {
                PlaceRecognitionEvent::WorkerFailure {
                    keyframe_id,
                    map_version,
                    error,
                } => {
                    eprintln!(
                        "descriptor worker failure (keyframe={keyframe_id:?}, version={}): {error}",
                        map_version.as_u64()
                    );
                }
                PlaceRecognitionEvent::WorkerPanic {
                    keyframe_id,
                    map_version,
                    message,
                    respawn_count,
                } => {
                    eprintln!(
                        "descriptor worker panic (keyframe={keyframe_id:?}, version={}): {message}",
                        map_version.as_u64()
                    );
                    self.emit_event(DiagnosticEvent::DescriptorWorkerDied { respawn_count });
                }
            }
        }
    }

    fn process_pending_loop_closure(&mut self) -> Result<Option<VerifiedLoop>, LoopDetectError> {
        let Some(place_recognition) = self.place_recognition.as_mut() else {
            return Ok(None);
        };
        let config = place_recognition.loop_config();
        let Some(pending) = place_recognition.take_pending_loop() else {
            return Ok(None);
        };
        let query_quantized: Vec<_> = pending
            .detections
            .descriptors()
            .iter()
            .map(crate::Descriptor::quantize)
            .collect();

        let mut first_error: Option<LoopDetectError> = None;
        for candidate in pending.candidates {
            let correspondences = match_quantized_descriptors_for_loop(
                &query_quantized,
                candidate.candidate,
                self.global_map.map(),
                config.descriptor_match_threshold(),
            )
            .unwrap_or_else(|err| {
                eprintln!(
                    "loop descriptor matching skipped for candidate {:?}: {err}",
                    candidate.candidate
                );
                Vec::new()
            });

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
                similarity: candidate.similarity,
            };

            let verified = match loop_candidate.verify(
                pending.detections.keypoints(),
                &correspondences,
                self.global_map.map(),
                self.frontend.intrinsics(),
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

            let Some(match_pose) = self
                .global_map
                .keyframe(candidate.candidate)
                .map(|entry| entry.pose())
            else {
                if first_error.is_none() {
                    first_error = Some(LoopDetectError::ApplyFailed(
                        LoopApplyError::MissingKeyframe,
                    ));
                }
                continue;
            };
            let (translation, rotation_deg) =
                LoopManager::correction_magnitude(match_pose, verified.query_pose_world());
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
                let detect_err = LoopDetectError::ApplyFailed(LoopManager::apply_error_kind(&err));
                self.emit_event(DiagnosticEvent::LoopClosureRejected {
                    reason: LoopManager::reject_reason(&detect_err),
                });
                return Err(detect_err);
            }
            self.emit_event(DiagnosticEvent::LoopClosureDetected {
                query: pending.query_kf,
                match_kf: candidate.candidate,
                similarity: candidate.similarity,
            });
            return Ok(Some(verified));
        }

        if let Some(err) = first_error {
            self.emit_event(DiagnosticEvent::LoopClosureRejected {
                reason: LoopManager::reject_reason(&err),
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
            self.global_map.clone_map(),
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
                        match apply_correction_event(
                            self.global_map.map_mut(),
                            self.map_version,
                            &correction,
                        ) {
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

    fn maybe_enter_relocalization(
        &mut self,
        tracking_health: TrackingHealth,
        detections: Arc<Detections>,
    ) {
        if let Some(session) = Self::initial_relocalization_session(
            tracking_health,
            self.config.relocalization_config().is_some(),
            detections,
        ) {
            self.emit_event(DiagnosticEvent::RelocalizationStarted);
            if let Some(place_recognition) = self.place_recognition.as_mut() {
                place_recognition.clear_pending();
            }
            self.state = TrackerState::Relocalizing(session);
            if self.trace_transitions {
                eprintln!("entering relocalization after tracking loss");
            }
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
        self.output_with_diagnostics(
            self.last_pose_world,
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
        let next_attempts = session.attempts.saturating_add(1);
        if self.trace_transitions {
            eprintln!(
                "relocalization failure frame={} attempt={}/{}",
                frame_id.as_u64(),
                next_attempts,
                cfg.max_attempts()
            );
        }
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
    ) -> Option<crate::loop_closure::VerifiedRelocalization> {
        let place_recognition = self.place_recognition.as_ref()?;
        let global_descriptor = aggregate_global_descriptor(current.descriptors()).ok()?;
        let candidates =
            place_recognition.relocalization_matches(&global_descriptor, cfg.max_candidates());
        let query_quantized: Vec<_> = current
            .descriptors()
            .iter()
            .map(crate::Descriptor::quantize)
            .collect();
        for candidate in candidates {
            let correspondences = match_quantized_descriptors_for_loop(
                &query_quantized,
                candidate.candidate,
                self.global_map.map(),
                cfg.descriptor_match_threshold(),
            )
            .unwrap_or_else(|err| {
                eprintln!(
                    "relocalization descriptor matching skipped for candidate {:?}: {err}",
                    candidate.candidate
                );
                Vec::new()
            });
            if correspondences.len() < MIN_PNP_CORRESPONDENCES {
                continue;
            }
            let relocalization_candidate = RelocalizationCandidate {
                match_kf: candidate.candidate,
                similarity: candidate.similarity,
            };
            let verified = match relocalization_candidate.verify(
                current.keypoints(),
                &correspondences,
                self.global_map.map(),
                self.frontend.intrinsics(),
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
                        confirmations: NonZeroUsize::new(next_confirmations)
                            .unwrap_or(NonZeroUsize::MIN),
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
                    confirmations: NonZeroUsize::MIN,
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
        let (left, right) = pair.into_parts();
        let frame_id = left.frame_id();
        let Some(cfg) = self.config.relocalization_config() else {
            self.state = TrackerState::NeedKeyframe;
            return Ok(self.relocalization_output(frame_id, TrackingHealth::Lost));
        };

        let current =
            self.frontend
                .detect(&left, self.config.downscale, self.config.max_keypoints())?;

        if current.is_empty() {
            return Ok(self.fail_relocalization(frame_id, cfg, session, Arc::clone(&current)));
        }

        if self.place_recognition.is_none() {
            self.state = TrackerState::NeedKeyframe;
            return Ok(self.relocalization_output(frame_id, TrackingHealth::Lost));
        }

        let Some(verified) = self.relocalization_candidate(current.as_ref(), cfg) else {
            return Ok(self.fail_relocalization(frame_id, cfg, session, current));
        };
        let candidate_id = verified.match_kf();
        let pose_world = verified.pose_world();
        if self.trace_transitions {
            eprintln!(
                "relocalization candidate frame={} candidate={candidate_id:?} inliers={}",
                frame_id.as_u64(),
                verified.inlier_count()
            );
        }

        session.last_detections = current;
        match Self::relocalization_step(session, candidate_id, pose_world, cfg) {
            RelocalizationStep::Recovered { pose_world } => {
                self.last_pose_world = Some(pose_world);
                self.emit_event(DiagnosticEvent::RelocalizationSucceeded {
                    keyframe_id: candidate_id,
                });
                #[cfg(feature = "vio")]
                self.align_map_from_odom_to_pose(pose_world);
                if let Some(place_recognition) = self.place_recognition.as_mut() {
                    place_recognition.clear_pending();
                }
                self.state = TrackerState::NeedKeyframe;
                if self.trace_transitions {
                    eprintln!(
                        "relocalization recovered frame={} candidate={candidate_id:?}; creating bootstrap keyframe from recovered pose",
                        frame_id.as_u64()
                    );
                }
                return self.create_keyframe(StereoPair::from_parts(left, right), pose_world);
            }
            RelocalizationStep::Continue(next_session) => {
                self.state = TrackerState::Relocalizing(next_session);
                if self.trace_transitions {
                    eprintln!(
                        "relocalization confirmation pending frame={}",
                        frame_id.as_u64()
                    );
                }
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
        let (left, right) = pair.into_parts();
        let frame_id = left.frame_id();

        let current =
            self.frontend
                .detect(&left, self.config.downscale, self.config.max_keypoints())?;

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
            self.maybe_enter_relocalization(tracking_health, Arc::clone(&current));
            let mut diagnostics = self.empty_diagnostics();
            diagnostics.features_detected = Some(current.len());
            diagnostics.features_matched = Some(0);
            diagnostics.tracking_time = Some(tracking_start.elapsed());
            return Ok(self.tracking_failure_output(frame_id, tracking_health, diagnostics));
        } else {
            self.frontend
                .match_tracking(current.clone(), keyframe.detections().clone())?
        };

        let verified = match matches.with_landmarks(keyframe) {
            Ok(verified) => verified,
            Err(err) => {
                return Err(TrackerError::Inference(InferenceError::Match(err)));
            }
        };

        let tracked_observations = match self.frontend.build_map_observations(
            self.global_map.map(),
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
                self.maybe_enter_relocalization(tracking_health, Arc::clone(&current));
                let mut diagnostics = self.empty_diagnostics();
                diagnostics.features_detected = Some(current.len());
                diagnostics.features_matched = Some(matches.len());
                diagnostics.tracking_time = Some(tracking_start.elapsed());
                return Ok(self.tracking_failure_output(frame_id, tracking_health, diagnostics));
            }
            Err(err) => return Err(TrackerError::Pnp(err)),
        };

        let result = match self
            .frontend
            .solve_tracking_pose(&tracked_observations.observations, self.config.ransac)
        {
            Ok(result) => result,
            Err(crate::PnpError::NotEnoughPoints { .. } | crate::PnpError::NoSolution) => {
                if self.trace_transitions {
                    eprintln!(
                        "tracking failure frame={} reason=pnp_failed observations={} matches={} verified={}",
                        frame_id.as_u64(),
                        tracked_observations.len(),
                        matches.len(),
                        verified.len()
                    );
                }
                let tracking_health = self.tracking_failure_health();
                self.maybe_enter_relocalization(tracking_health, Arc::clone(&current));
                let mut diagnostics = self.empty_diagnostics();
                diagnostics.features_detected = Some(current.len());
                diagnostics.features_matched = Some(matches.len());
                diagnostics.pnp_observations = Some(tracked_observations.len());
                diagnostics.tracking_time = Some(tracking_start.elapsed());
                return Ok(self.tracking_failure_output(frame_id, tracking_health, diagnostics));
            }
            Err(err) => return Err(TrackerError::Pnp(err)),
        };

        let mut map_observations = Vec::with_capacity(result.inliers.len());
        for &idx in &result.inliers {
            let verified_idx = *tracked_observations.verified_match_indices.get(idx).ok_or(
                TrackerError::Inference(InferenceError::InvariantViolation {
                    context: "tracked observation index out of bounds",
                }),
            )?;
            let (ci, ki) = *verified
                .indices()
                .get(verified_idx)
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
            let keypoint_ref = self.global_map.keyframe_keypoint(keyframe_id, ki)?;
            map_observations.push(MapObservation::new(keypoint_ref, pixel));
        }

        let parallax_px = median_parallax_px(
            &verified,
            &tracked_observations.verified_match_indices,
            &result.inliers,
            keyframe,
        );
        let covisibility = if keyframe.landmarks().is_empty() {
            0.0
        } else {
            result.inliers.len() as f32 / keyframe.landmarks().len() as f32
        };

        let pose_world = result.pose;
        #[cfg(feature = "vio")]
        let refined_world = match &self.local_estimator {
            LocalEstimator::VisualOnly => {
                ObservationSet::new(map_observations, self.ba.min_observations())
                    .ok()
                    .and_then(|set| self.ba.push_frame(self.global_map.map(), pose_world, set))
            }
            LocalEstimator::Inertial(_) => None,
        };
        #[cfg(not(feature = "vio"))]
        let refined_world = ObservationSet::new(map_observations, self.ba.min_observations())
            .ok()
            .and_then(|set| self.ba.push_frame(self.global_map.map(), pose_world, set));

        let pose_world = refined_world.unwrap_or(pose_world);
        if self.consecutive_tracking_failures > 0 {
            self.emit_event(DiagnosticEvent::TrackingRecovered);
        }
        self.consecutive_tracking_failures = 0;
        let inlier_observations: Vec<_> = result
            .inliers
            .iter()
            .filter_map(|&idx| tracked_observations.observations.get(idx).copied())
            .collect();
        let mut output_keyframe = None;
        let mut output_matches = None;
        let mut keyframe_status = None;
        let mut triangulation_stats = None;
        let mut ba_result = None;

        let keyframe_decision =
            self.config
                .keyframe_policy
                .decide(result.inliers.len(), parallax_px, covisibility);

        #[cfg(feature = "vio")]
        if !matches!(keyframe_decision, KeyframeDecision::Insert(_)) {
            self.correct_tracking_pose_from_vio(pose_world, inlier_observations.clone())?;
        }

        if matches!(keyframe_decision, KeyframeDecision::Insert(_)) {
            let new_pose = pose_world;
            let shared = build_shared_matches(
                keyframe_id,
                &verified,
                &tracked_observations.verified_match_indices,
                &result.inliers,
            );
            if let Ok((keyframe_output, keyframe_id)) = self.create_keyframe_internal(
                left,
                right,
                new_pose,
                Some(current.clone()),
                Some(shared),
                Some(tracked_observations.observations.clone()),
            ) {
                keyframe_status = Some(KeyframeStatus::Created);
                triangulation_stats = keyframe_output.diagnostics.triangulation;
                ba_result = keyframe_output.diagnostics.ba_result.clone();
                if let Some(keyframe) = keyframe_output.keyframe {
                    let redundant = self
                        .config
                        .redundancy
                        .map(|policy| {
                            is_redundant(
                                self.global_map.map(),
                                keyframe_id,
                                policy.max_covisibility(),
                            )
                        })
                        .transpose()?
                        .unwrap_or(false);
                    if redundant {
                        if let Err(err) =
                            remove_keyframe_from_graph_and_db(&mut self.global_map, keyframe_id)
                        {
                            eprintln!("failed to remove redundant keyframe {keyframe_id:?}: {err}");
                        } else {
                            self.emit_event(DiagnosticEvent::KeyframeRemoved {
                                keyframe_id,
                                reason: KeyframeRemovalReason::Redundant,
                            });
                            if let Some(place_recognition) = self.place_recognition.as_mut() {
                                place_recognition.remove_keyframe(keyframe_id);
                            }
                            self.bump_map_version();
                        }
                    } else {
                        let window = self
                            .global_map
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
                                    let result = self.ba.optimize_keyframe_window(
                                        self.global_map.map_mut(),
                                        &window,
                                    );
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
                                    let result = self.ba.optimize_keyframe_window(
                                        self.global_map.map_mut(),
                                        &window,
                                    );
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

        let inlier_errors = crate::pnp::reprojection_errors(
            &pose_world,
            &inlier_observations,
            self.frontend.intrinsics(),
        );

        let mut diagnostics = self.empty_diagnostics();
        diagnostics.inlier_ratio =
            Some(result.inliers.len() as f32 / tracked_observations.len().max(1) as f32);
        diagnostics.pnp_observations = Some(tracked_observations.len());
        diagnostics.ransac_iterations = Some(result.iterations);
        diagnostics.reprojection_rmse_px = crate::pnp::reprojection_rmse(&inlier_errors);
        diagnostics.reprojection_max_px = crate::pnp::reprojection_max(&inlier_errors);
        diagnostics.parallax_px = parallax_px;
        diagnostics.covisibility = Some(covisibility);
        diagnostics.keyframe_status = keyframe_status;
        diagnostics.triangulation = triangulation_stats;
        diagnostics.ba_result = ba_result;
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
        let (left, right) = pair.into_parts();
        let frame_id = left.frame_id();
        let (output, keyframe_id) = match self
            .create_keyframe_internal(left, right, pose_world, None, None, None)
        {
            Ok(value) => value,
            Err(TrackerError::KeyframeRejected { landmarks }) => {
                if self.trace_transitions {
                    eprintln!(
                        "keyframe bootstrap rejected frame={} landmarks={} -> staying in NeedKeyframe",
                        frame_id.as_u64(),
                        landmarks
                    );
                }
                let mut diagnostics = self.empty_diagnostics();
                diagnostics.keyframe_status = Some(KeyframeStatus::Rejected);
                return Ok(self.tracking_failure_output(
                    frame_id,
                    TrackingHealth::Degraded,
                    diagnostics,
                ));
            }
            Err(err) => {
                if self.trace_transitions {
                    eprintln!(
                        "keyframe bootstrap rejected frame={} error={err}",
                        frame_id.as_u64()
                    );
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
        visual_observations: Option<Vec<Observation>>,
    ) -> Result<(TrackerOutput, KeyframeId), TrackerError> {
        let frame_id = left.frame_id();
        let max_keypoints = self.config.max_keypoints();

        let (left_arc, right_arc) = match left_det {
            Some(left_arc) => {
                let right_det =
                    self.frontend
                        .detect(&right, self.config.downscale, max_keypoints)?;
                (left_arc, right_det)
            }
            None => {
                let left_det = self
                    .frontend
                    .detect(&left, self.config.downscale, max_keypoints)?;
                let right_det =
                    self.frontend
                        .detect(&right, self.config.downscale, max_keypoints)?;

                (left_det, right_det)
            }
        };

        let matches = if left_arc.is_empty() || right_arc.is_empty() {
            return Err(TrackerError::KeyframeRejected { landmarks: 0 });
        } else {
            self.frontend
                .match_stereo(left_arc.clone(), right_arc.clone())?
        };

        let result = self.frontend.triangulate_matches(&matches)?;
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
        let keyframe_id = insert_keyframe_into_map(
            self.global_map.map_mut(),
            &keyframe,
            left.timestamp(),
            pose_world,
            shared.as_ref(),
        )?;
        self.emit_event(DiagnosticEvent::KeyframeCreated {
            keyframe_id,
            landmarks,
        });
        self.global_map.add_keyframe_to_graph(keyframe_id);
        self.bump_map_version();
        if let Some(place_recognition) = self.place_recognition.as_mut() {
            place_recognition.on_keyframe(
                keyframe_id,
                keyframe.detections(),
                &left,
                self.map_version,
            );
        }
        #[cfg(not(feature = "vio"))]
        let _ = visual_observations;
        #[cfg(feature = "vio")]
        self.on_keyframe_for_vio(
            keyframe_id,
            pose_world,
            visual_observations.unwrap_or_default(),
        )?;

        let mut diagnostics = self.empty_diagnostics();
        diagnostics.keyframe_status = Some(KeyframeStatus::Created);
        diagnostics.triangulation = Some(triangulation_stats);
        diagnostics.features_detected = Some(left_arc.len());
        diagnostics.features_matched = Some(matches.len());

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
            vio_telemetry: None,
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

    // Keep singleton points by default so active keyframes retain enough
    // point associations for robust PnP. Enable stronger culling only via env.
    let cull_min_observations = crate::env::env_usize("KIKO_MAP_CULL_MIN_OBSERVATIONS")
        .unwrap_or(DEFAULT_CULL_MIN_OBSERVATIONS)
        .max(DEFAULT_CULL_MIN_OBSERVATIONS);
    if cull_min_observations > 1 && map.num_points() > 0 {
        let points_before = map.num_points();
        let culled_points = map.cull_points(cull_min_observations);
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
    global_map: &mut GlobalMap,
    keyframe_id: KeyframeId,
) -> Result<(), TrackerError> {
    global_map.remove_keyframe_from_graph(keyframe_id)?;
    global_map.remove_keyframe(keyframe_id)?;
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

#[cfg(feature = "vio")]
fn tracking_pose_from_vio_output(
    map_from_odom: &MapFromOdom,
    cam_from_odom: Pose64,
    pose_map_measurement: Option<Pose>,
) -> TrackingPose {
    let cam_from_map_corrected = map_from_odom.odom_to_map(cam_from_odom);
    let cam_from_map_visual_measurement = pose_map_measurement.map(Pose64::from_pose32);
    TrackingPose::new(
        cam_from_odom,
        cam_from_map_corrected,
        cam_from_map_visual_measurement,
    )
}

fn build_shared_matches(
    keyframe_id: KeyframeId,
    matches: &Matches<Verified>,
    verified_match_indices: &[usize],
    inliers: &[usize],
) -> SharedMatches {
    let mut pairs = Vec::with_capacity(inliers.len());
    for &idx in inliers {
        let Some(&verified_idx) = verified_match_indices.get(idx) else {
            continue;
        };
        if let Some(&(ci, ki)) = matches.indices().get(verified_idx) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loop_closure::KeyframeDatabase;
    use crate::loop_manager::LoopManager;
    use crate::map::assert_map_invariants;
    use crate::place_recognition::{
        DescriptorExtractorFactory, DescriptorRequest, DescriptorSupervisor, DescriptorWorker,
        DescriptorWorkerResponse,
    };
    use crate::pose_graph::{EssentialEdgeKind, EssentialGraph, PoseGraphConfig};
    use crate::{
        CompactDescriptor, Descriptor, Detections, Keypoint, PlaceDescriptorExtractor, Point3,
        SensorId, Timestamp,
    };
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
    use std::time::Duration;

    #[cfg(feature = "vio")]
    use crate::MapFromOdom;

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

    #[test]
    fn build_map_observations_preserves_verified_match_indices_after_filtering() {
        let keypoints = vec![
            Keypoint { x: 80.0, y: 70.0 },
            Keypoint { x: 95.0, y: 72.0 },
            Keypoint { x: 110.0, y: 75.0 },
            Keypoint { x: 125.0, y: 78.0 },
            Keypoint { x: 140.0, y: 82.0 },
        ];
        let scores = vec![1.0; keypoints.len()];
        let descriptors = vec![make_descriptor(); keypoints.len()];
        let keyframe_detections = Arc::new(
            Detections::new(
                SensorId::StereoLeft,
                FrameId::new(40),
                320,
                240,
                keypoints.clone(),
                scores.clone(),
                descriptors.clone(),
            )
            .expect("keyframe detections"),
        );
        let keyframe = Keyframe::from_arc(
            Arc::clone(&keyframe_detections),
            vec![
                Point3 {
                    x: -0.2,
                    y: -0.1,
                    z: 3.0,
                },
                Point3 {
                    x: -0.1,
                    y: -0.1,
                    z: 3.1,
                },
                Point3 {
                    x: 0.0,
                    y: 0.0,
                    z: 3.2,
                },
                Point3 {
                    x: 0.1,
                    y: 0.1,
                    z: 3.3,
                },
                Point3 {
                    x: 0.2,
                    y: 0.2,
                    z: 3.4,
                },
            ],
            vec![0, 1, 2, 3, 4],
        )
        .expect("keyframe");

        let mut map = SlamMap::new();
        let keyframe_id = map
            .add_keyframe_from_detections(
                keyframe.detections().as_ref(),
                Timestamp::from_nanos(1),
                Pose::identity(),
            )
            .expect("map keyframe");

        for &det_idx in &[0_usize, 2, 3, 4] {
            let keypoint_ref = map
                .keyframe_keypoint(keyframe_id, det_idx)
                .expect("keypoint ref");
            map.add_map_point(
                keyframe
                    .landmark_for_detection(det_idx)
                    .expect("landmark for detection"),
                make_descriptor().quantize(),
                keypoint_ref,
            )
            .expect("map point");
        }

        let current = Arc::new(
            Detections::new(
                SensorId::StereoLeft,
                FrameId::new(41),
                320,
                240,
                keypoints,
                scores,
                descriptors,
            )
            .expect("current detections"),
        );
        let matches = Matches::new(
            Arc::clone(&current),
            Arc::clone(&keyframe_detections),
            vec![(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)],
            vec![1.0; 5],
        )
        .expect("matches");
        let verified = matches.with_landmarks(&keyframe).expect("verified matches");
        let intrinsics =
            crate::test_helpers::make_pinhole_intrinsics(320, 240, 300.0, 300.0, 160.0, 120.0)
                .expect("intrinsics");

        let tracked = crate::frontend::build_map_observations(
            &map,
            keyframe_id,
            &verified,
            current.as_ref(),
            intrinsics,
        )
        .expect("tracked observations");

        assert_eq!(tracked.observations.len(), 4);
        assert_eq!(tracked.verified_match_indices, vec![0, 2, 3, 4]);
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
        let mut worker =
            BackendWorker::spawn(backend_cfg, intrinsics, ba_cfg).expect("spawn backend worker");

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

    #[test]
    fn descriptor_supervisor_retries_after_transient_spawn_failure() {
        let config = GlobalDescriptorConfig::new(2).expect("config");
        let spawn_count = Arc::new(AtomicUsize::new(0));
        let calls = Arc::new(Mutex::new(0_usize));
        let descriptor = make_global_descriptor_basis(23);

        let factory: DescriptorExtractorFactory = {
            let spawn_count = Arc::clone(&spawn_count);
            let calls = Arc::clone(&calls);
            let descriptor = descriptor.clone();
            Arc::new(move || {
                let spawn_idx = spawn_count.fetch_add(1, AtomicOrdering::SeqCst);
                if spawn_idx == 0 {
                    None
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
        assert!(!supervisor.has_worker(), "initial worker should be absent");

        let frame = Frame::new(
            SensorId::StereoLeft,
            FrameId::new(79),
            Timestamp::from_nanos(79),
            16,
            12,
            vec![64_u8; 16 * 12],
        )
        .expect("frame");

        supervisor
            .submit(DescriptorRequest {
                keyframe_id: KeyframeId::default(),
                map_version: MapVersion::initial(),
                frame,
            })
            .expect("submit should trigger retry and succeed");

        let mut recovered = None;
        for _ in 0..50 {
            match supervisor.try_recv() {
                Some(DescriptorWorkerResponse::Descriptor(value)) => {
                    recovered = Some(value);
                    break;
                }
                Some(other) => panic!("unexpected descriptor response: {other:?}"),
                None => {}
            }
            std::thread::sleep(Duration::from_millis(5));
        }

        let recovered = recovered.expect("descriptor response after retry");
        assert_eq!(recovered.descriptor, descriptor);
        assert_eq!(supervisor.respawn_count(), 1);
        assert!(supervisor.has_worker());
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
        let (map, essential_graph, verified, query_kf, _) = make_loop_closure_apply_fixture();
        let mut global_map = GlobalMap::from_parts(map.clone(), essential_graph.clone());
        let loop_manager = LoopManager::new(PoseGraphConfig::default());

        let before = map
            .keyframe(query_kf)
            .expect("query pose")
            .pose()
            .translation();
        let before_error =
            ((before[0] - 2.0).powi(2) + (before[1]).powi(2) + (before[2]).powi(2)).sqrt();

        loop_manager
            .apply_verified_loop(&mut global_map, &verified)
            .expect("apply loop closure");

        let after = global_map
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
        let (map, essential_graph, verified, _query_kf, before_points) =
            make_loop_closure_apply_fixture();
        let mut global_map = GlobalMap::from_parts(map.clone(), essential_graph.clone());
        let loop_manager = LoopManager::new(PoseGraphConfig::default());

        loop_manager
            .apply_verified_loop(&mut global_map, &verified)
            .expect("apply loop closure");

        let moved_points = before_points
            .iter()
            .filter(|(point_id, before)| {
                let after = global_map.point(*point_id).expect("point").position();
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
        let (map, essential_graph, verified, _query_kf, _before_points) =
            make_loop_closure_apply_fixture();
        let mut global_map = GlobalMap::from_parts(map.clone(), essential_graph.clone());
        let loop_manager = LoopManager::new(PoseGraphConfig::default());

        assert_eq!(global_map.essential_graph().snapshot().loop_edges.len(), 0);
        loop_manager
            .apply_verified_loop(&mut global_map, &verified)
            .expect("apply loop closure");
        let snapshot = global_map.essential_graph().snapshot();
        assert_eq!(snapshot.loop_edges.len(), 1);
        assert_eq!(snapshot.loop_edges[0].kind, EssentialEdgeKind::Loop);
    }

    #[test]
    fn remove_keyframe_from_graph_and_db_cleans_all_structures() {
        let (map, essential_graph, _verified, removed_kf, _before_points) =
            make_loop_closure_apply_fixture();
        let mut global_map = GlobalMap::from_parts(map.clone(), essential_graph.clone());
        let mut loop_db = KeyframeDatabase::new(0);
        for (idx, (keyframe_id, _)) in map.keyframes().enumerate() {
            loop_db.insert_with_source(
                keyframe_id,
                make_global_descriptor_basis(idx),
                crate::loop_closure::DescriptorSource::Bootstrap,
            );
        }

        remove_keyframe_from_graph_and_db(&mut global_map, removed_kf).expect("remove keyframe");
        loop_db.remove(removed_kf);

        assert!(global_map.keyframe(removed_kf).is_none());
        assert!(global_map.essential_graph().parent_of(removed_kf).is_none());
        assert!(loop_db.descriptor_source(removed_kf).is_none());
        let input = global_map.essential_graph().pose_graph_input();
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
        let detections = make_test_detections(900);
        assert!(
            SlamTracker::initial_relocalization_session(
                TrackingHealth::Good,
                true,
                Arc::clone(&detections)
            )
            .is_none()
        );
        assert!(
            SlamTracker::initial_relocalization_session(
                TrackingHealth::Lost,
                false,
                Arc::clone(&detections)
            )
            .is_none()
        );

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

    #[cfg(feature = "vio")]
    #[test]
    fn map_from_odom_alignment_maps_current_odom_pose_to_measured_map_pose() {
        let cam_from_odom = Pose64::from_pose32(Pose::from_rt(
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            [0.3, -0.2, 1.4],
        ));
        let pose_map = Pose::from_rt(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [2.0, 3.0, -1.0],
        );
        let mut bridge = MapFromOdom::identity();
        bridge.align_to_pose(Pose64::from_pose32(pose_map), cam_from_odom);
        let mapped = bridge.odom_to_map(cam_from_odom).to_pose32();
        assert_eq!(mapped.translation(), pose_map.translation());
        assert_eq!(mapped.rotation(), pose_map.rotation());
    }

    #[cfg(feature = "vio")]
    #[test]
    fn vio_output_prefers_map_measurement_without_mutating_bridge() {
        let cam_from_odom = Pose64::from_pose32(Pose::from_rt(
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            [0.3, -0.2, 1.4],
        ));
        let bridge_pose = Pose64::from_pose32(Pose::from_rt(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [1.0, 2.0, 3.0],
        ));
        let measured_map_pose = Pose::from_rt(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [2.0, 3.0, -1.0],
        );
        let mut bridge = MapFromOdom::identity();
        bridge.set_pose_map_from_odom(bridge_pose);
        let expected_bridge_pose = bridge.odom_to_map(cam_from_odom).to_pose32();

        let tracking_pose =
            tracking_pose_from_vio_output(&bridge, cam_from_odom, Some(measured_map_pose));

        assert_eq!(
            tracking_pose.cam_from_map_pose32().translation(),
            expected_bridge_pose.translation()
        );
        assert_eq!(
            tracking_pose.cam_from_map_pose32().rotation(),
            expected_bridge_pose.rotation()
        );
        assert_eq!(
            tracking_pose
                .cam_from_map_visual_measurement_pose32()
                .expect("visual measurement")
                .translation(),
            measured_map_pose.translation()
        );
        assert_eq!(
            tracking_pose
                .cam_from_map_visual_measurement_pose32()
                .expect("visual measurement")
                .rotation(),
            measured_map_pose.rotation()
        );
    }
}
