use std::collections::HashSet;
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

/// Minimum 3D-2D correspondences needed for PnP pose estimation.
const MIN_PNP_CORRESPONDENCES: usize = 4;
/// During bootstrap and sparse-map tracking, accept a smaller inlier set instead
/// of requiring the fully-configured mature-map threshold immediately.
const MIN_TRACKING_RANSAC_INLIERS: usize = MIN_PNP_CORRESPONDENCES;
/// Target roughly 25% of tracked observations as the adaptive inlier gate until
/// the configured mature-map threshold becomes achievable.
const TRACKING_RANSAC_INLIER_DIVISOR: usize = 4;
/// Default maximum respawn attempts for backend and descriptor workers.
const DEFAULT_MAX_RESPAWNS: u32 = 3;
/// Minimum keyframes required for multi-frame optimization (BA or pose graph).
const MIN_OPTIMIZATION_KEYFRAMES: usize = 2;
/// Default minimum observations per map point to survive culling.
const DEFAULT_CULL_MIN_OBSERVATIONS: usize = 1;

use crate::frontend::{
    MapObservationError, StereoFrontend, TrackedMapObservations, median_parallax_px,
};
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
use crate::pose_graph::{EssentialGraphError, PoseGraphConfig, PoseGraphError};
use crate::{
    BaCorrection, BaResult, CalibrationBundle, CaptureBundle, CaptureBundleError, CaptureId,
    Detections, DiagnosticEvent, DownscaleFactor, Frame, FrameDiagnostics, FrameId, Keyframe,
    KeyframeRemovalReason, KeyframeStatus, KeypointLimit, LightGlue, LocalBaConfig,
    LocalBundleAdjuster, LoopClosureStatus, MapFromOdom, MapObservation, Matches, Observation,
    ObservationSet, PinholeIntrinsics, Point3, Pose, Pose64, RansacConfig, RansacConfigError, Raw,
    StereoPair, SuperPoint, Timestamp, TriangulationConfig, TriangulationError, Triangulator,
    Verified,
    map::{KeyframeId, MapPointId, SlamMap},
};
#[cfg(feature = "vio")]
use crate::{Gravity, ImuAccumulator};

use crate::inference::InferenceError;
use crossbeam_channel::{Receiver, Sender, TryRecvError, TrySendError};

#[derive(Clone, Copy, Debug)]
pub struct TrackerConfig {
    pub max_keypoints: KeypointLimit,
    pub downscale: DownscaleFactor,
    pub tracking_matcher: TrackingMatcher,
    pub min_keyframe_points: usize,
    pub ransac: RansacConfig,
    pub triangulation: TriangulationConfig,
    pub keyframe_policy: KeyframePolicy,
    pub ba: LocalBaConfig,
    pub redundancy: Option<RedundancyPolicy>,
    pub backend: Option<BackendConfig>,
    pub loop_subsystem: LoopSubsystemConfig,
    #[cfg(feature = "vio")]
    pub vio_enabled: bool,
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TrackingMatcher {
    LightGlue,
    Projected(ProjectedMatcherConfig),
}

impl TrackingMatcher {
    pub fn uses_speculative_lightglue(self) -> bool {
        matches!(self, Self::LightGlue)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProjectedMatcherConfig {
    pub search_radius_px: f32,
    pub min_similarity: f32,
    pub min_matches: usize,
    pub min_inliers: usize,
}

impl ProjectedMatcherConfig {
    pub fn jetson_default() -> Self {
        Self {
            search_radius_px: 32.0,
            min_similarity: 0.45,
            min_matches: 32,
            min_inliers: 24,
        }
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

impl std::error::Error for BackendWorkerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            BackendWorkerError::BuildCorrection(source) => Some(source),
        }
    }
}

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

impl std::error::Error for SubmitEventError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SubmitEventError::InvalidWindow(source) => Some(source),
            SubmitEventError::InvalidEvent(source) => Some(source),
            SubmitEventError::QueueFull | SubmitEventError::Disconnected => None,
        }
    }
}

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

impl std::error::Error for ApplyCorrectionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ApplyCorrectionError::Map(source) => Some(source),
            ApplyCorrectionError::StaleVersion { .. }
            | ApplyCorrectionError::MissingKeyframe { .. }
            | ApplyCorrectionError::MissingMapPoint { .. } => None,
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
enum BackendSubsystem {
    Disabled,
    Configured(BackendSupervisor),
}

impl BackendSubsystem {
    fn supervisor(&self) -> Option<&BackendSupervisor> {
        match self {
            Self::Configured(supervisor) => Some(supervisor),
            Self::Disabled => None,
        }
    }

    fn supervisor_mut(&mut self) -> Option<&mut BackendSupervisor> {
        match self {
            Self::Configured(supervisor) => Some(supervisor),
            Self::Disabled => None,
        }
    }

    fn is_configured(&self) -> bool {
        matches!(self, Self::Configured(_))
    }

    fn health_flags(&self) -> (bool, bool) {
        match self {
            Self::Disabled => (false, false),
            Self::Configured(supervisor) => (true, supervisor.has_worker()),
        }
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

impl KeyframeInsertReason {
    fn trace_label(self) -> String {
        match self {
            KeyframeInsertReason::TooFewInliers {
                inliers,
                min_required,
            } => {
                format!("too_few_inliers inliers={inliers} min_required={min_required}")
            }
            KeyframeInsertReason::HighParallax {
                parallax_px,
                threshold_px,
            } => {
                format!("high_parallax parallax_px={parallax_px:.2} threshold_px={threshold_px:.2}")
            }
            KeyframeInsertReason::LowCovisibility {
                covisibility,
                threshold,
            } => {
                format!("low_covisibility covisibility={covisibility:.3} threshold={threshold:.3}")
            }
        }
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
    RansacConfig(RansacConfigError),
    MapObservation(MapObservationError),
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
            TrackerError::RansacConfig(err) => write!(f, "RANSAC config error: {err}"),
            TrackerError::MapObservation(err) => write!(f, "map observation error: {err}"),
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
            TrackerError::Capture(source) => Some(source),
            TrackerError::Inference(source) => Some(source),
            TrackerError::Triangulation(source) => Some(source),
            TrackerError::Pnp(source) => Some(source),
            TrackerError::Map(source) => Some(source),
            TrackerError::EssentialGraph(source) => Some(source),
            TrackerError::PoseGraph(source) => Some(source),
            TrackerError::RansacConfig(source) => Some(source),
            TrackerError::MapObservation(source) => Some(source),
            #[cfg(feature = "vio")]
            TrackerError::Vio(_) => None,
            TrackerError::KeyframeRejected { .. } | TrackerError::InvariantViolation(_) => None,
        }
    }
}

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

impl From<RansacConfigError> for TrackerError {
    fn from(err: RansacConfigError) -> Self {
        TrackerError::RansacConfig(err)
    }
}

impl From<MapObservationError> for TrackerError {
    fn from(err: MapObservationError) -> Self {
        TrackerError::MapObservation(err)
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

#[derive(Clone, Debug)]
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

#[derive(Debug)]
pub enum PoseStatus {
    Current(TrackingPose),
    Predicted(TrackingPose),
    Stale {
        pose: TrackingPose,
        source_frame_id: FrameId,
    },
    Unavailable,
}

impl PoseStatus {
    pub fn current_estimate(&self) -> Option<&TrackingPose> {
        match self {
            Self::Current(pose) | Self::Predicted(pose) => Some(pose),
            Self::Stale { .. } | Self::Unavailable => None,
        }
    }

    pub fn last_known_pose(&self) -> Option<&TrackingPose> {
        match self {
            Self::Current(pose) | Self::Predicted(pose) | Self::Stale { pose, .. } => Some(pose),
            Self::Unavailable => None,
        }
    }

    pub fn stale_source_frame_id(&self) -> Option<FrameId> {
        match self {
            Self::Stale {
                source_frame_id, ..
            } => Some(*source_frame_id),
            Self::Current(_) | Self::Predicted(_) | Self::Unavailable => None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VioTelemetry {
    velocity_odom_mps: [f64; 3],
    accel_bias_mps2: [f64; 3],
    gyro_bias_radps: [f64; 3],
}

impl VioTelemetry {
    #[cfg(feature = "vio")]
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
    pub pose: PoseStatus,
    pub inliers: usize,
    pub keyframe: Option<Arc<Keyframe>>,
    pub stereo_matches: Option<Matches<Raw>>,
    pub frame_id: FrameId,
    pub health: SystemHealth,
    pub diagnostics: FrameDiagnostics,
    pub events: Vec<DiagnosticEvent>,
    pub vio_telemetry: Option<VioTelemetry>,
}

#[derive(Clone, Debug)]
struct LastAcceptedPose {
    frame_id: FrameId,
    pose_world: Pose,
    tracking_pose: TrackingPose,
}

#[derive(Debug)]
struct CreatedKeyframe {
    keyframe_id: KeyframeId,
    keyframe: Arc<Keyframe>,
    stereo_matches: Matches<Raw>,
    diagnostics: FrameDiagnostics,
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
    reference_cam_from_odom: Option<Pose64>,
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

#[derive(Clone, Copy, Debug)]
struct DepthSummary {
    min_m: f32,
    median_m: f32,
    max_m: f32,
}

fn summarize_depths(points: &[Point3]) -> Option<DepthSummary> {
    if points.is_empty() {
        return None;
    }
    let mut depths: Vec<f32> = points.iter().map(|point| point.z).collect();
    depths.sort_by(|a, b| a.total_cmp(b));
    let len = depths.len();
    let median_m = if len % 2 == 1 {
        depths[len / 2]
    } else {
        (depths[len / 2 - 1] + depths[len / 2]) * 0.5
    };
    Some(DepthSummary {
        min_m: depths[0],
        median_m,
        max_m: depths[len - 1],
    })
}

fn adaptive_tracking_ransac_config(
    base: RansacConfig,
    observation_count: usize,
) -> Result<RansacConfig, RansacConfigError> {
    let target_min_inliers = observation_count.saturating_add(TRACKING_RANSAC_INLIER_DIVISOR - 1)
        / TRACKING_RANSAC_INLIER_DIVISOR;
    let max_configured = base.min_inliers().max(MIN_TRACKING_RANSAC_INLIERS);
    base.try_with_min_inliers(target_min_inliers.clamp(MIN_TRACKING_RANSAC_INLIERS, max_configured))
}

#[cfg(feature = "vio")]
fn decide_vio_pose_adoption(
    current_frame_visual_residual_count: usize,
    visual_metrics: &PoseReprojectionMetrics,
    vio_metrics: &PoseReprojectionMetrics,
) -> crate::VioProposalDisposition {
    if current_frame_visual_residual_count == 0 {
        return crate::VioProposalDisposition::RejectedInsufficientCurrentVioObservationSupport;
    }
    let shared = visual_metrics.shared_with(vio_metrics);
    if shared.count < MIN_PNP_CORRESPONDENCES {
        return crate::VioProposalDisposition::RejectedInsufficientSharedAcceptedInlierSupport;
    }
    if shared.count != visual_metrics.projectable_count()
        || shared.count != vio_metrics.projectable_count()
    {
        return crate::VioProposalDisposition::RejectedChangedAcceptedInlierProjectability;
    }
    match (shared.lhs_rmse_px, shared.rhs_rmse_px) {
        (Some(visual_rmse_px), Some(vio_rmse_px)) if vio_rmse_px <= visual_rmse_px => {
            crate::VioProposalDisposition::Adopted
        }
        (Some(_), Some(_)) => {
            crate::VioProposalDisposition::RejectedHigherSharedAcceptedInlierReprojectionRmse
        }
        _ => crate::VioProposalDisposition::RejectedInsufficientSharedAcceptedInlierSupport,
    }
}

#[cfg(feature = "vio")]
fn should_adopt_visual_ba_proposal(
    visual_metrics: &PoseReprojectionMetrics,
    visual_ba_metrics: &PoseReprojectionMetrics,
) -> bool {
    let shared = visual_metrics.shared_with(visual_ba_metrics);
    if shared.count < MIN_PNP_CORRESPONDENCES {
        return false;
    }
    if shared.count != visual_metrics.projectable_count()
        || shared.count != visual_ba_metrics.projectable_count()
    {
        return false;
    }
    matches!(
        (shared.lhs_rmse_px, shared.rhs_rmse_px),
        (Some(visual_rmse_px), Some(visual_ba_rmse_px)) if visual_ba_rmse_px <= visual_rmse_px
    )
}

#[cfg(feature = "vio")]
fn pose_odom_from_body_from_camera_pose(
    camera_from_odom: Pose64,
    camera_from_body: Pose64,
) -> Pose64 {
    camera_from_odom.inverse().compose(camera_from_body)
}

#[cfg(feature = "vio")]
fn camera_from_odom_from_pose_odom_from_body(
    pose_odom_from_body: Pose64,
    camera_from_body: Pose64,
) -> Pose64 {
    camera_from_body.compose(pose_odom_from_body.inverse())
}

#[cfg(feature = "vio")]
enum LocalEstimator {
    VisualOnly,
    Inertial(Box<VioRuntime>),
}

#[cfg(feature = "vio")]
struct VioRuntime {
    camera_from_body: Pose64,
    noise: crate::ImuNoiseModel,
    pending_imu: ImuAccumulator,
    predicted_state: Option<crate::NavState>,
    last_visual_measurement_body_odom: Option<(Timestamp, Pose64)>,
    calibrated_bias: Option<crate::ImuBias>,
    /// The last BA-optimized NavState. Used as base for preintegration.
    last_optimized_state: Option<crate::NavState>,
    /// VIO solve config (immutable after construction).
    solve_config: crate::VioSolveConfig,
    /// Sliding window for tightly-coupled VIO BA.
    vio_window: Option<crate::local_ba::VioWindow>,
    /// Max window size before oldest frame rolls off.
    max_window: usize,
}

#[cfg(feature = "vio")]
impl VioRuntime {
    fn set_capture_imu_interval(
        &mut self,
        batch: Option<&crate::ImuBatch>,
    ) -> Result<(), crate::ImuAccumulatorError> {
        self.pending_imu.clear();
        if let Some(batch) = batch {
            self.pending_imu.extend_batch(batch)?;
        }
        Ok(())
    }

    fn reset_runtime_continuity(&mut self) {
        self.pending_imu.clear();
        self.predicted_state = None;
        self.last_visual_measurement_body_odom = None;
        self.last_optimized_state = None;
        self.vio_window = None;
    }

    fn bias_seed(&self) -> crate::ImuBias {
        self.last_optimized_state
            .as_ref()
            .map(|state| state.bias().clone())
            .or_else(|| self.calibrated_bias.clone())
            .unwrap_or_default()
    }

    fn velocity_seed(&self, current_body_odom: Pose64, capture_time: Timestamp) -> [f64; 3] {
        self.visual_velocity_seed(current_body_odom, capture_time)
            .or_else(|| {
                self.last_optimized_state
                    .as_ref()
                    .map(|state| state.velocity_odom_mps())
            })
            .unwrap_or([0.0; 3])
    }

    #[cfg(test)]
    fn commit_authoritative_visual_anchor(&mut self, anchor: crate::local_ba::VioAnchor) {
        let nav_state = anchor.synced.nav_state().clone();
        self.last_optimized_state = Some(nav_state.clone());
        self.predicted_state = Some(nav_state);
        self.vio_window = Some(crate::local_ba::VioWindow {
            anchor,
            successors: Vec::new(),
        });
        self.pending_imu.clear();
    }

    fn commit_authoritative_pose(
        &mut self,
        capture_time: Timestamp,
        body_odom: Pose64,
        observations: Option<ObservationSet>,
    ) {
        let velocity = self.velocity_seed(body_odom, capture_time);
        let bias = self.bias_seed();
        let Ok(nav_state) = crate::NavState::try_new(body_odom, velocity, bias) else {
            self.reset_runtime_continuity();
            return;
        };
        self.record_visual_measurement(capture_time, body_odom);
        self.pending_imu.clear();
        self.last_optimized_state = Some(nav_state.clone());
        self.predicted_state = Some(nav_state.clone());
        self.vio_window = observations.map(|observations| crate::local_ba::VioWindow {
            anchor: crate::local_ba::VioAnchor {
                synced: crate::local_ba::SyncedPose::new(nav_state.clone()),
                observations: Some(observations),
                anchor_velocity_odom_mps: nav_state.velocity_odom_mps(),
            },
            successors: Vec::new(),
        });
    }

    fn visual_velocity_seed(
        &self,
        current_body_odom: Pose64,
        capture_time: Timestamp,
    ) -> Option<[f64; 3]> {
        let (previous_time, previous_body_odom) = self.last_visual_measurement_body_odom?;
        let dt = capture_time.seconds_since(previous_time);
        if !dt.is_finite() || dt <= 0.0 {
            return None;
        }
        let current = current_body_odom.translation();
        let previous = previous_body_odom.translation();
        Some([
            (current[0] - previous[0]) / dt,
            (current[1] - previous[1]) / dt,
            (current[2] - previous[2]) / dt,
        ])
    }

    fn record_visual_measurement(&mut self, capture_time: Timestamp, body_odom: Pose64) {
        self.last_visual_measurement_body_odom = Some((capture_time, body_odom));
    }
}

#[cfg(feature = "vio")]
#[derive(Debug)]
enum PoseRefinementProposal {
    None,
    VisualBa(VisualBaPoseProposal),
    Vio(Box<VioPoseProposal>),
}

#[cfg(feature = "vio")]
#[derive(Debug)]
struct VisualBaPoseProposal {
    pose_world: Pose,
    optimized_ba: LocalBundleAdjuster,
}

#[cfg(feature = "vio")]
#[derive(Debug)]
struct VioPoseProposal {
    pose_world: Pose,
    solve_result: crate::VioSolveResult,
    optimized_state: crate::NavState,
    optimized_window: crate::local_ba::VioWindow,
}

#[derive(Clone, Debug)]
struct PoseReprojectionMetrics {
    errors: Vec<Option<f32>>,
}

impl PoseReprojectionMetrics {
    fn from_pose(pose: &Pose, observations: &[Observation], intrinsics: PinholeIntrinsics) -> Self {
        Self {
            errors: crate::pnp::reprojection_errors(pose, observations, intrinsics),
        }
    }

    fn projectable_count(&self) -> usize {
        self.errors.iter().filter(|error| error.is_some()).count()
    }

    fn rmse_px(&self) -> Option<f32> {
        crate::pnp::reprojection_rmse(&self.errors)
    }

    fn max_px(&self) -> Option<f32> {
        crate::pnp::reprojection_max(&self.errors)
    }

    fn mse_per_axis_px2(&self) -> Option<f64> {
        crate::pnp::reprojection_mse_per_axis_px2(&self.errors)
    }

    #[cfg(all(test, feature = "vio"))]
    fn from_errors(errors: Vec<Option<f32>>) -> Self {
        Self { errors }
    }

    #[cfg(feature = "vio")]
    fn shared_with(&self, other: &Self) -> SharedPoseReprojectionMetrics {
        let mut lhs = Vec::new();
        let mut rhs = Vec::new();
        for (lhs_error, rhs_error) in self.errors.iter().zip(&other.errors) {
            if let (Some(lhs_px), Some(rhs_px)) = (lhs_error, rhs_error) {
                lhs.push(Some(*lhs_px));
                rhs.push(Some(*rhs_px));
            }
        }
        SharedPoseReprojectionMetrics {
            count: lhs.len(),
            lhs_rmse_px: crate::pnp::reprojection_rmse(&lhs),
            rhs_rmse_px: crate::pnp::reprojection_rmse(&rhs),
        }
    }
}

#[cfg(feature = "vio")]
#[derive(Clone, Copy, Debug)]
struct SharedPoseReprojectionMetrics {
    count: usize,
    lhs_rmse_px: Option<f32>,
    rhs_rmse_px: Option<f32>,
}

struct TrackingAttempt {
    matches: Matches<Raw>,
    verified: Matches<Verified>,
    tracked_observations: TrackedMapObservations,
    tracking_ransac: RansacConfig,
    result: crate::PnpResult,
}

enum TrackingAttemptError {
    NotEnoughMapPoints {
        matches: usize,
        verified: usize,
        observations: usize,
        required_observations: usize,
    },
    PnpFailed {
        matches: usize,
        verified: usize,
        observations: usize,
        required_inliers: usize,
    },
    Fatal(TrackerError),
}

impl TrackingAttemptError {
    fn trace_label(&self) -> &'static str {
        match self {
            Self::NotEnoughMapPoints { .. } => "not_enough_map_points",
            Self::PnpFailed { .. } => "pnp_failed",
            Self::Fatal(_) => "fatal",
        }
    }
}

#[derive(Clone, Copy)]
struct ProjectedMatchCandidate {
    current_idx: usize,
    keyframe_idx: usize,
    score: f32,
    distance_sq: f32,
}

struct CurrentKeypointGrid {
    cell_size: f32,
    cols: usize,
    rows: usize,
    cells: Vec<Vec<usize>>,
}

impl CurrentKeypointGrid {
    fn new(detections: &Detections, search_radius_px: f32) -> Self {
        let cell_size = search_radius_px.max(1.0);
        let cols = ((detections.width() as f32) / cell_size).ceil().max(1.0) as usize;
        let rows = ((detections.height() as f32) / cell_size).ceil().max(1.0) as usize;
        let mut cells = vec![Vec::new(); cols * rows];
        for (idx, kp) in detections.keypoints().iter().enumerate() {
            if kp.x < 0.0 || kp.y < 0.0 {
                continue;
            }
            let col = ((kp.x / cell_size).floor() as usize).min(cols.saturating_sub(1));
            let row = ((kp.y / cell_size).floor() as usize).min(rows.saturating_sub(1));
            cells[row * cols + col].push(idx);
        }
        Self {
            cell_size,
            cols,
            rows,
            cells,
        }
    }

    fn for_each(&self, x: f32, y: f32, radius: f32, mut visit: impl FnMut(usize)) {
        let min_col = (((x - radius) / self.cell_size).floor() as isize)
            .clamp(0, self.cols.saturating_sub(1) as isize) as usize;
        let max_col = (((x + radius) / self.cell_size).floor() as isize)
            .clamp(0, self.cols.saturating_sub(1) as isize) as usize;
        let min_row = (((y - radius) / self.cell_size).floor() as isize)
            .clamp(0, self.rows.saturating_sub(1) as isize) as usize;
        let max_row = (((y + radius) / self.cell_size).floor() as isize)
            .clamp(0, self.rows.saturating_sub(1) as isize) as usize;
        for row in min_row..=max_row {
            for col in min_col..=max_col {
                for &idx in &self.cells[row * self.cols + col] {
                    visit(idx);
                }
            }
        }
    }
}

fn descriptor_similarity(a: &crate::Descriptor, b: &crate::Descriptor) -> f32 {
    a.as_slice()
        .iter()
        .zip(b.as_slice())
        .map(|(lhs, rhs)| lhs * rhs)
        .sum()
}

fn project_world_point(
    pose_world_to_camera: Pose,
    point: Point3,
    intrinsics: PinholeIntrinsics,
) -> Option<(f32, f32)> {
    let pc = crate::math::transform_point(
        pose_world_to_camera.rotation(),
        pose_world_to_camera.translation(),
        [point.x, point.y, point.z],
    );
    if pc[2] <= 0.0 {
        return None;
    }
    Some((
        intrinsics.fx() * (pc[0] / pc[2]) + intrinsics.cx(),
        intrinsics.fy() * (pc[1] / pc[2]) + intrinsics.cy(),
    ))
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
    backend: BackendSubsystem,
    backend_stats: BackendStats,
    place_recognition: Option<PlaceRecognition>,
    pending_events: Vec<DiagnosticEvent>,
    tracking_health: TrackingHealth,
    consecutive_tracking_failures: usize,
    last_accepted_pose: Option<LastAcceptedPose>,
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
        let local_estimator = match (
            config.vio_enabled,
            calibration.imu_noise(),
            calibration.has_imu(),
        ) {
            (true, Some(noise), true) => {
                let gravity = Gravity::try_new([0.0, calibration.gravity_magnitude_mps2(), 0.0])
                    .map_err(|err| TrackerInitError::VioInvalidGravity {
                        message: err.to_string(),
                    })?;
                let camera_from_body = calibration
                    .imu_extrinsics()
                    .map(|extrinsics| extrinsics.t_cam_imu())
                    .unwrap_or_else(Pose64::identity);
                let vio_window_size = crate::env::env_usize("KIKO_VIO_WINDOW").unwrap_or(5);
                let vio_max_iters = crate::env::env_usize("KIKO_VIO_ITERS").unwrap_or(3);
                let calibrated_bias = calibration.initial_bias().cloned();
                let bias_prior = calibrated_bias
                    .clone()
                    .map(|bias| crate::VioBiasPrior::new(100.0, bias))
                    .transpose()
                    .map_err(|err| TrackerInitError::VioInvalidGravity {
                        message: err.to_string(),
                    })?;
                let solve_config = crate::VioSolveConfig::new(
                    gravity,
                    camera_from_body,
                    intrinsics,
                    config.ba.lm(),
                    std::num::NonZeroUsize::new(vio_max_iters)
                        .unwrap_or(NonZeroUsize::new(3).unwrap()),
                    f64::from(config.ba.huber_delta_px()),
                    10.0, // anchor velocity info
                    bias_prior,
                )
                .map_err(|err| TrackerInitError::VioInvalidGravity {
                    message: err.to_string(),
                })?;
                LocalEstimator::Inertial(Box::new(VioRuntime {
                    camera_from_body,
                    noise: noise.clone(),
                    pending_imu: ImuAccumulator::new(),
                    predicted_state: None,
                    last_visual_measurement_body_odom: None,
                    calibrated_bias,
                    last_optimized_state: None,
                    solve_config,
                    vio_window: None,
                    max_window: vio_window_size,
                }))
            }
            _ => LocalEstimator::VisualOnly,
        };
        let backend_max_respawns = crate::env::env_usize("KIKO_BACKEND_MAX_RESPAWNS")
            .and_then(|value| u32::try_from(value).ok())
            .unwrap_or(DEFAULT_MAX_RESPAWNS);
        let backend = config
            .backend
            .map_or(BackendSubsystem::Disabled, |backend_cfg| {
                BackendSubsystem::Configured(BackendSupervisor::spawn_with_max_respawns(
                    backend_cfg,
                    intrinsics,
                    config.ba,
                    backend_max_respawns,
                ))
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
            last_accepted_pose: None,
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

    /// Take the prefetch SP session for background detection.
    pub fn take_prefetch_sp(&mut self) -> Option<SuperPoint> {
        self.frontend.take_prefetch_sp()
    }

    /// Return the prefetch SP session after background use.
    pub fn return_prefetch_sp(&mut self, sp: SuperPoint) {
        self.frontend.return_prefetch_sp(sp);
    }

    pub fn take_prefetch_lg(&mut self) -> Option<LightGlue> {
        self.frontend.take_prefetch_lg()
    }

    pub fn return_prefetch_lg(&mut self, lg: LightGlue) {
        self.frontend.return_prefetch_lg(lg);
    }

    /// Return current tracking keyframe info for speculative LightGlue prefetch.
    pub fn current_tracking_keyframe_detections(
        &self,
    ) -> Option<(KeyframeId, std::sync::Arc<Detections>)> {
        match &self.state {
            TrackerState::Tracking {
                keyframe,
                keyframe_id,
            } => Some((*keyframe_id, keyframe.tracking_detections().clone())),
            _ => None,
        }
    }

    pub fn process_capture(
        &mut self,
        capture: CaptureBundle,
    ) -> Result<TrackerOutput, TrackerError> {
        self.process_capture_with_prefetch(capture, None, None)
    }

    pub fn process_capture_with_prefetch(
        &mut self,
        capture: CaptureBundle,
        prefetched_left: Option<(crate::FrameId, std::sync::Arc<Detections>)>,
        prefetched_matches: Option<(KeyframeId, Matches<Raw>)>,
    ) -> Result<TrackerOutput, TrackerError> {
        #[cfg(feature = "vio")]
        {
            self.set_capture_imu_interval(capture.imu().batch())?;
            self.drain_vio_responses();
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
            self.track_with_prefetch(
                pair,
                &keyframe,
                keyframe_id,
                prefetched_left,
                prefetched_matches,
            )
        } else {
            let bootstrap_pose = self.last_pose_world().unwrap_or(Pose::identity());
            if self.trace_transitions {
                eprintln!(
                    "tracker bootstrap keyframe: pose_source={} tx={:.3} ty={:.3} tz={:.3}",
                    if self.last_accepted_pose.is_some() {
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
    fn set_capture_imu_interval(
        &mut self,
        batch: Option<&crate::ImuBatch>,
    ) -> Result<(), TrackerError> {
        let LocalEstimator::Inertial(vio_runtime) = &mut self.local_estimator else {
            return Ok(());
        };
        vio_runtime
            .set_capture_imu_interval(batch)
            .map_err(|err| TrackerError::Vio(err.to_string()))
    }

    // --- Dead VIO methods deleted (M0 cleanup) ---
    // drain_vio_responses, apply_vio_odometry, update_last_pose_from_vio_state,
    // refresh_predicted_pose_from_vio, correct_predicted_pose_from_visual_measurement,
    // on_keyframe_for_vio — all replaced by tightly-coupled BA in M2.

    /// Run tightly-coupled VIO BA if IMU data is available, otherwise
    /// fall back to visual-only BA.
    #[cfg(feature = "vio")]
    fn run_vio_or_visual_ba(
        &mut self,
        capture_time: Timestamp,
        pose_world: Pose,
        map_observations: Vec<MapObservation>,
    ) -> PoseRefinementProposal {
        let LocalEstimator::Inertial(vio_runtime) = &mut self.local_estimator else {
            // Visual-only mode
            return ObservationSet::new(map_observations, self.ba.min_observations())
                .ok()
                .and_then(|set| self.propose_visual_ba_pose(pose_world, set))
                .map_or(
                    PoseRefinementProposal::None,
                    PoseRefinementProposal::VisualBa,
                );
        };

        // Build NavState for this frame from visual pose
        let camera_odom = self
            .map_from_odom
            .map_to_odom(Pose64::from_pose32(pose_world));
        let body_odom =
            pose_odom_from_body_from_camera_pose(camera_odom, vio_runtime.camera_from_body);
        let visual_velocity_seed = vio_runtime.visual_velocity_seed(body_odom, capture_time);
        let prev_vel = vio_runtime.velocity_seed(body_odom, capture_time);
        let prev_bias = vio_runtime.bias_seed();

        let nav_state = match crate::NavState::try_new(body_odom, prev_vel, prev_bias.clone()) {
            Ok(s) => s,
            Err(_) => {
                return ObservationSet::new(map_observations, self.ba.min_observations())
                    .ok()
                    .and_then(|set| self.propose_visual_ba_pose(pose_world, set))
                    .map_or(
                        PoseRefinementProposal::None,
                        PoseRefinementProposal::VisualBa,
                    );
            }
        };

        // Preintegrate pending IMU
        let obs_set = ObservationSet::new(map_observations, self.ba.min_observations()).ok();

        let preintegrated = match vio_runtime.pending_imu.batch() {
            Ok(Some(batch)) if batch.len() >= 2 => {
                crate::PreintegratedImu::integrate(&batch, &prev_bias, &vio_runtime.noise).ok()
            }
            _ => None,
        };

        let Some(preintegrated) = preintegrated else {
            // No IMU data — fall back to visual BA
            return obs_set
                .and_then(|set| self.propose_visual_ba_pose(pose_world, set))
                .map_or(
                    PoseRefinementProposal::None,
                    PoseRefinementProposal::VisualBa,
                );
        };

        // Build or extend the VIO window
        use crate::local_ba::{SyncedPose, VioAnchor, VioSuccessor, VioWindow, optimize_vio};

        if vio_runtime.vio_window.is_none() {
            if let (Some(anchor_state), Some(successor_observations)) =
                (vio_runtime.last_optimized_state.clone(), obs_set.clone())
            {
                let mut candidate_window = VioWindow {
                    anchor: VioAnchor {
                        synced: SyncedPose::new(anchor_state.clone()),
                        observations: None,
                        anchor_velocity_odom_mps: anchor_state.velocity_odom_mps(),
                    },
                    successors: vec![VioSuccessor {
                        synced: SyncedPose::new(nav_state.clone()),
                        observations: Some(successor_observations),
                        preintegrated,
                    }],
                };
                let result = optimize_vio(
                    &mut candidate_window,
                    &vio_runtime.solve_config,
                    self.global_map.map(),
                    &self.map_from_odom,
                );

                let optimized = candidate_window
                    .successors
                    .last()
                    .map(|s| s.synced.nav_state().clone())
                    .unwrap_or_else(|| candidate_window.anchor.synced.nav_state().clone());
                let cam_from_odom = camera_from_odom_from_pose_odom_from_body(
                    optimized.pose_odom_from_body(),
                    vio_runtime.camera_from_body,
                );
                let cam_from_map = self.map_from_odom.odom_to_map(cam_from_odom).to_pose32();

                if self.trace_transitions {
                    let vel = optimized.velocity_odom_mps();
                    let bias = optimized.bias();
                    let delta = crate::local_ba::se3_delta_between(pose_world, cam_from_map);
                    let delta_t =
                        (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
                    let delta_r =
                        (delta[3] * delta[3] + delta[4] * delta[4] + delta[5] * delta[5]).sqrt();
                    eprintln!(
                        "vio ba: frames={} iters={} cost={:.1} reproj={:.1} imu={:.1} vel_anchor={:.1} bias_rw={:.1} bias_prior={:.1} vel=[{:.3},{:.3},{:.3}] ba=[{:.3},{:.3},{:.3}] pose_delta_mm={:.2} rot_delta_mdeg={:.1}",
                        candidate_window.len(),
                        result.iterations,
                        result.final_cost,
                        result.cost_breakdown.reprojection_cost,
                        result.cost_breakdown.imu_cost,
                        result.cost_breakdown.velocity_anchor_cost,
                        result.cost_breakdown.bias_random_walk_cost,
                        result.cost_breakdown.bias_prior_cost,
                        vel[0],
                        vel[1],
                        vel[2],
                        bias.accel[0],
                        bias.accel[1],
                        bias.accel[2],
                        delta_t * 1000.0,
                        delta_r.to_degrees() * 1000.0,
                    );
                }

                return PoseRefinementProposal::Vio(Box::new(VioPoseProposal {
                    pose_world: cam_from_map,
                    solve_result: result,
                    optimized_state: optimized,
                    optimized_window: candidate_window,
                }));
            }

            if visual_velocity_seed.is_none() {
                if self.trace_transitions {
                    eprintln!(
                        "vio bootstrap: deferring inertial anchor until a visual velocity seed is available"
                    );
                }
                vio_runtime.last_optimized_state = Some(nav_state.clone());
                vio_runtime.predicted_state = Some(nav_state);
                vio_runtime.pending_imu.clear();
                return obs_set
                    .and_then(|set| self.propose_visual_ba_pose(pose_world, set))
                    .map_or(
                        PoseRefinementProposal::None,
                        PoseRefinementProposal::VisualBa,
                    );
            }
            // First frame: create anchor
            let anchor = VioAnchor {
                synced: SyncedPose::new(nav_state.clone()),
                observations: obs_set.clone(),
                anchor_velocity_odom_mps: nav_state.velocity_odom_mps(),
            };
            vio_runtime.vio_window = Some(VioWindow {
                anchor,
                successors: Vec::new(),
            });
            vio_runtime.last_optimized_state = Some(nav_state.clone());
            vio_runtime.predicted_state = Some(nav_state);
            vio_runtime.pending_imu.clear();
            return PoseRefinementProposal::None;
        }

        // Build a candidate window. It only becomes authoritative if the caller
        // accepts the proposal against the visual metric on the same support.
        let mut candidate_window = vio_runtime.vio_window.clone().unwrap();
        candidate_window.successors.push(VioSuccessor {
            synced: SyncedPose::new(nav_state.clone()),
            observations: obs_set.clone(),
            preintegrated,
        });

        // Trim window
        while candidate_window.len() > vio_runtime.max_window {
            if candidate_window.successors.len() <= 1 {
                break;
            }
            // Promote second frame to anchor, drop first successor's preintegration
            let old_succ = candidate_window.successors.remove(0);
            let anchor_velocity_odom_mps = old_succ.synced.nav_state().velocity_odom_mps();
            candidate_window.anchor = VioAnchor {
                synced: old_succ.synced,
                observations: old_succ.observations,
                anchor_velocity_odom_mps,
            };
        }

        // Run the VIO optimizer
        let result = optimize_vio(
            &mut candidate_window,
            &vio_runtime.solve_config,
            self.global_map.map(),
            &self.map_from_odom,
        );

        // Extract optimized state from the last frame
        let optimized = candidate_window
            .successors
            .last()
            .map(|s| s.synced.nav_state().clone())
            .unwrap_or_else(|| candidate_window.anchor.synced.nav_state().clone());

        let cam_from_odom = camera_from_odom_from_pose_odom_from_body(
            optimized.pose_odom_from_body(),
            vio_runtime.camera_from_body,
        );
        let cam_from_map = self.map_from_odom.odom_to_map(cam_from_odom).to_pose32();

        if self.trace_transitions {
            let vel = optimized.velocity_odom_mps();
            let bias = optimized.bias();
            let delta = crate::local_ba::se3_delta_between(pose_world, cam_from_map);
            let delta_t = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
            let delta_r = (delta[3] * delta[3] + delta[4] * delta[4] + delta[5] * delta[5]).sqrt();
            eprintln!(
                "vio ba: frames={} iters={} cost={:.1} reproj={:.1} imu={:.1} vel_anchor={:.1} bias_rw={:.1} bias_prior={:.1} vel=[{:.3},{:.3},{:.3}] ba=[{:.3},{:.3},{:.3}] pose_delta_mm={:.2} rot_delta_mdeg={:.1}",
                candidate_window.len(),
                result.iterations,
                result.final_cost,
                result.cost_breakdown.reprojection_cost,
                result.cost_breakdown.imu_cost,
                result.cost_breakdown.velocity_anchor_cost,
                result.cost_breakdown.bias_random_walk_cost,
                result.cost_breakdown.bias_prior_cost,
                vel[0],
                vel[1],
                vel[2],
                bias.accel[0],
                bias.accel[1],
                bias.accel[2],
                delta_t * 1000.0,
                delta_r.to_degrees() * 1000.0,
            );
        }

        PoseRefinementProposal::Vio(Box::new(VioPoseProposal {
            pose_world: cam_from_map,
            solve_result: result,
            optimized_state: optimized,
            optimized_window: candidate_window,
        }))
    }

    #[cfg(feature = "vio")]
    fn propose_visual_ba_pose(
        &self,
        pose_world: Pose,
        observations: ObservationSet,
    ) -> Option<VisualBaPoseProposal> {
        let mut candidate_ba = self.ba.clone();
        candidate_ba
            .push_frame(self.global_map.map(), pose_world, observations)
            .map(|refined_pose| VisualBaPoseProposal {
                pose_world: refined_pose,
                optimized_ba: candidate_ba,
            })
    }

    #[cfg(feature = "vio")]
    fn commit_visual_ba_proposal(&mut self, proposal: VisualBaPoseProposal) {
        self.ba = proposal.optimized_ba;
    }

    #[cfg(feature = "vio")]
    fn commit_vio_proposal(&mut self, capture_time: Timestamp, proposal: VioPoseProposal) {
        let LocalEstimator::Inertial(vio_runtime) = &mut self.local_estimator else {
            return;
        };
        vio_runtime.record_visual_measurement(
            capture_time,
            proposal.optimized_state.pose_odom_from_body(),
        );
        vio_runtime.last_optimized_state = Some(proposal.optimized_state.clone());
        vio_runtime.predicted_state = Some(proposal.optimized_state);
        vio_runtime.vio_window = Some(proposal.optimized_window);
        vio_runtime.pending_imu.clear();
    }

    #[cfg(feature = "vio")]
    fn commit_authoritative_visual_pose(
        &mut self,
        capture_time: Timestamp,
        pose_world: Pose,
        map_observations: &[MapObservation],
    ) {
        let LocalEstimator::Inertial(vio_runtime) = &mut self.local_estimator else {
            return;
        };
        let camera_odom = self
            .map_from_odom
            .map_to_odom(Pose64::from_pose32(pose_world));
        let body_odom =
            pose_odom_from_body_from_camera_pose(camera_odom, vio_runtime.camera_from_body);
        let observations =
            ObservationSet::new(map_observations.to_vec(), self.ba.min_observations()).ok();
        vio_runtime.commit_authoritative_pose(capture_time, body_odom, observations);
    }

    #[cfg(feature = "vio")]
    fn reset_inertial_runtime_continuity(&mut self) {
        let LocalEstimator::Inertial(vio_runtime) = &mut self.local_estimator else {
            return;
        };
        vio_runtime.reset_runtime_continuity();
    }

    #[cfg(feature = "vio")]
    fn drain_vio_responses(&mut self) {}

    #[cfg(feature = "vio")]
    fn refresh_predicted_pose_from_vio(&mut self) -> Result<(), TrackerError> {
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
        let (backend_expected, backend_alive) = self.backend.health_flags();
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
        {
            self.realign_map_from_odom(&corrected);
            self.reset_inertial_runtime_continuity();
        }
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
    fn realign_map_from_odom(&mut self, corrected_poses: &[(KeyframeId, Pose)]) {
        // After loop closure, corrected_poses contains updated map-frame poses.
        // Use the most recent one to realign map_from_odom so that the odom
        // trajectory remains continuous while the map frame absorbs the correction.
        let Some(cam_from_odom) = self.current_odom_pose() else {
            return;
        };
        // Use the last corrected pose (most recent keyframe) as the alignment target
        if let Some((_, corrected_map_pose)) = corrected_poses.last() {
            self.map_from_odom
                .align_to_pose(Pose64::from_pose32(*corrected_map_pose), cam_from_odom);
        }
    }

    #[cfg(feature = "vio")]
    fn current_odom_pose(&self) -> Option<Pose64> {
        match &self.local_estimator {
            LocalEstimator::VisualOnly => None,
            LocalEstimator::Inertial(vio_runtime) => {
                vio_runtime.predicted_state.as_ref().map(|state| {
                    camera_from_odom_from_pose_odom_from_body(
                        state.pose_odom_from_body(),
                        vio_runtime.camera_from_body,
                    )
                })
            }
        }
    }

    #[cfg(feature = "vio")]
    fn current_vio_telemetry(&self) -> Option<VioTelemetry> {
        match &self.local_estimator {
            LocalEstimator::VisualOnly => None,
            LocalEstimator::Inertial(vio_runtime) => vio_runtime
                .predicted_state
                .as_ref()
                .map(VioTelemetry::from_nav_state),
        }
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
        visual_pose: Option<Pose>,
        inliers: usize,
        keyframe: Option<Arc<Keyframe>>,
        stereo_matches: Option<Matches<Raw>>,
        frame_id: FrameId,
        tracking: TrackingHealth,
        diagnostics: FrameDiagnostics,
    ) -> TrackerOutput {
        #[cfg(feature = "vio")]
        let current_odom_pose = self.current_odom_pose();
        #[cfg(not(feature = "vio"))]
        let current_odom_pose = None;
        let pose = classify_pose_status(
            &self.map_from_odom,
            current_odom_pose,
            visual_pose,
            self.last_accepted_pose.as_ref(),
        );
        #[cfg(feature = "vio")]
        let vio_telemetry = self.current_vio_telemetry();
        #[cfg(not(feature = "vio"))]
        let vio_telemetry = None;
        if let PoseStatus::Current(pose_world) = &pose {
            self.last_accepted_pose = Some(LastAcceptedPose {
                frame_id,
                pose_world: pose_world.cam_from_map_pose32(),
                tracking_pose: pose_world.clone(),
            });
            #[cfg(feature = "vio")]
            if self.trace_transitions {
                if let Some(measurement) = pose_world.cam_from_map_visual_measurement_pose32() {
                    let delta = crate::local_ba::se3_delta_between(
                        pose_world.cam_from_map_pose32(),
                        measurement,
                    );
                    if delta.iter().all(|value| value.is_finite()) {
                        let translation_m =
                            (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2])
                                .sqrt();
                        let rotation_deg =
                            (delta[3] * delta[3] + delta[4] * delta[4] + delta[5] * delta[5])
                                .sqrt()
                                .to_degrees();
                        eprintln!(
                            "vio pose delta translation_m={translation_m:.3} rotation_deg={rotation_deg:.3}"
                        );
                    }
                }
            }
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

    /// Build an output whose pose is explicitly predicted, stale, or unavailable.
    fn tracking_failure_output(
        &mut self,
        frame_id: FrameId,
        health: TrackingHealth,
        diagnostics: FrameDiagnostics,
    ) -> TrackerOutput {
        self.output_with_diagnostics(None, 0, None, None, frame_id, health, diagnostics)
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
        let Some(supervisor) = self.backend.supervisor_mut() else {
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
                let Some(supervisor) = self.backend.supervisor_mut() else {
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
                    if let Some(supervisor) = self.backend.supervisor_mut() {
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
        if tracking_health != TrackingHealth::Lost {
            return;
        }
        #[cfg(feature = "vio")]
        let reference_cam_from_odom = self.current_odom_pose();
        #[cfg(not(feature = "vio"))]
        let reference_cam_from_odom = None;
        if let Some(session) = Self::initial_relocalization_session(
            tracking_health,
            self.config.relocalization_config().is_some(),
            detections,
            reference_cam_from_odom,
        ) {
            #[cfg(feature = "vio")]
            self.reset_inertial_runtime_continuity();
            self.emit_event(DiagnosticEvent::RelocalizationStarted);
            if let Some(place_recognition) = self.place_recognition.as_mut() {
                place_recognition.clear_pending();
            }
            self.state = TrackerState::Relocalizing(session);
            if self.trace_transitions {
                eprintln!("entering relocalization after tracking loss");
            }
        } else {
            #[cfg(feature = "vio")]
            self.reset_inertial_runtime_continuity();
            self.state = TrackerState::NeedKeyframe;
            if self.trace_transitions {
                eprintln!("tracking lost without relocalization; resetting to NeedKeyframe");
            }
        }
    }

    fn initial_relocalization_session(
        tracking_health: TrackingHealth,
        relocalization_enabled: bool,
        detections: Arc<Detections>,
        reference_cam_from_odom: Option<Pose64>,
    ) -> Option<RelocalizationSession> {
        if tracking_health != TrackingHealth::Lost || !relocalization_enabled {
            return None;
        }
        Some(RelocalizationSession {
            attempts: 0,
            phase: RelocalizationPhase::Searching,
            last_detections: detections,
            reference_cam_from_odom,
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
        let next_attempts = session.attempts.saturating_add(1);
        if self.trace_transitions {
            eprintln!(
                "relocalization failure frame={} attempt={}/{}",
                frame_id.as_u64(),
                next_attempts,
                cfg.max_attempts()
            );
        }
        let next_state = Self::next_state_after_relocalization_failure(cfg, session, current);
        if matches!(next_state, TrackerState::NeedKeyframe) {
            #[cfg(feature = "vio")]
            self.reset_inertial_runtime_continuity();
        }
        self.state = next_state;
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
                    reference_cam_from_odom: session.reference_cam_from_odom,
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
                reference_cam_from_odom: session.reference_cam_from_odom,
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
            #[cfg(feature = "vio")]
            self.reset_inertial_runtime_continuity();
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
            #[cfg(feature = "vio")]
            self.reset_inertial_runtime_continuity();
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
        #[cfg(feature = "vio")]
        let reference_cam_from_odom = session.reference_cam_from_odom;
        match Self::relocalization_step(session, candidate_id, pose_world, cfg) {
            RelocalizationStep::Recovered { pose_world } => {
                self.emit_event(DiagnosticEvent::RelocalizationSucceeded {
                    keyframe_id: candidate_id,
                });
                #[cfg(feature = "vio")]
                {
                    if let Some(reference_cam_from_odom) =
                        reference_cam_from_odom.or_else(|| self.current_odom_pose())
                    {
                        self.map_from_odom.align_to_pose(
                            Pose64::from_pose32(pose_world),
                            reference_cam_from_odom,
                        );
                    }
                    self.reset_inertial_runtime_continuity();
                }
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

    fn build_tracking_attempt(
        &self,
        keyframe: &Keyframe,
        keyframe_id: KeyframeId,
        matches: Matches<Raw>,
    ) -> Result<TrackingAttempt, TrackingAttemptError> {
        let match_count = matches.len();
        let verified = matches
            .with_landmarks(self.global_map.map().instance_id(), keyframe_id, keyframe)
            .map_err(|err| {
                TrackingAttemptError::Fatal(TrackerError::Inference(InferenceError::Match(err)))
            })?;
        let verified_count = verified.len();

        let tracked_observations = match self
            .frontend
            .build_map_observations(self.global_map.map(), &verified)
        {
            Ok(obs) => obs,
            Err(MapObservationError::NotEnoughPoints { required, actual }) => {
                return Err(TrackingAttemptError::NotEnoughMapPoints {
                    matches: match_count,
                    verified: verified_count,
                    observations: actual,
                    required_observations: required,
                });
            }
            Err(err) => {
                return Err(TrackingAttemptError::Fatal(TrackerError::MapObservation(
                    err,
                )));
            }
        };
        let tracking_ransac =
            adaptive_tracking_ransac_config(self.config.ransac, tracked_observations.len())
                .map_err(|err| TrackingAttemptError::Fatal(TrackerError::RansacConfig(err)))?;

        let result = match self
            .frontend
            .solve_tracking_pose(&tracked_observations.observations, tracking_ransac)
        {
            Ok(result) => result,
            Err(crate::PnpError::NotEnoughPoints { .. } | crate::PnpError::NoSolution) => {
                return Err(TrackingAttemptError::PnpFailed {
                    matches: match_count,
                    verified: verified_count,
                    observations: tracked_observations.len(),
                    required_inliers: tracking_ransac.min_inliers(),
                });
            }
            Err(err) => return Err(TrackingAttemptError::Fatal(TrackerError::Pnp(err))),
        };

        Ok(TrackingAttempt {
            matches,
            verified,
            tracked_observations,
            tracking_ransac,
            result,
        })
    }

    fn lightglue_tracking_matches(
        &mut self,
        current: Arc<Detections>,
        keyframe: &Keyframe,
        keyframe_id: KeyframeId,
        prefetched_matches: Option<(KeyframeId, Matches<Raw>)>,
        frame_id: FrameId,
    ) -> Result<Matches<Raw>, TrackerError> {
        let tracking_matches = if let Some((prefetch_keyframe_id, prefetched_raw)) =
            prefetched_matches
        {
            if prefetch_keyframe_id == keyframe_id {
                if self.trace_transitions {
                    eprintln!("speculative LightGlue hit: frame={}", frame_id.as_u64());
                }
                prefetched_raw
            } else {
                if self.trace_transitions {
                    eprintln!(
                        "speculative LightGlue miss: prefetched={prefetch_keyframe_id:?} current={keyframe_id:?}"
                    );
                }
                self.frontend
                    .match_tracking(current, keyframe.tracking_detections().clone())?
            }
        } else {
            self.frontend
                .match_tracking(current, keyframe.tracking_detections().clone())?
        };
        keyframe
            .remap_tracking_matches(&tracking_matches)
            .map_err(|err| TrackerError::Inference(InferenceError::Match(err)))
    }

    fn predicted_tracking_pose(&self) -> Option<Pose> {
        #[cfg(feature = "vio")]
        if let Some(cam_from_odom) = self.current_odom_pose() {
            return Some(self.map_from_odom.odom_to_map(cam_from_odom).to_pose32());
        }
        self.last_pose_world()
    }

    fn last_pose_world(&self) -> Option<Pose> {
        self.last_accepted_pose.as_ref().map(|last| last.pose_world)
    }

    fn projected_tracking_matches(
        &self,
        current: Arc<Detections>,
        keyframe: &Keyframe,
        keyframe_id: KeyframeId,
        config: ProjectedMatcherConfig,
    ) -> Result<Option<Matches<Raw>>, TrackerError> {
        let Some(predicted_pose) = self.predicted_tracking_pose() else {
            return Ok(None);
        };
        let radius = config.search_radius_px;
        let radius_sq = radius * radius;
        let intrinsics = self.frontend.intrinsics();
        let grid = CurrentKeypointGrid::new(&current, radius);
        let current_keypoints = current.keypoints();
        let current_descriptors = current.descriptors();
        let keyframe_descriptors = keyframe.detections().descriptors();
        let mut candidates = Vec::with_capacity(keyframe.landmark_indices().len());

        for &keyframe_idx in keyframe.landmark_indices() {
            let keypoint_ref = self
                .global_map
                .keyframe_keypoint(keyframe_id, keyframe_idx)?;
            let Some(point_id) = self.global_map.map_point_for_keypoint(keypoint_ref)? else {
                continue;
            };
            let Some(point) = self.global_map.point(point_id) else {
                continue;
            };
            let Some((u, v)) = project_world_point(predicted_pose, point.position(), intrinsics)
            else {
                continue;
            };
            if u < -radius
                || v < -radius
                || u >= current.width() as f32 + radius
                || v >= current.height() as f32 + radius
            {
                continue;
            }

            let key_desc = &keyframe_descriptors[keyframe_idx];
            let mut best: Option<ProjectedMatchCandidate> = None;
            grid.for_each(u, v, radius, |current_idx| {
                let kp = current_keypoints[current_idx];
                let dx = kp.x - u;
                let dy = kp.y - v;
                let distance_sq = dx * dx + dy * dy;
                if distance_sq > radius_sq {
                    return;
                }
                let similarity = descriptor_similarity(&current_descriptors[current_idx], key_desc);
                if similarity < config.min_similarity {
                    return;
                }
                let spatial_penalty = 0.05 * (distance_sq / radius_sq);
                let score = similarity - spatial_penalty;
                match best {
                    Some(prev) if prev.score >= score => {}
                    _ => {
                        best = Some(ProjectedMatchCandidate {
                            current_idx,
                            keyframe_idx,
                            score,
                            distance_sq,
                        });
                    }
                }
            });
            if let Some(best) = best {
                candidates.push(best);
            }
        }

        if candidates.len() < config.min_matches {
            return Ok(None);
        }
        candidates.sort_unstable_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.distance_sq.total_cmp(&b.distance_sq))
        });

        let mut used_current = vec![false; current.len()];
        let mut used_keyframe = vec![false; keyframe.detections().len()];
        let mut indices = Vec::with_capacity(candidates.len());
        let mut scores = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            if used_current[candidate.current_idx] || used_keyframe[candidate.keyframe_idx] {
                continue;
            }
            used_current[candidate.current_idx] = true;
            used_keyframe[candidate.keyframe_idx] = true;
            indices.push((candidate.current_idx, candidate.keyframe_idx));
            scores.push(candidate.score);
        }

        if indices.len() < config.min_matches {
            return Ok(None);
        }
        Matches::new(current, Arc::clone(keyframe.detections()), indices, scores)
            .map(Some)
            .map_err(|err| TrackerError::Inference(InferenceError::Match(err)))
    }

    fn track_with_prefetch(
        &mut self,
        pair: StereoPair,
        keyframe: &Arc<Keyframe>,
        keyframe_id: KeyframeId,
        prefetched_left: Option<(crate::FrameId, std::sync::Arc<Detections>)>,
        prefetched_matches: Option<(KeyframeId, Matches<Raw>)>,
    ) -> Result<TrackerOutput, TrackerError> {
        let tracking_start = Instant::now();
        let (left, right) = pair.into_parts();
        let frame_id = left.frame_id();

        let current = self.frontend.detect_or_use_prefetched(
            &left,
            self.config.downscale,
            self.config.max_keypoints(),
            prefetched_left,
        )?;

        let attempt = if current.is_empty() || keyframe.tracking_detections().is_empty() {
            if self.trace_transitions {
                eprintln!(
                    "tracking failure frame={} reason=empty_features current={} keyframe={}",
                    frame_id.as_u64(),
                    current.len(),
                    keyframe.tracking_detections().len()
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
            let mut attempt = None;
            if let TrackingMatcher::Projected(config) = self.config.tracking_matcher {
                match self.projected_tracking_matches(
                    current.clone(),
                    keyframe,
                    keyframe_id,
                    config,
                )? {
                    Some(projected_matches) => {
                        let projected_match_count = projected_matches.len();
                        match self.build_tracking_attempt(keyframe, keyframe_id, projected_matches)
                        {
                            Ok(projected_attempt)
                                if projected_attempt.result.inliers.len() >= config.min_inliers =>
                            {
                                if self.trace_transitions {
                                    eprintln!(
                                        "projected tracking accepted frame={} matches={} observations={} inliers={}",
                                        frame_id.as_u64(),
                                        projected_match_count,
                                        projected_attempt.tracked_observations.len(),
                                        projected_attempt.result.inliers.len(),
                                    );
                                }
                                attempt = Some(projected_attempt);
                            }
                            Ok(projected_attempt) => {
                                if self.trace_transitions {
                                    eprintln!(
                                        "projected tracking fallback frame={} reason=low_inliers matches={} observations={} inliers={} min_inliers={}",
                                        frame_id.as_u64(),
                                        projected_match_count,
                                        projected_attempt.tracked_observations.len(),
                                        projected_attempt.result.inliers.len(),
                                        config.min_inliers,
                                    );
                                }
                            }
                            Err(err) => {
                                if self.trace_transitions {
                                    eprintln!(
                                        "projected tracking fallback frame={} reason={} matches={}",
                                        frame_id.as_u64(),
                                        err.trace_label(),
                                        projected_match_count,
                                    );
                                }
                            }
                        }
                    }
                    None => {
                        if self.trace_transitions {
                            eprintln!(
                                "projected tracking fallback frame={} reason=not_enough_projected_matches",
                                frame_id.as_u64()
                            );
                        }
                    }
                }
            }

            match attempt {
                Some(attempt) => attempt,
                None => {
                    let lightglue_matches = self.lightglue_tracking_matches(
                        current.clone(),
                        keyframe,
                        keyframe_id,
                        prefetched_matches,
                        frame_id,
                    )?;
                    match self.build_tracking_attempt(keyframe, keyframe_id, lightglue_matches) {
                        Ok(attempt) => attempt,
                        Err(TrackingAttemptError::NotEnoughMapPoints {
                            matches,
                            verified,
                            observations,
                            required_observations,
                        }) => {
                            if self.trace_transitions {
                                eprintln!(
                                    "tracking failure frame={} reason=not_enough_map_points matches={} verified={} observations={} required_observations={} current={}",
                                    frame_id.as_u64(),
                                    matches,
                                    verified,
                                    observations,
                                    required_observations,
                                    current.len()
                                );
                            }
                            let tracking_health = self.tracking_failure_health();
                            self.maybe_enter_relocalization(tracking_health, Arc::clone(&current));
                            let mut diagnostics = self.empty_diagnostics();
                            diagnostics.features_detected = Some(current.len());
                            diagnostics.features_matched = Some(matches);
                            diagnostics.tracking_time = Some(tracking_start.elapsed());
                            return Ok(self.tracking_failure_output(
                                frame_id,
                                tracking_health,
                                diagnostics,
                            ));
                        }
                        Err(TrackingAttemptError::PnpFailed {
                            matches,
                            verified,
                            observations,
                            required_inliers,
                        }) => {
                            if self.trace_transitions {
                                eprintln!(
                                    "tracking failure frame={} reason=pnp_failed observations={} matches={} verified={} required_inliers={}",
                                    frame_id.as_u64(),
                                    observations,
                                    matches,
                                    verified,
                                    required_inliers,
                                );
                            }
                            let tracking_health = self.tracking_failure_health();
                            self.maybe_enter_relocalization(tracking_health, Arc::clone(&current));
                            let mut diagnostics = self.empty_diagnostics();
                            diagnostics.features_detected = Some(current.len());
                            diagnostics.features_matched = Some(matches);
                            diagnostics.pnp_tracked_observations =
                                Some(crate::PnpTrackedObservationCountMetric::new(observations));
                            diagnostics.tracking_time = Some(tracking_start.elapsed());
                            return Ok(self.tracking_failure_output(
                                frame_id,
                                tracking_health,
                                diagnostics,
                            ));
                        }
                        Err(TrackingAttemptError::Fatal(err)) => return Err(err),
                    }
                }
            }
        };
        let TrackingAttempt {
            matches,
            verified,
            tracked_observations,
            tracking_ransac,
            result,
        } = attempt;

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
        let inlier_observations: Vec<_> = result
            .inliers
            .iter()
            .filter_map(|&idx| tracked_observations.observations.get(idx).copied())
            .collect();

        let parallax_px = median_parallax_px(
            &verified,
            &tracked_observations.verified_match_indices,
            &result.inliers,
        );
        let covisibility = if keyframe.landmarks().is_empty() {
            0.0
        } else {
            result.inliers.len() as f32 / keyframe.landmarks().len() as f32
        };

        let visual_pose_world = result.pose;
        #[cfg(feature = "vio")]
        let map_observations_for_authoritative_visual_pose = map_observations.clone();
        #[cfg(feature = "vio")]
        let refinement =
            self.run_vio_or_visual_ba(left.timestamp(), visual_pose_world, map_observations);
        #[cfg(not(feature = "vio"))]
        let refined_world = ObservationSet::new(map_observations, self.ba.min_observations())
            .ok()
            .and_then(|set| {
                self.ba
                    .push_frame(self.global_map.map(), visual_pose_world, set)
            });
        #[cfg(not(feature = "vio"))]
        let pose_world = refined_world.unwrap_or(visual_pose_world);
        #[cfg(not(feature = "vio"))]
        let tracking_pose_source = if refined_world.is_some() {
            crate::TrackingPoseSource::VisualBundleAdjustment
        } else {
            crate::TrackingPoseSource::VisualTracking
        };
        #[cfg(feature = "vio")]
        let intrinsics = self.frontend.intrinsics();
        #[cfg(feature = "vio")]
        let visual_proposal_tracked_metrics = PoseReprojectionMetrics::from_pose(
            &visual_pose_world,
            &tracked_observations.observations,
            intrinsics,
        );
        #[cfg(feature = "vio")]
        let visual_proposal_accepted_inlier_metrics = PoseReprojectionMetrics::from_pose(
            &visual_pose_world,
            &inlier_observations,
            intrinsics,
        );
        #[cfg(feature = "vio")]
        let (
            pose_world,
            tracking_pose_source,
            vio_proposal_disposition,
            vio_proposal_tracked_metrics,
            vio_proposal_accepted_inlier_metrics,
            vio_solve_result,
        ) = match refinement {
            PoseRefinementProposal::None => {
                self.commit_authoritative_visual_pose(
                    left.timestamp(),
                    visual_pose_world,
                    &map_observations_for_authoritative_visual_pose,
                );
                (
                    visual_pose_world,
                    crate::TrackingPoseSource::VisualTracking,
                    crate::VioProposalDisposition::NotRun,
                    None,
                    None,
                    None,
                )
            }
            PoseRefinementProposal::VisualBa(proposal) => {
                let visual_ba_accepted_inlier_metrics = PoseReprojectionMetrics::from_pose(
                    &proposal.pose_world,
                    &inlier_observations,
                    intrinsics,
                );
                if should_adopt_visual_ba_proposal(
                    &visual_proposal_accepted_inlier_metrics,
                    &visual_ba_accepted_inlier_metrics,
                ) {
                    let adopted_pose = proposal.pose_world;
                    self.commit_visual_ba_proposal(proposal);
                    self.commit_authoritative_visual_pose(
                        left.timestamp(),
                        adopted_pose,
                        &map_observations_for_authoritative_visual_pose,
                    );
                    (
                        adopted_pose,
                        crate::TrackingPoseSource::VisualBundleAdjustment,
                        crate::VioProposalDisposition::NotRun,
                        None,
                        None,
                        None,
                    )
                } else {
                    self.commit_authoritative_visual_pose(
                        left.timestamp(),
                        visual_pose_world,
                        &map_observations_for_authoritative_visual_pose,
                    );
                    (
                        visual_pose_world,
                        crate::TrackingPoseSource::VisualTracking,
                        crate::VioProposalDisposition::NotRun,
                        None,
                        None,
                        None,
                    )
                }
            }
            PoseRefinementProposal::Vio(proposal) => {
                let proposal = *proposal;
                let vio_tracked_metrics = PoseReprojectionMetrics::from_pose(
                    &proposal.pose_world,
                    &tracked_observations.observations,
                    intrinsics,
                );
                let vio_accepted_inlier_metrics = PoseReprojectionMetrics::from_pose(
                    &proposal.pose_world,
                    &inlier_observations,
                    intrinsics,
                );
                let disposition = decide_vio_pose_adoption(
                    proposal.solve_result.last_frame_visual_residual_count,
                    &visual_proposal_accepted_inlier_metrics,
                    &vio_accepted_inlier_metrics,
                );
                let vio_solve_result = proposal.solve_result.clone();
                let (adopted_pose, source) = if disposition
                    == crate::VioProposalDisposition::Adopted
                {
                    let adopted_pose = proposal.pose_world;
                    self.commit_vio_proposal(left.timestamp(), proposal);
                    (adopted_pose, crate::TrackingPoseSource::VioRefined)
                } else {
                    if self.trace_transitions {
                        eprintln!(
                            "vio proposal rejected disposition={disposition:?}; reanchoring inertial runtime to current visual pose"
                        );
                    }
                    self.commit_authoritative_visual_pose(
                        left.timestamp(),
                        visual_pose_world,
                        &map_observations_for_authoritative_visual_pose,
                    );
                    (visual_pose_world, crate::TrackingPoseSource::VisualTracking)
                };
                (
                    adopted_pose,
                    source,
                    disposition,
                    Some(vio_tracked_metrics),
                    Some(vio_accepted_inlier_metrics),
                    Some(vio_solve_result),
                )
            }
        };
        if self.consecutive_tracking_failures > 0 {
            self.emit_event(DiagnosticEvent::TrackingRecovered);
        }
        self.consecutive_tracking_failures = 0;
        let mut output_keyframe = None;
        let mut output_matches = None;
        let mut keyframe_status = None;
        let mut triangulation_stats = None;
        let mut ba_result = None;
        let keyframe_decision =
            self.config
                .keyframe_policy
                .decide(result.inliers.len(), parallax_px, covisibility);
        if self.trace_transitions {
            eprintln!(
                "tracking success frame={} observations={} missing_associations={} inliers={} required_inliers={} matches={} verified={} parallax_px={} covisibility={:.3} decision={:?}",
                frame_id.as_u64(),
                tracked_observations.len(),
                tracked_observations.missing_map_point_associations,
                result.inliers.len(),
                tracking_ransac.min_inliers(),
                matches.len(),
                verified.len(),
                parallax_px
                    .map(|value| format!("{value:.2}"))
                    .unwrap_or_else(|| "none".to_string()),
                covisibility,
                keyframe_decision,
            );
        }

        if let KeyframeDecision::Insert(reason) = keyframe_decision {
            let new_pose = pose_world;
            let shared = build_shared_matches(
                keyframe_id,
                &verified,
                &tracked_observations.verified_match_indices,
                &result.inliers,
            );
            let shared_pairs = shared.pairs.len();
            if self.trace_transitions {
                eprintln!(
                    "keyframe insertion frame={} reason={} shared_pairs={}",
                    frame_id.as_u64(),
                    reason.trace_label(),
                    shared_pairs
                );
            }
            match self.create_keyframe_internal(
                left,
                right,
                new_pose,
                Some(current.clone()),
                Some(shared),
            ) {
                Ok(created) => {
                    let CreatedKeyframe {
                        keyframe_id,
                        keyframe,
                        stereo_matches,
                        diagnostics: keyframe_diagnostics,
                    } = created;
                    keyframe_status = Some(KeyframeStatus::Created);
                    triangulation_stats = keyframe_diagnostics.triangulation;
                    ba_result = keyframe_diagnostics.ba_result.clone();
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
                        remove_keyframe_from_graph_and_db(&mut self.global_map, keyframe_id)?;
                        self.emit_event(DiagnosticEvent::KeyframeRemoved {
                            keyframe_id,
                            reason: KeyframeRemovalReason::Redundant,
                        });
                        if let Some(place_recognition) = self.place_recognition.as_mut() {
                            place_recognition.remove_keyframe(keyframe_id);
                        }
                        self.bump_map_version();
                    } else {
                        let window = self
                            .global_map
                            .covisible_window(keyframe_id, self.ba.window_size())?;
                        if window.len() >= 2 {
                            if self.backend.is_configured() {
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
                                            if let Some(supervisor) = self.backend.supervisor() {
                                                self.backend_stats.respawn_count =
                                                    supervisor.respawn_count();
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
                        output_matches = Some(stereo_matches);
                    }
                }
                Err(TrackerError::KeyframeRejected { landmarks }) => {
                    keyframe_status = Some(KeyframeStatus::Rejected);
                    if self.trace_transitions {
                        eprintln!(
                            "keyframe insertion rejected frame={} reason={} shared_pairs={} landmarks={}",
                            frame_id.as_u64(),
                            reason.trace_label(),
                            shared_pairs,
                            landmarks
                        );
                    }
                }
                Err(err) => return Err(err),
            }
        }

        #[cfg(not(feature = "vio"))]
        let intrinsics = self.frontend.intrinsics();
        let tracked_metrics = PoseReprojectionMetrics::from_pose(
            &pose_world,
            &tracked_observations.observations,
            intrinsics,
        );
        let projectable_tracked_observations = tracked_metrics.projectable_count();
        let accepted_inlier_metrics =
            PoseReprojectionMetrics::from_pose(&pose_world, &inlier_observations, intrinsics);
        // VIO visual correction will be handled by tightly-coupled BA (M2).

        let mut diagnostics = self.empty_diagnostics();
        diagnostics.pnp_inlier_ratio = Some(
            crate::PnpInlierRatioMetric::new(
                crate::PnpAcceptedInlierCountMetric::new(result.inliers.len()),
                crate::PnpTrackedObservationCountMetric::new(tracked_observations.len()),
            )
            .expect("tracker must emit a finite inlier ratio over non-empty tracked observations"),
        );
        diagnostics.pnp_tracked_observations = Some(crate::PnpTrackedObservationCountMetric::new(
            tracked_observations.len(),
        ));
        diagnostics.pnp_accepted_inliers = Some(crate::PnpAcceptedInlierCountMetric::new(
            result.inliers.len(),
        ));
        diagnostics.tracking_pose_source = Some(tracking_pose_source);
        diagnostics.pnp_projectable_tracked_observations =
            Some(crate::PnpProjectableTrackedObservationCountMetric::new(
                projectable_tracked_observations,
            ));
        diagnostics.ransac_iterations = Some(result.iterations);
        diagnostics.pnp_projectable_tracked_observation_reprojection_rmse_px =
            tracked_metrics.rmse_px().map(|value| {
                crate::PnpProjectableTrackedObservationPixelResidualMetric::new(value)
                    .expect("projectable tracked reprojection RMSE must be finite and non-negative")
            });
        diagnostics.pnp_projectable_tracked_observation_reprojection_max_px =
            tracked_metrics.max_px().map(|value| {
                crate::PnpProjectableTrackedObservationPixelResidualMetric::new(value)
                    .expect("projectable tracked reprojection max must be finite and non-negative")
            });
        diagnostics.pnp_projectable_tracked_observation_reprojection_mse_per_axis_px2 =
            tracked_metrics.mse_per_axis_px2().map(|value| {
                crate::PnpProjectableTrackedObservationReprojectionMsePerAxisPx2Metric::new(value)
                    .expect(
                        "projectable tracked reprojection MSE per axis must be finite and non-negative",
                    )
            });
        #[cfg(feature = "vio")]
        {
            diagnostics.visual_proposal_projectable_tracked_observations = Some(
                crate::VisualProposalProjectableTrackedObservationCountMetric::new(
                    visual_proposal_tracked_metrics.projectable_count(),
                ),
            );
            diagnostics.visual_proposal_projectable_tracked_observation_reprojection_rmse_px =
                visual_proposal_tracked_metrics.rmse_px().map(|value| {
                    crate::VisualProposalProjectableTrackedObservationPixelResidualMetric::new(
                        value,
                    )
                    .expect(
                        "visual proposal tracked reprojection RMSE must be finite and non-negative",
                    )
                });
            diagnostics.visual_proposal_projectable_accepted_inliers =
                Some(crate::PnpAcceptedInlierCountMetric::new(
                    visual_proposal_accepted_inlier_metrics.projectable_count(),
                ));
            diagnostics.visual_proposal_accepted_inlier_reprojection_rmse_px =
                visual_proposal_accepted_inlier_metrics.rmse_px().map(|value| {
                    crate::PnpAcceptedInlierPixelResidualMetric::new(value).expect(
                        "visual proposal accepted-inlier reprojection RMSE must be finite and non-negative",
                    )
                });
            diagnostics.vio_proposal_disposition = Some(vio_proposal_disposition);
            diagnostics.vio_solve_result = vio_solve_result;
            diagnostics.vio_calibrated_bias_prior_active = match &self.local_estimator {
                LocalEstimator::VisualOnly => None,
                LocalEstimator::Inertial(vio_runtime) => {
                    Some(vio_runtime.solve_config.has_anchor_bias_prior())
                }
            };
            if let Some(vio_tracked_metrics) = vio_proposal_tracked_metrics.as_ref() {
                let shared_metrics =
                    visual_proposal_tracked_metrics.shared_with(vio_tracked_metrics);
                diagnostics.vio_proposal_projectable_tracked_observations = Some(
                    crate::VioProposalProjectableTrackedObservationCountMetric::new(
                        vio_tracked_metrics.projectable_count(),
                    ),
                );
                diagnostics.vio_proposal_projectable_tracked_observation_reprojection_rmse_px =
                    vio_tracked_metrics.rmse_px().map(|value| {
                        crate::VioProposalProjectableTrackedObservationPixelResidualMetric::new(
                            value,
                        )
                        .expect(
                            "VIO proposal tracked reprojection RMSE must be finite and non-negative",
                        )
                    });
                diagnostics.shared_projectable_tracked_observations = Some(
                    crate::VisualVsVioSharedProjectableTrackedObservationCountMetric::new(
                        shared_metrics.count,
                    ),
                );
                diagnostics
                    .visual_proposal_shared_projectable_tracked_observation_reprojection_rmse_px =
                    shared_metrics.lhs_rmse_px.map(|value| {
                        crate::VisualVsVioSharedProjectableTrackedObservationPixelResidualMetric::new(
                            value,
                        )
                        .expect(
                            "visual shared tracked reprojection RMSE must be finite and non-negative",
                        )
                    });
                diagnostics
                    .vio_proposal_shared_projectable_tracked_observation_reprojection_rmse_px =
                    shared_metrics.rhs_rmse_px.map(|value| {
                        crate::VisualVsVioSharedProjectableTrackedObservationPixelResidualMetric::new(
                            value,
                        )
                        .expect(
                            "VIO shared tracked reprojection RMSE must be finite and non-negative",
                        )
                    });
            }
            if let Some(vio_accepted_inlier_metrics) = vio_proposal_accepted_inlier_metrics.as_ref()
            {
                let shared_accepted_inlier_metrics = visual_proposal_accepted_inlier_metrics
                    .shared_with(vio_accepted_inlier_metrics);
                diagnostics.vio_proposal_projectable_accepted_inliers =
                    Some(crate::PnpAcceptedInlierCountMetric::new(
                        vio_accepted_inlier_metrics.projectable_count(),
                    ));
                diagnostics.vio_proposal_accepted_inlier_reprojection_rmse_px =
                    vio_accepted_inlier_metrics.rmse_px().map(|value| {
                        crate::PnpAcceptedInlierPixelResidualMetric::new(value).expect(
                            "VIO proposal accepted-inlier reprojection RMSE must be finite and non-negative",
                        )
                    });
                diagnostics.shared_projectable_accepted_inliers = Some(
                    crate::PnpAcceptedInlierCountMetric::new(shared_accepted_inlier_metrics.count),
                );
                diagnostics.visual_proposal_shared_accepted_inlier_reprojection_rmse_px =
                    shared_accepted_inlier_metrics.lhs_rmse_px.map(|value| {
                        crate::PnpAcceptedInlierPixelResidualMetric::new(value).expect(
                            "visual proposal shared accepted-inlier reprojection RMSE must be finite and non-negative",
                        )
                    });
                diagnostics.vio_proposal_shared_accepted_inlier_reprojection_rmse_px =
                    shared_accepted_inlier_metrics.rhs_rmse_px.map(|value| {
                        crate::PnpAcceptedInlierPixelResidualMetric::new(value).expect(
                            "VIO proposal shared accepted-inlier reprojection RMSE must be finite and non-negative",
                        )
                    });
            }
        }
        diagnostics.pnp_inlier_reprojection_rmse_px =
            accepted_inlier_metrics.rmse_px().map(|value| {
                crate::PnpAcceptedInlierPixelResidualMetric::new(value)
                    .expect("reprojection RMSE must be finite and non-negative")
            });
        diagnostics.pnp_inlier_reprojection_max_px =
            accepted_inlier_metrics.max_px().map(|value| {
                crate::PnpAcceptedInlierPixelResidualMetric::new(value)
                    .expect("reprojection max must be finite and non-negative")
            });
        diagnostics.pnp_inlier_reprojection_mse_per_axis_px2 =
            accepted_inlier_metrics.mse_per_axis_px2().map(|value| {
                crate::PnpAcceptedInlierReprojectionMsePerAxisPx2Metric::new(value)
                    .expect("PnP inlier reprojection MSE per axis must be finite and non-negative")
            });
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
        #[cfg(feature = "vio")]
        let capture_time = left.timestamp();
        let created = match self.create_keyframe_internal(left, right, pose_world, None, None) {
            Ok(value) => value,
            Err(TrackerError::KeyframeRejected { landmarks }) => {
                if self.trace_transitions {
                    eprintln!(
                        "keyframe bootstrap rejected frame={} landmarks={} -> staying in NeedKeyframe",
                        frame_id.as_u64(),
                        landmarks
                    );
                }
                #[cfg(feature = "vio")]
                self.reset_inertial_runtime_continuity();
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
        let keyframe = Arc::clone(&created.keyframe);
        self.state = TrackerState::Tracking {
            keyframe,
            keyframe_id: created.keyframe_id,
        };
        self.ba.reset();
        #[cfg(feature = "vio")]
        self.commit_authoritative_visual_pose(capture_time, pose_world, &[]);
        self.consecutive_tracking_failures = 0;
        Ok(self.output_with_diagnostics(
            Some(pose_world),
            0,
            Some(created.keyframe),
            Some(created.stereo_matches),
            frame_id,
            TrackingHealth::Good,
            created.diagnostics,
        ))
    }

    fn create_keyframe_internal(
        &mut self,
        left: Frame,
        right: Frame,
        pose_world: Pose,
        left_det: Option<Arc<Detections>>,
        shared: Option<SharedMatches>,
    ) -> Result<CreatedKeyframe, TrackerError> {
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
        let depth_summary = summarize_depths(result.keyframe.landmarks());
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
        if self.trace_transitions {
            let shared_pairs = shared.as_ref().map_or(0, |shared| shared.pairs.len());
            match depth_summary {
                Some(depths) => eprintln!(
                    "keyframe created frame={} landmarks={} shared_pairs={} matches={} tri_kept={} tri_dropped_disparity={} tri_dropped_depth={} tri_dropped_duplicate={} depth_min_m={:.2} depth_median_m={:.2} depth_max_m={:.2}",
                    frame_id.as_u64(),
                    landmarks,
                    shared_pairs,
                    matches.len(),
                    triangulation_stats.kept,
                    triangulation_stats.dropped_disparity,
                    triangulation_stats.dropped_depth,
                    triangulation_stats.dropped_duplicate,
                    depths.min_m,
                    depths.median_m,
                    depths.max_m,
                ),
                None => eprintln!(
                    "keyframe created frame={} landmarks={} shared_pairs={} matches={} tri_kept={} tri_dropped_disparity={} tri_dropped_depth={} tri_dropped_duplicate={}",
                    frame_id.as_u64(),
                    landmarks,
                    shared_pairs,
                    matches.len(),
                    triangulation_stats.kept,
                    triangulation_stats.dropped_disparity,
                    triangulation_stats.dropped_depth,
                    triangulation_stats.dropped_duplicate,
                ),
            }
        }
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

        let mut diagnostics = self.empty_diagnostics();
        diagnostics.keyframe_status = Some(KeyframeStatus::Created);
        diagnostics.triangulation = Some(triangulation_stats);
        diagnostics.features_detected = Some(left_arc.len());
        diagnostics.features_matched = Some(matches.len());

        Ok(CreatedKeyframe {
            keyframe_id,
            keyframe,
            stereo_matches: matches,
            diagnostics,
        })
    }
}

fn insert_keyframe_into_map(
    map: &mut SlamMap,
    keyframe: &Arc<Keyframe>,
    timestamp: Timestamp,
    pose_world: Pose,
    shared: Option<&SharedMatches>,
) -> Result<KeyframeId, TrackerError> {
    let mut candidate = map.clone();
    let keyframe_id =
        insert_keyframe_into_candidate(&mut candidate, keyframe, timestamp, pose_world, shared)?;
    *map = candidate;
    Ok(keyframe_id)
}

fn insert_keyframe_into_candidate(
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
    let mut candidate = global_map.clone();
    candidate.remove_keyframe_from_graph(keyframe_id)?;
    candidate.remove_keyframe(keyframe_id)?;
    *global_map = candidate;
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

fn classify_pose_status(
    map_from_odom: &MapFromOdom,
    current_odom_pose: Option<Pose64>,
    visual_pose_map: Option<Pose>,
    last_accepted_pose: Option<&LastAcceptedPose>,
) -> PoseStatus {
    match (current_odom_pose, visual_pose_map) {
        (Some(cam_from_odom), visual_pose_map) => {
            let pose = tracking_pose_from_vio_output(map_from_odom, cam_from_odom, visual_pose_map);
            if visual_pose_map.is_some() {
                PoseStatus::Current(pose)
            } else {
                PoseStatus::Predicted(pose)
            }
        }
        (None, Some(pose_map)) => {
            let cam_from_map = Pose64::from_pose32(pose_map);
            let cam_from_odom = map_from_odom.map_to_odom(cam_from_map);
            PoseStatus::Current(TrackingPose::new(
                cam_from_odom,
                cam_from_map,
                Some(cam_from_map),
            ))
        }
        (None, None) => {
            last_accepted_pose.map_or(PoseStatus::Unavailable, |last| PoseStatus::Stale {
                pose: last.tracking_pose.clone(),
                source_frame_id: last.frame_id,
            })
        }
    }
}

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
    use crate::pose_graph::{EssentialEdge, EssentialEdgeKind, EssentialGraph, PoseGraphConfig};
    use crate::{
        CompactDescriptor, Descriptor, Detections, Keypoint, PlaceDescriptorExtractor, Point3,
        SensorId, Timestamp,
    };
    use std::error::Error as _;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
    use std::time::Duration;

    #[cfg(feature = "vio")]
    use crate::{MapFromOdom, Pose64};

    fn make_descriptor() -> Descriptor {
        Descriptor([0.0; 256])
    }

    #[test]
    fn tracker_error_preserves_inference_domain_source_chain() {
        let error = TrackerError::Inference(InferenceError::Frame(
            crate::FrameError::DimensionMismatch {
                expected: 4,
                actual: 3,
            },
        ));

        let inference = error.source().expect("inference source");
        let frame = inference.source().expect("frame source");
        assert_eq!(frame.to_string(), "dimension mismatch: expected 4, got 3");
        assert!(frame.source().is_none());
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
        let verified = matches
            .with_landmarks(map.instance_id(), keyframe_id, &keyframe)
            .expect("verified matches");
        let intrinsics =
            crate::test_helpers::make_pinhole_intrinsics(320, 240, 300.0, 300.0, 160.0, 120.0)
                .expect("intrinsics");

        let tracked = crate::frontend::build_map_observations(&map, &verified, intrinsics)
            .expect("tracked observations");

        assert_eq!(tracked.observations.len(), 4);
        assert_eq!(tracked.verified_match_indices, vec![0, 2, 3, 4]);
        assert_eq!(tracked.missing_map_point_associations, 1);

        let wrong_detections = Detections::new(
            SensorId::StereoLeft,
            FrameId::new(42),
            320,
            240,
            vec![Keypoint { x: 10.0, y: 10.0 }],
            vec![1.0],
            vec![make_descriptor()],
        )
        .expect("wrong detections");
        let wrong_keyframe_id = map
            .add_keyframe_from_detections(
                &wrong_detections,
                Timestamp::from_nanos(2),
                Pose::identity(),
            )
            .expect("wrong map keyframe");
        let wrong_provenance = matches
            .with_landmarks(map.instance_id(), wrong_keyframe_id, &keyframe)
            .expect("verified matches with wrong map id");
        assert!(matches!(
            crate::frontend::build_map_observations(&map, &wrong_provenance, intrinsics),
            Err(MapObservationError::KeyframeProvenanceMismatch { .. })
        ));

        let foreign_map = SlamMap::new();
        let foreign_provenance = matches
            .with_landmarks(foreign_map.instance_id(), keyframe_id, &keyframe)
            .expect("verified matches with foreign map id");
        assert!(matches!(
            crate::frontend::build_map_observations(&map, &foreign_provenance, intrinsics),
            Err(MapObservationError::MapInstanceMismatch { expected, actual })
                if expected == map.instance_id() && actual == foreign_map.instance_id()
        ));
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

        let candidate_detections = Detections::new(
            SensorId::StereoLeft,
            FrameId::new(11),
            keyframe.detections().width(),
            keyframe.detections().height(),
            keyframe.detections().keypoints().to_vec(),
            keyframe.detections().scores().to_vec(),
            keyframe.detections().descriptors().to_vec(),
        )
        .expect("candidate detections");
        let candidate_keyframe = Arc::new(
            Keyframe::from_arc(
                Arc::new(candidate_detections),
                keyframe.landmarks().to_vec(),
                keyframe.landmark_indices().to_vec(),
            )
            .expect("candidate keyframe"),
        );
        let invalid_shared = SharedMatches {
            keyframe_id,
            pairs: vec![(usize::MAX, 0)],
        };
        let generation_before = map.generation();
        let keyframes_before = map.num_keyframes();
        let points_before = map.num_points();

        let error = insert_keyframe_into_map(
            &mut map,
            &candidate_keyframe,
            Timestamp::from_nanos(43),
            Pose::identity(),
            Some(&invalid_shared),
        )
        .expect_err("invalid shared association must reject the candidate transaction");

        assert!(matches!(
            error,
            TrackerError::Map(crate::map::MapError::KeypointIndexOutOfBounds { .. })
        ));
        assert_eq!(map.generation(), generation_before);
        assert_eq!(map.num_keyframes(), keyframes_before);
        assert_eq!(map.num_points(), points_before);
        assert_map_invariants(&map).expect("failed insertion preserved map invariants");
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

        let backend = BackendSubsystem::Configured(supervisor);
        assert_eq!(backend.health_flags(), (true, false));
        assert!(backend.is_configured());
    }

    #[test]
    fn disabled_backend_remains_distinct_from_configured_failure() {
        let backend = BackendSubsystem::Disabled;
        assert_eq!(backend.health_flags(), (false, false));
        assert!(!backend.is_configured());
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
    fn loop_closure_failure_leaves_global_map_unchanged() {
        let (map, essential_graph, verified, query_kf, before_points) =
            make_loop_closure_apply_fixture();
        let mut global_map = GlobalMap::from_parts(map, essential_graph);
        let before_generation = global_map.map().generation();
        let before_query_pose = global_map.keyframe(query_kf).expect("query pose").pose();
        let before_loop_edges = global_map.essential_graph().snapshot().loop_edges;
        let loop_manager = LoopManager::new(PoseGraphConfig {
            max_iterations: 0,
            ..PoseGraphConfig::default()
        });

        let error = loop_manager
            .apply_verified_loop(&mut global_map, &verified)
            .expect_err("iteration exhaustion must reject the loop");

        assert!(matches!(
            error,
            TrackerError::PoseGraph(PoseGraphError::NotConverged { iterations: 0, .. })
        ));
        assert_eq!(global_map.map().generation(), before_generation);
        let after_query_pose = global_map.keyframe(query_kf).expect("query pose").pose();
        assert_eq!(after_query_pose.rotation(), before_query_pose.rotation());
        assert_eq!(
            after_query_pose.translation(),
            before_query_pose.translation()
        );
        assert_eq!(
            global_map.essential_graph().snapshot().loop_edges.len(),
            before_loop_edges.len()
        );
        for (point_id, before) in before_points {
            let after = global_map.point(point_id).expect("point").position();
            assert_eq!([after.x, after.y, after.z], [before.x, before.y, before.z]);
        }
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
    fn failed_keyframe_removal_preserves_essential_graph() {
        let mut map = SlamMap::new();
        let root = map
            .add_keyframe_from_detections(
                make_test_detections(1).as_ref(),
                Timestamp::from_nanos(1),
                Pose::identity(),
            )
            .expect("root keyframe");

        let mut other_map = SlamMap::new();
        other_map
            .add_keyframe_from_detections(
                make_test_detections(2).as_ref(),
                Timestamp::from_nanos(2),
                Pose::identity(),
            )
            .expect("other root");
        let foreign_id = other_map
            .add_keyframe_from_detections(
                make_test_detections(3).as_ref(),
                Timestamp::from_nanos(3),
                Pose::identity(),
            )
            .expect("foreign keyframe");

        let mut graph = EssentialGraph::new(1);
        graph.add_keyframe(root, None, &map);
        graph.add_loop_edge(EssentialEdge {
            a: root,
            b: foreign_id,
            kind: EssentialEdgeKind::Loop,
            relative_pose: crate::Pose64::identity(),
            information: [[0.0; 6]; 6],
        });
        let mut global_map = GlobalMap::from_parts(map, graph);
        let generation_before = global_map.map().generation();
        let loop_edges_before = global_map.essential_graph().snapshot().loop_edges.len();

        let error = remove_keyframe_from_graph_and_db(&mut global_map, foreign_id)
            .expect_err("foreign map ID must not be removable");

        assert!(matches!(
            error,
            TrackerError::Map(crate::map::MapError::KeyframeNotFound(id)) if id == foreign_id
        ));
        assert_eq!(global_map.map().generation(), generation_before);
        assert_eq!(
            global_map.essential_graph().snapshot().loop_edges.len(),
            loop_edges_before
        );
        assert_eq!(
            global_map.essential_graph().parent_of(foreign_id),
            Some(root)
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
    fn adaptive_tracking_ransac_relaxes_sparse_observation_sets() {
        let base = RansacConfig::default()
            .try_with_min_inliers(8)
            .expect("RANSAC config");

        let adaptive = |count| {
            adaptive_tracking_ransac_config(base, count)
                .expect("derived RANSAC config")
                .min_inliers()
        };
        assert_eq!(adaptive(4), 4);
        assert_eq!(adaptive(14), 4);
        assert_eq!(adaptive(19), 5);
        assert_eq!(adaptive(32), 8);
        assert_eq!(adaptive(200), 8);
    }

    #[test]
    fn relocalization_initial_session_requires_lost_tracking_and_enabled_config() {
        let detections = make_test_detections(900);
        assert!(
            SlamTracker::initial_relocalization_session(
                TrackingHealth::Good,
                true,
                Arc::clone(&detections),
                None,
            )
            .is_none()
        );
        assert!(
            SlamTracker::initial_relocalization_session(
                TrackingHealth::Lost,
                false,
                Arc::clone(&detections),
                None,
            )
            .is_none()
        );

        let session = SlamTracker::initial_relocalization_session(
            TrackingHealth::Lost,
            true,
            Arc::clone(&detections),
            None,
        )
        .expect("lost tracking should create relocalization session");
        assert_eq!(session.attempts, 0);
        assert!(matches!(session.phase, RelocalizationPhase::Searching));
    }

    #[cfg(feature = "vio")]
    #[test]
    fn relocalization_initial_session_preserves_reference_cam_from_odom() {
        let detections = make_test_detections(9001);
        let reference = Some(Pose64::from_pose32(Pose::from_rt(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [0.3, -0.2, 1.4],
        )));

        let session = SlamTracker::initial_relocalization_session(
            TrackingHealth::Lost,
            true,
            detections,
            reference,
        )
        .expect("session");

        assert_eq!(session.reference_cam_from_odom, reference);
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
                reference_cam_from_odom: None,
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
                reference_cam_from_odom: None,
            },
            detections,
        );
        assert!(matches!(give_up, TrackerState::NeedKeyframe));
    }

    #[cfg(feature = "vio")]
    #[test]
    fn relocalization_reference_cam_from_odom_prefers_session_reference() {
        let session_reference = Pose64::from_pose32(Pose::from_rt(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [1.0, 2.0, 3.0],
        ));
        let current_reference = Pose64::from_pose32(Pose::from_rt(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [-1.0, -2.0, -3.0],
        ));
        let session = RelocalizationSession {
            attempts: 0,
            phase: RelocalizationPhase::Searching,
            last_detections: make_test_detections(9002),
            reference_cam_from_odom: Some(session_reference),
        };

        assert_eq!(
            session.reference_cam_from_odom.or(Some(current_reference)),
            Some(session_reference)
        );
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
                reference_cam_from_odom: None,
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
                reference_cam_from_odom: None,
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

    #[test]
    fn pose_status_distinguishes_stale_snapshot_from_current_estimate() {
        let tracking_pose = TrackingPose::new(
            Pose64::identity(),
            Pose64::identity(),
            Some(Pose64::identity()),
        );
        let last = LastAcceptedPose {
            frame_id: FrameId::new(41),
            pose_world: Pose::identity(),
            tracking_pose,
        };

        let status = classify_pose_status(&MapFromOdom::identity(), None, None, Some(&last));
        assert!(status.current_estimate().is_none());
        assert!(status.last_known_pose().is_some());
        assert_eq!(status.stale_source_frame_id(), Some(FrameId::new(41)));
        assert!(matches!(status, PoseStatus::Stale { .. }));
    }

    #[test]
    fn pose_status_marks_inertial_only_output_as_predicted() {
        let cam_from_odom = Pose64::try_from_rt(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [0.4, -0.2, 0.1],
        )
        .expect("odom pose");
        let status =
            classify_pose_status(&MapFromOdom::identity(), Some(cam_from_odom), None, None);

        let PoseStatus::Predicted(pose) = status else {
            panic!("inertial-only output must be predicted");
        };
        assert_eq!(pose.cam_from_odom(), cam_from_odom);
        assert_eq!(pose.cam_from_map(), cam_from_odom);
        assert!(pose.cam_from_map_visual_measurement().is_none());
    }

    #[test]
    fn visual_output_without_prediction_preserves_map_and_odom_frames() {
        let mut bridge = MapFromOdom::identity();
        bridge.set_pose_map_from_odom(
            Pose64::try_from_rt(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [1.0, 0.0, 0.0],
            )
            .expect("map-from-odom"),
        );
        let visual_pose = Pose::from_rt(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [2.0, 0.0, 0.0],
        );
        let expected_map = Pose64::from_pose32(visual_pose);
        let expected_odom = bridge.map_to_odom(expected_map);

        let status = classify_pose_status(&bridge, None, Some(visual_pose), None);
        let PoseStatus::Current(pose) = status else {
            panic!("accepted visual output must be current");
        };
        assert_eq!(pose.cam_from_map(), expected_map);
        assert_eq!(pose.cam_from_odom(), expected_odom);
        assert_eq!(pose.cam_from_map_visual_measurement(), Some(expected_map));
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

    #[cfg(feature = "vio")]
    #[test]
    fn vio_camera_pose_conversion_round_trips_identity_extrinsics() {
        let cam_from_odom = Pose64::from_pose32(Pose::from_rt(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [1.5, -2.0, 0.3],
        ));
        let pose_odom_from_body =
            pose_odom_from_body_from_camera_pose(cam_from_odom, Pose64::identity());
        let recovered =
            camera_from_odom_from_pose_odom_from_body(pose_odom_from_body, Pose64::identity());

        assert_eq!(
            recovered.to_pose32().translation(),
            cam_from_odom.to_pose32().translation()
        );
        assert_eq!(
            recovered.to_pose32().rotation(),
            cam_from_odom.to_pose32().rotation()
        );
    }

    #[cfg(feature = "vio")]
    #[test]
    fn vio_camera_pose_conversion_round_trips_non_identity_extrinsics() {
        let cam_from_odom = Pose64::from_pose32(Pose::from_rt(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [1.5, -2.0, 0.3],
        ));
        let camera_from_body = Pose64::from_pose32(Pose::from_rt(
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            [0.04, -0.01, 0.02],
        ));
        let pose_odom_from_body =
            pose_odom_from_body_from_camera_pose(cam_from_odom, camera_from_body);
        let recovered =
            camera_from_odom_from_pose_odom_from_body(pose_odom_from_body, camera_from_body);

        assert_eq!(
            recovered.to_pose32().translation(),
            cam_from_odom.to_pose32().translation()
        );
        assert_eq!(
            recovered.to_pose32().rotation(),
            cam_from_odom.to_pose32().rotation()
        );
    }

    #[cfg(feature = "vio")]
    #[test]
    fn vio_pose_adoption_rejects_changed_projectable_support_even_when_counts_match() {
        let visual = PoseReprojectionMetrics::from_errors(vec![
            Some(1.0),
            Some(1.0),
            Some(1.0),
            Some(1.0),
            Some(1.0),
            None,
        ]);
        let vio = PoseReprojectionMetrics::from_errors(vec![
            None,
            Some(0.1),
            Some(0.1),
            Some(0.1),
            Some(0.1),
            Some(0.1),
        ]);
        assert_eq!(
            decide_vio_pose_adoption(5, &visual, &vio),
            crate::VioProposalDisposition::RejectedChangedAcceptedInlierProjectability
        );
    }

    #[cfg(feature = "vio")]
    #[test]
    fn vio_pose_adoption_rejects_missing_current_vio_observation_support() {
        let visual =
            PoseReprojectionMetrics::from_errors(vec![Some(1.0), Some(1.0), Some(1.0), Some(1.0)]);
        let vio =
            PoseReprojectionMetrics::from_errors(vec![Some(0.5), Some(0.5), Some(0.5), Some(0.5)]);

        assert_eq!(
            decide_vio_pose_adoption(0, &visual, &vio),
            crate::VioProposalDisposition::RejectedInsufficientCurrentVioObservationSupport
        );
    }

    #[cfg(feature = "vio")]
    #[test]
    fn visual_ba_adoption_requires_exact_projectable_support_match() {
        let visual = PoseReprojectionMetrics::from_errors(vec![
            Some(1.0),
            Some(1.0),
            Some(1.0),
            Some(1.0),
            Some(1.0),
            None,
        ]);
        let visual_ba = PoseReprojectionMetrics::from_errors(vec![
            None,
            Some(0.1),
            Some(0.1),
            Some(0.1),
            Some(0.1),
            Some(0.1),
        ]);
        assert!(!should_adopt_visual_ba_proposal(&visual, &visual_ba));
    }

    #[cfg(feature = "vio")]
    fn make_single_observation_set() -> ObservationSet {
        let (map, keyframe_id, _) = make_map_with_single_point();
        let keypoint = map.keyframe_keypoint(keyframe_id, 0).expect("keypoint ref");
        let pixel = map.keypoint(keypoint).expect("pixel");
        ObservationSet::new(
            vec![MapObservation::new(keypoint, pixel)],
            std::num::NonZeroUsize::new(1).expect("nonzero"),
        )
        .expect("observation set")
    }

    #[cfg(feature = "vio")]
    fn make_test_vio_intrinsics() -> crate::PinholeIntrinsics {
        crate::PinholeIntrinsics::try_from(&crate::dataset::CameraIntrinsics {
            fx: 420.0,
            fy: 418.0,
            cx: 320.0,
            cy: 240.0,
            width: 640,
            height: 480,
        })
        .expect("intrinsics")
    }

    #[cfg(feature = "vio")]
    fn make_test_vio_solve_config() -> crate::VioSolveConfig {
        let lm = crate::LmConfig::new(1e-3, 10.0, 1e-6, 1e6, 0.25, 0.75).expect("lm");
        let gravity = crate::Gravity::try_new([0.0, 9.81, 0.0]).expect("gravity");
        crate::VioSolveConfig::new(
            gravity,
            Pose64::identity(),
            make_test_vio_intrinsics(),
            lm,
            std::num::NonZeroUsize::new(3).expect("iters"),
            2.0,
            10.0,
            None,
        )
        .expect("solve config")
    }

    #[cfg(feature = "vio")]
    #[test]
    fn vio_runtime_visual_velocity_seed_uses_visual_pose_delta() {
        let runtime = VioRuntime {
            camera_from_body: Pose64::identity(),
            noise: crate::ImuNoiseModel::new(0.1, 0.01, 0.001, 0.0001).expect("noise"),
            pending_imu: crate::ImuAccumulator::new(),
            predicted_state: None,
            last_visual_measurement_body_odom: Some((
                Timestamp::from_nanos(1_000_000_000),
                Pose64::from_pose32(Pose::from_rt(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    [1.0, 2.0, 3.0],
                )),
            )),
            calibrated_bias: None,
            last_optimized_state: None,
            solve_config: make_test_vio_solve_config(),
            vio_window: None,
            max_window: 5,
        };

        let current_body_odom = Pose64::from_pose32(Pose::from_rt(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [2.5, 1.0, 5.0],
        ));
        let velocity = runtime
            .visual_velocity_seed(current_body_odom, Timestamp::from_nanos(2_000_000_000))
            .expect("velocity seed");

        assert!((velocity[0] - 1.5).abs() < 1e-12);
        assert!((velocity[1] + 1.0).abs() < 1e-12);
        assert!((velocity[2] - 2.0).abs() < 1e-12);
    }

    #[cfg(feature = "vio")]
    #[test]
    fn vio_runtime_set_capture_imu_interval_replaces_previous_interval() {
        let mut runtime = VioRuntime {
            camera_from_body: Pose64::identity(),
            noise: crate::ImuNoiseModel::new(0.1, 0.01, 0.001, 0.0001).expect("noise"),
            pending_imu: crate::ImuAccumulator::new(),
            predicted_state: None,
            last_visual_measurement_body_odom: None,
            calibrated_bias: None,
            last_optimized_state: None,
            solve_config: make_test_vio_solve_config(),
            vio_window: None,
            max_window: 5,
        };

        let first = crate::ImuBatch::new(vec![
            crate::ImuSample::new(Timestamp::from_nanos(10), [0.0; 3], [0.0; 3]).expect("imu 0"),
            crate::ImuSample::new(Timestamp::from_nanos(20), [1.0; 3], [2.0; 3]).expect("imu 1"),
        ])
        .expect("first batch");
        let second = crate::ImuBatch::new(vec![
            crate::ImuSample::new(Timestamp::from_nanos(30), [3.0; 3], [4.0; 3]).expect("imu 2"),
            crate::ImuSample::new(Timestamp::from_nanos(40), [5.0; 3], [6.0; 3]).expect("imu 3"),
        ])
        .expect("second batch");

        runtime
            .set_capture_imu_interval(Some(&first))
            .expect("set first interval");
        runtime
            .set_capture_imu_interval(Some(&second))
            .expect("replace with second interval");

        let pending = runtime
            .pending_imu
            .batch()
            .expect("pending batch")
            .expect("pending interval");
        assert_eq!(pending.len(), 2);
        assert_eq!(pending.start_time(), Timestamp::from_nanos(30));
        assert_eq!(pending.end_time(), Timestamp::from_nanos(40));
    }

    #[cfg(feature = "vio")]
    #[test]
    fn vio_runtime_set_capture_imu_interval_clears_on_absent_batch() {
        let mut runtime = VioRuntime {
            camera_from_body: Pose64::identity(),
            noise: crate::ImuNoiseModel::new(0.1, 0.01, 0.001, 0.0001).expect("noise"),
            pending_imu: crate::ImuAccumulator::new(),
            predicted_state: None,
            last_visual_measurement_body_odom: None,
            calibrated_bias: None,
            last_optimized_state: None,
            solve_config: make_test_vio_solve_config(),
            vio_window: None,
            max_window: 5,
        };

        let batch = crate::ImuBatch::new(vec![
            crate::ImuSample::new(Timestamp::from_nanos(10), [0.0; 3], [0.0; 3]).expect("imu 0"),
            crate::ImuSample::new(Timestamp::from_nanos(20), [1.0; 3], [2.0; 3]).expect("imu 1"),
        ])
        .expect("batch");

        runtime
            .set_capture_imu_interval(Some(&batch))
            .expect("set interval");
        runtime
            .set_capture_imu_interval(None)
            .expect("clear interval on absent capture imu");

        assert!(
            runtime.pending_imu.is_empty(),
            "absent capture IMU must not inherit the previous frame's interval"
        );
    }

    #[cfg(feature = "vio")]
    #[test]
    fn vio_runtime_visual_reanchor_replaces_stale_window_and_clears_pending_imu() {
        let observations = make_single_observation_set();
        let noise = crate::ImuNoiseModel::new(0.1, 0.01, 0.001, 0.0001).expect("noise");

        let stale_state = crate::NavState::try_new(
            Pose64::from_pose32(Pose::from_rt(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [0.0, 0.0, 0.0],
            )),
            [0.1, 0.2, 0.3],
            crate::ImuBias {
                accel: [0.01, 0.02, 0.03],
                gyro: [0.001, 0.002, 0.003],
            },
        )
        .expect("stale nav state");
        let replacement_state = crate::NavState::try_new(
            Pose64::from_pose32(Pose::from_rt(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [1.0, -0.5, 0.25],
            )),
            [0.4, 0.5, 0.6],
            crate::ImuBias {
                accel: [0.3, -0.1, 0.2],
                gyro: [0.01, -0.02, 0.03],
            },
        )
        .expect("replacement nav state");

        let stale_anchor = crate::local_ba::VioAnchor {
            synced: crate::local_ba::SyncedPose::new(stale_state.clone()),
            observations: Some(observations.clone()),
            anchor_velocity_odom_mps: stale_state.velocity_odom_mps(),
        };
        let replacement_anchor = crate::local_ba::VioAnchor {
            synced: crate::local_ba::SyncedPose::new(replacement_state.clone()),
            observations: Some(observations),
            anchor_velocity_odom_mps: replacement_state.velocity_odom_mps(),
        };

        let mut pending_imu = crate::ImuAccumulator::new();
        let pending_batch = crate::ImuBatch::new(vec![
            crate::ImuSample::new(Timestamp::from_nanos(10), [0.0, 9.81, 0.0], [0.0; 3])
                .expect("imu sample a"),
            crate::ImuSample::new(Timestamp::from_nanos(20), [0.0, 9.81, 0.0], [0.0; 3])
                .expect("imu sample b"),
        ])
        .expect("imu batch");
        pending_imu
            .extend_batch(&pending_batch)
            .expect("extend pending imu");

        let mut runtime = VioRuntime {
            camera_from_body: Pose64::identity(),
            noise,
            pending_imu,
            predicted_state: Some(stale_state.clone()),
            last_visual_measurement_body_odom: None,
            calibrated_bias: None,
            last_optimized_state: Some(stale_state.clone()),
            solve_config: make_test_vio_solve_config(),
            vio_window: Some(crate::local_ba::VioWindow {
                anchor: stale_anchor,
                successors: Vec::new(),
            }),
            max_window: 5,
        };

        runtime.commit_authoritative_visual_anchor(replacement_anchor);

        assert!(
            runtime.pending_imu.is_empty(),
            "current frame IMU interval must be consumed after visual reanchor"
        );
        let committed = runtime
            .last_optimized_state
            .as_ref()
            .expect("last optimized state");
        assert_eq!(
            committed.pose_odom_from_body().translation(),
            replacement_state.pose_odom_from_body().translation()
        );
        assert_eq!(
            committed.velocity_odom_mps(),
            replacement_state.velocity_odom_mps()
        );
        assert_eq!(committed.bias().accel, replacement_state.bias().accel);
        assert_eq!(committed.bias().gyro, replacement_state.bias().gyro);
        let predicted = runtime.predicted_state.as_ref().expect("predicted state");
        assert_eq!(
            predicted.pose_odom_from_body().translation(),
            replacement_state.pose_odom_from_body().translation()
        );
        let window = runtime.vio_window.as_ref().expect("vio window");
        assert_eq!(
            window.len(),
            1,
            "visual reanchor must replace the stale multi-frame window with a single authoritative anchor"
        );
        assert_eq!(
            window
                .anchor
                .synced
                .nav_state()
                .pose_odom_from_body()
                .translation(),
            replacement_state.pose_odom_from_body().translation()
        );
    }

    #[cfg(feature = "vio")]
    #[test]
    fn vio_runtime_commit_authoritative_pose_keeps_authoritative_bias_and_uses_pose_delta_velocity()
    {
        let previous_bias = crate::ImuBias {
            accel: [0.3, -0.1, 0.2],
            gyro: [0.01, -0.02, 0.03],
        };
        let previous_state = crate::NavState::try_new(
            Pose64::from_pose32(Pose::from_rt(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [0.5, -0.5, 1.0],
            )),
            [1.0, 2.0, 3.0],
            previous_bias.clone(),
        )
        .expect("previous nav state");
        let observations = make_single_observation_set();
        let mut pending_imu = crate::ImuAccumulator::new();
        let pending_batch = crate::ImuBatch::new(vec![
            crate::ImuSample::new(Timestamp::from_nanos(10), [0.0, 9.81, 0.0], [0.0; 3])
                .expect("imu sample a"),
            crate::ImuSample::new(Timestamp::from_nanos(20), [0.0, 9.81, 0.0], [0.0; 3])
                .expect("imu sample b"),
        ])
        .expect("imu batch");
        pending_imu
            .extend_batch(&pending_batch)
            .expect("extend pending imu");
        let mut runtime = VioRuntime {
            camera_from_body: Pose64::identity(),
            noise: crate::ImuNoiseModel::new(0.1, 0.01, 0.001, 0.0001).expect("noise"),
            pending_imu,
            predicted_state: Some(previous_state.clone()),
            last_visual_measurement_body_odom: Some((
                Timestamp::from_nanos(1_000_000_000),
                previous_state.pose_odom_from_body(),
            )),
            calibrated_bias: Some(crate::ImuBias {
                accel: [9.0, 9.0, 9.0],
                gyro: [9.0, 9.0, 9.0],
            }),
            last_optimized_state: Some(previous_state),
            solve_config: make_test_vio_solve_config(),
            vio_window: None,
            max_window: 5,
        };

        let body_odom = Pose64::from_pose32(Pose::from_rt(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [2.0, 0.0, 5.0],
        ));
        runtime.commit_authoritative_pose(
            Timestamp::from_nanos(2_000_000_000),
            body_odom,
            Some(observations),
        );

        assert!(runtime.pending_imu.is_empty());
        let committed = runtime
            .last_optimized_state
            .as_ref()
            .expect("committed state");
        assert_eq!(
            committed.pose_odom_from_body().translation(),
            [2.0, 0.0, 5.0]
        );
        assert_eq!(committed.bias().accel, previous_bias.accel);
        assert_eq!(committed.bias().gyro, previous_bias.gyro);
        assert!((committed.velocity_odom_mps()[0] - 1.5).abs() < 1e-12);
        assert!((committed.velocity_odom_mps()[1] - 0.5).abs() < 1e-12);
        assert!((committed.velocity_odom_mps()[2] - 4.0).abs() < 1e-12);
        let window = runtime.vio_window.as_ref().expect("vio window");
        assert_eq!(window.len(), 1);
        assert_eq!(
            window.anchor.synced.nav_state().bias().accel,
            previous_bias.accel
        );
        assert!(window.anchor.observations.is_some());
    }

    #[cfg(feature = "vio")]
    #[test]
    fn vio_runtime_commit_authoritative_pose_without_observations_keeps_authoritative_state() {
        let previous_state = crate::NavState::try_new(
            Pose64::from_pose32(Pose::from_rt(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [0.5, -0.5, 1.0],
            )),
            [1.0, 2.0, 3.0],
            crate::ImuBias {
                accel: [0.3, -0.1, 0.2],
                gyro: [0.01, -0.02, 0.03],
            },
        )
        .expect("previous nav state");
        let mut runtime = VioRuntime {
            camera_from_body: Pose64::identity(),
            noise: crate::ImuNoiseModel::new(0.1, 0.01, 0.001, 0.0001).expect("noise"),
            pending_imu: crate::ImuAccumulator::new(),
            predicted_state: Some(previous_state.clone()),
            last_visual_measurement_body_odom: Some((
                Timestamp::from_nanos(1_000_000_000),
                previous_state.pose_odom_from_body(),
            )),
            calibrated_bias: None,
            last_optimized_state: Some(previous_state.clone()),
            solve_config: make_test_vio_solve_config(),
            vio_window: Some(crate::local_ba::VioWindow {
                anchor: crate::local_ba::VioAnchor {
                    synced: crate::local_ba::SyncedPose::new(previous_state.clone()),
                    observations: Some(make_single_observation_set()),
                    anchor_velocity_odom_mps: previous_state.velocity_odom_mps(),
                },
                successors: Vec::new(),
            }),
            max_window: 5,
        };

        let body_odom = Pose64::from_pose32(Pose::from_rt(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [2.0, 0.0, 5.0],
        ));
        runtime.commit_authoritative_pose(Timestamp::from_nanos(2_000_000_000), body_odom, None);

        let committed = runtime
            .last_optimized_state
            .as_ref()
            .expect("committed state");
        assert_eq!(
            committed.pose_odom_from_body().translation(),
            [2.0, 0.0, 5.0]
        );
        assert!(runtime.vio_window.is_none());
        assert_eq!(
            runtime
                .last_visual_measurement_body_odom
                .expect("visual measurement")
                .0,
            Timestamp::from_nanos(2_000_000_000)
        );
    }

    #[cfg(feature = "vio")]
    #[test]
    fn vio_runtime_reset_runtime_continuity_clears_state_and_history() {
        let observations = make_single_observation_set();
        let nav_state = crate::NavState::try_new(
            Pose64::identity(),
            [0.1, 0.2, 0.3],
            crate::ImuBias::default(),
        )
        .expect("nav state");
        let mut pending_imu = crate::ImuAccumulator::new();
        let pending_batch = crate::ImuBatch::new(vec![
            crate::ImuSample::new(Timestamp::from_nanos(10), [0.0, 9.81, 0.0], [0.0; 3])
                .expect("imu sample a"),
            crate::ImuSample::new(Timestamp::from_nanos(20), [0.0, 9.81, 0.0], [0.0; 3])
                .expect("imu sample b"),
        ])
        .expect("imu batch");
        pending_imu
            .extend_batch(&pending_batch)
            .expect("extend pending imu");
        let mut runtime = VioRuntime {
            camera_from_body: Pose64::identity(),
            noise: crate::ImuNoiseModel::new(0.1, 0.01, 0.001, 0.0001).expect("noise"),
            pending_imu,
            predicted_state: Some(nav_state.clone()),
            last_visual_measurement_body_odom: Some((
                Timestamp::from_nanos(100),
                Pose64::identity(),
            )),
            calibrated_bias: None,
            last_optimized_state: Some(nav_state.clone()),
            solve_config: make_test_vio_solve_config(),
            vio_window: Some(crate::local_ba::VioWindow {
                anchor: crate::local_ba::VioAnchor {
                    synced: crate::local_ba::SyncedPose::new(nav_state),
                    observations: Some(observations),
                    anchor_velocity_odom_mps: [0.1, 0.2, 0.3],
                },
                successors: Vec::new(),
            }),
            max_window: 5,
        };

        runtime.reset_runtime_continuity();

        assert!(runtime.pending_imu.is_empty());
        assert!(runtime.predicted_state.is_none());
        assert!(runtime.last_visual_measurement_body_odom.is_none());
        assert!(runtime.last_optimized_state.is_none());
        assert!(runtime.vio_window.is_none());
    }
}
