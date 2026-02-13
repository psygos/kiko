use std::cmp::Ordering;
use std::collections::HashSet;
use std::num::{NonZeroU64, NonZeroUsize};
use std::sync::Arc;
use std::thread;

use crate::{
    map::{KeyframeId, MapPointId, SlamMap},
    solve_pnp_ransac, Detections, DownscaleFactor, Frame, FrameId, Keyframe, KeypointLimit,
    LightGlue, LocalBaConfig, LocalBundleAdjuster, MapObservation, Matches, ObservationSet,
    PinholeIntrinsics, Point3, Pose, RansacConfig, Raw, RectifiedStereo, StereoPair, SuperPoint,
    Timestamp, TriangulationConfig, TriangulationError, Triangulator, Verified,
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
}

impl TrackerConfig {
    pub fn max_keypoints(&self) -> usize {
        self.max_keypoints.get()
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

#[derive(Debug)]
struct KeyframeEvent {
    request_id: BackendRequestId,
    map_version: MapVersion,
    trigger_keyframe: KeyframeId,
    window: BackendWindow,
    map_snapshot: SlamMap,
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
        if !window.as_slice().iter().any(|&id| id == trigger_keyframe) {
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
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct PoseCorrection {
    keyframe_id: KeyframeId,
    pose: Pose,
}

#[derive(Clone, Copy, Debug)]
struct LandmarkCorrection {
    point_id: MapPointId,
    position: Point3,
}

#[derive(Debug)]
struct CorrectionEvent {
    request_id: BackendRequestId,
    map_version: MapVersion,
    trigger_keyframe: KeyframeId,
    pose_corrections: Vec<PoseCorrection>,
    landmark_corrections: Vec<LandmarkCorrection>,
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
    ) -> Result<Self, CorrectionBuildError> {
        let mut pose_corrections = Vec::with_capacity(event.window.as_slice().len());
        for &keyframe_id in event.window.as_slice() {
            let entry = optimized_map
                .keyframe(keyframe_id)
                .ok_or(CorrectionBuildError::MissingKeyframe { keyframe_id })?;
            pose_corrections.push(PoseCorrection {
                keyframe_id,
                pose: entry.pose(),
            });
        }

        let point_ids = collect_window_points(optimized_map, &event.window)?;
        let mut landmark_corrections = Vec::with_capacity(point_ids.len());
        for point_id in point_ids {
            let point = optimized_map
                .point(point_id)
                .ok_or(CorrectionBuildError::MissingMapPoint { point_id })?;
            landmark_corrections.push(LandmarkCorrection {
                point_id,
                position: point.position(),
            });
        }

        Ok(Self {
            request_id: event.request_id,
            map_version: event.map_version,
            trigger_keyframe: event.trigger_keyframe,
            pose_corrections,
            landmark_corrections,
        })
    }
}

#[derive(Debug)]
enum BackendWorkerError {
    OptimizationFailed { keyframe_id: KeyframeId },
    BuildCorrection(CorrectionBuildError),
}

impl std::fmt::Display for BackendWorkerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendWorkerError::OptimizationFailed { keyframe_id } => write!(
                f,
                "backend optimization failed for trigger keyframe {keyframe_id:?}"
            ),
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

        thread::spawn(move || {
            let mut ba = LocalBundleAdjuster::new(intrinsics, ba_config);
            while let Ok(event) = rx_req.recv() {
                let mut optimized_map = event.map_snapshot.clone();
                if !ba.optimize_keyframe_window(&mut optimized_map, event.window.as_slice()) {
                    let _ = tx_resp.send(BackendResponse::Failure {
                        request_id: event.request_id,
                        map_version: event.map_version,
                        error: BackendWorkerError::OptimizationFailed {
                            keyframe_id: event.trigger_keyframe,
                        },
                    });
                    continue;
                }
                match CorrectionEvent::from_optimized_map(&event, &optimized_map) {
                    Ok(correction) => {
                        let _ = tx_resp.send(BackendResponse::Correction(correction));
                    }
                    Err(err) => {
                        let _ = tx_resp.send(BackendResponse::Failure {
                            request_id: event.request_id,
                            map_version: event.map_version,
                            error: BackendWorkerError::BuildCorrection(err),
                        });
                    }
                }
            }
        });

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

    fn try_recv(&self) -> Option<BackendResponse> {
        match self.rx.try_recv() {
            Ok(response) => Some(response),
            Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => None,
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
        if !min_covisibility.is_finite() || min_covisibility < 0.0 || min_covisibility > 1.0 {
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
        if !max_covisibility.is_finite() || max_covisibility < 0.0 || max_covisibility > 1.0 {
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
    KeyframeRejected { landmarks: usize },
}

impl std::fmt::Display for TrackerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrackerError::Inference(err) => write!(f, "inference error: {err}"),
            TrackerError::Triangulation(err) => write!(f, "triangulation error: {err}"),
            TrackerError::Pnp(err) => write!(f, "pnp error: {err}"),
            TrackerError::Map(err) => write!(f, "map error: {err}"),
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrackingHealth {
    Good,
    Degraded,
    Lost,
}

#[derive(Debug)]
pub struct TrackerOutput {
    pub pose: Option<Pose>,
    pub inliers: usize,
    pub keyframe: Option<Arc<Keyframe>>,
    pub stereo_matches: Option<Matches<Raw>>,
    pub frame_id: FrameId,
    pub health: TrackingHealth,
}

#[derive(Debug)]
enum TrackerState {
    NeedKeyframe,
    Tracking {
        keyframe: Arc<Keyframe>,
        keyframe_id: KeyframeId,
    },
}

#[derive(Debug)]
struct SharedMatches {
    keyframe_id: KeyframeId,
    pairs: Vec<(usize, usize)>,
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
    map_version: MapVersion,
    backend: Option<BackendWorker>,
    backend_stats: BackendStats,
    consecutive_tracking_failures: usize,
}

impl SlamTracker {
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
            .map(|backend_cfg| BackendWorker::spawn(backend_cfg, intrinsics, config.ba));
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
            map_version: MapVersion::initial(),
            backend,
            backend_stats: BackendStats::default(),
            consecutive_tracking_failures: 0,
        }
    }

    pub fn process(&mut self, pair: StereoPair) -> Result<TrackerOutput, TrackerError> {
        self.drain_backend_responses();
        let tracking = match &self.state {
            TrackerState::NeedKeyframe => None,
            TrackerState::Tracking {
                keyframe,
                keyframe_id,
            } => Some((Arc::clone(keyframe), *keyframe_id)),
        };

        if let Some((keyframe, keyframe_id)) = tracking {
            self.track(pair, &keyframe, keyframe_id)
        } else {
            self.create_keyframe(pair, Pose::identity())
        }
    }

    pub fn covisibility_snapshot(&self) -> crate::map::CovisibilitySnapshot {
        self.map.covisibility_snapshot()
    }

    pub fn backend_stats(&self) -> BackendStats {
        self.backend_stats
    }

    fn bump_map_version(&mut self) {
        self.map_version = self.map_version.next();
    }

    fn submit_backend_event(
        &mut self,
        trigger_keyframe: KeyframeId,
        window_ids: Vec<KeyframeId>,
    ) -> Result<(), SubmitEventError> {
        let Some(worker) = self.backend.as_mut() else {
            return Err(SubmitEventError::Disconnected);
        };

        let window = BackendWindow::try_new(window_ids).map_err(SubmitEventError::InvalidWindow)?;
        let request_id = worker.next_request_id();
        let event = KeyframeEvent::try_new(
            request_id,
            self.map_version,
            trigger_keyframe,
            window,
            self.map.clone(),
        )
        .map_err(SubmitEventError::InvalidEvent)?;
        worker.try_submit(event)?;
        self.backend_stats.submitted = self.backend_stats.submitted.saturating_add(1);
        Ok(())
    }

    fn drain_backend_responses(&mut self) {
        loop {
            let response = {
                let Some(worker) = self.backend.as_ref() else {
                    return;
                };
                worker.try_recv()
            };

            let Some(response) = response else {
                break;
            };
            match response {
                BackendResponse::Correction(correction) => {
                    match apply_correction_event(&mut self.map, self.map_version, &correction) {
                        Ok(()) => {
                            self.bump_map_version();
                            self.backend_stats.applied =
                                self.backend_stats.applied.saturating_add(1);
                        }
                        Err(ApplyCorrectionError::StaleVersion { .. }) => {
                            self.backend_stats.stale = self.backend_stats.stale.saturating_add(1);
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
            }
        }
    }

    fn tracking_failure_health(&mut self) -> TrackingHealth {
        const LOST_AFTER_CONSECUTIVE_FAILURES: usize = 3;
        self.consecutive_tracking_failures = self.consecutive_tracking_failures.saturating_add(1);
        if self.consecutive_tracking_failures >= LOST_AFTER_CONSECUTIVE_FAILURES {
            TrackingHealth::Lost
        } else {
            TrackingHealth::Degraded
        }
    }

    fn track(
        &mut self,
        pair: StereoPair,
        keyframe: &Arc<Keyframe>,
        keyframe_id: KeyframeId,
    ) -> Result<TrackerOutput, TrackerError> {
        let StereoPair { left, right } = pair;
        let frame_id = left.frame_id();

        let current = self
            .superpoint_left
            .detect_with_downscale(&left, self.config.downscale)?
            .top_k(self.config.max_keypoints());
        let current = Arc::new(current);

        let matches = if current.is_empty() || keyframe.detections().is_empty() {
            return Ok(TrackerOutput {
                pose: None,
                inliers: 0,
                keyframe: None,
                stereo_matches: None,
                frame_id,
                health: self.tracking_failure_health(),
            });
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
                return Ok(TrackerOutput {
                    pose: None,
                    inliers: 0,
                    keyframe: None,
                    stereo_matches: None,
                    frame_id,
                    health: self.tracking_failure_health(),
                });
            }
            Err(err) => return Err(TrackerError::Pnp(err)),
        };

        let result = match solve_pnp_ransac(&observations, self.intrinsics, self.config.ransac) {
            Ok(result) => result,
            Err(crate::PnpError::NotEnoughPoints { .. } | crate::PnpError::NoSolution) => {
                return Ok(TrackerOutput {
                    pose: None,
                    inliers: 0,
                    keyframe: None,
                    stereo_matches: None,
                    frame_id,
                    health: self.tracking_failure_health(),
                });
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
        self.consecutive_tracking_failures = 0;

        let mut output = TrackerOutput {
            pose: Some(pose_world),
            inliers: result.inliers.len(),
            keyframe: None,
            stereo_matches: None,
            frame_id,
            health: TrackingHealth::Good,
        };

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
                        let _ = self.map.remove_keyframe(keyframe_id);
                        self.bump_map_version();
                    } else {
                        let window = self
                            .map
                            .covisible_window(keyframe_id, self.ba.window_size())?;
                        if self.backend.is_some() {
                            if let Err(err) = self.submit_backend_event(keyframe_id, window.clone())
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
                                        self.backend = None;
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
                                if self.ba.optimize_keyframe_window(&mut self.map, &window) {
                                    self.bump_map_version();
                                }
                            }
                        } else if self.ba.optimize_keyframe_window(&mut self.map, &window) {
                            self.bump_map_version();
                        }
                        self.state = TrackerState::Tracking {
                            keyframe: keyframe.clone(),
                            keyframe_id,
                        };
                        self.ba.reset();
                        output.keyframe = Some(keyframe);
                        output.stereo_matches = keyframe_output.stereo_matches;
                    }
                }
            }
        }

        Ok(output)
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
        Ok(TrackerOutput {
            pose: Some(pose_world),
            inliers: 0,
            keyframe: output.keyframe,
            stereo_matches: output.stereo_matches,
            frame_id,
            health: TrackingHealth::Good,
        })
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
        self.bump_map_version();

        Ok((
            TrackerOutput {
                pose: None,
                inliers: 0,
                keyframe: Some(keyframe),
                stereo_matches: Some(matches),
                frame_id,
                health: TrackingHealth::Good,
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
        let descriptor = keyframe.detections().descriptors()[det_idx];
        let world = camera_to_world(pose_world, *landmark);
        map.add_map_point(world, descriptor, keypoint_ref)?;
    }
    Ok(keyframe_id)
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

    for pose_correction in &correction.pose_corrections {
        if map.keyframe(pose_correction.keyframe_id).is_none() {
            return Err(ApplyCorrectionError::MissingKeyframe {
                keyframe_id: pose_correction.keyframe_id,
            });
        }
    }
    for landmark_correction in &correction.landmark_corrections {
        if map.point(landmark_correction.point_id).is_none() {
            return Err(ApplyCorrectionError::MissingMapPoint {
                point_id: landmark_correction.point_id,
            });
        }
    }

    for pose_correction in &correction.pose_corrections {
        map.set_keyframe_pose(pose_correction.keyframe_id, pose_correction.pose)?;
    }
    for landmark_correction in &correction.landmark_corrections {
        map.set_map_point_position(landmark_correction.point_id, landmark_correction.position)?;
    }
    Ok(())
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
    use crate::{Descriptor, Detections, Keypoint, Point3, SensorId, Timestamp};

    fn make_descriptor() -> Descriptor {
        Descriptor([0.0; 256])
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
                make_descriptor(),
                keypoint,
            )
            .expect("map point");
        (map, keyframe_id, point_id)
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
            pose_corrections: vec![PoseCorrection {
                keyframe_id,
                pose: Pose::identity(),
            }],
            landmark_corrections: vec![LandmarkCorrection {
                point_id,
                position: Point3 {
                    x: 1.0,
                    y: 2.0,
                    z: 3.0,
                },
            }],
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
        let correction = CorrectionEvent {
            request_id: BackendRequestId(NonZeroU64::new(2).expect("non-zero")),
            map_version: MapVersion::initial(),
            trigger_keyframe: keyframe_id,
            pose_corrections: vec![PoseCorrection {
                keyframe_id,
                pose: corrected_pose,
            }],
            landmark_corrections: vec![LandmarkCorrection {
                point_id,
                position: corrected_point,
            }],
        };

        apply_correction_event(&mut map, MapVersion::initial(), &correction)
            .expect("correction apply");
        assert_map_invariants(&map).expect("post-correction invariants");

        let stored_pose = map.keyframe(keyframe_id).expect("keyframe").pose();
        assert_eq!(stored_pose.translation(), corrected_pose.translation());
        assert_eq!(stored_pose.rotation(), corrected_pose.rotation());

        let stored_point = map.point(point_id).expect("map point").position();
        assert_eq!(stored_point.x, corrected_point.x);
        assert_eq!(stored_point.y, corrected_point.y);
        assert_eq!(stored_point.z, corrected_point.z);
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
