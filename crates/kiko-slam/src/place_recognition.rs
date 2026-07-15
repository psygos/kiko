use std::collections::{HashSet, VecDeque};
use std::ffi::OsString;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;

use crate::loop_closure::{
    DescriptorSource, GlobalDescriptor, GlobalDescriptorError, KeyframeDatabase,
    KeyframeDatabaseError, LoopClosureConfig, PlaceMatch, RelocalizationMatch,
    aggregate_global_descriptor,
};
use crate::map::KeyframeId;
use crate::{
    Detections, EigenPlaces, Frame, GlobalDescriptorConfig, InferenceError,
    PlaceDescriptorExtractor,
};
use crossbeam_channel::{Receiver, Sender, TryRecvError, TrySendError};

use crate::MapSnapshot;

const EIGENPLACES_MODEL_ENV: &str = "KIKO_EIGENPLACES_MODEL";

#[derive(Clone, Copy, Debug, Default)]
pub struct DescriptorStats {
    pub submitted: u64,
    pub dropped_full: u64,
    pub dropped_unavailable: u64,
    pub applied: u64,
    pub worker_failures: u64,
    pub restart_failures: u64,
    pub respawn_count: u32,
    pub respawn_exhausted: bool,
    pub panics: u64,
}

#[derive(Debug)]
pub enum BootstrapDescriptorError {
    Aggregation { source: GlobalDescriptorError },
    Database { source: KeyframeDatabaseError },
}

impl std::fmt::Display for BootstrapDescriptorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Aggregation { source } => {
                write!(f, "bootstrap descriptor aggregation failed: {source}")
            }
            Self::Database { source } => {
                write!(
                    f,
                    "bootstrap descriptor database operation failed: {source}"
                )
            }
        }
    }
}

impl std::error::Error for BootstrapDescriptorError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Aggregation { source } => Some(source),
            Self::Database { source } => Some(source),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct PendingLoopCandidate {
    pub(crate) query_kf: KeyframeId,
    pub(crate) detections: Arc<Detections>,
    pub(crate) candidates: Vec<PlaceMatch>,
}

#[derive(Debug)]
pub(crate) struct PreparedKeyframeRecognition {
    database: KeyframeDatabase,
    pending_loop: Option<PendingLoopCandidate>,
    loop_streak: std::collections::HashMap<KeyframeId, usize>,
}

#[derive(Debug, Clone)]
pub(crate) struct DescriptorRequest {
    pub(crate) keyframe_id: KeyframeId,
    pub(crate) source_snapshot: MapSnapshot,
    pub(crate) frame: Frame,
}

#[derive(Debug, Clone)]
pub(crate) struct DescriptorResponse {
    pub(crate) keyframe_id: KeyframeId,
    pub(crate) source_snapshot: MapSnapshot,
    pub(crate) descriptor: GlobalDescriptor,
}

#[derive(Debug)]
pub(crate) enum DescriptorWorkerResponse {
    Descriptor(Box<DescriptorResponse>),
    Failure {
        keyframe_id: KeyframeId,
        source_snapshot: MapSnapshot,
        error: InferenceError,
    },
    WorkerPanic {
        keyframe_id: KeyframeId,
        source_snapshot: MapSnapshot,
        message: String,
    },
}

#[derive(Debug)]
pub(crate) enum SubmitDescriptorError {
    QueueFull,
    Unavailable {
        exhausted: bool,
        source: Option<Arc<DescriptorInitError>>,
    },
}

impl std::fmt::Display for SubmitDescriptorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::QueueFull => write!(f, "descriptor request queue is full"),
            Self::Unavailable {
                exhausted,
                source: Some(source),
            } => write!(
                f,
                "descriptor worker is unavailable (restart_exhausted={exhausted}): {source}"
            ),
            Self::Unavailable {
                exhausted,
                source: None,
            } => write!(
                f,
                "descriptor worker is unavailable (restart_exhausted={exhausted})"
            ),
        }
    }
}

impl std::error::Error for SubmitDescriptorError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Unavailable {
                source: Some(source),
                ..
            } => Some(source.as_ref()),
            Self::QueueFull | Self::Unavailable { source: None, .. } => None,
        }
    }
}

pub(crate) type DescriptorExtractorFactory =
    Arc<dyn Fn() -> Result<Box<dyn PlaceDescriptorExtractor>, DescriptorInitError> + Send + Sync>;

#[derive(Debug)]
pub enum DescriptorInitError {
    EmptyModelPath {
        variable: &'static str,
    },
    ModelMissing {
        path: PathBuf,
    },
    ModelLoad {
        path: PathBuf,
        source: InferenceError,
    },
    WorkerThread {
        source: std::io::Error,
    },
}

impl std::fmt::Display for DescriptorInitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyModelPath { variable } => {
                write!(
                    f,
                    "{variable} must not be empty when learned descriptors are enabled"
                )
            }
            Self::ModelMissing { path } => {
                write!(f, "descriptor model does not exist at {}", path.display())
            }
            Self::ModelLoad { path, source } => write!(
                f,
                "failed to initialize descriptor model at {}: {source}",
                path.display()
            ),
            Self::WorkerThread { source } => {
                write!(f, "failed to spawn descriptor worker thread: {source}")
            }
        }
    }
}

impl std::error::Error for DescriptorInitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ModelLoad { source, .. } => Some(source),
            Self::WorkerThread { source } => Some(source),
            Self::EmptyModelPath { .. } | Self::ModelMissing { .. } => None,
        }
    }
}

pub(crate) struct DescriptorWorker {
    tx: Sender<DescriptorRequest>,
    rx: Receiver<DescriptorWorkerResponse>,
    _thread: thread::JoinHandle<()>,
}

#[derive(Debug)]
pub(crate) enum DescriptorWorkerSubmitError {
    QueueFull(DescriptorRequest),
    Disconnected(DescriptorRequest),
}

impl DescriptorWorker {
    fn model_path_from_override(
        override_path: Option<OsString>,
    ) -> Result<PathBuf, DescriptorInitError> {
        if let Some(path) = override_path {
            if path.is_empty() {
                return Err(DescriptorInitError::EmptyModelPath {
                    variable: EIGENPLACES_MODEL_ENV,
                });
            }
            return Ok(path.into());
        }
        Ok(PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("models")
            .join("eigenplaces.onnx"))
    }

    fn model_path() -> Result<PathBuf, DescriptorInitError> {
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
                            error: err,
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
    pub(crate) fn spawn_with_extractor(
        config: GlobalDescriptorConfig,
        extractor: Box<dyn PlaceDescriptorExtractor>,
    ) -> Result<Self, std::io::Error> {
        Self::spawn(config, extractor)
    }

    pub(crate) fn submit(
        &self,
        request: DescriptorRequest,
    ) -> Result<(), DescriptorWorkerSubmitError> {
        match self.tx.try_send(request) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(request)) => {
                Err(DescriptorWorkerSubmitError::QueueFull(request))
            }
            Err(TrySendError::Disconnected(request)) => {
                Err(DescriptorWorkerSubmitError::Disconnected(request))
            }
        }
    }

    pub(crate) fn try_recv(&self) -> Result<Option<DescriptorWorkerResponse>, ()> {
        match self.rx.try_recv() {
            Ok(value) => Ok(Some(value)),
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => Err(()),
        }
    }

    #[cfg(test)]
    fn is_finished(&self) -> bool {
        self._thread.is_finished()
    }
}

pub(crate) struct DescriptorSupervisor {
    worker: Option<DescriptorWorker>,
    config: GlobalDescriptorConfig,
    factory: DescriptorExtractorFactory,
    respawn_count: u32,
    max_respawns: u32,
    spawn_exhausted: bool,
    last_spawn_error: Option<Arc<DescriptorInitError>>,
    pending_outputs: VecDeque<DescriptorSupervisorOutput>,
}

#[derive(Clone, Debug)]
pub(crate) struct DescriptorRestartFailure {
    pub(crate) respawn_count: u32,
    pub(crate) max_respawns: u32,
    pub(crate) exhausted: bool,
    pub(crate) error: Arc<DescriptorInitError>,
}

#[derive(Debug)]
pub(crate) enum DescriptorSupervisorOutput {
    Worker(DescriptorWorkerResponse),
    RestartFailure(DescriptorRestartFailure),
}

impl DescriptorSupervisor {
    fn default_factory() -> Result<DescriptorExtractorFactory, DescriptorInitError> {
        let path = DescriptorWorker::model_path()?;
        let backend = crate::InferenceBackend::auto();
        Ok(Arc::new(move || {
            let extractor = EigenPlaces::new_with_backend(&path, backend).map_err(|source| {
                DescriptorInitError::ModelLoad {
                    path: path.clone(),
                    source,
                }
            })?;
            Ok(Box::new(extractor) as Box<dyn PlaceDescriptorExtractor>)
        }))
    }

    fn spawn_with_max_respawns(
        config: GlobalDescriptorConfig,
        max_respawns: u32,
    ) -> Result<Self, DescriptorInitError> {
        let factory = Self::default_factory()?;
        let worker = Self::spawn_worker(config, &factory)?;
        Ok(Self {
            worker: Some(worker),
            config,
            factory,
            respawn_count: 0,
            max_respawns,
            spawn_exhausted: false,
            last_spawn_error: None,
            pending_outputs: VecDeque::new(),
        })
    }

    #[cfg(test)]
    pub(crate) fn with_factory_and_max_respawns(
        config: GlobalDescriptorConfig,
        factory: DescriptorExtractorFactory,
        max_respawns: u32,
    ) -> Self {
        let (worker, last_spawn_error) = match Self::spawn_worker(config, &factory) {
            Ok(worker) => (Some(worker), None),
            Err(error) => {
                eprintln!("descriptor worker unavailable: {error}");
                (None, Some(Arc::new(error)))
            }
        };
        let spawn_exhausted = worker.is_none() && max_respawns == 0;
        Self {
            worker,
            config,
            factory,
            respawn_count: 0,
            max_respawns,
            spawn_exhausted,
            last_spawn_error,
            pending_outputs: VecDeque::new(),
        }
    }

    fn spawn_worker(
        config: GlobalDescriptorConfig,
        factory: &DescriptorExtractorFactory,
    ) -> Result<DescriptorWorker, DescriptorInitError> {
        let extractor = factory()?;
        DescriptorWorker::spawn(config, extractor)
            .map_err(|source| DescriptorInitError::WorkerThread { source })
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
        let respawn_count = self.respawn_count.saturating_add(1);
        self.respawn_count = respawn_count;
        match Self::spawn_worker(self.config, &self.factory) {
            Ok(worker) => {
                self.worker = Some(worker);
                self.last_spawn_error = None;
            }
            Err(error) => {
                let error = Arc::new(error);
                let exhausted = respawn_count >= self.max_respawns;
                eprintln!("descriptor worker respawn failed: {error}");
                if exhausted {
                    self.spawn_exhausted = true;
                    eprintln!(
                        "descriptor worker respawn exhausted after {} attempts",
                        self.max_respawns
                    );
                } else {
                    eprintln!("descriptor worker remains unavailable; will retry");
                }
                self.last_spawn_error = Some(Arc::clone(&error));
                self.pending_outputs
                    .push_back(DescriptorSupervisorOutput::RestartFailure(
                        DescriptorRestartFailure {
                            respawn_count,
                            max_respawns: self.max_respawns,
                            exhausted,
                            error,
                        },
                    ));
            }
        }
    }

    fn disconnect_worker(&mut self) {
        let Some(worker) = self.worker.take() else {
            return;
        };
        while let Ok(Some(response)) = worker.try_recv() {
            self.pending_outputs
                .push_back(DescriptorSupervisorOutput::Worker(response));
        }
    }

    fn unavailable_error(&self) -> SubmitDescriptorError {
        SubmitDescriptorError::Unavailable {
            exhausted: self.spawn_exhausted,
            source: self.last_spawn_error.as_ref().map(Arc::clone),
        }
    }

    pub(crate) fn submit(
        &mut self,
        request: DescriptorRequest,
    ) -> Result<(), SubmitDescriptorError> {
        if self.worker.is_none() {
            self.check_health();
        }
        let Some(worker) = self.worker.as_ref() else {
            return Err(self.unavailable_error());
        };
        let request = match worker.submit(request) {
            Ok(()) => return Ok(()),
            Err(DescriptorWorkerSubmitError::QueueFull(_request)) => {
                return Err(SubmitDescriptorError::QueueFull);
            }
            Err(DescriptorWorkerSubmitError::Disconnected(request)) => request,
        };

        self.disconnect_worker();
        self.check_health();
        let Some(worker) = self.worker.as_ref() else {
            return Err(self.unavailable_error());
        };
        match worker.submit(request) {
            Ok(()) => Ok(()),
            Err(DescriptorWorkerSubmitError::QueueFull(_request)) => {
                Err(SubmitDescriptorError::QueueFull)
            }
            Err(DescriptorWorkerSubmitError::Disconnected(_request)) => {
                self.disconnect_worker();
                Err(self.unavailable_error())
            }
        }
    }

    pub(crate) fn try_recv(&mut self) -> Option<DescriptorSupervisorOutput> {
        if let Some(output) = self.pending_outputs.pop_front() {
            return Some(output);
        }
        let response = self.worker.as_ref()?.try_recv();
        match response {
            Ok(Some(response)) => {
                if matches!(response, DescriptorWorkerResponse::WorkerPanic { .. }) {
                    self.disconnect_worker();
                    self.check_health();
                }
                Some(DescriptorSupervisorOutput::Worker(response))
            }
            Ok(None) => None,
            Err(()) => {
                self.disconnect_worker();
                self.check_health();
                self.pending_outputs.pop_front()
            }
        }
    }

    pub(crate) fn respawn_count(&self) -> u32 {
        self.respawn_count
    }

    pub(crate) fn respawn_exhausted(&self) -> bool {
        self.spawn_exhausted
    }

    pub(crate) fn has_worker(&self) -> bool {
        self.worker.is_some()
    }

    #[cfg(test)]
    pub(crate) fn worker_thread_is_finished(&self) -> bool {
        self.worker
            .as_ref()
            .is_some_and(DescriptorWorker::is_finished)
    }
}

#[derive(Debug)]
pub(crate) enum PlaceRecognitionEvent {
    InferenceFailed {
        keyframe_id: KeyframeId,
        source_snapshot: MapSnapshot,
        error: InferenceError,
    },
    Panicked {
        keyframe_id: KeyframeId,
        source_snapshot: MapSnapshot,
        message: String,
        respawn_count: u32,
    },
    RestartFailed {
        respawn_count: u32,
        max_respawns: u32,
        exhausted: bool,
        error: Arc<DescriptorInitError>,
    },
    IndexFailed {
        keyframe_id: KeyframeId,
        source_snapshot: MapSnapshot,
        error: KeyframeDatabaseError,
    },
}

pub(crate) struct PlaceRecognition {
    database: KeyframeDatabase,
    descriptor_worker: DescriptorSupervisor,
    descriptor_stats: DescriptorStats,
    loop_config: LoopClosureConfig,
    pending_loop: Option<PendingLoopCandidate>,
    loop_streak: std::collections::HashMap<KeyframeId, usize>,
}

impl PlaceRecognition {
    pub(crate) fn new(
        loop_config: LoopClosureConfig,
        descriptor_config: GlobalDescriptorConfig,
        max_respawns: u32,
    ) -> Result<Self, DescriptorInitError> {
        let descriptor_worker =
            DescriptorSupervisor::spawn_with_max_respawns(descriptor_config, max_respawns)?;
        Ok(Self {
            database: KeyframeDatabase::new(loop_config.temporal_gap()),
            descriptor_worker,
            descriptor_stats: DescriptorStats::default(),
            loop_config,
            pending_loop: None,
            loop_streak: std::collections::HashMap::new(),
        })
    }

    pub(crate) fn descriptor_stats(&self) -> DescriptorStats {
        DescriptorStats {
            respawn_count: self.descriptor_worker.respawn_count(),
            respawn_exhausted: self.descriptor_worker.respawn_exhausted(),
            ..self.descriptor_stats
        }
    }

    pub(crate) fn loop_config(&self) -> LoopClosureConfig {
        self.loop_config
    }

    pub(crate) fn has_worker(&self) -> bool {
        self.descriptor_worker.has_worker()
    }

    pub(crate) fn pending_candidate_count(&self) -> usize {
        self.pending_loop
            .as_ref()
            .map_or(0, |pending| pending.candidates.len())
    }

    pub(crate) fn clear_pending(&mut self) {
        self.pending_loop = None;
        self.loop_streak.clear();
    }

    /// Forget every keyframe belonging to the previous map instance while
    /// retaining the learned-descriptor worker and its model allocation.
    pub(crate) fn reset_mapping_session(&mut self) {
        self.database = KeyframeDatabase::new(self.loop_config.temporal_gap());
        self.clear_pending();
    }

    pub(crate) fn take_pending_loop(&mut self) -> Option<PendingLoopCandidate> {
        self.pending_loop.take()
    }

    pub(crate) fn relocalization_matches(
        &self,
        descriptor: &GlobalDescriptor,
        max_candidates: usize,
    ) -> Vec<RelocalizationMatch> {
        self.database
            .query_for_relocalization(descriptor, max_candidates)
    }

    pub(crate) fn remove_keyframe(
        &mut self,
        keyframe_id: KeyframeId,
    ) -> Result<(), KeyframeDatabaseError> {
        self.database.remove(keyframe_id)?;
        self.loop_streak.remove(&keyframe_id);
        if let Some(pending) = self.pending_loop.as_mut() {
            if pending.query_kf == keyframe_id {
                self.pending_loop = None;
            } else {
                pending
                    .candidates
                    .retain(|candidate| candidate.candidate() != keyframe_id);
                if pending.candidates.is_empty() {
                    self.pending_loop = None;
                }
            }
        }
        Ok(())
    }

    pub(crate) fn prepare_keyframe(
        &self,
        keyframe_id: KeyframeId,
        detections: &Arc<Detections>,
    ) -> Result<PreparedKeyframeRecognition, BootstrapDescriptorError> {
        let global_descriptor = aggregate_global_descriptor(detections.descriptors())
            .map_err(|source| BootstrapDescriptorError::Aggregation { source })?;
        let mut database = self.database.clone();
        database
            .register_keyframe(keyframe_id)
            .map_err(|source| BootstrapDescriptorError::Database { source })?;
        let mut candidates = database
            .query_loop_candidates(
                keyframe_id,
                &global_descriptor,
                self.loop_config.max_candidates(),
            )
            .map_err(|source| BootstrapDescriptorError::Database { source })?;
        database
            .set_descriptor(keyframe_id, global_descriptor, DescriptorSource::Bootstrap)
            .map_err(|source| BootstrapDescriptorError::Database { source })?;
        candidates.retain(|candidate| {
            candidate.cosine_similarity().value() >= self.loop_config.similarity_threshold()
        });
        let mut pending_loop = self.pending_loop.clone();
        let mut loop_streak = self.loop_streak.clone();
        Self::update_loop_candidates(
            keyframe_id,
            detections,
            candidates,
            self.loop_config,
            &mut pending_loop,
            &mut loop_streak,
        );
        Ok(PreparedKeyframeRecognition {
            database,
            pending_loop,
            loop_streak,
        })
    }

    pub(crate) fn commit_keyframe(
        &mut self,
        prepared: PreparedKeyframeRecognition,
        keyframe_id: KeyframeId,
        frame: &Frame,
        source_snapshot: MapSnapshot,
    ) {
        self.database = prepared.database;
        self.pending_loop = prepared.pending_loop;
        self.loop_streak = prepared.loop_streak;
        self.enqueue_descriptor_request(keyframe_id, frame, source_snapshot);
    }

    pub(crate) fn drain_responses(
        &mut self,
        current_source_snapshot: MapSnapshot,
        keyframe_exists: impl Fn(KeyframeId) -> bool,
    ) -> Vec<PlaceRecognitionEvent> {
        let mut events = Vec::new();
        loop {
            let Some(output) = self.descriptor_worker.try_recv() else {
                break;
            };
            let response = match output {
                DescriptorSupervisorOutput::Worker(response) => response,
                DescriptorSupervisorOutput::RestartFailure(failure) => {
                    self.descriptor_stats.restart_failures =
                        self.descriptor_stats.restart_failures.saturating_add(1);
                    events.push(PlaceRecognitionEvent::RestartFailed {
                        respawn_count: failure.respawn_count,
                        max_respawns: failure.max_respawns,
                        exhausted: failure.exhausted,
                        error: failure.error,
                    });
                    continue;
                }
            };
            match response {
                DescriptorWorkerResponse::Descriptor(response) => {
                    if !response
                        .source_snapshot
                        .is_same_or_older_than(current_source_snapshot)
                    {
                        continue;
                    }
                    if !keyframe_exists(response.keyframe_id) {
                        continue;
                    }
                    match self.database.set_descriptor(
                        response.keyframe_id,
                        response.descriptor,
                        DescriptorSource::Learned,
                    ) {
                        Ok(_) => {
                            self.descriptor_stats.applied =
                                self.descriptor_stats.applied.saturating_add(1);
                        }
                        Err(error) => events.push(PlaceRecognitionEvent::IndexFailed {
                            keyframe_id: response.keyframe_id,
                            source_snapshot: response.source_snapshot,
                            error,
                        }),
                    }
                }
                DescriptorWorkerResponse::Failure {
                    keyframe_id,
                    source_snapshot,
                    error,
                } => {
                    self.descriptor_stats.worker_failures =
                        self.descriptor_stats.worker_failures.saturating_add(1);
                    events.push(PlaceRecognitionEvent::InferenceFailed {
                        keyframe_id,
                        source_snapshot,
                        error,
                    });
                }
                DescriptorWorkerResponse::WorkerPanic {
                    keyframe_id,
                    source_snapshot,
                    message,
                } => {
                    self.descriptor_stats.panics = self.descriptor_stats.panics.saturating_add(1);
                    self.descriptor_stats.worker_failures =
                        self.descriptor_stats.worker_failures.saturating_add(1);
                    events.push(PlaceRecognitionEvent::Panicked {
                        keyframe_id,
                        source_snapshot,
                        message,
                        respawn_count: self.descriptor_worker.respawn_count(),
                    });
                }
            }
        }
        events
    }

    fn update_loop_candidates(
        keyframe_id: KeyframeId,
        detections: &Arc<Detections>,
        candidates: Vec<PlaceMatch>,
        loop_config: LoopClosureConfig,
        pending_loop: &mut Option<PendingLoopCandidate>,
        loop_streak: &mut std::collections::HashMap<KeyframeId, usize>,
    ) {
        if candidates.is_empty() {
            loop_streak.clear();
            return;
        }

        let present: HashSet<KeyframeId> = candidates
            .iter()
            .map(|candidate| candidate.candidate())
            .collect();
        loop_streak.retain(|candidate, _| present.contains(candidate));
        for candidate in &candidates {
            let streak = loop_streak.entry(candidate.candidate()).or_insert(0);
            *streak = streak.saturating_add(1);
        }

        if pending_loop.is_some() {
            return;
        }

        let promoted: Vec<PlaceMatch> = candidates
            .into_iter()
            .filter(|candidate| {
                loop_streak
                    .get(&candidate.candidate())
                    .copied()
                    .unwrap_or(0)
                    >= loop_config.min_streak()
            })
            .collect();

        if promoted.is_empty() {
            return;
        }

        *pending_loop = Some(PendingLoopCandidate {
            query_kf: keyframe_id,
            detections: Arc::clone(detections),
            candidates: promoted,
        });
    }

    fn enqueue_descriptor_request(
        &mut self,
        keyframe_id: KeyframeId,
        frame: &Frame,
        source_snapshot: MapSnapshot,
    ) {
        let request = DescriptorRequest {
            keyframe_id,
            source_snapshot,
            frame: frame.clone(),
        };
        match self.descriptor_worker.submit(request) {
            Ok(()) => {
                self.descriptor_stats.submitted = self.descriptor_stats.submitted.saturating_add(1);
            }
            Err(SubmitDescriptorError::QueueFull) => {
                self.descriptor_stats.dropped_full =
                    self.descriptor_stats.dropped_full.saturating_add(1);
                eprintln!("descriptor worker queue full; keeping bootstrap descriptor");
            }
            Err(error @ SubmitDescriptorError::Unavailable { .. }) => {
                self.descriptor_stats.dropped_unavailable =
                    self.descriptor_stats.dropped_unavailable.saturating_add(1);
                eprintln!("descriptor request not submitted: {error}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error as _;

    #[test]
    fn descriptor_model_path_rejects_an_empty_override() {
        let error = DescriptorWorker::model_path_from_override(Some(OsString::new()))
            .expect_err("empty model override must fail during initialization");
        assert!(matches!(
            error,
            DescriptorInitError::EmptyModelPath {
                variable: EIGENPLACES_MODEL_ENV
            }
        ));
    }

    fn descriptor_with_basis(index: usize) -> GlobalDescriptor {
        let mut values = [0.0_f32; crate::loop_closure::GLOBAL_DESCRIPTOR_DIM];
        values[index] = 1.0;
        GlobalDescriptor::try_new(values).expect("valid descriptor")
    }

    fn recognition_without_worker() -> PlaceRecognition {
        let descriptor_config = GlobalDescriptorConfig::new(1).expect("descriptor config");
        let factory: DescriptorExtractorFactory = Arc::new(|| {
            Err(DescriptorInitError::ModelMissing {
                path: "intentionally-absent-test-model.onnx".into(),
            })
        });
        PlaceRecognition {
            database: KeyframeDatabase::new(LoopClosureConfig::default().temporal_gap()),
            descriptor_worker: DescriptorSupervisor::with_factory_and_max_respawns(
                descriptor_config,
                factory,
                0,
            ),
            descriptor_stats: DescriptorStats::default(),
            loop_config: LoopClosureConfig::default(),
            pending_loop: None,
            loop_streak: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn bootstrap_failure_is_atomic_and_late_response_reports_index_error() {
        let mut recognition = recognition_without_worker();
        let keyframe_id = KeyframeId::default();
        let source_snapshot = crate::map::SlamMap::new().snapshot();
        let frame_id = crate::FrameId::new(1);
        let detections = Arc::new(
            Detections::new(
                crate::SensorId::StereoLeft,
                frame_id,
                2,
                2,
                vec![crate::Keypoint { x: 0.0, y: 0.0 }],
                vec![1.0],
                vec![crate::Descriptor::ZERO],
            )
            .expect("finite zero-norm detections"),
        );
        let bootstrap_error = recognition
            .prepare_keyframe(keyframe_id, &detections)
            .expect_err("zero-norm bootstrap descriptor must fail");
        assert!(matches!(
            bootstrap_error,
            BootstrapDescriptorError::Aggregation {
                source: GlobalDescriptorError::ZeroNorm
            }
        ));
        assert_eq!(recognition.database.registered_len(), 0);
        assert_eq!(recognition.database.descriptor_len(), 0);
        assert_eq!(recognition.descriptor_stats().dropped_unavailable, 0);

        recognition.descriptor_worker.pending_outputs.push_back(
            DescriptorSupervisorOutput::Worker(DescriptorWorkerResponse::Descriptor(Box::new(
                DescriptorResponse {
                    keyframe_id,
                    source_snapshot,
                    descriptor: descriptor_with_basis(7),
                },
            ))),
        );

        let events = recognition.drain_responses(source_snapshot, |id| id == keyframe_id);

        assert!(matches!(
            events.as_slice(),
            [PlaceRecognitionEvent::IndexFailed {
                keyframe_id: failed_id,
                error: KeyframeDatabaseError::KeyframeNotRegistered { keyframe_id: missing },
                ..
            }] if *failed_id == keyframe_id && *missing == keyframe_id
        ));
        assert_eq!(recognition.database.registered_len(), 0);
        assert_eq!(recognition.database.descriptor_len(), 0);
        assert_eq!(recognition.descriptor_stats().applied, 0);
    }

    #[test]
    fn prepared_keyframe_registration_commits_as_one_database_transition() {
        let mut recognition = recognition_without_worker();
        let keyframe_id = KeyframeId::default();
        let frame_id = crate::FrameId::new(2);
        let mut descriptor = [0.0_f32; crate::DESCRIPTOR_DIM];
        descriptor[0] = 1.0;
        let detections = Arc::new(
            Detections::new(
                crate::SensorId::StereoLeft,
                frame_id,
                2,
                2,
                vec![crate::Keypoint { x: 0.0, y: 0.0 }],
                vec![1.0],
                vec![crate::Descriptor::try_new(descriptor).expect("finite descriptor")],
            )
            .expect("detections"),
        );
        let frame = Frame::new(
            crate::SensorId::StereoLeft,
            frame_id,
            crate::Timestamp::from_nanos(2),
            2,
            2,
            vec![0; 4],
        )
        .expect("frame");
        let source_snapshot = crate::map::SlamMap::new().snapshot();

        let prepared = recognition
            .prepare_keyframe(keyframe_id, &detections)
            .expect("valid descriptor registration");
        assert_eq!(recognition.database.registered_len(), 0);

        recognition.commit_keyframe(prepared, keyframe_id, &frame, source_snapshot);
        assert_eq!(recognition.database.registered_len(), 1);
        assert_eq!(recognition.database.descriptor_len(), 1);
        assert_eq!(
            recognition.database.descriptor_source(keyframe_id),
            Some(DescriptorSource::Bootstrap)
        );
    }

    #[test]
    fn bootstrap_database_error_preserves_source() {
        let error = BootstrapDescriptorError::Database {
            source: KeyframeDatabaseError::SequenceExhausted {
                next_sequence: u64::MAX,
            },
        };

        assert!(matches!(
            error.source(),
            Some(source) if source.to_string().contains("sequence exhausted")
        ));
    }
}
