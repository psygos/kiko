use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;

use crate::loop_closure::{
    DescriptorSource, GlobalDescriptor, KeyframeDatabase, LoopClosureConfig, PlaceMatch,
    RelocalizationMatch, aggregate_global_descriptor,
};
use crate::map::KeyframeId;
use crate::{
    Detections, EigenPlaces, Frame, GlobalDescriptorConfig, InferenceError,
    PlaceDescriptorExtractor,
};
use crossbeam_channel::{Receiver, Sender, TryRecvError, TrySendError};

use crate::MapSnapshot;

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

#[derive(Debug, Clone)]
pub(crate) struct PendingLoopCandidate {
    pub(crate) query_kf: KeyframeId,
    pub(crate) detections: Arc<Detections>,
    pub(crate) candidates: Vec<PlaceMatch>,
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
    Disconnected,
}

pub(crate) type DescriptorExtractorFactory =
    Arc<dyn Fn() -> Result<Box<dyn PlaceDescriptorExtractor>, DescriptorInitError> + Send + Sync>;

#[derive(Debug)]
pub enum DescriptorInitError {
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
            Self::ModelMissing { .. } => None,
        }
    }
}

pub(crate) struct DescriptorWorker {
    tx: Sender<DescriptorRequest>,
    rx: Receiver<DescriptorWorkerResponse>,
    _thread: thread::JoinHandle<()>,
}

impl DescriptorWorker {
    fn model_path() -> PathBuf {
        if let Some(path) = std::env::var_os("KIKO_EIGENPLACES_MODEL") {
            return path.into();
        }
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("models")
            .join("eigenplaces.onnx")
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

    pub(crate) fn submit(&self, request: DescriptorRequest) -> Result<(), SubmitDescriptorError> {
        match self.tx.try_send(request) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(_)) => Err(SubmitDescriptorError::QueueFull),
            Err(TrySendError::Disconnected(_)) => Err(SubmitDescriptorError::Disconnected),
        }
    }

    pub(crate) fn try_recv(&self) -> Result<Option<DescriptorWorkerResponse>, ()> {
        match self.rx.try_recv() {
            Ok(value) => Ok(Some(value)),
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => Err(()),
        }
    }
}

pub(crate) struct DescriptorSupervisor {
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
            let extractor = EigenPlaces::try_load(&path, crate::InferenceBackend::auto())
                .map_err(|source| DescriptorInitError::ModelLoad {
                    path: path.clone(),
                    source,
                })?
                .ok_or_else(|| DescriptorInitError::ModelMissing { path })?;
            Ok(Box::new(extractor) as Box<dyn PlaceDescriptorExtractor>)
        })
    }

    fn spawn_with_max_respawns(
        config: GlobalDescriptorConfig,
        max_respawns: u32,
    ) -> Result<Self, DescriptorInitError> {
        let factory = Self::default_factory();
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

    #[cfg(test)]
    pub(crate) fn with_factory_and_max_respawns(
        config: GlobalDescriptorConfig,
        factory: DescriptorExtractorFactory,
        max_respawns: u32,
    ) -> Self {
        let worker = Self::spawn_worker(config, &factory)
            .inspect_err(|error| eprintln!("descriptor worker unavailable: {error}"))
            .ok();
        let spawn_exhausted = worker.is_none() && max_respawns == 0;
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
        self.worker = Self::spawn_worker(self.config, &self.factory)
            .inspect_err(|error| eprintln!("descriptor worker respawn failed: {error}"))
            .ok();
        self.respawn_count = self.respawn_count.saturating_add(1);
        if self.worker.is_none() {
            if self.respawn_count >= self.max_respawns {
                self.spawn_exhausted = true;
                eprintln!(
                    "descriptor worker respawn exhausted after {} attempts",
                    self.max_respawns
                );
            } else {
                eprintln!("descriptor worker remains unavailable; will retry");
            }
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
            return Err(SubmitDescriptorError::Disconnected);
        };
        let result = worker.submit(request);
        if matches!(result, Err(SubmitDescriptorError::Disconnected)) {
            self.worker = None;
            self.check_health();
        }
        result
    }

    pub(crate) fn try_recv(&mut self) -> Option<DescriptorWorkerResponse> {
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

    pub(crate) fn respawn_count(&self) -> u32 {
        self.respawn_count
    }

    pub(crate) fn has_worker(&self) -> bool {
        self.worker.is_some()
    }
}

#[derive(Debug)]
pub(crate) enum PlaceRecognitionEvent {
    WorkerFailure {
        keyframe_id: KeyframeId,
        source_snapshot: MapSnapshot,
        error: InferenceError,
    },
    WorkerPanic {
        keyframe_id: KeyframeId,
        source_snapshot: MapSnapshot,
        message: String,
        respawn_count: u32,
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
        self.descriptor_stats
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

    pub(crate) fn remove_keyframe(&mut self, keyframe_id: KeyframeId) {
        self.database.remove(keyframe_id);
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
    }

    pub(crate) fn on_keyframe(
        &mut self,
        keyframe_id: KeyframeId,
        detections: &Arc<Detections>,
        frame: &Frame,
        source_snapshot: MapSnapshot,
    ) {
        self.enqueue_loop_candidates(keyframe_id, detections);
        self.enqueue_descriptor_request(keyframe_id, frame, source_snapshot);
    }

    pub(crate) fn drain_responses(
        &mut self,
        current_source_snapshot: MapSnapshot,
        keyframe_exists: impl Fn(KeyframeId) -> bool,
    ) -> Vec<PlaceRecognitionEvent> {
        let mut events = Vec::new();
        loop {
            let Some(response) = self.descriptor_worker.try_recv() else {
                break;
            };
            self.descriptor_stats.respawn_count = self.descriptor_worker.respawn_count();
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
                    if self.database.replace_descriptor(
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
                    source_snapshot,
                    error,
                } => {
                    self.descriptor_stats.worker_failures =
                        self.descriptor_stats.worker_failures.saturating_add(1);
                    events.push(PlaceRecognitionEvent::WorkerFailure {
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
                    events.push(PlaceRecognitionEvent::WorkerPanic {
                        keyframe_id,
                        source_snapshot,
                        message,
                        respawn_count: self.descriptor_stats.respawn_count,
                    });
                }
            }
        }
        events
    }

    fn enqueue_loop_candidates(&mut self, keyframe_id: KeyframeId, detections: &Arc<Detections>) {
        let Ok(global_descriptor) = aggregate_global_descriptor(detections.descriptors()) else {
            return;
        };
        self.database.insert_with_source(
            keyframe_id,
            global_descriptor.clone(),
            DescriptorSource::Bootstrap,
        );

        let mut candidates = self
            .database
            .query(&global_descriptor, self.loop_config.max_candidates());
        candidates
            .retain(|candidate| candidate.similarity >= self.loop_config.similarity_threshold());

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
                    >= self.loop_config.min_streak()
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
            Err(SubmitDescriptorError::Disconnected) => {
                self.descriptor_stats.dropped_disconnected =
                    self.descriptor_stats.dropped_disconnected.saturating_add(1);
                self.descriptor_stats.respawn_count = self.descriptor_worker.respawn_count();
                eprintln!("descriptor worker disconnected; retrying with supervisor");
            }
        }
    }
}
