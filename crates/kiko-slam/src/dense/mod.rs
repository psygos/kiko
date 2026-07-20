pub mod backend;
pub mod command_mapper;
pub mod occupancy;
pub mod occupancy_persistence;
pub mod occupancy_runtime;
pub mod ring_buffer;

use std::collections::{HashMap, VecDeque};
use std::marker::PhantomData;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

#[cfg(test)]
use crate::Pose;
use crate::dense::backend::{TsdfBackend, TsdfBackendFactory, TsdfConfig, TsdfError};
use crate::map::{KeyframeId, MapInstanceId};
use crate::{
    ChannelCapacity, DepthImage, KeyframePoseUpdate, MappingSessionTransition, PinholeIntrinsics,
    Timestamp, WorldToCamera,
};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Commands sent from the pipeline to the dense reconstruction thread.
///
/// Live commands must be routed through [`DenseCommandSender::route`] so that
/// every accepted command retains producer order. Integrations are bounded by
/// a separate data quota and drop newest at that boundary. Control commands
/// wait for the configured timeout and are never silently dropped.
#[derive(Debug)]
pub enum DenseCommand {
    IntegrateKeyframe {
        keyframe_id: KeyframeId,
        pose: WorldToCamera,
        depth: DepthImage,
        timestamp: Timestamp,
    },
    RemoveKeyframe {
        keyframe_id: KeyframeId,
        timestamp: Timestamp,
    },
    /// Establish a hard boundary between independent maps.
    ///
    /// An accepted command clears the host depth store and advances
    /// `generation` even when TSDF clearing fails. On that failure the
    /// uncertain backend is dropped and reconstruction reports [`ReconState::Down`].
    ResetMappingSession {
        transition: MappingSessionTransition,
        generation: u64,
        timestamp: Timestamp,
    },
    /// Apply committed authoritative tracker poses to retained reconstruction
    /// sources. Updates may be a subset because integrations can be dropped or
    /// keyframes can already have been evicted.
    ApplyPoseUpdates {
        updates: Vec<KeyframePoseUpdate>,
        generation: u64,
        timestamp: Timestamp,
    },
}

/// Result of routing one live dense command.
#[must_use = "dense command routing failures and drops must be handled"]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DenseCommandSendOutcome {
    /// The command was accepted into the ordered queue.
    Enqueued,
    /// An integration was not accepted because the configured data quota or
    /// the ordered queue was full.
    IntegrationDroppedNewest,
    /// A control command could not be accepted before its configured timeout.
    ControlTimedOut,
    /// The dense worker receiver has disconnected.
    Disconnected,
}

/// Monotonic ordered-command queue event counters, saturating at `u64::MAX`.
///
/// `commands_enqueued` includes both integration and control commands. The
/// remaining fields count rejected send attempts by their exact cause. A
/// snapshot reads each counter independently; compare cross-field totals only
/// after the producer and consumer have quiesced.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DenseCommandQueueStats {
    pub commands_enqueued: u64,
    pub integrations_dropped_newest: u64,
    pub controls_timed_out: u64,
    pub disconnected: u64,
}

#[derive(Debug)]
struct DenseCommandQueueState {
    receiver_alive: AtomicBool,
    integrations_queued: AtomicUsize,
    commands_enqueued: AtomicU64,
    integrations_dropped_newest: AtomicU64,
    controls_timed_out: AtomicU64,
    disconnected: AtomicU64,
}

impl DenseCommandQueueState {
    fn new() -> Self {
        Self {
            receiver_alive: AtomicBool::new(true),
            integrations_queued: AtomicUsize::new(0),
            commands_enqueued: AtomicU64::new(0),
            integrations_dropped_newest: AtomicU64::new(0),
            controls_timed_out: AtomicU64::new(0),
            disconnected: AtomicU64::new(0),
        }
    }

    fn snapshot(&self) -> DenseCommandQueueStats {
        DenseCommandQueueStats {
            commands_enqueued: self.commands_enqueued.load(Ordering::Relaxed),
            integrations_dropped_newest: self.integrations_dropped_newest.load(Ordering::Relaxed),
            controls_timed_out: self.controls_timed_out.load(Ordering::Relaxed),
            disconnected: self.disconnected.load(Ordering::Relaxed),
        }
    }

    fn record(counter: &AtomicU64) {
        let _ = counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
            value.checked_add(1)
        });
    }

    fn release_integration(&self) {
        let _ = self
            .integrations_queued
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                current.checked_sub(1)
            })
            .expect("dequeued integration must own a data slot");
    }
}

/// Read-only handle for ordered dense-command queue statistics.
#[derive(Clone, Debug)]
pub struct DenseCommandQueueStatsHandle {
    state: Arc<DenseCommandQueueState>,
}

impl DenseCommandQueueStatsHandle {
    pub fn snapshot(&self) -> DenseCommandQueueStats {
        self.state.snapshot()
    }
}

/// Error constructing an ordered dense-command queue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DenseCommandChannelError {
    CapacityOverflow {
        data_capacity: usize,
        reserved_control_capacity: usize,
    },
}

impl std::fmt::Display for DenseCommandChannelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CapacityOverflow {
                data_capacity,
                reserved_control_capacity,
            } => write!(
                f,
                "dense command queue capacities overflow usize: data={data_capacity}, reserved_control={reserved_control_capacity}"
            ),
        }
    }
}

impl std::error::Error for DenseCommandChannelError {}

/// Single-producer endpoint for the causally ordered dense-command queue.
///
/// This endpoint is intentionally non-cloneable and non-`Sync`: all commands
/// in one mapping session must pass through one producer in program order.
#[derive(Debug)]
pub struct DenseCommandSender {
    tx: crossbeam_channel::Sender<DenseCommand>,
    state: Arc<DenseCommandQueueState>,
    data_capacity: usize,
    control_timeout: Duration,
    _single_producer: PhantomData<std::cell::Cell<()>>,
}

impl DenseCommandSender {
    /// Route one command according to its domain variant.
    ///
    /// Integrations never block and drop newest once the data quota is full.
    /// Controls wait up to the configured timeout. Both use the same FIFO, so
    /// every accepted command retains this producer's causal order.
    pub fn route(&self, command: DenseCommand) -> DenseCommandSendOutcome {
        if !self.state.receiver_alive.load(Ordering::Acquire) {
            DenseCommandQueueState::record(&self.state.disconnected);
            return DenseCommandSendOutcome::Disconnected;
        }

        match command {
            command @ DenseCommand::IntegrateKeyframe { .. } => self.route_integration(command),
            command => self.route_control(command),
        }
    }

    fn route_integration(&self, command: DenseCommand) -> DenseCommandSendOutcome {
        let reserved = self
            .state
            .integrations_queued
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                (current < self.data_capacity).then_some(current + 1)
            })
            .is_ok();
        if !reserved {
            DenseCommandQueueState::record(&self.state.integrations_dropped_newest);
            return DenseCommandSendOutcome::IntegrationDroppedNewest;
        }

        match self.tx.try_send(command) {
            Ok(()) => {
                DenseCommandQueueState::record(&self.state.commands_enqueued);
                DenseCommandSendOutcome::Enqueued
            }
            Err(crossbeam_channel::TrySendError::Full(_)) => {
                self.state.release_integration();
                DenseCommandQueueState::record(&self.state.integrations_dropped_newest);
                DenseCommandSendOutcome::IntegrationDroppedNewest
            }
            Err(crossbeam_channel::TrySendError::Disconnected(_)) => {
                self.state.release_integration();
                DenseCommandQueueState::record(&self.state.disconnected);
                DenseCommandSendOutcome::Disconnected
            }
        }
    }

    fn route_control(&self, command: DenseCommand) -> DenseCommandSendOutcome {
        match self.tx.send_timeout(command, self.control_timeout) {
            Ok(()) => {
                DenseCommandQueueState::record(&self.state.commands_enqueued);
                DenseCommandSendOutcome::Enqueued
            }
            Err(crossbeam_channel::SendTimeoutError::Timeout(_)) => {
                DenseCommandQueueState::record(&self.state.controls_timed_out);
                DenseCommandSendOutcome::ControlTimedOut
            }
            Err(crossbeam_channel::SendTimeoutError::Disconnected(_)) => {
                DenseCommandQueueState::record(&self.state.disconnected);
                DenseCommandSendOutcome::Disconnected
            }
        }
    }
}

/// Sole consumer endpoint for the causally ordered dense-command queue.
///
/// Like the sender, this endpoint is non-cloneable and non-`Sync` so an
/// integration quota slot is released exactly once by one consumer.
#[derive(Debug)]
pub struct DenseCommandReceiver {
    rx: crossbeam_channel::Receiver<DenseCommand>,
    state: Arc<DenseCommandQueueState>,
    _sole_consumer: PhantomData<std::cell::Cell<()>>,
}

impl DenseCommandReceiver {
    pub fn recv(&self) -> Result<DenseCommand, crossbeam_channel::RecvError> {
        self.rx.recv().inspect(|command| {
            self.release_data_quota(command);
        })
    }

    pub fn try_recv(&self) -> Result<DenseCommand, crossbeam_channel::TryRecvError> {
        self.rx.try_recv().inspect(|command| {
            self.release_data_quota(command);
        })
    }

    fn release_data_quota(&self, command: &DenseCommand) {
        if matches!(command, DenseCommand::IntegrateKeyframe { .. }) {
            self.state.release_integration();
        }
    }
}

impl Drop for DenseCommandReceiver {
    fn drop(&mut self) {
        self.state.receiver_alive.store(false, Ordering::Release);
    }
}

/// Build one FIFO with `data_capacity + reserved_control_capacity` slots.
///
/// At most `data_capacity` queued integrations may occupy it, reserving the
/// remaining capacity for controls. Controls may use otherwise idle data
/// slots, but the total backlog always remains bounded by the summed capacity.
pub fn dense_command_channel(
    data_capacity: ChannelCapacity,
    reserved_control_capacity: ChannelCapacity,
    control_timeout: Duration,
) -> Result<
    (
        DenseCommandSender,
        DenseCommandReceiver,
        DenseCommandQueueStatsHandle,
    ),
    DenseCommandChannelError,
> {
    let total_capacity = data_capacity
        .get()
        .checked_add(reserved_control_capacity.get())
        .ok_or(DenseCommandChannelError::CapacityOverflow {
            data_capacity: data_capacity.get(),
            reserved_control_capacity: reserved_control_capacity.get(),
        })?;
    let (tx, rx) = crossbeam_channel::bounded(total_capacity);
    let state = Arc::new(DenseCommandQueueState::new());
    Ok((
        DenseCommandSender {
            tx,
            state: Arc::clone(&state),
            data_capacity: data_capacity.get(),
            control_timeout,
            _single_producer: PhantomData,
        },
        DenseCommandReceiver {
            rx,
            state: Arc::clone(&state),
            _sole_consumer: PhantomData,
        },
        DenseCommandQueueStatsHandle { state },
    ))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReconState {
    AwaitingBackend,
    Nominal,
    Rebuilding { generation: u64 },
    Down,
}

#[derive(Debug)]
pub struct TsdfModeConfigError(crate::dense::backend::TsdfConfigError);

impl std::fmt::Display for TsdfModeConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid TSDF configuration: {}", self.0)
    }
}

impl std::error::Error for TsdfModeConfigError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.0)
    }
}

#[derive(Clone, Debug)]
pub struct TsdfModeConfig {
    config: TsdfConfig,
    intrinsics: PinholeIntrinsics,
}

impl TsdfModeConfig {
    pub fn try_new(
        config: TsdfConfig,
        intrinsics: PinholeIntrinsics,
    ) -> Result<Self, TsdfModeConfigError> {
        config.validate().map_err(TsdfModeConfigError)?;
        Ok(Self { config, intrinsics })
    }

    pub fn config(&self) -> &TsdfConfig {
        &self.config
    }

    pub fn intrinsics(&self) -> PinholeIntrinsics {
        self.intrinsics
    }
}

#[derive(Clone, Debug, Default)]
pub enum DenseMode {
    #[default]
    DepthStoreOnly,
    Tsdf(TsdfModeConfig),
}

#[derive(Clone, Debug)]
pub struct DenseStats {
    pub integrated_count: u64,
    pub removed_count: u64,
    pub rebuild_count: u64,
    pub stored_keyframes: usize,
    pub state: ReconState,
}

impl Default for DenseStats {
    fn default() -> Self {
        Self {
            integrated_count: 0,
            removed_count: 0,
            rebuild_count: 0,
            stored_keyframes: 0,
            state: ReconState::Nominal,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DenseConfigError {
    ZeroMaxStoredKeyframes,
}

impl std::fmt::Display for DenseConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroMaxStoredKeyframes => {
                write!(f, "maximum stored keyframes must be nonzero")
            }
        }
    }
}

impl std::error::Error for DenseConfigError {}

#[derive(Debug)]
pub struct DenseConfig {
    /// Maximum number of depth keyframes to store. Insertion-order eviction
    /// kicks in when this limit is reached (a safety net for exploration
    /// trajectories where the tracker never culls keyframes).
    pub max_stored_keyframes: NonZeroUsize,
    /// Dense reconstruction mode.
    pub mode: DenseMode,
}

impl DenseConfig {
    pub fn try_new(max_stored_keyframes: usize, mode: DenseMode) -> Result<Self, DenseConfigError> {
        let max_stored_keyframes = NonZeroUsize::new(max_stored_keyframes)
            .ok_or(DenseConfigError::ZeroMaxStoredKeyframes)?;
        Ok(Self {
            max_stored_keyframes,
            mode,
        })
    }
}

impl Default for DenseConfig {
    fn default() -> Self {
        Self {
            max_stored_keyframes: NonZeroUsize::new(300)
                .expect("default dense store capacity is nonzero"),
            mode: DenseMode::DepthStoreOnly,
        }
    }
}

// ---------------------------------------------------------------------------
// Depth store
// ---------------------------------------------------------------------------

/// Bounded store of depth images keyed by keyframe ID.
///
/// Primary eviction is via `RemoveKeyframe` commands (driven by
/// `DiagnosticEvent::KeyframeRemoved`). The insertion-order cap is a safety
/// net for unbounded growth in exploration scenarios. Updating an existing
/// keyframe does not change its eviction position.
#[derive(Debug)]
pub(crate) struct DepthStore {
    map: HashMap<KeyframeId, StoredDepth>,
    order: VecDeque<KeyframeId>,
    cap: NonZeroUsize,
}

#[derive(Debug)]
struct StoredDepth {
    pose: WorldToCamera,
    depth: DepthImage,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct DepthStoreInsertOutcome {
    evicted: Option<KeyframeId>,
    replaced_existing: bool,
}

impl DepthStore {
    pub fn new(cap: NonZeroUsize) -> Self {
        Self {
            map: HashMap::with_capacity(cap.get().min(64)),
            order: VecDeque::with_capacity(cap.get().min(64)),
            cap,
        }
    }

    #[cfg(test)]
    pub fn insert(&mut self, keyframe_id: KeyframeId, depth: DepthImage) {
        let _ = self.insert_with_pose(keyframe_id, WorldToCamera::identity(), depth);
    }

    fn insert_with_pose(
        &mut self,
        keyframe_id: KeyframeId,
        pose: WorldToCamera,
        depth: DepthImage,
    ) -> DepthStoreInsertOutcome {
        if let std::collections::hash_map::Entry::Occupied(mut entry) = self.map.entry(keyframe_id)
        {
            // Update in place, don't change order.
            entry.insert(StoredDepth { pose, depth });
            return DepthStoreInsertOutcome {
                evicted: None,
                replaced_existing: true,
            };
        }

        // Insertion-order eviction if at cap.
        let mut evicted = None;
        while self.map.len() >= self.cap.get() {
            if let Some(oldest) = self.order.pop_front() {
                self.map.remove(&oldest);
                evicted = Some(oldest);
            } else {
                break;
            }
        }

        self.map.insert(keyframe_id, StoredDepth { pose, depth });
        self.order.push_back(keyframe_id);
        DepthStoreInsertOutcome {
            evicted,
            replaced_existing: false,
        }
    }

    /// Remove a keyframe. No-op if the ID is unknown.
    pub fn remove(&mut self, keyframe_id: KeyframeId) -> bool {
        if self.map.remove(&keyframe_id).is_some() {
            self.order.retain(|id| *id != keyframe_id);
            true
        } else {
            false
        }
    }

    pub fn clear(&mut self) {
        self.map.clear();
        self.order.clear();
    }

    #[cfg(test)]
    pub fn get(&self, keyframe_id: KeyframeId) -> Option<&DepthImage> {
        self.map.get(&keyframe_id).map(|stored| &stored.depth)
    }

    fn update_pose(&mut self, update: KeyframePoseUpdate) -> bool {
        let Some(stored) = self.map.get_mut(&update.keyframe_id()) else {
            return false;
        };
        stored.pose = update.pose();
        true
    }

    fn ordered_entries(&self) -> impl Iterator<Item = (WorldToCamera, &DepthImage)> {
        self.order.iter().map(|keyframe_id| {
            let stored = self
                .map
                .get(keyframe_id)
                .expect("depth-store order must reference a stored keyframe");
            (stored.pose, &stored.depth)
        })
    }

    #[cfg(test)]
    pub fn contains(&self, keyframe_id: KeyframeId) -> bool {
        self.map.contains_key(&keyframe_id)
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }
}

// ---------------------------------------------------------------------------
// State machine
// ---------------------------------------------------------------------------

pub struct DenseState {
    store: DepthStore,
    state: ReconState,
    map_instance_id: Option<MapInstanceId>,
    generation: u64,
    stats: DenseStats,
    mode: DenseMode,
    backend: Option<Box<dyn TsdfBackend>>,
}

impl DenseState {
    pub fn new(config: &DenseConfig) -> Self {
        let initial_state = match &config.mode {
            DenseMode::DepthStoreOnly => ReconState::Nominal,
            DenseMode::Tsdf(_) => ReconState::AwaitingBackend,
        };
        Self {
            store: DepthStore::new(config.max_stored_keyframes),
            state: initial_state,
            map_instance_id: None,
            generation: 0,
            stats: DenseStats::default(),
            mode: config.mode.clone(),
            backend: None,
        }
    }

    /// Attach a TSDF backend. Called from the worker thread after factory
    /// construction succeeds.
    pub fn set_backend(&mut self, backend: Box<dyn TsdfBackend>) {
        match self.mode {
            DenseMode::DepthStoreOnly => {
                eprintln!("dense: ignoring backend attachment in DepthStoreOnly mode");
            }
            DenseMode::Tsdf(_) => {
                self.backend = Some(backend);
                self.state = ReconState::Nominal;
            }
        }
    }

    pub fn has_backend(&self) -> bool {
        self.backend.is_some()
    }

    pub fn stats(&self) -> DenseStats {
        DenseStats {
            integrated_count: self.stats.integrated_count,
            removed_count: self.stats.removed_count,
            rebuild_count: self.stats.rebuild_count,
            stored_keyframes: self.store.len(),
            state: self.state,
        }
    }

    pub fn state(&self) -> ReconState {
        self.state
    }

    fn accepts_keyframe(&mut self, keyframe_id: KeyframeId) -> bool {
        let command_map = keyframe_id.map_instance_id();
        match self.map_instance_id {
            Some(active_map) => active_map == command_map,
            None => {
                self.map_instance_id = Some(command_map);
                true
            }
        }
    }

    fn accepts_pose_updates(&mut self, updates: &[KeyframePoseUpdate]) -> bool {
        let Some((first, rest)) = updates.split_first() else {
            return false;
        };
        let command_map = first.keyframe_id().map_instance_id();
        if rest
            .iter()
            .any(|update| update.keyframe_id().map_instance_id() != command_map)
        {
            return false;
        }
        match self.map_instance_id {
            Some(active_map) => active_map == command_map,
            None => {
                self.map_instance_id = Some(command_map);
                true
            }
        }
    }
}

fn rebuild_tsdf(
    tsdf: &TsdfModeConfig,
    backend: &mut dyn TsdfBackend,
    store: &DepthStore,
) -> Result<(), crate::dense::backend::TsdfError> {
    backend.clear()?;
    for (pose, depth) in store.ordered_entries() {
        backend.integrate(pose.into_legacy_pose(), depth, tsdf.intrinsics)?;
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DenseProcessingOperation {
    IntegrateKeyframe,
    RebuildAfterIntegration,
    RebuildAfterRemoval,
    ResetMappingSession,
    RebuildAfterPoseUpdate,
}

impl std::fmt::Display for DenseProcessingOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let operation = match self {
            Self::IntegrateKeyframe => "integrating a keyframe",
            Self::RebuildAfterIntegration => "rebuilding after integration replacement or eviction",
            Self::RebuildAfterRemoval => "rebuilding after keyframe removal",
            Self::ResetMappingSession => "clearing for a mapping-session reset",
            Self::RebuildAfterPoseUpdate => "rebuilding after authoritative pose updates",
        };
        f.write_str(operation)
    }
}

#[derive(Debug)]
pub enum DenseProcessingError {
    BackendUnavailable {
        operation: DenseProcessingOperation,
    },
    Tsdf {
        operation: DenseProcessingOperation,
        source: TsdfError,
    },
    Panicked {
        detail: String,
    },
}

impl std::fmt::Display for DenseProcessingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BackendUnavailable { operation } => {
                write!(f, "TSDF backend is unavailable while {operation}")
            }
            Self::Tsdf { operation, source } => {
                write!(f, "TSDF failed while {operation}: {source}")
            }
            Self::Panicked { detail } => write!(f, "dense command processing panicked: {detail}"),
        }
    }
}

impl std::error::Error for DenseProcessingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Tsdf { source, .. } => Some(source),
            Self::BackendUnavailable { .. } | Self::Panicked { .. } => None,
        }
    }
}

#[derive(Debug)]
pub enum DenseWorkerError {
    BackendFactory { source: TsdfError },
    BackendFactoryPanicked { detail: String },
    BackendFactoryRequired,
    Processing { source: DenseProcessingError },
}

impl std::fmt::Display for DenseWorkerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BackendFactory { source } => {
                write!(f, "TSDF backend construction failed: {source}")
            }
            Self::BackendFactoryPanicked { detail } => {
                write!(f, "TSDF backend factory panicked: {detail}")
            }
            Self::BackendFactoryRequired => {
                write!(f, "TSDF mode requires a backend factory")
            }
            Self::Processing { source } => write!(f, "dense worker failed: {source}"),
        }
    }
}

impl std::error::Error for DenseWorkerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::BackendFactory { source } => Some(source),
            Self::Processing { source } => Some(source),
            Self::BackendFactoryPanicked { .. } | Self::BackendFactoryRequired => None,
        }
    }
}

/// Process one dense command and return updated stats.
///
/// TSDF failures leave the state fail-closed in [`ReconState::Down`] and
/// preserve their typed source. The caller may inspect [`DenseState::stats`]
/// after an error to publish the terminal state.
pub fn process_dense_command(
    state: &mut DenseState,
    cmd: DenseCommand,
) -> Result<DenseStats, DenseProcessingError> {
    if state.state == ReconState::Down && !matches!(&cmd, DenseCommand::ResetMappingSession { .. })
    {
        // Drain without processing.
        return Ok(state.stats());
    }

    match cmd {
        DenseCommand::IntegrateKeyframe {
            keyframe_id,
            pose,
            depth,
            timestamp: _,
        } => {
            if !state.accepts_keyframe(keyframe_id) {
                eprintln!(
                    "dense: ignoring integration from inactive map {:?}",
                    keyframe_id.map_instance_id()
                );
                return Ok(state.stats());
            }
            let insertion = state
                .store
                .insert_with_pose(keyframe_id, pose, depth.clone());
            let operation = if insertion.evicted.is_some() || insertion.replaced_existing {
                DenseProcessingOperation::RebuildAfterIntegration
            } else {
                DenseProcessingOperation::IntegrateKeyframe
            };
            let failure = match (&state.mode, state.backend.as_mut()) {
                (DenseMode::DepthStoreOnly, _) => {
                    state.stats.integrated_count = state.stats.integrated_count.saturating_add(1);
                    None
                }
                (DenseMode::Tsdf(tsdf), Some(backend)) => {
                    let result = if insertion.evicted.is_some() || insertion.replaced_existing {
                        rebuild_tsdf(tsdf, backend.as_mut(), &state.store)
                    } else {
                        backend.integrate(pose.into_legacy_pose(), &depth, tsdf.intrinsics)
                    };
                    match result {
                        Ok(()) => {
                            state.stats.integrated_count =
                                state.stats.integrated_count.saturating_add(1);
                            None
                        }
                        Err(source) => Some(DenseProcessingError::Tsdf { operation, source }),
                    }
                }
                (DenseMode::Tsdf(_), None) => {
                    Some(DenseProcessingError::BackendUnavailable { operation })
                }
            };
            if let Some(error) = failure {
                state.backend = None;
                state.state = ReconState::Down;
                return Err(error);
            }
        }
        DenseCommand::RemoveKeyframe {
            keyframe_id,
            timestamp: _,
        } => {
            if state.map_instance_id != Some(keyframe_id.map_instance_id()) {
                eprintln!(
                    "dense: ignoring removal from inactive map {:?}",
                    keyframe_id.map_instance_id()
                );
                return Ok(state.stats());
            }
            if state.store.remove(keyframe_id) {
                state.stats.removed_count = state.stats.removed_count.saturating_add(1);
                let operation = DenseProcessingOperation::RebuildAfterRemoval;
                let failure = match (&state.mode, state.backend.as_mut()) {
                    (DenseMode::DepthStoreOnly, _) => None,
                    (DenseMode::Tsdf(tsdf), Some(backend)) => {
                        rebuild_tsdf(tsdf, backend.as_mut(), &state.store)
                            .err()
                            .map(|source| DenseProcessingError::Tsdf { operation, source })
                    }
                    (DenseMode::Tsdf(_), None) => {
                        Some(DenseProcessingError::BackendUnavailable { operation })
                    }
                };
                if let Some(error) = failure {
                    state.backend = None;
                    state.state = ReconState::Down;
                    return Err(error);
                }
            }
        }
        DenseCommand::ResetMappingSession {
            transition,
            generation,
            timestamp: _,
        } => {
            if generation <= state.generation {
                return Ok(state.stats());
            }
            let old_map = transition.old_map();
            let new_map = transition.new_map();
            // Accepting `active_map == new_map` is intentional for offline
            // replay and direct processing: the reset remains an isolation
            // barrier even if new-session state was preloaded out of order.
            // The live ordered queue prevents this ordering. An unrelated
            // third map means the transition is stale or belongs elsewhere,
            // so it must not erase current data.
            if let Some(active_map) = state.map_instance_id
                && active_map != old_map
                && active_map != new_map
            {
                eprintln!(
                    "dense: ignoring mapping-session reset for inactive map {old_map:?}; active map is {active_map:?}"
                );
                return Ok(state.stats());
            }
            if state.map_instance_id == Some(new_map) {
                eprintln!(
                    "dense: new-session data arrived before its reset barrier; clearing it to preserve session isolation"
                );
            }

            // Commit the session boundary before invoking the backend. If
            // clear fails or panics, stale generations and old-map commands
            // must still remain ineligible after the uncertain backend is
            // discarded.
            state.store.clear();
            state.map_instance_id = Some(new_map);
            state.generation = generation;

            let operation = DenseProcessingOperation::ResetMappingSession;
            let failure = match (&state.mode, state.backend.as_mut()) {
                (DenseMode::DepthStoreOnly, _) => {
                    state.state = ReconState::Nominal;
                    None
                }
                (DenseMode::Tsdf(_), Some(backend)) => match backend.clear() {
                    Ok(()) => {
                        state.state = ReconState::Nominal;
                        None
                    }
                    Err(source) => Some(DenseProcessingError::Tsdf { operation, source }),
                },
                (DenseMode::Tsdf(_), None) => {
                    // There is no reachable backend state to contaminate the
                    // new session, but reconstruction cannot truthfully be
                    // reported as available until a backend is attached.
                    Some(DenseProcessingError::BackendUnavailable { operation })
                }
            };
            if let Some(error) = failure {
                state.backend = None;
                state.state = ReconState::Down;
                return Err(error);
            }
        }
        DenseCommand::ApplyPoseUpdates {
            updates,
            generation,
            timestamp: _,
        } => {
            if generation <= state.generation {
                // Stale pose-update request — skip.
                return Ok(state.stats());
            }
            if !state.accepts_pose_updates(&updates) {
                eprintln!("dense: ignoring empty or mixed-session authoritative pose-update batch");
                return Ok(state.stats());
            }
            state.state = ReconState::Rebuilding { generation };
            let mut updated = 0usize;
            for update in updates {
                updated = updated.saturating_add(usize::from(state.store.update_pose(update)));
            }
            if updated == 0 {
                eprintln!(
                    "dense: authoritative pose update had no retained depth sources; accepting its generation without rebuilding"
                );
            }

            let operation = DenseProcessingOperation::RebuildAfterPoseUpdate;
            let failure = match (&state.mode, state.backend.as_mut()) {
                (DenseMode::DepthStoreOnly, _) => None,
                (DenseMode::Tsdf(tsdf), Some(backend)) => {
                    if updated == 0 {
                        None
                    } else {
                        rebuild_tsdf(tsdf, backend.as_mut(), &state.store)
                            .err()
                            .map(|source| DenseProcessingError::Tsdf { operation, source })
                    }
                }
                (DenseMode::Tsdf(_), None) => {
                    Some(DenseProcessingError::BackendUnavailable { operation })
                }
            };

            if let Some(error) = failure {
                state.backend = None;
                state.state = ReconState::Down;
                return Err(error);
            }
            state.generation = generation;
            if updated > 0 {
                state.stats.rebuild_count = state.stats.rebuild_count.saturating_add(1);
            }
            state.state = ReconState::Nominal;
        }
    }

    Ok(state.stats())
}

// ---------------------------------------------------------------------------
// Worker
// ---------------------------------------------------------------------------

fn process_command_with_recovery(
    state: &mut DenseState,
    cmd: DenseCommand,
    stats_tx: Option<&crate::DropSender<DenseStats>>,
) -> Result<DenseStats, DenseProcessingError> {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        process_dense_command(state, cmd)
    }));
    let result = match result {
        Ok(result) => result,
        Err(payload) => {
            // The backend may have panicked after a partial mutation. Drop it
            // and stop processing instead of reusing uncertain state.
            state.backend = None;
            state.state = ReconState::Down;
            Err(DenseProcessingError::Panicked {
                detail: crate::panic_payload_to_string(payload.as_ref()),
            })
        }
    };
    if let Some(tx) = stats_tx {
        tx.try_send(state.stats());
    }
    result
}

/// Run the dense worker loop.
///
/// `command_rx` retains the producer's accepted command order. Integrations
/// and controls must never arrive through separate transports because reset,
/// rebuild, integration, and removal semantics are causally dependent.
/// `backend_factory` is required when `config.mode` is [`DenseMode::Tsdf`].
/// The optional stats channel is diagnostic and best-effort; this result is
/// the authoritative worker outcome.
pub fn run_dense_worker(
    config: &DenseConfig,
    command_rx: &DenseCommandReceiver,
    backend_factory: Option<TsdfBackendFactory>,
    stats_tx: Option<&crate::DropSender<DenseStats>>,
) -> Result<(), DenseWorkerError> {
    let mut state = DenseState::new(config);
    let initialization = match (&config.mode, backend_factory) {
        (DenseMode::DepthStoreOnly, _) => Ok(()),
        (DenseMode::Tsdf(tsdf), Some(factory)) => {
            let created = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                factory(tsdf.config.clone())
            }));
            match created {
                Ok(Ok(backend)) => {
                    state.set_backend(backend);
                    Ok(())
                }
                Ok(Err(source)) => Err(DenseWorkerError::BackendFactory { source }),
                Err(payload) => Err(DenseWorkerError::BackendFactoryPanicked {
                    detail: crate::panic_payload_to_string(payload.as_ref()),
                }),
            }
        }
        (DenseMode::Tsdf(_), None) => Err(DenseWorkerError::BackendFactoryRequired),
    };
    if let Err(error) = initialization {
        state.backend = None;
        state.state = ReconState::Down;
        if let Some(tx) = stats_tx {
            tx.try_send(state.stats());
        }
        return Err(error);
    }
    if let Some(tx) = stats_tx {
        tx.try_send(state.stats());
    }

    while let Ok(command) = command_rx.recv() {
        process_command_with_recovery(&mut state, command, stats_tx)
            .map_err(|source| DenseWorkerError::Processing { source })?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::dataset::CameraIntrinsics;
    use crate::dense::backend::{Mesh, TsdfError};
    use crate::map::{ImageSize, SlamMap};
    use crate::test_helpers::make_depth_image;
    use crate::{FrameId, Keypoint, Timestamp, WorldToCamera};

    struct FaultBackend {
        fail_clear: bool,
        panic_clear: bool,
        fail_integration_at: Option<usize>,
        panic_integration_at: Option<usize>,
        integrations: usize,
    }

    impl FaultBackend {
        fn healthy() -> Self {
            Self {
                fail_clear: false,
                panic_clear: false,
                fail_integration_at: None,
                panic_integration_at: None,
                integrations: 0,
            }
        }
    }

    impl TsdfBackend for FaultBackend {
        fn integrate(
            &mut self,
            _pose: Pose,
            _depth: &DepthImage,
            _intrinsics: PinholeIntrinsics,
        ) -> Result<(), TsdfError> {
            self.integrations = self.integrations.saturating_add(1);
            assert_ne!(
                self.panic_integration_at,
                Some(self.integrations),
                "fault-injected backend panic"
            );
            if self.fail_integration_at == Some(self.integrations) {
                return Err(TsdfError::Integration("fault injected".to_string()));
            }
            Ok(())
        }

        fn clear(&mut self) -> Result<(), TsdfError> {
            assert!(!self.panic_clear, "fault-injected clear panic");
            if self.fail_clear {
                return Err(TsdfError::Internal("fault-injected clear".to_string()));
            }
            self.integrations = 0;
            Ok(())
        }

        fn extract_mesh(&self) -> Result<Mesh, TsdfError> {
            Ok(Mesh::empty())
        }
    }

    struct CountingClearBackend {
        entries: Arc<AtomicUsize>,
        clears: Arc<AtomicUsize>,
    }

    impl TsdfBackend for CountingClearBackend {
        fn integrate(
            &mut self,
            _pose: Pose,
            _depth: &DepthImage,
            _intrinsics: PinholeIntrinsics,
        ) -> Result<(), TsdfError> {
            self.entries.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        fn clear(&mut self) -> Result<(), TsdfError> {
            self.entries.store(0, Ordering::SeqCst);
            self.clears.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        fn extract_mesh(&self) -> Result<Mesh, TsdfError> {
            Ok(Mesh::empty())
        }
    }

    fn kf(n: u64) -> KeyframeId {
        KeyframeId::for_test(n as usize)
    }

    fn pose_update(keyframe_id: KeyframeId) -> KeyframePoseUpdate {
        KeyframePoseUpdate::new(keyframe_id, WorldToCamera::identity())
    }

    fn map_keyframe(frame: u64) -> (MapInstanceId, KeyframeId) {
        let mut map = SlamMap::new();
        let keyframe_id = map
            .add_keyframe(
                FrameId::new(frame),
                Timestamp::from_nanos(i64::try_from(frame).expect("test frame fits i64")),
                WorldToCamera::identity(),
                ImageSize::try_new(2, 2).expect("nonzero test image size"),
                vec![Keypoint { x: 1.0, y: 1.0 }],
            )
            .expect("test keyframe");
        (map.snapshot().instance_id(), keyframe_id)
    }

    fn transition(old_map: MapInstanceId, new_map: MapInstanceId) -> MappingSessionTransition {
        MappingSessionTransition::try_new(old_map, new_map).expect("distinct test maps")
    }

    fn make_config(cap: usize) -> DenseConfig {
        DenseConfig::try_new(cap, DenseMode::DepthStoreOnly).expect("test dense config")
    }

    fn make_tsdf_config() -> DenseConfig {
        let intrinsics = PinholeIntrinsics::try_from(&CameraIntrinsics {
            fx: 100.0,
            fy: 100.0,
            cx: 1.0,
            cy: 1.0,
            width: 2,
            height: 2,
        })
        .expect("test intrinsics");
        let mode =
            TsdfModeConfig::try_new(TsdfConfig::default(), intrinsics).expect("test TSDF config");
        DenseConfig::try_new(10, DenseMode::Tsdf(mode)).expect("test dense config")
    }

    fn dummy_depth() -> DepthImage {
        make_depth_image(FrameId::new(0), Timestamp::from_nanos(0), 2, 2, 1.0)
    }

    fn depth_store(capacity: usize) -> DepthStore {
        DepthStore::new(NonZeroUsize::new(capacity).expect("nonzero test capacity"))
    }

    fn channel_capacity(capacity: usize) -> ChannelCapacity {
        ChannelCapacity::try_from(capacity).expect("nonzero test channel capacity")
    }

    fn process_dense_command(state: &mut DenseState, command: DenseCommand) -> DenseStats {
        super::process_dense_command(state, command).expect("test dense command must succeed")
    }

    // -- DepthStore tests --

    #[test]
    fn dense_config_rejects_zero_store_capacity() {
        assert_eq!(
            DenseConfig::try_new(0, DenseMode::DepthStoreOnly).unwrap_err(),
            DenseConfigError::ZeroMaxStoredKeyframes
        );
    }

    #[test]
    fn depth_store_insert_and_retrieve() {
        let mut store = depth_store(10);
        let id = kf(0);
        let depth = dummy_depth();
        store.insert(id, depth.clone());
        assert_eq!(store.len(), 1);
        assert!(store.get(id).is_some());
    }

    #[test]
    fn depth_store_capacity_two_evicts_by_insertion_order() {
        let mut store = depth_store(2);
        let id1 = kf(0);
        let id2 = kf(1);
        let id3 = kf(2);
        store.insert(id1, dummy_depth());
        store.insert(id2, dummy_depth());
        assert_eq!(store.len(), 2);
        store.insert(id1, dummy_depth());
        store.insert(id3, dummy_depth());
        assert_eq!(store.len(), 2);
        assert!(
            store.get(id1).is_none(),
            "updating the oldest entry must not change its eviction position"
        );
        assert!(store.get(id2).is_some());
        assert!(store.get(id3).is_some());
    }

    #[test]
    fn remove_keyframe_known_id() {
        let mut store = depth_store(10);
        let id = kf(0);
        store.insert(id, dummy_depth());
        assert_eq!(store.len(), 1);
        store.remove(id);
        assert_eq!(store.len(), 0);
        assert!(store.get(id).is_none());
    }

    #[test]
    fn remove_keyframe_unknown_id_is_noop() {
        let mut store = depth_store(10);
        let id_in = kf(0);
        let id_out = kf(1);
        store.insert(id_in, dummy_depth());
        store.remove(id_out); // unknown — no panic, no change
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn depth_store_clear_removes_map_and_eviction_order() {
        let mut store = depth_store(1);
        store.insert(kf(0), dummy_depth());

        store.clear();
        store.insert(kf(1), dummy_depth());

        assert_eq!(store.len(), 1);
        assert!(store.contains(kf(1)));
    }

    #[test]
    fn depth_store_capacity_one_keeps_only_latest_insertion() {
        let mut store = depth_store(1);
        let id1 = kf(0);
        let id2 = kf(1);
        store.insert(id1, dummy_depth());
        store.insert(id2, dummy_depth());
        assert_eq!(store.len(), 1);
        assert!(store.get(id1).is_none());
        assert!(store.get(id2).is_some());
    }

    // -- process_dense_command tests --

    #[test]
    fn integrate_increments_count() {
        let config = make_config(10);
        let mut state = DenseState::new(&config);
        let stats = process_dense_command(
            &mut state,
            DenseCommand::IntegrateKeyframe {
                keyframe_id: kf(0),
                pose: WorldToCamera::identity(),
                depth: dummy_depth(),
                timestamp: Timestamp::from_nanos(0),
            },
        );
        assert_eq!(stats.integrated_count, 1);
        assert_eq!(stats.stored_keyframes, 1);
    }

    #[test]
    fn failed_tsdf_integration_is_not_counted_as_success() {
        let config = make_tsdf_config();
        let mut state = DenseState::new(&config);
        state.set_backend(Box::new(FaultBackend {
            fail_integration_at: Some(1),
            ..FaultBackend::healthy()
        }));

        let error = super::process_dense_command(
            &mut state,
            DenseCommand::IntegrateKeyframe {
                keyframe_id: kf(0),
                pose: WorldToCamera::identity(),
                depth: dummy_depth(),
                timestamp: Timestamp::from_nanos(0),
            },
        )
        .expect_err("fault-injected integration must retain its typed error");
        let stats = state.stats();

        assert!(matches!(
            error,
            DenseProcessingError::Tsdf {
                operation: DenseProcessingOperation::IntegrateKeyframe,
                source: TsdfError::Integration(ref detail),
            } if detail == "fault injected"
        ));
        assert_eq!(stats.integrated_count, 0);
        assert_eq!(stats.stored_keyframes, 1);
        assert_eq!(stats.state, ReconState::Down);
        assert!(!state.has_backend());
    }

    #[test]
    fn tsdf_state_is_not_nominal_until_backend_is_attached() {
        let config = make_tsdf_config();
        let mut state = DenseState::new(&config);
        assert_eq!(state.state(), ReconState::AwaitingBackend);

        state.set_backend(Box::new(FaultBackend::healthy()));
        assert_eq!(state.state(), ReconState::Nominal);
    }

    #[test]
    fn direct_tsdf_processing_without_backend_is_typed_and_fail_closed() {
        let config = make_tsdf_config();
        let mut state = DenseState::new(&config);

        let error = super::process_dense_command(
            &mut state,
            DenseCommand::IntegrateKeyframe {
                keyframe_id: kf(0),
                pose: WorldToCamera::identity(),
                depth: dummy_depth(),
                timestamp: Timestamp::from_nanos(0),
            },
        )
        .expect_err("TSDF integration requires an attached backend");

        assert!(matches!(
            error,
            DenseProcessingError::BackendUnavailable {
                operation: DenseProcessingOperation::IntegrateKeyframe,
            }
        ));
        assert_eq!(state.state(), ReconState::Down);
        assert_eq!(state.stats().stored_keyframes, 1);
    }

    #[test]
    fn tsdf_worker_constructs_backend_on_worker_thread() {
        let config = make_tsdf_config();
        let (command_tx, command_rx, _) =
            dense_command_channel(channel_capacity(1), channel_capacity(1), Duration::ZERO)
                .expect("command channel");
        let (stats_tx, stats_rx, _) = crate::bounded_channel(
            crate::ChannelCapacity::try_from(4_usize).expect("stats capacity"),
            crate::DropPolicy::DropNewest,
        );
        assert_eq!(
            command_tx.route(DenseCommand::IntegrateKeyframe {
                keyframe_id: kf(0),
                pose: WorldToCamera::identity(),
                depth: dummy_depth(),
                timestamp: Timestamp::from_nanos(0),
            }),
            DenseCommandSendOutcome::Enqueued
        );
        drop(command_tx);

        let worker = std::thread::spawn(move || {
            let factory: TsdfBackendFactory = Box::new(|_| Ok(Box::new(FaultBackend::healthy())));
            run_dense_worker(&config, &command_rx, Some(factory), Some(&stats_tx))
                .expect("healthy dense worker");
        });
        let initial = stats_rx
            .as_receiver()
            .recv_timeout(std::time::Duration::from_secs(1))
            .expect("initial worker state");
        let integrated = stats_rx
            .as_receiver()
            .recv_timeout(std::time::Duration::from_secs(1))
            .expect("integration state");
        worker.join().expect("dense worker");

        assert_eq!(initial.state, ReconState::Nominal);
        assert_eq!(integrated.state, ReconState::Nominal);
        assert_eq!(integrated.integrated_count, 1);
    }

    #[test]
    fn tsdf_worker_without_factory_reports_down() {
        let config = make_tsdf_config();
        let (command_tx, command_rx, _) =
            dense_command_channel(channel_capacity(1), channel_capacity(1), Duration::ZERO)
                .expect("command channel");
        let (stats_tx, stats_rx, _) = crate::bounded_channel(
            crate::ChannelCapacity::try_from(1_usize).expect("stats capacity"),
            crate::DropPolicy::DropNewest,
        );
        drop(command_tx);

        let error = run_dense_worker(&config, &command_rx, None, Some(&stats_tx))
            .expect_err("missing factory must be authoritative worker failure");

        let stats = stats_rx.try_recv().expect("initial worker state");
        assert_eq!(stats.state, ReconState::Down);
        assert!(matches!(error, DenseWorkerError::BackendFactoryRequired));
    }

    #[test]
    fn tsdf_worker_factory_error_preserves_source_and_reports_down() {
        let config = make_tsdf_config();
        let (command_tx, command_rx, _) =
            dense_command_channel(channel_capacity(1), channel_capacity(1), Duration::ZERO)
                .expect("command channel");
        let (stats_tx, stats_rx, _) =
            crate::bounded_channel(channel_capacity(1), crate::DropPolicy::DropNewest);
        drop(command_tx);
        let factory: TsdfBackendFactory = Box::new(|_| {
            Err(TsdfError::Internal(
                "fault-injected factory failure".to_owned(),
            ))
        });

        let error = run_dense_worker(&config, &command_rx, Some(factory), Some(&stats_tx))
            .expect_err("factory failure must be returned");

        assert!(matches!(
            error,
            DenseWorkerError::BackendFactory {
                source: TsdfError::Internal(ref detail),
            } if detail == "fault-injected factory failure"
        ));
        assert_eq!(
            stats_rx.try_recv().expect("terminal worker state").state,
            ReconState::Down
        );
    }

    #[test]
    fn tsdf_worker_factory_panic_is_typed_and_reports_down() {
        let config = make_tsdf_config();
        let (command_tx, command_rx, _) =
            dense_command_channel(channel_capacity(1), channel_capacity(1), Duration::ZERO)
                .expect("command channel");
        let (stats_tx, stats_rx, _) =
            crate::bounded_channel(channel_capacity(1), crate::DropPolicy::DropNewest);
        drop(command_tx);
        let factory: TsdfBackendFactory = Box::new(|_| panic!("fault-injected factory panic"));

        let error = run_dense_worker(&config, &command_rx, Some(factory), Some(&stats_tx))
            .expect_err("factory panic must be returned");

        assert!(matches!(
            error,
            DenseWorkerError::BackendFactoryPanicked { ref detail }
                if detail.contains("fault-injected factory panic")
        ));
        assert_eq!(
            stats_rx.try_recv().expect("terminal worker state").state,
            ReconState::Down
        );
    }

    #[test]
    fn tsdf_worker_processing_error_preserves_source_and_terminal_stats() {
        let config = make_tsdf_config();
        let (command_tx, command_rx, _) =
            dense_command_channel(channel_capacity(1), channel_capacity(1), Duration::ZERO)
                .expect("command channel");
        let (stats_tx, stats_rx, _) =
            crate::bounded_channel(channel_capacity(2), crate::DropPolicy::DropNewest);
        assert_eq!(
            command_tx.route(DenseCommand::IntegrateKeyframe {
                keyframe_id: kf(0),
                pose: WorldToCamera::identity(),
                depth: dummy_depth(),
                timestamp: Timestamp::from_nanos(0),
            }),
            DenseCommandSendOutcome::Enqueued
        );
        drop(command_tx);
        let factory: TsdfBackendFactory = Box::new(|_| {
            Ok(Box::new(FaultBackend {
                fail_integration_at: Some(1),
                ..FaultBackend::healthy()
            }))
        });

        let error = run_dense_worker(&config, &command_rx, Some(factory), Some(&stats_tx))
            .expect_err("processing failure must be returned");

        assert!(matches!(
            error,
            DenseWorkerError::Processing {
                source: DenseProcessingError::Tsdf {
                    operation: DenseProcessingOperation::IntegrateKeyframe,
                    source: TsdfError::Integration(ref detail),
                },
            } if detail == "fault injected"
        ));
        let terminal = std::iter::from_fn(|| stats_rx.try_recv().ok())
            .last()
            .expect("initial and terminal stats");
        assert_eq!(terminal.state, ReconState::Down);
        assert_eq!(terminal.integrated_count, 0);
    }

    #[test]
    fn ordered_queue_reserves_control_capacity_and_releases_data_quota_on_receive() {
        let (sender, receiver, stats) =
            dense_command_channel(channel_capacity(1), channel_capacity(1), Duration::ZERO)
                .expect("command channel");
        let first = kf(0);
        let second = kf(1);

        assert_eq!(
            sender.route(DenseCommand::IntegrateKeyframe {
                keyframe_id: first,
                pose: WorldToCamera::identity(),
                depth: dummy_depth(),
                timestamp: Timestamp::from_nanos(0),
            }),
            DenseCommandSendOutcome::Enqueued
        );
        assert_eq!(
            sender.route(DenseCommand::IntegrateKeyframe {
                keyframe_id: second,
                pose: WorldToCamera::identity(),
                depth: dummy_depth(),
                timestamp: Timestamp::from_nanos(0),
            }),
            DenseCommandSendOutcome::IntegrationDroppedNewest
        );
        assert_eq!(
            sender.route(DenseCommand::RemoveKeyframe {
                keyframe_id: first,
                timestamp: Timestamp::from_nanos(0)
            }),
            DenseCommandSendOutcome::Enqueued,
            "the integration quota must leave the reserved control slot available"
        );

        assert!(matches!(
            receiver.try_recv(),
            Ok(DenseCommand::IntegrateKeyframe { keyframe_id, .. }) if keyframe_id == first
        ));
        assert_eq!(
            sender.route(DenseCommand::IntegrateKeyframe {
                keyframe_id: second,
                pose: WorldToCamera::identity(),
                depth: dummy_depth(),
                timestamp: Timestamp::from_nanos(0),
            }),
            DenseCommandSendOutcome::Enqueued,
            "dequeueing an integration must release its data slot"
        );
        assert!(matches!(
            receiver.try_recv(),
            Ok(DenseCommand::RemoveKeyframe { keyframe_id, .. }) if keyframe_id == first
        ));
        assert!(matches!(
            receiver.try_recv(),
            Ok(DenseCommand::IntegrateKeyframe { keyframe_id, .. }) if keyframe_id == second
        ));
        assert_eq!(
            stats.snapshot(),
            DenseCommandQueueStats {
                commands_enqueued: 3,
                integrations_dropped_newest: 1,
                ..DenseCommandQueueStats::default()
            }
        );
    }

    #[test]
    fn ordered_queue_reports_control_timeout_and_disconnects() {
        let (sender, receiver, stats) =
            dense_command_channel(channel_capacity(1), channel_capacity(1), Duration::ZERO)
                .expect("command channel");
        assert_eq!(
            sender.route(DenseCommand::RemoveKeyframe {
                keyframe_id: kf(0),
                timestamp: Timestamp::from_nanos(0)
            }),
            DenseCommandSendOutcome::Enqueued
        );
        assert_eq!(
            sender.route(DenseCommand::RemoveKeyframe {
                keyframe_id: kf(1),
                timestamp: Timestamp::from_nanos(0)
            }),
            DenseCommandSendOutcome::Enqueued,
            "controls may use idle data capacity while total backlog stays bounded"
        );
        assert_eq!(
            sender.route(DenseCommand::RemoveKeyframe {
                keyframe_id: kf(2),
                timestamp: Timestamp::from_nanos(0)
            }),
            DenseCommandSendOutcome::ControlTimedOut
        );
        drop(receiver);
        assert_eq!(
            sender.route(DenseCommand::RemoveKeyframe {
                keyframe_id: kf(3),
                timestamp: Timestamp::from_nanos(0)
            }),
            DenseCommandSendOutcome::Disconnected
        );
        assert_eq!(
            sender.route(DenseCommand::IntegrateKeyframe {
                keyframe_id: kf(4),
                pose: WorldToCamera::identity(),
                depth: dummy_depth(),
                timestamp: Timestamp::from_nanos(0),
            }),
            DenseCommandSendOutcome::Disconnected
        );
        assert_eq!(
            stats.snapshot(),
            DenseCommandQueueStats {
                commands_enqueued: 2,
                controls_timed_out: 1,
                disconnected: 2,
                ..DenseCommandQueueStats::default()
            }
        );
    }

    #[test]
    fn ordered_queue_capacity_overflow_is_rejected_before_channel_creation() {
        let error = dense_command_channel(
            channel_capacity(usize::MAX),
            channel_capacity(1),
            Duration::ZERO,
        )
        .expect_err("summed capacity must fit usize");
        assert_eq!(
            error,
            DenseCommandChannelError::CapacityOverflow {
                data_capacity: usize::MAX,
                reserved_control_capacity: 1,
            }
        );
    }

    fn run_ordered_depth_store_worker(commands: Vec<DenseCommand>) -> DenseStats {
        let (sender, receiver, _) =
            dense_command_channel(channel_capacity(4), channel_capacity(4), Duration::ZERO)
                .expect("command channel");
        for command in commands {
            assert_eq!(
                sender.route(command),
                DenseCommandSendOutcome::Enqueued,
                "test command must fit the queue"
            );
        }
        drop(sender);
        let (stats_tx, stats_rx, _) =
            crate::bounded_channel(channel_capacity(8), crate::DropPolicy::DropNewest);

        run_dense_worker(&make_config(10), &receiver, None, Some(&stats_tx))
            .expect("depth-store worker");

        std::iter::from_fn(|| stats_rx.try_recv().ok())
            .last()
            .expect("worker emits initial and per-command stats")
    }

    #[test]
    fn ordered_worker_reset_then_integrate_retains_new_session_keyframe() {
        let old_map = SlamMap::new().snapshot().instance_id();
        let (new_map, new_keyframe) = map_keyframe(1);

        let final_stats = run_ordered_depth_store_worker(vec![
            DenseCommand::ResetMappingSession {
                transition: transition(old_map, new_map),
                generation: 1,
                timestamp: Timestamp::from_nanos(0),
            },
            DenseCommand::IntegrateKeyframe {
                keyframe_id: new_keyframe,
                pose: WorldToCamera::identity(),
                depth: dummy_depth(),
                timestamp: Timestamp::from_nanos(0),
            },
        ]);

        assert_eq!(final_stats.integrated_count, 1);
        assert_eq!(final_stats.stored_keyframes, 1);
    }

    #[test]
    fn ordered_worker_reset_integrate_remove_leaves_keyframe_removed() {
        let old_map = SlamMap::new().snapshot().instance_id();
        let (new_map, new_keyframe) = map_keyframe(1);

        let final_stats = run_ordered_depth_store_worker(vec![
            DenseCommand::ResetMappingSession {
                transition: transition(old_map, new_map),
                generation: 1,
                timestamp: Timestamp::from_nanos(0),
            },
            DenseCommand::IntegrateKeyframe {
                keyframe_id: new_keyframe,
                pose: WorldToCamera::identity(),
                depth: dummy_depth(),
                timestamp: Timestamp::from_nanos(0),
            },
            DenseCommand::RemoveKeyframe {
                keyframe_id: new_keyframe,
                timestamp: Timestamp::from_nanos(0),
            },
        ]);

        assert_eq!(final_stats.integrated_count, 1);
        assert_eq!(final_stats.removed_count, 1);
        assert_eq!(final_stats.stored_keyframes, 0);
    }

    #[test]
    fn pose_updates_ignore_unretained_sources_and_rebuild_retained_sources() {
        let config = make_tsdf_config();
        let mut state = DenseState::new(&config);
        state.set_backend(Box::new(FaultBackend::healthy()));
        let present = kf(0);
        state.store.insert(present, dummy_depth());

        let stats = process_dense_command(
            &mut state,
            DenseCommand::ApplyPoseUpdates {
                updates: vec![pose_update(present), pose_update(kf(1))],
                generation: 4,
                timestamp: Timestamp::from_nanos(0),
            },
        );

        assert_eq!(state.generation, 4);
        assert_eq!(stats.rebuild_count, 1);
        assert_eq!(stats.state, ReconState::Nominal);
    }

    #[test]
    fn failed_or_partial_rebuild_drops_backend_and_goes_down() {
        for backend in [
            FaultBackend {
                fail_clear: true,
                ..FaultBackend::healthy()
            },
            FaultBackend {
                fail_integration_at: Some(2),
                ..FaultBackend::healthy()
            },
        ] {
            let config = make_tsdf_config();
            let mut state = DenseState::new(&config);
            state.set_backend(Box::new(backend));
            let first = kf(0);
            let second = kf(1);
            state.store.insert(first, dummy_depth());
            state.store.insert(second, dummy_depth());

            let error = super::process_dense_command(
                &mut state,
                DenseCommand::ApplyPoseUpdates {
                    updates: vec![pose_update(first), pose_update(second)],
                    generation: 1,
                    timestamp: Timestamp::from_nanos(0),
                },
            )
            .expect_err("fault-injected rebuild must retain its typed error");
            let stats = state.stats();

            assert!(matches!(
                error,
                DenseProcessingError::Tsdf {
                    operation: DenseProcessingOperation::RebuildAfterPoseUpdate,
                    ..
                }
            ));
            assert_eq!(state.generation, 0);
            assert_eq!(stats.rebuild_count, 0);
            assert_eq!(stats.state, ReconState::Down);
            assert!(!state.has_backend());
        }
    }

    #[test]
    fn backend_panic_never_reuses_half_mutated_state() {
        let config = make_tsdf_config();
        let mut state = DenseState::new(&config);
        state.set_backend(Box::new(FaultBackend {
            panic_integration_at: Some(1),
            ..FaultBackend::healthy()
        }));

        let error = process_command_with_recovery(
            &mut state,
            DenseCommand::IntegrateKeyframe {
                keyframe_id: kf(0),
                pose: WorldToCamera::identity(),
                depth: dummy_depth(),
                timestamp: Timestamp::from_nanos(0),
            },
            None,
        )
        .expect_err("backend panic must be returned");

        assert!(matches!(
            error,
            DenseProcessingError::Panicked { ref detail }
                if detail.contains("fault-injected backend panic")
        ));
        assert_eq!(state.state(), ReconState::Down);
        assert!(!state.has_backend());
    }

    #[test]
    fn remove_increments_count() {
        let config = make_config(10);
        let mut state = DenseState::new(&config);
        let id = kf(0);
        process_dense_command(
            &mut state,
            DenseCommand::IntegrateKeyframe {
                keyframe_id: id,
                pose: WorldToCamera::identity(),
                depth: dummy_depth(),
                timestamp: Timestamp::from_nanos(0),
            },
        );
        let stats = process_dense_command(
            &mut state,
            DenseCommand::RemoveKeyframe {
                keyframe_id: id,
                timestamp: Timestamp::from_nanos(0),
            },
        );
        assert_eq!(stats.removed_count, 1);
        assert_eq!(stats.stored_keyframes, 0);
    }

    #[test]
    fn mapping_session_reset_prevents_delayed_old_data_from_mixing() {
        let config = make_config(10);
        let mut state = DenseState::new(&config);
        let (old_map, old_keyframe) = map_keyframe(1);
        let (new_map, new_keyframe) = map_keyframe(2);

        process_dense_command(
            &mut state,
            DenseCommand::IntegrateKeyframe {
                keyframe_id: old_keyframe,
                pose: WorldToCamera::identity(),
                depth: dummy_depth(),
                timestamp: Timestamp::from_nanos(0),
            },
        );
        let reset_stats = process_dense_command(
            &mut state,
            DenseCommand::ResetMappingSession {
                transition: transition(old_map, new_map),
                generation: 1,
                timestamp: Timestamp::from_nanos(0),
            },
        );

        assert_eq!(reset_stats.stored_keyframes, 0);
        assert_eq!(state.map_instance_id, Some(new_map));
        assert_eq!(state.generation, 1);

        process_dense_command(
            &mut state,
            DenseCommand::IntegrateKeyframe {
                keyframe_id: old_keyframe,
                pose: WorldToCamera::identity(),
                depth: dummy_depth(),
                timestamp: Timestamp::from_nanos(0),
            },
        );
        let new_stats = process_dense_command(
            &mut state,
            DenseCommand::IntegrateKeyframe {
                keyframe_id: new_keyframe,
                pose: WorldToCamera::identity(),
                depth: dummy_depth(),
                timestamp: Timestamp::from_nanos(0),
            },
        );

        assert_eq!(new_stats.integrated_count, 2);
        assert_eq!(new_stats.stored_keyframes, 1);
        assert!(!state.store.contains(old_keyframe));
        assert!(state.store.contains(new_keyframe));
    }

    #[test]
    fn reset_for_inactive_old_map_preserves_current_session() {
        let config = make_config(10);
        let mut state = DenseState::new(&config);
        let (active_map, active_keyframe) = map_keyframe(1);
        let (inactive_map, _) = map_keyframe(2);
        let (proposed_map, _) = map_keyframe(3);
        process_dense_command(
            &mut state,
            DenseCommand::IntegrateKeyframe {
                keyframe_id: active_keyframe,
                pose: WorldToCamera::identity(),
                depth: dummy_depth(),
                timestamp: Timestamp::from_nanos(0),
            },
        );

        let stats = process_dense_command(
            &mut state,
            DenseCommand::ResetMappingSession {
                transition: transition(inactive_map, proposed_map),
                generation: 1,
                timestamp: Timestamp::from_nanos(0),
            },
        );

        assert_eq!(state.map_instance_id, Some(active_map));
        assert_eq!(state.generation, 0);
        assert_eq!(stats.stored_keyframes, 1);
        assert!(state.store.contains(active_keyframe));
    }

    #[test]
    fn failed_session_clear_drops_backend_but_commits_isolation_boundary() {
        let config = make_tsdf_config();
        let mut state = DenseState::new(&config);
        state.set_backend(Box::new(FaultBackend {
            fail_clear: true,
            ..FaultBackend::healthy()
        }));
        let (old_map, old_keyframe) = map_keyframe(1);
        let (new_map, new_keyframe) = map_keyframe(2);
        process_dense_command(
            &mut state,
            DenseCommand::IntegrateKeyframe {
                keyframe_id: old_keyframe,
                pose: WorldToCamera::identity(),
                depth: dummy_depth(),
                timestamp: Timestamp::from_nanos(0),
            },
        );

        let error = super::process_dense_command(
            &mut state,
            DenseCommand::ResetMappingSession {
                transition: transition(old_map, new_map),
                generation: 7,
                timestamp: Timestamp::from_nanos(0),
            },
        )
        .expect_err("session-clear failure must be returned");
        let failed = state.stats();

        assert!(matches!(
            error,
            DenseProcessingError::Tsdf {
                operation: DenseProcessingOperation::ResetMappingSession,
                source: TsdfError::Internal(ref detail),
            } if detail == "fault-injected clear"
        ));
        assert_eq!(failed.state, ReconState::Down);
        assert_eq!(failed.stored_keyframes, 0);
        assert_eq!(state.map_instance_id, Some(new_map));
        assert_eq!(state.generation, 7);
        assert!(!state.has_backend());

        state.set_backend(Box::new(FaultBackend::healthy()));
        process_dense_command(
            &mut state,
            DenseCommand::IntegrateKeyframe {
                keyframe_id: old_keyframe,
                pose: WorldToCamera::identity(),
                depth: dummy_depth(),
                timestamp: Timestamp::from_nanos(0),
            },
        );
        let recovered = process_dense_command(
            &mut state,
            DenseCommand::IntegrateKeyframe {
                keyframe_id: new_keyframe,
                pose: WorldToCamera::identity(),
                depth: dummy_depth(),
                timestamp: Timestamp::from_nanos(0),
            },
        );
        assert_eq!(recovered.state, ReconState::Nominal);
        assert_eq!(recovered.stored_keyframes, 1);
        assert!(!state.store.contains(old_keyframe));
        assert!(state.store.contains(new_keyframe));
    }

    #[test]
    fn reset_clears_new_session_data_that_raced_ahead_of_the_barrier() {
        let config = make_tsdf_config();
        let mut state = DenseState::new(&config);
        // Model one old-session volume entry that predates the host-side
        // session binding, then let one new-session integration race ahead.
        let backend_entries = Arc::new(AtomicUsize::new(1));
        let clear_count = Arc::new(AtomicUsize::new(0));
        state.set_backend(Box::new(CountingClearBackend {
            entries: Arc::clone(&backend_entries),
            clears: Arc::clone(&clear_count),
        }));
        let (old_map, _) = map_keyframe(1);
        let (new_map, new_keyframe) = map_keyframe(2);

        process_dense_command(
            &mut state,
            DenseCommand::IntegrateKeyframe {
                keyframe_id: new_keyframe,
                pose: WorldToCamera::identity(),
                depth: dummy_depth(),
                timestamp: Timestamp::from_nanos(0),
            },
        );
        assert_eq!(backend_entries.load(Ordering::SeqCst), 2);
        assert_eq!(state.store.len(), 1);

        let reset = process_dense_command(
            &mut state,
            DenseCommand::ResetMappingSession {
                transition: transition(old_map, new_map),
                generation: 1,
                timestamp: Timestamp::from_nanos(0),
            },
        );

        assert_eq!(clear_count.load(Ordering::SeqCst), 1);
        assert_eq!(backend_entries.load(Ordering::SeqCst), 0);
        assert_eq!(reset.stored_keyframes, 0);
        assert_eq!(reset.state, ReconState::Nominal);
        assert_eq!(state.map_instance_id, Some(new_map));
        assert_eq!(state.generation, 1);
    }

    #[test]
    fn panicking_session_clear_keeps_the_committed_isolation_boundary() {
        let config = make_tsdf_config();
        let mut state = DenseState::new(&config);
        state.set_backend(Box::new(FaultBackend {
            panic_clear: true,
            ..FaultBackend::healthy()
        }));
        let (old_map, old_keyframe) = map_keyframe(1);
        let (new_map, _) = map_keyframe(2);
        process_dense_command(
            &mut state,
            DenseCommand::IntegrateKeyframe {
                keyframe_id: old_keyframe,
                pose: WorldToCamera::identity(),
                depth: dummy_depth(),
                timestamp: Timestamp::from_nanos(0),
            },
        );

        let error = process_command_with_recovery(
            &mut state,
            DenseCommand::ResetMappingSession {
                transition: transition(old_map, new_map),
                generation: 9,
                timestamp: Timestamp::from_nanos(0),
            },
            None,
        )
        .expect_err("panicking clear must be returned");

        assert!(matches!(
            error,
            DenseProcessingError::Panicked { ref detail }
                if detail.contains("fault-injected clear panic")
        ));
        assert_eq!(state.state(), ReconState::Down);
        assert_eq!(state.store.len(), 0);
        assert_eq!(state.map_instance_id, Some(new_map));
        assert_eq!(state.generation, 9);
        assert!(!state.has_backend());
    }

    #[test]
    fn reset_generation_and_map_identity_order_pose_updates() {
        let config = make_config(10);
        let mut state = DenseState::new(&config);
        let (old_map, old_keyframe) = map_keyframe(1);
        let (new_map, new_keyframe) = map_keyframe(2);
        process_dense_command(
            &mut state,
            DenseCommand::IntegrateKeyframe {
                keyframe_id: old_keyframe,
                pose: WorldToCamera::identity(),
                depth: dummy_depth(),
                timestamp: Timestamp::from_nanos(0),
            },
        );
        process_dense_command(
            &mut state,
            DenseCommand::ResetMappingSession {
                transition: transition(old_map, new_map),
                generation: 2,
                timestamp: Timestamp::from_nanos(0),
            },
        );

        process_dense_command(
            &mut state,
            DenseCommand::ApplyPoseUpdates {
                updates: vec![pose_update(old_keyframe)],
                generation: 1,
                timestamp: Timestamp::from_nanos(0),
            },
        );
        process_dense_command(
            &mut state,
            DenseCommand::ApplyPoseUpdates {
                updates: vec![pose_update(old_keyframe)],
                generation: 3,
                timestamp: Timestamp::from_nanos(0),
            },
        );
        assert_eq!(state.generation, 2);
        assert_eq!(state.stats.rebuild_count, 0);

        process_dense_command(
            &mut state,
            DenseCommand::IntegrateKeyframe {
                keyframe_id: new_keyframe,
                pose: WorldToCamera::identity(),
                depth: dummy_depth(),
                timestamp: Timestamp::from_nanos(0),
            },
        );
        let current = process_dense_command(
            &mut state,
            DenseCommand::ApplyPoseUpdates {
                updates: vec![pose_update(new_keyframe)],
                generation: 3,
                timestamp: Timestamp::from_nanos(0),
            },
        );
        assert_eq!(state.generation, 3);
        assert_eq!(current.rebuild_count, 1);
        assert!(state.store.contains(new_keyframe));
    }

    #[test]
    fn retained_pose_update_increments_rebuild_count() {
        let config = make_config(10);
        let mut state = DenseState::new(&config);
        let keyframe_id = kf(0);
        process_dense_command(
            &mut state,
            DenseCommand::IntegrateKeyframe {
                keyframe_id,
                pose: WorldToCamera::identity(),
                depth: dummy_depth(),
                timestamp: Timestamp::from_nanos(0),
            },
        );
        let stats = process_dense_command(
            &mut state,
            DenseCommand::ApplyPoseUpdates {
                updates: vec![pose_update(keyframe_id)],
                generation: 1,
                timestamp: Timestamp::from_nanos(1),
            },
        );
        assert_eq!(stats.rebuild_count, 1);
        assert_eq!(stats.state, ReconState::Nominal);
    }

    #[test]
    fn unretained_pose_update_advances_generation_without_reporting_a_rebuild() {
        let config = make_config(10);
        let mut state = DenseState::new(&config);

        let stats = process_dense_command(
            &mut state,
            DenseCommand::ApplyPoseUpdates {
                updates: vec![pose_update(kf(0))],
                generation: 1,
                timestamp: Timestamp::from_nanos(1),
            },
        );

        assert_eq!(state.generation, 1);
        assert_eq!(stats.rebuild_count, 0);
        assert_eq!(stats.stored_keyframes, 0);
        assert_eq!(stats.state, ReconState::Nominal);
    }

    #[test]
    fn stale_pose_update_is_skipped() {
        let config = make_config(10);
        let mut state = DenseState::new(&config);
        let keyframe_id = kf(0);
        process_dense_command(
            &mut state,
            DenseCommand::IntegrateKeyframe {
                keyframe_id,
                pose: WorldToCamera::identity(),
                depth: dummy_depth(),
                timestamp: Timestamp::from_nanos(0),
            },
        );
        process_dense_command(
            &mut state,
            DenseCommand::ApplyPoseUpdates {
                updates: vec![pose_update(keyframe_id)],
                generation: 5,
                timestamp: Timestamp::from_nanos(1),
            },
        );
        let stats = process_dense_command(
            &mut state,
            DenseCommand::ApplyPoseUpdates {
                updates: vec![pose_update(keyframe_id)],
                generation: 3, // stale
                timestamp: Timestamp::from_nanos(2),
            },
        );
        assert_eq!(stats.rebuild_count, 1, "stale update should not increment");
    }

    #[test]
    fn successive_authoritative_pose_generations_are_applied() {
        let config = make_config(10);
        let mut state = DenseState::new(&config);
        let keyframe_id = kf(0);
        process_dense_command(
            &mut state,
            DenseCommand::IntegrateKeyframe {
                keyframe_id,
                pose: WorldToCamera::identity(),
                depth: dummy_depth(),
                timestamp: Timestamp::from_nanos(0),
            },
        );
        process_dense_command(
            &mut state,
            DenseCommand::ApplyPoseUpdates {
                updates: vec![pose_update(keyframe_id)],
                generation: 1,
                timestamp: Timestamp::from_nanos(1),
            },
        );
        let stats = process_dense_command(
            &mut state,
            DenseCommand::ApplyPoseUpdates {
                updates: vec![pose_update(keyframe_id)],
                generation: 2,
                timestamp: Timestamp::from_nanos(2),
            },
        );
        assert_eq!(stats.rebuild_count, 2);
    }

    #[test]
    fn down_state_drains_without_processing() {
        let config = make_config(10);
        let mut state = DenseState::new(&config);
        state.state = ReconState::Down;
        let stats = process_dense_command(
            &mut state,
            DenseCommand::IntegrateKeyframe {
                keyframe_id: kf(0),
                pose: WorldToCamera::identity(),
                depth: dummy_depth(),
                timestamp: Timestamp::from_nanos(0),
            },
        );
        assert_eq!(stats.integrated_count, 0, "Down state should not process");
        assert_eq!(stats.stored_keyframes, 0);
    }

    #[test]
    fn empty_pose_update_batch_is_ignored_without_advancing_generation() {
        let config = make_config(10);
        let mut state = DenseState::new(&config);
        let stats = process_dense_command(
            &mut state,
            DenseCommand::ApplyPoseUpdates {
                updates: vec![],
                generation: 1,
                timestamp: Timestamp::from_nanos(0),
            },
        );
        assert_eq!(stats.state, ReconState::Nominal);
        assert_eq!(stats.rebuild_count, 0);
        assert_eq!(state.generation, 0);
    }

    #[test]
    fn dense_stats_default_is_zero() {
        let stats = DenseStats::default();
        assert_eq!(stats.integrated_count, 0);
        assert_eq!(stats.removed_count, 0);
        assert_eq!(stats.rebuild_count, 0);
        assert_eq!(stats.stored_keyframes, 0);
        assert_eq!(stats.state, ReconState::Nominal);
    }

    #[test]
    fn dense_command_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<DenseCommand>();
    }

    #[test]
    fn dense_command_queue_endpoints_can_move_to_their_own_threads() {
        fn assert_send<T: Send>() {}
        assert_send::<DenseCommandSender>();
        assert_send::<DenseCommandReceiver>();
    }

    #[test]
    fn dense_stats_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<DenseStats>();
    }
}
