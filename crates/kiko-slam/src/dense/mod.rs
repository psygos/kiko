pub mod backend;
pub mod command_mapper;
pub mod ring_buffer;

use std::collections::{HashMap, VecDeque};
use std::marker::PhantomData;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

use crate::dense::backend::{TsdfBackend, TsdfBackendFactory, TsdfConfig};
use crate::map::{KeyframeId, MapInstanceId};
use crate::{ChannelCapacity, DepthImage, MappingSessionTransition, PinholeIntrinsics, Pose};

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
        pose: Pose,
        depth: DepthImage,
    },
    RemoveKeyframe {
        keyframe_id: KeyframeId,
    },
    /// Establish a hard boundary between independent maps.
    ///
    /// An accepted command clears the host depth store and advances
    /// `generation` even when TSDF clearing fails. On that failure the
    /// uncertain backend is dropped and reconstruction reports [`ReconState::Down`].
    ResetMappingSession {
        transition: MappingSessionTransition,
        generation: u64,
    },
    RebuildFromSnapshot {
        corrected_poses: Vec<(KeyframeId, Pose)>,
        generation: u64,
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
    Degraded { generation: u64 },
    Down,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RebuildPolicy {
    /// Rebuild only when every corrected keyframe still has a stored depth image.
    #[default]
    Strict,
    /// Rebuild when enough corrected keyframes have depth coverage.
    BestEffort { min_coverage_percent: u8 },
}

#[derive(Debug)]
pub enum TsdfModeConfigError {
    Tsdf(crate::dense::backend::TsdfConfigError),
    InvalidCoveragePercent { value: u8 },
}

impl std::fmt::Display for TsdfModeConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tsdf(err) => write!(f, "invalid TSDF configuration: {err}"),
            Self::InvalidCoveragePercent { value } => write!(
                f,
                "best-effort rebuild coverage must be in 1..=100 percent, got {value}"
            ),
        }
    }
}

impl std::error::Error for TsdfModeConfigError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Tsdf(err) => Some(err),
            Self::InvalidCoveragePercent { .. } => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TsdfModeConfig {
    config: TsdfConfig,
    intrinsics: PinholeIntrinsics,
    rebuild_policy: RebuildPolicy,
}

impl TsdfModeConfig {
    pub fn try_new(
        config: TsdfConfig,
        intrinsics: PinholeIntrinsics,
        rebuild_policy: RebuildPolicy,
    ) -> Result<Self, TsdfModeConfigError> {
        config.validate().map_err(TsdfModeConfigError::Tsdf)?;
        if let RebuildPolicy::BestEffort {
            min_coverage_percent,
        } = rebuild_policy
            && !(1..=100).contains(&min_coverage_percent)
        {
            return Err(TsdfModeConfigError::InvalidCoveragePercent {
                value: min_coverage_percent,
            });
        }
        Ok(Self {
            config,
            intrinsics,
            rebuild_policy,
        })
    }

    pub fn config(&self) -> &TsdfConfig {
        &self.config
    }

    pub fn intrinsics(&self) -> PinholeIntrinsics {
        self.intrinsics
    }

    pub fn rebuild_policy(&self) -> RebuildPolicy {
        self.rebuild_policy
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
    map: HashMap<KeyframeId, DepthImage>,
    order: VecDeque<KeyframeId>,
    cap: NonZeroUsize,
}

impl DepthStore {
    pub fn new(cap: NonZeroUsize) -> Self {
        Self {
            map: HashMap::with_capacity(cap.get().min(64)),
            order: VecDeque::with_capacity(cap.get().min(64)),
            cap,
        }
    }

    pub fn insert(&mut self, keyframe_id: KeyframeId, depth: DepthImage) {
        if let std::collections::hash_map::Entry::Occupied(mut entry) = self.map.entry(keyframe_id)
        {
            // Update in place, don't change order.
            entry.insert(depth);
            return;
        }

        // Insertion-order eviction if at cap.
        while self.map.len() >= self.cap.get() {
            if let Some(oldest) = self.order.pop_front() {
                self.map.remove(&oldest);
            } else {
                break;
            }
        }

        self.map.insert(keyframe_id, depth);
        self.order.push_back(keyframe_id);
    }

    /// Remove a keyframe. No-op if the ID is unknown.
    pub fn remove(&mut self, keyframe_id: KeyframeId) {
        if self.map.remove(&keyframe_id).is_some() {
            self.order.retain(|id| *id != keyframe_id);
        }
    }

    pub fn clear(&mut self) {
        self.map.clear();
        self.order.clear();
    }

    pub fn get(&self, keyframe_id: KeyframeId) -> Option<&DepthImage> {
        self.map.get(&keyframe_id)
    }

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

    fn accepts_rebuild(&mut self, corrected_poses: &[(KeyframeId, Pose)]) -> bool {
        let Some((first, rest)) = corrected_poses.split_first() else {
            // Empty rebuilds carry no map-scoped key. Their generation is the
            // only ordering evidence, so the usual stale-generation check is
            // authoritative for them.
            return true;
        };
        let command_map = first.0.map_instance_id();
        if rest
            .iter()
            .any(|(keyframe_id, _)| keyframe_id.map_instance_id() != command_map)
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

fn rebuild_allowed(policy: RebuildPolicy, rebuildable: usize, total: usize) -> bool {
    if total == 0 {
        return true;
    }
    match policy {
        RebuildPolicy::Strict => rebuildable == total,
        RebuildPolicy::BestEffort {
            min_coverage_percent,
        } => {
            let threshold = usize::from(min_coverage_percent);
            rebuildable.saturating_mul(100) >= total.saturating_mul(threshold)
        }
    }
}

/// Process a single dense command. Returns updated stats.
///
/// This is a pure function (no thread, no I/O) so that tests can exercise
/// the full command processing logic without spawning threads.
pub fn process_dense_command(state: &mut DenseState, cmd: DenseCommand) -> DenseStats {
    if state.state == ReconState::Down && !matches!(&cmd, DenseCommand::ResetMappingSession { .. })
    {
        // Drain without processing.
        return state.stats();
    }

    match cmd {
        DenseCommand::IntegrateKeyframe {
            keyframe_id,
            pose,
            depth,
        } => {
            if !state.accepts_keyframe(keyframe_id) {
                eprintln!(
                    "dense: ignoring integration from inactive map {:?}",
                    keyframe_id.map_instance_id()
                );
                return state.stats();
            }
            state.store.insert(keyframe_id, depth.clone());
            let backend_failed = match (&state.mode, state.backend.as_mut()) {
                (DenseMode::DepthStoreOnly, _) => {
                    state.stats.integrated_count = state.stats.integrated_count.saturating_add(1);
                    false
                }
                (DenseMode::Tsdf(tsdf), Some(backend)) => {
                    match backend.integrate(pose, &depth, tsdf.intrinsics) {
                        Ok(()) => {
                            state.stats.integrated_count =
                                state.stats.integrated_count.saturating_add(1);
                            false
                        }
                        Err(e) => {
                            eprintln!("dense: tsdf integration error: {e}");
                            true
                        }
                    }
                }
                (DenseMode::Tsdf(_), None) => {
                    eprintln!("dense: tsdf integration requested without a backend");
                    true
                }
            };
            if backend_failed {
                state.backend = None;
                state.state = ReconState::Down;
            }
        }
        DenseCommand::RemoveKeyframe { keyframe_id } => {
            if state.map_instance_id != Some(keyframe_id.map_instance_id()) {
                eprintln!(
                    "dense: ignoring removal from inactive map {:?}",
                    keyframe_id.map_instance_id()
                );
                return state.stats();
            }
            state.store.remove(keyframe_id);
            state.stats.removed_count = state.stats.removed_count.saturating_add(1);
        }
        DenseCommand::ResetMappingSession {
            transition,
            generation,
        } => {
            if generation <= state.generation {
                return state.stats();
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
                return state.stats();
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

            match (&state.mode, state.backend.as_mut()) {
                (DenseMode::DepthStoreOnly, _) => {
                    state.state = ReconState::Nominal;
                }
                (DenseMode::Tsdf(_), Some(backend)) => {
                    if let Err(error) = backend.clear() {
                        eprintln!("dense: tsdf session-reset clear error: {error}");
                        state.backend = None;
                        state.state = ReconState::Down;
                    } else {
                        state.state = ReconState::Nominal;
                    }
                }
                (DenseMode::Tsdf(_), None) => {
                    // There is no reachable backend state to contaminate the
                    // new session, but reconstruction cannot truthfully be
                    // reported as available until a backend is attached.
                    state.state = ReconState::Down;
                }
            }
        }
        DenseCommand::RebuildFromSnapshot {
            corrected_poses,
            generation,
        } => {
            if generation <= state.generation {
                // Stale rebuild request — skip.
                return state.stats();
            }
            if !state.accepts_rebuild(&corrected_poses) {
                eprintln!("dense: ignoring rebuild whose keyframes are not all in the active map");
                return state.stats();
            }
            state.state = ReconState::Rebuilding { generation };

            // Count how many corrected keyframes still have depth snapshots.
            let mut rebuildable = 0usize;
            for (kf_id, _new_pose) in &corrected_poses {
                if state.store.contains(*kf_id) {
                    rebuildable = rebuildable.saturating_add(1);
                }
            }
            if rebuildable < corrected_poses.len() {
                eprintln!(
                    "dense rebuild missing depth snapshots for {} keyframes",
                    corrected_poses.len().saturating_sub(rebuildable)
                );
            }

            let mut backend_poisoned = false;
            let rebuild_succeeded = match (&state.mode, state.backend.as_mut()) {
                (DenseMode::DepthStoreOnly, _) => true,
                (DenseMode::Tsdf(tsdf), Some(backend)) => {
                    let total = corrected_poses.len();
                    if !rebuild_allowed(tsdf.rebuild_policy, rebuildable, total) {
                        eprintln!(
                            "dense: skipping tsdf rebuild due to policy {:?} (coverage={rebuildable}/{total})",
                            tsdf.rebuild_policy
                        );
                        false
                    } else if let Err(e) = backend.clear() {
                        eprintln!("dense: tsdf clear error: {e}");
                        backend_poisoned = true;
                        false
                    } else {
                        let mut succeeded = true;
                        for (kf_id, new_pose) in &corrected_poses {
                            if let Some(depth) = state.store.get(*kf_id)
                                && let Err(e) = backend.integrate(*new_pose, depth, tsdf.intrinsics)
                            {
                                eprintln!("dense: tsdf rebuild integration error: {e}");
                                backend_poisoned = true;
                                succeeded = false;
                                break;
                            }
                        }
                        succeeded
                    }
                }
                (DenseMode::Tsdf(_), None) => {
                    eprintln!("dense: tsdf rebuild requested without a backend");
                    backend_poisoned = true;
                    false
                }
            };

            if backend_poisoned {
                state.backend = None;
                state.state = ReconState::Down;
            } else if rebuild_succeeded {
                state.generation = generation;
                state.stats.rebuild_count = state.stats.rebuild_count.saturating_add(1);
                state.state = ReconState::Nominal;
            } else {
                state.state = ReconState::Degraded {
                    generation: state.generation,
                };
            }
        }
    }

    state.stats()
}

// ---------------------------------------------------------------------------
// Worker
// ---------------------------------------------------------------------------

fn process_command_with_recovery(
    state: &mut DenseState,
    cmd: DenseCommand,
    stats_tx: Option<&crate::DropSender<DenseStats>>,
) {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        process_dense_command(state, cmd)
    }));
    match result {
        Ok(stats) => {
            if let Some(tx) = stats_tx {
                tx.try_send(stats);
            }
        }
        Err(_) => {
            // The backend may have panicked after a partial mutation. Drop it
            // and stop processing instead of reusing uncertain state.
            state.backend = None;
            state.state = ReconState::Down;
            if let Some(tx) = stats_tx {
                tx.try_send(state.stats());
            }
        }
    }
}

/// Run the dense worker loop.
///
/// `command_rx` retains the producer's accepted command order. Integrations
/// and controls must never arrive through separate transports because reset,
/// rebuild, integration, and removal semantics are causally dependent.
/// `backend_factory` is required when `config.mode` is [`DenseMode::Tsdf`].
pub fn run_dense_worker(
    config: &DenseConfig,
    command_rx: &DenseCommandReceiver,
    backend_factory: Option<TsdfBackendFactory>,
    stats_tx: Option<&crate::DropSender<DenseStats>>,
) {
    let mut state = DenseState::new(config);
    match (&config.mode, backend_factory) {
        (DenseMode::DepthStoreOnly, _) => {}
        (DenseMode::Tsdf(tsdf), Some(factory)) => {
            let created = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                factory(tsdf.config.clone())
            }));
            match created {
                Ok(Ok(backend)) => state.set_backend(backend),
                Ok(Err(err)) => {
                    eprintln!("dense: failed to create TSDF backend: {err}");
                    state.state = ReconState::Down;
                }
                Err(_) => {
                    eprintln!("dense: TSDF backend factory panicked");
                    state.state = ReconState::Down;
                }
            }
        }
        (DenseMode::Tsdf(_), None) => {
            eprintln!("dense: TSDF mode requires a backend factory");
            state.state = ReconState::Down;
        }
    }
    if let Some(tx) = stats_tx {
        tx.try_send(state.stats());
    }

    while let Ok(command) = command_rx.recv() {
        process_command_with_recovery(&mut state, command, stats_tx);
    }
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

    fn make_tsdf_config(policy: RebuildPolicy) -> DenseConfig {
        let intrinsics = PinholeIntrinsics::try_from(&CameraIntrinsics {
            fx: 100.0,
            fy: 100.0,
            cx: 1.0,
            cy: 1.0,
            width: 2,
            height: 2,
        })
        .expect("test intrinsics");
        let mode = TsdfModeConfig::try_new(TsdfConfig::default(), intrinsics, policy)
            .expect("test TSDF config");
        DenseConfig::try_new(10, DenseMode::Tsdf(mode)).expect("test dense config")
    }

    fn test_intrinsics() -> PinholeIntrinsics {
        PinholeIntrinsics::try_from(&CameraIntrinsics {
            fx: 100.0,
            fy: 100.0,
            cx: 1.0,
            cy: 1.0,
            width: 2,
            height: 2,
        })
        .expect("test intrinsics")
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
                pose: Pose::identity(),
                depth: dummy_depth(),
            },
        );
        assert_eq!(stats.integrated_count, 1);
        assert_eq!(stats.stored_keyframes, 1);
    }

    #[test]
    fn failed_tsdf_integration_is_not_counted_as_success() {
        let config = make_tsdf_config(RebuildPolicy::Strict);
        let mut state = DenseState::new(&config);
        state.set_backend(Box::new(FaultBackend {
            fail_integration_at: Some(1),
            ..FaultBackend::healthy()
        }));

        let stats = process_dense_command(
            &mut state,
            DenseCommand::IntegrateKeyframe {
                keyframe_id: kf(0),
                pose: Pose::identity(),
                depth: dummy_depth(),
            },
        );

        assert_eq!(stats.integrated_count, 0);
        assert_eq!(stats.stored_keyframes, 1);
        assert_eq!(stats.state, ReconState::Down);
        assert!(!state.has_backend());
    }

    #[test]
    fn tsdf_state_is_not_nominal_until_backend_is_attached() {
        let config = make_tsdf_config(RebuildPolicy::Strict);
        let mut state = DenseState::new(&config);
        assert_eq!(state.state(), ReconState::AwaitingBackend);

        state.set_backend(Box::new(FaultBackend::healthy()));
        assert_eq!(state.state(), ReconState::Nominal);
    }

    #[test]
    fn tsdf_mode_rejects_invalid_best_effort_coverage() {
        for value in [0, 101] {
            let result = TsdfModeConfig::try_new(
                TsdfConfig::default(),
                test_intrinsics(),
                RebuildPolicy::BestEffort {
                    min_coverage_percent: value,
                },
            );
            assert!(matches!(
                result,
                Err(TsdfModeConfigError::InvalidCoveragePercent { value: actual })
                    if actual == value
            ));
        }
    }

    #[test]
    fn tsdf_worker_constructs_backend_on_worker_thread() {
        let config = make_tsdf_config(RebuildPolicy::Strict);
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
                pose: Pose::identity(),
                depth: dummy_depth(),
            }),
            DenseCommandSendOutcome::Enqueued
        );
        drop(command_tx);

        let worker = std::thread::spawn(move || {
            let factory: TsdfBackendFactory = Box::new(|_| Ok(Box::new(FaultBackend::healthy())));
            run_dense_worker(&config, &command_rx, Some(factory), Some(&stats_tx));
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
        let config = make_tsdf_config(RebuildPolicy::Strict);
        let (command_tx, command_rx, _) =
            dense_command_channel(channel_capacity(1), channel_capacity(1), Duration::ZERO)
                .expect("command channel");
        let (stats_tx, stats_rx, _) = crate::bounded_channel(
            crate::ChannelCapacity::try_from(1_usize).expect("stats capacity"),
            crate::DropPolicy::DropNewest,
        );
        drop(command_tx);

        run_dense_worker(&config, &command_rx, None, Some(&stats_tx));

        let stats = stats_rx.try_recv().expect("initial worker state");
        assert_eq!(stats.state, ReconState::Down);
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
                pose: Pose::identity(),
                depth: dummy_depth(),
            }),
            DenseCommandSendOutcome::Enqueued
        );
        assert_eq!(
            sender.route(DenseCommand::IntegrateKeyframe {
                keyframe_id: second,
                pose: Pose::identity(),
                depth: dummy_depth(),
            }),
            DenseCommandSendOutcome::IntegrationDroppedNewest
        );
        assert_eq!(
            sender.route(DenseCommand::RemoveKeyframe { keyframe_id: first }),
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
                pose: Pose::identity(),
                depth: dummy_depth(),
            }),
            DenseCommandSendOutcome::Enqueued,
            "dequeueing an integration must release its data slot"
        );
        assert!(matches!(
            receiver.try_recv(),
            Ok(DenseCommand::RemoveKeyframe { keyframe_id }) if keyframe_id == first
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
            sender.route(DenseCommand::RemoveKeyframe { keyframe_id: kf(0) }),
            DenseCommandSendOutcome::Enqueued
        );
        assert_eq!(
            sender.route(DenseCommand::RemoveKeyframe { keyframe_id: kf(1) }),
            DenseCommandSendOutcome::Enqueued,
            "controls may use idle data capacity while total backlog stays bounded"
        );
        assert_eq!(
            sender.route(DenseCommand::RemoveKeyframe { keyframe_id: kf(2) }),
            DenseCommandSendOutcome::ControlTimedOut
        );
        drop(receiver);
        assert_eq!(
            sender.route(DenseCommand::RemoveKeyframe { keyframe_id: kf(3) }),
            DenseCommandSendOutcome::Disconnected
        );
        assert_eq!(
            sender.route(DenseCommand::IntegrateKeyframe {
                keyframe_id: kf(4),
                pose: Pose::identity(),
                depth: dummy_depth(),
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

        run_dense_worker(&make_config(10), &receiver, None, Some(&stats_tx));

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
            },
            DenseCommand::IntegrateKeyframe {
                keyframe_id: new_keyframe,
                pose: Pose::identity(),
                depth: dummy_depth(),
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
            },
            DenseCommand::IntegrateKeyframe {
                keyframe_id: new_keyframe,
                pose: Pose::identity(),
                depth: dummy_depth(),
            },
            DenseCommand::RemoveKeyframe {
                keyframe_id: new_keyframe,
            },
        ]);

        assert_eq!(final_stats.integrated_count, 1);
        assert_eq!(final_stats.removed_count, 1);
        assert_eq!(final_stats.stored_keyframes, 0);
    }

    #[test]
    fn rejected_rebuild_does_not_advance_generation_or_success_count() {
        let config = make_tsdf_config(RebuildPolicy::Strict);
        let mut state = DenseState::new(&config);
        state.set_backend(Box::new(FaultBackend::healthy()));
        let present = kf(0);
        state.store.insert(present, dummy_depth());

        let stats = process_dense_command(
            &mut state,
            DenseCommand::RebuildFromSnapshot {
                corrected_poses: vec![(present, Pose::identity()), (kf(1), Pose::identity())],
                generation: 4,
            },
        );

        assert_eq!(state.generation, 0);
        assert_eq!(stats.rebuild_count, 0);
        assert_eq!(stats.state, ReconState::Degraded { generation: 0 });
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
            let config = make_tsdf_config(RebuildPolicy::Strict);
            let mut state = DenseState::new(&config);
            state.set_backend(Box::new(backend));
            let first = kf(0);
            let second = kf(1);
            state.store.insert(first, dummy_depth());
            state.store.insert(second, dummy_depth());

            let stats = process_dense_command(
                &mut state,
                DenseCommand::RebuildFromSnapshot {
                    corrected_poses: vec![(first, Pose::identity()), (second, Pose::identity())],
                    generation: 1,
                },
            );

            assert_eq!(state.generation, 0);
            assert_eq!(stats.rebuild_count, 0);
            assert_eq!(stats.state, ReconState::Down);
            assert!(!state.has_backend());
        }
    }

    #[test]
    fn backend_panic_never_reuses_half_mutated_state() {
        let config = make_tsdf_config(RebuildPolicy::Strict);
        let mut state = DenseState::new(&config);
        state.set_backend(Box::new(FaultBackend {
            panic_integration_at: Some(1),
            ..FaultBackend::healthy()
        }));

        process_command_with_recovery(
            &mut state,
            DenseCommand::IntegrateKeyframe {
                keyframe_id: kf(0),
                pose: Pose::identity(),
                depth: dummy_depth(),
            },
            None,
        );

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
                pose: Pose::identity(),
                depth: dummy_depth(),
            },
        );
        let stats =
            process_dense_command(&mut state, DenseCommand::RemoveKeyframe { keyframe_id: id });
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
                pose: Pose::identity(),
                depth: dummy_depth(),
            },
        );
        let reset_stats = process_dense_command(
            &mut state,
            DenseCommand::ResetMappingSession {
                transition: transition(old_map, new_map),
                generation: 1,
            },
        );

        assert_eq!(reset_stats.stored_keyframes, 0);
        assert_eq!(state.map_instance_id, Some(new_map));
        assert_eq!(state.generation, 1);

        process_dense_command(
            &mut state,
            DenseCommand::IntegrateKeyframe {
                keyframe_id: old_keyframe,
                pose: Pose::identity(),
                depth: dummy_depth(),
            },
        );
        let new_stats = process_dense_command(
            &mut state,
            DenseCommand::IntegrateKeyframe {
                keyframe_id: new_keyframe,
                pose: Pose::identity(),
                depth: dummy_depth(),
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
                pose: Pose::identity(),
                depth: dummy_depth(),
            },
        );

        let stats = process_dense_command(
            &mut state,
            DenseCommand::ResetMappingSession {
                transition: transition(inactive_map, proposed_map),
                generation: 1,
            },
        );

        assert_eq!(state.map_instance_id, Some(active_map));
        assert_eq!(state.generation, 0);
        assert_eq!(stats.stored_keyframes, 1);
        assert!(state.store.contains(active_keyframe));
    }

    #[test]
    fn failed_session_clear_drops_backend_but_commits_isolation_boundary() {
        let config = make_tsdf_config(RebuildPolicy::Strict);
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
                pose: Pose::identity(),
                depth: dummy_depth(),
            },
        );

        let failed = process_dense_command(
            &mut state,
            DenseCommand::ResetMappingSession {
                transition: transition(old_map, new_map),
                generation: 7,
            },
        );

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
                pose: Pose::identity(),
                depth: dummy_depth(),
            },
        );
        let recovered = process_dense_command(
            &mut state,
            DenseCommand::IntegrateKeyframe {
                keyframe_id: new_keyframe,
                pose: Pose::identity(),
                depth: dummy_depth(),
            },
        );
        assert_eq!(recovered.state, ReconState::Nominal);
        assert_eq!(recovered.stored_keyframes, 1);
        assert!(!state.store.contains(old_keyframe));
        assert!(state.store.contains(new_keyframe));
    }

    #[test]
    fn reset_clears_new_session_data_that_raced_ahead_of_the_barrier() {
        let config = make_tsdf_config(RebuildPolicy::Strict);
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
                pose: Pose::identity(),
                depth: dummy_depth(),
            },
        );
        assert_eq!(backend_entries.load(Ordering::SeqCst), 2);
        assert_eq!(state.store.len(), 1);

        let reset = process_dense_command(
            &mut state,
            DenseCommand::ResetMappingSession {
                transition: transition(old_map, new_map),
                generation: 1,
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
        let config = make_tsdf_config(RebuildPolicy::Strict);
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
                pose: Pose::identity(),
                depth: dummy_depth(),
            },
        );

        process_command_with_recovery(
            &mut state,
            DenseCommand::ResetMappingSession {
                transition: transition(old_map, new_map),
                generation: 9,
            },
            None,
        );

        assert_eq!(state.state(), ReconState::Down);
        assert_eq!(state.store.len(), 0);
        assert_eq!(state.map_instance_id, Some(new_map));
        assert_eq!(state.generation, 9);
        assert!(!state.has_backend());
    }

    #[test]
    fn reset_generation_orders_empty_rebuilds_and_map_ids_order_nonempty_rebuilds() {
        let config = make_config(10);
        let mut state = DenseState::new(&config);
        let (old_map, old_keyframe) = map_keyframe(1);
        let (new_map, new_keyframe) = map_keyframe(2);
        process_dense_command(
            &mut state,
            DenseCommand::IntegrateKeyframe {
                keyframe_id: old_keyframe,
                pose: Pose::identity(),
                depth: dummy_depth(),
            },
        );
        process_dense_command(
            &mut state,
            DenseCommand::ResetMappingSession {
                transition: transition(old_map, new_map),
                generation: 2,
            },
        );

        process_dense_command(
            &mut state,
            DenseCommand::RebuildFromSnapshot {
                corrected_poses: Vec::new(),
                generation: 1,
            },
        );
        process_dense_command(
            &mut state,
            DenseCommand::RebuildFromSnapshot {
                corrected_poses: vec![(old_keyframe, Pose::identity())],
                generation: 3,
            },
        );
        assert_eq!(state.generation, 2);
        assert_eq!(state.stats.rebuild_count, 0);

        process_dense_command(
            &mut state,
            DenseCommand::IntegrateKeyframe {
                keyframe_id: new_keyframe,
                pose: Pose::identity(),
                depth: dummy_depth(),
            },
        );
        let current = process_dense_command(
            &mut state,
            DenseCommand::RebuildFromSnapshot {
                corrected_poses: Vec::new(),
                generation: 3,
            },
        );
        assert_eq!(state.generation, 3);
        assert_eq!(current.rebuild_count, 1);
        assert!(state.store.contains(new_keyframe));
    }

    #[test]
    fn rebuild_increments_count() {
        let config = make_config(10);
        let mut state = DenseState::new(&config);
        let stats = process_dense_command(
            &mut state,
            DenseCommand::RebuildFromSnapshot {
                corrected_poses: vec![],
                generation: 1,
            },
        );
        assert_eq!(stats.rebuild_count, 1);
        assert_eq!(stats.state, ReconState::Nominal);
    }

    #[test]
    fn stale_rebuild_is_skipped() {
        let config = make_config(10);
        let mut state = DenseState::new(&config);
        process_dense_command(
            &mut state,
            DenseCommand::RebuildFromSnapshot {
                corrected_poses: vec![],
                generation: 5,
            },
        );
        let stats = process_dense_command(
            &mut state,
            DenseCommand::RebuildFromSnapshot {
                corrected_poses: vec![],
                generation: 3, // stale
            },
        );
        assert_eq!(stats.rebuild_count, 1, "stale rebuild should not increment");
    }

    #[test]
    fn rebuild_aborts_on_higher_generation() {
        let config = make_config(10);
        let mut state = DenseState::new(&config);
        process_dense_command(
            &mut state,
            DenseCommand::RebuildFromSnapshot {
                corrected_poses: vec![],
                generation: 1,
            },
        );
        let stats = process_dense_command(
            &mut state,
            DenseCommand::RebuildFromSnapshot {
                corrected_poses: vec![],
                generation: 2,
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
                pose: Pose::identity(),
                depth: dummy_depth(),
            },
        );
        assert_eq!(stats.integrated_count, 0, "Down state should not process");
        assert_eq!(stats.stored_keyframes, 0);
    }

    #[test]
    fn empty_rebuild_snapshot() {
        let config = make_config(10);
        let mut state = DenseState::new(&config);
        let stats = process_dense_command(
            &mut state,
            DenseCommand::RebuildFromSnapshot {
                corrected_poses: vec![],
                generation: 1,
            },
        );
        assert_eq!(stats.state, ReconState::Nominal);
        assert_eq!(stats.rebuild_count, 1);
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
