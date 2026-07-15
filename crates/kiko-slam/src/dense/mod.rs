pub mod backend;
pub mod command_mapper;
pub mod ring_buffer;

use std::collections::{HashMap, VecDeque};
use std::num::NonZeroUsize;

use crate::dense::backend::{TsdfBackend, TsdfBackendFactory, TsdfConfig};
use crate::map::KeyframeId;
use crate::{DepthImage, PinholeIntrinsics, Pose};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Commands sent from the pipeline to the dense reconstruction thread.
///
/// Routing: `IntegrateKeyframe` goes on the bounded data channel (DropNewest).
/// `RemoveKeyframe` and `RebuildFromSnapshot` go on the unbounded control
/// channel to guarantee delivery.
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
    RebuildFromSnapshot {
        corrected_poses: Vec<(KeyframeId, Pose)>,
        generation: u64,
    },
}

impl DenseCommand {
    /// Returns `true` for commands that must never be dropped.
    pub fn is_control(&self) -> bool {
        !matches!(self, DenseCommand::IntegrateKeyframe { .. })
    }
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
    if state.state == ReconState::Down {
        // Drain without processing.
        return state.stats();
    }

    match cmd {
        DenseCommand::IntegrateKeyframe {
            keyframe_id,
            pose,
            depth,
        } => {
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
            state.store.remove(keyframe_id);
            state.stats.removed_count = state.stats.removed_count.saturating_add(1);
        }
        DenseCommand::RebuildFromSnapshot {
            corrected_poses,
            generation,
        } => {
            if generation <= state.generation {
                // Stale rebuild request — skip.
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

/// Prevent control bursts from starving integration commands indefinitely.
const MAX_CONTROL_BURST: usize = 8;

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
/// `ctrl_rx` is the bounded control channel (RemoveKeyframe, RebuildFromSnapshot).
/// `data_rx` is the bounded data channel (IntegrateKeyframe, DropNewest).
/// `backend_factory` is required when `config.mode` is [`DenseMode::Tsdf`].
///
/// The worker prioritises control commands: on each iteration it drains
/// the control channel before blocking on either channel.
pub fn run_dense_worker(
    config: &DenseConfig,
    ctrl_rx: &crossbeam_channel::Receiver<DenseCommand>,
    data_rx: &crossbeam_channel::Receiver<DenseCommand>,
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

    loop {
        // Priority: drain all pending control commands.
        let mut drained = 0usize;
        while drained < MAX_CONTROL_BURST {
            match ctrl_rx.try_recv() {
                Ok(cmd) => {
                    process_command_with_recovery(&mut state, cmd, stats_tx);
                    drained = drained.saturating_add(1);
                }
                Err(crossbeam_channel::TryRecvError::Empty) => break,
                Err(crossbeam_channel::TryRecvError::Disconnected) => return,
            }
        }

        // Block on either channel.
        crossbeam_channel::select! {
            recv(ctrl_rx) -> msg => match msg {
                Ok(cmd) => process_command_with_recovery(&mut state, cmd, stats_tx),
                Err(_) => return,
            },
            recv(data_rx) -> msg => match msg {
                Ok(cmd) => process_command_with_recovery(&mut state, cmd, stats_tx),
                Err(_) => return,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::CameraIntrinsics;
    use crate::dense::backend::{Mesh, TsdfError};
    use crate::test_helpers::make_depth_image;
    use crate::{FrameId, Timestamp};

    struct FaultBackend {
        fail_clear: bool,
        fail_integration_at: Option<usize>,
        panic_integration_at: Option<usize>,
        integrations: usize,
    }

    impl FaultBackend {
        fn healthy() -> Self {
            Self {
                fail_clear: false,
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

    fn kf(n: u64) -> KeyframeId {
        KeyframeId::for_test(n as usize)
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
        let (ctrl_tx, ctrl_rx) = crossbeam_channel::bounded(1);
        let (data_tx, data_rx) = crossbeam_channel::bounded(1);
        let (stats_tx, stats_rx, _) = crate::bounded_channel(
            crate::ChannelCapacity::try_from(4_usize).expect("stats capacity"),
            crate::DropPolicy::DropNewest,
        );
        data_tx
            .send(DenseCommand::IntegrateKeyframe {
                keyframe_id: kf(0),
                pose: Pose::identity(),
                depth: dummy_depth(),
            })
            .expect("queue integration");
        drop(data_tx);

        let worker = std::thread::spawn(move || {
            let factory: TsdfBackendFactory = Box::new(|_| Ok(Box::new(FaultBackend::healthy())));
            run_dense_worker(&config, &ctrl_rx, &data_rx, Some(factory), Some(&stats_tx));
        });
        let initial = stats_rx
            .as_receiver()
            .recv_timeout(std::time::Duration::from_secs(1))
            .expect("initial worker state");
        let integrated = stats_rx
            .as_receiver()
            .recv_timeout(std::time::Duration::from_secs(1))
            .expect("integration state");
        drop(ctrl_tx);
        worker.join().expect("dense worker");

        assert_eq!(initial.state, ReconState::Nominal);
        assert_eq!(integrated.state, ReconState::Nominal);
        assert_eq!(integrated.integrated_count, 1);
    }

    #[test]
    fn tsdf_worker_without_factory_reports_down() {
        let config = make_tsdf_config(RebuildPolicy::Strict);
        let (ctrl_tx, ctrl_rx) = crossbeam_channel::bounded(1);
        let (data_tx, data_rx) = crossbeam_channel::bounded(1);
        let (stats_tx, stats_rx, _) = crate::bounded_channel(
            crate::ChannelCapacity::try_from(1_usize).expect("stats capacity"),
            crate::DropPolicy::DropNewest,
        );
        drop((ctrl_tx, data_tx));

        run_dense_worker(&config, &ctrl_rx, &data_rx, None, Some(&stats_tx));

        let stats = stats_rx.try_recv().expect("initial worker state");
        assert_eq!(stats.state, ReconState::Down);
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
    fn dense_stats_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<DenseStats>();
    }
}
