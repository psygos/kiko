pub mod command_mapper;
pub mod ring_buffer;

use std::collections::{HashMap, VecDeque};

use crate::map::KeyframeId;
use crate::{DepthImage, Pose};

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
    Nominal,
    Rebuilding { generation: u64 },
    Down,
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

#[derive(Debug)]
pub struct DenseConfig {
    /// Maximum number of depth keyframes to store. LRU eviction kicks in
    /// when this limit is reached (safety net for exploration trajectories
    /// where the tracker never culls keyframes).
    pub max_stored_keyframes: usize,
}

impl Default for DenseConfig {
    fn default() -> Self {
        Self {
            max_stored_keyframes: 300,
        }
    }
}

// ---------------------------------------------------------------------------
// Depth store
// ---------------------------------------------------------------------------

/// Bounded store of depth images keyed by keyframe ID.
///
/// Primary eviction is via `RemoveKeyframe` commands (driven by
/// `DiagnosticEvent::KeyframeRemoved`). The LRU cap is a safety net
/// for unbounded growth in exploration scenarios.
#[derive(Debug)]
pub(crate) struct DepthStore {
    map: HashMap<KeyframeId, DepthImage>,
    order: VecDeque<KeyframeId>,
    cap: usize,
}

impl DepthStore {
    pub fn new(cap: usize) -> Self {
        let cap = cap.max(1);
        Self {
            map: HashMap::with_capacity(cap.min(64)),
            order: VecDeque::with_capacity(cap.min(64)),
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

        // LRU eviction if at cap.
        while self.map.len() >= self.cap {
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

    #[cfg(test)]
    pub fn get(&self, keyframe_id: KeyframeId) -> Option<&DepthImage> {
        self.map.get(&keyframe_id)
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
    consecutive_panics: u32,
}

impl DenseState {
    pub fn new(config: &DenseConfig) -> Self {
        Self {
            store: DepthStore::new(config.max_stored_keyframes),
            state: ReconState::Nominal,
            generation: 0,
            stats: DenseStats::default(),
            consecutive_panics: 0,
        }
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
            pose: _,
            depth,
        } => {
            state.store.insert(keyframe_id, depth);
            state.stats.integrated_count = state.stats.integrated_count.saturating_add(1);
            // Commit 2: call backend.integrate(pose, depth, intrinsics) here.
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
            state.generation = generation;
            state.state = ReconState::Rebuilding { generation };

            // Commit 2: backend.clear() here, then re-integrate all stored
            // keyframes with their corrected poses.
            //
            // For now, update stored poses in-place (no TSDF backend yet).
            for (kf_id, new_pose) in &corrected_poses {
                // The depth data stays the same; only the pose used for
                // integration changes. In commit 2 this drives
                // backend.integrate(new_pose, stored_depth, intrinsics).
                let _ = (kf_id, new_pose);
            }

            state.stats.rebuild_count = state.stats.rebuild_count.saturating_add(1);
            state.state = ReconState::Nominal;
        }
    }

    state.consecutive_panics = 0;
    state.stats()
}

// ---------------------------------------------------------------------------
// Worker
// ---------------------------------------------------------------------------

/// Maximum consecutive panics before the dense worker transitions to `Down`.
const MAX_CONSECUTIVE_PANICS: u32 = 3;
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
            state.consecutive_panics = 0;
            if let Some(tx) = stats_tx {
                tx.try_send(stats);
            }
        }
        Err(_) => {
            state.consecutive_panics = state.consecutive_panics.saturating_add(1);
            if state.consecutive_panics >= MAX_CONSECUTIVE_PANICS {
                state.state = ReconState::Down;
            }
        }
    }
}

/// Run the dense worker loop.
///
/// `ctrl_rx` is the unbounded control channel (RemoveKeyframe, RebuildFromSnapshot).
/// `data_rx` is the bounded data channel (IntegrateKeyframe, DropNewest).
///
/// The worker prioritises control commands: on each iteration it drains
/// the control channel before blocking on either channel.
pub fn run_dense_worker(
    config: &DenseConfig,
    ctrl_rx: &crossbeam_channel::Receiver<DenseCommand>,
    data_rx: &crossbeam_channel::Receiver<DenseCommand>,
    stats_tx: Option<&crate::DropSender<DenseStats>>,
) {
    let mut state = DenseState::new(config);

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
    use crate::test_helpers::make_depth_image;
    use crate::{FrameId, Timestamp};

    fn kf(n: u64) -> KeyframeId {
        // Use default + offset trick — slotmap KeyframeId is opaque.
        // For unit tests, we need distinct IDs. We can use a small slotmap.
        use slotmap::SlotMap;
        let mut sm = SlotMap::<KeyframeId, ()>::with_key();
        let mut id = sm.insert(());
        for _ in 1..=n {
            id = sm.insert(());
        }
        id
    }

    fn make_config(cap: usize) -> DenseConfig {
        DenseConfig {
            max_stored_keyframes: cap,
        }
    }

    fn dummy_depth() -> DepthImage {
        make_depth_image(FrameId::new(0), Timestamp::from_nanos(0), 2, 2, 1.0)
    }

    // -- DepthStore tests --

    #[test]
    fn depth_store_insert_and_retrieve() {
        let mut store = DepthStore::new(10);
        let id = kf(0);
        let depth = dummy_depth();
        store.insert(id, depth.clone());
        assert_eq!(store.len(), 1);
        assert!(store.get(id).is_some());
    }

    #[test]
    fn depth_store_cap_evicts_oldest() {
        let mut store = DepthStore::new(2);
        let id1 = kf(0);
        let id2 = kf(1);
        let id3 = kf(2);
        store.insert(id1, dummy_depth());
        store.insert(id2, dummy_depth());
        assert_eq!(store.len(), 2);
        store.insert(id3, dummy_depth());
        assert_eq!(store.len(), 2);
        assert!(store.get(id1).is_none(), "oldest should be evicted");
        assert!(store.get(id2).is_some());
        assert!(store.get(id3).is_some());
    }

    #[test]
    fn remove_keyframe_known_id() {
        let mut store = DepthStore::new(10);
        let id = kf(0);
        store.insert(id, dummy_depth());
        assert_eq!(store.len(), 1);
        store.remove(id);
        assert_eq!(store.len(), 0);
        assert!(store.get(id).is_none());
    }

    #[test]
    fn remove_keyframe_unknown_id_is_noop() {
        let mut store = DepthStore::new(10);
        let id_in = kf(0);
        let id_out = kf(1);
        store.insert(id_in, dummy_depth());
        store.remove(id_out); // unknown — no panic, no change
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn depth_store_cap_one() {
        let mut store = DepthStore::new(1);
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
