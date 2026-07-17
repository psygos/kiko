//! Ordered occupancy-command processing and deterministic snapshot scheduling.

use std::num::NonZeroUsize;

use super::occupancy::{
    OccupancyConfig, OccupancyError, OccupancyGridSnapshot, OccupancyMapper, OccupancyRemoveOutcome,
};
use super::{DenseCommand, DenseCommandReceiver, DenseStats, ReconState};
use crate::{DropSender, SendOutcome, Timestamp};

/// Number of successful keyframe integrations between regular snapshots.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OccupancySnapshotCadence(NonZeroUsize);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OccupancySnapshotCadenceError;

impl std::fmt::Display for OccupancySnapshotCadenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "occupancy snapshot cadence must be nonzero")
    }
}

impl std::error::Error for OccupancySnapshotCadenceError {}

impl OccupancySnapshotCadence {
    pub fn try_new(value: usize) -> Result<Self, OccupancySnapshotCadenceError> {
        NonZeroUsize::new(value)
            .map(Self)
            .ok_or(OccupancySnapshotCadenceError)
    }
    pub fn get(self) -> usize {
        self.0.get()
    }
}

impl Default for OccupancySnapshotCadence {
    fn default() -> Self {
        Self(NonZeroUsize::new(5).expect("default occupancy cadence is nonzero"))
    }
}

#[derive(Clone, Debug)]
pub struct OccupancyRuntimeConfig {
    mapper: OccupancyConfig,
    snapshot_cadence: OccupancySnapshotCadence,
}

impl OccupancyRuntimeConfig {
    pub fn new(mapper: OccupancyConfig, snapshot_cadence: OccupancySnapshotCadence) -> Self {
        Self {
            mapper,
            snapshot_cadence,
        }
    }

    pub fn mapper(&self) -> &OccupancyConfig {
        &self.mapper
    }

    pub fn snapshot_cadence(&self) -> OccupancySnapshotCadence {
        self.snapshot_cadence
    }
}

/// One map snapshot paired with the tracker capture time that caused it.
#[derive(Debug)]
pub struct TimedOccupancySnapshot {
    timestamp: Timestamp,
    snapshot: OccupancyGridSnapshot,
}

impl TimedOccupancySnapshot {
    pub fn timestamp(&self) -> Timestamp {
        self.timestamp
    }

    pub fn snapshot(&self) -> &OccupancyGridSnapshot {
        &self.snapshot
    }

    pub fn into_parts(self) -> (Timestamp, OccupancyGridSnapshot) {
        (self.timestamp, self.snapshot)
    }
}

#[derive(Debug)]
pub enum OccupancyRuntimeError {
    Mapping(OccupancyError),
    Snapshot(OccupancyError),
    MappingAndSnapshot {
        mapping: OccupancyError,
        snapshot: OccupancyError,
    },
}

impl std::fmt::Display for OccupancyRuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Mapping(error) => write!(f, "occupancy command failed: {error}"),
            Self::Snapshot(error) => write!(f, "occupancy snapshot failed: {error}"),
            Self::MappingAndSnapshot { mapping, snapshot } => write!(
                f,
                "occupancy command failed: {mapping}; an earlier occupancy snapshot also failed: {snapshot}"
            ),
        }
    }
}

impl std::error::Error for OccupancyRuntimeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Mapping(error) | Self::Snapshot(error) => Some(error),
            Self::MappingAndSnapshot { mapping, .. } => Some(mapping),
        }
    }
}

impl OccupancyRuntimeError {
    /// Preserve an earlier non-authoritative publication failure when an
    /// authoritative mapping failure later terminates command draining.
    pub fn with_deferred_snapshot(self, deferred_snapshot: &mut Option<OccupancyError>) -> Self {
        match (self, deferred_snapshot.take()) {
            (Self::Mapping(mapping), Some(snapshot)) => {
                Self::MappingAndSnapshot { mapping, snapshot }
            }
            (error, None) => error,
            (error, Some(snapshot)) => {
                *deferred_snapshot = Some(snapshot);
                error
            }
        }
    }
}

#[derive(Debug)]
pub struct OccupancyProcessOutcome {
    stats: DenseStats,
    snapshot: Option<TimedOccupancySnapshot>,
}

impl OccupancyProcessOutcome {
    pub fn stats(&self) -> &DenseStats {
        &self.stats
    }

    pub fn into_parts(self) -> (DenseStats, Option<TimedOccupancySnapshot>) {
        (self.stats, self.snapshot)
    }
}

/// Single-threaded owner of occupancy evidence and its publication schedule.
pub struct OccupancyRuntime {
    mapper: OccupancyMapper,
    cadence: OccupancySnapshotCadence,
    successful_integrations_since_snapshot: usize,
    dirty_timestamp: Option<Timestamp>,
    generation: u64,
    stats: DenseStats,
}

impl OccupancyRuntime {
    pub fn try_new(config: OccupancyRuntimeConfig) -> Result<Self, OccupancyError> {
        Ok(Self {
            mapper: OccupancyMapper::try_new(config.mapper)?,
            cadence: config.snapshot_cadence,
            successful_integrations_since_snapshot: 0,
            dirty_timestamp: None,
            generation: 0,
            stats: DenseStats::default(),
        })
    }

    pub fn stats(&self) -> DenseStats {
        self.stats.clone()
    }

    pub fn mapper(&self) -> &OccupancyMapper {
        &self.mapper
    }

    pub fn process(
        &mut self,
        command: DenseCommand,
        snapshots_enabled: bool,
    ) -> Result<OccupancyProcessOutcome, OccupancyRuntimeError> {
        let mut force_snapshot = false;
        let mut cadence_snapshot = false;
        let timestamp;

        let mapping_result: Result<(), OccupancyError> = match command {
            DenseCommand::IntegrateKeyframe {
                keyframe_id,
                pose,
                depth,
                timestamp: command_timestamp,
            } => {
                timestamp = command_timestamp;
                self.mapper.integrate(keyframe_id, pose, &depth).map(|_| {
                    self.stats.integrated_count = self.stats.integrated_count.saturating_add(1);
                    self.successful_integrations_since_snapshot = self
                        .successful_integrations_since_snapshot
                        .saturating_add(1);
                    cadence_snapshot =
                        self.successful_integrations_since_snapshot >= self.cadence.get();
                    self.dirty_timestamp = Some(command_timestamp);
                })
            }
            DenseCommand::RemoveKeyframe {
                keyframe_id,
                timestamp: command_timestamp,
            } => {
                timestamp = command_timestamp;
                self.mapper.remove(keyframe_id).map(|outcome| {
                    if matches!(outcome, OccupancyRemoveOutcome::Removed { .. }) {
                        self.stats.removed_count = self.stats.removed_count.saturating_add(1);
                        self.dirty_timestamp = Some(command_timestamp);
                        force_snapshot = true;
                    }
                })
            }
            DenseCommand::ResetMappingSession {
                transition,
                generation,
                timestamp: command_timestamp,
            } => {
                timestamp = command_timestamp;
                let unrelated_session = self.mapper.map_instance_id().is_some_and(|active| {
                    active != transition.old_map() && active != transition.new_map()
                });
                if generation <= self.generation || unrelated_session {
                    Ok(())
                } else {
                    self.mapper.reset_to_map(transition.new_map()).map(|_| {
                        self.generation = generation;
                        self.successful_integrations_since_snapshot = 0;
                        self.dirty_timestamp = Some(command_timestamp);
                        force_snapshot = true;
                    })
                }
            }
            DenseCommand::ApplyPoseUpdates {
                updates,
                generation,
                timestamp: command_timestamp,
            } => {
                timestamp = command_timestamp;
                if generation <= self.generation || updates.is_empty() {
                    Ok(())
                } else {
                    let mut poses = Vec::new();
                    if poses.try_reserve(updates.len()).is_err() {
                        Err(OccupancyError::AllocationFailed {
                            context: "runtime pose-update batch",
                            requested: updates.len(),
                        })
                    } else {
                        poses.extend(
                            updates
                                .into_iter()
                                .map(|update| (update.keyframe_id(), update.pose())),
                        );
                        self.mapper.update_poses(&poses).map(|outcome| {
                            self.generation = generation;
                            if outcome.updated_keyframes > 0 {
                                self.stats.rebuild_count =
                                    self.stats.rebuild_count.saturating_add(1);
                                self.dirty_timestamp = Some(command_timestamp);
                                force_snapshot = true;
                            }
                        })
                    }
                }
            }
        };

        if let Err(error) = mapping_result {
            self.stats.state = ReconState::Down;
            return Err(OccupancyRuntimeError::Mapping(error));
        }

        self.stats.stored_keyframes = self.mapper.retained_keyframes();
        let snapshot = if snapshots_enabled && (cadence_snapshot || force_snapshot) {
            let snapshot = self
                .mapper
                .snapshot()
                .map_err(OccupancyRuntimeError::Snapshot)?;
            self.successful_integrations_since_snapshot = 0;
            self.dirty_timestamp = None;
            Some(TimedOccupancySnapshot {
                timestamp,
                snapshot,
            })
        } else {
            None
        };

        Ok(OccupancyProcessOutcome {
            stats: self.stats(),
            snapshot,
        })
    }

    /// Emit the final unpublished map state, if any.
    pub fn finish(
        &mut self,
        snapshots_enabled: bool,
    ) -> Result<Option<TimedOccupancySnapshot>, OccupancyRuntimeError> {
        let Some(timestamp) = self.dirty_timestamp else {
            return Ok(None);
        };
        if !snapshots_enabled {
            return Ok(None);
        }
        let snapshot = self
            .mapper
            .snapshot()
            .map_err(OccupancyRuntimeError::Snapshot)?;
        self.successful_integrations_since_snapshot = 0;
        self.dirty_timestamp = None;
        Ok(Some(TimedOccupancySnapshot {
            timestamp,
            snapshot,
        }))
    }
}

/// Run occupancy mapping on the sole dense-command consumer.
///
/// Snapshot transport is optional and non-authoritative. Once its consumer
/// disconnects, snapshot creation is disabled while mapping continues. A
/// snapshot creation failure also disables further snapshots so authoritative
/// mapping can drain, but the first failure is returned at worker completion.
pub fn run_occupancy_worker(
    config: OccupancyRuntimeConfig,
    command_rx: &DenseCommandReceiver,
    stats_tx: Option<&DropSender<DenseStats>>,
    mut snapshot_tx: Option<DropSender<TimedOccupancySnapshot>>,
) -> Result<(), OccupancyRuntimeError> {
    let mut runtime = OccupancyRuntime::try_new(config).map_err(OccupancyRuntimeError::Mapping)?;
    let mut deferred_snapshot_error = None;
    if let Some(sender) = stats_tx {
        let _ = sender.try_send(runtime.stats());
    }

    while let Ok(command) = command_rx.recv() {
        match runtime.process(command, snapshot_tx.is_some()) {
            Ok(outcome) => {
                let (stats, snapshot) = outcome.into_parts();
                if let Some(sender) = stats_tx {
                    let _ = sender.try_send(stats);
                }
                if let (Some(sender), Some(snapshot)) = (snapshot_tx.as_ref(), snapshot)
                    && matches!(sender.try_send(snapshot), SendOutcome::Disconnected)
                {
                    snapshot_tx = None;
                }
            }
            Err(OccupancyRuntimeError::Snapshot(error)) => {
                eprintln!(
                    "occupancy: snapshot publication failed; mapping will drain before the failure is returned: {error}"
                );
                deferred_snapshot_error.get_or_insert(error);
                snapshot_tx = None;
                if let Some(sender) = stats_tx {
                    let _ = sender.try_send(runtime.stats());
                }
            }
            Err(error @ OccupancyRuntimeError::Mapping(_)) => {
                if let Some(sender) = stats_tx {
                    let _ = sender.try_send(runtime.stats());
                }
                return Err(error.with_deferred_snapshot(&mut deferred_snapshot_error));
            }
            Err(error @ OccupancyRuntimeError::MappingAndSnapshot { .. }) => return Err(error),
        }
    }

    if let Some(sender) = snapshot_tx.as_ref() {
        match runtime.finish(true) {
            Ok(Some(snapshot)) => {
                let _ = sender.try_send(snapshot);
            }
            Ok(None) => {}
            Err(OccupancyRuntimeError::Snapshot(error)) => {
                deferred_snapshot_error.get_or_insert(error);
            }
            Err(error @ OccupancyRuntimeError::Mapping(_)) => {
                return Err(error.with_deferred_snapshot(&mut deferred_snapshot_error));
            }
            Err(error @ OccupancyRuntimeError::MappingAndSnapshot { .. }) => return Err(error),
        }
    }
    deferred_snapshot_result(deferred_snapshot_error)
}

fn deferred_snapshot_result(
    deferred_snapshot_error: Option<OccupancyError>,
) -> Result<(), OccupancyRuntimeError> {
    deferred_snapshot_error.map_or(Ok(()), |error| Err(OccupancyRuntimeError::Snapshot(error)))
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::dense::{DenseCommandSendOutcome, dense_command_channel};
    use crate::map::{ImageSize, KeyframeId, SlamMap};
    use crate::{
        ChannelCapacity, DepthImage, DropPolicy, FrameDimensions, FrameId, KeyframePoseUpdate,
        Keypoint, MappingSessionTransition, PinholeIntrinsics, Timestamp, WorldToCamera,
        bounded_channel,
    };

    use super::super::occupancy::{
        DepthCameraModel, DepthRangeMeters, DepthToTrackingCamera, HeightRangeMeters,
        OccupancyEvidenceModel, OccupancyGridGeometry, WorldToOccupancy,
    };

    fn runtime_config(cadence: usize) -> OccupancyRuntimeConfig {
        let dimensions = FrameDimensions::try_new(1, 1).expect("test depth dimensions");
        let camera = DepthCameraModel::new(
            PinholeIntrinsics::try_new(100.0, 100.0, 0.0, 0.0).expect("test intrinsics"),
            dimensions,
            DepthToTrackingCamera::identity(),
        );
        let mapper = OccupancyConfig::try_new(
            OccupancyGridGeometry::try_new(1.0, [-2.0, 0.0], 6, 6, 36).expect("test geometry"),
            WorldToOccupancy::level_optical_world(1.0).expect("test occupancy frame"),
            camera,
            HeightRangeMeters::try_new(0.0, 2.0).expect("test height range"),
            DepthRangeMeters::try_new(0.1, 10.0).expect("test depth range"),
            1,
            OccupancyEvidenceModel::try_new(-1, 3, -1, 1).expect("test evidence"),
            8,
        )
        .expect("test occupancy config");
        OccupancyRuntimeConfig::new(
            mapper,
            OccupancySnapshotCadence::try_new(cadence).expect("test cadence"),
        )
    }

    fn mapped_keyframes(count: usize) -> Vec<KeyframeId> {
        let mut map = SlamMap::new();
        let image_size = ImageSize::try_new(1, 1).expect("test image size");
        (0..count)
            .map(|index| {
                map.add_keyframe(
                    FrameId::new(u64::try_from(index + 1).expect("test frame ID")),
                    Timestamp::from_nanos(i64::try_from(index + 1).expect("test timestamp")),
                    WorldToCamera::identity(),
                    image_size,
                    vec![Keypoint { x: 0.0, y: 0.0 }],
                )
                .expect("test keyframe")
            })
            .collect()
    }

    fn keyframe_from_new_map(frame_id: u64) -> KeyframeId {
        let mut map = SlamMap::new();
        map.add_keyframe(
            FrameId::new(frame_id),
            Timestamp::from_nanos(i64::try_from(frame_id).expect("test timestamp")),
            WorldToCamera::identity(),
            ImageSize::try_new(1, 1).expect("test image size"),
            vec![Keypoint { x: 0.0, y: 0.0 }],
        )
        .expect("test keyframe")
    }

    fn depth(frame_id: u64, timestamp_ns: i64, depth_m: f32) -> DepthImage {
        DepthImage::new(
            FrameId::new(frame_id),
            Timestamp::from_nanos(timestamp_ns),
            1,
            1,
            vec![depth_m],
        )
        .expect("test depth")
    }

    fn integrate(keyframe_id: KeyframeId, timestamp_ns: i64) -> DenseCommand {
        DenseCommand::IntegrateKeyframe {
            keyframe_id,
            pose: WorldToCamera::identity(),
            depth: depth(
                u64::try_from(timestamp_ns).expect("positive test timestamp"),
                timestamp_ns,
                2.0,
            ),
            timestamp: Timestamp::from_nanos(timestamp_ns),
        }
    }

    #[test]
    fn cadence_counts_only_successful_integrations_and_finish_emits_dirty_state_once() {
        let ids = mapped_keyframes(3);
        let mut runtime = OccupancyRuntime::try_new(runtime_config(2)).expect("runtime");

        let first = runtime.process(integrate(ids[0], 10), true).expect("first");
        assert!(first.snapshot.is_none());
        assert_eq!(first.stats.integrated_count, 1);

        let second = runtime
            .process(integrate(ids[1], 20), true)
            .expect("second");
        let second = second.snapshot.expect("cadence snapshot");
        assert_eq!(second.timestamp(), Timestamp::from_nanos(20));
        assert_eq!(second.snapshot().metadata().revision(), 2);

        let third = runtime.process(integrate(ids[2], 30), true).expect("third");
        assert!(third.snapshot.is_none());
        let final_snapshot = runtime
            .finish(true)
            .expect("finish")
            .expect("dirty snapshot");
        assert_eq!(final_snapshot.timestamp(), Timestamp::from_nanos(30));
        assert_eq!(final_snapshot.snapshot().metadata().revision(), 3);
        assert!(runtime.finish(true).expect("second finish").is_none());
    }

    #[test]
    fn final_snapshot_resets_cadence_if_the_runtime_is_reused() {
        let ids = mapped_keyframes(2);
        let mut runtime = OccupancyRuntime::try_new(runtime_config(2)).expect("runtime");

        assert!(
            runtime
                .process(integrate(ids[0], 10), true)
                .expect("first integration")
                .snapshot
                .is_none()
        );
        assert!(runtime.finish(true).expect("final snapshot").is_some());
        assert!(
            runtime
                .process(integrate(ids[1], 20), true)
                .expect("post-finish integration")
                .snapshot
                .is_none(),
            "one integration after a final snapshot must not inherit the previous cadence count"
        );
    }

    #[test]
    fn disabled_publication_does_not_consume_a_due_cadence() {
        let ids = mapped_keyframes(3);
        let mut runtime = OccupancyRuntime::try_new(runtime_config(2)).expect("runtime");

        assert!(
            runtime
                .process(integrate(ids[0], 10), false)
                .expect("first")
                .snapshot
                .is_none()
        );
        assert!(
            runtime
                .process(integrate(ids[1], 20), false)
                .expect("second")
                .snapshot
                .is_none()
        );
        let resumed = runtime
            .process(integrate(ids[2], 30), true)
            .expect("resumed publication")
            .snapshot
            .expect("overdue cadence snapshot");
        assert_eq!(resumed.timestamp(), Timestamp::from_nanos(30));
        assert_eq!(resumed.snapshot().metadata().revision(), 3);
    }

    #[test]
    fn removal_pose_update_and_reset_force_timestamped_snapshots() {
        let ids = mapped_keyframes(1);
        let mut runtime = OccupancyRuntime::try_new(runtime_config(8)).expect("runtime");
        runtime
            .process(integrate(ids[0], 10), false)
            .expect("integration");

        let pose_update = runtime
            .process(
                DenseCommand::ApplyPoseUpdates {
                    updates: vec![KeyframePoseUpdate::new(ids[0], WorldToCamera::identity())],
                    generation: 1,
                    timestamp: Timestamp::from_nanos(20),
                },
                true,
            )
            .expect("pose update");
        let pose_snapshot = pose_update.snapshot.expect("forced pose snapshot");
        assert_eq!(pose_snapshot.timestamp(), Timestamp::from_nanos(20));
        assert_eq!(pose_update.stats.rebuild_count, 1);

        let removal = runtime
            .process(
                DenseCommand::RemoveKeyframe {
                    keyframe_id: ids[0],
                    timestamp: Timestamp::from_nanos(30),
                },
                true,
            )
            .expect("removal");
        let removal_snapshot = removal.snapshot.expect("forced removal snapshot");
        assert_eq!(removal_snapshot.timestamp(), Timestamp::from_nanos(30));
        assert_eq!(removal.stats.removed_count, 1);
        assert_eq!(removal.stats.stored_keyframes, 0);

        let new_map_keyframe = keyframe_from_new_map(99);
        let transition = MappingSessionTransition::try_new(
            ids[0].map_instance_id(),
            new_map_keyframe.map_instance_id(),
        )
        .expect("test map transition");
        let reset = runtime
            .process(
                DenseCommand::ResetMappingSession {
                    transition,
                    generation: 2,
                    timestamp: Timestamp::from_nanos(40),
                },
                true,
            )
            .expect("reset");
        let reset_snapshot = reset.snapshot.expect("forced reset snapshot");
        assert_eq!(reset_snapshot.timestamp(), Timestamp::from_nanos(40));
        assert_eq!(
            reset_snapshot.snapshot().metadata().map_instance_id(),
            Some(new_map_keyframe.map_instance_id())
        );
    }

    #[test]
    fn capacity_one_snapshot_transport_preserves_latest_revision() {
        let ids = mapped_keyframes(3);
        let (command_tx, command_rx, _) = dense_command_channel(
            ChannelCapacity::try_from(3_usize).expect("data capacity"),
            ChannelCapacity::try_from(1_usize).expect("control capacity"),
            Duration::from_millis(1),
        )
        .expect("command channel");
        let (snapshot_tx, snapshot_rx, snapshot_stats) = bounded_channel(
            ChannelCapacity::try_from(1_usize).expect("snapshot capacity"),
            DropPolicy::DropOldest,
        );
        let worker = std::thread::spawn(move || {
            run_occupancy_worker(runtime_config(1), &command_rx, None, Some(snapshot_tx))
        });

        for (index, keyframe_id) in ids.into_iter().enumerate() {
            assert_eq!(
                command_tx.route(integrate(
                    keyframe_id,
                    i64::try_from(index + 1).expect("test timestamp"),
                )),
                DenseCommandSendOutcome::Enqueued
            );
        }
        drop(command_tx);
        worker
            .join()
            .expect("worker thread")
            .expect("worker result");

        let latest = snapshot_rx.try_recv().expect("latest snapshot");
        assert_eq!(latest.timestamp(), Timestamp::from_nanos(3));
        assert_eq!(latest.snapshot().metadata().revision(), 3);
        assert!(snapshot_rx.try_recv().is_err());
        assert_eq!(snapshot_stats.snapshot().enqueued, 3);
        assert_eq!(snapshot_stats.snapshot().dropped_oldest, 2);
    }

    #[test]
    fn final_dirty_snapshot_replaces_an_older_queued_revision() {
        let ids = mapped_keyframes(3);
        let (command_tx, command_rx, _) = dense_command_channel(
            ChannelCapacity::try_from(3_usize).expect("data capacity"),
            ChannelCapacity::try_from(1_usize).expect("control capacity"),
            Duration::from_millis(1),
        )
        .expect("command channel");
        let (snapshot_tx, snapshot_rx, snapshot_stats) = bounded_channel(
            ChannelCapacity::try_from(1_usize).expect("snapshot capacity"),
            DropPolicy::DropOldest,
        );
        let worker = std::thread::spawn(move || {
            run_occupancy_worker(runtime_config(2), &command_rx, None, Some(snapshot_tx))
        });

        for (index, keyframe_id) in ids.into_iter().enumerate() {
            assert_eq!(
                command_tx.route(integrate(
                    keyframe_id,
                    i64::try_from(index + 1).expect("test timestamp"),
                )),
                DenseCommandSendOutcome::Enqueued
            );
        }
        drop(command_tx);
        worker
            .join()
            .expect("worker thread")
            .expect("worker result");

        let latest = snapshot_rx.try_recv().expect("final snapshot");
        assert_eq!(latest.timestamp(), Timestamp::from_nanos(3));
        assert_eq!(latest.snapshot().metadata().revision(), 3);
        assert!(snapshot_rx.try_recv().is_err());
        assert_eq!(snapshot_stats.snapshot().enqueued, 2);
        assert_eq!(snapshot_stats.snapshot().dropped_oldest, 1);
    }

    #[test]
    fn disconnected_snapshot_consumer_does_not_stop_mapping() {
        let ids = mapped_keyframes(2);
        let (command_tx, command_rx, _) = dense_command_channel(
            ChannelCapacity::try_from(2_usize).expect("data capacity"),
            ChannelCapacity::try_from(1_usize).expect("control capacity"),
            Duration::from_millis(1),
        )
        .expect("command channel");
        let (snapshot_tx, snapshot_rx, snapshot_stats) = bounded_channel(
            ChannelCapacity::try_from(1_usize).expect("snapshot capacity"),
            DropPolicy::DropOldest,
        );
        drop(snapshot_rx);
        let worker = std::thread::spawn(move || {
            run_occupancy_worker(runtime_config(1), &command_rx, None, Some(snapshot_tx))
        });

        assert_eq!(
            command_tx.route(integrate(ids[0], 1)),
            DenseCommandSendOutcome::Enqueued
        );
        assert_eq!(
            command_tx.route(integrate(ids[1], 2)),
            DenseCommandSendOutcome::Enqueued
        );
        drop(command_tx);
        worker
            .join()
            .expect("worker thread")
            .expect("worker result");
        assert_eq!(snapshot_stats.snapshot().disconnected, 1);
    }

    #[test]
    fn deferred_snapshot_failure_is_returned_at_worker_completion() {
        let source = OccupancyError::AllocationFailed {
            context: "test snapshot",
            requested: 42,
        };

        assert!(matches!(
            deferred_snapshot_result(Some(source)),
            Err(OccupancyRuntimeError::Snapshot(error)) if error == source
        ));
        assert!(deferred_snapshot_result(None).is_ok());
    }

    #[test]
    fn later_mapping_failure_preserves_an_earlier_snapshot_failure() {
        let mapping = OccupancyError::RevisionExhausted;
        let snapshot = OccupancyError::AllocationFailed {
            context: "test snapshot",
            requested: 42,
        };
        let mut deferred = Some(snapshot);

        let combined =
            OccupancyRuntimeError::Mapping(mapping).with_deferred_snapshot(&mut deferred);

        assert!(matches!(
            combined,
            OccupancyRuntimeError::MappingAndSnapshot {
                mapping: actual_mapping,
                snapshot: actual_snapshot,
            } if actual_mapping == mapping && actual_snapshot == snapshot
        ));
        assert!(deferred.is_none());
    }
}
