use std::collections::HashSet;

use crate::dense::DenseCommand;
use crate::dense::ring_buffer::{DepthAssociationWindow, DepthRingBuffer};
use crate::map::KeyframeId;
use crate::{
    DiagnosticEvent, KeyframePoseUpdate, PoseStatus, Timestamp, TrackerOutput, TrackingHealth,
    WorldToCamera,
};

/// Operational depth-association policy: the nearest frame at an inclusive
/// timestamp distance of at most 20 ms is eligible.
///
/// This is a scheduling policy, not a motion-error guarantee. A defensible
/// geometric bound additionally depends on measured camera motion, scene
/// depth, and map resolution, which are not inputs to this mapper.
pub const DEPTH_ASSOCIATION_WINDOW: DepthAssociationWindow =
    DepthAssociationWindow::from_nanos(20_000_000);

/// Producer-side sequence for dense reset and pose-update control commands.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DenseCommandGeneration(u64);

impl DenseCommandGeneration {
    pub const fn from_current(current: u64) -> Self {
        Self(current)
    }

    pub const fn current(self) -> u64 {
        self.0
    }

    fn checked_advance_for(
        self,
        reset_generations: usize,
        pose_update_generation: bool,
    ) -> Result<Self, DenseCommandGenerationError> {
        let error = DenseCommandGenerationError {
            current: self.0,
            reset_generations,
            pose_update_generation,
        };
        let required = u64::try_from(reset_generations)
            .ok()
            .and_then(|resets| resets.checked_add(u64::from(pose_update_generation)))
            .ok_or(error)?;
        self.0.checked_add(required).map(Self).ok_or(error)
    }

    fn checked_next(&mut self) -> Option<u64> {
        let next = self.0.checked_add(1)?;
        self.0 = next;
        Some(next)
    }
}

/// A dense control-command batch cannot fit in the remaining generation domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DenseCommandGenerationError {
    current: u64,
    reset_generations: usize,
    pose_update_generation: bool,
}

impl DenseCommandGenerationError {
    pub const fn current(self) -> u64 {
        self.current
    }

    pub const fn reset_generations(self) -> usize {
        self.reset_generations
    }

    pub const fn includes_pose_updates(self) -> bool {
        self.pose_update_generation
    }

    pub fn required(self) -> Option<u64> {
        u64::try_from(self.reset_generations)
            .ok()?
            .checked_add(u64::from(self.pose_update_generation))
    }
}

impl std::fmt::Display for DenseCommandGenerationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "dense control-command generation {} cannot reserve {} reset generation(s) and {} pose-update generation(s)",
            self.current,
            self.reset_generations,
            u8::from(self.pose_update_generation)
        )
    }
}

impl std::error::Error for DenseCommandGenerationError {}

/// A tracker output cannot be mapped into one transactional dense-command batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DenseCommandMappingError {
    Generation(DenseCommandGenerationError),
    MissingAuthoritativePose { keyframe_id: KeyframeId },
}

impl std::fmt::Display for DenseCommandMappingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Generation(source) => source.fmt(f),
            Self::MissingAuthoritativePose { keyframe_id } => write!(
                f,
                "cannot integrate keyframe {keyframe_id:?}: authoritative stored pose is missing"
            ),
        }
    }
}

impl std::error::Error for DenseCommandMappingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Generation(source) => Some(source),
            Self::MissingAuthoritativePose { .. } => None,
        }
    }
}

impl From<DenseCommandGenerationError> for DenseCommandMappingError {
    fn from(source: DenseCommandGenerationError) -> Self {
        Self::Generation(source)
    }
}

/// Map one tracker output and its committed keyframe-pose updates into dense
/// commands.
///
/// Rules:
/// - `IntegrateKeyframe` is emitted for each `KeyframeCreated` event that
///   has an associated depth frame and whose keyframe was NOT also removed
///   in the same frame. Its pose comes exclusively from `keyframe_pose`, the
///   authoritative stored-keyframe lookup; [`TrackerOutput::pose`] is never
///   used as a keyframe pose. Once all other integration conditions hold, a
///   missing authoritative pose is an invariant error for the whole batch.
/// - `RemoveKeyframe` is emitted for each `KeyframeRemoved` event.
/// - `ResetMappingSession` is emitted before every other command when the
///   output contains a mapping-session boundary.
/// - One `ApplyPoseUpdates` command is emitted when `pose_updates` is nonempty.
/// - No `IntegrateKeyframe` commands are emitted when tracking is `Lost`.
/// - Every emitted command carries the supplied tracker-output `timestamp`.
///
/// On any mapping error, no commands are returned and `generation` remains
/// unchanged.
pub fn map_output_to_dense_commands(
    output: &TrackerOutput,
    pose_updates: Vec<KeyframePoseUpdate>,
    mut keyframe_pose: impl FnMut(KeyframeId) -> Option<WorldToCamera>,
    depth_buffer: &DepthRingBuffer,
    timestamp: Timestamp,
    generation: &mut DenseCommandGeneration,
) -> Result<Vec<DenseCommand>, DenseCommandMappingError> {
    let reset_count = output
        .events
        .iter()
        .filter(|event| matches!(event, DiagnosticEvent::MappingSessionReset { .. }))
        .count();
    let pose_update_generation = !pose_updates.is_empty();
    let final_generation = generation
        .checked_advance_for(reset_count, pose_update_generation)
        .map_err(DenseCommandMappingError::Generation)?;
    let batch_error = DenseCommandGenerationError {
        current: generation.current(),
        reset_generations: reset_count,
        pose_update_generation,
    };
    let mut staged_generation = *generation;
    let mut commands = Vec::new();

    // A reset must be sent on the control path before any data command from
    // the new session. Advancing the generation makes every reset a sequencing
    // barrier for already queued pose-update requests.
    let mut output_map = None;
    for event in &output.events {
        if let DiagnosticEvent::MappingSessionReset { transition } = event {
            let reset_generation = staged_generation
                .checked_next()
                .ok_or(DenseCommandMappingError::Generation(batch_error))?;
            commands.push(DenseCommand::ResetMappingSession {
                transition: *transition,
                generation: reset_generation,
                timestamp,
            });
            output_map = Some(transition.new_map());
        }
    }

    // Collect KeyframeIds that were removed this frame so we can suppress
    // IntegrateKeyframe for the same id (coalesce create+remove).
    let removed_this_frame: HashSet<KeyframeId> = output
        .events
        .iter()
        .filter_map(|e| match e {
            DiagnosticEvent::KeyframeRemoved { keyframe_id, .. } => Some(*keyframe_id),
            _ => None,
        })
        .collect();

    // Gate integration on tracking health — don't integrate when lost.
    let tracking_ok =
        output.health.tracking != TrackingHealth::Lost && output.pose_status == PoseStatus::Current;
    let mut associated_depth = None;

    for event in &output.events {
        match event {
            DiagnosticEvent::KeyframeCreated { keyframe_id, .. } => {
                if output_map.is_some_and(|map| map != keyframe_id.map_instance_id()) {
                    continue;
                }
                if !tracking_ok {
                    continue;
                }
                if removed_this_frame.contains(keyframe_id) {
                    continue;
                }
                let depth = associated_depth
                    .get_or_insert_with(|| {
                        depth_buffer.find_closest(timestamp, DEPTH_ASSOCIATION_WINDOW)
                    })
                    .clone();
                let Some(depth) = depth else {
                    continue;
                };
                let Some(pose) = keyframe_pose(*keyframe_id) else {
                    return Err(DenseCommandMappingError::MissingAuthoritativePose {
                        keyframe_id: *keyframe_id,
                    });
                };
                commands.push(DenseCommand::IntegrateKeyframe {
                    keyframe_id: *keyframe_id,
                    pose,
                    depth,
                    timestamp,
                });
            }
            DiagnosticEvent::KeyframeRemoved { keyframe_id, .. } => {
                if output_map.is_some_and(|map| map != keyframe_id.map_instance_id()) {
                    continue;
                }
                commands.push(DenseCommand::RemoveKeyframe {
                    keyframe_id: *keyframe_id,
                    timestamp,
                });
            }
            _ => {}
        }
    }

    if !pose_updates.is_empty() {
        let pose_update_generation = staged_generation
            .checked_next()
            .ok_or(DenseCommandMappingError::Generation(batch_error))?;
        commands.push(DenseCommand::ApplyPoseUpdates {
            updates: pose_updates,
            generation: pose_update_generation,
            timestamp,
        });
    }

    debug_assert_eq!(staged_generation, final_generation);
    *generation = final_generation;
    Ok(commands)
}

/// Build one owned pose-update command and advance its generation only when
/// at least one update exists and the generation can advance. The supplied
/// timestamp is forwarded unchanged to the command.
pub fn apply_pose_updates_command(
    updates: Vec<KeyframePoseUpdate>,
    timestamp: Timestamp,
    generation: &mut DenseCommandGeneration,
) -> Result<Option<DenseCommand>, DenseCommandGenerationError> {
    if updates.is_empty() {
        return Ok(None);
    }
    let final_generation = generation.checked_advance_for(0, true)?;
    let command = DenseCommand::ApplyPoseUpdates {
        updates,
        generation: final_generation.current(),
        timestamp,
    };
    *generation = final_generation;
    Ok(Some(command))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense::ring_buffer::DepthRingBuffer;
    use crate::diagnostics::KeyframeRemovalReason;
    use crate::map::SlamMap;
    use crate::test_helpers::make_depth_image;
    use crate::tracker::BackendStats;
    use crate::{
        ComponentHealth, DegradationLevel, FrameDiagnostics, FrameId, Pose, PoseStatus,
        SystemHealth, TrackerOutput,
    };

    fn ts(ns: i64) -> Timestamp {
        Timestamp::from_nanos(ns)
    }

    fn depth_at(t_ns: i64) -> crate::DepthImage {
        make_depth_image(FrameId::new(0), ts(t_ns), 2, 2, 1.0)
    }

    fn depth_buffer() -> DepthRingBuffer {
        DepthRingBuffer::try_new(4).expect("nonzero test capacity")
    }

    fn kf_id() -> KeyframeId {
        KeyframeId::for_test(0)
    }

    fn translated_pose(x: f32) -> WorldToCamera {
        WorldToCamera::from_legacy_pose(
            Pose::try_from_rt(Pose::identity().rotation(), [x, 0.0, 0.0])
                .expect("finite translated pose"),
        )
    }

    fn pose_update(keyframe_id: KeyframeId, x: f32) -> KeyframePoseUpdate {
        KeyframePoseUpdate::new(keyframe_id, translated_pose(x))
    }

    fn healthy() -> SystemHealth {
        SystemHealth {
            tracking: TrackingHealth::Good,
            backend: ComponentHealth::Alive,
            descriptor: ComponentHealth::Alive,
            backend_stats: BackendStats::default(),
            degradation: DegradationLevel::Nominal,
        }
    }

    fn lost_health() -> SystemHealth {
        SystemHealth {
            tracking: TrackingHealth::Lost,
            backend: ComponentHealth::Alive,
            descriptor: ComponentHealth::Alive,
            backend_stats: BackendStats::default(),
            degradation: DegradationLevel::Lost,
        }
    }

    fn base_output(events: Vec<DiagnosticEvent>) -> TrackerOutput {
        TrackerOutput {
            pose: Some(crate::WorldToCamera::identity()),
            pose_status: PoseStatus::Current,
            inliers: 0,
            keyframe: None,
            stereo_matches: None,
            frame_id: FrameId::new(1),
            health: healthy(),
            diagnostics: FrameDiagnostics::empty(1, 1),
            events,
        }
    }

    fn map_commands(
        output: &TrackerOutput,
        pose_updates: Vec<KeyframePoseUpdate>,
        depth_buffer: &DepthRingBuffer,
        timestamp: Timestamp,
        generation: &mut DenseCommandGeneration,
    ) -> Vec<DenseCommand> {
        map_output_to_dense_commands(
            output,
            pose_updates,
            |_| Some(WorldToCamera::identity()),
            depth_buffer,
            timestamp,
            generation,
        )
        .expect("dense command mapping")
    }

    #[test]
    fn keyframe_created_with_depth_emits_integrate() {
        let kf = kf_id();
        let mut buf = depth_buffer();
        buf.push(depth_at(100));

        let output = base_output(vec![DiagnosticEvent::KeyframeCreated {
            keyframe_id: kf,
            landmarks: 10,
        }]);

        let mut gen_ = DenseCommandGeneration::default();
        let cmds = map_commands(&output, Vec::new(), &buf, ts(100), &mut gen_);
        assert!(matches!(
            cmds.as_slice(),
            [DenseCommand::IntegrateKeyframe {
                keyframe_id,
                pose,
                timestamp,
                ..
            }] if *keyframe_id == kf
                && pose.translation() == [0.0, 0.0, 0.0]
                && *timestamp == ts(100)
        ));
    }

    #[test]
    fn keyframe_created_uses_authoritative_pose_not_frame_pose() {
        let kf = kf_id();
        let mut buf = depth_buffer();
        buf.push(depth_at(100));
        let mut output = base_output(vec![DiagnosticEvent::KeyframeCreated {
            keyframe_id: kf,
            landmarks: 10,
        }]);
        output.pose = Some(translated_pose(99.0));
        let authoritative_pose = translated_pose(4.0);
        let mut generation = DenseCommandGeneration::default();

        let commands = map_output_to_dense_commands(
            &output,
            Vec::new(),
            |keyframe_id| (keyframe_id == kf).then_some(authoritative_pose),
            &buf,
            ts(100),
            &mut generation,
        )
        .expect("authoritative command mapping");

        let [DenseCommand::IntegrateKeyframe { pose, .. }] = commands.as_slice() else {
            panic!("one integration command expected");
        };
        assert_eq!(pose.translation(), [4.0, 0.0, 0.0]);
        assert_ne!(
            pose.translation(),
            output.pose.expect("frame pose").translation()
        );
    }

    #[test]
    fn missing_authoritative_pose_is_typed_and_mapping_is_transactional() {
        let kf = kf_id();
        let mut buf = depth_buffer();
        buf.push(depth_at(100));
        let output = base_output(vec![DiagnosticEvent::KeyframeCreated {
            keyframe_id: kf,
            landmarks: 10,
        }]);
        let mut lookups = 0;
        let mut generation = DenseCommandGeneration::from_current(7);

        let error = map_output_to_dense_commands(
            &output,
            vec![pose_update(kf, 1.0)],
            |_| {
                lookups += 1;
                None
            },
            &buf,
            ts(100),
            &mut generation,
        )
        .expect_err("missing authoritative pose must fail the whole batch");

        assert!(matches!(
            error,
            DenseCommandMappingError::MissingAuthoritativePose { keyframe_id }
                if keyframe_id == kf
        ));
        assert_eq!(lookups, 1);
        assert_eq!(
            generation.current(),
            7,
            "reserved pose-update generation must not commit on mapping failure"
        );
    }

    #[test]
    fn keyframe_removed_emits_remove() {
        let kf = kf_id();
        let buf = depth_buffer();

        let output = base_output(vec![DiagnosticEvent::KeyframeRemoved {
            keyframe_id: kf,
            reason: KeyframeRemovalReason::Redundant,
        }]);

        let mut gen_ = DenseCommandGeneration::default();
        let cmds = map_commands(&output, Vec::new(), &buf, ts(100), &mut gen_);
        assert!(matches!(
            cmds.as_slice(),
            [DenseCommand::RemoveKeyframe {
                keyframe_id,
                timestamp,
            }] if *keyframe_id == kf && *timestamp == ts(100)
        ));
    }

    #[test]
    fn pose_updates_emit_one_typed_command() {
        let buf = depth_buffer();
        let output = base_output(vec![]);
        let kf = kf_id();
        let updates = vec![pose_update(kf, 2.0)];
        let mut gen_ = DenseCommandGeneration::default();

        let cmds = map_commands(&output, updates, &buf, ts(100), &mut gen_);

        let [
            DenseCommand::ApplyPoseUpdates {
                updates,
                generation,
                timestamp,
            },
        ] = cmds.as_slice()
        else {
            panic!("one pose-update command expected");
        };
        assert_eq!(*generation, 1);
        assert_eq!(*timestamp, ts(100));
        assert_eq!(updates.len(), 1);
        assert_eq!(updates[0].keyframe_id(), kf);
        assert_eq!(updates[0].pose().translation(), [2.0, 0.0, 0.0]);
        assert_eq!(gen_.current(), 1);
    }

    #[test]
    fn partial_pose_update_batch_is_forwarded_without_synthesis() {
        let output = base_output(Vec::new());
        let updated_keyframe = KeyframeId::for_test(1);
        let mut generation = DenseCommandGeneration::default();

        let commands = map_output_to_dense_commands(
            &output,
            vec![pose_update(updated_keyframe, 5.0)],
            |_| panic!("pose updates do not require an authoritative lookup"),
            &depth_buffer(),
            ts(123),
            &mut generation,
        )
        .expect("partial update batch");

        let [DenseCommand::ApplyPoseUpdates { updates, .. }] = commands.as_slice() else {
            panic!("one pose-update command expected");
        };
        assert_eq!(updates.len(), 1);
        assert_eq!(updates[0].keyframe_id(), updated_keyframe);
        assert_eq!(updates[0].pose().translation(), [5.0, 0.0, 0.0]);
    }

    #[test]
    fn no_command_when_no_depth_available() {
        let kf = kf_id();
        let buf = depth_buffer(); // empty

        let output = base_output(vec![DiagnosticEvent::KeyframeCreated {
            keyframe_id: kf,
            landmarks: 5,
        }]);

        let mut gen_ = DenseCommandGeneration::default();
        let cmds = map_commands(&output, Vec::new(), &buf, ts(100), &mut gen_);
        assert!(cmds.is_empty(), "no depth available → no integrate command");
    }

    #[test]
    fn no_integrate_when_tracking_lost() {
        let kf = kf_id();
        let mut buf = depth_buffer();
        buf.push(depth_at(100));

        let mut output = base_output(vec![DiagnosticEvent::KeyframeCreated {
            keyframe_id: kf,
            landmarks: 5,
        }]);
        output.health = lost_health();

        let mut gen_ = DenseCommandGeneration::default();
        let cmds = map_commands(&output, Vec::new(), &buf, ts(100), &mut gen_);
        assert!(
            cmds.is_empty(),
            "should not integrate when tracking is lost"
        );
    }

    #[test]
    fn no_integrate_with_stale_fallback_pose() {
        let kf = kf_id();
        let mut buf = depth_buffer();
        buf.push(depth_at(100));
        let mut output = base_output(vec![DiagnosticEvent::KeyframeCreated {
            keyframe_id: kf,
            landmarks: 5,
        }]);
        output.pose_status = PoseStatus::Stale;

        let mut gen_ = DenseCommandGeneration::default();
        let cmds = map_commands(&output, Vec::new(), &buf, ts(100), &mut gen_);

        assert!(cmds.is_empty(), "stale poses must not be integrated");
    }

    #[test]
    fn coalesce_create_and_remove_same_frame() {
        let kf = kf_id();
        let mut buf = depth_buffer();
        buf.push(depth_at(100));

        let output = base_output(vec![
            DiagnosticEvent::KeyframeCreated {
                keyframe_id: kf,
                landmarks: 5,
            },
            DiagnosticEvent::KeyframeRemoved {
                keyframe_id: kf,
                reason: KeyframeRemovalReason::Redundant,
            },
        ]);

        let mut gen_ = DenseCommandGeneration::default();
        let cmds = map_commands(&output, Vec::new(), &buf, ts(100), &mut gen_);
        // Should only have RemoveKeyframe, not IntegrateKeyframe.
        assert_eq!(cmds.len(), 1);
        assert!(matches!(cmds[0], DenseCommand::RemoveKeyframe { .. }));
    }

    #[test]
    fn depth_outside_window_not_associated() {
        let kf = kf_id();
        let mut buf = depth_buffer();
        buf.push(depth_at(0)); // very old

        let output = base_output(vec![DiagnosticEvent::KeyframeCreated {
            keyframe_id: kf,
            landmarks: 5,
        }]);

        let mut gen_ = DenseCommandGeneration::default();
        let cmds = map_commands(
            &output,
            Vec::new(),
            &buf,
            ts(100_000_000), // 100ms after depth
            &mut gen_,
        );
        assert!(cmds.is_empty(), "depth too old for association");
    }

    #[test]
    fn generation_increments_on_successive_pose_update_batches() {
        let buf = depth_buffer();
        let output = base_output(vec![]);
        let mut gen_ = DenseCommandGeneration::default();

        map_commands(
            &output,
            vec![pose_update(kf_id(), 1.0)],
            &buf,
            ts(100),
            &mut gen_,
        );
        assert_eq!(gen_.current(), 1);

        map_commands(
            &output,
            vec![pose_update(kf_id(), 2.0)],
            &buf,
            ts(200),
            &mut gen_,
        );
        assert_eq!(gen_.current(), 2);
    }

    #[test]
    fn mapping_session_reset_is_first_and_advances_pose_update_order() {
        let old_map = SlamMap::new().snapshot().instance_id();
        let new_map = SlamMap::new().snapshot().instance_id();
        let transition =
            crate::MappingSessionTransition::try_new(old_map, new_map).expect("distinct test maps");
        let output = base_output(vec![DiagnosticEvent::MappingSessionReset { transition }]);
        let mut generation = DenseCommandGeneration::default();

        let commands = map_commands(
            &output,
            vec![pose_update(kf_id(), 1.0)],
            &depth_buffer(),
            ts(100),
            &mut generation,
        );

        assert_eq!(commands.len(), 2);
        assert!(matches!(
            commands[0],
            DenseCommand::ResetMappingSession {
                transition: actual,
                generation: 1,
                timestamp,
            } if actual == transition
                && timestamp == ts(100)
        ));
        assert!(matches!(
            commands[1],
            DenseCommand::ApplyPoseUpdates {
                generation: 2,
                timestamp,
                ..
            } if timestamp == ts(100)
        ));
        assert_eq!(generation.current(), 2);
    }

    #[test]
    fn generation_exhaustion_is_typed_and_does_not_mutate_the_counter() {
        let output = base_output(Vec::new());
        let mut generation = DenseCommandGeneration::from_current(u64::MAX);

        let mapping_error = map_output_to_dense_commands(
            &output,
            vec![pose_update(kf_id(), 1.0)],
            |_| None,
            &depth_buffer(),
            ts(100),
            &mut generation,
        )
        .expect_err("generation exhaustion");
        let DenseCommandMappingError::Generation(error) = mapping_error else {
            panic!("generation error expected");
        };

        assert_eq!(error.current(), u64::MAX);
        assert_eq!(error.reset_generations(), 0);
        assert!(error.includes_pose_updates());
        assert_eq!(error.required(), Some(1));
        assert_eq!(generation.current(), u64::MAX);
    }

    #[test]
    fn reset_and_pose_update_generation_exhaustion_is_transactional() {
        let old_map = SlamMap::new().snapshot().instance_id();
        let new_map = SlamMap::new().snapshot().instance_id();
        let transition =
            crate::MappingSessionTransition::try_new(old_map, new_map).expect("distinct test maps");
        let output = base_output(vec![DiagnosticEvent::MappingSessionReset { transition }]);
        let mut generation = DenseCommandGeneration::from_current(u64::MAX - 1);

        for _ in 0..2 {
            let mapping_error = map_output_to_dense_commands(
                &output,
                vec![pose_update(kf_id(), 1.0)],
                |_| None,
                &depth_buffer(),
                ts(100),
                &mut generation,
            )
            .expect_err("reset plus pose update exceeds the generation domain");
            let DenseCommandMappingError::Generation(error) = mapping_error else {
                panic!("generation error expected");
            };
            assert_eq!(error.current(), u64::MAX - 1);
            assert_eq!(error.reset_generations(), 1);
            assert!(error.includes_pose_updates());
            assert_eq!(error.required(), Some(2));
            assert_eq!(generation.current(), u64::MAX - 1);
        }
    }

    #[test]
    fn reset_and_empty_pose_updates_consume_only_the_reset_generation() {
        let old_map = SlamMap::new().snapshot().instance_id();
        let new_map = SlamMap::new().snapshot().instance_id();
        let transition =
            crate::MappingSessionTransition::try_new(old_map, new_map).expect("distinct test maps");
        let output = base_output(vec![DiagnosticEvent::MappingSessionReset { transition }]);
        let mut generation = DenseCommandGeneration::from_current(u64::MAX - 1);

        let commands = map_commands(
            &output,
            Vec::new(),
            &depth_buffer(),
            ts(100),
            &mut generation,
        );

        assert!(matches!(
            commands.as_slice(),
            [DenseCommand::ResetMappingSession {
                generation: u64::MAX,
                timestamp,
                ..
            }] if *timestamp == ts(100)
        ));
        assert_eq!(generation.current(), u64::MAX);
    }

    #[test]
    fn non_generation_commands_succeed_at_generation_maximum() {
        let empty_output = base_output(Vec::new());
        let mut generation = DenseCommandGeneration::from_current(u64::MAX);
        assert!(
            map_commands(
                &empty_output,
                Vec::new(),
                &depth_buffer(),
                ts(100),
                &mut generation,
            )
            .is_empty()
        );
        assert_eq!(generation.current(), u64::MAX);

        let keyframe_id = kf_id();
        let output = base_output(vec![DiagnosticEvent::KeyframeRemoved {
            keyframe_id,
            reason: KeyframeRemovalReason::Redundant,
        }]);

        let commands = map_commands(
            &output,
            Vec::new(),
            &depth_buffer(),
            ts(100),
            &mut generation,
        );

        assert!(matches!(
            commands.as_slice(),
            [DenseCommand::RemoveKeyframe {
                keyframe_id: actual,
                timestamp,
            }] if *actual == keyframe_id && *timestamp == ts(100)
        ));
        assert_eq!(generation.current(), u64::MAX);
    }

    #[test]
    fn empty_pose_update_builder_does_not_advance_or_fail_at_maximum() {
        let mut generation = DenseCommandGeneration::from_current(u64::MAX);

        let command = apply_pose_updates_command(Vec::new(), ts(100), &mut generation)
            .expect("empty updates need no generation");

        assert!(command.is_none());
        assert_eq!(generation.current(), u64::MAX);
    }

    #[test]
    fn owned_pose_update_builder_commits_only_a_successful_generation() {
        let mut generation = DenseCommandGeneration::from_current(u64::MAX - 1);
        let command =
            apply_pose_updates_command(vec![pose_update(kf_id(), 1.0)], ts(100), &mut generation)
                .expect("last available generation")
                .expect("nonempty update command");
        let DenseCommand::ApplyPoseUpdates {
            updates,
            generation: command_generation,
            timestamp,
        } = command
        else {
            panic!("pose-update command expected");
        };
        assert_eq!(command_generation, u64::MAX);
        assert_eq!(timestamp, ts(100));
        assert_eq!(updates.len(), 1);
        assert_eq!(updates[0].pose().translation(), [1.0, 0.0, 0.0]);
        assert_eq!(generation.current(), u64::MAX);

        let error =
            apply_pose_updates_command(vec![pose_update(kf_id(), 2.0)], ts(200), &mut generation)
                .expect_err("exhausted generation");
        assert_eq!(error.current(), u64::MAX);
        assert!(error.includes_pose_updates());
        assert_eq!(error.required(), Some(1));
        assert_eq!(generation.current(), u64::MAX);
    }
}
