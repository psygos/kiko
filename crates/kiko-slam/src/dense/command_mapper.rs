use std::collections::HashSet;

use crate::dense::DenseCommand;
use crate::dense::ring_buffer::DepthRingBuffer;
use crate::map::KeyframeId;
use crate::{DiagnosticEvent, Pose, PoseStatus, Timestamp, TrackerOutput, TrackingHealth};

/// Maximum timestamp distance (nanoseconds) between a depth frame and a
/// keyframe's stereo pair for valid association. Derived from RIEMANN's
/// bound: δt < v / (2 · max(v_cam, ω · z_max)) ≈ 19 ms.
pub const MAX_ASSOCIATION_WINDOW_NS: i64 = 20_000_000;

/// Producer-side sequence for dense reset and rebuild control commands.
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
        rebuild_generation: bool,
    ) -> Result<Self, DenseCommandGenerationError> {
        let error = DenseCommandGenerationError {
            current: self.0,
            reset_generations,
            rebuild_generation,
        };
        let required = u64::try_from(reset_generations)
            .ok()
            .and_then(|resets| resets.checked_add(u64::from(rebuild_generation)))
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
    rebuild_generation: bool,
}

impl DenseCommandGenerationError {
    pub const fn current(self) -> u64 {
        self.current
    }

    pub const fn reset_generations(self) -> usize {
        self.reset_generations
    }

    pub const fn includes_rebuild(self) -> bool {
        self.rebuild_generation
    }

    pub fn required(self) -> Option<u64> {
        u64::try_from(self.reset_generations)
            .ok()?
            .checked_add(u64::from(self.rebuild_generation))
    }
}

impl std::fmt::Display for DenseCommandGenerationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "dense control-command generation {} cannot reserve {} reset generation(s) and {} rebuild generation(s)",
            self.current,
            self.reset_generations,
            u8::from(self.rebuild_generation)
        )
    }
}

impl std::error::Error for DenseCommandGenerationError {}

/// Map a single `TrackerOutput` (plus any pending loop correction) into
/// dense commands.
///
/// Rules:
/// - `IntegrateKeyframe` is emitted for each `KeyframeCreated` event that
///   has an associated depth frame and whose keyframe was NOT also removed
///   in the same frame.
/// - `RemoveKeyframe` is emitted for each `KeyframeRemoved` event.
/// - `ResetMappingSession` is emitted before every other command when the
///   output contains a mapping-session boundary.
/// - `RebuildFromSnapshot` is emitted when `correction` is `Some`.
/// - No `IntegrateKeyframe` commands are emitted when tracking is `Lost`.
///
/// If the control-command generation is exhausted, no commands are returned and
/// `generation` remains unchanged.
pub fn map_output_to_dense_commands(
    output: &TrackerOutput,
    correction: Option<Vec<(KeyframeId, Pose)>>,
    depth_buffer: &DepthRingBuffer,
    timestamp: Timestamp,
    generation: &mut DenseCommandGeneration,
) -> Result<Vec<DenseCommand>, DenseCommandGenerationError> {
    let reset_count = output
        .events
        .iter()
        .filter(|event| matches!(event, DiagnosticEvent::MappingSessionReset { .. }))
        .count();
    let rebuild_generation = correction.is_some();
    let final_generation = generation.checked_advance_for(reset_count, rebuild_generation)?;
    let batch_error = DenseCommandGenerationError {
        current: generation.current(),
        reset_generations: reset_count,
        rebuild_generation,
    };
    let mut staged_generation = *generation;
    let mut commands = Vec::new();

    // A reset must be sent on the control path before any data command from
    // the new session. Advancing the rebuild generation makes every reset a
    // sequencing barrier for already queued rebuild requests.
    let mut output_map = None;
    for event in &output.events {
        if let DiagnosticEvent::MappingSessionReset { transition } = event {
            let reset_generation = staged_generation.checked_next().ok_or(batch_error)?;
            commands.push(DenseCommand::ResetMappingSession {
                transition: *transition,
                generation: reset_generation,
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
                let pose = match output.pose {
                    Some(p) => p.into_legacy_pose(),
                    None => continue,
                };
                if let Some(depth) = depth_buffer.find_closest(timestamp, MAX_ASSOCIATION_WINDOW_NS)
                {
                    commands.push(DenseCommand::IntegrateKeyframe {
                        keyframe_id: *keyframe_id,
                        pose,
                        depth,
                    });
                }
            }
            DiagnosticEvent::KeyframeRemoved { keyframe_id, .. } => {
                if output_map.is_some_and(|map| map != keyframe_id.map_instance_id()) {
                    continue;
                }
                commands.push(DenseCommand::RemoveKeyframe {
                    keyframe_id: *keyframe_id,
                });
            }
            _ => {}
        }
    }

    // Emit rebuild if loop correction is available.
    if let Some(poses) = correction {
        let rebuild_generation = staged_generation.checked_next().ok_or(batch_error)?;
        commands.push(DenseCommand::RebuildFromSnapshot {
            corrected_poses: poses,
            generation: rebuild_generation,
        });
    }

    debug_assert_eq!(staged_generation, final_generation);
    *generation = final_generation;
    Ok(commands)
}

/// Build one owned rebuild command and commit its generation only on success.
pub fn rebuild_from_snapshot_command(
    corrected_poses: Vec<(KeyframeId, Pose)>,
    generation: &mut DenseCommandGeneration,
) -> Result<DenseCommand, DenseCommandGenerationError> {
    let final_generation = generation.checked_advance_for(0, true)?;
    let command = DenseCommand::RebuildFromSnapshot {
        corrected_poses,
        generation: final_generation.current(),
    };
    *generation = final_generation;
    Ok(command)
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
        ComponentHealth, DegradationLevel, FrameDiagnostics, FrameId, PoseStatus, SystemHealth,
        TrackerOutput,
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
        correction: Option<Vec<(KeyframeId, Pose)>>,
        depth_buffer: &DepthRingBuffer,
        timestamp: Timestamp,
        generation: &mut DenseCommandGeneration,
    ) -> Vec<DenseCommand> {
        map_output_to_dense_commands(output, correction, depth_buffer, timestamp, generation)
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
        let cmds = map_commands(&output, None, &buf, ts(100), &mut gen_);
        assert_eq!(cmds.len(), 1);
        assert!(
            matches!(cmds[0], DenseCommand::IntegrateKeyframe { keyframe_id, .. } if keyframe_id == kf)
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
        let cmds = map_commands(&output, None, &buf, ts(100), &mut gen_);
        assert_eq!(cmds.len(), 1);
        assert!(
            matches!(cmds[0], DenseCommand::RemoveKeyframe { keyframe_id } if keyframe_id == kf)
        );
    }

    #[test]
    fn loop_correction_emits_rebuild() {
        let buf = depth_buffer();
        let output = base_output(vec![]);

        let correction = vec![(kf_id(), Pose::identity())];
        let mut gen_ = DenseCommandGeneration::default();
        let cmds = map_commands(&output, Some(correction), &buf, ts(100), &mut gen_);
        assert_eq!(cmds.len(), 1);
        assert!(matches!(
            cmds[0],
            DenseCommand::RebuildFromSnapshot { generation: 1, .. }
        ));
        assert_eq!(gen_.current(), 1);
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
        let cmds = map_commands(&output, None, &buf, ts(100), &mut gen_);
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
        let cmds = map_commands(&output, None, &buf, ts(100), &mut gen_);
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
        let cmds = map_commands(&output, None, &buf, ts(100), &mut gen_);

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
        let cmds = map_commands(&output, None, &buf, ts(100), &mut gen_);
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
            None,
            &buf,
            ts(100_000_000), // 100ms after depth
            &mut gen_,
        );
        assert!(cmds.is_empty(), "depth too old for association");
    }

    #[test]
    fn generation_increments_on_successive_corrections() {
        let buf = depth_buffer();
        let output = base_output(vec![]);
        let correction = vec![(kf_id(), Pose::identity())];
        let mut gen_ = DenseCommandGeneration::default();

        map_commands(&output, Some(correction.clone()), &buf, ts(100), &mut gen_);
        assert_eq!(gen_.current(), 1);

        map_commands(&output, Some(correction), &buf, ts(200), &mut gen_);
        assert_eq!(gen_.current(), 2);
    }

    #[test]
    fn mapping_session_reset_is_first_and_advances_rebuild_order() {
        let old_map = SlamMap::new().snapshot().instance_id();
        let new_map = SlamMap::new().snapshot().instance_id();
        let transition =
            crate::MappingSessionTransition::try_new(old_map, new_map).expect("distinct test maps");
        let output = base_output(vec![DiagnosticEvent::MappingSessionReset { transition }]);
        let correction = vec![(kf_id(), Pose::identity())];
        let mut generation = DenseCommandGeneration::default();

        let commands = map_commands(
            &output,
            Some(correction),
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
            } if actual == transition
        ));
        assert!(matches!(
            commands[1],
            DenseCommand::RebuildFromSnapshot { generation: 2, .. }
        ));
        assert_eq!(generation.current(), 2);
    }

    #[test]
    fn generation_exhaustion_is_typed_and_does_not_mutate_the_counter() {
        let output = base_output(Vec::new());
        let correction = Vec::new();
        let mut generation = DenseCommandGeneration::from_current(u64::MAX);

        let error = map_output_to_dense_commands(
            &output,
            Some(correction),
            &depth_buffer(),
            ts(100),
            &mut generation,
        )
        .expect_err("generation exhaustion");

        assert_eq!(error.current(), u64::MAX);
        assert_eq!(error.reset_generations(), 0);
        assert!(error.includes_rebuild());
        assert_eq!(error.required(), Some(1));
        assert_eq!(generation.current(), u64::MAX);
    }

    #[test]
    fn reset_and_rebuild_generation_exhaustion_is_transactional() {
        let old_map = SlamMap::new().snapshot().instance_id();
        let new_map = SlamMap::new().snapshot().instance_id();
        let transition =
            crate::MappingSessionTransition::try_new(old_map, new_map).expect("distinct test maps");
        let output = base_output(vec![DiagnosticEvent::MappingSessionReset { transition }]);
        let mut generation = DenseCommandGeneration::from_current(u64::MAX - 1);

        for _ in 0..2 {
            let error = map_output_to_dense_commands(
                &output,
                Some(Vec::new()),
                &depth_buffer(),
                ts(100),
                &mut generation,
            )
            .expect_err("reset plus rebuild exceeds the generation domain");
            assert_eq!(error.current(), u64::MAX - 1);
            assert_eq!(error.reset_generations(), 1);
            assert!(error.includes_rebuild());
            assert_eq!(error.required(), Some(2));
            assert_eq!(generation.current(), u64::MAX - 1);
        }
    }

    #[test]
    fn reset_and_empty_rebuild_exactly_fit_the_generation_domain() {
        let old_map = SlamMap::new().snapshot().instance_id();
        let new_map = SlamMap::new().snapshot().instance_id();
        let transition =
            crate::MappingSessionTransition::try_new(old_map, new_map).expect("distinct test maps");
        let output = base_output(vec![DiagnosticEvent::MappingSessionReset { transition }]);
        let correction = Vec::new();
        let mut generation = DenseCommandGeneration::from_current(u64::MAX - 2);

        let commands = map_commands(
            &output,
            Some(correction),
            &depth_buffer(),
            ts(100),
            &mut generation,
        );

        assert!(matches!(
            commands.as_slice(),
            [
                DenseCommand::ResetMappingSession {
                    generation: reset_generation,
                    ..
                },
                DenseCommand::RebuildFromSnapshot {
                    generation: rebuild_generation,
                    corrected_poses,
                }
            ] if *reset_generation == u64::MAX - 1
                && *rebuild_generation == u64::MAX
                && corrected_poses.is_empty()
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
                None,
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

        let commands = map_commands(&output, None, &depth_buffer(), ts(100), &mut generation);

        assert!(matches!(
            commands.as_slice(),
            [DenseCommand::RemoveKeyframe { keyframe_id: actual }] if *actual == keyframe_id
        ));
        assert_eq!(generation.current(), u64::MAX);
    }

    #[test]
    fn owned_rebuild_builder_commits_only_a_successful_generation() {
        let mut generation = DenseCommandGeneration::from_current(u64::MAX - 1);
        let command = rebuild_from_snapshot_command(Vec::new(), &mut generation)
            .expect("last available generation");
        assert!(matches!(
            command,
            DenseCommand::RebuildFromSnapshot {
                generation: u64::MAX,
                corrected_poses,
            } if corrected_poses.is_empty()
        ));
        assert_eq!(generation.current(), u64::MAX);

        let error = rebuild_from_snapshot_command(Vec::new(), &mut generation)
            .expect_err("exhausted generation");
        assert_eq!(error.current(), u64::MAX);
        assert_eq!(error.required(), Some(1));
        assert_eq!(generation.current(), u64::MAX);
    }
}
