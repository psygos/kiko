use std::collections::HashSet;

use crate::dense::DenseCommand;
use crate::dense::ring_buffer::DepthRingBuffer;
use crate::map::KeyframeId;
use crate::{DiagnosticEvent, Pose, Timestamp, TrackerOutput, TrackingHealth};

/// Maximum timestamp distance (nanoseconds) between a depth frame and a
/// keyframe's stereo pair for valid association. Derived from RIEMANN's
/// bound: δt < v / (2 · max(v_cam, ω · z_max)) ≈ 19 ms.
pub const MAX_ASSOCIATION_WINDOW_NS: i64 = 20_000_000;

/// Map a single `TrackerOutput` (plus any pending loop correction) into
/// dense commands.
///
/// Rules:
/// - `IntegrateKeyframe` is emitted for each `KeyframeCreated` event that
///   has an associated depth frame and whose keyframe was NOT also removed
///   in the same frame.
/// - `RemoveKeyframe` is emitted for each `KeyframeRemoved` event.
/// - `RebuildFromSnapshot` is emitted when `correction` is `Some`.
/// - No `IntegrateKeyframe` commands are emitted when tracking is `Lost`.
pub fn map_output_to_dense_commands(
    output: &TrackerOutput,
    correction: Option<&[(KeyframeId, Pose)]>,
    depth_buffer: &DepthRingBuffer,
    timestamp: Timestamp,
    generation: &mut u64,
) -> Vec<DenseCommand> {
    let mut commands = Vec::new();

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
    let tracking_ok = output.health.tracking != TrackingHealth::Lost;

    for event in &output.events {
        match event {
            DiagnosticEvent::KeyframeCreated { keyframe_id, .. } => {
                if !tracking_ok {
                    continue;
                }
                if removed_this_frame.contains(keyframe_id) {
                    continue;
                }
                let pose = match output.pose {
                    Some(p) => p,
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
                commands.push(DenseCommand::RemoveKeyframe {
                    keyframe_id: *keyframe_id,
                });
            }
            _ => {}
        }
    }

    // Emit rebuild if loop correction is available.
    if let Some(poses) = correction {
        *generation = generation.saturating_add(1);
        commands.push(DenseCommand::RebuildFromSnapshot {
            corrected_poses: poses.to_vec(),
            generation: *generation,
        });
    }

    commands
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense::ring_buffer::DepthRingBuffer;
    use crate::diagnostics::KeyframeRemovalReason;
    use crate::test_helpers::make_depth_image;
    use crate::tracker::BackendStats;
    use crate::{DegradationLevel, FrameDiagnostics, FrameId, SystemHealth, TrackerOutput};

    fn ts(ns: i64) -> Timestamp {
        Timestamp::from_nanos(ns)
    }

    fn depth_at(t_ns: i64) -> crate::DepthImage {
        make_depth_image(FrameId::new(0), ts(t_ns), 2, 2, 1.0)
    }

    fn kf_id() -> KeyframeId {
        use slotmap::SlotMap;
        let mut sm = SlotMap::<KeyframeId, ()>::with_key();
        sm.insert(())
    }

    fn healthy() -> SystemHealth {
        SystemHealth {
            tracking: TrackingHealth::Good,
            backend_alive: true,
            descriptor_alive: true,
            backend_stats: BackendStats::default(),
            degradation: DegradationLevel::Nominal,
        }
    }

    fn lost_health() -> SystemHealth {
        SystemHealth {
            tracking: TrackingHealth::Lost,
            backend_alive: true,
            descriptor_alive: true,
            backend_stats: BackendStats::default(),
            degradation: DegradationLevel::Lost,
        }
    }

    fn base_output(events: Vec<DiagnosticEvent>) -> TrackerOutput {
        TrackerOutput {
            pose: Some(Pose::identity()),
            inliers: 0,
            keyframe: None,
            stereo_matches: None,
            frame_id: FrameId::new(1),
            health: healthy(),
            diagnostics: FrameDiagnostics::empty(1, 1),
            events,
        }
    }

    #[test]
    fn keyframe_created_with_depth_emits_integrate() {
        let kf = kf_id();
        let mut buf = DepthRingBuffer::new(4);
        buf.push(depth_at(100));

        let output = base_output(vec![DiagnosticEvent::KeyframeCreated {
            keyframe_id: kf,
            landmarks: 10,
        }]);

        let mut gen_ = 0;
        let cmds = map_output_to_dense_commands(&output, None, &buf, ts(100), &mut gen_);
        assert_eq!(cmds.len(), 1);
        assert!(
            matches!(cmds[0], DenseCommand::IntegrateKeyframe { keyframe_id, .. } if keyframe_id == kf)
        );
    }

    #[test]
    fn keyframe_removed_emits_remove() {
        let kf = kf_id();
        let buf = DepthRingBuffer::new(4);

        let output = base_output(vec![DiagnosticEvent::KeyframeRemoved {
            keyframe_id: kf,
            reason: KeyframeRemovalReason::Redundant,
        }]);

        let mut gen_ = 0;
        let cmds = map_output_to_dense_commands(&output, None, &buf, ts(100), &mut gen_);
        assert_eq!(cmds.len(), 1);
        assert!(
            matches!(cmds[0], DenseCommand::RemoveKeyframe { keyframe_id } if keyframe_id == kf)
        );
    }

    #[test]
    fn loop_correction_emits_rebuild() {
        let buf = DepthRingBuffer::new(4);
        let output = base_output(vec![]);

        let correction = vec![(kf_id(), Pose::identity())];
        let mut gen_ = 0;
        let cmds =
            map_output_to_dense_commands(&output, Some(&correction), &buf, ts(100), &mut gen_);
        assert_eq!(cmds.len(), 1);
        assert!(matches!(
            cmds[0],
            DenseCommand::RebuildFromSnapshot { generation: 1, .. }
        ));
        assert_eq!(gen_, 1);
    }

    #[test]
    fn no_command_when_no_depth_available() {
        let kf = kf_id();
        let buf = DepthRingBuffer::new(4); // empty

        let output = base_output(vec![DiagnosticEvent::KeyframeCreated {
            keyframe_id: kf,
            landmarks: 5,
        }]);

        let mut gen_ = 0;
        let cmds = map_output_to_dense_commands(&output, None, &buf, ts(100), &mut gen_);
        assert!(cmds.is_empty(), "no depth available → no integrate command");
    }

    #[test]
    fn no_integrate_when_tracking_lost() {
        let kf = kf_id();
        let mut buf = DepthRingBuffer::new(4);
        buf.push(depth_at(100));

        let mut output = base_output(vec![DiagnosticEvent::KeyframeCreated {
            keyframe_id: kf,
            landmarks: 5,
        }]);
        output.health = lost_health();

        let mut gen_ = 0;
        let cmds = map_output_to_dense_commands(&output, None, &buf, ts(100), &mut gen_);
        assert!(
            cmds.is_empty(),
            "should not integrate when tracking is lost"
        );
    }

    #[test]
    fn coalesce_create_and_remove_same_frame() {
        let kf = kf_id();
        let mut buf = DepthRingBuffer::new(4);
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

        let mut gen_ = 0;
        let cmds = map_output_to_dense_commands(&output, None, &buf, ts(100), &mut gen_);
        // Should only have RemoveKeyframe, not IntegrateKeyframe.
        assert_eq!(cmds.len(), 1);
        assert!(matches!(cmds[0], DenseCommand::RemoveKeyframe { .. }));
    }

    #[test]
    fn depth_outside_window_not_associated() {
        let kf = kf_id();
        let mut buf = DepthRingBuffer::new(4);
        buf.push(depth_at(0)); // very old

        let output = base_output(vec![DiagnosticEvent::KeyframeCreated {
            keyframe_id: kf,
            landmarks: 5,
        }]);

        let mut gen_ = 0;
        let cmds = map_output_to_dense_commands(
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
        let buf = DepthRingBuffer::new(4);
        let output = base_output(vec![]);
        let correction = vec![(kf_id(), Pose::identity())];
        let mut gen_ = 0;

        map_output_to_dense_commands(&output, Some(&correction), &buf, ts(100), &mut gen_);
        assert_eq!(gen_, 1);

        map_output_to_dense_commands(&output, Some(&correction), &buf, ts(200), &mut gen_);
        assert_eq!(gen_, 2);
    }
}
