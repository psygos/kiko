use crate::global_map::GlobalMap;
use crate::map::KeyframeId;
use crate::pose_graph::{
    EssentialEdge, EssentialEdgeKind, PoseGraphConfig, PoseGraphError, PoseGraphOptimizer,
    PoseGraphTermination,
};
use crate::{
    LoopApplyError, LoopClosureRejectReason, LoopDetectError, Point3, Pose, Pose64, VerifiedLoop,
};
use std::collections::HashMap;

const MIN_OPTIMIZATION_KEYFRAMES: usize = 2;

pub(crate) struct LoopManager {
    optimizer: PoseGraphOptimizer,
}

impl LoopManager {
    pub(crate) fn new(config: PoseGraphConfig) -> Self {
        Self {
            optimizer: PoseGraphOptimizer::new(config),
        }
    }

    #[cfg(any(not(feature = "vio"), test))]
    pub(crate) fn apply_verified_loop(
        &self,
        global_map: &mut GlobalMap,
        verified: &VerifiedLoop,
    ) -> Result<Vec<(KeyframeId, Pose)>, LoopApplyError> {
        let (candidate, corrections) = self.prepare_verified_loop(global_map, verified)?;
        *global_map = candidate;
        Ok(corrections)
    }

    pub(crate) fn prepare_verified_loop(
        &self,
        global_map: &GlobalMap,
        verified: &VerifiedLoop,
    ) -> Result<(GlobalMap, Vec<(KeyframeId, Pose)>), LoopApplyError> {
        let current = global_map.map().snapshot();
        if verified.map_snapshot() != current {
            return Err(LoopApplyError::StaleCorrection {
                proof: verified.map_snapshot(),
                current,
            });
        }
        let mut candidate = global_map.clone();
        let corrections = self.apply_verified_loop_to_candidate(&mut candidate, verified)?;
        Ok((candidate, corrections))
    }

    fn apply_verified_loop_to_candidate(
        &self,
        global_map: &mut GlobalMap,
        verified: &VerifiedLoop,
    ) -> Result<Vec<(KeyframeId, Pose)>, LoopApplyError> {
        let (map, essential_graph) = global_map.split_mut();
        let query_kf = verified.query_kf();
        let match_kf = verified.match_kf();
        let match_pose = map
            .keyframe(match_kf)
            .ok_or(crate::map::MapError::KeyframeNotFound(match_kf))?
            .pose();
        let loop_relative = Pose64::from_pose32(Self::loop_relative_pose(
            match_pose,
            verified.query_pose_world(),
        ));

        essential_graph.add_loop_edge(EssentialEdge {
            a: match_kf,
            b: query_kf,
            kind: EssentialEdgeKind::Loop,
            relative_pose: loop_relative,
            information: heuristic_loop_information_from_inlier_count(verified.inlier_count()),
        });

        let input = essential_graph.pose_graph_input()?;
        if input.keyframe_ids.len() < MIN_OPTIMIZATION_KEYFRAMES || input.edges.is_empty() {
            return Ok(Vec::new());
        }

        let mut old_poses = HashMap::with_capacity(input.keyframe_ids.len());
        let mut initial_poses = Vec::with_capacity(input.keyframe_ids.len());
        for &keyframe_id in &input.keyframe_ids {
            let pose = map
                .keyframe(keyframe_id)
                .ok_or(crate::map::MapError::KeyframeNotFound(keyframe_id))?
                .pose();
            old_poses.insert(keyframe_id, pose);
            initial_poses.push(Pose64::from_pose32(pose));
        }

        let result = self.optimizer.optimize(&input.edges, &mut initial_poses)?;
        if !matches!(result.termination, PoseGraphTermination::Converged { .. }) {
            return Err(LoopApplyError::PoseGraph {
                source: PoseGraphError::NotConverged {
                    outer_iterations: result.outer_iterations,
                    last_linear_solve_residual_norm: result.last_linear_solve_residual_norm,
                },
            });
        }
        let corrected_poses: HashMap<KeyframeId, Pose> = input
            .keyframe_ids
            .iter()
            .copied()
            .zip(
                result
                    .corrected_poses
                    .into_iter()
                    .map(|pose| pose.to_pose32()),
            )
            .collect();

        for (keyframe_id, corrected_pose) in &corrected_poses {
            map.set_keyframe_pose(*keyframe_id, *corrected_pose)?;
        }

        let mut point_updates = Vec::new();
        for (point_id, point) in map.points() {
            let world = point.position();
            let world_vec = [world.x, world.y, world.z];
            let mut accum = [0.0_f32; 3];
            let mut count = 0usize;

            for observation in point.observations() {
                let keyframe_id = observation.keyframe_id();
                let Some(old_pose) = old_poses.get(&keyframe_id).copied() else {
                    continue;
                };
                let Some(new_pose) = corrected_poses.get(&keyframe_id).copied() else {
                    continue;
                };

                let camera = crate::math::transform_point(
                    old_pose.rotation(),
                    old_pose.translation(),
                    world_vec,
                );
                let corrected_world = camera_to_world(
                    new_pose,
                    Point3 {
                        x: camera[0],
                        y: camera[1],
                        z: camera[2],
                    },
                );
                accum[0] += corrected_world.x;
                accum[1] += corrected_world.y;
                accum[2] += corrected_world.z;
                count = count.saturating_add(1);
            }

            if count > 0 {
                let inv_count = 1.0_f32 / count as f32;
                point_updates.push((
                    point_id,
                    Point3 {
                        x: accum[0] * inv_count,
                        y: accum[1] * inv_count,
                        z: accum[2] * inv_count,
                    },
                ));
            }
        }

        for (point_id, corrected_world) in point_updates {
            map.set_map_point_position(point_id, corrected_world)?;
        }

        Ok(corrected_poses.into_iter().collect())
    }

    pub(crate) fn correction_magnitude(match_pose: Pose, query_pose_world: Pose) -> (f32, f32) {
        let loop_relative = Self::loop_relative_pose(match_pose, query_pose_world);
        (
            loop_translation_norm(loop_relative),
            loop_rotation_angle_deg(loop_relative),
        )
    }

    pub(crate) fn reject_reason(error: &LoopDetectError) -> LoopClosureRejectReason {
        match error {
            LoopDetectError::TooFewCorrespondences { count } => {
                LoopClosureRejectReason::TooFewCorrespondences { count: *count }
            }
            LoopDetectError::VerificationFailed(_) => LoopClosureRejectReason::VerificationFailed,
            LoopDetectError::CorrectionTooLarge {
                translation,
                rotation_deg,
            } => LoopClosureRejectReason::CorrectionTooLarge {
                translation_m: *translation,
                rotation_deg: *rotation_deg,
            },
            LoopDetectError::ApplyFailed(_) => LoopClosureRejectReason::ApplyFailed,
        }
    }

    fn loop_relative_pose(match_pose: Pose, query_pose_world: Pose) -> Pose {
        Pose64::from_pose32(match_pose)
            .inverse()
            .compose(Pose64::from_pose32(query_pose_world))
            .to_pose32()
    }
}

fn camera_to_world(pose_world: Pose, point: Point3) -> Point3 {
    let inv = pose_world.inverse();
    let v = crate::math::transform_point(
        inv.rotation(),
        inv.translation(),
        [point.x, point.y, point.z],
    );
    Point3 {
        x: v[0],
        y: v[1],
        z: v[2],
    }
}

fn loop_translation_norm(pose: Pose) -> f32 {
    let t = pose.translation();
    (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt()
}

fn loop_rotation_angle_deg(pose: Pose) -> f32 {
    let r = pose.rotation();
    let trace = r[0][0] + r[1][1] + r[2][2];
    let cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0);
    cos_theta.acos().to_degrees()
}

/// Heuristic isotropic information scaled by verified PnP support. This is not
/// a covariance-derived uncertainty estimate.
fn heuristic_loop_information_from_inlier_count(inlier_count: usize) -> [[f64; 6]; 6] {
    let weight = inlier_count as f64;
    let mut info = [[0.0_f64; 6]; 6];
    for (axis, row) in info.iter_mut().enumerate() {
        row[axis] = weight;
    }
    info
}
