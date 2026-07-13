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
        let loop_relative = Self::try_loop_relative_pose64(match_pose, verified.query_pose_world())
            .map_err(|source| LoopApplyError::PoseConversion {
                operation: "relative loop-pose construction",
                keyframe_id: None,
                source,
            })?;

        essential_graph
            .add_loop_edge(EssentialEdge {
                a: match_kf,
                b: query_kf,
                kind: EssentialEdgeKind::Loop,
                relative_pose: loop_relative,
                information: heuristic_loop_information_from_inlier_count(verified.inlier_count()),
            })
            .map_err(|source| LoopApplyError::EssentialGraph { source })?;

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
            initial_poses.push(Pose64::try_from_pose32(pose).map_err(|source| {
                LoopApplyError::PoseConversion {
                    operation: "initial pose widening",
                    keyframe_id: Some(keyframe_id),
                    source,
                }
            })?);
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
        let mut corrected_poses = HashMap::with_capacity(input.keyframe_ids.len());
        let mut pose_updates = Vec::with_capacity(input.keyframe_ids.len());
        for (keyframe_id, corrected_pose) in input
            .keyframe_ids
            .iter()
            .copied()
            .zip(result.corrected_poses)
        {
            let corrected_pose = corrected_pose.try_to_pose32().map_err(|source| {
                LoopApplyError::PoseConversion {
                    operation: "optimized pose narrowing",
                    keyframe_id: Some(keyframe_id),
                    source,
                }
            })?;
            corrected_poses.insert(keyframe_id, corrected_pose);
            pose_updates.push((keyframe_id, corrected_pose));
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

        map.apply_geometry_updates(&pose_updates, &point_updates)?;

        Ok(pose_updates)
    }

    pub(crate) fn correction_magnitude(
        current_query_pose_world: Pose,
        estimated_query_pose_world: Pose,
    ) -> Result<(f64, f64), crate::Pose64Error> {
        let correction =
            Self::try_pose_correction64(current_query_pose_world, estimated_query_pose_world)?;
        Ok((
            translation_norm_m(correction),
            rotation_angle_deg(correction),
        ))
    }

    pub(crate) fn reject_reason(error: &LoopDetectError) -> LoopClosureRejectReason {
        match error {
            LoopDetectError::TooFewCorrespondences { count } => {
                LoopClosureRejectReason::TooFewCorrespondences { count: *count }
            }
            LoopDetectError::VerificationFailed(_) => LoopClosureRejectReason::VerificationFailed,
            LoopDetectError::CorrectionEvaluation { .. } => {
                LoopClosureRejectReason::CorrectionEvaluationFailed
            }
            LoopDetectError::CorrectionTooLarge {
                translation_m,
                rotation_deg,
            } => LoopClosureRejectReason::CorrectionTooLarge {
                translation_m: *translation_m,
                rotation_deg: *rotation_deg,
            },
            LoopDetectError::ApplyFailed(_) => LoopClosureRejectReason::ApplyFailed,
        }
    }

    fn try_loop_relative_pose64(
        match_pose: Pose,
        query_pose_world: Pose,
    ) -> Result<Pose64, crate::Pose64Error> {
        let match_pose = Pose64::try_from_pose32(match_pose)?;
        let query_pose_world = Pose64::try_from_pose32(query_pose_world)?;
        Ok(match_pose.inverse().compose(query_pose_world))
    }

    fn try_pose_correction64(
        current_pose_world: Pose,
        estimated_pose_world: Pose,
    ) -> Result<Pose64, crate::Pose64Error> {
        let current_pose_world = Pose64::try_from_pose32(current_pose_world)?;
        let estimated_pose_world = Pose64::try_from_pose32(estimated_pose_world)?;
        Ok(estimated_pose_world.compose(current_pose_world.inverse()))
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

fn translation_norm_m(pose: Pose64) -> f64 {
    let t = pose.translation();
    t[0].hypot(t[1]).hypot(t[2])
}

fn rotation_angle_deg(pose: Pose64) -> f64 {
    let r = pose.rotation();
    let sin_theta = 0.5
        * (r[2][1] - r[1][2])
            .hypot(r[0][2] - r[2][0])
            .hypot(r[1][0] - r[0][1]);
    let cos_theta = 0.5 * (r[0][0] + r[1][1] + r[2][2] - 1.0);
    sin_theta.atan2(cos_theta).to_degrees()
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
