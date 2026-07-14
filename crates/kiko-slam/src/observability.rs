use crate::{
    BaResult, ComponentHealth, DegradationLevel, DiagnosticEvent, FrameDiagnostics, KeyframeStatus,
    LoopClosureRejectReason, LoopClosureStatus, RerunSink, SystemHealth, Timestamp, TrackingHealth,
    VizLogError,
};

const TIMELINE_CAPTURE_NS: &str = "capture_ns";

const PATH_HEALTH_PNP_INLIER_RATIO: &str = "diagnostics/health/pnp_inlier_ratio";
const PATH_HEALTH_VISUAL_PROPOSAL_ACCEPTED_INLIER_REPROJECTION_RMSE: &str =
    "diagnostics/health/visual_proposal_pnp_accepted_inlier_reprojection_rmse_px";
const PATH_HEALTH_VIO_PROPOSAL_ACCEPTED_INLIER_REPROJECTION_RMSE: &str =
    "diagnostics/health/vio_proposal_pnp_accepted_inlier_reprojection_rmse_px";
const PATH_HEALTH_VISUAL_PROPOSAL_SHARED_ACCEPTED_INLIER_REPROJECTION_RMSE: &str =
    "diagnostics/health/visual_proposal_shared_accepted_inlier_reprojection_rmse_px";
const PATH_HEALTH_VIO_PROPOSAL_SHARED_ACCEPTED_INLIER_REPROJECTION_RMSE: &str =
    "diagnostics/health/vio_proposal_shared_accepted_inlier_reprojection_rmse_px";
const PATH_HEALTH_PNP_PROJECTABLE_TRACKED_REPROJECTION_RMSE: &str =
    "diagnostics/health/pnp_projectable_tracked_observation_reprojection_rmse_px";
const PATH_HEALTH_PNP_PROJECTABLE_TRACKED_REPROJECTION_MAX: &str =
    "diagnostics/health/pnp_projectable_tracked_observation_reprojection_max_px";
const PATH_HEALTH_PNP_PROJECTABLE_TRACKED_REPROJECTION_MSE_PER_AXIS: &str =
    "diagnostics/health/pnp_projectable_tracked_observation_reprojection_mse_per_axis_px2";
const PATH_HEALTH_VISUAL_PROPOSAL_PNP_PROJECTABLE_TRACKED_REPROJECTION_RMSE: &str =
    "diagnostics/health/visual_proposal_pnp_projectable_tracked_observation_reprojection_rmse_px";
const PATH_HEALTH_VIO_PROPOSAL_PNP_PROJECTABLE_TRACKED_REPROJECTION_RMSE: &str =
    "diagnostics/health/vio_proposal_pnp_projectable_tracked_observation_reprojection_rmse_px";
const PATH_HEALTH_VISUAL_PROPOSAL_SHARED_TRACKED_REPROJECTION_RMSE: &str = "diagnostics/health/visual_proposal_shared_projectable_tracked_observation_reprojection_rmse_px";
const PATH_HEALTH_VIO_PROPOSAL_SHARED_TRACKED_REPROJECTION_RMSE: &str =
    "diagnostics/health/vio_proposal_shared_projectable_tracked_observation_reprojection_rmse_px";
const PATH_HEALTH_PNP_INLIER_REPROJECTION_RMSE: &str =
    "diagnostics/health/pnp_inlier_reprojection_rmse_px";
const PATH_HEALTH_PNP_INLIER_REPROJECTION_MAX: &str =
    "diagnostics/health/pnp_inlier_reprojection_max_px";
const PATH_HEALTH_PNP_INLIER_REPROJECTION_MSE_PER_AXIS: &str =
    "diagnostics/health/pnp_inlier_reprojection_mse_per_axis_px2";
const PATH_HEALTH_TRACKING_STATE: &str = "diagnostics/health/tracking_state";
const PATH_HEALTH_DEGRADATION_LEVEL: &str = "diagnostics/health/degradation_level";
const PATH_HEALTH_BACKEND_STATE: &str = "diagnostics/health/backend_state";
const PATH_HEALTH_DESCRIPTOR_STATE: &str = "diagnostics/health/descriptor_state";
const PATH_HEALTH_BACKEND_ALIVE: &str = "diagnostics/health/backend_alive";
const PATH_HEALTH_DESCRIPTOR_ALIVE: &str = "diagnostics/health/descriptor_alive";
const PATH_HEALTH_BACKEND_SUBMITTED: &str = "diagnostics/health/backend_submitted";
const PATH_HEALTH_BACKEND_APPLIED: &str = "diagnostics/health/backend_applied";
const PATH_HEALTH_BACKEND_UNCHANGED: &str = "diagnostics/health/backend_unchanged";
const PATH_HEALTH_BACKEND_DROPPED_FULL: &str = "diagnostics/health/backend_dropped_full";
const PATH_HEALTH_BACKEND_DROPPED_UNAVAILABLE: &str =
    "diagnostics/health/backend_dropped_unavailable";
const PATH_HEALTH_BACKEND_STALE: &str = "diagnostics/health/backend_stale";
const PATH_HEALTH_BACKEND_REJECTED: &str = "diagnostics/health/backend_rejected";
const PATH_HEALTH_BACKEND_WORKER_FAILURES: &str = "diagnostics/health/backend_worker_failures";
const PATH_HEALTH_BACKEND_RESTART_FAILURES: &str = "diagnostics/health/backend_restart_failures";
const PATH_HEALTH_BACKEND_RESPAWN_COUNT: &str = "diagnostics/health/backend_respawn_count";
const PATH_HEALTH_BACKEND_RESPAWN_EXHAUSTED: &str = "diagnostics/health/backend_respawn_exhausted";
const PATH_HEALTH_BACKEND_PANICS: &str = "diagnostics/health/backend_panics";

const PATH_TRACKING_FEATURES_DETECTED: &str = "diagnostics/tracking/features_detected";
const PATH_TRACKING_FEATURES_MATCHED: &str = "diagnostics/tracking/features_matched";
const PATH_TRACKING_PNP_TRACKED_OBSERVATIONS: &str =
    "diagnostics/tracking/pnp_tracked_observations";
const PATH_TRACKING_PNP_ACCEPTED_INLIERS: &str = "diagnostics/tracking/pnp_accepted_inliers";
const PATH_TRACKING_VISUAL_PROPOSAL_PROJECTABLE_ACCEPTED_INLIERS: &str =
    "diagnostics/tracking/visual_proposal_projectable_accepted_inliers";
const PATH_TRACKING_VIO_PROPOSAL_PROJECTABLE_ACCEPTED_INLIERS: &str =
    "diagnostics/tracking/vio_proposal_projectable_accepted_inliers";
const PATH_TRACKING_SHARED_PROJECTABLE_ACCEPTED_INLIERS: &str =
    "diagnostics/tracking/shared_projectable_accepted_inliers";
const PATH_TRACKING_PNP_PROJECTABLE_TRACKED_OBSERVATIONS: &str =
    "diagnostics/tracking/pnp_projectable_tracked_observations";
const PATH_TRACKING_VISUAL_PROPOSAL_PNP_PROJECTABLE_TRACKED_OBSERVATIONS: &str =
    "diagnostics/tracking/visual_proposal_pnp_projectable_tracked_observations";
const PATH_TRACKING_VIO_PROPOSAL_PNP_PROJECTABLE_TRACKED_OBSERVATIONS: &str =
    "diagnostics/tracking/vio_proposal_pnp_projectable_tracked_observations";
const PATH_TRACKING_SHARED_PROJECTABLE_TRACKED_OBSERVATIONS: &str =
    "diagnostics/tracking/shared_projectable_tracked_observations";
const PATH_TRACKING_POSE_SOURCE: &str = "diagnostics/tracking/pose_source";
const PATH_TRACKING_VIO_PROPOSAL_RAN: &str = "diagnostics/tracking/vio_proposal_ran";
const PATH_TRACKING_VIO_PROPOSAL_ADOPTED: &str = "diagnostics/tracking/vio_proposal_adopted";
const PATH_TRACKING_VIO_PROPOSAL_REJECTED_UNUSABLE_SOLVE: &str =
    "diagnostics/tracking/vio_proposal_rejected_unusable_solve";
const PATH_TRACKING_VIO_PROPOSAL_REJECTED_INSUFFICIENT_CURRENT_VIO_SUPPORT: &str =
    "diagnostics/tracking/vio_proposal_rejected_insufficient_current_vio_observation_support";
const PATH_TRACKING_VIO_PROPOSAL_REJECTED_INSUFFICIENT_SHARED_SUPPORT: &str =
    "diagnostics/tracking/vio_proposal_rejected_insufficient_shared_accepted_inlier_support";
const PATH_TRACKING_VIO_PROPOSAL_REJECTED_CHANGED_SUPPORT: &str =
    "diagnostics/tracking/vio_proposal_rejected_changed_accepted_inlier_projectability";
const PATH_TRACKING_VIO_PROPOSAL_REJECTED_HIGHER_SHARED_RMSE: &str =
    "diagnostics/tracking/vio_proposal_rejected_higher_shared_accepted_inlier_reprojection_rmse";
const PATH_TRACKING_RANSAC_ITERATIONS: &str = "diagnostics/tracking/ransac_iterations";
const PATH_TRACKING_PNP_RANSAC_CANDIDATE_PROJECTION_REJECTIONS: &str =
    "diagnostics/tracking/pnp_ransac_candidate_projection_rejections";
const PATH_TRACKING_PNP_REFINEMENT_APPLIED: &str = "diagnostics/tracking/pnp_refinement_applied";
const PATH_TRACKING_PNP_REFINEMENT_ITERATIONS: &str =
    "diagnostics/tracking/pnp_refinement_iterations";
const PATH_TRACKING_PARALLAX: &str = "diagnostics/tracking/parallax_px";
const PATH_TRACKING_COVISIBILITY: &str = "diagnostics/tracking/covisibility";
const PATH_TRACKING_KEYFRAME_CREATED: &str = "diagnostics/tracking/keyframe_created";

const PATH_TIMING_TRACKING_MS: &str = "diagnostics/timing/tracking_ms";

const PATH_MAP_KEYFRAMES: &str = "diagnostics/map/keyframes";
const PATH_MAP_POINTS: &str = "diagnostics/map/points";
const PATH_DEPTH_REORDER_WARNINGS: &str = "diagnostics/depth/reorder_warnings";

const PATH_TRI_CANDIDATES: &str = "diagnostics/triangulation/candidates";
const PATH_TRI_KEPT: &str = "diagnostics/triangulation/kept";
const PATH_TRI_DROPPED_DISPARITY: &str = "diagnostics/triangulation/dropped_disparity";
const PATH_TRI_DROPPED_DEPTH: &str = "diagnostics/triangulation/dropped_depth";
const PATH_TRI_DROPPED_NUMERICAL: &str = "diagnostics/triangulation/dropped_numerical";
const PATH_TRI_DROPPED_DUPLICATE: &str = "diagnostics/triangulation/dropped_duplicate";

const PATH_BA_FINAL_COST: &str = "diagnostics/ba/final_cost";
const PATH_BA_ITERATIONS: &str = "diagnostics/ba/iterations";
const PATH_BA_ACCEPTED_STEPS: &str = "diagnostics/ba/accepted_steps";
const PATH_BA_STALLED: &str = "diagnostics/ba/stalled";
const PATH_BA_STATIONARY: &str = "diagnostics/ba/stationary";
const PATH_POSE_BA_ITERATIONS: &str = "diagnostics/pose_ba/iterations";
const PATH_POSE_BA_CONVERGED: &str = "diagnostics/pose_ba/converged";
#[cfg(feature = "vio")]
const PATH_VIO_FINAL_MIXED_OBJECTIVE: &str = "diagnostics/vio/final_mixed_objective";
#[cfg(feature = "vio")]
const PATH_VIO_ATTEMPTED_ITERATIONS: &str = "diagnostics/vio/attempted_iterations";
#[cfg(feature = "vio")]
const PATH_VIO_CONVERGED: &str = "diagnostics/vio/converged";
#[cfg(feature = "vio")]
const PATH_VIO_TERMINATION: &str = "diagnostics/vio/termination";
#[cfg(feature = "vio")]
const PATH_VIO_ACCEPTED_STEPS: &str = "diagnostics/vio/accepted_steps";
#[cfg(feature = "vio")]
const PATH_VIO_REJECTED_STEPS: &str = "diagnostics/vio/rejected_steps";
#[cfg(feature = "vio")]
const PATH_VIO_REJECTED_NONPROJECTABLE_CANDIDATE_STEPS: &str =
    "diagnostics/vio/rejected_nonprojectable_candidate_steps";
#[cfg(feature = "vio")]
const PATH_VIO_LAST_FRAME_ACTIVE_VISUAL_FACTORS: &str =
    "diagnostics/vio/last_frame_active_visual_factors";
#[cfg(feature = "vio")]
const PATH_VIO_INITIALLY_EXCLUDED_NONPROJECTABLE_VISUAL_FACTORS: &str =
    "diagnostics/vio/initially_excluded_nonprojectable_visual_factors";
#[cfg(feature = "vio")]
const PATH_VIO_REGULARIZED_IMU_RESIDUAL_FACTORS: &str =
    "diagnostics/vio/regularized_imu_residual_factors";
#[cfg(feature = "vio")]
const PATH_VIO_FLOORED_ACCEL_BIAS_RANDOM_WALK_FACTORS: &str =
    "diagnostics/vio/floored_accel_bias_random_walk_factors";
#[cfg(feature = "vio")]
const PATH_VIO_FLOORED_GYRO_BIAS_RANDOM_WALK_FACTORS: &str =
    "diagnostics/vio/floored_gyro_bias_random_walk_factors";
#[cfg(feature = "vio")]
const PATH_VIO_CALIBRATED_BIAS_PRIOR_ACTIVE: &str = "diagnostics/vio/calibrated_bias_prior_active";
#[cfg(feature = "vio")]
const PATH_VIO_OBJECTIVE_REPROJECTION_ROBUST_PX2: &str =
    "diagnostics/vio/objective/reprojection_robust_px2";
#[cfg(feature = "vio")]
const PATH_VIO_OBJECTIVE_IMU_MAHALANOBIS: &str = "diagnostics/vio/objective/imu_mahalanobis";
#[cfg(feature = "vio")]
const PATH_VIO_OBJECTIVE_BIAS_RANDOM_WALK_MAHALANOBIS: &str =
    "diagnostics/vio/objective/bias_random_walk_mahalanobis";
#[cfg(feature = "vio")]
const PATH_VIO_OBJECTIVE_VELOCITY_ANCHOR_MAHALANOBIS: &str =
    "diagnostics/vio/objective/velocity_anchor_mahalanobis";
#[cfg(feature = "vio")]
const PATH_VIO_OBJECTIVE_BIAS_PRIOR_MAHALANOBIS: &str =
    "diagnostics/vio/objective/bias_prior_mahalanobis";

const PATH_LOOP_CANDIDATES: &str = "diagnostics/loop/candidates";
const PATH_LOOP_APPLIED: &str = "diagnostics/loop/applied";

const PATH_EVENTS_LOG: &str = "diagnostics/events/log";

fn set_capture_time(rec: &rerun::RecordingStream, timestamp: Timestamp) {
    rec.set_time(
        TIMELINE_CAPTURE_NS,
        rerun::TimeCell::from_duration_nanos(timestamp.as_nanos()),
    );
}

fn diagnostics_scalars(diag: &FrameDiagnostics) -> Vec<(&'static str, f64)> {
    let mut scalars = Vec::new();
    scalars.push((PATH_MAP_KEYFRAMES, diag.map_keyframes as f64));
    scalars.push((PATH_MAP_POINTS, diag.map_points as f64));
    if let Some(v) = diag.depth_reorder_warnings {
        scalars.push((PATH_DEPTH_REORDER_WARNINGS, v as f64));
    }
    scalars.push((PATH_LOOP_CANDIDATES, diag.loop_candidate_count as f64));
    scalars.push((
        PATH_TRACKING_KEYFRAME_CREATED,
        if diag.keyframe_status == Some(KeyframeStatus::Created) {
            1.0
        } else {
            0.0
        },
    ));
    scalars.push((
        PATH_LOOP_APPLIED,
        if diag.loop_closure_status == Some(LoopClosureStatus::Applied) {
            1.0
        } else {
            0.0
        },
    ));

    if let Some(v) = diag.pnp_inlier_ratio {
        scalars.push((PATH_HEALTH_PNP_INLIER_RATIO, v.value() as f64));
    }
    if let Some(v) = diag.visual_proposal_accepted_inlier_reprojection_rmse_px {
        scalars.push((
            PATH_HEALTH_VISUAL_PROPOSAL_ACCEPTED_INLIER_REPROJECTION_RMSE,
            v.value_px() as f64,
        ));
    }
    if let Some(v) = diag.vio_proposal_accepted_inlier_reprojection_rmse_px {
        scalars.push((
            PATH_HEALTH_VIO_PROPOSAL_ACCEPTED_INLIER_REPROJECTION_RMSE,
            v.value_px() as f64,
        ));
    }
    if let Some(v) = diag.visual_proposal_shared_accepted_inlier_reprojection_rmse_px {
        scalars.push((
            PATH_HEALTH_VISUAL_PROPOSAL_SHARED_ACCEPTED_INLIER_REPROJECTION_RMSE,
            v.value_px() as f64,
        ));
    }
    if let Some(v) = diag.vio_proposal_shared_accepted_inlier_reprojection_rmse_px {
        scalars.push((
            PATH_HEALTH_VIO_PROPOSAL_SHARED_ACCEPTED_INLIER_REPROJECTION_RMSE,
            v.value_px() as f64,
        ));
    }
    if let Some(v) = diag.pnp_projectable_tracked_observation_reprojection_rmse_px {
        scalars.push((
            PATH_HEALTH_PNP_PROJECTABLE_TRACKED_REPROJECTION_RMSE,
            v.value_px() as f64,
        ));
    }
    if let Some(v) = diag.pnp_projectable_tracked_observation_reprojection_max_px {
        scalars.push((
            PATH_HEALTH_PNP_PROJECTABLE_TRACKED_REPROJECTION_MAX,
            v.value_px() as f64,
        ));
    }
    if let Some(v) = diag.pnp_projectable_tracked_observation_reprojection_mse_per_axis_px2 {
        scalars.push((
            PATH_HEALTH_PNP_PROJECTABLE_TRACKED_REPROJECTION_MSE_PER_AXIS,
            v.value_px2(),
        ));
    }
    if let Some(v) = diag.visual_proposal_projectable_tracked_observation_reprojection_rmse_px {
        scalars.push((
            PATH_HEALTH_VISUAL_PROPOSAL_PNP_PROJECTABLE_TRACKED_REPROJECTION_RMSE,
            v.value_px() as f64,
        ));
    }
    if let Some(v) = diag.vio_proposal_projectable_tracked_observation_reprojection_rmse_px {
        scalars.push((
            PATH_HEALTH_VIO_PROPOSAL_PNP_PROJECTABLE_TRACKED_REPROJECTION_RMSE,
            v.value_px() as f64,
        ));
    }
    if let Some(v) =
        diag.visual_proposal_shared_projectable_tracked_observation_reprojection_rmse_px
    {
        scalars.push((
            PATH_HEALTH_VISUAL_PROPOSAL_SHARED_TRACKED_REPROJECTION_RMSE,
            v.value_px() as f64,
        ));
    }
    if let Some(v) = diag.vio_proposal_shared_projectable_tracked_observation_reprojection_rmse_px {
        scalars.push((
            PATH_HEALTH_VIO_PROPOSAL_SHARED_TRACKED_REPROJECTION_RMSE,
            v.value_px() as f64,
        ));
    }
    if let Some(v) = diag.pnp_inlier_reprojection_rmse_px {
        scalars.push((
            PATH_HEALTH_PNP_INLIER_REPROJECTION_RMSE,
            v.value_px() as f64,
        ));
    }
    if let Some(v) = diag.pnp_inlier_reprojection_max_px {
        scalars.push((PATH_HEALTH_PNP_INLIER_REPROJECTION_MAX, v.value_px() as f64));
    }
    if let Some(v) = diag.pnp_inlier_reprojection_mse_per_axis_px2 {
        scalars.push((
            PATH_HEALTH_PNP_INLIER_REPROJECTION_MSE_PER_AXIS,
            v.value_px2(),
        ));
    }
    if let Some(v) = diag.features_detected {
        scalars.push((PATH_TRACKING_FEATURES_DETECTED, v as f64));
    }
    if let Some(v) = diag.features_matched {
        scalars.push((PATH_TRACKING_FEATURES_MATCHED, v as f64));
    }
    if let Some(v) = diag.pnp_tracked_observations {
        scalars.push((PATH_TRACKING_PNP_TRACKED_OBSERVATIONS, v.count() as f64));
    }
    if let Some(v) = diag.pnp_accepted_inliers {
        scalars.push((PATH_TRACKING_PNP_ACCEPTED_INLIERS, v.count() as f64));
    }
    if let Some(v) = diag.visual_proposal_projectable_accepted_inliers {
        scalars.push((
            PATH_TRACKING_VISUAL_PROPOSAL_PROJECTABLE_ACCEPTED_INLIERS,
            v.count() as f64,
        ));
    }
    if let Some(v) = diag.vio_proposal_projectable_accepted_inliers {
        scalars.push((
            PATH_TRACKING_VIO_PROPOSAL_PROJECTABLE_ACCEPTED_INLIERS,
            v.count() as f64,
        ));
    }
    if let Some(v) = diag.shared_projectable_accepted_inliers {
        scalars.push((
            PATH_TRACKING_SHARED_PROJECTABLE_ACCEPTED_INLIERS,
            v.count() as f64,
        ));
    }
    if let Some(v) = diag.pnp_projectable_tracked_observations {
        scalars.push((
            PATH_TRACKING_PNP_PROJECTABLE_TRACKED_OBSERVATIONS,
            v.count() as f64,
        ));
    }
    if let Some(v) = diag.visual_proposal_projectable_tracked_observations {
        scalars.push((
            PATH_TRACKING_VISUAL_PROPOSAL_PNP_PROJECTABLE_TRACKED_OBSERVATIONS,
            v.count() as f64,
        ));
    }
    if let Some(v) = diag.vio_proposal_projectable_tracked_observations {
        scalars.push((
            PATH_TRACKING_VIO_PROPOSAL_PNP_PROJECTABLE_TRACKED_OBSERVATIONS,
            v.count() as f64,
        ));
    }
    if let Some(v) = diag.shared_projectable_tracked_observations {
        scalars.push((
            PATH_TRACKING_SHARED_PROJECTABLE_TRACKED_OBSERVATIONS,
            v.count() as f64,
        ));
    }
    if let Some(v) = diag.tracking_pose_source {
        scalars.push((PATH_TRACKING_POSE_SOURCE, tracking_pose_source_scalar(v)));
    }
    if let Some(v) = diag.vio_proposal_disposition {
        scalars.push((
            PATH_TRACKING_VIO_PROPOSAL_RAN,
            if v == crate::VioProposalDisposition::NotRun {
                0.0
            } else {
                1.0
            },
        ));
        scalars.push((
            PATH_TRACKING_VIO_PROPOSAL_ADOPTED,
            if v == crate::VioProposalDisposition::Adopted {
                1.0
            } else {
                0.0
            },
        ));
        scalars.push((
            PATH_TRACKING_VIO_PROPOSAL_REJECTED_UNUSABLE_SOLVE,
            if v == crate::VioProposalDisposition::RejectedUnusableSolve {
                1.0
            } else {
                0.0
            },
        ));
        scalars.push((
            PATH_TRACKING_VIO_PROPOSAL_REJECTED_INSUFFICIENT_CURRENT_VIO_SUPPORT,
            if v == crate::VioProposalDisposition::RejectedInsufficientCurrentVioObservationSupport
            {
                1.0
            } else {
                0.0
            },
        ));
        scalars.push((
            PATH_TRACKING_VIO_PROPOSAL_REJECTED_INSUFFICIENT_SHARED_SUPPORT,
            if v == crate::VioProposalDisposition::RejectedInsufficientSharedAcceptedInlierSupport {
                1.0
            } else {
                0.0
            },
        ));
        scalars.push((
            PATH_TRACKING_VIO_PROPOSAL_REJECTED_CHANGED_SUPPORT,
            if v == crate::VioProposalDisposition::RejectedChangedAcceptedInlierProjectability {
                1.0
            } else {
                0.0
            },
        ));
        scalars.push((
            PATH_TRACKING_VIO_PROPOSAL_REJECTED_HIGHER_SHARED_RMSE,
            if v
                == crate::VioProposalDisposition::RejectedHigherSharedAcceptedInlierReprojectionRmse
            {
                1.0
            } else {
                0.0
            },
        ));
    }
    if let Some(v) = diag.ransac_iterations {
        scalars.push((PATH_TRACKING_RANSAC_ITERATIONS, v as f64));
    }
    if let Some(v) = diag.pnp_ransac_candidate_projection_rejections {
        scalars.push((
            PATH_TRACKING_PNP_RANSAC_CANDIDATE_PROJECTION_REJECTIONS,
            v as f64,
        ));
    }
    if let Some(refinement) = diag.pnp_refinement.as_ref() {
        scalars.push((
            PATH_TRACKING_PNP_REFINEMENT_APPLIED,
            if refinement.applied() { 1.0 } else { 0.0 },
        ));
        if let Some(iterations) = refinement.iterations() {
            scalars.push((
                PATH_TRACKING_PNP_REFINEMENT_ITERATIONS,
                iterations.get() as f64,
            ));
        }
    }
    if let Some(v) = diag.parallax_px {
        scalars.push((PATH_TRACKING_PARALLAX, v as f64));
    }
    if let Some(v) = diag.covisibility {
        scalars.push((PATH_TRACKING_COVISIBILITY, v as f64));
    }
    if let Some(v) = diag.tracking_time {
        scalars.push((PATH_TIMING_TRACKING_MS, v.as_secs_f64() * 1_000.0));
    }

    if let Some(stats) = diag.triangulation {
        scalars.push((PATH_TRI_CANDIDATES, stats.candidate_matches as f64));
        scalars.push((PATH_TRI_KEPT, stats.kept as f64));
        scalars.push((PATH_TRI_DROPPED_DISPARITY, stats.dropped_disparity as f64));
        scalars.push((PATH_TRI_DROPPED_DEPTH, stats.dropped_depth as f64));
        scalars.push((PATH_TRI_DROPPED_NUMERICAL, stats.dropped_numerical as f64));
        scalars.push((PATH_TRI_DROPPED_DUPLICATE, stats.dropped_duplicate as f64));
    }

    if let Some(ba_result) = diag.ba_result.as_ref() {
        match ba_result {
            BaResult::Optimized(optimization) => {
                scalars.push((
                    PATH_BA_ITERATIONS,
                    optimization.termination().iterations().get() as f64,
                ));
                scalars.push((
                    PATH_BA_ACCEPTED_STEPS,
                    optimization.accepted_steps().get() as f64,
                ));
                scalars.push((PATH_BA_FINAL_COST, optimization.final_cost().get()));
                scalars.push((PATH_BA_STALLED, 0.0));
                scalars.push((PATH_BA_STATIONARY, 0.0));
            }
            BaResult::Stationary(stationary) => {
                scalars.push((
                    PATH_BA_ITERATIONS,
                    stationary.detected_at_iteration().get() as f64,
                ));
                scalars.push((PATH_BA_ACCEPTED_STEPS, 0.0));
                scalars.push((PATH_BA_FINAL_COST, stationary.retained_cost().get()));
                scalars.push((PATH_BA_STALLED, 0.0));
                scalars.push((PATH_BA_STATIONARY, 1.0));
            }
            BaResult::Stalled(stall) => {
                scalars.push((
                    PATH_BA_ITERATIONS,
                    stall.attempted_iterations().get() as f64,
                ));
                scalars.push((PATH_BA_ACCEPTED_STEPS, 0.0));
                scalars.push((PATH_BA_FINAL_COST, stall.retained_cost().get()));
                scalars.push((PATH_BA_STALLED, 1.0));
                scalars.push((PATH_BA_STATIONARY, 0.0));
            }
            BaResult::Degenerate { .. } => {}
        }
    }
    if let Some(termination) = diag.pose_ba_termination {
        let (iterations, converged) = match termination {
            crate::PoseBaTermination::Converged { iterations } => (iterations, 1.0),
            crate::PoseBaTermination::IterationLimit { iterations } => (iterations, 0.0),
        };
        scalars.push((PATH_POSE_BA_ITERATIONS, iterations.get() as f64));
        scalars.push((PATH_POSE_BA_CONVERGED, converged));
    }

    #[cfg(feature = "vio")]
    if let Some(vio_result) = diag.vio_solve_result.as_ref() {
        scalars.push((
            PATH_VIO_ATTEMPTED_ITERATIONS,
            vio_result.attempted_iterations() as f64,
        ));
        scalars.push((
            PATH_VIO_FINAL_MIXED_OBJECTIVE,
            vio_result.final_mixed_objective(),
        ));
        scalars.push((
            PATH_VIO_CONVERGED,
            if vio_result.termination().is_converged() {
                1.0
            } else {
                0.0
            },
        ));
        let termination = match vio_result.termination() {
            crate::VioSolveTermination::NotRequired => 0.0,
            crate::VioSolveTermination::Converged { .. } => 1.0,
            crate::VioSolveTermination::IterationLimit => 2.0,
            crate::VioSolveTermination::StalledNoObjectiveImprovement => 3.0,
        };
        scalars.push((PATH_VIO_TERMINATION, termination));
        scalars.push((PATH_VIO_ACCEPTED_STEPS, vio_result.accepted_steps() as f64));
        scalars.push((PATH_VIO_REJECTED_STEPS, vio_result.rejected_steps() as f64));
        scalars.push((
            PATH_VIO_REJECTED_NONPROJECTABLE_CANDIDATE_STEPS,
            vio_result.rejected_nonprojectable_candidate_steps() as f64,
        ));
        scalars.push((
            PATH_VIO_LAST_FRAME_ACTIVE_VISUAL_FACTORS,
            vio_result.last_frame_active_visual_factor_count() as f64,
        ));
        scalars.push((
            PATH_VIO_INITIALLY_EXCLUDED_NONPROJECTABLE_VISUAL_FACTORS,
            vio_result.initially_excluded_nonprojectable_visual_factor_count() as f64,
        ));
        scalars.push((
            PATH_VIO_REGULARIZED_IMU_RESIDUAL_FACTORS,
            vio_result.regularized_imu_residual_factor_count() as f64,
        ));
        scalars.push((
            PATH_VIO_FLOORED_ACCEL_BIAS_RANDOM_WALK_FACTORS,
            vio_result.floored_accel_bias_random_walk_factor_count() as f64,
        ));
        scalars.push((
            PATH_VIO_FLOORED_GYRO_BIAS_RANDOM_WALK_FACTORS,
            vio_result.floored_gyro_bias_random_walk_factor_count() as f64,
        ));
        let objective = vio_result.objective_breakdown();
        scalars.push((
            PATH_VIO_OBJECTIVE_REPROJECTION_ROBUST_PX2,
            objective.reprojection_robust_px2(),
        ));
        scalars.push((
            PATH_VIO_OBJECTIVE_IMU_MAHALANOBIS,
            objective.imu_mahalanobis(),
        ));
        scalars.push((
            PATH_VIO_OBJECTIVE_BIAS_RANDOM_WALK_MAHALANOBIS,
            objective.bias_random_walk_mahalanobis(),
        ));
        scalars.push((
            PATH_VIO_OBJECTIVE_VELOCITY_ANCHOR_MAHALANOBIS,
            objective.velocity_anchor_mahalanobis(),
        ));
        scalars.push((
            PATH_VIO_OBJECTIVE_BIAS_PRIOR_MAHALANOBIS,
            objective.bias_prior_mahalanobis(),
        ));
    }
    #[cfg(feature = "vio")]
    if let Some(active) = diag.vio_calibrated_bias_prior_active {
        scalars.push((
            PATH_VIO_CALIBRATED_BIAS_PRIOR_ACTIVE,
            if active { 1.0 } else { 0.0 },
        ));
    }

    scalars
}

fn format_event(event: &DiagnosticEvent) -> (String, &'static str) {
    match event {
        DiagnosticEvent::TrackingLost {
            consecutive_failures,
        } => (
            format!("tracking lost after {consecutive_failures} consecutive failures"),
            rerun::TextLogLevel::WARN,
        ),
        DiagnosticEvent::TrackingRecovered => {
            ("tracking recovered".to_string(), rerun::TextLogLevel::INFO)
        }
        DiagnosticEvent::ProjectedTrackingFallback { reason } => (
            format!("projected tracking fell back to LightGlue: {reason}"),
            rerun::TextLogLevel::WARN,
        ),
        DiagnosticEvent::KeyframeCreated {
            keyframe_id,
            landmarks,
        } => (
            format!("keyframe {keyframe_id:?} created with {landmarks} landmarks"),
            rerun::TextLogLevel::INFO,
        ),
        DiagnosticEvent::KeyframeRemoved {
            keyframe_id,
            reason,
        } => (
            format!("keyframe {keyframe_id:?} removed ({reason:?})"),
            rerun::TextLogLevel::INFO,
        ),
        DiagnosticEvent::LoopClosureDetected {
            query,
            match_kf,
            cosine_similarity,
        } => (
            format!(
                "loop closure applied: query={query:?}, match={match_kf:?}, cosine_similarity={:.3}",
                cosine_similarity.value()
            ),
            rerun::TextLogLevel::INFO,
        ),
        DiagnosticEvent::LoopDescriptorMatchDegraded {
            candidate_keyframe,
            zero_norm_query_descriptors,
            zero_norm_candidate_descriptors,
        } => (
            format!(
                "loop descriptor matching skipped undefined cosine comparisons (candidate={candidate_keyframe:?}, zero_norm_query_descriptors={zero_norm_query_descriptors}, zero_norm_candidate_descriptors={zero_norm_candidate_descriptors})"
            ),
            rerun::TextLogLevel::WARN,
        ),
        DiagnosticEvent::LoopClosureRejected { reason } => {
            let reason_text = match reason {
                LoopClosureRejectReason::TooFewCorrespondences { count } => {
                    format!("too few correspondences ({count})")
                }
                LoopClosureRejectReason::VerificationFailed => "verification failed".to_string(),
                LoopClosureRejectReason::CorrectionEvaluationFailed => {
                    "correction evaluation failed".to_string()
                }
                LoopClosureRejectReason::CorrectionTooLarge {
                    translation_m,
                    rotation_deg,
                } => format!(
                    "correction too large (translation={translation_m:.3}m, rotation={rotation_deg:.2}deg)"
                ),
                LoopClosureRejectReason::ApplyFailed => "apply failed".to_string(),
            };
            (
                format!("loop closure rejected: {reason_text}"),
                rerun::TextLogLevel::WARN,
            )
        }
        DiagnosticEvent::BackendWorkerDied {
            respawn_count,
            message,
        } => (
            format!("backend worker died (respawns={respawn_count}): {message}"),
            rerun::TextLogLevel::ERROR,
        ),
        DiagnosticEvent::BackendWorkerRestartFailed {
            respawn_count,
            max_respawns,
            exhausted,
            error,
        } => (
            format!(
                "backend worker restart failed (attempt={respawn_count}/{max_respawns}, exhausted={exhausted}): {error}"
            ),
            rerun::TextLogLevel::ERROR,
        ),
        DiagnosticEvent::DescriptorWorkerDied { respawn_count } => (
            format!("descriptor worker died (respawns={respawn_count})"),
            rerun::TextLogLevel::ERROR,
        ),
        DiagnosticEvent::DescriptorWorkerRestartFailed {
            respawn_count,
            max_respawns,
            exhausted,
            error,
        } => (
            format!(
                "descriptor worker restart failed (attempt={respawn_count}/{max_respawns}, exhausted={exhausted}): {error}"
            ),
            rerun::TextLogLevel::ERROR,
        ),
        DiagnosticEvent::DescriptorInferenceFailed {
            keyframe_id,
            source_snapshot,
            error,
        } => (
            format!(
                "descriptor inference failed (keyframe={keyframe_id:?}, snapshot={source_snapshot}): {error}"
            ),
            rerun::TextLogLevel::ERROR,
        ),
        DiagnosticEvent::BootstrapDescriptorUnavailable {
            keyframe_id,
            source_snapshot,
            error,
        } => (
            format!(
                "bootstrap loop descriptor unavailable (keyframe={keyframe_id:?}, snapshot={source_snapshot}): {error}"
            ),
            rerun::TextLogLevel::WARN,
        ),
        DiagnosticEvent::DescriptorIndexFailed {
            keyframe_id,
            source_snapshot,
            error,
        } => (
            format!(
                "learned descriptor index update failed (keyframe={keyframe_id:?}, snapshot={source_snapshot}): {error}"
            ),
            rerun::TextLogLevel::ERROR,
        ),
        DiagnosticEvent::RelocalizationStarted => (
            "relocalization started".to_string(),
            rerun::TextLogLevel::WARN,
        ),
        DiagnosticEvent::RelocalizationSucceeded { keyframe_id } => (
            format!("relocalization succeeded against keyframe {keyframe_id:?}"),
            rerun::TextLogLevel::INFO,
        ),
        DiagnosticEvent::BaDegenerate { reason } => (
            format!("backend BA degenerate: {reason:?}"),
            rerun::TextLogLevel::WARN,
        ),
        DiagnosticEvent::BaStalled {
            attempted_iterations,
        } => (
            format!(
                "backend BA stalled without an accepted step after {attempted_iterations} iterations"
            ),
            rerun::TextLogLevel::WARN,
        ),
    }
}

fn tracking_state_scalar(state: TrackingHealth) -> f64 {
    match state {
        TrackingHealth::Good => 0.0,
        TrackingHealth::Degraded => 1.0,
        TrackingHealth::Lost => 2.0,
    }
}

fn degradation_scalar(level: DegradationLevel) -> f64 {
    match level {
        DegradationLevel::Nominal => 0.0,
        DegradationLevel::TrackingDegraded => 1.0,
        DegradationLevel::DescriptorDown => 2.0,
        DegradationLevel::BackendDown => 3.0,
        DegradationLevel::Lost => 4.0,
    }
}

fn component_state_scalar(state: ComponentHealth) -> f64 {
    match state {
        ComponentHealth::Disabled => 0.0,
        ComponentHealth::Alive => 1.0,
        ComponentHealth::Down => 2.0,
    }
}

fn tracking_pose_source_scalar(source: crate::TrackingPoseSource) -> f64 {
    match source {
        crate::TrackingPoseSource::VisualTracking => 0.0,
        crate::TrackingPoseSource::VisualBundleAdjustment => 1.0,
        crate::TrackingPoseSource::VioRefined => 2.0,
    }
}

impl RerunSink {
    /// Recommended Rerun dashboard:
    /// 1) `diagnostics/health/*` and `diagnostics/timing/*` as top time-series plots.
    /// 2) `diagnostics/map/*` and `diagnostics/loop/*` for operational state.
    /// 3) `diagnostics/events/log` as an always-visible text log panel.
    pub fn log_diagnostics(
        &self,
        timestamp: Timestamp,
        diagnostics: &FrameDiagnostics,
    ) -> Result<(), VizLogError> {
        let rec = self.recording();
        set_capture_time(rec, timestamp);
        for (path, value) in diagnostics_scalars(diagnostics) {
            rec.log(path, &rerun::Scalars::single(value))?;
        }
        Ok(())
    }

    pub fn log_event(
        &self,
        timestamp: Timestamp,
        event: &DiagnosticEvent,
    ) -> Result<(), VizLogError> {
        let rec = self.recording();
        set_capture_time(rec, timestamp);
        let (message, level) = format_event(event);
        let text = rerun::TextLog::new(message).with_level(level);
        rec.log(PATH_EVENTS_LOG, &text)?;
        Ok(())
    }

    pub fn log_system_health(
        &self,
        timestamp: Timestamp,
        health: &SystemHealth,
    ) -> Result<(), VizLogError> {
        let rec = self.recording();
        set_capture_time(rec, timestamp);
        rec.log(
            PATH_HEALTH_TRACKING_STATE,
            &rerun::Scalars::single(tracking_state_scalar(health.tracking)),
        )?;
        rec.log(
            PATH_HEALTH_DEGRADATION_LEVEL,
            &rerun::Scalars::single(degradation_scalar(health.degradation)),
        )?;
        rec.log(
            PATH_HEALTH_BACKEND_STATE,
            &rerun::Scalars::single(component_state_scalar(health.backend)),
        )?;
        rec.log(
            PATH_HEALTH_DESCRIPTOR_STATE,
            &rerun::Scalars::single(component_state_scalar(health.descriptor)),
        )?;
        rec.log(
            PATH_HEALTH_BACKEND_ALIVE,
            &rerun::Scalars::single(if health.backend.is_alive() { 1.0 } else { 0.0 }),
        )?;
        rec.log(
            PATH_HEALTH_DESCRIPTOR_ALIVE,
            &rerun::Scalars::single(if health.descriptor.is_alive() {
                1.0
            } else {
                0.0
            }),
        )?;
        rec.log(
            PATH_HEALTH_BACKEND_SUBMITTED,
            &rerun::Scalars::single(health.backend_stats.submitted as f64),
        )?;
        rec.log(
            PATH_HEALTH_BACKEND_APPLIED,
            &rerun::Scalars::single(health.backend_stats.applied as f64),
        )?;
        rec.log(
            PATH_HEALTH_BACKEND_UNCHANGED,
            &rerun::Scalars::single(health.backend_stats.unchanged as f64),
        )?;
        rec.log(
            PATH_HEALTH_BACKEND_DROPPED_FULL,
            &rerun::Scalars::single(health.backend_stats.dropped_full as f64),
        )?;
        rec.log(
            PATH_HEALTH_BACKEND_DROPPED_UNAVAILABLE,
            &rerun::Scalars::single(health.backend_stats.dropped_unavailable as f64),
        )?;
        rec.log(
            PATH_HEALTH_BACKEND_STALE,
            &rerun::Scalars::single(health.backend_stats.stale as f64),
        )?;
        rec.log(
            PATH_HEALTH_BACKEND_REJECTED,
            &rerun::Scalars::single(health.backend_stats.rejected as f64),
        )?;
        rec.log(
            PATH_HEALTH_BACKEND_WORKER_FAILURES,
            &rerun::Scalars::single(health.backend_stats.worker_failures as f64),
        )?;
        rec.log(
            PATH_HEALTH_BACKEND_RESTART_FAILURES,
            &rerun::Scalars::single(health.backend_stats.restart_failures as f64),
        )?;
        rec.log(
            PATH_HEALTH_BACKEND_RESPAWN_COUNT,
            &rerun::Scalars::single(health.backend_stats.respawn_count as f64),
        )?;
        rec.log(
            PATH_HEALTH_BACKEND_RESPAWN_EXHAUSTED,
            &rerun::Scalars::single(if health.backend_stats.respawn_exhausted {
                1.0
            } else {
                0.0
            }),
        )?;
        rec.log(
            PATH_HEALTH_BACKEND_PANICS,
            &rerun::Scalars::single(health.backend_stats.panics as f64),
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        PATH_BA_ACCEPTED_STEPS, PATH_BA_FINAL_COST, PATH_BA_ITERATIONS, PATH_BA_STALLED,
        PATH_BA_STATIONARY, PATH_HEALTH_PNP_INLIER_RATIO,
        PATH_HEALTH_PNP_PROJECTABLE_TRACKED_REPROJECTION_MAX,
        PATH_HEALTH_PNP_PROJECTABLE_TRACKED_REPROJECTION_MSE_PER_AXIS,
        PATH_HEALTH_PNP_PROJECTABLE_TRACKED_REPROJECTION_RMSE,
        PATH_HEALTH_VIO_PROPOSAL_ACCEPTED_INLIER_REPROJECTION_RMSE,
        PATH_HEALTH_VIO_PROPOSAL_SHARED_ACCEPTED_INLIER_REPROJECTION_RMSE,
        PATH_HEALTH_VISUAL_PROPOSAL_ACCEPTED_INLIER_REPROJECTION_RMSE,
        PATH_HEALTH_VISUAL_PROPOSAL_SHARED_ACCEPTED_INLIER_REPROJECTION_RMSE, PATH_MAP_KEYFRAMES,
        PATH_MAP_POINTS, PATH_POSE_BA_CONVERGED, PATH_POSE_BA_ITERATIONS,
        PATH_TRACKING_PNP_ACCEPTED_INLIERS, PATH_TRACKING_PNP_PROJECTABLE_TRACKED_OBSERVATIONS,
        PATH_TRACKING_PNP_RANSAC_CANDIDATE_PROJECTION_REJECTIONS,
        PATH_TRACKING_PNP_REFINEMENT_APPLIED, PATH_TRACKING_PNP_REFINEMENT_ITERATIONS,
        PATH_TRACKING_SHARED_PROJECTABLE_ACCEPTED_INLIERS,
        PATH_TRACKING_SHARED_PROJECTABLE_TRACKED_OBSERVATIONS, PATH_TRACKING_VIO_PROPOSAL_ADOPTED,
        PATH_TRACKING_VIO_PROPOSAL_PNP_PROJECTABLE_TRACKED_OBSERVATIONS,
        PATH_TRACKING_VIO_PROPOSAL_PROJECTABLE_ACCEPTED_INLIERS,
        PATH_TRACKING_VISUAL_PROPOSAL_PROJECTABLE_ACCEPTED_INLIERS, diagnostics_scalars,
        format_event,
    };
    #[cfg(feature = "vio")]
    use super::{
        PATH_VIO_ACCEPTED_STEPS, PATH_VIO_ATTEMPTED_ITERATIONS,
        PATH_VIO_CALIBRATED_BIAS_PRIOR_ACTIVE, PATH_VIO_FINAL_MIXED_OBJECTIVE,
        PATH_VIO_FLOORED_ACCEL_BIAS_RANDOM_WALK_FACTORS,
        PATH_VIO_FLOORED_GYRO_BIAS_RANDOM_WALK_FACTORS,
        PATH_VIO_INITIALLY_EXCLUDED_NONPROJECTABLE_VISUAL_FACTORS,
        PATH_VIO_LAST_FRAME_ACTIVE_VISUAL_FACTORS, PATH_VIO_OBJECTIVE_BIAS_PRIOR_MAHALANOBIS,
        PATH_VIO_OBJECTIVE_BIAS_RANDOM_WALK_MAHALANOBIS, PATH_VIO_OBJECTIVE_IMU_MAHALANOBIS,
        PATH_VIO_OBJECTIVE_REPROJECTION_ROBUST_PX2, PATH_VIO_OBJECTIVE_VELOCITY_ANCHOR_MAHALANOBIS,
        PATH_VIO_REGULARIZED_IMU_RESIDUAL_FACTORS,
        PATH_VIO_REJECTED_NONPROJECTABLE_CANDIDATE_STEPS, PATH_VIO_REJECTED_STEPS,
        PATH_VIO_TERMINATION,
    };
    #[cfg(feature = "vio")]
    use crate::local_ba::VioFactorDiagnostics;
    use crate::{
        DiagnosticEvent, FrameDiagnostics, KeyframeRemovalReason, LoopClosureRejectReason,
        PnpAcceptedInlierCountMetric, PnpAcceptedInlierPixelResidualMetric, PnpInlierRatioMetric,
        PnpProjectableTrackedObservationCountMetric,
        PnpProjectableTrackedObservationPixelResidualMetric,
        PnpProjectableTrackedObservationReprojectionMsePerAxisPx2Metric, TrackingPoseSource,
        TriangulationStats, VioProposalDisposition,
        VioProposalProjectableTrackedObservationCountMetric,
        VioProposalProjectableTrackedObservationPixelResidualMetric,
        VisualProposalProjectableTrackedObservationCountMetric,
        VisualProposalProjectableTrackedObservationPixelResidualMetric,
        VisualVsVioSharedProjectableTrackedObservationCountMetric,
        VisualVsVioSharedProjectableTrackedObservationPixelResidualMetric,
    };
    #[cfg(feature = "vio")]
    use crate::{VioObjectiveBreakdown, VioSolveResult, VioSolveTermination};

    #[test]
    fn diagnostics_scalars_empty_has_baselines() {
        let diag = FrameDiagnostics::empty(5, 13);
        let scalars = diagnostics_scalars(&diag);
        assert!(
            scalars
                .iter()
                .any(|(path, value)| *path == PATH_MAP_KEYFRAMES && *value == 5.0)
        );
        assert!(
            scalars
                .iter()
                .any(|(path, value)| *path == PATH_MAP_POINTS && *value == 13.0)
        );
    }

    #[test]
    fn diagnostics_scalars_distinguish_optimized_stationary_and_stalled_ba() {
        let iterations = std::num::NonZeroUsize::new(3).expect("nonzero");
        let accepted_steps = std::num::NonZeroUsize::new(2).expect("nonzero");
        let optimized = crate::BaResult::Optimized(
            crate::BaOptimization::new(
                crate::BaTermination::IterationLimit { iterations },
                accepted_steps,
                crate::BaCost::new(1.5).expect("valid cost"),
            )
            .expect("valid result"),
        );
        let stationary = crate::BaResult::Stationary(crate::BaStationary::new(
            std::num::NonZeroUsize::MIN,
            crate::BaCost::new(2.5).expect("valid cost"),
        ));
        let stalled = crate::BaResult::Stalled(crate::BaStall::new(
            iterations,
            crate::BaCost::new(3.5).expect("valid cost"),
        ));

        for (result, expected) in [
            (optimized, (3.0, 2.0, 1.5, 0.0, 0.0)),
            (stationary, (1.0, 0.0, 2.5, 0.0, 1.0)),
            (stalled, (3.0, 0.0, 3.5, 1.0, 0.0)),
        ] {
            let mut diagnostics = FrameDiagnostics::empty(0, 0);
            diagnostics.ba_result = Some(result);
            let scalars = diagnostics_scalars(&diagnostics);
            let value = |path| {
                scalars
                    .iter()
                    .find_map(|(actual, value)| (*actual == path).then_some(*value))
                    .expect("BA scalar")
            };
            assert_eq!(value(PATH_BA_ITERATIONS), expected.0);
            assert_eq!(value(PATH_BA_ACCEPTED_STEPS), expected.1);
            assert_eq!(value(PATH_BA_FINAL_COST), expected.2);
            assert_eq!(value(PATH_BA_STALLED), expected.3);
            assert_eq!(value(PATH_BA_STATIONARY), expected.4);
        }
    }

    #[test]
    fn diagnostics_scalars_include_present_fields() {
        let mut diag = FrameDiagnostics::empty(1, 2);
        diag.pnp_inlier_ratio = Some(
            PnpInlierRatioMetric::new(
                PnpAcceptedInlierCountMetric::new(3),
                crate::PnpTrackedObservationCountMetric::new(4),
            )
            .expect("ratio"),
        );
        diag.pnp_accepted_inliers = Some(PnpAcceptedInlierCountMetric::new(6));
        diag.pnp_ransac_candidate_projection_rejections = Some(2);
        diag.pnp_refinement = Some(crate::PnpRefinementStatus::Applied {
            termination: crate::PnpRefinementTermination::Converged {
                iterations: std::num::NonZeroUsize::new(2).expect("literal is non-zero"),
            },
        });
        diag.pnp_projectable_tracked_observations =
            Some(PnpProjectableTrackedObservationCountMetric::new(7));
        diag.pnp_projectable_tracked_observation_reprojection_rmse_px = Some(
            PnpProjectableTrackedObservationPixelResidualMetric::new(1.5).expect("tracked rmse"),
        );
        diag.pnp_projectable_tracked_observation_reprojection_max_px = Some(
            PnpProjectableTrackedObservationPixelResidualMetric::new(3.0).expect("tracked max"),
        );
        diag.pnp_projectable_tracked_observation_reprojection_mse_per_axis_px2 = Some(
            PnpProjectableTrackedObservationReprojectionMsePerAxisPx2Metric::new(1.125)
                .expect("tracked mse"),
        );
        diag.tracking_pose_source = Some(TrackingPoseSource::VioRefined);
        diag.visual_proposal_projectable_tracked_observations = Some(
            VisualProposalProjectableTrackedObservationCountMetric::new(7),
        );
        diag.visual_proposal_projectable_tracked_observation_reprojection_rmse_px = Some(
            VisualProposalProjectableTrackedObservationPixelResidualMetric::new(1.2)
                .expect("visual proposal rmse"),
        );
        diag.visual_proposal_projectable_accepted_inliers =
            Some(PnpAcceptedInlierCountMetric::new(6));
        diag.visual_proposal_accepted_inlier_reprojection_rmse_px = Some(
            PnpAcceptedInlierPixelResidualMetric::new(1.0)
                .expect("visual proposal accepted-inlier rmse"),
        );
        diag.vio_proposal_projectable_tracked_observations =
            Some(VioProposalProjectableTrackedObservationCountMetric::new(8));
        diag.vio_proposal_projectable_tracked_observation_reprojection_rmse_px = Some(
            VioProposalProjectableTrackedObservationPixelResidualMetric::new(0.9)
                .expect("vio proposal rmse"),
        );
        diag.vio_proposal_projectable_accepted_inliers = Some(PnpAcceptedInlierCountMetric::new(6));
        diag.vio_proposal_accepted_inlier_reprojection_rmse_px = Some(
            PnpAcceptedInlierPixelResidualMetric::new(0.8)
                .expect("vio proposal accepted-inlier rmse"),
        );
        diag.shared_projectable_tracked_observations =
            Some(VisualVsVioSharedProjectableTrackedObservationCountMetric::new(6));
        diag.visual_proposal_shared_projectable_tracked_observation_reprojection_rmse_px = Some(
            VisualVsVioSharedProjectableTrackedObservationPixelResidualMetric::new(1.1)
                .expect("visual shared rmse"),
        );
        diag.vio_proposal_shared_projectable_tracked_observation_reprojection_rmse_px = Some(
            VisualVsVioSharedProjectableTrackedObservationPixelResidualMetric::new(0.8)
                .expect("vio shared rmse"),
        );
        diag.shared_projectable_accepted_inliers = Some(PnpAcceptedInlierCountMetric::new(6));
        diag.visual_proposal_shared_accepted_inlier_reprojection_rmse_px = Some(
            PnpAcceptedInlierPixelResidualMetric::new(1.0)
                .expect("visual shared accepted-inlier rmse"),
        );
        diag.vio_proposal_shared_accepted_inlier_reprojection_rmse_px = Some(
            PnpAcceptedInlierPixelResidualMetric::new(0.8)
                .expect("vio shared accepted-inlier rmse"),
        );
        diag.vio_proposal_disposition = Some(VioProposalDisposition::Adopted);
        diag.pose_ba_termination = Some(crate::PoseBaTermination::IterationLimit {
            iterations: std::num::NonZeroUsize::new(7).expect("literal is non-zero"),
        });
        #[cfg(feature = "vio")]
        {
            diag.vio_calibrated_bias_prior_active = Some(true);
            diag.vio_solve_result = Some(
                VioSolveResult::try_evaluated(
                    VioSolveTermination::IterationLimit,
                    2,
                    1,
                    1,
                    VioObjectiveBreakdown::new(4.0, 0.0, 0.0, 0.0, 0.0).expect("valid objective"),
                    VioFactorDiagnostics {
                        last_frame_active_visual_factor_count: 6,
                        initially_excluded_nonprojectable_visual_factor_count: 2,
                        regularized_imu_residual_factor_count: 2,
                        floored_accel_bias_random_walk_factor_count: 1,
                        floored_gyro_bias_random_walk_factor_count: 2,
                    },
                )
                .expect("valid VIO result"),
            );
        }
        diag.depth_reorder_warnings = Some(3);
        diag.features_detected = Some(400);
        diag.triangulation = Some(TriangulationStats {
            candidate_matches: 10,
            kept: 8,
            dropped_disparity: 1,
            dropped_depth: 1,
            dropped_numerical: 0,
            dropped_duplicate: 0,
        });
        let scalars = diagnostics_scalars(&diag);
        assert!(scalars.iter().any(|(path, value)| {
            *path == PATH_POSE_BA_ITERATIONS && (*value - 7.0).abs() < f64::EPSILON
        }));
        assert!(scalars.iter().any(|(path, value)| {
            *path == PATH_POSE_BA_CONVERGED && value.abs() < f64::EPSILON
        }));
        assert!(scalars.iter().any(|(path, value)| {
            *path == PATH_TRACKING_PNP_RANSAC_CANDIDATE_PROJECTION_REJECTIONS
                && (*value - 2.0).abs() < f64::EPSILON
        }));
        #[cfg(feature = "vio")]
        assert!(scalars.iter().any(|(path, value)| {
            *path == PATH_VIO_ATTEMPTED_ITERATIONS && (*value - 3.0).abs() < f64::EPSILON
        }));
        #[cfg(feature = "vio")]
        assert!(scalars.iter().any(|(path, value)| {
            *path == PATH_VIO_FINAL_MIXED_OBJECTIVE && (*value - 4.0).abs() < f64::EPSILON
        }));
        #[cfg(feature = "vio")]
        for (expected_path, expected_value) in [
            (PATH_VIO_OBJECTIVE_REPROJECTION_ROBUST_PX2, 4.0),
            (PATH_VIO_OBJECTIVE_IMU_MAHALANOBIS, 0.0),
            (PATH_VIO_OBJECTIVE_BIAS_RANDOM_WALK_MAHALANOBIS, 0.0),
            (PATH_VIO_OBJECTIVE_VELOCITY_ANCHOR_MAHALANOBIS, 0.0),
            (PATH_VIO_OBJECTIVE_BIAS_PRIOR_MAHALANOBIS, 0.0),
        ] {
            assert!(scalars.iter().any(|(path, value)| {
                *path == expected_path && (*value - expected_value).abs() < f64::EPSILON
            }));
        }
        #[cfg(feature = "vio")]
        assert!(scalars.iter().all(|(path, _)| !matches!(
            *path,
            "diagnostics/vio/iterations"
                | "diagnostics/vio/final_cost"
                | "diagnostics/vio/cost/reprojection"
                | "diagnostics/vio/cost/imu"
                | "diagnostics/vio/cost/bias_random_walk"
                | "diagnostics/vio/cost/velocity_anchor"
                | "diagnostics/vio/cost/bias_prior"
        )));
        #[cfg(feature = "vio")]
        assert!(scalars.iter().any(|(path, value)| {
            *path == PATH_VIO_REJECTED_NONPROJECTABLE_CANDIDATE_STEPS
                && (*value - 1.0).abs() < f64::EPSILON
        }));
        #[cfg(feature = "vio")]
        assert!(scalars.iter().any(|(path, value)| {
            *path == PATH_VIO_LAST_FRAME_ACTIVE_VISUAL_FACTORS
                && (*value - 6.0).abs() < f64::EPSILON
        }));
        #[cfg(feature = "vio")]
        assert!(scalars.iter().any(|(path, value)| {
            *path == PATH_VIO_INITIALLY_EXCLUDED_NONPROJECTABLE_VISUAL_FACTORS
                && (*value - 2.0).abs() < f64::EPSILON
        }));
        #[cfg(feature = "vio")]
        assert!(scalars.iter().any(|(path, value)| {
            *path == PATH_VIO_REGULARIZED_IMU_RESIDUAL_FACTORS
                && (*value - 2.0).abs() < f64::EPSILON
        }));
        #[cfg(feature = "vio")]
        assert!(scalars.iter().any(|(path, value)| {
            *path == PATH_VIO_FLOORED_ACCEL_BIAS_RANDOM_WALK_FACTORS
                && (*value - 1.0).abs() < f64::EPSILON
        }));
        #[cfg(feature = "vio")]
        assert!(scalars.iter().any(|(path, value)| {
            *path == PATH_VIO_FLOORED_GYRO_BIAS_RANDOM_WALK_FACTORS
                && (*value - 2.0).abs() < f64::EPSILON
        }));
        assert!(scalars.iter().any(|(path, value)| {
            *path == PATH_TRACKING_PNP_REFINEMENT_APPLIED && (*value - 1.0).abs() < f64::EPSILON
        }));
        assert!(scalars.iter().any(|(path, value)| {
            *path == PATH_TRACKING_PNP_REFINEMENT_ITERATIONS && (*value - 2.0).abs() < f64::EPSILON
        }));
        assert!(
            scalars
                .iter()
                .any(|(path, value)| *path == PATH_HEALTH_PNP_INLIER_RATIO
                    && (*value - 0.75).abs() < 1e-6)
        );
        assert!(scalars.iter().any(|(path, value)| {
            *path == PATH_HEALTH_PNP_PROJECTABLE_TRACKED_REPROJECTION_RMSE
                && (*value - 1.5).abs() < 1e-6
        }));
        assert!(scalars.iter().any(|(path, value)| {
            *path == PATH_TRACKING_PNP_ACCEPTED_INLIERS && (*value - 6.0).abs() < 1e-6
        }));
        assert!(scalars.iter().any(|(path, value)| {
            *path == PATH_TRACKING_VISUAL_PROPOSAL_PROJECTABLE_ACCEPTED_INLIERS
                && (*value - 6.0).abs() < 1e-6
        }));
        assert!(scalars.iter().any(|(path, value)| {
            *path == PATH_TRACKING_VIO_PROPOSAL_PROJECTABLE_ACCEPTED_INLIERS
                && (*value - 6.0).abs() < 1e-6
        }));
        assert!(scalars.iter().any(|(path, value)| {
            *path == PATH_TRACKING_SHARED_PROJECTABLE_ACCEPTED_INLIERS
                && (*value - 6.0).abs() < 1e-6
        }));
        assert!(scalars.iter().any(|(path, value)| {
            *path == PATH_HEALTH_VISUAL_PROPOSAL_ACCEPTED_INLIER_REPROJECTION_RMSE
                && (*value - 1.0).abs() < 1e-6
        }));
        assert!(scalars.iter().any(|(path, value)| {
            *path == PATH_HEALTH_VIO_PROPOSAL_ACCEPTED_INLIER_REPROJECTION_RMSE
                && (*value - 0.8).abs() < 1e-6
        }));
        assert!(scalars.iter().any(|(path, value)| {
            *path == PATH_HEALTH_VISUAL_PROPOSAL_SHARED_ACCEPTED_INLIER_REPROJECTION_RMSE
                && (*value - 1.0).abs() < 1e-6
        }));
        assert!(scalars.iter().any(|(path, value)| {
            *path == PATH_HEALTH_VIO_PROPOSAL_SHARED_ACCEPTED_INLIER_REPROJECTION_RMSE
                && (*value - 0.8).abs() < 1e-6
        }));
        assert!(scalars.iter().any(|(path, value)| {
            *path == PATH_HEALTH_PNP_PROJECTABLE_TRACKED_REPROJECTION_MAX
                && (*value - 3.0).abs() < 1e-6
        }));
        assert!(scalars.iter().any(|(path, value)| {
            *path == PATH_HEALTH_PNP_PROJECTABLE_TRACKED_REPROJECTION_MSE_PER_AXIS
                && (*value - 1.125).abs() < 1e-9
        }));
        assert!(scalars.iter().any(|(path, value)| {
            *path == PATH_TRACKING_PNP_PROJECTABLE_TRACKED_OBSERVATIONS
                && (*value - 7.0).abs() < 1e-6
        }));
        assert!(scalars.iter().any(|(path, value)| {
            *path == PATH_TRACKING_VIO_PROPOSAL_PNP_PROJECTABLE_TRACKED_OBSERVATIONS
                && (*value - 8.0).abs() < 1e-6
        }));
        assert!(scalars.iter().any(|(path, value)| {
            *path == PATH_TRACKING_SHARED_PROJECTABLE_TRACKED_OBSERVATIONS
                && (*value - 6.0).abs() < 1e-6
        }));
        assert!(scalars.iter().any(|(path, value)| {
            *path == PATH_TRACKING_VIO_PROPOSAL_ADOPTED && (*value - 1.0).abs() < 1e-6
        }));
        #[cfg(feature = "vio")]
        assert!(scalars.iter().any(|(path, value)| {
            *path == PATH_VIO_CALIBRATED_BIAS_PRIOR_ACTIVE && (*value - 1.0).abs() < 1e-6
        }));
        #[cfg(feature = "vio")]
        assert!(
            scalars.iter().any(|(path, value)| {
                *path == PATH_VIO_TERMINATION && (*value - 2.0).abs() < 1e-6
            })
        );
        #[cfg(feature = "vio")]
        assert!(scalars.iter().any(|(path, value)| {
            *path == PATH_VIO_ACCEPTED_STEPS && (*value - 2.0).abs() < 1e-6
        }));
        #[cfg(feature = "vio")]
        assert!(scalars.iter().any(|(path, value)| {
            *path == PATH_VIO_REJECTED_STEPS && (*value - 1.0).abs() < 1e-6
        }));
        assert!(
            scalars
                .iter()
                .any(|(path, _)| *path == "diagnostics/tracking/features_detected")
        );
        assert!(scalars.iter().any(
            |(path, value)| *path == "diagnostics/depth/reorder_warnings"
                && (*value - 3.0).abs() < 1e-6
        ));
        assert!(
            scalars
                .iter()
                .any(|(path, _)| *path == "diagnostics/triangulation/candidates")
        );
    }

    #[test]
    fn format_event_maps_rejection_to_warn_text() {
        let (text, level) = format_event(&DiagnosticEvent::LoopClosureRejected {
            reason: LoopClosureRejectReason::TooFewCorrespondences { count: 3 },
        });
        assert!(text.contains("rejected"));
        assert_eq!(level, rerun::TextLogLevel::WARN);
    }

    #[test]
    fn format_event_maps_worker_death_to_error() {
        let (text, level) = format_event(&DiagnosticEvent::BackendWorkerDied {
            respawn_count: 2,
            message: "forced panic".to_string(),
        });
        assert!(text.contains("backend worker died"));
        assert!(text.contains("forced panic"));
        assert_eq!(level, rerun::TextLogLevel::ERROR);
    }

    #[test]
    fn format_event_exposes_bootstrap_descriptor_degradation() {
        let (text, level) = format_event(&DiagnosticEvent::BootstrapDescriptorUnavailable {
            keyframe_id: crate::map::KeyframeId::default(),
            source_snapshot: crate::map::SlamMap::new().snapshot(),
            error: std::sync::Arc::new(crate::BootstrapDescriptorError::Aggregation {
                source: crate::loop_closure::GlobalDescriptorError::ZeroNorm,
            }),
        });

        assert!(text.contains("bootstrap loop descriptor unavailable"));
        assert!(text.contains("norm must be > 0"));
        assert_eq!(level, rerun::TextLogLevel::WARN);
    }

    #[test]
    fn format_event_exposes_projected_tracking_fallback() {
        let (text, level) = format_event(&DiagnosticEvent::ProjectedTrackingFallback {
            reason: crate::ProjectedTrackingFallbackReason::TooFewInliers {
                matches: 30,
                verified: 28,
                observations: 24,
                inliers: 18,
                required_inliers: 24,
            },
        });

        assert!(text.contains("fell back to LightGlue"));
        assert!(text.contains("verified=28"));
        assert!(text.contains("inliers=18"));
        assert!(text.contains("required_inliers=24"));
        assert_eq!(level, rerun::TextLogLevel::WARN);
    }

    #[test]
    fn format_event_supports_all_variants() {
        let _ = format_event(&DiagnosticEvent::TrackingLost {
            consecutive_failures: 4,
        });
        let _ = format_event(&DiagnosticEvent::TrackingRecovered);
        let _ = format_event(&DiagnosticEvent::ProjectedTrackingFallback {
            reason: crate::ProjectedTrackingFallbackReason::TooFewCandidateMatches {
                candidates: 12,
                required: 32,
            },
        });
        let _ = format_event(&DiagnosticEvent::KeyframeCreated {
            keyframe_id: crate::map::KeyframeId::default(),
            landmarks: 12,
        });
        let _ = format_event(&DiagnosticEvent::KeyframeRemoved {
            keyframe_id: crate::map::KeyframeId::default(),
            reason: KeyframeRemovalReason::Redundant,
        });
        let _ = format_event(&DiagnosticEvent::LoopClosureDetected {
            query: crate::map::KeyframeId::default(),
            match_kf: crate::map::KeyframeId::default(),
            cosine_similarity: crate::CosineSimilarity::try_new(0.8).expect("valid similarity"),
        });
        let _ = format_event(&DiagnosticEvent::LoopDescriptorMatchDegraded {
            candidate_keyframe: crate::map::KeyframeId::default(),
            zero_norm_query_descriptors: 1,
            zero_norm_candidate_descriptors: 2,
        });
        let _ = format_event(&DiagnosticEvent::LoopClosureRejected {
            reason: LoopClosureRejectReason::VerificationFailed,
        });
        let _ = format_event(&DiagnosticEvent::BackendWorkerDied {
            respawn_count: 1,
            message: "forced panic".to_string(),
        });
        let _ = format_event(&DiagnosticEvent::BackendWorkerRestartFailed {
            respawn_count: 1,
            max_respawns: 2,
            exhausted: false,
            error: std::sync::Arc::new(std::io::Error::other("test restart failure")),
        });
        let _ = format_event(&DiagnosticEvent::DescriptorWorkerDied { respawn_count: 1 });
        let _ = format_event(&DiagnosticEvent::DescriptorWorkerRestartFailed {
            respawn_count: 1,
            max_respawns: 2,
            exhausted: false,
            error: std::sync::Arc::new(crate::DescriptorInitError::WorkerThread {
                source: std::io::Error::other("test restart failure"),
            }),
        });
        let _ = format_event(&DiagnosticEvent::DescriptorInferenceFailed {
            keyframe_id: crate::KeyframeId::default(),
            source_snapshot: crate::map::SlamMap::new().snapshot(),
            error: std::sync::Arc::new(crate::InferenceError::InvariantViolation {
                context: "test descriptor failure",
            }),
        });
        let _ = format_event(&DiagnosticEvent::BootstrapDescriptorUnavailable {
            keyframe_id: crate::KeyframeId::default(),
            source_snapshot: crate::map::SlamMap::new().snapshot(),
            error: std::sync::Arc::new(crate::BootstrapDescriptorError::Aggregation {
                source: crate::loop_closure::GlobalDescriptorError::ZeroNorm,
            }),
        });
        let _ = format_event(&DiagnosticEvent::DescriptorIndexFailed {
            keyframe_id: crate::KeyframeId::default(),
            source_snapshot: crate::map::SlamMap::new().snapshot(),
            error: std::sync::Arc::new(
                crate::loop_closure::KeyframeDatabaseError::SequenceExhausted {
                    next_sequence: usize::MAX,
                },
            ),
        });
        let _ = format_event(&DiagnosticEvent::RelocalizationStarted);
        let _ = format_event(&DiagnosticEvent::RelocalizationSucceeded {
            keyframe_id: crate::map::KeyframeId::default(),
        });
        let _ = format_event(&DiagnosticEvent::BaDegenerate {
            reason: crate::DegenerateReason::NoFactors,
        });
        let _ = format_event(&DiagnosticEvent::BaStalled {
            attempted_iterations: std::num::NonZeroUsize::MIN,
        });
    }
}
