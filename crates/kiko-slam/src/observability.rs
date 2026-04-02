use crate::{
    BaResult, ComponentHealth, DegradationLevel, DiagnosticEvent, FrameDiagnostics, KeyframeStatus,
    LoopClosureRejectReason, LoopClosureStatus, RerunSink, SystemHealth, Timestamp, TrackingHealth,
    VizLogError,
};

const TIMELINE_CAPTURE_NS: &str = "capture_ns";

const PATH_HEALTH_PNP_INLIER_RATIO: &str = "diagnostics/health/pnp_inlier_ratio";
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
const PATH_HEALTH_BACKEND_DROPPED_FULL: &str = "diagnostics/health/backend_dropped_full";
const PATH_HEALTH_BACKEND_DROPPED_DISCONNECTED: &str =
    "diagnostics/health/backend_dropped_disconnected";
const PATH_HEALTH_BACKEND_STALE: &str = "diagnostics/health/backend_stale";
const PATH_HEALTH_BACKEND_REJECTED: &str = "diagnostics/health/backend_rejected";
const PATH_HEALTH_BACKEND_WORKER_FAILURES: &str = "diagnostics/health/backend_worker_failures";
const PATH_HEALTH_BACKEND_RESPAWN_COUNT: &str = "diagnostics/health/backend_respawn_count";
const PATH_HEALTH_BACKEND_PANICS: &str = "diagnostics/health/backend_panics";

const PATH_TRACKING_FEATURES_DETECTED: &str = "diagnostics/tracking/features_detected";
const PATH_TRACKING_FEATURES_MATCHED: &str = "diagnostics/tracking/features_matched";
const PATH_TRACKING_PNP_TRACKED_OBSERVATIONS: &str =
    "diagnostics/tracking/pnp_tracked_observations";
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
const PATH_TRACKING_VIO_PROPOSAL_REJECTED_INSUFFICIENT_SHARED_SUPPORT: &str =
    "diagnostics/tracking/vio_proposal_rejected_insufficient_shared_projectable_support";
const PATH_TRACKING_VIO_PROPOSAL_REJECTED_CHANGED_SUPPORT: &str =
    "diagnostics/tracking/vio_proposal_rejected_changed_projectable_tracked_support";
const PATH_TRACKING_VIO_PROPOSAL_REJECTED_HIGHER_SHARED_RMSE: &str = "diagnostics/tracking/vio_proposal_rejected_higher_shared_projectable_tracked_reprojection_rmse";
const PATH_TRACKING_RANSAC_ITERATIONS: &str = "diagnostics/tracking/ransac_iterations";
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
const PATH_TRI_DROPPED_OUT_OF_BOUNDS: &str = "diagnostics/triangulation/dropped_out_of_bounds";
const PATH_TRI_DROPPED_DUPLICATE: &str = "diagnostics/triangulation/dropped_duplicate";

const PATH_BA_FINAL_COST: &str = "diagnostics/ba/final_cost";
const PATH_BA_ITERATIONS: &str = "diagnostics/ba/iterations";
#[cfg(feature = "vio")]
const PATH_VIO_FINAL_COST: &str = "diagnostics/vio/final_cost";
#[cfg(feature = "vio")]
const PATH_VIO_ITERATIONS: &str = "diagnostics/vio/iterations";
#[cfg(feature = "vio")]
const PATH_VIO_CONVERGED: &str = "diagnostics/vio/converged";
#[cfg(feature = "vio")]
const PATH_VIO_COST_REPROJECTION: &str = "diagnostics/vio/cost/reprojection";
#[cfg(feature = "vio")]
const PATH_VIO_COST_IMU: &str = "diagnostics/vio/cost/imu";
#[cfg(feature = "vio")]
const PATH_VIO_COST_BIAS_RANDOM_WALK: &str = "diagnostics/vio/cost/bias_random_walk";
#[cfg(feature = "vio")]
const PATH_VIO_COST_VELOCITY_ANCHOR: &str = "diagnostics/vio/cost/velocity_anchor";
#[cfg(feature = "vio")]
const PATH_VIO_COST_BIAS_PRIOR: &str = "diagnostics/vio/cost/bias_prior";

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
            PATH_TRACKING_VIO_PROPOSAL_REJECTED_INSUFFICIENT_SHARED_SUPPORT,
            if v == crate::VioProposalDisposition::RejectedInsufficientSharedProjectableSupport {
                1.0
            } else {
                0.0
            },
        ));
        scalars.push((
            PATH_TRACKING_VIO_PROPOSAL_REJECTED_CHANGED_SUPPORT,
            if v == crate::VioProposalDisposition::RejectedChangedProjectableTrackedSupport {
                1.0
            } else {
                0.0
            },
        ));
        scalars.push((
            PATH_TRACKING_VIO_PROPOSAL_REJECTED_HIGHER_SHARED_RMSE,
            if v
                == crate::VioProposalDisposition::RejectedHigherSharedProjectableTrackedReprojectionRmse
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
        scalars.push((
            PATH_TRI_DROPPED_OUT_OF_BOUNDS,
            stats.dropped_out_of_bounds as f64,
        ));
        scalars.push((PATH_TRI_DROPPED_DUPLICATE, stats.dropped_duplicate as f64));
    }

    if let Some(ba_result) = diag.ba_result.as_ref() {
        match ba_result {
            BaResult::Converged {
                iterations,
                final_cost,
            }
            | BaResult::MaxIterations {
                iterations,
                final_cost,
            } => {
                scalars.push((PATH_BA_ITERATIONS, *iterations as f64));
                scalars.push((PATH_BA_FINAL_COST, *final_cost));
            }
            BaResult::Degenerate { .. } => {}
        }
    }

    #[cfg(feature = "vio")]
    if let Some(vio_result) = diag.vio_solve_result.as_ref() {
        scalars.push((PATH_VIO_ITERATIONS, vio_result.iterations as f64));
        scalars.push((PATH_VIO_FINAL_COST, vio_result.final_cost));
        scalars.push((
            PATH_VIO_CONVERGED,
            if vio_result.converged { 1.0 } else { 0.0 },
        ));
        scalars.push((
            PATH_VIO_COST_REPROJECTION,
            vio_result.cost_breakdown.reprojection_cost,
        ));
        scalars.push((PATH_VIO_COST_IMU, vio_result.cost_breakdown.imu_cost));
        scalars.push((
            PATH_VIO_COST_BIAS_RANDOM_WALK,
            vio_result.cost_breakdown.bias_random_walk_cost,
        ));
        scalars.push((
            PATH_VIO_COST_VELOCITY_ANCHOR,
            vio_result.cost_breakdown.velocity_anchor_cost,
        ));
        scalars.push((
            PATH_VIO_COST_BIAS_PRIOR,
            vio_result.cost_breakdown.bias_prior_cost,
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
            similarity,
        } => (
            format!(
                "loop closure applied: query={query:?}, match={match_kf:?}, similarity={similarity:.3}"
            ),
            rerun::TextLogLevel::INFO,
        ),
        DiagnosticEvent::LoopClosureRejected { reason } => {
            let reason_text = match reason {
                LoopClosureRejectReason::TooFewCorrespondences { count } => {
                    format!("too few correspondences ({count})")
                }
                LoopClosureRejectReason::VerificationFailed => "verification failed".to_string(),
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
        DiagnosticEvent::BackendWorkerDied { respawn_count } => (
            format!("backend worker died (respawns={respawn_count})"),
            rerun::TextLogLevel::ERROR,
        ),
        DiagnosticEvent::DescriptorWorkerDied { respawn_count } => (
            format!("descriptor worker died (respawns={respawn_count})"),
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
            PATH_HEALTH_BACKEND_DROPPED_FULL,
            &rerun::Scalars::single(health.backend_stats.dropped_full as f64),
        )?;
        rec.log(
            PATH_HEALTH_BACKEND_DROPPED_DISCONNECTED,
            &rerun::Scalars::single(health.backend_stats.dropped_disconnected as f64),
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
            PATH_HEALTH_BACKEND_RESPAWN_COUNT,
            &rerun::Scalars::single(health.backend_stats.respawn_count as f64),
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
        PATH_HEALTH_PNP_INLIER_RATIO, PATH_HEALTH_PNP_PROJECTABLE_TRACKED_REPROJECTION_MAX,
        PATH_HEALTH_PNP_PROJECTABLE_TRACKED_REPROJECTION_MSE_PER_AXIS,
        PATH_HEALTH_PNP_PROJECTABLE_TRACKED_REPROJECTION_RMSE, PATH_MAP_KEYFRAMES, PATH_MAP_POINTS,
        PATH_TRACKING_PNP_PROJECTABLE_TRACKED_OBSERVATIONS,
        PATH_TRACKING_SHARED_PROJECTABLE_TRACKED_OBSERVATIONS, PATH_TRACKING_VIO_PROPOSAL_ADOPTED,
        PATH_TRACKING_VIO_PROPOSAL_PNP_PROJECTABLE_TRACKED_OBSERVATIONS, diagnostics_scalars,
        format_event,
    };
    use crate::{
        DiagnosticEvent, FrameDiagnostics, KeyframeRemovalReason, LoopClosureRejectReason,
        PnpInlierRatioMetric, PnpProjectableTrackedObservationCountMetric,
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
    fn diagnostics_scalars_include_present_fields() {
        let mut diag = FrameDiagnostics::empty(1, 2);
        diag.pnp_inlier_ratio = Some(PnpInlierRatioMetric::new(0.75).expect("ratio"));
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
        diag.vio_proposal_projectable_tracked_observations =
            Some(VioProposalProjectableTrackedObservationCountMetric::new(8));
        diag.vio_proposal_projectable_tracked_observation_reprojection_rmse_px = Some(
            VioProposalProjectableTrackedObservationPixelResidualMetric::new(0.9)
                .expect("vio proposal rmse"),
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
        diag.vio_proposal_disposition = Some(VioProposalDisposition::Adopted);
        diag.depth_reorder_warnings = Some(3);
        diag.features_detected = Some(400);
        diag.triangulation = Some(TriangulationStats {
            candidate_matches: 10,
            kept: 8,
            dropped_disparity: 1,
            dropped_out_of_bounds: 0,
            dropped_depth: 1,
            dropped_duplicate: 0,
        });
        let scalars = diagnostics_scalars(&diag);
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
        let (text, level) = format_event(&DiagnosticEvent::BackendWorkerDied { respawn_count: 2 });
        assert!(text.contains("backend worker died"));
        assert_eq!(level, rerun::TextLogLevel::ERROR);
    }

    #[test]
    fn format_event_supports_all_variants() {
        let _ = format_event(&DiagnosticEvent::TrackingLost {
            consecutive_failures: 4,
        });
        let _ = format_event(&DiagnosticEvent::TrackingRecovered);
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
            similarity: 0.8,
        });
        let _ = format_event(&DiagnosticEvent::LoopClosureRejected {
            reason: LoopClosureRejectReason::VerificationFailed,
        });
        let _ = format_event(&DiagnosticEvent::BackendWorkerDied { respawn_count: 1 });
        let _ = format_event(&DiagnosticEvent::DescriptorWorkerDied { respawn_count: 1 });
        let _ = format_event(&DiagnosticEvent::RelocalizationStarted);
        let _ = format_event(&DiagnosticEvent::RelocalizationSucceeded {
            keyframe_id: crate::map::KeyframeId::default(),
        });
        let _ = format_event(&DiagnosticEvent::BaDegenerate {
            reason: crate::DegenerateReason::NoFactors,
        });
    }
}
