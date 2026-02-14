use crate::{
    BaResult, DiagnosticEvent, FrameDiagnostics, LoopClosureRejectReason, RerunSink, Timestamp,
    VizLogError,
};

const TIMELINE_CAPTURE_NS: &str = "capture_ns";

const PATH_HEALTH_INLIER_RATIO: &str = "diagnostics/health/inlier_ratio";
const PATH_HEALTH_REPROJECTION_RMSE: &str = "diagnostics/health/reprojection_rmse_px";
const PATH_HEALTH_REPROJECTION_MAX: &str = "diagnostics/health/reprojection_max_px";

const PATH_TRACKING_FEATURES_DETECTED: &str = "diagnostics/tracking/features_detected";
const PATH_TRACKING_FEATURES_MATCHED: &str = "diagnostics/tracking/features_matched";
const PATH_TRACKING_PNP_OBSERVATIONS: &str = "diagnostics/tracking/pnp_observations";
const PATH_TRACKING_RANSAC_ITERATIONS: &str = "diagnostics/tracking/ransac_iterations";
const PATH_TRACKING_PARALLAX: &str = "diagnostics/tracking/parallax_px";
const PATH_TRACKING_COVISIBILITY: &str = "diagnostics/tracking/covisibility";
const PATH_TRACKING_KEYFRAME_CREATED: &str = "diagnostics/tracking/keyframe_created";

const PATH_TIMING_TRACKING_MS: &str = "diagnostics/timing/tracking_ms";

const PATH_MAP_KEYFRAMES: &str = "diagnostics/map/keyframes";
const PATH_MAP_POINTS: &str = "diagnostics/map/points";

const PATH_TRI_CANDIDATES: &str = "diagnostics/triangulation/candidates";
const PATH_TRI_KEPT: &str = "diagnostics/triangulation/kept";
const PATH_TRI_DROPPED_DISPARITY: &str = "diagnostics/triangulation/dropped_disparity";
const PATH_TRI_DROPPED_DEPTH: &str = "diagnostics/triangulation/dropped_depth";
const PATH_TRI_DROPPED_OUT_OF_BOUNDS: &str = "diagnostics/triangulation/dropped_out_of_bounds";
const PATH_TRI_DROPPED_DUPLICATE: &str = "diagnostics/triangulation/dropped_duplicate";

const PATH_BA_FINAL_COST: &str = "diagnostics/ba/final_cost";
const PATH_BA_ITERATIONS: &str = "diagnostics/ba/iterations";

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
    scalars.push((PATH_LOOP_CANDIDATES, diag.loop_candidate_count as f64));
    scalars.push((
        PATH_TRACKING_KEYFRAME_CREATED,
        if diag.keyframe_created { 1.0 } else { 0.0 },
    ));
    scalars.push((
        PATH_LOOP_APPLIED,
        if diag.loop_closure_applied { 1.0 } else { 0.0 },
    ));

    if let Some(v) = diag.inlier_ratio {
        scalars.push((PATH_HEALTH_INLIER_RATIO, v as f64));
    }
    if let Some(v) = diag.reprojection_rmse_px {
        scalars.push((PATH_HEALTH_REPROJECTION_RMSE, v as f64));
    }
    if let Some(v) = diag.reprojection_max_px {
        scalars.push((PATH_HEALTH_REPROJECTION_MAX, v as f64));
    }
    if let Some(v) = diag.features_detected {
        scalars.push((PATH_TRACKING_FEATURES_DETECTED, v as f64));
    }
    if let Some(v) = diag.features_matched {
        scalars.push((PATH_TRACKING_FEATURES_MATCHED, v as f64));
    }
    if let Some(v) = diag.pnp_observations {
        scalars.push((PATH_TRACKING_PNP_OBSERVATIONS, v as f64));
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
        DiagnosticEvent::TrackingRecovered => (
            "tracking recovered".to_string(),
            rerun::TextLogLevel::INFO,
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
                LoopClosureRejectReason::VerificationFailed => {
                    "verification failed".to_string()
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
}

#[cfg(test)]
mod tests {
    use super::{
        diagnostics_scalars, format_event, PATH_HEALTH_INLIER_RATIO, PATH_MAP_KEYFRAMES,
        PATH_MAP_POINTS,
    };
    use crate::{
        DiagnosticEvent, FrameDiagnostics, KeyframeRemovalReason, LoopClosureRejectReason,
        TriangulationStats,
    };

    #[test]
    fn diagnostics_scalars_empty_has_baselines() {
        let diag = FrameDiagnostics::empty(5, 13);
        let scalars = diagnostics_scalars(&diag);
        assert!(scalars
            .iter()
            .any(|(path, value)| *path == PATH_MAP_KEYFRAMES && *value == 5.0));
        assert!(scalars
            .iter()
            .any(|(path, value)| *path == PATH_MAP_POINTS && *value == 13.0));
    }

    #[test]
    fn diagnostics_scalars_include_present_fields() {
        let mut diag = FrameDiagnostics::empty(1, 2);
        diag.inlier_ratio = Some(0.75);
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
                .any(|(path, value)| *path == PATH_HEALTH_INLIER_RATIO
                    && (*value - 0.75).abs() < 1e-6)
        );
        assert!(scalars
            .iter()
            .any(|(path, _)| *path == "diagnostics/tracking/features_detected"));
        assert!(scalars
            .iter()
            .any(|(path, _)| *path == "diagnostics/triangulation/candidates"));
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
