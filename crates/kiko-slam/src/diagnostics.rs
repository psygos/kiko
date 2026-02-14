use std::time::Duration;

use crate::local_ba::{BaResult, DegenerateReason};
use crate::map::KeyframeId;
use crate::triangulation::TriangulationStats;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyframeRemovalReason {
    Redundant,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LoopClosureRejectReason {
    TooFewCorrespondences {
        count: usize,
    },
    VerificationFailed,
    CorrectionTooLarge {
        translation_m: f32,
        rotation_deg: f32,
    },
    ApplyFailed,
}

#[derive(Clone, Debug)]
pub struct FrameDiagnostics {
    pub inlier_ratio: Option<f32>,
    pub pnp_observations: Option<usize>,
    pub ransac_iterations: Option<usize>,
    pub reprojection_rmse_px: Option<f32>,
    pub reprojection_max_px: Option<f32>,
    pub parallax_px: Option<f32>,
    pub covisibility: Option<f32>,
    pub keyframe_created: bool,
    pub triangulation: Option<TriangulationStats>,
    pub ba_result: Option<BaResult>,
    pub loop_candidate_count: usize,
    pub loop_closure_applied: bool,
    pub tracking_time: Option<Duration>,
    pub map_keyframes: usize,
    pub map_points: usize,
    pub features_detected: Option<usize>,
    pub features_matched: Option<usize>,
}

impl FrameDiagnostics {
    pub fn empty(map_keyframes: usize, map_points: usize) -> Self {
        Self {
            inlier_ratio: None,
            pnp_observations: None,
            ransac_iterations: None,
            reprojection_rmse_px: None,
            reprojection_max_px: None,
            parallax_px: None,
            covisibility: None,
            keyframe_created: false,
            triangulation: None,
            ba_result: None,
            loop_candidate_count: 0,
            loop_closure_applied: false,
            tracking_time: None,
            map_keyframes,
            map_points,
            features_detected: None,
            features_matched: None,
        }
    }
}

#[derive(Clone, Debug)]
pub enum DiagnosticEvent {
    TrackingLost {
        consecutive_failures: usize,
    },
    TrackingRecovered,
    KeyframeCreated {
        keyframe_id: KeyframeId,
        landmarks: usize,
    },
    KeyframeRemoved {
        keyframe_id: KeyframeId,
        reason: KeyframeRemovalReason,
    },
    LoopClosureDetected {
        query: KeyframeId,
        match_kf: KeyframeId,
        similarity: f32,
    },
    LoopClosureRejected {
        reason: LoopClosureRejectReason,
    },
    BackendWorkerDied {
        respawn_count: u32,
    },
    DescriptorWorkerDied {
        respawn_count: u32,
    },
    RelocalizationStarted,
    RelocalizationSucceeded {
        keyframe_id: KeyframeId,
    },
    BaDegenerate {
        reason: DegenerateReason,
    },
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::mem::discriminant;

    use super::{
        DiagnosticEvent, FrameDiagnostics, KeyframeRemovalReason, LoopClosureRejectReason,
    };
    use crate::map::KeyframeId;
    use crate::DegenerateReason;

    #[test]
    fn empty_diagnostics_has_all_none() {
        let diag = FrameDiagnostics::empty(3, 9);
        assert!(diag.inlier_ratio.is_none());
        assert!(diag.pnp_observations.is_none());
        assert!(diag.ransac_iterations.is_none());
        assert!(diag.reprojection_rmse_px.is_none());
        assert!(diag.reprojection_max_px.is_none());
        assert!(diag.parallax_px.is_none());
        assert!(diag.covisibility.is_none());
        assert!(!diag.keyframe_created);
        assert!(diag.triangulation.is_none());
        assert!(diag.ba_result.is_none());
        assert_eq!(diag.loop_candidate_count, 0);
        assert!(!diag.loop_closure_applied);
        assert!(diag.tracking_time.is_none());
        assert!(diag.features_detected.is_none());
        assert!(diag.features_matched.is_none());
    }

    #[test]
    fn empty_diagnostics_carries_map_counts() {
        let diag = FrameDiagnostics::empty(11, 42);
        assert_eq!(diag.map_keyframes, 11);
        assert_eq!(diag.map_points, 42);
    }

    #[test]
    fn diagnostic_event_variants_are_distinct() {
        let kf = KeyframeId::default();
        let variants = vec![
            DiagnosticEvent::TrackingLost {
                consecutive_failures: 3,
            },
            DiagnosticEvent::TrackingRecovered,
            DiagnosticEvent::KeyframeCreated {
                keyframe_id: kf,
                landmarks: 8,
            },
            DiagnosticEvent::KeyframeRemoved {
                keyframe_id: kf,
                reason: KeyframeRemovalReason::Redundant,
            },
            DiagnosticEvent::LoopClosureDetected {
                query: kf,
                match_kf: kf,
                similarity: 0.9,
            },
            DiagnosticEvent::LoopClosureRejected {
                reason: LoopClosureRejectReason::VerificationFailed,
            },
            DiagnosticEvent::BackendWorkerDied { respawn_count: 1 },
            DiagnosticEvent::DescriptorWorkerDied { respawn_count: 2 },
            DiagnosticEvent::RelocalizationStarted,
            DiagnosticEvent::RelocalizationSucceeded { keyframe_id: kf },
            DiagnosticEvent::BaDegenerate {
                reason: DegenerateReason::NoFactors,
            },
        ];
        let mut uniq = HashSet::new();
        for event in variants {
            uniq.insert(discriminant(&event));
        }
        assert_eq!(uniq.len(), 11);
    }

    #[test]
    fn frame_diagnostics_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<FrameDiagnostics>();
    }
}
