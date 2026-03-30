use std::marker::PhantomData;
use std::time::Duration;

use crate::local_ba::{BaResult, DegenerateReason};
use crate::map::KeyframeId;
use crate::triangulation::TriangulationStats;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObservationSupport {
    PnpAcceptedInliers,
    PnpTrackedObservations,
    HeldOutObservations,
    AllObservations,
}

impl ObservationSupport {
    pub fn label(self) -> &'static str {
        match self {
            ObservationSupport::PnpAcceptedInliers => "pnp_accepted_inliers",
            ObservationSupport::PnpTrackedObservations => "pnp_tracked_observations",
            ObservationSupport::HeldOutObservations => "held_out_observations",
            ObservationSupport::AllObservations => "all_observations",
        }
    }
}

pub trait ObservationSupportMarker: Clone + Copy + std::fmt::Debug + Send + Sync + 'static {
    const SUPPORT: ObservationSupport;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PnpAcceptedInliersSupport;

impl ObservationSupportMarker for PnpAcceptedInliersSupport {
    const SUPPORT: ObservationSupport = ObservationSupport::PnpAcceptedInliers;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PnpTrackedObservationsSupport;

impl ObservationSupportMarker for PnpTrackedObservationsSupport {
    const SUPPORT: ObservationSupport = ObservationSupport::PnpTrackedObservations;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeldOutObservationsSupport;

impl ObservationSupportMarker for HeldOutObservationsSupport {
    const SUPPORT: ObservationSupport = ObservationSupport::HeldOutObservations;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AllObservationsSupport;

impl ObservationSupportMarker for AllObservationsSupport {
    const SUPPORT: ObservationSupport = ObservationSupport::AllObservations;
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DiagnosticMetricError {
    NonFinite {
        metric: &'static str,
        value: f32,
    },
    Negative {
        metric: &'static str,
        value: f32,
    },
    OutOfRange {
        metric: &'static str,
        value: f32,
        min: f32,
        max: f32,
    },
}

impl std::fmt::Display for DiagnosticMetricError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DiagnosticMetricError::NonFinite { metric, value } => {
                write!(f, "{metric} must be finite, got {value}")
            }
            DiagnosticMetricError::Negative { metric, value } => {
                write!(f, "{metric} must be >= 0, got {value}")
            }
            DiagnosticMetricError::OutOfRange {
                metric,
                value,
                min,
                max,
            } => {
                write!(f, "{metric} must be in [{min}, {max}], got {value}")
            }
        }
    }
}

impl std::error::Error for DiagnosticMetricError {}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RatioMetric<S>
where
    S: ObservationSupportMarker,
{
    value: f32,
    support: PhantomData<S>,
}

impl<S> RatioMetric<S>
where
    S: ObservationSupportMarker,
{
    pub fn new(value: f32) -> Result<Self, DiagnosticMetricError> {
        if !value.is_finite() {
            return Err(DiagnosticMetricError::NonFinite {
                metric: "ratio metric",
                value,
            });
        }
        if !(0.0..=1.0).contains(&value) {
            return Err(DiagnosticMetricError::OutOfRange {
                metric: "ratio metric",
                value,
                min: 0.0,
                max: 1.0,
            });
        }
        Ok(Self {
            value,
            support: PhantomData,
        })
    }

    pub fn value(self) -> f32 {
        self.value
    }

    pub fn support(self) -> ObservationSupport {
        S::SUPPORT
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CountMetric<S>
where
    S: ObservationSupportMarker,
{
    count: usize,
    support: PhantomData<S>,
}

impl<S> CountMetric<S>
where
    S: ObservationSupportMarker,
{
    pub fn new(count: usize) -> Self {
        Self {
            count,
            support: PhantomData,
        }
    }

    pub fn count(self) -> usize {
        self.count
    }

    pub fn support(self) -> ObservationSupport {
        S::SUPPORT
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PixelResidualMetric<S>
where
    S: ObservationSupportMarker,
{
    value_px: f32,
    support: PhantomData<S>,
}

impl<S> PixelResidualMetric<S>
where
    S: ObservationSupportMarker,
{
    pub fn new(value_px: f32) -> Result<Self, DiagnosticMetricError> {
        if !value_px.is_finite() {
            return Err(DiagnosticMetricError::NonFinite {
                metric: "pixel residual metric",
                value: value_px,
            });
        }
        if value_px < 0.0 {
            return Err(DiagnosticMetricError::Negative {
                metric: "pixel residual metric",
                value: value_px,
            });
        }
        Ok(Self {
            value_px,
            support: PhantomData,
        })
    }

    pub fn value_px(self) -> f32 {
        self.value_px
    }

    pub fn support(self) -> ObservationSupport {
        S::SUPPORT
    }
}

pub type PnpInlierRatioMetric = RatioMetric<PnpTrackedObservationsSupport>;
pub type PnpTrackedObservationCountMetric = CountMetric<PnpTrackedObservationsSupport>;
pub type PnpAcceptedInlierPixelResidualMetric = PixelResidualMetric<PnpAcceptedInliersSupport>;

/// Normalized Innovation Squared: r^T Σ^{-1} r / dim.
/// Should be ≈ 1.0 when the noise model matches reality.
/// >> 1 means the model is overconfident (noise is worse than assumed).
/// << 1 means the model is conservative (noise is better than assumed).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormalizedInnovationSquared<S>
where
    S: ObservationSupportMarker,
{
    value: f64,
    dim: usize,
    support: PhantomData<S>,
}

impl<S> NormalizedInnovationSquared<S>
where
    S: ObservationSupportMarker,
{
    pub fn new(value: f64, dim: usize) -> Result<Self, DiagnosticMetricError> {
        if !value.is_finite() {
            return Err(DiagnosticMetricError::NonFinite {
                metric: "normalized innovation squared",
                value: value as f32,
            });
        }
        Ok(Self {
            value,
            dim,
            support: PhantomData,
        })
    }

    /// NIS value: should be ≈ 1.0.
    pub fn value(&self) -> f64 {
        self.value
    }

    /// Residual dimension used for normalization.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Support set this was computed over.
    pub fn support(&self) -> ObservationSupport {
        S::SUPPORT
    }
}

pub type PnpInlierNis = NormalizedInnovationSquared<PnpAcceptedInliersSupport>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyframeRemovalReason {
    Redundant,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyframeStatus {
    Created,
    Rejected,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoopClosureStatus {
    Applied,
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
    pub pnp_inlier_ratio: Option<PnpInlierRatioMetric>,
    pub pnp_tracked_observations: Option<PnpTrackedObservationCountMetric>,
    pub ransac_iterations: Option<usize>,
    pub pnp_inlier_reprojection_rmse_px: Option<PnpAcceptedInlierPixelResidualMetric>,
    pub pnp_inlier_reprojection_max_px: Option<PnpAcceptedInlierPixelResidualMetric>,
    /// Normalized Innovation Squared for PnP inliers: r^T Σ^{-1} r / dim.
    /// Should be ≈ 1.0 when noise model matches reality.
    pub pnp_inlier_nis: Option<PnpInlierNis>,
    pub parallax_px: Option<f32>,
    pub covisibility: Option<f32>,
    pub keyframe_status: Option<KeyframeStatus>,
    pub triangulation: Option<TriangulationStats>,
    pub ba_result: Option<BaResult>,
    pub loop_candidate_count: usize,
    pub loop_closure_status: Option<LoopClosureStatus>,
    pub tracking_time: Option<Duration>,
    pub map_keyframes: usize,
    pub map_points: usize,
    pub depth_reorder_warnings: Option<u64>,
    pub features_detected: Option<usize>,
    pub features_matched: Option<usize>,
}

impl FrameDiagnostics {
    pub fn empty(map_keyframes: usize, map_points: usize) -> Self {
        Self {
            pnp_inlier_ratio: None,
            pnp_tracked_observations: None,
            ransac_iterations: None,
            pnp_inlier_reprojection_rmse_px: None,
            pnp_inlier_reprojection_max_px: None,
            pnp_inlier_nis: None,
            parallax_px: None,
            covisibility: None,
            keyframe_status: None,
            triangulation: None,
            ba_result: None,
            loop_candidate_count: 0,
            loop_closure_status: None,
            tracking_time: None,
            map_keyframes,
            map_points,
            depth_reorder_warnings: None,
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
        DiagnosticEvent, DiagnosticMetricError, FrameDiagnostics,
        LoopClosureRejectReason, ObservationSupport, PnpAcceptedInlierPixelResidualMetric,
        PnpInlierRatioMetric, PnpTrackedObservationCountMetric,
    };
    use crate::DegenerateReason;

    #[test]
    fn ratio_metric_enforces_unit_interval() {
        assert!(matches!(
            PnpInlierRatioMetric::new(1.2),
            Err(DiagnosticMetricError::OutOfRange { .. })
        ));
        assert!(matches!(
            PnpInlierRatioMetric::new(f32::NAN),
            Err(DiagnosticMetricError::NonFinite { .. })
        ));
        let metric = PnpInlierRatioMetric::new(0.25).expect("ratio metric");
        assert_eq!(metric.value(), 0.25);
        assert_eq!(metric.support(), ObservationSupport::PnpTrackedObservations);
    }

    #[test]
    fn pixel_residual_metric_rejects_negative_values() {
        assert!(matches!(
            PnpAcceptedInlierPixelResidualMetric::new(-0.1),
            Err(DiagnosticMetricError::Negative { .. })
        ));
        assert!(matches!(
            PnpAcceptedInlierPixelResidualMetric::new(f32::INFINITY),
            Err(DiagnosticMetricError::NonFinite { .. })
        ));
        let metric = PnpAcceptedInlierPixelResidualMetric::new(1.5).expect("px residual");
        assert_eq!(metric.value_px(), 1.5);
        assert_eq!(metric.support(), ObservationSupport::PnpAcceptedInliers);
    }

    #[test]
    fn count_metric_preserves_support() {
        let metric = PnpTrackedObservationCountMetric::new(12);
        assert_eq!(metric.count(), 12);
        assert_eq!(metric.support(), ObservationSupport::PnpTrackedObservations);
    }

    #[test]
    fn empty_diagnostics_has_all_none() {
        let diag = FrameDiagnostics::empty(2, 5);
        assert!(diag.pnp_inlier_ratio.is_none());
        assert!(diag.pnp_tracked_observations.is_none());
        assert!(diag.ransac_iterations.is_none());
        assert!(diag.pnp_inlier_reprojection_rmse_px.is_none());
        assert!(diag.pnp_inlier_reprojection_max_px.is_none());
        assert!(diag.parallax_px.is_none());
        assert!(diag.covisibility.is_none());
        assert!(diag.keyframe_status.is_none());
        assert!(diag.triangulation.is_none());
        assert!(diag.ba_result.is_none());
        assert_eq!(diag.loop_candidate_count, 0);
        assert!(diag.loop_closure_status.is_none());
        assert!(diag.tracking_time.is_none());
    }

    #[test]
    fn empty_diagnostics_carries_map_counts() {
        let diag = FrameDiagnostics::empty(7, 42);
        assert_eq!(diag.map_keyframes, 7);
        assert_eq!(diag.map_points, 42);
    }

    #[test]
    fn support_labels_are_distinct() {
        let labels = [
            ObservationSupport::PnpAcceptedInliers.label(),
            ObservationSupport::PnpTrackedObservations.label(),
            ObservationSupport::HeldOutObservations.label(),
            ObservationSupport::AllObservations.label(),
        ];
        let unique: HashSet<_> = labels.into_iter().collect();
        assert_eq!(unique.len(), 4);
    }

    #[test]
    fn diagnostic_event_variants_are_distinct() {
        let events = [
            DiagnosticEvent::TrackingLost {
                consecutive_failures: 1,
            },
            DiagnosticEvent::TrackingRecovered,
            DiagnosticEvent::LoopClosureRejected {
                reason: LoopClosureRejectReason::VerificationFailed,
            },
            DiagnosticEvent::BackendWorkerDied { respawn_count: 1 },
            DiagnosticEvent::DescriptorWorkerDied { respawn_count: 2 },
            DiagnosticEvent::RelocalizationStarted,
            DiagnosticEvent::BaDegenerate {
                reason: DegenerateReason::NoFactors,
            },
        ];

        let mut kinds = HashSet::new();
        for event in events {
            kinds.insert(discriminant(&event));
        }
        assert_eq!(kinds.len(), 7);
    }

    #[test]
    fn frame_diagnostics_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<FrameDiagnostics>();
    }
}
