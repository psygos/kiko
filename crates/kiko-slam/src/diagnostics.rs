use std::marker::PhantomData;
use std::time::Duration;

use crate::local_ba::{BaResult, DegenerateReason};
use crate::map::KeyframeId;
use crate::triangulation::TriangulationStats;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObservationSupport {
    PnpAcceptedInliers,
    PnpTrackedObservations,
    PnpProjectableTrackedObservations,
    VisualProposalProjectableTrackedObservations,
    VioProposalProjectableTrackedObservations,
    VisualVsVioSharedProjectableTrackedObservations,
    StableSurfaceRetainedRawObservations,
    HeldOutObservations,
    AllObservations,
}

impl ObservationSupport {
    pub fn label(self) -> &'static str {
        match self {
            ObservationSupport::PnpAcceptedInliers => "pnp_accepted_inliers",
            ObservationSupport::PnpTrackedObservations => "pnp_tracked_observations",
            ObservationSupport::PnpProjectableTrackedObservations => {
                "pnp_projectable_tracked_observations"
            }
            ObservationSupport::VisualProposalProjectableTrackedObservations => {
                "visual_proposal_projectable_tracked_observations"
            }
            ObservationSupport::VioProposalProjectableTrackedObservations => {
                "vio_proposal_projectable_tracked_observations"
            }
            ObservationSupport::VisualVsVioSharedProjectableTrackedObservations => {
                "visual_vs_vio_shared_projectable_tracked_observations"
            }
            ObservationSupport::StableSurfaceRetainedRawObservations => {
                "stable_surface_retained_raw_observations"
            }
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
pub struct PnpProjectableTrackedObservationsSupport;

impl ObservationSupportMarker for PnpProjectableTrackedObservationsSupport {
    const SUPPORT: ObservationSupport = ObservationSupport::PnpProjectableTrackedObservations;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VisualProposalProjectableTrackedObservationsSupport;

impl ObservationSupportMarker for VisualProposalProjectableTrackedObservationsSupport {
    const SUPPORT: ObservationSupport =
        ObservationSupport::VisualProposalProjectableTrackedObservations;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VioProposalProjectableTrackedObservationsSupport;

impl ObservationSupportMarker for VioProposalProjectableTrackedObservationsSupport {
    const SUPPORT: ObservationSupport =
        ObservationSupport::VioProposalProjectableTrackedObservations;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VisualVsVioSharedProjectableTrackedObservationsSupport;

impl ObservationSupportMarker for VisualVsVioSharedProjectableTrackedObservationsSupport {
    const SUPPORT: ObservationSupport =
        ObservationSupport::VisualVsVioSharedProjectableTrackedObservations;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StableSurfaceRetainedRawObservationsSupport;

impl ObservationSupportMarker for StableSurfaceRetainedRawObservationsSupport {
    const SUPPORT: ObservationSupport = ObservationSupport::StableSurfaceRetainedRawObservations;
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
    ZeroDenominator {
        metric: &'static str,
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
            DiagnosticMetricError::ZeroDenominator { metric } => {
                write!(f, "{metric} denominator must be > 0")
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PnpInlierRatioMetric {
    value: f32,
    accepted_inliers: PnpAcceptedInlierCountMetric,
    tracked_observations: PnpTrackedObservationCountMetric,
}

impl PnpInlierRatioMetric {
    pub fn new(
        accepted_inliers: PnpAcceptedInlierCountMetric,
        tracked_observations: PnpTrackedObservationCountMetric,
    ) -> Result<Self, DiagnosticMetricError> {
        if tracked_observations.count() == 0 {
            return Err(DiagnosticMetricError::ZeroDenominator {
                metric: "pnp inlier ratio",
            });
        }
        let value = accepted_inliers.count() as f32 / tracked_observations.count() as f32;
        if !value.is_finite() {
            return Err(DiagnosticMetricError::NonFinite {
                metric: "pnp inlier ratio",
                value,
            });
        }
        if !(0.0..=1.0).contains(&value) {
            return Err(DiagnosticMetricError::OutOfRange {
                metric: "pnp inlier ratio",
                value,
                min: 0.0,
                max: 1.0,
            });
        }
        Ok(Self {
            value,
            accepted_inliers,
            tracked_observations,
        })
    }

    pub fn value(self) -> f32 {
        self.value
    }

    pub fn accepted_inliers(self) -> PnpAcceptedInlierCountMetric {
        self.accepted_inliers
    }

    pub fn tracked_observations(self) -> PnpTrackedObservationCountMetric {
        self.tracked_observations
    }
}

pub type PnpTrackedObservationCountMetric = CountMetric<PnpTrackedObservationsSupport>;
pub type PnpAcceptedInlierCountMetric = CountMetric<PnpAcceptedInliersSupport>;
pub type PnpProjectableTrackedObservationCountMetric =
    CountMetric<PnpProjectableTrackedObservationsSupport>;
pub type VisualProposalProjectableTrackedObservationCountMetric =
    CountMetric<VisualProposalProjectableTrackedObservationsSupport>;
pub type VioProposalProjectableTrackedObservationCountMetric =
    CountMetric<VioProposalProjectableTrackedObservationsSupport>;
pub type VisualVsVioSharedProjectableTrackedObservationCountMetric =
    CountMetric<VisualVsVioSharedProjectableTrackedObservationsSupport>;
pub type PnpAcceptedInlierPixelResidualMetric = PixelResidualMetric<PnpAcceptedInliersSupport>;
pub type PnpProjectableTrackedObservationPixelResidualMetric =
    PixelResidualMetric<PnpProjectableTrackedObservationsSupport>;
pub type VisualProposalProjectableTrackedObservationPixelResidualMetric =
    PixelResidualMetric<VisualProposalProjectableTrackedObservationsSupport>;
pub type VioProposalProjectableTrackedObservationPixelResidualMetric =
    PixelResidualMetric<VioProposalProjectableTrackedObservationsSupport>;
pub type VisualVsVioSharedProjectableTrackedObservationPixelResidualMetric =
    PixelResidualMetric<VisualVsVioSharedProjectableTrackedObservationsSupport>;
pub type StableSurfaceRetainedRawPixelResidualMetric =
    PixelResidualMetric<StableSurfaceRetainedRawObservationsSupport>;

/// Mean squared pixel residual per image axis in px².
///
/// This is the average squared reprojection error divided by the 2 image residual
/// axes. It is not a true NIS metric because no measurement covariance model is
/// applied.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeanSquaredPixelResidualMetric<S>
where
    S: ObservationSupportMarker,
{
    value_px2: f64,
    support: PhantomData<S>,
}

impl<S> MeanSquaredPixelResidualMetric<S>
where
    S: ObservationSupportMarker,
{
    pub fn new(value_px2: f64) -> Result<Self, DiagnosticMetricError> {
        if !value_px2.is_finite() {
            return Err(DiagnosticMetricError::NonFinite {
                metric: "mean squared pixel residual metric",
                value: value_px2 as f32,
            });
        }
        if value_px2 < 0.0 {
            return Err(DiagnosticMetricError::Negative {
                metric: "mean squared pixel residual metric",
                value: value_px2 as f32,
            });
        }
        Ok(Self {
            value_px2,
            support: PhantomData,
        })
    }

    pub fn value_px2(&self) -> f64 {
        self.value_px2
    }

    pub fn support(&self) -> ObservationSupport {
        S::SUPPORT
    }
}

pub type PnpAcceptedInlierReprojectionMsePerAxisPx2Metric =
    MeanSquaredPixelResidualMetric<PnpAcceptedInliersSupport>;
pub type PnpProjectableTrackedObservationReprojectionMsePerAxisPx2Metric =
    MeanSquaredPixelResidualMetric<PnpProjectableTrackedObservationsSupport>;
pub type VisualVsVioSharedProjectableTrackedObservationReprojectionMsePerAxisPx2Metric =
    MeanSquaredPixelResidualMetric<VisualVsVioSharedProjectableTrackedObservationsSupport>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrackingPoseSource {
    VisualTracking,
    VisualBundleAdjustment,
    VioRefined,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VioProposalDisposition {
    NotRun,
    Adopted,
    RejectedUnusableSolve,
    RejectedInsufficientCurrentVioObservationSupport,
    RejectedInsufficientSharedAcceptedInlierSupport,
    RejectedChangedAcceptedInlierProjectability,
    RejectedHigherSharedAcceptedInlierReprojectionRmse,
}

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
    pub pnp_accepted_inliers: Option<PnpAcceptedInlierCountMetric>,
    pub ransac_iterations: Option<usize>,
    pub tracking_pose_source: Option<TrackingPoseSource>,
    pub pnp_projectable_tracked_observations: Option<PnpProjectableTrackedObservationCountMetric>,
    /// Reprojection RMSE in px over projectable tracked PnP observations under the solved pose.
    pub pnp_projectable_tracked_observation_reprojection_rmse_px:
        Option<PnpProjectableTrackedObservationPixelResidualMetric>,
    /// Maximum reprojection error in px over projectable tracked PnP observations under the solved pose.
    pub pnp_projectable_tracked_observation_reprojection_max_px:
        Option<PnpProjectableTrackedObservationPixelResidualMetric>,
    /// Mean squared reprojection error per image axis in px² over projectable tracked PnP observations.
    pub pnp_projectable_tracked_observation_reprojection_mse_per_axis_px2:
        Option<PnpProjectableTrackedObservationReprojectionMsePerAxisPx2Metric>,
    pub visual_proposal_projectable_tracked_observations:
        Option<VisualProposalProjectableTrackedObservationCountMetric>,
    pub visual_proposal_projectable_tracked_observation_reprojection_rmse_px:
        Option<VisualProposalProjectableTrackedObservationPixelResidualMetric>,
    pub vio_proposal_projectable_tracked_observations:
        Option<VioProposalProjectableTrackedObservationCountMetric>,
    pub vio_proposal_projectable_tracked_observation_reprojection_rmse_px:
        Option<VioProposalProjectableTrackedObservationPixelResidualMetric>,
    pub shared_projectable_tracked_observations:
        Option<VisualVsVioSharedProjectableTrackedObservationCountMetric>,
    pub visual_proposal_shared_projectable_tracked_observation_reprojection_rmse_px:
        Option<VisualVsVioSharedProjectableTrackedObservationPixelResidualMetric>,
    pub vio_proposal_shared_projectable_tracked_observation_reprojection_rmse_px:
        Option<VisualVsVioSharedProjectableTrackedObservationPixelResidualMetric>,
    pub visual_proposal_projectable_accepted_inliers: Option<PnpAcceptedInlierCountMetric>,
    pub visual_proposal_accepted_inlier_reprojection_rmse_px:
        Option<PnpAcceptedInlierPixelResidualMetric>,
    pub vio_proposal_projectable_accepted_inliers: Option<PnpAcceptedInlierCountMetric>,
    pub vio_proposal_accepted_inlier_reprojection_rmse_px:
        Option<PnpAcceptedInlierPixelResidualMetric>,
    pub shared_projectable_accepted_inliers: Option<PnpAcceptedInlierCountMetric>,
    pub visual_proposal_shared_accepted_inlier_reprojection_rmse_px:
        Option<PnpAcceptedInlierPixelResidualMetric>,
    pub vio_proposal_shared_accepted_inlier_reprojection_rmse_px:
        Option<PnpAcceptedInlierPixelResidualMetric>,
    pub vio_proposal_disposition: Option<VioProposalDisposition>,
    pub pnp_inlier_reprojection_rmse_px: Option<PnpAcceptedInlierPixelResidualMetric>,
    pub pnp_inlier_reprojection_max_px: Option<PnpAcceptedInlierPixelResidualMetric>,
    /// Mean squared reprojection error per image axis in px² over accepted PnP inliers.
    pub pnp_inlier_reprojection_mse_per_axis_px2:
        Option<PnpAcceptedInlierReprojectionMsePerAxisPx2Metric>,
    pub parallax_px: Option<f32>,
    pub covisibility: Option<f32>,
    pub keyframe_status: Option<KeyframeStatus>,
    pub triangulation: Option<TriangulationStats>,
    pub ba_result: Option<BaResult>,
    pub pose_ba_termination: Option<crate::PoseBaTermination>,
    #[cfg(feature = "vio")]
    pub vio_solve_result: Option<crate::VioSolveResult>,
    #[cfg(feature = "vio")]
    pub vio_calibrated_bias_prior_active: Option<bool>,
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
            pnp_accepted_inliers: None,
            ransac_iterations: None,
            tracking_pose_source: None,
            pnp_projectable_tracked_observations: None,
            pnp_projectable_tracked_observation_reprojection_rmse_px: None,
            pnp_projectable_tracked_observation_reprojection_max_px: None,
            pnp_projectable_tracked_observation_reprojection_mse_per_axis_px2: None,
            visual_proposal_projectable_tracked_observations: None,
            visual_proposal_projectable_tracked_observation_reprojection_rmse_px: None,
            vio_proposal_projectable_tracked_observations: None,
            vio_proposal_projectable_tracked_observation_reprojection_rmse_px: None,
            shared_projectable_tracked_observations: None,
            visual_proposal_shared_projectable_tracked_observation_reprojection_rmse_px: None,
            vio_proposal_shared_projectable_tracked_observation_reprojection_rmse_px: None,
            visual_proposal_projectable_accepted_inliers: None,
            visual_proposal_accepted_inlier_reprojection_rmse_px: None,
            vio_proposal_projectable_accepted_inliers: None,
            vio_proposal_accepted_inlier_reprojection_rmse_px: None,
            shared_projectable_accepted_inliers: None,
            visual_proposal_shared_accepted_inlier_reprojection_rmse_px: None,
            vio_proposal_shared_accepted_inlier_reprojection_rmse_px: None,
            vio_proposal_disposition: None,
            pnp_inlier_reprojection_rmse_px: None,
            pnp_inlier_reprojection_max_px: None,
            pnp_inlier_reprojection_mse_per_axis_px2: None,
            parallax_px: None,
            covisibility: None,
            keyframe_status: None,
            triangulation: None,
            ba_result: None,
            pose_ba_termination: None,
            #[cfg(feature = "vio")]
            vio_solve_result: None,
            #[cfg(feature = "vio")]
            vio_calibrated_bias_prior_active: None,
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
        DiagnosticEvent, DiagnosticMetricError, FrameDiagnostics, LoopClosureRejectReason,
        ObservationSupport, PnpAcceptedInlierPixelResidualMetric, PnpInlierRatioMetric,
        PnpProjectableTrackedObservationPixelResidualMetric, PnpTrackedObservationCountMetric,
        StableSurfaceRetainedRawPixelResidualMetric,
        VisualVsVioSharedProjectableTrackedObservationPixelResidualMetric,
    };
    use crate::{DegenerateReason, PnpAcceptedInlierCountMetric};

    #[test]
    fn ratio_metric_enforces_unit_interval() {
        let accepted = PnpAcceptedInlierCountMetric::new(3);
        let tracked = PnpTrackedObservationCountMetric::new(4);
        assert!(matches!(
            PnpInlierRatioMetric::new(
                PnpAcceptedInlierCountMetric::new(5),
                PnpTrackedObservationCountMetric::new(4)
            ),
            Err(DiagnosticMetricError::OutOfRange { .. })
        ));
        assert!(matches!(
            PnpInlierRatioMetric::new(accepted, PnpTrackedObservationCountMetric::new(0)),
            Err(DiagnosticMetricError::ZeroDenominator { .. })
        ));
        let metric = PnpInlierRatioMetric::new(accepted, tracked).expect("ratio metric");
        assert_eq!(metric.value(), 0.75);
        assert_eq!(
            metric.accepted_inliers().support(),
            ObservationSupport::PnpAcceptedInliers
        );
        assert_eq!(
            metric.tracked_observations().support(),
            ObservationSupport::PnpTrackedObservations
        );
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
    fn stable_surface_pixel_residual_metric_preserves_support() {
        let metric =
            StableSurfaceRetainedRawPixelResidualMetric::new(0.25).expect("surface px residual");
        assert_eq!(metric.value_px(), 0.25);
        assert_eq!(
            metric.support(),
            ObservationSupport::StableSurfaceRetainedRawObservations
        );
    }

    #[test]
    fn projectable_tracked_pnp_pixel_residual_metric_preserves_support() {
        let metric = PnpProjectableTrackedObservationPixelResidualMetric::new(2.0)
            .expect("tracked px residual");
        assert_eq!(metric.value_px(), 2.0);
        assert_eq!(
            metric.support(),
            ObservationSupport::PnpProjectableTrackedObservations
        );
    }

    #[test]
    fn shared_projectable_tracked_pnp_pixel_residual_metric_preserves_support() {
        let metric = VisualVsVioSharedProjectableTrackedObservationPixelResidualMetric::new(1.0)
            .expect("shared tracked px residual");
        assert_eq!(metric.value_px(), 1.0);
        assert_eq!(
            metric.support(),
            ObservationSupport::VisualVsVioSharedProjectableTrackedObservations
        );
    }

    #[test]
    fn accepted_inlier_count_metric_preserves_support() {
        let metric = crate::PnpAcceptedInlierCountMetric::new(6);
        assert_eq!(metric.count(), 6);
        assert_eq!(metric.support(), ObservationSupport::PnpAcceptedInliers);
    }

    #[test]
    fn empty_diagnostics_has_all_none() {
        let diag = FrameDiagnostics::empty(2, 5);
        assert!(diag.pnp_inlier_ratio.is_none());
        assert!(diag.pnp_tracked_observations.is_none());
        assert!(diag.pnp_accepted_inliers.is_none());
        assert!(diag.ransac_iterations.is_none());
        assert!(diag.tracking_pose_source.is_none());
        assert!(diag.pnp_projectable_tracked_observations.is_none());
        assert!(
            diag.pnp_projectable_tracked_observation_reprojection_rmse_px
                .is_none()
        );
        assert!(
            diag.pnp_projectable_tracked_observation_reprojection_max_px
                .is_none()
        );
        assert!(
            diag.pnp_projectable_tracked_observation_reprojection_mse_per_axis_px2
                .is_none()
        );
        assert!(
            diag.visual_proposal_projectable_tracked_observations
                .is_none()
        );
        assert!(
            diag.visual_proposal_projectable_tracked_observation_reprojection_rmse_px
                .is_none()
        );
        assert!(diag.vio_proposal_projectable_tracked_observations.is_none());
        assert!(
            diag.vio_proposal_projectable_tracked_observation_reprojection_rmse_px
                .is_none()
        );
        assert!(diag.shared_projectable_tracked_observations.is_none());
        assert!(
            diag.visual_proposal_shared_projectable_tracked_observation_reprojection_rmse_px
                .is_none()
        );
        assert!(
            diag.vio_proposal_shared_projectable_tracked_observation_reprojection_rmse_px
                .is_none()
        );
        assert!(diag.visual_proposal_projectable_accepted_inliers.is_none());
        assert!(
            diag.visual_proposal_accepted_inlier_reprojection_rmse_px
                .is_none()
        );
        assert!(diag.vio_proposal_projectable_accepted_inliers.is_none());
        assert!(
            diag.vio_proposal_accepted_inlier_reprojection_rmse_px
                .is_none()
        );
        assert!(diag.shared_projectable_accepted_inliers.is_none());
        assert!(
            diag.visual_proposal_shared_accepted_inlier_reprojection_rmse_px
                .is_none()
        );
        assert!(
            diag.vio_proposal_shared_accepted_inlier_reprojection_rmse_px
                .is_none()
        );
        assert!(diag.vio_proposal_disposition.is_none());
        assert!(diag.pnp_inlier_reprojection_rmse_px.is_none());
        assert!(diag.pnp_inlier_reprojection_max_px.is_none());
        assert!(diag.pnp_inlier_reprojection_mse_per_axis_px2.is_none());
        assert!(diag.parallax_px.is_none());
        assert!(diag.covisibility.is_none());
        assert!(diag.keyframe_status.is_none());
        assert!(diag.triangulation.is_none());
        assert!(diag.ba_result.is_none());
        #[cfg(feature = "vio")]
        assert!(diag.vio_solve_result.is_none());
        #[cfg(feature = "vio")]
        assert!(diag.vio_calibrated_bias_prior_active.is_none());
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
            ObservationSupport::VisualVsVioSharedProjectableTrackedObservations.label(),
            ObservationSupport::HeldOutObservations.label(),
            ObservationSupport::AllObservations.label(),
        ];
        let unique: HashSet<_> = labels.into_iter().collect();
        assert_eq!(unique.len(), 5);
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
