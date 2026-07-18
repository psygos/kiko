//! Hardware-independent navigation contracts and algorithms.

mod cell_inflation;
mod frames;
mod global_planner;
mod ingress;
mod local_costmap;
pub mod mpc;
mod odometry;
mod shadow_command;

pub use frames::{
    BaseFrame, BaseToOdom, LocalCostmapFrame, LocalCostmapToOdom, MapFrame, MapToOdom, OdomFrame,
    OdomToBase, OdomToLocalCostmap, OdomToMap, PlanarAxis, PlanarFrame, PlanarPoint,
    PlanarPointError, PlanarTransform, PlanarTransformComponent, PlanarTransformError,
    PlanarTransformOperation,
};
pub use global_planner::{
    GlobalPath, GlobalPlanError, GlobalPlanIdentity, GlobalPlanner, GlobalPlannerConfig,
    GlobalPlannerInstanceId, GlobalPlannerInvocationId, GlobalPlannerRevision, MapPoint, PlanStart,
    PointGoal, UnknownSpacePolicy,
};
pub use ingress::{
    AcceptedDepthIngress, ControlTickIngress, CurrentMapEpochBinding,
    MAX_NAVIGATION_INGRESS_RECORDS, MapEpochStartedIngress, MapEpochTransition,
    MapPointGoalIngress, NAVIGATION_INGRESS_FORMAT_VERSION, NAVIGATION_INGRESS_STREAM_FILE,
    NavigationClockEpoch, NavigationClockOffset, NavigationGoalReplayError,
    NavigationIngressBoundaryError, NavigationIngressCapacity, NavigationIngressCapacityError,
    NavigationIngressEvent, NavigationIngressLog, NavigationIngressParseError,
    NavigationIngressReadStage, NavigationIngressReader, NavigationIngressRecord,
    NavigationIngressSequence, NavigationIngressStreamReadError, NavigationIngressStreamWriteError,
    NavigationIngressWriteError, NavigationIngressWriteStage, NavigationIngressWriter,
    NavigationMapEpochCoordinator, NavigationRecordingId, NavigationRecordingIdError,
    NavigationReplayClock, NavigationReplayClockError, RecordedImuReport, RecordedMapEpochId,
    RecordedMapEpochIdError, ReplayMapEpochBinding, VisualAttemptIngress, VisualAttemptOutcome,
};
pub use local_costmap::{
    DepthFrameKey, LocalCostmap, LocalCostmapCell, LocalCostmapClockRegression, LocalCostmapConfig,
    LocalCostmapConfigError, LocalCostmapCoordinateError, LocalCostmapError, LocalCostmapFreshness,
    LocalCostmapProvenance, LocalCostmapQuery, LocalCostmapUpdateOutcome, LocalCostmapView,
    LocalDepthObservation, LocalDepthObservationError, TrackingCameraToBase,
};
pub use odometry::{
    BaseAcceleration, BaseAngularVelocity, CalibratedQuantity, CalibrationMatrix,
    CalibrationVector, DurationParameter, ImuCalibrationProvenance, ImuUpdate, OdomPlanarTwist,
    OdomSegmentId, OdomSegmentIdError, OdometryError, OdometryEstimate, OdometryQuality,
    OdometryState, OdometryUnavailable, PlanarOdometry, PlanarOdometryConfig,
    PlanarOdometryConfigDto, PlanarOdometryConfigError, PlanarityComponent, PoseHistoryQuery,
    PredictionTranslationModel, RawImuCalibration, RawImuCalibrationDto, RawImuCalibrationError,
    ReanchorReason, ScalarParameter, TimeAlignedOdomPose, TimeAlignment, TranslationIntegration,
    VisualCaptureProvenance,
};
pub use shadow_command::{
    MAX_SHADOW_COMMAND_RECORDS, MotorPacketsSent, ShadowCommandConfig, ShadowCommandConfigDto,
    ShadowCommandConfigError, ShadowCommandDisposition, ShadowCommandError, ShadowCommandRecord,
    ShadowCommandSession, ShadowDecisionId, ShadowPwmPair, ShadowPwmPairError,
};
