//! Hardware-independent navigation contracts and algorithms.

mod cell_inflation;
mod frames;
mod goal_input;
mod global_planner;
mod ingress;
mod local_costmap;
pub mod mpc;
mod odometry;
mod reference;
mod safety;
mod shadow_command;
pub use frames::{
    BaseFrame, BaseToOdom, LocalCostmapFrame, LocalCostmapToOdom, MapFrame, MapToOdom, OdomFrame,
    OdomToBase, OdomToLocalCostmap, OdomToMap, PlanarAxis, PlanarFrame, PlanarPoint,
    PlanarPointError, PlanarTransform, PlanarTransformComponent, PlanarTransformError,
    PlanarTransformOperation,
};
pub use goal_input::{NavigationGoalArg, NavigationGoalArgError, NavigationGoalAxis};
pub use global_planner::{
    GlobalPath, GlobalPlanError, GlobalPlanIdentity, GlobalPlanner, GlobalPlannerConfig,
    GlobalPlannerInstanceId, GlobalPlannerInvocationId, GlobalPlannerRevision, MapPoint, PlanStart,
    PointGoal, UnknownSpacePolicy,
};
pub use ingress::{
    AcceptedDepthIngress, ControlTickIngress, CurrentMapEpochBinding, FinalizedNavigationIngress,
    MAX_NAVIGATION_INGRESS_RECORDS, MapEpochStartedIngress, MapEpochTransition,
    MapPointGoalIngress, NAVIGATION_INGRESS_FORMAT_VERSION, NAVIGATION_INGRESS_HEADER_BYTES,
    NAVIGATION_INGRESS_RECORD_BYTES, NAVIGATION_INGRESS_STREAM_FILE,
    NAVIGATION_INGRESS_STREAM_FORMAT, NavigationClockEpoch, NavigationClockOffset,
    NavigationGoalReplayError, NavigationIngressBoundaryError, NavigationIngressCapacity,
    NavigationIngressCapacityError, NavigationIngressEvent, NavigationIngressLog,
    NavigationIngressParseError, NavigationIngressReadStage, NavigationIngressReader,
    NavigationIngressRecord, NavigationIngressScope, NavigationIngressSequence,
    NavigationIngressSidecarDescriptor, NavigationIngressSidecarDescriptorError,
    NavigationIngressStreamReadError, NavigationIngressStreamWriteError, NavigationIngressTimebase,
    NavigationIngressWriteError, NavigationIngressWriteStage, NavigationIngressWriter,
    NavigationMapEpochCoordinator, NavigationRecordingId, NavigationRecordingIdError,
    NavigationReplayClock, NavigationReplayClockError, PendingVisualAttemptIngress,
    RecordedImuReport, RecordedMapEpochId, RecordedMapEpochIdError, ReplayMapEpochBinding,
    VisualAttemptIngress, VisualAttemptOutcome,
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
pub use reference::{
    EpochPathMismatchV1, FORWARD_MOST_NEAREST_SEGMENT_V1, MAX_PATH_REFERENCE_POINTS,
    MAX_SUPPORTED_ABS_REFERENCE_YAW_RATE_RAD_S, MAX_SUPPORTED_PATH_LENGTH_M,
    MAX_SUPPORTED_PROJECTION_DISTANCE_M, NearestSegmentTiePolicyV1, PATH_REFERENCE_CONFIG_V1,
    PathReferenceBuildError, PathReferenceBuilderV1, PathReferenceConfigParseError,
    PathReferenceConfigV1, PathReferenceConfigV1Dto,
};
pub use safety::{
    BoundPredictedTrajectory, FinalValidationMismatch, SafetyControllerDecision,
    SafetyDecideError, SafetyDecision, SafetyDecisionOutcome, SafetyFatalError,
    SafetyNotReadyReason, SafetyReadyTick, SafetySolverFailure, SafetySolverRequestContext,
    SafetyStopCause, SafetyStoppedDecision, SafetySupervisorCreateError, SafetyTickInput,
    ShadowSafetySupervisor, SolverBudgetError, SolverBudgetNs,
};
pub use shadow_command::{
    MAX_SHADOW_COMMAND_RECORDS, MotorPacketsSent, ShadowCommandConfig, ShadowCommandConfigDto,
    ShadowCommandConfigError, ShadowCommandDisposition, ShadowCommandError, ShadowCommandRecord,
    ShadowCommandSession, ShadowDecisionId, ShadowPwmPair, ShadowPwmPairError,
};
