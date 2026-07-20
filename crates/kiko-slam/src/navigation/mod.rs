//! Hardware-independent navigation contracts and algorithms.

#[cfg(feature = "actuation")]
pub mod actuation;
#[cfg(feature = "actuation")]
mod actuation_config;
mod cell_inflation;
mod coordinator;
mod frames;
mod frontier;
mod global_planner;
mod goal_input;
mod ingress;
mod local_costmap;
mod manual_drive;
pub mod mpc;
mod odometry;
mod reference;
mod safety;
mod shadow_command;
mod shadow_config;

#[cfg(feature = "actuation")]
pub use actuation_config::{
    ActuationConfigParseError, ActuatorConfigFingerprint, ControllerUid,
    MAX_NAVIGATION_ACTUATION_CONFIG_JSON_BYTES, NAVIGATION_ACTUATION_CONFIG_V1,
    NavigationActuationConfigV1, NavigationConfigSha256, OperatorClaimedPhysicalApprovalV1,
};
pub use coordinator::{
    CoordinatorAdmissionError, CoordinatorLatch, CoordinatorTickBlocker, CoordinatorTickError,
    CoordinatorTickOutcome, DepthAdmissionOutcome, GlobalMapAdmissionOutcome,
    GlobalPlanningOutcome, GoalSelectionOutcome, ImuAdmissionOutcome, NavigationGoalState,
    NavigationIngressSink, PlanStartBuildError, ShadowNavigationCoordinator, StoredPlanFault,
    VisualAdmission, VisualAdmissionError, VisualAdmissionOutcome,
};
pub use frames::{
    BaseFrame, BaseToOdom, LocalCostmapFrame, LocalCostmapToOdom, MapFrame, MapToOdom, OdomFrame,
    OdomToBase, OdomToLocalCostmap, OdomToMap, PlanarAxis, PlanarFrame, PlanarPoint,
    PlanarPointError, PlanarTransform, PlanarTransformComponent, PlanarTransformError,
    PlanarTransformOperation,
};
pub use frontier::{
    FrontierBuildError, FrontierExplorer, FrontierExplorerConfig, FrontierExplorerConfigError,
    FrontierGoal, FrontierInPlaceScan, FrontierScore, FrontierSearchError, FrontierSearchOutcome,
    FrontierUnknownDirection, FrontierUnknownDirections,
};
pub use global_planner::{
    GlobalPath, GlobalPlanError, GlobalPlanIdentity, GlobalPlanner, GlobalPlannerConfig,
    GlobalPlannerInstanceId, GlobalPlannerInvocationId, GlobalPlannerRevision, MapPoint, PlanStart,
    PointGoal, UnknownSpacePolicy,
};
pub use goal_input::{
    MapPointGoalSelection, MapPointGoalSelectionDto, MapPointGoalSelectionParseError,
    NavigationGoalArg, NavigationGoalArgError, NavigationGoalAxis,
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
pub use manual_drive::{
    BODY_VELOCITY_TARGET_V1, BodyVelocityTargetV1, MANUAL_DRIVE_COMMAND_V1, MANUAL_DRIVE_CONFIG_V1,
    ManualAuthoritySnapshot, ManualDriveAcceptedIntent, ManualDriveAcceptedTarget,
    ManualDriveCommandDto, ManualDriveCommandKindDto, ManualDriveConfigParseError,
    ManualDriveConfigV1, ManualDriveConfigV1Dto, ManualDriveCore, ManualDriveOutput,
    ManualDriveSequence, ManualDriveStopCause, ManualDriveStopped,
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
    BoundPredictedTrajectory, FinalValidationMismatch, SafetyControllerDecision, SafetyDecideError,
    SafetyDecision, SafetyDecisionOutcome, SafetyFatalError, SafetyNotReadyReason, SafetyReadyTick,
    SafetySolverFailure, SafetySolverRequestContext, SafetyStopCause, SafetyStoppedDecision,
    SafetySupervisorCreateError, SafetyTickInput, ShadowSafetySupervisor, SolverBudgetError,
    SolverBudgetNs,
};
pub use shadow_command::{
    MAX_SHADOW_COMMAND_RECORDS, MotorPacketsSent, ShadowCommandConfig, ShadowCommandConfigDto,
    ShadowCommandConfigError, ShadowCommandDisposition, ShadowCommandError, ShadowCommandRecord,
    ShadowCommandSession, ShadowDecisionId, ShadowPwmPair, ShadowPwmPairError,
};
pub use shadow_config::{
    ControlPeriodNs, FreshnessParameter, MAX_COMMAND_LEASE_CONTROL_PERIODS,
    MAX_SHADOW_NAVIGATION_CONFIG_JSON_BYTES, SHADOW_NAVIGATION_CONFIG_V1,
    ShadowNavigationConfigParseError, ShadowNavigationConfigV1, ShadowNavigationRuntimePartsV1,
};
