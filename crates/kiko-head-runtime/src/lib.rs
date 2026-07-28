#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

mod actor;
mod config;
mod framing;
pub mod gaze_control;
mod motion;
mod probe;
mod transport;

pub use actor::{
    ActorExit, ActorTermination, ArmingFreshnessCheck, CancellationCause, FrameWriteError,
    HeadActorHandle, HeadActorSpawnError, HeadActorStartError, HeadActorTask, HeadCommandError,
    HeadGazeActuationConfig, HeadGazeActuationConfigError, HeadGazeBaseZeroExclusiveLease,
    HeadGazeHardwareApplication, HeadGazeProposalCommandError, HeadGazeServiceError,
    HeadGazeServiceOutcome, HeadGoalRegisterBoundary, HeadGoalRegisterBoundaryEvidence,
    HeadGoalRegisterError, HeadGoalRegisterFailure, HeadHealthCheckError, HeadHealthClockBoundary,
    HeadHealthFailure, HeadHealthJointEvidence, HeadHealthObservationError, HeadHealthRequestError,
    HeadHoldTarget, HeadReturnActorHandle, HeadReturnError, HeadRuntimeError,
    HeadStartupTorqueEvidence, HeadTelemetrySetEvidence, HeadWaypointBatchFailure,
    HeadWaypointBatchWriteError, HeadWaypointEvidence, HoldPreservingOwnershipReleaseEvidence,
    InterruptedTelemetryRead, PhysicalHeadMotionConsent, PhysicalTorqueEnableConsent,
    PositionObservationEvidence, ProductionTensionPreservingTakeoverConsent, ReadbackEvidence,
    RequestError, ResponseEvidence, RuntimeStage, ShutdownError, StartupReceipt,
    StartupReceiptError, TensionPreservingHeadActorExit, TensionPreservingHeadActorTask,
    TensionPreservingHeadGazeActorHandle, TensionPreservingHeadReturnActorHandle,
    TorqueDisableJointOutcome, TorqueDisableReport, VerificationSample,
    VerifiedHeadGazeControlStep, VerifiedHeadGoalRegisterEvidence, VerifiedHeadHealthEvidence,
    VerifiedHeadReturnEvidence, VerifiedNaturalHoldEvidence, WriteEvidence, WritePurpose,
    spawn_head_actor, spawn_head_return_actor, spawn_tension_preserving_head_gaze_actor,
    spawn_tension_preserving_head_return_actor, start_serial_head_actor,
    start_serial_head_return_actor, start_serial_tension_preserving_head_gaze_actor,
    start_serial_tension_preserving_head_return_actor,
};
pub use config::{
    ArmingFreshness, ConfigParseError, ConfiguredHeadPoseBound, ConfiguredHeadPoseBounds,
    ConfiguredHeadPoseBoundsError, DeviceIdentity, DeviceIdentityKind,
    HEAD_PRE_ENABLE_TELEMETRY_MAXIMUM_AGE, HEAD_RETURN_CONTROL_PERIOD, HEAD_RETURN_MOTION_TIMEOUT,
    HEAD_RETURN_NO_PROGRESS_TIMEOUT, HEAD_RETURN_POSITION_STEP_TICKS,
    HEAD_RETURN_TELEMETRY_SET_MAX_AGE, HeadPoseBoundsAdmissionError,
    HeadPoseWithinConfiguredBounds, HeadProbeConfig, HeadProbeConfigInput, HeadRuntimeConfig,
    HeadRuntimeConfigInput, HeadTelemetrySafetyLimits, HeadTelemetrySafetyViolation,
    KIKO_MAXIMUM_ENERGIZED_HEAD_TEMPERATURE_RAW_EXCLUSIVE, KIKO_MAXIMUM_HEAD_VOLTAGE_RAW_INCLUSIVE,
    KIKO_MAXIMUM_PRE_TORQUE_HEAD_TEMPERATURE_RAW_INCLUSIVE,
    KIKO_MINIMUM_HEAD_VOLTAGE_RAW_INCLUSIVE, MAX_CONFIGURED_POSE_WINDOW_SPAN_TICKS,
    MAX_HEAD_RETURN_TRAVEL_TICKS, ObservedHoldConfig, ObservedHoldConfigInput,
    ObservedHoldConfigParseError, OperationTimeout, ReturnToTargetConfig,
    ReturnToTargetConfigInput, ReturnToTargetConfigParseError, WriteAttemptLimit,
};
pub use framing::{FrameReadError, MAX_RESPONSE_BYTES};
pub use motion::HeadMotionError;
pub use probe::{
    HeadProbeError, HeadProbeReport, ProbeRequest, ProbeResponseEvidence, SerialHeadProbeError,
    ServoProbeReport, probe_serial_head,
};
pub use transport::{
    AsyncByteTransport, MonotonicClock, MonotonicTime, SerialConfigurationEvidence,
    SerialOpenError, SerialSetting, TokioClock, TransportFailure, TransportFailureKind,
    TransportOperation,
};
