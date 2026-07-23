#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

mod actor;
mod config;
mod framing;
mod motion;
mod probe;
mod transport;

pub use actor::{
    ActorExit, ActorTermination, ArmingFreshnessCheck, CancellationCause, FrameWriteError,
    HeadActorHandle, HeadActorSpawnError, HeadActorStartError, HeadActorTask, HeadCommandError,
    HeadHealthCheckError, HeadHealthClockBoundary, HeadHealthFailure, HeadHealthJointEvidence,
    HeadHealthObservationError, HeadHealthRequestError, HeadReturnActorHandle, HeadReturnError,
    HeadRuntimeError, HeadTelemetrySetEvidence, HeadWaypointBatchFailure,
    HeadWaypointBatchWriteError, HeadWaypointEvidence, InterruptedTelemetryRead,
    PhysicalHeadMotionConsent, PhysicalTorqueEnableConsent, PositionObservationEvidence,
    ReadbackEvidence, RequestError, ResponseEvidence, RuntimeStage, ShutdownError, StartupReceipt,
    StartupReceiptError, TorqueDisableJointOutcome, TorqueDisableReport, VerificationSample,
    VerifiedHeadHealthEvidence, VerifiedHeadReturnEvidence, VerifiedNaturalHoldEvidence,
    WriteEvidence, WritePurpose, spawn_head_actor, spawn_head_return_actor,
    start_serial_head_actor, start_serial_head_return_actor,
};
pub use config::{
    ArmingFreshness, ConfigParseError, ConfiguredHeadPoseBound, ConfiguredHeadPoseBounds,
    ConfiguredHeadPoseBoundsError, DeviceIdentity, DeviceIdentityKind, HEAD_RETURN_CONTROL_PERIOD,
    HEAD_RETURN_MOTION_TIMEOUT, HEAD_RETURN_NO_PROGRESS_TIMEOUT, HEAD_RETURN_POSITION_STEP_TICKS,
    HEAD_RETURN_TELEMETRY_SET_MAX_AGE, HeadPoseBoundsAdmissionError,
    HeadPoseWithinConfiguredBounds, HeadProbeConfig, HeadProbeConfigInput, HeadRuntimeConfig,
    HeadRuntimeConfigInput, MAX_CONFIGURED_POSE_WINDOW_SPAN_TICKS, MAX_HEAD_RETURN_TRAVEL_TICKS,
    ObservedHoldConfig, ObservedHoldConfigInput, ObservedHoldConfigParseError, OperationTimeout,
    ReturnToTargetConfig, ReturnToTargetConfigInput, ReturnToTargetConfigParseError,
    WriteAttemptLimit,
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
