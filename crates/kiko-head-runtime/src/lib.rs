#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

mod actor;
mod config;
mod framing;
mod probe;
mod transport;

pub use actor::{
    ActorExit, ActorTermination, ArmingFreshnessCheck, CancellationCause, FrameWriteError,
    HeadActorHandle, HeadActorSpawnError, HeadActorStartError, HeadActorTask, HeadRuntimeError,
    PhysicalTorqueEnableConsent, PositionObservationEvidence, ReadbackEvidence, RequestError,
    ResponseEvidence, RuntimeStage, ShutdownError, StartupReceipt, StartupReceiptError,
    TorqueDisableJointOutcome, TorqueDisableReport, VerificationSample,
    VerifiedNaturalHoldEvidence, WriteEvidence, WritePurpose, spawn_head_actor,
    start_serial_head_actor,
};
pub use config::{
    ArmingFreshness, ConfigParseError, ConfiguredHeadPoseBound, ConfiguredHeadPoseBounds,
    ConfiguredHeadPoseBoundsError, DeviceIdentity, DeviceIdentityKind,
    HeadPoseBoundsAdmissionError, HeadPoseWithinConfiguredBounds, HeadProbeConfig,
    HeadProbeConfigInput, HeadRuntimeConfig, HeadRuntimeConfigInput,
    MAX_CONFIGURED_POSE_WINDOW_SPAN_TICKS, ObservedHoldConfig, ObservedHoldConfigInput,
    ObservedHoldConfigParseError, OperationTimeout, WriteAttemptLimit,
};
pub use framing::{FrameReadError, MAX_RESPONSE_BYTES};
pub use probe::{
    HeadProbeError, HeadProbeReport, ProbeRequest, ProbeResponseEvidence, SerialHeadProbeError,
    ServoProbeReport, probe_serial_head,
};
pub use transport::{
    AsyncByteTransport, MonotonicClock, MonotonicTime, SerialConfigurationEvidence,
    SerialOpenError, SerialSetting, TokioClock, TransportFailure, TransportFailureKind,
    TransportOperation,
};
