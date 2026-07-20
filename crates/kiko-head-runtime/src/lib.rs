#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

mod actor;
mod config;
mod framing;
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
    ArmingFreshness, ConfigParseError, DeviceIdentity, DeviceIdentityKind, HeadRuntimeConfig,
    HeadRuntimeConfigInput, OperationTimeout, WriteAttemptLimit,
};
pub use framing::{FrameReadError, MAX_RESPONSE_BYTES};
pub use transport::{
    AsyncByteTransport, MonotonicClock, MonotonicTime, SerialConfigurationEvidence,
    SerialOpenError, SerialSetting, TokioClock, TransportFailure, TransportFailureKind,
    TransportOperation,
};
