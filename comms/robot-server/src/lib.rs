#![forbid(unsafe_code)]

//! Host-side V2 controller ownership for Kiko's sole in-process Nano runtime
//! and bounded identity/transport qualification tools.

mod actuation_v2;
pub mod config;
mod controller_owner;
mod deadline;

pub use actuation_v2::{
    ActuationActorError, ActuationFaultEvidence, ActuationOutputEvidence, ActuationSnapshot,
    ActuationStartError, ActuationStartupPhase, ActuationTelemetry, ControllerHelloRejection,
    ControllerReadyRejection, NoopActuationTelemetry, ObservedActuationOutput,
    PreSessionHeartbeatRejection, SerialResynchronizationOutcome, SerialTransmitError,
    SerialTransmitInterruption, SerialTransmitPhase, SessionHeartbeatRejection,
    ShutdownForceStopOutcome, ShutdownInterruptedTransmitRecovery, StopFaultEvidence,
    UdpServiceError,
};
#[cfg(feature = "qualification-fault-injection")]
pub use actuation_v2::{
    OperatorSupervisedCandidateSerialFaultInjection, QualificationPartialUartPrefixTransmitOutcome,
    QualificationPartialUartRecordInjectionRecovery, QualificationPartialUartRecordPrefixError,
};
pub use controller_owner::{
    ActuationTaskOutcome, UdpTaskOutcome, V2ControllerOwner, V2ControllerOwnerExitTrigger,
    V2ControllerOwnerStartCleanup, V2ControllerOwnerStartError, V2ControllerOwnerTerminationError,
};
