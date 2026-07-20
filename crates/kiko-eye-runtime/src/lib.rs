//! Exclusive, bounded host-side serial ownership for Kiko's KEP2 eyes.
//!
//! Weak deployment configuration is parsed once into
//! [`StaticEyeRuntimeConfig`]. A per-start [`EyeSessionMaterialGenerator`] must
//! then produce one-shot session material before the non-cloneable
//! [`EyeRuntimeConfig`] can be constructed. A single non-cloneable
//! [`EyeActorHandle`] drives the resulting exact identity challenge, control
//! acquisition, one-in-flight intent admission, and explicit release over one
//! exclusively owned serial transport. Every fault enters the
//! transport-independent session's fallback state and records whether its
//! best-effort release was unavailable, written, or failed.
//! The injected [`MonotonicClock`] must be the same clock epoch used to stamp
//! every [`kiko_expression_runtime::PreparedEyeIntent`].
//!
//! [`FirmwareAdmissionEvidence`] is intentionally not display evidence. KEP2
//! reports firmware admission and a renderer sequence; this crate has no
//! sensor that observes photons or proves that pixels were physically visible.

#![forbid(unsafe_code)]

mod actor;
mod config;
mod framing;
mod identity_probe;
mod transport;

pub use actor::{
    ActorExit, ActorTermination, CancellationCause, CleanupOutcome, EyeActorHandle,
    EyeActorSpawnError, EyeActorStartError, EyeActorTask, EyeRuntimeFault,
    FirmwareAdmissionEvidence, FrameWriteError, FrameWriteEvidence, FrameWriteFailure,
    HandleRequestError, PriorReleaseAttempt, ProtocolExchange, ReleaseEvidence, ReleaseReport,
    RuntimeFaultCause, StartupEvidence, StartupReceipt, StartupReceiptError, spawn_eye_actor,
    start_serial_eye_actor,
};

pub use config::{
    BaudRate, ConfigParseError, DeviceIdentity, DeviceIdentityKind, EyeRuntimeConfig,
    EyeSessionMaterial, EyeSessionMaterialError, EyeSessionMaterialGenerator,
    EyeSessionMaterialInput, OperationTimeout, OsEyeSessionMaterialError,
    OsEyeSessionMaterialGenerator, StaticEyeRuntimeConfig, StaticEyeRuntimeConfigInput,
    WriteAttemptLimit,
};
pub use framing::{FrameReadError, MAX_READ_CHUNK_BYTES};
pub use identity_probe::{
    EyeIdentityObservation, IdentityProbeConfig, IdentityProbeConfigError,
    IdentityProbeConfigInput, IdentityProbeError, IdentityQueryWriteError,
    probe_serial_eye_identity,
};
pub use transport::{
    AsyncByteTransport, ClockError, MonotonicClock, SerialConfigurationEvidence, SerialOpenError,
    SerialSetting, TokioClock, TransportFailure, TransportFailureKind, TransportOperation,
};

#[cfg(test)]
mod tests;
