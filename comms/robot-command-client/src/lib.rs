#![forbid(unsafe_code)]
#![doc = include_str!("../README.md")]

mod client;
mod config;
mod domain;
mod transport;

pub mod fake;

pub use client::{
    AcquireFailure, ApplyFailure, ArmedCommandClient, DisarmFailure, DisarmedCommandClient,
    EvidenceError, FailureCause, LatchedCommandClient, LatchedStopKnowledge,
    RecoveryAttemptFailure, StopRecoveryReport,
};
pub use config::{
    ClientConfig, ClientConfigInput, ConfigError, MAX_IO_TIMEOUT_NS, MAX_STOP_RECOVERY_ATTEMPTS,
    StopRecoveryPolicy, TimeoutNs, UdpEndpoint,
};
pub use domain::{
    AppliedCommandReceipt, ControllerSession, DisarmReceipt, MonotonicInstant,
    PendingPhysicalCommand,
};
pub use robot_protocol::v2::{
    ActuatorConfigFingerprint, ControlEpoch, ControllerBootId, ControllerUid, TimerPwm,
    V2CommandLeaseMs, V2CommandSequence,
};
pub use transport::{
    MonotonicClock, RobotProtocolV2CodecError, RobotProtocolV2WireAdapter, SystemMonotonicClock,
    UdpTransportBuildError, UdpTransportError, UdpTransportPhase, UdpV2Transport,
    V2CommandTransport, V2WireAdapter,
};
