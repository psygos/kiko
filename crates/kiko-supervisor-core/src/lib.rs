#![no_std]
#![forbid(unsafe_code)]

//! Transport-independent robot lifecycle and authority supervision.

mod evidence;
mod state;
mod time;

pub use evidence::{
    ConfirmedBaseZero, EvidenceValueError, ReadinessBinding, ReadinessEpoch, Sha256Digest,
    ZeroEvidenceError,
};
pub use state::{
    AuthorityLease, AuthorityLeaseId, AuthorityMode, FaultKind, RobotSupervisor, StopReason,
    SupervisorAction, SupervisorConfig, SupervisorConfigError, SupervisorError, SupervisorState,
    SupervisorStateKind,
};
pub use time::{AuthorityDuration, MonotonicInstant, TimeValueError};
