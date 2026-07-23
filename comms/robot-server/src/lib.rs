#![forbid(unsafe_code)]

//! Host-side V2 controller ownership shared by the standalone robot-server
//! process and Kiko's in-process Nano runtime.

mod actuation_v2;
pub mod config;
mod controller_owner;
mod deadline;

pub use actuation_v2::{
    unavailable_udp_service, ActuationActorError, ActuationSnapshot, ActuationStartError,
    ActuationTelemetry, NoopActuationTelemetry, UdpServiceError,
};
pub use controller_owner::{
    ActuationTaskOutcome, UdpTaskOutcome, V2ControllerOwner, V2ControllerOwnerExitTrigger,
    V2ControllerOwnerStartError, V2ControllerOwnerTerminationError,
};
