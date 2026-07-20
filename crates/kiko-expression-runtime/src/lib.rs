//! Transport-independent host boundary for Kiko's expression pipeline.
//!
//! This crate owns three narrow responsibilities:
//!
//! - deterministic scene-motion extraction from an already checked, borrowed
//!   RGB frame;
//! - an exact semantic and numeric mapping from expression-core eye values to
//!   the KEP2 eye protocol; and
//! - the protocol session state machine above an arbitrary byte transport.
//!
//! It performs no camera, serial, USB, filesystem, logging, head, or display
//! I/O. In particular, a firmware admission report is not evidence that the
//! requested pixels were physically visible.

#![no_std]
#![forbid(unsafe_code)]

mod adapter;
mod scene_motion;
mod session;

pub use adapter::{
    AdaptError, EyeRenderStyle, PreparedEyeIntent, adapt_eye_intention, adapt_reaction_output,
    map_expression,
};
pub use scene_motion::{
    MAX_SCENE_SAMPLES, MotionThresholds, SamplingGeometry, SamplingGeometryError, SceneAnalysis,
    SceneMotionConfig, SceneMotionConfigError, SceneMotionError, SceneMotionExtractor,
};
pub use session::{
    ControlBinding, ExpectedEyeIdentity, EyeSession, EyeSessionFault, EyeSessionFaultKind,
    EyeSessionPlan, FirmwareAdmission, InboundMessageKind, REQUIRED_EYE_CAPABILITIES, SessionEvent,
    SessionNonce, SessionPhase, SessionPlanError,
};
