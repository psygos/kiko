//! Transport-independent host boundary for Kiko's expression pipeline.
//!
//! This crate owns four narrow responsibilities:
//!
//! - deterministic scene-motion extraction from an already checked, borrowed
//!   RGB frame;
//! - typed, transport-free OAK-camera to neutral-head gaze geometry;
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
mod gaze_geometry;
mod scene_motion;
mod session;

pub use adapter::{
    AdaptError, EyeRenderStyle, PreparedEyeIntent, adapt_eye_intention, adapt_reaction_output,
    map_expression,
};
pub use gaze_geometry::{
    CameraForwardDepthMeters, CameraGazeTargetError, CameraToHeadGazeExtrinsics,
    CameraToHeadGazeExtrinsicsInput, CartesianAxis, GazeExtrinsicsParseError, HeadGazeAngle,
    HeadGazeProjectionError, HeadRelativeGaze, MAX_CAMERA_FORWARD_DEPTH_M,
    MAX_CAMERA_TARGET_AXIS_ABS_M, MAX_HEAD_ORIGIN_DISTANCE_M, MIN_CAMERA_FORWARD_DEPTH_M,
    OakCameraTargetPoint, OakCameraTargetRay, QuaternionComponent, RayHeadGazeProjectionError,
};
pub use scene_motion::{
    MAX_SCENE_SAMPLES, MonotonicLatestAdmission, MonotonicLatestGap, MonotonicLatestSceneAnalysis,
    MotionThresholds, SamplingGeometry, SamplingGeometryError, SceneAnalysis, SceneMotionConfig,
    SceneMotionConfigError, SceneMotionError, SceneMotionExtractor,
};
pub use session::{
    ControlBinding, ExpectedEyeIdentity, EyeSession, EyeSessionFault, EyeSessionFaultKind,
    EyeSessionPlan, FirmwareAdmission, InboundMessageKind, REQUIRED_EYE_CAPABILITIES, SessionEvent,
    SessionNonce, SessionPhase, SessionPlanError,
};
