//! Transport-independent host boundary for Kiko's expression pipeline.
//!
//! This crate owns five narrow responsibilities:
//!
//! - deterministic scene-motion extraction from an already checked, borrowed
//!   RGB frame;
//! - bounded, confidence-free face-target association from typed detector
//!   results;
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
mod autonomic;
mod face_tracking;
mod gaze_geometry;
mod head_gaze_calibration;
mod scene_motion;
mod session;

pub use adapter::{
    AdaptError, EyeRenderStyle, PreparedEyeIntent, adapt_eye_intention, adapt_reaction_output,
    map_expression,
};
pub use autonomic::{
    AutonomicCharacterEngine, CHARACTER_HEAD_SCALE, CharacterAct, CharacterAttention,
    CharacterFaceAttention, CharacterFaceAttentionState, CharacterHeadAmount,
    CharacterHeadAmountError, CharacterHeadAxis, CharacterHeadOverlay,
    CharacterHeadOverlayParseError, CharacterInputs, CharacterMode, CharacterPetEpisode,
    CharacterPetEpisodeError, CharacterPetReaction, CharacterPetState, PreparedCharacterFrame,
};
pub use face_tracking::{
    AcquiringFaceTarget, CloserFaceWidthRatio, CoastingFaceTarget, ConsecutiveFaceResults,
    DEFAULT_FACE_ACQUISITION_DISTANCE_PX, DEFAULT_FACE_ACQUISITION_RESULTS,
    DEFAULT_FACE_ASSOCIATION_DISTANCE_PX, DEFAULT_FACE_COASTING_DURATION_NS,
    DEFAULT_FACE_SMOOTHING_ALPHA_BASIS_POINTS, DEFAULT_FACE_SWITCH_RESULTS, DetectorLevelWeight,
    DetectorResultSequence, FaceCoastingDuration, FaceDetection, FaceDetectionBatch,
    FaceDetectionBatchError, FaceDetectionError, FaceDetectorSource, FacePixelDistance,
    FaceRectangle, FaceResultAdmission, FaceTargetState, FaceTracker, FaceTrackingConfig,
    FaceTrackingConfigError, FaceTrackingError, FaceTrackingUpdate, LostFaceTarget,
    MAX_CLOSER_FACE_WIDTH_RATIO, MAX_FACE_COASTING_DURATION_NS, MAX_FACE_CONSECUTIVE_RESULTS,
    MAX_FACE_DETECTIONS, MAX_FACE_PIXEL_DISTANCE_PX, SwitchedFaceTarget, TrackedFaceObservation,
};
pub use gaze_geometry::{
    CameraForwardDepthMeters, CameraGazeTargetError, CameraToHeadGazeExtrinsics,
    CameraToHeadGazeExtrinsicsInput, CartesianAxis, GazeExtrinsicsParseError, HeadGazeAngle,
    HeadGazeProjectionError, HeadRelativeGaze, MAX_CAMERA_FORWARD_DEPTH_M,
    MAX_CAMERA_TARGET_AXIS_ABS_M, MAX_HEAD_ORIGIN_DISTANCE_M, MIN_CAMERA_FORWARD_DEPTH_M,
    OakCameraTargetPoint, OakCameraTargetRay, QuaternionComponent, RayHeadGazeProjectionError,
};
pub use head_gaze_calibration::{
    CameraRayHeadProposalError, CharacterHeadMappingDeclaration,
    CharacterHeadMappingDeclarationParseError, CharacterHeadOverlayMappingError,
    DECLARED_HEAD_ORIGIN_IN_OAK_CAMERA_M, DECLARED_NEUTRAL_HEAD_FROM_OAK_ROTATION_ROWS,
    HEAD_GAZE_FOCUS_PLANE_CAMERA_FORWARD_DEPTH_M, HeadAssemblyId, HeadCalibrationProvenanceId,
    HeadGazeCoordinate, HeadGazeIdentifierError, HeadGazeIdentifierField,
    HeadGazeMappingDeclaration, HeadGazeMappingDeclarationInput,
    HeadGazeMappingDeclarationParseError, HeadGazeProposalMappingError, HeadGazeTargetProposal,
    HeadGazeTickOffsetsPerRadianInput, HeadNaturalPoseDeclaration, HeadTickEnvelope,
    HeadTickEnvelopeBound, HeadTickEnvelopeInput, MAX_HEAD_GAZE_IDENTIFIER_BYTES,
    NamedCharacterHeadFullScaleTickOffsetsInput, NamedHeadTickEnvelopesInput,
    NamedHeadTickOffsetsPerRadianInput, NamedNaturalHeadTicksInput,
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
