//! Deterministic expression-domain types and reaction mixing for Kiko.
//!
//! This crate is intentionally `no_std`, performs no I/O, allocates no memory,
//! and contains no hardware calibration. Boundary adapters parse camera and
//! agent data into the checked types here exactly once. Hardware actors then
//! map the bounded, normalized output into separately qualified servo or eye
//! firmware commands.
//!
//! The default [`ReactionMixer`] always emits [`HeadIntention::NaturalHold`].
//! This lets visual expressions move Kiko's eyes without quietly authorizing
//! neck motion. Expressive head following is an explicit policy selection and
//! is still limited to normalized offsets; physical joint limits remain the
//! responsibility of the head actor.

#![no_std]
#![forbid(unsafe_code)]

mod amount;
mod intent;
mod mixer;
mod observation;
mod time;

pub use amount::{AmountError, PositiveUnitAmount, SignedUnitAmount, UnitAmount};
pub use intent::{
    ExpressionIntent, ExpressionKind, ExpressionPriority, GazeTarget, HeadMotionPolicy,
};
pub use mixer::{
    EyeIntention, HeadIntention, HeadOffset, ReactionInputs, ReactionMixer, ReactionMode,
    ReactionOutput, VisualReactionSource,
};
pub use observation::{
    ChannelOrder, DistanceMillimeters, FrameId, ImageLayout, ImageLayoutError, ImagePoint,
    ObservationValueError, PersonObservation, PersonTrackId, RgbFrameView, RgbObservation,
    SceneMotion, SceneObservation, StreamEpochId,
};
pub use time::{Deadline, FreshnessWindow, MonotonicTimestamp, NonZeroDuration, TimeError};
