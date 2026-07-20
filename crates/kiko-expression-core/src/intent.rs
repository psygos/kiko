//! High-level, expiring expression requests.

use crate::{FreshnessWindow, ImagePoint, PositiveUnitAmount, UnitAmount};

/// Semantic expression requested by a director or deterministic behavior.
///
/// These are intentionally not servo poses or eye-firmware opcodes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ExpressionKind {
    #[default]
    Neutral,
    Attentive,
    Friendly,
    Curious,
    Concerned,
    Calm,
}

/// Coarse priority prevents an unbounded numeric priority arms race.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ExpressionPriority {
    Background,
    #[default]
    Normal,
    Important,
    Urgent,
}

/// A high-level request with strength and an explicit freshness window.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ExpressionIntent {
    kind: ExpressionKind,
    strength: PositiveUnitAmount,
    priority: ExpressionPriority,
    gaze_target: Option<GazeTarget>,
    freshness: FreshnessWindow,
}

impl ExpressionIntent {
    pub const fn new(
        kind: ExpressionKind,
        strength: PositiveUnitAmount,
        priority: ExpressionPriority,
        gaze_target: Option<GazeTarget>,
        freshness: FreshnessWindow,
    ) -> Self {
        Self {
            kind,
            strength,
            priority,
            gaze_target,
            freshness,
        }
    }

    pub const fn kind(self) -> ExpressionKind {
        self.kind
    }

    pub const fn strength(self) -> PositiveUnitAmount {
        self.strength
    }

    pub const fn priority(self) -> ExpressionPriority {
        self.priority
    }

    pub const fn gaze_target(self) -> Option<GazeTarget> {
        self.gaze_target
    }

    pub const fn freshness(self) -> FreshnessWindow {
        self.freshness
    }
}

/// An attention target in the RGB image coordinate convention.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GazeTarget(ImagePoint);

impl GazeTarget {
    pub const fn new(point: ImagePoint) -> Self {
        Self(point)
    }

    pub const fn point(self) -> ImagePoint {
        self.0
    }
}

/// Whether reaction output may request bounded head offsets.
///
/// `NaturalHold` is the production default. `FollowGaze` only produces a
/// normalized intention; a qualified head actor must still map it through its
/// own versioned physical envelope.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum HeadMotionPolicy {
    #[default]
    NaturalHold,
    FollowGaze {
        gain: UnitAmount,
    },
}
