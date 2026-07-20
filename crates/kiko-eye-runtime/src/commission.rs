//! Fixed, eye-only KEP2 commissioning sequence.
//!
//! This module prepares typed expression intents only. Serial ownership,
//! firmware admission, release, and fallback remain owned by the KEP2 actor.

use std::fmt;
use std::time::Duration;

use kiko_expression_core::{
    AmountError, ExpressionIntent, ExpressionKind, ExpressionPriority, FreshnessWindow, GazeTarget,
    ImagePoint, MonotonicTimestamp, NonZeroDuration, PositiveUnitAmount, ReactionInputs,
    ReactionMixer, TimeError, UnitAmount,
};
use kiko_expression_runtime::{
    AdaptError, EyeRenderStyle, PreparedEyeIntent, adapt_reaction_output,
};

/// Firmware lease used by the commissioning CLI. Every visual hold is
/// deliberately shorter, so an interrupted host returns to firmware fallback
/// no later than this lease after the last admitted intent.
pub const COMMISSIONING_INTENT_LEASE_MS: u16 = 1_800;

/// Number of fixed, finite visual steps in one commissioning run.
pub const COMMISSIONING_STEP_COUNT: usize = 9;

/// Longest requested host-side hold in the fixed recipe.
pub const COMMISSIONING_MAX_HOLD_MS: u64 = 900;

const PREPARED_INTENT_FRESHNESS_MS: u64 = 2_000;
const BRIGHTNESS_BASIS_POINTS: u16 = 10_000;

/// One immutable visual request in the bounded commissioning recipe.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EyeCommissioningStep {
    name: &'static str,
    semantic_expression: ExpressionKind,
    gaze_basis_points: Option<[u16; 2]>,
    color_rgb: [u8; 3],
    blink: bool,
    hold_duration_ms: u64,
}

impl EyeCommissioningStep {
    pub const fn name(self) -> &'static str {
        self.name
    }

    pub const fn semantic_expression(self) -> ExpressionKind {
        self.semantic_expression
    }

    pub const fn hold_duration(self) -> Duration {
        Duration::from_millis(self.hold_duration_ms)
    }

    /// Prepare this fixed step using the same monotonic clock epoch as the
    /// actor which will admit it.
    pub fn prepare(
        self,
        now: MonotonicTimestamp,
    ) -> Result<PreparedEyeIntent, CommissioningPrepareError> {
        let freshness = FreshnessWindow::from_ttl(
            now,
            NonZeroDuration::try_from_nanos(PREPARED_INTENT_FRESHNESS_MS * 1_000_000)
                .map_err(CommissioningPrepareError::Time)?,
        )
        .map_err(CommissioningPrepareError::Time)?;
        let gaze_target = self
            .gaze_basis_points
            .map(|[x, y]| {
                Ok(GazeTarget::new(ImagePoint::new(
                    UnitAmount::try_from_basis_points(x)
                        .map_err(CommissioningPrepareError::Amount)?,
                    UnitAmount::try_from_basis_points(y)
                        .map_err(CommissioningPrepareError::Amount)?,
                )))
            })
            .transpose()?;
        let intent = ExpressionIntent::new(
            self.semantic_expression,
            PositiveUnitAmount::ONE,
            ExpressionPriority::Important,
            gaze_target,
            freshness,
        );
        let output = ReactionMixer::default().mix(
            now,
            ReactionInputs {
                rgb: None,
                people: &[],
                scene: None,
                intents: &[intent],
            },
        );
        let style = EyeRenderStyle::new(
            UnitAmount::try_from_basis_points(BRIGHTNESS_BASIS_POINTS)
                .map_err(CommissioningPrepareError::Amount)?,
            self.color_rgb,
            self.blink,
        );
        adapt_reaction_output(output, self.semantic_expression, style, now)
            .map_err(CommissioningPrepareError::Adapt)
    }
}

/// Return the complete recipe by value. It cannot be extended at runtime.
pub fn eye_commissioning_steps()
-> impl ExactSizeIterator<Item = EyeCommissioningStep> + DoubleEndedIterator {
    STEPS.into_iter()
}

const STEPS: [EyeCommissioningStep; COMMISSIONING_STEP_COUNT] = [
    EyeCommissioningStep {
        name: "white_center",
        semantic_expression: ExpressionKind::Neutral,
        gaze_basis_points: None,
        color_rgb: [255, 255, 255],
        blink: false,
        hold_duration_ms: 800,
    },
    EyeCommissioningStep {
        name: "red_full_left_1",
        semantic_expression: ExpressionKind::Curious,
        gaze_basis_points: Some([0, 5_000]),
        color_rgb: [255, 0, 0],
        blink: false,
        hold_duration_ms: 900,
    },
    EyeCommissioningStep {
        name: "red_full_left_2",
        semantic_expression: ExpressionKind::Curious,
        gaze_basis_points: Some([0, 5_000]),
        color_rgb: [255, 0, 0],
        blink: false,
        hold_duration_ms: 900,
    },
    EyeCommissioningStep {
        name: "blue_full_right_1",
        semantic_expression: ExpressionKind::Friendly,
        gaze_basis_points: Some([10_000, 5_000]),
        color_rgb: [0, 0, 255],
        blink: false,
        hold_duration_ms: 900,
    },
    EyeCommissioningStep {
        name: "blue_full_right_2",
        semantic_expression: ExpressionKind::Friendly,
        gaze_basis_points: Some([10_000, 5_000]),
        color_rgb: [0, 0, 255],
        blink: false,
        hold_duration_ms: 900,
    },
    EyeCommissioningStep {
        name: "white_blink_1",
        semantic_expression: ExpressionKind::Friendly,
        gaze_basis_points: Some([5_000, 5_000]),
        color_rgb: [255, 255, 255],
        blink: true,
        hold_duration_ms: 450,
    },
    EyeCommissioningStep {
        name: "white_blink_2",
        semantic_expression: ExpressionKind::Friendly,
        gaze_basis_points: Some([5_000, 5_000]),
        color_rgb: [255, 255, 255],
        blink: true,
        hold_duration_ms: 450,
    },
    EyeCommissioningStep {
        name: "white_blink_3",
        semantic_expression: ExpressionKind::Friendly,
        gaze_basis_points: Some([5_000, 5_000]),
        color_rgb: [255, 255, 255],
        blink: true,
        hold_duration_ms: 450,
    },
    EyeCommissioningStep {
        name: "neutral_finish",
        semantic_expression: ExpressionKind::Neutral,
        gaze_basis_points: None,
        color_rgb: [64, 160, 255],
        blink: false,
        hold_duration_ms: 800,
    },
];

const _: () = {
    let mut index = 0;
    while index < STEPS.len() {
        assert!(STEPS[index].hold_duration_ms < COMMISSIONING_INTENT_LEASE_MS as u64);
        assert!(STEPS[index].hold_duration_ms <= COMMISSIONING_MAX_HOLD_MS);
        index += 1;
    }
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CommissioningPrepareError {
    Amount(AmountError),
    Time(TimeError),
    Adapt(AdaptError),
}

impl fmt::Display for CommissioningPrepareError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "could not prepare fixed eye commissioning step: {self:?}"
        )
    }
}

impl std::error::Error for CommissioningPrepareError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Amount(source) => Some(source),
            Self::Time(source) => Some(source),
            Self::Adapt(source) => Some(source),
        }
    }
}

#[cfg(test)]
mod tests {
    use kiko_eye_protocol::{Expression, EyeFlags};

    use super::*;

    #[test]
    fn recipe_is_finite_distinct_centered_at_both_ends_and_lease_bounded() {
        let steps: Vec<_> = eye_commissioning_steps().collect();
        assert_eq!(steps.len(), COMMISSIONING_STEP_COUNT);
        assert_eq!(steps.first().map(|step| step.name()), Some("white_center"));
        assert_eq!(steps.last().map(|step| step.name()), Some("neutral_finish"));
        assert!(steps.iter().all(|step| {
            step.hold_duration() < Duration::from_millis(COMMISSIONING_INTENT_LEASE_MS.into())
        }));

        let prepared: Vec<_> = steps
            .into_iter()
            .map(|step| {
                step.prepare(MonotonicTimestamp::from_nanos_since_epoch(1))
                    .expect("fixed recipe parses")
                    .intent()
            })
            .collect();
        assert_eq!(prepared[0].expression(), Expression::Neutral);
        assert_eq!(prepared[0].gaze_x().get(), 0);
        assert_eq!(prepared[1].expression(), Expression::Curious);
        assert_eq!(prepared[1].gaze_x().get(), -1_000);
        assert_eq!(prepared[1].color_rgb(), [255, 0, 0]);
        assert_eq!(prepared[2].gaze_x().get(), -1_000);
        assert_eq!(prepared[3].expression(), Expression::Greet);
        assert_eq!(prepared[3].gaze_x().get(), 1_000);
        assert_eq!(prepared[3].color_rgb(), [0, 0, 255]);
        assert_eq!(prepared[4].gaze_x().get(), 1_000);
        assert!(prepared[5..8]
            .iter()
            .all(|intent| intent.flags().bits() == EyeFlags::BLINK));
        assert_eq!(prepared[8].expression(), Expression::Neutral);
        assert_eq!(prepared[8].gaze_x().get(), 0);
        assert!(prepared
            .iter()
            .all(|intent| intent.brightness().get() == 1_000));
    }

    #[test]
    fn every_prepared_step_has_natural_head_policy_and_fresh_host_time() {
        let now = MonotonicTimestamp::from_nanos_since_epoch(42);
        for step in eye_commissioning_steps() {
            let prepared = step.prepare(now).expect("fixed recipe parses");
            assert_eq!(prepared.generated_at(), now);
            assert!(
                prepared
                    .valid_until_exclusive()
                    .expect("active recipe has a deadline")
                    .is_alive_at(now)
            );
        }
    }
}
