//! Exact expression-core to KEP2 eye-intent mapping.

use core::fmt;

use kiko_expression_core::{
    Deadline, ExpressionKind, EyeIntention, HeadIntention, MonotonicTimestamp, ReactionMode,
    ReactionOutput, UnitAmount as CoreUnitAmount,
};
use kiko_eye_protocol::{
    DomainError, Expression, EyeFlags, EyeIntent, NORMALIZED_SCALE, SignedUnit,
    UnitAmount as KepUnitAmount,
};

const CORE_SCALE: i32 = CoreUnitAmount::ONE.basis_points() as i32;
const KEP2_SCALE: i32 = NORMALIZED_SCALE as i32;
const SCALE_DIVISOR: i32 = CORE_SCALE / KEP2_SCALE;

/// Rendering fields that are intentionally not produced by the reaction mixer.
///
/// Brightness is expressed on the expression core's 10,000-point unit scale
/// and is converted with the same checked rounding as other unit quantities.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EyeRenderStyle {
    brightness: CoreUnitAmount,
    color_rgb: [u8; 3],
    blink: bool,
}

impl EyeRenderStyle {
    pub const fn new(brightness: CoreUnitAmount, color_rgb: [u8; 3], blink: bool) -> Self {
        Self {
            brightness,
            color_rgb,
            blink,
        }
    }

    pub const fn brightness(self) -> CoreUnitAmount {
        self.brightness
    }

    pub const fn color_rgb(self) -> [u8; 3] {
        self.color_rgb
    }

    pub const fn blink(self) -> bool {
        self.blink
    }
}

/// A KEP2 intent that remains bound to its host-side reaction freshness.
///
/// The device lease is deliberately not stored here: KEP2 leases begin on the
/// device clock when firmware admits a command, while this deadline is on the
/// process-local host clock. [`crate::EyeSession`] keeps those two clocks and
/// their meanings separate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PreparedEyeIntent {
    intent: EyeIntent,
    generated_at: MonotonicTimestamp,
    valid_until_exclusive: Option<Deadline>,
}

impl PreparedEyeIntent {
    #[cfg(test)]
    pub(crate) const fn from_parts(
        intent: EyeIntent,
        generated_at: MonotonicTimestamp,
        valid_until_exclusive: Option<Deadline>,
    ) -> Self {
        Self {
            intent,
            generated_at,
            valid_until_exclusive,
        }
    }

    pub const fn intent(self) -> EyeIntent {
        self.intent
    }

    /// Replace only the already parsed KEP2 rendering intent while preserving
    /// the exact host freshness carried by this reaction.
    ///
    /// This is the narrow composition seam for deterministic autonomic eye
    /// animation. [`EyeIntent`] construction has already made every numeric
    /// field and protocol enum valid; callers cannot retag the source clock or
    /// extend the reaction lifetime through this method.
    pub const fn with_intent(self, intent: EyeIntent) -> Self {
        Self { intent, ..self }
    }

    pub const fn generated_at(self) -> MonotonicTimestamp {
        self.generated_at
    }

    pub const fn valid_until_exclusive(self) -> Option<Deadline> {
        self.valid_until_exclusive
    }

    pub(crate) const fn is_fresh_at(self, now: MonotonicTimestamp) -> bool {
        if now.nanos_since_epoch() < self.generated_at.nanos_since_epoch() {
            return false;
        }
        match self.valid_until_exclusive {
            Some(deadline) => deadline.is_alive_at(now),
            None => true,
        }
    }
}

/// Map a checked eye intention to KEP2.
///
/// Coordinate contract:
///
/// - expression-core `gaze_x_right`: positive means image/right;
/// - expression-core `gaze_y_down`: positive means image/down;
/// - the source renderer's KEP2 physical plane uses positive Y upward, so Y is
///   negated while X is preserved;
/// - 10,000 core basis points map to 1,000 KEP2 units, rounding to nearest with
///   exact half cases away from zero for signed values;
/// - core `openness` is inverted because the source eye renderer's `lid` is
///   closure (`0=open`, `1000=closed`); and
/// - pupil dilation and brightness are direct unit-scale mappings.
///
/// Panel mounting polarity is firmware/build calibration. This host mapping
/// never guesses or silently flips a physical eye.
pub fn adapt_eye_intention(
    eyes: EyeIntention,
    semantic_expression: ExpressionKind,
    style: EyeRenderStyle,
) -> Result<EyeIntent, AdaptError> {
    let gaze_x = SignedUnit::try_new(scale_signed(eyes.gaze_x_right().basis_points()))?;
    let gaze_y = SignedUnit::try_new(-scale_signed(eyes.gaze_y_down().basis_points()))?;
    let openness = scale_unit(eyes.openness().basis_points());
    let lid_closure = u16::try_from(KEP2_SCALE)
        .expect("KEP2 normalized scale is a positive u16")
        .checked_sub(openness)
        .expect("scaled openness is bounded by the KEP2 scale");
    let lid = KepUnitAmount::try_new(lid_closure)?;
    let pupil = KepUnitAmount::try_new(scale_unit(eyes.pupil_dilation().basis_points()))?;
    let brightness = KepUnitAmount::try_new(scale_unit(style.brightness.basis_points()))?;
    let flags = EyeFlags::try_from_bits(if style.blink { EyeFlags::BLINK } else { 0 })?;

    Ok(EyeIntent::new(
        gaze_x,
        gaze_y,
        lid,
        pupil,
        brightness,
        map_expression(semantic_expression),
        flags,
        style.color_rgb,
    ))
}

/// Convert one complete reaction while preserving its host freshness.
///
/// This crate does not own head hardware. An offset-bearing output is rejected
/// instead of silently dropping an actuation request. Neutral fallback always
/// maps to the neutral KEP2 expression, regardless of the caller's semantic
/// decoration.
pub fn adapt_reaction_output(
    output: ReactionOutput,
    semantic_expression: ExpressionKind,
    style: EyeRenderStyle,
    now: MonotonicTimestamp,
) -> Result<PreparedEyeIntent, AdaptError> {
    if !matches!(output.head(), HeadIntention::NaturalHold) {
        return Err(AdaptError::HeadMotionNotOwned);
    }
    if now.nanos_since_epoch() < output.generated_at().nanos_since_epoch() {
        return Err(AdaptError::ReactionFromFuture {
            generated_at_ns: output.generated_at().nanos_since_epoch(),
            now_ns: now.nanos_since_epoch(),
        });
    }
    if let Some(deadline) = output.valid_until_exclusive()
        && !deadline.is_alive_at(now)
    {
        return Err(AdaptError::ReactionStale {
            deadline_ns: deadline.timestamp().nanos_since_epoch(),
            now_ns: now.nanos_since_epoch(),
        });
    }

    let semantic_expression = match output.mode() {
        ReactionMode::NeutralFallback => ExpressionKind::Neutral,
        ReactionMode::Active => semantic_expression,
    };
    let intent = adapt_eye_intention(output.eyes(), semantic_expression, style)?;
    Ok(PreparedEyeIntent {
        intent,
        generated_at: output.generated_at(),
        valid_until_exclusive: output.valid_until_exclusive(),
    })
}

/// Project the richer expression-core vocabulary onto KEP2's finite renderer
/// vocabulary. The projection is deliberate and exhaustive.
pub const fn map_expression(kind: ExpressionKind) -> Expression {
    match kind {
        ExpressionKind::Neutral | ExpressionKind::Calm => Expression::Neutral,
        ExpressionKind::Attentive | ExpressionKind::Curious => Expression::Curious,
        ExpressionKind::Friendly => Expression::Greet,
        ExpressionKind::Concerned => Expression::Concerned,
    }
}

fn scale_signed(value: i16) -> i16 {
    let value = i32::from(value);
    let rounded = if value >= 0 {
        (value + SCALE_DIVISOR / 2) / SCALE_DIVISOR
    } else {
        (value - SCALE_DIVISOR / 2) / SCALE_DIVISOR
    };
    i16::try_from(rounded).expect("bounded core signed amount fits KEP2 i16")
}

fn scale_unit(value: u16) -> u16 {
    let rounded = (u32::from(value) + u32::try_from(SCALE_DIVISOR / 2).expect("positive divisor"))
        / u32::try_from(SCALE_DIVISOR).expect("positive divisor");
    u16::try_from(rounded).expect("bounded core unit amount fits KEP2 u16")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdaptError {
    ProtocolDomain(DomainError),
    HeadMotionNotOwned,
    ReactionFromFuture { generated_at_ns: u64, now_ns: u64 },
    ReactionStale { deadline_ns: u64, now_ns: u64 },
}

impl From<DomainError> for AdaptError {
    fn from(value: DomainError) -> Self {
        Self::ProtocolDomain(value)
    }
}

impl fmt::Display for AdaptError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "cannot adapt expression output to KEP2: {self:?}"
        )
    }
}

impl core::error::Error for AdaptError {}

#[cfg(test)]
mod tests {
    extern crate std;

    use kiko_expression_core::{
        ExpressionIntent, ExpressionPriority, FreshnessWindow, GazeTarget, HeadMotionPolicy,
        ImagePoint, NonZeroDuration, PositiveUnitAmount, ReactionInputs, ReactionMixer,
        SignedUnitAmount, UnitAmount,
    };

    use super::*;

    fn style() -> EyeRenderStyle {
        EyeRenderStyle::new(
            UnitAmount::try_from_basis_points(7_555).expect("brightness"),
            [1, 2, 3],
            false,
        )
    }

    fn active_output(head_motion: HeadMotionPolicy) -> ReactionOutput {
        let freshness = FreshnessWindow::from_ttl(
            MonotonicTimestamp::from_nanos_since_epoch(10),
            NonZeroDuration::try_from_nanos(100).expect("ttl"),
        )
        .expect("freshness");
        let intent = ExpressionIntent::new(
            ExpressionKind::Friendly,
            PositiveUnitAmount::ONE,
            ExpressionPriority::Normal,
            Some(GazeTarget::new(ImagePoint::new(
                UnitAmount::try_from_basis_points(10_000).expect("x"),
                UnitAmount::try_from_basis_points(0).expect("y"),
            ))),
            freshness,
        );
        ReactionMixer::new(head_motion).mix(
            MonotonicTimestamp::from_nanos_since_epoch(10),
            ReactionInputs {
                rgb: None,
                people: &[],
                scene: None,
                intents: &[intent],
            },
        )
    }

    #[test]
    fn every_core_basis_point_scales_in_range_with_symmetric_rounding() {
        for value in -10_000_i16..=10_000 {
            let scaled = scale_signed(value);
            assert!((-1_000..=1_000).contains(&scaled));
            assert_eq!(scale_signed(-value), -scaled);
        }
        assert_eq!(scale_signed(4), 0);
        assert_eq!(scale_signed(5), 1);
        assert_eq!(scale_signed(-5), -1);

        for value in 0_u16..=10_000 {
            assert!(scale_unit(value) <= 1_000);
        }
        assert_eq!(scale_unit(4), 0);
        assert_eq!(scale_unit(5), 1);
    }

    #[test]
    fn mapping_is_exhaustive_and_lid_is_openness_inverted_to_closure() {
        assert_eq!(map_expression(ExpressionKind::Neutral), Expression::Neutral);
        assert_eq!(
            map_expression(ExpressionKind::Attentive),
            Expression::Curious
        );
        assert_eq!(map_expression(ExpressionKind::Friendly), Expression::Greet);
        assert_eq!(map_expression(ExpressionKind::Curious), Expression::Curious);
        assert_eq!(
            map_expression(ExpressionKind::Concerned),
            Expression::Concerned
        );
        assert_eq!(map_expression(ExpressionKind::Calm), Expression::Neutral);

        let intent = adapt_eye_intention(EyeIntention::neutral(), ExpressionKind::Neutral, style())
            .expect("neutral maps");
        assert_eq!(intent.lid().get(), 200);
        assert_eq!(intent.pupil().get(), 500);
        assert_eq!(intent.brightness().get(), 756);
        assert_eq!(intent.color_rgb(), [1, 2, 3]);
    }

    #[test]
    fn reaction_adapter_preserves_axes_and_freshness_and_forces_fallback_neutral() {
        let output = active_output(HeadMotionPolicy::NaturalHold);
        let prepared = adapt_reaction_output(
            output,
            ExpressionKind::Friendly,
            style(),
            MonotonicTimestamp::from_nanos_since_epoch(10),
        )
        .expect("fresh output");
        assert_eq!(prepared.intent().gaze_x().get(), 1_000);
        assert_eq!(prepared.intent().gaze_y().get(), 1_000);
        assert_eq!(prepared.intent().expression(), Expression::Greet);
        assert_eq!(
            prepared.valid_until_exclusive(),
            output.valid_until_exclusive()
        );

        let fallback = ReactionMixer::default().mix(
            MonotonicTimestamp::from_nanos_since_epoch(20),
            ReactionInputs::empty(),
        );
        let fallback = adapt_reaction_output(
            fallback,
            ExpressionKind::Concerned,
            style(),
            MonotonicTimestamp::from_nanos_since_epoch(20),
        )
        .expect("fallback maps");
        assert_eq!(fallback.intent().expression(), Expression::Neutral);
    }

    #[test]
    fn stale_future_and_head_offset_outputs_are_rejected() {
        let output = active_output(HeadMotionPolicy::NaturalHold);
        assert!(matches!(
            adapt_reaction_output(
                output,
                ExpressionKind::Friendly,
                style(),
                MonotonicTimestamp::from_nanos_since_epoch(9)
            ),
            Err(AdaptError::ReactionFromFuture { .. })
        ));
        assert!(matches!(
            adapt_reaction_output(
                output,
                ExpressionKind::Friendly,
                style(),
                MonotonicTimestamp::from_nanos_since_epoch(110)
            ),
            Err(AdaptError::ReactionStale { .. })
        ));

        let output = active_output(HeadMotionPolicy::FollowGaze {
            gain: UnitAmount::try_from_basis_points(10_000).expect("gain"),
        });
        assert_eq!(
            adapt_reaction_output(
                output,
                ExpressionKind::Friendly,
                style(),
                MonotonicTimestamp::from_nanos_since_epoch(10)
            ),
            Err(AdaptError::HeadMotionNotOwned)
        );
    }

    #[test]
    fn signed_source_domain_is_exact() {
        for value in -10_000_i16..=10_000 {
            let value = SignedUnitAmount::try_from_basis_points(value).expect("source value");
            assert_eq!(
                scale_signed(value.basis_points()),
                SignedUnit::try_new(scale_signed(value.basis_points()))
                    .expect("KEP2 value")
                    .get()
            );
        }
    }
}
