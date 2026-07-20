//! Deterministic, allocation-free reaction mixing.

use crate::{
    Deadline, ExpressionIntent, ExpressionKind, ExpressionPriority, FrameId, HeadMotionPolicy,
    ImagePoint, MonotonicTimestamp, PersonObservation, PersonTrackId, RgbObservation,
    SceneObservation, SignedUnitAmount, UnitAmount,
};

/// Inputs sampled for one expression decision.
#[derive(Clone, Copy, Debug)]
pub struct ReactionInputs<'a> {
    pub rgb: Option<&'a RgbObservation>,
    pub people: &'a [PersonObservation],
    pub scene: Option<&'a SceneObservation>,
    pub intents: &'a [ExpressionIntent],
}

impl<'a> ReactionInputs<'a> {
    pub const fn empty() -> Self {
        Self {
            rgb: None,
            people: &[],
            scene: None,
            intents: &[],
        }
    }
}

/// Bounded, hardware-independent eye request.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EyeIntention {
    gaze_x_right: SignedUnitAmount,
    gaze_y_down: SignedUnitAmount,
    openness: UnitAmount,
    pupil_dilation: UnitAmount,
}

impl EyeIntention {
    pub const fn neutral() -> Self {
        Self {
            gaze_x_right: SignedUnitAmount::ZERO,
            gaze_y_down: SignedUnitAmount::ZERO,
            openness: UnitAmount::from_basis_points_proven(8_000),
            pupil_dilation: UnitAmount::from_basis_points_proven(5_000),
        }
    }

    pub const fn gaze_x_right(self) -> SignedUnitAmount {
        self.gaze_x_right
    }

    pub const fn gaze_y_down(self) -> SignedUnitAmount {
        self.gaze_y_down
    }

    pub const fn openness(self) -> UnitAmount {
        self.openness
    }

    pub const fn pupil_dilation(self) -> UnitAmount {
        self.pupil_dilation
    }
}

/// Normalized offsets relative to a separately calibrated natural pose.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct HeadOffset {
    yaw_right: SignedUnitAmount,
    pitch_down: SignedUnitAmount,
    roll_right: SignedUnitAmount,
    bow_forward: SignedUnitAmount,
}

impl HeadOffset {
    pub const fn new(
        yaw_right: SignedUnitAmount,
        pitch_down: SignedUnitAmount,
        roll_right: SignedUnitAmount,
        bow_forward: SignedUnitAmount,
    ) -> Self {
        Self {
            yaw_right,
            pitch_down,
            roll_right,
            bow_forward,
        }
    }

    pub const fn yaw_right(self) -> SignedUnitAmount {
        self.yaw_right
    }

    pub const fn pitch_down(self) -> SignedUnitAmount {
        self.pitch_down
    }

    pub const fn roll_right(self) -> SignedUnitAmount {
        self.roll_right
    }

    pub const fn bow_forward(self) -> SignedUnitAmount {
        self.bow_forward
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum HeadIntention {
    #[default]
    NaturalHold,
    Offset(HeadOffset),
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ReactionMode {
    #[default]
    NeutralFallback,
    Active,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum VisualReactionSource {
    #[default]
    None,
    SceneMotion,
    Person {
        frame_id: FrameId,
        track_id: PersonTrackId,
    },
}

/// One deterministic output sample.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ReactionOutput {
    generated_at: MonotonicTimestamp,
    valid_until_exclusive: Option<Deadline>,
    mode: ReactionMode,
    active_priority: Option<ExpressionPriority>,
    mixed_intent_count: usize,
    visual_source: VisualReactionSource,
    eyes: EyeIntention,
    head: HeadIntention,
}

impl ReactionOutput {
    pub const fn generated_at(self) -> MonotonicTimestamp {
        self.generated_at
    }

    /// The earliest deadline among every contribution used in this output.
    /// Neutral fallback has no deadline because it carries no stale stimulus.
    pub const fn valid_until_exclusive(self) -> Option<Deadline> {
        self.valid_until_exclusive
    }

    pub const fn mode(self) -> ReactionMode {
        self.mode
    }

    pub const fn active_priority(self) -> Option<ExpressionPriority> {
        self.active_priority
    }

    pub const fn mixed_intent_count(self) -> usize {
        self.mixed_intent_count
    }

    pub const fn visual_source(self) -> VisualReactionSource {
        self.visual_source
    }

    pub const fn eyes(self) -> EyeIntention {
        self.eyes
    }

    pub const fn head(self) -> HeadIntention {
        self.head
    }
}

/// Stateless deterministic mixer. Identical checked inputs produce identical
/// output; randomness and autonomic animation belong in a separately seeded
/// layer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ReactionMixer {
    head_motion: HeadMotionPolicy,
}

impl ReactionMixer {
    pub const fn new(head_motion: HeadMotionPolicy) -> Self {
        Self { head_motion }
    }

    pub fn mix(self, now: MonotonicTimestamp, inputs: ReactionInputs<'_>) -> ReactionOutput {
        let visual = valid_visual(now, &inputs);
        let mut highest_priority = visual.map(VisualCandidate::priority);

        for intent in inputs.intents {
            if intent.freshness().is_fresh_at(now) {
                highest_priority = Some(match highest_priority {
                    Some(current) => current.max(intent.priority()),
                    None => intent.priority(),
                });
            }
        }

        let Some(highest_priority) = highest_priority else {
            return self.neutral(now);
        };

        let mut accumulator = Accumulator::default();
        let mut deadline = None;
        let mut mixed_intent_count = 0_usize;
        let mut visual_source = VisualReactionSource::None;

        if let Some(candidate) = visual.filter(|candidate| candidate.priority() == highest_priority)
        {
            accumulator.add(
                candidate.kind(),
                candidate.weight_basis_points(),
                Some(candidate.center()),
            );
            deadline = earlier_deadline(deadline, candidate.deadline());
            visual_source = candidate.source();
        }

        for intent in inputs.intents {
            if intent.priority() != highest_priority || !intent.freshness().is_fresh_at(now) {
                continue;
            }
            accumulator.add(
                intent.kind(),
                intent.strength().basis_points(),
                intent.gaze_target().map(|target| target.point()),
            );
            deadline = earlier_deadline(deadline, intent.freshness().valid_until_exclusive());
            mixed_intent_count = mixed_intent_count.saturating_add(1);
        }

        if accumulator.total_weight == 0 {
            return self.neutral(now);
        }

        let eyes = accumulator.finish();
        let head = match self.head_motion {
            HeadMotionPolicy::NaturalHold => HeadIntention::NaturalHold,
            HeadMotionPolicy::FollowGaze { gain } => HeadIntention::Offset(HeadOffset::new(
                eyes.gaze_x_right.scaled_by(gain),
                eyes.gaze_y_down.scaled_by(gain),
                SignedUnitAmount::ZERO,
                SignedUnitAmount::ZERO,
            )),
        };

        ReactionOutput {
            generated_at: now,
            valid_until_exclusive: deadline,
            mode: ReactionMode::Active,
            active_priority: Some(highest_priority),
            mixed_intent_count,
            visual_source,
            eyes,
            head,
        }
    }

    fn neutral(self, now: MonotonicTimestamp) -> ReactionOutput {
        ReactionOutput {
            generated_at: now,
            valid_until_exclusive: None,
            mode: ReactionMode::NeutralFallback,
            active_priority: None,
            mixed_intent_count: 0,
            visual_source: VisualReactionSource::None,
            eyes: EyeIntention::neutral(),
            head: HeadIntention::NaturalHold,
        }
    }
}

fn earlier_deadline(current: Option<Deadline>, candidate: Deadline) -> Option<Deadline> {
    Some(match current {
        Some(current) => current.earlier(candidate),
        None => candidate,
    })
}

#[derive(Clone, Copy, Debug)]
enum VisualCandidate {
    Person(PersonObservation, Deadline),
    Scene(crate::SceneMotion, Deadline),
}

impl VisualCandidate {
    const fn priority(self) -> ExpressionPriority {
        match self {
            Self::Person(_, _) => ExpressionPriority::Normal,
            Self::Scene(_, _) => ExpressionPriority::Background,
        }
    }

    const fn kind(self) -> ExpressionKind {
        match self {
            Self::Person(_, _) => ExpressionKind::Attentive,
            Self::Scene(_, _) => ExpressionKind::Curious,
        }
    }

    const fn weight_basis_points(self) -> u16 {
        match self {
            Self::Person(person, _) => person.confidence().basis_points(),
            Self::Scene(motion, _) => motion.strength().basis_points(),
        }
    }

    const fn center(self) -> ImagePoint {
        match self {
            Self::Person(person, _) => person.center(),
            Self::Scene(motion, _) => motion.center(),
        }
    }

    const fn deadline(self) -> Deadline {
        match self {
            Self::Person(_, deadline) | Self::Scene(_, deadline) => deadline,
        }
    }

    const fn source(self) -> VisualReactionSource {
        match self {
            Self::Person(person, _) => VisualReactionSource::Person {
                frame_id: person.frame_id(),
                track_id: person.track_id(),
            },
            Self::Scene(_, _) => VisualReactionSource::SceneMotion,
        }
    }
}

fn valid_visual(now: MonotonicTimestamp, inputs: &ReactionInputs<'_>) -> Option<VisualCandidate> {
    let rgb = inputs.rgb.filter(|rgb| rgb.freshness().is_fresh_at(now))?;
    let rgb_id = rgb.frame_id();
    let rgb_deadline = rgb.freshness().valid_until_exclusive();

    let mut best_person: Option<PersonObservation> = None;
    for person in inputs.people {
        if person.frame_id() != rgb_id || !person.freshness().is_fresh_at(now) {
            continue;
        }
        if best_person.is_none_or(|best| person_is_better(*person, best)) {
            best_person = Some(*person);
        }
    }
    if let Some(person) = best_person {
        return Some(VisualCandidate::Person(
            person,
            rgb_deadline.earlier(person.freshness().valid_until_exclusive()),
        ));
    }

    let scene = inputs
        .scene
        .filter(|scene| scene.frame_id() == rgb_id && scene.freshness().is_fresh_at(now))?;
    let motion = scene.motion()?;
    Some(VisualCandidate::Scene(
        motion,
        rgb_deadline.earlier(scene.freshness().valid_until_exclusive()),
    ))
}

fn person_is_better(candidate: PersonObservation, current: PersonObservation) -> bool {
    let candidate_confidence = candidate.confidence().basis_points();
    let current_confidence = current.confidence().basis_points();
    if candidate_confidence != current_confidence {
        return candidate_confidence > current_confidence;
    }
    match (candidate.distance_mm(), current.distance_mm()) {
        (Some(candidate), Some(current)) if candidate != current => candidate < current,
        (Some(_), None) => true,
        (None, Some(_)) => false,
        _ => candidate.track_id() < current.track_id(),
    }
}

#[derive(Default)]
struct Accumulator {
    total_weight: u128,
    openness: u128,
    pupil: u128,
    gaze_x: i128,
    gaze_y: i128,
    gaze_weight: u128,
}

impl Accumulator {
    fn add(&mut self, kind: ExpressionKind, weight: u16, gaze: Option<ImagePoint>) {
        let weight = u128::from(weight);
        let preset = preset(kind);
        self.total_weight += weight;
        self.openness += u128::from(preset.openness) * weight;
        self.pupil += u128::from(preset.pupil) * weight;
        if let Some(gaze) = gaze {
            self.gaze_x += i128::from(point_axis_to_signed(gaze.x_right())) * weight as i128;
            self.gaze_y += i128::from(point_axis_to_signed(gaze.y_down())) * weight as i128;
            self.gaze_weight += weight;
        }
    }

    fn finish(self) -> EyeIntention {
        let openness = rounded_unsigned_average(self.openness, self.total_weight);
        let pupil = rounded_unsigned_average(self.pupil, self.total_weight);
        let (gaze_x, gaze_y) = if self.gaze_weight == 0 {
            (0, 0)
        } else {
            (
                rounded_signed_average(self.gaze_x, self.gaze_weight),
                rounded_signed_average(self.gaze_y, self.gaze_weight),
            )
        };
        EyeIntention {
            gaze_x_right: SignedUnitAmount::from_basis_points_proven(gaze_x),
            gaze_y_down: SignedUnitAmount::from_basis_points_proven(gaze_y),
            openness: UnitAmount::from_basis_points_proven(openness),
            pupil_dilation: UnitAmount::from_basis_points_proven(pupil),
        }
    }
}

#[derive(Clone, Copy)]
struct Preset {
    openness: u16,
    pupil: u16,
}

const fn preset(kind: ExpressionKind) -> Preset {
    match kind {
        ExpressionKind::Neutral => Preset {
            openness: 8_000,
            pupil: 5_000,
        },
        ExpressionKind::Attentive => Preset {
            openness: 9_000,
            pupil: 5_500,
        },
        ExpressionKind::Friendly => Preset {
            openness: 8_500,
            pupil: 6_000,
        },
        ExpressionKind::Curious => Preset {
            openness: 9_500,
            pupil: 6_500,
        },
        ExpressionKind::Concerned => Preset {
            openness: 6_500,
            pupil: 4_500,
        },
        ExpressionKind::Calm => Preset {
            openness: 7_000,
            pupil: 4_500,
        },
    }
}

fn point_axis_to_signed(axis: UnitAmount) -> i16 {
    (i32::from(axis.basis_points()) * 2 - 10_000) as i16
}

fn rounded_unsigned_average(numerator: u128, denominator: u128) -> u16 {
    ((numerator + denominator / 2) / denominator) as u16
}

fn rounded_signed_average(numerator: i128, denominator: u128) -> i16 {
    let denominator = denominator as i128;
    let rounded = if numerator >= 0 {
        (numerator + denominator / 2) / denominator
    } else {
        (numerator - denominator / 2) / denominator
    };
    rounded as i16
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ChannelOrder, DistanceMillimeters, FreshnessWindow, GazeTarget, ImageLayout,
        NonZeroDuration, PersonTrackId, PositiveUnitAmount, SceneMotion, StreamEpochId,
    };

    fn at(value: u64) -> MonotonicTimestamp {
        MonotonicTimestamp::from_nanos_since_epoch(value)
    }

    fn window(start: u64, ttl: u64) -> FreshnessWindow {
        FreshnessWindow::from_ttl(at(start), NonZeroDuration::try_from_nanos(ttl).unwrap()).unwrap()
    }

    fn point(x: u16, y: u16) -> ImagePoint {
        ImagePoint::new(
            UnitAmount::try_from_basis_points(x).unwrap(),
            UnitAmount::try_from_basis_points(y).unwrap(),
        )
    }

    fn frame_id(stream_epoch: u64, sequence: u64) -> FrameId {
        FrameId::new(StreamEpochId::try_new(stream_epoch).unwrap(), sequence)
    }

    fn rgb(frame_id: FrameId, freshness: FreshnessWindow) -> RgbObservation {
        RgbObservation::new(
            frame_id,
            ImageLayout::try_new(2, 2, 6, ChannelOrder::Bgr).unwrap(),
            freshness,
        )
    }

    #[test]
    fn empty_and_stale_inputs_fall_back_to_neutral_natural_hold() {
        let mixer = ReactionMixer::default();
        let empty = mixer.mix(at(10), ReactionInputs::empty());
        assert_eq!(empty.mode(), ReactionMode::NeutralFallback);
        assert_eq!(empty.eyes(), EyeIntention::neutral());
        assert_eq!(empty.head(), HeadIntention::NaturalHold);
        assert_eq!(empty.valid_until_exclusive(), None);

        let stale = ExpressionIntent::new(
            ExpressionKind::Friendly,
            PositiveUnitAmount::ONE,
            ExpressionPriority::Urgent,
            None,
            window(0, 5),
        );
        let output = mixer.mix(
            at(5),
            ReactionInputs {
                intents: &[stale],
                ..ReactionInputs::empty()
            },
        );
        assert_eq!(output.mode(), ReactionMode::NeutralFallback);
    }

    #[test]
    fn person_requires_a_fresh_matching_rgb_frame() {
        let frame = frame_id(3, 7);
        let other_frame = frame_id(3, 8);
        let camera = rgb(frame, window(0, 20));
        let person = PersonObservation::new(
            other_frame,
            window(0, 20),
            PersonTrackId::try_new(1).unwrap(),
            point(10_000, 0),
            PositiveUnitAmount::ONE,
            None,
        );
        let output = ReactionMixer::default().mix(
            at(10),
            ReactionInputs {
                rgb: Some(&camera),
                people: &[person],
                scene: None,
                intents: &[],
            },
        );
        assert_eq!(output.mode(), ReactionMode::NeutralFallback);
    }

    #[test]
    fn best_person_selection_is_order_independent() {
        let frame = frame_id(3, 7);
        let camera = rgb(frame, window(0, 20));
        let farther = PersonObservation::new(
            frame,
            window(0, 18),
            PersonTrackId::try_new(2).unwrap(),
            point(0, 5_000),
            PositiveUnitAmount::try_from_basis_points(8_000).unwrap(),
            Some(DistanceMillimeters::try_new(2_000).unwrap()),
        );
        let nearer = PersonObservation::new(
            frame,
            window(0, 15),
            PersonTrackId::try_new(1).unwrap(),
            point(10_000, 5_000),
            PositiveUnitAmount::try_from_basis_points(8_000).unwrap(),
            Some(DistanceMillimeters::try_new(1_000).unwrap()),
        );
        let mixer = ReactionMixer::default();
        let a = mixer.mix(
            at(10),
            ReactionInputs {
                rgb: Some(&camera),
                people: &[farther, nearer],
                scene: None,
                intents: &[],
            },
        );
        let b = mixer.mix(
            at(10),
            ReactionInputs {
                rgb: Some(&camera),
                people: &[nearer, farther],
                scene: None,
                intents: &[],
            },
        );
        assert_eq!(a, b);
        assert_eq!(
            a.visual_source(),
            VisualReactionSource::Person {
                frame_id: frame,
                track_id: PersonTrackId::try_new(1).unwrap(),
            }
        );
        assert_eq!(a.eyes().gaze_x_right(), SignedUnitAmount::MAX);
        assert_eq!(a.valid_until_exclusive().unwrap().timestamp(), at(15));
    }

    #[test]
    fn higher_priority_intent_suppresses_visual_reaction() {
        let frame = frame_id(1, 1);
        let camera = rgb(frame, window(0, 20));
        let person = PersonObservation::new(
            frame,
            window(0, 20),
            PersonTrackId::try_new(1).unwrap(),
            point(10_000, 10_000),
            PositiveUnitAmount::ONE,
            None,
        );
        let urgent = ExpressionIntent::new(
            ExpressionKind::Calm,
            PositiveUnitAmount::ONE,
            ExpressionPriority::Urgent,
            None,
            window(0, 12),
        );
        let output = ReactionMixer::default().mix(
            at(10),
            ReactionInputs {
                rgb: Some(&camera),
                people: &[person],
                scene: None,
                intents: &[urgent],
            },
        );
        assert_eq!(output.active_priority(), Some(ExpressionPriority::Urgent));
        assert_eq!(output.visual_source(), VisualReactionSource::None);
        assert_eq!(output.eyes().gaze_x_right(), SignedUnitAmount::ZERO);
        assert_eq!(output.mixed_intent_count(), 1);
    }

    #[test]
    fn same_priority_intents_mix_by_exact_fixed_point_weight() {
        let left = ExpressionIntent::new(
            ExpressionKind::Friendly,
            PositiveUnitAmount::try_from_basis_points(2_500).unwrap(),
            ExpressionPriority::Important,
            Some(GazeTarget::new(point(0, 5_000))),
            window(0, 30),
        );
        let right = ExpressionIntent::new(
            ExpressionKind::Concerned,
            PositiveUnitAmount::try_from_basis_points(7_500).unwrap(),
            ExpressionPriority::Important,
            Some(GazeTarget::new(point(10_000, 5_000))),
            window(0, 20),
        );
        let output = ReactionMixer::default().mix(
            at(10),
            ReactionInputs {
                intents: &[left, right],
                ..ReactionInputs::empty()
            },
        );
        assert_eq!(output.eyes().gaze_x_right().basis_points(), 5_000);
        assert_eq!(output.eyes().openness().basis_points(), 7_000);
        assert_eq!(output.eyes().pupil_dilation().basis_points(), 4_875);
        assert_eq!(output.valid_until_exclusive().unwrap().timestamp(), at(20));
    }

    #[test]
    fn default_locks_head_and_opt_in_follow_is_bounded() {
        let intent = ExpressionIntent::new(
            ExpressionKind::Attentive,
            PositiveUnitAmount::ONE,
            ExpressionPriority::Normal,
            Some(GazeTarget::new(point(10_000, 0))),
            window(0, 20),
        );
        let inputs = ReactionInputs {
            intents: &[intent],
            ..ReactionInputs::empty()
        };
        assert_eq!(
            ReactionMixer::default().mix(at(1), inputs).head(),
            HeadIntention::NaturalHold
        );

        let follow = ReactionMixer::new(HeadMotionPolicy::FollowGaze {
            gain: UnitAmount::try_from_basis_points(2_500).unwrap(),
        })
        .mix(at(1), inputs);
        let HeadIntention::Offset(offset) = follow.head() else {
            panic!("follow-gaze policy must produce an offset");
        };
        assert_eq!(offset.yaw_right().basis_points(), 2_500);
        assert_eq!(offset.pitch_down().basis_points(), -2_500);
        assert_eq!(offset.roll_right(), SignedUnitAmount::ZERO);
        assert_eq!(offset.bow_forward(), SignedUnitAmount::ZERO);
    }

    #[test]
    fn scene_motion_is_used_only_without_a_valid_person() {
        let frame = frame_id(1, 2);
        let camera = rgb(frame, window(0, 10));
        let scene = SceneObservation::new(
            frame,
            window(0, 8),
            UnitAmount::try_from_basis_points(5_000).unwrap(),
            Some(SceneMotion::new(
                PositiveUnitAmount::try_from_basis_points(4_000).unwrap(),
                point(0, 10_000),
            )),
        );
        let output = ReactionMixer::default().mix(
            at(5),
            ReactionInputs {
                rgb: Some(&camera),
                people: &[],
                scene: Some(&scene),
                intents: &[],
            },
        );
        assert_eq!(output.visual_source(), VisualReactionSource::SceneMotion);
        assert_eq!(output.eyes().gaze_x_right(), SignedUnitAmount::MIN);
        assert_eq!(output.eyes().gaze_y_down(), SignedUnitAmount::MAX);
    }

    #[test]
    fn mixed_outputs_remain_bounded_for_many_generated_inputs() {
        let kinds = [
            ExpressionKind::Neutral,
            ExpressionKind::Attentive,
            ExpressionKind::Friendly,
            ExpressionKind::Curious,
            ExpressionKind::Concerned,
            ExpressionKind::Calm,
        ];
        for first_weight in (1_u16..=10_000).step_by(127) {
            for second_weight in (1_u16..=10_000).step_by(521) {
                let first = ExpressionIntent::new(
                    kinds[usize::from(first_weight) % kinds.len()],
                    PositiveUnitAmount::try_from_basis_points(first_weight).unwrap(),
                    ExpressionPriority::Normal,
                    Some(GazeTarget::new(point(first_weight, second_weight))),
                    window(0, 10),
                );
                let second = ExpressionIntent::new(
                    kinds[usize::from(second_weight) % kinds.len()],
                    PositiveUnitAmount::try_from_basis_points(second_weight).unwrap(),
                    ExpressionPriority::Normal,
                    Some(GazeTarget::new(point(second_weight, first_weight))),
                    window(0, 10),
                );
                let output = ReactionMixer::default().mix(
                    at(1),
                    ReactionInputs {
                        intents: &[first, second],
                        ..ReactionInputs::empty()
                    },
                );
                let eyes = output.eyes();
                assert!((-10_000..=10_000).contains(&eyes.gaze_x_right().basis_points()));
                assert!((-10_000..=10_000).contains(&eyes.gaze_y_down().basis_points()));
                assert!(eyes.openness().basis_points() <= 10_000);
                assert!(eyes.pupil_dilation().basis_points() <= 10_000);
            }
        }
    }
}
