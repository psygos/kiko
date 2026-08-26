//! Bounded autonomic eye and four-joint character behavior.
//!
//! The visual mixer owns face/scene attention. This module owns only the
//! character-like timing that makes an otherwise idle renderer feel alive:
//! greeting, loss/search/sleep transitions, blinks, micro-saccades, and a
//! finite act library derived from Kiko's retained expression-engine behavior.
//! Eyes and head share one act clock so eye contact can lead a delayed,
//! minimum-jerk head response instead of looking like unrelated animations.
//!
//! It performs no allocation, I/O, sleeping, wall-clock access, or head
//! actuation. Head output is a normalized semantic overlay, never encoder
//! ticks or motion authority. Only the evidence-bound calibrated head path may
//! convert it into a physical target.

use core::fmt;
use core::num::NonZeroU64;
use core::time::Duration;
use kiko_expression_core::MonotonicTimestamp;
use kiko_eye_protocol::{
    Expression, EyeFlags, EyeIntent, NORMALIZED_SCALE, SignedUnit, UnitAmount,
};

use crate::PreparedEyeIntent;

const SCALE: i32 = NORMALIZED_SCALE as i32;
const NS_PER_MS: u64 = 1_000_000;
const NS_PER_SECOND: u64 = 1_000 * NS_PER_MS;
const GREETING_MIN_NS: u64 = 900 * NS_PER_MS;
const GREETING_MAX_NS: u64 = 1_400 * NS_PER_MS;
const GREETING_COOLDOWN_NS: u64 = 10 * NS_PER_SECOND;
const LOST_DURATION_NS: u64 = 700 * NS_PER_MS;
const SEARCH_MIN_NS: u64 = 2_200 * NS_PER_MS;
const SEARCH_MAX_NS: u64 = 3_800 * NS_PER_MS;
const SLEEPY_AFTER_IDLE_NS: u64 = 60 * NS_PER_SECOND;
const FIRST_ACT_MIN_NS: u64 = 2 * NS_PER_SECOND;
const FIRST_ACT_MAX_NS: u64 = 5 * NS_PER_SECOND;
const TRACK_ACT_GAP_MIN_NS: u64 = 3 * NS_PER_SECOND;
const TRACK_ACT_GAP_MAX_NS: u64 = 9 * NS_PER_SECOND;
const IDLE_ACT_GAP_MIN_NS: u64 = 4 * NS_PER_SECOND;
const IDLE_ACT_GAP_MAX_NS: u64 = 12 * NS_PER_SECOND;
const FIRST_BLINK_MIN_NS: u64 = 3 * NS_PER_SECOND;
const FIRST_BLINK_MAX_NS: u64 = 5 * NS_PER_SECOND;
const BLINK_MIN_NS: u64 = 2 * NS_PER_SECOND;
const BLINK_MAX_NS: u64 = 12 * NS_PER_SECOND;
const SACCADE_MIN_NS: u64 = 900 * NS_PER_MS;
const SACCADE_MAX_NS: u64 = 2_400 * NS_PER_MS;
const SACCADE_HOLD_NS: u64 = 120 * NS_PER_MS;
const SACCADE_DECAY_NS: u64 = 280 * NS_PER_MS;
const HEAD_EYE_LEAD_NS: u64 = 120 * NS_PER_MS;
const REST_AFTER_IDLE_NS: u64 = 20 * NS_PER_SECOND;
const REST_EASE_NS: u64 = 6 * NS_PER_SECOND;
const NEVER_RUN: u64 = u64::MAX;

/// Dimensionless signed character displacement scale.
///
/// `1_000` means the full *reviewed expressive excursion* for the named
/// joint. Its sign is `character-positive`, whose encoder polarity must be an
/// explicit physical mapping declaration; it never means a raw encoder
/// position or the hardware hard limit.
pub const CHARACTER_HEAD_SCALE: i16 = 1_000;

/// A bounded normalized displacement for one named head joint.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct CharacterHeadAmount(i16);

impl CharacterHeadAmount {
    pub const ZERO: Self = Self(0);

    pub const fn try_new(value: i16) -> Result<Self, CharacterHeadAmountError> {
        if value < -CHARACTER_HEAD_SCALE || value > CHARACTER_HEAD_SCALE {
            return Err(CharacterHeadAmountError { value });
        }
        Ok(Self(value))
    }

    const fn from_clamped(value: i32) -> Self {
        let clamped = if value < -(CHARACTER_HEAD_SCALE as i32) {
            -(CHARACTER_HEAD_SCALE as i32)
        } else if value > CHARACTER_HEAD_SCALE as i32 {
            CHARACTER_HEAD_SCALE as i32
        } else {
            value
        };
        Self(clamped as i16)
    }

    pub const fn get(self) -> i16 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CharacterHeadAmountError {
    pub value: i16,
}

impl fmt::Display for CharacterHeadAmountError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "character-head amount {} is outside [{}, {}]",
            self.value, -CHARACTER_HEAD_SCALE, CHARACTER_HEAD_SCALE
        )
    }
}

impl core::error::Error for CharacterHeadAmountError {}

/// Named semantic axis at the transport-independent character boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CharacterHeadAxis {
    Bow,
    Curl,
    Yaw,
    Roll,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CharacterHeadOverlayParseError {
    pub axis: CharacterHeadAxis,
    pub source: CharacterHeadAmountError,
}

impl fmt::Display for CharacterHeadOverlayParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid {:?} character-head axis: {}",
            self.axis, self.source
        )
    }
}

impl core::error::Error for CharacterHeadOverlayParseError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        Some(&self.source)
    }
}

/// Semantic four-servo overlay relative to Kiko's reviewed natural pose.
///
/// Axes are named, but encoder polarity is deliberately absent. This prevents
/// the character director from guessing mounting signs. The order of physical
/// servo IDs is likewise absent from this domain type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CharacterHeadOverlay {
    bow: CharacterHeadAmount,
    curl: CharacterHeadAmount,
    yaw: CharacterHeadAmount,
    roll: CharacterHeadAmount,
}

impl CharacterHeadOverlay {
    pub const NATURAL: Self = Self {
        bow: CharacterHeadAmount::ZERO,
        curl: CharacterHeadAmount::ZERO,
        yaw: CharacterHeadAmount::ZERO,
        roll: CharacterHeadAmount::ZERO,
    };

    pub fn try_new(
        bow: i16,
        curl: i16,
        yaw: i16,
        roll: i16,
    ) -> Result<Self, CharacterHeadOverlayParseError> {
        Ok(Self {
            bow: CharacterHeadAmount::try_new(bow).map_err(|source| {
                CharacterHeadOverlayParseError {
                    axis: CharacterHeadAxis::Bow,
                    source,
                }
            })?,
            curl: CharacterHeadAmount::try_new(curl).map_err(|source| {
                CharacterHeadOverlayParseError {
                    axis: CharacterHeadAxis::Curl,
                    source,
                }
            })?,
            yaw: CharacterHeadAmount::try_new(yaw).map_err(|source| {
                CharacterHeadOverlayParseError {
                    axis: CharacterHeadAxis::Yaw,
                    source,
                }
            })?,
            roll: CharacterHeadAmount::try_new(roll).map_err(|source| {
                CharacterHeadOverlayParseError {
                    axis: CharacterHeadAxis::Roll,
                    source,
                }
            })?,
        })
    }

    const fn from_clamped(fields: HeadFields) -> Self {
        Self {
            bow: CharacterHeadAmount::from_clamped(fields.bow),
            curl: CharacterHeadAmount::from_clamped(fields.curl),
            yaw: CharacterHeadAmount::from_clamped(fields.yaw),
            roll: CharacterHeadAmount::from_clamped(fields.roll),
        }
    }

    pub const fn bow(self) -> CharacterHeadAmount {
        self.bow
    }

    pub const fn curl(self) -> CharacterHeadAmount {
        self.curl
    }

    pub const fn yaw(self) -> CharacterHeadAmount {
        self.yaw
    }

    pub const fn roll(self) -> CharacterHeadAmount {
        self.roll
    }

    pub const fn amounts(self) -> [CharacterHeadAmount; 4] {
        [self.bow, self.curl, self.yaw, self.roll]
    }

    pub const fn is_natural(self) -> bool {
        self.bow.get() == 0 && self.curl.get() == 0 && self.yaw.get() == 0 && self.roll.get() == 0
    }
}

/// One time-coherent, transport-independent character decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PreparedCharacterFrame {
    eye: PreparedEyeIntent,
    head: CharacterHeadOverlay,
    mode: CharacterMode,
    act: Option<CharacterAct>,
}

impl PreparedCharacterFrame {
    pub const fn eyes_only(eye: PreparedEyeIntent) -> Self {
        Self {
            eye,
            head: CharacterHeadOverlay::NATURAL,
            mode: CharacterMode::Idle,
            act: None,
        }
    }

    const fn new(
        eye: PreparedEyeIntent,
        head: CharacterHeadOverlay,
        mode: CharacterMode,
        act: Option<CharacterAct>,
    ) -> Self {
        Self {
            eye,
            head,
            mode,
            act,
        }
    }

    pub const fn eye(self) -> PreparedEyeIntent {
        self.eye
    }

    /// Compatibility projection for diagnostics that previously received an
    /// eye-only prepared value.
    pub const fn intent(self) -> EyeIntent {
        self.eye.intent()
    }

    pub const fn head(self) -> CharacterHeadOverlay {
        self.head
    }

    pub const fn mode(self) -> CharacterMode {
        self.mode
    }

    pub const fn act(self) -> Option<CharacterAct> {
        self.act
    }
}

/// High-level autonomic state. This is eye-rendering state, not navigation or
/// physical authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CharacterMode {
    Idle,
    Greeting,
    Tracking,
    Lost,
    Searching,
    Sleepy,
}

/// Finite retained character-act vocabulary.
///
/// The names intentionally match the earlier Kiko expression engine so a
/// migration audit cannot silently collapse distinct behaviors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CharacterAct {
    CuriousTilt,
    DoubleTake,
    ExcitedWiggle,
    LeanIn,
    Nod,
    SoftNod,
    HappySquint,
    PuppyEyes,
    ShyDip,
    Sparkle,
    BlinkFlourish,
    LookAround,
    PerkUp,
    Daydream,
    Stretch,
    SweepScan,
    HeadBob,
    Sneeze,
    Dance,
    Sigh,
    BowBob,
    PlayBow,
    StartleBoop,
    AffectionMelt,
}

impl CharacterAct {
    const ALL: [Self; 24] = [
        Self::CuriousTilt,
        Self::DoubleTake,
        Self::ExcitedWiggle,
        Self::LeanIn,
        Self::Nod,
        Self::SoftNod,
        Self::HappySquint,
        Self::PuppyEyes,
        Self::ShyDip,
        Self::Sparkle,
        Self::BlinkFlourish,
        Self::LookAround,
        Self::PerkUp,
        Self::Daydream,
        Self::Stretch,
        Self::SweepScan,
        Self::HeadBob,
        Self::Sneeze,
        Self::Dance,
        Self::Sigh,
        Self::BowBob,
        Self::PlayBow,
        Self::StartleBoop,
        Self::AffectionMelt,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CuriousTilt => "curious_tilt",
            Self::DoubleTake => "double_take",
            Self::ExcitedWiggle => "excited_wiggle",
            Self::LeanIn => "lean_in",
            Self::Nod => "nod",
            Self::SoftNod => "soft_nod",
            Self::HappySquint => "happy_squint",
            Self::PuppyEyes => "puppy_eyes",
            Self::ShyDip => "shy_dip",
            Self::Sparkle => "sparkle",
            Self::BlinkFlourish => "blink_flourish",
            Self::LookAround => "look_around",
            Self::PerkUp => "perk_up",
            Self::Daydream => "daydream",
            Self::Stretch => "stretch",
            Self::SweepScan => "sweep_scan",
            Self::HeadBob => "head_bob",
            Self::Sneeze => "sneeze",
            Self::Dance => "dance",
            Self::Sigh => "sigh",
            Self::BowBob => "bow_bob",
            Self::PlayBow => "play_bow",
            Self::StartleBoop => "startle_boop",
            Self::AffectionMelt => "affection_melt",
        }
    }

    const fn index(self) -> usize {
        match self {
            Self::CuriousTilt => 0,
            Self::DoubleTake => 1,
            Self::ExcitedWiggle => 2,
            Self::LeanIn => 3,
            Self::Nod => 4,
            Self::SoftNod => 5,
            Self::HappySquint => 6,
            Self::PuppyEyes => 7,
            Self::ShyDip => 8,
            Self::Sparkle => 9,
            Self::BlinkFlourish => 10,
            Self::LookAround => 11,
            Self::PerkUp => 12,
            Self::Daydream => 13,
            Self::Stretch => 14,
            Self::SweepScan => 15,
            Self::HeadBob => 16,
            Self::Sneeze => 17,
            Self::Dance => 18,
            Self::Sigh => 19,
            Self::BowBob => 20,
            Self::PlayBow => 21,
            Self::StartleBoop => 22,
            Self::AffectionMelt => 23,
        }
    }

    const fn eligible(self, mode: CharacterMode) -> bool {
        match self {
            Self::LookAround | Self::PerkUp | Self::Daydream | Self::Stretch => {
                matches!(mode, CharacterMode::Idle)
            }
            Self::Sparkle
            | Self::BlinkFlourish
            | Self::SweepScan
            | Self::Sneeze
            | Self::Dance
            | Self::Sigh => {
                matches!(mode, CharacterMode::Idle | CharacterMode::Tracking)
            }
            Self::CuriousTilt
            | Self::DoubleTake
            | Self::ExcitedWiggle
            | Self::LeanIn
            | Self::Nod
            | Self::SoftNod
            | Self::HappySquint
            | Self::PuppyEyes
            | Self::ShyDip
            | Self::HeadBob
            | Self::BowBob
            | Self::PlayBow => matches!(mode, CharacterMode::Tracking),
            Self::StartleBoop | Self::AffectionMelt => false,
        }
    }

    const fn cooldown_ns(self) -> u64 {
        (match self {
            Self::CuriousTilt => 9,
            Self::DoubleTake => 16,
            Self::ExcitedWiggle => 14,
            Self::LeanIn => 15,
            Self::Nod => 11,
            Self::SoftNod => 12,
            Self::HappySquint => 12,
            Self::PuppyEyes => 18,
            Self::ShyDip => 25,
            Self::Sparkle => 10,
            Self::BlinkFlourish => 8,
            Self::LookAround => 7,
            Self::PerkUp => 13,
            Self::Daydream => 12,
            Self::Stretch => 40,
            Self::SweepScan => 16,
            Self::HeadBob => 13,
            Self::Sneeze => 45,
            Self::Dance => 30,
            Self::Sigh => 22,
            Self::BowBob => 9,
            Self::PlayBow => 20,
            Self::StartleBoop | Self::AffectionMelt => 0,
        }) * NS_PER_SECOND
    }

    const fn duration_bounds_ns(self) -> (u64, u64) {
        let milliseconds = match self {
            Self::CuriousTilt => (2_200, 3_600),
            Self::DoubleTake => (1_300, 1_900),
            Self::ExcitedWiggle => (1_600, 2_600),
            Self::LeanIn => (2_800, 4_400),
            Self::Nod => (1_600, 2_300),
            Self::SoftNod => (2_400, 3_400),
            Self::HappySquint => (1_600, 2_600),
            Self::PuppyEyes => (2_600, 3_800),
            Self::ShyDip => (1_800, 2_600),
            Self::Sparkle => (1_000, 1_700),
            Self::BlinkFlourish => (900, 1_300),
            Self::LookAround => (3_200, 5_500),
            Self::PerkUp => (1_400, 2_200),
            Self::Daydream => (4_000, 6_500),
            Self::Stretch => (2_600, 3_600),
            Self::SweepScan => (3_400, 5_200),
            Self::HeadBob => (1_800, 2_800),
            Self::Sneeze => (2_400, 3_200),
            Self::Dance => (4_200, 6_800),
            Self::Sigh => (2_600, 3_400),
            Self::BowBob => (1_000, 1_400),
            Self::PlayBow => (2_400, 3_200),
            Self::StartleBoop => (1_600, 2_100),
            Self::AffectionMelt => (2_800, 3_600),
        };
        (milliseconds.0 * NS_PER_MS, milliseconds.1 * NS_PER_MS)
    }
}

/// Exact, transport-free facts from one completed compliant contact episode.
///
/// The character layer receives no raw servo registers and makes no claim
/// about why contact occurred. It classifies only the head controller's
/// completed episode evidence. A zero-duration episode or a non-zero delta
/// sum without samples is rejected at this boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CharacterPetEpisode {
    duration_ns: NonZeroU64,
    accumulated_max_delta_ticks: u64,
    delta_samples: u64,
    reached_comfy: bool,
    tap: bool,
}

impl CharacterPetEpisode {
    pub fn try_new(
        duration: Duration,
        accumulated_max_delta_ticks: u64,
        delta_samples: u64,
        reached_comfy: bool,
        tap: bool,
    ) -> Result<Self, CharacterPetEpisodeError> {
        let duration_ns = duration.as_nanos();
        let duration_ns = u64::try_from(duration_ns)
            .ok()
            .and_then(NonZeroU64::new)
            .ok_or(CharacterPetEpisodeError::InvalidDuration { duration_ns })?;
        if delta_samples == 0 && accumulated_max_delta_ticks != 0 {
            return Err(CharacterPetEpisodeError::DeltaWithoutSamples {
                accumulated_max_delta_ticks,
            });
        }
        Ok(Self {
            duration_ns,
            accumulated_max_delta_ticks,
            delta_samples,
            reached_comfy,
            tap,
        })
    }

    pub const fn duration_ns(self) -> NonZeroU64 {
        self.duration_ns
    }

    pub const fn accumulated_max_delta_ticks(self) -> u64 {
        self.accumulated_max_delta_ticks
    }

    pub const fn delta_samples(self) -> u64 {
        self.delta_samples
    }

    pub const fn reached_comfy(self) -> bool {
        self.reached_comfy
    }

    pub const fn was_tap(self) -> bool {
        self.tap
    }

    pub const fn reaction(self) -> CharacterPetReaction {
        if self.tap {
            return CharacterPetReaction::Boop;
        }
        let playful = match self.delta_samples.checked_mul(6) {
            Some(threshold) => self.accumulated_max_delta_ticks >= threshold,
            None => false,
        };
        if playful && !self.reached_comfy {
            CharacterPetReaction::Play
        } else {
            CharacterPetReaction::Affection
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CharacterPetEpisodeError {
    InvalidDuration { duration_ns: u128 },
    DeltaWithoutSamples { accumulated_max_delta_ticks: u64 },
}

impl fmt::Display for CharacterPetEpisodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid character pet episode: {self:?}")
    }
}

impl core::error::Error for CharacterPetEpisodeError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CharacterPetReaction {
    Boop,
    Play,
    Affection,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RunningAct {
    act: CharacterAct,
    started_ns: u64,
    duration_ns: u64,
    side: i32,
    style: u8,
    energy_milli: u16,
}

#[derive(Clone, Copy, Debug)]
struct EyeFields {
    gaze_x: i32,
    gaze_y: i32,
    lid: i32,
    pupil: i32,
    brightness: i32,
    expression: Expression,
    blink: bool,
    color_rgb: [u8; 3],
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct HeadFields {
    bow: i32,
    curl: i32,
    yaw: i32,
    roll: i32,
}

impl From<EyeIntent> for EyeFields {
    fn from(intent: EyeIntent) -> Self {
        Self {
            gaze_x: i32::from(intent.gaze_x().get()),
            gaze_y: i32::from(intent.gaze_y().get()),
            lid: i32::from(intent.lid().get()),
            pupil: i32::from(intent.pupil().get()),
            brightness: i32::from(intent.brightness().get()),
            expression: intent.expression(),
            blink: intent.flags().requests_blink(),
            color_rgb: intent.color_rgb(),
        }
    }
}

impl EyeFields {
    fn into_intent(self) -> EyeIntent {
        EyeIntent::new(
            SignedUnit::try_new(clamp_signed(self.gaze_x))
                .expect("clamped gaze x is a valid signed unit"),
            SignedUnit::try_new(clamp_signed(self.gaze_y))
                .expect("clamped gaze y is a valid signed unit"),
            UnitAmount::try_new(clamp_unit(self.lid)).expect("clamped lid is a valid unit"),
            UnitAmount::try_new(clamp_unit(self.pupil)).expect("clamped pupil is a valid unit"),
            UnitAmount::try_new(clamp_unit(self.brightness))
                .expect("clamped brightness is a valid unit"),
            self.expression,
            EyeFlags::try_from_bits(if self.blink { EyeFlags::BLINK } else { 0 })
                .expect("blink is the only requested flag"),
            self.color_rgb,
        )
    }
}

/// Stateful, deterministic-for-one-seed autonomic eye director.
///
/// A camera stream epoch is a suitable seed: it creates varied behavior
/// without adding an entropy or global-RNG dependency to the frame path.
pub struct AutonomicCharacterEngine {
    random_state: u64,
    initialized: bool,
    created_ns: u64,
    life_phase_milli: [u16; 6],
    ever_saw_face: bool,
    mode: CharacterMode,
    mode_started_ns: u64,
    last_face_ns: u64,
    greeting_until_ns: u64,
    greeting_style: u8,
    greeting_blink_pending: bool,
    searching_until_ns: u64,
    next_blink_ns: u64,
    next_saccade_ns: u64,
    saccade_peak_x: i32,
    saccade_peak_y: i32,
    saccade_hold_until_ns: u64,
    saccade_decay_until_ns: u64,
    saccade_x: i32,
    saccade_y: i32,
    next_act_ns: u64,
    active_act: Option<RunningAct>,
    last_run_ns: [u64; CharacterAct::ALL.len()],
    playfulness_milli: u16,
    playfulness_updated_ns: u64,
    playfulness_decay_remainder: u64,
}

impl AutonomicCharacterEngine {
    pub const fn new(seed: u64) -> Self {
        Self {
            random_state: if seed == 0 {
                0x9e37_79b9_7f4a_7c15
            } else {
                seed
            },
            initialized: false,
            created_ns: 0,
            life_phase_milli: [0; 6],
            ever_saw_face: false,
            mode: CharacterMode::Idle,
            mode_started_ns: 0,
            last_face_ns: 0,
            greeting_until_ns: 0,
            greeting_style: 0,
            greeting_blink_pending: false,
            searching_until_ns: 0,
            next_blink_ns: 0,
            next_saccade_ns: 0,
            saccade_peak_x: 0,
            saccade_peak_y: 0,
            saccade_hold_until_ns: 0,
            saccade_decay_until_ns: 0,
            saccade_x: 0,
            saccade_y: 0,
            next_act_ns: 0,
            active_act: None,
            last_run_ns: [NEVER_RUN; CharacterAct::ALL.len()],
            playfulness_milli: 0,
            playfulness_updated_ns: 0,
            playfulness_decay_remainder: 0,
        }
    }

    pub const fn mode(&self) -> CharacterMode {
        self.mode
    }

    pub const fn active_act(&self) -> Option<CharacterAct> {
        match self.active_act {
            Some(running) => Some(running.act),
            None => None,
        }
    }

    pub const fn playfulness_milli(&self) -> u16 {
        self.playfulness_milli
    }

    /// Preempt the scheduled act with the social response to one completed
    /// compliant-contact episode. The head controller decides contact facts;
    /// this character layer decides only their presentation.
    pub fn note_pet_episode(
        &mut self,
        now: MonotonicTimestamp,
        episode: CharacterPetEpisode,
    ) -> CharacterPetReaction {
        let now_ns = now.nanos_since_epoch();
        self.initialize_if_needed(now_ns);
        self.decay_playfulness(now_ns);
        let reaction = episode.reaction();
        if matches!(
            reaction,
            CharacterPetReaction::Boop | CharacterPetReaction::Play
        ) {
            self.playfulness_milli = self.playfulness_milli.saturating_add(450).min(1_000);
        }
        let act = match reaction {
            CharacterPetReaction::Boop => CharacterAct::StartleBoop,
            CharacterPetReaction::Play => CharacterAct::PlayBow,
            CharacterPetReaction::Affection => CharacterAct::AffectionMelt,
        };
        let (minimum, maximum) = act.duration_bounds_ns();
        let duration_ns = self.random_range(minimum, maximum);
        let side = if self.next_u64() & 1 == 0 { -1 } else { 1 };
        let style = (self.next_u64() % 3) as u8;
        self.last_run_ns[act.index()] = now_ns;
        self.active_act = Some(RunningAct {
            act,
            started_ns: now_ns,
            duration_ns,
            side,
            style,
            energy_milli: self.playfulness_milli,
        });
        reaction
    }

    /// Apply one autonomic sample without changing the reaction's freshness.
    ///
    /// This compatibility entrypoint discards the semantic head overlay. New
    /// integrated callers should use [`Self::render_character`] so the eyes
    /// and all four head joints retain their shared timing.
    pub fn render(
        &mut self,
        now: MonotonicTimestamp,
        face_present: bool,
        prepared: PreparedEyeIntent,
    ) -> PreparedEyeIntent {
        self.render_character(now, face_present, prepared).eye()
    }

    /// Prepare one coherent eye plus four-joint character sample.
    pub fn render_character(
        &mut self,
        now: MonotonicTimestamp,
        face_present: bool,
        prepared: PreparedEyeIntent,
    ) -> PreparedCharacterFrame {
        let now_ns = now.nanos_since_epoch();
        self.initialize_if_needed(now_ns);
        self.decay_playfulness(now_ns);
        self.update_mode(face_present, now_ns);
        self.update_saccade(now_ns);
        self.update_act(now_ns);

        let mut fields = EyeFields::from(prepared.intent());
        // The static bridge style is the fallback used when this director is
        // absent. Once enabled, blink becomes a timed event; carrying a static
        // blink bit into every refreshed frame would retrigger it continuously.
        fields.blink = false;
        self.apply_mode(&mut fields, now_ns);
        self.apply_act(&mut fields, now_ns);
        self.apply_blink(&mut fields, now_ns);
        let mut head = HeadFields::default();
        self.apply_head_mode(&mut head, now_ns);
        self.apply_head_act(&mut head, now_ns);
        let rest_envelope = self.rest_envelope(now_ns);
        head.bow = head.bow * rest_envelope / SCALE;
        head.curl = head.curl * rest_envelope / SCALE;
        head.yaw = head.yaw * rest_envelope / SCALE;
        head.roll = head.roll * rest_envelope / SCALE;
        PreparedCharacterFrame::new(
            prepared.with_intent(fields.into_intent()),
            CharacterHeadOverlay::from_clamped(head),
            self.mode,
            self.active_act(),
        )
    }

    fn initialize_if_needed(&mut self, now_ns: u64) {
        if self.initialized {
            return;
        }
        self.initialized = true;
        self.created_ns = now_ns;
        self.mode_started_ns = now_ns;
        self.last_face_ns = now_ns;
        self.next_blink_ns = saturating_add(
            now_ns,
            self.random_range(FIRST_BLINK_MIN_NS, FIRST_BLINK_MAX_NS),
        );
        self.next_saccade_ns =
            saturating_add(now_ns, self.random_range(SACCADE_MIN_NS, SACCADE_MAX_NS));
        self.next_act_ns = saturating_add(
            now_ns,
            self.random_range(FIRST_ACT_MIN_NS, FIRST_ACT_MAX_NS),
        );
        let life_phase_milli = core::array::from_fn(|_| {
            u16::try_from(
                self.next_u64() % u64::try_from(SCALE).expect("positive normalized scale"),
            )
            .expect("normalized phase seed fits u16")
        });
        self.life_phase_milli = life_phase_milli;
        self.playfulness_updated_ns = now_ns;
    }

    fn decay_playfulness(&mut self, now_ns: u64) {
        if self.playfulness_milli == 0 {
            self.playfulness_decay_remainder = 0;
            self.playfulness_updated_ns = now_ns;
            return;
        }
        let elapsed_ns = now_ns.saturating_sub(self.playfulness_updated_ns);
        let denominator = u128::from(45 * NS_PER_SECOND);
        let numerator =
            u128::from(elapsed_ns) * 1_000 + u128::from(self.playfulness_decay_remainder);
        let decrease = numerator / denominator;
        self.playfulness_decay_remainder = u64::try_from(numerator % denominator)
            .expect("decay remainder is below a u64 denominator");
        let decrease = u16::try_from(decrease.min(u128::from(u16::MAX)))
            .expect("bounded playfulness decay fits u16");
        self.playfulness_milli = self.playfulness_milli.saturating_sub(decrease);
        self.playfulness_updated_ns = now_ns;
    }

    fn rest_envelope(&self, now_ns: u64) -> i32 {
        if matches!(self.mode, CharacterMode::Greeting | CharacterMode::Tracking) {
            return SCALE;
        }
        let idle_ns = now_ns.saturating_sub(self.last_face_ns);
        if idle_ns <= REST_AFTER_IDLE_NS {
            return SCALE;
        }
        let easing_ns = idle_ns - REST_AFTER_IDLE_NS;
        if easing_ns >= REST_EASE_NS {
            return 0;
        }
        SCALE - minimum_jerk_ramp(normalized_phase(easing_ns, REST_EASE_NS))
    }

    fn update_mode(&mut self, face_present: bool, now_ns: u64) {
        if face_present {
            let greeting_due = !self.ever_saw_face
                || now_ns.saturating_sub(self.last_face_ns) > GREETING_COOLDOWN_NS;
            if matches!(
                self.mode,
                CharacterMode::Idle
                    | CharacterMode::Sleepy
                    | CharacterMode::Searching
                    | CharacterMode::Lost
            ) {
                if greeting_due {
                    self.enter(CharacterMode::Greeting, now_ns);
                    self.greeting_style = (self.next_u64() % 3) as u8;
                    self.greeting_blink_pending = true;
                    self.greeting_until_ns =
                        saturating_add(now_ns, self.random_range(GREETING_MIN_NS, GREETING_MAX_NS));
                } else {
                    self.enter(CharacterMode::Tracking, now_ns);
                }
            }
            self.ever_saw_face = true;
            self.last_face_ns = now_ns;
        } else {
            match self.mode {
                CharacterMode::Tracking | CharacterMode::Greeting => {
                    self.enter(CharacterMode::Lost, now_ns);
                }
                CharacterMode::Lost
                    if now_ns.saturating_sub(self.mode_started_ns) >= LOST_DURATION_NS =>
                {
                    self.enter(CharacterMode::Searching, now_ns);
                    self.searching_until_ns =
                        saturating_add(now_ns, self.random_range(SEARCH_MIN_NS, SEARCH_MAX_NS));
                }
                CharacterMode::Searching if now_ns >= self.searching_until_ns => {
                    self.enter(CharacterMode::Idle, now_ns);
                }
                CharacterMode::Idle
                    if now_ns.saturating_sub(self.last_face_ns) >= SLEEPY_AFTER_IDLE_NS =>
                {
                    self.enter(CharacterMode::Sleepy, now_ns);
                }
                CharacterMode::Idle
                | CharacterMode::Lost
                | CharacterMode::Searching
                | CharacterMode::Sleepy => {}
            }
        }
        if self.mode == CharacterMode::Greeting && now_ns >= self.greeting_until_ns {
            self.enter(CharacterMode::Tracking, now_ns);
        }
    }

    fn enter(&mut self, mode: CharacterMode, now_ns: u64) {
        if mode == self.mode {
            return;
        }
        self.mode = mode;
        self.mode_started_ns = now_ns;
        if !matches!(mode, CharacterMode::Idle | CharacterMode::Tracking) {
            self.active_act = None;
        }
    }

    fn update_saccade(&mut self, now_ns: u64) {
        // Face tracking already carries authoritative gaze. Adding random
        // offsets here made the eyes visibly ping-pong around a real person,
        // so tracking explicitly suppresses and reschedules autonomic
        // saccades instead of merely hiding the current sample.
        if self.mode == CharacterMode::Tracking {
            self.saccade_peak_x = 0;
            self.saccade_peak_y = 0;
            self.saccade_x = 0;
            self.saccade_y = 0;
            self.saccade_hold_until_ns = now_ns;
            self.saccade_decay_until_ns = now_ns;
            if now_ns >= self.next_saccade_ns {
                self.next_saccade_ns =
                    saturating_add(now_ns, self.random_range(SACCADE_MIN_NS, SACCADE_MAX_NS));
            }
            return;
        }

        if now_ns >= self.next_saccade_ns {
            self.saccade_peak_x = self.random_signed(14);
            self.saccade_peak_y = self.random_signed(9);
            self.saccade_x = self.saccade_peak_x;
            self.saccade_y = self.saccade_peak_y;
            self.saccade_hold_until_ns = saturating_add(now_ns, SACCADE_HOLD_NS);
            self.saccade_decay_until_ns =
                saturating_add(self.saccade_hold_until_ns, SACCADE_DECAY_NS);
            self.next_saccade_ns =
                saturating_add(now_ns, self.random_range(SACCADE_MIN_NS, SACCADE_MAX_NS));
            return;
        }

        if now_ns <= self.saccade_hold_until_ns {
            self.saccade_x = self.saccade_peak_x;
            self.saccade_y = self.saccade_peak_y;
            return;
        }
        if now_ns >= self.saccade_decay_until_ns {
            self.saccade_x = 0;
            self.saccade_y = 0;
            return;
        }

        let elapsed = now_ns.saturating_sub(self.saccade_hold_until_ns);
        let progress = normalized_phase(elapsed, SACCADE_DECAY_NS);
        let remaining = SCALE - minimum_jerk_ramp(progress);
        self.saccade_x = self.saccade_peak_x * remaining / SCALE;
        self.saccade_y = self.saccade_peak_y * remaining / SCALE;
    }

    fn update_act(&mut self, now_ns: u64) {
        if let Some(running) = self.active_act {
            if now_ns.saturating_sub(running.started_ns) < running.duration_ns {
                return;
            }
            self.active_act = None;
            let (minimum, maximum) = if self.mode == CharacterMode::Tracking {
                (TRACK_ACT_GAP_MIN_NS, TRACK_ACT_GAP_MAX_NS)
            } else {
                (IDLE_ACT_GAP_MIN_NS, IDLE_ACT_GAP_MAX_NS)
            };
            self.next_act_ns = saturating_add(now_ns, self.random_range(minimum, maximum));
        }
        if self.active_act.is_some()
            || now_ns < self.next_act_ns
            || !matches!(self.mode, CharacterMode::Idle | CharacterMode::Tracking)
            || self.rest_envelope(now_ns) == 0
        {
            return;
        }

        let eligible_count = CharacterAct::ALL
            .iter()
            .copied()
            .filter(|act| {
                act.eligible(self.mode)
                    && (self.last_run_ns[act.index()] == NEVER_RUN
                        || now_ns.saturating_sub(self.last_run_ns[act.index()])
                            >= act.cooldown_ns())
            })
            .count();
        if eligible_count == 0 {
            self.next_act_ns = saturating_add(now_ns, 2 * NS_PER_SECOND);
            return;
        }
        let selected = usize::try_from(self.next_u64() % eligible_count as u64)
            .expect("eligible act count fits usize");
        let act = CharacterAct::ALL
            .iter()
            .copied()
            .filter(|act| {
                act.eligible(self.mode)
                    && (self.last_run_ns[act.index()] == NEVER_RUN
                        || now_ns.saturating_sub(self.last_run_ns[act.index()])
                            >= act.cooldown_ns())
            })
            .nth(selected)
            .expect("selected eligible act exists");
        let (minimum, maximum) = act.duration_bounds_ns();
        let duration_ns = self.random_range(minimum, maximum);
        let side = if self.next_u64() & 1 == 0 { -1 } else { 1 };
        let style = (self.next_u64() % 3) as u8;
        self.last_run_ns[act.index()] = now_ns;
        self.active_act = Some(RunningAct {
            act,
            started_ns: now_ns,
            duration_ns,
            side,
            style,
            energy_milli: self.playfulness_milli,
        });
    }

    fn apply_mode(&self, fields: &mut EyeFields, now_ns: u64) {
        let elapsed = now_ns.saturating_sub(self.mode_started_ns);
        match self.mode {
            CharacterMode::Greeting => {
                fields.expression = if self.greeting_style == 2 {
                    Expression::Curious
                } else {
                    Expression::Greet
                };
                fields.brightness = fields.brightness.max(800);
                fields.pupil = fields.pupil.max(720);
                fields.color_rgb = match self.greeting_style {
                    0 => [255, 180, 70],
                    1 => [80, 255, 95],
                    _ => [255, 120, 150],
                };
                fields.gaze_x += self.saccade_x;
                fields.gaze_y += self.saccade_y;
            }
            CharacterMode::Tracking => {
                fields.gaze_x += self.saccade_x;
                fields.gaze_y += self.saccade_y;
            }
            CharacterMode::Lost => {
                fields.expression = Expression::Concerned;
                fields.gaze_x = self.saccade_x * 3;
                fields.gaze_y = 0;
                fields.lid = 140;
                fields.pupil = 520;
                fields.brightness = 600;
                fields.color_rgb = [200, 120, 180];
            }
            CharacterMode::Searching => {
                fields.expression = Expression::Curious;
                fields.gaze_x = scale_wave(elapsed, 3_000 * NS_PER_MS, 650);
                fields.gaze_y = 80;
                fields.lid = 100;
                fields.pupil = 560;
                fields.brightness = 650;
                fields.color_rgb = [150, 150, 210];
            }
            CharacterMode::Sleepy => {
                fields.expression = Expression::Sleepy;
                fields.gaze_x = 0;
                fields.gaze_y = 250;
                fields.lid = 620;
                fields.pupil = 400;
                fields.brightness = 180 + scale_wave(elapsed, 12_000 * NS_PER_MS, 30).abs();
                fields.color_rgb = [25, 60, 130];
                fields.blink = false;
            }
            CharacterMode::Idle => {
                fields.expression = Expression::Neutral;
                fields.gaze_x = scale_wave(elapsed, 20_000 * NS_PER_MS, 200) + self.saccade_x / 2;
                fields.gaze_y = scale_wave(
                    saturating_add(elapsed, 2_000 * NS_PER_MS),
                    30_000 * NS_PER_MS,
                    120,
                );
                fields.lid = 90;
                fields.pupil = 550;
                fields.brightness = 360 + scale_wave(elapsed, 10_000 * NS_PER_MS, 60).abs();
                fields.color_rgb = drifting_palette(elapsed);
            }
        }
    }

    fn apply_act(&self, fields: &mut EyeFields, now_ns: u64) {
        let Some(running) = self.active_act else {
            return;
        };
        let elapsed = now_ns.saturating_sub(running.started_ns);
        let phase = normalized_phase(elapsed, running.duration_ns);
        let pulse = symmetric_pulse(phase);
        let side = running.side;
        match running.act {
            CharacterAct::CuriousTilt => {
                fields.expression = Expression::Curious;
                fields.gaze_x += side * pulse * 120 / SCALE;
                fields.gaze_y += side * pulse * 70 / SCALE;
                fields.pupil += pulse * 160 / SCALE;
            }
            CharacterAct::DoubleTake => {
                let kick = if phase < 450 {
                    side * smooth_ramp(phase.min(220), 220) * 520 / SCALE
                } else {
                    -side * symmetric_pulse(phase.saturating_sub(450)) * 80 / SCALE
                };
                fields.gaze_x += kick;
                fields.blink |= (680..=760).contains(&phase);
            }
            CharacterAct::ExcitedWiggle => {
                fields.expression = Expression::Greet;
                fields.gaze_x += scale_wave(elapsed, 420 * NS_PER_MS, 180);
                fields.pupil += pulse * 220 / SCALE;
                fields.brightness += pulse * 320 / SCALE;
                fields.color_rgb = [255, 170, 45];
            }
            CharacterAct::LeanIn => {
                fields.expression = Expression::Curious;
                fields.pupil += pulse * 220 / SCALE;
                fields.gaze_y += pulse * 70 / SCALE;
            }
            CharacterAct::Nod => {
                fields.gaze_y += scale_wave(elapsed, 700 * NS_PER_MS, 90) * pulse / SCALE;
            }
            CharacterAct::SoftNod => {
                fields.expression = Expression::Greet;
                fields.gaze_y += scale_wave(elapsed, 850 * NS_PER_MS, 60) * pulse / SCALE;
                fields.brightness += pulse * 160 / SCALE;
            }
            CharacterAct::HappySquint => {
                fields.expression = Expression::Greet;
                fields.lid += pulse * 410 / SCALE;
                fields.pupil -= pulse * 90 / SCALE;
                fields.brightness += pulse * 320 / SCALE;
                fields.color_rgb = [255, 205, 60];
            }
            CharacterAct::PuppyEyes => {
                fields.expression = Expression::Curious;
                fields.pupil += pulse * 360 / SCALE;
                fields.gaze_y += pulse * 120 / SCALE;
                fields.brightness += pulse * 180 / SCALE;
                fields.color_rgb = [255, 155, 210];
            }
            CharacterAct::ShyDip => {
                fields.expression = Expression::Concerned;
                fields.gaze_y += pulse * 380 / SCALE;
                fields.gaze_x += side * pulse * 80 / SCALE;
                fields.lid += pulse * 250 / SCALE;
                fields.color_rgb = [230, 120, 180];
            }
            CharacterAct::Sparkle => {
                fields.pupil += pulse * 140 / SCALE;
                let flicker = if (phase / 120) % 2 == 0 { 420 } else { 60 };
                fields.brightness += flicker;
                fields.color_rgb = [90, 225, 255];
            }
            CharacterAct::BlinkFlourish => {
                fields.blink |= (120..=220).contains(&phase)
                    || (running.style != 0 && (470..=570).contains(&phase));
                fields.gaze_x += side * pulse * 40 / SCALE;
            }
            CharacterAct::LookAround => {
                fields.gaze_x += scale_wave(elapsed, 1_300 * NS_PER_MS, 680) * pulse / SCALE;
                fields.gaze_y += scale_wave(elapsed, 2_100 * NS_PER_MS, 180) * pulse / SCALE;
            }
            CharacterAct::PerkUp => {
                fields.expression = Expression::Curious;
                fields.pupil -= pulse * 120 / SCALE;
                fields.brightness += pulse * 380 / SCALE;
                fields.gaze_y -= pulse * 80 / SCALE;
            }
            CharacterAct::Daydream => {
                fields.gaze_x += scale_wave(elapsed, 2_400 * NS_PER_MS, 300) * pulse / SCALE;
                fields.gaze_y += scale_wave(elapsed, 3_100 * NS_PER_MS, 180) * pulse / SCALE;
                fields.brightness -= pulse * 220 / SCALE;
            }
            CharacterAct::Stretch => {
                fields.lid += pulse * 360 / SCALE;
                fields.gaze_y += scale_wave(elapsed, 1_600 * NS_PER_MS, 100) * pulse / SCALE;
            }
            CharacterAct::SweepScan => {
                fields.expression = Expression::Curious;
                fields.gaze_x += side * scale_wave(elapsed, 2_000 * NS_PER_MS, 620) * pulse / SCALE;
                fields.gaze_y += scale_wave(elapsed, 3_200 * NS_PER_MS, 110) * pulse / SCALE;
            }
            CharacterAct::HeadBob => {
                fields.gaze_y += scale_wave(elapsed, 520 * NS_PER_MS, 100) * pulse / SCALE;
                fields.brightness += pulse * 220 / SCALE;
            }
            CharacterAct::Sneeze => {
                fields.expression = Expression::Concerned;
                fields.gaze_y += if phase < 520 { -220 } else { 420 };
                fields.lid += if (430..=680).contains(&phase) {
                    850
                } else {
                    pulse * 260 / SCALE
                };
                fields.brightness += if (480..=650).contains(&phase) { 500 } else { 0 };
                fields.blink |= (520..=620).contains(&phase);
                fields.color_rgb = [255, 240, 210];
            }
            CharacterAct::Dance => {
                fields.expression = Expression::Greet;
                fields.gaze_x += scale_wave(elapsed, 650 * NS_PER_MS, 560) * pulse / SCALE;
                fields.gaze_y += scale_wave(elapsed, 430 * NS_PER_MS, 120) * pulse / SCALE;
                fields.brightness += pulse * 300 / SCALE;
                fields.color_rgb = dance_palette(phase, running.style);
            }
            CharacterAct::Sigh => {
                fields.lid += keyed_minimum_jerk(phase, &[(0, 0), (300, 90), (600, 70), (1000, 0)]);
                fields.brightness += keyed_minimum_jerk(phase, &[(0, 0), (350, -250), (1000, 0)]);
            }
            CharacterAct::BowBob => {
                fields.expression = Expression::Greet;
                fields.pupil += pulse * 170 / SCALE;
                fields.brightness += pulse * 300 / SCALE;
            }
            CharacterAct::StartleBoop => {
                fields.expression = Expression::Greet;
                fields.lid += keyed_minimum_jerk(phase, &[(0, 0), (80, -60), (500, 20), (1000, 0)]);
                fields.pupil +=
                    keyed_minimum_jerk(phase, &[(0, 0), (80, 260), (600, 120), (1000, 0)]);
                fields.brightness += pulse * 500 / SCALE;
                fields.color_rgb = [255, 190, 25];
                fields.blink |= (35..=80).contains(&phase) || (875..=930).contains(&phase);
            }
            CharacterAct::PlayBow => {
                fields.expression = Expression::Greet;
                fields.gaze_y +=
                    keyed_minimum_jerk(phase, &[(0, 0), (200, 500), (700, 300), (1000, 0)]);
                fields.pupil += pulse * 260 / SCALE;
                fields.brightness += pulse * 450 / SCALE;
                fields.color_rgb = [70, 255, 105];
                fields.blink |= (135..=185).contains(&phase);
            }
            CharacterAct::AffectionMelt => {
                fields.expression = Expression::Greet;
                fields.lid +=
                    keyed_minimum_jerk(phase, &[(0, 0), (350, 240), (550, 240), (1000, 0)]);
                fields.pupil += pulse * 170 / SCALE;
                fields.brightness += keyed_minimum_jerk(phase, &[(0, 0), (400, -200), (1000, 0)]);
                fields.color_rgb = [255, 150, 90];
                fields.blink |= (280..=340).contains(&phase);
            }
        }
    }

    fn apply_head_mode(&self, fields: &mut HeadFields, now_ns: u64) {
        let elapsed = now_ns.saturating_sub(self.mode_started_ns);
        match self.mode {
            CharacterMode::Greeting => {
                let duration = self
                    .greeting_until_ns
                    .saturating_sub(self.mode_started_ns)
                    .max(1);
                let phase = normalized_phase(elapsed, duration);
                let pulse = delayed_symmetric_pulse(elapsed, duration, HEAD_EYE_LEAD_NS);
                let side = if self.greeting_style & 1 == 0 { -1 } else { 1 };
                fields.bow += pulse * 85 / SCALE;
                fields.curl -= pulse * 55 / SCALE;
                fields.yaw += side * pulse * 45 / SCALE;
                fields.roll += side * pulse * 80 / SCALE;
                if phase > 820 {
                    fields.roll += side * symmetric_pulse((phase - 820) * 5) * 18 / SCALE;
                }
            }
            CharacterMode::Lost => {
                let pulse = delayed_symmetric_pulse(elapsed, LOST_DURATION_NS, HEAD_EYE_LEAD_NS);
                let side = if self.saccade_x < 0 { -1 } else { 1 };
                fields.bow -= pulse * 35 / SCALE;
                fields.curl += pulse * 45 / SCALE;
                fields.yaw += side * pulse * 55 / SCALE;
                fields.roll -= side * pulse * 45 / SCALE;
            }
            CharacterMode::Searching => {
                let duration = self
                    .searching_until_ns
                    .saturating_sub(self.mode_started_ns)
                    .max(1);
                let envelope = delayed_symmetric_pulse(elapsed, duration, HEAD_EYE_LEAD_NS);
                fields.bow += envelope * 28 / SCALE;
                fields.curl += scale_wave(elapsed, 2_200 * NS_PER_MS, 36) * envelope / SCALE;
                fields.yaw += scale_wave(elapsed, 3_000 * NS_PER_MS, 145) * envelope / SCALE;
                fields.roll -= scale_wave(elapsed, 3_000 * NS_PER_MS, 42) * envelope / SCALE;
            }
            CharacterMode::Sleepy => {
                let settled = minimum_jerk_ramp(normalized_phase(elapsed, 3 * NS_PER_SECOND));
                let breathe = scale_wave(elapsed, 9 * NS_PER_SECOND, 12);
                fields.bow += settled * 45 / SCALE + breathe;
                fields.curl += settled * 70 / SCALE + breathe;
                fields.yaw += scale_wave(elapsed, 17 * NS_PER_SECOND, 10);
                fields.roll += scale_wave(elapsed, 13 * NS_PER_SECOND, 14);
            }
            CharacterMode::Idle | CharacterMode::Tracking => {
                let living_elapsed = now_ns.saturating_sub(self.created_ns);
                let phase = self.life_phase_milli;
                // Two incommensurate slow waves make the energy ebb instead
                // of exposing one endlessly repeated mechanical loop.
                let energy_tide = 720
                    + phase_shifted_sine_wave(living_elapsed, phase[1], 299 * NS_PER_SECOND, SCALE)
                        * phase_shifted_sine_wave(
                            living_elapsed,
                            phase[3],
                            483 * NS_PER_SECOND,
                            280,
                        )
                        / SCALE;
                let pitch =
                    (phase_shifted_sine_wave(living_elapsed, phase[0], 14_600 * NS_PER_MS, 70)
                        + phase_shifted_sine_wave(living_elapsed, phase[1], 6_900 * NS_PER_MS, 32))
                        * energy_tide
                        / SCALE;
                let posture =
                    (phase_shifted_sine_wave(living_elapsed, phase[0], 17_000 * NS_PER_MS, 45)
                        + phase_shifted_sine_wave(living_elapsed, phase[5], 7_600 * NS_PER_MS, 22))
                        * energy_tide
                        / SCALE;
                fields.bow += pitch * 2 / 5 + posture;
                fields.curl += pitch * 3 / 5 + posture;
                fields.yaw +=
                    (phase_shifted_sine_wave(living_elapsed, phase[2], 20_300 * NS_PER_MS, 80)
                        + phase_shifted_sine_wave(living_elapsed, phase[3], 9_400 * NS_PER_MS, 35))
                        * energy_tide
                        / SCALE;
                fields.roll +=
                    (phase_shifted_sine_wave(living_elapsed, phase[4], 23_300 * NS_PER_MS, 70)
                        + phase_shifted_sine_wave(living_elapsed, phase[5], 8_600 * NS_PER_MS, 38))
                        * energy_tide
                        / SCALE
                        + phase_shifted_sine_wave(living_elapsed, phase[4], 5_556 * NS_PER_MS, 55);
            }
        }
    }

    fn apply_head_act(&self, fields: &mut HeadFields, now_ns: u64) {
        let Some(running) = self.active_act else {
            return;
        };
        let elapsed = now_ns.saturating_sub(running.started_ns);
        let phase = normalized_phase(elapsed, running.duration_ns);
        let pulse = delayed_symmetric_pulse(elapsed, running.duration_ns, HEAD_EYE_LEAD_NS);
        let side = running.side;
        let wave = |period_ms, amplitude| {
            scale_wave(elapsed, period_ms * NS_PER_MS, amplitude) * pulse / SCALE
        };

        match running.act {
            CharacterAct::CuriousTilt => {
                fields.bow += pulse * 45 / SCALE;
                fields.curl -= pulse * 38 / SCALE;
                fields.yaw += side * pulse * 65 / SCALE;
                fields.roll += side * pulse * 120 / SCALE;
            }
            CharacterAct::DoubleTake => {
                let kick = if phase < 430 {
                    minimum_jerk_ramp((phase * SCALE / 430).clamp(0, SCALE)) * 150 / SCALE
                } else {
                    -symmetric_pulse(((phase - 430) * SCALE / 570).clamp(0, SCALE)) * 45 / SCALE
                };
                fields.bow -= pulse * 28 / SCALE;
                fields.curl -= pulse * 25 / SCALE;
                fields.yaw += side * kick * pulse / SCALE;
                fields.roll -= side * kick * 2 / 5 * pulse / SCALE;
            }
            CharacterAct::ExcitedWiggle => {
                fields.bow += pulse * 75 / SCALE;
                fields.curl += wave(520, 48);
                fields.yaw += wave(420, 130);
                fields.roll -= wave(420, 155);
            }
            CharacterAct::LeanIn => {
                fields.bow += pulse * 165 / SCALE;
                fields.curl -= pulse * 62 / SCALE;
                fields.yaw += side * pulse * 24 / SCALE;
                fields.roll += side * pulse * 32 / SCALE;
            }
            CharacterAct::Nod => {
                fields.bow += wave(760, 58);
                fields.curl += wave(700, 135);
                fields.yaw += side * pulse * 12 / SCALE;
                fields.roll += side * pulse * 14 / SCALE;
            }
            CharacterAct::SoftNod => {
                fields.bow += wave(980, 42);
                fields.curl += wave(850, 82);
                fields.yaw += side * pulse * 14 / SCALE;
                fields.roll += side * pulse * 24 / SCALE;
            }
            CharacterAct::HappySquint => {
                fields.bow += pulse * 70 / SCALE;
                fields.curl -= pulse * 42 / SCALE;
                fields.yaw += side * pulse * 18 / SCALE;
                fields.roll += side * pulse * 38 / SCALE;
            }
            CharacterAct::PuppyEyes => {
                fields.bow += pulse * 92 / SCALE;
                fields.curl -= pulse * 96 / SCALE;
                fields.yaw += side * pulse * 42 / SCALE;
                fields.roll += side * pulse * 112 / SCALE;
            }
            CharacterAct::ShyDip => {
                fields.bow += pulse * 105 / SCALE;
                fields.curl += pulse * 125 / SCALE;
                fields.yaw += side * pulse * 92 / SCALE;
                fields.roll -= side * pulse * 98 / SCALE;
            }
            CharacterAct::Sparkle => {
                fields.bow += pulse * 35 / SCALE;
                fields.curl -= pulse * 32 / SCALE;
                fields.yaw += side * wave(360, 28);
                fields.roll += side * wave(310, 38);
            }
            CharacterAct::BlinkFlourish => {
                fields.bow += pulse * 30 / SCALE;
                fields.curl -= pulse * 24 / SCALE;
                fields.yaw += side * pulse * 34 / SCALE;
                fields.roll += side * pulse * 74 / SCALE;
            }
            CharacterAct::LookAround => {
                fields.bow += pulse * 24 / SCALE;
                fields.curl += wave(2_100, 34);
                fields.yaw += wave(1_300, 165);
                fields.roll -= wave(1_300, 52);
            }
            CharacterAct::PerkUp => {
                fields.bow -= pulse * 72 / SCALE;
                fields.curl -= pulse * 92 / SCALE;
                fields.yaw += side * pulse * 24 / SCALE;
                fields.roll += side * pulse * 45 / SCALE;
            }
            CharacterAct::Daydream => {
                fields.bow += pulse * 52 / SCALE;
                fields.curl += wave(3_100, 48);
                fields.yaw += wave(2_400, 95);
                fields.roll -= wave(2_900, 78);
            }
            CharacterAct::Stretch => {
                fields.bow -= pulse * 118 / SCALE;
                fields.curl -= pulse * 145 / SCALE;
                fields.yaw += side * pulse * 30 / SCALE;
                fields.roll += side * pulse * 52 / SCALE;
            }
            CharacterAct::SweepScan => {
                fields.bow += pulse * 28 / SCALE;
                fields.curl += wave(3_200, 35);
                fields.yaw += side * wave(2_000, 185);
                fields.roll -= side * wave(2_000, 55);
            }
            CharacterAct::HeadBob => {
                fields.bow += wave(600, 82);
                fields.curl += wave(520, 142);
                fields.yaw += side * wave(1_040, 22);
                fields.roll += side * wave(1_040, 28);
            }
            CharacterAct::Sneeze => {
                let anticipation = minimum_jerk_ramp((phase * SCALE / 620).clamp(0, SCALE));
                let release = if phase < 620 {
                    0
                } else {
                    symmetric_pulse(((phase - 620) * SCALE / 380).clamp(0, SCALE))
                };
                fields.bow -= anticipation * 55 / SCALE;
                fields.bow += release * 150 / SCALE;
                fields.curl -= anticipation * 80 / SCALE;
                fields.curl += release * 205 / SCALE;
                fields.yaw += side * release * 34 / SCALE;
                fields.roll -= side * release * 65 / SCALE;
                fields.bow = fields.bow * pulse / SCALE;
                fields.curl = fields.curl * pulse / SCALE;
                fields.yaw = fields.yaw * pulse / SCALE;
                fields.roll = fields.roll * pulse / SCALE;
            }
            CharacterAct::Dance => {
                fields.bow += pulse * 64 / SCALE + wave(860, 42);
                fields.curl += wave(430, 78);
                fields.yaw += wave(650, 175);
                fields.roll -= wave(650, 190);
            }
            CharacterAct::Sigh => {
                let posture = keyed_minimum_jerk(phase, &[(0, 0), (300, 70), (550, 64), (1000, 0)]);
                fields.bow += posture;
                fields.curl += posture;
                fields.yaw += side * pulse * 12 / SCALE;
                fields.roll += side * pulse * 16 / SCALE;
            }
            CharacterAct::BowBob => {
                let bob = keyed_minimum_jerk(
                    phase,
                    &[
                        (0, 0),
                        (100, 0),
                        (200, 105),
                        (310, -26),
                        (500, 105),
                        (610, -26),
                        (1000, 0),
                    ],
                );
                fields.bow += bob;
                fields.curl += bob;
                fields.yaw += side * pulse * 18 / SCALE;
                fields.roll += side * pulse * 26 / SCALE;
            }
            CharacterAct::StartleBoop => {
                let energy = i32::from(running.energy_milli);
                let recoil = 42 + 25 * energy / SCALE;
                let dip = 52 + 31 * energy / SCALE;
                let posture = keyed_minimum_jerk(
                    phase,
                    &[
                        (0, 0),
                        (100, -recoil),
                        (300, -recoil * 3 / 5),
                        (480, dip),
                        (620, -dip * 3 / 10),
                        (780, dip * 9 / 20),
                        (1000, 0),
                    ],
                );
                fields.bow += posture;
                fields.curl += posture;
                fields.yaw += side * pulse * 22 / SCALE;
                fields.roll += side * pulse * 52 / SCALE;
            }
            CharacterAct::PlayBow => {
                let energy = i32::from(running.energy_milli);
                let depth = 110 + 55 * energy / SCALE;
                let posture = keyed_minimum_jerk(
                    phase,
                    &[
                        (0, 0),
                        (180, depth),
                        (550, depth * 92 / 100),
                        (720, -depth * 22 / 100),
                        (850, depth * 8 / 100),
                        (1000, 0),
                    ],
                );
                let wiggle = if (220..=540).contains(&phase) {
                    scale_wave(
                        elapsed.saturating_sub(running.duration_ns * 22 / 100),
                        190 * NS_PER_MS,
                        78,
                    )
                } else {
                    0
                };
                fields.bow += posture;
                fields.curl += posture;
                fields.yaw += side * pulse * 52 / SCALE;
                fields.roll += wiggle * pulse / SCALE;
            }
            CharacterAct::AffectionMelt => {
                let nod = keyed_minimum_jerk(phase, &[(0, 0), (400, 68), (1000, 0)]);
                fields.bow += nod * 2 / 5;
                fields.curl += nod * 3 / 5;
                fields.yaw += side * pulse * 12 / SCALE;
                fields.roll += side * pulse * 34 / SCALE;
            }
        }
    }

    fn apply_blink(&mut self, fields: &mut EyeFields, now_ns: u64) {
        if self.greeting_blink_pending {
            fields.blink = true;
            self.greeting_blink_pending = false;
            let next_interval = self.random_range(BLINK_MIN_NS, BLINK_MAX_NS);
            self.next_blink_ns = saturating_add(now_ns, next_interval);
            return;
        }
        if now_ns < self.next_blink_ns {
            return;
        }
        if self.mode != CharacterMode::Sleepy {
            fields.blink = true;
        }
        self.next_blink_ns = saturating_add(now_ns, self.random_range(BLINK_MIN_NS, BLINK_MAX_NS));
    }

    fn next_u64(&mut self) -> u64 {
        let mut value = self.random_state;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.random_state = value;
        value
    }

    fn random_range(&mut self, minimum: u64, maximum: u64) -> u64 {
        debug_assert!(minimum <= maximum);
        let width = maximum.saturating_sub(minimum).saturating_add(1);
        minimum.saturating_add(self.next_u64() % width)
    }

    fn random_signed(&mut self, magnitude: i32) -> i32 {
        let width = u64::try_from(magnitude * 2 + 1).expect("small positive random width");
        i32::try_from(self.next_u64() % width).expect("small random sample") - magnitude
    }
}

const fn saturating_add(left: u64, right: u64) -> u64 {
    left.saturating_add(right)
}

fn clamp_signed(value: i32) -> i16 {
    i16::try_from(value.clamp(-SCALE, SCALE)).expect("normalized signed range fits i16")
}

fn clamp_unit(value: i32) -> u16 {
    u16::try_from(value.clamp(0, SCALE)).expect("normalized unit range fits u16")
}

fn normalized_phase(elapsed_ns: u64, duration_ns: u64) -> i32 {
    if duration_ns == 0 {
        return SCALE;
    }
    let scaled = u128::from(elapsed_ns.min(duration_ns)) * u128::from(SCALE as u32)
        / u128::from(duration_ns);
    i32::try_from(scaled).expect("normalized phase is bounded")
}

fn smooth_ramp(value: i32, maximum: i32) -> i32 {
    if maximum <= 0 {
        return SCALE;
    }
    let x = i64::from(value.clamp(0, maximum)) * i64::from(SCALE) / i64::from(maximum);
    // 3x² - 2x³ on the normalized integer scale.
    let x2 = x * x / i64::from(SCALE);
    let smooth = x2 * (3 * i64::from(SCALE) - 2 * x) / i64::from(SCALE);
    i32::try_from(smooth).expect("smooth ramp is normalized")
}

/// Quintic minimum-jerk position profile `10x^3 - 15x^4 + 6x^5`.
///
/// Integer arithmetic is bounded well below `i128::MAX` at the normalized
/// scale. The exact endpoints are retained, including zero velocity and zero
/// acceleration in the corresponding continuous polynomial.
fn minimum_jerk_ramp(value: i32) -> i32 {
    let x = i128::from(value.clamp(0, SCALE));
    let scale = i128::from(SCALE);
    let x2 = x * x;
    let x3 = x2 * x;
    let x4 = x3 * x;
    let x5 = x4 * x;
    let denominator = scale * scale * scale * scale;
    let numerator = 10 * x3 * scale * scale - 15 * x4 * scale + 6 * x5;
    let result = (numerator + denominator / 2) / denominator;
    i32::try_from(result.clamp(0, scale)).expect("minimum-jerk ramp is normalized")
}

fn minimum_jerk_pulse(phase: i32) -> i32 {
    let phase = phase.clamp(0, SCALE);
    if phase <= SCALE / 2 {
        minimum_jerk_ramp(phase * 2)
    } else {
        minimum_jerk_ramp((SCALE - phase) * 2)
    }
}

/// Interpolate a small, ordered keyframe curve with a minimum-jerk segment
/// between every pair. Values may be signed; phases are normalized to
/// `0..=SCALE`. Exact key values and endpoints are retained.
fn keyed_minimum_jerk(phase: i32, keys: &[(i32, i32)]) -> i32 {
    debug_assert!(keys.len() >= 2);
    debug_assert!(keys.windows(2).all(|pair| pair[0].0 < pair[1].0));
    let phase = phase.clamp(0, SCALE);
    if phase <= keys[0].0 {
        return keys[0].1;
    }
    for pair in keys.windows(2) {
        let (start_phase, start_value) = pair[0];
        let (end_phase, end_value) = pair[1];
        if phase <= end_phase {
            let span = end_phase - start_phase;
            let local = (phase - start_phase) * SCALE / span;
            let progress = minimum_jerk_ramp(local);
            let delta = i64::from(end_value) - i64::from(start_value);
            let interpolated =
                i64::from(start_value) + delta * i64::from(progress) / i64::from(SCALE);
            return i32::try_from(interpolated).expect("small keyframe values fit i32");
        }
    }
    keys[keys.len() - 1].1
}

fn delayed_symmetric_pulse(elapsed_ns: u64, duration_ns: u64, delay_ns: u64) -> i32 {
    if elapsed_ns <= delay_ns || duration_ns <= delay_ns {
        return 0;
    }
    minimum_jerk_pulse(normalized_phase(
        elapsed_ns - delay_ns,
        duration_ns - delay_ns,
    ))
}

fn symmetric_pulse(phase: i32) -> i32 {
    let phase = phase.clamp(0, SCALE);
    if phase <= SCALE / 2 {
        smooth_ramp(phase, SCALE / 2)
    } else {
        smooth_ramp(SCALE - phase, SCALE / 2)
    }
}

fn scale_wave(elapsed_ns: u64, period_ns: u64, amplitude: i32) -> i32 {
    if period_ns == 0 {
        return 0;
    }
    let phase = i32::try_from(
        u128::from(elapsed_ns % period_ns) * u128::from((SCALE * 4) as u32) / u128::from(period_ns),
    )
    .expect("wave phase is bounded");
    let normalized = match phase {
        0..=1000 => phase,
        1001..=2000 => 2000 - phase,
        2001..=3000 => -(phase - 2000),
        _ => -(4000 - phase),
    };
    normalized * amplitude / SCALE
}

fn phase_shifted_sine_wave(
    elapsed_ns: u64,
    phase_offset_milli: u16,
    period_ns: u64,
    amplitude: i32,
) -> i32 {
    if period_ns == 0 {
        return 0;
    }
    let phase_offset_ns = u64::try_from(
        u128::from(period_ns) * u128::from(phase_offset_milli)
            / u128::try_from(SCALE).expect("positive normalized scale"),
    )
    .expect("normalized offset never exceeds its u64 period");
    let shifted = u64::try_from(
        (u128::from(elapsed_ns % period_ns) + u128::from(phase_offset_ns)) % u128::from(period_ns),
    )
    .expect("modulo period fits u64");
    // Fast integer sine approximation. Unlike the retained triangle wave
    // used by deliberately rhythmic acts, this has no velocity reversal at
    // zero crossings and its slope joins across the period boundary. The
    // 0.225 correction keeps the maximum error small without adding a float
    // or libm dependency to this no-allocation frame path.
    let doubled_phase =
        i64::try_from(u128::from(shifted) * u128::from((2 * SCALE) as u32) / u128::from(period_ns))
            .expect("sine phase is bounded");
    let x = if doubled_phase <= i64::from(SCALE) {
        doubled_phase
    } else {
        doubled_phase - 2 * i64::from(SCALE)
    };
    let scale = i64::from(SCALE);
    let parabola = 4 * x - 4 * x * x.abs() / scale;
    let corrected = parabola + 225 * (parabola * parabola.abs() / scale - parabola) / 1_000;
    i32::try_from(corrected * i64::from(amplitude) / scale)
        .expect("bounded sine amplitude fits i32")
}

fn drifting_palette(elapsed_ns: u64) -> [u8; 3] {
    const COLORS: [[u8; 3]; 6] = [
        [70, 180, 220],
        [85, 150, 235],
        [120, 120, 235],
        [165, 105, 220],
        [100, 155, 235],
        [65, 195, 210],
    ];
    let index = usize::try_from((elapsed_ns / (8 * NS_PER_SECOND)) % COLORS.len() as u64)
        .expect("palette index fits usize");
    COLORS[index]
}

fn dance_palette(phase: i32, style: u8) -> [u8; 3] {
    const COLORS: [[u8; 3]; 6] = [
        [255, 75, 80],
        [255, 175, 45],
        [120, 235, 70],
        [60, 210, 235],
        [105, 105, 255],
        [235, 80, 220],
    ];
    let phase = usize::try_from(phase.clamp(0, SCALE)).expect("phase fits usize");
    COLORS[(phase / 167 + usize::from(style)) % COLORS.len()]
}

#[cfg(test)]
mod tests {
    extern crate std;

    use kiko_expression_core::{Deadline, MonotonicTimestamp};
    use kiko_eye_protocol::{EyeFlags, SignedUnit};

    use super::*;

    fn prepared(now_ns: u64) -> PreparedEyeIntent {
        let intent = EyeIntent::new(
            SignedUnit::try_new(0).unwrap(),
            SignedUnit::try_new(0).unwrap(),
            UnitAmount::try_new(60).unwrap(),
            UnitAmount::try_new(500).unwrap(),
            UnitAmount::try_new(700).unwrap(),
            Expression::Neutral,
            EyeFlags::NONE,
            [10, 20, 30],
        );
        let generated_at = MonotonicTimestamp::from_nanos_since_epoch(now_ns);
        let deadline = Deadline::after(
            generated_at,
            kiko_expression_core::NonZeroDuration::try_from_nanos(NS_PER_SECOND).unwrap(),
        )
        .unwrap();
        PreparedEyeIntent::from_parts(intent, generated_at, Some(deadline))
    }

    #[test]
    fn first_face_greets_then_tracks_and_loss_searches_idles_and_sleeps() {
        let mut engine = AutonomicCharacterEngine::new(7);
        let start = 10 * NS_PER_SECOND;
        engine.render(
            MonotonicTimestamp::from_nanos_since_epoch(start),
            true,
            prepared(start),
        );
        assert_eq!(engine.mode(), CharacterMode::Greeting);

        let tracked_at = start + GREETING_MAX_NS + 1;
        engine.render(
            MonotonicTimestamp::from_nanos_since_epoch(tracked_at),
            true,
            prepared(tracked_at),
        );
        assert_eq!(engine.mode(), CharacterMode::Tracking);

        let lost_at = tracked_at + NS_PER_MS;
        engine.render(
            MonotonicTimestamp::from_nanos_since_epoch(lost_at),
            false,
            prepared(lost_at),
        );
        assert_eq!(engine.mode(), CharacterMode::Lost);

        let search_at = lost_at + LOST_DURATION_NS;
        engine.render(
            MonotonicTimestamp::from_nanos_since_epoch(search_at),
            false,
            prepared(search_at),
        );
        assert_eq!(engine.mode(), CharacterMode::Searching);

        let idle_at = search_at + SEARCH_MAX_NS + 1;
        engine.render(
            MonotonicTimestamp::from_nanos_since_epoch(idle_at),
            false,
            prepared(idle_at),
        );
        assert_eq!(engine.mode(), CharacterMode::Idle);

        let sleepy_at = tracked_at + SLEEPY_AFTER_IDLE_NS + SEARCH_MAX_NS + NS_PER_SECOND;
        engine.render(
            MonotonicTimestamp::from_nanos_since_epoch(sleepy_at),
            false,
            prepared(sleepy_at),
        );
        assert_eq!(engine.mode(), CharacterMode::Sleepy);
    }

    #[test]
    fn every_retained_act_stays_inside_protocol_domains() {
        let start = 5 * NS_PER_SECOND;
        for (index, act) in CharacterAct::ALL.iter().copied().enumerate() {
            let mut engine = AutonomicCharacterEngine::new(index as u64 + 1);
            engine.initialize_if_needed(start);
            engine.mode = if act.eligible(CharacterMode::Tracking) {
                CharacterMode::Tracking
            } else {
                CharacterMode::Idle
            };
            engine.active_act = Some(RunningAct {
                act,
                started_ns: start,
                duration_ns: 5 * NS_PER_SECOND,
                side: if index % 2 == 0 { -1 } else { 1 },
                style: (index % 3) as u8,
                energy_milli: 500,
            });
            let mut joint_moved = [false; 4];
            for step in 0..=100_u64 {
                let now_ns = start + step * 50 * NS_PER_MS;
                let output = engine.render_character(
                    MonotonicTimestamp::from_nanos_since_epoch(now_ns),
                    engine.mode == CharacterMode::Tracking,
                    prepared(now_ns),
                );
                let intent = output.eye().intent();
                assert!((-1000..=1000).contains(&intent.gaze_x().get()));
                assert!((-1000..=1000).contains(&intent.gaze_y().get()));
                assert!(intent.lid().get() <= 1000);
                assert!(intent.pupil().get() <= 1000);
                assert!(intent.brightness().get() <= 1000);
                for (moved, amount) in joint_moved.iter_mut().zip(output.head().amounts()) {
                    *moved |= amount.get() != 0;
                    assert!((-CHARACTER_HEAD_SCALE..=CHARACTER_HEAD_SCALE).contains(&amount.get()));
                }
            }
            assert_eq!(
                joint_moved,
                [true; 4],
                "{} must choreograph all four head joints",
                act.as_str()
            );
        }
    }

    #[test]
    fn eyes_lead_each_head_act_and_the_head_returns_to_exact_natural() {
        let start = 8 * NS_PER_SECOND;
        let duration = 2 * NS_PER_SECOND;
        let mut engine = AutonomicCharacterEngine::new(19);
        engine.initialize_if_needed(start);
        engine.mode = CharacterMode::Tracking;
        engine.active_act = Some(RunningAct {
            act: CharacterAct::CuriousTilt,
            started_ns: start,
            duration_ns: duration,
            side: 1,
            style: 0,
            energy_milli: 0,
        });

        let sample_act = |now_ns| {
            let mut fields = HeadFields::default();
            engine.apply_head_act(&mut fields, now_ns);
            CharacterHeadOverlay::from_clamped(fields)
        };

        assert!(sample_act(start).is_natural());
        assert!(sample_act(start + HEAD_EYE_LEAD_NS).is_natural());
        assert!(!sample_act(start + 500 * NS_PER_MS).is_natural());
        assert!(sample_act(start + duration).is_natural());
    }

    #[test]
    fn idle_living_motion_uses_every_joint_then_rests_and_wakes() {
        let start = 8 * NS_PER_SECOND;
        let mut engine = AutonomicCharacterEngine::new(20);
        engine.initialize_if_needed(start);
        engine.mode = CharacterMode::Idle;
        engine.mode_started_ns = start;
        engine.last_face_ns = start;
        engine.life_phase_milli = [0; 6];
        engine.next_act_ns = u64::MAX;

        let mut moved = [false; 4];
        for step in 1..=200_u64 {
            let now_ns = start + step * 50 * NS_PER_MS;
            let frame = engine.render_character(
                MonotonicTimestamp::from_nanos_since_epoch(now_ns),
                false,
                prepared(now_ns),
            );
            for (axis_moved, amount) in moved.iter_mut().zip(frame.head().amounts()) {
                *axis_moved |= amount.get() != 0;
            }
        }
        assert_eq!(moved, [true; 4]);

        let rested_at = start + REST_AFTER_IDLE_NS + REST_EASE_NS;
        let rested = engine.render_character(
            MonotonicTimestamp::from_nanos_since_epoch(rested_at),
            false,
            prepared(rested_at),
        );
        assert!(rested.head().is_natural());

        let sleepy_at = start + SLEEPY_AFTER_IDLE_NS + NS_PER_SECOND;
        let still_rested = engine.render_character(
            MonotonicTimestamp::from_nanos_since_epoch(sleepy_at),
            false,
            prepared(sleepy_at),
        );
        assert_eq!(still_rested.mode(), CharacterMode::Sleepy);
        assert!(still_rested.head().is_natural());

        let wake_at = sleepy_at + NS_PER_MS;
        engine.render_character(
            MonotonicTimestamp::from_nanos_since_epoch(wake_at),
            true,
            prepared(wake_at),
        );
        let moving_at = wake_at + HEAD_EYE_LEAD_NS + 200 * NS_PER_MS;
        let awake = engine.render_character(
            MonotonicTimestamp::from_nanos_since_epoch(moving_at),
            true,
            prepared(moving_at),
        );
        assert_eq!(awake.mode(), CharacterMode::Greeting);
        assert!(!awake.head().is_natural());
    }

    #[test]
    fn unattended_rest_envelope_has_exact_endpoints_and_never_rises() {
        let start = 9 * NS_PER_SECOND;
        let mut engine = AutonomicCharacterEngine::new(21);
        engine.initialize_if_needed(start);
        engine.mode = CharacterMode::Idle;
        engine.last_face_ns = start;

        assert_eq!(engine.rest_envelope(start + REST_AFTER_IDLE_NS), SCALE);
        let mut previous = SCALE;
        for step in 1..=600_u64 {
            let now_ns = start + REST_AFTER_IDLE_NS + step * 10 * NS_PER_MS;
            let envelope = engine.rest_envelope(now_ns);
            assert!(envelope <= previous);
            assert!((0..=SCALE).contains(&envelope));
            previous = envelope;
        }
        assert_eq!(previous, 0);

        engine.mode = CharacterMode::Sleepy;
        assert_eq!(engine.rest_envelope(start + SLEEPY_AFTER_IDLE_NS), 0);
        engine.mode = CharacterMode::Tracking;
        assert_eq!(engine.rest_envelope(start + SLEEPY_AFTER_IDLE_NS), SCALE);
    }

    #[test]
    fn minimum_jerk_curve_has_exact_endpoints_and_is_monotonic() {
        assert_eq!(minimum_jerk_ramp(0), 0);
        assert_eq!(minimum_jerk_ramp(SCALE), SCALE);
        let mut previous = 0;
        for sample in 0..=SCALE {
            let actual = minimum_jerk_ramp(sample);
            assert!(actual >= previous);
            assert!((0..=SCALE).contains(&actual));
            previous = actual;
        }
    }

    #[test]
    fn living_sine_has_exact_quadrants_and_stays_bounded() {
        let period = 4 * NS_PER_SECOND;
        assert_eq!(phase_shifted_sine_wave(0, 0, period, 100), 0);
        assert_eq!(phase_shifted_sine_wave(NS_PER_SECOND, 0, period, 100), 100);
        assert_eq!(
            phase_shifted_sine_wave(2 * NS_PER_SECOND, 0, period, 100),
            0
        );
        assert_eq!(
            phase_shifted_sine_wave(3 * NS_PER_SECOND, 0, period, 100),
            -100
        );
        assert_eq!(phase_shifted_sine_wave(0, 250, period, 100), 100);
        for step in 0..=4_000_u64 {
            let sample = phase_shifted_sine_wave(step * NS_PER_MS, 0, period, 100);
            assert!((-100..=100).contains(&sample));
        }
    }

    #[test]
    fn external_head_overlay_parser_rejects_each_invalid_axis() {
        assert_eq!(
            CharacterHeadOverlay::try_new(1_001, 0, 0, 0),
            Err(CharacterHeadOverlayParseError {
                axis: CharacterHeadAxis::Bow,
                source: CharacterHeadAmountError { value: 1_001 },
            })
        );
        assert_eq!(
            CharacterHeadOverlay::try_new(0, -1_001, 0, 0),
            Err(CharacterHeadOverlayParseError {
                axis: CharacterHeadAxis::Curl,
                source: CharacterHeadAmountError { value: -1_001 },
            })
        );
        assert_eq!(
            CharacterHeadOverlay::try_new(0, 0, 1_001, 0),
            Err(CharacterHeadOverlayParseError {
                axis: CharacterHeadAxis::Yaw,
                source: CharacterHeadAmountError { value: 1_001 },
            })
        );
        assert_eq!(
            CharacterHeadOverlay::try_new(0, 0, 0, -1_001),
            Err(CharacterHeadOverlayParseError {
                axis: CharacterHeadAxis::Roll,
                source: CharacterHeadAmountError { value: -1_001 },
            })
        );
    }

    #[test]
    fn overlay_preserves_exact_freshness_identity() {
        let mut engine = AutonomicCharacterEngine::new(11);
        let input = prepared(1_000);
        let output = engine.render(
            MonotonicTimestamp::from_nanos_since_epoch(1_000),
            false,
            input,
        );
        assert_eq!(output.generated_at(), input.generated_at());
        assert_eq!(
            output.valid_until_exclusive(),
            input.valid_until_exclusive()
        );
    }

    #[test]
    fn greeting_blink_is_one_command_edge_not_a_streamed_level() {
        let mut engine = AutonomicCharacterEngine::new(13);
        let start = NS_PER_SECOND;
        let first = engine.render(
            MonotonicTimestamp::from_nanos_since_epoch(start),
            true,
            prepared(start),
        );
        let second_at = start + 50 * NS_PER_MS;
        let second = engine.render(
            MonotonicTimestamp::from_nanos_since_epoch(second_at),
            true,
            prepared(second_at),
        );

        assert!(first.intent().flags().requests_blink());
        assert!(!second.intent().flags().requests_blink());
    }

    #[test]
    fn micro_saccade_holds_then_returns_to_center_with_a_smooth_decay() {
        let start = 5 * NS_PER_SECOND;
        let mut engine = AutonomicCharacterEngine::new(23);
        engine.initialize_if_needed(start);
        engine.mode = CharacterMode::Idle;
        engine.next_saccade_ns = start;

        engine.update_saccade(start);
        let peak = (engine.saccade_x, engine.saccade_y);
        assert!(peak.0.abs() <= 14);
        assert!(peak.1.abs() <= 9);
        assert_ne!(peak, (0, 0), "the retained seed produces a visible sample");
        assert!(
            (start + SACCADE_MIN_NS..=start + SACCADE_MAX_NS).contains(&engine.next_saccade_ns)
        );

        engine.update_saccade(start + SACCADE_HOLD_NS);
        assert_eq!((engine.saccade_x, engine.saccade_y), peak);

        engine.update_saccade(start + SACCADE_HOLD_NS + SACCADE_DECAY_NS / 2);
        assert!(engine.saccade_x.abs() <= peak.0.abs());
        assert!(engine.saccade_y.abs() <= peak.1.abs());

        engine.update_saccade(start + SACCADE_HOLD_NS + SACCADE_DECAY_NS);
        assert_eq!((engine.saccade_x, engine.saccade_y), (0, 0));
    }

    #[test]
    fn face_tracking_suppresses_random_saccade_offsets() {
        let start = 7 * NS_PER_SECOND;
        let mut engine = AutonomicCharacterEngine::new(29);
        engine.initialize_if_needed(start);
        engine.mode = CharacterMode::Tracking;
        engine.mode_started_ns = start;
        engine.ever_saw_face = true;
        engine.last_face_ns = start;
        engine.next_act_ns = u64::MAX;
        engine.saccade_peak_x = 14;
        engine.saccade_peak_y = -9;
        engine.saccade_x = 14;
        engine.saccade_y = -9;
        engine.next_saccade_ns = start + NS_PER_SECOND;

        let output = engine.render_character(
            MonotonicTimestamp::from_nanos_since_epoch(start),
            true,
            prepared(start),
        );
        assert_eq!(output.eye().intent().gaze_x().get(), 0);
        assert_eq!(output.eye().intent().gaze_y().get(), 0);
        assert_eq!((engine.saccade_x, engine.saccade_y), (0, 0));
    }

    #[test]
    fn pet_episode_boundary_rejects_impossible_facts_and_classifies_exactly() {
        assert_eq!(
            CharacterPetEpisode::try_new(Duration::ZERO, 0, 0, false, true),
            Err(CharacterPetEpisodeError::InvalidDuration { duration_ns: 0 })
        );
        assert_eq!(
            CharacterPetEpisode::try_new(Duration::from_secs(1), 1, 0, false, false),
            Err(CharacterPetEpisodeError::DeltaWithoutSamples {
                accumulated_max_delta_ticks: 1,
            })
        );

        let boop = CharacterPetEpisode::try_new(Duration::from_millis(600), 2, 1, false, true)
            .expect("valid tap episode");
        let just_below_play =
            CharacterPetEpisode::try_new(Duration::from_secs(2), 17, 3, false, false)
                .expect("valid delta evidence");
        let play = CharacterPetEpisode::try_new(Duration::from_secs(2), 18, 3, false, false)
            .expect("valid delta evidence");
        let comfy = CharacterPetEpisode::try_new(Duration::from_secs(6), 18, 3, true, false)
            .expect("valid comfy evidence");

        assert_eq!(boop.reaction(), CharacterPetReaction::Boop);
        assert_eq!(just_below_play.reaction(), CharacterPetReaction::Affection);
        assert_eq!(play.reaction(), CharacterPetReaction::Play);
        assert_eq!(comfy.reaction(), CharacterPetReaction::Affection);
    }

    #[test]
    fn pet_reaction_preempts_scheduled_act_and_playfulness_decays_without_tick_loss() {
        let start = 11 * NS_PER_SECOND;
        let mut engine = AutonomicCharacterEngine::new(31);
        engine.initialize_if_needed(start);
        engine.active_act = Some(RunningAct {
            act: CharacterAct::Daydream,
            started_ns: start,
            duration_ns: 5 * NS_PER_SECOND,
            side: -1,
            style: 0,
            energy_milli: 0,
        });
        let boop = CharacterPetEpisode::try_new(Duration::from_millis(600), 0, 0, false, true)
            .expect("valid tap episode");
        assert_eq!(
            engine.note_pet_episode(MonotonicTimestamp::from_nanos_since_epoch(start), boop,),
            CharacterPetReaction::Boop
        );
        assert_eq!(engine.active_act(), Some(CharacterAct::StartleBoop));
        assert_eq!(engine.playfulness_milli(), 450);

        for tick in 1..=900_u64 {
            let now = start + tick * 50 * NS_PER_MS;
            engine.decay_playfulness(now);
        }
        assert_eq!(engine.playfulness_milli(), 0);
        assert_eq!(engine.playfulness_decay_remainder, 0);
    }

    #[test]
    fn same_seed_and_inputs_are_deterministic() {
        let mut left = AutonomicCharacterEngine::new(42);
        let mut right = AutonomicCharacterEngine::new(42);
        for step in 0..1_000_u64 {
            let now_ns = step * 50 * NS_PER_MS;
            let face = (step / 100) % 3 != 0;
            let left_output = left.render(
                MonotonicTimestamp::from_nanos_since_epoch(now_ns),
                face,
                prepared(now_ns),
            );
            let right_output = right.render(
                MonotonicTimestamp::from_nanos_since_epoch(now_ns),
                face,
                prepared(now_ns),
            );
            assert_eq!(left.mode(), right.mode());
            assert_eq!(left.active_act(), right.active_act());
            assert_eq!(left_output, right_output);
        }
    }
}
