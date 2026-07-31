//! Bounded autonomic eye behavior layered over fresh visual reactions.
//!
//! The visual mixer owns face/scene attention. This module owns only the
//! character-like timing that makes an otherwise idle renderer feel alive:
//! greeting, loss/search/sleep transitions, blinks, micro-saccades, and a
//! finite act library derived from Kiko's retained expression-engine behavior.
//! It performs no allocation, I/O, sleeping, wall-clock access, or head
//! actuation. Physical head motion remains exclusively owned by the calibrated
//! gaze path.

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
const SACCADE_MIN_NS: u64 = 180 * NS_PER_MS;
const SACCADE_MAX_NS: u64 = 550 * NS_PER_MS;
const NEVER_RUN: u64 = u64::MAX;

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
}

impl CharacterAct {
    const ALL: [Self; 19] = [
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
        }
    }

    const fn eligible(self, mode: CharacterMode) -> bool {
        match self {
            Self::LookAround | Self::PerkUp | Self::Daydream | Self::Stretch => {
                matches!(mode, CharacterMode::Idle)
            }
            Self::Sparkle | Self::BlinkFlourish | Self::SweepScan | Self::Sneeze | Self::Dance => {
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
            | Self::HeadBob => matches!(mode, CharacterMode::Tracking),
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
        };
        (milliseconds.0 * NS_PER_MS, milliseconds.1 * NS_PER_MS)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RunningAct {
    act: CharacterAct,
    started_ns: u64,
    duration_ns: u64,
    side: i32,
    style: u8,
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
    saccade_x: i32,
    saccade_y: i32,
    next_act_ns: u64,
    active_act: Option<RunningAct>,
    last_run_ns: [u64; CharacterAct::ALL.len()],
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
            saccade_x: 0,
            saccade_y: 0,
            next_act_ns: 0,
            active_act: None,
            last_run_ns: [NEVER_RUN; CharacterAct::ALL.len()],
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

    /// Apply one autonomic sample without changing the reaction's freshness.
    pub fn render(
        &mut self,
        now: MonotonicTimestamp,
        face_present: bool,
        prepared: PreparedEyeIntent,
    ) -> PreparedEyeIntent {
        let now_ns = now.nanos_since_epoch();
        self.initialize_if_needed(now_ns);
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
        prepared.with_intent(fields.into_intent())
    }

    fn initialize_if_needed(&mut self, now_ns: u64) {
        if self.initialized {
            return;
        }
        self.initialized = true;
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
        if now_ns < self.next_saccade_ns {
            return;
        }
        self.saccade_x = self.random_signed(26);
        self.saccade_y = self.random_signed(18);
        self.next_saccade_ns =
            saturating_add(now_ns, self.random_range(SACCADE_MIN_NS, SACCADE_MAX_NS));
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
            });
            for step in 0..=100_u64 {
                let now_ns = start + step * 50 * NS_PER_MS;
                let output = engine.render(
                    MonotonicTimestamp::from_nanos_since_epoch(now_ns),
                    engine.mode == CharacterMode::Tracking,
                    prepared(now_ns),
                );
                let intent = output.intent();
                assert!((-1000..=1000).contains(&intent.gaze_x().get()));
                assert!((-1000..=1000).contains(&intent.gaze_y().get()));
                assert!(intent.lid().get() <= 1000);
                assert!(intent.pupil().get() <= 1000);
                assert!(intent.brightness().get() <= 1000);
            }
        }
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
