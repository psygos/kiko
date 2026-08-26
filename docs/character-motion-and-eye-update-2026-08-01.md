# Four-joint character motion and eye-update cue

Date: 2026-08-01

Scope: canonical host/runtime and RP2350 eye-firmware source

Hardware action in this change: none

Calibration supersession: the character design and source evidence in this
document remain relevant, but any natural pose, curl range, pitch share, or
torque value predating the 2026-08-06 bow-servo replacement is historical.
Current values are in `docs/kiko-head-assembly-policy-2026-08-27.md`.

## Outcome

Kiko now has one deterministic character decision per accepted RGB frame. The
decision contains the existing KEP2 eye intent plus normalized bow, curl, yaw,
and roll overlays. Eyes and head therefore share one mode, act, clock, and
freshness decision instead of two independent animation loops.

This does not add a servo-bus owner. The existing reviewed head actor remains
the only path that may convert a character overlay into an encoder target. It
combines face gaze and expression before submitting one target. The same
base-zero exclusive lease, service result, readback, hard envelopes, and
terminal-fault behavior continue to apply.

## Motion grammar

The director retains the six modes and 19 acts recovered from the earlier
expression-engine source. It uses the four joints as a coordinated mechanism:

| Intent | Primary read | Secondary detail |
| --- | --- | --- |
| greeting | small lift and friendly tilt | tiny yaw acknowledgement and settling roll |
| curious / puppy eyes | asymmetric roll | slight yaw and forward bow |
| nod / soft nod / head bob | curl with bow support | deliberately small yaw/roll prevents mechanical symmetry |
| shy dip | curl and bow down | gaze-away yaw with counter-roll |
| perk up / stretch | lift through curl and bow | restrained side bias |
| look around / sweep scan | yaw sweep | slower curl and counter-roll make the eyes lead the scan |
| excited wiggle / dance | yaw-roll counter-motion | bow/curl bounce |
| sneeze | minimum-jerk anticipation then release | brief asymmetric recovery |
| sleepy | slow settled droop | low-amplitude breathing and long yaw/roll drift |

The eyes lead physical head intent by exactly 120 ms. Head pulses use the
quintic minimum-jerk position curve `10x^3 - 15x^4 + 6x^5`, with exact integer
endpoints. Each finite act returns all axes to the exact semantic natural pose;
there is no accumulating offset or floating-point drift.

The director is fixed-size and allocation-free. Its pseudorandom choices are
seeded by the stream epoch, so identical typed inputs and sample times replay
identically. No performance improvement is claimed because this change does
not include a reproducible runtime benchmark.

## Physical mapping boundary

Character axes are dimensionless values in `[-1000, 1000]`. Positive means
`character-positive`; it does not mean increasing encoder ticks. A reviewed
head-gaze policy may declare the signed encoder offset corresponding to
`+1000` for each named joint. Parsing rejects:

- a missing physical mapping when physical character motion is requested;
- a zero full-scale offset for any of the four joints;
- either signed full-scale endpoint outside the existing hard envelope; and
- any face-gaze-plus-character composition outside a hard envelope.

Mapping uses signed nearest-tick rounding. It does not clamp. If a mapping is
absent or a combined target is unsafe, the returned disposition says exactly
why the expressive overlay was withheld. Face gaze can still use its safe base
proposal.

The four-axis calibration was already present and active on the Nano. Read-only
inspection bound it to configuration SHA-256
`0d98af8c9fde9f34686375ff49a963c9f290386e3bdb7d623390cc3fbe82085f`
and source SHA-256
`a41f6d4717dd1b301da04e89906587aa6d4f1b9238e041ab9e9480e8bc1938ad`.
The qualification template now retains its exact natural pose, per-axis
software envelopes, gaze signs/scales, and character-positive offsets. Its
remaining placeholders concern controller timing/limits and review-artifact
binding, not servo recalibration.

## Rerun seam

`PreparedCharacterFrame` exposes the mode, optional named act, four normalized
joint values, eye intent, generated-at timestamp, and exclusive freshness
deadline without exposing transport bytes or servo authority. This is the
canonical diagnostic/Rerun seam. Logging these values may say “prepared”; it
must not say “physically applied” unless the separate head and KEP2 receipts
are joined to the same decision.

The integration avoids an extra RGB clone: face perception, character
preparation, and head evaluation borrow the same authoritative queued frame.

## Matrix-green eye update cue

The RP2350 eye application renders a 2.4-second falling green Matrix pattern
on both measured 56-LED panel geometries before it enumerates the KEP2 USB
endpoint. The panels are phase-shifted, the renderer allocates nothing, and
every output channel remains under the existing brightness ceiling. Delaying
USB is required so firmware cannot acknowledge an eye command while a boot
animation is still overriding the pixels.

ROM BOOTSEL owns the RP2350 during a UF2 copy, so the application cannot drive
the physical LEDs during that interval. The implemented cue begins immediately
when the copied application actually boots. A continuous physical Matrix
pattern through the copy itself would require a custom bootloader or an
independent display controller. This implementation reports that limitation
instead of pretending application code runs while BOOTSEL owns the MCU.

## Read-only live Nano evidence

The live Nano guardian was inspected read-only and was not stopped, restarted,
or replaced. At inspection time it retained the proven four-servo natural
ticks `[2174, 2570, 1637, 3047]` and an above/behind OAK-to-head declaration of
`0.25 m` and `0.20 m`.

The live source established the current roll semantic polarity and the
existing face-follow decomposition. Those facts informed the named mapping
contract; they were not silently installed as a reviewed canonical physical
mapping. The existing guardian remained the active hardware owner throughout
this source change.

## Verification boundary

Host tests cover deterministic replay, all 19 acts, all four joints, exact eye
lead, exact return-to-natural, monotonic minimum-jerk arithmetic, invalid
normalized inputs, signed nearest-tick conversion, mapping completeness,
hard-envelope rejection without clamping, missing-mapping disposition, and
the single-worker head-before-eye failure order. Firmware tests cover dynamic
green output, left/right phase separation, power limits, and `u64::MAX` time
arithmetic. The embedded target is compile-checked separately.

These checks prove software behavior only. No eye firmware was flashed and no
servo was moved by this change. A physical rollout still requires an explicit
guardian handoff, a fresh non-reused firmware build identity, exact KEP2
identity admission, binding the existing four-axis calibration into the
reviewed physical policy, and observed motion with the wheels safely removed.
