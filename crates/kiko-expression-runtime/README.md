# kiko-expression-runtime

Transport-independent host boundary between `kiko-expression-core`, borrowed
RGB frames, and the KEP2 contract in `kiko-eye-protocol`. The crate is
`no_std`, forbids unsafe code, performs no I/O, and makes no heap allocation.
It does not own the camera, serial/USB port, clocks, head, firmware, or display.

## RGB scene motion

`SceneMotionExtractor` accepts only `RgbFrameView`, whose layout and exact byte
length have already been parsed by `kiko-expression-core`. Its sampling is
fully specified:

- a caller-selected `columns × rows` grid, bounded to 4,096 samples;
- cell-centre pixel `floor(((2*i + 1) * extent) / (2*cell_count))`;
- frames smaller than the grid are rejected, so samples are unique per axis;
- interleaved RGB and BGR are handled explicitly and row stride/padding is
  honoured; and
- integer BT.601 luminance `(77R + 150G + 29B + 128) >> 8`.

Two fixed sampled-luminance buffers are retained; source pixels stay borrowed
and no frame is copied. For adjacent frames, the extractor finds the median of
all signed luminance deltas and subtracts it before motion scoring. An
even-sized sample set's half-luminance median is retained exactly in doubled
integer units instead of being rounded toward either sign. This suppresses
uniform exposure changes without hiding localized change. Active samples cross
an explicit residual-luminance threshold. Strength is the active residual
energy divided by the equivalent `sample_count × 510` full-luminance scale;
centroid is residual-weighted in image coordinates. All reported
normalization uses integer round-to-nearest.

The first accepted frame is `ColdStart`, a valid comparison without detected
motion is `NoMotion`, and a nonzero result is `Motion`. These are not collapsed
into one ambiguous `None`. After cold start, accepted frames must have the same
stream epoch and layout, the exact next frame sequence, a strictly increasing
observation timestamp, a non-regressing host clock, and a freshness window
alive at processing time. Gaps, duplicates, reordering, clock faults, stale or
future frames, epoch changes, and layout changes are errors which do not
advance extractor state. A stream restart requires an explicit `reset()`.

## Camera-to-head gaze geometry

`CameraToHeadGazeExtrinsics` parses one finite camera/head origin and one
non-degenerate camera-to-neutral-head quaternion, normalizes the rotation once,
and retains its matrix for allocation-free projection. The head-origin vector
is limited to `1 m`: this is deliberately more than three times the current
roughly `0.32 m` assembly offset while rejecting centimetre-as-metre and
millimetre-as-metre mistakes. The OAK camera convention is `+x` image-right,
`+y` image-down, `+z` forward. Output is explicitly `yaw_right_rad` and
`pitch_down_rad` in SI radians. Projection subtracts the camera-frame head
origin before applying the camera-to-neutral-head rotation. Fused dot products
and stable `atan2`/`hypot` angle geometry are used near the forward plane; a
plane classification or strict forward-hemisphere angle that cannot be
represented is a typed error, never a rounded valid gaze.

Target points are finite metres with each `x`/`y` coordinate in `[-10, 10] m`
and camera-forward depth in `[0.1, 10] m`. These are conservative local
head-gaze input-policy limits, not claims about OAK ranging accuracy. A camera
ray is normalized on parsing but cannot be projected by itself: the head and
camera do not share an origin, so a bounded positive camera-forward depth is
required to resolve the target point. This prevents an unknown range from being
silently treated as infinite range and rejects common unit typos without
clamping.

The Nano schema-v1 RGB policy stores this boundary as an optional, independently
versioned nested `gaze_geometry`. Absence means projection is unavailable; it
never assumes identity rotation or zero translation. Geometry output is not a
head intention, servo pose, calibration certificate, or permission to move
hardware.

## Expression to KEP2 mapping

The adapter maps core basis points (`10,000 = 1`) to KEP2 normalized units
(`1,000 = 1`) with nearest rounding; signed half cases round away from zero.

| Expression-core meaning | KEP2 field | Mapping |
| --- | --- | --- |
| `gaze_x_right` | `gaze_x` | positive right, same sign |
| `gaze_y_down` | `gaze_y` | negate: core positive-down to renderer positive-up |
| `openness` | `lid` | `1000 - openness`; KEP2/source renderer uses closure |
| `pupil_dilation` | `pupil` | direct normalized scale |
| render brightness | `brightness` | direct normalized scale |

The physical panel mounting sign is firmware calibration, not a value this
host layer can infer. Exact firmware-build matching in the session prevents a
calibrated polarity from silently crossing builds.

## Autonomic character layer

`AutonomicCharacterEngine` decorates each fresh, already typed reaction without
extending or replacing its host freshness. It is a fixed-size state machine
with no allocation, I/O, wall clock, or background task. One stream epoch seeds
its local xorshift sequence, so the same seed, monotonic samples, typed
character inputs, and base intentions produce identical output. The character
input does not accept a loose face-present boolean: an established face
carries track identity, normalized image bearing, non-metric apparent width,
freshness, and observed/switched/coasting provenance. Acquisition and expired
tracks are absent by construction.

The retained modes are idle, greeting, tracking, lost, searching, and sleepy.
The finite act vocabulary preserves the prior Nano behaviors and the retained
pet responses. Cooldowns, bounded durations, timed blinks, and micro-saccades
are state, not sleeps. Live compliant phases also cross a typed feedback edge:
candidate, yielding, release dwell, resting, and controller-progress recovery
drive one minimum-jerk eye envelope before the completed episode selects its
social act. This lets the eyes soften, shrink, and look down while touch is
still happening; expressive head scripts are suppressed while the compliant
controller owns motion. The Nano owner records each complete controller
episode through a bounded Fable-compatible NDJSON journal before admitting the
social response; disk append/sync runs on a separate writer thread and any
writer failure is a terminal accessory fault. All output is reconstructed
through KEP2's signed/unit domain constructors after explicit saturation to
`[-1000,1000]` or `[0,1000]`; the frame path retains no untyped eye values.

`render_character` returns one `PreparedCharacterFrame`: the KEP2 eye intent,
mode, optional named act, and a semantic overlay for bow, curl, yaw, and roll.
The eyes begin the reaction first; the four head axes remain exactly natural
for 120 ms, follow a quintic minimum-jerk pulse, and return exactly to natural.
Every retained act exercises every joint, but small secondary-axis motion is
deliberate so a nod still reads as a nod rather than four unrelated gestures.
The named values use a dimensionless `character-positive` scale of
`[-1000,1000]`, not encoder ticks. This coherent frame is also the stable seam
for Rerun or other diagnostics: telemetry can record mode, act, eye values,
and four normalized axes without parsing transport bytes or claiming physical
motion.

Physical conversion remains separate and fail-closed. An optional declaration
in the reviewed head-gaze policy binds the positive full-scale offset for each
named joint. That declaration is parsed once against the same natural pose and
hard envelopes as gaze. Composition uses signed nearest-tick rounding and
rejects an out-of-envelope result; it never clamps. Without the exact mapping,
the overlay is explicitly withheld. With physical head gaze enabled, the
single existing head actor evaluates face gaze and character overlay together,
requires the sole base owner's exact-zero lease, and owns the only bus write.
This avoids a second servo owner and avoids treating signs copied from the
older Python process as calibration evidence.

Semantic expressions project exhaustively:

| Core | KEP2 |
| --- | --- |
| `Neutral`, `Calm` | `Neutral` |
| `Attentive`, `Curious` | `Curious` |
| `Friendly` | `Greet` |
| `Concerned` | `Concerned` |

The caller supplies the active semantic label because a mixed
`ReactionOutput` intentionally has no single dominant `ExpressionKind` field.
Neutral fallback is always forced to KEP2 `Neutral`. A core reaction containing
a head offset is still rejected: this crate owns no head actor. The production
default remains `ReactionMixer::default()` → `HeadIntention::NaturalHold`;
the separate typed character overlay cannot grant actuation authority.

## Eye ownership session

`EyeSession` yields and consumes KEP2 `Message` values; a transport owns byte
framing, deadlines, reconnects, and writes. The required sequence is:

1. send a nonzero-nonce-bound identity query;
2. require an exact report nonce, device UID, firmware build ID, capability
   set, boot ID, and non-regressing device uptime;
3. explicitly acquire the reported boot with a distinct nonce and selected
   nonzero control epoch;
4. send one intent at a time, bound to exact boot/epoch/sequence/lease and its
   host reaction freshness; and
5. require a matching `AppliedNew` report with exact lease duration before
   advancing the sequence. An acknowledgement received at or after the full
   lease duration on the host clock is expired. `DuplicateCached` is a fault
   because this state machine does not retransmit intent messages.

Duplicate, stale, rebooted, out-of-order, wrong-session, rejected, regressing
clock/renderer, or transport cases enter `Fallback`. When the current
boot/epoch is still trustworthy, the fault includes a best-effort KEP2 release
for the transport to send. A reboot mismatch deliberately produces no stale
release addressed to the new boot. Normal release also requires a bound
`Released` acknowledgement with a zero device interval. KEP2 parses result
code and timing together into a private-field `IntentResult`; reversed,
unbounded, or result-inconsistent intervals are rejected before they can reach
this session and are never subtracted with wrapping arithmetic.

Intent sequences never wrap: the final value is reserved for release instead
of silently returning to zero. Device-clock milliseconds, host-clock
nanoseconds, intent sequences, and the firmware's independent 32-bit rendered
frame counter remain distinct types. The renderer counter is compared with
half-range serial arithmetic, so a real `u32::MAX → 0` wrap is accepted while
ambiguous or backward reports fail.

`FirmwareAdmission` proves only that matching firmware admitted the state and
reported a renderer sequence. It is not optical evidence that an LED lit.
Likewise, returning a release message does not prove autonomous fallback was
physically rendered.

## Source-drop provenance and limits

This boundary was derived after a read-only audit on 2026-07-20 of the
unversioned local source drop `Downloads/Kiko_expression_engine`. It was not a
Git worktree, so file hashes are recorded instead of inventing a commit:

- `README.md`: `56ebf6931a86b223a5323c3254b900fa9ee3ff95411956134a03516803462a74`
- `eyes/src/host.rs`: `14b7afd7b3d5972177b85a7a2f22cd2bdca8b123d170d9f5543c66c95bbc7853`
- `eyes/src/animation/host_eye.rs`: `65bcf56c9e779ca9ed8244756d969ab10e2d2b6ce15067c2454014a07a4c7901`
- `eyes/src/config.rs`: `39dc3a3121599607ec0a7db0d3e6a8ce1bc18ddc6a3ef9b93596088b09befe7a`
- `eyes/src/geometry.rs`: `38fbbf822516f2172976346a29badae3053d4d6c7124c99e1d0bc8720b5a706d`

Those files establish the legacy closure-lid convention, autonomous fallback,
conjugate-gaze renderer, and two explicitly unverified physical sign choices.
The ASCII legacy transport, port probing, RP2350 renderer, prebuilt UF2, neck
control, and hardware calibration were not copied into this crate. KEP2 is the
canonical typed wire contract.

Before robot use, the Nano integration still must provide a bounded transport,
select the exact USB identity/port, configure the expected UID/build/capability
allow-list, and perform hardware-in-the-loop checks for both global gaze sign,
right-panel mounting sign, lid direction, brightness/power ceiling, watchdog
fallback, and visible application. No host test can establish those physical
facts.
