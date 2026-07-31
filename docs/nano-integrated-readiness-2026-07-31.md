# Nano integrated-readiness checkpoint — 2026-07-31

This checkpoint records what is implemented, what was observed on the Jetson
Orin Nano, and exactly why Kiko is not yet at the wheel-attachment gate. It is
not a deployment, physical calibration, camera-accuracy, SLAM-accuracy,
occupancy-quality, MPC-tracking, thermal, or motion-safety claim.

## Frozen source

The integrated source revision is
`6cc51ec5557e91bcdd0f047ac1e4d79a0bbf008b` on
`codex/nano-expression-integration-stage`. The host and Nano checkouts were
clean and exact at this revision before target-native verification. The branch
was pushed to the `psygos` remote.

The preceding reviewable commits provide:

- one evidence-bound physical gaze path;
- one unified loopback browser/agent authority and control surface;
- one foreground attended navigation graph joining OAK capture, accessories,
  sparse/dense SLAM, local/global 2D occupancy, map persistence, Rerun,
  manual control, frontier exploration, map-click goals, MPC, and exactly one
  STM32 owner; and
- one bounded autonomic eye director integrated into the production RGB
  bridge.

## Expression-engine integration

The retained Fable behavior is no longer treated as a second architecture or
an independent production daemon. `AutonomicCharacterEngine` is a fixed-size
state machine inside the canonical expression runtime. It retains all six
modes—idle, greeting, tracking, lost, searching, and sleepy—and all 19 named
acts:

```text
curious_tilt  double_take  excited_wiggle  lean_in
nod  soft_nod  happy_squint  puppy_eyes  shy_dip
sparkle  blink_flourish  look_around  perk_up  daydream
stretch  sweep_scan  head_bob  sneeze  dance
```

It adds deterministic, bounded act selection, cooldowns, timed blinks,
micro-saccades, palette changes, and loss/search/sleep transitions to each
fresh typed RGB reaction. It performs no allocation, file or device I/O,
sleeping, wall-clock access, locking, or background-thread creation. Every
numeric eye field is explicitly saturated and reconstructed through the KEP2
domain constructors. Replacing the eye rendering intent cannot alter the
source timestamp or extend the reaction deadline.

The eye firmware interprets blink as an edge on a newly sequenced command.
The greeting therefore emits one blink request rather than streaming a high
level that would continuously restart the device timer.

These acts animate the eyes only. The words `tilt`, `nod`, and `head` retain
the character intent of the earlier engine; they do not authorize copied
servo offsets. Physical head movement remains owned only by the separately
calibrated face-gaze path. This preserves natural face following without
silently promoting older Python servo constants into canonical calibration.

## Host verification

All host OAK checks below used `OAK_SYS_CHECK_ONLY=1`; they validate the exact
compile boundary but do not open or emulate an OAK device.

- `kiko-expression-runtime`: 77 unit tests and one compile-fail documentation
  test passed.
- Exact `nano-attended-navigation-trial` graph: 1,459 library tests and 78
  main-binary tests passed.
- The same all-target feature graph passed strict Clippy with `-D warnings`.
- The expression runtime passed strict all-target Clippy with `-D warnings`.
- Operator-console view-model tests and all seven browser drive-safety tests
  passed; both JavaScript assets passed syntax checks.
- `cargo fmt --all -- --check` and `git diff --check` passed.

The first restricted host test run produced exactly 23 local socket/listener
`Operation not permitted` failures. The unchanged test command was rerun with
local socket permission and all 1,537 attended tests passed. The restriction
failure is retained here so it is not mislabeled as a code regression or
silently omitted.

## Orin Nano target-native evidence

The exact source revision compiled on the Jetson Orin Nano as an optimized
aarch64 Linux PIE with the real native DepthAI and OpenCV bridge; the
compile-only OAK override was not set.

```text
binary:
  /home/makerspace/kiko/target/nano-attended-navigation-trial/release/kiko-slam
SHA-256:
  7973681299e151710d7232b819c32329d2f43791276ae0086e4d32cc175e9ae6
ELF build ID:
  bcfbee68b0e38d417a6e4a8d2de9882f3ee3e4cd
```

The attended-navigation help boundary ran successfully. `ldd` reported no
unresolved libraries when given the retained ONNX Runtime, inference,
DepthAI, DepthAI vcpkg, and aarch64 system library roots. This is build and
loadability evidence, not a production installation: those runtime libraries
still need to be placed and bound by the immutable deployment rather than
inherited from developer/evidence directories.

The companion base-commissioning and plant-promotion binary identities are
from the same exact source revision:

```text
kiko-nano-base-commission:
  SHA-256 239139739b4d60247ace9445d00ee1236727dc906b188a76c66aa7a075f15843
  ELF build ID 0d63166d162fc772d004b8ff7bdb635b299fb909
kiko-nano-plant-promote:
  SHA-256 1d6ffa6ce047a94667e20495abba58a94f95832e7d9a2a2516a90e862ccbe077
  ELF build ID c2d11a485f9ab57df9b2eba4058f56813e974ba8
```

Both are aarch64 Linux PIE executables, both had zero unresolved libraries
under the same retained runtime library roots, and both displayed their
expected help boundary. The promotion tool describes itself as offline with
no motion authority. The 77 expression-runtime tests and its compile-fail
documentation test also passed natively in the Nano's optimized test profile.

## Live hardware ownership retained

The existing `/home/makerspace/kiko-follow/engine-guardian.sh` and its one
Python child remained alive throughout source update and native builds. That
child continued to own the OAK, head serial endpoint, and eye serial endpoint.
The STM32 endpoint remained separate and free. No process was killed,
restarted, or displaced, and no motor command was sent during this
checkpoint.

At `2026-07-31T14:35:46+05:30`, guardian PID `12740` and Python PID `13632`
were alive. The Python process held `/dev/bus/usb/002/049`, `/dev/ttyACM1`,
and `/dev/ttyACM2`; `/dev/ttyACM0` had no owner. Its log was still advancing
at `14:35:55` and reported fresh IDLE/GREET/TRACK/LOST/SEARCH transitions,
`sparkle`, `perk_up`, `look_around`, and `blink_flourish` acts, and
`head=TRACKING`. The guardian still has both `@reboot` and minute-level
crontab launch authority, which is an explicit handoff conflict rather than a
canonical startup mechanism.

Consequently, two different facts must not be conflated:

1. the current legacy guardian process and fresh logs report its existing
   expression, face-follow, and active neck-hold behavior; and
2. the newly integrated canonical 19-act engine is compiled and tested but is
   not yet the live hardware owner.

Starting the canonical graph while the guardian owns the same devices would
only recreate the observed `X_LINK_DEVICE_ALREADY_IN_USE` failure. A single,
explicit handoff is required; running both is not integration.

The OAK remained enumerated through a SuperSpeed path while the guardian was
live: the Orin's 10,000 Mbit/s root and hub path reported the OAK vendor node
at 5,000 Mbit/s. This confirms current USB3 enumeration, not full
canonical-stream delivery at this source revision. Earlier bounded full-stream
evidence remains in `docs/nano-live-readiness-2026-07-29.md`.

## STM32 state and blocking fault

The installed STM32 image reports ABI 2, build `131074`, fingerprint
`KIKO-NO-ACT-V1!!`, capability bits `319`, maximum PWM `0%`, outputs disabled,
and no motion authority. Its identity is deliberately motor-inert.

A fresh 20 Hz transport qualification was attempted against this non-actuating
image and rejected before its timed run because the idle heartbeat reported
controller fault bits `0x00000001` (`SERIAL_INTEGRITY`). The 50 Hz stage was
not run. This failure is preserved rather than cleared, retried, or converted
into a readiness claim.

The current image cannot drive or calibrate the wheels. The motion-capable
commissioning image must be flashed only under the documented attended
preconditions with motor power physically disconnected, then must prove exact
identity, applied-zero, disarm, watchdog, restart, disconnect, and fault
behavior before wheel attachment.

## Remaining evidence required before “attach the wheels”

The next gate is finite:

1. Resolve and explain the retained STM32 serial-integrity fault while the
   controller remains non-actuating.
2. Measure and retain the physical values that templates deliberately do not
   invent: wheelbase, native-IMU-to-base rotation, tracking-camera-to-base
   transform, footprint, obstacle-height frames, motion bounds, and plant-fit
   acceptance limits.
3. Render a sentinel-free immutable base-commissioning deployment that binds
   those measurements, the exact controller contract/firmware, models,
   cascades, calibration, runtime libraries, OAK MXID, and binary hashes.
4. With wheels removed, motor power disconnected, head supported, attendance
   confirmed, and the independent power cut reachable, flash and qualify the
   exact motion-capable commissioning firmware.
5. Perform one deliberate guardian-to-canonical handoff. Require live OAK
   SuperSpeed RGB/stereo/depth/IMU freshness, active natural neck hold, visible
   canonical expressions, face-gaze behavior, live SLAM, occupancy, Rerun,
   console Stop/Disarm, and an exact applied-zero STM32 receipt.
6. Exercise timeout, release, browser-loss, controller-disconnect, camera-loss,
   and accessory-fault shutdown paths without nonzero motor power.

Only when all six items have retained evidence is the next truthful request:
attach the wheels for attended polarity, velocity/PWM mapping, plant
identification, map/localization, occupancy, manual control, autonomous
map-click, and MPC acceptance.
