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

## STM32 motor-inert transport qualification

The installed STM32 image reports ABI 2, build `131074`, fingerprint
`KIKO-NO-ACT-V1!!`, capability bits `319`, maximum PWM `0%`, outputs disabled,
and no motion authority. Its identity is deliberately motor-inert.

A first 20 Hz transport qualification against this non-actuating image was
rejected before its timed run because the idle heartbeat reported controller
fault bits `0x00000001` (`SERIAL_INTEGRITY`). That failure remains preserved as
evidence rather than being omitted or converted into a readiness claim.

The source audit narrows the meaning of this observation. Candidate admission
checks an exact Hello and idle-safe Heartbeat before writing its first
freshness challenge, so this failed attempt wrote no diagnostic probe bytes
and did not create the reported fault. Firmware initializes the fault word
only at boot and subsequently ORs fault bits into it; it has no in-session
clear path. `SERIAL_INTEGRITY` can represent RX queue invalidation, a decode or
oversize-record failure, an unsupported host-direction message, or an
unavailable transmit/report path. The latched bit does not retain which one
occurred. Given the prior interactive history, the source cannot identify a
root cause from that Heartbeat alone.

Follow-up commit `e6e6b5c87c73ddd8cd4e58460b08add65fc44f1e`
made that phase boundary explicit in the qualifier's typed error without
changing the firmware or attended runtime. The exact native aarch64
`v2_transport_qualify` binary has SHA-256
`3ca4bde44ca00683023575fa10c5b35c586297d107c4b2430c6228444907f283`
and ELF build ID `7bf83043d8fc1c15bc9ac7dc0126edbcf251e44d`; it had no unresolved
libraries and displayed the expected CLI boundary. Repeating the 20 Hz
invocation produced:

```text
read-only candidate admission rejected controller fault bits 0x00000001
before any diagnostic probe bytes were written
```

This confirms error provenance; it did not by itself clear or qualify the
controller.

With wheels removed, motor power physically disconnected, the head supported,
and an independent stop available, the controller was then reset once through
the exact ST-Link serial `066EFF313946303143221230`. OpenOCD reported target
voltage `3.249012 V` and completed normally. A read-only identity probe after
reset returned the same exact motor-inert contract and no controller fault.
The boot ID also repeated. That is expected for this deliberately motor-inert
image because it has no session-unique boot-ID source; it must not be reported
as proof of a restart.

The clean boot separated the historical sticky fault from a currently
reproducible transport fault. Fresh-nonce timed qualifications then passed:

| Rate | Run ID | Planned/completed/unique | Missing/duplicate/reordered | Host or writer skips | Late writes | Max in flight | Max host RTT | Max controller service | Max RX/TX queue | Max heartbeat gap | Final idle-safe |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 20 Hz | `2852397741463210674` | 200/200/200 | 0/0/0 | 0 | 0 | 1 | 20,886,967 ns | 0 ms | 0/80 B | 257,657,711 ns | yes |
| 50 Hz | `3656527987827278333` | 500/500/500 | 0/0/0 | 0 | 0 | 2 | 25,541,436 ns | 1 ms | 0/91 B | 259,643,269 ns | yes |

Both runs remained motor-inert: maximum PWM was `0%`, actuation stayed
disabled, and the final Heartbeat was idle-safe after the last write. The
heartbeat-gap acceptance bound was `375 ms`; both observed maxima were below
it. These runs prove this transport test under the stated conditions. They do
not prove powered motor behavior, closed-loop control quality, SLAM throughput,
or a general performance improvement.

## Journal provisioning and wheels-off candidate installation

After both motor-inert rates passed, the complete installed 512 KiB flash was
read back before any write. It has SHA-256
`dfda9a32a6dede174ce55a29acfb59fc754277c421d23db886c8155d0f40dd55`;
its 384 KiB motor-inert main prefix has SHA-256
`270e553f5c18a53393f0234f334d3ccc71be32ac7827240b54c939c6d6def38d`.
The sector-7 parser rejected the original 128 KiB suffix as `HeaderErased`, so
journal-capable firmware was not started against an invented or malformed
identity source.

A fresh journal was generated with the repository tool. Its initial sector
image has SHA-256
`13673248185cbb7ea60f945839940bc1468f69a8383f79bfe77b298e6c8a4862`,
seed `3ea1f8bf266d75f2`, zero committed records, and a planned first record at
byte offset 32. The complete motor-inert-main-plus-journal image was written
and the 524,288-byte readback compared byte-for-byte; both have SHA-256
`df58f846c0b1590717d5f40da0ad22b5f0d883d63e56e7eaed2792747c7f2e8d`.
The motor-inert identity remained exact after that operation.

The operator-supervised wheels-off candidate was then built twice in distinct
clean Nano checkouts from source commit
`beb9441435f598cee9654bf6f0731114a27bdfab`, using Rust 1.88.0 and source-root
remapping. Both ELFs were byte-identical with SHA-256
`fcb0624f6422309f8cd4a383e31e6fd3596465a4efbb98317310d7febefa63e0`.
Its final flash-backed byte ends at `0x0800beb4`, below the journal boundary at
`0x08060000`; the padded 384 KiB main image has SHA-256
`8b8fa6f2b7498ac9c7e5ad86d5e44d98a6fe8fb4d50035fb9ec9f6828b564424`.
Strict target Clippy passed and the exact journal inspector passed 6/6 tests.

With motor power still physically disconnected, only sectors 0 through 6
were replaced. The complete candidate main readback matched byte-for-byte and
sector 7 matched both its immediate pre-write and post-write snapshots. Two
controlled run windows then produced exactly the two canonically planned
16-byte journal appends. The next boot ID predicted from the second snapshot
was `4513161801469097461`; the subsequent read-only UART identity reported
that exact boot ID, UID `2c0018001750314242353320`, ABI 2, build `135169`,
fingerprint `KIKO-4PWM-CAND1!`, capabilities `575`, a 30% hard cap, both-low
neutral, unverified physical-stop semantics, and disabled output.

This is now the installed image. It grants a bounded software motion class,
but no command owner or serial writer was started, no motor command was sent,
and motor power remained physically disconnected. The result proves image,
journal, readback, and reported identity boundaries; it does not prove a PWM
output, motor direction, physical stop, velocity, SLAM, or MPC behavior.

## Remaining evidence required before “attach the wheels”

The next gate is finite:

1. Measure and retain the physical values that templates deliberately do not
   invent: wheelbase, native-IMU-to-base rotation, tracking-camera-to-base
   transform, footprint, distinct floor- and axle-frame obstacle slabs, motion
   bounds, and plant-fit acceptance limits.
2. Render a sentinel-free immutable wheels-off qualification deployment that
   binds those measurements, the exact candidate contract, models, cascades,
   calibration, runtime libraries, OAK MXID, and executable hashes.
3. Perform one deliberate guardian-to-canonical handoff. Require live OAK
   SuperSpeed RGB/stereo/depth/IMU freshness, active natural neck hold, visible
   canonical expressions, face-gaze behavior, live SLAM, occupancy, Rerun,
   console Stop/Disarm, and an exact applied-zero/disarm STM32 receipt while
   motor power remains physically disconnected.
4. With wheels still removed and the independent cut reachable, complete the
   attended candidate direction checks and the bounded timeout, release,
   browser-loss, controller-disconnect, camera-loss, and accessory-fault
   shutdown matrix. Retain uncertainty rather than promoting an incomplete
   or ambiguous physical stop.
5. Return motor power to disconnected, reproduce and install the distinct
   attended wheel-on commissioning image while preserving the monotonic
   journal, and require its exact identity and stopped output before asking
   for wheel attachment.

Only when all five items have retained evidence is the next truthful request:
attach the wheels for attended polarity, velocity/PWM mapping, plant
identification, map/localization, occupancy, manual control, autonomous
map-click, and MPC acceptance.
