# Fable expression and Nano integration audit

> Historical source/evidence audit. Replacement-servo calibration, completed
> host merge-back, and the current attended transaction are superseded by
> `docs/unification-plan-2026-08-03.md` and
> `docs/nano-gate-a-live-qualification-2026-08-30.md`.

Date: 2026-07-23
Scope: read-only source and evidence review; no device I/O, process changes,
firmware changes, or physical claims were made by this audit.

This document is the integration boundary for the Fable work. Preserve its
artifact-backed evidence and character direction. Replace its prototype device
owners with the canonical typed Kiko owners. Do not copy prototype authority,
fallback, flashing, or lifecycle shortcuts into the robot runtime.

## Audited inputs

The expression drop at
`/Users/ttrb/Downloads/Kiko_expression_engine` contained 1,281 regular files,
occupied approximately 151 MiB, and had no Git metadata. A sorted whole-tree
content inventory produced SHA-256
`7b0ff876aac7fd2c88727142012990973be8098905c890d724982f269b6d3e71`.
The repository's earlier source-drop classification remains in
`docs/expression-integration-provenance.md`.

The most recent read-only copies of the Nano prototype had these identities:

| Artifact | SHA-256 |
| --- | --- |
| `kiko_face_follow.py` | `ef0c9fb48743bd51ec8af317084273682553ac6b30bed384c74731a0eb3daf4e` |
| `config.json` | `6444ce331d0fe66faf7de9b2696c8d0640881678975831505dd3e7a4e1eebcbc` |
| `engine-guardian.sh` | `1255a9563b1e03ef917b74f220698a1ee80804c3c474f30f1d0e3f3d703b4336` |
| old `kiko_dash.py` | `20392e2e8a292adc9615dc43588843104cee9815b5a2d8c74da875e416ae7c99` |
| wheels-off motor owner | `3a5d2da4de1f57606b7a53b3baf28902b0f4c59d83e65bb2bf0fe2ded80de4fc` |
| wheels-off dashboard | `da57bf7b1afbbd2609d7b186ea5c1b49d68f53d0b943e6204726d1031fb411a4` |

The current face-follow hash differs from some historical report copies. The
hash above is authoritative only for the file inspected during this audit.

At `2026-07-26T20:27:26+05:30`, a fresh read-only Nano check established the
prototype's exact classifier inputs. Python OpenCV `4.13.0` resolved
`cv2.data.haarcascades` to
`/home/makerspace/.local/lib/python3.10/site-packages/cv2/data/`; the two
user-owned mode-`0664` files were:

| Classifier | Bytes | SHA-256 |
| --- | ---: | --- |
| `haarcascade_frontalface_default.xml` | 930,127 | `0f7d4527844eb514d4a4948e822da90fbb16a34a0bbbbc6adc6498747a5aafb0` |
| `haarcascade_profileface.xml` | 828,514 | `b39a4a3be45539db146a7fc1d3e761a292c196eb88421185e6a615b3055e612d` |

The Nano's native C++ OpenCV development package reported `4.5.4` with
`/usr/include/opencv4`. A C++17 probe read the two exact files above into
memory, opened each with `FileStorage(READ | MEMORY)`, selected the first
top-level `cascade` node, and obtained `read=true` with a nonempty classifier
for both. This proves native parser acceptance of those exact current bytes,
not detector-output equivalence, full `oak-sys` linkage, camera behavior, or
runtime latency. Production must copy exact reviewed classifier bytes into the
deployment, bind their sizes and digests, retain those verified bytes, and
pass the retained buffers directly to the in-memory native constructor; the
mutable per-user files are not production assets.

A second read-only source inspection on `2026-07-26` retained the prototype's
exact search policy instead of inferring it from the classifier names:

- convert BGR to grayscale and equalize the histogram once;
- search frontal faces first with `scaleFactor=1.15`,
  `minNeighbors=6`, and `minSize=(30,30)`;
- only when that result is empty, search profile faces with the same scale and
  size but `minNeighbors=4`;
- only when that is also empty, run the same profile search on a horizontal
  mirror and map each rectangle back into the original image grid.

Those are algorithm inputs, not a detector-quality or latency claim. The
prototype used Python's `detectMultiScale` rectangle output and selected by
apparent width. OpenCV's optional cascade level weight is an arbitrary finite
ranking value, not a calibrated probability or person-confidence value.

## Evidence retained

Artifact-backed or bounded historical evidence includes:

- STS/SCS framing at 1,000,000 baud; DTR low; RTS high; logical joints bow,
  curl, yaw, and roll at servo IDs 1 through 4.
- The source drop's provisional encoder calibration:
  `[2127,2558,2925,2930]` ticks, signs `[-1,+1,+1,-1]`, and
  `651.8986469044033` ticks/radian. The source artifact is
  `neck/sysid/calib.json`, whose recorded SHA-256 is
  `8ec182d486dace3cc3747ba19448d06c4153a4b23af21df5dcd211d16cf97533`.
  It is historical calibration evidence, not current natural-pose authority.
- Held-out encoder-model RMS residuals of 0.319, 0.383, 0.309, and 0.252
  degrees in `neck/sysid/fit_report.json`. Most fitted parameters in that
  report are explicitly flat or non-identifiable. Friction and other flat
  parameters must not be promoted as measured plant truth.
- The later operator-observed and typed natural return to
  `[2155,2545,2943,2876]` ticks. This is the current reviewed natural target,
  subject to the canonical actor's bounded admission and continuous health
  checks.
- Exact KEP2 UID/build/session/capability evidence and the operator-observed
  high-contrast color, gaze, blink, and neutral-return recipe recorded in the
  Fable handoff.
- Separate, finite, wheels-off commissioning observations that each shaft
  moved during a 30 percent pulse. They do not establish installed-wheel
  direction, velocity, breakaway, stop distance, or a production plant.

The assembly geometry is operator-declared, not measured extrinsic
calibration: the head center is 0.25 m above and 0.20 m behind the OAK, with
neutral axes parallel. In the selected camera convention, positive Y is down
and positive Z is forward, so the configured translation
`[0.0,-0.25,-0.20]` metres is consistent with that declaration.

The following remain heuristics or proposals:

- Haar frontal/profile face detection;
- range inferred from an assumed 0.16 m face width;
- prototype servo gains, signs, offsets, and permissive windows;
- ALIVENESS, DIRECTOR, and MIND CPU, timing, ODE, person-pipeline, residual
  network, and dynamics numbers;
- simulation-only verification reports and estimated mass, range, torque,
  friction, or contact values.

## Preserve, replace, and reject

| Fable element | Preserve | Canonical replacement | Do not port |
| --- | --- | --- | --- |
| Eye renderer | measured layout, mirroring, bounded gaze, pupil, brightness, blink/easing, double buffering | `firmware/eyes-kep2` and the KEP2 actor | legacy ASCII `E/B/P`, copied UF2, frame-rate-dependent smoothing |
| Eye identity/control | exact UID/build/capabilities and optical recipe evidence | manifest-bound, nonce-challenged KEP2 sessions with fresh leases | VID/PID or serial path as identity; swallowed release errors |
| Natural head | active natural hold, reviewed target, ordered redundant reads | `kiko-head-protocol` and one continuously owning `kiko-head-runtime` actor | Python as a service, `read ... or 0`, unacknowledged writes, raw register guesses |
| Head geometry | corrected 0.25 m above and 0.20 m behind declaration | typed `CameraToHeadGazeExtrinsics` | stale 0.18 m / 0.15 m prototype defaults |
| Character | eye-leads-head feel, blink/saccade timing, micro-saccades, color warmth, novelty/cooldowns, act vocabulary | deterministic typed scheduler and declarative bounded clips, initially eyes-only | direct RGB-to-servo ticks or expression bypass of the supervisor |
| Character acts | curious tilt, double take, excited wiggle, lean-in, nods, squint, puppy eyes, shy dip, sparkle, look-around, perk-up, daydream, stretch, sweep, bob, sneeze, dance | replayable eye clips; later separately qualified head clips | wholesale Python state-machine copy |
| Camera | exact MXID, bounded nonblocking queues, calibrated intrinsics; the current prototype deliberately requests `dai.UsbSpeed.HIGH` | one native OAK owner requesting and verifying its launch-bound SuperSpeed contract while fanning RGB/stereo/depth/IMU | infer a cable/port defect from the prototype's requested USB-2 enumeration; second dashboard pipeline, first-device fallback, silent calibration fallback |
| Person attention | product intent and graceful loss behavior | typed frame/freshness/depth/association observations | assumed-face-width ranging as navigation or safety evidence |
| Guardian | start expressions with the robot and preserve tension continuity | one least-privilege systemd lifecycle with readiness and coordinated handoff | `pgrep`, blind respawn, long-duration shell loops |
| Dashboard | arrow/WASD UX, 150 ms browser lease, stop on release/blur/hide/disconnect, receipt visibility | a local adapter to the typed control socket | raw/legacy PWM, public unauthenticated binding, HTTP 202 as applied evidence |
| Motor owner | exclusive owner intent, finite messages, monotonic receipt sequence, disconnect stop | canonical in-process KRP2 V2 STM32 owner and exact applied receipts | standalone production motor service, ASCII `CMD`/`PWM`, swallowed stop errors, inconsistent 10/20 percent claims |
| Layering | one mixer and “director proposes; safety engine disposes” | versioned typed policy and deterministic engine layers | DWA/pure-pursuit bypasses, guessed dynamics, unsupported CPU/latency claims |
| Recovery/flashing | provenance and operator procedure history | reproducible source-controlled build/install procedure | auto-flash, servo-ID assignment, `pkill -9`, or reset actions in the runtime |
| Copied build/media trees | no runtime content | pinned canonical dependencies and selected reviewed assets | `target`, `node_modules`, browser `dist`, copied UF2, or CAD/media as code authority |

## Production integration direction

The target is one `kiko-nano-agent` process, or an equivalently named single
production owner, composed from the existing canonical modules:

1. Parse one strict policy, manifest, inventory observation, and artifact set.
2. Keep the base at exact controller-acknowledged zero and start disarmed.
3. Own one exact-MXID OAK graph for RGB, rectified stereo, metric
   rectified-left depth, and IMU.
4. Fan that graph to live SLAM, occupancy, local collision checking, Rerun,
   and the expression bridge without opening a second camera.
5. Own one natural-hold head actor and perform periodic, complete, read-only
   health checks.
6. Own one KEP2 actor with fresh session material. Expired or failed eye
   sessions enter firmware fallback without inventing optical evidence.
7. Own one supervisor and local control socket. Manual, mapping, exploration,
   point navigation, persistence, and shutdown cannot overlap ambiguously.
8. Keep expressive head displacement disabled until its physical signs,
   angular ratio, bounds, backlash, stop behavior, and raw telemetry policy
   have separate evidence.

The production V3 bridge now adds bounded face attention to scene motion. Its
native detector and Fable-derived association policy carry exact frame
identity, freshness, loss/coast state, and an opaque detector rank; they do
not manufacture a `PersonObservation`, identity, calibrated confidence,
metric range, or navigation obstacle. Porting face-directed eye behavior
therefore still does not mean claiming semantic person tracking. A future
person boundary used by navigation must additionally carry metric depth or
explicit unknown depth, calibrated association evidence, and reviewed loss
behavior.

## Wheel-attachment gate

Do not ask for the wheels until one immutable wheels-off deployment proves:

- exact device/build/boot/artifact admission;
- a single owner for every physical endpoint;
- exact applied base zero through startup and every fault;
- active natural head hold with repeated health checks;
- live RGB expressions and eye fallback;
- continuing stereo, depth, IMU, SLAM pose, occupancy revisions, and Rerun;
- bounded manual/MPC command streaming under simulated and wheels-off fault
  injection;
- stop on command expiry, client disconnect, controller reset, receipt
  timeout, stale depth/localization, camera failure, head fault, and process
  cancellation;
- coordinated shutdown with every cleanup result reported separately.

Static wheels-off operation can qualify this graph, not whole-room mapping,
floor signs, velocity, stop distance, or navigation.

After attachment, one supervised encoderless commissioning session must bind
exact applied PWM receipts to visual base-frame translation and calibrated
base-frame IMU yaw rate on a common timebase. IMU alone cannot identify
drift-free translation or PWM-to-linear-velocity gain. The measured sequence
is: wheel signs and breakaway, PWM-to-velocity response, effective wheelbase,
timing and stop distance, held-out plant fit, manual deadman, MPC path
tracking, manual mapping, frontier exploration, atomic save/replay plus fresh
relocalization, and finally map-epoch/revision-bound click-to-goal.

Only after the complete wheels-off gate is defensible should the operator be
asked:

> Please attach the wheels. I am ready to run supervised encoderless
> calibration, drive Kiko while online SLAM builds the map, and then qualify
> MPC and click-to-goal navigation.
