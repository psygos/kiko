# Kiko Nano integration handoff for Fable — 2026-07-21

This is the authoritative handoff at the point where the operator paused Codex
implementation and switched to Fable. It records what was actually observed,
what was changed, what passed, what failed, and what remains. It is deliberately
conservative: a host test, controller receipt, USB enumeration, or parsed file
is not promoted into a physical-robot claim.

## Read this safety state first

- The robot's wheels were physically removed throughout the documented motor
  tests. Do not infer that wheels remain removed now; ask the operator before
  any new nonzero base command.
- At `2026-07-21T05:55:47+05:30`, Nano PID `28752` was the only Kiko process:
  `kiko-head-commission --return-to-target` using
  `/tmp/kiko-head-return-natural-close-20260721.json`. It was actively holding
  the four neck servos at `[2155, 2545, 2943, 2876]` ticks.
- The process subsequently reached its configured 900,000 ms ceiling and
  exited with `hold_stop reason=MaximumDuration`,
  `hold_exit termination=RequestedShutdown`, and
  `torque_disable_complete all_writes_completed=true`. The output separately
  confirmed completed torque-disable writes for bow, curl, yaw, and roll.
  **The neck is now torque-disabled, not actively held. Physically support it
  before touching or reopening the bus.**
- This is a bounded commissioning lease, not a production head supervisor. It
  performs no periodic telemetry while holding. Starting a second commissioning
  process is not a safe handoff: a new actor deliberately disables all torque
  before observing positions, creating a gravity-drop interval. Physically
  support the head before any new head process, and never extend the hold by
  blindly restarting this tool.
- No Kiko systemd unit was installed or enabled. `systemctl list-unit-files
  'kiko*'` returned zero units.
- The last documented STM32 state is the canonical KRP2 motion-disabled image:
  maximum PWM 0, disabled safe output, no motion authority. Re-probe its typed
  identity before relying on that state.

## Primary goal at handoff

Implementation is paused. This report is the primary handoff deliverable. The
still-open product goal is:

> Deliver and finish the production-grade Kiko Nano integration end to end on
> `codex/nano-expression-integration`: strict manifest and device admission,
> fail-closed supervisor and control API, safe mapping/manual/explore/point
> navigation, RGB-driven expression eyes and natural head hold, truthful map
> persistence/relocalization, Nano packaging and cold-boot/fault verification,
> physical validation only where safely available, focused tests, reviewable
> commits, a pushed branch, and a clean worktree with all remaining
> hardware-dependent claims documented.

The final user-visible example remains:

1. Start the robot; one agent inventories exact devices, establishes applied
   zero, locks the natural head pose, checks OAK streams and admitted artifacts,
   and remains disarmed.
2. Calibrate the encoderless base with bounded, operator-approved excitation,
   using visual forward motion plus calibrated IMU yaw rate to fit PWM-to-motion.
3. Drive manually while online SLAM and local occupancy continue updating.
4. Let the agent explore a declared bounded area and build the map.
5. Display that exact map, submit a revision/epoch-bound clicked point through
   the local typed control API, and navigate there with global planning, local
   depth safety, MPC, and exact STM32 applied receipts.
6. Run deterministic eye expression from the already-owned RGB stream, with a
   natural supervised neck hold and later separately qualified gaze motion.

## Repository topology and exact state

All paths below are real paths, not aliases.

### Development Mac

- Authoritative integration worktree:
  `/Users/ttrb/Documents/Codex/2026-07-10/ss/work/kiko`
- Branch: `codex/nano-expression-integration`
- Committed HEAD: `b89065d3112bbb871ca193c2626addcc040fd849`
  (`docs(nano): record qualified bench evidence`)
- The branch has no configured upstream. A fresh `git ls-remote` attempt failed
  authentication, so do **not** claim that `b89065d` or the dirty work below is
  pushed.
- Default shell checkout, which must not be confused with the worktree above:
  `/Users/ttrb/Documents/Codex/2026-07-10/ss/kiko`
- That default checkout remains on clean `main` at
  `27d63e7b91fd1b1903b37233a13e2efc083f6f3c`.
- Origin URL:
  `makerspace@192.168.50.2:/home/makerspace/kiko`
- `codex/core-hardening` is at `8478899` and tracks
  `origin/codex/core-hardening`. The Nano-expression branch descends from that
  completed host-hardening work; it is not a second product repository.

### Jetson Orin Nano

- SSH endpoint: `makerspace@192.168.50.2`. The password is intentionally not
  written into this repository; the operator supplied it in the task.
- Hostname: `ubuntu`.
- Platform observed now: aarch64, Ubuntu 22.04 lineage, Linux
  `5.15.148-tegra` (`#1 SMP PREEMPT Tue Jan 7 17:14:38 PST 2025`).
- Integration worktree:
  `/home/makerspace/kiko-nano-expression-integration`
- Branch: `codex/nano-expression-integration`
- Committed HEAD: `bb0f78d` (`feat(stm32): add bounded single-wheel breakaway
  probes`). This is the same Git repository as `/home/makerspace/kiko`, linked
  as another worktree; it is not a separate expression project.
- The Nano worktree contains only the uncommitted head-return slice listed
  below. The Mac is ahead by committed evidence `b89065d` and also contains the
  uncommitted navigation and persistence work. Do not overwrite either side.
- Primary Nano checkout: `/home/makerspace/kiko`, clean on
  `codex/jetson-hardware-validation` at `482023e`.

### Dirty Mac integration worktree — preserve exactly

No final commit was made for these files. Do not reset, checkout, clean, or
bulk-copy over this worktree.

Head-return slice:

- modified `crates/kiko-head-protocol/src/lib.rs`
- modified `crates/kiko-head-protocol/src/pose.rs`
- modified `crates/kiko-head-runtime/src/actor.rs`
- modified `crates/kiko-head-runtime/src/bin/kiko-head-commission.rs`
- modified `crates/kiko-head-runtime/src/config.rs`
- modified `crates/kiko-head-runtime/src/lib.rs`
- new `crates/kiko-head-runtime/src/motion.rs`

Map replay/persistence slice:

- modified `crates/kiko-slam/src/dense/occupancy.rs`
- modified `crates/kiko-slam/src/dense/occupancy_persistence.rs`

Manual-control/dispatcher slice:

- modified `crates/kiko-slam/src/navigation/agent_config.rs`
- modified `crates/kiko-slam/src/navigation/control_api.rs`
- modified `crates/kiko-slam/src/navigation/control_socket.rs`
- modified `crates/kiko-slam/src/navigation/coordinator.rs`
- modified `crates/kiko-slam/src/navigation/manual_drive.rs`
- modified `crates/kiko-slam/src/navigation/mod.rs`
- modified `crates/kiko-slam/src/navigation/mpc.rs`
- modified `crates/kiko-slam/src/navigation/safety.rs`
- new `crates/kiko-slam/src/navigation/agent_dispatch.rs`
- new `crates/kiko-slam/src/navigation/agent_manual.rs`

Temporary, untracked operator-session configs:

- `.head-current-hold-20260721.json`
- `.head-return-natural-20260721.json`

Those two temporary files must not be committed. Review and delete them with an
intentional patch only after their evidence has been transferred to docs.

At the freeze point the dirty diff was about 2,689 inserted and 86 deleted
lines across 16 tracked files, plus the three new Rust modules and two temporary
JSON files. `git diff --check` passed.

### Dirty Nano integration worktree

The Nano worktree at `bb0f78d` has the same seven head source changes/new module
listed above. It does not have the Mac-only manual or persistence changes. Its
currently running release binary was built from an earlier synchronized form of
the return logic; the latest source classification patch was not rebuilt for
deployment after the final edit. Never identify a running binary merely from
the source directory.

## Current hardware inventory and port facts

Observed at `2026-07-21T05:55+05:30`:

| Role | Exact admitted identity | Current kernel path / USB |
| --- | --- | --- |
| STM32 ST-Link VCP | ST-Link serial `066EFF313946303143221230` | `/dev/serial/by-id/usb-STMicroelectronics_STM32_STLink_066EFF313946303143221230-if02` -> `/dev/ttyACM0`; `0483:374b` |
| Four-servo STS head bus | adapter serial `5B14031114` | `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14031114-if00` -> `/dev/ttyACM1`; `1a86:55d3` |
| Eye RP2350 | KEP2 UID `98c47919804f9f1aaacfd5fa0a20bf74` | `/dev/serial/by-id/usb-kiko_kiko-eyes-kep2_98c47919804f9f1aaacfd5fa0a20bf74-if00` -> `/dev/ttyACM2`; `c0de:cafe` |
| OAK | MXID `19443010F1B43A2E00` | Intel Movidius MyriadX `03e7:2485` |

The three serial endpoints are persistent identities. Do not fall back to
`ttyACM0/1/2`, because enumeration order has changed across boots.

The Nano exposes a 10,000 Mbit/s USB root/hub on Bus 002, but at this snapshot
the OAK enumerated under the Bus 001 USB 2 hub at 480 Mbit/s. This is direct
evidence that the current OAK connection is not SuperSpeed. It does not prove a
bad camera or cable. The operator explicitly states that both are known-good.

## Physical and device findings

### OAK camera

What worked:

- DepthAI Python 3.4.0 discovered exactly MXID `19443010F1B43A2E00`.
- Low-rate RGB worked: a 320x200, 5 fps request delivered 20 valid `uint8`
  frames with sequence `0..19` and increasing device timestamps in 3.905 s.
- The corrected exact-owner diagnostic constructed one device for the exact
  MXID, capped transport at `HIGH`, reused that owner for calibration and the
  pipeline, and checked negotiated speed. In 15 s it captured 447 left frames,
  447 right frames, 447 pairs, zero orphans, zero recorded drops, and 5,908 IMU
  samples.
- Camera calibration SHA-256:
  `15c7d334857d6e5291e02219484b8e8d44d8da0b3525adce902f415038f6ccd9`.
- Manifest SHA-256:
  `9d4e48c548a18217219e5524182c8f771541ae0ed3b01c6b08598745db14d83f`.
- Commit `1824110` added exact-device USB transport admission.

What failed or remains open:

- Earlier diagnostics reopened a default device after calibration and showed
  XLink disconnect/reconnect behavior. Correcting owner identity removed that
  flaw in one `HIGH` run, but one success does not prove causation or long-term
  stability.
- Current enumeration is USB 2 / 480 Mbit/s, not required production `SUPER`.
- The Nano has Python DepthAI, but no independently proven matching native
  `depthai/depthai.hpp` plus `libdepthai-core.so` installation for the Rust C++
  bridge. Existing old SLAM binaries also reported missing
  `libonnxruntime.so.1` under `ldd`.
- The production RGB + rectified stereo + aligned metric depth + IMU graph has
  not been built and cold-booted from this branch. Full online SLAM on the Nano
  is therefore not yet qualified.

### Eye controller and expression engine

What happened:

- The original controller answered legacy ASCII `P` with `kiko-eyes 1`, but
  repeated far-left/blink/far-right/neutral writes produced no visible change.
- The operator put only the eye RP2350 into BOOTSEL/UF2 mode. An exact legacy
  flash backup was unavailable because the volume did not expose `CURRENT.UF2`.
- The first KEP2 build enumerated USB descriptors but failed configuration with
  `-110`: a 32-character UID needs a 66-byte UTF-16LE serial descriptor, while
  the Embassy control buffer was only 64 bytes. The fix uses 128 bytes plus a
  compile-time size proof.
- Corrected firmware build ID:
  `08134c20df747e68d38bea8af1eb8e62e86b085d347d8e18d5bf18301f368076`.
- Corrected ELF SHA-256:
  `5d0d8b962b33f5c2154fc7f7fe8c9f3d60f942a47296a78570553a9808f1ddd9`.
- Corrected UF2 SHA-256:
  `13dee99d699a0840e53874fee64b56da3fcc33054ea92a1e97da12124e5d4c94`.
- The exact native identity probe verified UID, build ID, nonzero boot ID
  `420323034556454353`, protocol 2, and all required capability bits 255.
- The operator saw a centered eye with small autonomous movements after flash.
- A deliberately obvious fresh KEP2 session then commanded white center, red
  full-left twice, blue full-right twice, three white blinks, and cyan neutral.
  The controller admitted sequences 6..14 and confirmed release; the operator
  explicitly confirmed seeing that exact sequence.

What this proves and does not prove:

- It proves exact-identity KEP2 control, color, conjugate left/right gaze,
  blink, and neutral return for that bounded run.
- It does not calibrate gaze angle in radians, prove every pixel, prove
  long-duration reliability, or provide RGB person/face tracking.
- Committed host code includes deterministic expression intent, borrowed-RGB
  adaptation, exclusive KEP2 ownership, and expiry/fallback behavior. It still
  needs the one production OAK owner and production Nano runtime.

### Neck/head

Assembly geometry supplied and corrected by the operator:

- head center is **0.25 m above** and **0.20 m behind** the OAK;
- neutral head axes are parallel to the OAK optical axes;
- in the chosen OAK convention this is stored as translation
  `[0.0, -0.25, -0.20]` m;
- this is declared assembly geometry, not a measured extrinsic calibration.

The bus is the exact adapter above at 1,000,000 baud, DTR low, RTS high. Servo
IDs and logical order are bow 1, curl 2, yaw 3, roll 4.

The new typed return implementation adds exact target types, bounded config,
explicit physical consent, redundant ordered telemetry, fixed-size waypoints,
direction/path/travel gates, progress and total timeouts, stopped final samples,
and typed evidence. Current intended limits were speed 50 ticks/s and torque
permille `[600,400,400,400]`.

Physical sequence:

1. A first return began at `[2337,2938,2748,2748]` with 10-tick waypoints. Curl
   made no accepted progress for 2 s. Actor cleanup torque-disabled all joints;
   gravity moved the unsupported head and the operator reported that it fell.
2. The implementation changed the waypoint increment to 50 ticks while keeping
   the same speed/torque and added a one-time active present-pose fallback for
   recoverable kinematic faults. The latest source narrows that fallback to
   motion timeout, path-corridor, direction-regression, no-progress, or final
   sample disagreement; clock, telemetry-order, and device-status faults still
   terminate and clean up.
3. A second return began at `[2211,2576,2858,2906]`, reached
   `[2155,2545,2943,2876]` in 18 waypoint cycles, and entered active hold.
4. Final redundant readback was approximately:
   bow 2160/2160, curl 2548/2548, yaw 2941/2941, roll 2874/2874. All device
   status bytes were zero; the yaw and roll raw load readings were 1048 and must
   not be casually interpreted without the servo's signed-load representation.

Software verification on aarch64 after the source changes:

- `kiko-head-protocol`: 16 tests passed;
- `kiko-head-runtime`: 35 tests passed;
- `kiko-head-commission` CLI tests: 7 passed;
- strict Clippy for head protocol/runtime all targets passed after boxing the
  large error evidence.

Unfinished safety boundary:

- The commissioning hold does not periodically read pose, device status,
  temperature, current, load, voltage, movement, or bus liveness.
- A one-time present-pose write is not supervised hold.
- Normal exit attempts all-joint torque disable; abort, power loss, or a severed
  bus cannot guarantee it.
- Production requires one continuously serial-owning actor, fixed-cadence full
  telemetry, freshness/clock/pose/electrical/thermal limits, typed fault and
  cleanup evidence, and service signal ownership. Even that only proves
  host-supervised hold unless a hardware watchdog or mechanical support is
  separately verified.

### STM32 and motors

Initial controller state was legacy ASCII telemetry (`ODO,...`, `DBG,...`), not
the completed typed KRP2 contract. Before any flash, two complete 512 KiB flash
reads matched byte-for-byte:

- legacy flash SHA-256:
  `8e8f658e5ee65b2eca3ca8de7cb045ea2b08dbf3ec82d70b654fe6fa02bec7dc`;
- option-byte SHA-256:
  `d292558017cf9ca0a2e40e262a5c1daa4b305ccf084ce06128133d282f905115`;
- the timestamped `stm32f446re-20260721T0349IST` backup was copied to both Nano
  and Mac. Locate and hash it before depending on it.

Canonical KRP2 provisioning:

- STM32F446, UID `2c0018001750314242353320`, firmware ABI 2, build ID
  `0x00020001`, profile `KIKO-NO-ACT-V1!!`, capabilities 63, PWM 20 kHz,
  maximum PWM 0, disabled safe output, no motion authority.
- Canonical raw image: 39,216 bytes, SHA-256
  `11a4b23b66302d306c7530566d36f122eaba4ca3abe49fd9f5b8622f78c8b2b3`;
  flash readback matched exactly.

Wheels-off commissioning used separate feature-gated one-shot images; it did
not weaken canonical firmware:

- An 8% left/right recipe and a 10 s both-forward 8% recipe produced valid
  nonce/CRC/timer/terminal-safe evidence, but the operator saw no shaft motion.
- KMC3 then admitted exactly one wheel at 30% forward for 500 ms after a 500 ms
  zero dwell. Its raw image was 20,276 bytes, SHA-256
  `e100add1b2c6fcc5483a64f80e9c353775c8d21cefdc8d1c59b72568ddb3ecef`,
  and matched readback.
- The operator saw the right shaft move and the left shaft move on their
  separate KMC3 pulses. A repeated right pulse also moved with historical Nano
  enable candidates held low. This establishes finite motion for each output
  under those conditions; it does not prove direction, velocity, minimum duty,
  driver identity, or a safe operational envelope.
- The contrast between 8% no-motion and 30% motion supports, but does not prove,
  a breakaway threshold explanation.
- After one-shot tests the canonical motion-disabled image was restored and
  exactly read back. Re-establish fresh identity/applied-zero evidence before
  any next action.

No PWM-to-linear-velocity fit exists. No wheel sign, wheelbase, stop distance,
MPC tracking, or wheels-on driving is qualified. Because Kiko intentionally
skips encoders, IMU alone is insufficient for drift-free translation or
PWM-to-linear-velocity gain. Commissioning must bind controller PWM to visual
forward velocity and calibrated IMU yaw rate on one monotonic timeline.

## Software completed and committed

The branch contains the complete host-hardening lineage plus these major Nano
foundations:

- strict bounded robot/device manifest and immutable artifact bindings;
- exact readiness identities and exact OAK device provenance;
- supervisor state machine with mutually exclusive authority and fresh applied
  zero handoff requirements;
- completed typed KRP2 host/server/firmware contract and zero-only session;
- strict local agent protocol and bounded Unix-socket transport;
- deterministic 2D occupancy mapping, revision/map identity, Rerun output,
  checksummed bounded persistence codec and atomic publication;
- point-goal parsing, global planning, local costmap, unknown-is-blocked policy,
  correction-safe odometry, deadline-bounded MPC, shadow command sessions, and
  exact applied-evidence gates;
- frontier/manual/point-goal supervisor concepts and architecture contract;
- expression core, RGB adapter, KEP2 protocol/actor/firmware/commission tools;
- typed STS protocol, present-pose natural-hold commissioning owner, and the
  camera-to-head geometry boundary;
- a wheels-off bench runtime and systemd template that are deliberately
  fail-closed and zero-only.

Important committed documents:

- `docs/nano-agent-architecture.md`
- `docs/nano-bench-evidence-2026-07-21.md`
- `docs/nano-validation-evidence-2026-07-20.md`
- `docs/nano-wheels-off-bench.md`
- `docs/expression-integration-provenance.md`
- `docs/actuation-safety-contract.md`
- `docs/core-hardening-evidence-2026-07-17.md`

The `configs/nano-wheels-off-example` directory is intentionally non-launchable:
it contains `REPLACE`, `DEAD`, placeholder artifacts, and descending head bounds.
Do not broaden those sentinels into permissive values. Construct a qualified
deployment directory from observed identities and content hashes instead.

## Uncommitted software handoff

### Head return

The head slice described above compiled and passed its focused tests/Clippy on
the Nano. It still needs review of docs, removal of temporary JSON, a final
release rebuild, a reviewable commit, and integration into a continuous owner.
Do not confuse the successful bounded return with production hold.

### Manual control and dispatcher

The dirty navigation slice adds:

- typed manual policy/config and plant-envelope binding;
- explicit `begin_manual` rather than implicit authority acquisition;
- coordinator manual-stop evidence and safety-stop reasons;
- bounded control-socket task and dispatcher;
- manual authority/deadman core;
- routing for velocity, manual stop, and global stop;
- lifecycle/supervisor agreement and clock-fault destruction of command state.

Verified before the last type-only patch:

- `cargo check -p kiko-slam --features agent-runtime,actuation` passed;
- manual-core tests 11/11 passed;
- dispatcher tests 2/2 passed.

The final `manual_drive.rs` patch introduced finite parsed-command types but was
not recompiled. Duplicate finite validation remains and the parsed type is not
fully wired into ingress. No live owner connects dispatcher, coordinator,
`PhysicalActuationSession`, and response receipts. Global stop is routed, but a
future owner must release authority, apply zero, and respond truthfully. Typed
live-actuation fault classification and receipt timeout handling remain. Do not
claim end-to-end manual driving from this slice.

### Map persistence and replay binding

The dirty map slice adds:

- internal-only rebinding of a loaded occupancy grid to a replayed map;
- `PersistedOccupancyMap`, `ReplayOccupancyEvidence`, and
  `ReplayMatchedOccupancyMap` boundaries;
- typed load/decode helpers;
- exact comparison of retained sparse-map identity, dimensions, resolution,
  lower bounds, all transform/height `f64` bit patterns, revision, cell count,
  and every class byte;
- no-copy rebinding after an exact match.

Verification completed:

- focused persistence tests 18/18 passed;
- full `cargo test -p kiko-slam --lib`: 983/983 passed;
- strict library/test Clippy passed with only `-A dead-code` for an unrelated
  in-progress control-socket helper;
- `git diff --check` passed.

Production save/replay assembly is not wired. A warm start must replay the
named SLAM dataset through the tracker, retain that live sparse tracker/map
identity, quiesce final occupancy, compare it to the persisted artifact, and
only then bind it. Matching replay data is **not** live-camera relocalization;
fresh localization remains mandatory before motion. Occupancy bytes alone
never establish robot pose.

## Architecture gap: why there is no production robot agent yet

Two partial owners currently exist:

- `kiko-slam live` owns online SLAM/occupancy/navigation/Rerun and optional
  actuation, but it does not own the strict Nano policy/inventory/supervisor,
  KEP2 eyes, head, production control socket, or a shared RGB expression path.
  Its current OAK graph does not expose the required single-owner RGB fan-out.
- `kiko-nano-wheels-off-bench` assembles manifest/OAK RGB/head/eye/supervisor
  concepts, but it is intentionally zero-only and does not run SLAM or
  navigation.

There is no `kiko-nano-agent` production binary, no installed production unit,
and no cold-boot/fault acceptance harness covering the complete graph. Creating
one owner is the next integration milestone; adding more independent demo
processes would worsen transport ownership and safety.

## Mapping/navigation truth model

- The occupancy map is deterministic geometric mapping from metric aligned
  depth and localized pose. It is not an “occupancy network” and does not need
  to be learned for the current SLAM architecture.
- Fresh local depth can stop or replan around a moving human as a dynamic
  obstacle. It does not classify a human or predict their trajectory. Unknown,
  occluded, stale, and out-of-range space remains blocked.
- The global map provides a path; the local costmap and depth safety gate react
  to current obstacles; MPC tracks the safe reference through the calibrated
  plant; the STM32 must return exact applied evidence before the next command.
- Rerun is diagnostic/output-only in the pinned SDK. A map click must be adapted
  into the typed local API as `(map epoch, revision, x_m, y_m)`. Do not claim a
  native Rerun callback that the SDK does not provide.

## Recommended continuation plan

### P0 — establish safe, reproducible state

1. Read this file and the evidence docs. Run `git status` in both Mac and Nano
   worktrees. Do not mutate until their differences are understood.
2. Check the head process. If it has ended, assume torque is disabled and
   physically support the neck before opening the bus. If it is somehow still
   running, do not start a competing owner.
3. Re-probe the STM32 identity without motion. Keep canonical maximum-PWM-zero
   firmware until a separately authorized physical test.
4. Confirm wheels-on/off, clear space, emergency power cut, and head support
   directly with the operator before hardware motion.

### P1 — preserve and close the three dirty slices

1. Review and re-run the final head tests/Clippy; update head README and the
   evidence doc with both the fall and successful return; remove temp configs;
   commit head only.
2. Compile/test the final manual patch; remove duplicate validation; add typed
   actuation fault/receipt timeout behavior; commit navigation only.
3. Review persistence API/tests and commit map replay binding separately.
4. Never mix these into one opaque commit.

### P2 — build one `kiko-nano-agent`

1. Parse one strict production policy and immutable artifacts once.
2. Inventory exact OAK MXID, STM32 UID/build/profile, eye UID/build/boot, head
   adapter and four servo IDs before control ownership.
3. Start unarmed and establish fresh STM32 applied zero.
4. Keep one OAK device owner and fan bounded observations to RGB expression,
   stereo/depth/IMU SLAM, occupancy, navigation, and Rerun.
5. Keep one serial owner each for STM32, eyes, and head.
6. Wire control socket -> dispatcher -> supervisor -> coordinator -> MPC ->
   physical session -> exact response receipt.
7. Make manual, explore, point goal, commissioning, and disarmed mutually
   exclusive, with zero on expiry, disconnect, fault, or transition.
8. Integrate periodic supervised natural hold. Keep expressive servo motion
   disabled until angle/sign/envelope/backlash/stop qualification exists.

### P3 — native Nano packaging and fault acceptance

1. Install or build a pinned native DepthAI header/library pair and matching
   ONNX Runtime on aarch64; record hashes/provenance. Resolve OAK SuperSpeed
   topology, then qualify exact RGB/stereo/depth/IMU graph.
2. Install one least-privilege systemd service with exact supplementary groups,
   deployment paths, state directory, restart behavior, and fail-safe cleanup.
3. Build a cold-boot harness proving wrong/missing/rebooted identity rejection,
   unarmed applied zero, head admission, RGB expiry, SLAM/occupancy readiness,
   authority exclusion, point-goal path/MPC handoff, replay-bound restore, and
   zero on camera/depth/localization/controller/serial/process/clock faults.
4. Run this wheels-off first and record exact logs/artifact hashes.

### P4 — operator-gated physical closure

1. With wheels still off, verify wheel sign and repeat bounded left/right
   control through the production KRP2 owner—not commissioning firmware.
2. Ask the operator to install wheels only after every non-motion gate passes.
3. In a clear bounded area with an independent stop, collect synchronized
   visual/IMU/applied-PWM calibration data, fit and holdout-test the plant, and
   admit the exact artifact only if conditioning/residual gates pass.
4. Qualify low-speed stop distance, manual deadman, local obstacle stop, MPC
   tracking, continued online mapping, frontier exploration, map save/replay
   relocalization, and revision-bound click-to-point navigation.
5. Report any missing physical evidence as unknown. Do not tune by anecdote or
   claim performance without a reproducible measurement.

## First commands for Fable

Read-only Mac snapshot:

```sh
cd /Users/ttrb/Documents/Codex/2026-07-10/ss/work/kiko
git status --short --branch
git log --oneline --decorate -12
git diff --stat
git diff --check
```

Read-only Nano snapshot:

```sh
ssh makerspace@192.168.50.2
cd /home/makerspace/kiko-nano-expression-integration
git status --short --branch
git log --oneline --decorate -12
ps -eo pid,etime,stat,args | grep -E '[k]iko|[r]obot-server'
ls -l /dev/serial/by-id
lsusb
lsusb -t
systemctl list-unit-files 'kiko*' --no-pager
```

Do not begin with `git pull`, `git reset`, `git clean`, a firmware flash, a
servo process, a GPIO write, or a motor command. Reconcile the two dirty
worktrees first. The development origin is an SSH working repository, not a
normal hosted bare remote, and the Nano integration worktree is checked out
from it.

## Evidence boundary at handoff

Proven now:

- exact current USB identities;
- exact KEP2 identity and one operator-confirmed expression recipe;
- one exact-owner OAK `HIGH` stereo/IMU capture and low-rate RGB capture;
- typed head return to the reviewed target and bounded active hold for the
  observed interval;
- exact canonical KRP2 identity and finite wheels-off shaft motion for each
  selected motor output at 30% in commissioning mode;
- extensive host/aarch64 typed-boundary and simulation tests described above.

Not proven now:

- USB `SUPER` production OAK stability or native production SLAM build;
- a continuously supervised or power-loss-safe neck hold;
- RGB face/person detection or head gaze following;
- production-server nonzero actuation, wheel direction, speed calibration,
  physical watchdog behavior, stop distance, or MPC tracking;
- live manual driving, autonomous full-area exploration, persisted-map
  relocalization, or click-to-goal navigation;
- a production Nano agent, installed service, clean cold boot, pushed final
  branch, or clean integration worktree.

That distinction must remain intact until each item has its own reproducible
software and, where applicable, operator-observed physical evidence.
