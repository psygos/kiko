# Kiko Nano post-freeze continuation for Fable — 2026-07-21

This report supplements, but does not rewrite,
`docs/FABLE-NANO-HANDOFF-2026-07-21.md`. The earlier file is the complete
historical handoff at commit `d87e2d3`; its SHA-256 is
`55234548e469955fb7454f78f79b369311e69cbd4b6aca77efd5a33a7205631d`.
This continuation records everything that changed or was learned after that
freeze. Where a live-state statement conflicts with the earlier snapshot, the
newer timestamp here wins only for live state; the historical evidence remains
unchanged.

The operator paused Nano implementation and switched to Fable. The production
integration goal is paused, not complete. No motor, firmware, servo, camera, or
other hardware command was issued while preparing this continuation.

This is a dated handoff, not proof of present state. Until the operator
explicitly resumes and authorizes Nano access, issue no device command and do
not refresh hardware state. On resumption, treat wheel installation, head
support/power/torque/pose/temperature, process ownership, STM32 image/applied
zero, and OAK path/health as unknown until each is safely re-established.

## Safety corrections to the historical handoff

The earlier report is preserved as a historical record, but these newer
corrections supersede unsafe implications in it:

- Four completed torque-disable writes are not an independent readback of the
  servo torque-switch state. Replace historical statements that the neck "is
  now torque-disabled" with: no active head owner was observed at the stated
  snapshot, four disable writes were reported complete, and actual
  power/torque/pose state remained unverified.
- The historical Nano dirty-tree phrase "currently running release binary"
  refers to the release binary that ran during the earlier bounded return; that
  process subsequently exited.
- A wheels-off test can qualify selected motor-channel/shaft response and
  polarity, but it cannot establish vehicle body-forward wheel sign. Historical
  phrases such as "30% forward" mean protocol-commanded positive polarity, not
  measured physical forward travel. Body-forward sign needs separately
  authorized, assembled, restrained, very-low-speed validation.

## Immediate safety state

At `2026-07-21T07:12:18+05:30`:

- no owner was observed by the recorded `/proc` file-descriptor check for
  `/dev/ttyACM0` (STM32), `/dev/ttyACM1` (head), or `/dev/ttyACM2` (eyes);
- the temporary face-follow process described below had exited;
- its final log reported successful torque-disable writes for bow, curl, yaw,
  and roll;
- those writes have no independent torque-switch-state readback or physical
  observation. Current pose, mechanical support, power, torque-switch state,
  and surface temperature are unknown. Treat the head as gravity-prone and
  potentially energized; keep people clear of its drop/pinch envelope;
- issue no enable, position, or bus command. A future re-enable requires a
  diagnosis of the raw temperature-register threshold trip, a qualified exact
  servo encoding and limit, repeated stable telemetry, physical inspection as
  appropriate, and fresh explicit operator approval;
- wheel installation state remains unknown. Obtain fresh operator confirmation
  before any nonzero base command.

The OAK, eye controller, head adapter, and STM32 still enumerated after the
face-follow exit. The OAK re-enumerated as Bus 001 Device 046 at 480 Mbit/s
(`HIGH` / USB 2), not production-required `SUPER` speed. Enumeration is not a
health, stream, calibration, or ownership proof.

## Newly observed temporary face-follow run

An earlier ad-hoc command had left this process running:

```text
python3 /home/makerspace/kiko-follow/kiko_face_follow.py \
  --snapshot --duration-s 3600
```

It was not a Kiko service and was not part of the Git worktree. It was a
temporary Python prototype with these evidence identities:

- script SHA-256:
  `d91b05a6caf5e5b02b06b5bf591219ebf704817b2402be16cb93f542926d87eb`;
- config path: `/home/makerspace/kiko-follow/config.json`; config SHA-256:
  `5e36132a2168899d4b05e4c88dca27f5b1309df452aaecd0dcb0eef01e63803b`;
- final log: `/tmp/follow-v5.log`, 152 lines / 6,103 bytes, SHA-256
  `7924bc84a7ce5eb860590016dc8a24ee541b5e811dc3d019e740dcc1f8bf7012`;
- read-only preserved evidence copy observed at handoff:
  `/home/makerspace/kiko-hardware-backups/face-follow-v5-20260721T0706IST/follow-v5.log`,
  mode 0444, with the same SHA-256.

The process was observed holding:

- OAK USB endpoint `/dev/bus/usb/001/045` during the run;
- head adapter `/dev/ttyACM1`;
- eye controller `/dev/ttyACM2`.

It did not own the STM32. Its startup log reported:

```text
camera_ready usb=HIGH fx=574.7 cx=316.6 cy=202.9
eyes_session boot_id=420323034556454353 capabilities=255
admit joint=bow  pos=2145 temp=31 volt=120
admit joint=curl pos=2551 temp=31 volt=119
admit joint=yaw  pos=2941 temp=36 volt=119
admit joint=roll pos=2875 temp=35 volt=119
engaged_at=[2145, 2551, 2941, 2875]
head_at_natural tracking_enabled
```

The script used OpenCV Haar frontal/profile face cascades, an assumed 0.16 m
physical face width, apparent face size for range, and bounded servo offsets.
Its log showed `TRACK`, `LOST`, `SEARCH`, `IDLE`, and `GREET` eye modes and
reported detections for roughly four minutes. These are program-log claims;
there was no operator confirmation of correct gaze direction, tracking quality,
distance, or temperature for this run.

This prototype must not be treated as calibrated or production-safe:

- its effective camera-to-head geometry remained the script defaults of 0.18 m
  above and 0.15 m behind because the external config did not override them;
- the operator-declared assembly geometry is 0.25 m above and 0.20 m behind, with
  neutral head axes parallel to the OAK optical axes;
- its external config changed servo signs, scale factors, and motion bounds but
  had no admitted artifact/version binding;
- it was unversioned, outside the canonical repository, and not supervised by
  systemd;
- it used heuristic face ranging rather than measured 3D person localization;
- no reproducible physical calibration or holdout evaluation exists.

The run terminated itself after this exact fault sequence:

```text
status t=240.6s head=TRACKING eyes=TRACK person=True prox=0.66
fault StsError('bow overtemp 86')
park_begin
[depthai] ... Couldn't read data from stream ... (X_LINK_ERROR)
torque_disable bow=True curl=True yaw=True roll=True
park_complete clean=True
shutdown_complete
```

The `86` value is a raw bow temperature-register threshold trip against the
script's raw threshold of 58. Do not silently relabel the raw unit, infer a
physical temperature, assume the cause, or clear the fault based on elapsed
time. The process log says that park and all torque-disable writes completed,
but this is not an independent torque-switch-state readback or a physical
observation. The OAK XLink error in the same fault/shutdown sequence is a
second failure, and the camera subsequently re-enumerated from USB device 045
to 046. Neither failure has been diagnosed.

## Git state after the original freeze

### Development Mac

- worktree:
  `/Users/ttrb/Documents/Codex/2026-07-10/ss/work/kiko`;
- branch: `codex/nano-expression-integration`;
- current committed HEAD before this continuation is `e0e8db3`;
- the branch still has no proven upstream/push for these commits;
- the default checkout at
  `/Users/ttrb/Documents/Codex/2026-07-10/ss/kiko` remains the separate clean
  `main` checkout and must not be used for this integration work.

Three commits were made after the earlier report's `b89065d` snapshot:

1. `d87e2d3 docs(nano): preserve Fable integration handoff`
2. `98e711b feat(map): bind persisted occupancy to exact replay`
3. `e0e8db3 fix(slam): keep Rerun failures diagnostic`

The map commit makes persisted occupancy usable only after exact retained-map
replay agreement, including sparse map identity, map dimensions/resolution,
lower bounds, transform/height bit patterns, revision, cell count, and every
cell class. Rebinding moves the already-decoded grid after equality; it does
not clone the grid. This proves an exact artifact seam, not live-camera
relocalization or robot pose.

Map verification before commit passed 18 focused persistence tests, all 983
`kiko-slam` library tests, strict library/test Clippy (with a documented
`dead_code` allowance for an unrelated in-progress socket helper), and
`git diff --check`.

The Rerun commit separates visualization failures from authoritative live
worker failures in the type system. Dataset publication depends only on the
authoritative failure ledger and finalized dataset descriptor. A Rerun failure
is now explicitly diagnostic and cannot by itself invalidate or validate the
SLAM dataset.

Rerun verification on the Nano passed the focused
`navigation_dataset_publication_uses_only_authoritative_inputs` test under the
explicit compile-only `OAK_SYS_CHECK_ONLY=1` boundary, Rustfmt, and strict
no-dependency Clippy for the record feature. A native build failed closed at
the expected missing DepthAI header check. The compile-only result is not
deployment, camera, or runtime evidence.

The Mac worktree remains deliberately dirty and uncommitted. At the last
freeze it contained 29 modified tracked files plus these three untracked Rust
modules:

- `crates/kiko-head-runtime/src/motion.rs`;
- `crates/kiko-slam/src/navigation/agent_dispatch.rs`;
- `crates/kiko-slam/src/navigation/agent_manual.rs`.

The exact frozen snapshot used for the post-freeze software evidence was:

- `git diff --binary -- .` SHA-256:
  `a5d23a070595cfca4f84806d6b81a4f1dd53ab568efece25d5ea75432544eac5`;
- 29 tracked files, 5,133 insertions, 379 deletions;
- `motion.rs` SHA-256:
  `d273a81d9f2f58d3289fe2f9fc9fee67caca446c1fbb639fd5d7a5030b187327`;
- `agent_dispatch.rs` SHA-256:
  `e48c6e47b9c11444c7281460205775fd7224292e36eb12d1bb2ba2ce149d7017`;
- `agent_manual.rs` SHA-256:
  `f764c18ebc393e0e150d5eeb01262c929aad8f30eb453f484e10325e8a17dea0`.

The patch fingerprint is reproducible only while the dirty snapshot is
preserved; it is not an archived patch artifact or commit. `git diff --check`
passed. Do not infer reviewability or correctness from the counts or hashes,
and do not reset, clean, bulk-copy, or commit the aggregate tree.

### Jetson Orin Nano

- worktree: `/home/makerspace/kiko-nano-expression-integration`;
- branch: `codex/nano-expression-integration`;
- committed HEAD: `bb0f78d`;
- the original Fable report is present there as an untracked file with the same
  SHA-256 as the committed Mac copy;
- source-only copies of the head work, an earlier partial manual slice, and the
  Rerun `kiko_slam.rs` change are dirty there. They are not identical to the
  final Mac dirty tree and are not a deployed binary identity;
- no Kiko systemd unit exists;
- native Rust SLAM remains blocked by the missing pinned native DepthAI
  headers/library, and old binaries also lacked `libonnxruntime.so.1`.

Do not run `git pull`, `git reset`, `git clean`, or a bulk rsync on either
worktree. Their committed bases and dirty slices differ intentionally.

Push topology also matters. The Mac origin is the Nano SSH working repository,
and noninteractive authentication from the Mac currently fails. The Nano
repository's hosted origin is `https://github.com/psygos/kiko.git`. After the
dirty slices are reviewed and committed, the safe route is to create a Git
bundle on the Mac, copy it to a neutral Nano path, fetch it into a temporary
ref, compare the exact commit graph, and push that verified ref to GitHub
without force. Do not push into the Nano's checked-out dirty branch or rewrite
either worktree.

## Post-freeze software slices at the pause

All results below are software-only. No post-freeze physical qualification was
performed.

### Exact inventory/admission

Changed files are limited to the Nano example inventory plus the
`kiko-device-inventory` README, benchmark, source, and tests. The slice adds:

- a private, non-forgeable `ExactInventoryAdmission`;
- one consuming `admit_exact_inventory(expected, observed)` boundary;
- a fixed-capacity, lossless mismatch report retaining both snapshots;
- exact capacity accounting for every singleton and repeated servo mismatch;
- a distinct observed OAK DTO;
- precise `compiled_depthai_header_*` provenance names.

The OAK provenance fields now state only what the C++ bridge was compiled from.
They explicitly do not claim a dynamically linked DepthAI library match or
physical device firmware/bootloader readback. `InventoryComparison` equality
compares only the mismatch prefix, so intentionally ignored STM32/eye boot IDs
cannot make two successful comparisons unequal.

The field rename deliberately changes an unreleased draft V1 schema. Repository
history checks found no tag containing the inventory crate, no hosted remote
branch containing this integration branch, and no production observed caller;
old field names fail closed. This is a one-time draft migration rationale, not
permission to make a future incompatible V1 change without a version bump.

The new `ExactInventoryAdmission` is not production-wired. Current calls to
`admit_exact_inventory` / `into_exact_admission` are confined to the inventory
crate's own implementation and tests; no Nano startup owner consumes the proof.

Nano aarch64 evidence passed:

- formatting check;
- 44 tests: 10 unit, 11 host-boundary, and 23 integration tests;
- strict package Clippy for all targets with `--no-deps -D warnings`;
- rustdoc with `-D warnings`;
- `git diff --check`.

This inventory source was tested in an ephemeral staged clone on the Nano and
then removed. It was not copied into the dirty Nano integration worktree.

Still open: two navigation JSON fixtures in `agent_config.rs` and
`zero_only_config.rs` use the intermediate `linked_depthai_*` names. They must
be migrated before the slice can be committed with a green cross-crate parse
boundary. Dependency-inclusive Clippy also still encounters an unchanged
`robot-protocol` `uninlined_format_args` warning.

### Head return and commissioning owner

The uncommitted head slice now includes:

- a config-bound specialized return owner, so a plan cannot be executed under
  different limits;
- two fresh, stopped, status-zero, ordered command-start telemetry sets within
  a bounded span;
- distinct final/path/regression tolerance invariants;
- all-joint corridor admission before recovery;
- exact-target write before success;
- I/O-inclusive deadlines;
- lossless partial-batch and bounded-history error evidence;
- startup device-status rejection;
- signal-raced startup/return cancellation and immediate fault cleanup;
- `O_NOFOLLOW` config opening and truthful CLI cleanup reporting.

Nano aarch64 evidence passed:

- formatting check;
- `kiko-head-protocol`: 17/17 tests;
- `kiko-head-runtime` library: 41/41 tests;
- `kiko-head-commission`: 8/8 tests;
- doc tests;
- strict Clippy for all head targets;
- rustdoc with warnings denied;
- head-slice `git diff --check`.

Still open: none of this post-run hardening has been physically qualified; no
continuous periodic-telemetry production head supervisor exists; the complete
dirty workspace was not verified; macOS compilation was unavailable in this
session; and the new raw bow temperature-register threshold trip plus XLink
event must be diagnosed before any head re-enable.

### Manual control, dispatcher, and supervisor seam

The uncommitted navigation/supervisor slice now includes:

- explicit `BeginManual` instead of implicit motion-authority acquisition;
- exact pending-begin cancellation with a new fresh-zero barrier;
- fail-closed authority renewal and clock-regression handling;
- finite body-velocity domain values parsed at ingress;
- combined differential-drive body-twist/individual-wheel plant bounds;
- manual deadman and explicit release semantics;
- consumed physical-session fault classification that requires reinventory
  instead of pretending a new applied zero exists;
- dual dispatcher error variants that preserve both the original clock/manual
  fault and a response-delivery failure;
- global-stop obligations represented as obligations, not falsely reported as
  already executed.

Software evidence passed in an isolated Nano aarch64 staging worktree, using
the `agent-runtime,actuation` feature set for `kiko-slam`:

- `kiko-supervisor-core`: 10/10 tests;
- feature-enabled `kiko-slam`: 1,040/1,040 tests before a final Box-only error
  size correction;
- targeted final-tree renewal regression;
- four dispatcher dual-failure regressions;
- final-tree `kiko-slam` Clippy with `--no-deps -D warnings`;
- targeted Rustfmt checks.

The full 1,040-test suite predates both the Box-only correction and the final
inventory rename to `compiled_depthai_header_*`. The latter intentionally
leaves two known-stale navigation fixtures. Therefore the current aggregate
tree is not full-suite green; do not promote the count to final-tree evidence.

Still open and blocking a production owner:

- `AgentControlSocketTask::bind_and_spawn` still uses `thread::spawn`, so OS
  thread-creation failure can panic rather than return a typed startup error;
- dropping `AgentControlSocketTask` still detaches the socket owner without
  signalling shutdown, joining it, or guaranteeing inode cleanup/receiver
  disconnection;
- no bounded Drop-cleanup regression exists;
- `AgentControlCommandV1` has no explicit `Arm` or `Disarm` command. `BeginManual`
  begins the manual authority transition and must not be mistaken for the
  architecture's explicit production arm gate;
- no live owner executes manual/global-stop obligations through the physical
  actuation session and exact applied response;
- this is a tested core seam, not a deployable agent.

## What is actually proven now

In addition to the evidence in the original handoff:

- persisted occupancy can be admitted only against an exact replayed retained
  map in the committed host code;
- Rerun failure is structurally diagnostic rather than authoritative;
- the three dirty software slices have the focused aarch64 evidence listed
  above;
- one temporary prototype opened the exact OAK/head/eye endpoints and produced
  face/eye/head state logs for about four minutes;
- that prototype then reported a raw bow temperature-register threshold trip
  of 86 against 58, an OAK XLink failure, four successful torque-disable
  writes, and clean process exit;
- all four USB identities remained enumerated after exit, with the OAK still at
  USB 2 `HIGH` speed.

## What is not proven

- The face-follow prototype is not correct, calibrated, production-owned, or
  safe to restart. Its geometry did not match the operator-declared assembly
  geometry, and it terminated on two faults.
- No active head owner was observed at the timestamp, but actual head
  power/torque/pose/temperature state is unverified. The head is not thermally
  qualified, periodically supervised, or power-loss safe.
- The OAK is not on `SUPER`, and its production RGB/stereo/depth/IMU graph is
  not cold-boot qualified.
- The uncommitted inventory, head, and manual slices have not received final
  independent integration review or atomic commits.
- There is still no `kiko-nano-agent`, no installed service, no production
  control socket owner, no production head/eyes/OAK fan-out, and no fault
  acceptance harness.
- PWM-to-velocity calibration, wheel signs, stop distance, wheels-on MPC,
  online exploration, live relocalization, and click-to-goal navigation remain
  unproven.
- The current branch is not proven pushed, and neither integration worktree is
  clean.

## Primary continuation goal

The paused product goal remains one production Nano owner that can safely
deliver the user's final example:

1. admit the exact immutable policy, artifacts, OAK, STM32, eye controller,
   head adapter, and four servos;
2. start disarmed, establish fresh exact STM32 applied zero, and supervise a
   thermally/electrically bounded natural head hold;
3. own one OAK and fan bounded RGB, stereo, depth, IMU, SLAM, occupancy,
   expression, and Rerun observations without reopening the device;
4. expose one typed local control socket for status, explicit arm/disarm,
   manual drive, map-only, frontier exploration, map save, revision-bound point
   selection, stop, and shutdown;
5. keep manual/explore/point/commissioning authorities mutually exclusive and
   execute zero before truthful completion on expiry, disconnect, mode change,
   or fault;
6. replay and exactly bind a persisted map, then separately establish fresh
   live localization before allowing motion;
7. qualify the whole graph wheels-off through cold boot and injected faults;
8. only after explicit operator permission and physical safety preparation,
   calibrate the encoderless base with visual forward velocity plus calibrated
   IMU yaw rate, then qualify MPC and click-to-goal navigation.

## Recommended first actions for Fable

1. Read this continuation and the original handoff in full.
2. Confirm no face-follow/head process has restarted and keep the head
   uncommanded. Treat it as potentially energized and gravity-prone. Before any
   bus access or command, have the operator secure the assembly against gravity
   with appropriate support. Manual handling must follow verified
   de-energization and thermal precautions rather than rely on the cleanup log.
3. Verify the read-only evidence copy above by its hash, then investigate the
   raw temperature-register trajectory and OAK XLink failure without issuing an
   enable or position command.
4. Reconcile Mac and Nano worktrees read-only. Do not pull, reset, clean, or
   bulk-copy.
5. Migrate the two stale inventory fixture keys; fix the two socket-task
   lifecycle defects; rerun the complete feature-enabled suites; independently
   review; then make three small commits for inventory, head, and manual work.
6. Design and implement one production Nano owner. Do not revive the temporary
   Python prototype as the product architecture.
7. Keep all physical motion gated on fresh explicit operator consent, known
   wheel state, a safely supported and independently qualified head, clear
   space, and an independent stop.

Useful read-only snapshot commands:

```sh
cd /Users/ttrb/Documents/Codex/2026-07-10/ss/work/kiko
git status --short --branch
git log --oneline --decorate -12
git diff --stat
git diff --check

ssh makerspace@192.168.50.2
cd /home/makerspace/kiko-nano-expression-integration
git status --short --branch
git log --oneline --decorate -12
ps -eo pid,etime,stat,args | grep -E '[k]iko|[r]obot-server'
ls -l /dev/serial/by-id
lsusb
lsusb -t
sha256sum /tmp/follow-v5.log
tail -40 /tmp/follow-v5.log
```

Never place the operator-supplied SSH password in source, documentation,
history, or logs.
