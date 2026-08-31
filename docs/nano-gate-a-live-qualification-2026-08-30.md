# Nano Gate A live qualification ledger — 2026-08-30

This ledger records the attended wheels-off transactions without promoting an
incomplete run into hardware evidence. It supersedes the candidate-selection
state in `nano-current-candidate-readiness-2026-08-27.md`; that older document
remains a historical account of the isolated build and the then-missing
inputs.

## Current disposition

The `qualification-run-20260830-2.typescript` transaction accepted all four
fresh attended preflight statements. Its launch-bound bootstrap then rejected
the live OAK metadata before acquiring the candidate controller or starting a
controller owner:

```text
CommonBootstrap(Stereo(FocalMismatch {
  left_fx: 398.1716,
  right_fx: 396.992,
  left_fy: 398.1898,
  right_fy: 397.00247,
  tolerance_px: 0.001
}))
```

The process confirmed that the OAK closed. The transcript also retains the
required post-failure statement that motor-output power remained physically
disconnected. This was a safe bootstrap rejection, not a failed controller
stop and not a live SLAM qualification result.

The defect was duplicated boundary interpretation, not an inaccurate
calibration or a reason to loosen the numerical tolerance. DepthAI's
rectified-right frame can retain CAM_C source metadata even though its
delivered pixels have been remapped into the common rectified-left
projection. Commit `09880d1cc5b9106412867021018aaeae7bebe84d` centralizes
that rule in `oak_stereo_calibration_from_frame_metadata`: rectified graphs
bind both delivered images to the left projection, while unrectified graphs
retain independent source projections. Recording, qualification, production
bootstrap, and base commissioning now consume that one typed conversion.

An exact source archive for `09880d1` was transferred and built in isolation
at `/home/makerspace/kiko-candidate-09880d1`. The corrected executable was
then passed through the canonical schema-V4 renderer. The renderer regenerated
the executable binding, render evidence, and launch hash rather than allowing
an in-place binary replacement. The prior root-owned bundle remains intact at
`/opt/kiko/qualification-retired-3ff220c-20260830T1833+0530`; the corrected
root-owned bundle is installed at `/opt/kiko/qualification`.

A new foreground transaction was started at
`2026-08-30T18:38:30+05:30` and retained at:

```text
/home/makerspace/kiko-candidate-09880d1/evidence/
  qualification-refresh-09880d1/
  qualification-run-20260830-3.typescript
```

The operator declined to provide its first newly generated wheels-removed
statement. The attended SSH input ended, and the qualifier exited with
`EndOfInput` before opening any device. A subsequent read-only check found no
qualifier, robot server, Fable, DepthAI/OAK, STM32, head, or eye owner and no
listener on ports 8080, 9876, or 9877. No physical attestation was synthesized
or replayed. The corrected software is installed, but Gate A remains
unqualified.

Neither `kiko-nano-agent.service` nor `kiko-robot-server.service` is active.
The production service is not installed. The qualifier is not a boot service
and has not been promoted to production.

## Exact executable provenance

The immutable installed qualifier is:

| Field | Value |
| --- | --- |
| path | `/opt/kiko/qualification/bin/kiko-nano-wheels-off-qualification` |
| bytes | `31,212,784` |
| SHA-256 | `71fb284e47f5484dcd39250b6552a8d984812179f30b5fd79b1d8c2237d7019b` |
| GNU build ID | `cc72d535594c607eeaf5a11cb7abf8fe748bc002` |
| source qualification graph | `09880d1cc5b9106412867021018aaeae7bebe84d` |

The installed executable is mode `0555`, owned by `root:root`, and has no
unresolved dynamic library. Its loader closure resolves the seven
launch-bound native roles from `/opt/kiko/qualification/lib`. A byte-for-byte
`diff -qr` between the 23-file renderer staging tree and the installed tree
passed before this ledger update. The previous executable remains at the
retired bundle path with SHA-256
`3ff220c0848d2216a21695fd38f4c532ab0a5b2a588a41ce328e46443b311435`.

## Fresh OAK capture and calibration inputs

The exact OAK-D S2 `19443010F1B43A2E00` completed a `49.5 s` candidate capture
using DepthAI SDK `3.4.0` over observed `SUPER` USB transport. The capture
recorded:

- `745` left, `745` right, and `745` rectified-left depth frames, including the
  bootstrap frame;
- `709` stereo pairs and `36` explicitly accounted unmatched frames per side;
- `9,889` IMU reports at an observed `199.6 Hz`;
- `12,052` logical payload units written and `0` logical payload drops; and
- 640 by 400 stereo/depth at the observed timed rate of `15.0 Hz`.

The raw capture and log are retained below
`evidence/attended-handoff-20260827T174024+0530/oak-record-full`. The common
rectified-left projection contract is fixed by `3ceb1f4`; left and right use
the same rectified intrinsics, and the baseline is `0.07503394 m`.

The prepared calibration artifact is
`kiko-gate-a-3ceb1f4`, SHA-256
`b16e63c3135fb74acbf8a3cc4db282b930508852b0e41994b397532700a95ebb`.
It binds the OAK identity, rectified stereo model, raw IMU calibration, the
native-IMU-to-base proper rotation, and the tracking-camera-to-base transform.
The declared tracking-camera translation is `[0.1485, 0.0, 0.30] m` in the
axle-centred base frame; the transform and navigation slabs were parsed by the
transactional preparer rather than reconstructed at runtime.

## Immutable Gate A bundle

The root-owned qualification bundle is installed at
`/opt/kiko/qualification`. Directories are mode `0755`, ordinary files are
mode `0444`, and the qualifier is mode `0555`. The launch V4 document is
`4,934` bytes with SHA-256
`f06cb5fd765505f13fac536d2d5e932ca306f6b6481567934956efd8a59e6536`.
The render-evidence document has SHA-256
`f3a5123f4be5ae442d48ba8b0e907503de560c6c0f16885871e32e1d1d84ed16`.
Its exact bindings include:

- OAK RGB, rectified stereo, and rectified-left depth at 640 by 400 and 15 Hz;
- raw IMU at 200 Hz and mandatory `SUPER` USB transport;
- CPU SuperPoint and LightGlue with bounded `512` keypoints;
- a 400 by 400 global occupancy grid at `0.05 m` resolution;
- a 120 by 120 local costmap at `0.05 m` resolution, a `0.343 m` inflated
  footprint radius, and unknown space treated as blocked;
- loopback-only Rerun on `127.0.0.1:9876` with a 128 MiB memory limit;
- a loopback-only unified console on `127.0.0.1:9877` with a capability file,
  20 ms deadman tick, and software stop path;
- a four-servo continuous natural hold at ticks
  `[1505, 3937, 1551, 3018]`, bounded travel and per-axis torque limits;
- KEP2 RGB scene-motion eyes using the one OAK owner; and
- shadow-only MPC using an explicitly synthetic, unvalidated plant.

Gate A starts `disarmed_map_only`. Manual, frontier-exploration, and point-goal
motion permissions are disabled. The STM32 contract is the separate
operator-supervised four-PWM wheels-off candidate, capped at 30 percent, with
its physical stop semantics truthfully marked `unverified`.

The installed eye identity remains the retained KEP2 build
`08134c20df747e68d38bea8af1eb8e62e86b085d347d8e18d5bf18301f368076`.
Matrix-green upload feedback exists in source and passes its host/firmware
tests, but it has not been flashed and is not active on this robot. Gate A also
does not enable expressive physical gaze: it permits the reviewed natural
return and continuous hold only. These boundaries must not be described as
deployed expression behavior.

## Verification already complete

The corrected source graph passed:

- the exact qualifier graph: 1,540 library tests, 7 deployment tests, 104
  runtime tests, 7 template tests, and doctests;
- the complete all-feature graph: 1,581 library tests, 3 base-commissioning
  CLI tests, 14 calibration-preparation tests, 7 deployment tests, 2 plant
  promotion tests, 106 runtime tests, 7 template tests, and doctests;
- strict all-target Clippy for both the exact qualifier and all-feature
  graphs;
- all 85 retained Fable Python behavior/lifecycle tests;
- 8 eye renderer tests, 14 KEP2 contract tests, strict eye-firmware Clippy,
  and an RP2350 release cross-build with a synthetic non-deployable CI identity;
- STM32 embedded logic and firmware build graphs, typed robot protocol/client/
  server, inventory, renderer, deployment gate, supervisor, base
  commissioning, and head/eye protocol/runtime tests; and
- all 8 operator-console JavaScript tests.

The source tree was clean after commit `09880d1`. The first sandbox test run
encountered 27 expected socket `EPERM` failures; the identical graph passed in
the permitted environment. No performance claim is made: these results are
correctness, integration, and build evidence, not a benchmark.

## Evidence still required from a future attended run

Gate A is not passed. The stopped transaction cannot be resumed or represented
as successful. Only if an operator later initiates another attended run and
supplies each challenge at the physical boundary where it is generated can
that new foreground process attempt to prove:

1. current exclusive OAK, STM32, head, and eye acquisition and exact typed
   identities;
2. applied base zero and disarmed state with motor output power disconnected;
3. continuous natural head hold and current RGB-derived eye behavior;
4. live stereo/IMU SLAM evidence, rectified-left occupancy, Rerun delivery,
   bounded persistence, and the unified console;
5. manual deadman, emergency stop, watchdog/fault transitions, and MPC shadow
   without granting production motion authority;
6. the separate attended candidate motor window, if explicitly reached with
   both wheels still removed and the independent physical cut in hand; and
7. final applied zero, disarm, motor-power disconnection, orderly accessory
   shutdown, and a complete terminal transcript.

Until those observations finish, this ledger does not authorize attaching the
wheels, wheel-on plant identification, live MPC drive, autonomous exploration,
click-to-goal navigation, a production service, or a claim that live SLAM is
qualified.
