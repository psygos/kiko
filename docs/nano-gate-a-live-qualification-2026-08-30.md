# Nano Gate A live qualification ledger — 2026-08-30

This ledger records the current attended wheels-off transaction without
promoting an incomplete run into hardware evidence. It supersedes the
candidate-selection state in
`nano-current-candidate-readiness-2026-08-27.md`; that older document remains a
historical account of the isolated build and the then-missing inputs.

## Current disposition

At `2026-08-30T16:54:27+05:30`, one foreground qualifier was still alive on the
Jetson Orin Nano. It was started at `2026-08-30T15:57:42+05:30` through
`script(1)` so its attended terminal transcript is retained. The process is
waiting at its first physical preflight prompt, before it opens any hardware
endpoint:

```text
WHEELS REMOVED 538853e7f949604571b7ac7061f9011a
```

That response must come from the person physically observing the robot. It has
not been entered by software and is not claimed by this ledger. The live
process and its SSH tunnels have deliberately remained running; no restart is
required to continue this exact challenge.

The transcript is retained at:

```text
/home/makerspace/kiko-candidate-3ceb1f4/evidence/
  attended-handoff-20260827T174024+0530/
  qualification-run-20260830-2.typescript
```

Neither `kiko-nano-agent.service` nor `kiko-robot-server.service` is active.
The production service is not installed. The qualifier is not a boot service
and has not been promoted to production.

## Exact executable provenance

The immutable installed qualifier is:

| Field | Value |
| --- | --- |
| path | `/opt/kiko/qualification/bin/kiko-nano-wheels-off-qualification` |
| bytes | `31,216,704` |
| SHA-256 | `3ff220c0848d2216a21695fd38f4c532ab0a5b2a588a41ce328e46443b311435` |
| GNU build ID | `fd49573f1ee242e84db901eaf7cb793a1e89097a` |
| source qualification graph | `3ceb1f413c345e3d0ff6225a38bb3e57a21e45bf` |

The verified code tip before this ledger commit is
`ae85b1e0d926736f705bb409431f8cce760db16d`. Its complete diff from `3ceb1f4`
is CI-only: qualification source, manifests, lockfile, models, and runtime
configuration are unchanged. An isolated Orin rebuild from that restored
source graph produced the same byte length, SHA-256, GNU build ID, and a
successful byte-for-byte `cmp` against the installed executable. Therefore the
preserved physical run is executable evidence for that code tip and later
documentation-only commits; no new process was substituted under the live
terminal.

This equality was checked after rejecting two intermediate builds. A
source-scoped conditional lint attribute changed the release ELF even though
it did not change the recording feature's behavior. The final CI keeps strict
minimal-library linting and exercises the complete no-default-feature graph
with the two record-only dead-code warnings allowed at the command boundary,
leaving the qualification source graph reproducible.

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
`/opt/kiko/qualification`. Ordinary files are mode `0444`; the qualifier is
mode `0555`. The launch V4 document is `4,934` bytes with SHA-256
`6d018f29dfda2978f5acd7a95e30295758f14662d1217c5d30b714ea8c33fc56`.
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

Before this ledger, the current source graph passed:

- the complete all-feature Kiko SLAM test graph, including 1,579 library tests
  and 106 CLI tests;
- strict all-feature Clippy for Kiko SLAM and the expression/core/head crates;
- strict minimal-library Clippy plus complete minimal-target Clippy with the
  scoped record-provenance allowance;
- all 85 retained Fable Python behavior/lifecycle tests;
- 8 eye renderer tests, 14 KEP2 contract tests, strict eye-firmware Clippy,
  and an RP2350 release cross-build with a synthetic non-deployable CI identity;
- STM32 embedded logic and firmware build graphs, typed robot protocol/client/
  server, inventory, renderer, deployment gate, supervisor, base
  commissioning, and head/eye protocol/runtime tests; and
- all 8 operator-console JavaScript tests.

The source tree was clean after commit `ae85b1e`. No performance claim is made:
these results are correctness, integration, and build evidence, not a
benchmark.

## Evidence still required from this preserved run

Gate A is not passed yet. After the human supplies each fresh challenge at the
physical boundary where it is generated, the same foreground process must
still prove:

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
