# Attended Nano navigation trial

## Purpose and claim boundary

`kiko-slam nano-attended-navigation-trial` is the foreground, wheel-on bridge
between the already hardened base-commissioning controller and Kiko's canonical
live robot graph. It is intended for the first attended map-and-drive session,
not unattended service startup.

One invocation owns and connects:

- the launch-bound OAK-D S2 RGB, rectified stereo, depth, and IMU streams;
- natural head hold, eye expressions, RGB face tracking, and the optional
  evidence-gated physical gaze lease;
- sparse SLAM, dense depth, the local and global 2D occupancy grid, persistence,
  and Rerun diagnostics;
- the unified loopback operator/agent control service;
- manual body-frame velocity control, mapping-only operation, frontier
  exploration, map-click point goals, and the existing MPC controller; and
- exactly one attended STM32 V2 controller owner.

This executable does not prove camera delivery, SLAM accuracy, occupancy
quality, actuator polarity, wheel velocity, MPC tracking, emergency-cut
behavior, or safe autonomous navigation. Those remain physical acceptance
results. It also does not promote a newly fitted plant. It consumes the exact
plant already bound by the commissioning launch.

## Static admission

The three CLI inputs are the complete public launch boundary:

```text
kiko-slam nano-attended-navigation-trial \
  --deployment-root /ABSOLUTE/DEPLOYMENT/ROOT \
  --launch-config RELATIVE/BASE-COMMISSIONING-LAUNCH-V1.json \
  --state-root /ABSOLUTE/STATE/ROOT
```

Build and install the exact feature-qualified Nano artifact:

```text
cargo build --locked --release -p kiko-slam \
  --target-dir target/nano-attended-navigation-trial \
  --no-default-features --features nano-attended-navigation-trial \
  --bin kiko-slam
```

The separate target directory prevents a later production-only build from
silently replacing the local executable with one that correctly omits this
trial entry point. A native Nano build must not set `OAK_SYS_CHECK_ONLY`; that
variable is exclusively for host compile/test checks and disables the native
OAK bridge.

There are deliberately no CLI or environment flags for wheels attached,
operator attendance, clear motion area, reachable independent power cut, or a
PWM override. Those claims are collected only by the fresh, nonce-bound
controlling-TTY ceremony after the camera, accessories, inference runtime,
SLAM, occupancy, map persistence, and Rerun software have prepared
successfully.

Preparation parses and binds the exact launch, policy, controller profile,
controller contract, device manifest, calibration, live graph, accessory
policy, navigation configuration, plant, inference runtime, models, and face
cascades. The physical attestation directly records the navigation and plant
digests in addition to the parent live-graph digest. A missing, changed,
aliased, oversized, mismatched, or weakly typed input rejects before motion
authority is created.

The OAK graph requires a configured and observed USB SuperSpeed transport.
DepthAI's connected MXID and live stereo geometry must match the calibration.
Navigation is parsed against the live depth-camera model and the exact plant,
then checked against the calibration before the STM32 is opened.

## Ownership and startup sequence

The runtime has one owner for each physical device. It does not start a
commissioning camera and later race a second SLAM camera for the same OAK.

1. Parse the immutable deployment and state roots.
2. Start the canonical accessory worker. The worker returns the neck to its
   configured natural pose and holds it continuously, starts the eye expression
   scheduler, and consumes the RGB stream for face tracking.
3. Open the exact OAK once, require USB SuperSpeed evidence, verify its MXID and
   stereo geometry, and retain the bootstrap stereo pair for the full tracker.
4. Build the sparse/dense SLAM, occupancy, map-persistence, Rerun, planner, and
   MPC graph.
5. Collect the fresh physical TTY attestation.
6. Start one STM32 controller and require its exact applied-zero receipt.
7. Construct the unified authority supervisor in `Disarmed`. No manual or
   autonomous authority exists yet.
8. Start the private loopback console and control socket. The browser and an
   agent use the same typed intent ingress and the same sole-owner authority
   state machine.

The attended controller retains the profile's command ceiling (at most 20%
timer duty) and attestation expiry for every physical application. An Arm
intent still requires a new controller-applied zero before the supervisor can
enter `ReadyStopped`. Manual velocity or autonomous intent cannot bypass that
barrier.

## Unified control

The browser recognizes `attended_navigation_trial` as an SI navigation
authority, not as production and not as raw-PWM qualification. It provides:

- held Arrow/WASD body-frame velocity commands with browser release handling
  and the server's monotonic deadman;
- Arm, Disarm, Stop, mapping-only, frontier exploration, and Save Map;
- a live occupancy grid with pose, planned path, MPC prediction, and applied
  command evidence;
- a map click that is accepted only on a currently free cell of the exact
  displayed revision while localization and OAK motion evidence are current;
  and
- a one-way software safety-stop latch. The physical emergency cut remains an
  independent mechanism and is never represented as a browser substitute.

The HTTP service remains loopback-only. Access it through an explicit SSH
forward to the configured console port and read the per-boot capability from
the configured private runtime path. Neither value is guessed by this
runbook.

## Failure and shutdown semantics

Any preparation failure closes the sole OAK before releasing the accessory
hold. If the STM32 had opened, controller shutdown is attempted first and its
result is retained alongside the primary failure.

During live operation, a stale sensor stream, control-transport failure,
authority mismatch, controller fault, expired attended admission, missed
deadman, or shutdown request enters the existing fail-closed owner path.
Navigation motion is stopped before the accessory worker releases the neck and
eyes. OAK and controller cleanup failures are reported rather than hidden by
the primary error.

## Verification before asking for wheels

Host verification is necessary but cannot satisfy the wheel-attachment gate:

```text
OAK_SYS_CHECK_ONLY=1 cargo test --locked -p kiko-slam \
  --lib --bin kiko-slam --no-default-features \
  --features nano-attended-navigation-trial

OAK_SYS_CHECK_ONLY=1 cargo clippy --locked -p kiko-slam \
  --all-targets --no-default-features \
  --features nano-attended-navigation-trial -- -D warnings

node crates/kiko-slam/src/operator-console/view-model.test.js
node --check crates/kiko-slam/src/operator-console/app.js
```

Before wheels are attached, the Nano must additionally prove, with motor power
disconnected and without replacing the currently running head/eye owner until
the canonical handoff is ready:

1. the rendered deployment has no placeholder hashes or paths;
2. the exact Nano aarch64 binary and native OAK bridge start successfully;
3. the OAK reports SuperSpeed and all RGB/stereo/depth/IMU streams remain
   healthy;
4. the neck is actively held at the reviewed neutral pose and the current eye
   expression and RGB tracking are visibly present;
5. the motion-capable STM32 firmware returns the expected V2 identity, exact
   applied-zero, disarm, watchdog, restart, disconnect, and fault evidence with
   motor power disconnected;
6. the private console can observe the grid and exercise Stop/Disarm without
   granting nonzero motion; and
7. the independent physical power cut is reachable and ready for the attended
   wheel-on ceremony.

Only after those checks pass is the truthful next instruction: attach the
wheels for attended polarity, velocity, plant, localization, occupancy, and
MPC acceptance. Until then, this code is software-ready evidence, not a claim
that Kiko is physically ready to drive.
