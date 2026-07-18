# Encoderless navigation shadow architecture

## Closure boundary

This lane closes at **replay-bound live host shadow mode**. Kiko accepts an exact map-frame point at
the typed command-line boundary, maintains a continuous local pose, plans a revision-bound route,
incorporates the newest admitted depth frame as local collision evidence, and computes the
direct-PWM command that a bounded controller would request. Every admitted control tick produces
one fail-closed shadow decision. The process has no motor-command transport and emits zero STM32
motor packets.

Live sensor, visual-attempt, map, goal, and tick admissions are appended in the order actually
observed by the coordinator and bound to the captured dataset. Operating-system thread scheduling
is neither controlled nor claimed deterministic. The journal is the replay authority for that
observed order; a replay consumes those typed admissions rather than reconstructing a schedule.
Given the same admitted events and configuration, the algorithms are deterministic, but no
bit-for-bit cross-platform floating-point claim is made for platform `libm` operations.
The recording binds payloads to the journal; it does not embed the JSON configuration or software
revision. Reproduction therefore also requires the same external configuration and code revision.

Physical actuation, plant identification on the robot, Jetson power/thermal work, and deployment
remain outside this host lane. Removing the shadow lock requires a separate reviewed change and
physical validation.

## Data flow and authority

```text
OAK stereo --------> existing visual SLAM --------> map pose/corrections
       |                                             |
OAK IMU -----------> continuous planar estimator <---+
       |                    | odom pose       | map<-odom
       |                    v                 v
       +-----------> versioned dataset +      global occupancy snapshot
                         ingress journal
                                                   |
newest OAK depth --> dynamic local costmap          +--> revision-bound global plan
       |                    |                                  |
       +--------------------+--> fail-closed safety <----------+
                                      |
versioned plant model + limits ------> deadline-bounded MPC
                                      |
                                      +--> shadow command + Rerun only
```

The existing SLAM map remains authoritative for global localization and loop correction. `odom`
is the locally continuous control frame; global corrections update `map <- odom` instead of
jumping the controller state. `base` is the planar robot frame. OAK optical and IMU measurements
remain in their native frames until an explicit calibrated extrinsic converts them.

IMU data alone cannot determine drift-free planar translation. With no wheel encoders, continuous
motion therefore comes from visual odometry constrained by calibrated inertial measurements. If
camera-to-base, IMU-to-camera, gravity/ground-plane, or clock-session authority is missing, the
estimator and controller must stay unavailable rather than assume alignment.

## Boundary contracts

- Every OAK connection is one nonzero device-clock session. A reconnect creates a new session;
  sequence or timestamp restarts are never silently joined to the old session.
- Accelerometer and gyroscope samples retain independent device timestamps. Acceleration is
  metres per second squared and angular velocity is radians per second in the native OAK IMU
  frame. Host dequeue time uses a separate monotonic process clock with no wall-clock meaning.
- Depth is metric and tied to its exact optical projection contract. The live navigation branch is
  capacity one/drop-oldest so it exposes the newest complete observation without delaying SLAM.
- Command-line goals, environment values, manifests, calibration files, and plant models are
  parsed once. Nonfinite values, ambiguous units, unsupported versions, stale identities, and
  incompatible frames are typed failures. Live navigation requires `--navigation-config`,
  `--navigation-goal X_M,Y_M`, and `--navigation-record` together.
- A point goal identifies its map epoch. A computed path identifies the exact occupancy revision
  and safety profile used to produce it. Map changes require explicit path revalidation or
  replanning; they never silently bless an old path.
- A global map received before the first visual localization anchor is journaled, but planning is
  explicitly deferred until a later map revision arrives after that anchor. The current
  coordinator does not silently reinterpret the pre-anchor revision.

## Dynamic-obstacle semantics

The local costmap is deterministic geometry, not a learned occupancy network. Every accepted
depth frame replaces short-lived local evidence. A person crossing the camera field can therefore
block or invalidate the current trajectory before the slower global map changes. This is collision
evidence, not semantic human detection or motion prediction: occluded, out-of-range, missing, or
expired space is non-traversable, and a stale local map forces a stop.

The local map clears only the raw cells conservatively covered by the robot's embodied footprint
at the current pose before obstacle inflation. It does not clear the clearance ring or unseen
space. With a forward-looking depth camera, unseen rear and side space therefore remains blocked;
this can correctly stop a rollout even when the body's current cell is known free.

Robot footprint and clearance are applied before trajectory acceptance. The safety supervisor
also rejects missing localization, stale inputs, map/path identity mismatch, infeasible controller
results, deadline misses, and command-session loss.

## Controller and plant-model contract

The navigation controller models the completed direct-PWM STM32 contract; it does not invent an
encoder interface and the host shadow adapter does not import or open a command transport. Its
differential-drive plant parameters must come from a versioned identified model with units,
dataset provenance, fit residuals, and a declared validity envelope. Shadow mode may exercise
controller mathematics with a synthetic fixture model, but that does not establish physical
accuracy and cannot unlock actuation. The checked-in
`configs/navigation-shadow-v1.example.json` is deliberately marked synthetic, non-actuating, and
not physically validated; its values are a schema example, not robot calibration.

Each optimization has a host-monotonic deadline and a typed safe fallback. A late or infeasible
result is a stop request, never a reused or partially computed command. Output limits, slew limits,
horizon timing, signs, and left/right channel ordering belong to parsed types shared with the
shadow command session.

## Rerun evidence

Shadow mode emits the map and local grids, exact frame transforms, pose quality, goal and path
provenance, predicted trajectory, solver status, safety reason, and requested PWM on one coherent
Rerun timeline. It also exposes an explicit `motor_packets_sent = 0` counter. Rerun is diagnostic,
output-only evidence; it is not part of the safety decision, and its failure cannot enable motion.
The durable dataset binds the sensor payloads to the exact coordinator-admission journal; Rerun is
not the replay authority.

The global goal and path use the registered map frame. The predicted trajectory and local grid are
kept under explicitly named odom and capture-time local frames, with exact transform scalars and
provenance logged alongside them. This closure does not claim that Rerun spatially overlays the
capture-time local grid onto the global map.

Rerun is output-only in the pinned host SDK. The live closure therefore accepts a typed map point
through `--navigation-goal X_M,Y_M` and renders that goal in Rerun. A future viewer-click adapter
may construct the same typed goal, but this closure has no map-click callback and does not claim
interactive goal selection.

## Verification boundary

Host unit, property, regression, compile, lint, and benchmark evidence can establish parsing,
provenance, numerical, replay, fail-closed, and structural no-transport behavior. It cannot prove
camera/IMU calibration, plant identification, physical collision clearance, motor response, or
closed-loop navigation. No Jetson/Nano run, deployment, GPU benchmark, power/thermal tuning, or
physical STM32 validation is part of this closure.

Dataset publication proves that the captured payload writers and synchronized ingress journal
completed and revalidated. OAK device close happens afterward; a close failure is returned as a
separate session error but does not retroactively invalidate an already complete dataset. OS
thread creation still uses the existing `std::thread::spawn` panic boundary, so resource-exhaustion
spawn failure and multi-worker fault injection remain explicit hardening debt rather than verified
typed-error paths.
