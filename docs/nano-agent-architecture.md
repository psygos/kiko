# Kiko Nano agent architecture and acceptance contract

This document defines the production integration target for the canonical Kiko
repository. It distinguishes implemented software invariants from claims that
still require the assembled robot. Passing host tests is not physical motion
evidence.

## One owner per physical stream

The deployed system has these ownership boundaries:

- `robot-server` is the sole owner of the configured STM32 serial device. It
  exposes only the typed KRP2 V2 loopback service and reports exact applied
  results; legacy packets cannot reach the V2 actuator.
- the Kiko agent is the sole owner of the exact configured OAK MXID. It fans
  borrowed or bounded observations to SLAM, local occupancy, expression, and
  Rerun without allowing a second camera pipeline to compete for frames.
- one head actor exclusively owns the exact configured Waveshare adapter. No
  motion is possible until read-only inventory and redundant position reads
  succeed for exactly servos 1 through 4.
- one eye actor exclusively owns the exact configured eye UID. It uses the
  versioned KEP2 session protocol; a USB path or VID/PID alone is not identity.
- one supervisor owns lifecycle and motion authority. Commissioning, manual
  driving, point-goal navigation, and frontier exploration are mutually
  exclusive and every transition requires a fresh STM32-applied zero.

No Rerun, HTTP, UI, or expression task owns an actuator transport. They submit
typed intentions to the appropriate owner and can be disconnected without
enabling motion.

## Startup sequence

Every cold boot starts unarmed and follows this order:

1. Parse one bounded, versioned robot manifest. Resolve only exact persistent
   device identities; never choose the first matching serial port or camera.
2. Inventory the OAK, STM32, eye controller, head adapter, and four head servos.
   Compare observed boot/build/config identities with the manifest.
3. Establish device-clock epochs and measure freshness. A restart or timestamp
   regression creates a new epoch and invalidates prior authority.
4. Keep the base at a confirmed applied zero. Lock each head joint at its
   redundantly read present position before any natural-pose transition.
5. Load only calibration and plant artifacts whose content identities match the
   manifest. Parsing proves agreement, not physical truth.
6. Start RGB, stereo, metric rectified-left depth, IMU, online SLAM, occupancy,
   expression, and Rerun streams. Occupancy readiness requires a localized pose
   and a fresh aligned depth integration; a file that merely parses is not a
   live-map readiness claim.
7. Enter `Disarmed`. An explicit arm request can select exactly one authority
   only after the supervisor receives a fresh identity-bound applied zero.

Startup does not automatically perform motion-based plant identification or
servo sign discovery. Those are supervised commissioning operations with
physical hazards, not harmless boot calibration.

## Encoderless commissioning

Kiko deliberately has no wheel-encoder contract. The commissioning dataset
binds each canonical applied left/right PWM sample to:

- visual forward velocity in the base frame;
- calibrated IMU yaw rate in the base frame;
- a common monotonic timebase and device/session identities; and
- a stationary applied-zero segment before and after excitation.

With known wheelbase `b`, the fitted wheel observations are
`v_left = v_forward - b*yaw_rate/2` and
`v_right = v_forward + b*yaw_rate/2`. IMU data alone cannot identify
drift-free translation or PWM-to-linear-velocity gain. It can contribute yaw
rate; visual motion is required for translation.

Commissioning emits only a bounded, reviewed excitation schedule. Missing or
stale visual, IMU, controller, collision, or clock evidence immediately returns
to a required-zero state. A fit is rejected when excitation coverage,
conditioning, sample timing, parameter domain, or holdout residual gates fail.
An accepted result is still an operator-reviewed physical artifact before it
can appear in the actuation manifest.

## Mapping and navigation modes

All motion modes keep online SLAM and fresh local depth collision checking
active:

- **manual** accepts expiring body-frame motion intentions. Expiry, disconnect,
  or replacement by another authority produces zero; it is not raw persistent
  PWM ownership.
- **point goal** accepts a finite map-frame point bound to the exact displayed
  map epoch and revision. A map reset invalidates it. The global planner and
  MPC retain their existing frame, clearance, plant, deadline, and KRP2 applied
  evidence contracts.
- **explore** selects only reachable frontier goals inside an operator-supplied
  map boundary. It stops when no reachable frontier remains, localization is
  lost, a resource budget is exhausted, or the operator cancels.

The geometric occupancy grid is not learned. Fresh local depth can react to a
moving person as a dynamic obstacle, but it does not classify or predict the
person. Unknown, occluded, stale, or out-of-range space remains blocked.

Map persistence is a versioned, checksummed, bounded artifact written by atomic
replacement. Occupancy reload accelerates visualization and planning but does
not itself prove localization. Sparse-map/relocalization state must be rebuilt
from or bound to the corresponding recorded SLAM dataset before motion.

## Expression and head behavior

The RGB expression path samples an already-owned OAK frame. It produces
deterministic scene-motion/person intentions with explicit frame identity and
freshness, mixes semantic reactions, and sends bounded KEP2 eye intentions.
Stale RGB or a failed eye session returns the eyes to firmware fallback.

The default head intention is always `NaturalHold`. RGB does not directly map
to servo ticks. An optional, explicitly configured camera-to-neutral-head
extrinsic can produce typed yaw-right/pitch-down radians for observation and
future qualification; absence makes that projection unavailable, and its
presence grants no head-motion authority. Expressive head offsets remain
disabled until physical yaw ratio/sign, joint envelopes, backlash, stop
behavior, voltage/temperature
limits, process-kill behavior, and safe natural-pose approach have been
qualified on this assembly.

## Rerun and control adapters

Rerun is the shared diagnostic view for RGB, stereo, pose, map, local costmap,
frontiers, selected goal, path, MPC rollout, applied controller receipt,
supervisor state, expression source, and head/eye health. Every item is logged
on its real device or host timeline with explicit transforms. Rerun is not a
safety authority or complete decision ledger.

The pinned Rerun SDK is output-only. A click adapter therefore submits the same
typed `(map_epoch, revision, x_m, y_m)` command through the local control API;
the agent never pretends Rerun supplied a callback it does not provide.

## Cold-boot acceptance

The software acceptance harness must prove, with simulated transports and fault
injection, that:

1. exact inventory succeeds and every wrong/missing/rebooted identity fails;
2. the robot remains unarmed and at confirmed applied zero through startup;
3. the head locks at present pose and never approaches natural after a failed
   telemetry or approval gate;
4. RGB motion produces an expiring eye intention and stale RGB falls back;
5. online SLAM produces a localized, checksummed occupancy artifact;
6. manual, explore, and point-goal authorities cannot overlap;
7. a selected map point reaches the planner/MPC and only an exact applied
   result permits the next command;
8. save, reload, dataset-bound relocalization, and continued mapping preserve
   the declared map/frame identities; and
9. camera loss, stale depth, localization loss, controller reset, serial loss,
   process cancellation, and clock faults all require or confirm zero.

Physical acceptance additionally requires the assembled devices to be visible
on the Nano and an independent emergency stop. The read-only inventory on
2026-07-20 found the OAK, STM32 ST-Link serial function, eye controller, and
head adapter on an NVIDIA Jetson Orin Nano. That establishes USB presence only:
no camera stream, STM32 control identity, KEP2 eye session, head hold, emergency
stop, motor motion, or drive result is claimed by this document. See
`nano-validation-evidence-2026-07-20.md` for the exact observations and native
aarch64 host-test evidence.
