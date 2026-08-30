# Kiko integrated goal completion audit — 2026-08-30

This audit evaluates the requested end state rather than redefining completion
around the software already written. The target is one robot process that
starts Kiko's character, owns every physical endpoint, builds and visualizes a
live two-dimensional map, exposes one safe manual/agent control plane, and—only
after encoderless plant commissioning—uses MPC for exploration and map-click
point-goal navigation.

The current attended transaction and exact runtime bytes are recorded in
`nano-gate-a-live-qualification-2026-08-30.md`. This document maps each product
requirement to its strongest current evidence and explicitly names what is
still missing.

## Status vocabulary

- **Implemented and host-verified** means the canonical source path is wired
  and its directly relevant tests passed. It is not a physical claim.
- **Target-observed** means exact Jetson/hardware evidence exists for the named
  boundary.
- **Open** means the available evidence is insufficient to claim the requested
  behavior on the assembled robot.

## Requirement audit

| Requirement | Current evidence | Status | Required closure evidence |
| --- | --- | --- | --- |
| Preserve all of Fable's field behavior without retaining a second production owner | `deploy/expression/` is the 85-test incident specification. `kiko-expression-runtime`, `kiko-head-runtime`, `expression_bridge.rs`, and `nano_accessory_worker.rs` implement and connect semantic trace replay, four-axis acts, compliant pet phases, torque switching, organic dynamics, thermal facts, turn posture, residual eyes, lifecycle presentation, and durable pet evidence. | Implemented and host-verified | Attended physical parity for natural hold, face gaze, rest/wake, pet yield/recovery, representative acts, fault presentation, and goodnight using the sole Rust owner. |
| Make two-position eye jitter impossible and keep eye/head transitions coherent | `micro_saccade_holds_then_returns_to_center_with_a_smooth_decay`, `face_tracking_suppresses_random_saccade_offsets`, residual-eye composition tests, and the one-clock expression bridge cover this failure class. | Implemented and host-verified | Observe stable tracked fixation and smooth loss/coast/recovery on the installed eye/head assembly. |
| Use all four servos with organic, graceful, backdrivable pet behavior | The physical policy parser requires complete bow/curl/yaw/roll mapping; compliant and organic policies are parsed once; the actor transaction proves all four goal registers; pet reaction and resting choreography tests cover look-down, shrink, dwell, and varied recovery. | Implemented and host-verified | Materialize and cross-bind the production physical-review asset, then observe contact, bounded yield, rest, return, all-axis health, torque restoration, temperature supervision, and safe shutdown. |
| Show Matrix-green while eye firmware is updated | `firmware/eyes-kep2` owns a dynamic, bounded, panel-phased 2.4-second Matrix boot renderer before KEP2 enumeration; maximum-time arithmetic and total-power bounds are tested. The character engine separately labels host-visible firmware-maintenance facts without pretending it can render during BOOTSEL. | Implemented and firmware-built | Flash a release whose real build identity is bound into the manifest and observe the post-upload Matrix cue. The installed build is still `08134c20…`; Matrix is not currently active. |
| Start as one integrated robot service, replacing Fable's guardian | Production bootstrap exact-loads policy/review/assets, probes and owns head/eyes before OAK/controller admission, starts one accessory worker and pet journal, and binds its completed four-joint health counter to the OAK-loop systemd watchdog. The base unit is content-gated, least-privilege, `Type=notify`, restart-bounded, and only becomes enableable through the exact qualified-boot drop-in. | Implemented and host-verified | Complete wheels-off, wheel-on, and production admission; install the production bundle/unit/drop-in; verify cold boot, readiness, watchdog restart, start limiting, coordinated shutdown, and sole endpoint ownership under PID 1. |
| Use one OAK-D S2 graph over USB3 for RGB, stereo, depth, IMU, expression, SLAM, and occupancy | The exact MXID completed a 49.5-second capture with runtime `SUPER` readback, 640×400 at 15 Hz, 9,889 IMU reports at 199.6 Hz, and zero logical payload drops. `nano-agent` fans one device owner into inference, face/expression, depth, IMU, occupancy, navigation, and Rerun routes. | Partly target-observed | The preserved qualifier must reacquire the same MXID, read back `SUPER`, and simultaneously sustain all consumers with fresh health and bounded queues. |
| Build a real-time 2D occupancy map suitable for SLAM/navigation | Rectified-left metric depth is projected through explicit calibrated transforms into bounded global and local geometric occupancy. Unknown space is blocked, body clearance is inflated, revisions are versioned, and persistence is quota-bound. The installed Gate A graph binds a 400×400 global grid and 120×120 local costmap at 0.05 m. | Implemented and host-verified | Observe continuing live SLAM pose and occupancy revisions in the preserved qualifier, then later drive through representative space and validate obstacle geometry and map consistency. |
| Provide live Rerun visualization designed into the graph | One bounded Rerun thread receives capture, RGB, face diagnostics, sparse SLAM, occupancy, path, costmap, and MPC trajectory data with explicit timeline-domain switching, decimation, memory cap, and typed finalization. | Implemented and host-verified | Observe delivery and display from the qualifier's loopback `127.0.0.1:9876` endpoint; a queue acceptance alone is not display evidence. |
| Give the user and agent one GUI/API with manual control and emergency stop | The embedded loopback console and typed control socket share one linear authority. Browser lease, key/button release, blur, hiding, disconnect, stale state, deadman, stop, save-map, actual authority, localization, map, path, and receipt semantics have Rust and JavaScript tests. | Implemented and host-verified | Exercise the live console through its per-boot capability: stopped map view, manual lease, deadman, browser loss, emergency stop, controller reset, stale localization/depth, and orderly recovery. |
| Stream commands reliably to the STM32 instead of repeating the old UART failure | KRP2 V2 uses finite typed frames, exact controller identity, boot/epoch/sequence, leases, deadlines, applied acknowledgements, explicit zero/disarm, watchdog, one owner, and terminal stop evidence. Separate firmware feature classes prevent a wheels-off candidate image from becoming production firmware. | Implemented and host-verified | Complete the qualifier's applied-zero/fault matrix and bounded wheels-off candidate window, including real watchdog/reset/disconnect and terminal power-cut evidence. |
| Calibrate an encoderless differential-drive plant | The commissioning pipeline binds synchronized visual translation, calibrated IMU yaw, applied PWM receipts, common time, wheel signs, breakaway, response, wheelbase, timing, stop distance, held-out residuals, and an attended review before promotion. IMU alone is explicitly rejected as translation evidence. | Implemented as a commissioning mechanism; physical model open | After Gate A, attach wheels and collect the supervised identification dataset. Fit, validate on held-out runs, review, and promote the exact plant artifact. |
| Drive manually while online SLAM forms a map | The live coordinator joins current localization, depth, occupancy, manual intentions, safety, MPC, and exact physical receipts. Manual mode is mutually exclusive and stops on deadman, stale visual/depth evidence, explicit stop, or owner fault. | Implemented and host-verified | Requires the promoted physical plant and an attended wheel-on drive with measured tracking, stop, and map behavior. |
| Let the agent scan the place autonomously | Frontier selection, bounded exploration policy, in-place yaw scans, map-revision binding, unknown-space blocking, safety, MPC, maximum runtime/goals, and exclusive authority are implemented. | Implemented and host-verified | Qualify wheel-on MPC first, then observe bounded exploration in a prepared area, including replanning, no-frontier completion, obstruction, localization loss, and stop. |
| Click a point on the map and have Kiko navigate there | The console supplies map-epoch/revision-bound goals; point-goal preparation is non-mutating, commit consumes the exact snapshot proof, ABA changes fail closed, and the live MPC driver follows the versioned path through the same safety/receipt owner. | Implemented and host-verified | Save a physically built map, replay/relocalize from fresh camera evidence, click a reachable point, and demonstrate arrival plus stop under the promoted plant. |
| Keep the repository coherent, typed, truthful, and clean | Weak JSON/environment/device boundaries parse once into bounded domain types; runtime claims distinguish configured, selected, applied, displayed, and physically observed evidence. Current code tests and strict Clippy pass; the branch is clean. No performance claim is made without a benchmark. | Implemented for the audited source | Preserve cleanliness through qualification evidence and production materialization; run the full locked test/Clippy/build graph for the final source and exact native artifacts. |

## End-to-end source trace

The production path is not a collection of disconnected modules:

1. `run_nano_agent` parses systemd supervision and creates one bootstrap
   request and one controller-owner runtime.
2. `bootstrap_nano_production` exact-loads the immutable launch assets, waits
   for exact device presence, probes identities, starts the sole accessory
   worker, opens the exact OAK, establishes applied zero, and returns linear
   controller ownership.
3. `prepare_nano_common_live_software` constructs one rectified stereo/IMU
   graph, bounded dataset and map owners, learned inference sessions, geometric
   occupancy, Rerun, and navigation runtime.
4. `run_prepared_live_session` starts bounded capture, inference, occupancy,
   navigation, visualization, and accessory routes. Sparse-SLAM completion—not
   camera freshness—is the localization evidence.
5. `run_live_navigation_worker` owns one coordinator and selects exactly one
   compatibility, production, attended-trial, or wheels-off physical authority.
6. Production control admits manual, frontier, and point-goal requests through
   the same `LiveMotionOwner`; every physical tick crosses current localization,
   collision, lease/deadline, plant validity, MPC, controller, and applied-
   receipt boundaries.
7. Shutdown retains operation, controller, accessory, Rerun, dataset, map, and
   device-close outcomes separately instead of allowing a later cleanup result
   to erase the first failure.

## Current closing sequence

The goal is not complete. The shortest truthful remaining sequence is:

1. Continue the existing foreground Gate A process without restart and finish
   all fresh physical challenges, live graph observations, fault cases, and
   terminal shutdown evidence.
2. If Gate A passes, ask for wheel attachment and run the attended encoderless
   base commissioning workflow. Do not tune MPC against the synthetic Gate A
   plant.
3. Promote only the held-out-validated plant, render the exact production
   launch/policy/review bundle, and pass offline installation admission.
4. Run attended manual mapping, MPC tracking, bounded frontier exploration,
   save/replay with fresh relocalization, and map-click point-goal acceptance.
5. Install and explicitly enable the byte-qualified systemd owner, then verify
   cold boot, expression readiness, live mapping/control, watchdog/fault
   recovery, and clean shutdown with no legacy guardian or second endpoint
   owner.

Only after all five steps have direct evidence can the full integrated goal be
marked complete.
