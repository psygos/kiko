# Kiko Nano agent architecture and acceptance contract

This document defines the production integration target for the canonical Kiko
repository. It distinguishes implemented software invariants from claims that
still require the assembled robot. Passing host tests is not physical motion
evidence.

## One owner per physical stream

The deployed system has these ownership boundaries:

- `kiko-slam nano-agent` retains the sole in-process owner of the configured
  STM32 serial device and its typed KRP2 V2 loopback service. It reports exact
  applied results and joins that owner during shutdown; legacy packets cannot
  reach the V2 actuator. The former standalone `robot-server` systemd unit and
  split wheels-off executable have been removed; an older installed copy must
  not run beside production or qualification.
- the Kiko agent is the sole owner of the exact configured OAK MXID. It fans
  borrowed or bounded observations to SLAM, local occupancy, expression, and
  Rerun without allowing a second camera pipeline to compete for frames.
- one head actor exclusively owns the exact configured Waveshare adapter. No
  head motion is possible until the head/eye policy is bound to the parsed
  manifest and finite read-only identity plus redundant position reads succeed
  for exactly servos 1 through 4. Production adopts that fresh pose without
  first torque-disabling the neck; prior torque state remains unknown.
- one eye actor exclusively owns the exact configured eye UID. It uses the
  versioned KEP2 session protocol; a USB path or VID/PID alone is not identity.
- one supervisor owns lifecycle and motion authority. Commissioning, manual
  driving, point-goal navigation, and frontier exploration are mutually
  exclusive and every transition requires a fresh STM32-applied zero.

No Rerun, HTTP, UI, or expression task owns an actuator transport. They submit
typed intentions to the appropriate owner and can be disconnected without
enabling motion.

## Startup sequence

Every cold boot starts with no base owner and follows this order:

1. Load and parse the bounded launch assets and versioned robot manifest once.
   The canonical calibration artifact is launch-, policy-, manifest-, path-,
   and digest-bound before hardware access. Bind the expected head/eye policy
   to exact persistent identities; never choose the first matching serial port
   or camera.
2. Within the compiled fixed 30-second device-presence window, poll only until
   one sequential composite polling pass reports the exact head, eye, and
   STM32 serial-by-id character devices and exact OAK MXID in `Available` or
   `InUse` state. It observes immediately, then pauses for at most 100 ms after
   each unsuccessful observation; shutdown is checked before every observation
   and sleep and between serial metadata calls before native OAK discovery.
   `Bootloader` and `Unknown` OAK states keep the wait active. An OAK reported
   `InUse` counts as present for timing only: every serial probe, OAK connect,
   and controller open below is still attempted exactly once, so a competing
   owner or any other exclusive-open failure returns immediately instead of
   restarting this poll. Timeout retains each serial-presence bit and the exact
   OAK's last observed `Available`, `InUse`, `Bootloader`, `Unknown`, or missing
   state when a complete prior pass exists; a timeout during the first partial
   pass has no composite snapshot. Presence proves no identity, ownership, USB
   speed, or continued availability. The fixed duration cannot preempt a
   filesystem or native DepthAI call already in progress, but it prevents the
   next component call from starting once the preceding boundary returns.
3. Complete finite read-only eye, adapter, and four-servo probes. The probe
   sessions close before the single production accessory owner opens either
   serial device.
4. Start the production accessory owner before opening OAK or starting the
   STM32 owner. Its dedicated face-perception thread first constructs both
   OpenCV classifiers from the exact retained launch-bound cascade bytes and
   must publish bounded readiness; no head or eye actor starts after detector
   construction fails or times out. The head owner then observes each joint
   twice, admits the complete pose inside the exact policy window and freshness
   budget, writes that same pose with bounded speed and torque limits, enables
   or refreshes torque, and verifies two stopped readbacks before any
   natural-pose transition. It performs no pre-observation torque-disable.
5. Open only the exact OAK MXID, require SuperSpeed, and bit-exactly bind its
   observed rectified stereo geometry to the retained calibration artifact.
   Require the artifact's raw IMU and tracking-camera-to-base values to equal
   the parsed navigation configuration and its three calibration IDs to equal
   the production actuation approval. Continuously reject a terminal accessory
   fault while waiting for the first stereo pair.
6. Start the exact STM32 owner in a stopped state, acquire an acknowledged
   applied zero, and compare the complete observed OAK/STM32/accessory/artifact
   inventory with the manifest. No base motion authority exists during the
   earlier head return.
7. Establish device-clock epochs and measure freshness. A restart or timestamp
   regression creates a new epoch and invalidates prior authority.
8. Start RGB, stereo, metric rectified-left depth, IMU, online SLAM, occupancy,
   expression, and Rerun streams. Occupancy readiness requires a localized pose
   and a fresh aligned depth integration; a file that merely parses is not a
   live-map readiness claim.
9. Enter `Disarmed`. An explicit arm request can select exactly one authority
   only after the supervisor receives a fresh identity-bound applied zero.

After the accessory owner is ready, a later bootstrap failure first stops and
joins any base owner, closes OAK when it was opened, and only then requests eye
release plus head serial-ownership release. The head release sends no torque
switch write and therefore preserves the last admitted goal, but it does not
prove that physical torque remains present.

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

Map persistence is a versioned, checksummed, bounded occupancy artifact written
by atomic replacement. In production, **Finalize map & stop** is always a
terminal checkpoint, not a live snapshot button: it closes capture,
drains inference and occupancy in causal order, stops the controller, finalizes
and synchronizes the navigation journal, reads it back to derive the final
accepted map epoch/revision, and requires that identity to exactly equal the
retained occupancy before finalizing the session manifest. It writes and quota-verifies the
final occupancy once at the configured staging path, moves that exact inode
without replacement into the session as `occupancy.kmap`, then atomically replaces
`navigation/selected-warm-start-v1.json`. That selection names one direct-child
session and records the exact manifest and occupancy byte lengths, SHA-256
digests, map epoch, and final revision. A failed pre-publication checkpoint
does not select the incomplete session.

The direct socket gives this terminal transaction its own parsed five-minute
deadline; ordinary commands keep their shorter deadline. The main/worker
dataset handoff and finalization acknowledgement use the remaining time from
that same absolute boundary. For browser-originated requests, the loopback HTTP
owner stays available through the final response record and for a bounded
two-second polling grace (or until that exact record is observed), then shuts
down.

Warm start ignores an unselected mutable session and resolves only that
selection. It retains the selected session, manifest, and occupancy descriptors;
the occupancy parser and manifest parser each digest the exact bytes consumed
from that handle rather than reopening a previously hashed pathname. It checks
those selected digests before replay and again before binding the replay result.
Before processing stereo payloads,
it streams the manifest-bound fixed-record navigation journal in constant
memory under its 1,048,576-record format cap. The final journal epoch must
contain an accepted global map, and that event's epoch and revision must equal
the atomic selection; a later empty epoch or any mismatch fails closed.
Historical frame IDs use a disjoint reserved namespace; replay requires exact
rectified-left depth geometry and must reconstruct occupancy bytes that match
the persisted artifact. That match publishes the map as `lost`, never
`localized`. Only a fresh OAK frame that produces the tracker's typed
multi-frame relocalization success, good tracking health, and a current pose
on the same advancing map lineage opens the localization gate.

The content claim is deliberately narrow: the selection cryptographically
binds `manifest.json` and `occupancy.kmap`; it is not a whole-directory payload
tree digest. `DatasetReader` still validates manifest-defined payload
structure, identities, and lengths, and exact reconstructed occupancy is
required, but arbitrary payload bytes—including navigation journal records—are
not independently content-addressed. Journal records are structurally parsed
and order-checked before their final map identity is accepted. The selection
read is capped at 4 KiB, terminal manifest hashing at 64 MiB, and
selected-occupancy hashing at 256 MiB (with the produced artifact's exact
encoded length used during publication). No terminal hashing step walks the
complete quota-bounded dataset payload tree.

The retained handles close pathname replacement between selection hashing and
manifest/occupancy parsing. They do not make the generic `DatasetReader`
payload tree immutable: metadata, calibration, frame, depth, IMU, and sidecar
payload opens remain path-based, and those payload bytes are not selected
digests. An active same-UID process that can mutate the service-owned dataset
during replay is therefore outside this checkpoint's integrity claim. Nano
operation must give the runtime exclusive ownership of its private state tree;
a future whole-dataset descriptor-relative/content-addressed reader is required
before claiming resistance to that actor.

The launch storage policy separates map persistence from navigation-dataset
retention. The exact encoded map snapshot cannot exceed
`maximum_map_snapshot_bytes`, and atomic replacement must leave at least
`minimum_free_bytes_after_map_save` available. Dataset payloads, sidecars, IMU,
journal, and manifest share independent cumulative logical-byte and
regular-file ceilings; journal records also have an independent count ceiling.
Every dataset-owned write must preserve the configured descriptor-relative
free-space floor. Logical byte totals are exact file lengths, not a claim about
fragment-rounded physical allocation. The file ceiling is currently capped at
65,536 because finalization builds one bounded monolithic manifest. Longer
sessions require a reviewed chunked-manifest format rather than a larger
launch value. The launch record ceiling cannot exceed the journal format's
existing 1,048,576-record hard bound.

Open-ended capture cannot consume
`navigation_dataset_terminal_reserve_bytes`. Admission requires that reserve
to be below the dataset logical-byte maximum and at least the
4096-byte-fragment-rounded sum of the configured map ceiling, bounded 64 MiB
manifest, and 4 KiB warm-selection ceiling. Final occupancy and selection
remain map-persistence artifacts: their bytes are not adopted by the dataset
logical-byte counter, while the withheld reserve protects their terminal
allocation. Descriptor retention closes path replacement at quota checks, but
concurrent external writes can still race a reservation; exact post-write
verification must report a violated floor or identity rather than claim safe
publication. Startup evidence and total state-root usage have no aggregate
byte-limit claim.

## Expression and head behavior

The canonical V3 RGB expression path moves each already-owned OAK frame into a
capacity-one detector lane after one authoritative ingress-time sample. That
lane parses the frame identity once, borrows the same pixel allocation for the
bound OpenCV frontal/profile cascades, applies the bounded Fable-derived face
association policy, and moves the exact parsed frame and result into the
existing expression actor. The actor combines deterministic scene motion with
an `Important` face-attention intent and sends KEP2 eye intentions. Haar level
weight is retained only as an opaque ranking value: it is not a probability,
person identity, distance estimate, or `PersonObservation`. This provides
face-directed eye attention, not semantic human identification, prediction,
or safety occupancy. Stale RGB, detector failure, or a failed eye session is a
typed terminal accessory fault which disarms the base; coordinated shutdown
then attempts and verifies the eye release to firmware fallback and the
hold-preserving release of the reviewed natural head pose. Missing release or
join evidence is a shutdown failure, not a successful fallback claim.

OpenCV is used only for this bounded, host-side Haar rectangle detector. The
cascades are pretrained static assets; Kiko performs no online training or
model update. This choice preserves the audited Fable behavior without adding
a second OAK owner or making expression attention part of SLAM, MPC, obstacle
avoidance, identity, or the safety case. Replacing it with an OAK-NPU detector
is a separate measured optimization and accuracy qualification, not a current
performance claim.

The default head intention is always `NaturalHold`. RGB does not directly map
to servo ticks. An optional, explicitly configured camera-to-neutral-head
extrinsic can produce typed yaw-right/pitch-down radians for observation and
future qualification; absence makes that projection unavailable, and its
presence grants no head-motion authority. Expressive head offsets remain
disabled until physical yaw ratio/sign, joint envelopes, backlash, stop
behavior, voltage/temperature
limits, process-kill behavior, and safe natural-pose approach have been
qualified on this assembly.

The production start window is the exact per-joint union of the evidenced Fable
return-start envelope and the reviewed natural target plus/minus its 20-tick
readback tolerance:
`[2135..2227,2525..2592,2842..2963,2856..2922]`. Policy parsing rejects a
window which excludes any part of the reviewed hold envelope or widens beyond
the evidenced union. The production head handle has no torque-disable
operation. A terminal health/accessory fault stops the base/eye path but keeps
the head owner and hold alive. A startup fault closes ownership without
altering or claiming the prior servo state. Return faults, handle loss, and
ordinary process or systemd shutdown preserve the last admitted goal while
eventually releasing serial ownership without a torque-switch write. That is
not torque readback: power loss, servo protection, or another bus owner can
still release the neck. Intentional torque release remains a separately
supported commissioning action.

## Rerun and control adapters

Rerun is the shared high-bandwidth diagnostic view for decimated RGB, stereo,
depth, pose, occupancy map, local costmap, selected goal, global path, MPC
rollout, control-tick timing, exact applied-controller evidence, Haar face
rectangles, and the selected face target. RGB is copied only after strict BGR8
layout admission into a capacity-one, drop-oldest diagnostic queue; that copy
cannot feed SLAM, expression, control, or safety.

A face result can overlay only the RGB image with the exact same stream epoch,
device capture sequence, host delivery sequence, timestamp/reference, and
dimensions. RGB and detector-result identities use separate timelines, and
timeline-domain switches explicitly clear sticky foreign time state. Every
new RGB image clears the prior rectangles and target before an exact-key match
is applied, so dropped, late, or unmatched detector work cannot remain visible
on a newer frame. A matched empty batch is published as empty evidence rather
than leaving an old face behind.

Capture-derived items use their device timeline and navigation items use their
explicit host/tick timelines and frame transforms. Haar rank and rectangles
are not probability, identity, range, occupancy, collision, or navigation
evidence. Rerun does not currently publish the accessory health snapshot,
semantic expression source, frontier candidate set, or complete supervisor
state; those must not be inferred from the images. Rerun is not a safety
authority or complete decision ledger.

The pinned Rerun SDK is output-only. A click adapter therefore submits the same
typed `(map_epoch, revision, x_m, y_m)` command through the local control API;
the agent never pretends Rerun supplied a callback it does not provide.

The loopback operator console and agent API share that one typed ingress and
the same downstream request sequence. The browser opens a per-session
capability only after the operator supplies the mode-`0600` per-boot
capability; neither secret appears in a URL or browser storage. Arrow/WASD and
buttons stream admitted SI velocity intentions through a monotonic server
deadman. Blur, page hide, network loss, key release, and manual release reduce
manual authority toward an exact applied zero; opening a replacement session
does not itself revoke the current owner. Autonomous authority deliberately
continues across browser or client loss until completion, its configured
runtime bound, a safety fault, or an explicit global stop from any authenticated
session. Manual, map-only, frontier, and revision-bound point-goal requests use
the same arbitrator. A process-lifetime software safety stop has priority over
queued work and cannot be reset remotely, but it is explicitly not the
independent physical emergency stop.

The console renders only typed map metadata/cells, composed map-frame pose,
goal, global path, MPC rollout, requested actuation, exact STM32 applied
receipt, stop certainty, health, and timing that the live owners publish. Head
health is refreshed from complete four-joint transactions, eye health requires
at least one acknowledged RGB-derived expression after startup, and OAK health
does not become ready until visual, depth, and IMU inputs have each been
admitted. Ready is not an ever-seen latch: the projection requires recent
activity from all three streams and the coordinator's at-now odometry and
depth-aligned local-costmap freshness gates. A closed stream faults it. The
same typed coordinator readiness check rejects new manual, frontier, and point
authority from both the browser and agent API before any authority is granted;
periodic control retains its independent stricter stop-on-stale checks. The
console never opens the OAK or STM32 and never labels an accepted request as
applied.
The configured `100 mm/s` and `500 mrad/s` browser steps are requested
body-frame magnitudes inside the admitted manual envelope, not physical speed
measurements.

## Service shutdown guard

`kiko-nano-agent.service` uses `TimeoutStopSec=420` as an operational kill
guard, not as proof that graceful cleanup always completes. The parsed hard
maxima cover a 120-second Rerun flush, at most 150 seconds for an in-flight eye
intent followed by eye release/cleanup, and at most 1.2 seconds for coordinated
controller-task collection. Production head ownership release performs no
protocol write. These bounded phases total at most 271.2 seconds when treated
sequentially.

The remaining margin is not a derived guarantee. Inference, dense-map,
navigation, visualization, and control-socket thread joins; dataset
finalization, abort, and filesystem synchronization; OAK `Device::close()`; and
native or operating-system I/O have no common outer application deadline.
Reaching the systemd timeout can interrupt durable-state cleanup and cannot
prove controller stop, eye release, head torque state, or OAK closure. Do not
lower the guard or describe it as a graceful-shutdown bound until those phases
have typed deadlines and measured termination evidence.

## Cold-boot acceptance

The software acceptance harness must prove, with simulated transports and fault
injection, that:

1. exact inventory succeeds and every wrong/missing/rebooted identity fails;
2. the robot remains unarmed and at confirmed applied zero through startup;
3. the head adopts and verifies the present pose without a torque-disable gap;
   failed telemetry or approval gates issue no motion or cleanup torque write
   and never approach natural;
4. RGB motion produces an expiring eye intention and stale RGB falls back;
5. online SLAM produces a localized, checksummed occupancy artifact;
6. manual, explore, and point-goal authorities cannot overlap;
7. a selected map point reaches the planner/MPC and only an exact applied
   result permits the next command;
8. save, reload, dataset-bound relocalization, and continued mapping preserve
   the declared map/frame identities; and
9. camera loss, stale depth, localization loss, controller reset, serial loss,
   process cancellation, and clock faults all require or confirm zero.

Run the bounded offline component suite with:

```bash
tools/nano-cold-boot-fault-acceptance.sh
```

The exact case-to-contract mapping and the deliberately unproven physical
claims are recorded in `docs/nano-cold-boot-fault-acceptance.md`. A passing
software run is required evidence, but is not a Nano cold-power, hardware
watchdog, physical stop, camera, SLAM-accuracy, MPC-tracking, or performance
result.

Physical acceptance additionally requires the assembled devices to be visible
on the Nano and an independent emergency stop. The read-only inventory on
2026-07-20 found the OAK, STM32 ST-Link serial function, eye controller, and
head adapter on an NVIDIA Jetson Orin Nano. That establishes USB presence only:
no camera stream, STM32 control identity, KEP2 eye session, head hold, emergency
stop, motor motion, or drive result is claimed by this document. See
`nano-validation-evidence-2026-07-20.md` for the exact observations and native
aarch64 host-test evidence.
