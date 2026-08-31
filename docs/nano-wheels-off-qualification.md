# Attended Nano wheels-off full-stack qualification

This runbook is for the separate
`nano-wheels-off-qualification` executable surface. It qualifies the host,
camera/SLAM/occupancy/Rerun graph, accessories, and provisional STM32 streaming
path while both drive wheels are physically removed. It is not production
admission and it does not produce wheel-ground calibration.

Do not attach the wheels for this runbook. Do not enable a service, create a
boot unit, or treat a successful process start as physical evidence.

## Safety and ownership boundary

One operator must remain beside Kiko for the entire motion window. Before
starting:

1. physically remove both drive wheels;
2. support the head so a loss of torque cannot let it fall;
3. physically disconnect the motor output power supply while leaving only the
   controller logic/serial path available for stopped-device qualification;
4. keep an independent motor-power cut immediately reachable;
5. inspect the current owners of the OAK, STM32, head bus, and eye serial
   endpoints; and
6. refuse to start while any competing device owner or automatic launcher is
   present.

Never use `killall`, `pkill`, or an unrelated process kill to obtain a device.
The qualifier is the sole OAK, accessory, in-process UDP, and STM32 owner only
after exact acquisition succeeds. An ownership conflict is a failed
qualification, not a reason to kill other workloads.

Before acquisition, perform a fresh endpoint-by-endpoint owner check and
follow the exact
[exclusive-endpoint acquisition](nano-qualified-deployment.md#exact-exclusive-endpoint-acquisition).
If no conflicting owner exists, start exactly one canonical Kiko owner after
proving the endpoints free. A historical
`makerspace` snapshot found two separate crontab entries preserving one
guardian, but that observation is not current authority to edit either entry.
Preserve all unrelated crontab entries and processes. If a current owner or
respawn authority is found, retain the conflict and stop this procedure; do
not disable, signal, or kill it here.

The candidate contract deliberately says physical stop semantics are
`unverified`. The operator's reachable power cut remains the independent stop.
The one-way software safety stop, deadman, lease, watchdog, applied-zero
receipt, and disarm receipt are additional typed controls; none proves how the
uncharacterized motor driver stops.

## Build the qualification-only binary

Build the reviewed commit on Linux aarch64 with the lockfile:

```bash
cargo build --locked --release -p kiko-slam \
  --no-default-features \
  --features nano-wheels-off-qualification \
  --bin kiko-slam
```

This feature includes the common live Nano stack and the qualification
surface. The raw-PWM qualification modules are compiled out of an ordinary
production-only `--features nano-agent` build.

Canonical production and wheels-off startup use a fixed geometric/worker
tracker policy rather than ambient keyframe, bundle-adjustment, loop,
descriptor-worker, culling, or transition-trace overrides. Loop closure and
relocalization use deterministic descriptors aggregated from the admitted
SuperPoint features, so these paths do not open or require an EigenPlaces model
at startup. This makes the graph runnable without an unbound model path; it
does not prove place-recognition quality equivalent to EigenPlaces. Retain
representative-room loop-closure and relocalization observations as physical
evidence before promoting either behavior. The offline compatibility command
retains its documented environment-driven learned-descriptor boundary.
ONNX Runtime session tuning remains a separate compatibility boundary through
the documented `KIKO_ORT_*` variables; this gate does not infer an
environment-free inference runtime.

The exact resulting executable is a mandatory renderer input. Start from
`configs/nano-wheels-off-qualification-template/bundle-render-input-v4.json.template`.
The prepared qualification render-input boundary includes:

```json
{
  "schema_version": 4,
  "bundle": {
    "kind": "wheels_off_qualification",
    "qualification_executable_path": "/absolute/path/to/target/release/kiko-slam"
  }
}
```

Qualification render-input/launch V1 through V3 were already published. Do
not relabel an old document: V2 incorrectly selected the system ABI name
`libusb-1.0.so.0` for the pinned DepthAI libusb role, and V3 had no exact
face-cascade or optional head-gaze-policy binding. The current renderer
requires qualification input V4 and emits qualification launch V4. Production
now uses its separate input V2 and launch V4, which mandate an attended-review
bound physical head-gaze policy; that does not change this proposal-only
qualification lane.

Gate A omits `assets.head_gaze_policy_source_path`. Consequently its bundle
contains no head-gaze policy file or hash, bootstrap returns no policy, and no
gaze adapter is claimed. This does not disable the separately reviewed natural
head hold. The current return-and-hold target, startup envelope, exact software
travel caps, retained torque limits, and remaining physical claim boundary are
recorded in `docs/nano-head-neutral-policy-2026-07-29.md`; older return evidence
must not be substituted for that superseding policy.

For later proposal-only qualification work, start from
`configs/nano-wheels-off-qualification-template/head-gaze-policy-v1.json.template`.
It fixes the declared camera/head geometry, parallel neutral axes, assumed
`1.5 m` gaze plane, and natural encoder declaration, but deliberately leaves
every unmeasured mapping and controller value as an `UNVALIDATED` sentinel.
It is a non-deployable template, not calibration evidence. If supplied, the
policy must parse as `proposal_only` before bootstrap opens hardware; it
cannot authorize torque or head motion.

Do not install a qualification systemd unit. Do not add qualification to the
production unit, qualified-boot drop-in, cron, a legacy guardian, or any other
automatic startup path.

Never replace `/opt/kiko/bin/kiko-slam` with this feature-expanded binary.
The renderer retains it as
`/opt/kiko/qualification/bin/kiko-nano-wheels-off-qualification`; production
and qualification executable bytes therefore have independent paths and
hashes. Updating either one requires rerendering its exact bundle.

`/run/kiko` is created explicitly because this attended command has no systemd
`RuntimeDirectory` owner. Before reuse, inspect that exact directory for a
stale socket or capability. If either exists, first prove its owning process
has exited and complete that owner's documented cleanup; do not unlink a live
owner's endpoint.

## Render and review the immutable bundle

Start from
`configs/nano-wheels-off-qualification-template`. Render into a separate
staging root. The canonical installed layout is:

```text
/opt/kiko/qualification/
├── bin/
│   └── kiko-nano-wheels-off-qualification
├── agent-policy-v3.json
├── candidate-controller-policy-v1.json
├── controller-server-candidate-v2.json
├── device-inventory-candidate-v2.json
├── nano-wheels-off-qualification-launch-v4.json
├── navigation-shadow-v2.json
├── native-runtime-v1.json
├── artifacts/
│   ├── calibration/<exact calibration asset>
│   └── plant/<exact shadow-only plant asset>
├── evidence/
│   ├── render-input-v4.json
│   └── render-evidence-v1.json
├── lib/
│   ├── libdepthai-core.so
│   ├── libdynamic_calibration.so
│   ├── libusb-1.0.so
│   ├── libonnxruntime.so.1
│   ├── libopencv_core.so.4.5d
│   ├── libopencv_imgproc.so.4.5d
│   └── libopencv_objdetect.so.4.5d
└── models/
│   ├── opencv/
│   │   ├── haarcascade_frontalface_default.xml
│   │   └── haarcascade_profileface.xml
│   └── <exact SuperPoint and LightGlue model paths>
```

`head-gaze-policy-v1.json` appears only in a later bundle whose render input
explicitly supplies a complete proposal-only policy source. Its absence is the
typed Gate A state.

The deployment tool must discover and render the following instead of
guessing them:

- exact persistent `/dev/serial/by-id/...` paths;
- STM32 UID in both 24-lowercase-hex and 12-decimal-byte forms;
- exact OAK MXID and compiled DepthAI header provenance;
- exact head and eye identities;
- exact canonical calibration/plant IDs and paths; the calibration asset owns
  one OAK MXID, rectified stereo intrinsics/dimensions/baseline, raw IMU
  calibration, tracking-camera-to-base transform, and three later production
  approval IDs;
- exact 32-byte calibration and plant SHA arrays;
- exact byte counts and lowercase SHA-256 values for every file in the
  renderer's typed bundle plan, including the qualification executable, the
  seven required native-runtime roles, and `native-runtime-v1.json`; and
- the source files for the fixed SuperPoint, LightGlue, face-cascade, and
  native-library destinations.

OAK stream geometry, occupancy, inference, Rerun, storage, head-return, and
RGB-expression policy are fixed by the checked-in V4 render-input template.
Changing them is a reviewed source change with tests, not a deployment-time
substitution.

The current physical connection assumptions may seed device discovery, but
they are not evidence. The rendered server, inventory, and live acquisition
must still agree exactly. Qualification bootstrap parses the retained
calibration once before any base owner is acquired, then requires its
canonical MXID and stereo model to match the exact opened OAK and its raw IMU
and tracking-camera-to-base values to match the parsed navigation
configuration. Production separately cross-binds the artifact's three
calibration IDs to the physical-actuation approval.

Render the leaf assets and documents first. Render
`nano-wheels-off-qualification-launch-v4.json` last. On the Nano, inspect the
result before installation:

```bash
find STAGING_ROOT -type f -name '*.json' -print0 |
  xargs -0 -n1 jq -e .
if rg -n '\$\{' STAGING_ROOT; then
  echo 'unresolved deployment token' >&2
  exit 1
fi
find STAGING_ROOT -type f -printf '%s %p\n' | sort -k2
find STAGING_ROOT -type f -print0 |
  sort -z |
  xargs -0 sha256sum
```

`STAGING_ROOT` above is an instruction placeholder, not a literal path. The
reviewed deployment tool should retain its render-input record and the final
hash listing as evidence. The renderer sets `maximum_bytes` to the exact
installed byte count. A smaller value cannot admit that file, and a larger
ceiling must never be copied from a sample artifact.

Publish only after review, with the qualifier stopped. The destination must be
new and empty; never merge a staging bundle over old bytes. On an update,
archive the stopped old root, create a fresh `/opt/kiko/qualification`, and
then run:

```bash
sudo install -d -o root -g root -m 0755 /opt/kiko/qualification
sudo cp -a STAGING_ROOT/. /opt/kiko/qualification/
sudo chown -R root:root /opt/kiko/qualification
sudo find /opt/kiko/qualification -type d -exec chmod 0755 '{}' +
sudo find /opt/kiko/qualification -type f \
  ! -path /opt/kiko/qualification/bin/kiko-nano-wheels-off-qualification \
  -exec chmod 0444 '{}' +
sudo chmod 0555 \
  /opt/kiko/qualification/bin/kiko-nano-wheels-off-qualification
sudo install -d -o root -g root -m 0750 /var/lib/kiko-nano-qualification
sudo install -d -o root -g root -m 0700 /run/kiko
```

Do not edit the installed tree in place. A changed byte requires a new staging
render and a new launch hash.

## Stationary integrated lab mode

Use `nano-stationary-lab` when the immediate task is integrated expression,
camera, SLAM, occupancy, Rerun, or console bring-up and no base-motion test is
needed. This mode deliberately has no physical-attestation dialogue. It uses
the same launch-bound device identities and native assets, establishes the
STM32's exact applied zero and disarm receipts, then latches the console's
process-lifetime software safety stop before the first runtime tick. Its
motion-attestation gate starts terminal and contains no worker or token, so no
HTTP, agent, flag, or later readiness transition can enable candidate PWM.

Run it in the foreground:

```bash
sudo /usr/bin/env LD_LIBRARY_PATH=/opt/kiko/qualification/lib \
  /opt/kiko/qualification/bin/kiko-nano-wheels-off-qualification \
  nano-stationary-lab \
  --deployment-root /opt/kiko/qualification \
  --launch-config nano-wheels-off-qualification-launch-v4.json \
  --state-root /var/lib/kiko-nano-stationary-lab
```

The mode still owns and supervises the exact STM32 serial endpoint because an
applied zero/disarm receipt and exclusive ownership are useful integration
evidence. It does not reconnect motor power, issue a nonzero command, run a
qualification fault injection, or ask for a post-run motor-power statement.
It is not Gate-A motion evidence, physical stop evidence, plant calibration,
or production authority. An attended motion or wheel-on run remains a
separate later operation.

## Preflight

Confirm no automatic or standalone motor/camera owner is active. Inspect
specific units and processes; do not broadly kill anything:

```bash
systemctl is-active kiko-nano-agent.service
systemctl is-active kiko-robot-server.service
systemctl is-enabled kiko-nano-agent.service
```

Production and any legacy standalone controller service must be inactive. The
repository no longer ships the standalone service; `unknown` is therefore the
expected result on a clean installation. The qualification process must remain
a foreground attended command. If any other owner or automatic launcher can
hold the OAK, head, or eye endpoint, do not start qualification. Retain the
conflict and resolve that workload separately, then repeat the read-only owner
check.

Recheck:

- wheels physically removed;
- head physically supported;
- motor output power physically disconnected while controller logic/serial
  remains available;
- independent power cut reachable;
- motor area clear;
- correct immutable bundle selected; and
- operator and Nano terminal clocks/log destination identified.

## Start exactly once, in the foreground

Before starting the foreground process, establish the loopback SSH forwards
shown in the next section. A forward can wait while the Nano port is not yet
listening; do not expose either listener on the LAN.

Run from an attended TTY:

```bash
sudo /usr/bin/env LD_LIBRARY_PATH=/opt/kiko/qualification/lib \
  /opt/kiko/qualification/bin/kiko-nano-wheels-off-qualification \
  nano-wheels-off-qualification \
  --deployment-root /opt/kiko/qualification \
  --launch-config nano-wheels-off-qualification-launch-v4.json \
  --state-root /var/lib/kiko-nano-qualification
```

There are no physical-attestation flags or environment aliases. Before any
device is opened, the process generates a fresh 128-bit challenge for each
claim and requires the exact displayed phrase plus its 32-lowercase-hex
challenge:

```text
WHEELS REMOVED <32-lowercase-hex-challenge>
HEAD SUPPORTED <32-lowercase-hex-challenge>
MOTOR POWER PHYSICALLY DISCONNECTED <32-lowercase-hex-challenge>
POWER CUT REACHABLE <32-lowercase-hex-challenge>
```

Each challenge is generated only when its physical boundary is reached.
Prequeued, pasted-ahead, static, or replayed replies cannot satisfy a later
claim without predicting its fresh 128-bit challenge.

Before opening a device, the Linux-only process proves that `/proc/self/exe`
is the exact launch-bound executable by byte identity and filesystem
device/inode, parses the launch-bound native-runtime manifest, requires all
seven native-runtime roles, stream-verifies every exact executable/library
identity without retaining a second executable or native-library copy,
initializes ONNX Runtime through its launch-bound exact path, and requires the
executable plus all seven expected native-library device/inode identities each
to have a file-backed executable mapping in `/proc/self/maps`.
It also parses and cross-binds the canonical calibration, plant, and navigation
artifacts to the candidate controller. The admitted MPC is limited to at most
30% absolute PWM and 5% PWM slew per step, and its control/service interval
must not exceed 54,999,999 ns. These checks finish before device enumeration,
serial probes, or OAK connection. A plant declaring `synthetic_fixture`
evidence remains explicitly synthetic and is not physical-identification
evidence. The process then probes exact
OAK/head/eye identity, requires OAK SuperSpeed readback, starts and acquires
the exact candidate STM32 session, observes exact inventory, applies zero,
confirms zero, disarms, prepares storage/models/SLAM, and starts the natural
head hold. The candidate controller remains stopped through fallible
preparation. Motor output power must remain physically disconnected throughout
all of those steps. The software cannot measure that physical supply state;
the initial exact reply is an operator claim, and the later readiness reply
reconfirms it before reconnection is permitted.

This admission proves the launch-bound executable and seven required
native-runtime files each have a file-backed executable mapping with those
exact device/inode identities. It does not parse ELF dynamic sections or prove
a hermetic transitive OS-library closure. Retain exact
`DT_NEEDED`/loader-graph evidence from the final ELF as a separate target-side
`readelf` release gate.

After the fallible setup completes, the process creates the exactly stopped
runtime owner and binds the loopback console with its manual-motion boundary
still disabled. OAK capture, online SLAM, occupancy, Rerun, accessories,
console telemetry, MPC shadow decisions, and the exact applied-zero/disarm
STM32 evidence can run while nonzero authority is structurally locked and
motor output power remains physically disconnected. This does not relabel the
motion-capable candidate firmware as motor-inert. The transition worker remains
absent until one stopped control tick has fresh admitted visual, depth, and IMU
observations; a current ready head/eye/RGB-expression health observation; one
admitted occupancy revision published to the console; successful coordinator
`motion_start_readiness_at` evidence; and acceptance of that tick's navigation
diagnostic by the bounded visualization queue. Queue acceptance does not claim
that Rerun has consumed, displayed, or persisted the message.

While the console reports `motion-attestation-pending`, read the newly created
capability in a separate authenticated Nano shell, open the console through
the already-established tunnel, and confirm that every motion-capable request
is rejected. The reduction-only ordinary `stop` and one-way software safety
stop remain available. The worker discards pending TTY input before generating
each fresh challenge. It first requires this through-setup reply:

```text
MOTOR POWER REMAINED PHYSICALLY DISCONNECTED THROUGH SETUP <32-lowercase-hex-challenge>
```

Only after accepting that reply does the terminal instruct the operator to
physically reconnect motor power. With both wheels still removed, the head
still supported, and the independent cut still immediately reachable, it then
requires:

```text
MOTOR POWER RECONNECTED WHEELS OFF HEAD SUPPORTED POWER CUT READY <32-lowercase-hex-challenge>
```

Only after that reply does the existing stopped runtime recheck all integrated
readiness evidence, the process-running authority, and the console safety
boundary at the enable linearization. Readiness loss during the operator delay
closes the one-shot gate and requires a process restart. A software safety
stop, frontend loss, runtime receiver loss, process shutdown, controller-owner
exit, or propagated accessory failure cancels and joins the TTY worker without
enabling motion. Cancellation is rechecked after terminal input is discarded,
after the fresh challenge is generated, immediately before prompt output, and
while input is polled. Under a concurrent stop, the hard guarantee is that
no authority is enabled when the stop wins the console/process linearization;
the documentation does not claim that racing prompt bytes can never already
have reached the terminal. If enablement linearizes first, no queued command
can be consumed until a later owner tick whose process-running check passes,
and a subsequently linearized software safety stop preempts candidate motion.
The attestation may authorize
nonzero candidate requests for at most 30 seconds from its monotonic creation;
read-only startup and mapping do not consume that window. It is not a
30-second continuous-motion request. A safely stopped mapping session may
continue after expiry, but a new nonzero request—or a retained nonzero target
crossing the deadline—fails closed. Restart and repeat the attended preflight
for a new motion window.

Treat the pending prompt as a short, continuously attended transition check,
not an unattended pause. Sensor/navigation processing continues while motion
authority remains unrepresentable. If console inspection is delayed or the
frontend exits, do not attest into an aged startup: stop and restart the
qualification.

Sessions opened while motion attestation is pending are stamped with that
pending authority generation. They remain authenticated for the one-way
software safety stop and the reduction-only ordinary `stop`; neither can create
or retain nonzero authority. Every motion-capable intent remains
generation-bound, so no prebuilt motion request or idempotency key can cross
enablement. Browser and agent clients must open a fresh session after
enablement. The browser keeps the old stop-capable session until its
replacement exists, atomically switches to the replacement, and only then
retires the old session. It clears every held key/pointer state before that
handshake, so motion requires a later physical release and new input edge.

## Use the unified human/agent console

The qualifier listens only on Nano loopback at `127.0.0.1:9877`. From the
operator computer, keep this tunnel open:

```bash
ssh -N \
  -L 9877:127.0.0.1:9877 \
  -L 9876:127.0.0.1:9876 \
  makerspace@NANO_IP
```

Port `9876` is the launch-bound Rerun gRPC stream. Connect a compatible local
Rerun viewer through that forwarded port; do not expose the Nano listener. The
console displays
`rerun --connect rerun+http://127.0.0.1:9876/proxy` only when the admitted
launch graph configures the loopback Rerun server. This is the operator side of
the same-port SSH forward, not a Nano network address. Its presence describes
configuration, not diagnostic-worker health or control authority.

In a separate authenticated Nano shell, read the per-process capability:

```bash
sudo cat /run/kiko/operator-console.capability
```

Open exactly:

```text
http://127.0.0.1:9877/
```

Paste the capability into the console. Never place it in a URL, shell command,
log, screenshot, issue, or checked-in file. The capability is created for this
process and removed only after the console owner shuts down.

The browser and an agent use the same authority/session arbiter. The
qualification API is:

```text
POST /api/v1/wheels-off-qualification/intents
```

API clients authenticate with `x-kiko-console-capability`, open a session,
then use the returned session ID and `x-kiko-session-capability`. The ordinary
production velocity-intent endpoint is not a raw-PWM alias and must not be used
for this session.

The nested `wheels_off_qualification` snapshot projection is schema V2. V2
adds the required frontend lifecycle, attended-motion-authority, and stop-latch
state. Clients must reject retired V1 snapshots rather than deriving readiness
without those fields. Intent submissions remain the separate request schema
V1.

The UI must show the qualification banner, raw left/right timer-duty requests,
last exact applied receipt, stop/disarm state, runtime health, pose, live
occupancy grid, path, and shadow MPC state. Use arrow/WASD only while observing
the lifted drivetrain. Release, focus loss, tab hiding, tunnel loss, session
close, deadman expiry, or the UI software safety stop must request terminal
zero. The independent physical E-stop/power cut remains separate. Do not
continue if the UI lacks fresh applied-receipt or stop-barrier evidence.

The fixed manual request magnitude is exactly 10% raw timer duty; the absolute
candidate cap is 30%. Firmware slew is bounded separately. Neither number is a
velocity, wheel-direction calibration, torque claim, or permission to attach
the wheels.

## Qualification sequence

With the wheels removed and the power cut in hand:

1. Verify the page reports the exact candidate profile and an applied zero.
2. Verify OAK RGB, rectified stereo, depth, IMU, live pose, occupancy, and
   Rerun update without opening another camera process.
3. Verify the head reaches and continuously holds the reviewed natural pose
   and the RGB-driven eye behavior updates.
4. Hold each manual direction briefly, one at a time. Observe motor direction
   only; do not infer ground velocity.
5. Release each control and require an exact applied-zero/stop-barrier update.
6. Exercise the UI software safety stop and confirm it latches against later
   motion; separately keep the physical E-stop/power cut reachable.
7. End the process normally and confirm the exact disarm receipt and capability
   cleanup.

After the software returns from its cleanup path on every success or failure
following the initial preflight, physically disconnect motor power. This
requirement applies whether a controller owner started and whether its cleanup
proved an exact stop. The foreground process requires this final exact terminal
fresh challenged reply:

```text
MOTOR POWER PHYSICALLY DISCONNECTED <32-lowercase-hex-challenge>
```

If the terminal is lost, the disconnect reply is wrong, or controller stop is
uncertain, the run cannot succeed. Use the independent physical cut first,
retain the uncertainty, and do not leave motor power connected merely to
obtain another software receipt.

Run exactly one fault in each fresh qualification process. Retain the process
ID, controller boot/session identity, last applied receipt, terminal stop or
disarm receipt, and whether output certainty is exact or uncertain. A stopped
process must not be reused for another fault. Keep the wheels removed and the
independent power cut reachable throughout. If software stop is uncertain, cut
motor power and retain that uncertainty; never promote the run.

The minimum separate-session matrix is:

- close the browser or SSH tunnel while a brief request is being refreshed,
  then require the server-side deadman to produce a journaled exact zero;
- let one request expire without refresh, then require the same terminal-zero
  evidence;
- replay a source sequence or idempotency key and require rejection without a
  new nonzero applied step;
- stop refreshing the controller lease, without killing its owner, and require
  the firmware lease/watchdog stop before clean host recovery;
- reset the STM32 and, in a different session, physically disconnect its
  serial link; treat either result as uncertain unless an exact zero from the
  same reacquired boot/session can be proved;
- physically disconnect the OAK for the camera-loss test and require the host
  safety latch and terminal stop;
- send `SIGTERM` to the exact foreground qualifier PID, never a broad process
  signal, and require normal cleanup, exact stop/disarm evidence, and capability
  removal;
- after a proved stop and disarm with motor power cut, cold-restart the
  controller and qualifier and require a new boot/session that begins at zero
  with no stale authority; and
- run each qualifier-only deterministic seam below in its own fresh process.

Four host faults have closed, qualifier-only declarations. Add exactly one of
these arguments to the normal `nano-wheels-off-qualification` invocation:

```text
--fault-injection host-monotonic-clock-regression-on-first-nonzero-command
--fault-injection partial-uart-record-on-first-nonzero-command
--fault-injection stale-depth-on-first-nonzero-command
--fault-injection localization-loss-on-first-nonzero-command
```

Run them separately. The clock declaration arms only after bootstrap zero and
reacquisition; the first nonzero candidate request then makes the command
client observe one strict regression. It must latch before transmitting that
nonzero command and retain the exact `HostStop` recovery result. The UART
declaration lets bootstrap and reacquisition zero pass normally, writes exactly
one checked non-delimiter byte of the first nonzero `ApplyPwm` record, writes a
delimiter, issues `ForceStop`, and terminates the controller owner with the
typed resynchronization and stop outcome. A controller serial-integrity fault
is retained, not rewritten as a clean run. The stale-depth declaration latches
depth stale only after a controller-confirmed nonzero applied step, prevents
later depth observations from making the injected navigation state fresh, and
queues the existing terminal stop. The localization-loss declaration first
requires established localization at that confirmed step, then latches
localization lost and queues the same terminal stop; if localization was never
established, the seam fails closed instead of fabricating a loss.

The stale-depth and localization-loss declarations are synthetic software
seams inside a live hardware qualification process. They do not prove a
physical OAK disconnect, frame loss, or real localization failure. Retain the
selected and triggered declaration with the applied receipt and stop evidence,
and run the physical OAK-disconnect session separately.

There is no free-form regression size, prefix length, repeat count, or combined
mode. Unknown declarations fail command-line parsing. The production
`nano-agent` subcommand has no `--fault-injection` argument, and an ordinary
qualification run omits the injection state. A selected declaration that
reaches normal teardown without exercising its exact path is a typed failed
session, not a clean run. These host tests prove
deterministic fault routing; only a later attended wheels-off session can
provide controller and output evidence from the physical link.

Live SLAM/occupancy/Rerun and shadow MPC are observational in this lane. No MPC
output, point goal, frontier exploration, or body-frame velocity command is
converted to candidate PWM.

## Evidence and exit gate

Retain:

- Git commit and Linux aarch64 binary SHA-256;
- rendered input record plus exact byte count/SHA-256 for every installed
  asset;
- exact device identities, controller boot/session identity, and OAK
  SuperSpeed readback;
- initial zero and disarm receipts;
- each requested raw-PWM pair and exact applied receipt;
- deadman, release, software-safety-stop, disconnect, reset, and shutdown stop
  evidence, plus the separately observed physical power-cut/E-stop check;
- head final/held telemetry and eye identity/behavior evidence;
- map/pose/occupancy/Rerun observations; and
- every missing or uncertain result, explicitly labeled.

A wheels-off run can qualify this provisional streaming and software path. It
cannot qualify PWM-to-velocity, motor signs under ground load, traction,
braking, stopping distance, plant parameters, or MPC-driven navigation. Those
remain wheel-attached calibration and controlled-drive work after this gate is
reviewed.
