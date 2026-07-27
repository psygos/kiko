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
3. keep an independent motor-power cut immediately reachable;
4. inspect the current owners of the OAK, STM32, head bus, and eye serial
   endpoints; and
5. coordinate a normal Fable/legacy-owner handoff.

Never use `killall`, `pkill`, or an unrelated process kill to obtain a device.
The qualifier is the sole OAK, accessory, in-process UDP, and STM32 owner only
after exact acquisition succeeds. An ownership conflict is a failed
qualification, not a reason to kill other workloads.

Follow the exact
[Fable handoff](nano-qualified-deployment.md#exact-fable-handoff) before
acquisition. The audited `makerspace` owner had two separate crontab entries
that preserved its guardian; killing only `kiko_face_follow.py` lets that
guardian respawn it in about eight seconds. Preserve all unrelated crontab
entries and processes, disable only the two identified owner entries, and
terminate only the exact guardian/child PIDs in the documented order.

The candidate contract deliberately says physical stop semantics are
`unverified`. The operator's reachable power cut remains the independent stop.
The one-way software safety stop, deadman, lease, watchdog, applied-zero
receipt, and disarm receipt are additional typed controls; none proves how the
uncharacterized motor driver stops.

## Build the qualification-only binary

Build the reviewed commit on Linux aarch64 with the lockfile:

```bash
cargo build --locked --release -p kiko-slam \
  --features nano-wheels-off-qualification \
  --bin kiko-slam
```

This feature includes the common live Nano stack and the qualification
surface. The raw-PWM qualification modules are compiled out of an ordinary
production-only `--features nano-agent` build.

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
face-cascade or head-gaze-policy binding. The current renderer requires
qualification input V4 and emits qualification launch V4. Production input V1
and launch V3 are unchanged.

Do not install a qualification systemd unit. Do not add qualification to the
production unit, qualified-boot drop-in, cron, Fable guardian, or any other
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
├── head-gaze-policy-v1.json
├── nano-wheels-off-qualification-launch-v4.json
├── navigation-shadow-v1.json
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
- reviewed OAK, occupancy, inference, Rerun, and storage resource limits.

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
a foreground attended command. If a Fable owner currently holds the
OAK/head/eye, stop it through its documented normal handoff and verify that it
exited before continuing.

Recheck:

- wheels physically removed;
- head physically supported;
- independent power cut reachable;
- motor area clear;
- correct immutable bundle selected; and
- operator and Nano terminal clocks/log destination identified.

## Start exactly once, in the foreground

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
device is opened, the process requires these exact separate terminal replies:

```text
WHEELS REMOVED
HEAD SUPPORTED
POWER CUT REACHABLE
```

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
preparation.

This admission proves the launch-bound executable and seven required
native-runtime files each have a file-backed executable mapping with those
exact device/inode identities. It does not parse ELF dynamic sections or prove
a hermetic transitive OS-library closure. Retain exact
`DT_NEEDED`/loader-graph evidence from the final ELF as a separate target-side
`readelf` release gate.

Immediately before the short raw-PWM window, the process requires one fresh
exact reply:

```text
WHEELS OFF HEAD SUPPORTED POWER CUT READY
```

That attestation may authorize nonzero candidate requests for at most 30
seconds from its monotonic creation. It is not a 30-second continuous-motion
request. A safely stopped mapping session may continue after expiry, but a new
nonzero request—or a retained nonzero target crossing the deadline—fails
closed. Restart and repeat the attended preflight for a new motion window.

## Open the unified human/agent console

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

Run separate fault sessions for STM32 reset, serial disconnect, OAK disconnect,
browser/tunnel loss, host termination, stale/deferred command, partial serial
record, and controller lease expiry. Keep the wheels removed. On every fault,
record whether stop certainty is exact or uncertain. If software stop is
uncertain, use the independent power cut and report uncertainty; never promote
that run.

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
