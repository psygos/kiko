# Nano wheels-off bench bring-up

This is the fastest supported path to a physical Kiko camera/head/eye check.
It is deliberately a non-driving gate. The wheels must be physically removed
and the robot supported before either service is started.

Passing this gate means the exact configured STM32 repeatedly admitted zero,
the exact OAK produced two ordered startup samples from RGB, depth, and IMU and
continued producing RGB within a bounded liveness budget, the head startup
transaction admitted an already-safe observed pose inside reviewed tick
windows, and the eye firmware admitted RGB-derived expressions. It does not
validate SLAM, wheel
calibration, MPC, navigation, camera optics, depth accuracy, IMU calibration,
visible eye photons, thermal behavior, or any nonzero motor command.

## What “ready normal position” means

The bench never seeks a named neutral pose and never discovers servo geometry.
With head torque off, an operator first places the head in a known safe natural
pose. The head actor then:

1. attempts torque-disable writes for all four joints and requires all four host
   writes to complete before issuing a read request;
2. reads each exact servo twice;
3. rejects disagreement, stale telemetry, and any pose outside the configured
   windows; and
4. only then writes that observed goal, bounded speed/torque, and torque enable,
   followed by a startup readback.

The array order is always `bow, curl, yaw, roll`. Those windows are deployment
acceptance limits, not calibration evidence. Do not copy historical demo ticks
after an EEPROM-offset, servo, gear, sign, or yaw-scale change. Do not use
`[0, 0, 0, 0]..=[4095, 4095, 4095, 4095]` merely to make the gate pass. The
parser rejects every per-joint window wider than 256 raw ticks, so a full-domain
window is not representable.

The pre-observation disable is host-write completion evidence, not an STS
register acknowledgement or independent physical torque-off proof. The rest is
also startup-only host-write and readback evidence. The current bench does not
continuously poll head position or torque state and therefore does not claim
that the pose remains centered for the full run.

The repository template intentionally has descending bounds. It therefore
fails during typed configuration parsing, before any device I/O, until a human
replaces them with narrow reviewed bounds.

## Ownership and launch boundary

There are two processes:

- `robot-server` exclusively owns the STM32 stable serial path and exposes the
  unauthenticated KRP2 V2 command socket only on `127.0.0.1:8080`.
- `kiko-nano-wheels-off-bench` owns the exact OAK, head adapter, and eye serial
  path. It contains only zero-only base authority; it has no nonzero motion API.

The systemd units are
`deploy/systemd/kiko-robot-server.service` and
`deploy/systemd/kiko-nano-wheels-off-bench.service`. The bench unit has both
`Requires=` and `After=` on the serial owner. It has no `[Install]` section and
`Restart=no`, so ordinary `systemctl enable` cannot make the physical bench
start at boot. The supported systemd launch is a deliberate manual operation
and enforces fresh root-owned attestations. Invoking the installed bench binary
directly only checks freely supplied operator-claim flags; it does not prove
that the root helper ran and is therefore an unsupported physical launch path.

Both units run as the observed Nano account `makerspace`. The robot-server unit
retains `dialout`; the bench also retains `video`, `render`, and `plugdev`.
They intentionally do not enable `PrivateDevices`, because the exact serial
nodes and OAK USB device must remain visible. They do retain filesystem,
privilege, kernel, and address-family hardening. If the deployed account or
udev groups differ, edit and re-verify the units explicitly; do not run either
process as root as a shortcut.

## Deployment-shaped template

`configs/nano-wheels-off-example/` mirrors `/opt/kiko/deployment`:

```text
nano-wheels-off-bench-v1.json
agent-policy-v1.json
nano-zero-only-v1.json
controller-server-v1.json
device-inventory-v1.json
artifacts/calibration/camera-placeholder.json
artifacts/plant/drive-placeholder.json
```

The bench document resolves its three policy assets relative to the deployment
root. The agent policy uses exact absolute paths for the inventory and artifact
root. The server unit and bench document both select the same
`controller-server-v1.json`; the server bind, inventory endpoint, zero-only
endpoint, and bench bind must all equal `127.0.0.1:8080`.

The template is structurally shaped like a deployment but is intentionally not
launchable. Values containing `REPLACE`, the `DEAD...` OAK MXID, `DE`/`AD`
identity bytes, descending pose bounds, and both placeholder artifacts are not
physical facts. Their matching example hashes demonstrate content binding
only. Replace every one from qualified assembly evidence. In particular,
`both_low`, `coast_verified`, PWM frequency, watchdog timing, torque limits,
and camera/drive artifacts are review-required hardware claims even though
their syntax parses.

After replacing an artifact, update its manifest digest. This read-only helper
prints the exact decimal byte array expected by the manifest:

```bash
python3 -c 'import hashlib,json,sys; print(json.dumps(list(hashlib.sha256(open(sys.argv[1], "rb").read()).digest())))' /opt/kiko/deployment/artifacts/calibration/qualified-camera.json
```

## Build and install on the Nano

Build the exact checked-out commit on aarch64 with the lockfile:

```bash
cargo build --locked --release -p robot-server --bin robot-server --bin v2_identity_probe
cargo build --locked --release -p kiko-slam --features nano-bench --bin kiko-nano-wheels-off-bench
cargo build --locked --release -p kiko-wheels-off-attest
```

Install immutable launch assets outside the writable checkout:

```bash
sudo install -d -o root -g root -m 0755 /opt/kiko/bin
sudo install -o root -g root -m 0755 target/release/robot-server /opt/kiko/bin/robot-server
sudo install -o root -g root -m 0755 target/release/kiko-nano-wheels-off-bench /opt/kiko/bin/kiko-nano-wheels-off-bench
sudo install -o root -g root -m 0755 target/release/kiko-wheels-off-attest /opt/kiko/bin/kiko-wheels-off-attest
sudo install -d -o root -g root -m 0755 /opt/kiko/deployment/artifacts/calibration
sudo install -d -o root -g root -m 0755 /opt/kiko/deployment/artifacts/plant
sudo install -o root -g root -m 0444 configs/nano-wheels-off-example/nano-wheels-off-bench-v1.json /opt/kiko/deployment/nano-wheels-off-bench-v1.json
sudo install -o root -g root -m 0444 configs/nano-wheels-off-example/agent-policy-v1.json /opt/kiko/deployment/agent-policy-v1.json
sudo install -o root -g root -m 0444 configs/nano-wheels-off-example/nano-zero-only-v1.json /opt/kiko/deployment/nano-zero-only-v1.json
sudo install -o root -g root -m 0444 configs/nano-wheels-off-example/controller-server-v1.json /opt/kiko/deployment/controller-server-v1.json
sudo install -o root -g root -m 0444 configs/nano-wheels-off-example/device-inventory-v1.json /opt/kiko/deployment/device-inventory-v1.json
sudo install -o root -g root -m 0444 configs/nano-wheels-off-example/artifacts/calibration/camera-placeholder.json /opt/kiko/deployment/artifacts/calibration/camera-placeholder.json
sudo install -o root -g root -m 0444 configs/nano-wheels-off-example/artifacts/plant/drive-placeholder.json /opt/kiko/deployment/artifacts/plant/drive-placeholder.json
sudo install -o root -g root -m 0644 deploy/systemd/kiko-robot-server.service /etc/systemd/system/kiko-robot-server.service
sudo install -o root -g root -m 0644 deploy/systemd/kiko-nano-wheels-off-bench.service /etc/systemd/system/kiko-nano-wheels-off-bench.service
sudo systemctl daemon-reload
sudo systemd-analyze verify /etc/systemd/system/kiko-robot-server.service /etc/systemd/system/kiko-nano-wheels-off-bench.service
```

The commands above install the fail-closed template. Qualify and replace it
before starting either service. Never edit `/opt/kiko/deployment` while a
service is running.

## First read-only SSH discovery

Once the operator confirms the wheels are off, the first remote session should
only inspect host and device presence:

```bash
ssh makerspace@192.168.50.2
uname -m
cat /proc/sys/kernel/random/boot_id
id makerspace
find /dev/serial/by-id -maxdepth 1 -type l -print -exec readlink -f {} \;
lsusb
systemctl --no-pager --full status kiko-robot-server.service kiko-nano-wheels-off-bench.service
```

No service should already be running. `makerspace` must be able to read and
write each selected stable serial path through its groups. Before generating
the controller contract, byte-read one KRP2 `ControllerHello` from the
exact STM32 path while `robot-server` remains stopped:

```bash
cargo run --locked -p robot-server --bin v2_identity_probe -- --serial-device /dev/serial/by-id/usb-STMicroelectronics_STM32_STLink_066EFF313946303143221230-if02 --timeout-ms 5000
```

The probe acquires that one serial path exclusively, transmits no protocol
bytes, uses the canonical bounded KRP2 decoder, and prints JSON containing the reported
UID, boot ID, firmware ABI/build, actuator fingerprint, capabilities, output
state, PWM/watchdog/neutral claims, and stop semantics. It is byte-read-only,
not electrically passive: opening a USB/UART endpoint can affect adapter or
line state. Copy only fields that
belong in the controller/inventory contracts. The output is a decoded firmware
claim, not physical proof of wiring, neutral levels, braking/coasting, or motor
behavior. A non-safe reported output state is an immediate power-cut condition.
The reported `maximum_command_lease_ms` must also cover the zero-only policy's
`zero_acquisition_lease_ms` (250 ms); do not shorten or extend either by guess.

Discover the OAK MXID without opening an actuator:

```bash
cargo run --locked -p oak-sys --bin list_devices
```

USB presence, a serial symlink, a byte-read controller hello, or an OAK MXID is
identity input, not calibration or physical-behavior proof. Eye
UID/build/capabilities, camera provenance, unreported STM32 timing/error
bounds, and artifact identities must come from the exact flashed
build/deployment evidence and then be challenged by the runtime. If any value
is unavailable, stop and report it as unknown.

## Preflight after replacing the template

With both wheels still removed and power-cut access in reach:

```bash
jq empty /opt/kiko/deployment/nano-wheels-off-bench-v1.json /opt/kiko/deployment/agent-policy-v1.json /opt/kiko/deployment/nano-zero-only-v1.json /opt/kiko/deployment/controller-server-v1.json /opt/kiko/deployment/device-inventory-v1.json
grep -R -n -E 'REPLACE|DEADDEAD|placeholder|template_only|replacement_required' /opt/kiko/deployment
find /opt/kiko/deployment -type l -print
jq -e '.ready_pose.minimum_ticks as $lo | .ready_pose.maximum_ticks as $hi | [range(0;4) as $i | ($lo[$i] <= $hi[$i] and ($hi[$i] - $lo[$i]) <= 256)] | all' /opt/kiko/deployment/nano-wheels-off-bench-v1.json
sudo -u makerspace test -r /opt/kiko/deployment/nano-wheels-off-bench-v1.json
sudo -u makerspace test -r /opt/kiko/deployment/controller-server-v1.json
```

The `grep` and `find` commands must produce no output, and `jq -e` must return
success. Also review every digest against the installed bytes, every absolute
path, all four narrow pose windows, head speed/torque limits, and all STM32
electrical/watchdog/stop claims. Confirm no other process owns any selected
serial node.

Start the sole STM32 owner first and inspect it. Controller startup failure is
fatal and exits the service; an active process plus clean startup logs are both
required before the bench can make its stronger fresh applied-zero check:

```bash
sudo systemctl start kiko-robot-server.service
sudo systemctl is-active kiko-robot-server.service
sudo journalctl -u kiko-robot-server.service -b -n 100 --no-pager
```

The bench's fresh applied-zero gate is the authoritative controller check. No
other local KRP2 client may run during this acceptance. `robot-server` is V2
only by default, and this unit does not opt into `--legacy-http-camera`; legacy
HTTP/camera routes are therefore absent from this bench deployment.

## Create one-shot physical attestations and run

Immediately before every start, physically re-check all three statements, then
create root-owned files in volatile `/run`:

```bash
sudo install -d -o root -g root -m 0700 /run/kiko-wheels-off.pending
sudo install -o root -g root -m 0400 /dev/null /run/kiko-wheels-off.pending/wheels-removed
sudo install -o root -g root -m 0400 /dev/null /run/kiko-wheels-off.pending/head-path-clear
sudo install -o root -g root -m 0400 /dev/null /run/kiko-wheels-off.pending/power-cut-reachable
sudo systemctl start kiko-nano-wheels-off-bench.service
sudo journalctl -fu kiko-nano-wheels-off-bench.service
```

The unit atomically renames the pending directory, requires the exact three
empty root-owned/root-group regular non-symlink files at mode `0400` in a
root-owned mode-`0700` directory, and rejects any file older than 60 seconds or
dated in the future. It consumes the complete transaction whether verification
succeeds or fails. A killed helper leaves only a consuming transaction that the
next attempt deletes; it cannot be replayed. `/run` is cleared by reboot and
the service never restarts itself. These files record a deliberate operator
act; software cannot prove that a wheel was removed or a hand is not in the
head path.

## View the live camera through Rerun

The example serves Rerun only on Nano loopback port `9876`. Its proxy cache is
explicitly bounded to 128 MiB; older telemetry may be evicted instead of
letting a disconnected viewer exhaust Nano memory. On the Mac, open a second
terminal:

```bash
ssh -N -L 9876:127.0.0.1:9876 makerspace@192.168.50.2
rerun --connect
```

The primary paths are:

- `bench/camera/rgb`: startup and ongoing tightly packed BGR frames;
- `bench/camera/depth/rectified_left_mm`: rectified-left `uint16` depth in
  millimetres, with `0` invalid and Rerun's metre scale set to `1000`
  (startup samples only);
- `bench/camera/imu/sensor_native/accel_m_s2_xyz`: raw startup accelerometer
  values;
- `bench/camera/imu/sensor_native/gyro_rad_s_xyz`: raw startup gyroscope values;
  and
- `bench/lifecycle`: ordered readiness events.

IMU vectors are in the OAK sensor-native frame, not calibrated or transformed
to the robot base frame. The camera readiness gate requires two observations
from each stream with strictly increasing capture/delivery/device timestamps;
accelerometer and gyroscope timestamps are checked separately. After the eye
session is ready, a third strictly newer RGB frame is processed immediately and
its bounded expression must be admitted by firmware. The OAK and host clocks
have unrelated epochs, so the gate cannot certify the frame's queue age.
Admission is not proof that the LEDs were physically visible. During the running phase, each cycle checks
the zero keeper and requests another RGB frame. Consecutive RGB timeouts are
bounded by the same parsed timeout/attempt budget used at startup, and a
replayed/non-increasing RGB sequence is rejected rather than counted as live.
Depth and IMU are not polled after startup, so this gate makes no ongoing
liveness claim for either stream.

## Runtime and cleanup order

The base zero-only keeper is established before accessory I/O and refreshes a
new STM32-applied zero within its lease. The physical sequence is:

1. parse and cross-bind every bounded document and exact artifact;
2. prove a fresh applied zero, connect the exact OAK, and establish RGB,
   rectified-left depth, and IMU continuity;
3. prove a newer zero, require all four pre-observation torque-disable writes,
   redundantly read the current head pose, apply the narrow pose-window gate
   before any energising write, then establish the bounded startup hold;
4. prove a newer zero, establish a fresh challenged KEP2 eye session, and
   admit an expression from a strictly newer third RGB frame whose queue age
   is not certified; and
5. remain in zero-only RGB observation until `SIGINT`, `SIGTERM`, a typed
   fault, or the configured 15-minute budget expires.

Signal acquisition runs independently during startup. A queued `SIGINT` or
`SIGTERM` is checked before head/eye operations and between each bounded camera
operation, then enters the same evidence-preserving cleanup path. During the
running phase, loss of keeper health or the bounded RGB liveness budget is a
typed fault and also triggers cleanup.

Cleanup first requests a newer applied base zero, then releases the eye,
attempts torque-disable for every head joint, closes OAK, flushes Rerun, and
finally disarms the zero keeper. Every result is retained separately; one
successful cleanup action does not hide another failure.

The current head shutdown report proves completion of the host serial writes;
it is not an STS register acknowledgement and does not independently prove
that physical torque is off. Keep the power cut reachable and isolate power
before touching the mechanism whenever its state is uncertain.

Stop with systemd and wait for normal cleanup:

```bash
sudo systemctl stop kiko-nano-wheels-off-bench.service
sudo systemctl show kiko-nano-wheels-off-bench.service -p ActiveState -p Result -p ExecMainStatus
sudo journalctl -u kiko-nano-wheels-off-bench.service -b --no-pager
sudo systemctl stop kiko-robot-server.service
```

The parser rejects a running-phase budget above 900 seconds. Systemd's
`RuntimeMaxSec=1200` is a separate outer ceiling that also leaves room for
bounded startup and normal cleanup; it is not a second claimed running phase.
A manual stop gives `SIGTERM` cleanup up to 120 seconds. `SIGKILL`, power loss,
or a stop-timeout escalation cannot run host cleanup. If the final zero, eye
release, torque-disable, OAK close, Rerun flush, or keeper disarm is uncertain,
cut physical power and do not attach wheels.

## What can be tuned in this gate

With wheels off, tune only from recorded evidence:

- narrow bow/curl/yaw/roll acceptance windows around an independently accepted
  natural pose;
- head serial deadlines, agreement tolerances, hold speed, and torque clamp,
  without adding a pose-seeking move;
- RGB/stereo dimensions and rates, IMU rate, nonblocking queue depth, and
  bounded capture attempts supported by the exact OAK; and
- RGB scene-motion thresholds and eye brightness after both firmware admission
  and visible behavior are observed.

The current bench has no exposure/focus calibration API, no IMU-to-base
calibration, and no wheel PWM-to-velocity fitting. Do not claim those are tuned
from a good-looking RGB frame or raw IMU stream.

Only ask to reinstall the wheels after one complete run shows ongoing RGB
liveness; startup depth and IMU continuity; a narrow accepted startup head
hold; eye admission and the desired visible expression; repeated applied-zero
evidence; and fully clean teardown with no uncertainty. Wheel installation and
encoderless visual/IMU plant identification are a separate, explicitly
approved commissioning gate.
