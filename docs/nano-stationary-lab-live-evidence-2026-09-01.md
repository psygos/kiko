# Nano stationary-lab live evidence — 2026-09-01

This record covers the prompt-free, zero-motion-authority stationary entry
point only. It is integration evidence, not wheels-off motion qualification,
wheel-on plant calibration, occupancy-map acceptance, or autonomous-navigation
evidence.

## Candidate identity

- Source commit: `2355be9e4777b99ec0b5a380651293e29ea8d81e`
- Source archive SHA-256:
  `052ea00004a892f1921bc213acd2721e5adf02cf3ba3bc0e508ac7ea53110d7c`
- Native Linux aarch64 executable size: `31,283,704` bytes
- Native executable SHA-256:
  `8c2f5f02e78412cc3b2ede2f8df5b9b10e404438011149559f405e6a51a40ad7`
- Rendered launch-record SHA-256:
  `a22054aaef38cba582d3d48da804666a2f7778ee3558dfdd12d1992c605d179b`
- Render evidence SHA-256:
  `6d89ad813dc0a73af1d8b032d08f839a88f9bf1c4707ecbf66fd739c6dfd0e7f`
- Installed immutable root: `/opt/kiko/qualification`

The source archive was transferred to the Jetson, verified before extraction,
built natively with the locked dependency graph, and rendered through the
canonical bundle renderer. The installed tree matched the staging tree with
`diff -qr`. `ldd`, with the launch-bound library path, reported no missing
libraries. A preliminary binary-only replacement was rejected before device
ownership because the old launch record's executable-size bound did not match;
the installed candidate is the subsequently rerendered bundle, not that
rejected tree.

## Invocation and observation

The candidate was left running in the foreground with:

```bash
sudo /usr/bin/env LD_LIBRARY_PATH=/opt/kiko/qualification/lib \
  /opt/kiko/qualification/bin/kiko-nano-wheels-off-qualification \
  nano-stationary-lab \
  --deployment-root /opt/kiko/qualification \
  --launch-config nano-wheels-off-qualification-launch-v4.json \
  --state-root /var/lib/kiko-nano-qualification
```

At `2026-09-01T00:00:53+05:30`, PID `95537` had remained alive for more than
five minutes, beyond the earlier approximately 174-second raw-dataset quota
failure. The state root measured `8,135,808,633` bytes before and after the
observation: delta `0` bytes. No file under that root had been modified during
the observation window. The pre-existing size includes retained files from
older aborted persistent sessions; they were not deleted or rewritten.

The operator snapshot after 5 minutes 4 seconds reported:

- runtime `ready_stopped`, requested owner absent, and actual authority absent;
- process-lifetime software safety stop latched;
- controller output disabled, requested/applied left and right PWM all exactly
  zero, and controller fault bits zero;
- STM32, head, eyes, and OAK `ready`;
- SLAM `degraded`, with 161 started pairs, 159 successful pairs, one
  recoverable failure, and zero fatal failures; and
- map, pose, path, goal, and MPC prediction absent.

The SLAM degradation is truthful: CPU SuperPoint and LightGlue inference was
repeatedly slower than the configured 200 ms warning threshold. This run is
therefore evidence that live perception stayed operational, not evidence of a
real-time SLAM rate or a usable occupancy map.

The OAK enumerated as a 5,000 Mbit/s USB device beneath a 10,000 Mbit/s hub and
root hub. Rerun and the operator console listened only on loopback ports 9876
and 9877. The current console schema still exposes legacy
`wheels_off_qualification` labels and raw-PWM pattern metadata even though the
stationary entry point cannot acquire motion authority; that presentation debt
must not be mistaken for an enabled command path.

## Result and remaining gates

The stationary entry point now starts without a physical-attestation dialog,
retains constant-memory navigation-ingress ordering checks, does not allocate
or publish a replay dataset, and keeps nonzero base authority structurally
unavailable. Motion-enabled attended and production modes continue to use the
persistent journal and their existing safety gates.

This run does not establish occupancy output, physical head-gaze behavior,
grounded wheel signs, PWM-to-velocity calibration, the drive plant, MPC
tracking, click-to-goal navigation, or recovery behavior. Those remain explicit
acceptance work; no claim is inferred from the stationary observation.

## Stationary console and accessory-lifecycle refresh

The later `a02c26bec85336f35f42565cf1c64cd9e344c6a7` candidate supersedes
the console presentation described above. Its stationary mode has a distinct
`stationary_lab` authority identity, omits the qualification/control-profile
projection, does not render manual or mode controls, and returns HTTP 404 for
the raw qualification-intent route before request-body parsing. The shared
binary and bundle kind retain their qualification names because the same
reviewed artifact also contains the explicitly attended lane; those internal
names do not grant the stationary entry point another authority.

Two intermediate candidates were rejected rather than relabelled successful:

- `8a0b7418c260c99f72cf2267f2873f58d39182d3` removed the stationary
  controls, but later telemetry ticks incorrectly projected the shared
  `wheels_off_qualification` identity. The runtime rejected that contradiction
  with a typed `ProfileMismatch` fault.
- `83386e3b9144385210f38de046715666bf1f2634` retained stationary identity
  on later ticks. Live use then exposed an independent cancellation defect: a
  health or frame select branch could drop an already-submitted KEP2 reply
  receiver. The eye actor correctly failed closed with
  `ActorStoppedBeforeReporting`. A separate startup attempt also failed closed
  when bow telemetry had only 116 ms of freshness remaining against the
  configured 200 ms worst-case write budget.

Commit `a02c26b` makes the accessory loop select observation-only triggers and
runs every head/eye transaction to a typed completion after selection. Its
regression fixture deliberately combines a 20 ms eye acknowledgement with a
1 ms health cadence. The acknowledgement completes before the overdue health
transaction; the old cancellation structure cannot satisfy that test.

### Refreshed candidate identity

- Source archive SHA-256:
  `2a1721a8663c5d1a1fb3322e8c11ac408e9e9404123ec4a0f83ebd2deefc7f37`
- Native Linux aarch64 executable size: `31,359,480` bytes
- Native executable SHA-256:
  `dfce412a696a5455caecb361a4b4188012d4c583d81c0cf8a02f74bf57bb27fb`
- Bundle renderer SHA-256:
  `b436597feba9c8804c62023d494f0776f28544d44ada392e8e220977e0bd5aca`
- Render-input SHA-256:
  `4aaf722c21d87e8eed19d4b8beca26592bcb19f7fd81dc10d3f7da124afdd776`
- Render-evidence SHA-256:
  `456e10a7c7df352d8ac6032311cfd602a1be4ea707cf1c3a5e129288e141719b`
- Launch-record SHA-256:
  `b8976b3e0c8b127e80f73bc5f460b4273955b1cc7f016e7d798a52d3881fee1a`

The archive hash matched before extraction. The commit was built natively from
a clean extraction with the locked graph and a deterministic source-path
remap. Renderer `check` and `stage` both succeeded. The installed immutable
root matched the staged tree under `diff -qr`; the rejected `83386e3` install
was retained as a separate rollback directory.

Host evidence for the lifecycle change is 1,475 `nano-agent` library tests and
109 exact `nano-wheels-off-qualification` binary tests passing, plus strict
Clippy, rustfmt, and `git diff --check`. The 64 focused accessory-worker tests
are included in the library count.

### Refreshed live result

The foreground stationary runtime reached `ready_stopped`. Two authenticated
snapshots advanced from revision 399 to 1,094 while SLAM advanced from 10 to 28
successful pairs with zero recoverable and zero fatal failures in that window.
Both snapshots reported:

- `authority_kind: stationary_lab`, with requested owner and actual authority
  absent;
- no `control_profile` or `wheels_off_qualification` member;
- STM32, head, eyes, and OAK ready;
- requested and applied left/right PWM exactly zero, controller output
  disabled, controller faults zero, and the software safety stop latched; and
- map, pose, path, goal, and MPC prediction absent.

An authenticated POST to
`/api/v1/wheels-off-qualification/intents` returned HTTP 404 with
`{"error":"not_found"}`. The delivered application contains the explicit
`STATIONARY LAB — BASE CONTROL DISABLED` banner. After OAK pipeline boot,
`lsusb -t` placed the camera at 5,000 Mbit/s beneath the 10,000 Mbit/s Tegra
hub. The unbooted device had initially appeared on the USB 2 tree; that was not
used as a false USB3 claim.

At 5 minutes 3 seconds, PID 110291 remained live and the state root was still
exactly `8,135,808,633` bytes: delta zero from the pre-launch measurement. The
snapshot had reached revision 6,169 with 156 started pairs, 155 successful
pairs, zero recoverable failures, and zero fatal failures. STM32, head, and eyes
remained ready. The final point-in-time OAK and SLAM health fields were
degraded, although paired frames continued and the OAK remained enumerated at
5,000 Mbit/s; this is not presented as continuous OAK readiness or a real-time
SLAM result.

The refreshed run remains deliberately honest about missing motion evidence:
SLAM health is degraded by measured CPU inference latency, and a stationary
camera cannot create a useful map or current localization. This deployment
does not establish wheel signs, plant calibration, MPC tracking, or
click-to-goal navigation.

## Exact fast-geometry candidate and shutdown evidence

The final stationary refresh was built from source commit
`b3d7edaab62fc984ec538cf88cd17e2922586050`. Its source archive was transferred
to the Orin and matched SHA-256
`c325833a8b53cb598db3b3cea2f2b7d0aa1f405a06ae2c784476d15acf186733`
before extraction. The locked native Linux aarch64 build produced:

- qualifier executable: `31,400,296` bytes, SHA-256
  `7111a847b599ba595335539eca89ee888e401c8d88770a3d060dccff294cbc6f`;
- bundle renderer: `2,971,440` bytes, SHA-256
  `e4cfbc39587a60b058fe8b307da8d19a885614fa408ae6d465466f0a936956c6`;
- render input SHA-256
  `039d30b405d9b4203f50ef6a0b543d7df0e352cf7aa993a120383012a0e3ed9c`;
- render evidence SHA-256
  `028cbc1428d6ad904873a366af7ca2022975baa2ff5418a6b7517599d73e708b`;
  and
- launch record SHA-256
  `51350a640d8d3afcf8e9b7f6dda682e7114c78a8f354a43101e69a99b1e8d7e9`.

Renderer `check` and `stage` both passed. The staging tree contains 23 bound
files, its launch-bound `ldd` closure has no missing library, the installed
root matched staging under `diff -qr`, and the installed executable is
`root:root` mode `0555`. The previous working bundle remains intact at
`/opt/kiko/qualification-retired-feb02a1-cpu-d8-k128-final` for rollback.

### Inference admission and measurements

The tracked SuperPoint model contract now has one shared source of truth: each
admitted downscale factor must divide both 640 by 400 rectified axes, each
resulting model axis must be at least the graph's eight-pixel stride, and the
requested keypoint ceiling cannot exceed the graph's 512-element TopK output.
The installed stationary profile requests CPU for both models, downscale 8,
and at most 128 keypoints. These are parsed launch values, not ambient tuning
defaults.

Intermediate runs were retained as negative or comparative evidence rather
than promoted:

- strict CUDA failed closed because the tracked SuperPoint graph retained CPU
  nodes while CPU fallback was explicitly disabled;
- CUDA-plus-CPU hybrid stayed operational but its observed 64-completion
  window was `0.439467 Hz`, so no acceleration claim is made;
- CPU downscale 4 with 256 keypoints observed `1.544239 Hz` over its retained
  window; and
- downscale 6 was discovered to be geometrically invalid for 640 by 400 after
  318 recoverable runtime failures. The launch parser now rejects that input
  before device ownership instead of repeating invalid inference.

Those observations were functional qualification runs under different load
and scene conditions, not GPU benchmarks, power-mode tests, or thermal tests.
They support rejecting unsuitable configurations; they do not establish a
general performance comparison.

### Exact live result

After more than five minutes, the authenticated schema-5 snapshot reported:

- runtime `ready_stopped`, `authority_kind: stationary_lab`, and no requested
  owner, actual authority, manual command envelope, requested command, or
  requested actuation;
- STM32, head, eyes, OAK, and SLAM all `ready`;
- exact applied stop with requested/applied PWM `0/0`, output disabled,
  controller fault bits zero, and controller-reported safe stop certainty;
- 2,438 started stereo pairs, 2,436 successful pairs, one recoverable startup
  triangulation miss, and zero fatal failures;
- a latest 64-success window spanning `7,896,418,413 ns`, equal to
  `7.978300630 Hz` over its 63 completion intervals; and
- no map, pose, path, goal, MPC prediction, or solver-duration claim.

The state root was exactly `8,135,808,633` bytes before and after the run. The
OAK was again enumerated at 5,000 Mbit/s beneath the 10,000 Mbit/s Tegra hub.
An authenticated same-origin POST to the removed qualification-intent route
returned HTTP 404 with `{"error":"not_found"}`. The stationary surface has no
qualifier/manual controls and cannot acquire nonzero base authority.

The absent map is expected and material: the camera and robot remained
stationary, so this run cannot prove representative-motion localization or
occupancy quality. It proves that the live graph, bounded routes, diagnostics,
accessory ownership, and typed zero owner remain operational; it does not turn
stationary frames into mapping evidence.

The predecessor candidate exposed a deliberate-stop race: navigation closed
its lossless receiver immediately while one already-admitted inference result
was still completing, and that expected causal tail was mislabeled as a
`VisualAdmissionRoute::Disconnected` failure. Commit `da88d4f` distinguishes
only the already-requested shutdown tail; live-process timeout and disconnect
remain failures, and fatal tracker or panic outcomes remain authoritative. A
live Ctrl-C of the exact candidate then ended with pipeline state `Closed`,
2,645 successful of 2,646 started pairs, the same one recoverable startup miss,
zero fatal failures, and no `LiveRunError`. The exact candidate was restarted
after that test and left running in stationary mode.

Current-source host evidence is 1,551 exact-feature library tests, 110
exact-feature runtime-binary tests, and all 36 renderer tests passing. Strict
all-target/all-feature Clippy for `kiko-slam` and the renderer passed, as did
the renderer's default-feature release check, formatting, and
`git diff --check`.

This closes the prompt-free stationary runtime refresh, not Gate A. The
attended candidate fault matrix, shaft/body sign evidence with motor power
available and wheels absent, final disarmed power-cut handoff, wheel-on plant
identification, map-quality acceptance, MPC tracking, and click-to-goal
qualification remain separate physical gates.
