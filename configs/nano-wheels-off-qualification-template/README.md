# Nano wheels-off qualification bundle

This directory is the source for the manually invoked, qualification-only Nano
bundle. It is not a rendered deployment, physical evidence, a production
configuration, or permission to attach the wheels.

The qualification contracts are deliberately different from production:

- `bundle-render-input-v4.json.template` is the qualification-only renderer
  boundary with its bundle kind, exact face-cascade sources, optional
  head-gaze input disabled by absence, cold-start policy, conservative
  software policy, and all seven exact native-runtime SONAMEs fixed;
- `device-inventory-candidate-v2.json.template` is candidate inventory schema
  V2;
- `candidate-controller-policy-v1.json` is candidate host-policy schema V1;
- `controller-server-candidate-v2.json.template` is controller-server schema
  V2;
- `head-gaze-policy-v1.json.template` is a non-deployable, proposal-only
  head-gaze source template that fixes only the facts already declared by the
  host contract; and
- `agent-policy-v3.json.template` supplies the common OAK, accessory,
  persistence, and loopback-console policy while leaving all production motion
  modes disabled.

There is deliberately no checked-in launch-V4 template. The renderer derives
`nano-wheels-off-qualification-launch-v4.json` from the one typed V4 render
input and writes it last. Keeping a second tokenized launch template would make
the fixed Gate-A software policy ambiguous.

The remaining unexpanded `${...}` tokens are intentional deployment
boundaries: exact observed device identities, source paths, generated
calibration identity, and native build provenance. The V4 renderer computes
every byte count and SHA-256 itself; no checked-in digest is presented as
evidence about a future installed byte sequence.

## Fixed Gate-A software policy

The V4 render-input template now fixes one bounded Gate-A graph instead of
asking deployment-time string replacement to make software-policy decisions:

- RGB and rectified stereo/depth are `640x400@15`, IMU is `200 Hz`, and each
  nonblocking OAK queue has capacity 4;
- the OAK requests and requires DepthAI `SUPER`, so a fresh same-owner
  readback must be the qualified 5 Gbit/s USB-3 transport;
- global occupancy is a 20 m square at 5 cm resolution with 160,000 cells,
  4,096 retained keyframes, and a snapshot every 20 keyframes;
- SuperPoint and LightGlue use the reviewed CPU boundary, downscale 2, and at
  most 512 keypoints;
- Rerun is decimated by 2 with a 128 MiB memory bound and 2 s flush timeout;
- map/dataset storage has explicit 64 MiB, 4 GiB, file-count, ingress-count,
  free-space, and terminal-reserve bounds; and
- the previously reviewed natural-head and RGB-expression values are copied
  exactly, including the operator-declared head origin
  `[0.0,-0.25,-0.20] m` and parallel neutral axes.

These are admission choices for an attended, wheels-off qualification. They
are not frame-rate, latency, control-tick-jitter, solver-budget,
mapping-duration, thermal, storage-throughput, gaze-calibration, or performance
evidence. The fresh OAK calibration must have the exact selected stereo
dimensions. Enforced solver, lease, freshness, quota, and live-contract
failures remain qualification failures rather than reasons to change the
running configuration implicitly. Control-tick lateness is retained as
diagnostic evidence; this policy does not claim a fail-closed jitter threshold.

## Proposal-only head gaze

The checked-in head-gaze template is deliberately not a calibration artifact
and is not valid deployment JSON until every `${...}` token is replaced. It
fixes only these currently declared facts:

- the head centre is `[0.0, -0.25, -0.20] m` in OAK camera coordinates, where
  OAK `+y` is image-down and `+z` is forward;
- the neutral head axes are parallel to the OAK optical axes;
- a two-dimensional face ray uses an assumed `1.5 m` camera-forward focus
  plane, not observed range;
- the natural encoder declaration is bow/curl/yaw/roll
  `[2155, 2545, 2943, 2876]`; and
- the lifecycle is `proposal_only`.

The assembly identifier, retained proposal-evidence identifier and digest,
hard encoder envelopes, encoder signs/scales, controller timing, hysteresis,
and per-joint motion limits remain visibly named `UNVALIDATED` sentinels.
Replacing a sentinel proves neither its physical value nor its safety. The
rendered policy must continue to use `proposal_only`; it must not be converted
to `operator_claimed_physical_review` without a separate, retained, physically
witnessed review transaction.

The controller declaration remains non-command metadata even after it parses.
Gate A leaves `assets.head_gaze_policy_source_path` absent, so the renderer
emits no policy leaf or hash and bootstrap makes no gaze-adapter claim. A
future qualification bundle may supply the field only with complete exact
proposal bytes. Bootstrap then requires a `proposal_only` lifecycle before
opening hardware. Even then, the policy supplies no torque consent, motion
consent, or conversion into a commanded head pose. Gate A only requires the
separately reviewed natural-return-and-hold policy in `agent-policy-v3.json`;
head-gaze activation is not part of the wheels-off attachment gate.

## Synthetic shadow plant

`qualification-shadow-only-synthetic-unvalidated-plant-v2.json` is a
checked-in, qualification-shadow-only synthetic fixture. Its numeric values
are test inputs, not measurements, physical identification, performance
evidence, or permission to actuate. The explicitly non-calibrated
`../navigation-shadow-v1.example.json` duplicates the same plant declaration
and is parser-tested against this artifact. It is only an example: its
identity camera/base transform and synthetic IMU calibration are not suitable
for a rendered qualification bundle.

V2 uses a 50 ms sample period and the candidate's ±30% PWM envelope. The V1
fixture's 100 ms sample period could never pass the candidate runtime's strict
54,999,999 ns service-interval ceiling because plant, MPC, and control periods
must match exactly. The version change is an input-contract correction, not a
performance claim or physical plant identification.

The fixture is never wired into production. The qualification renderer does
not discover it by ambient path: deployment supplies one explicit source, and
the renderer binds that source to the exact V2 artifact ID, destination,
semantic identity, and checked-in SHA-256 before staging it. The recommended
`navigation-shadow-preparation-v1.json.template` embeds the same synthetic
plant domain so Gate A does not circularly claim wheel-on physical
identification before wheel attachment. A qualification render must still
supply a reviewed navigation document whose camera/base transform and IMU
calibration bind to the canonical physical calibration artifact. Bootstrap
requires the embedded and separate plant domains to be exactly equal.
Production admission rejects this synthetic evidence as physical plant
identification.

## Fixed candidate contract

The checked-in candidate controller policy requests raw signed timer-duty
percentages through `127.0.0.1:8080`. It fixes:

- an absolute raw timer-duty request cap of 30%;
- a manual test magnitude of 10%;
- a 150 ms manual deadman;
- a 20 ms non-stop command interval;
- a 100 ms controller lease; and
- a maximum 30 s age for the attended wheels-off attestation.

The firmware/server identity is the canonical provisional four-PWM candidate:
ABI 2, build `135169`, fingerprint
`4b494b4f2d3450574d2d43414e443121`, capabilities `575`, and unverified
physical stop semantics. These constants identify software. They do not prove
pin wiring, driver behavior, motor direction, useful duty, PWM-to-velocity,
coast/brake behavior, stopping time, or stopping distance.

Raw timer PWM is not a body velocity and never enters the production MPC
actuation API. SLAM, occupancy, Rerun, and the navigation/MPC shadow may run
during this session, but autonomous, point-goal, frontier-explore, and
production manual velocity actuation remain disabled.

## Rendering order

Render into a staging directory, never directly into the live deployment:

1. prepare `bundle-render-input-v4.json.template`, setting
   `bundle.qualification_executable_path` to the absolute path of the reviewed
   Linux-aarch64 qualification executable and supplying the distinct exact
   frontal/profile cascades; its qualification bundle kind, schema version,
   and cold-start selection are fixed rather than caller placeholders. Leave
   `assets.head_gaze_policy_source_path` absent for Gate A. A later
   proposal-only qualification may render `head-gaze-policy-v1.json.template`
   to a separate source, retain the exact proposal evidence named by its
   lifecycle claim, and add that field; never point the renderer at the
   checked-in `.template`;
2. install the exact canonical calibration artifact, shadow-only plant,
   navigation-shadow configuration, model bytes, and all seven required roles
   in the closed qualification native-runtime manifest under their reviewed
   exact SONAMEs:
   `libdepthai-core.so`, `libdynamic_calibration.so`, `libusb-1.0.so`,
   `libonnxruntime.so.1`, `libopencv_core.so.4.5d`,
   `libopencv_imgproc.so.4.5d`, and `libopencv_objdetect.so.4.5d`.
   The shared source schema is
   `../nano-agent-template/calibration-artifact-v1.json.template`; it binds
   one canonical OAK MXID to exact rectified stereo geometry, raw IMU
   calibration, tracking-camera-to-base transform, and three later production
   approval IDs;
3. retain the executable at
   `bin/kiko-nano-wheels-off-qualification`, retain it as mode `0555`, and
   compute each installed leaf's exact byte count and lowercase SHA-256;
4. render `native-runtime-v1.json` from the seven exact native leaves;
5. render candidate inventory V2 from exact observed identities and the
   calibration/plant digests expressed as 32 decimal bytes;
6. render the qualification agent policy and candidate server contract;
7. copy the candidate controller policy without editing it;
8. compute the byte count and lowercase SHA-256 of every rendered JSON input;
9. render the evidence manifest and launch document last from those exact
   values, including exact bindings for the executable, native-runtime
   manifest, two cascades, and the optional head-gaze policy only when
   supplied. Retain the qualification render input as
   `evidence/render-input-v4.json`; and
10. reject the staging tree if any `${` token remains or any rendered JSON
   fails `jq -e .`.

The deployment renderer must produce both the 64-character lowercase
hexadecimal digest used by launch bindings and the exact 32-element decimal
byte array used by inventory artifact entries from the same digest. Do not
transcribe either representation manually.

The two cascade leaves are independently bounded to 4 MiB and must have
distinct destinations and exact content. When supplied, the qualification-only
head-gaze policy is bounded to 256 KiB, parsed as exact JSON, retained at
`head-gaze-policy-v1.json`, rejected if its path or content aliases another
launch-bound input, and admitted by bootstrap only as `proposal_only`.

Qualification render-input/launch V1, V2, and V3 were already published
contracts. They must not be relabelled: V2 incorrectly selected the system ABI
name `libusb-1.0.so.0` for the pinned DepthAI libusb role, while V3 had no
face-cascade or head-gaze-policy bindings. The current renderer therefore
rejects qualification render-input V1 through V3 and emits launch V4.
Production render-input schema V1 and production launch V3 remain unchanged.

The pinned DepthAI v3.4.0 `libdepthai-core.so` directly needs
`libusb-1.0.so`, so that exact file is the qualification role retained and
hashed here. A separate system `libusb-1.0.so.0` may still appear transitively
through the Nano's OpenCV/libdc1394 stack; it belongs to the measured OS ABI
closure and is not a substitute for the pinned DepthAI role.

The launch storage placeholders are also mandatory in qualification. Dataset
logical bytes, regular-file count, and ingress-record count are independent
cumulative limits; the descriptor-relative free-space floor is a physical
filesystem constraint. The file ceiling cannot exceed 65,536 while
finalization uses one bounded monolithic manifest; a longer scan needs a
reviewed chunked-manifest format. The terminal reserve must be below the dataset byte
maximum and at least the 4096-byte-fragment-rounded sum of the configured map
ceiling, 64 MiB manifest ceiling, and 4 KiB selection ceiling. Occupancy and
selection remain map-persistence artifacts rather than dataset-quota bytes.

The exact attended installation and operation procedure is
[`docs/nano-wheels-off-qualification.md`](../../docs/nano-wheels-off-qualification.md).
There is intentionally no qualification systemd unit.

Use the offline
[`kiko-nano-bundle-renderer`](../../crates/kiko-nano-bundle-renderer/README.md)
for this order. Its `check` mode creates nothing; its `stage` mode accepts
only a new or empty destination, writes a content-addressed evidence manifest,
and writes the qualification launch document last. It never installs the
bundle or touches a live device.
