# Nano wheels-off qualification bundle

This directory is the source for the manually invoked, qualification-only Nano
bundle. It is not a rendered deployment, physical evidence, a production
configuration, or permission to attach the wheels.

The qualification contracts are deliberately different from production:

- `bundle-render-input-v3.json.template` is the qualification-only renderer
  boundary with its bundle kind, null face assets, warm-start policy, and all
  seven exact native-runtime SONAMEs fixed;
- `nano-wheels-off-qualification-launch-v3.json.template` is launch schema V3;
- `device-inventory-candidate-v2.json.template` is candidate inventory schema
  V2;
- `candidate-controller-policy-v1.json` is candidate host-policy schema V1;
- `controller-server-candidate-v2.json.template` is controller-server schema
  V2; and
- `agent-policy-v3.json.template` supplies the common OAK, accessory,
  persistence, and loopback-console policy while leaving all production motion
  modes disabled.

The unexpanded `${...}` tokens are intentional. In particular, every SHA-256
is a deployment-tool output placeholder. No checked-in digest is presented as
evidence about a future installed byte sequence.

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

1. prepare `bundle-render-input-v3.json.template`, setting
   `bundle.qualification_executable_path` to the absolute path of the reviewed
   Linux-aarch64 qualification executable; its qualification bundle kind,
   null face assets, schema version, and cold-start selection are fixed rather
   than caller placeholders;
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
   values, including exact bindings for the executable and native-runtime
   manifest. Retain the qualification render input as
   `evidence/render-input-v3.json`; and
10. reject the staging tree if any `${` token remains or any rendered JSON
   fails `jq -e .`.

The deployment renderer must produce both the 64-character lowercase
hexadecimal digest used by launch bindings and the exact 32-element decimal
byte array used by inventory artifact entries from the same digest. Do not
transcribe either representation manually.

Qualification render-input/launch V1 and V2 were already published contracts.
They must not be relabelled: V2 incorrectly selected the system ABI name
`libusb-1.0.so.0` for the pinned DepthAI libusb role. The current renderer
therefore rejects qualification render-input V1 and V2 and emits launch V3.
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
