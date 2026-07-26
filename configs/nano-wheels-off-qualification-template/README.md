# Nano wheels-off qualification bundle

This directory is the source for the manually invoked, qualification-only Nano
bundle. It is not a rendered deployment, physical evidence, a production
configuration, or permission to attach the wheels.

The qualification contracts are deliberately different from production:

- `nano-wheels-off-qualification-launch-v1.json.template` is launch schema V1;
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

1. prepare the shared illustrative render input at
   `../nano-agent-template/bundle-render-input-v1.json.template` with
   `bundle.kind` equal to `wheels_off_qualification` and the whole-value
   `FACE_PERCEPTION_ASSETS_JSON` token equal to `null`; qualification rejects
   production cascade assets;
2. install the exact canonical calibration artifact, shadow-only plant,
   navigation-shadow configuration, model bytes, and all seven required direct
   native-library roles: DepthAI core, dynamic calibration, libusb 1.0, ONNX
   Runtime, OpenCV core, OpenCV imgproc, and OpenCV objdetect.
   The shared source schema is
   `../nano-agent-template/calibration-artifact-v1.json.template`; it binds
   one canonical OAK MXID to exact rectified stereo geometry, raw IMU
   calibration, tracking-camera-to-base transform, and three later production
   approval IDs;
3. compute each installed leaf's exact byte count and lowercase SHA-256;
4. render `native-runtime-v1.json` from the seven exact native leaves;
5. render candidate inventory V2 from exact observed identities and the
   calibration/plant digests expressed as 32 decimal bytes;
6. render the qualification agent policy and candidate server contract;
7. copy the candidate controller policy without editing it;
8. compute the byte count and lowercase SHA-256 of every rendered JSON input;
9. render the evidence manifest and launch document last from those exact
   values; and
10. reject the staging tree if any `${` token remains or any rendered JSON
   fails `jq -e .`.

The deployment renderer must produce both the 64-character lowercase
hexadecimal digest used by launch bindings and the exact 32-element decimal
byte array used by inventory artifact entries from the same digest. Do not
transcribe either representation manually.

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
