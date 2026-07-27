# Immutable Nano bundle rendering

The canonical renderer is:

```bash
cargo run --locked -p kiko-nano-bundle-renderer -- \
  check --input /canonical/absolute/path/render-input.json

cargo run --locked -p kiko-nano-bundle-renderer -- \
  stage \
  --input /canonical/absolute/path/render-input.json \
  --destination /canonical/absolute/path/new-empty-staging-directory
```

Both commands are offline host operations. They do not discover hardware,
install a file, open a Nano device, run a service, enable boot, flash
firmware, or assert that physical qualification passed. `stage` writes only
to its explicit destination and rejects a nonempty destination before writing
anything.

## Boundary contract

The schema version is selected with the bundle kind. Production remains
render-input schema V1 and its illustrative source template is
`configs/nano-agent-template/bundle-render-input-v1.json.template`.
Wheels-off qualification requires render-input schema V2. Its field layout
is the exact
`configs/nano-wheels-off-qualification-template/bundle-render-input-v2.json.template`;
it fixes the qualification bundle kind, null face assets, cold-start
selection, and reviewed SONAMEs while requiring the canonical absolute
`qualification_executable_path`. Production instead requires the complete
two-cascade object and has no executable-path field. Unresolved `${...}`
fields make either prepared file non-deployable. The strict renderer rejects
cross-version and bundle/asset mismatches.

Qualification render-input V1 and qualification launch V1 were already
published. The executable and native-runtime bindings added later therefore
belong to qualification render-input V2 and
`nano-wheels-off-qualification-launch-v2.json`; they are not silent mutations
of V1. Production render-input V1 and launch V3 are unchanged.

Prepare the JSON input from one retained discovery record and reviewed source
files. The strict parser:

- denies unknown and duplicate object fields;
- accepts only canonical absolute source paths;
- rejects source or destination path traversal and every source-path symlink
  component;
- accepts controller, head, and eye ports only as one
  `/dev/serial/by-id/<identity>` path;
- parses controller, eye, OAK, accessory, stream, resource, geometry, and
  numeric identities into bounded domain values;
- requires the four legacy native roles—DepthAI core, dynamic calibration,
  libusb 1.0, and ONNX Runtime—plus `opencv_core`, `opencv_imgproc`, and
  `opencv_objdetect` roles for both bundle kinds. The attended wheels-off
  binary includes the production dispatch under `nano-agent`, so its ELF
  required native-runtime manifest cannot omit the detector libraries even
  though its qualification path rejects and never loads face-cascade assets.
  Qualification pins all seven reviewed direct SONAMEs exactly:
  `libdepthai-core.so`, `libdynamic_calibration.so`, `libusb-1.0.so.0`,
  `libonnxruntime.so.1`, `libopencv_core.so.4.5d`,
  `libopencv_imgproc.so.4.5d`, and `libopencv_objdetect.so.4.5d`.
  Production continues to pin the three current Nano OpenCV SONAMEs exactly
  while retaining its existing V1 behavior for the other four role names;
- requires all artifact output paths to remain beneath `artifacts/`;
- rejects unresolved `${` tokens in every JSON boundary;
- checks the exact provisional controller ABI, build, fingerprint, and
  capability set before it can render a wheels-off candidate bundle; and
- retains the qualification executable at its fixed bundle path with mode
  `0555`, then launch-binds its exact size and SHA-256 along with the strict
  seven-role native-runtime manifest.

The serial-by-id values and OAK MXID are observations supplied by discovery;
rendering does not prove those devices remain connected. The renderer never
resolves a by-id path to a transient `ttyACM*` name.

The qualification executable, navigation-shadow document, canonical
calibration artifact, plant artifact, models, production frontal/profile face
cascades, and native libraries are exact leaf sources. The renderer retains
their exact bytes.
Production requires both cascade sources as one typed set; wheels-off
qualification rejects that unused set. The calibration input key is deliberately
`assets.calibration`, not `camera_calibration`: the one artifact owns the
exact OAK MXID, rectified stereo intrinsics/dimensions/baseline, raw IMU
calibration, tracking-camera-to-base transform, and three approval calibration
IDs. The renderer does not rewrite floating-point values, infer units, invent
a calibration, or copy a digest from user input. Offline deployment
qualification and runtime bootstrap parse that retained artifact and require
its exact manifest, live-OAK, navigation, and production-approval bindings.

## One digest, both representations

Each source is read into retained bytes once. Its exact byte length and
SHA-256 are computed from that retained sequence. Copied staging files reuse
that identity; rendered files compute their identity when their final byte
sequence is created. Inventory decimal digest arrays and launch hexadecimal
digests are emitted from the same `[u8; 32]`; neither representation is a
manual input.

After each staging write the file is read back and compared byte-for-byte to
the retained sequence. This readback does not create a second claimed digest.
The qualification executable alone is set to mode `0555`. Every other staged
file uses the renderer's established `set_readonly(true)` behavior, and each
staging directory is made read-only. A staging tree is still not a root-owned
production installation.

## Deterministic construction order

The renderer constructs and writes:

1. the exact qualification executable when selected, plus calibration, plant,
   navigation-shadow, model, production face-cascade, and native-library
   leaves;
2. the native-runtime manifest, inventory, controller contract, motion
   contract when applicable, candidate policy when applicable, and agent
   policy;
3. the exact render-input evidence copy—V2 for qualification and V1 for
   production—and the production-profile evidence copy when applicable;
4. `evidence/render-evidence-v1.json`; and
5. the bundle launch document, always last.

The evidence manifest records the exact source paths, source byte counts and
digests—including the qualification executable source—every content-bound
output except its own recursive identity, the launch identity, and the
deterministic write order. It deliberately records that installation,
ownership, device presence, hardware qualification, physical stop behavior,
and performance were not established.

`check` runs the complete parse/hash/render validation but creates no
destination. This makes it suitable before any Nano handoff.

## Production motion remains fail-closed

A production render input selects:

```json
{
  "kind": "production",
  "production_controller_profile_path": "/canonical/absolute/path/production-controller-profile-v1.json"
}
```

The profile is a separate schema-V1 record. Its source template is
`configs/nano-agent-template/production-controller-profile-v1.json.template`.
It has a fixed admission scope, exact controller identity, verified (not
`unverified`) physical stop semantics, controller envelope, physical plant
approval, timing bounds, and complete live-mode policy. Its identity must
match the discovery record at every controller identity field.

If the profile path is absent, production rendering fails before loading any
deployment leaf. The renderer does not silently manufacture a map-only
actuation contract or downgrade production to the raw candidate profile.
Production rendering likewise fails if either exact face-cascade source is
absent; both are copied, hashed, and bound into launch V3.
It also fails unless all three directly linked OpenCV shared-library sources
are supplied under their pinned SONAMEs. The renderer copies the regular
source bytes into `lib/<SONAME>` and hashes them; do not supply or install
symlink objects as deployment leaves.
The typed SONAME field is a required filename/role declaration, not an ELF
dynamic-section parser. Exact `DT_NEEDED`/loader-graph evidence from the final
ELF remains a separate target-side `readelf` release gate.
Supplying a syntactically valid profile is still not proof that its physical
claims are true; later typed deployment admission and attended physical gates
remain authoritative.

For production, the renderer derives
`navigation_config_sha256_hex` directly from the exact staged navigation
bytes, binds the same loopback port across inventory, controller, actuation,
and launch, and immediately parses the rendered controller document with the
real controller-server parser. The production controller identity and
capability set cause that parser to derive the
`production_external_interlocks` session class; schema V1 has no caller-set
session-class field that could overstate the connected firmware.

It also emits navigation-actuation schema V2 with two noninterchangeable
bindings. The production profile's canonical
`plant_dataset_content_id` identifies the exact physical-evidence dataset
named by the parsed plant model. `plant_artifact_sha256_hex` is derived
independently from the retained bytes of the exact staged plant-model file.
The dataset identity is never substituted for the plant-artifact digest; the
actuation parser rejects both legacy schema V1 and an equal reused digest.
Commissioning output remains a non-activatable proposal until an operator
reviews the dataset and plant separately and renders a new manifest binding.

## Persisted-map restart is an explicit bundle choice

`runtime.storage.warm_start` in the render input accepts exactly one of:

```json
{ "kind": "none" }
```

or, for production only:

```json
{ "kind": "dataset_replay" }
```

`dataset_replay` renders the canonical pair
`/var/lib/kiko-nano-agent/maps/current.kmap` and
`/var/lib/kiko-nano-agent/navigation` into the parsed agent policy.
Wheels-off-qualification bundles reject it because qualification must not
inherit an older map or dataset.

Use `none` for the first mapping deployment. Production always labels the map
operation **Finalize map & stop**, including that first cold session: capture
closes, the causal pipeline drains, controller stop and journal finalization
complete, and the runtime selects one finalized `navigation/session-*`
directory by an atomic `navigation/selected-warm-start-v1.json` replacement.
The selected session contains its final `occupancy.kmap`; the selection
exposes exact manifest/occupancy byte lengths and SHA-256 digests as restart
evidence. Ordinary command responses retain a 30-second upper bound; terminal
`save_map` has its own parsed 300-second upper bound for capture drain,
journal finalization, hashing, and synchronization. The final journal is read
back after synchronization, and its last accepted map epoch/revision must
exactly equal the retained occupancy identity before completion. A completion means restart inputs were
durably selected and the current control process is ending, not that mapping
continues. A cold (`none`) run creates the selection but does not consume it;
review that evidence, then render the next immutable bundle with
`dataset_replay`.

On restart, the mutable configured map pathname is not replay authority. The
loader parses the selection once, admits only its named direct-child session,
and retains exact descriptors for the selected `manifest.json` and
`occupancy.kmap`. Hashing and parsing consume the same handles, and those
handles are rechecked before and after replay binding. It also streams the
manifest-bound fixed-record navigation
journal in constant memory under its 1,048,576-record format cap; the final
epoch must have an accepted global map whose epoch and revision exactly match
the atomic selection. Rendering or replaying those files does not claim that
the live camera has localized. Replay must reproduce and bind the historical
map, and the live tracker must separately produce current-map relocalization
evidence before point-goal or autonomous motion can be authorized.

The digest claim is limited to the manifest and final occupancy artifact; it
does not content-address every frame/depth/IMU payload or the journal record
bytes. Journal records are nevertheless structurally parsed, order-checked,
and matched to the selected final map identity before replay. Selection reads
are bounded to 4 KiB, manifest hashing to 64 MiB, and occupancy hashing to
256 MiB. The streaming dataset directory itself remains outside the
byte-quota contract and must be provisioned and monitored independently.
The generic dataset reader still opens metadata, calibration, frames, depth,
IMU, and sidecars by path; those payload bytes are not selected digests.
Consequently this contract requires exclusive same-UID ownership of the Nano
state tree during replay and does not claim resistance to an active same-UID
payload mutator.

For wheels-off qualification, raw timer PWM remains isolated in the candidate
policy. Manual, point-goal, and frontier motion permissions remain disabled;
SLAM, occupancy, Rerun, and MPC shadow do not acquire production authority.

## Source templates

The production template directory describes every derived production
contract:

- `agent-policy-v3.json.template`;
- `controller-server-v1.json.template`;
- `device-inventory-v1.json.template`;
- `navigation-actuation-v2.json.template`;
- `native-runtime-v1.json.template`; and
- `nano-agent-launch-v3.json.template`.

These files document the on-disk shapes and remain intentionally invalid
until expanded. Do not expand them with shell substitution or manually
transcribe hashes. The Rust renderer constructs the equivalent documents from
the parsed domain record, so strings are escaped correctly and all digest
bindings share one source identity. The navigation-shadow configuration is an
exact independently reviewed leaf, not a generated template.
