# Production Nano launch template

This directory is a source template, not a qualified deployment and not
evidence that any device or motion contract is satisfied. The canonical
offline renderer and its strict input/evidence contract are documented in
[`docs/nano-bundle-renderer.md`](../../docs/nano-bundle-renderer.md). The
`nano-agent-launch-v3.json.template` file deliberately contains `${...}`
tokens, so it is not JSON and cannot be admitted until a deployment tool has
replaced every token with measured or independently reviewed values.
`native-runtime-v1.json.template` is likewise a non-deployable source
template. Render it from the exact staged native-library names, byte
ceilings, and lowercase SHA-256 values. The complete offline qualification,
native closure, exclusive-endpoint acquisition, and qualified-only boot
procedure is in
`docs/nano-qualified-deployment.md`.

Launch schema V3 retains V2's separate map and navigation-dataset contracts
and adds mandatory, distinct, exact frontal/profile face-cascade bindings.
V2 replaced V1's aggregate navigation-record,
startup-evidence, and total-state fields with separate enforceable map and
navigation-dataset contracts. Do not relabel an older document as V3: render
and review the complete V3 storage and `face_perception` sections explicitly.

A qualified deployment must:

1. install the binary and every input asset under a root-owned, non-writable
   deployment root such as `/opt/kiko/deployment`;
2. hash the exact installed bytes for every asset and place the lowercase
   SHA-256 in the rendered launch document;
3. use the byte count (or a deliberately smaller reviewed ceiling) for each
   `maximum_bytes` field;
4. bind the plant ID and bytes to the exact admitted device manifest;
5. bind one canonical calibration artifact by ID, path, byte ceiling, and
   digest to that same manifest. The artifact must contain the exact canonical
   OAK MXID, rectified left/right intrinsics and dimensions, metric baseline,
   raw IMU calibration, tracking-camera-to-base transform, and the three
   production approval calibration IDs. Runtime admission compares every
   retained value exactly with the same opened OAK, parsed navigation
   configuration, and production actuation approval;
6. cross-check the in-process V2 controller endpoint and contract against the
   physical-actuation configuration. The retained `controller_server` schema
   name describes that bound contract; it does not authorize a second
   `robot-server` process;
7. bind the local control socket and per-boot operator-console capability below
   `/run/kiko/`, the service-owned runtime directory. Its exact mode is `0700`;
   the host boundary rejects a group- or world-accessible parent and writes the
   capability itself as `0600`;
8. use the exact OAK stream dimensions and rates selected during wheels-off
   qualification;
9. set every `occupancy` field from the reviewed global-map resource envelope.
   This section owns only grid extent, maximum retained evidence, and snapshot
   cadence. The exact `navigation-shadow-v1.json` owns the level
   optical-world/camera-height transform, runtime rectified-left depth camera
   and intrinsics, height/depth ranges, and sampling block; do not duplicate
   or override them with environment variables;
10. select inference backends only as requested providers. A selection is not
   a claim of availability, compatibility, latency, throughput, or speedup;
11. set `maximum_map_snapshot_bytes` from the largest reviewed encoded map and
    set `minimum_free_bytes_after_map_save` from the required post-publication
    filesystem headroom. These two limits are reserved before and verified
    after each map replacement. Set
    `maximum_navigation_dataset_bytes`,
    `maximum_navigation_dataset_files`, and
    `maximum_navigation_ingress_records` from the reviewed session envelope,
    and set `minimum_free_bytes_after_navigation_dataset_write` from the
    required descriptor-relative filesystem headroom. Dataset bytes are exact
    cumulative logical regular-file lengths for payloads, sidecars, IMU,
    journal, and manifest; they are not a claim about fragment-rounded
    physical allocation. The record ceiling is an independent upper bound,
    not a conversion from the byte or file ceilings. The current file ceiling
    is at most 65,536 because finalization still builds one bounded monolithic
    manifest; longer sessions require a reviewed chunked-manifest format
    rather than raising this launch value. Set
    `navigation_dataset_terminal_reserve_bytes` below the dataset byte maximum
    and at least to the 4096-byte-fragment-rounded sum of the map ceiling,
    64 MiB manifest ceiling, and 4 KiB warm-selection ceiling. Open-ended
    capture cannot consume that reserve. Occupancy and warm-selection files
    remain owned and counted by the map-persistence contract, not by dataset
    logical bytes; the reserve protects allocation for their terminal
    publication. Concurrent external writes can still race a reservation, so
    post-write verification must fail the session truthfully if the retained
    filesystem floor is violated;
12. select `runtime.storage.warm_start` deliberately. Use `{"kind":"none"}`
    for first mapping and all wheels-off qualification. Production's typed
    **Finalize map & stop** operation is terminal even under `none`: it drains
    the exact session, publishes its final session-local occupancy, and
    atomically writes the manifest/occupancy digest selection for a later
    restart. Only after reviewing that selection may a new production bundle
    select `{"kind":"dataset_replay"}`. Replay reconstructs historical map
    context. Before stereo replay it streams the manifest-bound navigation
    journal in constant memory under the format's 1,048,576-record cap and
    requires the final epoch's accepted map epoch/revision to equal the atomic
    selection. Autonomous motion remains gated until the live tracker proves
    current-map relocalization. The selected manifest hash does not
    content-address every dataset payload (including journal record bytes);
13. stage the exact frontal and profile OpenCV cascade files at the two
    launch-bound paths. Production bootstrap retains and verifies both files
    before hardware acquisition; this binding is not by itself a claim that
    detector execution has succeeded;
14. install the rendered document as
   `/opt/kiko/deployment/nano-agent-launch-v3.json`; and
15. mint and verify the exact offline-install marker, then start
    `kiko-nano-agent.service` manually for qualification. The supplied service
    has no `[Install]` section and therefore is not automatically enabled at
    boot.

Parsing the rendered document proves only structural validity and equality
with its content bindings. Runtime admission must still verify exact
inventory, OAK identity and SuperSpeed readback, model loading, controller
session/receipts, accessory health, calibration, plant evidence, and physical
stop behavior.

The current renderer writes `maximum_usb_speed=SUPER_PLUS` and
`minimum_usb_speed=SUPER`. This requests the fastest DepthAI USB-3 mode while
requiring at least a USB-3 link. Runtime retains the exact requested maximum,
required minimum, and observed readback. Older retained launch documents that
explicitly cap both values at `SUPER` remain parseable as capped 5 Gbit/s
inputs; they are not silently promoted.

Do not expand these templates with shell substitution. Feed exact discovery
and reviewed source paths to `kiko-nano-bundle-renderer`; it emits the agent
policy, production controller, inventory, navigation-actuation,
native-runtime, and launch documents from domain types. Digest arrays and
hexadecimal digests are derived from the same retained bytes, and the launch
document is written last.

The illustrative `bundle-render-input-v1.json.template` is the production
render-input schema-V1 source. Replace its whole-value
`${FACE_PERCEPTION_ASSETS_JSON}` token with the complete object below, using
canonical absolute source paths. The strict renderer rejects production
without both distinct assets.

The wheels-off renderer uses the separate exact schema-V4 source at
`configs/nano-wheels-off-qualification-template/bundle-render-input-v4.json.template`.
It fixes the qualification bundle kind, exact face-cascade and optional
proposal-only head-gaze inputs, cold-start selection, and reviewed SONAMEs while
requiring `qualification_executable_path`, the canonical absolute path of the
exact Linux-aarch64 executable retained into the qualification bundle. The
strict renderer rejects cross-version or bundle-specific fields instead of
silently reinterpreting them.

```json
{
  "frontal_face_cascade": {
    "source_path": "/canonical/absolute/path/haarcascade_frontalface_default.xml",
    "destination_relative_path": "models/opencv/haarcascade_frontalface_default.xml"
  },
  "profile_face_cascade": {
    "source_path": "/canonical/absolute/path/haarcascade_profileface.xml",
    "destination_relative_path": "models/opencv/haarcascade_profileface.xml"
  }
}
```

Both bundle kinds must provide the exact regular bytes for the current Nano's
three directly linked OpenCV libraries as `opencv_core`, `opencv_imgproc`, and
`opencv_objdetect`: the attended wheels-off binary also contains the
production dispatch. Their accepted SONAMEs are pinned in the template and
typed renderer. Both production and V4 wheels-off qualification require the
two exact, distinct face-cascade leaves because both launch the common
accessory graph. This direct closure does not make unstaged transitive OpenCV
or OS libraries hermetic; complete the final-ELF loader-trace review in
`docs/nano-qualified-deployment.md` before treating the runtime as ready.

## Native build and manual service installation

Build the exact reviewed commit on Linux aarch64 with the lockfile. This base
command enables the production entry point but makes no provider-performance
claim:

```bash
cargo build --locked --release -p kiko-nano-deployment-gate --bin kiko-nano-deployment-gate
cargo build --locked --release -p kiko-slam --features nano-agent --bin kiko-slam --bin kiko-nano-deployment-qualify
```

If the launch document requests a feature-gated ONNX provider, add only its
reviewed Cargo feature and verify that exact provider on the Nano. A requested
provider is not proof that it loaded or improved performance.

With the service stopped, prepare the deployment root and install the binary,
this README, and the unit as root-owned files:

```bash
sudo install -d -o root -g root -m 0755 /opt/kiko
sudo install -d -o root -g root -m 0755 /opt/kiko/bin /opt/kiko/deployment /opt/kiko/deployment/lib
sudo install -o root -g root -m 0755 target/release/kiko-slam /opt/kiko/bin/kiko-slam
sudo install -o root -g root -m 0755 target/release/kiko-nano-deployment-gate /opt/kiko/bin/kiko-nano-deployment-gate
sudo install -o root -g root -m 0444 configs/nano-agent-template/README.md /opt/kiko/deployment/README.md
sudo install -o root -g root -m 0644 deploy/systemd/kiko-nano-agent.service /etc/systemd/system/kiko-nano-agent.service
sudo systemctl daemon-reload
sudo systemd-analyze verify /etc/systemd/system/kiko-nano-agent.service
```

Install the rendered launch document and every referenced asset separately
from a reviewed staging bundle; never install the tokenized template as JSON.
Record their exact byte identities before publication and do not edit the
deployment tree while the service is running.

Install `deploy/systemd/kiko-nano-agent-qualified-boot.conf` before offline
qualification because its exact bytes are marker-bound, but do not enable the
service during the manual wheels-off phase. The base unit itself always runs
the verifier and deliberately remains non-enableable. Only after the complete
wheels-off evidence has been reviewed may boot enablement be requested as
described in `docs/nano-qualified-deployment.md`.

The production unit invokes
`/opt/kiko/bin/kiko-slam nano-agent`. That process owns the admitted OAK,
accessories, control API, and in-process typed STM32 lifecycle. The repository
no longer ships a standalone `kiko-robot-server.service` or the former
wheels-off bench unit. Do not kill an older installed unit or any endpoint
owner: inspect ownership and let exact endpoint/serial acquisition fail closed
if anything still competes. A conflicting process or automatic launcher is a
failed precondition; do not disable, signal, or kill it from this launch
workflow. Resolve the other workload separately, then repeat the read-only
owner check. An older installation may still have the retired standalone unit
enabled; production must remain stopped until that separate installation has
been deliberately resolved.

After the immutable bundle, exact device admission, independent power cut,
and wheels-off fault prerequisites have been reviewed, start production
explicitly:

```bash
systemctl is-active kiko-robot-server.service
sudo systemctl start kiko-nano-agent.service
sudo systemctl --no-pager --full status kiko-nano-agent.service
sudo journalctl -u kiko-nano-agent.service -b --no-pager
```

The first command must report `inactive` (or `unknown` when the retired unit is
not installed). While only the manual base unit is installed, never use
`systemctl enable`. Boot enablement is permitted only through the later exact
qualified drop-in and marker procedure. An active service proves only that the
process has not exited; the application's typed readiness and wheels-off
evidence remain authoritative.

The operator console deliberately listens only on Nano loopback. From the
operator computer, create an SSH tunnel and leave it open:

```bash
ssh -N \
  -L 9877:127.0.0.1:9877 \
  -L 9876:127.0.0.1:9876 \
  makerspace@NANO_IP
```

In a separate authenticated Nano shell, read the fresh capability owned by
that boot:

```bash
cat /run/kiko/operator-console.capability
```

Then open `http://127.0.0.1:9877/` locally and paste the exact 64-hex
capability. Do not put the capability in a command-line URL, shell history,
browser storage, screenshot, or log. It is regenerated on each process start
and removed only after the HTTP owner stops. The page and agent API share the
same downstream owner; neither opens a second camera or motor transport.
The second forwarding rule carries the launch-bound loopback Rerun stream on
the same local port; it grants no control authority.
The same page exposes the typed **Finalize map & stop** operation; it closes
capture and ends the control process after selecting the exact restart inputs.
It does not silently change the running bundle's warm-start policy. Review the
published selection and render a new immutable production bundle when restart
replay is intended.
