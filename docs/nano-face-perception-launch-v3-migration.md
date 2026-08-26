# Nano face-perception launch V3 migration boundary

Historical migration record. Canonical production now uses launch V4, which
retains this complete V3 face-perception graph and adds mandatory exact
physical-head-gaze policy/review bindings. V3 remains an explicit compatibility
parser/loader and must not be selected by current production bootstrap.

Status at completion of this historical migration: production rendering,
offline qualification, systemd startup, and bootstrap used V3. Production
bootstrap verified and retained both
face-cascade assets before hardware acquisition. The production accessory
startup then moves those exact retained buffers into a named perception
thread, constructs the detector there, and requires typed detector readiness
before starting either the eye actor or the natural-head-hold actor. Nano
target detection behavior and latency are still unqualified.

`NanoAgentLaunchV3` extends the unchanged V2 runtime graph with one mandatory
`face_perception` object:

```json
{
  "face_perception": {
    "frontal_face_cascade_asset": {
      "relative_path": "models/opencv/haarcascade_frontalface_default.xml",
      "maximum_bytes": 1048576,
      "sha256_hex": "<lowercase SHA-256 of exact installed bytes>"
    },
    "profile_face_cascade_asset": {
      "relative_path": "models/opencv/haarcascade_profileface.xml",
      "maximum_bytes": 1048576,
      "sha256_hex": "<lowercase SHA-256 of exact installed bytes>"
    }
  }
}
```

Both paths parse into canonical deployment-relative identities. Each file has a
nonzero launch bound, a 4 MiB role ceiling, an exact SHA-256 identity, and must
have a different path and content digest from the other cascade. Both paths
must also be distinct from all existing launch inputs. The V3
loader also rejects either path if it aliases the launch document. V2 retains
its own strict DTO, parser, type, and loader; V2 neither accepts nor implies
these assets.

`LoadedNanoBootstrapFacePerceptionAssets` loads both bindings through the
existing no-follow deployment loader and retains the exact verified bytes,
digest, and relative path. Bootstrap separates this move-only value from the
model/runtime assets, consumes it into `NanoFacePerceptionAssets`, and does not
return a stale second asset owner in `PreparedNanoBootstrap.assets`. No
missing-value or V2 fallback state reaches production hardware startup.

## Native retained-byte boundary

The canonical `oak-sys` constructor consumes borrowed frontal and profile XML
buffers from the already verified bootstrap assets. It rejects empty,
embedded-NUL, and larger-than-4-MiB payloads before entering C++, copies each
slice into C++-owned memory, and uses OpenCV `FileStorage` in `READ | MEMORY`
mode followed by `CascadeClassifier::read` on the first top-level node. Neither
classifier retains the Rust slices, and there is no later pathname reopen or
pathname TOCTOU assumption.

The deployment path, size, and SHA-256 bindings still matter: they prove which
bytes the bootstrap retained. No environment variable, Python package data
directory, home directory, or caller-supplied arbitrary path may replace those
bindings. This removes the pathname race. A Nano-native C++17 probe using
system OpenCV 4.5.4 subsequently read the exact 930,127-byte frontal asset
(`0f7d4527…aafb0`) and 828,514-byte profile asset (`b39a4a3b…e612d`) through
this same `FileStorage`/first-node/`CascadeClassifier::read` sequence; both
returned true and nonempty. That proves parser acceptance of those exact
current bytes, but not full `oak-sys` linkage, detector output equivalence,
camera behavior, or runtime latency. Those remain deployment-target
qualification gates.

The production native manifest also binds the detector's three direct OpenCV
libraries under the current Nano's exact 4.5-series SONAMEs. That direct binding
does not hermetically capture objdetect's system DNN/calib3d/features2d/flann
and lower ABI closure. Follow the final-ELF `readelf`/loader-trace gate in
`docs/nano-qualified-deployment.md`; readiness remains withheld until that
target evidence exists.

## Coordinated production migration

The canonical deployment migration moves these coupled consumers together:

1. bundle-renderer input schema, V3 output filename, rendered face assets, and
   renderer tests;
2. the production launch template and its rendering instructions;
3. base commissioning's bound live-graph asset;
4. offline qualifier enumeration, exact loading, marker roles, and readback;
5. Nano bootstrap's launch type and mandatory face-asset load before any
   hardware acquisition;
6. CLI defaults, systemd command line, qualified deployment gate expectations,
   and associated tests; and
7. the dedicated face-perception worker that consumes the exact typed assets
   without opening a second OAK owner.

Items 1–7 then required `nano-agent-launch-v3.json`. V2 remains available only
through its explicit parser/loader compatibility API and tests; production
bootstrap never chooses it or invents face bindings. A successful production
accessory start now proves that the exact two retained assets were accepted by
the in-memory native constructor and that the dedicated detector lane reported
readiness before head/eye actor startup. It does not prove a correct detection
on an OAK frame, detector throughput, end-to-end eye latency, or equivalence to
the Python prototype; those require Nano-native runtime evidence.

## Runtime ownership and failure semantics

The sole live OAK owner moves each RGB allocation into a capacity-one
replace-latest ingress after sampling the shared monotonic clock once.
`kiko-nano-face-perception` parses that ingress metadata once into
`ParsedIngressRgbFrame`, borrows the same `ImageFrame` allocation for OpenCV,
updates `FaceTracker` with `FaceTrackingConfig::default()`, and moves the
parsed frame plus its exact tracking result into a second capacity-one
handoff. No pixel clone, observation retag, second OAK connection, pathname
open, or physical head-gaze command occurs.

The detector is intentionally `!Send` and is both constructed and retained on
that named OS thread; there is no unsafe `Send` implementation and no Tokio
`spawn_blocking` transfer. The head/eye actor remains on its separate
current-thread runtime, so Haar work cannot block natural-hold health checks or
eye protocol servicing. Load failure, per-frame parse/detection/tracking
failure, detector panic, handoff poison/disconnect, expression failure, and
actor failure are distinct typed startup or terminal outcomes. Coordinated
shutdown requests both lanes and retains the detector-thread exit in accessory
shutdown evidence. Detector startup readiness is bounded to 10 seconds.
Coordinated shutdown first joins eye/head cleanup, then gives the
hardware-free detector only the time remaining in the original 2-second face
join deadline. If it is still running, shutdown records
`DetachedAfterTimeout`. That disposition is classified as shutdown
uncertainty: it prevents a healthy-cleanup claim and prevents checkpoint
publication. It proves only that the thread had not exited by its deadline; it
does not identify its phase or cancel an OpenCV call. A synchronous native
load/detection may therefore continue consuming CPU after detachment, but the
thread owns no OAK, serial bus, head, eye, or base actuator. Native latency
qualification remains required before these conservative deadlines can be
tuned from evidence.
