# Nano calibration preparation

`kiko-nano-calibration-prepare` is an offline, fail-closed assembler for the
canonical `NanoCalibrationArtifactV1` and the two duplicated calibration
sections in `navigation-shadow-v2.json`. It opens no hardware, performs no
calibration, and grants no motion authority.

The input template is
`configs/nano-agent-template/calibration-preparation-v1.json.template`. Every
source identifier and lowercase SHA-256 is required. The exact expanded input
is copied into the new output directory beside the generated files so the
source claims and transformation values are retained for review.

The recommended navigation input is
`configs/nano-wheels-off-qualification-template/navigation-shadow-preparation-v2.json.template`.
It covers the complete shadow-navigation schema, fixes the schema versions,
the fail-closed `blocked` unknown-space policy, and the exact checked-in
qualification-only synthetic shadow plant. That plant is deliberately not
physical identification; fixing it here avoids a circular demand for
wheel-on measurements before Gate A permits wheel attachment. The template
also fixes one conservative, shadow-only navigation policy: 50 ms
plant/MPC/control periods, a 100 ms shadow lease, candidate-compatible
±30%/5% PWM bounds, a search lattice that can leave STOP, bounded odometry and
costmap freshness, and explicit solver/journal limits. Those values are
software admission choices, not tuned navigation or performance evidence.

The operator geometry declarations currently available are retained in
`docs/kiko-assembly-geometry-declaration-2026-08-27.md`. They include a
0.293 m physical footprint radius, 0.05 m drive-axle midpoint height, and an
OAK-housing offset whose “centre” datum remains ambiguous. The footprint is
not yet a margin-inflated navigation radius, and the housing measurement is
not yet an optical-centre-to-base transform. The preparer must continue to
reject unresolved camera/base and native-IMU rotations rather than infer them.

Only seven physically dependent `NAV_SHADOW_UNVALIDATED_*` leaves remain:
world-to-occupancy rotation and translation, footprint radius, the minimum
and maximum floor-relative global occupancy heights, and the minimum and
maximum axle-relative base-frame obstacle `z` values. Render those seven from
retained physical evidence before invoking the assembler. Do not copy one
height pair into the other: the base origin is the drive-wheel axle midpoint,
not the floor. Leave the two quoted
`CALIBRATION_PREPARER_REPLACES_*` placeholders in place: the assembler
replaces those complete values before the production parser sees the
navigation document. They must be the exact complete string values at
`coordinate_frames.tracking_camera_to_base` and
`odometry.raw_imu_calibration`; already-filled values are rejected rather than
silently overwritten.

`configs/navigation-shadow-v2.example.json` is a synthetic parser/test fixture.
It is not a recommended preparation input and must not be used for a
qualification or deployment render.

Navigation V1 is not accepted by the live parser or qualification path.
Its single `local_costmap.obstacle_height_{minimum,maximum}_m` pair did not
declare whether zero meant the floor or the axle-centred base origin, and the
runtime consumed it in both frames. Migrate only from the retained physical
measurement record:

- write floor-relative bounds to
  `global_occupancy.obstacle_floor_height_{minimum,maximum}_m`; and
- write axle-midpoint-base-frame bounds to
  `local_costmap.obstacle_base_z_{minimum,maximum}_m`.

There is deliberately no compatibility fallback that duplicates the old
numbers into both fields.

Both obstacle slabs are closed intervals: an endpoint exactly equal to either
the minimum or maximum is classified as an obstacle. A measured floor plane is
therefore not a valid minimum by itself. Select each minimum above the highest
plausible floor return in that field's own frame, including retained depth
noise, extrinsic uncertainty, floor unevenness, and the intended low-obstacle
clearance margin. In particular,
`local_costmap.obstacle_base_z_minimum_m` must be greater than the most
positive plausible floor `z` in the axle-midpoint base frame; setting it equal
to nominal floor `z` makes nominal floor endpoints occupied. Retain the
measurement, uncertainty calculation, and chosen margin. Apply the same
evidence discipline to the maximum so over-robot points are excluded without
excluding obstacles Kiko can collide with.

Run the assembler only after replacing every required preparation and
navigation token:

```text
cargo run --locked -p kiko-slam \
  --features nano-calibration-prepare \
  --bin kiko-nano-calibration-prepare -- \
  --input PREPARATION.json \
  --navigation-template NAVIGATION-SHADOW.json \
  --output-dir NEW-EMPTY-DIRECTORY
```

The source identifiers and digests are caller declarations. The assembler
does not open the named source records or recompute their SHA-256 values.
Calculate each digest from the exact retained source file before preparing the
input; do not transcribe a digest from notes.

The assembler:

- rejects unknown, missing, trailing, oversized, non-finite, singular, and
  unsupported input;
- rejects every unresolved `${...}` token in the preparation input and every
  navigation token except the two exact replacement markers above; it checks
  bounded raw bytes before JSON parsing and decoded string values afterward,
  so JSON escaping cannot hide a token in an identifier;
- rejects duplicate object keys recursively in the navigation template before
  constructing its mutable JSON value, so neither top-level nor nested
  last-key-wins collapse can alter the admitted declaration;
- requires an explicitly sourced proper native-IMU-to-base rotation and
  tracking-camera-to-base rigid transform;
- requires closed Basalt parameter-unit declarations: acceleration
  `b[0..2]` is `metres_per_second_squared`, angular-velocity `b[0..2]` is
  `radians_per_second`, and every `s` affine parameter is `dimensionless`;
- converts Basalt `CalibAccelBias` and `CalibGyroBias` parameters from
  `A * raw - b` into Kiko's `A * (raw - bias)` representation by solving
  `A * bias = b` with pivoting;
- requires the caller to declare the corroborating baseline
  `independently_derived`, rejects a corroborating source with either the same
  source identifier or the same content SHA-256 as the live stereo source,
  and requires the two baseline values to agree within 2%, using the live
  value in the artifact;
- parses both generated documents through the production domain parsers and
  requires their raw-IMU and camera/base values to match bit-for-bit.

The typed independence declaration is a retained operator claim, not proof
that the derivations are independent. Distinct identifiers and content hashes
are necessary admission checks, not sufficient calibration evidence. For
example, relabelling two values extracted from the same OAK EEPROM does not
make one an independent corroboration.

The 2% comparison is a consistency gate, not a calibration tolerance or a
quality claim. Runtime admission remains stricter: the artifact's rectified
stereo model must match the connected OAK observation bit-for-bit.

Output publication is transactional within the output path's parent
filesystem. The assembler creates a unique sibling staging directory, writes
and syncs all three complete files, syncs that directory, then atomically
renames it with the operating system's no-replace flag to the requested absent
output path and syncs the parent directory. The no-replace operation also
rejects an empty directory or other destination created after the earlier
absence checks; there is no check/rename overwrite race. Failures before or
during the rename remove the staging directory; a cleanup failure is reported
alongside the original typed publication phase. If the final parent sync
fails, the error says that a complete output is already visible but its
directory-entry durability is unconfirmed. Existing output paths are never
replaced.

## Retained Nano evidence and baseline selection

Read-only audit on 2026-07-27 found these relevant retained sources:

- factory calibration:
  `/home/makerspace/calibration/oakd-basalt/factory_calibration_full.json`,
  SHA-256
  `8a6261456674796dcda3ece150196b6e234176e309feab2d2712908aa787ede3`;
- raw dataset calibration:
  `/home/makerspace/calibration/oakd-basalt/datasets/imu_dynamic/calibration.json`,
  SHA-256
  `15c7d334857d6e5291e02219484b8e8d44d8da0b3525adce902f415038f6ccd9`;
- Basalt stable-first-pass candidate:
  `/home/makerspace/calibration/oakd-basalt/results_rectified_imu_seed/calibration_recommended_first_pass.json`,
  SHA-256
  `1361d60c4cad7e5c4aa0d43945e6eeea72fd23922759be42281c094620c58c27`;
- Basalt notes:
  `/home/makerspace/calibration/oakd-basalt/results_rectified_imu_seed/NOTES.md`,
  SHA-256
  `f2a91a286fad0c10ccf9a771c611e00589dc853c6df16bb7ac72ac8e7df081c2`.

These are candidates, not a canonical artifact or a calibration-quality
certificate.

The retained recorder labels the DepthAI result in metres and stores
`0.075 m`. Its later EuRoC converter computes the OpenCV rectified projection
baseline and then divides by `1000`, producing
`0.0075033944182863015 m`. The corresponding retained dynamic conversion
summary has SHA-256
`e2d9e048508762f968810ca0eea0613a78344047e94ec22f9f4507937780f53a`.
That tenfold disagreement is rejected by the assembler and the affected
conversion summary must not be used as canonical baseline evidence.

Fresh evidence on 2026-07-29 supersedes that defective conversion as the
selected baseline pair:

- live rectified stereo:
  `/home/makerspace/kiko-native-evidence/4238f14-calibration-stereo-20260729T001905+0530/calibration.json`,
  SHA-256
  `13cdf7c81036b9983f4f12f51b4353295468b88de896ec0d7bd3a37a1147503e`,
  baseline `0.07503394 m`; and
- independent static OAK-D S2 board design:
  `/home/makerspace/kiko-native-evidence/4238f14-oak-s2-board-baseline-20260729T005342+0530/OAK-D-S2.json`,
  SHA-256
  `b6c50050a9d45bd28d76102cfc44ff399591f10f5518b0eedc47d15b83c28281`,
  baseline `0.075 m`.

The design declaration comes from clean `depthai-boards` commit
`1dda9d332864c9139616282d63b043cab0ee65fa` and is not a re-export of the
connected device's EEPROM/API value. Exact decimal arithmetic gives a relative
discrepancy of `0.045232864%`; the preparer's parsed `f32` inputs give
`0.0452294%`. Both are below its 2% gate. Exact interpretation and claim
limits are recorded in `docs/nano-oak-s2-baseline-evidence-2026-07-29.md`.

## Exact live-capture-to-preparer transaction

Use one fresh `kiko-slam record` dataset to bind the connected OAK observation
to the values selected for the V4 runtime graph. This is an observation and
provenance transaction, not a camera- or IMU-calibration algorithm.

Before capture:

1. require the exact expected MXID and SuperSpeed connection with no competing
   OAK owner;
2. select the final V4 stereo width, height, FPS, and IMU rate; and
3. choose a new absent dataset path. The recorder deliberately refuses to
   reuse an existing dataset directory.

Run the exact release executable selected for the candidate bundle. Substitute
literal numeric values from the same V4 render input; do not rely on defaults
or a second configuration file:

```text
KIKO_RECORD_DEPTH=1 \
  /path/to/exact/kiko-slam record /new/evidence/dataset \
  --oak-device-id EXACT_MXID \
  --width EXACT_STEREO_WIDTH_PX \
  --height EXACT_STEREO_HEIGHT_PX \
  --fps EXACT_STEREO_FPS \
  --rectified \
  --imu-rate-hz EXACT_IMU_RATE_HZ
```

Stop with one normal shutdown signal only after paired stereo, rectified-left
depth, and IMU records have been observed. Admit the capture only if the
recorder reports successful finalization and device close. Retain the complete
dataset, invocation, log, source revision, executable SHA-256, `meta.json`,
`calibration.json`, and finalized dataset manifest. An interrupted, dropped,
writer-failed, unfinalized, or device-close-failed capture is evidence of that
failure, not calibration input.

Raw IMU capture does not require or infer an EEPROM IMU-to-camera extrinsic.
The dataset declares its raw IMU extrinsic provenance as
`uncalibrated_unknown`, and `calibration.json` may therefore omit `oak_eeprom`
even when the dataset contains IMU reports. This is the truthful state for a
supported device whose vendor API reports that IMU calibration data is not
available. The separately sourced native-IMU-to-base proper rotation and
Basalt calibration remain mandatory preparer inputs; successful raw capture
must never be relabelled as either one.

Map the retained `calibration.json` into `rectified_stereo` exactly once:

- `left.{fx,fy,cx,cy,width,height}` becomes
  `left.{fx_px,fy_px,cx_px,cy_px,width_px,height_px}`;
- the right camera maps identically;
- `rectified` and `baseline_m` retain their exact serialized values; and
- the SHA-256 of those exact `calibration.json` bytes is
  `rectified_stereo.provenance.source_sha256_hex`.

When present, the embedded `oak_eeprom` matrices remain raw vendor-API
evidence. In particular, `imu_to_camera_b_m` is not a native-IMU-to-base
rotation and the raw left-rectification matrix has no asserted transform
direction. Their absence is not filled with a default, and neither may be
silently relabelled into the two physical transforms required by the
preparer.

Use the retained static OAK-D S2 board declaration above as the corroborating
baseline source. Its exact bytes, identifier, design method, centimetre units,
source revisions, operator model identification, and SHA-256 populate the
corroborating fields. The preparer relationship remains
`independently_derived`; re-exporting, reformatting, or renaming the live OAK
value would not satisfy that relationship.

The current preparer retains caller-supplied provenance declarations but does
not reopen the source files. Therefore the release review must recompute both
source hashes from the retained bytes, compare every mapped stereo value to
`calibration.json`, and retain that review beside the preparer output. Any
mismatch closes the gate. The generated calibration and navigation documents
are subsequently parsed and cross-bound by the assembler and qualification
bootstrap; that later binding does not replace this source-byte review.

## Remaining physical inputs

The Nano files do not establish either of these required facts:

1. the rigid transform from the rectified-left tracking-camera optical frame
   (`+x` right, `+y` down, `+z` forward) into the robot base frame whose origin
   is the drive-wheel axle midpoint (`+x` forward, `+y` left, `+z` up); or
2. the proper rotation from the raw OAK IMU axes emitted by DepthAI into that
   base frame.

The previously supplied head-centre geometry—0.25 m above and 0.20 m behind
the OAK—is not the camera-to-base transform. Do not substitute it.

Before preparing the canonical files, retain a measurement record that states
the camera optical-centre offsets from the wheel-axle midpoint in metres
(forward, left, up), the camera mounting orientation, the IMU-axis derivation,
the assembly identity, method, date, and operator. Put the exact record hashes
and derived proper rotations in the preparation input. A fresh canonical OAK
capture must supply the exact rectified intrinsics, dimensions, and baseline
used by the runtime graph.
