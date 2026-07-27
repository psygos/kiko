# Nano calibration preparation

`kiko-nano-calibration-prepare` is an offline, fail-closed assembler for the
canonical `NanoCalibrationArtifactV1` and the two duplicated calibration
sections in `navigation-shadow-v1.json`. It opens no hardware, performs no
calibration, and grants no motion authority.

The input template is
`configs/nano-agent-template/calibration-preparation-v1.json.template`. Every
source identifier and lowercase SHA-256 is required. The exact expanded input
is copied into the new output directory beside the generated files so the
source claims and transformation values are retained for review.

The recommended navigation input is
`configs/nano-wheels-off-qualification-template/navigation-shadow-preparation-v1.json.template`.
It covers the complete shadow-navigation schema, fixes only schema versions
and the fail-closed `blocked` unknown-space policy, and leaves every
configuration or physically identified value as an explicitly named
`UNVALIDATED` token. Render every `NAV_SHADOW_UNVALIDATED_*` token from
retained, reviewed evidence before invoking the assembler. Leave the two
quoted `CALIBRATION_PREPARER_REPLACES_*` placeholders in place: the assembler
replaces those complete values before the production parser sees the
navigation document. They must be the exact complete string values at
`coordinate_frames.tracking_camera_to_base` and
`odometry.raw_imu_calibration`; already-filled values are rejected rather than
silently overwritten.

`configs/navigation-shadow-v1.example.json` is a synthetic parser/test fixture.
It is not a recommended preparation input and must not be used for a
qualification or deployment render.

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

## Retained Nano evidence and the baseline defect

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
