# Meta Review

## Commit Goal

Make runtime IMU calibration override explicit and truthful in dataset/live modes. This tranche adds a typed runtime IMU override path that can replace only the IMU block from either `KIKO_IMU_CALIBRATION_FILE` or `KIKO_IMU_*` env, rejects ambiguous mixed sources, and supports direct Basalt import without silently collapsing unsupported higher-order bias terms into a fake simple bias.

## Current Repo Starting Point

The code already had an honest VIO proposal path and an optional calibrated bias prior representation, but dataset SLAM still loaded IMU calibration only from the dataset `calibration.json` while its own error/help text claimed `KIKO_IMU_*` runtime support. `record` already had an env-based IMU block builder, but that logic was duplicated, limited to direct env vars, and unavailable to dataset replay.

## Previous Invariants

- The dataset replay reader applied IMU time offset before interval selection, so tracker-side replay must not reapply it.
- `record` could synthesize an IMU calibration block from `KIKO_IMU_*` env, but dataset `slam` could not.
- The runtime had no explicit typed path for “override only the IMU block while preserving dataset visual calibration”.
- Basalt calibration results existed on disk, but there was no principled ingestion path for them.

## New Invariants Claimed

- Dataset `slam` and live/record paths share one runtime IMU override loader instead of maintaining separate semantics.
- `KIKO_IMU_CALIBRATION_FILE` and direct `KIKO_IMU_*` env cannot be combined in one run; ambiguity is rejected at parse time.
- Runtime IMU override replaces only the dataset IMU block and preserves the dataset’s visual intrinsics/baseline/rectification.
- Basalt import inverts `T_imu_cam[0]` into runtime `T_cam_imu` explicitly.
- Basalt import refuses anisotropic noise vectors because the runtime model currently only represents isotropic scalar noise.
- Basalt import refuses non-zero higher-order bias calibration terms instead of silently truncating them into a 3-vector bias.

## Touched Files

- `crates/kiko-slam/src/runtime_imu.rs`
- `crates/kiko-slam/src/lib.rs`
- `crates/kiko-slam/src/bin/kiko_slam/record.rs`
- `crates/kiko-slam/src/bin/kiko_slam/slam.rs`
- `crates/kiko-slam/src/bin/kiko_slam/live.rs`

## New Or Changed Metrics

- none

## New Or Changed Solver Outputs

- none

## Tests Added

- `runtime_imu::tests::direct_env_override_requires_complete_block`
- `runtime_imu::tests::direct_env_override_builds_imu_calibration`
- `runtime_imu::tests::file_and_direct_env_sources_conflict`
- `runtime_imu::tests::basalt_override_inverts_left_camera_extrinsics_and_extracts_biases`
- `runtime_imu::tests::apply_override_preserves_visual_calibration_and_replaces_imu_block`
- `slam::tests::slam_env_help_mentions_runtime_imu_override_file_env`

## Tests Run

- `cargo test -p kiko-slam --manifest-path /home/makerspace/kiko-vio/Cargo.toml`
- `cargo test -p kiko-slam --features vio --manifest-path /home/makerspace/kiko-vio/Cargo.toml`
- `cargo test -p kiko-slam --features 'record,vio' --manifest-path /home/makerspace/kiko-vio/Cargo.toml --no-run`
  This failed for an environment reason outside the new code: `oak-sys` could not find `depthai/depthai.hpp` on this machine.

## Known Risks Or Deferred Follow-Ups

- The runtime override path is explicit, but it does not yet auto-discover recommended calibration artifacts. The operator must still point `KIKO_IMU_CALIBRATION_FILE` at the intended file.
- Basalt import currently accepts only the simple “bias terms present, higher-order terms zero” case. That is intentional, but full scale/misalignment calibration support would need a richer runtime IMU model.
- The new tests mutate process environment under a serialized mutex because Rust 2024 makes env mutation `unsafe`. That keeps the tests lawful, but environment-global testing remains inherently less isolated than pure parsing tests.
- Review agents were requested for the tranche process, but no automated findings returned before timeout. Findings below are from direct code review and green verification.

## Findings

- none blocking in this tranche
- follow-up: log the active IMU override source at startup so dataset runs show whether they are using dataset calibration, direct env, or a file override
- follow-up: if we standardize around Basalt first-pass files operationally, add a first-class operator doc with the exact `KIKO_IMU_CALIBRATION_FILE` invocation

## Invariant Verdict

- strengthened: explicit runtime source selection, no ambiguous mixed overrides, no silent visual-calibration replacement, principled Basalt transform/bias import
- weakened or ambiguous: no new weakening in code semantics; remaining ambiguity is operational selection of which calibration artifact to point at

## Metric Verdict

- trustworthy: unchanged from the previous tranche; this work is calibration ingestion, not a metric change
- partial or misleading: the runtime still does not expose a scalar saying which IMU override source is active

## Test Verdict

- covered: direct env parsing, conflict rejection, Basalt conversion, IMU-block-only merge semantics, dataset SLAM help text
- missing: an integration test that loads an actual calibration file path through `KIKO_IMU_CALIBRATION_FILE` end-to-end in `slam`

## Merge Decision

`accept with follow-up`
