# Meta Review

## Commit Goal

Make runtime IMU calibration override authoritative on both replay and live paths so replayed IMU timestamps are shifted and interval-selected under the same calibration that VIO actually uses, and so a bad embedded dataset `imu` block does not prevent a lawful runtime override from becoming the source of truth.

## Current Repo Starting Point

Before this change, dataset `slam` opened `DatasetReader`, which loaded and time-shifted replay IMU samples using the embedded dataset calibration before any runtime IMU override from `KIKO_IMU_CALIBRATION_FILE` or `KIKO_IMU_*` env was applied. Live mode advertised the same override env vars but did not apply them at all.

## Previous Invariants

- Replay IMU interval selection is based on the timestamps loaded by `DatasetReader`.
- Runtime IMU override is intended to replace only the IMU calibration block, not the visual calibration.
- VIO metrics are only trustworthy if replay timestamps and runtime calibration refer to the same IMU model.

## New Invariants Claimed

- On dataset replay, the active runtime IMU override becomes authoritative before calibration deserialization of the `imu` subtree, before IMU timestamp shifting, and before bundle interval selection.
- On live capture, the active runtime IMU override is applied before extracting `imu_time_offset_ns` for device IMU timestamp shifting.
- A non-deserializable embedded dataset `imu` block no longer prevents a valid runtime override from becoming the authoritative IMU source.

## Touched Files

- `/home/makerspace/kiko-vio/crates/kiko-slam/src/dataset/mod.rs`
- `/home/makerspace/kiko-vio/crates/kiko-slam/src/dataset/reader.rs`
- `/home/makerspace/kiko-vio/crates/kiko-slam/src/bin/kiko_slam/slam.rs`
- `/home/makerspace/kiko-vio/crates/kiko-slam/src/bin/kiko_slam/live.rs`

## New Or Changed Metrics

- none

## New Or Changed Solver Outputs

- none

## Tests Added

- `dataset::reader::tests::runtime_imu_override_reshifts_loaded_samples_before_bundle_selection`
- `dataset::reader::tests::open_with_runtime_imu_override_applies_override_before_loading_samples`
- `dataset::reader::tests::open_with_runtime_imu_override_ignores_non_deserializable_embedded_imu_block`

## Tests Run

- `cargo test -p kiko-slam --features vio --manifest-path /home/makerspace/kiko-vio/Cargo.toml --quiet`
- `bash /home/makerspace/kiko-vio/scripts/check_meta_review.sh`

## Known Risks Or Deferred Follow-Ups

- This commit does not yet fix tracker-side IMU interval continuity; `pending_imu` and per-capture preintegration semantics still need their own invariant pass.
- This commit does not change gravity or bias initialization, only calibration authority and replay/live consistency.

## Findings

- none in final reviewed scope

## Invariant Verdict

- strengthened: runtime IMU override authority on replay now begins before calibration deserialization, replay timestamp shifting, and interval selection
- strengthened: live and replay both apply runtime IMU override before deriving IMU time offset for timestamp shifting
- strengthened: invalid embedded dataset IMU blocks no longer poison a valid runtime override
- weakened or ambiguous: none in reviewed scope

## Metric Verdict

- trustworthy: no metric semantics changed; this commit improves the trustworthiness of downstream VIO residuals by aligning replay timestamps with the calibration actually used by VIO
- partial or misleading: none introduced in reviewed scope

## Test Verdict

- covered: replay override affects bundle timestamps, override authority precedes sample-load overflow failure, and override can replace a non-deserializable embedded IMU block
- missing: dataset-level end-to-end replay trace showing changed per-frame VIO cost on a real recording remains a follow-up, not a blocker for this calibration-authority patch

## Merge Decision

`accept`
