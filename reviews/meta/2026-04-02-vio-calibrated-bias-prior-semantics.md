# Meta Review

## Commit Goal

Remove the silent false certainty around IMU bias initialization. This tranche changes the type surface so a calibrated VIO bias prior only exists if there is an actual calibrated accel+gyro bias, instead of silently turning missing calibration into a strong prior toward zero.

## Current Repo Starting Point

The previous commit made VIO proposal adoption truthful and transactional. After that landed, the next major code-side risk was still inside the inertial configuration surface: `tracker.rs` passed `[0; 3]` accel/gyro fallback arrays into `VioSolveConfig` and always enabled a strong all-frame bias prior, even when calibration did not contain a lawful initial bias estimate.

## Previous Invariants

- VIO proposal adoption was transactional and judged on exact shared projectable tracked support.
- The VIO solver reported accepted-state cost and per-factor breakdown honestly.
- Calibration could omit IMU bias initialization, but the VIO config still represented that absence as a strong prior toward zero.

## New Invariants Claimed

- A calibrated VIO bias prior is represented as `Option<VioBiasPrior>`, so “strong prior with no calibrated bias” is no longer representable.
- `CalibrationBundle` stores initial IMU bias as a single `Option<ImuBias>`; partial accel-only or gyro-only initial bias blocks are rejected at parse time.
- Tracker VIO initialization only installs a strong bias prior when calibration supplies a complete initial bias estimate.
- Missing initial bias still allows VIO to run, but only as “no calibrated bias prior”, not as a fake zero-bias prior.

## Touched Files

- `crates/kiko-slam/src/calibration.rs`
- `crates/kiko-slam/src/local_ba.rs`
- `crates/kiko-slam/src/tracker.rs`
- `crates/kiko-slam/src/lib.rs`

## New Or Changed Metrics

- none

## New Or Changed Solver Outputs

- `VioBiasPrior`
- `VioSolveConfig` now carries `anchor_bias_prior: Option<VioBiasPrior>` instead of raw bias-info plus fallback bias arrays

## Tests Added

- `calibration::tests::from_dataset_calibration_preserves_complete_initial_bias`
- `calibration::tests::from_dataset_calibration_rejects_partial_initial_bias`

## Tests Run

- `cargo test -p kiko-slam --manifest-path /home/makerspace/kiko-vio/Cargo.toml`
- `cargo test -p kiko-slam --features vio --manifest-path /home/makerspace/kiko-vio/Cargo.toml`

## Known Risks Or Deferred Follow-Ups

- This change removes a false prior, but it does not yet provide a better bias estimate. Static initialization or calibrated bias ingestion is still needed for best VIO performance.
- Gravity initialization is still brittle and remains a separate follow-up.
- IMU-camera extrinsics and time-offset calibration remain dominant real-world error sources if dataset calibration is still placeholder quality.
- Review agents were requested for the tranche process, but no automated findings returned before timeout. Findings below are from direct code review and the green test matrix.

## Findings

- none blocking in this tranche
- follow-up: add explicit observability for whether a calibrated bias prior is active, so dataset runs can confirm they are not accidentally operating without one

## Invariant Verdict

- strengthened: complete-bias-or-none parsing, typed optional bias-prior semantics, no more silent strong prior toward zero
- weakened or ambiguous: VIO still falls back to zero initial state bias when calibration has none, but that fallback is no longer mislabeled as a calibrated prior

## Metric Verdict

- trustworthy: unchanged from the previous tranche
- partial or misleading: there is still no direct scalar that says “calibrated bias prior active/inactive” in runtime observability

## Test Verdict

- covered: calibration parsing for complete and partial initial bias blocks; full default and `vio` test matrices
- missing: integration test proving tracker initialization disables the bias prior when calibration lacks initial bias

## Merge Decision

`accept with follow-up`
