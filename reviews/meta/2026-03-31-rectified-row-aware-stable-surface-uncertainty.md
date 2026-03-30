# Meta Review

## Commit Goal

Strengthen stable sparse stereo surface observations before voxel fusion by making per-observation positional uncertainty respond to actual stereo geometry: use a shared midpoint row for the reconstructed point, propagate lateral image uncertainty explicitly, and inflate vertical uncertainty when the rectified stereo match disagrees in row space.

## Current Repo Starting Point

This tranche starts immediately after `1dd01f3` on `measurement-system`, where the surface belief map already required support, consistency, and posterior sigma before a voxel could be rendered. The weak point moved upstream: `generate_stable_surface_points` still used a fixed disparity-only uncertainty prior and the left-image row directly, so stable-point ranking and rejection did not yet account for rectified row disagreement except as a side-channel metric.

## Previous Invariants

- Stable sparse stereo observations used disparity-induced depth uncertainty as the only positional uncertainty model.
- Geometry used the left-image row directly even when the stereo correspondence had a rectified row mismatch.
- `StableSurfaceStats::{mean,max}_accepted_position_sigma_m` reflected only the disparity-driven scalar variance model.
- Rectified row mismatch was exported as telemetry but did not affect the observation uncertainty used for filtering and ranking.

## New Invariants Claimed

- Stable sparse stereo geometry uses the midpoint rectified row of the stereo match rather than the left-image row alone.
- Stable sparse stereo positional variance now includes disparity uncertainty, lateral image-plane uncertainty, and conservative vertical inflation from rectified row disagreement.
- The variance model uses `sample.rectified_row_mismatch_px` as the uncertainty source of truth, so filtering/ranking and exported row-mismatch telemetry are driven by the same validated scalar.
- Retained stable-surface sigma metrics now honestly reflect the stronger uncertainty model instead of the earlier disparity-only approximation.

## Touched Files

- `crates/kiko-slam/src/dense_cloud.rs`

## New Or Changed Metrics

- `StableSurfaceStats::mean_accepted_position_sigma_m` now reflects disparity, lateral pixel, and row-mismatch uncertainty rather than disparity-only uncertainty.
- `StableSurfaceStats::max_accepted_position_sigma_m` now reflects disparity, lateral pixel, and row-mismatch uncertainty rather than disparity-only uncertainty.

## New Or Changed Solver Outputs

- none

## Tests Added

- `dense_cloud::tests::stereo_position_variance_increases_with_rectified_row_mismatch`
- `dense_cloud::tests::stable_surface_rejects_large_row_mismatch_when_uncertainty_exceeds_threshold`

## Tests Run

- `rustfmt --edition 2024 /home/makerspace/kiko-vio/crates/kiko-slam/src/dense_cloud.rs`
- `cargo test -p kiko-slam dense_cloud:: --manifest-path /home/makerspace/kiko-vio/Cargo.toml`
- `cargo test -p kiko-slam --manifest-path /home/makerspace/kiko-vio/Cargo.toml`
- `cargo test -p kiko-slam --features vio --manifest-path /home/makerspace/kiko-vio/Cargo.toml`
- reviewer agent `Euclid`: `accept`

## Known Risks Or Deferred Follow-Ups

- `sigma_v_sq = sigma_feature^2 + 0.25 * mismatch^2` is a conservative inflation rule, not yet a calibrated stereo covariance model.
- `SparseStereoSample` still does not enforce that `right_v` and `rectified_row_mismatch_px` remain algebraically consistent for manually constructed values.
- The interpolated dense visualization path still passes `row_mismatch_px = 0.0`; that is acceptable only because it remains explicitly non-authoritative.
- The new tests prove monotonicity and threshold behavior, but they do not yet assert midpoint-row symmetry or off-axis ordering effects directly.

## Findings

- `major`: stable-point ranking and rejection now depend on the same row-mismatch quantity already exported as telemetry, reducing a real semantic gap between measurement quality reporting and measurement filtering.
- `major`: midpoint-row geometry is more symmetric and less left-camera-biased for rectified stereo than the previous left-row-only reconstruction.
- `medium`: retained position-sigma metrics now changed meaning in a more truthful direction, so downstream interpretation should treat them as stronger uncertainty summaries rather than directly comparable to older runs.

## Invariant Verdict

- strengthened: stable sparse stereo geometry and uncertainty now both acknowledge rectified row disagreement; the uncertainty model includes lateral image uncertainty instead of pretending depth noise is the whole story; filtering and ranking use the same row-mismatch source of truth as telemetry.
- weakened or ambiguous: the uncertainty model remains scalar and conservative rather than fully calibrated or anisotropic; sample-type consistency between `right_v` and stored row mismatch is still not compiler-enforced.

## Metric Verdict

- trustworthy: retained stable-surface sigma metrics are more honest than before because they now encode the row-mismatch-aware uncertainty model actually used for filtering.
- partial or misleading: these sigma metrics are still conservative scalar summaries, not calibrated posterior surface covariance.

## Test Verdict

- covered: row mismatch increases computed positional variance; sufficiently large row mismatch can now push an otherwise similar observation above the stable-surface uncertainty threshold; full default and `vio` package suites remain green.
- missing: midpoint-row symmetry tests; direct off-axis ranking tests; type-level enforcement of `SparseStereoSample` row-consistency invariants.

## Merge Decision

`accept`
