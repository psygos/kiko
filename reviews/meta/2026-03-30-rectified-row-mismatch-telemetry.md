# Meta Review

## Commit Goal

Add truthful retained-observation stereo row-mismatch telemetry to the stable surface path without weakening the existing “measured sparse stereo only” invariant. The tranche should expose what the rectified stereo pairs actually disagreed on vertically, keep missing support as missing rather than reporting fake zero quality, make the quantity itself lawful at the type level, and preserve support-set semantics all the way to the exported metric path.

## Current Repo Starting Point

This tranche starts from the `da3b180` stable-surface hardening state: same-batch same-voxel observations are already grouped into one correlated support view before persistent fusion, confirmed voxels depend on support views rather than raw sample count, interpolated stereo depth remains banned from the stereo-only TSDF path, and stable-surface telemetry already distinguishes raw observations from grouped support views. However, the stable surface path still lacked any explicit rectified stereo mismatch telemetry, and the first attempt at adding it misreported empty support as `0 px`, used a broader `epipolar_error` name than the underlying quantity justified, and did not preserve support-set semantics at the export path.

## Previous Invariants

- Stable surface fusion consumed measured sparse stereo observations, not interpolated dense depth.
- Stable surface confirmation depended on grouped support views rather than raw same-batch duplicates.
- Stable-surface sigma metrics described the retained observations that actually entered the map, but empty retained support still relied on omission by convention rather than explicit type-level absence.
- Stereo row mismatch was not tracked as a first-class diagnostic quantity.

## New Invariants Claimed

- Retained-observation rectified row mismatch is represented by a lawful `RectifiedRowMismatchPx` newtype rather than a raw `f32`.
- Empty retained support cannot be exported as `0 px` row mismatch or `0 mm` accepted sigma; these values remain absent in `StableSurfaceStats`.
- Retained stable-surface row-mismatch telemetry carries an explicit support marker via `StableSurfaceRetainedRawPixelResidualMetric`.
- The exported Rerun metric path preserves the support set explicitly as `diagnostics/surface/retained_raw_observations/...`.
- The stable surface path continues to treat row mismatch as descriptive retained-observation telemetry, not as posterior quality or calibrated uncertainty.

## Touched Files

- `crates/kiko-slam/src/triangulation.rs`
- `crates/kiko-slam/src/dense_cloud.rs`
- `crates/kiko-slam/src/surface_map.rs`
- `crates/kiko-slam/src/viz.rs`
- `crates/kiko-slam/src/diagnostics.rs`
- `crates/kiko-slam/src/lib.rs`

## New Or Changed Metrics

- `diagnostics/surface/retained_raw_observations/mean_rectified_row_mismatch_px`
- `diagnostics/surface/retained_raw_observations/max_rectified_row_mismatch_px`
- `StableSurfaceStats::mean_accepted_position_sigma_m` now uses explicit missing-value semantics
- `StableSurfaceStats::max_accepted_position_sigma_m` now uses explicit missing-value semantics

## New Or Changed Solver Outputs

- none

## Tests Added

- `triangulation::tests::rectified_row_mismatch_rejects_invalid_values`
- `diagnostics::tests::stable_surface_pixel_residual_metric_preserves_support`
- `dense_cloud::tests::stable_surface_rejects_high_uncertainty_observations` now also asserts missing retained-observation metrics
- `dense_cloud::tests::stable_surface_keeps_most_stable_points_when_capped` now asserts retained-observation metric presence
- `dense_cloud::tests::stable_surface_reports_retained_rectified_row_mismatch_metrics`

## Tests Run

- `rustfmt --edition 2024 /home/makerspace/kiko-vio/crates/kiko-slam/src/triangulation.rs /home/makerspace/kiko-vio/crates/kiko-slam/src/dense_cloud.rs /home/makerspace/kiko-vio/crates/kiko-slam/src/surface_map.rs /home/makerspace/kiko-vio/crates/kiko-slam/src/viz.rs /home/makerspace/kiko-vio/crates/kiko-slam/src/diagnostics.rs /home/makerspace/kiko-vio/crates/kiko-slam/src/lib.rs`
- `cargo test -p kiko-slam --manifest-path /home/makerspace/kiko-vio/Cargo.toml`
- `cargo test -p kiko-slam --features vio --manifest-path /home/makerspace/kiko-vio/Cargo.toml`
- reviewer agent `Pascal`: `accept-with-follow-up`
- reviewer agent `Zeno`: `accept-with-follow-up`

## Known Risks Or Deferred Follow-Ups

- `SparseStereoSample` is still publicly constructible field-by-field, so `right_v` and `rectified_row_mismatch_px` can still be made semantically inconsistent by external callers even though each field is individually lawful.
- The sigma telemetry paths still use the older `diagnostics/surface/accepted_*_point_sigma_mm` naming rather than the more explicit retained-support namespace used for row mismatch.
- Missing-value semantics are now truthful in types and logs, but Rerun still represents “undefined this frame” by omission rather than an explicit undefined marker.
- Rectified row mismatch remains descriptive retained-observation telemetry only; it is not yet propagated into a calibrated stereo uncertainty model.

## Findings

- `major`: the previous false-zero telemetry bug is fixed because empty retained support now yields `None` in `StableSurfaceStats` and the logger emits no row-mismatch or sigma value for that frame.
- `major`: metric honesty improved because the quantity is now named `rectified_row_mismatch` rather than the broader `epipolar_error`, matching the actual rectified-row construction in `Triangulator::extract_stereo_samples`.
- `major`: the metric path now preserves support semantics explicitly at export time, so the telemetry can no longer be mistaken for fused-surface or posterior quality.
- `medium`: row mismatch is now lawful by construction as a non-negative finite newtype, preventing malformed retained-observation telemetry from entering the stable-surface path through a raw float.
- `medium`: the dedicated support-specific pixel-residual metric alias aligns the telemetry with the repo’s typed diagnostic-metric philosophy instead of pushing bare `f32` values through the stats path.

## Invariant Verdict

- strengthened: retained-observation row mismatch is lawful by construction; missing retained support no longer misreports zero quality; support-set semantics now survive both the type layer and the export path; the metric name now matches the underlying geometric quantity.
- weakened or ambiguous: `SparseStereoSample` still allows semantically inconsistent field combinations when built manually; retained sigma path naming remains less explicit than the new row-mismatch path.

## Metric Verdict

- trustworthy: retained raw-observation rectified row-mismatch mean/max; support-specific pixel residual typing for retained stable-surface observations; absence of retained-observation metrics when no retained support exists.
- partial or misleading: accepted sigma path names remain support-implicit; row mismatch is still descriptive raw-observation telemetry rather than calibrated uncertainty or posterior surface quality.

## Test Verdict

- covered: invalid row-mismatch rejection; support-set preservation for the new typed metric; empty-retained-support omission semantics; retained-metric aggregation over two known row-mismatch samples; continued default and `vio` package coverage.
- missing: an API-level constructor or parser test that makes `SparseStereoSample` semantically consistent as a whole rather than only field-wise lawful; an explicit viewer/consumer contract test for omitted-frame semantics in exported telemetry.

## Merge Decision

`accept with follow-up`
