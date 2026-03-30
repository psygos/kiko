# Meta Review

## Commit Goal

Harden the low-resolution stable surface map so it behaves like a conservative posterior surface belief rather than a lightly renamed point accumulator: one keyframe must not be able to confirm a voxel merely by depositing multiple correlated stereo features into the same voxel, the surface-consistency metric must stop overstating itself as chi-squared, the emitted stable-surface quality scalars must describe the points that actually survive capping and enter the map, rejection-only keyframes must still emit diagnostics, and the metric stream must distinguish raw accepted observations from grouped support views.

## Current Repo Starting Point

This tranche starts from a branch state where measured sparse stereo observations were already separated from interpolated dense visualization, stereo-only SLAM no longer fed interpolated depth into TSDF, and the stable surface path fused `StableSurfacePoint`s into `SurfaceBeliefMap`. However, the map still counted every per-keyframe point landing in a voxel as a separate observation, which let support count act as a false proxy for independent evidence. The map also exposed `chi_squared` naming for a residual-energy score that is not yet backed by a full innovation covariance model, the accepted-point sigma metrics were accumulated before the point cap so they could describe discarded observations, and the telemetry layer still hid all-rejected keyframes while exposing only raw accepted-point counts rather than the grouped support views actually admitted into the map.

## Previous Invariants

- Stable surface observations were measured sparse stereo points, not interpolated dense points.
- TSDF authority remained disabled for stereo-derived interpolated depth.
- Surface fusion used inverse-variance weighting, but correlated same-keyframe points in one voxel could still increase support count as if they were independent evidence.
- The surface map exposed a partially misleading `chi_squared` name for an uncalibrated consistency score.
- Stable-surface point sigma metrics could misreport retained-point quality after the cap.
- Surface telemetry could suppress failure cases entirely when a keyframe produced zero surviving stable points.
- Surface telemetry exposed raw accepted-point count but not the grouped support-view count that the map actually retained.

## New Invariants Claimed

- A single integration batch contributes at most one correlated support view per voxel to the persistent surface belief map.
- Persistent voxel confirmation is based on distinct support views, not raw same-voxel sample count.
- Within-batch grouped evidence carries conservative variance that includes both per-observation variance and within-voxel positional spread.
- The persistent-map consistency gate is named as a consistency score rather than chi-squared.
- Stable-surface sigma metrics are computed from the points that actually survive capping and enter the map.
- Confirmed voxel extraction is deterministic before render capping.
- Stable-surface diagnostics are emitted even when a keyframe contributes zero surviving stable points.
- Surface telemetry names raw observation counts explicitly and exposes grouped support-view count separately.

## Touched Files

- `crates/kiko-slam/src/surface_map.rs`
- `crates/kiko-slam/src/dense_cloud.rs`
- `crates/kiko-slam/src/viz.rs`
- `crates/kiko-slam/src/bin/kiko_slam/slam.rs`

## New Or Changed Metrics

- `diagnostics/surface/accepted_mean_point_sigma_mm`
- `diagnostics/surface/accepted_max_point_sigma_mm`
- `diagnostics/surface/input_raw_observations`
- `diagnostics/surface/accepted_raw_observations`
- `diagnostics/surface/integrated_support_views`
- `diagnostics/surface/mean_confirmed_support_views`
- `diagnostics/surface/mean_confirmed_raw_observations`
- Stable-surface confirmation now uses `max_consistency_score` terminology rather than `max_chi_squared`.

## New Or Changed Solver Outputs

- none

## Tests Added

- `surface_map::tests::single_batch_duplicate_points_do_not_count_as_multiple_support_views`
- `surface_map::tests::three_consistent_support_views_confirmed`
- `surface_map::tests::single_observation_not_confirmed` now also asserts the returned batch integration summary
- `surface_map::tests::single_batch_duplicate_points_do_not_count_as_multiple_support_views` now also asserts grouped support-view count

## Tests Run

- `rustfmt --edition 2024 /home/makerspace/kiko-vio/crates/kiko-slam/src/surface_map.rs /home/makerspace/kiko-vio/crates/kiko-slam/src/dense_cloud.rs /home/makerspace/kiko-vio/crates/kiko-slam/src/viz.rs /home/makerspace/kiko-vio/crates/kiko-slam/src/bin/kiko_slam/slam.rs`
- `cargo test -p kiko-slam --manifest-path /home/makerspace/kiko-vio/Cargo.toml`
- `cargo test -p kiko-slam --features vio --manifest-path /home/makerspace/kiko-vio/Cargo.toml`
- `bash /home/makerspace/kiko-vio/scripts/check_meta_review.sh`

## Known Risks Or Deferred Follow-Ups

- The stereo uncertainty path still uses a conservative fixed disparity-noise prior; calibrated epipolar/rectification-aware uncertainty remains a later M7 task.
- Support views are now conservative per-batch groups, but they are still not a full independence model across nearby keyframes or shared-pose windows.
- The surface consistency score is honest about not being chi-squared, but it is still not a predictive or covariance-normalized residual metric.
- Surface-map confirmation still depends on a thresholded policy layer; held-out predictive surface evaluation remains a later M11 task.
- Dense voxel rendering is now deterministic under the current map, but it remains a visualization of the posterior cache rather than a separately verified predictive evaluation product.

## Findings

- `major`: the previous false-confidence bug is materially reduced because `SurfaceBeliefMap::integrate()` now groups same-batch same-voxel observations before they can affect persistent support counts, so one keyframe cannot confirm a voxel via duplicate local samples alone.
- `major`: surface-map metric honesty improved because the prior `chi_squared` naming has been replaced with an explicit consistency-score concept, avoiding a stronger statistical claim than the code currently supports.
- `major`: surface telemetry now emits even for zero-survivor keyframes, so rejection-heavy failures are no longer silently censored from the metric stream.
- `major`: the telemetry stream now names raw observation counts explicitly and emits grouped support-view count separately, so the dashboard no longer has to infer independent support from raw accepted points.
- `medium`: stable-surface observation-quality scalars are now aligned with the retained observations rather than the pre-cap candidate pool, eliminating a concrete misreport path.
- `medium`: confirmed voxel extraction is now deterministic before capping, which avoids hash-order-dependent visual subsets and makes replay behavior more reproducible.

## Invariant Verdict

- strengthened: support count no longer tracks raw same-voxel same-batch duplicates; persistent confirmation is tied to distinct support views; emitted stable-surface sigma metrics now match retained observations; rejection-only keyframes still surface diagnostics; render output selection is deterministic; raw-vs-support telemetry semantics are explicit.
- weakened or ambiguous: support-view count is still only a conservative grouping heuristic for correlation, not a full independence model across temporally adjacent viewpoints; the disparity-noise prior remains fixed rather than calibrated.

## Metric Verdict

- trustworthy: accepted-point sigma metrics for retained stable-surface observations; raw input/accepted observation counts; integrated support-view count; confirmed voxel count; confirmed ratio; mean confirmed support views; mean confirmed raw observations; the consistency-score name itself is now semantically honest.
- partial or misleading: the surface consistency score remains an unnormalized residual-energy statistic rather than a predictive or covariance-normalized metric; support views still must not be interpreted as true independent evidence count; PnP reprojection metrics outside this tranche remain accepted-inlier metrics rather than held-out predictive metrics.

## Test Verdict

- covered: new support-view invariant against same-batch duplicate confirmation; batch integration summary assertions; continued default and `vio` feature test coverage across the package; retained-point sigma metric path implicitly exercised by package tests; meta-review artifact validation.
- missing: explicit property test for deterministic confirmed-voxel extraction under hash-order variation; synthetic correlation test for nearby-keyframe duplicated evidence; predictive evaluation tests relating surface beliefs to future-view residuals.

## Merge Decision

`accept with follow-up`
