# Meta Review

## Commit Goal

Establish the first runtime measurement-system invariants in code, not prose: separate measured depth from interpolated stereo depth at the type boundary, prevent TSDF from consuming interpolated pseudo-depth as authoritative input, make PnP diagnostic support sets compile-time specific rather than runtime tags, and add continuous `vio` test coverage in CI.

## Current Repo Starting Point

This tranche starts from a branch that already had the roadmap/meta-review documents plus initial geometry and diagnostics scaffolding, but the runtime still fed `generate_dense_depth_image()` directly into `TsdfIntegrateMsg`, and the PnP diagnostics still stored generic metric wrappers with support tags that could be mismatched in release builds.

## Previous Invariants

- Dense interpolation was documented as visualization-only, but the type system still allowed it to flow into TSDF as if it were measured depth.
- PnP support semantics existed only as runtime labels on generic metric wrappers.
- CI validated `cargo test -p kiko-slam`, but not the `vio` feature path.

## New Invariants Claimed

- TSDF integration now requires `DepthImage<MeasuredDepth>` through `TsdfIntegrateMsg::try_new`.
- Dense stereo interpolation now yields `DepthImage<InterpolatedDepth>` and therefore cannot be passed to TSDF by accident.
- Dataset SLAM no longer integrates interpolated stereo depth into TSDF.
- `FrameDiagnostics` PnP metric fields are support-specific types, so support mismatches are not representable at those field boundaries.
- CI now runs `cargo test -p kiko-slam --features vio` in addition to the default host test path.

## Touched Files

- `.github/workflows/ci.yml`
- `crates/kiko-slam/src/bin/kiko_slam/slam.rs`
- `crates/kiko-slam/src/dense_cloud.rs`
- `crates/kiko-slam/src/depth.rs`
- `crates/kiko-slam/src/diagnostics.rs`
- `crates/kiko-slam/src/lib.rs`
- `crates/kiko-slam/src/observability.rs`
- `crates/kiko-slam/src/tracker.rs`
- `crates/kiko-slam/src/tsdf.rs`

## New Or Changed Metrics

- `pnp_inlier_ratio` is now `PnpInlierRatioMetric`.
- `pnp_tracked_observations` is now `PnpTrackedObservationCountMetric`.
- `pnp_inlier_reprojection_{rmse,max}_px` are now `PnpAcceptedInlierPixelResidualMetric`.

## New Or Changed Solver Outputs

- none

## Tests Added

- `depth::tests::interpolated_depth_image_preserves_provenance`
- `tsdf::tests::tsdf_intrinsics_reject_non_positive_focal_length`
- `tsdf::tests::tsdf_msg_rejects_grayscale_dimension_mismatch`
- `tsdf::tests::tsdf_msg_accepts_measured_depth_image`
- `tsdf.rs` compile-fail doctest proving `DepthImage<InterpolatedDepth>` cannot be passed to `TsdfIntegrateMsg::try_new`
- diagnostics tests updated to exercise support-specific metric aliases

## Tests Run

- `cargo fmt --package kiko-slam --manifest-path /home/makerspace/kiko-vio/Cargo.toml`
- `cargo test -p kiko-slam --manifest-path /home/makerspace/kiko-vio/Cargo.toml`
- `cargo test -p kiko-slam --features vio --manifest-path /home/makerspace/kiko-vio/Cargo.toml`

## Known Risks Or Deferred Follow-Ups

- `optimize_vio()` and tracker output semantics still have unresolved `map_from_odom` identity assumptions.
- PnP reprojection metrics are still computed on accepted tracking observations under the refined pose, not a fully support-normalized predictive metric.
- Dense point cloud generation is still a visualization-oriented interpolant; only the invalid TSDF promotion path was removed in this tranche.
- BA/VIO `final_cost` remains a solver objective and is still exported as a scalar without a more explicit semantic wrapper.

## Findings

- `major`: the typed depth boundary is materially stronger; interpolated depth can no longer enter TSDF by construction, and the binary path that previously violated the measurement/model distinction is removed.
- `major`: PnP diagnostics are less brittle because support-specific metric aliases eliminate the previous release-build loophole where the wrong support tag could be logged under the wrong path.
- `major`: the `map_from_odom` seam remains the largest unresolved architectural gap for the measurement roadmap.
- `medium`: CI verifiability is improved because the `vio` feature path is now part of the host test job.
- `medium`: dense reconstruction quality is still limited by the interpolation model itself; this tranche only prevents false authority, it does not yet introduce uncertainty-aware fusion or predictive evaluation.

## Invariant Verdict

- strengthened: depth provenance at the TSDF boundary; compile-time support typing for emitted PnP diagnostics; continuous verification of the `vio` feature path.
- weakened or ambiguous: none in the touched paths, but map/odom authority and solver-objective semantics remain unresolved elsewhere.

## Metric Verdict

- trustworthy: support set for the touched PnP diagnostics is now encoded in the field type; TSDF no longer pretends interpolated stereo depth is authoritative.
- partial or misleading: `pnp_inlier_reprojection_*` still describe accepted tracking observations under the refined pose, and `ba/final_cost` is still not a physically normalized quality metric.

## Test Verdict

- covered: default host build, `vio` feature build, new depth/TSDF unit tests, compile-fail provenance guard.
- missing: dataset-level predictive evaluation, uncertainty calibration tests, loop-closure/map-bridge reintegration tests.

## Merge Decision

`accept with follow-up`
