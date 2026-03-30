# Meta Review

## Commit Goal

Prevent stable-surface observations from entering the low-resolution surface map through weak frame poses by gating fusion on typed tracked-pose quality metrics: a minimum projectable tracked observation count and a maximum projectable tracked reprojection RMSE.

## Current Repo Starting Point

This tranche starts immediately after `4c72b6d` on `measurement-system`. At the start of this change, stable-surface observability had already become support-honest: repeated same-view evidence no longer counted as new support, and raw/support/redundant grouped-view accounting was exposed in telemetry. But surface fusion still happened unconditionally once stable stereo observations existed and a pose was present, even if the current tracked pose was weak or only supported by a tiny projectable set.

## Previous Invariants

- Stable-surface fusion used stereo observation uncertainty and support-view diversity, but not frame pose quality.
- Any frame with `output.pose` and generated stable surface points could integrate into the surface map.
- The surface pipeline exposed integration counters honestly, but it could still accumulate geometry from weakly supported tracked poses.
- `pnp_projectable_tracked_observations` and `pnp_projectable_tracked_observation_reprojection_rmse_px` were available diagnostics, but they were not yet used to gate surface fusion.

## New Invariants Claimed

- Stable-surface fusion now requires a typed minimum projectable tracked observation count and a typed maximum tracked reprojection RMSE.
- Missing tracked-count or tracked-RMSE diagnostics reject surface fusion conservatively.
- Rejected frames still emit explicit pose-gate observability metrics, zero integration counters, and the unchanged surface-map state.
- The pose-quality gate lives in the policy layer (`viz.rs`), not inside the surface estimator or tracker core.

## Touched Files

- `crates/kiko-slam/src/viz.rs`
- `crates/kiko-slam/src/bin/kiko_slam/slam.rs`

## New Or Changed Metrics

- `diagnostics/surface/pose_gate/accepted`
- `diagnostics/surface/pose_gate/rejected_low_projectable_tracked_observations`
- `diagnostics/surface/pose_gate/rejected_missing_projectable_tracked_observations`
- `diagnostics/surface/pose_gate/rejected_missing_projectable_tracked_reprojection_rmse`
- `diagnostics/surface/pose_gate/rejected_high_projectable_tracked_reprojection_rmse`
- `diagnostics/surface/pose_gate/projectable_tracked_observations`
- `diagnostics/surface/pose_gate/min_required_projectable_tracked_observations`
- `diagnostics/surface/pose_gate/projectable_tracked_reprojection_rmse_px`
- `diagnostics/surface/pose_gate/max_allowed_projectable_tracked_reprojection_rmse_px`

## New Or Changed Solver Outputs

- none

## Tests Added

- `viz::tests::surface_pose_quality_gate_accepts_low_tracked_reprojection_rmse`
- `viz::tests::surface_pose_quality_gate_rejects_missing_tracked_observation_count`
- `viz::tests::surface_pose_quality_gate_rejects_low_tracked_observation_count`
- `viz::tests::surface_pose_quality_gate_rejects_missing_tracked_reprojection_rmse`
- `viz::tests::surface_pose_quality_gate_rejects_high_tracked_reprojection_rmse`
- `viz::tests::surface_pose_quality_scalars_export_decision_and_threshold`
- `slam::tests::slam_env_help_mentions_surface_pose_quality_gate_env`
- `slam::tests::slam_env_help_mentions_surface_pose_quality_support_env`

## Tests Run

- `rustfmt --edition 2024 /home/makerspace/kiko-vio/crates/kiko-slam/src/viz.rs /home/makerspace/kiko-vio/crates/kiko-slam/src/bin/kiko_slam/slam.rs`
- `cargo test -p kiko-slam --manifest-path /home/makerspace/kiko-vio/Cargo.toml`
- `cargo test -p kiko-slam --features vio --manifest-path /home/makerspace/kiko-vio/Cargo.toml`
- reviewer agent `Euclid`: `accept`

## Known Risks Or Deferred Follow-Ups

- The defaults `KIKO_SURFACE_MIN_PROJECTABLE_TRACKED_OBSERVATIONS=8` and `KIKO_SURFACE_MAX_TRACKED_REPROJECTION_RMSE_PX=1.5` are still policy thresholds, even though they are typed, explicit, and observable.
- The current tests validate the gate helpers and exported scalar paths, but not a full end-to-end `log_surface_observations` accept/reject path against `surface_map` mutation.
- The gate currently reasons over projectable tracked observations only; it does not use future predictive metrics or full pose covariance yet.

## Findings

- `major`: the surface map is materially harder to pollute now because stable stereo observations no longer bypass tracked-pose quality checks.
- `major`: the gating signal is semantically honest because both support size and RMSE are explicitly scoped to projectable tracked observations and represented by typed metrics.
- `medium`: the pose-quality policy is visible rather than hidden; every reject mode has an explicit scalar path and both thresholds are discoverable in CLI help.

## Invariant Verdict

- strengthened: stable-surface fusion now depends on both stereo observation quality and pose support quality; missing diagnostics fail closed; pose-quality policy remains outside estimator kernels.
- weakened or ambiguous: no new invariant weakening beyond the fact that policy thresholds still require tuning by dataset and deployment.

## Metric Verdict

- trustworthy: the new pose-gate metrics are support-honest, typed, and branch-explicit; they distinguish missing count, low count, missing RMSE, and high RMSE rather than collapsing them into one opaque skip.
- partial or misleading: none new, provided consumers remember that these are policy-gate metrics rather than posterior confidence estimates.

## Test Verdict

- covered: gate accept/reject decision logic; exported scalar paths and values; env help discoverability; full default and `vio` test suites.
- missing: one integration-style test proving the full `log_surface_observations` reject path leaves `surface_map` unchanged while still exporting zero integration counters.

## Merge Decision

`accept`
