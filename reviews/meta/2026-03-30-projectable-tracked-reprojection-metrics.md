# Meta Review

## Commit Goal

Reduce optimistic reprojection telemetry by adding support-honest tracked-observation reprojection metrics beside the existing accepted-inlier metrics. The change should make the denominator explicit, preserve support-set semantics in types and exported paths, and avoid pretending these same-sample solved-pose residuals are held-out quality.

## Current Repo Starting Point

This tranche starts immediately after `fc8ffeb` on `measurement-system`, where stable-surface retained-observation telemetry was already made type-safe and support-honest. In the tracking path, however, reprojection quality was still only exposed for accepted PnP inliers. That made the primary reprojection stream optimistic by construction and hid the broader tracked-observation error distribution. The first pass at adding tracked residuals still mislabeled the support set because behind-camera tracked observations dropped out silently; that bug has been corrected in the final version of this tranche by explicitly scoping the metrics to `projectable tracked observations`.

## Previous Invariants

- PnP inlier ratio was reported over all tracked observations.
- PnP reprojection RMSE / max / MSE-per-axis were only reported for accepted inliers.
- No explicit metric existed for the number of tracked observations that remained projectable under the solved pose.
- Export paths were support-honest for the recent stable-surface telemetry work, but tracking reprojection telemetry remained optimistic.

## New Invariants Claimed

- Solved-pose reprojection residuals over tracked observations are now explicitly scoped to `projectable tracked observations`.
- The tracker emits both `pnp_tracked_observations` and `pnp_projectable_tracked_observations`, so the residual denominator is explicit.
- Projectable-tracked reprojection RMSE / max / MSE-per-axis use support-typed metric aliases rather than bare numbers.
- Observability export paths preserve the support set explicitly as `pnp_projectable_tracked_observation_*`.
- Inlier-only reprojection metrics remain separate rather than being silently broadened or conflated with the new projectable-tracked metrics.

## Touched Files

- `crates/kiko-slam/src/diagnostics.rs`
- `crates/kiko-slam/src/lib.rs`
- `crates/kiko-slam/src/tracker.rs`
- `crates/kiko-slam/src/observability.rs`

## New Or Changed Metrics

- `FrameDiagnostics::pnp_projectable_tracked_observations`
- `FrameDiagnostics::pnp_projectable_tracked_observation_reprojection_rmse_px`
- `FrameDiagnostics::pnp_projectable_tracked_observation_reprojection_max_px`
- `FrameDiagnostics::pnp_projectable_tracked_observation_reprojection_mse_per_axis_px2`
- `diagnostics/tracking/pnp_projectable_tracked_observations`
- `diagnostics/health/pnp_projectable_tracked_observation_reprojection_rmse_px`
- `diagnostics/health/pnp_projectable_tracked_observation_reprojection_max_px`
- `diagnostics/health/pnp_projectable_tracked_observation_reprojection_mse_per_axis_px2`

## New Or Changed Solver Outputs

- none

## Tests Added

- `diagnostics::tests::projectable_tracked_pnp_pixel_residual_metric_preserves_support`
- `diagnostics::tests::empty_diagnostics_has_all_none` now asserts the projectable-tracked fields are unset by default
- `observability::tests::diagnostics_scalars_include_present_fields` now asserts projectable-tracked count and residual export paths

## Tests Run

- `rustfmt --edition 2024 /home/makerspace/kiko-vio/crates/kiko-slam/src/diagnostics.rs /home/makerspace/kiko-vio/crates/kiko-slam/src/lib.rs /home/makerspace/kiko-vio/crates/kiko-slam/src/tracker.rs /home/makerspace/kiko-vio/crates/kiko-slam/src/observability.rs`
- `cargo test -p kiko-slam --manifest-path /home/makerspace/kiko-vio/Cargo.toml`
- `cargo test -p kiko-slam --features vio --manifest-path /home/makerspace/kiko-vio/Cargo.toml`
- reviewer agent `Pascal`: `accept`
- reviewer agent `Zeno`: `accept-with-follow-up`

## Known Risks Or Deferred Follow-Ups

- These metrics are still same-sample solved-pose diagnostics, not held-out predictive quality.
- `pnp_inlier_ratio` still uses all tracked observations as its denominator while the new residual metrics use the smaller projectable-tracked subset, so the numbers should not be compared as if they shared support.
- On PnP failure, the projectable-tracked residuals remain absent because no solved pose exists; downstream consumers must interpret absence as undefined, not zero.
- The new metrics still live under `diagnostics/health/...`, so top-line dashboards need to keep the `projectable tracked observation` qualifier visible to avoid over-reading them.

## Findings

- `major`: the optimistic “inlier-only” blind spot is materially reduced because the tracker now exports reprojection residuals for the broader projectable tracked set alongside the existing inlier-only metrics.
- `major`: the support-set honesty bug from the first draft is fixed by explicitly renaming the metrics to `projectable tracked observations` and exporting the projectable-tracked count separately.
- `medium`: the observability layer now exposes both denominator counts and the residual metrics, which makes it much easier to spot when a low RMSE is coming from a tiny projectable subset.
- `medium`: the metric aliases in `diagnostics.rs` keep the new telemetry aligned with the repo’s typed support-marker approach instead of introducing another raw float channel.

## Invariant Verdict

- strengthened: tracked reprojection telemetry now distinguishes all tracked count, projectable tracked count, and accepted-inlier residuals; exported metric paths make the support set explicit; same-sample residual support no longer silently drops behind-camera observations without reflecting that in naming.
- weakened or ambiguous: these remain solved-pose diagnostics rather than predictive metrics; `pnp_inlier_ratio` and projectable-tracked residuals still describe different denominators.

## Metric Verdict

- trustworthy: projectable-tracked count; projectable-tracked reprojection RMSE / max / MSE-per-axis; accepted-inlier reprojection metrics remain honest about their narrower support.
- partial or misleading: none of these residuals are held-out predictive quality; dashboards that collapse away the `projectable tracked observation` qualifier could still invite over-interpretation.

## Test Verdict

- covered: support preservation for the new projectable-tracked metric alias; default-empty diagnostics; observability export of projectable-tracked count and residual paths; full package coverage under default and `vio`.
- missing: a tracker-level regression test that constructs a pose with both projectable and behind-camera tracked observations and asserts the count/residual split directly at the diagnostics boundary.

## Merge Decision

`accept with follow-up`
